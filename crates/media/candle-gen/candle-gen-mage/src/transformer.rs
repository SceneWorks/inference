//! Mage dual-stream NR-MMDiT.

use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_gen_boogu::loader::{linear, Weights};
use candle_nn::Linear;

use crate::config::{HEAD_DIM, NORM_EPS};
use crate::rope::{self, PackLayout, RopeTable};

fn rms(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let norm = x
        .to_dtype(DType::F32)?
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1., NORM_EPS)?
        .sqrt()?;
    x.to_dtype(DType::F32)?
        .broadcast_div(&norm)?
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(x.dtype())
}

fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let centered = xf.broadcast_sub(&mean)?;
    let denom = centered
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1., NORM_EPS)?
        .sqrt()?;
    centered.broadcast_div(&denom)?.to_dtype(x.dtype())
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.)?)?.broadcast_add(shift)
}

struct TimestepEmbedder {
    l1: Linear,
    l2: Linear,
}

impl TimestepEmbedder {
    fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            l1: linear(w, "time_text_embed.timestep_embedder.linear_1", true)?,
            l2: linear(w, "time_text_embed.timestep_embedder.linear_2", true)?,
        })
    }

    fn forward(&self, sigma: &Tensor, dtype: DType) -> Result<Tensor> {
        let half = 128usize;
        let mut freqs: Vec<f32> = (0..half)
            .map(|i| (-10_000f32.ln() * i as f32 / half as f32).exp())
            .collect();
        if dtype == DType::BF16 {
            freqs = Tensor::from_vec(freqs, half, sigma.device())?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .to_vec1()?;
        }
        let f = Tensor::from_vec(freqs, (1, half), sigma.device())?;
        let t = sigma.to_dtype(DType::F32)?.unsqueeze(1)?;
        let a = t.broadcast_mul(&f)?.affine(1000., 0.)?;
        let emb = Tensor::cat(&[a.cos()?, a.sin()?], 1)?.to_dtype(dtype)?;
        self.l2.forward(&self.l1.forward(&emb)?.silu()?)
    }
}

struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj: linear(w, &format!("{prefix}.net.0.proj"), true)?,
            out: linear(w, &format!("{prefix}.net.2"), true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle's `gelu` is the tanh approximation used by diffusers' gelu-approximate.
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }
}

struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    add_out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    norm_add_q: Tensor,
    norm_add_k: Tensor,
    heads: usize,
}

impl JointAttention {
    fn load(w: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        let l = |name: &str| linear(w, &format!("{prefix}.{name}"), true);
        Ok(Self {
            to_q: l("to_q")?,
            to_k: l("to_k")?,
            to_v: l("to_v")?,
            to_out: l("to_out.0")?,
            add_q: l("add_q_proj")?,
            add_k: l("add_k_proj")?,
            add_v: l("add_v_proj")?,
            add_out: l("to_add_out")?,
            norm_q: w.get(&format!("{prefix}.norm_q.weight"))?,
            norm_k: w.get(&format!("{prefix}.norm_k.weight"))?,
            norm_add_q: w.get(&format!("{prefix}.norm_added_q.weight"))?,
            norm_add_k: w.get(&format!("{prefix}.norm_added_k.weight"))?,
            heads,
        })
    }

    fn forward(
        &self,
        image: &Tensor,
        text: &Tensor,
        table: &RopeTable,
        layout: &PackLayout,
    ) -> Result<(Tensor, Tensor)> {
        let (_, ni, _) = image.dims3()?;
        let (_, nt, _) = text.dims3()?;
        let shape_i = (ni, self.heads, HEAD_DIM);
        let shape_t = (nt, self.heads, HEAD_DIM);
        let iq = rope::apply(
            &rms(&self.to_q.forward(image)?.reshape(shape_i)?, &self.norm_q)?,
            table,
        )?;
        let ik = rope::apply(
            &rms(&self.to_k.forward(image)?.reshape(shape_i)?, &self.norm_k)?,
            table,
        )?;
        let iv = self.to_v.forward(image)?.reshape(shape_i)?;
        let tq = rms(
            &self.add_q.forward(text)?.reshape(shape_t)?,
            &self.norm_add_q,
        )?;
        let tk = rms(
            &self.add_k.forward(text)?.reshape(shape_t)?,
            &self.norm_add_k,
        )?;
        let tv = self.add_v.forward(text)?.reshape(shape_t)?;

        let image_cu = layout.image_cu();
        let text_cu = layout.text_cu();
        let mut image_parts = Vec::with_capacity(layout.segments());
        let mut text_parts = Vec::with_capacity(layout.segments());
        for s in 0..layout.segments() {
            let il = image_cu[s + 1] - image_cu[s];
            let tl = text_cu[s + 1] - text_cu[s];
            let joint = |t: &Tensor, i: &Tensor| -> Result<Tensor> {
                Tensor::cat(
                    &[t.narrow(0, text_cu[s], tl)?, i.narrow(0, image_cu[s], il)?],
                    0,
                )?
                .transpose(0, 1)?
                .unsqueeze(0)
            };
            let q = joint(&tq, &iq)?;
            let k = joint(&tk, &ik)?;
            let v = joint(&tv, &iv)?;
            let o = candle_gen::sdpa_budgeted_bhsd(
                &q,
                &k,
                &v,
                (HEAD_DIM as f64).powf(-0.5),
                None,
                candle_nn::ops::softmax_last_dim,
                candle_gen::ATTN_SCORES_BUDGET,
            )?
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((tl + il, self.heads * HEAD_DIM))?;
            text_parts.push(o.narrow(0, 0, tl)?);
            image_parts.push(o.narrow(0, tl, il)?);
        }
        let image = Tensor::cat(&image_parts.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        let text = Tensor::cat(&text_parts.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        Ok((self.to_out.forward(&image)?, self.add_out.forward(&text)?))
    }
}

struct Block {
    image_mod: Linear,
    text_mod: Linear,
    attention: JointAttention,
    image_ff: FeedForward,
    text_ff: FeedForward,
}

struct Mods {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl Block {
    fn load(w: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            image_mod: linear(w, &format!("{prefix}.img_mod.1"), true)?,
            text_mod: linear(w, &format!("{prefix}.txt_mod.1"), true)?,
            attention: JointAttention::load(w, &format!("{prefix}.attn"), heads)?,
            image_ff: FeedForward::load(w, &format!("{prefix}.img_mlp"))?,
            text_ff: FeedForward::load(w, &format!("{prefix}.txt_mlp"))?,
        })
    }

    fn mods(linear: &Linear, temb: &Tensor, ids: &Tensor, tokens: usize) -> Result<(Mods, Mods)> {
        let p = linear.forward(&temb.silu()?)?;
        let dim = p.dim(1)? / 6;
        let one = |offset: usize| -> Result<Mods> {
            let expand = |part: usize| -> Result<Tensor> {
                p.narrow(1, offset + part * dim, dim)?
                    .contiguous()?
                    .index_select(ids, 0)?
                    .reshape((1, tokens, dim))
            };
            Ok(Mods {
                shift: expand(0)?,
                scale: expand(1)?,
                gate: expand(2)?,
            })
        };
        Ok((one(0)?, one(3 * dim)?))
    }

    fn forward(
        &self,
        image: Tensor,
        text: Tensor,
        temb: &Tensor,
        table: &RopeTable,
        layout: &PackLayout,
    ) -> Result<(Tensor, Tensor)> {
        let image_ids = layout.image_segment_ids(image.device())?;
        let text_ids: Vec<u32> = layout
            .text_lens()
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat_n(i as u32, *n))
            .collect();
        let text_ids = Tensor::from_vec(text_ids, layout.text_tokens(), text.device())?;
        let (im1, im2) = Self::mods(&self.image_mod, temb, &image_ids, layout.image_tokens())?;
        let (tx1, tx2) = Self::mods(&self.text_mod, temb, &text_ids, layout.text_tokens())?;
        let imn = modulate(&layer_norm(&image)?, &im1.shift, &im1.scale)?;
        let txn = modulate(&layer_norm(&text)?, &tx1.shift, &tx1.scale)?;
        let (ia, ta) = self.attention.forward(&imn, &txn, table, layout)?;
        let image = (&image + ia.broadcast_mul(&im1.gate)?)?;
        let text = (&text + ta.broadcast_mul(&tx1.gate)?)?;
        let iff =
            self.image_ff
                .forward(&modulate(&layer_norm(&image)?, &im2.shift, &im2.scale)?)?;
        let tff = self
            .text_ff
            .forward(&modulate(&layer_norm(&text)?, &tx2.shift, &tx2.scale)?)?;
        Ok((
            (&image + iff.broadcast_mul(&im2.gate)?)?,
            (&text + tff.broadcast_mul(&tx2.gate)?)?,
        ))
    }
}

pub struct MageTransformer {
    image_in: Linear,
    text_norm: Tensor,
    text_in: Linear,
    timestep: TimestepEmbedder,
    blocks: Vec<Block>,
    final_mod: Linear,
    output: Linear,
    dtype: DType,
}

impl MageTransformer {
    pub fn load(dir: &Path, cfg: &crate::config::MageConfig, device: &Device) -> Result<Self> {
        let weights = Weights::from_dir(dir, device, DType::BF16)?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::load(
                &weights,
                &format!("transformer_blocks.{i}"),
                cfg.num_heads,
            )?);
        }
        Ok(Self {
            image_in: linear(&weights, "img_in", true)?,
            text_norm: weights.get("txt_norm.weight")?,
            text_in: linear(&weights, "txt_in", true)?,
            timestep: TimestepEmbedder::load(&weights)?,
            blocks,
            final_mod: linear(&weights, "norm_out.linear", true)?,
            output: linear(&weights, "proj_out", true)?,
            dtype: DType::BF16,
        })
    }

    /// Inputs are packed `[1, image_tokens, 128]`, `[1, text_tokens, 2560]`, and one sigma per
    /// attention segment. Output is packed flow velocity `[1, image_tokens, 128]`.
    pub fn forward(
        &self,
        image: &Tensor,
        text: &Tensor,
        sigma: &Tensor,
        layout: &PackLayout,
    ) -> Result<Tensor> {
        let table = RopeTable::build(layout, self.dtype, image.device())?;
        let mut image = self.image_in.forward(&image.to_dtype(self.dtype)?)?;
        let mut text = self
            .text_in
            .forward(&rms(&text.to_dtype(self.dtype)?, &self.text_norm)?)?;
        let temb = self
            .timestep
            .forward(&sigma.to_dtype(self.dtype)?, self.dtype)?;
        for block in &self.blocks {
            (image, text) = block.forward(image, text, &temb, &table, layout)?;
        }
        let params = self.final_mod.forward(&temb.silu()?)?;
        let dim = params.dim(1)? / 2;
        let ids = layout.image_segment_ids(image.device())?;
        // Output head is scale,shift—the opposite of block shift,scale,gate.
        let scale = params
            .narrow(1, 0, dim)?
            .contiguous()?
            .index_select(&ids, 0)?
            .reshape((1, layout.image_tokens(), dim))?;
        let shift = params
            .narrow(1, dim, dim)?
            .contiguous()?
            .index_select(&ids, 0)?
            .reshape((1, layout.image_tokens(), dim))?;
        self.output
            .forward(&modulate(&layer_norm(&image)?, &shift, &scale)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_head_is_scale_then_shift() {
        let d = Device::Cpu;
        let x = Tensor::new(&[[[0f32, 4.]]], &d).unwrap();
        let scale = Tensor::ones((1, 1, 2), DType::F32, &d).unwrap();
        let shift = Tensor::new(&[[[10f32, 20.]]], &d).unwrap();
        let got = modulate(&layer_norm(&x).unwrap(), &shift, &scale)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((got[0][0][0] - 8.).abs() < 1e-4);
        assert!((got[0][0][1] - 22.).abs() < 1e-4);
    }
}
