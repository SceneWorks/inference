//! The frozen MOSS graph. The small geometry constructor is private to numerical tests.
use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{self, Module};
use candle_gen::gen_core::{moss_image::MossImageConfig, CancelFlag};
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub dim: usize,
    pub ffn: usize,
    pub heads: usize,
    pub layers: usize,
    pub text: usize,
    pub freq: usize,
    pub channels: usize,
}
impl From<&MossImageConfig> for Geometry {
    fn from(c: &MossImageConfig) -> Self {
        Self {
            dim: c.dim,
            ffn: c.ffn_dim,
            heads: c.num_heads,
            layers: c.num_layers,
            text: c.text_dim,
            freq: c.freq_dim,
            channels: c.latent_channels,
        }
    }
}

pub(crate) fn check_cancel(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(CandleError::Canceled)
    } else {
        Ok(())
    }
}

struct LowRank {
    down: Tensor,
    up: Tensor,
    scale: f64,
}
struct Linear {
    base: candle_nn::Linear,
    low_rank: Option<LowRank>,
}
impl Linear {
    fn load(
        w: &Weights,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        adapter: Option<&Weights>,
        consumed: &mut BTreeSet<String>,
    ) -> Result<Self> {
        let weight_name = format!("{name}.weight");
        let weight = w.require(&weight_name)?;
        if weight.dims() != [output, input] {
            return Err(CandleError::Msg(format!("MOSS shape {weight_name}")));
        }
        consumed.insert(weight_name);
        let b = if bias {
            let key = format!("{name}.bias");
            let value = w.require(&key)?;
            if value.dims() != [output] {
                return Err(CandleError::Msg(format!("MOSS shape {key}")));
            }
            consumed.insert(key);
            Some(value)
        } else {
            None
        };
        let low_rank = if let Some(a) = adapter {
            let prefix = format!("diffusion_model.{name}");
            let down = a.require(&format!("{prefix}.lora_down.weight"))?;
            let up = a.require(&format!("{prefix}.lora_up.weight"))?;
            let alpha = a
                .require(&format!("{prefix}.alpha"))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let rank = down.dim(0)?;
            if down.dims() != [rank, input]
                || up.dims() != [output, rank]
                || alpha.len() != 1
                || !alpha[0].is_finite()
                || rank == 0
                || down.dtype() != DType::F32
                || up.dtype() != DType::F32
            {
                return Err(CandleError::Msg(format!(
                    "MOSS invalid low rank projection {prefix}"
                )));
            }
            Some(LowRank {
                down,
                up,
                scale: f64::from(alpha[0]) / rank as f64,
            })
        } else {
            None
        };
        Ok(Self {
            base: candle_nn::Linear::new(weight, b),
            low_rank,
        })
    }
    fn dtype(&self) -> DType {
        self.base.weight().dtype()
    }
    fn base_forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle's CPU matmul has no BF16 kernel. This explicit FP32 accumulation over the
        // already-rounded BF16 operands, followed by BF16 rounding, is the CPU numerical oracle.
        // CUDA/Metal use their native BF16 GEMM; no production GPU operand is widened here.
        if x.device().is_cpu() && self.dtype() == DType::BF16 {
            let linear = candle_nn::Linear::new(
                self.base.weight().to_dtype(DType::F32)?,
                self.base
                    .bias()
                    .map(|b| b.to_dtype(DType::F32))
                    .transpose()?,
            );
            Ok(linear
                .forward(&x.to_dtype(DType::F32)?)?
                .to_dtype(DType::BF16)?)
        } else {
            Ok(self.base.forward(x)?)
        }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(a) = &self.low_rank {
            let base = self
                .base_forward(&x.to_dtype(self.dtype())?)?
                .to_dtype(DType::F32)?;
            let low =
                candle_nn::Linear::new(a.down.clone(), None).forward(&x.to_dtype(DType::F32)?)?;
            let low = candle_nn::Linear::new(a.up.clone(), None).forward(&low)?;
            Ok(base.add(&(low * a.scale)?)?)
        } else {
            self.base_forward(x)
        }
    }
}

struct Norm(Tensor);
impl Norm {
    fn load(w: &Weights, name: &str, dim: usize, keys: &mut BTreeSet<String>) -> Result<Self> {
        let key = format!("{name}.weight");
        let t = w.require(&key)?;
        if t.dims() != [dim] {
            return Err(CandleError::Msg(format!("MOSS norm shape {key}")));
        }
        keys.insert(key);
        Ok(Self(t))
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let f = x.to_dtype(DType::F32)?;
        let normalized = f.broadcast_mul(
            &((f.sqr()?.mean_keepdim(candle_gen::candle_core::D::Minus1)? + 1e-6)?
                .sqrt()?
                .recip()?),
        )?;
        // PyTorch promotes the normalized activation when the norm weight is an FP32 island.
        let dtype = if x.dtype() == DType::F32 || self.0.dtype() == DType::F32 {
            DType::F32
        } else {
            x.dtype()
        };
        Ok(normalized
            .to_dtype(x.dtype())?
            .to_dtype(dtype)?
            .broadcast_mul(&self.0.to_dtype(dtype)?)?)
    }
}
pub(crate) fn situ(hidden: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let a = ((gate / 4.0)?.tanh()? * 4.0)?;
    let a = a.mul(&candle_nn::ops::sigmoid(gate)?)?;
    Ok(a.mul(&((hidden / 25.0)?.tanh()? * 25.0)?)?)
}

/// Source FP32 frequency construction, coordinates and sin/cos, then activation cast.
pub(crate) fn rope(
    height: usize,
    width: usize,
    text: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let bands = head_dim / 8;
    let frequencies: Vec<f32> = (0..bands)
        .map(|i| 1.0 / 10000_f32.powf(i as f32 / bands as f32))
        .collect();
    let mut values = Vec::with_capacity((height * width + text) * head_dim);
    for n in 0..height * width + text {
        let coordinate = if n < height * width {
            [
                0.0,
                text as f32,
                n.div_euclid(width) as f32 - (height - 1) as f32 / 2.0,
                (n % width) as f32 - (width - 1) as f32 / 2.0,
            ]
        } else {
            [0.0, (n - height * width) as f32, 0.0, 0.0]
        };
        for trig in 0..2 {
            for c in coordinate {
                for f in &frequencies {
                    let p = c * f;
                    values.push(if trig == 0 { p.sin() } else { p.cos() });
                }
            }
        }
    }
    Ok(Tensor::from_vec(values, (1, height * width + text, head_dim), device)?.to_dtype(dtype)?)
}
fn rotate(x: &Tensor, rope: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let sin = rope.narrow(2, 0, half)?.unsqueeze(2)?;
    let cos = rope.narrow(2, half, half)?.unsqueeze(2)?;
    let sin = Tensor::cat(&[&sin, &sin], 3)?;
    let cos = Tensor::cat(&[&cos, &cos], 3)?;
    let paired = Tensor::cat(
        &[&x.narrow(3, half, half)?.neg()?, &x.narrow(3, 0, half)?],
        3,
    )?;
    // Source multiplication promotes BF16 rope when a widened DMD Q/K enters.
    let dtype = x.dtype();
    Ok(x.broadcast_mul(&cos.to_dtype(dtype)?)?
        .add(&paired.broadcast_mul(&sin.to_dtype(dtype)?)?)?)
}
fn attention_qk_operands(
    q: &Tensor,
    k: &Tensor,
    value_dtype: DType,
    r: &Tensor,
    heads: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, s, d) = q.dims3()?;
    let width = d / heads;
    let q = rotate(&q.reshape((b, s, heads, width))?, r)?
        .transpose(1, 2)?
        .to_dtype(value_dtype)?;
    let k = rotate(&k.reshape((b, s, heads, width))?, r)?
        .transpose(1, 2)?
        .to_dtype(value_dtype)?;
    // CUDA batched GEMM needs packed heads. BF16 -> FP32 already packs these views,
    // but DMD's FP32 -> FP32 is a no-op and retains the interleaved head strides.
    // Pack K before its final matrix transpose, which cuBLAS supports directly.
    Ok((
        q.to_dtype(DType::F32)?.contiguous()?,
        k.to_dtype(DType::F32)?.contiguous()?.transpose(2, 3)?,
    ))
}
fn attention_value_operand(v: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, s, d) = v.dims3()?;
    // The value GEMM has the same FP32 no-op-cast layout hazard as Q/K.
    Ok(v.reshape((b, s, heads, d / heads))?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .contiguous()?)
}
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, r: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, s, d) = q.dims3()?;
    let output_dtype = v.dtype();
    // Explicit FP32 score and value accumulation, then round to the activation dtype.
    let scores = {
        let (q, kt) = attention_qk_operands(q, k, output_dtype, r, heads)?;
        (q.matmul(&kt)? / ((d / heads) as f64).sqrt())?
    };
    let p = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(p.matmul(&attention_value_operand(v, heads)?)?
        .to_dtype(output_dtype)?
        .transpose(1, 2)?
        .reshape((b, s, d))?)
}
struct TextBlock {
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    qkv: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    heads: usize,
}
impl TextBlock {
    fn load(w: &Weights, p: &str, g: Geometry, keys: &mut BTreeSet<String>) -> Result<Self> {
        Ok(Self {
            n1: Norm::load(w, &format!("{p}.norm1"), g.dim, keys)?,
            n2: Norm::load(w, &format!("{p}.norm2"), g.dim, keys)?,
            qn: Norm::load(w, &format!("{p}.q_norm"), g.dim / g.heads, keys)?,
            kn: Norm::load(w, &format!("{p}.k_norm"), g.dim / g.heads, keys)?,
            qkv: Linear::load(w, &format!("{p}.qkv"), g.dim, 3 * g.dim, true, None, keys)?,
            o: Linear::load(w, &format!("{p}.proj"), g.dim, g.dim, true, None, keys)?,
            gate: Linear::load(w, &format!("{p}.ffn_gate"), g.dim, g.ffn, true, None, keys)?,
            up: Linear::load(w, &format!("{p}.ffn_up"), g.dim, g.ffn, true, None, keys)?,
            down: Linear::load(w, &format!("{p}.ffn_down"), g.ffn, g.dim, true, None, keys)?,
            heads: g.heads,
        })
    }
    fn forward(&self, x: &Tensor, r: &Tensor) -> Result<Tensor> {
        let qkv = self.qkv.forward(&self.n1.forward(x)?)?;
        let (b, s, three) = qkv.dims3()?;
        let d = three / 3;
        let q = self
            .qn
            .forward(
                &qkv.narrow(2, 0, d)?
                    .reshape((b, s, self.heads, d / self.heads))?,
            )?
            .reshape((b, s, d))?;
        let k = self
            .kn
            .forward(
                &qkv.narrow(2, d, d)?
                    .reshape((b, s, self.heads, d / self.heads))?,
            )?
            .reshape((b, s, d))?;
        let x = x.add(&self.o.forward(&attention(
            &q,
            &k,
            &qkv.narrow(2, 2 * d, d)?,
            r,
            self.heads,
        )?)?)?;
        let y = self.n2.forward(&x)?;
        Ok(x.add(
            &self
                .down
                .forward(&situ(&self.up.forward(&y)?, &self.gate.forward(&y)?)?)?,
        )?)
    }
}
struct Block {
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up_gate: Linear,
    down: Linear,
    heads: usize,
}
impl Block {
    fn load(
        w: &Weights,
        p: &str,
        g: Geometry,
        a: Option<&Weights>,
        keys: &mut BTreeSet<String>,
    ) -> Result<Self> {
        Ok(Self {
            n1: Norm::load(w, &format!("{p}.norm1"), g.dim, keys)?,
            n2: Norm::load(w, &format!("{p}.norm2"), g.dim, keys)?,
            qn: Norm::load(w, &format!("{p}.q_norm"), g.dim / g.heads, keys)?,
            kn: Norm::load(w, &format!("{p}.k_norm"), g.dim / g.heads, keys)?,
            q: Linear::load(w, &format!("{p}.proj.q"), g.dim, g.dim, false, a, keys)?,
            k: Linear::load(w, &format!("{p}.proj.k"), g.dim, g.dim, false, a, keys)?,
            v: Linear::load(w, &format!("{p}.proj.v"), g.dim, g.dim, false, a, keys)?,
            o: Linear::load(w, &format!("{p}.proj.o"), g.dim, g.dim, false, a, keys)?,
            up_gate: Linear::load(
                w,
                &format!("{p}.proj.ffn_up_gate"),
                g.dim,
                2 * g.ffn,
                false,
                a,
                keys,
            )?,
            down: Linear::load(
                w,
                &format!("{p}.proj.ffn_down"),
                g.ffn,
                g.dim,
                false,
                a,
                keys,
            )?,
            heads: g.heads,
        })
    }
    fn forward(&self, x: &Tensor, r: &Tensor, m: &Tensor) -> Result<Tensor> {
        let part = |i| m.i((.., i, ..))?.unsqueeze(1);
        let dtype = if x.dtype() == DType::F32 || m.dtype() == DType::F32 {
            DType::F32
        } else {
            x.dtype()
        };
        let y = self
            .n1
            .forward(x)?
            .to_dtype(dtype)?
            .broadcast_mul(&(part(1)?.to_dtype(dtype)? + 1.0)?)?
            .broadcast_add(&part(0)?.to_dtype(dtype)?)?
            .to_dtype(self.q.dtype())?;
        let (b, s, d) = y.dims3()?;
        let q = self
            .qn
            .forward(
                &self
                    .q
                    .forward(&y)?
                    .reshape((b, s, self.heads, d / self.heads))?,
            )?
            .reshape((b, s, d))?;
        let k = self
            .kn
            .forward(
                &self
                    .k
                    .forward(&y)?
                    .reshape((b, s, self.heads, d / self.heads))?,
            )?
            .reshape((b, s, d))?;
        let residual = self
            .o
            .forward(&attention(&q, &k, &self.v.forward(&y)?, r, self.heads)?)?
            .broadcast_mul(&part(2)?.to_dtype(q.dtype())?)?;
        let dtype = residual.dtype();
        let x = x.to_dtype(dtype)?.add(&residual)?;
        let y = self
            .n2
            .forward(&x)?
            .broadcast_mul(&(part(4)?.to_dtype(dtype)? + 1.0)?)?
            .broadcast_add(&part(3)?.to_dtype(dtype)?)?
            .to_dtype(self.q.dtype())?;
        let hg = self.up_gate.forward(&y)?;
        let f = hg.dim(2)? / 2;
        let residual = self
            .down
            .forward(&situ(&hg.narrow(2, 0, f)?, &hg.narrow(2, f, f)?)?)?;
        Ok(x.to_dtype(residual.dtype())?
            .add(&residual.broadcast_mul(&part(5)?.to_dtype(residual.dtype())?)?)?)
    }
}

pub(crate) struct Transformer {
    resident_bytes: u64,
    g: Geometry,
    video: Linear,
    text: Linear,
    text_blocks: Vec<TextBlock>,
    time1: Linear,
    time2: Linear,
    shared: Linear,
    head: Linear,
    layers: Vec<Block>,
    norm: Norm,
    final_linear: Linear,
}
impl Transformer {
    pub(crate) fn channels(&self) -> usize {
        self.g.channels
    }
    pub(crate) fn device(&self) -> &Device {
        self.video.base.weight().device()
    }
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    pub(crate) fn load(w: &Weights, g: Geometry, adapter: Option<&Weights>) -> Result<Self> {
        let mut keys = BTreeSet::new();
        let mut result = Self {
            resident_bytes: 0,
            g,
            video: Linear::load(
                w,
                "video_embedder",
                g.channels,
                g.dim,
                true,
                None,
                &mut keys,
            )?,
            text: Linear::load(
                w,
                "text_embedder.proj",
                g.text,
                g.dim,
                true,
                None,
                &mut keys,
            )?,
            text_blocks: (0..2)
                .map(|i| TextBlock::load(w, &format!("text_embedder.blocks.{i}"), g, &mut keys))
                .collect::<Result<_>>()?,
            time1: Linear::load(w, "time_embedding.0", g.freq, g.dim, true, None, &mut keys)?,
            time2: Linear::load(w, "time_embedding.2", g.dim, g.dim, true, None, &mut keys)?,
            shared: Linear::load(
                w,
                "shared_block_modulation",
                g.dim,
                6 * g.dim,
                true,
                None,
                &mut keys,
            )?,
            head: Linear::load(
                w,
                "video_head_modulation",
                g.dim,
                2 * g.dim,
                true,
                None,
                &mut keys,
            )?,
            layers: (0..g.layers)
                .map(|i| Block::load(w, &format!("layers.{i}"), g, adapter, &mut keys))
                .collect::<Result<_>>()?,
            norm: Norm::load(w, "final_norm_video", g.dim, &mut keys)?,
            final_linear: Linear::load(
                w,
                "final_linear_video",
                g.dim,
                g.channels,
                false,
                None,
                &mut keys,
            )?,
        };
        for key in &keys {
            let t = w.require(key)?;
            result.resident_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        }
        if let Some(adapter) = adapter {
            for key in adapter.keys().filter(|k| !k.ends_with(".alpha")) {
                let t = adapter.require(key)?;
                result.resident_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            }
        }
        // Derived RoPE bands are checkpoint-compatible but are reconstructed in source FP32.
        keys.insert("rope.bands".into());
        let extra: Vec<_> = w.keys().filter(|k| !keys.contains(*k)).collect();
        if !extra.is_empty() {
            return Err(CandleError::Msg(format!(
                "MOSS unconsumed transformer tensors {extra:?}"
            )));
        }
        Ok(result)
    }
    pub(crate) fn project_context(&self, context: &Tensor) -> Result<Tensor> {
        self.text.forward(&context.to_dtype(self.text.dtype())?)
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        time: f32,
        context: &Tensor,
        mask: &[u8],
        skip: &[u32],
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        check_cancel(cancel)?;
        let (batch, c, h, w) = x.dims4()?;
        let (cb, s, td) = context.dims3()?;
        if c != self.g.channels
            || cb != batch
            || td != self.g.text
            || mask.len() != batch * s
            || skip.iter().any(|i| *i as usize >= self.g.layers)
        {
            return Err(CandleError::Msg("MOSS forward geometry/mask/skip".into()));
        }
        let features = if self.g.freq == 256 {
            candle_gen::gen_core::moss_image::moss_time_features(time)?.to_vec()
        } else {
            let half = self.g.freq / 2;
            let phase: Vec<_> = (0..half)
                .map(|i| f64::from(time) * 10000_f64.powf(-(i as f64) / half as f64))
                .collect();
            phase
                .iter()
                .map(|p| p.cos() as f32)
                .chain(phase.iter().map(|p| p.sin() as f32))
                .collect()
        };
        let t = Tensor::from_vec(features, (1, self.g.freq), x.device())?
            .to_dtype(self.time1.dtype())?;
        let t = self
            .time2
            .forward(&candle_nn::ops::silu(&self.time1.forward(&t)?)?)?
            .to_dtype(DType::F32)?;
        let m = self
            .shared
            .forward(&t.to_dtype(self.shared.dtype())?)?
            .reshape((1, 6, self.g.dim))?;
        let head = self
            .head
            .forward(&t.to_dtype(self.head.dtype())?)?
            .reshape((1, 2, self.g.dim))?;
        let mut result = Vec::with_capacity(batch);
        for b in 0..batch {
            check_cancel(cancel)?;
            let indices: Vec<u32> = mask[b * s..(b + 1) * s]
                .iter()
                .enumerate()
                .filter_map(|(i, v)| (*v == 1).then_some(i as u32))
                .collect();
            if indices.is_empty() || mask[b * s..(b + 1) * s].iter().any(|v| *v > 1) {
                return Err(CandleError::Msg("MOSS empty/nonbinary mask".into()));
            }
            let ctx = context.narrow(0, b, 1)?.index_select(
                &Tensor::from_vec(indices.clone(), indices.len(), x.device())?,
                1,
            )?;
            let video = x
                .narrow(0, b, 1)?
                .reshape((1, c, h * w))?
                .transpose(1, 2)?
                .to_dtype(self.video.dtype())?
                .contiguous()?;
            let video = self.video.forward(&video)?;
            let r = rope(
                h,
                w,
                indices.len(),
                self.g.dim / self.g.heads,
                video.dtype(),
                x.device(),
            )?;
            let mut text = self.project_context(&ctx)?;
            for block in &self.text_blocks {
                check_cancel(cancel)?;
                text = block.forward(&text, &r.narrow(1, h * w, indices.len())?)?;
                x.device().synchronize()?;
                check_cancel(cancel)?;
            }
            let mut y = Tensor::cat(&[video, text], 1)?;
            for (i, block) in self.layers.iter().enumerate() {
                check_cancel(cancel)?;
                if !skip.contains(&(i as u32)) {
                    y = block.forward(&y, &r, &m)?;
                    x.device().synchronize()?;
                    check_cancel(cancel)?;
                }
            }
            let y = self
                .norm
                .forward(&y.narrow(1, 0, h * w)?.to_dtype(self.norm.0.dtype())?)?;
            let dtype = if y.dtype() == DType::F32 || head.dtype() == DType::F32 {
                DType::F32
            } else {
                y.dtype()
            };
            let y = y
                .to_dtype(dtype)?
                .broadcast_mul(&(head.i((.., 1, ..))?.unsqueeze(1)?.to_dtype(dtype)? + 1.0)?)?
                .broadcast_add(&head.i((.., 0, ..))?.unsqueeze(1)?.to_dtype(dtype)?)?;
            result.push(
                self.final_linear
                    .forward(&y.to_dtype(self.final_linear.dtype())?)?
                    .transpose(1, 2)?
                    .reshape((1, c, h, w))?,
            );
        }
        Ok(Tensor::cat(&result, 0)?)
    }
}

/// Promote only source-mandated FP32 islands, then replace all six published tensors.
pub(crate) fn install_replacements(base: &Weights, replacements: &Weights) -> Result<Weights> {
    let mut map = HashMap::new();
    for key in base.keys() {
        let mut t = base.require(key)?;
        if [
            "time_embedding.",
            "shared_block_modulation.",
            "video_head_modulation.",
        ]
        .iter()
        .any(|p| key.starts_with(p))
        {
            t = t.to_dtype(DType::F32)?;
        }
        if replacements.contains(key) {
            t = replacements.require(key)?;
        }
        map.insert(key.clone(), t);
    }
    Ok(Weights::from_map(map))
}

#[cfg(test)]
pub(crate) fn test_projection(
    w: &Weights,
    name: &str,
    dim: usize,
    adapter: Option<&Weights>,
    input: &Tensor,
) -> Result<Tensor> {
    Linear::load(w, name, dim, dim, false, adapter, &mut BTreeSet::new())?.forward(input)
}

#[cfg(test)]
pub(crate) fn test_norm(x: &Tensor, weight: Tensor) -> Result<Tensor> {
    Norm(weight).forward(x)
}

#[cfg(test)]
mod attention_layout_tests {
    use super::*;

    const SEQUENCE: usize = 5;
    const HEADS: usize = 3;
    const WIDTH: usize = 8;

    fn inputs(
        device: &Device,
        batch: usize,
        qk_dtype: DType,
        value_dtype: DType,
        rope_dtype: DType,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let input = |offset: usize, dtype| {
            let values: Vec<f32> = (0..batch * SEQUENCE * HEADS * WIDTH)
                .map(|i| (((i * 7 + offset) % 29) as f32 - 14.) / 32.)
                .collect();
            Tensor::from_vec(values, (batch, SEQUENCE, HEADS * WIDTH), device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        };
        (
            input(0, qk_dtype),
            input(3, qk_dtype),
            input(11, value_dtype),
            rope(1, 2, 3, WIDTH, rope_dtype, device).unwrap(),
        )
    }

    fn policies() -> [(DType, DType, DType); 4] {
        [
            (DType::BF16, DType::BF16, DType::BF16),
            (DType::F32, DType::BF16, DType::BF16),
            (DType::F32, DType::F32, DType::BF16),
            (DType::F32, DType::F32, DType::F32),
        ]
    }

    #[test]
    fn production_attention_operands_pack_each_head_for_both_matmuls() {
        for batch in [1, 2] {
            for (qk_dtype, value_dtype, rope_dtype) in policies() {
                let (q, k, v, r) = inputs(&Device::Cpu, batch, qk_dtype, value_dtype, rope_dtype);
                let (q, kt) = attention_qk_operands(&q, &k, v.dtype(), &r, HEADS).unwrap();
                let v = attention_value_operand(&v, HEADS).unwrap();
                // These are the actual operands consumed by attention, not a CPU numerical
                // proxy: CPU matmul accepts the interleaved strides that CUDA rejects.
                let packed = [HEADS * SEQUENCE * WIDTH, SEQUENCE * WIDTH, WIDTH, 1];
                assert_eq!(q.stride(), packed, "Q must pack its head matrices");
                assert_eq!(v.stride(), packed, "V must pack its head matrices");
                assert_eq!(
                    kt.stride(),
                    [HEADS * SEQUENCE * WIDTH, SEQUENCE * WIDTH, 1, WIDTH],
                    "K must pack its head matrices before its cuBLAS-supported transpose"
                );
                assert_eq!(q.dims(), [batch, HEADS, SEQUENCE, WIDTH]);
                assert_eq!(kt.dims(), [batch, HEADS, WIDTH, SEQUENCE]);
                assert_eq!(v.dims(), q.dims());
                for operand in [&q, &kt, &v] {
                    assert_eq!(operand.dtype(), DType::F32);
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an explicitly admitted CUDA device; no model weights"]
    fn production_attention_cuda_accepts_ordinary_and_dmd_layouts() {
        let device = Device::new_cuda(0).expect("admitted logical CUDA device 0");
        for batch in [1, 2] {
            for (qk_dtype, value_dtype, rope_dtype) in policies() {
                let (q, k, v, r) = inputs(&device, batch, qk_dtype, value_dtype, rope_dtype);
                let actual = attention(&q, &k, &v, &r, HEADS).unwrap();
                let (q, k, v, r) = inputs(&Device::Cpu, batch, qk_dtype, value_dtype, rope_dtype);
                let expected = attention(&q, &k, &v, &r, HEADS).unwrap();
                assert_eq!(actual.dims(), [batch, SEQUENCE, HEADS * WIDTH]);
                assert_eq!(actual.dtype(), value_dtype);
                let flatten = |x: Tensor| {
                    x.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                };
                let actual = flatten(actual);
                let expected = flatten(expected);
                assert_eq!(actual.len(), expected.len());
                let tolerance = if value_dtype == DType::BF16 {
                    0.005
                } else {
                    0.0001
                };
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!(actual.is_finite() && expected.is_finite());
                    assert!((actual - expected).abs() <= tolerance);
                }
            }
        }
    }
}
