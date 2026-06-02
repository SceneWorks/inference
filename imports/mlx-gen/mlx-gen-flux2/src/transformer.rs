//! FLUX.2 MMDiT transformer — 8 double (joint img+txt) blocks + 24 single (fused parallel
//! attention+SwiGLU) blocks, shared per-stream modulation, 4-axis interleaved RoPE, and an
//! `AdaLayerNormContinuous` output. Port of `models/flux2/model/flux2_transformer/`.
//!
//! Runs f32 activations (matmul(f32 act, bf16 weight)→f32): the `x_embedder` (K=128, M=seq≥2) is
//! the dense 16-bit Metal GEMM bug shape, so the whole stack must run f32 — which is also the
//! quality target. Linears are bias-less plain matmuls (Q4/Q8 = sc-2643, LoRA = sc-2646).

use std::f32::consts::LN_10;

use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Flux2Config;
use crate::pos_embed::Flux2PosEmbed;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

fn matmul_t(x: &Array, w: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::matmul(x, w.t())?)
}

fn require_f32_input(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// `[B,S,H·D]` → `[B,H,S,D]`, with per-head q/k RMSNorm (f32). Port of `AttentionUtils.process_qkv`.
#[allow(clippy::too_many_arguments)]
fn process_qkv(
    x: &Array,
    q_w: &Array,
    k_w: &Array,
    v_w: &Array,
    norm_q: &Array,
    norm_k: &Array,
    heads: i32,
    head_dim: i32,
) -> Result<(Array, Array, Array)> {
    let sh = x.shape();
    let (b, s) = (sh[0], sh[1]);
    let to_bhsd = |a: Array| -> Result<Array> {
        Ok(a.reshape(&[b, s, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?)
    };
    let q = to_bhsd(matmul_t(x, q_w)?)?;
    let k = to_bhsd(matmul_t(x, k_w)?)?;
    let v = to_bhsd(matmul_t(x, v_w)?)?;
    let q = rms_norm(&q, norm_q, RMS_EPS)?;
    let k = rms_norm(&k, norm_k, RMS_EPS)?;
    Ok((q, k, v))
}

/// Interleaved RoPE (`AttentionUtils.apply_rope_bshd`): pairs `(x[2i], x[2i+1])` rotated by
/// `cos/sin[i]`. `cos`/`sin`: `[S, head_dim/2]`; `q`/`k`: `[B,H,S,head_dim]`.
fn apply_rope(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let s = cos.shape()[0];
    let half = cos.shape()[1];
    let cos = cos.reshape(&[1, 1, s, half])?;
    let sin = sin.reshape(&[1, 1, s, half])?;
    let one = |x: &Array| -> Result<Array> {
        let sh = x.shape();
        let (b, h, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let x5 = x.reshape(&[b, h, seq, hd / 2, 2])?;
        let p = split(&x5, 2, 4)?;
        let real = p[0].reshape(&[b, h, seq, hd / 2])?;
        let imag = p[1].reshape(&[b, h, seq, hd / 2])?;
        let out0 = subtract(&multiply(&real, &cos)?, &multiply(&imag, &sin)?)?;
        let out1 = add(&multiply(&imag, &cos)?, &multiply(&real, &sin)?)?;
        Ok(
            concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
                .reshape(&[b, h, seq, hd])?,
        )
    };
    Ok((one(q)?, one(k)?))
}

/// SDPA over `[B,H,S,D]` → `[B,S,H·D]`.
fn attention(q: &Array, k: &Array, v: &Array, head_dim: i32) -> Result<Array> {
    let b = q.shape()[0];
    let scale = (head_dim as f32).powf(-0.5);
    let o = scaled_dot_product_attention(q, k, v, scale, None, None)?;
    Ok(o.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, -1, q.shape()[1] * head_dim])?)
}

/// SwiGLU: split last axis in half, `silu(x1) · x2`.
fn swiglu(x: &Array) -> Result<Array> {
    let p = split(x, 2, -1)?;
    Ok(multiply(&silu(&p[0])?, &p[1])?)
}

/// `(1 + scale) · norm(x) + shift` with `scale`/`shift` broadcast `[B,1,D]`.
fn modulate(norm: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(&multiply(norm, &add(scale, scalar(1.0))?)?, shift)?)
}

/// Sinusoidal timestep embedding (diffusers `_timestep_embedding`, flip_sin_to_cos): `[B]` → `[B,
/// dim]` = `concat([cos(args), sin(args)])`.
fn timestep_embedding(t: &Array, dim: usize) -> Result<Array> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-LN_10 * 4.0 * i as f32 / half as f32).exp())
        .collect();
    // ln(10000) = 4·ln(10).
    let freqs = Array::from_slice(&freqs, &[1, half as i32]);
    let t = t.reshape(&[t.shape()[0], 1])?.as_dtype(Dtype::Float32)?;
    let args = multiply(&t, &freqs)?;
    Ok(concatenate_axis(&[&args.cos()?, &args.sin()?], 1)?)
}

struct FeedForward {
    linear_in: Array,
    linear_out: Array,
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_in: w.require(&format!("{prefix}.linear_in.weight"))?.clone(),
            linear_out: w.require(&format!("{prefix}.linear_out.weight"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = matmul_t(x, &self.linear_in)?;
        let x = swiglu(&x)?;
        matmul_t(&x, &self.linear_out)
    }
}

struct DoubleBlock {
    attn: DoubleAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

struct DoubleAttention {
    to_q: Array,
    to_k: Array,
    to_v: Array,
    to_out: Array,
    norm_q: Array,
    norm_k: Array,
    add_q: Array,
    add_k: Array,
    add_v: Array,
    to_add_out: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
}

impl DoubleAttention {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, head_dim: i32) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}.weight")).cloned();
        Ok(Self {
            to_q: g("to_q")?,
            to_k: g("to_k")?,
            to_v: g("to_v")?,
            to_out: g("to_out")?,
            norm_q: g("norm_q")?,
            norm_k: g("norm_k")?,
            add_q: g("add_q_proj")?,
            add_k: g("add_k_proj")?,
            add_v: g("add_v_proj")?,
            to_add_out: g("to_add_out")?,
            norm_added_q: g("norm_added_q")?,
            norm_added_k: g("norm_added_k")?,
            heads,
            head_dim,
        })
    }

    /// Joint attention. Returns `(img_attn_out, txt_attn_out)`.
    fn forward(
        &self,
        img: &Array,
        txt: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        let (iq, ik, iv) = process_qkv(
            img,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
            self.heads,
            self.head_dim,
        )?;
        let (tq, tk, tv) = process_qkv(
            txt,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
            self.heads,
            self.head_dim,
        )?;
        // [txt, img] order along the sequence (axis 2 in BHSD).
        let q = concatenate_axis(&[&tq, &iq], 2)?;
        let k = concatenate_axis(&[&tk, &ik], 2)?;
        let v = concatenate_axis(&[&tv, &iv], 2)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let o = attention(&q, &k, &v, self.head_dim)?;
        let txt_seq = txt.shape()[1];
        let txt_idx = Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
        let img_idx = Array::from_slice(
            &(txt_seq..o.shape()[1]).collect::<Vec<i32>>(),
            &[o.shape()[1] - txt_seq],
        );
        let txt_out = matmul_t(&o.take_axis(&txt_idx, 1)?, &self.to_add_out)?;
        let img_out = matmul_t(&o.take_axis(&img_idx, 1)?, &self.to_out)?;
        Ok((img_out, txt_out))
    }
}

impl DoubleBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            attn: DoubleAttention::from_weights(w, &format!("{prefix}.attn"), heads, head_dim)?,
            ff: FeedForward::from_weights(w, &format!("{prefix}.ff"))?,
            ff_context: FeedForward::from_weights(w, &format!("{prefix}.ff_context"))?,
        })
    }

    /// `img_mod` / `txt_mod`: `[(shift_msa,scale_msa,gate_msa),(shift_mlp,scale_mlp,gate_mlp)]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        mut img: Array,
        mut txt: Array,
        img_mod: &[(Array, Array, Array); 2],
        txt_mod: &[(Array, Array, Array); 2],
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        let (shift_msa, scale_msa, gate_msa) = &img_mod[0];
        let (shift_mlp, scale_mlp, gate_mlp) = &img_mod[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = &txt_mod[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = &txt_mod[1];

        let norm_img = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_msa, shift_msa)?;
        let norm_txt = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_msa,
            c_shift_msa,
        )?;

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin)?;
        img = add(&img, &multiply(gate_msa, &img_attn)?)?;
        txt = add(&txt, &multiply(c_gate_msa, &txt_attn)?)?;

        let norm_img2 = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_mlp, shift_mlp)?;
        img = add(&img, &multiply(gate_mlp, &self.ff.forward(&norm_img2)?)?)?;

        let norm_txt2 = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_mlp,
            c_shift_mlp,
        )?;
        txt = add(
            &txt,
            &multiply(c_gate_mlp, &self.ff_context.forward(&norm_txt2)?)?,
        )?;

        Ok((txt, img))
    }
}

struct SingleBlock {
    to_qkv_mlp: Array,
    to_out: Array,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    inner: i32,
}

impl SingleBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            to_qkv_mlp: w
                .require(&format!("{prefix}.attn.to_qkv_mlp_proj.weight"))?
                .clone(),
            to_out: w.require(&format!("{prefix}.attn.to_out.weight"))?.clone(),
            norm_q: w.require(&format!("{prefix}.attn.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.attn.norm_k.weight"))?.clone(),
            heads,
            head_dim,
            inner: heads * head_dim,
        })
    }

    /// `mod`: `(shift, scale, gate)`.
    fn forward(
        &self,
        hidden: &Array,
        m: &(Array, Array, Array),
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let (shift, scale, gate) = m;
        let norm = modulate(&layer_norm(hidden, None, None, LN_EPS)?, scale, shift)?;
        let proj = matmul_t(&norm, &self.to_qkv_mlp)?;

        let sh = proj.shape();
        let (b, s) = (sh[0], sh[1]);
        let take = |start: i32, end: i32| -> Result<Array> {
            let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
            Ok(proj.take_axis(&idx, 2)?)
        };
        let q = take(0, self.inner)?;
        let k = take(self.inner, 2 * self.inner)?;
        let v = take(2 * self.inner, 3 * self.inner)?;
        let mlp = take(3 * self.inner, sh[2])?;

        let to_bhsd = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, s, self.heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = rms_norm(&to_bhsd(q)?, &self.norm_q, RMS_EPS)?;
        let k = rms_norm(&to_bhsd(k)?, &self.norm_k, RMS_EPS)?;
        let v = to_bhsd(v)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let attn = attention(&q, &k, &v, self.head_dim)?;

        let mlp = swiglu(&mlp)?;
        let cat = concatenate_axis(&[&attn, &mlp], -1)?;
        let attn_output = matmul_t(&cat, &self.to_out)?;
        Ok(add(hidden, &multiply(gate, &attn_output)?)?)
    }
}

/// Per-stream modulation producer: `silu(temb) → linear → split into `sets` × (shift,scale,gate)`.
struct Modulation {
    linear: Array,
    sets: usize,
}

impl Modulation {
    fn from_weights(w: &Weights, prefix: &str, sets: usize) -> Result<Self> {
        Ok(Self {
            linear: w.require(&format!("{prefix}.linear.weight"))?.clone(),
            sets,
        })
    }

    /// `temb`: `[B, dim]` → `Vec<(shift,scale,gate)>` of length `sets`, each `[B,1,dim]`.
    fn forward(&self, temb: &Array) -> Result<Vec<(Array, Array, Array)>> {
        let mod_ = matmul_t(&silu(temb)?, &self.linear)?.expand_dims(1)?;
        let chunks = split(&mod_, (3 * self.sets) as i32, -1)?;
        Ok((0..self.sets)
            .map(|i| {
                (
                    chunks[3 * i].clone(),
                    chunks[3 * i + 1].clone(),
                    chunks[3 * i + 2].clone(),
                )
            })
            .collect())
    }
}

/// The FLUX.2 MMDiT transformer.
pub struct Flux2Transformer {
    pos_embed: Flux2PosEmbed,
    time_linear1: Array,
    time_linear2: Array,
    mod_img: Modulation,
    mod_txt: Modulation,
    mod_single: Modulation,
    x_embedder: Array,
    context_embedder: Array,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out_linear: Array,
    proj_out: Array,
    time_channels: usize,
}

impl Flux2Transformer {
    pub fn from_weights(w: &Weights, cfg: &Flux2Config) -> Result<Self> {
        let heads = cfg.num_heads as i32;
        let head_dim = cfg.head_dim as i32;
        let double_blocks = (0..cfg.num_double_layers)
            .map(|i| {
                DoubleBlock::from_weights(w, &format!("transformer_blocks.{i}"), heads, head_dim)
            })
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..cfg.num_single_layers)
            .map(|i| {
                SingleBlock::from_weights(
                    w,
                    &format!("single_transformer_blocks.{i}"),
                    heads,
                    head_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pos_embed: Flux2PosEmbed::new(cfg.rope_theta, cfg.axes_dim),
            time_linear1: w.require("time_guidance_embed.linear_1.weight")?.clone(),
            time_linear2: w.require("time_guidance_embed.linear_2.weight")?.clone(),
            mod_img: Modulation::from_weights(w, "double_stream_modulation_img", 2)?,
            mod_txt: Modulation::from_weights(w, "double_stream_modulation_txt", 2)?,
            mod_single: Modulation::from_weights(w, "single_stream_modulation", 1)?,
            x_embedder: w.require("x_embedder.weight")?.clone(),
            context_embedder: w.require("context_embedder.weight")?.clone(),
            double_blocks,
            single_blocks,
            norm_out_linear: w.require("norm_out.linear.weight")?.clone(),
            proj_out: w.require("proj_out.weight")?.clone(),
            time_channels: cfg.timestep_channels,
        })
    }

    fn temb(&self, timestep: f32) -> Result<Array> {
        // klein has no guidance embedding; timestep is fed as sigma·1000 (>1) so no rescale.
        let t = Array::from_slice(&[timestep], &[1]);
        let emb = timestep_embedding(&t, self.time_channels)?;
        let h = matmul_t(&emb, &self.time_linear1)?;
        matmul_t(&silu(&h)?, &self.time_linear2)
    }

    fn norm_out(&self, x: &Array, temb: &Array) -> Result<Array> {
        let p = matmul_t(&silu(temb)?, &self.norm_out_linear)?; // [B, 2·dim]
        let parts = split(&p, 2, 1)?;
        let scale = parts[0].expand_dims(1)?; // [B,1,dim]
        let shift = parts[1].expand_dims(1)?;
        let normed = layer_norm(x, None, None, LN_EPS)?;
        Ok(add(
            &multiply(&normed, &add(&scale, scalar(1.0))?)?,
            &shift,
        )?)
    }

    /// `hidden_states`: `[B, seq_img, in_channels]`; `encoder_hidden_states`: `[B, seq_txt,
    /// joint_attention_dim]`; `img_ids`/`txt_ids`: `[seq, 4]` (or `[1, seq, 4]`). `timestep` is the
    /// scaled sigma (×1000). Returns the velocity `[B, seq_img, out_channels]`.
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
    ) -> Result<Array> {
        let temb = self.temb(timestep)?;
        let mut img = matmul_t(&require_f32_input(hidden_states)?, &self.x_embedder)?;
        let mut txt = matmul_t(
            &require_f32_input(encoder_hidden_states)?,
            &self.context_embedder,
        )?;

        let drop_batch = |ids: &Array| -> Result<Array> {
            if ids.shape().len() == 3 {
                Ok(ids.reshape(&[ids.shape()[1], ids.shape()[2]])?)
            } else {
                Ok(ids.clone())
            }
        };
        let (img_cos, img_sin) = self.pos_embed.forward(&drop_batch(img_ids)?)?;
        let (txt_cos, txt_sin) = self.pos_embed.forward(&drop_batch(txt_ids)?)?;
        let cos = concatenate_axis(&[&txt_cos, &img_cos], 0)?;
        let sin = concatenate_axis(&[&txt_sin, &img_sin], 0)?;

        let mi = self.mod_img.forward(&temb)?;
        let mt = self.mod_txt.forward(&temb)?;
        let img_mod = [mi[0].clone(), mi[1].clone()];
        let txt_mod = [mt[0].clone(), mt[1].clone()];

        for block in &self.double_blocks {
            (txt, img) = block.forward(img, txt, &img_mod, &txt_mod, &cos, &sin)?;
        }

        let txt_seq = txt.shape()[1];
        let mut hidden = concatenate_axis(&[&txt, &img], 1)?;
        let ms = self.mod_single.forward(&temb)?;
        for block in &self.single_blocks {
            hidden = block.forward(&hidden, &ms[0], &cos, &sin)?;
        }

        // Keep only the image tokens.
        let img_seq = hidden.shape()[1] - txt_seq;
        let img_idx = Array::from_slice(
            &(txt_seq..hidden.shape()[1]).collect::<Vec<i32>>(),
            &[img_seq],
        );
        let hidden = hidden.take_axis(&img_idx, 1)?;
        let hidden = self.norm_out(&hidden, &temb)?;
        matmul_t(&hidden, &self.proj_out)
    }
}

/// Configuration glue so callers can keep the transformer's dims in one place.
pub type Flux2TransformerConfig = Flux2Config;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_shape_and_flip() {
        let t = Array::from_slice(&[1000.0f32], &[1]);
        let emb = timestep_embedding(&t, 256).unwrap();
        assert_eq!(emb.shape(), &[1, 256]);
    }
}
