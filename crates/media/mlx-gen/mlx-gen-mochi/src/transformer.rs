//! Mochi 1 **AsymmDiT** denoiser — port of `MochiTransformer3DModel` + `MochiTransformerBlock` +
//! `MochiAttnProcessor2_0` (diffusers `transformer_mochi.py` / `attention_processor.py`).
//!
//! A dual-stream MMDiT: a **visual** stream (patch-embedded latent tokens, `inner_dim = 3072`) and a
//! **text** stream (caption-projected T5 tokens, `pooled_projection_dim = 1536`) that interact only
//! through a single **joint** attention per block. Each block:
//!
//!  1. modulates both streams with `MochiRMSNormZero` (weightless f32 RMS-norm → `(1 + scale)`);
//!  2. runs joint attention — visual `to_{q,k,v}` (3072→3072) and text `add_{q,k,v}_proj`
//!     (1536→3072), per-head `qk_norm` (weighted RMS, eps 1e-5) on q/k **and** the added q/k, learned
//!     3-D RoPE on the **visual** q/k only, then one masked SDPA over the concatenated
//!     `[visual | text]` keys (padded text keys get additive `−inf`), split back to `to_out`
//!     (3072→3072) + `to_add_out` (3072→1536);
//!  3. applies **tanh-gated** dual residuals (`MochiModulatedRMSNorm`) and a SwiGLU FFN per stream.
//!
//! The final block is `context_pre_only` — it drops the text-stream output path (no `to_add_out` /
//! `ff_context`, and `norm1_context` is a `MochiLayerNormContinuous` instead). The whole model runs in
//! **f32** here (the reference runs bf16; f32 is the high-precision truth the bf16 goldens are a
//! rounding of — the same stance as the T5 `te_parity`), with RoPE/norms already f32 in the reference.

use std::path::Path;

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, concatenate_axis, matmul, mean_axis, multiply, rsqrt, split, tanh};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::silu;
use mlx_gen::weights::{join, Weights};
use mlx_gen::{Error, Result};

use crate::rope::MochiRope;

/// Load the AsymmDiT transformer weights from `<root>/transformer/` — the **bf16** variant shards
/// referenced by `diffusion_pytorch_model.safetensors.index.bf16.json` (the f32 set on the hub is
/// incomplete; the reference loads `variant="bf16"`). Weights are returned **as-is** (bf16); each
/// module casts the tensors it reads to its working dtype at construction, so only the tensors a given
/// block/model actually touches are upcast (block_parity builds one block → casts one block's weights).
pub fn load_transformer_weights(root: &Path) -> Result<Weights> {
    let dir = root.join("transformer");
    let index = dir.join("diffusion_pytorch_model.safetensors.index.bf16.json");
    if !index.exists() {
        return Weights::from_dir(&dir);
    }
    let text = std::fs::read_to_string(&index)
        .map_err(|e| Error::Msg(format!("mochi dit index {}: {e}", index.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("mochi dit index {}: {e}", index.display())))?;
    let map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::Msg(format!("mochi dit index {}: no weight_map", index.display())))?;

    let mut shard_files: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    let mut combined = Weights::empty();
    for f in shard_files {
        let shard = Weights::from_file(dir.join(&f))?;
        let keys: Vec<String> = shard.keys().map(String::from).collect();
        for k in keys {
            if let Some(t) = shard.get(&k) {
                combined.insert(k, t.clone());
            }
        }
    }
    Ok(combined)
}

/// AsymmDiT geometry (`transformer/config.json`). `inner_dim = num_heads · head_dim = 3072`.
#[derive(Debug, Clone, Copy)]
pub struct MochiDitConfig {
    pub patch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub pooled_dim: usize,
    pub in_channels: usize,
    pub text_embed_dim: usize,
    pub time_embed_dim: usize,
    /// Normalization epsilon for the block's weightless norms (`1e-6`).
    pub eps: f32,
}

impl Default for MochiDitConfig {
    fn default() -> Self {
        Self {
            patch_size: 2,
            num_heads: 24,
            head_dim: 128,
            num_layers: 48,
            pooled_dim: 1536,
            in_channels: 12,
            text_embed_dim: 4096,
            time_embed_dim: 256,
            eps: 1e-6,
        }
    }
}

impl MochiDitConfig {
    /// `inner_dim = num_heads · head_dim`.
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Per-head `qk_norm` epsilon (`MochiAttention(eps=1e-5)`), distinct from the block's `1e-6`.
const QK_NORM_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------- primitives

/// `y = x · Wᵀ` for a stored `[out, in]` weight, no bias. Batched over any leading dims.
fn linear_nb(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, &w.t())?)
}

/// `y = x · Wᵀ + b` (mlx-gen core fused `addmm`).
fn linear_b(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    mlx_gen::nn::linear(x, w, b)
}

/// Weightless RMS norm over the last axis, computed in f32 (`RMSNorm(0, eps, False)` —
/// `MochiRMSNormZero.norm` / `MochiModulatedRMSNorm.norm`). `x / sqrt(mean(x²) + eps)`.
fn rms_weightless(x: &Array, eps: f32) -> Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let ms = mean_axis(&xf.square()?, -1, true)?;
    Ok(multiply(&xf, &rsqrt(&add(&ms, Array::from_f32(eps))?)?)?)
}

/// Weighted RMS norm over the last axis in f32 (`MochiRMSNorm(dim_head, eps, True)` — the per-head
/// `qk_norm`). `weight` is `[head_dim]`, broadcast over the leading `[B, S, heads]`.
fn rms_weighted(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let normed = rms_weightless(x, eps)?;
    Ok(multiply(&normed, &weight.as_dtype(Dtype::Float32)?)?)
}

/// `emb.chunk(n, dim=1)` — split a `[B, n·d]` modulation vector into `n` `[B, d]` parts (order
/// preserved). Used for the `(scale_msa, gate_msa, scale_mlp, gate_mlp)` unpacking.
fn chunk_last(x: &Array, n: i32) -> Result<Vec<Array>> {
    Ok(split(x, n, x.shape().len() as i32 - 1)?)
}

/// `x[:, None, :]` — insert a length-1 sequence axis so a `[B, d]` modulation broadcasts over
/// `[B, S, d]`.
fn unsqueeze1(x: &Array) -> Result<Array> {
    Ok(x.expand_dims(1)?)
}

// ---------------------------------------------------------------------------- SwiGLU FFN

/// SwiGLU feed-forward (`FeedForward(activation="swiglu", bias=False)`): `proj` (`d → 2·inner`),
/// split into `(value, gate)`, `value · silu(gate)`, then `out` (`inner → d`).
#[derive(Clone)]
struct SwiGlu {
    proj_w: Array, // [2·inner, d]
    out_w: Array,  // [d, inner]
}

impl SwiGlu {
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            proj_w: w
                .require(&join(prefix, "net.0.proj.weight"))?
                .as_dtype(dtype)?,
            out_w: w.require(&join(prefix, "net.2.weight"))?.as_dtype(dtype)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = linear_nb(x, &self.proj_w)?;
        let parts = chunk_last(&h, 2)?;
        let gated = multiply(&parts[0], &silu(&parts[1])?)?;
        linear_nb(&gated, &self.out_w)
    }
}

// ---------------------------------------------------------------------------- attention

/// Mochi joint attention (`MochiAttention` + `MochiAttnProcessor2_0`).
#[derive(Clone)]
pub struct MochiAttention {
    to_q: Array,
    to_k: Array,
    to_v: Array,
    add_q: Array,
    add_k: Array,
    add_v: Array,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    to_out_w: Array,
    to_out_b: Array,
    /// `(weight, bias)` for `to_add_out` — absent when `context_pre_only`.
    to_add_out: Option<(Array, Array)>,
    num_heads: usize,
    head_dim: usize,
}

impl MochiAttention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MochiDitConfig,
        context_pre_only: bool,
        dtype: Dtype,
    ) -> Result<Self> {
        let g = |name: &str| -> Result<Array> {
            Ok(w.require(&join(prefix, name))?.as_dtype(dtype)?)
        };
        let to_add_out = if context_pre_only {
            None
        } else {
            Some((g("to_add_out.weight")?, g("to_add_out.bias")?))
        };
        Ok(Self {
            to_q: g("to_q.weight")?,
            to_k: g("to_k.weight")?,
            to_v: g("to_v.weight")?,
            add_q: g("add_q_proj.weight")?,
            add_k: g("add_k_proj.weight")?,
            add_v: g("add_v_proj.weight")?,
            norm_q: g("norm_q.weight")?,
            norm_k: g("norm_k.weight")?,
            norm_added_q: g("norm_added_q.weight")?,
            norm_added_k: g("norm_added_k.weight")?,
            to_out_w: g("to_out.0.weight")?,
            to_out_b: g("to_out.0.bias")?,
            to_add_out,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Split `[B, S, inner]` → `[B, S, heads, head_dim]`.
    fn to_heads(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        Ok(x.reshape(&[sh[0], sh[1], self.num_heads as i32, self.head_dim as i32])?)
    }

    /// Joint attention. `visual [B, Sv, inner]`, `text [B, St, pooled]`, `enc_mask [B, St]` (0/1).
    /// Returns `(visual_out [B, Sv, inner], Some(text_out [B, St, pooled]))` (text `None` when
    /// `context_pre_only`).
    pub fn forward(
        &self,
        visual: &Array,
        text: &Array,
        rope: &MochiRope,
        enc_mask: &Array,
    ) -> Result<(Array, Option<Array>)> {
        let sv = visual.shape()[1];
        let st = text.shape()[1];

        // Visual q/k/v (+ per-head qk_norm) with RoPE on q/k.
        let q = self.to_heads(&linear_nb(visual, &self.to_q)?)?;
        let k = self.to_heads(&linear_nb(visual, &self.to_k)?)?;
        let v = self.to_heads(&linear_nb(visual, &self.to_v)?)?;
        let q = rope.apply(&rms_weighted(&q, &self.norm_q, QK_NORM_EPS)?)?;
        let k = rope.apply(&rms_weighted(&k, &self.norm_k, QK_NORM_EPS)?)?;

        // Text q/k/v (+ per-head qk_norm), no RoPE.
        let eq = self.to_heads(&linear_nb(text, &self.add_q)?)?;
        let ek = self.to_heads(&linear_nb(text, &self.add_k)?)?;
        let ev = self.to_heads(&linear_nb(text, &self.add_v)?)?;
        let eq = rms_weighted(&eq, &self.norm_added_q, QK_NORM_EPS)?;
        let ek = rms_weighted(&ek, &self.norm_added_k, QK_NORM_EPS)?;

        // → [B, heads, S, head_dim]; concat visual + text along the sequence axis.
        let t = |a: &Array| -> Result<Array> { Ok(a.transpose_axes(&[0, 2, 1, 3])?) };
        let full_q = concatenate_axis(&[&t(&q)?, &t(&eq)?], 2)?;
        let full_k = concatenate_axis(&[&t(&k)?, &t(&ek)?], 2)?;
        let full_v = concatenate_axis(&[&t(&v)?, &t(&ev)?], 2)?;

        // Additive key-padding mask [B, 1, 1, Sv+St]: 0 for visual + valid text, −inf for padded text.
        let mask = build_joint_mask(enc_mask, sv)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = scaled_dot_product_attention(&full_q, &full_k, &full_v, scale, &mask, None)?;

        // → [B, Sv+St, inner]; split back to visual / text.
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[out.shape()[0], sv + st, (self.num_heads * self.head_dim) as i32])?;
        let vis_idx = Array::from_slice(&(0..sv).collect::<Vec<i32>>(), &[sv]);
        let txt_idx = Array::from_slice(&(sv..sv + st).collect::<Vec<i32>>(), &[st]);
        let vis = out.take_axis(&vis_idx, 1)?;
        let txt = out.take_axis(&txt_idx, 1)?;

        let hidden = linear_b(&vis, &self.to_out_w, &self.to_out_b)?;
        let enc = match &self.to_add_out {
            Some((w, b)) => Some(linear_b(&txt, w, b)?),
            None => None,
        };
        Ok((hidden, enc))
    }
}

/// Build the additive joint attention mask `[B, 1, 1, num_visual + St]`: `0` for the visual keys and
/// valid text keys, `−inf` for padded text keys (`enc_mask == 0`). Broadcasts over query + heads. This
/// is the joint-SDPA equivalent of the reference's gather-valid-keys path: padded keys get softmax
/// weight 0, so the valid query rows are identical (padded text *query* rows differ — masked out of
/// the parity gate for `block_out.1`).
fn build_joint_mask(enc_mask: &Array, num_visual: i32) -> Result<Array> {
    let sh = enc_mask.shape();
    if sh.len() != 2 {
        return Err(Error::Msg(format!(
            "mochi attention: enc_mask must be [B, St], got {sh:?}"
        )));
    }
    let (b, st) = (sh[0], sh[1]);
    let m: Vec<f32> = enc_mask.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
    let total = num_visual + st;
    let mut data = vec![0f32; (b * total) as usize];
    for bi in 0..b {
        for j in 0..st {
            // valid iff mask == 1; padded text key → −inf.
            if m[(bi * st + j) as usize] == 0.0 {
                data[(bi * total + num_visual + j) as usize] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(&data, &[b, 1, 1, total]))
}

// ---------------------------------------------------------------------------- block

/// `norm1_context` variant: a `MochiRMSNormZero` (non-final blocks) or a `MochiLayerNormContinuous`
/// (the final `context_pre_only` block).
#[derive(Clone)]
enum NormContext {
    /// `MochiRMSNormZero`: `linear [4·pooled, inner]` → 4 modulation chunks.
    Zero { lin_w: Array, lin_b: Array },
    /// `MochiLayerNormContinuous`: `linear_1 [pooled, inner]` → a single scale.
    Continuous { lin_w: Array, lin_b: Array },
}

/// One `MochiTransformerBlock` — the dual-stream MMDiT block.
#[derive(Clone)]
pub struct MochiTransformerBlock {
    norm1_w: Array, // [4·inner, inner]
    norm1_b: Array,
    norm1_context: NormContext,
    attn: MochiAttention,
    ff: SwiGlu,
    /// `None` on the final `context_pre_only` block — the text output path is dropped.
    ff_context: Option<SwiGlu>,
    eps: f32,
}

impl MochiTransformerBlock {
    /// Load block `prefix` (e.g. `transformer_blocks.0`). `context_pre_only` (the final block) drops
    /// the text output path.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MochiDitConfig,
        context_pre_only: bool,
        dtype: Dtype,
    ) -> Result<Self> {
        let norm1_context = if context_pre_only {
            NormContext::Continuous {
                lin_w: w
                    .require(&join(prefix, "norm1_context.linear_1.weight"))?
                    .as_dtype(dtype)?,
                lin_b: w
                    .require(&join(prefix, "norm1_context.linear_1.bias"))?
                    .as_dtype(dtype)?,
            }
        } else {
            NormContext::Zero {
                lin_w: w
                    .require(&join(prefix, "norm1_context.linear.weight"))?
                    .as_dtype(dtype)?,
                lin_b: w
                    .require(&join(prefix, "norm1_context.linear.bias"))?
                    .as_dtype(dtype)?,
            }
        };
        let ff_context = if context_pre_only {
            None
        } else {
            Some(SwiGlu::from_weights(
                w,
                &join(prefix, "ff_context"),
                dtype,
            )?)
        };
        Ok(Self {
            norm1_w: w
                .require(&join(prefix, "norm1.linear.weight"))?
                .as_dtype(dtype)?,
            norm1_b: w
                .require(&join(prefix, "norm1.linear.bias"))?
                .as_dtype(dtype)?,
            norm1_context,
            attn: MochiAttention::from_weights(
                w,
                &join(prefix, "attn1"),
                cfg,
                context_pre_only,
                dtype,
            )?,
            ff: SwiGlu::from_weights(w, &join(prefix, "ff"), dtype)?,
            ff_context,
            eps: cfg.eps,
        })
    }

    /// Forward the block. `hidden [B, Sv, inner]`, `enc [B, St, pooled]`, `temb [B, inner]`,
    /// `enc_mask [B, St]`. Returns the updated `(hidden, enc)`.
    pub fn forward(
        &self,
        hidden: &Array,
        enc: &Array,
        temb: &Array,
        rope: &MochiRope,
        enc_mask: &Array,
    ) -> Result<(Array, Array)> {
        let eps = self.eps;
        let silu_temb = silu(&temb.as_dtype(Dtype::Float32)?)?;

        // norm1 (visual): (scale_msa, gate_msa, scale_mlp, gate_mlp).
        let emb = linear_b(&silu_temb, &self.norm1_w, &self.norm1_b)?;
        let c = chunk_last(&emb, 4)?;
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = (&c[0], &c[1], &c[2], &c[3]);
        let norm_h = multiply(
            &rms_weightless(hidden, eps)?,
            &add(&unsqueeze1(scale_msa)?, Array::from_f32(1.0))?,
        )?;

        // norm1_context (text).
        let (norm_e, ctx_gates) = match &self.norm1_context {
            NormContext::Zero { lin_w, lin_b } => {
                let emb_c = linear_b(&silu_temb, lin_w, lin_b)?;
                let cc = chunk_last(&emb_c, 4)?;
                let norm_e = multiply(
                    &rms_weightless(enc, eps)?,
                    &add(&unsqueeze1(&cc[0])?, Array::from_f32(1.0))?,
                )?;
                (norm_e, Some((cc[1].clone(), cc[2].clone(), cc[3].clone())))
            }
            NormContext::Continuous { lin_w, lin_b } => {
                let scale_c = linear_b(&silu_temb, lin_w, lin_b)?;
                let norm_e = multiply(
                    &rms_weightless(enc, eps)?,
                    &add(&unsqueeze1(&scale_c)?, Array::from_f32(1.0))?,
                )?;
                (norm_e, None)
            }
        };

        // Joint attention.
        let (attn_h, attn_e) = self.attn.forward(&norm_h, &norm_e, rope, enc_mask)?;

        // Visual residuals: tanh-gated attn (norm2), SwiGLU FFN with (1+scale_mlp) mod (norm3),
        // tanh-gated ff (norm4).
        let hidden = add(
            hidden,
            &multiply(
                &rms_weightless(&attn_h, eps)?,
                &unsqueeze1(&tanh(gate_msa)?)?,
            )?,
        )?;
        let norm_h2 = multiply(
            &rms_weightless(&hidden, eps)?,
            &add(&unsqueeze1(scale_mlp)?, Array::from_f32(1.0))?,
        )?;
        let ff_out = self.ff.forward(&norm_h2)?;
        let hidden = add(
            &hidden,
            &multiply(&rms_weightless(&ff_out, eps)?, &unsqueeze1(&tanh(gate_mlp)?)?)?,
        )?;

        // Text residuals (skipped on the final context_pre_only block).
        let enc = if let (Some((e_gate_msa, e_scale_mlp, e_gate_mlp)), Some(attn_e), Some(ff_ctx)) =
            (ctx_gates, attn_e, &self.ff_context)
        {
            let enc = add(
                enc,
                &multiply(
                    &rms_weightless(&attn_e, eps)?,
                    &unsqueeze1(&tanh(&e_gate_msa)?)?,
                )?,
            )?;
            let norm_e2 = multiply(
                &rms_weightless(&enc, eps)?,
                &add(&unsqueeze1(&e_scale_mlp)?, Array::from_f32(1.0))?,
            )?;
            let ff_e = ff_ctx.forward(&norm_e2)?;
            add(
                &enc,
                &multiply(&rms_weightless(&ff_e, eps)?, &unsqueeze1(&tanh(&e_gate_mlp)?)?)?,
            )?
        } else {
            enc.clone()
        };

        Ok((hidden, enc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::subtract;

    /// Deterministic small "random" fill, bounded so the block stays well-conditioned.
    fn rnd(shape: &[i32], seed: u64) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                (((i as u64).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 1e-6).sin()
                    * 0.05
            })
            .collect();
        Array::from_slice(&data, shape)
    }

    /// A tiny DiT config: 2 heads × 8 head-dim → inner 16, pooled 8. GroupNorm-free, so any sizes work.
    fn tiny_cfg() -> MochiDitConfig {
        MochiDitConfig {
            patch_size: 2,
            num_heads: 2,
            head_dim: 8,
            num_layers: 1,
            pooled_dim: 8,
            in_channels: 12,
            text_embed_dim: 4096,
            time_embed_dim: 256,
            eps: 1e-6,
        }
    }

    /// Build a synthetic weight set for one non-final block of `tiny_cfg`.
    fn tiny_block_weights(cfg: &MochiDitConfig, prefix: &str) -> Weights {
        let inner = cfg.inner_dim() as i32; // 16
        let pooled = cfg.pooled_dim as i32; // 8
        let hd = cfg.head_dim as i32; // 8
        let ff_inner = ((4 * cfg.inner_dim() * 2) / 3) as i32;
        let ff_ctx_inner = ((4 * cfg.pooled_dim * 2) / 3) as i32;
        let mut w = Weights::empty();
        let mut put = |k: String, a: Array| w.insert(k, a);
        let p = |s: &str| format!("{prefix}.{s}");
        put(p("norm1.linear.weight"), rnd(&[4 * inner, inner], 1));
        put(p("norm1.linear.bias"), rnd(&[4 * inner], 2));
        put(p("norm1_context.linear.weight"), rnd(&[4 * pooled, inner], 3));
        put(p("norm1_context.linear.bias"), rnd(&[4 * pooled], 4));
        put(p("attn1.to_q.weight"), rnd(&[inner, inner], 5));
        put(p("attn1.to_k.weight"), rnd(&[inner, inner], 6));
        put(p("attn1.to_v.weight"), rnd(&[inner, inner], 7));
        put(p("attn1.add_q_proj.weight"), rnd(&[inner, pooled], 8));
        put(p("attn1.add_k_proj.weight"), rnd(&[inner, pooled], 9));
        put(p("attn1.add_v_proj.weight"), rnd(&[inner, pooled], 10));
        put(p("attn1.norm_q.weight"), rnd(&[hd], 11));
        put(p("attn1.norm_k.weight"), rnd(&[hd], 12));
        put(p("attn1.norm_added_q.weight"), rnd(&[hd], 13));
        put(p("attn1.norm_added_k.weight"), rnd(&[hd], 14));
        put(p("attn1.to_out.0.weight"), rnd(&[inner, inner], 15));
        put(p("attn1.to_out.0.bias"), rnd(&[inner], 16));
        put(p("attn1.to_add_out.weight"), rnd(&[pooled, inner], 17));
        put(p("attn1.to_add_out.bias"), rnd(&[pooled], 18));
        put(p("ff.net.0.proj.weight"), rnd(&[2 * ff_inner, inner], 19));
        put(p("ff.net.2.weight"), rnd(&[inner, ff_inner], 20));
        put(p("ff_context.net.0.proj.weight"), rnd(&[2 * ff_ctx_inner, pooled], 21));
        put(p("ff_context.net.2.weight"), rnd(&[pooled, ff_ctx_inner], 22));
        w
    }

    #[test]
    fn block_forward_shapes_and_determinism() {
        let cfg = tiny_cfg();
        let w = tiny_block_weights(&cfg, "transformer_blocks.0");
        let block =
            MochiTransformerBlock::from_weights(&w, "transformer_blocks.0", &cfg, false, Dtype::Float32)
                .unwrap();

        // 1 frame × 2 × 2 = 4 visual tokens, 3 text tokens (2 valid, 1 pad), inner 16, pooled 8.
        let hidden = rnd(&[1, 4, 16], 100);
        let enc = rnd(&[1, 3, 8], 101);
        let temb = rnd(&[1, 16], 102);
        let enc_mask = Array::from_slice(&[1.0f32, 1.0, 0.0], &[1, 3]);
        let pf = rnd(&[3, 2, 4], 103); // [3, heads, head_dim/2]
        let rope = MochiRope::new(&pf, 1, 2, 2).unwrap();

        let (h1, e1) = block.forward(&hidden, &enc, &temb, &rope, &enc_mask).unwrap();
        assert_eq!(h1.shape(), &[1, 4, 16]);
        assert_eq!(e1.shape(), &[1, 3, 8]);

        // Determinism.
        let (h2, e2) = block.forward(&hidden, &enc, &temb, &rope, &enc_mask).unwrap();
        let close = |a: &Array, b: &Array| {
            mlx_rs::ops::max(mlx_rs::ops::abs(&subtract(a, b).unwrap()).unwrap(), None)
                .unwrap()
                .item::<f32>()
                < 1e-6
        };
        assert!(close(&h1, &h2));
        assert!(close(&e1, &e2));
        assert!(h1.as_slice::<f32>().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn context_pre_only_block_returns_enc_unchanged() {
        // The final block drops the text output path: enc must be returned identical to the input.
        let cfg = tiny_cfg();
        let mut w = tiny_block_weights(&cfg, "transformer_blocks.0");
        // Swap in the continuous norm1_context + drop the text-only weights for a final block.
        let inner = cfg.inner_dim() as i32;
        let pooled = cfg.pooled_dim as i32;
        w.remove("transformer_blocks.0.norm1_context.linear.weight");
        w.remove("transformer_blocks.0.norm1_context.linear.bias");
        w.insert(
            "transformer_blocks.0.norm1_context.linear_1.weight".to_string(),
            rnd(&[pooled, inner], 30),
        );
        w.insert(
            "transformer_blocks.0.norm1_context.linear_1.bias".to_string(),
            rnd(&[pooled], 31),
        );
        let block =
            MochiTransformerBlock::from_weights(&w, "transformer_blocks.0", &cfg, true, Dtype::Float32)
                .unwrap();
        let hidden = rnd(&[1, 4, 16], 100);
        let enc = rnd(&[1, 3, 8], 101);
        let temb = rnd(&[1, 16], 102);
        let enc_mask = Array::from_slice(&[1.0f32, 1.0, 0.0], &[1, 3]);
        let pf = rnd(&[3, 2, 4], 103);
        let rope = MochiRope::new(&pf, 1, 2, 2).unwrap();
        let (h, e) = block.forward(&hidden, &enc, &temb, &rope, &enc_mask).unwrap();
        assert_eq!(h.shape(), &[1, 4, 16]);
        // enc is bit-identical to the input (no context update on the final block).
        let same = mlx_rs::ops::max(mlx_rs::ops::abs(&subtract(&e, &enc).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert_eq!(same, 0.0);
    }
}
