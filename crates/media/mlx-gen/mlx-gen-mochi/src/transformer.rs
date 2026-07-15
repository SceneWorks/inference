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

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{
    add, concatenate_axis, matmul, maximum, mean_axis, multiply, quantized_matmul, rsqrt, split,
    sum_axis, tanh,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv2d, silu, timestep_sincos};
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
        .ok_or_else(|| {
            Error::Msg(format!(
                "mochi dit index {}: no weight_map",
                index.display()
            ))
        })?;

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

/// A pre-quantized tier's affine-quant geometry, read from the tier dir's `split_model.json` (the
/// [`crate::config::MochiSplitModel`] manifest). When present on [`MochiDitConfig`], the packed-load
/// path ([`MochiLinear::load`]) consumes the on-disk `.weight`(u32)/`.scales`/`.biases` packs for the
/// [`MOCHI_QUANT_SUFFIXES`](crate::convert::MOCHI_QUANT_SUFFIXES) Linears instead of loading dense
/// weights. `bits` (4 → Q4, 8 → Q8) and `group` (the group-scale width, 64) must match the packing
/// `convert.rs` emitted — they are read from the manifest, never hardcoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MochiQuant {
    /// Quantization bits (4 → Q4, 8 → Q8).
    pub bits: i32,
    /// Affine-quant group size (the reference/mflux default 64).
    pub group: i32,
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
    /// `Some` iff the transformer weights are a pre-quantized tier (the tier dir's `split_model.json`
    /// carries `quantized:true`). Drives the packed-load path; `None` = dense (the default / the raw
    /// bf16 snapshot). Rides on the config so the parity-tested `from_weights(w, cfg, dtype)` seam is
    /// unchanged — mirrors `WanModelConfig.quantization`.
    pub quantization: Option<MochiQuant>,
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
            quantization: None,
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
    Ok(matmul(x, w.t())?)
}

/// `y = x · Wᵀ + b` (mlx-gen core fused `addmm`).
fn linear_b(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    mlx_gen::nn::linear(x, w, b)
}

/// A DiT projection Linear that is **dense** (raw `[out, in]` weight) or **quantized** (group-wise
/// affine `.weight`(u32)/`.scales`/`.biases` packs). The visual/text attention `to_q/k/v` +
/// `to_out.0`/`to_add_out` and the SwiGLU `net.0.proj`/`net.2` are quantized in a pre-quantized tier
/// (they carry `.scales` on disk); every other tensor (norms, adaLN modulation, patchify, pooler,
/// caption/time embed, proj_out) stays dense. `.scales` presence is the per-Linear signal, exactly as
/// the Wan packed-load path and mirroring the reference `nn.quantize` predicate.
#[derive(Clone)]
enum MochiLinear {
    /// Dense base — `matmul(x, Wᵀ)` (`+ b` fused via `addmm` when biased). Bit-identical to the
    /// previous raw-`Array` path (f32 activations make `addmm == matmul + add`).
    Dense { w: Array, b: Option<Array> },
    /// Quantized base — `quantized_matmul` (fp32-accumulate) over the packed weight, then optional
    /// dense-bias add. Activations feed in AS-IS (Mochi's f32 compute), matching the Wan path.
    Quant {
        wq: Array,
        scales: Array,
        biases: Array,
        b: Option<Array>,
        group: i32,
        bits: i32,
    },
}

impl MochiLinear {
    /// Load `{prefix}.weight` (+ `{prefix}.bias` when `bias`). When `quant` is `Some` **and** this
    /// Linear carries a `{prefix}.scales` pack on disk, consume the packed parts directly (the
    /// pre-quantized tier / `convert.rs` output). Otherwise load dense at `dtype`. The `.scales` probe
    /// is what makes a single tier dir mix packed attention/FFN Linears with dense norms/embeds.
    fn load(
        w: &Weights,
        prefix: &str,
        bias: bool,
        quant: Option<MochiQuant>,
        dtype: Dtype,
    ) -> Result<Self> {
        let wkey = format!("{prefix}.weight");
        let bkey = format!("{prefix}.bias");
        if let (Some(q), Some(scales)) = (quant, w.get(&format!("{prefix}.scales"))) {
            // Packed parts are consumed as-is (u32 codes + bf16 scales/biases) — no re-quantize.
            let b = if bias {
                Some(w.require(&bkey)?.clone())
            } else {
                w.get(&bkey).cloned()
            };
            return Ok(MochiLinear::Quant {
                wq: w.require(&wkey)?.clone(),
                scales: scales.clone(),
                biases: w.require(&format!("{prefix}.biases"))?.clone(),
                b,
                group: q.group,
                bits: q.bits,
            });
        }
        let b = if bias {
            Some(w.require(&bkey)?.as_dtype(dtype)?)
        } else {
            None
        };
        Ok(MochiLinear::Dense {
            w: w.require(&wkey)?.as_dtype(dtype)?,
            b,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            MochiLinear::Dense { w, b } => match b {
                Some(b) => linear_b(x, w, b),
                None => linear_nb(x, w),
            },
            MochiLinear::Quant {
                wq,
                scales,
                biases,
                b,
                group,
                bits,
            } => {
                let mut y = quantized_matmul(x, wq, scales, biases, true, *group, *bits)?;
                if let Some(b) = b {
                    y = add(&y, b)?;
                }
                Ok(y)
            }
        }
    }
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
/// split into `(value, gate)`, `value · silu(gate)`, then `out` (`inner → d`). Both projections are
/// quantized in a pre-quantized tier (`net.0.proj` / `net.2` — see [`MochiLinear`]).
#[derive(Clone)]
struct SwiGlu {
    proj: MochiLinear, // [2·inner, d]
    out: MochiLinear,  // [d, inner]
}

impl SwiGlu {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        quant: Option<MochiQuant>,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            proj: MochiLinear::load(w, &join(prefix, "net.0.proj"), false, quant, dtype)?,
            out: MochiLinear::load(w, &join(prefix, "net.2"), false, quant, dtype)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.proj.forward(x)?;
        let parts = chunk_last(&h, 2)?;
        let gated = multiply(&parts[0], &silu(&parts[1])?)?;
        self.out.forward(&gated)
    }
}

// ---------------------------------------------------------------------------- attention

/// Mochi joint attention (`MochiAttention` + `MochiAttnProcessor2_0`). The `to_q/k/v`, added
/// `add_{q,k,v}_proj`, `to_out.0`, and `to_add_out` projections are quantized in a pre-quantized tier
/// (they carry `.scales` on disk); the per-head `qk_norm` weights stay dense.
#[derive(Clone)]
pub struct MochiAttention {
    to_q: MochiLinear,
    to_k: MochiLinear,
    to_v: MochiLinear,
    add_q: MochiLinear,
    add_k: MochiLinear,
    add_v: MochiLinear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    /// `to_out.0` (biased).
    to_out: MochiLinear,
    /// `to_add_out` (biased) — absent when `context_pre_only`.
    to_add_out: Option<MochiLinear>,
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
        let q = cfg.quantization;
        let lin = |name: &str, bias: bool| -> Result<MochiLinear> {
            MochiLinear::load(w, &join(prefix, name), bias, q, dtype)
        };
        let arr =
            |name: &str| -> Result<Array> { Ok(w.require(&join(prefix, name))?.as_dtype(dtype)?) };
        let to_add_out = if context_pre_only {
            None
        } else {
            Some(lin("to_add_out", true)?)
        };
        Ok(Self {
            to_q: lin("to_q", false)?,
            to_k: lin("to_k", false)?,
            to_v: lin("to_v", false)?,
            add_q: lin("add_q_proj", false)?,
            add_k: lin("add_k_proj", false)?,
            add_v: lin("add_v_proj", false)?,
            norm_q: arr("norm_q.weight")?,
            norm_k: arr("norm_k.weight")?,
            norm_added_q: arr("norm_added_q.weight")?,
            norm_added_k: arr("norm_added_k.weight")?,
            to_out: lin("to_out.0", true)?,
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
        let q = self.to_heads(&self.to_q.forward(visual)?)?;
        let k = self.to_heads(&self.to_k.forward(visual)?)?;
        let v = self.to_heads(&self.to_v.forward(visual)?)?;
        let q = rope.apply(&rms_weighted(&q, &self.norm_q, QK_NORM_EPS)?)?;
        let k = rope.apply(&rms_weighted(&k, &self.norm_k, QK_NORM_EPS)?)?;

        // Text q/k/v (+ per-head qk_norm), no RoPE.
        let eq = self.to_heads(&self.add_q.forward(text)?)?;
        let ek = self.to_heads(&self.add_k.forward(text)?)?;
        let ev = self.to_heads(&self.add_v.forward(text)?)?;
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
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            out.shape()[0],
            sv + st,
            (self.num_heads * self.head_dim) as i32,
        ])?;
        let vis_idx = Array::from_slice(&(0..sv).collect::<Vec<i32>>(), &[sv]);
        let txt_idx = Array::from_slice(&(sv..sv + st).collect::<Vec<i32>>(), &[st]);
        let vis = out.take_axis(&vis_idx, 1)?;
        let txt = out.take_axis(&txt_idx, 1)?;

        let hidden = self.to_out.forward(&vis)?;
        let enc = match &self.to_add_out {
            Some(l) => Some(l.forward(&txt)?),
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
    let m: Vec<f32> = enc_mask
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec();
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
                cfg.quantization,
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
            ff: SwiGlu::from_weights(w, &join(prefix, "ff"), cfg.quantization, dtype)?,
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
            &multiply(
                &rms_weightless(&ff_out, eps)?,
                &unsqueeze1(&tanh(gate_mlp)?)?,
            )?,
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
                &multiply(
                    &rms_weightless(&ff_e, eps)?,
                    &unsqueeze1(&tanh(&e_gate_mlp)?)?,
                )?,
            )?
        } else {
            enc.clone()
        };

        Ok((hidden, enc))
    }
}

// ---------------------------------------------------------------------- time embedding

/// `MochiAttentionPool` — a single learned query (the masked-mean "class" token) attends over the raw
/// T5 tokens to pool them into one `[B, output_dim]` conditioning vector.
#[derive(Clone)]
struct AttentionPool {
    to_kv_w: Array,
    to_kv_b: Array,
    to_q_w: Array,
    to_q_b: Array,
    to_out_w: Array,
    to_out_b: Array,
    num_heads: usize,
    embed_dim: usize,
}

impl AttentionPool {
    fn from_weights(w: &Weights, prefix: &str, num_heads: usize, dtype: Dtype) -> Result<Self> {
        let g =
            |name: &str| -> Result<Array> { Ok(w.require(&join(prefix, name))?.as_dtype(dtype)?) };
        let to_kv_w = g("to_kv.weight")?;
        let embed_dim = to_kv_w.shape()[1] as usize; // to_kv: [2·embed, embed]
        Ok(Self {
            to_kv_w,
            to_kv_b: g("to_kv.bias")?,
            to_q_w: g("to_q.weight")?,
            to_q_b: g("to_q.bias")?,
            to_out_w: g("to_out.weight")?,
            to_out_b: g("to_out.bias")?,
            num_heads,
            embed_dim,
        })
    }

    /// `x [B, L, D]`, `mask [B, L]` (0/1) → pooled `[B, output_dim]`.
    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, l, d) = (sh[0], sh[1], sh[2]);
        let head_dim = self.embed_dim / self.num_heads;

        // pool_tokens: weighted mean over valid tokens → the query "class" token.
        let m = mask.as_dtype(Dtype::Float32)?.reshape(&[b, l, 1])?;
        let denom = maximum(&sum_axis(&m, 1, true)?, Array::from_f32(1.0))?; // [B,1,1] clamp≥1
        let mnorm = mlx_rs::ops::divide(&m, &denom)?;
        let x_pool = sum_axis(&multiply(x, &mnorm)?, 1, true)?; // [B, 1, D]

        // Concat pooled + tokens; KV over all, Q from the pooled token only.
        let xcat = concatenate_axis(&[&x_pool, x], 1)?; // [B, 1+L, D]
        let kv = linear_b(&xcat, &self.to_kv_w, &self.to_kv_b)?; // [B, 1+L, 2D]
        let q = linear_b(&x_pool.reshape(&[b, d])?, &self.to_q_w, &self.to_q_b)?; // [B, D]

        // Heads: kv [B, 1+L, 2, H, hd] → [B, H, 2, 1+L, hd] → k, v.
        let lk = l + 1;
        let kv = kv
            .reshape(&[b, lk, 2, self.num_heads as i32, head_dim as i32])?
            .transpose_axes(&[0, 3, 2, 1, 4])?; // [B, H, 2, 1+L, hd]
        let parts = split(&kv, 2, 2)?;
        let k = parts[0].reshape(&[b, self.num_heads as i32, lk, head_dim as i32])?;
        let v = parts[1].reshape(&[b, self.num_heads as i32, lk, head_dim as i32])?;
        let q = q.reshape(&[b, self.num_heads as i32, 1, head_dim as i32])?; // [B, H, 1, hd]

        // Additive mask [B, 1, 1, 1+L]: key 0 (pooled) always valid; text keys 0/−inf per `mask`.
        let mvals: Vec<f32> = mask.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let mut mdata = vec![0f32; (b * lk) as usize];
        for bi in 0..b {
            for j in 0..l {
                if mvals[(bi * l + j) as usize] == 0.0 {
                    mdata[(bi * lk + 1 + j) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let attn_mask = Array::from_slice(&mdata, &[b, 1, 1, lk]);

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, &attn_mask, None)?; // [B,H,1,hd]
        let out = out.reshape(&[b, self.embed_dim as i32])?; // squeeze(2).flatten(1,2)
        linear_b(&out, &self.to_out_w, &self.to_out_b)
    }
}

/// `MochiCombinedTimestepCaptionEmbedding` — sinusoidal-timestep MLP + masked attention-pool of the
/// raw T5 tokens (summed into `temb`), plus the `caption_proj` that projects the raw T5 tokens into
/// the 1536-dim text stream. Returns `(temb [B, inner], caption [B, L, pooled])`.
#[derive(Clone)]
struct TimeEmbed {
    ts_lin1_w: Array,
    ts_lin1_b: Array,
    ts_lin2_w: Array,
    ts_lin2_b: Array,
    pooler: AttentionPool,
    caption_w: Array,
    caption_b: Array,
    time_embed_dim: usize,
}

impl TimeEmbed {
    fn from_weights(w: &Weights, prefix: &str, cfg: &MochiDitConfig, dtype: Dtype) -> Result<Self> {
        let g =
            |name: &str| -> Result<Array> { Ok(w.require(&join(prefix, name))?.as_dtype(dtype)?) };
        Ok(Self {
            ts_lin1_w: g("timestep_embedder.linear_1.weight")?,
            ts_lin1_b: g("timestep_embedder.linear_1.bias")?,
            ts_lin2_w: g("timestep_embedder.linear_2.weight")?,
            ts_lin2_b: g("timestep_embedder.linear_2.bias")?,
            // The reference builds the pooler with num_attention_heads=8.
            pooler: AttentionPool::from_weights(w, &join(prefix, "pooler"), 8, dtype)?,
            caption_w: g("caption_proj.weight")?,
            caption_b: g("caption_proj.bias")?,
            time_embed_dim: cfg.time_embed_dim,
        })
    }

    fn forward(&self, timestep: &Array, enc: &Array, enc_mask: &Array) -> Result<(Array, Array)> {
        // Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0.0) → [B, time_embed_dim].
        let time_proj = timestep_sincos(
            &timestep.as_dtype(Dtype::Float32)?,
            self.time_embed_dim,
            10000.0,
            0.0,
        )?;
        let te = linear_b(&time_proj, &self.ts_lin1_w, &self.ts_lin1_b)?;
        let te = silu(&te)?;
        let te = linear_b(&te, &self.ts_lin2_w, &self.ts_lin2_b)?; // [B, inner]

        let pooled = self.pooler.forward(enc, enc_mask)?; // [B, inner]
        let caption = linear_b(enc, &self.caption_w, &self.caption_b)?; // [B, L, pooled]
        let temb = add(&te, &pooled)?;
        Ok((temb, caption))
    }
}

// ---------------------------------------------------------------------- full model

/// The full Mochi 1 AsymmDiT — `MochiTransformer3DModel`. One `forward` = one CFG-branch velocity
/// prediction (call with the `[neg, pos]` batch for a full CFG step; combine downstream via
/// [`crate::scheduler::cfg_combine`]).
pub struct MochiTransformer3DModel {
    patch_w: Array, // mlx [out, kh, kw, in]
    patch_b: Array,
    pos_frequencies: Array, // [3, heads, head_dim/2]
    time_embed: TimeEmbed,
    blocks: Vec<MochiTransformerBlock>,
    norm_out_w: Array, // [2·inner, inner]
    norm_out_b: Array,
    proj_out_w: Array, // [patch²·out_ch, inner]
    proj_out_b: Array,
    cfg: MochiDitConfig,
}

impl MochiTransformer3DModel {
    /// Build the full model from the transformer weights (see [`load_transformer_weights`]).
    pub fn from_weights(w: &Weights, cfg: &MochiDitConfig, dtype: Dtype) -> Result<Self> {
        // PatchEmbed Conv2d weight [out, in, kh, kw] → mlx NHWC [out, kh, kw, in].
        let patch_w = w
            .require("patch_embed.proj.weight")?
            .transpose_axes(&[0, 2, 3, 1])?
            .as_dtype(dtype)?;
        let blocks = (0..cfg.num_layers)
            .map(|i| {
                MochiTransformerBlock::from_weights(
                    w,
                    &format!("transformer_blocks.{i}"),
                    cfg,
                    i == cfg.num_layers - 1, // final block is context_pre_only
                    dtype,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_w,
            patch_b: w.require("patch_embed.proj.bias")?.as_dtype(dtype)?,
            pos_frequencies: w.require("pos_frequencies")?.as_dtype(dtype)?,
            time_embed: TimeEmbed::from_weights(w, "time_embed", cfg, dtype)?,
            blocks,
            norm_out_w: w.require("norm_out.linear.weight")?.as_dtype(dtype)?,
            norm_out_b: w.require("norm_out.linear.bias")?.as_dtype(dtype)?,
            proj_out_w: w.require("proj_out.weight")?.as_dtype(dtype)?,
            proj_out_b: w.require("proj_out.bias")?.as_dtype(dtype)?,
            cfg: *cfg,
        })
    }

    /// Forward. `hidden [B, in_ch, F, H, W]` (latent), `enc [B, L, text_embed]` (raw T5), `timestep
    /// [B]`, `enc_mask [B, L]` (0/1). Returns the velocity `noise_pred [B, in_ch, F, H, W]`.
    pub fn forward(
        &self,
        hidden: &Array,
        enc: &Array,
        timestep: &Array,
        enc_mask: &Array,
    ) -> Result<Array> {
        let sh = hidden.shape();
        let (b, c, f, h, wd) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let p = self.cfg.patch_size as i32;
        let ph = h / p;
        let pw = wd / p;
        let inner = self.cfg.inner_dim() as i32;

        // Time / caption embedding (raw T5 → temb + 1536-dim text stream).
        let (temb, mut enc_stream) = self.time_embed.forward(timestep, enc, enc_mask)?;

        // Patchify: [B, C, F, H, W] → [B·F, H, W, C] (NHWC) → Conv2d(patch) → [B, F·ph·pw, inner].
        let x = hidden
            .as_dtype(Dtype::Float32)?
            .transpose_axes(&[0, 2, 1, 3, 4])? // [B, F, C, H, W]
            .reshape(&[b * f, c, h, wd])?
            .transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = conv2d(&x, &self.patch_w, Some(&self.patch_b), p, 0)?; // [B·F, ph, pw, inner]
        let mut hs = x
            .reshape(&[b * f, ph * pw, inner])?
            .reshape(&[b, f * ph * pw, inner])?;

        // Learned 3-D RoPE over the post-patch grid.
        let rope = MochiRope::new(&self.pos_frequencies, f as usize, ph as usize, pw as usize)?;

        for block in &self.blocks {
            let (h_new, e_new) = block.forward(&hs, &enc_stream, &temb, &rope, enc_mask)?;
            hs = h_new;
            enc_stream = e_new;
        }

        // AdaLayerNormContinuous (layer_norm, no affine) → proj_out.
        let emb = linear_b(&silu(&temb)?, &self.norm_out_w, &self.norm_out_b)?;
        let so = chunk_last(&emb, 2)?;
        let (scale, shift) = (&so[0], &so[1]);
        let normed = layer_norm(&hs, None, None, 1e-6)?;
        let hs = add(
            &multiply(&normed, &add(&unsqueeze1(scale)?, Array::from_f32(1.0))?)?,
            &unsqueeze1(shift)?,
        )?;
        let hs = linear_b(&hs, &self.proj_out_w, &self.proj_out_b)?; // [B, seq, p²·out_ch]

        // Unpatchify: [B, F, ph, pw, p, p, out_ch] → [B, out_ch, F, H, W].
        let out_ch = c; // out_channels == in_channels (12)
        let hs = hs
            .reshape(&[b, f, ph, pw, p, p, out_ch])?
            .transpose_axes(&[0, 6, 1, 2, 4, 3, 5])? // [B, out_ch, F, ph, p, pw, p]
            .reshape(&[b, out_ch, f, ph * p, pw * p])?;
        Ok(hs)
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
            quantization: None,
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
        put(
            p("norm1_context.linear.weight"),
            rnd(&[4 * pooled, inner], 3),
        );
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
        put(
            p("ff_context.net.0.proj.weight"),
            rnd(&[2 * ff_ctx_inner, pooled], 21),
        );
        put(
            p("ff_context.net.2.weight"),
            rnd(&[pooled, ff_ctx_inner], 22),
        );
        w
    }

    #[test]
    fn block_forward_shapes_and_determinism() {
        let cfg = tiny_cfg();
        let w = tiny_block_weights(&cfg, "transformer_blocks.0");
        let block = MochiTransformerBlock::from_weights(
            &w,
            "transformer_blocks.0",
            &cfg,
            false,
            Dtype::Float32,
        )
        .unwrap();

        // 1 frame × 2 × 2 = 4 visual tokens, 3 text tokens (2 valid, 1 pad), inner 16, pooled 8.
        let hidden = rnd(&[1, 4, 16], 100);
        let enc = rnd(&[1, 3, 8], 101);
        let temb = rnd(&[1, 16], 102);
        let enc_mask = Array::from_slice(&[1.0f32, 1.0, 0.0], &[1, 3]);
        let pf = rnd(&[3, 2, 4], 103); // [3, heads, head_dim/2]
        let rope = MochiRope::new(&pf, 1, 2, 2).unwrap();

        let (h1, e1) = block
            .forward(&hidden, &enc, &temb, &rope, &enc_mask)
            .unwrap();
        assert_eq!(h1.shape(), &[1, 4, 16]);
        assert_eq!(e1.shape(), &[1, 3, 8]);

        // Determinism.
        let (h2, e2) = block
            .forward(&hidden, &enc, &temb, &rope, &enc_mask)
            .unwrap();
        let close = |a: &Array, b: &Array| {
            mlx_rs::ops::max(mlx_rs::ops::abs(subtract(a, b).unwrap()).unwrap(), None)
                .unwrap()
                .item::<f32>()
                < 1e-6
        };
        assert!(close(&h1, &h2));
        assert!(close(&e1, &e2));
        assert!(h1.as_slice::<f32>().iter().all(|x| x.is_finite()));
    }

    /// A tiny full-model config: 2 heads × 8 head-dim → inner 16, pooled 8, 4 latent channels, 2
    /// layers (so block 0 is normal and block 1 is `context_pre_only`).
    fn tiny_full_cfg() -> MochiDitConfig {
        MochiDitConfig {
            patch_size: 2,
            num_heads: 2,
            head_dim: 8,
            num_layers: 2,
            pooled_dim: 8,
            in_channels: 4,
            text_embed_dim: 16,
            time_embed_dim: 8,
            eps: 1e-6,
            quantization: None,
        }
    }

    /// Merge `src`'s tensors into `dst`.
    fn merge(dst: &mut Weights, src: Weights) {
        let keys: Vec<String> = src.keys().map(String::from).collect();
        for k in keys {
            if let Some(t) = src.get(&k) {
                dst.insert(k, t.clone());
            }
        }
    }

    fn tiny_full_weights(cfg: &MochiDitConfig) -> Weights {
        let inner = cfg.inner_dim() as i32; // 16
        let pooled = cfg.pooled_dim as i32; // 8
        let te = cfg.text_embed_dim as i32; // 16
        let ted = cfg.time_embed_dim as i32; // 8
        let in_ch = cfg.in_channels as i32; // 4
        let half = (cfg.head_dim / 2) as i32; // 4
        let out_dims = (cfg.patch_size * cfg.patch_size * cfg.in_channels) as i32; // 16

        let mut w = Weights::empty();
        w.insert("patch_embed.proj.weight", rnd(&[inner, in_ch, 2, 2], 200));
        w.insert("patch_embed.proj.bias", rnd(&[inner], 201));
        w.insert(
            "pos_frequencies",
            rnd(&[3, cfg.num_heads as i32, half], 202),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_1.weight",
            rnd(&[inner, ted], 203),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_1.bias",
            rnd(&[inner], 204),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_2.weight",
            rnd(&[inner, inner], 205),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_2.bias",
            rnd(&[inner], 206),
        );
        w.insert("time_embed.pooler.to_kv.weight", rnd(&[2 * te, te], 207));
        w.insert("time_embed.pooler.to_kv.bias", rnd(&[2 * te], 208));
        w.insert("time_embed.pooler.to_q.weight", rnd(&[te, te], 209));
        w.insert("time_embed.pooler.to_q.bias", rnd(&[te], 210));
        w.insert("time_embed.pooler.to_out.weight", rnd(&[inner, te], 211));
        w.insert("time_embed.pooler.to_out.bias", rnd(&[inner], 212));
        w.insert("time_embed.caption_proj.weight", rnd(&[pooled, te], 213));
        w.insert("time_embed.caption_proj.bias", rnd(&[pooled], 214));
        w.insert("norm_out.linear.weight", rnd(&[2 * inner, inner], 215));
        w.insert("norm_out.linear.bias", rnd(&[2 * inner], 216));
        w.insert("proj_out.weight", rnd(&[out_dims, inner], 217));
        w.insert("proj_out.bias", rnd(&[out_dims], 218));

        merge(&mut w, tiny_block_weights(cfg, "transformer_blocks.0"));
        merge(&mut w, tiny_block_weights(cfg, "transformer_blocks.1"));
        // Block 1 is the final context_pre_only block → needs norm1_context.linear_1.
        w.insert(
            "transformer_blocks.1.norm1_context.linear_1.weight",
            rnd(&[pooled, inner], 219),
        );
        w.insert(
            "transformer_blocks.1.norm1_context.linear_1.bias",
            rnd(&[pooled], 220),
        );
        w
    }

    #[test]
    fn full_model_forward_shapes_and_determinism() {
        let cfg = tiny_full_cfg();
        let w = tiny_full_weights(&cfg);
        let model = MochiTransformer3DModel::from_weights(&w, &cfg, Dtype::Float32).unwrap();

        // [B=2, C=4, F=1, H=4, W=4] latent, 3 text tokens (2 valid), timestep per batch element.
        let hidden = rnd(&[2, 4, 1, 4, 4], 300);
        let enc = rnd(&[2, 3, 16], 301);
        let timestep = Array::from_slice(&[0.0f32, 25.0], &[2]);
        let enc_mask = Array::from_slice(&[1.0f32, 1.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]);

        let out = model.forward(&hidden, &enc, &timestep, &enc_mask).unwrap();
        assert_eq!(
            out.shape(),
            &[2, 4, 1, 4, 4],
            "noise_pred matches latent shape"
        );
        assert!(out.as_slice::<f32>().iter().all(|x| x.is_finite()));

        let out2 = model.forward(&hidden, &enc, &timestep, &enc_mask).unwrap();
        let d = mlx_rs::ops::max(
            mlx_rs::ops::abs(subtract(&out, &out2).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        assert_eq!(d, 0.0, "forward is deterministic");
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
        let block = MochiTransformerBlock::from_weights(
            &w,
            "transformer_blocks.0",
            &cfg,
            true,
            Dtype::Float32,
        )
        .unwrap();
        let hidden = rnd(&[1, 4, 16], 100);
        let enc = rnd(&[1, 3, 8], 101);
        let temb = rnd(&[1, 16], 102);
        let enc_mask = Array::from_slice(&[1.0f32, 1.0, 0.0], &[1, 3]);
        let pf = rnd(&[3, 2, 4], 103);
        let rope = MochiRope::new(&pf, 1, 2, 2).unwrap();
        let (h, e) = block
            .forward(&hidden, &enc, &temb, &rope, &enc_mask)
            .unwrap();
        assert_eq!(h.shape(), &[1, 4, 16]);
        // enc is bit-identical to the input (no context update on the final block).
        let same = mlx_rs::ops::max(mlx_rs::ops::abs(subtract(&e, &enc).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert_eq!(same, 0.0);
    }
}
