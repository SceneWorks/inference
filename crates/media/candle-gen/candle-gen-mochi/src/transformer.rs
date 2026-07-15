//! Mochi 1 **AsymmDiT** denoiser — port of `MochiTransformer3DModel` + `MochiTransformerBlock` +
//! `MochiAttnProcessor2_0` (diffusers `transformer_mochi.py` / `attention_processor.py`).
//!
//! A dual-stream MMDiT: a **visual** stream (patch-embedded latent tokens, `inner_dim = 3072`) and a
//! **text** stream (caption-projected T5 tokens, `pooled_projection_dim = 1536`) that interact only
//! through a single **joint** attention per block. Each block:
//!
//!  1. modulates both streams with `MochiRMSNormZero` (weightless f32 RMS-norm → `(1 + scale)`);
//!  2. runs joint attention — visual `to_{q,k,v}` (3072→3072) and text `add_{q,k,v}_proj`
//!     (1536→3072), per-head `qk_norm` (weighted RMS) on q/k **and** the added q/k, learned 3-D RoPE on
//!     the **visual** q/k only, then one masked SDPA over the concatenated `[visual | text]` keys
//!     (padded text keys get additive `−inf`), split back to `to_out` (3072→3072) + `to_add_out`
//!     (3072→1536);
//!  3. applies **tanh-gated** dual residuals and a SwiGLU FFN per stream.
//!
//! The final block is `context_pre_only` — it drops the text-stream output path (no `to_add_out` /
//! `ff_context`, and `norm1_context` is a `MochiLayerNormContinuous` instead).
//!
//! **Dtype (parity regime):** weights are stored at the loaded dtype (bf16 in production), but the whole
//! forward runs **f32 activations** — the exact `mlx-gen-mochi` compute regime the parity tolerances were
//! calibrated against. The `nn::linear_*` helpers upcast each bf16 weight to f32 at the matmul (a
//! transient per-weight view), so f32 activations flow through bf16-stored projections; when the weight
//! is already f32 (the CPU synthetic path) it is a no-op.

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::QLinear;
use candle_gen::{CandleError, Result};

use crate::nn::{
    layer_norm_no_affine, linear_b, rms_weighted, rms_weightless, silu, timestep_sincos,
};
use crate::rope::MochiRope;

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
    pub eps: f64,
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

    /// Visual SwiGLU hidden width (`FeedForward(mult=4)` × swiglu `2/3`): `inner·4·2/3` (= 8192). The
    /// `ff.net.0.proj` maps `inner → 2·ff_inner`, `ff.net.2` maps `ff_inner → inner`. Matches the
    /// crate's own block-fixture arithmetic and the real snapshot shapes.
    pub fn ff_inner(&self) -> usize {
        (self.inner_dim() * 4 * 2) / 3
    }

    /// Text (`ff_context`) SwiGLU hidden width: `pooled·4·2/3` (= 4096).
    pub fn ff_ctx_inner(&self) -> usize {
        (self.pooled_dim * 4 * 2) / 3
    }
}

/// Per-head `qk_norm` epsilon (`MochiAttention(eps=1e-5)`), distinct from the block's `1e-6`.
const QK_NORM_EPS: f64 = 1e-5;

// ---------------------------------------------------------------------------- primitives

/// `emb.chunk(n, dim=-1)` — split a `[.., n·d]` modulation vector into `n` `[.., d]` parts.
fn chunk_last(x: &Tensor, n: usize) -> Result<Vec<Tensor>> {
    Ok(x.chunk(n, D::Minus1)?)
}

/// `(1 + scale)` broadcast over a length-1 sequence axis: `[B, d]` → `[B, 1, d]`.
fn scale_plus_one_seq(scale: &Tensor) -> Result<Tensor> {
    Ok(scale.affine(1.0, 1.0)?.unsqueeze(1)?)
}

/// `tanh(gate)` with a length-1 sequence axis inserted: `[B, d]` → `[B, 1, d]`.
fn tanh_gate_seq(gate: &Tensor) -> Result<Tensor> {
    Ok(gate.tanh()?.unsqueeze(1)?)
}

// ---------------------------------------------------------------------------- SwiGLU FFN

/// SwiGLU feed-forward (`FeedForward(activation="swiglu", bias=False)`): `proj` (`d → 2·ff_inner`),
/// split into `(value, gate)`, `value · silu(gate)`, then `out` (`ff_inner → d`). Both projections are
/// [`MOCHI_QUANT_SUFFIXES`](../../mlx-gen/mlx-gen-mochi/src/convert.rs) targets, so each **packed-detects**
/// on its `.scales` sibling (a pre-quantized q4/q8 tier) via [`QLinear::linear_detect`], else stays dense.
struct SwiGlu {
    proj: QLinear, // [2·ff_inner, d]
    out: QLinear,  // [d, ff_inner]
}

impl SwiGlu {
    /// `d` is the stream width (visual `inner`, text `pooled`); `ff_inner` the SwiGLU hidden width. The
    /// dense fallback shapes off `(d, ff_inner)`; the packed path ignores them (dims come from the pack).
    fn load(vb: &VarBuilder, d: usize, ff_inner: usize) -> Result<Self> {
        Ok(Self {
            proj: QLinear::linear_detect(d, 2 * ff_inner, vb, "net.0.proj", false)?,
            out: QLinear::linear_detect(ff_inner, d, vb, "net.2", false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `forward_upcast`: the dense arm upcasts the bf16 weight to the f32 activation dtype per matmul
        // (byte-identical to the old `linear_nb`); the packed arm dequantizes to f32 and matmuls (sc-7702).
        let h = self.proj.forward_upcast(x)?;
        let parts = chunk_last(&h, 2)?;
        let gated = (&parts[0] * silu(&parts[1])?)?;
        Ok(self.out.forward_upcast(&gated)?)
    }
}

// ---------------------------------------------------------------------------- attention

/// Mochi joint attention (`MochiAttention` + `MochiAttnProcessor2_0`). The `to_q/k/v`, added
/// `add_{q,k,v}_proj`, `to_out.0`, and `to_add_out` projections are [`MOCHI_QUANT_SUFFIXES`] targets, so
/// each **packed-detects** on its `.scales` sibling (a pre-quantized q4/q8 tier) via
/// [`QLinear::linear_detect`], else loads dense unchanged. The per-head `qk_norm` weights stay dense.
pub struct MochiAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    norm_added_q: Tensor,
    norm_added_k: Tensor,
    /// `to_out.0` (biased).
    to_out: QLinear,
    /// `to_add_out` (biased) — absent when `context_pre_only`.
    to_add_out: Option<QLinear>,
    num_heads: usize,
    head_dim: usize,
}

impl MochiAttention {
    fn load(vb: &VarBuilder, cfg: &MochiDitConfig, context_pre_only: bool) -> Result<Self> {
        let inner = cfg.inner_dim();
        let pooled = cfg.pooled_dim;
        let g = |n: &str| vb.get_unchecked(n);
        let to_add_out = if context_pre_only {
            None
        } else {
            // to_add_out: [pooled, inner] — the visual→text output projection (biased).
            Some(QLinear::linear_detect(
                inner,
                pooled,
                vb,
                "to_add_out",
                true,
            )?)
        };
        Ok(Self {
            to_q: QLinear::linear_detect(inner, inner, vb, "to_q", false)?,
            to_k: QLinear::linear_detect(inner, inner, vb, "to_k", false)?,
            to_v: QLinear::linear_detect(inner, inner, vb, "to_v", false)?,
            // add_{q,k,v}_proj: [inner, pooled] — the text stream is projected pooled → inner.
            add_q: QLinear::linear_detect(pooled, inner, vb, "add_q_proj", false)?,
            add_k: QLinear::linear_detect(pooled, inner, vb, "add_k_proj", false)?,
            add_v: QLinear::linear_detect(pooled, inner, vb, "add_v_proj", false)?,
            norm_q: g("norm_q.weight")?,
            norm_k: g("norm_k.weight")?,
            norm_added_q: g("norm_added_q.weight")?,
            norm_added_k: g("norm_added_k.weight")?,
            to_out: QLinear::linear_detect(inner, inner, vb, "to_out.0", true)?,
            to_add_out,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Split `[B, S, inner]` → `[B, S, heads, head_dim]`.
    fn to_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        Ok(x.reshape((b, s, self.num_heads, self.head_dim))?)
    }

    /// Joint attention. `visual [B, Sv, inner]`, `text [B, St, pooled]`, `enc_mask [B, St]` (0/1).
    /// Returns `(visual_out [B, Sv, inner], Some(text_out [B, St, pooled]))` (text `None` when
    /// `context_pre_only`). All-f32.
    pub fn forward(
        &self,
        visual: &Tensor,
        text: &Tensor,
        rope: &MochiRope,
        enc_mask: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let sv = visual.dim(1)?;
        let st = text.dim(1)?;

        // Visual q/k/v (+ per-head qk_norm) with RoPE on q/k. `forward_upcast`: dense arm upcasts the
        // bf16 weight to the f32 activation dtype (== the old `linear_nb`); packed arm dequantizes to f32.
        let q = self.to_heads(&self.to_q.forward_upcast(visual)?)?;
        let k = self.to_heads(&self.to_k.forward_upcast(visual)?)?;
        let v = self.to_heads(&self.to_v.forward_upcast(visual)?)?;
        let q = rope.apply(&rms_weighted(&q, &self.norm_q, QK_NORM_EPS)?)?;
        let k = rope.apply(&rms_weighted(&k, &self.norm_k, QK_NORM_EPS)?)?;

        // Text q/k/v (+ per-head qk_norm), no RoPE.
        let eq = self.to_heads(&self.add_q.forward_upcast(text)?)?;
        let ek = self.to_heads(&self.add_k.forward_upcast(text)?)?;
        let ev = self.to_heads(&self.add_v.forward_upcast(text)?)?;
        let eq = rms_weighted(&eq, &self.norm_added_q, QK_NORM_EPS)?;
        let ek = rms_weighted(&ek, &self.norm_added_k, QK_NORM_EPS)?;
        // v/ev come from bf16-stored projections; upcast to f32 to match q/k (rms/rope already f32).
        let v = v.to_dtype(DType::F32)?;
        let ev = ev.to_dtype(DType::F32)?;

        // → [B, heads, S, head_dim]; concat visual + text along the sequence axis.
        let t = |a: &Tensor| -> Result<Tensor> { Ok(a.transpose(1, 2)?.contiguous()?) };
        let full_q = Tensor::cat(&[&t(&q)?, &t(&eq)?], 2)?.contiguous()?;
        let full_k = Tensor::cat(&[&t(&k)?, &t(&ek)?], 2)?.contiguous()?;
        let full_v = Tensor::cat(&[&t(&v)?, &t(&ev)?], 2)?.contiguous()?;

        let mask = build_joint_mask(enc_mask, sv)?;
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let out = crate::nn::sdpa(&full_q, &full_k, &full_v, scale, Some(&mask))?;

        // → [B, Sv+St, inner]; split back to visual / text.
        let (b, _, _, _) = out.dims4()?;
        let inner = self.num_heads * self.head_dim;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, sv + st, inner))?;
        let vis = out.narrow(1, 0, sv)?;
        let txt = out.narrow(1, sv, st)?;

        let hidden = self.to_out.forward_upcast(&vis)?;
        let enc = match &self.to_add_out {
            Some(l) => Some(l.forward_upcast(&txt)?),
            None => None,
        };
        Ok((hidden, enc))
    }
}

/// Build the additive joint attention mask `[B, 1, 1, num_visual + St]`: `0` for the visual keys and
/// valid text keys, `−inf` for padded text keys (`enc_mask == 0`). Broadcasts over query + heads.
fn build_joint_mask(enc_mask: &Tensor, num_visual: usize) -> Result<Tensor> {
    let (b, st) = enc_mask.dims2()?;
    let m = enc_mask.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let total = num_visual + st;
    let mut data = vec![0f32; b * total];
    for (bi, row) in m.iter().enumerate() {
        for (j, &mv) in row.iter().enumerate() {
            if mv == 0.0 {
                data[bi * total + num_visual + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, (b, 1, 1, total), enc_mask.device())?)
}

// ---------------------------------------------------------------------------- block

/// `norm1_context` variant: a `MochiRMSNormZero` (non-final blocks) or a `MochiLayerNormContinuous`
/// (the final `context_pre_only` block).
enum NormContext {
    /// `MochiRMSNormZero`: `linear [4·pooled, inner]` → 4 modulation chunks.
    Zero { lin_w: Tensor, lin_b: Tensor },
    /// `MochiLayerNormContinuous`: `linear_1 [pooled, inner]` → a single scale.
    Continuous { lin_w: Tensor, lin_b: Tensor },
}

/// One `MochiTransformerBlock` — the dual-stream MMDiT block.
pub struct MochiTransformerBlock {
    norm1_w: Tensor, // [4·inner, inner]
    norm1_b: Tensor,
    norm1_context: NormContext,
    attn: MochiAttention,
    ff: SwiGlu,
    /// `None` on the final `context_pre_only` block — the text output path is dropped.
    ff_context: Option<SwiGlu>,
    eps: f64,
}

impl MochiTransformerBlock {
    /// Load block `vb` (e.g. `transformer_blocks.0`). `context_pre_only` (the final block) drops the
    /// text output path.
    pub fn load(vb: &VarBuilder, cfg: &MochiDitConfig, context_pre_only: bool) -> Result<Self> {
        let norm1_context = if context_pre_only {
            NormContext::Continuous {
                lin_w: vb.get_unchecked("norm1_context.linear_1.weight")?,
                lin_b: vb.get_unchecked("norm1_context.linear_1.bias")?,
            }
        } else {
            NormContext::Zero {
                lin_w: vb.get_unchecked("norm1_context.linear.weight")?,
                lin_b: vb.get_unchecked("norm1_context.linear.bias")?,
            }
        };
        let ff_context = if context_pre_only {
            None
        } else {
            Some(SwiGlu::load(
                &vb.pp("ff_context"),
                cfg.pooled_dim,
                cfg.ff_ctx_inner(),
            )?)
        };
        Ok(Self {
            norm1_w: vb.get_unchecked("norm1.linear.weight")?,
            norm1_b: vb.get_unchecked("norm1.linear.bias")?,
            norm1_context,
            attn: MochiAttention::load(&vb.pp("attn1"), cfg, context_pre_only)?,
            ff: SwiGlu::load(&vb.pp("ff"), cfg.inner_dim(), cfg.ff_inner())?,
            ff_context,
            eps: cfg.eps,
        })
    }

    /// Forward the block. `hidden [B, Sv, inner]`, `enc [B, St, pooled]`, `temb [B, inner]`,
    /// `enc_mask [B, St]`. Returns the updated `(hidden, enc)`. All-f32.
    pub fn forward(
        &self,
        hidden: &Tensor,
        enc: &Tensor,
        temb: &Tensor,
        rope: &MochiRope,
        enc_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let eps = self.eps;
        let silu_temb = silu(&temb.to_dtype(DType::F32)?)?;

        // norm1 (visual): (scale_msa, gate_msa, scale_mlp, gate_mlp).
        let emb = linear_b(&silu_temb, &self.norm1_w, &self.norm1_b)?;
        let c = chunk_last(&emb, 4)?;
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = (&c[0], &c[1], &c[2], &c[3]);
        let norm_h = rms_weightless(hidden, eps)?.broadcast_mul(&scale_plus_one_seq(scale_msa)?)?;

        // norm1_context (text).
        let (norm_e, ctx_gates) = match &self.norm1_context {
            NormContext::Zero { lin_w, lin_b } => {
                let emb_c = linear_b(&silu_temb, lin_w, lin_b)?;
                let cc = chunk_last(&emb_c, 4)?;
                let norm_e =
                    rms_weightless(enc, eps)?.broadcast_mul(&scale_plus_one_seq(&cc[0])?)?;
                (norm_e, Some((cc[1].clone(), cc[2].clone(), cc[3].clone())))
            }
            NormContext::Continuous { lin_w, lin_b } => {
                let scale_c = linear_b(&silu_temb, lin_w, lin_b)?;
                let norm_e =
                    rms_weightless(enc, eps)?.broadcast_mul(&scale_plus_one_seq(&scale_c)?)?;
                (norm_e, None)
            }
        };

        // Joint attention.
        let (attn_h, attn_e) = self.attn.forward(&norm_h, &norm_e, rope, enc_mask)?;

        // Visual residuals: tanh-gated attn, SwiGLU FFN with (1+scale_mlp) mod, tanh-gated ff.
        let hidden =
            (hidden + rms_weightless(&attn_h, eps)?.broadcast_mul(&tanh_gate_seq(gate_msa)?)?)?;
        let norm_h2 =
            rms_weightless(&hidden, eps)?.broadcast_mul(&scale_plus_one_seq(scale_mlp)?)?;
        let ff_out = self.ff.forward(&norm_h2)?;
        let hidden =
            (&hidden + rms_weightless(&ff_out, eps)?.broadcast_mul(&tanh_gate_seq(gate_mlp)?)?)?;

        // Text residuals (skipped on the final context_pre_only block).
        let enc = match (ctx_gates, attn_e, &self.ff_context) {
            (Some((e_gate_msa, e_scale_mlp, e_gate_mlp)), Some(attn_e), Some(ff_ctx)) => {
                let enc = (enc
                    + rms_weightless(&attn_e, eps)?
                        .broadcast_mul(&tanh_gate_seq(&e_gate_msa)?)?)?;
                let norm_e2 =
                    rms_weightless(&enc, eps)?.broadcast_mul(&scale_plus_one_seq(&e_scale_mlp)?)?;
                let ff_e = ff_ctx.forward(&norm_e2)?;
                (&enc + rms_weightless(&ff_e, eps)?.broadcast_mul(&tanh_gate_seq(&e_gate_mlp)?)?)?
            }
            _ => enc.clone(),
        };

        Ok((hidden, enc))
    }
}

// ---------------------------------------------------------------------- time embedding

/// `MochiAttentionPool` — a single learned query (the masked-mean "class" token) attends over the raw
/// T5 tokens to pool them into one `[B, output_dim]` conditioning vector.
struct AttentionPool {
    to_kv_w: Tensor,
    to_kv_b: Tensor,
    to_q_w: Tensor,
    to_q_b: Tensor,
    to_out_w: Tensor,
    to_out_b: Tensor,
    num_heads: usize,
    embed_dim: usize,
}

impl AttentionPool {
    fn load(vb: &VarBuilder, num_heads: usize) -> Result<Self> {
        let g = |n: &str| vb.get_unchecked(n);
        let to_kv_w = g("to_kv.weight")?;
        let embed_dim = to_kv_w.dim(1)?; // to_kv: [2·embed, embed]
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

    /// `x [B, L, D]` (f32), `mask [B, L]` (0/1 f32) → pooled `[B, output_dim]`.
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;
        let head_dim = self.embed_dim / self.num_heads;

        // pool_tokens: weighted mean over valid tokens → the query "class" token.
        let m = mask.to_dtype(DType::F32)?.reshape((b, l, 1))?;
        let denom = m.sum_keepdim(1)?.clamp(1.0, f64::INFINITY)?; // [B,1,1] clamp≥1
        let mnorm = m.broadcast_div(&denom)?;
        let x_pool = x.broadcast_mul(&mnorm)?.sum_keepdim(1)?; // [B, 1, D]

        // Concat pooled + tokens; KV over all, Q from the pooled token only.
        let xcat = Tensor::cat(&[&x_pool, x], 1)?; // [B, 1+L, D]
        let kv = linear_b(&xcat, &self.to_kv_w, &self.to_kv_b)?; // [B, 1+L, 2D]
        let q = linear_b(&x_pool.reshape((b, d))?, &self.to_q_w, &self.to_q_b)?; // [B, D]

        // Heads: kv [B, 1+L, 2, H, hd] → [B, H, 2, 1+L, hd] → k, v.
        let lk = l + 1;
        let kv = kv
            .reshape((b, lk, 2, self.num_heads, head_dim))?
            .permute((0, 3, 2, 1, 4))?
            .contiguous()?; // [B, H, 2, 1+L, hd]
        let k = kv.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [B, H, 1+L, hd]
        let v = kv.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let q = q.reshape((b, self.num_heads, 1, head_dim))?; // [B, H, 1, hd]

        // Additive mask [B, 1, 1, 1+L]: key 0 (pooled) always valid; text keys 0/−inf per `mask`.
        let mvals = mask.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut mdata = vec![0f32; b * lk];
        for (bi, row) in mvals.iter().enumerate() {
            for (j, &mv) in row.iter().enumerate() {
                if mv == 0.0 {
                    mdata[bi * lk + 1 + j] = f32::NEG_INFINITY;
                }
            }
        }
        let attn_mask = Tensor::from_vec(mdata, (b, 1, 1, lk), x.device())?;

        let scale = 1.0f64 / (head_dim as f64).sqrt();
        let out = crate::nn::sdpa(&q, &k, &v, scale, Some(&attn_mask))?; // [B,H,1,hd]
        let out = out.reshape((b, self.embed_dim))?; // squeeze(2).flatten(1,2)
        linear_b(&out, &self.to_out_w, &self.to_out_b)
    }
}

/// `MochiCombinedTimestepCaptionEmbedding` — sinusoidal-timestep MLP + masked attention-pool of the raw
/// T5 tokens (summed into `temb`), plus the `caption_proj` that projects the raw T5 tokens into the
/// 1536-dim text stream. Returns `(temb [B, inner], caption [B, L, pooled])`.
struct TimeEmbed {
    ts_lin1_w: Tensor,
    ts_lin1_b: Tensor,
    ts_lin2_w: Tensor,
    ts_lin2_b: Tensor,
    pooler: AttentionPool,
    caption_w: Tensor,
    caption_b: Tensor,
    time_embed_dim: usize,
}

impl TimeEmbed {
    fn load(vb: &VarBuilder, cfg: &MochiDitConfig) -> Result<Self> {
        let g = |n: &str| vb.get_unchecked(n);
        Ok(Self {
            ts_lin1_w: g("timestep_embedder.linear_1.weight")?,
            ts_lin1_b: g("timestep_embedder.linear_1.bias")?,
            ts_lin2_w: g("timestep_embedder.linear_2.weight")?,
            ts_lin2_b: g("timestep_embedder.linear_2.bias")?,
            // The reference builds the pooler with num_attention_heads=8.
            pooler: AttentionPool::load(&vb.pp("pooler"), 8)?,
            caption_w: g("caption_proj.weight")?,
            caption_b: g("caption_proj.bias")?,
            time_embed_dim: cfg.time_embed_dim,
        })
    }

    fn forward(
        &self,
        timestep: &Tensor,
        enc: &Tensor,
        enc_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0.0) → [B, time_embed_dim] (f32).
        let time_proj = timestep_sincos(
            &timestep.to_dtype(DType::F32)?,
            self.time_embed_dim,
            10000.0,
            timestep.device(),
        )?;
        let te = linear_b(&time_proj, &self.ts_lin1_w, &self.ts_lin1_b)?;
        let te = silu(&te)?;
        let te = linear_b(&te, &self.ts_lin2_w, &self.ts_lin2_b)?; // [B, inner]

        let pooled = self.pooler.forward(enc, enc_mask)?; // [B, inner]
        let caption = linear_b(enc, &self.caption_w, &self.caption_b)?; // [B, L, pooled]
        let temb = (te + pooled)?;
        Ok((temb, caption))
    }
}

// ---------------------------------------------------------------------- full model

/// The full Mochi 1 AsymmDiT — `MochiTransformer3DModel`. One `forward` = one CFG-branch velocity
/// prediction (call with the `[neg, pos]` batch for a full CFG step; combine downstream via
/// [`crate::scheduler::cfg_combine`]).
pub struct MochiTransformer3DModel {
    patch_w: Tensor, // torch [out, in, kh, kw]
    patch_b: Tensor,
    pos_frequencies: Tensor, // [3, heads, head_dim/2]
    time_embed: TimeEmbed,
    blocks: Vec<MochiTransformerBlock>,
    norm_out_w: Tensor, // [2·inner, inner]
    norm_out_b: Tensor,
    proj_out_w: Tensor, // [patch²·out_ch, inner]
    proj_out_b: Tensor,
    cfg: MochiDitConfig,
    device: Device,
}

impl MochiTransformer3DModel {
    /// Build the full model from a VarBuilder rooted at the transformer root (`patch_embed.*`,
    /// `pos_frequencies`, `time_embed.*`, `transformer_blocks.*`, `norm_out.*`, `proj_out.*`).
    pub fn new(vb: VarBuilder, cfg: &MochiDitConfig, device: &Device) -> Result<Self> {
        let bvb = vb.pp("transformer_blocks");
        let blocks = (0..cfg.num_layers)
            .map(|i| MochiTransformerBlock::load(&bvb.pp(i), cfg, i == cfg.num_layers - 1))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_w: vb.get_unchecked("patch_embed.proj.weight")?,
            patch_b: vb.get_unchecked("patch_embed.proj.bias")?,
            pos_frequencies: vb.get_unchecked("pos_frequencies")?,
            time_embed: TimeEmbed::load(&vb.pp("time_embed"), cfg)?,
            blocks,
            norm_out_w: vb.get_unchecked("norm_out.linear.weight")?,
            norm_out_b: vb.get_unchecked("norm_out.linear.bias")?,
            proj_out_w: vb.get_unchecked("proj_out.weight")?,
            proj_out_b: vb.get_unchecked("proj_out.bias")?,
            cfg: *cfg,
            device: device.clone(),
        })
    }

    /// Forward. `hidden [B, in_ch, F, H, W]` (latent), `enc [B, L, text_embed]` (raw T5), `timestep
    /// [B]`, `enc_mask [B, L]` (0/1). Returns the velocity `noise_pred [B, in_ch, F, H, W]` (f32).
    pub fn forward(
        &self,
        hidden: &Tensor,
        enc: &Tensor,
        timestep: &Tensor,
        enc_mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, c, f, h, wd) = hidden.dims5()?;
        let p = self.cfg.patch_size;
        let ph = h / p;
        let pw = wd / p;
        let inner = self.cfg.inner_dim();

        // f32 activation stream (the parity regime).
        let hidden = hidden.to_dtype(DType::F32)?;
        let enc = enc.to_dtype(DType::F32)?;
        let enc_mask = enc_mask.to_dtype(DType::F32)?;

        // Time / caption embedding (raw T5 → temb + 1536-dim text stream).
        let (temb, mut enc_stream) = self.time_embed.forward(timestep, &enc, &enc_mask)?;

        // Patchify (channels-first, candle-native NCHW conv2d): [B,C,F,H,W] → [B·F, C, H, W] →
        // Conv2d(patch, stride p) → [B·F, inner, ph, pw] → [B, F·ph·pw, inner].
        let x = hidden
            .permute((0, 2, 1, 3, 4))? // [B, F, C, H, W]
            .reshape((b * f, c, h, wd))?
            .contiguous()?;
        let pw_f32 = self.patch_w.to_dtype(DType::F32)?;
        let pb_f32 = self.patch_b.to_dtype(DType::F32)?;
        let x = x.conv2d(&pw_f32, 0, p, 1, 1)?; // [B·F, inner, ph, pw]
        let x = x.broadcast_add(&pb_f32.reshape((1, inner, 1, 1))?)?;
        let mut hs = x
            .reshape((b * f, inner, ph * pw))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, f * ph * pw, inner))?;

        // Learned 3-D RoPE over the post-patch grid.
        let rope = MochiRope::new(&self.pos_frequencies, f, ph, pw, &self.device)?;

        for block in &self.blocks {
            let (h_new, e_new) = block.forward(&hs, &enc_stream, &temb, &rope, &enc_mask)?;
            hs = h_new;
            enc_stream = e_new;
        }

        // AdaLayerNormContinuous (layer_norm, no affine) → proj_out.
        let emb = linear_b(&silu(&temb)?, &self.norm_out_w, &self.norm_out_b)?;
        let so = chunk_last(&emb, 2)?;
        let (scale, shift) = (&so[0], &so[1]);
        let normed = layer_norm_no_affine(&hs, 1e-6)?;
        let hs = normed
            .broadcast_mul(&scale_plus_one_seq(scale)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        let hs = linear_b(&hs, &self.proj_out_w, &self.proj_out_b)?; // [B, seq, p²·out_ch]

        // Unpatchify: [B, F, ph, pw, p, p, out_ch] → [B, out_ch, F, H, W].
        let out_ch = c; // out_channels == in_channels
        let hs = hs
            .reshape(vec![b, f, ph, pw, p, p, out_ch])?
            .permute([0usize, 6, 1, 2, 4, 3, 5])?
            .contiguous()?
            .reshape((b, out_ch, f, ph * p, pw * p))?;
        Ok(hs)
    }
}

/// Load the AsymmDiT transformer VarBuilder from `<root>/transformer/` — the **bf16** variant shards
/// referenced by `diffusion_pytorch_model.safetensors.index.bf16.json` (index-filtered so the overlapping
/// shard sets don't collide, like the T5 loader), at `dtype`.
pub fn load_transformer_var_builder(
    root: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let dir = root.join("transformer");
    let bf16_index = dir.join("diffusion_pytorch_model.safetensors.index.bf16.json");
    if bf16_index.exists() {
        return load_index_named(
            &dir,
            "diffusion_pytorch_model.safetensors.index.bf16.json",
            dtype,
            device,
        );
    }
    let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
    if index.exists() {
        return load_index_named(
            &dir,
            "diffusion_pytorch_model.safetensors.index.json",
            dtype,
            device,
        );
    }
    candle_gen::load_sorted_mmap(&dir, dtype, device, "mochi dit")
}

/// mmap a VarBuilder over only the shards referenced by `<dir>/<index_name>`'s `weight_map`.
fn load_index_named(
    dir: &std::path::Path,
    index_name: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let index = dir.join(index_name);
    let text = std::fs::read_to_string(&index)
        .map_err(|e| CandleError::Msg(format!("mochi dit index {}: {e}", index.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| CandleError::Msg(format!("mochi dit index {}: {e}", index.display())))?;
    let map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "mochi dit index {}: no weight_map",
                index.display()
            ))
        })?;
    let shard_files: std::collections::BTreeSet<String> = map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    let files: Vec<std::path::PathBuf> = shard_files.into_iter().map(|f| dir.join(f)).collect();
    candle_gen::mmap_var_builder(&files, dtype, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// Deterministic small "random" fill, bounded so the block stays well-conditioned.
    fn rnd(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                (((i as u64).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 1e-6).sin()
                    * 0.05
            })
            .collect();
        Tensor::from_vec(data, shape, dev).unwrap()
    }

    /// A tiny full-model config: 2 heads × 8 head-dim → inner 16, pooled 8, 4 latent channels, 2 layers
    /// (block 0 normal, block 1 `context_pre_only`).
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
        }
    }

    fn insert_block(
        w: &mut HashMap<String, Tensor>,
        cfg: &MochiDitConfig,
        prefix: &str,
        dev: &Device,
    ) {
        let inner = cfg.inner_dim();
        let pooled = cfg.pooled_dim;
        let hd = cfg.head_dim;
        let ff_inner = (4 * cfg.inner_dim() * 2) / 3;
        let ff_ctx_inner = (4 * cfg.pooled_dim * 2) / 3;
        let p = |s: &str| format!("{prefix}.{s}");
        let mut put = |k: String, t: Tensor| {
            w.insert(k, t);
        };
        put(p("norm1.linear.weight"), rnd(&[4 * inner, inner], 1, dev));
        put(p("norm1.linear.bias"), rnd(&[4 * inner], 2, dev));
        put(
            p("norm1_context.linear.weight"),
            rnd(&[4 * pooled, inner], 3, dev),
        );
        put(p("norm1_context.linear.bias"), rnd(&[4 * pooled], 4, dev));
        put(p("attn1.to_q.weight"), rnd(&[inner, inner], 5, dev));
        put(p("attn1.to_k.weight"), rnd(&[inner, inner], 6, dev));
        put(p("attn1.to_v.weight"), rnd(&[inner, inner], 7, dev));
        put(p("attn1.add_q_proj.weight"), rnd(&[inner, pooled], 8, dev));
        put(p("attn1.add_k_proj.weight"), rnd(&[inner, pooled], 9, dev));
        put(p("attn1.add_v_proj.weight"), rnd(&[inner, pooled], 10, dev));
        put(p("attn1.norm_q.weight"), rnd(&[hd], 11, dev));
        put(p("attn1.norm_k.weight"), rnd(&[hd], 12, dev));
        put(p("attn1.norm_added_q.weight"), rnd(&[hd], 13, dev));
        put(p("attn1.norm_added_k.weight"), rnd(&[hd], 14, dev));
        put(p("attn1.to_out.0.weight"), rnd(&[inner, inner], 15, dev));
        put(p("attn1.to_out.0.bias"), rnd(&[inner], 16, dev));
        put(p("attn1.to_add_out.weight"), rnd(&[pooled, inner], 17, dev));
        put(p("attn1.to_add_out.bias"), rnd(&[pooled], 18, dev));
        put(
            p("ff.net.0.proj.weight"),
            rnd(&[2 * ff_inner, inner], 19, dev),
        );
        put(p("ff.net.2.weight"), rnd(&[inner, ff_inner], 20, dev));
        put(
            p("ff_context.net.0.proj.weight"),
            rnd(&[2 * ff_ctx_inner, pooled], 21, dev),
        );
        put(
            p("ff_context.net.2.weight"),
            rnd(&[pooled, ff_ctx_inner], 22, dev),
        );
    }

    fn tiny_full_weights(cfg: &MochiDitConfig, dev: &Device) -> HashMap<String, Tensor> {
        let inner = cfg.inner_dim();
        let pooled = cfg.pooled_dim;
        let te = cfg.text_embed_dim;
        let ted = cfg.time_embed_dim;
        let in_ch = cfg.in_channels;
        let half = cfg.head_dim / 2;
        let out_dims = cfg.patch_size * cfg.patch_size * cfg.in_channels;

        let mut w = HashMap::new();
        w.insert(
            "patch_embed.proj.weight".into(),
            rnd(&[inner, in_ch, 2, 2], 200, dev),
        );
        w.insert("patch_embed.proj.bias".into(), rnd(&[inner], 201, dev));
        w.insert(
            "pos_frequencies".into(),
            rnd(&[3, cfg.num_heads, half], 202, dev),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_1.weight".into(),
            rnd(&[inner, ted], 203, dev),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_1.bias".into(),
            rnd(&[inner], 204, dev),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_2.weight".into(),
            rnd(&[inner, inner], 205, dev),
        );
        w.insert(
            "time_embed.timestep_embedder.linear_2.bias".into(),
            rnd(&[inner], 206, dev),
        );
        w.insert(
            "time_embed.pooler.to_kv.weight".into(),
            rnd(&[2 * te, te], 207, dev),
        );
        w.insert(
            "time_embed.pooler.to_kv.bias".into(),
            rnd(&[2 * te], 208, dev),
        );
        w.insert(
            "time_embed.pooler.to_q.weight".into(),
            rnd(&[te, te], 209, dev),
        );
        w.insert("time_embed.pooler.to_q.bias".into(), rnd(&[te], 210, dev));
        w.insert(
            "time_embed.pooler.to_out.weight".into(),
            rnd(&[inner, te], 211, dev),
        );
        w.insert(
            "time_embed.pooler.to_out.bias".into(),
            rnd(&[inner], 212, dev),
        );
        w.insert(
            "time_embed.caption_proj.weight".into(),
            rnd(&[pooled, te], 213, dev),
        );
        w.insert(
            "time_embed.caption_proj.bias".into(),
            rnd(&[pooled], 214, dev),
        );
        w.insert(
            "norm_out.linear.weight".into(),
            rnd(&[2 * inner, inner], 215, dev),
        );
        w.insert("norm_out.linear.bias".into(), rnd(&[2 * inner], 216, dev));
        w.insert("proj_out.weight".into(), rnd(&[out_dims, inner], 217, dev));
        w.insert("proj_out.bias".into(), rnd(&[out_dims], 218, dev));

        insert_block(&mut w, cfg, "transformer_blocks.0", dev);
        insert_block(&mut w, cfg, "transformer_blocks.1", dev);
        // Block 1 is the final context_pre_only block → needs norm1_context.linear_1.
        w.insert(
            "transformer_blocks.1.norm1_context.linear_1.weight".into(),
            rnd(&[pooled, inner], 219, dev),
        );
        w.insert(
            "transformer_blocks.1.norm1_context.linear_1.bias".into(),
            rnd(&[pooled], 220, dev),
        );
        w
    }

    /// A single joint-attention block forwards to the right shapes and is deterministic — the CPU
    /// CI-green DiT-block gate (no model weights).
    #[test]
    fn block_forward_shapes_and_determinism() {
        let dev = Device::Cpu;
        let cfg = MochiDitConfig {
            num_layers: 1,
            ..tiny_full_cfg()
        };
        let mut wmap = HashMap::new();
        insert_block(&mut wmap, &cfg, "transformer_blocks.0", &dev);
        let vb = VarBuilder::from_tensors(wmap, DType::F32, &dev);
        let block =
            MochiTransformerBlock::load(&vb.pp("transformer_blocks").pp(0), &cfg, false).unwrap();

        // 1 frame × 2 × 2 = 4 visual tokens, 3 text tokens (2 valid, 1 pad), inner 16, pooled 8.
        let hidden = rnd(&[1, 4, 16], 100, &dev);
        let enc = rnd(&[1, 3, 8], 101, &dev);
        let temb = rnd(&[1, 16], 102, &dev);
        let enc_mask = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], (1, 3), &dev).unwrap();
        let pf = rnd(&[3, 2, 4], 103, &dev);
        let rope = MochiRope::new(&pf, 1, 2, 2, &dev).unwrap();

        let (h1, e1) = block
            .forward(&hidden, &enc, &temb, &rope, &enc_mask)
            .unwrap();
        assert_eq!(h1.dims(), &[1, 4, 16]);
        assert_eq!(e1.dims(), &[1, 3, 8]);

        let (h2, e2) = block
            .forward(&hidden, &enc, &temb, &rope, &enc_mask)
            .unwrap();
        let close = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                < 1e-6
        };
        assert!(close(&h1, &h2) && close(&e1, &e2));
        assert!(h1
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }

    /// The full model forwards a `[neg, pos]` batch to a velocity of the latent shape, is
    /// deterministic, and finite — the CPU CI-green full-DiT gate (no model weights).
    #[test]
    fn full_model_forward_shapes_and_determinism() {
        let dev = Device::Cpu;
        let cfg = tiny_full_cfg();
        let vb = VarBuilder::from_tensors(tiny_full_weights(&cfg, &dev), DType::F32, &dev);
        let model = MochiTransformer3DModel::new(vb, &cfg, &dev).unwrap();

        // [B=2, C=4, F=1, H=4, W=4] latent, 3 text tokens (2 valid), timestep per batch element.
        let hidden = rnd(&[2, 4, 1, 4, 4], 300, &dev);
        let enc = rnd(&[2, 3, 16], 301, &dev);
        let timestep = Tensor::from_vec(vec![0.0f32, 25.0], 2, &dev).unwrap();
        let enc_mask =
            Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 1.0, 0.0, 0.0], (2, 3), &dev).unwrap();

        let out = model.forward(&hidden, &enc, &timestep, &enc_mask).unwrap();
        assert_eq!(
            out.dims(),
            &[2, 4, 1, 4, 4],
            "noise_pred matches latent shape"
        );
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));

        let out2 = model.forward(&hidden, &enc, &timestep, &enc_mask).unwrap();
        let d = (&out - &out2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(d, 0.0, "forward is deterministic");
    }

    /// The final `context_pre_only` block returns `enc` unchanged (the text output path is dropped).
    #[test]
    fn context_pre_only_block_returns_enc_unchanged() {
        let dev = Device::Cpu;
        let cfg = MochiDitConfig {
            num_layers: 1,
            ..tiny_full_cfg()
        };
        let mut wmap = HashMap::new();
        insert_block(&mut wmap, &cfg, "transformer_blocks.0", &dev);
        // Final block: continuous norm1_context; the Zero linear + text-only weights are ignored.
        wmap.insert(
            "transformer_blocks.0.norm1_context.linear_1.weight".into(),
            rnd(&[cfg.pooled_dim, cfg.inner_dim()], 30, &dev),
        );
        wmap.insert(
            "transformer_blocks.0.norm1_context.linear_1.bias".into(),
            rnd(&[cfg.pooled_dim], 31, &dev),
        );
        let vb = VarBuilder::from_tensors(wmap, DType::F32, &dev);
        let block =
            MochiTransformerBlock::load(&vb.pp("transformer_blocks").pp(0), &cfg, true).unwrap();

        let hidden = rnd(&[1, 4, 16], 100, &dev);
        let enc = rnd(&[1, 3, 8], 101, &dev);
        let temb = rnd(&[1, 16], 102, &dev);
        let enc_mask = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], (1, 3), &dev).unwrap();
        let pf = rnd(&[3, 2, 4], 103, &dev);
        let rope = MochiRope::new(&pf, 1, 2, 2, &dev).unwrap();
        let (h, e) = block
            .forward(&hidden, &enc, &temb, &rope, &enc_mask)
            .unwrap();
        assert_eq!(h.dims(), &[1, 4, 16]);
        // enc is bit-identical to the input (no context update on the final block).
        let same = (&e - &enc)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(same, 0.0);
    }
}
