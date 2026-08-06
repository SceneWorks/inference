//! SANA **Linear Diffusion Transformer trunk** — faithful mlx-rs port of diffusers
//! `SanaTransformer2DModel` / `SanaTransformerBlock` (epic 8485, story sc-8487).
//!
//! Port target: `Efficient-Large-Model/Sana_1600M_1024px_diffusers` (the 1.6B model Clark Labs
//! ported to MLX). We write the **bf16/fp16** path (the checkpoint dtype is preserved through the
//! forward — every op is dtype-preserving); we do NOT copy Clark Labs' 2-bit ternary quant (that was
//! a small-machine fit, not a fidelity requirement).
//!
//! ## Architecture (the four story pillars)
//!
//!  - **ReLU linear self-attention** (`attn1`, `SanaLinearAttnProcessor2_0`) — O(N) attention:
//!    `ReLU(Q),ReLU(K)`, then the `value`-padded-with-a-ones-row trick collapsed to the algebraically
//!    identical numerator/denominator split `num = (Vᵀ·K)·Q`, `den = (Σ_n K)·Q`, divided with a
//!    `1/(·+1e-15)` normalizer — the SAME f32 linear-attention kernel the DC-AE spike
//!    (`crate::dc_ae::LinearAttn`) uses, minus the multiscale QKV projections (the trunk's `attn1`
//!    is plain single-scale). `attention_bias=false` for SANA-1.6B → `to_q/k/v` have no bias;
//!    `to_out.0` carries a bias.
//!  - **Cross-attention** (`attn2`, standard softmax SDPA) to the caption embeddings — `to_q/k/v` all
//!    bias-carrying, KV from the projected+normed caption.
//!  - **Mix-FFN** (`ff`, `GLUMBConv`) — `conv_inverted(1×1) → SiLU → conv_depth(3×3 depthwise) → gated
//!    SiLU → conv_point(1×1, no bias)`. The 3×3 depthwise conv is the token-mixer; the FFN runs over
//!    the un-flattened `[B, inner, H, W]` grid (channels-first in the reference; channels-last here).
//!    No residual/norm inside the block's `ff` (the block owns the residual + gate).
//!  - **NoPE** — `interpolation_scale=None` ⇒ `patch_embed` has no `pos_embed`; the conv patchify
//!    (here `patch_size=1`, a 1×1 conv) plus the Mix-FFN depthwise conv provide all locality.
//!
//! Per-block adaLN-single modulation `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
//! gate_mlp)` comes from `block.scale_shift_table[6,dim] + timestep_emb.reshape(B,6,-1)`; the
//! timestep path is `Timesteps(256) → timestep_embedder(MLP) = embedded_timestep`, then
//! `time_embed.linear(SiLU(embedded_timestep)) → [B, 6·dim]`. Output: `SanaModulatedNorm`
//! (affine-free LayerNorm + `top.scale_shift_table[2,dim] + embedded_timestep`) → `proj_out` →
//! unpatchify to `[B, out_channels, H, W]` (32 channels = the DC-AE f32c32 latent, so the trunk's
//! output feeds [`crate::dc_ae::DcAeDecoder::decode`] directly — sc-8489 composition).
//!
//! Tensor keys are the diffusers `SanaTransformer2DModel` names exactly, so a converted checkpoint
//! loads unchanged. Layout convention follows [`crate::dc_ae`]: channels-last NHWC for the conv ops,
//! `[B, N, C]` token layout for the attention/Linear ops (diffusers' `flatten/permute` between the
//! two is mirrored explicitly).

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{
    add, clip, concatenate_axis, divide, matmul, multiply, softmax_axis, split_sections, subtract,
    sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::attention::AttentionBudget;
use mlx_gen::block_residency::BlockPlan;
use mlx_gen::nn::{gelu_tanh, silu, timestep_sincos};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

use crate::block_stream::SanaBlockStream;
use crate::config::SanaTransformerConfig;

const F32: Dtype = Dtype::Float32;

/// The two constrained rungs a single trunk forward may run under (SC-15523).
///
/// Rung 3 is [`Self::attention`] — the score budget SANA's `attn2` cross-attention chunks its query
/// rows against. Rung 4 is [`Self::window`] — the block cadence the trunk materializes its
/// `transformer_blocks` stack in. [`Self::resident`] is the historical path: an unbounded budget and
/// no window, so a request that selects neither rung runs byte-for-byte the pre-SC-15523 forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SanaForwardPlan {
    pub(crate) attention: AttentionBudget,
    pub(crate) window: Option<BlockPlan>,
}

impl SanaForwardPlan {
    /// The unbounded path — no attention chunking, no block window. Byte-for-byte the pre-SC-15523
    /// forward.
    pub(crate) const RESIDENT: Self = Self {
        attention: AttentionBudget::UNBOUNDED,
        window: None,
    };
}

/// Test-only observation of how many query chunks the last [`CrossAttn::forward`] actually ran.
///
/// Without it every "chunked == unbounded" equivalence assertion in this crate passes with the
/// chunking deleted, because the claim is trivially true when the lever never engages. Mirrors
/// [`mlx_gen::attention`]'s probe. Compiled out entirely in release.
#[cfg(test)]
mod cross_attn_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static LAST_CHUNK_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Chunks run by the most recent cross-attention call (`1` = the unchunked path).
    pub(crate) fn last_chunk_count() -> usize {
        LAST_CHUNK_COUNT.load(Ordering::Relaxed)
    }

    pub(crate) fn reset() {
        LAST_CHUNK_COUNT.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
pub(crate) use cross_attn_probe::{last_chunk_count, reset as reset_chunk_count};

#[cfg(test)]
fn record_chunk_count(n: usize) {
    cross_attn_probe::LAST_CHUNK_COUNT.store(n, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn record_chunk_count(_n: usize) {}

/// Contiguous `[.., start..start+len, ..]` slice along `axis` via boundary splits — no host index
/// vector and no gather, matching [`mlx_gen::attention`]'s own helper.
fn slice_axis(a: &Array, axis: i32, start: i32, len: i32) -> Result<Array> {
    Ok(a.split_axis(&[start, start + len], axis)?.swap_remove(1))
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn relu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::relu(x)?)
}

// ----------------------------------------------------------------------------------------------
// Linear / norm primitives (dtype-preserving; bf16/fp16 weights flow through unchanged).
// ----------------------------------------------------------------------------------------------

// Every trunk `nn.Linear` loads through [`crate::quant::lin`] as an [`AdaptableLinear`] — packed
// (Q4/Q8) when the on-disk `{base}.scales` is present, else dense (bf16), with identical numerics to
// the former bespoke `Linear` (`x · Wᵀ (+ b)`). The `.base_shape()[0]` accessor recovers the output
// (inner) dim the attention code needs, dense or packed alike (sc-8489, Group-B sc-8669).

/// `RMSNorm(elementwise_affine=True, bias=False)` over the last axis, f32 reduction (diffusers
/// `caption_norm`). `weight` is `[C]`.
fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let dt = x.dtype();
    let rank = x.shape().len();
    let ax = (rank - 1) as i32;
    let xf = x.as_dtype(F32)?;
    let var = mlx_rs::ops::mean_axes(&multiply(&xf, &xf)?, &[ax], true)?;
    let normed = multiply(&xf, &add(&var, scalar(eps))?.rsqrt()?)?;
    Ok(multiply(&normed.as_dtype(dt)?, weight)?)
}

/// adaLN-single affine `norm·(1 + scale) + shift` (diffusers `hidden * (1 + scale) + shift`).
fn modulate(norm: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    let one = scalar(1.0).as_dtype(scale.dtype())?;
    Ok(add(&multiply(norm, &add(scale, &one)?)?, shift)?)
}

// ----------------------------------------------------------------------------------------------
// Conv (channels-last NHWC; PyTorch [O, I/groups, H, W] → mlx [O, H, W, I/groups] at load).
// ----------------------------------------------------------------------------------------------

struct Conv {
    w: Array,
    b: Option<Array>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv {
    fn load(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        groups: i32,
        bias: bool,
    ) -> Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?;
        let b = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.clone())
        } else {
            None
        };
        Ok(Self {
            w: weight,
            b,
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = mlx_rs::ops::conv2d(
            x,
            &self.w,
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            self.groups,
        )?;
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

// ----------------------------------------------------------------------------------------------
// ReLU linear self-attention (attn1).
// ----------------------------------------------------------------------------------------------

/// `SanaLinearAttnProcessor2_0`: ReLU linear attention over the token axis. Input/output `[B, N, C]`.
struct LinearSelfAttn {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    /// Sprint `qk_norm = "rms_norm_across_heads"` (sc-8490): RMSNorm over the full projected query /
    /// key (the whole `inner_dim`), applied BEFORE the head split and the ReLU. `None` for base SANA.
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    heads: i32,
    attn_eps: f32,
    /// qk-norm RMSNorm eps (`1e-5`, diffusers `Attention.__init__` default). NOT `cfg.norm_eps`
    /// (`1e-6`), which governs only the affine-free LayerNorms.
    qk_norm_eps: f32,
}

impl LinearSelfAttn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(w.require(&format!("{prefix}.norm_q.weight"))?.clone()),
                Some(w.require(&format!("{prefix}.norm_k.weight"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            // attention_bias=false → q/k/v bias-free; to_out.0 carries a bias.
            to_q: crate::quant::lin(w, &format!("{prefix}.to_q"), false)?,
            to_k: crate::quant::lin(w, &format!("{prefix}.to_k"), false)?,
            to_v: crate::quant::lin(w, &format!("{prefix}.to_v"), false)?,
            to_out: crate::quant::lin(w, &format!("{prefix}.to_out.0"), true)?,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads,
            attn_eps: cfg.attn_eps,
            qk_norm_eps: cfg.attn_qk_norm_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let inner = self.to_q.base_shape()[0];
        let hd = inner / self.heads;

        // qk_norm = "rms_norm_across_heads": RMSNorm over the full `inner_dim`, BEFORE the head split
        // (diffusers applies `attn.norm_q(query)` / `attn.norm_k(key)` to the `[B,N,inner]` projection).
        let q_proj = self.to_q.forward(x)?;
        let q_proj = match &self.norm_q {
            Some(g) => rms_norm(&q_proj, g, self.qk_norm_eps)?,
            None => q_proj,
        };
        let k_proj = self.to_k.forward(x)?;
        let k_proj = match &self.norm_k {
            Some(g) => rms_norm(&k_proj, g, self.qk_norm_eps)?,
            None => k_proj,
        };

        // [B,N,inner] → [B, heads, hd, N]  (diffusers: transpose(1,2).unflatten(1,(heads,-1)))
        let to_bh_d_n = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.heads, hd])?
                .transpose_axes(&[0, 2, 3, 1])?)
        };
        let q = relu(&to_bh_d_n(q_proj)?)?.as_dtype(F32)?; // [B,H,hd,N]
        let k = relu(&to_bh_d_n(k_proj)?)?.as_dtype(F32)?; // [B,H,hd,N]
        let v = to_bh_d_n(self.to_v.forward(x)?)?.as_dtype(F32)?; // [B,H,hd,N]

        // Reference pads value with a ones-row then divides by it. Algebraically identical f32 split:
        //   num = (V·Kᵀ)·Q : [B,H,hd,N]   den = (Σ_n K)·Q : [B,H,1,N]
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?; // [B,H,N,hd]
        let num = matmul(&matmul(&v, &k_t)?, &q)?; // [B,H,hd,N]
        let k_sum = sum_axes(&k, &[3], true)?; // [B,H,hd,1]
        let den = matmul(&k_sum.transpose_axes(&[0, 1, 3, 2])?, &q)?; // [B,H,1,N]
        let out = divide(&num, &add(&den, scalar(self.attn_eps))?)?; // [B,H,hd,N]

        // [B,H,hd,N] → [B,N,inner]
        let out = out
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, n, inner])?
            .as_dtype(x.dtype())?;
        let out = self.to_out.forward(&out)?;

        // Reference (`SanaLinearAttnProcessor2_0`) clips `to_out` to fp16's representable range as an
        // overflow guard — but only when the *input* dtype was fp16 (`if original_dtype ==
        // torch.float16: hidden_states.clip(-65504, 65504)`). bf16/f32 are left unchanged.
        if x.dtype() == Dtype::Float16 {
            Ok(clip(&out, (-65504.0, 65504.0))?)
        } else {
            Ok(out)
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Standard cross-attention (attn2) to the caption embedding.
// ----------------------------------------------------------------------------------------------

struct CrossAttn {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    /// Sprint `qk_norm = "rms_norm_across_heads"` (sc-8490): RMSNorm over the full projected query /
    /// key (the whole cross `inner_dim`), applied BEFORE the head split. `None` for base SANA.
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    heads: i32,
    /// qk-norm RMSNorm eps (`1e-5`, diffusers `Attention.__init__` default). NOT `cfg.norm_eps`
    /// (`1e-6`), which governs only the affine-free LayerNorms.
    qk_norm_eps: f32,
}

impl CrossAttn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(w.require(&format!("{prefix}.norm_q.weight"))?.clone()),
                Some(w.require(&format!("{prefix}.norm_k.weight"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: crate::quant::lin(w, &format!("{prefix}.to_q"), true)?,
            to_k: crate::quant::lin(w, &format!("{prefix}.to_k"), true)?,
            to_v: crate::quant::lin(w, &format!("{prefix}.to_v"), true)?,
            to_out: crate::quant::lin(w, &format!("{prefix}.to_out.0"), true)?,
            norm_q,
            norm_k,
            heads: cfg.num_cross_attention_heads,
            qk_norm_eps: cfg.attn_qk_norm_eps,
        })
    }

    /// `x` (query) `[B, N, C]`, `kv` (caption) `[B, M, C]`, `kv_mask` (optional `[B, M]`, `1.0` real /
    /// `0.0` padding) — the caption padding mask diffusers passes as `encoder_attention_mask`. Applied
    /// additively to the pre-softmax logits so PAD keys contribute nothing.
    ///
    /// ## Rung 3 (SC-15523): why the chunking lives here and not on the shared MLX kernel
    ///
    /// [`mlx_gen::attention::sdpa_budgeted_bhsd`] is the MLX kernel for the **fused**
    /// `scaled_dot_product_attention`, which never materializes a score tensor — on that kernel the
    /// only rung-3 lever is the per-chunk graph cut. SANA's `attn2` is the other case the shared
    /// planner's module doc names explicitly: a hand-rolled `matmul → mask → softmax → matmul` that
    /// **does** materialize `[B, H, N, M]` in f32. Routing it through the fused kernel would swap
    /// SANA's arithmetic for Metal's fused one on the RESIDENT path too, which is a numerics change,
    /// not a memory rung — so the kernel stays SANA's and only the **planner** is shared. That is
    /// exactly the split [`gen_core::attention_budget`] prescribes (shared planner, per-backend and
    /// per-shape kernel); no budget arithmetic is re-derived here.
    ///
    /// Chunking is along the query axis only. Each output row `n` depends on `q[.., n, ..]` and the
    /// **complete** k/v, and both reductions (`hd` for the logits, `M` for the context) are untouched
    /// by the split, so the chunked result is the unchunked result row for row.
    ///
    /// `attn1` is deliberately **not** chunked: SANA's ReLU-linear self-attention has no score tensor
    /// at all (`num = (V·Kᵀ)·Q` collapses the key axis into a `[B, H, hd, hd]` gram matrix), so there
    /// is nothing for a score budget to bound there.
    fn forward(
        &self,
        x: &Array,
        kv: &Array,
        kv_mask: Option<&Array>,
        budget: AttentionBudget,
    ) -> Result<Array> {
        let xsh = x.shape();
        let (b, n) = (xsh[0], xsh[1]);
        let m = kv.shape()[1];
        let inner = self.to_q.base_shape()[0];
        let hd = inner / self.heads;
        let scale = scalar(1.0 / (hd as f32).sqrt());

        // qk_norm = "rms_norm_across_heads": RMSNorm over the full cross `inner_dim`, BEFORE the head
        // split (diffusers `attn.norm_q(query)` / `attn.norm_k(key)` on the `[B,*,inner]` projection).
        let q_proj = self.to_q.forward(x)?;
        let q_proj = match &self.norm_q {
            Some(g) => rms_norm(&q_proj, g, self.qk_norm_eps)?,
            None => q_proj,
        };
        let k_proj = self.to_k.forward(kv)?;
        let k_proj = match &self.norm_k {
            Some(g) => rms_norm(&k_proj, g, self.qk_norm_eps)?,
            None => k_proj,
        };

        let split_heads = |a: Array, len: i32| -> Result<Array> {
            // [B,len,inner] → [B,heads,len,hd]
            Ok(a.reshape(&[b, len, self.heads, hd])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = split_heads(q_proj, n)?; // [B,H,N,hd]
        let k = split_heads(k_proj, m)?; // [B,H,M,hd]
        let v = split_heads(self.to_v.forward(kv)?, m)?; // [B,H,M,hd]

        // Softmax SDPA in f32 (caption seq is short; full attention).
        let qf = q.as_dtype(F32)?;
        let kt = k.as_dtype(F32)?.transpose_axes(&[0, 1, 3, 2])?; // [B,H,hd,M]
        let vf = v.as_dtype(F32)?; // [B,H,M,hd]
                                   // Additive caption padding mask: PAD keys (mask==0) get a large negative bias → ~0 after
                                   // softmax. Broadcast [B,M] → [B,1,1,M] over heads and query positions. Without this, a short
                                   // prompt (300 slots dominated by PAD) lets padding embeddings swamp the real conditioning.
                                   // It is broadcast over the query axis, so every chunk shares it unmodified.
        let bias = kv_mask
            .map(|mask| -> Result<Array> {
                let mask = mask.as_dtype(F32)?.reshape(&[b, 1, 1, m])?; // 1.0 real / 0.0 pad
                Ok(multiply(&subtract(scalar(1.0), &mask)?, scalar(-1e9))?) // 0 / -1e9
            })
            .transpose()?;
        let attend = |q_rows: &Array| -> Result<Array> {
            let scores = multiply(&matmul(q_rows, &kt)?, &scale)?; // [B,H,n,M]
            let scores = match &bias {
                Some(bias) => add(&scores, bias)?,
                None => scores,
            };
            Ok(matmul(&softmax_axis(&scores, -1, None)?, &vf)?) // [B,H,n,hd]
        };

        let block = budget.query_block(b, self.heads, n, m);
        let ctx = if block >= n {
            record_chunk_count(1);
            attend(&qf)?
        } else {
            let mut outs: Vec<Array> = Vec::with_capacity(n.div_euclid(block) as usize + 1);
            let mut start = 0;
            while start < n {
                let len = block.min(n - start);
                let out = attend(&slice_axis(&qf, 2, start, len)?)?;
                if budget.eval_per_chunk() {
                    mlx_rs::transforms::eval([&out])?;
                }
                outs.push(out);
                start += len;
            }
            record_chunk_count(outs.len());
            let refs: Vec<&Array> = outs.iter().collect();
            concatenate_axis(&refs, 2)?
        };

        let ctx = ctx
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, inner])?
            .as_dtype(x.dtype())?;
        self.to_out.forward(&ctx)
    }
}

// ----------------------------------------------------------------------------------------------
// GLUMBConv Mix-FFN (block `ff`: norm_type=None, residual_connection=False).
// ----------------------------------------------------------------------------------------------

struct GluMbConv {
    conv_inverted: Conv, // 1×1, in → 2·hidden  (+bias)
    conv_depth: Conv,    // 3×3 depthwise, 2·hidden → 2·hidden (+bias)
    conv_point: Conv,    // 1×1, hidden → out (no bias)
    hidden: i32,
}

impl GluMbConv {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hidden = (cfg.mlp_ratio * inner as f32) as i32;
        Ok(Self {
            conv_inverted: Conv::load(w, &format!("{prefix}.conv_inverted"), 1, 0, 1, true)?,
            conv_depth: Conv::load(w, &format!("{prefix}.conv_depth"), 1, 1, 2 * hidden, true)?,
            conv_point: Conv::load(w, &format!("{prefix}.conv_point"), 1, 0, 1, false)?,
            hidden,
        })
    }

    /// `x` is NHWC `[B, H, W, inner]`. Returns NHWC `[B, H, W, out]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv_inverted.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        let parts = split_sections(&h, &[self.hidden], 3)?; // chunk(2) over the channel (NHWC) axis
        let h = multiply(&parts[0], &silu(&parts[1])?)?;
        self.conv_point.forward(&h)
    }
}

// ----------------------------------------------------------------------------------------------
// SanaTransformerBlock.
// ----------------------------------------------------------------------------------------------

pub(crate) struct SanaBlock {
    scale_shift_table: Array, // [6, dim]
    attn1: LinearSelfAttn,
    attn2: CrossAttn,
    ff: GluMbConv,
    norm_eps: f32,
}

impl SanaBlock {
    pub(crate) fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        Ok(Self {
            scale_shift_table: w.require(&format!("{prefix}.scale_shift_table"))?.clone(),
            attn1: LinearSelfAttn::load(w, &format!("{prefix}.attn1"), cfg)?,
            attn2: CrossAttn::load(w, &format!("{prefix}.attn2"), cfg)?,
            ff: GluMbConv::load(w, &format!("{prefix}.ff"), cfg)?,
            norm_eps: cfg.norm_eps,
        })
    }

    /// `hidden` `[B, N, dim]` (N = H·W tokens), `caption` `[B, M, dim]`, `temb` `[B, 6·dim]`.
    ///
    /// `budget` is rung 3: [`AttentionBudget::UNBOUNDED`] (the default everywhere) keeps the
    /// historical single-call cross-attention.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        hidden: &Array,
        caption: &Array,
        caption_mask: Option<&Array>,
        temb: &Array,
        h: i32,
        w: i32,
        budget: AttentionBudget,
    ) -> Result<Array> {
        let dim = self.scale_shift_table.shape()[1];
        let b = hidden.shape()[0];
        // 1. Modulation: scale_shift_table[None] + temb.reshape(B,6,-1)  → chunk(6) along axis 1.
        let ss = self.scale_shift_table.reshape(&[1, 6, dim])?;
        let modg = add(&ss, &temb.reshape(&[b, 6, dim])?)?; // [B,6,dim]
        let mc = split_sections(&modg, &[1, 2, 3, 4, 5], 1)?; // 6 × [B,1,dim]
        let chunk = |i: usize| -> Result<Array> { Ok(mc[i].reshape(&[b, 1, dim])?) };
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);

        // 2. Self linear-attention.
        let norm_h = layer_norm(hidden, None, None, self.norm_eps)?;
        let norm_h = modulate(&norm_h, &scale_msa, &shift_msa)?;
        let attn_out = self.attn1.forward(&norm_h)?;
        let hidden = add(hidden, &multiply(&gate_msa, &attn_out)?)?;

        // 3. Cross-attention (no pre-norm in SANA — attn2 reads `hidden` directly).
        let cross = self.attn2.forward(&hidden, caption, caption_mask, budget)?;
        let hidden = add(&cross, &hidden)?;

        // 4. Mix-FFN. norm2 → modulate → un-flatten to [B,H,W,dim] → GLUMBConv → flatten → gate.
        let norm_h = layer_norm(&hidden, None, None, self.norm_eps)?;
        let norm_h = modulate(&norm_h, &scale_mlp, &shift_mlp)?;
        let grid = norm_h.reshape(&[b, h, w, dim])?; // [B,N,dim] → NHWC (channels-last)
        let ff = self.ff.forward(&grid)?;
        let ff = ff.reshape(&[b, h * w, dim])?;
        Ok(add(&hidden, &multiply(&gate_mlp, &ff)?)?)
    }
}

// ----------------------------------------------------------------------------------------------
// Full trunk.
// ----------------------------------------------------------------------------------------------

/// SANA Linear-DiT trunk (`SanaTransformer2DModel`).
pub struct SanaTransformer {
    cfg: SanaTransformerConfig,
    patch_embed: Conv, // proj: in → inner (kernel/stride = patch_size)
    // timestep path (AdaLayerNormSingle.emb + .linear, or — Sprint — the combined
    // timestep+guidance embedder, see `guidance_embedder`)
    ts_embedder_1: AdaptableLinear,
    ts_embedder_2: AdaptableLinear,
    time_linear: AdaptableLinear, // → 6·inner
    /// Sprint (sc-8490): the extra guidance embedder (`SanaCombinedTimestepGuidanceEmbeddings`). The
    /// embedded guidance scalar runs through the same `Timesteps(256)` sincos projection as the
    /// timestep, then this two-linear MLP, and is summed into the timestep conditioning. `None` for
    /// base SANA (`AdaLayerNormSingle`).
    guidance_embedder: Option<(AdaptableLinear, AdaptableLinear)>,
    // caption path
    caption_proj_1: AdaptableLinear,
    caption_proj_2: AdaptableLinear,
    caption_norm: Array, // RMSNorm weight [inner]
    blocks: Vec<SanaBlock>,
    scale_shift_table: Array, // [2, inner] (output modulated norm)
    proj_out: AdaptableLinear,
    /// Rung 4 (SC-15523): the re-openable `transformer/` source a window rebuilds blocks from.
    ///
    /// `None` for a load with no re-openable source (nothing to stream from), and the contract
    /// declares `BoundedTransformerResidency` unavailable for exactly those loads. The resident
    /// [`Self::blocks`] stay lazy MLX handles until something evaluates them, so holding both costs
    /// nothing: a windowed forward never touches the resident stack, and it never materializes.
    block_stream: Option<SanaBlockStream>,
}

impl SanaTransformer {
    pub fn from_weights(w: &Weights, cfg: SanaTransformerConfig) -> Result<Self> {
        let p = cfg.patch_size;
        let patch_embed = Conv::load(w, "patch_embed.proj", p, 0, 1, true)?;
        let mut blocks = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            blocks.push(SanaBlock::load(
                w,
                &format!("transformer_blocks.{i}"),
                &cfg,
            )?);
        }
        // Sprint's guidance variant (`SanaCombinedTimestepGuidanceEmbeddings`) drops the `.emb.`
        // nesting AdaLayerNormSingle introduces and adds a parallel `guidance_embedder`.
        let (ts1_key, ts2_key, guidance_embedder) = if cfg.guidance_embeds {
            (
                "time_embed.timestep_embedder.linear_1",
                "time_embed.timestep_embedder.linear_2",
                Some((
                    crate::quant::lin(w, "time_embed.guidance_embedder.linear_1", true)?,
                    crate::quant::lin(w, "time_embed.guidance_embedder.linear_2", true)?,
                )),
            )
        } else {
            (
                "time_embed.emb.timestep_embedder.linear_1",
                "time_embed.emb.timestep_embedder.linear_2",
                None,
            )
        };
        Ok(Self {
            patch_embed,
            ts_embedder_1: crate::quant::lin(w, ts1_key, true)?,
            ts_embedder_2: crate::quant::lin(w, ts2_key, true)?,
            time_linear: crate::quant::lin(w, "time_embed.linear", true)?,
            guidance_embedder,
            caption_proj_1: crate::quant::lin(w, "caption_projection.linear_1", true)?,
            caption_proj_2: crate::quant::lin(w, "caption_projection.linear_2", true)?,
            caption_norm: w.require("caption_norm.weight")?.clone(),
            blocks,
            scale_shift_table: w.require("scale_shift_table")?.clone(),
            proj_out: crate::quant::lin(w, "proj_out", true)?,
            cfg,
            block_stream: None,
        })
    }

    /// Attach the rung-4 block stream. Called by the loader for a re-openable snapshot load; the
    /// stream must describe the same stack this trunk just built, which
    /// [`Self::forward_with_memory`] re-checks before it runs a window.
    pub(crate) fn with_block_stream(mut self, stream: SanaBlockStream) -> Self {
        self.block_stream = Some(stream);
        self
    }

    /// The number of `transformer_blocks` this trunk runs — the rung-4 plan's `n_blocks`.
    pub(crate) fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Forward one denoise step.
    ///
    /// * `latent_nchw` — `[B, in_channels, H, W]` (channels-first, diffusers-native).
    /// * `caption` — `[B, M, caption_channels]` caption embedding (M = 300 for SANA-1.6B).
    /// * `timestep` — `[B]` (or `[1]`) scalar timestep(s).
    ///
    /// Returns the noise prediction `[B, out_channels, H, W]` (channels-first), where
    /// `out_channels == 32` matches the DC-AE f32c32 latent so the output feeds
    /// [`crate::dc_ae::DcAeDecoder::decode`] directly (sc-8489 composition).
    pub fn forward(&self, latent_nchw: &Array, caption: &Array, timestep: &Array) -> Result<Array> {
        self.forward_with_guidance(latent_nchw, caption, timestep, None, None)
    }

    /// [`Self::forward_with_guidance`] on the historical unbounded path (no rung 3, no rung 4).
    pub fn forward_with_guidance(
        &self,
        latent_nchw: &Array,
        caption: &Array,
        timestep: &Array,
        guidance: Option<&Array>,
        caption_mask: Option<&Array>,
    ) -> Result<Array> {
        self.forward_with_memory(
            latent_nchw,
            caption,
            timestep,
            guidance,
            caption_mask,
            SanaForwardPlan::RESIDENT,
            &CancelFlag::default(),
        )
    }

    /// [`Self::forward`] with an optional **embedded guidance scalar** (SANA-Sprint, sc-8490).
    ///
    /// * `guidance` — `[B]` (or `[1]`) the CFG-free guidance scalar (already multiplied by the
    ///   `guidance_embeds_scale` by the caller). `Some` only for a Sprint-config trunk
    ///   (`guidance_embeds = true`); `None` runs the base AdaLN-single path. Sprint feeds the scale
    ///   as an embedded conditioning input — it is NOT classifier-free guidance (no uncond forward).
    ///
    /// `plan` carries the two constrained rungs (SC-15523). Everything outside the block stack —
    /// patchify, the timestep/guidance embedders, the caption projection, the output modulated norm,
    /// `proj_out` and unpatchify — is byte-for-byte the resident path, and a streamed block is built
    /// by the SAME [`SanaBlock::load`] constructor from the SAME on-disk tensors as its resident
    /// twin (SANA's tiers are packed-detected, and adapters are refused at load, so there is no
    /// per-block quantization or adapter replay that could diverge). Only *when the weights exist*
    /// differs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_memory(
        &self,
        latent_nchw: &Array,
        caption: &Array,
        timestep: &Array,
        guidance: Option<&Array>,
        caption_mask: Option<&Array>,
        plan: SanaForwardPlan,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim();
        let lsh = latent_nchw.shape();
        let (b, height, width) = (lsh[0], lsh[2], lsh[3]);
        let p = cfg.patch_size;
        let (ph, pw) = (height / p, width / p);
        let dt = latent_nchw.dtype();

        // 1. Patch embed (NHWC). [B,C,H,W] → NHWC → conv → [B,ph,pw,dim] → tokens [B,N,dim].
        let x = latent_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = self.patch_embed.forward(&x)?; // [B,ph,pw,dim]
        let mut hidden = x.reshape(&[b, ph * pw, dim])?;

        // 2. Timestep embedding → embedded_timestep [B,dim] and modulation temb [B,6·dim].
        let ts_proj = timestep_sincos(timestep, 256, 10_000.0, 0.0)?.as_dtype(dt)?; // [B,256]
        let timesteps_emb = self
            .ts_embedder_2
            .forward(&silu(&self.ts_embedder_1.forward(&ts_proj)?)?)?; // [B,dim]
                                                                       // Sprint: conditioning = timesteps_emb + guidance_emb (the guidance scalar through the same
                                                                       // sincos(256) projection + a parallel MLP). embedded_timestep (the output-modnorm input) is
                                                                       // this combined conditioning, exactly as diffusers `SanaCombinedTimestepGuidanceEmbeddings`.
        let emb =
            match (&self.guidance_embedder, guidance) {
                (Some((g1, g2)), Some(g)) => {
                    let g_proj = timestep_sincos(g, 256, 10_000.0, 0.0)?.as_dtype(dt)?;
                    let guidance_emb = g2.forward(&silu(&g1.forward(&g_proj)?)?)?;
                    add(&timesteps_emb, &guidance_emb)?
                }
                (None, None) => timesteps_emb,
                // F-092: the two remaining combinations are contract violations that previously fell into
                // the `_ => timesteps_emb` arm and SILENTLY dropped the guidance conditioning. Surface them:
                // a Sprint trunk (embedder loaded) MUST be given a guidance scalar, and a base trunk must
                // NOT (the caller mis-routed the request).
                (Some(_), None) => return Err(Error::Msg(
                    "sana: guidance_embeds trunk requires a guidance scalar, but none was supplied"
                        .into(),
                )),
                (None, Some(_)) => return Err(Error::Msg(
                    "sana: a guidance scalar was supplied but this trunk has no guidance embedder \
                     (base AdaLN-single config)"
                        .into(),
                )),
            };
        let temb = self.time_linear.forward(&silu(&emb)?)?; // [B,6·dim]

        // 3. Caption projection + RMSNorm.
        let cap = self.caption_proj_1.forward(caption)?;
        let cap = self.caption_proj_2.forward(&gelu_tanh(&cap)?)?;
        let cap = cap.reshape(&[b, -1, dim])?;
        let caption = rms_norm(&cap, &self.caption_norm, cfg.caption_norm_eps)?;

        // 4. Transformer blocks. The caption padding mask (if any) applies unchanged to every block's
        // attn2 — the per-token caption projection above preserves the M (=300) axis it indexes.
        hidden = match plan.window {
            None => {
                for block in &self.blocks {
                    hidden = block.forward(
                        &hidden,
                        &caption,
                        caption_mask,
                        &temb,
                        ph,
                        pw,
                        plan.attention,
                    )?;
                }
                hidden
            }
            Some(window) => self.run_windowed_blocks(
                hidden,
                &caption,
                caption_mask,
                &temb,
                ph,
                pw,
                plan.attention,
                window,
                cancel,
            )?,
        };

        // 5. Output: SanaModulatedNorm(embedded_timestep) → proj_out → unpatchify.
        let ss = self.scale_shift_table.reshape(&[1, 2, dim])?;
        let modg = add(&ss, &emb.reshape(&[b, 1, dim])?)?; // [B,2,dim]
        let parts = split_sections(&modg, &[1], 1)?; // 2 × [B,1,dim]
        let shift = parts[0].reshape(&[b, 1, dim])?;
        let scale = parts[1].reshape(&[b, 1, dim])?;
        // F-092: use the config eps, not a hardcoded `1e-6`. (For every shipped SANA config
        // `cfg.norm_eps == 1e-6`, so this is a no-op cleanup that removes the divergence risk.)
        let normed = layer_norm(&hidden, None, None, cfg.norm_eps)?;
        let one = scalar(1.0).as_dtype(scale.dtype())?;
        let hidden = add(&multiply(&normed, &add(&scale, &one)?)?, &shift)?;

        let out = self.proj_out.forward(&hidden)?; // [B,N, p·p·out_channels]
                                                   // unpatchify: [B,ph,pw,p,p,out_c] → permute(0,5,1,3,2,4) → [B,out_c,ph·p,pw·p].
        let oc = cfg.out_channels;
        let out = out.reshape(&[b, ph, pw, p, p, oc])?;
        let out = out.transpose_axes(&[0, 5, 1, 3, 2, 4])?;
        Ok(out.reshape(&[b, oc, ph * p, pw * p])?)
    }

    /// Rung 4: walk the 20 `transformer_blocks` in windows, rebuilding each window's weights from the
    /// snapshot, running them, and releasing them before advancing.
    ///
    /// The lifecycle — open a fresh view → apply → **force evaluation of the carried activation** →
    /// drop the view → release the allocator, with a cancellation check at every window boundary —
    /// belongs entirely to [`mlx_gen::block_residency::run_windowed`]. Nothing is re-implemented
    /// here: SC-15750 measured that hand-rolling the drop-before-eval order frees *nothing* on MLX
    /// (238.4 MiB vs 8.0 MiB at window = 1) while still producing correct images, so a family-local
    /// copy of the loop is a silent failure waiting to happen.
    #[allow(clippy::too_many_arguments)]
    fn run_windowed_blocks(
        &self,
        hidden: Array,
        caption: &Array,
        caption_mask: Option<&Array>,
        temb: &Array,
        h: i32,
        w: i32,
        budget: AttentionBudget,
        window: BlockPlan,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let stream = self.block_stream.as_ref().ok_or_else(|| {
            Error::Msg(
                "sana: bounded transformer residency was requested but this trunk has no \
                 re-openable weights source; the contract declares rung 4 unavailable for such a \
                 load"
                    .to_owned(),
            )
        })?;
        // The plan, the resident stack and the stream must all describe the same 20 blocks. A
        // disagreement would silently skip or double-run layers, so it is an error, not a clamp.
        if window.n_blocks() != self.blocks.len() || stream.n_blocks() != self.blocks.len() {
            return Err(Error::Msg(format!(
                "sana: block plan covers {} blocks and the stream {}, but the trunk has {}",
                window.n_blocks(),
                stream.n_blocks(),
                self.blocks.len()
            )));
        }
        mlx_gen::block_residency::run_windowed(
            &window,
            cancel,
            hidden,
            || stream.open(),
            |state, view, range| {
                let mut cur = state;
                for index in range {
                    let block = stream.materialize(view, index)?;
                    cur = block.forward(&cur, caption, caption_mask, temb, h, w, budget)?;
                }
                Ok(cur)
            },
            |state| Ok(mlx_rs::transforms::eval([state])?),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET;

    /// A synthetic cross-attention with the production head geometry but a small token axis, so the
    /// rung-3 kernel is exercised without weights.
    pub(super) fn cross_attn(
        heads: i32,
        inner: i32,
        qk_norm: bool,
    ) -> (CrossAttn, SanaTransformerConfig) {
        let cfg = SanaTransformerConfig {
            num_cross_attention_heads: heads,
            cross_attention_head_dim: inner / heads,
            num_attention_heads: heads,
            attention_head_dim: inner / heads,
            qk_norm,
            ..SanaTransformerConfig::sana_1600m()
        };
        let mut map = std::collections::HashMap::new();
        let fill = |n: i32, seed: f32, shape: &[i32]| {
            Array::from_slice(
                &(0..n)
                    .map(|k| (k as f32 * 0.013 + seed).sin())
                    .collect::<Vec<f32>>(),
                shape,
            )
        };
        for (i, leaf) in ["to_q", "to_k", "to_v", "to_out.0"].iter().enumerate() {
            map.insert(
                format!("attn2.{leaf}.weight"),
                fill(inner * inner, i as f32, &[inner, inner]),
            );
            map.insert(
                format!("attn2.{leaf}.bias"),
                fill(inner, i as f32, &[inner]),
            );
        }
        if qk_norm {
            for leaf in ["norm_q", "norm_k"] {
                map.insert(format!("attn2.{leaf}.weight"), fill(inner, 0.25, &[inner]));
            }
        }
        let weights = Weights::from_map(map);
        (CrossAttn::load(&weights, "attn2", &cfg).unwrap(), cfg)
    }

    pub(super) fn tokens(b: i32, n: i32, inner: i32, seed: f32) -> Array {
        Array::from_slice(
            &(0..b * n * inner)
                .map(|k| (k as f32 * 0.007 + seed).cos())
                .collect::<Vec<f32>>(),
            &[b, n, inner],
        )
    }

    /// **The rung-3 kernel is bit-exact and the lever actually engages.**
    ///
    /// The equivalence half alone is worthless — it is trivially true when the chunking never runs —
    /// so the chunk-count probe is asserted alongside it. Both `qk_norm` variants are covered because
    /// the norm is applied before the head split and therefore before any chunk boundary.
    #[test]
    fn chunked_cross_attention_is_bit_exact_and_actually_chunks() {
        for qk_norm in [false, true] {
            let (attn, _cfg) = cross_attn(4, 16, qk_norm);
            let x = tokens(1, 64, 16, 0.1);
            let kv = tokens(1, 12, 16, 0.2);
            let mask = Array::from_slice(
                &(0..12)
                    .map(|k| if k < 7 { 1.0f32 } else { 0.0 })
                    .collect::<Vec<f32>>(),
                &[1, 12],
            );
            for kv_mask in [None, Some(&mask)] {
                reset_chunk_count();
                let full = attn
                    .forward(&x, &kv, kv_mask, AttentionBudget::UNBOUNDED)
                    .unwrap();
                assert_eq!(last_chunk_count(), 1, "the unbounded path must be one call");

                // rows_per_query = B·H·Sk = 1·4·12 = 48, so a 480-score budget is 10 query rows.
                reset_chunk_count();
                let chunked = attn
                    .forward(
                        &x,
                        &kv,
                        kv_mask,
                        AttentionBudget::from_score_elements(480, true),
                    )
                    .unwrap();
                let chunks = last_chunk_count();
                assert!(
                    chunks > 1,
                    "the budget must actually chunk (qk_norm={qk_norm}), got {chunks} chunk(s)"
                );
                assert_eq!(full.shape(), chunked.shape());
                assert_eq!(
                    full.as_slice::<f32>(),
                    chunked.as_slice::<f32>(),
                    "query-row chunking must be bit-exact (qk_norm={qk_norm}, mask={})",
                    kv_mask.is_some()
                );
            }
        }
    }

    /// **The masked path must narrow nothing.** SANA's caption mask is `[B, 1, 1, M]` — broadcast
    /// over the query axis — so it is shared by every chunk. A kernel that sliced it along the query
    /// axis would silently mask the wrong rows; a kernel that dropped it would let 300 PAD slots
    /// swamp the conditioning. The bit-exactness assertion above covers the first; this pins that the
    /// mask is load-bearing at all, so "bit-exact" is not being satisfied by two identical no-ops.
    #[test]
    fn the_caption_mask_changes_the_chunked_result() {
        let (attn, _cfg) = cross_attn(4, 16, false);
        let x = tokens(1, 64, 16, 0.1);
        let kv = tokens(1, 12, 16, 0.2);
        let mask = Array::from_slice(
            &(0..12)
                .map(|k| if k < 7 { 1.0f32 } else { 0.0 })
                .collect::<Vec<f32>>(),
            &[1, 12],
        );
        let budget = AttentionBudget::from_score_elements(480, true);
        let masked = attn.forward(&x, &kv, Some(&mask), budget).unwrap();
        let unmasked = attn.forward(&x, &kv, None, budget).unwrap();
        assert_ne!(
            masked.as_slice::<f32>(),
            unmasked.as_slice::<f32>(),
            "the caption padding mask must reach the chunked kernel"
        );
    }

    /// **The sibling families' 64 Mi operating point is inert at every advertised SANA geometry.**
    ///
    /// This is the measurement that forced SANA to publish its own budgets, and it is arithmetic, so
    /// it belongs in a weights-free test rather than in a comment. `attn2`'s domain is
    /// `B · H · N · 300` with `N = (edge/32)²`.
    #[test]
    fn the_shared_64mi_budget_never_chunks_sana_and_the_published_ones_do() {
        let cfg = SanaTransformerConfig::sana_1600m();
        let heads = cfg.num_cross_attention_heads;
        let caption = 300;
        for edge in [256_i32, 512, 1024] {
            let n = (edge / 32) * (edge / 32);
            let shared = AttentionBudget::from_score_elements(CONSTRAINED_ATTN_SCORES_BUDGET, true);
            assert_eq!(
                shared.query_block(1, heads, n, caption),
                n,
                "64 Mi must not chunk SANA at {edge}²"
            );
        }
        let default = AttentionBudget::from_score_elements(
            u64::from(crate::memory_strategy::ATTENTION_CHUNK_SIZE),
            true,
        );
        // The published default chunks at 512² and at the native 1024².
        for (edge, expected_chunks) in [(512_i32, 2), (1024, 6)] {
            let n = (edge / 32) * (edge / 32);
            let block = default.query_block(1, heads, n, caption);
            assert!(
                block < n,
                "the published default must chunk at {edge}² (block {block} of {n} rows)"
            );
            assert_eq!(
                (n + block - 1) / block,
                expected_chunks,
                "chunk count at {edge}² moved"
            );
        }
        // **And it is INERT at the 256² floor** — `8·8·20·300 = 384 Ki` scores is already inside
        // every published budget. Recorded rather than papered over with a smaller budget: 1.5 MiB
        // of f32 scratch is not a thing this rung exists to bound, and a budget that chunked it
        // would be publishing an operating point with no measurement behind it.
        let floor_tokens = (256 / 32) * (256 / 32);
        assert_eq!(
            (i64::from(heads) * i64::from(floor_tokens) * i64::from(caption)) as u64,
            384_000
        );
        assert_eq!(
            default.query_block(1, heads, floor_tokens, caption),
            floor_tokens,
            "no published budget chunks the 256² floor"
        );
    }

    #[test]
    fn the_resident_plan_is_the_historical_forward() {
        assert_eq!(
            SanaForwardPlan::RESIDENT.attention,
            AttentionBudget::UNBOUNDED
        );
        assert!(SanaForwardPlan::RESIDENT.window.is_none());
        assert!(AttentionBudget::UNBOUNDED.is_unbounded());
    }
}

#[cfg(test)]
mod query_row_boundary_tests {
    use super::*;
    use crate::config::SanaTransformerConfig;

    /// **A single-query-row chunk is NOT bit-exact, and the published domain cannot reach one.**
    ///
    /// This is the boundary of rung 3's exactness claim, measured rather than asserted. Query-row
    /// chunking leaves each row's complete k/v and both reductions untouched, so the *arithmetic* is
    /// identical — but it changes the query GEMM's `M` dimension, and at `M = 1` MLX dispatches a
    /// different (gemv) kernel whose accumulation order differs. Measured on a synthetic block the
    /// divergence is ~1e-6 relative: invisible in an image, and still a numerics change rather than a
    /// memory schedule.
    ///
    /// `gen_core::attention_budget` says exactly this in the abstract — "query-row chunking changes
    /// `M` and may move results by a few ULP" — and SANA is where it was measured. The response is
    /// the epic's standard one: the DOMAIN excludes the degenerate case, and that exclusion is
    /// checked over the whole advertised size range rather than at one convenient point.
    #[test]
    fn a_single_query_row_chunk_is_not_bit_exact_and_the_domain_cannot_reach_one() {
        // The exactness boundary, on the smallest shape that exhibits it.
        let (attn, _) = super::tests::cross_attn(4, 16, false);
        let x = super::tests::tokens(1, 64, 16, 0.1);
        let kv = super::tests::tokens(1, 12, 16, 0.2);
        let full = attn
            .forward(&x, &kv, None, AttentionBudget::UNBOUNDED)
            .unwrap();
        // rows_per_query = 1·4·12 = 48.
        let one_row = attn
            .forward(
                &x,
                &kv,
                None,
                AttentionBudget::from_score_elements(48, true),
            )
            .unwrap();
        let two_rows = attn
            .forward(
                &x,
                &kv,
                None,
                AttentionBudget::from_score_elements(96, true),
            )
            .unwrap();
        assert_eq!(last_chunk_count(), 32, "96 scores must be two rows a chunk");
        assert_eq!(
            full.as_slice::<f32>(),
            two_rows.as_slice::<f32>(),
            "two query rows a chunk must stay bit-exact"
        );
        assert_ne!(
            full.as_slice::<f32>(),
            one_row.as_slice::<f32>(),
            "a single-row chunk is expected NOT to be bit-exact — if MLX ever made it exact, this \
             test is the place that records the change, and the domain guard below can relax"
        );

        // The domain guard: the tightest published budget over the whole advertised size range.
        let cfg = SanaTransformerConfig::sana_1600m();
        let heads = cfg.num_cross_attention_heads;
        let caption = 300;
        let tightest = AttentionBudget::from_score_elements(
            u64::from(crate::memory_strategy::ATTENTION_CHUNK_SIZE),
            true,
        );
        let mut narrowest = i32::MAX;
        // SANA advertises 256..=1024 on a 32-pixel stride; every one of those is a reachable
        // request, so every one of them is checked.
        let mut edge = 256;
        while edge <= 1024 {
            let n = (edge / 32) * (edge / 32);
            for budget in crate::memory_strategy::ATTENTION_CHUNK_SIZES {
                let plan = AttentionBudget::from_score_elements(u64::from(*budget), true);
                let block = plan.query_block(1, heads, n, caption);
                assert!(
                    block > 1,
                    "budget {budget} degenerates to a {block}-row chunk at {edge}sq"
                );
                if block < n {
                    narrowest = narrowest.min(block);
                }
            }
            let _ = tightest;
            edge += 32;
        }
        // Recorded, not merely bounded: the narrowest chunk any published budget produces anywhere in
        // the advertised range is 174 query rows, three orders of magnitude clear of the degenerate
        // case. A new budget that narrowed this would have to move this number.
        assert_eq!(
            narrowest, 174,
            "the narrowest published chunk width moved; re-measure the exactness boundary before \
             publishing the new budget"
        );
    }
}
