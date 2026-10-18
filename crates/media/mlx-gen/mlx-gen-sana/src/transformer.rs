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
use mlx_gen::attention::{slice_axis, AttentionBudget};
use mlx_gen::block_residency::BlockPlan;
use mlx_gen::nn::{conv2d_general, gelu_tanh, silu, timestep_sincos};
use mlx_gen::qkv::{FusedQkvProjection, QkvPart};
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
/// chunking deleted, because the claim is trivially true when the lever never engages.
///
/// [`mlx_gen::attention`] has the same probe and this is **not** a fork of convenience: that one is
/// `#[cfg(test)]`, and `cfg(test)` is set only for the crate being tested, never for its
/// dependencies. A dependent crate links `mlx-gen` compiled *without* `cfg(test)`, so the symbol
/// does not exist to import — sharing it would mean promoting it to a cargo feature, i.e. a change
/// to `mlx-gen`'s public surface for a counter that is compiled out of release entirely.
/// `slice_axis`, which carries no such constraint, IS imported from there rather than copied.
#[cfg(test)]
mod cross_attn_probe {
    use std::cell::Cell;

    // Per test thread: libtest runs tests in parallel, and a process-wide counter let one test's
    // cross-attention overwrite another's reading between its call and its assert.
    thread_local! {
        pub(super) static LAST_CHUNK_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    /// Chunks run by the most recent cross-attention call on this thread (`1` = the unchunked
    /// path).
    pub(crate) fn last_chunk_count() -> usize {
        LAST_CHUNK_COUNT.with(Cell::get)
    }

    pub(crate) fn reset() {
        LAST_CHUNK_COUNT.with(|c| c.set(0));
    }
}

#[cfg(test)]
pub(crate) use cross_attn_probe::{last_chunk_count, reset as reset_chunk_count};

#[cfg(test)]
fn record_chunk_count(n: usize) {
    cross_attn_probe::LAST_CHUNK_COUNT.with(|c| c.set(n));
}

#[cfg(not(test))]
#[inline(always)]
fn record_chunk_count(_n: usize) {}

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
        conv2d_general(
            x,
            &self.w,
            self.b.as_ref(),
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            self.groups,
        )
    }
}

// ----------------------------------------------------------------------------------------------
// ReLU linear self-attention (attn1).
// ----------------------------------------------------------------------------------------------

/// `SanaLinearAttnProcessor2_0`: ReLU linear attention over the token axis. Input/output `[B, N, C]`.
struct LinearSelfAttn {
    /// SC-18319 P4: `to_q`/`to_k`/`to_v` behind one adapter/quant-aware packed matrix. All three read
    /// the SAME activation `x` (this is a genuine self-attention), which is the precondition for
    /// packing them; `attn2` is deliberately NOT fused for exactly the opposite reason (see
    /// [`CrossAttn`]).
    ///
    /// Note this is orthogonal to SANA's entry in `qkv::EXEMPTIONS`: that exemption is about the
    /// *prologue* (ReLU linear attention has no softmax and no RoPE, so `qkv::prepare` cannot serve
    /// it), not about the projection weights, which pack on the same terms as any other family's.
    qkv: FusedQkvProjection,
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
            qkv: FusedQkvProjection::new(
                crate::quant::lin(w, &format!("{prefix}.to_q"), false)?,
                crate::quant::lin(w, &format!("{prefix}.to_k"), false)?,
                crate::quant::lin(w, &format!("{prefix}.to_v"), false)?,
            ),
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
        // The query projection's own `out` — read from the PROBE surface, so asking the geometry
        // question never dismantles the pack (SC-18319).
        let inner = self.qkv.part_facts(QkvPart::Q).base_shape[0];
        let hd = inner / self.heads;

        // One matmul when the pack is engaged, three when it is not; the packed weight is the row-wise
        // concatenation of the three bases and a matmul's output rows are independent, so the three
        // projections are bit-identical either way.
        let (q_proj, k_proj, v_proj) = self.qkv.forward(x)?;

        // qk_norm = "rms_norm_across_heads": RMSNorm over the full `inner_dim`, BEFORE the head split
        // (diffusers applies `attn.norm_q(query)` / `attn.norm_k(key)` to the `[B,N,inner]` projection).
        let q_proj = match &self.norm_q {
            Some(g) => rms_norm(&q_proj, g, self.qk_norm_eps)?,
            None => q_proj,
        };
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
        let v = to_bh_d_n(v_proj)?.as_dtype(F32)?; // [B,H,hd,N]

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

/// **Deliberately NOT fused** (SC-18319 P4). `to_q` reads the image stream `x` while `to_k`/`to_v`
/// read the projected caption `kv` — two different tensors. `FusedQkvProjection` computes all three
/// from ONE activation, so packing this triple would silently compute the keys and values from the
/// query stream: a running model with garbage cross-attention and no error.
///
/// `try_pack` cannot catch it. `caption_projection` has already mapped the caption to `inner_dim`
/// (2240), so all three `in_features` match, every part shares `bits`/`group_size`, and 2240 = 64·35
/// is group-aligned on both axes — the predicate compares *widths*, never provenance. The refusal has
/// to be structural in the code, which is what keeping the three fields separate here is.
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

    /// Expanded channels (`2 · hidden`) from the loaded tensors.
    fn ff_channels(&self) -> u64 {
        2 * self.hidden as u64
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

/// Bytes of a stored tensor's copy promoted to `activation`: its elements at the activation width
/// when it is stored narrower (MLX promotes a mixed-dtype op's narrower operand), else 0.
fn promoted_bytes(stored: &Array, activation: Dtype) -> u64 {
    promoted_elements_bytes(stored.size() as u64, stored.dtype(), activation)
}

fn dtype_width(d: Dtype) -> u64 {
    match d {
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        _ => 4,
    }
}

fn promoted_elements_bytes(elements: u64, stored: Dtype, activation: Dtype) -> u64 {
    if stored != activation && dtype_width(stored) < dtype_width(activation) {
        elements * dtype_width(activation)
    } else {
        0
    }
}

/// Block quantum from the CONFIGURATION alone (no tensors): the geometry of the iOS synthetic
/// primitive probe, whose only weight is the F16 inverted 1x1 conv `[2·hidden, dim]` — so exactly
/// that one promoted f32 copy is charged. A LOADED block is priced by `SanaBlock::quantum`, which
/// sums every promoted copy the real block makes (its three GLUMBConv convolutions and eight
/// attention projections are all live together); never price a real block from this.
pub fn configured_block_quantum(
    cfg: &SanaTransformerConfig,
    batch: i32,
    h: i32,
    w: i32,
    caption_tokens: i32,
    budget: AttentionBudget,
) -> crate::quantum::SanaBlockQuantum {
    let dim = cfg.inner_dim() as u64;
    let hidden = (cfg.mlp_ratio * cfg.inner_dim() as f32) as u64;
    let tokens = h.saturating_mul(w);
    crate::quantum::SanaBlockQuantum {
        batch: batch as u64,
        tokens: tokens as u64,
        dim,
        self_heads: cfg.num_attention_heads as u64,
        self_head_dim: cfg.attention_head_dim as u64,
        caption_tokens: caption_tokens as u64,
        cross_heads: cfg.num_cross_attention_heads as u64,
        query_rows: budget
            .query_block(batch, cfg.num_cross_attention_heads, tokens, caption_tokens)
            .min(tokens) as u64,
        ff_channels: 2 * hidden,
        qk_norm: cfg.qk_norm,
        element_bytes: 4,
        weight_bytes: 2 * hidden * dim * 4,
    }
}

impl SanaBlock {
    /// The promoted copies of attn1's q/k/v when dense and stored narrower: the packed
    /// `[3·inner, in]` matrix when the pack is engaged (one matmul), else all three projections
    /// (three matmuls in one quantum, each copy held until its matmul completes). The projections
    /// share attn1 `to_out`'s storage dtype (one checkpoint, one tier).
    fn qkv_promoted_bytes(&self, activation: Dtype) -> Option<u64> {
        let facts = self.attn1.qkv.part_facts(QkvPart::Q);
        if facts.is_quantized {
            return None;
        }
        let stored = self.attn1.to_out.dense_weight()?.0.dtype();
        let elements = |shape: &[i32]| shape.iter().map(|d| *d as u64).product::<u64>();
        let qkv = match self.attn1.qkv.packed_shape() {
            Some(shape) => elements(&shape),
            None => [QkvPart::Q, QkvPart::K, QkvPart::V]
                .into_iter()
                .map(|part| elements(&self.attn1.qkv.part_facts(part).base_shape))
                .sum(),
        };
        Some(promoted_elements_bytes(qkv, stored, activation))
    }

    /// The block quantum from THIS block's loaded tensors and the request, not production
    /// constants. `activation` is the trunk's compute dtype (the latent's).
    ///
    /// Weight copies are SUMMED over the block, not maxed. Every op of the block — attn1's q/k/v
    /// and `to_out`, attn2's four projections, the three GLUMBConv convolutions — is one graph
    /// evaluated by the quantum's single `eval`, and MLX encodes ahead of execution: a promoted
    /// copy is allocated when its cast is encoded and released only when the command buffer (Metal)
    /// or queued task (CPU) of the matmul/conv consuming it completes. Nothing inside the block
    /// synchronizes, so in op order every copy can be encoded before the first consumer retires —
    /// the SDXL retention model. Proven at a weight-dominated shape (F16 weights, f32 activations,
    /// wide `dim`, four tokens) in `quantum_proofs::trunk_quanta_track_their_measured_peaks`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantum(
        &self,
        batch: i32,
        h: i32,
        w: i32,
        caption_tokens: i32,
        budget: AttentionBudget,
        activation: Dtype,
    ) -> crate::quantum::SanaBlockQuantum {
        let dim = self.scale_shift_table.shape()[1] as u64;
        let tokens = h.saturating_mul(w);
        let self_heads = self.attn1.heads as u64;
        let linears = [
            &self.attn2.to_q,
            &self.attn2.to_k,
            &self.attn2.to_v,
            &self.attn2.to_out,
            &self.attn1.to_out,
        ];
        let weight_bytes = [
            &self.ff.conv_inverted,
            &self.ff.conv_depth,
            &self.ff.conv_point,
        ]
        .into_iter()
        .map(|conv| promoted_bytes(&conv.w, activation))
        .chain(
            linears
                .into_iter()
                .filter_map(|lin| lin.dense_weight())
                .map(|(w, _)| promoted_bytes(w, activation)),
        )
        .chain(self.qkv_promoted_bytes(activation))
        .fold(0u64, u64::saturating_add);
        let width = dtype_width(activation);
        crate::quantum::SanaBlockQuantum {
            batch: batch as u64,
            tokens: tokens as u64,
            dim,
            self_heads,
            self_head_dim: dim / self_heads.max(1),
            caption_tokens: caption_tokens as u64,
            cross_heads: self.attn2.heads as u64,
            query_rows: budget
                .query_block(batch, self.attn2.heads, tokens, caption_tokens)
                .min(tokens) as u64,
            ff_channels: self.ff.ff_channels(),
            qk_norm: self.attn1.norm_q.is_some(),
            element_bytes: width,
            weight_bytes,
        }
    }

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
        let scratch = crate::quantum::block_bytes(self.quantum(
            b,
            h,
            w,
            caption.shape()[1],
            budget,
            hidden.dtype(),
        ));
        mlx_gen::ios_quantum::run(crate::quantum::labels::BLOCK, scratch, || {
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
        })
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

    /// The promoted copies of the dense linears the entry / output quanta run, SUMMED: each quantum
    /// is one graph under one `eval`, and a copy lives until its consuming matmul's command buffer
    /// (or CPU task) completes, so with MLX's encode-ahead they can all be live together (see
    /// [`SanaBlock::quantum`]).
    fn promoted_linear_bytes(linears: &[&AdaptableLinear], activation: Dtype) -> u64 {
        linears
            .iter()
            .filter_map(|lin| lin.dense_weight())
            .map(|(w, _)| promoted_bytes(w, activation))
            .fold(0u64, u64::saturating_add)
    }

    /// The entry quantum (patchify, timestep/guidance embedding, caption projection) for this
    /// trunk's loaded tensors and the request. Excludes the previous step's lazy sampler update,
    /// which the call site adds as [`mlx_gen::ios_quantum::sampler_step_bytes`].
    pub(crate) fn entry_quantum(
        &self,
        latent_shape: &[i32],
        caption_shape: &[i32],
        caption_dtype: Dtype,
        activation: Dtype,
    ) -> crate::quantum::SanaEntryQuantum {
        let p = self.cfg.patch_size.max(1);
        let mut linears = vec![
            &self.ts_embedder_1,
            &self.ts_embedder_2,
            &self.time_linear,
            &self.caption_proj_1,
            &self.caption_proj_2,
        ];
        if let Some((g1, g2)) = &self.guidance_embedder {
            linears.push(g1);
            linears.push(g2);
        }
        let weight_bytes = Self::promoted_linear_bytes(&linears, activation)
            .saturating_add(promoted_bytes(&self.patch_embed.w, activation));
        crate::quantum::SanaEntryQuantum {
            batch: latent_shape[0] as u64,
            latent_channels: latent_shape[1] as u64,
            tokens: (latent_shape[2] / p) as u64 * (latent_shape[3] / p) as u64,
            patch: p as u64,
            dim: self.cfg.inner_dim() as u64,
            caption_tokens: caption_shape[1] as u64,
            caption_channels: caption_shape[2] as u64,
            caption_element_bytes: dtype_width(caption_dtype),
            element_bytes: dtype_width(activation),
            weight_bytes,
        }
    }

    /// The output quantum (modulated norm, `proj_out`, unpatchify).
    pub(crate) fn output_quantum(
        &self,
        batch: i32,
        tokens: i32,
        activation: Dtype,
    ) -> crate::quantum::SanaOutputQuantum {
        let p = self.cfg.patch_size as u64;
        crate::quantum::SanaOutputQuantum {
            batch: batch as u64,
            tokens: tokens as u64,
            dim: self.cfg.inner_dim() as u64,
            out_features: p * p * self.cfg.out_channels as u64,
            element_bytes: dtype_width(activation),
            weight_bytes: Self::promoted_linear_bytes(&[&self.proj_out], activation),
        }
    }

    /// The entry quantum's body: patchify, the timestep (+ Sprint guidance) embedding and the
    /// caption projection. Returns `(hidden [B,N,dim], caption [B,M,dim], temb [B,6·dim],
    /// embedded_timestep [B,dim])`.
    pub(crate) fn entry(
        &self,
        latent_nchw: &Array,
        caption: &Array,
        timestep: &Array,
        guidance: Option<&Array>,
    ) -> Result<(Array, Array, Array, Array)> {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim();
        let lsh = latent_nchw.shape();
        let b = lsh[0];
        let p = cfg.patch_size;
        let (ph, pw) = (lsh[2] / p, lsh[3] / p);
        let dt = latent_nchw.dtype();
        // 1. Patch embed (NHWC). [B,C,H,W] → NHWC → conv → [B,ph,pw,dim] → tokens [B,N,dim].
        let x = latent_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = self.patch_embed.forward(&x)?; // [B,ph,pw,dim]
        let hidden = x.reshape(&[b, ph * pw, dim])?;

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
        Ok((hidden, caption, temb, emb))
    }

    /// The output quantum's body: `SanaModulatedNorm(embedded_timestep)` → `proj_out` →
    /// unpatchify to `[B, out_channels, H, W]`.
    pub(crate) fn output(
        &self,
        hidden: &Array,
        emb: &Array,
        b: i32,
        ph: i32,
        pw: i32,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim();
        let p = cfg.patch_size;
        let ss = self.scale_shift_table.reshape(&[1, 2, dim])?;
        let modg = add(&ss, &emb.reshape(&[b, 1, dim])?)?; // [B,2,dim]
        let parts = split_sections(&modg, &[1], 1)?; // 2 × [B,1,dim]
        let shift = parts[0].reshape(&[b, 1, dim])?;
        let scale = parts[1].reshape(&[b, 1, dim])?;
        // F-092: use the config eps, not a hardcoded `1e-6`. (For every shipped SANA config
        // `cfg.norm_eps == 1e-6`, so this is a no-op cleanup that removes the divergence risk.)
        let normed = layer_norm(hidden, None, None, cfg.norm_eps)?;
        let one = scalar(1.0).as_dtype(scale.dtype())?;
        let hidden = add(&multiply(&normed, &add(&scale, &one)?)?, &shift)?;

        let out = self.proj_out.forward(&hidden)?; // [B,N, p·p·out_channels]
                                                   // unpatchify: [B,ph,pw,p,p,out_c] → permute(0,5,1,3,2,4) → [B,out_c,ph·p,pw·p].
        let oc = cfg.out_channels;
        let out = out.reshape(&[b, ph, pw, p, p, oc])?;
        let out = out.transpose_axes(&[0, 5, 1, 3, 2, 4])?;
        let out = out.reshape(&[b, oc, ph * p, pw * p])?;
        Ok(out)
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
        let lsh = latent_nchw.shape();
        let (b, height, width) = (lsh[0], lsh[2], lsh[3]);
        let p = cfg.patch_size;
        let (ph, pw) = (height / p, width / p);
        let dt = latent_nchw.dtype();
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        // The previous step's CFG combine + sampler update is lazy and materializes here.
        mlx_gen::ios_quantum::before(
            crate::quantum::labels::ENTRY,
            crate::quantum::entry_bytes(self.entry_quantum(
                latent_nchw.shape(),
                caption.shape(),
                caption.dtype(),
                dt,
            ))
            .saturating_add(mlx_gen::ios_quantum::sampler_step_bytes(
                latent_nchw.size() as u64,
            )),
        )?;
        let (mut hidden, caption, temb, emb) =
            self.entry(latent_nchw, caption, timestep, guidance)?;
        mlx_gen::ios_quantum::finish(
            crate::quantum::labels::ENTRY,
            &[&hidden, &caption, &temb, &emb],
        )?;

        // 4. Transformer blocks. The caption padding mask (if any) applies unchanged to every block's
        // attn2 — the per-token caption projection above preserves the M (=300) axis it indexes.
        hidden = match plan.window {
            None => {
                for block in &self.blocks {
                    // Observe cancellation before constructing each subsequent block quantum.
                    if cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
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
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        mlx_gen::ios_quantum::before(
            crate::quantum::labels::OUTPUT,
            crate::quantum::output_bytes(self.output_quantum(b, ph * pw, hidden.dtype())),
        )?;
        let out = self.output(&hidden, &emb, b, ph, pw)?;
        mlx_gen::ios_quantum::finish(crate::quantum::labels::OUTPUT, &[&out])?;
        Ok(out)
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
                    if cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
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

    /// sc-16780: the quantum floor's FFN geometry comes from the loaded GLUMBConv, here at a
    /// non-production `mlp_ratio = 3` where a fixed `5 * dim` expansion would be wrong.
    #[test]
    fn glumbconv_quantum_geometry_uses_loaded_nondefault_expansion() {
        let cfg = SanaTransformerConfig {
            num_attention_heads: 2,
            attention_head_dim: 8,
            mlp_ratio: 3.0,
            ..SanaTransformerConfig::sana_1600m()
        };
        let inner = cfg.inner_dim();
        let hidden = 48;
        let mut map = std::collections::HashMap::new();
        let zeros = |shape: &[i32]| Array::zeros::<f32>(shape).unwrap();
        map.insert(
            "ff.conv_inverted.weight".to_string(),
            zeros(&[2 * hidden, inner, 1, 1]),
        );
        map.insert("ff.conv_inverted.bias".to_string(), zeros(&[2 * hidden]));
        map.insert(
            "ff.conv_depth.weight".to_string(),
            zeros(&[2 * hidden, 1, 3, 3]),
        );
        map.insert("ff.conv_depth.bias".to_string(), zeros(&[2 * hidden]));
        map.insert(
            "ff.conv_point.weight".to_string(),
            zeros(&[inner, hidden, 1, 1]),
        );
        let ff = GluMbConv::load(&Weights::from_map(map), "ff", &cfg).unwrap();
        let channels = ff.ff_channels();
        assert_eq!(channels, 96);
        assert_ne!(channels, 5 * inner as u64);
        // The tensor-free configured quantum (rung-4 fallback, iOS primitive probe) prices the
        // configured expansion and charges the promoted F16 inverted conv `[2·hidden, dim]` f32.
        let configured = configured_block_quantum(&cfg, 1, 8, 8, 300, AttentionBudget::UNBOUNDED);
        assert_eq!(configured.ff_channels, channels);
        assert_eq!(configured.weight_bytes, channels * inner as u64 * 4);
        assert_eq!(
            (configured.dim, configured.tokens, configured.caption_tokens),
            (16, 64, 300)
        );
        assert_eq!(configured.cross_heads, cfg.num_cross_attention_heads as u64);
        assert_eq!(configured.query_rows, 64);
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

        // The domain guard: EVERY published budget over the whole advertised size range — not just
        // the tightest, because the narrowest chunk is not necessarily produced by the smallest
        // budget once the token count varies.
        let cfg = SanaTransformerConfig::sana_1600m();
        let heads = cfg.num_cross_attention_heads;
        let caption = 300;
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

/// Measured-peak proofs of the counted trunk quanta ([`crate::quantum::block_bytes`],
/// [`crate::quantum::entry_bytes`], [`crate::quantum::output_bytes`]) on the MLX CPU device with
/// Metal trapped, in a child process: REAL modules built from F16 weights (the shipped storage) at
/// f32 activations (the latent's dtype), at small shapes where activations dominate weights, one
/// shape per phase that can dominate (self-attention, cross-attention unchunked and rung-3 chunked,
/// Mix-FFN), base and Sprint (qk-norm).
#[cfg(all(test, target_os = "macos"))]
pub(crate) mod quantum_proofs {
    use super::*;
    use crate::quantum::MAX_RATIO;
    use mlx_gen::ios_quantum::{assert_tracks_measured, measured_peak_bytes};
    use std::collections::HashMap;

    fn rand(shape: &[i32], seed: u64) -> Array {
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
            .multiply(scalar(0.05))
            .unwrap()
    }

    pub(crate) struct Builder {
        pub(crate) map: HashMap<String, Array>,
        pub(crate) seed: u64,
        pub(crate) dtype: Dtype,
    }
    impl Builder {
        fn put(&mut self, key: String, shape: &[i32]) {
            self.seed += 1;
            let a = rand(shape, self.seed).as_dtype(self.dtype).unwrap();
            a.eval().unwrap();
            self.map.insert(key, a);
        }
        fn lin(&mut self, key: &str, out: i32, inp: i32, bias: bool) {
            self.put(format!("{key}.weight"), &[out, inp]);
            if bias {
                self.put(format!("{key}.bias"), &[out]);
            }
        }
        pub(crate) fn block(&mut self, p: &str, cfg: &SanaTransformerConfig) {
            let d = cfg.inner_dim();
            let hidden = (cfg.mlp_ratio * d as f32) as i32;
            self.put(format!("{p}.scale_shift_table"), &[6, d]);
            for leaf in ["to_q", "to_k", "to_v"] {
                self.lin(&format!("{p}.attn1.{leaf}"), d, d, false);
                self.lin(&format!("{p}.attn2.{leaf}"), d, d, true);
            }
            self.lin(&format!("{p}.attn1.to_out.0"), d, d, true);
            self.lin(&format!("{p}.attn2.to_out.0"), d, d, true);
            if cfg.qk_norm {
                for a in ["attn1", "attn2"] {
                    self.put(format!("{p}.{a}.norm_q.weight"), &[d]);
                    self.put(format!("{p}.{a}.norm_k.weight"), &[d]);
                }
            }
            self.put(
                format!("{p}.ff.conv_inverted.weight"),
                &[2 * hidden, d, 1, 1],
            );
            self.put(format!("{p}.ff.conv_inverted.bias"), &[2 * hidden]);
            self.put(format!("{p}.ff.conv_depth.weight"), &[2 * hidden, 1, 3, 3]);
            self.put(format!("{p}.ff.conv_depth.bias"), &[2 * hidden]);
            self.put(format!("{p}.ff.conv_point.weight"), &[d, hidden, 1, 1]);
        }
        pub(crate) fn trunk(&mut self, cfg: &SanaTransformerConfig) {
            let d = cfg.inner_dim();
            let p = cfg.patch_size;
            self.put(
                "patch_embed.proj.weight".into(),
                &[d, cfg.in_channels, p, p],
            );
            self.put("patch_embed.proj.bias".into(), &[d]);
            let ts = if cfg.guidance_embeds {
                self.lin("time_embed.guidance_embedder.linear_1", d, 256, true);
                self.lin("time_embed.guidance_embedder.linear_2", d, d, true);
                "time_embed.timestep_embedder"
            } else {
                "time_embed.emb.timestep_embedder"
            };
            self.lin(&format!("{ts}.linear_1"), d, 256, true);
            self.lin(&format!("{ts}.linear_2"), d, d, true);
            self.lin("time_embed.linear", 6 * d, d, true);
            self.lin("caption_projection.linear_1", d, cfg.caption_channels, true);
            self.lin("caption_projection.linear_2", d, d, true);
            self.put("caption_norm.weight".into(), &[d]);
            self.put("scale_shift_table".into(), &[2, d]);
            self.lin("proj_out", p * p * cfg.out_channels, d, true);
        }
        pub(crate) fn weights(&self) -> Weights {
            Weights::from_map(self.map.clone())
        }
    }

    pub(crate) fn cfg(heads: i32, hd: i32, mlp_ratio: f32, qk_norm: bool) -> SanaTransformerConfig {
        SanaTransformerConfig {
            num_attention_heads: heads,
            attention_head_dim: hd,
            num_cross_attention_heads: heads,
            cross_attention_head_dim: hd,
            num_layers: 0,
            mlp_ratio,
            caption_channels: 96,
            guidance_embeds: qk_norm,
            qk_norm,
            ..SanaTransformerConfig::sana_1600m()
        }
    }

    /// Runs per measurement; the proofs assert against the WORST (largest) peak. MLX's CPU executor
    /// releases buffers asynchronously, so one run's peak varies; never throttled (no memory limit).
    pub(crate) const RUNS: usize = 5;

    /// The worst measured peak of [`RUNS`] unthrottled runs of `run`.
    pub(crate) fn worst_peak(mut run: impl FnMut() -> Result<Vec<Array>>) -> u64 {
        (0..RUNS)
            .map(|_| measured_peak_bytes(&mut run).unwrap())
            .max()
            .unwrap()
    }

    /// Print every proof's ratio, then assert them all (so one run reports every site).
    fn ratio(proofs: &mut Vec<(String, u64, u64)>, label: &str, formula: u64, measured: u64) {
        println!(
            "[{label}] formula {formula} measured {measured} ratio {:.3}",
            formula as f64 / measured as f64
        );
        proofs.push((label.to_string(), formula, measured));
    }

    #[test]
    fn trunk_quanta_track_their_measured_peaks() {
        const CHILD: &str = "SANA_TRUNK_QUANTUM_PROOF_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "transformer::quantum_proofs::trunk_quanta_track_their_measured_peaks",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        // Two 512-row query chunks: the chunked scores still dominate a narrow-FFN block.
        let chunked = AttentionBudget::from_score_elements(4 * 512 * 300, true);
        // (label, mlp_ratio, caption tokens, budget): which phase dominates.
        let mut proofs = Vec::new();
        let mut superseded = Vec::new();
        // (qk_norm, activation-dominated, caption dtype, formula, measured) of every entry proof.
        let mut entries = Vec::new();
        // (label, heads, head_dim, mlp_ratio, side, caption tokens, budget). The last case is
        // WEIGHT-dominated: F16 weights at dim 512 against f32 activations over 2x2 = 4 tokens, so
        // the block's promoted weight copies (~13 MB) dwarf every activation (~8 KB each) and the
        // proof sees whether they are live together (summed) or one at a time.
        let cases = [
            ("cross", 4, 16, 2.5, 32, 300, AttentionBudget::UNBOUNDED),
            ("cross rung-3", 4, 16, 0.25, 32, 300, chunked),
            ("ffn", 4, 16, 2.5, 32, 8, AttentionBudget::UNBOUNDED),
            (
                "self-attention",
                4,
                16,
                0.25,
                32,
                4,
                AttentionBudget::UNBOUNDED,
            ),
            (
                "weight copies",
                8,
                64,
                2.5,
                2,
                4,
                AttentionBudget::UNBOUNDED,
            ),
        ];
        for qk_norm in [false, true] {
            for (name, heads, head_dim, mlp_ratio, side, caption_tokens, budget) in cases {
                let cfg = cfg(heads, head_dim, mlp_ratio, qk_norm);
                let d = cfg.inner_dim();
                let mut b = Builder {
                    map: HashMap::new(),
                    seed: 900,
                    dtype: Dtype::Float16,
                };
                b.block("blk", &cfg);
                let block = SanaBlock::load(&b.weights(), "blk", &cfg).unwrap();
                let (h, w) = (side, side);
                let hidden = rand(&[1, h * w, d], 1);
                let caption = rand(&[1, caption_tokens, d], 2);
                let mask = Array::ones::<f32>(&[1, caption_tokens]).unwrap();
                let temb = rand(&[1, 6 * d], 3);
                mlx_rs::transforms::eval([&hidden, &caption, &mask, &temb]).unwrap();
                let q = block.quantum(1, h, w, caption_tokens, budget, Dtype::Float32);
                if name == "cross rung-3" {
                    assert!(q.query_rows < q.tokens, "the rung-3 case must chunk: {q:?}");
                }
                let measured = worst_peak(|| {
                    Ok(vec![block.forward(
                        &hidden,
                        &caption,
                        Some(&mask),
                        &temb,
                        h,
                        w,
                        budget,
                    )?])
                });
                ratio(
                    &mut proofs,
                    &format!("SANA block {name} qk_norm={qk_norm}"),
                    crate::quantum::block_bytes(q),
                    measured,
                );
                if name == "weight copies" {
                    // The superseded pricing (only the largest single copy) under-prices this
                    // shape: more than one copy is live at once.
                    let largest = [
                        &block.ff.conv_inverted.w,
                        &block.ff.conv_depth.w,
                        &block.ff.conv_point.w,
                    ]
                    .into_iter()
                    .map(|w| promoted_bytes(w, Dtype::Float32))
                    .max()
                    .unwrap();
                    superseded.push((
                        format!("SANA block {name} qk_norm={qk_norm}"),
                        crate::quantum::block_bytes(crate::quantum::SanaBlockQuantum {
                            weight_bytes: largest,
                            ..q
                        }),
                        measured,
                    ));
                }
            }

            // Entry and output quanta of a whole (blockless) trunk, at a bf16 caption (Gemma's
            // dtype) and an f32 one; activation-dominated (dim 64, 32x32 latent, 300 caption
            // tokens) and weight-dominated (dim 512, 2x2 latent, 4 caption tokens: the entry's
            // promoted projection copies, ~7 MB, dwarf its activations).
            for (scale, heads, head_dim, side, caption_tokens) in
                [("", 4, 16, 32, 300), (" weight copies", 8, 64, 2, 4)]
            {
                let cfg = cfg(heads, head_dim, 2.5, qk_norm);
                let mut b = Builder {
                    map: HashMap::new(),
                    seed: 1900,
                    dtype: Dtype::Float16,
                };
                b.trunk(&cfg);
                let trunk = SanaTransformer::from_weights(&b.weights(), cfg.clone()).unwrap();
                for caption_dtype in [Dtype::Bfloat16, Dtype::Float32] {
                    let latent = rand(&[1, cfg.in_channels, side, side], 4);
                    let caption = rand(&[1, caption_tokens, cfg.caption_channels], 5)
                        .as_dtype(caption_dtype)
                        .unwrap();
                    let t = Array::from_slice(&[500.0f32], &[1]);
                    let g = Array::from_slice(&[0.45f32], &[1]);
                    mlx_rs::transforms::eval([&latent, &caption, &t, &g]).unwrap();
                    let guidance = qk_norm.then_some(&g);
                    let measured = worst_peak(|| {
                        let (a, b, c, d) = trunk.entry(&latent, &caption, &t, guidance)?;
                        Ok(vec![a, b, c, d])
                    });
                    let entry = trunk.entry_quantum(
                        latent.shape(),
                        caption.shape(),
                        caption_dtype,
                        Dtype::Float32,
                    );
                    let label =
                        format!("SANA entry{scale} qk_norm={qk_norm} caption={caption_dtype:?}");
                    entries.push((
                        qk_norm,
                        scale.is_empty(),
                        caption_dtype,
                        crate::quantum::entry_bytes(entry),
                        measured,
                    ));
                    ratio(
                        &mut proofs,
                        &label,
                        crate::quantum::entry_bytes(entry),
                        measured,
                    );
                    if !scale.is_empty() {
                        let mut linears = vec![
                            &trunk.ts_embedder_1,
                            &trunk.ts_embedder_2,
                            &trunk.time_linear,
                            &trunk.caption_proj_1,
                            &trunk.caption_proj_2,
                        ];
                        if let Some((g1, g2)) = &trunk.guidance_embedder {
                            linears.push(g1);
                            linears.push(g2);
                        }
                        let largest = linears
                            .into_iter()
                            .map(|lin| {
                                promoted_bytes(lin.dense_weight().unwrap().0, Dtype::Float32)
                            })
                            .chain([promoted_bytes(&trunk.patch_embed.w, Dtype::Float32)])
                            .max()
                            .unwrap();
                        superseded.push((
                            label,
                            crate::quantum::entry_bytes(crate::quantum::SanaEntryQuantum {
                                weight_bytes: largest,
                                ..entry
                            }),
                            measured,
                        ));
                    }
                    let (hidden, _, _, emb) = trunk.entry(&latent, &caption, &t, guidance).unwrap();
                    mlx_rs::transforms::eval([&hidden, &emb]).unwrap();
                    let measured =
                        worst_peak(|| Ok(vec![trunk.output(&hidden, &emb, 1, side, side)?]));
                    ratio(
                        &mut proofs,
                        &format!("SANA output{scale} qk_norm={qk_norm}"),
                        crate::quantum::output_bytes(trunk.output_quantum(
                            1,
                            side * side,
                            Dtype::Float32,
                        )),
                        measured,
                    );
                }
            }
        }
        // The bf16 caption's f32 widening copy is released before gelu's chain (the caption term's
        // peak): at the chain-dominated shape a bf16 and an f32 caption peak alike, so their
        // formulas must not differ by more than the measurement does (a page of slack). Pricing
        // the copy on top of the chain over-counts it (CI: 1.524x).
        for &(qk, dominated, dtype, formula, measured) in &entries {
            if !dominated || dtype != Dtype::Bfloat16 {
                continue;
            }
            let &(_, _, _, f32_formula, f32_measured) = entries
                .iter()
                .find(|e| e.0 == qk && e.1 && e.2 == Dtype::Float32)
                .unwrap();
            assert!(
                formula.abs_diff(f32_formula) <= measured.abs_diff(f32_measured) + 16384,
                "qk_norm={qk}: the bf16 caption is priced {formula} vs f32 {f32_formula}, but they \
                 measure {measured} vs {f32_measured}"
            );
        }
        // Every proof against the same report threshold: the weight-dominated block's ratio
        // (~1.6-1.8: the CPU's 11-task window retires the attention projections' copies the
        // device still holds) is reported, not excused.
        for (label, formula, measured) in proofs {
            let _ = assert_tracks_measured(&label, formula, measured, MAX_RATIO);
        }
        // Discrimination: pricing only the largest copy (the pre-fix model) is BELOW the measured
        // peak at the weight-dominated shapes.
        assert_eq!(superseded.len(), 6);
        for (label, single, measured) in superseded {
            println!("[{label}] largest-copy-only formula {single} measured {measured}");
            assert!(
                single < measured,
                "{label}: the largest-copy-only pricing ({single} B) must under-price the \
                 measured peak ({measured} B), or this shape cannot tell the models apart"
            );
        }
    }
}
