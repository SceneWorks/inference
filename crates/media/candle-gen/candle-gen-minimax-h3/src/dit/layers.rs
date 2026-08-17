//! The two leaf modules every DiT block and every token-refiner block is built from.
//!
//! Both are **bias-free** — `MiniMaxH3Attention` constructs its four projections with
//! `bias=False` and the block's `FeedForward(..., bias=False)` does the same, which the published
//! tensor index corroborates: no `.bias` exists under `attn.` or `ff.` anywhere in
//! `transformer/`. Only `adaln_proj.linear`, the input/output projections and the timestep MLP
//! carry biases.
//!
//! # The attention transient is a genuine cross-backend difference (sc-17152)
//!
//! MLX's fused `scaled_dot_product_attention` **streams the scores**: sc-17152 measured its peak
//! tracking `4·B·H·S·D` exactly, i.e. there is no `[B, H, Sq, Sk]` tensor in memory at all. candle
//! has no fused kernel here, so [`DitAttention::forward`] routes through
//! `candle_gen::sdpa_budgeted_bhsd`, which **does** materialize scores — in bounded row blocks
//! rather than all at once (`ATTN_SCORES_BUDGET`, the shared i32-overflow-safe helper from
//! sc-9116).
//!
//! The consequence is stated rather than glossed: at the shipped 1344×768 canvas the 345-frame
//! lattice ceiling (14.375 s — 15 s has no legal frame count) packs ~104k rows — 102 latent frames
//! × 42×24 patches = 102,816 video rows, plus 1,150 audio and the text rows — over 56 heads, where
//! the full score tensor would be `56 · (1.04e5)²` ≈ **6.1e11 elements**, ~280× past `i32::MAX`
//! and far past any card — so the budgeted split is **not optional on this lane** the way MLX's
//! streaming kernel makes it optional there. sc-17152 additionally measured chunked SDPA on MLX at
//! **+50.3 % peak and ~3× wall**, so its conclusion ("do not chunk") is an MLX result and must not
//! be carried across: the two backends have different attention memory shapes, and this lane's
//! numbers are sc-17156's to measure on real hardware.
//!
//! sc-17152 also probed `matmul` correctness above `i32::MAX` **on MLX** and found it exact. That
//! is an MLX result. Nothing here assumes it transfers to candle/CUDA, and the widest single tensor
//! this module writes is the SwiGLU projection `[1, S, 2·ffn_dim]` — `[1, S, 28672]` at the shipped
//! geometry — not anything in the attention.
//!
//! # The SwiGLU intermediate is chunked for the same i32 reason as the attention
//!
//! That ceiling is now **guarded, not merely observed**. At the same ~104k rows the paragraph above
//! derives (the 345-frame lattice ceiling on the 1344×768 canvas), `[1, 104000, 28672]` is ~2.98e9
//! elements — past `i32::MAX` (2.147e9) in a single unchunked GEMM, which is exactly the corruption
//! class `candle_gen::ATTN_SCORES_BUDGET` exists to prevent on the score tensor. Unlike the
//! attention's ~6.1e11, this one is only ~1.4× over the limit: a **silent-corruption** ceiling
//! rather than an out-of-memory one, which is why it went unguarded.
//! [`DitFeedForward::forward`] therefore runs the projection over
//! blocks of **token rows** through the same shared planner the attention path uses
//! ([`candle_gen::attention::AttentionBudget::query_block_rows`], sc-15796) at the same 1e9 setting
//! ([`FFN_INTERMEDIATE_BUDGET`]), so the two guards cannot plan different boundaries from the same
//! declared budget.
//!
//! Every token's SwiGLU is independent of every other token's — no reduction crosses the token axis
//! — so the split is mathematically exact. It is **not** bitwise exact: narrowing the token axis
//! changes the GEMM's `M`, and candle's CPU `gemm` and cuBLAS may accumulate in a different order at
//! a different `M`. `chunked_swiglu_matches_the_unchunked_projection` gates that on **relative
//! max-abs**, the crate's established measure, and not on cosine — cosine cannot see a scale error.
//! Do not build an exact-equality assertion on this path (sc-15943 is the sibling defect).
//!
//! Geometries whose intermediate already fits stay on the single un-chunked GEMM, byte-identical to
//! the pre-guard forward.

use candle_gen::attention::AttentionBudget;
use candle_gen::candle_core::{DType, Tensor};
use candle_gen::quant::AdaptLinear;
use candle_gen::{CandleError, Result, Weights};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::layout::split_gate_value;
use crate::nn::{rms_weighted, sdpa, silu};

/// A **forward-time additive LoRA residual** attached to a [`LinearNoBias`] (sc-18728).
///
/// `scale · ((x·a)·b)` with `a` `[in, rank]` (= `downᵀ`) and `b` `[rank, out]` (= `upᵀ` with
/// `alpha/rank` already folded in at `b`'s own dtype). Deliberately the two-small-matmul form and
/// never the `[out, in]` product, so the residual costs `rank·(in+out)` elements rather than a
/// second copy of the projection — the property that lets the same install ride a packed tier
/// unchanged when one lands.
///
/// **The base weight is never mutated.** That is the candle twin of the MLX lane's decision
/// (`mlx_gen_minimax_h3::adapters` installs over `AdaptableLinear` and never merges), and it is what
/// makes the fold strength independent of the quant tier: a merge would have to dequantize.
///
/// # This is a RECORD of what was installed, not the mechanism (sc-20267)
///
/// The residual arithmetic now lives in the shared [`candle_gen::quant::AdaptLinear`], which
/// [`LinearNoBias`] wraps so a LoRA composes over a **packed** base as well as a dense one. That
/// seam holds its adapters privately and exposes no per-residual accessor, so the same factors are
/// retained here — `Tensor` is `Arc`-backed, so the retention costs pointers, not weights — to keep
/// the installed dtype / shapes / strength inspectable. `tests/turbo_lora.rs` reads exactly those
/// three, and they are the only thing that can see a stray widening cast in the install.
#[derive(Debug, Clone)]
pub struct LoraResidual {
    /// `[in, rank]` — `downᵀ`, at the factor's loaded dtype.
    a: Tensor,
    /// `[rank, out]` — `upᵀ` scaled by `alpha/rank`, at the factor's loaded dtype.
    b: Tensor,
    /// The caller's per-adapter strength (`AdapterSpec::scale`), applied at the residual.
    scale: f64,
}

impl LoraResidual {
    /// `a`'s dtype — the loaded factor dtype. Read by
    /// `adapters::tests::the_installed_fold_keeps_the_factor_dtype`, which is the **only** thing
    /// that can see a stray `to_dtype` in the install: every published turbo fold is an exact power
    /// of two, so bf16 and f32 products are bit-identical and no numeric assertion can distinguish
    /// them.
    pub fn dtype(&self) -> DType {
        self.b.dtype()
    }

    /// The `(in, rank)` of the down factor and the `(rank, out)` of the up factor, for a guard that
    /// wants to check the installed shapes without reaching into the tensors.
    pub fn shapes(&self) -> ((usize, usize), (usize, usize)) {
        let a = self.a.dims();
        let b = self.b.dims();
        ((a[0], a[1]), (b[0], b[1]))
    }

    /// The per-adapter strength this residual folds at.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// `y = x · Wᵀ` for a stored `[out, in]` weight — an `nn.Linear(..., bias=False)` — plus any
/// [`LoraResidual`] stacked onto it.
///
/// # The base is the shared [`AdaptLinear`], so it may be DENSE **or PACKED** (sc-20267)
///
/// This is one of exactly two loaders the MLX tiers pack (the other is
/// [`crate::dit::block::AdaLnProjection`]); [`crate::quant::lin`] detects the `{prefix}.scales`
/// sibling and builds the quantized base straight from the MLX triple with no dense transient. The
/// residual mechanism is the shared `candle_gen::quant::adapt` seam rather than a third hand-rolled
/// copy, which is what makes a LoRA compose over a packed base: the base weight is never mutated,
/// so there is nothing to dequantize.
///
/// The forward is [`AdaptLinear::forward_upcast`], not `forward`: the base is stored at the
/// loader's dtype and must be upcast to the activation's per call — the same contract
/// [`crate::nn::linear_nb`] held. It is inert (an `Arc` clone, no copy) when the two already match,
/// which is every shipped path.
#[derive(Debug, Clone)]
pub struct LinearNoBias {
    inner: AdaptLinear,
    /// Device bytes the frozen base holds, recorded at load — a packed base has no dense weight to
    /// measure afterwards. See [`crate::quant::TieredLinear`].
    base_bytes: usize,
    /// Record of the forward-time residuals installed on `inner`, in push order, for the
    /// dtype/shape/strength introspection the shared seam does not expose. Empty on every
    /// un-adapted render, in which case [`Self::forward`] is byte-identical to the bare base.
    adapters: Vec<LoraResidual>,
}

impl LinearNoBias {
    /// Load `{prefix}` — **packed** when `{prefix}.scales` is present, else dense at `dtype`.
    pub fn from_weights(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        let loaded = crate::quant::lin(w, prefix, false, dtype)?;
        Ok(Self {
            inner: loaded.linear,
            base_bytes: loaded.base_bytes,
            adapters: Vec::new(),
        })
    }

    /// `y = x · Wᵀ` over the last axis of `x`, plus every attached residual.
    ///
    /// The residual runs at `x`'s dtype — matching the base, which is upcast the same way — and is
    /// narrowed to the host output dtype before the add, exactly as MLX's
    /// `AdaptableLinear::apply_adapters` does. A zero-scale residual is skipped entirely, so a
    /// disabled adapter is byte-identical to the bare base.
    ///
    /// # Why the activation is flattened here rather than handed straight to [`AdaptLinear`]
    ///
    /// This preserves [`crate::nn::linear_nb`]'s rank contract exactly, which the shared seam does
    /// not have: [`AdaptLinear`]'s dense base is a `candle_nn::Linear`, whose forward special-cases
    /// ranks 2–4 and falls through to a same-rank `matmul` otherwise — so a **rank-1** activation
    /// (`[in]`) fails with `shape mismatch in matmul`, and `linear_nb` accepted it.
    /// `DitFeedForward::forward_budgeted` has an explicit rank-<2 arm, so that is a reachable input
    /// on this crate's own seam, not a hypothetical.
    ///
    /// Folding every leading dim into the GEMM's `M` is additionally the *same* single large-`M`
    /// GEMM `linear_nb` issued, rather than the batched broadcast `Linear` would run — so the
    /// un-adapted forward keeps its previous shape as well as its previous result, and the residual
    /// legs see the flattened 2-D activation the hand-rolled `apply_factor` used to build for them.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let in_features = *dims.last().expect("LinearNoBias::forward: x has no axes");
        let rows = x.elem_count() / in_features;
        let y = self
            .inner
            .forward_upcast(&x.reshape((rows, in_features))?)?;
        let mut out_dims = dims;
        *out_dims
            .last_mut()
            .expect("LinearNoBias::forward: x has no axes") = self.inner.out_features();
        Ok(y.reshape(out_dims)?)
    }

    /// Attach a forward-time LoRA residual. `a` is `[in, rank]`, `b` is `[rank, out]` with
    /// `alpha/rank` already folded in. Multiple pushes stack in order.
    ///
    /// Shape-checked against the projection's own `[out, in]`: a factor pair that would broadcast
    /// into the wrong contraction is refused here rather than producing a runnable, wrong model.
    /// The shape is read off the [`AdaptLinear`], which recovers it from the packed `scales` on a
    /// quantized base — so the check holds on every tier, not just the dense one.
    pub fn push_lora(&mut self, a: Tensor, b: Tensor, scale: f64) -> Result<()> {
        let (out, inn) = self.inner.base_shape();
        if a.rank() != 2 || b.rank() != 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adapter: LoRA factors must be 2-D, got a={:?} b={:?}",
                a.dims(),
                b.dims()
            )));
        }
        let (a0, a1) = (a.dim(0)?, a.dim(1)?);
        let (b0, b1) = (b.dim(0)?, b.dim(1)?);
        if a0 != inn || b1 != out || a1 != b0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adapter: factors a={:?} b={:?} do not compose onto a [{out}, {inn}] \
                 projection — expected a [{inn}, rank] and b [rank, {out}]",
                a.dims(),
                b.dims()
            )));
        }
        self.inner.push_lora(a.clone(), b.clone(), scale);
        self.adapters.push(LoraResidual { a, b, scale });
        Ok(())
    }

    /// Drop every attached residual, reverting to the bare base.
    pub fn clear_adapters(&mut self) {
        self.inner.clear_adapters();
        self.adapters.clear();
    }

    /// The residuals attached to this projection, in push order.
    pub fn adapters(&self) -> &[LoraResidual] {
        &self.adapters
    }

    /// Whether any residual is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
    }

    /// Whether the base loaded from a **packed** (pre-quantized) tier.
    pub fn is_packed(&self) -> bool {
        self.inner.is_packed()
    }

    /// The projection's logical `(out_features, in_features)` — recoverable from a packed base,
    /// where no dense weight exists to measure.
    pub fn base_shape(&self) -> Result<(usize, usize)> {
        Ok(self.inner.base_shape())
    }

    /// Device bytes this projection holds — the frozen base plus every attached residual factor.
    pub fn nbytes(&self) -> usize {
        self.base_bytes
            + self
                .adapters
                .iter()
                .map(|ad| {
                    ad.a.elem_count() * ad.a.dtype().size_in_bytes()
                        + ad.b.elem_count() * ad.b.dtype().size_in_bytes()
                })
                .sum::<usize>()
    }

    /// The one tensor name this projection consumes.
    ///
    /// The **dense** spelling: a packed tier additionally carries `{prefix}.scales` /
    /// `{prefix}.biases` under the same base, which [`crate::quant::lin`] finds by construction.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
}

/// Full self-attention with **affine per-head q/k RMSNorm** and MM-RoPE.
///
/// # qk-norm placement and shape
///
/// `norm_q` / `norm_k` are `nn.RMSNorm(dim_head, eps=qk_norm_eps)` — normalizing over a single
/// **head's** 128 channels, not over the 7168-wide projection — and they run **after** the
/// `[B, S, H·D] → [B, S, H, D]` unflatten and **before** the rotary. All three of those choices are
/// invisible in the tensor index, which only shows a `[128]` vector:
///
/// * normalizing the flat projection instead would mix all 56 heads into one RMS;
/// * running the rotary first would rotate un-normalized vectors, changing every attention score.
///
/// The video VAE's equivalent is non-affine and leaves **no tensors at all**, which is why
/// `tests/dit_parity.rs` gates this on *behaviour* rather than on the presence of `norm_q.weight`.
#[derive(Debug, Clone)]
pub struct DitAttention {
    to_q: LinearNoBias,
    to_k: LinearNoBias,
    to_v: LinearNoBias,
    to_out: LinearNoBias,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    qk_norm_eps: f64,
}

impl DitAttention {
    /// Load `{prefix}` (e.g. `transformer_blocks.0.attn`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: DType,
    ) -> Result<Self> {
        let expect = |t: &Tensor, want: &[usize], what: &str| -> Result<()> {
            if t.dims() != want {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit {prefix}.{what}: expected {want:?}, got {:?}",
                    t.dims()
                )));
            }
            Ok(())
        };
        let norm_q = w
            .require(&format!("{prefix}.norm_q.weight"))?
            .to_dtype(dtype)?;
        let norm_k = w
            .require(&format!("{prefix}.norm_k.weight"))?
            .to_dtype(dtype)?;
        // A qk-norm sized to the projection rather than to a head is the plausible wrong shape,
        // and it broadcasts silently against `[B, S, H, D]` only when it is `[D]`.
        expect(&norm_q, &[cfg.attention_head_dim], "norm_q")?;
        expect(&norm_k, &[cfg.attention_head_dim], "norm_k")?;

        Ok(Self {
            to_q: LinearNoBias::from_weights(w, &format!("{prefix}.to_q"), dtype)?,
            to_k: LinearNoBias::from_weights(w, &format!("{prefix}.to_k"), dtype)?,
            to_v: LinearNoBias::from_weights(w, &format!("{prefix}.to_v"), dtype)?,
            to_out: LinearNoBias::from_weights(w, &format!("{prefix}.to_out.0"), dtype)?,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            qk_norm_eps: cfg.qk_norm_eps,
        })
    }

    /// Every tensor name this attention consumes — six.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v: Vec<String> = ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .flat_map(|p| LinearNoBias::names(&format!("{prefix}.{p}")))
            .collect();
        v.push(format!("{prefix}.norm_q.weight"));
        v.push(format!("{prefix}.norm_k.weight"));
        v
    }

    /// Resolve one adapter path segment run **relative to this attention** to its projection.
    ///
    /// The four spellings are the published diffusers ones, `to_out.0` included — that trailing `0`
    /// is an `nn.Sequential` index and therefore a real path segment, which is why the lookup takes
    /// a segment slice rather than a dotted string.
    ///
    /// `norm_q` / `norm_k` are deliberately **not** addressable: they are `[head_dim]` vectors, not
    /// projections, and nothing in the published turbo set targets them.
    pub fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut LinearNoBias> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out", "0"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    /// Drop every residual on all four projections.
    pub fn clear_adapters(&mut self) {
        self.to_q.clear_adapters();
        self.to_k.clear_adapters();
        self.to_v.clear_adapters();
        self.to_out.clear_adapters();
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.to_q.nbytes()
            + self.to_k.nbytes()
            + self.to_v.nbytes()
            + self.to_out.nbytes()
            + self.norm_q.elem_count() * self.norm_q.dtype().size_in_bytes()
            + self.norm_k.elem_count() * self.norm_k.dtype().size_in_bytes()
    }

    /// `rope` is `None` for the token refiner, which runs **without** any positional embedding.
    pub fn forward(&self, x: &Tensor, rope: Option<(&MmRope, &MmRopeTables)>) -> Result<Tensor> {
        let s = x.dims();
        if s.len() != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit attention: expected [B, S, hidden], got {s:?}"
            )));
        }
        let (b, seq) = (s[0], s[1]);
        let (h, d) = (self.heads, self.head_dim);

        // The published checkpoint ships `to_q`/`to_k`/`to_v` already split, so this applies NO
        // fused-QKV transform. See `crate::dit::qkv` for which transform produced them and why
        // that must be asserted rather than assumed.
        let q = self.to_q.forward(x)?.reshape((b, seq, h, d))?;
        let k = self.to_k.forward(x)?.reshape((b, seq, h, d))?;
        let v = self.to_v.forward(x)?.reshape((b, seq, h, d))?;

        // qk-norm first, rotary second.
        let q = rms_weighted(&q, &self.norm_q, self.qk_norm_eps)?;
        let k = rms_weighted(&k, &self.norm_k, self.qk_norm_eps)?;
        let (q, k) = match rope {
            Some((rope, tables)) => (rope.apply(&q, tables)?, rope.apply(&k, tables)?),
            None => (q, k),
        };

        // MiniMax-H3 packs one request into a single attention document: no mask, not causal.
        let qh = q.transpose(1, 2)?.contiguous()?;
        let kh = k.transpose(1, 2)?.contiguous()?;
        let vh = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (d as f64).sqrt();
        let out = sdpa(&qh, &kh, &vh, scale)?;

        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, h * d))?;
        self.to_out.forward(&out)
    }
}

/// Max elements in one SwiGLU intermediate `[.., block, 2·ffn_dim]` before the token axis is
/// chunked.
///
/// The **same 1e9 setting** as `candle_gen::ATTN_SCORES_BUDGET`, for the same reason: candle CUDA
/// kernels index elements with i32, so a single tensor over `i32::MAX` (~2.147e9) silently corrupts
/// its tail. 1e9 keeps each block well under that while leaving every geometry whose full
/// intermediate is already `≤ 1e9` a single un-chunked GEMM — byte-identical to the pre-guard path.
///
/// This is a **correctness guard**, not a memory operating point; it is deliberately much larger
/// than the memory rung's `CONSTRAINED_ATTN_SCORES_BUDGET` and is not published as a strategy
/// parameter.
pub const FFN_INTERMEDIATE_BUDGET: u64 = 1_000_000_000;

/// The planned token-block length for a SwiGLU intermediate of `leading × seq × width` elements —
/// `seq` itself when the whole intermediate already fits, otherwise a value in `1..seq`.
///
/// `leading` is the product of the dims **before** the token axis (`B`, i.e. 1 on every shipped
/// path) and `width` is the projection's `out_features` (`2 · ffn_dim`), so `leading · width` is the
/// element count contributed by ONE token row — the shared planner's `rows_per_query`.
///
/// **Pure delegation to [`AttentionBudget::query_block_rows`].** This must not apply arithmetic of
/// its own, for the same reason `candle_gen::attention::query_block` must not: two guards computing
/// their own boundaries from one declared budget is the divergence the shared planner exists to
/// prevent. All arithmetic is `u64`/`usize` — the element counts this plans for are precisely the
/// ones that do not fit in `i32`.
pub fn ffn_token_block(leading: usize, width: usize, seq: usize, budget: AttentionBudget) -> usize {
    let rows_per_token = (leading as u64).saturating_mul(width as u64);
    budget.query_block_rows(rows_per_token, seq as u64) as usize
}

/// Bias-free SwiGLU feed-forward: `w2( silu(gate) · value )`.
///
/// **The published `ff.net.0.proj` emits `[value | gate]`**, so the gate is the SECOND half — the
/// split is delegated to [`crate::layout::split_gate_value`], the crate's single implementation.
/// `convert_minimax_h3_to_diffusers.py` swaps the halves of the DiT's `mlp.fc1` on the way in
/// exactly as it does the video VAE's `ff.w1`, and reading the first half as the gate computes
/// `w2( silu(value) · gate )` — shape-identical, and the sc-18740 defect.
#[derive(Debug, Clone)]
pub struct DitFeedForward {
    proj: LinearNoBias,
    out: LinearNoBias,
}

impl DitFeedForward {
    /// Load `{prefix}` (e.g. `transformer_blocks.0.ff`).
    pub fn from_weights(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: LinearNoBias::from_weights(w, &format!("{prefix}.net.0.proj"), dtype)?,
            out: LinearNoBias::from_weights(w, &format!("{prefix}.net.2"), dtype)?,
        })
    }

    /// The two tensor names the feed-forward consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = LinearNoBias::names(&format!("{prefix}.net.0.proj")).to_vec();
        v.extend(LinearNoBias::names(&format!("{prefix}.net.2")));
        v
    }

    /// Resolve one adapter path segment run **relative to this feed-forward** to its projection.
    ///
    /// `net.0.proj` and `net.2` keep their `nn.Sequential` indices, so both are multi-segment.
    pub fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut LinearNoBias> {
        match path {
            ["net", "0", "proj"] => Some(&mut self.proj),
            ["net", "2"] => Some(&mut self.out),
            _ => None,
        }
    }

    /// Drop every residual on both projections.
    pub fn clear_adapters(&mut self) {
        self.proj.clear_adapters();
        self.out.clear_adapters();
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.proj.nbytes() + self.out.nbytes()
    }

    /// `w2( silu(gate) · value )`, over token blocks bounded by [`FFN_INTERMEDIATE_BUDGET`].
    ///
    /// The `[1, S, 2·ffn_dim]` intermediate this writes is the **widest single tensor in the whole
    /// model** — `[1, 104000, 28672]` ≈ 2.98e9 elements at the lattice ceiling, past `i32::MAX` — which is
    /// the element-count ceiling on this lane rather than anything in the attention (sc-17152). See
    /// the module doc for why the token axis is the exact split and why the equivalence is
    /// tolerance-level rather than bitwise.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_budgeted(
            x,
            AttentionBudget::from_score_elements(FFN_INTERMEDIATE_BUDGET, false),
        )
    }

    /// [`Self::forward`] at an explicit budget.
    ///
    /// Exposed so the equivalence test can drive the chunked path at a small geometry: asserting
    /// *chunked == unchunked* is a false green if the chunking never engages, and the shipped
    /// geometry's intermediate cannot be allocated in a test.
    pub fn forward_budgeted(&self, x: &Tensor, budget: AttentionBudget) -> Result<Tensor> {
        let dims = x.dims();
        let rank = dims.len();
        if rank < 2 {
            // A rank-<2 activation has no token axis to split, so this is a span — record it, or
            // the probe would report the *previous* call's count and every assertion keyed on it
            // would read a stale value.
            record_span_count(1);
            return self.forward_span(x);
        }
        let axis = rank - 2;
        let seq = dims[axis];
        let leading: usize = dims[..axis].iter().product();
        let width = self.proj.base_shape()?.0;
        let block = ffn_token_block(leading, width, seq, budget);
        if block >= seq {
            record_span_count(1);
            return self.forward_span(x);
        }

        let mut blocks = Vec::new();
        let mut start = 0;
        while start < seq {
            let len = block.min(seq - start);
            blocks.push(self.forward_span(&x.narrow(axis, start, len)?.contiguous()?)?);
            start += len;
        }
        record_span_count(blocks.len());
        Ok(Tensor::cat(&blocks, axis)?)
    }

    /// The un-chunked SwiGLU over whatever token span it is handed — the whole forward when the
    /// intermediate fits, one block when it does not.
    fn forward_span(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(x)?;
        let (gate, value) = split_gate_value(&h)?;
        self.out.forward(&(silu(&gate)?.mul(&value)?))
    }
}

/// Test-only observation of how many token spans the last [`DitFeedForward::forward_budgeted`] call
/// actually ran.
///
/// Without this every equivalence assertion below is a **false green**: they compare *chunked ==
/// unchunked*, which is trivially true when the chunking never engages — so the whole suite would
/// keep passing with the lever deleted, or with the loop sizing its blocks from arithmetic of its
/// own instead of the shared planner. Compiled out entirely in release. `RUST_TEST_THREADS=1` is
/// forced repo-wide (`.cargo/config.toml`), so a process-global counter is safe. Mirrors
/// `candle_gen::attention::chunk_probe`.
#[cfg(test)]
static LAST_SPAN_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
fn record_span_count(n: usize) {
    LAST_SPAN_COUNT.store(n, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn record_span_count(_n: usize) {}

/// Affine RMSNorm as a loadable block-level norm (`norm1` / `norm2` / `final_norm` /
/// `norm_out.norm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Load `{prefix}.weight` at `dtype`.
    pub fn from_weights(w: &Weights, prefix: &str, eps: f64, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
            eps,
        })
    }

    /// `weight · x / sqrt(mean(x²) + eps)` over the last axis.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_weighted(x, &self.weight, self.eps)
    }

    /// The one tensor name this norm consumes.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// Deterministic spread values — a fixed pseudo-random fixture, so the equivalence measurement
    /// is reproducible run to run.
    fn spread(shape: &[usize], phase: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.37) + phase).sin() * 0.9)
            .collect();
        Tensor::from_vec(vals, shape, &Device::Cpu).expect("fixture tensor")
    }

    fn linear(shape: (usize, usize), phase: f32) -> LinearNoBias {
        let mut map = std::collections::HashMap::new();
        map.insert("p.weight".to_string(), spread(&[shape.0, shape.1], phase));
        LinearNoBias::from_weights(&Weights::from_map(map), "p", DType::F32).expect("dense fixture")
    }

    /// A tiny bias-free SwiGLU: `proj` is `[2·ffn, hidden]`, `out` is `[hidden, ffn]`.
    fn tiny_ff(hidden: usize, ffn: usize) -> DitFeedForward {
        DitFeedForward {
            proj: linear((2 * ffn, hidden), 0.0),
            out: linear((hidden, ffn), 1.3),
        }
    }

    /// Token spans the last [`DitFeedForward::forward_budgeted`] ran — `1` is the un-chunked path.
    fn spans_run() -> usize {
        LAST_SPAN_COUNT.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// **Relative max-abs-diff** — `max|a-b| / max|b|`, the crate's established measure (the same
    /// one `tests/turbo_lora.rs` and `tests/dit_parity.rs` gate on). Deliberately not cosine:
    /// cosine is blind to a uniform scale error, which is precisely what a truncated chunk tail
    /// would look like.
    fn rel_max_abs(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims(), "shape");
        let max_abs = |t: &Tensor| -> f32 {
            t.abs()
                .expect("abs")
                .flatten_all()
                .expect("flat")
                .max(0)
                .expect("max")
                .to_vec0::<f32>()
                .expect("scalar")
        };
        let d = max_abs(&(a - b).expect("difference"));
        let scale = max_abs(b);
        if scale == 0.0 {
            d
        } else {
            d / scale
        }
    }

    /// Chunking the token axis is *mathematically* exact — no reduction crosses tokens — but not
    /// bitwise, because narrowing the axis changes the GEMM's `M`. Gate it on relative max-abs.
    ///
    /// [`spans_run`] is what keeps this from being a false green: without it the test passes
    /// trivially with the chunking deleted, since both sides would then run the same single GEMM.
    #[test]
    fn chunked_swiglu_matches_the_unchunked_projection() {
        let (hidden, ffn, seq) = (8usize, 6usize, 37usize);
        let ff = tiny_ff(hidden, ffn);
        let x = spread(&[1, seq, hidden], 0.7);

        // rows_per_token = 1 · 2·ffn = 12, so a 40-element budget plans 3-token blocks.
        let bounded = AttentionBudget::from_score_elements(40, false);
        let block = ffn_token_block(1, 2 * ffn, seq, bounded);
        assert!(
            block < seq && block > 0,
            "the chunking must actually engage, got block {block} for seq {seq}"
        );
        assert_eq!(block, 3, "40 / (1 · 12) = 3 token rows per block");

        let want = ff.forward(&x).expect("unchunked forward");
        assert_eq!(spans_run(), 1, "the default budget must not chunk 37 × 12");
        let got = ff.forward_budgeted(&x, bounded).expect("chunked forward");
        assert_eq!(
            spans_run(),
            seq.div_ceil(block),
            "the forward must actually run the planned number of spans"
        );
        assert_eq!(
            got.dims(),
            want.dims(),
            "the chunked forward must recompose"
        );

        let drift = rel_max_abs(&got, &want);
        println!("[swiglu] chunked vs unchunked rel-max-abs = {drift:.3e}");
        assert!(
            drift < 1e-5,
            "chunked SwiGLU must agree with the single-pass projection to 1e-5; got {drift:.3e}"
        );

        // A ragged tail (37 = 12·3 + 1) is covered: a boundary that dropped the remainder would
        // change the shape, and one that double-counted it would move the values.
        let uneven = ff
            .forward_budgeted(&x, AttentionBudget::from_score_elements(12 * 5, false))
            .expect("ragged chunked forward");
        assert_eq!(
            spans_run(),
            8,
            "37 = 5·7 + 2 — eight spans, the last ragged"
        );
        assert!(rel_max_abs(&uneven, &want) < 1e-5, "ragged tail");
    }

    /// The default budget leaves a small geometry on the single un-chunked GEMM, so nothing that
    /// already fits changes at all.
    #[test]
    fn a_geometry_that_fits_stays_on_the_unchunked_path() {
        let (hidden, ffn, seq) = (8usize, 6usize, 37usize);
        assert_eq!(
            ffn_token_block(
                1,
                2 * ffn,
                seq,
                AttentionBudget::from_score_elements(FFN_INTERMEDIATE_BUDGET, false)
            ),
            seq,
            "a 37 × 12 intermediate is nine orders under the budget"
        );
        let ff = tiny_ff(hidden, ffn);
        let x = spread(&[1, seq, hidden], 0.7);
        let a = ff.forward(&x).expect("forward");
        let b = ff.forward_span(&x).expect("span");
        assert_eq!(rel_max_abs(&a, &b), 0.0, "byte-identical to the bare span");
    }

    /// **The shipped ceiling, as arithmetic.** `[1, 104000, 28672]` is ~2.98e9 elements — over
    /// `i32::MAX` — and this proves the plan splits it into blocks that each stay under the budget,
    /// cover the sequence exactly once, and never exceed `i32::MAX`. Pure `usize`/`u64`: allocating
    /// the tensor would need ~11.9 GB at f32, and the point is the plan, not the bytes.
    ///
    /// `SEQ` is the module doc's ~104k packed rows at the 345-frame lattice ceiling (102,816 video
    /// + 1,150 audio + the text rows), rounded down — the plan only gets *more* chunked above it.
    #[test]
    fn the_shipped_ffn_intermediate_is_planned_below_i32_max() {
        const SEQ: usize = 104_000;
        const WIDTH: usize = 28_672; // 2 · ffn_dim
        let unchunked = SEQ as u64 * WIDTH as u64;
        assert!(
            unchunked > i32::MAX as u64,
            "the premise: one unchunked GEMM writes {unchunked} elements, over i32::MAX"
        );

        let budget = AttentionBudget::from_score_elements(FFN_INTERMEDIATE_BUDGET, false);
        let block = ffn_token_block(1, WIDTH, SEQ, budget);
        assert!(block > 0 && block < SEQ, "the plan must chunk, got {block}");
        assert!(
            block as u64 * WIDTH as u64 <= FFN_INTERMEDIATE_BUDGET,
            "each block must fit the declared budget"
        );
        assert!(
            block as u64 * WIDTH as u64 <= i32::MAX as u64,
            "each block must be i32-indexable"
        );

        // The loop `forward_budgeted` runs, in the same order, over the same arithmetic.
        let mut covered = 0usize;
        let mut chunks = 0usize;
        let mut start = 0usize;
        while start < SEQ {
            let len = block.min(SEQ - start);
            covered += len;
            chunks += 1;
            start += len;
        }
        assert_eq!(covered, SEQ, "the blocks must tile the sequence exactly");
        assert_eq!(chunks, SEQ.div_ceil(block), "no stray or missing block");
        assert!(chunks > 1, "the shipped geometry is genuinely chunked");
    }

    /// Rank-1 activations have no token axis to chunk; the plan must not divide by, or narrow, an
    /// axis that is not there.
    #[test]
    fn a_rank_one_activation_takes_the_span_path() {
        let ff = tiny_ff(8, 6);
        // Poison the probe first: the rank-<2 branch returns before the planner runs, so if it did
        // not record its own span count the reader below would report this stale value and every
        // assertion keyed on the probe after a rank-1 call would be reading the previous forward.
        let wide = spread(&[6, 8], 0.3);
        ff.forward_budgeted(&wide, AttentionBudget::from_score_elements(1, false))
            .expect("chunked forward");
        assert!(spans_run() > 1, "the poisoning call must have chunked");

        let x = spread(&[8], 0.2);
        let got = ff
            .forward_budgeted(&x, AttentionBudget::from_score_elements(1, false))
            .expect("rank-1 forward");
        assert_eq!(got.dims(), &[8]);
        assert_eq!(
            spans_run(),
            1,
            "the rank-<2 branch is one span and records it"
        );
        assert_eq!(ffn_token_block(1, 12, 0, AttentionBudget::UNBOUNDED), 0);
    }

    // ─── the packed tier (sc-20267) ─────────────────────────────────────────────────────────────

    /// **The detect fires through the real loader**, on the real key spelling — a `.scales` sibling
    /// under `{prefix}` makes `LinearNoBias` load its base quantized, with no dense weight
    /// materialized, and the same call on a dense key set is unchanged.
    #[test]
    fn a_packed_prefix_loads_the_projection_quantized() {
        let (out, in_) = (32usize, 128usize);
        let mut map = std::collections::HashMap::new();
        crate::quant::testkit::insert_packed(&mut map, "attn.to_q", out, in_, 11);
        let packed = LinearNoBias::from_weights(&Weights::from_map(map), "attn.to_q", DType::F32)
            .expect("packed load");
        assert!(packed.is_packed(), "the `.scales` sibling must pack");
        assert_eq!(packed.base_shape().unwrap(), (out, in_));
        // The resident base is the Q4_1 block stream, not a dense f32 weight.
        assert!(
            packed.nbytes() < out * in_ * 4,
            "a packed base must not cost a dense weight: {} vs {}",
            packed.nbytes(),
            out * in_ * 4
        );

        let dense = linear((out, in_), 0.4);
        assert!(
            !dense.is_packed(),
            "no `.scales` ⇒ the dense path, unchanged"
        );
        assert_eq!(dense.nbytes(), out * in_ * 4);
    }

    /// **The packed forward matches a dense reference of the same dequantized weights**, gated on
    /// relative max-abs (never cosine — cosine cannot see a mis-decoded group scale).
    ///
    /// The reference is the affine grid `scale·q + bias` the tier's producer quantized, loaded
    /// through the *same* `LinearNoBias::from_weights` on a dense key set. So this measures the
    /// packed **path**, not quantization error.
    #[test]
    fn the_packed_forward_matches_a_dense_reference() {
        let (out, in_) = (32usize, 128usize);
        let mut map = std::collections::HashMap::new();
        let grid = crate::quant::testkit::insert_packed(&mut map, "ff.net.2", out, in_, 17);
        let packed = LinearNoBias::from_weights(&Weights::from_map(map), "ff.net.2", DType::F32)
            .expect("packed load");

        let mut dense_map = std::collections::HashMap::new();
        dense_map.insert("ff.net.2.weight".to_string(), grid);
        let dense =
            LinearNoBias::from_weights(&Weights::from_map(dense_map), "ff.net.2", DType::F32)
                .expect("dense load");

        // Rank 3, so the leading-dim handling is exercised as it is in the real block.
        let x = spread(&[1, 5, in_], 0.9);
        let got = packed.forward(&x).expect("packed forward");
        let want = dense.forward(&x).expect("dense forward");
        assert_eq!(got.dims(), want.dims());
        let drift = rel_max_abs(&got, &want);
        println!("[linear] packed vs dense-reference rel-max-abs = {drift:.3e}");
        assert!(
            drift < 5e-3,
            "the Q4_1 repack is lossless up to the f16 scale/bias cast; got {drift:.3e}"
        );
    }

    /// **A LoRA composes over a PACKED base** — the property the shared `AdaptLinear` adoption
    /// buys, and the one the hand-rolled residual could never have: the base stays quantized while
    /// the residual rides unmerged, and its contribution equals the same residual measured over the
    /// dense reference.
    #[test]
    fn a_lora_rides_a_packed_base_unmerged() {
        let (out, in_, rank) = (32usize, 128usize, 4usize);
        let mut map = std::collections::HashMap::new();
        let grid = crate::quant::testkit::insert_packed(&mut map, "attn.to_v", out, in_, 23);
        let mut packed =
            LinearNoBias::from_weights(&Weights::from_map(map), "attn.to_v", DType::F32)
                .expect("packed load");
        let mut dense_map = std::collections::HashMap::new();
        dense_map.insert("attn.to_v.weight".to_string(), grid);
        let mut dense =
            LinearNoBias::from_weights(&Weights::from_map(dense_map), "attn.to_v", DType::F32)
                .expect("dense load");

        let a = spread(&[in_, rank], 0.2);
        let b = spread(&[rank, out], 1.1);
        let x = spread(&[1, 5, in_], 0.9);
        let packed_base = packed.forward(&x).expect("bare packed");
        let dense_base = dense.forward(&x).expect("bare dense");

        packed.push_lora(a.clone(), b.clone(), 0.75).expect("push");
        dense.push_lora(a, b, 0.75).expect("push");
        assert!(
            packed.is_packed(),
            "the install must NOT dequantize the base"
        );
        assert!(packed.is_adapted());

        let packed_residual = (packed.forward(&x).unwrap() - packed_base.clone()).unwrap();
        let dense_residual = (dense.forward(&x).unwrap() - dense_base).unwrap();
        let drift = rel_max_abs(&packed_residual, &dense_residual);
        println!("[linear] packed-vs-dense LoRA residual rel-max-abs = {drift:.3e}");
        assert!(
            drift < 1e-5,
            "the residual is computed from the ACTIVATION and the factors only, so it must be \
             identical on either base; got {drift:.3e}"
        );

        // Clearing reverts to the bare packed base exactly.
        packed.clear_adapters();
        assert!(!packed.is_adapted());
        assert_eq!(
            rel_max_abs(&packed.forward(&x).unwrap(), &packed_base),
            0.0,
            "clear_adapters must restore the bare base bit-for-bit"
        );
    }

    /// A factor pair that does not compose is refused on a **packed** base too — the shape check
    /// reads the `AdaptLinear`'s recovered `[out, in]`, which exists with no dense weight.
    #[test]
    fn a_mismatched_factor_is_refused_on_a_packed_base() {
        let mut map = std::collections::HashMap::new();
        crate::quant::testkit::insert_packed(&mut map, "attn.to_k", 32, 128, 29);
        let mut packed =
            LinearNoBias::from_weights(&Weights::from_map(map), "attn.to_k", DType::F32)
                .expect("packed load");
        let err = packed
            .push_lora(spread(&[64, 4], 0.1), spread(&[4, 32], 0.2), 1.0)
            .expect_err("a [64, rank] factor must not compose onto a [32, 128] projection");
        assert!(err.to_string().contains("do not compose"), "{err}");
    }
}
