//! Adapter framework — LoRA + LoKr applied as forward-time residuals over a shared
//! base. Quantized-safe: the base is never fused/mutated. Ported from the sc-2338
//! spike; mirrors the Python mflux fork's `LoKrLinear` / `FusedLoRALinear` (sc-2216).
//!
//! The base is a real `nn::Linear` *or* `nn::QuantizedLinear` (sc-2342), so quantization
//! and adapters compose: `base(x) + Σ adapter.residual(x)`. Forward is taken by `&self`
//! (we call the underlying ops directly rather than the `&mut self` `Module` trait), so a
//! whole model tree can be evaluated through shared references.
//!
//! Adapters are installed by dotted path via [`AdaptableHost`] / [`install_adapter`] — the
//! Rust stand-in for Python's dynamic `getattr`-swap, since mlx-rs flattens module params to
//! `Array` leaves and cannot replace a submodule in place.

use mlx_rs::{
    module::Param,
    nn::{Linear as MlxLinear, QuantizedLinear},
    ops::{
        add, addmm, einsum, indexing::TryIndexOp, kron, matmul, multiply, quantized_matmul,
        quantized_matmul_bias as quantized_matmul_bias_op,
    },
    Array, Dtype,
};

use crate::scalar;
use crate::{Error, Result};

pub mod facts;

/// Canonical backend linear projection type alias on the shared `mlx-layers` spine (`BACK-1`, `BACK-4`).
pub type Linear = AdaptableLinear;

/// Request-scoped / per-pass adapter scale overrides (`PIPE-14`, `BACK-1`, `BACK-4`).
///
/// Shared resident modules remain immutable during a run; callers pass `ScaleContext` into
/// [`AdaptableLinear::forward_with_scales`], [`AdaptableLinear::forward_split_with_scales`], or
/// [`AdaptableLinear::forward_upcast_with_scales`] to override or modulate adapter scales per
/// request or per pass without mutating `&self`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScaleContext<'a> {
    /// Optional per-adapter scale overrides indexed by adapter position in [`AdaptableLinear::adapters`].
    pub adapter_scales: Option<&'a [f32]>,
    /// Optional uniform scale multiplier applied across all adapters (defaults to `1.0` when `None`).
    pub uniform_scale: Option<f32>,
    /// Pass index for multi-pass pipelines (e.g., conditional / unconditional / guidance passes).
    pub pass: usize,
}

impl<'a> ScaleContext<'a> {
    /// Default pass-0 context with no scale overrides.
    pub const fn new() -> Self {
        Self {
            adapter_scales: None,
            uniform_scale: None,
            pass: 0,
        }
    }

    /// Override per-adapter scales by position.
    pub const fn with_adapter_scales(adapter_scales: &'a [f32]) -> Self {
        Self {
            adapter_scales: Some(adapter_scales),
            uniform_scale: None,
            pass: 0,
        }
    }

    /// Apply a uniform scale multiplier across all installed adapters for this request/pass.
    pub const fn with_uniform_scale(scale: f32) -> Self {
        Self {
            adapter_scales: None,
            uniform_scale: Some(scale),
            pass: 0,
        }
    }

    /// Attach a pass index to this scale context.
    pub const fn with_pass(mut self, pass: usize) -> Self {
        self.pass = pass;
        self
    }

    /// Resolve the effective scale for adapter at `index` with installed `default_scale`.
    #[inline]
    pub fn scale_for(&self, index: usize, default_scale: f32) -> f32 {
        let base = self
            .adapter_scales
            .and_then(|s| s.get(index).copied())
            .unwrap_or(default_scale);
        match self.uniform_scale {
            Some(mult) => base * mult,
            None => base,
        }
    }
}

/// Affine Q4/Q8 matmul over an already-packed `[out, in]` weight, with an optional dense output
/// bias. With P3 selected, the biased case first attempts the strict exact native epilogue. Any
/// unsupported dtype, shape, group size, or Metal dispatcher executes the original
/// `quantized_matmul` then `add` expression and emits a truthful per-operation fallback receipt.
/// Biasless paths are not a P3 composition and remain a plain quantized matmul.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_with_bias(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    output_bias: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    const EXACT_QUANTIZED_MATMUL_BIAS: &str = "quantized_matmul_bias";
    if let Some(output_bias) = output_bias {
        if crate::exact_epilogues_enabled() {
            match quantized_matmul_bias_op(
                x,
                w,
                scales,
                biases,
                output_bias,
                true,
                group_size,
                bits,
            ) {
                Ok(y) => {
                    crate::record_exact_epilogue_applied(EXACT_QUANTIZED_MATMUL_BIAS);
                    return Ok(y);
                }
                Err(_) => {
                    crate::record_exact_epilogue_fallback(EXACT_QUANTIZED_MATMUL_BIAS);
                }
            }
        }
    }

    let y = quantized_matmul(x, w, scales, biases, true, group_size, bits)?;
    match output_bias {
        Some(output_bias) => Ok(add(&y, output_bias)?),
        None => Ok(y),
    }
}

/// A row-slice over a raw LoRA factor (applied BEFORE the `[r,in]`/`[out,r]`→residual transpose),
/// porting the fork's `LoraTransforms` byte-for-byte (sc-2743).
#[derive(Clone, Debug)]
pub enum LoraRowSlice {
    /// Chunk `index` of `n` equal dim-0 chunks (`chunk = shape[0] / n`), ALWAYS sliced.
    Chunk { n: i32, index: i32 },
    /// Chunk `index` of `n` equal dim-0 chunks IFF `shape[0] % n == 0`, else the whole tensor.
    ChunkIfDivisible { n: i32, index: i32 },
    /// The dim-0 slice `[Σdims[..index] .. Σdims[..=index]]`.
    Dims { dims: Vec<i32>, index: i32 },
}

impl LoraRowSlice {
    /// Apply this row-slice to `t`.
    pub fn apply(&self, t: &Array) -> Result<Array> {
        let rows = t.shape()[0];
        let (start, end) = match self {
            LoraRowSlice::Chunk { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                if rows % n != 0 {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: rows ({rows}) not divisible by n ({n}); \
                         the fused factor is mis-shaped for this split"
                    )));
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::ChunkIfDivisible { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::ChunkIfDivisible: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                if rows % n != 0 {
                    return Ok(t.clone());
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::Dims { dims, index } => {
                let i = *index as usize;
                if *index < 0 || i >= dims.len() {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Dims: index {index} out of range for {} dims",
                        dims.len()
                    )));
                }
                let start: i32 = dims[..i].iter().sum();
                (start, start + dims[i])
            }
        };
        Ok(t.try_index((start..end, ..))?)
    }
}

/// One BFL/ComfyUI adapter target: a set of source key spellings mapping to a diffusers module
/// `target_path`, with an optional [`LoraRowSlice`] on the up/down factor.
#[derive(Clone, Debug)]
pub struct BflTarget {
    /// Diffusers module path that [`AdaptableHost::adaptable_mut`] resolves (concrete, no `{block}`).
    pub target_path: String,
    /// Source up-factor (`lora_up`/`lora_B`) key spellings.
    pub up_keys: Vec<String>,
    /// Source down-factor (`lora_down`/`lora_A`) key spellings.
    pub down_keys: Vec<String>,
    /// Source `alpha` key spellings (no transform, no transpose).
    pub alpha_keys: Vec<String>,
    /// Row-slice applied to the up factor (the qkv split). `None` for a plain rename.
    pub up_slice: Option<LoraRowSlice>,
    /// Row-slice applied to the down factor (shared-or-split). `None` for a plain rename.
    pub down_slice: Option<LoraRowSlice>,
}

/// Default actual LyCORIS residual dtype, shared with header-only host normalization.
pub const DEFAULT_LYCORIS_DTYPE: Dtype = Dtype::Bfloat16;

/// The existing PEFT/global-metadata scale gate, shared with allocation-free header facts.
/// Third-party per-module rank selection is a different format mechanism in `loader`.
pub fn validate_lokr_metadata_scale(alpha: f32, rank: f32) -> Result<()> {
    if !rank.is_finite() || rank <= 0.0 || !alpha.is_finite() {
        return Err(format!(
            "LoKr: invalid metadata scale (rank = {rank}, alpha = {alpha}); rank must be > 0 and \
            alpha finite — a `rank = 0` / non-finite `alpha` would make the delta NaN"
        )
        .into());
    }
    Ok(())
}

/// Read a LoRA `.alpha` scalar tensor. Adapter files are external input, so this never panics: an
/// unreadable or empty tensor (a truncated file, sc-11129) and a NaN/inf value (which would turn the
/// `alpha / rank · scale` fold into a NaN residual) are typed errors naming `label` (the tensor key
/// or adapter path). The one shared reader for every provider's LoRA installer.
pub fn read_scalar_alpha(a: &Array, label: &str) -> Result<f32> {
    // An empty array has no data buffer to read, so name it before the slice read does.
    if a.size() == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: empty .alpha tensor (malformed adapter file)"
        )));
    }
    let alpha = a
        .as_dtype(Dtype::Float32)?
        .try_as_slice::<f32>()
        .map_err(|e| {
            crate::Error::Msg(format!(
                "{label}: .alpha is not a readable scalar array: {e}"
            ))
        })?
        .first()
        .copied()
        .ok_or_else(|| {
            crate::Error::Msg(format!(
                "{label}: empty .alpha tensor (malformed adapter file)"
            ))
        })?;
    if !alpha.is_finite() {
        return Err(crate::Error::Msg(format!(
            "{label}: .alpha {alpha} is not finite"
        )));
    }
    Ok(alpha)
}

/// Reconstruct a LoKr weight delta `ΔW = (alpha/rank) · kron(w1, w2)`, reshaped to the
/// base weight's logical `[out, in]` and cast to `out_dtype`. Each Kronecker factor is either
/// full (`w1` / `w2`) or a low-rank product (`w1_a @ w1_b` / `w2_a @ w2_b`). Mirrors
/// PEFT/LyCORIS `LoKrLayer.get_delta_weight` (pending the sc-2324 cross-impl parity check).
///
/// `out_dtype` is `Bfloat16` for the fork-parity residual path (Z-Image/Qwen — PARITY-BF16,
/// sc-2609) and `Float32` for the SDXL merge path (f32-everywhere, no fork to match — sc-2640).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    alpha: f32,
    rank: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    // Guard the metadata-derived `alpha`/`rank` at the shared seam so EVERY consumer inherits it
    // (F-141, sc-11129). A file that stamps `rank = 0` (alpha then defaults to rank) makes the scale
    // `0/0 = NaN`, baked into the reconstructed delta and merged as NaN — NaN-poisoning every
    // subsequent render while the load reports success. `lora_delta` already rejects rank 0 (sc-5252),
    // but this middle path was missed; hoisting the guard here retires the class rather than patching
    // each callsite.
    validate_lokr_metadata_scale(alpha, rank)?;
    // The SceneWorks peft path bakes `alpha/rank` as the scale and is always linear (no tucker
    // factor, equal-rank factors) — delegate to the general scaled form, which is then byte-identical.
    reconstruct_lokr_delta_scaled(
        alpha / rank,
        base_shape,
        w1,
        w1_a,
        w1_b,
        w2,
        None,
        w2_a,
        w2_b,
        out_dtype,
    )
}

/// Reconstruct a LoKr weight delta `ΔW = scale · kron(w1, w2)`, reshaped to `base_shape` and cast to
/// `out_dtype`. Generalizes [`reconstruct_lokr_delta`] for **third-party LyCORIS** LoKr (sc-3642):
/// the caller passes the final `scale` directly (lycoris derives it per module — `alpha/rank`, or a
/// forced `1.0` when both factors are full), and `w2` may be a **tucker/CP** factor
/// (`w2_t2` + `w2_a` + `w2_b`, lycoris `use_cp`). Each Kronecker factor is full (`w1`/`w2`), a
/// low-rank product (`w1_a@w1_b` / `w2_a@w2_b`), or — for `w2` only — the tucker rebuild
/// `einsum("ij…,ip,jr->pr…", t2, w2_a, w2_b)`. Mirrors LyCORIS `LokrModule.get_weight` + `make_kron`
/// (w1's trailing dims are unsqueezed to w2's rank before the product — the conv case).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta_scaled(
    scale: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_t2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_t2, w2_a, w2_b) {
        (Some(w), _, _, _) => w.clone(),
        (_, Some(t2), Some(a), Some(b)) => rebuild_tucker(t2, a, b)?,
        (_, None, Some(a), Some(b)) => matmul(a, b)?,
        _ => {
            return Err("LoKr: w2 missing (need full w2, tucker t2+w2_a+w2_b, or w2_a@w2_b)".into())
        }
    };
    // LyCORIS `make_kron` unsqueezes w1's TRAILING dims to w2's rank before `torch.kron` (the conv
    // case: w1 `[a,c]` → `[a,c,1,1]` against w2 `[b,d,kH,kW]`). Linear factors share rank → no-op.
    let factor1 = match_trailing_rank(&factor1, factor2.shape().len())?;
    let delta = multiply(&kron(&factor1, &factor2)?, scalar(scale))?;
    Ok(delta.reshape(base_shape)?.as_dtype(out_dtype)?)
}

/// Reshape `a` to append trailing length-1 dims until it has `ndim` dimensions (LyCORIS `make_kron`'s
/// `w1.unsqueeze(-1)` loop). A no-op when `a` already has ≥ `ndim` dims.
fn match_trailing_rank(a: &Array, ndim: usize) -> Result<Array> {
    let mut shape = a.shape().to_vec();
    if shape.len() >= ndim {
        return Ok(a.clone());
    }
    shape.resize(ndim, 1);
    Ok(a.reshape(&shape)?)
}

/// LyCORIS `rebuild_tucker(t, wa, wb) = einsum("i j …, i p, j r -> p r …", t, wa, wb)` — the CP/tucker
/// rebuild for a conv `w2` (`use_cp`). `t2` is `[i, j, kH, kW]`, `wa` is `[i, p]`, `wb` is `[j, r]`,
/// yielding `[p, r, kH, kW]`. Only the 4-D conv form lycoris emits is supported (others error loudly).
fn rebuild_tucker(t2: &Array, wa: &Array, wb: &Array) -> Result<Array> {
    if t2.shape().len() != 4 {
        return Err(format!(
            "LoKr tucker: expected a 4-D lokr_t2 [in,out,kH,kW], got shape {:?}",
            t2.shape()
        )
        .into());
    }
    Ok(einsum("ijhw,ip,jr->prhw", [t2, wa, wb])?)
}

/// The two small Kronecker factors of a LoKr delta, kept UNMATERIALIZED for a deferred
/// structured forward (sc-10050). `ΔW = scale · kron(w1, w2)` reshapes to the base's `[out, in]`,
/// but the Kronecker–vector identity lets us apply it WITHOUT ever forming that `[out, in]` tensor:
/// with `w1` `[a, c]` (LyCORIS `shape[0][0] × shape[1][0]`) and `w2` `[b, d]`
/// (`shape[0][1] × shape[1][1]`), so `out = a·b` and `in = c·d`, the residual `y = x · ΔWᵀ` is
///   `Y = w1 · X · w2ᵀ`  (then flatten row-major `[.., a, b] → [.., out]`),
/// where `X = reshape(x, [.., c, d])`. Two small matmuls (`[a,c]·[..,c,d]` then `·[d,b]`) touch only
/// the factor shapes — never `[out, in]` — so a LoKr applies on a packed Q4/Q8 base at the same
/// memory profile as a plain LoRA (sc-10050 / epic 10043). `w1`/`w2` are the *small* Kronecker
/// factors, materialized from their low-rank `w_a·w_b` inner products if decomposed (that product is
/// bounded by the factor dims, NOT `out×in`), so every linear LoKr variant is deferrable this way.
#[derive(Clone)]
pub struct LokrFactors {
    /// `[a, c]` — the left Kronecker factor (`out = a·b`, `in = c·d`).
    pub w1: Array,
    /// `[b, d]` — the right Kronecker factor.
    pub w2: Array,
    /// `a` — leading (row) count of `w1`, so the flattened output index is `p·b + q`.
    pub a: i32,
    /// `b` — leading (row) count of `w2`.
    pub b: i32,
    /// `c` — trailing (col) count of `w1`, so the flattened input index is `r·d + s`.
    pub c: i32,
    /// `d` — trailing (col) count of `w2`.
    pub d: i32,
    /// The **pre-bake** `scale` (sc-15265). `scale` is folded into [`w2`](Self::w2) at build time and
    /// is not otherwise recoverable from the factors — but [`Adapter::is_disabled`] needs it to
    /// short-circuit a disabled structured LoKr exactly as it short-circuits LoRA/LoKr. Keeping it
    /// here is what makes the short-circuit *universal* instead of a two-of-three exemption: at
    /// `scale == 0` the residual is never computed, so a `w2` factor carrying a non-finite value (an
    /// `Inf` in a third-party checkpoint — [`build_lokr_factors`] guards the *scale* for finiteness
    /// but cannot vouch for the factors) can no longer turn `Inf · 0.0` into a `NaN` that poisons the
    /// output. It is otherwise unused: the residual math still reads the already-baked `w2`.
    pub scale: f32,
}

/// Exact selector shared by actual packed installation and header normalization.
pub fn lokr_structured_layout(
    base_shape: &[i32],
    factor1_shape: &[i32],
    factor2_shape: &[i32],
    has_tucker: bool,
) -> Option<[i32; 4]> {
    if has_tucker || base_shape.len() != 2 || factor1_shape.len() != 2 || factor2_shape.len() != 2 {
        return None;
    }
    let (a, c) = (factor1_shape[0], factor1_shape[1]);
    let (b, d) = (factor2_shape[0], factor2_shape[1]);
    if a.checked_mul(b) != Some(base_shape[0]) || c.checked_mul(d) != Some(base_shape[1]) {
        return None;
    }
    Some([a, b, c, d])
}

/// Build the small [`LokrFactors`] from a LoKr module's factors (full `w1`/`w2` or low-rank
/// `w_a·w_b`), baking `scale` into `w2` and casting to `out_dtype` — the deferred, allocation-free
/// counterpart to [`reconstruct_lokr_delta_scaled`] (which materializes the full `[out, in]` delta).
/// Only the **linear** (2-D-factor) LoKr forms are deferrable via the vec-trick; a **tucker/CP** `w2`
/// (`w2_t2`, conv-only in LyCORIS) has no 2-D matrix form, so this returns `Ok(None)` and the caller
/// falls back to materialization (dense) or a clear error (packed). `base_shape` is the target's
/// logical `[out, in]`, used to confirm `a·b == out` and `c·d == in`.
#[allow(clippy::too_many_arguments)]
pub fn build_lokr_factors(
    scale: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_t2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Option<LokrFactors>> {
    // Guard the caller-derived `scale` at the shared seam so the packed/deferred path inherits the
    // F-141 protection too (sc-11129): the SDXL/Wan callers compute `(alpha/rank)·strength`, so a
    // `rank = 0` metadata makes `scale = NaN`, which would bake into the structured `w2` and
    // NaN-poison the deferred residual. Refuse it here rather than silently install a NaN factor.
    if !scale.is_finite() {
        return Err(format!("LoKr: non-finite scale ({scale}) — rank = 0 / invalid alpha").into());
    }
    // Tucker/CP `w2` is a 4-D conv factor with no 2-D matrix form — not deferrable via the vec-trick.
    // (LyCORIS only emits `lokr_t2` for conv layers; the Wan DiT adapter surface is all Linear, so
    // this never fires there, but guard it so a conv LoKr falls back rather than silently mis-applies.)
    if w2_t2.is_some() {
        return Ok(None);
    }
    // The two small Kronecker factors — full, or the low-rank product (bounded by the factor dims,
    // NEVER `out×in`). Building `w1_a @ w1_b` yields the small `[a, c]`, not the packed delta.
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w2 missing (need full w2 or w2_a@w2_b)".into()),
    };
    // A conv factor or a factor/base mismatch retains the materialization fallback. Metadata
    // normalization consumes this same selector rather than inferring structured storage from kind.
    let Some([a, b, c, d]) = lokr_structured_layout(
        base_shape,
        factor1.shape(),
        factor2.shape(),
        w2_t2.is_some(),
    ) else {
        return Ok(None);
    };
    // Bake `scale` into `w2` (either factor works; `w2` keeps `w1` a clean copy) and match dtype.
    let factor2 = multiply(&factor2, scalar(scale))?.as_dtype(out_dtype)?;
    let factor1 = factor1.as_dtype(out_dtype)?;
    Ok(Some(LokrFactors {
        w1: factor1,
        w2: factor2,
        a,
        b,
        c,
        d,
        // Retained so a `scale == 0` structured LoKr is recoverably disabled (sc-15265) — see the
        // field doc. The value baked into `w2` above is the authoritative one for the math.
        scale,
    }))
}

/// Reconstruct a **LoHa** (Hadamard-product) weight delta `ΔW = scale · ((w1_a·w1_b) ⊙ (w2_a·w2_b))`,
/// reshaped to `base_shape` and cast to `out_dtype` (sc-3643). Third-party LyCORIS LoHa decomposes a
/// delta as the elementwise product of TWO low-rank products (vs LoKr's Kronecker). With tucker
/// factors (`t1`/`t2`, lycoris `use_cp` for conv) each side is a CP rebuild
/// `einsum("ij…,jr,ip->pr…", t, w_b, w_a)`. `scale` is `alpha/rank` (the caller derives it per
/// module). Mirrors LyCORIS `LohaModule.get_weight` / `loha_diff_weight` (`HadaWeight`/`HadaWeightTucker`):
/// the saved factors map as `w1d=hada_w1_b, w1u=hada_w1_a, w2d=hada_w2_b, w2u=hada_w2_a`, so the
/// non-tucker product is `(w1_a @ w1_b) ⊙ (w2_a @ w2_b)` (conv folds the kernel into the factors' dim 1,
/// then `reshape(base_shape)` restores `[out,in,kH,kW]`).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_loha_delta(
    scale: f32,
    base_shape: &[i32],
    w1_a: &Array,
    w1_b: &Array,
    w2_a: &Array,
    w2_b: &Array,
    t1: Option<&Array>,
    t2: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    let (m1, m2) = match (t1, t2) {
        (Some(t1), Some(t2)) => (
            loha_rebuild_tucker(t1, w1_b, w1_a)?,
            loha_rebuild_tucker(t2, w2_b, w2_a)?,
        ),
        _ => (matmul(w1_a, w1_b)?, matmul(w2_a, w2_b)?),
    };
    let delta = multiply(&multiply(&m1, &m2)?, scalar(scale))?;
    Ok(delta.reshape(base_shape)?.as_dtype(out_dtype)?)
}

/// LoHa tucker rebuild: `einsum("i j …, j r, i p -> p r …", t, w_b, w_a)` (lycoris `HadaWeightTucker`).
/// `t` is `[dim, dim, kH, kW]`, `w_b` is `[dim, in]`, `w_a` is `[dim, out]` → `[out, in, kH, kW]`.
fn loha_rebuild_tucker(t: &Array, w_b: &Array, w_a: &Array) -> Result<Array> {
    if t.shape().len() != 4 {
        return Err(format!(
            "LoHa tucker: expected a 4-D hada_t [dim,dim,kH,kW], got shape {:?}",
            t.shape()
        )
        .into());
    }
    Ok(einsum("ijhw,jr,ip->prhw", [t, w_b, w_a])?)
}

/// Fuse a **conv-layer** LoRA pair into a single conv-weight delta, returned in the trained-file
/// NCHW `[out, in, kH, kW]` layout (sc-2919). Conv LoRAs decompose a conv into a spatial `down`
/// (`lora_down`, `[rank, in, kH, kW]`) followed by a 1×1 `up` (`lora_up`, `[out, rank, 1, 1]`); the
/// fused weight is the composition of those two convs:
///   `δ[o, i, y, x] = Σ_r up[o, r] · down[r, i, y, x]`,
/// which is exactly `up[out, rank] @ down[rank, in·kH·kW]` reshaped back to `[out, in, kH, kW]` —
/// bit-identical to PEFT/diffusers' `Conv2d` LoRA fusion (`F.conv2d(down.permute(1,0,2,3), up)`),
/// and uniform across 1×1 and k×k kernels. Then scaled by `(alpha/rank)·scale`.
///
/// Precision mirrors the SDXL Linear `lora_delta`: the `up @ down` matmul runs in **f32** (correct,
/// and avoids the former NAX 16-bit-GEMM bug — now fixed at the toolchain level, sc-2772) and is
/// rounded back through the factors' source dtype (f16 for community/accel LoRAs), so the result is
/// the same value an f16 reference fusion would produce, returned as f32 for the caller to cast to
/// the conv weight's dtype on merge. The merge itself is the chaos-safe `W += δ` (the SDXL ancestral
/// sampler needs a merged weight, not a forward-time residual — cf. [`AdaptableLinear::merge_dense_delta`]).
pub fn conv_lora_delta(
    down: &Array,
    up: &Array,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Array> {
    let src = up.dtype(); // f16 for kohya/community LoRAs; f32 makes the round-trip a no-op.
    let ds = down.shape(); // [rank, in, kH, kW]
    let us = up.shape(); // [out, rank, 1, 1]
                         // A malformed conv LoRA with 2-D factors would panic on the `ds[2]`/`ds[3]` slice below; surface a
                         // typed error up front instead, matching the tucker reconstructors' style (F-006).
    if ds.len() != 4 || us.len() != 4 {
        return Err(format!(
            "conv LoRA: expected 4-D factors (down [rank,in,kH,kW], up [out,rank,1,1]), got down \
             {ds:?} up {us:?}"
        )
        .into());
    }
    if us[1] != ds[0] {
        return Err(format!(
            "conv LoRA: rank mismatch between factors — down[0]={} but up[1]={} (down {ds:?} up {us:?})",
            ds[0], us[1]
        )
        .into());
    }
    let (r, cin, kh, kw) = (ds[0], ds[1], ds[2], ds[3]);
    let out = us[0];
    let down2 = down.reshape(&[r, cin * kh * kw])?; // [rank, in·kH·kW]
    let up2 = up.reshape(&[out, r])?; // [out, rank]
    let ba = matmul(
        &up2.as_dtype(Dtype::Float32)?,
        &down2.as_dtype(Dtype::Float32)?,
    )?;
    let ba = ba.as_dtype(src)?.as_dtype(Dtype::Float32)?;
    // effective_scale in f64 then f32, matching a reference's Python-float arithmetic.
    let eff = ((alpha as f64 / rank as f64) * scale as f64) as f32;
    Ok(multiply(&ba, scalar(eff))?.reshape(&[out, cin, kh, kw])?)
}

/// One adapter's contribution WITHOUT the base, so a host can sum stacked adapters over
/// a single base application.
#[derive(Clone)]
pub enum Adapter {
    /// LoRA: `residual = scale · x·A·B`.
    Lora { a: Array, b: Array, scale: f32 },
    /// LoKr: `residual = scale · x·ΔWᵀ`; `delta` stored bf16 (see [`reconstruct_lokr_delta`]).
    Lokr { delta: Array, scale: f32 },
    /// LoKr applied as a **structured, deferred Kronecker** residual (sc-10050 / epic 10043) — the
    /// full `[out, in]` delta is NEVER materialized. `scale` is baked into `factors.w2`, so the
    /// residual is `vec(w1 · reshape(x) · w2ᵀ)` (see [`LokrFactors`]) — two small matmuls in compute
    /// precision over the packed (or dense) base, matching the folded delta within tolerance. This is
    /// the packed-tier path so a LoKr works on q4/q8 at plain-LoRA memory cost.
    LokrStructured { factors: LokrFactors },
}

impl Adapter {
    /// Evaluate only this residual's adapter-owned payload arrays.
    ///
    /// Safetensors loads and the transpose/reconstruction graph layered over them are lazy. Provider
    /// loaders call this while the adapter file's immutable token is still guarded, so no LoRA/LoKr
    /// payload can first touch disk after the pin's post-check. Base Linear weights are intentionally
    /// absent from this evaluation set.
    pub fn materialize(&self) -> Result<()> {
        mlx_rs::transforms::eval(self.arrays())?;
        Ok(())
    }

    /// The adapter-owned payloads, without base parameters or reconstruction-only inputs.
    /// Transpose/reshape views alias these payloads rather than adding another parameter.
    pub fn arrays(&self) -> Vec<&Array> {
        match self {
            Adapter::Lora { a, b, .. } => vec![a, b],
            Adapter::Lokr { delta, .. } => vec![delta],
            Adapter::LokrStructured { factors } => vec![&factors.w1, &factors.w2],
        }
    }

    /// One adapter's forward-time contribution `scale · …`, replicating the fork's `LoRALinear`
    /// / `LoKrLinear` `.residual` **byte-for-byte** (sc-2718). No dtype is forced: the earlier f32
    /// upcast (sc-2602/2719) was a workaround for the NAX 16-bit dense GEMM returning garbage on the
    /// low-rank `[seq,r]·[r,out]` matmul (`K = rank ≤ 512`, `M ≥ 2`); that GEMM is now correct at the
    /// toolchain level (sc-2772 — Metal target ≥ 26.2), so the math runs in the natural promoted
    /// dtype exactly as the fork does — restoring parity (the f32 forcing was the DEVIATION):
    ///   * LoRA — `scale · (x·A)·B` with `A`/`B` kept at their loaded (file) dtype. The fork never
    ///     casts the factors, so a bf16 `x` against f32 factors (the goldens ship f32) promotes to
    ///     f32; a bf16-factor file runs bf16 (the formerly-buggy shape, now safe).
    ///   * LoKr — `scale · x·ΔWᵀ` with `ΔW` (stored bf16) cast to the **activation dtype** — bf16 on
    ///     the bf16 path — mirroring the fork's `delta.astype(x.dtype)`.
    ///
    /// The result is NOT cast back **here** — this returns the residual in its natural promoted
    /// dtype, byte-for-byte as the fork's `.residual` does. The hand-off to the host is a separate
    /// concern and lives in `AdaptableLinear`'s adapter accumulation, which narrows the residual to the
    /// host's output dtype before the add (sc-15265) so installing an adapter cannot widen the host
    /// Linear — and therefore the whole downstream chain — from bf16 to f32. An f32-activation
    /// target is unchanged either way (FLUX.2; Qwen's f32 image stream; SDXL merges
    /// instead) — the residual was f32 before and stays f32. A bf16-activation target now runs the
    /// residual in bf16 like the fork (Z-Image's latents; Qwen's bf16 text stream); validated against
    /// the fork goldens (Z-Image / Qwen LoRA+LoKr) — px>8 byte-identical to the old forced-f32 path,
    /// i.e. the dtype change is sub-threshold while restoring fork-faithfulness (sc-2718). `scale` is
    /// applied through a dtype-matched scalar so the multiply preserves the residual's dtype, matching
    /// the fork's weak Python-float `scale * …` (a strong f32 scalar would wrongly promote a bf16
    /// residual to f32; verified against MLX).
    /// Installed default scale for this adapter.
    #[inline]
    pub fn default_scale(&self) -> f32 {
        match self {
            Adapter::Lora { scale, .. } | Adapter::Lokr { scale, .. } => *scale,
            Adapter::LokrStructured { factors } => factors.scale,
        }
    }

    pub fn residual(&self, x: &Array) -> Result<Array> {
        self.residual_with_scale(x, self.default_scale())
    }

    /// Evaluate this adapter's residual under a request-scoped `override_scale` (`PIPE-14`).
    pub fn residual_with_scale(&self, x: &Array, override_scale: f32) -> Result<Array> {
        let (r, scale) = match self {
            Adapter::Lora { a, b, .. } => (matmul(&matmul(x, a)?, b)?, override_scale),
            Adapter::Lokr { delta, .. } => {
                let d = delta.as_dtype(x.dtype())?;
                (matmul(x, d.t())?, override_scale)
            }
            // Structured LoKr (sc-10050): the vec-trick, `factors.scale` is already baked into
            // `factors.w2`. When `override_scale` matches `factors.scale`, return directly without
            // any extra op; when overridden and `factors.scale != 0.0`, rescale by the ratio.
            Adapter::LokrStructured { factors } => {
                let y = factors.residual(x)?;
                if (override_scale - factors.scale).abs() <= f32::EPSILON || factors.scale == 0.0 {
                    return Ok(y);
                }
                let ratio = override_scale / factors.scale;
                return Ok(multiply(&y, &scalar(ratio).as_dtype(y.dtype())?)?);
            }
        };
        // Dtype-matched scalar → preserves the residual's dtype (the fork's weak-float `scale * …`).
        Ok(multiply(&r, &scalar(scale).as_dtype(r.dtype())?)?)
    }

    /// `true` when this adapter contributes nothing — a `scale` of exactly zero (sc-15265).
    ///
    /// [`AdaptableLinear::forward`] skips a disabled adapter entirely rather than adding its
    /// (mathematically zero) residual, so **"install an adapter at scale 0" is byte-identical to
    /// "install no adapter at all"**: no residual is formed, nothing is added, and the host's
    /// output array is returned untouched. That is what makes scale 0 usable both as the
    /// "disabled LoRA" model a UI/worker reaches for and as an A/B control when validating an
    /// adapter. (It is also strictly cheaper: two matmuls per adapted Linear are skipped.)
    ///
    /// [`Adapter::LokrStructured`] bakes its `scale` into [`LokrFactors::w2`], so it carries the
    /// pre-bake value in [`LokrFactors::scale`] purely to answer this question (sc-15265). It
    /// deliberately is NOT exempted: a zero `scale` does multiply `w2` to exact zeros, and adding
    /// exact zeros would already be bit-exact — but only for *finite* factors. `Inf · 0.0 = NaN`,
    /// so a checkpoint carrying a non-finite `w2` entry would otherwise NaN-poison the output at
    /// exactly the setting the user chose to turn the adapter OFF. Skipping the residual outright
    /// makes that unreachable rather than merely unlikely.
    pub fn is_disabled(&self) -> bool {
        match self {
            Adapter::Lora { scale, .. } | Adapter::Lokr { scale, .. } => *scale == 0.0,
            Adapter::LokrStructured { factors } => factors.scale == 0.0,
        }
    }
}

impl LokrFactors {
    /// The deferred, allocation-free LoKr residual `scale · x·ΔWᵀ` via the Kronecker–vector identity
    /// (`scale` is baked into [`w2`](Self::w2)). For an activation `x` of shape `[.., in]` (`in = c·d`):
    /// reshape to `[.., c, d]`, compute `Y = w1 · X · w2ᵀ` (`[.., a, b]`), and flatten row-major to
    /// `[.., out]` (`out = a·b`). The two matmuls (`w1 · X` then `· w2ᵀ`) touch only the small factor
    /// shapes `[a,c]`/`[b,d]` — the full `[out, in]` delta is NEVER materialized, so this holds the
    /// same memory profile on a packed Q4/Q8 base as a plain LoRA. Factors are cast to the activation
    /// dtype (mirroring [`Adapter::Lokr`]'s `delta.astype(x.dtype)`) so a bf16 stream runs bf16.
    pub fn residual(&self, x: &Array) -> Result<Array> {
        let w1 = self.w1.as_dtype(x.dtype())?;
        let w2t = self.w2.as_dtype(x.dtype())?.t();
        let lead: Vec<i32> = x.shape()[..x.shape().len() - 1].to_vec();
        let n: i32 = lead.iter().product::<i32>().max(1);
        // Collapse every leading dim into ONE batch axis, then reshape `[N, in] → [N, c, d]`. Doing the
        // batched matmul against an explicit `[N, …]` batch (rather than broadcasting a bare 2-D `w1`
        // over `[.., c, d]`) avoids the numpy/MLX matmul ambiguity when a leading batch dim happens to
        // equal `a`/`b`. `w1`/`w2ᵀ` are prepended a length-1 batch so they broadcast cleanly over `N`.
        let xr = x.reshape(&[n, self.c, self.d])?;
        let w1b = w1.reshape(&[1, self.a, self.c])?;
        let w2tb = w2t.reshape(&[1, self.d, self.b])?;
        // Y = w1 · X · w2ᵀ  → [N, a, b].
        let y = matmul(&matmul(&w1b, &xr)?, &w2tb)?;
        // [N, a, b] → [.., out] (out = a·b), restoring the original leading dims.
        let mut ys = lead;
        ys.push(self.a * self.b);
        Ok(y.reshape(&ys)?)
    }
}

/// A linear base — dense or quantized — evaluated through a shared reference. Mirrors the
/// `forward` of mlx-rs's `nn::Linear` / `nn::QuantizedLinear` but without requiring `&mut`.
#[derive(Clone)]
pub enum LinearBase {
    Dense(MlxLinear),
    Quantized(QuantizedLinear),
    /// Descriptor-driven symmetric INT8, distinct from affine Q4/Q8.
    ConvRot(crate::convrot_int8::ConvRotLinear),
    /// Descriptor-driven packed W4A8 (asym_w4a8_int8), distinct from both ConvRot INT8 and affine
    /// Q4/Q8: two int4 codes per stored byte, a codebook and per-group F8 scales (sc-26685).
    W4A8(crate::w4a8_int8::W4A8Linear),
}

impl LinearBase {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            LinearBase::Dense(l) => {
                // Mirror MLX `nn.Linear` exactly: the biased case is a FUSED `addmm(bias, x, Wᵀ)`
                // — accumulate `x·Wᵀ`, add bias, round to the output dtype ONCE. A separate
                // `matmul` then `add` rounds the matmul, *then* rounds the bias add again — a
                // ~1.4e-3 double-rounding error per biased Linear in bf16 that compounds over a
                // deep net (sc-2779; localized in the Wan DiT, q_proj 1.4e-3 → ~4e-7 with addmm).
                // f32-INVISIBLE and therefore safe for every crate today: with f32 activations
                // (the current Z-Image/Qwen/FLUX path, even with bf16 weights) `addmm == matmul+add`
                // bit-for-bit, because nothing rounds to bf16 mid-op (verified, sc-2779). It bites
                // only once a path runs bf16 activations (the sc-2718–2721 reverts). The unbiased
                // case stays a plain `matmul`, as mlx-rs's own `Linear::forward` does.
                match l.bias.value.as_ref() {
                    Some(b) => addmm(b, x, l.weight.value.t(), 1.0, 1.0)?,
                    None => matmul(x, l.weight.value.t())?,
                }
            }
            LinearBase::Quantized(q) => {
                // Activations are fed to `quantized_matmul` AS-IS — no dtype upcast. `quantized_matmul`
                // accumulates in fp32 (mlx#963) and is correct at every activation shape/dtype, so it
                // was never the buggy op: the NAX 16-bit-GEMM bug lived in the *dense* 16-bit×16-bit
                // Metal GEMM, and that is now fixed at the toolchain level (sc-2772 — metal target ≥26.2).
                // The former bf16→f32 upcast here (sc-2719) guarded a proven non-bug and is removed:
                // feeding bf16 activations straight in matches the fork's own quantized compute dtype
                // (bf16 latents → `quantized_matmul` → bf16), so it is strictly *more* faithful, not less.
                // Weights stay Q4/Q8 throughout. (`q8_smoke.rs` exercises the bf16-activation path.)
                quantized_matmul_with_bias(
                    x,
                    &q.inner.weight.value,
                    &q.scales.value,
                    &q.biases.value,
                    q.inner.bias.value.as_ref(),
                    q.group_size,
                    q.bits,
                )?
            }
            LinearBase::ConvRot(q) => q.forward(x)?,
            LinearBase::W4A8(q) => q.forward(x)?,
        })
    }
}

/// A linear base plus a stack of adapters, applied as `base(x) + Σ adapter.residual(x)`.
/// Quantized-safe: the base weight is never mutated.
#[derive(Clone)]
pub struct AdaptableLinear {
    pub base: LinearBase,
    adapters: Vec<Adapter>,
    /// `true` only for a **training** adapter stack (installed via
    /// [`set_training_adapters`](Self::set_training_adapters)), which opts OUT of both sc-15265
    /// rules in `apply_adapters` — see that method's doc for why.
    training_residual: bool,
}

impl AdaptableLinear {
    /// Build from a raw `[out, in]` weight (and optional bias) — the common path when
    /// loading dense (bf16/fp16/fp32) checkpoints via the `weights` module.
    pub fn dense(weight: Array, bias: Option<Array>) -> Self {
        Self::from_linear(MlxLinear {
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Wrap an existing dense `nn::Linear`.
    pub fn from_linear(linear: MlxLinear) -> Self {
        Self {
            base: LinearBase::Dense(linear),
            adapters: Vec::new(),
            training_residual: false,
        }
    }

    /// Wrap an existing `nn::QuantizedLinear` (sc-2342 quantized weights).
    pub fn from_quantized(q: QuantizedLinear) -> Self {
        Self {
            base: LinearBase::Quantized(q),
            adapters: Vec::new(),
            training_residual: false,
        }
    }
    /// Wrap a validated, native signed INT8 base without a floating weight copy.
    pub fn from_convrot(q: crate::convrot_int8::ConvRotLinear) -> Self {
        Self {
            base: LinearBase::ConvRot(q),
            adapters: Vec::new(),
            training_residual: false,
        }
    }

    /// Wrap a validated, native packed W4A8 base without a dense weight copy.
    pub fn from_w4a8(q: crate::w4a8_int8::W4A8Linear) -> Self {
        Self {
            base: LinearBase::W4A8(q),
            adapters: Vec::new(),
            training_residual: false,
        }
    }

    /// Native INT8 resident base facts; affine Q8 is intentionally not reported here.
    pub fn convrot_base(&self) -> Option<&crate::convrot_int8::ConvRotLinear> {
        match &self.base {
            LinearBase::ConvRot(q) => Some(q),
            _ => None,
        }
    }

    /// Native packed-W4A8 resident base facts; the affine and ConvRot bases are not reported here.
    pub fn w4a8_base(&self) -> Option<&crate::w4a8_int8::W4A8Linear> {
        match &self.base {
            LinearBase::W4A8(q) => Some(q),
            _ => None,
        }
    }

    /// Build a quantized base from **already-packed** parts read off disk — a *pre-quantized*
    /// checkpoint (group-wise affine `weight` u32 codes + `scales` + `biases`, optional dense
    /// `bias`). The consume-side counterpart to [`quantize`](Self::quantize): no re-quantization
    /// happens, the on-disk scales are used as-is. Mirrors the fork's `loading.py` — `nn.quantize`
    /// stubs then `load_weights` of the packed tensors — but as a direct construction. `group_size`
    /// and `bits` come from the checkpoint's manifest (e.g. Wan's `config.json` `quantization` block).
    pub fn from_quantized_parts(
        weight: Array,
        scales: Array,
        biases: Array,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::from_quantized(QuantizedLinear {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: MlxLinear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            },
        })
    }

    /// Stack a new adapter (LoRA or LoKr) on top of any already installed.
    pub fn push(&mut self, adapter: Adapter) {
        self.adapters.push(adapter);
    }

    /// Replace the entire adapter stack under the normal (inference) dtype rules — see
    /// `apply_adapters`. An empty `Vec` clears the stack back to the bare
    /// frozen base, which is what every non-training caller uses this for.
    pub fn set_adapters(&mut self, adapters: Vec<Adapter>) {
        self.adapters = adapters;
        self.training_residual = false;
    }

    /// Replace the entire adapter stack as a **training** stack. The forward-time counterpart to
    /// [`push`](Self::push) used by training (sc-3042/3039): each optimizer step produces new
    /// trainable LoRA factor arrays, so the trainer re-injects a single fresh `Adapter::Lora` per
    /// target every step rather than accumulating residuals. Setting the SAME
    /// `(transpose, alpha/rank fold, scale)` an inference reload applies
    /// (`adapters::loader::install_lora_groups`) makes the trained adapter round-trip bit-for-bit.
    ///
    /// The stack is flagged so `apply_adapters` leaves the trainer's
    /// numerics **exactly as they shipped** — no scale-0 skip, no narrowing cast (sc-15265). Both
    /// the traced `loss_fn` install (`train::lora::install_training_lora_as` /
    /// `install_training_lokr`) and every family's gradient-checkpoint recompute go through this
    /// method, so the train/recompute consistency invariant is preserved by construction.
    pub fn set_training_adapters(&mut self, adapters: Vec<Adapter>) {
        self.adapters = adapters;
        self.training_residual = true;
    }

    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    /// Evaluate every adapter residual attached to this Linear without touching its base weight.
    pub fn materialize_adapters(&self) -> Result<()> {
        for adapter in &self.adapters {
            adapter.materialize()?;
        }
        Ok(())
    }

    /// Evaluate this linear's retained base and adapter arrays.  Each call is deliberately scoped to
    /// one projection: load-time quantizers can invoke it while walking a model so MLX never has to
    /// materialize the complete dense source model before retaining the packed Q4/Q8 result.
    pub fn materialize_weights(&self) -> Result<()> {
        match &self.base {
            LinearBase::Dense(linear) => {
                mlx_rs::transforms::eval(
                    std::iter::once(&linear.weight.value).chain(linear.bias.value.iter()),
                )?;
            }
            LinearBase::Quantized(quantized) => {
                mlx_rs::transforms::eval(
                    [
                        &quantized.inner.weight.value,
                        &quantized.scales.value,
                        &quantized.biases.value,
                    ]
                    .into_iter()
                    .chain(quantized.inner.bias.value.iter()),
                )?;
            }
            LinearBase::ConvRot(q) => q.materialize()?,
            LinearBase::W4A8(q) => q.materialize()?,
        }
        self.materialize_adapters()
    }

    /// Merge a precomputed `[out, in]` delta into the dense base weight (`W += δ`) — the in-place
    /// LoRA/LoKr *merge*, distinct from the forward-time [`Adapter::residual`] stack. The merge
    /// reproduces a reference's merged-weight forward (`(W+δ)·x`) bit-for-bit, where a residual
    /// (`W·x + δ·x`) differs by ~1 ULP; on a chaos-sensitive sampler (SDXL's ancestral) that 1-ULP
    /// cascades to a visible whole-image divergence, so the SDXL provider merges (matching the
    /// vendored `lora.py` `module.weight += delta`) rather than stacking residuals. `delta` is cast
    /// to the base weight's dtype before the add. Errors on a quantized base — a LoRA must be merged
    /// into the dense (e.g. f32) weight BEFORE quantization (the fork merges pre-quantize too).
    pub fn merge_dense_delta(&mut self, delta: &Array) -> Result<()> {
        match &mut self.base {
            LinearBase::Dense(l) => {
                let merged = add(&l.weight.value, &delta.as_dtype(l.weight.value.dtype())?)?;
                l.weight = Param::new(merged);
                Ok(())
            }
            LinearBase::Quantized(_) | LinearBase::ConvRot(_) | LinearBase::W4A8(_) => Err(
                "merge_dense_delta: base is quantized; a LoRA must be merged before quantization"
                    .into(),
            ),
        }
    }

    /// Merge a precomputed bias delta into the base **bias** (`b += δ`) — the bias-channel
    /// analog of [`merge_dense_delta`](Self::merge_dense_delta), for a ComfyUI/lightx2v `.diff_b`
    /// diff-patch (a full bias delta a low-rank adapter cannot express). `delta` is cast to the base
    /// bias's dtype before the add. Errors only on a base with **no** bias — the caller (the
    /// diff-patch fold) treats that as a surfaced skip, never inventing a bias the reference module
    /// doesn't have.
    ///
    /// **Tier-independent in *coverage*, unlike [`merge_dense_delta`](Self::merge_dense_delta)
    /// (sc-15326).** A quantized base packs only its *weight*: `QuantizedLinear` keeps the bias as a
    /// plain dense vector and its forward is `quantized_matmul(x, wq, …) + b`, so `b += δ_b` is exactly
    /// as correct over a Q4/Q8 base as over a bf16 one. That is what lets Krea Realtime apply a
    /// lightx2v step-distill LoRA's 407 `.diff_b` bias deltas on **every** tier instead of only on the
    /// dense one. Quant tier is a *creative* choice in this product, so an adapter result that varied
    /// with it would be a real defect.
    ///
    /// *Coverage*, not bit-for-bit: every `.diff_b` lands on every tier, but the fold dtype can still
    /// differ, because [`quantize`](Self::quantize) casts the bias to bf16 along with the weight
    /// (PARITY-BF16, sc-2609). A base that was f32 on disk therefore folds in f32 when dense and in
    /// bf16 when packed. Immaterial for the bf16-native Krea DiT, where both sides are bf16 anyway.
    pub fn merge_bias_delta(&mut self, delta: &Array) -> Result<()> {
        let bias = match &mut self.base {
            LinearBase::Dense(l) => &mut l.bias,
            LinearBase::Quantized(q) => &mut q.inner.bias,
            LinearBase::ConvRot(q) => {
                let b = q
                    .bias
                    .as_ref()
                    .ok_or("merge_bias_delta: ConvRot base has no bias")?;
                q.bias = Some(add(b, &delta.as_dtype(b.dtype())?)?);
                return Ok(());
            }
            // W4A8 packs only its WEIGHT: the bias stays a dense float vector that the
            // `((acc·xs)·s_channel) + bias` epilogue adds last, so `b += δ_b` is exactly as correct
            // here as over a packed Q4/Q8 base — sc-15326's tier-independent coverage.
            LinearBase::W4A8(q) => {
                let b = q
                    .bias
                    .as_ref()
                    .ok_or("merge_bias_delta: W4A8 base has no bias")?;
                q.bias = Some(add(b, &delta.as_dtype(b.dtype())?)?);
                return Ok(());
            }
        };
        match bias.value.as_ref() {
            Some(b) => {
                let merged = add(b, &delta.as_dtype(b.dtype())?)?;
                *bias = Param::new(Some(merged));
                Ok(())
            }
            None => Err(
                "merge_bias_delta: base linear has no bias; a `.diff_b` cannot fold into a bias-free module"
                    .into(),
            ),
        }
    }

    /// The base's dense bias, packed or not — the tier-independent half of a diff-patch target.
    /// `Some` for both a dense and a quantized base (a `QuantizedLinear` never packs its bias), so a
    /// `.diff_b` shape check does not have to first ask whether the weight is packed (sc-15326).
    pub fn bias(&self) -> Option<&Array> {
        match &self.base {
            LinearBase::Dense(l) => l.bias.value.as_ref(),
            LinearBase::Quantized(q) => q.inner.bias.value.as_ref(),
            LinearBase::ConvRot(q) => q.bias.as_ref(),
            LinearBase::W4A8(q) => q.bias.as_ref(),
        }
    }

    /// Cast the dense base weight (and bias) to `dtype` in place — the training-time compute-dtype
    /// switch (sc-4887: bf16 mixed-precision training over an f32-on-disk checkpoint). Quantized
    /// bases are left untouched (their compute dtype is fixed by the packed format). Destructive for
    /// a narrowing cast (f32→bf16 drops mantissa bits): reload the model to get the f32 weights back.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        if let LinearBase::Dense(l) = &mut self.base {
            if l.weight.value.dtype() != dtype {
                l.weight = Param::new(l.weight.value.as_dtype(dtype)?);
            }
            if let Some(b) = l.bias.value.as_ref() {
                if b.dtype() != dtype {
                    l.bias = Param::new(Some(b.as_dtype(dtype)?));
                }
            }
        }
        Ok(())
    }

    /// The dense base weight's dtype, or `None` for a quantized base. Lets a forward stay
    /// dtype-following (cast its inputs to the weight's compute dtype) without assuming f32.
    pub fn weight_dtype(&self) -> Option<Dtype> {
        match &self.base {
            LinearBase::Dense(l) => Some(l.weight.value.dtype()),
            LinearBase::Quantized(_) | LinearBase::ConvRot(_) | LinearBase::W4A8(_) => None,
        }
    }

    /// `true` once the base has been quantized (Q4/Q8).
    pub fn is_quantized(&self) -> bool {
        matches!(
            self.base,
            LinearBase::Quantized(_) | LinearBase::ConvRot(_) | LinearBase::W4A8(_)
        )
    }

    /// Diagnostic accessor: the dense base's `(weight, bias)`, or `None` if already quantized.
    /// Used by the sc-2604 diagnostic to inspect the loaded weight dtype before quantization.
    pub fn dense_weight(&self) -> Option<(&Array, Option<&Array>)> {
        match &self.base {
            LinearBase::Dense(l) => Some((&l.weight.value, l.bias.value.as_ref())),
            LinearBase::Quantized(_) | LinearBase::ConvRot(_) | LinearBase::W4A8(_) => None,
        }
    }

    /// Diagnostic accessor: the quantized base's `(packed_weight, scales, biases, bias, group_size,
    /// bits)`, or `None` if the base is still dense. Used by the sc-2604 Q8 root-cause diagnostic to
    /// byte-compare the *loaded* model's quantization against the fork's `mx.quantize` (the
    /// `qmm_smallk` probe only exercised the free `quantize` op, not `try_from_linear`).
    #[allow(clippy::type_complexity)]
    pub fn quantized_params(&self) -> Option<(&Array, &Array, &Array, Option<&Array>, i32, i32)> {
        match &self.base {
            LinearBase::Quantized(q) => Some((
                &q.inner.weight.value,
                &q.scales.value,
                &q.biases.value,
                q.inner.bias.value.as_ref(),
                q.group_size,
                q.bits,
            )),
            LinearBase::Dense(_) | LinearBase::ConvRot(_) | LinearBase::W4A8(_) => None,
        }
    }

    /// The base weight's logical `[out, in]` shape — what a LoKr delta must reshape to.
    /// For a quantized base the packed weight is opaque, so recover it from the scales grid
    /// (`[out, in/group_size]`) times the group size.
    pub fn base_shape(&self) -> Vec<i32> {
        match &self.base {
            LinearBase::Dense(l) => l.weight.value.shape().to_vec(),
            LinearBase::Quantized(q) => {
                // Recover `in = scales_cols · group_size`. Exact only when `in % group_size == 0`,
                // which always holds for the group-quantized linears here (in is a multiple of the
                // group size by construction) (F-089).
                let s = q.scales.value.shape();
                vec![s[0], s[1] * q.group_size]
            }
            LinearBase::ConvRot(q) => vec![
                q.descriptor.out_features as i32,
                q.descriptor.in_features as i32,
            ],
            LinearBase::W4A8(q) => vec![
                q.descriptor.out_features as i32,
                q.descriptor.in_features as i32,
            ],
        }
    }

    /// Quantize the dense base in place to Q4/Q8 (`group_size` defaults to 64), the mlx-rs
    /// equivalent of `nn.quantize` over this Linear. No-op if already quantized. Adapters are
    /// forward-time residuals over the (now quantized) base, so they are unaffected — this is
    /// why the base is never fused: fusing would force re-quantization on every adapter swap.
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        if matches!(self.base, LinearBase::ConvRot(_)) {
            return Err(
                "ConvRot INT8 refuses reinterpretation or re-quantization as affine Q4/Q8".into(),
            );
        }
        if matches!(self.base, LinearBase::W4A8(_)) {
            return Err(
                "packed W4A8 refuses reinterpretation or re-quantization as affine Q4/Q8".into(),
            );
        }
        if let LinearBase::Dense(l) = &self.base {
            // PARITY-BF16 (sc-2609): downcast for fork parity. f32 quantization (f32 group scales)
            // is *more* accurate; we cast to bf16 only to byte-match the fork's golden. Flip to f32
            // for quality once parity is no longer the goal — f32 is safe (the qmm path never hits
            // the bf16-GEMM bug). Rationale below.
            //
            // The fork (mflux) loads every weight at bf16 — its compute dtype — and quantizes THAT.
            // Some checkpoints (e.g. Z-Image-Turbo's transformer) ship f32 on disk; quantizing the
            // as-loaded f32 weight yields group `scales` that differ from the fork's bf16 scales by
            // ~0.13% (the integer `wq` codes and `biases` survive the perturbation, the scales do
            // not), which compounds into the base-model Q8/Q4 e2e residual (sc-2604). Cast weight +
            // bias to bf16 first so the packing is byte-identical to the fork. No-op when already
            // bf16 (e.g. Qwen, whose checkpoint is bf16-native — which is why its Q8 already matched).
            let weight = l.weight.value.as_dtype(Dtype::Bfloat16)?;
            let bias = l
                .bias
                .value
                .as_ref()
                .map(|b| b.as_dtype(Dtype::Bfloat16))
                .transpose()?;
            let linear = MlxLinear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            };
            let q = QuantizedLinear::try_from_linear(
                linear,
                group_size.unwrap_or(crate::quant::DEFAULT_GROUP_SIZE),
                bits,
            )?;
            self.base = LinearBase::Quantized(q);
        }
        Ok(())
    }

    /// Whether this Linear's logical in-features ([`base_shape`](Self::base_shape)`[1]`) are a
    /// multiple of `group_size` (default [`DEFAULT_GROUP_SIZE`](crate::quant::DEFAULT_GROUP_SIZE)),
    /// i.e. whether [`quantize_if_aligned`](Self::quantize_if_aligned) would (or did) pack it.
    /// Quantization-count diagnostics use this so they agree with the quantize predicate (sc-27320).
    pub fn is_quant_aligned(&self, group_size: Option<i32>) -> bool {
        crate::quant::is_quant_aligned(
            self.base_shape()[1],
            group_size.unwrap_or(crate::quant::DEFAULT_GROUP_SIZE),
        )
    }

    /// Quantize the base to `bits` **iff** it is still dense and its in-features divide the group
    /// size ([`is_quant_aligned`](Self::is_quant_aligned)); otherwise leave it untouched. Returns
    /// whether it quantized. The shared form of the "quantize iff divisible" predicate the per-model
    /// quantize cascades each re-implemented (sc-27320); the packing itself is
    /// [`quantize`](Self::quantize), so the bytes are unchanged.
    pub fn quantize_if_aligned(&mut self, bits: i32, group_size: Option<i32>) -> Result<bool> {
        if !matches!(self.base, LinearBase::Dense(_)) || !self.is_quant_aligned(group_size) {
            return Ok(false);
        }
        self.quantize(bits, group_size)?;
        Ok(true)
    }

    /// Accumulate the adapter stack over an already-computed base output, in the **host's** dtype
    /// (sc-15265). Two rules, both about the host staying the host:
    ///
    /// 1. A [disabled](Adapter::is_disabled) adapter (`scale == 0`) is skipped outright, so `out`
    ///    is returned as the base produced it — installing an adapter and setting its scale to 0
    ///    is byte-identical to never installing one.
    /// 2. A live adapter's residual is cast to `out`'s dtype **before** the add.
    ///
    /// Rule 2 is the consequential one and is not confined to scale 0. Adapter factors carry their
    /// on-disk dtype and [`Adapter::residual`] deliberately does not force one (sc-2718,
    /// fork-faithful); the goldens ship **f32** factors, so on a **bf16** host `add(bf16, f32)`
    /// promoted the Linear's output to f32 — and from there the whole downstream chain, since
    /// every subsequent op inherits the widened activation. That is a global precision/allocation
    /// change triggered by merely installing a LoRA, measured e2e at ≈1.9e-4 (dense) / ≈2.9e-4
    /// (Q4) versus the unadapted model. Training already had to work around it per-call-site
    /// (`install_training_lora_as`'s compute-dtype cast, sc-4887, whose doc names exactly this
    /// "silently re-promotes the whole chain to f32"); this fixes it once at the shared seam.
    ///
    /// The residual itself is still computed in its natural promoted dtype — only the *hand-off*
    /// to the host is narrowed — so an f32-activation host (FLUX.2, Qwen's image stream) is
    /// completely unaffected: `out` is already f32 and the cast is a no-op.
    ///
    /// **What rule 2 costs, stated honestly.** It is not free and it is not an accuracy *gain* — it
    /// moves the adapted Linear AWAY from a fully-f32 reference, not toward it. Two independent
    /// seam-level measurements (bf16 host, f32 rank-4 factors, relative L2 vs an all-f32 reference):
    ///
    /// | fixture   | unadapted bf16 base floor | err before | err after |
    /// |-----------|---------------------------|------------|-----------|
    /// | 64-wide   | 6.3014e-4                 | 6.3066e-4  | 7.4336e-4 |
    /// | 1024-wide | 1.95376e-3                | 1.95376e-3 | 2.0238e-3 |
    ///
    /// The load-bearing observation is that `err(before)` equals the unadapted base's own bf16 floor
    /// to five/six digits: the promoted residual was essentially exact and contributed ~nothing on
    /// top of that floor. Narrowing it costs a further `≈4.7e-4` / `≈5.3e-4` relative. Expressed as a
    /// *fraction of the adapter's own effect* at that Linear the number is NOT stable — ~19% at the
    /// first fixture, ~92% at the second — because it depends entirely on how large the residual is
    /// relative to the host's bf16 ulp; quote it and you will mislead yourself. The claim that IS
    /// stable is the absolute bound. The narrowing path rounds **twice** — once casting the residual
    /// to the host dtype, and once in the host-dtype `add` — each at most half an ulp, so the
    /// constructive bound is **at most about one ulp of the host dtype**, not half. (The measured
    /// extra `≈4.7e-4` / `≈5.3e-4` relative above is consistent with ~1 ulp: bf16's relative ulp is
    /// ≈3.9e-3.) That is still **a bounded accuracy loss of the same order as the host's own bf16
    /// rounding, with a non-systematic end-to-end sign** — the qualitative conclusion and every
    /// measured number are unaffected by the correction. The krea e2e tier sweep
    /// bears the second half out: −0.17% (bf16), −1.1% (Q8), **+**1.5% (Q4) — non-monotonic,
    /// consistent with chaotic propagation of rounding rather than a directional bias.
    ///
    /// **User-visible consequence at inference.** Rule 2's dead zone is not confined to training:
    /// on a bf16 host a sufficiently low user LoRA strength is now quantized to the host dtype
    /// rather than accumulated in f32, so it can round away *entirely* where pre-fix it had a small
    /// but visible f32 effect. The measured threshold is roughly half a bf16 ulp of the host output
    /// — i.e. a residual/output ratio of about **2e-3** or less contributes nothing. This is the
    /// accepted diffusers convention and is not a defect, but it is the answer to "my LoRA at very
    /// low strength does nothing now": raise the strength, or run an f32-activation host.
    ///
    /// This is the diffusers convention (LoRA cast to the model dtype); it diverges from the frozen
    /// fork, which adds the promoted residual. **No image-level A/B on a shipped LoRA was run**, and
    /// the Z-Image / Qwen LoRA+LoKr **fork-parity goldens named in [`Adapter::residual`] were not
    /// re-run** against this narrowing — they need a licensed host. Both are open follow-ups, not
    /// claims. What is bought is the invariant above: installing an adapter no longer silently
    /// re-dtypes the model, and "installed at scale 0" is genuinely nothing.
    ///
    /// `candle-gen`'s `quant::adapt` carries the same defect at its own seam and is NOT fixed here
    /// (tracked as sc-15444).
    fn apply_adapters_with_scales(
        &self,
        x: &Array,
        mut out: Array,
        scales: Option<&ScaleContext<'_>>,
    ) -> Result<Array> {
        for (idx, adapter) in self.adapters.iter().enumerate() {
            // sc-15265 rule 0 — the TRAINING exemption. A training stack
            // ([`set_training_adapters`]) opts out of both rules and runs the pre-sc-15265 add
            // verbatim, so this change is inference-only and the shipped trainers are
            // bit-identical. Reasons, in order:
            //   * Neither rule is *for* training. Rule 1 exists so "installed at scale 0" is a
            //     no-op; a trainer always installs at `scale = 1` and has no disabled case.
            //   * Rule 2 would change shipped trainer numerics, and this is an inference-scoped
            //     fix. Wan installs its **f32 master** factors unchanged (`install_training_lora`
            //     passes `dtype = None`; see
            //     `mlx_gen_wan::transformer::WanTransformer::forward_train_checkpointed`) over a
            //     bf16 block stream. `b` initializes to zeros and the residual grows *from* zero,
            //     so for an initial stretch it sits below the bf16 ulp of the base and narrowing
            //     `base + r` to bf16 would round the adapter contribution away entirely. That
            //     FORWARD dead zone is real and measured at the seam: at `|b|` up to 1e-4 the
            //     residual peaks at 3.3e-7 and the narrowed output is bit-identical to the
            //     unadapted base.
            //     It does NOT, however, stall the optimizer — that would be an overclaim. The
            //     `astype`/`add` VJP is straight-through, so gradient still reaches the LoRA
            //     parameters through the narrowing path: `grad` of a squared-sum loss at `b == 0`
            //     measures max|g| = 4.42e-2, and `b` grows out of the dead zone. The justification
            //     for the exemption is therefore the narrower and sufficient one — it keeps the
            //     shipped trainers **bit-identical** while this change stays inference-only, and
            //     avoids a forward dead zone whose end-to-end training effect was never measured.
            //     The wider accumulation is the master-weights pattern working as intended.
            //   * The alternative — casting the factors to bf16 at the install site — moves the
            //     low-rank matmul itself into bf16, which is a LARGER departure from shipped
            //     trainer numerics than doing nothing, so it cannot be justified as conservative.
            // No before/after Wan training run was measured (a 14B run is not a CI-scale
            // experiment); this exemption is what makes that measurement unnecessary.
            let eff_scale = scales.map_or_else(
                || adapter.default_scale(),
                |ctx| ctx.scale_for(idx, adapter.default_scale()),
            );
            if self.training_residual {
                out = add(&out, &adapter.residual_with_scale(x, eff_scale)?)?;
                continue;
            }
            if eff_scale == 0.0 {
                continue;
            }
            let residual = adapter.residual_with_scale(x, eff_scale)?;
            let residual = if residual.dtype() == out.dtype() {
                residual
            } else {
                residual.as_dtype(out.dtype())?
            };
            out = add(&out, &residual)?;
        }
        Ok(out)
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_with_scales(x, None)
    }

    /// Evaluate `base(x) + Σ adapter.residual(x)` with optional request-scoped [`ScaleContext`]
    /// overrides (`PIPE-14`, `BACK-1`, `BACK-4`), keeping shared resident weights and adapters
    /// immutable.
    pub fn forward_with_scales(
        &self,
        x: &Array,
        scales: Option<&ScaleContext<'_>>,
    ) -> Result<Array> {
        let out = self.base.forward(x)?;
        self.apply_adapters_with_scales(x, out, scales)
    }

    /// [`forward`](Self::forward) with the base reading `base_x` and the adapter stack reading
    /// `x`: for a host whose base was built over a zero-padded input (an unaligned projection
    /// packed at a group-multiple width — `[x, 0] · [W, 0]ᵀ = x · Wᵀ`) while its adapters keep the
    /// checkpoint's input width.
    pub fn forward_split(&self, base_x: &Array, x: &Array) -> Result<Array> {
        self.forward_split_with_scales(base_x, x, None)
    }

    /// [`forward_split`](Self::forward_split) with optional request-scoped [`ScaleContext`]
    /// overrides (`PIPE-14`).
    pub fn forward_split_with_scales(
        &self,
        base_x: &Array,
        x: &Array,
        scales: Option<&ScaleContext<'_>>,
    ) -> Result<Array> {
        let out = self.base.forward(base_x)?;
        self.apply_adapters_with_scales(x, out, scales)
    }

    /// Evaluate with dense base weights widened to the activation dtype for this operation.
    /// Packed bases already compute from the activation dtype and are left packed. This supports
    /// bf16-resident/f32-compute text encoders without permanently widening their weight store.
    pub fn forward_upcast(&self, x: &Array) -> Result<Array> {
        self.forward_upcast_with_scales(x, None)
    }

    /// [`forward_upcast`](Self::forward_upcast) with optional request-scoped [`ScaleContext`]
    /// overrides (`PIPE-14`).
    pub fn forward_upcast_with_scales(
        &self,
        x: &Array,
        scales: Option<&ScaleContext<'_>>,
    ) -> Result<Array> {
        let out = match &self.base {
            LinearBase::Dense(l) => {
                let weight = l.weight.value.as_dtype(x.dtype())?;
                match l.bias.value.as_ref() {
                    Some(bias) => addmm(&bias.as_dtype(x.dtype())?, x, weight.t(), 1.0, 1.0)?,
                    None => matmul(x, weight.t())?,
                }
            }
            LinearBase::Quantized(_) | LinearBase::ConvRot(_) | LinearBase::W4A8(_) => {
                self.base.forward(x)?
            }
        };
        self.apply_adapters_with_scales(x, out, scales)
    }
}

/// A dense Conv2d weight (mlx NHWC `[out, kH, kW, in]`) plus its optional bias, that can have a
/// conv-layer LoRA delta merged into it (sc-2919). Convs in this codebase are **merge-only**: they
/// are never quantized and never carry a forward-time residual, so unlike [`AdaptableLinear`] there
/// is no adapter stack or quantized variant — just the mergeable weight and the accessors a forward
/// pass needs. The merge takes a delta in the trained-file NCHW layout and folds it in chaos-safely
/// (`W += δ`), the conv analog of [`AdaptableLinear::merge_dense_delta`].
#[derive(Clone)]
pub struct AdaptableConv2d {
    /// NHWC `[out, kH, kW, in]` — the layout `mlx_gen::nn::conv2d` expects.
    weight: Array,
    bias: Option<Array>,
}

impl AdaptableConv2d {
    /// Wrap an already-NHWC conv weight (`[out, kH, kW, in]`) and optional bias.
    pub fn new(weight_nhwc: Array, bias: Option<Array>) -> Self {
        Self {
            weight: weight_nhwc,
            bias,
        }
    }

    /// The NHWC `[out, kH, kW, in]` weight, to feed `mlx_gen::nn::conv2d`.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// The optional conv bias.
    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }

    /// Merge a conv LoRA `delta` — given in the **trained-file NCHW** `[out, in, kH, kW]` layout (what
    /// [`conv_lora_delta`] returns) — into the stored NHWC weight: transpose NCHW→NHWC, cast to the
    /// weight's dtype, and add (`W += δ`). Reproduces a reference's merged-weight conv forward
    /// bit-for-bit (a residual would differ by ~1 ULP and cascade on the chaos sampler — see
    /// [`AdaptableLinear::merge_dense_delta`]). A zero delta is a bit-exact no-op (`W + 0 == W`).
    pub fn merge_conv_delta(&mut self, delta_nchw: &Array) -> Result<()> {
        // [out, in, kH, kW] → [out, kH, kW, in] to match the stored NHWC weight.
        let delta_nhwc = delta_nchw.transpose_axes(&[0, 2, 3, 1])?;
        self.weight = add(&self.weight, &delta_nhwc.as_dtype(self.weight.dtype())?)?;
        Ok(())
    }

    /// Cast the conv weight (and bias) to `dtype` in place — the conv analog of
    /// [`AdaptableLinear::cast_weights`], for bf16 mixed-precision training (sc-4878/sc-4941). Convs
    /// are never quantized, so this always applies. Destructive for a narrowing cast (reload to get
    /// the f32 weights back).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        if self.weight.dtype() != dtype {
            self.weight = self.weight.as_dtype(dtype)?;
        }
        if let Some(b) = &self.bias {
            if b.dtype() != dtype {
                self.bias = Some(b.as_dtype(dtype)?);
            }
        }
        Ok(())
    }

    /// The conv weight's dtype — lets a forward stay dtype-following without assuming f32.
    pub fn weight_dtype(&self) -> Dtype {
        self.weight.dtype()
    }

    /// Evaluate this dense convolution without evaluating neighboring model weights.
    pub fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval(std::iter::once(&self.weight).chain(self.bias.iter()))?;
        Ok(())
    }
}

/// A read-only snapshot of one adaptable projection — **everything the probe half of the
/// adapter-resolution surface can answer**, carried by value.
///
/// This type is the whole point of [`AdaptableHost::adaptable_facts`] (SC-18319). Before it existed,
/// the only way to ask "is there a projection here, how big is it, is it packed, does it already
/// carry an adapter?" was to take a `&mut AdaptableLinear` through
/// [`adaptable_mut`](AdaptableHost::adaptable_mut) — and once a family's `to_q`/`to_k`/`to_v` live
/// behind a `FusedQkvProjection`, *handing out that `&mut` is
/// exactly what unfuses the block*. A mere scan (the LoRA/LyCORIS/BFL installers' pass 1, a
/// `.diff`-patch existence check, a trainer's target-shape read) would then dismantle the fusion it
/// just built, over every block, for nothing.
///
/// A probe therefore gets facts, never a handle: **mutation is not expressible through this type**,
/// so a probe cannot install, clear, merge, quantize or unfuse anything, and a fused host is free to
/// answer from its packed representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearFacts {
    /// The logical `[out, in]` base shape — [`AdaptableLinear::base_shape`].
    pub base_shape: Vec<i32>,
    /// Whether the base is group-quantized (packed) rather than dense.
    pub is_quantized: bool,
    /// Whether **any** installed adapter is live (`scale != 0`). A disabled adapter reads `false`,
    /// matching `apply_adapters` rule 1 and `FusedQkvProjection`'s
    /// pack predicate, which both treat "installed at scale 0" as "never installed".
    pub has_live_adapters: bool,
    /// How many adapters are on the stack **including disabled ones** — the "is there anything here
    /// at all to read" question, which is what a capture/replay walk needs (anima's block stream) and
    /// which [`has_live_adapters`](Self::has_live_adapters) deliberately does not answer.
    pub adapter_count: usize,
    /// The dense bias's shape, when the projection carries one.
    pub bias_shape: Option<Vec<i32>>,
}

impl LinearFacts {
    /// Snapshot a projection. The `&` receiver is the point: facts are readable without the `&mut`
    /// that would unfuse a packed QKV triple.
    pub fn of(lin: &AdaptableLinear) -> Self {
        Self {
            base_shape: lin.base_shape(),
            is_quantized: lin.is_quantized(),
            has_live_adapters: lin.adapters().iter().any(|a| !a.is_disabled()),
            adapter_count: lin.adapters().len(),
            bias_shape: lin.bias().map(|b| b.shape().to_vec()),
        }
    }
}

/// A module tree that can resolve a dotted parameter path (split into segments) to the
/// [`AdaptableLinear`] living there, so an adapter can be installed onto it. This is the
/// hand-written form of the macro the full adapter framework (sc-2343) will generate.
///
/// # Probe vs mutate (SC-18319)
///
/// The surface is deliberately split in two, and a caller must pick the half that matches its
/// intent:
///
/// * [`adaptable_facts`](Self::adaptable_facts) — **the probe.** "Is there a projection at this
///   path, and what is it like?" Returns [`LinearFacts`] by value.
/// * [`adaptable_mut`](Self::adaptable_mut) — **the mutation.** "Give me the projection so I can
///   change it." Returns `&mut AdaptableLinear`.
///
/// Only the mutation half may unfuse a `FusedQkvProjection`.
/// **A host holding one MUST override `adaptable_facts`** for the paths that resolve into it, and
/// answer from the packed representation
/// (`FusedQkvProjection::part_facts`); the default
/// below routes through `adaptable_mut`, which unfuses. Because the routing is a hand-written tree,
/// every intermediate node on the way down to a fused leaf must forward `adaptable_facts` too — the
/// default's `adaptable_mut` delegation would otherwise take over at the first node that forgot.
/// Each adopting family pins this with a `probe_does_not_unfuse` test.
pub trait AdaptableHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear>;

    /// **The probe half** — resolve a dotted path to a read-only [`LinearFacts`] snapshot, without
    /// taking the `&mut AdaptableLinear` that unfuses a packed QKV triple. See the trait doc.
    ///
    /// The receiver is `&mut self` rather than `&self` only because the routing tree is: a host with
    /// lazily-materialized blocks (anima's block stream) resolves a path by touching that tree. The
    /// *return type* is what makes this half non-mutating — no handle escapes, so no probe can
    /// install, merge, clear, quantize or unfuse anything.
    ///
    /// The default answers through [`adaptable_mut`](Self::adaptable_mut), which is correct for
    /// every host whose projections are plain [`AdaptableLinear`]s and wrong for one holding a
    /// `FusedQkvProjection` — see the trait doc.
    fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        self.adaptable_mut(path).map(|lin| LinearFacts::of(lin))
    }

    /// The dtype a LyCORIS (LoKr / LoHa) install builds its residual in — the reconstructed
    /// `[out, in]` delta and the structured Kronecker factors. The default is bf16 (PARITY-BF16,
    /// sc-2609), which every fork-parity family relies on; a host computing in f32 (e.g. Iris under
    /// `Precision::Fp32`) overrides it so the adapter is not rounded below its own compute.
    fn lycoris_dtype(&self) -> Dtype {
        DEFAULT_LYCORIS_DTYPE
    }

    /// Resolve a dotted path to the [`AdaptableConv2d`] living there, for conv-layer LoRA merging
    /// (sc-2919) — the conv analog of [`adaptable_mut`](Self::adaptable_mut). The default is empty:
    /// only the SDXL U-Net (the one conv-bearing adapter host) overrides it; DiT/MMDiT families
    /// (Z-Image, Qwen, FLUX) have no conv adapter targets (their only convs live in the un-adapted
    /// VAE / patch-embed), so a conv-shaped key applied to them surfaces as skipped, never merged.
    fn adaptable_conv_mut(&mut self, _path: &[&str]) -> Option<&mut AdaptableConv2d> {
        None
    }

    /// Enumerate every adapter target reachable through the kohya `lora_unet_` convention, as
    /// dotted paths in the trained-file (diffusers) naming that [`adaptable_mut`](Self::adaptable_mut)
    /// accepts. Used to build the `flattened → dotted` lookup that disambiguates kohya keys (whose
    /// `.`→`_` flattening cannot be re-split blindly — module names like `to_out.0` / `feed_forward.w1`
    /// already contain underscores). Mirrors the fork's explicit per-target `lora_unet_…` patterns
    /// (sc-2618): block-indexed layer targets only — the families' fork mappings carry no `lora_unet_`
    /// pattern for global targets, which stay reachable via the diffusers/peft dotted form.
    ///
    /// Every returned path MUST resolve via [`adaptable_mut`](Self::adaptable_mut) and the set MUST be
    /// collision-free once flattened (both guarded by tests). The default is empty — a host that does
    /// not override it has no kohya support and a kohya file applied to it surfaces every key as
    /// unmatched (loud), never silently dropped.
    fn adaptable_paths(&self) -> Vec<String> {
        Vec::new()
    }

    /// Enumerate the host's **BFL / ComfyUI** fused→split adapter targets (sc-2743), the orthogonal
    /// axis to the kohya `lora_unet_` flattening of [`adaptable_paths`](Self::adaptable_paths). A
    /// [`BflTarget`] maps one source key spelling (in any of the BFL prefix
    /// conventions — `lora_unet_…`, `diffusion_model.…`, `base_model.model.…`) to a diffusers module
    /// path, optionally row-slicing the up/down factor so a *fused* source linear (BFL `…img_attn.qkv`,
    /// `…linear1`) fans out into the model's *split* targets (`attn.to_q/to_k/to_v`, …). Mirrors the
    /// fork's `Flux2LoRAMapping._get_bfl_*` + the `base_model.model.` global renames.
    ///
    /// The default is empty — only FLUX.2/FLUX.1 expose a BFL surface (Z-Image/Qwen/SDXL have none),
    /// so a BFL file applied to a host without one surfaces every key as unmatched (loud), never
    /// silently dropped. The per-target slices MUST be byte-faithful to `LoraTransforms` (guarded by
    /// tests).
    fn bfl_targets(&self) -> Vec<BflTarget> {
        Vec::new()
    }

    /// Resolve a dotted path to a **non-Linear dense parameter** a ComfyUI/lightx2v diff-patch delta
    /// can fold into — the norm-layer analog of [`adaptable_mut`](Self::adaptable_mut) (sc-15326).
    ///
    /// A step-distill / lightning LoRA for a Wan-family backbone patches its **norms** as well as its
    /// Linears: the lightx2v `Wan2.1-T2V-14B` cfg-step-distill file carries 200 `.diff` weight deltas
    /// on `self_attn.norm_q`/`norm_k`, `cross_attn.norm_q`/`norm_k` and `norm3`, plus 40 `norm3.diff_b`
    /// bias deltas. Those are RMSNorm/LayerNorm parameters, not `AdaptableLinear`s, so they are
    /// unreachable through the Linear surface at any width — and before this existed they were dropped
    /// without a word.
    ///
    /// Every parameter reachable here MUST be dense on **every quant tier** (norm weights and biases
    /// are: the Wan `_quantize_predicate` packs only the per-block attention/FFN Linears), which is
    /// what makes a fold through this surface tier-independent — the property that distinguishes it
    /// from a `.diff` weight fold into a Linear, which cannot happen over a packed base at all.
    ///
    /// The default is `None` for every host: a family with no norm-adaptable surface reports such a
    /// delta as unmatched (loud), never silently dropped.
    fn diff_patch_param_mut(&mut self, _path: &[&str], _part: DiffPatchPart) -> Option<&mut Array> {
        None
    }
}

/// Which half of a module a ComfyUI/lightx2v diff-patch delta addresses: `‹module›.diff` patches the
/// [`Weight`](DiffPatchPart::Weight), `‹module›.diff_b` the [`Bias`](DiffPatchPart::Bias). The
/// selector [`AdaptableHost::diff_patch_param_mut`] takes, so one path resolves both halves of e.g.
/// an affine LayerNorm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffPatchPart {
    /// `‹module›.diff` — the weight/gain parameter.
    Weight,
    /// `‹module›.diff_b` — the bias/offset parameter.
    Bias,
}

/// Prefix each of `host`'s [`AdaptableHost::adaptable_paths`] with `‹prefix›.` — the enumeration
/// analog of a parent's `["‹prefix›", rest @ ..] => sub.adaptable_mut(rest)` delegation, so a
/// composite host can build its full path list from its children's relative ones (sc-2618 kohya).
pub fn prefixed_paths(prefix: &str, host: &impl AdaptableHost) -> Vec<String> {
    host.adaptable_paths()
        .iter()
        .map(|p| format!("{prefix}.{p}"))
        .collect()
}

/// Install an adapter onto the [`AdaptableLinear`] addressed by `dotted` (e.g.
/// `"attention.to_q"`). Errors if the path resolves to no adaptable linear.
pub fn install_adapter(
    host: &mut impl AdaptableHost,
    dotted: &str,
    adapter: Adapter,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    host.adaptable_mut(&parts)
        .ok_or_else(|| format!("no adaptable linear at path: {dotted}"))?
        .push(adapter);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_context_resolves_per_adapter_and_uniform_overrides_without_mutation() {
        let ctx = ScaleContext::new();
        assert_eq!(ctx.pass, 0);
        assert_eq!(ctx.scale_for(0, 0.75), 0.75);

        let overrides = [0.25f32, 0.0, 1.5];
        let per_adapter = ScaleContext::with_adapter_scales(&overrides).with_pass(2);
        assert_eq!(per_adapter.pass, 2);
        assert_eq!(per_adapter.scale_for(0, 1.0), 0.25);
        assert_eq!(per_adapter.scale_for(1, 1.0), 0.0);
        assert_eq!(per_adapter.scale_for(2, 1.0), 1.5);
        assert_eq!(per_adapter.scale_for(3, 0.8), 0.8);

        let uniform = ScaleContext::with_uniform_scale(0.5);
        assert_eq!(uniform.scale_for(0, 0.8), 0.4);

        let combined = ScaleContext {
            adapter_scales: Some(&overrides),
            uniform_scale: Some(2.0),
            pass: 1,
        };
        assert_eq!(combined.scale_for(0, 1.0), 0.5);
        assert_eq!(combined.scale_for(1, 1.0), 0.0);
        assert_eq!(combined.scale_for(3, 0.75), 1.5);
    }

    #[test]
    fn validate_lokr_metadata_scale_rejects_non_positive_or_non_finite() {
        assert!(validate_lokr_metadata_scale(8.0, 4.0).is_ok());
        assert!(validate_lokr_metadata_scale(0.0, 4.0).is_ok());
        assert!(validate_lokr_metadata_scale(8.0, 0.0).is_err());
        assert!(validate_lokr_metadata_scale(8.0, -1.0).is_err());
        assert!(validate_lokr_metadata_scale(f32::NAN, 4.0).is_err());
        assert!(validate_lokr_metadata_scale(8.0, f32::INFINITY).is_err());
    }

    #[test]
    fn lokr_structured_layout_validates_kronecker_factor_dimensions() {
        assert_eq!(
            lokr_structured_layout(&[64, 128], &[8, 16], &[8, 8], false),
            Some([8, 8, 16, 8])
        );
        assert_eq!(
            lokr_structured_layout(&[64, 128], &[8, 16], &[8, 8], true),
            None
        );
        assert_eq!(
            lokr_structured_layout(&[64, 128], &[8, 15], &[8, 8], false),
            None
        );
    }
}
