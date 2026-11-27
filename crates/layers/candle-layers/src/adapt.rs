//! Shared residual-capable linear for the **forward-time additive (unmerged) LoRA / LoKr** path
//! (sc-11091, epic 10765) — the one core [`AdaptLinear`] that the per-crate copies in
//! `candle-gen-wan/src/quant.rs` (sc-10094) and `candle-gen-anima/src/adapt.rs` (sc-10640) collapse
//! into, and the first-class seam a new consumer (qwen-image-edit Lightning on a packed q4/q8 tier)
//! adopts instead of hand-copying a third time. The candle twin of mlx-gen's `AdaptableLinear`.
//!
//! A frozen **base** — dense (`candle_nn::Linear`) or an MLX-**packed** [`super::QLinear`] that
//! dequantizes-on-forward (sc-7702) — plus zero or more **forward-time additive residuals**
//! `y = base(x) + Σ scale·((x·A)·B)`. Two residual forms, both **memory-free on a packed tier**:
//!   * `Lora` — `scale·((x·a)·b)`, two small matmuls (`a = downᵀ [in,rank]`, `b = upᵀ·(alpha/rank)
//!     [rank,out]`); never the `[out,in]` product, so a q4 base keeps its q4 footprint.
//!   * `LokrStructured` — the Kronecker vec-trick `vec(w1·reshape(x)·w2ᵀ)` (the candle port of
//!     mlx-gen's `Adapter::LokrStructured`), which applies a LoKr WITHOUT ever forming the `[out,in]`
//!     delta — the one path Wan's old copy lacked (it fell back to a dense `[out,in]` delta, packed-
//!     rejected).
//!
//! A full-weight diff or an already reconstructed LoHa/LoKr can also use one `MatrixDelta`
//! projection while the external checkpoint base retains its native activation policy.
//!
//! **The base weight is NEVER mutated.** On a packed q4/q8 DiT that is the whole point: the packed
//! codes survive load (`is_packed()` stays true, no dense `[out,in]` weight is materialized), and each
//! adapter rides *unmerged* as small matmuls per forward. On a **dense** base the fold path
//! (`crate::train::merge`) is still preferred for real runs, because a merge into a real `.weight`
//! is byte-for-byte what the chaos-sensitive samplers' goldens expect (`(W+δ)·x ≠ W·x + δ·x` to ~1
//! ULP — see `crate::train::lora::reconstruct_lora_delta`); the additive branch equals the fold to
//! f32 tolerance and is the *only* viable path where a merge is impossible (a quant-resident weight).
//! With **no** adapter attached the forward is byte-identical to the bare base, so swapping this in for
//! a projection leaves the plain-model / dense-fold paths unchanged.

use std::sync::{Arc, Mutex, OnceLock};

use crate::quant::ComponentFold;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear as CandleNnLinear, Module, VarBuilder};

use crate::quant::{
    lin_gs, reject_rowwise_as_legacy_quant, DenseLinear, MatmulStrategy, Nvfp4Linear, QLinear,
    MLX_GROUP_SIZE,
};
use crate::{convrot_int8, w4a8_int8, CandleError, Result};
use gen_core::Quant;

/// Canonical backend linear projection type alias on the shared `candle-layers` spine (`BACK-1`, `BACK-4`).
pub type Linear = AdaptLinear;

pub use gen_core::phase::ScaleContext;

struct ReferenceMathHooks {
    on_apply_factor: fn() -> bool,
    is_reference_math: fn() -> bool,
}

static REFERENCE_MATH_HOOKS: OnceLock<ReferenceMathHooks> = OnceLock::new();

/// Register `candle-gen`'s thread-local training comparison reference-math predicates without an
/// upward crate dependency. Safe to call repeatedly; the first registration wins.
pub fn register_reference_math_hook(
    on_apply_factor: fn() -> bool,
    is_reference_math: fn() -> bool,
) {
    let _ = REFERENCE_MATH_HOOKS.set(ReferenceMathHooks {
        on_apply_factor,
        is_reference_math,
    });
}

#[inline]
fn on_apply_factor_reference_math() -> bool {
    REFERENCE_MATH_HOOKS
        .get()
        .is_some_and(|h| (h.on_apply_factor)())
}

#[inline]
fn is_reference_math() -> bool {
    REFERENCE_MATH_HOOKS
        .get()
        .is_some_and(|h| (h.is_reference_math)())
}

/// Apply a two-dimensional factor without materializing one copy per leading batch.
pub fn apply_factor(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    if on_apply_factor_reference_math() || w.rank() != 2 || x.rank() <= 2 {
        return x.broadcast_matmul(w);
    }
    let dims = x.dims();
    let mut output = dims[..dims.len() - 1].to_vec();
    let rows: usize = output.iter().product();
    output.push(w.dim(1)?);
    x.reshape((rows, dims[dims.len() - 1]))?
        .matmul(w)?
        .reshape(output)
}

/// Scale the rank-sized intermediate only in f32. Half precision keeps the reference
/// output-scale rounding/underflow behavior; live factors are always supplied by the caller.
pub fn scaled_lora(
    x: &Tensor,
    down: &Tensor,
    up: &Tensor,
    scale: f64,
) -> candle_core::Result<Tensor> {
    let rank = apply_factor(x, down)?;
    if is_reference_math() {
        return apply_factor(&rank, up)? * scale;
    }
    if scale == 1.0 {
        // The common alpha == rank convention needs no scaling kernel at any dtype.
        apply_factor(&rank, up)
    } else if x.dtype() == DType::F32 && down.dim(1)? < up.dim(1)? && scale.is_finite() {
        apply_factor(&(rank * scale)?, up)
    } else {
        apply_factor(&rank, up)? * scale
    }
}

/// The frozen base weight — **dense** (`candle_nn::Linear`), **MLX-packed** ([`super::QLinear`],
/// dequant-on-forward), or **NVFP4** ([`Nvfp4Linear`], E2M1 block-16 + UE4M3 scales served
/// either through the FP4 tensor-core GEMM or its transparent dequant→bf16 fallback — sc-21483).
/// All three compute `x·Wᵀ (+ b)`; none is ever mutated by an adapter.
///
/// The NVFP4 leg is held behind an [`Arc`] because [`Nvfp4Linear`] owns a staged device
/// weight (and, on cuda, a cuBLASLt resident) that must not be deep-copied when a job-local DiT
/// clones the resident trunk to install its own residual stack. The base is read-only by contract,
/// so sharing it is exactly right: cloning an [`AdaptLinear`] to adapt it per job costs a refcount,
/// never a re-pack.
#[derive(Clone)]
pub enum Base {
    Dense(CandleNnLinear),
    Packed(QLinear),
    Nvfp4(std::sync::Arc<Nvfp4Linear>),
    ConvRot(Arc<convrot_int8::ConvRotLinear>),
    W4A8(Arc<w4a8_int8::W4A8Linear>),
    #[cfg(feature = "cuda")]
    Fp8(Arc<crate::eight_bit_linear::Fp8Linear>),
    #[cfg(feature = "cuda")]
    Int8(Arc<crate::eight_bit_linear::Int8Linear>),
    /// Metadata ONLY for a caller-owned strict packed GEMM. This wrapper owns no
    /// base tensor and can apply residuals only to an externally computed output.
    External {
        compute: DType,
        device: Device,
    },
}

impl std::fmt::Debug for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(l) => f.debug_tuple("Dense").field(l).finish(),
            Self::Packed(_) => f.write_str("Packed(QLinear)"),
            Self::Nvfp4(_) => f.write_str("Nvfp4(Nvfp4Linear)"),
            Self::ConvRot(_) => f.write_str("ConvRot(ConvRotLinear)"),
            Self::W4A8(_) => f.write_str("W4A8(W4A8Linear)"),
            #[cfg(feature = "cuda")]
            Self::Fp8(_) => f.write_str("Fp8(Fp8Linear)"),
            #[cfg(feature = "cuda")]
            Self::Int8(_) => f.write_str("Int8(Int8Linear)"),
            Self::External { compute, device } => f
                .debug_struct("External")
                .field("compute", compute)
                .field("device", &device.location())
                .finish(),
        }
    }
}

/// The one refusal every fold entry point returns on an **NVFP4** base (sc-21483, epic 11037 E2).
/// An NVFP4 projection has exactly one correct answer to "re-quantize yourself": no. Folding it to
/// `Q4`/`Q8` (or dequantizing it to BF16) to make some downstream path fit would silently replace the
/// numeric regime the caller asked for — the exact silent-conversion class this story forbids.
fn nvfp4_refold_refusal(quant: Quant) -> candle_core::Error {
    candle_core::Error::Msg(format!(
        "NVFP4 base: refusing to re-quantize an NVFP4 projection as {quant:?}; an NVFP4 weight is \
         packed from the bf16 master and is never silently converted to another numeric regime. \
         Adapters ride as forward-time additive residuals over the packed base instead."
    ))
}

fn external_base_refusal(action: &str) -> candle_core::Error {
    candle_core::Error::wrap(CandleError::Unsupported(format!(
        "external packed base: cannot {action}; the caller owns the strict packed kernel and weight; use apply_residuals on its computed output"
    )))
}

impl Base {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Base::Dense(l) => l.forward(x),
            Base::Packed(q) => q.forward(x),
            Base::Nvfp4(q) => q.forward(x),
            Base::ConvRot(q) => q.forward(x),
            Base::W4A8(q) => q.forward(x),
            #[cfg(feature = "cuda")]
            Base::Fp8(q) => q.forward(x),
            #[cfg(feature = "cuda")]
            Base::Int8(q) => q.forward(x),
            Base::External { .. } => Err(external_base_refusal("compute the base forward")),
        }
    }

    /// [`Self::forward`] with a **dense** base weight (and bias) upcast to `x`'s dtype per call — the
    /// storage-dtype ≠ compute-dtype regime (bf16 weights, f32 activations, sc-12828). Only the one
    /// weight in flight is transiently materialized at the compute dtype, so the resident footprint
    /// stays at the storage dtype's. Inert when the base already matches `x`'s dtype (`Tensor::to_dtype`
    /// short-circuits to an `Arc` clone), so byte-identical to [`Self::forward`]. The **packed** base
    /// already dequantizes to the activation dtype, so it delegates unchanged.
    pub fn forward_upcast(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Base::Dense(l) => {
                let dt = x.dtype();
                let w = l.weight().to_dtype(dt)?;
                let b = l.bias().map(|b| b.to_dtype(dt)).transpose()?;
                __model_proof_output!(CandleNnLinear::new(w, b).forward(x))
            }
            Base::Packed(q) => q.forward(x),
            // NVFP4 already produces its output in the activation dtype (the packed weight is never
            // stored at the activation dtype in the first place), so there is nothing to upcast.
            Base::Nvfp4(q) => q.forward(x),
            Base::ConvRot(q) => q.forward(x),
            Base::W4A8(q) => q.forward(x),
            #[cfg(feature = "cuda")]
            Base::Fp8(q) => q.forward(x),
            #[cfg(feature = "cuda")]
            Base::Int8(q) => q.forward(x),
            Base::External { .. } => Err(external_base_refusal("compute the base forward")),
        }
    }

    pub fn is_packed(&self) -> bool {
        matches!(self, Base::Packed(_) | Base::ConvRot(_) | Base::W4A8(_))
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Base::Nvfp4(_))
    }

    pub fn is_fp8(&self) -> bool {
        if let Base::Packed(q) = self {
            use gen_core::checkpoint_codec::{
                FP8_E4M3_ROWWISE_CODEC, FP8_E4M3_SCALAR_CODEC, FP8_E5M2_SCALAR_CODEC, MXFP8_CODEC,
            };
            return q.native_format_id().is_some_and(|id| {
                [
                    FP8_E4M3_SCALAR_CODEC,
                    FP8_E5M2_SCALAR_CODEC,
                    MXFP8_CODEC,
                    FP8_E4M3_ROWWISE_CODEC,
                ]
                .iter()
                .any(|codec| codec.codec_id == id)
            });
        }
        #[cfg(feature = "cuda")]
        {
            matches!(self, Base::Fp8(_))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    pub fn is_int8(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, Base::Int8(_))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// The device the frozen base weight lives on — the device a residual factor must already be on
    /// before it can be admitted. `None` on the MLX-packed arm, whose resident weight is a `QTensor`
    /// behind a [`MatmulStrategy`] and exposes no device accessor; an admission check then
    /// verifies shape and dtype only, exactly as it did before this seam existed.
    pub fn device(&self) -> Option<Device> {
        match self {
            Base::Dense(l) => Some(l.device().clone()),
            Base::Packed(q) => q.native_device().cloned(),
            Base::Nvfp4(q) => Some(q.device().clone()),
            Base::ConvRot(q) => Some(q.device().clone()),
            Base::W4A8(q) => Some(q.device().clone()),
            #[cfg(feature = "cuda")]
            Base::Fp8(q) => Some(q.device().clone()),
            #[cfg(feature = "cuda")]
            Base::Int8(q) => Some(q.device().clone()),
            Base::External { device, .. } => Some(device.clone()),
        }
    }

    /// Fold a **dense** base to an MLX-packed base in place (`Q4`/`Q8`), treating an already-packed
    /// base as an idempotent no-op and refusing re-quantization on `Nvfp4`, `ConvRot`, `W4A8`, `Fp8`,
    /// `Int8`, and `External` bases.
    pub fn quantize(&mut self, quant: Quant) -> candle_core::Result<()> {
        self.fold(quant, ComponentFold::DequantDense)
    }

    pub fn quantize_dequant_onto(
        &mut self,
        quant: Quant,
        device: &Device,
    ) -> candle_core::Result<()> {
        self.fold(quant, ComponentFold::DequantDenseOnto(device))
    }

    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> candle_core::Result<()> {
        self.fold(quant, ComponentFold::Int8FastOnto(device))
    }

    /// Apply a component tier while retaining an immutable checkpoint operand and its device binding.
    pub fn quantize_component(
        &mut self,
        quant: Quant,
        fold: ComponentFold<'_>,
    ) -> candle_core::Result<()> {
        reject_rowwise_as_legacy_quant(quant)?;
        match self {
            Base::Packed(q) if q.native_format_id().is_some() => q.quantize_component(quant, fold),
            Base::ConvRot(q) => {
                if fold
                    .target()
                    .is_some_and(|dev| !dev.same_device(q.device()))
                {
                    candle_core::bail!(
                        "ConvRot INT8 refuses cross-device migration of its staged base"
                    );
                }
                Ok(())
            }
            _ => self.fold(quant, fold),
        }
    }

    fn fold(&mut self, quant: Quant, fold: ComponentFold<'_>) -> candle_core::Result<()> {
        reject_rowwise_as_legacy_quant(quant)?;
        match self {
            Base::Packed(q) if q.native_format_id().is_some() => fold.apply(q, quant),
            Base::Packed(QLinear::W4A8(_)) | Base::W4A8(_) => {
                candle_core::bail!("W4A8 refuses re-quantization to {quant:?}")
            }
            Base::Packed(_) => Ok(()),
            Base::Nvfp4(_) => Err(nvfp4_refold_refusal(quant)),
            Base::ConvRot(_) => {
                candle_core::bail!("ConvRot INT8 refuses re-quantization to {quant:?}")
            }
            #[cfg(feature = "cuda")]
            Base::Fp8(_) => candle_core::bail!("FP8 refuses re-quantization to {quant:?}"),
            #[cfg(feature = "cuda")]
            Base::Int8(_) => candle_core::bail!("INT8 refuses re-quantization to {quant:?}"),
            Base::External { .. } => Err(external_base_refusal("re-quantize the base")),
            Base::Dense(l) => {
                let mut q = QLinear::from_dense(DenseLinear::Linear(l.clone()));
                fold.apply(&mut q, quant)?;
                *self = Base::Packed(q);
                Ok(())
            }
        }
    }
}

/// A forward-time additive residual attached to an [`AdaptLinear`] — it never touches the frozen base
/// weight, so it is memory-free on a packed q4/q8 tier. Factors are held **f32** (the merge/train
/// dtype) and cast to the activation dtype per forward (they are tiny, so the cast is cheap).
#[derive(Clone)]
pub enum Adapter {
    /// LoRA residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b` `[rank, out]` (= `upᵀ` with
    /// the `alpha/rank` ratio folded in at resolution). The **deferred two-small-matmul** form — never
    /// the `[out,in]` product — so it stays memory-free on any quant.
    Lora {
        a: Tensor,
        b: Tensor,
        scale: f64,
        /// Frozen adapter factors have no live `Var` leaves. Prepare their compute-dtype casts once
        /// per installed adapter/dtype instead of rebuilding them for every denoise step.
        prepared: PreparedLora,
    },
    /// Training LoRA keeps the canonical `down [rank,in]` / `up [out,rank]` variable leaves and
    /// transposes them inside every forward. Holding a precomputed transpose here would retain an
    /// eager graph node from adapter installation rather than make the owning `Var`s leaves of each
    /// new loss graph, so gradients and subsequent `Var::set` updates would be silently lost.
    TrainableLora {
        down: Tensor,
        up: Tensor,
        scale: f64,
        /// [`AdaptLinear::set_trainable_frozen`]: read detached (storage-sharing) views of the
        /// leaves, so a preview render between optimizer steps builds no autograd graph.
        frozen: bool,
    },
    /// Training LoKr retains live `Var` leaves and reconstructs only its bounded structured factors
    /// inside the current forward graph. This is the LoKr twin of `TrainableLora`.
    TrainableLokr {
        w1: Tensor,
        w2: TrainableLokrW2,
        base_shape: (usize, usize),
        scale: f64,
        /// See `TrainableLora::frozen`.
        frozen: bool,
    },
    /// Structured LoKr residual via the Kronecker vec-trick — the FULL `(alpha/rank)·strength` scale is
    /// baked into [`LokrFactors::w2`], so a LoKr applies WITHOUT ever forming the `[out,in]` delta (the
    /// packed-capable path the whole hoist adds over Wan's old dense-only delta).
    LokrStructured { factors: LokrFactors },
    /// Already scaled, full-rank `[out, in]` residual (diff patch or reconstructed adapter).
    MatrixDelta(MatrixDelta),
}

#[derive(Clone, Debug)]
pub struct MatrixDelta {
    pub weight: Tensor,
}

impl MatrixDelta {
    pub fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(device)?,
        })
    }

    pub fn nbytes(&self) -> usize {
        tensor_bytes(&self.weight)
    }

    pub fn forward_with_scale(&self, x: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
        let output = self.forward(x)?;
        if scale == 1.0 {
            Ok(output)
        } else {
            output * scale
        }
    }
}

impl Module for MatrixDelta {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        CandleNnLinear::new(self.weight.to_dtype(x.dtype())?, None).forward(x)
    }
}

/// The second Kronecker factor of a trainable LoKr: a full `w2` leaf, or PEFT's low-rank
/// `w2_a·w2_b` pair (`use_w2` off). The low-rank product is rebuilt inside every forward — never
/// cached at install — so each loss graph ends at the live `w2_a`/`w2_b` leaves and observes every
/// optimizer update.
#[derive(Clone)]
pub enum TrainableLokrW2 {
    Full(Tensor),
    LowRank { a: Tensor, b: Tensor },
}

/// A trainable leaf as the current forward should read it: the live tensor, or — while the owning
/// projection is frozen for a graph-free preview — a detached view sharing its storage (so it
/// still reflects every in-place `Var::set` the optimizer made).
fn leaf(t: &Tensor, frozen: bool) -> Tensor {
    if frozen {
        t.detach()
    } else {
        t.clone()
    }
}

/// Compute-dtype views of a frozen LoRA's already-oriented factors. This is deliberately absent from
/// `TrainableLora`: caching an eager transpose/cast graph there would detach the next loss from its
/// `Var` leaves and silently hide optimizer updates.
#[derive(Clone, Default)]
pub struct PreparedLora {
    pub by_dtype: Arc<Mutex<Vec<(DType, Tensor, Tensor)>>>,
}

impl PreparedLora {
    pub fn factors(
        &self,
        a: &Tensor,
        b: &Tensor,
        dtype: DType,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let mut prepared = self.by_dtype.lock().map_err(|_| {
            candle_core::Error::Msg("frozen LoRA preparation cache lock poisoned".into())
        })?;
        if let Some((_, a, b)) = prepared.iter().find(|(cached, _, _)| *cached == dtype) {
            return Ok((a.clone(), b.clone()));
        }
        let a = a.to_dtype(dtype)?;
        let b = b.to_dtype(dtype)?;
        prepared.push((dtype, a.clone(), b.clone()));
        Ok((a, b))
    }

    pub fn count(&self) -> usize {
        self.by_dtype.lock().map_or(0, |prepared| prepared.len())
    }

    /// Bytes of the cached copies that own storage. A copy at the factors' own dtype is an `Arc`
    /// clone of them (`Tensor::to_dtype` short-circuits), so it holds nothing new.
    pub fn owned_bytes(&self, source: DType) -> usize {
        self.by_dtype.lock().map_or(0, |prepared| {
            prepared
                .iter()
                .filter(|(dtype, _, _)| *dtype != source)
                .map(|(_, a, b)| tensor_bytes(a) + tensor_bytes(b))
                .sum()
        })
    }

    pub fn clear(&self) -> candle_core::Result<()> {
        self.by_dtype
            .lock()
            .map_err(|_| {
                candle_core::Error::Msg("frozen LoRA preparation cache lock poisoned".into())
            })?
            .clear();
        Ok(())
    }
}

/// Storage bytes of `t` at its dtype.
fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

impl Adapter {
    /// Installed default scale for this adapter.
    #[inline]
    pub fn default_scale(&self) -> f64 {
        match self {
            Adapter::Lora { scale, .. }
            | Adapter::TrainableLora { scale, .. }
            | Adapter::TrainableLokr { scale, .. } => *scale,
            Adapter::LokrStructured { factors } => factors.scale,
            Adapter::MatrixDelta(_) => 1.0,
        }
    }

    /// A scalar-zero LoRA contributes exactly nothing. Detect it before reading either factor: this
    /// keeps disabled adapters byte-identical to the bare host even when an unused factor contains a
    /// non-finite value, and avoids two needless matmuls. Structured LoKr retains its pre-bake scale
    /// solely for the same disabled check.
    pub fn is_zero(&self) -> bool {
        self.default_scale() == 0.0
    }

    /// The residual this adapter contributes, in the activation dtype of `x`.
    pub fn residual(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.residual_with_scale(x, self.default_scale())
    }

    /// Evaluate this adapter's residual under a request-scoped `override_scale` (`PIPE-14`).
    pub fn residual_with_scale(
        &self,
        x: &Tensor,
        override_scale: f64,
    ) -> candle_core::Result<Tensor> {
        match self {
            Adapter::MatrixDelta(delta) => delta.forward_with_scale(x, override_scale),
            Adapter::Lora { a, b, prepared, .. } => {
                let xd = x.dtype();
                let (a, b) = prepared.factors(a, b, xd)?;
                scaled_lora(x, &a, &b, override_scale)
            }
            Adapter::TrainableLora {
                down, up, frozen, ..
            } => {
                let xd = x.dtype();
                let down = leaf(down, *frozen).to_dtype(xd)?;
                let up = leaf(up, *frozen).to_dtype(xd)?;
                scaled_lora(x, &down.t()?, &up.t()?, override_scale)
            }
            Adapter::TrainableLokr {
                w1,
                w2,
                base_shape,
                frozen,
                ..
            } => {
                let w1 = leaf(w1, *frozen);
                let (full, low_a, low_b) = match w2 {
                    TrainableLokrW2::Full(w) => (Some(leaf(w, *frozen)), None, None),
                    TrainableLokrW2::LowRank { a, b } => {
                        (None, Some(leaf(a, *frozen)), Some(leaf(b, *frozen)))
                    }
                };
                LokrFactors::build(
                    override_scale,
                    *base_shape,
                    Some(&w1),
                    None,
                    None,
                    full.as_ref(),
                    None,
                    low_a.as_ref(),
                    low_b.as_ref(),
                )
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                .ok_or_else(|| {
                    candle_core::Error::Msg("trainable LoKr factors lost 2-D shape".into())
                })?
                .residual(x)
            }
            // The `scale` is already baked into `factors.w2`; rescale only when overridden.
            // A zero-baked `factors.scale == 0.0` destroyed the pre-bake `w2` amplitude, so a
            // nonzero request-scoped override cannot recover the unscaled factors and is refused.
            Adapter::LokrStructured { factors } => {
                if factors.scale == 0.0 && override_scale != 0.0 {
                    return Err(candle_core::Error::Msg(
                        "LokrStructured: cannot apply nonzero ScaleContext override when baked factors.scale is 0.0"
                            .into(),
                    ));
                }
                let y = factors.residual(x)?;
                if override_scale == factors.scale || factors.scale == 0.0 {
                    Ok(y)
                } else {
                    y * (override_scale / factors.scale)
                }
            }
        }
    }

    /// Move this residual's factors onto `device`, in place (the dense-leaf migration seam, sc-11105).
    fn migrate_to(&mut self, device: &Device) -> candle_core::Result<()> {
        match self {
            Adapter::MatrixDelta(delta) => *delta = delta.to_device(device)?,
            Adapter::Lora { a, b, prepared, .. } => {
                *a = a.to_device(device)?;
                *b = b.to_device(device)?;
                prepared.clear()?;
            }
            Adapter::TrainableLora { down, up, .. } => {
                *down = down.to_device(device)?;
                *up = up.to_device(device)?;
            }
            Adapter::TrainableLokr { w1, w2, .. } => {
                *w1 = w1.to_device(device)?;
                match w2 {
                    TrainableLokrW2::Full(w) => *w = w.to_device(device)?,
                    TrainableLokrW2::LowRank { a, b } => {
                        *a = a.to_device(device)?;
                        *b = b.to_device(device)?;
                    }
                }
            }
            Adapter::LokrStructured { factors } => {
                factors.migrate_to(device)?;
            }
        }
        Ok(())
    }
}

/// The two small Kronecker factors of a LoKr delta, kept **unmaterialized** for a deferred structured
/// forward (the candle port of mlx-gen's `LokrFactors`, sc-10713 / epic 10043). `ΔW = scale·kron(w1,
/// w2)` reshapes to the base's `[out, in]`, but the Kronecker–vector identity lets us apply it WITHOUT
/// ever forming that `[out, in]` tensor: with `w1` `[a, c]` and `w2` `[b, d]` (so `out = a·b`,
/// `in = c·d`), the residual `y = x·ΔWᵀ` is
///   `Y = w1 · X · w2ᵀ`  (then flatten row-major `[.., a, b] → [.., out]`),
/// where `X = reshape(x, [.., c, d])`. Two small matmuls (`[a,c]·[..,c,d]` then `·[d,b]`) touch only
/// the factor shapes — never `[out, in]` — so a LoKr applies on a packed q4/q8 base at the same memory
/// profile as a plain LoRA. The row-major kron ordering here matches `crate::train::lora`'s `kron2d`
/// (`out[i·b+k, j·d+l] = w1[i,j]·w2[k,l]`), so the structured residual equals the folded delta. The
/// full `(alpha/rank)·strength` scale is baked into `w2` at build time.
///
/// `Clone`/`Debug` so a caller that stacks these residuals on its own adaptable seam (the SDXL
/// `crate::train::lora::LoraLinear` additive channel, sc-11103) can hold them in a `#[derive(Clone,
/// Debug)]` module without re-implementing the vec-trick — the factors are `Tensor`s (cheap `Arc`
/// clone) plus `usize` shape metadata.
#[derive(Clone, Debug)]
pub struct LokrFactors {
    /// `[a, c]` — the left Kronecker factor (`out = a·b`, `in = c·d`).
    pub w1: Tensor,
    /// `[b, d]` — the right Kronecker factor, with the full scale baked in.
    pub w2: Tensor,
    /// `a` — row count of `w1`; the flattened output index is `p·b + q`.
    pub a: usize,
    /// `b` — row count of `w2`.
    pub b: usize,
    /// `c` — col count of `w1`; the flattened input index is `r·d + s`.
    pub c: usize,
    /// `d` — col count of `w2`.
    pub d: usize,
    /// Optional output-row slice for a fused source projection routed onto one split host projection.
    pub output_slice: Option<(usize, usize)>,
    /// Pre-bake scale retained solely so a disabled structured LoKr can be skipped before reading
    /// factors. The residual math continues to use the scale already baked into `w2`.
    pub scale: f64,
    /// Frozen factors are immutable, so their activation-dtype casts/transposed right leg can be
    /// retained once per compute dtype. Unlike the trainable LoKr variant this cache has no eager
    /// graph connected to `Var` leaves.
    pub prepared: Arc<Mutex<Vec<(DType, Tensor, Tensor)>>>,
}

impl LokrFactors {
    /// Device bytes retained by the structured residual. The shape metadata and scalar scale are
    /// not allocations; the owned factor tensors are the complete LoKr storage payload.
    pub fn nbytes(&self) -> usize {
        self.w1.elem_count() * self.w1.dtype().size_in_bytes()
            + self.w2.elem_count() * self.w2.dtype().size_in_bytes()
    }

    /// Bytes retained by the structured residual after source factors are converted to f32 and any
    /// low-rank inner products are materialized. This is the device allocation installed on an
    /// [`AdaptLinear`], not the adapter container payload.
    pub fn resident_f32_bytes(&self) -> usize {
        self.w1
            .elem_count()
            .saturating_add(self.w2.elem_count())
            .saturating_mul(std::mem::size_of::<f32>())
    }

    /// Build the small `[a,c]`/`[b,d]` Kronecker factors from a LoKr module's factors (full `w1`/`w2`
    /// or a low-rank `w_a·w_b` product — that product is bounded by the factor dims, NEVER `out×in`),
    /// baking the FULL `scale` into `w2`. The allocation-free counterpart to
    /// `crate::train::lora::reconstruct_lokr_delta` (which materializes the full `[out,in]` delta).
    ///
    /// **Scale differs from the fold path by design** (the two-conventions trap, sc-10578): the
    /// materialized `reconstruct_lokr_delta` bakes only `alpha/rank` and rides the user `strength` in
    /// the merge `scale`, whereas the structured residual carries no separate scale field, so the FULL
    /// `(alpha/rank)·strength` must be baked here (the caller derives it). Mismatching these two is a
    /// silent mis-scale, not a crash.
    ///
    /// Returns `Ok(None)` when the module has **no 2-D matrix form** deferrable via the vec-trick — a
    /// tucker/CP `lokr_t2` (conv-only), a factor that is not 2-D, or a base that does not factor as
    /// `a·b × c·d`. The caller then rejects (a packed base cannot materialize) or folds (a dense base).
    /// A missing `w1`/`w2` leg is a typed error (a malformed file), never a panic.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        scale: f64,
        base_shape: (usize, usize),
        w1: Option<&Tensor>,
        w1_a: Option<&Tensor>,
        w1_b: Option<&Tensor>,
        w2: Option<&Tensor>,
        w2_t2: Option<&Tensor>,
        w2_a: Option<&Tensor>,
        w2_b: Option<&Tensor>,
    ) -> Result<Option<Self>> {
        Self::build_inner(
            scale, base_shape, None, w1, w1_a, w1_b, w2, w2_t2, w2_a, w2_b,
        )
    }

    /// Re-bake a request's absolute scale against the F32 right factor from this installation.
    pub fn with_scaled_right(
        &self,
        unscaled_right: &Tensor,
        scale: f64,
    ) -> candle_core::Result<Self> {
        if !scale.is_finite() {
            candle_core::bail!("lokr: derived scale must be finite, got {scale}");
        }
        let w2 = (unscaled_right * scale)?
            .contiguous()?
            .to_device(self.w2.device())?;
        Ok(Self {
            w2,
            scale,
            prepared: Arc::default(),
            ..self.clone()
        })
    }

    /// Build factors for a fused source projection and retain only `output_slice` in the residual.
    /// This routes BFL/ComfyUI fused FLUX QKV LoKr onto split q/k/v host projections without ever
    /// materializing the fused `[out,in]` weight delta.
    #[allow(clippy::too_many_arguments)]
    pub fn build_sliced(
        scale: f64,
        input_features: usize,
        output_slice: (usize, usize),
        w1: Option<&Tensor>,
        w1_a: Option<&Tensor>,
        w1_b: Option<&Tensor>,
        w2: Option<&Tensor>,
        w2_t2: Option<&Tensor>,
        w2_a: Option<&Tensor>,
        w2_b: Option<&Tensor>,
    ) -> Result<Option<Self>> {
        Self::build_inner(
            scale,
            (0, input_features),
            Some(output_slice),
            w1,
            w1_a,
            w1_b,
            w2,
            w2_t2,
            w2_a,
            w2_b,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_inner(
        scale: f64,
        base_shape: (usize, usize),
        output_slice: Option<(usize, usize)>,
        w1: Option<&Tensor>,
        w1_a: Option<&Tensor>,
        w1_b: Option<&Tensor>,
        w2: Option<&Tensor>,
        w2_t2: Option<&Tensor>,
        w2_a: Option<&Tensor>,
        w2_b: Option<&Tensor>,
    ) -> Result<Option<Self>> {
        if !scale.is_finite() {
            return Err(CandleError::Msg(format!(
                "lokr: derived scale must be finite, got {scale}"
            )));
        }
        // A tucker/CP `w2` (a 4-D conv factor, lycoris `lokr_t2`) has no 2-D matrix form — not
        // deferrable via the vec-trick. The peft LoKr format never carries it; guard anyway so a conv
        // LoKr falls back to reject/fold rather than silently mis-applying.
        if w2_t2.is_some() {
            return Ok(None);
        }
        let f32d = |t: &Tensor| t.to_dtype(DType::F32);
        // The small Kronecker factors — full, or the low-rank inner product (bounded by the factor
        // dims, NEVER `out×in`): `w1_a @ w1_b` yields the small `[a, c]`, not the packed delta.
        let factor1 = match (w1, w1_a, w1_b) {
            (Some(w), _, _) => f32d(w)?,
            (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
            _ => {
                return Err(CandleError::Msg(
                    "lokr: w1 missing (need full lokr_w1 or lokr_w1_a·lokr_w1_b)".into(),
                ))
            }
        };
        let factor2 = match (w2, w2_a, w2_b) {
            (Some(w), _, _) => f32d(w)?,
            (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
            _ => {
                return Err(CandleError::Msg(
                    "lokr: w2 missing (need full lokr_w2 or lokr_w2_a·lokr_w2_b)".into(),
                ))
            }
        };
        // A conv-shaped (>2-D) factor is not a plain matrix — defer to reject/fold.
        if factor1.dims().len() != 2 || factor2.dims().len() != 2 {
            return Ok(None);
        }
        let (a, c) = (factor1.dims()[0], factor1.dims()[1]);
        let (b, d) = (factor2.dims()[0], factor2.dims()[1]);
        let (out_f, in_f) = base_shape;
        // The base must factor as `out = a·b`, `in = c·d` (a plain 2-D linear); anything else (a conv
        // weight with kernel dims, or a factor/base mismatch) is not this linear vec-trick.
        let fused_out = a * b;
        let output_matches = match output_slice {
            Some((start, len)) => out_f == 0 && start + len <= fused_out,
            None => fused_out == out_f,
        };
        if !output_matches || c * d != in_f {
            return Ok(None);
        }
        if output_slice.is_some_and(|(start, len)| start + len > fused_out) {
            return Ok(None);
        }
        // Bake the full scale into `w2` (keeps `w1` a clean copy); hold f32, contiguous for the matmuls.
        let w2 = (&factor2 * scale)?.contiguous()?;
        let w1 = factor1.contiguous()?;
        Ok(Some(Self {
            w1,
            w2,
            a,
            b,
            c,
            d,
            output_slice,
            scale,
            prepared: Arc::default(),
        }))
    }

    /// The `[out, in]` projection shape this residual adapts (`out` is the output slice's length
    /// when one is set).
    pub fn shape(&self) -> (usize, usize) {
        let out = self.output_slice.map_or(self.a * self.b, |(_, len)| len);
        (out, self.c * self.d)
    }

    /// The f32 `[out, in]` weight delta this residual applies, `kron(w1, w2)` with the scale baked
    /// into `w2` (the output slice, if any, applied) — a merged **view** for parity checks; the
    /// forward never forms it.
    pub fn dense_delta(&self) -> candle_core::Result<Tensor> {
        let delta = kron2d(
            &self.w1.to_dtype(DType::F32)?,
            &self.w2.to_dtype(DType::F32)?,
        )?;
        match self.output_slice {
            Some((start, len)) => delta.narrow(0, start, len),
            None => Ok(delta),
        }
    }

    /// Move the (CPU-read) factors onto `device` — the base lives on the DiT's device, so the residual
    /// matmul would be a device mismatch otherwise.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            w1: self.w1.to_device(device)?,
            w2: self.w2.to_device(device)?,
            a: self.a,
            b: self.b,
            c: self.c,
            d: self.d,
            output_slice: self.output_slice,
            scale: self.scale,
            // Migration installs a new frozen adapter payload; retain only factors on the target
            // device and lazily prepare its target-dtype views there.
            prepared: Arc::default(),
        })
    }

    pub fn prepared_factors(&self, dtype: DType) -> candle_core::Result<(Tensor, Tensor)> {
        let mut prepared = self.prepared.lock().map_err(|_| {
            candle_core::Error::Msg("frozen LoKr preparation cache lock poisoned".into())
        })?;
        if let Some((_, w1, w2t)) = prepared.iter().find(|(cached, _, _)| *cached == dtype) {
            return Ok((w1.clone(), w2t.clone()));
        }
        let w1 = self.w1.to_dtype(dtype)?;
        let w2t = self.w2.to_dtype(dtype)?.t()?.contiguous()?;
        prepared.push((dtype, w1.clone(), w2t.clone()));
        Ok((w1, w2t))
    }

    fn preparation_count(&self) -> usize {
        self.prepared.lock().map_or(0, |prepared| prepared.len())
    }

    /// Device bytes this structured residual holds right now: its factors ([`Self::nbytes`]) plus
    /// the compute-dtype copies its forwards have cached so far. What an overlay pricing such as
    /// `gen_core::LokrKroneckerDims::resident_bytes` must cover (sc-24163).
    pub fn device_bytes(&self) -> usize {
        self.nbytes() + self.prepared_owned_bytes()
    }

    /// Bytes of the cached compute-dtype copies that own storage: `w1` unless it is the factor's own
    /// dtype (an `Arc` clone then), and the transposed `w2`, which `contiguous` always copies.
    fn prepared_owned_bytes(&self) -> usize {
        let source = self.w1.dtype();
        self.prepared.lock().map_or(0, |prepared| {
            prepared
                .iter()
                .map(|(dtype, w1, w2t)| {
                    let w1 = if *dtype == source {
                        0
                    } else {
                        tensor_bytes(w1)
                    };
                    w1 + tensor_bytes(w2t)
                })
                .sum()
        })
    }

    fn migrate_to(&mut self, device: &Device) -> candle_core::Result<()> {
        self.w1 = self.w1.to_device(device)?;
        self.w2 = self.w2.to_device(device)?;
        self.prepared
            .lock()
            .map_err(|_| {
                candle_core::Error::Msg("frozen LoKr preparation cache lock poisoned".into())
            })?
            .clear();
        Ok(())
    }

    /// The deferred, allocation-free LoKr residual via the Kronecker–vector identity (`scale` already
    /// baked into `w2`). For an activation `x` of shape `[.., in]` (`in = c·d`): reshape to
    /// `[.., c, d]`, compute `Y = w1 · X · w2ᵀ` (`[.., a, b]`), and flatten row-major to `[.., out]`
    /// (`out = a·b`). The two matmuls touch only the small factor shapes — the full `[out, in]` delta
    /// is NEVER materialized, so this holds the same memory profile on a packed q4/q8 base as a plain
    /// LoRA. Factors are cast to the activation dtype so a bf16 stream runs bf16.
    ///
    /// Public so a caller stacking this residual on its own seam (the SDXL `LoraLinear` additive
    /// channel, sc-11103) applies it without reaching through an [`AdaptLinear`].
    pub fn residual(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let xd = x.dtype();
        let (w1, w2t) = self.prepared_factors(xd)?; // [d, b]
        let dims = x.dims();
        let lead = &dims[..dims.len() - 1];
        let n: usize = lead.iter().product::<usize>().max(1);
        // Vec-trick `Y = w1 · X · w2ᵀ`, ordered so the **expensive** contraction (over `d`, the big
        // inner factor) is ONE large-`M` GEMM rather than `N` tiny-`M` batched GEMMs. The prior form
        // `w1b.broadcast_matmul(xr).broadcast_matmul(w2tb)` ran the `·w2ᵀ` leg as a batch of `N`
        // `[a,d]·[d,b]` products with `M = a` — for a factor-4 LoKr `a = 4`, and `N` = the token count.
        // On CUDA that batch of tiny-`M` GEMMs is pathologically slow at long sequences (a Krea
        // image-edit's `N ≈ 6.6k` tokens turned an 8-step turbo edit into a multi-minute grind, and it
        // inflated VRAM), while the flop count barely rises. Fold the `N·c` rows into the GEMM's `M`:
        //   T = X · w2ᵀ : `[N·c, d] · [d, b] → [N·c, b]`  (single GEMM, `M = N·c`),
        // then apply the tiny outer `w1` (contraction over `c` — cheap however it batches). Associativity
        // makes this bit-equal (to f32 rounding) to the old order, so the `structured_lokr_*` deltas hold.
        let t = x
            .contiguous()?
            .reshape((n * self.c, self.d))?
            .matmul(&w2t)? // [N·c, b]
            .reshape((n, self.c, self.b))?;
        let w1b = w1.reshape((1, self.a, self.c))?;
        // Y = w1 · T  → [N, a, b] (batch `N`, but `M = a` / `K = c` are the tiny outer factor ⇒ cheap).
        let y = w1b.broadcast_matmul(&t)?;
        // [N, a, b] → [.., out] (out = a·b), restoring the original leading dims (row-major flatten).
        let mut ys = lead.to_vec();
        ys.push(self.a * self.b);
        let y = y.contiguous()?.reshape(ys)?;
        match self.output_slice {
            Some((start, len)) => y.narrow(y.rank() - 1, start, len),
            None => Ok(y),
        }
    }
}

/// A projection with a frozen base (dense or MLX-packed) and stacked forward-time LoRA/LoKr residuals.
/// Built dense ([`Self::linear`] / [`Self::linear_no_bias`] / [`Self::from_dense`] / [`Self::dense`] /
/// [`Self::dense_bias`]) or packed-detecting ([`Self::linear_detect`] / [`Self::linear_detect_gs`] /
/// [`Self::detect`]). `forward` = `base(x)` plus every residual, in push order; with no adapter it is
/// byte-identical to the bare base.
#[derive(Clone)]
pub struct AdaptLinear {
    pub base: Base,
    /// The projection's logical `(out_features, in_features)` — captured at construction (recoverable
    /// even from a packed base, where the dense weight is never materialized) so the residual installer
    /// can shape-check a factor without reading the quantized weight back.
    pub out_features: usize,
    pub in_features: usize,
    /// Forward-time additive residuals, applied in push order (adapters stack). Empty on the plain /
    /// dense-fold path ⇒ forward is byte-identical to the bare base.
    pub adapters: Vec<Adapter>,
}

impl std::fmt::Debug for AdaptLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptLinear")
            .field("packed", &self.is_packed())
            .field("out_features", &self.out_features)
            .field("in_features", &self.in_features)
            .field("adapters", &self.adapters.len())
            .finish()
    }
}

impl AdaptLinear {
    /// Inspect executable frozen LoRA storage in an explicit model-proof diagnostic.
    /// Never prepares casts, changes caches, or reads the provider's inspection records.
    /// Every unsupported adapter variant refuses the complete observation.
    #[doc(hidden)]
    #[cfg(feature = "model-inference-proof")]
    pub fn __model_proof_frozen_lora_factors(
        &self,
    ) -> candle_core::Result<Vec<(Tensor, Tensor, f64)>> {
        self.adapters
            .iter()
            .map(|adapter| match adapter {
                Adapter::Lora { a, b, scale, .. } => Ok((a.clone(), b.clone(), *scale)),
                Adapter::LokrStructured { .. }
                | Adapter::MatrixDelta(_)
                | Adapter::TrainableLora { .. }
                | Adapter::TrainableLokr { .. } => {
                    candle_core::bail!("frozen LoRA proof observation requires frozen LoRA factors")
                }
            })
            .collect()
    }

    /// Copy a prepared base and every residual onto a bounded execution window.
    pub fn copy_to_device(&self, device: &Device) -> candle_core::Result<Self> {
        let mut copy = self.clone();
        // Ordinary clones deliberately share immutable-factor preparation caches. A bounded
        // device copy has a different owner/lifetime: migration and its subsequent forward
        // must neither clear the host source's cache nor publish device tensors back into it.
        for adapter in &mut copy.adapters {
            match adapter {
                Adapter::Lora { prepared, .. } => *prepared = PreparedLora::default(),
                Adapter::LokrStructured { factors } => factors.prepared = Arc::default(),
                Adapter::MatrixDelta(_)
                | Adapter::TrainableLora { .. }
                | Adapter::TrainableLokr { .. } => {}
            }
        }
        if let Base::Packed(base) = &self.base {
            copy.base = Base::Packed(base.copy_to_device(device)?);
        }
        copy.to_device(device)?;
        Ok(copy)
    }

    /// Checked residual storage for a caller-owned PACKED base. No dummy tensor,
    /// dense weight, quantization, or additional kernel handle is allocated here.
    /// Call [`Self::apply_residuals`] after the caller's strict packed GEMM;
    /// [`Self::forward`] and every folding/requantization entry point refuse it.
    pub fn for_external_base(
        in_dim: usize,
        out_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if in_dim == 0
            || out_dim == 0
            || in_dim.checked_mul(out_dim).is_none()
            || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
        {
            return Err(CandleError::Unsupported("external packed base requires nonzero checked dimensions and a float compute dtype".into()));
        }
        Ok(Self {
            base: Base::External {
                compute: dtype,
                device: device.clone(),
            },
            in_features: in_dim,
            out_features: out_dim,
            adapters: Vec::new(),
        })
    }

    /// Compute metadata only. The packed payload remains exclusively caller-owned.
    pub fn external_base_compute_dtype(&self) -> Option<DType> {
        match &self.base {
            Base::External { compute, .. } => Some(*compute),
            _ => None,
        }
    }

    /// Wrap an already-built dense `candle_nn::Linear` with its logical `[out, in]` dims — the seam a
    /// caller uses when it holds a `Linear` directly (e.g. a folded-delta reference, or a test grid).
    pub fn from_dense(l: CandleNnLinear, in_dim: usize, out_dim: usize) -> Self {
        if l.is_native() {
            return Self::from_packed(
                QLinear::from_dense(super::DenseLinear::Linear(l)),
                in_dim,
                out_dim,
            );
        }
        Self {
            base: Base::Dense(l),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }

    /// Wrap an already-built **packed** base [`super::QLinear`] with its logical `[out, in]` dims — the
    /// raw-tensor twin of [`Self::from_dense`], for a caller that builds the packed base directly from
    /// the MLX triple tensors rather than through a [`VarBuilder`] (the ideogram DiT loader's
    /// `Weights`-based `linear_detect`, sc-11104; the krea loader's `MmapedSafetensors` seam, sc-11105).
    /// The base stays quantized (dequant-on-forward); pushed residuals ride unmerged, so a q4/q8 tier
    /// keeps its footprint.
    pub fn from_packed(base: QLinear, in_dim: usize, out_dim: usize) -> Self {
        Self {
            base: match base {
                QLinear::ConvRot(q) => Base::ConvRot(q),
                other => Base::Packed(other),
            },
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }

    /// Convert a [`QLinear`] into an [`AdaptLinear`] on the shared spine (`SW-6956`, `BACK-1`).
    /// Standard dense `QLinear::Dense(DenseLinear::Linear(l))` unwraps to `Base::Dense(l)` so
    /// dense inspection and dense delta folding remain available; quantized, W4A8, and pre-transposed
    /// projections wrap as `Base::Packed(base)`.
    pub fn from_qlinear(base: QLinear, in_dim: usize, out_dim: usize) -> Self {
        match base {
            QLinear::Dense(DenseLinear::Linear(l)) => Self::from_dense(l, in_dim, out_dim),
            other => Self::from_packed(other, in_dim, out_dim),
        }
    }

    /// Wrap an already-built **NVFP4** [`Nvfp4Linear`] as an adapter-capable base (sc-21483,
    /// epic 11037) — the NVFP4 twin of [`Self::from_packed`]. The logical `[out, in]` dims come from
    /// the packed tensor itself, so no dense weight is read back.
    ///
    /// The point of routing NVFP4 through the ONE shared additive wrapper (sc-11091) rather than
    /// giving it a parallel residual stack: a LoRA/LoKr rides as `scale·((x·a)·b)` (or the Kronecker
    /// vec-trick) *alongside* the packed forward, so the E2M1 codes are never dequantized, never
    /// re-packed, and never converted to q4/BF16 to make an adapter fit. Removing the adapter
    /// ([`Self::clear_adapters`]) restores the base output exactly, because the base was never
    /// touched.
    pub fn from_nvfp4(base: Nvfp4Linear) -> Self {
        let (out_dim, in_dim) = base.shape();
        Self {
            base: Base::Nvfp4(std::sync::Arc::new(base)),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }
    /// Native symmetric INT8 base with shared adapter residual semantics.
    pub fn from_convrot(base: super::convrot_int8::ConvRotLinear) -> Self {
        let d = base.descriptor();
        Self {
            base: Base::ConvRot(Arc::new(base)),
            in_features: d.in_features,
            out_features: d.out_features,
            adapters: Vec::new(),
        }
    }

    /// Distinguish descriptor-driven integer execution from affine packed Q8.
    pub fn is_convrot_int8(&self) -> bool {
        matches!(self.base, Base::ConvRot(_))
            || matches!(&self.base,Base::Packed(q) if q.native_format_id()==Some(gen_core::convrot_int8::CODEC_ID))
    }

    /// Packed W4A8 base; residuals remain additive in the original, unrotated input basis.
    pub fn from_w4a8(base: w4a8_int8::W4A8Linear) -> Self {
        let d = base.descriptor();
        Self {
            base: Base::W4A8(Arc::new(base)),
            in_features: d.in_features,
            out_features: d.out_features,
            adapters: Vec::new(),
        }
    }

    /// Distinguish native codebook W4A8 from affine Q4 and symmetric ConvRot INT8.
    pub fn is_w4a8(&self) -> bool {
        matches!(self.base, Base::W4A8(_))
    }

    /// Wrap a CUDA [`crate::eight_bit_linear::Fp8Linear`] on the shared spine (`SW-6956`).
    #[cfg(feature = "cuda")]
    pub fn from_fp8(base: crate::eight_bit_linear::Fp8Linear) -> Self {
        let (out_dim, in_dim) = base.shape();
        Self {
            base: Base::Fp8(Arc::new(base)),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }

    /// Whether the base is a CUDA `Fp8Linear`.
    pub fn is_fp8(&self) -> bool {
        self.base.is_fp8()
    }

    /// Wrap a CUDA [`crate::eight_bit_linear::Int8Linear`] on the shared spine (`SW-6956`).
    #[cfg(feature = "cuda")]
    pub fn from_int8(base: crate::eight_bit_linear::Int8Linear) -> Self {
        let (out_dim, in_dim) = base.shape();
        Self {
            base: Base::Int8(Arc::new(base)),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }

    /// Whether the base is a CUDA `Int8Linear`.
    pub fn is_int8(&self) -> bool {
        self.base.is_int8()
    }

    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`), shape-
    /// checked exactly like `candle_nn::linear` — so it loads unchanged on `VarMap`-backed test
    /// fixtures.
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self::from_dense(
            candle_nn::linear(in_dim, out_dim, vb)?,
            in_dim,
            out_dim,
        ))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`).
    pub fn linear_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self::from_dense(
            candle_nn::linear_no_bias(in_dim, out_dim, vb)?,
            in_dim,
            out_dim,
        ))
    }

    /// **Packed-detecting** `[out, in]` loader at an explicit MLX `group_size`: if `{base}.scales` is
    /// present under `vb` (a pre-quantized MLX tier), build a `Base::Packed` straight from the packed
    /// parts on `vb`'s device via the shared [`crate::quant::lin_gs`] — **no dense weight is materialized**.
    /// Otherwise the **dense** path is taken unchanged (`{base}.weight` [+ `{base}.bias`], shape-
    /// checked).
    ///
    /// `base` is the full dotted key prefix relative to `vb` (e.g. `to_out.0`), so the
    /// `.scales`/`.biases`/`.bias` siblings survive any `to_out.0`-style nesting: build the base string
    /// first, then detect — never `.pp()` past the scales sibling (the key-remap trap the shared loader
    /// guards).
    pub fn linear_detect_gs(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
        group_size: usize,
    ) -> candle_core::Result<Self> {
        if vb.contains_tensor(&format!("{base}.scales")) {
            return Ok(Self {
                base: Base::Packed(lin_gs(vb, base, in_dim, out_dim, bias, group_size)?),
                out_features: out_dim,
                in_features: in_dim,
                adapters: Vec::new(),
            });
        }
        let sub = vb.pp(base);
        if bias {
            Self::linear(in_dim, out_dim, sub)
        } else {
            Self::linear_no_bias(in_dim, out_dim, sub)
        }
    }

    /// **Packed-detecting** `[out, in]` loader at the default MLX [`MLX_GROUP_SIZE`] (64) — thin wrapper
    /// over [`Self::linear_detect_gs`] for the callers whose hosted tiers pack at 64.
    pub fn linear_detect(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilder,
        base: &str,
        bias: bool,
    ) -> candle_core::Result<Self> {
        Self::linear_detect_gs(in_dim, out_dim, vb, base, bias, MLX_GROUP_SIZE)
    }

    /// Bias-less, **packed-detecting** `[out, in]` projection from `{name}` on `vb` — the variant that
    /// **recovers the logical dims** from the packed `scales` shape (`[out, in/group]`) instead of
    /// taking them as arguments (the Anima DiT loader shape). If `{name}.scales` is present, load the
    /// MLX-packed triple at their native dtypes (u32 codes must NOT be cast through the vb's float
    /// dtype) at group 64; otherwise read the dense `{name}.weight` unchanged.
    pub fn detect(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        let scales_key = format!("{name}.scales");
        if vb.contains_tensor(&scales_key) {
            let device = vb.device().clone();
            let wq = vb.get_unchecked_dtype(&format!("{name}.weight"), DType::U32)?;
            let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
            let biases = vb.get_unchecked_dtype(&format!("{name}.biases"), DType::F32)?;
            // scales is [out, in/group] — recover the logical dims without touching the packed codes.
            let sdims = scales.dims();
            let out_features = sdims[0];
            let in_features = sdims[1] * MLX_GROUP_SIZE;
            let q = QLinear::from_packed_gs(&wq, &scales, &biases, None, MLX_GROUP_SIZE, &device)?;
            Ok(Self {
                base: Base::Packed(q),
                out_features,
                in_features,
                adapters: Vec::new(),
            })
        } else {
            Self::dense(vb, name)
        }
    }

    /// Bias-less **dense** `[out, in]` projection from `{name}.weight` (a component that never packs,
    /// e.g. Anima's conditioner q/k/v/o — dense bf16 in every tier).
    pub fn dense(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        Self::from_builder(vb, name, false)
    }

    pub fn dense_bias(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        Self::from_builder(vb, name, true)
    }

    fn from_builder(vb: &VarBuilder, name: &str, bias: bool) -> candle_core::Result<Self> {
        let sub = vb.pp(name);
        let layer = CandleNnLinear::from_var_builder(&sub, bias)?;
        let (out_features, in_features) = layer.weight_dims()?;
        Ok(Self::from_dense(layer, in_features, out_features))
    }

    /// `x·Wᵀ (+ b)` plus every attached additive residual, in push order. A disabled residual is
    /// skipped entirely; each live residual keeps `Adapter::residual`'s activation-dtype math and is
    /// cast only at the hand-off to the host output dtype before addition. With no live adapter this
    /// is byte-identical to the bare base forward.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_with_scales(x, None)
    }

    /// Evaluate `base(x) + Σ adapter(x)` with optional request-scoped [`ScaleContext`] overrides
    /// (`PIPE-14`, `BACK-1`, `BACK-4`), keeping shared resident weights and adapters immutable.
    pub fn forward_with_scales(
        &self,
        x: &Tensor,
        scales: Option<&ScaleContext<'_>>,
    ) -> candle_core::Result<Tensor> {
        self.apply_residuals_with_scales(x, self.base.forward(x)?, scales)
    }

    /// Evaluate an explicitly selected stack in caller order over the same frozen base. The
    /// provider resolves sparse source slots to this projection's residuals before calling.
    pub fn forward_with_selected_adapters<'a>(
        &self,
        x: &Tensor,
        selected: impl IntoIterator<Item = candle_core::Result<(std::borrow::Cow<'a, Adapter>, f64)>>,
    ) -> candle_core::Result<Tensor> {
        let mut y = self.base.forward(x)?;
        for adapter in selected {
            let (adapter, scale) = adapter?;
            if scale != 0.0 {
                let residual = adapter.residual_with_scale(x, scale)?.to_dtype(y.dtype())?;
                y = (y + residual)?;
            }
        }
        Ok(y)
    }

    /// Dense base inspection without dequantization. Preparation refuses tracked weights.
    pub fn dense_base(&self) -> Option<&CandleNnLinear> {
        match &self.base {
            Base::Dense(base) => Some(base),
            _ => None,
        }
    }

    pub fn replace_dense_base(&mut self, base: CandleNnLinear) {
        self.base = Base::Dense(base);
    }

    /// Add this layer's existing residual stack to an already computed base output. Providers
    /// with an explicit base precision policy (FLUX3's autocast/opmath boundary) use the same
    /// adapter math without recomputing or changing that base. No adapters returns `y` untouched.
    pub fn apply_residuals(&self, x: &Tensor, y: Tensor) -> candle_core::Result<Tensor> {
        self.apply_residuals_with_scales(x, y, None)
    }

    /// [`Self::apply_residuals`] with optional request-scoped [`ScaleContext`] overrides (`PIPE-14`).
    pub fn apply_residuals_with_scales(
        &self,
        x: &Tensor,
        mut y: Tensor,
        scales: Option<&ScaleContext<'_>>,
    ) -> candle_core::Result<Tensor> {
        for (idx, ad) in self.adapters.iter().enumerate() {
            let eff_scale = scales.map_or_else(
                || ad.default_scale(),
                |ctx| ctx.scale_for_f64(idx, ad.default_scale()),
            );
            if eff_scale == 0.0 {
                continue;
            }
            let residual = ad.residual_with_scale(x, eff_scale)?.to_dtype(y.dtype())?;
            y = (y + residual)?;
        }
        Ok(y)
    }

    /// [`Self::forward`] for a **storage-dtype ≠ compute-dtype** site (sc-12828): the dense base weight
    /// is upcast to `x`'s dtype per call, so a bf16-resident projection runs against f32 activations
    /// without materializing the whole weight at f32 (only the one weight in flight is transient). The
    /// additive residuals retain their `x`-dtype arithmetic (`Adapter::residual`) and narrow only to
    /// the resulting host output dtype at the add. Inert (byte-identical to [`Self::forward`], an
    /// `Arc` clone with no copy) when `x` already matches the base dtype — the f32-store
    /// training/control paths. Used by the Qwen3-VL text encoders (krea/boogu), whose bf16-stored
    /// projections run against an f32 hidden state.
    pub fn forward_upcast(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_upcast_with_scales(x, None)
    }

    /// [`Self::forward_upcast`] with optional request-scoped [`ScaleContext`] overrides (`PIPE-14`).
    pub fn forward_upcast_with_scales(
        &self,
        x: &Tensor,
        scales: Option<&ScaleContext<'_>>,
    ) -> candle_core::Result<Tensor> {
        let y = self.base.forward_upcast(x)?;
        self.apply_residuals_with_scales(x, y, scales)
    }

    /// Whether the base loaded from an MLX-packed tier (its codes are quantized) — used to gate the
    /// residual-vs-fold route and asserted by the tests (packed survives load, no dense weight
    /// materialized).
    pub fn is_packed(&self) -> bool {
        self.base.is_packed()
    }

    /// Whether the base is an **NVFP4** projection (sc-21483). Distinct from [`Self::is_packed`],
    /// which stays the MLX-packed (`Q4_1`/`Q8_0`) predicate its existing callers key off — an NVFP4
    /// base holds no GGUF block weight and has no [`MatmulStrategy`].
    pub fn is_nvfp4(&self) -> bool {
        self.base.is_nvfp4()
    }

    /// Whether the base weight is held in *any* quantized regime — MLX-packed, NVFP4, FP8, INT8, or External.
    pub fn is_quantized(&self) -> bool {
        self.is_packed()
            || self.is_nvfp4()
            || self.is_fp8()
            || self.is_int8()
            || matches!(self.base, Base::External { .. })
    }

    pub fn matmul_strategy(&self) -> Option<MatmulStrategy> {
        match &self.base {
            Base::Packed(linear) => linear.matmul_strategy(),
            _ => None,
        }
    }

    pub fn quant_dtype(&self) -> Option<candle_core::quantized::GgmlDType> {
        match &self.base {
            Base::Packed(linear) => linear.quant_dtype(),
            _ => None,
        }
    }

    /// The base projection's `(out_features, in_features)` — the shape a resolved LoRA factor / LoKr
    /// delta is checked against, recoverable even from a packed base.
    pub fn base_shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// The packed base's inner shared [`super::QLinear`] (for a consumer's test to inspect the GGUF
    /// block dtype / device of the folded leaf), or `None` on a dense base. sc-11105.
    pub fn base_qlinear(&self) -> Option<&QLinear> {
        match &self.base {
            Base::Packed(q) => Some(q),
            _ => None,
        }
    }

    /// The NVFP4 base projection (for a consumer's footprint/regime accounting), or `None` on a
    /// dense / MLX-packed base. sc-21483.
    pub fn base_nvfp4(&self) -> Option<&Nvfp4Linear> {
        match &self.base {
            Base::Nvfp4(q) => Some(q),
            _ => None,
        }
    }

    /// The native W4A8 base, including its packed residency and transient weight-scratch bound.
    pub fn base_w4a8(&self) -> Option<&w4a8_int8::W4A8Linear> {
        match &self.base {
            Base::W4A8(q) => Some(q),
            _ => None,
        }
    }

    /// The projection's contraction (`in_features`) — the last-dim an `[in, rank]` LoRA `a` factor
    /// contracts against.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The projection's output width (`out_features`).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Fold an `[out, in]` delta into a **dense** base weight (`W ← W + δ`), for an adapter type that
    /// has no deferred additive form (LyCORIS **LoHa** — a Hadamard product of two low-rank pairs).
    ///
    /// The sum is computed in f32 on the base weight's device and cast back to the base's storage
    /// dtype; a fresh weight tensor replaces the old one, so a tensor shared with another holder
    /// (an mmap'd snapshot, a cloned trunk) is never mutated in place. Attached residuals are kept.
    ///
    /// A **packed** (MLX q4/q8) or **NVFP4** base has no dense weight to fold into and is refused
    /// with a typed `Unsupported` rather than silently dequantized; a delta whose shape is not the
    /// base's `[out, in]` is refused too.
    pub fn fold_dense_delta(&mut self, delta: &Tensor) -> Result<()> {
        let (out_f, in_f) = self.base_shape();
        if delta.dims() != [out_f, in_f] {
            return Err(CandleError::Msg(format!(
                "dense fold: delta {:?} does not match the base [out={out_f}, in={in_f}]",
                delta.dims()
            )));
        }
        match &mut self.base {
            Base::Dense(l) => {
                let w = l.weight();
                let merged = (w.to_dtype(DType::F32)?
                    + delta.to_device(w.device())?.to_dtype(DType::F32)?)?
                .to_dtype(w.dtype())?;
                let bias = l.bias().cloned();
                *l = CandleNnLinear::new(merged, bias);
                Ok(())
            }
            Base::External { .. } => Err(CandleError::Unsupported("dense fold: an external packed base has no owned dense weight; install an additive residual instead".into())),
            Base::Packed(_) => Err(CandleError::Unsupported(
                "dense fold: the base is MLX-packed (q4/q8); a delta cannot be folded into \
                 quantized codes"
                    .into(),
            )),
            Base::Nvfp4(_) => Err(CandleError::Unsupported(
                "dense fold: the base is NVFP4; a delta cannot be folded into packed E2M1 codes"
                    .into(),
            )),
            Base::ConvRot(_) => Err(CandleError::Unsupported("ConvRot INT8 base is immutable; install an additive residual instead".into())),
            Base::W4A8(_) => Err(CandleError::Unsupported(
                "W4A8 base is immutable; install an additive residual instead".into(),
            )),
            #[cfg(feature = "cuda")]
            Base::Fp8(_) => Err(CandleError::Unsupported(
                "FP8 base is immutable; install an additive residual instead".into(),
            )),
            #[cfg(feature = "cuda")]
            Base::Int8(_) => Err(CandleError::Unsupported(
                "INT8 base is immutable; install an additive residual instead".into(),
            )),
        }
    }

    /// Whether any additive residual is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
    }

    /// Device bytes the **frozen** forward-time adapters on this projection hold right now: each
    /// LoRA's factors and each structured LoKr's factors, plus the compute-dtype copies their
    /// forwards have cached so far (a copy at the factors' own dtype shares their storage, so it adds
    /// nothing). Trainable adapters are not counted; the optimizer owns their leaves. A narrowed
    /// factor counts its own elements, not the storage it views. This is what an overlay pricing such
    /// as `gen_core::adapter_stack_upcast_resident_bytes` must cover (sc-24163).
    pub fn frozen_adapter_bytes(&self) -> usize {
        self.adapters
            .iter()
            .map(|adapter| match adapter {
                Adapter::Lora { a, b, prepared, .. } => {
                    tensor_bytes(a) + tensor_bytes(b) + prepared.owned_bytes(a.dtype())
                }
                Adapter::LokrStructured { factors } => factors.device_bytes(),
                Adapter::MatrixDelta(delta) => delta.nbytes(),
                Adapter::TrainableLora { .. } | Adapter::TrainableLokr { .. } => 0,
            })
            .sum()
    }

    /// Attach an already reconstructed matrix. The adapter owns its scale-adjusted coefficients.
    pub fn push_matrix_delta(&mut self, weight: Tensor) -> Result<()> {
        if weight.dims2()? != self.base_shape() {
            return Err(CandleError::Msg(format!(
                "adapter delta {:?} does not match projection {:?}",
                weight.dims(),
                self.base_shape()
            )));
        }
        self.check_factor_admissible("matrix delta", &weight)?;
        self.adapters
            .push(Adapter::MatrixDelta(MatrixDelta { weight }));
        Ok(())
    }

    /// Attach a forward-time **LoRA** residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b`
    /// `[rank, out]` (= `upᵀ` with `alpha/rank` folded in), `scale` the caller's per-adapter strength.
    /// Multiple pushes stack. Valid on **any** base — the base weight is untouched, so a packed q4/q8
    /// tier keeps its footprint.
    ///
    /// **On an NVFP4 base this delegates to [`Self::push_lora_checked`]** (sc-11045 fix round,
    /// MAJOR 7): an NVFP4 host has no fallback — it can neither fold a mismatched delta nor
    /// dequantize to make one fit — so the unchecked bypass is structurally closed and a
    /// mis-shaped factor is a typed refusal at admission, never a first-forward shape panic.
    /// Dense/MLX-packed bases keep the historical unchecked behaviour (their installers carry
    /// their own shape screening) and always return `Ok`.
    pub fn push_lora(&mut self, a: Tensor, b: Tensor, scale: f64) -> Result<()> {
        if self.is_fp8()
            || self.is_convrot_int8()
            || self.is_w4a8()
            || matches!(self.base, Base::Nvfp4(_) | Base::External { .. })
        {
            return self.push_lora_checked(a, b, scale);
        }
        self.adapters.push(Adapter::Lora {
            a,
            b,
            scale,
            prepared: PreparedLora::default(),
        });
        Ok(())
    }

    /// [`Self::push_lora`] with **admission validation** (sc-21483): the factor pair must match the
    /// base projection's contraction and output width, carry a float dtype, and already live on the
    /// base's device. A mismatch is a typed error *here* — at load/admission, before the first
    /// sampler step — rather than a shape panic inside the first forward, and the base is never
    /// re-quantized or dequantized to make a mismatched factor fit.
    ///
    /// This is the entry point a host with **no fallback** must use. A dense base could in principle
    /// fold a delta and a packed base could be dequantized, but an NVFP4 base can do neither: its
    /// only correct answer to a factor it cannot host is to refuse.
    pub fn push_lora_checked(&mut self, a: Tensor, b: Tensor, scale: f64) -> Result<()> {
        let (out_f, in_f) = self.base_shape();
        let (a_dims, b_dims) = (a.dims().to_vec(), b.dims().to_vec());
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(CandleError::Msg(format!(
                "{}: LoRA factors must be rank-2, got a{a_dims:?} b{b_dims:?}",
                self.admission_subject()
            )));
        }
        if a_dims[0] != in_f || b_dims[1] != out_f || a_dims[1] != b_dims[0] {
            return Err(CandleError::Msg(format!(
                "{}: LoRA factor shapes a{a_dims:?}·b{b_dims:?} do not compose against the base \
                 [out={out_f}, in={in_f}]",
                self.admission_subject()
            )));
        }
        self.check_factor_admissible("LoRA `a`", &a)?;
        self.check_factor_admissible("LoRA `b`", &b)?;
        self.adapters.push(Adapter::Lora {
            a,
            b,
            scale,
            prepared: PreparedLora::default(),
        });
        Ok(())
    }

    /// [`Self::push_lokr_structured`] with the same admission validation as
    /// [`Self::push_lora_checked`] (sc-21483). The Kronecker factors are checked against the base
    /// shape they were built for, plus dtype/device, before anything is attached.
    pub fn push_lokr_structured_checked(&mut self, factors: LokrFactors) -> Result<()> {
        let (out_f, in_f) = self.base_shape();
        // A fused source projection routed onto one split host narrows its residual to `len` output
        // rows, so the admitted width is the slice's when one is present.
        let residual_out = match factors.output_slice {
            Some((_, len)) => len,
            None => factors.a * factors.b,
        };
        let residual_in = factors.c * factors.d;
        if residual_out != out_f || residual_in != in_f {
            return Err(CandleError::Msg(format!(
                "{}: LoKr factors reconstruct [out={residual_out}, in={residual_in}], not the base \
                 [out={out_f}, in={in_f}]",
                self.admission_subject(),
            )));
        }
        self.check_factor_admissible("LoKr `w1`", &factors.w1)?;
        self.check_factor_admissible("LoKr `w2`", &factors.w2)?;
        self.adapters.push(Adapter::LokrStructured { factors });
        Ok(())
    }

    /// The base regime named in an admission error, so the message says *which* numeric contract
    /// refused the factor.
    fn admission_subject(&self) -> &'static str {
        match &self.base {
            Base::Dense(_) => "dense base",
            Base::Packed(q) if q.native_device().is_some() => "FP8 base",
            Base::Packed(_) => "MLX-packed base",
            Base::Nvfp4(_) => "NVFP4 base",
            Base::ConvRot(_) => "ConvRot INT8 base",
            Base::W4A8(_) => "W4A8 base",
            #[cfg(feature = "cuda")]
            Base::Fp8(_) => "FP8 base",
            #[cfg(feature = "cuda")]
            Base::Int8(_) => "INT8 base",
            Base::External { .. } => "external packed base",
        }
    }

    /// Reject a residual factor whose dtype is not a float, or which lives on a different device
    /// than the frozen base. Both would otherwise surface as a mid-render failure (or, worse, a
    /// silent host-side copy) on the first denoise step.
    fn check_factor_admissible(&self, what: &str, factor: &Tensor) -> Result<()> {
        match factor.dtype() {
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {}
            other => {
                return Err(CandleError::Msg(format!(
                    "{}: {what} has non-float dtype {other:?}; an additive residual is cast to the \
                     activation dtype per forward and cannot be quantized storage",
                    self.admission_subject()
                )))
            }
        }
        if let Some(base_device) = self.base.device() {
            if !factor.device().same_device(&base_device) {
                return Err(CandleError::Msg(format!(
                    "{}: {what} lives on {:?} but the base weight is on {:?}; move the factor onto \
                     the base device before installing",
                    self.admission_subject(),
                    factor.device().location(),
                    base_device.location(),
                )));
            }
        }
        Ok(())
    }

    /// Number of frozen LoRA/LoKr factor preparations retained by this projection. There is at most
    /// one preparation per installed frozen adapter and compute dtype; trainable adapters contribute
    /// zero because their factors must remain live in every eager loss graph.
    pub fn frozen_adapter_preparation_count(&self) -> usize {
        self.adapters
            .iter()
            .map(|adapter| match adapter {
                Adapter::Lora { prepared, .. } => prepared.count(),
                Adapter::LokrStructured { factors } => factors.preparation_count(),
                Adapter::MatrixDelta(_) => 0,
                Adapter::TrainableLora { .. } | Adapter::TrainableLokr { .. } => 0,
            })
            .sum()
    }

    /// Attach a trainable LoRA residual whose canonical factors are `Var`-backed. Unlike
    /// [`Self::push_lora`], this deliberately performs the factor transposes during each forward so
    /// every loss graph terminates at the live factor leaves and observes optimizer updates.
    pub fn push_trainable_lora(&mut self, down: Tensor, up: Tensor, scale: f64) {
        self.adapters.push(Adapter::TrainableLora {
            down,
            up,
            scale,
            frozen: false,
        });
    }

    /// Attach live full-factor LoKr leaves for training. The base shape is captured here so the
    /// bounded Kronecker residual can be rebuilt inside every eager loss graph.
    pub fn push_trainable_lokr(&mut self, w1: Tensor, w2: Tensor, scale: f64) {
        self.adapters.push(Adapter::TrainableLokr {
            w1,
            w2: TrainableLokrW2::Full(w2),
            base_shape: self.base_shape(),
            scale,
            frozen: false,
        });
    }

    /// [`Self::push_trainable_lokr`] with PEFT's **low-rank** second factor (`use_w2` off): `w2` is
    /// `w2_a [out_b, rank] · w2_b [rank, in_b]`, rebuilt from the live leaves inside every forward
    /// (never cached at install, so no optimizer update is hidden behind a stale product).
    pub fn push_trainable_lokr_low_rank(
        &mut self,
        w1: Tensor,
        w2_a: Tensor,
        w2_b: Tensor,
        scale: f64,
    ) {
        self.adapters.push(Adapter::TrainableLokr {
            w1,
            w2: TrainableLokrW2::LowRank { a: w2_a, b: w2_b },
            base_shape: self.base_shape(),
            scale,
            frozen: false,
        });
    }

    /// Freeze (`true`) or thaw (`false`) every **trainable** residual on this projection. A frozen
    /// trainable residual reads detached, storage-sharing views of its `Var` leaves, so a forward run
    /// while frozen (a preview render between optimizer steps) builds no autograd graph yet still sees
    /// the current factor values; thawing restores the live leaves for the next loss. Frozen
    /// (inference) residuals are unaffected. The [`AdaptLinear`] twin of
    /// `train::lora::LoraLinear::freeze_adapter`/`thaw_adapter`.
    pub fn set_trainable_frozen(&mut self, freeze: bool) {
        for adapter in &mut self.adapters {
            match adapter {
                Adapter::TrainableLora { frozen, .. } | Adapter::TrainableLokr { frozen, .. } => {
                    *frozen = freeze
                }
                Adapter::MatrixDelta(_) | Adapter::Lora { .. } | Adapter::LokrStructured { .. } => {
                }
            }
        }
    }

    /// Whether any trainable LoRA/LoKr adapter is currently attached.
    pub fn has_trainable_adapter(&self) -> bool {
        self.adapters.iter().any(|a| {
            matches!(
                a,
                Adapter::TrainableLora { .. } | Adapter::TrainableLokr { .. }
            )
        })
    }

    /// Whether all attached trainable adapters are currently frozen (detached for graph-free preview).
    pub fn is_trainable_frozen(&self) -> bool {
        let mut found = false;
        for adapter in &self.adapters {
            match adapter {
                Adapter::TrainableLora { frozen, .. } | Adapter::TrainableLokr { frozen, .. } => {
                    if !*frozen {
                        return false;
                    }
                    found = true;
                }
                Adapter::MatrixDelta(_) | Adapter::Lora { .. } | Adapter::LokrStructured { .. } => {
                }
            }
        }
        found
    }

    /// Attach a forward-time **structured LoKr** residual via the Kronecker vec-trick: the full
    /// `(alpha/rank)·strength` scale is already baked into `factors.w2`, so `[out,in]` is never
    /// materialized. Valid on **any** base — the base weight is untouched, so a packed q4/q8 tier keeps
    /// its footprint. Multiple pushes stack, and it mixes freely with LoRA residuals (push order).
    ///
    /// **On an NVFP4 base this delegates to [`Self::push_lokr_structured_checked`]** — same
    /// structural closure as [`Self::push_lora`] (sc-11045 fix round, MAJOR 7).
    pub fn push_lokr_structured(&mut self, factors: LokrFactors) -> Result<()> {
        if self.is_fp8()
            || self.is_convrot_int8()
            || self.is_w4a8()
            || matches!(self.base, Base::Nvfp4(_) | Base::External { .. })
        {
            return self.push_lokr_structured_checked(factors);
        }
        self.adapters.push(Adapter::LokrStructured { factors });
        Ok(())
    }

    /// Drop **every** attached forward-time residual, reverting the projection to its bare base — so the
    /// next [`Self::forward`] is byte-identical to the un-adapted base (`is_adapted()` returns `false`).
    /// The base weight is never touched (it was never mutated by a residual), so clearing is a cheap,
    /// exact toggle-off. The candle twin of mlx-gen's `AdaptableLinear::clear_adapters`: it is the
    /// per-phase adapter *toggle* a multi-phase render runs on its **job-local** DiT between phases —
    /// clear the prior phase's residuals, then push the next phase's subset — without ever mutating the
    /// shared resident base tensors (which stay Arc-shared and read-only across concurrent generates).
    pub fn clear_adapters(&mut self) {
        self.adapters.clear();
    }

    /// Fold a **dense** base to an MLX-packed base in place (Q4/Q8), preserving any attached residuals —
    /// or an **idempotent no-op** on an already-packed base. The shared-core twin of
    /// [`super::QLinear::quantize`] for a consumer whose DiT quantizes AFTER any dense adapter fold
    /// (lens / sd3, sc-11105): on a **dense** tier the projection folds dense→Q4/Q8 here; on a **packed**
    /// tier it is the no-op the additive install relies on, so the forward-time residuals survive. Uses
    /// the sc-7702-safe [`MatmulStrategy::DequantDense`] forward (via `QLinear::quantize`); the
    /// residual stack is untouched — the deltas ride on top of the now-packed base. Only the **base**
    /// weight is quantized (never a residual factor), so a dense base carrying residuals stays correct.
    ///
    /// Native checkpoint operands preserve the source precision and refuse a fold.
    /// An **NVFP4** base is a hard error, never a no-op and never a re-fold (sc-21483): re-quantizing
    /// it to `Q4`/`Q8` — or dequantizing it to BF16 to make some other path fit — would swap the
    /// projection's numeric regime out from under a caller that asked for NVFP4. `Quant::Nvfp4` is
    /// refused too: NVFP4 is packed from the bf16 master, so "re-NVFP4ing" an NVFP4 base could only
    /// mean a double quantization.
    pub fn quantize(&mut self, quant: Quant) -> candle_core::Result<()> {
        self.base.quantize(quant)
    }

    /// As [`Self::quantize`] but folds a **dense** base to a packed base landing on an explicit `device`
    /// (the CPU-stage → quantize-onto-GPU path, sc-8504 / sd3) — the base is quantized on its current
    /// device and placed on `device` via [`super::QLinear::quantize_dequant_onto`]. An already-packed
    /// base is an **idempotent no-op**. Only the base is folded; this is only used on the dense-fold
    /// route (which carries no forward-time residuals), so the residual stack — empty there — is untouched.
    pub fn quantize_dequant_onto(
        &mut self,
        quant: Quant,
        device: &Device,
    ) -> candle_core::Result<()> {
        self.base.quantize_dequant_onto(quant, device)
    }

    /// Quantize the base projection with [`QLinear::quantize_onto`] while preserving all attached
    /// forward-time residuals.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> candle_core::Result<()> {
        self.base.quantize_onto(quant, device)
    }

    /// Apply a component tier, retaining the source-native precision and migrating attached factors.
    pub fn quantize_component(
        &mut self,
        quant: Quant,
        fold: ComponentFold<'_>,
    ) -> candle_core::Result<()> {
        self.base.quantize_component(quant, fold)?;
        if let Some(device) = fold.target() {
            self.to_device(device)?;
        }
        Ok(())
    }

    /// Move the projection — base + every attached residual factor — onto `device`, in place. The seam
    /// a consumer's **dense-kept** leaf migrates through on the CPU-stage quant path (sd3's AdaLN /
    /// embedder linears, sc-11105): they build on the CPU with the rest of the DiT, then migrate to the
    /// GPU alongside the quantized projections. Mirrors [`super::QLinear::to_device`] (a dense base
    /// moves; a packed base is a no-op — its quantized weight already lives on its device), and also
    /// moves any forward-time residual so a migrated adapted leaf stays device-consistent.
    pub fn to_device(&mut self, device: &Device) -> candle_core::Result<()> {
        match &mut self.base {
            Base::Dense(l) => {
                let w = l.weight().to_device(device)?;
                let b = match l.bias() {
                    Some(b) => Some(b.to_device(device)?),
                    None => None,
                };
                *l = CandleNnLinear::new(w, b);
            }
            Base::External { device: owner, .. } => {
                if !owner.same_device(device) {
                    return Err(external_base_refusal(
                        "migrate the caller-owned weight to another device",
                    ));
                }
            }
            Base::Packed(q) => q.to_device(device)?,
            // The NVFP4 weight is staged on its device at construction and cannot be re-staged
            // without re-packing, so a cross-device move is refused rather than silently dropped
            // (which would leave the residuals on one device and the base on another). A move onto
            // the device it already occupies is the ordinary no-op.
            Base::Nvfp4(q) => {
                if !q.device().same_device(device) {
                    return Err(candle_core::Error::Msg(format!(
                        "NVFP4 base: refusing to migrate a staged NVFP4 projection from {:?} to \
                         {:?}; the packed weight would have to be re-staged",
                        q.device().location(),
                        device.location(),
                    )));
                }
            }
            Base::ConvRot(q) => {
                if !q.device().same_device(device) {
                    candle_core::bail!(
                        "ConvRot INT8 refuses cross-device migration of its staged base"
                    );
                }
            }
            Base::W4A8(q) => {
                if !q.device().same_device(device) {
                    candle_core::bail!("W4A8 refuses cross-device migration of its packed base");
                }
            }
            #[cfg(feature = "cuda")]
            Base::Fp8(q) => {
                if !q.device().same_device(device) {
                    candle_core::bail!("FP8 refuses cross-device migration of its staged base");
                }
            }
            #[cfg(feature = "cuda")]
            Base::Int8(q) => {
                if !q.device().same_device(device) {
                    candle_core::bail!("INT8 refuses cross-device migration of its staged base");
                }
            }
        }
        for ad in &mut self.adapters {
            ad.migrate_to(device)?;
        }
        Ok(())
    }
}

impl From<CandleNnLinear> for AdaptLinear {
    fn from(linear: CandleNnLinear) -> Self {
        let dims = linear.weight().dims();
        let (out_dim, in_dim) = (dims[0], dims[1]);
        Self::from_dense(linear, in_dim, out_dim)
    }
}

impl Module for AdaptLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        AdaptLinear::forward(self, x)
    }
}

/// 2-D Kronecker product `kron(a[m,n], b[p,q]) = [m·p, n·q]` via broadcast — differentiable, so grads
/// flow to `a`/`b`. `out[i·p+k, j·q+l] = a[i,j]·b[k,l]`.
pub fn kron2d(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let (m, n) = a.dims2()?;
    let (p, q) = b.dims2()?;
    let a4 = a.reshape((m, 1, n, 1))?;
    let b4 = b.reshape((1, p, 1, q))?;
    a4.broadcast_mul(&b4)?.reshape((m * p, n * q))
}

/// Reconstruct the LoRA weight delta `ΔW = (alpha/rank)·scale·(B·A)` as an `[out, in]` **f32** tensor
/// — the inference-side **merge** counterpart to `LoraLinear`'s training **forward**, which adds the
/// mathematically-identical residual `scale·(x·Aᵀ)·Bᵀ` with the install `scale = alpha/rank`. `down`
/// is `A` `[rank, in]`, `up` is `B` `[out, rank]`; `scale` is the caller's per-adapter strength
/// (`gen_core::AdapterSpec::scale`, `1.0` reconstructs the trained delta verbatim). Computed in f32 so
/// a candle-trained adapter round-trips through inference exactly.
///
/// SDXL merges this into the dense UNet weight (`W += ΔW`) rather than adding it live: the ancestral
/// sampler is chaos-sensitive and the merged forward `(W+ΔW)·x` differs from the residual form
/// `W·x + ΔW·x` by ~1 ULP, which cascades to a visibly different image (see `candle-gen-sdxl`'s
/// adapter merge). Holding both forms to the same f32 reconstruction keeps train and infer in lockstep.
pub fn reconstruct_lora_delta(
    down: &Tensor,
    up: &Tensor,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Tensor> {
    let down = down.to_dtype(DType::F32)?;
    let up = up.to_dtype(DType::F32)?;
    let ba = up.matmul(&down)?; // [out, rank] · [rank, in] → [out, in]
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    Ok((ba * eff)?)
}

/// Parse PEFT LoKr `rank`/`alpha` metadata without collapsing a present malformed value to a default.
/// Missing rank defaults to 1 and missing alpha defaults to rank; present values must match the same
/// finite, positive-rank contract enforced by [`reconstruct_lokr_delta`].
pub fn parse_lokr_metadata(rank: Option<&str>, alpha: Option<&str>) -> Result<(f32, f32)> {
    let parse = |field: &str, raw: &str| {
        raw.trim().parse::<f32>().map_err(|_| {
            CandleError::Msg(format!(
                "LoKr adapter metadata `{field}` = `{raw}` is not a valid number"
            ))
        })
    };
    let rank = match rank {
        Some(raw) => parse("rank", raw)?,
        None => 1.0,
    };
    let alpha = match alpha {
        Some(raw) => parse("alpha", raw)?,
        None => rank,
    };
    if !rank.is_finite() || rank <= 0.0 || !alpha.is_finite() {
        return Err(CandleError::Msg(format!(
            "LoKr: invalid metadata scale (rank = {rank}, alpha = {alpha}); rank must be finite and \
             > 0 and alpha must be finite"
        )));
    }
    Ok((rank, alpha))
}

/// Reconstruct the LoKr weight delta `ΔW = (alpha/rank)·scale·kron(w1, w2)` reshaped to `base_shape`
/// (`[out, in]`), as **f32** — the inference-side merge counterpart to `LoraLinear`'s LoKr forward,
/// using the *same* `kron2d` reconstruction. Each Kronecker leg is either a full factor (`w1`/`w2`)
/// or a low-rank product (`w1_a·w1_b` / `w2_a·w2_b`, e.g. the trainer's zero-init `w2_a`/`w2_b` form).
/// Errors if a leg is missing or if metadata/user scaling is invalid. Linear-only: pass 2-D factors
/// (the SDXL trainer adapts Linears; Tucker / conv reconstruction is not handled here).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    w1: Option<&Tensor>,
    w1_a: Option<&Tensor>,
    w1_b: Option<&Tensor>,
    w2: Option<&Tensor>,
    w2_a: Option<&Tensor>,
    w2_b: Option<&Tensor>,
    alpha: f32,
    rank: f32,
    scale: f32,
    base_shape: impl Into<Option<(usize, usize)>>,
) -> Result<Tensor> {
    if !scale.is_finite() {
        return Err(CandleError::Msg(format!(
            "LoKr: adapter scale must be finite (got {scale})"
        )));
    }
    if !rank.is_finite() || rank <= 0.0 || !alpha.is_finite() {
        return Err(CandleError::Msg(format!(
            "LoKr: invalid metadata scale (rank = {rank}, alpha = {alpha}); rank must be finite and \
             > 0 and alpha must be finite"
        )));
    }
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    if !eff.is_finite() {
        return Err(CandleError::Msg(format!(
            "LoKr: derived (alpha/rank) * adapter scale must be finite (got {eff})"
        )));
    }
    let f32d = |t: &Tensor| t.to_dtype(DType::F32);
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => f32d(w)?,
        (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
        _ => {
            return Err(CandleError::Msg(
                "lokr: w1 missing (need full lokr_w1 or lokr_w1_a·lokr_w1_b)".into(),
            ))
        }
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => f32d(w)?,
        (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
        _ => {
            return Err(CandleError::Msg(
                "lokr: w2 missing (need full lokr_w2 or lokr_w2_a·lokr_w2_b)".into(),
            ))
        }
    };
    let delta = kron2d(&factor1, &factor2)?;
    let delta = match base_shape.into() {
        Some(shape) => delta.reshape(shape)?,
        None => delta,
    };
    Ok((delta * eff)?)
}

/// Fuse a **conv-layer** LoRA pair into a single conv-weight delta in the trained-file **NCHW**
/// `[out, in, kH, kW]` layout (sc-5225) — the conv analog of [`reconstruct_lora_delta`]. Community SDXL
/// LoRAs adapt convs (resnet `conv1`/`conv2`/`conv_shortcut`, the down/up-samplers, `conv_in`/`conv_out`)
/// by decomposing a conv into a spatial `down` (`lora_down`, `[rank, in, kH, kW]`) followed by a 1×1
/// `up` (`lora_up`, `[out, rank, 1, 1]`); the fused weight is the composition of those two convs:
///   `δ[o, i, y, x] = Σ_r up[o, r] · down[r, i, y, x]`,
/// which is exactly `up[out, rank] · down[rank, in·kH·kW]` reshaped back to `[out, in, kH, kW]` —
/// bit-identical to PEFT/diffusers' `Conv2d` LoRA fusion, uniform across 1×1 and k×k kernels — then
/// scaled by `(alpha/rank)·scale`. Computed in **f32** (the candle SDXL merge path is f32-everywhere,
/// matching [`reconstruct_lora_delta`]); the caller folds it into the conv weight (`W += δ`).
///
/// Unlike mlx-gen's `conv_lora_delta` (which returns NCHW and transposes to NHWC at the merge site,
/// because mlx stores conv weights NHWC), candle convs are already NCHW (`candle_nn::Conv2d`), so the
/// returned delta merges into the diffusers `{path}.weight` tensor with no transpose. A non-4-D factor
/// (a malformed conv LoRA) is a typed error rather than a panic on the kernel-dim reshape.
pub fn conv_lora_delta(
    down: &Tensor,
    up: &Tensor,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Tensor> {
    let (ds, us) = (down.dims(), up.dims());
    if ds.len() != 4 || us.len() != 4 {
        return Err(CandleError::Msg(format!(
            "conv LoRA: expected 4-D factors (down [rank,in,kH,kW], up [out,rank,1,1]), got down \
             {ds:?} up {us:?}"
        )));
    }
    if us[1] != ds[0] {
        return Err(CandleError::Msg(format!(
            "conv LoRA: rank mismatch between factors — down[0]={} but up[1]={} (down {ds:?} up {us:?})",
            ds[0], us[1]
        )));
    }
    let (r, cin, kh, kw) = (ds[0], ds[1], ds[2], ds[3]);
    let out = us[0];
    let down2 = down.to_dtype(DType::F32)?.reshape((r, cin * kh * kw))?; // [rank, in·kH·kW]
    let up2 = up.to_dtype(DType::F32)?.reshape((out, r))?; // [out, rank]
    let ba = up2.matmul(&down2)?; // [out, in·kH·kW]
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    Ok((ba * eff)?.reshape((out, cin, kh, kw))?)
}

/// Reconstruct a **LoHa** (LyCORIS Hadamard-product) weight delta `ΔW = scale · ((w1_a·w1_b) ⊙
/// (w2_a·w2_b))` as an `[out, in]` **f32** tensor (sc-5225). Third-party LoHa decomposes a delta as the
/// elementwise product of TWO low-rank products (vs LoKr's Kronecker). `w1_a`/`w2_a` are `[out, rank]`,
/// `w1_b`/`w2_b` are `[rank, in]`; `scale` is the fully-effective multiplier (the lycoris `alpha/rank`
/// times the caller's per-adapter strength). Mirrors LyCORIS `LohaModule.get_weight` / `HadaWeight`
/// (`w1d=hada_w1_b, w1u=hada_w1_a`, so the product is `(hada_w1_a·hada_w1_b) ⊙ (hada_w2_a·hada_w2_b)`).
///
/// **Linear-only**: pass 2-D factors. The conv/tucker LoHa form (lycoris `use_cp`, `hada_t1`/`hada_t2`)
/// is out of the candle SDXL adapter surface — like third-party LoKr, LoHa merges only into the
/// attention/proj Linears (the conv surface is LoRA-only), so a conv-shaped LoHa is surfaced as skipped.
///
/// **Orientation is checked, not just element count** (sc-24157): every factor must be 2-D with
/// `w*_a` `[out, r]` and `w*_b` `[r, in]` for `base_shape = (out, in)`. A transposed module
/// (`[in, r]`·`[r, out]`) has the same `out·in` elements and would otherwise reshape into a
/// scrambled `[out, in]` delta and merge silently.
pub fn reconstruct_loha_delta(
    w1_a: &Tensor,
    w1_b: &Tensor,
    w2_a: &Tensor,
    w2_b: &Tensor,
    scale: f32,
    base_shape: impl Into<Option<(usize, usize)>>,
) -> Result<Tensor> {
    let (out_f, in_f) = match base_shape.into() {
        Some(shape) => shape,
        None => (w1_a.dims2()?.0, w1_b.dims2()?.1),
    };
    let oriented = |a: &Tensor, b: &Tensor| {
        let (a, b) = (a.dims(), b.dims());
        a.len() == 2 && b.len() == 2 && a[0] == out_f && b[1] == in_f && a[1] == b[0]
    };
    if !(oriented(w1_a, w1_b) && oriented(w2_a, w2_b)) {
        return Err(CandleError::Msg(format!(
            "loha: factors w1_a {:?} / w1_b {:?} / w2_a {:?} / w2_b {:?} do not reconstruct the \
             base [out={out_f}, in={in_f}] (expected w*_a [out, r] and w*_b [r, in])",
            w1_a.dims(),
            w1_b.dims(),
            w2_a.dims(),
            w2_b.dims()
        )));
    }
    let f32d = |t: &Tensor| t.to_dtype(DType::F32);
    let m1 = f32d(w1_a)?.matmul(&f32d(w1_b)?)?; // [out, rank]·[rank, in] → [out, in]
    let m2 = f32d(w2_a)?.matmul(&f32d(w2_b)?)?;
    let delta = (m1 * m2)?.reshape((out_f, in_f))?;
    Ok((delta * scale as f64)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_context_resolves_per_adapter_and_uniform_overrides() {
        let ctx = ScaleContext::new();
        assert_eq!(ctx.pass, 0);
        assert_eq!(ctx.scale_for(0, 0.75), 0.75);
        assert_eq!(ctx.scale_for_f64(0, 0.75), 0.75);

        let overrides = [0.25f32, 0.0, 1.5];
        let per_adapter = ScaleContext::with_adapter_scales(&overrides).with_pass(3);
        assert_eq!(per_adapter.pass, 3);
        assert_eq!(per_adapter.scale_for(0, 1.0), 0.25);
        assert_eq!(per_adapter.scale_for_f64(1, 1.0), 0.0);
        assert_eq!(per_adapter.scale_for_f64(2, 1.0), 1.5);
        assert_eq!(per_adapter.scale_for(3, 0.5), 0.5);

        let uniform = ScaleContext::with_uniform_scale(0.5);
        assert_eq!(uniform.scale_for_f64(0, 0.8), 0.4);
    }

    #[test]
    fn parse_lokr_metadata_validates_rank_and_alpha() {
        let (rank, alpha) = parse_lokr_metadata(Some("8"), Some("4.0")).unwrap();
        assert_eq!((rank, alpha), (8.0, 4.0));
        let (default_rank, default_alpha) = parse_lokr_metadata(None, None).unwrap();
        assert_eq!((default_rank, default_alpha), (1.0, 1.0));
        assert!(parse_lokr_metadata(Some("0"), Some("4.0")).is_err());
        assert!(parse_lokr_metadata(Some("-2"), None).is_err());
        assert!(parse_lokr_metadata(Some("NaN"), None).is_err());
    }

    #[test]
    fn adapt_linear_spine_introspection_and_trainable_freeze_transitions() {
        let dev = Device::Cpu;
        let ext = AdaptLinear::for_external_base(8, 4, DType::BF16, &dev).unwrap();
        assert_eq!(ext.base_shape(), (4, 8));
        assert_eq!(ext.in_features(), 8);
        assert_eq!(ext.out_features(), 4);
        assert!(ext.is_quantized());
        assert!(!ext.is_packed());
        assert!(!ext.is_nvfp4());
        assert!(!ext.is_fp8());
        assert!(!ext.is_int8());
        assert_eq!(ext.external_base_compute_dtype(), Some(DType::BF16));

        let w = Tensor::zeros((4, 8), DType::F32, &dev).unwrap();
        let qlin = QLinear::from_dense(DenseLinear::Linear(CandleNnLinear::new(w, None)));
        let mut lin = AdaptLinear::from_qlinear(qlin, 8, 4);
        assert!(!lin.is_packed());
        assert!(lin.dense_base().is_some());
        assert!(!lin.has_trainable_adapter());
        assert!(!lin.is_trainable_frozen());

        let down = Tensor::zeros((2, 8), DType::F32, &dev).unwrap();
        let up = Tensor::zeros((4, 2), DType::F32, &dev).unwrap();
        lin.push_trainable_lora(down, up, 1.0);
        assert!(lin.is_adapted());
        assert!(lin.has_trainable_adapter());
        assert!(!lin.is_trainable_frozen());

        lin.set_trainable_frozen(true);
        assert!(lin.is_trainable_frozen());
        lin.set_trainable_frozen(false);
        assert!(!lin.is_trainable_frozen());

        lin.clear_adapters();
        assert!(!lin.is_adapted());
        assert!(!lin.has_trainable_adapter());
        assert!(!lin.is_trainable_frozen());
    }

    #[test]
    fn lokr_structured_zero_baked_scale_rejects_nonzero_override() {
        let dev = Device::Cpu;
        let w1 = Tensor::ones((2, 2), DType::F32, &dev).unwrap();
        let w2 = Tensor::ones((2, 2), DType::F32, &dev).unwrap();
        let adapter = Adapter::LokrStructured {
            factors: LokrFactors::build(
                0.0,
                (4, 4),
                Some(&w1),
                None,
                None,
                Some(&w2),
                None,
                None,
                None,
            )
            .unwrap()
            .unwrap(),
        };
        let x = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let err = adapter
            .residual_with_scale(&x, 0.5)
            .expect_err("zero-baked LokrStructured must reject nonzero override");
        assert!(err.to_string().contains("baked factors.scale is 0.0"));

        let small = Adapter::LokrStructured {
            factors: LokrFactors::build(
                5e-17,
                (4, 4),
                Some(&w1),
                None,
                None,
                Some(&w2),
                None,
                None,
                None,
            )
            .unwrap()
            .unwrap(),
        };
        let x_ones = Tensor::ones((1, 4), DType::F32, &dev).unwrap();
        let pos = small
            .residual_with_scale(&x_ones, 5e-17)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let neg = small
            .residual_with_scale(&x_ones, -5e-17)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(pos[0] > 0.0);
        assert_eq!(neg[0], -pos[0]);
    }
}
