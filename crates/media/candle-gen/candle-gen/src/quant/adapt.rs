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
//! **The base weight is NEVER mutated.** On a packed q4/q8 DiT that is the whole point: the packed
//! codes survive load (`is_packed()` stays true, no dense `[out,in]` weight is materialized), and each
//! adapter rides *unmerged* as small matmuls per forward. On a **dense** base the fold path
//! ([`crate::train::merge`]) is still preferred for real runs, because a merge into a real `.weight`
//! is byte-for-byte what the chaos-sensitive samplers' goldens expect (`(W+δ)·x ≠ W·x + δ·x` to ~1
//! ULP — see [`crate::train::lora::reconstruct_lora_delta`]); the additive branch equals the fold to
//! f32 tolerance and is the *only* viable path where a merge is impossible (a quant-resident weight).
//! With **no** adapter attached the forward is byte-identical to the bare base, so swapping this in for
//! a projection leaves the plain-model / dense-fold paths unchanged.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{lin_gs, DenseLinear, QLinear, MLX_GROUP_SIZE};
use crate::{CandleError, Result};
use gen_core::Quant;

/// The frozen base weight — **dense** (`candle_nn::Linear`), **MLX-packed** ([`super::QLinear`],
/// dequant-on-forward), or **NVFP4** ([`super::Nvfp4Linear`], E2M1 block-16 + UE4M3 scales served
/// either through the FP4 tensor-core GEMM or its transparent dequant→bf16 fallback — sc-21483).
/// All three compute `x·Wᵀ (+ b)`; none is ever mutated by an adapter.
///
/// The NVFP4 leg is held behind an [`Arc`] because [`super::Nvfp4Linear`] owns a staged device
/// weight (and, on cuda, a cuBLASLt resident) that must not be deep-copied when a job-local DiT
/// clones the resident trunk to install its own residual stack. The base is read-only by contract,
/// so sharing it is exactly right: cloning an [`AdaptLinear`] to adapt it per job costs a refcount,
/// never a re-pack.
#[derive(Clone)]
enum Base {
    Dense(Linear),
    Packed(QLinear),
    Nvfp4(std::sync::Arc<super::Nvfp4Linear>),
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

impl Base {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Base::Dense(l) => l.forward(x),
            Base::Packed(q) => q.forward(x),
            Base::Nvfp4(q) => q.forward(x),
        }
    }

    /// [`Self::forward`] with a **dense** base weight (and bias) upcast to `x`'s dtype per call — the
    /// storage-dtype ≠ compute-dtype regime (bf16 weights, f32 activations, sc-12828). Only the one
    /// weight in flight is transiently materialized at the compute dtype, so the resident footprint
    /// stays at the storage dtype's. Inert when the base already matches `x`'s dtype (`Tensor::to_dtype`
    /// short-circuits to an `Arc` clone), so byte-identical to [`Self::forward`]. The **packed** base
    /// already dequantizes to the activation dtype, so it delegates unchanged.
    fn forward_upcast(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Base::Dense(l) => {
                let dt = x.dtype();
                let w = l.weight().to_dtype(dt)?;
                let b = l.bias().map(|b| b.to_dtype(dt)).transpose()?;
                Linear::new(w, b).forward(x)
            }
            Base::Packed(q) => q.forward(x),
            // NVFP4 already produces its output in the activation dtype (the packed weight is never
            // stored at the activation dtype in the first place), so there is nothing to upcast.
            Base::Nvfp4(q) => q.forward(x),
        }
    }

    fn is_packed(&self) -> bool {
        matches!(self, Base::Packed(_))
    }

    fn is_nvfp4(&self) -> bool {
        matches!(self, Base::Nvfp4(_))
    }

    /// The device the frozen base weight lives on — the device a residual factor must already be on
    /// before it can be admitted. `None` on the MLX-packed arm, whose resident weight is a `QTensor`
    /// behind a [`super::MatmulStrategy`] and exposes no device accessor; an admission check then
    /// verifies shape and dtype only, exactly as it did before this seam existed.
    fn device(&self) -> Option<Device> {
        match self {
            Base::Dense(l) => Some(l.weight().device().clone()),
            Base::Packed(_) => None,
            Base::Nvfp4(q) => Some(q.device().clone()),
        }
    }
}

/// A forward-time additive residual attached to an [`AdaptLinear`] — it never touches the frozen base
/// weight, so it is memory-free on a packed q4/q8 tier. Factors are held **f32** (the merge/train
/// dtype) and cast to the activation dtype per forward (they are tiny, so the cast is cheap).
#[derive(Clone)]
enum Adapter {
    /// LoRA residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b` `[rank, out]` (= `upᵀ` with
    /// the `alpha/rank` ratio folded in at resolution). The **deferred two-small-matmul** form — never
    /// the `[out,in]` product — so it stays memory-free on any quant.
    Lora { a: Tensor, b: Tensor, scale: f64 },
    /// Training LoRA keeps the canonical `down [rank,in]` / `up [out,rank]` variable leaves and
    /// transposes them inside every forward. Holding a precomputed transpose here would retain an
    /// eager graph node from adapter installation rather than make the owning `Var`s leaves of each
    /// new loss graph, so gradients and subsequent `Var::set` updates would be silently lost.
    TrainableLora {
        down: Tensor,
        up: Tensor,
        scale: f64,
    },
    /// Training LoKr retains live `Var` leaves and reconstructs only its bounded structured factors
    /// inside the current forward graph. This is the LoKr twin of `TrainableLora`.
    TrainableLokr {
        w1: Tensor,
        w2: Tensor,
        base_shape: (usize, usize),
        scale: f64,
    },
    /// Structured LoKr residual via the Kronecker vec-trick — the FULL `(alpha/rank)·strength` scale is
    /// baked into [`LokrFactors::w2`], so a LoKr applies WITHOUT ever forming the `[out,in]` delta (the
    /// packed-capable path the whole hoist adds over Wan's old dense-only delta).
    LokrStructured { factors: LokrFactors },
}

/// Apply a **2-D** factor `w` `[in, out]` to an activation `x` whose last dim is `in`, folding every
/// leading dim into the GEMM's `M`.
///
/// **Never `broadcast_matmul` here.** candle materializes a broadcast rhs through `.contiguous()`
/// (`candle-core`'s own `broadcast_matmul` carries the `TODO: Avoid concretising the broadcasted
/// matrixes` note), so a 2-D factor against a `[N, S, in]` activation is physically **copied `N`
/// times** and then multiplied as `N` batched GEMMs with `M = S`. That is free where `N = 1` — every
/// DiT trunk site — and pathological where a site folds a token count into the *batch* dim: the Krea
/// text-fusion `layerwise_blocks` run `[n_tokens, num_layers, d]`, so a 2048² image **edit**
/// (`n_tokens = 4107` vision+text tokens, `d = 2560`, rank 256) turned each rank-256 factor into a
/// 2.7e9-element / **5.4 GB** copy and then a 4107-batch `M = 12` GEMM against it — 32 such legs per
/// denoise step across the two layerwise blocks' eight adapted projections. The first denoise step
/// never finished (hours), while the same render without an adapter took 55 s.
///
/// Flattening is mathematically identical (`matmul` contracts the last dim either way), allocates
/// nothing beyond the result, and issues ONE large-`M` GEMM — the same lesson sc-11785 applied to the
/// LoKr vec-trick's expensive leg, here for the plain-LoRA legs.
fn apply_factor(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    // A non-2-D factor has no flattened form; keep the general (broadcasting) path.
    if w.rank() != 2 || dims.len() <= 2 {
        return x.broadcast_matmul(w);
    }
    let (lead, k) = dims.split_at(dims.len() - 1);
    let m: usize = lead.iter().product();
    let mut out_dims = lead.to_vec();
    out_dims.push(w.dim(1)?);
    x.reshape((m, k[0]))?.matmul(w)?.reshape(out_dims)
}

impl Adapter {
    /// A scalar-zero LoRA contributes exactly nothing. Detect it before reading either factor: this
    /// keeps disabled adapters byte-identical to the bare host even when an unused factor contains a
    /// non-finite value, and avoids two needless matmuls. Structured LoKr retains its pre-bake scale
    /// solely for the same disabled check.
    fn is_zero(&self) -> bool {
        match self {
            Adapter::Lora { scale, .. } => *scale == 0.0,
            Adapter::TrainableLora { scale, .. } => *scale == 0.0,
            Adapter::TrainableLokr { scale, .. } => *scale == 0.0,
            Adapter::LokrStructured { factors } => factors.scale == 0.0,
        }
    }

    /// The residual this adapter contributes, in the activation dtype of `x`.
    fn residual(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Adapter::Lora { a, b, scale } => {
                let xd = x.dtype();
                let r = apply_factor(&apply_factor(x, &a.to_dtype(xd)?)?, &b.to_dtype(xd)?)?;
                r * *scale
            }
            Adapter::TrainableLora { down, up, scale } => {
                let xd = x.dtype();
                let down = down.to_dtype(xd)?;
                let up = up.to_dtype(xd)?;
                let r = apply_factor(&apply_factor(x, &down.t()?)?, &up.t()?)?;
                r * *scale
            }
            Adapter::TrainableLokr {
                w1,
                w2,
                base_shape,
                scale,
            } => LokrFactors::build(
                *scale,
                *base_shape,
                Some(w1),
                None,
                None,
                Some(w2),
                None,
                None,
                None,
            )
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::Msg("trainable LoKr factors lost 2-D shape".into()))?
            .residual(x),
            // The `scale` is already baked into `factors.w2`, so the vec-trick returns directly.
            Adapter::LokrStructured { factors } => factors.residual(x),
        }
    }

    /// Move this residual's factors onto `device`, in place (the dense-leaf migration seam, sc-11105).
    fn migrate_to(&mut self, device: &Device) -> candle_core::Result<()> {
        match self {
            Adapter::Lora { a, b, .. } => {
                *a = a.to_device(device)?;
                *b = b.to_device(device)?;
            }
            Adapter::TrainableLora { down, up, .. } => {
                *down = down.to_device(device)?;
                *up = up.to_device(device)?;
            }
            Adapter::TrainableLokr { w1, w2, .. } => {
                *w1 = w1.to_device(device)?;
                *w2 = w2.to_device(device)?;
            }
            // `LokrFactors` fields are same-module-private; move `w1`/`w2` directly (candle_core::Result).
            Adapter::LokrStructured { factors } => {
                factors.w1 = factors.w1.to_device(device)?;
                factors.w2 = factors.w2.to_device(device)?;
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
/// profile as a plain LoRA. The row-major kron ordering here matches [`crate::train::lora`]'s `kron2d`
/// (`out[i·b+k, j·d+l] = w1[i,j]·w2[k,l]`), so the structured residual equals the folded delta. The
/// full `(alpha/rank)·strength` scale is baked into `w2` at build time.
///
/// `Clone`/`Debug` so a caller that stacks these residuals on its own adaptable seam (the SDXL
/// [`crate::train::lora::LoraLinear`] additive channel, sc-11103) can hold them in a `#[derive(Clone,
/// Debug)]` module without re-implementing the vec-trick — the factors are `Tensor`s (cheap `Arc`
/// clone) plus `usize` shape metadata.
#[derive(Clone, Debug)]
pub struct LokrFactors {
    /// `[a, c]` — the left Kronecker factor (`out = a·b`, `in = c·d`).
    w1: Tensor,
    /// `[b, d]` — the right Kronecker factor, with the full scale baked in.
    w2: Tensor,
    /// `a` — row count of `w1`; the flattened output index is `p·b + q`.
    a: usize,
    /// `b` — row count of `w2`.
    b: usize,
    /// `c` — col count of `w1`; the flattened input index is `r·d + s`.
    c: usize,
    /// `d` — col count of `w2`.
    d: usize,
    /// Optional output-row slice for a fused source projection routed onto one split host projection.
    output_slice: Option<(usize, usize)>,
    /// Pre-bake scale retained solely so a disabled structured LoKr can be skipped before reading
    /// factors. The residual math continues to use the scale already baked into `w2`.
    pub(crate) scale: f64,
}

impl LokrFactors {
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
    /// [`crate::train::lora::reconstruct_lokr_delta`] (which materializes the full `[out,in]` delta).
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
        let w2 = (factor2 * scale)?.contiguous()?;
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
        }))
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
        })
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
        let w1 = self.w1.to_dtype(xd)?;
        let w2t = self.w2.to_dtype(xd)?.t()?.contiguous()?; // [d, b]
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
    base: Base,
    /// The projection's logical `(out_features, in_features)` — captured at construction (recoverable
    /// even from a packed base, where the dense weight is never materialized) so the residual installer
    /// can shape-check a factor without reading the quantized weight back.
    out_features: usize,
    in_features: usize,
    /// Forward-time additive residuals, applied in push order (adapters stack). Empty on the plain /
    /// dense-fold path ⇒ forward is byte-identical to the bare base.
    adapters: Vec<Adapter>,
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
    /// Wrap an already-built dense `candle_nn::Linear` with its logical `[out, in]` dims — the seam a
    /// caller uses when it holds a `Linear` directly (e.g. a folded-delta reference, or a test grid).
    pub fn from_dense(l: Linear, in_dim: usize, out_dim: usize) -> Self {
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
            base: Base::Packed(base),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
    }

    /// Wrap an already-built **NVFP4** [`super::Nvfp4Linear`] as an adapter-capable base (sc-21483,
    /// epic 11037) — the NVFP4 twin of [`Self::from_packed`]. The logical `[out, in]` dims come from
    /// the packed tensor itself, so no dense weight is read back.
    ///
    /// The point of routing NVFP4 through the ONE shared additive wrapper (sc-11091) rather than
    /// giving it a parallel residual stack: a LoRA/LoKr rides as `scale·((x·a)·b)` (or the Kronecker
    /// vec-trick) *alongside* the packed forward, so the E2M1 codes are never dequantized, never
    /// re-packed, and never converted to q4/BF16 to make an adapter fit. Removing the adapter
    /// ([`Self::clear_adapters`]) restores the base output exactly, because the base was never
    /// touched.
    pub fn from_nvfp4(base: super::Nvfp4Linear) -> Self {
        let (out_dim, in_dim) = base.shape();
        Self {
            base: Base::Nvfp4(std::sync::Arc::new(base)),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        }
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
    /// parts on `vb`'s device via the shared [`super::lin_gs`] — **no dense weight is materialized**.
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
            let w = vb.get_unchecked(&format!("{name}.weight"))?;
            let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
            Ok(Self {
                base: Base::Dense(Linear::new(w, None)),
                out_features,
                in_features,
                adapters: Vec::new(),
            })
        }
    }

    /// Bias-less **dense** `[out, in]` projection from `{name}.weight` (a component that never packs,
    /// e.g. Anima's conditioner q/k/v/o — dense bf16 in every tier).
    pub fn dense(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?;
        let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
        Ok(Self {
            base: Base::Dense(Linear::new(w, None)),
            out_features,
            in_features,
            adapters: Vec::new(),
        })
    }

    /// **Dense** `[out, in]` projection with bias from `{name}.weight` + `{name}.bias`.
    pub fn dense_bias(vb: &VarBuilder, name: &str) -> candle_core::Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?;
        let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
        let b = vb.get_unchecked(&format!("{name}.bias"))?;
        Ok(Self {
            base: Base::Dense(Linear::new(w, Some(b))),
            out_features,
            in_features,
            adapters: Vec::new(),
        })
    }

    /// `x·Wᵀ (+ b)` plus every attached additive residual, in push order. A disabled residual is
    /// skipped entirely; each live residual keeps `Adapter::residual`'s activation-dtype math and is
    /// cast only at the hand-off to the host output dtype before addition. With no live adapter this
    /// is byte-identical to the bare base forward.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut y = self.base.forward(x)?;
        for ad in &self.adapters {
            if ad.is_zero() {
                continue;
            }
            let residual = ad.residual(x)?.to_dtype(y.dtype())?;
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
        let mut y = self.base.forward_upcast(x)?;
        for ad in &self.adapters {
            if ad.is_zero() {
                continue;
            }
            let residual = ad.residual(x)?.to_dtype(y.dtype())?;
            y = (y + residual)?;
        }
        Ok(y)
    }

    /// Whether the base loaded from an MLX-packed tier (its codes are quantized) — used to gate the
    /// residual-vs-fold route and asserted by the tests (packed survives load, no dense weight
    /// materialized).
    pub fn is_packed(&self) -> bool {
        self.base.is_packed()
    }

    /// Whether the base is an **NVFP4** projection (sc-21483). Distinct from [`Self::is_packed`],
    /// which stays the MLX-packed (`Q4_1`/`Q8_0`) predicate its existing callers key off — an NVFP4
    /// base holds no GGUF block weight and has no [`super::MatmulStrategy`].
    pub fn is_nvfp4(&self) -> bool {
        self.base.is_nvfp4()
    }

    /// Whether the base weight is held in *any* quantized regime — MLX-packed or NVFP4.
    pub fn is_quantized(&self) -> bool {
        self.is_packed() || self.is_nvfp4()
    }

    pub fn matmul_strategy(&self) -> Option<super::MatmulStrategy> {
        match &self.base {
            Base::Dense(_) | Base::Nvfp4(_) => None,
            Base::Packed(linear) => linear.matmul_strategy(),
        }
    }

    pub fn quant_dtype(&self) -> Option<candle_core::quantized::GgmlDType> {
        match &self.base {
            Base::Dense(_) | Base::Nvfp4(_) => None,
            Base::Packed(linear) => linear.quant_dtype(),
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
            Base::Dense(_) | Base::Nvfp4(_) => None,
        }
    }

    /// The NVFP4 base projection (for a consumer's footprint/regime accounting), or `None` on a
    /// dense / MLX-packed base. sc-21483.
    pub fn base_nvfp4(&self) -> Option<&super::Nvfp4Linear> {
        match &self.base {
            Base::Nvfp4(q) => Some(q),
            Base::Dense(_) | Base::Packed(_) => None,
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

    /// Whether any additive residual is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
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
        if matches!(self.base, Base::Nvfp4(_)) {
            return self.push_lora_checked(a, b, scale);
        }
        self.adapters.push(Adapter::Lora { a, b, scale });
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
        self.adapters.push(Adapter::Lora { a, b, scale });
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
            Base::Packed(_) => "MLX-packed base",
            Base::Nvfp4(_) => "NVFP4 base",
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

    /// Attach a trainable LoRA residual whose canonical factors are `Var`-backed. Unlike
    /// [`Self::push_lora`], this deliberately performs the factor transposes during each forward so
    /// every loss graph terminates at the live factor leaves and observes optimizer updates.
    pub fn push_trainable_lora(&mut self, down: Tensor, up: Tensor, scale: f64) {
        self.adapters
            .push(Adapter::TrainableLora { down, up, scale });
    }

    /// Attach live full-factor LoKr leaves for training. The base shape is captured here so the
    /// bounded Kronecker residual can be rebuilt inside every eager loss graph.
    pub fn push_trainable_lokr(&mut self, w1: Tensor, w2: Tensor, scale: f64) {
        self.adapters.push(Adapter::TrainableLokr {
            w1,
            w2,
            base_shape: self.base_shape(),
            scale,
        });
    }

    /// Attach a forward-time **structured LoKr** residual via the Kronecker vec-trick: the full
    /// `(alpha/rank)·strength` scale is already baked into `factors.w2`, so `[out,in]` is never
    /// materialized. Valid on **any** base — the base weight is untouched, so a packed q4/q8 tier keeps
    /// its footprint. Multiple pushes stack, and it mixes freely with LoRA residuals (push order).
    ///
    /// **On an NVFP4 base this delegates to [`Self::push_lokr_structured_checked`]** — same
    /// structural closure as [`Self::push_lora`] (sc-11045 fix round, MAJOR 7).
    pub fn push_lokr_structured(&mut self, factors: LokrFactors) -> Result<()> {
        if matches!(self.base, Base::Nvfp4(_)) {
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
    /// the sc-7702-safe [`super::MatmulStrategy::DequantDense`] forward (via `QLinear::quantize`); the
    /// residual stack is untouched — the deltas ride on top of the now-packed base. Only the **base**
    /// weight is quantized (never a residual factor), so a dense base carrying residuals stays correct.
    ///
    /// An **NVFP4** base is a hard error, never a no-op and never a re-fold (sc-21483): re-quantizing
    /// it to `Q4`/`Q8` — or dequantizing it to BF16 to make some other path fit — would swap the
    /// projection's numeric regime out from under a caller that asked for NVFP4. `Quant::Nvfp4` is
    /// refused too: NVFP4 is packed from the bf16 master, so "re-NVFP4ing" an NVFP4 base could only
    /// mean a double quantization.
    pub fn quantize(&mut self, quant: Quant) -> candle_core::Result<()> {
        match &mut self.base {
            // Already packed (a packed-tier load, or a prior fold) → idempotent no-op.
            Base::Packed(_) => Ok(()),
            Base::Nvfp4(_) => Err(nvfp4_refold_refusal(quant)),
            Base::Dense(l) => {
                let mut q = QLinear::from_dense(DenseLinear::Linear(l.clone()));
                q.quantize(quant)?;
                self.base = Base::Packed(q);
                Ok(())
            }
        }
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
        match &mut self.base {
            Base::Packed(_) => Ok(()),
            Base::Nvfp4(_) => Err(nvfp4_refold_refusal(quant)),
            Base::Dense(l) => {
                let mut q = QLinear::from_dense(DenseLinear::Linear(l.clone()));
                q.quantize_dequant_onto(quant, device)?;
                self.base = Base::Packed(q);
                Ok(())
            }
        }
    }

    /// Quantize the base projection with [`QLinear::quantize_onto`] while preserving all attached
    /// forward-time residuals.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> candle_core::Result<()> {
        match &mut self.base {
            Base::Packed(_) => Ok(()),
            Base::Nvfp4(_) => Err(nvfp4_refold_refusal(quant)),
            Base::Dense(l) => {
                let mut q = QLinear::from_dense(DenseLinear::Linear(l.clone()));
                q.quantize_onto(quant, device)?;
                self.base = Base::Packed(q);
                Ok(())
            }
        }
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
                *l = Linear::new(w, b);
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
        }
        for ad in &mut self.adapters {
            ad.migrate_to(device)?;
        }
        Ok(())
    }
}

impl Module for AdaptLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        AdaptLinear::forward(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::lora::reconstruct_lokr_delta;
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer (group 64): per-element 4-bit codes → u32 words (LSB-first nibbles).
    /// Returns `(wq [out, in/8] u32, scales [out, in/g], biases [out, in/g], affine grid [out, in])`.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let g = MLX_GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let gpr = in_dim / g;
        let groups = out_dim * gpr;
        // Small, BOUNDED scales/biases so the dequantized grid stays ~O(1). A large-magnitude grid
        // makes `base.forward` huge, and the residual-isolation test (adapted − base) then recovers a
        // tiny residual from a catastrophic f32 cancellation. Bounded via `% k` so it holds at any
        // group count.
        let scales: Vec<f32> = (0..groups)
            .map(|gi| 0.01 * ((gi % 5) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.03 * (gi % 7) as f32).collect();
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let gi = row * gpr + col / g;
                scales[gi] * codes[i] as f32 + biases[gi]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// A per-call unique suffix so parallel tests never share a temp file (`cargo test` runs threads in
    /// ONE process, so a pid-only name would collide — one test truncating/deleting the file another is
    /// mid-mmap on → corrupt reads / flaky failures).
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Build a packed [`AdaptLinear`] via [`AdaptLinear::detect`] on a written `.weight`/`.scales`/
    /// `.biases` triple (the round-trip the DiT loader takes) plus the affine grid it represents.
    fn packed_adapt(
        tmp: &tempfile::TempDir,
        out_dim: usize,
        in_dim: usize,
    ) -> (AdaptLinear, Tensor) {
        let dev = Device::Cpu;
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = tmp.path().join(format!("adapt_core_{}.safetensors", uniq));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let lin = AdaptLinear::detect(&vb, "p").unwrap();
        (
            lin,
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    fn deterministic_weight(out_dim: usize, in_dim: usize, dtype: DType) -> Tensor {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
            .collect();
        Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn deterministic_input(rows: usize, in_dim: usize, dtype: DType) -> Tensor {
        let values: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i % 11) as f32 + 1.0) / 8.0)
            .collect();
        Tensor::from_vec(values, (rows, in_dim), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn dense_bf16_adapt(out_dim: usize, in_dim: usize) -> AdaptLinear {
        AdaptLinear::from_dense(
            Linear::new(deterministic_weight(out_dim, in_dim, DType::BF16), None),
            in_dim,
            out_dim,
        )
    }

    fn int8_fast_adapt(quant: Quant, out_dim: usize, in_dim: usize) -> AdaptLinear {
        let mut base = QLinear::from_dense(DenseLinear::Linear(Linear::new(
            deterministic_weight(out_dim, in_dim, DType::F32),
            None,
        )));
        // `cast_back=false` is the production SAM3-style host regime: a bf16 activation is promoted
        // for QMatMul and the host result stays f32. This is the dtype boundary the residual must
        // follow, rather than assuming the input dtype is also the host-output dtype.
        base.quantize_int8_fast(quant, false, false, false).unwrap();
        AdaptLinear::from_packed(base, in_dim, out_dim)
    }

    fn poison_lora(in_dim: usize, out_dim: usize) -> (Tensor, Tensor) {
        let a =
            Tensor::from_vec(vec![f32::INFINITY; in_dim * 2], (in_dim, 2), &Device::Cpu).unwrap();
        let b = Tensor::ones((2, out_dim), DType::F32, &Device::Cpu).unwrap();
        (a, b)
    }

    fn exact_values(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
    }

    fn assert_tensor_exact(actual: &Tensor, expected: &Tensor, context: &str) {
        assert_eq!(actual.dtype(), expected.dtype(), "{context}: dtype differs");
        assert_eq!(
            exact_values(actual),
            exact_values(expected),
            "{context}: element values differ"
        );
    }

    /// sc-15444: a strength-zero LoRA is not merely numerically close to disabled: its factors must
    /// not be evaluated. Poisoned factors make the distinction observable (∞·0 would become NaN).
    /// The ordinary dense forward and the bf16-store/f32-compute `forward_upcast` must return the
    /// exact no-adapter values and dtype. Candle's CPU backend does not implement bf16 matmul, so the
    /// ordinary-forward leg uses its production f32 regime; the bf16-resident leg is exercised through
    /// the public upcast entry point used for that storage dtype.
    #[test]
    fn dense_scale_zero_skips_residual_exactly_in_forward_and_bf16_upcast() {
        let (out_dim, in_dim) = (16usize, 32usize);
        let x_f32 = deterministic_input(2, in_dim, DType::F32);
        let bare_forward_host = AdaptLinear::from_dense(
            Linear::new(deterministic_weight(out_dim, in_dim, DType::F32), None),
            in_dim,
            out_dim,
        );
        let bare_forward = bare_forward_host.forward(&x_f32).unwrap();

        let mut zero_forward_host = AdaptLinear::from_dense(
            Linear::new(deterministic_weight(out_dim, in_dim, DType::F32), None),
            in_dim,
            out_dim,
        );
        let (a, b) = poison_lora(in_dim, out_dim);
        zero_forward_host.push_lora(a, b, 0.0).unwrap();
        let zero_forward = zero_forward_host.forward(&x_f32).unwrap();
        assert_tensor_exact(&zero_forward, &bare_forward, "dense f32 forward scale=0");

        let poison_w1 = Tensor::from_vec(vec![f32::INFINITY; 4 * 4], (4, 4), &Device::Cpu).unwrap();
        let poison_w2 = Tensor::ones((4, 8), DType::F32, &Device::Cpu).unwrap();
        let factors = LokrFactors::build(
            0.0,
            (out_dim, in_dim),
            Some(&poison_w1),
            None,
            None,
            Some(&poison_w2),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        let mut zero_lokr_host = AdaptLinear::from_dense(
            Linear::new(deterministic_weight(out_dim, in_dim, DType::F32), None),
            in_dim,
            out_dim,
        );
        zero_lokr_host.push_lokr_structured(factors).unwrap();
        assert_tensor_exact(
            &zero_lokr_host.forward(&x_f32).unwrap(),
            &bare_forward,
            "dense structured LoKr scale=0",
        );

        let bare_bf16 = dense_bf16_adapt(out_dim, in_dim);
        let bare_upcast = bare_bf16.forward_upcast(&x_f32).unwrap();
        let mut zero_bf16 = dense_bf16_adapt(out_dim, in_dim);
        let (a, b) = poison_lora(in_dim, out_dim);
        zero_bf16.push_lora(a, b, 0.0).unwrap();
        let zero_upcast = zero_bf16.forward_upcast(&x_f32).unwrap();
        assert_tensor_exact(
            &zero_upcast,
            &bare_upcast,
            "dense bf16 forward_upcast scale=0",
        );
    }

    /// sc-15444: Q8/Q4 int8-fast hosts can intentionally retain an f32 result for non-f32 inputs. A
    /// live residual must be cast to that host-result dtype before addition; a zero residual must be
    /// skipped. The expected output is constructed with the same explicit host-dtype seam, then
    /// compared element-for-element for both public forward entry points. CPU Candle does not implement
    /// bf16 matmul, so f64 is the executable CPU proxy for the same `input dtype != host output dtype`
    /// boundary (the production bf16/CUDA path crosses that identical boundary).
    #[test]
    fn quantized_hosts_preserve_output_dtype_and_exact_adapter_semantics() {
        let (out_dim, in_dim, rank) = (32usize, 32usize, 2usize);
        let x = deterministic_input(2, in_dim, DType::F64);
        let a = Tensor::ones((in_dim, rank), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::ones((rank, out_dim), DType::F32, &Device::Cpu).unwrap();
        let residual = ((x
            .broadcast_matmul(&a.to_dtype(DType::F64).unwrap())
            .unwrap()
            .broadcast_matmul(&b.to_dtype(DType::F64).unwrap())
            .unwrap())
            * 0.5)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            exact_values(&residual),
            vec![vec![23.375; out_dim], vec![23.5; out_dim]],
            "the deterministic live adapter must inject the measured positive residual"
        );

        for quant in [Quant::Q8, Quant::Q4] {
            let bare = int8_fast_adapt(quant, out_dim, in_dim);
            let bare_forward = bare.forward(&x).unwrap();
            assert_eq!(bare_forward.dtype(), DType::F32);

            let mut zero = int8_fast_adapt(quant, out_dim, in_dim);
            let (poison_a, poison_b) = poison_lora(in_dim, out_dim);
            zero.push_lora(poison_a, poison_b, 0.0).unwrap();
            assert_tensor_exact(
                &zero.forward(&x).unwrap(),
                &bare_forward,
                "quantized forward scale=0",
            );
            assert_tensor_exact(
                &zero.forward_upcast(&x).unwrap(),
                &bare_forward,
                "quantized forward_upcast scale=0",
            );

            let mut live = int8_fast_adapt(quant, out_dim, in_dim);
            live.push_lora(a.clone(), b.clone(), 0.5).unwrap();
            let expected = (&bare_forward + &residual).unwrap();
            let got = live.forward(&x).unwrap();
            assert_tensor_exact(&got, &expected, "quantized forward live adapter");
            assert_tensor_exact(
                &live.forward_upcast(&x).unwrap(),
                &expected,
                "quantized forward_upcast live adapter",
            );
            assert!(
                exact_values(&got)
                    .iter()
                    .flatten()
                    .zip(exact_values(&bare_forward).iter().flatten())
                    .all(|(adapted, base)| adapted > base),
                "positive live adapter must shift every host output upward"
            );
        }
    }

    /// The hardware counterpart to the CPU dtype-boundary test: Metal supports bf16 dense matmul, so
    /// this runs the production activation dtype directly across dense, Q8, and Q4 hosts. It is gated
    /// with the backend feature and is exercised by the macOS validation lane.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_bf16_dense_q8_q4_forwards_are_exact_at_the_adapter_seam() {
        let dev = Device::new_metal(0).unwrap();
        let (out_dim, in_dim, rank) = (32usize, 32usize, 2usize);
        let x = deterministic_input(2, in_dim, DType::BF16)
            .to_device(&dev)
            .unwrap();
        let dense_weight = deterministic_weight(out_dim, in_dim, DType::BF16)
            .to_device(&dev)
            .unwrap();
        let bare_dense =
            AdaptLinear::from_dense(Linear::new(dense_weight.clone(), None), in_dim, out_dim);
        let bare_dense_y = bare_dense.forward(&x).unwrap();
        assert_eq!(bare_dense_y.dtype(), DType::BF16);

        let mut zero_dense =
            AdaptLinear::from_dense(Linear::new(dense_weight, None), in_dim, out_dim);
        let poison_a = Tensor::full(f32::INFINITY, (in_dim, rank), &dev).unwrap();
        let poison_b = Tensor::ones((rank, out_dim), DType::F32, &dev).unwrap();
        zero_dense
            .push_lora(poison_a, poison_b, 0.0)
            .expect("scale=0 poison factors are admitted on a dense bf16 host");
        assert_tensor_exact(
            &zero_dense.forward(&x).unwrap(),
            &bare_dense_y,
            "Metal dense bf16 forward scale=0",
        );
        assert_tensor_exact(
            &zero_dense.forward_upcast(&x).unwrap(),
            &bare_dense_y,
            "Metal dense bf16 forward_upcast scale=0",
        );

        let a = Tensor::ones((in_dim, rank), DType::F32, &dev).unwrap();
        let b = Tensor::ones((rank, out_dim), DType::F32, &dev).unwrap();
        let residual_bf16 = ((x
            .broadcast_matmul(&a.to_dtype(DType::BF16).unwrap())
            .unwrap()
            .broadcast_matmul(&b.to_dtype(DType::BF16).unwrap())
            .unwrap())
            * 0.5)
            .unwrap();
        let mut live_dense = AdaptLinear::from_dense(
            Linear::new(
                deterministic_weight(out_dim, in_dim, DType::BF16)
                    .to_device(&dev)
                    .unwrap(),
                None,
            ),
            in_dim,
            out_dim,
        );
        live_dense
            .push_lora(a.clone(), b.clone(), 0.5)
            .expect("rank-2 factors are admitted on a dense bf16 host");
        let dense_expected = (&bare_dense_y + &residual_bf16).unwrap();
        assert_tensor_exact(
            &live_dense.forward(&x).unwrap(),
            &dense_expected,
            "Metal dense bf16 live adapter",
        );
        assert_tensor_exact(
            &live_dense.forward_upcast(&x).unwrap(),
            &dense_expected,
            "Metal dense bf16 live adapter upcast entry",
        );

        for quant in [Quant::Q8, Quant::Q4] {
            let build = || {
                let weight = deterministic_weight(out_dim, in_dim, DType::F32)
                    .to_device(&dev)
                    .unwrap();
                let mut base = QLinear::from_dense(DenseLinear::Linear(Linear::new(weight, None)));
                base.quantize_int8_fast(quant, false, false, false).unwrap();
                AdaptLinear::from_packed(base, in_dim, out_dim)
            };

            let bare = build();
            let base_y = bare.forward(&x).unwrap();
            assert_eq!(base_y.dtype(), DType::F32);

            let mut zero = build();
            let poison_a = Tensor::full(f32::INFINITY, (in_dim, rank), &dev).unwrap();
            let poison_b = Tensor::ones((rank, out_dim), DType::F32, &dev).unwrap();
            zero.push_lora(poison_a, poison_b, 0.0)
                .expect("scale=0 poison factors are admitted on a packed host");
            assert_tensor_exact(
                &zero.forward(&x).unwrap(),
                &base_y,
                "Metal quantized bf16 forward scale=0",
            );
            assert_tensor_exact(
                &zero.forward_upcast(&x).unwrap(),
                &base_y,
                "Metal quantized bf16 forward_upcast scale=0",
            );

            let mut live = build();
            live.push_lora(a.clone(), b.clone(), 0.5)
                .expect("rank-2 factors are admitted on a packed host");
            let expected = (&base_y + residual_bf16.to_dtype(DType::F32).unwrap()).unwrap();
            assert_tensor_exact(
                &live.forward(&x).unwrap(),
                &expected,
                "Metal quantized bf16 live adapter",
            );
            assert_tensor_exact(
                &live.forward_upcast(&x).unwrap(),
                &expected,
                "Metal quantized bf16 live adapter upcast entry",
            );
        }
    }

    /// `detect` recovers the logical dims from the packed `scales` shape and keeps the base packed.
    #[test]
    fn detect_recovers_dims_and_stays_packed() {
        let tmp = tempfile::tempdir().unwrap();
        let (out_dim, in_dim) = (64usize, 128usize); // in divisible by group 64
        let (lin, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        assert!(lin.is_packed(), "`.scales` ⇒ packed base");
        assert_eq!(
            lin.base_shape(),
            (out_dim, in_dim),
            "dims from scales shape"
        );
        assert_eq!(lin.in_features(), in_dim);
        assert_eq!(lin.out_features(), out_dim);
        assert!(!lin.is_adapted());
    }

    /// The dense arm of `linear_detect` is byte-identical to the legacy `candle_nn::linear` read — a
    /// dense checkpoint (no `.scales`) loads dense, unchanged; a `.scales` sibling fires the packed arm.
    #[test]
    fn linear_detect_dense_and_packed_arms() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        // Dense arm: byte-identical to `Linear::new`.
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("d.weight".into(), w.clone());
        map.insert("d.bias".into(), bias.clone());
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard
            .path()
            .join(format!("adapt_core_{}.safetensors", uniq));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let dense = AdaptLinear::linear_detect(in_dim, out_dim, &vb, "d", true).unwrap();
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense");
        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let want = Linear::new(w, Some(bias));
        let dev_max = (dense.forward(&x).unwrap() - want.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            dev_max, 0.0,
            "dense arm deviates from the legacy linear read"
        );

        let packed = AdaptLinear::linear_detect(in_dim, out_dim, &vb, "p", false).unwrap();
        assert!(
            packed.is_packed(),
            "`.scales` ⇒ packed load, not a silent dense fallback"
        );
    }

    /// The additive LoRA residual `scale·((x·a)·b)` reproduces the **folded** `x·(W + δ)ᵀ` with
    /// `δ = (alpha/rank)·scale·(up·down)` on a dense f32 base — the additive==folded identity (tight in
    /// f32), the core weight-level property (no GPU needed).
    #[test]
    fn additive_lora_matches_folded_dense() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(); // A [rank, in]
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap(); // B [out, rank]
        let (alpha, user_scale) = (8.0f64, 0.7f64);
        let ratio = alpha / rank as f64;

        // a = downᵀ [in, rank]; b = (upᵀ·ratio) [rank, out].
        let a = down.t().unwrap().contiguous().unwrap();
        let b = (up.t().unwrap().contiguous().unwrap() * ratio).unwrap();
        let mut lin = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a, b, user_scale).unwrap();
        assert!(lin.is_adapted());

        // Folded reference: δ = ratio·user_scale·(up·down); W_merged = W + δ.
        let delta = ((up.matmul(&down).unwrap()) * (ratio * user_scale)).unwrap();
        let folded = Linear::new((w + delta).unwrap(), None);

        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();
        let dev_max = (lin.forward(&x).unwrap() - folded.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(dev_max < 1e-4, "additive vs folded deviates by {dev_max}");
    }

    /// A LoRA applied additively onto a **packed** base shifts the output, keeps the base **packed** (no
    /// dense weight materialized), and stays finite — the core acceptance on a quantized tier. A scale-0
    /// residual is an exact no-op (the mutation anchor: break the scale and this equality breaks).
    #[test]
    fn additive_lora_on_packed_shifts_stays_packed_and_scale0_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 8usize);
        let (packed_base, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        assert!(packed_base.is_packed());

        let (mut adapted, _) = packed_adapt(&tmp, out_dim, in_dim);
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        adapted.push_lora(a.clone(), b.clone(), 1.0).unwrap();
        assert!(adapted.is_packed(), "adapter must not un-pack the base");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let base_y = packed_base.forward(&x).unwrap();
        let adapted_y = adapted.forward(&x).unwrap();
        let shift = (adapted_y.sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            shift > 1e-4,
            "additive LoRA on packed did not shift ({shift})"
        );
        assert!(
            adapted_y
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite(),
            "packed additive output non-finite"
        );

        // scale 0 ⇒ exact no-op vs the un-adapted packed base.
        let (mut zero, _) = packed_adapt(&tmp, out_dim, in_dim);
        zero.push_lora(a, b, 0.0).unwrap();
        let zero_dev = (zero.forward(&x).unwrap().sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 residual must be an exact no-op on packed"
        );
    }

    /// `clear_adapters` reverts an adapted projection to its **bare base** — the per-phase toggle-off a
    /// multi-phase render runs on its job-local DiT between phases. After clearing, the forward is
    /// **byte-identical** to the un-adapted base, and a re-push installs a fresh residual — proving a
    /// phase's adapter set is authoritative regardless of what the prior phase installed, and that the
    /// base weight is never disturbed by the toggle.
    #[test]
    fn clear_adapters_reverts_to_bare_base_and_repush_reinstalls() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();

        // The bare-base reference (same weight, no residual).
        let base = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        let base_y = base.forward(&x).unwrap();

        let mut lin = AdaptLinear::from_dense(Linear::new(w, None), in_dim, out_dim);
        lin.push_lora(a.clone(), b.clone(), 0.7).unwrap();
        assert!(lin.is_adapted());
        let adapted_shift = (lin.forward(&x).unwrap() - &base_y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(adapted_shift > 1e-4, "the residual must shift the forward");

        // Toggle off: clear ⇒ byte-identical to the bare base (an exact 0 deviation, not a tolerance).
        lin.clear_adapters();
        assert!(!lin.is_adapted(), "clear must drop every residual");
        let cleared_dev = (lin.forward(&x).unwrap() - &base_y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            cleared_dev, 0.0,
            "after clear the forward must equal the bare base exactly"
        );

        // Re-push a fresh (different-scale) residual — the phase's set is authoritative after a clear.
        lin.push_lora(a, b, 0.3).unwrap();
        assert!(lin.is_adapted());
        let repush_shift = (lin.forward(&x).unwrap() - &base_y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            repush_shift > 1e-4 && (repush_shift - adapted_shift).abs() > 1e-6,
            "re-push installs a fresh residual at the new scale"
        );
    }

    /// The **acceptance parity** on a packed base, isolated from the base's own quant error: the
    /// residual the adapter contributes equals `scale·((x·a)·b)` **exactly** (f32). `adapted.forward −
    /// base.forward` cancels the (bit-identical) packed base — the dequant-repack quant error included —
    /// leaving only the residual. This proves the LoRA is added **additively over the packed weight**
    /// (never folded into a re-quantized dense weight).
    #[test]
    fn additive_on_packed_adds_exact_residual_over_base() {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);
        let (base, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(&tmp, out_dim, in_dim); // bit-identical packed base
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        adapted.push_lora(a.clone(), b.clone(), scale).unwrap();
        assert!(adapted.is_packed(), "base stays packed under the residual");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        // The residual the adapter contributes = adapted − base (identical packed bases cancel exactly).
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        // Expected: scale·((x·a)·b).
        let residual_expected = ((x.matmul(&a).unwrap().matmul(&b).unwrap()) * scale).unwrap();
        let dev_max = (residual_actual - residual_expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-5,
            "packed residual != scale·(x·a)·b (max diff {dev_max})"
        );
    }

    /// `quantize` folds a **dense** base to packed in place (idempotent no-op if already packed), keeping
    /// any attached residual — the lens/sd3 "quantize after the dense fold" contract (sc-11105). The
    /// folded-packed forward equals the packed base forward plus the same residual (f32 tol), and a
    /// second quantize on the now-packed base is an exact no-op (the additive residuals survive).
    #[test]
    fn quantize_folds_dense_base_and_preserves_residual() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();

        let mut lin = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a.clone(), b.clone(), 0.7).unwrap();
        assert!(!lin.is_packed());
        lin.quantize(Quant::Q8).unwrap();
        assert!(lin.is_packed(), "dense base must fold to packed");
        assert!(lin.is_adapted(), "residual must survive the fold");

        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();
        let residual = ((x.matmul(&a).unwrap().matmul(&b).unwrap()) * 0.7).unwrap();
        // The Q8-packed base alone (no residual) — same weight, same fold — plus the residual.
        let mut base_only = AdaptLinear::from_dense(Linear::new(w, None), in_dim, out_dim);
        base_only.quantize(Quant::Q8).unwrap();
        let expected = (base_only.forward(&x).unwrap() + residual).unwrap();
        let dev_max = (lin.forward(&x).unwrap() - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-4,
            "adapted-packed forward != packed base + residual ({dev_max})"
        );

        // Idempotent: a second quantize is a no-op (packed stays packed; forward unchanged).
        let y0 = lin.forward(&x).unwrap();
        lin.quantize(Quant::Q4).unwrap();
        let y1 = lin.forward(&x).unwrap();
        let noop = (y1 - y0)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            noop, 0.0,
            "quantize on an already-packed base must be a no-op"
        );
    }

    // ---- Structured (deferred) LoKr — the Kronecker vec-trick ---------------------------------------

    /// `x·ΔWᵀ` — the materialized-delta residual, the reference the vec-trick must reproduce.
    fn delta_residual(x: &Tensor, delta: &Tensor) -> Tensor {
        x.matmul(&delta.t().unwrap().contiguous().unwrap()).unwrap()
    }

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// **Full-factor parity.** The structured vec-trick residual `vec(w1·X·w2ᵀ)` equals the
    /// materialized-delta residual `x·(kron(w1,w2))ᵀ` for a full `w1⊗w2` LoKr — proving the row-major
    /// Kronecker identity the whole port rests on, against candle's own `reconstruct_lokr_delta` fold.
    /// The built factors are the SMALL `[a,c]`/`[b,d]` matrices — the `[out,in]` delta is NEVER formed.
    #[test]
    fn structured_lokr_full_matches_reconstruct_delta() {
        let dev = Device::Cpu;
        let (a, b, c, d) = (2usize, 3, 4, 5);
        let (out, inp) = (a * b, c * d);
        let w1 = Tensor::from_vec(
            (0..(a * c))
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            (a, c),
            &dev,
        )
        .unwrap();
        let w2 = Tensor::from_vec(
            (0..(b * d))
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            (b, d),
            &dev,
        )
        .unwrap();
        let scale = 0.9f64;
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out, inp),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out, inp),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        assert_eq!(factors.w1.dims(), &[a, c], "left factor stays [a,c]");
        assert_eq!(factors.w2.dims(), &[b, d], "right factor stays [b,d]");
        assert!(
            factors.w1.elem_count() + factors.w2.elem_count() < out * inp,
            "structured factors must be « the [out,in] delta"
        );

        let x = Tensor::from_vec(
            (0..(2 * inp))
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            (2, inp),
            &dev,
        )
        .unwrap();
        let want = delta_residual(&x, &delta);
        let got = factors.residual(&x).unwrap();
        assert_eq!(got.dims(), &[2, out]);
        let dev_max = max_abs(&(got.clone() - &want).unwrap());
        assert!(
            dev_max < 1e-4,
            "structured LoKr residual != materialized-delta residual ({dev_max})"
        );
        assert!(
            max_abs(&got) > 1e-2,
            "the LoKr residual must be materially non-zero"
        );
    }

    /// **Decomposed-factor parity.** Same identity for a low-rank LoKr (`w1_a·w1_b`, `w2_a·w2_b`): the
    /// inner products are materialized only as the SMALL `[a,c]`/`[b,d]` factors, never `[out,in]`.
    #[test]
    fn structured_lokr_decomposed_matches_reconstruct_delta() {
        let dev = Device::Cpu;
        let (a, b, c, d, r) = (3usize, 2, 5, 4, 2);
        let (out, inp) = (a * b, c * d);
        let mk = |rows: usize, cols: usize, seed: f32| {
            Tensor::from_vec(
                (0..(rows * cols))
                    .map(|i| (i as f32 * 0.09 + seed).sin() * 0.3)
                    .collect::<Vec<_>>(),
                (rows, cols),
                &dev,
            )
            .unwrap()
        };
        let (w1a, w1b) = (mk(a, r, 0.1), mk(r, c, 0.2)); // w1 = [a,c]
        let (w2a, w2b) = (mk(b, r, 0.3), mk(r, d, 0.4)); // w2 = [b,d]
        let scale = 1.3f64;
        let delta = reconstruct_lokr_delta(
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            Some(&w2a),
            Some(&w2b),
            1.0,
            1.0,
            scale as f32,
            (out, inp),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out, inp),
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
        )
        .unwrap()
        .expect("a decomposed linear LoKr is deferrable");
        assert_eq!(factors.w1.dims(), &[a, c]);
        assert_eq!(factors.w2.dims(), &[b, d]);

        let x = Tensor::from_vec(
            (0..inp)
                .map(|i| (i as f32 * 0.02).cos())
                .collect::<Vec<_>>(),
            (1, inp),
            &dev,
        )
        .unwrap();
        let dev_max =
            max_abs(&(factors.residual(&x).unwrap() - delta_residual(&x, &delta)).unwrap());
        assert!(
            dev_max < 1e-4,
            "decomposed structured LoKr != materialized delta ({dev_max})"
        );
    }

    /// **Acceptance parity on a PACKED base.** The structured LoKr installs on a packed q4 base, the
    /// base stays **packed** (no `[out,in]` weight materialized), and `packed_forward + residual`
    /// reproduces `packed_forward + folded_delta` within quant tolerance. Also the mutation anchor: a
    /// scale-0 LoKr is an exact no-op.
    #[test]
    fn structured_lokr_on_packed_matches_folded_and_stays_packed() {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        // in = c·d = 8·16 = 128 (divisible by group 64); out = a·b = 4·16 = 64.
        let (a, b, c, d) = (4usize, 16, 8, 16);
        let (out_dim, in_dim) = (a * b, c * d);
        let (base, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(&tmp, out_dim, in_dim); // bit-identical packed base
        assert!(base.is_packed() && adapted.is_packed());

        let w1 = (Tensor::randn(0f32, 1f32, (a, c), &dev).unwrap() * 0.1).unwrap();
        let w2 = (Tensor::randn(0f32, 1f32, (b, d), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        let factors = LokrFactors::build(
            scale,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("deferrable");
        assert!(factors.w1.elem_count() + factors.w2.elem_count() < out_dim * in_dim);
        adapted.push_lokr_structured(factors).unwrap();
        assert!(
            adapted.is_packed(),
            "structured LoKr must not un-pack the base"
        );

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out_dim, in_dim),
        )
        .unwrap();
        let dev_max = max_abs(&(residual_actual.clone() - delta_residual(&x, &delta)).unwrap());
        assert!(
            dev_max < 1e-4,
            "packed structured LoKr residual != folded delta ({dev_max})"
        );
        assert!(
            max_abs(&residual_actual) > 1e-3,
            "the LoKr must shift the packed forward"
        );

        // Mutation: a scale-0 structured LoKr is an exact no-op over the packed base.
        let (mut zero, _) = packed_adapt(&tmp, out_dim, in_dim);
        let f0 = LokrFactors::build(
            0.0,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        zero.push_lokr_structured(f0).unwrap();
        let zero_dev = max_abs(&(zero.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap());
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 structured LoKr must be an exact no-op"
        );
    }

    /// A tucker/CP `w2` (`lokr_t2`, conv-only) and a base that does not factor as `a·b × c·d` are both
    /// NOT deferrable via the 2-D vec-trick → `Ok(None)`, so the installer rejects them on a packed tier
    /// rather than materializing. A missing `w1`/`w2` leg is a typed error, never a panic.
    #[test]
    fn structured_lokr_non_deferrable_and_missing_legs() {
        let dev = Device::Cpu;
        let w1 = Tensor::zeros((3usize, 4), DType::F32, &dev).unwrap();
        let w2a = Tensor::zeros((2usize, 4), DType::F32, &dev).unwrap();
        let w2b = Tensor::zeros((2usize, 5), DType::F32, &dev).unwrap();
        let t2 = Tensor::zeros((2usize, 2, 3, 3), DType::F32, &dev).unwrap();
        // Tucker `w2_t2` present → None (the guard fires before any shape check).
        let got = LokrFactors::build(
            1.0,
            (24, 180),
            Some(&w1),
            None,
            None,
            None,
            Some(&t2),
            Some(&w2a),
            Some(&w2b),
        )
        .unwrap();
        assert!(got.is_none(), "tucker/CP LoKr must be non-deferrable");

        // A base that does not factor as a·b × c·d (here a·b = 3·2 = 6 ≠ out = 7) → None.
        let w2 = Tensor::zeros((2usize, 5), DType::F32, &dev).unwrap();
        let mism = LokrFactors::build(
            1.0,
            (7, 20),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(
            mism.is_none(),
            "a base that doesn't factor a·b×c·d is non-deferrable"
        );

        // A missing w2 leg (no full, no a/b) is a typed error, not a panic.
        let err = LokrFactors::build(1.0, (6, 20), Some(&w1), None, None, None, None, None, None);
        assert!(err.is_err(), "missing w2 must be a typed error");
    }

    /// A LoRA residual on an activation with a **non-1 leading (batch) dim** must fold those leading
    /// dims into the GEMM's `M` — never `broadcast_matmul`, which physically copies the 2-D factor once
    /// per batch element. The Krea text-fusion `layerwise_blocks` run `[n_tokens, num_layers, d]`, so at
    /// a 2048² image edit (`n_tokens = 4107`, `d = 2560`, rank 256) the old broadcast path materialized a
    /// 5.4 GB copy per residual leg and the first denoise step never finished. This pins the flattened
    /// path's NUMERICS against a per-batch-slice reference, so the perf fix can't silently change the math.
    #[test]
    fn lora_residual_on_batched_activation_matches_per_slice_reference() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank, scale) = (12usize, 16usize, 4usize, 0.7f64);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let a = Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap(); // [in, rank]
        let b = Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap(); // [rank, out]
        let mut lin = AdaptLinear::from_dense(Linear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a.clone(), b.clone(), scale).unwrap();

        // `[N, S, in]`: N is a BATCH of token stacks (the layerwise-block shape), S the stack length.
        let (n, s) = (5usize, 3usize);
        let x = Tensor::randn(0f32, 1f32, (n, s, in_dim), &dev).unwrap();
        let got = lin.forward(&x).unwrap();
        assert_eq!(got.dims(), &[n, s, out_dim]);

        for i in 0..n {
            let xi = x.narrow(0, i, 1).unwrap().reshape((s, in_dim)).unwrap();
            let base = Linear::new(w.clone(), None).forward(&xi).unwrap();
            let res = (xi.matmul(&a).unwrap().matmul(&b).unwrap() * scale).unwrap();
            let want = (base + res).unwrap();
            let mine = got.narrow(0, i, 1).unwrap().reshape((s, out_dim)).unwrap();
            let d = (mine - want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_vec0::<f32>()
                .unwrap();
            assert!(d < 1e-5, "batch slice {i} diverged from the reference: {d}");
        }
    }

    // ------------------------------------------------------------------------------------------
    // NVFP4 as an adapter-capable base (sc-21483, epic 11037).
    //
    // These run on the CPU lane: `Nvfp4Linear` retains the packed host container in every regime and
    // serves the transparent dequant→bf16 fallback off `sm_120`, so the ADDITIVE arm — the thing this
    // story adds — is fully exercisable without a Blackwell device. The CUDA lane adds the FP4 GEMM
    // under the same arm; that is what the krea real-weight render covers.
    // ------------------------------------------------------------------------------------------

    /// An NVFP4-based [`AdaptLinear`] plus the dense master it was packed from.
    fn nvfp4_host(out_dim: usize, in_dim: usize) -> AdaptLinear {
        let dev = Device::Cpu;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 11.0)
            .collect();
        let w = Tensor::from_vec(w, (out_dim, in_dim), &dev).unwrap();
        let lin = super::super::Nvfp4Linear::from_dense(
            &w,
            None,
            &dev,
            super::super::ActPrecision::W4A16,
        )
        .unwrap();
        AdaptLinear::from_nvfp4(lin)
    }

    /// A deterministic `a [in, rank]` / `b [rank, out]` LoRA factor pair on the CPU.
    fn lora_pair(in_dim: usize, rank: usize, out_dim: usize) -> (Tensor, Tensor) {
        let dev = Device::Cpu;
        let a: Vec<f32> = (0..in_dim * rank).map(|i| (i % 5) as f32 * 0.03).collect();
        let b: Vec<f32> = (0..rank * out_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        (
            Tensor::from_vec(a, (in_dim, rank), &dev).unwrap(),
            Tensor::from_vec(b, (rank, out_dim), &dev).unwrap(),
        )
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// AC#1 — identity / delta / removal at the shared seam. A LoRA installed over an NVFP4 base
    /// contributes exactly `scale·((x·a)·b)` on top of the packed forward, and `clear_adapters`
    /// restores the base output **exactly**: the packed weight was never mutated, so this is bit
    /// equality, not a tolerance.
    #[test]
    fn nvfp4_base_hosts_an_additive_lora_and_restores_exactly_on_removal() {
        let (out_dim, in_dim, rank) = (32, 64, 4);
        let mut host = nvfp4_host(out_dim, in_dim);
        assert!(host.is_nvfp4(), "the base must be an NVFP4 projection");
        assert!(host.is_quantized(), "NVFP4 is a quantized regime");
        assert!(!host.is_packed(), "NVFP4 is not the MLX-packed regime");
        assert_eq!(host.base_shape(), (out_dim, in_dim));
        assert!(host.base_nvfp4().is_some());

        let x = Tensor::from_vec(
            (0..8 * in_dim)
                .map(|i| (i % 13) as f32 * 0.05)
                .collect::<Vec<_>>(),
            (8, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let bare = host.forward(&x).unwrap();

        let (a, b) = lora_pair(in_dim, rank, out_dim);
        host.push_lora_checked(a.clone(), b.clone(), 0.75).unwrap();
        assert!(host.is_adapted());
        let adapted = host.forward(&x).unwrap();

        // The delta is the LoRA's own product — the packed base contributed nothing extra.
        let want_delta = (x.matmul(&a).unwrap().matmul(&b).unwrap() * 0.75).unwrap();
        let got_delta = (&adapted - &bare).unwrap();
        assert!(
            max_abs(&(got_delta - &want_delta).unwrap()) < 1e-5,
            "the additive arm did not contribute the LoRA delta"
        );
        assert!(
            max_abs(&(adapted - &bare).unwrap()) > 1e-4,
            "the adapter must actually move the output"
        );

        // Removal restores the base output EXACTLY.
        host.clear_adapters();
        assert!(!host.is_adapted());
        assert_eq!(
            flat(&host.forward(&x).unwrap()),
            flat(&bare),
            "clearing the adapter must restore the exact base output"
        );
        assert!(host.is_nvfp4(), "the base regime survived the whole cycle");
    }

    /// AC#1 — a structured LoKr rides the same NVFP4 base through the Kronecker vec-trick, matching
    /// the reconstructed dense delta, and removal is again exact.
    #[test]
    fn nvfp4_base_hosts_a_structured_lokr_residual() {
        let (a, b, c, d) = (4usize, 8usize, 8usize, 8usize);
        let (out_dim, in_dim) = (a * b, c * d);
        let dev = Device::Cpu;
        let mut host = nvfp4_host(out_dim, in_dim);
        let w1 = Tensor::from_vec(
            (0..(a * c))
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            (a, c),
            &dev,
        )
        .unwrap();
        let w2 = Tensor::from_vec(
            (0..(b * d))
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            (b, d),
            &dev,
        )
        .unwrap();
        let scale = 0.5f64;
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out_dim, in_dim),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");

        let x = Tensor::from_vec(
            (0..(4 * in_dim))
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            (4, in_dim),
            &dev,
        )
        .unwrap();
        let bare = host.forward(&x).unwrap();
        host.push_lokr_structured_checked(factors).unwrap();
        let adapted = host.forward(&x).unwrap();

        let want = (&bare + delta_residual(&x, &delta)).unwrap();
        assert!(
            max_abs(&(adapted - &want).unwrap()) < 1e-4,
            "structured LoKr over NVFP4 diverged from the reconstructed delta"
        );

        host.clear_adapters();
        assert_eq!(flat(&host.forward(&x).unwrap()), flat(&bare));
    }

    /// **sc-11045 fix round (MAJOR 7, minor): the unchecked pushes delegate to the checked forms
    /// on an NVFP4 base**, so the bypass is structurally closed — no caller can attach an
    /// unvalidated factor to a base that has no fallback. Dense/MLX-packed bases keep the
    /// historical unchecked behaviour.
    ///
    /// # Mutation
    ///
    /// Remove the `Base::Nvfp4` delegation from `push_lora`/`push_lokr_structured` (restore the
    /// plain `self.adapters.push(...)`): both `unwrap_err`s below go red.
    #[test]
    fn the_unchecked_pushes_delegate_to_the_checked_forms_on_an_nvfp4_base() {
        let (out_dim, in_dim) = (32, 64);
        let mut host = nvfp4_host(out_dim, in_dim);
        // A mis-shaped LoRA through the UNCHECKED entry point refuses on the NVFP4 base.
        let (bad_a, bad_b) = lora_pair(16, 4, out_dim);
        let error = host.push_lora(bad_a, bad_b, 1.0).unwrap_err().to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(!host.is_adapted());
        // A mis-reconstructing LoKr likewise: factors built for a [16, 64] base.
        let dev = Device::Cpu;
        let w1 = Tensor::ones((2, 8), DType::F32, &dev).unwrap();
        let w2 = Tensor::ones((8, 8), DType::F32, &dev).unwrap();
        let wrong = LokrFactors::build(
            1.0,
            (16, 64),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        let error = host.push_lokr_structured(wrong).unwrap_err().to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(!host.is_adapted());
        // Well-shaped factors still install through the same unchecked names.
        let (a, b) = lora_pair(in_dim, 4, out_dim);
        host.push_lora(a, b, 1.0).unwrap();
        assert!(host.is_adapted());
        // And a dense base keeps the historical unchecked semantics: no validation, always Ok.
        let mut dense = AdaptLinear::from_dense(
            Linear::new(
                Tensor::zeros((out_dim, in_dim), DType::F32, &dev).unwrap(),
                None,
            ),
            in_dim,
            out_dim,
        );
        let (bad_a, bad_b) = lora_pair(16, 4, out_dim);
        dense.push_lora(bad_a, bad_b, 1.0).unwrap();
    }

    /// AC#2, shape half — a factor that does not compose against the base is refused at ADMISSION,
    /// naming the NVFP4 regime, and nothing is attached. It never reaches a sampler step.
    #[test]
    fn nvfp4_base_refuses_a_mis_shaped_adapter_at_admission() {
        let (out_dim, in_dim) = (32, 64);
        let mut host = nvfp4_host(out_dim, in_dim);

        // `a` contracts against the wrong input width…
        let (a, b) = lora_pair(in_dim / 2, 4, out_dim);
        let error = host.push_lora_checked(a, b, 1.0).unwrap_err().to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(error.contains("do not compose"), "{error}");
        assert!(!host.is_adapted(), "a refused factor must not be attached");

        // …and the wrong output width.
        let (a, b) = lora_pair(in_dim, 4, out_dim + 1);
        assert!(host.push_lora_checked(a, b, 1.0).is_err());
        assert!(!host.is_adapted());

        // A LoKr whose Kronecker factors reconstruct a different projection is refused too.
        let w1 = Tensor::from_vec(vec![0.1f32; 4 * 8], (4, 8), &Device::Cpu).unwrap();
        let w2 = Tensor::from_vec(vec![0.1f32; 4 * 8], (4, 8), &Device::Cpu).unwrap();
        let factors = LokrFactors::build(
            0.5,
            (32, 64),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap();
        if let Some(factors) = factors {
            // Re-host the same factors on a DIFFERENT projection shape: admission must reject.
            let mut other = nvfp4_host(64, 32);
            let error = other
                .push_lokr_structured_checked(factors)
                .unwrap_err()
                .to_string();
            assert!(error.contains("NVFP4 base"), "{error}");
            assert!(error.contains("not the base"), "{error}");
            assert!(!other.is_adapted());
        }
    }

    /// AC#2, dtype/device half — an additive residual is cast to the activation dtype per forward,
    /// so a non-float factor is refused rather than silently reinterpreted.
    #[test]
    fn nvfp4_base_refuses_a_non_float_adapter_factor() {
        let (out_dim, in_dim, rank) = (32, 64, 4);
        let mut host = nvfp4_host(out_dim, in_dim);
        let (a, b) = lora_pair(in_dim, rank, out_dim);
        let a_u32 = a.to_dtype(DType::U32).unwrap();
        let error = host
            .push_lora_checked(a_u32, b, 1.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(error.contains("non-float dtype"), "{error}");
        assert!(!host.is_adapted());
    }

    /// AC#2, regime half — an NVFP4 base is NEVER silently converted to q4/q8 (or dequantized to
    /// BF16) to make something fit. Every fold entry point refuses, and the base stays NVFP4.
    #[test]
    fn nvfp4_base_refuses_every_requantization() {
        let mut host = nvfp4_host(32, 64);
        for quant in [Quant::Q4, Quant::Q8, Quant::Nvfp4] {
            let error = host.quantize(quant).unwrap_err().to_string();
            assert!(error.contains("refusing to re-quantize"), "{error}");
            let error = host
                .quantize_onto(quant, &Device::Cpu)
                .unwrap_err()
                .to_string();
            assert!(error.contains("refusing to re-quantize"), "{error}");
            let error = host
                .quantize_dequant_onto(quant, &Device::Cpu)
                .unwrap_err()
                .to_string();
            assert!(error.contains("refusing to re-quantize"), "{error}");
        }
        assert!(
            host.is_nvfp4(),
            "the base must still be NVFP4 after the refusals"
        );
        assert!(host.base_nvfp4().is_some());
        assert!(host.matmul_strategy().is_none());
        assert!(host.quant_dtype().is_none());
    }

    /// A same-device migration is the ordinary no-op (and still migrates residual factors); the
    /// NVFP4 base is never re-staged behind the caller's back.
    #[test]
    fn nvfp4_base_same_device_migration_is_a_noop() {
        let (out_dim, in_dim, rank) = (32, 64, 4);
        let mut host = nvfp4_host(out_dim, in_dim);
        let (a, b) = lora_pair(in_dim, rank, out_dim);
        host.push_lora_checked(a, b, 0.5).unwrap();
        host.to_device(&Device::Cpu).unwrap();
        assert!(host.is_nvfp4());
        assert!(host.is_adapted());
    }

    /// The checked pushes are not NVFP4-only: they are the shared admission gate, so a dense host
    /// validates identically (and a valid factor still installs).
    #[test]
    fn checked_push_validates_a_dense_host_too() {
        let (out_dim, in_dim, rank) = (16, 8, 2);
        let w = Tensor::from_vec(
            (0..out_dim * in_dim)
                .map(|i| i as f32 * 0.01)
                .collect::<Vec<_>>(),
            (out_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let mut host = AdaptLinear::from_dense(Linear::new(w, None), in_dim, out_dim);
        let (a, b) = lora_pair(in_dim, rank, out_dim);
        host.push_lora_checked(a, b, 1.0).unwrap();
        assert!(host.is_adapted());

        let (a, b) = lora_pair(in_dim + 1, rank, out_dim);
        let error = host.push_lora_checked(a, b, 1.0).unwrap_err().to_string();
        assert!(error.contains("dense base"), "{error}");
    }
}
