//! The fused **video-VAE norm + SiLU + temporal pad** (SW-6133, feature SW-F287 S8): one nvrtc
//! launch ([`VAE_NORM_SILU_PAD_SRC`], compiled once per device) reads a raw activation
//! `x [B, C, T, H, W]` and writes `pad(silu(norm(x)))` — the temporally padded input a VAE causal
//! 3-D convolution consumes — directly, so the normalized tensor, the SiLU tensor and the
//! concatenated / padded copies the op chain builds are never materialized.
//!
//! # The two norms and the pads it serves
//!
//! - [`VaeNorm::ChannelL2`] — Wan's `WanRMS_norm` (candle-gen-wan `ChanNorm`):
//!   `x / max(‖x‖₂ over C, eps) · scale · γ` (`scale = √C`, `eps = 1e-12`).
//! - [`VaeNorm::PixelRms`] — LTX's `PixelNorm` (candle-gen-ltx `pixel_norm`):
//!   `x / √(mean(x² over C) + eps)` (no scale, no γ).
//!
//! The time axis of the output is `[zeros][cache][replicated first][act(x)][replicated last][zeros]`
//! ([`TemporalPad`] + an optional cache): Wan's causal conv prepends its streaming `feat_cache`
//! (frames that are *already* activated, copied bit for bit) and zero-pads the frames the cache
//! has not warmed yet; LTX's conv replicates the first / last frame. The output is channel-major
//! `[B, C, Tp, H, W]` (the op chain's own layout) or frame-major `[B, Tp, C, H, W]`
//! ([`ConvInputLayout::FrameMajor`]), where every conv tap's `T` frames are already the merged
//! `[B·T, C, H, W]` conv2d batch — a view, not a permute copy, when `B == 1`.
//!
//! # Numerics (epic E2)
//!
//! The kernel reduces `Σ x²` in f32 and applies the norm and SiLU in f32, rounding **once** to the
//! output dtype; cache frames and zero frames are exact. On f32 inputs every operation is the op
//! chain's own, in its order and without contraction — candle's `fast_sum` reduction tree included —
//! so the f32 result is **bit-identical** to the chain (E2's 1e-5 bound, met with zero error). The
//! bf16 / f16 op chains round
//! after every op (the L2 scalar, the divide, the scale, the γ product and SiLU's exp / add /
//! divide), so the fused 16-bit result is held to the **one-rounding** bound of the f32 math —
//! within half an output ULP of the f32 oracle — and to being no worse than the 16-bit chain; see
//! the GPU tests in `cuda_tests`.
//!
//! # Path selection (epic E1/E5)
//!
//! Callers go through [`VAE_NORM_SILU_PAD_PATH`] (the media fused-path seam,
//! [`crate::fused_path`]; switch `CANDLE_FUSED_VAE_NORM_SILU_PAD`) with their own op chain as the
//! reference. [`check_norm_silu_pad`] is the pure plan: a CPU or Metal tensor, a dtype other than
//! F32 / BF16 / F16, a gradient-tracking input (E9), a shape the kernel does not serve, an
//! `H × W` plane that is not contiguous, or more than [`VAE_NSP_MAX_ELEMS`] elements in any operand
//! (E6) declines with a typed reason. [`norm_silu_pad_reference`] reproduces the providers' chains
//! op for op as this kernel's parity oracle and bench baseline.

use std::fmt;

use candle_core::{DType, Result, Tensor};

use crate::fused_path::{Attempt, FusedKernelPath, PathReason};
use crate::nvrtc::KernelSource;

/// The fused norm + SiLU + pad source; compiled once per device on first use.
pub const VAE_NORM_SILU_PAD_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_vae_norm_silu_pad_v1",
    src: include_str!("vae_norm_silu_pad.cu"),
    // f32 arithmetic, `cvt.f32.f16` / `cvt.rn.f16.f32` and software bf16 rounding only: no
    // instruction newer than sm_70, the oldest this workspace targets.
    cc_floor: (7, 0),
};

/// The kernel's path record (Wan `ChanNorm` and LTX `pixel_norm` conv inputs).
pub static VAE_NORM_SILU_PAD_PATH: FusedKernelPath =
    FusedKernelPath::new("vae_norm_silu_pad", "CANDLE_FUSED_VAE_NORM_SILU_PAD");

/// Largest operand (input, cache or output) the kernel launches on (epic E6).
pub const VAE_NSP_MAX_ELEMS: usize = i32::MAX as usize;
/// Threads per block (`NSP_THREADS` in the source).
pub const VAE_NSP_THREADS: u32 = 256;
/// Most blocks one launch uses (grid-stride beyond it).
pub const VAE_NSP_MAX_BLOCKS: u32 = 1 << 20;

/// The channel norm applied before SiLU.
#[derive(Clone, Copy, Debug)]
pub enum VaeNorm<'a> {
    /// Wan `x / min(max(‖x‖₂, eps), 1e30) · scale · γ_c` (`γ`: `[C]`, `x`'s dtype).
    ChannelL2 {
        /// The per-channel gain, `[C]` contiguous.
        gamma: &'a Tensor,
        /// `√C` for Wan.
        scale: f64,
        /// The L2 floor (`1e-12` for Wan).
        eps: f64,
    },
    /// LTX `x / √(Σ x² · (1/C) + eps)`.
    PixelRms {
        /// `1e-8` (decoder) or `1e-6` (encoder) for LTX.
        eps: f64,
    },
}

impl<'a> VaeNorm<'a> {
    fn gamma(&self) -> Option<&'a Tensor> {
        match *self {
            Self::ChannelL2 { gamma, .. } => Some(gamma),
            Self::PixelRms { .. } => None,
        }
    }
}

/// The time-axis padding around `[cache][act(x)]`, in frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TemporalPad {
    /// Zero frames before everything (Wan's causal pad).
    pub zero_front: usize,
    /// Copies of `act(x[0])` before `act(x)` (LTX).
    pub replicate_front: usize,
    /// Copies of `act(x[T-1])` after `act(x)` (LTX's non-causal decoder conv).
    pub replicate_back: usize,
    /// Zero frames after everything.
    pub zero_back: usize,
}

/// Where the padded time axis sits in the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvInputLayout {
    /// `[B, C, Tp, H, W]` — the op chain's layout.
    ChannelMajor,
    /// `[B, Tp, C, H, W]` — each frame's `[C, H, W]` contiguous, the merged conv2d batch layout.
    FrameMajor,
}

/// One operand's metadata for the plan check.
#[derive(Clone, Copy, Debug)]
pub struct NspOperand<'a> {
    /// The operand's dtype.
    pub dtype: DType,
    /// Its dims.
    pub dims: &'a [usize],
    /// Its element strides.
    pub strides: &'a [usize],
}

impl<'a> NspOperand<'a> {
    /// The metadata of `t`.
    pub fn of(t: &'a Tensor) -> Self {
        Self {
            dtype: t.dtype(),
            dims: t.dims(),
            strides: t.layout().stride(),
        }
    }
}

/// Everything [`check_norm_silu_pad`] decides on (host metadata only).
#[derive(Clone, Copy, Debug)]
pub struct NspRequest<'a> {
    /// Whether `x` lives on a CUDA device.
    pub on_cuda: bool,
    /// Whether any operand carries gradient tracking.
    pub tracks_grad: bool,
    /// The raw activation, `[B, C, T, H, W]`.
    pub x: NspOperand<'a>,
    /// `γ` for [`VaeNorm::ChannelL2`]; `None` for [`VaeNorm::PixelRms`].
    pub gamma: Option<NspOperand<'a>>,
    /// Whether the norm needs `γ`.
    pub needs_gamma: bool,
    /// The activated cache frames, `[B, C, n, H, W]`.
    pub cache: Option<NspOperand<'a>>,
    /// The padding.
    pub pad: TemporalPad,
}

impl<'a> NspRequest<'a> {
    /// The request for a [`norm_silu_pad`] call.
    pub fn of(
        x: &'a Tensor,
        norm: &VaeNorm<'a>,
        cache: Option<&'a Tensor>,
        pad: TemporalPad,
    ) -> Self {
        let gamma = norm.gamma();
        Self {
            on_cuda: x.device().is_cuda(),
            tracks_grad: x.track_op()
                || gamma.is_some_and(Tensor::track_op)
                || cache.is_some_and(Tensor::track_op),
            x: NspOperand::of(x),
            gamma: gamma.map(NspOperand::of),
            needs_gamma: gamma.is_some(),
            cache: cache.map(NspOperand::of),
            pad,
        }
    }
}

/// Why the fused kernel does not serve a call. The reference serves it instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NspRefusal {
    /// `x` is not on a CUDA device.
    NotCuda,
    /// Only `F32`, `BF16` and `F16` are served, with `γ` and the cache in `x`'s dtype.
    Dtype(DType),
    /// An operand tracks gradients: training graphs keep the composable ops (E9).
    Autograd,
    /// A rank or extent the kernel does not serve.
    Shape(&'static str),
    /// An operand's `H × W` plane (or `γ`) is not contiguous.
    Layout(&'static str),
    /// An operand has more than [`VAE_NSP_MAX_ELEMS`] elements.
    TooLarge(usize),
}

impl NspRefusal {
    /// The fused-path reason this refusal is counted under.
    pub fn reason(&self) -> PathReason {
        match self {
            Self::NotCuda => PathReason::NotCuda,
            Self::Dtype(d) => PathReason::Dtype(*d),
            Self::Autograd => PathReason::Autograd,
            Self::Shape(_) => PathReason::Shape,
            Self::Layout(_) => PathReason::Layout,
            Self::TooLarge(_) => PathReason::IndexOverflow,
        }
    }
}

impl fmt::Display for NspRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "tensor is not on a CUDA device"),
            Self::Dtype(d) => write!(f, "dtype {d:?} is not served (F32, BF16 or F16, all alike)"),
            Self::Autograd => write!(f, "an operand tracks gradients"),
            Self::Shape(why) => write!(f, "shape not served: {why}"),
            Self::Layout(why) => write!(f, "layout not served: {why}"),
            Self::TooLarge(n) => write!(f, "{n} elements exceed {VAE_NSP_MAX_ELEMS}"),
        }
    }
}

/// A served call's geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NspPlan {
    /// The dtype of every operand.
    pub dtype: DType,
    /// `B`.
    pub batch: usize,
    /// `C`.
    pub channels: usize,
    /// `T` of `x`.
    pub frames: usize,
    /// Cache frames.
    pub cache_frames: usize,
    /// Output frames: `zero_front + cache + replicate_front + T + replicate_back + zero_back`.
    pub padded_frames: usize,
    /// `H`.
    pub height: usize,
    /// `W`.
    pub width: usize,
    /// Output elements, `B · C · Tp · H · W`.
    pub out_elems: usize,
    /// Blocks of the launch.
    pub blocks: u32,
}

fn hw_contiguous(dims: &[usize], strides: &[usize]) -> bool {
    (dims[4] == 1 || strides[4] == 1) && (dims[3] == 1 || strides[3] == dims[4])
}

fn checked_elems(dims: &[usize]) -> std::result::Result<usize, NspRefusal> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .filter(|&n| n <= VAE_NSP_MAX_ELEMS)
        .ok_or(NspRefusal::TooLarge(
            dims.iter().fold(1usize, |acc, &d| acc.saturating_mul(d)),
        ))
}

/// The fused kernel's input check (pure; every lane runs it).
pub fn check_norm_silu_pad(req: &NspRequest<'_>) -> std::result::Result<NspPlan, NspRefusal> {
    if !req.on_cuda {
        return Err(NspRefusal::NotCuda);
    }
    let dtype = req.x.dtype;
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(NspRefusal::Dtype(dtype));
    }
    for operand in req.gamma.iter().chain(req.cache.iter()) {
        if operand.dtype != dtype {
            return Err(NspRefusal::Dtype(operand.dtype));
        }
    }
    if req.tracks_grad {
        return Err(NspRefusal::Autograd);
    }
    let &[batch, channels, frames, height, width] = req.x.dims else {
        return Err(NspRefusal::Shape("x must be rank 5 [B, C, T, H, W]"));
    };
    if batch == 0 || channels == 0 || frames == 0 || height == 0 || width == 0 {
        return Err(NspRefusal::Shape("x is empty"));
    }
    if !hw_contiguous(req.x.dims, req.x.strides) {
        return Err(NspRefusal::Layout("x's H x W plane is not contiguous"));
    }
    match (req.needs_gamma, req.gamma) {
        (true, None) => return Err(NspRefusal::Shape("the channel-L2 norm needs gamma")),
        (true, Some(g)) => {
            if g.dims.iter().product::<usize>() != channels || g.dims.is_empty() {
                return Err(NspRefusal::Shape("gamma must hold C elements"));
            }
            let last = g.dims.len() - 1;
            if channels > 1 && g.strides[last] != 1 {
                return Err(NspRefusal::Layout("gamma is not contiguous"));
            }
            if g.dims.iter().filter(|&&d| d > 1).count() > 1 {
                return Err(NspRefusal::Shape("gamma must be a [C] vector"));
            }
        }
        (false, _) => {}
    }
    let mut cache_frames = 0;
    if let Some(cache) = req.cache {
        let &[cb, cc, n, ch, cw] = cache.dims else {
            return Err(NspRefusal::Shape(
                "the cache must be rank 5 [B, C, n, H, W]",
            ));
        };
        if (cb, cc, ch, cw) != (batch, channels, height, width) {
            return Err(NspRefusal::Shape("the cache's B, C, H, W must equal x's"));
        }
        if n > 0 && !hw_contiguous(cache.dims, cache.strides) {
            return Err(NspRefusal::Layout(
                "the cache's H x W plane is not contiguous",
            ));
        }
        checked_elems(cache.dims)?;
        cache_frames = n;
    }
    checked_elems(req.x.dims)?;
    let pad = req.pad;
    let padded_frames = [
        pad.zero_front,
        cache_frames,
        pad.replicate_front,
        frames,
        pad.replicate_back,
        pad.zero_back,
    ]
    .iter()
    .try_fold(0usize, |acc, &n| acc.checked_add(n))
    .filter(|&n| n <= i32::MAX as usize)
    .ok_or(NspRefusal::TooLarge(usize::MAX))?;
    let out_elems = checked_elems(&[batch, channels, padded_frames, height, width])?;
    let positions = batch * padded_frames * height * width;
    let blocks = positions
        .div_ceil(VAE_NSP_THREADS as usize)
        .min(VAE_NSP_MAX_BLOCKS as usize) as u32;
    Ok(NspPlan {
        dtype,
        batch,
        channels,
        frames,
        cache_frames,
        padded_frames,
        height,
        width,
        out_elems,
        blocks,
    })
}

/// The op chain the kernel replaces, op for op as the providers run it (candle-gen-wan
/// `ChanNorm::forward` → `.silu()` → `CausalConv3d`'s `cat` / `pad_with_zeros`; candle-gen-ltx
/// `pixel_norm` → `silu` → frame replication): the parity oracle and bench baseline. The providers
/// keep their own chains as the fallback path; this one is held equal to them by construction.
pub fn norm_silu_pad_reference(
    x: &Tensor,
    norm: &VaeNorm<'_>,
    cache: Option<&Tensor>,
    pad: TemporalPad,
    layout: ConvInputLayout,
) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    let normed = match norm {
        VaeNorm::ChannelL2 { gamma, scale, eps } => {
            let dt = x.dtype();
            let l2 = x
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_keepdim(1)?
                .sqrt()?
                .clamp(*eps, 1e30)?
                .to_dtype(dt)?;
            (x.broadcast_div(&l2)? * *scale)?.broadcast_mul(&gamma.reshape((1, c, 1, 1, 1))?)?
        }
        VaeNorm::PixelRms { eps } => {
            let denom = ((x.sqr()?.sum_keepdim(1)? / c as f64)? + *eps)?.sqrt()?;
            x.broadcast_div(&denom)?
        }
    };
    let act = normed.silu()?;
    let zeros = |n: usize| Tensor::zeros((b, c, n, h, w), x.dtype(), x.device());
    let mut parts = Vec::new();
    if pad.zero_front > 0 {
        parts.push(zeros(pad.zero_front)?);
    }
    if let Some(cache) = cache {
        parts.push(cache.clone());
    }
    let first = act.narrow(2, 0, 1)?;
    let last = act.narrow(2, t - 1, 1)?;
    parts.extend(std::iter::repeat_n(first, pad.replicate_front));
    parts.push(act);
    parts.extend(std::iter::repeat_n(last, pad.replicate_back));
    if pad.zero_back > 0 {
        parts.push(zeros(pad.zero_back)?);
    }
    let padded = Tensor::cat(&parts, 2)?;
    match layout {
        ConvInputLayout::ChannelMajor => padded.contiguous(),
        ConvInputLayout::FrameMajor => padded.permute((0, 2, 1, 3, 4))?.contiguous(),
    }
}

/// The fused attempt: `Attempt::Fused(pad(silu(norm(x))))` in `layout`, or
/// `Attempt::Reference(reason)` when the kernel does not serve the call (a typed plan refusal, a
/// build without `cuda`, a cached compile failure). It neither consults the switch nor records
/// the outcome — [`VAE_NORM_SILU_PAD_PATH`]'s `dispatch` does both.
///
/// `x`: `[B, C, T, H, W]`; `cache`: activated frames `[B, C, n, H, W]` (any strides with a
/// contiguous `H × W` plane, e.g. a permuted frame-major buffer).
pub fn norm_silu_pad(
    x: &Tensor,
    norm: &VaeNorm<'_>,
    cache: Option<&Tensor>,
    pad: TemporalPad,
    layout: ConvInputLayout,
) -> Result<Attempt<Tensor>> {
    let plan = match check_norm_silu_pad(&NspRequest::of(x, norm, cache, pad)) {
        Ok(plan) => plan,
        Err(refusal) => return Ok(Attempt::Reference(refusal.reason())),
    };
    #[cfg(feature = "cuda")]
    {
        cuda_impl::launch(x, norm, cache, pad, layout, &plan)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (plan, layout);
        Ok(Attempt::Reference(PathReason::CudaFeatureOff))
    }
}

/// [`norm_silu_pad`] through [`VAE_NORM_SILU_PAD_PATH`] with [`norm_silu_pad_reference`] as the
/// reference (the bench's and tests' entry point; providers pass their own chain to `dispatch`).
pub fn norm_silu_pad_or_reference(
    x: &Tensor,
    norm: &VaeNorm<'_>,
    cache: Option<&Tensor>,
    pad: TemporalPad,
    layout: ConvInputLayout,
) -> Result<Tensor> {
    VAE_NORM_SILU_PAD_PATH.dispatch(
        || norm_silu_pad(x, norm, cache, pad, layout),
        || norm_silu_pad_reference(x, norm, cache, pad, layout),
    )
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::nvrtc::KernelCompileError;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaFunction, CudaSlice, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::CudaDType;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Storage};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    /// One device's three entry points, resolved once.
    struct Functions {
        f32: CudaFunction,
        bf16: CudaFunction,
        f16: CudaFunction,
    }

    impl Functions {
        fn resolve(dev: &CudaDevice) -> std::result::Result<Self, KernelCompileError> {
            let m = VAE_NORM_SILU_PAD_SRC.compiled(dev)?;
            Ok(Self {
                f32: m.function("vae_norm_silu_pad_f32")?,
                bf16: m.function("vae_norm_silu_pad_bf16")?,
                f16: m.function("vae_norm_silu_pad_f16")?,
            })
        }
    }

    /// Per-ordinal function tables; only successes are held (a compile failure stays cached in
    /// the nvrtc seam and is re-read from there).
    fn functions(dev: &CudaDevice) -> std::result::Result<&'static Functions, KernelCompileError> {
        static TABLES: OnceLock<RwLock<HashMap<usize, &'static Functions>>> = OnceLock::new();
        let tables = TABLES.get_or_init(Default::default);
        let ordinal = dev.cuda_stream().context().ordinal();
        if let Some(t) = tables
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&ordinal)
        {
            return Ok(t);
        }
        let resolved = Functions::resolve(dev)?;
        let mut tables = tables.write().unwrap_or_else(|p| p.into_inner());
        Ok(*tables
            .entry(ordinal)
            .or_insert_with(|| Box::leak(Box::new(resolved))))
    }

    fn drv(e: DriverError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("fused vae norm+silu+pad: {e:?}").into())
    }

    /// `(start offset, batch, channel, time)` element strides of a rank-5 `[B, C, T, H, W]` view.
    fn view_strides(t: &Tensor) -> (i64, i64, i64, i64) {
        let s = t.layout().stride();
        (
            t.layout().start_offset() as i64,
            s[0] as i64,
            s[1] as i64,
            s[2] as i64,
        )
    }

    fn slice<T: CudaDType>(storage: &Storage) -> Result<&CudaSlice<T>> {
        match storage {
            Storage::Cuda(cs) => cs.as_cuda_slice::<T>(),
            _ => candle_core::bail!("fused vae norm+silu+pad: operand is not on CUDA"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_typed<T: CudaDType + DeviceRepr>(
        f: &CudaFunction,
        dev: &CudaDevice,
        x: &Tensor,
        norm: &VaeNorm<'_>,
        cache: Option<&Tensor>,
        pad: TemporalPad,
        layout: ConvInputLayout,
        plan: &NspPlan,
    ) -> Result<Tensor> {
        let (xs, _) = x.storage_and_layout();
        let x_slice = slice::<T>(&xs)?;
        let gamma = norm.gamma();
        let gs = gamma.map(|g| g.storage_and_layout());
        let g_slice = match &gs {
            Some((s, _)) => slice::<T>(s)?,
            None => x_slice,
        };
        let g_off = gamma.map_or(0i64, |g| g.layout().start_offset() as i64);
        let cache = cache.filter(|_| plan.cache_frames > 0);
        let cs = cache.map(|c| c.storage_and_layout());
        let c_slice = match &cs {
            Some((s, _)) => slice::<T>(s)?,
            None => x_slice,
        };
        let (x_off, xsb, xsc, xst) = view_strides(x);
        let (c_off, csb, csc, cst) = cache.map_or((0, 0, 0, 0), view_strides);
        let hw = (plan.height * plan.width) as i64;
        let (c, tp) = (plan.channels as i64, plan.padded_frames as i64);
        let (osb, osc, ost) = match layout {
            ConvInputLayout::ChannelMajor => (c * tp * hw, tp * hw, hw),
            ConvInputLayout::FrameMajor => (tp * c * hw, hw, c * hw),
        };
        let (norm_kind, scale, eps) = match norm {
            VaeNorm::ChannelL2 { scale, eps, .. } => (0i32, *scale as f32, *eps as f32),
            VaeNorm::PixelRms { eps } => (1i32, 1.0f32, *eps as f32),
        };
        let inv_c = (1.0 / plan.channels as f64) as f32;
        let (batch, channels, frames, padded_frames) = (
            plan.batch as i32,
            plan.channels as i32,
            plan.frames as i32,
            plan.padded_frames as i32,
        );
        let n_cache = plan.cache_frames as i32;
        let (zero_front, rep_front, zero_back) = (
            pad.zero_front as i32,
            pad.replicate_front as i32,
            pad.zero_back as i32,
        );
        // SAFETY: every one of the `out_elems` elements is written by the launch below (each
        // output position writes all C channels of its frame, zero frames included).
        let mut out = unsafe { dev.alloc::<T>(plan.out_elems) }?;
        let stream = dev.cuda_stream();
        let mut b = stream.launch_builder(f);
        b.arg(x_slice).arg(&x_off).arg(&xsb).arg(&xsc).arg(&xst);
        b.arg(g_slice).arg(&g_off);
        b.arg(c_slice)
            .arg(&c_off)
            .arg(&csb)
            .arg(&csc)
            .arg(&cst)
            .arg(&n_cache);
        b.arg(&mut out).arg(&osb).arg(&osc).arg(&ost);
        b.arg(&batch)
            .arg(&channels)
            .arg(&frames)
            .arg(&padded_frames)
            .arg(&hw);
        b.arg(&zero_front).arg(&rep_front).arg(&zero_back);
        b.arg(&norm_kind).arg(&scale).arg(&eps).arg(&inv_c);
        let cfg = LaunchConfig {
            grid_dim: (plan.blocks, 1, 1),
            block_dim: (VAE_NSP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match `NSP_ENTRY` in the source; every index the
        // kernel forms is inside an operand the plan check bounded (64-bit math).
        unsafe { b.launch(cfg) }.map_err(drv)?;
        drop((xs, gs, cs));
        let shape = match layout {
            ConvInputLayout::ChannelMajor => (
                plan.batch,
                plan.channels,
                plan.padded_frames,
                plan.height,
                plan.width,
            ),
            ConvInputLayout::FrameMajor => (
                plan.batch,
                plan.padded_frames,
                plan.channels,
                plan.height,
                plan.width,
            ),
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    pub(super) fn launch(
        x: &Tensor,
        norm: &VaeNorm<'_>,
        cache: Option<&Tensor>,
        pad: TemporalPad,
        layout: ConvInputLayout,
        plan: &NspPlan,
    ) -> Result<Attempt<Tensor>> {
        let Device::Cuda(dev) = x.device() else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let f = match functions(dev) {
            Ok(f) => f,
            Err(e) => return Ok(Attempt::Reference(e.into())),
        };
        let out = match plan.dtype {
            DType::F32 => launch_typed::<f32>(&f.f32, dev, x, norm, cache, pad, layout, plan)?,
            DType::BF16 => {
                launch_typed::<half::bf16>(&f.bf16, dev, x, norm, cache, pad, layout, plan)?
            }
            DType::F16 => {
                launch_typed::<half::f16>(&f.f16, dev, x, norm, cache, pad, layout, plan)?
            }
            other => return Ok(Attempt::Reference(PathReason::Dtype(other))),
        };
        Ok(Attempt::Fused(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const CONTIG: [usize; 5] = [2 * 3 * 4 * 5 * 6 / 2, 4 * 5 * 6, 5 * 6, 6, 1];

    fn x_op(dims: &[usize; 5]) -> NspOperand<'_> {
        NspOperand {
            dtype: DType::BF16,
            dims,
            strides: &CONTIG,
        }
    }

    fn request<'a>(x: NspOperand<'a>, gamma: Option<NspOperand<'a>>) -> NspRequest<'a> {
        NspRequest {
            on_cuda: true,
            tracks_grad: false,
            x,
            gamma,
            needs_gamma: true,
            cache: None,
            pad: TemporalPad {
                zero_front: 2,
                ..Default::default()
            },
        }
    }

    #[test]
    fn the_plan_refuses_what_the_kernel_does_not_serve() {
        let dims = [1usize, 3, 4, 5, 6];
        let gdims = [3usize];
        let gamma = NspOperand {
            dtype: DType::BF16,
            dims: &gdims,
            strides: &[1],
        };
        let ok = request(x_op(&dims), Some(gamma));
        let plan = check_norm_silu_pad(&ok).expect("served");
        assert_eq!(plan.padded_frames, 6);
        assert_eq!(plan.out_elems, 3 * 6 * 5 * 6);

        let mut r = ok;
        r.on_cuda = false;
        assert_eq!(check_norm_silu_pad(&r), Err(NspRefusal::NotCuda));
        let mut r = ok;
        r.x.dtype = DType::U8;
        assert_eq!(check_norm_silu_pad(&r), Err(NspRefusal::Dtype(DType::U8)));
        let mut r = ok;
        r.gamma = Some(NspOperand {
            dtype: DType::F32,
            ..gamma
        });
        assert_eq!(check_norm_silu_pad(&r), Err(NspRefusal::Dtype(DType::F32)));
        let mut r = ok;
        r.tracks_grad = true;
        assert_eq!(check_norm_silu_pad(&r), Err(NspRefusal::Autograd));
        let rank4 = [1usize, 3, 5, 6];
        let r = request(
            NspOperand {
                dtype: DType::BF16,
                dims: &rank4,
                strides: &[90, 30, 6, 1],
            },
            Some(gamma),
        );
        assert!(matches!(check_norm_silu_pad(&r), Err(NspRefusal::Shape(_))));
        let empty = [1usize, 3, 0, 5, 6];
        let r = request(x_op(&empty), Some(gamma));
        assert!(matches!(check_norm_silu_pad(&r), Err(NspRefusal::Shape(_))));
        // A transposed H x W plane is not served.
        let r = request(
            NspOperand {
                dtype: DType::BF16,
                dims: &dims,
                strides: &[120, 40, 1, 4, 20],
            },
            Some(gamma),
        );
        assert!(matches!(
            check_norm_silu_pad(&r),
            Err(NspRefusal::Layout(_))
        ));
        let mut r = ok;
        r.gamma = None;
        assert!(matches!(check_norm_silu_pad(&r), Err(NspRefusal::Shape(_))));
        let short = [2usize];
        let r = request(
            x_op(&dims),
            Some(NspOperand {
                dims: &short,
                ..gamma
            }),
        );
        assert!(matches!(check_norm_silu_pad(&r), Err(NspRefusal::Shape(_))));
        let strided = request(
            x_op(&dims),
            Some(NspOperand {
                strides: &[2],
                ..gamma
            }),
        );
        assert!(matches!(
            check_norm_silu_pad(&strided),
            Err(NspRefusal::Layout(_))
        ));
        // The cache must match x's B, C, H, W.
        let cdims = [1usize, 4, 2, 5, 6];
        let mut r = ok;
        r.cache = Some(x_op(&cdims));
        assert!(matches!(check_norm_silu_pad(&r), Err(NspRefusal::Shape(_))));
        let cdims = [1usize, 3, 2, 5, 6];
        let mut r = ok;
        r.cache = Some(x_op(&cdims));
        r.pad.zero_front = 0;
        let plan = check_norm_silu_pad(&r).expect("cache served");
        assert_eq!((plan.cache_frames, plan.padded_frames), (2, 6));
        // E6: an operand past i32::MAX elements is refused.
        let huge = [1usize, 1024, 4, 1024, 1024];
        let mut r = request(
            NspOperand {
                dtype: DType::BF16,
                dims: &huge,
                strides: &[0, 0, 1 << 20, 1024, 1],
            },
            None,
        );
        r.needs_gamma = false;
        assert!(matches!(
            check_norm_silu_pad(&r),
            Err(NspRefusal::TooLarge(_))
        ));
    }

    #[test]
    fn refusals_map_to_typed_path_reasons() {
        assert_eq!(NspRefusal::NotCuda.reason().label(), "not_cuda");
        assert_eq!(NspRefusal::Dtype(DType::U8).reason().label(), "dtype");
        assert_eq!(NspRefusal::Autograd.reason().label(), "autograd");
        assert_eq!(NspRefusal::Shape("s").reason().label(), "shape");
        assert_eq!(NspRefusal::Layout("l").reason().label(), "layout");
        assert_eq!(NspRefusal::TooLarge(1).reason().label(), "index_overflow");
    }

    #[test]
    fn the_grid_covers_every_position_within_the_cap() {
        for (dims, pad, blocks) in [
            ([1usize, 3, 1, 1, 1], TemporalPad::default(), 1u32),
            (
                [1, 96, 4, 90, 160],
                TemporalPad {
                    zero_front: 2,
                    ..Default::default()
                },
                (6 * 90 * 160usize).div_ceil(256) as u32,
            ),
            (
                [1, 96, 4, 720, 1280],
                TemporalPad {
                    replicate_front: 1,
                    replicate_back: 1,
                    ..Default::default()
                },
                (6 * 720 * 1280usize).div_ceil(256) as u32,
            ),
            (
                [1, 1, 100, 2048, 2048],
                TemporalPad::default(),
                VAE_NSP_MAX_BLOCKS,
            ),
        ] {
            let strides = [0usize, 0, dims[3] * dims[4], dims[4], 1];
            let mut r = request(
                NspOperand {
                    dtype: DType::F32,
                    dims: &dims,
                    strides: &strides,
                },
                None,
            );
            r.needs_gamma = false;
            r.pad = pad;
            let plan = check_norm_silu_pad(&r).expect("served");
            assert_eq!(plan.blocks, blocks, "{dims:?}");
        }
    }

    /// The reference chain lays the time axis out as `[zeros][cache][first..][act][..last][zeros]`
    /// (pure tensor ops on the CPU; the kernel's GPU tests compare against this chain).
    #[test]
    fn the_reference_chain_lays_out_the_padded_time_axis() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::arange(0f32, 2. * 3. * 2. * 2., &dev)?.reshape((1, 2, 3, 2, 2))?;
        let x = (x - 5.0)?;
        let cache = Tensor::full(7f32, (1, 2, 1, 2, 2), &dev)?;
        let pad = TemporalPad {
            zero_front: 1,
            replicate_front: 1,
            replicate_back: 2,
            zero_back: 1,
        };
        let norm = VaeNorm::PixelRms { eps: 1e-8 };
        let cm =
            norm_silu_pad_reference(&x, &norm, Some(&cache), pad, ConvInputLayout::ChannelMajor)?;
        assert_eq!(cm.dims(), &[1, 2, 9, 2, 2]);
        let act = {
            let denom = ((x.sqr()?.sum_keepdim(1)? / 2.0)? + 1e-8)?.sqrt()?;
            x.broadcast_div(&denom)?.silu()?
        };
        let frame = |t: &Tensor, i: usize| -> Result<Vec<f32>> {
            t.narrow(2, i, 1)?.flatten_all()?.to_vec1::<f32>()
        };
        let zero = vec![0f32; 8];
        assert_eq!(frame(&cm, 0)?, zero);
        assert_eq!(frame(&cm, 1)?, vec![7f32; 8]);
        assert_eq!(frame(&cm, 2)?, frame(&act, 0)?);
        for t in 0..3 {
            assert_eq!(frame(&cm, 3 + t)?, frame(&act, t)?);
        }
        assert_eq!(frame(&cm, 6)?, frame(&act, 2)?);
        assert_eq!(frame(&cm, 7)?, frame(&act, 2)?);
        assert_eq!(frame(&cm, 8)?, zero);
        let fm =
            norm_silu_pad_reference(&x, &norm, Some(&cache), pad, ConvInputLayout::FrameMajor)?;
        assert_eq!(fm.dims(), &[1, 9, 2, 2, 2]);
        let back = fm.permute((0, 2, 1, 3, 4))?.contiguous()?;
        assert_eq!(
            back.flatten_all()?.to_vec1::<f32>()?,
            cm.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    /// Without a CUDA tensor the attempt declines with a typed reason (never an error).
    #[test]
    fn a_cpu_tensor_declines_with_not_cuda() -> Result<()> {
        let x = Tensor::ones((1, 2, 1, 2, 2), DType::F32, &Device::Cpu)?;
        let a = norm_silu_pad(
            &x,
            &VaeNorm::PixelRms { eps: 1e-8 },
            None,
            TemporalPad::default(),
            ConvInputLayout::FrameMajor,
        )?;
        assert!(matches!(a, Attempt::Reference(PathReason::NotCuda)));
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    //! GPU parity of the fused kernel against [`norm_silu_pad_reference`] (the providers' chain).
    //! `#[ignore]`d CUDA compute (registered in `scripts/ci/test_suites.json`); run them on a CUDA
    //! box with `cargo test -p candle-quant-kernels --features cuda --lib vae_norm_silu_pad --
    //! --ignored`.
    //!
    //! Declared tolerances (epic E2):
    //! - **f32**: max relative error <= 1e-5 against the f32 chain — and bit-identical, since the
    //!   kernel reproduces candle's `fast_sum` order and every op without contraction.
    //! - **bf16 / f16**: the kernel computes in f32 and rounds once, so each element is held within
    //!   half an output ULP of the f32 oracle (the chain on the upcast inputs): `|fused - oracle| <=
    //!   (2^-8 + 1e-5)·|oracle|` for bf16 (8 significant bits), `(2^-11 + 1e-5)·|oracle| + 2^-25`
    //!   for f16 (11 bits; the absolute term covers f16 subnormals). The 16-bit op chain rounds after
    //!   every op and is looser; the fused rel-L2 error against the oracle must not exceed it.
    //! - zero frames are exactly zero and cache frames are bit-identical copies.
    use super::*;
    use crate::fused_path::media_fused_policy_guard;
    use candle_core::{Device, Var};

    fn cuda() -> Device {
        Device::new_cuda(0).expect("CUDA device 0")
    }

    /// Deterministic activations with a heavy tail: mostly |v| < 3, every 53rd up to ~60.
    fn activations(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                let v = ((u * 2.0 - 1.0) * 3.0) as f32;
                if i % 53 == 0 {
                    v * 20.0
                } else {
                    v
                }
            })
            .collect()
    }

    fn tensor(dims: &[usize], seed: u64, dev: &Device) -> Result<Tensor> {
        let n = dims.iter().product();
        Tensor::from_vec(activations(n, seed), dims, dev)
    }

    fn host(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }

    fn fused(a: Attempt<Tensor>) -> Tensor {
        match a {
            Attempt::Fused(t) => t,
            Attempt::Reference(r) => panic!("the fused kernel declined: {r}"),
        }
    }

    /// Max relative error (exact zeros on both sides count as equal).
    fn max_rel(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                if x == y {
                    0.0
                } else {
                    ((x - y).abs() as f64) / (y.abs() as f64).max(1e-30)
                }
            })
            .fold(0.0, f64::max)
    }

    fn rel_l2(a: &[f32], oracle: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (&x, &o) in a.iter().zip(oracle) {
            num += ((x - o) as f64).powi(2);
            den += (o as f64).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// Half an output ULP of one rounding, per element.
    fn one_rounding_bound(dtype: DType, oracle: f32) -> f64 {
        let o = oracle.abs() as f64;
        match dtype {
            DType::BF16 => (2f64.powi(-8) + 1e-5) * o + 1e-38,
            DType::F16 => (2f64.powi(-11) + 1e-5) * o + 2f64.powi(-25),
            _ => unreachable!(),
        }
    }

    struct Case {
        name: &'static str,
        dims: [usize; 5],
        /// `x` is a view narrowed out of a wider time axis (non-contiguous strides, an offset).
        strided_x: bool,
        /// Cache frames (0 = none); `fm_cache` stores them frame-major, viewed channel-major.
        cache: usize,
        fm_cache: bool,
        pad: TemporalPad,
        ltx: bool,
    }

    fn cases() -> Vec<Case> {
        let zf = |n| TemporalPad {
            zero_front: n,
            ..Default::default()
        };
        vec![
            Case {
                name: "wan first chunk (zero front pad, no cache)",
                dims: [1, 96, 1, 12, 20],
                strided_x: false,
                cache: 0,
                fm_cache: false,
                pad: zf(2),
                ltx: false,
            },
            Case {
                name: "wan warming chunk (1 cached frame + 1 zero frame)",
                dims: [1, 96, 2, 12, 20],
                strided_x: false,
                cache: 1,
                fm_cache: false,
                pad: zf(1),
                ltx: false,
            },
            Case {
                name: "wan cached chunk (2 cached frames)",
                dims: [1, 384, 4, 9, 16],
                strided_x: false,
                cache: 2,
                fm_cache: true,
                pad: zf(0),
                ltx: false,
            },
            Case {
                name: "wan single pass, strided x, odd extents",
                dims: [2, 3, 5, 7, 5],
                strided_x: true,
                cache: 0,
                fm_cache: false,
                pad: zf(2),
                ltx: false,
            },
            Case {
                name: "wan cached, batch 2, strided x",
                dims: [2, 17, 3, 6, 11],
                strided_x: true,
                cache: 2,
                fm_cache: true,
                pad: zf(0),
                ltx: false,
            },
            Case {
                name: "z48 mid width (C = 1024, the widest tree)",
                dims: [1, 1024, 2, 4, 6],
                strided_x: false,
                cache: 2,
                fm_cache: false,
                pad: zf(0),
                ltx: false,
            },
            Case {
                name: "C = 1500 (fast_sum leaves of two squares)",
                dims: [1, 1500, 1, 3, 5],
                strided_x: false,
                cache: 0,
                fm_cache: false,
                pad: zf(2),
                ltx: false,
            },
            Case {
                name: "ltx decoder (replicate first + last)",
                dims: [1, 128, 5, 8, 12],
                strided_x: false,
                cache: 0,
                fm_cache: false,
                pad: TemporalPad {
                    replicate_front: 1,
                    replicate_back: 1,
                    ..Default::default()
                },
                ltx: true,
            },
            Case {
                name: "ltx encoder (causal replicate front)",
                dims: [1, 48, 3, 10, 6],
                strided_x: true,
                cache: 0,
                fm_cache: false,
                pad: TemporalPad {
                    replicate_front: 2,
                    ..Default::default()
                },
                ltx: true,
            },
        ]
    }

    struct Inputs {
        x: Tensor,
        gamma: Tensor,
        cache: Option<Tensor>,
    }

    fn inputs(case: &Case, seed: u64, dev: &Device) -> Result<Inputs> {
        let [b, c, t, h, w] = case.dims;
        let x = if case.strided_x {
            tensor(&[b, c, t + 3, h, w], seed, dev)?.narrow(2, 2, t)?
        } else {
            tensor(&case.dims, seed, dev)?
        };
        // Wan's gamma is a trained gain around 1 (occasionally negative-leaning).
        let gamma = (tensor(&[c], seed ^ 0x5eed, dev)? * 0.2)?.affine(1.0, 1.0)?;
        let cache = match case.cache {
            0 => None,
            n if case.fm_cache => {
                Some(tensor(&[b, n, c, h, w], seed ^ 0xcace, dev)?.permute((0, 2, 1, 3, 4))?)
            }
            n => Some(tensor(&[b, c, n, h, w], seed ^ 0xcace, dev)?),
        };
        Ok(Inputs { x, gamma, cache })
    }

    fn norm_of<'a>(case: &Case, gamma: &'a Tensor, c: usize) -> VaeNorm<'a> {
        if case.ltx {
            VaeNorm::PixelRms { eps: 1e-8 }
        } else {
            VaeNorm::ChannelL2 {
                gamma,
                scale: (c as f64).sqrt(),
                eps: 1e-12,
            }
        }
    }

    fn cast(i: &Inputs, dtype: DType) -> Result<Inputs> {
        Ok(Inputs {
            x: i.x.to_dtype(dtype)?,
            gamma: i.gamma.to_dtype(dtype)?,
            cache: i.cache.as_ref().map(|c| c.to_dtype(dtype)).transpose()?,
        })
    }

    /// The item's parity test: `pad(silu(norm(x)))` over every case, dtype and layout.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_norm_silu_pad_gpu() -> Result<()> {
        let dev = cuda();
        for (k, case) in cases().iter().enumerate() {
            let base = inputs(case, 11 + k as u64, &dev)?;
            let c = case.dims[1];
            for layout in [ConvInputLayout::FrameMajor, ConvInputLayout::ChannelMajor] {
                for dtype in [DType::F32, DType::BF16, DType::F16] {
                    let label = format!("{} {dtype:?} {layout:?}", case.name);
                    // Rounding the f32 inputs to the dtype first makes the oracle (the f32 chain on
                    // the upcast operands) exact for what the 16-bit kernel actually reads.
                    let typed = cast(&base, dtype)?;
                    let up = cast(&typed, DType::F32)?;
                    let norm = norm_of(case, &typed.gamma, c);
                    let got = fused(norm_silu_pad(
                        &typed.x,
                        &norm,
                        typed.cache.as_ref(),
                        case.pad,
                        layout,
                    )?);
                    let reference = norm_silu_pad_reference(
                        &typed.x,
                        &norm,
                        typed.cache.as_ref(),
                        case.pad,
                        layout,
                    )?;
                    assert_eq!(got.dims(), reference.dims(), "{label}");
                    assert_eq!(got.dtype(), dtype, "{label}");
                    let oracle = norm_silu_pad_reference(
                        &up.x,
                        &norm_of(case, &up.gamma, c),
                        up.cache.as_ref(),
                        case.pad,
                        layout,
                    )?;
                    let (g, r, o) = (host(&got)?, host(&reference)?, host(&oracle)?);
                    // Zero and cache frames are exact in every dtype.
                    let cm = |t: &Tensor| -> Result<Tensor> {
                        match layout {
                            ConvInputLayout::ChannelMajor => Ok(t.clone()),
                            ConvInputLayout::FrameMajor => t.permute((0, 2, 1, 3, 4)),
                        }
                    };
                    let (gcm, rcm) = (cm(&got)?, cm(&reference)?);
                    let copies = case.pad.zero_front + case.cache;
                    if copies > 0 {
                        assert_eq!(
                            host(&gcm.narrow(2, 0, copies)?)?,
                            host(&rcm.narrow(2, 0, copies)?)?,
                            "{label}: zero / cache frames must be exact"
                        );
                    }
                    if dtype == DType::F32 {
                        let rel = max_rel(&g, &r);
                        let differing = g.iter().zip(&r).filter(|(a, b)| a != b).count();
                        println!(
                            "{label}: max rel err {rel:e}, {differing} of {} values differ",
                            g.len()
                        );
                        assert!(rel <= 1e-5, "{label}: f32 max rel err {rel:e} > 1e-5");
                        assert_eq!(
                            differing, 0,
                            "{label}: the f32 kernel must equal the chain bit for bit"
                        );
                    } else {
                        for (i, (&gv, &ov)) in g.iter().zip(&o).enumerate() {
                            let err = (gv - ov).abs() as f64;
                            assert!(
                                err <= one_rounding_bound(dtype, ov),
                                "{label}: element {i}: fused {gv} vs f32 oracle {ov} (err {err:e})"
                            );
                        }
                        let (fe, re) = (rel_l2(&g, &o), rel_l2(&r, &o));
                        assert!(
                            fe <= re * 1.0001 + 1e-7,
                            "{label}: fused rel-L2 {fe:e} worse than the {dtype:?} chain's {re:e}"
                        );
                        println!("{label}: fused rel-L2 {fe:.3e}, chain rel-L2 {re:.3e}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Mutation check: writing the causal zero pad on the wrong temporal side (after the frames
    /// instead of before) or dropping the cache must fail the parity criterion.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn wrong_side_pad_and_dropped_cache_fail_parity() -> Result<()> {
        let dev = cuda();
        let case = &cases()[1]; // 1 cached frame + 1 zero frame
        let i = inputs(case, 7, &dev)?;
        let norm = norm_of(case, &i.gamma, case.dims[1]);
        let layout = ConvInputLayout::FrameMajor;
        let reference = host(&norm_silu_pad_reference(
            &i.x,
            &norm,
            i.cache.as_ref(),
            case.pad,
            layout,
        )?)?;
        let wrong_side = TemporalPad {
            zero_front: 0,
            zero_back: case.pad.zero_front,
            ..case.pad
        };
        let mutated = host(&fused(norm_silu_pad(
            &i.x,
            &norm,
            i.cache.as_ref(),
            wrong_side,
            layout,
        )?))?;
        let rel = max_rel(&mutated, &reference);
        assert!(
            rel > 1e-2,
            "a wrong-side pad passed parity (max rel {rel:e})"
        );
        // Same frame count without the cache: zeros where the history belongs.
        let no_cache = TemporalPad {
            zero_front: case.pad.zero_front + case.cache,
            ..case.pad
        };
        let mutated = host(&fused(norm_silu_pad(&i.x, &norm, None, no_cache, layout)?))?;
        let rel = max_rel(&mutated, &reference);
        assert!(
            rel > 1e-2,
            "a dropped cache passed parity (max rel {rel:e})"
        );
        Ok(())
    }

    /// The dispatching entry records its path: fused on CUDA, `disabled` with the switch off
    /// (same values), `not_cuda` for a CPU tensor and `autograd` for a variable.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn the_dispatching_entry_reports_its_path() -> Result<()> {
        let dev = cuda();
        let case = &cases()[0];
        let i = inputs(case, 3, &dev)?;
        let norm = norm_of(case, &i.gamma, case.dims[1]);
        let layout = ConvInputLayout::FrameMajor;
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = VAE_NORM_SILU_PAD_PATH.guard(Some(true));
        let path = &VAE_NORM_SILU_PAD_PATH;

        let start = path.thread_counts();
        let on = norm_silu_pad_or_reference(&i.x, &norm, None, case.pad, layout)?;
        let d = path.thread_counts().since(&start);
        assert_eq!((d.fused, d.reference_total()), (1, 0));

        path.set_enabled(Some(false));
        let mid = path.thread_counts();
        let off = norm_silu_pad_or_reference(&i.x, &norm, None, case.pad, layout)?;
        let d = path.thread_counts().since(&mid);
        assert_eq!((d.fused, d.reference_for("disabled")), (0, 1));
        assert!(max_rel(&host(&on)?, &host(&off)?) <= 1e-5);
        path.set_enabled(Some(true));

        let (xc, gc) = (
            i.x.to_device(&Device::Cpu)?,
            i.gamma.to_device(&Device::Cpu)?,
        );
        let cpu_norm = norm_of(case, &gc, case.dims[1]);
        let before = path.thread_counts();
        norm_silu_pad_or_reference(&xc, &cpu_norm, None, case.pad, layout)?;
        assert_eq!(
            path.thread_counts()
                .since(&before)
                .reference_for("not_cuda"),
            1
        );

        let var = Var::from_tensor(&i.x)?;
        let before = path.thread_counts();
        norm_silu_pad_or_reference(var.as_tensor(), &norm, None, case.pad, layout)?;
        assert_eq!(
            path.thread_counts()
                .since(&before)
                .reference_for("autograd"),
            1
        );
        Ok(())
    }
}
