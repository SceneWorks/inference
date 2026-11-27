//! The fused **AdaLN modulation** (SW-6127, feature SW-F287 S2): `norm(x) · (1 + scale) + shift`
//! (or `· scale`) as one launch over [`ADALN_SRC`] (nvrtc, compiled once per device), for RMSNorm
//! and LayerNorm, with an optional learned norm weight (and LayerNorm bias), the modulation
//! broadcast per sample (`[B, 1, D]`) or per token (`[B, S, D]`).
//!
//! The public home of this API is `candle_gen::modulate` (re-exported there for the DiT
//! providers); it lives here, beside the nvrtc seam, because the kernel, its reference and its
//! parity tests belong together.
//!
//! # The reference it must equal bit for bit
//!
//! [`adaln_reference`] is today's op chain, verbatim: candle's `ops::rms_norm` /
//! `ops::layer_norm` (in `x`'s dtype, or on an f32 copy cast back when the spec says
//! [`AdaLn::norm_in_f32`](field@AdaLn::norm_in_f32)), then `broadcast_mul(scale + 1)` and `broadcast_add(shift)` in `x`'s
//! dtype. It is the parity oracle and the path for everything the kernel does not serve (epic E1):
//! a CPU / Metal tensor, a build without `cuda`, an unserved dtype, shape or layout, a tensor above
//! `i32::MAX` elements, a gradient-tracked operand (E9: training keeps the composable ops), either
//! switch off, or a cached compile failure.
//!
//! The kernel reproduces each reference step with the reference's own operation (the contract at
//! the top of `adaln.cu`): candle's rmsnorm / layernorm launch shape and reduction order, its
//! `rsqrtf`, one rounding to `x`'s dtype per stored tensor of the chain. The GPU tests in
//! `cuda_tests` hold it to **bit identity** for f32, bf16 and f16 (the declared tolerance is zero
//! ulp, which also satisfies the epic's f32 `<= 1e-5` relative bound).
//!
//! # Path selection (epic E1 / E5)
//!
//! Every call goes through [`ADALN_PATH`] (the media fused-path seam, [`crate::fused_path`]):
//! `CANDLE_FUSED_ADALN` (and `CANDLE_MEDIA_FUSED_KERNELS`) switch it, and each call records fused
//! or reference with a typed reason. [`check_adaln`] is the pure plan both the launch and the CPU
//! tests use. The kernel's compute-capability floor is sm_70 (f32 arithmetic, `cvt` and warp
//! shuffles only).

use std::fmt;

use candle_core::{DType, Result, Tensor, D};

use crate::fused_path::{Attempt, FusedKernelPath, PathReason};
use crate::nvrtc::KernelSource;

/// The fused AdaLN source; compiled once per device on first use.
pub const ADALN_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_adaln_v1",
    src: include_str!("adaln.cu"),
    // f32 arithmetic, `cvt` conversions (`cvt.rn.bf16.f32` only behind `__CUDA_ARCH__ >= 800`)
    // and warp shuffles: sm_70 is the oldest this workspace targets.
    cc_floor: (7, 0),
};

/// The fused AdaLN's path record (every [`adaln`] call).
pub static ADALN_PATH: FusedKernelPath =
    FusedKernelPath::new("adaln_modulate", "CANDLE_FUSED_ADALN");

/// Largest `x` the fused AdaLN launches (epic E6); larger ones take the reference.
pub const ADALN_MAX_ELEMS: usize = i32::MAX as usize;

/// candle's rmsnorm / layernorm block size for a row of `n` columns — the launch shape the kernel
/// reproduces so its f32 summation order matches the reference.
pub fn adaln_block_dim(n: usize) -> u32 {
    if n < 1024 {
        32
    } else {
        1024
    }
}

/// Which normalization precedes the modulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaNorm {
    /// `x / sqrt(mean(x²) + eps) · weight` — candle's `ops::rms_norm`.
    Rms,
    /// `(x - mean) / sqrt(var + eps) · weight + bias` — candle's `ops::layer_norm`.
    Layer,
}

/// How the modulation scale enters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleForm {
    /// `norm · (1 + scale) + shift` — the DiT AdaLN / AdaLN-Zero convention.
    OnePlus,
    /// `norm · scale + shift` — for a caller whose scale already carries the `1`.
    Direct,
}

/// One AdaLN call's normalization and modulation convention. Build it with [`AdaLn::rms`] /
/// [`AdaLn::layer`] and the `with_*` / [`AdaLn::norm_in_f32`](method@AdaLn::norm_in_f32) / [`AdaLn::direct_scale`]
/// modifiers.
#[derive(Clone, Copy, Debug)]
pub struct AdaLn<'a> {
    /// RMSNorm or LayerNorm.
    pub norm: AdaNorm,
    /// The learned per-column norm weight (`[D]`), or none (an all-ones weight).
    pub weight: Option<&'a Tensor>,
    /// The learned per-column LayerNorm bias (`[D]`), or none. Never set for RMSNorm.
    pub bias: Option<&'a Tensor>,
    /// The norm's epsilon (candle's norm ops take it as `f32`).
    pub eps: f32,
    /// Normalize an f32 copy of `x` (weight and bias then f32) and cast the result back to `x`'s
    /// dtype before modulating — the `F.rms_norm(x.float(), w).to(dtype)` convention. Otherwise
    /// the norm runs in `x`'s dtype (f32 accumulation inside candle's kernel) with weight and bias
    /// in `x`'s dtype.
    pub norm_in_f32: bool,
    /// `(1 + scale)` or `scale`.
    pub scale_form: ScaleForm,
}

impl<'a> AdaLn<'a> {
    /// RMSNorm, no weight, `(1 + scale)`, norm in `x`'s dtype.
    pub fn rms(eps: f64) -> Self {
        Self {
            norm: AdaNorm::Rms,
            weight: None,
            bias: None,
            eps: eps as f32,
            norm_in_f32: false,
            scale_form: ScaleForm::OnePlus,
        }
    }

    /// LayerNorm, no weight or bias, `(1 + scale)`, norm in `x`'s dtype.
    pub fn layer(eps: f64) -> Self {
        Self {
            norm: AdaNorm::Layer,
            ..Self::rms(eps)
        }
    }

    /// With the learned norm weight `weight` (`[D]`).
    pub fn with_weight(self, weight: &'a Tensor) -> Self {
        Self {
            weight: Some(weight),
            ..self
        }
    }

    /// With the learned LayerNorm bias `bias` (`[D]`); [`adaln`] refuses it on RMSNorm.
    pub fn with_bias(self, bias: &'a Tensor) -> Self {
        Self {
            bias: Some(bias),
            ..self
        }
    }

    /// Normalize in f32 and cast back (see [`AdaLn::norm_in_f32`](field@AdaLn::norm_in_f32)).
    pub fn norm_in_f32(self) -> Self {
        Self {
            norm_in_f32: true,
            ..self
        }
    }

    /// Modulate with `scale` itself rather than `1 + scale`.
    pub fn direct_scale(self) -> Self {
        Self {
            scale_form: ScaleForm::Direct,
            ..self
        }
    }

    fn validate(&self) -> Result<()> {
        if self.norm == AdaNorm::Rms && self.bias.is_some() {
            candle_core::bail!("adaln: a bias is a LayerNorm parameter, not an RMSNorm one");
        }
        Ok(())
    }
}

/// The shared modulation `normed · (1 + scale) + shift` (scale and shift broadcast over `normed`)
/// as candle's composable ops — the body every DiT's private `modulate` copy computes.
pub fn modulate(normed: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    modulate_with(normed, scale, shift, ScaleForm::OnePlus)
}

/// [`modulate`] with an explicit [`ScaleForm`].
pub fn modulate_with(
    normed: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    form: ScaleForm,
) -> Result<Tensor> {
    let scaled = match form {
        ScaleForm::OnePlus => normed.broadcast_mul(&(scale + 1.0)?)?,
        ScaleForm::Direct => normed.broadcast_mul(scale)?,
    };
    scaled.broadcast_add(shift)
}

/// The normalization half of the reference chain: candle's norm op on `x` (or an f32 copy, cast
/// back), with ones / zeros standing in for an absent weight / bias.
pub fn adaln_norm_reference(x: &Tensor, spec: &AdaLn<'_>) -> Result<Tensor> {
    spec.validate()?;
    let dtype = x.dtype();
    let xc = if spec.norm_in_f32 {
        x.to_dtype(DType::F32)?
    } else {
        x.clone()
    }
    .contiguous()?;
    let n = xc.dim(D::Minus1)?;
    let ones = || Tensor::ones(n, xc.dtype(), xc.device());
    let weight = match spec.weight {
        Some(w) => w.clone(),
        None => ones()?,
    };
    let normed = match spec.norm {
        AdaNorm::Rms => candle_nn::ops::rms_norm(&xc, &weight, spec.eps)?,
        AdaNorm::Layer => {
            let bias = match spec.bias {
                Some(b) => b.clone(),
                None => Tensor::zeros(n, xc.dtype(), xc.device())?,
            };
            candle_nn::ops::layer_norm(&xc, &weight, &bias, spec.eps)?
        }
    };
    normed.to_dtype(dtype)
}

/// The reference op chain (epic E1): [`adaln_norm_reference`] then [`modulate_with`] in `x`'s
/// dtype. The fused kernel's parity oracle and the path for every input it does not serve.
pub fn adaln_reference(
    x: &Tensor,
    spec: &AdaLn<'_>,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    modulate_with(
        &adaln_norm_reference(x, spec)?,
        scale,
        shift,
        spec.scale_form,
    )
}

/// `norm(x) · (1 + scale) + shift` (per `spec`): the fused kernel on CUDA when it serves the
/// call, else [`adaln_reference`] — bit-identical either way, and recorded in [`ADALN_PATH`].
///
/// `x` is `[B, S, D]`; `scale` and `shift` are `[B or 1, S or 1, D]` in `x`'s dtype (any batch /
/// sequence strides, contiguous along `D`) for the fused path; anything else the reference
/// broadcasts as candle does.
pub fn adaln(x: &Tensor, spec: &AdaLn<'_>, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    spec.validate()?;
    ADALN_PATH.dispatch(
        || fused_attempt(x, spec, scale, shift),
        || adaln_reference(x, spec, scale, shift),
    )
}

/// [`adaln`] for RMSNorm with an optional weight, `(1 + scale)`, norm in `x`'s dtype.
pub fn rms_adaln(
    x: &Tensor,
    weight: Option<&Tensor>,
    eps: f64,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    let spec = AdaLn {
        weight,
        ..AdaLn::rms(eps)
    };
    adaln(x, &spec, scale, shift)
}

/// [`adaln`] for LayerNorm with an optional weight and bias, `(1 + scale)`, norm in `x`'s dtype.
pub fn layer_adaln(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    let spec = AdaLn {
        weight,
        bias,
        ..AdaLn::layer(eps)
    };
    adaln(x, &spec, scale, shift)
}

#[cfg(feature = "cuda")]
fn fused_attempt(
    x: &Tensor,
    spec: &AdaLn<'_>,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Attempt<Tensor>> {
    cuda_impl::fused_adaln(x, spec, scale, shift)
}

#[cfg(not(feature = "cuda"))]
fn fused_attempt(
    _x: &Tensor,
    _spec: &AdaLn<'_>,
    _scale: &Tensor,
    _shift: &Tensor,
) -> Result<Attempt<Tensor>> {
    // Never called: `dispatch` records `cuda_feature_off` before any attempt without `cuda`.
    Ok(Attempt::Reference(PathReason::CudaFeatureOff))
}

/// Why the fused AdaLN does not serve a call. The reference serves it instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaLnRefusal {
    /// An operand is not on `x`'s CUDA device.
    NotCuda,
    /// An operand carries a gradient (training keeps the composable ops, epic E9).
    Autograd,
    /// `x` is not F32 / BF16 / F16, or an operand's dtype is not the one the chain computes in.
    Dtype(DType),
    /// `x` is not a non-empty `[B, S, D]`, or a weight / bias / modulation shape does not fit it.
    Shape,
    /// A weight / bias is not contiguous, or a modulation tensor is strided along `D`.
    Layout,
    /// More than [`ADALN_MAX_ELEMS`] elements.
    TooLarge(usize),
}

impl AdaLnRefusal {
    /// The fused-path reason this refusal is counted under.
    pub fn reason(&self) -> PathReason {
        match self {
            Self::NotCuda => PathReason::NotCuda,
            Self::Autograd => PathReason::Autograd,
            Self::Dtype(d) => PathReason::Dtype(*d),
            Self::Shape => PathReason::Shape,
            Self::Layout => PathReason::Layout,
            Self::TooLarge(_) => PathReason::IndexOverflow,
        }
    }
}

impl fmt::Display for AdaLnRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "an operand is not on x's CUDA device"),
            Self::Autograd => write!(f, "an operand carries a gradient"),
            Self::Dtype(d) => write!(f, "dtype {d:?} is not served"),
            Self::Shape => write!(f, "the operand shapes are not served"),
            Self::Layout => write!(f, "an operand's layout is not served"),
            Self::TooLarge(n) => write!(f, "{n} elements exceed {ADALN_MAX_ELEMS}"),
        }
    }
}

/// One operand's metadata, as the plan sees it.
#[derive(Clone, Copy, Debug)]
pub struct OperandMeta<'a> {
    /// Element dtype.
    pub dtype: DType,
    /// Dimensions.
    pub dims: &'a [usize],
    /// Element strides.
    pub stride: &'a [usize],
    /// Whether it is on `x`'s CUDA device.
    pub on_device: bool,
    /// Whether a gradient can flow through it.
    pub tracked: bool,
}

impl<'a> OperandMeta<'a> {
    /// `t`'s metadata, `on_device` being whether it sits on `x`'s CUDA device.
    pub fn of(t: &'a Tensor, x: &Tensor) -> Self {
        Self {
            dtype: t.dtype(),
            dims: t.dims(),
            stride: t.stride(),
            on_device: t.device().is_cuda() && t.device().same_device(x.device()),
            tracked: t.track_op(),
        }
    }
}

/// How the kernel reads an optional per-column parameter (`weight_kind` / `bias_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// Absent: weight 1 / bias 0.
    Absent = 0,
    /// In `x`'s dtype.
    Native = 1,
    /// In f32.
    F32 = 2,
}

/// A served call's launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaLnPlan {
    /// `x`'s dtype (F32, BF16 or F16); the output's too.
    pub dtype: DType,
    /// Rows (`B · S`), one block each.
    pub rows: usize,
    /// Columns (`D`).
    pub n: usize,
    /// Rows per sample (`S`).
    pub seq: usize,
    /// Threads per block ([`adaln_block_dim`]).
    pub block: u32,
    /// How the weight is read.
    pub weight: ParamKind,
    /// How the bias is read.
    pub bias: ParamKind,
    /// Element strides of `scale`'s batch and sequence axes (0 broadcasts).
    pub scale_strides: (usize, usize),
    /// Element strides of `shift`'s batch and sequence axes (0 broadcasts).
    pub shift_strides: (usize, usize),
}

/// The batch / sequence strides of a `[B or 1, S or 1, D]` modulation operand (0 for a broadcast
/// axis), or why it does not fit `x`'s `(b, s, n)`.
fn modulation_strides(
    m: &OperandMeta<'_>,
    (b, s, n): (usize, usize, usize),
) -> std::result::Result<(usize, usize), AdaLnRefusal> {
    let [mb, ms, mn] = *m.dims else {
        return Err(AdaLnRefusal::Shape);
    };
    if mn != n || !(mb == b || mb == 1) || !(ms == s || ms == 1) {
        return Err(AdaLnRefusal::Shape);
    }
    if n > 1 && m.stride[2] != 1 {
        return Err(AdaLnRefusal::Layout);
    }
    let axis = |size: usize, stride: usize| if size == 1 { 0 } else { stride };
    Ok((axis(mb, m.stride[0]), axis(ms, m.stride[1])))
}

/// The fused AdaLN's input check (pure; every lane runs it). `x_on_cuda` is whether `x` lives on a
/// CUDA device; the other operands' [`OperandMeta::on_device`] whether they share it.
#[allow(clippy::too_many_arguments)]
pub fn check_adaln(
    x_on_cuda: bool,
    x: &OperandMeta<'_>,
    norm: AdaNorm,
    norm_in_f32: bool,
    weight: Option<&OperandMeta<'_>>,
    bias: Option<&OperandMeta<'_>>,
    scale: &OperandMeta<'_>,
    shift: &OperandMeta<'_>,
) -> std::result::Result<AdaLnPlan, AdaLnRefusal> {
    let params = [weight, bias];
    let all = || {
        [Some(x), Some(scale), Some(shift)]
            .into_iter()
            .chain(params)
            .flatten()
    };
    if !x_on_cuda || all().skip(1).any(|m| !m.on_device) {
        return Err(AdaLnRefusal::NotCuda);
    }
    if all().any(|m| m.tracked) {
        return Err(AdaLnRefusal::Autograd);
    }
    if !matches!(x.dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(AdaLnRefusal::Dtype(x.dtype));
    }
    for m in [scale, shift] {
        if m.dtype != x.dtype {
            return Err(AdaLnRefusal::Dtype(m.dtype));
        }
    }
    let norm_dtype = if norm_in_f32 { DType::F32 } else { x.dtype };
    for m in params.into_iter().flatten() {
        if m.dtype != norm_dtype {
            return Err(AdaLnRefusal::Dtype(m.dtype));
        }
    }
    if norm == AdaNorm::Rms && bias.is_some() {
        return Err(AdaLnRefusal::Shape);
    }
    let [b, s, n] = *x.dims else {
        return Err(AdaLnRefusal::Shape);
    };
    let rows = b * s;
    if rows == 0 || n == 0 {
        return Err(AdaLnRefusal::Shape);
    }
    let elems = rows.saturating_mul(n);
    if elems > ADALN_MAX_ELEMS {
        return Err(AdaLnRefusal::TooLarge(elems));
    }
    for m in params.into_iter().flatten() {
        if m.dims != [n] {
            return Err(AdaLnRefusal::Shape);
        }
        if n > 1 && m.stride != [1] {
            return Err(AdaLnRefusal::Layout);
        }
    }
    let scale_strides = modulation_strides(scale, (b, s, n))?;
    let shift_strides = modulation_strides(shift, (b, s, n))?;
    let kind = |m: Option<&OperandMeta<'_>>| match m {
        None => ParamKind::Absent,
        Some(_) if norm_in_f32 && x.dtype != DType::F32 => ParamKind::F32,
        Some(_) => ParamKind::Native,
    };
    Ok(AdaLnPlan {
        dtype: x.dtype,
        rows,
        n,
        seq: s,
        block: adaln_block_dim(n),
        weight: kind(weight),
        bias: kind(bias),
        scale_strides,
        shift_strides,
    })
}

/// [`check_adaln`] over live tensors.
pub fn plan_adaln(
    x: &Tensor,
    spec: &AdaLn<'_>,
    scale: &Tensor,
    shift: &Tensor,
) -> std::result::Result<AdaLnPlan, AdaLnRefusal> {
    let weight = spec.weight.map(|w| OperandMeta::of(w, x));
    let bias = spec.bias.map(|b| OperandMeta::of(b, x));
    check_adaln(
        x.device().is_cuda(),
        &OperandMeta::of(x, x),
        spec.norm,
        spec.norm_in_f32,
        weight.as_ref(),
        bias.as_ref(),
        &OperandMeta::of(scale, x),
        &OperandMeta::of(shift, x),
    )
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::nvrtc::KernelCompileError;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Storage};
    use cudarc::driver::{CudaFunction, CudaView, DeviceRepr, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    /// The three entry points of one device's module, resolved once.
    struct Functions {
        f32: CudaFunction,
        bf16: CudaFunction,
        f16: CudaFunction,
    }

    impl Functions {
        fn resolve(dev: &CudaDevice) -> std::result::Result<Self, KernelCompileError> {
            let m = ADALN_SRC.compiled(dev)?;
            Ok(Self {
                f32: m.function("adaln_modulate_f32")?,
                bf16: m.function("adaln_modulate_bf16")?,
                f16: m.function("adaln_modulate_f16")?,
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

    fn drv(e: cudarc::driver::DriverError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("fused adaln: {e:?}").into())
    }

    fn launch_i32(v: usize, what: &str) -> Result<i32> {
        i32::try_from(v).map_err(|_| candle_core::Error::Msg(format!("adaln: {what} {v} > i32")))
    }

    fn launch_i64(v: usize) -> i64 {
        // Every stride / extent here indexes a tensor of at most i32::MAX elements.
        v as i64
    }

    /// An optional weight / bias as the view its [`ParamKind`] reads: `x`'s dtype or f32.
    #[allow(clippy::type_complexity)]
    fn param_views<T: CudaDType>(
        p: Option<(&CudaStorage, usize)>,
        kind: ParamKind,
    ) -> Result<(Option<CudaView<'_, T>>, Option<CudaView<'_, f32>>)> {
        Ok(match (p, kind) {
            (Some((s, o)), ParamKind::Native) => (Some(s.as_cuda_slice::<T>()?.slice(o..)), None),
            (Some((s, o)), ParamKind::F32) => (None, Some(s.as_cuda_slice::<f32>()?.slice(o..))),
            _ => (None, None),
        })
    }

    fn as_cuda(s: &Storage) -> Option<&CudaStorage> {
        match s {
            Storage::Cuda(c) => Some(c),
            _ => None,
        }
    }

    /// An optional parameter's locked storage as `Some(Some(cuda))`, `Some(None)` when absent, or
    /// `None` when it is not a CUDA tensor.
    #[allow(clippy::type_complexity)]
    fn guarded_param<'a>(
        p: &'a Option<(std::sync::RwLockReadGuard<'_, Storage>, usize)>,
    ) -> Option<Option<(&'a CudaStorage, usize)>> {
        match p {
            Some((s, o)) => as_cuda(s).map(|c| Some((c, *o))),
            None => Some(None),
        }
    }

    /// One call's device operands: each storage with its element offset.
    struct Operands<'a> {
        x: (&'a CudaStorage, usize),
        weight: Option<(&'a CudaStorage, usize)>,
        bias: Option<(&'a CudaStorage, usize)>,
        scale: (&'a CudaStorage, usize),
        shift: (&'a CudaStorage, usize),
    }

    /// An operand's elements from its offset on, as `T`.
    fn view<T: CudaDType>((s, o): (&CudaStorage, usize)) -> Result<CudaView<'_, T>> {
        Ok(s.as_cuda_slice::<T>()?.slice(o..))
    }

    /// Launch with every same-dtype operand as `T`.
    fn launch<T: CudaDType + DeviceRepr>(
        dev: &CudaDevice,
        func: &CudaFunction,
        plan: &AdaLnPlan,
        spec: &AdaLn<'_>,
        ops: &Operands<'_>,
    ) -> Result<CudaStorage> {
        let xs = view::<T>(ops.x)?;
        let sc = view::<T>(ops.scale)?;
        let sh = view::<T>(ops.shift)?;
        let (w_t, w_f) = param_views::<T>(ops.weight, plan.weight)?;
        let (b_t, b_f) = param_views::<T>(ops.bias, plan.bias)?;
        let null: u64 = 0;
        let mut out = unsafe { dev.alloc::<T>(plan.rows * plan.n) }?;
        let n = launch_i32(plan.n, "columns")?;
        let grid = u32::try_from(plan.rows)
            .ok()
            .filter(|r| *r <= i32::MAX as u32)
            .ok_or_else(|| candle_core::Error::Msg(format!("adaln: {} rows", plan.rows)))?;
        let (weight_kind, bias_kind) = (plan.weight as i32, plan.bias as i32);
        let layer = i32::from(spec.norm == AdaNorm::Layer);
        let one_plus = i32::from(spec.scale_form == ScaleForm::OnePlus);
        let seq = launch_i64(plan.seq);
        let (scale_sb, scale_ss) = (
            launch_i64(plan.scale_strides.0),
            launch_i64(plan.scale_strides.1),
        );
        let (shift_sb, shift_ss) = (
            launch_i64(plan.shift_strides.0),
            launch_i64(plan.shift_strides.1),
        );
        let stream = dev.cuda_stream();
        let mut b = stream.launch_builder(func);
        b.arg(&xs);
        match (&w_t, &w_f) {
            (Some(v), _) => b.arg(v),
            (_, Some(v)) => b.arg(v),
            _ => b.arg(&null),
        };
        b.arg(&weight_kind);
        match (&b_t, &b_f) {
            (Some(v), _) => b.arg(v),
            (_, Some(v)) => b.arg(v),
            _ => b.arg(&null),
        };
        b.arg(&bias_kind)
            .arg(&sc)
            .arg(&sh)
            .arg(&mut out)
            .arg(&n)
            .arg(&spec.eps)
            .arg(&layer)
            .arg(&one_plus)
            .arg(&seq)
            .arg(&scale_sb)
            .arg(&scale_ss)
            .arg(&shift_sb)
            .arg(&shift_ss);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (plan.block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(drv)?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    /// The fused AdaLN: `Attempt::Fused(out)` or `Attempt::Reference(reason)` when it does not
    /// serve the call. It neither consults the switch nor records the outcome — [`ADALN_PATH`]'s
    /// `dispatch` does both.
    pub fn fused_adaln(
        x: &Tensor,
        spec: &AdaLn<'_>,
        scale: &Tensor,
        shift: &Tensor,
    ) -> Result<Attempt<Tensor>> {
        let plan = match plan_adaln(x, spec, scale, shift) {
            Ok(plan) => plan,
            Err(refusal) => return Ok(Attempt::Reference(refusal.reason())),
        };
        let Device::Cuda(dev) = x.device() else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let f = match functions(dev) {
            Ok(f) => f,
            Err(e) => return Ok(Attempt::Reference(e.into())),
        };
        // The kernel reads `x` as contiguous rows (the reference makes it contiguous too).
        let x = x.contiguous()?;
        macro_rules! cuda_storage {
            ($t:expr) => {{
                let (storage, layout) = $t.storage_and_layout();
                let offset = layout.start_offset();
                (storage, offset)
            }};
        }
        let (xs, xo) = cuda_storage!(x);
        let (ss, so) = cuda_storage!(scale);
        let (hs, ho) = cuda_storage!(shift);
        let weight = spec.weight.map(|w| cuda_storage!(w));
        let bias = spec.bias.map(|b| cuda_storage!(b));
        let (Some(xc), Some(sc), Some(hc)) = (as_cuda(&xs), as_cuda(&ss), as_cuda(&hs)) else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let (Some(wc), Some(bc)) = (guarded_param(&weight), guarded_param(&bias)) else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let ops = Operands {
            x: (xc, xo),
            weight: wc,
            bias: bc,
            scale: (sc, so),
            shift: (hc, ho),
        };
        let storage = match plan.dtype {
            DType::F32 => launch::<f32>(dev, &f.f32, &plan, spec, &ops)?,
            DType::BF16 => launch::<half::bf16>(dev, &f.bf16, &plan, spec, &ops)?,
            DType::F16 => launch::<half::f16>(dev, &f.f16, &plan, spec, &ops)?,
            other => return Ok(Attempt::Reference(PathReason::Dtype(other))),
        };
        Ok(Attempt::Fused(Tensor::from_storage(
            Storage::Cuda(storage),
            x.shape().clone(),
            BackpropOp::none(),
            false,
        )))
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::fused_adaln;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused_path::{media_fused_policy_guard, PathCounts};
    use candle_core::{Device, Var};

    fn meta<'a>(dtype: DType, dims: &'a [usize], stride: &'a [usize]) -> OperandMeta<'a> {
        OperandMeta {
            dtype,
            dims,
            stride,
            on_device: true,
            tracked: false,
        }
    }

    const X: [usize; 3] = [2, 5, 7];
    const XS: [usize; 3] = [35, 7, 1];

    fn plan_with(
        x_dtype: DType,
        scale: OperandMeta<'_>,
        weight: Option<OperandMeta<'_>>,
        norm_in_f32: bool,
    ) -> std::result::Result<AdaLnPlan, AdaLnRefusal> {
        let x = meta(x_dtype, &X, &XS);
        check_adaln(
            true,
            &x,
            AdaNorm::Rms,
            norm_in_f32,
            weight.as_ref(),
            None,
            &scale,
            &scale,
        )
    }

    #[test]
    fn the_plan_serves_per_sample_and_per_token_modulation() {
        // Per-sample [B,1,D] as a chunk of a wider [B,1,6D] table (Krea's layout).
        let per_sample = meta(DType::BF16, &[2, 1, 7], &[42, 42, 1]);
        let plan = plan_with(DType::BF16, per_sample, None, false).unwrap();
        assert_eq!((plan.rows, plan.n, plan.seq, plan.block), (10, 7, 5, 32));
        assert_eq!(plan.scale_strides, (42, 0));
        assert_eq!(plan.weight, ParamKind::Absent);
        // Per-token [B,S,D], contiguous.
        let per_token = meta(DType::BF16, &X, &XS);
        let plan = plan_with(DType::BF16, per_token, None, false).unwrap();
        assert_eq!(plan.scale_strides, (35, 7));
        // One modulation row for every sample.
        let shared = meta(DType::BF16, &[1, 1, 7], &[7, 7, 1]);
        assert_eq!(
            plan_with(DType::BF16, shared, None, false)
                .unwrap()
                .scale_strides,
            (0, 0)
        );
        // An f32 weight under norm_in_f32 for a bf16 x is read as f32; a native one natively.
        let w = meta(DType::F32, &[7], &[1]);
        let plan = plan_with(DType::BF16, shared, Some(w), true).unwrap();
        assert_eq!(plan.weight, ParamKind::F32);
        let w = meta(DType::BF16, &[7], &[1]);
        let plan = plan_with(DType::BF16, shared, Some(w), false).unwrap();
        assert_eq!(plan.weight, ParamKind::Native);
        assert_eq!(adaln_block_dim(1023), 32);
        assert_eq!(adaln_block_dim(1024), 1024);
        assert_eq!(adaln_block_dim(6144), 1024);
    }

    #[test]
    fn the_plan_refuses_what_the_kernel_does_not_serve() {
        let ok = meta(DType::BF16, &[2, 1, 7], &[7, 7, 1]);
        let x = meta(DType::BF16, &X, &XS);
        let run = |x_on_cuda: bool,
                   x: &OperandMeta<'_>,
                   norm: AdaNorm,
                   weight: Option<OperandMeta<'_>>,
                   bias: Option<OperandMeta<'_>>,
                   scale: OperandMeta<'_>| {
            check_adaln(
                x_on_cuda,
                x,
                norm,
                false,
                weight.as_ref(),
                bias.as_ref(),
                &scale,
                &ok,
            )
        };
        assert_eq!(
            run(false, &x, AdaNorm::Rms, None, None, ok),
            Err(AdaLnRefusal::NotCuda)
        );
        let elsewhere = OperandMeta {
            on_device: false,
            ..ok
        };
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, None, elsewhere),
            Err(AdaLnRefusal::NotCuda)
        );
        let tracked = OperandMeta {
            tracked: true,
            ..ok
        };
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, None, tracked),
            Err(AdaLnRefusal::Autograd)
        );
        for dtype in [DType::F64, DType::U8, DType::F8E4M3] {
            let xd = meta(dtype, &X, &XS);
            assert_eq!(
                run(true, &xd, AdaNorm::Rms, None, None, ok),
                Err(AdaLnRefusal::Dtype(dtype))
            );
        }
        let f32_scale = meta(DType::F32, &[2, 1, 7], &[7, 7, 1]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, None, f32_scale),
            Err(AdaLnRefusal::Dtype(DType::F32))
        );
        // A weight in f32 without norm_in_f32 is not the chain's dtype.
        let w32 = meta(DType::F32, &[7], &[1]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, Some(w32), None, ok),
            Err(AdaLnRefusal::Dtype(DType::F32))
        );
        let bias = meta(DType::BF16, &[7], &[1]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, Some(bias), ok),
            Err(AdaLnRefusal::Shape),
            "a bias is LayerNorm-only"
        );
        assert!(run(true, &x, AdaNorm::Layer, None, Some(bias), ok).is_ok());
        let rank2 = meta(DType::BF16, &[5, 7], &[7, 1]);
        assert_eq!(
            run(true, &rank2, AdaNorm::Rms, None, None, ok),
            Err(AdaLnRefusal::Shape)
        );
        let empty = meta(DType::BF16, &[2, 0, 7], &[0, 7, 1]);
        assert_eq!(
            run(true, &empty, AdaNorm::Rms, None, None, ok),
            Err(AdaLnRefusal::Shape)
        );
        let wrong_d = meta(DType::BF16, &[2, 1, 6], &[6, 6, 1]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, None, wrong_d),
            Err(AdaLnRefusal::Shape)
        );
        let wrong_b = meta(DType::BF16, &[3, 1, 7], &[7, 7, 1]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, None, wrong_b),
            Err(AdaLnRefusal::Shape)
        );
        let strided_d = meta(DType::BF16, &[2, 1, 7], &[14, 14, 2]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, None, None, strided_d),
            Err(AdaLnRefusal::Layout)
        );
        let short_w = meta(DType::BF16, &[6], &[1]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, Some(short_w), None, ok),
            Err(AdaLnRefusal::Shape)
        );
        let strided_w = meta(DType::BF16, &[7], &[2]);
        assert_eq!(
            run(true, &x, AdaNorm::Rms, Some(strided_w), None, ok),
            Err(AdaLnRefusal::Layout)
        );
        let huge = [1, 1 << 16, 1 << 16];
        let big = meta(DType::BF16, &huge, &[1 << 32, 1 << 16, 1]);
        let big_mod = meta(DType::BF16, &[1, 1, 1 << 16], &[1 << 16, 1 << 16, 1]);
        assert_eq!(
            run(true, &big, AdaNorm::Rms, None, None, big_mod),
            Err(AdaLnRefusal::TooLarge(1 << 32))
        );
    }

    #[test]
    fn refusals_map_to_typed_path_reasons() {
        assert_eq!(AdaLnRefusal::NotCuda.reason().label(), "not_cuda");
        assert_eq!(AdaLnRefusal::Autograd.reason().label(), "autograd");
        assert_eq!(AdaLnRefusal::Dtype(DType::U8).reason().label(), "dtype");
        assert_eq!(AdaLnRefusal::Shape.reason().label(), "shape");
        assert_eq!(AdaLnRefusal::Layout.reason().label(), "layout");
        assert_eq!(
            AdaLnRefusal::TooLarge(1 << 40).reason().label(),
            "index_overflow"
        );
    }

    /// The reference chain is today's Krea / DiT op chain, and `adaln` on CPU is that chain,
    /// recorded as a reference call (`cuda_feature_off` without `cuda`, `not_cuda` with it).
    #[test]
    fn adaln_on_cpu_is_the_reference_chain_and_is_recorded() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 5, 7), &dev)?;
        let w = (Tensor::randn(0f32, 0.1, 7, &dev)? + 1.0)?;
        let scale = Tensor::randn(0f32, 0.1, (2, 1, 7), &dev)?;
        let shift = Tensor::randn(0f32, 0.1, (2, 1, 7), &dev)?;
        let spec = AdaLn::rms(1e-6).with_weight(&w).norm_in_f32();
        let by_hand = candle_nn::ops::rms_norm(&x.contiguous()?, &w, 1e-6)?
            .broadcast_mul(&(&scale + 1.0)?)?
            .broadcast_add(&shift)?;
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ADALN_PATH.guard(Some(true));
        let start = ADALN_PATH.thread_counts();
        let got = adaln(&x, &spec, &scale, &shift)?;
        let d: PathCounts = ADALN_PATH.thread_counts().since(&start);
        let expected = if cfg!(feature = "cuda") {
            "not_cuda"
        } else {
            "cuda_feature_off"
        };
        assert_eq!((d.fused, d.reference_for(expected)), (0, 1), "{d}");
        let diff = (got - by_hand)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);

        // The direct form drops the `+ 1`; LayerNorm with a bias is the candle layer_norm chain.
        let direct = adaln_reference(&x, &spec.direct_scale(), &scale, &shift)?;
        let by_hand = candle_nn::ops::rms_norm(&x, &w, 1e-6)?
            .broadcast_mul(&scale)?
            .broadcast_add(&shift)?;
        assert_eq!(
            (direct - by_hand)?.abs()?.max_all()?.to_scalar::<f32>()?,
            0.0
        );
        let bias = Tensor::randn(0f32, 0.1, 7, &dev)?;
        let got = layer_adaln(&x, Some(&w), Some(&bias), 1e-5, &scale, &shift)?;
        let by_hand = candle_nn::ops::layer_norm(&x, &w, &bias, 1e-5)?
            .broadcast_mul(&(&scale + 1.0)?)?
            .broadcast_add(&shift)?;
        assert_eq!((got - by_hand)?.abs()?.max_all()?.to_scalar::<f32>()?, 0.0);
        // A bias on RMSNorm is refused outright.
        assert!(adaln(&x, &AdaLn::rms(1e-6).with_bias(&bias), &scale, &shift).is_err());
        Ok(())
    }

    /// A gradient-carrying operand is planned to the composable reference (epic E9).
    #[test]
    fn a_tracked_operand_is_an_autograd_refusal() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 3, 4), &dev)?;
        let scale = Var::from_tensor(&Tensor::zeros((1, 1, 4), DType::F32, &dev)?)?;
        let shift = Tensor::zeros((1, 1, 4), DType::F32, &dev)?;
        let x_meta = OperandMeta::of(&x, &x);
        let s_meta = OperandMeta {
            on_device: true,
            ..OperandMeta::of(scale.as_tensor(), &x)
        };
        let h_meta = OperandMeta {
            on_device: true,
            ..OperandMeta::of(&shift, &x)
        };
        assert!(s_meta.tracked);
        assert_eq!(
            check_adaln(
                true,
                &x_meta,
                AdaNorm::Rms,
                false,
                None,
                None,
                &s_meta,
                &h_meta
            ),
            Err(AdaLnRefusal::Autograd)
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    //! GPU bit identity of the fused AdaLN against the reference op chain. `#[ignore]`d CUDA
    //! compute (registered in `scripts/ci/test_suites.json`); run them on a CUDA box with
    //! `cargo test -p candle-quant-kernels --features cuda --lib adaln -- --ignored`.
    //!
    //! Declared tolerance: **zero ulp** for f32, bf16 and f16. Justification: the kernel performs
    //! the reference chain's exact operations in its exact order (see `adaln.cu`), so there is no
    //! approximation to bound; any difference is a defect (and the epic's f32 bound, max relative
    //! error <= 1e-5, is implied).
    use super::*;
    use crate::fused_path::media_fused_policy_guard;
    use candle_core::Device;

    fn cuda() -> Device {
        Device::new_cuda(0).expect("CUDA device 0")
    }

    /// Deterministic values in about [-amp, amp] with a heavy tail every 61st.
    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
                if i % 61 == 0 {
                    u * amp * 40.0
                } else {
                    u * amp
                }
            })
            .collect()
    }

    fn tensor(dev: &Device, shape: &[usize], seed: u64, amp: f32, dtype: DType) -> Result<Tensor> {
        let n = shape.iter().product();
        Tensor::from_vec(values(n, seed, amp), shape, dev)?.to_dtype(dtype)
    }

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        let t = t.flatten_all()?;
        Ok(match t.dtype() {
            DType::F32 => t.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect(),
            DType::BF16 => t
                .to_vec1::<half::bf16>()?
                .into_iter()
                .map(|v| v.to_bits() as u32)
                .collect(),
            DType::F16 => t
                .to_vec1::<half::f16>()?
                .into_iter()
                .map(|v| v.to_bits() as u32)
                .collect(),
            other => candle_core::bail!("no bit view of {other:?}"),
        })
    }

    /// Index, fused bits, reference bits of the first difference.
    fn first_mismatch(a: &[u32], b: &[u32]) -> Option<(usize, u32, u32)> {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .position(|(x, y)| x != y)
            .map(|i| (i, a[i], b[i]))
    }

    fn fused(x: &Tensor, spec: &AdaLn<'_>, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        match fused_adaln(x, spec, scale, shift)? {
            Attempt::Fused(t) => Ok(t),
            Attempt::Reference(r) => panic!("the fused adaln declined: {r}"),
        }
    }

    /// Every norm x weight x precision x scale-form combination, f32 / bf16 / f16, per-sample
    /// and per-token modulation, D below 1024 (one-warp blocks), above it (1024-thread blocks),
    /// at Krea's 6144 and at sizes not divisible by any vector width (1, 7, 1027, 6151).
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_adaln_gpu_is_bit_identical_to_the_reference_chain() -> Result<()> {
        let dev = cuda();
        let mut checked = 0usize;
        for &(b, s, n) in &[
            (1, 3, 1),
            (2, 5, 7),
            (1, 9, 96),
            (2, 17, 1027),
            (1, 33, 6144),
            (2, 4, 6151),
        ] {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let seed = (b * 1000 + s * 10 + n) as u64;
                let x = tensor(&dev, &[b, s, n], seed, 3.0, dtype)?;
                let per_sample = (
                    tensor(&dev, &[b, 1, n], seed + 1, 0.5, dtype)?,
                    tensor(&dev, &[b, 1, n], seed + 2, 0.5, dtype)?,
                );
                let per_token = (
                    tensor(&dev, &[b, s, n], seed + 3, 0.5, dtype)?,
                    tensor(&dev, &[b, s, n], seed + 4, 0.5, dtype)?,
                );
                for norm in [AdaNorm::Rms, AdaNorm::Layer] {
                    for norm_in_f32 in [false, true] {
                        let pdt = if norm_in_f32 { DType::F32 } else { dtype };
                        let w = (tensor(&dev, &[n], seed + 5, 0.3, DType::F32)? + 1.0)?
                            .to_dtype(pdt)?;
                        let bias = tensor(&dev, &[n], seed + 6, 0.2, pdt)?;
                        for (weighted, biased) in [(false, false), (true, false), (true, true)] {
                            if biased && norm == AdaNorm::Rms {
                                continue;
                            }
                            for form in [ScaleForm::OnePlus, ScaleForm::Direct] {
                                let mut spec = AdaLn {
                                    norm,
                                    scale_form: form,
                                    norm_in_f32,
                                    ..AdaLn::rms(if norm == AdaNorm::Rms { 1e-6 } else { 1e-5 })
                                };
                                if weighted {
                                    spec = spec.with_weight(&w);
                                }
                                if biased {
                                    spec = spec.with_bias(&bias);
                                }
                                for (label, (scale, shift)) in
                                    [("per-sample", &per_sample), ("per-token", &per_token)]
                                {
                                    let want = adaln_reference(&x, &spec, scale, shift)?;
                                    let got = fused(&x, &spec, scale, shift)?;
                                    assert_eq!(got.dims(), want.dims());
                                    assert_eq!(got.dtype(), dtype);
                                    assert_eq!(
                                        first_mismatch(&bits(&got)?, &bits(&want)?),
                                        None,
                                        "[{b},{s},{n}] {dtype:?} {norm:?} f32norm={norm_in_f32} \
                                         w={weighted} b={biased} {form:?} {label}: (index, fused \
                                         bits, reference bits)"
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!("[adaln] {checked} fused calls bit-identical to the reference chain");
        Ok(())
    }

    /// Krea's own layout: a bf16 `[1, S, 6144]` stream, an f32 `+1`-folded weight with
    /// `norm_in_f32`, and the scale / shift as chunks of one `[B, 1, 6·D]` table (strided views,
    /// no copy), at B = 1 and B = 2 (CFG batch).
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_adaln_gpu_serves_krea_chunked_modulation_bit_identically() -> Result<()> {
        let dev = cuda();
        let n = 6144;
        for b in [1usize, 2] {
            let x = tensor(&dev, &[b, 300, n], 11 + b as u64, 2.0, DType::BF16)?;
            let w = (tensor(&dev, &[n], 12, 0.3, DType::F32)? + 1.0)?;
            let table = tensor(&dev, &[b, 1, 6 * n], 13, 0.4, DType::BF16)?;
            let chunks = table.chunk(6, D::Minus1)?;
            let spec = AdaLn::rms(1e-6).with_weight(&w).norm_in_f32();
            for (scale, shift) in [(&chunks[0], &chunks[1]), (&chunks[3], &chunks[4])] {
                let want = adaln_reference(&x, &spec, scale, shift)?;
                let got = fused(&x, &spec, scale, shift)?;
                assert_eq!(first_mismatch(&bits(&got)?, &bits(&want)?), None, "B={b}");
            }
        }
        Ok(())
    }

    /// Mutation check: a kernel that dropped the `+ 1` (the Direct form standing in for it) or
    /// used the wrong epsilon is caught by the bit-identity comparison.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_adaln_gpu_parity_catches_a_dropped_one_plus() -> Result<()> {
        let dev = cuda();
        let x = tensor(&dev, &[1, 64, 6144], 21, 2.0, DType::BF16)?;
        let w = (tensor(&dev, &[6144], 22, 0.3, DType::F32)? + 1.0)?;
        let scale = tensor(&dev, &[1, 1, 6144], 23, 0.4, DType::BF16)?;
        let shift = tensor(&dev, &[1, 1, 6144], 24, 0.4, DType::BF16)?;
        let spec = AdaLn::rms(1e-6).with_weight(&w).norm_in_f32();
        let want = bits(&adaln_reference(&x, &spec, &scale, &shift)?)?;
        let mutated = bits(&fused(&x, &spec.direct_scale(), &scale, &shift)?)?;
        assert!(first_mismatch(&mutated, &want).is_some());
        let wrong_eps = AdaLn { eps: 1e-2, ..spec };
        let mutated = bits(&fused(&x, &wrong_eps, &scale, &shift)?)?;
        assert!(first_mismatch(&mutated, &want).is_some());
        Ok(())
    }

    /// The public `adaln` records fused on CUDA; the switch sends the same call to the reference
    /// (`disabled`, identical output); a CPU input is `not_cuda`; a tracked operand `autograd`.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn adaln_reports_its_path() -> Result<()> {
        let dev = cuda();
        let x = tensor(&dev, &[1, 16, 512], 31, 2.0, DType::BF16)?;
        let scale = tensor(&dev, &[1, 1, 512], 32, 0.4, DType::BF16)?;
        let shift = tensor(&dev, &[1, 1, 512], 33, 0.4, DType::BF16)?;
        let spec = AdaLn::rms(1e-6);
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ADALN_PATH.guard(Some(true));

        let start = ADALN_PATH.thread_counts();
        let on = adaln(&x, &spec, &scale, &shift)?;
        let d = ADALN_PATH.thread_counts().since(&start);
        assert_eq!((d.fused, d.reference_total()), (1, 0), "{d}");

        ADALN_PATH.set_enabled(Some(false));
        let mid = ADALN_PATH.thread_counts();
        let off = adaln(&x, &spec, &scale, &shift)?;
        let d = ADALN_PATH.thread_counts().since(&mid);
        assert_eq!((d.fused, d.reference_for("disabled")), (0, 1), "{d}");
        assert_eq!(first_mismatch(&bits(&on)?, &bits(&off)?), None);
        ADALN_PATH.set_enabled(Some(true));

        let before = ADALN_PATH.thread_counts();
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu);
        adaln(&cpu(&x)?, &spec, &cpu(&scale)?, &cpu(&shift)?)?;
        let tracked = candle_core::Var::from_tensor(&scale)?;
        adaln(&x, &spec, tracked.as_tensor(), &shift)?;
        let d = ADALN_PATH.thread_counts().since(&before);
        assert_eq!(
            (
                d.fused,
                d.reference_for("not_cuda"),
                d.reference_for("autograd")
            ),
            (0, 1, 1),
            "{d}"
        );
        Ok(())
    }
}
