//! The fused AdaLN for the **composable** norm chains (SW-6129, feature SW-F287 S4): the DiT
//! providers' own `mean_keepdim` / `broadcast_sub` / `sqr` / `+ eps` / `sqrt` / `broadcast_div`
//! LayerNorm and RMS chains (and candle_nn's LayerNorm without a bias, which is the same chain)
//! followed by the AdaLN modulation, as one launch over [`ADALN_CHAIN_SRC`] — the sibling of S2's
//! [`crate::adaln`] kernel, which reproduces candle's *fused* `ops::rms_norm` / `ops::layer_norm`.
//!
//! # The reference it must equal bit for bit
//!
//! [`adaln_chain_reference`] is the crates' chain, op for op (see [`AdaLnChain`]):
//!
//! ```text
//! xc = x.to_dtype(F32)            (ChainPrecision::F32)  |  x  (ChainPrecision::Native)
//! Layer:    c = xc - xc.mean_keepdim(-1);  y = c / (c.sqr().mean_keepdim(-1) + eps).sqrt()
//! Rms:      y = xc / (xc.sqr().mean_keepdim(-1) + eps).sqrt()
//! RmsRecip: y = xc * (xc.sqr().mean_keepdim(-1) + eps).sqrt().recip()
//! [y = y * weight] [y = y + bias]
//! out = modulate_with(y.to_dtype(M), scale, shift, form).to_dtype(O)
//! ```
//!
//! On CUDA the kernel replays each of those candle kernels' arithmetic in their order — candle's
//! `fast_sum` reduction tree (one block per row, `next_pow2(min(1024, n))` threads, strided
//! per-thread accumulation then a shared-memory halving tree), the affine `fma`, IEEE f32 sqrt /
//! div / rcp, and for native bf16 the `hsqrt` / `__hdiv` approximations of `cuda_bf16.hpp` (the
//! contract at the top of `adaln_chain.cu`). Everything else — CPU, an unserved dtype / shape /
//! layout, a gradient-tracked operand (E9), the switch off — runs the reference itself, so the
//! CPU path is the crates' chain exactly. Every call is recorded in [`ADALN_PATH`] (switch
//! `CANDLE_FUSED_ADALN`), the same record as the S2 kernel: one AdaLN path for every DiT.

use candle_core::{DType, Result, Tensor, D};

pub use crate::adaln::ADALN_PATH;
use crate::adaln::{modulate_with, AdaLnRefusal, OperandMeta, ScaleForm, ADALN_MAX_ELEMS};
use crate::fused_path::{Attempt, PathReason};
use crate::nvrtc::KernelSource;

/// The composable-chain AdaLN source; compiled once per device on first use.
pub const ADALN_CHAIN_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_adaln_chain_v1",
    src: include_str!("adaln_chain.cu"),
    // f32 arithmetic, `cvt` (bf16 only behind `__CUDA_ARCH__ >= 800`), approx sqrt / div, shared
    // memory: sm_70.
    cc_floor: (7, 0),
};

/// candle `fast_sum`'s block size for a row of `n` columns: `next_pow2(min(1024, n))`.
pub fn chain_block_dim(n: usize) -> u32 {
    n.clamp(1, 1024).next_power_of_two() as u32
}

/// Which composable normalization precedes the modulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainNorm {
    /// `c = x - mean(x); c / sqrt(mean(c²) + eps)`.
    Layer,
    /// `x / sqrt(mean(x²) + eps)`.
    Rms,
    /// `x * recip(sqrt(mean(x²) + eps))`.
    RmsRecip,
}

/// The dtype the norm chain computes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainPrecision {
    /// `x.to_dtype(F32)` first (the identity for an f32 `x`).
    F32,
    /// Every op in `x`'s own dtype.
    Native,
}

/// One composable AdaLN call. Build with [`AdaLnChain::layer`] / [`AdaLnChain::rms`] /
/// [`AdaLnChain::rms_recip`] and the modifiers.
#[derive(Clone, Copy, Debug)]
pub struct AdaLnChain<'a> {
    /// The norm chain.
    pub norm: ChainNorm,
    /// The `+ eps` operand (candle's `affine` rounds it to the compute dtype).
    pub eps: f64,
    /// Compute dtype of the norm chain.
    pub precision: ChainPrecision,
    /// `y * weight` after the norm (compute dtype, `[D]`).
    pub weight: Option<&'a Tensor>,
    /// `y + bias` after the weight (compute dtype, `[D]`).
    pub bias: Option<&'a Tensor>,
    /// The dtype the norm is cast to and the modulation runs in (`None`: `x`'s dtype).
    pub modulate_dtype: Option<DType>,
    /// The dtype the modulated result is cast to (`None`: the modulation dtype).
    pub out_dtype: Option<DType>,
    /// `(1 + scale)` or `scale`.
    pub scale_form: ScaleForm,
}

impl<'a> AdaLnChain<'a> {
    fn new(norm: ChainNorm, eps: f64) -> Self {
        Self {
            norm,
            eps,
            precision: ChainPrecision::F32,
            weight: None,
            bias: None,
            modulate_dtype: None,
            out_dtype: None,
            scale_form: ScaleForm::OnePlus,
        }
    }

    /// The composable LayerNorm chain (f32, cast back to `x`'s dtype, `(1 + scale)`).
    pub fn layer(eps: f64) -> Self {
        Self::new(ChainNorm::Layer, eps)
    }

    /// The composable RMS chain `x / sqrt(mean(x²) + eps)`.
    pub fn rms(eps: f64) -> Self {
        Self::new(ChainNorm::Rms, eps)
    }

    /// The composable RMS chain `x * recip(sqrt(mean(x²) + eps))`.
    pub fn rms_recip(eps: f64) -> Self {
        Self::new(ChainNorm::RmsRecip, eps)
    }

    /// Compute the chain in `x`'s own dtype (no f32 upcast).
    pub fn native(self) -> Self {
        Self {
            precision: ChainPrecision::Native,
            ..self
        }
    }

    /// Multiply the norm by `weight` (compute dtype) before the cast.
    pub fn with_weight(self, weight: &'a Tensor) -> Self {
        Self {
            weight: Some(weight),
            ..self
        }
    }

    /// Add `bias` (compute dtype) after the weight.
    pub fn with_bias(self, bias: &'a Tensor) -> Self {
        Self {
            bias: Some(bias),
            ..self
        }
    }

    /// Cast the norm to `dtype` and modulate in it (default `x`'s dtype).
    pub fn modulate_in(self, dtype: DType) -> Self {
        Self {
            modulate_dtype: Some(dtype),
            ..self
        }
    }

    /// Cast the modulated result to `dtype` (default the modulation dtype).
    pub fn output(self, dtype: DType) -> Self {
        Self {
            out_dtype: Some(dtype),
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

    fn compute_dtype(&self, x: DType) -> DType {
        match self.precision {
            ChainPrecision::F32 => DType::F32,
            ChainPrecision::Native => x,
        }
    }
}

/// The norm half of the reference chain, ending in the modulation dtype.
pub fn adaln_chain_norm_reference(x: &Tensor, spec: &AdaLnChain<'_>) -> Result<Tensor> {
    let cdt = spec.compute_dtype(x.dtype());
    let xc = x.to_dtype(cdt)?;
    let ms =
        |t: &Tensor| -> Result<Tensor> { (t.sqr()?.mean_keepdim(D::Minus1)? + spec.eps)?.sqrt() };
    let mut y = match spec.norm {
        ChainNorm::Layer => {
            let c = xc.broadcast_sub(&xc.mean_keepdim(D::Minus1)?)?;
            c.broadcast_div(&ms(&c)?)?
        }
        ChainNorm::Rms => xc.broadcast_div(&ms(&xc)?)?,
        ChainNorm::RmsRecip => xc.broadcast_mul(&ms(&xc)?.recip()?)?,
    };
    if let Some(w) = spec.weight {
        y = y.broadcast_mul(&w.to_dtype(cdt)?)?;
    }
    if let Some(b) = spec.bias {
        y = y.broadcast_add(&b.to_dtype(cdt)?)?;
    }
    y.to_dtype(spec.modulate_dtype.unwrap_or(x.dtype()))
}

/// The reference op chain (epic E1) — the providers' composable chain, verbatim.
pub fn adaln_chain_reference(
    x: &Tensor,
    spec: &AdaLnChain<'_>,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    let mdt = spec.modulate_dtype.unwrap_or(x.dtype());
    let out = modulate_with(
        &adaln_chain_norm_reference(x, spec)?,
        scale,
        shift,
        spec.scale_form,
    )?;
    out.to_dtype(spec.out_dtype.unwrap_or(mdt))
}

/// `norm(x) · (1 + scale) + shift` over the composable chain `spec`: the fused kernel on CUDA
/// when it serves the call, else [`adaln_chain_reference`] — bit-identical either way, recorded in
/// [`ADALN_PATH`].
///
/// `x` is `[B, S, D]`; `scale` / `shift` `[B or 1, S or 1, D]` in the modulation dtype (any batch
/// / sequence strides, unit stride along `D`) for the fused path; anything else broadcasts as candle
/// does on the reference.
pub fn adaln_chain(
    x: &Tensor,
    spec: &AdaLnChain<'_>,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    ADALN_PATH.dispatch(
        || chain_attempt(x, spec, scale, Some(shift)),
        || adaln_chain_reference(x, spec, scale, shift),
    )
}

/// The scale-only chain `norm(x) · (1 + scale)` (no shift): [`adaln_chain_norm_reference`] then
/// one `broadcast_mul`, then the output cast.
pub fn adaln_chain_scale_only_reference(
    x: &Tensor,
    spec: &AdaLnChain<'_>,
    scale: &Tensor,
) -> Result<Tensor> {
    let mdt = spec.modulate_dtype.unwrap_or(x.dtype());
    let normed = adaln_chain_norm_reference(x, spec)?;
    let out = match spec.scale_form {
        ScaleForm::OnePlus => normed.broadcast_mul(&(scale + 1.0)?)?,
        ScaleForm::Direct => normed.broadcast_mul(scale)?,
    };
    out.to_dtype(spec.out_dtype.unwrap_or(mdt))
}

/// [`adaln_chain`] without a shift, recorded in [`ADALN_PATH`].
pub fn adaln_chain_scale_only(x: &Tensor, spec: &AdaLnChain<'_>, scale: &Tensor) -> Result<Tensor> {
    ADALN_PATH.dispatch(
        || chain_attempt(x, spec, scale, None),
        || adaln_chain_scale_only_reference(x, spec, scale),
    )
}

#[cfg(feature = "cuda")]
fn chain_attempt(
    x: &Tensor,
    spec: &AdaLnChain<'_>,
    scale: &Tensor,
    shift: Option<&Tensor>,
) -> Result<Attempt<Tensor>> {
    cuda_impl::fused_adaln_chain(x, spec, scale, shift)
}

#[cfg(not(feature = "cuda"))]
fn chain_attempt(
    _x: &Tensor,
    _spec: &AdaLnChain<'_>,
    _scale: &Tensor,
    _shift: Option<&Tensor>,
) -> Result<Attempt<Tensor>> {
    // Never called: `dispatch` records `cuda_feature_off` before any attempt without `cuda`.
    Ok(Attempt::Reference(PathReason::CudaFeatureOff))
}

/// A served call's launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainPlan {
    /// `x`'s dtype.
    pub x_dtype: DType,
    /// Whether the chain computes in bf16 (native precision on a bf16 `x`).
    pub bf16_compute: bool,
    /// The modulation dtype (scale / shift, the cast norm).
    pub modulate_dtype: DType,
    /// The output dtype.
    pub out_dtype: DType,
    /// Rows (`B · S`), one block each.
    pub rows: usize,
    /// Columns (`D`).
    pub n: usize,
    /// Rows per sample (`S`).
    pub seq: usize,
    /// Threads per block ([`chain_block_dim`]).
    pub block: u32,
    /// Element strides of `scale`'s batch and sequence axes (0 broadcasts).
    pub scale_strides: (usize, usize),
    /// Element strides of `shift`'s batch and sequence axes (0 broadcasts).
    pub shift_strides: (usize, usize),
}

fn served(d: DType) -> bool {
    matches!(d, DType::F32 | DType::BF16 | DType::F16)
}

/// The composable AdaLN's input check (pure; every lane runs it).
#[allow(clippy::too_many_arguments)]
pub fn check_adaln_chain(
    x_on_cuda: bool,
    x: &OperandMeta<'_>,
    precision: ChainPrecision,
    modulate_dtype: DType,
    out_dtype: DType,
    weight: Option<&OperandMeta<'_>>,
    bias: Option<&OperandMeta<'_>>,
    scale: &OperandMeta<'_>,
    shift: &OperandMeta<'_>,
) -> std::result::Result<ChainPlan, AdaLnRefusal> {
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
    for d in [x.dtype, modulate_dtype, out_dtype] {
        if !served(d) {
            return Err(AdaLnRefusal::Dtype(d));
        }
    }
    // Native precision is replayed for f32 (IEEE) and bf16 (cuda_bf16.hpp); f16 native is not.
    let compute = match precision {
        ChainPrecision::F32 => DType::F32,
        ChainPrecision::Native if x.dtype == DType::F16 => {
            return Err(AdaLnRefusal::Dtype(DType::F16))
        }
        ChainPrecision::Native => x.dtype,
    };
    for m in [scale, shift] {
        if m.dtype != modulate_dtype {
            return Err(AdaLnRefusal::Dtype(m.dtype));
        }
    }
    for m in params.into_iter().flatten() {
        if m.dtype != compute {
            return Err(AdaLnRefusal::Dtype(m.dtype));
        }
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
    let scale_strides = crate::adaln::modulation_strides(scale, (b, s, n))?;
    let shift_strides = crate::adaln::modulation_strides(shift, (b, s, n))?;
    Ok(ChainPlan {
        x_dtype: x.dtype,
        bf16_compute: compute == DType::BF16,
        modulate_dtype,
        out_dtype,
        rows,
        n,
        seq: s,
        block: chain_block_dim(n),
        scale_strides,
        shift_strides,
    })
}

/// [`check_adaln_chain`] over live tensors.
pub fn plan_adaln_chain(
    x: &Tensor,
    spec: &AdaLnChain<'_>,
    scale: &Tensor,
    shift: &Tensor,
) -> std::result::Result<ChainPlan, AdaLnRefusal> {
    let weight = spec.weight.map(|w| OperandMeta::of(w, x));
    let bias = spec.bias.map(|b| OperandMeta::of(b, x));
    let mdt = spec.modulate_dtype.unwrap_or(x.dtype());
    check_adaln_chain(
        x.device().is_cuda(),
        &OperandMeta::of(x, x),
        spec.precision,
        mdt,
        spec.out_dtype.unwrap_or(mdt),
        weight.as_ref(),
        bias.as_ref(),
        &OperandMeta::of(scale, x),
        &OperandMeta::of(shift, x),
    )
}

/// The affine operand `v` as candle passes it to the compute dtype's kernel (`T::from_f64`).
pub fn chain_affine_operand(v: f64, bf16_compute: bool) -> f32 {
    if bf16_compute {
        half::bf16::from_f64(v).to_f32()
    } else {
        v as f32
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::nvrtc::KernelCompileError;
    use candle_core::cuda_backend::cudarc;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Storage};
    use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    /// One device's entry point, resolved once (E8); a compile failure stays cached in the seam.
    fn function(
        dev: &CudaDevice,
    ) -> std::result::Result<&'static CudaFunction, KernelCompileError> {
        static TABLES: OnceLock<RwLock<HashMap<usize, &'static CudaFunction>>> = OnceLock::new();
        let tables = TABLES.get_or_init(Default::default);
        let ordinal = dev.cuda_stream().context().ordinal();
        if let Some(f) = tables
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&ordinal)
        {
            return Ok(f);
        }
        let f = ADALN_CHAIN_SRC
            .compiled(dev)?
            .function("adaln_chain_modulate")?;
        let mut tables = tables.write().unwrap_or_else(|p| p.into_inner());
        Ok(*tables
            .entry(ordinal)
            .or_insert_with(|| Box::leak(Box::new(f))))
    }

    fn drv(e: cudarc::driver::DriverError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("fused adaln chain: {e:?}").into())
    }

    /// The kernel's dtype code: 0 f32, 1 bf16, 2 f16.
    fn dtype_code(d: DType) -> i32 {
        match d {
            DType::BF16 => 1,
            DType::F16 => 2,
            _ => 0,
        }
    }

    fn as_cuda(s: &Storage) -> Option<&CudaStorage> {
        match s {
            Storage::Cuda(c) => Some(c),
            _ => None,
        }
    }

    /// An optional parameter's storage: `None` absent, `Some(None)` not on CUDA.
    #[allow(clippy::type_complexity)]
    fn param<'a>(
        p: &'a Option<(
            std::sync::RwLockReadGuard<'_, Storage>,
            &candle_core::Layout,
        )>,
    ) -> Option<Option<(&'a CudaStorage, usize)>> {
        p.as_ref()
            .map(|(s, l)| as_cuda(s).map(|c| (c, l.start_offset())))
    }

    /// A device operand of any served dtype, viewed from its element offset.
    enum Raw<'a> {
        F32(CudaView<'a, f32>),
        Bf16(CudaView<'a, half::bf16>),
        F16(CudaView<'a, half::f16>),
    }

    fn raw(s: &CudaStorage, offset: usize, dtype: DType) -> Result<Raw<'_>> {
        Ok(match dtype {
            DType::F32 => Raw::F32(s.as_cuda_slice::<f32>()?.slice(offset..)),
            DType::BF16 => Raw::Bf16(s.as_cuda_slice::<half::bf16>()?.slice(offset..)),
            _ => Raw::F16(s.as_cuda_slice::<half::f16>()?.slice(offset..)),
        })
    }

    /// The output buffer in its dtype.
    enum Out {
        F32(CudaSlice<f32>),
        Bf16(CudaSlice<half::bf16>),
        F16(CudaSlice<half::f16>),
    }

    pub fn fused_adaln_chain(
        x: &Tensor,
        spec: &AdaLnChain<'_>,
        scale: &Tensor,
        shift: Option<&Tensor>,
    ) -> Result<Attempt<Tensor>> {
        let has_shift = i32::from(shift.is_some());
        let shift = shift.unwrap_or(scale);
        let plan = match plan_adaln_chain(x, spec, scale, shift) {
            Ok(plan) => plan,
            Err(refusal) => return Ok(Attempt::Reference(refusal.reason())),
        };
        let Device::Cuda(dev) = x.device() else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let func = match function(dev) {
            Ok(f) => f,
            Err(e) => return Ok(Attempt::Reference(e.into())),
        };
        let x = x.contiguous()?;
        let (xs, xl) = x.storage_and_layout();
        let (ss, sl) = scale.storage_and_layout();
        let (hs, hl) = shift.storage_and_layout();
        let ws = spec.weight.map(|w| w.storage_and_layout());
        let bs = spec.bias.map(|b| b.storage_and_layout());
        let (Some(xc), Some(sc), Some(hc)) = (as_cuda(&xs), as_cuda(&ss), as_cuda(&hs)) else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let cdt = spec.compute_dtype(plan.x_dtype);
        let (wp, bp) = (param(&ws), param(&bs));
        if matches!(wp, Some(None)) || matches!(bp, Some(None)) {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        }
        let xv = raw(xc, xl.start_offset(), plan.x_dtype)?;
        let sv = raw(sc, sl.start_offset(), plan.modulate_dtype)?;
        let hv = raw(hc, hl.start_offset(), plan.modulate_dtype)?;
        let wv = wp.flatten().map(|(c, o)| raw(c, o, cdt)).transpose()?;
        let bv = bp.flatten().map(|(c, o)| raw(c, o, cdt)).transpose()?;

        let elems = plan.rows * plan.n;
        // SAFETY: every element is written by the kernel before it is read.
        let mut out = match plan.out_dtype {
            DType::F32 => Out::F32(unsafe { dev.alloc::<f32>(elems) }?),
            DType::BF16 => Out::Bf16(unsafe { dev.alloc::<half::bf16>(elems) }?),
            _ => Out::F16(unsafe { dev.alloc::<half::f16>(elems) }?),
        };
        let grid = u32::try_from(plan.rows)
            .ok()
            .filter(|r| *r <= i32::MAX as u32)
            .ok_or_else(|| candle_core::Error::Msg(format!("adaln chain: {} rows", plan.rows)))?;
        let n = i32::try_from(plan.n)
            .map_err(|_| candle_core::Error::Msg(format!("adaln chain: {} columns", plan.n)))?;
        let (dt_x, dt_m, dt_o) = (
            dtype_code(plan.x_dtype),
            dtype_code(plan.modulate_dtype),
            dtype_code(plan.out_dtype),
        );
        let norm = match spec.norm {
            ChainNorm::Layer => 0i32,
            ChainNorm::Rms => 1,
            ChainNorm::RmsRecip => 2,
        };
        let bf = i32::from(plan.bf16_compute);
        let inv_n = chain_affine_operand(1f64 / plan.n as f64, plan.bf16_compute);
        let eps = chain_affine_operand(spec.eps, plan.bf16_compute);
        let one_plus = i32::from(spec.scale_form == ScaleForm::OnePlus);
        let (wk, bk) = (i32::from(wv.is_some()), i32::from(bv.is_some()));
        let i64_of = |v: usize| v as i64;
        let (seq, ssb, sss, hsb, hss) = (
            i64_of(plan.seq),
            i64_of(plan.scale_strides.0),
            i64_of(plan.scale_strides.1),
            i64_of(plan.shift_strides.0),
            i64_of(plan.shift_strides.1),
        );
        let null: u64 = 0;
        let stream = dev.cuda_stream();
        let mut b = stream.launch_builder(func);
        macro_rules! arg_raw {
            ($v:expr) => {
                match $v {
                    Raw::F32(v) => b.arg(v),
                    Raw::Bf16(v) => b.arg(v),
                    Raw::F16(v) => b.arg(v),
                }
            };
        }
        arg_raw!(&xv);
        b.arg(&dt_x);
        match &wv {
            Some(v) => arg_raw!(v),
            None => b.arg(&null),
        };
        b.arg(&wk);
        match &bv {
            Some(v) => arg_raw!(v),
            None => b.arg(&null),
        };
        b.arg(&bk);
        arg_raw!(&sv);
        arg_raw!(&hv);
        b.arg(&dt_m);
        match &mut out {
            Out::F32(o) => b.arg(o),
            Out::Bf16(o) => b.arg(o),
            Out::F16(o) => b.arg(o),
        };
        b.arg(&dt_o)
            .arg(&n)
            .arg(&norm)
            .arg(&bf)
            .arg(&inv_n)
            .arg(&eps)
            .arg(&one_plus)
            .arg(&seq)
            .arg(&ssb)
            .arg(&sss)
            .arg(&hsb)
            .arg(&hss)
            .arg(&has_shift);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (plan.block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(drv)?;
        drop(b);
        let storage = match out {
            Out::F32(o) => CudaStorage::wrap_cuda_slice(o, dev.clone()),
            Out::Bf16(o) => CudaStorage::wrap_cuda_slice(o, dev.clone()),
            Out::F16(o) => CudaStorage::wrap_cuda_slice(o, dev.clone()),
        };
        Ok(Attempt::Fused(Tensor::from_storage(
            Storage::Cuda(storage),
            x.shape().clone(),
            BackpropOp::none(),
            false,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused_path::media_fused_policy_guard;
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

    #[test]
    fn the_block_dim_is_candle_fast_sums() {
        for (n, bd) in [
            (1, 1),
            (3, 4),
            (32, 32),
            (96, 128),
            (1000, 1024),
            (1024, 1024),
            (3072, 1024),
        ] {
            assert_eq!(chain_block_dim(n), bd, "n={n}");
        }
    }

    #[test]
    fn affine_operands_round_like_candle() {
        let inv = 1f64 / 3072.0;
        assert_eq!(chain_affine_operand(inv, false), inv as f32);
        assert_eq!(
            chain_affine_operand(1e-6, true),
            half::bf16::from_f64(1e-6).to_f32()
        );
        assert_ne!(chain_affine_operand(1e-6, true), 1e-6f32);
    }

    #[test]
    fn the_plan_serves_and_refuses() {
        let (x, xs) = ([2usize, 5, 7], [35usize, 7, 1]);
        let (m, ms) = ([2usize, 1, 7], [7usize, 7, 1]);
        let w = [7usize];
        let ws = [1usize];
        let plan = |xd: DType,
                    p: ChainPrecision,
                    md: DType,
                    sd: DType,
                    wd: Option<DType>|
         -> std::result::Result<ChainPlan, AdaLnRefusal> {
            let wm = wd.map(|d| meta(d, &w, &ws));
            check_adaln_chain(
                true,
                &meta(xd, &x, &xs),
                p,
                md,
                md,
                wm.as_ref(),
                None,
                &meta(sd, &m, &ms),
                &meta(sd, &m, &ms),
            )
        };
        let ok = plan(
            DType::BF16,
            ChainPrecision::F32,
            DType::BF16,
            DType::BF16,
            None,
        )
        .unwrap();
        assert_eq!(
            (ok.rows, ok.n, ok.seq, ok.block, ok.bf16_compute),
            (10, 7, 5, 8, false)
        );
        assert_eq!(ok.scale_strides, (7, 0));
        let native = plan(
            DType::BF16,
            ChainPrecision::Native,
            DType::BF16,
            DType::BF16,
            None,
        );
        assert!(native.unwrap().bf16_compute);
        assert_eq!(
            plan(
                DType::F16,
                ChainPrecision::Native,
                DType::F16,
                DType::F16,
                None
            ),
            Err(AdaLnRefusal::Dtype(DType::F16))
        );
        assert_eq!(
            plan(
                DType::BF16,
                ChainPrecision::F32,
                DType::F32,
                DType::BF16,
                None
            ),
            Err(AdaLnRefusal::Dtype(DType::BF16))
        );
        assert_eq!(
            plan(
                DType::BF16,
                ChainPrecision::F32,
                DType::BF16,
                DType::BF16,
                Some(DType::BF16)
            ),
            Err(AdaLnRefusal::Dtype(DType::BF16)),
            "an f32-precision chain takes an f32 weight"
        );
        let rank2 = check_adaln_chain(
            true,
            &meta(DType::F32, &[10, 7], &[7, 1]),
            ChainPrecision::F32,
            DType::F32,
            DType::F32,
            None,
            None,
            &meta(DType::F32, &m, &ms),
            &meta(DType::F32, &m, &ms),
        );
        assert_eq!(rank2, Err(AdaLnRefusal::Shape));
        let cpu = check_adaln_chain(
            false,
            &meta(DType::F32, &x, &xs),
            ChainPrecision::F32,
            DType::F32,
            DType::F32,
            None,
            None,
            &meta(DType::F32, &m, &ms),
            &meta(DType::F32, &m, &ms),
        );
        assert_eq!(cpu, Err(AdaLnRefusal::NotCuda));
    }

    /// The reference is the providers' chains verbatim: the Lens / Qwen-Image f32 LayerNorm cast
    /// back, the LTX `recip` RMS, the FLUX.2 f32 LayerNorm modulated in f32, a native chain.
    #[test]
    fn the_reference_is_the_crates_chain() -> Result<()> {
        let dev = Device::Cpu;
        let x = (Tensor::randn(0f32, 2.0, (2, 6, 40), &dev)? + 0.5)?.to_dtype(DType::BF16)?;
        let scale = Tensor::randn(0f32, 0.3, (2, 1, 40), &dev)?.to_dtype(DType::BF16)?;
        let shift = Tensor::randn(0f32, 0.3, (2, 1, 40), &dev)?.to_dtype(DType::BF16)?;
        let eps = 1e-6;
        // Lens `layer_norm` + `modulate`.
        let lens = {
            let xf = x.to_dtype(DType::F32)?;
            let mean = xf.mean_keepdim(D::Minus1)?;
            let xc = xf.broadcast_sub(&mean)?;
            let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
            let n = xc
                .broadcast_div(&(var + eps)?.sqrt()?)?
                .to_dtype(DType::BF16)?;
            n.broadcast_mul(&(&scale + 1.0)?)?.broadcast_add(&shift)?
        };
        let got = adaln_chain_reference(&x, &AdaLnChain::layer(eps), &scale, &shift)?;
        assert_eq!(bits(&got)?, bits(&lens)?);
        // LTX `rms_noweight`.
        let ltx = {
            let xf = x.to_dtype(DType::F32)?.contiguous()?;
            let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
                .sqrt()?
                .recip()?;
            let n = xf.broadcast_mul(&inv)?.to_dtype(DType::BF16)?;
            n.broadcast_mul(&(&scale + 1.0)?)?.broadcast_add(&shift)?
        };
        let got = adaln_chain_reference(&x, &AdaLnChain::rms_recip(eps), &scale, &shift)?;
        assert_eq!(bits(&got)?, bits(&ltx)?);
        // Native bf16 LayerNorm (no upcast).
        let native = {
            let mean = x.mean_keepdim(D::Minus1)?;
            let xc = x.broadcast_sub(&mean)?;
            let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
            let n = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
            n.broadcast_mul(&(&scale + 1.0)?)?.broadcast_add(&shift)?
        };
        let got = adaln_chain_reference(&x, &AdaLnChain::layer(eps).native(), &scale, &shift)?;
        assert_eq!(bits(&got)?, bits(&native)?);
        // f32 modulation then a cast (Wan: modulate in f32, cast to the stream dtype).
        let (s32, h32) = (scale.to_dtype(DType::F32)?, shift.to_dtype(DType::F32)?);
        let wan = {
            let xf = x.to_dtype(DType::F32)?;
            let mean = xf.mean_keepdim(D::Minus1)?;
            let xc = xf.broadcast_sub(&mean)?;
            let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
            let n = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
            n.broadcast_mul(&(&s32 + 1.0)?)?
                .broadcast_add(&h32)?
                .to_dtype(DType::BF16)?
        };
        let spec = AdaLnChain::layer(eps)
            .modulate_in(DType::F32)
            .output(DType::BF16);
        let got = adaln_chain_reference(&x, &spec, &s32, &h32)?;
        assert_eq!(bits(&got)?, bits(&wan)?);
        Ok(())
    }

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn adaln_chain_on_cpu_is_the_reference_and_is_recorded() -> Result<()> {
        let _guard = media_fused_policy_guard(Some(true));
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 3, 16), &dev)?;
        let s = Tensor::randn(0f32, 0.2, (1, 1, 16), &dev)?;
        let start = ADALN_PATH.thread_counts();
        let got = adaln_chain(&x, &AdaLnChain::layer(1e-6), &s, &s)?;
        let counts = ADALN_PATH.thread_counts().since(&start);
        let want = adaln_chain_reference(&x, &AdaLnChain::layer(1e-6), &s, &s)?;
        assert_eq!(bits(&got)?, bits(&want)?);
        assert_eq!((counts.fused, counts.reference_total()), (0, 1), "{counts}");
        Ok(())
    }

    #[test]
    fn a_tracked_operand_backpropagates_through_the_reference() -> Result<()> {
        let dev = Device::Cpu;
        let x = Var::new(&[[[1f32, 2.0, 4.0, -1.0]]], &dev)?;
        let s = Tensor::zeros((1, 1, 4), DType::F32, &dev)?;
        let out = adaln_chain(x.as_tensor(), &AdaLnChain::layer(1e-6), &s, &s)?;
        let grads = (out.sqr()?.sum_all()?).backward()?;
        assert!(grads.get(x.as_tensor()).is_some());
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    //! GPU bit identity of the composable-chain AdaLN against [`adaln_chain_reference`] (the
    //! candle op chain on the same device). Declared tolerance: **0 ulp**, every variant and dtype
    //! (the kernel replays the reference kernels' arithmetic; `adaln_chain.cu`). `#[ignore]`d CUDA
    //! compute, registered in `scripts/ci/test_suites.json`.
    use super::*;
    use crate::fused_path::media_fused_policy_guard;
    use candle_core::Device;

    fn values(n: usize, seed: u64, amp: f32, bias: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp + bias
            })
            .collect()
    }

    fn tensor(dev: &Device, dims: &[usize], seed: u64, amp: f32, bias: f32) -> Result<Tensor> {
        Tensor::from_vec(values(dims.iter().product(), seed, amp, bias), dims, dev)
    }

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    fn fused(x: &Tensor, spec: &AdaLnChain<'_>, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        match cuda_impl::fused_adaln_chain(x, spec, scale, Some(shift))? {
            Attempt::Fused(t) => Ok(t),
            Attempt::Reference(r) => panic!("the fused chain declined: {r}"),
        }
    }

    /// Every variant (Layer / Rms / RmsRecip; f32 and native precision; with and without weight
    /// and bias; per-sample and per-token modulation; OnePlus and Direct; norm cast back, f32
    /// modulation, a final cast) over D in {1, 7, 96, 1000, 1024, 3072, 6151} and x in f32 / bf16 /
    /// f16 is bit-identical to the reference chain on the same device.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_adaln_chain_gpu_is_bit_identical_to_the_reference_chain() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut cases = 0usize;
        for &n in &[1usize, 7, 96, 1000, 1024, 3072, 6151] {
            let (b, s) = (2usize, 5usize);
            // An offset mean so the LayerNorm centring matters.
            let x32 = tensor(&dev, &[b, s, n], 11 + n as u64, 3.0, 0.7)?;
            let w32 = tensor(&dev, &[n], 3, 0.5, 1.0)?;
            let b32 = tensor(&dev, &[n], 5, 0.5, 0.0)?;
            for xd in [DType::F32, DType::BF16, DType::F16] {
                let x = x32.to_dtype(xd)?;
                for precision in [ChainPrecision::F32, ChainPrecision::Native] {
                    if precision == ChainPrecision::Native && xd == DType::F16 {
                        continue;
                    }
                    let cdt = if precision == ChainPrecision::F32 {
                        DType::F32
                    } else {
                        xd
                    };
                    let (w, bb) = (w32.to_dtype(cdt)?, b32.to_dtype(cdt)?);
                    for norm in [ChainNorm::Layer, ChainNorm::Rms, ChainNorm::RmsRecip] {
                        for affine in 0..3 {
                            for (md, od) in [(xd, xd), (DType::F32, DType::F32), (DType::F32, xd)] {
                                for per_token in [false, true] {
                                    for direct in [false, true] {
                                        let ms = if per_token { s } else { 1 };
                                        let sc =
                                            tensor(&dev, &[b, ms, n], 7, 0.4, 0.0)?.to_dtype(md)?;
                                        let sh =
                                            tensor(&dev, &[b, ms, n], 9, 0.4, 0.0)?.to_dtype(md)?;
                                        let mut spec =
                                            AdaLnChain::new(norm, 1e-6).modulate_in(md).output(od);
                                        if precision == ChainPrecision::Native {
                                            spec = spec.native();
                                        }
                                        if affine >= 1 {
                                            spec = spec.with_weight(&w);
                                        }
                                        if affine == 2 {
                                            spec = spec.with_bias(&bb);
                                        }
                                        if direct {
                                            spec = spec.direct_scale();
                                        }
                                        let want = adaln_chain_reference(&x, &spec, &sc, &sh)?;
                                        let got = fused(&x, &spec, &sc, &sh)?;
                                        assert_eq!(got.dtype(), want.dtype());
                                        let (g, r) = (bits(&got)?, bits(&want)?);
                                        let bad = g.iter().zip(&r).filter(|(a, b)| a != b).count();
                                        assert_eq!(
                                            bad, 0,
                                            "n={n} x={xd:?} {precision:?} {norm:?} affine={affine} \
                                             m={md:?} o={od:?} per_token={per_token} \
                                             direct={direct}: {bad} of {} differ",
                                            g.len()
                                        );
                                        cases += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("adaln_chain: {cases} variant x shape x dtype cases bit-identical (0 ulp)");
        Ok(())
    }

    /// Production rows, bf16 x, the crates' commonest form (f32 LayerNorm cast back, per-sample
    /// modulation), at D 3072 / 5120 and a 4096-token sequence.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_adaln_chain_gpu_is_bit_identical_at_production_rows() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for (b, s, n) in [
            (1usize, 4096usize, 3072usize),
            (2, 4608, 3072),
            (1, 4096, 5120),
        ] {
            let x = tensor(&dev, &[b, s, n], 1, 4.0, 0.3)?.to_dtype(DType::BF16)?;
            let sc = tensor(&dev, &[b, 1, n], 2, 0.5, 0.0)?.to_dtype(DType::BF16)?;
            let sh = tensor(&dev, &[b, 1, n], 3, 0.5, 0.0)?.to_dtype(DType::BF16)?;
            for spec in [AdaLnChain::layer(1e-6), AdaLnChain::rms_recip(1e-6)] {
                let want = adaln_chain_reference(&x, &spec, &sc, &sh)?;
                let got = fused(&x, &spec, &sc, &sh)?;
                assert_eq!(bits(&got)?, bits(&want)?, "[{b},{s},{n}] {:?}", spec.norm);
            }
        }
        Ok(())
    }

    /// The comparison is sensitive: a dropped `+ 1`, a different eps, the other norm, candle's
    /// fused LayerNorm (another reduction order) and an f32 chain in place of the native bf16 one
    /// each break parity. (A manual kernel mutation — the reduction tree pairing adjacent slots
    /// instead of candle's halving stride — failed `..._at_production_rows`; reverted.)
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_adaln_chain_gpu_parity_catches_mutations() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = tensor(&dev, &[1, 64, 3072], 4, 3.0, 0.5)?.to_dtype(DType::BF16)?;
        let sc = tensor(&dev, &[1, 1, 3072], 5, 0.5, 0.0)?.to_dtype(DType::BF16)?;
        let sh = tensor(&dev, &[1, 1, 3072], 6, 0.5, 0.0)?.to_dtype(DType::BF16)?;
        let differ = |a: &Tensor, b: &Tensor| -> Result<usize> {
            Ok(bits(a)?
                .iter()
                .zip(&bits(b)?)
                .filter(|(p, q)| p != q)
                .count())
        };
        let base = AdaLnChain::layer(1e-6);
        let got = fused(&x, &base, &sc, &sh)?;
        let total = got.elem_count();
        let mutants = [
            ("direct scale (dropped +1)", base.direct_scale()),
            ("eps 1e-5", AdaLnChain::layer(1e-5)),
            ("rms in place of layer", AdaLnChain::rms(1e-6)),
        ];
        for (what, spec) in mutants {
            let m = adaln_chain_reference(&x, &spec, &sc, &sh)?;
            let d = differ(&got, &m)?;
            println!("mutation {what}: {d} of {total} differ");
            assert!(d > 0, "{what} not caught");
        }
        // candle's fused `ops::layer_norm` (S2's chain: warp-shuffle reduction, E[x²] - mean²
        // variance, rsqrt) is the same math in another order: it must not pass for this chain.
        let s2 = crate::adaln::adaln_reference(
            &x,
            &crate::adaln::AdaLn::layer(1e-6).norm_in_f32(),
            &sc,
            &sh,
        )?;
        let d = differ(&got, &s2)?;
        println!("mutation fused ops::layer_norm order: {d} of {total} differ");
        assert!(d > 0);
        // Native vs f32 precision on the same bf16 x: the bf16 sum / approx sqrt / div differ.
        let native = fused(&x, &base.native(), &sc, &sh)?;
        let d = differ(&got, &native)?;
        println!("mutation native vs f32 chain: {d} of {total} differ");
        assert!(d > 0);
        Ok(())
    }

    /// The scale-only forms (`norm(x) * (1 + scale)`, no shift) of both AdaLN kernels — this
    /// composable chain and S2's candle `ops::rms_norm` / `ops::layer_norm` kernel — are
    /// bit-identical to their references (norm then one `broadcast_mul`), f32 / bf16 / f16.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fused_scale_only_adaln_gpu_is_bit_identical() -> Result<()> {
        use crate::adaln::{adaln_scale_only_reference, fused_adaln_opt, AdaLn};
        let dev = Device::new_cuda(0)?;
        let mut cases = 0usize;
        for &n in &[7usize, 96, 2304, 3840] {
            let x32 = tensor(&dev, &[2, 9, n], 21, 3.0, 0.4)?;
            let w32 = tensor(&dev, &[n], 22, 0.5, 1.0)?;
            for xd in [DType::F32, DType::BF16, DType::F16] {
                let x = x32.to_dtype(xd)?;
                for per_token in [false, true] {
                    let sc = tensor(&dev, &[2, if per_token { 9 } else { 1 }, n], 23, 0.4, 0.0)?
                        .to_dtype(xd)?;
                    for spec in [AdaLnChain::rms(1e-5), AdaLnChain::layer(1e-6)] {
                        let want = adaln_chain_scale_only_reference(&x, &spec, &sc)?;
                        let got = match cuda_impl::fused_adaln_chain(&x, &spec, &sc, None)? {
                            Attempt::Fused(t) => t,
                            Attempt::Reference(r) => panic!("chain scale-only declined: {r}"),
                        };
                        assert_eq!(
                            bits(&got)?,
                            bits(&want)?,
                            "chain n={n} {xd:?} {:?}",
                            spec.norm
                        );
                        cases += 1;
                    }
                    let w = w32.to_dtype(xd)?;
                    for spec in [AdaLn::rms(1e-5).with_weight(&w), AdaLn::layer(1e-6)] {
                        let want = adaln_scale_only_reference(&x, &spec, &sc)?;
                        let got = match fused_adaln_opt(&x, &spec, &sc, None)? {
                            Attempt::Fused(t) => t,
                            Attempt::Reference(r) => panic!("ops scale-only declined: {r}"),
                        };
                        assert_eq!(
                            bits(&got)?,
                            bits(&want)?,
                            "ops n={n} {xd:?} {:?}",
                            spec.norm
                        );
                        cases += 1;
                    }
                }
            }
        }
        println!("scale-only adaln: {cases} cases bit-identical (0 ulp)");
        Ok(())
    }

    /// Path telemetry: fused on CUDA, disabled with the switch off (identical output), autograd
    /// for a tracked operand.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn adaln_chain_reports_its_path() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ADALN_PATH.guard(Some(true));
        let x = tensor(&dev, &[1, 8, 96], 1, 1.0, 0.0)?.to_dtype(DType::BF16)?;
        let s = tensor(&dev, &[1, 1, 96], 2, 0.3, 0.0)?.to_dtype(DType::BF16)?;
        let spec = AdaLnChain::layer(1e-6);
        let start = ADALN_PATH.thread_counts();
        let on = adaln_chain(&x, &spec, &s, &s)?;
        let c = ADALN_PATH.thread_counts().since(&start);
        assert_eq!((c.fused, c.reference_total()), (1, 0), "{c}");
        ADALN_PATH.set_enabled(Some(false));
        let start = ADALN_PATH.thread_counts();
        let off = adaln_chain(&x, &spec, &s, &s)?;
        let c = ADALN_PATH.thread_counts().since(&start);
        assert_eq!(c.reference_for("disabled"), 1, "{c}");
        ADALN_PATH.set_enabled(Some(true));
        assert_eq!(bits(&on)?, bits(&off)?);
        let v = candle_core::Var::from_tensor(&x.to_dtype(DType::F32)?)?;
        let s32 = s.to_dtype(DType::F32)?;
        let start = ADALN_PATH.thread_counts();
        adaln_chain(v.as_tensor(), &spec.modulate_in(DType::F32), &s32, &s32)?;
        let c = ADALN_PATH.thread_counts().since(&start);
        assert_eq!(c.reference_for("autograd"), 1, "{c}");
        Ok(())
    }
}
