//! Fused decode primitives (sc-24137, epic sc-24128): RMSNorm(+residual), SwiGLU and
//! QK-norm+RoPE as single CUDA launches, nvrtc-compiled from [`FUSED_DECODE_SRC`] through the
//! [`nvrtc`](crate::nvrtc) compile-once seam.
//!
//! Each primitive is a drop-in for a candle op chain and is **bit-identical** to it on bf16 and
//! f32 (see the contract at the top of `fused_decode.cu`): the op chain stays the reference and
//! the CPU path in the engine that calls these, and the fused kernel is a pure speed change.
//!
//! **Validation before launch (epic E5).** Every kernel declares its shape/dtype constraints in a
//! `check_*` function that builds on every lane (CPU too) and returns a typed [`FusedRefusal`]
//! naming why an input is not served; the launch functions call the same check, so what a CPU
//! test asserts is exactly what the GPU refuses. Unsupported input is never coerced: the caller
//! routes it to the reference chain and reports the refusal.
//!
//! **Toolkit dependency.** bf16 SwiGLU is bit-identical to candle's `usilu_bf16` because the kernel
//! replicates CUDA 12.9's `cuda_bf16.hpp` `hexp` / `__hdiv` (`ex2.approx` and `div.approx` with the
//! 2^126 guard). A toolkit that changes those definitions changes candle's reference, not this
//! kernel, and nothing fails to compile — the GPU parity tests
//! (`cuda_tests::swiglu_matches_reference_on_qwen35_and_edge_shapes` here and `candle-llm`'s
//! `fused_primitives::cuda::fused_on_and_off_are_bit_identical_and_both_visible`) are the guard;
//! re-run them on any CUDA toolkit bump.
//!
//! The six entry points (three kernels × f32/bf16) are resolved once per device into a table, so a
//! leaf launch does one map lookup — no symbol formatting and no `cuModuleGetFunction`.
//!
//! Only the launch functions (`rms_norm`, `rms_norm_residual`, `swiglu`, `rms_norm_rope`) are
//! `cuda`-only.

use std::fmt;

use candle_core::{DType, Tensor};

use crate::nvrtc::{KernelCompileError, KernelSource};

/// The fused decode kernel source; compiled once per device on first use.
pub const FUSED_DECODE_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_fused_decode_v1",
    src: include_str!("fused_decode.cu"),
    // Builtins and `ex2.approx` / `div.approx` PTX only; sm_70 is the oldest this workspace
    // targets, and the bf16 conversions are software so no sm_80 instruction is required.
    cc_floor: (7, 0),
};

/// Longest head (`head_dim`) the fused QK-norm+RoPE kernel serves: its per-row staging buffer is
/// one 1024-float shared array.
pub const FUSED_ROPE_MAX_HEAD_DIM: usize = 1024;

/// Why a fused primitive refuses an input. The reference chain serves it instead; the label is
/// what telemetry records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedRefusal {
    /// The tensor is not on a CUDA device.
    NotCuda,
    /// Only `F32` and `BF16` are served.
    Dtype(DType),
    /// The operands do not share one dtype.
    DtypeMismatch,
    /// The operands' shapes disagree (or the weight is not `[last_dim]`).
    Shape,
    /// The normalized axis is empty.
    EmptyRow,
    /// RoPE: `head_dim` exceeds [`FUSED_ROPE_MAX_HEAD_DIM`].
    HeadDimTooLarge,
    /// RoPE: `rotary_dim` is zero, odd, or larger than `head_dim`.
    RotaryDim,
    /// RoPE: cos/sin are not `[1 | batch, seq, rotary_dim]`.
    CosSinShape,
    /// More rows than one launch grid addresses.
    TooManyRows,
}

impl FusedRefusal {
    /// Stable lower-case label for telemetry.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotCuda => "not_cuda",
            Self::Dtype(_) => "dtype",
            Self::DtypeMismatch => "dtype_mismatch",
            Self::Shape => "shape",
            Self::EmptyRow => "empty_row",
            Self::HeadDimTooLarge => "head_dim_too_large",
            Self::RotaryDim => "rotary_dim",
            Self::CosSinShape => "cos_sin_shape",
            Self::TooManyRows => "too_many_rows",
        }
    }
}

impl fmt::Display for FusedRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "tensor is not on a CUDA device"),
            Self::Dtype(d) => write!(f, "dtype {d:?} is not served (F32 or BF16 only)"),
            Self::DtypeMismatch => write!(f, "operands do not share one dtype"),
            Self::Shape => write!(f, "operand shapes disagree"),
            Self::EmptyRow => write!(f, "normalized axis is empty"),
            Self::HeadDimTooLarge => {
                write!(f, "head_dim exceeds {FUSED_ROPE_MAX_HEAD_DIM}")
            }
            Self::RotaryDim => write!(f, "rotary_dim must be even, > 0 and <= head_dim"),
            Self::CosSinShape => write!(f, "cos/sin must be [1 | batch, seq, rotary_dim]"),
            Self::TooManyRows => write!(f, "too many rows for one launch"),
        }
    }
}

/// A fused primitive's failure: a typed refusal (route to the reference), a cached compile
/// error (the seam's; also route to the reference), or a candle error from the surrounding ops.
#[derive(Debug)]
pub enum FusedError {
    /// Input outside the kernel's declared constraints.
    Refused(FusedRefusal),
    /// The kernel source does not compile / load on this device (cached by the seam).
    Compile(KernelCompileError),
    /// A candle error (allocation, contiguity copy, …).
    Candle(candle_core::Error),
}

impl FusedError {
    /// Stable telemetry label: the refusal's label, the compile error's label, or `candle`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Refused(r) => r.label(),
            Self::Compile(e) => e.label(),
            Self::Candle(_) => "candle",
        }
    }
}

impl fmt::Display for FusedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(r) => write!(f, "fused primitive refused: {r}"),
            Self::Compile(e) => write!(f, "fused primitive unavailable: {e}"),
            Self::Candle(e) => write!(f, "fused primitive failed: {e}"),
        }
    }
}

impl std::error::Error for FusedError {}

impl From<FusedRefusal> for FusedError {
    fn from(r: FusedRefusal) -> Self {
        Self::Refused(r)
    }
}

impl From<KernelCompileError> for FusedError {
    fn from(e: KernelCompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<candle_core::Error> for FusedError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}

/// The validated geometry of one RMSNorm launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormPlan {
    /// Rows normalized (product of the leading dims).
    pub rows: usize,
    /// Length of the normalized (last) axis.
    pub n: usize,
    /// Operand dtype.
    pub dtype: DType,
}

/// The validated geometry of one QK-norm+RoPE launch over `x: [batch, seq, heads, head_dim]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopePlan {
    /// `batch`.
    pub batch: usize,
    /// `seq`.
    pub seq: usize,
    /// `heads`.
    pub heads: usize,
    /// `head_dim`.
    pub head_dim: usize,
    /// `rotary_dim` (cos/sin last dim).
    pub rotary_dim: usize,
    /// Whether cos/sin carry a per-batch leading dim (`batch`) rather than `1`.
    pub cos_batched: bool,
    /// Operand dtype.
    pub dtype: DType,
}

fn served_dtype(dtype: DType) -> Result<(), FusedRefusal> {
    match dtype {
        DType::F32 | DType::BF16 => Ok(()),
        other => Err(FusedRefusal::Dtype(other)),
    }
}

fn rows_fit(rows: usize) -> Result<(), FusedRefusal> {
    if u32::try_from(rows).is_err() {
        return Err(FusedRefusal::TooManyRows);
    }
    Ok(())
}

/// Validate an RMSNorm (+ optional same-shape residual) over the last axis of `x` with `w:
/// [last_dim]`, without touching a device.
pub fn check_rms_norm(
    x: &Tensor,
    residual: Option<&Tensor>,
    w: &Tensor,
) -> Result<RmsNormPlan, FusedRefusal> {
    served_dtype(x.dtype())?;
    if w.dtype() != x.dtype() || residual.is_some_and(|r| r.dtype() != x.dtype()) {
        return Err(FusedRefusal::DtypeMismatch);
    }
    let n = x.dims().last().copied().unwrap_or(0);
    if n == 0 {
        return Err(FusedRefusal::EmptyRow);
    }
    if w.dims() != [n] {
        return Err(FusedRefusal::Shape);
    }
    if residual.is_some_and(|r| r.dims() != x.dims()) {
        return Err(FusedRefusal::Shape);
    }
    let rows = x.elem_count() / n;
    rows_fit(rows)?;
    Ok(RmsNormPlan {
        rows,
        n,
        dtype: x.dtype(),
    })
}

/// Validate a SwiGLU over same-shape `gate` / `up`; returns the element count.
pub fn check_swiglu(gate: &Tensor, up: &Tensor) -> Result<usize, FusedRefusal> {
    served_dtype(gate.dtype())?;
    if up.dtype() != gate.dtype() {
        return Err(FusedRefusal::DtypeMismatch);
    }
    if up.dims() != gate.dims() {
        return Err(FusedRefusal::Shape);
    }
    let n = gate.elem_count();
    if n == 0 {
        return Err(FusedRefusal::EmptyRow);
    }
    if u32::try_from(n).is_err() {
        return Err(FusedRefusal::TooManyRows);
    }
    Ok(n)
}

/// Validate a per-head RMSNorm + RoPE over `x: [batch, seq, heads, head_dim]` with `w:
/// [head_dim]` and `cos`/`sin: [1 | batch, seq, rotary_dim]`.
pub fn check_rms_norm_rope(
    x: &Tensor,
    w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<RopePlan, FusedRefusal> {
    served_dtype(x.dtype())?;
    if [w, cos, sin].iter().any(|t| t.dtype() != x.dtype()) {
        return Err(FusedRefusal::DtypeMismatch);
    }
    let &[batch, seq, heads, head_dim] = x.dims() else {
        return Err(FusedRefusal::Shape);
    };
    if head_dim == 0 || batch == 0 || seq == 0 || heads == 0 {
        return Err(FusedRefusal::EmptyRow);
    }
    if head_dim > FUSED_ROPE_MAX_HEAD_DIM {
        return Err(FusedRefusal::HeadDimTooLarge);
    }
    if w.dims() != [head_dim] {
        return Err(FusedRefusal::Shape);
    }
    let &[cb, cs, rotary_dim] = cos.dims() else {
        return Err(FusedRefusal::CosSinShape);
    };
    if sin.dims() != cos.dims() || cs != seq || (cb != 1 && cb != batch) {
        return Err(FusedRefusal::CosSinShape);
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err(FusedRefusal::RotaryDim);
    }
    rows_fit(batch * seq * heads)?;
    Ok(RopePlan {
        batch,
        seq,
        heads,
        head_dim,
        rotary_dim,
        cos_batched: cb != 1,
        dtype: x.dtype(),
    })
}

/// candle's `fast_sum` block size for a row of `n` elements — the launch shape the fused RMSNorm
/// reproduces so its f32 summation order matches the op chain.
pub fn reduce_block_dim(n: usize) -> u32 {
    usize::min(1024, n).next_power_of_two() as u32
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Shape, Storage};
    use cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    fn drv(e: cudarc::driver::DriverError) -> FusedError {
        FusedError::Candle(candle_core::Error::Cuda(
            format!("fused decode kernel: {e:?}").into(),
        ))
    }

    fn cuda_device(t: &Tensor) -> Result<&CudaDevice, FusedRefusal> {
        match t.device() {
            Device::Cuda(d) => Ok(d),
            _ => Err(FusedRefusal::NotCuda),
        }
    }

    /// The three fused entry points; indexes [`FusedFunctions`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Kernel {
        RmsNormResidual = 0,
        Swiglu = 1,
        RmsNormRope = 2,
    }

    impl Kernel {
        const ALL: [Kernel; 3] = [Kernel::RmsNormResidual, Kernel::Swiglu, Kernel::RmsNormRope];

        pub(super) fn symbol(self, dtype: DType) -> &'static str {
            match (self, dtype) {
                (Kernel::RmsNormResidual, DType::F32) => "rms_norm_residual_f32",
                (Kernel::RmsNormResidual, _) => "rms_norm_residual_bf16",
                (Kernel::Swiglu, DType::F32) => "swiglu_f32",
                (Kernel::Swiglu, _) => "swiglu_bf16",
                (Kernel::RmsNormRope, DType::F32) => "rms_norm_rope_f32",
                (Kernel::RmsNormRope, _) => "rms_norm_rope_bf16",
            }
        }
    }

    /// Every fused entry point of one device's module, resolved once (`cuModuleGetFunction` and
    /// the symbol names stay off the per-leaf launch path).
    pub(super) struct FusedFunctions {
        f32: [CudaFunction; 3],
        bf16: [CudaFunction; 3],
    }

    impl FusedFunctions {
        fn resolve(dev: &CudaDevice) -> Result<Self, KernelCompileError> {
            let module = FUSED_DECODE_SRC.compiled(dev)?;
            let load = |dtype: DType| -> Result<[CudaFunction; 3], KernelCompileError> {
                let [a, b, c] = Kernel::ALL;
                Ok([
                    module.function(a.symbol(dtype))?,
                    module.function(b.symbol(dtype))?,
                    module.function(c.symbol(dtype))?,
                ])
            };
            Ok(Self {
                f32: load(DType::F32)?,
                bf16: load(DType::BF16)?,
            })
        }
    }

    /// Per-ordinal resolved function tables. Only successes are held here (leaked once per device
    /// for the life of the process); a compile failure stays cached in the nvrtc seam and is
    /// re-read from there.
    fn tables() -> &'static RwLock<HashMap<usize, &'static FusedFunctions>> {
        static TABLES: OnceLock<RwLock<HashMap<usize, &'static FusedFunctions>>> = OnceLock::new();
        TABLES.get_or_init(Default::default)
    }

    pub(super) fn functions(dev: &CudaDevice) -> Result<&'static FusedFunctions, FusedError> {
        let ordinal = dev.cuda_stream().context().ordinal();
        if let Some(table) = tables()
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&ordinal)
        {
            return Ok(table);
        }
        let resolved = FusedFunctions::resolve(dev)?;
        let mut tables = tables()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(*tables
            .entry(ordinal)
            .or_insert_with(|| Box::leak(Box::new(resolved))))
    }

    fn function(
        dev: &CudaDevice,
        kernel: Kernel,
        dtype: DType,
    ) -> Result<&'static CudaFunction, FusedError> {
        let table = match dtype {
            DType::F32 | DType::BF16 => functions(dev)?,
            other => return Err(FusedRefusal::Dtype(other).into()),
        };
        let row = if dtype == DType::F32 {
            &table.f32
        } else {
            &table.bf16
        };
        Ok(&row[kernel as usize])
    }

    fn wrap<T: CudaDType>(slice: CudaSlice<T>, dev: &CudaDevice, shape: Shape) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    /// Run `f` with the contiguous CUDA slices of `tensors` (each already `.contiguous()`).
    macro_rules! with_slices {
        ($t:ty, [$($name:ident),*], $body:expr) => {{
            $(
                let ($name, __l) = $name.storage_and_layout();
                let $name = match &*$name {
                    Storage::Cuda(c) => c.as_cuda_slice::<$t>()?.slice(__l.start_offset()..),
                    _ => return Err(FusedRefusal::NotCuda.into()),
                };
            )*
            $body
        }};
    }

    fn rms_norm_launch<T: CudaDType + DeviceRepr>(
        dev: &CudaDevice,
        func: &CudaFunction,
        x: &Tensor,
        residual: Option<&Tensor>,
        w: &Tensor,
        plan: RmsNormPlan,
        eps: f64,
    ) -> Result<(Option<Tensor>, Tensor), FusedError> {
        let total = plan.rows * plan.n;
        let mut y = unsafe { dev.alloc::<T>(total) }?;
        let mut h = match residual {
            Some(_) => Some(unsafe { dev.alloc::<T>(total) }?),
            None => None,
        };
        let cfg = LaunchConfig {
            grid_dim: (plan.rows as u32, 1, 1),
            block_dim: (reduce_block_dim(plan.n), 1, 1),
            shared_mem_bytes: 0,
        };
        let n = plan.n as i32;
        let scale = (1.0f64 / plan.n as f64) as f32;
        let eps = eps as f32;
        let stream = dev.cuda_stream();
        let null = 0u64;
        with_slices!(T, [x, w], {
            let mut b = stream.launch_builder(func);
            b.arg(&x);
            let residual_guard;
            let residual_slice;
            match residual {
                Some(r) => {
                    let (g, l) = r.storage_and_layout();
                    residual_guard = g;
                    residual_slice = match &*residual_guard {
                        Storage::Cuda(c) => c.as_cuda_slice::<T>()?.slice(l.start_offset()..),
                        _ => return Err(FusedRefusal::NotCuda.into()),
                    };
                    b.arg(&residual_slice);
                }
                None => {
                    b.arg(&null);
                }
            }
            b.arg(&w);
            match h.as_mut() {
                Some(h) => {
                    b.arg(h);
                }
                None => {
                    b.arg(&null);
                }
            }
            b.arg(&mut y).arg(&n).arg(&scale).arg(&eps);
            unsafe { b.launch(cfg) }.map_err(drv)?;
        });
        let shape = Shape::from(x.dims());
        Ok((h.map(|h| wrap(h, dev, shape.clone())), wrap(y, dev, shape)))
    }

    /// `rms_norm(x, w, eps)`: `x / sqrt(mean(x², last) + eps) * w`, f32 math, `x`'s dtype out.
    pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor, FusedError> {
        let plan = check_rms_norm(x, None, w)?;
        let dev = cuda_device(x)?;
        let func = function(dev, Kernel::RmsNormResidual, plan.dtype)?;
        let x = x.contiguous()?;
        let w = w.contiguous()?;
        let (_, y) = match plan.dtype {
            DType::F32 => rms_norm_launch::<f32>(dev, func, &x, None, &w, plan, eps)?,
            _ => rms_norm_launch::<half::bf16>(dev, func, &x, None, &w, plan, eps)?,
        };
        Ok(y)
    }

    /// `h = x + residual` (rounded to the dtype) and `rms_norm(h, w, eps)`, one launch; returns
    /// `(h, normed)` — `h` is the residual stream the caller carries forward.
    pub fn rms_norm_residual(
        x: &Tensor,
        residual: &Tensor,
        w: &Tensor,
        eps: f64,
    ) -> Result<(Tensor, Tensor), FusedError> {
        let plan = check_rms_norm(x, Some(residual), w)?;
        let dev = cuda_device(x)?;
        let func = function(dev, Kernel::RmsNormResidual, plan.dtype)?;
        let x = x.contiguous()?;
        let r = residual.contiguous()?;
        let w = w.contiguous()?;
        let (h, y) = match plan.dtype {
            DType::F32 => rms_norm_launch::<f32>(dev, func, &x, Some(&r), &w, plan, eps)?,
            _ => rms_norm_launch::<half::bf16>(dev, func, &x, Some(&r), &w, plan, eps)?,
        };
        Ok((h.expect("residual launch returns h"), y))
    }

    fn swiglu_launch<T: CudaDType + DeviceRepr>(
        dev: &CudaDevice,
        func: &CudaFunction,
        gate: &Tensor,
        up: &Tensor,
        n: usize,
    ) -> Result<Tensor, FusedError> {
        let mut out = unsafe { dev.alloc::<T>(n) }?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = n as u32;
        let stream = dev.cuda_stream();
        with_slices!(T, [gate, up], {
            let mut b = stream.launch_builder(func);
            b.arg(&gate).arg(&up).arg(&mut out).arg(&count);
            unsafe { b.launch(cfg) }.map_err(drv)?;
        });
        Ok(wrap(out, dev, Shape::from(gate.dims())))
    }

    /// `silu(gate) * up`, elementwise, one launch.
    pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor, FusedError> {
        let n = check_swiglu(gate, up)?;
        let dev = cuda_device(gate)?;
        let func = function(dev, Kernel::Swiglu, gate.dtype())?;
        let gate = gate.contiguous()?;
        let up = up.contiguous()?;
        match gate.dtype() {
            DType::F32 => swiglu_launch::<f32>(dev, func, &gate, &up, n),
            _ => swiglu_launch::<half::bf16>(dev, func, &gate, &up, n),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_launch<T: CudaDType + DeviceRepr>(
        dev: &CudaDevice,
        func: &CudaFunction,
        x: &Tensor,
        w: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        plan: RopePlan,
        interleaved: bool,
        eps: f64,
    ) -> Result<Tensor, FusedError> {
        let rows = plan.batch * plan.seq * plan.heads;
        let mut y = unsafe { dev.alloc::<T>(rows * plan.head_dim) }?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (reduce_block_dim(plan.head_dim), 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = plan.head_dim as i32;
        let rd = plan.rotary_dim as i32;
        let heads = plan.heads as i32;
        let seq = plan.seq as i32;
        let cos_batched = plan.cos_batched as i32;
        let interleaved = interleaved as i32;
        let scale = (1.0f64 / plan.head_dim as f64) as f32;
        let eps = eps as f32;
        let stream = dev.cuda_stream();
        with_slices!(T, [x, w, cos, sin], {
            let mut b = stream.launch_builder(func);
            b.arg(&x)
                .arg(&w)
                .arg(&cos)
                .arg(&sin)
                .arg(&mut y)
                .arg(&hd)
                .arg(&rd)
                .arg(&heads)
                .arg(&seq)
                .arg(&cos_batched)
                .arg(&interleaved)
                .arg(&scale)
                .arg(&eps);
            unsafe { b.launch(cfg) }.map_err(drv)?;
        });
        Ok(wrap(y, dev, Shape::from(x.dims())))
    }

    /// Per-head `rms_norm(x, w, eps)` followed by (partial) RoPE over the first `rotary_dim`
    /// dims of `x: [batch, seq, heads, head_dim]`; `interleaved` selects the GPT-J even/odd
    /// pairing instead of NeoX half-split. One launch.
    pub fn rms_norm_rope(
        x: &Tensor,
        w: &Tensor,
        eps: f64,
        cos: &Tensor,
        sin: &Tensor,
        interleaved: bool,
    ) -> Result<Tensor, FusedError> {
        let plan = check_rms_norm_rope(x, w, cos, sin)?;
        let dev = cuda_device(x)?;
        let func = function(dev, Kernel::RmsNormRope, plan.dtype)?;
        let x = x.contiguous()?;
        let w = w.contiguous()?;
        let cos = cos.contiguous()?;
        let sin = sin.contiguous()?;
        match plan.dtype {
            DType::F32 => rope_launch::<f32>(dev, func, &x, &w, &cos, &sin, plan, interleaved, eps),
            _ => rope_launch::<half::bf16>(dev, func, &x, &w, &cos, &sin, plan, interleaved, eps),
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{rms_norm, rms_norm_residual, rms_norm_rope, swiglu};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t(dims: &[usize], dtype: DType) -> Tensor {
        Tensor::zeros(dims, dtype, &Device::Cpu).unwrap()
    }

    #[test]
    fn reduce_block_dim_matches_candle_fast_sum() {
        assert_eq!(reduce_block_dim(5120), 1024);
        assert_eq!(reduce_block_dim(256), 256);
        assert_eq!(reduce_block_dim(100), 128);
        assert_eq!(reduce_block_dim(1), 1);
        assert_eq!(reduce_block_dim(17408), 1024);
    }

    #[test]
    fn rms_norm_checks_declare_their_constraints() {
        let x = t(&[1, 7, 5120], DType::BF16);
        let w = t(&[5120], DType::BF16);
        let plan = check_rms_norm(&x, None, &w).unwrap();
        assert_eq!(
            plan,
            RmsNormPlan {
                rows: 7,
                n: 5120,
                dtype: DType::BF16
            }
        );
        assert_eq!(
            check_rms_norm(&x, Some(&x), &w).unwrap().rows,
            7,
            "same-shape residual is served"
        );
        assert_eq!(
            check_rms_norm(&t(&[1, 5120], DType::F16), None, &t(&[5120], DType::F16)),
            Err(FusedRefusal::Dtype(DType::F16))
        );
        assert_eq!(
            check_rms_norm(&x, None, &t(&[5120], DType::F32)),
            Err(FusedRefusal::DtypeMismatch)
        );
        assert_eq!(
            check_rms_norm(&x, None, &t(&[512], DType::BF16)),
            Err(FusedRefusal::Shape)
        );
        assert_eq!(
            check_rms_norm(&x, Some(&t(&[1, 6, 5120], DType::BF16)), &w),
            Err(FusedRefusal::Shape)
        );
        assert_eq!(
            check_rms_norm(&t(&[2, 0], DType::F32), None, &t(&[0], DType::F32)),
            Err(FusedRefusal::EmptyRow)
        );
        assert_eq!(FusedRefusal::Shape.label(), "shape");
    }

    #[test]
    fn swiglu_checks_declare_their_constraints() {
        let g = t(&[1, 1, 17408], DType::BF16);
        assert_eq!(check_swiglu(&g, &g), Ok(17408));
        assert_eq!(
            check_swiglu(&g, &t(&[1, 1, 17408], DType::F32)),
            Err(FusedRefusal::DtypeMismatch)
        );
        assert_eq!(
            check_swiglu(&g, &t(&[1, 2, 17408], DType::BF16)),
            Err(FusedRefusal::Shape)
        );
        assert_eq!(
            check_swiglu(&t(&[0], DType::F32), &t(&[0], DType::F32)),
            Err(FusedRefusal::EmptyRow)
        );
    }

    #[test]
    fn rope_checks_declare_their_constraints() {
        // qwen35-27B: 24 q heads of 256, rotary_dim 64 (partial_rotary_factor 0.25), cos [1, s, 64].
        let x = t(&[1, 7, 24, 256], DType::BF16);
        let w = t(&[256], DType::BF16);
        let cos = t(&[1, 7, 64], DType::BF16);
        let plan = check_rms_norm_rope(&x, &w, &cos, &cos).unwrap();
        assert_eq!(
            plan,
            RopePlan {
                batch: 1,
                seq: 7,
                heads: 24,
                head_dim: 256,
                rotary_dim: 64,
                cos_batched: false,
                dtype: DType::BF16
            }
        );
        let batched = t(&[2, 7, 64], DType::BF16);
        let x2 = t(&[2, 7, 24, 256], DType::BF16);
        assert!(
            check_rms_norm_rope(&x2, &w, &batched, &batched)
                .unwrap()
                .cos_batched
        );
        assert_eq!(
            check_rms_norm_rope(&x, &w, &batched, &batched),
            Err(FusedRefusal::CosSinShape),
            "cos batch must be 1 or the tensor batch"
        );
        assert_eq!(
            check_rms_norm_rope(&x, &w, &t(&[1, 6, 64], DType::BF16), &cos),
            Err(FusedRefusal::CosSinShape)
        );
        assert_eq!(
            check_rms_norm_rope(
                &x,
                &w,
                &t(&[1, 7, 63], DType::BF16),
                &t(&[1, 7, 63], DType::BF16)
            ),
            Err(FusedRefusal::RotaryDim)
        );
        assert_eq!(
            check_rms_norm_rope(
                &x,
                &w,
                &t(&[1, 7, 512], DType::BF16),
                &t(&[1, 7, 512], DType::BF16)
            ),
            Err(FusedRefusal::RotaryDim)
        );
        assert_eq!(
            check_rms_norm_rope(&t(&[7, 24, 256], DType::BF16), &w, &cos, &cos),
            Err(FusedRefusal::Shape),
            "rank 4 only"
        );
        assert_eq!(
            check_rms_norm_rope(&x, &t(&[128], DType::BF16), &cos, &cos),
            Err(FusedRefusal::Shape)
        );
        assert_eq!(
            check_rms_norm_rope(&x, &w, &t(&[1, 7, 64], DType::F32), &cos),
            Err(FusedRefusal::DtypeMismatch)
        );
        let wide = t(&[1, 1, 1, 2048], DType::F32);
        assert_eq!(
            check_rms_norm_rope(
                &wide,
                &t(&[2048], DType::F32),
                &t(&[1, 1, 64], DType::F32),
                &t(&[1, 1, 64], DType::F32)
            ),
            Err(FusedRefusal::HeadDimTooLarge)
        );
    }
}

/// GPU parity tests: each fused kernel against the candle op chain it replaces, on the
/// qwen3_5-27B decode shapes and the edge shapes the story names. The op chains here are copies of
/// `candle-llm`'s `primitives::nn::rms_norm` / `primitives::rope::apply_rope` / `silu · up` — the
/// engine's own tests compare its entry points fused-vs-reference; these pin the kernels.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::{Device, D};

    /// Declared tolerances. f32: the story's `<= 1e-5` relative. bf16: **bit-identical** — the
    /// kernels reproduce candle's reduction order and per-op bf16 rounding (see `fused_decode.cu`),
    /// so the test asserts zero bf16 ulps of difference rather than a loose band.
    const F32_MAX_REL: f64 = 1e-5;
    const BF16_MAX_ULP: u32 = 0;

    fn device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    /// The entry points are resolved once per device: a second lookup (and a second `CudaDevice`
    /// handle for the same ordinal) returns the same table, and each slot is the symbol it names.
    #[test]
    fn entry_points_are_resolved_once_per_device() {
        use super::cuda_impl::{functions, Kernel};
        let (Some(Device::Cuda(dev)), Some(Device::Cuda(again))) = (device(), device()) else {
            return;
        };
        let first = functions(&dev).expect("resolves");
        assert!(std::ptr::eq(first, functions(&dev).unwrap()));
        assert!(std::ptr::eq(first, functions(&again).unwrap()));
        assert_eq!(FUSED_DECODE_SRC.compile_attempts(&dev), 1);
        assert_eq!(
            Kernel::RmsNormResidual.symbol(DType::F32),
            "rms_norm_residual_f32"
        );
        assert_eq!(Kernel::Swiglu.symbol(DType::BF16), "swiglu_bf16");
        assert_eq!(
            Kernel::RmsNormRope.symbol(DType::BF16),
            "rms_norm_rope_bf16"
        );
    }

    fn ramp(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn tensor(dims: &[usize], seed: u64, scale: f32, dtype: DType, dev: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        Tensor::from_vec(ramp(n, seed, scale), dims, dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    // ---- the reference op chains (copies of candle-llm's primitives) ----

    fn ref_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
        let orig = x.dtype();
        let xf = x.to_dtype(DType::F32).unwrap();
        let last = xf.rank() - 1;
        let mean = xf.sqr().unwrap().mean_keepdim(last).unwrap();
        let denom = (mean + eps).unwrap().sqrt().unwrap();
        let normed = xf.broadcast_div(&denom).unwrap();
        let wf = weight.to_dtype(DType::F32).unwrap();
        normed.broadcast_mul(&wf).unwrap().to_dtype(orig).unwrap()
    }

    fn ref_apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Tensor {
        let head_dim = x.dim(3).unwrap();
        let rd = cos.dim(D::Minus1).unwrap();
        let cos = cos.unsqueeze(2).unwrap();
        let sin = sin.unsqueeze(2).unwrap();
        let x_rot = x.narrow(3, 0, rd).unwrap();
        let rotated = if interleaved {
            let mut pair_shape = x_rot.dims().to_vec();
            let last = pair_shape.len() - 1;
            pair_shape[last] = rd / 2;
            pair_shape.push(2);
            let xr = x_rot.reshape(pair_shape).unwrap();
            let ax = xr.rank() - 1;
            let even = xr.narrow(ax, 0, 1).unwrap();
            let odd = xr.narrow(ax, 1, 1).unwrap();
            let rot = Tensor::cat(&[&odd.neg().unwrap(), &even], ax)
                .unwrap()
                .reshape(x_rot.shape())
                .unwrap();
            (x_rot.broadcast_mul(&cos).unwrap() + rot.broadcast_mul(&sin).unwrap()).unwrap()
        } else {
            let half = rd / 2;
            let x1 = x_rot.narrow(3, 0, half).unwrap();
            let x2 = x_rot.narrow(3, half, half).unwrap();
            let rot = Tensor::cat(&[&x2.neg().unwrap(), &x1], 3).unwrap();
            (x_rot.broadcast_mul(&cos).unwrap() + rot.broadcast_mul(&sin).unwrap()).unwrap()
        };
        if rd < head_dim {
            let x_pass = x.narrow(3, rd, head_dim - rd).unwrap();
            Tensor::cat(&[&rotated, &x_pass], 3).unwrap()
        } else {
            rotated
        }
    }

    fn ref_swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
        candle_nn::ops::silu(gate)
            .unwrap()
            .broadcast_mul(up)
            .unwrap()
    }

    // ---- comparison ----

    /// f32: max relative difference against `F32_MAX_REL`; bf16: max ulp distance against
    /// `BF16_MAX_ULP`. Both report how many elements differ at all.
    // `BF16_MAX_ULP` is 0 (bit-identical), which makes `<=` look absurd to clippy; it stays a
    // `<=` against the named tolerance so loosening the declaration is a one-line change.
    #[allow(clippy::absurd_extreme_comparisons)]
    fn compare(what: &str, got: &Tensor, want: &Tensor) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape");
        assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
        match got.dtype() {
            DType::F32 => {
                let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let mut max_rel = 0f64;
                let mut differing = 0usize;
                for (a, b) in g.iter().zip(&w) {
                    if a.to_bits() != b.to_bits() {
                        differing += 1;
                    }
                    let rel = (*a as f64 - *b as f64).abs()
                        / (a.abs() as f64).max(b.abs() as f64).max(1e-6);
                    max_rel = max_rel.max(rel);
                }
                assert!(
                    max_rel <= F32_MAX_REL,
                    "{what}: f32 max rel {max_rel:e} > {F32_MAX_REL:e} ({differing}/{} differ)",
                    g.len()
                );
            }
            DType::BF16 => {
                let g = got.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap();
                let w = want.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap();
                let mut max_ulp = 0u32;
                let mut differing = 0usize;
                for (a, b) in g.iter().zip(&w) {
                    let (ab, bb) = (a.to_bits(), b.to_bits());
                    if ab != bb {
                        differing += 1;
                        // Same-sign bf16 bit patterns are monotone in magnitude; a sign flip is
                        // counted as a large distance.
                        let ulp = if (ab ^ bb) & 0x8000 == 0 {
                            u32::from((ab & 0x7fff).abs_diff(bb & 0x7fff))
                        } else {
                            u32::from(ab & 0x7fff) + u32::from(bb & 0x7fff)
                        };
                        max_ulp = max_ulp.max(ulp);
                    }
                }
                assert!(
                    max_ulp <= BF16_MAX_ULP,
                    "{what}: bf16 max {max_ulp} ulp > {BF16_MAX_ULP} ({differing}/{} differ)",
                    g.len()
                );
            }
            other => panic!("{what}: unexpected dtype {other:?}"),
        }
    }

    const EPS: f64 = 1e-6;

    /// (rows..., n) shapes: the 27B hidden (5120) at decode (s=1) and prefill (s=7, s=33), the
    /// attention-head width, and hidden sizes that are not a multiple of any block size.
    const RMS_SHAPES: &[&[usize]] = &[
        &[1, 1, 5120],
        &[1, 7, 5120],
        &[2, 33, 5120],
        &[1, 1, 24, 256],
        &[1, 1, 100],
        &[3, 5, 1000],
        &[1, 1, 17],
        &[4, 1, 1],
        &[1, 3, 2049],
    ];

    #[test]
    fn rms_norm_matches_reference_on_qwen35_and_edge_shapes() {
        let Some(dev) = device() else { return };
        for dtype in [DType::F32, DType::BF16] {
            for (i, dims) in RMS_SHAPES.iter().enumerate() {
                let n = *dims.last().unwrap();
                let x = tensor(dims, 11 + i as u64, 3.0, dtype, &dev);
                let w = tensor(&[n], 97 + i as u64, 1.5, dtype, &dev);
                let got = rms_norm(&x, &w, EPS).unwrap();
                compare(
                    &format!("rms_norm {dtype:?} {dims:?}"),
                    &got,
                    &ref_rms_norm(&x, &w, EPS),
                );
            }
        }
    }

    #[test]
    fn rms_norm_residual_matches_reference_add_then_norm() {
        let Some(dev) = device() else { return };
        for dtype in [DType::F32, DType::BF16] {
            for (i, dims) in RMS_SHAPES.iter().enumerate() {
                let n = *dims.last().unwrap();
                let x = tensor(dims, 211 + i as u64, 3.0, dtype, &dev);
                let r = tensor(dims, 311 + i as u64, 3.0, dtype, &dev);
                let w = tensor(&[n], 411 + i as u64, 1.5, dtype, &dev);
                let (h, y) = rms_norm_residual(&x, &r, &w, EPS).unwrap();
                let h_ref = x.broadcast_add(&r).unwrap();
                compare(&format!("residual h {dtype:?} {dims:?}"), &h, &h_ref);
                compare(
                    &format!("residual norm {dtype:?} {dims:?}"),
                    &y,
                    &ref_rms_norm(&h_ref, &w, EPS),
                );
            }
        }
    }

    #[test]
    fn rms_norm_serves_a_strided_view_by_making_it_contiguous() {
        let Some(dev) = device() else { return };
        let wide = tensor(&[1, 4, 2, 256], 5, 2.0, DType::BF16, &dev);
        // A head-dim narrow (start offset + non-unit outer stride), as the qwen35 q/gate split makes.
        let x = wide.narrow(3, 0, 128).unwrap();
        let w = tensor(&[128], 6, 1.0, DType::BF16, &dev);
        let got = rms_norm(&x, &w, EPS).unwrap();
        compare("strided rms_norm", &got, &ref_rms_norm(&x, &w, EPS));
    }

    #[test]
    fn swiglu_matches_reference_on_qwen35_and_edge_shapes() {
        let Some(dev) = device() else { return };
        let shapes: &[&[usize]] = &[
            &[1, 1, 17408],
            &[1, 7, 17408],
            &[2, 33, 1000],
            &[1, 1, 17],
            &[5, 3, 1],
        ];
        for dtype in [DType::F32, DType::BF16] {
            for (i, dims) in shapes.iter().enumerate() {
                // Wide range so silu's saturating tails and the near-zero region are both hit.
                let gate = tensor(dims, 21 + i as u64, 12.0, dtype, &dev);
                let up = tensor(dims, 31 + i as u64, 4.0, dtype, &dev);
                let got = swiglu(&gate, &up).unwrap();
                compare(
                    &format!("swiglu {dtype:?} {dims:?}"),
                    &got,
                    &ref_swiglu(&gate, &up),
                );
            }
        }
    }

    /// (batch, seq, heads, head_dim, rotary_dim, cos batched?) — the 27B q (24×256, rd 64) and
    /// k (4×256) heads at decode and prefill, a full-rotary head, an odd head width, and a
    /// batched cos table.
    const ROPE_SHAPES: &[(usize, usize, usize, usize, usize, bool)] = &[
        (1, 1, 24, 256, 64, false),
        (1, 1, 4, 256, 64, false),
        (1, 7, 24, 256, 64, false),
        (2, 33, 4, 256, 64, true),
        (1, 5, 3, 128, 128, false),
        (1, 3, 2, 96, 24, false),
        (2, 2, 2, 10, 4, true),
        (1, 1, 1, 1024, 1024, false),
    ];

    fn cos_sin(
        batch: usize,
        seq: usize,
        rd: usize,
        batched: bool,
        seed: u64,
        dtype: DType,
        dev: &Device,
    ) -> (Tensor, Tensor) {
        let cb = if batched { batch } else { 1 };
        let angles = tensor(&[cb, seq, rd], seed, 6.0, DType::F32, dev);
        (
            angles.cos().unwrap().to_dtype(dtype).unwrap(),
            angles.sin().unwrap().to_dtype(dtype).unwrap(),
        )
    }

    #[test]
    fn rms_norm_rope_matches_norm_then_rope_neox_and_interleaved() {
        let Some(dev) = device() else { return };
        for dtype in [DType::F32, DType::BF16] {
            for (i, &(b, s, h, hd, rd, batched)) in ROPE_SHAPES.iter().enumerate() {
                let x = tensor(&[b, s, h, hd], 41 + i as u64, 3.0, dtype, &dev);
                let w = tensor(&[hd], 51 + i as u64, 1.5, dtype, &dev);
                let (cos, sin) = cos_sin(b, s, rd, batched, 61 + i as u64, dtype, &dev);
                // qwen3_5 uses the NeoX half-split pairing (`apply_rope(.., false)`); the GPT-J
                // interleaved variant is served too and checked on the same shapes.
                for interleaved in [false, true] {
                    let got = rms_norm_rope(&x, &w, EPS, &cos, &sin, interleaved).unwrap();
                    let want = ref_apply_rope(&ref_rms_norm(&x, &w, EPS), &cos, &sin, interleaved);
                    compare(
                        &format!(
                            "rms_norm_rope {dtype:?} b{b} s{s} h{h} hd{hd} rd{rd} \
                             interleaved={interleaved}"
                        ),
                        &got,
                        &want,
                    );
                }
            }
        }
    }

    #[test]
    fn refusals_are_typed_and_leave_no_launch_behind() {
        let Some(dev) = device() else { return };
        let x = tensor(&[1, 2, 64], 1, 1.0, DType::F16, &dev);
        let w = tensor(&[64], 2, 1.0, DType::F16, &dev);
        match rms_norm(&x, &w, EPS) {
            Err(FusedError::Refused(FusedRefusal::Dtype(DType::F16))) => {}
            other => panic!("expected an F16 refusal, got {other:?}"),
        }
        let x = tensor(&[1, 2, 64], 1, 1.0, DType::F32, &dev);
        let w = tensor(&[32], 2, 1.0, DType::F32, &dev);
        match rms_norm(&x, &w, EPS) {
            Err(FusedError::Refused(FusedRefusal::Shape)) => {}
            other => panic!("expected a shape refusal, got {other:?}"),
        }
        let cpu = Tensor::zeros((1, 2, 64), DType::F32, &Device::Cpu).unwrap();
        let wc = Tensor::zeros(64, DType::F32, &Device::Cpu).unwrap();
        match rms_norm(&cpu, &wc, EPS) {
            Err(FusedError::Refused(FusedRefusal::NotCuda)) => {}
            other => panic!("expected a not-cuda refusal, got {other:?}"),
        }
    }
}
