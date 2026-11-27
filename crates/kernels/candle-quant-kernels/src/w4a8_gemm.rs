//! The fused **W4A8 GEMM** (SW-6134, feature SW-F287): packed int4 weight codes decoded to int8 in
//! shared memory and multiplied on the int8 tensor cores (`mma.sync m16n8k32 s8`), with the W4A8
//! epilogue folded into the same launch.
//!
//! The reference it must equal is the existing two-step path in
//! `candle_layers::w4a8_int8::W4A8Linear`: the device decode of a bounded output-row tile
//! (`CublasLt::dequant_w4a8_packed_tile`, `w4a8_dequant.cu`) followed by a cuBLASLt INT8 IGEMM and
//! the eager F32 epilogue `((acc · xs) · s_channel) + bias`. Integer accumulation is exact, the
//! weight decode is the same helper, and the epilogue repeats the same unfused F32 operations, so
//! the fused output is **bit-identical** to the reference on every shape both accept — it is an
//! exact kernel, not an additional approximation (the W4A8 tier itself is the lossy, opt-in part).
//!
//! # Constraints (validated before launch by [`check_w4a8_gemm`])
//!
//! * compute capability `>=` [`W4A8_GEMM_COMPUTE_CAP_FLOOR`] (`sm_80`: `mma.sync` int8 tensor
//!   cores);
//! * `K` a multiple of [`W4A8_GEMM_TILE_K`] (every W4A8 projection is: the format requires
//!   `K % 256 == 0`) and `K <=` [`W4A8_GEMM_MAX_K`] (exact `i32` accumulation);
//! * nonempty `M`/`N`, `M·K` and `M·N` within `i32::MAX` elements (epic E6), and a launch grid
//!   inside the CUDA limits;
//! * output dtype `F32`, `BF16` or `F16`.
//!
//! A refusal is typed ([`W4A8GemmRefusal`], with a stable telemetry [`label`](W4A8GemmRefusal::label))
//! and the caller runs the reference visibly; a compile failure is cached per device by the nvrtc
//! seam and reported the same way.
//!
//! The same module carries the fused **activation prep** (`w4a8_quantize_fht256`,
//! [`check_w4a8_quantize`]): the ConvRot FHT (block 256) and Kitchen's row-wise int8 quantizer in
//! one launch, byte-identical to the eager `convrot::convrot_fht_f32` + `quantize_kitchen_int8`
//! reference it replaces (the eager FHT is a chain of candle ops over F32 intermediates and costs
//! several times the GEMM at DiT shapes).
//!
//! The W4A8 **tier** floor ([`W4A8_COMPUTE_CAP_FLOOR`], [`w4a8_tier_refusal`]) lives here too, so a
//! plan-time tier selector and the loader share one definition.

use crate::nvrtc::{KernelCompileError, KernelSource};
use candle_core::DType;
use std::fmt;

/// The fused GEMM's compute-capability floor: Ampere `sm_80`, the first architecture with the
/// `mma.sync.m16n8k32` int8 tensor-core instruction.
pub const W4A8_GEMM_COMPUTE_CAP_FLOOR: (i32, i32) = (8, 0);

/// The W4A8 tier's compute-capability floor (`sm_80`, comfy-kitchen's
/// `AsymW4A8Int8Layout.MIN_SM_VERSION`). Below it a W4A8 checkpoint is refused, never downgraded.
pub const W4A8_COMPUTE_CAP_FLOOR: (i32, i32) = (8, 0);

/// Output rows (`M`) per thread block.
pub const W4A8_GEMM_TILE_M: usize = 128;
/// Output columns (`N`) per thread block.
pub const W4A8_GEMM_TILE_N: usize = 128;
/// Contraction step per shared-memory stage; `K` must be a multiple.
pub const W4A8_GEMM_TILE_K: usize = 64;
/// Threads per block.
pub const W4A8_GEMM_THREADS: u32 = 256;
/// Largest contraction width whose int32 accumulation is exact: `K · 127 · 128 < 2³¹`.
pub const W4A8_GEMM_MAX_K: usize = 131_072;
/// The GEMM entry point.
pub const W4A8_GEMM_FUNCTION: &str = "w4a8_int4_gemm";
/// The fused activation prep entry point (ConvRot FHT + row-wise int8 quantizer).
pub const W4A8_QUANTIZE_FUNCTION: &str = "w4a8_quantize_fht256";
/// The rotation block the fused activation prep implements (the frozen W4A8 dialect's).
pub const W4A8_QUANTIZE_GROUP: usize = 256;

/// The fused W4A8 GEMM source, compiled once per device through the [`crate::nvrtc`] seam.
pub const W4A8_GEMM_SRC: KernelSource = KernelSource {
    name: "w4a8_int4_int8mma_gemm_v1",
    src: include_str!("w4a8_gemm.cu"),
    cc_floor: W4A8_GEMM_COMPUTE_CAP_FLOOR,
};

/// Why the fused kernel declined an input. The caller runs the dequant + IGEMM reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4A8GemmRefusal {
    /// The device is below [`W4A8_GEMM_COMPUTE_CAP_FLOOR`].
    ComputeFloor { found: (i32, i32) },
    /// `M`, `N` or `K` is zero.
    Empty,
    /// `K` is not a multiple of [`W4A8_GEMM_TILE_K`].
    KTile { k: usize },
    /// `K` exceeds [`W4A8_GEMM_MAX_K`] (the int32 accumulator could wrap).
    KOverflow { k: usize },
    /// An operand or the output would exceed `i32::MAX` elements, or the grid the CUDA limits.
    TooLarge,
    /// The output dtype is not `F32`, `BF16` or `F16`.
    Dtype(DType),
    /// An operand is not on a CUDA device, or the operands' devices/dtypes/shapes disagree.
    Operands,
}

impl W4A8GemmRefusal {
    /// Stable lower-case label for telemetry.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ComputeFloor { .. } => "compute_floor",
            Self::Empty => "empty",
            Self::KTile { .. } => "k_tile",
            Self::KOverflow { .. } => "k_overflow",
            Self::TooLarge => "too_large",
            Self::Dtype(_) => "dtype",
            Self::Operands => "operands",
        }
    }
}

impl fmt::Display for W4A8GemmRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComputeFloor { found } => write!(
                f,
                "fused W4A8 GEMM needs sm_{}{} (device is sm_{}{})",
                W4A8_GEMM_COMPUTE_CAP_FLOOR.0, W4A8_GEMM_COMPUTE_CAP_FLOOR.1, found.0, found.1
            ),
            Self::Empty => write!(f, "fused W4A8 GEMM: empty M, N or K"),
            Self::KTile { k } => write!(
                f,
                "fused W4A8 GEMM: K {k} is not a multiple of {W4A8_GEMM_TILE_K}"
            ),
            Self::KOverflow { k } => write!(
                f,
                "fused W4A8 GEMM: K {k} exceeds the exact int32 bound {W4A8_GEMM_MAX_K}"
            ),
            Self::TooLarge => write!(f, "fused W4A8 GEMM: shape exceeds the i32 / grid limits"),
            Self::Dtype(d) => write!(f, "fused W4A8 GEMM: unsupported output dtype {d:?}"),
            Self::Operands => write!(f, "fused W4A8 GEMM: operand device/dtype/shape mismatch"),
        }
    }
}

/// A fused-GEMM outcome other than success.
#[derive(Debug)]
pub enum W4A8GemmError {
    /// Input outside the kernel's declared constraints.
    Refused(W4A8GemmRefusal),
    /// The source does not compile / load on this device (cached by the seam).
    Compile(KernelCompileError),
    /// A candle / driver failure (allocation, launch).
    Candle(candle_core::Error),
}

impl W4A8GemmError {
    /// Stable telemetry label: the refusal's, the compile error's, or `candle`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Refused(r) => r.label(),
            Self::Compile(e) => e.label(),
            Self::Candle(_) => "candle",
        }
    }
}

impl fmt::Display for W4A8GemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(r) => write!(f, "fused W4A8 GEMM refused: {r}"),
            Self::Compile(e) => write!(f, "fused W4A8 GEMM unavailable: {e}"),
            Self::Candle(e) => write!(f, "fused W4A8 GEMM failed: {e}"),
        }
    }
}

impl std::error::Error for W4A8GemmError {}

impl From<W4A8GemmRefusal> for W4A8GemmError {
    fn from(r: W4A8GemmRefusal) -> Self {
        Self::Refused(r)
    }
}
impl From<KernelCompileError> for W4A8GemmError {
    fn from(e: KernelCompileError) -> Self {
        Self::Compile(e)
    }
}
impl From<candle_core::Error> for W4A8GemmError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}

/// The validated launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4A8GemmPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// `(ceil(N / TILE_N), ceil(M / TILE_M))`.
    pub grid: (u32, u32),
    /// 0 = F32, 1 = BF16, 2 = F16 (the kernel's `out_dtype`).
    pub out_code: i32,
}

/// Pure (CPU-testable) admission of one fused launch: `cap` is the device's compute capability,
/// `[m, k] x [n, k]ᵀ -> [m, n]` in `out`.
pub fn check_w4a8_gemm(
    cap: (i32, i32),
    m: usize,
    n: usize,
    k: usize,
    out: DType,
) -> Result<W4A8GemmPlan, W4A8GemmRefusal> {
    if cap < W4A8_GEMM_COMPUTE_CAP_FLOOR {
        return Err(W4A8GemmRefusal::ComputeFloor { found: cap });
    }
    let out_code = match out {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => return Err(W4A8GemmRefusal::Dtype(other)),
    };
    if m == 0 || n == 0 || k == 0 {
        return Err(W4A8GemmRefusal::Empty);
    }
    if !k.is_multiple_of(W4A8_GEMM_TILE_K) {
        return Err(W4A8GemmRefusal::KTile { k });
    }
    if k > W4A8_GEMM_MAX_K {
        return Err(W4A8GemmRefusal::KOverflow { k });
    }
    let max = i32::MAX as usize;
    let fits = |a: usize, b: usize| a.checked_mul(b).is_some_and(|e| e <= max);
    let grid_x = n.div_ceil(W4A8_GEMM_TILE_N);
    let grid_y = m.div_ceil(W4A8_GEMM_TILE_M);
    if !fits(m, k) || !fits(m, n) || !fits(n, k) || grid_x > i32::MAX as usize || grid_y > 65_535 {
        return Err(W4A8GemmRefusal::TooLarge);
    }
    Ok(W4A8GemmPlan {
        m,
        n,
        k,
        grid: (grid_x as u32, grid_y as u32),
        out_code,
    })
}

/// Pure admission of the fused activation prep for `[m, k]` activations of `dtype`: returns the
/// kernel's dtype code (0 = F32, 1 = BF16, 2 = F16).
pub fn check_w4a8_quantize(
    cap: (i32, i32),
    m: usize,
    k: usize,
    dtype: DType,
) -> Result<i32, W4A8GemmRefusal> {
    if cap < W4A8_GEMM_COMPUTE_CAP_FLOOR {
        return Err(W4A8GemmRefusal::ComputeFloor { found: cap });
    }
    let code = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => return Err(W4A8GemmRefusal::Dtype(other)),
    };
    if m == 0 || k == 0 {
        return Err(W4A8GemmRefusal::Empty);
    }
    if !k.is_multiple_of(W4A8_QUANTIZE_GROUP) {
        return Err(W4A8GemmRefusal::KTile { k });
    }
    if m.checked_mul(k).is_none_or(|e| e > i32::MAX as usize) || m > i32::MAX as usize {
        return Err(W4A8GemmRefusal::TooLarge);
    }
    Ok(code)
}

/// A W4A8 tier refusal: the device cannot execute W4A8 at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4A8TierRefusal {
    pub found: (i32, i32),
}

impl fmt::Display for W4A8TierRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "W4A8 requires compute capability sm_{}{} or newer (device is sm_{}{})",
            W4A8_COMPUTE_CAP_FLOOR.0, W4A8_COMPUTE_CAP_FLOOR.1, self.found.0, self.found.1
        )
    }
}

impl std::error::Error for W4A8TierRefusal {}

/// Plan-time W4A8 tier admission for a CUDA device of capability `cap`: `None` when the tier may
/// be offered, the typed refusal otherwise.
pub fn w4a8_tier_refusal(cap: (i32, i32)) -> Option<W4A8TierRefusal> {
    (cap < W4A8_COMPUTE_CAP_FLOOR).then_some(W4A8TierRefusal { found: cap })
}

/// Operands of one fused launch, all resident on one CUDA device.
#[cfg(feature = "cuda")]
pub struct W4A8GemmOperands<'a> {
    /// `U8` int8 activation codes, `m · k` contiguous bytes (`[m, k]` or flat).
    pub qx: &'a candle_core::Tensor,
    /// `F32 [m, 1]` (or `[m]`) activation row scales.
    pub xs: &'a candle_core::Tensor,
    /// `U8 [n, k/2]` packed int4 codes.
    pub packed: &'a candle_core::Tensor,
    /// `U8 [n, k/16]` raw E4M3FN relative scales.
    pub s_rel: &'a candle_core::Tensor,
    /// `F32 [16]` codebook.
    pub codebook: &'a candle_core::Tensor,
    /// `F32 [n]` per-output-channel scales.
    pub s_channel: &'a candle_core::Tensor,
    /// `F32 [n]` bias.
    pub bias: Option<&'a candle_core::Tensor>,
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{w4a8_gemm, w4a8_quantize_activation};

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Storage, Tensor};
    use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    /// The module's two entry points, resolved once per device ordinal and shared by every layer
    /// (E8). Only successes are held here; a compile failure stays cached in the nvrtc seam.
    struct Functions {
        gemm: CudaFunction,
        quantize: CudaFunction,
    }

    fn functions(dev: &CudaDevice) -> Result<&'static Functions, W4A8GemmError> {
        static TABLE: OnceLock<RwLock<HashMap<usize, &'static Functions>>> = OnceLock::new();
        let table = TABLE.get_or_init(Default::default);
        let ordinal = dev.cuda_stream().context().ordinal();
        if let Some(f) = table
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&ordinal)
        {
            return Ok(f);
        }
        let module = W4A8_GEMM_SRC.compiled(dev)?;
        let f = Functions {
            gemm: module.function(W4A8_GEMM_FUNCTION)?,
            quantize: module.function(W4A8_QUANTIZE_FUNCTION)?,
        };
        let mut w = table.write().unwrap_or_else(|p| p.into_inner());
        Ok(*w.entry(ordinal).or_insert_with(|| Box::leak(Box::new(f))))
    }

    fn device_cap(dev: &CudaDevice) -> Result<(i32, i32), W4A8GemmError> {
        crate::nvrtc::device_compute_cap(dev).map_err(|message| {
            W4A8GemmError::Compile(KernelCompileError::Device {
                name: W4A8_GEMM_SRC.name,
                message,
            })
        })
    }

    fn drv(e: cudarc::driver::DriverError) -> W4A8GemmError {
        W4A8GemmError::Candle(candle_core::Error::Cuda(
            format!("fused W4A8 kernel: {e:?}").into(),
        ))
    }

    /// The fused activation prep: ConvRot FHT (block 256) + Kitchen's row-wise int8 quantizer in
    /// one launch, byte-identical to `convrot::convrot_fht_f32` followed by
    /// `CublasLt::quantize_kitchen_int8`. `x` is `[m, k]` F32/BF16/F16 on CUDA (copied to
    /// contiguous if it is not); returns `U8 [m, k]` codes and `F32 [m, 1]` row scales.
    pub fn w4a8_quantize_activation(x: &Tensor) -> Result<(Tensor, Tensor), W4A8GemmError> {
        let Device::Cuda(dev) = x.device() else {
            return Err(W4A8GemmRefusal::Operands.into());
        };
        let (m, k) = x.dims2().map_err(|_| W4A8GemmRefusal::Operands)?;
        let code = check_w4a8_quantize(device_cap(dev)?, m, k, x.dtype())?;
        let f = functions(dev)?;
        let x = x.contiguous()?;
        let mut q = unsafe { dev.alloc::<u8>(m * k) }?;
        let mut scales = unsafe { dev.alloc::<f32>(m) }?;
        let ki = k as i32;
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = dev.cuda_stream();
        let (g, l) = x.storage_and_layout();
        macro_rules! launch {
            ($ty:ty) => {{
                let xv = view::<$ty>(&g, l)?;
                let mut b = stream.launch_builder(&f.quantize);
                b.arg(&xv).arg(&mut q).arg(&mut scales).arg(&ki).arg(&code);
                unsafe { b.launch(cfg) }.map_err(drv)?;
            }};
        }
        match x.dtype() {
            DType::F32 => launch!(f32),
            DType::BF16 => launch!(half::bf16),
            _ => launch!(half::f16),
        }
        drop(g);
        let q = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(q, dev.clone())),
            (m, k),
            BackpropOp::none(),
            false,
        );
        let scales = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(scales, dev.clone())),
            (m, 1),
            BackpropOp::none(),
            false,
        );
        Ok((q, scales))
    }

    fn view<'a, T: candle_core::cuda_backend::CudaDType>(
        storage: &'a Storage,
        layout: &candle_core::Layout,
    ) -> Result<cudarc::driver::CudaView<'a, T>, W4A8GemmError> {
        let Storage::Cuda(c) = storage else {
            return Err(W4A8GemmRefusal::Operands.into());
        };
        Ok(c.as_cuda_slice::<T>()?.slice(layout.start_offset()..))
    }

    fn operand_shape(t: &Tensor, dtype: DType, elems: usize) -> Result<(), W4A8GemmRefusal> {
        if t.dtype() != dtype || t.elem_count() != elems || !t.is_contiguous() {
            return Err(W4A8GemmRefusal::Operands);
        }
        Ok(())
    }

    /// Launch the fused W4A8 GEMM: `[m, k]` activation codes against the resident packed weight,
    /// returning `[m, n]` in `out_dtype`. Every operand must already be contiguous on one CUDA
    /// device; a violation is a typed [`W4A8GemmRefusal::Operands`].
    pub fn w4a8_gemm(
        ops: &W4A8GemmOperands<'_>,
        out_dtype: DType,
    ) -> Result<Tensor, W4A8GemmError> {
        let Device::Cuda(dev) = ops.qx.device() else {
            return Err(W4A8GemmRefusal::Operands.into());
        };
        // Geometry comes from the resident weight; the staged activation codes may be `[m, k]` or
        // flat (`CublasLt::stage_int8_bytes` keeps them flat).
        let (n, k_half) = ops.packed.dims2().map_err(|_| W4A8GemmRefusal::Operands)?;
        let k = k_half * 2;
        if k == 0 || !ops.qx.elem_count().is_multiple_of(k) {
            return Err(W4A8GemmRefusal::Operands.into());
        }
        let m = ops.qx.elem_count() / k;
        let plan = check_w4a8_gemm(device_cap(dev)?, m, n, k, out_dtype)?;
        let all = [
            ops.qx,
            ops.xs,
            ops.packed,
            ops.s_rel,
            ops.codebook,
            ops.s_channel,
        ];
        if all
            .iter()
            .chain(ops.bias.iter())
            .any(|t| !t.device().same_device(ops.qx.device()))
        {
            return Err(W4A8GemmRefusal::Operands.into());
        }
        operand_shape(ops.qx, DType::U8, m * k)?;
        operand_shape(ops.xs, DType::F32, m)?;
        operand_shape(ops.packed, DType::U8, n * (k / 2))?;
        operand_shape(ops.s_rel, DType::U8, n * (k / 16))?;
        operand_shape(ops.codebook, DType::F32, 16)?;
        operand_shape(ops.s_channel, DType::F32, n)?;
        if let Some(b) = ops.bias {
            operand_shape(b, DType::F32, n)?;
        }
        let func = &functions(dev)?.gemm;
        let guards = [
            ops.qx.storage_and_layout(),
            ops.xs.storage_and_layout(),
            ops.packed.storage_and_layout(),
            ops.s_rel.storage_and_layout(),
            ops.codebook.storage_and_layout(),
            ops.s_channel.storage_and_layout(),
        ];
        let qx = view::<u8>(&guards[0].0, guards[0].1)?;
        let xs = view::<f32>(&guards[1].0, guards[1].1)?;
        let packed = view::<u8>(&guards[2].0, guards[2].1)?;
        let s_rel = view::<u8>(&guards[3].0, guards[3].1)?;
        let codebook = view::<f32>(&guards[4].0, guards[4].1)?;
        let s_channel = view::<f32>(&guards[5].0, guards[5].1)?;
        // Without a bias the pointer is never read (`has_bias == 0`); s_channel stands in.
        let bias_guard = ops.bias.map(|b| b.storage_and_layout());
        let bias_view = match &bias_guard {
            Some((g, l)) => Some(view::<f32>(g, l)?),
            None => None,
        };
        let bias = bias_view.as_ref().unwrap_or(&s_channel);
        let has_bias = i32::from(ops.bias.is_some());
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (plan.grid.0, plan.grid.1, 1),
            block_dim: (W4A8_GEMM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = dev.cuda_stream();
        let storage = match out_dtype {
            DType::F32 => {
                let mut y = unsafe { dev.alloc::<f32>(m * n) }?;
                let mut b = stream.launch_builder(func);
                b.arg(&qx).arg(&xs).arg(&packed).arg(&s_rel).arg(&codebook);
                b.arg(&s_channel).arg(bias).arg(&mut y);
                b.arg(&mi)
                    .arg(&ni)
                    .arg(&ki)
                    .arg(&has_bias)
                    .arg(&plan.out_code);
                unsafe { b.launch(cfg) }.map_err(drv)?;
                CudaStorage::wrap_cuda_slice(y, dev.clone())
            }
            DType::BF16 => {
                let mut y = unsafe { dev.alloc::<half::bf16>(m * n) }?;
                let mut b = stream.launch_builder(func);
                b.arg(&qx).arg(&xs).arg(&packed).arg(&s_rel).arg(&codebook);
                b.arg(&s_channel).arg(bias).arg(&mut y);
                b.arg(&mi)
                    .arg(&ni)
                    .arg(&ki)
                    .arg(&has_bias)
                    .arg(&plan.out_code);
                unsafe { b.launch(cfg) }.map_err(drv)?;
                CudaStorage::wrap_cuda_slice(y, dev.clone())
            }
            _ => {
                let mut y = unsafe { dev.alloc::<half::f16>(m * n) }?;
                let mut b = stream.launch_builder(func);
                b.arg(&qx).arg(&xs).arg(&packed).arg(&s_rel).arg(&codebook);
                b.arg(&s_channel).arg(bias).arg(&mut y);
                b.arg(&mi)
                    .arg(&ni)
                    .arg(&ki)
                    .arg(&has_bias)
                    .arg(&plan.out_code);
                unsafe { b.launch(cfg) }.map_err(drv)?;
                CudaStorage::wrap_cuda_slice(y, dev.clone())
            }
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(storage),
            (m, n),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_admits_every_w4a8_projection_geometry_at_sm80_and_above() {
        for cap in [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)] {
            let p = check_w4a8_gemm(cap, 4096, 6144, 6144, DType::BF16).unwrap();
            assert_eq!(p.grid, (48, 32));
            assert_eq!(p.out_code, 1);
        }
        // Partial edge tiles in M and N are served (bounds-guarded in the kernel).
        let p = check_w4a8_gemm((12, 0), 17, 1552, 256, DType::F32).unwrap();
        assert_eq!(p.grid, (13, 1));
    }

    #[test]
    fn plan_refuses_with_typed_labelled_reasons() {
        use W4A8GemmRefusal as R;
        let c = (12, 0);
        assert_eq!(
            check_w4a8_gemm((7, 5), 16, 16, 256, DType::BF16),
            Err(R::ComputeFloor { found: (7, 5) })
        );
        assert_eq!(check_w4a8_gemm(c, 0, 16, 256, DType::BF16), Err(R::Empty));
        assert_eq!(
            check_w4a8_gemm(c, 16, 16, 96, DType::BF16),
            Err(R::KTile { k: 96 })
        );
        assert_eq!(
            check_w4a8_gemm(c, 16, 16, 131_072 + 64, DType::BF16),
            Err(R::KOverflow { k: 131_136 })
        );
        assert_eq!(
            check_w4a8_gemm(c, 1 << 20, 1 << 12, 256, DType::BF16),
            Err(R::TooLarge)
        );
        assert_eq!(
            check_w4a8_gemm(c, 16, 16, 256, DType::U8),
            Err(R::Dtype(DType::U8))
        );
        for (r, label) in [
            (R::ComputeFloor { found: (7, 0) }, "compute_floor"),
            (R::Empty, "empty"),
            (R::KTile { k: 1 }, "k_tile"),
            (R::KOverflow { k: 1 }, "k_overflow"),
            (R::TooLarge, "too_large"),
            (R::Dtype(DType::U8), "dtype"),
            (R::Operands, "operands"),
        ] {
            assert_eq!(r.label(), label);
            assert_eq!(W4A8GemmError::Refused(r).label(), label);
        }
    }

    #[test]
    fn activation_prep_plan_admits_w4a8_rows_and_refuses_the_rest() {
        use W4A8GemmRefusal as R;
        assert_eq!(check_w4a8_quantize((8, 0), 4096, 6144, DType::BF16), Ok(1));
        assert_eq!(check_w4a8_quantize((12, 0), 1, 256, DType::F16), Ok(2));
        assert_eq!(check_w4a8_quantize((12, 0), 3, 16384, DType::F32), Ok(0));
        assert_eq!(
            check_w4a8_quantize((7, 5), 1, 256, DType::F32),
            Err(R::ComputeFloor { found: (7, 5) })
        );
        assert_eq!(
            check_w4a8_quantize((12, 0), 1, 320, DType::F32),
            Err(R::KTile { k: 320 })
        );
        assert_eq!(
            check_w4a8_quantize((12, 0), 0, 256, DType::F32),
            Err(R::Empty)
        );
        assert_eq!(
            check_w4a8_quantize((12, 0), 1, 256, DType::U8),
            Err(R::Dtype(DType::U8))
        );
        assert_eq!(
            check_w4a8_quantize((12, 0), 1 << 24, 1 << 10, DType::F32),
            Err(R::TooLarge)
        );
    }

    #[test]
    fn tier_floor_is_sm80() {
        assert_eq!(
            w4a8_tier_refusal((7, 5)),
            Some(W4A8TierRefusal { found: (7, 5) })
        );
        assert!(w4a8_tier_refusal((8, 0)).is_none());
        assert!(w4a8_tier_refusal((12, 0)).is_none());
        assert!(w4a8_tier_refusal((7, 5))
            .unwrap()
            .to_string()
            .contains("sm_80"));
    }

    #[test]
    fn source_is_single_file_with_its_attribution_and_entry_point() {
        let src = W4A8_GEMM_SRC.src;
        assert!(!src.contains("#include"), "nvrtc sources take no headers");
        assert!(src.contains("comfy-kitchen") && src.contains("Apache-2.0"));
        assert!(src.contains(&format!(
            "void __launch_bounds__(W4A8_THREADS) {W4A8_GEMM_FUNCTION}("
        )));
        assert!(src.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"));
        assert!(src.contains(&format!(
            "void __launch_bounds__(256) {W4A8_QUANTIZE_FUNCTION}("
        )));
        // The tile constants the host plans with are the kernel's.
        assert!(src.contains(&format!("#define W4A8_BM {W4A8_GEMM_TILE_M}")));
        assert!(src.contains(&format!("#define W4A8_BN {W4A8_GEMM_TILE_N}")));
        assert!(src.contains(&format!("#define W4A8_BK {W4A8_GEMM_TILE_K}")));
        assert!(src.contains(&format!("#define W4A8_THREADS {W4A8_GEMM_THREADS}")));
    }
}
