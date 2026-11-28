//! The fused **per-tensor dynamic activation quantizer** (SW-6126, feature SW-F287 S1): amax →
//! scale → cast as two device passes over [`ACT_QUANT_SRC`] (nvrtc, compiled once per device),
//! emitting E4M3FN bytes or int8 codes directly and leaving the scale **on the device**.
//!
//! # What it replaces, and the reference it must equal byte for byte
//!
//! - `cublaslt::quantize_activation_fp8` — the candle op chain `max|x|` → host scalar →
//!   `x / scale` → **a CPU round trip** for the `f32 → F8E4M3` cast. On CUDA the fused kernel
//!   now serves it ([`Fp8ScaleRule::Reciprocal`]); the chain stays, verbatim, as
//!   `cublaslt::quantize_activation_fp8_reference`.
//! - `cublaslt::quantize_activation_int8` — `max|x|` → host scalar → `round(x / scale)` →
//!   `clamp(±127)` (six launches and three host transfers). Fused: one amax pass and one cast pass,
//!   F32 codes ([`Int8Output::F32Codes`]) for that function's contract or the i8 bytes cuBLASLt
//!   reads ([`Int8Output::I8Bytes`]) for `Int8Linear`'s forward.
//! - the dynamic-scale branch of `CublasLt::scaled_fp8_activation` (`Fp8Linear`'s forward):
//!   device cast, but its amax was a candle reduction plus a 4-byte device → host read to pass the
//!   scale back to the GEMM. Fused ([`Fp8ScaleRule::Divide`]): the scale stays on the device and the
//!   cuBLASLt fp8 GEMM reads it through its `B_SCALE_POINTER`.
//!
//! There is no stock device alternative to beat: candle's own CUDA `f32 → F8E4M3` cast cannot be
//! launched (candle-core asks for `cast_f32_f8e4m3`, candle-kernels defines `cast_f32_f8_e4m3`;
//! `CUDA_ERROR_NOT_FOUND` on sm_120 too), which is why the reference round-trips through the CPU.
//! candle-gen's `fused_kernels_bench` example measures the fused kernel against the references.
//!
//! The kernel reproduces each reference step with the reference's own operation (see the contract
//! at the top of `act_quant.cu`): the reduction's NaN policy per dtype, the f32 scale with its
//! `f32::MIN_POSITIVE` floor, candle's `affine(1/scale, 0)` as `fma(x, mul, 0)` (or the Fp8Linear
//! route's IEEE divide), the `float8` crate's E4M3 rounding ported bit for bit, and candle's
//! `round().clamp(-127, 127)`. The GPU tests in `cuda_tests` hold it to byte identity.
//!
//! # Path selection (epic E1/E5)
//!
//! Each call goes through [`ACT_QUANT_FP8_PATH`] / [`ACT_QUANT_INT8_PATH`] (the media fused-path
//! seam, [`crate::fused_path`]): the reference runs, and is counted with its reason, for a CPU or
//! Metal tensor, a dtype other than F32 / BF16 / F16, an empty tensor, more than
//! [`ACT_QUANT_MAX_ELEMS`] elements, a build without `cuda`, either switch off
//! (`CANDLE_FUSED_ACT_QUANT_FP8` / `CANDLE_FUSED_ACT_QUANT_INT8`, `CANDLE_MEDIA_FUSED_KERNELS`),
//! or a cached compile failure. [`check_act_quant`] is the pure plan both the launch and the CPU
//! tests use.

use std::fmt;

use candle_core::{DType, Tensor};

use crate::fused_path::{FusedKernelPath, PathReason};
use crate::nvrtc::KernelSource;

/// The fused activation-quantizer source; compiled once per device on first use.
pub const ACT_QUANT_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_act_quant_v1",
    src: include_str!("act_quant.cu"),
    // Integer / f32 / f64 arithmetic and `cvt.f32.f16` only — the E4M3 rounding is software, so no
    // sm_89 `cvt.*.e4m3x2` is needed; sm_70 is the oldest this workspace targets.
    cc_floor: (7, 0),
};

/// The fp8 activation quantizer's path record (`quantize_activation_fp8`, `Fp8Linear`).
pub static ACT_QUANT_FP8_PATH: FusedKernelPath =
    FusedKernelPath::new("act_quant_fp8", "CANDLE_FUSED_ACT_QUANT_FP8");

/// The int8 activation quantizer's path record (`quantize_activation_int8`, `Int8Linear`).
pub static ACT_QUANT_INT8_PATH: FusedKernelPath =
    FusedKernelPath::new("act_quant_int8", "CANDLE_FUSED_ACT_QUANT_INT8");

/// Largest tensor the fused quantizer launches (epic E6); larger ones take the reference.
pub const ACT_QUANT_MAX_ELEMS: usize = i32::MAX as usize;

/// Threads per block of both passes (`AQ_THREADS` in the source).
pub const ACT_QUANT_THREADS: u32 = 256;
/// Most blocks the amax pass launches (grid-stride; bounds the atomics).
pub const ACT_QUANT_AMAX_BLOCKS: u32 = 1024;
/// Most blocks the cast pass launches (grid-stride).
pub const ACT_QUANT_CAST_BLOCKS: u32 = 16384;

/// How the fp8 route scales `x` before the E4M3 cast — the two references in the tree differ here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8ScaleRule {
    /// `quantize_activation_fp8`'s candle chain: `x · (1/scale)` as candle's `affine` computes it,
    /// with the amax reduced in `x`'s dtype (NaN wins for F16 / BF16).
    Reciprocal,
    /// `CublasLt::scaled_fp8_activation` (Fp8Linear): IEEE `x / scale`, amax of the F32 cast
    /// (NaN ignored).
    Divide,
}

/// The int8 route's code format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int8Output {
    /// Rounded codes in `F32` — `QuantizedActivation::q`'s contract.
    F32Codes,
    /// Two's-complement `i8` bytes (a `U8` tensor) — what the cuBLASLt int8 GEMM reads.
    I8Bytes,
}

/// Why the fused quantizer does not serve an input. The reference serves it instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActQuantRefusal {
    /// The tensor is not on a CUDA device.
    NotCuda,
    /// Only `F32`, `BF16` and `F16` are served.
    Dtype(DType),
    /// The tensor has no elements (the reference's reduction refuses it too).
    Empty,
    /// More than [`ACT_QUANT_MAX_ELEMS`] elements.
    TooLarge(usize),
}

impl ActQuantRefusal {
    /// The fused-path reason this refusal is counted under.
    pub fn reason(&self) -> PathReason {
        match self {
            Self::NotCuda => PathReason::NotCuda,
            Self::Dtype(d) => PathReason::Dtype(*d),
            Self::Empty => PathReason::Shape,
            Self::TooLarge(_) => PathReason::IndexOverflow,
        }
    }
}

impl fmt::Display for ActQuantRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "tensor is not on a CUDA device"),
            Self::Dtype(d) => write!(f, "dtype {d:?} is not served (F32, BF16 or F16)"),
            Self::Empty => write!(f, "tensor is empty"),
            Self::TooLarge(n) => write!(f, "{n} elements exceed {ACT_QUANT_MAX_ELEMS}"),
        }
    }
}

/// A served input's launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActQuantPlan {
    /// Input dtype (F32, BF16 or F16).
    pub dtype: DType,
    /// Element count (1..=[`ACT_QUANT_MAX_ELEMS`]).
    pub elems: usize,
    /// Blocks of the amax pass.
    pub amax_blocks: u32,
    /// Blocks of the cast pass.
    pub cast_blocks: u32,
}

/// The fused quantizer's input check (pure; every lane runs it): `on_cuda` is whether the tensor
/// lives on a CUDA device.
pub fn check_act_quant(
    on_cuda: bool,
    dtype: DType,
    elems: usize,
) -> Result<ActQuantPlan, ActQuantRefusal> {
    if !on_cuda {
        return Err(ActQuantRefusal::NotCuda);
    }
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(ActQuantRefusal::Dtype(dtype));
    }
    if elems == 0 {
        return Err(ActQuantRefusal::Empty);
    }
    if elems > ACT_QUANT_MAX_ELEMS {
        return Err(ActQuantRefusal::TooLarge(elems));
    }
    // One thread per group of four, grid-stride beyond the caps.
    let blocks = elems.div_ceil(4).div_ceil(ACT_QUANT_THREADS as usize);
    let capped = |cap: u32| blocks.min(cap as usize) as u32;
    Ok(ActQuantPlan {
        dtype,
        elems,
        amax_blocks: capped(ACT_QUANT_AMAX_BLOCKS),
        cast_blocks: capped(ACT_QUANT_CAST_BLOCKS),
    })
}

/// A quantized activation whose scale stays on the device: the codes (`F8E4M3`, `F32` codes or
/// `U8` i8 bytes, `x`'s shape) and a one-element `F32` scale tensor on the same device
/// (`dequant = q · scale`).
#[derive(Debug, Clone)]
pub struct DeviceQuantized {
    /// The codes, in `x`'s shape.
    pub codes: Tensor,
    /// `[1]` `F32`: `absmax / qmax`, floored at `f32::MIN_POSITIVE`.
    pub scale: Tensor,
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::fused_path::Attempt;
    use crate::nvrtc::KernelCompileError;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Result, Shape, Storage};
    use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    /// The six entry points of one device's module, resolved once.
    struct Functions {
        amax_f32: CudaFunction,
        amax_16: CudaFunction,
        fp8_f32: CudaFunction,
        fp8_16: CudaFunction,
        int8_f32: CudaFunction,
        int8_16: CudaFunction,
    }

    impl Functions {
        fn resolve(dev: &CudaDevice) -> std::result::Result<Self, KernelCompileError> {
            let m = ACT_QUANT_SRC.compiled(dev)?;
            Ok(Self {
                amax_f32: m.function("act_quant_amax_f32")?,
                amax_16: m.function("act_quant_amax_16")?,
                fp8_f32: m.function("act_quant_fp8_f32")?,
                fp8_16: m.function("act_quant_fp8_16")?,
                int8_f32: m.function("act_quant_int8_f32")?,
                int8_16: m.function("act_quant_int8_16")?,
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
        candle_core::Error::Cuda(format!("fused activation quantizer: {e:?}").into())
    }

    fn wrap<T: candle_core::cuda_backend::CudaDType>(
        slice: CudaSlice<T>,
        dev: &CudaDevice,
        shape: impl Into<Shape>,
    ) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    /// The pass-2 output buffer.
    enum Codes {
        Fp8(CudaSlice<float8::F8E4M3>),
        F32(CudaSlice<f32>),
        Bytes(CudaSlice<u8>),
    }

    /// A finished launch: the codes buffer, the one-element device scale and the input shape.
    struct Raw {
        codes: Codes,
        scale: CudaSlice<f32>,
        shape: Shape,
    }

    /// Plan, resolve and run both passes; `Attempt::Reference` for an unserved input or a cached
    /// compile failure. `fp8` is `Some(rule)` for E4M3, `None` for int8 (`int8` then says how).
    fn run(x: &Tensor, fp8: Option<Fp8ScaleRule>, int8: Int8Output) -> Result<Attempt<Raw>> {
        let plan = match check_act_quant(x.device().is_cuda(), x.dtype(), x.elem_count()) {
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
        // The kernels index from 0 with vector loads: a contiguous, offset-0 operand (candle's
        // allocations are 256-byte aligned).
        let x = x.contiguous()?;
        let x = if x.layout().start_offset() == 0 {
            x
        } else {
            x.force_contiguous()?
        };
        let n = plan.elems as u64;
        let nan_propagates = i32::from(match fp8 {
            Some(Fp8ScaleRule::Divide) => false,
            _ => plan.dtype != DType::F32,
        });
        let bf16 = i32::from(plan.dtype == DType::BF16);
        let divide = i32::from(fp8 == Some(Fp8ScaleRule::Divide));
        let f32_codes = i32::from(int8 == Int8Output::F32Codes);
        let stream = dev.cuda_stream();
        let mut amax = dev.alloc_zeros::<u32>(1)?;
        let mut scale = unsafe { dev.alloc::<f32>(1) }?;
        let mut codes = match (fp8, int8) {
            (Some(_), _) => Codes::Fp8(unsafe { dev.alloc::<float8::F8E4M3>(plan.elems) }?),
            (None, Int8Output::F32Codes) => Codes::F32(unsafe { dev.alloc::<f32>(plan.elems) }?),
            (None, Int8Output::I8Bytes) => Codes::Bytes(unsafe { dev.alloc::<u8>(plan.elems) }?),
        };
        let cfg = |blocks: u32| LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (ACT_QUANT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (storage, _) = x.storage_and_layout();
        let Storage::Cuda(cs) = &*storage else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        // Pass 1: amax (bits of a non-negative float, atomicMax'd). Pass 2: scale (block 0
        // publishes it) + cast. Both on the device's stream, so pass 2 sees the finished amax.
        macro_rules! launch {
            ($amax_fn:expr, $cast_fn:expr, $src:expr, $out:expr, $mode:expr $(, $extra:expr)?) => {{
                let mut b = stream.launch_builder($amax_fn);
                b.arg($src).arg(&n).arg(&mut amax).arg(&nan_propagates);
                $( b.arg($extra); )?
                unsafe { b.launch(cfg(plan.amax_blocks)) }.map_err(drv)?;
                let mut b = stream.launch_builder($cast_fn);
                b.arg($src).arg(&n).arg(&amax).arg(&mut scale).arg($out).arg(&$mode);
                $( b.arg($extra); )?
                unsafe { b.launch(cfg(plan.cast_blocks)) }.map_err(drv)?;
            }};
        }
        match (&cs.slice, &mut codes) {
            (CudaStorageSlice::F32(src), Codes::Fp8(out)) => {
                launch!(&f.amax_f32, &f.fp8_f32, src, out, divide)
            }
            (CudaStorageSlice::F32(src), Codes::F32(out)) => {
                launch!(&f.amax_f32, &f.int8_f32, src, out, f32_codes)
            }
            (CudaStorageSlice::F32(src), Codes::Bytes(out)) => {
                launch!(&f.amax_f32, &f.int8_f32, src, out, f32_codes)
            }
            (CudaStorageSlice::BF16(src), Codes::Fp8(out)) => {
                launch!(&f.amax_16, &f.fp8_16, src, out, divide, &bf16)
            }
            (CudaStorageSlice::BF16(src), Codes::F32(out)) => {
                launch!(&f.amax_16, &f.int8_16, src, out, f32_codes, &bf16)
            }
            (CudaStorageSlice::BF16(src), Codes::Bytes(out)) => {
                launch!(&f.amax_16, &f.int8_16, src, out, f32_codes, &bf16)
            }
            (CudaStorageSlice::F16(src), Codes::Fp8(out)) => {
                launch!(&f.amax_16, &f.fp8_16, src, out, divide, &bf16)
            }
            (CudaStorageSlice::F16(src), Codes::F32(out)) => {
                launch!(&f.amax_16, &f.int8_16, src, out, f32_codes, &bf16)
            }
            (CudaStorageSlice::F16(src), Codes::Bytes(out)) => {
                launch!(&f.amax_16, &f.int8_16, src, out, f32_codes, &bf16)
            }
            _ => return Ok(Attempt::Reference(PathReason::Dtype(plan.dtype))),
        }
        drop(storage);
        Ok(Attempt::Fused(Raw {
            codes,
            scale,
            shape: x.shape().clone(),
        }))
    }

    fn into_tensors(raw: Raw, dev: &CudaDevice) -> DeviceQuantized {
        let codes = match raw.codes {
            Codes::Fp8(c) => wrap(c, dev, raw.shape),
            Codes::F32(c) => wrap(c, dev, raw.shape),
            Codes::Bytes(c) => wrap(c, dev, raw.shape),
        };
        DeviceQuantized {
            codes,
            scale: wrap(raw.scale, dev, 1),
        }
    }

    fn tensors(x: &Tensor, attempt: Attempt<Raw>) -> Result<Attempt<DeviceQuantized>> {
        attempt.try_map(|raw| match x.device() {
            Device::Cuda(dev) => Ok(into_tensors(raw, dev)),
            _ => candle_core::bail!("a fused activation quantizer run implies a CUDA input"),
        })
    }

    /// The fused E4M3 quantizer: `Attempt::Fused` with `F8E4M3` codes and the device scale, or
    /// `Attempt::Reference(reason)` when it does not serve `x` (the caller then runs the reference
    /// `rule` names). It neither consults the switch nor records the outcome —
    /// [`ACT_QUANT_FP8_PATH`]'s `dispatch` does both.
    pub fn fused_fp8(x: &Tensor, rule: Fp8ScaleRule) -> Result<Attempt<DeviceQuantized>> {
        tensors(x, run(x, Some(rule), Int8Output::F32Codes)?)
    }

    /// The fused int8 quantizer (codes in the `output` format), with [`fused_fp8`]'s contract.
    pub fn fused_int8(x: &Tensor, output: Int8Output) -> Result<Attempt<DeviceQuantized>> {
        tensors(x, run(x, None, output)?)
    }

    /// The E4M3 codes (in the input's row-major order) and device scale of a fused fp8 run as raw
    /// buffers, for the cuBLASLt fp8 GEMM to consume without a copy (the scale is its
    /// `B_SCALE_POINTER`).
    pub(crate) struct RawFp8 {
        pub(crate) codes: CudaSlice<float8::F8E4M3>,
        pub(crate) scale: CudaSlice<f32>,
    }

    /// [`fused_fp8`] as [`RawFp8`] buffers.
    pub(crate) fn fused_fp8_raw(x: &Tensor, rule: Fp8ScaleRule) -> Result<Attempt<RawFp8>> {
        run(x, Some(rule), Int8Output::F32Codes)?.try_map(|raw| match raw.codes {
            Codes::Fp8(codes) => Ok(RawFp8 {
                codes,
                scale: raw.scale,
            }),
            _ => candle_core::bail!("a fused fp8 run produced non-fp8 codes"),
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda_impl::fused_fp8_raw;
#[cfg(feature = "cuda")]
pub use cuda_impl::{fused_fp8, fused_int8};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_refuses_what_the_kernel_does_not_serve() {
        assert_eq!(
            check_act_quant(false, DType::F32, 16),
            Err(ActQuantRefusal::NotCuda)
        );
        for dtype in [DType::U8, DType::F64, DType::I64, DType::F8E4M3] {
            assert_eq!(
                check_act_quant(true, dtype, 16),
                Err(ActQuantRefusal::Dtype(dtype))
            );
        }
        assert_eq!(
            check_act_quant(true, DType::BF16, 0),
            Err(ActQuantRefusal::Empty)
        );
        let too_many = ACT_QUANT_MAX_ELEMS + 1;
        assert_eq!(
            check_act_quant(true, DType::F16, too_many),
            Err(ActQuantRefusal::TooLarge(too_many))
        );
        assert!(check_act_quant(true, DType::F16, ACT_QUANT_MAX_ELEMS).is_ok());
    }

    #[test]
    fn refusals_map_to_typed_path_reasons() {
        assert_eq!(ActQuantRefusal::NotCuda.reason().label(), "not_cuda");
        assert_eq!(ActQuantRefusal::Dtype(DType::U8).reason().label(), "dtype");
        assert_eq!(ActQuantRefusal::Empty.reason().label(), "shape");
        assert_eq!(
            ActQuantRefusal::TooLarge(1 << 40).reason().label(),
            "index_overflow"
        );
    }

    #[test]
    fn the_grid_covers_every_group_within_the_caps() {
        for (elems, amax, cast) in [
            (1, 1, 1),
            (4, 1, 1),
            (1024, 1, 1),
            (1025, 2, 2),
            (4118 * 6144, ACT_QUANT_AMAX_BLOCKS, ACT_QUANT_CAST_BLOCKS),
            (256 * 3072, 768, 768),
        ] {
            let plan = check_act_quant(true, DType::BF16, elems).unwrap();
            assert_eq!(
                (plan.amax_blocks, plan.cast_blocks),
                (amax, cast),
                "{elems}"
            );
            assert_eq!(plan.elems, elems);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    //! GPU byte-identity of the fused quantizer against the in-tree references. `#[ignore]`d CUDA
    //! compute (registered in `scripts/ci/test_suites.json`); run them on a CUDA box with
    //! `cargo test -p candle-quant-kernels --features cuda --lib act_quant -- --ignored`.
    use super::*;
    use crate::cublaslt::{
        quantize_activation_fp8, quantize_activation_fp8_reference, quantize_activation_int8,
        quantize_activation_int8_reference,
    };
    use crate::fused_path::{media_fused_policy_guard, Attempt};
    use candle_core::{Device, Result};

    fn cuda() -> Device {
        Device::new_cuda(0).expect("CUDA device 0")
    }

    /// Deterministic values with a heavy tail: mostly |v| < 1, every 97th an outlier up to ~300x.
    fn outlier_heavy(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                let v = (u * 2.0 - 1.0) as f32;
                if i % 97 == 0 {
                    v * 300.0
                } else {
                    v
                }
            })
            .collect()
    }

    fn bytes(t: &Tensor) -> Result<Vec<u8>> {
        match t.dtype() {
            DType::F8E4M3 => Ok(t
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect()),
            DType::U8 => t.flatten_all()?.to_vec1::<u8>(),
            DType::F32 => Ok(t
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect()),
            other => candle_core::bail!("no byte view of {other:?}"),
        }
    }

    fn first_mismatch(a: &[u8], b: &[u8]) -> Option<usize> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).position(|(x, y)| x != y)
    }

    fn fused(a: Attempt<DeviceQuantized>) -> DeviceQuantized {
        match a {
            Attempt::Fused(q) => q,
            Attempt::Reference(r) => panic!("the fused quantizer declined: {r}"),
        }
    }

    /// Inputs in every served dtype and several shapes, including tails that are not a multiple
    /// of the 4-element vector group, a single element, all zeros, signed zeros, exact E4M3 ties
    /// and int8 half-way points, subnormals, ±inf and NaN (with the NaN policy per dtype).
    fn cases(dev: &Device) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (rows, cols) in [(1, 1), (1, 3), (3, 5), (7, 129), (64, 3072), (4118, 64)] {
            let v = outlier_heavy(rows * cols, (rows * 31 + cols) as u64);
            out.push((
                format!("outlier {rows}x{cols}"),
                Tensor::from_vec(v, (rows, cols), dev)?,
            ));
        }
        out.push(("zeros".into(), Tensor::zeros((5, 7), DType::F32, dev)?));
        // amax = 448 makes scale exactly 1, so every listed value lands on an E4M3 tie / an int8
        // half-way point: a rounding-mode change flips them.
        let mut ties = vec![
            448.0f32, -0.0, 0.0, 1.0625, -1.0625, 1.1875, 2.5, -2.5, 3.5, 0.5, -0.5,
        ];
        ties.extend([
            17.0,
            19.0,
            0.0029296875,
            -0.0029296875,
            0.0068359375,
            464.0,
            -460.0,
        ]);
        let len = ties.len();
        out.push(("e4m3 ties".into(), Tensor::from_vec(ties, len, dev)?));
        // amax = 127 makes the int8 scale exactly 1.
        let half = vec![127.0f32, 2.5, -2.5, 0.5, -0.5, 1.5, -126.5, 3.4999998, -0.0];
        let len = half.len();
        out.push(("int8 half-way".into(), Tensor::from_vec(half, len, dev)?));
        // A subnormal next to a large amax underflows `x * (1/scale)` to a signed zero.
        let tiny = vec![1.0e10f32, -1.0e-40, 1.0e-40, -3.0e-39, 0.0, 7.0];
        let len = tiny.len();
        out.push(("underflow".into(), Tensor::from_vec(tiny, len, dev)?));
        let specials = vec![
            1.0f32,
            f32::NAN,
            -2.0,
            f32::INFINITY,
            -f32::INFINITY,
            -f32::NAN,
            3.0,
        ];
        let len = specials.len();
        out.push(("inf/nan".into(), Tensor::from_vec(specials, len, dev)?));
        let nan_only = vec![1.5f32, -f32::NAN, 0.25, 9.0, -4.0];
        let len = nan_only.len();
        out.push(("nan".into(), Tensor::from_vec(nan_only, len, dev)?));
        let mut typed = Vec::new();
        for (name, t) in out {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                typed.push((format!("{name} {dtype:?}"), t.to_dtype(dtype)?));
            }
        }
        // A strided (non-contiguous) and an offset view.
        let base =
            Tensor::from_vec(outlier_heavy(32 * 48, 5), (32, 48), dev)?.to_dtype(DType::BF16)?;
        typed.push(("transposed BF16".into(), base.t()?));
        typed.push(("offset BF16".into(), base.narrow(0, 3, 17)?));
        Ok(typed)
    }

    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fp8_codes_are_byte_identical_to_the_candle_chain() -> Result<()> {
        let dev = cuda();
        for (name, x) in cases(&dev)? {
            let reference = quantize_activation_fp8_reference(&x)?;
            let q = fused(fused_fp8(&x, Fp8ScaleRule::Reciprocal)?);
            let scale = q.scale.to_vec1::<f32>()?[0];
            assert_eq!(scale.to_bits(), reference.scale.to_bits(), "{name}: scale");
            assert_eq!(q.codes.dims(), x.dims(), "{name}");
            let (a, b) = (bytes(&q.codes)?, bytes(&reference.q)?);
            assert_eq!(
                first_mismatch(&a, &b),
                None,
                "{name}: fused vs reference codes"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn int8_codes_are_byte_identical_to_the_candle_chain() -> Result<()> {
        let dev = cuda();
        let lt = crate::cublaslt::CublasLt::new(&dev)?;
        for (name, x) in cases(&dev)? {
            let reference = quantize_activation_int8_reference(&x)?;
            let q = fused(fused_int8(&x, Int8Output::F32Codes)?);
            let scale = q.scale.to_vec1::<f32>()?[0];
            assert_eq!(scale.to_bits(), reference.scale.to_bits(), "{name}: scale");
            let (a, b) = (bytes(&q.codes)?, bytes(&reference.q)?);
            assert_eq!(first_mismatch(&a, &b), None, "{name}: F32 codes");
            // The i8 bytes equal what the cuBLASLt stage narrows the reference's F32 codes to.
            let q8 = fused(fused_int8(&x, Int8Output::I8Bytes)?);
            assert_eq!(q8.codes.dtype(), DType::U8);
            let flat = reference.q.flatten_all()?;
            let staged = lt.stage_int8(&flat.reshape((1, flat.elem_count()))?)?;
            let (a, b) = (bytes(&q8.codes)?, staged.bytes_to_host()?);
            assert_eq!(
                first_mismatch(&a, &b),
                None,
                "{name}: i8 bytes vs staged reference"
            );
        }
        Ok(())
    }

    /// The Fp8Linear route: codes equal `scaled_fp8_activation`'s (hardware `cvt.rn.satfinite`)
    /// device cast for the same input, and the scale equals its host scalar.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn fp8_divide_rule_matches_the_fp8_linear_quantizer() -> Result<()> {
        let dev = cuda();
        let lt = crate::cublaslt::CublasLt::new(&dev)?;
        if !lt.meets_fp8_floor()? {
            eprintln!("skipping: the hardware E4M3 cast needs sm_89");
            return Ok(());
        }
        for (name, x) in cases(&dev)? {
            let x = x.flatten_all()?;
            let x = x.reshape((1, x.elem_count()))?;
            let (dev_fp8, scale) = lt.scaled_fp8_activation(&x, None, false)?;
            let q = fused(fused_fp8(&x, Fp8ScaleRule::Divide)?);
            assert_eq!(
                q.scale.to_vec1::<f32>()?[0].to_bits(),
                scale.to_bits(),
                "{name}: scale"
            );
            let (a, b) = (bytes(&q.codes)?, dev_fp8.codes_to_host(&lt)?);
            assert_eq!(
                first_mismatch(&a, &b),
                None,
                "{name}: fused vs scaled_fp8_activation"
            );
        }
        Ok(())
    }

    /// The software E4M3 rounding equals the `float8` crate on a sweep of binary32 bit patterns
    /// (every 251st, every E4M3 tie and grid point with its neighbours, and the specials), with
    /// scale 1 (an amax of exactly 448 pinned in).
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn the_e4m3_rounding_matches_float8_on_a_bit_pattern_sweep() -> Result<()> {
        let dev = cuda();
        let mut v: Vec<f32> = (0..=u32::MAX)
            .step_by(251)
            .map(f32::from_bits)
            .filter(|f| f.is_nan() || f.abs() <= 448.0)
            .collect();
        v.extend([448.0, -448.0, 0.0, -0.0, f32::MIN_POSITIVE, 1.0e-45]);
        // Every E4M3 tie and representable point (and their binary32 neighbours) across the
        // normal and subnormal range: the top five mantissa bits at each exponent 2^-27..2^9.
        for exp in 100u32..=136 {
            for top in 0u32..32 {
                let bits = (exp << 23) | (top << 18);
                for b in [bits - 1, bits, bits + 1] {
                    for sign in [0, 1u32 << 31] {
                        let f = f32::from_bits(b | sign);
                        if f.abs() <= 448.0 {
                            v.push(f);
                        }
                    }
                }
            }
        }
        let n = v.len();
        let want: Vec<u8> = v
            .iter()
            .map(|f| float8::F8E4M3::from_f32(*f).to_bits())
            .collect();
        let x = Tensor::from_vec(v.clone(), n, &dev)?;
        let q = fused(fused_fp8(&x, Fp8ScaleRule::Divide)?);
        assert_eq!(q.scale.to_vec1::<f32>()?[0], 1.0);
        let got = bytes(&q.codes)?;
        // NaN: the device divide canonicalises the NaN (its sign is lost), float8 keeps the
        // input's sign; compare those by NaN-ness only.
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            if v[i].is_nan() {
                assert_eq!(g & 0x7F, 0x7F, "NaN at {i}");
            } else {
                assert_eq!(g, w, "bits {:#010x}", v[i].to_bits());
            }
        }
        Ok(())
    }

    /// The public functions take the fused path on CUDA (recorded), and the switch sends the same
    /// call to the reference (recorded as `disabled`) with identical output.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn the_public_quantizers_report_their_path() -> Result<()> {
        let dev = cuda();
        let x =
            Tensor::from_vec(outlier_heavy(64 * 96, 9), (64, 96), &dev)?.to_dtype(DType::BF16)?;
        let _media = media_fused_policy_guard(Some(true));
        for (path, int8) in [(&ACT_QUANT_FP8_PATH, false), (&ACT_QUANT_INT8_PATH, true)] {
            let _kernel = path.guard(Some(true));
            let quantize = |x: &Tensor| {
                if int8 {
                    quantize_activation_int8(x)
                } else {
                    quantize_activation_fp8(x)
                }
            };
            let start = path.thread_counts();
            let on = quantize(&x)?;
            let d = path.thread_counts().since(&start);
            assert_eq!((d.fused, d.reference_total()), (1, 0), "{}", path.name());

            path.set_enabled(Some(false));
            let mid = path.thread_counts();
            let off = quantize(&x)?;
            let d = path.thread_counts().since(&mid);
            assert_eq!(
                (d.fused, d.reference_for("disabled")),
                (0, 1),
                "{}",
                path.name()
            );
            assert_eq!(on.scale.to_bits(), off.scale.to_bits());
            assert_eq!(first_mismatch(&bytes(&on.q)?, &bytes(&off.q)?), None);

            // CPU input: the reference, counted as `not_cuda`.
            path.set_enabled(Some(true));
            let cpu = x.to_device(&Device::Cpu)?;
            let before = path.thread_counts();
            quantize(&cpu)?;
            let d = path.thread_counts().since(&before);
            assert_eq!(d.reference_for("not_cuda"), 1, "{}", path.name());
        }
        Ok(())
    }
}
