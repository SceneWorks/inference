//! **cuBLASLt 8-bit GEMM wrapper** (sc-9299 spike, epic 9083) — the compute foundation for a
//! future fp8 fast tier AND the community INT8-ConvRot consume path (sc-9300). This is the piece the
//! sc-8523 NO-GO said was missing: candle's own int8 MMQ kernels (`fast_mmq`) run *slower* than plain
//! bf16 cuBLAS at DiT shapes on this box, and candle has no fp8 GEMM at all. cuBLASLt runs both at the
//! near-peak (≈2×-bf16 on Ada/Blackwell for fp8) the whole 8-bit track always assumed.
//!
//! # Why a candle-gen wrapper and not a candle fork
//! At our pin (`c1e6756a89`) candle-core already exposes [`F8E4M3`](candle_core::DType::F8E4M3) as a
//! first-class CUDA dtype (storage, casts, const-set — only the *matmul dispatch* is missing) and
//! re-exports `cudarc`, whose `cublaslt` module ships the raw sys bindings + a thin `result` layer
//! (`create_matmul_desc` / `create_matrix_layout` / `matmul` / heuristic search). The safe
//! `CudaBlasLT::matmul` only wires f32/f16/bf16 (scale-type hard-coded `CUDA_R_32F`, compute-type from
//! a per-`T` trait) — no fp8, no int8. So we drop to that `result` layer and drive `cublasLtMatmul`
//! ourselves for the two 8-bit configs. No fork; if we ever upstream, the dtype groundwork suggests a
//! PR would be welcome.
//!
//! # The two configs (locked-decision-7: sm_89 floor, modern CUDA 12.x layouts only — no col32/IMMA)
//! * **fp8 E4M3** — A/B both `CUDA_R_8F_E4M3`, `CUBLAS_COMPUTE_32F` accumulate, per-tensor `scaleA` /
//!   `scaleB` passed as **device pointers** (`A_SCALE_POINTER` / `B_SCALE_POINTER`), output bf16.
//!   cuBLASLt fp8 requires **TN** (A transposed, B non-transposed, both stored row-major) and
//!   16-byte-aligned leading dims. `FAST_ACCUM` is left OFF (the numerically-safe default; it trades
//!   accuracy for speed and the epic wants the parity proof first).
//! * **int8 IGEMM** — A/B `CUDA_R_8I`, `CUBLAS_COMPUTE_32I` → **int32** accumulate, output int32. On
//!   CUDA 12.x the relaxed **TN row-major** path removes the legacy `COL32` re-layout; scales are
//!   applied *after* the kernel (we return the raw int32 and let the caller fold `scaleA·scaleB`),
//!   which keeps the accumulate exact and matches how a ConvRot checkpoint wants to re-scale.
//!
//! # Layout contract (the fiddly part)
//! candle stores a `Linear` weight row-major as `(N, K)` (out, in) and the activation as `(M, K)`.
//! For `D = X · Wᵀ` (`(M,N)`), the TN cuBLASLt call is, in cuBLAS's **column-major** worldview:
//! `A = W` declared `(K, N)` col-major (== our row-major `(N,K)`, i.e. `op(A)=Aᵀ`, `transa=T`),
//! `B = X` declared `(K, M)` col-major (== our row-major `(M,K)`, `transb=N`), `D` declared `(N, M)`
//! col-major == our row-major `(M, N)`. Leading dims are all `K` for A/B and `N` for D. This is the
//! identity the tests pin against an f32 reference.
//!
//! Everything on the cuBLASLt handle is `#[cfg(feature = "cuda")]`; the CPU/Metal builds see only the
//! small dtype helpers ([`quantize_activation_fp8`] / [`quantize_activation_int8`], pure candle ops).

use candle_core::{DType, Device, Result, Tensor};

/// Result of a per-tensor dynamic activation quantization: the packed 8-bit tensor plus the scalar
/// `scale` such that `dequant = q * scale` (fp8: absmax/448; int8: absmax/127).
pub struct QuantizedActivation {
    /// The 8-bit tensor (`F8E4M3` for fp8; rounded int codes in `F32` for int8 — narrowed to device
    /// `i8` at the cuBLASLt seam so the pure-candle helper stays dtype-portable).
    pub q: Tensor,
    /// Per-tensor scale, `absmax / dtype_max`.
    pub scale: f32,
}

/// fp8 E4M3 finite absmax (values above this saturate to NaN/inf). Used as the dynamic-range divisor.
pub const F8E4M3_MAX: f32 = 448.0;
/// int8 symmetric range.
pub const I8_MAX: f32 = 127.0;

/// Dynamic per-tensor **fp8 E4M3** activation quant (v1, pure candle ops — a fused amax→scale→cast
/// kernel is a later optimization if the bench shows it dominates). `scale = absmax / 448`, then
/// `q = cast_f8e4m3(x / scale)`.
///
/// **The final `f32 → F8E4M3` cast is done on the CPU** and the result moved back to `x`'s device.
/// candle's CUDA fp8-cast kernel (`cast_f32_f8_e4m3`) is `#if __CUDA_ARCH__ >= 800`-gated in
/// candle-kernels and is **not present** in the repo's cap=80 fatbin on this sm_120 box (the cast
/// dispatch fails with `CUDA_ERROR_NOT_FOUND "named symbol not found"` — sc-7544 territory, but for a
/// dense-PTX cast rather than a quant matmul). The CPU cast is byte-identical (the `float8` crate does
/// the same E4M3 rounding on host) so it is a correctness-neutral workaround; wiring the device cast
/// (re-vendor candle-kernels with the fp8 casts in the multi-arch fatbin) is a perf follow-up (sc-9299
/// note → sc-9300). The divide/amax stay on-device; only the tiny elementwise cast round-trips.
pub fn quantize_activation_fp8(x: &Tensor) -> Result<QuantizedActivation> {
    let absmax = x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?;
    let scale = (absmax.to_scalar::<f32>()? / F8E4M3_MAX).max(f32::MIN_POSITIVE);
    let scaled = (x.to_dtype(DType::F32)? / scale as f64)?;
    let q = scaled
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F8E4M3)?
        .to_device(x.device())?;
    Ok(QuantizedActivation { q, scale })
}

/// Dynamic per-tensor **int8** activation quant (v1). `scale = absmax / 127`, `q = round(x/scale)`
/// clamped to `[-127, 127]`, returned in `F32` (the rounded codes).
pub fn quantize_activation_int8(x: &Tensor) -> Result<QuantizedActivation> {
    let absmax = x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?;
    let scale = (absmax.to_scalar::<f32>()? / I8_MAX).max(f32::MIN_POSITIVE);
    let scaled = (x.to_dtype(DType::F32)? / scale as f64)?;
    let q = scaled.round()?.clamp(-I8_MAX, I8_MAX)?;
    Ok(QuantizedActivation { q, scale })
}

/// Quantize a weight `(N, K)` to fp8 E4M3 once (per-tensor), returning the packed tensor + scale.
/// Weight quant is static (done at load), so this is the load-time twin of [`quantize_activation_fp8`].
pub fn quantize_weight_fp8(w: &Tensor) -> Result<QuantizedActivation> {
    quantize_activation_fp8(w)
}

/// Quantize a weight `(N, K)` to int8 (per-tensor), returning rounded codes (`F32`) + scale.
pub fn quantize_weight_int8(w: &Tensor) -> Result<QuantizedActivation> {
    quantize_activation_int8(w)
}

/// Result of a **per-output-channel** int8 weight quantization: the int8 codes (`F32`) plus a `[out]`
/// scale vector such that `dequant[o, :] = q[o, :] * scale[o]`. This is the granularity a community
/// INT8-ConvRot checkpoint stores (`{base}.weight_scale`, `[out, 1]` f32), and the strict superset of
/// the per-tensor path (a per-tensor scale is a per-channel vector with one distinct value).
pub struct PerChannelInt8Weight {
    /// `(N, K)` int8 codes carried in `F32` (rounded, clamped to `[-127, 127]`).
    pub q: Tensor,
    /// `[N]` (per-output-row) dequant scale, `absmax_row / 127`.
    pub scale: Vec<f32>,
}

/// Dynamic **per-output-channel** int8 weight quant — `scale[o] = absmax(row o) / 127`,
/// `q[o, :] = round(w[o, :] / scale[o])` clamped to `[-127, 127]`. The candle-side dequant fold in
/// `CublasLt::matmul_int8_per_channel` applies this `[out]` vector to the int32 accumulator.
///
/// This is the load-time twin of a ConvRot checkpoint's stored per-row weight scale; a loader that
/// already has the on-disk `{base}.weight_scale` and int8 `{base}.weight` skips this and builds the
/// [`PerChannelInt8Weight`] directly from the parts (no re-quantization). Present so an int8 tier can
/// be produced from a dense weight for tests / a from-dense fold.
pub fn quantize_weight_int8_per_channel(w: &Tensor) -> Result<PerChannelInt8Weight> {
    let (n, _k) = w.dims2()?;
    let absmax = w.abs()?.max(1)?.to_dtype(DType::F32)?; // [N]
    let absmax_v = absmax.to_vec1::<f32>()?;
    let scale: Vec<f32> = absmax_v
        .iter()
        .map(|&a| (a / I8_MAX).max(f32::MIN_POSITIVE))
        .collect();
    let scale_col = Tensor::from_vec(scale.clone(), (n, 1), w.device())?;
    let q = w
        .to_dtype(DType::F32)?
        .broadcast_div(&scale_col)?
        .round()?
        .clamp(-I8_MAX, I8_MAX)?;
    Ok(PerChannelInt8Weight { q, scale })
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::super::nvfp4::{Nvfp4Tensor, SF_ATOM_COLS, SF_ATOM_ROWS};
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, Storage};
    use cudarc::cublaslt::{result as lt, sys};
    use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::sync::Arc;

    /// A cuBLASLt handle bound to a candle CUDA device, plus a reusable workspace. Cheap to build
    /// per matmul in a spike/bench; a real integration caches one per device.
    pub struct CublasLt {
        handle: sys::cublasLtHandle_t,
        stream: Arc<CudaStream>,
        device: candle_core::CudaDevice,
        workspace: cudarc::driver::CudaSlice<u8>,
        workspace_size: usize,
        /// Cached NVFP4 algo per `(m, k, n)` — the cuBLASLt heuristic search is a host-side cost that
        /// dominates a small FP4 GEMM; picking the algo once per shape and reusing it is what makes the
        /// resident-weight forward (and the throughput probe) reflect real FP4 tensor-core speed rather
        /// than repeated heuristic searches. The fp8/int8 paths do not cache (they were spike/bench
        /// scaffolding); the NVFP4 path is a shipping compute lane so it does.
        nvfp4_algos: std::sync::Mutex<
            std::collections::HashMap<(usize, usize, usize), sys::cublasLtMatmulAlgo_t>,
        >,
        /// Cached device-resident **gather index** (`U32`) for the on-device NVFP4 activation
        /// quantizer (sc-11044), keyed by `(rows, n_blocks)`. For each byte offset in cuBLASLt's
        /// row-major scale-factor-atom layout it holds the source index into the row-major
        /// `(row, 16-block)` scales — the inverse of the permutation `cublaslt_scale_layout` applies on
        /// the host for a staged weight. Built once per activation shape (a small host loop) and reused so
        /// the per-forward activation quantize is pure on-device tensor math + one cached gather
        /// (`index_select`), never a host round-trip or an atomic scatter (sc-12207).
        nvfp4_act_scale_gather_idx:
            std::sync::Mutex<std::collections::HashMap<(usize, usize), Tensor>>,
        /// Compiled + cached fused NVFP4 activation-quantize kernels (sc-12078). The outer `Option` is
        /// "have we tried to compile?"; the inner is `Some(functions)` on success or `None` on a compile
        /// failure. nvrtc compilation is a one-time host cost, so success is held for the handle's life —
        /// **and a failure is cached too**, so the `forward_fp4` fused→unfused fallback does not
        /// re-attempt (and re-fail) nvrtc on every projection of every denoise step.
        nvfp4_quant_kernels: std::sync::Mutex<Option<Option<Arc<Nvfp4QuantKernels>>>>,
    }

    /// The fused NVFP4 activation-quantize CUDA source (sc-12078), compiled once via nvrtc. Uses only
    /// standard device intrinsics (no fp8/fp4 headers) so nvrtc needs no extra include path.
    const NVFP4_QUANT_CU: &str = include_str!("nvfp4_quant.cu");

    /// The two nvrtc-compiled kernels of the fused activation quantizer, cached on the handle. The
    /// `CudaFunction`s keep their owning module alive. Force `Send`/`Sync` to match [`CublasLt`]'s own
    /// contract (the handle is used single-threaded per `RUST_TEST_THREADS=1`).
    struct Nvfp4QuantKernels {
        amax: cudarc::driver::CudaFunction,
        pack: cudarc::driver::CudaFunction,
    }
    unsafe impl Send for Nvfp4QuantKernels {}
    unsafe impl Sync for Nvfp4QuantKernels {}

    // The Lt handle is an opaque device-side object; guarded by the owning stream/device.
    unsafe impl Send for CublasLt {}
    unsafe impl Sync for CublasLt {}

    impl Drop for CublasLt {
        fn drop(&mut self) {
            unsafe {
                let _ = lt::destroy_handle(self.handle);
            }
        }
    }

    impl CublasLt {
        /// 32 MiB workspace — enough for the Split-K / stream-K algos cuBLASLt picks at DiT shapes.
        const WORKSPACE: usize = 32 * 1024 * 1024;

        pub fn new(dev: &Device) -> Result<Self> {
            let device = match dev {
                Device::Cuda(c) => c.clone(),
                _ => candle_core::bail!("CublasLt::new requires a CUDA device"),
            };
            let stream = device.cuda_stream();
            let handle = lt::create_handle().map_err(cublas_err)?;
            let workspace = stream.alloc_zeros::<u8>(Self::WORKSPACE).map_err(drv_err)?;
            Ok(Self {
                handle,
                stream,
                device,
                workspace,
                workspace_size: Self::WORKSPACE,
                nvfp4_algos: std::sync::Mutex::new(std::collections::HashMap::new()),
                nvfp4_act_scale_gather_idx: std::sync::Mutex::new(std::collections::HashMap::new()),
                nvfp4_quant_kernels: std::sync::Mutex::new(None),
            })
        }

        /// Compute capability of the bound device as `(major, minor)` — the sm_89 eligibility gate
        /// (locked-decision-7) reads this. For the worker, this is queryable straight off the candle
        /// device via `CudaDevice::cuda_stream().context().attribute(..)` (see module writeup).
        pub fn compute_cap(&self) -> Result<(i32, i32)> {
            let ctx = self.stream.context();
            let major = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(drv_err)?;
            let minor = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(drv_err)?;
            Ok((major, minor))
        }

        /// True iff the device meets the sc-9299 sm_89 floor (fp8 needs 8.9; int8 IGEMM 8.0, but the
        /// whole 8-bit track is floored at 8.9 per locked-decision-7).
        pub fn meets_fp8_floor(&self) -> Result<bool> {
            let (maj, min) = self.compute_cap()?;
            Ok(maj > 8 || (maj == 8 && min >= 9))
        }

        /// True iff the device can run the NVFP4 block-scaled FP4 GEMM (sc-11039). The master-gate
        /// spike (sc-11038) confirmed cuBLASLt dispatches a real `CUDA_R_4F_E2M1` + `VEC16_UE4M3`
        /// tensor-core kernel on **consumer** Blackwell `sm_120` (cap 12.0) — plain `sm_120`, not
        /// `sm_120a`. Datacenter `sm_100` is out of epic scope; the floor is cap ≥ 12.0. Below this the
        /// caller falls back to the dequant→bf16 dense path (the same fallback the non-cuda build takes
        /// by construction — this whole handle is `#[cfg(feature = "cuda")]`).
        pub fn meets_nvfp4_floor(&self) -> Result<bool> {
            let (maj, _min) = self.compute_cap()?;
            Ok(maj >= 12)
        }

        /// **fp8 E4M3 GEMM**: `D = (scale_w·scale_x) · (X · Wᵀ)` → bf16 `(M, N)`.
        /// `w_fp8` is `(N, K)` `F8E4M3`, `x_fp8` is `(M, K)` `F8E4M3`, both contiguous. Stages both
        /// operands then delegates to [`Self::matmul_fp8_staged`].
        pub fn matmul_fp8(
            &self,
            w_fp8: &Tensor,
            scale_w: f32,
            x_fp8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let a = DevFp8::stage(self, w_fp8)?;
            let b = DevFp8::stage(self, x_fp8)?;
            self.matmul_fp8_staged(&a, scale_w, &b, scale_x)
        }

        /// **NVFP4 block-scaled FP4 GEMM** (sc-11039, epic 11037) — the primary FP4 compute path.
        /// `D = (α)·(X · Wᵀ)` → bf16 `(M, N)`, where `α = w.global_scale · x.global_scale` and the two
        /// operands are packed NVFP4 ([`super::super::nvfp4::Nvfp4Tensor`], sc-11040): E2M1 nibbles +
        /// per-16-block UE4M3 micro-scales (in cuBLASLt's `VEC16_UE4M3` 128×4-swizzled layout) + the
        /// FP32 per-tensor scale folded into `alpha`.
        ///
        /// `w` is the packed weight `[N=out, K=in]`; `x` is the packed activation `[M=tokens, K=in]`
        /// (W4A4). Both must share the padded contraction width `cols_padded`. Delegates to
        /// [`Self::matmul_nvfp4_staged`] after uploading both operands.
        ///
        /// This is the recipe the master-gate spike (sc-11038) proved dispatches a real FP4
        /// tensor-core kernel on `sm_120`: `CUDA_R_4F_E2M1` A/B operands, `CUBLAS_COMPUTE_32F`
        /// accumulate, `CUDA_R_32F` scale type, `VEC16_UE4M3` A/B block-scale mode, bf16 output, the
        /// per-tensor scale in `alpha` (`D_SCALE_POINTER` is `NOT_SUPPORTED` for a bf16 D there).
        pub fn matmul_nvfp4(&self, w: &Nvfp4Tensor, x: &Nvfp4Tensor) -> Result<Tensor> {
            let a = DevNvfp4::stage(self, w)?;
            let b = DevNvfp4::stage(self, x)?;
            self.matmul_nvfp4_staged(&a, &b)
        }

        /// NVFP4 FP4 GEMM over **pre-staged** device operands (weight staged at load, activation per
        /// forward) — the honest resident-weight compute path the throughput probe times. Returns the
        /// bf16 `(M, N)` product. See [`Self::matmul_nvfp4`] for the recipe.
        pub fn matmul_nvfp4_staged(&self, w: &DevNvfp4, x: &DevNvfp4) -> Result<Tensor> {
            let (m, n) = (x.rows, w.rows);
            let k = w.cols_padded;
            if x.cols_padded != w.cols_padded {
                candle_core::bail!(
                    "nvfp4 staged K mismatch: x.cols_padded={} w.cols_padded={}",
                    x.cols_padded,
                    w.cols_padded
                );
            }
            check_nvfp4_alignment(k, n)?;
            // The FP32 per-tensor scales of both operands fold into a single `alpha` (the master-gate
            // spike wrinkle: `D_SCALE_POINTER` is `NOT_SUPPORTED` for a bf16 output).
            let alpha = w.global_scale * x.global_scale;
            let mut out = unsafe { self.stream.alloc::<half::bf16>(m * n) }.map_err(drv_err)?;
            {
                let (a_ptr, _ga) = w.packed.device_ptr(&self.stream);
                let (b_ptr, _gb) = x.packed.device_ptr(&self.stream);
                let (sa_ptr, _gsa) = w.scales.device_ptr(&self.stream);
                let (sb_ptr, _gsb) = x.scales.device_ptr(&self.stream);
                let (d_ptr, _gd) = out.device_ptr_mut(&self.stream);
                unsafe {
                    self.run_nvfp4(m, k, n, a_ptr, b_ptr, sa_ptr, sb_ptr, d_ptr, alpha)?;
                }
            }
            let storage = CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: self.device.clone(),
            };
            Ok(Tensor::from_storage(
                Storage::Cuda(storage),
                (m, n),
                BackpropOp::none(),
                false,
            ))
        }

        /// Stage a packed [`Nvfp4Tensor`] (host) as owned device buffers (nibbles + swizzled UE4M3
        /// scales) for repeated FP4 GEMMs — the load-time twin for a resident NVFP4 weight.
        pub fn stage_nvfp4(&self, t: &Nvfp4Tensor) -> Result<DevNvfp4> {
            DevNvfp4::stage(self, t)
        }

        /// **On-device NVFP4 activation quantize (sc-11044)** — the unfused **reference** quantizer.
        ///
        /// # Not a production path (sc-12078 fallback policy)
        ///
        /// This is the readable candle-op spelling of the NVFP4 activation recipe, and it is the
        /// **parity oracle** [`Self::quantize_nvfp4_activation_fused`] is gated against — the fused
        /// kernel is a hand-written two-pass CUDA reimplementation of exactly this, and the only thing
        /// keeping it honest is that the two agree bit-for-bit. That is what this function is for now.
        ///
        /// It is **not** the W4A4 forward's fallback and must not be reinstated as one. At ~19 ms per
        /// projection against the fused kernel's ~0.38 ms (K=6144, M=4118), serving W4A4 through it
        /// measured **0.01× vs dense bf16** end-to-end — ~100× slower than simply not using NVFP4.
        /// `Nvfp4Linear` therefore gates W4A4 on the fused kernel compiling and falls back to W4A16
        /// (~1.00×) when it does not, rather than routing here. See [`Nvfp4Regime::DequantBf16`].
        ///
        /// [`Nvfp4Regime::DequantBf16`]: super::super::nvfp4_linear::Nvfp4Regime::DequantBf16
        ///
        /// # Recipe
        ///
        /// Quantizes a device activation `x = [M, K]` (bf16/f32, already M-row-padded by the caller)
        /// to a packed [`DevNvfp4`] operand — E2M1 nibbles + per-16-block UE4M3 scales (in cuBLASLt's
        /// row-major scale-factor-atom layout) + the FP32 per-tensor scale — **entirely on the GPU**,
        /// feeding [`Self::matmul_nvfp4_staged`] directly. This replaces sc-11041's per-forward CPU
        /// round-trip (`Nvfp4Tensor::pack`, which copied the activation host-side, quantized in a scalar
        /// loop, and re-uploaded); here the only host transfer is the single per-tensor amax scalar
        /// (needed as the GEMM `alpha`), not the activation data.
        ///
        /// `cols_padded` is the weight's padded contraction width (a multiple of 16, and of
        /// [`NVFP4_K_ALIGN`]); `K` is padded up to it with zero columns so the staged operand matches
        /// the resident weight. The numeric recipe mirrors [`Nvfp4Tensor::pack_from_slice`] exactly
        /// (per-tensor `amax/(6·448)` global scale; per-block UE4M3 micro-scale; nearest-E2M1 element
        /// codes) so the on-device result tracks the CPU packer within a tiny rel-RMS. The quantizer is
        /// **NaN-free by construction**: E2M1 saturates at ±6 and every divisor is clamped positive, so
        /// an all-zero or outlier-carrying block yields finite codes (the W4A4 risk is signal collapse
        /// over steps, not a quantizer NaN — spike sc-11038).
        pub fn quantize_nvfp4_activation(&self, x: &Tensor, cols_padded: usize) -> Result<DevNvfp4> {
            use super::super::nvfp4::{E2M1_MAX, E4M3_MAX, NVFP4_BLOCK};
            let dev = Device::Cuda(self.device.clone());
            let (m, k) = x.dims2()?;
            if !cols_padded.is_multiple_of(NVFP4_BLOCK) || cols_padded < k {
                candle_core::bail!(
                    "quantize_nvfp4_activation: cols_padded ({cols_padded}) must be >= K ({k}) and a \
                     multiple of {NVFP4_BLOCK}"
                );
            }
            let x = x.to_dtype(DType::F32)?.contiguous()?;
            // Pad K -> cols_padded with zero columns (they carry no signal and don't perturb any
            // block amax; matches the packer's K-padding policy).
            let x = if cols_padded > k {
                x.pad_with_zeros(1, 0, cols_padded - k)?
            } else {
                x
            };
            let kp = cols_padded;
            let n_blocks = kp / NVFP4_BLOCK;

            // Per-tensor amax -> global scale. One scalar leaves the device (the GEMM alpha needs it on
            // the host); the activation tensor itself never does.
            let amax = x.abs()?.max_all()?.to_scalar::<f32>()?;
            let global_scale = if amax > 0.0 {
                amax / (E2M1_MAX * E4M3_MAX)
            } else {
                1.0
            };

            // Per-block amax over the 16-element blocks along K.
            let xb = x.reshape((m, n_blocks, NVFP4_BLOCK))?;
            let block_amax = xb.abs()?.max_keepdim(2)?; // [m, n_blocks, 1]

            // UE4M3 block micro-scale that maps the block amax -> 6.0 relative to the per-tensor scale,
            // then round it onto the E4M3 grid (arithmetic, matching OCP E4M3 nearest).
            let sf_real = block_amax.affine(1.0 / (E2M1_MAX * global_scale) as f64, 0.0)?;
            let sf_dec = e4m3_round_tensor(&sf_real)?; // decoded E4M3 block scale (on-grid), [m,n_blocks,1]

            // Per-element E2M1 codes: value / (block_scale · global_scale), nearest E2M1, sign in bit 3.
            let elem_scale = sf_dec.affine(global_scale as f64, 0.0)?;
            let elem_scale = elem_scale.clamp(f32::MIN_POSITIVE as f64, f64::INFINITY)?; // all-zero block -> 0/eps = 0
            let ratio = xb.broadcast_div(&elem_scale)?; // [m, n_blocks, 16]
            let code = e2m1_code_tensor(&ratio)?; // f32 codes 0..15, [m, n_blocks, 16]

            // Pack two E2M1 codes per byte, little-endian nibble order (col 2j -> low, 2j+1 -> high).
            let code = code.reshape((m, kp))?.reshape((m, kp / 2, 2))?;
            let low = code.narrow(2, 0, 1)?;
            let high = code.narrow(2, 1, 1)?;
            let byte = low.add(&high.affine(16.0, 0.0)?)?.reshape((m, kp / 2))?;
            let packed = extract_u8_slice(&byte)?; // CudaSlice<u8>, len m*kp/2

            // UE4M3 scale BYTES from the on-grid decoded value (exact inverse of the E4M3 decode). Gather
            // them into cuBLASLt's row-major scale-factor-atom layout via the cached inverse-permutation
            // index. A single zero is appended to the source so padding atoms gather 0 (matching the CPU
            // packer's zeroed padding). This swizzle is a pure bijection — the former `scatter_add` did it
            // with atomic accumulation (250 ms vs 0.04 ms on Krea shapes; sc-12207).
            let sf_byte = e4m3_byte_from_decoded(&sf_dec)?.reshape((m * n_blocks,))?;
            let sf_byte = Tensor::cat(&[&sf_byte, &Tensor::zeros(1, DType::F32, &dev)?], 0)?;
            let gather_idx = self.nvfp4_act_scale_gather_index(m, n_blocks, &dev)?;
            let swz = sf_byte.index_select(&gather_idx, 0)?;
            let scales = extract_u8_slice(&swz)?;

            Ok(DevNvfp4 {
                packed,
                scales,
                rows: m,
                cols_padded: kp,
                global_scale,
            })
        }

        /// Build (or fetch the cached) device `U32` **gather** index that permutes the row-major
        /// activation block scales into cuBLASLt's row-major scale-factor-atom layout, i.e.
        /// `out[dst] = sf_byte_padded[idx[dst]]`. `sf_byte_padded` is the logical `(row, 16-block)` scale
        /// in `row*n_blocks + blk` source order with a single zero appended at index `rows*n_blocks`.
        /// This is the **inverse** of the host-side [`cublaslt_scale_layout`] swizzle (intra-atom
        /// `((32,4),4):((16,4),1)`, **row-major** atom tiling `atom = k_atom + num_k_atoms · m_atom`);
        /// padding atoms (rows ≥ M or blocks ≥ n_blocks) index the appended zero so they decode to 0,
        /// matching the packer's zeroed padding. A gather over this permutation replaces the former
        /// `scatter_add` — a bijection wrongly implemented with atomic accumulation (sc-12207). Cached per
        /// `(rows, n_blocks)`.
        fn nvfp4_act_scale_gather_index(
            &self,
            rows: usize,
            n_blocks: usize,
            dev: &Device,
        ) -> Result<Tensor> {
            if let Some(t) = crate::lock_recover(&self.nvfp4_act_scale_gather_idx)
                .get(&(rows, n_blocks))
                .cloned()
            {
                return Ok(t);
            }
            let sf_rows = round_up_usize(rows, SF_ATOM_ROWS);
            let sf_cols = round_up_usize(n_blocks, SF_ATOM_COLS);
            let scales_len = sf_rows * sf_cols;
            let num_k_atoms = sf_cols / SF_ATOM_COLS;
            // Padding atoms gather the appended zero at source index `rows*n_blocks`.
            let sentinel = (rows * n_blocks) as u32;
            let mut idx = vec![sentinel; scales_len];
            for r in 0..rows {
                let m_atom = r / SF_ATOM_ROWS;
                let mr = r % SF_ATOM_ROWS;
                for blk in 0..n_blocks {
                    let k_atom = blk / SF_ATOM_COLS;
                    let kc = blk % SF_ATOM_COLS;
                    let atom_index = k_atom + num_k_atoms * m_atom;
                    let intra = (mr % 32) * 16 + (mr / 32) * 4 + kc;
                    let dst = atom_index * (SF_ATOM_ROWS * SF_ATOM_COLS) + intra;
                    idx[dst] = (r * n_blocks + blk) as u32;
                }
            }
            let t = Tensor::from_vec(idx, scales_len, dev)?;
            crate::lock_recover(&self.nvfp4_act_scale_gather_idx).insert((rows, n_blocks), t.clone());
            Ok(t)
        }

        /// Set (to anything) to make [`Self::nvfp4_fused_quantizer_available`] report `false` without
        /// breaking the real nvrtc install — the fused kernel is then treated as uncompilable.
        ///
        /// **This exists so the fallback policy is testable.** A genuine nvrtc failure cannot be
        /// induced on a healthy rig, so the W4A4→W4A16 capability gate would otherwise be a branch
        /// that ships having never once been executed — the same class of defect (an unobserved
        /// fallback) that the gate itself was written to remove. `nvfp4_fused_unavailable_forces_w4a16`
        /// drives the gate through this. It is read inside the compile closure, so a forced failure is
        /// cached exactly like a real one and exercises the same code path.
        pub const NVFP4_FORCE_NO_FUSED_QUANT_ENV: &str = "SC12078_DISABLE_FUSED_QUANT";

        /// Compile (once) and fetch the fused NVFP4 activation-quantize kernels (sc-12078). nvrtc turns
        /// [`NVFP4_QUANT_CU`] into a module JITed for the live device; the two functions are cached so
        /// the compile is paid once per handle, not per forward.
        fn nvfp4_quant_kernels(&self) -> Result<Arc<Nvfp4QuantKernels>> {
            let mut guard = crate::lock_recover(&self.nvfp4_quant_kernels);
            match guard.as_ref() {
                Some(Some(k)) => return Ok(k.clone()),
                Some(None) => {
                    candle_core::bail!("sc-12078 fused NVFP4 quant kernels previously failed to compile")
                }
                None => {}
            }
            let compiled = (|| -> Result<Arc<Nvfp4QuantKernels>> {
                if std::env::var_os(Self::NVFP4_FORCE_NO_FUSED_QUANT_ENV).is_some() {
                    candle_core::bail!(
                        "sc-12078 fused NVFP4 quant kernels disabled by ${} (test seam)",
                        Self::NVFP4_FORCE_NO_FUSED_QUANT_ENV
                    );
                }
                let ctx = self.stream.context();
                let ptx = cudarc::nvrtc::compile_ptx(NVFP4_QUANT_CU).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "sc-12078 nvrtc compile of nvfp4_quant.cu failed: {e}"
                    ))
                })?;
                let module = ctx.load_module(ptx).map_err(drv_err)?;
                let amax = module
                    .load_function("nvfp4_block_amax_f32")
                    .map_err(drv_err)?;
                let pack = module.load_function("nvfp4_pack_f32").map_err(drv_err)?;
                Ok(Arc::new(Nvfp4QuantKernels { amax, pack }))
            })();
            // Cache the outcome either way — a failed compile must not be retried per forward.
            *guard = Some(compiled.as_ref().ok().cloned());
            compiled
        }

        /// True iff the **fused** NVFP4 activation quantizer (sc-12078) is usable on this handle —
        /// i.e. its nvrtc compile succeeds, or already did. Compiles on first call and caches the
        /// outcome (success *and* failure), so repeat queries are free and calling it early costs
        /// nothing the first forward would not have paid anyway.
        ///
        /// **A capability gate, not a diagnostic.** [`Nvfp4Linear::from_packed`] consults this before
        /// selecting W4A4, alongside [`Self::meets_nvfp4_floor`], because W4A4 *without* the fused
        /// quantizer is not a regime worth running — see [`Nvfp4Regime::DequantBf16`] for the
        /// measurements behind that.
        ///
        /// [`Nvfp4Linear::from_packed`]: super::super::nvfp4_linear::Nvfp4Linear::from_packed
        /// [`Nvfp4Regime::DequantBf16`]: super::super::nvfp4_linear::Nvfp4Regime::DequantBf16
        pub fn nvfp4_fused_quantizer_available(&self) -> bool {
            self.nvfp4_quant_kernels().is_ok()
        }

        /// **Fused** on-device NVFP4 activation quantize (sc-12078) — the throughput replacement for
        /// [`Self::quantize_nvfp4_activation`]'s ~40-op candle chain. Two nvrtc kernels over the
        /// activation — (1) per-16-block amax + per-tensor amax, (2) E4M3 block scale + E2M1 codes +
        /// nibble pack + swizzle into the row-major 128×4 UE4M3 SF-atom layout — produce byte-identical
        /// [`DevNvfp4`] to the CPU packer, with the **single** unavoidable host sync (the per-tensor amax
        /// scalar that becomes the GEMM `alpha`). Same numeric recipe as [`Nvfp4Tensor::pack_from_slice`]
        /// and [`Self::quantize_nvfp4_activation`]; `cols_padded` matches the resident weight.
        pub fn quantize_nvfp4_activation_fused(
            &self,
            x: &Tensor,
            cols_padded: usize,
        ) -> Result<DevNvfp4> {
            use super::super::nvfp4::{E2M1_MAX, E4M3_MAX, NVFP4_BLOCK};
            use cudarc::driver::{LaunchConfig, PushKernelArg};

            let (m, k) = x.dims2()?;
            if !cols_padded.is_multiple_of(NVFP4_BLOCK) || cols_padded < k {
                candle_core::bail!(
                    "quantize_nvfp4_activation_fused: cols_padded ({cols_padded}) must be >= K ({k}) \
                     and a multiple of {NVFP4_BLOCK}"
                );
            }
            let kernels = self.nvfp4_quant_kernels()?;
            let n_blocks = cols_padded / NVFP4_BLOCK;
            let sf_rows = round_up_usize(m, SF_ATOM_ROWS);
            let sf_cols = round_up_usize(n_blocks, SF_ATOM_COLS);
            let total = m * n_blocks;

            // The activation as a contiguous f32 device slice — the kernels read it directly (K padding
            // is handled in-kernel by bounds-checking against the real K, so no `pad_with_zeros`).
            let xf = x.to_dtype(DType::F32)?.contiguous()?;
            let (x_storage, _xl) = xf.storage_and_layout();
            let x_slice = match &*x_storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::F32(s) => s,
                    _ => candle_core::bail!("fused quantize: expected F32 CUDA storage"),
                },
                _ => candle_core::bail!("fused quantize: activation is not on CUDA"),
            };

            // Outputs + scratch. `packed`/`scales` are zeroed so K-padding nibbles and padding scale
            // atoms (rows ≥ m or blocks ≥ n_blocks, up to the 128×4 atom) stay 0 — matching the packer.
            let mut block_amax = self.stream.alloc_zeros::<f32>(total).map_err(drv_err)?;
            let mut g_amax = self.stream.alloc_zeros::<f32>(1).map_err(drv_err)?;
            let mut packed = self
                .stream
                .alloc_zeros::<u8>(m * (cols_padded / 2))
                .map_err(drv_err)?;
            let mut scales = self.stream.alloc_zeros::<u8>(sf_rows * sf_cols).map_err(drv_err)?;

            let (mi, ki, nbi) = (m as i32, k as i32, n_blocks as i32);
            let (cpi, sci) = (cols_padded as i32, sf_cols as i32);
            let block = 256u32;
            let grid = (total as u32).div_ceil(block);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };

            // Pass 1: per-block amax + per-tensor amax (atomicMax into a pre-zeroed scalar).
            unsafe {
                self.stream
                    .launch_builder(&kernels.amax)
                    .arg(x_slice)
                    .arg(&mut block_amax)
                    .arg(&mut g_amax)
                    .arg(&mi)
                    .arg(&ki)
                    .arg(&nbi)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }

            // The one host sync: read the per-tensor amax scalar for the GEMM alpha (same global scale
            // recipe as the CPU packer). `clone_dtoh` orders after pass 1 on the stream.
            let amax = self.stream.clone_dtoh(&g_amax).map_err(drv_err)?[0];
            let global_scale = if amax > 0.0 {
                amax / (E2M1_MAX * E4M3_MAX)
            } else {
                1.0
            };

            // Pass 2: E4M3 block scale + E2M1 element codes + nibble pack + swizzled scale scatter.
            let gs = global_scale;
            unsafe {
                self.stream
                    .launch_builder(&kernels.pack)
                    .arg(x_slice)
                    .arg(&block_amax)
                    .arg(&mut packed)
                    .arg(&mut scales)
                    .arg(&mi)
                    .arg(&ki)
                    .arg(&nbi)
                    .arg(&cpi)
                    .arg(&sci)
                    .arg(&gs)
                    .launch(cfg)
                    .map_err(drv_err)?;
            }

            Ok(DevNvfp4 {
                packed,
                scales,
                rows: m,
                cols_padded,
                global_scale,
            })
        }

        /// **int8 IGEMM**: raw `int32 = X_i8 · W_i8ᵀ` (no scale folded — caller multiplies by
        /// `scale_w·scale_x`). `w_i8`/`x_i8` carry the rounded int8 codes in `F32`; narrowed to
        /// device `i8` here. Returns `(M, N)` on-device `I32`.
        pub fn matmul_int8_raw(&self, w_i8: &Tensor, x_i8: &Tensor) -> Result<Tensor> {
            let a = DevInt8::stage(self, w_i8)?; // A = W (N,K)
            let b = DevInt8::stage(self, x_i8)?; // B = X (M,K)
            self.matmul_int8_staged(&a, &b)
        }

        /// int8 IGEMM over **pre-staged** device operands (the honest compute path — no per-call host
        /// narrowing). Returns `(M, N)` on-device `I32`. This is what the bench times.
        pub fn matmul_int8_staged(&self, w: &DevInt8, x: &DevInt8) -> Result<Tensor> {
            let (m, k, n) = (x.rows, x.cols, w.rows);
            if x.cols != w.cols {
                candle_core::bail!(
                    "int8 staged shape mismatch: x=(.,{}) w=({},{})",
                    x.cols,
                    w.rows,
                    w.cols
                );
            }
            check_alignment(k, n)?;
            let mut out = self.stream.alloc_zeros::<i32>(m * n).map_err(drv_err)?;
            {
                let (a_ptr, _ga) = w.buf.device_ptr(&self.stream);
                let (b_ptr, _gb) = x.buf.device_ptr(&self.stream);
                let (d_ptr, _gd) = out.device_ptr_mut(&self.stream);
                unsafe {
                    self.run(
                        sys::cudaDataType_t::CUDA_R_8I,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                        sys::cudaDataType_t::CUDA_R_32I,
                        sys::cudaDataType_t::CUDA_R_32I,
                        m,
                        k,
                        n,
                        a_ptr,
                        b_ptr,
                        d_ptr,
                        None,
                        Alpha::I32(1),
                    )?;
                }
            }
            // sc-11260 (F-100): no per-call `stream.synchronize()` here. The GEMM enqueues on the
            // device stream and stream ordering already sequences every downstream consumer.
            // Correctness rests on two LOAD-BEARING invariants:
            //   (1) Same-stream ordering — the GEMM, the `out` allocation, and the host read-back
            //       copy all run on `self.stream` (= `device.cuda_stream()`), so the GEMM is
            //       ordered strictly before the read.
            //   (2) Blocking read-back — the host-fold paths
            //       (`matmul_int8[_per_channel[_staged]]`) read the accumulate back via `clone_dtoh`
            //       into a PAGEABLE (non-pinned) `Vec`. A device→pageable-host `cudaMemcpyAsync`
            //       blocks the host until the copy — hence the GEMM — completes, so no explicit sync
            //       is needed. The on-device fold (`matmul_int8_per_channel_staged_ondevice`) chains
            //       stream-ordered candle ops, and the fp8 twin (`matmul_fp8_staged`) is sync-free
            //       for the same reasons.
            // CAVEAT: this holds ONLY while read-backs land in pageable host memory. If a caller
            // ever switches the read-back to a PINNED host buffer, the blocking guarantee is lost
            // and an explicit stream sync (or event) would be required again — do not reintroduce a
            // race by that route. Draining the pipeline once per int8 projection defeated async
            // enqueue-ahead on the ConvRot resident forward.
            let storage = CudaStorage {
                slice: CudaStorageSlice::I32(out),
                device: self.device.clone(),
            };
            Ok(Tensor::from_storage(
                Storage::Cuda(storage),
                (m, n),
                BackpropOp::none(),
                false,
            ))
        }

        /// **On-device** per-output-channel int8 linear (sc-9601 perf) — the resident-weight ConvRot
        /// forward with the dequant fold kept entirely on the GPU. Runs the exact int32 IGEMM
        /// ([`Self::matmul_int8_staged`], `CudaStorageSlice::I32` on device), casts the accumulate `i32 →
        /// f32` **on-device** (`Tensor::to_dtype`, via the sc-9601 vendored `cast_i32_f32` kernel), then
        /// folds the per-row weight scale × per-tensor activation scale with a candle broadcast multiply
        /// and casts to bf16 — **no int32→host copy, no CPU fold**. This is the fast twin of
        /// [`Self::matmul_int8_per_channel_staged`] (which reads the int32 accumulate back to host because
        /// stock candle-kernels ships no CUDA `i32 → f32` cast; the vendored kernel closes that gap). The
        /// i32→f32 conversion is an exact hardware I2F (round-to-nearest, rel ≤ 6e-8 for the few outputs
        /// beyond 2²⁴ — negligible beside int8's ~1 %). `D[m, o] = (scale_w[o]·scale_x) · acc[m, o]` → bf16.
        pub fn matmul_int8_per_channel_staged_ondevice(
            &self,
            w: &DevInt8,
            scale_w: &[f32],
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let x = DevInt8::stage(self, x_i8)?;
            let n = w.rows;
            if scale_w.len() != n {
                candle_core::bail!(
                    "matmul_int8_per_channel_staged_ondevice: scale_w len {} != N (out) {n}",
                    scale_w.len()
                );
            }
            let acc = self.matmul_int8_staged(w, &x)?.to_dtype(DType::F32)?; // (M, N), on device
                                                                             // Fold the dequant scale on-device: per-row weight scale (a [1, N] row broadcast over M) times
                                                                             // the scalar activation scale. `scale_x` is folded into the row vector so it's one multiply.
            let row: Vec<f32> = scale_w.iter().map(|&s| s * scale_x).collect();
            let row = Tensor::from_vec(row, (1, n), acc.device())?;
            acc.broadcast_mul(&row)?.to_dtype(DType::BF16)
        }

        /// fp8 GEMM over a **pre-staged** device weight + activation (no per-call operand clone). The
        /// bench's honest fp8 compute path.
        pub fn matmul_fp8_staged(
            &self,
            w: &DevFp8,
            scale_w: f32,
            x: &DevFp8,
            scale_x: f32,
        ) -> Result<Tensor> {
            let (m, k, n) = (x.rows, x.cols, w.rows);
            if x.cols != w.cols {
                candle_core::bail!(
                    "fp8 staged shape mismatch: x=(.,{}) w=({},{})",
                    x.cols,
                    w.rows,
                    w.cols
                );
            }
            check_alignment(k, n)?;
            let mut out = unsafe { self.stream.alloc::<half::bf16>(m * n) }.map_err(drv_err)?;
            let sa = self.upload_scalar(scale_w)?;
            let sb = self.upload_scalar(scale_x)?;
            {
                let (a_ptr, _ga) = w.buf.device_ptr(&self.stream);
                let (b_ptr, _gb) = x.buf.device_ptr(&self.stream);
                let (sa_ptr, _gsa) = sa.device_ptr(&self.stream);
                let (sb_ptr, _gsb) = sb.device_ptr(&self.stream);
                let (d_ptr, _gd) = out.device_ptr_mut(&self.stream);
                unsafe {
                    self.run(
                        sys::cudaDataType_t::CUDA_R_8F_E4M3,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        sys::cudaDataType_t::CUDA_R_32F,
                        sys::cudaDataType_t::CUDA_R_16BF,
                        m,
                        k,
                        n,
                        a_ptr,
                        b_ptr,
                        d_ptr,
                        Some((sa_ptr, sb_ptr)),
                        Alpha::F32(1.0),
                    )?;
                }
            }
            let storage = CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: self.device.clone(),
            };
            Ok(Tensor::from_storage(
                Storage::Cuda(storage),
                (m, n),
                BackpropOp::none(),
                false,
            ))
        }

        /// Full int8 linear `D = (scale_w·scale_x)·(X·Wᵀ)` → bf16, folding the dequant scale on the
        /// candle side after the exact int32 accumulate.
        ///
        /// The int32 accumulate is read back to the host and the `scale_w·scale_x` fold + bf16 cast
        /// happen there: candle-kernels ships **no** CUDA `i32 → f32` cast (only `u32`/`i64` have full
        /// cast coverage), so an on-device `I32` tensor cannot be `to_dtype`'d. The staged
        /// [`Self::matmul_int8_staged`] keeps the accumulate on-device for the bench (which never
        /// casts it). The **on-device** fold that avoids this host round-trip entirely — via the
        /// f32-output IGEMM epilogue — is [`Self::matmul_int8_per_channel_staged_ondevice`] (sc-9601);
        /// this per-tensor convenience path keeps the host fold (it is not on a hot resident path).
        pub fn matmul_int8(
            &self,
            w_i8: &Tensor,
            scale_w: f32,
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let a = DevInt8::stage(self, w_i8)?;
            let b = DevInt8::stage(self, x_i8)?;
            let (m, n) = (b.rows, a.rows);
            let acc = self.matmul_int8_staged(&a, &b)?;
            // Read the on-device I32 accumulate back to host (no CUDA i32 cast exists), fold the
            // dequant scale, upload as bf16.
            let (storage, _l) = acc.storage_and_layout();
            let host_i32: Vec<i32> = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::I32(s) => self.stream.clone_dtoh(s).map_err(drv_err)?,
                    _ => candle_core::bail!("matmul_int8: expected I32 accumulate"),
                },
                _ => candle_core::bail!("matmul_int8: accumulate not on CUDA"),
            };
            let s = scale_w * scale_x;
            let host_f32: Vec<f32> = host_i32.iter().map(|&v| v as f32 * s).collect();
            Tensor::from_vec(host_f32, (m, n), acc.device())?.to_dtype(DType::BF16)
        }

        /// Full int8 linear with a **per-output-channel** weight scale — the community INT8-ConvRot
        /// dequant (sc-9300): `D[m, o] = (scale_w[o] · scale_x) · (X_i8 · W_i8ᵀ)[m, o]` → bf16. The
        /// `[N]` `scale_w` is the checkpoint's stored per-row `{base}.weight_scale`; `scale_x` is the
        /// dynamic per-tensor activation scale. A strict superset of [`Self::matmul_int8`] (pass an
        /// all-equal `scale_w` to recover the per-tensor fold). Like [`Self::matmul_int8`] the int32
        /// accumulate is read back to host for the fold + bf16 cast (candle-kernels ships no CUDA
        /// `i32 → f32` cast), so the exact accumulate is preserved and only the scale application
        /// differs.
        ///
        /// This is the exact `X·Wᵀ` compute for a per-channel-quantized int8 weight. For a ConvRot
        /// checkpoint the stored `W_i8` is the *rotated* weight `W·R`, so the consume path applies the
        /// matching online activation rotation `RHT(x)` ([`super::super::convrot`], sc-9601) before this call,
        /// making `RHT(x)·(W·R)ᵀ = x·Wᵀ`. The compute here is rotation-agnostic and correct either way.
        pub fn matmul_int8_per_channel(
            &self,
            w_i8: &Tensor,
            scale_w: &[f32],
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let a = DevInt8::stage(self, w_i8)?;
            let b = DevInt8::stage(self, x_i8)?;
            let (m, n) = (b.rows, a.rows);
            if scale_w.len() != n {
                candle_core::bail!(
                    "matmul_int8_per_channel: scale_w len {} != N (out) {n}",
                    scale_w.len()
                );
            }
            let acc = self.matmul_int8_staged(&a, &b)?;
            let (storage, _l) = acc.storage_and_layout();
            let host_i32: Vec<i32> = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::I32(s) => self.stream.clone_dtoh(s).map_err(drv_err)?,
                    _ => candle_core::bail!("matmul_int8_per_channel: expected I32 accumulate"),
                },
                _ => candle_core::bail!("matmul_int8_per_channel: accumulate not on CUDA"),
            };
            // Row-major `(M, N)`: element `(row, col)` dequants by `scale_w[col] · scale_x`.
            let host_f32: Vec<f32> = host_i32
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * scale_w[i % n] * scale_x)
                .collect();
            Tensor::from_vec(host_f32, (m, n), acc.device())?.to_dtype(DType::BF16)
        }

        /// Per-output-channel int8 over a **pre-staged** device weight (sc-9300) — the resident-weight
        /// form the ConvRot consume path uses so the `(N, K)` int8 codes live on-device as native `i8`
        /// (1 byte/elem) rather than as an 8×-larger I64 tensor. `w` is the staged weight, `scale_w` its
        /// `[N]` per-row dequant scale, `x_i8`/`scale_x` the dynamically-quantized activation. Same fold
        /// as [`Self::matmul_int8_per_channel`], only the weight is not re-staged per call. This is the
        /// **exact host fold** (int32→host); the on-device fast twin
        /// [`Self::matmul_int8_per_channel_staged_ondevice`] (sc-9601) folds on-device via the f32-output
        /// IGEMM and is the path the ConvRot resident forward uses when the device supports it.
        pub fn matmul_int8_per_channel_staged(
            &self,
            w: &DevInt8,
            scale_w: &[f32],
            x_i8: &Tensor,
            scale_x: f32,
        ) -> Result<Tensor> {
            let x = DevInt8::stage(self, x_i8)?;
            let (m, n) = (x.rows, w.rows);
            if scale_w.len() != n {
                candle_core::bail!(
                    "matmul_int8_per_channel_staged: scale_w len {} != N (out) {n}",
                    scale_w.len()
                );
            }
            let acc = self.matmul_int8_staged(w, &x)?;
            let (storage, _l) = acc.storage_and_layout();
            let host_i32: Vec<i32> = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::I32(s) => self.stream.clone_dtoh(s).map_err(drv_err)?,
                    _ => candle_core::bail!(
                        "matmul_int8_per_channel_staged: expected I32 accumulate"
                    ),
                },
                _ => candle_core::bail!("matmul_int8_per_channel_staged: accumulate not on CUDA"),
            };
            let host_f32: Vec<f32> = host_i32
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * scale_w[i % n] * scale_x)
                .collect();
            Tensor::from_vec(host_f32, (m, n), acc.device())?.to_dtype(DType::BF16)
        }

        /// Probe (once) whether this device can cast an **int32 tensor to f32 on-device** — the capability
        /// the on-device dequant fast path (sc-9601) needs, provided by the vendored `cast_i32_f32` kernel.
        /// Stock candle-kernels omits the I32 source casts, so on a build without the vendored fork this
        /// returns `false` and the caller keeps the exact int32→host fold. Runs a tiny `i32 → f32`
        /// `to_dtype` and reports success; swallows errors deliberately (a probe, not a compute).
        pub fn supports_ondevice_int8_dequant(&self) -> bool {
            let dev = Device::Cuda(self.device.clone());
            let probe = || -> Result<()> {
                let dummy = Tensor::zeros((16, 16), DType::F32, &dev)?;
                let (a, b) = (DevInt8::stage(self, &dummy)?, DevInt8::stage(self, &dummy)?);
                let acc = self.matmul_int8_staged(&a, &b)?; // I32 on device
                acc.to_dtype(DType::F32)?; // the cast_i32_f32 kernel (fails if absent)
                Ok(())
            };
            probe().is_ok()
        }

        /// Stage an fp8 weight/activation tensor (`F8E4M3`) as an owned device buffer + its shape.
        pub fn stage_fp8(&self, t: &Tensor) -> Result<DevFp8> {
            DevFp8::stage(self, t)
        }
        /// Stage an int8-code tensor (`F32`/`I64`) as an owned device `i8` buffer + its shape.
        pub fn stage_int8(&self, t: &Tensor) -> Result<DevInt8> {
            DevInt8::stage(self, t)
        }

        // --- internals -------------------------------------------------------------------------

        #[allow(clippy::too_many_arguments)]
        unsafe fn run(
            &self,
            io_dtype: sys::cudaDataType_t,
            compute: sys::cublasComputeType_t,
            scale_dtype: sys::cudaDataType_t,
            out_dtype: sys::cudaDataType_t,
            m: usize,
            k: usize,
            n: usize,
            a_ptr: cudarc::driver::sys::CUdeviceptr,
            b_ptr: cudarc::driver::sys::CUdeviceptr,
            d_ptr: cudarc::driver::sys::CUdeviceptr,
            scales: Option<(
                cudarc::driver::sys::CUdeviceptr,
                cudarc::driver::sys::CUdeviceptr,
            )>,
            alpha: Alpha,
        ) -> Result<()> {
            let desc = lt::create_matmul_desc(compute, scale_dtype).map_err(cublas_err)?;
            // TN: op(A)=Aᵀ (transa=T), op(B)=B (transb=N) — the only fp8 layout cuBLASLt accepts and
            // the modern relaxed-TN int8 row-major path (no COL32). The transpose op is a raw i32
            // (`1 == T, 0 == N`) — `cublasOperation_t` lives in `cublas::sys`, not `cublaslt::sys`,
            // and cuBLASLt reads TRANSA/TRANSB as a 32-bit int (mirrors cudarc's own `set_transpose`).
            let transa: i32 = 1; // CUBLAS_OP_T
            let transb: i32 = 0; // CUBLAS_OP_N
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &transa,
            )?;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &transb,
            )?;

            if let Some((sa, sb)) = scales {
                set_attr(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &sa,
                )?;
                set_attr(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &sb,
                )?;
            }

            // Layouts (see module doc): A=(K,N) col-major ld=K, B=(K,M) col-major ld=K,
            // D=(N,M) col-major ld=N.
            let a_layout = lt::create_matrix_layout(io_dtype, k as u64, n as u64, k as i64)
                .map_err(cublas_err)?;
            let b_layout = lt::create_matrix_layout(io_dtype, k as u64, m as u64, k as i64)
                .map_err(cublas_err)?;
            let d_layout = lt::create_matrix_layout(out_dtype, n as u64, m as u64, n as i64)
                .map_err(cublas_err)?;

            let pref = lt::create_matmul_pref().map_err(cublas_err)?;
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &self.workspace_size as *const _ as *const c_void,
                size_of::<usize>(),
            )
            .map_err(cublas_err)?;

            let heuristic = lt::get_matmul_algo_heuristic(
                self.handle,
                desc,
                a_layout,
                b_layout,
                d_layout,
                d_layout,
                pref,
            )
            .map_err(cublas_err)?;

            let (alpha_ptr, beta_ptr): (*const c_void, *const c_void) = match &alpha {
                Alpha::F32(a) => (
                    a as *const f32 as *const c_void,
                    &F32_ZERO as *const f32 as *const c_void,
                ),
                Alpha::I32(a) => (
                    a as *const i32 as *const c_void,
                    &I32_ZERO as *const i32 as *const c_void,
                ),
            };

            let (ws_ptr, _gw) = self.workspace.device_ptr(&self.stream);

            let res = lt::matmul(
                self.handle,
                desc,
                alpha_ptr,
                beta_ptr,
                a_ptr as *const c_void,
                a_layout,
                b_ptr as *const c_void,
                b_layout,
                d_ptr as *const c_void,
                d_layout,
                d_ptr as *mut c_void,
                d_layout,
                &heuristic.algo as *const _,
                ws_ptr as *mut c_void,
                self.workspace_size,
                // driver::sys::CUstream and cublaslt::sys::cudaStream_t are both `*mut CUstream_st`
                // but distinct newtype aliases across the two sys modules — cast the raw pointer.
                self.stream.cu_stream() as sys::cudaStream_t,
            );

            let _ = lt::destroy_matrix_layout(a_layout);
            let _ = lt::destroy_matrix_layout(b_layout);
            let _ = lt::destroy_matrix_layout(d_layout);
            let _ = lt::destroy_matmul_pref(pref);
            let _ = lt::destroy_matmul_desc(desc);
            res.map_err(cublas_err)
        }

        fn upload_scalar(&self, v: f32) -> Result<cudarc::driver::CudaSlice<f32>> {
            self.stream.clone_htod(&[v]).map_err(drv_err)
        }

        /// The NVFP4 block-scaled FP4 matmul (sc-11039). Same TN col-major layout identity as
        /// [`Self::run`] (A=W `(K,N)` ld=K, B=X `(K,M)` ld=K, D `(N,M)` ld=N) — only the operand dtype
        /// (`CUDA_R_4F_E2M1`), the per-operand **block-scale mode** (`VEC16_UE4M3` + block-scale
        /// pointers, in place of the fp8 per-tensor SCALAR scale pointers), and `alpha` (the folded
        /// FP32 per-tensor scales) differ. Output bf16, `CUBLAS_COMPUTE_32F` / `CUDA_R_32F` scale.
        ///
        /// `sa_ptr` / `sb_ptr` are the A/B UE4M3 block-scale tensors in the CUTLASS
        /// `Sm1xxBlockScaledConfig` 128×4 swizzle the sc-11040 packer emits — cuBLASLt's `VEC16`
        /// UE4M3 mode consumes them directly (handoff item (a): the packer's column-major atom tiling
        /// is what this descriptor expects; validated on the live GPU by the round-trip test).
        #[allow(clippy::too_many_arguments)]
        unsafe fn run_nvfp4(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a_ptr: cudarc::driver::sys::CUdeviceptr,
            b_ptr: cudarc::driver::sys::CUdeviceptr,
            sa_ptr: cudarc::driver::sys::CUdeviceptr,
            sb_ptr: cudarc::driver::sys::CUdeviceptr,
            d_ptr: cudarc::driver::sys::CUdeviceptr,
            alpha: f32,
        ) -> Result<()> {
            let io_dtype = sys::cudaDataType_t::CUDA_R_4F_E2M1;
            let out_dtype = sys::cudaDataType_t::CUDA_R_16BF;
            let desc = lt::create_matmul_desc(
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType_t::CUDA_R_32F,
            )
            .map_err(cublas_err)?;

            let transa: i32 = 1; // CUBLAS_OP_T  (A = W declared (K,N) col-major == our row-major (N,K))
            let transb: i32 = 0; // CUBLAS_OP_N  (B = X declared (K,M) col-major == our row-major (M,K))
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &transa,
            )?;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &transb,
            )?;

            // Per-operand VEC16 UE4M3 block-scale mode + the swizzled scale tensors. The scale mode is
            // a `cublasLtMatmulMatrixScale_t` (u32); `VEC16_UE4M3` is what the sc-11040 packer emits.
            let scale_mode =
                sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                &scale_mode,
            )?;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                &scale_mode,
            )?;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &sa_ptr,
            )?;
            set_attr(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &sb_ptr,
            )?;

            // Layouts (see [`Self::run`] doc): A=(K,N) ld=K, B=(K,M) ld=K, D=(N,M) ld=N. For FP4 the
            // leading dim is in *elements* (K / N), not bytes — cuBLASLt handles the 2-nibbles/byte
            // packing internally from the `CUDA_R_4F_E2M1` dtype.
            let a_layout = lt::create_matrix_layout(io_dtype, k as u64, n as u64, k as i64)
                .map_err(cublas_err)?;
            let b_layout = lt::create_matrix_layout(io_dtype, k as u64, m as u64, k as i64)
                .map_err(cublas_err)?;
            let d_layout = lt::create_matrix_layout(out_dtype, n as u64, m as u64, n as i64)
                .map_err(cublas_err)?;

            // Algo: reuse the cached pick for this shape if present (the heuristic search is a host-side
            // cost that otherwise dominates a small FP4 GEMM); else run the heuristic once and cache it.
            // On an sm_120 device the heuristic returns the FP4 tensor-core algos (the spike saw 6); an
            // empty result surfaces as a cuBLASLt error (the caller reads that as "cuBLASLt did not
            // deliver an FP4 kernel" → the MMQ fallback would be needed).
            let cached = crate::lock_recover(&self.nvfp4_algos).get(&(m, k, n)).copied();
            let algo = match cached {
                Some(a) => a,
                None => {
                    let pref = lt::create_matmul_pref().map_err(cublas_err)?;
                    lt::set_matmul_pref_attribute(
                        pref,
                        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &self.workspace_size as *const _ as *const c_void,
                        size_of::<usize>(),
                    )
                    .map_err(cublas_err)?;
                    let heuristic = lt::get_matmul_algo_heuristic(
                        self.handle,
                        desc,
                        a_layout,
                        b_layout,
                        d_layout,
                        d_layout,
                        pref,
                    );
                    let _ = lt::destroy_matmul_pref(pref);
                    let algo = match heuristic {
                        Ok(h) => h.algo,
                        Err(e) => {
                            let _ = lt::destroy_matrix_layout(a_layout);
                            let _ = lt::destroy_matrix_layout(b_layout);
                            let _ = lt::destroy_matrix_layout(d_layout);
                            let _ = lt::destroy_matmul_desc(desc);
                            return Err(cublas_err(e));
                        }
                    };
                    crate::lock_recover(&self.nvfp4_algos).insert((m, k, n), algo);
                    algo
                }
            };

            let beta = F32_ZERO;
            // Hold the workspace device-ptr guard alive across the whole matmul call.
            let (ws_ptr, _gw) = self.workspace.device_ptr(&self.stream);
            let res = lt::matmul(
                self.handle,
                desc,
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                a_ptr as *const c_void,
                a_layout,
                b_ptr as *const c_void,
                b_layout,
                d_ptr as *const c_void,
                d_layout,
                d_ptr as *mut c_void,
                d_layout,
                &algo as *const _,
                ws_ptr as *mut c_void,
                self.workspace_size,
                self.stream.cu_stream() as sys::cudaStream_t,
            );

            let _ = lt::destroy_matrix_layout(a_layout);
            let _ = lt::destroy_matrix_layout(b_layout);
            let _ = lt::destroy_matrix_layout(d_layout);
            let _ = lt::destroy_matmul_desc(desc);
            res.map_err(cublas_err)
        }
    }

    /// A `(rows, cols)` fp8 weight/activation staged as an owned, contiguous, offset-0 device
    /// `F8E4M3` buffer. Staged once (at load for weights; per-forward for activations) so a repeated
    /// GEMM does not re-clone the operand — the difference between the bench's honest compute number
    /// and the naive per-call one.
    pub struct DevFp8 {
        buf: cudarc::driver::CudaSlice<float8::F8E4M3>,
        rows: usize,
        cols: usize,
    }

    impl DevFp8 {
        fn stage(lt: &CublasLt, t: &Tensor) -> Result<Self> {
            expect_dtype(t, DType::F8E4M3)?;
            let (rows, cols) = t.dims2()?;
            let t = t.contiguous()?;
            let (storage, _l) = t.storage_and_layout();
            let buf = match &*storage {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::F8E4M3(s) => s.try_clone().map_err(drv_err)?,
                    _ => candle_core::bail!("DevFp8::stage: expected F8E4M3 storage"),
                },
                _ => candle_core::bail!("DevFp8::stage: not a CUDA tensor"),
            };
            let _ = lt;
            Ok(Self { buf, rows, cols })
        }
    }

    /// A packed NVFP4 operand staged on-device: the E2M1 nibble bytes + the UE4M3 128×4-swizzled
    /// block-scale bytes + the FP32 per-tensor scale, plus the logical shape. Built from a host
    /// [`Nvfp4Tensor`] (the sc-11040 offline packer output); the two byte buffers upload once so a
    /// resident FP4 weight is not re-uploaded per forward. cuBLASLt reads `packed` as
    /// `CUDA_R_4F_E2M1` and `scales` as the `VEC16_UE4M3` block-scale tensor.
    pub struct DevNvfp4 {
        packed: cudarc::driver::CudaSlice<u8>,
        scales: cudarc::driver::CudaSlice<u8>,
        rows: usize,
        cols_padded: usize,
        global_scale: f32,
    }

    impl DevNvfp4 {
        /// Total **resident device** bytes of this staged NVFP4 operand — the E2M1 nibble buffer plus
        /// the UE4M3 block-scale buffer (the FP32 per-tensor scale is a host scalar). This is the honest
        /// VRAM footprint of a resident NVFP4 weight, used by the SC#6 packed-forward gate (sc-11041) to
        /// assert a resident `Nvfp4Linear` weight stays at the ~4.5-eff-bit NVFP4 footprint and never
        /// expands to bf16. Both buffers are `u8`, so `len()` is the byte count directly.
        pub fn resident_bytes(&self) -> usize {
            self.packed.len() + self.scales.len()
        }

        /// The logical `[rows, cols_padded]` shape of the staged operand (padding included).
        pub fn shape_padded(&self) -> (usize, usize) {
            (self.rows, self.cols_padded)
        }

        /// The per-tensor FP32 scale cuBLASLt consumes via `alpha`.
        pub fn global_scale(&self) -> f32 {
            self.global_scale
        }

        /// Copy the staged **UE4M3 block-scale bytes** back to the host — the swizzled buffer exactly as
        /// cuBLASLt reads it (see `cublaslt_scale_layout`, private to this module, for the offset of a
        /// logical `(row, block)`).
        ///
        /// Byte-level test/debug support, deliberately not on any hot path. It exists because an
        /// end-to-end GEMM rel-RMS check **cannot** see a single wrong scale byte at an exact E4M3
        /// rounding tie: ties are measure-zero under random activations, so only a crafted input
        /// inspected at the byte level catches them (sc-12078 review).
        pub fn scales_to_host(&self, lt: &CublasLt) -> Result<Vec<u8>> {
            lt.stream.clone_dtoh(&self.scales).map_err(drv_err)
        }

        /// Copy the staged **E2M1 nibble bytes** back to the host (row-major `[rows, cols_padded/2]`,
        /// low nibble = even column). Byte-level test/debug support; see [`Self::scales_to_host`].
        pub fn packed_to_host(&self, lt: &CublasLt) -> Result<Vec<u8>> {
            lt.stream.clone_dtoh(&self.packed).map_err(drv_err)
        }

        fn stage(lt: &CublasLt, t: &Nvfp4Tensor) -> Result<Self> {
            // Sanity: the packer's buffers must match its declared shape (guards a malformed handoff).
            let expect_packed = t.rows * (t.cols_padded / 2);
            if t.packed.len() != expect_packed {
                candle_core::bail!(
                    "DevNvfp4::stage: packed len {} != rows*cols_padded/2 {}",
                    t.packed.len(),
                    expect_packed
                );
            }
            if t.scales.len() != t.sf_rows * t.sf_cols {
                candle_core::bail!(
                    "DevNvfp4::stage: scales len {} != sf_rows*sf_cols {}",
                    t.scales.len(),
                    t.sf_rows * t.sf_cols
                );
            }
            let packed = lt.stream.clone_htod(&t.packed).map_err(drv_err)?;
            // Re-tile the UE4M3 block scales into the **row-major scale-factor-atom** order cuBLASLt's
            // matrix-scale descriptor expects (sc-11039 handoff item (a), GPU-confirmed). The sc-11040
            // packer emits the scales in *column-major* atom order (m-atom fastest); the two coincide
            // when the operand has a single 128-row atom, but for >128 rows cuBLASLt reads the wrong
            // atoms (round-trip rel-RMS jumps ~0.002 → ~0.082). We rebuild the buffer here by reading
            // each logical `(row, block)` scale through the packer's own `scale_offset` (so this stays
            // correct even if sc-11040 later flips its internal swizzle) and writing it at the
            // row-major-atom offset. Follow-up (sc-11040): emit row-major natively and drop this shim.
            let scales_dev = cublaslt_scale_layout(t);
            let scales = lt.stream.clone_htod(&scales_dev).map_err(drv_err)?;
            Ok(Self {
                packed,
                scales,
                rows: t.rows,
                cols_padded: t.cols_padded,
                global_scale: t.global_scale,
            })
        }
    }

    /// Rebuild a packed [`Nvfp4Tensor`]'s UE4M3 block scales into the **row-major scale-factor-atom**
    /// layout cuBLASLt's `VEC16_UE4M3` block-scale mode consumes (sc-11039 handoff (a)). The intra-atom
    /// swizzle `((32,4),4):((16,4),1)` (CUTLASS `Sm1xxBlockScaledConfig` SF atom) is unchanged — only
    /// the tiling **across** atoms flips from the packer's column-major (m-atom fastest) to row-major
    /// (k-atom fastest: `atom_index = k_atom + num_k_atoms * m_atom`). Reads logical scales through
    /// [`Nvfp4Tensor::scale_offset`] so it is agnostic to the packer's internal order.
    fn cublaslt_scale_layout(t: &Nvfp4Tensor) -> Vec<u8> {
        let mut out = vec![0u8; t.scales.len()];
        let num_k_atoms = t.sf_cols / SF_ATOM_COLS;
        for r in 0..t.sf_rows {
            for blk in 0..t.sf_cols {
                let v = t.scales[t.scale_offset(r, blk)];
                let m_atom = r / SF_ATOM_ROWS;
                let k_atom = blk / SF_ATOM_COLS;
                let atom_index = k_atom + num_k_atoms * m_atom;
                let mr = r % SF_ATOM_ROWS;
                let kc = blk % SF_ATOM_COLS;
                let intra = (mr % 32) * 16 + (mr / 32) * 4 + kc;
                out[atom_index * (SF_ATOM_ROWS * SF_ATOM_COLS) + intra] = v;
            }
        }
        out
    }

    // ---- on-device NVFP4 activation-quantize primitives (sc-11044) ------------------------------
    //
    // All operate on candle CUDA tensors so the W4A4 activation quantize is one on-device dataflow
    // (no host round-trip). The arithmetic mirrors the CPU packer's `e2m1_from_f32` / `e4m3_from_f32`
    // closely enough that the staged operand tracks the CPU reference within a small rel-RMS.

    #[inline]
    fn round_up_usize(x: usize, m: usize) -> usize {
        x.div_ceil(m) * m
    }

    /// Extract an owned device `CudaSlice<u8>` from a candle tensor of non-negative integer values in
    /// `[0, 255]` (cast to `U8` on-device, same idiom as `DevInt8::stage`'s CUDA leg). The values are
    /// exact integers so the `f32 -> u8` cast is lossless.
    fn extract_u8_slice(t: &Tensor) -> Result<cudarc::driver::CudaSlice<u8>> {
        let u8t = t.to_dtype(DType::U8)?.flatten_all()?.contiguous()?;
        let (storage, _l) = u8t.storage_and_layout();
        match &*storage {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s.try_clone().map_err(drv_err),
                _ => candle_core::bail!("extract_u8_slice: expected U8 CUDA storage"),
            },
            _ => candle_core::bail!("extract_u8_slice: tensor is not on CUDA"),
        }
    }

    const LN2: f64 = std::f64::consts::LN_2;
    const E4M3_MIN_NORMAL: f64 = 0.015_625; // 2^-6
    const E4M3_MIN_SUBNORMAL: f64 = 1.0 / 512.0; // 2^-9
    const E4M3_MAXV: f64 = 448.0;

    /// Round a non-negative tensor onto the OCP E4M3 grid, returning the **decoded** value (nearest
    /// representable E4M3, saturating at 448). `q = rn_even(v / ulp) · ulp` with `ulp = 2^(e-3)` and
    /// `e = floor(log2(max(v, 2^-6)))` — a single formula that covers both the normal grid (ULP
    /// `2^e/8` in `[2^e, 2^{e+1}]`) and the subnormal grid (`e` clamped to `-6` -> ULP `2^-9`).
    ///
    /// **Ties MUST round to even.** The canonical encoder
    /// [`e4m3_from_f32`](super::super::nvfp4::e4m3_from_f32) tie-breaks on
    /// `code.is_multiple_of(2)`, and an even E4M3 byte is an even mantissa LSB. candle's `.round()`
    /// rounds halves **away from zero** and is therefore wrong here: it emitted 0x51 (9.0) for
    /// `sf_real` 8.5 where the CPU emits 0x50 (8.0), rescaling that whole 16-element block. candle's
    /// tensor API has no RN-even round, so the tie is derived explicitly — `t.round()` away from a
    /// tie, and `floor(t) + (floor(t) mod 2)` on one (which is `floor(t)` when it is already even and
    /// `floor(t)+1`, also even, when it is odd). This is exactly the fused kernel's `__float2int_rn`
    /// ([`NVFP4_QUANT_CU`], sc-12078) expressed in candle ops. Guarded by
    /// `nvfp4_unfused_e4m3_block_scale_bytes_match_cpu_at_exact_ties` — the GEMM rel-RMS gate cannot
    /// see this (exact midpoints are measure-zero under random activations, so it reads 0.000000
    /// throughout).
    fn e4m3_round_tensor(v: &Tensor) -> Result<Tensor> {
        let v = v.clamp(0.0, E4M3_MAXV)?;
        // Exponent from a floor of clamped v (avoid log2(0); subnormals floor to -6).
        let vc = v.clamp(E4M3_MIN_NORMAL, E4M3_MAXV)?;
        let e = vc.log()?.affine(1.0 / LN2, 0.0)?.floor()?; // floor(log2(vc)) in [-6, 8]
        let ulp = e.affine(1.0, -3.0)?.affine(LN2, 0.0)?.exp()?; // 2^(e-3)
        let t = v.broadcast_div(&ulp)?; // v ≥ 0 ⟹ t ∈ [0, 16); ulp is a power of two, so a tie is exact
        let fl = t.floor()?;
        let is_tie = t.sub(&fl)?.eq(0.5)?;
        // floor(t) mod 2, as `fl - 2·floor(fl/2)`: 0 when fl is even, 1 when odd. `fl ≤ 15` here, so
        // every step is exact in f32.
        let parity = fl.sub(&fl.affine(0.5, 0.0)?.floor()?.affine(2.0, 0.0)?)?;
        let rn_even = is_tie.where_cond(&fl.add(&parity)?, &t.round()?)?;
        rn_even.broadcast_mul(&ulp)?.clamp(0.0, E4M3_MAXV)
    }

    /// The OCP E4M3 **byte** (0..=254) whose decode equals the on-grid value `q` (as produced by
    /// [`e4m3_round_tensor`]). Because `q` already sits exactly on the E4M3 grid this inversion is exact
    /// (the `round`s only clean float noise): normals -> `((e+7)<<3) | (round(q/2^e·8) - 8)` (mantissa
    /// overflow to `2^{e+1}` folds into the next exponent automatically), subnormals -> `round(q·512)`,
    /// zero -> 0.
    fn e4m3_byte_from_decoded(q: &Tensor) -> Result<Tensor> {
        let is_zero = q.lt(E4M3_MIN_SUBNORMAL / 2.0)?; // q >= 0; below half the min subnormal == 0
        let is_sub = q.lt(E4M3_MIN_NORMAL)?;
        let qc = q.clamp(E4M3_MIN_SUBNORMAL, E4M3_MAXV)?;
        let e = qc.log()?.affine(1.0 / LN2, 0.0)?.floor()?; // floor(log2 q)
        let pow_e = e.affine(LN2, 0.0)?.exp()?; // 2^e
        // Normal: E = e + 7, M = round(q / 2^e · 8) - 8  -> byte = E·8 + M = (e+7)·8 + round(...) - 8.
        let mant = q.broadcast_div(&pow_e)?.affine(8.0, 0.0)?.round()?; // round(q/2^e·8), 8..16
        let byte_normal = e.affine(8.0, 8.0 * 7.0 - 8.0)?.add(&mant)?; // (e·8) + (56-8) + mant
        // Subnormal: byte = round(q · 512).
        let byte_sub = q.affine(512.0, 0.0)?.round()?;
        let zeros = q.zeros_like()?;
        let byte = is_sub.where_cond(&byte_sub, &byte_normal)?;
        let byte = is_zero.where_cond(&zeros, &byte)?;
        byte.clamp(0.0, 254.0)
    }

    /// Nearest-E2M1 codes (0..=15, sign in bit 3) for a signed ratio tensor `value / block_scale`.
    /// Magnitude index = count of the 7 grid midpoints the magnitude passes (`{0,.5,1,1.5,2,3,4,6}`
    /// -> midpoints `{.25,.75,1.25,1.75,2.5,3.5,5}`); saturates at index 7 (±6).
    ///
    /// **Ties round to even**, matching the canonical
    /// [`e2m1_from_f32`](super::super::nvfp4::e2m1_from_f32) (which tie-breaks on
    /// `i.is_multiple_of(2)` over the magnitude table — an even index is an even mantissa LSB). This
    /// is why the comparison alternates: midpoint `i` sits between magnitude index `i` and `i+1`, so
    /// for an **even** `i` the tie keeps the lower index and the magnitude must be *strictly* past the
    /// midpoint to advance (`>`), while for an **odd** `i` the tie advances onto the even index `i+1`
    /// (`>=`). A uniform `>=` — as this used before — rounds every tie UP, which is wrong at 4 of the
    /// 7 midpoints (0.25, 1.25, 2.5, 5.0) and rescales the affected element. Same thresholds as the
    /// fused kernel's `e2m1_code` ([`NVFP4_QUANT_CU`], sc-12078), which spells them as `<=`/`<`.
    /// Guarded by `nvfp4_e2m1_element_nibbles_match_cpu_at_exact_ties` (which runs both routes through
    /// one assertion); like the E4M3 tie defect this is invisible to a GEMM rel-RMS gate (exact
    /// midpoints are measure-zero).
    fn e2m1_code_tensor(ratio: &Tensor) -> Result<Tensor> {
        let mag = ratio.abs()?;
        const MIDS: [f64; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
        let mut idx = mag.zeros_like()?;
        for (i, &mid) in MIDS.iter().enumerate() {
            let past = if i.is_multiple_of(2) { mag.gt(mid)? } else { mag.ge(mid)? };
            idx = idx.add(&past.to_dtype(DType::F32)?)?;
        }
        // sign bit (8) where negative; -0.0 (sign set, magnitude 0) decodes to 0.0 so is harmless.
        let sign = ratio.lt(0.0)?.to_dtype(DType::F32)?.affine(8.0, 0.0)?;
        idx.add(&sign)
    }

    /// A `(rows, cols)` int8-code operand as an owned device **byte** buffer. candle has no `i8` dtype,
    /// so the int8 codes are carried as their two's-complement `u8` byte pattern (`c as i8 as u8`);
    /// cuBLASLt reads the buffer as `CUDA_R_8I`, so byte `b` reads back as the signed code `b as i8`.
    /// Holding `u8` (candle's native byte dtype) is what lets a **CUDA** operand stage entirely
    /// on-device (sc-9601 perf): the per-forward activation narrow becomes a handful of candle kernels
    /// instead of a DtoH copy + CPU round loop + HtoD copy (the dominant cost once the dequant fold
    /// moved on-device). A CPU operand (weights loaded to host, tests) still narrows on the host.
    pub struct DevInt8 {
        buf: cudarc::driver::CudaSlice<u8>,
        rows: usize,
        cols: usize,
    }

    impl DevInt8 {
        fn stage(lt: &CublasLt, t: &Tensor) -> Result<Self> {
            let (rows, cols) = t.dims2()?;
            if t.device().is_cuda() {
                // On-device narrow to the int8 byte pattern (sc-9601): round to [-127, 127], wrap the
                // negatives into [128, 255] (two's complement: `c + 256`), then cast f32→u8. The values
                // are exact integers so the f32→u8 cast is lossless. cuBLASLt reads these bytes as i8.
                let codes = t.to_dtype(DType::F32)?.round()?.clamp(-I8_MAX, I8_MAX)?;
                let neg256 = codes.lt(0f32)?.to_dtype(DType::F32)?.affine(256.0, 0.0)?;
                let u8t = codes
                    .add(&neg256)?
                    .to_dtype(DType::U8)?
                    .flatten_all()?
                    .contiguous()?;
                let (storage, _l) = u8t.storage_and_layout();
                let buf = match &*storage {
                    Storage::Cuda(cs) => match &cs.slice {
                        CudaStorageSlice::U8(s) => s.try_clone().map_err(drv_err)?,
                        _ => candle_core::bail!("DevInt8::stage: expected U8 storage"),
                    },
                    _ => candle_core::bail!("DevInt8::stage: not a CUDA tensor"),
                };
                return Ok(Self { buf, rows, cols });
            }
            // CPU operand (weights staged from host, tests): narrow on the host to the same byte pattern.
            let host = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let bytes: Vec<u8> = host.iter().map(|&v| v.round() as i8 as u8).collect();
            let buf = lt.stream.clone_htod(&bytes).map_err(drv_err)?;
            Ok(Self { buf, rows, cols })
        }
    }

    /// alpha carrier so int32 vs f32 compute get correctly-typed scalars.
    enum Alpha {
        F32(f32),
        I32(i32),
    }
    static F32_ZERO: f32 = 0.0;
    static I32_ZERO: i32 = 0;

    unsafe fn set_attr<T>(
        desc: sys::cublasLtMatmulDesc_t,
        attr: sys::cublasLtMatmulDescAttributes_t,
        val: &T,
    ) -> Result<()> {
        lt::set_matmul_desc_attribute(desc, attr, val as *const T as *const c_void, size_of::<T>())
            .map_err(cublas_err)
    }

    fn check_alignment(k: usize, n: usize) -> Result<()> {
        if !k.is_multiple_of(16) || !n.is_multiple_of(16) {
            candle_core::bail!("cuBLASLt 8-bit: K ({k}) and N ({n}) must be multiples of 16");
        }
        Ok(())
    }

    /// NVFP4 operand-K / N alignment (sc-11039 handoff item (b)). The NVFP4 block is 16 elements and
    /// cuBLASLt's FP4 block-scaled path requires the contraction dim `K` a multiple of the scale-factor
    /// atom's K-extent. The sc-11040 packer pads its **scale tensor** to 4 blocks (`SF_ATOM_COLS`) =
    /// **64 elements** along K, so a K that is a multiple of 64 always has a fully-populated scale atom
    /// and is unconditionally safe; the live-GPU probe (`nvfp4_k_alignment_probe`) reports the actual
    /// minimum cuBLASLt accepts. `N` (out) must be a multiple of 16. Enforced at K a multiple of 64 —
    /// the conservative bound that matches the packer's padded scale-atom width and never trips
    /// `CUBLAS_STATUS_NOT_SUPPORTED` on a partial scale atom.
    /// The required padded-K multiple for the NVFP4 cuBLASLt path (sc-11039 handoff item (b),
    /// GPU-confirmed on the RTX PRO 6000). The `nvfp4_k_alignment_probe` swept K on live hardware:
    /// K∈{32,64,128} are ACCEPTED (bit-accurate), K∈{16,48} return `CUBLAS_STATUS_NOT_SUPPORTED` — so
    /// cuBLASLt's FP4 block-scaled path requires **K a multiple of 32** (two NVFP4 blocks), *not* the
    /// single 16-element block and *not* the 64-element 4-block scale atom. The sc-11040 packer pads
    /// `cols_padded` only to a multiple of 16, so an `in_features` in e.g. `[33,47]` packs to K=48 and
    /// is rejected here with a clear message (follow-up: have the packer pad K to 32).
    pub const NVFP4_K_ALIGN: usize = 32;

    fn check_nvfp4_alignment(k: usize, n: usize) -> Result<()> {
        if !k.is_multiple_of(NVFP4_K_ALIGN) {
            candle_core::bail!(
                "cuBLASLt NVFP4: padded K ({k}) must be a multiple of {NVFP4_K_ALIGN} (two NVFP4 \
                 blocks — cuBLASLt returns NOT_SUPPORTED otherwise, e.g. K=16 or K=48); pack with an \
                 in_features that rounds to a multiple of {NVFP4_K_ALIGN}"
            );
        }
        if !n.is_multiple_of(16) {
            candle_core::bail!("cuBLASLt NVFP4: N ({n}) must be a multiple of 16");
        }
        Ok(())
    }

    fn expect_dtype(t: &Tensor, want: DType) -> Result<()> {
        if t.dtype() != want {
            candle_core::bail!("expected {want:?}, got {:?}", t.dtype());
        }
        Ok(())
    }

    fn cublas_err(e: cudarc::cublaslt::result::CublasError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("cublasLt: {e:?}").into())
    }
    fn drv_err(e: cudarc::driver::DriverError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("cuda driver: {e:?}").into())
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{CublasLt, DevFp8, DevInt8, DevNvfp4, NVFP4_K_ALIGN};

/// **One `CublasLt` handle per device**, shared by every INT8 projection built against it (sc-12301)
/// — the int8 twin of [`Nvfp4Context`](super::nvfp4_linear::Nvfp4Context).
///
/// # Why this exists
///
/// `CublasLt::new` eagerly allocates a `CublasLt::WORKSPACE` (32 MiB) buffer it holds for life, and
/// nothing on the handle is per-layer (its caches are keyed by shape, or not at all — this module's own
/// docs say *"a real integration caches one per device"*). `QLinear::convrot_int8` nevertheless built one
/// **per projection**, so a ConvRot DiT's ~224 int8 projections carried ~7 GiB of duplicated scratch that
/// a weights-only byte accounting cannot see. sc-12274 measured the identical defect on the NVFP4 lane at
/// 32.00 MiB/handle and recovered 7.5 GiB by sharing; this is that fix for int8.
///
/// # Why it is not `Nvfp4Context`
///
/// Two deliberate differences, either of which would be a bug if copied across:
///
/// 1. **No capability floor here.** `Nvfp4Context::new` gates on `meets_nvfp4_floor` (sm_120) and drops
///    the handle below it. The int8 lane's floor is **sm_89** (locked decision 7), enforced once at load
///    by `candle_gen_krea::pipeline`'s `ensure_int8_floor` — which builds *this* context and reads the cap
///    off its handle rather than discarding a throwaway probe. Gating on sm_120 here would wrongly deny
///    int8 on every sm_89..sm_120 card.
/// 2. **A missing handle is an error, not a fallback.** An absent NVFP4 handle simply means dequant→bf16,
///    so `Nvfp4Context` returns [`none`](Self::none) rather than failing. INT8-ConvRot has no such lane:
///    without the handle the forward drops to a cross-device dequant-dense matmul — correct but
///    catastrophically slow, and silently so. F-121 / sc-11208 settled that this must surface as a typed
///    error at **load** (where `?` is available) instead of an `.expect()` panic — or, worse, a silent
///    collapse — on the first sampler forward. So [`Self::new`] propagates a handle failure and
///    [`handle_for`](Self::handle_for) returns a `Result`.
///
/// # Contract
///
/// Cfg-neutral **by design** — a zero-sized type on a non-cuda build, so the `*_in` constructors have one
/// signature everywhere. It lives here rather than beside `Int8Linear` because that module is cuda-only
/// as a whole, while this one compiles everywhere with the handle gated
/// inside. An **empty** context is the honest and correct state on a non-CUDA device (the CPU/Metal
/// dequant-dense fallback is test-only); it is only an error to hand one to a projection *on CUDA*.
///
/// Safe to share across layers: every handle on a device already resolves to the same stream
/// (`CublasLt::new` takes `device.cuda_stream()`), so sharing introduces no stream coupling that did not
/// already exist. The one genuinely shared mutable resource is the 32 MiB workspace scratch, and a
/// denoise is sequential on that one stream, so cuBLASLt serializes access to it — pinned bit-exact by
/// `int8_linear_shares_one_cublaslt_workspace_across_layers`.
#[derive(Clone, Default)]
pub struct Int8Context {
    #[cfg(feature = "cuda")]
    inner: Option<Int8Ctx>,
}

/// The live half of an [`Int8Context`]: the shared handle + the device it is bound to (the binding is
/// what [`Int8Context::handle_for`] checks — a hazard that only exists once handles are shared).
#[cfg(feature = "cuda")]
#[derive(Clone)]
struct Int8Ctx {
    lt: std::sync::Arc<CublasLt>,
    device: Device,
}

impl Int8Context {
    /// The **empty** context: no shared handle. Correct on any non-CUDA device, where the int8 legs are
    /// `None` and the forward takes its dequant-dense fallback. What [`Self::new`] returns there.
    pub fn none() -> Self {
        Self::default()
    }

    /// Build **one** cuBLASLt handle for `device`, to be shared by every INT8 projection on it.
    ///
    /// A non-CUDA device (or a non-cuda build) yields [`Self::none`] — the fallback is the honest answer
    /// there, not a failure. On a CUDA device a handle failure is propagated as a **typed error**
    /// (F-121 / sc-11208): the int8 lane cannot run without it, and load time is where that must be said.
    ///
    /// Deliberately does **not** probe the sm_89 floor — see the [type docs](Self).
    pub fn new(device: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if device.is_cuda() {
                return Ok(Self {
                    inner: Some(Int8Ctx {
                        lt: std::sync::Arc::new(CublasLt::new(device)?),
                        device: device.clone(),
                    }),
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(Self::none())
    }

    /// True iff this context carries a live handle (i.e. projections built with it take the int8 IGEMM).
    pub fn is_int8(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            return self.inner.is_some();
        }
        #[allow(unreachable_code)]
        false
    }

    /// The shared handle, **iff** it is bound to `device`. Call only for a CUDA `device` — a projection
    /// off CUDA has no int8 leg to build and must not ask.
    ///
    /// Both failures are typed errors rather than a silent `None`, because neither has a fallback worth
    /// taking quietly (see the [type docs](Self)):
    ///
    /// * **No handle** — an empty context reached a CUDA projection.
    /// * **Wrong device** — a context built on `cuda:0` handed to a projection on `cuda:1` would stage
    ///   the int8 codes through the wrong device's stream. This hazard is **new to sharing**: a
    ///   per-layer handle could never be bound to the wrong device.
    #[cfg(feature = "cuda")]
    pub fn handle_for(&self, device: &Device) -> Result<&std::sync::Arc<CublasLt>> {
        let c = self.inner.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "[sc-12301] Int8Context: an empty context reached an int8 projection on {:?}. The int8 \
                 lane has no fallback worth taking silently, so this is a typed error at load rather \
                 than a quiet collapse to a cross-device dequant-dense matmul mid-render (F-121).",
                device.location()
            ))
        })?;
        if c.device.same_device(device) {
            Ok(&c.lt)
        } else {
            Err(candle_core::Error::Msg(format!(
                "[sc-12301] Int8Context: the shared cuBLASLt handle is bound to {:?} but this int8 \
                 projection is on {:?}; staging its codes through the wrong device's stream would \
                 corrupt the layer.",
                c.device.location(),
                device.location()
            )))
        }
    }
}
