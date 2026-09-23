//! Fused NVFP4 decode GEMV (sc-24136, epic sc-24128): `y = x·Wᵀ` for a resident
//! [`Nvfp4Weight`] and a **bf16** activation of 1..=[`NVFP4_GEMV_MAX_ROWS`] token rows, as one
//! launch of [`NVFP4_GEMV_SRC`] compiled through the [`nvrtc`](crate::nvrtc) compile-once seam.
//!
//! # Why a second NVFP4 forward
//!
//! [`Nvfp4Weight::forward`] is the W4A4 cuBLASLt path: it pads M to 16, quantizes the activation
//! to NVFP4 with the fused quantizer (two launches **and one host sync** for the per-tensor amax
//! that becomes the GEMM `alpha`) and runs the block-scaled FP4 GEMM. At decode (M = 1, or K+1
//! rows under MTP) that fixed cost dwarfs the GEMM itself. This kernel instead streams the packed
//! weight once, dequantizes it in registers (exactly, to bf16) and multiplies it by the
//! **unquantized** bf16 activation on the tensor cores (`mma.m16n8k16`, f32 accumulate) — no
//! activation quantization, no host sync, one launch, and a cost that is flat in the row count
//! (the activation rows ride in the MMA's 8-wide `N` dimension). See `nvfp4_gemv.cu` for the
//! layout contract and work split.
//!
//! # Numerics
//!
//! The GEMV is **not** bit-identical to the W4A4 path, and is not meant to be: it skips the
//! activation's FP4 quantization and its error. Its reference is the dequantize-then-matmul
//! product `x · dequant(W)ᵀ` (f32), which it matches within the declared tolerance
//! [`GEMV_REL_RMS_TOL`] / [`gemv_abs_bound`]: the residual is the single bf16 rounding of the
//! output plus the f32 accumulation order.
//!
//! # Validation before launch (epic E5)
//!
//! [`check_nvfp4_gemv`] builds on every lane (CPU too) and returns a typed
//! [`Nvfp4GemvRefusal`] for any input the kernel does not serve (not CUDA, another device, not
//! bf16, 0 or more than eight rows, a `K` that does not match the weight). The caller routes a
//! refused input to [`Nvfp4Weight::forward`] (cuBLASLt) and reports the refusal's label, so which
//! path ran is never silent.

use std::fmt;

use candle_core::{DType, Tensor};

use crate::nvfp4_weight::Nvfp4Weight;
use crate::nvrtc::{KernelCompileError, KernelSource};

/// The fused NVFP4 GEMV source; compiled once per device on first use.
pub const NVFP4_GEMV_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_nvfp4_gemv_v1",
    src: include_str!("nvfp4_gemv.cu"),
    // The true floor of the code: `mma.sync.m16n8k16` with bf16 operands and `fma.rn.bf16x2`
    // are sm_80 instructions. NVFP4 weights themselves only exist on sm_120+
    // (`Nvfp4Context::require` refuses the load below it), so this floor never binds in practice.
    cc_floor: (8, 0),
};

/// Largest token-row count (`M`, the product of the activation's leading dims) the GEMV serves.
/// More rows go to the cuBLASLt W4A4 GEMM, which amortizes its activation quantization.
pub const NVFP4_GEMV_MAX_ROWS: usize = 8;

/// The kernel's entry point in [`NVFP4_GEMV_SRC`] (one kernel for every row count).
pub const NVFP4_GEMV_FUNCTION: &str = "nvfp4_gemv_bf16";

/// Threads per block (`MMA_WARPS` × 32 in the source): eight warps split `K`.
pub const NVFP4_GEMV_THREADS: u32 = 256;

/// Output rows per block: one `m16n8k16` tile's 16 weight rows (every NVFP4 weight has
/// `N % 16 == 0`, so the grid is exactly `N / 16`).
pub const NVFP4_GEMV_ROWS_PER_BLOCK: usize = 16;

/// Declared parity tolerance, part 1: the relative RMS of the GEMV output against the
/// dequantize-then-matmul f32 reference, `‖y − ref‖₂ / ‖ref‖₂ ≤ 2⁻⁷`. bf16 carries 8 significant
/// bits, so the output's single rounding is at most 2⁻⁸ relative per element and ~2⁻⁸/√3 in RMS;
/// 2⁻⁷ leaves ≥ 3× margin for the f32 accumulation order.
pub const GEMV_REL_RMS_TOL: f64 = 1.0 / 128.0;

/// Declared parity tolerance, part 2: the per-element bound
/// `|y − ref| ≤ 2⁻⁷·|ref| + K·2⁻²⁴·Σₖ|x·W|` — twice the worst-case bf16 output rounding (half an
/// ulp is up to 2⁻⁸ relative) plus the classic worst-case f32 dot-product accumulation error
/// `γ_K ≈ K·u` (u = 2⁻²⁴) over the absolute products. `abs_dot` is `Σₖ |x[k]·W[n,k]|` for the
/// element.
pub fn gemv_abs_bound(reference: f64, abs_dot: f64, k: usize) -> f64 {
    reference.abs() / 128.0 + (k as f64) * abs_dot / (1u64 << 24) as f64
}

/// Why the GEMV refuses an input. The cuBLASLt W4A4 path serves it instead; the label is what
/// telemetry records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4GemvRefusal {
    /// The activation is not on a CUDA device.
    NotCuda,
    /// The activation is on a different CUDA device than the weight.
    DeviceMismatch,
    /// Only `BF16` activations are served.
    Dtype(DType),
    /// The activation has zero rows or more than [`NVFP4_GEMV_MAX_ROWS`].
    Rows(usize),
    /// The activation's last dim is not the weight's `in` features (or the activation is a
    /// scalar).
    Shape {
        /// The weight's input features.
        expected_k: usize,
        /// The activation's last dim (0 for a scalar).
        got_k: usize,
    },
}

impl Nvfp4GemvRefusal {
    /// Stable lower-case label for telemetry.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotCuda => "not_cuda",
            Self::DeviceMismatch => "device",
            Self::Dtype(_) => "dtype",
            Self::Rows(_) => "rows",
            Self::Shape { .. } => "shape",
        }
    }
}

impl fmt::Display for Nvfp4GemvRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "activation is not on a CUDA device"),
            Self::DeviceMismatch => write!(f, "activation is on another device than the weight"),
            Self::Dtype(d) => write!(f, "activation dtype {d:?} is not served (BF16 only)"),
            Self::Rows(m) => write!(
                f,
                "{m} token rows is outside the GEMV's 1..={NVFP4_GEMV_MAX_ROWS}"
            ),
            Self::Shape { expected_k, got_k } => write!(
                f,
                "activation last dim {got_k} does not match the weight's {expected_k} input \
                 features"
            ),
        }
    }
}

/// The GEMV's failure: a typed refusal (route to cuBLASLt), a cached compile error (the seam's;
/// also route to cuBLASLt), or a candle error from the surrounding ops (propagate).
#[derive(Debug)]
pub enum Nvfp4GemvError {
    /// Input outside the kernel's declared constraints.
    Refused(Nvfp4GemvRefusal),
    /// The kernel source does not compile / load on this device (cached by the seam).
    Compile(KernelCompileError),
    /// A candle / driver error (allocation, contiguity copy, launch).
    Candle(candle_core::Error),
}

impl Nvfp4GemvError {
    /// Stable telemetry label: the refusal's label, the compile error's label, or `candle`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Refused(r) => r.label(),
            Self::Compile(e) => e.label(),
            Self::Candle(_) => "candle",
        }
    }
}

impl fmt::Display for Nvfp4GemvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(r) => write!(f, "nvfp4 GEMV refused: {r}"),
            Self::Compile(e) => write!(f, "nvfp4 GEMV unavailable: {e}"),
            Self::Candle(e) => write!(f, "nvfp4 GEMV failed: {e}"),
        }
    }
}

impl std::error::Error for Nvfp4GemvError {}

impl From<Nvfp4GemvRefusal> for Nvfp4GemvError {
    fn from(r: Nvfp4GemvRefusal) -> Self {
        Self::Refused(r)
    }
}

impl From<KernelCompileError> for Nvfp4GemvError {
    fn from(e: KernelCompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<candle_core::Error> for Nvfp4GemvError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}

/// The validated geometry of one GEMV launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4GemvPlan {
    /// Token rows (`M`, the product of the activation's leading dims), 1..=8.
    pub rows: usize,
    /// Output features (`N`).
    pub n: usize,
    /// Input features (`K`, logical — before the weight's padding to 32).
    pub k: usize,
}

/// Validate a GEMV of activation `x: [..., k]` against a `[n, k]` NVFP4 weight, without touching
/// a device beyond comparing locations. `weight_device` is the weight's device.
pub fn check_nvfp4_gemv(
    x: &Tensor,
    weight_shape: (usize, usize),
    weight_device: &candle_core::Device,
) -> Result<Nvfp4GemvPlan, Nvfp4GemvRefusal> {
    if !x.device().is_cuda() {
        return Err(Nvfp4GemvRefusal::NotCuda);
    }
    if !x.device().same_device(weight_device) {
        return Err(Nvfp4GemvRefusal::DeviceMismatch);
    }
    if x.dtype() != DType::BF16 {
        return Err(Nvfp4GemvRefusal::Dtype(x.dtype()));
    }
    let (n, expected_k) = weight_shape;
    let got_k = x.dims().last().copied().unwrap_or(0);
    if x.rank() == 0 || got_k != expected_k {
        return Err(Nvfp4GemvRefusal::Shape { expected_k, got_k });
    }
    let rows = x.elem_count() / got_k.max(1);
    if rows == 0 || rows > NVFP4_GEMV_MAX_ROWS {
        return Err(Nvfp4GemvRefusal::Rows(rows));
    }
    Ok(Nvfp4GemvPlan {
        rows,
        n,
        k: expected_k,
    })
}

impl Nvfp4Weight {
    /// `y = x·Wᵀ (+ b)` through the fused decode GEMV: a bf16 activation `[..., in]` with at most
    /// [`NVFP4_GEMV_MAX_ROWS`] rows, unquantized, against the resident packed weight; returns
    /// `[..., out]` bf16. Refuses (typed, before any launch) what [`check_nvfp4_gemv`] refuses.
    #[cfg(feature = "cuda")]
    pub fn forward_gemv(&self, x: &Tensor) -> Result<Tensor, Nvfp4GemvError> {
        let plan = check_nvfp4_gemv(x, self.shape(), self.device())?;
        cuda_impl::launch(self, x, plan)
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, Device, Shape, Storage};
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    use crate::nvfp4::{NVFP4_BLOCK, SF_ATOM_COLS};

    fn drv(e: cudarc::driver::DriverError) -> Nvfp4GemvError {
        Nvfp4GemvError::Candle(candle_core::Error::Cuda(
            format!("nvfp4 GEMV kernel: {e:?}").into(),
        ))
    }

    pub(super) fn launch(
        weight: &Nvfp4Weight,
        x: &Tensor,
        plan: Nvfp4GemvPlan,
    ) -> Result<Tensor, Nvfp4GemvError> {
        let Device::Cuda(dev) = weight.device() else {
            return Err(Nvfp4GemvRefusal::NotCuda.into());
        };
        let func = NVFP4_GEMV_SRC
            .compiled(dev)?
            .function(NVFP4_GEMV_FUNCTION)?;

        let staged = weight.staged();
        let (_, cols_padded) = staged.shape_padded();
        let n_blocks = cols_padded / NVFP4_BLOCK;
        let num_k_atoms = n_blocks.div_ceil(SF_ATOM_COLS);

        // The kernel reads x as a dense [rows, k] bf16 matrix; 16-byte vector loads need K % 8 == 0
        // and a start offset on a 16-byte boundary (a narrowed view may not be).
        let x2 = x.reshape((plan.rows, plan.k))?.contiguous()?;
        let x2 = if x2.layout().start_offset() % 8 != 0 {
            x2.copy()?
        } else {
            x2
        };
        let vec_x = i32::from(plan.k.is_multiple_of(8));

        let mut y = unsafe { dev.alloc::<half::bf16>(plan.rows * plan.n) }?;
        // `Nvfp4Weight::quantize` refuses N % 16 != 0, so the 16-row tiles cover N exactly.
        let cfg = LaunchConfig {
            grid_dim: ((plan.n / NVFP4_GEMV_ROWS_PER_BLOCK) as u32, 1, 1),
            block_dim: (NVFP4_GEMV_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let m_rows = plan.rows as i32;
        let n_rows = plan.n as i32;
        let k = plan.k as i32;
        let cp = cols_padded as i32;
        let nka = num_k_atoms as i32;
        let gs = staged.global_scale();
        let stream = dev.cuda_stream();
        {
            let (xs, xl) = x2.storage_and_layout();
            let xs = match &*xs {
                Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..),
                _ => return Err(Nvfp4GemvRefusal::NotCuda.into()),
            };
            let mut b = stream.launch_builder(&func);
            b.arg(staged.packed_slice())
                .arg(staged.scales_slice())
                .arg(&xs)
                .arg(&mut y)
                .arg(&n_rows)
                .arg(&k)
                .arg(&cp)
                .arg(&nka)
                .arg(&gs)
                .arg(&vec_x)
                .arg(&m_rows);
            unsafe { b.launch(cfg) }.map_err(drv)?;
        }

        let mut out_dims = x.dims().to_vec();
        *out_dims.last_mut().expect("rank checked") = plan.n;
        let y = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(y, dev.clone())),
            Shape::from((plan.rows, plan.n)),
            BackpropOp::none(),
            false,
        )
        .reshape(out_dims)?;
        Ok(match weight.bias() {
            Some(b) => y.broadcast_add(&b.to_dtype(DType::BF16)?)?,
            None => y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn checks_declare_their_constraints_on_any_lane() {
        // CPU activations are refused first, whatever else is wrong with them.
        let x = Tensor::zeros((1, 64), DType::BF16, &Device::Cpu).unwrap();
        assert_eq!(
            check_nvfp4_gemv(&x, (16, 64), &Device::Cpu),
            Err(Nvfp4GemvRefusal::NotCuda)
        );
        assert_eq!(Nvfp4GemvRefusal::NotCuda.label(), "not_cuda");
        assert_eq!(Nvfp4GemvRefusal::DeviceMismatch.label(), "device");
        assert_eq!(Nvfp4GemvRefusal::Dtype(DType::F32).label(), "dtype");
        assert_eq!(Nvfp4GemvRefusal::Rows(9).label(), "rows");
        assert_eq!(
            Nvfp4GemvRefusal::Shape {
                expected_k: 64,
                got_k: 32
            }
            .label(),
            "shape"
        );
        let msg = Nvfp4GemvRefusal::Rows(9).to_string();
        assert!(msg.contains("1..=8"), "{msg}");
    }

    #[test]
    fn the_source_declares_the_entry_point_and_its_true_floor() {
        assert!(NVFP4_GEMV_SRC.src.contains(&format!(
            "__global__ void __launch_bounds__(MMA_THREADS) {NVFP4_GEMV_FUNCTION}("
        )));
        assert!(NVFP4_GEMV_SRC.src.contains("#define MMA_WARPS 8"));
        assert_eq!(NVFP4_GEMV_THREADS, 8 * 32);
        // mma.sync bf16 + fma.rn.bf16x2 are sm_80; nothing newer is used.
        assert_eq!(NVFP4_GEMV_SRC.cc_floor, (8, 0));
        assert!(NVFP4_GEMV_SRC
            .src
            .contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
    }

    #[test]
    fn declared_abs_bound_grows_with_k_and_the_reference() {
        let small = gemv_abs_bound(1.0, 10.0, 128);
        let tall = gemv_abs_bound(1.0, 10.0, 17_408);
        assert!(tall > small);
        assert!((gemv_abs_bound(128.0, 0.0, 1) - 1.0).abs() < 1e-12);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::nvfp4_linear::Nvfp4Context;
    use candle_core::Device;

    /// A CUDA device that meets the NVFP4 floor, or `None` (the GPU tests then skip loudly).
    fn nvfp4_device() -> Option<(Device, Nvfp4Context)> {
        let device = Device::new_cuda(0).ok()?;
        let ctx = Nvfp4Context::require(&device).ok()?;
        Some((device, ctx))
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

    fn bf16(dims: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        Tensor::from_vec(ramp(n, seed, scale), dims, dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    /// The declared parity check: `got` (the GEMV's `[m, n]` output) against the f64
    /// dequantize-then-matmul reference over the weight read back through the codec: every
    /// element within [`gemv_abs_bound`] and the relative RMS within [`GEMV_REL_RMS_TOL`].
    fn assert_matches_dequant_reference(what: &str, w: &Nvfp4Weight, x: &Tensor, got: &Tensor) {
        let (n, k) = w.shape();
        let w_deq = w
            .to_host()
            .unwrap()
            .dequantize()
            .unwrap()
            .narrow(1, 0, k)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let m = x.elem_count() / k;
        let xh = x
            .reshape((m, k))
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let reference = xh.matmul(&w_deq.t().unwrap()).unwrap();
        let abs_dot = xh
            .abs()
            .unwrap()
            .matmul(&w_deq.abs().unwrap().t().unwrap())
            .unwrap();
        let reference = reference.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let abs_dot = abs_dot.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let got = got
            .reshape((m, n))
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..got.len() {
            let err = (got[i] - reference[i]).abs();
            num += err * err;
            den += reference[i] * reference[i];
            let bound = gemv_abs_bound(reference[i], abs_dot[i], k);
            assert!(
                err <= bound,
                "{what}: element {i} (m={}, n={}) |{} - {}| = {err} > {bound}",
                i / n,
                i % n,
                got[i],
                reference[i]
            );
        }
        let rel = (num / den.max(1e-300)).sqrt();
        assert!(rel <= GEMV_REL_RMS_TOL, "{what}: rel-RMS {rel}");
    }

    /// Every row count 1..=8 on shapes that exercise the layout's
    /// edges: K not a multiple of the 16-block (17, 1000, 2049), K padded to 32 but not to the
    /// 64-column unit (80 → 96), K not a multiple of 8 (the element-wise activation path), more
    /// than one 128-row scale atom (272 rows) and more than one 4-block atom column.
    #[test]
    fn gemv_matches_the_dequant_reference_on_edge_shapes() {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return;
        };
        for (n, k) in [(16, 17), (96, 80), (32, 1000), (48, 2049), (272, 512)] {
            let w = Nvfp4Weight::quantize(
                &bf16(&[n, k], 0x9e37 + (n * k) as u64, 0.5, &device),
                None,
                &ctx,
            )
            .unwrap();
            for m in 1..=NVFP4_GEMV_MAX_ROWS {
                let x = bf16(&[m, k], 0xacdc + m as u64, 2.0, &device);
                let y = w.forward_gemv(&x).unwrap();
                assert_eq!(y.dims(), &[m, n]);
                assert_eq!(y.dtype(), DType::BF16);
                assert_matches_dequant_reference(&format!("[{n},{k}] m={m}"), &w, &x, &y);
            }
        }
    }

    /// Rank-3 input, a strided (narrowed, offset) activation view and a bias.
    #[test]
    fn gemv_serves_rank3_views_and_bias() {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return;
        };
        let (n, k) = (64, 256);
        let bias = bf16(&[n], 7, 1.0, &device);
        let w = Nvfp4Weight::quantize(&bf16(&[n, k], 3, 0.5, &device), Some(bias.clone()), &ctx)
            .unwrap();
        let wide = bf16(&[1, 5, k + 8], 11, 1.0, &device);
        let x = wide.narrow(1, 2, 3).unwrap().narrow(2, 3, k).unwrap(); // [1, 3, k], strided
        let y = w.forward_gemv(&x).unwrap();
        assert_eq!(y.dims(), &[1, 3, n]);
        // The bias is added in bf16 after the GEMV's own rounding; compare the GEMV part in f32.
        let y_nobias = y
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_sub(&bias.to_dtype(DType::F32).unwrap())
            .unwrap()
            .reshape((3, n))
            .unwrap();
        let unbiased = Nvfp4Weight::quantize(&bf16(&[n, k], 3, 0.5, &device), None, &ctx).unwrap();
        let direct = unbiased.forward_gemv(&x).unwrap().reshape((3, n)).unwrap();
        let diff = (y_nobias - direct.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff <= 0.05, "bias applied: {diff}");
        assert_matches_dequant_reference(
            "rank-3 strided",
            &unbiased,
            &x.contiguous().unwrap(),
            &direct,
        );
    }

    /// Unsupported inputs are refused, typed, before any launch; the cuBLASLt path serves them.
    #[test]
    fn refusals_are_typed() {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return;
        };
        let w = Nvfp4Weight::quantize(&bf16(&[32, 64], 5, 0.5, &device), None, &ctx).unwrap();
        let refusal = |x: &Tensor| match w.forward_gemv(x) {
            Err(Nvfp4GemvError::Refused(r)) => r,
            Err(other) => panic!("expected a refusal, got {other}"),
            Ok(_) => panic!("expected a refusal"),
        };
        let f32x = bf16(&[1, 64], 1, 1.0, &device)
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(refusal(&f32x), Nvfp4GemvRefusal::Dtype(DType::F32));
        assert_eq!(
            refusal(&bf16(&[9, 64], 1, 1.0, &device)),
            Nvfp4GemvRefusal::Rows(9)
        );
        assert_eq!(
            refusal(&bf16(&[2, 5, 64], 1, 1.0, &device)),
            Nvfp4GemvRefusal::Rows(10)
        );
        assert_eq!(
            refusal(&bf16(&[1, 48], 1, 1.0, &device)),
            Nvfp4GemvRefusal::Shape {
                expected_k: 64,
                got_k: 48
            }
        );
        assert_eq!(
            refusal(&bf16(&[1, 64], 1, 1.0, &Device::Cpu)),
            Nvfp4GemvRefusal::NotCuda
        );
        assert_eq!(
            w.forward(&bf16(&[9, 64], 1, 1.0, &device)).unwrap().dims(),
            &[9, 32]
        );
    }

    /// The GEMV compiles through the shared nvrtc seam: once per device, the module shared by
    /// every later launch and every weight, each (rows, rows-per-warp) entry point present.
    #[test]
    fn gemv_compiles_once_through_the_seam() {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return;
        };
        let Device::Cuda(dev) = &device else {
            unreachable!()
        };
        let a = Nvfp4Weight::quantize(&bf16(&[16, 32], 1, 0.5, &device), None, &ctx).unwrap();
        let b = Nvfp4Weight::quantize(&bf16(&[32, 64], 2, 0.5, &device), None, &ctx).unwrap();
        a.forward_gemv(&bf16(&[1, 32], 3, 1.0, &device)).unwrap();
        let attempts = NVFP4_GEMV_SRC.compile_attempts(dev);
        assert_eq!(attempts, 1, "compiled exactly once for this device");
        b.forward_gemv(&bf16(&[4, 64], 4, 1.0, &device)).unwrap();
        a.forward_gemv(&bf16(&[8, 32], 5, 1.0, &device)).unwrap();
        assert_eq!(NVFP4_GEMV_SRC.compile_attempts(dev), attempts);
        let module = NVFP4_GEMV_SRC
            .cached(dev)
            .expect("outcome cached")
            .expect("compiled");
        assert_eq!(module.name(), NVFP4_GEMV_SRC.name);
        module.function(NVFP4_GEMV_FUNCTION).expect("entry point");
    }

    /// The fused path is more accurate than W4A4 (the activation is not quantized): against the
    /// exact dense product it is closer than the cuBLASLt path on the same weight.
    #[test]
    fn gemv_is_closer_to_the_dense_product_than_w4a4() {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return;
        };
        let (n, k) = (256, 1024);
        let dense = bf16(&[n, k], 21, 0.5, &device);
        let w = Nvfp4Weight::quantize(&dense, None, &ctx).unwrap();
        let x = bf16(&[1, k], 22, 1.0, &device);
        let exact = x
            .to_dtype(DType::F32)
            .unwrap()
            .matmul(&dense.to_dtype(DType::F32).unwrap().t().unwrap())
            .unwrap();
        let err = |y: Tensor| -> f32 {
            let d = (y.to_dtype(DType::F32).unwrap() - &exact).unwrap();
            let num = d
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let den = exact
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            (num / den).sqrt()
        };
        let gemv = err(w.forward_gemv(&x).unwrap());
        let w4a4 = err(w.forward(&x).unwrap());
        assert!(gemv < w4a4, "gemv rel {gemv} vs w4a4 rel {w4a4}");
    }
}
