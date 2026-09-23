//! Load-time NVFP4 weights with a **strict** capability gate (sc-24135, epic sc-24128).
//!
//! The media lane's [`Nvfp4Linear`](crate::Nvfp4Linear) is built from an offline-packed host
//! container and, by design (sc-12078), falls back to dequant→bf16 when the FP4 path is unavailable.
//! The LLM lane needs the opposite contract on both counts:
//!
//! - **Quantize at load, on the device.** A 27B decoder holds ~54 GB of bf16 projection weights;
//!   packing them through the scalar CPU packer (and retaining the host container, as
//!   `Nvfp4Linear` does) would cost minutes and ~15 GB of host RAM for nothing. [`Nvfp4Weight`]
//!   instead runs the device-resident bf16 weight through the **same fused quantizer** the W4A4
//!   forward uses for activations. For `K` a multiple of [`NVFP4_K_ALIGN`] (32) its output is
//!   byte-identical to [`Nvfp4Tensor::pack`](crate::Nvfp4Tensor::pack) + staging; otherwise it
//!   equals the packer's output on the input zero-padded to the next multiple of 32 columns (the
//!   packer itself pads `K` only to 16). Either way it is the sc-12078 parity contract: the same
//!   per-tensor `amax / (6·448)` global scale, per-16-block UE4M3 scale and nearest-E2M1 codes,
//!   emitted in cuBLASLt's scale-factor layout. Only the packed bytes stay resident.
//! - **Refuse, never downgrade.** Requesting NVFP4 where the FP4 GEMM cannot run is a load error
//!   carrying a typed [`Nvfp4Refusal`] that names the capability — not a silent dense fallback that
//!   would serve a bf16 footprint under an NVFP4 label (epic E2 / E5).
//!
//! The forward is the one W4A4 implementation shared with `Nvfp4Linear`
//! (`nvfp4_linear::w4a4_forward`): pad M to 16, fused on-device activation quantize, cuBLASLt
//! block-scaled FP4 GEMM, bias, cast back.

use candle_core::{Device, Result, Tensor};

use crate::cublaslt::{
    compute_cap_meets_nvfp4_floor, NVFP4_COMPUTE_CAP_FLOOR, NVFP4_K_ALIGN, NVFP4_N_ALIGN,
};
use crate::nvfp4_linear::Nvfp4Context;

/// The capability name every [`Nvfp4Refusal`] carries.
pub const NVFP4_CAPABILITY: &str = "nvfp4";

/// Why NVFP4 was refused for a load — the typed, capability-naming refusal (sc-24135 AC2).
///
/// Every variant's message starts with the capability name ([`NVFP4_CAPABILITY`]) and states the
/// requirement it failed, so a caller can surface it verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Nvfp4Refusal {
    /// The load device is not a CUDA device (CPU, Metal) — or this is a build without the `cuda`
    /// feature, which cannot produce a CUDA device at all.
    NotCudaDevice {
        /// The device the load was asked to use (`Cpu`, `Metal(..)`).
        device: String,
    },
    /// A CUDA device below the block-scaled FP4 GEMM floor ([`NVFP4_COMPUTE_CAP_FLOOR`], sm_120).
    BelowComputeFloor {
        /// The device's `(major, minor)` compute capability.
        found: (i32, i32),
    },
    /// The cuBLASLt handle could not be created on an otherwise eligible device.
    HandleUnavailable {
        /// The driver / cuBLASLt error text.
        reason: String,
    },
    /// The fused NVFP4 quantizer (nvrtc) does not compile on this device. Weights and activations
    /// are both quantized by it, so without it there is no NVFP4 path worth serving (sc-12078).
    FusedQuantizerUnavailable,
    /// A projection's output dimension is not a multiple of [`NVFP4_N_ALIGN`] (16), which the
    /// cuBLASLt FP4 GEMM requires.
    ShapeIneligible {
        /// Output features (`N`).
        rows: usize,
        /// Input features (`K`).
        cols: usize,
    },
    /// A projection too large for the fused quantizer, which indexes the `K`-padded
    /// `[rows, round_up(cols, 32)]` grid with 32-bit integers: more than `i32::MAX` elements would
    /// silently overflow them.
    ShapeTooLarge {
        /// Output features (`N`).
        rows: usize,
        /// Input features (`K`).
        cols: usize,
    },
}

impl Nvfp4Refusal {
    /// The capability this refusal is about — always [`NVFP4_CAPABILITY`].
    pub fn capability(&self) -> &'static str {
        NVFP4_CAPABILITY
    }
}

impl std::fmt::Display for Nvfp4Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (major, minor) = NVFP4_COMPUTE_CAP_FLOOR;
        write!(f, "{NVFP4_CAPABILITY}: ")?;
        match self {
            Self::NotCudaDevice { device } => write!(
                f,
                "NVFP4 projections need a CUDA device with compute capability >= sm_{major}{minor} \
                 (cuBLASLt block-scaled FP4 GEMM); the load device is {device}"
            ),
            Self::BelowComputeFloor { found } => write!(
                f,
                "NVFP4 projections need compute capability >= sm_{major}{minor} (cuBLASLt \
                 block-scaled FP4 GEMM); this GPU is sm_{}{}",
                found.0, found.1
            ),
            Self::HandleUnavailable { reason } => write!(
                f,
                "NVFP4 projections need a cuBLASLt handle, which could not be created: {reason}"
            ),
            Self::FusedQuantizerUnavailable => write!(
                f,
                "NVFP4 projections need the fused NVFP4 quantizer, which nvrtc could not compile \
                 on this device"
            ),
            Self::ShapeIneligible { rows, cols } => write!(
                f,
                "NVFP4 projection [{rows}, {cols}] is ineligible: the cuBLASLt FP4 GEMM needs the \
                 output dimension to be a multiple of {NVFP4_N_ALIGN}"
            ),
            Self::ShapeTooLarge { rows, cols } => write!(
                f,
                "NVFP4 projection [{rows}, {cols}] is too large: the fused quantizer indexes the \
                 K-padded weight with 32-bit integers, so rows x round_up(cols, {NVFP4_K_ALIGN}) \
                 must not exceed {}",
                i32::MAX
            ),
        }
    }
}

impl std::error::Error for Nvfp4Refusal {}

/// The pure half of the device gate: the refusal a CUDA device of compute capability `cap` earns,
/// or `None` when it meets the NVFP4 floor. Split out so the sub-sm_120 refusal is testable with a
/// mocked capability on any host (sc-24135 AC2).
pub fn nvfp4_refusal_for_compute_cap(cap: (i32, i32)) -> Option<Nvfp4Refusal> {
    (!compute_cap_meets_nvfp4_floor(cap)).then_some(Nvfp4Refusal::BelowComputeFloor { found: cap })
}

/// The shape half of the gate: `Ok` iff a `[rows, cols]` weight can be served by the FP4 GEMM.
/// `K` is padded to the GEMM's alignment at quantization time, so its alignment never
/// disqualifies; `N` must be a multiple of [`NVFP4_N_ALIGN`], and the `K`-padded element count
/// must fit the fused quantizer's 32-bit indexing ([`Nvfp4Refusal::ShapeTooLarge`]). Pure
/// arithmetic — no allocation — so a loader can settle it before reading the weight.
pub fn nvfp4_shape_refusal(rows: usize, cols: usize) -> std::result::Result<(), Nvfp4Refusal> {
    if rows == 0 || cols == 0 || !rows.is_multiple_of(NVFP4_N_ALIGN) {
        return Err(Nvfp4Refusal::ShapeIneligible { rows, cols });
    }
    let fits_i32 = cols
        .checked_next_multiple_of(NVFP4_K_ALIGN)
        .and_then(|cols_padded| rows.checked_mul(cols_padded))
        .is_some_and(|elems| elems <= i32::MAX as usize);
    if !fits_i32 {
        return Err(Nvfp4Refusal::ShapeTooLarge { rows, cols });
    }
    Ok(())
}

impl Nvfp4Context {
    /// The **strict** twin of [`Nvfp4Context::new`]: one shared cuBLASLt handle for `device`, or a
    /// typed [`Nvfp4Refusal`] naming why NVFP4 cannot run there. Never an empty context.
    ///
    /// Settles, in order: a CUDA device; a cuBLASLt handle; compute capability ≥ sm_120; the fused
    /// quantizer compiling. This is the whole capability floor, so a loader calls it **before**
    /// reading any weights (epic E5: declared and validated before launch).
    pub fn require(device: &Device) -> std::result::Result<Self, Nvfp4Refusal> {
        #[cfg(feature = "cuda")]
        {
            Self::require_with(device, crate::cublaslt::CublasLt::compute_cap)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Nvfp4Refusal::NotCudaDevice {
                device: format!("{:?}", device.location()),
            })
        }
    }

    /// [`Self::require`] with the compute-capability probe injected: `cap_probe` is asked for the
    /// device's `(major, minor)` in place of the handle's real query, so the whole gate — handle,
    /// capability refusal, quantizer — can be exercised with a mocked sub-sm_120 capability on an
    /// sm_120 host. [`Self::require`] is exactly this with [`CublasLt::compute_cap`](crate::CublasLt::compute_cap).
    #[cfg(feature = "cuda")]
    pub fn require_with(
        device: &Device,
        cap_probe: impl FnOnce(&crate::cublaslt::CublasLt) -> Result<(i32, i32)>,
    ) -> std::result::Result<Self, Nvfp4Refusal> {
        if !device.is_cuda() {
            return Err(Nvfp4Refusal::NotCudaDevice {
                device: format!("{:?}", device.location()),
            });
        }
        let lt = crate::cublaslt::CublasLt::new(device).map_err(|e| {
            Nvfp4Refusal::HandleUnavailable {
                reason: e.to_string(),
            }
        })?;
        let cap = cap_probe(&lt).map_err(|e| Nvfp4Refusal::HandleUnavailable {
            reason: e.to_string(),
        })?;
        if let Some(refusal) = nvfp4_refusal_for_compute_cap(cap) {
            return Err(refusal);
        }
        if !lt.nvfp4_fused_quantizer_available() {
            return Err(Nvfp4Refusal::FusedQuantizerUnavailable);
        }
        Ok(Self {
            inner: Some(crate::nvfp4_linear::Fp4Ctx {
                lt: std::sync::Arc::new(lt),
                device: device.clone(),
            }),
        })
    }
}

/// A projection weight quantized to NVFP4 **at load**, resident on-device as packed E2M1 nibbles +
/// UE4M3 block scales (~4.5 bits/weight), served by the W4A4 cuBLASLt FP4 GEMM.
///
/// Only constructible with the `cuda` feature (through [`Self::quantize`] with a context from
/// [`Nvfp4Context::require`]); on other builds the type exists so callers compile unchanged, but it
/// is uninhabited.
pub struct Nvfp4Weight {
    rows: usize,
    cols: usize,
    bias: Option<Tensor>,
    #[cfg(feature = "cuda")]
    lt: std::sync::Arc<crate::cublaslt::CublasLt>,
    #[cfg(feature = "cuda")]
    staged: crate::cublaslt::DevNvfp4,
    /// The CUDA device the packed weight lives on (the decode GEMV refuses an activation on
    /// another device rather than reading across contexts).
    #[cfg(feature = "cuda")]
    device: Device,
    #[cfg(not(feature = "cuda"))]
    _uninhabited: std::convert::Infallible,
}

impl Nvfp4Weight {
    /// Quantize a dense `[out, in]` weight (any float dtype, on `ctx`'s device) to a resident NVFP4
    /// weight. The dense tensor is not retained; drop it to release its memory.
    ///
    /// Errors when the shape is ineligible or too large for the quantizer's 32-bit indexing
    /// ([`nvfp4_shape_refusal`] — call it first to get the typed refusal), when `ctx` holds no
    /// handle for the weight's device, or on a device fault.
    pub fn quantize(weight: &Tensor, bias: Option<Tensor>, ctx: &Nvfp4Context) -> Result<Self> {
        let (rows, cols) = weight.dims2()?;
        if let Err(refusal) = nvfp4_shape_refusal(rows, cols) {
            candle_core::bail!("{refusal}");
        }
        #[cfg(feature = "cuda")]
        {
            let lt = ctx.handle_for(weight.device()).map_err(|why| {
                candle_core::Error::Msg(format!(
                    "{NVFP4_CAPABILITY}: no NVFP4 handle for the weight's device ({why:?})"
                ))
            })?;
            // `nvfp4_shape_refusal` above bounds `rows * cols_padded` by `i32::MAX`.
            let cols_padded = cols.div_ceil(NVFP4_K_ALIGN) * NVFP4_K_ALIGN;
            // The fused quantizer is layout-agnostic between operands: it emits the row-major
            // `[rows, cols_padded]` nibbles and cuBLASLt's row-major scale-factor-atom layout for
            // whichever operand it is given. For `cols % 32 == 0` that is byte-identical to
            // `Nvfp4Tensor::pack` + staging; otherwise it equals the packer's output on the input
            // zero-padded to `cols_padded` columns (the packer pads `K` only to 16).
            let staged = lt.quantize_nvfp4_activation_fused(&weight.contiguous()?, cols_padded)?;
            Ok(Self {
                rows,
                cols,
                bias,
                lt: std::sync::Arc::clone(lt),
                staged,
                device: weight.device().clone(),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (bias, ctx);
            candle_core::bail!(
                "{}",
                Nvfp4Refusal::NotCudaDevice {
                    device: format!("{:?}", weight.device().location()),
                }
            )
        }
    }

    /// `y = x·Wᵀ (+ b)` through the W4A4 FP4 GEMM. Accepts a rank-≥1 activation `[..., in]` and
    /// returns `[..., out]` in the activation's dtype.
    ///
    /// This is the cuBLASLt path at every row count. For decode-sized inputs (≤
    /// [`NVFP4_GEMV_MAX_ROWS`](crate::NVFP4_GEMV_MAX_ROWS) rows) the fused W4A16 GEMV,
    /// `forward_gemv` (only compiled with the `cuda` feature), is the other implementation; the
    /// caller picks (sc-24136).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            crate::nvfp4_linear::w4a4_forward(&self.lt, &self.staged, x, self.bias.as_ref())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = x;
            match self._uninhabited {}
        }
    }

    /// The logical `[out, in]` weight shape.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The optional `[out]` bias.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// The staged packed operand (nibbles + swizzled block scales + global scale).
    #[cfg(feature = "cuda")]
    pub(crate) fn staged(&self) -> &crate::cublaslt::DevNvfp4 {
        &self.staged
    }

    /// The device the packed weight lives on.
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resident device bytes of the packed weight (E2M1 nibbles + UE4M3 block scales) plus the
    /// bias, if any. This is the whole on-device footprint of the projection.
    pub fn resident_bytes(&self) -> usize {
        let bias = self
            .bias
            .as_ref()
            .map_or(0, |b| b.elem_count() * b.dtype().size_in_bytes());
        #[cfg(feature = "cuda")]
        {
            self.staged.resident_bytes() + bias
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = bias;
            match self._uninhabited {}
        }
    }

    /// Read the resident weight back as a host [`Nvfp4Tensor`](crate::Nvfp4Tensor) (the codec's
    /// canonical layout), for parity tests against the CPU packer and the dequant reference.
    /// Byte-level test support, never on a hot path.
    #[cfg(feature = "cuda")]
    pub fn to_host(&self) -> Result<crate::Nvfp4Tensor> {
        use crate::nvfp4::{NVFP4_BLOCK, SF_ATOM_COLS, SF_ATOM_ROWS};
        let (_, cols_padded) = self.staged.shape_padded();
        let packed = self.staged.packed_to_host(&self.lt)?;
        let device_scales = self.staged.scales_to_host(&self.lt)?;
        let n_blocks = cols_padded / NVFP4_BLOCK;
        let sf_rows = self.rows.div_ceil(SF_ATOM_ROWS) * SF_ATOM_ROWS;
        let sf_cols = n_blocks.div_ceil(SF_ATOM_COLS) * SF_ATOM_COLS;
        // Invert cuBLASLt's row-major atom tiling back into the container's own offset.
        let num_k_atoms = sf_cols / SF_ATOM_COLS;
        let mut scales = vec![0u8; sf_rows * sf_cols];
        for r in 0..sf_rows {
            for blk in 0..sf_cols {
                let (mr, kc) = (r % SF_ATOM_ROWS, blk % SF_ATOM_COLS);
                let atom = blk / SF_ATOM_COLS + num_k_atoms * (r / SF_ATOM_ROWS);
                let intra = (mr % 32) * 16 + (mr / 32) * 4 + kc;
                scales[crate::Nvfp4Tensor::scale_offset_for(r, blk, sf_rows)] =
                    device_scales[atom * SF_ATOM_ROWS * SF_ATOM_COLS + intra];
            }
        }
        Ok(crate::Nvfp4Tensor {
            rows: self.rows,
            cols: cols_padded,
            cols_padded,
            packed,
            scales,
            sf_rows,
            sf_cols,
            global_scale: self.staged.global_scale(),
        })
    }
}

impl std::fmt::Debug for Nvfp4Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nvfp4Weight")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_sm120_capabilities_are_refused_by_name() {
        for cap in [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (11, 0)] {
            let refusal = nvfp4_refusal_for_compute_cap(cap).expect("below the floor");
            assert_eq!(refusal, Nvfp4Refusal::BelowComputeFloor { found: cap });
            assert_eq!(refusal.capability(), "nvfp4");
            let msg = refusal.to_string();
            assert!(msg.starts_with("nvfp4: "), "{msg}");
            assert!(msg.contains("sm_120"), "names the floor: {msg}");
            assert!(
                msg.contains(&format!("sm_{}{}", cap.0, cap.1)),
                "names the device: {msg}"
            );
        }
        for cap in [(12, 0), (12, 1), (13, 0)] {
            assert_eq!(nvfp4_refusal_for_compute_cap(cap), None, "{cap:?}");
        }
    }

    #[test]
    fn cpu_device_is_refused_before_any_handle_is_built() {
        let Err(refusal) = Nvfp4Context::require(&Device::Cpu) else {
            panic!("CPU cannot serve NVFP4");
        };
        assert!(matches!(refusal, Nvfp4Refusal::NotCudaDevice { .. }));
        let msg = refusal.to_string();
        assert!(
            msg.starts_with("nvfp4: ") && msg.contains("sm_120"),
            "{msg}"
        );
        assert!(msg.contains("Cpu"), "names the device: {msg}");
    }

    #[test]
    fn only_the_output_dimension_can_disqualify_a_shape() {
        assert!(nvfp4_shape_refusal(5120, 5120).is_ok());
        assert!(
            nvfp4_shape_refusal(16, 8).is_ok(),
            "K is padded at quantization"
        );
        assert_eq!(
            nvfp4_shape_refusal(8, 64),
            Err(Nvfp4Refusal::ShapeIneligible { rows: 8, cols: 64 })
        );
        assert!(nvfp4_shape_refusal(0, 64).is_err());
    }

    /// The fused quantizer indexes the K-padded grid with 32-bit integers (`long` is 32-bit on
    /// MSVC), so a shape whose `rows × round_up(cols, 32)` exceeds `i32::MAX` is refused up front
    /// — settled by arithmetic alone, nothing allocated.
    #[test]
    fn a_shape_past_the_quantizers_32_bit_indexing_is_refused() {
        // 65536 × 32768 = 2^31 = i32::MAX + 1: one element too many.
        assert_eq!(
            nvfp4_shape_refusal(65_536, 32_768),
            Err(Nvfp4Refusal::ShapeTooLarge {
                rows: 65_536,
                cols: 32_768
            })
        );
        // K = 32_737 pads to 32_768, so the padded (not the logical) count is what overflows.
        assert_eq!(
            nvfp4_shape_refusal(65_536, 32_737),
            Err(Nvfp4Refusal::ShapeTooLarge {
                rows: 65_536,
                cols: 32_737
            })
        );
        // The largest padded grid that fits: 65536 × 32736 < 2^31.
        assert!(nvfp4_shape_refusal(65_536, 32_736).is_ok());
        // usize overflow in the product is a refusal, not a wrap (2^40 · 2^40 wraps to 0).
        assert!(nvfp4_shape_refusal(1 << 40, 1 << 40).is_err());
        assert!(nvfp4_shape_refusal(usize::MAX - 15, usize::MAX).is_err());
        let msg = Nvfp4Refusal::ShapeTooLarge {
            rows: 65_536,
            cols: 32_768,
        }
        .to_string();
        assert!(
            msg.starts_with("nvfp4: ") && msg.contains("32-bit"),
            "{msg}"
        );
    }

    #[test]
    fn quantize_on_cpu_is_an_error_not_a_dense_fallback() -> Result<()> {
        let w = Tensor::zeros((16, 32), candle_core::DType::F32, &Device::Cpu)?;
        let err = Nvfp4Weight::quantize(&w, None, &Nvfp4Context::none())
            .expect_err("no NVFP4 weight on CPU");
        assert!(err.to_string().contains("nvfp4"), "{err}");
        Ok(())
    }

    /// A CUDA device that meets the NVFP4 floor, or `None` (the GPU tests then skip: CI's CUDA
    /// runners are sm_120, a developer's older GPU is not).
    #[cfg(feature = "cuda")]
    fn nvfp4_device() -> Option<(Device, Nvfp4Context)> {
        let device = Device::new_cuda(0).ok()?;
        let ctx = Nvfp4Context::require(&device).ok()?;
        Some((device, ctx))
    }

    #[cfg(feature = "cuda")]
    fn ramp(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..rows * cols)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The load-time on-device quantization is the codec: byte-identical nibbles, block scales and
    /// global scale to the CPU packer's [`Nvfp4Tensor::pack_from_slice`](crate::Nvfp4Tensor), on a
    /// multi-atom grid (256 rows = 2 row atoms, 256 cols = 16 blocks = 4 block atoms).
    #[cfg(feature = "cuda")]
    #[test]
    fn load_time_quantization_matches_the_cpu_packer_byte_for_byte() -> Result<()> {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return Ok(());
        };
        let (rows, cols) = (256, 256);
        let data = ramp(rows, cols, 0x5EED_2413_5000_0001);
        let w = Tensor::from_vec(data.clone(), (rows, cols), &device)?;
        let weight = Nvfp4Weight::quantize(&w, None, &ctx)?;
        let host = weight.to_host()?;
        let cpu = crate::Nvfp4Tensor::pack_from_slice(&data, rows, cols)?;
        assert_eq!(host.global_scale, cpu.global_scale);
        assert_eq!(host.packed, cpu.packed, "E2M1 nibbles");
        assert_eq!(host.scales, cpu.scales, "UE4M3 block scales");
        assert_eq!(weight.resident_bytes(), cpu.packed.len() + cpu.scales.len());
        Ok(())
    }

    /// The fused NVFP4 quantizer compiles through the shared nvrtc seam (sc-24137): a fresh
    /// `CublasLt` handle resolves its functions from the process-wide module, so nvrtc runs once
    /// per device however many handles probe it.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_fused_quantizer_compiles_once_per_device_through_the_seam() {
        let Some((device, _ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return;
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let src = crate::cublaslt::NVFP4_QUANT_SRC;
        for _ in 0..3 {
            let lt = crate::CublasLt::new(&device).expect("handle");
            assert!(lt.nvfp4_fused_quantizer_available());
        }
        assert!(matches!(src.cached(cuda), Some(Ok(_))), "the seam holds it");
        assert_eq!(src.compile_attempts(cuda), 1, "nvrtc ran once");
    }

    /// For `K` not a multiple of 32 the device pads `K` to 32 while the CPU packer pads only to 16,
    /// so parity is against the packer run on the input zero-padded to 32 (here 80 → 96 columns;
    /// 96 rows is below one 128-row scale atom).
    #[cfg(feature = "cuda")]
    #[test]
    fn load_time_quantization_of_an_unaligned_k_matches_the_packer_on_the_padded_input(
    ) -> Result<()> {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return Ok(());
        };
        let (rows, cols, cols_padded) = (96, 80, 96);
        let data = ramp(rows, cols, 0x5EED_2413_5000_0003);
        let w = Tensor::from_vec(data.clone(), (rows, cols), &device)?;
        let weight = Nvfp4Weight::quantize(&w, None, &ctx)?;
        assert_eq!(weight.shape(), (rows, cols));
        let host = weight.to_host()?;
        let padded: Vec<f32> = data
            .chunks(cols)
            .flat_map(|row| {
                row.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, cols_padded - cols))
            })
            .collect();
        let cpu = crate::Nvfp4Tensor::pack_from_slice(&padded, rows, cols_padded)?;
        assert_eq!(host.cols_padded, cpu.cols_padded);
        assert_eq!(host.global_scale, cpu.global_scale);
        assert_eq!(host.packed, cpu.packed, "E2M1 nibbles");
        assert_eq!(host.scales, cpu.scales, "UE4M3 block scales");
        assert_eq!(weight.resident_bytes(), cpu.packed.len() + cpu.scales.len());
        Ok(())
    }

    /// A contiguous f32 view with a start offset (a row narrow) must quantize the rows it names,
    /// not the storage's first rows — the fused quantizer used to index from element 0.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_row_narrowed_f32_view_quantizes_its_own_rows() -> Result<()> {
        let Some((device, ctx)) = nvfp4_device() else {
            eprintln!("skipping: no sm_120 CUDA device");
            return Ok(());
        };
        let full = Tensor::from_vec(ramp(64, 64, 0xA11C_E5EE_D000_0002), (64, 64), &device)?;
        let view = full.narrow(0, 32, 32)?;
        assert!(view.is_contiguous() && view.layout().start_offset() != 0);
        let from_view = Nvfp4Weight::quantize(&view, None, &ctx)?.to_host()?;
        let from_copy = Nvfp4Weight::quantize(&view.copy()?, None, &ctx)?.to_host()?;
        assert_eq!(from_view.packed, from_copy.packed);
        assert_eq!(from_view.scales, from_copy.scales);
        Ok(())
    }
}
