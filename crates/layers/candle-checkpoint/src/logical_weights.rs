//! Candle **mapped logical-weight reader** and the engine's **codec table** (epic 20398,
//! sc-20385) — the Candle twin of `mlx_gen::logical_weights`.
//!
//! The provider supplies its adapter-owned [`LogicalKeyMapping`]; [`plan_logical_weights`]
//! compiles the safetensors header plus the `.comfy_quant` descriptor payloads into a
//! [`LogicalWeightPlan`] against [`baseline_codec_registry`] and this backend's
//! [`CandleCodecResidency`] policy; [`read_logical_weights`] then materializes exactly the planned
//! tensors, decoding **per layer** (mixed checkpoints dispatch tensor-by-tensor) and measuring
//! what each codec left resident.
//!
//! Matrix constructors retain original E4M3/E5M2 scalar codes, BFL rowwise E4M3,
//! MXFP8/E8M0 blocks, NVFP4 blocks or ConvRot INT8 codes according to CandleCodecResidency.
//! Each CUDA kernel has its own device and layout requirements. Model-declared shapes
//! determine the logical dimensions of padded blocks.
//!
//! Coefficient consumers decode supported source formats to the requested F32, BF16 or F16 dtype.
//! Scalar weight-only projections retain their codes and preserve activation precision.
//! Measured receipts charge the actual resident
//! representation, retained scale operands and any dense coefficient copies.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::OnceLock;

#[cfg(feature = "model-inference-proof")]
static DENSE_FP8_REFERENCE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

use gen_core::checkpoint_codec::{
    CheckpointCodecRegistration, CheckpointCodecRegistry, CodecResidencyPolicy, LogicalKeyMapping,
    LogicalTensorPlan, LogicalWeightPlan, LogicalWeightReceipt, ResidencyMode, ScalarScaleSource,
    TensorCodecSpec, WeightEncoding, DENSE_BF16_CODEC, DENSE_F16_CODEC, DENSE_F32_CODEC,
    FP8_E4M3_ROWWISE_CODEC, FP8_E4M3_SCALAR_CODEC, FP8_E5M2_SCALAR_CODEC, INT8_PER_ROW_CODEC,
    MXFP8_CODEC, NVFP4_CODEC,
};
use gen_core::checkpoint_facts::{
    CheckpointWeightFacts, ExecutionRepresentation, NativeExecutionCapability,
};
use gen_core::ProviderRegistryBuilder;

use crate::nvfp4_kitchen::Nvfp4KitchenExt as _;
use candle_quant_kernels::Nvfp4Tensor;

use candle_core::safetensors::{self as cst, MmapedSafetensors};
use candle_core::{DType, Device, Tensor};
use candle_core::{Error as CandleError, Result};

/// The codec rows this engine implements and registers — identical to the MLX table (codecs are
/// backend-portable declarations; each engine owns its implementation).
pub const BASELINE_CODECS: &[CheckpointCodecRegistration] =
    gen_core::checkpoint_codec::BASELINE_CHECKPOINT_CODECS;

/// The codec ids this engine has a decode implementation for. Must equal the ids of
/// [`BASELINE_CODECS`] (the catalog test proves it).
pub const CODEC_IMPLEMENTATION_IDS: &[&str] = &[
    DENSE_BF16_CODEC.codec_id,
    DENSE_F16_CODEC.codec_id,
    DENSE_F32_CODEC.codec_id,
    FP8_E4M3_SCALAR_CODEC.codec_id,
    FP8_E5M2_SCALAR_CODEC.codec_id,
    FP8_E4M3_ROWWISE_CODEC.codec_id,
    MXFP8_CODEC.codec_id,
    gen_core::convrot_int8::CODEC_ID,
    INT8_PER_ROW_CODEC.codec_id,
    NVFP4_CODEC.codec_id,
];

/// Register the engine's codec table exactly once into a platform catalog.
pub fn register_checkpoint_codecs(mut builder: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    for codec in BASELINE_CODECS {
        builder = builder.register_checkpoint_codec(*codec);
    }
    builder
}

/// The validated registry the loaders plan against.
pub fn baseline_codec_registry() -> &'static CheckpointCodecRegistry {
    static REGISTRY: OnceLock<CheckpointCodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CheckpointCodecRegistry::new(BASELINE_CODECS.iter().copied())
            .expect("the engine codec table is a valid registry")
    })
}

/// Candle's packed-vs-dense residency decision. Layout facts are enforced by the plan compiler (and,
/// for NVFP4, by [`nvfp4_layout_is_native`]); the hardware facts — a CUDA device at the cuBLASLt fp8
/// leg's sm_89 floor, and at the NVFP4 leg's sm_120 floor — are probed once per device and carried
/// here, so the policy itself stays a pure, unit-testable predicate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandleCodecResidency {
    /// The bound device runs the cuBLASLt E4M3 GEMM (CUDA, compute capability ≥ 8.9).
    pub fp8_e4m3_native: bool,
    /// The bound device runs the cuBLASLt NVFP4 block-scaled FP4 GEMM (CUDA, compute capability
    /// ≥ 12.0 — consumer Blackwell `sm_120`).
    pub nvfp4_native: bool,
    /// Native VEC32_UE8M0 block scaling is available on Blackwell.
    pub mxfp8_native: bool,
}

impl CandleCodecResidency {
    /// One-way reference policy for a fresh, separately guarded model-proof process.
    /// Ordinary product builds expose no precision override.
    #[cfg(feature = "model-inference-proof")]
    pub fn force_dense_fp8_for_model_proof() {
        DENSE_FP8_REFERENCE.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The dense-only policy (CPU, Metal, or CUDA below both floors).
    pub const DENSE: Self = Self {
        fp8_e4m3_native: false,
        nvfp4_native: false,
        mxfp8_native: false,
    };

    /// Render this policy as the backend-neutral **host capability** that crosses the worker
    /// boundary (sc-21484): the codec rows this host can execute in their stored packing, with no
    /// checkpoint in hand.
    ///
    /// This is the declaration [`gen_core::checkpoint_facts::CheckpointWeightFacts`] checks a
    /// native receipt row against, so a host below a floor cannot produce facts that label its run
    /// native. Datacenter `sm_100` is below the NVFP4 leg's `(12, 0)` floor
    /// ([`candle_quant_kernels::NVFP4_COMPUTE_CAP_FLOOR`]) and so lists nothing here — the exclusion is the
    /// floor comparison itself, not a special case.
    pub fn native_execution_capability(&self) -> NativeExecutionCapability {
        let mut ids: Vec<&'static str> = Vec::new();
        if self.fp8_e4m3_native {
            ids.extend([
                FP8_E4M3_SCALAR_CODEC.codec_id,
                FP8_E5M2_SCALAR_CODEC.codec_id,
                FP8_E4M3_ROWWISE_CODEC.codec_id,
            ]);
            ids.push(gen_core::convrot_int8::CODEC_ID);
        }
        if self.mxfp8_native {
            ids.push(MXFP8_CODEC.codec_id);
        }
        if self.nvfp4_native {
            ids.push(NVFP4_CODEC.codec_id);
        }
        NativeExecutionCapability::new(ids)
    }

    /// Probe the device's eligibility for each native leg. Non-CUDA devices (and every build
    /// without the `cuda` feature) are dense-only; a CUDA device whose capability cannot be read is
    /// treated as below both floors (dense fallback — the safe direction, never the silent-packed
    /// one).
    pub fn probe(device: &Device) -> Self {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = device {
            if let Ok(cap) = candle_quant_kernels::device_compute_cap(dev) {
                #[allow(unused_mut)]
                let mut policy = Self::from_cuda_compute_cap(cap);
                #[cfg(feature = "model-inference-proof")]
                if DENSE_FP8_REFERENCE.load(std::sync::atomic::Ordering::SeqCst) {
                    policy.fp8_e4m3_native = false;
                }
                return policy;
            }
        }
        let _ = device;
        Self::DENSE
    }

    /// Probe the visible default CUDA ordinal used by the media loader without creating a
    /// device context, cuBLAS handle or workspace. Unprobed devices receive dense admission.
    pub fn default_device_for_admission() -> Self {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::{
                result, sys::CUdevice_attribute as Attr,
            };
            static POLICY: std::sync::OnceLock<CandleCodecResidency> = std::sync::OnceLock::new();
            *POLICY.get_or_init(|| {
                let cap = || {
                    result::init().ok()?;
                    let dev = result::device::get(0).ok()?;
                    // SAFETY: dev is a successfully enumerated CUdevice; both attributes are scalar.
                    unsafe {
                        Some((
                            result::device::get_attribute(
                                dev,
                                Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                            )
                            .ok()?,
                            result::device::get_attribute(
                                dev,
                                Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                            )
                            .ok()?,
                        ))
                    }
                };
                cap()
                    .map(Self::from_cuda_compute_cap)
                    .unwrap_or(Self::DENSE)
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::DENSE
        }
    }

    /// Eligibility from a CUDA capability already measured by the process (e.g. NVML at startup).
    pub fn from_cuda_compute_cap(cap: (i32, i32)) -> Self {
        Self {
            fp8_e4m3_native: candle_quant_kernels::compute_cap_meets_fp8_floor(cap),
            nvfp4_native: candle_quant_kernels::compute_cap_meets_nvfp4_floor(cap),
            mxfp8_native: candle_quant_kernels::cublaslt::compute_cap_meets_mxfp8_floor(cap),
        }
    }
}

/// Whether one NVFP4 layer's **stored layout** is one the cuBLASLt FP4 GEMM accepts, independent of
/// hardware: the contraction K (the logical `in_features`) must be a multiple of
/// [`candle_quant_kernels::NVFP4_K_ALIGN`] and the output N a multiple of [`candle_quant_kernels::NVFP4_N_ALIGN`] —
/// the same pair `check_nvfp4_alignment` enforces at GEMM time, read from the one definition rather
/// than re-spelled here.
///
/// ComfyUI pads NVFP4 storage only to 16, so a layer with `in_features = 48` is a legitimate NVFP4
/// checkpoint this leg still cannot run; it takes the dense fallback at *plan* time (and is priced
/// as dense) instead of failing at the first forward.
///
/// **Alignment is not the whole native contract.** ComfyUI's padding also means a layer can be
/// stored wider than it is: `in_features = 60` stores `K = 64`, which *is* 32-aligned. Repacking
/// that grid would hand `Nvfp4Linear` 4 columns of padding as real contraction elements, so
/// [`CodecResidencyPolicy::residency`] additionally requires `logical_shape == stored_shape`
/// (sc-20641). This predicate answers only the GEMM's alignment question.
pub fn nvfp4_layout_is_native(stored_shape: [usize; 2]) -> bool {
    stored_shape[1].is_multiple_of(candle_quant_kernels::NVFP4_K_ALIGN)
        && stored_shape[0].is_multiple_of(candle_quant_kernels::NVFP4_N_ALIGN)
}

impl CodecResidencyPolicy for CandleCodecResidency {
    fn residency(
        &self,
        codec: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        stored_shape: &[usize],
    ) -> ResidencyMode {
        // Scalar FP8 has a packed weight-only operator; other full-precision codecs still decode.
        if spec.full_precision_matrix_mult() && !matches!(spec, TensorCodecSpec::ScalarFp8 { .. }) {
            return ResidencyMode::Dense;
        }
        if self.mxfp8_native && matches!(spec, TensorCodecSpec::Mxfp8 { .. }) {
            return ResidencyMode::Packed;
        }
        if let TensorCodecSpec::ConvRotInt8 { geometry, .. } = spec {
            return if self.fp8_e4m3_native && geometry.out_features.is_multiple_of(16) {
                ResidencyMode::Packed
            } else {
                ResidencyMode::Dense
            };
        }
        // Scalar and per-token scaled multiplication both require a two-dimensional,
        // aligned GEMM operand. Biases and positional coefficients take the dense path.
        if self.fp8_e4m3_native
            && [
                FP8_E4M3_SCALAR_CODEC.codec_id,
                FP8_E5M2_SCALAR_CODEC.codec_id,
                FP8_E4M3_ROWWISE_CODEC.codec_id,
            ]
            .contains(&codec.codec_id)
            && candle_quant_kernels::cublaslt::eight_bit_layout_supported(stored_shape)
        {
            return ResidencyMode::Packed;
        }
        // NVFP4 needs the hardware floor, a layout the FP4 leg accepts, *and* a stored grid that is
        // the layer itself. The compiler's `stored_shape` argument is the on-disk byte shape
        // `[rows, cols / 2]`; the codec spec carries both element shapes the rules are written
        // against. `logical != stored` means ComfyUI padded the layer, and the packed container has
        // nowhere to record the unpad — the repacked operand would contract over padding — so a
        // padded layer takes the dense fallback no matter how well the padded grid aligns.
        if self.nvfp4_native && codec.codec_id == NVFP4_CODEC.codec_id {
            if let TensorCodecSpec::Nvfp4 {
                stored_shape,
                logical_shape,
                ..
            } = spec
            {
                if logical_shape == stored_shape && nvfp4_layout_is_native(*stored_shape) {
                    return ResidencyMode::Packed;
                }
            }
        }
        ResidencyMode::Dense
    }
}

/// Header-plus-descriptors plan for one safetensors file under the given mapping and residency
/// policy. Refuses before any tensor is created; the error names the exact on-disk tensor.
pub fn plan_logical_weights(
    path: &Path,
    mapping: &dyn LogicalKeyMapping,
    residency: &dyn CodecResidencyPolicy,
) -> Result<LogicalWeightPlan> {
    gen_core::checkpoint_codec::plan_safetensors(
        path,
        mapping,
        baseline_codec_registry(),
        residency,
    )
    .map_err(|e| CandleError::Msg(e.to_string()))
}

/// [`plan_logical_weights`] for a native single file whose **W4A8** projections are served by
/// `W4A8Linear` instead of the plan (SW-6135): see
/// [`gen_core::checkpoint_codec::plan_safetensors_delegating_w4a8`]. Open the reader with
/// [`LogicalWeightReader::open_with_capability_delegated`] and the returned delegation's keys.
pub fn plan_logical_weights_delegating_w4a8(
    path: &Path,
    mapping: &dyn LogicalKeyMapping,
    residency: &dyn CodecResidencyPolicy,
) -> Result<(LogicalWeightPlan, gen_core::w4a8_int8::W4A8Delegation)> {
    gen_core::checkpoint_codec::plan_safetensors_delegating_w4a8(
        path,
        mapping,
        baseline_codec_registry(),
        residency,
    )
    .map_err(|e| CandleError::Msg(e.to_string()))
}

/// One decoded logical tensor.
pub enum LogicalTensor {
    /// Original, padded E4M3 matrix and contiguous swizzled E8M0 block-scale surface.
    PackedRowwiseFp8 { codes: Tensor, scales: Tensor },
    PackedMxFp8 {
        codes: Tensor,
        scales: Tensor,
        logical: (usize, usize),
    },
    /// Exact two's-complement storage bytes, F32 row scales, and required online rotation.
    PackedConvRot {
        codes: Tensor,
        scales: Tensor,
        geometry: gen_core::convrot_int8::ConvRotInt8,
    },
    /// The dense resident form (pass-through dtype for dense rows, bf16 for dequantized rows).
    Dense(Tensor),
    /// The packed-native fp8 E4M3 form: the stored codes (dtype `F8E4M3`, on device) plus the
    /// per-tensor scales — exactly the `CublasLt::matmul_fp8` weight-side operands.
    PackedScalarFp8 {
        codes: Tensor,
        /// The `weight_scale` value (1.0 for the plain undescribed cast).
        weight_scale: f32,
        /// Whether the sibling `weight_scale` was **read from a `{layer}.weight_scale` companion row**
        /// (`true`) or is the plain undescribed cast's synthetic unit scale (`false`).
        ///
        /// This is a residency-accounting fact, not a decode fact — `matmul_fp8` takes the same
        /// `f32` either way. The plan prices a retained `weight_scale` on its
        /// [`gen_core::checkpoint_codec::CompanionTensorPlan`] row, and
        /// [`gen_core::checkpoint_codec::ScalarScaleSource::Unit`] has **no** such row (there is no
        /// tensor in the file to price). Counting four bytes unconditionally in
        /// [`Self::resident_bytes`] therefore made the receipt exceed the plan by exactly
        /// `4 × (undescribed fp8 layers)` — a plan/receipt inequality on the one pairing the two
        /// exist to cross-check.
        weight_scale_from_companion: bool,
        /// The retained `input_scale` companion value, when the checkpoint carries one.
        input_scale: Option<f32>,
        /// Keep activations at their compute dtype instead of quantizing them to FP8.
        full_precision_matrix_mult: bool,
    },
    /// The packed-native NVFP4 form (sc-20641): the checkpoint's E2M1 nibbles and both scale levels
    /// repacked into the canonical container [`candle_quant_kernels::Nvfp4Linear::from_packed_in`] consumes.
    ///
    /// Unlike [`Self::PackedScalarFp8`] this variant is **not** `cuda`-gated. `Nvfp4Tensor` is a host
    /// container (`Vec<u8>` nibbles + scales), so the repack is pure host code that is correct — and
    /// testable — on every lane; only the *decision* to plan `Packed` is hardware-gated, by
    /// [`CandleCodecResidency::nvfp4_native`]. Uploading it to the FP4 GEMM is `Nvfp4Linear`'s job
    /// and stays behind the `cuda` feature there.
    PackedNvfp4 {
        tensor: Box<Nvfp4Tensor>,
        /// The retained `input_scale` companion value, when the checkpoint carries one.
        input_scale: Option<f32>,
    },
}

impl LogicalTensor {
    /// Bytes this tensor keeps resident, measured from the actual representation — the reader's
    /// **only** definition of resident cost (see [`read_logical_weights`]).
    ///
    /// The packed variant counts its retained scales here because it owns them: they survive the
    /// decode as `f32` values on this struct, not as file rows. That is what makes the receipt an
    /// independent measurement of the plan's `Packed` pricing (stored bytes on the tensor row,
    /// `weight_scale`/`input_scale` on their companion rows) rather than a copy of it.
    pub fn resident_bytes(&self) -> u64 {
        /// One retained `f32` scale.
        const SCALE_BYTES: u64 = std::mem::size_of::<f32>() as u64;
        match self {
            Self::PackedMxFp8 { codes, scales, .. } => {
                (codes.elem_count() + scales.elem_count()) as u64
            }
            Self::PackedRowwiseFp8 { codes, scales }
            | Self::PackedConvRot { codes, scales, .. } => {
                (codes.elem_count() + scales.elem_count() * scales.dtype().size_in_bytes()) as u64
            }
            Self::Dense(tensor) => (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64,
            Self::PackedScalarFp8 {
                codes,
                weight_scale_from_companion,
                input_scale,
                ..
            } => {
                // Count a retained scale exactly when the plan priced a companion row for it: an
                // undescribed (`Unit`) fp8 layer has no `weight_scale` tensor in the file, so the
                // plan prices none and the receipt must not invent four bytes of it.
                (codes.elem_count() * codes.dtype().size_in_bytes()) as u64
                    + if *weight_scale_from_companion {
                        SCALE_BYTES
                    } else {
                        0
                    }
                    + if input_scale.is_some() {
                        SCALE_BYTES
                    } else {
                        0
                    }
            }
            // Both NVFP4 scale levels are owned here: the block scales as the container's padded
            // byte buffer, `weight_scale_2` as its `global_scale` f32. Measured off the container,
            // never copied from the plan.
            Self::PackedNvfp4 {
                tensor,
                input_scale,
            } => {
                (tensor.packed.len() + tensor.scales.len()) as u64
                    + SCALE_BYTES
                    + if input_scale.is_some() {
                        SCALE_BYTES
                    } else {
                        0
                    }
            }
        }
    }
}

/// The reader's output: logical-keyed tensors plus the receipt.
pub struct LogicalWeights {
    pub tensors: HashMap<String, LogicalTensor>,
    pub receipt: LogicalWeightReceipt,
}

#[cfg(feature = "cuda")]
fn matrix_slice_range(
    row: &gen_core::checkpoint_codec::LogicalTensorPlan,
    shape: &[usize],
    source_len: usize,
    first: usize,
    rows: usize,
    cols: usize,
) -> Result<std::ops::Range<usize>> {
    let key = &row.logical_key;
    let Some(len) = rows.checked_mul(cols).filter(|n| *n > 0) else {
        candle_core::bail!("{key}: invalid slice shape")
    };
    if shape != row.shape
        || shape.len() < 2
        || !(shape.last() == Some(&cols)
            || (first == 0 && shape.first() == Some(&rows) && source_len == len))
        || !source_len.is_multiple_of(len)
        || !first.is_multiple_of(rows)
    {
        candle_core::bail!(
            "{key}: matrix slice does not partition checkpoint shape {:?}",
            shape
        );
    }
    let start = first
        .checked_mul(cols)
        .ok_or_else(|| CandleError::Msg(format!("{key}: row offset overflow")))?;
    let end = start
        .checked_add(len)
        .filter(|&v| v <= source_len)
        .ok_or_else(|| CandleError::Msg(format!("{key}: slice is outside checkpoint")))?;
    Ok(start..end)
}

fn decode_scaled_fp8(bytes: &[u8], element: fn(u8) -> f32, scale: f32) -> Vec<f32> {
    let table: [f32; 256] = std::array::from_fn(|i| element(i as u8) * scale);
    bytes.iter().map(|&b| table[b as usize]).collect()
}

fn scalar_f32(bytes: &[u8], what: &str) -> Result<f32> {
    if bytes.len() != 4 {
        return Err(CandleError::Msg(format!(
            "{what}: expected one F32 scalar (4 bytes), got {} bytes",
            bytes.len()
        )));
    }
    Ok(cst::read_source_bytes(bytes, |bytes| {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }))
}

// Legacy companion-scaled checkpoints may store calibration scalars in either half type.
fn scalar_at(st: &MmapedSafetensors, key: &str) -> Result<f32> {
    let view = st.get(key)?;
    match view.dtype() {
        ::safetensors::Dtype::F32 => scalar_f32(view.data(), key),
        ::safetensors::Dtype::F16 | ::safetensors::Dtype::BF16 if view.data().len() == 2 => {
            let dtype = if view.dtype() == ::safetensors::Dtype::F16 {
                DType::F16
            } else {
                DType::BF16
            };
            cst::read_tensor_bytes(&view, |bytes| {
                Tensor::from_raw_buffer(bytes, dtype, &[], &Device::Cpu)
            })?
            .to_dtype(DType::F32)?
            .to_scalar()
        }
        other => candle_core::bail!("{key}: invalid checkpoint scalar type {other:?}"),
    }
}
fn companion_bytes<'a>(
    st: &'a MmapedSafetensors,
    tensor: &LogicalTensorPlan,
    key: &str,
) -> Result<&'a [u8]> {
    let view = st.get(key).map_err(|error| {
        CandleError::Msg(format!(
            "codec {}: tensor {:?} planned companion {key:?}: {error}",
            tensor.codec_id, tensor.physical_key
        ))
    })?;
    Ok(view.data())
}

/// Load exactly the planned tensors from `path` onto `device`, rename them to their logical keys,
/// and decode each through its planned codec. The file's tensor set must equal the plan's physical
/// key surface (weights **and** companions); any difference is source drift and refuses.
pub fn read_logical_weights(
    path: &Path,
    plan: &LogicalWeightPlan,
    device: &Device,
) -> Result<LogicalWeights> {
    let reader = LogicalWeightReader::open(path, plan.clone(), device)?;
    let mut tensors = HashMap::new();
    for tensor in &plan.tensors {
        tensors.insert(
            tensor.logical_key.clone(),
            reader.read(&tensor.logical_key)?,
        );
    }
    Ok(LogicalWeights {
        tensors,
        receipt: reader.receipt(),
    })
}

/// The engine's **incremental** shared logical-weight reader (sc-21482): the same plan-driven,
/// codec-dispatched decode as [`read_logical_weights`], exposed one logical tensor at a time so a
/// provider that constructs its model layer-by-layer (Krea's trunk, ~260 projections) can
/// materialize each planned tensor exactly when its layer is built — without a whole-file
/// intermediate map and without owning any decode or classification logic of its own.
///
/// * **Opening** validates the on-disk tensor set against the plan's full physical-key surface
///   (weights **and** companions); any difference is source drift and refuses before a single
///   tensor is materialized. Per-tensor drift after opening (shape/dtype changes under the mmap)
///   is still refused by the decode's stored-header guards.
/// * **Reading** dispatches the one shared per-layer decode — the same code path
///   [`read_logical_weights`] uses — and records what the codec actually left resident.
/// * **The receipt** ([`Self::receipt`]) is measured over the tensors materialized *so far*: each
///   codec row's `resident_bytes` comes off the decoded values themselves
///   ([`LogicalTensor::resident_bytes`]), never copied from the plan's pricing, so plan and
///   receipt remain the independent pair the contract requires. A companion's source bytes are
///   attributed to its owner's codec row only once that owner has materialized.
pub struct LogicalWeightReader {
    pub(crate) st: std::sync::Arc<MmapedSafetensors>,
    resources: candle_quant_kernels::convrot_int8::ProjectionResources,
    plan: std::sync::Arc<LogicalWeightPlan>,
    device: Device,
    /// Logical key → index into `plan.tensors`, built once at [`Self::open`].
    ///
    /// Without it [`Self::planned`] is a linear scan, and a whole-file read (or Krea's trunk,
    /// which resolves the same key three times per projection) turns the load into O(n²) over a
    /// ~2600-tensor plan (sc-21482 review).
    by_logical_key: BTreeMap<String, usize>,
    accounting: std::sync::Arc<gen_core::CheckpointReadAccounting>,
}
impl LogicalWeightReader {
    /// mmap `path` and verify its tensor surface equals the plan's before anything materializes.
    ///
    /// The host capability is probed from `device`. A loader that priced its plan under an
    /// explicitly constructed [`CandleCodecResidency`] must use
    /// [`Self::open_with_capability`] instead, so the capability the facts are validated against is
    /// the one the plan was priced under.
    pub fn open(path: &Path, plan: LogicalWeightPlan, device: &Device) -> Result<Self> {
        let capability = CandleCodecResidency::probe(device).native_execution_capability();
        Self::open_with_capability(path, plan, device, capability)
    }

    /// [`Self::open`] with the host capability supplied by the caller — the residency policy that
    /// priced this plan, rendered through
    /// [`CandleCodecResidency::native_execution_capability`].
    pub fn open_with_capability(
        path: &Path,
        plan: LogicalWeightPlan,
        device: &Device,
        capability: NativeExecutionCapability,
    ) -> Result<Self> {
        Self::open_with_capability_delegated(
            path,
            plan,
            device,
            capability,
            &std::collections::BTreeSet::new(),
        )
    }

    /// [`Self::open_with_capability`] for a plan compiled by [`plan_logical_weights_delegating_w4a8`]:
    /// `delegated` names the physical keys of the W4A8 projections served outside the plan. They
    /// must all still be on disk, and they are excluded from the plan-surface comparison — every
    /// other tensor is held to exactly the same equality as an undelegated open.
    pub fn open_with_capability_delegated(
        path: &Path,
        plan: LogicalWeightPlan,
        device: &Device,
        capability: NativeExecutionCapability,
        delegated: &std::collections::BTreeSet<String>,
    ) -> Result<Self> {
        // SAFETY: the provider-provisioned checkpoint must remain unmodified while mapped.
        let st = unsafe { MmapedSafetensors::new(path)? };
        let mut on_disk: Vec<String> = st.tensors().into_iter().map(|(name, _)| name).collect();
        on_disk.sort_unstable();
        let delegated_missing = delegated
            .iter()
            .filter(|key| on_disk.binary_search(key).is_err())
            .count();
        if delegated_missing > 0 {
            return Err(CandleError::Msg(format!(
                "logical weight read of {}: {delegated_missing} delegated W4A8 tensor(s) missing \
                 since planning; refusing to load a different checkpoint",
                path.display(),
            )));
        }
        on_disk.retain(|name| !delegated.contains(name));
        let mut planned: Vec<&str> = plan.all_physical_keys().collect();
        planned.sort_unstable();
        // A transformed tensor contributes one plan entry per logical output, all naming the same
        // physical key (sc-21547), so the plan's *surface* is the deduplicated set. Without this the
        // comparison below reads a fused-QKV split as three unplanned tensors and refuses every
        // checkpoint the transforms exist to load.
        planned.dedup();
        if on_disk
            .iter()
            .map(String::as_str)
            .ne(planned.iter().copied())
        {
            let missing = planned
                .iter()
                .filter(|key| {
                    on_disk
                        .binary_search_by(|name| name.as_str().cmp(key))
                        .is_err()
                })
                .count();
            let unplanned = on_disk
                .iter()
                .filter(|name| planned.binary_search(&name.as_str()).is_err())
                .count();
            return Err(CandleError::Msg(format!(
                "logical weight read of {}: tensor set changed since planning ({missing} planned \
                 tensor(s) missing, {unplanned} unplanned tensor(s) present); refusing to load a \
                 different checkpoint",
                path.display(),
            )));
        }
        Ok(Self::from_mapped(
            std::sync::Arc::new(st),
            plan,
            device,
            capability,
            &[path.to_owned()],
        ))
    }

    pub(crate) fn from_mapped(
        st: std::sync::Arc<MmapedSafetensors>,
        plan: LogicalWeightPlan,
        device: &Device,
        capability: NativeExecutionCapability,
        paths: &[std::path::PathBuf],
    ) -> Self {
        let by_logical_key = plan
            .tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.logical_key.clone(), index))
            .collect();
        let plan = std::sync::Arc::new(plan);
        let accounting = gen_core::CheckpointReadAccounting::new(plan.clone(), capability, paths);
        Self {
            st,
            plan,
            device: device.clone(),
            by_logical_key,
            resources: Default::default(),
            accounting,
        }
    }

    pub fn set_int8_context(&self, context: candle_quant_kernels::Int8Context) {
        self.resources.set_int8_context(context)
    }

    pub fn convrot_projection(
        &self,
        key: &str,
        bias: Option<Tensor>,
    ) -> Result<Option<candle_quant_kernels::convrot_int8::ConvRotLinear>> {
        if !self
            .planned(key)
            .is_some_and(|row| matches!(row.codec, TensorCodecSpec::ConvRotInt8 { .. }))
        {
            return Ok(None);
        }
        let operand = self.prepare_operand(key, None, &self.device)?;
        let bytes = operand.resident_bytes();
        let LogicalTensor::PackedConvRot {
            codes,
            scales,
            geometry,
        } = operand
        else {
            candle_core::bail!("{key}: expected ConvRot operands")
        };
        let op = candle_quant_kernels::convrot_int8::ConvRotLinear::from_native_parts(
            &self.resources,
            geometry,
            codes,
            scales,
            bias,
        )?;
        self.accounting
            .record(key, ExecutionRepresentation::NativePacked, bytes)
            .map_err(CandleError::wrap)?;
        Ok(Some(op))
    }

    /// Whether the native projection constructor owns this row of the compiled plan.
    pub fn has_native_projection(&self, key: &str) -> bool {
        self.planned(key).is_some_and(|row| {
            [
                FP8_E4M3_SCALAR_CODEC.codec_id,
                FP8_E5M2_SCALAR_CODEC.codec_id,
                FP8_E4M3_ROWWISE_CODEC.codec_id,
                MXFP8_CODEC.codec_id,
            ]
            .contains(&row.codec_id)
                && row.residency.mode == ResidencyMode::Packed
        })
    }

    /// Build an adapter-capable projection from the checkpoint codes when this row's plan
    /// selected native execution. Unhandled rows remain available to their existing owner.
    #[cfg(feature = "cuda")]
    pub fn fp8_projection(
        &self,
        key: &str,
        bias: Option<Tensor>,
    ) -> Result<Option<candle_quant_kernels::fp8_linear::Fp8Linear>> {
        self.fp8_projection_with_shape(key, bias, None)
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_projection_with_shape(
        &self,
        key: &str,
        bias: Option<Tensor>,
        shape: Option<(usize, usize)>,
    ) -> Result<Option<candle_quant_kernels::fp8_linear::Fp8Linear>> {
        self.fp8_projection_oriented(key, bias, shape, false)
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn fp8_projection_oriented(
        &self,
        key: &str,
        bias: Option<Tensor>,
        shape: Option<(usize, usize)>,
        transpose: bool,
    ) -> Result<Option<candle_quant_kernels::fp8_linear::Fp8Linear>> {
        if !self.has_native_projection(key) {
            return Ok(None);
        }
        if transpose
            && !self
                .planned(key)
                .is_some_and(|r| matches!(r.codec, TensorCodecSpec::ScalarFp8 { .. }))
        {
            return Ok(None);
        }
        let operand = self.prepare_operand(key, shape, &self.device)?;
        self.bind_fp8_operand(key, &operand, bias, shape, transpose)
            .map(Some)
    }

    /// Prepare the exact frozen operand once, either on the execution device or on the host
    /// for a bounded block window. Preparation alone does not report a model projection.
    pub(crate) fn prepare_operand(
        &self,
        key: &str,
        shape: Option<(usize, usize)>,
        device: &Device,
    ) -> Result<LogicalTensor> {
        let row = self.planned(key).ok_or_else(|| {
            CandleError::Msg(format!(
                "logical weight read: `{key}` has no entry in the compiled plan (mapping {})",
                self.plan.mapping_id
            ))
        })?;
        let row = Self::shaped_row(row, shape)?;
        decode(&self.st, &row, device, DType::BF16)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn bind_fp8_operand(
        &self,
        key: &str,
        operand: &LogicalTensor,
        bias: Option<Tensor>,
        shape: Option<(usize, usize)>,
        transpose: bool,
    ) -> Result<candle_quant_kernels::fp8_linear::Fp8Linear> {
        use candle_quant_kernels::fp8_linear::Fp8Linear;
        let ctx = self.resources.int8_context(&self.device)?;
        let lt = ctx.handle_for(&self.device)?.clone();
        let layer = match operand {
            LogicalTensor::PackedScalarFp8 {
                codes,
                weight_scale,
                input_scale,
                full_precision_matrix_mult,
                ..
            } => {
                let codes = if transpose {
                    codes.t()?.contiguous()?
                } else {
                    codes.clone()
                };
                if shape.is_some_and(|(n, k)| codes.dims() != [n, k]) {
                    candle_core::bail!(
                        "{key}: oriented weight {:?} differs from model shape {shape:?}",
                        codes.dims()
                    );
                }
                let codes = codes.to_device(&self.device)?;
                if *full_precision_matrix_mult {
                    Fp8Linear::from_weight_only_codes(codes, *weight_scale, bias, lt)?
                } else if codes.dtype() == DType::U8 {
                    Fp8Linear::from_e5m2_parts(codes, *weight_scale, *input_scale, bias, lt)?
                } else {
                    Fp8Linear::from_parts(codes, *weight_scale, *input_scale, bias, lt)?
                }
            }
            LogicalTensor::PackedRowwiseFp8 { codes, scales } => {
                if transpose || shape.is_some_and(|(n, k)| codes.dims() != [n, k]) {
                    candle_core::bail!("{key}: FP8R source shape/orientation mismatch");
                }
                Fp8Linear::from_rowwise_parts(
                    codes.to_device(&self.device)?,
                    &scales.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
                    bias,
                    lt,
                )?
            }
            LogicalTensor::PackedMxFp8 {
                codes,
                scales,
                logical,
            } => Fp8Linear::from_mxfp8_parts(
                codes.to_device(&self.device)?,
                scales.to_device(&self.device)?,
                *logical,
                bias,
                lt,
            )?,
            _ => candle_core::bail!("{key}: expected packed FP8 projection"),
        };
        self.accounting
            .record(
                key,
                ExecutionRepresentation::NativePacked,
                operand.resident_bytes(),
            )
            .map_err(CandleError::wrap)?;
        Ok(layer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dense_matrix_slice(
        &self,
        key: &str,
        first: usize,
        rows: usize,
        cols: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let row = self
            .planned(key)
            .ok_or_else(|| CandleError::Msg(format!("unknown source tensor {key}")))?;
        let TensorCodecSpec::ScalarFp8 { scale, .. } = &row.codec else {
            candle_core::bail!("{key}: scalar FP8 slice required")
        };
        let view = self.st.get(&row.physical_key)?;
        let range = matrix_slice_range(row, view.shape(), view.data().len(), first, rows, cols)?;
        let decode = match row.encoding {
            WeightEncoding::Fp8E4M3 => gen_core::fp8_e4m3fn_to_f32,
            WeightEncoding::Fp8E5M2 => gen_core::fp8_e5m2_to_f32,
            _ => candle_core::bail!("{key}: invalid scalar FP8 encoding"),
        };
        let sw = match scale {
            ScalarScaleSource::Unit => 1.,
            ScalarScaleSource::Companion { physical_key } => scalar_at(&self.st, physical_key)?,
        };
        cst::read_source_bytes(&view.data()[range], |bytes| {
            upload_values(
                decode_scaled_fp8(bytes, decode, sw),
                &[rows, cols],
                device,
                dtype,
            )
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn record_coefficient_cache(
        &self,
        key: &str,
        first: Option<usize>,
        t: &Tensor,
    ) -> Result<()> {
        let encoding = match t.dtype() {
            DType::BF16 => WeightEncoding::DenseBf16,
            DType::F16 => WeightEncoding::DenseF16,
            DType::F32 => WeightEncoding::DenseF32,
            DType::F64 => WeightEncoding::DenseF64,
            d => candle_core::bail!("{key}: unexpected decoded coefficient dtype {d:?}"),
        };
        self.accounting
            .record_coefficient_cache(
                key,
                first,
                t.dims(),
                encoding,
                (t.elem_count() * t.dtype().size_in_bytes()) as u64,
            )
            .map_err(CandleError::wrap)
    }

    /// Load one contiguous matrix from a scalar-coded stacked operand, without uploading the rest.
    #[cfg(feature = "cuda")]
    pub(crate) fn fp8_matrix_slice(
        &self,
        key: &str,
        first: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Option<(candle_quant_kernels::fp8_linear::Fp8Linear, u64)>> {
        let Some(row) = self.planned(key) else {
            return Ok(None);
        };
        let TensorCodecSpec::ScalarFp8 {
            scale,
            input_scale,
            full_precision_matrix_mult,
        } = &row.codec
        else {
            return Ok(None);
        };
        if row.transform.is_some() {
            return Ok(None);
        }
        let view = self.st.get(&row.physical_key)?;
        let range = matrix_slice_range(row, view.shape(), view.data().len(), first, rows, cols)?;
        if !self
            .native_execution_capability()
            .executes_natively(row.codec_id)
            || !candle_quant_kernels::cublaslt::eight_bit_layout_supported(&[rows, cols])
        {
            return Ok(None);
        }
        let expected = match row.encoding {
            WeightEncoding::Fp8E4M3 => (::safetensors::Dtype::F8_E4M3, DType::F8E4M3),
            WeightEncoding::Fp8E5M2 => (::safetensors::Dtype::F8_E5M2, DType::U8),
            _ => return Ok(None),
        };
        guard_stored(row, view.dtype(), view.shape(), expected.0, &row.shape)?;
        let (sw, scalar_bytes) = match scale {
            ScalarScaleSource::Unit => (1., 0),
            ScalarScaleSource::Companion { physical_key } => {
                (scalar_at(&self.st, physical_key)?, 4)
            }
        };
        let sx = input_scale
            .as_deref()
            .map(|key| scalar_at(&self.st, key))
            .transpose()?;
        let codes = cst::read_source_bytes(&view.data()[range], |bytes| {
            Tensor::from_raw_buffer(bytes, expected.1, &[rows, cols], &self.device)
        })?;
        let resident = codes.elem_count() as u64 + scalar_bytes + if sx.is_some() { 4 } else { 0 };
        let ctx = self.resources.int8_context(&self.device)?;
        let lt = ctx.handle_for(&self.device)?.clone();
        use candle_quant_kernels::fp8_linear::Fp8Linear;
        let op = if *full_precision_matrix_mult {
            Fp8Linear::from_weight_only_codes(codes, sw, None, lt)?
        } else if expected.1 == DType::U8 {
            Fp8Linear::from_e5m2_parts(codes, sw, sx, None, lt)?
        } else {
            Fp8Linear::from_parts(codes, sw, sx, None, lt)?
        };
        self.accounting
            .record_matrix_slice(key, first, rows, cols, resident)
            .map_err(CandleError::wrap)?;
        Ok(Some((op, resident)))
    }

    /// Record that the provider served `logical_key` — a row the plan priced
    /// [`ResidencyMode::Packed`] — as a **dense BF16** weight holding `resident_bytes` on-device
    /// (sc-11045 fix round, epic E5).
    ///
    /// This is the seam that keeps the receipt truthful about W4A16 and the transparent
    /// construction-time fallbacks: the plan prices from checkpoint geometry and device floors
    /// alone, but a provider's role table (or a missing fused quantizer, a staging failure) can
    /// still hold a `Packed`-priced row as the dequantized dense weight. Without this call the
    /// receipt would report that row `NativePacked` at packed byte counts — dense BF16 execution
    /// labelled native, which is exactly the dishonest report E5 forbids.
    ///
    /// `resident_bytes` is measured off the constructed layer
    /// (`Nvfp4Linear::resident_weight_bytes`), never derived here. Refuses a key the plan does not
    /// contain and a row the plan priced `Dense` (nothing packed exists to demote). Re-demoting the
    /// same key with the same measurement is idempotent (a staged provider can rebuild a layer);
    /// a *different* measurement for the same key is a contradiction and refuses.
    pub fn demote_to_dense_fallback(&self, logical_key: &str, resident_bytes: u64) -> Result<()> {
        let tensor = self.planned(logical_key).ok_or_else(|| {
            CandleError::Msg(format!(
                "logical weight demotion: `{logical_key}` has no entry in the compiled plan \
                 (mapping {})",
                self.plan.mapping_id
            ))
        })?;
        if tensor.residency.mode != ResidencyMode::Packed {
            return Err(CandleError::Msg(format!(
                "logical weight demotion: `{logical_key}` was priced Dense by the plan; there is \
                 no packed pricing to demote from"
            )));
        }
        self.accounting
            .record(
                logical_key,
                ExecutionRepresentation::DenseFallback,
                resident_bytes,
            )
            .map_err(CandleError::wrap)
    }

    /// This host's native-execution declaration — the capability the plan was priced under.
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn materialized_bytes(&self, key: &str) -> Option<u64> {
        self.accounting.materialized_bytes(key)
    }
    pub fn native_execution_capability(&self) -> &NativeExecutionCapability {
        self.accounting.capability()
    }

    /// The plan this reader materializes.
    pub fn plan(&self) -> &LogicalWeightPlan {
        &self.plan
    }

    /// The planned entry for one **logical** key, or `None` when the plan has no such tensor.
    ///
    /// O(log n) through the index built at [`Self::open`] — the hot path of a whole-plan read.
    pub fn planned(&self, logical_key: &str) -> Option<&LogicalTensorPlan> {
        self.by_logical_key
            .get(logical_key)
            .map(|index| &self.plan.tensors[*index])
    }

    /// ConvRot applies activation-dtype rounding to the bias at execution. Keep the original
    /// floating operand so the same projection can receive inputs at different precisions.
    pub fn projection_bias_dtype(&self, weight_key: &str) -> Option<DType> {
        self.projection_bias_storage(weight_key)
            .map(|(_, dtype)| dtype)
    }

    fn projection_bias_storage(&self, weight_key: &str) -> Option<(String, DType)> {
        use ::safetensors::Dtype as StDtype;
        let key = self.planned(weight_key)?.retained_convrot_bias_key()?;
        let dtype = match self.st.get(&key).ok()?.dtype() {
            StDtype::F32 | StDtype::F8_E4M3 | StDtype::F8_E5M2 => DType::F32,
            StDtype::F16 => DType::F16,
            StDtype::BF16 => DType::BF16,
            _ => return None,
        };
        Some((key, dtype))
    }

    /// Read a bias whose native epilogue retains source precision. A mapped or split bias
    /// follows its own logical row. A sparse plan keeps ordinary biases in the original mmap.
    pub fn retained_projection_bias(
        &self,
        weight_key: &str,
        bias_key: &str,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let Some((physical, dtype)) = self.projection_bias_storage(weight_key) else {
            return Ok(None);
        };
        let bias = if self.planned(bias_key).is_some() {
            self.read_dense(bias_key, dtype, device)?
        } else {
            self.st.load(&physical, device)?.to_dtype(dtype)?
        };
        Ok(Some(bias))
    }

    /// Materialize one planned logical tensor through its codec and record what it keeps resident.
    pub fn read(&self, logical_key: &str) -> Result<LogicalTensor> {
        let decoded = self.prepare_operand(logical_key, None, &self.device)?;
        let tensor = self.planned(logical_key).expect("prepared logical row");
        let repr = if tensor.residency.mode == ResidencyMode::Packed {
            ExecutionRepresentation::NativePacked
        } else {
            ExecutionRepresentation::DenseFallback
        };
        self.accounting
            .record(logical_key, repr, decoded.resident_bytes())
            .map_err(CandleError::wrap)?;
        Ok(decoded)
    }

    fn shaped_row(
        tensor: &LogicalTensorPlan,
        shape: Option<(usize, usize)>,
    ) -> Result<std::borrow::Cow<'_, LogicalTensorPlan>> {
        Ok(
            if let (
                Some((n, k)),
                TensorCodecSpec::Mxfp8 {
                    stored_shape,
                    logical_shape_declared: false,
                    ..
                },
            ) = (shape, &tensor.codec)
            {
                if gen_core::mxfp8_padded_shape([n, k]) != *stored_shape {
                    candle_core::bail!(
                        "{}: MXFP8 stored {stored_shape:?} does not pad model shape [{n},{k}]",
                        tensor.logical_key
                    );
                }
                let mut row = tensor.clone();
                let TensorCodecSpec::Mxfp8 {
                    logical_shape_declared,
                    ..
                } = &mut row.codec
                else {
                    unreachable!()
                };
                *logical_shape_declared = true;
                row.shape = vec![n, k];
                std::borrow::Cow::Owned(row)
            } else {
                std::borrow::Cow::Borrowed(tensor)
            },
        )
    }

    /// Bytes planned for the packed weight itself; companion allocations are separate plan rows.
    pub fn planned_packed_bytes(&self, key: &str) -> Option<u64> {
        let row = self.planned(key)?;
        (row.residency.mode == ResidencyMode::Packed).then_some(row.residency.resident_bytes)
    }

    /// Validate model-supplied matrix dimensions without decoding or accounting an allocation.
    pub fn validate_matrix_shape(&self, key: &str, expected: [usize; 2]) -> Result<()> {
        let row = self
            .planned(key)
            .ok_or_else(|| CandleError::Msg(format!("unplanned weight {key}")))?;
        let actual = Self::shaped_row(row, Some((expected[0], expected[1])))?;
        if actual.shape != expected {
            candle_core::bail!(
                "{key}: model expects {expected:?}, checkpoint declares {:?}",
                actual.shape
            );
        }
        Ok(())
    }

    /// Decode the requested coefficient view without changing an existing projection's execution regime.
    pub(crate) fn dense_coefficients(
        &self,
        key: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let row = self
            .planned(key)
            .ok_or_else(|| CandleError::Msg(format!("unplanned weight {key}")))?;
        let mut dense = Self::shaped_row(row, shape)?.into_owned();
        dense.residency.mode = ResidencyMode::Dense;
        let LogicalTensor::Dense(t) = decode(&self.st, &dense, device, dtype)? else {
            candle_core::bail!("{key}: checkpoint operand has no dense coefficient representation")
        };
        t.to_dtype(dtype)
    }

    /// Read a true dense operand for a caller that needs coefficients (embedding, convolution,
    /// weight fusion), accounting for the actual storage when its plan priced a GEMM.
    pub fn read_dense(&self, key: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        self.read_dense_with_shape(key, None, dtype, device)
    }
    pub fn read_dense_with_shape(
        &self,
        key: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let row = self
            .planned(key)
            .ok_or_else(|| CandleError::Msg(format!("unplanned weight {key}")))?;
        let shaped = Self::shaped_row(row, shape)?;
        let t = self.dense_coefficients(key, shape, dtype, device)?;
        self.accounting
            .declare_dense_layout(key, &shaped.shape, dtype.size_in_bytes())
            .map_err(CandleError::wrap)?;
        let bytes = (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        self.accounting
            .record(key, ExecutionRepresentation::DenseFallback, bytes)
            .map_err(CandleError::wrap)?;
        Ok(t)
    }

    /// The receipt **measured** over everything materialized through [`Self::read`] so far.
    ///
    /// A reader none of whose tensors has materialized yet reports zero rows — the truth about
    /// this instant, not a claim that the load was free. Once every planned tensor has been read,
    /// the receipt covers the plan's whole surface: its per-codec `resident_bytes` are directly
    /// comparable to the plan's pricing and its `source_bytes` total the plan's
    /// (weights + companions + descriptor payloads all attributed).
    ///
    /// # Source bytes count each *physical* tensor once
    ///
    /// A transformed tensor contributes one plan entry per logical output, all naming the same
    /// physical key (sc-21547). Their source bytes must still total the file **once**, and the
    /// rule that makes that true lives in the plan compiler, not here: exactly one output per
    /// physical tensor carries `source_bytes` and its siblings carry zero
    /// ([`LogicalTensorPlan::transform`]). So this loop simply *sums* — summing a plain sum is
    /// what makes it correct, and de-duplicating by physical key here would be actively wrong,
    /// because `plan.tensors` is sorted by logical key and the byte-carrying output is not
    /// generally the first one a fused key is seen at.
    ///
    /// `gen_core::checkpoint_facts::SourceCodecSummary` sums under the same convention, and
    /// [`CheckpointWeightFacts::new`] cross-checks the two totals per codec. `tensor_count` and
    /// `resident_bytes` stay **per logical row** — each output is separately counted and
    /// separately occupies memory; only the source bytes are shared.
    pub fn receipt(&self) -> LogicalWeightReceipt {
        self.accounting.receipt()
    }

    pub fn checkpoint_weight_facts(&self) -> Result<CheckpointWeightFacts> {
        self.accounting.facts().map_err(CandleError::wrap)
    }
}

fn guard_stored(
    tensor: &LogicalTensorPlan,
    view_dtype: ::safetensors::Dtype,
    view_shape: &[usize],
    expected_dtype: ::safetensors::Dtype,
    expected_shape: &[usize],
) -> Result<()> {
    if view_shape != expected_shape {
        return Err(CandleError::Msg(format!(
            "codec {}: tensor {:?} was planned with stored shape {expected_shape:?} but the file \
             now holds shape {view_shape:?}; the file changed between planning and reading",
            tensor.codec_id, tensor.physical_key
        )));
    }
    if view_dtype != expected_dtype {
        return Err(CandleError::Msg(format!(
            "codec {}: tensor {:?} was planned as {} but the file now holds {view_dtype:?}",
            tensor.codec_id,
            tensor.physical_key,
            tensor.encoding.label()
        )));
    }
    Ok(())
}

fn upload_values(
    values: Vec<f32>,
    shape: &[usize],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    Tensor::from_vec(values, shape, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
}

/// Run the planned codec on one stored tensor — the per-layer dispatch.
fn decode(
    st: &MmapedSafetensors,
    tensor: &LogicalTensorPlan,
    device: &Device,
    dtype: DType,
) -> Result<LogicalTensor> {
    use ::safetensors::Dtype as StDtype;

    // A block-padded row (MXFP8/NVFP4) whose adapter declared no logical shape would decode its
    // PADDING as weights — see [`LogicalTensorPlan::undeclared_padded_storage`]. Planning such a
    // file stays legal (the padded grid is a conservative pricing over-estimate, and the plans
    // compiled for admission never materialize a tensor); materializing it is not. The refusal text
    // is gen-core's, so both engines refuse the same checkpoint with the same diagnosis.
    if let Some(refusal) = tensor.undeclared_padded_storage_refusal() {
        return Err(CandleError::Msg(refusal));
    }

    // Rank floor for the arms below that index the logical shape positionally — see
    // [`LogicalTensorPlan::matrix_rank_refusal`]. Checked before `guard_stored` so a wrong-rank plan
    // is diagnosed as the rank defect it is rather than as a stored-shape drift. The refusal text is
    // gen-core's, so both engines refuse the same plan with the same diagnosis.
    if let Some(refusal) = tensor.matrix_rank_refusal() {
        return Err(CandleError::Msg(refusal));
    }

    let view = st.get(&tensor.physical_key)?;
    // The codec decodes the *physical* tensor, so every geometry below is the pre-transform shape;
    // `tensor.shape` is the post-transform shape the model sees and is applied by
    // [`apply_transform`] at the end. For an untransformed entry the two are the same slice.
    let source_shape = tensor.source_shape();
    let decoded = match &tensor.codec {
        TensorCodecSpec::Dense => {
            let expected = match tensor.codec_id {
                id if id == DENSE_BF16_CODEC.codec_id => StDtype::BF16,
                id if id == DENSE_F16_CODEC.codec_id => StDtype::F16,
                id if id == DENSE_F32_CODEC.codec_id => StDtype::F32,
                other => {
                    return Err(CandleError::Msg(format!(
                        "codec {other:?} is registered but this engine has no dense implementation \
                         for it (tensor {:?})",
                        tensor.physical_key
                    )))
                }
            };
            guard_stored(tensor, view.dtype(), view.shape(), expected, source_shape)?;
            // Byte-preserving: no cast, no substitution can hide inside the dense rows.
            LogicalTensor::Dense(st.load(&tensor.physical_key, device)?)
        }
        TensorCodecSpec::ScalarFp8 {
            scale,
            input_scale,
            full_precision_matrix_mult,
        } => {
            #[cfg(not(feature = "cuda"))]
            let _ = full_precision_matrix_mult;
            let (expected_dtype, element) = match tensor.encoding {
                WeightEncoding::Fp8E4M3 => (
                    StDtype::F8_E4M3,
                    gen_core::fp8_e4m3fn_to_f32 as fn(u8) -> f32,
                ),
                WeightEncoding::Fp8E5M2 => {
                    (StDtype::F8_E5M2, gen_core::fp8_e5m2_to_f32 as fn(u8) -> f32)
                }
                other => {
                    return Err(CandleError::Msg(format!(
                        "codec {}: tensor {:?} planned scalar-fp8 decode for non-fp8 encoding {}",
                        tensor.codec_id,
                        tensor.physical_key,
                        other.label()
                    )))
                }
            };
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                expected_dtype,
                source_shape,
            )?;
            let (weight_scale, weight_scale_from_companion) = match scale {
                ScalarScaleSource::Unit => (1.0_f32, false),
                ScalarScaleSource::Companion { physical_key } => {
                    (scalar_at(st, physical_key)?, true)
                }
            };
            match tensor.residency.mode {
                ResidencyMode::Dense => {
                    let values: Vec<f32> = cst::read_tensor_bytes(&view, |bytes| {
                        decode_scaled_fp8(bytes, element, weight_scale)
                    });
                    LogicalTensor::Dense(upload_values(values, source_shape, device, dtype)?)
                }
                ResidencyMode::Packed => {
                    #[cfg(feature = "cuda")]
                    {
                        let input_scale = input_scale
                            .as_deref()
                            .map(|key| scalar_at(st, key))
                            .transpose()?;
                        let codes = if tensor.encoding == WeightEncoding::Fp8E5M2 {
                            Tensor::from_raw_buffer(view.data(), DType::U8, source_shape, device)?
                        } else {
                            st.load(&tensor.physical_key, device)?
                        };
                        LogicalTensor::PackedScalarFp8 {
                            codes,
                            weight_scale,
                            weight_scale_from_companion,
                            input_scale,
                            full_precision_matrix_mult: *full_precision_matrix_mult,
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = (input_scale, weight_scale_from_companion);
                        return Err(CandleError::Msg(format!(
                            "codec {}: tensor {:?} was planned with packed-native fp8 residency, \
                             but this build has no CUDA fp8 leg; replan with a dense residency \
                             policy instead of silently substituting the dense fallback",
                            tensor.codec_id, tensor.physical_key
                        )));
                    }
                }
            }
        }
        TensorCodecSpec::RowwiseFp8 { scale } => {
            let n = source_shape[0];
            let k = source_shape[1];
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                StDtype::F8_E4M3,
                source_shape,
            )?;
            let sv = st.get(scale)?;
            if sv.dtype() != StDtype::F32 || sv.shape() != [n] {
                candle_core::bail!("{scale}: FP8R source scale geometry changed");
            }
            let scales = cst::read_tensor_bytes(&sv, |b| {
                b.chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>()
            });
            if tensor.residency.mode == ResidencyMode::Packed {
                LogicalTensor::PackedRowwiseFp8 {
                    codes: cst::read_tensor_bytes(&view, |b| {
                        Tensor::from_raw_buffer(b, DType::F8E4M3, source_shape, device)
                    })?,
                    scales: Tensor::from_vec(scales, n, &Device::Cpu)?,
                }
            } else {
                let values = cst::read_tensor_bytes(&view, |b| {
                    b.chunks_exact(k)
                        .zip(&scales)
                        .flat_map(|(codes, &scale)| {
                            decode_scaled_fp8(codes, gen_core::fp8_e4m3fn_to_f32, scale)
                        })
                        .collect()
                });
                LogicalTensor::Dense(upload_values(values, source_shape, device, dtype)?)
            }
        }
        TensorCodecSpec::Mxfp8 {
            scale,
            stored_shape,
            ..
        } => {
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                StDtype::F8_E4M3,
                stored_shape,
            )?;
            let scales = companion_bytes(st, tensor, scale)?;
            if tensor.residency.mode == ResidencyMode::Packed {
                let scale_shape = gen_core::mxfp8_scale_shape(*stored_shape);
                let scale_tensor =
                    Tensor::from_raw_buffer(scales, DType::U8, &scale_shape, device)?;
                let codes = st.load(&tensor.physical_key, device)?;
                let logical = (source_shape[0], source_shape[1]);
                return apply_transform(
                    tensor,
                    LogicalTensor::PackedMxFp8 {
                        codes,
                        scales: scale_tensor,
                        logical,
                    },
                );
            }
            let mut values = Vec::new();
            cst::read_source_byte_sets(&[view.data(), scales], || {
                gen_core::decode_mxfp8(
                    view.data(),
                    scales,
                    *stored_shape,
                    [source_shape[0], source_shape[1]],
                    &mut values,
                )
            })
            .map_err(|error| {
                CandleError::Msg(format!(
                    "codec {}: tensor {:?}: {error}",
                    tensor.codec_id, tensor.physical_key
                ))
            })?;
            LogicalTensor::Dense(upload_values(values, source_shape, device, dtype)?)
        }
        TensorCodecSpec::Nvfp4 {
            block_scale,
            global_scale,
            input_scale,
            stored_shape,
            ..
        } => {
            // The on-disk byte matrix is `[rows, cols / 2]` — two E2M1 codes per `U8`.
            let packed_shape = [stored_shape[0], stored_shape[1] / 2];
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                StDtype::U8,
                &packed_shape,
            )?;
            let scales = companion_bytes(st, tensor, block_scale)?;
            let global = scalar_f32(companion_bytes(st, tensor, global_scale)?, global_scale)?;
            match tensor.residency.mode {
                ResidencyMode::Dense => {
                    let mut values = Vec::new();
                    cst::read_source_byte_sets(&[view.data(), scales], || {
                        gen_core::decode_nvfp4(
                            view.data(),
                            scales,
                            global,
                            *stored_shape,
                            [source_shape[0], source_shape[1]],
                            &mut values,
                        )
                    })
                    .map_err(|error| {
                        CandleError::Msg(format!(
                            "codec {}: tensor {:?}: {error}",
                            tensor.codec_id, tensor.physical_key
                        ))
                    })?;
                    LogicalTensor::Dense(upload_values(values, source_shape, device, dtype)?)
                }
                ResidencyMode::Packed => {
                    // The repack into the container `Nvfp4Linear` consumes: nibble-order swap plus
                    // the block-scale atom-order permutation (see `Nvfp4Tensor::from_kitchen_parts`).
                    // The container holds exactly the stored grid and carries no unpad, so a packed
                    // decode is only sound when the layer *is* that grid. `CandleCodecResidency`
                    // plans Dense otherwise; refuse here rather than trust it, because silently
                    // repacking a padded layer widens it and feeds the GEMM padding as real
                    // contraction elements (sc-20641).
                    if source_shape != stored_shape.as_slice() {
                        return Err(CandleError::Msg(format!(
                            "codec {}: tensor {:?} planned packed-native NVFP4 residency but its \
                             logical shape {source_shape:?} is not its stored shape \
                             {stored_shape:?}; ComfyUI padded this layer and the packed container \
                             cannot express the unpad",
                            tensor.codec_id, tensor.physical_key
                        )));
                    }
                    // Codec-owned scale-payload value check (sc-21482): the dense arm gets this
                    // inside `decode_nvfp4`; the packed repack preserves scale bytes verbatim, so
                    // a NaN or sign-bit E4M3 block scale must be refused here or it reaches the
                    // GEMM as a corrupted multiplier no later stage re-examines.
                    cst::read_source_byte_sets(&[scales], || {
                        gen_core::validate_nvfp4_block_scale_payload(
                            scales,
                            *stored_shape,
                            [source_shape[0], source_shape[1]],
                        )
                    })
                    .map_err(|error| {
                        CandleError::Msg(format!(
                            "codec {}: tensor {:?}: {error}",
                            tensor.codec_id, tensor.physical_key
                        ))
                    })?;
                    let tensor_packed = cst::read_source_byte_sets(&[view.data(), scales], || {
                        Nvfp4Tensor::from_kitchen_parts(
                            view.data(),
                            scales,
                            global,
                            stored_shape[0],
                            stored_shape[1],
                        )
                    })
                    .map_err(|error| {
                        CandleError::Msg(format!(
                            "codec {}: tensor {:?} planned packed-native NVFP4 residency but its \
                             stored layout does not repack: {error}",
                            tensor.codec_id, tensor.physical_key
                        ))
                    })?;
                    let input_scale = input_scale
                        .as_deref()
                        .map(|key| scalar_at(st, key))
                        .transpose()?;
                    LogicalTensor::PackedNvfp4 {
                        tensor: Box::new(tensor_packed),
                        input_scale,
                    }
                }
            }
        }
        TensorCodecSpec::ConvRotInt8 { scale, geometry } => {
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                StDtype::I8,
                source_shape,
            )?;
            let companion = st.get(scale)?;
            let shape = [geometry.out_features, geometry.in_features];
            if tensor.residency.mode != ResidencyMode::Packed
                || shape != source_shape
                || companion.dtype() != StDtype::F32
                || match (geometry.flat_scales, companion.shape()) {
                    (true, [n]) | (false, [n, 1]) => *n != geometry.out_features,
                    _ => true,
                }
            {
                return Err(CandleError::Msg(format!(
                    "{}: expected native ConvRot I8 matrix and F32 row scales",
                    tensor.physical_key
                )));
            }
            let (start, n) = match &tensor.transform {
                None => (0, geometry.out_features),
                Some(t) => (t.rows.start, t.rows.len),
            };
            let k = geometry.in_features;
            if start > geometry.out_features
                || n > geometry.out_features - start
                || tensor.shape.as_slice() != [n, k]
            {
                return Err(CandleError::Msg(format!(
                    "{}: ConvRot row slice is outside stored weights",
                    tensor.logical_key
                )));
            }
            let slice = gen_core::convrot_int8::ConvRotInt8 {
                out_features: n,
                ..*geometry
            };
            let scale_bytes = &companion.data()[start * 4..(start + n) * 4];
            cst::read_source_bytes(scale_bytes, |bytes| {
                slice.validate_scale_values(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                )
            })
            .map_err(CandleError::Msg)?;
            let codes = Tensor::from_raw_buffer(
                &view.data()[start * k..(start + n) * k],
                DType::U8,
                &[n, k],
                device,
            )?;
            let scales = Tensor::from_raw_buffer(scale_bytes, DType::F32, &[n, 1], device)?;
            let (codes, scales) = if tensor.transform.as_ref().is_some_and(|t| t.half_swap) {
                if !n.is_multiple_of(2) {
                    return Err(CandleError::Msg(format!(
                        "{}: odd ConvRot half swap",
                        tensor.logical_key
                    )));
                }
                let half_swap = |x: Tensor| {
                    candle_core::Tensor::cat(
                        &[x.narrow(0, n / 2, n / 2)?, x.narrow(0, 0, n / 2)?],
                        0,
                    )
                };
                (half_swap(codes)?, half_swap(scales)?)
            } else {
                (codes, scales)
            };
            return Ok(LogicalTensor::PackedConvRot {
                codes,
                scales,
                geometry: slice,
            });
        }
        TensorCodecSpec::Int8PerRow { scale, .. } => {
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                StDtype::I8,
                source_shape,
            )?;
            let scale_bytes = companion_bytes(st, tensor, scale)?;
            let rows = source_shape[0];
            let cols = source_shape[1];
            let scales: Vec<f32> = if scale_bytes.len() == 4 && rows == 1 {
                vec![scalar_f32(scale_bytes, scale)?]
            } else {
                if scale_bytes.len() != rows * 4 {
                    return Err(CandleError::Msg(format!(
                        "codec {}: companion {scale:?} holds {} bytes, expected {} ({rows} F32 \
                         row scales)",
                        tensor.codec_id,
                        scale_bytes.len(),
                        rows * 4
                    )));
                }
                cst::read_source_bytes(scale_bytes, |bytes| {
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                })
            };
            let codes: &[u8] = view.data();
            let mut values = Vec::new();
            gen_core::decode_int8_per_row(
                // SAFETY-free reinterpretation: i8 and u8 share layout; map explicitly instead.
                &cst::read_source_bytes(codes, |bytes| {
                    bytes.iter().map(|&byte| byte as i8).collect::<Vec<i8>>()
                }),
                &scales,
                cols,
                &mut values,
            );
            LogicalTensor::Dense(upload_values(values, source_shape, device, dtype)?)
        }
    };
    apply_transform(tensor, decoded)
}

/// Apply the plan entry's adapter-declared logical transform to the codec's whole-tensor output
/// (sc-21547).
///
/// The transform is a **row re-labelling**, never arithmetic: a contiguous slice of the leading
/// axis and, for dense rows, an exchange of that slice's two halves. It runs after the codec so no
/// codec knowledge lives in the provider that declared it (epic requirement E1), and it runs on the
/// codec's *own* representation so a packed source is never dequantized to be split (E3): a packed
/// fp8 slice narrows the stored code matrix, and a packed NVFP4 slice goes through
/// [`Nvfp4Tensor::slice_rows`], which re-emits block scales without touching a nibble.
///
/// Every slice is made [`Tensor::contiguous`] (and the NVFP4 container copies), so the output owns
/// its bytes and [`LogicalTensor::resident_bytes`] measures the slice rather than the whole source
/// buffer it was cut from — the measurement the receipt is built on.
#[doc(hidden)]
pub fn apply_transform(
    tensor: &LogicalTensorPlan,
    decoded: LogicalTensor,
) -> Result<LogicalTensor> {
    let Some(transform) = tensor.transform.as_ref() else {
        return Ok(decoded);
    };
    let (start, len) = (transform.rows.start, transform.rows.len);
    match decoded {
        LogicalTensor::PackedMxFp8 {
            codes,
            scales,
            logical,
        } => {
            if transform.half_swap || !start.is_multiple_of(128) || !len.is_multiple_of(128) {
                return Err(CandleError::Msg(format!(
                    "{}: MXFP8 transform must select complete 128-row scale tiles",
                    tensor.logical_key
                )));
            }
            Ok(LogicalTensor::PackedMxFp8 {
                codes: codes.narrow(0, start, len)?.contiguous()?,
                scales: scales.narrow(0, start, len)?.contiguous()?,
                logical: (len, logical.1),
            })
        }
        LogicalTensor::PackedConvRot { .. } => Err(CandleError::Msg(
            "ConvRot slices read directly from source".into(),
        )),
        LogicalTensor::Dense(dense) => {
            let sliced = dense.narrow(0, start, len)?;
            let out = if transform.half_swap {
                let half = len / 2;
                Tensor::cat(
                    &[sliced.narrow(0, half, half)?, sliced.narrow(0, 0, half)?],
                    0,
                )?
            } else {
                sliced.contiguous()?
            };
            Ok(LogicalTensor::Dense(out))
        }
        LogicalTensor::PackedRowwiseFp8 { codes, scales } => {
            let apply = |t: Tensor| -> Result<Tensor> {
                let sliced = t.narrow(0, start, len)?;
                if transform.half_swap {
                    let half = len / 2;
                    Tensor::cat(
                        &[sliced.narrow(0, half, half)?, sliced.narrow(0, 0, half)?],
                        0,
                    )
                } else {
                    sliced.force_contiguous()
                }
            };
            Ok(LogicalTensor::PackedRowwiseFp8 {
                codes: apply(codes)?,
                scales: apply(scales)?,
            })
        }
        LogicalTensor::PackedScalarFp8 {
            codes,
            weight_scale,
            weight_scale_from_companion,
            input_scale,
            full_precision_matrix_mult,
        } => {
            let sliced = codes.narrow(0, start, len)?;
            let codes = if transform.half_swap {
                if !len.is_multiple_of(2) {
                    candle_core::bail!(
                        "{}: cannot half-swap odd row count {len}",
                        tensor.logical_key
                    );
                }
                let half = len / 2;
                Tensor::cat(
                    &[sliced.narrow(0, half, half)?, sliced.narrow(0, 0, half)?],
                    0,
                )?
            } else {
                sliced.contiguous()?
            };
            Ok(LogicalTensor::PackedScalarFp8 {
                codes,
                weight_scale,
                weight_scale_from_companion,
                input_scale,
                full_precision_matrix_mult,
            })
        }
        LogicalTensor::PackedNvfp4 {
            tensor: packed,
            input_scale,
        } => {
            // Same reasoning as the packed-fp8 arm above: a real refusal, not a `debug_assert!`.
            if transform.half_swap {
                return Err(half_swap_on_packed(tensor));
            }
            let sliced = packed.slice_rows(start, len).map_err(|error| {
                CandleError::Msg(format!(
                    "codec {}: tensor {:?} logical output {:?}: {error}",
                    tensor.codec_id, tensor.physical_key, tensor.logical_key
                ))
            })?;
            Ok(LogicalTensor::PackedNvfp4 {
                tensor: Box::new(sliced),
                input_scale,
            })
        }
    }
}

/// The read-time refusal for a `half_swap` declared on a tensor this backend decoded **packed**.
///
/// The plan compiler already refuses this combination by name
/// (`LogicalTransformError::HalfSwapOnPackedResidency`); this is the
/// local restatement at the site that depends on it, because
/// [`LogicalTensorPlan::transform`] is a public field on a public struct reached through the public
/// [`read_logical_weights`] entry. Permuting packed rows would require re-deriving the codec's
/// scale surface, so the packed arms cannot honour the swap — and quietly returning the unswapped
/// rows under the swapped logical key is silent corruption, so they refuse instead. The wording
/// mirrors gen-core's, plus the physical and logical keys the reader knows and the compiler's
/// error does not carry.
fn half_swap_on_packed(tensor: &LogicalTensorPlan) -> CandleError {
    CandleError::Msg(format!(
        "codec {}: tensor {:?} logical output {:?} declares a half swap but this backend decoded a \
         packed-native residency for it; permuting packed rows would require re-deriving the \
         codec's scale surface. Plan this file with a dense residency policy, or declare the swap \
         on a dense layer",
        tensor.codec_id, tensor.physical_key, tensor.logical_key
    ))
}
