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
//! # Dense fallback vs native path
//!
//! * **Dense fallback** (every codec row, every device): the stored bytes are decoded on the host
//!   through the `gen_core::comfy_quant` reference functions — E4M3FN/E5M2 element decode ×
//!   per-tensor scale, MXFP8 block dequantization (cuBLAS 128×4 scale un-swizzle + 32-block
//!   shared exponents + unpadding), int8 × per-row scale — in exact f32, then cast once to the
//!   codec's resident bf16 and uploaded. Host decode is deliberate: Candle's Metal backend has no
//!   fp8 cast kernels and its CPU/CUDA `to_dtype` covers E4M3 only, while the reference functions
//!   cover every row identically on every lane.
//! * **Native paths** — two codec rows have one, and each is gated on its own layout + hardware
//!   contract by [`CandleCodecResidency`]:
//!   * `fp8-e4m3-scalar-v1`: a CUDA device at the sm_89 floor (`CublasLt::meets_fp8_floor`, via
//!     [`crate::quant::FP8_COMPUTE_CAP_FLOOR`]; locked decision 7), a rank-2 E4M3 weight, and no
//!     `full_precision_matrix_mult` flag. The reader keeps the stored `F8E4M3` codes + f32 scale
//!     resident ([`LogicalTensor::PackedFp8E4M3`]), the exact operands `CublasLt::matmul_fp8`
//!     consumes. On a build without the `cuda` feature a `Packed` fp8 plan entry is a typed
//!     refusal — never a silent dense substitution of what admission priced as packed.
//!   * `nvfp4-v1` (sc-20641): a CUDA device at the sm_120 floor (via
//!     [`crate::quant::NVFP4_COMPUTE_CAP_FLOOR`]), a K/N-aligned stored layout
//!     ([`nvfp4_layout_is_native`]) **and** a stored grid that is the layer itself (no ComfyUI
//!     padding — the packed container carries no unpad). The reader repacks the checkpoint's nibbles and both scale
//!     levels into the canonical [`Nvfp4Tensor`] container
//!     ([`LogicalTensor::PackedNvfp4`]) that `Nvfp4Linear` consumes. That container is host-side,
//!     so the repack itself compiles and is tested on every lane; only the residency *decision* is
//!     hardware-gated.
//!
//!   E5M2 has no weight-side GEMM leg here and MXFP8 has no block-scaled kernel in this workspace,
//!   so both always take the dense fallback.
//!
//! Residency is therefore priced per layer at plan time (packed = stored bytes + retained scales;
//! dense = logical shape × bf16) and the receipt measures the same quantity from what was actually
//! materialized; the reader asserts nothing silently.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::OnceLock;

use gen_core::checkpoint_codec::{
    CheckpointCodecRegistration, CheckpointCodecRegistry, CodecResidencyPolicy,
    CodecResidencyReport, LogicalKeyMapping, LogicalReadMaterialization, LogicalTensorPlan,
    LogicalWeightPlan, LogicalWeightReceipt, ResidencyMode, ScalarScaleSource, TensorCodecSpec,
    WeightEncoding, DENSE_BF16_CODEC, DENSE_F16_CODEC, DENSE_F32_CODEC, FP8_E4M3_SCALAR_CODEC,
    FP8_E5M2_SCALAR_CODEC, INT8_PER_ROW_CODEC, MXFP8_CODEC, NVFP4_CODEC,
};
use gen_core::ProviderRegistryBuilder;

use crate::quant::Nvfp4Tensor;

use crate::candle_core::safetensors::MmapedSafetensors;
use crate::candle_core::{DType, Device, Tensor};
use crate::{CandleError, Result};

/// Descriptor payloads are tiny JSON blobs; anything above this is not a `.comfy_quant` tensor.
const MAX_DESCRIPTOR_BYTES: u64 = 65_536;

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
    MXFP8_CODEC.codec_id,
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
}

impl CandleCodecResidency {
    /// The dense-only policy (CPU, Metal, or CUDA below both floors).
    pub const DENSE: Self = Self {
        fp8_e4m3_native: false,
        nvfp4_native: false,
    };

    /// Probe the device's eligibility for each native leg. Non-CUDA devices (and every build
    /// without the `cuda` feature) are dense-only; a CUDA device whose capability cannot be read is
    /// treated as below both floors (dense fallback — the safe direction, never the silent-packed
    /// one).
    pub fn probe(device: &Device) -> Self {
        Self {
            fp8_e4m3_native: cuda_meets_sm89_floor(device),
            nvfp4_native: cuda_meets_sm120_floor(device),
        }
    }
}

/// Whether one NVFP4 layer's **stored layout** is one the cuBLASLt FP4 GEMM accepts, independent of
/// hardware: the contraction K (the logical `in_features`) must be a multiple of
/// [`crate::quant::NVFP4_K_ALIGN`] and the output N a multiple of [`crate::quant::NVFP4_N_ALIGN`] —
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
    stored_shape[1].is_multiple_of(crate::quant::NVFP4_K_ALIGN)
        && stored_shape[0].is_multiple_of(crate::quant::NVFP4_N_ALIGN)
}

impl CodecResidencyPolicy for CandleCodecResidency {
    fn residency(
        &self,
        codec: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        stored_shape: &[usize],
    ) -> ResidencyMode {
        // The compiler already forces Dense for `full_precision_matrix_mult` layers; checking it
        // here keeps this predicate honest if that ever moved.
        if spec.full_precision_matrix_mult() {
            return ResidencyMode::Dense;
        }
        // The scalar E4M3 row. The plain undescribed cast (unit scale) qualifies too:
        // `matmul_fp8` takes any f32 weight scale.
        //
        // **Rank-2 only** — the contract this module's header already states ("a rank-2 E4M3
        // weight"), enforced rather than merely documented. `CublasLt::matmul_fp8` is a matrix
        // multiply: its weight-side operand is `[out, in]`. An fp8 checkpoint's rank-1 rows are
        // real — ComfyUI casts biases and modulation vectors to fp8 alongside the projections —
        // and planning one `Packed` hands `Nvfp4Linear`'s fp8 sibling a rank-1 container it cannot
        // read, while MLX (dense-only) decodes the same tensor correctly. That divergence is a
        // backend disagreement about the same file, so the floor is enforced here, at plan time,
        // where the fallback is the dense decode both backends share.
        if self.fp8_e4m3_native
            && codec.codec_id == FP8_E4M3_SCALAR_CODEC.codec_id
            && stored_shape.len() == 2
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

/// The locked-decision-7 sm_89 predicate, applied to a plan-time `Device`.
///
/// The threshold is **not** re-derived here: the capability is read off the device and handed to
/// [`crate::quant::compute_cap_meets_fp8_floor`], the same predicate
/// `CublasLt::meets_fp8_floor` (`cfg(cuda)`, hence unlinked here) applies to a
/// bound handle. Planning cannot go through the handle itself — `CublasLt::new` allocates the
/// handle's 32 MiB workspace, and residency is decided before any GEMM exists.
#[cfg(feature = "cuda")]
fn cuda_meets_sm89_floor(device: &Device) -> bool {
    use crate::candle_core::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    let Device::Cuda(cuda) = device else {
        return false;
    };
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let (Ok(major), Ok(minor)) = (
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
    ) else {
        return false;
    };
    crate::quant::compute_cap_meets_fp8_floor((major, minor))
}

#[cfg(not(feature = "cuda"))]
fn cuda_meets_sm89_floor(_device: &Device) -> bool {
    false
}

/// The NVFP4 `sm_120` predicate, applied to a plan-time `Device` (sc-20641) — the same shape as
/// [`cuda_meets_sm89_floor`], and for the same reason: the threshold is **not** re-derived, the
/// capability is read off the device and handed to
/// [`crate::quant::compute_cap_meets_nvfp4_floor`], the same predicate `CublasLt::meets_nvfp4_floor`
/// applies to a bound handle.
#[cfg(feature = "cuda")]
fn cuda_meets_sm120_floor(device: &Device) -> bool {
    use crate::candle_core::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    let Device::Cuda(cuda) = device else {
        return false;
    };
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let (Ok(major), Ok(minor)) = (
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
    ) else {
        return false;
    };
    crate::quant::compute_cap_meets_nvfp4_floor((major, minor))
}

#[cfg(not(feature = "cuda"))]
fn cuda_meets_sm120_floor(_device: &Device) -> bool {
    false
}

/// Header-plus-descriptors plan for one safetensors file under the given mapping and residency
/// policy. Refuses before any tensor is created; the error names the exact on-disk tensor.
pub fn plan_logical_weights(
    path: &Path,
    mapping: &dyn LogicalKeyMapping,
    residency: &dyn CodecResidencyPolicy,
) -> Result<LogicalWeightPlan> {
    let headers = gen_core::safetensors_path_tensor_headers(path)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    let descriptors = gen_core::read_safetensors_tensor_payloads(
        path,
        |header| header.name.ends_with(".comfy_quant"),
        MAX_DESCRIPTOR_BYTES,
    )
    .map_err(|error| CandleError::Msg(error.to_string()))?;
    // A checkpoint declares its quantization per layer (`.comfy_quant` tensors), file-wide
    // (`__metadata__._quantization_metadata`, sc-20641), or not at all; both routes are read and
    // the compiler refuses a layer the two disagree about.
    let quantization_metadata = gen_core::safetensors_path_quantization_metadata(path)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    gen_core::compile_logical_weight_plan_with_metadata(
        &headers,
        &descriptors,
        quantization_metadata.as_deref(),
        mapping,
        baseline_codec_registry(),
        residency,
    )
    .map_err(|error| {
        CandleError::Msg(format!(
            "logical weight plan for {} ({}): {error}",
            path.display(),
            mapping.mapping_id()
        ))
    })
}

/// One decoded logical tensor.
pub enum LogicalTensor {
    /// The dense resident form (pass-through dtype for dense rows, bf16 for dequantized rows).
    Dense(Tensor),
    /// The packed-native fp8 E4M3 form: the stored codes (dtype `F8E4M3`, on device) plus the
    /// per-tensor scales — exactly the `CublasLt::matmul_fp8` weight-side operands.
    PackedFp8E4M3 {
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
    },
    /// The packed-native NVFP4 form (sc-20641): the checkpoint's E2M1 nibbles and both scale levels
    /// repacked into the canonical container [`crate::quant::Nvfp4Linear::from_packed_in`] consumes.
    ///
    /// Unlike [`Self::PackedFp8E4M3`] this variant is **not** `cuda`-gated. `Nvfp4Tensor` is a host
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
            Self::Dense(tensor) => (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64,
            Self::PackedFp8E4M3 {
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

fn scalar_f32(bytes: &[u8], what: &str) -> Result<f32> {
    if bytes.len() != 4 {
        return Err(CandleError::Msg(format!(
            "{what}: expected one F32 scalar (4 bytes), got {} bytes",
            bytes.len()
        )));
    }
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
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
    // SAFETY: read-only mmap of a weight file; the standard candle loading path.
    let st = unsafe { MmapedSafetensors::new(path)? };
    let mut on_disk: Vec<String> = st.tensors().into_iter().map(|(name, _)| name).collect();
    on_disk.sort_unstable();
    let mut planned: Vec<&str> = plan.all_physical_keys().collect();
    planned.sort_unstable();
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

    let mut tensors = HashMap::new();
    let mut residency: BTreeMap<&'static str, CodecResidencyReport> = BTreeMap::new();
    let mut codec_by_owner: BTreeMap<&str, &'static str> = BTreeMap::new();
    for tensor in &plan.tensors {
        codec_by_owner.insert(tensor.physical_key.as_str(), tensor.codec_id);
        let decoded = decode(&st, tensor, device)?;
        let report = residency
            .entry(tensor.codec_id)
            .or_insert(CodecResidencyReport {
                codec_id: tensor.codec_id,
                tensor_count: 0,
                source_bytes: 0,
                resident_bytes: 0,
            });
        report.tensor_count += 1;
        report.source_bytes = report.source_bytes.saturating_add(tensor.source_bytes);
        // The single measured source of resident cost: [`LogicalTensor::resident_bytes`] reads it
        // off the decoded value itself — including the scale companions a packed load retained,
        // which the packed variant *owns* (it holds them as f32 values, not as file rows). The
        // companion loop below therefore attributes source bytes only. Reading the retained half
        // back off `plan.companions` would make `receipt == plan` self-referential on exactly the
        // packed row the pair exists to cross-check.
        report.resident_bytes = report
            .resident_bytes
            .saturating_add(decoded.resident_bytes());
        tensors.insert(tensor.logical_key.clone(), decoded);
    }
    for companion in &plan.companions {
        let Some(codec_id) = codec_by_owner.get(companion.owner_physical_key.as_str()) else {
            continue;
        };
        if let Some(report) = residency.get_mut(codec_id) {
            report.source_bytes = report.source_bytes.saturating_add(companion.source_bytes);
        }
    }
    Ok(LogicalWeights {
        tensors,
        receipt: LogicalWeightReceipt {
            mapping_id: plan.mapping_id,
            tensor_count: plan.tensors.len(),
            source_bytes: plan.source_bytes,
            materialization: LogicalReadMaterialization::Materialized,
            residency: residency.into_values().collect(),
        },
    })
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

fn upload_bf16(values: Vec<f32>, shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(values, shape, &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(device)?)
}

/// Run the planned codec on one stored tensor — the per-layer dispatch.
fn decode(
    st: &MmapedSafetensors,
    tensor: &LogicalTensorPlan,
    device: &Device,
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

    let view = st.get(&tensor.physical_key)?;
    match &tensor.codec {
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
            guard_stored(tensor, view.dtype(), view.shape(), expected, &tensor.shape)?;
            // Byte-preserving: no cast, no substitution can hide inside the dense rows.
            Ok(LogicalTensor::Dense(st.load(&tensor.physical_key, device)?))
        }
        TensorCodecSpec::ScalarFp8 {
            scale, input_scale, ..
        } => {
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
                &tensor.shape,
            )?;
            let (weight_scale, weight_scale_from_companion) = match scale {
                ScalarScaleSource::Unit => (1.0_f32, false),
                ScalarScaleSource::Companion { physical_key } => (
                    scalar_f32(companion_bytes(st, tensor, physical_key)?, physical_key)?,
                    true,
                ),
            };
            match tensor.residency.mode {
                ResidencyMode::Dense => {
                    let values: Vec<f32> = view
                        .data()
                        .iter()
                        .map(|&byte| element(byte) * weight_scale)
                        .collect();
                    Ok(LogicalTensor::Dense(upload_bf16(
                        values,
                        &tensor.shape,
                        device,
                    )?))
                }
                ResidencyMode::Packed => {
                    #[cfg(feature = "cuda")]
                    {
                        let input_scale = input_scale
                            .as_deref()
                            .map(|key| scalar_f32(companion_bytes(st, tensor, key)?, key))
                            .transpose()?;
                        let codes = st.load(&tensor.physical_key, device)?;
                        Ok(LogicalTensor::PackedFp8E4M3 {
                            codes,
                            weight_scale,
                            weight_scale_from_companion,
                            input_scale,
                        })
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = (input_scale, weight_scale_from_companion);
                        Err(CandleError::Msg(format!(
                            "codec {}: tensor {:?} was planned with packed-native fp8 residency, \
                             but this build has no CUDA fp8 leg; replan with a dense residency \
                             policy instead of silently substituting the dense fallback",
                            tensor.codec_id, tensor.physical_key
                        )))
                    }
                }
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
            let mut values = Vec::new();
            gen_core::decode_mxfp8(
                view.data(),
                scales,
                *stored_shape,
                [tensor.shape[0], tensor.shape[1]],
                &mut values,
            )
            .map_err(|error| {
                CandleError::Msg(format!(
                    "codec {}: tensor {:?}: {error}",
                    tensor.codec_id, tensor.physical_key
                ))
            })?;
            Ok(LogicalTensor::Dense(upload_bf16(
                values,
                &tensor.shape,
                device,
            )?))
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
                    gen_core::decode_nvfp4(
                        view.data(),
                        scales,
                        global,
                        *stored_shape,
                        [tensor.shape[0], tensor.shape[1]],
                        &mut values,
                    )
                    .map_err(|error| {
                        CandleError::Msg(format!(
                            "codec {}: tensor {:?}: {error}",
                            tensor.codec_id, tensor.physical_key
                        ))
                    })?;
                    Ok(LogicalTensor::Dense(upload_bf16(
                        values,
                        &tensor.shape,
                        device,
                    )?))
                }
                ResidencyMode::Packed => {
                    // The repack into the container `Nvfp4Linear` consumes: nibble-order swap plus
                    // the block-scale atom-order permutation (see `Nvfp4Tensor::from_kitchen_parts`).
                    // The container holds exactly the stored grid and carries no unpad, so a packed
                    // decode is only sound when the layer *is* that grid. `CandleCodecResidency`
                    // plans Dense otherwise; refuse here rather than trust it, because silently
                    // repacking a padded layer widens it and feeds the GEMM padding as real
                    // contraction elements (sc-20641).
                    if tensor.shape.as_slice() != stored_shape.as_slice() {
                        return Err(CandleError::Msg(format!(
                            "codec {}: tensor {:?} planned packed-native NVFP4 residency but its \
                             logical shape {:?} is not its stored shape {stored_shape:?}; ComfyUI \
                             padded this layer and the packed container cannot express the unpad",
                            tensor.codec_id, tensor.physical_key, tensor.shape
                        )));
                    }
                    let tensor_packed = Nvfp4Tensor::from_kitchen_parts(
                        view.data(),
                        scales,
                        global,
                        stored_shape[0],
                        stored_shape[1],
                    )
                    .map_err(|error| {
                        CandleError::Msg(format!(
                            "codec {}: tensor {:?} planned packed-native NVFP4 residency but its \
                             stored layout does not repack: {error}",
                            tensor.codec_id, tensor.physical_key
                        ))
                    })?;
                    let input_scale = input_scale
                        .as_deref()
                        .map(|key| scalar_f32(companion_bytes(st, tensor, key)?, key))
                        .transpose()?;
                    Ok(LogicalTensor::PackedNvfp4 {
                        tensor: Box::new(tensor_packed),
                        input_scale,
                    })
                }
            }
        }
        TensorCodecSpec::Int8PerRow { scale, .. } => {
            guard_stored(
                tensor,
                view.dtype(),
                view.shape(),
                StDtype::I8,
                &tensor.shape,
            )?;
            let scale_bytes = companion_bytes(st, tensor, scale)?;
            let rows = tensor.shape[0];
            let cols = tensor.shape[1];
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
                scale_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            };
            let codes: &[u8] = view.data();
            let mut values = Vec::new();
            gen_core::decode_int8_per_row(
                // SAFETY-free reinterpretation: i8 and u8 share layout; map explicitly instead.
                &codes.iter().map(|&byte| byte as i8).collect::<Vec<i8>>(),
                &scales,
                cols,
                &mut values,
            );
            Ok(LogicalTensor::Dense(upload_bf16(
                values,
                &tensor.shape,
                device,
            )?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StripModel;

    impl LogicalKeyMapping for StripModel {
        fn mapping_id(&self) -> &'static str {
            "strip-model-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
    }

    /// [`StripModel`] that also **declares** one logical shape for every key it maps.
    ///
    /// A block-padded codec row (MXFP8/NVFP4) is only materializable when the adapter states the
    /// layer's true geometry — otherwise the plan can do nothing but carry the padded stored grid
    /// forward, and decoding it would promote padding to weights (`gen_core` refuses by name). The
    /// NVFP4 fixtures below are built UNPADDED on purpose, so declaring the grid they were built at
    /// is the adapter making a true statement about them, not a restatement of the storage.
    struct StripModelDeclaring(Vec<usize>);

    impl LogicalKeyMapping for StripModelDeclaring {
        fn mapping_id(&self) -> &'static str {
            "strip-model-declaring-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
        fn logical_shape(&self, _logical_key: &str) -> Option<Vec<usize>> {
            Some(self.0.clone())
        }
    }

    fn fixture_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("candle-logical-weights-{}-", std::process::id()))
            .tempdir()
            .expect("fixture dir")
    }

    /// sc-20385 review: gen-core's own 256-code sweeps check `fp8_e4m3fn_to_f32` against a
    /// `spec_e4m3fn` helper written from the same OCP bit fields in the same file — a
    /// transliteration, so the two share any misreading of the table. `float8::F8E4M3` is a
    /// third-party implementation of the same standard; agreeing with it on all 256 codes is
    /// independent evidence. E5M2 gets the same treatment (its existing second oracle is the
    /// binary16-top-byte identity, which is genuinely independent, so this is belt and braces).
    ///
    /// NaN is compared by class: E4M3FN has no infinities and two NaN codes (`0x7F`, `0xFF`).
    #[test]
    fn fp8_references_agree_with_the_float8_crate_on_all_256_codes() {
        for code in 0..=u8::MAX {
            let ours = gen_core::fp8_e4m3fn_to_f32(code);
            let theirs = float8::F8E4M3::from_bits(code).to_f32();
            assert_eq!(
                ours.is_nan(),
                theirs.is_nan(),
                "e4m3 code {code:#04x}: NaN class disagrees ({ours} vs {theirs})"
            );
            if !ours.is_nan() {
                assert_eq!(
                    ours.to_bits(),
                    theirs.to_bits(),
                    "e4m3 code {code:#04x}: {ours} vs {theirs}"
                );
            }

            let ours = gen_core::fp8_e5m2_to_f32(code);
            let theirs = float8::F8E5M2::from_bits(code).to_f32();
            assert_eq!(
                ours.is_nan(),
                theirs.is_nan(),
                "e5m2 code {code:#04x}: NaN class disagrees ({ours} vs {theirs})"
            );
            if !ours.is_nan() {
                assert_eq!(
                    ours.to_bits(),
                    theirs.to_bits(),
                    "e5m2 code {code:#04x}: {ours} vs {theirs}"
                );
            }
        }
        // The sweep is only meaningful if the oracle actually distinguishes codes.
        assert_eq!(float8::F8E4M3::from_bits(0x38).to_f32(), 1.0);
        assert_eq!(float8::F8E5M2::from_bits(0x3C).to_f32(), 1.0);
    }

    /// sc-20385 review: the residency policy must not re-derive the locked-decision-7 sm_89
    /// threshold. `CublasLt::meets_fp8_floor` and `CandleCodecResidency::probe`'s device predicate
    /// both go through this one function, so the grid below pins the single definition. (The
    /// `probe` side is `cfg(cuda)`; the shared predicate is not, so this runs on every lane.)
    #[test]
    fn the_eight_bit_track_floor_has_one_definition_at_sm89() {
        assert_eq!(crate::quant::FP8_COMPUTE_CAP_FLOOR, (8, 9));
        for (cap, expected) in [
            ((7, 5), false),
            ((8, 0), false),
            ((8, 6), false),
            ((8, 8), false),
            ((8, 9), true),
            ((8, 10), true),
            ((9, 0), true),
            ((12, 0), true),
        ] {
            assert_eq!(
                crate::quant::compute_cap_meets_fp8_floor(cap),
                expected,
                "compute capability {cap:?}"
            );
        }
    }

    /// Write a minimal safetensors file from `(name, dtype, shape, little-endian payload)` rows.
    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header_entries = Vec::new();
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).expect("write safetensors fixture");
    }

    /// [`write_safetensors`] plus a file-level `__metadata__` map — the route a header-declared
    /// NVFP4 checkpoint takes (sc-20641).
    fn write_safetensors_with_metadata(
        path: &Path,
        tensors: &[(&str, &str, &[usize], Vec<u8>)],
        metadata: &[(&str, &str)],
    ) {
        let mut header_entries = vec![format!(
            "\"__metadata__\":{{{}}}",
            metadata
                .iter()
                .map(|(key, value)| format!("{key:?}:{value:?}"))
                .collect::<Vec<_>>()
                .join(",")
        )];
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).expect("write safetensors fixture");
    }

    // ---- an independent NVFP4 reference -------------------------------------------------------
    //
    // Written from the format specification and ComfyUI's own quantizer
    // (`comfy.quant_ops.TensorCoreNVFP4Layout.quantize`, `comfy.float.to_blocked` /
    // `stochastic_float_to_fp4_e2m1`), NOT by calling the codec under test. The FP8 E4M3 half goes
    // through the third-party `float8` crate; the E2M1 grid and the `to_blocked` swizzle are
    // derived here from their definitions.

    /// `comfy.float.to_blocked`, transliterated as an explicit index walk: view the padded
    /// `[R, B]` scale matrix as `(R/128, 128, B/4, 4)`, permute to `(R/128, B/4, 128, 4)`, reshape
    /// to `(-1, 4, 32, 4)`, transpose axes 1 and 2, flatten. Returns the flat destination index of
    /// logical `(row, block)`.
    fn reference_to_blocked_index(blocks: usize, row: usize, block: usize) -> usize {
        let n_col_blocks = blocks.div_ceil(4);
        // (R/128, B/4, 128, 4) — atoms walked row-major, then 128 rows × 4 block-cols inside.
        let atom = (row / 128) * n_col_blocks + block / 4;
        let (row_in_atom, col_in_atom) = (row % 128, block % 4);
        // (128, 4) viewed as (4, 32, 4) then transposed to (32, 4, 4): source (a, b, c) with
        // row_in_atom = a * 32 + b lands at b * 16 + a * 4 + c.
        let (a, b) = (row_in_atom / 32, row_in_atom % 32);
        atom * 512 + b * 16 + a * 4 + col_in_atom
    }

    /// The eight non-negative E2M1 magnitudes, derived from the bit fields (2 exponent bits at
    /// bias 1, 1 mantissa bit) rather than copied from a table.
    fn reference_e2m1_magnitudes() -> [f32; 8] {
        let mut grid = [0.0_f32; 8];
        for (code, slot) in grid.iter_mut().enumerate() {
            let (e, m) = ((code >> 1) as i32, (code & 1) as f32);
            *slot = if e == 0 {
                m / 2.0
            } else {
                (1.0 + m / 2.0) * 2f32.powi(e - 1)
            };
        }
        grid
    }

    /// Nearest E2M1 code (sign in bit 3) for a value already divided by its block scale.
    fn reference_e2m1_code(value: f32) -> u8 {
        let grid = reference_e2m1_magnitudes();
        let magnitude = value.abs().min(6.0);
        let mut best = 0_usize;
        for (index, candidate) in grid.iter().enumerate() {
            if (candidate - magnitude).abs() < (grid[best] - magnitude).abs() {
                best = index;
            }
        }
        (if value.is_sign_negative() { 0x08 } else { 0 }) | best as u8
    }

    /// Quantize a dense row-major `[rows, cols]` f32 matrix to the three stored NVFP4 artifacts,
    /// and return the values a correct decoder must reproduce.
    #[allow(clippy::type_complexity)]
    fn nvfp4_reference_quantize(
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> (Vec<u8>, Vec<u8>, f32, Vec<f32>) {
        assert_eq!(data.len(), rows * cols);
        assert!(cols.is_multiple_of(16) && rows.is_multiple_of(16));
        let blocks = cols / 16;
        // `scale = amax(|W|) / (F8_E4M3_MAX * F4_E2M1_MAX)` — quant_ops.py line 134.
        let amax = data.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
        let global = amax / (448.0 * 6.0);

        let scale_rows = rows.div_ceil(128) * 128;
        let scale_cols = blocks.div_ceil(4) * 4;
        let mut blocked_scales = vec![0_u8; scale_rows * scale_cols];
        let mut packed = vec![0_u8; rows * cols / 2];
        let mut reference = vec![0.0_f32; rows * cols];

        for row in 0..rows {
            for block in 0..blocks {
                let span = &data[row * cols + block * 16..row * cols + block * 16 + 16];
                // `clamp(amax(|blk|) / F4_E2M1_MAX / per_tensor_scale, max=448).to(e4m3)`.
                let block_amax = span.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
                let sf = (block_amax / 6.0 / global).min(448.0);
                let sf_byte = float8::F8E4M3::from_f32(sf).to_bits();
                blocked_scales[reference_to_blocked_index(blocks, row, block)] = sf_byte;

                let element_scale = float8::F8E4M3::from_bits(sf_byte).to_f32() * global;
                for (offset, value) in span.iter().enumerate() {
                    let col = block * 16 + offset;
                    let code = if element_scale > 0.0 {
                        reference_e2m1_code(value / element_scale)
                    } else {
                        0
                    };
                    // `packed = (even << 4) | odd` — float.py line 95: even element, high nibble.
                    let byte = &mut packed[(row * cols + col) / 2];
                    if col.is_multiple_of(2) {
                        *byte = (*byte & 0x0F) | (code << 4);
                    } else {
                        *byte = (*byte & 0xF0) | code;
                    }
                    let signed = if code & 0x08 != 0 { -1.0 } else { 1.0 };
                    reference[row * cols + col] = signed
                        * reference_e2m1_magnitudes()[(code & 0x07) as usize]
                        * element_scale;
                }
            }
        }
        (packed, blocked_scales, global, reference)
    }

    /// A deterministic spread of weights with a wide dynamic range across blocks, so per-block
    /// scales genuinely differ (a decode that used one scale everywhere would fail).
    fn nvfp4_fixture_values(rows: usize, cols: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * cols);
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for row in 0..rows {
            for col in 0..cols {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
                // Block-dependent decade so each 16-block gets its own exponent range.
                let decade = 2f32.powi(((row / 8 + col / 16) % 7) as i32 - 3);
                values.push(unit * decade);
            }
        }
        values
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// bf16 value of `x` (round-to-nearest-even) — the reference resident form.
    fn to_bf16(x: f32) -> f32 {
        if x.is_nan() {
            return x;
        }
        let bits = x.to_bits();
        f32::from_bits(bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)
    }

    fn dense_f32(weights: &LogicalWeights, key: &str) -> Vec<f32> {
        match weights.tensors.get(key).expect("logical tensor") {
            LogicalTensor::Dense(tensor) => tensor
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            LogicalTensor::PackedFp8E4M3 { .. } | LogicalTensor::PackedNvfp4 { .. } => {
                panic!("{key} unexpectedly packed")
            }
        }
    }

    fn plan_dense(path: &Path) -> LogicalWeightPlan {
        plan_logical_weights(path, &StripModel, &CandleCodecResidency::DENSE).expect("plan")
    }

    #[test]
    fn catalog_registration_carries_the_shared_baseline_codec_table() {
        let registry = baseline_codec_registry();
        let mut registered: Vec<&str> = registry.codecs().map(|codec| codec.codec_id).collect();
        registered.sort_unstable();
        let mut implemented = CODEC_IMPLEMENTATION_IDS.to_vec();
        implemented.sort_unstable();
        assert_eq!(registered, implemented);
        let catalog = register_checkpoint_codecs(ProviderRegistryBuilder::new())
            .build()
            .expect("codec table registers into an empty catalog");
        assert_eq!(
            catalog
                .checkpoint_codecs()
                .codecs()
                .copied()
                .collect::<Vec<_>>(),
            BASELINE_CODECS
        );
        // The table itself is gen-core's, not this crate's: a codec is a backend-portable
        // declaration, and two hand-kept copies could drift into a checkpoint one engine plans and
        // the other refuses with nothing comparing them.
        assert_eq!(
            BASELINE_CODECS,
            gen_core::checkpoint_codec::BASELINE_CHECKPOINT_CODECS
        );
    }

    /// Golden: `fp8-e4m3-scalar-v1` dense fallback decodes exactly to the gen-core reference
    /// (spec-derived; the fixture covers 1.0, max 448, min normal, min subnormal, negatives, ±0).
    #[test]
    fn fp8_e4m3_scalar_golden_decodes_exactly_on_the_dense_fallback() {
        let dir = fixture_dir();
        let path = dir.path().join("e4m3.safetensors");
        let codes: Vec<u8> = vec![0x38, 0x7E, 0x08, 0x01, 0xB9, 0xC0, 0x00, 0x80];
        let scale = 0.03125_f32;
        let descriptor = br#"{"format": "float8_e4m3fn"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[2, 4], codes.clone()),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    scale.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_dense(&path);
        assert_eq!(plan.codec_ids(), ["fp8-e4m3-scalar-v1"]);
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let mut expected = Vec::new();
        gen_core::decode_fp8_e4m3fn_scalar(&codes, scale, &mut expected);
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(dense_f32(&weights, "q.weight"), expected);
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
        assert_eq!(weights.receipt.resident_bytes(), 16);
    }

    /// Golden: `fp8-e5m2-scalar-v1` (scaled) plus the undescribed plain cast (unit scale).
    #[test]
    fn fp8_e5m2_scalar_and_plain_cast_golden_decode_exactly() {
        let dir = fixture_dir();
        let path = dir.path().join("e5m2.safetensors");
        let codes: Vec<u8> = vec![0x3C, 0x7B, 0x04, 0x01, 0xBF, 0x00, 0x80, 0x42];
        let scale = 0.25_f32;
        let descriptor = br#"{"format": "float8_e5m2"}"#;
        let plain: Vec<u8> = vec![0x3C, 0x40, 0xC2, 0x01];
        write_safetensors(
            &path,
            &[
                ("model.k.weight", "F8_E5M2", &[2, 4], codes.clone()),
                (
                    "model.k.weight_scale",
                    "F32",
                    &[1],
                    scale.to_le_bytes().to_vec(),
                ),
                (
                    "model.k.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
                ("model.k.bias", "F8_E5M2", &[4], plain.clone()),
            ],
        );
        let plan = plan_dense(&path);
        assert_eq!(plan.codec_ids(), ["fp8-e5m2-scalar-v1"]);
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let mut expected = Vec::new();
        gen_core::decode_fp8_e5m2_scalar(&codes, scale, &mut expected);
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        assert_eq!(dense_f32(&weights, "k.weight"), expected);
        let mut expected_plain = Vec::new();
        gen_core::decode_fp8_e5m2_scalar(&plain, 1.0, &mut expected_plain);
        let expected_plain: Vec<f32> = expected_plain.into_iter().map(to_bf16).collect();
        assert_eq!(dense_f32(&weights, "k.bias"), expected_plain);
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
    }

    /// Golden: `mxfp8-v1` with a declared logical shape [37, 70] inside stored [64, 96] — padding
    /// rows, a non-block-aligned column tail, poison in the padding, swizzled scales.
    ///
    /// Run for **both** on-disk spellings of the block-scale companion. `F8_E8M0` is what ComfyUI
    /// actually writes (`torch.float8_e8m0fnu`) and is the case that matters here: candle reads
    /// through `MmapedSafetensors` — the `safetensors` crate's parser, not gen-core's header
    /// reader — so a fixture that only ever stored `U8` proved nothing about the real dtype
    /// surviving that second parser. `U8` stays covered because re-serialized checkpoints in the
    /// wild carry the scales that way.
    #[test]
    fn mxfp8_golden_unswizzles_unpads_and_decodes_exactly() {
        for scale_dtype in ["F8_E8M0", "U8"] {
            mxfp8_golden_case(scale_dtype);
        }
    }

    fn mxfp8_golden_case(scale_dtype: &str) {
        struct DeclaredShape;
        impl LogicalKeyMapping for DeclaredShape {
            fn mapping_id(&self) -> &'static str {
                "declared-shape-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
                (logical_key == "v.weight").then(|| vec![37, 70])
            }
        }
        let dir = fixture_dir();
        let path = dir.path().join(format!("mxfp8-{scale_dtype}.safetensors"));
        // 64 rows exercises BOTH 32-row halves of a 128-row swizzle tile; 96 columns = 3 blocks,
        // so the swizzled scale matrix carries a padded fourth column; logical [37, 70] leaves a
        // non-block-aligned column tail (70 = 2 full blocks + 6) and 27 padded rows.
        let stored = [64_usize, 96];
        let mut values = vec![0x7E_u8; stored[0] * stored[1]];
        for row in 0..37 {
            for col in 0..70 {
                values[row * stored[1] + col] = 0x38 + ((row + col) % 4) as u8;
            }
        }
        let scale_shape = gen_core::mxfp8_scale_shape([stored[0], stored[1]]);
        let mut scales = vec![0xFF_u8; scale_shape[0] * scale_shape[1]];
        for row in 0..37 {
            for block in 0..3 {
                scales[gen_core::mxfp8_swizzled_scale_index([stored[0], stored[1]], row, block)] =
                    126 + ((row + block) % 3) as u8;
            }
        }
        let descriptor = br#"{"format": "mxfp8"}"#;
        write_safetensors(
            &path,
            &[
                ("model.v.weight", "F8_E4M3", &stored, values.clone()),
                (
                    "model.v.weight_scale",
                    scale_dtype,
                    &scale_shape,
                    scales.clone(),
                ),
                (
                    "model.v.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &DeclaredShape, &CandleCodecResidency::DENSE)
            .unwrap_or_else(|error| panic!("plan ({scale_dtype} scales): {error}"));
        assert_eq!(plan.codec_ids(), ["mxfp8-v1"], "{scale_dtype}");
        assert_eq!(plan.tensors[0].shape, vec![37, 70], "{scale_dtype}");
        let weights = read_logical_weights(&path, &plan, &Device::Cpu)
            .unwrap_or_else(|error| panic!("read ({scale_dtype} scales): {error}"));
        let mut expected = Vec::new();
        gen_core::decode_mxfp8(&values, &scales, stored, [37, 70], &mut expected).unwrap();
        let expected: Vec<f32> = expected.into_iter().map(to_bf16).collect();
        let got = dense_f32(&weights, "v.weight");
        assert_eq!(got, expected, "{scale_dtype}");
        assert!(
            got.iter().all(|value| value.is_finite() && *value < 400.0),
            "poison from the padding region leaked into the logical tensor ({scale_dtype})"
        );
        assert_eq!(
            weights.receipt.resident_bytes(),
            plan.resident_bytes(),
            "{scale_dtype}"
        );
        assert_eq!(
            weights.receipt.resident_bytes(),
            37 * 70 * 2,
            "{scale_dtype}"
        );
    }

    /// The int8 per-row row plus a mixed file: dense bf16 + e4m3 + int8 dispatch per layer.
    #[test]
    fn mixed_checkpoint_dispatches_per_layer_including_int8_per_row() {
        let dir = fixture_dir();
        let path = dir.path().join("mixed.safetensors");
        let e4m3_descriptor = br#"{"format": "float8_e4m3fn"}"#;
        let int8_descriptor = br#"{"format": "int8_tensorwise", "per_row": true}"#;
        write_safetensors(
            &path,
            &[
                ("model.dense.weight", "BF16", &[2], bf16_bytes(&[1.0, -1.0])),
                (
                    "model.q.weight",
                    "F8_E4M3",
                    &[1, 4],
                    vec![0x38, 0x40, 0x48, 0xB8],
                ),
                (
                    "model.q.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[e4m3_descriptor.len()],
                    e4m3_descriptor.to_vec(),
                ),
                (
                    "model.o.weight",
                    "I8",
                    &[2, 3],
                    vec![1, 0xFE, 3, 0xFC, 5, 0xFA],
                ),
                (
                    "model.o.weight_scale",
                    "F32",
                    &[2, 1],
                    [0.5_f32, 2.0]
                        .iter()
                        .flat_map(|scale| scale.to_le_bytes())
                        .collect(),
                ),
                (
                    "model.o.comfy_quant",
                    "U8",
                    &[int8_descriptor.len()],
                    int8_descriptor.to_vec(),
                ),
            ],
        );
        let plan = plan_dense(&path);
        assert_eq!(
            plan.codec_ids(),
            ["dense-bf16-v1", "fp8-e4m3-scalar-v1", "int8-per-row-v1"]
        );
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        assert_eq!(dense_f32(&weights, "dense.weight"), [1.0, -1.0]);
        assert_eq!(dense_f32(&weights, "q.weight"), [2.0, 4.0, 8.0, -2.0]);
        assert_eq!(
            dense_f32(&weights, "o.weight"),
            [0.5, -1.0, 1.5, -8.0, 10.0, -12.0]
        );
        assert_eq!(weights.receipt.residency.len(), 3);
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
        let total: u64 = weights
            .receipt
            .residency
            .iter()
            .map(|report| report.source_bytes)
            .sum();
        assert_eq!(total, plan.source_bytes);
    }

    /// The packed-native policy is a pure layout+hardware predicate: it selects `Packed` only for
    /// the E4M3 scalar row without `full_precision_matrix_mult`, and the plan prices packed
    /// entries at stored bytes + retained scales instead of dense bf16.
    #[test]
    fn candle_residency_policy_packs_only_the_eligible_e4m3_layers_and_prices_them() {
        let dir = fixture_dir();
        let path = dir.path().join("packed-pricing.safetensors");
        let e4m3 = br#"{"format": "float8_e4m3fn"}"#;
        let e4m3_fpmm = br#"{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}"#;
        let e5m2 = br#"{"format": "float8_e5m2"}"#;
        write_safetensors(
            &path,
            &[
                ("model.a.weight", "F8_E4M3", &[2, 4], vec![0x38; 8]),
                (
                    "model.a.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                ("model.a.comfy_quant", "U8", &[e4m3.len()], e4m3.to_vec()),
                ("model.b.weight", "F8_E4M3", &[2, 4], vec![0x38; 8]),
                (
                    "model.b.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.b.comfy_quant",
                    "U8",
                    &[e4m3_fpmm.len()],
                    e4m3_fpmm.to_vec(),
                ),
                ("model.c.weight", "F8_E5M2", &[2, 4], vec![0x3C; 8]),
                (
                    "model.c.weight_scale",
                    "F32",
                    &[],
                    2.0_f32.to_le_bytes().to_vec(),
                ),
                ("model.c.comfy_quant", "U8", &[e5m2.len()], e5m2.to_vec()),
            ],
        );
        let native = CandleCodecResidency {
            nvfp4_native: false,
            fp8_e4m3_native: true,
        };
        let plan = plan_logical_weights(&path, &StripModel, &native).expect("plan");
        let by_key: BTreeMap<&str, &LogicalTensorPlan> = plan
            .tensors
            .iter()
            .map(|tensor| (tensor.logical_key.as_str(), tensor))
            .collect();
        // Eligible: packed at stored bytes (8), its 4-byte scale retained on the companion row.
        assert_eq!(by_key["a.weight"].residency.mode, ResidencyMode::Packed);
        assert_eq!(by_key["a.weight"].residency.resident_bytes, 8);
        let scale_row = plan
            .companions
            .iter()
            .find(|companion| companion.physical_key == "model.a.weight_scale")
            .unwrap();
        assert_eq!(scale_row.resident_bytes, 4);
        // full_precision_matrix_mult: dense by contract, 2 × 4 × bf16 = 16.
        assert_eq!(by_key["b.weight"].residency.mode, ResidencyMode::Dense);
        assert_eq!(by_key["b.weight"].residency.resident_bytes, 16);
        // E5M2 has no native leg: dense.
        assert_eq!(by_key["c.weight"].residency.mode, ResidencyMode::Dense);
        // Packed vs dense pricing differs end to end.
        let dense_plan = plan_dense(&path);
        assert!(plan.resident_bytes() < dense_plan.resident_bytes());
        assert_eq!(dense_plan.resident_bytes(), 3 * 16);
        assert_eq!(plan.resident_bytes(), (8 + 4) + 16 + 16);

        // The probed policy on this machine's devices: CPU (and Metal) are dense-only.
        assert_eq!(
            CandleCodecResidency::probe(&Device::Cpu),
            CandleCodecResidency::DENSE
        );

        // On a build without the CUDA fp8 leg, reading a packed plan is a typed refusal — never a
        // silent dense substitution of what admission priced as packed.
        #[cfg(not(feature = "cuda"))]
        {
            let error = match read_logical_weights(&path, &plan, &Device::Cpu) {
                Ok(_) => panic!("packed plan must refuse on a non-cuda build"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("no CUDA fp8 leg"), "{error}");
        }
        // On a CUDA build the packed read keeps F8E4M3 codes + scales resident; the read itself is
        // exercised by the windows-cuda lane (this lane's CPU device cannot hold F8E4M3 GEMM
        // operands). The *accounting* the receipt would then report is not cuda-gated, so pin it
        // here against the plan rows above: the receipt measures the retained scales off the
        // decoded value, so this number is derived independently of `plan.companions`.
        let codes = Tensor::zeros((2, 4), DType::F8E4M3, &Device::Cpu).expect("codes");
        assert_eq!(
            LogicalTensor::PackedFp8E4M3 {
                codes: codes.clone(),
                weight_scale: 1.0,
                weight_scale_from_companion: true,
                input_scale: None,
            }
            .resident_bytes(),
            8 + 4,
            "packed codes + the retained weight_scale"
        );
        assert_eq!(
            LogicalTensor::PackedFp8E4M3 {
                codes,
                weight_scale: 1.0,
                weight_scale_from_companion: true,
                input_scale: Some(0.5),
            }
            .resident_bytes(),
            8 + 4 + 4,
            "a retained input_scale is priced too"
        );
    }

    /// AC1 (Candle). A header-declared NVFP4 golden fixture — packed E2M1 values, block-16 FP8
    /// scales in the `to_blocked` swizzle, a global tensor scale — decodes to exactly the values an
    /// independent reference (`nvfp4_reference_quantize`, written from the format spec and the
    /// third-party `float8` crate) says it holds, on the dense fallback.
    ///
    /// Shape `[256, 128]` is deliberately **multi-atom** in both scale dimensions (2 row atoms ×
    /// 2 block atoms), so an atom-order mistake in the un-swizzle is a wrong value, not a no-op.
    /// The fixture also carries no `.comfy_quant` tensors at all: the declaration lives in
    /// `__metadata__._quantization_metadata` under a *relative* layer name, the form the ComfyUI
    /// Kitchen converters write, so the prefix resolver is on the path too.
    #[test]
    fn nvfp4_golden_decodes_exactly_on_the_dense_fallback() {
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, reference) = nvfp4_reference_quantize(&values, rows, cols);
        assert_eq!(scales.len(), 256 * 8, "2 row atoms × 2 block atoms");

        write_safetensors_with_metadata(
            &path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed.clone()),
                ("model.q.weight_scale", "F8_E4M3", &[256, 8], scales.clone()),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                // Relative layer name: the tensors are `model.q.*`, the declaration says `q`.
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        // The fixture is built at exactly [rows, cols] — no ComfyUI padding — so the adapter
        // declaring that grid is a true statement about the layer, and it is REQUIRED: without a
        // declaration the plan could not tell this grid from a padded one and refuses to
        // materialize it (sc-20651).
        let mapping = StripModelDeclaring(vec![rows, cols]);
        let plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        assert_eq!(plan.codec_ids(), vec!["nvfp4-v1"]);
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.shape, vec![rows, cols], "the logical element grid");
        assert_eq!(tensor.residency.mode, ResidencyMode::Dense);
        // Dense residency is priced on the logical shape × bf16, not on the stored 4-bit bytes.
        assert_eq!(
            tensor.residency.resident_bytes,
            (rows * cols * 2) as u64,
            "dense fallback holds bf16"
        );
        // Both scale levels are consumed by a dense decode, so neither is priced resident.
        assert_eq!(plan.companions.len(), 2);
        assert!(plan
            .companions
            .iter()
            .all(|companion| companion.resident_bytes == 0));
        assert_eq!(
            plan.resident_bytes(),
            (rows * cols * 2) as u64,
            "dense plan prices only the decoded weight"
        );

        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let decoded = dense_f32(&weights, "q.weight");
        assert_eq!(decoded.len(), rows * cols);
        for (index, expected) in reference.iter().enumerate() {
            assert_eq!(
                decoded[index],
                to_bf16(*expected),
                "element {index} (row {}, col {})",
                index / cols,
                index % cols
            );
        }
        // The receipt measures what was materialized and agrees with the plan's pricing.
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());

        // Supplementary: round-trip error against the *original* f32 matrix. NVFP4 is lossy, so
        // this is a quantization-error bound, not a correctness proof — the exactness assertions
        // above are what pin the decode. Checked as a relative RMS; never a cosine.
        //
        // The band is the MEASURED error of this fixture (0.103) with headroom, not a spec figure:
        // uniform-random values are the worst case for E2M1's non-uniform 8-magnitude grid, whose
        // coarsest step (4 -> 6) is a third of its own magnitude. It exists to catch a decode that
        // drifts wholesale — a dropped scale level, the wrong global — which lands orders of
        // magnitude away, not to certify a precision claim.
        let (mut error, mut energy) = (0.0_f64, 0.0_f64);
        for (original, got) in values.iter().zip(decoded.iter()) {
            error += ((*original - *got) as f64).powi(2);
            energy += (*original as f64).powi(2);
        }
        let relative_rms = (error / energy).sqrt();
        assert!(
            (0.05..0.15).contains(&relative_rms),
            "NVFP4 relative RMS error {relative_rms} is outside the band this fixture measures \
             (0.103): far below means it stopped being quantized, far above means the decode drifted"
        );
    }

    /// AC2 (Candle, host half). The `sm_120` residency decision and the repack it selects.
    ///
    /// The hardware predicate is exercised through the one floor definition; the *repack* is host
    /// code, so a policy that reports the floor as met drives the real packed path on this lane and
    /// the resulting container decodes back to the same golden values. Live cuBLASLt execution of
    /// the resulting operands remains the windows-cuda box's job.
    #[test]
    fn nvfp4_packed_residency_repacks_and_prices_independently_of_the_dense_fallback() {
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-packed.safetensors");
        let (rows, cols) = (256_usize, 128_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, reference) = nvfp4_reference_quantize(&values, rows, cols);
        write_safetensors_with_metadata(
            &path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed.clone()),
                ("model.q.weight_scale", "F8_E4M3", &[256, 8], scales),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
        };
        // Unpadded fixture; the adapter declares the grid it was built at (see above).
        let mapping = StripModelDeclaring(vec![rows, cols]);
        let plan = plan_logical_weights(&path, &mapping, &native).expect("plan");
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.residency.mode, ResidencyMode::Packed);
        // Packed prices the stored 4-bit bytes; both scale levels are retained and priced on their
        // own companion rows — the two residencies are independent quantities, not a scaled copy.
        assert_eq!(tensor.residency.resident_bytes, (rows * cols / 2) as u64);
        let companions: u64 = plan
            .companions
            .iter()
            .map(|companion| companion.resident_bytes)
            .sum();
        assert_eq!(
            companions,
            (256 * 8) as u64 + 4,
            "block scales + F32 global"
        );
        assert_eq!(
            plan.resident_bytes(),
            (rows * cols / 2) as u64 + 256 * 8 + 4
        );
        let dense_plan =
            plan_logical_weights(&path, &mapping, &CandleCodecResidency::DENSE).expect("plan");
        assert!(plan.resident_bytes() < dense_plan.resident_bytes());

        // The read repacks into the container `Nvfp4Linear::from_packed_in` consumes...
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("read");
        let LogicalTensor::PackedNvfp4 {
            tensor: container,
            input_scale,
        } = weights.tensors.get("q.weight").expect("logical tensor")
        else {
            panic!("an sm_120-eligible NVFP4 layer must repack, not decode dense");
        };
        assert_eq!((container.rows, container.cols), (rows, cols));
        assert_eq!(container.global_scale, global);
        assert!(input_scale.is_none());
        // ...and that container still holds the golden values: the nibble swap and the block-scale
        // atom permutation are both lossless.
        let recovered = container.dequantize_to_vec();
        for (index, expected) in reference.iter().enumerate() {
            assert_eq!(recovered[index], *expected, "element {index}");
        }
        // Residency measured off the container, independently of the plan rows above.
        assert_eq!(weights.receipt.resident_bytes(), plan.resident_bytes());
    }

    /// sc-20641 review. ComfyUI pads NVFP4 storage to 16, so a layer can be stored WIDER than it
    /// is — `in_features = 60` stores `K = 64`, which passes the FP4 leg's 32-alignment rule. The
    /// packed container holds the stored grid and carries no unpad, so repacking such a layer hands
    /// `Nvfp4Linear` four columns of padding as real contraction elements.
    ///
    /// Two independent guards, tested separately: the residency policy plans **Dense** on sm_120
    /// hardware, and the decoder's Packed arm **refuses** if a plan ever asks for it anyway.
    #[test]
    fn a_padded_nvfp4_layer_plans_dense_and_a_forced_packed_decode_refuses() {
        struct DeclaresSixty;
        impl LogicalKeyMapping for DeclaresSixty {
            fn mapping_id(&self) -> &'static str {
                "declares-sixty-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
                (logical_key == "q.weight").then(|| vec![256, 60])
            }
        }
        /// Ignores every layout fact — stands in for a residency policy that regresses.
        struct ForcePacked;
        impl CodecResidencyPolicy for ForcePacked {
            fn residency(
                &self,
                _codec: &CheckpointCodecRegistration,
                _spec: &TensorCodecSpec,
                _stored_shape: &[usize],
            ) -> ResidencyMode {
                ResidencyMode::Packed
            }
        }

        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-padded.safetensors");
        let (rows, stored_cols, logical_cols) = (256_usize, 64_usize, 60_usize);
        // The pad columns must hold the zero code, which is what the geometry validator demands.
        let mut values = nvfp4_fixture_values(rows, stored_cols);
        for row in 0..rows {
            for col in logical_cols..stored_cols {
                values[row * stored_cols + col] = 0.0;
            }
        }
        let (packed, scales, global, _) = nvfp4_reference_quantize(&values, rows, stored_cols);
        assert_eq!(scales.len(), 256 * 4, "2 row atoms × 1 block atom");
        write_safetensors_with_metadata(
            &path,
            &[
                (
                    "model.q.weight",
                    "U8",
                    &[rows, stored_cols / 2],
                    packed.clone(),
                ),
                ("model.q.weight_scale", "F8_E4M3", &[256, 4], scales),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    global.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
        };
        let plan = plan_logical_weights(&path, &DeclaresSixty, &native).expect("plan");
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.shape, vec![rows, logical_cols], "the unpadded layer");
        assert!(
            matches!(
                tensor.codec,
                TensorCodecSpec::Nvfp4 {
                    stored_shape: [256, 64],
                    logical_shape: [256, 60],
                    ..
                }
            ),
            "the spec carries BOTH shapes: {:?}",
            tensor.codec
        );
        assert_eq!(
            tensor.residency.mode,
            ResidencyMode::Dense,
            "a padded layer takes the dense fallback even on sm_120 with an aligned stored grid"
        );
        // Dense is the honest plan and it reads back correctly at the logical width.
        let weights = read_logical_weights(&path, &plan, &Device::Cpu).expect("dense read");
        assert_eq!(
            dense_f32(&weights, "q.weight").len(),
            rows * logical_cols,
            "the dense fallback unpads"
        );

        // The decoder does not trust the policy: a Packed plan for this layer is refused, naming
        // both shapes, rather than silently widening the layer.
        let forced = plan_logical_weights(&path, &DeclaresSixty, &ForcePacked).expect("plan");
        assert_eq!(forced.tensors[0].residency.mode, ResidencyMode::Packed);
        let Err(error) = read_logical_weights(&path, &forced, &Device::Cpu) else {
            panic!("a packed decode of a padded NVFP4 layer must refuse");
        };
        let error = error.to_string();
        assert!(error.contains("model.q.weight"), "{error}");
        assert!(error.contains("[256, 60]"), "names the logical: {error}");
        assert!(error.contains("[256, 64]"), "names the stored: {error}");
    }

    /// AC2 (Candle). The `sm_120` floor has one definition, and a layer the FP4 leg cannot run
    /// takes the dense fallback rather than a packed plan that would fail at the first forward.
    #[test]
    fn the_nvfp4_floor_has_one_definition_at_sm120_and_the_layout_gate_is_separate() {
        assert_eq!(crate::quant::NVFP4_COMPUTE_CAP_FLOOR, (12, 0));
        for (cap, expected) in [
            ((8, 9), false),
            ((9, 0), false),
            ((10, 0), false),
            ((11, 8), false),
            ((12, 0), true),
            ((12, 1), true),
            ((13, 0), true),
        ] {
            assert_eq!(
                crate::quant::compute_cap_meets_nvfp4_floor(cap),
                expected,
                "compute capability {cap:?}"
            );
        }
        // The layout gate reads the same K/N alignment the GEMM enforces.
        assert_eq!(crate::quant::NVFP4_K_ALIGN, 32);
        assert_eq!(crate::quant::NVFP4_N_ALIGN, 16);
        assert!(nvfp4_layout_is_native([256, 128]));
        assert!(
            !nvfp4_layout_is_native([256, 48]),
            "K = 48 is NOT_SUPPORTED"
        );
        assert!(
            !nvfp4_layout_is_native([24, 128]),
            "N must be a multiple of 16"
        );

        // And the policy needs all three: sm_120 hardware, an accepted layout, AND an unpadded
        // layer (`spec_padded` below covers the last one).
        let spec_of = |stored_shape: [usize; 2],
                       logical_shape: [usize; 2],
                       full_precision: bool| TensorCodecSpec::Nvfp4 {
            block_scale: "q.weight_scale".to_owned(),
            global_scale: "q.weight_scale_2".to_owned(),
            input_scale: None,
            stored_shape,
            logical_shape,
            logical_shape_declared: false,
            full_precision_matrix_mult: full_precision,
        };
        let spec = |stored_shape: [usize; 2], full_precision: bool| {
            spec_of(stored_shape, stored_shape, full_precision)
        };
        let native = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
        };
        let stored_bytes = [256_usize, 64];
        assert_eq!(
            native.residency(&NVFP4_CODEC, &spec([256, 128], false), &stored_bytes),
            ResidencyMode::Packed
        );
        assert_eq!(
            native.residency(&NVFP4_CODEC, &spec([256, 48], false), &stored_bytes),
            ResidencyMode::Dense,
            "an unaligned K falls back rather than planning a GEMM that would refuse"
        );
        assert_eq!(
            native.residency(&NVFP4_CODEC, &spec([256, 128], true), &stored_bytes),
            ResidencyMode::Dense,
            "`full_precision_matrix_mult` never runs packed"
        );
        // sc-20641 review: ComfyUI pads to 16, so `in_features = 60` stores a 32-aligned K = 64.
        // Alignment alone would plan Packed and hand `Nvfp4Linear` 4 columns of padding as real
        // contraction elements; the logical-equals-stored condition is what keeps it dense.
        assert!(nvfp4_layout_is_native([256, 64]), "the PADDED grid aligns");
        assert_eq!(
            native.residency(
                &NVFP4_CODEC,
                &spec_of([256, 64], [256, 60], false),
                &[256, 32]
            ),
            ResidencyMode::Dense,
            "a padded layer must not repack, however well the padded grid aligns"
        );
        assert_eq!(
            CandleCodecResidency::DENSE.residency(
                &NVFP4_CODEC,
                &spec([256, 128], false),
                &stored_bytes
            ),
            ResidencyMode::Dense,
            "below the floor every device takes the dense fallback"
        );
        // This machine's CPU (and Metal) are below both floors.
        assert_eq!(
            CandleCodecResidency::probe(&Device::Cpu),
            CandleCodecResidency::DENSE
        );
    }

    /// **sc-20651 major 3: an undeclared block-padded layer plans but must not materialize.**
    ///
    /// NVFP4 storage is 16-padded on both axes and the file records no true geometry, so with no
    /// adapter-declared logical shape the plan can only carry the *stored* grid forward. Decoding
    /// that grid hands the model the pad rows/columns as real weights — silent corruption, not a
    /// shape error, because every downstream check that trusts the plan agrees with it.
    ///
    /// Planning stays legal: a plan is also the pricing artifact admission reads, the padded grid is
    /// a conservative over-estimate, and the memory-strategy paths that compile a plan never
    /// materialize a tensor. Materialization is where it stops, and the refusal names the tensor.
    #[test]
    fn an_undeclared_block_padded_layer_plans_but_refuses_to_materialize() {
        let (rows, cols) = (32_usize, 64_usize);
        let dir = fixture_dir();
        let path = dir.path().join("nvfp4-undeclared.safetensors");
        let packed = vec![0x11_u8; rows * cols / 2];
        let block_scales =
            vec![0x38_u8; gen_core::nvfp4_scale_shape([rows, cols]).iter().product()];
        write_safetensors_with_metadata(
            &path,
            &[
                ("model.q.weight", "U8", &[rows, cols / 2], packed),
                (
                    "model.q.weight_scale",
                    "F8_E4M3",
                    &gen_core::nvfp4_scale_shape([rows, cols]),
                    block_scales,
                ),
                (
                    "model.q.weight_scale_2",
                    "F32",
                    &[],
                    1.0_f32.to_le_bytes().to_vec(),
                ),
            ],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
            )],
        );

        // Plans — and prices — with nothing declared.
        let undeclared = plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE)
            .expect("an undeclared padded checkpoint still PLANS: pricing is not materialization");
        assert_eq!(undeclared.codec_ids(), vec!["nvfp4-v1"]);
        assert_eq!(
            undeclared.tensors[0].undeclared_padded_storage(),
            Some([rows, cols]),
            "the plan records that its logical shape is only the stored grid"
        );

        // ...but the read refuses, naming the tensor.
        let error = match read_logical_weights(&path, &undeclared, &Device::Cpu) {
            Ok(_) => panic!("materializing an undeclared padded grid must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("model.q.weight"), "{error}");
        assert!(error.contains("block-padded"), "{error}");
        assert!(error.contains("declares no logical shape"), "{error}");

        // The SAME file, with the adapter declaring the layer's true geometry, reads.
        let declared = plan_logical_weights(
            &path,
            &StripModelDeclaring(vec![rows, cols]),
            &CandleCodecResidency::DENSE,
        )
        .expect("plan");
        assert_eq!(declared.tensors[0].undeclared_padded_storage(), None);
        read_logical_weights(&path, &declared, &Device::Cpu)
            .expect("a declared layer materializes exactly as before");
    }

    /// **sc-20651 blocker 2: the fp8 packed leg is rank-2 only.**
    ///
    /// `CublasLt::matmul_fp8`'s weight-side operand is a matrix, and this module's header has
    /// always said so ("a rank-2 E4M3 weight"). The policy did not enforce it, and the plan
    /// compiler deliberately applies **no** rank constraint to *undescribed* fp8 (ComfyUI's
    /// `weight_dtype=fp8_e4m3fn` cast covers biases and modulation vectors too), so a rank-1 bias
    /// in such a checkpoint planned `Packed` on an sm_89 CUDA device — into a container the fp8 leg
    /// cannot read — while MLX dense-decoded the same tensor correctly. That is two backends
    /// disagreeing about one file.
    #[test]
    fn fp8_packed_residency_is_rank_two_only() {
        let native = CandleCodecResidency {
            fp8_e4m3_native: true,
            nvfp4_native: false,
        };
        let scalar_fp8 = TensorCodecSpec::ScalarFp8 {
            scale: ScalarScaleSource::Unit,
            input_scale: None,
            full_precision_matrix_mult: false,
        };
        assert_eq!(
            native.residency(&FP8_E4M3_SCALAR_CODEC, &scalar_fp8, &[2048, 512]),
            ResidencyMode::Packed,
            "a rank-2 projection is the packed leg's operand"
        );
        for stored in [
            vec![2048_usize],         // a bias / modulation vector — the real rank-1 case
            vec![],                   // a scalar
            vec![4_usize, 2048, 512], // a rank-3 payload
        ] {
            assert_eq!(
                native.residency(&FP8_E4M3_SCALAR_CODEC, &scalar_fp8, &stored),
                ResidencyMode::Dense,
                "stored rank {} must take the dense fallback both backends share",
                stored.len()
            );
        }
    }

    /// **sc-20651 minor: a packed fp8 receipt counts a retained scale only when one exists.**
    ///
    /// The plan prices a retained `weight_scale` on its companion row, and
    /// [`ScalarScaleSource::Unit`] — the plain undescribed cast — has no such row, because there is
    /// no tensor in the file to price. `resident_bytes` added four bytes unconditionally, so the
    /// receipt exceeded the plan by exactly `4 x (undescribed fp8 layers)` on the one pairing the
    /// two exist to cross-check.
    #[test]
    fn packed_fp8_resident_bytes_counts_only_companions_that_exist() {
        let codes = Tensor::zeros((8, 8), DType::F8E4M3, &Device::Cpu).expect("codes");
        let unit = LogicalTensor::PackedFp8E4M3 {
            codes: codes.clone(),
            weight_scale: 1.0,
            weight_scale_from_companion: false,
            input_scale: None,
        };
        let companion = LogicalTensor::PackedFp8E4M3 {
            codes: codes.clone(),
            weight_scale: 0.5,
            weight_scale_from_companion: true,
            input_scale: None,
        };
        let both = LogicalTensor::PackedFp8E4M3 {
            codes,
            weight_scale: 0.5,
            weight_scale_from_companion: true,
            input_scale: Some(0.25),
        };
        assert_eq!(
            unit.resident_bytes(),
            64,
            "the undescribed cast retains no scale ROW, so it prices none"
        );
        assert_eq!(companion.resident_bytes(), 64 + 4);
        assert_eq!(both.resident_bytes(), 64 + 4 + 4);
    }

    /// AC1 (Candle). Every NVFP4 layout defect fails closed, naming the layer.
    #[test]
    fn nvfp4_layout_defects_refuse_with_layer_specific_diagnostics() {
        let (rows, cols) = (32_usize, 64_usize);
        let values = nvfp4_fixture_values(rows, cols);
        let (packed, scales, global, _) = nvfp4_reference_quantize(&values, rows, cols);
        let scale_shape = [128_usize, 4];
        assert_eq!(scales.len(), scale_shape[0] * scale_shape[1]);

        let dir = fixture_dir();
        let case = |name: &str, tensors: &[(&str, &str, &[usize], Vec<u8>)]| -> String {
            let path = dir.path().join(format!("{name}.safetensors"));
            write_safetensors_with_metadata(
                &path,
                tensors,
                &[(
                    "_quantization_metadata",
                    r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#,
                )],
            );
            match plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE) {
                Ok(_) => panic!("{name}: must refuse"),
                Err(error) => error.to_string(),
            }
        };
        let weight = (
            "model.q.weight",
            "U8",
            &[rows, cols / 2][..],
            packed.clone(),
        );
        let block = (
            "model.q.weight_scale",
            "F8_E4M3",
            &scale_shape[..],
            scales.clone(),
        );
        let global_row = (
            "model.q.weight_scale_2",
            "F32",
            &[][..],
            global.to_le_bytes().to_vec(),
        );

        // A missing block scale, and a missing global scale — each named as the absent companion.
        let error = case("no-block-scale", &[weight.clone(), global_row.clone()]);
        assert!(
            error.contains("model.q.weight") && error.contains("weight_scale"),
            "{error}"
        );
        let error = case("no-global-scale", &[weight.clone(), block.clone()]);
        assert!(
            error.contains("model.q.weight_scale_2"),
            "a missing second scale level must name it: {error}"
        );
        // A mis-shaped block-scale surface (un-swizzled `[rows, blocks]` instead of the padded
        // swizzle) — the exact expected shape is named.
        let error = case(
            "unswizzled-scale",
            &[
                weight.clone(),
                (
                    "model.q.weight_scale",
                    "F8_E4M3",
                    &[rows, cols / 16],
                    scales[..rows * (cols / 16)].to_vec(),
                ),
                global_row.clone(),
            ],
        );
        assert!(
            error.contains("model.q.weight") && error.contains("[128, 4]"),
            "{error}"
        );
        // A global scale that is not a scalar F32.
        let error = case(
            "global-not-scalar",
            &[
                weight.clone(),
                block.clone(),
                ("model.q.weight_scale_2", "F32", &[4], vec![0_u8; 16]),
            ],
        );
        assert!(
            error.contains("model.q.weight_scale_2") && error.contains("scalar F32"),
            "{error}"
        );
        // Truncated nibbles: the byte matrix does not match its declared shape.
        let error = case(
            "truncated-nibbles",
            &[
                (
                    "model.q.weight",
                    "U8",
                    &[rows, cols / 2],
                    packed[..packed.len() - 8].to_vec(),
                ),
                block.clone(),
                global_row.clone(),
            ],
        );
        assert!(error.contains("model.q.weight"), "{error}");
        // Invalid padding: a logical column count that is not a multiple of 16.
        let error = case(
            "unpadded",
            &[
                ("model.q.weight", "U8", &[rows, 20], vec![0_u8; rows * 20]),
                (
                    "model.q.weight_scale",
                    "F8_E4M3",
                    &[128, 4],
                    vec![0_u8; 512],
                ),
                global_row.clone(),
            ],
        );
        assert!(
            error.contains("model.q.weight") && error.contains("multiple of 16"),
            "{error}"
        );
        // A shape mismatch between the weight and its block scales: a weight twice as wide still
        // has a valid 16-padded geometry, but needs 8 blocks per row, not 4.
        let error = case(
            "shape-mismatch",
            &[
                (
                    "model.q.weight",
                    "U8",
                    &[rows, cols],
                    vec![0_u8; rows * cols],
                ),
                block.clone(),
                global_row.clone(),
            ],
        );
        assert!(
            error.contains("model.q.weight") && error.contains("weight_scale"),
            "{error}"
        );
        // A `_quantization_metadata` layer name that matches nothing refuses rather than being
        // silently ignored.
        let path = dir.path().join("unmatched.safetensors");
        write_safetensors_with_metadata(
            &path,
            &[weight.clone(), block.clone(), global_row.clone()],
            &[(
                "_quantization_metadata",
                r#"{"format_version": "1.0", "layers": {"absent": {"format": "nvfp4"}}}"#,
            )],
        );
        let error = plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE)
            .expect_err("an unmatched declaration must refuse")
            .to_string();
        assert!(error.contains("absent"), "{error}");
    }

    #[test]
    fn descriptor_defects_and_source_drift_refuse_by_tensor() {
        let dir = fixture_dir();
        let path = dir.path().join("bad.safetensors");
        let descriptor = br#"{"format": "int4_awq"}"#;
        write_safetensors(
            &path,
            &[
                ("model.q.weight", "F8_E4M3", &[1, 2], vec![0x38, 0x40]),
                (
                    "model.q.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let error = plan_logical_weights(&path, &StripModel, &CandleCodecResidency::DENSE)
            .expect_err("nvfp4 must refuse")
            .to_string();
        assert!(
            error.contains("\"model.q\"") && error.contains("nvfp4"),
            "{error}"
        );

        // Drift between planning and reading refuses.
        let drift = dir.path().join("drift.safetensors");
        write_safetensors(
            &drift,
            &[("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0]))],
        );
        let plan = plan_dense(&drift);
        write_safetensors(
            &drift,
            &[
                ("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0])),
                ("model.extra", "BF16", &[1], bf16_bytes(&[3.0])),
            ],
        );
        let error = match read_logical_weights(&drift, &plan, &Device::Cpu) {
            Ok(_) => panic!("drift must refuse"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("tensor set changed since planning"),
            "{error}"
        );
    }
}
