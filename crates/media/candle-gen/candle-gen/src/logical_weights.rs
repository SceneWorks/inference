//! Candle **mapped logical-weight reader** and the engine's **codec table** (epic 20398,
//! sc-20385) — the Candle twin of `mlx_gen::logical_weights`.
//!
//! The provider supplies its adapter-owned [`LogicalKeyMapping`]; [`plan_logical_weights`]
//! compiles the safetensors header plus the `.comfy_quant` descriptor payloads into a
//! [`LogicalWeightPlan`] against [`baseline_codec_registry`] and this backend's
//! [`CandleFp8Residency`] policy; [`read_logical_weights`] then materializes exactly the planned
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
//! * **Native path** (`fp8-e4m3-scalar-v1` only): where the layout + hardware contract of the
//!   cuBLASLt fp8 leg holds — a CUDA device at the sm_89 floor
//!   (`CublasLt::meets_fp8_floor`, via [`crate::quant::FP8_COMPUTE_CAP_FLOOR`]; locked
//!   decision 7), a rank-2 E4M3 weight, and no `full_precision_matrix_mult` flag — the
//!   [`CandleFp8Residency`] policy plans `Packed` residency and the reader keeps the stored
//!   `F8E4M3` codes + f32 scale resident ([`LogicalTensor::PackedFp8E4M3`]), the exact operands
//!   `CublasLt::matmul_fp8` consumes. E5M2 has no weight-side GEMM leg here and MXFP8 has no
//!   block-scaled kernel in this workspace, so both always take the dense fallback. On a build
//!   without the `cuda` feature a `Packed` plan entry is a typed refusal — never a silent
//!   dense substitution of a plan that admission priced as packed.
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
    FP8_E5M2_SCALAR_CODEC, INT8_PER_ROW_CODEC, MXFP8_CODEC,
};
use gen_core::ProviderRegistryBuilder;

use crate::candle_core::safetensors::MmapedSafetensors;
use crate::candle_core::{DType, Device, Tensor};
use crate::{CandleError, Result};

/// Descriptor payloads are tiny JSON blobs; anything above this is not a `.comfy_quant` tensor.
const MAX_DESCRIPTOR_BYTES: u64 = 65_536;

/// The codec rows this engine implements and registers — identical to the MLX table (codecs are
/// backend-portable declarations; each engine owns its implementation).
pub const BASELINE_CODECS: &[CheckpointCodecRegistration] = &[
    DENSE_BF16_CODEC,
    DENSE_F16_CODEC,
    DENSE_F32_CODEC,
    FP8_E4M3_SCALAR_CODEC,
    FP8_E5M2_SCALAR_CODEC,
    MXFP8_CODEC,
    INT8_PER_ROW_CODEC,
];

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

/// Candle's packed-vs-dense residency decision. Layout facts (rank-2 E4M3 weight, no
/// `full_precision_matrix_mult`) are enforced by the plan compiler; the hardware fact — a CUDA
/// device at the cuBLASLt fp8 leg's sm_89 floor — is probed once per device and carried here, so
/// the policy itself is a pure, unit-testable predicate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandleFp8Residency {
    /// The bound device runs the cuBLASLt E4M3 GEMM (CUDA, compute capability ≥ 8.9).
    pub fp8_e4m3_native: bool,
}

impl CandleFp8Residency {
    /// The dense-only policy (CPU, Metal, or CUDA below the sm_89 floor).
    pub const DENSE: Self = Self {
        fp8_e4m3_native: false,
    };

    /// Probe the device's fp8 eligibility. Non-CUDA devices (and every build without the `cuda`
    /// feature) are dense-only; a CUDA device whose capability cannot be read is treated as below
    /// the floor (dense fallback — the safe direction, never the silent-packed one).
    pub fn probe(device: &Device) -> Self {
        Self {
            fp8_e4m3_native: cuda_meets_sm89_floor(device),
        }
    }
}

impl CodecResidencyPolicy for CandleFp8Residency {
    fn residency(
        &self,
        codec: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        _stored_shape: &[usize],
    ) -> ResidencyMode {
        // The scalar E4M3 row is the only codec with a native leg in this workspace. The plain
        // undescribed cast (unit scale) qualifies too: `matmul_fp8` takes any f32 weight scale.
        // (The compiler already forces Dense for `full_precision_matrix_mult` layers; the match
        // here keeps this predicate honest if that ever moved.)
        if self.fp8_e4m3_native
            && codec.codec_id == FP8_E4M3_SCALAR_CODEC.codec_id
            && !spec.full_precision_matrix_mult()
        {
            ResidencyMode::Packed
        } else {
            ResidencyMode::Dense
        }
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
    gen_core::compile_logical_weight_plan(
        &headers,
        &descriptors,
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
                codes, input_scale, ..
            } => {
                (codes.elem_count() * codes.dtype().size_in_bytes()) as u64
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
            let weight_scale = match scale {
                ScalarScaleSource::Unit => 1.0_f32,
                ScalarScaleSource::Companion { physical_key } => {
                    scalar_f32(companion_bytes(st, tensor, physical_key)?, physical_key)?
                }
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
                            input_scale,
                        })
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = input_scale;
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
    /// threshold. `CublasLt::meets_fp8_floor` and `CandleFp8Residency::probe`'s device predicate
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
            LogicalTensor::PackedFp8E4M3 { .. } => panic!("{key} unexpectedly packed"),
        }
    }

    fn plan_dense(path: &Path) -> LogicalWeightPlan {
        plan_logical_weights(path, &StripModel, &CandleFp8Residency::DENSE).expect("plan")
    }

    #[test]
    fn catalog_registration_carries_the_seven_row_table() {
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
        let plan = plan_logical_weights(&path, &DeclaredShape, &CandleFp8Residency::DENSE)
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
        let native = CandleFp8Residency {
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
            CandleFp8Residency::probe(&Device::Cpu),
            CandleFp8Residency::DENSE
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
                input_scale: Some(0.5),
            }
            .resident_bytes(),
            8 + 4 + 4,
            "a retained input_scale is priced too"
        );
    }

    #[test]
    fn descriptor_defects_and_source_drift_refuse_by_tensor() {
        let dir = fixture_dir();
        let path = dir.path().join("bad.safetensors");
        let descriptor = br#"{"format": "nvfp4"}"#;
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
        let error = plan_logical_weights(&path, &StripModel, &CandleFp8Residency::DENSE)
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
