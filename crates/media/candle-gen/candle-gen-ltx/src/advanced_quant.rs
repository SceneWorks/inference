//! Descriptor-driven, fail-closed LTX-2.5 advanced projection materialization.
//!
//! INT8-ConvRot consumes stored I8 `W*R` codes, per-row `weight_scale`, and the declared Hadamard
//! group, then executes activation rotation plus cuBLASLt IGEMM. NVFP4 consumes the checkpoint's
//! declared Kitchen NVFP4 triplet through the shared logical reader and requires a native W4A4
//! context. Neither arm has a dense fallback.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::gen_core::IdentityKeyMapping;
use candle_gen::logical_weights::{
    plan_logical_weights, CandleCodecResidency, LogicalTensor, LogicalWeightReader,
};
use candle_gen::quant::{Int8Context, Nvfp4Context, Nvfp4Tensor};
use safetensors::Dtype as SafetensorDtype;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::quant_eval::Ltx25QuantMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdvancedOperatorKind {
    Dense,
    MlxAffine,
    Int8ConvRotIgemm,
    Nvfp4W4A4,
}

impl AdvancedOperatorKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Dense => "dense-linear",
            Self::MlxAffine => "mlx-affine-dequant",
            Self::Int8ConvRotIgemm => "int8-convrot-rht-cublaslt-igemm",
            Self::Nvfp4W4A4 => "native-nvfp4-cublaslt-w4a4",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformerSourceInspection {
    pub mode: Ltx25QuantMode,
    pub projection_keys: BTreeSet<String>,
    pub operator_contract_sha256: String,
}

fn sha256(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        let bytes = part.as_ref();
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

fn projection_contract(mode: Ltx25QuantMode, keys: &BTreeSet<String>) -> String {
    sha256(
        std::iter::once(mode.id().as_bytes().to_vec())
            .chain(keys.iter().map(|key| key.as_bytes().to_vec())),
    )
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Msg(message.into())
}

/// Inspect the actual transformer header/descriptors. File names are deliberately irrelevant.
pub fn inspect_transformer_source(
    path: &Path,
    mode: Ltx25QuantMode,
) -> Result<TransformerSourceInspection> {
    // SAFETY: standard read-only safetensors mmap; no tensor is materialized by this inspection.
    let st = unsafe { MmapedSafetensors::new(path)? };
    let tensors = st.tensors();
    let has_mlx = tensors.iter().any(|(name, _)| name.ends_with(".scales"));
    let has_i8 = tensors
        .iter()
        .any(|(name, view)| name.ends_with(".weight") && view.dtype() == SafetensorDtype::I8);
    let quantization_metadata = candle_gen::gen_core::safetensors_path_quantization_metadata(path)
        .map_err(|error| invalid(error.to_string()))?;

    let projection_keys = match mode {
        Ltx25QuantMode::Bf16 => {
            if has_mlx || has_i8 || quantization_metadata.is_some() {
                return Err(invalid(
                    "LTX bf16 transformer carries packed/descriptor quantization; refusing a mislabeled baseline",
                ));
            }
            BTreeSet::new()
        }
        Ltx25QuantMode::Q4 | Ltx25QuantMode::PackedQ8 => {
            if has_i8 || quantization_metadata.is_some() {
                return Err(invalid(
                    "LTX MLX-affine source mixes native descriptor quantization; refusing ambiguous operators",
                ));
            }
            let expected_bits = if mode == Ltx25QuantMode::Q4 { 4 } else { 8 };
            let mut keys = BTreeSet::new();
            for (scale_name, scales) in tensors.iter().filter(|(name, _)| name.ends_with(".scales"))
            {
                let base = scale_name.strip_suffix(".scales").unwrap();
                let weight_name = format!("{base}.weight");
                let biases_name = format!("{base}.biases");
                let weight = st.get(&weight_name).map_err(|_| {
                    invalid(format!(
                        "packed projection `{base}` is missing `{weight_name}`"
                    ))
                })?;
                if weight.dtype() != SafetensorDtype::U32 || scales.shape().len() != 2 {
                    return Err(invalid(format!(
                        "packed projection `{base}` must have U32 rank-2 codes and rank-2 scales"
                    )));
                }
                let biases = st.get(&biases_name).map_err(|_| {
                    invalid(format!(
                        "packed projection `{base}` is missing `{biases_name}`"
                    ))
                })?;
                if biases.shape() != scales.shape() {
                    return Err(invalid(format!(
                        "packed projection `{base}` scales/biases shapes differ"
                    )));
                }
                let logical_cols = scales.shape()[1] * crate::quant::GROUP_SIZE;
                let bits = weight.shape()[1]
                    .checked_mul(32)
                    .and_then(|stored_bits| stored_bits.checked_div(logical_cols))
                    .unwrap_or(0);
                if bits != expected_bits {
                    return Err(invalid(format!(
                        "packed projection `{base}` is {bits}-bit by its stored shapes, expected {expected_bits}-bit"
                    )));
                }
                keys.insert(base.to_owned());
            }
            if keys.is_empty() {
                return Err(invalid(format!(
                    "LTX {} transformer contains no validated MLX-affine projection triples",
                    mode.id()
                )));
            }
            keys
        }
        Ltx25QuantMode::Int8ConvRot => inspect_convrot(&st)?,
        Ltx25QuantMode::Nvfp4 => {
            let plan = plan_logical_weights(
                path,
                &IdentityKeyMapping,
                &CandleCodecResidency {
                    fp8_e4m3_native: false,
                    nvfp4_native: true,
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
            let keys: BTreeSet<String> = plan
                .tensors
                .iter()
                .filter(|tensor| tensor.codec_id == "nvfp4-v1")
                .map(|tensor| {
                    tensor
                        .logical_key
                        .strip_suffix(".weight")
                        .ok_or_else(|| {
                            invalid(format!(
                                "NVFP4 codec row `{}` is not a projection weight",
                                tensor.logical_key
                            ))
                        })
                        .map(str::to_owned)
                })
                .collect::<Result<_>>()?;
            if keys.is_empty() {
                return Err(invalid(
                    "LTX NVFP4 transformer has no descriptor-declared nvfp4-v1 projection",
                ));
            }
            if has_mlx || has_i8 {
                return Err(invalid(
                    "LTX NVFP4 source mixes MLX-affine or I8 weights; refusing ambiguous operators",
                ));
            }
            keys
        }
    };
    Ok(TransformerSourceInspection {
        mode,
        operator_contract_sha256: projection_contract(mode, &projection_keys),
        projection_keys,
    })
}

fn inspect_convrot(st: &MmapedSafetensors) -> Result<BTreeSet<String>> {
    let mut keys = BTreeSet::new();
    for (descriptor_name, descriptor) in st
        .tensors()
        .into_iter()
        .filter(|(name, _)| name.ends_with(".comfy_quant"))
    {
        if descriptor.dtype() != SafetensorDtype::U8 || descriptor.shape().len() != 1 {
            return Err(invalid(format!(
                "ConvRot descriptor `{descriptor_name}` must be a rank-1 U8 JSON tensor"
            )));
        }
        let json: Value = serde_json::from_slice(descriptor.data()).map_err(|error| {
            invalid(format!(
                "ConvRot descriptor `{descriptor_name}` is invalid JSON: {error}"
            ))
        })?;
        if json.get("format").and_then(Value::as_str) != Some("int8_tensorwise")
            || json.get("convrot").and_then(Value::as_bool) != Some(true)
        {
            return Err(invalid(format!(
                "ConvRot descriptor `{descriptor_name}` must declare format=int8_tensorwise and convrot=true"
            )));
        }
        let group = json
            .get("convrot_groupsize")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                invalid(format!(
                    "ConvRot descriptor `{descriptor_name}` is missing convrot_groupsize"
                ))
            })? as usize;
        let base = descriptor_name.strip_suffix(".comfy_quant").unwrap();
        let weight = st.get(&format!("{base}.weight"))?;
        let [rows, cols] = weight.shape() else {
            return Err(invalid(format!("ConvRot `{base}.weight` must be rank 2")));
        };
        if weight.dtype() != SafetensorDtype::I8
            || !candle_gen::quant::is_power_of_four(group)
            || !cols.is_multiple_of(group)
        {
            return Err(invalid(format!(
                "ConvRot `{base}` requires I8 codes and a power-of-four group dividing K"
            )));
        }
        let scale = st.get(&format!("{base}.weight_scale"))?;
        if scale.dtype() != SafetensorDtype::F32
            || (scale.shape() != [*rows] && scale.shape() != [*rows, 1])
        {
            return Err(invalid(format!(
                "ConvRot `{base}.weight_scale` must be F32 [{rows}] or [{rows},1]"
            )));
        }
        keys.insert(base.to_owned());
    }
    if keys.is_empty() {
        return Err(invalid(
            "LTX INT8-ConvRot transformer has no validated convrot=true projection descriptors",
        ));
    }
    for (weight_name, _) in st
        .tensors()
        .into_iter()
        .filter(|(name, view)| name.ends_with(".weight") && view.dtype() == SafetensorDtype::I8)
    {
        let base = weight_name.strip_suffix(".weight").unwrap();
        if !keys.contains(base) {
            return Err(invalid(format!(
                "LTX INT8-ConvRot source contains orphan I8 projection `{weight_name}` without a validated convrot descriptor/scale"
            )));
        }
    }
    if st
        .tensors()
        .iter()
        .any(|(name, _)| name.ends_with(".scales"))
    {
        return Err(invalid(
            "LTX INT8-ConvRot source mixes MLX-affine triples; refusing ambiguous operators",
        ));
    }
    Ok(keys)
}

enum AdvancedBackend {
    ConvRot {
        st: MmapedSafetensors,
        context: Arc<Int8Context>,
    },
    Nvfp4 {
        reader: Box<LogicalWeightReader>,
        context: Arc<Nvfp4Context>,
    },
}

pub(crate) struct AdvancedQuantSource {
    inspection: TransformerSourceInspection,
    backend: AdvancedBackend,
}

impl AdvancedQuantSource {
    pub fn open(path: &Path, mode: Ltx25QuantMode, device: &Device) -> Result<Arc<Self>> {
        if !device.is_cuda() {
            return Err(invalid(format!(
                "LTX {} materialization requires CUDA; dense CPU fallback is forbidden",
                mode.id()
            )));
        }
        let inspection = inspect_transformer_source(path, mode)?;
        let backend = match mode {
            Ltx25QuantMode::Int8ConvRot => {
                let context = Arc::new(Int8Context::new(device)?);
                if !context.is_int8() {
                    return Err(invalid(
                        "LTX INT8-ConvRot could not construct a live cuBLASLt IGEMM context",
                    ));
                }
                // SAFETY: standard read-only checkpoint mmap.
                let st = unsafe { MmapedSafetensors::new(path)? };
                AdvancedBackend::ConvRot { st, context }
            }
            Ltx25QuantMode::Nvfp4 => {
                let residency = CandleCodecResidency {
                    fp8_e4m3_native: false,
                    nvfp4_native: true,
                };
                let plan = plan_logical_weights(path, &IdentityKeyMapping, &residency)
                    .map_err(|error| invalid(error.to_string()))?;
                let reader = LogicalWeightReader::open_with_capability(
                    path,
                    plan,
                    device,
                    residency.native_execution_capability(),
                )
                .map_err(|error| invalid(error.to_string()))?;
                let context = Arc::new(Nvfp4Context::new(device)?);
                if !context.is_fp4() || !context.fused_quantizer_available() {
                    return Err(invalid(
                        "LTX NVFP4 requires consumer Blackwell sm_120 plus the fused native FP4 quantizer; BF16 fallback is forbidden",
                    ));
                }
                AdvancedBackend::Nvfp4 {
                    reader: Box::new(reader),
                    context,
                }
            }
            _ => {
                return Err(invalid(format!(
                    "{} is not an advanced native LTX projection source",
                    mode.id()
                )))
            }
        };
        declare_advanced_source(mode, &inspection.projection_keys);
        Ok(Arc::new(Self {
            inspection,
            backend,
        }))
    }

    pub fn is_advanced_projection(&self, base: &str) -> bool {
        self.inspection.projection_keys.contains(base)
    }

    /// A tensor that structurally looks advanced but was absent from the descriptor-derived set is
    /// corruption, not permission to feed its bytes to the dense loader.
    pub fn refuse_undeclared_advanced_tensor(&self, base: &str) -> Result<()> {
        match &self.backend {
            AdvancedBackend::ConvRot { st, .. } => {
                refuse_undeclared_convrot_tensor(st, base)?;
            }
            AdvancedBackend::Nvfp4 { reader, .. } => {
                if reader
                    .planned(&format!("{base}.weight"))
                    .is_some_and(|plan| plan.codec_id == "nvfp4-v1")
                {
                    return Err(invalid(format!(
                        "LTX `{base}` is planned NVFP4 but was not selected for the native operator"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn load_projection(
        &self,
        base: &str,
        _device: &Device,
    ) -> Result<LoadedAdvancedProjection> {
        match &self.backend {
            AdvancedBackend::ConvRot { st, context } => {
                let descriptor_name = format!("{base}.comfy_quant");
                let descriptor = st.get(&descriptor_name)?;
                let json: Value = serde_json::from_slice(descriptor.data())
                    .map_err(|error| invalid(error.to_string()))?;
                let group_size = json["convrot_groupsize"]
                    .as_u64()
                    .ok_or_else(|| invalid(format!("{descriptor_name} lost convrot_groupsize")))?
                    as usize;
                let weight_name = format!("{base}.weight");
                let weight = st.get(&weight_name)?;
                if weight.dtype() != SafetensorDtype::I8 {
                    return Err(invalid(format!(
                        "ConvRot `{weight_name}` changed dtype after validation"
                    )));
                }
                let codes = Tensor::from_vec(
                    weight
                        .data()
                        .iter()
                        .map(|byte| *byte as i8 as i64)
                        .collect::<Vec<_>>(),
                    weight.shape().to_vec(),
                    &Device::Cpu,
                )?;
                let scale_name = format!("{base}.weight_scale");
                let scale_view = st.get(&scale_name)?;
                let scale = st
                    .load(&scale_name, &Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let attestation = sha256([
                    base.as_bytes(),
                    weight.data(),
                    scale_view.data(),
                    descriptor.data(),
                ]);
                Ok(LoadedAdvancedProjection::Int8ConvRot {
                    codes,
                    scale,
                    group_size,
                    context: Arc::clone(context),
                    attestation,
                })
            }
            AdvancedBackend::Nvfp4 { reader, context } => {
                let weight_key = format!("{base}.weight");
                let LogicalTensor::PackedNvfp4 {
                    tensor,
                    input_scale,
                } = reader
                    .read(&weight_key)
                    .map_err(|error| invalid(error.to_string()))?
                else {
                    return Err(invalid(format!(
                        "LTX `{weight_key}` did not materialize PackedNvfp4; dense fallback refused"
                    )));
                };
                if input_scale.is_some() {
                    return Err(invalid(format!(
                        "LTX `{weight_key}` carries an unsupported input_scale; native operator contract is not implemented"
                    )));
                }
                let packed = *tensor;
                let global = packed.global_scale.to_bits().to_le_bytes();
                let attestation = sha256([
                    base.as_bytes(),
                    packed.packed.as_slice(),
                    packed.scales.as_slice(),
                    global.as_slice(),
                ]);
                Ok(LoadedAdvancedProjection::Nvfp4 {
                    packed,
                    context: Arc::clone(context),
                    attestation,
                })
            }
        }
    }
}

fn refuse_undeclared_convrot_tensor(st: &MmapedSafetensors, base: &str) -> Result<()> {
    let is_i8_weight = st
        .get(&format!("{base}.weight"))
        .is_ok_and(|view| view.dtype() == SafetensorDtype::I8);
    if is_i8_weight
        || st.get(&format!("{base}.weight_scale")).is_ok()
        || st.get(&format!("{base}.comfy_quant")).is_ok()
    {
        return Err(invalid(format!(
            "LTX `{base}` has I8/ConvRot tensors but no validated convrot descriptor; dense cast/fallback refused"
        )));
    }
    Ok(())
}

pub(crate) enum LoadedAdvancedProjection {
    Int8ConvRot {
        codes: Tensor,
        scale: Vec<f32>,
        group_size: usize,
        context: Arc<Int8Context>,
        attestation: String,
    },
    Nvfp4 {
        packed: Nvfp4Tensor,
        context: Arc<Nvfp4Context>,
        attestation: String,
    },
}

thread_local! {
    static ACTIVE_SOURCE: RefCell<Vec<Arc<AdvancedQuantSource>>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn with_advanced_source<T>(
    source: Option<Arc<AdvancedQuantSource>>,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(source) = source else {
        return f();
    };
    ACTIVE_SOURCE.with(|active| active.borrow_mut().push(source));
    let result = f();
    ACTIVE_SOURCE.with(|active| {
        active.borrow_mut().pop();
    });
    result
}

pub(crate) fn active_source() -> Option<Arc<AdvancedQuantSource>> {
    ACTIVE_SOURCE.with(|active| active.borrow().last().cloned())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorAttestation {
    pub mode: Ltx25QuantMode,
    pub operator_kind: String,
    pub executed_projection_count: u32,
    pub declared_projection_count: u32,
    pub weight_inventory_sha256: String,
}

#[derive(Default)]
struct AttestationState {
    mode: Option<Ltx25QuantMode>,
    declared: BTreeSet<String>,
    executed: BTreeMap<String, (AdvancedOperatorKind, String)>,
}

thread_local! {
    static OPERATOR_ATTESTATION: RefCell<AttestationState> = RefCell::new(AttestationState::default());
}

pub fn begin_operator_attestation(mode: Ltx25QuantMode) {
    OPERATOR_ATTESTATION.with(|state| {
        *state.borrow_mut() = AttestationState {
            mode: Some(mode),
            ..AttestationState::default()
        };
    });
}

fn declare_advanced_source(mode: Ltx25QuantMode, keys: &BTreeSet<String>) {
    OPERATOR_ATTESTATION.with(|state| {
        let mut state = state.borrow_mut();
        if state.mode == Some(mode) {
            state.declared.extend(keys.iter().cloned());
        }
    });
}

pub(crate) fn record_projection_execution(
    key: String,
    kind: AdvancedOperatorKind,
    attestation: String,
) {
    OPERATOR_ATTESTATION.with(|state| {
        let mut state = state.borrow_mut();
        if state.mode.is_some() {
            state.executed.insert(key, (kind, attestation));
        }
    });
}

pub fn finish_operator_attestation() -> Result<OperatorAttestation> {
    OPERATOR_ATTESTATION.with(|state| finish_attestation(&state.borrow()))
}

fn finish_attestation(state: &AttestationState) -> Result<OperatorAttestation> {
    let mode = state
        .mode
        .ok_or_else(|| invalid("operator attestation was not started"))?;
    if matches!(mode, Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4)
        && (state.declared.is_empty()
            || state
                .declared
                .iter()
                .any(|key| !state.executed.contains_key(key)))
    {
        return Err(invalid(format!(
            "LTX {} did not execute every descriptor-declared advanced projection; dense fallback/replay refused",
            mode.id()
        )));
    }
    if state.executed.is_empty() {
        return Err(invalid("LTX generation executed no attested projections"));
    }
    let expected = match mode {
        Ltx25QuantMode::Bf16 => AdvancedOperatorKind::Dense,
        Ltx25QuantMode::Q4 | Ltx25QuantMode::PackedQ8 => AdvancedOperatorKind::MlxAffine,
        Ltx25QuantMode::Int8ConvRot => AdvancedOperatorKind::Int8ConvRotIgemm,
        Ltx25QuantMode::Nvfp4 => AdvancedOperatorKind::Nvfp4W4A4,
    };
    if !state.executed.values().any(|(kind, _)| *kind == expected) {
        return Err(invalid(format!(
            "LTX {} never executed its required operator {}",
            mode.id(),
            expected.id()
        )));
    }
    if matches!(mode, Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4)
        && state.declared.iter().any(|key| {
            state
                .executed
                .get(key)
                .is_some_and(|(kind, _)| *kind != expected)
        })
    {
        return Err(invalid(
            "descriptor-declared advanced projection executed a different operator",
        ));
    }
    let inventory = sha256(state.executed.iter().flat_map(|(key, (kind, hash))| {
        [
            key.as_bytes().to_vec(),
            kind.id().as_bytes().to_vec(),
            hash.as_bytes().to_vec(),
        ]
    }));
    Ok(OperatorAttestation {
        mode,
        operator_kind: expected.id().to_owned(),
        executed_projection_count: state.executed.len() as u32,
        declared_projection_count: state.declared.len() as u32,
        weight_inventory_sha256: inventory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn write_convrot(path: &Path) {
        write_convrot_with_orphan(path, false);
    }

    fn write_convrot_with_orphan(path: &Path, include_orphan: bool) {
        let codes = vec![0u8; 16];
        let scales = [1.0f32; 4]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let descriptor = br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":4}"#;
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.diffusion_model.block.to_q.weight",
            safetensors::tensor::TensorView::new(SafetensorDtype::I8, vec![4, 4], &codes).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.block.to_q.weight_scale",
            safetensors::tensor::TensorView::new(SafetensorDtype::F32, vec![4], &scales).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.block.to_q.comfy_quant",
            safetensors::tensor::TensorView::new(
                SafetensorDtype::U8,
                vec![descriptor.len()],
                descriptor,
            )
            .unwrap(),
        );
        if include_orphan {
            tensors.insert(
                "model.diffusion_model.block.to_k.weight",
                safetensors::tensor::TensorView::new(SafetensorDtype::I8, vec![4, 4], &codes)
                    .unwrap(),
            );
        }
        safetensors::serialize_to_file(tensors, None, path).unwrap();
    }

    fn write_nvfp4(path: &Path) {
        let (rows, cols) = (128usize, 32usize);
        let packed = vec![0u8; rows * cols / 2];
        let scale_shape = candle_gen::gen_core::nvfp4_scale_shape([rows, cols]).to_vec();
        let scales = vec![0x38u8; scale_shape.iter().product()];
        let global = 1.0f32.to_le_bytes();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.diffusion_model.block.to_q.weight",
            safetensors::tensor::TensorView::new(
                SafetensorDtype::U8,
                vec![rows, cols / 2],
                &packed,
            )
            .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.block.to_q.weight_scale",
            safetensors::tensor::TensorView::new(SafetensorDtype::F8_E4M3, scale_shape, &scales)
                .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.block.to_q.weight_scale_2",
            safetensors::tensor::TensorView::new(SafetensorDtype::F32, vec![], &global).unwrap(),
        );
        let metadata = HashMap::from([(
            "_quantization_metadata".to_owned(),
            r#"{"format_version":"1.0","layers":{"model.diffusion_model.block.to_q":{"format":"nvfp4"}}}"#
                .to_owned(),
        )]);
        safetensors::serialize_to_file(tensors, Some(metadata), path).unwrap();
    }

    #[test]
    fn advanced_operator_identities_are_not_dense_or_mlx_aliases() {
        let identities = [
            AdvancedOperatorKind::Dense.id(),
            AdvancedOperatorKind::MlxAffine.id(),
            AdvancedOperatorKind::Int8ConvRotIgemm.id(),
            AdvancedOperatorKind::Nvfp4W4A4.id(),
        ];
        let unique: BTreeSet<_> = identities.into_iter().collect();
        assert_eq!(unique.len(), identities.len());
        assert!(AdvancedOperatorKind::Int8ConvRotIgemm
            .id()
            .contains("rht-cublaslt-igemm"));
        assert!(AdvancedOperatorKind::Nvfp4W4A4
            .id()
            .contains("native-nvfp4-cublaslt-w4a4"));
    }

    #[test]
    fn semantic_source_guards_distinguish_convrot_nvfp4_and_mlx_labels() {
        let dir = tempfile::tempdir().unwrap();
        let convrot = dir.path().join("ordinary-name.safetensors");
        let nvfp4 = dir.path().join("also-ordinary.safetensors");
        write_convrot(&convrot);
        write_nvfp4(&nvfp4);

        let convrot_inspection =
            inspect_transformer_source(&convrot, Ltx25QuantMode::Int8ConvRot).unwrap();
        let nvfp4_inspection = inspect_transformer_source(&nvfp4, Ltx25QuantMode::Nvfp4).unwrap();
        assert_ne!(
            convrot_inspection.operator_contract_sha256,
            nvfp4_inspection.operator_contract_sha256
        );
        assert!(inspect_transformer_source(&convrot, Ltx25QuantMode::PackedQ8).is_err());
        assert!(inspect_transformer_source(&nvfp4, Ltx25QuantMode::Int8ConvRot).is_err());
    }

    #[test]
    fn a_dense_weight_cannot_satisfy_an_advanced_source_by_filename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pretend-int8-convrot-nvfp4.safetensors");
        let dense = vec![0u8; 4 * 4 * 4];
        let tensors = BTreeMap::from([(
            "model.diffusion_model.block.to_q.weight",
            safetensors::tensor::TensorView::new(SafetensorDtype::F32, vec![4, 4], &dense).unwrap(),
        )]);
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        assert!(inspect_transformer_source(&path, Ltx25QuantMode::Int8ConvRot).is_err());
        assert!(inspect_transformer_source(&path, Ltx25QuantMode::Nvfp4).is_err());
    }

    #[test]
    fn convrot_source_rejects_an_orphan_i8_projection_before_dense_loading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orphan-i8.safetensors");
        write_convrot_with_orphan(&path, true);
        let error = inspect_transformer_source(&path, Ltx25QuantMode::Int8ConvRot)
            .unwrap_err()
            .to_string();
        assert!(error.contains("orphan I8 projection"), "{error}");
        assert!(error.contains("to_k.weight"), "{error}");
        // SAFETY: test-owned immutable safetensors file.
        let st = unsafe { MmapedSafetensors::new(&path).unwrap() };
        let fallback = refuse_undeclared_convrot_tensor(&st, "model.diffusion_model.block.to_k")
            .unwrap_err()
            .to_string();
        assert!(
            fallback.contains("dense cast/fallback refused"),
            "{fallback}"
        );
    }

    #[test]
    fn missing_advanced_materialization_is_a_hard_error() {
        begin_operator_attestation(Ltx25QuantMode::Int8ConvRot);
        declare_advanced_source(
            Ltx25QuantMode::Int8ConvRot,
            &BTreeSet::from(["model.diffusion_model.block.to_q".to_owned()]),
        );
        let error = finish_operator_attestation().unwrap_err().to_string();
        assert!(error.contains("did not execute every"), "{error}");
    }

    #[test]
    fn advanced_attestation_refuses_a_dense_operator_for_declared_weight() {
        begin_operator_attestation(Ltx25QuantMode::Nvfp4);
        let key = "model.diffusion_model.block.to_q".to_owned();
        declare_advanced_source(Ltx25QuantMode::Nvfp4, &BTreeSet::from([key.clone()]));
        record_projection_execution(key, AdvancedOperatorKind::Dense, "a".repeat(64));
        let error = finish_operator_attestation().unwrap_err().to_string();
        assert!(error.contains("required operator"), "{error}");
    }
}
