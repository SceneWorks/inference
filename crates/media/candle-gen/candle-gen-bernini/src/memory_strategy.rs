//! Bernini Candle request-scoped memory contract.
//!
//! The CUDA provider has a real tiled Wan z16 VAE decode seam, so Resident and
//! BoundedDecode are declared. Older unconditional phase staging is not a
//! request-selected StagedResidency lever, so that rung remains Missing. Bernini has no Candle
//! deferred transformer loader or attention-chunking seam; those rungs remain
//! Missing rather than inheriting the MLX claims. Production calibration is
//! intentionally absent until the Windows/CUDA real-weight campaign exists.

use candle_gen::candle_core::Device;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use candle_gen::gen_core::weightsmeta::{Dtype, SafetensorsTensorHeader};
use candle_gen::gen_core::LoadShape;
use candle_gen::gen_core::{
    self, Conditioning, GenerationRequest, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges,
    MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization,
    ResidentRequestMemory,
};
#[cfg(any(feature = "cuda", test))]
use candle_gen::gen_core::{MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryMode};
use candle_gen::{CandleError, Result as CandleResult};

use crate::config::Defaults;
use candle_gen_wan::config::{DEFAULT_FRAMES_14B, MAX_AREA_14B, SIZE_MULTIPLE_14B};
use sha2::{Digest, Sha256};

pub const DECODE_OVERLAP: u32 = 64;
// The merged Wan z16 implementation validates the provider-selected cap against this exact,
// CUDA-safe domain. Both the V2V and still contracts must use the executable set.
pub const DECODE_TILE_EDGES: &[u32] = &[512, 448, 384, 320, 256, 192];
const PACK_GROUP_SIZE: usize = 64;
const STATIC_CALIBRATION: &str = "bernini-candle-registry-v2v-v1";
pub const ADVERTISED_GEOMETRIES: &[(u32, u32)] =
    &[(848, 480), (480, 848), (1280, 720), (720, 1280)];
const ADVERTISED_STILL_GEOMETRIES: &[(u32, u32)] = &[
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1280, 720),
    (720, 1280),
];

fn tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

fn validate_geometry(width: u32, height: u32) -> gen_core::Result<()> {
    if ADVERTISED_GEOMETRIES.contains(&(width, height)) {
        Ok(())
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "Bernini memory evidence requires one of the advertised geometries {ADVERTISED_GEOMETRIES:?}, got {width}x{height}"
        )))
    }
}

fn known_provider(provider_id: &str) -> gen_core::Result<()> {
    [crate::pipeline::MODEL_ID, crate::bernini::MODEL_ID]
        .contains(&provider_id)
        .then_some(())
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!("unknown Bernini provider {provider_id}"))
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComponentPacking {
    Dense,
    Q4,
    Q8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ComponentReceipt {
    resident_bytes: u64,
    projections: BTreeMap<String, (usize, usize)>,
}

fn nested_safetensors(path: &Path, depth: usize) -> gen_core::Result<Option<PathBuf>> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if entry.file_type()?.is_dir() {
            if let Some(found) = nested_safetensors(&child, depth + 1)? {
                return Ok(Some(found));
            }
        } else if depth > 0 && child.extension().is_some_and(|ext| ext == "safetensors") {
            return Ok(Some(child));
        }
    }
    Ok(None)
}

fn tensor_elements(component: &str, tensor: &SafetensorsTensorHeader) -> gen_core::Result<u64> {
    let elements = tensor.shape.iter().try_fold(1_u64, |total, &dimension| {
        total
            .checked_mul(u64::try_from(dimension).map_err(|_| {
                gen_core::Error::Msg(format!(
                    "Bernini {component} tensor {} has an unrepresentable dimension",
                    tensor.name
                ))
            })?)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "Bernini {component} tensor {} element count overflow",
                    tensor.name
                ))
            })
    })?;
    let stored = elements
        .checked_mul(tensor.dtype.size() as u64)
        .ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "Bernini {component} tensor {} stored byte count overflow",
                tensor.name
            ))
        })?;
    if stored != tensor.data_bytes {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} tensor {} declares {} bytes but its dtype/shape require {stored}",
            tensor.name, tensor.data_bytes
        )));
    }
    Ok(elements)
}

/// Inspect exactly the direct safetensors shards `component_vb` opens, with later shards winning
/// duplicate keys. The generic weightsmeta directory helper is recursive and is therefore not
/// load-exact for this provider.
fn component_headers(
    root: &Path,
    component: &str,
) -> gen_core::Result<BTreeMap<String, SafetensorsTensorHeader>> {
    let path = root.join(component);
    if let Some(foreign) = nested_safetensors(&path, 0)? {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} contains nested foreign safetensors {}; only direct component files are loadable evidence",
            foreign.display()
        )));
    }
    let direct_files = candle_gen::sorted_safetensors(&path, component).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "Bernini {component} direct artifacts at {} are not loadable: {error}",
            path.display()
        ))
    })?;
    let mut tensors = BTreeMap::new();
    for file in direct_files {
        let headers =
            gen_core::weightsmeta::safetensors_path_tensor_headers(&file).map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "Bernini {component} direct artifact {} is not safetensors: {error}",
                    file.display()
                ))
            })?;
        for tensor in headers {
            tensor_elements(component, &tensor)?;
            tensors.insert(tensor.name.clone(), tensor);
        }
    }
    if tensors.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} direct safetensors contain no tensors"
        )));
    }
    Ok(tensors)
}

fn packed_bits(
    component: &str,
    base: &str,
    weight: &SafetensorsTensorHeader,
    scales: &SafetensorsTensorHeader,
) -> gen_core::Result<ComponentPacking> {
    let [_, packed_columns] = weight.shape.as_slice() else {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} packed {base}.weight must be rank-2 U32 codes"
        )));
    };
    let [_, scale_columns] = scales.shape.as_slice() else {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} packed {base}.scales must be rank-2"
        )));
    };
    let input = scale_columns.checked_mul(PACK_GROUP_SIZE).ok_or_else(|| {
        gen_core::Error::Msg(format!(
            "Bernini {component} packed {base} input width overflow"
        ))
    })?;
    let encoded = packed_columns.checked_mul(32).ok_or_else(|| {
        gen_core::Error::Msg(format!(
            "Bernini {component} packed {base} code width overflow"
        ))
    })?;
    if input == 0 || !encoded.is_multiple_of(input) {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} packed {base} has inconsistent U32/scales geometry"
        )));
    }
    match encoded / input {
        4 => Ok(ComponentPacking::Q4),
        8 => Ok(ComponentPacking::Q8),
        bits => Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} packed {base} encodes unsupported Q{bits}"
        ))),
    }
}

/// Project one direct component into the bytes the loader actually retains. Dense source tensors
/// are cast to the component's load dtype. Affine U32/scales/biases triples are validated at group
/// 64, derive their tier from geometry, and are priced as the GGML Q4_1/Q8_0 tensors Candle builds;
/// the transient safetensors payload is never charged as resident memory.
fn component_receipt(
    root: &Path,
    component: &str,
    expected: ComponentPacking,
    float_bytes: u64,
) -> gen_core::Result<ComponentReceipt> {
    let tensors = component_headers(root, component)?;

    let mut packed_bases = BTreeSet::new();
    let mut derived_packing = None;
    let mut resident_bytes = 0_u64;
    let mut projections = BTreeMap::new();
    for scales in tensors
        .values()
        .filter(|tensor| tensor.name.ends_with(".scales"))
    {
        let base = scales.name.trim_end_matches(".scales");
        let weight = tensors.get(&format!("{base}.weight")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "Bernini {component} has orphan packed scales {}",
                scales.name
            ))
        })?;
        let biases = tensors.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("Bernini {component} packed {base} has no biases"))
        })?;
        if weight.dtype != Dtype::U32
            || !matches!(scales.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
            || !matches!(biases.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini {component} packed {base} requires U32 codes and F16/BF16/F32 scales/biases"
            )));
        }
        let packing = packed_bits(component, base, weight, scales)?;
        if derived_packing
            .replace(packing)
            .is_some_and(|prior| prior != packing)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini {component} mixes Q4 and Q8 packed geometry"
            )));
        }
        let projected = candle_gen::quant::mlx_packed_qtensor_resident_bytes(
            weight,
            scales,
            biases,
            PACK_GROUP_SIZE,
        )?;
        resident_bytes = resident_bytes.checked_add(projected).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "Bernini {component} packed resident byte total overflow"
            ))
        })?;
        let [output, scale_columns] = scales.shape.as_slice() else {
            unreachable!("packed_bits validated rank")
        };
        projections.insert(base.to_owned(), (*output, scale_columns * PACK_GROUP_SIZE));
        packed_bases.insert(base.to_owned());
    }
    for tensor in tensors.values() {
        if tensor.name.ends_with(".scales") || tensor.name.ends_with(".biases") {
            let base = tensor
                .name
                .trim_end_matches(".scales")
                .trim_end_matches(".biases");
            if !packed_bases.contains(base) {
                return Err(gen_core::Error::Unsupported(format!(
                    "Bernini {component} has an orphan packed leaf {}",
                    tensor.name
                )));
            }
            continue;
        }
        if let Some(base) = tensor.name.strip_suffix(".weight") {
            if packed_bases.contains(base) {
                continue;
            }
        }
        if !tensor.is_float() {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini {component} ordinary tensor {} must use floating storage; only validated packed weights may be U32",
                tensor.name
            )));
        }
        let projected = tensor_elements(component, tensor)?
            .checked_mul(float_bytes)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "Bernini {component} tensor {} projected byte count overflow",
                    tensor.name
                ))
            })?;
        resident_bytes = resident_bytes.checked_add(projected).ok_or_else(|| {
            gen_core::Error::Msg(format!("Bernini {component} resident byte total overflow"))
        })?;
        if let [output, input] = tensor.shape.as_slice() {
            if tensor.name.ends_with(".weight") {
                projections.insert(
                    tensor.name.trim_end_matches(".weight").to_owned(),
                    (*output, *input),
                );
            }
        }
    }
    let actual = derived_packing.unwrap_or(ComponentPacking::Dense);
    let marker = quant_marker(root, component)?;
    if marker != derived_packing {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} quantize_config disagrees with direct tensor geometry: marker={marker:?}, actual={derived_packing:?}"
        )));
    }
    if actual != expected {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} loads as {actual:?}, crossed requested tier {expected:?}"
        )));
    }
    if resident_bytes == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {component} has zero projected resident bytes"
        )));
    }
    Ok(ComponentReceipt {
        resident_bytes,
        projections,
    })
}

fn quant_marker(root: &Path, component: &str) -> gen_core::Result<Option<ComponentPacking>> {
    let path = root.join(component).join("quantize_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "{}: invalid quant marker: {error}",
                path.display()
            ))
        })?;
    let bits = value
        .get("bits")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!("{}: quant marker has no bits", path.display()))
        })?;
    let group = value
        .get("quantization")
        .and_then(|quantization| quantization.get("group_size"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(PACK_GROUP_SIZE as u64);
    if group != PACK_GROUP_SIZE as u64 {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: quant marker group size {group} disagrees with loader group {PACK_GROUP_SIZE}",
            path.display()
        )));
    }
    match bits {
        4 => Ok(Some(ComponentPacking::Q4)),
        8 => Ok(Some(ComponentPacking::Q8)),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{}: quant marker bits {bits} must be 4 or 8",
            path.display()
        ))),
    }
}

fn expected_packing(spec: &LoadSpec) -> gen_core::Result<ComponentPacking> {
    match spec.quantize {
        None => Ok(ComponentPacking::Dense),
        Some(gen_core::Quant::Q4) => Ok(ComponentPacking::Q4),
        Some(gen_core::Quant::Q8) => Ok(ComponentPacking::Q8),
        Some(other) => Err(gen_core::Error::Unsupported(format!(
            "Bernini Candle memory contract does not recognize quant tier {other:?}"
        ))),
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AdapterLoadReceipt {
    canonical_path: PathBuf,
    digest: [u8; 32],
    kind: gen_core::AdapterKind,
    scale_bits: u32,
    pass_scale_bits: Option<Vec<u32>>,
    expert: Option<gen_core::MoeExpert>,
    artifact_bytes: u64,
    applied_f32_bytes: u64,
}

impl AdapterLoadReceipt {
    fn identity(&self) -> String {
        let path_hex = self
            .canonical_path
            .to_string_lossy()
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let pass_scales = self.pass_scale_bits.as_ref().map_or_else(
            || "none".to_owned(),
            |bits| {
                bits.iter()
                    .map(|bits| format!("{bits:08x}"))
                    .collect::<Vec<_>>()
                    .join("/")
            },
        );
        format!(
            "artifact=safetensors;path_hex={path_hex};digest=sha256:{};kind={:?};scale_bits={:08x};pass_scale_bits={pass_scales};expert={:?};artifact_bytes={};applied_f32_bytes={};stable=true",
            self.digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
            self.kind,
            self.scale_bits,
            self.expert,
            self.artifact_bytes,
            self.applied_f32_bytes
        )
    }
}

fn read_adapter_receipt(
    adapter: &gen_core::AdapterSpec,
    projection_sets: &[&BTreeMap<String, (usize, usize)>],
) -> gen_core::Result<AdapterLoadReceipt> {
    if adapter
        .path
        .extension()
        .is_none_or(|extension| extension != "safetensors")
    {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini adapter {} is not a safetensors artifact",
            adapter.path.display()
        )));
    }
    let canonical_path = std::fs::canonicalize(&adapter.path).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "Bernini adapter {} cannot be resolved exactly: {error}",
            adapter.path.display()
        ))
    })?;
    let before = std::fs::metadata(&canonical_path)?;
    let mut file = File::open(&canonical_path)?;
    let opened = file.metadata()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let after = std::fs::metadata(&canonical_path)?;
    if bytes.is_empty()
        || before.len() != opened.len()
        || opened.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || u64::try_from(bytes.len()).ok() != Some(after.len())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini adapter {} changed while its load receipt was read",
            canonical_path.display()
        )));
    }
    safetensors::SafeTensors::deserialize(&bytes).map_err(|error| {
        gen_core::Error::Unsupported(format!(
            "Bernini adapter {} is not loadable safetensors: {error}",
            canonical_path.display()
        ))
    })?;
    let applied_f32_bytes = projection_sets
        .iter()
        .try_fold(0_u64, |total, projections| {
            let bytes = candle_gen_wan::adapters::applied_factor_f32_bytes(adapter, projections)
                .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?;
            total.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg("Bernini adapter retained F32 bytes overflow".into())
            })
        })?;
    let verified_again = std::fs::read(&canonical_path)?;
    if verified_again != bytes {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini adapter {} changed during its post-read stability check",
            canonical_path.display()
        )));
    }
    if !adapter.scale.is_finite()
        || adapter
            .pass_scales
            .as_ref()
            .is_some_and(|scales| scales.iter().any(|scale| !scale.is_finite()))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini adapter {} has non-finite scale evidence",
            canonical_path.display()
        )));
    }
    Ok(AdapterLoadReceipt {
        canonical_path,
        digest: Sha256::digest(&bytes).into(),
        kind: adapter.kind,
        scale_bits: adapter.scale.to_bits(),
        pass_scale_bits: adapter
            .pass_scales
            .as_ref()
            .map(|scales| scales.iter().map(|scale| scale.to_bits()).collect()),
        expert: adapter.moe_expert,
        artifact_bytes: u64::try_from(bytes.len())
            .map_err(|_| gen_core::Error::Msg("Bernini adapter artifact bytes overflow".into()))?,
        applied_f32_bytes,
    })
}

fn adapter_identity(
    spec: &LoadSpec,
    packing: ComponentPacking,
    high_projections: &BTreeMap<String, (usize, usize)>,
    low_projections: &BTreeMap<String, (usize, usize)>,
) -> gen_core::Result<(u64, String)> {
    if spec.adapters.is_empty() {
        return Ok((0, String::new()));
    }
    let mut total = 0_u64;
    let mut identities = Vec::with_capacity(spec.adapters.len());
    for adapter in &spec.adapters {
        let projection_sets = match (packing, adapter.moe_expert) {
            (ComponentPacking::Dense, _) => Vec::new(),
            (_, None) => vec![high_projections, low_projections],
            (_, Some(gen_core::MoeExpert::High)) => vec![high_projections],
            (_, Some(gen_core::MoeExpert::Low)) => vec![low_projections],
        };
        let receipt = read_adapter_receipt(adapter, &projection_sets)?;
        // Dense adapters are folded into each expert's base map and leave no independent resident
        // allocation. Packed tiers keep an additive residual alive; shared adapters are installed
        // on both loaded experts, while an expert-targeted one is installed once.
        if packing != ComponentPacking::Dense {
            total = total
                .checked_add(receipt.applied_f32_bytes)
                .ok_or_else(|| {
                    gen_core::Error::Msg("Bernini adapter resident bytes overflow".into())
                })?;
        }
        identities.push(receipt.identity());
    }
    Ok((total, format!("adapters:[{}]", identities.join(","))))
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::StagedResidency => MemoryStrategySupport::Missing,
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    MemoryStrategySupport::Missing
                }
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

fn contract(
    provider_id: &str,
    spec: &LoadSpec,
    calibration: Option<MemoryCalibrationIdentity>,
    facts: MemoryAssetFacts,
    adapter_identity: Option<String>,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::FrameCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
    ];
    let resident_components = if let Some(adapter_identity) = adapter_identity {
        variables.push(MemoryFormulaVariable::OverlayBytes);
        vec![MemoryResidentComponent {
            id: adapter_identity,
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: facts.overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        }]
    } else {
        Vec::new()
    };
    let formula = if resident_components.is_empty() {
        MemoryFormulaKind::PhaseEnvelope {
            phases: phases.clone(),
            variables,
        }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases.clone(),
            variables,
            resident_components,
        }
    };
    MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula,
        calibration,
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

/// Weights-free declaration used by registry conformance. This is not production evidence.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if !matches!(
        provider_id,
        crate::pipeline::MODEL_ID | crate::bernini::MODEL_ID
    ) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown Bernini provider {provider_id}"
        )));
    }
    Ok(contract(
        provider_id,
        spec,
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION,
            spec.load_shape,
        )),
        MemoryAssetFacts::default(),
        None,
    ))
}

fn production_assets(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<(MemoryAssetFacts, MemoryNumericTier, Option<String>)> {
    known_provider(provider_id)?;
    let gen_core::WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "Bernini Candle memory contract requires a snapshot directory".into(),
        ));
    };
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(gen_core::Error::Unsupported(
            "Bernini Candle memory contract requires EagerMaterialization".into(),
        ));
    }
    let expected = expected_packing(spec)?;
    let conditioning =
        component_receipt(root, "text_encoder", ComponentPacking::Dense, 4)?.resident_bytes;
    let high = component_receipt(root, "transformer", expected, 2)?;
    let low = component_receipt(root, "transformer_2", expected, 2)?;
    let transformer = high
        .resident_bytes
        .checked_add(low.resident_bytes)
        .ok_or_else(|| gen_core::Error::Msg("Bernini transformer bytes overflow".into()))?;
    let decoder = component_receipt(root, "vae", ComponentPacking::Dense, 4)?.resident_bytes;
    let planner = if provider_id == crate::bernini::MODEL_ID {
        let mllm = component_receipt(root, "mllm", expected, 2)?.resident_bytes;
        let connector =
            component_receipt(root, "connector", ComponentPacking::Dense, 2)?.resident_bytes;
        let vit_decoder =
            component_receipt(root, "vit_decoder", ComponentPacking::Dense, 2)?.resident_bytes;
        mllm.checked_add(connector)
            .and_then(|total| total.checked_add(vit_decoder))
            .ok_or_else(|| gen_core::Error::Msg("Bernini planner bytes overflow".into()))?
    } else {
        0
    };
    let (overlay_bytes, overlay_identity) =
        adapter_identity(spec, expected, &high.projections, &low.projections)?;
    let facts = MemoryAssetFacts {
        base_bytes: conditioning
            .checked_add(planner)
            .and_then(|value| value.checked_add(transformer))
            .and_then(|value| value.checked_add(decoder))
            .ok_or_else(|| gen_core::Error::Msg("Bernini base bytes overflow".into()))?,
        conditioning_bytes: conditioning
            .checked_add(planner)
            .ok_or_else(|| gen_core::Error::Msg("Bernini conditioning bytes overflow".into()))?,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes,
    };
    Ok((
        facts,
        tier(spec),
        (!overlay_identity.is_empty()).then_some(overlay_identity),
    ))
}

/// Real loads expose load-exact asset/tier identity, but no calibration identity until the
/// Windows/CUDA evidence campaign exists. That makes the shared selector reachable while every
/// optimized selection still refuses through the normal uncalibrated contract path.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let (facts, _tier, adapter_identity) = production_assets(provider_id, spec)?;
    Ok(contract(provider_id, spec, None, facts, adapter_identity))
}

/// Recompute the load-exact contract immediately around the lazy component load. A caller may keep
/// the same adapter path while replacing its contents after provider construction; that path must
/// never execute new bytes under the old receipt.
pub fn validate_loaded_contract(
    provider_id: &str,
    spec: &LoadSpec,
    expected_contract: &MemoryProviderContract,
    expected_tier: MemoryNumericTier,
) -> gen_core::Result<()> {
    let (facts, actual_tier, adapter_identity) = production_assets(provider_id, spec)?;
    let actual_contract = contract(provider_id, spec, None, facts, adapter_identity);
    if actual_tier != expected_tier || actual_contract != *expected_contract {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini {provider_id} lazy-load artifacts changed after the memory contract was sealed"
        )));
    }
    Ok(())
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported("Bernini bounded decode needs tile edge".into())
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported("Bernini bounded decode needs overlap".into())
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini bounded decode requires edge in {DECODE_TILE_EDGES:?} and overlap {DECODE_OVERLAP}, got {edge}/{overlap}"
        )));
    }
    Ok(())
}

fn route_ok(contract: &MemoryProviderContract, context: &MemoryRunContext) -> gen_core::Result<()> {
    if matches!(context.mode.as_key(), "text_to_image" | "edit") {
        let expected_references = u32::from(context.mode.as_key() == "edit");
        if context.geometry.reference_count != expected_references
            || context.has_reference != (expected_references == 1)
            || context.geometry.batch != 1
            || context.geometry.frames != 1
            || context.use_pid
            || context.has_phases
            || !ADVERTISED_STILL_GEOMETRIES
                .contains(&(context.geometry.width, context.geometry.height))
            || context
                .overlay
                .as_deref()
                .is_some_and(|overlay| overlay != "lora")
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{} still-memory evidence requires one advertised T2I/edit frame with its exact reference identity",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        return Ok(());
    }
    if context.mode.as_key() != "video_to_video"
        || context.geometry.reference_count != 1
        || !context.has_reference
        || context.geometry.batch != 1
        || context.use_pid
        || context.has_phases
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} memory evidence requires one video_to_video clip, one batch, and no PiD/phases",
            contract.provider_id
        )));
    }
    validate_geometry(context.geometry.width, context.geometry.height)?;
    if context.overlay.as_deref().is_some_and(|overlay| {
        overlay
            .split('+')
            .find(|axis| axis.starts_with("provider_video_mode:"))
            .is_some_and(|axis| axis != "provider_video_mode:v2v")
    }) {
        return Err(gen_core::Error::Unsupported(format!(
            "{} provider video mode overlay crossed the v2v contract",
            contract.provider_id
        )));
    }
    let adapter_axis = context.overlay.as_deref().and_then(|overlay| {
        overlay
            .split('+')
            .find(|axis| axis.starts_with("adapters:["))
    });
    let expected_adapter_axis = contract
        .resident_components()
        .iter()
        .find(|component| component.kind == MemoryComponentKind::AdapterStack)
        .map(|component| component.id.as_str());
    if adapter_axis != expected_adapter_axis {
        return Err(gen_core::Error::Unsupported(format!(
            "{} adapter artifact identity is missing or crossed the loaded contract",
            contract.provider_id
        )));
    }
    let area = u64::from(context.geometry.width) * u64::from(context.geometry.height);
    if !context.geometry.width.is_multiple_of(SIZE_MULTIPLE_14B)
        || !context.geometry.height.is_multiple_of(SIZE_MULTIPLE_14B)
        || area > MAX_AREA_14B as u64
        || !matches!(context.geometry.frames, 45 | 61 | 77)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} memory evidence does not cover {}x{} frames={}",
            contract.provider_id,
            context.geometry.width,
            context.geometry.height,
            context.geometry.frames
        )));
    }
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
        validate_decode(
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        )?;
    }
    Ok(())
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let gate = || route_ok(contract, context);
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&gate),
    )
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, tier(spec), context)
}

pub(crate) fn validate_request(request: &GenerationRequest) -> gen_core::Result<()> {
    if matches!(request.video_mode.as_deref(), Some("t2i") | Some("i2i")) {
        let references = request.image_reference_count();
        let expected_references = u32::from(request.video_mode.as_deref() == Some("i2i"));
        if request.frames != Some(1)
            || request.count != 1
            || request.phases.is_some()
            || request.use_pid
            || !request.video_clips().is_empty()
            || references != expected_references
            || !ADVERTISED_STILL_GEOMETRIES.contains(&(request.width, request.height))
        {
            return Err(gen_core::Error::Unsupported(
                "Bernini Candle still memory scope requires exact T2I/I2I single-frame identity"
                    .to_owned(),
            ));
        }
        if let Some(memory) = request.memory {
            if memory.stage_residency || memory.chunk_attention || memory.stream_transformer_blocks
            {
                return Err(gen_core::Error::Unsupported(
                    "Bernini Candle still request carries an unsupported memory mechanism"
                        .to_owned(),
                ));
            }
            if memory.tile_vae_decode {
                validate_decode(memory.decode_tile_edge, memory.decode_overlap)?;
            } else if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
                return Err(gen_core::Error::Unsupported(
                    "Bernini Candle decode parameters require bounded decode".to_owned(),
                ));
            }
        }
        return Ok(());
    }
    if request.video_mode.as_deref() != Some("v2v")
        || request.fps.unwrap_or(Defaults::FPS) != Defaults::FPS
        || request.count != 1
        || request.image_reference_count() != 0
        || request.video_clips().len() != 1
        || !matches!(
            request.conditioning.as_slice(),
            [Conditioning::VideoClip { .. }]
        )
        || !matches!(request.frames, Some(45 | 61 | 77))
    {
        return Err(gen_core::Error::Unsupported(
            "Bernini Candle memory scope requires exactly one VideoClip, v2v, FPS16, and 3/4/5s"
                .to_owned(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    Ok(())
}

struct BerniniMemoryRequestScope {
    inner: candle_gen::request_scope::CandleRequestScopeCore,
}

impl MemoryRequestScope for BerniniMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        validate_request(request)?;
        self.inner.configure_request(request)
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.leave_phase(phase)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        mut geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        geometry.reference_count = 0;
        self.inner.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, chunk: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.inner.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.inner.finish(outcome)
    }
}

#[cfg(any(feature = "cuda", test))]
pub fn contract_for_loaded(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<Option<(MemoryProviderContract, MemoryNumericTier)>> {
    let (facts, loaded_tier, adapter_identity) = match production_assets(provider_id, spec) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    Ok(Some((
        contract(provider_id, spec, None, facts, adapter_identity),
        loaded_tier,
    )))
}

#[cfg(any(feature = "cuda", test))]
pub fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("video_to_video".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.geometry.width = 848;
    context.geometry.height = 480;
    context.geometry.frames = 45;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free Bernini v2v memory behavior".to_owned();
    fixture.request.video_mode = Some("v2v".to_owned());
    fixture.request.fps = Some(16);
    fixture.request.conditioning.clear();
    fixture.request.conditioning.push(Conditioning::VideoClip {
        frames: vec![gen_core::Image {
            width: 2,
            height: 2,
            pixels: vec![0; 12],
        }],
        frame_idx: 0,
        strength: 1.0,
    });
    Ok(vec![fixture])
}

#[cfg(any(feature = "cuda", test))]
pub fn registered_begin_request(
    provider_id: &str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let Some((_, loaded_tier)) = contract_for_loaded(spec, provider_id)? else {
        return Ok(None);
    };
    begin_request(contract, loaded_tier, Device::Cpu, context)
}

pub fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, loaded_tier, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let is_v2v = context.mode.as_key() == "video_to_video";
    let mut geometry = context.geometry;
    if is_v2v {
        geometry.reference_count = 0;
    }
    let provider_id = match contract.provider_id.as_str() {
        crate::pipeline::MODEL_ID => crate::pipeline::MODEL_ID,
        crate::bernini::MODEL_ID => crate::bernini::MODEL_ID,
        _ => {
            return Err(gen_core::Error::Unsupported(
                "unknown Bernini provider".into(),
            ))
        }
    };
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        80,
        |_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.default_frames = if is_v2v { DEFAULT_FRAMES_14B } else { 1 };
    Ok(Some(Box::new(BerniniMemoryRequestScope {
        inner: candle_gen::request_scope::CandleRequestScopeCore::new(config),
    })))
}

pub fn selected_decode_cap(request: &GenerationRequest) -> CandleResult<Option<u32>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(CandleError::Msg(
                "Bernini decode parameters require bounded decode".into(),
            ));
        }
        return Ok(None);
    }
    validate_decode(memory.decode_tile_edge, memory.decode_overlap)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    Ok(memory.decode_tile_edge)
}

fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

pub const RENDERER_MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: crate::pipeline::MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(crate::pipeline::MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };

pub const FULL_MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: crate::bernini::MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(crate::bernini::MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };

pub const RENDERER_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: crate::pipeline::MODEL_ID,
        contract: |spec| memory_strategy_contract(crate::pipeline::MODEL_ID, spec),
        safety_check: registered_safety_check,
    };

pub const FULL_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::bernini::MODEL_ID,
    contract: |spec| memory_strategy_contract(crate::bernini::MODEL_ID, spec),
    safety_check: registered_safety_check,
};

#[cfg(any(feature = "cuda", test))]
pub const RENDERER_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::pipeline::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: |spec, contract, context| {
            registered_begin_request(crate::pipeline::MODEL_ID, spec, contract, context)
        },
    };

#[cfg(any(feature = "cuda", test))]
pub const FULL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::bernini::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: |spec, contract, context| {
            registered_begin_request(crate::bernini::MODEL_ID, spec, contract, context)
        },
    };

/// Add the weights-free Bernini contract surfaces to an external catalog walk. Production loads
/// still expose no contract until the Windows/CUDA evidence campaign mints one.
pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_contract_fixture(RENDERER_MEMORY_FIXTURE)
        .register_memory_contract_fixture(FULL_MEMORY_FIXTURE)
}

/// Provider-owned registration hook used by the CUDA catalog and source-derived wiring checks.
pub fn register_memory_strategy(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(RENDERER_MEMORY_REGISTRATION)
        .register_memory_strategy(FULL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(RENDERER_MEMORY_FIXTURE)
        .register_memory_contract_fixture(FULL_MEMORY_FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tensor_file(
        path: &Path,
        tensors: &[(&str, safetensors::Dtype, Vec<usize>)],
        metadata: Option<std::collections::HashMap<String, String>>,
    ) {
        let owned = tensors
            .iter()
            .map(|(name, dtype, shape)| {
                let elements = shape.iter().product::<usize>();
                (
                    (*name).to_owned(),
                    *dtype,
                    shape.clone(),
                    vec![0_u8; (elements * dtype.bitsize()).div_ceil(8)],
                )
            })
            .collect::<Vec<_>>();
        let views = owned
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(views, metadata, path).unwrap();
    }

    fn write_component_with_marker(
        root: &Path,
        component: &str,
        packing: ComponentPacking,
        marker: ComponentPacking,
    ) {
        let dir = root.join(component);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        match packing {
            ComponentPacking::Dense => write_tensor_file(
                &path,
                &[
                    ("proj_out.weight", safetensors::Dtype::F16, vec![2, 64]),
                    (
                        "blocks.0.attn1.to_q.weight",
                        safetensors::Dtype::BF16,
                        vec![2, 64],
                    ),
                ],
                None,
            ),
            ComponentPacking::Q4 | ComponentPacking::Q8 => {
                let columns = if packing == ComponentPacking::Q4 {
                    8
                } else {
                    16
                };
                write_tensor_file(
                    &path,
                    &[
                        ("proj_out.weight", safetensors::Dtype::U32, vec![2, columns]),
                        ("proj_out.scales", safetensors::Dtype::F16, vec![2, 1]),
                        ("proj_out.biases", safetensors::Dtype::BF16, vec![2, 1]),
                        (
                            "blocks.0.attn1.to_q.weight",
                            safetensors::Dtype::U32,
                            vec![2, columns],
                        ),
                        (
                            "blocks.0.attn1.to_q.scales",
                            safetensors::Dtype::BF16,
                            vec![2, 1],
                        ),
                        (
                            "blocks.0.attn1.to_q.biases",
                            safetensors::Dtype::F32,
                            vec![2, 1],
                        ),
                    ],
                    None,
                );
            }
        }
        let bits = match marker {
            ComponentPacking::Dense => None,
            ComponentPacking::Q4 => Some(4),
            ComponentPacking::Q8 => Some(8),
        };
        if let Some(bits) = bits {
            std::fs::write(
                dir.join("quantize_config.json"),
                format!(r#"{{"bits":{bits},"quantization":{{"group_size":64}}}}"#),
            )
            .unwrap();
        }
    }

    fn write_component(root: &Path, component: &str, packing: ComponentPacking) {
        write_component_with_marker(root, component, packing, packing);
    }

    fn write_full_snapshot(root: &Path, packing: ComponentPacking) {
        for component in ["text_encoder", "connector", "vit_decoder", "vae"] {
            write_component(root, component, ComponentPacking::Dense);
        }
        for component in ["transformer", "transformer_2", "mllm"] {
            write_component(root, component, packing);
        }
    }

    fn write_lora(path: &Path, dtype: safetensors::Dtype, extra_unknown: bool) {
        let mut tensors = vec![
            ("blocks.0.attn1.to_q.lora_A.weight", dtype, vec![2, 64]),
            ("blocks.0.attn1.to_q.lora_B.weight", dtype, vec![2, 2]),
        ];
        if extra_unknown {
            tensors.push(("text_encoder.unused.weight", dtype, vec![128, 128]));
        }
        write_tensor_file(path, &tensors, None);
    }

    fn write_lokr(path: &Path, dtype: safetensors::Dtype) {
        write_tensor_file(
            path,
            &[
                ("blocks.0.attn1.to_q.lokr_w1", dtype, vec![1, 8]),
                ("blocks.0.attn1.to_q.lokr_w2", dtype, vec![2, 8]),
            ],
            Some(std::collections::HashMap::from([
                ("networkType".to_owned(), "lokr".to_owned()),
                ("rank".to_owned(), "2".to_owned()),
                ("alpha".to_owned(), "2".to_owned()),
            ])),
        );
    }

    fn fixture_projections() -> BTreeMap<String, (usize, usize)> {
        BTreeMap::from([("blocks.0.attn1.to_q".to_owned(), (2, 64))])
    }

    #[test]
    fn candle_bernini_declares_missing_attention_and_transformer_rungs() {
        let spec = LoadSpec::new(gen_core::WeightsSource::Dir("/missing".into()));
        let contract =
            weights_free_memory_strategy_contract(crate::pipeline::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn candle_bernini_decode_cap_is_exactly_admitted() {
        let mut request = GenerationRequest {
            memory: Some(gen_core::GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(selected_decode_cap(&request).unwrap(), Some(512));
        request.memory.as_mut().unwrap().decode_tile_edge = Some(511);
        assert!(selected_decode_cap(&request).is_err());
    }

    #[test]
    fn candle_bernini_scope_rejects_plain_and_wrong_fps_requests() {
        let clip = Conditioning::VideoClip {
            frames: vec![gen_core::Image {
                width: 2,
                height: 2,
                pixels: vec![0; 12],
            }],
            frame_idx: 0,
            strength: 1.0,
        };
        let mut request = GenerationRequest {
            prompt: "v2v".to_owned(),
            width: 848,
            height: 480,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("v2v".to_owned()),
            conditioning: vec![clip],
            ..Default::default()
        };
        assert!(validate_request(&request).is_ok());
        request.video_mode = None;
        assert!(validate_request(&request).is_err());
        request.video_mode = Some("v2v".to_owned());
        request.fps = Some(24);
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn candle_bernini_scope_admits_only_advertised_geometry_and_frames() {
        for &(width, height) in ADVERTISED_GEOMETRIES {
            for frames in [45, 61, 77] {
                let request = GenerationRequest {
                    prompt: "v2v".to_owned(),
                    width,
                    height,
                    frames: Some(frames),
                    fps: Some(16),
                    video_mode: Some("v2v".to_owned()),
                    conditioning: vec![Conditioning::VideoClip {
                        frames: vec![gen_core::Image {
                            width: 2,
                            height: 2,
                            pixels: vec![0; 12],
                        }],
                        frame_idx: 0,
                        strength: 1.0,
                    }],
                    ..Default::default()
                };
                assert!(
                    validate_request(&request).is_ok(),
                    "{width}x{height}/{frames}"
                );
            }
        }
        let mut crossed = GenerationRequest {
            width: 640,
            height: 640,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("v2v".to_owned()),
            conditioning: vec![Conditioning::VideoClip {
                frames: vec![gen_core::Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 12],
                }],
                frame_idx: 0,
                strength: 1.0,
            }],
            ..Default::default()
        };
        assert!(validate_request(&crossed).is_err());
        crossed.width = 848;
        crossed.height = 480;
        crossed.fps = Some(24);
        assert!(validate_request(&crossed).is_err());
    }

    #[test]
    fn candle_bernini_loaded_contract_prices_shared_adapter_per_expert_and_exact_tier() {
        let root = tempfile::tempdir().unwrap();
        write_full_snapshot(root.path(), ComponentPacking::Q4);
        let adapter = root.path().join("adapter.safetensors");
        write_lora(&adapter, safetensors::Dtype::F16, true);
        let high_adapter = root.path().join("high-adapter.safetensors");
        write_lokr(&high_adapter, safetensors::Dtype::BF16);
        let projections = fixture_projections();
        let adapter_bytes = read_adapter_receipt(
            &gen_core::AdapterSpec::new(adapter.clone(), 0.5, gen_core::AdapterKind::Lora),
            &[&projections, &projections],
        )
        .unwrap()
        .applied_f32_bytes;
        let high_bytes = read_adapter_receipt(
            &gen_core::AdapterSpec::new(high_adapter.clone(), 1.0, gen_core::AdapterKind::Lokr),
            &[&projections],
        )
        .unwrap()
        .applied_f32_bytes;
        assert_eq!(adapter_bytes, 1_056, "F16 shared factors retain F32 twice");
        assert_eq!(
            high_bytes, 96,
            "BF16 LoKr factors retain their built F32 form"
        );
        let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()))
            .with_quant(gen_core::Quant::Q4);
        spec.adapters = vec![
            gen_core::AdapterSpec::new(adapter, 0.5, gen_core::AdapterKind::Lora),
            gen_core::AdapterSpec {
                path: high_adapter,
                scale: 1.0,
                kind: gen_core::AdapterKind::Lokr,
                moe_expert: Some(gen_core::MoeExpert::High),
                pass_scales: None,
            },
        ];
        let contract = memory_strategy_contract(crate::bernini::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.calibration, None);
        assert_eq!(
            contract.asset_facts.overlay_bytes,
            adapter_bytes + high_bytes
        );
        assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert!(contract.resident_components().iter().any(|component| {
            component.kind == MemoryComponentKind::AdapterStack
                && component.resident_bytes == adapter_bytes + high_bytes
                && component.id.contains("digest=sha256:")
                && component.id.contains("scale_bits=3f000000")
                && component.id.contains("expert=Some(High)")
                && component.id.contains("applied_f32_bytes=")
        }));
        let stale = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()));
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &stale).is_err());
    }

    #[test]
    fn adapter_receipt_binds_every_load_knob_and_post_read_artifact() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("adapter.safetensors");
        write_lora(&path, safetensors::Dtype::F16, false);
        let projections = fixture_projections();
        let base =
            gen_core::AdapterSpec::new(path.clone(), 0.123_456_79, gen_core::AdapterKind::Lora);
        let base_id = read_adapter_receipt(&base, &[&projections])
            .unwrap()
            .identity();
        assert!(base_id.contains(&format!("scale_bits={:08x}", base.scale.to_bits())));
        assert!(base_id.contains("pass_scale_bits=none"));
        assert!(base_id.contains("expert=None"));
        assert!(base_id.contains("artifact_bytes="));
        assert!(base_id.contains("applied_f32_bytes=528"));
        assert!(base_id.contains("stable=true"));

        let mut crossed_scale = base.clone();
        crossed_scale.scale = f32::from_bits(base.scale.to_bits() + 1);
        let mut crossed_pass = base.clone();
        crossed_pass.pass_scales = Some(vec![0.25, 0.75]);
        let crossed_high = base.clone().with_moe_expert(gen_core::MoeExpert::High);
        let mut crossed_kind = base.clone();
        crossed_kind.kind = gen_core::AdapterKind::Lokr;
        for crossed in [&crossed_scale, &crossed_pass, &crossed_high, &crossed_kind] {
            let result = read_adapter_receipt(crossed, &[&projections]);
            if crossed.kind == gen_core::AdapterKind::Lokr {
                assert!(result.is_err(), "kind/file disagreement must fail closed");
            } else {
                assert_ne!(result.unwrap().identity(), base_id);
            }
        }
        assert!(read_adapter_receipt(&crossed_pass, &[&projections])
            .unwrap()
            .identity()
            .contains(&format!(
                "pass_scale_bits={:08x}/{:08x}",
                0.25_f32.to_bits(),
                0.75_f32.to_bits()
            )));

        // A post-receipt artifact mutation cannot borrow the old identity, even when the file stays
        // valid safetensors and happens to retain the same filesystem path.
        write_lora(&path, safetensors::Dtype::BF16, true);
        let mutated_id = read_adapter_receipt(&base, &[&projections])
            .unwrap()
            .identity();
        assert_ne!(mutated_id, base_id);
        assert!(mutated_id.contains("applied_f32_bytes=528"));

        let unapplied = root.path().join("unapplied.safetensors");
        write_tensor_file(
            &unapplied,
            &[(
                "unknown.branch.weight",
                safetensors::Dtype::F16,
                vec![64, 64],
            )],
            None,
        );
        let unapplied = gen_core::AdapterSpec::new(unapplied, 1.0, gen_core::AdapterKind::Lora);
        assert!(
            read_adapter_receipt(&unapplied, &[&projections]).is_err(),
            "a valid adapter container with no factors applied to the selected expert must fail closed"
        );

        let second = root.path().join("second.safetensors");
        write_lora(&second, safetensors::Dtype::BF16, false);
        let mut stack = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()));
        stack.adapters = vec![
            base,
            gen_core::AdapterSpec::new(second, 1.0, gen_core::AdapterKind::Lora)
                .with_moe_expert(gen_core::MoeExpert::Low),
        ];
        let (_, multi) =
            adapter_identity(&stack, ComponentPacking::Q4, &projections, &projections).unwrap();
        assert_eq!(
            adapter_identity(&stack, ComponentPacking::Q4, &projections, &projections)
                .unwrap()
                .0,
            1_584,
            "shared retained f32 factors are installed on both experts and Low-targeted factors once"
        );
        assert_eq!(multi.matches("artifact=safetensors").count(), 2);
        assert!(multi.contains("expert=Some(Low)"));
    }

    #[test]
    fn dense_folded_adapters_keep_exact_identity_but_zero_independent_residency() {
        let root = tempfile::tempdir().unwrap();
        write_full_snapshot(root.path(), ComponentPacking::Dense);
        let shared = root.path().join("shared.safetensors");
        let low = root.path().join("low.safetensors");
        write_lora(&shared, safetensors::Dtype::F16, true);
        write_lokr(&low, safetensors::Dtype::BF16);
        let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()));
        spec.adapters = vec![
            gen_core::AdapterSpec::new(shared, 0.75, gen_core::AdapterKind::Lora),
            gen_core::AdapterSpec::new(low, 1.25, gen_core::AdapterKind::Lokr)
                .with_moe_expert(gen_core::MoeExpert::Low),
        ];
        let contract = memory_strategy_contract(crate::bernini::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.asset_facts.overlay_bytes, 0);
        let receipt = contract
            .resident_components()
            .iter()
            .find(|component| component.kind == MemoryComponentKind::AdapterStack)
            .expect("dense folded stack retains request identity");
        assert_eq!(receipt.resident_bytes, 0);
        assert_eq!(receipt.id.matches("artifact=safetensors").count(), 2);
        assert!(receipt.id.contains("expert=Some(Low)"));
    }

    #[test]
    fn component_receipt_uses_only_direct_parseable_loader_evidence() {
        let root = tempfile::tempdir().unwrap();
        write_full_snapshot(root.path(), ComponentPacking::Q4);
        let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()))
            .with_quant(gen_core::Quant::Q4);
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &spec).is_ok());

        // A stale marker cannot upgrade dense direct weights.
        write_component(root.path(), "transformer", ComponentPacking::Dense);
        std::fs::write(
            root.path().join("transformer/quantize_config.json"),
            r#"{"bits":4}"#,
        )
        .unwrap();
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &spec).is_err());

        // Restore the direct tier, then add a foreign nested container. Recursive byte counters
        // would price it even though component_vb never opens it.
        write_component(root.path(), "transformer", ComponentPacking::Q4);
        let nested = root.path().join("transformer/foreign");
        std::fs::create_dir_all(&nested).unwrap();
        write_tensor_file(
            &nested.join("model.safetensors"),
            &[("proj_out.scales", safetensors::Dtype::F16, vec![2, 1])],
            None,
        );
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &spec).is_err());
        std::fs::remove_dir_all(nested).unwrap();

        // A `.safetensors` suffix is not evidence unless the direct artifact parses.
        std::fs::write(
            root.path().join("transformer_2/model.safetensors"),
            b"not safetensors",
        )
        .unwrap();
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &spec).is_err());

        // Mixed expert tiers are rejected even if each marker is internally plausible.
        write_component(root.path(), "transformer_2", ComponentPacking::Q8);
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &spec).is_err());
    }

    #[test]
    fn packed_tier_and_resident_bytes_come_from_tensor_geometry_not_the_marker() {
        let q4 = tempfile::tempdir().unwrap();
        write_full_snapshot(q4.path(), ComponentPacking::Q4);
        let q4_spec = LoadSpec::new(gen_core::WeightsSource::Dir(q4.path().to_owned()))
            .with_quant(gen_core::Quant::Q4);
        let (q4_contract, q4_tier) = contract_for_loaded(&q4_spec, crate::pipeline::MODEL_ID)
            .unwrap()
            .unwrap();
        assert_eq!(q4_tier.quant, Some(gen_core::Quant::Q4));
        assert_eq!(q4_contract.asset_facts.transformer_bytes, 320);

        let q8 = tempfile::tempdir().unwrap();
        write_full_snapshot(q8.path(), ComponentPacking::Q8);
        let q8_spec = LoadSpec::new(gen_core::WeightsSource::Dir(q8.path().to_owned()))
            .with_quant(gen_core::Quant::Q8);
        let (q8_contract, q8_tier) = contract_for_loaded(&q8_spec, crate::pipeline::MODEL_ID)
            .unwrap()
            .unwrap();
        assert_eq!(q8_tier.quant, Some(gen_core::Quant::Q8));
        assert_eq!(q8_contract.asset_facts.transformer_bytes, 544);
        assert!(
            q8_contract.asset_facts.transformer_bytes > q4_contract.asset_facts.transformer_bytes
        );

        write_component_with_marker(
            q8.path(),
            "transformer",
            ComponentPacking::Q8,
            ComponentPacking::Q4,
        );
        assert!(memory_strategy_contract(crate::pipeline::MODEL_ID, &q8_spec).is_err());
        write_component_with_marker(
            q4.path(),
            "transformer_2",
            ComponentPacking::Q4,
            ComponentPacking::Q8,
        );
        assert!(memory_strategy_contract(crate::pipeline::MODEL_ID, &q4_spec).is_err());
    }

    #[test]
    fn lazy_load_rejects_post_contract_adapter_mutation() {
        let root = tempfile::tempdir().unwrap();
        write_full_snapshot(root.path(), ComponentPacking::Q4);
        let adapter = root.path().join("adapter.safetensors");
        write_lora(&adapter, safetensors::Dtype::F16, false);
        let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()))
            .with_quant(gen_core::Quant::Q4);
        spec.adapters.push(gen_core::AdapterSpec::new(
            adapter.clone(),
            0.5,
            gen_core::AdapterKind::Lora,
        ));
        let (sealed, tier) = contract_for_loaded(&spec, crate::pipeline::MODEL_ID)
            .unwrap()
            .unwrap();
        validate_loaded_contract(crate::pipeline::MODEL_ID, &spec, &sealed, tier).unwrap();

        // Same path and still-loadable LoRA, but different bytes after provider construction. The
        // before/after guard used by the lazy component load compares against the sealed contract.
        write_lora(&adapter, safetensors::Dtype::BF16, true);
        let error =
            validate_loaded_contract(crate::pipeline::MODEL_ID, &spec, &sealed, tier).unwrap_err();
        assert!(error.to_string().contains("changed after"), "{error}");
        let refreshed = memory_strategy_contract(crate::pipeline::MODEL_ID, &spec).unwrap();
        assert_ne!(refreshed, sealed);
    }
}
