//! Candle/CUDA Kolors memory contract.
//!
//! This is intentionally a narrow contract: the registered provider owns only base T2I and the
//! single-reference edit route.  IP-Adapter and pose ControlNet have independent model identities
//! and must never borrow this base receipt or memory evidence.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, GenerationMemory, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization, PidWeights, Precision,
    Quant, TransformerComponent, WeightsSource,
};
use sha2::{Digest, Sha256};

pub const REQUEST_EVIDENCE_REVISION: &str = "kolors-candle-request-contract-v1";
const CALIBRATION_FINGERPRINT: &str = "kolors-candle-staged-chatglm-unet-f32-vae-v1";
pub const KOLORS_REPOSITORY: &str = "SceneWorks/kolors-mlx";
pub const KOLORS_REVISION: &str = "aadbd49f53b66a33ef1be09384eac409cbc44061";
pub const IP_REPOSITORY: &str = "Kwai-Kolors/Kolors-IP-Adapter-Plus";
pub const IP_REVISION: &str = "5c72aa86cd8d9d23ff406d293c5473820e09e1d9";
pub const CONTROL_REPOSITORY: &str = "Kwai-Kolors/Kolors-ControlNet-Pose";
pub const CONTROL_REVISION: &str = "83e35a8033a89d2e75044b412d0e2474111578f7";
pub const PHYSICAL_RECEIPT_PREFIX: &str = "kolors.physical.sha256:";
pub const ADAPTER_RECEIPT_PREFIX: &str = "kolors.adapters.ordered.sha256:";
pub const IP_RECEIPT_PREFIX: &str = "kolors.ip.sha256:";
pub const CONTROL_RECEIPT_PREFIX: &str = "kolors.control.sha256:";
pub const PID_RECEIPT_PREFIX: &str = "kolors.pid.sdxl.sha256:";
const IP_CACHE_NAMESPACE: &str = "ipadapter-kolors";
const CONTROL_CACHE_NAMESPACE: &str = "controlnet-kolors";

#[derive(Default)]
struct ComponentReceipt {
    bytes: u64,
    runtime_bytes: u64,
    digest: Sha256,
}

fn update_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hex_digest(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> gen_core::Result<()> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt cannot inspect {}: {error}",
            path.display()
        ))
    })?;
    if metadata.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors receipt refuses non-file source {}",
            path.display()
        )));
    }
    let mut entries = std::fs::read_dir(path)
        .map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt reads {}: {error}", path.display()))
        })?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt enumeration failed: {error}"))
        })?;
    entries.sort();
    for entry in entries {
        collect_files(&entry, files)?;
    }
    Ok(())
}

fn receipt_for(label: &str, path: &Path) -> gen_core::Result<ComponentReceipt> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors receipt component {label} is empty"
        )));
    }
    files.sort();
    let mut receipt = ComponentReceipt::default();
    update_framed(&mut receipt.digest, label.as_bytes());
    for file in files {
        let relative = file.strip_prefix(path).unwrap_or(file.as_path());
        update_framed(&mut receipt.digest, relative.to_string_lossy().as_bytes());
        let mut reader = BufReader::new(File::open(&file).map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt opens {}: {error}", file.display()))
        })?);
        let mut file_digest = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let count = reader.read(&mut buffer).map_err(|error| {
                gen_core::Error::Msg(format!("kolors receipt reads {}: {error}", file.display()))
            })?;
            if count == 0 {
                break;
            }
            receipt.bytes = receipt.bytes.saturating_add(count as u64);
            file_digest.update(&buffer[..count]);
        }
        update_framed(&mut receipt.digest, &file_digest.finalize());
        if file.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            receipt.runtime_bytes = receipt
                .runtime_bytes
                .saturating_add(f32_runtime_tensor_bytes(&file)?);
        }
    }
    Ok(receipt)
}

fn f32_runtime_tensor_bytes(path: &Path) -> gen_core::Result<u64> {
    let mut reader = BufReader::new(File::open(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt opens header {}: {error}",
            path.display()
        ))
    })?);
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt reads header {}: {error}",
            path.display()
        ))
    })?;
    let header_len = usize::try_from(u64::from_le_bytes(length)).map_err(|_| {
        gen_core::Error::Unsupported(format!(
            "kolors receipt header too large: {}",
            path.display()
        ))
    })?;
    let mut header = vec![0_u8; header_len];
    reader.read_exact(&mut header).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors receipt reads tensor index {}: {error}",
            path.display()
        ))
    })?;
    let index: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|error| {
            gen_core::Error::Msg(format!("kolors receipt parses {}: {error}", path.display()))
        })?;
    index
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .try_fold(0_u64, |total, (name, value)| {
            let dtype = value
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!("kolors tensor {name} omits dtype"))
                })?;
            let width = match dtype {
                "F64" | "I64" | "U64" => 8,
                // Every floating tensor is materialized through this provider's F32 VarBuilder.
                "F32" | "F16" | "BF16" | "I32" | "U32" => 4,
                "I16" | "U16" => 2,
                "I8" | "U8" | "BOOL" => 1,
                other => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "kolors tensor {name} uses unsupported dtype {other}"
                    )))
                }
            };
            let elements = value
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!("kolors tensor {name} omits shape"))
                })?
                .iter()
                .try_fold(1_u64, |product, dimension| {
                    dimension
                        .as_u64()
                        .and_then(|dimension| product.checked_mul(dimension))
                        .ok_or_else(|| {
                            gen_core::Error::Unsupported(format!(
                                "kolors tensor {name} shape overflows"
                            ))
                        })
                })?;
            total
                .checked_add(elements.saturating_mul(width))
                .ok_or_else(|| {
                    gen_core::Error::Unsupported("kolors runtime tensor bytes overflow".into())
                })
        })
}

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

fn adapter_receipt(adapters: &[AdapterSpec]) -> gen_core::Result<Option<ComponentReceipt>> {
    if adapters.is_empty() {
        return Ok(None);
    }
    let mut combined = ComponentReceipt::default();
    update_framed(&mut combined.digest, b"ordered-kolors-adapters-v1");
    for (index, adapter) in adapters.iter().enumerate() {
        let item = receipt_for("adapter", &adapter.path)?;
        combined.bytes = combined.bytes.saturating_add(item.bytes);
        combined.runtime_bytes = combined.runtime_bytes.saturating_add(item.runtime_bytes);
        update_framed(&mut combined.digest, &(index as u64).to_le_bytes());
        update_framed(
            &mut combined.digest,
            match adapter.kind {
                AdapterKind::Lora => b"lora",
                AdapterKind::Lokr => b"lokr",
            },
        );
        update_framed(&mut combined.digest, &adapter.scale.to_bits().to_le_bytes());
        update_framed(&mut combined.digest, &item.digest.finalize());
    }
    Ok(Some(combined))
}

fn pid_receipt(pid: Option<&PidWeights>) -> gen_core::Result<Option<ComponentReceipt>> {
    let Some(pid) = pid else { return Ok(None) };
    let student = receipt_for("pid-student", source_path(&pid.checkpoint))?;
    let gemma = receipt_for("pid-gemma", source_path(&pid.gemma))?;
    let mut combined = ComponentReceipt {
        bytes: student.bytes.saturating_add(gemma.bytes),
        runtime_bytes: student.runtime_bytes.saturating_add(gemma.runtime_bytes),
        ..Default::default()
    };
    update_framed(&mut combined.digest, b"kolors-pid-sdxl-v1");
    update_framed(&mut combined.digest, &student.digest.finalize());
    update_framed(&mut combined.digest, &gemma.digest.finalize());
    Ok(Some(combined))
}

fn sealed_contract(
    provider_id: &str,
    root: &Path,
    adapters: &[AdapterSpec],
    auxiliary: Option<(&str, MemoryComponentKind, &Path, &str, &str, &str)>,
    pid: Option<&PidWeights>,
) -> gen_core::Result<MemoryProviderContract> {
    let canonical = std::fs::canonicalize(root).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors cannot canonicalize source {}: {error}",
            root.display()
        ))
    })?;
    let identity = canonical.to_string_lossy();
    let repository_marker = KOLORS_REPOSITORY.replace('/', "--");
    if !identity.contains(KOLORS_REVISION)
        || !(identity.contains(KOLORS_REPOSITORY) || identity.contains(&repository_marker))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors source must resolve inside immutable {KOLORS_REPOSITORY}@{KOLORS_REVISION}"
        )));
    }
    let _ = physical_tier(root)?;
    let conditioning_tokenizer = receipt_for("kolors-tokenizer", &root.join("tokenizer"))?;
    let conditioning_text = receipt_for("kolors-chatglm", &root.join("text_encoder"))?;
    let transformer = receipt_for("kolors-unet", &root.join("unet"))?;
    let decoder = receipt_for("kolors-vae-f32", &root.join("vae"))?;
    let conditioning_bytes = conditioning_text.runtime_bytes;
    if conditioning_bytes == 0 || transformer.runtime_bytes == 0 || decoder.runtime_bytes == 0 {
        return Err(gen_core::Error::Unsupported(
            "kolors receipt has an empty runtime component inventory".into(),
        ));
    }
    let mut physical = Sha256::new();
    update_framed(&mut physical, KOLORS_REPOSITORY.as_bytes());
    update_framed(&mut physical, KOLORS_REVISION.as_bytes());
    update_framed(&mut physical, &conditioning_tokenizer.digest.finalize());
    update_framed(&mut physical, &conditioning_text.digest.finalize());
    update_framed(&mut physical, &transformer.digest.finalize());
    update_framed(&mut physical, &decoder.digest.finalize());
    let mut components = vec![MemoryResidentComponent {
        id: format!("{PHYSICAL_RECEIPT_PREFIX}{}", hex_digest(physical)),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes: transformer.runtime_bytes,
        bounded_by: Some(MemoryStrategy::StagedResidency),
        residency: MemoryComponentResidency::WholeRender,
    }];
    let mut overlay_bytes = 0_u64;
    if let Some(adapter) = adapter_receipt(adapters)? {
        overlay_bytes = overlay_bytes.saturating_add(adapter.runtime_bytes);
        components.push(MemoryResidentComponent {
            id: format!("{ADAPTER_RECEIPT_PREFIX}{}", hex_digest(adapter.digest)),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: adapter.runtime_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    if let Some((prefix, kind, path, repository, revision, cache_namespace)) = auxiliary {
        let canonical =
            validate_immutable_component_source(path, repository, revision, cache_namespace)?;
        let receipt = receipt_for(prefix, &canonical)?;
        overlay_bytes = overlay_bytes.saturating_add(receipt.runtime_bytes);
        components.push(MemoryResidentComponent {
            id: format!("{prefix}{}", hex_digest(receipt.digest)),
            kind,
            resident_bytes: receipt.runtime_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    if let Some(pid) = pid_receipt(pid)? {
        overlay_bytes = overlay_bytes.saturating_add(pid.runtime_bytes);
        components.push(MemoryResidentComponent {
            id: format!("{PID_RECEIPT_PREFIX}{}", hex_digest(pid.digest)),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: pid.runtime_bytes,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    let mut contract = provider_contract_for(provider_id);
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: components,
    };
    contract.asset_facts = MemoryAssetFacts {
        conditioning_bytes,
        transformer_bytes: transformer.runtime_bytes,
        decoder_bytes: decoder.runtime_bytes,
        base_bytes: conditioning_bytes
            .saturating_add(transformer.runtime_bytes)
            .saturating_add(decoder.runtime_bytes),
        overlay_bytes,
    };
    Ok(contract)
}

fn validate_immutable_component_source(
    path: &Path,
    repository: &str,
    revision: &str,
    cache_namespace: &str,
) -> gen_core::Result<PathBuf> {
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "kolors cannot canonicalize component source {}: {error}",
            path.display()
        ))
    })?;
    let identity = canonical.to_string_lossy();
    let repository_marker = repository.replace('/', "--");
    let repository_name = repository.rsplit('/').next().unwrap_or(repository);
    let immutable_repository_source = identity.contains(repository)
        || identity.contains(&repository_marker)
        || identity.contains(repository_name);
    let immutable_app_cache = identity.contains(cache_namespace);
    if !identity.contains(revision) || !(immutable_repository_source || immutable_app_cache) {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors component source must resolve inside immutable {repository}@{revision}"
        )));
    }
    Ok(canonical)
}

pub fn provider_contract_for_load(
    root: &Path,
    adapters: &[AdapterSpec],
    pid: Option<&PidWeights>,
) -> gen_core::Result<MemoryProviderContract> {
    sealed_contract(crate::MODEL_ID, root, adapters, None, pid)
}

pub fn provider_contract_for_ip(
    paths: &crate::IpAdapterKolorsPaths,
) -> gen_core::Result<MemoryProviderContract> {
    sealed_contract(
        "candle_kolors_ipadapter",
        &paths.kolors_base,
        &paths.adapters,
        Some((
            IP_RECEIPT_PREFIX,
            MemoryComponentKind::IpAdapter,
            &paths.ip_adapter,
            IP_REPOSITORY,
            IP_REVISION,
            IP_CACHE_NAMESPACE,
        )),
        None,
    )
}

pub fn provider_contract_for_control(
    paths: &crate::KolorsControlPaths,
    pid: Option<&PidWeights>,
) -> gen_core::Result<MemoryProviderContract> {
    sealed_contract(
        "candle_kolors_control",
        &paths.kolors_base,
        &paths.adapters,
        Some((
            CONTROL_RECEIPT_PREFIX,
            MemoryComponentKind::ControlBranch,
            &paths.controlnet,
            CONTROL_REPOSITORY,
            CONTROL_REVISION,
            CONTROL_CACHE_NAMESPACE,
        )),
        pid,
    )
}

pub fn provider_overlay_identity(contract: &MemoryProviderContract) -> Option<String> {
    let prefixes = [
        ADAPTER_RECEIPT_PREFIX,
        IP_RECEIPT_PREFIX,
        CONTROL_RECEIPT_PREFIX,
        PID_RECEIPT_PREFIX,
    ];
    let identities = contract
        .resident_components()
        .iter()
        .filter(|component| {
            prefixes
                .iter()
                .any(|prefix| component.id.starts_with(prefix))
        })
        .map(|component| component.id.as_str())
        .collect::<Vec<_>>();
    (!identities.is_empty()).then(|| identities.join("+"))
}

pub fn provider_contract() -> MemoryProviderContract {
    provider_contract_for(crate::MODEL_ID)
}

/// Bespoke IP-Adapter and ControlNet routes deliberately mint independent contracts; they may share
/// a physical Kolors base but never a provider/evidence identity.
pub fn provider_contract_for(provider_id: &str) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.lifecycle = MemoryLifecycleCapabilities {
        // Edit has a VAE-init phase, but it is deliberately represented by the decoder/VAE phase:
        // no additional advertised optimization rung is claimed for it.
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        gen_core::LoadShape::EagerMaterialization,
    ));
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention
            | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }
    contract
}

/// The selected numeric tier comes from the physical component headers/configuration, never the
/// requested UI quantization label.  q4/q8 are only valid when both ChatGLM and UNet agree.
pub fn physical_tier(root: &Path) -> gen_core::Result<MemoryNumericTier> {
    let text = crate::pipeline::detect_packed_group(&root.join("text_encoder/config.json"))
        .map_err(gen_core::Error::backend)?;
    let unet = crate::pipeline::detect_packed_group(&root.join("unet/config.json"))
        .map_err(gen_core::Error::backend)?;
    let quant = match (text, unet) {
        (None, None) => None,
        (Some(text_group), Some(unet_group)) => {
            if text_group != unet_group {
                return Err(gen_core::Error::Unsupported(format!(
                    "kolors: packed ChatGLM group {text_group} does not match packed UNet group {unet_group}"
                )));
            }
            let packed_bits = |component: &str| -> gen_core::Result<u64> {
                let config =
                    std::fs::read(root.join(component).join("config.json")).map_err(|error| {
                        gen_core::Error::Msg(format!(
                            "kolors: read physical {component} tier: {error}"
                        ))
                    })?;
                let value: serde_json::Value =
                    serde_json::from_slice(&config).map_err(|error| {
                        gen_core::Error::Msg(format!(
                            "kolors: parse physical {component} tier: {error}"
                        ))
                    })?;
                value
                    .pointer("/quantization/bits")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| {
                        gen_core::Error::Unsupported(format!(
                            "kolors: packed {component} lacks quantization.bits"
                        ))
                    })
            };
            let bits = packed_bits("unet")?;
            if packed_bits("text_encoder")? != bits {
                return Err(gen_core::Error::Unsupported(
                    "kolors: packed ChatGLM and UNet bit widths differ".into(),
                ));
            }
            match bits {
                4 => Some(Quant::Q4),
                8 => Some(Quant::Q8),
                _ => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "kolors: unsupported physical packed width {bits}"
                    )))
                }
            }
        }
        _ => {
            return Err(gen_core::Error::Unsupported(
                "kolors: mixed dense/packed ChatGLM and UNet artifacts are refused".into(),
            ))
        }
    };
    // The canonical bf16 snapshot executes Candle's exact F32 loader recipe and F32 VAE.  The
    // numeric identity stays BF16 (physical bytes), while execution precision is documented by the
    // provider's receipt/phase implementation rather than pretending the source artifact is F32.
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> gen_core::Result<()> {
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: evidence revision {} does not match {}",
            context.evidence_revision, REQUEST_EVIDENCE_REVISION
        )));
    }
    let route = || match context.mode {
        MemoryMode::TextToImage
            if !context.has_reference && context.geometry.reference_count == 0 =>
        {
            Ok(())
        }
        MemoryMode::Edit | MemoryMode::ImageToImage
            if context.has_reference && context.geometry.reference_count == 1 =>
        {
            Ok(())
        }
        _ => Err(gen_core::Error::Unsupported(
            "kolors: memory evidence is bound only to base T2I or one-reference edit".into(),
        )),
    };
    match gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(tier),
        Some(&route),
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn validate_bespoke_context(
    contract: &MemoryProviderContract,
    root: &Path,
    context: &MemoryRunContext,
    require_reference: bool,
    require_pid: bool,
) -> gen_core::Result<()> {
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: evidence revision {} does not match {}",
            contract.provider_id, context.evidence_revision, REQUEST_EVIDENCE_REVISION
        )));
    }
    if context.has_reference != require_reference || context.use_pid != require_pid {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed bespoke reference/PiD receipt",
            contract.provider_id
        )));
    }
    if context.overlay != provider_overlay_identity(contract) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed exact overlay/adapter receipt",
            contract.provider_id
        )));
    }
    let route_matches = match contract.provider_id.as_str() {
        "candle_kolors_ipadapter" => {
            context.mode == MemoryMode::Other("character_image".to_owned())
                && context.geometry.reference_count == 1
        }
        "candle_kolors_control" => {
            (context.mode == MemoryMode::TextToImage
                || matches!(&context.mode, MemoryMode::Other(mode)
                    if mode == "style_variations" || mode == "character_image"))
                && context.geometry.reference_count == 0
        }
        _ => false,
    };
    if !route_matches {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed bespoke mode/reference receipt",
            contract.provider_id
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(physical_tier(root)?),
        None,
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn request_memory(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Option<GenerationMemory> {
    contract.generation_memory(&context.selection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterSpec, MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryOptimizationAuthority,
        MemorySelection,
    };

    fn packed(bits: u8, group: u32) -> String {
        format!(r#"{{"quantization":{{"bits":{bits},"group_size":{group}}}}}"#)
    }

    #[test]
    fn only_resident_and_staged_are_advertised() {
        let contract = provider_contract();
        for capability in contract.strategies {
            assert_eq!(
                capability.support,
                if matches!(
                    capability.strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                ) {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        assert!(contract.lifecycle.synchronized_phase_release);
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn physical_tier_requires_a_matching_packed_pair() {
        let temp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "unet"] {
            let dir = temp.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), packed(4, 64)).unwrap();
        }
        assert_eq!(physical_tier(temp.path()).unwrap().quant, Some(Quant::Q4));

        std::fs::write(temp.path().join("text_encoder/config.json"), packed(8, 64)).unwrap();
        assert!(physical_tier(temp.path()).is_err());
        std::fs::write(temp.path().join("text_encoder/config.json"), packed(4, 32)).unwrap();
        assert!(physical_tier(temp.path()).is_err());
    }

    #[test]
    fn dense_tier_is_not_relabelled_as_requested_quantization() {
        let temp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "unet"] {
            let dir = temp.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), "{}").unwrap();
        }
        let tier = physical_tier(temp.path()).unwrap();
        assert_eq!(tier.precision, Precision::Bf16);
        assert_eq!(tier.quant, None);
    }

    fn write_tensor(path: &Path, dtype: &str, shape: &[u64]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let elements = shape.iter().product::<u64>();
        let source_width = if dtype == "BF16" { 2 } else { 4 };
        let header = serde_json::json!({
            "weight": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, elements * source_width]
            }
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + (elements * source_width) as usize, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn canonical_base(temp: &tempfile::TempDir) -> PathBuf {
        let root = temp
            .path()
            .join("models--SceneWorks--kolors-mlx")
            .join("snapshots")
            .join(KOLORS_REVISION)
            .join("q4");
        for component in ["text_encoder", "unet", "vae"] {
            write_tensor(
                &root.join(component).join("model.safetensors"),
                if component == "unet" { "U32" } else { "BF16" },
                &[2, 3],
            );
            std::fs::write(
                root.join(component).join("config.json"),
                if component == "vae" {
                    "{}"
                } else {
                    r#"{"quantization":{"bits":4,"group_size":64}}"#
                },
            )
            .unwrap();
        }
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
        root
    }

    fn context(provider: &str, strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            optimization_authority: if strategy == MemoryStrategy::Resident {
                MemoryOptimizationAuthority::Resident
            } else {
                MemoryOptimizationAuthority::Estimated
            },
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: CALIBRATION_FINGERPRINT.to_owned(),
            mode: if provider == "candle_kolors_ipadapter" {
                MemoryMode::Other("character_image".into())
            } else {
                MemoryMode::TextToImage
            },
            load_shape: gen_core::LoadShape::EagerMaterialization,
            has_reference: provider == "candle_kolors_ipadapter",
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: u32::from(provider == "candle_kolors_ipadapter"),
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
        }
    }

    #[test]
    fn bespoke_receipts_bind_runtime_bytes_route_and_ordered_adapters() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let ip = temp.path().join(IP_CACHE_NAMESPACE).join(IP_REVISION);
        write_tensor(
            &ip.join("ip_adapter_plus_general.safetensors"),
            "BF16",
            &[3, 5],
        );
        write_tensor(&ip.join("image_encoder/model.safetensors"), "BF16", &[7, 2]);
        let adapter_a = temp.path().join("a.safetensors");
        let adapter_b = temp.path().join("b.safetensors");
        write_tensor(&adapter_a, "BF16", &[2, 2]);
        write_tensor(&adapter_b, "BF16", &[3, 2]);
        let adapters = vec![
            AdapterSpec::new(adapter_a.clone(), 0.25, AdapterKind::Lora),
            AdapterSpec::new(adapter_b.clone(), 0.75, AdapterKind::Lokr),
        ];
        let paths = crate::IpAdapterKolorsPaths {
            kolors_base: base.clone(),
            ip_adapter: ip,
            adapters: adapters.clone(),
        };
        let contract = provider_contract_for_ip(&paths).unwrap();
        assert_eq!(contract.provider_id, "candle_kolors_ipadapter");
        assert_eq!(contract.asset_facts.conditioning_bytes, 24);
        assert_eq!(contract.asset_facts.transformer_bytes, 24);
        assert_eq!(contract.asset_facts.decoder_bytes, 24);
        assert!(contract
            .resident_components()
            .iter()
            .any(|component| component.id.starts_with(IP_RECEIPT_PREFIX)));
        let mut staged = context("candle_kolors_ipadapter", MemoryStrategy::StagedResidency);
        staged.overlay = provider_overlay_identity(&contract);
        validate_bespoke_context(&contract, &base, &staged, true, false).unwrap();

        let mut reversed = paths.clone();
        reversed.adapters.reverse();
        let reversed = provider_contract_for_ip(&reversed).unwrap();
        let adapter_id = |contract: &MemoryProviderContract| {
            contract
                .resident_components()
                .iter()
                .find(|component| component.id.starts_with(ADAPTER_RECEIPT_PREFIX))
                .unwrap()
                .id
                .clone()
        };
        assert_ne!(adapter_id(&contract), adapter_id(&reversed));
    }

    #[test]
    fn bespoke_context_refuses_crossed_provider_mode_reference_and_revision() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let control = temp
            .path()
            .join(CONTROL_CACHE_NAMESPACE)
            .join(CONTROL_REVISION)
            .join("control.safetensors");
        write_tensor(&control, "BF16", &[4, 4]);
        let paths = crate::KolorsControlPaths {
            kolors_base: base.clone(),
            controlnet: control,
            adapters: vec![],
        };
        let contract = provider_contract_for_control(&paths, None).unwrap();
        let mut crossed = context("candle_kolors_control", MemoryStrategy::StagedResidency);
        crossed.overlay = provider_overlay_identity(&contract);
        crossed.has_reference = true;
        crossed.geometry.reference_count = 1;
        assert!(validate_bespoke_context(&contract, &base, &crossed, false, false).is_err());
        crossed.has_reference = false;
        crossed.geometry.reference_count = 0;
        crossed.mode = MemoryMode::Other("character_image".into());
        assert!(validate_bespoke_context(&contract, &base, &crossed, false, false).is_ok());
        crossed.evidence_revision = "crossed-family".into();
        assert!(validate_bespoke_context(&contract, &base, &crossed, false, false).is_err());
    }

    #[test]
    fn mutable_or_arbitrary_base_source_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let arbitrary = temp.path().join("kolors-main");
        for component in ["text_encoder", "unet", "vae"] {
            write_tensor(
                &arbitrary.join(component).join("model.safetensors"),
                "BF16",
                &[1],
            );
        }
        std::fs::create_dir_all(arbitrary.join("tokenizer")).unwrap();
        std::fs::write(arbitrary.join("tokenizer/tokenizer.json"), "{}").unwrap();
        assert!(provider_contract_for_load(&arbitrary, &[], None).is_err());
    }

    #[test]
    fn arbitrary_or_crossed_bespoke_component_source_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let base = canonical_base(&temp);
        let arbitrary = temp.path().join("control-main.safetensors");
        write_tensor(&arbitrary, "BF16", &[4, 4]);
        let arbitrary_paths = crate::KolorsControlPaths {
            kolors_base: base.clone(),
            controlnet: arbitrary,
            adapters: vec![],
        };
        assert!(provider_contract_for_control(&arbitrary_paths, None).is_err());

        let crossed = temp
            .path()
            .join(IP_CACHE_NAMESPACE)
            .join(IP_REVISION)
            .join("control.safetensors");
        write_tensor(&crossed, "BF16", &[4, 4]);
        let crossed_paths = crate::KolorsControlPaths {
            kolors_base: base,
            controlnet: crossed,
            adapters: vec![],
        };
        assert!(provider_contract_for_control(&crossed_paths, None).is_err());
    }
}
