//! Provider-owned resident memory facts for Krea Realtime I2V (SC-20770).
//!
//! Krea's currently selectable ladder is intentionally resident-only. The bounded causal cache,
//! automatic VAE tiling, and loader staging are unconditional implementation details, not request
//! strategies, so they remain `Missing`. This module nevertheless publishes exact, non-zero facts
//! before construction so the caller can reject an impossible resident load instead of discovering
//! it after Metal allocation has begun.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use mlx_gen::gen_core::{
    default_memory_strategy_safety_check, MemoryBackendRealization, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryPhase, MemoryProviderContract, MemoryRegistration,
    MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, ResidentOnlyMemoryContractRegistration,
};
use mlx_gen::{Error, LoadSpec, Result, WeightsSource};
use sha2::{Digest, Sha256};

use crate::MODEL_ID;

const CONFIG: &str = "config.json";
const TEXT_ENCODER: &str = "t5_encoder.safetensors";
const TOKENIZER: &str = "tokenizer.json";
const VAE: &str = "vae.safetensors";
const DIT: &str = "dit.safetensors";
const TRANSFORMER: &str = "transformer";
const FINGERPRINT_PREFIX: &str = "krea-realtime-i2v-resident-v2";
pub const CANONICAL_REPOSITORY: &str = "SceneWorks/krea-realtime-14b-mlx";
pub const CANONICAL_REVISION: &str = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626";
const PACKED_FILES: &[&str] = &[CONFIG, DIT, TEXT_ENCODER, TOKENIZER, VAE];
const BF16_FILES: &[&str] = &[
    CONFIG,
    TEXT_ENCODER,
    TOKENIZER,
    VAE,
    "transformer/dit-00001-of-00007.safetensors",
    "transformer/dit-00002-of-00007.safetensors",
    "transformer/dit-00003-of-00007.safetensors",
    "transformer/dit-00004-of-00007.safetensors",
    "transformer/dit-00005-of-00007.safetensors",
    "transformer/dit-00006-of-00007.safetensors",
    "transformer/dit-00007-of-00007.safetensors",
];

#[derive(Debug)]
struct PhysicalFile {
    logical_bytes: u64,
    content_digest: String,
}

fn root(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(path) => Ok(path),
        WeightsSource::File(_) => Err(Error::Unsupported(
            "krea_realtime_14b resident facts require the exact tier directory".to_owned(),
        )),
    }
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).map_err(|error| Error::Msg(format!("{}: {error}", dir.display())))?
    {
        let entry = entry.map_err(|error| Error::Msg(error.to_string()))?;
        let path = entry.path();
        let ty = entry
            .file_type()
            .map_err(|error| Error::Msg(format!("{}: {error}", path.display())))?;
        if ty.is_dir() {
            collect_files(root, &path, files)?;
        } else if ty.is_file() || ty.is_symlink() {
            let relative = path.strip_prefix(root).map_err(|_| {
                Error::Unsupported(format!("{MODEL_ID}: inventory escaped tier root"))
            })?;
            if !path.is_file() {
                return Err(Error::Unsupported(format!(
                    "{MODEL_ID}: inventory entry is not a readable file: {}",
                    path.display()
                )));
            }
            files.push(relative.to_string_lossy().replace('\\', "/"));
        } else {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: unsupported inventory entry: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn required_direct_files(root: &Path, resolved_tier: &str) -> Result<Vec<PathBuf>> {
    let expected = if resolved_tier == "bf16" {
        BF16_FILES
    } else {
        PACKED_FILES
    };
    let mut actual = Vec::new();
    collect_files(root, root, &mut actual)?;
    actual.sort();
    let mut expected_sorted = expected
        .iter()
        .map(|item| (*item).to_owned())
        .collect::<Vec<_>>();
    expected_sorted.sort();
    if actual != expected_sorted {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: incomplete or extra tier inventory (expected {expected_sorted:?}, got {actual:?})"
        )));
    }
    Ok(expected
        .iter()
        .map(|relative| root.join(relative))
        .collect())
}

fn dtype_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

fn inspect_safetensors(path: &Path) -> Result<PhysicalFile> {
    let mut file =
        File::open(path).map_err(|error| Error::Msg(format!("{}: {error}", path.display())))?;
    let size = file
        .metadata()
        .map_err(|error| Error::Msg(format!("{}: {error}", path.display())))?
        .len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).map_err(|error| {
        Error::Unsupported(format!(
            "{MODEL_ID}: invalid safetensors {}: {error}",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(prefix);
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| Error::Unsupported(format!("{MODEL_ID}: oversized safetensors header")))?;
    if header_len == 0
        || header_len > 100_000_000
        || 8_u64.checked_add(header_len).is_none_or(|n| n > size)
    {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: invalid safetensors header length in {}",
            path.display()
        )));
    }
    let mut header = vec![0_u8; header_len_usize];
    file.read_exact(&mut header).map_err(|error| {
        Error::Unsupported(format!(
            "{MODEL_ID}: truncated safetensors header {}: {error}",
            path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
        Error::Unsupported(format!(
            "{MODEL_ID}: invalid safetensors header {}: {error}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        Error::Unsupported(format!(
            "{MODEL_ID}: safetensors header is not an object: {}",
            path.display()
        ))
    })?;
    let mut tensors = Vec::new();
    for (name, info) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .and_then(dtype_bytes)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{MODEL_ID}: invalid tensor dtype in {}",
                    path.display()
                ))
            })?;
        let elements = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{MODEL_ID}: invalid tensor shape in {}",
                    path.display()
                ))
            })?
            .iter()
            .try_fold(1_u64, |product, dim| product.checked_mul(dim.as_u64()?))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{MODEL_ID}: tensor shape overflow in {}",
                    path.display()
                ))
            })?;
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{MODEL_ID}: invalid tensor offsets in {}",
                    path.display()
                ))
            })?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: invalid tensor offset")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: invalid tensor offset")))?;
        let logical = elements
            .checked_mul(dtype)
            .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: tensor byte overflow")))?;
        if end.checked_sub(start) != Some(logical) || logical == 0 {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: tensor geometry does not match packed data in {}",
                path.display()
            )));
        }
        tensors.push((start, end, logical));
    }
    tensors.sort_by_key(|tensor| tensor.0);
    if tensors.is_empty()
        || tensors
            .iter()
            .scan(0_u64, |cursor, (start, end, _)| {
                let valid = *start == *cursor;
                *cursor = *end;
                Some(valid)
            })
            .any(|valid| !valid)
    {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: empty or non-contiguous safetensors geometry in {}",
            path.display()
        )));
    }
    let payload = tensors.last().map(|tensor| tensor.1).unwrap_or(0);
    if 8_u64
        .checked_add(header_len)
        .and_then(|n| n.checked_add(payload))
        != Some(size)
    {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: safetensors inventory length does not match its header: {}",
            path.display()
        )));
    }
    let logical_bytes = tensors
        .iter()
        .try_fold(0_u64, |sum, tensor| sum.checked_add(tensor.2))
        .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: logical tensor byte overflow")))?;
    let content_digest = content_digest(path)?;
    Ok(PhysicalFile {
        logical_bytes,
        content_digest,
    })
}

fn content_digest(path: &Path) -> Result<String> {
    let file =
        File::open(path).map_err(|error| Error::Msg(format!("{}: {error}", path.display())))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut chunk)
            .map_err(|error| Error::Msg(format!("{}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn inspect_file(path: &Path) -> Result<PhysicalFile> {
    if path.extension().is_some_and(|ext| ext == "safetensors") {
        inspect_safetensors(path)
    } else {
        let size = std::fs::metadata(path)
            .map_err(|error| Error::Msg(format!("{}: {error}", path.display())))?
            .len();
        if size == 0 {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: direct artifact file is empty: {}",
                path.display()
            )));
        }
        Ok(PhysicalFile {
            logical_bytes: 0,
            content_digest: content_digest(path)?,
        })
    }
}

fn tier(spec: &LoadSpec, root: &Path) -> Result<&'static str> {
    let config = crate::KreaRealtimeConfig::from_model_dir(root)?;
    match (
        config.wan.quantization.map(|quant| quant.bits),
        spec.quantize,
    ) {
        (Some(4), None | Some(mlx_gen::Quant::Q4)) => Ok("q4"),
        (Some(8), None | Some(mlx_gen::Quant::Q8)) => Ok("q8"),
        (Some(bits), _) => Err(Error::Unsupported(format!(
            "{MODEL_ID}: unsupported or crossed packed tier Q{bits}"
        ))),
        (None, Some(mlx_gen::Quant::Q4)) => Ok("q4"),
        (None, Some(mlx_gen::Quant::Q8)) => Ok("q8"),
        (None, None) => Ok("bf16"),
        (None, Some(other)) => Err(Error::Unsupported(format!(
            "{MODEL_ID}: unsupported load-time tier {other:?}"
        ))),
    }
}

/// Provider-owned numeric tier for the exact prepared Krea directory. Packed q4/q8 snapshots carry
/// `quantize=None` in the load spec, so worker admission must not infer bf16 from that field.
pub fn resolved_numeric_tier(spec: &LoadSpec) -> Result<mlx_gen::gen_core::MemoryNumericTier> {
    let root = root(spec)?;
    let quant = match tier(spec, root)? {
        "q4" => Some(mlx_gen::Quant::Q4),
        "q8" => Some(mlx_gen::Quant::Q8),
        "bf16" => None,
        _ => unreachable!("tier() returns only the canonical matrix"),
    };
    Ok(mlx_gen::gen_core::MemoryNumericTier {
        precision: mlx_gen::Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

/// Canonical receipt for the exact tier/direct-file/adapter load identity.
///
/// The generator cache separately retains filesystem replacement tokens. This digest is the
/// provider-facing, telemetry-safe spelling: canonical repository/revision, complete inventory,
/// content digests, validated tensor geometry, resolved tier, adapter receipt, and contract version.
pub fn canonical_artifact_identity(spec: &LoadSpec) -> Result<String> {
    canonical_artifact_identity_for_source(spec, CANONICAL_REPOSITORY, CANONICAL_REVISION)
}

fn canonical_artifact_identity_for_source(
    spec: &LoadSpec,
    repository: &str,
    revision: &str,
) -> Result<String> {
    if repository != CANONICAL_REPOSITORY || revision != CANONICAL_REVISION {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: crossed canonical repository or revision"
        )));
    }
    spec.validate_prepared_file_pins()?;
    let root = root(spec)?;
    let resolved_tier = tier(spec, root)?;
    let mut direct = required_direct_files(root, resolved_tier)?;
    direct.sort();
    let mut digest = Sha256::new();
    digest.update(FINGERPRINT_PREFIX.as_bytes());
    digest.update(repository.as_bytes());
    digest.update(revision.as_bytes());
    digest.update(resolved_tier.as_bytes());
    for file in direct {
        let relative = file.strip_prefix(root).map_err(|_| {
            Error::Unsupported(format!("{MODEL_ID}: direct file escaped its tier root"))
        })?;
        let physical = inspect_file(&file)?;
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update(physical.content_digest.as_bytes());
        digest.update(physical.logical_bytes.to_le_bytes());
    }
    for adapter in &spec.adapters {
        let pin = spec.file_pin_for(&adapter.path)?;
        pin.ensure_unchanged()?;
        let physical = inspect_safetensors(&adapter.path)?;
        digest.update(pin.loader_path().to_string_lossy().as_bytes());
        digest.update(physical.content_digest.as_bytes());
        digest.update(physical.logical_bytes.to_le_bytes());
        digest.update(adapter.scale.to_bits().to_le_bytes());
        digest.update(
            format!(
                "{:?}:{:?}:{:?}",
                adapter.kind, adapter.pass_scales, adapter.moe_expert
            )
            .as_bytes(),
        );
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Exact resident-only contract, resolved before provider construction from the same direct files
/// the loader opens. Zero or crossed facts are errors, never a headroom-only candidate.
pub fn memory_strategy_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    let root = root(spec)?;
    let resolved_tier = tier(spec, root)?;
    let direct = required_direct_files(root, resolved_tier)?;
    let transformer_bytes = direct
        .iter()
        .filter(|file| {
            file.file_name().is_some_and(|name| name == DIT)
                || file
                    .parent()
                    .is_some_and(|parent| parent.ends_with(TRANSFORMER))
        })
        .try_fold(0_u64, |total, file| {
            total
                .checked_add(inspect_safetensors(file)?.logical_bytes)
                .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: transformer byte overflow")))
        })?;
    let conditioning_bytes = inspect_safetensors(&root.join(TEXT_ENCODER))?.logical_bytes;
    let decoder_bytes = inspect_safetensors(&root.join(VAE))?.logical_bytes;
    let base_bytes = conditioning_bytes
        .checked_add(transformer_bytes)
        .and_then(|bytes| bytes.checked_add(decoder_bytes))
        .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: resident byte sum overflow")))?;
    if base_bytes == 0 || conditioning_bytes == 0 || transformer_bytes == 0 || decoder_bytes == 0 {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: incomplete zero resident asset facts"
        )));
    }
    let overlay_bytes = spec.adapters.iter().try_fold(0_u64, |total, adapter| {
        total
            .checked_add(inspect_safetensors(&adapter.path)?.logical_bytes)
            .ok_or_else(|| Error::Unsupported(format!("{MODEL_ID}: adapter byte overflow")))
    })?;
    // Structural artifact identity is intentionally not written into `calibration`: that field
    // means a measured memory campaign. The registry/worker carry the immutable artifact receipt
    // separately while this contract remains truthfully estimate-backed.
    let _identity = canonical_artifact_identity(spec)?;
    contract_from_asset_facts(
        spec,
        mlx_gen::gen_core::MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
    )
}

fn contract_from_asset_facts(
    spec: &LoadSpec,
    asset_facts: mlx_gen::gen_core::MemoryAssetFacts,
) -> Result<MemoryProviderContract> {
    let mut contract = MemoryProviderContract::compatibility_default(
        MODEL_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: false,
            cache_eviction: false,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.calibration = None;
    contract.asset_facts = asset_facts;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        ..Default::default()
    };
    let mut resident_components = Vec::new();
    if contract.asset_facts.overlay_bytes > 0 {
        resident_components.push(MemoryResidentComponent {
            id: "adapter_stack".to_owned(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: contract.asset_facts.overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components,
    };
    for capability in &mut contract.strategies {
        capability.support = if capability.strategy == MemoryStrategy::Resident {
            MemoryStrategySupport::Implemented
        } else {
            MemoryStrategySupport::Missing
        };
    }
    let errors = contract.conformance_errors();
    if !errors.is_empty() {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: malformed resident contract: {}",
            errors.join("; ")
        )));
    }
    Ok(contract)
}

/// Weights-free registry witness for the provider's deliberately resident-only surface.
///
/// Production admission still uses [`memory_strategy_contract`] and its exact physical file facts.
/// This witness carries no asset magnitudes: it exists only so catalog reconciliation can prove,
/// for every shipped bf16/q4/q8 selector, that Krea Realtime exposes no optimized rung by omission.
pub(crate) fn weights_free_resident_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    if !matches!(
        spec.quantize,
        None | Some(mlx_gen::Quant::Q4) | Some(mlx_gen::Quant::Q8)
    ) {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: unsupported weights-free resident tier"
        )));
    }
    contract_from_asset_facts(spec, mlx_gen::gen_core::MemoryAssetFacts::default())
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let Ok(expected) = memory_strategy_contract(spec) else {
        return MemorySafetyDecision::Reject {
            reason: format!("{MODEL_ID}: resident artifact receipt no longer resolves"),
        };
    };
    if &expected != contract
        || context.mode.as_key() != "image_to_video"
        || context.geometry.reference_count != 1
        || canonical_artifact_identity(spec).is_err()
    {
        return MemorySafetyDecision::Reject {
            reason: format!("{MODEL_ID}: crossed resident artifact/mode/reference receipt"),
        };
    }
    default_memory_strategy_safety_check(contract, context)
}

pub(crate) fn loaded_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    loaded_artifact_identity: &str,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let Ok(current_identity) = canonical_artifact_identity(spec) else {
        return MemorySafetyDecision::Reject {
            reason: format!("{MODEL_ID}: resident physical receipt no longer resolves"),
        };
    };
    if current_identity != loaded_artifact_identity
        || !context
            .evidence_revision
            .contains(&format!(":artifact={loaded_artifact_identity}:"))
    {
        return MemorySafetyDecision::Reject {
            reason: format!("{MODEL_ID}: crossed or mutated physical artifact receipt"),
        };
    }
    safety_check(spec, contract, context)
}

fn registry_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    memory_strategy_contract(spec).map_err(|error| mlx_gen::gen_core::Error::Msg(error.to_string()))
}

fn registry_resident_witness(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    weights_free_resident_contract(spec)
        .map_err(|error| mlx_gen::gen_core::Error::Msg(error.to_string()))
}

pub const REGISTRATION: MemoryRegistration = MemoryRegistration {
    provider_id: MODEL_ID,
    contract: registry_contract,
    safety_check,
};

pub const RESIDENT_ONLY_WITNESS: ResidentOnlyMemoryContractRegistration =
    ResidentOnlyMemoryContractRegistration {
        provider_id: MODEL_ID,
        contract: registry_resident_witness,
        surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::Quant;

    fn tensor(path: &Path, bytes: u64, fill: u8) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut header = serde_json::to_vec(&serde_json::json!({
            "weight": {"dtype": "U8", "shape": [bytes], "data_offsets": [0, bytes]}
        }))
        .unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut output = Vec::with_capacity(8 + header.len() + bytes as usize);
        output.extend_from_slice(&(header.len() as u64).to_le_bytes());
        output.extend_from_slice(&header);
        output.extend(std::iter::repeat_n(fill, bytes as usize));
        std::fs::write(path, output).unwrap();
    }

    fn fixture(tier: &str, dit_bytes: u64) -> (tempfile::TempDir, LoadSpec) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join(tier);
        std::fs::create_dir_all(&root).unwrap();
        let mut config = crate::KreaRealtimeConfig::krea_realtime_14b();
        config.wan.quantization = match tier {
            "q4" => Some(mlx_gen_wan::config::WanQuant {
                bits: 4,
                group_size: 64,
            }),
            "q8" => Some(mlx_gen_wan::config::WanQuant {
                bits: 8,
                group_size: 64,
            }),
            "bf16" => None,
            _ => unreachable!(),
        };
        std::fs::write(
            root.join(CONFIG),
            serde_json::to_vec(&config.to_json()).unwrap(),
        )
        .unwrap();
        tensor(&root.join(TEXT_ENCODER), 101, 1);
        std::fs::write(root.join(TOKENIZER), b"{}").unwrap();
        tensor(&root.join(VAE), 211, 2);
        if tier == "bf16" {
            let per_shard = dit_bytes / 7;
            for shard in 1..=7 {
                let bytes = if shard == 7 {
                    dit_bytes - per_shard * 6
                } else {
                    per_shard
                };
                tensor(
                    &root
                        .join(TRANSFORMER)
                        .join(format!("dit-{shard:05}-of-00007.safetensors")),
                    bytes,
                    shard as u8,
                );
            }
        } else {
            tensor(&root.join(DIT), dit_bytes, if tier == "q4" { 4 } else { 8 });
        }
        (temp, LoadSpec::new(WeightsSource::Dir(root)))
    }

    #[test]
    fn q4_q8_and_bf16_publish_nonzero_differentiated_exact_facts() {
        let (_q4_root, q4_spec) = fixture("q4", 400);
        let (_q8_root, q8_spec) = fixture("q8", 800);
        let (_bf16_root, bf16_spec) = fixture("bf16", 1_600);
        let contracts =
            [&q4_spec, &q8_spec, &bf16_spec].map(|spec| memory_strategy_contract(spec).unwrap());
        assert_eq!(contracts[0].asset_facts.transformer_bytes, 400);
        assert_eq!(contracts[1].asset_facts.transformer_bytes, 800);
        assert_eq!(contracts[2].asset_facts.transformer_bytes, 1_600);
        for contract in contracts {
            assert!(contract.asset_facts.base_bytes > 0);
            assert!(contract.asset_facts.conditioning_bytes > 0);
            assert!(contract.asset_facts.decoder_bytes > 0);
            assert!(
                contract.calibration.is_none(),
                "structural facts are not calibration"
            );
            assert!(matches!(
                contract
                    .capability(MemoryStrategy::Resident)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            ));
            for strategy in MemoryStrategy::ALL {
                if strategy != MemoryStrategy::Resident {
                    assert!(matches!(
                        contract.capability(strategy).unwrap().support,
                        MemoryStrategySupport::Missing
                    ));
                }
            }
        }
        assert_ne!(
            canonical_artifact_identity(&q4_spec).unwrap(),
            canonical_artifact_identity(&q8_spec).unwrap()
        );
        assert_ne!(
            canonical_artifact_identity(&q8_spec).unwrap(),
            canonical_artifact_identity(&bf16_spec).unwrap()
        );
    }

    #[test]
    fn weights_free_witness_is_resident_only_across_every_shipped_selector() {
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            let contract = weights_free_resident_contract(&surface.spec).unwrap();
            assert_eq!(contract.provider_id, MODEL_ID);
            assert_eq!(
                contract.asset_facts,
                mlx_gen::gen_core::MemoryAssetFacts::default()
            );
            assert!(contract.conformance_errors().is_empty());
            assert!(contract.strategies.iter().all(|capability| {
                if capability.strategy == MemoryStrategy::Resident {
                    capability.support == MemoryStrategySupport::Implemented
                } else {
                    capability.support == MemoryStrategySupport::Missing
                }
            }));
        }
    }

    #[test]
    fn crossed_packed_tier_and_empty_direct_file_fail_before_load() {
        let (_root, mut spec) = fixture("q4", 400);
        spec.quantize = Some(Quant::Q8);
        assert!(memory_strategy_contract(&spec)
            .unwrap_err()
            .to_string()
            .contains("crossed packed tier"));

        let (_root, spec) = fixture("q8", 800);
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        std::fs::write(root.join(VAE), []).unwrap();
        assert!(memory_strategy_contract(&spec)
            .unwrap_err()
            .to_string()
            .contains("safetensors"));
    }

    #[test]
    fn physical_receipt_rejects_same_length_mutation_crossed_source_and_inventory_drift() {
        let (_root, spec) = fixture("q4", 400);
        let identity = canonical_artifact_identity(&spec).unwrap();
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        let path = root.join(DIT);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert_ne!(identity, canonical_artifact_identity(&spec).unwrap());
        assert!(canonical_artifact_identity_for_source(
            &spec,
            "other/repository",
            CANONICAL_REVISION
        )
        .is_err());
        assert!(canonical_artifact_identity_for_source(
            &spec,
            CANONICAL_REPOSITORY,
            "other-revision"
        )
        .is_err());
        std::fs::write(root.join("extra.json"), b"{}").unwrap();
        assert!(canonical_artifact_identity(&spec).is_err());
        std::fs::remove_file(root.join("extra.json")).unwrap();
        std::fs::remove_file(root.join(VAE)).unwrap();
        assert!(canonical_artifact_identity(&spec).is_err());
    }
}
