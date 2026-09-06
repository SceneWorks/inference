//! Provider-owned resident memory facts for Krea Realtime I2V/V2V (SC-20770/SC-20771).
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
    default_memory_strategy_safety_check, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
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
/// Cross-repository ABI domain used by the provider and the SceneWorks worker when sealing the
/// same direct-file/adapter receipt. Keep this public so paired fixtures can name the authoritative
/// provider spelling instead of reviving an I2V-only predecessor.
pub const ARTIFACT_RECEIPT_DOMAIN: &str = "krea-realtime-video-resident-v3";
/// Cross-repository ABI domain for the request envelope carrying that artifact receipt.
pub const REQUEST_RECEIPT_DOMAIN: &str = "provider-resident-video-request-v3";
pub const CANONICAL_REPOSITORY: &str = "SceneWorks/krea-realtime-14b-mlx";
pub const CANONICAL_REVISION: &str = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626";
/// Static-behavior identity family published by the weights-free registry declaration surface.
///
/// This seam opens no file, so it can only describe *behavior* — the resident-only rung set this
/// provider declares for a given requested tier and residency policy. It is deliberately a
/// different family from [`production_calibration_fingerprint`], whose strings name the resident
/// ladder measured against a real artifact on disk; nothing weights-free may ever produce one of
/// those.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "krea-realtime-14b-mlx-registry-behavior-v1";
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

/// Production calibration identity table of the Krea Realtime 14B resident ladder, keyed on
/// (provider, artifact tier).
///
/// Every shipped cell is `krea-realtime-14b-<tier>-mlx-resident-ladder-v1` for `tier` in
/// `bf16`/`q4`/`q8`; there is no pre-existing measured string for this provider to preserve, so
/// the three cells are uniform. A provider id other than [`MODEL_ID`] has no cell.
///
/// Offload policy and load shape are deliberately not inputs: the identity names the artifact the
/// evidence was captured against, and [`MemoryCalibrationIdentity::load_shape`] carries the
/// materialization axis separately.
///
/// This is the table, not the binding. The `tier` here is a caller-supplied label; only
/// [`production_calibration_identity`] — which proves the tier against the packed marker in the
/// snapshot's own `config.json` — may turn one of these strings into a contract identity.
pub fn production_calibration_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    if provider_id != MODEL_ID || !matches!(tier, "bf16" | "q4" | "q8") {
        return None;
    }
    Some(format!("krea-realtime-14b-{tier}-mlx-resident-ladder-v1"))
}

/// The production identity of `spec`, bound to the tier of the artifact actually on disk.
///
/// Fail-closed in every direction, and never an error: a file-backed source, a directory that is
/// not materialized, a snapshot whose packed `quantization.bits` crosses `LoadSpec::quantize`, or
/// a non-bf16 execution precision all publish `None` rather than a string no anchor measured.
/// Building the contract itself must not become fallible on account of the identity.
///
/// The materialized-directory check is load-bearing rather than redundant with the tier proof:
/// `KreaRealtimeConfig::from_model_dir` keeps the shipped preset for a snapshot with no
/// `config.json`, and a directory that does not exist has none either — so without it a
/// nonexistent path would publish the `bf16` anchor coordinate.
///
/// The tier comes from the snapshot's own packed marker, never from `spec.quantize` alone: the
/// SceneWorks worker resolves a Krea Realtime tier by directory and leaves `LoadSpec::quantize` at
/// `None` for all three tiers (the tier on disk *is* the quantization), so keying on that field
/// would collapse q4, q8 and bf16 onto one string.
pub fn production_calibration_identity(spec: &LoadSpec) -> Option<MemoryCalibrationIdentity> {
    if spec.precision != mlx_gen::Precision::Bf16 {
        return None;
    }
    let root = mlx_gen::architecture_facts::materialized_root(spec)?;
    let resolved_tier = tier(spec, root).ok()?;
    production_calibration_fingerprint(MODEL_ID, resolved_tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

/// Per-selector static behavior identity for the weights-free declaration surface.
///
/// The tier token comes from `spec.quantize` alone because this seam touches no filesystem: there
/// is no artifact whose packed marker could be read. `MemoryProviderContract::conformance_errors`
/// requires lowercase kebab tokens, so every component spelled in is already one.
fn static_behavior_identity(spec: &LoadSpec) -> MemoryCalibrationIdentity {
    let tier = match spec.quantize {
        None => "bf16",
        Some(mlx_gen::Quant::Q4) => "q4",
        Some(mlx_gen::Quant::Q8) => "q8",
        Some(_) => "unshipped",
    };
    let policy = match spec.offload_policy {
        mlx_gen::OffloadPolicy::Resident => "resident",
        mlx_gen::OffloadPolicy::Sequential => "sequential",
    };
    MemoryCalibrationIdentity::new(
        format!("{STATIC_BEHAVIOR_FINGERPRINT}-{tier}-{policy}"),
        spec.load_shape,
    )
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
    digest.update(ARTIFACT_RECEIPT_DOMAIN.as_bytes());
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
        pin.verify_unchanged()?;
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
    // `calibration` names the anchor cell this load belongs to — the (provider, artifact tier)
    // coordinate a memory campaign captures against — not a completed campaign. Publishing it is
    // what makes the cell measurable at all: the capture arm reads the identity off the loaded
    // generator and refuses a contract without one. The immutable artifact receipt below is a
    // separate, finer-grained quantity and stays out of the identity: it changes with every byte
    // on disk, while the anchor coordinate must be stable across snapshots of the same tier.
    let _identity = canonical_artifact_identity(spec)?;
    contract_from_asset_facts(
        spec,
        production_calibration_identity(spec),
        mlx_gen::gen_core::MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
    )
}

/// Architecture axes for the Krea Realtime 14B route (epic SC-22657, E2).
///
/// [`KreaRealtimeConfig::krea_realtime_14b`](crate::config::KreaRealtimeConfig::krea_realtime_14b)
/// carries `WanModelConfig::wan21_t2v_14b` — Krea Realtime is Wan 2.1 T2V-14B weight for weight —
/// and `KreaRealtimeConfig::from_model_dir` overlays a snapshot's own `config.json` onto it at load.
///
/// The autoencoder is the Wan z16 **video** VAE, so `vae_temporal_scale` is a real value here: four
/// frames per latent unit over a x8 spatial scale.
///
/// When `spec` names a materialized snapshot directory this re-runs
/// `KreaRealtimeConfig::from_model_dir` — the loader's own parse — so the published trunk axes are
/// the snapshot's rather than the preset's. Krea Realtime typically ships without a `config.json`,
/// and `from_model_dir` keeps the shipped preset in that case, so a config-less snapshot publishes
/// exactly what the weights-free surface does.
fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let wan = mlx_gen::architecture_facts::materialized_root(spec)
        .and_then(|root| crate::config::KreaRealtimeConfig::from_model_dir(root).ok())
        .unwrap_or_else(crate::config::KreaRealtimeConfig::krea_realtime_14b)
        .wan;
    let (_, patch_h, patch_w) = wan.patch_size;
    let (temporal_stride, spatial_stride, _) = wan.vae_stride;
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(wan.num_heads),
        // The exactness-gated helper, NOT `axis(wan.head_dim())` (SC-22667): `WanModelConfig::head_dim`
        // is a plain `dim / num_heads`, so a snapshot `config.json` whose `num_heads` does not divide
        // `dim` would publish a truncated width the loader cannot build, and `"num_heads": 0` divides
        // by zero before `axis` can decline the value. This mirrors `mlx_gen_wan`'s own derivation.
        head_dim: mlx_gen::architecture_facts::head_dim(wan.dim, wan.num_heads),
        transformer_blocks: mlx_gen::architecture_facts::axis(wan.num_layers),
        // A single scalar can only describe a square patch; an anisotropic one has no honest value.
        patch_size: (patch_h == patch_w)
            .then(|| mlx_gen::architecture_facts::axis(patch_h))
            .flatten(),
        latent_channels: mlx_gen::architecture_facts::axis(wan.vae_z_dim),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(spatial_stride),
        vae_temporal_scale: mlx_gen::architecture_facts::axis(temporal_stride),
        // `convert::TRANSFORMER_DTYPE` pins the reused Wan trunk to `Dtype::Bfloat16`.
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

/// Shared contract shape for both publication seams.
///
/// `calibration` is threaded in rather than derived here because the two callers are answering
/// different questions with it: the production path names the anchor cell of the artifact it just
/// inspected, and the weights-free declaration path names its own static behavior. Deriving one
/// rule inside this builder would let a weights-free witness publish a production string.
fn contract_from_asset_facts(
    spec: &LoadSpec,
    calibration: Option<MemoryCalibrationIdentity>,
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
    contract.architecture_facts = architecture_facts(spec);
    contract.calibration = calibration;
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
///
/// It publishes a [`static_behavior_identity`], never a
/// [`production_calibration_fingerprint`]: the two families are disjoint by construction, so a
/// weights-free witness can never be mistaken for measured-cell evidence.
pub(crate) fn weights_free_resident_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    if !matches!(
        spec.quantize,
        None | Some(mlx_gen::Quant::Q4) | Some(mlx_gen::Quant::Q8)
    ) {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: unsupported weights-free resident tier"
        )));
    }
    contract_from_asset_facts(
        spec,
        Some(static_behavior_identity(spec)),
        mlx_gen::gen_core::MemoryAssetFacts::default(),
    )
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
    let (mode, shape) = match context.mode.as_key() {
        "image_to_video" => ("image_to_video", "image"),
        "video_to_video" => ("video_to_video", "video_clip"),
        _ => {
            return MemorySafetyDecision::Reject {
                reason: format!("{MODEL_ID}: crossed resident request mode"),
            }
        }
    };
    let sealed_request_prefix =
        format!("{REQUEST_RECEIPT_DOMAIN}:{MODEL_ID}:mode={mode}:shape={shape}:");
    if &expected != contract
        || context.geometry.reference_count != 1
        || !context
            .evidence_revision
            .starts_with(&sealed_request_prefix)
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

/// Registry entry point for [`MemoryRegistration::contract`].
///
/// A materialized snapshot directory resolves the exact physical contract. A spec that names no
/// materialized directory is not a load at all — it is a declaration probe, the shape catalog
/// reconciliation and the SceneWorks planned-lane walk both synthesize — and it resolves the same
/// weights-free declaration contract [`RESIDENT_ONLY_WITNESS`] publishes, so a caller that reaches
/// this registration without a fixture still gets a well-formed, identity-carrying witness instead
/// of a directory-read error.
///
/// This does not widen admission. The declaration contract carries zero asset magnitudes, and the
/// paired [`safety_check`] independently re-resolves [`memory_strategy_contract`] and
/// [`canonical_artifact_identity`] against real files before any request is admitted, so a
/// nonexistent directory is still refused at the request boundary.
fn registry_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    if mlx_gen::architecture_facts::materialized_root(spec).is_none() {
        return registry_resident_witness(spec);
    }
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

    /// AC (SC-22662): the Krea Realtime contract publishes the axes of the Wan 2.1 T2V-14B trunk
    /// and z16 video VAE it reuses, and passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_the_reused_wan_2_1_config() {
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            let contract = weights_free_resident_contract(&surface.spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(40),
                    // 5120 / 40, derived by `WanModelConfig::head_dim`.
                    head_dim: Some(128),
                    transformer_blocks: Some(40),
                    // `patch_size` is `(1, 2, 2)`: the square spatial patch is 2.
                    patch_size: Some(2),
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    // A video autoencoder: four frames per latent unit.
                    vae_temporal_scale: Some(4),
                    activation_dtype_width: Some(2),
                },
                "{} architecture facts",
                surface.selector.id()
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        // The published pair IS this provider's assigned VAE geometry.
        let preset_facts = architecture_facts(&weights_free_spec());
        assert_eq!(
            crate::VAE_TILING.spatial_scale as u32,
            preset_facts.vae_spatial_scale.unwrap()
        );
        assert_eq!(
            crate::VAE_TILING.temporal_scale as u32,
            preset_facts.vae_temporal_scale.unwrap()
        );
    }

    /// A spec whose weights directory is the registry's never-created contract-surface sentinel:
    /// the weights-free path, where the preset is the only geometry there is.
    fn weights_free_spec() -> LoadSpec {
        LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ))
    }

    fn spec_for_config(dir: &Path, config: &serde_json::Value) -> LoadSpec {
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(dir.to_path_buf()))
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the trunk axes come from the
    /// snapshot's own `config.json` — the file `KreaRealtimeConfig::from_model_dir` overlays at
    /// load — rather than from the compile-time preset. A snapshot mirroring the reference config
    /// agrees with the weights-free path; a fixture with mutated keys publishes the mutated axes,
    /// which is what the unconditional `architecture_facts()` this replaced would fail.
    ///
    /// The third arm pins the shipped shape: Krea Realtime ships transformer-only with no
    /// `config.json` at all, and such a snapshot must still publish the Wan 2.1 14B preset.
    #[test]
    fn materialized_trunk_axes_come_from_the_snapshot_rather_than_the_preset() {
        let preset = crate::config::KreaRealtimeConfig::krea_realtime_14b().wan;

        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_config(mirror.path(), &preset.to_json())),
            architecture_facts(&weights_free_spec()),
            "a snapshot mirroring the reference config must publish the preset's axes"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = preset.to_json();
        mutated["num_layers"] = serde_json::json!(7);
        mutated["vae_z_dim"] = serde_json::json!(32);
        let mutated_facts = architecture_facts(&spec_for_config(mutated_dir.path(), &mutated));
        assert_eq!(
            (
                mutated_facts.transformer_blocks,
                mutated_facts.latent_channels
            ),
            (Some(7), Some(32)),
            "the materialized path must publish the snapshot's geometry, not the preset's"
        );

        let bare = tempfile::tempdir().unwrap();
        let bare_spec = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            bare.path().to_path_buf(),
        ));
        assert_eq!(
            architecture_facts(&bare_spec),
            architecture_facts(&weights_free_spec()),
            "the shipped config-less snapshot must publish the preset's axes"
        );
    }

    /// SC-22667: the head width goes through the exactness-gated shared helper, not
    /// `WanModelConfig::head_dim`'s integer division. A snapshot whose `num_heads` does not divide
    /// `dim` has no single head width — publishing the truncated quotient would declare a geometry
    /// the loader cannot build — and `"num_heads": 0` would divide by zero inside `head_dim()`
    /// before `axis` could decline it, panicking a contract build rather than declining an axis.
    ///
    /// Mutation that fails this: restoring `axis(wan.head_dim())` publishes `Some(731)` for the
    /// non-divisible fixture and panics on the zero fixture.
    #[test]
    fn a_non_uniform_head_stack_declines_the_head_width_instead_of_truncating_it() {
        let preset = crate::config::KreaRealtimeConfig::krea_realtime_14b().wan;
        assert_eq!(
            preset.dim, 5120,
            "fixture premise: the reused Wan-14B width"
        );

        let ragged_dir = tempfile::tempdir().unwrap();
        let mut ragged = preset.to_json();
        ragged["num_heads"] = serde_json::json!(7);
        let ragged_facts = architecture_facts(&spec_for_config(ragged_dir.path(), &ragged));
        assert_eq!(
            (ragged_facts.attention_heads, ragged_facts.head_dim),
            (Some(7), None),
            "5120 / 7 is not a head width any load has"
        );

        let zero_dir = tempfile::tempdir().unwrap();
        let mut zero = preset.to_json();
        zero["num_heads"] = serde_json::json!(0);
        let zero_facts = architecture_facts(&spec_for_config(zero_dir.path(), &zero));
        assert_eq!(
            (zero_facts.attention_heads, zero_facts.head_dim),
            (None, None),
            "a zero head count declines both axes rather than dividing by zero"
        );

        // The shipped geometry still publishes its exact quotient.
        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_config(mirror.path(), &preset.to_json())).head_dim,
            Some(128)
        );
    }

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

    /// Byte-for-byte twin of the SceneWorks worker's sealed q8 fixture. This deliberately does not
    /// call the provider fixture helper: both sides must independently reproduce the same receipt.
    fn worker_sealed_fixture() -> (tempfile::TempDir, LoadSpec) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().to_owned();
        std::fs::write(
            root.join(CONFIG),
            br#"{"quantization":{"bits":8},"denoising_step_list":[1000,937,833,625,0]}"#,
        )
        .unwrap();
        std::fs::write(root.join(TOKENIZER), b"{}").unwrap();
        tensor(&root.join(TEXT_ENCODER), 101, 101);
        tensor(&root.join(DIT), 401, 401_u64 as u8);
        tensor(&root.join(VAE), 211, 211);
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
            // sc-22735: the anchor coordinate IS published now — the capture arm refuses a
            // contract without one — but it names the (provider, tier) cell, never the
            // per-byte structural receipt, which must stay out of the identity.
            let calibration = contract
                .calibration
                .as_ref()
                .expect("every shipped tier publishes its anchor cell");
            assert_eq!(calibration.load_shape, contract.load_shape);
            assert!(
                !calibration
                    .fingerprint
                    .contains(&canonical_artifact_identity(&q4_spec).unwrap()),
                "the structural artifact receipt is not the anchor coordinate"
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

    /// The three production tiers, resolved from a real fixture directory the way the contract
    /// builder resolves them. Keyed on the tier set, never a frozen count.
    fn production_identities() -> std::collections::BTreeMap<&'static str, String> {
        ["q4", "q8", "bf16"]
            .into_iter()
            .map(|tier| {
                let (root, spec) = fixture(tier, 400);
                let fingerprint = memory_strategy_contract(&spec)
                    .unwrap()
                    .calibration
                    .expect("a shipped tier publishes its anchor cell")
                    .fingerprint;
                drop(root);
                (tier, fingerprint)
            })
            .collect()
    }

    /// AC (sc-22735 a): each shipped tier publishes its own anchor coordinate in the documented
    /// `krea-realtime-14b-<tier>-mlx-resident-ladder-v1` format, and the three are pairwise
    /// distinct — the defect this replaces is `LoadSpec::quantize` being `None` for all three
    /// tiers, which would have collapsed them onto one string.
    #[test]
    fn every_shipped_tier_publishes_its_own_production_anchor_identity() {
        let identities = production_identities();
        for (tier, fingerprint) in &identities {
            assert_eq!(
                *fingerprint,
                format!("krea-realtime-14b-{tier}-mlx-resident-ladder-v1"),
                "{tier} anchor coordinate"
            );
            assert_eq!(
                production_calibration_fingerprint(MODEL_ID, tier).as_ref(),
                Some(fingerprint)
            );
            assert!(mlx_gen::gen_core::validate_calibration_fingerprint(fingerprint).is_ok());
        }
        let distinct: std::collections::BTreeSet<&String> = identities.values().collect();
        assert_eq!(
            distinct.len(),
            identities.len(),
            "the tiers must not collapse onto one anchor string: {identities:?}"
        );
        // A foreign provider id has no cell in this table.
        assert_eq!(
            production_calibration_fingerprint("wan_2_1_14b", "q4"),
            None
        );
    }

    /// AC (sc-22735 b): the weights-free declaration surface publishes an identity that is
    /// `Some`, carries the spec's own load shape (the SceneWorks planned-lane gate asserts that
    /// equality), and can never be confused with a production anchor string.
    #[test]
    fn the_weights_free_surface_publishes_a_distinct_static_behavior_identity() {
        let production: std::collections::BTreeSet<String> =
            production_identities().into_values().collect();
        let mut seen = 0_usize;
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            let contract = weights_free_resident_contract(&surface.spec).unwrap();
            let calibration = contract
                .calibration
                .as_ref()
                .unwrap_or_else(|| panic!("{} publishes no identity", surface.selector.id()));
            assert_eq!(
                calibration.load_shape,
                surface.spec.load_shape,
                "{} static identity must carry its own load shape",
                surface.selector.id()
            );
            assert!(
                calibration
                    .fingerprint
                    .starts_with(STATIC_BEHAVIOR_FINGERPRINT),
                "{} published {}",
                surface.selector.id(),
                calibration.fingerprint
            );
            assert!(
                !production.contains(&calibration.fingerprint),
                "{} published a production anchor string {}",
                surface.selector.id(),
                calibration.fingerprint
            );
            assert!(contract.conformance_errors().is_empty());
            seen += 1;
        }
        assert!(seen > 0, "the surface walk must not be vacuous");

        // The tier and policy tokens are the two axes this weights-free seam can actually see.
        let mut spec = weights_free_spec();
        assert_eq!(
            static_behavior_identity(&spec).fingerprint,
            format!("{STATIC_BEHAVIOR_FINGERPRINT}-bf16-resident")
        );
        spec.quantize = Some(Quant::Q8);
        spec.offload_policy = mlx_gen::OffloadPolicy::Sequential;
        assert_eq!(
            static_behavior_identity(&spec).fingerprint,
            format!("{STATIC_BEHAVIOR_FINGERPRINT}-q8-sequential")
        );
    }

    /// AC (sc-22735 c): an unprovable tier publishes no identity and never turns a contract build
    /// into an error it was not before.
    #[test]
    fn an_unprovable_tier_fails_closed_to_no_identity() {
        // No directory at all, and a file-backed source: neither can prove an artifact tier.
        assert!(production_calibration_identity(&weights_free_spec()).is_none());
        assert!(
            production_calibration_identity(&LoadSpec::new(WeightsSource::File(
                "/nonexistent-krea-realtime/dit.safetensors".into()
            )))
            .is_none()
        );

        // A real q4 snapshot loaded at an execution precision no anchor was captured at: the
        // contract still builds, with its exact physical facts, and simply declines the cell.
        let (_root, mut spec) = fixture("q4", 400);
        assert!(production_calibration_identity(&spec).is_some());
        spec.precision = mlx_gen::Precision::Fp32;
        let contract = memory_strategy_contract(&spec).expect("the contract must still build");
        assert!(contract.calibration.is_none());
        assert!(contract.asset_facts.base_bytes > 0);
    }

    /// AC (sc-22735, second half): the SceneWorks planned-lane gate
    /// (`every_planned_mlx_lane_resolves_a_weights_free_provider_contract`) synthesizes a
    /// weights-free `LoadSpec` over a directory that does not exist and, finding no memory-contract
    /// fixture for this provider, falls back to `MemoryRegistration::contract`. That fallback must
    /// build, name this provider, and carry an identity whose load shape is the spec's.
    #[test]
    fn the_registry_registration_resolves_a_weights_free_contract_for_every_planned_lane() {
        for load_shape in [
            mlx_gen::gen_core::LoadShape::EagerMaterialization,
            mlx_gen::gen_core::LoadShape::DeferredMaterialization,
        ] {
            for quantize in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let mut spec = LoadSpec::new(WeightsSource::Dir("fixture".into()));
                spec.load_shape = load_shape;
                spec.quantize = quantize;
                let contract = (REGISTRATION.contract)(&spec).unwrap_or_else(|error| {
                    panic!("planned lane {quantize:?}/{load_shape:?} cannot build: {error}")
                });
                assert_eq!(contract.provider_id, MODEL_ID);
                assert_eq!(contract.load_shape, load_shape);
                let calibration = contract
                    .calibration
                    .as_ref()
                    .expect("a planned lane must resolve a calibratable contract");
                assert_eq!(calibration.load_shape, load_shape);
            }
        }

        // A materialized snapshot still resolves the exact physical contract, not the witness.
        let (_root, spec) = fixture("q8", 800);
        let contract = (REGISTRATION.contract)(&spec).unwrap();
        assert!(contract.asset_facts.base_bytes > 0);
        assert_eq!(
            contract.calibration.unwrap().fingerprint,
            "krea-realtime-14b-q8-mlx-resident-ladder-v1"
        );
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

    fn resident_context(
        mode: &str,
        shape: &str,
        artifact: &str,
        quant: Option<Quant>,
    ) -> MemoryRunContext {
        MemoryRunContext {
            selection: mlx_gen::gen_core::MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: mlx_gen::gen_core::MemoryNumericTier {
                    precision: mlx_gen::Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                },
            },
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Resident,
            calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
            // sc-22735: the contract now publishes an anchor coordinate, so an admitting caller
            // must carry the matching one through the calibration handshake. These fixtures pack
            // the tier they request, so the requested `quant` names the artifact tier here.
            calibration_fingerprint: production_calibration_fingerprint(
                MODEL_ID,
                match quant {
                    None => "bf16",
                    Some(Quant::Q4) => "q4",
                    Some(Quant::Q8) => "q8",
                    Some(other) => panic!("fixture premise: unshipped tier {other:?}"),
                },
            )
            .expect("every shipped tier has an anchor coordinate"),
            load_shape: mlx_gen::gen_core::LoadShape::EagerMaterialization,
            mode: mlx_gen::gen_core::MemoryMode::Other(mode.to_owned()),
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: mlx_gen::gen_core::MemoryGeometry {
                width: 832,
                height: 480,
                batch: 1,
                frames: 45,
                reference_count: 1,
            },
            overlay: None,
            budget: mlx_gen::gen_core::MemoryBudget {
                total_bytes: 1 << 30,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 713,
            cache_state: mlx_gen::gen_core::MemoryCacheState::Cold,
            evidence_revision: format!(
                "provider-resident-video-request-v3:{MODEL_ID}:mode={mode}:shape={shape}:sourceFrames=45:strength=3eb33333:nativeCadence=true:artifact={artifact}:adapter=none:digest"
            ),
        }
    }

    #[test]
    fn safety_accepts_exact_i2v_and_v2v_receipts_and_rejects_crossed_mode_shape() {
        for (tier, bytes, quant) in [
            ("q4", 400, Some(Quant::Q4)),
            ("q8", 800, Some(Quant::Q8)),
            ("bf16", 1_600, None),
        ] {
            let (_root, spec) = fixture(tier, bytes);
            let contract = memory_strategy_contract(&spec).unwrap();
            let artifact = canonical_artifact_identity(&spec).unwrap();
            for (mode, shape) in [
                ("image_to_video", "image"),
                ("video_to_video", "video_clip"),
            ] {
                let context = resident_context(mode, shape, &artifact, quant);
                assert!(matches!(
                    safety_check(&spec, &contract, &context),
                    MemorySafetyDecision::Accept
                ));
                assert!(matches!(
                    loaded_safety_check(&spec, &contract, &artifact, &context),
                    MemorySafetyDecision::Accept
                ));
            }
        }
        let (_root, spec) = fixture("q4", 400);
        let contract = memory_strategy_contract(&spec).unwrap();
        let artifact = canonical_artifact_identity(&spec).unwrap();
        let crossed = resident_context("video_to_video", "image", &artifact, Some(Quant::Q4));
        assert!(matches!(
            safety_check(&spec, &contract, &crossed),
            MemorySafetyDecision::Reject { .. }
        ));
        let mut crossed =
            resident_context("video_to_video", "video_clip", &artifact, Some(Quant::Q4));
        crossed.geometry.reference_count = 2;
        assert!(matches!(
            safety_check(&spec, &contract, &crossed),
            MemorySafetyDecision::Reject { .. }
        ));

        // sc-22735: a caller admitting against another cell's anchor coordinate — or against no
        // coordinate at all, the pre-story shape — is refused by the calibration handshake.
        for fingerprint in [
            production_calibration_fingerprint(MODEL_ID, "q8").unwrap(),
            String::new(),
        ] {
            let mut crossed =
                resident_context("video_to_video", "video_clip", &artifact, Some(Quant::Q4));
            crossed.calibration_fingerprint = fingerprint;
            assert!(matches!(
                safety_check(&spec, &contract, &crossed),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }

    #[test]
    fn worker_sealed_v3_artifact_receipt_is_provider_accepted() {
        const WORKER_SEALED_ARTIFACT: &str =
            "59024c46ca617e4a45071e685bbefcea282cad6e2e41c0ab909a78e1dff77e7c";
        let (_root, spec) = worker_sealed_fixture();
        let contract = memory_strategy_contract(&spec).unwrap();
        let artifact = canonical_artifact_identity(&spec).unwrap();
        assert_eq!(artifact, WORKER_SEALED_ARTIFACT);
        let context = resident_context(
            "video_to_video",
            "video_clip",
            WORKER_SEALED_ARTIFACT,
            Some(Quant::Q8),
        );
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        ));
        assert!(matches!(
            loaded_safety_check(&spec, &contract, WORKER_SEALED_ARTIFACT, &context),
            MemorySafetyDecision::Accept
        ));
    }
}
