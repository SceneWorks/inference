//! SenseNova-U1 MLX shared image-memory ladder.
//!
//! The checkpoint is one flat, fused dual-path Qwen3 model. There is no separately releasable text
//! encoder or VAE: conditioning and denoise use different weights interleaved in every resident
//! layer, while the final FM head already emits RGB patches. Consequently staged component
//! residency and bounded decode are structural N/A. Bounded attention is wired only through the
//! generation path used by denoise; understanding, VQA, interleave text, and think-token forwards
//! retain their historical unbounded attention path.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, Quant, WeightsSource};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, OnceLock};

/// Exact production parameters exercised by the serial real-Metal runner below.
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const QUALITY_CALIBRATION_FINGERPRINT: &str =
    "sensenova-u1-quality-q8-mlx-shared-ladder-2026-08-03-v1";
pub const FAST_CALIBRATION_FINGERPRINT: &str =
    "sensenova-u1-fast-q8-mlx-shared-ladder-2026-08-03-v1";
const QUALITY_Q8_ARTIFACT: &str =
    "8da38dde4c39722259a98cfc47643c88e48cea205595625fdbd9fec097f9dc4f";
const FAST_Q8_ARTIFACT: &str = "a9f8968d44ec440bdd7bfb2937a61b847d6f80bb563ffe60ca56be0e395bcf50";

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ArtifactFileIdentity {
    canonical_path: PathBuf,
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Clone, Debug)]
pub struct PinnedArtifact {
    identity: ArtifactFileIdentity,
    digest: String,
}

impl PinnedArtifact {
    /// Verify and pin one explicit safetensors file. Production directory loads additionally enforce
    /// the single-file inventory rule in the internal `verified_artifact` selector.
    pub fn verify_file(path: impl AsRef<Path>) -> Option<Self> {
        pinned_artifact(path.as_ref())
    }

    pub(crate) fn canonical_path(&self) -> &Path {
        &self.identity.canonical_path
    }

    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        let current = file_identity(&self.identity.canonical_path).map_err(|error| {
            CoreError::Msg(format!(
                "sensenova: verified checkpoint is no longer readable: {error}"
            ))
        })?;
        if current != self.identity {
            return Err(CoreError::Msg(
                "sensenova: verified checkpoint was replaced or mutated after load".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn open_weights(&self) -> mlx_gen::Result<mlx_gen::weights::Weights> {
        self.ensure_unchanged()
            .map_err(|error| mlx_gen::Error::Msg(error.to_string()))?;
        let weights = mlx_gen::weights::Weights::from_file(self.canonical_path())?;
        self.ensure_unchanged()
            .map_err(|error| mlx_gen::Error::Msg(error.to_string()))?;
        Ok(weights)
    }
}

fn file_identity(path: &Path) -> std::io::Result<ArtifactFileIdentity> {
    let canonical_path = std::fs::canonicalize(path)?;
    let metadata = std::fs::metadata(&canonical_path)?;
    Ok(ArtifactFileIdentity {
        canonical_path,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

#[derive(Clone, Debug)]
enum DigestState {
    Hashing,
    Ready(String),
}

struct DigestCache {
    entries: Mutex<HashMap<ArtifactFileIdentity, DigestState>>,
    ready: Condvar,
}

fn digest_cache() -> &'static DigestCache {
    static CACHE: OnceLock<DigestCache> = OnceLock::new();
    CACHE.get_or_init(|| DigestCache {
        entries: Mutex::new(HashMap::new()),
        ready: Condvar::new(),
    })
}

#[cfg(test)]
fn hash_operation_counts() -> &'static Mutex<HashMap<PathBuf, u64>> {
    static COUNTS: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hash_exact_file(identity: &ArtifactFileIdentity) -> Option<String> {
    #[cfg(test)]
    {
        *hash_operation_counts()
            .lock()
            .ok()?
            .entry(identity.canonical_path.clone())
            .or_default() += 1;
    }
    let file = File::open(&identity.canonical_path).ok()?;
    let opened = file.metadata().ok()?;
    if opened.dev() != identity.device || opened.ino() != identity.inode {
        return None;
    }
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).ok()?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

fn pinned_artifact(path: &Path) -> Option<PinnedArtifact> {
    loop {
        let identity = file_identity(path).ok()?;
        let cache = digest_cache();
        let mut entries = cache.entries.lock().ok()?;
        match entries.get(&identity).cloned() {
            Some(DigestState::Ready(digest)) => {
                drop(entries);
                // A cache hit is useful only if the path still resolves to the exact same object.
                // This closes the replacement race between the first stat and cache lookup.
                if file_identity(path).ok()? == identity {
                    return Some(PinnedArtifact { identity, digest });
                }
            }
            Some(DigestState::Hashing) => {
                entries = cache.ready.wait(entries).ok()?;
                drop(entries);
            }
            None => {
                entries.insert(identity.clone(), DigestState::Hashing);
                drop(entries);
                let digest = hash_exact_file(&identity);
                let unchanged = file_identity(path).ok().as_ref() == Some(&identity);
                let mut entries = cache.entries.lock().ok()?;
                entries.remove(&identity);
                let result = if unchanged {
                    digest.map(|digest| {
                        entries
                            .retain(|cached, _| cached.canonical_path != identity.canonical_path);
                        entries.insert(identity.clone(), DigestState::Ready(digest.clone()));
                        PinnedArtifact { identity, digest }
                    })
                } else {
                    None
                };
                cache.ready.notify_all();
                return result;
            }
        }
    }
}

/// SHA-256 of the exact checkpoint bytes, cached by a mutation-sensitive filesystem identity.
///
/// The cache key includes device/inode/size plus mtime and ctime at nanosecond precision. A
/// before/after identity comparison prevents caching a digest when the file changes while it is
/// being read. This keeps repeated selector/contract calls cheap without trusting an HF blob
/// basename (which is attacker-controlled local path text).
pub(crate) fn verified_artifact(spec: &LoadSpec) -> Option<PinnedArtifact> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return None;
    };
    let mut safetensors = std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| !mlx_gen::gen_core::weightsmeta::is_hidden_file(&entry.path()))
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        });
    let only = safetensors.next()?.path();
    if safetensors.next().is_some() || only.file_name()? != "model.safetensors" {
        return None;
    }
    pinned_artifact(&only)
}

pub fn verified_artifact_identity(spec: &LoadSpec) -> Option<String> {
    verified_artifact(spec).map(|artifact| artifact.digest)
}

fn calibration_fingerprint(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> Option<&'static str> {
    if spec.precision != mlx_gen::Precision::Bf16
        || spec.quantize != Some(Quant::Q8)
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return None;
    }
    match (provider_id, artifact?.digest()) {
        (crate::MODEL_ID, QUALITY_Q8_ARTIFACT) => Some(QUALITY_CALIBRATION_FINGERPRINT),
        (crate::MODEL_ID_FAST, FAST_Q8_ARTIFACT) if matches!(&spec.weights, WeightsSource::Dir(root) if root.join(crate::DISTILL_MERGED_MARKER).is_file()) => {
            Some(FAST_CALIBRATION_FINGERPRINT)
        }
        _ => None,
    }
}

pub(crate) fn can_stream_gen_with_artifact(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> bool {
    if spec.load_shape != LoadShape::DeferredMaterialization
        || spec.precision != mlx_gen::Precision::Bf16
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !matches!(spec.weights, WeightsSource::Dir(_))
        || artifact.is_none()
    {
        return false;
    }
    if provider_id == crate::MODEL_ID_FAST {
        let WeightsSource::Dir(root) = &spec.weights else {
            return false;
        };
        // Dense-base fast loads merge a curated LoRA into every Gen block at runtime. Streaming
        // those unmerged blocks would silently change the model; only a pre-merged turnkey is exact.
        return root.join(crate::DISTILL_MERGED_MARKER).is_file();
    }
    provider_id == crate::MODEL_ID
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let artifact = verified_artifact(spec);
    memory_strategy_contract_with_artifact(provider_id, spec, artifact.as_ref())
}

pub(crate) fn memory_strategy_contract_with_artifact(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> CoreResult<MemoryProviderContract> {
    let footprint = crate::model::component_footprint(spec)?;
    let streamable = can_stream_gen_with_artifact(provider_id, spec, artifact);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.calibration = calibration_fingerprint(provider_id, spec, artifact)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![MemoryPhase::Conditioning, MemoryPhase::Denoise],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.asset_facts.base_bytes = footprint.dit;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: Vec::new(),
        synchronized_phase_release: false,
        decode_tiling: false,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };

    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "SenseNova is one flat fused dual-path checkpoint; no independently releasable conditioning component exists".to_owned(),
                }
            }
            MemoryStrategy::BoundedDecode => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "SenseNova has no VAE/decoder phase; the FM head emits RGB patches and unpatchify only reshapes them".to_owned(),
                }
            }
            MemoryStrategy::BoundedAttention => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes =
                    vec![TRANSFORMER_WINDOW_SIZE];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Dit];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }
    Ok(contract)
}

fn expected_runner_identity(provider_id: &str) -> Option<(&'static str, &'static str)> {
    match provider_id {
        crate::MODEL_ID => Some((QUALITY_Q8_ARTIFACT, QUALITY_CALIBRATION_FINGERPRINT)),
        crate::MODEL_ID_FAST => Some((FAST_Q8_ARTIFACT, FAST_CALIBRATION_FINGERPRINT)),
        _ => None,
    }
}

/// Fail-closed gate used by the ignored real-weight runner before it loads or generates anything.
pub fn validate_runner_gate(
    provider_id: &str,
    artifact_sha256: &str,
    contract: &MemoryProviderContract,
) -> CoreResult<()> {
    let (expected_sha256, expected_calibration) = expected_runner_identity(provider_id)
        .ok_or_else(|| {
            CoreError::Unsupported(format!("unknown SenseNova provider {provider_id}"))
        })?;
    if artifact_sha256 != expected_sha256 {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: runner artifact SHA-256 does not match the calibrated checkpoint"
        )));
    }
    if contract.provider_id != provider_id
        || contract.calibration.as_ref().is_none_or(|calibration| {
            calibration.fingerprint != expected_calibration
                || calibration.load_shape != contract.load_shape
        })
    {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: runner contract does not carry the expected calibration identity"
        )));
    }
    Ok(())
}

/// Verify once, construct the contract from the same pin, and enforce the runner identity gates.
pub fn verified_runner_artifact(provider_id: &str, spec: &LoadSpec) -> CoreResult<String> {
    let artifact = verified_artifact(spec).ok_or_else(|| {
        CoreError::Unsupported(format!(
            "{provider_id}: runner requires one stable model.safetensors artifact"
        ))
    })?;
    let contract = memory_strategy_contract_with_artifact(provider_id, spec, Some(&artifact))?;
    validate_runner_gate(provider_id, artifact.digest(), &contract)?;
    artifact.ensure_unchanged()?;
    Ok(artifact.digest().to_owned())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    quant: Option<Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if !matches!(
            (&context.mode, context.geometry.reference_count),
            (MemoryMode::TextToImage, 0) | (MemoryMode::Edit, 1)
        ) {
            return Err(CoreError::Unsupported(format!(
                "{}: calibrated memory routes are exactly TextToImage with zero references and Edit with one reference",
                contract.provider_id
            )));
        }
        if context.use_pid || context.overlay.is_some() {
            return Err(CoreError::Unsupported(format!(
                "{}: calibrated memory route has no PiD or overlay",
                contract.provider_id
            )));
        }
        if context.geometry.width != 1024
            || context.geometry.height != 1024
            || context.geometry.batch != 1
            || context.geometry.frames != 1
        {
            return Err(CoreError::Unsupported(format!(
                "{}: calibrated memory geometry is exactly 1024x1024, batch 1, and one frame",
                contract.provider_id
            )));
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, spec.quantize, context)
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    quant: Option<Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, quant, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        42,
        move |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded decode is structurally not applicable"
            )))
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() || contract.calibration.is_none() {
        return Ok(Vec::new());
    }
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    [
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    ]
    .into_iter()
    .map(|route| {
        Ok(mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, tier, route)?,
        ))
    })
    .collect()
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request(
        provider_id,
        contract,
        spec.quantize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{MemoryBehaviorRoute, MemoryStrategySupport};
    use mlx_gen::{LoadShape, LoadSpec, Quant, WeightsSource};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_root(label: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "sensenova-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn fixture_spec() -> (std::path::PathBuf, LoadSpec) {
        let root = unique_root("memory-contract");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.safetensors"), [0_u8; 8]).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    #[test]
    fn contract_declares_only_the_current_truthful_surface() {
        let (root, spec) = fixture_spec();
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        assert!(contract.calibration.is_none());
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert!(matches!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::StructurallyNotApplicable { .. }
            ));
        }
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn fast_runtime_lora_route_refuses_streaming_until_premerged() {
        let (root, spec) = fixture_spec();
        let contract = memory_strategy_contract(crate::MODEL_ID_FAST, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        std::fs::write(root.join(crate::DISTILL_MERGED_MARKER), b"{}\n").unwrap();
        let contract = memory_strategy_contract(crate::MODEL_ID_FAST, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn sharded_inventory_does_not_advertise_deferred_streaming() {
        let (root, spec) = fixture_spec();
        std::fs::write(root.join("model-00001-of-00002.safetensors"), [1_u8; 8]).unwrap();
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(contract.calibration.is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn hidden_safetensors_sidecars_do_not_change_the_single_file_inventory() {
        let (root, spec) = fixture_spec();
        std::fs::write(root.join("._model.safetensors"), [1_u8; 8]).unwrap();
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(verified_artifact(&spec).is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn expected_digest_basename_with_arbitrary_bytes_does_not_calibrate() {
        let root = unique_root("calibration-contract");
        std::fs::create_dir_all(&root).unwrap();
        let blob = root.join(QUALITY_Q8_ARTIFACT);
        std::fs::write(&blob, [0_u8; 8]).unwrap();
        std::os::unix::fs::symlink(&blob, root.join("model.safetensors")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q8);
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let first = verified_artifact_identity(&spec).unwrap();
        assert_ne!(
            first, QUALITY_Q8_ARTIFACT,
            "the verifier must hash content instead of trusting the target basename"
        );
        assert!(contract.calibration.is_none());

        // Same path, inode and size: changing only the bytes must invalidate the cached result via
        // ctime/mtime and remain uncalibrated.
        std::fs::write(&blob, [1_u8; 8]).unwrap();
        let second = verified_artifact_identity(&spec).unwrap();
        assert_ne!(first, second);
        assert!(memory_strategy_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .calibration
            .is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn pinned_stream_source_fails_closed_after_atomic_replacement() {
        let root = unique_root("replacement-contract");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("model.safetensors");
        std::fs::write(&path, [0_u8; 8]).unwrap();
        let artifact = PinnedArtifact::verify_file(&path).unwrap();
        let replacement = root.join("replacement.safetensors");
        std::fs::write(&replacement, [1_u8; 8]).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        let error = match artifact.open_weights() {
            Ok(_) => panic!("replacement must fail before weights are opened"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("replaced or mutated"),
            "every deferred stream open must reject a changed source: {error}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn concurrent_verification_coalesces_one_content_hash() {
        let root = unique_root("coalesced-hash");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("model.safetensors");
        std::fs::write(&path, vec![7_u8; 4 * 1024 * 1024]).unwrap();
        let canonical = std::fs::canonicalize(&path).unwrap();
        let before = hash_operation_counts()
            .lock()
            .unwrap()
            .get(&canonical)
            .copied()
            .unwrap_or(0);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    PinnedArtifact::verify_file(path)
                        .unwrap()
                        .digest()
                        .to_owned()
                })
            })
            .collect();
        let digests: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(digests.iter().all(|digest| digest == &digests[0]));
        assert_eq!(
            hash_operation_counts()
                .lock()
                .unwrap()
                .get(&canonical)
                .copied()
                .unwrap_or(0)
                - before,
            1,
            "one file identity must be read only once even under concurrent first use"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn runner_gate_requires_exact_provider_sha_and_calibration() {
        let (root, spec) = fixture_spec();
        let mut contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            QUALITY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ));
        assert!(validate_runner_gate(crate::MODEL_ID, QUALITY_Q8_ARTIFACT, &contract).is_ok());
        assert!(validate_runner_gate(crate::MODEL_ID, FAST_Q8_ARTIFACT, &contract).is_err());
        assert!(
            validate_runner_gate(crate::MODEL_ID_FAST, QUALITY_Q8_ARTIFACT, &contract).is_err()
        );
        contract.calibration = Some(MemoryCalibrationIdentity::new("stale", spec.load_shape));
        assert!(validate_runner_gate(crate::MODEL_ID, QUALITY_Q8_ARTIFACT, &contract).is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn safety_accepts_only_the_two_measured_mode_reference_pairs() {
        let (root, spec) = fixture_spec();
        let spec = spec.with_quant(Quant::Q8);
        let mut contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "sensenova-route-test",
            spec.load_shape,
        ));
        let context = |mode, reference_count| {
            mlx_gen::gen_core::standard_memory_behavior_context(
                &contract,
                MemoryStrategy::BoundedAttention,
                MemoryNumericTier {
                    precision: mlx_gen::Precision::Bf16,
                    quant: Some(Quant::Q8),
                    component_precision_floors: &[],
                },
                MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )
            .unwrap()
        };
        for accepted in [
            context(MemoryMode::TextToImage, 0),
            context(MemoryMode::Edit, 1),
        ] {
            assert_eq!(
                safety_check(&contract, Some(Quant::Q8), &accepted),
                MemorySafetyDecision::Accept
            );
        }
        for rejected in [
            context(MemoryMode::ImageToImage, 1),
            context(MemoryMode::TextToImage, 1),
            context(MemoryMode::Edit, 0),
            context(MemoryMode::Edit, 2),
        ] {
            assert!(matches!(
                safety_check(&contract, Some(Quant::Q8), &rejected),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("exactly TextToImage with zero references and Edit with one reference")
            ));
        }

        let context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedAttention,
            MemoryNumericTier {
                precision: mlx_gen::Precision::Bf16,
                quant: Some(Quant::Q8),
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        assert_eq!(
            safety_check(&contract, Some(Quant::Q8), &context),
            MemorySafetyDecision::Accept
        );
        let fixtures =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedAttention).unwrap();
        assert_eq!(fixtures.len(), 2, "T2I and single-reference edit routes");
        let mut scope =
            registered_begin_request(crate::MODEL_ID, &spec, &contract, &fixtures[0].context)
                .unwrap()
                .unwrap();
        let mut request = mlx_gen::GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.chunk_attention);
        assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));

        let deferred_spec = spec
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        let mut deferred = memory_strategy_contract(crate::MODEL_ID, &deferred_spec).unwrap();
        deferred.calibration = Some(MemoryCalibrationIdentity::new(
            "sensenova-route-test",
            deferred_spec.load_shape,
        ));
        let fixtures = registered_valid_fixture(
            &deferred_spec,
            &deferred,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap();
        let mut scope = registered_begin_request(
            crate::MODEL_ID,
            &deferred_spec,
            &deferred,
            &fixtures[0].context,
        )
        .unwrap()
        .unwrap();
        let mut request = mlx_gen::GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.chunk_attention && memory.stream_transformer_blocks);
        assert_eq!(
            memory.transformer_window_size,
            Some(TRANSFORMER_WINDOW_SIZE)
        );
        std::fs::remove_dir_all(root).ok();
    }
}
