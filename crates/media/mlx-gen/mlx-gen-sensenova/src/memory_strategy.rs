//! SenseNova-U1 MLX shared image-memory ladder.
//!
//! The checkpoint is one flat, fused dual-path Qwen3 model. There is no separately releasable text
//! encoder or VAE: conditioning and denoise use different weights interleaved in every resident
//! layer, while the final FM head already emits RGB patches. Consequently staged component
//! residency and bounded decode are structural N/A. Bounded attention is wired only through the
//! generation path used by denoise; understanding, VQA, interleave text, and think-token forwards
//! retain their historical unbounded attention path.
//!
//! ## Declared ladder (sc-18608)
//!
//! The registry publishes 12 weights-free surfaces per provider — bf16/Q4/Q8 × resident/sequential
//! × eager/deferred — and both registered providers declare exactly:
//!
//! * [`MemoryStrategy::Resident`] — every surface;
//! * [`MemoryStrategy::BoundedAttention`] — every surface. `chunk_attention` +
//!   `attention_chunk_size` reach [`crate::t2i::T2iOptions::attention_score_budget`], which is what
//!   builds the budgeted `AttentionPlan` for both the T2I and the reference-conditioned edit
//!   denoise loops. Nothing about it depends on the artifact layout.
//! * [`MemoryStrategy::BoundedTransformerResidency`] — the **deferred** surfaces of a verified
//!   single-file snapshot only, and for `_fast` only once the distill LoRA is pre-merged. This is
//!   provider-local block windowing over the generation-path Qwen stack, reached through
//!   [`crate::t2i::T2iOptions::transformer_window_size`].
//!
//! Rung 4 keys off `LoadSpec::load_shape` and **not** `offload_policy`: SenseNova is deliberately
//! not on the shared `Residency` seam (`supports_sequential_offload: false`, F-176), so sequential
//! offload is a no-op fallback for it and would be a fabricated precondition if declared as one.
//! [`MemoryStrategy::StagedResidency`] and [`MemoryStrategy::BoundedDecode`] stay
//! [`MemoryStrategySupport::StructurallyNotApplicable`] on every surface for the structural reasons
//! above — there is no separable conditioning component to stage and no decoder phase to tile.

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
/// Source-owned weights-free behavior identity. Route/fixture semantics are versioned here so a
/// correction never restamps the measured calibration fingerprints above. v2 makes the registry
/// fixtures single-phase and fails phase-bearing contexts closed.
const STATIC_BEHAVIOR_CALIBRATION: &str = "sensenova-static-registry-behavior-v2";

/// Component key read by the fast loader. Shared with `model::load_inner` so the loader's
/// `reject_unknown_components` allow-list and this module's contract gate cannot drift apart.
pub(crate) const DISTILL_LORA_COMPONENT: &str = "distill_lora";

/// The exact production load compositions the SenseNova memory routes are wired for.
///
/// [`crate::model::load`] refuses a non-directory source, any precision override, user adapters,
/// and any unrecognized component key. Every remaining `LoadSpec` axis — control, extra controls,
/// IP adapter, PiD, identity, and an external text encoder — is silently *ignored* by that loader,
/// because NEO-Unify is one fused checkpoint with no seam for any of them. Publishing a memory
/// contract for either shape would declare a rung on a route that cannot load at all, or that
/// cannot honor the composition it was handed. Both are unreachable declarations, so admission
/// fails closed here rather than emitting a contract nothing can execute.
fn validate_load_contract(provider_id: &str, spec: &LoadSpec) -> CoreResult<()> {
    let known_components: &[&str] = match provider_id {
        crate::MODEL_ID => &[],
        crate::MODEL_ID_FAST => &[DISTILL_LORA_COMPONENT],
        _ => {
            return Err(CoreError::Unsupported(format!(
                "unknown SenseNova provider {provider_id}"
            )))
        }
    };
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: SenseNova memory routes require a snapshot directory, not a single file"
        )));
    }
    if spec.precision != mlx_gen::Precision::Bf16 {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: only the dense bf16 source is wired; drop the precision override"
        )));
    }
    // The descriptor advertises exactly Q4/Q8 over the dense bf16 source; nothing else can load.
    if !matches!(spec.quantize, None | Some(Quant::Q4 | Quant::Q8)) {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: SenseNova packs only Q4 or Q8 over the dense bf16 backbone"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: user-supplied adapters are not supported (supports_lora=false)"
        )));
    }
    mlx_gen::gen_core::reject_unknown_components(spec, known_components, provider_id)?;
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: SenseNova has no control, IP-adapter, PiD, identity, or external \
             text-encoder seam; its memory routes cannot honor that composition"
        )));
    }
    Ok(())
}

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

/// The pinned SNAPSHOT ENTRY, captured by `lstat` so a symlink is described as a symlink rather
/// than silently followed. This is the path runtime loaders open ([`PinnedArtifact::loader_path`]);
/// [`ArtifactFileIdentity`] describes what it resolves to. Mirrors `mlx-gen-flux`'s
/// `artifact_inventory::SourceEntryIdentity`, which pins the same two-level shape.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceEntryIdentity {
    absolute_path: PathBuf,
    is_symlink: bool,
    symlink_target: Option<PathBuf>,
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
    source: SourceEntryIdentity,
    identity: ArtifactFileIdentity,
    digest: String,
}

impl PinnedArtifact {
    /// Verify and pin one explicit safetensors file. Production directory loads additionally enforce
    /// the single-file inventory rule in the internal `verified_artifact` selector.
    pub fn verify_file(path: impl AsRef<Path>) -> Option<Self> {
        pinned_artifact(path.as_ref())
    }

    /// Snapshot entry consumed by format-dispatching runtime loaders.
    ///
    /// mlx-rs selects the safetensors loader from the path EXTENSION: `SafeTensors::load_device`
    /// rejects any path whose final component is not literally `*.safetensors` with
    /// `IoError::UnsupportedFormat` ("Unsupported file format"). A Hugging Face cache stores each
    /// file as an extensionless `blobs/<sha>` object and exposes it as a
    /// `snapshots/<rev>/…/model.safetensors` SYMLINK, so canonicalizing the entry — which pinning
    /// does, to resolve the identity — strips the extension. Opening that canonical blob is how
    /// every HF-cached SenseNova load died with `backend op failed: Unsupported file format`.
    ///
    /// Runtime opens therefore use this pinned entry, never the canonical blob. That does not
    /// weaken the pin: the entry, its symlink target, and the canonical file it resolves to are all
    /// re-checked by [`ensure_unchanged`](Self::ensure_unchanged) on both sides of every open, so a
    /// repointed symlink is rejected rather than followed.
    pub(crate) fn loader_path(&self) -> &Path {
        &self.source.absolute_path
    }

    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        // Canonical-target check FIRST so an in-place replacement of a regular file keeps reporting
        // "replaced or mutated" (the established message for that lane). The two entry-level checks
        // below are what the canonical stat cannot see: when the entry is an HF symlink, re-statting
        // the already-resolved blob describes the OLD target no matter where the link now points.
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
        let source = source_entry_identity(&self.source.absolute_path).map_err(|error| {
            CoreError::Msg(format!(
                "sensenova: pinned snapshot entry is no longer readable: {error}"
            ))
        })?;
        if source != self.source {
            return Err(CoreError::Msg(
                "sensenova: pinned snapshot entry or symlink target changed after verification"
                    .to_owned(),
            ));
        }
        let resolved = file_identity(&self.source.absolute_path).map_err(|error| {
            CoreError::Msg(format!(
                "sensenova: pinned snapshot entry no longer resolves: {error}"
            ))
        })?;
        if resolved != self.identity {
            return Err(CoreError::Msg(
                "sensenova: pinned snapshot entry resolves to a different canonical target"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn open_weights(&self) -> mlx_gen::Result<mlx_gen::weights::Weights> {
        self.ensure_unchanged()
            .map_err(|error| mlx_gen::Error::Msg(error.to_string()))?;
        let weights = mlx_gen::weights::Weights::from_file(self.loader_path())?;
        self.ensure_unchanged()
            .map_err(|error| mlx_gen::Error::Msg(error.to_string()))?;
        Ok(weights)
    }
}

fn source_entry_identity(path: &Path) -> std::io::Result<SourceEntryIdentity> {
    let absolute_path = std::path::absolute(path)?;
    let metadata = std::fs::symlink_metadata(&absolute_path)?;
    let is_symlink = metadata.file_type().is_symlink();
    Ok(SourceEntryIdentity {
        symlink_target: is_symlink
            .then(|| std::fs::read_link(&absolute_path))
            .transpose()?,
        absolute_path,
        is_symlink,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
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
        // Pin the ENTRY (lstat) and the object it resolves to (canonicalize) as one observation.
        // The digest cache stays keyed on the resolved identity, so two snapshot entries backed by
        // the same HF blob still coalesce onto one content hash.
        let source = source_entry_identity(path).ok()?;
        let identity = file_identity(&source.absolute_path).ok()?;
        let cache = digest_cache();
        let mut entries = cache.entries.lock().ok()?;
        match entries.get(&identity).cloned() {
            Some(DigestState::Ready(digest)) => {
                drop(entries);
                // A cache hit is useful only if the entry AND the object it resolves to are still
                // the ones just observed. This closes the replacement race between the first stat
                // and the cache lookup.
                if source_entry_identity(&source.absolute_path).ok().as_ref() == Some(&source)
                    && file_identity(&source.absolute_path).ok()? == identity
                {
                    return Some(PinnedArtifact {
                        source,
                        identity,
                        digest,
                    });
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
                let unchanged = source_entry_identity(&source.absolute_path).ok().as_ref()
                    == Some(&source)
                    && file_identity(&source.absolute_path).ok().as_ref() == Some(&identity);
                let mut entries = cache.entries.lock().ok()?;
                entries.remove(&identity);
                let result = if unchanged {
                    digest.map(|digest| {
                        entries
                            .retain(|cached, _| cached.canonical_path != identity.canonical_path);
                        entries.insert(identity.clone(), DigestState::Ready(digest.clone()));
                        PinnedArtifact {
                            source,
                            identity,
                            digest,
                        }
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

fn structurally_can_stream_gen(provider_id: &str, spec: &LoadSpec) -> bool {
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
    {
        return false;
    }
    matches!(provider_id, crate::MODEL_ID | crate::MODEL_ID_FAST)
}

pub(crate) fn can_stream_gen_with_artifact(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> bool {
    if !structurally_can_stream_gen(provider_id, spec) || artifact.is_none() {
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
    true
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let artifact = verified_artifact(spec);
    memory_strategy_contract_with_artifact(provider_id, spec, artifact.as_ref())
}

/// Production admission with the artifact the loader already pinned.
///
/// The loader resolves and verifies one artifact for the whole load; re-verifying here would hash
/// the checkpoint a second time. Keeping the composition gate in this wrapper is what stops a
/// loaded generator from publishing a contract that
/// [`memory_strategy_contract`] would have refused.
pub(crate) fn validated_memory_strategy_contract_with_artifact(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    memory_strategy_contract_with_artifact(provider_id, spec, artifact)
}

/// Weights-free contract used only by registry conformance. Production resolution never calls this
/// seam and therefore never receives its synthetic calibration identity.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let route = if provider_id == crate::MODEL_ID_FAST {
        "fast"
    } else {
        "quality"
    };
    build_memory_strategy_contract(
        provider_id,
        spec,
        mlx_gen::gen_core::PerComponentBytes::default(),
        structurally_can_stream_gen(provider_id, spec),
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_BEHAVIOR_CALIBRATION}-{route}"),
            spec.load_shape,
        )),
    )
}

pub(crate) fn memory_strategy_contract_with_artifact(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> CoreResult<MemoryProviderContract> {
    let footprint = crate::model::component_footprint(spec)?;
    let streamable = can_stream_gen_with_artifact(provider_id, spec, artifact);
    let calibration = calibration_fingerprint(provider_id, spec, artifact)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    build_memory_strategy_contract(provider_id, spec, footprint, streamable, calibration)
}

fn build_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
    footprint: mlx_gen::gen_core::PerComponentBytes,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
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
    contract.calibration = calibration;
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
    // Evidence must be captured on a composition production admission would accept; otherwise the
    // record would key a rung to a route the loader refuses or silently ignores.
    validate_load_contract(provider_id, spec)?;
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
        // `GenerationRequest::phases` is read only by the Krea MLX family; SenseNova's denoise
        // threads one AR trajectory through a per-step-mutated KV cache and ignores the field
        // entirely. A phase-bearing context would therefore be admitted against evidence for a
        // trajectory the engine never runs, so it fails closed here.
        if context.has_phases {
            return Err(CoreError::Unsupported(format!(
                "{}: SenseNova runs one single-phase trajectory and ignores request phases",
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
            has_phases: false,
            overlay: None,
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    ]
    .into_iter()
    .map(|route| {
        let mut fixture = mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, tier, route)?,
        );
        fixture.request.prompt = "weights-free SenseNova memory behavior".into();
        Ok(fixture)
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

    fn unique_root(tmp: &tempfile::TempDir, label: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        tmp.path().join(format!(
            "sensenova-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn fixture_spec(tmp: &tempfile::TempDir) -> (std::path::PathBuf, LoadSpec) {
        let root = unique_root(tmp, "memory-contract");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.safetensors"), [0_u8; 8]).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    /// The complete declared ladder, read back off the real registry rather than off one
    /// hand-picked `LoadSpec`. A caller-chosen spec cannot see a rung whose availability moves with
    /// numeric tier, residency policy, or materialization shape; the provider-owned witness set can.
    #[test]
    fn registry_publishes_the_exact_declared_ladder() {
        use mlx_gen::gen_core::{LoadShape, MemoryContractSurfaceTier};
        use std::collections::BTreeSet;

        let registry = crate::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        assert_eq!(
            surfaces.len(),
            24,
            "two providers × twelve witness surfaces"
        );

        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let provider: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider.len(), 12, "{provider_id}");
            let selectors: BTreeSet<_> = provider
                .iter()
                .map(|surface| surface.selector.id())
                .collect();
            let expected: BTreeSet<_> = [
                "bf16:resident:eager",
                "bf16:resident:deferred",
                "bf16:sequential:eager",
                "bf16:sequential:deferred",
                "q4:resident:eager",
                "q4:resident:deferred",
                "q4:sequential:eager",
                "q4:sequential:deferred",
                "q8:resident:eager",
                "q8:resident:deferred",
                "q8:sequential:eager",
                "q8:sequential:deferred",
            ]
            .into_iter()
            .collect();
            assert_eq!(selectors, expected, "{provider_id}");

            let mut bounded_transformer = 0;
            for surface in &provider {
                let selector = surface.selector;
                assert!(!surface.composed, "{provider_id}");
                assert!(matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Bf16
                        | MemoryContractSurfaceTier::Q4
                        | MemoryContractSurfaceTier::Q8
                ));
                assert!(surface
                    .contract
                    .calibration
                    .as_ref()
                    .is_some_and(|identity| identity
                        .fingerprint
                        .starts_with(STATIC_BEHAVIOR_CALIBRATION)));
                assert!(
                    surface.contract.conformance_errors().is_empty(),
                    "{provider_id} {}: {:?}",
                    selector.id(),
                    surface.contract.conformance_errors()
                );
                let support = |strategy| {
                    surface
                        .contract
                        .capability(strategy)
                        .expect("complete SenseNova ladder")
                        .support
                        .clone()
                };
                assert_eq!(
                    support(MemoryStrategy::Resident),
                    MemoryStrategySupport::Implemented,
                    "{provider_id} {}",
                    selector.id()
                );
                assert_eq!(
                    support(MemoryStrategy::BoundedAttention),
                    MemoryStrategySupport::Implemented,
                    "{provider_id} {}",
                    selector.id()
                );
                assert_eq!(
                    surface
                        .contract
                        .capability(MemoryStrategy::BoundedAttention)
                        .unwrap()
                        .parameters
                        .attention_chunk_sizes,
                    vec![ATTENTION_CHUNK_SIZE]
                );
                for structural in [
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                ] {
                    assert!(
                        matches!(
                            support(structural),
                            MemoryStrategySupport::StructurallyNotApplicable { .. }
                        ),
                        "{provider_id} {} {structural:?}",
                        selector.id()
                    );
                }
                // Rung 4 tracks the materialization shape ONLY. SenseNova is off the shared
                // residency seam, so keying it on `offload_policy` would invent a precondition.
                let deferred = selector.load_shape == LoadShape::DeferredMaterialization;
                assert_eq!(
                    support(MemoryStrategy::BoundedTransformerResidency)
                        == MemoryStrategySupport::Implemented,
                    deferred,
                    "{provider_id} {}",
                    selector.id()
                );
                if deferred {
                    bounded_transformer += 1;
                    let parameters = &surface
                        .contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .parameters;
                    assert_eq!(
                        parameters.transformer_window_sizes,
                        vec![TRANSFORMER_WINDOW_SIZE]
                    );
                    assert_eq!(
                        parameters.transformer_window_components,
                        vec![TransformerComponent::Dit]
                    );
                }
            }
            assert_eq!(bounded_transformer, 6, "{provider_id}");
        }
    }

    /// Every declared optimized rung must be *executable* from the registered behavior seam: the
    /// exact fixtures the conformance walker uses have to open a scope and configure a request the
    /// provider's own descriptor accepts.
    #[test]
    fn registered_fixtures_are_single_phase_and_executable() {
        let registry = crate::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        let mut executed = 0;
        for (provider_id, descriptor) in [
            (crate::MODEL_ID, crate::model::descriptor()),
            (crate::MODEL_ID_FAST, crate::model::descriptor_fast()),
        ] {
            let behavior = registry
                .memory_behavior_registrations()
                .find(|registration| registration.provider_id == provider_id)
                .unwrap_or_else(|| panic!("{provider_id} registers no memory behavior"));
            for surface in surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
            {
                for strategy in [
                    MemoryStrategy::BoundedAttention,
                    MemoryStrategy::BoundedTransformerResidency,
                ] {
                    if surface.contract.capability(strategy).unwrap().support
                        != MemoryStrategySupport::Implemented
                    {
                        continue;
                    }
                    let fixtures =
                        (behavior.valid_fixtures)(&surface.spec, &surface.contract, strategy)
                            .unwrap();
                    assert_eq!(fixtures.len(), 2, "T2I and single-reference edit routes");
                    for (fixture, mode) in fixtures
                        .iter()
                        .zip([MemoryMode::TextToImage, MemoryMode::Edit])
                    {
                        assert_eq!(fixture.context.mode, mode);
                        assert!(!fixture.context.has_phases, "{provider_id}");
                        assert!(fixture.request.phases.is_none(), "{provider_id}");
                        assert!(!fixture.context.use_pid && !fixture.request.use_pid);
                        assert_eq!(fixture.context.overlay, None);
                        assert_eq!(
                            fixture.request.conditioning.len(),
                            fixture.context.geometry.reference_count as usize
                        );
                        descriptor
                            .capabilities
                            .validate_request(provider_id, &fixture.request)
                            .unwrap();

                        let mut scope = (behavior.begin_request)(
                            &surface.spec,
                            &surface.contract,
                            &fixture.context,
                        )
                        .unwrap()
                        .expect("an implemented SenseNova rung must open a request scope");
                        let mut request = fixture.request.clone();
                        scope.configure_request(&mut request).unwrap();
                        let memory = request.memory.expect("configured memory knobs");
                        assert!(memory.chunk_attention);
                        assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
                        let rung4 = strategy == MemoryStrategy::BoundedTransformerResidency;
                        assert_eq!(memory.stream_transformer_blocks, rung4);
                        assert_eq!(
                            memory.transformer_window_size,
                            rung4.then_some(TRANSFORMER_WINDOW_SIZE)
                        );
                        executed += 1;
                    }
                }
            }
        }
        // Per provider: 12 bounded-attention surfaces + 6 deferred rung-4 surfaces, two routes each.
        assert_eq!(executed, 2 * (12 + 6) * 2);
    }

    /// A phase-bearing context must be refused, not silently admitted against single-phase
    /// evidence. SenseNova ignores `GenerationRequest::phases` outright.
    #[test]
    fn phase_bearing_context_is_refused_by_admission_and_by_begin_request() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        let contract = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let fixtures =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedAttention).unwrap();
        let mut phases = fixtures[0].context.clone();
        assert!(!phases.has_phases);
        phases.has_phases = true;
        assert!(matches!(
            registered_safety_check(&spec, &contract, &phases),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("single-phase trajectory")
        ));
        assert!(registered_begin_request(crate::MODEL_ID, &spec, &contract, &phases).is_err());
        std::fs::remove_dir_all(root).ok();
    }

    /// Admission must not publish a ladder for a composition `model::load` cannot load, or silently
    /// ignores. Each axis is mutated on its own so the assertion proves every gate, not the set.
    #[test]
    fn unloadable_compositions_publish_no_contract() {
        let base = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        weights_free_memory_strategy_contract(crate::MODEL_ID, &base).unwrap();
        memory_strategy_contract(crate::MODEL_ID, &base).unwrap();

        assert!(memory_strategy_contract("sensenova_u1_8b_imaginary", &base).is_err());

        let mutations: Vec<(&str, LoadSpec)> = vec![
            (
                "single-file source",
                LoadSpec::new(WeightsSource::File("/nonexistent/model.safetensors".into())),
            ),
            ("precision override", {
                let mut spec = base.clone();
                spec.precision = mlx_gen::Precision::Fp32;
                spec
            }),
            ("unsupported pack", base.clone().with_quant(Quant::Nvfp4)),
            ("user adapter", {
                let mut spec = base.clone();
                spec.adapters.push(mlx_gen::AdapterSpec::new(
                    "/nonexistent/user.safetensors".into(),
                    1.0,
                    mlx_gen::AdapterKind::Lora,
                ));
                spec
            }),
            ("unknown component", {
                let mut spec = base.clone();
                spec.components.insert(
                    "text_encoder".into(),
                    WeightsSource::Dir("/nonexistent/te".into()),
                );
                spec
            }),
            ("control", {
                let mut spec = base.clone();
                spec.control = Some(WeightsSource::Dir("/nonexistent/control".into()));
                spec
            }),
            ("identity", {
                let mut spec = base.clone();
                spec.identity = Some(mlx_gen::IdentityWeights {
                    encoder: Some(WeightsSource::File("/nonexistent/enc.safetensors".into())),
                    eva: None,
                    face_dir: None,
                });
                spec
            }),
            ("external text encoder", {
                let mut spec = base.clone();
                spec.text_encoder = Some(WeightsSource::Dir("/nonexistent/te".into()));
                spec
            }),
        ];
        for (label, spec) in mutations {
            for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
                assert!(
                    memory_strategy_contract(provider_id, &spec).is_err(),
                    "{provider_id}: {label} must publish no memory contract"
                );
                assert!(
                    weights_free_memory_strategy_contract(provider_id, &spec).is_err(),
                    "{provider_id}: {label} must publish no weights-free memory contract"
                );
            }
        }

        // The fast route's one recognized component stays admissible on the fast id only.
        let mut distill = base.clone();
        distill.components.insert(
            DISTILL_LORA_COMPONENT.into(),
            WeightsSource::File("/nonexistent/lora.safetensors".into()),
        );
        memory_strategy_contract(crate::MODEL_ID_FAST, &distill).unwrap();
        assert!(memory_strategy_contract(crate::MODEL_ID, &distill).is_err());
    }

    #[test]
    fn contract_declares_only_the_current_truthful_surface() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
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
    fn static_behavior_calibration_never_grants_unknown_runtime_artifacts() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let runtime = memory_strategy_contract(provider_id, &spec).unwrap();
            assert!(runtime.calibration.is_none());

            // This path does not exist: successful fixture construction proves the static builder
            // performs no footprint, inventory, marker, or artifact traversal.
            let fixture_contract =
                weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            assert_eq!(
                fixture_contract.asset_facts,
                mlx_gen::gen_core::MemoryAssetFacts::default()
            );
            assert!(fixture_contract
                .calibration
                .as_ref()
                .unwrap()
                .fingerprint
                .starts_with(STATIC_BEHAVIOR_CALIBRATION));
            for strategy in [
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    fixture_contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented
                );
                let fixtures =
                    registered_valid_fixture(&spec, &fixture_contract, strategy).unwrap();
                assert_eq!(fixtures.len(), 2);
                assert_eq!(
                    registered_safety_check(&spec, &fixture_contract, &fixtures[0].context),
                    MemorySafetyDecision::Accept
                );
                assert!(matches!(
                    registered_safety_check(&spec, &runtime, &fixtures[0].context),
                    MemorySafetyDecision::Reject { .. }
                ));
            }
        }
    }

    #[test]
    fn fast_runtime_lora_route_refuses_streaming_until_premerged() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
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
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
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
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
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
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "calibration-contract");
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

    /// Write a minimal but VALID safetensors file (one 2-element F32 tensor). The fixtures above
    /// write arbitrary bytes, which is enough for identity/digest assertions but cannot tell
    /// "the loader rejected the PATH" apart from "the loader rejected the CONTENTS" — and the path
    /// rejection is exactly what the two tests below pin.
    fn write_minimal_safetensors(path: &Path) {
        let mut header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
        // safetensors keeps the tensor payload 8-byte aligned; pad the header with spaces (JSON
        // insignificant whitespace) until the 8-byte length prefix plus header lands on a boundary.
        while !(8 + header.len()).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0_u8; 8]);
        std::fs::write(path, bytes).unwrap();
    }

    /// Build the Hugging Face cache shape: content in an extensionless `blobs/<sha>` object, exposed
    /// as a `model.safetensors` symlink in a snapshot directory. Returns `(snapshot_dir, entry)`.
    fn hf_cache_shape(root: &Path, blob_name: &str) -> (PathBuf, PathBuf) {
        let blobs = root.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        write_minimal_safetensors(&blobs.join(blob_name));
        let snapshot = root.join("snapshots/rev");
        std::fs::create_dir_all(&snapshot).unwrap();
        let entry = snapshot.join("model.safetensors");
        std::os::unix::fs::symlink(Path::new("../../blobs").join(blob_name), &entry).unwrap();
        (snapshot, entry)
    }

    /// An HF-cached checkpoint must LOAD, not merely verify.
    ///
    /// mlx-rs dispatches the safetensors loader on the path EXTENSION (`SafeTensors::load_device`
    /// → `IoError::UnsupportedFormat`), and pinning canonicalizes, so the pinned canonical path is
    /// the extensionless `blobs/<sha>` object. Opening THAT broke every HF-cached SenseNova load
    /// with `backend op failed: Unsupported file format`, while every identity/digest assertion in
    /// this module kept passing — raw byte hashing never reads an extension, which is precisely why
    /// the existing coverage could not see it. `open_weights` must go through `loader_path`.
    #[test]
    fn hf_cached_blob_symlink_loads_through_the_extension_bearing_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "hf-blob-symlink");
        let (snapshot, entry) = hf_cache_shape(&root, FAST_Q8_ARTIFACT);

        let spec = LoadSpec::new(WeightsSource::Dir(snapshot)).with_quant(Quant::Q8);
        let artifact =
            verified_artifact(&spec).expect("an HF blob symlink is still an exact single artifact");

        assert_eq!(artifact.loader_path(), std::path::absolute(&entry).unwrap());
        assert_eq!(
            artifact
                .loader_path()
                .extension()
                .and_then(|value| value.to_str()),
            Some("safetensors"),
            "mlx-rs dispatches the file format from this extension"
        );
        assert!(
            artifact.identity.canonical_path.extension().is_none(),
            "the canonical HF blob is extensionless — opening IT is the regression"
        );

        let weights = artifact
            .open_weights()
            .expect("an HF-cached checkpoint must load");
        assert!(weights.keys().any(|key| key == "weight"));
        artifact.ensure_unchanged().unwrap();

        std::fs::remove_dir_all(root).ok();
    }

    /// Opening the ENTRY instead of the resolved blob must not weaken the pin.
    ///
    /// Re-statting the already-resolved canonical path cannot see a repointed symlink — the old
    /// target is still there, unchanged — so pinning the entry is what keeps `loader_path` honest.
    /// Both blobs stay on disk here precisely so the canonical check passes and the entry-level
    /// check is the only thing standing between a repointed link and a different set of weights.
    #[test]
    fn repointed_snapshot_symlink_is_rejected_before_it_can_be_opened() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "hf-blob-repoint");
        let (snapshot, entry) = hf_cache_shape(&root, FAST_Q8_ARTIFACT);
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot)).with_quant(Quant::Q8);
        let artifact = verified_artifact(&spec).expect("initial pin");
        artifact
            .ensure_unchanged()
            .expect("pin is valid before the repoint");

        // A second, DIFFERENT blob; the originally-pinned one is deliberately left in place.
        write_minimal_safetensors(&root.join("blobs").join(QUALITY_Q8_ARTIFACT));
        std::fs::remove_file(&entry).unwrap();
        std::os::unix::fs::symlink(Path::new("../../blobs").join(QUALITY_Q8_ARTIFACT), &entry)
            .unwrap();

        let error = artifact
            .ensure_unchanged()
            .expect_err("a repointed snapshot entry must fail closed");
        assert!(
            error.to_string().contains("symlink target changed")
                || error.to_string().contains("different canonical target"),
            "{error}"
        );
        assert!(
            artifact.open_weights().is_err(),
            "no load may proceed through a repointed entry"
        );

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn pinned_stream_source_fails_closed_after_atomic_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "replacement-contract");
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
        let tmp = tempfile::tempdir().unwrap();
        let root = unique_root(&tmp, "coalesced-hash");
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
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
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
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
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
                    has_phases: false,
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
                has_phases: false,
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
