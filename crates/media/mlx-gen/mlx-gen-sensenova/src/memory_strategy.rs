//! SenseNova-U1 MLX shared image-memory ladder.
//!
//! The checkpoint is one flat, fused dual-path Qwen3 model. There is no separately releasable text
//! encoder or VAE: conditioning and denoise use different weights interleaved in every resident
//! layer, while the final FM head already emits RGB patches. Consequently staged component
//! residency and bounded decode are structural N/A. Bounded attention is request-scoped through
//! both paths, including VQA, interleave text/source-image context, and think-token forwards.
//!
//! ## Declared ladder (sc-18608)
//!
//! The registry publishes 12 weights-free surfaces per provider — bf16/Q4/Q8 × resident/sequential
//! × eager/deferred — and both registered providers declare exactly:
//!
//! * [`MemoryStrategy::Resident`] — every surface;
//! * [`MemoryStrategy::BoundedAttention`] — every surface. `chunk_attention` +
//!   `attention_chunk_size` reach [`crate::t2i::T2iOptions::attention_score_budget`], which is what
//!   builds the budgeted `AttentionPlan` for all understanding and generation forwards. Nothing
//!   about it depends on the artifact layout.
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
//! Bounded attention is request-scoped across both the understanding and generation paths, so it
//! also covers VQA text, interleave text/source-image context, and think-token forwards. Transformer
//! block residency stays generation-only: it applies to denoise in T2I/edit/character/interleave,
//! while VQA is explicitly structural N/A for that rung.
//!
//! ## Envelope vs. structure in the request route gate (sc-20569)
//!
//! The route gate carries two kinds of clause and they must not be confused. An **envelope** clause
//! (route mode/reference pair, request geometry) states what the calibration campaign MEASURED, so
//! a request outside it degrades to the caller's legacy/estimated admission instead of refusing —
//! the same disposition `AdmissionPath::Legacy` already gives an out-of-envelope or stale-identity
//! request upstream. A **structural** clause (PiD/overlay, request phases) states what the engine
//! cannot do at all and stays fail-closed on every authority.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryOptimizationAuthority, MemoryPhase, MemoryProviderContract, MemoryRequestScope,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
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
/// The measured `2026-08-03` key of the **(`sensenova_u1_8b`, q8)** cell, captured on the
/// checkpoint whose SHA-256 is [`QUALITY_Q8_ARTIFACT`]. Retained byte-for-byte by
/// [`production_calibration_fingerprint`] at that coordinate.
pub const QUALITY_CALIBRATION_FINGERPRINT: &str =
    "sensenova-u1-quality-q8-mlx-shared-ladder-2026-08-03-v1";
/// The measured `2026-08-03` key of the **(`sensenova_u1_8b_fast`, q8)** cell, captured on the
/// pre-merged turnkey whose SHA-256 is [`FAST_Q8_ARTIFACT`]. Retained byte-for-byte by
/// [`production_calibration_fingerprint`] at that coordinate.
pub const FAST_CALIBRATION_FINGERPRINT: &str =
    "sensenova-u1-fast-q8-mlx-shared-ladder-2026-08-03-v1";
/// SHA-256 of the exact quality-route q8 checkpoint [`QUALITY_CALIBRATION_FINGERPRINT`] was
/// measured on. It is no longer a precondition of publishing that string — sc-22734 binds the
/// identity to the artifact's *tier* rather than to one recorded digest, so every shipped cell can
/// be anchored — but it remains the provenance of that measurement and the fail-closed gate the
/// real-weight runner ([`validate_runner_gate`]) still holds the campaign to.
pub const QUALITY_Q8_ARTIFACT: &str =
    "8da38dde4c39722259a98cfc47643c88e48cea205595625fdbd9fec097f9dc4f";
/// SHA-256 of the exact pre-merged `_fast` q8 turnkey [`FAST_CALIBRATION_FINGERPRINT`] was measured
/// on. See [`QUALITY_Q8_ARTIFACT`] for why it is provenance rather than a publishing precondition.
pub const FAST_Q8_ARTIFACT: &str =
    "a9f8968d44ec440bdd7bfb2937a61b847d6f80bb563ffe60ca56be0e395bcf50";
/// Source-owned weights-free behavior identity. Route/fixture semantics are versioned here so a
/// correction never restamps the measured calibration fingerprints above. v2 makes the registry
/// fixtures single-phase and fails phase-bearing contexts closed.
const STATIC_BEHAVIOR_CALIBRATION: &str = "sensenova-static-registry-behavior-v2";

/// Component key read by the fast loader. Shared with `model::load_inner` so the loader's
/// `reject_unknown_components` allow-list and this module's contract gate cannot drift apart.
pub(crate) const DISTILL_LORA_COMPONENT: &str = "distill_lora";

pub const QUALITY_PUBLIC_ROUTES: &[&str] = &[
    "sensenova_u1_8b",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v3",
];
pub const FAST_PUBLIC_ROUTES: &[&str] = &[
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3_fast",
];

pub fn public_routes(provider_id: &str) -> CoreResult<&'static [&'static str]> {
    match provider_id {
        crate::MODEL_ID => Ok(QUALITY_PUBLIC_ROUTES),
        crate::MODEL_ID_FAST => Ok(FAST_PUBLIC_ROUTES),
        _ => Err(CoreError::Unsupported(format!(
            "unknown SenseNova provider {provider_id}"
        ))),
    }
}

fn expected_repository(route: &str) -> Option<String> {
    QUALITY_PUBLIC_ROUTES
        .iter()
        .chain(FAST_PUBLIC_ROUTES)
        .find(|candidate| **candidate == route)
        .map(|route| format!("{}-mlx", route.replace('_', "-")))
}

/// Bind the public route to the repository-bearing resolved path. [`PinnedArtifact`] then freezes
/// the exact snapshot entry and canonical target for every operation.
pub(crate) fn validate_resolved_artifact_binding(spec: &LoadSpec) -> CoreResult<()> {
    let (Some(route), WeightsSource::Dir(root)) = (spec.resolved_route.as_deref(), &spec.weights)
    else {
        return Ok(());
    };
    let expected = expected_repository(route).ok_or_else(|| {
        CoreError::Unsupported(format!("unknown SenseNova resolved route {route}"))
    })?;
    let expected_hf = format!("models--SceneWorks--{expected}");
    let expected_app = format!("SceneWorks__{expected}");
    if root.components().any(|component| {
        let component = component.as_os_str().to_string_lossy();
        component == expected || component == expected_hf || component == expected_app
    }) {
        return Ok(());
    }
    Err(CoreError::Unsupported(format!(
        "sensenova: resolved route {route} requires repository identity SceneWorks/{expected}, but weights path {} carries no matching repository component",
        root.display()
    )))
}

/// The exact production load compositions the SenseNova memory routes are wired for.
///
/// [`crate::model::load`] refuses a non-directory source, any precision override, user adapters,
/// and any unrecognized component key. Every remaining `LoadSpec` axis — control, extra controls,
/// IP adapter, PiD, identity, and an external text encoder — is silently *ignored* by that loader,
/// because NEO-Unify is one fused checkpoint with no seam for any of them. Publishing a memory
/// contract for either shape would declare a rung on a route that cannot load at all, or that
/// cannot honor the composition it was handed. Both are unreachable declarations, so admission
/// fails closed here rather than emitting a contract nothing can execute.
pub(crate) fn validate_load_contract(provider_id: &str, spec: &LoadSpec) -> CoreResult<()> {
    let routes = public_routes(provider_id)?;
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
    if let Some(route) = spec.resolved_route.as_deref() {
        if !routes.contains(&route) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: resolved route {route:?} does not belong to this SenseNova provider; expected one of {}",
                routes.join(", ")
            )));
        }
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

/// The route slug the calibration identity strings carry, for each of the **six** public catalog
/// routes the two SenseNova providers serve (sc-22734, epic sc-22723 E1/E4).
///
/// Before sc-22734 the identity was published only when `spec.resolved_route == provider_id`, so
/// the four infographic aliases published nothing at all and could never be anchored. They are
/// independently resolved checkpoints with their own repositories
/// (`validate_resolved_artifact_binding`), so each gets its own slug rather than borrowing a
/// sibling's evidence — which is exactly what the old veto was protecting against, now expressed as
/// a distinct key instead of an absent one.
pub fn route_label(route: &str) -> Option<&'static str> {
    match route {
        "sensenova_u1_8b" => Some("quality"),
        "sensenova_u1_8b_fast" => Some("fast"),
        "sensenova_u1_8b_infographic_v2" => Some("infographic-v2"),
        "sensenova_u1_8b_infographic_v2_fast" => Some("infographic-v2-fast"),
        "sensenova_u1_8b_infographic_v3" => Some("infographic-v3"),
        "sensenova_u1_8b_infographic_v3_fast" => Some("infographic-v3-fast"),
        _ => None,
    }
}

/// The catalog route a spec loads: its explicit `resolved_route` when the worker set one, else the
/// provider's own base route id (which is itself one of the six).
fn spec_route<'a>(provider_id: &'a str, spec: &'a LoadSpec) -> &'a str {
    spec.resolved_route.as_deref().unwrap_or(provider_id)
}

/// Tier label of a SenseNova load: `bf16` for the dense source, `q4`/`q8` for the two shipped
/// packed tiers (`validate_load_contract` refuses anything else). `None` for a tier this family
/// does not ship.
pub fn calibration_tier_label(quant: Option<Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(Quant::Q4) => Some("q4"),
        Some(Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// The tier of the artifact `spec` points at, read from the checkpoint's own tensor headers.
///
/// This deliberately does **not** consult `config.json`: `validate_artifact_tier`'s own doc says
/// an absent `quantization` marker is compatible with any declared tier, so a config probe would
/// accept a dense snapshot as evidence for a packed anchor — precisely the failure this binding
/// exists to close. The tier comes off the same seam `crate::quant::lin` packed-detects on: a
/// `{base}.scales` companion of a backbone decoder Linear, with the width inferred from the u32
/// codes / scales shape ratio at `crate::quant::GROUP_SIZE` (64).
///
/// "Backbone" is `crate::convert::is_backbone_linear`, the exact predicate the converter packs
/// by and the same key set the Candle sibling's `is_backbone_linear` scans, so the two lanes can
/// never disagree about which Linears carry the tier.
///
/// `Ok(None)` is a dense (bf16) checkpoint. `Err` fails closed on anything that is not a readable
/// shipped tier: a source that is not a snapshot directory, a root whose `model.safetensors`
/// cannot be read, a checkpoint with no backbone decoder Linears at all, a `.scales` with no codes,
/// a codes/scales ratio that is not an exact 4- or 8-bit pack, or two bases packed at different
/// widths.
pub fn resolved_artifact_tier(spec: &LoadSpec) -> CoreResult<Option<Quant>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Unsupported(
            "sensenova artifact tier: the load is not a snapshot directory".to_owned(),
        ));
    };
    // `verified_artifact` already establishes that a SenseNova snapshot is exactly one
    // `model.safetensors`; reading that file directly keeps this seam on the same shape.
    let path = root.join("model.safetensors");
    let fail = |detail: String| {
        CoreError::Unsupported(format!(
            "sensenova artifact tier: {} — {detail}",
            path.display()
        ))
    };
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path)
        .map_err(|error| fail(format!("no readable checkpoint weights ({error})")))?;
    let shapes: HashMap<&str, &[usize]> = headers
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect();
    let mut width: Option<Quant> = None;
    let mut backbone_linears = 0_usize;
    for tensor in &headers {
        if let Some(base) = tensor.name.strip_suffix(".weight") {
            if crate::convert::is_backbone_linear(base) {
                backbone_linears += 1;
            }
            continue;
        }
        let Some(base) = tensor.name.strip_suffix(".scales") else {
            continue;
        };
        if !crate::convert::is_backbone_linear(base) {
            continue;
        }
        let codes = shapes
            .get(format!("{base}.weight").as_str())
            .ok_or_else(|| fail(format!("`{base}.scales` has no `{base}.weight` codes")))?;
        let bits = packed_width_from_shapes(codes, &tensor.shape)
            .ok_or_else(|| fail(format!("`{base}` is not an exact 4- or 8-bit pack")))?;
        match width {
            None => width = Some(bits),
            Some(seen) if seen == bits => {}
            Some(seen) => {
                return Err(fail(format!(
                    "packed bases disagree on their width: `{base}` is {bits:?}, an earlier base \
                     is {seen:?}"
                )))
            }
        }
    }
    if backbone_linears == 0 {
        return Err(fail(
            "no backbone decoder Linears to read a tier from".to_owned(),
        ));
    }
    Ok(width)
}

/// Packed width from header shapes alone: `scales` is `[out, in / GROUP_SIZE]` and the u32 codes are
/// `[out, in · bits / 32]`, so `bits = codes.cols · 32 / (scales.cols · GROUP_SIZE)` when that
/// division is exact and lands on a shipped width. This is the same arithmetic the Candle sibling's
/// `detect_checkpoint_quantization` performs against `crate::quant::GROUP_SIZE`.
fn packed_width_from_shapes(codes: &[usize], scales: &[usize]) -> Option<Quant> {
    let (&[out, code_cols], &[scale_rows, scale_cols]) = (codes, scales) else {
        return None;
    };
    if out != scale_rows {
        return None;
    }
    let group = usize::try_from(crate::quant::GROUP_SIZE).ok()?;
    let in_dim = scale_cols.checked_mul(group)?;
    let packed_width = code_cols.checked_mul(32)?;
    if in_dim == 0 || packed_width % in_dim != 0 {
        return None;
    }
    match packed_width / in_dim {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// The tier a load ASKS for — `LoadSpec::quantize`, with no artifact fallback.
///
/// Unlike packed-detect-only families, SenseNova's MLX loader genuinely quantizes at load time
/// (`crate::model::load`: `if let Some(q) = spec.quantize { model.quantize(q.bits())? }`), so the
/// request knob is a real, independent fact about the load. Falling back to the artifact's own
/// width would let a `quantize = None` request over a packed root claim that tier's anchor while
/// the loader was asked for something else; `production_calibration_identity` instead requires
/// the two to AGREE before publishing anything.
pub fn requested_tier(spec: &LoadSpec) -> Option<Quant> {
    spec.quantize
}

/// Production calibration identity table of the MLX SenseNova cells, keyed on **(route, tier)** —
/// sc-22734, epic sc-22723 E1/E4. Six public catalog routes x three shipped tiers = 18 cells.
///
/// Before sc-22734 this published a string only for `(provider base route, q8, exact recorded
/// artifact digest)`: 16 of the 18 cells published nothing at all and could never be anchored.
///
/// **`offload_policy` is deliberately NOT in the key**, unlike the SANA table (sc-22731). SANA's
/// rung 4 is declared per offload policy, so its two policies are genuinely different ladders.
/// SenseNova's rung 4 keys off `LoadSpec::load_shape` and explicitly not `offload_policy` (this
/// module's header, and `supports_sequential_offload: false`, F-176), so sequential offload is a
/// no-op fallback here and a policy axis would split one measurement into two coordinates that
/// describe the same load. This follows the FLUX.1 precedent (sc-22726), whose table is likewise
/// policy-free. The materialization axis is not lost: `MemoryCalibrationIdentity::load_shape`
/// carries it alongside the fingerprint.
///
/// The two `2026-08-03` strings are the measured literals, retained byte-for-byte at the exact
/// coordinates they were captured on — `(quality, q8)` on [`QUALITY_Q8_ARTIFACT`] and `(fast, q8)`
/// on [`FAST_Q8_ARTIFACT`], the two checkpoint digests those campaigns ran against.
///
/// This is the TABLE, not the binding: the tier here is [`requested_tier`], the caller's knob. Only
/// `production_calibration_identity` — which proves that tier against the artifact on disk and
/// re-checks the load composition — may turn one of these strings into a published contract
/// identity.
pub fn production_calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<String> {
    let route = route_label(spec_route(provider_id, spec))?;
    let tier = calibration_tier_label(requested_tier(spec))?;
    Some(match (route, tier) {
        ("quality", "q8") => QUALITY_CALIBRATION_FINGERPRINT.to_owned(),
        ("fast", "q8") => FAST_CALIBRATION_FINGERPRINT.to_owned(),
        _ => format!("sensenova-u1-{route}-{tier}-mlx-shared-ladder-v1"),
    })
}

/// The identity a PRODUCTION load publishes: the (route, tier) string from
/// [`production_calibration_fingerprint`], but only once the load is one an anchor could describe.
///
/// This NEVER fails the load — every refusal is `None`, not `Err`.
///
/// * **Composition.** Every guard the pre-sc-22734 gate carried is retained: an overridden
///   precision, user adapters, component overlays, control/extra controls, IP adapter, PiD,
///   identity, or an external text encoder all zero the identity. The loader ignores most of those
///   rather than refusing, so a contract published under one would key a rung to a composition
///   nothing measured.
/// * **Pinned artifact.** A snapshot that is not the single stable `model.safetensors`
///   [`verified_artifact`] pins is not a shape any campaign ran on.
/// * **Pre-merged `_fast` turnkey.** A `_fast` root without [`crate::DISTILL_MERGED_MARKER`] loads
///   an unmerged base and merges a curated LoRA at runtime — a different resident shape from the
///   measured one, on every tier, matching [`can_stream_gen_with_artifact`]'s own requirement.
/// * **Artifact tier.** The requested tier must be the tier of the weights on disk. A dense
///   snapshot asked for at q4 is a load-time requantization no anchor measured, a packed snapshot
///   asked for the other packed tier is not a shipped load, and a `quantize = None` request over a
///   packed root is not the packed load either. An artifact whose width cannot be read publishes
///   `None` — fail closed.
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
    artifact: Option<&PinnedArtifact>,
) -> Option<MemoryCalibrationIdentity> {
    if spec.precision != mlx_gen::Precision::Bf16
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
    artifact?;
    if provider_id == crate::MODEL_ID_FAST {
        let WeightsSource::Dir(root) = &spec.weights else {
            return None;
        };
        if !root.join(crate::DISTILL_MERGED_MARKER).is_file() {
            return None;
        }
    }
    if resolved_artifact_tier(spec).ok()? != requested_tier(spec) {
        return None;
    }
    production_calibration_fingerprint(provider_id, spec)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

/// Validate converter-written packed-tier provenance. MLX may still quantize a dense source at load
/// time, so an absent marker is compatible with any declared tier; once a packed marker is present,
/// however, its bit-width and group size must match the declaration exactly.
pub(crate) fn validate_artifact_tier(spec: &LoadSpec) -> CoreResult<()> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(());
    };
    // Preserve the loader's existing actionable missing-source/component errors. Once a resolved
    // snapshot directory exists, its config becomes mandatory tier provenance.
    if !root.is_dir() {
        return Ok(());
    }
    let path = root.join("config.json");
    let body = std::fs::read_to_string(&path).map_err(|error| {
        CoreError::Unsupported(format!(
            "sensenova: cannot bind numeric tier without {}: {error}",
            path.display()
        ))
    })?;
    let config: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
        CoreError::Unsupported(format!(
            "sensenova: malformed tier provenance {}: {error}",
            path.display()
        ))
    })?;
    let Some(quantization) = config
        .get("quantization")
        .and_then(|value| value.as_object())
    else {
        return Ok(());
    };
    let recorded_bits = quantization.get("bits").and_then(|value| value.as_i64());
    let recorded_group = quantization
        .get("group_size")
        .and_then(|value| value.as_i64());
    let declared = spec.quantize.map(Quant::bits);
    if matches!(
        (declared, recorded_bits, recorded_group),
        (Some(declared), Some(recorded), Some(64)) if i64::from(declared) == recorded
    ) {
        return Ok(());
    }
    Err(CoreError::Unsupported(format!(
        "sensenova: declared numeric tier {:?} does not match config quantization provenance bits={recorded_bits:?} group_size={recorded_group:?}",
        declared
    )))
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
    validate_resolved_artifact_binding(spec)?;
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
    validate_resolved_artifact_binding(spec)?;
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
    let calibration = production_calibration_identity(provider_id, spec, artifact);
    build_memory_strategy_contract(provider_id, spec, footprint, streamable, calibration)
}

/// Architecture axes shared by both registered SenseNova-U1 routes (epic SC-22657, E2).
///
/// SenseNova has no separate DiT: generation runs through the dense Qwen3 MoT backbone plus a
/// shallow flow-matching head, so the attention axes are the backbone's.
/// [`NeoLlmConfig::default`](crate::config::NeoLlmConfig) is the shipped 8B-MoT `config.json`
/// resolved through the very parser `NeoChatConfig::from_dir` uses at load, so the two cannot drift.
/// The quality and fast routes share that backbone and differ only in the sampling profile.
///
/// `patch_size` is the flow-matching head's pixel patch — the top-level `patch_size` key of the
/// same `config.json`, which `NeoChatConfig` parses and `unpatchify` reshapes by. It is a real
/// config axis on both backends (the Candle sibling publishes the same key), so it is published
/// here rather than declared absent; on the weights-free surface it is the parser's own default,
/// [`DEFAULT_PATCH_SIZE`](crate::config::DEFAULT_PATCH_SIZE).
///
/// Three axes are declared structurally absent, all for one reason: **this provider has no VAE and
/// no latent at all.** The FM head emits RGB patches which `unpatchify` only reshapes, so there is
/// no latent channel count and no autoencoder scale — spatial or temporal — to declare. A
/// structurally absent axis is `None`, never zero.
///
/// `activation_dtype_width` is the store width the load will actually use, derived from the spec
/// the way `model.rs` admits it: only dense bf16 is wired (`spec.precision == Bf16`), so the width
/// is the half-precision one for an admitted spec and `None` for a precision the loader refuses —
/// a literal `2` would describe a load that never happens.
///
/// When `spec` names a materialized snapshot directory this re-runs `NeoChatConfig::from_dir` —
/// the loader's own parse — so the published backbone axes are the snapshot's rather than the
/// preset's. On the weights-free surface there is no `config.json` to read and the preset, which
/// resolves through that same parser, is the honest answer.
fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let chat = mlx_gen::architecture_facts::materialized_root(spec)
        .and_then(|root| crate::config::NeoChatConfig::from_dir(root).ok());
    let llm = chat
        .as_ref()
        .map(|chat| chat.llm.clone())
        .unwrap_or_default();
    let patch_size = chat
        .as_ref()
        .map_or(crate::config::DEFAULT_PATCH_SIZE, |chat| chat.patch_size);
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(llm.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(llm.head_dim()),
        transformer_blocks: mlx_gen::architecture_facts::axis(llm.num_hidden_layers),
        patch_size: mlx_gen::architecture_facts::axis(patch_size),
        latent_channels: None,
        vae_spatial_scale: None,
        vae_temporal_scale: None,
        activation_dtype_width: store_dtype_width(spec),
    }
}

/// Bytes per element of the store dtype `model.rs` loads `spec` at: bf16 for the only precision it
/// admits, and nothing for one it refuses.
fn store_dtype_width(spec: &LoadSpec) -> Option<u32> {
    match spec.precision {
        mlx_gen::Precision::Bf16 => Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
        mlx_gen::Precision::Fp32 => None,
    }
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
    contract.architecture_facts = architecture_facts(spec);
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
        // sc-20569: this gate describes what the 2026-08-03 calibration campaign MEASURED — clean
        // 1024x1024 text-to-image (and the single-reference edit) at batch 1, one frame. That is a
        // statement about which evidence exists, not about what the engine can render. The shipped
        // manifest advertises seven geometries from 1152 to 2048 per side, counts 1/2/4, and a
        // `character_image` (image-to-image) route; the measured cell is not among them, so an
        // unconditional refusal here rejected EVERY product-legal SenseNova request in production.
        //
        // Envelope clauses therefore DEGRADE instead of refusing. A context whose authority is
        // `Estimated` or `Resident` has already been told by the caller that it is not riding the
        // measured ladder — that is exactly `AdmissionPath::Legacy` in the SceneWorks fit gate,
        // which routes an out-of-envelope or stale-identity request to the legacy estimator rather
        // than killing it. Only a `Calibrated` claim ("grade me against the measured ladder") is
        // held to the measured cell, because admitting THAT off-cell would grade a request against
        // evidence captured at a different geometry — the false green this gate exists to stop.
        //
        // The structural clauses below are NOT envelope statements and stay unconditional: no
        // amount of estimate authority makes SenseNova grow a PiD seam or a multi-phase trajectory.
        let claims_measured_evidence =
            context.optimization_authority == MemoryOptimizationAuthority::Calibrated;
        let structurally_supported = match (&context.mode, context.geometry.reference_count) {
            (MemoryMode::TextToImage, 0) => true,
            (MemoryMode::Edit | MemoryMode::ImageToImage, 1..=5) => true,
            (MemoryMode::Other(mode), 1..=5)
                if mode == "edit_image" || mode == "character_image" =>
            {
                true
            }
            (MemoryMode::Other(mode), 1)
                if contract.provider_id == crate::MODEL_ID && mode == "vqa" =>
            {
                true
            }
            (MemoryMode::Other(mode), 0..=10)
                if contract.provider_id == crate::MODEL_ID && mode == "interleave" =>
            {
                true
            }
            _ => false,
        };
        if !structurally_supported {
            return Err(CoreError::Msg(format!(
                "{}: unsupported SenseNova route {}/{} references",
                contract.provider_id,
                context.mode.as_key(),
                context.geometry.reference_count
            )));
        }
        if matches!(&context.mode, MemoryMode::Other(mode) if mode == "interleave")
            && !(1..=10).contains(&context.geometry.batch)
        {
            return Err(CoreError::Msg(format!(
                "{}: interleave generated-image count must be 1..=10, got {}",
                contract.provider_id, context.geometry.batch
            )));
        }
        if matches!(&context.mode, MemoryMode::Other(mode) if mode == "vqa")
            && context.selection.strategy == MemoryStrategy::BoundedTransformerResidency
        {
            return Err(CoreError::Msg(format!(
                "{}: bounded transformer residency is structurally not applicable to VQA understanding",
                contract.provider_id
            )));
        }
        // Every refusal below is built with `CoreError::Msg`, not `CoreError::Unsupported`. The
        // shared pipeline stringifies a route-gate error into `MemorySafetyDecision::Reject`
        // immediately (`error.to_string()`), so nothing downstream can read the type, while
        // `begin_request` types the surviving string as `Unsupported` once at the request boundary.
        // Building `Unsupported` here too is what printed `unsupported: unsupported: …` in the
        // production log (sc-20569).
        if claims_measured_evidence
            && !matches!(
                (&context.mode, context.geometry.reference_count),
                (MemoryMode::TextToImage, 0) | (MemoryMode::Edit, 1)
            )
        {
            return Err(CoreError::Msg(format!(
                "{}: calibrated memory routes are exactly TextToImage with zero references and Edit with one reference",
                contract.provider_id
            )));
        }
        if context.use_pid || context.overlay.is_some() {
            return Err(CoreError::Msg(format!(
                "{}: SenseNova has no PiD or overlay seam",
                contract.provider_id
            )));
        }
        // `GenerationRequest::phases` is read only by the Krea MLX family; SenseNova's denoise
        // threads one AR trajectory through a per-step-mutated KV cache and ignores the field
        // entirely. A phase-bearing context would therefore be admitted against evidence for a
        // trajectory the engine never runs, so it fails closed here.
        if context.has_phases {
            return Err(CoreError::Msg(format!(
                "{}: SenseNova runs one single-phase trajectory and ignores request phases",
                contract.provider_id
            )));
        }
        if claims_measured_evidence
            && (context.geometry.width != 1024
                || context.geometry.height != 1024
                || context.geometry.batch != 1
                || context.geometry.frames != 1)
        {
            return Err(CoreError::Msg(format!(
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

pub(crate) fn validate_direct_operation_identity(
    provider_id: &str,
    context: &MemoryRunContext,
    actual_mode: &MemoryMode,
    actual_geometry: mlx_gen::gen_core::MemoryGeometry,
) -> CoreResult<()> {
    if &context.mode == actual_mode && context.geometry == actual_geometry {
        return Ok(());
    }
    Err(CoreError::Unsupported(format!(
        "{provider_id}: direct operation {}/{} references at {}x{} does not match admitted {}/{} references at {}x{}",
        actual_mode.as_key(),
        actual_geometry.reference_count,
        actual_geometry.width,
        actual_geometry.height,
        context.mode.as_key(),
        context.geometry.reference_count,
        context.geometry.width,
        context.geometry.height
    )))
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    quant: Option<Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    // The one place a refused route becomes a TYPED error. `MemorySafetyDecision::Reject` carries an
    // already-rendered string, so the route gate deliberately hands back plain reasons (see
    // `safety_check`) and the `unsupported: ` prefix is applied exactly once, here.
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

    /// The exact geometries `config/manifests/builtin.models.jsonc` advertises for all six shipped
    /// SenseNova ids (`sensenova_u1_8b`, `_fast`, `_infographic_v2`, `_v2_fast`, `_v3`, `_v3_fast` —
    /// six product ids over these two engine ids). None of them is the measured 1024x1024 cell; the
    /// smallest side offered anywhere is 1152.
    const MANIFEST_RESOLUTIONS: [(u32, u32); 7] = [
        (2048, 2048),
        (2048, 1152),
        (1152, 2048),
        (1888, 1248),
        (1248, 1888),
        (1760, 1312),
        (1312, 1760),
    ];

    /// The manifest's `limits.count`. SceneWorks pins the provider-facing `batch` to one forward
    /// pass, but the gate is a provider-owned seam and any caller may set it, so the degrade is
    /// proven across the full advertised count axis rather than at the worker's current pin.
    const MANIFEST_COUNTS: [u32; 3] = [1, 2, 4];

    fn q8_tier() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        }
    }

    /// A runtime contract stamped with a calibration identity, so the shared handshake passes and
    /// the provider's own route gate is the only thing left that can refuse.
    fn calibrated_contract(provider_id: &str, spec: &LoadSpec) -> MemoryProviderContract {
        let mut contract = memory_strategy_contract(provider_id, spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "sensenova-route-test",
            spec.load_shape,
        ));
        contract
    }

    fn route_context(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
        route: MemoryBehaviorRoute,
    ) -> MemoryRunContext {
        mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, q8_tier(), route)
            .unwrap()
    }

    fn t2i_route() -> MemoryBehaviorRoute {
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    }

    // ------------------------------------------------------------------------------------------
    // sc-22734 (epic sc-22723 E1/E4): every shipped (route, tier) cell publishes its own
    // production calibration identity, bound to the tier of the artifact on disk.
    // ------------------------------------------------------------------------------------------

    /// Write one safetensors file by hand: `(name, dtype, shape)` entries over a zero-filled data
    /// region. Only the HEADER matters to the tier seam, exactly as it does to the loader's
    /// packed-detect.
    fn write_safetensors(path: &std::path::Path, entries: &[(String, &str, Vec<usize>)]) {
        let width = |dtype: &str| match dtype {
            "BF16" => 2,
            "U32" | "F32" => 4,
            other => panic!("fixture dtype {other}"),
        };
        let mut offset = 0_usize;
        let mut fields = Vec::new();
        for (name, dtype, shape) in entries {
            let bytes = shape.iter().product::<usize>() * width(dtype);
            let shape = shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let end = offset + bytes;
            fields.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}"
            ));
            offset = end;
        }
        let mut json = format!("{{{}}}", fields.join(",")).into_bytes();
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(path, bytes).unwrap();
    }

    /// The packed triple `crate::convert` writes for one backbone Linear `{base}` of `in = 64` at
    /// `crate::quant::GROUP_SIZE` 64 — u32 codes `[out, 64·bits/32]`, bf16 scales/biases `[out, 1]`
    /// — or the dense bf16 `[out, 64]` weight when `bits` is `None`.
    fn linear_entries(
        base: &str,
        out: usize,
        bits: Option<usize>,
    ) -> Vec<(String, &'static str, Vec<usize>)> {
        match bits {
            Some(bits) => vec![
                (format!("{base}.weight"), "U32", vec![out, 64 * bits / 32]),
                (format!("{base}.scales"), "BF16", vec![out, 1]),
                (format!("{base}.biases"), "BF16", vec![out, 1]),
            ],
            None => vec![(format!("{base}.weight"), "BF16", vec![out, 64])],
        }
    }

    /// Two backbone decoder Linears — one attention projection, one MLP projection, on both the
    /// understanding and generation paths — packed at `bits` (`None` = dense bf16), plus a
    /// non-backbone tensor the tier scan must ignore.
    fn backbone_entries(bits: Option<usize>) -> Vec<(String, &'static str, Vec<usize>)> {
        let mut entries = linear_entries("language_model.model.layers.0.self_attn.q_proj", 8, bits);
        entries.extend(linear_entries(
            "language_model.model.layers.0.mlp_mot_gen.gate_proj",
            8,
            bits,
        ));
        // Not a backbone Linear: never carries the tier, and a `.scales`-less dense key here must
        // not make a packed checkpoint read as dense.
        entries.push((
            "language_model.model.embed_tokens.weight".to_owned(),
            "BF16",
            vec![16, 64],
        ));
        entries
    }

    /// A snapshot root shaped the way the SenseNova turnkey ships: exactly one `model.safetensors`
    /// (the shape [`verified_artifact`] pins), under a path component carrying the route's own
    /// repository identity so `validate_resolved_artifact_binding` admits it.
    fn tier_root(tmp: &tempfile::TempDir, route: &str, bits: Option<usize>) -> std::path::PathBuf {
        let repository = format!("SceneWorks__{}-mlx", route.replace('_', "-"));
        let root = unique_root(tmp, "tier").join(repository);
        std::fs::create_dir_all(&root).unwrap();
        write_safetensors(&root.join("model.safetensors"), &backbone_entries(bits));
        if route.ends_with("_fast") {
            std::fs::write(root.join(crate::DISTILL_MERGED_MARKER), b"{}\n").unwrap();
        }
        root
    }

    fn tier_spec(root: &std::path::Path, route: &str, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_resolved_route(route)
            .with_load_shape(LoadShape::EagerMaterialization);
        spec.quantize = quant;
        spec
    }

    /// The six public catalog routes, paired with the provider that serves each.
    fn every_route() -> Vec<(&'static str, &'static str)> {
        QUALITY_PUBLIC_ROUTES
            .iter()
            .map(|route| (crate::MODEL_ID, *route))
            .chain(
                FAST_PUBLIC_ROUTES
                    .iter()
                    .map(|route| (crate::MODEL_ID_FAST, *route)),
            )
            .collect()
    }

    /// The three shipped tiers, as `(fixture bits, LoadSpec::quantize)`.
    const SHIPPED_TIERS: [(Option<usize>, Option<Quant>); 3] = [
        (None, None),
        (Some(4), Some(Quant::Q4)),
        (Some(8), Some(Quant::Q8)),
    ];

    /// **All eighteen shipped MLX cells publish a distinct production identity, and the set is
    /// exactly the eighteen the SceneWorks anchor plan binds** (sc-22734). Six public catalog
    /// routes x three tiers; the `(quality, q8)` and `(fast, q8)` coordinates keep their measured
    /// `2026-08-03` literals byte-for-byte.
    ///
    /// Mutation that fails this: restoring the `resolved_route != provider_id` veto (the four
    /// infographic routes publish nothing) or the `quantize != Some(Q8)` veto (q4 and bf16 publish
    /// nothing) — sixteen of the eighteen cells go unanchorable, which is the sc-22734 defect.
    #[test]
    fn every_shipped_mlx_cell_publishes_its_own_production_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let mut expected = std::collections::BTreeSet::new();
        let mut published = std::collections::BTreeSet::new();
        for (provider, route) in every_route() {
            let slug = route_label(route).expect("a public route has a slug");
            for (bits, quant) in SHIPPED_TIERS {
                let tier = calibration_tier_label(quant).unwrap();
                expected.insert(match (slug, tier) {
                    ("quality", "q8") => QUALITY_CALIBRATION_FINGERPRINT.to_owned(),
                    ("fast", "q8") => FAST_CALIBRATION_FINGERPRINT.to_owned(),
                    _ => format!("sensenova-u1-{slug}-{tier}-mlx-shared-ladder-v1"),
                });
                let root = tier_root(&tmp, route, bits);
                let spec = tier_spec(&root, route, quant);
                let label = format!("{provider} {route} {tier}");
                assert_eq!(resolved_artifact_tier(&spec).unwrap(), quant, "{label}");
                let contract = memory_strategy_contract(provider, &spec).unwrap();
                let identity = contract
                    .calibration
                    .as_ref()
                    .unwrap_or_else(|| panic!("{label}: no production identity"));
                assert_eq!(identity.load_shape, spec.load_shape, "{label}");
                assert_eq!(
                    Some(identity.fingerprint.clone()),
                    production_calibration_fingerprint(provider, &spec),
                    "{label}"
                );
                assert!(
                    published.insert(identity.fingerprint.clone()),
                    "{label}: two cells share the identity {}",
                    identity.fingerprint
                );
            }
        }
        assert_eq!(published, expected);
        assert_eq!(published.len(), every_route().len() * SHIPPED_TIERS.len());
    }

    /// **The two measured `2026-08-03` literals are returned unchanged at the coordinates they were
    /// captured on**, and nowhere else (sc-22734). They are the only strings in the table that are
    /// not derived from the (route, tier) template, so a template change must not silently restamp
    /// them.
    #[test]
    fn the_measured_literals_survive_at_their_measured_coordinates() {
        let tmp = tempfile::tempdir().unwrap();
        for (provider, route, expected) in [
            (
                crate::MODEL_ID,
                "sensenova_u1_8b",
                "sensenova-u1-quality-q8-mlx-shared-ladder-2026-08-03-v1",
            ),
            (
                crate::MODEL_ID_FAST,
                "sensenova_u1_8b_fast",
                "sensenova-u1-fast-q8-mlx-shared-ladder-2026-08-03-v1",
            ),
        ] {
            let root = tier_root(&tmp, route, Some(8));
            let spec = tier_spec(&root, route, Some(Quant::Q8));
            assert_eq!(
                memory_strategy_contract(provider, &spec)
                    .unwrap()
                    .calibration
                    .unwrap()
                    .fingerprint,
                expected
            );
            // Every OTHER tier of the same route is the derived string, never the measured one.
            for (bits, quant) in [(None, None), (Some(4), Some(Quant::Q4))] {
                let other = tier_root(&tmp, route, bits);
                let spec = tier_spec(&other, route, quant);
                assert_ne!(
                    production_calibration_fingerprint(provider, &spec).as_deref(),
                    Some(expected)
                );
            }
        }
        assert_eq!(
            QUALITY_CALIBRATION_FINGERPRINT,
            "sensenova-u1-quality-q8-mlx-shared-ladder-2026-08-03-v1"
        );
        assert_eq!(
            FAST_CALIBRATION_FINGERPRINT,
            "sensenova-u1-fast-q8-mlx-shared-ladder-2026-08-03-v1"
        );
    }

    /// **No production string is ever a weights-free string.** A registry fixture contract carries
    /// [`STATIC_BEHAVIOR_CALIBRATION`], a namespace the production table cannot reach, so a
    /// weights-free declaration can never be filed as measured evidence of the cell it describes.
    #[test]
    fn no_production_identity_collides_with_the_weights_free_namespace() {
        let tmp = tempfile::tempdir().unwrap();
        let mut production = std::collections::BTreeSet::new();
        let mut weights_free = std::collections::BTreeSet::new();
        for (provider, route) in every_route() {
            for (bits, quant) in SHIPPED_TIERS {
                let root = tier_root(&tmp, route, bits);
                let spec = tier_spec(&root, route, quant);
                production.insert(
                    production_calibration_fingerprint(provider, &spec)
                        .expect("a shipped cell has a production identity"),
                );
                weights_free.insert(
                    weights_free_memory_strategy_contract(provider, &spec)
                        .unwrap()
                        .calibration
                        .unwrap()
                        .fingerprint,
                );
            }
        }
        assert!(production.is_disjoint(&weights_free));
        for fingerprint in &production {
            assert!(
                !fingerprint.starts_with(STATIC_BEHAVIOR_CALIBRATION),
                "{fingerprint} reaches the static behavior namespace"
            );
        }
    }

    /// **The identity is bound to the tier of the artifact on disk, at the seam the worker calls.**
    /// A packed root publishes its own tier's string; the SAME root asked for the other packed tier
    /// or for the dense one publishes nothing; a dense root asked for q4 publishes nothing while the
    /// dense request publishes the bf16 string.
    ///
    /// Mutation that fails this: deleting the
    /// `resolved_artifact_tier(spec) != requested_tier(spec)` refusal in
    /// `production_calibration_identity` — every mismatched cell publishes the requested tier's
    /// string over another tier's weights.
    #[test]
    fn the_production_identity_is_withheld_when_the_request_and_the_artifact_disagree() {
        let tmp = tempfile::tempdir().unwrap();
        let identity = |provider: &str, spec: &LoadSpec| {
            memory_strategy_contract(provider, spec)
                .unwrap()
                .calibration
                .map(|calibration| calibration.fingerprint)
        };
        for (provider, route) in every_route() {
            let slug = route_label(route).unwrap();
            // A q4-packed root: q4 publishes q4, q8 and bf16 publish nothing.
            let q4 = tier_root(&tmp, route, Some(4));
            assert_eq!(
                identity(provider, &tier_spec(&q4, route, Some(Quant::Q4))),
                Some(format!("sensenova-u1-{slug}-q4-mlx-shared-ladder-v1")),
                "{route}"
            );
            for mismatch in [Some(Quant::Q8), None] {
                assert_eq!(
                    identity(provider, &tier_spec(&q4, route, mismatch)),
                    None,
                    "{route}: q4 weights published an identity for {mismatch:?}"
                );
            }
            // A dense root: bf16 publishes bf16, q4 (a load-time requantization) publishes nothing.
            let dense = tier_root(&tmp, route, None);
            assert_eq!(
                identity(provider, &tier_spec(&dense, route, None)),
                Some(format!("sensenova-u1-{slug}-bf16-mlx-shared-ladder-v1")),
                "{route}"
            );
            assert_eq!(
                identity(provider, &tier_spec(&dense, route, Some(Quant::Q4))),
                None,
                "{route}: dense weights published a q4 identity"
            );
            // The TABLE still answers for the request knob in all of those cases — the refusal is
            // the binding's, so a table change can never be mistaken for the binding working.
            assert!(production_calibration_fingerprint(
                provider,
                &tier_spec(&dense, route, Some(Quant::Q4))
            )
            .is_some());
        }
    }

    /// **An unreadable or malformed checkpoint publishes no identity and NEVER fails the load.**
    /// The contract itself is still produced in every case — an admission gate that threw here
    /// would turn a measurement gap into an outage.
    #[test]
    fn an_unreadable_artifact_publishes_no_identity_without_failing_the_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let route = "sensenova_u1_8b";
        let repository = format!("SceneWorks__{}-mlx", route.replace('_', "-"));

        let make = |label: &str, entries: Option<Vec<(String, &'static str, Vec<usize>)>>| {
            let root = unique_root(&tmp, label).join(&repository);
            std::fs::create_dir_all(&root).unwrap();
            match entries {
                Some(entries) => write_safetensors(&root.join("model.safetensors"), &entries),
                // Not a safetensors file at all.
                None => std::fs::write(root.join("model.safetensors"), [0_u8; 8]).unwrap(),
            }
            root
        };

        // A packed base whose codes/scales ratio is not a shipped width.
        let two_bit = linear_entries("language_model.model.layers.0.self_attn.q_proj", 8, Some(2));
        // Two backbone bases packed at different widths.
        let mut mixed =
            linear_entries("language_model.model.layers.0.self_attn.q_proj", 8, Some(4));
        mixed.extend(linear_entries(
            "language_model.model.layers.0.mlp.gate_proj",
            8,
            Some(8),
        ));
        // A `.scales` with no codes companion.
        let orphan = vec![(
            "language_model.model.layers.0.self_attn.q_proj.scales".to_owned(),
            "BF16",
            vec![8, 1],
        )];
        // No backbone decoder Linears at all.
        let no_backbone = vec![(
            "language_model.model.embed_tokens.weight".to_owned(),
            "BF16",
            vec![16, 64],
        )];

        for (label, root) in [
            ("unreadable bytes", make("unreadable", None)),
            ("2-bit pack", make("two-bit", Some(two_bit))),
            ("mixed q4/q8 pack", make("mixed", Some(mixed))),
            ("orphan scales", make("orphan", Some(orphan))),
            (
                "no backbone linears",
                make("no-backbone", Some(no_backbone)),
            ),
        ] {
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let spec = tier_spec(&root, route, quant);
                assert!(
                    resolved_artifact_tier(&spec).is_err(),
                    "{label} {quant:?}: read a tier it cannot prove"
                );
                let contract = memory_strategy_contract(crate::MODEL_ID, &spec)
                    .unwrap_or_else(|error| panic!("{label} {quant:?}: contract failed: {error}"));
                assert!(
                    contract.calibration.is_none(),
                    "{label} {quant:?}: published an identity over unreadable weights"
                );
            }
        }
        // A source that is not a snapshot directory fails closed the same way.
        let mut file_spec = LoadSpec::new(WeightsSource::File("model.safetensors".into()));
        file_spec.quantize = Some(Quant::Q8);
        assert!(resolved_artifact_tier(&file_spec).is_err());
    }

    /// **A `_fast` root without the pre-merged distill marker publishes no identity, on any tier.**
    /// Such a root loads an unmerged base and merges a curated LoRA at runtime — a different
    /// resident shape from the one the `_fast` campaign measured, and the same requirement
    /// [`can_stream_gen_with_artifact`] holds rung 4 to.
    #[test]
    fn a_fast_root_without_the_merged_marker_publishes_no_identity() {
        let tmp = tempfile::tempdir().unwrap();
        for route in FAST_PUBLIC_ROUTES {
            for (bits, quant) in SHIPPED_TIERS {
                let root = tier_root(&tmp, route, bits);
                assert!(memory_strategy_contract(
                    crate::MODEL_ID_FAST,
                    &tier_spec(&root, route, quant)
                )
                .unwrap()
                .calibration
                .is_some());
                std::fs::remove_file(root.join(crate::DISTILL_MERGED_MARKER)).unwrap();
                let contract =
                    memory_strategy_contract(crate::MODEL_ID_FAST, &tier_spec(&root, route, quant))
                        .unwrap();
                assert!(
                    contract.calibration.is_none(),
                    "{route} {quant:?}: an unmerged fast root published an identity"
                );
            }
        }
    }

    /// **A composition the measured load never carried zeroes the identity**, on every route and
    /// tier. These are the guards the pre-sc-22734 gate held and sc-22734 retains: the loader
    /// silently ignores most of them, so a published identity would key a rung to a composition
    /// nothing measured.
    #[test]
    fn an_unmeasured_composition_publishes_no_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let route = "sensenova_u1_8b";
        let root = tier_root(&tmp, route, Some(8));
        let base = tier_spec(&root, route, Some(Quant::Q8));
        assert!(production_calibration_identity(
            crate::MODEL_ID,
            &base,
            verified_artifact(&base).as_ref()
        )
        .is_some());
        let overlay = || WeightsSource::File(root.join("overlay.safetensors"));
        // One unmeasured-composition mutation, applied to a spec that otherwise publishes.
        type Mutation<'a> = (&'a str, Box<dyn Fn(&mut LoadSpec) + 'a>);
        let mutations: Vec<Mutation<'_>> = vec![
            (
                "precision override",
                Box::new(|spec: &mut LoadSpec| spec.precision = mlx_gen::Precision::Fp32),
            ),
            (
                "user adapter",
                Box::new(|spec: &mut LoadSpec| {
                    spec.adapters.push(mlx_gen::gen_core::AdapterSpec::new(
                        "lora.safetensors".into(),
                        1.0,
                        mlx_gen::gen_core::AdapterKind::Lora,
                    ));
                }),
            ),
            (
                "component overlay",
                Box::new(|spec: &mut LoadSpec| {
                    spec.components
                        .insert(DISTILL_LORA_COMPONENT.to_owned(), overlay());
                }),
            ),
            (
                "control",
                Box::new(|spec: &mut LoadSpec| spec.control = Some(overlay())),
            ),
            (
                "extra control",
                Box::new(|spec: &mut LoadSpec| spec.extra_controls.push(overlay())),
            ),
            (
                "ip adapter",
                Box::new(|spec: &mut LoadSpec| spec.ip_adapter = Some(overlay())),
            ),
            (
                "pid",
                Box::new(|spec: &mut LoadSpec| {
                    spec.pid = Some(mlx_gen::gen_core::PidWeights {
                        checkpoint: overlay(),
                        gemma: WeightsSource::Dir(root.clone()),
                    });
                }),
            ),
            (
                "identity",
                Box::new(|spec: &mut LoadSpec| {
                    spec.identity = Some(mlx_gen::gen_core::IdentityWeights {
                        encoder: Some(overlay()),
                        eva: Some(overlay()),
                        face_dir: Some(WeightsSource::Dir(root.clone())),
                    });
                }),
            ),
            (
                "text encoder",
                Box::new(|spec: &mut LoadSpec| spec.text_encoder = Some(overlay())),
            ),
        ];
        for (label, mutate) in mutations {
            let mut spec = base.clone();
            mutate(&mut spec);
            let artifact = verified_artifact(&spec);
            assert!(
                production_calibration_identity(crate::MODEL_ID, &spec, artifact.as_ref())
                    .is_none(),
                "{label}: published an identity for an unmeasured composition"
            );
        }
        // The pinned artifact is a precondition too: no pin, no identity.
        assert!(production_calibration_identity(crate::MODEL_ID, &base, None).is_none());
    }

    /// The two recorded campaign artifact digests stay distinct, well-formed SHA-256 hex, and
    /// [`verified_artifact_identity`] still produces that shape over a real snapshot — the evidence
    /// [`QUALITY_CALIBRATION_FINGERPRINT`] and [`FAST_CALIBRATION_FINGERPRINT`] were measured on,
    /// and the gate [`validate_runner_gate`] still holds the real-weight runner to.
    #[test]
    fn the_recorded_campaign_artifact_digests_stay_distinct_sha256_hex() {
        let hex = |digest: &str| {
            digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        };
        assert!(hex(QUALITY_Q8_ARTIFACT), "{QUALITY_Q8_ARTIFACT}");
        assert!(hex(FAST_Q8_ARTIFACT), "{FAST_Q8_ARTIFACT}");
        assert_ne!(QUALITY_Q8_ARTIFACT, FAST_Q8_ARTIFACT);
        let tmp = tempfile::tempdir().unwrap();
        let root = tier_root(&tmp, "sensenova_u1_8b", Some(8));
        let spec = tier_spec(&root, "sensenova_u1_8b", Some(Quant::Q8));
        let observed = verified_artifact_identity(&spec).expect("a pinned single-file snapshot");
        assert!(hex(&observed), "{observed}");
        assert_ne!(observed, QUALITY_Q8_ARTIFACT);
    }

    /// sc-20569 (production outage): every geometry the manifest offers is outside the measured
    /// 1024x1024 cell, so an unconditional envelope refusal rejected every product-legal request.
    /// A caller that does NOT claim measured authority — `AdmissionPath::Legacy` in the SceneWorks
    /// fit gate, whether it synthesized an estimate ladder or froze to the resident baseline — must
    /// be ADMITTED, and must be able to open a request scope.
    #[test]
    fn every_manifest_geometry_and_count_degrades_to_legacy_admission() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        // Pre-merged turnkey: the only shape on which the `_fast` route declares rung 4, so the
        // deepest optimized rung is covered on BOTH engine ids rather than only the quality one.
        std::fs::write(root.join(crate::DISTILL_MERGED_MARKER), b"{}\n").unwrap();
        let spec = spec.with_quant(Quant::Q8);
        let mut admitted = 0;
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract = calibrated_contract(provider_id, &spec);
            for (width, height) in MANIFEST_RESOLUTIONS {
                for batch in MANIFEST_COUNTS {
                    for (authority, strategy) in [
                        (
                            MemoryOptimizationAuthority::Estimated,
                            MemoryStrategy::BoundedAttention,
                        ),
                        (
                            MemoryOptimizationAuthority::Estimated,
                            MemoryStrategy::BoundedTransformerResidency,
                        ),
                        (
                            MemoryOptimizationAuthority::Resident,
                            MemoryStrategy::Resident,
                        ),
                    ] {
                        let label =
                            format!("{provider_id} {width}x{height} batch {batch} {authority:?}");
                        let mut context = route_context(&contract, strategy, t2i_route());
                        context.optimization_authority = authority;
                        context.geometry.width = width;
                        context.geometry.height = height;
                        context.geometry.batch = batch;
                        assert_eq!(
                            safety_check(&contract, Some(Quant::Q8), &context),
                            MemorySafetyDecision::Accept,
                            "{label}"
                        );
                        assert!(
                            begin_request(
                                provider_id,
                                &contract,
                                Some(Quant::Q8),
                                &context,
                                mlx_gen::request_scope::MlxScopeCleanup::None,
                            )
                            .unwrap_or_else(|error| panic!("{label}: {error}"))
                            .is_some(),
                            "{label}"
                        );
                        admitted += 1;
                    }
                }
            }
        }
        assert_eq!(
            admitted,
            2 * MANIFEST_RESOLUTIONS.len() * MANIFEST_COUNTS.len() * 3,
            "two engine ids x seven manifest geometries x three counts x three legacy dispositions"
        );
        std::fs::remove_dir_all(root).ok();
    }

    /// The other half of the degrade: a context that DOES claim measured authority is still held to
    /// the measured cell, because admitting it would grade the request against evidence captured at
    /// a different geometry. Each geometry axis is mutated on its own so the assertion proves every
    /// conjunct rather than the set.
    #[test]
    fn a_measured_authority_claim_outside_the_campaign_cell_still_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract = calibrated_contract(provider_id, &spec);
            let measured = route_context(&contract, MemoryStrategy::BoundedAttention, t2i_route());
            assert_eq!(
                measured.optimization_authority,
                MemoryOptimizationAuthority::Calibrated,
                "the shared behavior context must still claim measured authority"
            );
            assert_eq!(
                safety_check(&contract, Some(Quant::Q8), &measured),
                MemorySafetyDecision::Accept,
                "{provider_id}: the measured cell itself must keep admitting"
            );

            let mut mutations: Vec<(String, MemoryRunContext)> = Vec::new();
            for (width, height) in MANIFEST_RESOLUTIONS {
                let mut context = measured.clone();
                context.geometry.width = width;
                context.geometry.height = height;
                mutations.push((format!("{width}x{height}"), context));
            }
            for batch in [2, 4] {
                let mut context = measured.clone();
                context.geometry.batch = batch;
                mutations.push((format!("batch {batch}"), context));
            }
            let mut frames = measured.clone();
            frames.geometry.frames = 2;
            mutations.push(("frames 2".to_owned(), frames));

            for (label, context) in mutations {
                assert!(
                    matches!(
                        safety_check(&contract, Some(Quant::Q8), &context),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains("calibrated memory geometry")
                    ),
                    "{provider_id} {label}: a measured claim off the campaign cell must fail closed"
                );
            }
        }
        let contract = calibrated_contract(crate::MODEL_ID, &spec);
        let mut vqa_btr = route_context(
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("vqa".into()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        );
        vqa_btr.optimization_authority = MemoryOptimizationAuthority::Estimated;
        assert!(matches!(
            safety_check(&contract, Some(Quant::Q8), &vqa_btr),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("structurally not applicable to VQA")
        ));
        std::fs::remove_dir_all(root).ok();
    }

    /// The mode/reference conjunct is the same kind of envelope statement as the geometry one, and
    /// `character_image` (image-to-image with one reference) is a shipped SenseNova capability. It
    /// degrades on a legacy claim and fails closed on a measured one.
    #[test]
    fn uncalibrated_route_modes_degrade_but_a_measured_claim_still_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract = calibrated_contract(provider_id, &spec);
            for (mode, reference_count) in [
                (MemoryMode::ImageToImage, 1),
                (MemoryMode::Edit, 2),
                (MemoryMode::Other("character_image".into()), 1),
            ] {
                let label = format!("{provider_id} {mode:?}/{reference_count}");
                let measured = route_context(
                    &contract,
                    MemoryStrategy::BoundedAttention,
                    MemoryBehaviorRoute {
                        mode,
                        reference_count,
                        use_pid: false,
                        has_phases: false,
                        overlay: None,
                    },
                );
                assert!(
                    matches!(
                        safety_check(&contract, Some(Quant::Q8), &measured),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains(
                                "exactly TextToImage with zero references and Edit with one reference"
                            )
                    ),
                    "{label}: a measured claim on an unmeasured route must fail closed"
                );
                let mut legacy = measured.clone();
                legacy.optimization_authority = MemoryOptimizationAuthority::Estimated;
                assert_eq!(
                    safety_check(&contract, Some(Quant::Q8), &legacy),
                    MemorySafetyDecision::Accept,
                    "{label}: a legacy/estimated claim must degrade, not refuse"
                );
            }
            for (mode, reference_count) in [(MemoryMode::TextToImage, 1), (MemoryMode::Edit, 0)] {
                let mut context = route_context(
                    &contract,
                    MemoryStrategy::BoundedAttention,
                    MemoryBehaviorRoute {
                        mode,
                        reference_count,
                        use_pid: false,
                        has_phases: false,
                        overlay: None,
                    },
                );
                context.optimization_authority = MemoryOptimizationAuthority::Estimated;
                assert!(matches!(
                    safety_check(&contract, Some(Quant::Q8), &context),
                    MemorySafetyDecision::Reject { reason }
                        if reason.contains("unsupported SenseNova route")
                ));
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// The degrade must not weaken the STRUCTURAL refusals. No amount of estimate authority makes
    /// SenseNova grow a PiD seam, an overlay axis, or a multi-phase trajectory. Each axis is mutated
    /// on its own so every guard is asked its own question.
    #[test]
    fn structural_refusals_are_not_weakened_by_the_legacy_degrade() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract = calibrated_contract(provider_id, &spec);
            let mut legacy =
                route_context(&contract, MemoryStrategy::BoundedAttention, t2i_route());
            legacy.optimization_authority = MemoryOptimizationAuthority::Estimated;
            // The unmutated legacy context is admitted, so each rejection below is attributable to
            // the one axis it mutates.
            assert_eq!(
                safety_check(&contract, Some(Quant::Q8), &legacy),
                MemorySafetyDecision::Accept,
                "{provider_id}"
            );

            let mut use_pid = legacy.clone();
            use_pid.use_pid = true;
            let mut overlay = legacy.clone();
            overlay.overlay = Some("character".to_owned());
            let mut has_phases = legacy.clone();
            has_phases.has_phases = true;
            for (label, context, needle) in [
                ("use_pid", use_pid, "no PiD or overlay seam"),
                ("overlay", overlay, "no PiD or overlay seam"),
                ("has_phases", has_phases, "single-phase trajectory"),
            ] {
                assert!(
                    matches!(
                        safety_check(&contract, Some(Quant::Q8), &context),
                        MemorySafetyDecision::Reject { reason } if reason.contains(needle)
                    ),
                    "{provider_id} {label}: structural refusal must survive the legacy degrade"
                );
                assert!(
                    begin_request(
                        provider_id,
                        &contract,
                        Some(Quant::Q8),
                        &context,
                        mlx_gen::request_scope::MlxScopeCleanup::None,
                    )
                    .is_err(),
                    "{provider_id} {label}"
                );
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// sc-20569 secondary: the production log read `unsupported: unsupported: sensenova_u1_8b_fast:
    /// …` because the route gate built a full `CoreError::Unsupported`, the decision kept only its
    /// RENDERED string, and `begin_request` typed that string as `Unsupported` again. Exactly one
    /// prefix must reach the user.
    #[test]
    fn a_route_refusal_surfaces_the_unsupported_prefix_exactly_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        let contract = calibrated_contract(crate::MODEL_ID_FAST, &spec);
        let mut context = route_context(&contract, MemoryStrategy::BoundedAttention, t2i_route());
        context.geometry.width = 2048;
        context.geometry.height = 2048;
        let error = match begin_request(
            crate::MODEL_ID_FAST,
            &contract,
            Some(Quant::Q8),
            &context,
            mlx_gen::request_scope::MlxScopeCleanup::None,
        ) {
            Ok(_) => panic!("a measured claim off the campaign cell must still refuse"),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            error.matches("unsupported: ").count(),
            1,
            "the rendered refusal must carry one `unsupported: ` prefix: {error}"
        );
        assert!(
            error.starts_with(&format!("unsupported: {}: ", crate::MODEL_ID_FAST)),
            "{error}"
        );
        std::fs::remove_dir_all(root).ok();
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

    /// AC (SC-22662): both registered SenseNova routes publish the axes of the Qwen3 MoT backbone
    /// they generate through, and pass the shared facts conformance check.
    ///
    /// The three latent/VAE axes stay absent for one structural reason: this provider has no
    /// autoencoder — the FM head emits RGB patches directly. `patch_size` is that head's pixel
    /// patch, a real `config.json` key, published on both backends (SC-22667 parity).
    #[test]
    fn architecture_facts_follow_the_backbone_config_and_declare_no_latent() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract = weights_free_memory_strategy_contract(provider_id, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(32),
                    head_dim: Some(128),
                    transformer_blocks: Some(42),
                    // The FM head's 16 px patch: `config.json:patch_size`, here the parser's
                    // `DEFAULT_PATCH_SIZE` because the fixture ships no config.
                    patch_size: Some(16),
                    latent_channels: None,
                    vae_spatial_scale: None,
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(2),
                },
                "{provider_id} architecture facts"
            );
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
        std::fs::remove_dir_all(root).ok();
    }

    fn spec_for_config(dir: &std::path::Path, config: &serde_json::Value) -> LoadSpec {
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    /// AC (SC-22662, review follow-up): on the **materialized** path the backbone axes come from
    /// the snapshot's own `config.json` — the file `NeoChatConfig::from_dir` parses at load —
    /// rather than from the compile-time preset. The shipped 8B-MoT fixture agrees with the
    /// weights-free path; a fixture with mutated `llm_config` keys publishes the mutated axes,
    /// which is the assertion the unconditional `architecture_facts()` this replaced would fail.
    #[test]
    fn materialized_backbone_axes_come_from_the_snapshot_rather_than_the_preset() {
        let shipped: serde_json::Value =
            serde_json::from_str(crate::config::MOT_8B_CONFIG).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let (_, weights_free) = fixture_spec(&tmp);
        let mirror = tempfile::tempdir().unwrap();
        assert_eq!(
            architecture_facts(&spec_for_config(mirror.path(), &shipped)),
            architecture_facts(&weights_free),
            "the shipped 8B-MoT config must publish the preset's axes"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = shipped.clone();
        mutated["llm_config"]["num_hidden_layers"] = serde_json::json!(7);
        mutated["llm_config"]["head_dim"] = serde_json::json!(64);
        let mutated_facts = architecture_facts(&spec_for_config(mutated_dir.path(), &mutated));
        assert_eq!(
            (mutated_facts.transformer_blocks, mutated_facts.head_dim),
            (Some(7), Some(64)),
            "the materialized path must publish the snapshot's geometry, not the preset's"
        );
    }

    /// Feature-end review (SC-22667, cross-story parity): the FM head's `patch_size` is a real
    /// top-level `config.json` key this crate parses, so it is published from the snapshot like the
    /// Candle sibling does, and the activation width follows the store dtype the loader admits
    /// rather than a literal.
    ///
    /// Mutations that fail this: `patch_size: None` (the shape under review) fails the first two
    /// assertions; `activation_dtype_width: Some(HALF_ACTIVATION_WIDTH)` unconditionally (the other
    /// shape under review) fails the `Fp32` assertion, because `model.rs` refuses that precision
    /// and no bf16 store is ever loaded for it.
    #[test]
    fn fm_head_patch_and_store_width_follow_the_snapshot_and_the_admitted_precision() {
        let shipped: serde_json::Value =
            serde_json::from_str(crate::config::MOT_8B_CONFIG).unwrap();
        let mirror = tempfile::tempdir().unwrap();
        let facts = architecture_facts(&spec_for_config(mirror.path(), &shipped));
        assert_eq!(
            facts.patch_size,
            Some(crate::config::DEFAULT_PATCH_SIZE as u32),
            "the shipped config's patch_size is the parser default"
        );

        let mutated_dir = tempfile::tempdir().unwrap();
        let mut mutated = shipped.clone();
        mutated["patch_size"] = serde_json::json!(8);
        assert_eq!(
            architecture_facts(&spec_for_config(mutated_dir.path(), &mutated)).patch_size,
            Some(8),
            "the materialized path must publish the snapshot's patch, not the default"
        );

        let bf16 = spec_for_config(mirror.path(), &shipped);
        assert_eq!(architecture_facts(&bf16).activation_dtype_width, Some(2));
        let mut fp32 = bf16.clone();
        fp32.precision = mlx_gen::Precision::Fp32;
        assert_eq!(
            architecture_facts(&fp32).activation_dtype_width,
            None,
            "a precision the loader refuses has no store width to publish"
        );
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
    fn public_route_identities_are_exact_and_provider_partitioned() {
        assert_eq!(
            public_routes(crate::MODEL_ID).unwrap(),
            [
                "sensenova_u1_8b",
                "sensenova_u1_8b_infographic_v2",
                "sensenova_u1_8b_infographic_v3",
            ]
        );
        assert_eq!(
            public_routes(crate::MODEL_ID_FAST).unwrap(),
            [
                "sensenova_u1_8b_fast",
                "sensenova_u1_8b_infographic_v2_fast",
                "sensenova_u1_8b_infographic_v3_fast",
            ]
        );

        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let tier = match quant {
                Some(quant) => spec.clone().with_quant(quant),
                None => spec.clone(),
            };
            for route in QUALITY_PUBLIC_ROUTES {
                assert!(weights_free_memory_strategy_contract(
                    crate::MODEL_ID,
                    &tier.clone().with_resolved_route(*route)
                )
                .is_ok());
                assert!(weights_free_memory_strategy_contract(
                    crate::MODEL_ID_FAST,
                    &tier.clone().with_resolved_route(*route)
                )
                .is_err());
            }
            for route in FAST_PUBLIC_ROUTES {
                assert!(weights_free_memory_strategy_contract(
                    crate::MODEL_ID_FAST,
                    &tier.clone().with_resolved_route(*route)
                )
                .is_ok());
                assert!(weights_free_memory_strategy_contract(
                    crate::MODEL_ID,
                    &tier.clone().with_resolved_route(*route)
                )
                .is_err());
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn each_public_route_is_bound_to_its_repository_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        for route in QUALITY_PUBLIC_ROUTES.iter().chain(FAST_PUBLIC_ROUTES) {
            let repository = expected_repository(route).unwrap();
            let exact = LoadSpec::new(WeightsSource::Dir(
                tmp.path().join(repository).join("snapshots/revision/q8"),
            ))
            .with_resolved_route(*route);
            assert!(
                validate_resolved_artifact_binding(&exact).is_ok(),
                "{route}"
            );

            let crossed_route = QUALITY_PUBLIC_ROUTES
                .iter()
                .chain(FAST_PUBLIC_ROUTES)
                .find(|candidate| *candidate != route)
                .unwrap();
            assert!(validate_resolved_artifact_binding(
                &exact.clone().with_resolved_route(*crossed_route)
            )
            .is_err());
        }
    }

    #[test]
    fn packed_tier_provenance_refuses_crossed_numeric_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("tier");
        std::fs::create_dir(&root).unwrap();
        let base = LoadSpec::new(WeightsSource::Dir(root.clone()));

        std::fs::write(root.join("config.json"), "{}").unwrap();
        assert!(validate_artifact_tier(&base).is_ok());
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q4)).is_ok());

        std::fs::write(
            root.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q4)).is_ok());
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q8)).is_err());
        assert!(validate_artifact_tier(&base).is_err());

        std::fs::write(
            root.join("config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        assert!(validate_artifact_tier(&base.with_quant(Quant::Q8)).is_ok());
    }

    #[test]
    fn direct_execution_requires_the_exact_admitted_mode_and_geometry() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        let contract = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let context = route_context(
            &contract,
            MemoryStrategy::BoundedAttention,
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("vqa".into()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        );
        assert!(validate_direct_operation_identity(
            crate::MODEL_ID,
            &context,
            &context.mode,
            context.geometry,
        )
        .is_ok());
        assert!(validate_direct_operation_identity(
            crate::MODEL_ID,
            &context,
            &MemoryMode::Other("interleave".into()),
            context.geometry,
        )
        .is_err());
        let mut crossed_geometry = context.geometry;
        crossed_geometry.reference_count = 2;
        assert!(validate_direct_operation_identity(
            crate::MODEL_ID,
            &context,
            &context.mode,
            crossed_geometry,
        )
        .is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn advertised_modes_have_exact_structural_applicability() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let spec = spec.with_quant(Quant::Q8);
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract = calibrated_contract(provider_id, &spec);
            let routes = [
                (MemoryMode::TextToImage, 0, true),
                (MemoryMode::Edit, 1, true),
                (MemoryMode::Edit, 5, true),
                (MemoryMode::ImageToImage, 5, true),
                (MemoryMode::Other("character_image".into()), 5, true),
                (
                    MemoryMode::Other("vqa".into()),
                    1,
                    provider_id == crate::MODEL_ID,
                ),
                (
                    MemoryMode::Other("interleave".into()),
                    10,
                    provider_id == crate::MODEL_ID,
                ),
                (MemoryMode::Edit, 6, false),
                (MemoryMode::Other("vqa".into()), 0, false),
                (MemoryMode::Other("interleave".into()), 11, false),
            ];
            for (mode, reference_count, supported) in routes {
                let mut context = route_context(
                    &contract,
                    MemoryStrategy::BoundedAttention,
                    MemoryBehaviorRoute {
                        mode,
                        reference_count,
                        use_pid: false,
                        has_phases: false,
                        overlay: None,
                    },
                );
                context.optimization_authority = MemoryOptimizationAuthority::Estimated;
                assert_eq!(
                    safety_check(&contract, Some(Quant::Q8), &context)
                        == MemorySafetyDecision::Accept,
                    supported,
                    "{provider_id} {}/{}",
                    context.mode.as_key(),
                    reference_count
                );
                if supported {
                    let scope = begin_request(
                        provider_id,
                        &contract,
                        Some(Quant::Q8),
                        &context,
                        mlx_gen::request_scope::MlxScopeCleanup::None,
                    )
                    .unwrap()
                    .expect("every advertised direct/registry route owns a request scope");
                    drop(scope);
                }
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn interleave_admission_binds_one_through_ten_generated_images() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture_spec(&tmp);
        let contract = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        for count in 1..=10 {
            let mut context = route_context(
                &contract,
                MemoryStrategy::BoundedAttention,
                MemoryBehaviorRoute {
                    mode: MemoryMode::Other("interleave".into()),
                    reference_count: 0,
                    use_pid: false,
                    has_phases: false,
                    overlay: None,
                },
            );
            context.optimization_authority = MemoryOptimizationAuthority::Estimated;
            context.geometry.batch = count;
            assert_eq!(
                safety_check(&contract, Some(Quant::Q8), &context),
                MemorySafetyDecision::Accept
            );
        }
        for count in [0, 11] {
            let mut context = route_context(
                &contract,
                MemoryStrategy::BoundedAttention,
                MemoryBehaviorRoute {
                    mode: MemoryMode::Other("interleave".into()),
                    reference_count: 0,
                    use_pid: false,
                    has_phases: false,
                    overlay: None,
                },
            );
            context.optimization_authority = MemoryOptimizationAuthority::Estimated;
            context.geometry.batch = count;
            assert!(matches!(
                safety_check(&contract, Some(Quant::Q8), &context),
                MemorySafetyDecision::Reject { .. }
            ));
        }
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
            context(MemoryMode::Edit, 2),
        ] {
            assert!(matches!(
                safety_check(&contract, Some(Quant::Q8), &rejected),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("exactly TextToImage with zero references and Edit with one reference")
            ));
        }
        for rejected in [
            context(MemoryMode::TextToImage, 1),
            context(MemoryMode::Edit, 0),
        ] {
            assert!(matches!(
                safety_check(&contract, Some(Quant::Q8), &rejected),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("unsupported SenseNova route")
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
