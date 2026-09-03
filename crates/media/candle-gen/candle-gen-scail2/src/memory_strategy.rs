//! Exact Resident-only Candle/CUDA memory contract for public SCAIL-2 character animation.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterKind, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryStructuralResidentEvidence, MemoryWindowMaterialization, Precision,
    ResidentOnlyMemoryContractRegistration, ResidentRequestMemory, WeightsSource,
    MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
};
use sha2::{Digest, Sha256};

pub const PROVIDER_ID: &str = crate::pipeline::MODEL_ID;
pub const CANONICAL_REPOSITORY: &str = "SceneWorks/scail2-mlx";
pub const CANONICAL_REVISION: &str = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";
pub const PUBLIC_BUCKETS: &[(u32, u32)] = &[(832, 480), (480, 832), (1280, 704), (704, 1280)];
pub const PUBLIC_FRAMES: &[u32] = &[45, 61, 77];
/// Domain separator for the provider-side animation-carrier content digest.
const CARRIER_DIGEST_DOMAIN: &str = "scail2-candle-carrier-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterRealization {
    Lora,
    Lokr,
    Loha,
    FoldedFullRank,
    Hybrid,
}

#[derive(Clone, Debug)]
struct AdapterReceipt {
    ordinal: usize,
    pin: gen_core::PinnedWeightsFile,
    canonical_path: PathBuf,
    digest: String,
    kind: AdapterKind,
    scale_bits: u32,
    pass_scales: Option<Vec<u32>>,
    expert: Option<gen_core::MoeExpert>,
    realization: AdapterRealization,
    physical_bytes: u64,
    transient_bytes: u64,
}

impl AdapterReceipt {
    fn capture(
        spec: &LoadSpec,
        ordinal: usize,
        adapter: &gen_core::AdapterSpec,
    ) -> gen_core::Result<Self> {
        if !adapter.scale.is_finite()
            || adapter
                .pass_scales
                .as_ref()
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter {ordinal} has a non-finite scale"
            )));
        }
        if adapter.pass_scales.is_some() || adapter.moe_expert.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter {ordinal} must use its full-run scale and no expert target"
            )));
        }
        let lexical = std::path::absolute(&adapter.path)?;
        let pin = if spec.prepared_file_pins().is_prepared() {
            spec.prepared_file_pins()
                .get(&lexical)
                .cloned()
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "{PROVIDER_ID}: sealed receipt is missing adapter {ordinal} at {}",
                        lexical.display()
                    ))
                })?
        } else {
            gen_core::PinnedWeightsFile::pin(&lexical)?
        };
        pin.ensure_unchanged()?;
        let canonical_path = pin.canonical_target_path().to_path_buf();
        let (digest, realization, physical_bytes) = pin.read_unchanged(|path| {
            let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
            if headers.is_empty() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: adapter {ordinal} contains no tensors"
                )));
            }
            let mut file = std::fs::File::open(path)?;
            let mut hasher = Sha256::new();
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            let mut has_diff = false;
            let mut has_lora = false;
            let mut has_lokr = false;
            let mut has_loha = false;
            let mut physical = 0_u64;
            for tensor in headers {
                physical = physical.checked_add(tensor.data_bytes).ok_or_else(|| {
                    gen_core::Error::Msg(format!("{PROVIDER_ID}: adapter byte overflow"))
                })?;
                let name = tensor.name.to_ascii_lowercase();
                has_diff |= name.ends_with(".diff") || name.ends_with(".diff_b");
                has_lokr |= name.contains("lokr");
                has_loha |= name.contains("hada_") || name.contains("loha");
                has_lora |= name.contains("lora_") || name.ends_with(".a") || name.ends_with(".b");
            }
            if physical == 0 {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: adapter {ordinal} contains no realized tensor bytes"
                )));
            }
            let factor_kinds = u8::from(has_lora) + u8::from(has_lokr) + u8::from(has_loha);
            let realization = if has_diff && factor_kinds > 0 {
                AdapterRealization::Hybrid
            } else if has_diff {
                AdapterRealization::FoldedFullRank
            } else if has_lokr {
                AdapterRealization::Lokr
            } else if has_loha {
                AdapterRealization::Loha
            } else {
                AdapterRealization::Lora
            };
            Ok::<_, gen_core::Error>((format!("{:x}", hasher.finalize()), realization, physical))
        })?;
        Ok(Self {
            ordinal,
            pin,
            canonical_path,
            digest,
            kind: adapter.kind,
            scale_bits: adapter.scale.to_bits(),
            pass_scales: adapter
                .pass_scales
                .as_ref()
                .map(|values| values.iter().map(|value| value.to_bits()).collect()),
            expert: adapter.moe_expert,
            realization,
            physical_bytes,
            // Every Candle adapter is folded into the dense CPU map and released after build.
            transient_bytes: physical_bytes,
        })
    }

    fn ensure_unchanged(
        &self,
        ordinal: usize,
        adapter: &gen_core::AdapterSpec,
    ) -> gen_core::Result<()> {
        let path = std::fs::canonicalize(&adapter.path)?;
        let pass_scales = adapter.pass_scales.as_ref().map(|values| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        });
        if self.ordinal != ordinal
            || path != self.canonical_path
            || self.kind != adapter.kind
            || self.scale_bits != adapter.scale.to_bits()
            || self.pass_scales != pass_scales
            || self.expert != adapter.moe_expert
            || self.digest.len() != 64
            || self.physical_bytes == 0
            || self.transient_bytes != self.physical_bytes
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter {ordinal} crossed its immutable loader receipt"
            )));
        }
        let _ = self.realization;
        self.pin.ensure_unchanged()
    }
}

#[derive(Clone, Debug)]
struct ArtifactReceipt {
    root: PathBuf,
    pins: Vec<gen_core::PinnedWeightsFile>,
    canonical: bool,
    facts: MemoryAssetFacts,
}

impl ArtifactReceipt {
    fn capture(spec: &LoadSpec) -> gen_core::Result<Self> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: requires directory weights"
            )));
        };
        let lexical_root = std::path::absolute(root)?;
        let root = std::fs::canonicalize(root)?;
        let config = crate::config::Scail2Config::from_model_dir(&root)
            .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
        if config.packed_quant.is_some() || spec.quantize.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: Candle public animation accepts dense bf16 only"
            )));
        }
        let files = crate::pipeline::SHARED_TIER_FILES
            .iter()
            .map(|name| root.join(name))
            .collect::<Vec<_>>();
        for file in &files {
            if !file.is_file() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: canonical tier is missing {}",
                    file.display()
                )));
            }
        }
        let pins = files
            .iter()
            .map(|file| {
                let lexical = lexical_root.join(file.file_name().expect("direct file"));
                if spec.prepared_file_pins().is_prepared() {
                    spec.prepared_file_pins()
                        .get(&lexical)
                        .cloned()
                        .ok_or_else(|| {
                            gen_core::Error::Unsupported(format!(
                                "{PROVIDER_ID}: sealed receipt is missing {}",
                                lexical.display()
                            ))
                        })
                } else {
                    gen_core::PinnedWeightsFile::pin(file)
                }
            })
            .collect::<gen_core::Result<Vec<_>>>()?;
        let component_bytes = |name: &str| -> gen_core::Result<u64> {
            let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(root.join(name))?;
            if headers.is_empty() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: {name} contains no tensors"
                )));
            }
            headers.into_iter().try_fold(0_u64, |total, tensor| {
                total
                    .checked_add(tensor.materialized_bytes(4)?)
                    .ok_or_else(|| {
                        gen_core::Error::Msg(format!("{PROVIDER_ID}: component byte overflow"))
                    })
            })
        };
        let conditioning_bytes = component_bytes("t5_encoder.safetensors")?
            .checked_add(component_bytes("clip.safetensors")?)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{PROVIDER_ID}: conditioning byte overflow"))
            })?;
        let transformer_bytes = component_bytes("dit.safetensors")?;
        let decoder_bytes = component_bytes("vae.safetensors")?;
        let base_bytes = conditioning_bytes
            .checked_add(transformer_bytes)
            .and_then(|value| value.checked_add(decoder_bytes))
            .ok_or_else(|| gen_core::Error::Msg(format!("{PROVIDER_ID}: base byte overflow")))?;
        let receipt = Self {
            root: root.clone(),
            pins,
            canonical: canonical_artifact_path(&root)
                && spec.resolved_route.as_deref() == Some(PROVIDER_ID),
            facts: MemoryAssetFacts {
                base_bytes,
                conditioning_bytes,
                transformer_bytes,
                decoder_bytes,
                // Dense-folded Candle adapters have zero steady overlay residency.
                overlay_bytes: 0,
            },
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    fn ensure_unchanged(&self) -> gen_core::Result<()> {
        for pin in &self.pins {
            pin.ensure_unchanged()?;
        }
        for name in crate::pipeline::SHARED_TIER_FILES {
            if !self.root.join(name).is_file() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: direct artifact inventory changed after admission"
                )));
            }
        }
        Ok(())
    }
}

fn canonical_artifact_path(root: &Path) -> bool {
    root.file_name().and_then(|name| name.to_str()) == Some("bf16")
        && root.components().any(|part| {
            matches!(
                part.as_os_str().to_str(),
                Some("models--SceneWorks--scail2-mlx") | Some("SceneWorks__scail2-mlx")
            )
        })
        && root
            .components()
            .any(|part| part.as_os_str().to_str() == Some(CANONICAL_REVISION))
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedMemory {
    pub(crate) contract: MemoryProviderContract,
    pub(crate) structural_evidence: MemoryStructuralResidentEvidence,
    tier: MemoryNumericTier,
    artifact: ArtifactReceipt,
    adapters: Vec<AdapterReceipt>,
    active_context: Arc<Mutex<Option<MemoryRunContext>>>,
}

impl PreparedMemory {
    pub(crate) fn prepare(spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_spec(spec)?;
        spec.validate_prepared_file_pins()?;
        let adapters = spec
            .adapters
            .iter()
            .enumerate()
            .map(|(ordinal, adapter)| AdapterReceipt::capture(spec, ordinal, adapter))
            .collect::<gen_core::Result<Vec<_>>>()?;
        let artifact = ArtifactReceipt::capture(spec)?;
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        };
        if !artifact.canonical {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: weights must be the canonical {CANONICAL_REPOSITORY}@{CANONICAL_REVISION}/bf16 artifact"
            )));
        }
        let contract = build_contract(spec, artifact.facts);
        let structural_evidence = build_structural_evidence(&artifact, &adapters, tier);
        structural_evidence.validate()?;
        Ok(Self {
            contract,
            structural_evidence,
            tier,
            artifact,
            adapters,
            active_context: Arc::new(Mutex::new(None)),
        })
    }

    pub(crate) fn ensure_unchanged(
        &self,
        adapters: &[gen_core::AdapterSpec],
    ) -> gen_core::Result<()> {
        self.artifact.ensure_unchanged()?;
        if adapters.len() != self.adapters.len() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter stack changed after admission"
            )));
        }
        for (ordinal, (receipt, adapter)) in self.adapters.iter().zip(adapters).enumerate() {
            receipt.ensure_unchanged(ordinal, adapter)?;
        }
        Ok(())
    }
}

/// Shape-only check of an admitted `evidence_revision`. Admission sees no [`GenerationRequest`], so
/// this is all that can be verified there — it deliberately does **not** return the embedded source
/// digest for reuse, because feeding that digest back into [`request_evidence_revision`] validated
/// the engine's own claim against itself and bound no carrier bytes at all.
fn validate_context_revision_shape(
    evidence: &MemoryStructuralResidentEvidence,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let parts = context.evidence_revision.split(':').collect::<Vec<_>>();
    if parts.len() != 4
        || parts[0] != "scail2-resident-v1"
        || parts[1] != evidence.receipt_sha256
        || parts[2].len() != 64
        || !parts[2].bytes().all(|byte| byte.is_ascii_hexdigit())
        || parts[3].len() != 64
        || !parts[3].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: request evidence identity does not match the sealed artifact"
        )));
    }
    Ok(())
}

/// Provider-owned digest of the *actual* animation carrier bytes: the character image, its paired
/// color mask, and every driving frame/mask pair, each bound by geometry **and** pixel content.
/// SVD hashes `reference.pixels` and Wan hashes every clip frame; SCAIL-2 previously bound only the
/// caller-declared `WxH`, so two requests that differed only in pixel content shared one identity.
pub fn carrier_source_digest(request: &GenerationRequest) -> gen_core::Result<String> {
    let carrier = request.scail2_animation_conditioning()?;
    let mut hasher = Sha256::new();
    hasher.update(CARRIER_DIGEST_DOMAIN.as_bytes());
    hasher.update(request.video_mode.as_deref().unwrap_or_default().as_bytes());
    for (role, image) in [
        ("character", carrier.character),
        ("character-mask", carrier.character_mask),
    ] {
        hasher.update(role.as_bytes());
        hasher.update(image.width.to_le_bytes());
        hasher.update(image.height.to_le_bytes());
        hasher.update(Sha256::digest(&image.pixels));
    }
    if carrier.driving_frames.len() != carrier.driving_masks.len() {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: driving frames and masks are not a one-to-one sequence"
        )));
    }
    hasher.update((carrier.driving_frames.len() as u64).to_le_bytes());
    for (index, (frame, mask)) in carrier
        .driving_frames
        .iter()
        .zip(carrier.driving_masks)
        .enumerate()
    {
        hasher.update((index as u64).to_le_bytes());
        hasher.update(frame.width.to_le_bytes());
        hasher.update(frame.height.to_le_bytes());
        hasher.update(Sha256::digest(&frame.pixels));
        hasher.update(mask.width.to_le_bytes());
        hasher.update(mask.height.to_le_bytes());
        hasher.update(Sha256::digest(&mask.pixels));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// The one place the executing identity is derived, for both this provider and any admitting
/// caller (the SVD/Wan `request_evidence_revision` shape). The carrier digest comes from the
/// request's own bytes, never from the revision being checked.
pub fn request_evidence_revision(
    evidence: &MemoryStructuralResidentEvidence,
    request: &GenerationRequest,
    selection: gen_core::MemorySelection,
) -> gen_core::Result<String> {
    validate_generation_request(request)?;
    let mode = request.video_mode.as_deref().unwrap_or_default();
    let carrier = request.scail2_animation_conditioning()?;
    let identity = gen_core::MemoryStructuralResidentRequestIdentity {
        source_digest: carrier_source_digest(request)?,
        mode: mode.to_owned(),
        carrier_shape: carrier.identity_shape(mode)?,
        width: request.width,
        height: request.height,
        frames: request.frames.unwrap_or_default(),
        fps: request.fps.unwrap_or_default(),
        reference_count: request.memory_reference_count(),
        seed: request.seed,
        sampler: request.sampler.clone(),
        scheduler: request.scheduler.clone(),
        steps: request.steps,
        guidance: request.guidance,
        scheduler_shift: request.scheduler_shift,
        selection,
    };
    evidence.evidence_revision(&identity)
}

fn validate_context_identity(
    memory: &PreparedMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    memory.structural_evidence.validate()?;
    validate_context_revision_shape(&memory.structural_evidence, context)
}

fn validate_request_identity(
    memory: &PreparedMemory,
    context: &MemoryRunContext,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    validate_context_revision_shape(&memory.structural_evidence, context)?;
    let actual =
        request_evidence_revision(&memory.structural_evidence, request, context.selection)?;
    if actual != context.evidence_revision {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: actual generation request crossed its admitted identity"
        )));
    }
    Ok(())
}

/// Refuse a memory-managed generate whose request is not the one the admitted scope opened.
/// `active_context` was armed and cleared but never read, so a crossed request could execute under
/// another request's admission (SVD's `ACTIVE_EVIDENCE` and Wan's `ActiveEvidenceGuard` both
/// enforce this).
pub(crate) fn validate_active_request(
    memory: &PreparedMemory,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    if request.memory.is_none() {
        return Ok(());
    }
    let active = candle_gen::lock_recover(&memory.active_context);
    let Some(context) = active.as_ref() else {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: memory-managed generation requires an admitted memory scope"
        )));
    };
    validate_request_identity(memory, context, request)
}

struct Scail2MemoryScope<'a> {
    inner: Box<dyn MemoryRequestScope + 'a>,
    memory: &'a PreparedMemory,
    context: MemoryRunContext,
    finished: bool,
}

impl MemoryRequestScope for Scail2MemoryScope<'_> {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.inner.configure_request(request)?;
        validate_request_identity(self.memory, &self.context, request)
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
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.inner.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, size: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(size)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.inner.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        let result = self.inner.finish(outcome);
        *self
            .memory
            .active_context
            .lock()
            .expect("SCAIL active context") = None;
        self.finished = true;
        result
    }
}

impl Drop for Scail2MemoryScope<'_> {
    fn drop(&mut self) {
        if !self.finished {
            *self
                .memory
                .active_context
                .lock()
                .expect("SCAIL active context") = None;
        }
    }
}

fn build_structural_evidence(
    artifact: &ArtifactReceipt,
    adapters: &[AdapterReceipt],
    tier: MemoryNumericTier,
) -> MemoryStructuralResidentEvidence {
    let mut hasher = Sha256::new();
    hasher.update(CANONICAL_REPOSITORY);
    hasher.update(CANONICAL_REVISION);
    hasher.update("bf16");
    for pin in &artifact.pins {
        hasher.update(format!("{pin:?}"));
    }
    for adapter in adapters {
        hasher.update(adapter.ordinal.to_le_bytes());
        hasher.update(adapter.canonical_path.to_string_lossy().as_bytes());
        hasher.update(adapter.digest.as_bytes());
        hasher.update(format!("{:?}", adapter.kind));
        hasher.update(adapter.scale_bits.to_le_bytes());
        hasher.update(adapter.physical_bytes.to_le_bytes());
        hasher.update(adapter.transient_bytes.to_le_bytes());
    }
    MemoryStructuralResidentEvidence {
        abi: MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
        provider_id: PROVIDER_ID.to_owned(),
        repository: CANONICAL_REPOSITORY.to_owned(),
        revision: CANONICAL_REVISION.to_owned(),
        variant: "bf16".to_owned(),
        receipt_sha256: format!("{:x}", hasher.finalize()),
        tier,
        load_shape: LoadShape::EagerMaterialization,
        asset_facts: artifact.facts,
        request_transient_bytes: adapters.iter().fold(0_u64, |sum, adapter| {
            sum.saturating_add(adapter.transient_bytes)
        }),
        direct_file_count: artifact.pins.len() as u32,
        adapter_count: adapters.len() as u32,
    }
}

fn validate_load_spec(spec: &LoadSpec) -> gen_core::Result<()> {
    if spec.precision != Precision::Bf16
        || spec.quantize.is_some()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: load is outside the dense-bf16 public animation surface"
        )));
    }
    gen_core::reject_unknown_components(spec, &[], PROVIDER_ID)
}

/// Snapshot-read architecture facts for the SCAIL-2 route (epic SC-22657, E2).
///
/// The DiT axes come from the very call the loader makes:
/// [`crate::config::Scail2Config::from_model_dir`] on the snapshot root, which reads the flat
/// `<root>/config.json` (`num_heads`, `num_layers`, `dim`, …) and keeps the shipped
/// [`crate::config::Scail2Config::scail2_14b`] value for any key the file omits. Both snapshot
/// layouts are handled the same way because the loader handles them the same way: `pipeline::load`
/// reads the config from the root for `SharedMlxTier` and `LegacyComponents` alike, and only the
/// *weight* files move under `transformer/`. So a snapshot whose config disagrees with the shipped
/// 14B config publishes what it actually says.
///
/// `head_dim` is [`crate::config::Scail2Config::head_dim`] — `dim / num_heads`, the same quotient
/// the model builds its attention from — and is declined when the config makes it inexact.
///
/// The VAE axes come from SCAIL-2's concrete VAE assignment: it decodes through
/// `candle_gen_wan::vae16::WanVae16` at `Vae16Config::wan21()`, whose geometry this crate re-exports
/// as [`crate::VAE_TILING`] (x8 spatial, x4 temporal).
///
/// **`activation_dtype_width` is 4, not 2.** This provider computes in f32 on purpose:
/// `pipeline::transformer_vb` loads the DiT at `DType::F32` because "bf16 overflows to NaN at high
/// token length" (`lib.rs`). Declaring the family-typical bf16 width here would halve every
/// activation estimate.
///
/// A weights-free contract — no materialized snapshot directory — publishes
/// `MemoryArchitectureFacts::default()`.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let Some(root) = af::snapshot_root(spec) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let Ok(config) = crate::config::Scail2Config::from_model_dir(root) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let tiling = crate::VAE_TILING;
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::declared(config.num_heads),
        // `head_dim()` is `dim / num_heads`; publish it only when the division is exact, so a config
        // with a non-uniform pair cannot claim a head width it does not have.
        head_dim: (config.num_heads != 0 && config.dim % config.num_heads == 0)
            .then(|| af::declared(config.head_dim()))
            .flatten(),
        transformer_blocks: af::declared(config.num_layers),
        // `patch` is `(p_t, p_h, p_w) = (1, 2, 2)` — a Rust-only field the config reader never
        // overrides; the spatial (index 1) entry is the axis this fact names.
        patch_size: af::declared(config.patch.1),
        latent_channels: af::declared(config.vae_z_dim),
        vae_spatial_scale: u32::try_from(tiling.spatial_scale)
            .ok()
            .filter(|scale| *scale != 0),
        vae_temporal_scale: u32::try_from(tiling.temporal_scale)
            .ok()
            .filter(|scale| *scale != 0),
        // f32, not bf16 — see the note above.
        activation_dtype_width: af::dtype_width(candle_gen::candle_core::DType::F32),
    }
}

fn build_contract(spec: &LoadSpec, facts: MemoryAssetFacts) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: PROVIDER_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if strategy == MemoryStrategy::Resident {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                parameters: MemoryParameterRanges::default(),
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: false,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::OverlayBytes,
            ],
        },
        calibration: None,
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

pub fn provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    PreparedMemory::prepare(spec).map(|prepared| prepared.contract)
}

/// Weights-free catalog witness for the sole shipped dense-bf16 Resident surface. Production loads
/// still require the immutable canonical receipt in [`PreparedMemory::prepare`].
fn weights_free_resident_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(spec)?;
    Ok(build_contract(spec, MemoryAssetFacts::default()))
}

fn resident_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.tier == gen_core::MemoryContractSurfaceTier::Bf16)
        .collect()
}

fn validate_route(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let geometry = context.geometry;
    if !matches!(context.mode.as_key(), "animation" | "replacement")
        || geometry.batch != 1
        || geometry.reference_count != 1
        || !context.has_reference
        || !PUBLIC_BUCKETS.contains(&(geometry.width, geometry.height))
        || !PUBLIC_FRAMES.contains(&geometry.frames)
        || context.use_pid
        || context.has_phases
        || context
            .overlay
            .as_deref()
            .is_some_and(|overlay| overlay != "adapter")
        || context.selection.strategy != MemoryStrategy::Resident
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: requires the exact one-character animation/replacement Resident route"
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(contract, context, Some(tier), None) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub(crate) fn safety_check(
    memory: &PreparedMemory,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match memory
        .artifact
        .ensure_unchanged()
        .and_then(|_| validate_context_identity(memory, context))
        .and_then(|_| validate_route(&memory.contract, memory.tier, context))
    {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

/// A declared geometry backed by exactly `width * height * 3` RGB8 bytes.
fn is_exact_rgb8(image: &gen_core::Image) -> bool {
    u64::from(image.width)
        .checked_mul(u64::from(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .is_some_and(|bytes| bytes != 0 && bytes == image.pixels.len())
}

pub(crate) fn validate_generation_request(request: &GenerationRequest) -> gen_core::Result<()> {
    let carrier = request.scail2_animation_conditioning()?;
    let frames = request.frames.unwrap_or_default();
    if !matches!(
        request.video_mode.as_deref(),
        Some("animation" | "replacement")
    ) || !PUBLIC_BUCKETS.contains(&(request.width, request.height))
        || !PUBLIC_FRAMES.contains(&frames)
        || request.fps != Some(16)
        || u32::try_from(carrier.driving_frames.len()).unwrap_or(u32::MAX) != frames
        || carrier.driving_masks.len() != carrier.driving_frames.len()
        || carrier
            .driving_frames
            .iter()
            .chain(carrier.driving_masks)
            .any(|frame| (frame.width, frame.height) != (request.width, request.height))
        // A declared `WxH` with an empty or short buffer is not a carrier. SVD and Wan both verify
        // `w*h*3 == pixels.len()`; without it a carrier of empty buffers validated.
        || [carrier.character, carrier.character_mask]
            .into_iter()
            .chain(carrier.driving_frames)
            .chain(carrier.driving_masks)
            .any(|image| !is_exact_rgb8(image))
        || request.count != 1
        || request.use_pid
        || request.phases.is_some()
        || request
            .memory
            .is_some_and(|memory| memory != gen_core::GenerationMemory::default())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: request left the admitted animation/replacement envelope"
        )));
    }
    Ok(())
}

pub(crate) fn begin_request<'a>(
    memory: &'a PreparedMemory,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    memory.artifact.ensure_unchanged()?;
    validate_context_identity(memory, context)?;
    validate_route(&memory.contract, memory.tier, context)?;
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        PROVIDER_ID,
        device,
        context.geometry,
        memory.contract.generation_memory(&context.selection),
        false,
        0,
        |_pid, _edge, _overlap| {
            Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: bounded decode is Missing"
            )))
        },
    )?;
    config.default_frames = context.geometry.frames;
    *candle_gen::lock_recover(&memory.active_context) = Some(context.clone());
    Ok(Some(Box::new(Scail2MemoryScope {
        inner: Box::new(candle_gen::request_scope::CandleRequestScopeCore::new(
            config,
        )),
        memory,
        context: context.clone(),
        finished: false,
    })))
}

pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: PROVIDER_ID,
    contract: provider_contract,
    safety_check: |spec, contract, context| match PreparedMemory::prepare(spec) {
        Ok(prepared) if contract != &prepared.contract => MemorySafetyDecision::Reject {
            reason: format!(
                "{PROVIDER_ID}: caller contract does not match the sealed load receipt"
            ),
        },
        Ok(prepared) => match validate_context_identity(&prepared, context)
            .and_then(|_| validate_route(&prepared.contract, prepared.tier, context))
        {
            Ok(()) => MemorySafetyDecision::Accept,
            Err(error) => MemorySafetyDecision::Reject {
                reason: error.to_string(),
            },
        },
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    },
};

pub const RESIDENT_ONLY_WITNESS: ResidentOnlyMemoryContractRegistration =
    ResidentOnlyMemoryContractRegistration {
        provider_id: PROVIDER_ID,
        contract: weights_free_resident_contract,
        surface_specs: resident_surface_specs,
    };

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, GenerationRequest, Image, ReplacementMode};

    fn image(width: u32, height: u32) -> Image {
        tinted_image(width, height, 7)
    }

    fn tinted_image(width: u32, height: u32, tint: u8) -> Image {
        Image {
            width,
            height,
            pixels: vec![tint; width as usize * height as usize * 3],
        }
    }

    fn request(mode: &str, width: u32, height: u32, frames: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "dance".to_owned(),
            width,
            height,
            count: 1,
            frames: Some(frames),
            fps: Some(16),
            video_mode: Some(mode.to_owned()),
            conditioning: vec![
                Conditioning::Reference {
                    image: image(2, 2),
                    strength: None,
                },
                Conditioning::Mask { image: image(2, 2) },
                Conditioning::ControlClip {
                    frames: (0..frames).map(|_| image(width, height)).collect(),
                    mask: (0..frames).map(|_| image(width, height)).collect(),
                    masking_strength: 1.0,
                    start_frame: 0,
                    mode: ReplacementMode::default(),
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn public_cells_and_resident_only_contract_are_exact() {
        for &(width, height) in PUBLIC_BUCKETS {
            for &frames in PUBLIC_FRAMES {
                for mode in ["animation", "replacement"] {
                    validate_generation_request(&request(mode, width, height, frames)).unwrap();
                }
            }
        }
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/unread")));
        let contract = build_contract(&spec, MemoryAssetFacts::default());
        for strategy in MemoryStrategy::ALL {
            let expected = if strategy == MemoryStrategy::Resident {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            };
            assert_eq!(contract.capability(strategy).unwrap().support, expected);
        }
    }

    /// AC (epic SC-22657, E2): the contract publishes the architecture axes read from the snapshot's
    /// own `config.json` — the very call `pipeline::load` makes — and the weights-free surface
    /// publishes none of them.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // The upstream `configs/config-14b.json` layout, cut down to the keys
        // `Scail2Config::from_model_dir` reads.
        std::fs::write(
            root.join("config.json"),
            br#"{
                "dim": 5120,
                "ffn_dim": 13824,
                "num_heads": 40,
                "num_layers": 40,
                "in_dim": 20,
                "out_dim": 16,
                "mask_dim": 28
            }"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
        let contract = build_contract(&spec, MemoryAssetFacts::default());
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // `config.json: num_heads`.
                attention_heads: Some(40),
                // `Scail2Config::head_dim()` = `dim / num_heads` = 5120 / 40.
                head_dim: Some(128),
                // `config.json: num_layers`.
                transformer_blocks: Some(40),
                // `Scail2Config::patch = (1, 2, 2)` — a Rust-only field; index 1 is the spatial
                // entry.
                patch_size: Some(2),
                // `Scail2Config::vae_z_dim`.
                latent_channels: Some(16),
                // `crate::VAE_TILING` (`WanVae16::VAE_TILING`, the Wan 2.1 z16 autoencoder).
                vae_spatial_scale: Some(8),
                vae_temporal_scale: Some(4),
                // f32, not bf16: `pipeline::transformer_vb` loads the DiT at `DType::F32` because
                // bf16 overflows to NaN at high token length.
                activation_dtype_width: Some(4),
            }
        );
        assert!(contract.architecture_facts.has_snapshot_read_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The facts are read, not asserted: a config declaring a different depth and width publishes
        // what it actually says.
        std::fs::write(
            root.join("config.json"),
            br#"{"dim": 2048, "num_heads": 16, "num_layers": 24}"#,
        )
        .unwrap();
        let mutated = architecture_facts(&spec);
        assert_eq!(mutated.attention_heads, Some(16));
        assert_eq!(mutated.head_dim, Some(128));
        assert_eq!(mutated.transformer_blocks, Some(24));

        // The registry surface names a sentinel that is not on disk.
        let weights_free = weights_free_resident_contract(&LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        )))
        .unwrap();
        assert!(weights_free.architecture_facts.is_empty());
    }

    #[test]
    fn crossed_animation_axes_refuse_before_generation() {
        let mut wrong = request("replacement", 832, 480, 45);
        wrong.fps = Some(24);
        assert!(validate_generation_request(&wrong).is_err());
        let mut wrong = request("animation", 832, 480, 45);
        wrong.video_mode = Some("other".to_owned());
        assert!(validate_generation_request(&wrong).is_err());
        let mut wrong = request("replacement", 832, 480, 45);
        wrong
            .conditioning
            .push(Conditioning::Mask { image: image(2, 2) });
        assert!(validate_generation_request(&wrong).is_err());
        let mut wrong = request("replacement", 832, 480, 45);
        let Conditioning::ControlClip { frames, .. } = &mut wrong.conditioning[2] else {
            unreachable!()
        };
        frames[0] = image(480, 832);
        assert!(validate_generation_request(&wrong).is_err());
        assert!(validate_generation_request(&request("replacement", 1024, 576, 45)).is_err());
    }

    pub(crate) fn write_tensor(path: &Path, name: &str, bytes: usize) {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        let data = vec![9_u8; bytes];
        let view = TensorView::new(Dtype::U8, vec![bytes], &data).unwrap();
        std::fs::write(path, serialize([(name, view)], None).unwrap()).unwrap();
    }

    fn write_hybrid(path: &Path) {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        let diff = vec![1_u8; 7];
        let lora = vec![2_u8; 9];
        let diff = TensorView::new(Dtype::U8, vec![7], &diff).unwrap();
        let lora = TensorView::new(Dtype::U8, vec![9], &lora).unwrap();
        std::fs::write(
            path,
            serialize([("block.diff", diff), ("block.lora_up", lora)], None).unwrap(),
        )
        .unwrap();
    }

    pub(crate) fn candle_fixture(adapter_name: Option<&str>) -> (tempfile::TempDir, LoadSpec) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .join("models--SceneWorks--scail2-mlx")
            .join("snapshots")
            .join(CANONICAL_REVISION)
            .join("bf16");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), "{}").unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        for (name, size) in [
            ("t5_encoder.safetensors", 3),
            ("clip.safetensors", 5),
            ("dit.safetensors", 7),
            ("vae.safetensors", 11),
        ] {
            write_tensor(&root.join(name), "weight", size);
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.resolved_route = Some(PROVIDER_ID.to_owned());
        if let Some(name) = adapter_name {
            let adapter_path = root.join(format!("adapter-{name}.safetensors"));
            let tensor_name = match name {
                "lora" => "block.lora_up",
                "lokr" => "block.lokr_w1",
                "loha" => "block.hada_w1_a",
                "full" => "block.diff",
                "hybrid" => "block.diff_b",
                _ => unreachable!(),
            };
            if name == "hybrid" {
                write_hybrid(&adapter_path);
            } else {
                write_tensor(&adapter_path, tensor_name, 17);
            }
            spec.adapters.push(gen_core::AdapterSpec::new(
                adapter_path,
                0.5,
                if name == "lokr" || name == "loha" {
                    AdapterKind::Lokr
                } else {
                    AdapterKind::Lora
                },
            ));
        }
        let mut pins = crate::pipeline::SHARED_TIER_FILES
            .iter()
            .map(|file| gen_core::PinnedWeightsFile::pin(root.join(file)).unwrap())
            .collect::<Vec<_>>();
        pins.extend(
            spec.adapters
                .iter()
                .map(|adapter| gen_core::PinnedWeightsFile::pin(&adapter.path).unwrap()),
        );
        spec.prepare_with_file_pins(pins).unwrap();
        (temp, spec)
    }

    #[test]
    fn prepared_memory_fixture_binds_dense_bf16_and_all_adapter_kinds() {
        for adapter in [
            None,
            Some("lora"),
            Some("lokr"),
            Some("loha"),
            Some("full"),
            Some("hybrid"),
        ] {
            let (_temp, spec) = candle_fixture(adapter);
            let prepared = PreparedMemory::prepare(&spec).unwrap();
            assert_eq!(prepared.contract.calibration, None);
            assert!(prepared.contract.asset_facts.base_bytes > 0);
            assert_eq!(prepared.contract.asset_facts.overlay_bytes, 0);
            assert_eq!(
                prepared.structural_evidence.adapter_count,
                spec.adapters.len() as u32
            );
            assert_eq!(
                prepared.structural_evidence.request_transient_bytes > 0,
                adapter.is_some()
            );
        }
    }

    fn admitted_context(
        prepared: &PreparedMemory,
        request: &GenerationRequest,
        cache_state: gen_core::MemoryCacheState,
    ) -> MemoryRunContext {
        let selection = gen_core::MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: gen_core::MemoryStrategyParameters::default(),
            tier: prepared.tier,
        };
        let mode = request.video_mode.as_deref().unwrap();
        MemoryRunContext {
            selection,
            optimization_authority: gen_core::MemoryOptimizationAuthority::Resident,
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: prepared.contract.load_shape,
            mode: gen_core::MemoryMode::Other(mode.to_owned()),
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: gen_core::MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: 1,
                frames: request.frames.unwrap(),
                reference_count: 1,
            },
            overlay: Some("adapter".to_owned()),
            budget: gen_core::MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: prepared.contract.total_resident_bytes(),
            cache_state,
            evidence_revision: request_evidence_revision(
                &prepared.structural_evidence,
                request,
                selection,
            )
            .unwrap(),
        }
    }

    #[test]
    fn candle_safety_begin_and_registration_bind_cold_warm_identity() {
        let (_temp, spec) = candle_fixture(Some("hybrid"));
        let prepared = PreparedMemory::prepare(&spec).unwrap();
        for mode in ["animation", "replacement"] {
            let mut req = request(mode, 832, 480, 45);
            req.seed = Some(44);
            for cache_state in [
                gen_core::MemoryCacheState::Cold,
                gen_core::MemoryCacheState::Warm,
            ] {
                let context = admitted_context(&prepared, &req, cache_state);
                assert_eq!(
                    safety_check(&prepared, &context),
                    MemorySafetyDecision::Accept
                );
                let mut scope = begin_request(&prepared, Device::Cpu, &context)
                    .unwrap()
                    .unwrap();
                let mut exact = req.clone();
                scope.configure_request(&mut exact).unwrap();
                scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
            }
        }
        let req = request("replacement", 832, 480, 45);
        let context = admitted_context(&prepared, &req, gen_core::MemoryCacheState::Cold);
        let mut wrong_contract = prepared.contract.clone();
        wrong_contract.asset_facts.base_bytes += 1;
        assert!(matches!(
            (MEMORY_REGISTRATION.safety_check)(&spec, &wrong_contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// The admitted identity must bind the carrier's *bytes*, not just its declared `WxH`. The
    /// provider used to lift the source digest straight out of the revision it was checking, so a
    /// request with identical geometry but different pixel content validated against another
    /// request's admission.
    #[test]
    fn crossed_carrier_pixels_refuse_the_admitted_identity() {
        let (_temp, spec) = candle_fixture(None);
        let prepared = PreparedMemory::prepare(&spec).unwrap();
        let admitted = request("animation", 832, 480, 45);
        let context = admitted_context(&prepared, &admitted, gen_core::MemoryCacheState::Cold);

        // Same mode, geometry, frame count and declared sizes — only the pixels differ.
        let mut crossed = admitted.clone();
        let Conditioning::ControlClip { frames, .. } = &mut crossed.conditioning[2] else {
            unreachable!()
        };
        frames[0] = tinted_image(832, 480, 9);
        assert_eq!(
            crossed
                .scail2_animation_conditioning()
                .unwrap()
                .identity_shape("animation")
                .unwrap(),
            admitted
                .scail2_animation_conditioning()
                .unwrap()
                .identity_shape("animation")
                .unwrap(),
            "the crossed carrier must be shape-identical, or this proves nothing"
        );
        assert_ne!(
            carrier_source_digest(&crossed).unwrap(),
            carrier_source_digest(&admitted).unwrap()
        );

        let mut scope = begin_request(&prepared, Device::Cpu, &context)
            .unwrap()
            .unwrap();
        let mut crossed_scope_request = crossed.clone();
        assert!(scope.configure_request(&mut crossed_scope_request).is_err());
        scope.configure_request(&mut admitted.clone()).unwrap();

        // ... and the executing generate must consult the admitted scope, not just the scope's own
        // `configure_request`.
        let mut executing = crossed;
        executing.memory = Some(gen_core::GenerationMemory::default());
        assert!(validate_active_request(&prepared, &executing).is_err());
        let mut exact = admitted;
        exact.memory = Some(gen_core::GenerationMemory::default());
        validate_active_request(&prepared, &exact).unwrap();
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();

        // The scope is closed: the same request no longer executes under a stale admission.
        assert!(validate_active_request(&prepared, &exact).is_err());
        // A request that asks for no memory at all is not gated by the scope.
        let plain = request("animation", 832, 480, 45);
        assert!(plain.memory.is_none());
        validate_active_request(&prepared, &plain).unwrap();
    }

    /// A declared `WxH` with an empty or short buffer is not a carrier.
    #[test]
    fn carrier_images_must_carry_exact_rgb8_bytes() {
        validate_generation_request(&request("animation", 832, 480, 45)).unwrap();
        for index in [0_usize, 1] {
            let mut empty = request("animation", 832, 480, 45);
            match &mut empty.conditioning[index] {
                Conditioning::Reference { image, .. } | Conditioning::Mask { image } => {
                    image.pixels.clear();
                }
                _ => unreachable!(),
            }
            assert!(validate_generation_request(&empty).is_err(), "{index}");
        }
        let mut short_frame = request("animation", 832, 480, 45);
        let Conditioning::ControlClip { frames, mask, .. } = &mut short_frame.conditioning[2]
        else {
            unreachable!()
        };
        frames[3].pixels.truncate(832 * 480 * 3 - 1);
        let _ = mask;
        assert!(validate_generation_request(&short_frame).is_err());
        let mut empty_mask = request("animation", 832, 480, 45);
        let Conditioning::ControlClip { mask, .. } = &mut empty_mask.conditioning[2] else {
            unreachable!()
        };
        mask[2].pixels.clear();
        assert!(validate_generation_request(&empty_mask).is_err());
    }

    #[test]
    fn candle_receipt_refuses_base_and_adapter_mutation() {
        let (_temp, spec) = candle_fixture(Some("loha"));
        let prepared = PreparedMemory::prepare(&spec).unwrap();
        std::fs::write(&spec.adapters[0].path, b"mutated").unwrap();
        assert!(prepared.ensure_unchanged(&spec.adapters).is_err());

        let (_temp, base_spec) = candle_fixture(None);
        let base_prepared = PreparedMemory::prepare(&base_spec).unwrap();
        let root = match &base_spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        std::fs::write(root.join("dit.safetensors"), b"mutated").unwrap();
        assert!(base_prepared.ensure_unchanged(&base_spec.adapters).is_err());
    }
}
