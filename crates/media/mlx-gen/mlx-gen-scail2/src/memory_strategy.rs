//! Exact Resident-only memory contract for the public SCAIL-2 character-animation route.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_gen::gen_core::{
    self, AdapterKind, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryStructuralResidentEvidence, Precision, Quant, ResidentOnlyMemoryContractRegistration,
    ResidentRequestMemory, WeightsSource, MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
};
use sha2::{Digest, Sha256};

pub const PROVIDER_ID: &str = crate::pipeline::MODEL_ID;
pub const CANONICAL_REPOSITORY: &str = "SceneWorks/scail2-mlx";
pub const CANONICAL_REVISION: &str = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";
pub const PUBLIC_BUCKETS: &[(u32, u32)] = &[(832, 480), (480, 832), (1280, 704), (704, 1280)];
pub const PUBLIC_FRAMES: &[u32] = &[45, 61, 77];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdapterRealization {
    Residual,
    FoldedFullRank,
    Hybrid,
}

#[derive(Clone, Debug)]
pub(crate) struct AdapterReceipt {
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
    resident_bytes: u64,
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
        // SCAIL-2 is a single-pass, single-expert DiT. Recording a target is insufficient when the
        // loader cannot execute it; refuse instead of silently widening it to the whole model.
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
        let (digest, realization, physical_bytes, resident_bytes, transient_bytes) = pin
            .read_unchanged(|path| {
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
                let mut folded = 0_u64;
                let mut residual = 0_u64;
                for tensor in headers {
                    if tensor.name.ends_with(".diff") || tensor.name.ends_with(".diff_b") {
                        folded = folded.checked_add(tensor.data_bytes).ok_or_else(|| {
                            gen_core::Error::Msg(format!("{PROVIDER_ID}: adapter byte overflow"))
                        })?;
                    } else {
                        residual = residual.checked_add(tensor.data_bytes).ok_or_else(|| {
                            gen_core::Error::Msg(format!("{PROVIDER_ID}: adapter byte overflow"))
                        })?;
                    }
                }
                let physical = folded.checked_add(residual).ok_or_else(|| {
                    gen_core::Error::Msg(format!("{PROVIDER_ID}: adapter byte overflow"))
                })?;
                if physical == 0 {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{PROVIDER_ID}: adapter {ordinal} contains no realized tensor bytes"
                    )));
                }
                let realization = match (folded > 0, residual > 0) {
                    (false, true) => AdapterRealization::Residual,
                    (true, false) => AdapterRealization::FoldedFullRank,
                    (true, true) => AdapterRealization::Hybrid,
                    (false, false) => unreachable!("nonempty headers carry bytes"),
                };
                Ok::<_, gen_core::Error>((
                    format!("{:x}", hasher.finalize()),
                    realization,
                    physical,
                    residual,
                    folded,
                ))
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
            resident_bytes,
            transient_bytes,
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
            || self.resident_bytes.saturating_add(self.transient_bytes) != self.physical_bytes
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
    tier: QuantTier,
    canonical: bool,
    facts: MemoryAssetFacts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuantTier {
    Bf16,
    Q8,
    Q4,
}

impl QuantTier {
    fn quant(self) -> Option<Quant> {
        match self {
            Self::Bf16 => None,
            Self::Q8 => Some(Quant::Q8),
            Self::Q4 => Some(Quant::Q4),
        }
    }

    fn directory(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }
}

impl ArtifactReceipt {
    fn capture(spec: &LoadSpec, adapters: &[AdapterReceipt]) -> gen_core::Result<Self> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: requires directory weights"
            )));
        };
        let lexical_root = std::path::absolute(root)?;
        let root = std::fs::canonicalize(root)?;
        let config = crate::config::Scail2Config::from_model_dir(&root)
            .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
        let tier = match config.wan.quantization.as_ref().map(|quant| quant.bits) {
            None => QuantTier::Bf16,
            Some(8) => QuantTier::Q8,
            Some(4) => QuantTier::Q4,
            other => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: unsupported on-disk quantization {other:?}"
                )))
            }
        };
        // Canonical tiers are already packed (or dense bf16). `quantize` would be an independent
        // on-load transform and must remain unset; the selected/loaded tier is bound by the exact
        // directory plus config/tensor receipt instead.
        if spec.quantize.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: canonical tier loads must not request a second on-load quantization"
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
                total.checked_add(tensor.data_bytes).ok_or_else(|| {
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
        let overlay_bytes = adapters.iter().try_fold(0_u64, |total, receipt| {
            total.checked_add(receipt.resident_bytes).ok_or_else(|| {
                gen_core::Error::Msg(format!("{PROVIDER_ID}: overlay byte overflow"))
            })
        })?;
        let base_bytes = conditioning_bytes
            .checked_add(transformer_bytes)
            .and_then(|value| value.checked_add(decoder_bytes))
            .ok_or_else(|| gen_core::Error::Msg(format!("{PROVIDER_ID}: base byte overflow")))?;
        let canonical = canonical_artifact_path(&root, tier)
            && spec.resolved_route.as_deref() == Some(PROVIDER_ID);
        let receipt = Self {
            root,
            pins,
            tier,
            canonical,
            facts: MemoryAssetFacts {
                base_bytes,
                conditioning_bytes,
                transformer_bytes,
                decoder_bytes,
                overlay_bytes,
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

fn canonical_artifact_path(root: &Path, tier: QuantTier) -> bool {
    root.file_name().and_then(|name| name.to_str()) == Some(tier.directory())
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
        let artifact = ArtifactReceipt::capture(spec, &adapters)?;
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: artifact.tier.quant(),
            component_precision_floors: &[],
        };
        if !artifact.canonical {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: weights must be the canonical {CANONICAL_REPOSITORY}@{CANONICAL_REVISION}/{} artifact",
                artifact.tier.directory()
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

fn structural_source_digest<'a>(
    evidence: &MemoryStructuralResidentEvidence,
    context: &'a MemoryRunContext,
) -> gen_core::Result<&'a str> {
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
    Ok(parts[2])
}

fn validate_context_identity(
    memory: &PreparedMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    memory.structural_evidence.validate()?;
    let _ = structural_source_digest(&memory.structural_evidence, context)?;
    Ok(())
}

fn validate_request_identity(
    memory: &PreparedMemory,
    context: &MemoryRunContext,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    let source_digest = structural_source_digest(&memory.structural_evidence, context)?.to_owned();
    let identity = gen_core::MemoryStructuralResidentRequestIdentity {
        source_digest,
        mode: request.video_mode.clone().unwrap_or_default(),
        carrier_shape: "character_animation".to_owned(),
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
        selection: context.selection,
    };
    if memory.structural_evidence.evidence_revision(&identity)? != context.evidence_revision {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: actual generation request crossed its admitted identity"
        )));
    }
    Ok(())
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
    hasher.update(artifact.tier.directory());
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
        hasher.update(adapter.resident_bytes.to_le_bytes());
        hasher.update(adapter.transient_bytes.to_le_bytes());
    }
    MemoryStructuralResidentEvidence {
        abi: MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
        provider_id: PROVIDER_ID.to_owned(),
        repository: CANONICAL_REPOSITORY.to_owned(),
        revision: CANONICAL_REVISION.to_owned(),
        variant: artifact.tier.directory().to_owned(),
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
            "{PROVIDER_ID}: load is outside the canonical bf16/q8/q4 animation surface"
        )));
    }
    gen_core::reject_unknown_components(spec, &[], PROVIDER_ID)
}

fn build_contract(spec: &LoadSpec, facts: MemoryAssetFacts) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        provider_id: PROVIDER_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
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

/// Weights-free witness for SCAIL-2's deliberately Resident-only animation route.
///
/// Runtime admission still seals real tier and adapter bytes through [`PreparedMemory`]; this
/// registry witness exists only to prove the static rung disposition for the supported load shape.
fn weights_free_resident_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(spec)?;
    Ok(build_contract(spec, MemoryAssetFacts::default()))
}

fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.spec.quantize.is_none())
        .collect()
}

fn validate_route(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let geometry = context.geometry;
    if context.mode.as_key() != "animation"
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
            "{PROVIDER_ID}: requires the exact one-character animation Resident route"
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

pub(crate) fn validate_generation_request(request: &GenerationRequest) -> gen_core::Result<()> {
    let carrier = request.scail2_animation_conditioning()?;
    let frames = request.frames.unwrap_or_default();
    if request.video_mode.as_deref() != Some("animation")
        || request.width == 0
        || request.height == 0
        || !PUBLIC_BUCKETS.contains(&(request.width, request.height))
        || !PUBLIC_FRAMES.contains(&frames)
        || request.fps != Some(16)
        || u32::try_from(carrier.driving_frames.len()).unwrap_or(u32::MAX) != frames
        || request.count != 1
        || request.use_pid
        || request.phases.is_some()
        || request
            .memory
            .is_some_and(|memory| memory != gen_core::GenerationMemory::default())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: request left the admitted animation envelope"
        )));
    }
    Ok(())
}

pub(crate) fn begin_request<'a>(
    memory: &'a PreparedMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    memory.artifact.ensure_unchanged()?;
    validate_context_identity(memory, context)?;
    validate_route(&memory.contract, memory.tier, context)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        PROVIDER_ID,
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
    config.load_shape = memory.contract.load_shape;
    *memory.active_context.lock().expect("SCAIL active context") = Some(context.clone());
    Ok(Some(Box::new(Scail2MemoryScope {
        inner: Box::new(mlx_gen::request_scope::MlxRequestScopeCore::new(config)),
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
        surface_specs: memory_contract_surface_specs,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{Conditioning, GenerationRequest, Image, ReplacementMode};

    fn image() -> Image {
        Image {
            width: 2,
            height: 2,
            pixels: vec![0; 12],
        }
    }

    fn request(width: u32, height: u32, frames: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "dance".to_owned(),
            width,
            height,
            count: 1,
            frames: Some(frames),
            fps: Some(16),
            video_mode: Some("animation".to_owned()),
            conditioning: vec![
                Conditioning::Reference {
                    image: image(),
                    strength: None,
                },
                Conditioning::Mask { image: image() },
                Conditioning::ControlClip {
                    frames: (0..frames).map(|_| image()).collect(),
                    mask: (0..frames).map(|_| image()).collect(),
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
                validate_generation_request(&request(width, height, frames)).unwrap();
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

    #[test]
    fn crossed_animation_axes_refuse_before_generation() {
        let mut wrong = request(832, 480, 45);
        wrong.fps = Some(24);
        assert!(validate_generation_request(&wrong).is_err());
        let mut wrong = request(832, 480, 45);
        wrong.video_mode = Some("replacement".to_owned());
        assert!(validate_generation_request(&wrong).is_err());
        let mut wrong = request(832, 480, 45);
        wrong
            .conditioning
            .push(Conditioning::Mask { image: image() });
        assert!(validate_generation_request(&wrong).is_err());
        assert!(validate_generation_request(&request(1024, 576, 45)).is_err());
    }

    fn write_tensor(path: &Path, name: &str, bytes: usize) {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        let data = vec![7_u8; bytes];
        let view = TensorView::new(Dtype::U8, vec![bytes], &data).unwrap();
        std::fs::write(path, serialize([(name, view)], None).unwrap()).unwrap();
    }

    fn write_hybrid(path: &Path) {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        let folded = vec![3_u8; 11];
        let residual = vec![5_u8; 13];
        let folded_view = TensorView::new(Dtype::U8, vec![folded.len()], &folded).unwrap();
        let residual_view = TensorView::new(Dtype::U8, vec![residual.len()], &residual).unwrap();
        std::fs::write(
            path,
            serialize(
                [
                    ("block.diff", folded_view),
                    ("block.lora_up", residual_view),
                ],
                None,
            )
            .unwrap(),
        )
        .unwrap();
    }

    fn fixture_spec(
        tier: &str,
        adapter_shape: Option<(&str, AdapterKind)>,
    ) -> (tempfile::TempDir, LoadSpec) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .join("models--SceneWorks--scail2-mlx")
            .join("snapshots")
            .join(CANONICAL_REVISION)
            .join(tier);
        std::fs::create_dir_all(&root).unwrap();
        let multiplier = match tier {
            "q4" => 1,
            "q8" => 2,
            "bf16" => 4,
            _ => unreachable!(),
        };
        let config = match tier {
            "q4" => r#"{"quantization":{"bits":4,"group_size":64}}"#,
            "q8" => r#"{"quantization":{"bits":8,"group_size":64}}"#,
            "bf16" => "{}",
            _ => unreachable!(),
        };
        std::fs::write(root.join("config.json"), config).unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        for (name, size) in [
            ("t5_encoder.safetensors", 3),
            ("clip.safetensors", 5),
            ("dit.safetensors", 7),
            ("vae.safetensors", 11),
        ] {
            write_tensor(&root.join(name), "weight", size * multiplier);
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.resolved_route = Some(PROVIDER_ID.to_owned());
        if let Some((shape, kind)) = adapter_shape {
            let adapter_path = root.join(format!("adapter-{shape}.safetensors"));
            match shape {
                "lora" => write_tensor(&adapter_path, "block.lora_up", 19),
                "lokr" => write_tensor(&adapter_path, "block.lokr_w1", 19),
                "loha" => write_tensor(&adapter_path, "block.hada_w1_a", 19),
                "full" => write_tensor(&adapter_path, "block.diff", 17),
                "hybrid" => write_hybrid(&adapter_path),
                _ => unreachable!(),
            }
            spec.adapters
                .push(gen_core::AdapterSpec::new(adapter_path, 0.75, kind));
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
    fn prepared_memory_fixtures_bind_all_tiers_and_adapter_realizations() {
        let variants = [
            None,
            Some(("lora", AdapterKind::Lora)),
            Some(("lokr", AdapterKind::Lokr)),
            // LoHa shares the metadata adapter-kind carrier with LoKr in the current ABI; its
            // tensor inventory remains distinct in the loader receipt.
            Some(("loha", AdapterKind::Lokr)),
            Some(("full", AdapterKind::Lora)),
            Some(("hybrid", AdapterKind::Lora)),
        ];
        let mut tier_bytes = Vec::new();
        for tier in ["q4", "q8", "bf16"] {
            for variant in variants {
                let (_temp, spec) = fixture_spec(tier, variant);
                let prepared = PreparedMemory::prepare(&spec).unwrap();
                assert_eq!(prepared.contract.calibration, None);
                assert!(prepared.contract.asset_facts.base_bytes > 0);
                assert_eq!(
                    prepared.structural_evidence.adapter_count,
                    spec.adapters.len() as u32
                );
                let folded = matches!(variant.map(|entry| entry.0), Some("full" | "hybrid"));
                assert_eq!(
                    prepared.structural_evidence.request_transient_bytes > 0,
                    folded
                );
                if variant.is_none() {
                    tier_bytes.push(prepared.contract.asset_facts.base_bytes);
                }
            }
        }
        assert!(tier_bytes.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn prepared_memory_refuses_repo_and_post_receipt_mutation() {
        let (_temp, spec) = fixture_spec("q4", Some(("hybrid", AdapterKind::Lora)));
        let prepared = PreparedMemory::prepare(&spec).unwrap();
        std::fs::write(&spec.adapters[0].path, b"mutated").unwrap();
        assert!(prepared.ensure_unchanged(&spec.adapters).is_err());

        let (temp, mut crossed) = fixture_spec("q8", None);
        let root = match &crossed.weights {
            WeightsSource::Dir(root) => root.clone(),
            _ => unreachable!(),
        };
        let wrong = temp
            .path()
            .join("models--Other--scail2-mlx")
            .join("snapshots")
            .join(CANONICAL_REVISION)
            .join("q8");
        std::fs::create_dir_all(wrong.parent().unwrap()).unwrap();
        std::fs::rename(root, &wrong).unwrap();
        crossed.weights = WeightsSource::Dir(wrong);
        assert!(PreparedMemory::prepare(&crossed).is_err());

        let (_temp, base_spec) = fixture_spec("bf16", None);
        let base_prepared = PreparedMemory::prepare(&base_spec).unwrap();
        let root = match &base_spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        std::fs::write(root.join("dit.safetensors"), b"mutated").unwrap();
        assert!(base_prepared.ensure_unchanged(&base_spec.adapters).is_err());
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
        let source_digest = format!("{:x}", Sha256::digest(b"character\0driving"));
        let identity = gen_core::MemoryStructuralResidentRequestIdentity {
            source_digest,
            mode: "animation".to_owned(),
            carrier_shape: "character_animation".to_owned(),
            width: request.width,
            height: request.height,
            frames: request.frames.unwrap(),
            fps: request.fps.unwrap(),
            reference_count: 1,
            seed: request.seed,
            sampler: request.sampler.clone(),
            scheduler: request.scheduler.clone(),
            steps: request.steps,
            guidance: request.guidance,
            scheduler_shift: request.scheduler_shift,
            selection,
        };
        MemoryRunContext {
            selection,
            optimization_authority: gen_core::MemoryOptimizationAuthority::Resident,
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: prepared.contract.load_shape,
            mode: gen_core::MemoryMode::Other("animation".to_owned()),
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
            overlay: (!prepared.adapters.is_empty()).then(|| "adapter".to_owned()),
            budget: gen_core::MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: prepared.contract.total_resident_bytes(),
            cache_state,
            evidence_revision: prepared
                .structural_evidence
                .evidence_revision(&identity)
                .unwrap(),
        }
    }

    #[test]
    fn safety_and_begin_bind_cold_warm_identity_and_reject_crossing() {
        let variants = [
            None,
            Some(("lora", AdapterKind::Lora)),
            Some(("lokr", AdapterKind::Lokr)),
            Some(("loha", AdapterKind::Lokr)),
            Some(("full", AdapterKind::Lora)),
            Some(("hybrid", AdapterKind::Lora)),
        ];
        for tier in ["q4", "q8", "bf16"] {
            for variant in variants {
                let (_temp, spec) = fixture_spec(tier, variant);
                let prepared = PreparedMemory::prepare(&spec).unwrap();
                let mut req = request(832, 480, 45);
                req.seed = Some(44);
                for cache in [
                    gen_core::MemoryCacheState::Cold,
                    gen_core::MemoryCacheState::Warm,
                ] {
                    let context = admitted_context(&prepared, &req, cache);
                    assert_eq!(
                        safety_check(&prepared, &context),
                        MemorySafetyDecision::Accept
                    );
                    let mut scope = begin_request(&prepared, &context).unwrap().unwrap();
                    let mut exact = req.clone();
                    scope.configure_request(&mut exact).unwrap();
                    scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
                    let mut crossed = context.clone();
                    crossed.evidence_revision.replace_range(0..1, "f");
                    assert!(matches!(
                        safety_check(&prepared, &crossed),
                        MemorySafetyDecision::Reject { .. }
                    ));
                }
            }
        }
        let (_temp, spec) = fixture_spec("q8", Some(("lora", AdapterKind::Lora)));
        let prepared = PreparedMemory::prepare(&spec).unwrap();
        let req = request(832, 480, 45);
        let context = admitted_context(&prepared, &req, gen_core::MemoryCacheState::Cold);
        let mut wrong_contract = prepared.contract.clone();
        wrong_contract.asset_facts.base_bytes += 1;
        assert!(matches!(
            (MEMORY_REGISTRATION.safety_check)(&spec, &wrong_contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
