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
    ResidentRequestMemory, WeightsSource, MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
};
use sha2::{Digest, Sha256};

pub const PROVIDER_ID: &str = crate::pipeline::MODEL_ID;
pub const CANONICAL_REPOSITORY: &str = "SceneWorks/scail2-mlx";
pub const CANONICAL_REVISION: &str = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";
pub const PUBLIC_BUCKETS: &[(u32, u32)] = &[(832, 480), (480, 832), (1280, 704), (704, 1280)];
pub const PUBLIC_FRAMES: &[u32] = &[45, 61, 77];

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

fn build_contract(spec: &LoadSpec, facts: MemoryAssetFacts) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
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
    *memory.active_context.lock().expect("SCAIL active context") = Some(context.clone());
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, GenerationRequest, Image, ReplacementMode};

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

    fn candle_fixture(adapter_name: Option<&str>) -> (tempfile::TempDir, LoadSpec) {
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
            overlay: Some("adapter".to_owned()),
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
    fn candle_safety_begin_and_registration_bind_cold_warm_identity() {
        let (_temp, spec) = candle_fixture(Some("hybrid"));
        let prepared = PreparedMemory::prepare(&spec).unwrap();
        let mut req = request(832, 480, 45);
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
        let context = admitted_context(&prepared, &req, gen_core::MemoryCacheState::Cold);
        let mut wrong_contract = prepared.contract.clone();
        wrong_contract.asset_facts.base_bytes += 1;
        assert!(matches!(
            (MEMORY_REGISTRATION.safety_check)(&spec, &wrong_contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
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
