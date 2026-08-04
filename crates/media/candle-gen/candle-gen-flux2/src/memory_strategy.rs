//! Candle/CUDA FLUX.2 adoption of the shared image-memory ladder (SC-15833, SC-15831).
//!
//! Dev and Klein deliberately share lifecycle and execution primitives while retaining distinct
//! provider identities, block domains, candidate ranges, and calibration fingerprints. The three
//! SceneWorks Klein catalog entries resolve to the one `flux2_klein_9b` Candle provider; entry-level
//! tier/mode/overlay measurements remain catalog-owned and cannot be inferred from this contract.

use crate::config::{Flux2Variant, FLUX2_DEV_ID, FLUX2_KLEIN_9B_ID};
use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryWindowMaterialization,
    PerComponentBytes, Precision, Quant, TransformerComponent, WeightsSource,
};
use std::sync::{Arc, Mutex};

/// Full output edge used by the bounded-decode hook at the representative 1024px calibration cell.
/// This intentionally does not spatially partition that cell: the saving comes from separating the
/// full-image attention-bearing head from the upsampling tail's live envelope, while preserving a
/// near-monolithic numerical path.
pub const DECODE_TILE_EDGE: u32 = 1024;
pub const DECODE_TILE_EDGES: &[u32] = &[DECODE_TILE_EDGE];
/// The shared contract requires a positive overlap domain. At the 1024px full-edge calibration cell
/// no neighboring tiles exist, so this value is an inert, exactly keyed sentinel rather than a claim
/// that spatial blending occurred.
pub const DECODE_OVERLAP: u32 = 1;
pub const DECODE_OVERLAPS: &[u32] = &[DECODE_OVERLAP];
pub const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
pub const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub const BASE_DOUBLE_BLOCKS: u32 = 8;
pub const BASE_SINGLE_BLOCKS: u32 = 48;
pub const BASE_TRANSFORMER_BLOCKS: u32 = BASE_DOUBLE_BLOCKS + BASE_SINGLE_BLOCKS;
pub const CONTROL_BLOCKS: u32 = 4;
pub const CALIBRATION_FINGERPRINT: &str =
    "flux2-dev-cuda-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v2";
pub const CONTROL_OVERLAY: &str = "control";

pub const KLEIN_DECODE_TILE_EDGE: u32 = 512;
pub const KLEIN_DECODE_TILE_EDGES: &[u32] = &[768, 640, KLEIN_DECODE_TILE_EDGE];
pub const KLEIN_DECODE_OVERLAP: u32 = 128;
pub const KLEIN_DECODE_OVERLAPS: &[u32] = &[KLEIN_DECODE_OVERLAP];
pub const KLEIN_BASE_DOUBLE_BLOCKS: u32 = 8;
pub const KLEIN_BASE_SINGLE_BLOCKS: u32 = 24;
pub const KLEIN_BASE_TRANSFORMER_BLOCKS: u32 = KLEIN_BASE_DOUBLE_BLOCKS + KLEIN_BASE_SINGLE_BLOCKS;
pub const KLEIN_CALIBRATION_FINGERPRINT: &str = "flux2-klein-cuda-shared-ladder-provider-abi-v1";

#[derive(Clone, Copy)]
struct ProviderProfile {
    provider_id: &'static str,
    decode_tile_edges: &'static [u32],
    decode_overlaps: &'static [u32],
    base_transformer_blocks: u32,
    calibration_fingerprint: &'static str,
}

fn profile(provider_id: &str) -> gen_core::Result<ProviderProfile> {
    match provider_id {
        FLUX2_DEV_ID => Ok(ProviderProfile {
            provider_id: FLUX2_DEV_ID,
            decode_tile_edges: DECODE_TILE_EDGES,
            decode_overlaps: DECODE_OVERLAPS,
            base_transformer_blocks: BASE_TRANSFORMER_BLOCKS,
            calibration_fingerprint: CALIBRATION_FINGERPRINT,
        }),
        FLUX2_KLEIN_9B_ID => Ok(ProviderProfile {
            provider_id: FLUX2_KLEIN_9B_ID,
            decode_tile_edges: KLEIN_DECODE_TILE_EDGES,
            decode_overlaps: KLEIN_DECODE_OVERLAPS,
            base_transformer_blocks: KLEIN_BASE_TRANSFORMER_BLOCKS,
            calibration_fingerprint: KLEIN_CALIBRATION_FINGERPRINT,
        }),
        _ => Err(gen_core::Error::Unsupported(format!(
            "unknown FLUX.2 memory provider {provider_id}"
        ))),
    }
}

fn path(source: &WeightsSource) -> &std::path::Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.adapters.is_empty()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none()
}

fn resident_components(provider_id: &str, spec: &LoadSpec) -> Vec<MemoryResidentComponent> {
    let mut out = Vec::new();
    if provider_id == FLUX2_DEV_ID {
        if let Some(control) = spec.control.as_ref() {
            let resident_bytes = gen_core::weightsmeta::safetensors_path_bytes(path(control));
            if resident_bytes > 0 {
                out.push(MemoryResidentComponent {
                    id: "flux2_dev_fun_controlnet_union".to_owned(),
                    kind: MemoryComponentKind::ControlBranch,
                    resident_bytes,
                    // SC-15833 windows the 56-block base. The four overlay blocks remain resident and
                    // are therefore charged explicitly rather than hidden inside the base estimate.
                    bounded_by: None,
                });
            }
        }
    }
    out
}

pub fn provider_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let profile = profile(provider_id)?;
    let streamable = streamable(spec);
    let components =
        PerComponentBytes::from_spec_subdirs(spec, &["text_encoder"], &["transformer"], &["vae"])
            .unwrap_or_default();
    let resident_components = resident_components(provider_id, spec);
    let overlay_bytes = resident_components
        .iter()
        .map(|component| component.resident_bytes)
        .sum();
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if strategy == MemoryStrategy::BoundedTransformerResidency && !streamable {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: profile.decode_tile_edges.to_vec(),
                    decode_overlaps: profile.decode_overlaps.to_vec(),
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                        transformer_window_components: vec![TransformerComponent::Dit],
                        ..Default::default()
                    }
                }
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect();

    Ok(MemoryProviderContract {
        provider_id: profile.provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        // This provider's constrained implementations load request-scoped phases. The explicit
        // edge is realization-owned, not an assumption made by the shared ladder.
        additional_prerequisites: [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            // The hook is implemented, but the sole 1024px production candidate is full-edge and
            // therefore does not spatially partition the representative calibration cell.
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components,
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            profile.calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

pub fn provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_for(FLUX2_DEV_ID, spec)
}

pub fn klein_provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_for(FLUX2_KLEIN_9B_ID, spec)
}

pub fn contract_for_variant(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_for(variant.id(), spec)
}

fn packed_quant(spec: &LoadSpec) -> gen_core::Result<Option<Quant>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "flux2_dev: numeric tier requires a snapshot directory".to_owned(),
        ));
    };
    let config = root.join("transformer/config.json");
    let packed = match std::fs::read_to_string(&config) {
        Ok(text) => {
            let value = serde_json::from_str::<serde_json::Value>(&text).map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "flux2_dev: parse {}: {error}",
                    config.display()
                ))
            })?;
            candle_gen::quant::PackedConfig::from_config(&value)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(gen_core::Error::Unsupported(format!(
                "flux2_dev: read {}: {error}",
                config.display()
            )))
        }
    };
    packed
        .map(|packed| match packed.bits {
            4 => Ok(Quant::Q4),
            8 => Ok(Quant::Q8),
            bits => Err(gen_core::Error::Unsupported(format!(
                "flux2_dev: transformer declares unsupported packed quantization width {bits}"
            ))),
        })
        .transpose()
}

pub fn resolved_quant(spec: &LoadSpec) -> gen_core::Result<Option<Quant>> {
    let packed = packed_quant(spec)?;
    match (spec.quantize, packed) {
        (Some(requested), Some(stored)) if requested != stored => {
            Err(gen_core::Error::Unsupported(format!(
                "flux2_dev: requested {requested:?} but snapshot stores {stored:?}"
            )))
        }
        (Some(requested), _) => Ok(Some(requested)),
        (None, stored) => Ok(stored),
    }
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: resolved_quant(spec)?,
        component_precision_floors: &[],
    })
}

fn route_is_supported(provider_id: &str, context: &MemoryRunContext) -> bool {
    match (
        &context.mode,
        context.geometry.reference_count,
        context.overlay.as_deref(),
    ) {
        (MemoryMode::TextToImage, 0, None) => true,
        (MemoryMode::Edit, 1..=8, None) => true,
        (MemoryMode::Other(mode), 1..=8, None)
            if mode == "character_image" || mode == "style_variations" =>
        {
            true
        }
        (MemoryMode::TextToImage, 0, Some(CONTROL_OVERLAY)) if provider_id == FLUX2_DEV_ID => true,
        _ => false,
    }
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } = gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: loaded_quant,
            component_precision_floors: &[],
        }),
        None,
    ) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    if !route_is_supported(&contract.provider_id, context) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: unsupported memory route mode={} references={} overlay={:?}",
            contract.provider_id,
            context.mode.as_key(),
            context.geometry.reference_count,
            context.overlay
        )));
    }
    if context.geometry.batch != 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: memory calibration is single-image only",
            contract.provider_id
        )));
    }
    if context.use_pid && context.selection.strategy.is_optimized() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: PiD cannot consume the native FLUX.2 VAE memory selection",
            contract.provider_id
        )));
    }
    if context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: optimized memory strategies do not cover multi-phase denoise",
            contract.provider_id
        )));
    }
    Ok(())
}

pub fn admission_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> MemorySafetyDecision {
    match validate_context(contract, context, loaded_quant) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn validate_registered_generator_context(
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    if context.mode != MemoryMode::TextToImage
        || context.geometry.reference_count != 0
        || context.has_reference
        || context.overlay.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "flux2_dev: registered generator admits text-to-image without references or overlays only"
                .to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Flux2RequestBinding {
    request_address: usize,
    geometry: MemoryGeometry,
    use_pid: bool,
    has_phases: bool,
    memory: Option<GenerationMemory>,
}

impl Flux2RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
        Self {
            request_address: std::ptr::from_ref(request).addr(),
            geometry: MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: request.frames.unwrap_or(1),
                reference_count: request.image_reference_count(),
            },
            use_pid: request.use_pid,
            has_phases: request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty()),
            memory: request.memory,
        }
    }
}

struct Flux2ActiveAdmission {
    token: u64,
    context: MemoryRunContext,
    expected_memory: Option<GenerationMemory>,
    binding: Option<Flux2RequestBinding>,
    consumed: bool,
}

#[derive(Default)]
struct Flux2AdmissionState {
    next_token: u64,
    approved_context: Option<MemoryRunContext>,
    active: Option<Flux2ActiveAdmission>,
}

/// Provider-local, one-shot authorization joining `begin`/`configure` to the exact request object
/// later passed to `generate`. The opaque token never enters `GenerationRequest`, so cloning or
/// copying its memory knobs cannot transfer authorization to another request.
#[derive(Clone)]
pub(crate) struct Flux2AdmissionRegistry {
    provider_id: &'static str,
    inner: Arc<Mutex<Flux2AdmissionState>>,
}

impl Flux2AdmissionRegistry {
    pub(crate) fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            inner: Arc::new(Mutex::new(Flux2AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: cannot replace safety approval while a request scope is active",
                self.provider_id
            )));
        }
        state.approved_context = Some(context.clone());
        Ok(())
    }

    pub(crate) fn clear_approval(&self) {
        candle_gen::lock_recover(&self.inner).approved_context = None;
    }

    fn begin(
        &self,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        expected_memory: Option<GenerationMemory>,
    ) -> gen_core::Result<u64> {
        if contract.provider_id != self.provider_id {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory contract belongs to {}",
                self.provider_id, contract.provider_id
            )));
        }
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request scope is already active",
                self.provider_id
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request begin skipped the safety handshake",
                self.provider_id
            ))
        })?;
        if approved != *context {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory context changed after safety approval",
                self.provider_id
            )));
        }
        state.next_token = state.next_token.wrapping_add(1).max(1);
        let token = state.next_token;
        state.active = Some(Flux2ActiveAdmission {
            token,
            context: context.clone(),
            expected_memory,
            binding: None,
            consumed: false,
        });
        Ok(token)
    }

    fn configure(&self, token: u64, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        let active = state.active.as_mut().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request scope is no longer active",
                self.provider_id
            ))
        })?;
        if active.token != token || active.binding.is_some() || active.consumed {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale, reused, or already-configured memory token",
                self.provider_id
            )));
        }
        let binding = Flux2RequestBinding::from_request(request);
        if binding.geometry != active.context.geometry
            || binding.use_pid != active.context.use_pid
            || binding.has_phases != active.context.has_phases
            || binding.memory != active.expected_memory
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed while configuring memory admission",
                self.provider_id
            )));
        }
        active.binding = Some(binding);
        Ok(())
    }

    pub(crate) fn consume_for_generate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        let constrained = request
            .memory
            .is_some_and(|memory| memory != GenerationMemory::default());
        let Some(active) = state.active.as_mut() else {
            return if constrained {
                Err(gen_core::Error::Unsupported(format!(
                    "{}: constrained memory request has no active admission token",
                    self.provider_id
                )))
            } else {
                Ok(())
            };
        };
        let binding = active.binding.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: active memory request was not configured",
                self.provider_id
            ))
        })?;
        if binding != &Flux2RequestBinding::from_request(request) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request or memory strategy changed after admission",
                self.provider_id
            )));
        }
        if active.consumed {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: memory admission token was already consumed",
                self.provider_id
            )));
        }
        active.consumed = true;
        Ok(())
    }

    fn finish(&self, token: u64) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: stale memory token cannot finish",
                self.provider_id
            )))
        }
    }

    fn abandon(&self, token: u64) {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
        }
    }
}

pub struct Flux2MemoryScope {
    device: Device,
    provider_id: String,
    decode_tile_edges: Vec<u32>,
    decode_overlaps: Vec<u32>,
    attention_chunk_sizes: Vec<u32>,
    base_transformer_blocks: u32,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    transformer_window: Option<u32>,
    use_pid: bool,
    has_phases: bool,
    admission: Option<Flux2AdmissionRegistry>,
    token: Option<u64>,
    finished: bool,
}

impl Flux2MemoryScope {
    pub fn new(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> Self {
        let decode = contract
            .capability(MemoryStrategy::BoundedDecode)
            .expect("FLUX.2 contract publishes bounded decode");
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .expect("FLUX.2 contract publishes bounded attention");
        Self {
            device,
            provider_id: contract.provider_id.clone(),
            decode_tile_edges: decode.parameters.decode_tile_edges.clone(),
            decode_overlaps: decode.parameters.decode_overlaps.clone(),
            attention_chunk_sizes: attention.parameters.attention_chunk_sizes.clone(),
            base_transformer_blocks: profile(&contract.provider_id)
                .expect("validated FLUX.2 provider contract")
                .base_transformer_blocks,
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            transformer_window: contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then_some(context.selection.parameters.transformer_window_size)
                .flatten(),
            use_pid: context.use_pid,
            has_phases: context.has_phases,
            admission: None,
            token: None,
            finished: false,
        }
    }

    pub(crate) fn new_bound(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: Flux2AdmissionRegistry,
    ) -> gen_core::Result<Self> {
        let token = admission.begin(
            contract,
            context,
            contract.generation_memory(&context.selection),
        )?;
        let mut scope = Self::new(device, contract, context);
        scope.admission = Some(admission);
        scope.token = Some(token);
        Ok(scope)
    }

    fn active(&self) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg(format!(
                "{}: memory request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }
}

impl MemoryRequestScope for Flux2MemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.active()?;
        if request.use_pid != self.use_pid
            || request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count != self.geometry.batch
            || request.image_reference_count() != self.geometry.reference_count
            || request.frames.unwrap_or(1) != self.geometry.frames
            || request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty())
                != self.has_phases
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request route or geometry changed after admission",
                self.provider_id
            )));
        }
        request.memory = self.memory;
        if let (Some(admission), Some(token)) = (&self.admission, self.token) {
            admission.configure(token, request)?;
        }
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        self.active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        self.active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.active()?;
        if geometry != self.geometry {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: decode geometry changed after admission",
                self.provider_id
            )));
        }
        if self.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: PiD has no admitted FLUX.2 VAE decode plan",
                self.provider_id
            )));
        }
        if self.decode_tile_edges.contains(&tile_edge) && self.decode_overlaps.contains(&overlap) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: decode does not publish {tile_edge}/{overlap}",
                self.provider_id
            )))
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.active()?;
        if self.attention_chunk_sizes.contains(&chunk_size) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: attention chunk size is not in {:?}, got {chunk_size}",
                self.provider_id, self.attention_chunk_sizes
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.active()?;
        let Some(window) = self.transformer_window else {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded transformer residency was not selected",
                self.provider_id
            )));
        };
        if window == 0 || block_count == 0 || !first_block.is_multiple_of(window) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: invalid transformer window {block_count} at {first_block}",
                self.provider_id
            )));
        }
        if first_block >= self.base_transformer_blocks {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: transformer window starts past the {}-block base",
                self.provider_id, self.base_transformer_blocks
            )));
        }
        let expected = window.min(self.base_transformer_blocks - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got {block_count}",
                self.provider_id
            )))
        }
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.active()?;
        self.device
            .synchronize()
            .map_err(gen_core::Error::backend)?;
        if let (Some(admission), Some(token)) = (&self.admission, self.token) {
            admission.finish(token)?;
        }
        self.finished = true;
        Ok(())
    }
}

impl Drop for Flux2MemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            if let (Some(admission), Some(token)) = (&self.admission, self.token) {
                admission.abandon(token);
            }
            self.finished = true;
        }
    }
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match resolved_quant(spec) {
        Ok(quant) => admission_safety_check(contract, context, quant),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let tier = resolved_numeric_tier(spec)?;
    let routes = [
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Other("character_image".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Other("style_variations".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    ];
    routes
        .into_iter()
        .map(|route| {
            gen_core::standard_memory_behavior_context(contract, strategy, tier, route)
                .map(gen_core::MemoryBehaviorFixture::new)
        })
        .collect()
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_context(contract, context, resolved_quant(spec)?)?;
    Ok(Some(Box::new(Flux2MemoryScope::new(
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spec() -> LoadSpec {
        let mut spec =
            LoadSpec::new(WeightsSource::Dir(PathBuf::from("/flux2-dev"))).with_quant(Quant::Q4);
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    fn capability(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> &MemoryStrategyCapability {
        contract
            .strategies
            .iter()
            .find(|capability| capability.strategy == strategy)
            .expect("strategy capability")
    }

    #[test]
    fn dev_contract_is_distinct_and_publishes_all_candidate_ranges() {
        let contract = provider_contract(&spec()).unwrap();
        assert!(contract.conformance_errors().is_empty());
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert_eq!(contract.provider_id, FLUX2_DEV_ID);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            CALIBRATION_FINGERPRINT
        );
        assert_ne!(
            CALIBRATION_FINGERPRINT,
            crate::RESIDENCY_CALIBRATION_FINGERPRINT
        );
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                capability(&contract, strategy).support,
                MemoryStrategySupport::Implemented
            );
        }
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_overlaps,
            DECODE_OVERLAPS
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedAttention)
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedTransformerResidency)
                .parameters
                .transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
    }

    #[test]
    fn klein_contract_is_distinct_and_publishes_production_candidate_ranges() {
        let contract = klein_provider_contract(&spec()).unwrap();
        assert!(contract.conformance_errors().is_empty());
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert_eq!(contract.provider_id, FLUX2_KLEIN_9B_ID);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            KLEIN_CALIBRATION_FINGERPRINT
        );
        assert_ne!(KLEIN_CALIBRATION_FINGERPRINT, CALIBRATION_FINGERPRINT);
        assert_eq!(KLEIN_BASE_TRANSFORMER_BLOCKS, 32);
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_tile_edges,
            KLEIN_DECODE_TILE_EDGES
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedDecode)
                .parameters
                .decode_overlaps,
            KLEIN_DECODE_OVERLAPS
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedAttention)
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedTransformerResidency)
                .parameters
                .transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
    }

    #[test]
    fn klein_scope_rejects_dev_candidate_and_block_domain() {
        let contract = klein_provider_contract(&spec()).unwrap();
        let context = registered_valid_fixture(
            &spec(),
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0)
        .context;
        let mut scope = Flux2MemoryScope::new(Device::Cpu, &contract, &context);
        assert!(scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, context.geometry)
            .is_err());
        assert!(scope
            .configure_decode(
                KLEIN_DECODE_TILE_EDGE,
                KLEIN_DECODE_OVERLAP,
                context.geometry,
            )
            .is_ok());
        assert!(scope
            .materialize_transformer_window(KLEIN_BASE_TRANSFORMER_BLOCKS, 1)
            .is_err());
        assert!(scope
            .materialize_transformer_window(KLEIN_BASE_TRANSFORMER_BLOCKS - 1, 1)
            .is_ok());
    }

    #[test]
    fn klein_route_rejects_dev_only_control_overlay() {
        let dev_contract = provider_contract(&spec()).unwrap();
        let mut context = registered_valid_fixture(
            &spec(),
            &dev_contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0)
        .context;
        context.overlay = Some(CONTROL_OVERLAY.to_owned());
        assert!(route_is_supported(FLUX2_DEV_ID, &context));
        assert!(!route_is_supported(FLUX2_KLEIN_9B_ID, &context));
    }

    #[test]
    fn klein_contract_never_inherits_dev_control_residency_identity() {
        let mut spec = spec();
        spec.control = Some(WeightsSource::File("control.safetensors".into()));
        let contract = klein_provider_contract(&spec).unwrap();
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = contract.formula
        else {
            panic!("FLUX.2 contract must use the component-phase formula")
        };
        assert!(resident_components.is_empty());
    }

    #[test]
    fn resident_load_shape_fails_closed_for_rung_four() {
        let mut spec = spec();
        spec.load_shape = LoadShape::EagerMaterialization;
        let contract = provider_contract(&spec).unwrap();
        assert_eq!(
            capability(&contract, MemoryStrategy::BoundedTransformerResidency).support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn block_domain_excludes_resident_control_overlay() {
        assert_eq!(BASE_TRANSFORMER_BLOCKS, 56);
        assert_eq!(CONTROL_BLOCKS, 4);
        assert_ne!(
            BASE_TRANSFORMER_BLOCKS,
            BASE_TRANSFORMER_BLOCKS + CONTROL_BLOCKS
        );
    }

    #[test]
    fn stale_identity_pid_and_route_mutation_fail_closed() {
        let spec = spec();
        let contract = provider_contract(&spec).unwrap();
        let mut fixture = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        fixture.context.calibration_fingerprint = "stale-flux2".to_owned();
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
        fixture.context.calibration_fingerprint = CALIBRATION_FINGERPRINT.to_owned();
        fixture.context.use_pid = true;
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
        fixture.context.use_pid = false;
        fixture.context.geometry.reference_count = 1;
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn active_admission_is_one_shot_request_local_and_non_transferable() {
        let spec = spec();
        let contract = provider_contract(&spec).unwrap();
        let context = registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .remove(0)
            .context;
        let registry = Flux2AdmissionRegistry::new(FLUX2_DEV_ID);

        let wrong_provider = Flux2AdmissionRegistry::new("not_flux2_dev");
        wrong_provider.approve(&context).unwrap();
        assert!(
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, wrong_provider,).is_err()
        );

        let manual_memory = contract.generation_memory(&context.selection).unwrap();
        let manual = GenerationRequest {
            prompt: "manual".to_owned(),
            memory: Some(manual_memory),
            ..Default::default()
        };
        assert!(registry.consume_for_generate(&manual).is_err());

        assert!(
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, registry.clone(),)
                .is_err(),
            "begin without safety approval must fail"
        );

        registry.approve(&context).unwrap();
        let mut unconfigured =
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, registry.clone())
                .unwrap();
        assert!(registry
            .consume_for_generate(&GenerationRequest {
                prompt: "unconfigured".to_owned(),
                ..Default::default()
            })
            .is_err());
        unconfigured.finish(MemoryRunOutcome::Canceled).unwrap();

        registry.approve(&context).unwrap();
        let mut scope =
            Flux2MemoryScope::new_bound(Device::Cpu, &contract, &context, registry.clone())
                .unwrap();
        let mut request = GenerationRequest {
            prompt: "bound".to_owned(),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let copied = request.clone();
        assert!(registry.consume_for_generate(&copied).is_err());

        request.width /= 2;
        assert!(registry.consume_for_generate(&request).is_err());
        request.width *= 2;
        let expected_memory = request.memory;
        request.memory = Some(GenerationMemory::default());
        assert!(registry.consume_for_generate(&request).is_err());
        request.memory = expected_memory;

        registry.consume_for_generate(&request).unwrap();
        assert!(registry.consume_for_generate(&request).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
    }
}
