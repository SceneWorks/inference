//! Shared Candle/CUDA image-memory ladder for the six Mage-Flow routes (SC-15813).
//!
//! Provider mechanics are shared, while the calibration identity remains route-local: an Edit
//! measurement must never authorize the text-to-image route (or a sibling checkpoint) merely because
//! the architecture and implementation are shared.

use crate::config;
use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyPrerequisite, MemoryStrategySupport,
    MemoryWindowMaterialization, Precision, Quant, TransformerComponent, WeightsSource,
};
use candle_gen::quant::PackedConfig;
use std::sync::{Arc, Mutex};

pub const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
pub const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub const TRANSFORMER_BLOCKS: u32 = config::DEPTH as u32;

const PROVIDER_IDS: &[&str] = &[
    config::MODEL_ID,
    config::BASE_MODEL_ID,
    config::TURBO_MODEL_ID,
    config::EDIT_MODEL_ID,
    config::EDIT_BASE_MODEL_ID,
    config::EDIT_TURBO_MODEL_ID,
];

fn is_edit(provider_id: &str) -> bool {
    matches!(
        provider_id,
        config::EDIT_MODEL_ID | config::EDIT_BASE_MODEL_ID | config::EDIT_TURBO_MODEL_ID
    )
}

fn fingerprint(provider_id: &str) -> gen_core::Result<String> {
    if PROVIDER_IDS.contains(&provider_id) {
        Ok(format!(
            "mage-flow-cuda-shared-ladder-provider-abi-v2-{}",
            provider_id.replace('_', "-")
        ))
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "unknown Mage-Flow memory provider {provider_id}"
        )))
    }
}

fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.adapters.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.pid.is_none()
        && spec.identity.is_none()
}

fn transformer_has_device_format(spec: &LoadSpec) -> gen_core::Result<bool> {
    if resolved_quant(spec)?.is_none() {
        return Ok(true);
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let path = root.join("transformer").join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(gen_core::Error::Msg(format!(
                "mage-flow: read {} while resolving streamed weight format: {error}",
                path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        gen_core::Error::Msg(format!(
            "mage-flow: parse {} while resolving streamed weight format: {error}",
            path.display()
        ))
    })?;
    Ok(PackedConfig::from_config(&config).is_some())
}

pub fn provider_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    provider_contract_with_components(provider_id, spec, crate::component_footprint(spec)?)
}

pub(crate) fn provider_contract_with_components(
    provider_id: &str,
    spec: &LoadSpec,
    components: gen_core::PerComponentBytes,
) -> gen_core::Result<MemoryProviderContract> {
    let calibration_fingerprint = fingerprint(provider_id)?;
    let streamable = streamable(spec) && transformer_has_device_format(spec)?;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                // Mage's CoD decoder normalizes over the complete latent feature field before its
                // pixel MLP. Spatial tiles would change that normalization and therefore the image;
                // a full-edge call is ordinary decode, not a bounded-memory implementation.
                MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: "Mage CoD decode contains full-frame normalization and has no parity-safe independent spatial tiles".to_owned(),
                    }
                }
                MemoryStrategy::BoundedTransformerResidency if !streamable => {
                    MemoryStrategySupport::Missing
                }
                _ => MemoryStrategySupport::Implemented,
            },
            parameters: match strategy {
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
        architecture_facts: candle_gen::gen_core::MemoryArchitectureFacts::default(),
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: [
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
            decode_tiling: false,
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
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components: Vec::new(),
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: if is_edit(provider_id) {
                components.text_encoder.saturating_add(components.vae)
            } else {
                components.text_encoder
            },
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes: 0,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

macro_rules! contract_fn {
    ($name:ident, $id:expr) => {
        pub fn $name(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
            provider_contract_for($id, spec)
        }
    };
}

contract_fn!(contract_rl, config::MODEL_ID);
contract_fn!(contract_base, config::BASE_MODEL_ID);
contract_fn!(contract_turbo, config::TURBO_MODEL_ID);
contract_fn!(contract_edit, config::EDIT_MODEL_ID);
contract_fn!(contract_edit_base, config::EDIT_BASE_MODEL_ID);
contract_fn!(contract_edit_turbo, config::EDIT_TURBO_MODEL_ID);

pub fn resolved_quant(spec: &LoadSpec) -> gen_core::Result<Option<Quant>> {
    match spec.quantize {
        Some(Quant::Q4) => Ok(Some(Quant::Q4)),
        Some(Quant::Q8) => Ok(Some(Quant::Q8)),
        None => Ok(None),
        Some(other) => Err(gen_core::Error::Unsupported(format!(
            "Mage-Flow does not support the {other:?} numeric tier"
        ))),
    }
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    let quant = resolved_quant(spec)?;
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: crate::quant::active_component_precision_floors(quant),
    })
}

fn route_is_supported(provider_id: &str, context: &MemoryRunContext) -> bool {
    if context.overlay.is_some() || context.use_pid || context.has_phases {
        return false;
    }
    if is_edit(provider_id) {
        context.mode == MemoryMode::Edit
            && (1..=8).contains(&context.geometry.reference_count)
            && context.has_reference
    } else {
        context.mode == MemoryMode::TextToImage
            && context.geometry.reference_count == 0
            && !context.has_reference
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
            component_precision_floors: crate::quant::active_component_precision_floors(
                loaded_quant,
            ),
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
    Ok(())
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match resolved_quant(spec).and_then(|quant| validate_context(contract, context, quant)) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestBinding {
    address: usize,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    has_phases: bool,
}

impl RequestBinding {
    fn from_request(request: &GenerationRequest) -> Self {
        Self {
            address: std::ptr::from_ref(request).addr(),
            geometry: MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: request.frames.unwrap_or(1),
                reference_count: request.image_reference_count(),
            },
            memory: request.memory,
            use_pid: request.use_pid,
            has_phases: request
                .phases
                .as_ref()
                .is_some_and(|phases| !phases.is_empty()),
        }
    }
}

struct ActiveAdmission {
    token: u64,
    context: MemoryRunContext,
    expected_memory: Option<GenerationMemory>,
    binding: Option<RequestBinding>,
    consumed: bool,
}

#[derive(Default)]
struct AdmissionState {
    next_token: u64,
    approved_context: Option<MemoryRunContext>,
    active: Option<ActiveAdmission>,
}

#[derive(Clone)]
pub(crate) struct AdmissionRegistry {
    provider_id: &'static str,
    inner: Arc<Mutex<AdmissionState>>,
}

impl AdmissionRegistry {
    pub(crate) fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            inner: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }

    pub(crate) fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.inner);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another memory request is active",
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
                "{}: another memory request scope is active",
                self.provider_id
            )));
        }
        let approved = state.approved_context.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: memory request skipped the safety handshake",
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
        state.active = Some(ActiveAdmission {
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
        let binding = RequestBinding::from_request(request);
        if active.token != token
            || active.binding.is_some()
            || active.consumed
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected_memory
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed memory request",
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
                    "{}: constrained request has no active admission",
                    self.provider_id
                )))
            } else {
                Ok(())
            };
        };
        if active.binding.as_ref() != Some(&RequestBinding::from_request(request))
            || active.consumed
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed or admission was already consumed",
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

pub struct MageMemoryScope {
    device: Device,
    provider_id: String,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    has_phases: bool,
    attention_chunk_sizes: Vec<u32>,
    transformer_window: Option<u32>,
    admission: Option<AdmissionRegistry>,
    token: Option<u64>,
    finished: bool,
}

impl MageMemoryScope {
    pub fn new(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> Self {
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .expect("Mage contract publishes bounded attention");
        Self {
            device,
            provider_id: contract.provider_id.clone(),
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            use_pid: context.use_pid,
            has_phases: context.has_phases,
            attention_chunk_sizes: attention.parameters.attention_chunk_sizes.clone(),
            transformer_window: contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then_some(context.selection.parameters.transformer_window_size)
                .flatten(),
            admission: None,
            token: None,
            finished: false,
        }
    }

    pub(crate) fn new_bound(
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
        admission: AdmissionRegistry,
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
                "{}: memory request scope is finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }
}

impl MemoryRequestScope for MageMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.active()?;
        let geometry = MemoryGeometry {
            width: request.width,
            height: request.height,
            batch: request.count,
            frames: request.frames.unwrap_or(1),
            reference_count: request.image_reference_count(),
        };
        if geometry != self.geometry {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request geometry changed after admission",
                self.provider_id
            )));
        }
        let has_phases = request
            .phases
            .as_ref()
            .is_some_and(|phases| !phases.is_empty());
        if request.use_pid != self.use_pid || has_phases != self.has_phases {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: forbidden request axes changed after admission",
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
        let _ = (tile_edge, overlap, geometry);
        Err(gen_core::Error::Unsupported(format!(
            "{}: bounded decode is structurally unavailable for the full-frame CoD decoder",
            self.provider_id
        )))
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.active()?;
        if self.attention_chunk_sizes.contains(&chunk_size) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: attention chunk {chunk_size} is not admitted",
                self.provider_id
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
                "{}: transformer streaming was not selected",
                self.provider_id
            )));
        };
        if first_block >= TRANSFORMER_BLOCKS
            || block_count == 0
            || !first_block.is_multiple_of(window)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: invalid transformer window {block_count} at {first_block}",
                self.provider_id
            )));
        }
        let expected = window.min(TRANSFORMER_BLOCKS - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: expected {expected} blocks at {first_block}, got {block_count}",
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

impl Drop for MageMemoryScope {
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

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let route = if is_edit(&contract.provider_id) {
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    } else {
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    };
    Ok(vec![gen_core::MemoryBehaviorFixture::new(
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            resolved_numeric_tier(spec)?,
            route,
        )?,
    )])
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_context(contract, context, resolved_quant(spec)?)?;
    Ok(Some(Box::new(MageMemoryScope::new(
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("sc15813-mage-packed-contract");
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    #[test]
    fn every_route_has_a_distinct_conformant_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let mut fingerprints = std::collections::BTreeSet::new();
        for id in PROVIDER_IDS {
            let contract = provider_contract_for(id, &spec(&tmp)).unwrap();
            assert!(
                contract.conformance_errors().is_empty(),
                "{id}: {:?}",
                contract.conformance_errors()
            );
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert!(fingerprints.insert(contract.calibration.as_ref().unwrap().fingerprint.clone()));
            for strategy in MemoryStrategy::ALL {
                let support = &contract.capability(strategy).unwrap().support;
                if strategy == MemoryStrategy::BoundedDecode {
                    assert!(matches!(
                        support,
                        MemoryStrategySupport::StructurallyNotApplicable { .. }
                    ));
                } else {
                    assert_eq!(support, &MemoryStrategySupport::Implemented);
                }
            }
        }
    }

    #[test]
    fn candidate_domains_and_block_geometry_are_exact() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract_rl(&spec(&tmp)).unwrap();
        assert_eq!(TRANSFORMER_BLOCKS, 12);
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert!(matches!(
            decode.support,
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        assert!(decode.parameters.decode_tile_edges.is_empty());
        assert!(decode.parameters.decode_overlaps.is_empty());
        assert!(!contract.lifecycle.decode_tiling);
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
    }

    #[test]
    fn t2i_and_edit_contexts_do_not_cross_authorize() {
        let tmp = tempfile::tempdir().unwrap();
        let t2i = contract_rl(&spec(&tmp)).unwrap();
        let edit = contract_edit(&spec(&tmp)).unwrap();
        let t2i_context =
            registered_valid_fixture(&spec(&tmp), &t2i, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;
        let edit_context =
            registered_valid_fixture(&spec(&tmp), &edit, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;
        assert!(validate_context(&t2i, &t2i_context, Some(Quant::Q4)).is_ok());
        assert!(validate_context(&edit, &edit_context, Some(Quant::Q4)).is_ok());
        assert!(validate_context(&t2i, &edit_context, Some(Quant::Q4)).is_err());
        assert!(validate_context(&edit, &t2i_context, Some(Quant::Q4)).is_err());
    }

    #[test]
    fn registered_receipts_bind_only_floors_active_for_the_loaded_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let base_spec = spec(&tmp);
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut selected_spec = base_spec.clone();
            selected_spec.quantize = quant;
            let contract = contract_rl(&selected_spec).unwrap();
            let tier = resolved_numeric_tier(&selected_spec).unwrap();
            assert_eq!(
                tier.component_precision_floors,
                crate::quant::active_component_precision_floors(quant),
                "resolved receipt must be tier-exact for {quant:?}"
            );
            let context = registered_valid_fixture(
                &selected_spec,
                &contract,
                MemoryStrategy::StagedResidency,
            )
            .unwrap()
            .remove(0)
            .context;
            assert!(validate_context(&contract, &context, quant).is_ok());

            if quant != Some(Quant::Q4) {
                let mut over_bound = context;
                over_bound.selection.tier.component_precision_floors =
                    crate::quant::COMPONENT_PRECISION_FLOORS;
                let error = validate_context(&contract, &over_bound, quant).unwrap_err();
                assert!(error.to_string().contains("does not match loaded tier"));
            } else {
                let mut under_bound = context;
                under_bound.selection.tier.component_precision_floors = &[];
                let error = validate_context(&contract, &under_bound, quant).unwrap_err();
                assert!(error.to_string().contains("does not match loaded tier"));
            }
        }
    }

    #[test]
    fn stale_fingerprint_and_resident_streaming_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut eager = spec(&tmp);
        eager.load_shape = LoadShape::EagerMaterialization;
        let contract = contract_rl(&eager).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );

        let contract = contract_rl(&spec(&tmp)).unwrap();
        let mut context =
            registered_valid_fixture(&spec(&tmp), &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .remove(0)
                .context;
        context.calibration_fingerprint.push_str(":stale");
        assert!(matches!(
            registered_safety_check(&spec(&tmp), &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn dense_quantized_directory_does_not_advertise_device_format_streaming() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        let mut dense = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        dense.load_shape = LoadShape::DeferredMaterialization;

        let contract = contract_rl(&dense).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn later_rungs_do_not_engage_structurally_unavailable_decode() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract_rl(&spec(&tmp)).unwrap();
        assert!(!contract.engages(
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedDecode
        ));
        let context =
            registered_valid_fixture(&spec(&tmp), &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .remove(0)
                .context;
        assert!(
            !contract
                .generation_memory(&context.selection)
                .unwrap()
                .tile_vae_decode
        );
        let mut scope = MageMemoryScope::new(Device::Cpu, &contract, &context);
        assert!(scope.configure_decode(1024, 1, context.geometry).is_err());
    }

    #[test]
    fn request_binding_rejects_pid_and_phase_mutation_before_or_after_configuration() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract_rl(&spec(&tmp)).unwrap();
        let context =
            registered_valid_fixture(&spec(&tmp), &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;

        let assert_mutation_rejected = |mutate: fn(&mut GenerationRequest)| {
            let admission = AdmissionRegistry::new(config::MODEL_ID);
            admission.approve(&context).unwrap();
            let mut scope =
                MageMemoryScope::new_bound(Device::Cpu, &contract, &context, admission.clone())
                    .unwrap();
            let mut request = GenerationRequest {
                prompt: "fixture".to_owned(),
                width: 1024,
                height: 1024,
                ..Default::default()
            };
            mutate(&mut request);
            assert!(scope.configure_request(&mut request).is_err());

            let admission = AdmissionRegistry::new(config::MODEL_ID);
            admission.approve(&context).unwrap();
            let mut scope =
                MageMemoryScope::new_bound(Device::Cpu, &contract, &context, admission.clone())
                    .unwrap();
            let mut request = GenerationRequest {
                prompt: "fixture".to_owned(),
                width: 1024,
                height: 1024,
                ..Default::default()
            };
            scope.configure_request(&mut request).unwrap();
            mutate(&mut request);
            assert!(admission.consume_for_generate(&request).is_err());
        };

        assert_mutation_rejected(|request| request.use_pid = true);
        assert_mutation_rejected(|request| {
            request.phases = Some(vec![gen_core::GenerationPhase {
                steps: 1,
                ..Default::default()
            }]);
        });
    }
}
