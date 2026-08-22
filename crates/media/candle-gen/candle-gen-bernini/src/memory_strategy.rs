//! Bernini Candle request-scoped memory contract.
//!
//! The CUDA provider has a real tiled Wan z16 VAE decode seam, so Resident and
//! BoundedDecode are declared. Older unconditional phase staging is not a
//! request-selected StagedResidency lever, so that rung remains Missing. Bernini has no Candle
//! deferred transformer loader or attention-chunking seam; those rungs remain
//! Missing rather than inheriting the MLX claims. Production calibration is
//! intentionally absent until the Windows/CUDA real-weight campaign exists.

use candle_gen::candle_core::Device;
use std::path::Path;

use candle_gen::gen_core::LoadShape;
use candle_gen::gen_core::{
    self, Conditioning, GenerationRequest, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges,
    MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization,
    ResidentRequestMemory,
};
#[cfg(any(feature = "cuda", test))]
use candle_gen::gen_core::{MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryMode};
use candle_gen::{CandleError, Result as CandleResult};

use crate::config::Defaults;
use candle_gen_wan::config::{DEFAULT_FRAMES_14B, MAX_AREA_14B, SIZE_MULTIPLE_14B};
use sha2::{Digest, Sha256};

pub const DECODE_OVERLAP: u32 = 64;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 384, 320, 256];
const STATIC_CALIBRATION: &str = "bernini-candle-registry-v2v-v1";
pub const ADVERTISED_GEOMETRIES: &[(u32, u32)] =
    &[(848, 480), (480, 848), (1280, 720), (720, 1280)];

fn tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

fn validate_geometry(width: u32, height: u32) -> gen_core::Result<()> {
    if ADVERTISED_GEOMETRIES.contains(&(width, height)) {
        Ok(())
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "Bernini memory evidence requires one of the advertised geometries {ADVERTISED_GEOMETRIES:?}, got {width}x{height}"
        )))
    }
}

fn known_provider(provider_id: &str) -> gen_core::Result<()> {
    [crate::pipeline::MODEL_ID, crate::bernini::MODEL_ID]
        .contains(&provider_id)
        .then_some(())
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!("unknown Bernini provider {provider_id}"))
        })
}

fn component_bytes(root: &Path, component: &str) -> gen_core::Result<u64> {
    let path = root.join(component);
    let bytes = gen_core::weightsmeta::safetensors_path_bytes(&path);
    if bytes == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini Candle memory contract requires a non-empty {component} safetensors component at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn quant_marker(root: &Path, component: &str) -> gen_core::Result<Option<u8>> {
    let path = root.join(component).join("quantize_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "{}: invalid quant marker: {error}",
                path.display()
            ))
        })?;
    let bits = value
        .get("bits")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!("{}: quant marker has no bits", path.display()))
        })?;
    u8::try_from(bits).map(Some).map_err(|_| {
        gen_core::Error::Unsupported(format!(
            "{}: quant marker bits {bits} is out of range",
            path.display()
        ))
    })
}

fn validate_quant_tier(root: &Path, spec: &LoadSpec, provider_id: &str) -> gen_core::Result<()> {
    let expected = match spec.quantize {
        None => 0,
        Some(gen_core::Quant::Q4) => 4,
        Some(gen_core::Quant::Q8) => 8,
        Some(other) => {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini Candle memory contract does not recognize quant tier {other:?}"
            )))
        }
    };
    let components = if provider_id == crate::bernini::MODEL_ID {
        ["transformer", "transformer_2", "mllm"].as_slice()
    } else {
        ["transformer", "transformer_2"].as_slice()
    };
    for component in components {
        let marker = quant_marker(root, component)?;
        let actual = marker.unwrap_or(0);
        if actual != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini Candle tier {:?} crossed {component} marker {actual}-bit (expected {expected}-bit)",
                spec.quantize
            )));
        }
    }
    Ok(())
}

fn adapter_identity(spec: &LoadSpec) -> gen_core::Result<(u64, String)> {
    if spec.adapters.is_empty() {
        return Ok((0, String::new()));
    }
    let mut total = 0_u64;
    let mut identities = Vec::with_capacity(spec.adapters.len());
    for adapter in &spec.adapters {
        let bytes = std::fs::read(&adapter.path).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "Bernini adapter {} is not readable: {error}",
                adapter.path.display()
            ))
        })?;
        if bytes.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini adapter {} is empty",
                adapter.path.display()
            )));
        }
        // The digest and resident bytes come from this same verified read. The metadata helper is
        // only a loadability/extension guard; a second stat must never become the priced source.
        let file_bytes = u64::try_from(bytes.len())
            .map_err(|_| gen_core::Error::Msg("Bernini adapter byte length overflow".into()))?;
        if gen_core::weightsmeta::safetensors_path_bytes(&adapter.path) != file_bytes
            || file_bytes == 0
        {
            return Err(gen_core::Error::Unsupported(format!(
                "Bernini adapter {} is not a non-empty load-exact safetensors artifact",
                adapter.path.display()
            )));
        }
        let expert_count = if adapter.moe_expert.is_none() { 2 } else { 1 };
        total = total
            .checked_add(file_bytes.saturating_mul(expert_count))
            .ok_or_else(|| {
                gen_core::Error::Msg("Bernini adapter resident bytes overflow".into())
            })?;
        identities.push(format!(
            "artifact={};digest=sha256:{:x};kind={:?};scale={:.9};expert={:?}",
            adapter.path.display(),
            Sha256::digest(&bytes),
            adapter.kind,
            adapter.scale,
            adapter.moe_expert
        ));
    }
    Ok((total, format!("adapters:[{}]", identities.join(","))))
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::StagedResidency => MemoryStrategySupport::Missing,
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    MemoryStrategySupport::Missing
                }
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

fn contract(
    provider_id: &str,
    spec: &LoadSpec,
    calibration: Option<MemoryCalibrationIdentity>,
    facts: MemoryAssetFacts,
    adapter_identity: Option<String>,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::FrameCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
    ];
    let resident_components = if facts.overlay_bytes > 0 {
        variables.push(MemoryFormulaVariable::OverlayBytes);
        vec![MemoryResidentComponent {
            id: adapter_identity.unwrap_or_else(|| "adapter_stack".to_owned()),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: facts.overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        }]
    } else {
        Vec::new()
    };
    let formula = if resident_components.is_empty() {
        MemoryFormulaKind::PhaseEnvelope {
            phases: phases.clone(),
            variables,
        }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases.clone(),
            variables,
            resident_components,
        }
    };
    MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula,
        calibration,
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

/// Weights-free declaration used by registry conformance. This is not production evidence.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if !matches!(
        provider_id,
        crate::pipeline::MODEL_ID | crate::bernini::MODEL_ID
    ) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown Bernini provider {provider_id}"
        )));
    }
    Ok(contract(
        provider_id,
        spec,
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION,
            spec.load_shape,
        )),
        MemoryAssetFacts::default(),
        None,
    ))
}

fn production_assets(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<(MemoryAssetFacts, MemoryNumericTier, Option<String>)> {
    known_provider(provider_id)?;
    let gen_core::WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "Bernini Candle memory contract requires a snapshot directory".into(),
        ));
    };
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(gen_core::Error::Unsupported(
            "Bernini Candle memory contract requires EagerMaterialization".into(),
        ));
    }
    validate_quant_tier(root, spec, provider_id)?;
    let conditioning = component_bytes(root, "text_encoder")?;
    let transformer = component_bytes(root, "transformer")?
        .checked_add(component_bytes(root, "transformer_2")?)
        .ok_or_else(|| gen_core::Error::Msg("Bernini transformer bytes overflow".into()))?;
    let decoder = component_bytes(root, "vae")?;
    let planner = if provider_id == crate::bernini::MODEL_ID {
        ["mllm", "connector", "vit_decoder"]
            .into_iter()
            .map(|component| component_bytes(root, component))
            .try_fold(0_u64, |total, next| {
                total
                    .checked_add(next?)
                    .ok_or_else(|| gen_core::Error::Msg("Bernini planner bytes overflow".into()))
            })?
    } else {
        0
    };
    let (overlay_bytes, overlay_identity) = adapter_identity(spec)?;
    let facts = MemoryAssetFacts {
        base_bytes: conditioning
            .checked_add(planner)
            .and_then(|value| value.checked_add(transformer))
            .and_then(|value| value.checked_add(decoder))
            .ok_or_else(|| gen_core::Error::Msg("Bernini base bytes overflow".into()))?,
        conditioning_bytes: conditioning
            .checked_add(planner)
            .ok_or_else(|| gen_core::Error::Msg("Bernini conditioning bytes overflow".into()))?,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes,
    };
    Ok((
        facts,
        tier(spec),
        (!overlay_identity.is_empty()).then_some(overlay_identity),
    ))
}

/// Real loads expose load-exact asset/tier identity, but no calibration identity until the
/// Windows/CUDA evidence campaign exists. That makes the shared selector reachable while every
/// optimized selection still refuses through the normal uncalibrated contract path.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let (facts, _tier, adapter_identity) = production_assets(provider_id, spec)?;
    Ok(contract(provider_id, spec, None, facts, adapter_identity))
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported("Bernini bounded decode needs tile edge".into())
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported("Bernini bounded decode needs overlap".into())
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "Bernini bounded decode requires edge in {DECODE_TILE_EDGES:?} and overlap {DECODE_OVERLAP}, got {edge}/{overlap}"
        )));
    }
    Ok(())
}

fn route_ok(contract: &MemoryProviderContract, context: &MemoryRunContext) -> gen_core::Result<()> {
    if context.mode.as_key() != "video_to_video"
        || context.geometry.reference_count != 1
        || !context.has_reference
        || context.geometry.batch != 1
        || context.use_pid
        || context.has_phases
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} memory evidence requires one video_to_video clip, one batch, and no PiD/phases",
            contract.provider_id
        )));
    }
    validate_geometry(context.geometry.width, context.geometry.height)?;
    if context.overlay.as_deref().is_some_and(|overlay| {
        overlay
            .split('+')
            .find(|axis| axis.starts_with("provider_video_mode:"))
            .is_some_and(|axis| axis != "provider_video_mode:v2v")
    }) {
        return Err(gen_core::Error::Unsupported(format!(
            "{} provider video mode overlay crossed the v2v contract",
            contract.provider_id
        )));
    }
    let adapter_axis = context.overlay.as_deref().and_then(|overlay| {
        overlay
            .split('+')
            .find(|axis| axis.starts_with("adapters:["))
    });
    let expected_adapter_axis = contract
        .resident_components()
        .iter()
        .find(|component| component.kind == MemoryComponentKind::AdapterStack)
        .map(|component| component.id.as_str());
    if adapter_axis != expected_adapter_axis {
        return Err(gen_core::Error::Unsupported(format!(
            "{} adapter artifact identity is missing or crossed the loaded contract",
            contract.provider_id
        )));
    }
    let area = u64::from(context.geometry.width) * u64::from(context.geometry.height);
    if !context.geometry.width.is_multiple_of(SIZE_MULTIPLE_14B)
        || !context.geometry.height.is_multiple_of(SIZE_MULTIPLE_14B)
        || area > MAX_AREA_14B as u64
        || !matches!(context.geometry.frames, 45 | 61 | 77)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} memory evidence does not cover {}x{} frames={}",
            contract.provider_id,
            context.geometry.width,
            context.geometry.height,
            context.geometry.frames
        )));
    }
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
        validate_decode(
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        )?;
    }
    Ok(())
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let gate = || route_ok(contract, context);
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&gate),
    )
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, tier(spec), context)
}

fn validate_request(request: &GenerationRequest) -> gen_core::Result<()> {
    if request.video_mode.as_deref() != Some("v2v")
        || request.fps.unwrap_or(Defaults::FPS) != Defaults::FPS
        || request.count != 1
        || request.image_reference_count() != 0
        || request.video_clips().len() != 1
        || !matches!(
            request.conditioning.as_slice(),
            [Conditioning::VideoClip { .. }]
        )
        || !matches!(request.frames, Some(45 | 61 | 77))
    {
        return Err(gen_core::Error::Unsupported(
            "Bernini Candle memory scope requires exactly one VideoClip, v2v, FPS16, and 3/4/5s"
                .to_owned(),
        ));
    }
    validate_geometry(request.width, request.height)?;
    Ok(())
}

struct BerniniMemoryRequestScope {
    inner: candle_gen::request_scope::CandleRequestScopeCore,
}

impl MemoryRequestScope for BerniniMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        validate_request(request)?;
        self.inner.configure_request(request)
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
        mut geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        geometry.reference_count = 0;
        self.inner.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, chunk: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.inner.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.inner.finish(outcome)
    }
}

#[cfg(any(feature = "cuda", test))]
pub fn contract_for_loaded(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<Option<(MemoryProviderContract, MemoryNumericTier)>> {
    let (facts, loaded_tier, adapter_identity) = match production_assets(provider_id, spec) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    Ok(Some((
        contract(provider_id, spec, None, facts, adapter_identity),
        loaded_tier,
    )))
}

#[cfg(any(feature = "cuda", test))]
pub fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("video_to_video".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.geometry.width = 848;
    context.geometry.height = 480;
    context.geometry.frames = 45;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free Bernini v2v memory behavior".to_owned();
    fixture.request.video_mode = Some("v2v".to_owned());
    fixture.request.fps = Some(16);
    fixture.request.conditioning.clear();
    fixture.request.conditioning.push(Conditioning::VideoClip {
        frames: vec![gen_core::Image {
            width: 2,
            height: 2,
            pixels: vec![0; 12],
        }],
        frame_idx: 0,
        strength: 1.0,
    });
    Ok(vec![fixture])
}

#[cfg(any(feature = "cuda", test))]
pub fn registered_begin_request(
    provider_id: &str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let Some((_, loaded_tier)) = contract_for_loaded(spec, provider_id)? else {
        return Ok(None);
    };
    begin_request(contract, loaded_tier, Device::Cpu, context)
}

pub fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, loaded_tier, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut geometry = context.geometry;
    geometry.reference_count = 0;
    let provider_id = match contract.provider_id.as_str() {
        crate::pipeline::MODEL_ID => crate::pipeline::MODEL_ID,
        crate::bernini::MODEL_ID => crate::bernini::MODEL_ID,
        _ => {
            return Err(gen_core::Error::Unsupported(
                "unknown Bernini provider".into(),
            ))
        }
    };
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        80,
        |_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.default_frames = DEFAULT_FRAMES_14B;
    Ok(Some(Box::new(BerniniMemoryRequestScope {
        inner: candle_gen::request_scope::CandleRequestScopeCore::new(config),
    })))
}

pub fn selected_decode_cap(request: &GenerationRequest) -> CandleResult<Option<u32>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(CandleError::Msg(
                "Bernini decode parameters require bounded decode".into(),
            ));
        }
        return Ok(None);
    }
    validate_decode(memory.decode_tile_edge, memory.decode_overlap)
        .map_err(|error| CandleError::Msg(error.to_string()))?;
    Ok(memory.decode_tile_edge)
}

fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

pub const RENDERER_MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: crate::pipeline::MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(crate::pipeline::MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };

pub const FULL_MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: crate::bernini::MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(crate::bernini::MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };

pub const RENDERER_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: crate::pipeline::MODEL_ID,
        contract: |spec| memory_strategy_contract(crate::pipeline::MODEL_ID, spec),
        safety_check: registered_safety_check,
    };

pub const FULL_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::bernini::MODEL_ID,
    contract: |spec| memory_strategy_contract(crate::bernini::MODEL_ID, spec),
    safety_check: registered_safety_check,
};

#[cfg(any(feature = "cuda", test))]
pub const RENDERER_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::pipeline::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: |spec, contract, context| {
            registered_begin_request(crate::pipeline::MODEL_ID, spec, contract, context)
        },
    };

#[cfg(any(feature = "cuda", test))]
pub const FULL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::bernini::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: |spec, contract, context| {
            registered_begin_request(crate::bernini::MODEL_ID, spec, contract, context)
        },
    };

/// Add the weights-free Bernini contract surfaces to an external catalog walk. Production loads
/// still expose no contract until the Windows/CUDA evidence campaign mints one.
pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_contract_fixture(RENDERER_MEMORY_FIXTURE)
        .register_memory_contract_fixture(FULL_MEMORY_FIXTURE)
}

/// Provider-owned registration hook used by the CUDA catalog and source-derived wiring checks.
pub fn register_memory_strategy(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(RENDERER_MEMORY_REGISTRATION)
        .register_memory_strategy(FULL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(RENDERER_MEMORY_FIXTURE)
        .register_memory_contract_fixture(FULL_MEMORY_FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_bernini_declares_missing_attention_and_transformer_rungs() {
        let spec = LoadSpec::new(gen_core::WeightsSource::Dir("/missing".into()));
        let contract =
            weights_free_memory_strategy_contract(crate::pipeline::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn candle_bernini_decode_cap_is_exactly_admitted() {
        let mut request = GenerationRequest {
            memory: Some(gen_core::GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(selected_decode_cap(&request).unwrap(), Some(512));
        request.memory.as_mut().unwrap().decode_tile_edge = Some(511);
        assert!(selected_decode_cap(&request).is_err());
    }

    #[test]
    fn candle_bernini_scope_rejects_plain_and_wrong_fps_requests() {
        let clip = Conditioning::VideoClip {
            frames: vec![gen_core::Image {
                width: 2,
                height: 2,
                pixels: vec![0; 12],
            }],
            frame_idx: 0,
            strength: 1.0,
        };
        let mut request = GenerationRequest {
            prompt: "v2v".to_owned(),
            width: 848,
            height: 480,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("v2v".to_owned()),
            conditioning: vec![clip],
            ..Default::default()
        };
        assert!(validate_request(&request).is_ok());
        request.video_mode = None;
        assert!(validate_request(&request).is_err());
        request.video_mode = Some("v2v".to_owned());
        request.fps = Some(24);
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn candle_bernini_scope_admits_only_advertised_geometry_and_frames() {
        for &(width, height) in ADVERTISED_GEOMETRIES {
            for frames in [45, 61, 77] {
                let request = GenerationRequest {
                    prompt: "v2v".to_owned(),
                    width,
                    height,
                    frames: Some(frames),
                    fps: Some(16),
                    video_mode: Some("v2v".to_owned()),
                    conditioning: vec![Conditioning::VideoClip {
                        frames: vec![gen_core::Image {
                            width: 2,
                            height: 2,
                            pixels: vec![0; 12],
                        }],
                        frame_idx: 0,
                        strength: 1.0,
                    }],
                    ..Default::default()
                };
                assert!(
                    validate_request(&request).is_ok(),
                    "{width}x{height}/{frames}"
                );
            }
        }
        let mut crossed = GenerationRequest {
            width: 640,
            height: 640,
            frames: Some(45),
            fps: Some(16),
            video_mode: Some("v2v".to_owned()),
            conditioning: vec![Conditioning::VideoClip {
                frames: vec![gen_core::Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 12],
                }],
                frame_idx: 0,
                strength: 1.0,
            }],
            ..Default::default()
        };
        assert!(validate_request(&crossed).is_err());
        crossed.width = 848;
        crossed.height = 480;
        crossed.fps = Some(24);
        assert!(validate_request(&crossed).is_err());
    }

    #[test]
    fn candle_bernini_loaded_contract_prices_shared_adapter_per_expert_and_exact_tier() {
        let root = tempfile::tempdir().unwrap();
        for component in [
            "text_encoder",
            "transformer",
            "transformer_2",
            "mllm",
            "connector",
            "vit_decoder",
            "vae",
        ] {
            std::fs::create_dir_all(root.path().join(component)).unwrap();
            std::fs::write(
                root.path().join(component).join("model.safetensors"),
                vec![7; 8],
            )
            .unwrap();
            if ["transformer", "transformer_2", "mllm"].contains(&component) {
                std::fs::write(
                    root.path().join(component).join("quantize_config.json"),
                    br#"{"bits":4,"quantization":{"group_size":64}}"#,
                )
                .unwrap();
            }
        }
        let adapter = root.path().join("adapter.safetensors");
        std::fs::write(&adapter, vec![3; 11]).unwrap();
        let high_adapter = root.path().join("high-adapter.safetensors");
        std::fs::write(&high_adapter, vec![5; 7]).unwrap();
        let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()))
            .with_quant(gen_core::Quant::Q4);
        spec.adapters = vec![
            gen_core::AdapterSpec::new(adapter, 0.5, gen_core::AdapterKind::Lora),
            gen_core::AdapterSpec {
                path: high_adapter,
                scale: 1.0,
                kind: gen_core::AdapterKind::Lokr,
                moe_expert: Some(gen_core::MoeExpert::High),
                pass_scales: None,
            },
        ];
        let contract = memory_strategy_contract(crate::bernini::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.calibration, None);
        assert_eq!(contract.asset_facts.overlay_bytes, 29);
        assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert!(contract.resident_components().iter().any(|component| {
            component.kind == MemoryComponentKind::AdapterStack
                && component.resident_bytes == 29
                && component.id.contains("digest=sha256:")
        }));
        let stale = LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_owned()));
        assert!(memory_strategy_contract(crate::bernini::MODEL_ID, &stale).is_err());
    }
}
