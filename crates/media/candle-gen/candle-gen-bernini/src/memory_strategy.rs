//! Bernini Candle request-scoped memory contract.
//!
//! The CUDA provider has a real tiled Wan z16 VAE decode seam, so Resident and
//! BoundedDecode are declared. Older unconditional phase staging is not a
//! request-selected StagedResidency lever, so that rung remains Missing. Bernini has no Candle
//! deferred transformer loader or attention-chunking seam; those rungs remain
//! Missing rather than inheriting the MLX claims. Production calibration is
//! intentionally absent until the Windows/CUDA real-weight campaign exists.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::LoadShape;
use candle_gen::gen_core::{
    self, Conditioning, GenerationRequest, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    MemoryWindowMaterialization, ResidentRequestMemory,
};
#[cfg(any(feature = "cuda", test))]
use candle_gen::gen_core::{MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryMode};
use candle_gen::{CandleError, Result as CandleResult};

use crate::config::Defaults;
use candle_gen_wan::config::{DEFAULT_FRAMES_14B, MAX_AREA_14B, SIZE_MULTIPLE_14B};

pub const DECODE_OVERLAP: u32 = 64;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 384, 320, 256];
const STATIC_CALIBRATION: &str = "bernini-candle-registry-v2v-v1";

fn tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
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
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
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
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::DecodeTileArea,
            ],
        },
        calibration,
        asset_facts: MemoryAssetFacts::default(),
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
    ))
}

/// Real loads currently expose no contract: no Windows/CUDA evidence has been minted.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let _ = (provider_id, spec);
    Err(gen_core::Error::Unsupported(
        "Bernini Candle memory contract has no production calibration evidence".to_owned(),
    ))
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
    if let Some(overlay) = context.overlay.as_deref() {
        if let Some(adapter_axis) = overlay
            .split('+')
            .find(|axis| axis.starts_with("adapters:["))
        {
            for field in ["artifact=", "digest=sha256:", "kind=", "scale=", "expert="] {
                if !adapter_axis.contains(field) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{} adapter overlay is missing exact {field} identity",
                        contract.provider_id
                    )));
                }
            }
        }
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
    _spec: &LoadSpec,
    _provider_id: &str,
) -> gen_core::Result<Option<(MemoryProviderContract, MemoryNumericTier)>> {
    // No production fingerprint or adapter artifact digest exists yet. Keep the generator's
    // historical path available while refusing optimized admission until the Windows/CUDA campaign.
    Ok(None)
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
}
