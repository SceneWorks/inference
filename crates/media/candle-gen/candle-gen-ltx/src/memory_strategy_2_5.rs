//! The production LTX-2.5 distilled Candle memory registration (SC-18797).
//!
//! This stays separate from the released LTX-2.3 q4 contract in [`crate::memory_strategy`]. The
//! registry, loaded generator, shared Candle request scope, and split transformer materializer all
//! consume this declaration. Removing any one of those links makes the reachability tests fail.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationRequest, LoadShape, LoadSpec, LtxComponent, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport, OffloadPolicy, Quant,
    ResidentRequestMemory, TransformerComponent, WeightsSource,
};

use crate::config::DEFAULT_FRAMES;
use crate::memory_strategy::{DECODE_OVERLAP, DECODE_TILE_EDGES};

/// The Candle LTX-2.5 engine id; never alias the released LTX-2.3 route.
pub const LTX_2_5_DISTILLED_MODEL_ID: &str = "ltx_2_5_distilled";

/// Distinguishable 48-block-window candidates. `48` is absent because it bounds nothing.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 16];

/// One AV block owns both audio and video attention, but not the separate Gemma-4 encoder.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Shared attention score-budget candidates, in score elements per chunk.
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[67_108_864, 16_777_216];

const CALIBRATION_FINGERPRINT: &str = "sc-18797-ltx-2-5-candle-ladder-v1";
const CANDLE_REGISTRY_CALIBRATION_FINGERPRINT: &str = "sc-18797-ltx-2-5-candle-registry-v1";

/// Request-local split-transformer construction choices after exact contract validation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TransformerExecution {
    pub attention_chunk_size: Option<u32>,
    pub transformer_window_size: Option<u32>,
}

impl TransformerExecution {
    pub fn is_streamed(self) -> bool {
        self.transformer_window_size.is_some()
    }
}

/// Rung 4 is meaningful only with a sequential, deferred, re-openable, adapter-free source.
/// [`crate::block_stream::LtxBlockStream`] enforces the same adapter restriction at execution.
fn streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.adapters.is_empty()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn uses_diffusion_decoder(spec: &LoadSpec) -> bool {
    spec.components
        .contains_key(LtxComponent::DiffusionVideoVae.id())
}

fn bounded_decode(spec: &LoadSpec) -> bool {
    // The convolutional decoder accepts the exact output-pixel tile domain below. DiffVAE owns an
    // automatic budgeted route but has no request-selectable edge/overlap seam, so it must not borrow
    // the convolutional rung.
    !uses_diffusion_decoder(spec)
}

fn strategies(spec: &LoadSpec) -> Vec<MemoryStrategyCapability> {
    let windowed = streamable(spec);
    let decode = bounded_decode(spec);
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedAttention => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedDecode if decode => MemoryStrategySupport::Implemented,
                MemoryStrategy::BoundedTransformerResidency if windowed => {
                    MemoryStrategySupport::Implemented
                }
                _ => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode if decode => MemoryParameterRanges {
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: ATTENTION_CHUNK_SIZES.to_vec(),
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if windowed => MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![TRANSFORMER_WINDOW_COMPONENT],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

fn collect_safetensors(path: &Path, files: &mut Vec<PathBuf>) -> gen_core::Result<()> {
    let metadata = std::fs::metadata(path)?;
    if metadata.is_file() {
        if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
            files.push(path.to_path_buf());
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: component source is neither a file nor directory: {}",
            path.display()
        )));
    }
    let mut children = std::fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    children.sort();
    for child in children {
        collect_safetensors(&child, files)?;
    }
    Ok(())
}

/// Count the exact tensor payload a component materializes: floating tensors at the component's
/// runtime width and packed integer codes at their physical stored width. This uses only headers;
/// no model tensor is allocated while the generator publishes its loaded contract.
fn required_component_bytes(path: &Path, float_width: u64, label: &str) -> gen_core::Result<u64> {
    let mut files = Vec::new();
    collect_safetensors(path, &mut files)?;
    let bytes = files.into_iter().try_fold(0_u64, |total, file| {
        let file_bytes = gen_core::weightsmeta::safetensors_path_tensor_headers(&file)?
            .into_iter()
            .try_fold(0_u64, |sum, header| {
                let bytes = if header.is_float() {
                    header.materialized_bytes(float_width)?
                } else {
                    header.data_bytes
                };
                sum.checked_add(bytes).ok_or_else(|| {
                    gen_core::Error::Msg(format!(
                        "{LTX_2_5_DISTILLED_MODEL_ID}: {label} tensor byte total overflows u64"
                    ))
                })
            })?;
        total.checked_add(file_bytes).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: {label} file byte total overflows u64"
            ))
        })
    })?;
    if bytes == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: {label} at {} has no countable safetensors tensor bytes",
            path.display()
        )));
    }
    Ok(bytes)
}

fn optional_component_bytes(path: &Path, float_width: u64, label: &str) -> gen_core::Result<u64> {
    if path.exists() {
        required_component_bytes(path, float_width, label)
    } else {
        Ok(0)
    }
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> gen_core::Result<u64> {
    values.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: {label} byte total overflows u64"
            ))
        })
    })
}

fn exact_load_receipt(
    spec: &LoadSpec,
    bundle: &gen_core::ltx_checkpoint::LtxBundle,
) -> gen_core::Result<MemoryAssetFacts> {
    let video_component = if uses_diffusion_decoder(spec) {
        LtxComponent::DiffusionVideoVae
    } else {
        LtxComponent::ConvVideoVae
    };
    let component_path = |component| bundle.require(component).map(|resolved| resolved.path());
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.as_path(),
        WeightsSource::File(path) => path.parent().unwrap_or_else(|| std::path::Path::new(".")),
    };
    let video_path = component_path(video_component)?;
    let encoder_path = crate::ltx25_encoder_path(root, video_component, video_path);
    let transformer_path = component_path(LtxComponent::Transformer)?;
    let connector_path = transformer_path.with_file_name("connector.safetensors");
    let separate_encoder = (encoder_path != video_path)
        .then(|| required_component_bytes(&encoder_path, 4, "video VAE encoder"))
        .transpose()?
        .unwrap_or(0);

    let conditioning_bytes = checked_sum(
        "conditioning component",
        [
            required_component_bytes(
                component_path(LtxComponent::TextEncoder)?,
                2,
                "Gemma-4 text encoder",
            )?,
            optional_component_bytes(&connector_path, 2, "audio/video connector")?,
            separate_encoder,
            required_component_bytes(
                component_path(LtxComponent::DurationHead)?,
                4,
                "duration head",
            )?,
        ],
    )?;
    let transformer_bytes = checked_sum(
        "transformer component",
        [
            required_component_bytes(transformer_path, 2, "AV transformer")?,
            required_component_bytes(
                component_path(LtxComponent::SpatialUpsampler)?,
                4,
                "spatial upsampler",
            )?,
            required_component_bytes(
                component_path(LtxComponent::TemporalUpsampler)?,
                4,
                "temporal upsampler",
            )?,
        ],
    )?;
    let decoder_bytes = checked_sum(
        "decoder component",
        [
            required_component_bytes(video_path, 4, "selected video decoder")?,
            required_component_bytes(
                component_path(LtxComponent::AudioVae)?,
                4,
                "audio VAE and vocoder",
            )?,
        ],
    )?;
    let overlay_bytes = spec.adapters.iter().try_fold(0_u64, |total, adapter| {
        total
            .checked_add(required_component_bytes(&adapter.path, 2, "adapter")?)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{LTX_2_5_DISTILLED_MODEL_ID}: adapter byte total overflows u64"
                ))
            })
    })?;
    let base_bytes = checked_sum(
        "base component",
        [conditioning_bytes, transformer_bytes, decoder_bytes],
    )?;
    Ok(MemoryAssetFacts {
        base_bytes,
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes,
    })
}

/// Build one LTX-2.5 Candle contract from an already resolved physical receipt. Rung 3 executes
/// through `candle_gen::sdpa_budgeted_bhsd`; rung 4 executes through Candle's binding over
/// `gen_core::block_window`; bounded decode is declared only for the convolutional VAE route.
fn build_contract(
    spec: &LoadSpec,
    asset_facts: MemoryAssetFacts,
    calibration_fingerprint: &str,
) -> gen_core::Result<MemoryProviderContract> {
    let windowed = streamable(spec);
    let decode = bounded_decode(spec);
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
        MemoryFormulaVariable::OverlayBytes,
    ];
    if decode {
        variables.push(MemoryFormulaVariable::DecodeTileArea);
    }
    Ok(MemoryProviderContract {
        architecture_facts: candle_gen::gen_core::MemoryArchitectureFacts::default(),
        provider_id: LTX_2_5_DISTILLED_MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(spec),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: decode,
            attention_chunking: true,
            transformer_window_materialization: windowed,
        },
        formula: MemoryFormulaKind::PhaseEnvelope { phases, variables },
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

/// Build the exact production contract by resolving and sizing every component the selected route
/// can materialize. Missing weights are an error here; only the fixture callback below is synthetic.
pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    memory_strategy_contract_for_bundle(spec, &bundle)
}

fn memory_strategy_contract_for_bundle(
    spec: &LoadSpec,
    bundle: &gen_core::ltx_checkpoint::LtxBundle,
) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        exact_load_receipt(spec, bundle)?,
        CALIBRATION_FINGERPRINT,
    )
}

fn weights_free_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    // Registry surfaces are executable shape declarations, not installed artifacts. Keeping the
    // physical receipt zero is the explicit gen-core convention for such fixtures; production
    // registration never calls this callback.
    build_contract(
        spec,
        MemoryAssetFacts::default(),
        CANDLE_REGISTRY_CALIBRATION_FINGERPRINT,
    )
}

/// Bind a shared selection to the exact numeric tier the provider loaded.
pub fn resolved_numeric_tier(spec: &LoadSpec) -> gen_core::Result<MemoryNumericTier> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(path) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: physical numeric-tier resolution requires a split bundle directory, got {}",
                path.display()
            )))
        }
    };
    let manifest_path = root.join(crate::tier::TIER_MANIFEST_FILE);
    let quant = if manifest_path.is_file() {
        let tier = crate::tier::Ltx25Tier::detect(root)
            .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "{LTX_2_5_DISTILLED_MODEL_ID}: {} does not declare an LTX-2.5 converted tier",
                    manifest_path.display()
                ))
            })?;
        let manifest = tier.manifest();
        let physical = match (manifest.quantized, manifest.quant.bits, manifest.tier.as_str()) {
            (false, _, "bf16") => None,
            (true, 4, "q4") => Some(Quant::Q4),
            (true, 8, "q8") => Some(Quant::Q8),
            _ => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{LTX_2_5_DISTILLED_MODEL_ID}: split_model.json tier {:?}, quantized={}, bits={} is contradictory or unsupported",
                    manifest.tier, manifest.quantized, manifest.quant.bits
                )))
            }
        };
        if spec.quantize != physical {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: requested quant tier {:?} disagrees with split_model.json tier {physical:?}",
                spec.quantize
            )));
        }
        physical
    } else {
        // Weights-free fixtures have no files and use the explicit selector as their synthetic
        // tier axis. Raw upstream split bundles likewise carry no converted-tier manifest.
        spec.quantize
    };
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
    bundle: &gen_core::ltx_checkpoint::LtxBundle,
) -> gen_core::Result<(MemoryProviderContract, MemoryNumericTier)> {
    Ok((
        memory_strategy_contract_for_bundle(spec, bundle)?,
        resolved_numeric_tier(spec)?,
    ))
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: bounded decode requires a selected tile edge"
        ))
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: bounded decode requires a selected overlap"
        ))
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: decode tile {edge}/{overlap} is outside the published convolutional-VAE domain"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.use_pid || context.has_phases {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: memory admission does not cover PiD or phased requests"
            )));
        }
        if context.overlay.as_deref() != expected_overlay {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: request overlay {:?} does not match the loaded adapter stack {:?}",
                context.overlay, expected_overlay
            )));
        }
        if contract.engages_selection(&context.selection, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&route_gate),
    )
}

fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let fixture = contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == CANDLE_REGISTRY_CALIBRATION_FINGERPRINT);
    let expected = if fixture {
        weights_free_contract(spec)
    } else {
        memory_strategy_contract(spec)
    };
    match expected {
        Ok(expected) if expected == *contract => match resolved_numeric_tier(spec) {
            Ok(tier) => safety_check(
                contract,
                tier,
                gen_core::adapter_stack_identity(&spec.adapters).as_deref(),
                context,
            ),
            Err(error) => MemorySafetyDecision::Reject {
                reason: error.to_string(),
            },
        },
        Ok(_) => MemorySafetyDecision::Reject {
            reason: format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: caller contract differs from the exact registered load contract"
            ),
        },
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

fn begin(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, loaded_tier, expected_overlay, context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        LTX_2_5_DISTILLED_MODEL_ID,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        48,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    config.default_frames = DEFAULT_FRAMES;
    if contract.engages_selection(&context.selection, MemoryStrategy::BoundedAttention) {
        config.attention_chunk_size = context.selection.parameters.attention_chunk_size;
    }
    if contract.engages_selection(
        &context.selection,
        MemoryStrategy::BoundedTransformerResidency,
    ) {
        config.transformer_window = context.selection.parameters.transformer_window_size;
    }
    Ok(Some(Box::new(
        candle_gen::request_scope::CandleRequestScopeCore::new(config),
    )))
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin(contract, loaded_tier, expected_overlay, device, context)
}

fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    match weights_free_contract(spec) {
        Ok(expected) if expected == *contract => begin(
            contract,
            resolved_numeric_tier(spec)?,
            gen_core::adapter_stack_identity(&spec.adapters).as_deref(),
            Device::Cpu,
            context,
        ),
        Ok(_) => Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: caller contract differs from the exact registered load contract"
        ))),
        Err(error) => Err(error),
    }
}

fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        resolved_numeric_tier(spec)?,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".to_owned()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.geometry.width = 512;
    context.geometry.height = 512;
    context.geometry.frames = 17;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request = GenerationRequest {
        prompt: "weights-free Candle LTX-2.5 memory behavior".to_owned(),
        width: 512,
        height: 512,
        frames: Some(17),
        ..Default::default()
    };
    Ok(vec![fixture.with_load_spec(spec.clone())])
}

/// The production Candle LTX-2.5 memory registration.
pub(crate) const LTX_2_5_DISTILLED_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: LTX_2_5_DISTILLED_MODEL_ID,
        contract: memory_strategy_contract,
        safety_check: registered_safety_check,
    };

/// Complete weights-free BF16/Q4/Q8 load surface for the registered Candle route — the shared
/// Candle surface, unfiltered (sc-18791). LTX-2.5 converts and ships a hosted `q8/` tier beside
/// `q4/`, so excluding the Q8 rungs would leave the shipped tier's request scope unexercised.
pub(crate) const LTX_2_5_DISTILLED_MEMORY_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: LTX_2_5_DISTILLED_MODEL_ID,
        contract: weights_free_contract,
        surface_specs: gen_core::candle_memory_contract_surface_specs,
    };

/// Executable weights-free request-scope behavior for every implemented optimized rung.
pub(crate) const LTX_2_5_DISTILLED_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: LTX_2_5_DISTILLED_MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

/// Validate and resolve request-local transformer loader choices. Both `Generator::validate` and
/// the split component constructor call this, so a direct trait caller cannot bypass the shared
/// request scope with an unadvertised value.
pub(crate) fn transformer_execution(
    request: &GenerationRequest,
    contract: Option<&MemoryProviderContract>,
    use_diffusion_decoder: bool,
) -> gen_core::Result<TransformerExecution> {
    let Some(memory) = request.memory else {
        return Ok(TransformerExecution::default());
    };
    let claims_ladder = memory.stage_residency
        || memory.tile_vae_decode
        || memory.chunk_attention
        || memory.stream_transformer_blocks
        || memory.decode_tile_edge.is_some()
        || memory.decode_overlap.is_some()
        || memory.attention_chunk_size.is_some()
        || memory.transformer_window_size.is_some()
        || memory.transformer_window_component.is_some();
    let contract = match (claims_ladder, contract) {
        (false, _) => return Ok(TransformerExecution::default()),
        (true, Some(contract)) => contract,
        (true, None) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: loaded route has no CUDA memory-strategy contract"
            )))
        }
    };

    let implemented = |strategy| {
        matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    };
    if memory.stage_residency && !implemented(MemoryStrategy::StagedResidency) {
        return Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: staged residency is not implemented by this loaded route"
        )));
    }
    if memory.tile_vae_decode {
        if use_diffusion_decoder || !implemented(MemoryStrategy::BoundedDecode) {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: request-selected bounded decode is available only on the convolutional VAE route"
            )));
        }
        validate_decode(memory.decode_tile_edge, memory.decode_overlap)?;
    } else if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: decode parameters were supplied without bounded decode"
        )));
    }

    let attention_chunk_size = if memory.chunk_attention {
        let chunk = memory.attention_chunk_size.ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: bounded attention requires a selected chunk size"
            ))
        })?;
        if !implemented(MemoryStrategy::BoundedAttention) || !ATTENTION_CHUNK_SIZES.contains(&chunk)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: attention chunk {chunk} is outside the published domain"
            )));
        }
        Some(chunk)
    } else {
        if memory.attention_chunk_size.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: attention chunk supplied without bounded attention"
            )));
        }
        None
    };

    let transformer_window_size = if memory.stream_transformer_blocks {
        let window = memory.transformer_window_size.ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: streamed transformer requires a selected window size"
            ))
        })?;
        let component = memory
            .transformer_window_component
            .unwrap_or(TransformerComponent::Dit);
        if component != TRANSFORMER_WINDOW_COMPONENT {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: transformer window component {component:?} is not the published AV DiT scope"
            )));
        }
        if !implemented(MemoryStrategy::BoundedTransformerResidency)
            || !TRANSFORMER_WINDOW_SIZES.contains(&window)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: transformer window {window} is outside the loaded route's published domain"
            )));
        }
        Some(window)
    } else {
        if memory.transformer_window_size.is_some() || memory.transformer_window_component.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: transformer window parameters were supplied without streamed transformer blocks"
            )));
        }
        None
    };

    Ok(TransformerExecution {
        attention_chunk_size,
        transformer_window_size,
    })
}

pub(crate) fn selected_decode_cap(
    request: &GenerationRequest,
    use_diffusion_decoder: bool,
) -> gen_core::Result<Option<(u32, u32)>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{LTX_2_5_DISTILLED_MODEL_ID}: decode parameters were supplied without bounded decode"
            )));
        }
        return Ok(None);
    }
    if use_diffusion_decoder {
        return Err(gen_core::Error::Unsupported(format!(
            "{LTX_2_5_DISTILLED_MODEL_ID}: DiffVAE uses its automatic budgeted route and does not accept convolutional tile parameters"
        )));
    }
    validate_decode(memory.decode_tile_edge, memory.decode_overlap)?;
    Ok(Some((
        memory.decode_tile_edge.expect("validated"),
        memory.decode_overlap.expect("validated"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};

    fn write_safetensors(path: &Path, metadata: &[(&str, &str)], elements: usize) {
        let metadata = metadata
            .iter()
            .map(|(key, value)| format!("{}:{}", serde_json::json!(key), serde_json::json!(value)))
            .collect::<Vec<_>>()
            .join(",");
        let data_bytes = elements * 4;
        let header = format!(
            r#"{{"__metadata__":{{{metadata}}},"w":{{"dtype":"F32","shape":[{elements}],"data_offsets":[0,{data_bytes}]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + data_bytes, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn add_component(
        spec: LoadSpec,
        root: &Path,
        component: LtxComponent,
        name: &str,
        config_key: &str,
        config: &str,
        elements: usize,
    ) -> LoadSpec {
        let path = root.join(name);
        write_safetensors(
            &path,
            &[("model_version", "2.5.0"), (config_key, config)],
            elements,
        );
        spec.with_component(component.id(), WeightsSource::File(path))
    }

    fn physical_spec(root: &Path, diffusion_vae: bool, with_encoder: bool) -> LoadSpec {
        let mut load = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_resolved_route(LTX_2_5_DISTILLED_MODEL_ID)
            .with_quant(Quant::Q4);
        load = add_component(
            load,
            root,
            LtxComponent::Transformer,
            "transformer.safetensors",
            "config",
            r#"{"transformer":{"_class_name":"AVTransformer3DModel"}}"#,
            11,
        );
        write_safetensors(&root.join("connector.safetensors"), &[], 43);
        load = add_component(
            load,
            root,
            LtxComponent::TextEncoder,
            "text_encoder.safetensors",
            "gemma_config",
            r#"{"model_type":"gemma4_unified","gemma_version":"gemma4-12b-ltx-v1"}"#,
            13,
        );
        load = add_component(
            load,
            root,
            if diffusion_vae {
                LtxComponent::DiffusionVideoVae
            } else {
                LtxComponent::ConvVideoVae
            },
            if diffusion_vae {
                "diffusion_vae_decoder.safetensors"
            } else {
                "vae_decoder.safetensors"
            },
            "config",
            if diffusion_vae {
                r#"{"vae":{"_class_name":"CausalDiffusionVAE"}}"#
            } else {
                r#"{"vae":{"_class_name":"CausalVideoAutoencoder"}}"#
            },
            17,
        );
        load = add_component(
            load,
            root,
            LtxComponent::AudioVae,
            "audio_vae.safetensors",
            "config",
            r#"{"audio_vae":{},"vocoder":{}}"#,
            19,
        );
        load = add_component(
            load,
            root,
            LtxComponent::DurationHead,
            "duration.safetensors",
            "config",
            r#"{"duration_head":{},"transformer":{}}"#,
            23,
        );
        load = add_component(
            load,
            root,
            LtxComponent::SpatialUpsampler,
            "spatial.safetensors",
            "config",
            r#"{"_class_name":"LatentUpsampler","spatial_upsample":true,"temporal_upsample":false}"#,
            29,
        );
        load = add_component(
            load,
            root,
            LtxComponent::TemporalUpsampler,
            "temporal.safetensors",
            "config",
            r#"{"_class_name":"LatentUpsampler","spatial_upsample":false,"temporal_upsample":true}"#,
            31,
        );
        if with_encoder {
            write_safetensors(&root.join("vae_encoder.safetensors"), &[], 37);
        }
        load
    }

    fn write_tier_manifest(root: &Path, tier: &str, quantized: bool, bits: usize) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join(crate::tier::TIER_MANIFEST_FILE),
            serde_json::json!({
                "tier": tier,
                "model_version": "2.5.0",
                "quantized": quantized,
                "quantization_bits": bits,
                "quantization_group_size": 64,
                "component_detail": []
            })
            .to_string(),
        )
        .unwrap();
    }

    fn spec(policy: OffloadPolicy, shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-2-5".into()))
            .with_resolved_route(LTX_2_5_DISTILLED_MODEL_ID)
            .with_quant(Quant::Q4)
            .with_offload_policy(policy)
            .with_load_shape(shape)
    }

    fn support(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> MemoryStrategySupport {
        contract
            .capability(strategy)
            .expect("all rungs declared")
            .support
            .clone()
    }

    #[test]
    fn registration_and_weights_free_surfaces_name_the_distilled_two_five_engine() {
        assert_eq!(
            LTX_2_5_DISTILLED_MEMORY_REGISTRATION.provider_id,
            LTX_2_5_DISTILLED_MODEL_ID
        );
        assert_eq!(
            LTX_2_5_DISTILLED_MEMORY_FIXTURE.provider_id,
            LTX_2_5_DISTILLED_MODEL_ID
        );
        assert_eq!(
            LTX_2_5_DISTILLED_MEMORY_BEHAVIOR.provider_id,
            LTX_2_5_DISTILLED_MODEL_ID
        );
        assert_ne!(
            LTX_2_5_DISTILLED_MEMORY_REGISTRATION.provider_id,
            crate::MODEL_ID
        );
        let surfaces = (LTX_2_5_DISTILLED_MEMORY_FIXTURE.surface_specs)();
        let mut tiers: Vec<_> = surfaces
            .iter()
            .map(|surface| surface.selector.tier)
            .collect();
        tiers.sort();
        tiers.dedup();
        assert_eq!(
            tiers,
            vec![
                gen_core::MemoryContractSurfaceTier::Bf16,
                gen_core::MemoryContractSurfaceTier::Q4,
                gen_core::MemoryContractSurfaceTier::Q8,
            ],
            "every released tier, and only released tiers, carries a weights-free surface"
        );
        assert_eq!(
            surfaces.len(),
            12,
            "3 tiers x resident/sequential x eager/deferred"
        );
        // sc-18791: the shipped q8 tier is a first-class Candle route, so its four rungs must be
        // present rather than filtered out of the weights-free surface.
        let q8: Vec<_> = surfaces
            .iter()
            .filter(|surface| surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q8)
            .collect();
        assert_eq!(q8.len(), 4, "q8 x resident/sequential x eager/deferred");
        for surface in &q8 {
            // Advertising the rung is not reaching it: each q8 surface must build its contract and
            // resolve to the q8 numeric tier the registered safety check re-derives.
            weights_free_contract(&surface.spec)
                .expect("every advertised q8 surface builds its weights-free contract");
            assert_eq!(
                resolved_numeric_tier(&surface.spec).unwrap().quant,
                Some(Quant::Q8)
            );
        }
    }

    #[test]
    fn production_contract_binds_physical_component_and_adapter_residency() {
        let directory = tempfile::tempdir().unwrap();
        let mut load = physical_spec(directory.path(), false, true);
        let adapter = directory.path().join("adapter.safetensors");
        write_safetensors(&adapter, &[], 41);
        load.adapters
            .push(AdapterSpec::new(adapter, 0.75, AdapterKind::Lora));

        let contract = memory_strategy_contract(&load).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 26 + 86 + 148 + 92);
        assert_eq!(contract.asset_facts.transformer_bytes, 22 + 116 + 124);
        assert_eq!(contract.asset_facts.decoder_bytes, 68 + 76);
        assert_eq!(contract.asset_facts.base_bytes, 352 + 262 + 144);
        assert_eq!(contract.asset_facts.overlay_bytes, 82);
        assert_eq!(
            weights_free_contract(&load).unwrap().asset_facts,
            MemoryAssetFacts::default()
        );

        let overlay = gen_core::adapter_stack_identity(&load.adapters).unwrap();
        let context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedAttention,
            resolved_numeric_tier(&load).unwrap(),
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("text_to_video".into()),
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: Some(overlay.clone()),
            },
        )
        .unwrap();
        assert_eq!(
            safety_check(
                &contract,
                resolved_numeric_tier(&load).unwrap(),
                Some(&overlay),
                &context,
            ),
            MemorySafetyDecision::Accept
        );
        assert_eq!(
            registered_safety_check(&load, &contract, &context),
            MemorySafetyDecision::Accept
        );
        let mut wrong_overlay = context;
        wrong_overlay.overlay = Some("adapters:wrong".into());
        assert!(matches!(
            safety_check(
                &contract,
                resolved_numeric_tier(&load).unwrap(),
                Some(&overlay),
                &wrong_overlay,
            ),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn physical_tier_manifest_must_match_the_requested_dense_or_q4_selector() {
        let directory = tempfile::tempdir().unwrap();
        write_tier_manifest(directory.path(), "q4", true, 4);
        let q4 =
            LoadSpec::new(WeightsSource::Dir(directory.path().to_path_buf())).with_quant(Quant::Q4);
        assert_eq!(resolved_numeric_tier(&q4).unwrap().quant, Some(Quant::Q4));
        assert!(resolved_numeric_tier(&LoadSpec::new(WeightsSource::Dir(
            directory.path().to_path_buf()
        )))
        .is_err());

        write_tier_manifest(directory.path(), "bf16", false, 16);
        let dense = LoadSpec::new(WeightsSource::Dir(directory.path().to_path_buf()));
        assert_eq!(resolved_numeric_tier(&dense).unwrap().quant, None);
        assert!(resolved_numeric_tier(&q4).is_err());
    }

    #[test]
    fn diffusion_vae_contract_does_not_require_a_convolutional_decoder() {
        let directory = tempfile::tempdir().unwrap();
        let load = physical_spec(directory.path(), true, false);
        assert!(!load
            .components
            .contains_key(LtxComponent::ConvVideoVae.id()));
        let contract = memory_strategy_contract(&load).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 26 + 86 + 92);
        assert_eq!(contract.asset_facts.decoder_bytes, 68 + 76);
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedDecode),
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn sequential_deferred_selects_windowing_but_resident_or_eager_does_not() {
        let sequential = weights_free_contract(&spec(
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        ))
        .unwrap();
        assert_eq!(
            support(&sequential, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Implemented
        );
        assert!(sequential.lifecycle.transformer_window_materialization);

        for unavailable in [
            spec(OffloadPolicy::Resident, LoadShape::DeferredMaterialization),
            spec(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
        ] {
            let contract = weights_free_contract(&unavailable).unwrap();
            assert_eq!(
                support(&contract, MemoryStrategy::BoundedTransformerResidency),
                MemoryStrategySupport::Missing
            );
            assert!(!contract.lifecycle.transformer_window_materialization);
        }
    }

    #[test]
    fn bounded_decode_is_exactly_the_convolutional_route() {
        let conv = spec(OffloadPolicy::Resident, LoadShape::EagerMaterialization);
        let conv_contract = weights_free_contract(&conv).unwrap();
        assert_eq!(
            support(&conv_contract, MemoryStrategy::BoundedDecode),
            MemoryStrategySupport::Implemented
        );
        assert!(conv_contract.lifecycle.decode_tiling);
        assert_eq!(
            conv_contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );

        let diff = conv.with_component(
            LtxComponent::DiffusionVideoVae.id(),
            WeightsSource::File("/nonexistent/diffvae.safetensors".into()),
        );
        let diff_contract = weights_free_contract(&diff).unwrap();
        assert_eq!(
            support(&diff_contract, MemoryStrategy::BoundedDecode),
            MemoryStrategySupport::Missing
        );
        assert!(!diff_contract.lifecycle.decode_tiling);
        assert!(diff_contract
            .capability(MemoryStrategy::BoundedDecode)
            .unwrap()
            .parameters
            .decode_tile_edges
            .is_empty());
    }

    #[test]
    fn shared_scope_threads_attention_and_window_into_request_memory() {
        let load = spec(
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        );
        let contract = weights_free_contract(&load).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
            resolved_numeric_tier(&load).unwrap(),
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("text_to_video".into()),
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        context.geometry.width = 512;
        context.geometry.height = 512;
        context.geometry.frames = 17;
        let mut scope = begin(
            &contract,
            resolved_numeric_tier(&load).unwrap(),
            None,
            Device::Cpu,
            &context,
        )
        .unwrap()
        .unwrap();
        let mut request = GenerationRequest {
            prompt: "scope".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let execution = transformer_execution(&request, Some(&contract), false).unwrap();
        assert_eq!(
            execution.attention_chunk_size,
            context.selection.parameters.attention_chunk_size
        );
        assert_eq!(
            execution.transformer_window_size,
            context.selection.parameters.transformer_window_size
        );
        assert!(execution.is_streamed());
    }

    #[test]
    fn direct_requests_cannot_smuggle_unpublished_transformer_or_decode_controls() {
        let load = spec(
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        );
        let contract = weights_free_contract(&load).unwrap();
        let mut request = GenerationRequest {
            memory: Some(gen_core::GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(123),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(transformer_execution(&request, Some(&contract), false).is_err());

        request.memory = Some(gen_core::GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(1),
            transformer_window_component: Some(TransformerComponent::TextEncoder),
            ..Default::default()
        });
        assert!(transformer_execution(&request, Some(&contract), false).is_err());

        request.memory = Some(gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE_EDGES[0]),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(transformer_execution(&request, Some(&contract), true).is_err());
        assert!(selected_decode_cap(&request, true).is_err());
    }

    #[test]
    fn registered_safety_is_contract_and_tier_exact() {
        let load = spec(
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        );
        let contract = weights_free_contract(&load).unwrap();
        let fixtures = registered_valid_fixtures(
            &load,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap();
        let context = &fixtures[0].context;
        assert_eq!(
            safety_check(
                &contract,
                resolved_numeric_tier(&load).unwrap(),
                None,
                context,
            ),
            MemorySafetyDecision::Accept
        );
        assert_eq!(
            registered_safety_check(&load, &contract, context),
            MemorySafetyDecision::Accept
        );
        let mut wrong_tier = context.clone();
        wrong_tier.selection.tier.quant = None;
        assert!(matches!(
            safety_check(
                &contract,
                resolved_numeric_tier(&load).unwrap(),
                None,
                &wrong_tier,
            ),
            MemorySafetyDecision::Reject { .. }
        ));
        let mut wrong_contract = contract.clone();
        wrong_contract.provider_id.push_str("-mutated");
        assert!(matches!(
            registered_safety_check(&load, &wrong_contract, context),
            MemorySafetyDecision::Reject { .. }
        ));
        assert!(registered_begin_request(&load, &wrong_contract, context).is_err());
    }

    #[test]
    fn every_declared_surface_conforms_and_behavior_is_executable() {
        for surface in (LTX_2_5_DISTILLED_MEMORY_FIXTURE.surface_specs)() {
            let contract = (LTX_2_5_DISTILLED_MEMORY_FIXTURE.contract)(&surface.spec).unwrap();
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        }
        let registry = crate::register_memory_contract_surfaces(
            gen_core::ProviderRegistryBuilder::new()
                .register_generator(crate::REGISTRATION)
                .register_generator(crate::REGISTRATION_25),
        )
        .build()
        .unwrap();
        gen_core_testkit::memory_contract_surface_registry_conformance(&registry);
        let generic = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &generic);
    }

    #[test]
    fn published_parameters_are_non_degenerate() {
        for pair in TRANSFORMER_WINDOW_SIZES.windows(2) {
            assert_eq!(pair[1], pair[0] * 2);
        }
        for &window in TRANSFORMER_WINDOW_SIZES {
            let plan = gen_core::block_window::BlockPlan::new(48, window as usize).unwrap();
            assert!(plan.is_bounded());
            assert_eq!(plan.window_count(), 48 / window as usize);
        }
        assert!(ATTENTION_CHUNK_SIZES
            .windows(2)
            .all(|pair| pair[0] > pair[1]));
    }
}
