//! Qwen-Image MLX shared memory-ladder contract (SC-15511, SC-16353).
//!
//! Rung 1 uses the existing staged `Residency` lifecycle; rung 2 drives the head-once/tail-tiled
//! Qwen VAE over its measured production tile ladder; rung 3 threads the shared MLX attention
//! planner through every one of the 60 joint-attention blocks; rung 4 uses the shared block-window
//! primitive from SC-16353. The provider contract is the only selector surface.

use mlx_gen::asset_facts::{
    projected_safetensors_bytes, projected_tensor_headers_bytes, ResidentProjection,
};
#[cfg(test)]
use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport, Result as CoreResult,
    TransformerComponent,
};
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

/// Load shape is a typed evidence-key axis; this content fingerprint remains shape-independent.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "qwen-image-mlx-shared-ladder-2026-08-01-v1";

/// Native Qwen-VAE production tile ladder in output pixels, measured against the exact untiled
/// decode on the real bf16 VAE. SC-15511's same-process Metal A/B found overlap 96 increased the
/// incremental active peak by 12 MiB at both 448- and 256-pixel tiles versus overlap 64, so only 64
/// is shipped. A candidate is not a production range merely because it changes seam blending.
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 448, 384, 320, 256];
pub const DECODE_OVERLAP: u32 = 64;
pub const REJECTED_SUB_512_OVERLAP: u32 = 96;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

/// The shared 64-Mi score-element budget used by the MLX rung-3 kernel.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
/// SC-16353 measured 1/2/4/8 plus the unbounded 60-block control at 1024² across Q4/Q8/BF16.
/// Only window 1 materially lowered the denoise counter versus the same-stream unbounded control;
/// windows above 1 were indistinguishable noise or worse and never moved the conditioning-bound
/// request peak. The all-covering 60-block arm is deliberately not publishable:
/// [`mlx_gen::block_residency::BlockPlan::is_bounded`] defines it as fully resident.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[TRANSFORMER_WINDOW_SIZE];

fn transformer_blocks(provider_id: &str) -> usize {
    if provider_id == crate::model_edit::MODEL_ID {
        crate::transformer::QwenTransformerConfig::qwen_image_edit().num_layers
    } else {
        crate::transformer::QwenTransformerConfig::qwen_image().num_layers
    }
}

pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    let matching_device_format = match (&spec.weights, spec.quantize) {
        (WeightsSource::Dir(root), None) => {
            matches!(
                mlx_gen::quant::packed_quant_bits(root, "transformer"),
                Ok(None)
            )
        }
        (WeightsSource::Dir(root), Some(quant)) => matches!(
            mlx_gen::quant::packed_quant_bits(root, "transformer"),
            Ok(Some(bits)) if bits == quant.bits()
        ),
        (WeightsSource::File(_), _) => false,
    };
    matching_device_format
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.precision, Precision::Bf16)
        && spec.pid.is_none()
}

/// Whether a provider may arm the Qwen transformer block stream at load time. Base and Edit own a
/// published transformer-window lifecycle; Control does not, because its separate five-block
/// branch remains unbounded.
pub(crate) fn should_arm_block_stream(provider_id: &str, spec: &LoadSpec) -> bool {
    provider_id != crate::model_control::MODEL_ID && is_streamable_spec(spec)
}

/// Resolve the physical DiT cadence for a request. Eligible deferred loads always use the stream.
/// SC-16353 found that reopening the view more often lowers the denoise-only counter but never the
/// whole-request peak at 1024². The published cadence is nevertheless genuinely bounded; the
/// all-covering plan remains an attribution control, never a selectable rung-4 parameter.
pub(crate) fn resolve_window_size(
    request: &GenerationRequest,
    contract: &MemoryProviderContract,
) -> mlx_gen::Result<Option<usize>> {
    let requested = request
        .memory
        .is_some_and(|memory| memory.stream_transformer_blocks);
    if requested && !contract.lifecycle.transformer_window_materialization {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{}: bounded transformer residency requires Sequential + DeferredMaterialization + directory weights",
            contract.provider_id
        )));
    }
    if !contract.lifecycle.transformer_window_materialization {
        return Ok(None);
    }
    if requested {
        let memory = request.memory.expect("requested checked");
        if memory.transformer_window_component.unwrap_or_default() != TransformerComponent::Dit {
            return Err(mlx_gen::Error::Unsupported(format!(
                "{}: only the DiT transformer window is implemented",
                contract.provider_id
            )));
        }
        return Ok(Some(
            memory
                .transformer_window_size
                .unwrap_or(TRANSFORMER_WINDOW_SIZE) as usize,
        ));
    }
    Ok(Some(transformer_blocks(&contract.provider_id)))
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let _ = crate::model::component_footprint(spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "qwen-image memory facts require a snapshot directory".to_owned(),
        ));
    };
    let project = |path: &std::path::Path, quant: Option<mlx_gen::Quant>| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |_| match quant {
            Some(quant) => ResidentProjection::GroupQuantized {
                bits: quant.bits(),
                group_size: crate::quant::GROUP_SIZE as usize,
            },
            None => ResidentProjection::Stored,
        })
    };
    let selected_text_encoder = crate::ENCODER_CONTRACT.source_for_load(spec, root)?;
    let conditioning_bytes =
        projected_tensor_headers_bytes(&selected_text_encoder.tensor_headers()?, |_| {
            ResidentProjection::Stored
        })?;
    let transformer_bytes = project(&root.join("transformer"), spec.quantize)?;
    let decoder_bytes = project(&root.join("vae"), None)?;
    let overlay_bytes = match &spec.control {
        Some(WeightsSource::Dir(path)) | Some(WeightsSource::File(path)) => {
            projected_safetensors_bytes(path, |_| match spec.quantize {
                Some(quant) => ResidentProjection::GroupQuantized {
                    bits: quant.bits(),
                    group_size: crate::quant::GROUP_SIZE as usize,
                },
                None => ResidentProjection::Stored,
            })?
        }
        None => 0,
    };
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes,
    )
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    memory_strategy_contract_with_asset_facts(provider_id, spec, 0, 0, 0, 0)
}

fn memory_strategy_contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    overlay_bytes: u64,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    let streamable = is_streamable_spec(spec);
    // The optional control route owns a separate five-block attention branch which is not yet
    // windowed by the shared block loader.  It may use the native tiled VAE, but must not inherit
    // the base/edit route's rung-3 or rung-4 claims merely because those providers share a crate.
    let has_unbounded_control_branch = provider_id == "qwen_image_control";
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    contract.formula = if overlay_bytes > 0 {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: vec![MemoryResidentComponent {
                id: "control_branch".to_owned(),
                kind: MemoryComponentKind::ControlBranch,
                resident_bytes: overlay_bytes,
                bounded_by: None,
            }],
        }
    } else {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    contract.asset_facts.overlay_bytes = overlay_bytes;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        decode_tiling: true,
        attention_chunking: !has_unbounded_control_branch,
        transformer_window_materialization: streamable && !has_unbounded_control_branch,
    };

    let mut implemented_scratch = vec![(
        MemoryStrategy::BoundedDecode,
        MemoryParameterRanges {
            decode_tile_edges: routes.published_edges(),
            decode_overlaps: routes.published_overlaps(),
            ..Default::default()
        },
    )];
    if !has_unbounded_control_branch {
        implemented_scratch.push((
            MemoryStrategy::BoundedAttention,
            MemoryParameterRanges {
                attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                ..Default::default()
            },
        ));
    }
    for (strategy, parameters) in implemented_scratch {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == strategy)
            .expect("compatibility contract contains every strategy");
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters = parameters;
    }
    contract.pid_decode_routes = Some(mlx_gen::gen_core::MemoryPidDecodeRoutes {
        native: mlx_gen::gen_core::MemoryDecodeRouteDomain {
            tile_edges: routes.native_edges().to_vec(),
            tile_overlap: DECODE_OVERLAP,
        },
        pid: mlx_gen::gen_core::MemoryDecodeRouteDomain {
            tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
            tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
        },
    });

    if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
        let staged = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("compatibility contract contains every strategy");
        staged.support = MemoryStrategySupport::Implemented;
    }
    if streamable && !has_unbounded_control_branch {
        let transformer_blocks = transformer_blocks(provider_id);
        let plan = mlx_gen::block_residency::BlockPlan::new(
            transformer_blocks,
            TRANSFORMER_WINDOW_SIZE as usize,
        )?;
        if !plan.is_bounded() {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: published transformer window {} does not bound the {}-block stack",
                TRANSFORMER_WINDOW_SIZE, transformer_blocks
            )));
        }
        let rung = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains every strategy");
        rung.support = MemoryStrategySupport::Implemented;
        rung.parameters.transformer_window_sizes = TRANSFORMER_WINDOW_SIZES.to_vec();
        rung.parameters.transformer_window_components = vec![TransformerComponent::Dit];
    }
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision,
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
    safety_check(contract, spec.precision, spec.quantize, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let (mode, has_reference) = if contract.provider_id.ends_with("_edit") {
        (mlx_gen::gen_core::MemoryMode::Edit, true)
    } else if contract.provider_id.ends_with("_control") {
        (mlx_gen::gen_core::MemoryMode::ImageToImage, true)
    } else {
        (mlx_gen::gen_core::MemoryMode::TextToImage, false)
    };
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let route = |use_pid| mlx_gen::gen_core::MemoryBehaviorRoute {
        mode: mode.clone(),
        reference_count: u32::from(has_reference),
        use_pid,
        has_phases: false,
        overlay: None,
    };
    let mut fixtures = vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            route(false),
        )?,
    )];
    if contract.engages(strategy, MemoryStrategy::BoundedDecode) {
        fixtures.push(mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                route(true),
            )?,
        ));
    }
    Ok(fixtures)
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        spec.quantize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

#[cfg(test)]
fn qwen_generation_memory(
    contract: &MemoryProviderContract,
    selection: &mlx_gen::gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    contract.generation_memory(selection)
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        precision,
        quant,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let transformer_blocks = transformer_blocks(provider_id);
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryNumericTier, MemorySelection, MemoryStrategyParameters, MemoryStrategySupport,
    };

    fn write_snapshot(root: &std::path::Path) {
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_control(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
    }

    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("qwen-memory-spec");
        write_snapshot(&root);
        LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn write_control(path: &std::path::Path) {
        let mut header =
            br#"{"control.weight":{"dtype":"BF16","shape":[2,64],"data_offsets":[0,256]}}"#
                .to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 256]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn empty_required_component_directory_cannot_be_reported_as_zero() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::remove_file(root.join("transformer/model.safetensors")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(memory_strategy_contract("qwen_image", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("qwen_image", &spec).is_ok());
    }

    #[test]
    fn sequential_deferred_directory_declares_the_exact_dit_window() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        let routes = decode_routes("qwen_image").unwrap();
        assert_eq!(
            decode.parameters.decode_tile_edges,
            routes.published_edges()
        );
        assert_eq!(
            decode.parameters.decode_overlaps,
            routes.published_overlaps()
        );
        let declared_routes = contract.pid_decode_routes.as_ref().unwrap();
        assert_eq!(declared_routes.native.tile_edges, DECODE_TILE_EDGES);
        assert_eq!(
            declared_routes.pid.tile_edges,
            mlx_gen_pid::DecodeRoutes::pid_edges()
        );
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung.parameters.transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
        assert_eq!(
            rung.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
        assert!(contract.engages(
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategy::StagedResidency
        ));
    }

    #[test]
    fn selected_encoder_pricing_ignores_nested_safetensors_not_loaded_as_shards() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp);
        let before = memory_strategy_contract("qwen_image", &spec)
            .unwrap()
            .asset_facts
            .conditioning_bytes;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        let nested = root.join("text_encoder/archive");
        std::fs::create_dir_all(&nested).unwrap();
        write_control(&nested.join("ignored.safetensors"));

        let after = memory_strategy_contract("qwen_image", &spec)
            .unwrap()
            .asset_facts
            .conditioning_bytes;
        assert_eq!(after, before);
    }

    #[test]
    fn block_stream_load_shape_predicate_is_exact_and_excludes_control() {
        let tmp = tempfile::tempdir().unwrap();
        let eligible = spec(&tmp);
        assert!(is_streamable_spec(&eligible));
        assert!(should_arm_block_stream(crate::model::MODEL_ID, &eligible));
        assert!(should_arm_block_stream(
            crate::model_edit::MODEL_ID,
            &eligible
        ));
        assert!(!should_arm_block_stream(
            crate::model_control::MODEL_ID,
            &eligible
        ));

        let mut eager = eligible.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        let mut resident = eligible.clone();
        resident.offload_policy = OffloadPolicy::Resident;
        let mut file = eligible.clone();
        file.weights = WeightsSource::File("transformer.safetensors".into());
        let mut load_time_quant = eligible;
        load_time_quant.quantize = Some(mlx_gen::Quant::Q4);
        for ineligible in [eager, resident, file, load_time_quant] {
            assert!(!is_streamable_spec(&ineligible));
            assert!(!should_arm_block_stream(
                crate::model::MODEL_ID,
                &ineligible
            ));
        }
    }

    #[test]
    fn checked_decode_routes_keep_native_geometry_out_of_pid_requests() {
        let routes = decode_routes("qwen_image").unwrap();
        assert_eq!(routes.native_edges(), DECODE_TILE_EDGES);
        routes
            .validate(false, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap();
        let error = routes
            .validate(true, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap_err();
        assert!(error.contains("PiD overlay"));
        assert!(error.contains("not a"));
    }

    #[test]
    fn control_route_does_not_overstate_its_unbounded_side_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = memory_strategy_contract("qwen_image_control", &spec(&tmp)).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "the separate five-block control branch is not yet bounded"
            );
        }
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn control_overlay_is_quant_projected_typed_and_excluded_from_base() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let control = root.join("control.safetensors");
        write_control(&control);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(mlx_gen::Quant::Q4)
            .with_control(WeightsSource::File(control));
        let contract = memory_strategy_contract("qwen_image_control", &spec).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 14_141_904_896);
        assert_eq!(contract.asset_facts.transformer_bytes, 72);
        assert_eq!(contract.asset_facts.decoder_bytes, 256);
        assert_eq!(contract.asset_facts.base_bytes, 14_141_905_224);
        assert_eq!(contract.asset_facts.overlay_bytes, 72);
        assert_eq!(contract.auxiliary_resident_bytes(), 72);
        assert!(matches!(
            contract.formula,
            MemoryFormulaKind::ComponentPhaseEnvelope { .. }
        ));
        assert!(contract.conformance_errors().is_empty());
    }

    fn selection(strategy: MemoryStrategy) -> MemorySelection {
        let mut parameters = MemoryStrategyParameters::default();
        if matches!(
            strategy,
            MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency
        ) {
            parameters.decode_tile_edge = Some(DECODE_TILE_EDGE);
            parameters.decode_overlap = Some(DECODE_OVERLAP);
        }
        if matches!(
            strategy,
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency
        ) {
            parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
        }
        if strategy == MemoryStrategy::BoundedTransformerResidency {
            parameters.transformer_window_size = Some(TRANSFORMER_WINDOW_SIZE);
            parameters.transformer_window_component = Some(TransformerComponent::Dit);
        }
        MemorySelection {
            strategy,
            parameters,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        }
    }

    #[test]
    fn selections_translate_to_the_shared_cumulative_request_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let resident = qwen_generation_memory(&contract, &selection(MemoryStrategy::Resident));
        assert_eq!(resident, None);

        let staged =
            qwen_generation_memory(&contract, &selection(MemoryStrategy::StagedResidency)).unwrap();
        assert!(staged.stage_residency);
        assert!(!staged.tile_vae_decode);
        assert!(!staged.chunk_attention);
        assert!(!staged.stream_transformer_blocks);

        let decode =
            qwen_generation_memory(&contract, &selection(MemoryStrategy::BoundedDecode)).unwrap();
        assert!(!decode.stage_residency);
        assert!(decode.tile_vae_decode);
        assert_eq!(decode.decode_tile_edge, Some(DECODE_TILE_EDGE));
        assert_eq!(decode.decode_overlap, Some(DECODE_OVERLAP));
        assert!(!decode.chunk_attention);

        let attention =
            qwen_generation_memory(&contract, &selection(MemoryStrategy::BoundedAttention))
                .unwrap();
        assert!(attention.tile_vae_decode);
        assert!(attention.chunk_attention);
        assert!(!attention.stage_residency);

        let streamed = qwen_generation_memory(
            &contract,
            &selection(MemoryStrategy::BoundedTransformerResidency),
        )
        .unwrap();
        assert!(
            streamed.stage_residency,
            "Qwen rung 4 explicitly requires rung 1"
        );
        assert!(streamed.tile_vae_decode);
        assert!(streamed.chunk_attention);
        assert!(streamed.stream_transformer_blocks);
        assert_eq!(
            streamed.transformer_window_component,
            Some(TransformerComponent::Dit)
        );
    }

    #[test]
    fn unpublished_parameters_are_rejected_instead_of_silently_coerced() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let mut decode = selection(MemoryStrategy::BoundedDecode);
        assert!(
            contract.validate_selection(&decode).is_ok(),
            "{:?}",
            contract.validate_selection(&decode)
        );
        decode.parameters.decode_overlap = Some(REJECTED_SUB_512_OVERLAP);
        assert!(contract.validate_selection(&decode).is_err());

        let mut attention = selection(MemoryStrategy::BoundedAttention);
        assert!(
            contract.validate_selection(&attention).is_ok(),
            "{:?}",
            contract.validate_selection(&attention)
        );
        attention.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE / 2);
        assert!(contract.validate_selection(&attention).is_err());
    }

    #[test]
    fn eager_and_resident_loads_do_not_advertise_rung_four() {
        let tmp = tempfile::tempdir().unwrap();
        let deferred_contract = memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let mut eager = spec(&tmp);
        eager.load_shape = LoadShape::EagerMaterialization;
        let eager_contract = memory_strategy_contract("qwen_image", &eager).unwrap();
        assert_eq!(
            deferred_contract.calibration.as_ref().unwrap().fingerprint,
            eager_contract.calibration.as_ref().unwrap().fingerprint
        );
        assert_ne!(
            deferred_contract.calibration.as_ref().unwrap().load_shape,
            eager_contract.calibration.as_ref().unwrap().load_shape
        );
        let mut resident = spec(&tmp);
        resident.offload_policy = OffloadPolicy::Resident;
        for spec in [eager, resident] {
            let contract = memory_strategy_contract("qwen_image", &spec).unwrap();
            assert!(matches!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|c| &c.support),
                Some(MemoryStrategySupport::Missing)
            ));
            assert!(!contract.lifecycle.transformer_window_materialization);
        }
    }

    #[test]
    fn dense_load_time_quantization_does_not_advertise_rung_four() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"dtype":"bfloat16"}"#,
        )
        .unwrap();
        let dense_q8 = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(mlx_gen::Quant::Q8)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = memory_strategy_contract("qwen_image", &dense_q8).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "per-window dense-to-Q8 conversion is not a device-format transfer"
        );

        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8}}"#,
        )
        .unwrap();
        let contract = memory_strategy_contract("qwen_image", &dense_q8).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
    }

    #[test]
    fn single_file_source_is_rejected_by_the_family_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let mut file = spec(&tmp);
        file.weights = WeightsSource::File("/nonexistent/qwen.safetensors".into());
        assert!(memory_strategy_contract("qwen_image", &file).is_err());
    }

    #[test]
    fn rung_four_rejects_an_unpublished_window() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = memory_strategy_contract("qwen_image", &spec(&tmp)).unwrap();
        let mut selection = selection(MemoryStrategy::BoundedTransformerResidency);
        assert!(
            contract.validate_selection(&selection).is_ok(),
            "{:?}",
            contract.validate_selection(&selection)
        );
        selection.parameters.transformer_window_size = Some(2);
        assert!(contract.validate_selection(&selection).is_err());
    }
}
