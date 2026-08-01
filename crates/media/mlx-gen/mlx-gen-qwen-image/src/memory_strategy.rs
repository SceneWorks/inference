//! Qwen-Image MLX shared memory-ladder contract (SC-15511, SC-16353).
//!
//! Rung 1 uses the existing staged `Residency` lifecycle; rung 2 drives the head-once/tail-tiled
//! Qwen VAE over its measured production tile ladder; rung 3 threads the shared MLX attention
//! planner through every one of the 60 joint-attention blocks; rung 4 uses the shared block-window
//! primitive from SC-16353. The provider contract is the only selector surface.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities,
    MemoryNumericTier, MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "qwen-image-mlx-shared-ladder-2026-08-01-v1-deferred";
pub const EAGER_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "qwen-image-mlx-shared-ladder-2026-08-01-v1-eager";

pub const fn memory_calibration_fingerprint(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => EAGER_MEMORY_CALIBRATION_FINGERPRINT,
        LoadShape::DeferredMaterialization => MEMORY_CALIBRATION_FINGERPRINT,
    }
}

/// Native Qwen-VAE production tile ladder in output pixels, measured against the exact untiled
/// decode on the real bf16 VAE. SC-15511's same-process Metal A/B found overlap 96 increased the
/// incremental active peak by 12 MiB at both 448- and 256-pixel tiles versus overlap 64, so only 64
/// is shipped. A candidate is not a production range merely because it changes seam blending.
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 448, 384, 320, 256];
pub const DECODE_OVERLAP: u32 = 64;
pub const REJECTED_SUB_512_OVERLAP: u32 = 96;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
    .map_err(|errors| CoreError::Unsupported(errors.join("; ")))
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
pub const TRANSFORMER_BLOCKS: u32 = 60;

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
    Ok(Some(TRANSFORMER_BLOCKS as usize))
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    let streamable = is_streamable_spec(spec);
    // The optional control route owns a separate five-block attention branch which is not yet
    // windowed by the shared block loader.  It may use the native tiled VAE, but must not inherit
    // the base/edit route's rung-3 or rung-4 claims merely because those providers share a crate.
    let has_unbounded_control_branch = provider_id == "qwen_image_control";
    let footprint = crate::model::component_footprint(spec)?;
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
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
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
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        memory_calibration_fingerprint(spec.load_shape),
    ));
    let overlay_bytes = spec.control.as_ref().and_then(source_bytes).unwrap_or(0);
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae)
        .saturating_add(overlay_bytes);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
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
            decode_tile_edges: routes.native_edges().to_vec(),
            decode_overlaps: vec![DECODE_OVERLAP],
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

    if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
        let staged = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("compatibility contract contains every strategy");
        staged.support = MemoryStrategySupport::Implemented;
    }
    if streamable && !has_unbounded_control_branch {
        let plan = mlx_gen::block_residency::BlockPlan::new(
            TRANSFORMER_BLOCKS as usize,
            TRANSFORMER_WINDOW_SIZE as usize,
        )?;
        if !plan.is_bounded() {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: published transformer window {} does not bound the {}-block stack",
                TRANSFORMER_WINDOW_SIZE, TRANSFORMER_BLOCKS
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

fn source_bytes(source: &WeightsSource) -> Option<u64> {
    match source {
        WeightsSource::File(path) => std::fs::metadata(path).ok().map(|metadata| metadata.len()),
        WeightsSource::Dir(path) => std::fs::read_dir(path).ok().map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| entry.metadata().ok())
                .filter(|metadata| metadata.is_file())
                .fold(0_u64, |sum, metadata| sum.saturating_add(metadata.len()))
        }),
    }
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

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(QwenMemoryScope {
        provider_id,
        decode_routes: decode_routes(provider_id)?,
        memory: qwen_generation_memory(contract, &context.selection),
        geometry: context.geometry,
        use_pid: context.use_pid,
        transformer_window: contract
            .engages(
                context.selection.strategy,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .then_some(context.selection.parameters.transformer_window_size)
            .flatten(),
        finished: false,
    })))
}

pub(crate) fn qwen_generation_memory(
    contract: &MemoryProviderContract,
    selection: &mlx_gen::gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    if selection.strategy == MemoryStrategy::Resident {
        return None;
    }
    let parameters = selection.parameters;
    let stage_residency = contract.engages(selection.strategy, MemoryStrategy::StagedResidency);
    let tile_vae_decode = contract.engages(selection.strategy, MemoryStrategy::BoundedDecode);
    let chunk_attention = contract.engages(selection.strategy, MemoryStrategy::BoundedAttention);
    let stream_transformer_blocks = contract.engages(
        selection.strategy,
        MemoryStrategy::BoundedTransformerResidency,
    );
    Some(GenerationMemory {
        stage_residency,
        tile_vae_decode,
        decode_tile_edge: tile_vae_decode
            .then_some(parameters.decode_tile_edge)
            .flatten(),
        decode_overlap: tile_vae_decode
            .then_some(parameters.decode_overlap)
            .flatten(),
        chunk_attention,
        stream_transformer_blocks,
        transformer_window_size: stream_transformer_blocks
            .then_some(parameters.transformer_window_size)
            .flatten(),
        transformer_window_component: stream_transformer_blocks
            .then(|| parameters.window_component()),
        ..Default::default()
    })
}

struct QwenMemoryScope {
    provider_id: &'static str,
    decode_routes: mlx_gen_pid::DecodeRoutes,
    memory: Option<GenerationMemory>,
    geometry: MemoryGeometry,
    use_pid: bool,
    transformer_window: Option<u32>,
    finished: bool,
}

impl QwenMemoryScope {
    fn finish_inner(&mut self) -> CoreResult<()> {
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl MemoryRequestScope for QwenMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        if request.use_pid != self.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: request use_pid={} does not match admitted use_pid={}",
                self.provider_id, request.use_pid, self.use_pid
            )));
        }
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "{}: request geometry {}x{}x{} exceeds admitted {}x{}x{}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        Ok(())
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        _geometry: MemoryGeometry,
    ) -> CoreResult<()> {
        self.decode_routes
            .validate(self.use_pid, Some(tile_edge), Some(overlap))
            .map_err(CoreError::Unsupported)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> CoreResult<()> {
        if chunk_size == ATTENTION_CHUNK_SIZE {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "{}: attention chunk size must be {ATTENTION_CHUNK_SIZE}, got {chunk_size}",
                self.provider_id
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        if self.transformer_window != Some(TRANSFORMER_WINDOW_SIZE)
            || block_count != TRANSFORMER_WINDOW_SIZE
            || first_block >= TRANSFORMER_BLOCKS
        {
            return Err(CoreError::Unsupported(format!(
                "{}: invalid DiT window ({first_block}, {block_count})",
                self.provider_id
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.finish_inner()
    }
}

impl Drop for QwenMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryNumericTier, MemorySelection, MemoryStrategyParameters, MemoryStrategySupport,
    };

    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/qwen".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    #[test]
    fn sequential_deferred_directory_declares_the_exact_dit_window() {
        let contract = memory_strategy_contract("qwen_image", &spec()).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        assert_eq!(decode.parameters.decode_tile_edges, DECODE_TILE_EDGES);
        assert_eq!(decode.parameters.decode_overlaps, vec![DECODE_OVERLAP]);
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
        let contract = memory_strategy_contract("qwen_image_control", &spec()).unwrap();
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
        let contract = memory_strategy_contract("qwen_image", &spec()).unwrap();
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
        let contract = memory_strategy_contract("qwen_image", &spec()).unwrap();
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
        let mut eager = spec();
        eager.load_shape = LoadShape::EagerMaterialization;
        let mut resident = spec();
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
        let root =
            std::env::temp_dir().join(format!("qwen-rung4-quant-contract-{}", std::process::id()));
        std::fs::create_dir_all(root.join("transformer")).unwrap();
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
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn single_file_source_is_rejected_by_the_family_contract() {
        let mut file = spec();
        file.weights = WeightsSource::File("/nonexistent/qwen.safetensors".into());
        assert!(memory_strategy_contract("qwen_image", &file).is_err());
    }

    #[test]
    fn rung_four_rejects_an_unpublished_window() {
        let contract = memory_strategy_contract("qwen_image", &spec()).unwrap();
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
