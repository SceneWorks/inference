//! Pure request geometry checks and weights-free lifecycle fixtures.
use super::*;
use candle_gen::gen_core::{
    MemoryBehaviorFixture, MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority,
    MemoryRequestScope,
};

pub(super) fn validate_decode(edge: u32, overlap: u32) -> candle_gen::gen_core::Result<()> {
    if edge != crate::spatial_tiling::TILE_SAMPLE_MIN_SIZE as u32
        || overlap != crate::spatial_tiling::TILE_SAMPLE_MIN_OVERLAP as u32
    {
        return Err(CoreError::Unsupported("minimax_h3: bounded decode requires reference tiles256/64 to preserve output semantics".into()));
    }
    Ok(())
}

pub(super) fn route_gate(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> candle_gen::gen_core::Result<()> {
    let g = context.geometry;
    let stride = crate::pipeline::SPATIAL_STRIDE;
    if context.use_pid
        || g.batch != 1
        || g.width == 0
        || g.height == 0
        || !g.width.is_multiple_of(stride)
        || !g.height.is_multiple_of(stride)
        || u64::from(g.width) * u64::from(g.height) > u64::from(crate::pipeline::CANVAS_MAX_PIXELS)
        || !crate::denoise::LEGAL_FRAME_COUNTS.contains(&(g.frames as usize))
    {
        return Err(CoreError::Unsupported(
            "minimax_h3: invalid joint audio/video geometry or unsupported PiD route".into(),
        ));
    }
    if let (Some(edge), Some(overlap)) = (
        context.selection.parameters.decode_tile_edge,
        context.selection.parameters.decode_overlap,
    ) {
        validate_decode(edge, overlap)?;
    }
    if let Some(elements) = context.selection.parameters.attention_chunk_size {
        // Necessary geometry-only lower bound. The product subsequently prices the complete
        // prompt/reference/source request; an unsupported minimum row must never be advertised
        // as a working bounded selection merely because its generic budget scalar fits.
        let request = candle_gen::gen_core::GenerationRequest {
            width: g.width,
            height: g.height,
            frames: Some(g.frames),
            count: g.batch,
            ..Default::default()
        };
        let (rows, _) = gen_core_h3::veda::workspace::h3_row_bound((&request).into())?;
        let heads = contract
            .architecture_facts
            .attention_heads
            .map_or(56, u64::from);
        if rows
            .checked_mul(heads)
            .is_none_or(|row| row > u64::from(elements))
        {
            return Err(CoreError::Unsupported(
                "minimax_h3: minimum materializing attention row exceeds selected workspace budget"
                    .into(),
            ));
        }
    }
    Ok(())
}

pub fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> candle_gen::gen_core::Result<Vec<MemoryBehaviorFixture>> {
    use candle_gen::gen_core::{MemoryBudget, MemoryCacheState, MemoryGeometry};
    if !strategy.is_optimized()
        || contract
            .capability(strategy)
            .is_none_or(|c| c.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    [
        (MemoryMode::TextToImage, 0),
        (MemoryMode::ImageToImage, 0),
        (MemoryMode::Other("ref2va".into()), 1),
    ]
    .into_iter()
    .map(|(mode, reference_count)| {
        let calibration = contract.calibration.as_ref();
        let mut selection = contract.representative_selection(strategy, tier, false)?;
        if selection.parameters.attention_chunk_size.is_some() {
            selection.parameters.attention_chunk_size = contract
                .capability(MemoryStrategy::BoundedAttention)
                .and_then(|cap| cap.parameters.attention_chunk_sizes.last().copied());
        }
        let context = MemoryRunContext {
            reference_resources: None,
            selection,
            optimization_authority: MemoryOptimizationAuthority::Estimated,
            calibration_abi: calibration.map_or(0, |c| c.abi),
            calibration_fingerprint: calibration
                .map_or_else(String::new, |c| c.fingerprint.clone()),
            load_shape: contract.load_shape,
            mode,
            has_reference: reference_count > 0,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 1344,
                height: 768,
                batch: 1,
                frames: 124,
                reference_count,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 8 << 30,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1 << 30,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "weights-free-h3-cuda-behavior".into(),
        };
        Ok(MemoryBehaviorFixture::new(context))
    })
    .collect()
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> candle_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    candle_gen::gen_core::standard_begin_request(
        &super::MEMORY_PROFILE,
        MODEL_ID,
        spec,
        contract,
        context,
        registered_scope,
    )
}

fn registered_scope(
    _provider_id: &'static str,
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _tier: MemoryNumericTier,
) -> candle_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    super::begin_request(contract, candle_gen::candle_core::Device::Cpu, context)
}

pub const MEMORY_BEHAVIOR: candle_gen::gen_core::MemoryBehaviorRegistration =
    candle_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_scope_applies_parameters_and_cleans_up_on_failure() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = super::super::weights_free_contract(&spec).unwrap();
        let fixtures = registered_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap();
        let context = &fixtures[0].context;
        let mut scope = registered_begin_request(&spec, &contract, context)
            .unwrap()
            .unwrap();
        let mut request = candle_gen::gen_core::GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(124),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(1));
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Both)
        );
        assert_eq!(super::super::active_context().as_ref(), Some(context));
        assert_eq!(crate::nn::attention_execution_report(), Default::default());
        drop(scope); // error/unwind path must clear the request-local selected policy too
        assert!(super::super::active_context().is_none());
        let next = registered_begin_request(&spec, &contract, context).unwrap();
        drop(next);
        assert!(super::super::active_context().is_none());
    }
}
