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

pub(super) fn route_gate(context: &MemoryRunContext) -> candle_gen::gen_core::Result<()> {
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
    };
    [
        (MemoryMode::TextToImage, 0),
        (MemoryMode::ImageToImage, 0),
        (MemoryMode::Other("ref2va".into()), 1),
    ]
    .into_iter()
    .map(|(mode, reference_count)| {
        let calibration = contract.calibration.as_ref();
        let context = MemoryRunContext {
            reference_resources: None,
            selection: contract.representative_selection(strategy, tier, false)?,
            optimization_authority: MemoryOptimizationAuthority::Estimated,
            calibration_abi: candle_gen::gen_core::MEMORY_CALIBRATION_ABI,
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
    candle_gen::request_scope::admit(super::safety_check(spec, contract, context))?;
    super::begin_request(contract, candle_gen::candle_core::Device::Cpu, context)
}

pub const MEMORY_BEHAVIOR: candle_gen::gen_core::MemoryBehaviorRegistration =
    candle_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_fixture,
        begin_request: registered_begin_request,
    };
