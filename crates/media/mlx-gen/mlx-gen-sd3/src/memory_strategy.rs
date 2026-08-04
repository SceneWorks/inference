//! Shared image-memory ladder for SD3.5 Large, Large Turbo, and Medium on MLX/Metal.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, OffloadPolicy};

use crate::config::{Sd3Variant, LARGE_NUM_LAYERS, MEDIUM_NUM_LAYERS};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sd3-5-large-bf16-mlx-shared-ladder-2026-08-03-v1";
pub const CALIBRATED_REVISION: &str = crate::artifact_inventory::CALIBRATED_REVISION;
const STATIC_CALIBRATION: &str = "sd3-5-mlx-registry-behavior-v1";
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

fn variant_for(provider_id: &str) -> mlx_gen::gen_core::Result<Sd3Variant> {
    match provider_id {
        crate::MODEL_ID => Ok(Sd3Variant::Large),
        crate::TURBO_MODEL_ID => Ok(Sd3Variant::LargeTurbo),
        crate::MEDIUM_MODEL_ID => Ok(Sd3Variant::Medium),
        _ => Err(CoreError::Unsupported(format!(
            "unknown SD3.5 memory provider {provider_id}"
        ))),
    }
}

fn provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    Ok(variant_for(provider_id)?.id())
}

fn clean(spec: &LoadSpec) -> bool {
    spec.adapters.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
}

pub(crate) fn structurally_streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && clean(spec)
        && matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    calibrated: bool,
    calibration: Option<MemoryCalibrationIdentity>,
    footprint: mlx_gen::PerComponentBytes,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    variant_for(provider_id)?;
    let staged = spec.offload_policy == OffloadPolicy::Sequential;
    let clean = clean(spec);
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
    contract.calibration = calibration;
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
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: staged,
        decode_tiling: clean,
        attention_chunking: clean,
        transformer_window_materialization: streamable,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged && calibrated => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode if clean && calibrated => {
                capability.parameters.decode_tile_edges = DECODE_TILE_EDGES.to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if clean && calibrated => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Dit];
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(contract)
}

pub(crate) fn contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    // The implementation topology is shared, but production admission is evidence-bound. Only the
    // exact Large artifact is present in this cache and calibrated by SC-15522; Turbo and Medium
    // therefore remain resident-only until their own exact artifacts and per-entry evidence exist.
    let exact_large = provider_id == crate::MODEL_ID
        && crate::artifact_inventory::Sd3ArtifactInventory::verify(spec)
            .map_err(mlx_gen::gen_core::Error::from)?
            .is_some();
    let streamable = exact_large && structurally_streamable(spec);
    let calibration = exact_large
        .then(|| MemoryCalibrationIdentity::new(MEMORY_CALIBRATION_FINGERPRINT, spec.load_shape));
    build_contract(
        provider_id,
        spec,
        streamable,
        exact_large,
        calibration,
        crate::model::component_footprint(spec)?,
    )
}

pub(crate) fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let calibrated = provider_id == crate::MODEL_ID;
    build_contract(
        provider_id,
        spec,
        calibrated && structurally_streamable(spec),
        calibrated,
        calibrated.then(|| {
            MemoryCalibrationIdentity::new(
                format!("{STATIC_CALIBRATION}-{}", provider_id.replace('_', "-")),
                spec.load_shape,
            )
        }),
        Default::default(),
    )
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let route_ok = matches!(context.mode, MemoryMode::TextToImage)
            && context.geometry.reference_count == 0
            || matches!(context.mode, MemoryMode::ImageToImage)
                && context.geometry.reference_count == 1;
        if !route_ok {
            return Err(CoreError::Unsupported(
                "sd3 memory route must be text-to-image or single-reference img2img".to_owned(),
            ));
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(
                "sd3 does not implement PiD decode".to_owned(),
            ));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let edge = context.selection.parameters.decode_tile_edge;
            let overlap = context.selection.parameters.decode_overlap;
            if !edge.is_some_and(|edge| DECODE_TILE_EDGES.contains(&edge))
                || overlap != Some(DECODE_OVERLAP)
            {
                return Err(CoreError::Unsupported(
                    "sd3 decode tile geometry is outside the calibrated domain".to_owned(),
                ));
            }
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) && (!context.has_phases || !contract.lifecycle.transformer_window_materialization)
        {
            return Err(CoreError::Unsupported(
                "sd3 transformer streaming requires Sequential + DeferredMaterialization and the exact calibrated Large artifact".to_owned(),
            ));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    if contract
        .capability(strategy)
        .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    [(MemoryMode::TextToImage, 0), (MemoryMode::ImageToImage, 1)]
        .into_iter()
        .map(|(mode, reference_count)| {
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                mlx_gen::gen_core::MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: spec.offload_policy == OffloadPolicy::Sequential,
                    overlay: None,
                },
            )
            .map(mlx_gen::gen_core::MemoryBehaviorFixture::new)
        })
        .collect()
}

fn begin_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let provider = provider_static(&contract.provider_id)?;
    let blocks = match variant_for(provider)? {
        Sd3Variant::Large | Sd3Variant::LargeTurbo => LARGE_NUM_LAYERS,
        Sd3Variant::Medium => MEDIUM_NUM_LAYERS,
    };
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        blocks,
        |_use_pid, edge, overlap| {
            if DECODE_TILE_EDGES.contains(&edge) && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(CoreError::Unsupported(
                    "sd3 decode tile geometry is outside the calibrated domain".to_owned(),
                ))
            }
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

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

pub(crate) fn transformer_window(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    if memory
        .transformer_window_component
        .unwrap_or(TransformerComponent::Dit)
        != TransformerComponent::Dit
    {
        return Err(mlx_gen::Error::Unsupported(
            "sd3 supports only the DiT transformer window component".to_owned(),
        ));
    }
    Ok(Some(
        memory
            .transformer_window_size
            .unwrap_or(TRANSFORMER_WINDOW_SIZE) as usize,
    ))
}

pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    Ok(Some(TilingConfig::spatial_only(
        memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE) as i32,
        memory.decode_overlap.unwrap_or(DECODE_OVERLAP) as i32,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;

    #[test]
    fn large_publishes_full_ladder_but_uncalibrated_entries_fail_closed() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = weights_free_contract(crate::MODEL_ID, &spec).unwrap();
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "Large must publish the representative full ladder"
            );
        }
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit]
        );
        assert!(contract.conformance_errors().is_empty());

        for provider in [crate::TURBO_MODEL_ID, crate::MEDIUM_MODEL_ID] {
            let contract = weights_free_contract(provider, &spec).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::Resident)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
            for strategy in MemoryStrategy::ALL
                .into_iter()
                .filter(|strategy| strategy.is_optimized())
            {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Missing,
                    "{provider} must not inherit Large's calibration without exact weights"
                );
            }
            assert!(contract.calibration.is_none());
            assert!(contract.conformance_errors().is_empty(), "{provider}");
        }
    }

    #[test]
    fn overlays_fail_closed_for_quality_sensitive_rungs() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_control(mlx_gen::WeightsSource::File("/nonexistent/control".into()));
        let contract = weights_free_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
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
}
