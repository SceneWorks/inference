//! Shared memory-strategy contract for base SANA and SANA-Sprint (SC-16783).
//!
//! The provider implements Resident, load-time staged residency, and the measured DC-AE tiled
//! decode ladder. Bounded attention remains Missing because only caption cross-attention has a
//! score tensor and its key axis is fixed at 300; transformer-window residency remains Missing
//! because this trunk has no block materialization driver.

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryDecodeRouteDomain,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPidDecodeRoutes, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, ResidentRequestMemory, Result as CoreResult,
};
#[cfg(test)]
use mlx_gen::LoadShape;
use mlx_gen::{LoadSpec, OffloadPolicy};

use crate::pipeline::{DECODE_OVERLAP, DECODE_TILE_EDGE};

pub const DECODE_TILE_EDGES: &[u32] = &[512, 384, 256, 192];
pub const DECODE_TILE_EDGES_REJECTED: &[u32] = &[128, 96, 64];
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sana-mlx-dcae-tiled-decode-v2-sequential";
pub const RESIDENT_MEMORY_CALIBRATION_FINGERPRINT: &str = "sana-mlx-dcae-tiled-decode-v2-resident";

pub const fn calibration_fingerprint(policy: OffloadPolicy) -> &'static str {
    match policy {
        OffloadPolicy::Sequential => MEMORY_CALIBRATION_FINGERPRINT,
        OffloadPolicy::Resident => RESIDENT_MEMORY_CALIBRATION_FINGERPRINT,
    }
}

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP as u32,
    )
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let components = crate::model::component_footprint(spec)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        components.text_encoder,
        components.dit,
        components.vae,
    )
}

pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts(provider_id, spec, 0, 0, 0)
}

fn contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
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
            MemoryFormulaVariable::DecodeTileArea,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        calibration_fingerprint(spec.offload_policy),
        spec.load_shape,
    ));
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    // Sequential loads ship the measured bounded-decode default. An explicit shared-contract
    // Resident selection must therefore write an all-disabled request block to override it.
    contract.resident_request_memory = ResidentRequestMemory::ExplicitResident;

    for strategy in [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
    ] {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == strategy)
            .expect("compatibility contract declares every rung");
        capability.support = MemoryStrategySupport::Implemented;
        if strategy == MemoryStrategy::BoundedDecode {
            capability.parameters = MemoryParameterRanges {
                decode_tile_edges: routes.published_edges(),
                decode_overlaps: routes.published_overlaps(),
                ..Default::default()
            };
        }
    }
    contract.pid_decode_routes = Some(MemoryPidDecodeRoutes {
        native: MemoryDecodeRouteDomain {
            tile_edges: routes.native_edges().to_vec(),
            tile_overlap: DECODE_OVERLAP as u32,
        },
        pid: MemoryDecodeRouteDomain {
            tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
            tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
        },
    });
    // The compatibility constructor supplies the required empty engagement-exclusion surface.
    debug_assert!(contract.default_engagement_exclusions.is_empty());
    Ok(contract)
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            if context.use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{}: SANA has no PiD decoder; mlx-gen-pid is used only for Gemma-2 conditioning",
                    contract.provider_id
                )));
            }
            decode_routes(&contract.provider_id)?
                .validate(
                    false,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !matches!(
        contract
            .capability(strategy)
            .map(|capability| &capability.support),
        Some(MemoryStrategySupport::Implemented)
    ) || !strategy.is_optimized()
    {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: matches!(spec.offload_policy, OffloadPolicy::Sequential),
            overlay: None,
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_with_cleanup(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::SanaTransformerConfig::sana_1600m().num_layers as usize,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{provider_id}: SANA has no PiD decoder"
                )));
            }
            routes
                .validate(false, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = None;
    config.transformer_window = None;
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub fn declared_parameters() -> mlx_gen::gen_core::MemoryStrategyParameters {
    mlx_gen::gen_core::MemoryStrategyParameters {
        decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
        decode_overlap: Some(DECODE_OVERLAP as u32),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;
    use mlx_gen::WeightsSource;

    fn spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-contract".into()))
            .with_offload_policy(policy)
            .with_load_shape(LoadShape::EagerMaterialization)
    }

    #[test]
    fn both_ids_publish_the_same_three_rung_ladder() {
        let spec = spec(OffloadPolicy::Sequential);
        let base = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let sprint = weights_free_memory_strategy_contract(crate::SPRINT_MODEL_ID, &spec).unwrap();
        assert!(base.conformance_errors().is_empty());
        for strategy in MemoryStrategy::ALL {
            assert_eq!(base.capability(strategy), sprint.capability(strategy));
        }
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                base.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented
            );
        }
        assert_eq!(
            base.resident_request_memory,
            ResidentRequestMemory::ExplicitResident
        );
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                base.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing
            );
        }
    }

    #[test]
    fn decode_domain_and_rejection_set_are_pinned() {
        assert_eq!(DECODE_TILE_EDGES, &[512, 384, 256, 192]);
        assert_eq!(DECODE_TILE_EDGES_REJECTED, &[128, 96, 64]);
        assert_eq!(DECODE_TILE_EDGE, 192);
        assert_eq!(DECODE_OVERLAP, 48);
        let routes = decode_routes(crate::MODEL_ID).unwrap();
        routes.validate(false, Some(192), Some(48)).unwrap();
        assert!(routes.validate(false, Some(128), Some(48)).is_err());
        assert!(routes.validate(true, Some(192), Some(48)).is_err());
    }

    #[test]
    fn calibration_identity_is_split_by_offload_policy() {
        let resident =
            weights_free_memory_strategy_contract(crate::MODEL_ID, &spec(OffloadPolicy::Resident))
                .unwrap();
        let sequential = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &spec(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_ne!(resident.calibration, sequential.calibration);
    }
}
