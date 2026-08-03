//! FLUX.1 MLX shared memory-provider contract foundation (SC-15514).
//!
//! This slice exposes the existing two-phase `Residency` lifecycle through the shared provider
//! contract. Decode tiling, bounded attention, and transformer windows remain `Missing` until their
//! runtime mechanisms and route-specific evidence land. Production contracts deliberately carry no
//! calibration identity; weights-free registry conformance receives an isolated synthetic identity.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Result as CoreResult,
};
use mlx_gen::{LoadSpec, OffloadPolicy};

const STATIC_CALIBRATION: &str = "flux-one-static-registry-behavior-v1";

fn is_known_provider(provider_id: &str) -> bool {
    matches!(
        provider_id,
        crate::FLUX1_SCHNELL_ID | crate::FLUX1_DEV_ID | crate::FLUX1_DEV_CONTROL_ID
    )
}

fn route_overlay(provider_id: &str, spec: &LoadSpec) -> Option<String> {
    let mut axes = Vec::new();
    if provider_id == crate::FLUX1_DEV_CONTROL_ID || spec.control.is_some() {
        axes.push("control");
    }
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.ip_adapter.is_some() {
        axes.push("ip-adapter");
    }
    if spec.pid.is_some() {
        axes.push("pid");
    }
    if spec.identity.is_some() {
        axes.push("identity");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn route_mode_and_references(provider_id: &str, spec: &LoadSpec) -> (MemoryMode, u32) {
    if provider_id == crate::FLUX1_DEV_CONTROL_ID
        || spec.control.is_some()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
    {
        (MemoryMode::ImageToImage, 1)
    } else {
        (MemoryMode::TextToImage, 0)
    }
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    footprint: mlx_gen::PerComponentBytes,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    if !is_known_provider(provider_id) {
        return Err(CoreError::Unsupported(format!(
            "unknown FLUX.1 provider {provider_id}"
        )));
    }
    let staged = matches!(spec.offload_policy, OffloadPolicy::Sequential);
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
            MemoryFormulaVariable::OverlayBytes,
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
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            _ => MemoryStrategySupport::Missing,
        };
    }
    Ok(contract)
}

/// Production contract. Filesystem-backed asset facts are real, but no optimized route is admitted
/// until an exact route/tier/overlay/artifact calibration exists.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    build_contract(
        provider_id,
        spec,
        crate::model::component_footprint(spec)?,
        None,
    )
}

/// Declaration-equivalent, zero-filesystem contract used only by registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let route = match provider_id {
        crate::FLUX1_SCHNELL_ID => "schnell",
        crate::FLUX1_DEV_ID => "dev",
        crate::FLUX1_DEV_CONTROL_ID => "dev-control",
        _ => "unknown",
    };
    build_contract(
        provider_id,
        spec,
        mlx_gen::PerComponentBytes::default(),
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_CALIBRATION}-{route}"),
            spec.load_shape,
        )),
    )
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let (expected_mode, expected_references) =
            route_mode_and_references(&contract.provider_id, spec);
        if context.mode != expected_mode
            || context.geometry.reference_count != expected_references
            || context.overlay != route_overlay(&contract.provider_id, spec)
        {
            return Err(CoreError::Unsupported(format!(
                "{}: memory route does not match the loaded mode/overlay",
                contract.provider_id
            )));
        }
        if context.use_pid && spec.pid.is_none() {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD route requested without a loaded PiD overlay",
                contract.provider_id
            )));
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
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

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(spec, contract, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let (mode, reference_count) = route_mode_and_references(&contract.provider_id, spec);
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode,
            reference_count,
            use_pid: false,
            has_phases: matches!(spec.offload_policy, OffloadPolicy::Sequential),
            overlay: route_overlay(&contract.provider_id, spec),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        57,
        move |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded decode is not implemented"
            )))
        },
    )?;
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(
            config,
            mlx_gen::request_scope::MlxScopeCleanup::None,
        ),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    fn sequential_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
    }

    #[test]
    fn static_contract_is_zero_filesystem_and_declares_only_existing_staging() {
        for provider in [
            crate::FLUX1_SCHNELL_ID,
            crate::FLUX1_DEV_ID,
            crate::FLUX1_DEV_CONTROL_ID,
        ] {
            let spec = sequential_spec();
            let contract = weights_free_memory_strategy_contract(provider, &spec).unwrap();
            assert_eq!(contract.asset_facts, Default::default());
            assert!(contract.conformance_errors().is_empty());
            assert_eq!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
            for missing in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    contract.capability(missing).unwrap().support,
                    MemoryStrategySupport::Missing
                );
            }
            let fixtures =
                registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                    .unwrap();
            assert_eq!(fixtures.len(), 1);
            assert_eq!(
                registered_safety_check(&spec, &contract, &fixtures[0].context),
                MemorySafetyDecision::Accept
            );
        }
    }

    #[test]
    fn route_tier_overlay_and_load_shape_are_fail_closed() {
        let mut spec = sequential_spec();
        spec.adapters.push(mlx_gen::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        spec.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
        spec = spec.with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        );
        let contract = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let mut context =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;
        assert_eq!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
        for mutation in 0..4 {
            let mut changed = context.clone();
            match mutation {
                0 => changed.overlay = Some("different".to_owned()),
                1 => changed.geometry.reference_count = 0,
                2 => changed.selection.tier.precision = mlx_gen::Precision::Fp32,
                _ => {
                    changed.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
                }
            }
            assert!(matches!(
                registered_safety_check(&spec, &contract, &changed),
                MemorySafetyDecision::Reject { .. }
            ));
        }
        context.calibration_fingerprint.push_str("-stale");
        assert!(matches!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn production_unknown_artifact_has_no_calibration_and_rejects_static_context() {
        let root = std::env::temp_dir().join(format!("flux-memory-{}", std::process::id()));
        for component in ["text_encoder", "text_encoder_2", "transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let runtime = memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        assert!(runtime.calibration.is_none());
        let fixture = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let context = registered_valid_fixture(&spec, &fixture, MemoryStrategy::StagedResidency)
            .unwrap()
            .remove(0)
            .context;
        assert!(matches!(
            registered_safety_check(&spec, &runtime, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        std::fs::remove_dir_all(root).ok();
    }
}
