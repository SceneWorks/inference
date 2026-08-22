//! Exact Candle/CUDA request-memory contract for the three Anima text-to-image routes (SC-20785).
//!
//! The only non-resident implementation is staged residency: Qwen3 + the bundled conditioner
//! produce immutable conditioning, are synchronized and dropped, then the DiT + VAE are loaded.
//! Decode tiling, attention chunking and transformer windows have no parity-safe implementation
//! here and are deliberately classified rather than implied by the shared ladder order.

use candle_gen::gen_core::{
    self, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization,
};

const IDS: &[&str] = &["anima_base", "anima_aesthetic", "anima_turbo"];
const FINGERPRINT: &str = "anima-candle-request-scoped-conditioning-v1";

pub fn contract(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    if !IDS.contains(&provider_id) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: not an Anima Candle memory provider"
        )));
    }
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    // The three routes share an implementation but not calibration evidence.  A base calibration
    // must never admit aesthetic/turbo merely because their component layout happens to match.
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        format!("{FINGERPRINT}-{}", provider_id.replace('_', "-")),
        spec.load_shape,
    ));
    let staged = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
        .expect("compatibility contract contains every strategy");
    // This is a route capability, not a statement about every request overlay.  The request
    // safety check refuses adapters only when staged residency is selected; resident LoRA/LoKr
    // remains the advertised and implemented load path.
    staged.support = MemoryStrategySupport::Implemented;
    Ok(contract)
}

pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    provider_safety_check(spec, contract, context)
}

pub fn provider_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let staged = contract.engages(context.selection.strategy, MemoryStrategy::StagedResidency);
    let route_gate = || {
        if !staged {
            return Ok(());
        }
        if !spec.adapters.is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: staged residency refuses LoRA/LoKr overlays because their conditioner and DiT loads are one atomic artifact",
                contract.provider_id
            )));
        }
        if context.mode == MemoryMode::TextToImage
            && context.geometry.reference_count == 0
            && !context.use_pid
            && !context.has_phases
            && context.overlay.is_none()
        {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: staged memory is only valid for the exact text_to_image/no-overlay route",
                contract.provider_id
            )))
        }
    };
    gen_core::standard_memory_strategy_safety_check(
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

/// Provider-owned behavior fixture for every exact Anima Candle route.  The fixture is deliberately
/// weights-free but carries the same tier, route and calibration identity the worker will admit.
pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    Ok(vec![gen_core::MemoryBehaviorFixture::new(
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            },
            gen_core::MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )?,
    )])
}

/// Weights-free executable admission seam.  It mirrors the loaded provider's validation before
/// creating the same request scope used by worker lifecycle telemetry.
pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    Ok(Some(Box::new(crate::AnimaMemoryScope::new(
        contract, context,
    ))))
}

macro_rules! registration {
    ($name:ident, $id:literal) => {
        pub const $name: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
            provider_id: $id,
            contract: |spec| contract($id, spec),
            safety_check,
        };
    };
}

registration!(BASE_MEMORY_REGISTRATION, "anima_base");
registration!(AESTHETIC_MEMORY_REGISTRATION, "anima_aesthetic");
registration!(TURBO_MEMORY_REGISTRATION, "anima_turbo");

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        AdapterKind, AdapterSpec, MemoryBehaviorRoute, MemoryNumericTier, MemoryStrategy,
        Precision, WeightsSource,
    };

    #[test]
    fn all_three_exact_plain_routes_publish_only_staged_residency() {
        for provider in IDS {
            let contract = contract(
                provider,
                &LoadSpec::new(WeightsSource::Dir("/anima".into())),
            )
            .unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented,
                "{provider}"
            );
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_ne!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{provider}: unsupported rung must not borrow staged evidence"
                );
            }
        }
    }

    #[test]
    fn staged_overlay_is_refused_but_resident_lora_and_lokr_remain_admitted() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        spec.adapters.push(AdapterSpec::new(
            "/anima/overlay.safetensors".into(),
            1.0,
            AdapterKind::Lora,
        ));
        let contract = contract("anima_base", &spec).unwrap();
        let resident = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::Resident,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: Some("lora:overlay".to_owned()),
            },
        )
        .unwrap();
        assert_eq!(
            safety_check(&spec, &contract, &resident),
            MemorySafetyDecision::Accept
        );

        spec.adapters[0].kind = AdapterKind::Lokr;
        assert_eq!(
            safety_check(&spec, &contract, &resident),
            MemorySafetyDecision::Accept
        );

        let mut staged =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .pop()
                .unwrap()
                .context;
        staged.overlay = Some("lokr:overlay".to_owned());
        assert!(matches!(
            safety_check(&spec, &contract, &staged),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn exact_route_tier_and_calibration_are_bound_before_admission() {
        let spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        let base = contract("anima_base", &spec).unwrap();
        let aesthetic = contract("anima_aesthetic", &spec).unwrap();
        let context = registered_valid_fixture(&spec, &base, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap()
            .context;
        assert_eq!(
            safety_check(&spec, &base, &context),
            MemorySafetyDecision::Accept
        );

        let mut crossed_tier = context.clone();
        crossed_tier.selection.tier.quant = Some(gen_core::Quant::Q4);
        assert!(matches!(
            safety_check(&spec, &base, &crossed_tier),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut crossed_route = context;
        crossed_route.calibration_fingerprint = aesthetic.calibration.unwrap().fingerprint;
        assert!(matches!(
            safety_check(&spec, &base, &crossed_route),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn every_route_has_provider_owned_fixture_and_begin_request_seam() {
        let spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        for provider in IDS {
            let contract = contract(provider, &spec).unwrap();
            let fixture =
                registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                    .unwrap()
                    .pop()
                    .unwrap();
            assert!(registered_begin_request(&spec, &contract, &fixture.context)
                .unwrap()
                .is_some());
        }
    }
}
