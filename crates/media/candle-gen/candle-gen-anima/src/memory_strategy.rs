//! Exact Candle/CUDA request-memory contract for the three Anima text-to-image routes (SC-20785).
//!
//! The only non-resident implementation is staged residency: Qwen3 + the bundled conditioner
//! produce immutable conditioning, are synchronized and dropped, then the DiT + VAE are loaded.
//! Decode tiling, attention chunking and transformer windows have no parity-safe implementation
//! here and are deliberately classified rather than implied by the shared ladder order.

use candle_gen::gen_core::{
    self, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryPhase, MemoryProviderContract,
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
    contract.calibration = Some(MemoryCalibrationIdentity::new(FINGERPRINT, spec.load_shape));
    let staged = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
        .expect("compatibility contract contains every strategy");
    staged.support = if spec.adapters.is_empty() {
        MemoryStrategySupport::Implemented
    } else {
        MemoryStrategySupport::StructurallyNotApplicable {
            reason: "Anima adapter overlays span the conditioner and DiT load boundary; staged residency refuses them rather than applying a partial overlay".to_owned(),
        }
    };
    Ok(contract)
}

pub fn safety_check(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    provider_safety_check(contract, context)
}

pub fn provider_safety_check(
    contract: &MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    let route_gate = || {
        if context.mode == gen_core::MemoryMode::TextToImage
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
    gen_core::standard_memory_strategy_safety_check(contract, context, None, Some(&route_gate))
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
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, MemoryStrategy, WeightsSource};

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
    fn overlay_is_classified_not_silently_staged() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/anima".into()));
        spec.adapters.push(AdapterSpec::new(
            "/anima/overlay.safetensors".into(),
            1.0,
            AdapterKind::Lora,
        ));
        assert!(matches!(
            contract("anima_base", &spec)
                .unwrap()
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
    }
}
