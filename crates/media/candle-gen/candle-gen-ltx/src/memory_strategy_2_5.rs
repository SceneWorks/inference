//! The LTX-2.5 distilled Candle memory registration (sc-18797).
//!
//! This stays separate from the released LTX-2.3 q4 contract in [`crate::memory_strategy`]. The
//! 2.5 Candle generator descriptor and split-bundle loader are owned by the paired engine story;
//! wiring this registration before that descriptor lands would make the registry reject the
//! catalog. The contract is therefore correct-but-inert until that dependency integrates it.

use candle_gen::gen_core::{
    self, LoadShape, LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges,
    MemoryPhase, MemoryProviderContract, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, OffloadPolicy, ResidentRequestMemory, TransformerComponent,
    WeightsSource,
};

/// The Candle LTX-2.5 engine id; never alias the released LTX-2.3 route.
pub const LTX_2_5_DISTILLED_MODEL_ID: &str = "ltx_2_5_distilled";

/// Distinguishable 48-block-window candidates. `48` is absent because it bounds nothing.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 16];

/// One AV block owns both audio and video attention, but not the separate Gemma-4 encoder.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Shared attention score-budget candidates, in score elements per chunk.
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[67_108_864, 16_777_216];

const CALIBRATION_FINGERPRINT: &str = "sc-18797-ltx-2-5-candle-ladder-v1";

/// Rung 4 is meaningful only with staged components and a re-openable, adapter-free source.
/// `LtxBlockStream` enforces the same adapter restriction at the execution seam.
fn streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.adapters.is_empty()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn strategies(spec: &LoadSpec) -> Vec<MemoryStrategyCapability> {
    let windowed = streamable(spec);
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedAttention => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency if windowed => {
                    MemoryStrategySupport::Implemented
                }
                _ => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
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

/// Build the 2.5 Candle contract. Rung 3 is `gen_core::attention_budget` via
/// `candle_gen::sdpa_budgeted_bhsd`; rung 4 is Candle's binding over `gen_core::block_window`, not
/// a provider-local window driver.
pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let windowed = streamable(spec);
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    Ok(MemoryProviderContract {
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
            decode_tiling: false,
            attention_chunking: true,
            transformer_window_materialization: windowed,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        asset_facts: gen_core::MemoryAssetFacts::default(),
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

/// The first Candle LTX-2.5 memory registration. Wiring waits for the matching generator.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: LTX_2_5_DISTILLED_MODEL_ID,
    contract: memory_strategy_contract,
    safety_check,
};

fn safety_check(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    gen_core::MemorySafetyDecision::Reject {
        reason: "ltx_2_5_distilled has no Candle generator descriptor yet; selection is owned by the LTX-2.5 engine story".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-2-5".into()))
            .with_offload_policy(policy)
            .with_load_shape(LoadShape::DeferredMaterialization)
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
    fn registration_names_the_distilled_two_five_engine() {
        assert_eq!(MEMORY_REGISTRATION.provider_id, LTX_2_5_DISTILLED_MODEL_ID);
        assert_ne!(MEMORY_REGISTRATION.provider_id, crate::MODEL_ID);
    }

    #[test]
    fn sequential_selects_windowing_but_resident_does_not() {
        let sequential = memory_strategy_contract(&spec(OffloadPolicy::Sequential)).unwrap();
        assert_eq!(
            support(&sequential, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Implemented
        );
        assert!(sequential.lifecycle.transformer_window_materialization);

        let resident = memory_strategy_contract(&spec(OffloadPolicy::Resident)).unwrap();
        assert_eq!(
            support(&resident, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing
        );
        assert!(!resident.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn shared_attention_and_window_contracts_are_advertised() {
        let contract = memory_strategy_contract(&spec(OffloadPolicy::Sequential)).unwrap();
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedAttention),
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            ATTENTION_CHUNK_SIZES
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit]
        );
    }
}
