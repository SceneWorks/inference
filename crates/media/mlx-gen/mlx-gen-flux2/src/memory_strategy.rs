//! FLUX.2-dev edit's provider-side memory-safety contract.
//!
//! The provider already bounds long multi-reference sequences internally with `MemoryConfig::LONG_SEQ`.
//! SceneWorks supplies request geometry, numeric tier, incremental live demand derived from the
//! evidence-owned absolute peak, and the live unified-memory budget. This module validates the
//! provider route and tier, then delegates the canonical budget comparison to `gen-core`;
//! calibration coefficients never live in a provider.

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryMode, MemoryNumericTier, MemoryProviderContract, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Quant,
};

use crate::config::{Flux2Variant, FLUX2_DEV_EDIT_ID};

pub const CALIBRATION_FINGERPRINT: &str = "sc-16593-flux2-dev-edit-evidence-v2";

pub fn contract_for_variant(variant: Flux2Variant) -> Option<MemoryProviderContract> {
    (variant == Flux2Variant::DevEdit).then(build_contract)
}

fn build_contract() -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_EDIT_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.formula = MemoryFormulaKind::Affine {
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    for capability in &mut contract.strategies {
        if capability.strategy != MemoryStrategy::Resident {
            capability.support = MemoryStrategySupport::Missing;
        }
    }
    contract
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_accepted = std::cell::Cell::new(false);
    let route_gate = || {
        if context.mode != MemoryMode::Edit || !context.has_reference {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must describe a referenced edit"
            )));
        }
        if context.geometry.reference_count < 2 {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must name at least two references"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        route_accepted.set(true);
        Ok(())
    };
    match standard_memory_strategy_safety_check(
        contract,
        context,
        Some(expected_tier),
        Some(&route_gate),
    ) {
        MemorySafetyDecision::Accept => MemorySafetyDecision::Accept,
        MemorySafetyDecision::Reject { reason }
            if route_accepted.get()
                && reason.contains("incremental live demand")
                && reason.contains("exceeds effective budget") =>
        {
            let reference_count = context.geometry.reference_count;
            let gib = 1024.0 * 1024.0 * 1024.0;
            MemorySafetyDecision::Reject {
                reason: format!(
                    "FLUX.2-dev multi-reference edit at {}×{} with {reference_count} reference \
                     images needs ~{} GB of unified memory (with headroom) but this machine has \
                     ~{} GB. Lower the output resolution, use a single reference image, choose a \
                     smaller numeric tier, or run on a Mac with more memory.",
                    context.geometry.width,
                    context.geometry.height,
                    (context
                        .budget
                        .required_total_bytes(context.predicted_peak_bytes)
                        as f64
                        / gib)
                        .round() as i64,
                    (context.budget.total_bytes as f64 / gib).round() as i64,
                ),
            }
        }
        MemorySafetyDecision::Reject { reason } => MemorySafetyDecision::Reject { reason },
    }
}

pub fn registered_contract(_spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract())
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(
        contract,
        context,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryNumericTier, MemorySelection,
    };
    use mlx_gen::{Precision, Quant};

    fn context(total_gb: f64) -> MemoryRunContext {
        let contract = build_contract();
        let calibration = contract.calibration.expect("calibration");
        let bytes = |gb: f64| (gb * 1024.0 * 1024.0 * 1024.0).round() as u64;
        MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            load_shape: calibration.load_shape,
            mode: MemoryMode::Edit,
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 2,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: bytes(total_gb),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: bytes(81.0),
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "worker-owned-exact-evidence".to_owned(),
        }
    }

    #[test]
    fn provider_safety_uses_the_caller_owned_peak_without_recomputing_it() {
        let contract = build_contract();
        let mut exact = context(81.0);
        assert_eq!(
            safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Accept
        );
        exact.budget.total_bytes -= 1;
        assert!(matches!(
            safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut four = context(81.0);
        four.geometry.reference_count = 4;
        assert_eq!(
            safety_check(&contract, &four, four.selection.tier),
            MemorySafetyDecision::Accept
        );
    }

    #[test]
    fn provider_safety_rejects_a_stale_calibration_identity() {
        let contract = build_contract();
        let mut stale = context(96.0);
        stale.calibration_fingerprint = "stale".to_owned();
        assert!(matches!(
            safety_check(&contract, &stale, stale.selection.tier),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn provider_contract_quarantines_structured_overlays_and_false_reference_summaries() {
        let contract = build_contract();
        assert!(contract.strategies.iter().all(|capability| {
            capability.strategy == MemoryStrategy::Resident
                || capability.support == MemoryStrategySupport::Missing
        }));

        let mut structured_overlay = context(128.0);
        structured_overlay.overlay = Some("references=2".to_owned());
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &structured_overlay,
            structured_overlay.selection.tier,
        ) else {
            panic!("structured overlay data must reject");
        };
        assert!(reason.contains("overlay is an identity axis"), "{reason}");

        let mut inconsistent = context(128.0);
        inconsistent.has_reference = false;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &inconsistent, inconsistent.selection.tier)
        else {
            panic!("inconsistent compatibility summary must reject");
        };
        assert!(
            reason.contains("inconsistent with reference_count=2"),
            "{reason}"
        );
    }

    #[test]
    fn provider_safety_owns_tier_identity_but_not_tier_peak_estimation() {
        let contract = build_contract();
        let q4_tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        for quant in [Some(Quant::Q8), None] {
            let mut larger_tier = context(128.0);
            larger_tier.selection.tier.quant = quant;
            let expected_tier = larger_tier.selection.tier;
            assert_eq!(
                safety_check(&contract, &larger_tier, expected_tier),
                MemorySafetyDecision::Accept
            );
            assert!(matches!(
                safety_check(&contract, &larger_tier, q4_tier),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }

    #[test]
    fn shared_rejections_keep_their_reason_before_provider_policy_and_budget_advice() {
        let contract = build_contract();

        let mut stale_and_wrong_route = context(1.0);
        stale_and_wrong_route.calibration_fingerprint = "stale".to_owned();
        stale_and_wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &stale_and_wrong_route,
            stale_and_wrong_route.selection.tier,
        ) else {
            panic!("stale handshake must reject");
        };
        assert!(
            reason.contains("calibration handshake mismatch"),
            "{reason}"
        );
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_tier = context(1.0);
        wrong_tier.selection.tier.quant = Some(Quant::Q8);
        let q4 = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let MemorySafetyDecision::Reject { reason } = safety_check(&contract, &wrong_tier, q4)
        else {
            panic!("wrong tier must reject");
        };
        assert!(reason.contains("does not match loaded tier"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut invalid_selection = context(1.0);
        invalid_selection.selection.parameters.decode_tile_edge = Some(512);
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &invalid_selection,
            invalid_selection.selection.tier,
        ) else {
            panic!("invalid selection must reject");
        };
        assert!(reason.contains("decode_tile_edge"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_route = context(1.0);
        wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &wrong_route, wrong_route.selection.tier)
        else {
            panic!("provider route policy must reject");
        };
        assert!(reason.contains("referenced edit"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let admitted_route = context(1.0);
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &admitted_route, admitted_route.selection.tier)
        else {
            panic!("under-budget request must reject");
        };
        assert!(reason.contains("Lower the output resolution"), "{reason}");
    }
}
