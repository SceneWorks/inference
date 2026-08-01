//! FLUX.2-dev edit's provider-side memory-safety contract.
//!
//! The provider already bounds long multi-reference sequences internally with `MemoryConfig::LONG_SEQ`.
//! SceneWorks supplies request geometry, numeric tier, reference count, and the live unified-memory
//! budget. This module owns the calibrated peak formula and final decision so the worker carries no
//! provider-specific tier or fit fork.

use mlx_gen::gen_core::{
    safetensors_path_bytes, standard_memory_strategy_safety_check, Error as CoreError, LoadShape,
    LoadSpec, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryMode, MemoryNumericTier, MemoryProviderContract, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport, Quant, WeightsSource,
};

use crate::config::{Flux2Variant, FLUX2_DEV_EDIT_ID};

pub const CALIBRATION_FINGERPRINT: &str = "sc-6211-flux2-dev-edit-chunked-v1";
/// SC-6211 measured activation/transient allowance above the chunked request peak. This is a
/// provider invariant, distinct from any caller-owned OS/foreign-process reserve.
pub const REQUIRED_TRANSIENT_HEADROOM_GB: f64 = 12.0;

/// SC-6211 chunked Q4 fit, measured at 1024²: two references ~81 GiB and four ~93 GiB.
/// The constants and provenance deliberately match the historical worker gate byte for byte.
const BASE_GB: f64 = 62.9;
const PER_TOKEN_GB: f64 = 0.001_489; // (93 − 81) GB / (20480 − 12288) tokens, sc-6211 (chunked).

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
    contract.calibration = Some(MemoryCalibrationIdentity::new(CALIBRATION_FINGERPRINT));
    for capability in &mut contract.strategies {
        if capability.strategy != MemoryStrategy::Resident {
            capability.support = MemoryStrategySupport::StructurallyNotApplicable {
                reason: "FLUX.2 bounds long-sequence activations inside its resident request path"
                    .to_owned(),
            };
        }
    }
    contract
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
    selected_weight_bytes: u64,
) -> MemorySafetyDecision {
    let reference_count = context
        .overlay
        .as_deref()
        .and_then(|overlay| overlay.strip_prefix("references="))
        .and_then(|count| count.parse::<usize>().ok())
        .filter(|count| *count >= 2);
    let required_transient_bytes =
        (REQUIRED_TRANSIENT_HEADROOM_GB * 1024.0 * 1024.0 * 1024.0).round() as u64;
    let tokens_per_image = (f64::from(context.geometry.width) / 16.0).ceil()
        * (f64::from(context.geometry.height) / 16.0).ceil();
    // Use a harmless placeholder while assembling the provider-owned context. The route gate below
    // rejects a missing/invalid count before the shared budget stage can observe this prediction.
    let total_tokens = (1.0 + reference_count.unwrap_or(2) as f64) * tokens_per_image;
    let q4_peak_bytes =
        ((BASE_GB + PER_TOKEN_GB * total_tokens) * 1024.0 * 1024.0 * 1024.0).round() as u64;
    let predicted_peak_bytes = match expected_tier.quant {
        Some(Quant::Q4) => q4_peak_bytes,
        Some(Quant::Q8) | None if selected_weight_bytes > 0 => {
            // Q8/BF16 lack request-peak measurements. Preserve their supported surface with a
            // conservative provider-owned bound: the measured Q4 request peak plus the entire
            // selected snapshot footprint. This may double-count packed resident weights, but it
            // cannot inherit the smaller tier's admission boundary or admit on fabricated evidence.
            q4_peak_bytes.saturating_add(selected_weight_bytes)
        }
        Some(Quant::Q8) | None => 0,
        Some(Quant::Nvfp4) => 0,
    };
    let mut provider_context = context.clone();
    provider_context.predicted_peak_bytes = predicted_peak_bytes;
    let route_accepted = std::cell::Cell::new(false);
    let route_gate = || {
        if context.mode != MemoryMode::Edit || !context.has_reference {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must describe a referenced edit"
            )));
        }
        let Some(_) = reference_count else {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must name at least two references"
            )));
        };
        if context.budget.reserved_headroom_bytes < required_transient_bytes {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context reserves less than the measured \
                 {REQUIRED_TRANSIENT_HEADROOM_GB:.0} GiB activation/transient allowance"
            )));
        }
        match expected_tier.quant {
            Some(Quant::Q8) | None if selected_weight_bytes == 0 => {
                return Err(CoreError::Unsupported(format!(
                    "{FLUX2_DEV_EDIT_ID}: {:?} multi-reference admission needs the selected snapshot footprint",
                    expected_tier.quant
                )));
            }
            Some(Quant::Nvfp4) => {
                return Err(CoreError::Unsupported(format!(
                    "{FLUX2_DEV_EDIT_ID}: NVFP4 is not implemented by the MLX provider"
                )));
            }
            _ => {}
        }
        route_accepted.set(true);
        Ok(())
    };
    match standard_memory_strategy_safety_check(
        contract,
        &provider_context,
        Some(expected_tier),
        Some(&route_gate),
    ) {
        MemorySafetyDecision::Accept => MemorySafetyDecision::Accept,
        MemorySafetyDecision::Reject { reason: _ }
            if route_accepted.get()
                && !provider_context
                    .budget
                    .fits(provider_context.predicted_peak_bytes) =>
        {
            let reference_count = reference_count.expect("accepted route has reference count");
            let gib = 1024.0 * 1024.0 * 1024.0;
            MemorySafetyDecision::Reject {
                reason: format!(
                    "FLUX.2-dev multi-reference edit at {}×{} with {reference_count} reference \
                     images needs ~{} GB of unified memory (with headroom) but this machine has \
                     ~{} GB. Lower the output resolution, use a single reference image, choose a \
                     smaller numeric tier, or run on a Mac with more memory.",
                    context.geometry.width,
                    context.geometry.height,
                    ((predicted_peak_bytes + context.budget.reserved_headroom_bytes) as f64 / gib)
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
    let selected_weight_bytes = match &spec.weights {
        WeightsSource::Dir(path) | WeightsSource::File(path) => safetensors_path_bytes(path),
    };
    safety_check(
        contract,
        context,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        selected_weight_bytes,
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
            mode: MemoryMode::Edit,
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            overlay: Some("references=2".to_owned()),
            budget: MemoryBudget {
                total_bytes: bytes(total_gb),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: bytes(REQUIRED_TRANSIENT_HEADROOM_GB),
            },
            // Deliberately bogus: provider safety must recompute rather than trust the worker.
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "sc-6211".to_owned(),
        }
    }

    #[test]
    fn provider_safety_preserves_measured_multireference_boundaries() {
        let contract = build_contract();
        assert_eq!(
            safety_check(&contract, &context(96.0), context(96.0).selection.tier, 0),
            MemorySafetyDecision::Accept
        );
        assert!(matches!(
            {
                let mut four = context(96.0);
                four.overlay = Some("references=4".to_owned());
                safety_check(&contract, &four, four.selection.tier, 0)
            },
            MemorySafetyDecision::Reject { .. }
        ));
        assert_eq!(
            {
                let mut four = context(128.0);
                four.overlay = Some("references=4".to_owned());
                safety_check(&contract, &four, four.selection.tier, 0)
            },
            MemorySafetyDecision::Accept
        );
    }

    #[test]
    fn provider_safety_rejects_a_stale_calibration_identity() {
        let contract = build_contract();
        let mut stale = context(96.0);
        stale.calibration_fingerprint = "stale".to_owned();
        assert!(matches!(
            safety_check(&contract, &stale, stale.selection.tier, 0),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn provider_safety_enforces_its_transient_reserve_invariant() {
        let contract = build_contract();
        for reserved_gb in [0.0_f64, 11.9] {
            let mut under_reserved = context(128.0);
            under_reserved.budget.reserved_headroom_bytes =
                (reserved_gb * 1024.0 * 1024.0 * 1024.0).round() as u64;
            assert!(matches!(
                safety_check(&contract, &under_reserved, under_reserved.selection.tier, 0),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }

    #[test]
    fn provider_safety_owns_tier_identity_and_conservative_large_tier_admission() {
        let contract = build_contract();
        let sixty_gib = (60.0 * 1024.0 * 1024.0 * 1024.0) as u64;
        let q4_tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        for quant in [Some(Quant::Q8), None] {
            let mut larger_tier = context(128.0);
            larger_tier.selection.tier.quant = quant;
            let expected_tier = larger_tier.selection.tier;
            assert!(matches!(
                safety_check(&contract, &larger_tier, expected_tier, sixty_gib),
                MemorySafetyDecision::Reject { .. }
            ));
            larger_tier.budget.total_bytes = (256.0 * 1024.0 * 1024.0 * 1024.0) as u64;
            assert_eq!(
                safety_check(&contract, &larger_tier, expected_tier, sixty_gib),
                MemorySafetyDecision::Accept
            );
            assert!(matches!(
                safety_check(&contract, &larger_tier, q4_tier, sixty_gib),
                MemorySafetyDecision::Reject { .. }
            ));
            assert!(matches!(
                safety_check(&contract, &larger_tier, expected_tier, 0),
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
            0,
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
        let MemorySafetyDecision::Reject { reason } = safety_check(&contract, &wrong_tier, q4, 0)
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
            0,
        ) else {
            panic!("invalid selection must reject");
        };
        assert!(reason.contains("decode_tile_edge"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_route = context(1.0);
        wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &wrong_route, wrong_route.selection.tier, 0)
        else {
            panic!("provider route policy must reject");
        };
        assert!(reason.contains("referenced edit"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let admitted_route = context(1.0);
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &admitted_route, admitted_route.selection.tier, 0)
        else {
            panic!("under-budget request must reject");
        };
        assert!(reason.contains("Lower the output resolution"), "{reason}");
    }
}
