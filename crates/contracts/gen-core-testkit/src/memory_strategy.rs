//! Weights-free conformance checks for the shared memory-strategy provider contract.

use gen_core::{
    LoadSpec, MemoryBudget, MemoryCacheState, MemoryCleanupSemantics, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryProviderContract, MemoryRegistration, MemoryRunContext,
    MemorySafetyDecision, MemorySelection, MemoryStrategy, MemoryStrategyParameters,
    MemoryStrategySupport, Precision, ProviderRegistry,
};

/// Check the static declaration and the safety-critical runtime semantics every provider must share.
pub fn check_memory_strategy_contract(
    contract: &MemoryProviderContract,
) -> Result<(), Vec<String>> {
    let mut errors = contract.conformance_errors();

    if !matches!(
        contract
            .capability(MemoryStrategy::Resident)
            .map(|capability| &capability.support),
        Some(MemoryStrategySupport::Implemented)
    ) {
        errors.push("Resident baseline must be implemented".to_owned());
    }
    if contract.runtime.cancellation
        != MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
    {
        errors.push("cancellation must synchronize and release active state".to_owned());
    }
    if contract.runtime.error != MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
    {
        errors.push("errors must synchronize and release active state".to_owned());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Panic-on-failure entry point for provider test suites.
pub fn memory_strategy_conformance(contract: &MemoryProviderContract) {
    if let Err(errors) = check_memory_strategy_contract(contract) {
        panic!(
            "memory-strategy conformance FAILED for '{}':\n- {}",
            contract.provider_id,
            errors.join("\n- ")
        );
    }
}

/// Weights-free behavioral walk over every memory-strategy registration in an explicit catalog.
///
/// Static contract conformance runs for every registration. A contract that declares native/PiD
/// decode routes receives four admission probes: each route's own geometry must be accepted, and the
/// same geometry presented to the opposite route must be rejected. The matching-route controls keep
/// the rejection proof non-vacuous — an always-rejecting safety check does not conform.
pub fn check_memory_strategy_registry(
    registry: &ProviderRegistry,
    spec: &LoadSpec,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for registration in registry.memory_strategy_registrations() {
        check_memory_registration(registration, spec, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Panic-on-failure catalog entry point.
pub fn memory_strategy_registry_conformance(registry: &ProviderRegistry, spec: &LoadSpec) {
    if let Err(errors) = check_memory_strategy_registry(registry, spec) {
        panic!(
            "memory-strategy registry conformance FAILED:\n- {}",
            errors.join("\n- ")
        );
    }
}

fn check_memory_registration(
    registration: &MemoryRegistration,
    spec: &LoadSpec,
    errors: &mut Vec<String>,
) {
    let contract = match (registration.contract)(spec) {
        Ok(contract) => contract,
        Err(error) => {
            errors.push(format!(
                "{}: weights-free contract construction failed: {error}",
                registration.provider_id
            ));
            return;
        }
    };
    if contract.provider_id != registration.provider_id {
        errors.push(format!(
            "{}: registration returned contract for {:?}",
            registration.provider_id, contract.provider_id
        ));
    }
    if let Err(contract_errors) = check_memory_strategy_contract(&contract) {
        errors.extend(
            contract_errors
                .into_iter()
                .map(|error| format!("{}: {error}", registration.provider_id)),
        );
        return;
    }

    let Some(routes) = contract.pid_decode_routes.as_ref() else {
        return;
    };
    let Some(calibration) = contract.calibration.as_ref() else {
        errors.push(format!(
            "{}: PiD route conformance needs a calibration identity for a valid admission probe",
            registration.provider_id
        ));
        return;
    };
    let Some(&native_edge) = routes.native.tile_edges.first() else {
        return;
    };
    let Some(&pid_edge) = routes.pid.tile_edges.first() else {
        return;
    };

    for (label, edge, overlap, use_pid, expected) in [
        (
            "native geometry on the native route",
            native_edge,
            routes.native.tile_overlap,
            false,
            "accept",
        ),
        (
            "PiD geometry on the PiD route",
            pid_edge,
            routes.pid.tile_overlap,
            true,
            "accept",
        ),
        (
            "native geometry on the PiD route",
            native_edge,
            routes.native.tile_overlap,
            true,
            "reject",
        ),
        (
            "PiD geometry on the native route",
            pid_edge,
            routes.pid.tile_overlap,
            false,
            "reject",
        ),
    ] {
        let context = route_context(
            calibration.abi,
            &calibration.fingerprint,
            edge,
            overlap,
            use_pid,
        );
        let decision = (registration.safety_check)(&contract, &context);
        let conforms = matches!(
            (expected, &decision),
            ("accept", MemorySafetyDecision::Accept)
                | ("reject", MemorySafetyDecision::Reject { .. })
        );
        if !conforms {
            errors.push(format!(
                "{}: safety_check must {expected} {label}, got {decision:?}",
                registration.provider_id
            ));
        }
    }
}

fn route_context(
    calibration_abi: u32,
    calibration_fingerprint: &str,
    edge: u32,
    overlap: u32,
    use_pid: bool,
) -> MemoryRunContext {
    MemoryRunContext {
        selection: MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(edge),
                decode_overlap: Some(overlap),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        },
        calibration_abi,
        calibration_fingerprint: calibration_fingerprint.to_owned(),
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid,
        has_phases: true,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: u64::MAX,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 0,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "weights-free-registry-conformance".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        default_memory_strategy_safety_check, MemoryBackendRealization, MemoryCalibrationIdentity,
        MemoryDecodeRouteDomain, MemoryLifecycleCapabilities, MemoryParameterRanges,
        MemoryPidDecodeRoutes, MemoryStrategyCapability, WeightsSource,
    };

    fn backend() -> MemoryBackendRealization {
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        }
    }

    #[test]
    fn resident_only_compatibility_contract_conforms_without_claiming_optimization() {
        let contract = MemoryProviderContract::compatibility_default("legacy", backend());
        check_memory_strategy_contract(&contract).unwrap();
        assert!(contract.calibration.is_none());
    }

    #[test]
    fn malformed_strategy_table_is_reported() {
        let mut contract = MemoryProviderContract::compatibility_default("bad", backend());
        contract.strategies.pop();
        let errors = check_memory_strategy_contract(&contract).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("exactly once")));
    }

    fn pid_contract(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let mut contract = MemoryProviderContract::compatibility_default("pid-provider", backend());
        let bounded = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .unwrap();
        *bounded = MemoryStrategyCapability {
            strategy: MemoryStrategy::BoundedDecode,
            support: MemoryStrategySupport::Implemented,
            parameters: MemoryParameterRanges {
                decode_tile_edges: vec![2048, 512],
                decode_overlaps: vec![256, 64],
                ..Default::default()
            },
        };
        contract.pid_decode_routes = Some(MemoryPidDecodeRoutes {
            native: MemoryDecodeRouteDomain {
                tile_edges: vec![512],
                tile_overlap: 64,
            },
            pid: MemoryDecodeRouteDomain {
                tile_edges: vec![2048],
                tile_overlap: 256,
            },
        });
        contract.lifecycle = MemoryLifecycleCapabilities {
            decode_tiling: true,
            ..Default::default()
        };
        contract.calibration = Some(MemoryCalibrationIdentity::new("pid-provider-v1"));
        Ok(contract)
    }

    fn route_aware_safety(
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        let routes = contract.pid_decode_routes.as_ref().unwrap();
        let edge = context.selection.parameters.decode_tile_edge;
        let overlap = context.selection.parameters.decode_overlap;
        let domain = if context.use_pid {
            &routes.pid
        } else {
            &routes.native
        };
        if edge.is_some_and(|edge| domain.tile_edges.contains(&edge))
            && overlap == Some(domain.tile_overlap)
        {
            default_memory_strategy_safety_check(contract, context)
        } else {
            MemorySafetyDecision::Reject {
                reason: "cross-route geometry".to_owned(),
            }
        }
    }

    #[test]
    fn route_aware_registration_accepts_matching_routes_and_rejects_cross_routes() {
        let registration = MemoryRegistration {
            provider_id: "pid-provider",
            contract: pid_contract,
            safety_check: route_aware_safety,
        };
        let mut errors = Vec::new();
        check_memory_registration(
            &registration,
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            &mut errors,
        );
        assert_eq!(errors, Vec::<String>::new());
    }

    #[test]
    fn route_blind_safety_fails_both_cross_route_probes() {
        let registration = MemoryRegistration {
            provider_id: "pid-provider",
            contract: pid_contract,
            safety_check: default_memory_strategy_safety_check,
        };
        let mut errors = Vec::new();
        check_memory_registration(
            &registration,
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            &mut errors,
        );
        assert_eq!(errors.len(), 2, "{errors:#?}");
        assert!(errors
            .iter()
            .any(|error| error.contains("native geometry on the PiD route")));
        assert!(errors
            .iter()
            .any(|error| error.contains("PiD geometry on the native route")));
    }
}
