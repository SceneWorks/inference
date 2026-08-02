//! Weights-free conformance checks for the shared memory-strategy provider contract.

use gen_core::{
    LoadSpec, MemoryBehaviorRegistration, MemoryBudget, MemoryCacheState, MemoryCleanupSemantics,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryPhase, MemoryProviderContract,
    MemoryRegistration, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemorySelection,
    MemoryStrategy, MemoryStrategyParameters, MemoryStrategySupport, Precision, ProviderRegistry,
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
        let behavior = registry
            .memory_behavior_registrations()
            .find(|behavior| behavior.provider_id == registration.provider_id);
        check_memory_registration(registration, behavior, spec, &mut errors);
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
    behavior: Option<&MemoryBehaviorRegistration>,
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

    let optimized = contract
        .strategies
        .iter()
        .filter(|capability| {
            capability.strategy.is_optimized()
                && matches!(capability.support, MemoryStrategySupport::Implemented)
        })
        .map(|capability| capability.strategy)
        .collect::<Vec<_>>();
    if !optimized.is_empty() && behavior.is_none() {
        errors.push(format!(
            "{}: optimized registration lacks a weights-free behavior seam",
            registration.provider_id
        ));
        return;
    }
    if let Some(behavior) = behavior {
        for strategy in optimized {
            check_behavior(registration, behavior, spec, &contract, strategy, errors);
        }
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
    let mut edges = routes.native.tile_edges.clone();
    edges.extend_from_slice(&routes.pid.tile_edges);
    edges.sort_unstable();
    edges.dedup();
    let mut overlaps = vec![routes.native.tile_overlap, routes.pid.tile_overlap];
    overlaps.sort_unstable();
    overlaps.dedup();

    for use_pid in [false, true] {
        let target = if use_pid { &routes.pid } else { &routes.native };
        let route_name = if use_pid { "PiD" } else { "native" };
        for &edge in &edges {
            for &overlap in &overlaps {
                let expected_accept =
                    target.tile_edges.contains(&edge) && overlap == target.tile_overlap;
                let context = route_context(
                    calibration.abi,
                    &calibration.fingerprint,
                    calibration.load_shape,
                    edge,
                    overlap,
                    use_pid,
                );
                let decision = (registration.safety_check)(spec, &contract, &context);
                let conforms = matches!(
                    (expected_accept, &decision),
                    (true, MemorySafetyDecision::Accept)
                        | (false, MemorySafetyDecision::Reject { .. })
                );
                if !conforms {
                    errors.push(format!(
                        "{}: safety_check must {} edge {edge} + overlap {overlap} on {route_name} route, got {decision:?}",
                        registration.provider_id,
                        if expected_accept { "accept" } else { "reject" }
                    ));
                }
            }
        }
    }
}

fn check_behavior(
    registration: &MemoryRegistration,
    behavior: &MemoryBehaviorRegistration,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    errors: &mut Vec<String>,
) {
    let fixtures = match (behavior.valid_fixtures)(spec, contract, strategy) {
        Ok(fixtures) if !fixtures.is_empty() => fixtures,
        Ok(_) => {
            errors.push(format!(
                "{}: implemented {strategy:?} has no provider-owned valid context",
                registration.provider_id
            ));
            return;
        }
        Err(error) => {
            errors.push(format!(
                "{}: valid {strategy:?} behavior fixture failed: {error}",
                registration.provider_id
            ));
            return;
        }
    };
    let needs_both_routes = contract.pid_decode_routes.is_some()
        && contract.engages(strategy, MemoryStrategy::BoundedDecode);
    if needs_both_routes {
        for use_pid in [false, true] {
            if !fixtures
                .iter()
                .any(|fixture| fixture.context.use_pid == use_pid)
            {
                errors.push(format!(
                    "{}: implemented {strategy:?} lacks a provider-owned {} route fixture",
                    registration.provider_id,
                    if use_pid { "PiD" } else { "native" }
                ));
            }
        }
    }
    for fixture in fixtures {
        check_behavior_fixture(
            registration,
            behavior,
            spec,
            contract,
            strategy,
            fixture,
            errors,
        );
    }
}

fn check_behavior_fixture(
    registration: &MemoryRegistration,
    behavior: &MemoryBehaviorRegistration,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    mut fixture: gen_core::MemoryBehaviorFixture,
    errors: &mut Vec<String>,
) {
    if fixture.context.selection.strategy != strategy {
        errors.push(format!(
            "{}: {strategy:?} fixture selected {:?}",
            registration.provider_id, fixture.context.selection.strategy
        ));
        return;
    }
    if fixture.context.predicted_peak_bytes == 0
        || fixture.context.predicted_peak_bytes == u64::MAX
        || fixture.context.budget.total_bytes == u64::MAX
        || !fixture
            .context
            .budget
            .fits(fixture.context.predicted_peak_bytes)
    {
        errors.push(format!(
            "{}: {strategy:?} fixture must declare a finite fitting budget and non-zero peak",
            registration.provider_id
        ));
        return;
    }
    let decision = (registration.safety_check)(spec, contract, &fixture.context);
    if !matches!(decision, MemorySafetyDecision::Accept) {
        errors.push(format!(
            "{}: implemented {strategy:?} rejects its provider-owned valid context: {decision:?}",
            registration.provider_id
        ));
        return;
    }

    let mut mutated_abi = fixture.context.clone();
    mutated_abi.calibration_abi = mutated_abi.calibration_abi.wrapping_add(1);
    if matches!(
        (registration.safety_check)(spec, contract, &mutated_abi),
        MemorySafetyDecision::Accept
    ) {
        errors.push(format!(
            "{}: {strategy:?} safety check is blind to mutated calibration ABI",
            registration.provider_id
        ));
    }
    let mut mutated_fingerprint = fixture.context.clone();
    mutated_fingerprint
        .calibration_fingerprint
        .push_str("-mutated");
    if matches!(
        (registration.safety_check)(spec, contract, &mutated_fingerprint),
        MemorySafetyDecision::Accept
    ) {
        errors.push(format!(
            "{}: {strategy:?} safety check is blind to mutated calibration fingerprint",
            registration.provider_id
        ));
    }
    let mut mutated_shape = fixture.context.clone();
    mutated_shape.load_shape = match mutated_shape.load_shape {
        gen_core::LoadShape::EagerMaterialization => gen_core::LoadShape::DeferredMaterialization,
        gen_core::LoadShape::DeferredMaterialization => gen_core::LoadShape::EagerMaterialization,
    };
    if matches!(
        (registration.safety_check)(spec, contract, &mutated_shape),
        MemorySafetyDecision::Accept
    ) {
        errors.push(format!(
            "{}: {strategy:?} safety check is blind to mutated calibration load shape",
            registration.provider_id
        ));
    }

    let mut tier = fixture.context.clone();
    tier.selection.tier.precision = match tier.selection.tier.precision {
        Precision::Bf16 => Precision::Fp32,
        _ => Precision::Bf16,
    };
    if matches!(
        (registration.safety_check)(spec, contract, &tier),
        MemorySafetyDecision::Accept
    ) {
        errors.push(format!(
            "{}: {strategy:?} safety check is blind to a numeric-tier mutation",
            registration.provider_id
        ));
    }

    let mut over_budget = fixture.context.clone();
    over_budget.predicted_peak_bytes = u64::MAX;
    over_budget.budget = MemoryBudget {
        total_bytes: 1024,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    if matches!(
        (registration.safety_check)(spec, contract, &over_budget),
        MemorySafetyDecision::Accept
    ) {
        errors.push(format!(
            "{}: {strategy:?} safety check is blind to an impossible budget",
            registration.provider_id
        ));
    }

    let expected_memory = contract.generation_memory(&fixture.context.selection);
    let mut scope = match (behavior.begin_request)(spec, contract, &fixture.context) {
        Ok(Some(scope)) => scope,
        Ok(None) => {
            errors.push(format!(
                "{}: implemented {strategy:?} begin_request returned no scope",
                registration.provider_id
            ));
            return;
        }
        Err(error) => {
            errors.push(format!(
                "{}: implemented {strategy:?} begin_request rejected valid context: {error}",
                registration.provider_id
            ));
            return;
        }
    };
    if let Err(error) = scope.configure_request(&mut fixture.request) {
        errors.push(format!(
            "{}: {strategy:?} configure_request failed: {error}",
            registration.provider_id
        ));
        return;
    }
    if fixture.request.memory != expected_memory {
        errors.push(format!(
            "{}: {strategy:?} configured {:?}, expected canonical {:?}",
            registration.provider_id, fixture.request.memory, expected_memory
        ));
    }
    if let Err(error) = scope.enter_phase(MemoryPhase::Denoise) {
        errors.push(format!(
            "{}: {strategy:?} enter_phase rejected valid scope: {error}",
            registration.provider_id
        ));
    }
    if contract.engages(strategy, MemoryStrategy::BoundedDecode) {
        let parameters = fixture.context.selection.parameters;
        if let (Some(edge), Some(overlap)) =
            (parameters.decode_tile_edge, parameters.decode_overlap)
        {
            if let Err(error) = scope.configure_decode(edge, overlap, fixture.context.geometry) {
                errors.push(format!(
                    "{}: {strategy:?} configure_decode rejected declared parameters: {error}",
                    registration.provider_id
                ));
            }
        }
    }
    if contract.engages(strategy, MemoryStrategy::BoundedAttention) {
        if let Some(chunk) = fixture.context.selection.parameters.attention_chunk_size {
            if let Err(error) = scope.configure_attention(chunk) {
                errors.push(format!(
                    "{}: {strategy:?} configure_attention rejected declared parameter: {error}",
                    registration.provider_id
                ));
            }
        }
    }
    if contract.engages(strategy, MemoryStrategy::BoundedTransformerResidency) {
        if let Some(window) = fixture.context.selection.parameters.transformer_window_size {
            if let Err(error) = scope.materialize_transformer_window(0, window) {
                errors.push(format!(
                    "{}: {strategy:?} transformer window rejected declared parameter: {error}",
                    registration.provider_id
                ));
            }
        }
    }
    if let Err(error) = scope.leave_phase(MemoryPhase::Denoise) {
        errors.push(format!(
            "{}: {strategy:?} leave_phase rejected valid scope: {error}",
            registration.provider_id
        ));
    }
    if let Err(error) = scope.finish(MemoryRunOutcome::Complete) {
        errors.push(format!(
            "{}: {strategy:?} first finish failed: {error}",
            registration.provider_id
        ));
        return;
    }
    if scope.finish(MemoryRunOutcome::Complete).is_ok() {
        errors.push(format!(
            "{}: {strategy:?} second finish succeeded",
            registration.provider_id
        ));
    }
    for (hook, result) in [
        ("enter_phase", scope.enter_phase(MemoryPhase::Denoise)),
        ("leave_phase", scope.leave_phase(MemoryPhase::Denoise)),
        (
            "configure_decode",
            scope.configure_decode(1, 0, fixture.context.geometry),
        ),
        ("configure_attention", scope.configure_attention(1)),
        (
            "materialize_transformer_window",
            scope.materialize_transformer_window(0, 1),
        ),
    ] {
        if result.is_ok() {
            errors.push(format!(
                "{}: {strategy:?} {hook} succeeded after finish",
                registration.provider_id
            ));
        }
    }
}

fn route_context(
    calibration_abi: u32,
    calibration_fingerprint: &str,
    load_shape: gen_core::LoadShape,
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
        load_shape,
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
            total_bytes: 8 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1024 * 1024 * 1024,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "weights-free-registry-conformance".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        standard_memory_strategy_safety_check, MemoryBackendRealization,
        MemoryBehaviorBeginRequest, MemoryCalibrationIdentity, MemoryDecodeRouteDomain,
        MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPidDecodeRoutes,
        MemoryRequestScope, MemoryRunOutcome, MemoryStrategyCapability, WeightsSource,
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
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "pid-provider-v1",
            contract.load_shape,
        ));
        Ok(contract)
    }

    fn route_aware_safety(
        _spec: &LoadSpec,
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
            standard_memory_strategy_safety_check(
                contract,
                context,
                Some(MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                }),
                None,
            )
        } else {
            MemorySafetyDecision::Reject {
                reason: "cross-route geometry".to_owned(),
            }
        }
    }

    struct FixtureScope {
        memory: Option<gen_core::GenerationMemory>,
        finished: bool,
    }

    impl MemoryRequestScope for FixtureScope {
        fn configure_request(
            &mut self,
            request: &mut gen_core::GenerationRequest,
        ) -> gen_core::Result<()> {
            if self.finished {
                return Err(gen_core::Error::Msg("finished".into()));
            }
            request.memory = self.memory;
            Ok(())
        }
        fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            if self.finished {
                Err(gen_core::Error::Msg("finished".into()))
            } else {
                Ok(())
            }
        }
        fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
            self.enter_phase(phase)
        }
        fn configure_decode(
            &mut self,
            _edge: u32,
            _overlap: u32,
            _geometry: MemoryGeometry,
        ) -> gen_core::Result<()> {
            self.enter_phase(MemoryPhase::Decode)
        }
        fn configure_attention(&mut self, _chunk: u32) -> gen_core::Result<()> {
            self.enter_phase(MemoryPhase::Denoise)
        }
        fn materialize_transformer_window(
            &mut self,
            _first: u32,
            _count: u32,
        ) -> gen_core::Result<()> {
            self.enter_phase(MemoryPhase::Denoise)
        }
        fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
            if self.finished {
                Err(gen_core::Error::Msg("finished".into()))
            } else {
                self.finished = true;
                Ok(())
            }
        }
    }

    fn pid_fixture(
        _spec: &LoadSpec,
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
        if strategy != MemoryStrategy::BoundedDecode {
            return Ok(Vec::new());
        }
        let routes = contract.pid_decode_routes.as_ref().unwrap();
        let calibration = contract.calibration.as_ref().unwrap();
        Ok(vec![
            gen_core::MemoryBehaviorFixture::new(route_context(
                calibration.abi,
                &calibration.fingerprint,
                calibration.load_shape,
                routes.native.tile_edges[0],
                routes.native.tile_overlap,
                false,
            )),
            gen_core::MemoryBehaviorFixture::new(route_context(
                calibration.abi,
                &calibration.fingerprint,
                calibration.load_shape,
                routes.pid.tile_edges[0],
                routes.pid.tile_overlap,
                true,
            )),
        ])
    }

    fn pid_begin(
        _spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
        Ok(Some(Box::new(FixtureScope {
            memory: contract.generation_memory(&context.selection),
            finished: false,
        })))
    }

    const PID_BEHAVIOR: MemoryBehaviorRegistration = MemoryBehaviorRegistration {
        provider_id: "pid-provider",
        valid_fixtures: pid_fixture,
        begin_request: pid_begin,
    };

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
            Some(&PID_BEHAVIOR),
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
            safety_check: |_spec, contract, context| {
                standard_memory_strategy_safety_check(
                    contract,
                    context,
                    Some(MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: None,
                        component_precision_floors: &[],
                    }),
                    None,
                )
            },
        };
        let mut errors = Vec::new();
        check_memory_registration(
            &registration,
            Some(&PID_BEHAVIOR),
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            &mut errors,
        );
        assert_eq!(errors.len(), 6, "{errors:#?}");
        assert!(errors
            .iter()
            .any(|error| error.contains("edge 512 + overlap 256 on native route")));
        assert!(errors
            .iter()
            .any(|error| error.contains("edge 2048 + overlap 64 on PiD route")));
    }

    fn always_accept(
        _spec: &LoadSpec,
        _contract: &MemoryProviderContract,
        _context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        MemorySafetyDecision::Accept
    }

    fn always_reject(
        _spec: &LoadSpec,
        _contract: &MemoryProviderContract,
        _context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        MemorySafetyDecision::Reject {
            reason: "always".into(),
        }
    }

    fn handshake_blind(
        _spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        let mut trusted = context.clone();
        let calibration = contract.calibration.as_ref().unwrap();
        trusted.calibration_abi = calibration.abi;
        trusted.calibration_fingerprint = calibration.fingerprint.clone();
        route_aware_safety(
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            contract,
            &trusted,
        )
    }

    fn tier_blind(
        _spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        let routes = contract.pid_decode_routes.as_ref().unwrap();
        let domain = if context.use_pid {
            &routes.pid
        } else {
            &routes.native
        };
        if context
            .selection
            .parameters
            .decode_tile_edge
            .is_some_and(|edge| domain.tile_edges.contains(&edge))
            && context.selection.parameters.decode_overlap == Some(domain.tile_overlap)
        {
            gen_core::default_memory_strategy_safety_check(contract, context)
        } else {
            MemorySafetyDecision::Reject {
                reason: "route".into(),
            }
        }
    }

    fn budget_blind(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        let mut admitted = context.clone();
        admitted.predicted_peak_bytes = 0;
        admitted.budget.total_bytes = u64::MAX;
        route_aware_safety(spec, contract, &admitted)
    }

    fn overlap_blind(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        let mut admitted = context.clone();
        let routes = contract.pid_decode_routes.as_ref().unwrap();
        admitted.selection.parameters.decode_overlap = Some(if admitted.use_pid {
            routes.pid.tile_overlap
        } else {
            routes.native.tile_overlap
        });
        route_aware_safety(spec, contract, &admitted)
    }

    fn wrong_begin(
        _spec: &LoadSpec,
        _contract: &MemoryProviderContract,
        _context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
        Ok(Some(Box::new(FixtureScope {
            memory: None,
            finished: false,
        })))
    }

    fn no_scope(
        _spec: &LoadSpec,
        _contract: &MemoryProviderContract,
        _context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
        Ok(None)
    }

    fn native_only_begin(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
        if context.use_pid {
            Ok(None)
        } else {
            pid_begin(spec, contract, context)
        }
    }

    struct DoubleFinishScope {
        memory: Option<gen_core::GenerationMemory>,
    }
    impl MemoryRequestScope for DoubleFinishScope {
        fn configure_request(
            &mut self,
            request: &mut gen_core::GenerationRequest,
        ) -> gen_core::Result<()> {
            request.memory = self.memory;
            Ok(())
        }
        fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            Err(gen_core::Error::Msg("finished".into()))
        }
        fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            Err(gen_core::Error::Msg("finished".into()))
        }
        fn configure_decode(
            &mut self,
            _edge: u32,
            _overlap: u32,
            _geometry: MemoryGeometry,
        ) -> gen_core::Result<()> {
            Err(gen_core::Error::Msg("finished".into()))
        }
        fn configure_attention(&mut self, _chunk: u32) -> gen_core::Result<()> {
            Err(gen_core::Error::Msg("finished".into()))
        }
        fn materialize_transformer_window(
            &mut self,
            _first: u32,
            _count: u32,
        ) -> gen_core::Result<()> {
            Err(gen_core::Error::Msg("finished".into()))
        }
        fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
            Ok(())
        }
    }
    fn double_finish_begin(
        _spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
        Ok(Some(Box::new(DoubleFinishScope {
            memory: contract.generation_memory(&context.selection),
        })))
    }

    struct PostFinishHooksScope {
        memory: Option<gen_core::GenerationMemory>,
        finished: bool,
    }
    impl MemoryRequestScope for PostFinishHooksScope {
        fn configure_request(
            &mut self,
            request: &mut gen_core::GenerationRequest,
        ) -> gen_core::Result<()> {
            request.memory = self.memory;
            Ok(())
        }
        fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_decode(
            &mut self,
            _edge: u32,
            _overlap: u32,
            _geometry: MemoryGeometry,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _chunk: u32) -> gen_core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(
            &mut self,
            _first: u32,
            _count: u32,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
            if self.finished {
                Err(gen_core::Error::Msg("finished".into()))
            } else {
                self.finished = true;
                Ok(())
            }
        }
    }
    fn post_finish_hooks_begin(
        _spec: &LoadSpec,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
        Ok(Some(Box::new(PostFinishHooksScope {
            memory: contract.generation_memory(&context.selection),
            finished: false,
        })))
    }

    fn errors_for(
        safety_check: fn(
            &LoadSpec,
            &MemoryProviderContract,
            &MemoryRunContext,
        ) -> MemorySafetyDecision,
        begin_request: MemoryBehaviorBeginRequest,
    ) -> Vec<String> {
        let registration = MemoryRegistration {
            provider_id: "pid-provider",
            contract: pid_contract,
            safety_check,
        };
        let behavior = MemoryBehaviorRegistration {
            provider_id: "pid-provider",
            valid_fixtures: pid_fixture,
            begin_request,
        };
        let mut errors = Vec::new();
        check_memory_registration(
            &registration,
            Some(&behavior),
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            &mut errors,
        );
        errors
    }

    #[test]
    fn overlap_blind_admission_is_killed_by_the_edge_overlap_cross_product() {
        let errors = errors_for(overlap_blind, pid_begin);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("overlap 256 on native route")),
            "{errors:#?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("overlap 64 on PiD route")),
            "{errors:#?}"
        );
    }

    #[test]
    fn native_only_scope_is_killed_by_the_pid_lifecycle_fixture() {
        let errors = errors_for(route_aware_safety, native_only_begin);
        assert!(
            errors.iter().any(|error| {
                error.contains("returned no scope") && error.contains("BoundedDecode")
            }),
            "{errors:#?}"
        );
    }

    #[test]
    fn rejecting_a_second_finish_does_not_mask_succeeding_post_finish_hooks() {
        let errors = errors_for(route_aware_safety, post_finish_hooks_begin);
        assert!(
            !errors
                .iter()
                .any(|error| error.contains("second finish succeeded")),
            "{errors:#?}"
        );
        for hook in [
            "enter_phase",
            "leave_phase",
            "configure_decode",
            "configure_attention",
            "materialize_transformer_window",
        ] {
            assert!(
                errors.iter().any(|error| {
                    error.contains(hook) && error.contains("succeeded after finish")
                }),
                "{hook} was not independently killed: {errors:#?}"
            );
        }
    }

    #[test]
    fn mutation_probes_reject_deliberately_broken_adopters() {
        let cases = [
            (
                "always accept",
                errors_for(always_accept, pid_begin),
                "mutated calibration ABI",
            ),
            (
                "always reject",
                errors_for(always_reject, pid_begin),
                "rejects its provider-owned valid context",
            ),
            (
                "handshake blind",
                errors_for(handshake_blind, pid_begin),
                "mutated calibration ABI",
            ),
            (
                "tier blind",
                errors_for(tier_blind, pid_begin),
                "numeric-tier mutation",
            ),
            (
                "budget blind",
                errors_for(budget_blind, pid_begin),
                "impossible budget",
            ),
            (
                "overlap blind",
                errors_for(overlap_blind, pid_begin),
                "overlap 256 on native route",
            ),
            (
                "wrong translation",
                errors_for(route_aware_safety, wrong_begin),
                "configured None",
            ),
            (
                "no scope",
                errors_for(route_aware_safety, no_scope),
                "returned no scope",
            ),
            (
                "native-only scope",
                errors_for(route_aware_safety, native_only_begin),
                "returned no scope",
            ),
            (
                "double finish",
                errors_for(route_aware_safety, double_finish_begin),
                "second finish succeeded",
            ),
            (
                "post-finish hooks",
                errors_for(route_aware_safety, post_finish_hooks_begin),
                "succeeded after finish",
            ),
        ];
        for (name, errors, expected) in cases {
            assert!(
                errors.iter().any(|error| error.contains(expected)),
                "{name} was not killed by {expected:?}: {errors:#?}"
            );
        }
    }
}
