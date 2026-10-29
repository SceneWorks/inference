//! Per-registration and per-behavior-fixture checks the registry suites run.

use super::*;

pub(super) fn check_memory_registration(
    registration: &MemoryRegistration,
    contract_fixture: Option<&MemoryContractFixtureRegistration>,
    resident_only_witness: Option<&ResidentOnlyMemoryContractRegistration>,
    behavior: Option<&MemoryBehaviorRegistration>,
    spec: &LoadSpec,
    errors: &mut Vec<String>,
) {
    let contract_factory = contract_fixture
        .map(|fixture| fixture.contract)
        .or_else(|| resident_only_witness.map(|witness| witness.contract))
        .unwrap_or(registration.contract);
    // A catalog can legitimately compose providers with opposite materialization requirements.
    // The caller-supplied spec carries all common fixture axes, but one global load shape cannot be
    // valid for both a block-window image provider and an eager-only video provider. Resolve that
    // single axis per registration and keep the successful spec for every subsequent behavior and
    // safety probe; otherwise the conformance walk would either reject a valid mixed catalog or test
    // a contract against a different load identity than the one used to construct it.
    let mut alternate_spec = None;
    let contract = match contract_factory(spec) {
        Ok(contract) => contract,
        Err(primary_error) => {
            let mut alternate = spec.clone();
            alternate.load_shape = match spec.load_shape {
                gen_core::LoadShape::EagerMaterialization => {
                    gen_core::LoadShape::DeferredMaterialization
                }
                gen_core::LoadShape::DeferredMaterialization => {
                    gen_core::LoadShape::EagerMaterialization
                }
            };
            match contract_factory(&alternate) {
                Ok(contract) => {
                    alternate_spec = Some(alternate);
                    contract
                }
                Err(alternate_error) => {
                    errors.push(format!(
                        "{}: weights-free contract construction failed for both {:?} ({primary_error}) and {:?} ({alternate_error})",
                        registration.provider_id,
                        spec.load_shape,
                        alternate.load_shape,
                    ));
                    return;
                }
            }
        }
    };
    let spec = alternate_spec.as_ref().unwrap_or(spec);
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
    let Some(_) = contract.calibration.as_ref() else {
        errors.push(format!(
            "{}: PiD route conformance needs a calibration identity for a valid admission probe",
            registration.provider_id
        ));
        return;
    };
    let Some(behavior) = behavior else {
        errors.push(format!(
            "{}: decode-route conformance needs provider-owned behavior fixtures",
            registration.provider_id
        ));
        return;
    };
    let decode_contexts =
        match (behavior.valid_fixtures)(spec, &contract, MemoryStrategy::BoundedDecode) {
            Ok(fixtures) => fixtures
                .into_iter()
                .map(|fixture| fixture.context)
                .collect::<Vec<_>>(),
            Err(error) => {
                errors.push(format!(
                    "{}: decode-route conformance could not build provider-owned contexts: {error}",
                    registration.provider_id
                ));
                return;
            }
        };
    let mut edges = routes.native.tile_edges.clone();
    edges.extend_from_slice(&routes.pid.tile_edges);
    edges.sort_unstable();
    edges.dedup();
    let mut overlaps = vec![routes.native.tile_overlap, routes.pid.tile_overlap];
    overlaps.sort_unstable();
    overlaps.dedup();

    for use_pid in [false, true] {
        let Some(base_context) = decode_contexts
            .iter()
            .find(|context| context.use_pid == use_pid)
        else {
            errors.push(format!(
                "{}: decode-route conformance lacks a provider-owned {} context",
                registration.provider_id,
                if use_pid { "PiD" } else { "native" }
            ));
            continue;
        };
        let target = if use_pid { &routes.pid } else { &routes.native };
        let route_name = if use_pid { "PiD" } else { "native" };
        for &edge in &edges {
            for &overlap in &overlaps {
                let expected_accept =
                    target.tile_edges.contains(&edge) && overlap == target.tile_overlap;
                let context = decode_probe_context(base_context, edge, overlap);
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

pub(super) fn check_behavior(
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

pub(super) fn check_behavior_fixture(
    registration: &MemoryRegistration,
    behavior: &MemoryBehaviorRegistration,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    mut fixture: gen_core::MemoryBehaviorFixture,
    errors: &mut Vec<String>,
) {
    let fixture_spec = fixture.load_spec.as_ref().unwrap_or(spec);
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
    let decision = (registration.safety_check)(fixture_spec, contract, &fixture.context);
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
        (registration.safety_check)(fixture_spec, contract, &mutated_abi),
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
        (registration.safety_check)(fixture_spec, contract, &mutated_fingerprint),
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
        (registration.safety_check)(fixture_spec, contract, &mutated_shape),
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
        (registration.safety_check)(fixture_spec, contract, &tier),
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
        (registration.safety_check)(fixture_spec, contract, &over_budget),
        MemorySafetyDecision::Accept
    ) {
        errors.push(format!(
            "{}: {strategy:?} safety check is blind to an impossible budget",
            registration.provider_id
        ));
    }

    let expected_memory = contract.generation_memory(&fixture.context.selection);
    let mut scope = match (behavior.begin_request)(fixture_spec, contract, &fixture.context) {
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
    let expected_attention_chunk = contract
        .engages(strategy, MemoryStrategy::BoundedAttention)
        .then_some(fixture.context.selection.parameters.attention_chunk_size)
        .flatten();
    let configured_attention_chunk = fixture
        .request
        .memory
        .and_then(|memory| memory.attention_chunk_size);
    if configured_attention_chunk != expected_attention_chunk {
        errors.push(format!(
            "{}: {strategy:?} configured attention chunk {:?}, expected engaged carrier {:?}",
            registration.provider_id, configured_attention_chunk, expected_attention_chunk
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

pub(super) fn decode_probe_context(
    base: &MemoryRunContext,
    edge: u32,
    overlap: u32,
) -> MemoryRunContext {
    let mut context = base.clone();
    context.selection.strategy = MemoryStrategy::BoundedDecode;
    context.selection.parameters = MemoryStrategyParameters {
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        ..Default::default()
    };
    context
}
