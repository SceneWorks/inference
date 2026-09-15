//! Weights-free conformance checks for the shared memory-strategy provider contract.

use gen_core::{
    LoadSpec, MemoryBehaviorRegistration, MemoryBudget, MemoryCleanupSemantics,
    MemoryContractFixtureRegistration, MemoryPhase, MemoryProviderContract, MemoryRegistration,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyParameters, MemoryStrategySupport, Precision, ProviderRegistry,
    ResidentOnlyMemoryContractRegistration,
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

/// Check the **E1 asset-facts half** of [`check_memory_contract_facts`] on its own.
///
/// This is separately public because a weights-free contract cannot satisfy E2 — it has no
/// `config.json` to read, so `MemoryArchitectureFacts::default()` is its honest state — while its
/// byte decomposition is still a claim that must hold. Such a test calls this entry point and
/// asserts `architecture_facts.is_empty()` alongside it.
///
/// Two defects are rejected:
///
/// 1. **A base total that is not its own decomposition.** `asset_facts.base_bytes` must equal
///    `conditioning_bytes + transformer_bytes + decoder_bytes` exactly. Auxiliary residency is
///    declared once in `overlay_bytes` and is never folded into `base_bytes` as well. A component
///    sum that overflows `u64` is reported rather than saturated: saturation would let a broken
///    `base_bytes` of `u64::MAX` match.
/// 2. **One total repeated in two component fields.** A provider that cannot price a component
///    separately must not paper over it by copying another component's byte count; a repeated
///    non-zero total is a dishonest decomposition even when the sum happens to add up.
pub fn check_memory_contract_asset_facts(
    contract: &MemoryProviderContract,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let facts = &contract.asset_facts;

    match facts
        .conditioning_bytes
        .checked_add(facts.transformer_bytes)
        .and_then(|bytes| bytes.checked_add(facts.decoder_bytes))
    {
        Some(decomposed) if decomposed != facts.base_bytes => {
            errors.push(format!(
                "asset_facts.base_bytes ({}) must equal conditioning ({}) + transformer ({}) + decoder ({}) = {decomposed}",
                facts.base_bytes, facts.conditioning_bytes, facts.transformer_bytes, facts.decoder_bytes
            ));
        }
        None => errors.push("base component byte sum overflow".to_owned()),
        _ => {}
    }

    for (left_name, left, right_name, right) in [
        (
            "conditioning_bytes",
            facts.conditioning_bytes,
            "transformer_bytes",
            facts.transformer_bytes,
        ),
        (
            "conditioning_bytes",
            facts.conditioning_bytes,
            "decoder_bytes",
            facts.decoder_bytes,
        ),
        (
            "transformer_bytes",
            facts.transformer_bytes,
            "decoder_bytes",
            facts.decoder_bytes,
        ),
    ] {
        if left != 0 && left == right {
            errors.push(format!(
                "asset_facts.{left_name} and asset_facts.{right_name} repeat the same total ({left}); \
                 a component that cannot be priced separately must not borrow another component's bytes"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Panic-on-failure entry point for [`check_memory_contract_asset_facts`].
pub fn assert_memory_contract_asset_facts_conform(contract: &MemoryProviderContract) {
    if let Err(errors) = check_memory_contract_asset_facts(contract) {
        panic!(
            "memory-contract asset-facts conformance FAILED for '{}':\n- {}",
            contract.provider_id,
            errors.join("\n- ")
        );
    }
}

/// Check that a contract's provider-owned *facts* are honest (epic SC-22657, E1 + E2).
///
/// This is deliberately a second, opt-in check rather than an addition to
/// [`check_memory_strategy_contract`]: providers adopt the facts axes one at a time, and a provider
/// which has not adopted them yet must keep passing the shared conformance walk unchanged.
///
/// The E1 byte-decomposition defects are delegated to [`check_memory_contract_asset_facts`]. On top
/// of them this rejects:
///
/// **E2 — no declared architecture facts on a lifecycle-phase provider.** A provider publishing
/// [`gen_core::MemoryFormulaKind::PhaseEnvelope`] or
/// [`gen_core::MemoryFormulaKind::ComponentPhaseEnvelope`] claims per-phase activation behavior,
/// which no caller can estimate from bytes alone. Such a contract must declare at least one axis
/// derived from a component config — parsed from the snapshot, or mirrored as a crate constant.
/// `activation_dtype_width` does not count: providers emit it from a crate-wide dtype constant, so
/// it is present whether or not any geometry was stated — see
/// [`gen_core::MemoryArchitectureFacts::has_declared_architecture_axis`].
///
/// A `Some(0)` on any architecture axis is rejected as well: an absent axis is `None`, and a zero
/// silently zeroes any activation estimate that multiplies by it.
///
/// **Which contracts this applies to is backend-dependent, so it is not the registry-wide gate.**
/// A Candle provider derives its axes only once a snapshot root exists, so its weights-free
/// contract publishes `MemoryArchitectureFacts::default()` and the E2 arm here would flag it —
/// call [`check_memory_contract_asset_facts`] and assert `architecture_facts.is_empty()` on that
/// path instead, and run this only against a contract built for a materialized snapshot. An MLX
/// provider mirrors compile-time presets, so its weights-free contract already carries axes and
/// passes this unchanged. [`check_memory_contract_surface_registry_facts`] is where that split is
/// resolved per backend for a whole registry.
pub fn check_memory_contract_facts(contract: &MemoryProviderContract) -> Result<(), Vec<String>> {
    let mut errors = check_memory_contract_asset_facts(contract)
        .err()
        .unwrap_or_default();

    let phase_formula = matches!(
        contract.formula,
        gen_core::MemoryFormulaKind::PhaseEnvelope { .. }
            | gen_core::MemoryFormulaKind::ComponentPhaseEnvelope { .. }
    );
    if phase_formula && !contract.architecture_facts.has_declared_architecture_axis() {
        errors.push(
            "a provider publishing a lifecycle-phase formula must declare at least one \
             config-derived architecture fact; activation_dtype_width alone is a crate-wide \
             compile-time constant, not evidence that any component geometry was stated"
                .to_owned(),
        );
    }
    for axis in contract.architecture_facts.zero_valued_axes() {
        errors.push(format!(
            "architecture_facts.{axis} is Some(0); a structurally absent axis is declared None, never zero"
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Panic-on-failure entry point for [`check_memory_contract_facts`].
pub fn assert_memory_contract_facts_conform(contract: &MemoryProviderContract) {
    if let Err(errors) = check_memory_contract_facts(contract) {
        panic!(
            "memory-contract facts conformance FAILED for '{}':\n- {}",
            contract.provider_id,
            errors.join("\n- ")
        );
    }
}

/// Weights-free behavioral walk over every memory-strategy registration in an explicit catalog.
///
/// Static contract conformance runs for every registration. A contract that declares native/PiD
/// decode routes receives four admission probes, starting from its provider-owned native/PiD
/// behavior contexts: each route's own geometry must be accepted, and the same geometry presented to
/// the opposite route must be rejected. Using those exact contexts keeps mode, reference, phase, and
/// overlay axes truthful. The matching-route controls keep the rejection proof non-vacuous — an
/// always-rejecting safety check does not conform.
pub fn check_memory_strategy_registry(
    registry: &ProviderRegistry,
    spec: &LoadSpec,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for registration in registry.memory_strategy_registrations() {
        let contract_fixture = registry
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == registration.provider_id);
        let resident_only_witness = registry
            .resident_only_memory_contract_registrations()
            .find(|witness| witness.provider_id == registration.provider_id);
        let behavior = registry
            .memory_behavior_registrations()
            .find(|behavior| behavior.provider_id == registration.provider_id);
        check_memory_registration(
            registration,
            contract_fixture,
            resident_only_witness,
            behavior,
            spec,
            &mut errors,
        );
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

/// Exhaustive, caller-independent conformance over every registered contract surface.
///
/// Unlike [`check_memory_strategy_registry`], this accepts no `LoadSpec`: the paired provider
/// fixture owns the complete registry-load witness matrix. Missing fixtures, duplicate selectors,
/// construction errors, and provider-id drift are reported by
/// [`ProviderRegistry::memory_contract_surfaces`] before static contract conformance runs.
pub fn check_memory_contract_surface_registry(
    registry: &ProviderRegistry,
) -> Result<(), Vec<String>> {
    let surfaces = registry
        .memory_contract_surfaces()
        .map_err(|error| vec![error.to_string()])?;
    let mut errors = Vec::new();
    for surface in surfaces {
        if let Err(contract_errors) = check_memory_strategy_contract(&surface.contract) {
            errors.extend(contract_errors.into_iter().map(|error| {
                format!(
                    "{} [{}]: {error}",
                    surface.contract.provider_id,
                    surface.selector.id()
                )
            }));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Panic-on-failure entry point for complete catalog contract-surface inventories.
pub fn memory_contract_surface_registry_conformance(registry: &ProviderRegistry) {
    if let Err(errors) = check_memory_contract_surface_registry(registry) {
        panic!(
            "memory-contract surface registry conformance FAILED:\n- {}",
            errors.join("\n- ")
        );
    }
}

/// A materialized snapshot root to re-derive one provider's architecture facts against.
///
/// Supplied by the caller of [`check_memory_contract_surface_registry_facts`] for the providers
/// whose own test fixtures can stand one up. Returning `None` for a provider means "this registry
/// has no root to offer here", and that provider is checked on the weights-free arm only.
pub type MaterializedRootLookup<'a> = &'a dyn Fn(&str) -> Option<std::path::PathBuf>;

/// What a registry-wide facts walk actually covered, so its caller can refuse a vacuous run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryContractSurfaceFactsCoverage {
    /// Contract surfaces walked on the weights-free arm.
    pub surfaces_checked: usize,
    /// Distinct providers re-derived against a caller-supplied materialized snapshot root and
    /// required to declare an architecture axis.
    pub materialized_providers_checked: usize,
}

/// Registry-wide facts walk over every registered contract surface (SC-22657, E1 + E2).
///
/// # What every surface is held to
///
/// The weights-free surface is built with no snapshot on disk, so [`check_memory_contract_facts`]
/// would wrongly flag a lifecycle-phase provider on this path. Its byte decomposition still holds,
/// so [`check_memory_contract_asset_facts`] runs on every surface, and a `Some(0)` on any axis is
/// rejected everywhere: an absent axis is `None`.
///
/// # The weights-free architecture rule is keyed on the backend
///
/// Whether a weights-free surface may declare an architecture axis is **not** a single rule; it
/// depends on where that backend's providers get their geometry, so
/// [`gen_core::MemoryBackendRealization`] selects between two:
///
/// * **[`gen_core::MemoryBackendRealization::CandleCuda`] — must publish
///   [`gen_core::MemoryArchitectureFacts::default`].** A Candle provider's axes are gated on
///   `candle_gen::architecture_facts::snapshot_root`, which yields nothing for the sentinel path the
///   registry builds surfaces against. An axis appearing here therefore did *not* come from a
///   component config; it was fabricated from the provider id, and the estimate built on it would
///   describe whatever model the id was assumed to name rather than the one on disk.
/// * **[`gen_core::MemoryBackendRealization::MlxMetal`] — must declare at least one axis.** An MLX
///   provider builds its geometry from compile-time preset constants that exist before any snapshot
///   does, so there is no snapshot read to wait for and nothing is fabricated by stating them. The
///   silent state worth catching on that backend is the opposite one: a surface that declares
///   nothing has skipped facts it already holds, so the *absence* of an axis is the defect.
///
/// `activation_dtype_width` is exempt from both arms — see
/// [`gen_core::MemoryArchitectureFacts::has_declared_architecture_axis`], which excludes it: it is a
/// crate-wide dtype constant, so it neither proves a Candle surface read a config nor counts as an
/// MLX surface having stated its geometry.
///
/// # The materialized arm
///
/// The weights-free walk alone cannot see whether a Candle provider *would* derive anything: a
/// provider that returned `MemoryArchitectureFacts::default()` unconditionally passes it. When
/// `materialized_root` yields a root for a provider, this rebuilds that provider's contract through
/// its real [`gen_core::MemoryRegistration::contract`] factory against that root and requires
/// [`gen_core::MemoryArchitectureFacts::has_declared_architecture_axis`] — which is exactly the
/// assertion the unconditional-`default()` mutation fails.
pub fn check_memory_contract_surface_registry_facts(
    registry: &ProviderRegistry,
    materialized_root: Option<MaterializedRootLookup<'_>>,
) -> Result<MemoryContractSurfaceFactsCoverage, Vec<String>> {
    let surfaces = registry
        .memory_contract_surfaces()
        .map_err(|error| vec![error.to_string()])?;
    let mut errors = Vec::new();
    let mut coverage = MemoryContractSurfaceFactsCoverage::default();
    let mut materialized_seen = std::collections::BTreeSet::new();
    for surface in surfaces {
        coverage.surfaces_checked += 1;
        let label = format!(
            "{} [{}]",
            surface.contract.provider_id,
            surface.selector.id()
        );
        if let Err(contract_errors) = check_memory_contract_asset_facts(&surface.contract) {
            errors.extend(
                contract_errors
                    .into_iter()
                    .map(|error| format!("{label}: {error}")),
            );
        }
        let declares = surface
            .contract
            .architecture_facts
            .has_declared_architecture_axis();
        match surface.contract.backend {
            gen_core::MemoryBackendRealization::CandleCuda { .. } if declares => {
                errors.push(format!(
                    "{label}: a Candle weights-free contract surface must publish \
                     MemoryArchitectureFacts::default(), but this one declares an architecture \
                     axis; Candle axes are gated on a materialized snapshot root, so an axis here \
                     was inferred from the provider id rather than from a component config"
                ));
            }
            gen_core::MemoryBackendRealization::MlxMetal { .. } if !declares => {
                errors.push(format!(
                    "{label}: an MLX weights-free contract surface must declare at least one \
                     architecture axis; MLX geometry is mirrored from compile-time preset \
                     constants that exist before any snapshot does, so declaring nothing withholds \
                     facts the provider already holds (activation_dtype_width does not count)"
                ));
            }
            _ => {}
        }
        for axis in surface.contract.architecture_facts.zero_valued_axes() {
            errors.push(format!(
                "{label}: architecture_facts.{axis} is Some(0); a structurally absent axis is \
                 declared None, never zero"
            ));
        }

        let Some(lookup) = materialized_root else {
            continue;
        };
        if !materialized_seen.insert(surface.contract.provider_id.clone()) {
            continue;
        }
        let Some(root) = lookup(&surface.contract.provider_id) else {
            continue;
        };
        let Some(registration) = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == surface.contract.provider_id)
        else {
            errors.push(format!(
                "{label}: a materialized root was supplied, but the registry has no \
                 memory-strategy registration to rebuild the contract through"
            ));
            continue;
        };
        let mut spec = surface.spec.clone();
        spec.weights = gen_core::WeightsSource::Dir(root.clone());
        match (registration.contract)(&spec) {
            Ok(contract) => {
                coverage.materialized_providers_checked += 1;
                if !contract.architecture_facts.has_declared_architecture_axis() {
                    errors.push(format!(
                        "{label}: rebuilt against the materialized snapshot root {}, the contract \
                         still declares no architecture axis; the provider derives none of the \
                         geometry its loader builds (activation_dtype_width does not count)",
                        root.display()
                    ));
                }
                for axis in contract.architecture_facts.zero_valued_axes() {
                    errors.push(format!(
                        "{label}: rebuilt against a materialized root, architecture_facts.{axis} \
                         is Some(0); a structurally absent axis is declared None, never zero"
                    ));
                }
            }
            Err(error) => errors.push(format!(
                "{label}: a materialized root was supplied, but the provider's own contract \
                 factory rejected {}: {error}",
                root.display()
            )),
        }
    }
    if errors.is_empty() {
        Ok(coverage)
    } else {
        Err(errors)
    }
}

/// Panic-on-failure entry point for [`check_memory_contract_surface_registry_facts`].
///
/// Returns what the walk covered so the caller can assert it was not vacuous.
pub fn memory_contract_surface_registry_facts_conformance(
    registry: &ProviderRegistry,
    materialized_root: Option<MaterializedRootLookup<'_>>,
) -> MemoryContractSurfaceFactsCoverage {
    match check_memory_contract_surface_registry_facts(registry, materialized_root) {
        Ok(coverage) => coverage,
        Err(errors) => panic!(
            "memory-contract surface registry facts conformance FAILED:\n- {}",
            errors.join("\n- ")
        ),
    }
}

fn check_memory_registration(
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

fn decode_probe_context(base: &MemoryRunContext, edge: u32, overlap: u32) -> MemoryRunContext {
    let mut context = base.clone();
    context.selection.strategy = MemoryStrategy::BoundedDecode;
    context.selection.parameters = MemoryStrategyParameters {
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        ..Default::default()
    };
    context
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        standard_memory_strategy_safety_check, MemoryBackendRealization,
        MemoryBehaviorBeginRequest, MemoryCacheState, MemoryCalibrationIdentity,
        MemoryDecodeRouteDomain, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
        MemoryNumericTier, MemoryOptimizationAuthority, MemoryParameterRanges,
        MemoryPidDecodeRoutes, MemoryRequestScope, MemoryRunOutcome, MemorySelection,
        MemoryStrategyCapability, WeightsSource,
    };

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
            optimization_authority: MemoryOptimizationAuthority::Calibrated,
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
                reference_count: 0,
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

    fn backend() -> MemoryBackendRealization {
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        }
    }

    /// Honest baseline for the facts check: a decomposable base total, three distinct component
    /// totals, a lifecycle-phase formula, and one declared architecture axis.
    fn facts_contract() -> MemoryProviderContract {
        let mut contract = MemoryProviderContract::compatibility_default("facts", backend());
        contract.asset_facts = gen_core::MemoryAssetFacts {
            base_bytes: 60,
            conditioning_bytes: 10,
            transformer_bytes: 20,
            decoder_bytes: 30,
            overlay_bytes: 0,
        };
        contract.formula = gen_core::MemoryFormulaKind::PhaseEnvelope {
            phases: vec![MemoryPhase::Denoise],
            variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
        };
        contract.architecture_facts = gen_core::MemoryArchitectureFacts {
            attention_heads: Some(30),
            head_dim: Some(128),
            transformer_blocks: Some(30),
            patch_size: Some(2),
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };
        contract
    }

    #[test]
    fn honest_contract_facts_conform() {
        let contract = facts_contract();
        check_memory_contract_facts(&contract).unwrap();
        assert_memory_contract_facts_conform(&contract);
    }

    #[test]
    fn base_bytes_that_is_not_its_own_decomposition_is_rejected() {
        let mut contract = facts_contract();
        // The classic dishonest shape: an overlay folded into the total but into no component.
        contract.asset_facts.base_bytes += 5;
        let errors = check_memory_contract_facts(&contract).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("must equal conditioning")),
            "{errors:?}"
        );
    }

    #[test]
    fn one_total_repeated_in_two_component_fields_is_rejected() {
        let mut contract = facts_contract();
        // Decoder bytes borrowed for the conditioning phase: the sum still adds up, the
        // decomposition is still a lie.
        contract.asset_facts.conditioning_bytes = 30;
        contract.asset_facts.base_bytes = 30 + 20 + 30;
        let errors = check_memory_contract_facts(&contract).unwrap_err();
        assert!(
            errors.iter().any(|error| error
                .contains("asset_facts.conditioning_bytes and asset_facts.decoder_bytes repeat")),
            "{errors:?}"
        );
    }

    #[test]
    fn a_zero_component_pair_is_not_treated_as_a_repeat() {
        let mut contract = facts_contract();
        contract.asset_facts.conditioning_bytes = 0;
        contract.asset_facts.decoder_bytes = 0;
        contract.asset_facts.base_bytes = contract.asset_facts.transformer_bytes;
        check_memory_contract_facts(&contract).unwrap();
    }

    #[test]
    fn all_absent_architecture_facts_on_a_phase_provider_are_rejected() {
        for formula in [
            gen_core::MemoryFormulaKind::PhaseEnvelope {
                phases: vec![MemoryPhase::Denoise],
                variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
            },
            gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
                phases: vec![MemoryPhase::Denoise],
                variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
                resident_components: Vec::new(),
            },
        ] {
            let mut contract = facts_contract();
            contract.formula = formula;
            contract.architecture_facts = gen_core::MemoryArchitectureFacts::default();
            let errors = check_memory_contract_facts(&contract).unwrap_err();
            assert!(
                errors.iter().any(|error| error
                    .contains("must declare at least one config-derived architecture fact")),
                "{errors:?}"
            );
        }
    }

    /// The E2 gate must not be satisfiable by a compile-time constant. `activation_dtype_width` is
    /// emitted from the provider's pinned dtype whether or not a single component `config.json` was
    /// found, so a contract carrying only that axis has read nothing and must still be rejected.
    #[test]
    fn activation_dtype_width_alone_does_not_satisfy_the_architecture_gate() {
        let mut contract = facts_contract();
        contract.formula = gen_core::MemoryFormulaKind::PhaseEnvelope {
            phases: vec![MemoryPhase::Denoise],
            variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
        };
        contract.architecture_facts = gen_core::MemoryArchitectureFacts {
            activation_dtype_width: Some(2),
            ..Default::default()
        };
        assert!(!contract.architecture_facts.is_empty());
        assert!(!contract.architecture_facts.has_declared_architecture_axis());
        let errors = check_memory_contract_facts(&contract).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("must declare at least one config-derived")),
            "{errors:?}"
        );
        // One genuinely config-derived axis alongside it is accepted.
        contract.architecture_facts.transformer_blocks = Some(30);
        check_memory_contract_facts(&contract).unwrap();
    }

    // ---- the backend-keyed weights-free architecture rule (SC-22661 reconciliation) ------------

    const CANDLE_BACKEND: MemoryBackendRealization = MemoryBackendRealization::CandleCuda {
        device_residency: true,
        host_backed_weights: false,
        host_to_device_block_materialization: true,
        block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
    };

    /// One config-derived axis set, the shape either backend publishes once geometry is known.
    fn declared_axes() -> gen_core::MemoryArchitectureFacts {
        gen_core::MemoryArchitectureFacts {
            transformer_blocks: Some(30),
            activation_dtype_width: Some(2),
            ..Default::default()
        }
    }

    /// `activation_dtype_width` only — the crate-wide constant that must satisfy neither arm.
    fn dtype_only_axes() -> gen_core::MemoryArchitectureFacts {
        gen_core::MemoryArchitectureFacts {
            activation_dtype_width: Some(2),
            ..Default::default()
        }
    }

    fn surface_contract(
        provider_id: &str,
        backend: MemoryBackendRealization,
        facts: gen_core::MemoryArchitectureFacts,
    ) -> gen_core::Result<MemoryProviderContract> {
        let mut contract = MemoryProviderContract::compatibility_default(provider_id, backend);
        contract.architecture_facts = facts;
        Ok(contract)
    }

    /// The honest Candle shape: axes appear only once the snapshot root exists on disk, exactly
    /// like `candle_gen::architecture_facts::snapshot_root` gates them.
    fn candle_honest(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let materialized = matches!(&spec.weights, WeightsSource::Dir(root) if root.is_dir());
        surface_contract(
            "candle_route",
            CANDLE_BACKEND,
            if materialized {
                declared_axes()
            } else {
                Default::default()
            },
        )
    }

    /// The Candle defect the weights-free arm exists to catch: geometry with nothing to read it
    /// from, so it came from the provider id.
    fn candle_fabricating(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let _ = spec;
        surface_contract("candle_route", CANDLE_BACKEND, declared_axes())
    }

    /// The Candle defect only the materialized arm can catch: `default()` unconditionally, which
    /// the weights-free walk cannot distinguish from honesty.
    fn candle_never_derives(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let _ = spec;
        surface_contract("candle_route", CANDLE_BACKEND, Default::default())
    }

    /// A Candle route that publishes only the crate-wide dtype constant on a materialized root has
    /// still derived no geometry.
    fn candle_dtype_only(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let _ = spec;
        surface_contract("candle_route", CANDLE_BACKEND, dtype_only_axes())
    }

    /// The honest MLX shape: preset constants exist before any snapshot does.
    fn mlx_preset(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let _ = spec;
        surface_contract("mlx_route", backend(), declared_axes())
    }

    /// The MLX defect: a weights-free surface withholding geometry the crate already holds.
    fn mlx_silent(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let _ = spec;
        surface_contract("mlx_route", backend(), Default::default())
    }

    /// ...and the constant that must not be mistaken for it.
    fn mlx_dtype_only(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        let _ = spec;
        surface_contract("mlx_route", backend(), dtype_only_axes())
    }

    fn registry_of(
        provider_id: &'static str,
        contract: fn(&LoadSpec) -> gen_core::Result<MemoryProviderContract>,
    ) -> ProviderRegistry {
        gen_core::ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(MemoryRegistration {
                provider_id,
                contract,
                safety_check: gen_core::default_registered_memory_strategy_safety_check,
            })
            .register_memory_contract_fixture(MemoryContractFixtureRegistration {
                provider_id,
                contract,
                surface_specs: gen_core::mlx_memory_contract_surface_specs,
            })
            .build()
            .expect("a single paired memory route builds")
    }

    /// AC (SC-22661 reconciliation, arm 1): a **Candle** weights-free surface must publish
    /// `MemoryArchitectureFacts::default()`. Its axes are gated on a materialized snapshot root the
    /// registry deliberately does not create, so an axis here was inferred from the provider id.
    #[test]
    fn a_candle_weights_free_surface_may_not_declare_an_architecture_axis() {
        let honest = registry_of("candle_route", candle_honest);
        let coverage = memory_contract_surface_registry_facts_conformance(&honest, None);
        assert!(
            coverage.surfaces_checked > 0,
            "the walk had nothing to check"
        );
        assert_eq!(coverage.materialized_providers_checked, 0);

        let errors = check_memory_contract_surface_registry_facts(
            &registry_of("candle_route", candle_fabricating),
            None,
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("Candle weights-free contract surface must publish")),
            "{errors:?}"
        );
        // The crate-wide dtype constant is not an axis, so publishing only it is still honest here.
        check_memory_contract_surface_registry_facts(
            &registry_of("candle_route", candle_dtype_only),
            None,
        )
        .unwrap();
    }

    /// AC (SC-22661 reconciliation, arm 2): an **MLX** weights-free surface must declare at least
    /// one axis. MLX geometry is mirrored from compile-time presets that exist before a snapshot
    /// does, so declaring nothing withholds facts the crate already holds.
    #[test]
    fn an_mlx_weights_free_surface_must_declare_an_architecture_axis() {
        memory_contract_surface_registry_facts_conformance(
            &registry_of("mlx_route", mlx_preset),
            None,
        );

        for (label, contract) in [
            ("nothing at all", mlx_silent as fn(&LoadSpec) -> _),
            ("only the dtype constant", mlx_dtype_only),
        ] {
            let errors = check_memory_contract_surface_registry_facts(
                &registry_of("mlx_route", contract),
                None,
            )
            .unwrap_err();
            assert!(
                errors.iter().any(|error| error
                    .contains("MLX weights-free contract surface must declare at least one")),
                "{label}: {errors:?}"
            );
        }
    }

    /// AC (SC-22661): the materialized arm is what makes the registry walk non-vacuous for E2. A
    /// provider that returns `MemoryArchitectureFacts::default()` unconditionally passes every
    /// weights-free assertion and fails only here.
    #[test]
    fn a_provider_that_never_derives_an_axis_fails_only_the_materialized_arm() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().to_path_buf();
        let lookup = |_: &str| Some(path.clone());

        // Honest: the same root yields geometry, and the walk records that it checked it.
        let coverage = memory_contract_surface_registry_facts_conformance(
            &registry_of("candle_route", candle_honest),
            Some(&lookup),
        );
        assert_eq!(coverage.materialized_providers_checked, 1);

        // The `::default()` mutation: invisible weights-free, rejected here.
        let never = registry_of("candle_route", candle_never_derives);
        check_memory_contract_surface_registry_facts(&never, None)
            .expect("the weights-free walk cannot see this defect");
        let errors =
            check_memory_contract_surface_registry_facts(&never, Some(&lookup)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("still declares no architecture axis")),
            "{errors:?}"
        );

        // A provider the lookup declines is checked on the weights-free arm only.
        let none = |_: &str| None;
        let coverage = memory_contract_surface_registry_facts_conformance(&never, Some(&none));
        assert_eq!(coverage.materialized_providers_checked, 0);
    }

    /// A component sum that overflows `u64` must be reported, not saturated into agreement with a
    /// `base_bytes` of `u64::MAX`.
    #[test]
    fn an_overflowing_base_component_sum_is_rejected_rather_than_saturated() {
        let mut contract = facts_contract();
        contract.asset_facts.conditioning_bytes = u64::MAX;
        contract.asset_facts.transformer_bytes = 1;
        contract.asset_facts.decoder_bytes = 2;
        contract.asset_facts.base_bytes = u64::MAX;
        let errors = check_memory_contract_facts(&contract).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error == "base component byte sum overflow"),
            "{errors:?}"
        );
    }

    /// The E1 half stands alone for the weights-free path, and the E2 gate is *not* applied there.
    #[test]
    fn the_asset_facts_entry_point_checks_e1_without_the_architecture_gate() {
        let mut contract = facts_contract();
        contract.architecture_facts = gen_core::MemoryArchitectureFacts::default();
        check_memory_contract_asset_facts(&contract).unwrap();
        assert_memory_contract_asset_facts_conform(&contract);
        // ...and it still rejects a dishonest decomposition.
        contract.asset_facts.base_bytes += 5;
        let errors = check_memory_contract_asset_facts(&contract).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("must equal conditioning")),
            "{errors:?}"
        );
    }

    #[test]
    #[should_panic(expected = "memory-contract asset-facts conformance FAILED for 'facts'")]
    fn the_panicking_asset_facts_entry_point_names_the_provider() {
        let mut contract = facts_contract();
        contract.asset_facts.base_bytes += 1;
        assert_memory_contract_asset_facts_conform(&contract);
    }

    #[test]
    fn a_provider_without_a_phase_formula_may_still_declare_no_architecture_facts() {
        let mut contract = facts_contract();
        contract.formula = gen_core::MemoryFormulaKind::AssetBytesPlusHeadroom;
        contract.architecture_facts = gen_core::MemoryArchitectureFacts::default();
        check_memory_contract_facts(&contract).unwrap();
    }

    #[test]
    fn a_zero_valued_architecture_axis_is_rejected() {
        let mut contract = facts_contract();
        contract.architecture_facts.vae_temporal_scale = Some(0);
        let errors = check_memory_contract_facts(&contract).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("architecture_facts.vae_temporal_scale is Some(0)")),
            "{errors:?}"
        );
    }

    #[test]
    #[should_panic(expected = "memory-contract facts conformance FAILED for 'facts'")]
    fn the_panicking_entry_point_names_the_provider() {
        let mut contract = facts_contract();
        contract.asset_facts.base_bytes += 1;
        assert_memory_contract_facts_conform(&contract);
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

    fn eager_only_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        if spec.load_shape != gen_core::LoadShape::EagerMaterialization {
            return Err(gen_core::Error::Unsupported("eager only".into()));
        }
        let mut contract = MemoryProviderContract::compatibility_default("eager-only", backend());
        contract.load_shape = spec.load_shape;
        Ok(contract)
    }

    fn always_reject_safety(
        _spec: &LoadSpec,
        _contract: &MemoryProviderContract,
        _context: &MemoryRunContext,
    ) -> MemorySafetyDecision {
        MemorySafetyDecision::Reject {
            reason: "no optimized strategies".into(),
        }
    }

    #[test]
    fn mixed_catalog_conformance_uses_each_providers_valid_load_shape() {
        let registration = MemoryRegistration {
            provider_id: "eager-only",
            contract: eager_only_contract,
            safety_check: always_reject_safety,
        };
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        let mut errors = Vec::new();

        check_memory_registration(&registration, None, None, None, &spec, &mut errors);

        assert_eq!(errors, Vec::<String>::new());
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
            None,
            None,
            Some(&PID_BEHAVIOR),
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            &mut errors,
        );
        assert_eq!(errors, Vec::<String>::new());
    }

    #[test]
    fn conformance_uses_the_explicit_weights_free_factory() {
        fn production_requires_assets(
            _spec: &LoadSpec,
        ) -> gen_core::Result<MemoryProviderContract> {
            Err(gen_core::Error::Msg(
                "production contract touched required assets".to_owned(),
            ))
        }

        let registration = MemoryRegistration {
            provider_id: "pid-provider",
            contract: production_requires_assets,
            safety_check: route_aware_safety,
        };
        let contract_fixture = MemoryContractFixtureRegistration {
            provider_id: "pid-provider",
            contract: pid_contract,
            surface_specs: gen_core::mlx_memory_contract_surface_specs,
        };
        let mut errors = Vec::new();
        check_memory_registration(
            &registration,
            Some(&contract_fixture),
            None,
            Some(&PID_BEHAVIOR),
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            &mut errors,
        );
        assert_eq!(errors, Vec::<String>::new());
    }

    #[test]
    fn conformance_uses_the_explicit_resident_only_witness_factory() {
        fn production_requires_assets(
            _spec: &LoadSpec,
        ) -> gen_core::Result<MemoryProviderContract> {
            Err(gen_core::Error::Msg(
                "production contract touched required assets".to_owned(),
            ))
        }

        let registration = MemoryRegistration {
            provider_id: "eager-only",
            contract: production_requires_assets,
            safety_check: always_reject_safety,
        };
        let witness = ResidentOnlyMemoryContractRegistration {
            provider_id: "eager-only",
            contract: eager_only_contract,
            surface_specs: gen_core::mlx_memory_contract_surface_specs,
        };
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        let mut errors = Vec::new();
        check_memory_registration(
            &registration,
            None,
            Some(&witness),
            None,
            &spec,
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
            None,
            None,
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
            None,
            None,
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
