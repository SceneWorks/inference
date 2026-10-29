//! Registry-wide memory-strategy and memory-contract-surface checks.

use super::*;

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
