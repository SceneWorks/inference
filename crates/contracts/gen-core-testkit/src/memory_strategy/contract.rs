//! Per-contract memory-strategy and memory-contract fact checks.

use super::*;

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
