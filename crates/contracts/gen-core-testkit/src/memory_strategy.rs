//! Weights-free conformance checks for the shared memory-strategy provider contract.

use gen_core::{
    MemoryCleanupSemantics, MemoryProviderContract, MemoryStrategy, MemoryStrategySupport,
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

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::MemoryBackendRealization;

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
}
