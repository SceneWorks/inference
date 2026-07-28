//! Weights-free conformance checks for the shared image-memory provider contract.

use gen_core::{
    ImageMemoryCleanupSemantics, ImageMemoryProviderContract, ImageMemoryStrategy,
    ImageMemoryStrategySupport,
};

/// Check the static declaration and the safety-critical runtime semantics every provider must share.
pub fn check_image_memory_contract(
    contract: &ImageMemoryProviderContract,
) -> Result<(), Vec<String>> {
    let mut errors = contract.conformance_errors();

    if !matches!(
        contract
            .capability(ImageMemoryStrategy::Resident)
            .map(|capability| &capability.support),
        Some(ImageMemoryStrategySupport::Implemented)
    ) {
        errors.push("Resident baseline must be implemented".to_owned());
    }
    if contract.runtime.cancellation
        != ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
    {
        errors.push("cancellation must synchronize and release active state".to_owned());
    }
    if contract.runtime.error
        != ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
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
pub fn image_memory_conformance(contract: &ImageMemoryProviderContract) {
    if let Err(errors) = check_image_memory_contract(contract) {
        panic!(
            "image-memory conformance FAILED for '{}':\n- {}",
            contract.provider_id,
            errors.join("\n- ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::ImageMemoryBackendRealization;

    fn backend() -> ImageMemoryBackendRealization {
        ImageMemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        }
    }

    #[test]
    fn resident_only_compatibility_contract_conforms_without_claiming_optimization() {
        let contract = ImageMemoryProviderContract::compatibility_default("legacy", backend());
        check_image_memory_contract(&contract).unwrap();
        assert!(contract.calibration.is_none());
    }

    #[test]
    fn malformed_strategy_table_is_reported() {
        let mut contract = ImageMemoryProviderContract::compatibility_default("bad", backend());
        contract.strategies.pop();
        let errors = check_image_memory_contract(&contract).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("exactly once")));
    }
}
