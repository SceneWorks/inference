//! Weights-free conformance checks for the shared memory-strategy provider contract.

use gen_core::{
    LoadSpec, MemoryBehaviorRegistration, MemoryBudget, MemoryCleanupSemantics,
    MemoryContractFixtureRegistration, MemoryPhase, MemoryProviderContract, MemoryRegistration,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyParameters, MemoryStrategySupport, Precision, ProviderRegistry,
    ResidentOnlyMemoryContractRegistration,
};

mod contract;
pub use contract::*;

mod registry;
pub use registry::*;

mod registration;
use registration::*;

#[cfg(test)]
mod tests;
