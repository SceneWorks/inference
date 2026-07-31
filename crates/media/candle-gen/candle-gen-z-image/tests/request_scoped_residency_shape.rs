//! SC-15806 cross-backend conformance: the shared request shape is sufficient for Candle's future
//! request-scoped residency refactor (SC-15811) without changing `Generator::generate(&self, ...)`.
//!
//! This deliberately does not alter Candle's current residency implementation. It pins the mapping
//! that SC-15811 can adopt while retaining its existing load-policy/env override as the fallback.

use candle_gen::gen_core::{GenerationMemory, GenerationRequest, OffloadPolicy};

fn effective_policy(load_policy: OffloadPolicy, request: &GenerationRequest) -> OffloadPolicy {
    if request.memory.is_some_and(|memory| memory.stage_residency) {
        OffloadPolicy::Sequential
    } else {
        load_policy
    }
}

fn request(stage_residency: bool) -> GenerationRequest {
    GenerationRequest {
        memory: Some(GenerationMemory {
            stage_residency,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn one_candle_load_can_represent_warm_staged_warm_requests() {
    let load_policy = OffloadPolicy::Resident;
    let policies = [false, true, false].map(|stage| effective_policy(load_policy, &request(stage)));
    assert_eq!(
        policies,
        [
            OffloadPolicy::Resident,
            OffloadPolicy::Sequential,
            OffloadPolicy::Resident,
        ]
    );
}

#[test]
fn an_existing_sequential_fallback_remains_sequential() {
    assert_eq!(
        effective_policy(OffloadPolicy::Sequential, &request(false)),
        OffloadPolicy::Sequential
    );
}
