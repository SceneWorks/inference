//! Registry round-trip and conformance checks, and the shared id-lookup / failure-panic helpers every kind's harness uses.

use super::*;

/// The registry round-trip body every provider kind shares (sc-27315): `id` must be one of the
/// descriptor ids the caller's registry publishes for that kind.
pub(crate) fn check_registry_ids(
    ids: impl IntoIterator<Item = &'static str>,
    id: &str,
) -> Result<(), String> {
    if ids.into_iter().any(|candidate| candidate == id) {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found in the explicit provider registry (gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// The aggregate-and-panic tail every `*_conformance` suite shares (sc-27315): no-op when
/// `failures` is empty, otherwise one panic naming the suite, provider (and backend) and every
/// failure.
pub(crate) fn panic_on_failures(suite: &str, id: &str, backend: Option<&str>, failures: &[String]) {
    if failures.is_empty() {
        return;
    }
    let backend = backend
        .map(|b| format!("{b} backend, "))
        .unwrap_or_default();
    panic!(
        "{suite} FAILED for `{id}` ({backend}gen-core {}):\n  - {}",
        gen_core::VERSION,
        failures.join("\n  - ")
    );
}

/// **Registry round-trip.** The provider's descriptor `id` is present in the explicit registry
/// supplied by the caller (a missing catalog entry is the runtime "engine not found" trap).
pub fn check_registry_roundtrip(
    registry: &gen_core::ProviderRegistry,
    g: &dyn Generator,
) -> Result<(), String> {
    crate::check_registry_ids(
        registry
            .generators()
            .map(|registration| (registration.descriptor)().id),
        g.descriptor().id,
    )
}

/// **Descriptor sweep (weights-free).** Run the registry-wide descriptor-level conformance sweep
/// over the explicit catalog supplied by the caller and panic with the aggregated violations — the
/// test-helper idiom of [`conformance`], minus any model load. Because no weights are touched,
/// providers wire this as a default (non-`#[ignore]`d) test; behavioral checks stay weights-gated.
pub fn registry_conformance(registry: &gen_core::ProviderRegistry) {
    let errs = registry.descriptor_conformance_errors();
    if !errs.is_empty() {
        panic!(
            "gen-core descriptor conformance FAILED ({} violations, gen-core {}):\n  - {}",
            errs.len(),
            gen_core::VERSION,
            errs.join("\n  - ")
        );
    }
}
