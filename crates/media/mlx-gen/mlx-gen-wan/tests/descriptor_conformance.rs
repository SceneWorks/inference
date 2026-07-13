//! Weights-free, default-run descriptor-level gen-core conformance (sc-9098, F-009): every
//! registration this provider explicitly exports (including any reused sibling providers) satisfies
//! the descriptor/capability invariants checkable without loading weights — id/family/backend
//! shape, coherent size/count bounds, duplicate-free curated names and conditioning kinds,
//! modality-consistent conditioning, and per-kind registry id uniqueness. Behavioral conformance
//! (progress/cancel/seed) stays weights-gated in the crate's `#[ignore]`d suites.

#[test]
fn registered_descriptors_conform() {
    let registry = mlx_gen_wan::provider_registry().expect("provider registry should build");
    assert!(
        registry.generators().len() > 0,
        "provider registry must contain a generator"
    );
    let errs = registry.descriptor_conformance_errors();
    assert!(
        errs.is_empty(),
        "descriptor conformance FAILED ({} violations):\n  - {}",
        errs.len(),
        errs.join("\n  - ")
    );
}
