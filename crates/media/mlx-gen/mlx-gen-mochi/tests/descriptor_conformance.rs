//! Weights-free, default-run descriptor-level gen-core conformance (mirrors LTX): the Mochi
//! registration satisfies the descriptor/capability invariants checkable without loading weights —
//! id/family/backend shape, coherent size/count bounds, duplicate-free curated names + conditioning
//! kinds, modality-consistent conditioning. Behavioral conformance (progress/cancel/seed) stays
//! weights-gated in the crate's `#[ignore]`d `e2e_parity` suite.

#[test]
fn registered_descriptors_conform() {
    let registry = mlx_gen_mochi::provider_registry().expect("provider registry should build");
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
