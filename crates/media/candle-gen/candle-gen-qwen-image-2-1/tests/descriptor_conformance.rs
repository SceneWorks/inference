//! Weights-free, default-run descriptor-level gen-core conformance: the registration this provider
//! exports satisfies the descriptor/capability invariants checkable without loading weights.
//! Behavioural conformance (progress/cancel/seed) runs on the committed miniature snapshot in
//! `generator_contract.rs`.

#[test]
fn registered_descriptors_conform() {
    let registry =
        candle_gen_qwen_image_2_1::provider_registry().expect("provider registry should build");
    assert!(registry.generators().len() > 0);
    let errs = registry.descriptor_conformance_errors();
    assert!(
        errs.is_empty(),
        "descriptor conformance FAILED ({} violations):\n  - {}",
        errs.len(),
        errs.join("\n  - ")
    );
}
