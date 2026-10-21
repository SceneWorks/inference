//! Public weights-free catalog checks.
#[test]
fn registers_both_native_routes() {
    let r = candle_gen_moss_image::register_providers(
        candle_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    assert!(r.descriptor_conformance_errors().is_empty());
    let ids: Vec<_> = r.generators().map(|r| (r.descriptor)().id).collect();
    assert_eq!(ids, vec!["moss_image_base", "moss_image_sft"]);
}
