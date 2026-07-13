//! Explicit provider-catalog coverage for the three Qwen-Image variants. No weights required.

#[test]
fn qwen_image_variants_are_exported_by_provider_catalog() {
    for id in ["qwen_image", "qwen_image_control", "qwen_image_edit"] {
        let reg = mlx_gen_qwen_image::provider_registry()
            .unwrap()
            .generators()
            .copied()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("provider catalog should export {id}"));
        assert_eq!((reg.descriptor)().family, "qwen-image");
    }
}
