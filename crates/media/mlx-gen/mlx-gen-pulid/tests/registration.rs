//! Explicit provider-catalog coverage for the PuLID-FLUX variant. No weights required.

#[test]
fn pulid_flux_is_exported_by_provider_catalog() {
    let reg = mlx_gen_pulid::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "pulid_flux")
        .expect("provider catalog should export pulid_flux");
    assert_eq!((reg.descriptor)().family, "pulid");
}
