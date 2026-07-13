//! Verifies the explicit Z-Image provider catalog: every engine id is visible, descriptors retain
//! their capability surface, and resolve-by-id dispatch reaches the provider's loader.

use mlx_gen::{LoadSpec, WeightsSource};

#[test]
fn z_image_turbo_resolves_through_provider_catalog() {
    let reg = mlx_gen_z_image::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "z_image_turbo")
        .expect("provider catalog should export z_image_turbo");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "z_image_turbo");
    assert_eq!(d.family, "z-image");

    // `provider_registry().load(id, …)` routes to *this* provider's loader: a bogus spec surfaces the
    // provider's own snapshot-layout error, not the registry's "no generator registered".
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load("z_image_turbo", &spec)
        .err()
        .expect("a single-file spec is rejected by the loader")
        .to_string();
    assert!(
        err.contains("snapshot directory"),
        "expected the z-image loader's error, got: {err}"
    );
}

#[test]
fn z_image_turbo_visible_in_registry_iteration() {
    assert!(mlx_gen_z_image::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .any(|r| (r.descriptor)().id == "z_image_turbo"));
}

#[test]
fn base_z_image_resolves_through_provider_catalog() {
    // sc-8320: the base (non-Turbo) model has its own id alongside Turbo, without an id clash.
    let reg = mlx_gen_z_image::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "z_image")
        .expect("provider catalog should export z_image");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "z_image");
    assert_eq!(d.family, "z-image");
    // The base is the full-CFG variant (Turbo is guidance-distilled).
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_negative_prompt);

    // `provider_registry().load("z_image", …)` routes to the base loader (its own snapshot-layout error).
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load("z_image", &spec)
        .err()
        .expect("a single-file spec is rejected by the loader")
        .to_string();
    assert!(
        err.contains("snapshot directory"),
        "expected the base z-image loader's error, got: {err}"
    );
}

#[test]
fn base_turbo_and_control_all_coexist() {
    // The three z-image engine ids are distinct and all visible in registry iteration — no id
    // collision when the base (sc-8320) was added to the crate that already hosts turbo + control.
    let ids: Vec<&str> = mlx_gen_z_image::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .map(|r| (r.descriptor)().id)
        .filter(|id| id.starts_with("z_image"))
        .collect();
    for want in [
        "z_image",
        "z_image_turbo",
        "z_image_turbo_control",
        "z_image_control",
    ] {
        assert!(ids.contains(&want), "missing {want} in {ids:?}");
    }
}

#[test]
fn unknown_id_still_errors() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    assert!(mlx_gen_z_image::provider_registry()
        .unwrap()
        .load("not_a_model", &spec)
        .is_err());
}
