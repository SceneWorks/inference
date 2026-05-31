//! Proves the architecture's central claim (docs/MODEL_ARCHITECTURE.md §4): linking the
//! provider crate self-registers Z-Image into `mlx-gen`'s link-time `inventory` registry — the
//! core has no central match to edit — so `mlx_gen::load("z_image_turbo", …)` resolves across
//! the crate boundary. This is the Rust stand-in for a DI container's resolve-by-id.
//!
//! NOTE: a provider must actually be *linked* into the consumer for its `inventory::submit!` to
//! take effect — a dependency that is declared but never referenced can have its link-section
//! statics dropped by the linker. The `use … as _` below forces the link (the SceneWorks worker
//! references every provider it serves, so this is automatic there). This is the "DI container
//! must know about the assembly" detail.

use mlx_gen::{LoadSpec, WeightsSource};
use mlx_gen_z_image as _;

#[test]
fn z_image_turbo_resolves_through_core_registry() {
    // The descriptor resolves across the crate boundary without loading weights — proof the
    // provider's `inventory::submit!` fired and the core can find it by id.
    let reg = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "z_image_turbo")
        .expect("provider self-registered via inventory");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "z_image_turbo");
    assert_eq!(d.family, "z-image");

    // `mlx_gen::load(id, …)` routes to *this* provider's loader: a bogus spec surfaces the
    // provider's own snapshot-layout error, not the registry's "no generator registered".
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen::load("z_image_turbo", &spec)
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
    assert!(mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "z_image_turbo"));
}

#[test]
fn unknown_id_still_errors() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    assert!(mlx_gen::load("not_a_model", &spec).is_err());
}
