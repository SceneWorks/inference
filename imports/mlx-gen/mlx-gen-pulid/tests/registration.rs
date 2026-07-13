//! sc-7780 dead-strip gate: the PuLID-FLUX provider self-registers through the core registry once
//! the crate is linked (`use mlx_gen_pulid as _;`). This proves the macro-emitted
//! `inventory::submit!` (register_generators!) still fires; it needs no weights.

use mlx_gen_pulid as _;

#[test]
fn pulid_flux_resolves_through_core_registry() {
    let reg = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "pulid_flux")
        .expect("pulid_flux provider should self-register");
    assert_eq!((reg.descriptor)().family, "pulid");
}
