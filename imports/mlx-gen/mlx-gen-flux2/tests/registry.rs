//! sc-2346 S0: the FLUX.2-klein variants self-register and are introspectable through the core
//! registry without loading weights; loading is guarded until the model modules land (S1–S3).

use mlx_gen::{ConditioningKind, LoadSpec, WeightsSource};
use mlx_gen_flux2 as _;

#[test]
fn flux2_variants_resolve_through_core_registry() {
    for id in ["flux2_klein_9b", "flux2_klein_9b_edit"] {
        let reg = mlx_gen::registry::generators()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("{id} provider should self-register"));
        let d = (reg.descriptor)();
        assert_eq!(d.family, "flux2");
        assert!(d.capabilities.requires_sigma_shift);
        assert!(d.capabilities.schedulers.contains(&"flow_match_euler"));
    }
}

#[test]
fn edit_advertises_single_reference_txt2img_does_not() {
    let edit = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b_edit")
        .map(|r| (r.descriptor)())
        .unwrap();
    assert!(edit.capabilities.accepts(ConditioningKind::Reference));
    // Multi-reference edit is sc-2645, not this story.
    assert!(!edit.capabilities.accepts(ConditioningKind::MultiReference));

    let t2i = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b")
        .map(|r| (r.descriptor)())
        .unwrap();
    // img2img (Reference) is sc-2644, not this story's txt2img variant.
    assert!(!t2i.capabilities.accepts(ConditioningKind::Reference));
}

#[test]
fn loading_is_guarded_until_modules_land() {
    for id in ["flux2_klein_9b", "flux2_klein_9b_edit"] {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = mlx_gen::load(id, &spec)
            .err()
            .expect("S0 load is guarded")
            .to_string();
        // The guard names the slices, not a generic "no generator registered" miss.
        assert!(
            err.contains("S1") && err.contains("S3"),
            "expected the S0 load guard, got: {err}"
        );
    }
}
