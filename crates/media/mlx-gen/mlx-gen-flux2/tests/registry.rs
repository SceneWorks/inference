//! sc-2346 S0: the explicit FLUX.2 catalog exposes every variant for weights-free introspection.

use mlx_gen::{ConditioningKind, LoadSpec, WeightsSource};

#[test]
fn flux2_variants_resolve_through_provider_catalog() {
    for id in [
        "flux2_klein_9b",
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
    ] {
        let reg = mlx_gen_flux2::provider_registry()
            .unwrap()
            .generators()
            .copied()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("provider catalog should export {id}"));
        let d = (reg.descriptor)();
        assert_eq!(d.family, "flux2");
        assert!(d.capabilities.requires_sigma_shift);
        assert!(d.capabilities.schedulers.contains(&"flow_match_euler"));
    }
}

#[test]
fn only_kv_variant_advertises_kv_cache() {
    let kv = mlx_gen_flux2::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b_kv_edit")
        .map(|r| (r.descriptor)())
        .expect("provider catalog should export the 9b-kv edit variant (sc-2347)");
    // The KV-cache edit variant accepts the same reference conditioning as the plain edit.
    assert!(kv.capabilities.supports_kv_cache);
    assert!(kv.capabilities.accepts(ConditioningKind::Reference));
    assert!(kv.capabilities.accepts(ConditioningKind::MultiReference));
    // The base txt2img + plain edit variants do NOT advertise the cache.
    for id in ["flux2_klein_9b", "flux2_klein_9b_edit"] {
        let d = mlx_gen_flux2::provider_registry()
            .unwrap()
            .generators()
            .copied()
            .find(|r| (r.descriptor)().id == id)
            .map(|r| (r.descriptor)())
            .unwrap();
        assert!(!d.capabilities.supports_kv_cache, "{id} must not cache");
    }
}

#[test]
fn variants_advertise_expected_conditioning() {
    let edit = mlx_gen_flux2::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b_edit")
        .map(|r| (r.descriptor)())
        .unwrap();
    // Edit accepts a single `Reference` (token concat) and N-image `MultiReference` (sc-2645).
    assert!(edit.capabilities.accepts(ConditioningKind::Reference));
    assert!(edit.capabilities.accepts(ConditioningKind::MultiReference));

    let t2i = mlx_gen_flux2::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b")
        .map(|r| (r.descriptor)())
        .unwrap();
    // txt2img consumes a `Reference` as an img2img init image (sc-2644); multi-image editing is
    // the edit variant only.
    assert!(t2i.capabilities.accepts(ConditioningKind::Reference));
    assert!(!t2i.capabilities.accepts(ConditioningKind::MultiReference));
}

#[test]
fn load_resolves_then_fails_on_missing_snapshot() {
    for id in [
        "flux2_klein_9b",
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
    ] {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = mlx_gen_flux2::provider_registry()
            .unwrap()
            .load(id, &spec)
            .err()
            .expect("a missing snapshot dir must error")
            .to_string();
        // The id resolves through the registry and reaches the loader (which then fails to read the
        // snapshot) — i.e. NOT a "no generator registered" miss.
        assert!(
            !err.contains("no generator registered"),
            "id should resolve through the registry, got: {err}"
        );
    }
}

#[test]
fn dev_control_is_exported_and_requires_control_weights() {
    // sc-2292: the strict-pose control variant resolves through the provider catalog.
    let d = mlx_gen_flux2::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == "flux2_dev_control")
        .map(|r| (r.descriptor)())
        .expect("provider catalog should export flux2_dev_control (sc-2292)");
    assert_eq!(d.family, "flux2");
    assert!(d.capabilities.accepts(ConditioningKind::Control));
    assert!(d.capabilities.accepts(ConditioningKind::Reference));
    assert!(d.capabilities.mac_only && !d.capabilities.supports_kv_cache);

    // …and a load through the registry reaches the loader, which requires the control checkpoint
    // (proving the overlay is a hard requirement, not a "no generator registered" miss).
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let err = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load("flux2_dev_control", &spec)
        .err()
        .expect("missing control weights must error")
        .to_string();
    assert!(!err.contains("no generator registered"), "got: {err}");
    assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
}

#[test]
fn single_file_spec_is_rejected() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load("flux2_klein_9b", &spec)
        .err()
        .expect("a single-file spec is rejected")
        .to_string();
    assert!(err.contains("snapshot directory"), "got: {err}");
}
