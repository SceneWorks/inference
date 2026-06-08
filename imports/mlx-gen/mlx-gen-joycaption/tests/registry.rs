use mlx_gen::{LoadSpec, Precision, Quant, WeightsSource};
use mlx_gen_joycaption as _;

#[test]
fn joycaption_resolves_through_core_caption_registry() {
    let id = mlx_gen::caption::joycaption::JOY_CAPTION_MODEL_ID;
    let reg = mlx_gen::registry::captioners()
        .find(|r| (r.descriptor)().id == id)
        .expect("provider self-registered via inventory");
    let d = (reg.descriptor)();
    assert_eq!(d.id, id);
    assert_eq!(d.family, mlx_gen::caption::joycaption::JOY_CAPTION_FAMILY);
    assert!(d.capabilities.mac_only);

    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen::load_captioner(id, &spec)
        .err()
        .expect("a single-file spec is rejected by the loader")
        .to_string();
    assert!(
        err.contains("snapshot directory"),
        "expected the joycaption loader's error, got: {err}"
    );
}

#[test]
fn joycaption_rejects_unvalidated_load_features() {
    let id = mlx_gen::caption::joycaption::JOY_CAPTION_MODEL_ID;
    let root = WeightsSource::Dir("/nonexistent/joycaption".into());

    let q8 = LoadSpec::new(root.clone()).with_quant(Quant::Q8);
    let err = mlx_gen::load_captioner(id, &q8)
        .err()
        .expect("quantized specs are rejected before disk access")
        .to_string();
    assert!(err.contains("quantized"), "{err}");

    let mut fp32 = LoadSpec::new(root);
    fp32.precision = Precision::Fp32;
    let err = mlx_gen::load_captioner(id, &fp32)
        .err()
        .expect("fp32 specs are rejected before disk access")
        .to_string();
    assert!(err.contains("dense bf16"), "{err}");
}

#[test]
fn unknown_captioner_id_still_errors() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    assert!(mlx_gen::load_captioner("not_a_captioner", &spec).is_err());
}
