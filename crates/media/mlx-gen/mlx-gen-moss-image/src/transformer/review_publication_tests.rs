use super::author_fixture_tests::*;
use super::*;
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn final_optimizer_values_are_validated_before_adapter_file_is_written() {
    let _device = crate::cpu_test_device();
    let f = fixture("ordinary");
    let mut model =
        Transformer::load(&weights(&f["weights"], Dtype::Float32), geometry(&f)).unwrap();
    let (targets, mut params) =
        mlx_gen::train::lora::build_lora_targets(&mut model, &["layers.0.proj.q".into()], 2, 13)
            .unwrap();
    let key = params
        .keys()
        .find(|key| key.ends_with(".lora_b"))
        .unwrap()
        .clone();
    params.insert(key, Array::from_slice(&[1e30f32; 16 * 2], &[16, 2]));
    let adapter = mlx_gen::train::lora::TrainAdapter::Lora { targets };
    let config = mlx_gen::TrainingConfig {
        rank: 2,
        alpha: 1e30,
        ..Default::default()
    };
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("must-not-publish.safetensors");
    assert!(
        crate::adapters::save_validated(&mut model, &adapter, &params, &config, &[], &path)
            .is_err()
    );
    assert!(
        !path.exists(),
        "invalid effective adapter was published as a trained artifact"
    );
}
