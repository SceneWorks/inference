use super::*;
use candle_gen::gen_core::{GenerationRequest, Image};
use serde_json::Value;
fn array(v: &Value) -> Tensor {
    let shape: Vec<usize> = serde_json::from_value(v["shape"].clone()).unwrap();
    let values: Vec<f32> = serde_json::from_value(v["values"].clone()).unwrap();
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}
#[test]
fn frozen_cfg_euler_is_fp32_and_rejects_narrowed_accumulator() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let sampling = &case["sampling"];
        let steps = sampling["steps"].as_array().unwrap();
        let sigmas: Vec<f32> =
            serde_json::from_value(sampling["sigmas"]["values"].clone()).unwrap();
        let mut current = array(&steps[0]["input"]);
        for (i, step) in steps.iter().enumerate() {
            current = cfg_euler_step(
                &current,
                &array(&step["conditional"]),
                Some(&array(&step["unconditional"])),
                sampling["cfg"].as_f64().unwrap() as f32,
                sigmas[i + 1] - sigmas[i],
            )
            .unwrap();
            assert_eq!(current.dtype(), DType::F32);
            let expected: Vec<f32> =
                serde_json::from_value(step["output"]["values"].clone()).unwrap();
            let shape: Vec<usize> =
                serde_json::from_value(step["output"]["shape"].clone()).unwrap();
            assert_eq!(current.dims(), shape);
            let error = |actual: &Tensor| {
                let values = tensor_f32(actual).unwrap();
                assert_eq!(values.len(), expected.len());
                assert!(values.iter().chain(&expected).all(|x| x.is_finite()));
                values
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max)
            };
            assert!(error(&current) <= 1e-6, "source CFG/Euler precision");
            assert!(
                error(
                    &current
                        .to_dtype(DType::BF16)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                ) > 1e-6,
                "narrowed accumulator must fail the same bound"
            );
        }
    }
    let current = Tensor::new(&[0.1234567f32], &Device::Cpu).unwrap();
    let conditional = Tensor::new(&[0.6789123f32], &Device::Cpu).unwrap();
    let negative = Tensor::new(&[-0.2345678f32], &Device::Cpu).unwrap();
    for cfg in [0f32, 1., 4.] {
        let output = cfg_euler_step(
            &current,
            &conditional,
            if cfg == 1. { None } else { Some(&negative) },
            cfg,
            -0.137,
        )
        .unwrap();
        let expected = 0.1234567f32
            - 0.137
                * if cfg == 1. {
                    0.6789123
                } else {
                    -0.2345678 + cfg * (0.6789123 + 0.2345678)
                };
        assert_eq!(output.dtype(), DType::F32);
        assert!((tensor_f32(&output).unwrap()[0] - expected).abs() <= 1e-7);
    }
}
#[test]
fn edit_peak_prices_reference_context_and_refuses_before_assets() {
    let references = [Image {
        width: 512,
        height: 512,
        pixels: vec![0; 512 * 512 * 3],
    }];
    let inputs = MossImageEditInputs {
        references: &references,
        ..Default::default()
    };
    let mut request = GenerationRequest {
        width: 512,
        height: 512,
        ..Default::default()
    };
    let full = extra_peak(inputs, &request).unwrap();
    assert!(
        full > extra_peak(
            MossImageEditInputs {
                reference_max_edge: 256,
                ..inputs
            },
            &request
        )
        .unwrap()
            + 100_000_000
    );
    request.moss_image = Some(Default::default());
    request.moss_image.as_mut().unwrap().max_sequence_length = 256;
    assert!(extra_peak(inputs, &request).unwrap() < full);
    let spec = LoadSpec::new(WeightsSource::Dir("unused-cold-snapshot".into()));
    request.moss_image.as_mut().unwrap().uncond_skip_layers = Some(vec![22]);
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        let error = crate::memory_strategy::edit_request_peak(variant, &spec, &request, inputs)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("full-layer CFG"),
            "control refusal must precede asset access: {error}"
        );
    }
}
#[test]
fn edit_package_missing_corrupt_and_prepared_identity_refuse_without_device() {
    let temp = tempfile::tempdir().unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(temp.path().join("unused-base")));
    assert!(EditAssets::prepare(&spec).is_err());
    spec.components.insert(
        EDIT_COMPONENT.into(),
        WeightsSource::Dir(temp.path().to_path_buf()),
    );
    std::fs::write(temp.path().join("adapter_config.json"), b"{}").unwrap();
    std::fs::write(
        temp.path().join("adapter_model.safetensors"),
        b"not-safetensors",
    )
    .unwrap();
    assert!(EditAssets::prepare(&spec).is_err());
    spec.prepare_with_file_pins([]).unwrap();
    assert!(EditAssets::prepare(&spec).is_err());
}
