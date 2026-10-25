//! Unchanged external Diffusers encoder oracle shared with MOSS edit. No real weights.
use super::*;
use candle_gen::candle_core::Device;
use serde_json::Value;
use std::collections::HashMap;
fn tensor(v: &Value, device: &Device) -> Tensor {
    let shape: Vec<usize> = serde_json::from_value(v["shape"].clone()).unwrap();
    let values: Vec<f32> = serde_json::from_value(v["values"].clone()).unwrap();
    Tensor::from_vec(values, shape, device).unwrap()
}
fn values(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}
fn delta(actual: &Tensor, expected: &Tensor) -> f32 {
    assert_eq!(actual.dims(), expected.dims());
    let (a, b) = (values(actual), values(expected));
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(&b).all(|x| x.is_finite()));
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
fn run(device: &Device) {
    let f: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_encode/fixture.json"
    ))
    .unwrap();
    let mut weights = HashMap::new();
    for recipe in f["weights"].as_array().unwrap() {
        let shape: Vec<usize> = serde_json::from_value(recipe["shape"].clone()).unwrap();
        let seed = recipe["seed"].as_u64().unwrap() as u32;
        let data: Vec<f32> = (0..shape.iter().product::<usize>() as u32)
            .map(|i| {
                let mut bits = (i + 1) ^ seed;
                bits ^= bits << 13;
                bits ^= bits >> 17;
                bits ^= bits << 5;
                let signed = (bits & 65535) as i32 - 32768;
                match recipe["kind"].as_str().unwrap() {
                    "bn_mean" => ((i % 17) as i32 - 8) as f32 / 64.,
                    "bn_var" => 1. + ((i * 7) % 31) as f32 / 32.,
                    "norm" => 1. + signed as f32 / 262144.,
                    "bias" => signed as f32 / 1048576.,
                    "weight" => signed as f32 / 8388608.,
                    "pointwise_identity" => {
                        signed as f32 / 8388608.
                            + if i / shape[1] as u32 == i % shape[1] as u32 {
                                1.
                            } else {
                                0.
                            }
                    }
                    kind => panic!("unknown source recipe {kind}"),
                }
            })
            .collect();
        let name = recipe["name"].as_str().unwrap();
        let probe: Vec<f32> =
            serde_json::from_value(f["weight_prefix_probes"][name].clone()).unwrap();
        assert_eq!(&data[..data.len().min(4)], probe.as_slice());
        weights.insert(
            name.to_string(),
            Tensor::from_vec(data, shape, device).unwrap(),
        );
    }
    let vae = Flux2Vae::build_with_encoder_channels(
        VarBuilder::from_tensors(weights, DType::F32, device),
        true,
        &Flux2VaeDecoderConfig {
            block_out_channels: [32; 4],
        },
        &[32; 4],
    )
    .unwrap();
    let pixels = tensor(&f["pixels_nchw"], device);
    let tolerance = f["absolute_tolerance"].as_f64().unwrap() as f32;
    let mean = vae.encode(&pixels).unwrap();
    let packed = vae.encode_packed(&pixels).unwrap();
    for (actual, key) in [(&mean, "mean_nchw"), (&packed, "normalized_packed_nchw")] {
        assert_eq!(actual.dtype(), DType::F32);
        let error = delta(actual, &tensor(&f[key], device));
        assert!(error <= tolerance, "{key}: error {error} > {tolerance}");
    }
    for (name, mutant) in f["mutants"].as_object().unwrap() {
        let error = delta(&packed, &tensor(mutant, device));
        assert!(
            error > 100. * tolerance,
            "undiscriminated encoder control {name}: {error}"
        );
    }
    device.synchronize().unwrap();
}
#[test]
fn frozen_moss_edit_encoder_cpu() {
    run(&Device::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn frozen_moss_edit_encoder_cuda() {
    assert!(!candle_gen::candle_core::cuda_backend::gemm_reduced_precision_f32());
    run(&Device::new_cuda(0).expect("admitted CUDA lane logical device 0"));
}
