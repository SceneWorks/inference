//! Independent frozen Diffusers 0.38 CPU pixels for MOSS's actual asymmetric small decoder.
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/small_vae/fixture.json")).unwrap()
}
fn values(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn recipe_value(recipe: &Value, index: u32) -> f32 {
    let mut bits = (index + 1) ^ recipe["seed"].as_u64().unwrap() as u32;
    bits ^= bits << 13;
    bits ^= bits >> 17;
    bits ^= bits << 5;
    let signed = (bits & 65535) as i32 - 32768;
    match recipe["kind"].as_str().unwrap() {
        "bn_mean" => ((index % 17) as i32 - 8) as f32 / 64.0,
        "bn_var" => 1.0 + ((index * 7) % 31) as f32 / 32.0,
        "norm" => 1.0 + signed as f32 / 262144.0,
        "bias" => signed as f32 / 1048576.0,
        "weight" => signed as f32 / 8388608.0,
        "post_quant_identity" => {
            signed as f32 / 8388608.0 + if index / 32 == index % 32 { 1.0 } else { 0.0 }
        }
        other => panic!("unknown frozen weight recipe {other}"),
    }
}
fn pixel_error(actual: &[f32], expected: &[f32]) -> Result<f32, &'static str> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("pixel count mismatch");
    }
    if actual.iter().chain(expected).any(|v| !v.is_finite()) {
        return Err("nonfinite pixel");
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max))
}

#[test]
fn cpu_recipe_and_pixel_oracle_reject_corruption_and_author_mutants() {
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("fixtures/small_vae/fixture.json"))
        ),
        "89572df808130ece07ce344982aed1c2f3462ebbd64cd5138b49d23852310a0c"
    );
    let f = fixture();
    assert_eq!(
        f["config"]["decoder_block_out_channels"],
        serde_json::json!([96, 192, 384, 384])
    );
    assert_eq!(f["diffusers_version"], "0.38.0");
    let expected = values(&f["pixels_nchw"]["values"]);
    assert_eq!(f["pixels_nchw"]["shape"], serde_json::json!([1, 3, 16, 16]));
    assert_eq!(expected.len(), 768);
    assert_eq!(pixel_error(&expected, &expected).unwrap(), 0.0);
    assert!(pixel_error(&expected[..767], &expected).is_err());
    let mut nonfinite = expected.clone();
    nonfinite[7] = f32::NAN;
    assert!(pixel_error(&nonfinite, &expected).is_err());
    nonfinite[7] = f32::INFINITY;
    assert!(pixel_error(&expected, &nonfinite).is_err());
    for recipe in f["weights"].as_array().unwrap() {
        let name = recipe["name"].as_str().unwrap();
        let probe = values(&f["weight_prefix_probes"][name]);
        for (i, value) in probe.into_iter().enumerate() {
            assert_eq!(recipe_value(recipe, i as u32), value, "{name}[{i}]");
        }
    }
    let tolerance = f["absolute_tolerance"].as_f64().unwrap() as f32;
    for mutant in ["wrong_patch_order", "missing_bn"] {
        let wrong = values(&f["mutants"][mutant]["values"]);
        assert!(
            pixel_error(&wrong, &expected).unwrap() > 20.0 * tolerance,
            "{mutant}"
        );
    }
}

#[test]
#[ignore = "requires coordinator-admitted Metal ownership; never execute as a CPU fixture"]
fn actual_moss_small_decoder_matches_frozen_author_pixels_on_metal() {
    assert_eq!(
        std::env::var("MOSS_NATIVE_TEST_ADMITTED").as_deref(),
        Ok("1")
    );
    use mlx_gen::weights::Weights;
    use mlx_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
    use mlx_rs::Array;
    use std::collections::HashMap;
    mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
        let f = fixture();
        let mut tensors = HashMap::new();
        for recipe in f["weights"].as_array().unwrap() {
            let shape: Vec<i32> = recipe["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let count = shape.iter().map(|n| *n as usize).product::<usize>();
            let data: Vec<f32> = (0..count).map(|i| recipe_value(recipe, i as u32)).collect();
            tensors.insert(
                recipe["name"].as_str().unwrap().into(),
                Array::from_slice(&data, &shape),
            );
        }
        let weights = Weights::from_map(tensors);
        // A standard large decoder must fail on these actual small-decoder shapes.
        assert!(
            Flux2Vae::from_weights_decode_only(&weights, &Flux2VaeDecoderConfig::default())
                .is_err()
        );
        let decoder = Flux2Vae::from_weights_decode_only(
            &weights,
            &Flux2VaeDecoderConfig {
                block_out_channels: [96, 192, 384, 384],
            },
        )
        .unwrap();
        weights.materialize_accessed().unwrap();
        assert!(weights.unused_keys().is_empty());
        let packed = Array::from_slice(&values(&f["packed_nchw"]["values"]), &[1, 128, 1, 1])
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap();
        let actual = decoder
            .decode_packed_latents(&packed)
            .unwrap()
            .transpose_axes(&[0, 3, 1, 2])
            .unwrap();
        assert_eq!(actual.shape(), [1, 3, 16, 16]);
        mlx_rs::transforms::eval([&actual]).unwrap();
        let actual = actual.try_as_slice::<f32>().unwrap();
        let expected = values(&f["pixels_nchw"]["values"]);
        let error = pixel_error(actual, &expected).unwrap();
        let tolerance = f["absolute_tolerance"].as_f64().unwrap() as f32;
        assert!(
            error <= tolerance,
            "author CPU vs MLX Metal small-VAE max_abs={error}, tolerance={tolerance}"
        );
        eprintln!("MOSS small VAE author pixels: finite=768 max_abs={error} tolerance={tolerance}");
    });
}
