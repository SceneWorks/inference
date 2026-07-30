//! Real Mage-VAE encoder parity and deterministic mean-latent round trip.

mod common;

use std::path::{Path, PathBuf};

use common::error;
use mlx_gen::weights::Weights;
use mlx_gen_mage::vae::{self, VaePart};
use mlx_rs::Dtype;

fn snapshot() -> PathBuf {
    std::env::var("MAGE_SNAPSHOT")
        .expect("set MAGE_SNAPSHOT to a complete Mage-Flow snapshot")
        .into()
}

fn golden_dir() -> PathBuf {
    std::env::var("MAGE_VAE_GOLDEN_DIR")
        .expect("set MAGE_VAE_GOLDEN_DIR to the directory containing Mage-VAE CPU goldens")
        .into()
}

fn check_moments(codec: &vae::MageVae, path: &Path, max_gate: f32, mean_gate: f32) {
    let golden = Weights::from_file(path).unwrap();
    let pixels = golden.require("pixels").unwrap();
    let moments = codec.encode_moments(pixels).unwrap();
    mlx_rs::transforms::eval([&moments.mean, &moments.logvar]).unwrap();
    for (label, got, want) in [
        ("mean", &moments.mean, golden.require("enc_mean").unwrap()),
        (
            "logvar",
            &moments.logvar,
            golden.require("enc_logvar").unwrap(),
        ),
    ] {
        let (max_abs, _, mean_rel) = error(got, want);
        println!(
            "{} {label}: max_abs={max_abs:.6} mean_rel={mean_rel:.6}",
            path.file_name().unwrap().to_string_lossy()
        );
        assert!(
            max_abs <= max_gate && mean_rel <= mean_gate,
            "{} {label}: max_abs={max_abs} mean_rel={mean_rel}",
            path.display()
        );
    }
    assert!(
        moments
            .logvar
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .all(|&value| (-20.0..=10.0).contains(&value)),
        "logvar escaped the reference clamp"
    );
}

#[test]
#[ignore = "needs real VAE weights, CPU goldens, and an authorized Metal device"]
fn encoder_matches_five_geometry_f32_and_bf16_goldens() {
    let dir = golden_dir();
    let f32_codec = vae::load(snapshot(), VaePart::Both, Dtype::Float32).unwrap();
    for name in [
        "mage_flow_vae_f32_256.safetensors",
        "mage_flow_vae_f32_1024.safetensors",
        "mage_flow_vae_f32_2048.safetensors",
        "mage_flow_vae_f32_512x2048.safetensors",
        "mage_flow_vae_f32_768x1152.safetensors",
    ] {
        check_moments(&f32_codec, &dir.join(name), 0.001, 0.0001);
    }

    let bf16_codec = vae::load(snapshot(), VaePart::Both, Dtype::Bfloat16).unwrap();
    check_moments(
        &bf16_codec,
        &dir.join("mage_flow_vae_golden.safetensors"),
        0.21,
        0.03,
    );
}

#[test]
#[ignore = "needs real VAE weights, CPU golden, and an authorized Metal device"]
fn mean_latent_round_trip_is_finite_and_shape_exact() {
    let golden =
        Weights::from_file(golden_dir().join("mage_flow_vae_f32_256.safetensors")).unwrap();
    let codec = vae::load(snapshot(), VaePart::Both, Dtype::Float32).unwrap();
    let mean = codec
        .encode_mean(golden.require("pixels").unwrap())
        .unwrap();
    let decoded = codec.decode(&mean).unwrap();
    mlx_rs::transforms::eval([&mean, &decoded]).unwrap();
    assert_eq!(mean.shape(), [1, 128, 16, 16]);
    assert_eq!(decoded.shape(), [1, 3, 256, 256]);
    assert!(
        decoded
            .as_slice::<f32>()
            .iter()
            .all(|value| value.is_finite()),
        "mean-latent round trip produced non-finite pixels"
    );
}
