//! Real-weight CUDA parity gate for the LTX CausalVideoAutoencoder encoder (sc-13865).
//!
//! The committed golden was produced by the MLX reference from the real LTX-2.3 encoder. This test
//! loads the matching split MLX decoder/encoder weights through candle's production remappers and
//! compares normalized latents with a PSNR tolerance.
//!
//! Run:
//! `LTX_BASE_DIR=<snapshot>/q8 cargo test -p candle-gen-ltx --features cuda --test integration --
//! vae_encode_parity:: --ignored --nocapture`

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{safetensors, DType, Device, Result as CoreResult, Tensor};
use candle_gen_ltx::config::LATENT_CHANNELS;
use candle_gen_ltx::tier::TierPaths;
use candle_gen_ltx::vae::LtxVideoVae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx_vae_golden.safetensors"
);

fn psnr_db(got: &Tensor, reference: &Tensor) -> CoreResult<f64> {
    let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let reference = reference
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(got.len(), reference.len());
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut squared_error = 0f64;
    for (&actual, &expected) in got.iter().zip(reference.iter()) {
        min = min.min(expected);
        max = max.max(expected);
        let error = actual as f64 - expected as f64;
        squared_error += error * error;
    }
    let mse = squared_error / got.len() as f64;
    if mse == 0.0 {
        return Ok(f64::INFINITY);
    }
    let peak = (max - min).max(1e-12) as f64;
    Ok(20.0 * (peak / mse.sqrt()).log10())
}

#[test]
#[ignore = "needs real LTX-2.3 split VAE weights and a CUDA GPU"]
fn encode_matches_mlx_golden_psnr() -> candle_gen::Result<()> {
    let base = std::env::var("LTX_BASE_DIR").expect("set LTX_BASE_DIR to the q4/q8 tier directory");
    let paths = TierPaths::detect(std::path::Path::new(&base), None)
        .expect("LTX_BASE_DIR must contain quantize_config.json and transformer.safetensors");
    let device = Device::new_cuda(0)?;
    let decoder = paths.vae_vb(DType::F32, &device)?;
    let encoder = paths.vae_encoder_vb(DType::F32, &device)?;
    let vae =
        LtxVideoVae::new_with_encoder(decoder.pp("vae"), encoder.pp("vae"), LATENT_CHANNELS, 4)?;

    let golden = safetensors::load(GOLDEN, &device)?;
    let input = golden["enc_in"].to_dtype(DType::F32)?;
    let reference = golden["enc_out"].to_dtype(DType::F32)?;
    let got = vae.encode(&input)?;
    assert_eq!(got.dims(), &[1, LATENT_CHANNELS, 2, 2, 2]);
    assert_eq!(got.dims(), reference.dims());
    let psnr = psnr_db(&got, &reference)?;
    eprintln!("LTX encoder MLX-golden PSNR: {psnr:.2} dB");
    assert!(
        psnr >= 35.0,
        "LTX encoder diverged from the real MLX golden: PSNR {psnr:.2} dB < 35 dB"
    );
    Ok(())
}
