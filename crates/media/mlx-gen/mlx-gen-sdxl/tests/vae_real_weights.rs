//! sc-2400 S4/S6: SDXL VAE decode + encode parity vs the vendored Apple reference (f32).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from `tools/dump_sdxl_vae_golden.py`.
//! Run with:
//!   cargo test -p mlx-gen-sdxl --release --test vae_real_weights -- --ignored --nocapture

mod common;

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder};
use mlx_gen_sdxl::{decode_image, decoded_to_image, load_vae, SdxlLatentDecoder};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_vae_golden.safetensors"
);

use common::snapshot;

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

#[test]
#[ignore = "needs the real SDXL snapshot + VAE golden"]
fn vae_decode_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    let decoded = vae.decode(g.require("latents").unwrap()).unwrap();
    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");
    let pr = peak_rel(&decoded, golden);
    println!("vae decode {:?}: peak_rel={pr:.3e}", decoded.shape());
    assert!(pr < 5e-3, "VAE decode diverged: peak_rel {pr:.3e}");
    println!("✓ SDXL VAE decode matches the vendored reference (f32)");
}

/// SC-18309 N1 hardware gate: execute the exact pre-seam native expression and the full engine
/// no-override route over the same real normalized latent. Equality is bitwise, not tolerance-based:
/// the seam may only transpose around the native VAE and must not alter normalization or readback.
#[test]
#[ignore = "needs the real SDXL snapshot + VAE golden"]
fn native_decode_seam_is_byte_exact_to_pre_seam_engine() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();
    let latents_nhwc = g.require("latents").unwrap();

    let legacy_tensor = vae.decode(latents_nhwc).unwrap();
    let latents_nchw = latents_nhwc.transpose_axes(&[0, 3, 1, 2]).unwrap();
    let seam_nchw = SdxlLatentDecoder::new(&vae).decode(&latents_nchw).unwrap();
    let seam_nhwc = seam_nchw.transpose_axes(&[0, 2, 3, 1]).unwrap();
    assert_eq!(seam_nhwc.shape(), legacy_tensor.shape());
    assert_eq!(
        seam_nhwc.reshape(&[-1]).unwrap().as_slice::<f32>(),
        legacy_tensor.reshape(&[-1]).unwrap().as_slice::<f32>(),
        "native seam changed real-VAE output bytes"
    );

    let legacy_image = decoded_to_image(&legacy_tensor).unwrap();
    let engine_image = decode_image(&vae, latents_nhwc, None).unwrap();
    assert_eq!(engine_image, legacy_image, "engine RGB bytes changed");

    let malformed = Array::from_slice(&[1.0f32], &[1]);
    let cancel = CancelFlag::new();
    cancel.cancel();
    for cfg in [
        mlx_gen::tiling::TilingConfig::spatial_only(8, 2),
        mlx_gen::tiling::TilingConfig::spatial_only(4096, 64),
    ] {
        assert!(matches!(
            vae.decode_tiled(&malformed, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            SdxlLatentDecoder::new(&vae).decode_tiled(&malformed, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }
}

#[test]
#[ignore = "needs the real SDXL snapshot + VAE golden"]
fn vae_encode_mean_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    let mean = vae.encode_mean(g.require("image").unwrap()).unwrap();
    let golden = g.require("enc_mean").unwrap();
    assert_eq!(mean.shape(), golden.shape(), "enc_mean shape");
    let pr = peak_rel(&mean, golden);
    println!("vae encode_mean {:?}: peak_rel={pr:.3e}", mean.shape());
    assert!(pr < 5e-3, "VAE encode diverged: peak_rel {pr:.3e}");
    println!("✓ SDXL VAE encode (mean) matches the vendored reference (f32)");
}
