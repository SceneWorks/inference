//! sc-2344: full VAE decoder-assembly parity vs the fork, plus the `Vae::decode` scale/shift
//! wrapper. Fixture `tests/fixtures/vae_decoder.safetensors` ← `tools/dump_z_image_vae_decoder.py`
//! (small decoder mirroring `Decoder.__call__`: conv_in → mid → 2 up-blocks → norm-out →
//! SiLU → conv_out). Tol 1e-2 (Metal fp32 convs).

use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder};
use mlx_gen_z_image::vae::{Decoder, Vae, VaeDecoderConfig};
use mlx_rs::ops::{add, all_close, multiply};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vae_decoder.safetensors"
);

fn small_cfg() -> VaeDecoderConfig {
    VaeDecoderConfig {
        up_blocks: vec![(3, true), (3, false)],
    }
}

#[test]
fn decoder_assembly_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let decoder = Decoder::from_weights(&w, "", &small_cfg()).unwrap();

    let latent = w.require("in.latent").unwrap();
    let image = decoder.forward(latent).unwrap();
    let want = w.require("out.image").unwrap();

    assert_eq!(image.shape(), want.shape(), "decoder output shape");
    assert!(
        all_close(&image, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "VAE decoder assembly diverged from the fork"
    );
}

#[test]
fn vae_decode_applies_scale_shift_and_frame_axis() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let vae = Vae::from_weights(&w, "", &small_cfg()).unwrap();

    let latent = w.require("in.latent").unwrap(); // (1,16,8,8)
    let sh = latent.shape();
    let latent5 = latent.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap(); // (1,16,1,8,8)

    let decoded = vae.decode(&latent5).unwrap();
    assert_eq!(
        decoded.shape(),
        &[1, 3, 1, 16, 16],
        "decode restores the frame axis"
    );

    // Reference: decoder.forward((latent / scaling) + shift), with the frame axis added back.
    let scaled = add(
        multiply(
            latent,
            Array::from_slice(&[1.0 / Vae::SCALING_FACTOR], &[1]),
        )
        .unwrap(),
        Array::from_slice(&[Vae::SHIFT_FACTOR], &[1]),
    )
    .unwrap();
    let ref_img = vae.decoder().forward(&scaled).unwrap();
    let d = ref_img.shape();
    let ref5 = ref_img.reshape(&[d[0], d[1], 1, d[2], d[3]]).unwrap();

    assert!(
        all_close(&decoded, &ref5, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "Vae::decode scale/shift/frame-axis wrapper is wrong"
    );
}

/// SC-18309 N1: the real tiny VAE fixture's trait default is bit-for-bit the historical inherent
/// decode, including scale/shift de-normalization and the singleton output frame. This is an exact
/// comparison (not the tolerance used for cross-framework parity above).
#[test]
fn native_trait_decode_is_byte_exact_to_inherent_decode() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let vae = Vae::from_weights(&w, "", &small_cfg()).unwrap();
    let latent = w.require("in.latent").unwrap();

    let legacy = Vae::decode(&vae, latent).unwrap();
    let seam = LatentDecoder::decode(&vae, latent).unwrap();
    assert_eq!(seam.shape(), legacy.shape());
    assert_eq!(
        seam.reshape(&[-1]).unwrap().as_slice::<f32>(),
        legacy.reshape(&[-1]).unwrap().as_slice::<f32>()
    );
}

/// A pre-tripped cancel wins before rank validation, tiling selection, de-normalization, or the
/// monolithic/head path. Both a plan that would tile and one that would fall back must return the
/// same typed cancellation even for a deliberately malformed latent.
#[test]
fn native_tiled_decode_rejects_pre_cancel_before_all_tensor_work() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let vae = Vae::from_weights(&w, "", &small_cfg()).unwrap();
    let malformed = Array::from_slice(&[1.0f32], &[1]);
    let cancel = CancelFlag::new();
    cancel.cancel();
    for cfg in [
        mlx_gen::tiling::TilingConfig::spatial_only(8, 2),
        mlx_gen::tiling::TilingConfig::spatial_only(4096, 64),
    ] {
        assert!(matches!(
            LatentDecoder::decode_tiled(&vae, &malformed, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            Vae::decode_tiled(&vae, &malformed, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }
}

/// sc-19753: exercise the production `Vae::decode_tiled` seam, not only the shared convolution
/// primitive. The fixture's position-dependent latent makes the old whole-tail/per-crop GroupNorm
/// implementation diverge, while layer-wise global normalization tracks the dense decode.
#[test]
fn vae_tiled_decode_tracks_dense_with_global_group_norm() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let vae = Vae::from_weights(&w, "", &small_cfg()).unwrap();
    let latent = w.require("in.latent").unwrap();
    let sh = latent.shape();
    let latent5 = latent.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();

    let dense = vae.decode(&latent5).unwrap();
    let tiled = vae
        .decode_tiled(&latent5, &TilingConfig::spatial_only(8, 0), None)
        .unwrap();
    dense.eval().unwrap();
    tiled.eval().unwrap();
    assert_eq!(tiled.shape(), dense.shape());

    let max_delta = dense
        .as_slice::<f32>()
        .iter()
        .zip(tiled.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_delta < 2e-3,
        "layer-wise tiled VAE diverged from dense decode: max|delta|={max_delta:.3e}"
    );
}

#[test]
fn vae_tiled_decode_honors_pretripped_cancel() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let vae = Vae::from_weights(&w, "", &small_cfg()).unwrap();
    let malformed = Array::from_slice(&[1.0_f32], &[1]);
    let cancel = CancelFlag::new();
    cancel.cancel();

    for cfg in [
        TilingConfig::spatial_only(8, 0),
        TilingConfig::spatial_only(4096, 64),
    ] {
        let result = vae.decode_tiled(&malformed, &cfg, Some(&cancel));
        assert!(matches!(result, Err(Error::Canceled)));
    }
}
