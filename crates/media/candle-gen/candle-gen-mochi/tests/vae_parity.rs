//! **Real-weight CUDA** AsymmVAE-decoder parity for Mochi 1 (A5, sc-11989) — the candle twin of
//! `mlx-gen-mochi`'s ignored `vae_parity` gate. Gated on `feature = "cuda"` + `#[ignore]`d. Feeds the
//! golden's teacher-forced `denormalized_latents` through the f32 decoder and checks the decoded
//! `video` reproduces `mochi_vae_golden.safetensors` (pixel space).
//!
//! Windows run:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p candle-gen-mochi --features cuda --test vae_parity -- --ignored --nocapture`
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::Weights;
use candle_gen_mochi::MochiVaeDecoder;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/mochi_vae_golden.safetensors"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn max_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn mean_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (vae safetensors) + tools/golden/mochi_vae_golden.safetensors (CUDA)"]
fn vae_decode_matches_golden() {
    let device = candle_gen::default_device().unwrap();
    let root = snapshot_dir();
    let dec = MochiVaeDecoder::load(&root, &device).expect("load vae decoder");
    let g = Weights::from_file(Path::new(GOLDEN), &device, DType::F32).expect("vae golden");

    // Teacher-forced: feed the golden's de-normalized latent straight into the decoder.
    let denorm = g.require("denormalized_latents").unwrap();
    let video = dec.decode_denormalized(&denorm).expect("decode");

    let want = g.require("video").unwrap();
    assert_eq!(video.dims(), want.dims(), "decoded video shape");

    // A valid (f32) golden video is ~[-1, 1]; if the provisioned golden is out of range it was dumped
    // in bf16 (the Mochi AsymmVAE is numerically f32-only) and is NOT a valid parity target — surface
    // that rather than gate against it (the MLX vae_parity finding).
    let want_range = max_abs(&want);
    assert!(
        want_range < 1.1,
        "golden `video` range is ±{want_range:.2} — this is a bf16 decode (the Mochi AsymmVAE is \
         f32-only). Re-provision the golden with the f32-fixed dump before running this gate."
    );

    let want_f = want.to_dtype(DType::F32).unwrap();
    let video_f = video.to_dtype(DType::F32).unwrap();
    let diff = (&video_f - &want_f).unwrap();
    let max_px = max_abs(&diff);
    let mean_px = mean_abs(&diff);
    eprintln!("VAE decode: max_px {max_px:.3e}  mean_px {mean_px:.3e}  (video ~[-1,1])");

    // Cross-impl pixel-space tolerance (the MLX vae_parity bar): a real bug (wrong unpatchify, pad mode,
    // groupnorm, or frame-drop) diverges by O(1) in pixel space.
    assert!(
        mean_px < 5e-3,
        "VAE mean pixel error {mean_px:.3e} too high"
    );
    assert!(max_px < 5e-2, "VAE max pixel error {max_px:.3e} too high");
}
