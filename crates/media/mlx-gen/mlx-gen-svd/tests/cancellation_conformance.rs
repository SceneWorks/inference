//! Real-weight typed-cancellation conformance gate for `svd_xt` (story 6883, task 5).
//!
//! Binds the repo-wide typed-cancellation contract to the SVD provider: drives the registered
//! image→video `Generator` through the reusable testkit check
//! (`gen_core_testkit::check_cancellation_with`), which trips `req.cancel` at the first emitted
//! `Progress::Step` and asserts `generate()` returns the typed `Err(Error::Canceled)` within ≤2
//! further steps. `#[ignore]` because it needs the real `stabilityai/stable-video-diffusion-img2vid-xt`
//! checkpoint in the HF cache (matching the crate's existing `svd_provider_generates_video` smoke).

use std::path::PathBuf;

use mlx_gen::{Conditioning, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_svd::MODEL_ID;

/// The cached SVD checkpoint snapshot dir (mirrors `tests/registration.rs`).
fn svd_snapshot_dir() -> PathBuf {
    let cache = std::path::PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)"));
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path()
}

/// A small RGB gradient reference image (the `svd_provider_generates_video` conditioning source).
fn gradient_image(w: u32, h: u32) -> Image {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = (x * 255 / w) as u8;
            pixels[i + 1] = (y * 255 / h) as u8;
            pixels[i + 2] = 128;
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

#[test]
#[ignore = "needs the SVD checkpoint in the HF cache (loads the full f32 model)"]
fn svd_xt_honors_typed_cancellation() {
    let snap = svd_snapshot_dir();
    let gen = mlx_gen_svd::provider_registry()
        .unwrap()
        .load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(snap)))
        .expect("load svd");

    // Reference-only image→video request at the descriptor's minimum size (256², ÷16), with step
    // headroom so a honoring provider visibly stops before completion.
    let req = GenerationRequest {
        width: 256,
        height: 256,
        frames: Some(3),
        steps: Some(6),
        fps: Some(7),
        seed: Some(7),
        conditioning: vec![Conditioning::Reference {
            image: gradient_image(48, 48),
            strength: None,
        }],
        ..Default::default()
    };

    gen_core_testkit::check_cancellation_with(gen.as_ref(), &req)
        .expect("svd_xt must honor the typed-cancellation contract");
}
