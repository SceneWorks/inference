//! Real-weight typed-cancellation conformance gate for `seedvr2` (story 6883, task 5).
//!
//! Binds the repo-wide typed-cancellation contract to the SeedVR2 super-resolution upscaler: drives
//! the registered `Generator` through the reusable testkit check
//! (`gen_core_testkit::check_cancellation_with`), which trips `req.cancel` at the first emitted
//! `Progress::Step` and asserts `generate()` returns the typed `Err(Error::Canceled)` within ≤2
//! further steps. The image-upscale path carries a `Conditioning::Reference` low-res input image.
//!
//! Snapshot-gated (skips when absent), matching the crate's `registry_e2e.rs` convention: it needs
//! the raw `numz/SeedVR2_comfyUI` checkpoint dir (the 3B fp16 DiT) under `MLX_GEN_MODELS_ROOT`.

use std::path::PathBuf;

use mlx_gen::{Conditioning, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_seedvr2::registry::MODEL_ID;

/// The raw `numz/SeedVR2_comfyUI` checkpoint snapshot dir (mirrors `registry_e2e.rs::raw_dir`):
/// returns `None` (→ skip) when the 3B checkpoint is not under `MLX_GEN_MODELS_ROOT`.
fn raw_dir() -> Option<PathBuf> {
    let base = PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").ok()?)
        .join("models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("seedvr2_ema_3b_fp16.safetensors")
        .exists()
        .then_some(snap)
}

/// A small synthetic low-res RGB8 input image (the `registry_e2e.rs` upscale source).
fn lr_image(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: (0..(w * h * 3)).map(|i| (i % 256) as u8).collect(),
    }
}

#[test]
fn seedvr2_honors_typed_cancellation() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw SeedVR2 3B checkpoint absent");
        return;
    };
    let gen = mlx_gen_seedvr2::provider_registry()
        .unwrap()
        .load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(snap)))
        .expect("load seedvr2");

    // Image upscale: a low-res Reference image → a 128×128 (÷16) target. `steps` is overridden for
    // headroom though SeedVR2 is a 1-step model — the helper trips on whatever step it emits.
    let req = GenerationRequest {
        width: 128,
        height: 128,
        steps: Some(6),
        seed: Some(7),
        conditioning: vec![Conditioning::Reference {
            image: lr_image(96, 96),
            strength: None,
        }],
        ..Default::default()
    };

    gen_core_testkit::check_cancellation_with(gen.as_ref(), &req)
        .expect("seedvr2 must honor the typed-cancellation contract");
}
