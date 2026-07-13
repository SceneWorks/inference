//! Real-weight typed-cancellation conformance gate for `scail2_14b` (story 6883, task 5).
//!
//! Binds the repo-wide typed-cancellation contract to the SCAIL-2 character-animation renderer:
//! drives the registered `Generator` through the reusable testkit check
//! (`gen_core_testkit::check_cancellation_with`), which trips `req.cancel` at the first emitted
//! `Progress::Step` and asserts `generate()` returns the typed `Err(Error::Canceled)` within ≤2
//! further steps. The request carries the full single-character animation conditioning (Reference +
//! Mask + ControlClip), mirroring `generate_smoke.rs::run_mode`.
//!
//! `#[ignore]` because it loads the real ~46 GB assembled snapshot (the crate's e2e gating
//! convention); set `SCAIL2_SNAPSHOT_DIR` or populate `~/.cache/scail2-mlx-convert`.

use std::path::PathBuf;

use mlx_gen::{Conditioning, GenerationRequest, Image, LoadSpec, ReplacementMode, WeightsSource};
use mlx_gen_scail2::pipeline::MODEL_ID;

fn snapshot_dir() -> PathBuf {
    std::env::var("SCAIL2_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/scail2-mlx-convert")
        })
}

/// A deterministic gradient image (reference / driving frame stand-in; `generate_smoke.rs`).
fn gradient(w: usize, h: usize, phase: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[
                ((x + phase) % 256) as u8,
                ((y + phase) % 256) as u8,
                ((x + y + phase) % 256) as u8,
            ]);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// A two-region color-coded mask (`generate_smoke.rs`).
fn color_mask(w: usize, h: usize, split: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let rgb = if x < split {
                [255u8, 255, 255]
            } else {
                [255u8, 0, 0]
            };
            pixels.extend_from_slice(&rgb);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

#[test]
#[ignore = "real ~46 GB snapshot; run with --ignored on macOS (set SCAIL2_SNAPSHOT_DIR)"]
fn scail2_honors_typed_cancellation() {
    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists(),
        "missing snapshot at {} — assemble it first (sc-5445)",
        root.display()
    );

    let (w, h) = (256usize, 256usize);
    let n_frames = 13usize;
    let reference = gradient(w, h, 0);
    let ref_mask = color_mask(w, h, w / 2);
    let driving: Vec<Image> = (0..n_frames).map(|i| gradient(w, h, i * 7)).collect();
    let masks: Vec<Image> = (0..n_frames)
        .map(|i| color_mask(w, h, w / 4 + (i % (w / 2))))
        .collect();

    // Full single-character animation conditioning, with step headroom so a honoring provider
    // visibly stops before completion.
    let req = GenerationRequest {
        prompt: "a person dancing, cinematic".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: w as u32,
        height: h as u32,
        steps: Some(6),
        seed: Some(7),
        fps: Some(16),
        conditioning: vec![
            Conditioning::Reference {
                image: reference,
                strength: None,
            },
            Conditioning::Mask { image: ref_mask },
            Conditioning::ControlClip {
                frames: driving,
                mask: masks,
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ],
        ..Default::default()
    };

    let gen = mlx_gen_scail2::provider_registry()
        .unwrap()
        .load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(root)))
        .expect("load scail2");

    gen_core_testkit::check_cancellation_with(gen.as_ref(), &req)
        .expect("scail2_14b must honor the typed-cancellation contract");
}
