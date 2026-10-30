//! SCAIL-2 CLIP **real-weight** parity gate (sc-5445 / sc-5446).
//!
//! Loads the real de-prefixed `clip.safetensors` (the stock open-CLIP XLM-RoBERTa ViT-H/14 visual
//! tower, 32 layers / 1280-dim) and runs the full CLIP path ([`mlx_gen_scail2::clip_preprocess`] →
//! [`mlx_gen_scail2::ScailClip::encode`]) against the upstream `VisionTransformer.visual(use_31_block
//! =True)` reference. The tiny-seeded `clip_parity.rs` proves the algorithm; this proves the real
//! weights load + run.
//!
//! Generate the snapshot's `clip.safetensors` + the reference on the Mac:
//!
//! ```text
//! SCAIL2_PARITY_DIR=~/.cache/scail2-parity \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_realweight_fixtures.py
//! ```
//!
//! `#[ignore]` (needs the ~2.5 GB weights + fixtures, off CI). Run with
//! `cargo test -p mlx-gen-scail2 --test integration clip_real_parity:: -- --ignored --nocapture`.
//! `SCAIL2_SNAPSHOT_DIR` overrides the snapshot dir (default `~/.cache/scail2-mlx-convert`).

use std::path::PathBuf;

use mlx_gen::testing::{cosine_host_f64, env_path_or_home, Denom};
use mlx_gen::weights::Weights;
use mlx_gen_scail2::{clip_preprocess, ClipVisionConfig, ScailClip};

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn parity_dir() -> PathBuf {
    std::env::var("SCAIL2_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".cache/scail2-parity"))
}

fn snapshot_dir() -> PathBuf {
    env_path_or_home("SCAIL2_SNAPSHOT_DIR", ".cache/scail2-mlx-convert")
}

#[test]
#[ignore = "real ~2.5 GB ViT-H/14 weights + fixtures (see module doc); run with --ignored on macOS"]
fn clip_real_parity() {
    let clip_path = snapshot_dir().join("clip.safetensors");
    let io_path = parity_dir().join("clip_real/io.safetensors");
    assert!(
        clip_path.exists() && io_path.exists(),
        "missing real CLIP weights/fixtures — generate with \
         `~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_realweight_fixtures.py` \
         ({} / {})",
        clip_path.display(),
        io_path.display(),
    );

    let w = Weights::from_file(&clip_path).unwrap();
    let clip = ScailClip::from_weights(&w, &ClipVisionConfig::vit_h_14()).unwrap();

    let io = Weights::from_file(&io_path).unwrap();
    let pixel = clip_preprocess(io.require("image").unwrap(), 224).unwrap();
    let out = clip.encode(&pixel).unwrap();
    let reference = io.require("output").unwrap();
    assert_eq!(out.shape(), reference.shape(), "real CLIP feature shape");
    let cos = cosine_host_f64(&out, reference, Denom::Exact) as f32;
    println!("real ViT-H/14 penultimate (clip_preprocess → encode): cosine {cos:.7}");
    // f32-vs-f32 over 31 layers; residual is MLX Metal matmul reduced precision.
    assert!(cos > 0.999, "real CLIP cosine {cos} below 0.999");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED clip_real_parity::clip_real_parity");
}
