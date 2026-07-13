//! SCAIL-2 CLIP visual-tower parity gate (sc-5443 / sc-5446).
//!
//! Compares [`mlx_gen_scail2::ScailClip::encode`] (f32) against the upstream open-CLIP
//! `VisionTransformer.visual(x, use_31_block=True)` penultimate path on a tiny-seeded tower (fused
//! qkv, exact-GELU, pre-norm, `use_31_block`). Reference fixtures are generated on the Mac by:
//!
//! ```text
//! SCAIL2_PARITY_DIR=~/.cache/scail2-parity \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_parity_fixtures.py
//! ```
//!
//! `#[ignore]` (needs the locally-generated fixtures). Run with
//! `cargo test -p mlx-gen-scail2 -- --ignored`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{ClipVisionConfig, ScailClip};
use mlx_rs::{Array, Dtype};

fn parity_dir() -> PathBuf {
    std::env::var("SCAIL2_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/scail2-parity")
        })
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn compare(a: &Array, b: &Array) -> (f32, f32) {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut max_abs = 0f32;
    for (x, y) in va.iter().zip(vb.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    ((dot / (na.sqrt() * nb.sqrt())) as f32, max_abs)
}

#[test]
#[ignore = "needs locally-generated fixtures (see module doc); run with --ignored on macOS"]
fn clip_parity_tiny_seeded() {
    let dir = parity_dir().join("clip");
    let model_path = dir.join("model.safetensors");
    assert!(
        model_path.exists(),
        "missing fixtures at {} — generate with \
         `SCAIL2_PARITY_DIR={} ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_parity_fixtures.py`",
        dir.display(),
        parity_dir().display(),
    );

    // Tiny tower matching _gen_clip_parity_fixtures.py: image 28 / patch 14 → 4 patches + cls = 5
    // tokens; num_layers 3 → use_31_block runs 2 blocks.
    let cfg = ClipVisionConfig {
        image_size: 28,
        patch_size: 14,
        dim: 128,
        num_heads: 2,
        num_layers: 3,
        eps: 1e-5,
    };
    let w = Weights::from_file(&model_path).unwrap();
    let clip = ScailClip::from_weights(&w, &cfg).unwrap();

    let io = Weights::from_file(dir.join("io.safetensors")).unwrap();
    let out = clip.encode(io.require("pixel").unwrap()).unwrap();
    let reference = io.require("output").unwrap();
    assert_eq!(out.shape(), reference.shape(), "clip feature shape");
    let (cos, max_abs) = compare(&out, reference);
    println!("clip penultimate: cosine {cos:.7}  max|Δ| {max_abs:.5}");
    assert!(
        cos > 0.9999,
        "clip cosine {cos} below 0.9999 (max|Δ| {max_abs})"
    );
}
