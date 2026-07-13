//! SCAIL-2 28-channel mask-build parity gate (sc-5443 / sc-5446).
//!
//! Compares [`mlx_gen_scail2::extract_and_compress_mask_to_latent`] against the upstream
//! `wan/utils/scail_utils.py` `extract_and_compress_mask_to_latent` on a synthetic color-coded mask.
//! Fixtures are generated on the Mac by:
//!
//! ```text
//! SCAIL2_PARITY_DIR=~/.cache/scail2-parity \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_mask_parity_fixtures.py
//! ```
//!
//! `#[ignore]` (needs the locally-generated fixtures). Run with
//! `cargo test -p mlx-gen-scail2 -- --ignored`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{extract_and_compress_mask_to_latent, preprocess::TEMPORAL_STRIDE};
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
fn mask_parity() {
    let dir = parity_dir().join("mask");
    let io_path = dir.join("io.safetensors");
    assert!(
        io_path.exists(),
        "missing fixtures at {} — generate with \
         `SCAIL2_PARITY_DIR={} ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_mask_parity_fixtures.py`",
        dir.display(),
        parity_dir().display(),
    );

    let io = Weights::from_file(&io_path).unwrap();
    let out =
        extract_and_compress_mask_to_latent(io.require("mask").unwrap(), TEMPORAL_STRIDE).unwrap();
    let reference = io.require("output").unwrap();
    assert_eq!(out.shape(), reference.shape(), "mask latent shape");
    let (cos, max_abs) = compare(&out, reference);
    println!("28-ch mask: cosine {cos:.7}  max|Δ| {max_abs:.2e}");
    // Pure tensor op (threshold / products / exact 8× avg-pool / temporal pack) → bit-exact up to f32.
    assert!(max_abs < 1e-5, "mask max|Δ| {max_abs} exceeds 1e-5");
}
