//! Shared helpers for the Qwen-Image 2.1 parity tests: fixture paths, tensor readback, and the
//! peak-relative comparison every component gate reports through.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

/// `tests/fixtures/` of this crate.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The committed miniature snapshot in the exact `Qwen/Qwen-Image-2.1` layout.
pub fn tiny_snapshot() -> PathBuf {
    fixtures().join("tiny-snapshot")
}

pub fn fixture(name: &str) -> Weights {
    Weights::from_file(fixtures().join(name)).expect("committed fixture loads")
}

pub fn meta_usize(w: &Weights, key: &str) -> usize {
    w.metadata(key)
        .unwrap_or_else(|| panic!("fixture metadata {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("fixture metadata {key} is an integer"))
}

pub fn meta_f32(w: &Weights, key: &str) -> f32 {
    w.metadata(key)
        .unwrap_or_else(|| panic!("fixture metadata {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("fixture metadata {key} is a float"))
}

pub fn meta_str<'a>(w: &'a Weights, key: &str) -> &'a str {
    w.metadata(key)
        .unwrap_or_else(|| panic!("fixture metadata {key}"))
}

pub fn host_f32(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// `(max |got − want|, max |want|, mean |got − want|)`.
pub fn errors(got: &Array, want: &Array) -> (f32, f32, f32) {
    assert_eq!(got.shape(), want.shape(), "shape mismatch");
    let (g, w) = (host_f32(got), host_f32(want));
    let mut max_abs = 0f32;
    let mut sum = 0f32;
    for (a, b) in g.iter().zip(&w) {
        let d = (a - b).abs();
        assert!(d.is_finite(), "non-finite difference ({a} vs {b})");
        max_abs = max_abs.max(d);
        sum += d;
    }
    let peak = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    (max_abs, peak, sum / g.len().max(1) as f32)
}

/// Assert `max |got − want| <= tol · max(1, max |want|)` and print the measured numbers so a
/// tolerance is never tighter than the evidence behind it.
pub fn assert_close(name: &str, got: &Array, want: &Array, tol: f32) {
    let (max_abs, peak, mean) = errors(got, want);
    let bound = tol * peak.max(1.0);
    eprintln!("{name}: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} peak={peak:.3e} bound={bound:.3e}");
    assert!(
        max_abs <= bound,
        "{name}: max|Δ|={max_abs:.3e} exceeds {bound:.3e} (tol {tol:.0e} × peak {peak:.3e})"
    );
}
