//! Shared helpers for the Iris parity tests: fixture paths, tensor readback and the peak-relative
//! comparison every gate reports through.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::IrisConfig;
use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

/// `tests/fixtures/` of this crate.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The miniature generation backbone (`config.yaml` + `model.safetensors`).
pub fn tiny_backbone() -> PathBuf {
    fixtures().join("tiny-snapshot/iris")
}

/// The miniature Qwen3-VL text-encoder snapshot.
pub fn tiny_text_encoder() -> PathBuf {
    fixtures().join("tiny-snapshot/text_encoder")
}

pub fn tiny_config() -> IrisConfig {
    IrisConfig::from_dir(&tiny_backbone()).expect("tiny config.yaml parses")
}

/// Run `f` with the MLX **CPU** stream as this thread's default. MLX's Metal GEMM accumulates f32
/// in a reduced-precision (TF32/bf16-class, ~1e-3 relative) path, so an FP32 parity gate against
/// upstream's FP32 CPU oracle runs on the CPU stream, where f32 is f32 (measured: the miniature
/// backbone matches to ~3e-5 there and only ~1e-2 on the GPU stream).
pub fn on_cpu<T>(f: impl FnOnce() -> T) -> T {
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), f)
}

pub fn fixture(name: &str) -> Weights {
    Weights::from_file(fixtures().join(name)).expect("committed fixture loads")
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

pub fn host_i32(a: &Array) -> Vec<i32> {
    let n: i32 = a.shape().iter().product();
    a.as_dtype(Dtype::Int32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<i32>()
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
