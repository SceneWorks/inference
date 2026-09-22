//! Shared helpers for the Qwen-Image 2.1 candle parity tests: fixture paths, tensor readback, and
//! the peak-relative comparison every component gate reports through.
//!
//! The fixtures and the miniature snapshot are the **same committed files** the MLX twin consumes
//! — one oracle for both backends — reached across the backend boundary by relative path, the way
//! `candle-gen-ltx/tests/connector_parity.rs` reaches `mlx-gen-ltx`'s golden.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use safetensors::{Dtype, SafeTensors};

/// `tests/fixtures/` of `mlx-gen-qwen-image-2-1` — shared with the MLX twin.
pub fn fixtures() -> PathBuf {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures"
    ))
    .to_path_buf()
}

/// The committed miniature snapshot in the exact `Qwen/Qwen-Image-2.1` layout.
pub fn tiny_snapshot() -> PathBuf {
    fixtures().join("tiny-snapshot")
}

/// A committed `.safetensors` fixture: its tensors (read back as CPU f32) plus its `__metadata__`.
pub struct Fixture {
    path: PathBuf,
    data: Vec<u8>,
    meta: HashMap<String, String>,
}

impl Fixture {
    fn open(path: PathBuf) -> Self {
        let data = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("committed fixture {} reads: {e}", path.display()));
        let meta = SafeTensors::read_metadata(&data)
            .unwrap_or_else(|e| panic!("fixture {} has a safetensors header: {e}", path.display()))
            .1
            .metadata()
            .clone()
            .unwrap_or_default();
        Self { path, data, meta }
    }

    /// Tensor `name` as a CPU f32 tensor, with the fixture's own shape.
    pub fn tensor(&self, name: &str) -> Tensor {
        let st = SafeTensors::deserialize(&self.data)
            .unwrap_or_else(|e| panic!("fixture {} deserializes: {e}", self.path.display()));
        let view = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("fixture lacks {name}"));
        let dtype = match view.dtype() {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::F64 => DType::F64,
            other => panic!("fixture {name} has unsupported dtype {other:?}"),
        };
        Tensor::from_raw_buffer(view.data(), dtype, view.shape(), &Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .unwrap_or_else(|e| panic!("fixture {name} converts to a candle tensor: {e}"))
    }

    /// A `__metadata__` entry.
    pub fn meta(&self, key: &str) -> &str {
        self.meta
            .get(key)
            .unwrap_or_else(|| panic!("fixture metadata {key}"))
            .as_str()
    }

    pub fn meta_usize(&self, key: &str) -> usize {
        self.meta(key)
            .parse()
            .unwrap_or_else(|_| panic!("fixture metadata {key} is an integer"))
    }

    pub fn meta_f32(&self, key: &str) -> f32 {
        self.meta(key)
            .parse()
            .unwrap_or_else(|_| panic!("fixture metadata {key} is a float"))
    }
}

/// Open the named fixture under [`fixtures`].
pub fn fixture(name: &str) -> Fixture {
    Fixture::open(fixtures().join(name))
}

pub fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("tensor reads back as f32")
}

/// `(max |got − want|, max |want|, mean |got − want|)`.
pub fn errors(got: &Tensor, want: &Tensor) -> (f32, f32, f32) {
    assert_eq!(got.dims(), want.dims(), "shape mismatch");
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
pub fn assert_close(name: &str, got: &Tensor, want: &Tensor, tol: f32) {
    let (max_abs, peak, mean) = errors(got, want);
    let bound = tol * peak.max(1.0);
    eprintln!("{name}: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} peak={peak:.3e} bound={bound:.3e}");
    assert!(
        max_abs <= bound,
        "{name}: max|Δ|={max_abs:.3e} exceeds {bound:.3e} (tol {tol:.0e} × peak {peak:.3e})"
    );
}
