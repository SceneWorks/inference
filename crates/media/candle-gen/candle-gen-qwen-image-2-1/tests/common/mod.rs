//! Shared helpers for the Qwen-Image 2.1 parity tests: fixture paths, tensor readback, and the
//! peak-relative comparison every component gate reports through.
//!
//! The fixtures are the **same committed `.safetensors` the MLX twin reads** — they live in
//! `crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/` and are reached across the backend
//! boundary by relative path (the convention `candle-gen-ltx`, `candle-gen-mochi` and
//! `candle-gen-krea` already use), so both backends are held to one numeric reference instead of two
//! copies that can drift.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

/// `tests/fixtures/` of the MLX twin — the single home of the Qwen-Image 2.1 parity fixtures.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures")
}

/// The committed miniature snapshot in the exact `Qwen/Qwen-Image-2.1` layout.
pub fn tiny_snapshot() -> PathBuf {
    fixtures().join("tiny-snapshot")
}

/// One committed `.safetensors` fixture: its tensors (read to CPU f32) and its `__metadata__`.
pub struct Fixture {
    tensors: HashMap<String, Tensor>,
    metadata: HashMap<String, String>,
}

impl Fixture {
    pub fn open(name: &str) -> Self {
        let path = fixtures().join(name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("committed fixture {} loads: {e}", path.display()));
        let parsed = safetensors::SafeTensors::deserialize(&bytes)
            .unwrap_or_else(|e| panic!("fixture {} parses: {e}", path.display()));
        let mut tensors = HashMap::new();
        for (key, view) in parsed.tensors() {
            let tensor = Tensor::from_raw_buffer(
                view.data(),
                dtype_of(view.dtype(), &key),
                view.shape(),
                &Device::Cpu,
            )
            .unwrap_or_else(|e| panic!("fixture tensor {key}: {e}"))
            .to_dtype(DType::F32)
            .unwrap_or_else(|e| panic!("fixture tensor {key} casts to f32: {e}"));
            tensors.insert(key, tensor);
        }
        let metadata = safetensors::SafeTensors::read_metadata(&bytes)
            .unwrap_or_else(|e| panic!("fixture {} header parses: {e}", path.display()))
            .1
            .metadata()
            .clone()
            .unwrap_or_default();
        Self { tensors, metadata }
    }

    /// One fixture tensor, on CPU in f32.
    pub fn tensor(&self, name: &str) -> Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("fixture lacks {name}"))
            .clone()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// One fixture tensor with a length-1 axis removed — the VAE fixtures are the 5-D
    /// `[B, C, T = 1, H, W]` upstream writes and the port is single-frame NCHW.
    pub fn tensor_squeezed(&self, name: &str, axis: usize) -> Tensor {
        self.tensor(name)
            .squeeze(axis)
            .unwrap_or_else(|e| panic!("fixture {name} squeeze({axis}): {e}"))
    }

    pub fn meta(&self, key: &str) -> &str {
        self.metadata
            .get(key)
            .unwrap_or_else(|| panic!("fixture metadata {key}"))
    }

    pub fn meta_opt(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
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

fn dtype_of(dtype: safetensors::Dtype, key: &str) -> DType {
    match dtype {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::U32 => DType::U32,
        safetensors::Dtype::U8 => DType::U8,
        // The fixtures' token-id tensors are int32; candle has no I32, so read the raw bytes as
        // U32 (every id is non-negative) and let the caller cast.
        safetensors::Dtype::I32 => DType::U32,
        other => panic!("fixture tensor {key} has unported dtype {other:?}"),
    }
}

/// Flatten a tensor to a host `Vec<f32>`.
pub fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
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

/// The CPU device every parity test runs on: candle's CPU backend is the only one reachable in the
/// default lane, and it is the f32 reference the tolerances below are measured against.
pub fn device() -> Device {
    Device::Cpu
}
