//! Shared helpers for the Qwen-Image 2.1 candle parity tests: fixture paths, tensor readback, and
//! the peak-relative comparison every component gate reports through. The candle twin of
//! `mlx-gen-qwen-image-2-1`'s `tests/common/mod.rs`.
//!
//! The `.safetensors` fixtures and the miniature snapshot are **shared with the MLX crate** and
//! reached across the backend boundary by relative path (the idiom `candle-gen-ltx`'s
//! `connector_parity.rs` uses), so there is exactly one committed oracle per component. They are
//! read with the `safetensors` crate rather than candle's loader because the case geometry
//! (`<case>/height`, `/width`, `/text_len`, `/timestep`) lives in the file's `__metadata__` map,
//! which candle's loader does not surface.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

/// `tests/fixtures/` of the **MLX** crate — the single home of the committed oracles.
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

/// One committed `.safetensors` fixture: every tensor on the CPU in f32, plus the `__metadata__`
/// map the dump script wrote the case geometry into.
pub struct Fixture {
    tensors: HashMap<String, Tensor>,
    metadata: HashMap<String, String>,
}

impl Fixture {
    pub fn open(name: &str) -> Self {
        let path = fixtures().join(name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("committed fixture {} loads: {e}", path.display()));
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
            .unwrap_or_else(|e| panic!("{}: header: {e}", path.display()));
        let metadata = header.metadata().clone().unwrap_or_default();
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .unwrap_or_else(|e| panic!("{}: deserialize: {e}", path.display()));
        let mut tensors = HashMap::new();
        for (key, view) in st.tensors() {
            assert_eq!(
                view.dtype(),
                safetensors::Dtype::F32,
                "{}: {key} is not f32",
                path.display()
            );
            let values: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let shape: Vec<usize> = view.shape().to_vec();
            let tensor = Tensor::from_vec(values, shape, &Device::Cpu)
                .unwrap_or_else(|e| panic!("{}: {key}: {e}", path.display()));
            tensors.insert(key, tensor);
        }
        Self { tensors, metadata }
    }

    /// The fixture tensor at `name`, on the CPU in f32.
    pub fn tensor(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("fixture lacks tensor {name}"))
    }

    /// A raw `__metadata__` entry.
    pub fn meta(&self, key: &str) -> &str {
        self.metadata
            .get(key)
            .unwrap_or_else(|| panic!("fixture metadata {key}"))
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

/// Every element of `t`, flattened, as f32 on the host.
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
