//! Shared helpers for the Iris parity tests: the MLX twin's committed fixtures (one numeric
//! reference for both backends), tensor readback and the peak-relative comparison every gate
//! reports through.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::iris::IrisConfig;

/// The MLX twin's `tests/fixtures/` — produced once by `crates/media/mlx-gen/tools/dump_iris_*.py`
/// from the pinned upstream and read by both backends.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mlx-gen/mlx-gen-iris/tests/fixtures")
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

/// The parity lane's device: Candle CPU (true f32, like the oracle's FP32 CPU path).
pub fn cpu() -> Device {
    Device::Cpu
}

/// A committed fixture: its tensors (on CPU) and its header metadata.
pub struct Fixture {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}

impl Fixture {
    pub fn require(&self, key: &str) -> &Tensor {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("fixture tensor {key} missing"))
    }

    pub fn meta(&self, key: &str) -> &str {
        self.metadata
            .get(key)
            .unwrap_or_else(|| panic!("fixture metadata {key} missing"))
    }
}

pub fn fixture(name: &str) -> Fixture {
    fixture_at(&fixtures().join(name))
}

pub fn fixture_at(path: &Path) -> Fixture {
    let bytes = std::fs::read(path).expect("fixture readable");
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).expect("fixture header");
    let metadata = header.metadata().clone().unwrap_or_default();
    let tensors = candle_gen::candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)
        .expect("fixture tensors");
    Fixture { tensors, metadata }
}

pub fn host_f32(a: &Tensor) -> Vec<f32> {
    a.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

pub fn host_i32(a: &Tensor) -> Vec<i32> {
    a.flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect()
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
