//! Shared helpers for the Iris parity tests: the MLX twin's committed fixtures (one numeric
//! reference for both backends), tensor readback and the peak-relative comparison every gate
//! reports through.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use gen_core_iris::IrisConfig;

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

/// The build's device — `cuda:0` under `--features cuda`, else CPU. The tests that use it compare
/// the device's own kernels (bf16 GEMM / attention on CUDA) to the committed fixtures; the CUDA
/// dispatch lane runs them by name.
pub fn device() -> Device {
    candle_gen::default_device().expect("the build's device")
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

/// The declared bound of the bf16 policy's activation-checkpointed gradients (sc-25690). Under bf16
/// a checkpointed step can round differently from the plain one at a region boundary; shared
/// conditioning cotangents are accumulated before their producer's backward, never cast separately
/// per reader. What is asserted is that checkpointing adds no error of its own: per trained tensor
/// its distance from the f32 gradient
/// stays within `AC_BF16_NOISE_FACTOR` of the plain bf16 gradient's own distance (plus
/// `AC_BF16_NOISE_FLOOR` of the tensor's norm), and the two bf16 gradients point the same way.
pub const AC_BF16_NOISE_FACTOR: f64 = 2.0;
pub const AC_BF16_NOISE_FLOOR: f64 = 1e-3;
pub const AC_BF16_MIN_COSINE: f64 = 0.999;

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

/// `(‖a − b‖ / ‖b‖, cos(a, b))`.
pub fn rel_and_cos(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let (a, b) = (host_f32(a), host_f32(b));
    let diff: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
    (l2(&diff) / l2(&b), dot / (l2(&a) * l2(&b)))
}

/// The f32 gate: every checkpointed gradient within `tol · max(1, peak)` of the plain one.
pub fn assert_ac_f32_equal<K: Ord + std::fmt::Display>(
    what: &str,
    tol: f32,
    plain: &std::collections::BTreeMap<K, Tensor>,
    checkpointed: &std::collections::BTreeMap<K, Tensor>,
) {
    assert_eq!(plain.len(), checkpointed.len(), "{what}");
    let mut worst = 0f32;
    for (k, want) in plain {
        let (max_abs, peak, _) = errors(&checkpointed[k], want);
        let rel = max_abs / peak.max(1.0);
        assert!(
            rel <= tol,
            "{what}: {k}: max|Δ| {max_abs:e} of peak {peak:e}"
        );
        worst = worst.max(rel);
    }
    eprintln!("{what}: f32 checkpointed vs plain: worst max|Δ| / max(1, peak) = {worst:.3e}");
}

/// The bf16 gate (see [`AC_BF16_NOISE_FACTOR`]); `f32_plain` is the reference both are measured
/// from. Returns `(mean e_none, mean e_full, worst e_full / e_none, worst cosine)`.
pub fn assert_ac_bf16_noise_only<K: Ord + std::fmt::Display>(
    what: &str,
    f32_plain: &std::collections::BTreeMap<K, Tensor>,
    bf16_plain: &std::collections::BTreeMap<K, Tensor>,
    bf16_checkpointed: &std::collections::BTreeMap<K, Tensor>,
) -> (f64, f64, f64, f64) {
    let (mut sum_none, mut sum_full, mut worst_ratio, mut worst_cos, mut n) =
        (0f64, 0f64, 0f64, 1f64, 0usize);
    for (k, reference) in f32_plain {
        if l2(&host_f32(reference)) == 0.0 {
            continue;
        }
        let (e_none, _) = rel_and_cos(&bf16_plain[k], reference);
        let (e_full, _) = rel_and_cos(&bf16_checkpointed[k], reference);
        let (_, cos) = rel_and_cos(&bf16_checkpointed[k], &bf16_plain[k]);
        assert!(
            e_full <= AC_BF16_NOISE_FACTOR * e_none + AC_BF16_NOISE_FLOOR,
            "{what}: {k}: checkpointing adds error under bf16 (e_full {e_full:.3e} vs e_none {e_none:.3e})"
        );
        assert!(
            cos >= AC_BF16_MIN_COSINE,
            "{what}: {k}: cos(checkpointed, plain) = {cos:.6} under bf16"
        );
        sum_none += e_none;
        sum_full += e_full;
        worst_ratio = worst_ratio.max(e_full / e_none.max(1e-12));
        worst_cos = worst_cos.min(cos);
        n += 1;
    }
    assert!(n > 0, "{what}: no non-zero gradient to compare");
    let out = (
        sum_none / n as f64,
        sum_full / n as f64,
        worst_ratio,
        worst_cos,
    );
    eprintln!(
        "{what}: bf16 vs f32 over {n} tensors: mean e_none {:.3e}, mean e_full {:.3e}, worst e_full/e_none {:.3}, worst cos(full, none) {:.6}",
        out.0, out.1, out.2, out.3
    );
    out
}
