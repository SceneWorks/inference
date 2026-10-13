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
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures")
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

/// `Qwen/Qwen-Image-2.1-Turbo`'s `model_index.json` `sample_sigmas`, verbatim (terminal 0 excluded).
pub const TURBO_SAMPLE_SIGMAS: [f64; 8] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
];

/// A **Turbo-shaped** copy of the committed miniature snapshot, built in a temp dir (sc-25901) —
/// the four things `Qwen/Qwen-Image-2.1-Turbo` changes against the base repository, applied to
/// the base fixture so nothing binary is committed twice:
///
/// * `model_index.json` gains `sample_sigmas` (`sigmas`, so a test can prove the schedule is READ
///   rather than assumed by saving a different list);
/// * `scheduler/scheduler_config.json` becomes the static scheduler (`use_dynamic_shifting:
///   false`, `shift_terminal: null`; `shift: 1.0` already);
/// * `vae/diffusion_pytorch_model.safetensors` is rewritten **bf16** (the base ships it f32);
/// * `text_encoder/` holds one `model.safetensors` — already true of the miniature, asserted here
///   so the fixture cannot silently stop being single-file.
///
/// The transformer and the tokenizer are byte copies. Dropping the returned guard deletes the
/// tree.
pub fn turbo_snapshot_with(sigmas: &[f64]) -> tempfile::TempDir {
    let temp = copy_of_tiny_snapshot();
    let root = temp.path();

    let index_path = root.join("model_index.json");
    let mut index = read_json(&index_path);
    index["sample_sigmas"] = serde_json::json!(sigmas);
    write_json(&index_path, &index);

    let scheduler_path = root.join("scheduler").join("scheduler_config.json");
    let mut scheduler = read_json(&scheduler_path);
    scheduler["use_dynamic_shifting"] = serde_json::json!(false);
    scheduler["shift_terminal"] = serde_json::Value::Null;
    assert_eq!(scheduler["shift"], serde_json::json!(1.0));
    write_json(&scheduler_path, &scheduler);

    let vae_path = root.join("vae").join("diffusion_pytorch_model.safetensors");
    let bf16: HashMap<String, Tensor> = candle_core::safetensors::load(&vae_path, &Device::Cpu)
        .expect("the miniature VAE reads")
        .into_iter()
        .map(|(name, tensor)| {
            let cast = tensor
                .to_dtype(DType::BF16)
                .unwrap_or_else(|e| panic!("{name} casts to bf16: {e}"));
            (name, cast)
        })
        .collect();
    candle_core::safetensors::save(&bf16, &vae_path).expect("the bf16 VAE writes");

    assert_eq!(
        safetensors_files(&root.join("text_encoder")),
        ["model.safetensors"],
        "the Turbo tower is one file"
    );
    temp
}

/// [`turbo_snapshot_with`] the released eight sigmas.
pub fn turbo_snapshot() -> tempfile::TempDir {
    turbo_snapshot_with(&TURBO_SAMPLE_SIGMAS)
}

/// A byte copy of the committed (base) miniature snapshot in a temp dir, for tests that edit one.
pub fn copy_of_tiny_snapshot() -> tempfile::TempDir {
    let temp = tempfile::tempdir().expect("a temp dir for a snapshot copy");
    copy_tree(&tiny_snapshot(), temp.path());
    temp
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap_or_else(|e| panic!("create {}: {e}", to.display()));
    for entry in std::fs::read_dir(from).unwrap_or_else(|e| panic!("list {}: {e}", from.display()))
    {
        let entry = entry.expect("a directory entry");
        let (source, target) = (entry.path(), to.join(entry.file_name()));
        if source.is_dir() {
            copy_tree(&source, &target);
        } else {
            std::fs::copy(&source, &target)
                .unwrap_or_else(|e| panic!("copy {}: {e}", source.display()));
        }
    }
}

pub fn read_json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn write_json(path: &Path, value: &serde_json::Value) {
    let text = serde_json::to_string_pretty(value).expect("JSON serialises");
    std::fs::write(path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// The `.safetensors` file names directly under `dir`, sorted.
pub fn safetensors_files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("list {}: {e}", dir.display()))
        .map(|entry| entry.expect("a directory entry").file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    names.sort();
    names
}

/// The distinct on-disk dtypes of one `.safetensors` file, read from its header — what the file
/// STORES, independent of whatever dtype a loader then casts to.
pub fn stored_dtypes(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let parsed = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("{} parses: {e}", path.display()));
    let mut dtypes: Vec<String> = parsed
        .tensors()
        .into_iter()
        .map(|(_, view)| format!("{:?}", view.dtype()))
        .collect();
    dtypes.sort();
    dtypes.dedup();
    dtypes
}
