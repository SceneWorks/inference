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

/// `Qwen/Qwen-Image-2.1-Turbo`'s saved sampling sigmas (`model_index.json` → `sample_sigmas` at
/// `UPSTREAM_TURBO_HF_REVISION`), verbatim — what [`turbo_snapshot`] writes into its fixture.
pub const TURBO_SAMPLE_SIGMAS: [f64; 8] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
];

fn rewrite_json(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value = serde_json::from_slice(
        &mlx_gen::gen_core::__model_input_choose!(
            std::fs::read(path),
            mlx_gen::gen_core::model_input_proof::read_configuration_bytes(path)
        )
        .unwrap_or_else(|e| panic!("{path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("{path:?}: {e}"));
    edit(&mut value);
    std::fs::write(path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
}

/// A **Turbo-shaped** copy of the committed miniature snapshot under `dir` (sc-25900) — every way
/// the released `Qwen/Qwen-Image-2.1-Turbo` layout differs from the base one, applied to the tiny
/// fixture at test time instead of committing a second set of binaries:
///
/// * `model_index.json` saves `sample_sigmas` ([`TURBO_SAMPLE_SIGMAS`]);
/// * `scheduler/scheduler_config.json` is the static identity shift (`use_dynamic_shifting:
///   false`, `shift: 1.0`, `shift_terminal: null`);
/// * `vae/diffusion_pytorch_model.safetensors` is **bf16** (the base repository ships f32);
/// * `text_encoder/` is one `model.safetensors` (the base repository ships four shards) — the
///   tiny snapshot already is, which this asserts rather than assumes.
pub fn turbo_snapshot(dir: &Path) -> PathBuf {
    let root = dir.join("tiny-turbo-snapshot");
    mlx_gen::quant::copy_dir(&tiny_snapshot(), &root).expect("copy the tiny snapshot");

    rewrite_json(&root.join("model_index.json"), |index| {
        index["sample_sigmas"] = serde_json::json!(TURBO_SAMPLE_SIGMAS);
    });
    rewrite_json(&root.join("scheduler/scheduler_config.json"), |config| {
        assert_eq!(config["shift"], serde_json::json!(1.0));
        config["use_dynamic_shifting"] = serde_json::json!(false);
        config["shift_terminal"] = serde_json::Value::Null;
    });

    // The VAE at bf16. Cast and evaluate before the source file is replaced: an MLX safetensors
    // load is lazy, so the bytes must be read while the f32 file still exists.
    let vae = root.join("vae/diffusion_pytorch_model.safetensors");
    let source = Weights::from_file(&vae).expect("tiny vae");
    let mut names: Vec<String> = source.keys().map(str::to_owned).collect();
    names.sort();
    let cast: Vec<(String, Array)> = names
        .into_iter()
        .map(|name| {
            let tensor = source.require(&name).unwrap();
            assert_eq!(
                tensor.dtype(),
                Dtype::Float32,
                "{name}: the tiny vae is f32"
            );
            let tensor = tensor.as_dtype(Dtype::Bfloat16).unwrap();
            (name, tensor)
        })
        .collect();
    mlx_rs::transforms::eval(cast.iter().map(|(_, tensor)| tensor)).unwrap();
    let staged = root.join("vae/bf16-staged.safetensors");
    Array::save_safetensors(
        cast.iter().map(|(name, tensor)| (name.as_str(), tensor)),
        None,
        &staged,
    )
    .unwrap();
    drop((cast, source));
    std::fs::rename(&staged, &vae).unwrap();

    let mut shards: Vec<String> = std::fs::read_dir(root.join("text_encoder"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    shards.sort();
    assert_eq!(shards, ["model.safetensors"], "single-file text encoder");
    root
}
