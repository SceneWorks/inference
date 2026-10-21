//! Shared helpers for the FLUX 3 parity tests: fixture paths, tensor readback and the
//! peak-relative comparison every gate reports through (the `mlx-gen-iris` conventions).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub mod metal_guard;

use mlx_gen::gen_core::flux3::{resolve_checkpoint, Flux3Config};
use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

/// `tests/fixtures/` of this crate.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The miniature package (`config.json` + `model.safetensors` + `ae.safetensors` +
/// `text_encoder/`), laid out exactly like the pinned Hugging Face package.
pub fn tiny_package() -> PathBuf {
    fixtures().join("tiny-snapshot")
}

/// The miniature Qwen3-VL text-encoder snapshot inside the package.
pub fn tiny_text_encoder() -> PathBuf {
    tiny_package().join("text_encoder")
}

/// The miniature autoencoder inside the package.
pub fn tiny_autoencoder() -> PathBuf {
    tiny_package().join("ae.safetensors")
}

/// The miniature distilled package's config, resolved from its weights' headers exactly as the
/// loaders resolve a real package (the `ref_time_in` pair decides `flow_map`).
pub fn tiny_config() -> Flux3Config {
    resolve_checkpoint(&tiny_package()).expect("the tiny distilled package resolves")
}

/// The miniature base (CFG) package: `config.json` + `model.safetensors` without `ref_time_in`.
pub fn tiny_base_package() -> PathBuf {
    fixtures().join("tiny-base-snapshot")
}

/// The miniature base package's config, resolved from its headers — never a distilled config
/// with `flow_map` hand-flipped.
pub fn tiny_base_config() -> Flux3Config {
    resolve_checkpoint(&tiny_base_package()).expect("the tiny base package resolves")
}

/// Run `f` with the MLX **CPU** stream as this thread's default. MLX's Metal GEMM accumulates f32
/// in a reduced-precision path, so an FP32 parity gate against upstream's FP32 CPU oracle runs on
/// the CPU stream, where f32 is f32.
pub fn on_cpu<T>(f: impl FnOnce() -> T) -> T {
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), f)
}

const CPU_CHILD_ENV: &str = "FLUX3_TEST_CPU_CHILD";
const CPU_CHILD_DONE: &str = "FLUX3_TEST_CPU_CHILD_DONE";

/// Run `body` in a child process of this test binary whose MLX **default device** is the CPU.
/// [`on_cpu`] sets only mlx-rs's task-local stream, which the array operators, `index` and a few
/// other entry points do not consult — they target the default device (the GPU on a Mac). A
/// process-wide CPU default in a dedicated child keeps every op of `body` off the GPU without
/// changing the device under the binary's other tests, and [`metal_guard`] makes any Metal work
/// the child attempts anyway fatal. `test` is the test's full path in the integration binary
/// (`module::name`).
pub fn cpu_process(test: &str, body: fn()) {
    if std::env::var(CPU_CHILD_ENV).as_deref() == Ok(test) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        metal_guard::forbid_metal();
        on_cpu(body);
        std::fs::write(std::env::var(CPU_CHILD_DONE).unwrap(), b"ok").unwrap();
        return;
    }
    let done = tempfile::tempdir().unwrap();
    let marker = done.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            test,
            "--nocapture",
            "--test-threads",
            "1",
            "--include-ignored",
        ])
        .env(CPU_CHILD_ENV, test)
        .env(CPU_CHILD_DONE, &marker)
        .status()
        .expect("spawn the CPU child");
    assert!(status.success(), "{test} failed in its CPU child");
    assert!(marker.is_file(), "{test}: the CPU child ran no body");
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

/// `[C, H, W]` or `[1, C, H, W]` (the oracle's layout) → `[1, H, W, C]` f32 (this crate's).
pub fn nchw_to_nhwc(a: &Array) -> Array {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let a = if a.ndim() == 3 {
        a.expand_dims(0).unwrap()
    } else {
        a
    };
    a.transpose_axes(&[0, 2, 3, 1]).unwrap()
}

/// `[1, H, W, C]` → `[1, C, H, W]` f32.
pub fn nhwc_to_nchw(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap()
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
