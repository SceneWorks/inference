//! Shared helpers for the Iris parity tests: fixture paths, tensor readback and the peak-relative
//! comparison every gate reports through.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub mod metal_guard;

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

const CPU_CHILD_ENV: &str = "IRIS_TEST_CPU_CHILD";
const CPU_CHILD_DONE: &str = "IRIS_TEST_CPU_CHILD_DONE";

/// Run `body` in a child process of this test binary whose MLX **default device** is the CPU
/// (sc-25689). [`on_cpu`] sets only mlx-rs's task-local stream, which the array operators, `index`
/// and a few other entry points do not consult — they target the default device (the GPU on a
/// Mac). A process-wide CPU default in a dedicated child keeps every op of `body` off the GPU
/// without changing the device under the binary's other tests, and [`metal_guard`] makes any Metal
/// work the child attempts anyway fatal (sc-25690). `test` is the test's full path in the
/// integration binary (`module::name`).
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

/// The declared bound of the bf16 policy's activation-checkpointed gradients (sc-25690; the Candle
/// crate's rule, spelled alike). Under bf16 a checkpointed step rounds differently from the plain
/// one (a region's cotangent crosses its boundary once per reader), so equality is not available;
/// what is asserted is that checkpointing adds no error of its own: per trained tensor its distance
/// from the f32 gradient stays within `AC_BF16_NOISE_FACTOR` of the plain bf16 gradient's own
/// distance (plus `AC_BF16_NOISE_FLOOR` of the tensor's norm), and the two bf16 gradients point the
/// same way.
pub const AC_BF16_NOISE_FACTOR: f64 = 2.0;
pub const AC_BF16_NOISE_FLOOR: f64 = 1e-3;
pub const AC_BF16_MIN_COSINE: f64 = 0.999;

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

/// `(‖a − b‖ / ‖b‖, cos(a, b))`.
pub fn rel_and_cos(a: &Array, b: &Array) -> (f64, f64) {
    let (a, b) = (host_f32(a), host_f32(b));
    let diff: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
    (l2(&diff) / l2(&b), dot / (l2(&a) * l2(&b)))
}

type Grads = std::collections::HashMap<std::rc::Rc<str>, Array>;

/// The f32 gate: every checkpointed gradient within `tol · max(1, peak)` of the plain one. Returns
/// the worst `max|Δ| / max(1, peak)`.
pub fn assert_ac_f32_equal(what: &str, tol: f32, plain: &Grads, checkpointed: &Grads) -> f32 {
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
    worst
}

/// The bf16 gate (see [`AC_BF16_NOISE_FACTOR`]); `f32_plain` is the reference both are measured
/// from. Returns `(mean e_none, mean e_full, worst e_full / e_none, worst cosine, worst max|Δ|)` —
/// the last is the largest element-wise distance between the checkpointed and the plain **bf16**
/// gradient over every trained tensor (0 = the two bf16 gradients are bit-equal), which this gate
/// measures and prints but does not bound: a caller that holds bit-equality asserts it.
pub fn assert_ac_bf16_noise_only(
    what: &str,
    f32_plain: &Grads,
    bf16_plain: &Grads,
    bf16_checkpointed: &Grads,
) -> (f64, f64, f64, f64, f32) {
    assert_eq!(bf16_plain.len(), bf16_checkpointed.len(), "{what}");
    let mut worst_abs = 0f32;
    for (k, plain) in bf16_plain {
        worst_abs = worst_abs.max(errors(&bf16_checkpointed[k], plain).0);
    }
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
        worst_abs,
    );
    eprintln!(
        "{what}: bf16 vs f32 over {n} tensors: mean e_none {:.3e}, mean e_full {:.3e}, worst e_full/e_none {:.3}, worst cos(full, none) {:.6}; bf16 checkpointed vs bf16 plain: worst max|Δ| = {:.3e}",
        out.0, out.1, out.2, out.3, out.4
    );
    out
}
