//! Shared helpers for the Supra2 Candle integration tests.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::supra2::Supra2Config;

/// Every device this build can run: the CPU (the exact-f32 structural gate) always, plus the
/// process-default accelerator when the build selected one (`cuda:0` under `--features cuda`).
pub fn devices() -> Vec<(&'static str, Device)> {
    let mut out = vec![("cpu", Device::Cpu)];
    let default = candle_gen::default_device().expect("default device");
    if default.is_cuda() {
        out.push(("cuda", default));
    } else if default.is_metal() {
        out.push(("metal", default));
    }
    out
}

/// Declared `(max_rel, mean_rel)` bound, relative to the reference abs-max, for a device: the CPU
/// bound is the exact-f32 structural gate; the accelerator bound covers reduction-order differences
/// (cuBLAS f32 GEMM tiling and the device softmax/mean reductions; candle leaves reduced-precision
/// f32 GEMM — TF32 — disabled, so no reduced-mantissa floor is included).
pub fn tol(device: &Device, cpu: (f32, f32), accel: (f32, f32)) -> (f32, f32) {
    if device.is_cpu() {
        cpu
    } else {
        accel
    }
}

/// Row-major f32 host copy.
pub fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// `(max_abs, mean_abs, ref_absmax)` between two equal-length f32 slices.
pub fn diff_stats(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut max_abs = 0f32;
    let mut sum = 0f64;
    let mut ref_max = 0f32;
    for (&g, &w) in got.iter().zip(want) {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        sum += d as f64;
        ref_max = ref_max.max(w.abs());
    }
    (max_abs, (sum / got.len() as f64) as f32, ref_max)
}

/// Assert a stage against its reference under a declared relative bound, printing the measurement.
pub fn report(stage: &str, got: &[f32], want: &[f32], (max_rel, mean_rel): (f32, f32)) {
    let (max_abs, mean_abs, ref_max) = diff_stats(got, want);
    eprintln!(
        "PARITY {stage}: max_abs {max_abs:.3e} mean_abs {mean_abs:.3e} ref_absmax {ref_max:.3e} \
         (max_rel {:.3e} <= {max_rel:.0e}, mean_rel {:.3e} <= {mean_rel:.0e})",
        max_abs / ref_max,
        mean_abs / ref_max
    );
    assert!(
        max_abs <= max_rel * ref_max,
        "{stage}: max_abs {max_abs} ref {ref_max}"
    );
    assert!(
        mean_abs <= mean_rel * ref_max,
        "{stage}: mean_abs {mean_abs} ref {ref_max}"
    );
}

/// The MLX twin's committed weights-free fixture (`tools/dump_supra2_reference.py tiny`): a tiny
/// random-weight upstream `SupraDiT` plus its torch outputs. Read in place — one fixture, two
/// backends.
pub fn tiny_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/mlx-gen-supra2/tests/fixtures/tiny_dit_parity.safetensors")
}

/// The tiny fixture's geometry.
pub fn tiny() -> Supra2Config {
    Supra2Config {
        latent_channels: 4,
        latent_size: 16,
        patch: 2,
        d_model: 32,
        depth: 2,
        num_heads: 2,
        mlp_hidden: 128,
        ctx_dim: 24,
        ctx_len: 8,
        freq_dim: 256,
    }
}

/// `(weights, inputs/outputs)` of the tiny fixture on `device`: weights with the `w.` prefix
/// stripped, everything else keyed as stored (`in.*`, `out.*`).
pub fn tiny_fixture(device: &Device) -> (HashMap<String, Tensor>, HashMap<String, Tensor>) {
    let all = candle_gen::candle_core::safetensors::load(tiny_fixture_path(), device).unwrap();
    let mut weights = HashMap::new();
    let mut rest = HashMap::new();
    for (key, value) in all {
        match key.strip_prefix("w.") {
            Some(name) => {
                weights.insert(name.to_string(), value);
            }
            None => {
                rest.insert(key, value);
            }
        }
    }
    (weights, rest)
}
