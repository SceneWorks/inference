//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (FLUX.2 klein-9b).
//!
//! The companion to Wan's `tests/perf.rs`. sc-2963 gated the compiled glue on the committed
//! `transformer_golden` fixture (`compile_parity.rs`) and measured the fusion win with the
//! weight-independent `compile_micro` micros, but never ran the `perf.rs`-style A/B on the real
//! ~18 GB klein-9b checkpoint. This file closes that gap on the same real transformer: it times
//! `Flux2Transformer::forward` warm, eager (`set_compile_glue(false)`) vs compiled
//! (`set_compile_glue(true)`), and checks the f32 whole-forward result within a peak-relative
//! end-to-end bound. The primitive contract remains the shared four-ULP f32 cap and exact bf16.
//!
//! Run it (klein-9b from the HF cache, or point at a snapshot):
//! ```text
//! cargo test --release -p mlx-gen-flux2 --test integration perf:: -- --ignored --nocapture
//! MLX_GEN_FLUX2_SNAPSHOT=<snapshot> cargo test --release -p mlx-gen-flux2 --test integration perf:: -- --ignored --nocapture
//! ```
//! Override geometry with `FLUX2_PERF_WIDTH` / `FLUX2_PERF_HEIGHT` (default 1024×1024).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_flux2::config::Flux2Config;
use mlx_gen_flux2::loader::load_transformer;
use mlx_gen_flux2::pipeline::{create_noise, prepare_grid_ids, prepare_text_ids};
use mlx_gen_flux2::transformer::set_compile_glue;
use mlx_rs::{random, Array, Dtype};

/// Peak-relative bound for the real 32-block f32 transformer forward. This is deliberately an
/// end-to-end bound, not the four-ULP primitive allowance: sub-ULP changes can amplify through the
/// residual stack. `1e-2` is FLUX.2's existing full-f32-transformer parity bar and remains an order
/// of magnitude below the O(1e-1) class of a wrong fusion. Do not loosen it without a geometry sweep.
const COMPILED_GLUE_F32_REAL_FWD_REL_TOL: f32 = 1.0e-2;

fn snapshot() -> Option<PathBuf> {
    let p = std::env::var("MLX_GEN_FLUX2_SNAPSHOT").ok()?;
    Some(PathBuf::from(p))
}

fn env_u32(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn timed(eval: &[&Array], start: Instant) -> f64 {
    mlx_rs::transforms::eval(eval.iter().copied()).unwrap();
    start.elapsed().as_secs_f64()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>()
}

fn max_abs(a: &Array) -> f32 {
    mlx_rs::ops::max(mlx_rs::ops::abs(a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b weights (MLX_GEN_FLUX2_SNAPSHOT or HF cache)"]
fn flux2_per_step_compiled_vs_eager() {
    let snap = match snapshot() {
        Some(p) => p,
        None => {
            eprintln!(
                "skip: set MLX_GEN_FLUX2_SNAPSHOT or populate the HF cache for FLUX.2-klein-9b"
            );
            return;
        }
    };
    let cfg = Flux2Config::klein_9b();
    let width = env_u32("FLUX2_PERF_WIDTH", 1024);
    let height = env_u32("FLUX2_PERF_HEIGHT", 1024);
    let (lat_h, lat_w) = ((height / 16) as usize, (width / 16) as usize);
    let txt_seq = cfg.max_sequence_length;
    println!(
        "FLUX.2 klein-9b: {height}x{width} -> seq={}, txt_seq={txt_seq}, joint_dim={}",
        lat_h * lat_w,
        cfg.joint_attention_dim
    );

    let t = load_transformer(&snap).expect("load FLUX.2 transformer");

    // Production-shaped f32 inputs (FLUX.2 runs an f32 latent stream; model.rs). Timing is
    // value-independent; the position ids + encoder shape drive the kernels.
    let key = random::key(0).unwrap();
    let hidden = create_noise(0, width, height, cfg.in_channels).expect("noise");
    let encoder = random::normal::<f32>(
        &[1, txt_seq as i32, cfg.joint_attention_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let img_ids = prepare_grid_ids(lat_h, lat_w, 0);
    let txt_ids = prepare_text_ids(txt_seq);
    mlx_rs::transforms::eval([&hidden, &encoder, &img_ids, &txt_ids]).unwrap();
    let timestep = 500.0f32;

    let run = || {
        t.forward(&hidden, &encoder, &img_ids, &txt_ids, timestep)
            .unwrap()
    };

    // --- real-weight f32 numerical contract ---
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "v shape");
    assert_eq!(eager0.dtype(), Dtype::Float32, "eager v dtype");
    assert_eq!(comp0.dtype(), eager0.dtype(), "compiled v dtype");
    let max_diff = max_abs_diff(&comp0, &eager0);
    let max_out = max_abs(&eager0);
    let rel = mlx_gen::nn::max_rel_diff(&comp0, &eager0);

    let warmup = 2usize;
    let iters = 6usize;

    set_compile_glue(false);
    let mut eager = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let v = run();
        let dt = timed(&[&v], start);
        if i >= warmup {
            eager.push(dt);
        }
    }

    set_compile_glue(true);
    let mut compiled = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let v = run();
        let dt = timed(&[&v], start);
        if i >= warmup {
            compiled.push(dt);
        }
    }
    set_compile_glue(false);

    let eag = median(eager);
    let cmp = median(compiled);
    println!(
        "[warm s/step] eager={eag:.4}  compiled-glue={cmp:.4}  speedup={:.3}×  \
         (recovers {:.1}% of step)  compiled-vs-eager: max|Δ|={max_diff:.3e}  \
         max|out|={max_out:.3e}  rel={rel:.3e}  bound={COMPILED_GLUE_F32_REAL_FWD_REL_TOL:.3e}",
        eag / cmp,
        (eag - cmp) / eag * 100.0
    );
    assert!(
        rel <= COMPILED_GLUE_F32_REAL_FWD_REL_TOL,
        "FLUX.2 compiled glue diverged from eager on real weights: rel={rel:e} \
         (max|Δ|={max_diff:e} / max|out|={max_out:e}) exceeds \
         {COMPILED_GLUE_F32_REAL_FWD_REL_TOL:e}"
    );
}
