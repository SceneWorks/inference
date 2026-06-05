//! sc-2963 compile-mechanism microbenchmark (rollout of the Wan sc-2957 template) — does
//! `mx.compile` fuse the FLUX.2 MMDiT's fusable elementwise *glue* into faster kernels?
//!
//! No weights — it times the fusable chains in isolation at klein-9b production shapes (inner dim
//! 4096 = 32×128, SwiGLU hidden 12288, 8 double + 24 single blocks), eager vs `compile`d, so the
//! kernel-fusion win can be measured + attributed per chain BEFORE the real-weight A/B. The chains
//! (all f32 — FLUX.2 runs f32 activations):
//!   * **swiglu** — `silu(a)·b` on `[B, S, ffn]` (the split is eager; the arithmetic fuses).
//!   * **modulate** — adaLN affine `norm·(1+scale)+shift` on `[B, S, dim]`.
//!   * **gated** — gated residual `x + gate·y` on `[B, S, dim]`.
//!   * **rope_rotate** — the complex rotation on `[B, H, S, head_dim/2]` (applied to q and k).
//!
//! Three variants per chain (eager / oneshot `compile(f,true)(x)` / held `f.compile(true)`) — see
//! `mlx-gen-wan/tests/compile_micro.rs` for the erase-per-call trap discussion.
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-flux2 --test compile_micro -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_gen::array::scalar;
use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, sigmoid, subtract};
use mlx_rs::transforms::compile::{compile, CallMut, Compile};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench(warmup: usize, iters: usize, mut f: impl FnMut() -> Array) -> f64 {
    let mut times = Vec::new();
    for i in 0..(warmup + iters) {
        let t0 = Instant::now();
        let out = f();
        eval([&out]).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= warmup {
            times.push(dt);
        }
    }
    median(times)
}

fn swiglu_body((a, b): (&Array, &Array)) -> Result<Array, Exception> {
    multiply(&multiply(a, &sigmoid(a)?)?, b)
}

fn modulate_body((n, s, sh): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(&multiply(n, &add(s, scalar(1.0))?)?, sh)
}

fn gated_body((x, g, y): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(x, &multiply(g, y)?)
}

fn rope_body(inp: &[Array]) -> Result<Vec<Array>, Exception> {
    let (r, i, c, s) = (&inp[0], &inp[1], &inp[2], &inp[3]);
    let out0 = subtract(&multiply(r, c)?, &multiply(i, s)?)?;
    let out1 = add(&multiply(i, c)?, &multiply(r, s)?)?;
    Ok(vec![out0, out1])
}

fn normal(shape: &[i32]) -> Array {
    let key = random::key(0).unwrap();
    let x = random::normal::<f32>(shape, None, None, Some(&key)).unwrap();
    eval([&x]).unwrap();
    x
}

fn max_abs_diff(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>() as f64
}

#[test]
#[ignore = "perf microbenchmark (no weights) — run with --ignored --nocapture"]
fn compile_glue_micro() {
    let b = env_usize("FLUX2_PERF_BATCH", 1) as i32;
    let s = env_usize("FLUX2_PERF_SEQ", 4096) as i32; // 1024² image tokens
    let dim = env_usize("FLUX2_DIM", 4096) as i32;
    let ffn = env_usize("FLUX2_FFN", 12288) as i32; // mlp_ratio 3 × 4096
    let heads = env_usize("FLUX2_HEADS", 32) as i32;
    let half = env_usize("FLUX2_HALF", 64) as i32; // head_dim/2
    let warmup = 3usize;
    let iters = 12usize;
    println!("shapes: B={b} S={s} dim={dim} ffn={ffn} heads={heads} half={half}  (warmup={warmup} iters={iters})");

    // ---- swiglu silu(a)·b on [B, S, ffn] f32 (~40 / step) ----
    {
        let a = normal(&[b, s, ffn]);
        let bb = normal(&[b, s, ffn]);
        let eager = bench(warmup, iters, || swiglu_body((&a, &bb)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(swiglu_body, true)((&a, &bb)).unwrap()
        });
        let mut held = swiglu_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&a, &bb)).unwrap());
        let diff = max_abs_diff(
            &swiglu_body((&a, &bb)).unwrap(),
            &compile(swiglu_body, true)((&a, &bb)).unwrap(),
        );
        println!(
            "[swiglu f32 {b}x{s}x{ffn}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×40={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 40.0
        );
    }

    // ---- modulate (adaLN affine) on [B, S, dim] f32 (~56 / step) ----
    {
        let m = normal(&[b, s, dim]);
        let sc = normal(&[b, 1, dim]);
        let sh = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || modulate_body((&m, &sc, &sh)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(modulate_body, true)((&m, &sc, &sh)).unwrap()
        });
        let mut held = modulate_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&m, &sc, &sh)).unwrap());
        let diff = max_abs_diff(
            &modulate_body((&m, &sc, &sh)).unwrap(),
            &compile(modulate_body, true)((&m, &sc, &sh)).unwrap(),
        );
        println!(
            "[modulate f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×56={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 56.0
        );
    }

    // ---- gated residual on [B, S, dim] f32 (~56 / step) ----
    {
        let x = normal(&[b, s, dim]);
        let y = normal(&[b, s, dim]);
        let g = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || gated_body((&x, &g, &y)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(gated_body, true)((&x, &g, &y)).unwrap()
        });
        let mut held = gated_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&x, &g, &y)).unwrap());
        let diff = max_abs_diff(
            &gated_body((&x, &g, &y)).unwrap(),
            &compile(gated_body, true)((&x, &g, &y)).unwrap(),
        );
        println!(
            "[gated f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×56={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 56.0
        );
    }

    // ---- rope_rotate on [B, H, S, half] f32 (q and k, ~64 / step) ----
    {
        let r = normal(&[b, heads, s, half]);
        let im = normal(&[b, heads, s, half]);
        let c = normal(&[1, 1, s, half]);
        let sn = normal(&[1, 1, s, half]);
        let args = [r.clone(), im.clone(), c.clone(), sn.clone()];
        let eager = bench(warmup, iters, || rope_body(&args).unwrap().pop().unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(rope_body, true)(&args).unwrap().pop().unwrap()
        });
        let mut held = rope_body.compile(true);
        let heldt = bench(warmup, iters, || {
            held.call_mut(&args).unwrap().pop().unwrap()
        });
        let diff = max_abs_diff(
            &rope_body(&args).unwrap()[0],
            &compile(rope_body, true)(&args).unwrap()[0],
        );
        println!(
            "[rope_rotate f32 {b}x{heads}x{s}x{half}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×64={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 64.0
        );
    }
}
