//! sc-2963 compile-mechanism microbenchmark (rollout of the Wan sc-2957 template) — does
//! `mx.compile` fuse SDXL's remaining eager glue (the **SiLU** activation) into a faster kernel?
//!
//! No weights — it times `silu(x) = x·sigmoid(x)` eager vs `compile`d at representative **fp16** UNet
//! ResNet feature-map shapes (CFG batch 2). The GEGLU/erf-GELU activations are already `mx.compile`'d
//! in core `nn` (sc-2721); SiLU is the only remaining fusable chain. SiLU is just 2 ops, so the win is
//! a single saved kernel launch + one fewer 2-pass tensor round-trip — modest, but it recurs in every
//! ResNet block (~2 SiLU/block × the down/mid/up stacks). `max|Δ|=0` (fusing fp16 SiLU is bit-exact).
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-sdxl --test compile_micro -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::ops::{multiply, sigmoid};
use mlx_rs::transforms::compile::{compile, CallMut, Compile};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

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

fn silu_body(x: &Array) -> Result<Array, Exception> {
    multiply(x, &sigmoid(x)?)
}

fn normal(shape: &[i32]) -> Array {
    let key = random::key(0).unwrap();
    let x = random::normal::<f32>(shape, None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    eval([&x]).unwrap();
    x
}

fn max_abs_diff(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>() as f64
}

#[test]
#[ignore = "perf microbenchmark (no weights) — run with --ignored --nocapture"]
fn compile_silu_micro() {
    let warmup = 3usize;
    let iters = 12usize;
    // (B=2 CFG) × representative SDXL UNet ResNet feature maps (NHWC): high-res/low-channel down to
    // low-res/high-channel. ~50 SiLU calls per CFG forward across the down/mid/up ResNet stacks.
    let shapes: [[i32; 4]; 3] = [
        [2, 128, 128, 320], // top level, 1024² input
        [2, 64, 64, 640],
        [2, 32, 32, 1280],
    ];
    for sh in shapes {
        let x = normal(&sh);
        let eager = bench(warmup, iters, || silu_body(&x).unwrap());
        let oneshot = bench(warmup, iters, || compile(silu_body, true)(&x).unwrap());
        let mut held = silu_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut(&x).unwrap());
        let diff = max_abs_diff(
            &silu_body(&x).unwrap(),
            &compile(silu_body, true)(&x).unwrap(),
        );
        println!(
            "[silu fp16 {}x{}x{}x{}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×50={:.1}ms | max|Δ|={diff:.2e}",
            sh[0],
            sh[1],
            sh[2],
            sh[3],
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 50.0
        );
    }
}
