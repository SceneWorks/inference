//! SW-6140 block micro-bench: the shared RoPE / modulate routes, before vs after, at representative
//! shapes — time (median ms/iter) and MLX peak memory per arm, plus the arms' numerical agreement.
//!
//! No weights; random activations at production geometry, one process, arms run back to back with
//! `clear_cache()` + `reset_peak_memory()` between them. Two rows, one per route:
//!
//! * **cos/sin-route crate — a FLUX.1-shaped DiT block prologue** (`[1, 4352, 3072]`, 24 × 128
//!   heads, 3-axis `[16, 56, 56]` RoPE, f32 stream): `layer_norm → modulate → q/k RoPE → SDPA`.
//!   *before* = the eager private formulation every family carried (`nn::modulate_eager`,
//!   `qkv::apply_rope_eager`); *after* = the shared compiled-glue tail (`nn::modulate` +
//!   `qkv::apply_rope` under a `CompileGlueGuard`). A multi-axis table cannot take `fast::rope`,
//!   so both arms rotate through cos/sin.
//! * **fused-route crate — PID's Gemma-2 text-encoder attention prologue** (`[1, 512, 2304]`,
//!   8 × 256 query heads, contiguous 1-D positions, bf16 stream, `RopeDtype::Stream`): q/k RoPE →
//!   SDPA. *before* = `qkv::apply_rope_1d` with the fused route off (the cos/sin route, the
//!   default); *after* = the same call under `qkv::FusedRopeGuard::set(true)`, i.e. MLX's fused
//!   `fast::rope` kernel.
//!
//! Run on the Mac (Apple Silicon only; label the chip in the evidence):
//! ```text
//! cargo run --release -p mlx-gen --example rope_modulate_block_bench
//! ROPE_BENCH_ITERS=50 cargo run --release -p mlx-gen --example rope_modulate_block_bench
//! ```

use std::time::Instant;

use mlx_gen::nn::{max_rel_diff, modulate, modulate_eager, rope_sincos_from_ids, CompileGlueGuard};
use mlx_gen::qkv::{
    apply_rope, apply_rope_1d, apply_rope_eager, FusedRopeGuard, Rope1d, RopeDtype, RopeStyle,
    RopeTables, RotationAxes,
};
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn chip() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown chip".to_string())
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Median ms/iter and peak MiB of `f` (warmup excluded), evaluating its output each iteration.
fn measure(warmup: usize, iters: usize, mut f: impl FnMut() -> Array) -> (f64, f64, Array) {
    clear_cache();
    let mut last = f();
    eval([&last]).unwrap();
    for _ in 1..warmup {
        last = f();
        eval([&last]).unwrap();
    }
    clear_cache();
    reset_peak_memory();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        last = f();
        eval([&last]).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], mib(get_peak_memory()), last)
}

fn normal(shape: &[i32], dtype: Dtype, seed: u64) -> Array {
    let key = random::key(seed).unwrap();
    let a = random::normal::<f32>(shape, None, None, Some(&key))
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
    eval([&a]).unwrap();
    a
}

fn to_heads(x: &Array, heads: i32, head_dim: i32) -> Array {
    let sh = x.shape();
    x.reshape(&[sh[0], sh[1], heads, head_dim])
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap()
}

struct Row {
    case: &'static str,
    route: &'static str,
    before: (f64, f64),
    after: (f64, f64),
    agreement: f32,
}

/// FLUX.1-shaped DiT block prologue on the cos/sin route.
fn flux_like(warmup: usize, iters: usize) -> Row {
    let (txt, gh, gw) = (256i32, 64i32, 64i32);
    let (seq, dim, heads, hd) = (txt + gh * gw, 3072i32, 24i32, 128i32);
    let x = normal(&[1, seq, dim], Dtype::Float32, 1);
    let q = to_heads(&normal(&[1, seq, dim], Dtype::Float32, 2), heads, hd);
    let k = to_heads(&normal(&[1, seq, dim], Dtype::Float32, 3), heads, hd);
    let v = to_heads(&normal(&[1, seq, dim], Dtype::Float32, 4), heads, hd);
    let scale = normal(&[1, 1, dim], Dtype::Bfloat16, 5);
    let shift = normal(&[1, 1, dim], Dtype::Bfloat16, 6);
    let mut ids = vec![0f32; (seq * 3) as usize];
    for r in 0..gh {
        for c in 0..gw {
            let row = (txt + r * gw + c) as usize;
            ids[row * 3 + 1] = r as f32;
            ids[row * 3 + 2] = c as f32;
        }
    }
    let ids = Array::from_slice(&ids, &[seq, 3]);
    let (cos, sin) = rope_sincos_from_ids(&ids, &[16, 56, 56], 10_000.0).unwrap();
    eval([&cos, &sin]).unwrap();
    let tables = RopeTables::new(&cos, &sin);
    let block = |compiled: bool| {
        let normed = layer_norm(&x, None, None, 1e-6).unwrap();
        let (m, qr, kr) = if compiled {
            let _glue = CompileGlueGuard::enable();
            let rot = |t: &Array| {
                apply_rope(
                    t,
                    tables,
                    RopeStyle::AdjacentPair,
                    RotationAxes::HeadMajor,
                    None,
                    RopeDtype::Promoted,
                )
                .unwrap()
            };
            (
                modulate(&normed, &scale, &shift, true).unwrap(),
                rot(&q),
                rot(&k),
            )
        } else {
            let rot = |t: &Array| {
                apply_rope_eager(
                    t,
                    tables,
                    RopeStyle::AdjacentPair,
                    RotationAxes::HeadMajor,
                    None,
                    RopeDtype::Promoted,
                )
                .unwrap()
            };
            (
                modulate_eager(&normed, &scale, &shift, true).unwrap(),
                rot(&q),
                rot(&k),
            )
        };
        let o =
            scaled_dot_product_attention(&qr, &kr, &v, (hd as f32).powf(-0.5), None, None).unwrap();
        // One output carrying both the modulated stream and the attention.
        mlx_rs::ops::add(
            o.transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[1, seq, dim])
                .unwrap(),
            &m,
        )
        .unwrap()
    };
    let (tb, mb, ob) = measure(warmup, iters, || block(false));
    let (ta, ma, oa) = measure(warmup, iters, || block(true));
    Row {
        case: "FLUX.1-shaped DiT block [1,4352,3072] 24x128 f32, 3-axis RoPE",
        route: "cos/sin (eager -> compiled glue)",
        before: (tb, mb),
        after: (ta, ma),
        agreement: max_rel_diff(&oa, &ob),
    }
}

/// PID Gemma-2 text-encoder attention prologue on the fused route.
fn gemma_like(warmup: usize, iters: usize) -> Row {
    let (len, heads, kv_heads, hd) = (512i32, 8i32, 4i32, 256i32);
    let base = 10_000.0f32;
    let q = to_heads(
        &normal(&[1, len, heads * hd], Dtype::Bfloat16, 11),
        heads,
        hd,
    );
    let k = to_heads(
        &normal(&[1, len, kv_heads * hd], Dtype::Bfloat16, 12),
        kv_heads,
        hd,
    );
    let v = to_heads(
        &normal(&[1, len, kv_heads * hd], Dtype::Bfloat16, 13),
        kv_heads,
        hd,
    );
    // PID's host `rope_tables`: `[L, D]`, `emb = cat(freqs, freqs)`, f64 trig rounded to f32.
    let half = (hd / 2) as usize;
    let mut c = Vec::with_capacity((len * hd) as usize);
    let mut s = Vec::with_capacity((len * hd) as usize);
    for p in 0..len as usize {
        for j in 0..hd as usize {
            let f = 1.0 / (base as f64).powf((2 * (j % half)) as f64 / hd as f64);
            c.push((p as f64 * f).cos() as f32);
            s.push((p as f64 * f).sin() as f32);
        }
    }
    let cos = Array::from_slice(&c, &[len, hd]);
    let sin = Array::from_slice(&s, &[len, hd]);
    let tables = RopeTables::new(&cos, &sin);
    let layout = Rope1d::standard(base, 0);
    let block = |fused: bool| {
        let _route = FusedRopeGuard::set(fused);
        let rot = |t: &Array| {
            apply_rope_1d(
                t,
                tables,
                layout,
                RopeStyle::RotateHalf,
                RotationAxes::HeadMajor,
                None,
                RopeDtype::Stream,
            )
            .unwrap()
        };
        let (qr, kr) = (rot(&q), rot(&k));
        let kr = mlx_gen::qkv::repeat_kv(&kr, heads / kv_heads, RotationAxes::HeadMajor).unwrap();
        let vr = mlx_gen::qkv::repeat_kv(&v, heads / kv_heads, RotationAxes::HeadMajor).unwrap();
        scaled_dot_product_attention(&qr, &kr, &vr, (hd as f32).powf(-0.5), None, None).unwrap()
    };
    let (tb, mb, ob) = measure(warmup, iters, || block(false));
    let (ta, ma, oa) = measure(warmup, iters, || block(true));
    Row {
        case: "PID Gemma-2 TE attention [1,512,8x256] bf16, 1-D RoPE",
        route: "cos/sin -> fused fast::rope",
        before: (tb, mb),
        after: (ta, ma),
        agreement: max_rel_diff(&oa, &ob),
    }
}

fn main() {
    let warmup = env_usize("ROPE_BENCH_WARMUP", 3);
    let iters = env_usize("ROPE_BENCH_ITERS", 20);
    let rows = [flux_like(warmup, iters), gemma_like(warmup, iters)];
    println!(
        "SW-6140 RoPE/modulate block micro-bench — {} (median of {iters}, {warmup} warmup)",
        chip()
    );
    println!(
        "| case | route | before ms | after ms | before peak MiB | after peak MiB | max rel diff |"
    );
    println!("|---|---|---|---|---|---|---|");
    for r in rows {
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.1} | {:.1} | {:e} |",
            r.case, r.route, r.before.0, r.after.0, r.before.1, r.after.1, r.agreement
        );
    }
}
