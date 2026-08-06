//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (Qwen-Image, T2I + Edit).
//!
//! The companion to Wan's `tests/perf.rs`. sc-2963 proved the compiled glue (modulate / gated /
//! tanh-GELU FFN / RoPE) **bit-exact** via in-crate `#[cfg(test)] sc2963` helper gates and measured
//! the fusion win with `compile_micro`, but never ran the `perf.rs`-style A/B on the real ~40 GB
//! 60-layer transformer. This file closes that gap on the SAME real transformer for **both** paths:
//! it times `QwenTransformer::forward` warm, eager (`set_compile_glue(false)`) vs compiled
//! (`set_compile_glue(true)`), and gates the compiled forward against the eager one on the real
//! weights. The Edit path additionally exercises the `zero_cond_t` dual-latent `modulate_index`
//! route (cond_grids non-empty).
//!
//! **That comparison used to assert `max|Δ| == 0`, which had never held on real weights** — sc-17284
//! ran this file for the first time and measured 1.897e-3 / 1.475e-3 / 9.739e-3 at 256² / 512² /
//! 1024² T2I and 1.345e-3 on Edit. sc-17513 swept it and replaced the gate with a peak-**relative**
//! bound: see [`COMPILED_GLUE_WHOLE_FWD_REL_TOL`] for the number, the sweep behind it, and — the
//! part that decides it — why the residual is this stack's accumulation floor and not a fusion
//! defect. Do not loosen it further without repeating that sweep.
//!
//! Qwen runs mixed precision: f32 latents, bf16 text embeds (`model.rs`). Timing is value-independent.
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-qwen-image --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `QWEN_PERF_SIZE` (square px, default 1024) / `QWEN_PERF_TXT` (text seq, 128).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_qwen_image::loader::{load_transformer, load_transformer_edit};
use mlx_gen_qwen_image::transformer::{set_compile_glue, QwenTransformer};
use mlx_rs::{random, Array, Dtype};

/// sc-17284: this used to return `Option`, and both tests below turned `None` into an
/// `eprintln!("skip: …")` + `return`. That is a FALSE GREEN, not a skip: libtest reports
/// `test result: ok. 1 passed` in 0.00s, so neither a `--exact` selection nor a run-count
/// assertion — the two things that catch a renamed or filtered-out test — can see that the gate
/// never ran. If you want to skip it, do not run it.
///
/// The HF-cache fallback went with it. It resolved a repository ROOT, but `load_transformer` needs
/// a TIER directory (`…/bf16`, `…/q8`) of the SceneWorks re-hosts these engines actually load, so
/// the fallback could never have succeeded — and deriving a cache location is what epic 13657
/// forbids anyway.
fn snapshot(env: &str) -> PathBuf {
    PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the required snapshot TIER dir; inference never self-fetches or derives a cache location (epic 13657)")))
}

fn env_i32(var: &str, default: i32) -> i32 {
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
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(&a, &b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>()
}

/// Peak magnitude of `a`, the denominator of the relative bound (printed so any future run log
/// carries the ratio, not just the numerator — the gap sc-17513 had to close before it could set a
/// bound at all).
fn max_abs(a: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::ops::max(mlx_rs::ops::abs(&a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

/// Peak-**relative** bound on the compiled-glue-vs-eager divergence of the **whole 60-layer
/// forward**, applied via [`mlx_gen::nn::max_rel_diff`] (`max|Δ| / max|eager|`).
///
/// # This is not a ULP tolerance, and must not be read as one
///
/// `1.6e-2` is ~1.3e5 f32 ULP. The sibling whole-forward gates are 16 ULP
/// (`mlx-gen-z-image/tests/compile_parity.rs`) and the op-level one is 4
/// ([`mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL`]). The four orders of magnitude between them are not
/// slack: they are this stack's measured **amplification** of a sub-ULP input difference, and no
/// gate on a 60-layer output can be tighter than that floor. What follows is the measurement.
///
/// # The sweep — sc-17513, 2026-08-06, nax-macos (M5 Max, 128 GiB), MLX pin 932beb4e
///
/// T2I on `SceneWorks/qwen-image-mlx@8080a417` **bf16**, 13 shapes, `txt_seq=128`, all at the full
/// 60 layers. `max|eager|` sat in 3.74…4.99 across the whole range, so `rel` tracks `max|Δ|`:
///
/// | px | img_seq | `max|Δ|` | `max|out|` | rel |
/// |---|---|---|---|---|
/// | 128 | 64 | 1.086e-3 | 3.891 | 2.79e-4 |
/// | 192 | 144 | 1.049e-3 | 3.745 | 2.80e-4 |
/// | 256 | 256 | 1.897e-3 | 4.151 | 4.57e-4 |
/// | 320 | 400 | 8.721e-4 | 4.508 | 1.93e-4 |
/// | 384 | 576 | 1.011e-2 | 4.426 | **2.285e-3** ← worst measured |
/// | 448 | 784 | 1.659e-3 | 4.190 | 3.96e-4 |
/// | 512 | 1024 | 1.475e-3 | 4.452 | 3.31e-4 |
/// | 640 | 1600 | 2.665e-3 | 4.843 | 5.50e-4 |
/// | 768 | 2304 | 1.683e-3 | 4.714 | 3.57e-4 |
/// | 896 | 3136 | 9.621e-4 | 4.764 | 2.02e-4 |
/// | 1024 | 4096 | 9.739e-3 | 4.707 | 2.069e-3 |
/// | 1152 | 5184 | 2.454e-3 | 4.990 | 4.92e-4 |
/// | 1280 | 6400 | 5.920e-3 | 4.633 | 1.278e-3 |
///
/// Edit on `qwen-image-edit-2511-mlx@0dfbf3a0` **q8** (the tier the lane runs; that repo's `bf16/`
/// is metadata-only), 5 shapes, `max|eager|` 5.91…7.14: 4.79e-5 (256²), **5.70e-4** (384², worst),
/// 1.17e-4 (512²), 2.70e-4 (768²), 1.88e-4 (1024²).
///
/// # Why this is the accumulation floor and not a fusion defect — the two decisive controls
///
/// 1. **Depth.** Same weights, same 512² input, `num_layers` truncated. At **1 layer the compiled
///    forward is bit-identical to eager on the real weights — `max|Δ|` is exactly `0.0`**; at 2
///    layers it is 3.815e-6 (rel 3.38e-8 = **0.3 ULP**), i.e. the ≤1-ULP-per-fused-op f32 rounding
///    sc-12747 documented for the MLX 0.32.0 pin. It then climbs with depth: 7.8e-5 (4), 5.9e-5
///    (8), 1.1e-4 (15), 2.6e-4 (30), 2.0e-3 (45), 3.3e-4 (60). A wrong fusion — a swapped operand,
///    a mis-broadcast scale, a dropped `+1` — is a per-op O(1e-1) error and would be loudest at
///    layer 1. This is zero at layer 1.
/// 2. **Conditioning.** Perturbing the eager forward's *own input* by one f32 ULP relative, with
///    compiled glue never switched on, moves its output by the SAME order as the whole
///    compiled-vs-eager gap, at every shape tried: 8.90e-4 / 4.45e-4 (256², uniform-gain and
///    random-noise perturbations) vs 4.57e-4 measured; 3.67e-4 / 3.61e-4 (512²) vs 3.31e-4;
///    2.26e-3 / 2.53e-3 (1024²) vs 2.069e-3. The stack's measured amplification is 1.3e3…2.2e4×,
///    and it is that amplification — not the fusion — that is non-monotonic in sequence length,
///    which is why `rel` jumps an order of magnitude between neighbouring shapes (384² and 1024²
///    are the peaks in both series).
///
/// Both paths are also **exactly reproducible**: eager-vs-eager and compiled-vs-compiled repeats
/// are `0.0` at every shape and every depth above, and the 1024² T2I (9.739e-3) and 1024² Edit
/// (1.345e-3) figures reproduce sc-17284's two-days-earlier run to every digit it printed. So the
/// headroom below is real headroom, not a noise margin.
///
/// # The number
///
/// `1.6e-2` is **7.0× the worst measured** (2.285e-3, T2I 384²) over both arms and 18 shapes. The
/// headroom is that size, rather than the ~2× a single-host measurement would justify, because
/// `rw-mage` is carried by TWO Macs and this sweep ran on one of them; sc-12747 measured the same
/// fused f32 kernels rounding differently between the NAX and non-NAX metallib paths, so the other
/// box may sit a per-op ULP away and be amplified to a different peak.
///
/// **What this gate can and cannot catch.** It cannot see anything below ~2e-3 relative — the
/// amplification floor above bounds it, and that is a property of the model, not of the bound.
/// What it does catch is the regression class that matters: a genuinely wrong fusion is O(1e-1)
/// relative or worse, 6…60× above this bound, and trips it. The tight sensitivity lives where it
/// can: the in-crate `#[cfg(test)] mod sc2963` gates in `transformer/{block,attention,
/// feed_forward}.rs` still hold each glue primitive to `0.0` (bf16) or 4 ULP (f32) in the default
/// weight-free `cargo test`, and the depth-1 result above is the real-weight corroboration that
/// the extrapolation from those primitives to a single block is exact.
const COMPILED_GLUE_WHOLE_FWD_REL_TOL: f32 = 1.6e-2;

/// Run the warm A/B + the real-weight compiled-vs-eager parity check for one forward closure.
/// `label` names the path.
fn ab<F: Fn() -> Array>(label: &str, run: F) {
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "{label} v shape");
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
        "[{label} warm s/step] eager={eag:.4}  compiled-glue={cmp:.4}  speedup={:.3}×  \
         (recovers {:.1}% of step)  compiled-vs-eager: max|Δ|={max_diff:.3e}  max|out|={max_out:.3e}  \
         rel={rel:.3e}  (bound {COMPILED_GLUE_WHOLE_FWD_REL_TOL:.3e}, {:.1}× headroom)",
        eag / cmp,
        (eag - cmp) / eag * 100.0,
        COMPILED_GLUE_WHOLE_FWD_REL_TOL / rel.max(f32::MIN_POSITIVE),
    );
    assert!(
        rel <= COMPILED_GLUE_WHOLE_FWD_REL_TOL,
        "Qwen {label} compiled glue diverged from eager on real weights: rel={rel:e} \
         (max|Δ|={max_diff:e} / max|out|={max_out:e}) exceeds {COMPILED_GLUE_WHOLE_FWD_REL_TOL:e}. \
         That bound is 7x the worst of an 18-shape two-arm sweep whose floor is this stack's \
         amplification of a sub-ULP rounding difference, so exceeding it is a real fusion \
         divergence, not drift — see COMPILED_GLUE_WHOLE_FWD_REL_TOL"
    );
}

/// f32 packed image latents `[1, seq, 64]` and bf16 text embeds `[1, txt, 3584]` at production dtype.
fn inputs(img_seq: i32, txt_seq: i32) -> (Array, Array) {
    let key = random::key(0).unwrap();
    let hidden = random::normal::<f32>(&[1, img_seq, 64], None, None, Some(&key)).unwrap();
    let encoder = random::normal::<f32>(&[1, txt_seq, 3584], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    mlx_rs::transforms::eval([&hidden, &encoder]).unwrap();
    (hidden, encoder)
}

#[test]
#[ignore = "needs real Qwen-Image weights (MLX_GEN_QWEN_SNAPSHOT tier dir)"]
fn qwen_t2i_per_step_compiled_vs_eager() {
    let snap = snapshot("MLX_GEN_QWEN_SNAPSHOT");
    let size = env_i32("QWEN_PERF_SIZE", 1024);
    let txt_seq = env_i32("QWEN_PERF_TXT", 128);
    let lat = size / 16; // patched grid (VAE/8 then 2×2 patch)
    let img_seq = lat * lat;
    println!(
        "Qwen-Image T2I: {size}x{size} -> img_seq={img_seq} (grid {lat}x{lat}), txt_seq={txt_seq}"
    );

    let t: QwenTransformer = load_transformer(&snap).expect("load Qwen-Image transformer");
    let (hidden, encoder) = inputs(img_seq, txt_seq);
    let (lat, sigma) = (lat as usize, 1.0f32);
    ab("T2I", || {
        t.forward(&hidden, &encoder, None, sigma, lat, lat, &[])
            .unwrap()
    });
}

#[test]
#[ignore = "needs real Qwen-Image-Edit-2511 weights (QWEN_IMAGE_EDIT_SNAPSHOT tier dir)"]
fn qwen_edit_per_step_compiled_vs_eager() {
    let snap = snapshot("QWEN_IMAGE_EDIT_SNAPSHOT");
    let size = env_i32("QWEN_PERF_SIZE", 1024);
    let txt_seq = env_i32("QWEN_PERF_TXT", 128);
    let lat = (size / 16) as usize; // noise grid; one same-size reference (cond_grids=[(lat,lat)])
    let noise_seq = (lat * lat) as i32;
    let img_seq = noise_seq * 2; // noise + one reference, concatenated (dual-latent edit)
    println!("Qwen-Image-Edit: {size}x{size} -> img_seq={img_seq} (noise {lat}x{lat} + ref {lat}x{lat}), txt_seq={txt_seq}");

    let t: QwenTransformer =
        load_transformer_edit(&snap).expect("load Qwen-Image-Edit transformer");
    let (hidden, encoder) = inputs(img_seq, txt_seq);
    let sigma = 1.0f32;
    let cond_grids = [(lat, lat)];
    ab("Edit", || {
        t.forward(&hidden, &encoder, None, sigma, lat, lat, &cond_grids)
            .unwrap()
    });
}
