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
//! 1024² T2I and 1.345e-3 on Edit. sc-17513 swept it and replaced the gate with a **per-arm**
//! peak-**relative** bound: see [`COMPILED_GLUE_WHOLE_FWD_REL_TOL_T2I`] for the derivation, the
//! sweep behind it, and — the part that decides it — why the residual is this stack's accumulation
//! floor and not a fusion defect. Do not loosen either bound without repeating that sweep, which is
//! committed and re-runnable: `examples/qwen_compiled_glue_sweep.rs` produced every number below.
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

/// Peak-**relative** bound on the compiled-glue-vs-eager divergence of the **whole 60-layer T2I
/// forward**, applied via [`mlx_gen::nn::max_rel_diff`] (`max|Δ| / max|eager|`). The Edit arm has
/// its own, [`COMPILED_GLUE_WHOLE_FWD_REL_TOL_EDIT`]; see "The numbers" below for why one constant
/// could not serve both.
///
/// # This is not a ULP tolerance, and must not be read as one
///
/// `1.0e-2` is ~8.4e4 f32 ULP, and the op-level gate is 4 ULP
/// ([`mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL`]). The four orders of magnitude between them are not
/// slack: they are this stack's measured **amplification** of a sub-ULP input difference, and no
/// gate on a 60-layer output can be tighter than that floor. What follows is the measurement.
///
/// The sibling whole-forward gate is 16 ULP (`mlx-gen-z-image/tests/compile_parity.rs`), and the
/// gap is depth, not rigour: that gate runs a **2-layer, dim-96 synthetic** DiT (`compile_parity.rs`
/// `base_cfg()`), not a 60-layer real transformer. Compared at equal depth the two **agree** —
/// z-image measures 2.48 ULP at 2 layers on its synthetic weights (`compile_parity.rs`
/// `COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL`), and the depth series below measures 0.3 ULP at 2 layers
/// on real ones. That is the strongest corroboration available that the fusion itself is sound;
/// everything above it is stack depth.
///
/// # The sweep — sc-17513, 2026-08-06, nax-macos (M5 Max, 128 GiB), MLX pin 932beb4e
///
/// Reproduce with `examples/qwen_compiled_glue_sweep.rs`, one mode per process.
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
/// **1. Depth.** Same weights, same 512² input, `num_layers` truncated (which sizes the `blocks`
/// Vec only — `set_compile_glue` remains enabled on the render thread, so the fusion is fully on at
/// every row). Every column is measured, not derived:
///
/// | layers | `max|Δ|` | `max|out|` | rel | rel in ULP |
/// |---|---|---|---|---|
/// | 1 | **0.0000e0** | 114.108 | **0.0** | **0** |
/// | 2 | 3.815e-6 | 112.923 | 3.378e-8 | 0.3 |
/// | 4 | 8.789e-3 | 112.397 | 7.820e-5 | 656 |
/// | 8 | 6.542e-3 | 111.808 | 5.851e-5 | 491 |
/// | 15 | 1.206e-2 | 108.111 | 1.116e-4 | 936 |
/// | 30 | 2.667e-2 | 103.701 | 2.572e-4 | 2157 |
/// | 45 | 2.019e-1 | 98.830 | 2.043e-3 | 17141 |
/// | 60 | 1.475e-3 | 4.452 | 3.314e-4 | 2780 |
///
/// **Read the `rel` column, not `max|Δ|`.** The two are not comparable across these rows: a
/// truncated stack is not a trained one and its output magnitude is ~99…114, while the trained
/// 60-layer output is 4.45 — a 22…26× smaller denominator. That is why 60 layers shows a *smaller*
/// `max|Δ|` (1.475e-3) than 30 does (2.667e-2) while being 1.3× *worse* relatively.
///
/// At **1 layer the compiled forward is bit-identical to eager on the real weights — `max|Δ|` is
/// exactly `0.0`** — and that zero is not vacuous: the output it is zero against peaks at 114.108.
/// At 2 layers it is 0.3 ULP, the ≤1-ULP-per-fused-op f32 rounding sc-12747 documented for the MLX
/// 0.32.0 pin. From there `rel` climbs ~4 orders of magnitude to 60 layers (3.378e-8 → 3.314e-4)
/// **non-monotonically** — it falls at 8 and falls 6.2× again at 60. A wrong fusion (a swapped
/// operand, a mis-broadcast scale, a dropped `+1`) is a per-op O(1e-1) error and would be loudest
/// at layer 1. This is zero at layer 1.
///
/// **2. Conditioning — and this is what the bound is derived from.** Perturb the eager forward's
/// *own input* by k f32 ULP relative, with compiled glue never switched on, and measure how far the
/// output moves. This bounds what the stack does to **any** sub-ULP difference, independently of
/// the fusion. Run per arm (the tiers are different weight encodings) over 4 shapes × {uniform
/// gain, random noise} × {1, 4} ULP = 16 probes each; the worst `out_rel` per shape:
///
/// | px | T2I bf16 envelope | T2I compiled-vs-eager | Edit q8 envelope | Edit compiled-vs-eager |
/// |---|---|---|---|---|
/// | 256 | 9.527e-4 | 4.57e-4 | 4.391e-4 | 4.79e-5 |
/// | 384 | 2.035e-3 | 2.285e-3 | **6.525e-4** | **5.70e-4** |
/// | 512 | 7.397e-4 | 3.31e-4 | 3.372e-4 | 1.17e-4 |
/// | 1024 | **2.526e-3** | 2.069e-3 | 4.414e-4 | 1.88e-4 |
///
/// Three things follow, and each is load-bearing:
///
/// - **The envelope brackets the fusion.** Each arm's whole compiled-vs-eager series (13 T2I + 5
///   Edit shapes) sits under that arm's envelope peak — 2.285e-3 < 2.526e-3, 5.70e-4 < 6.525e-4.
///   The fusion is doing nothing the stack does not already do to a rounding difference. (At 384²
///   T2I the compiled figure is 1.12× the *shape-matched* envelope; the envelope bounds the series
///   globally, not cell by cell, because the two perturbation directions sampled are not the
///   fusion's actual per-element difference.)
/// - **It is a plateau, not a slope.** Quadrupling the input perturbation does **not** quadruple
///   the response: across the 16 paired probes the largest increase is 2.1× and 6 of 16 pairs
///   *decreased* (T2I 384² gain falls 2.035e-3 → 6.336e-4). Any sub-ULP-scale input difference
///   lands on the same ~1e-3 output plateau. So the margin below is for **sampling**, not for a
///   magnitude extrapolation — extrapolating the magnitude linearly would overstate it.
/// - **The non-monotonicity belongs to the amplification, not the fusion.** The amplification is
///   itself non-monotonic in sequence length, which is why `rel` jumps an order of magnitude
///   between neighbouring shapes. The two largest envelope entries are 1024² and 384² — **the same
///   two shapes that top the 13-shape compiled-vs-eager series**, which is why 384² was measured
///   here even though it is not a CI geometry: it is the shape the T2I bound comes from.
///
/// Both paths are also **exactly reproducible**: eager-vs-eager and compiled-vs-compiled repeats
/// are `0.0` at every shape and every depth above, and the 1024² T2I (9.739e-3) and 1024² Edit
/// (1.345e-3) figures reproduce sc-17284's two-days-earlier run to every digit it printed. So the
/// headroom below is real headroom, not a noise margin.
///
/// # The numbers
///
/// **Each arm's bound is 4× its own measured conditioning envelope**, rounded down:
///
/// | arm | envelope | ×4 | bound | vs that arm's worst measured |
/// |---|---|---|---|---|
/// | T2I bf16 | 2.526e-3 | 1.010e-2 | **1.0e-2** | 4.4× (2.285e-3) |
/// | Edit q8 | 6.525e-4 | 2.610e-3 | **2.6e-3** | 4.6× (5.70e-4) |
///
/// The envelope is the right quantity to bound: it is what the stack does to an arbitrary sub-ULP
/// difference, so it covers rounding differences this sweep did not happen to produce, whereas the
/// worst *observed* fusion residual is one draw from that distribution. The ×4 is margin for
/// **shapes not sampled** and for the second host (below); it is not a magnitude extrapolation,
/// because the response plateaus rather than scaling. It leaves both bounds 10…38× below the
/// O(1e-1) a genuinely wrong fusion produces.
///
/// **One constant could not serve both arms.** The previous revision gated Edit at the T2I-derived
/// number, which left it 28× loose at its own worst shape and 85× loose at the geometry CI runs — a
/// gate that would not have caught a 20× regression on that arm. Splitting restores comparable
/// sensitivity: ~4.4× and ~4.6× of headroom against each arm's own worst.
///
/// **On the second host.** `rw-mage` is carried by two Macs and this sweep ran on one of them, so
/// the bounds are single-host. Note what this does *not* rest on: sc-12747's cross-metallib finding
/// is about the NAX vs non-NAX/dt15.0 split, and **both** `rw-mage` boxes are self-hosted NAX, so
/// by that finding's own text they should agree bit-for-bit — it is not a reason to inflate. The
/// honest statement is narrower: the second box's metallib and deployment target are **unverified**
/// here, and the ×4 absorbs a per-op ULP of disagreement if they turn out to differ. If the first
/// scheduled run on the other box lands near the bound, re-measure there and say so; do not widen.
///
/// **What this gate can and cannot catch.** Two different numbers, and conflating them overstates
/// it. The *floor* is the envelope — ~2.5e-3 (T2I) / ~6.5e-4 (Edit); nothing below that is
/// distinguishable from the stack's own conditioning, and that is a property of the model, not of
/// the bound. But the gate does not *trip* until 1.0e-2 / 2.6e-3, so the band between floor and
/// bound is a blind spot: a 5e-3 T2I divergence — 2× the floor, and a genuine anomaly — passes
/// silently. What it does catch is the regression class that matters: a genuinely wrong fusion is
/// O(1e-1) relative or worse and trips both bounds by 10…38×. The tight sensitivity lives where it
/// can: the in-crate `#[cfg(test)] mod sc2963` gates in `transformer/{block,attention,
/// feed_forward}.rs` still hold each glue primitive to `0.0` (bf16) or 4 ULP (f32) in the default
/// weight-free `cargo test`, and the depth-1 result above is the real-weight corroboration that
/// the extrapolation from those primitives to a single block is exact.
const COMPILED_GLUE_WHOLE_FWD_REL_TOL_T2I: f32 = 1.0e-2;

/// Peak-relative bound for the **Edit q8** arm — 4× that arm's own measured conditioning envelope
/// (6.525e-4), 4.6× its worst measured divergence (5.70e-4, 384²). Derived and justified alongside
/// the T2I bound on [`COMPILED_GLUE_WHOLE_FWD_REL_TOL_T2I`]; it is a separate constant because the
/// Edit tier is q8 rather than bf16 and its envelope is 3.9× smaller, so sharing one number would
/// cost this arm most of its sensitivity.
const COMPILED_GLUE_WHOLE_FWD_REL_TOL_EDIT: f32 = 2.6e-3;

/// Run the warm A/B + the real-weight compiled-vs-eager parity check for one forward closure.
/// `label` names the path; `tol` is that arm's own peak-relative bound (the two arms' conditioning
/// envelopes differ by 3.9×, so they do not share one).
fn ab<F: Fn() -> Array>(label: &str, tol: f32, run: F) {
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
         rel={rel:.3e}  (bound {tol:.3e}, {:.1}× headroom)",
        eag / cmp,
        (eag - cmp) / eag * 100.0,
        tol / rel.max(f32::MIN_POSITIVE),
    );
    assert!(
        rel <= tol,
        "Qwen {label} compiled glue diverged from eager on real weights: rel={rel:e} \
         (max|Δ|={max_diff:e} / max|out|={max_out:e}) exceeds {tol:e}. That bound is 4x this arm's \
         MEASURED conditioning envelope — how far its own eager forward moves when its input is \
         perturbed by 1-4 f32 ULP — so exceeding it is a divergence larger than this stack's \
         amplification of any sub-ULP rounding difference, i.e. a real fusion divergence and not \
         drift. See COMPILED_GLUE_WHOLE_FWD_REL_TOL_T2I; re-measure with \
         `examples/qwen_compiled_glue_sweep.rs` before touching the number"
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
    ab("T2I", COMPILED_GLUE_WHOLE_FWD_REL_TOL_T2I, || {
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
    ab("Edit", COMPILED_GLUE_WHOLE_FWD_REL_TOL_EDIT, || {
        t.forward(&hidden, &encoder, None, sigma, lat, lat, &cond_grids)
            .unwrap()
    });
}
