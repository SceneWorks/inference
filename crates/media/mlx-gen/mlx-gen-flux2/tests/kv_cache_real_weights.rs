//! sc-2347 — real-weights validation of the FLUX.2-klein-9b-kv reference-K/V cache: the ~2.4×
//! single-reference edit speedup + output coherence. `#[ignore]`d (needs the ~49 GB `-kv`
//! snapshot). Run:
//!
//!   MLX_GEN_FLUX2_KV_SNAPSHOT=... \
//!   cargo test -p mlx-gen-flux2 --test kv_cache_real_weights -- --ignored --nocapture
//!
//! **A/B isolation.** Both ids load the *same* `-kv` checkpoint, so the only difference is the
//! cache: `flux2_klein_9b_kv_edit` runs the cache (step-0 extract, steps 1+ cached);
//! `flux2_klein_9b_edit` runs every step over the full `[txt, target, ref]` sequence. The speedup
//! is the cache mechanism in isolation (no weights confound). Override resolution / reference count
//! with `MLX_GEN_FLUX2_KV_RES` (default 1024) / `MLX_GEN_FLUX2_KV_NREF` (default 1).
//!
//! **Verified ground truth (M5 Max, 1024², 4 steps, this port vs the fork's own fair A/B):**
//! the steady-state single-reference cache speedup is ~1.4–1.5× — Rust **1.47×** (44.2→30.0s,
//! f32 acts) tracks the fork's **1.41×** (18.4→13.1s, bf16) within noise. The cache saves work
//! proportional to the reference:(text+target) token ratio, so it scales with reference count:
//! BFL's headline "up to 2.5×" is **multi-reference** editing. (sc-2163's "2.4× single-ref" figure
//! compared `-kv`-cache against an *inflated* base-9b baseline — the fork's own cache-off on these
//! weights is 18.4s, not the 33.0s that figure used.)

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::media::Image;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

const PROMPT: &str = "make it look like a cold winter morning";

/// Measured runs per arm in [`time_generate`], reduced with `min` (sc-19556). Two is the cheapest
/// count that can distinguish "this run was descheduled" from "this arm is slow"; each run here is
/// a full 4-step edit, so the cost is real and the count is deliberately small.
const TIMED_RUNS: usize = 2;

fn kv_snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_FLUX2_KV_SNAPSHOT").unwrap_or_else(|_| panic!("set MLX_GEN_FLUX2_KV_SNAPSHOT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

fn res() -> u32 {
    std::env::var("MLX_GEN_FLUX2_KV_RES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

fn nref() -> usize {
    std::env::var("MLX_GEN_FLUX2_KV_NREF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1)
}

/// A deterministic RGB reference image (a diagonal gradient varied by `seed`). The edit
/// preprocessing resizes it to the target size, so any dimensions work; the speedup is
/// content-independent.
fn synthetic_ref(size: u32, seed: usize) -> Image {
    let s = size as usize;
    let mut pixels = Vec::with_capacity(s * s * 3);
    for y in 0..s {
        for x in 0..s {
            let r = (((x + seed * 40) * 255) / s) as u8;
            let g = ((y * 255) / s) as u8;
            let b = (((x + y + seed * 17) * 127) / s) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
    }
}

fn edit_request(size: u32, nref: usize) -> GenerationRequest {
    let conditioning = if nref == 1 {
        vec![Conditioning::Reference {
            image: synthetic_ref(size, 0),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference {
            images: (0..nref).map(|i| synthetic_ref(size, i)).collect(),
        }]
    };
    GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(0),
        steps: Some(4),
        conditioning,
        ..Default::default()
    }
}

fn render(id: &str, size: u32, nref: usize) -> Image {
    render_quant(id, size, nref, None)
}

/// As [`render`], with an optional whole-model quantization (sc-2643) applied at load — exercises
/// the cache over quantized linears (their `quantized_matmul` still emits f32 K/V, so the cache is
/// orthogonal; this proves it runs + stays coherent).
fn render_quant(id: &str, size: u32, nref: usize, quant: Option<Quant>) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(kv_snapshot()));
    spec.quantize = quant;
    let gen = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load(id, &spec)
        .unwrap();
    let req = edit_request(size, nref);
    let GenerationOutput::Images(mut images) = gen.generate(&req, &mut |_| {}).unwrap() else {
        panic!("expected images");
    };
    images.pop().unwrap()
}

/// Wall-clock of a warm generate for `id`, reduced with `min` over [`TIMED_RUNS`] measured runs.
///
/// sc-19556: this used to time ONE run. The cache's published effect is a wall-clock speedup, so
/// this gate keeps a duration — but a single sample makes the ratio below a comparison of two
/// arbitrary points on a contended machine, and the two arms are timed minutes apart with a ~30 GB
/// model load in between, so they do not even see the same load. Each arm runs identical work every
/// time, so contention can only push a run SLOWER and the fastest run is a lower bound on what the
/// hardware did. The ratio of two lower bounds is the honest form of "the cache saves work".
///
/// The model is dropped before returning so only one ~30 GB model is resident at a time.
fn time_generate(id: &str, size: u32, nref: usize) -> f64 {
    let gen = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load(id, &LoadSpec::new(WeightsSource::Dir(kv_snapshot())))
        .unwrap();
    let req = edit_request(size, nref);
    // Warmup: first call pays kernel compilation / lazy graph setup.
    let _ = gen.generate(&req, &mut |_| {}).unwrap();
    let mut fastest = f64::INFINITY;
    let mut last: Option<Image> = None;
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let out = gen.generate(&req, &mut |_| {}).unwrap();
        fastest = fastest.min(t0.elapsed().as_secs_f64());
        let GenerationOutput::Images(mut images) = out else {
            panic!("expected images");
        };
        last = images.pop();
    }
    // The timed path must produce a real image, not merely a fast one: a generator that returned
    // early would read as an excellent speedup (sc-19556). This gates the run that was ACTUALLY
    // TIMED, on BOTH arms of the ratio, and costs nothing extra — the output was already being
    // computed and thrown away with `let _ =`. Re-rendering the id in the test body to check the
    // same thing would instead have cost a third ~49 GB model load and covered only one arm.
    let img = last.expect("TIMED_RUNS must be > 0");
    let (mean, std) = coherence(&img);
    assert!(
        mean > 2.0 && mean < 253.0 && std > 5.0,
        "the timed `{id}` output is degenerate (mean={mean:.1}, std={std:.1}) — a speedup measured \
         against garbage is not a speedup"
    );
    fastest
}

/// Output coherence: finite, in range, and not degenerate (a flat/black frame would mean the cache
/// produced garbage). A real per-channel spread proves the cached edit is a real image.
fn coherence(img: &Image) -> (f64, f64) {
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

#[test]
#[ignore = "needs the real FLUX.2-klein-9b-kv snapshot (~49 GB)"]
fn kv_cache_edit_is_coherent() {
    let (size, nref) = (res(), nref());
    let img = render("flux2_klein_9b_kv_edit", size, nref);
    let (mean, std) = coherence(&img);
    println!("flux2 9b-kv edit ({size}², {nref} ref) cache-on output: mean={mean:.1} std={std:.1}");
    assert!(
        mean > 2.0 && mean < 253.0 && std > 5.0,
        "cache-on edit looks degenerate (mean={mean}, std={std}) — cache produced garbage?"
    );
}

#[test]
#[ignore = "needs the real FLUX.2-klein-9b-kv snapshot (~49 GB)"]
fn q8_kv_cache_edit_is_coherent() {
    // The cache is orthogonal to weight quantization (it stores f32 activations; quant only touches
    // weights), so Q8 + cache must produce a coherent edit. This validates the `-kv` variant's
    // inherited Q4/Q8 path (sc-2643) running *with* the cache.
    let (size, nref) = (res(), nref());
    let img = render_quant("flux2_klein_9b_kv_edit", size, nref, Some(Quant::Q8));
    let (mean, std) = coherence(&img);
    println!(
        "flux2 9b-kv edit ({size}², {nref} ref) Q8 + cache-on output: mean={mean:.1} std={std:.1}"
    );
    assert!(
        mean > 2.0 && mean < 253.0 && std > 5.0,
        "Q8 cache-on edit looks degenerate (mean={mean}, std={std}) — cache×quant broke?"
    );
}

#[test]
#[ignore = "needs the real FLUX.2-klein-9b-kv snapshot (~49 GB); heavy (6 generates: 2 arms × 1 \
            warmup + TIMED_RUNS timed)"]
fn kv_cache_delivers_edit_speedup() {
    let (size, nref) = (res(), nref());
    // Same -kv weights, cache OFF (plain edit id) vs cache ON (kv edit id).
    let t_off = time_generate("flux2_klein_9b_edit", size, nref);
    let t_on = time_generate("flux2_klein_9b_kv_edit", size, nref);
    let speedup = t_off / t_on;
    println!(
        "flux2 9b-kv edit ({size}², 4 steps, {nref} ref): cache-off {t_off:.2}s vs cache-on \
         {t_on:.2}s → {speedup:.2}× (fastest of {TIMED_RUNS} runs per arm; fork fair A/B: ~1.4× \
         single-ref, higher multi-ref)"
    );
    // Coherence of the timed output is gated inside `time_generate`, on the run that was actually
    // timed and on both arms — see the comment there for why it does not happen here.
    //
    // The cache must materially reduce work. The steady-state single-ref effect is ~1.4× at 1024²
    // (verified equal to the fork); the floor is set below that to tolerate timing noise, and scales
    // up with reference count (each extra ref adds `target`-many cached-away tokens).
    let floor = match (size >= 768, nref) {
        (true, 1) => 1.25,
        (true, _) => 1.5,
        (false, _) => 1.05,
    };
    assert!(
        speedup > floor,
        "KV-cache speedup {speedup:.2}× below the {floor}× floor ({size}², {nref} ref) — \
         cache not reducing work"
    );
}
