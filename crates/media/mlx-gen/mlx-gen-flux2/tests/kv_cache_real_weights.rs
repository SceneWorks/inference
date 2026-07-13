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

fn kv_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_KV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b-kv/snapshots");
    std::fs::read_dir(&snaps)
        .expect("the FLUX.2-klein-9b-kv snapshot dir (set MLX_GEN_FLUX2_KV_SNAPSHOT)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a -kv snapshot dir")
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

/// Wall-clock of a single warm generate for `id` (load + one warmup generate, then the timed one).
/// The model is dropped before returning so only one ~30 GB model is resident at a time.
fn time_generate(id: &str, size: u32, nref: usize) -> f64 {
    let gen = mlx_gen_flux2::provider_registry()
        .unwrap()
        .load(id, &LoadSpec::new(WeightsSource::Dir(kv_snapshot())))
        .unwrap();
    let req = edit_request(size, nref);
    // Warmup: first call pays kernel compilation / lazy graph setup.
    let _ = gen.generate(&req, &mut |_| {}).unwrap();
    let t0 = Instant::now();
    let _ = gen.generate(&req, &mut |_| {}).unwrap();
    t0.elapsed().as_secs_f64()
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
#[ignore = "needs the real FLUX.2-klein-9b-kv snapshot (~49 GB); heavy (4 generates)"]
fn kv_cache_delivers_edit_speedup() {
    let (size, nref) = (res(), nref());
    // Same -kv weights, cache OFF (plain edit id) vs cache ON (kv edit id).
    let t_off = time_generate("flux2_klein_9b_edit", size, nref);
    let t_on = time_generate("flux2_klein_9b_kv_edit", size, nref);
    let speedup = t_off / t_on;
    println!(
        "flux2 9b-kv edit ({size}², 4 steps, {nref} ref): cache-off {t_off:.2}s vs cache-on \
         {t_on:.2}s → {speedup:.2}× (fork fair A/B: ~1.4× single-ref, higher multi-ref)"
    );
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
