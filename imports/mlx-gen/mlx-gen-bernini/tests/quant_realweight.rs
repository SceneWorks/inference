//! sc-5146: full-Bernini Q4/Q8 vs bf16 — `#[ignore]`, real weights, run locally.
//!
//! Loads the full `bernini` pipeline at bf16, Q8, and Q4 from the combined snapshot and generates the
//! same t2i prompt at a fixed seed through each, measuring the peak GPU footprint per tier + the image
//! cosine vs the bf16 reference. Quantization is a quality/size knob, not a bit-parity target, so the
//! gate is directional: Q8 ≈ bf16 (near-lossless) and Q4 coherent — the same bar as the sensenova/lens
//! quant smokes. Validates the load-time quant seam end-to-end across the whole stack: the Qwen2.5-VL
//! LLM linears + both renderer experts are quantized; the vision tower (group-64-misaligned linears),
//! connector, and clip_diff flow head are kept dense.
//!
//! Run: `cargo test -p mlx-gen-bernini --test quant_realweight -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen_bernini as _; // force-link the inventory registration for `mlx_gen::load("bernini")`.

use mlx_gen::media::Image;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

/// The combined planner+renderer snapshot (assembled by `bernini_e2e::ensure_snapshot`).
fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/bernini_full_mlx_bf16")
}

/// Render one 256² / 4-step t2i and return (image, peak GPU GB) for this load tier.
fn render(spec: &LoadSpec) -> (Image, f64) {
    clear_cache();
    reset_peak_memory();
    let model = mlx_gen::load("bernini", spec).expect("load bernini");
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("t2i".into()),
        ..Default::default()
    };
    let mut noop = |_: Progress| {};
    let img = match model.generate(&req, &mut noop).expect("generate") {
        GenerationOutput::Images(mut imgs) => imgs.pop().unwrap(),
        _ => panic!("expected Images for a 1-frame request"),
    };
    let peak = get_peak_memory() as f64 / 1e9;
    (img, peak)
}

fn cosine(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let dot: f64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let na: f64 = a
        .pixels
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .pixels
        .iter()
        .map(|&y| (y as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    dot / (na * nb + 1e-12)
}

fn coherent(img: &Image) -> bool {
    img.pixels.iter().any(|&p| p != 0) && img.pixels.iter().any(|&p| p != 255)
}

#[test]
#[ignore = "real weights: loads the full Bernini pipeline at bf16 + Q8 + Q4 and renders t2i through each"]
fn quant_vs_bf16_t2i() {
    let snap = snapshot();
    if !snap.join("high_noise_model.safetensors").is_file() {
        eprintln!("skipping: combined snapshot missing at {}", snap.display());
        return;
    }
    let src = WeightsSource::Dir(snap);
    let (bf16, p_bf16) = render(&LoadSpec::new(src.clone()));
    let (q8, p_q8) = render(&LoadSpec::new(src.clone()).with_quant(Quant::Q8));
    let (q4, p_q4) = render(&LoadSpec::new(src).with_quant(Quant::Q4));

    let c8 = cosine(&bf16, &q8);
    let c4 = cosine(&bf16, &q4);
    // The peak is the process-global peak (`get_peak_memory`); the three tiers run in one process (so
    // later tiers can carry some cache residual). With the sc-5360 streaming load (load+quantize each
    // expert before the next, `WanTransformer::quantize` eval-freeing the bf16 dequant) only one bf16
    // expert is resident at a time, so the quantized **load** peak now drops below bf16 rather than
    // spiking above it. Reference run: bf16≈60.8GB, Q8≈58.0GB, Q4≈43.9GB.
    println!(
        "load peak: bf16={p_bf16:.1}GB  Q8={p_q8:.1}GB  Q4={p_q4:.1}GB\n\
         t2i cosine vs bf16: Q8={c8:.5}  Q4={c4:.5}"
    );

    assert!(
        coherent(&bf16) && coherent(&q8) && coherent(&q4),
        "all tiers render"
    );
    // Q8 is the fidelity tier — near-lossless affine quant.
    assert!(
        c8 > 0.98,
        "Q8 should be near-lossless vs bf16 (got {c8:.5})"
    );
    // Q4 is the aggressive footprint tier — 4-bit weight error shifts the image but stays the same
    // content (coherent, not pixel-close), matching the sensenova/lens quant bar.
    assert!(c4 > 0.80, "Q4 should stay coherent vs bf16 (got {c4:.5})");
    // sc-5360: the streaming load must keep the quantized load peak at/under bf16 (regression guard).
    assert!(
        p_q8 <= p_bf16 && p_q4 < p_bf16,
        "quantized load peak must be <= bf16 (bf16={p_bf16:.1} Q8={p_q8:.1} Q4={p_q4:.1})"
    );
}
