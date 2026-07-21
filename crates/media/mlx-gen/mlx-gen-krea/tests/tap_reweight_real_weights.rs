//! sc-11884 — Mac-GPU A/B for the Krea "text style" tap-reweight gain (`text_style_gain`), the MLX
//! parity twin of candle-gen-krea's `text_style_gain_sweep_1024` (sc-11878). Drives the SHIPPED
//! `Generator` contract (`mlx_gen_krea::load` → `gen.generate`) with the gain set on the request, so it
//! exercises the real worker seam (`model::Krea::encode_contexts` → `pipeline::maybe_apply_style_gain`),
//! not just the primitive. Confirms the two invariants the story calls for:
//!   1. `g = 1.0` (and `None`) is a **no-op** — byte-identical pixels to the untouched baseline.
//!   2. `g = 1.75` produces the early-tap warm/rich shift the candle A/B showed — a coherent image that
//!      moves materially MORE than the late-biased `g = 0.5`.
//!
//! `#[ignore]` — needs the real Krea 2 Turbo snapshot (bf16 ≈ 32 GB resident, or `KREA_QUANT=q8`):
//! ```sh
//! KREA_TURBO_DIR=/path/to/models--SceneWorks--krea-2-turbo-mlx/snapshots/<rev>/bf16 \
//!   cargo test -p mlx-gen-krea --release --test tap_reweight_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

fn snapshot() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer — a coherent natural
/// image has a broad histogram AND spatial smoothness; pure noise has a high adjacent Δ and flat std.
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

/// Mean per-channel (R, G, B) over an interleaved RGB8 buffer — a "warm" shift raises R relative to B.
fn channel_means(px: &[u8]) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = (0f64, 0f64, 0f64);
    for c in px.chunks_exact(3) {
        r += c[0] as f64;
        g += c[1] as f64;
        b += c[2] as f64;
    }
    let n = (px.len() / 3).max(1) as f64;
    ((r / n) as f32, (g / n) as f32, (b / n) as f32)
}

/// A real Turbo render has a broad histogram (`std`/`distinct`) and spatial smoothness (`adjΔ`); pure
/// noise (the failure mode of a flow-sign / schedule-direction bug) fails the `adjΔ` gate.
fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

/// Mean absolute per-byte difference between two same-size RGB8 buffers (0 = byte-identical).
fn mad(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "MAD needs same-size buffers");
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len().max(1) as f32
}

fn save(img: &Image, name: &str) {
    let dir = std::path::Path::new("/tmp/krea_text_style_sweep");
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

fn render_gain(gen: &dyn mlx_gen::Generator, gain: Option<f32>) -> Image {
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        text_style_gain: gain,
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let GenerationOutput::Images(mut imgs) = out else {
        panic!("expected GenerationOutput::Images");
    };
    imgs.pop().expect("one image")
}

/// sc-11884 Mac-GPU A/B: the tap-reweight gain is read + applied on the real MLX Turbo wire, `g=1.0`
/// is a byte-exact no-op, and early-emphasis (`g=1.75`) moves the image more than late-bias (`g=0.5`).
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR to the bf16/q8 tier)"]
fn text_style_gain_sweep_1024() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };

    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    match std::env::var("KREA_QUANT").ok().as_deref() {
        Some("q8") => spec = spec.with_quant(Quant::Q8),
        Some("q4") => spec = spec.with_quant(Quant::Q4),
        _ => {}
    }
    let gen = mlx_gen_krea::load(&spec).expect("load krea_2_turbo engine");

    // (label, gain): None is the untouched baseline; g=1.0 must hit the no-op fast path (identical to
    // None); g>1 early-emphasis, g<1 late-bias. Mirrors the candle sweep set.
    let cases: [(&str, Option<f32>); 5] = [
        ("gain_none", None),
        ("gain_1p00", Some(1.0)),
        ("gain_1p50", Some(1.5)),
        ("gain_1p75", Some(1.75)),
        ("gain_0p50", Some(0.5)),
    ];

    let mut baseline: Option<Vec<u8>> = None;
    let mut mad_of = std::collections::HashMap::<&str, f32>::new();

    eprintln!(
        "\n{:<11} {:>7} {:>8} {:>6} {:>8} {:>16} {:>4}",
        "label", "std", "distinct", "adjΔ", "MAD", "meanRGB", "coh"
    );
    for (label, gain) in cases {
        let img = render_gain(gen.as_ref(), gain);
        assert_eq!((img.width, img.height), (1024, 1024), "output dims");
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let (mr, mg, mb) = channel_means(&img.pixels);
        let coh = is_coherent(&img);
        let mad_v = baseline
            .as_ref()
            .map(|b| mad(b, &img.pixels))
            .unwrap_or(0.0);
        mad_of.insert(label, mad_v);
        eprintln!(
            "{label:<11} {std:>7.1} {distinct:>8} {adj:>6.1} {mad_v:>8.3} \
             {:>16} {coh:>4}",
            format!("{mr:.1}/{mg:.1}/{mb:.1}")
        );
        save(&img, label);

        assert!(
            coh,
            "{label} (gain={gain:?}) must render a coherent image, not noise"
        );
        if label == "gain_none" {
            baseline = Some(img.pixels.clone());
        }
    }

    // Invariant 1 — the no-op fast path: g=1.0 is byte-identical to the untouched baseline (both skip
    // `maybe_apply_style_gain`'s reweight). This is the recipes-stay-byte-identical guarantee.
    let mad_1p00 = mad_of["gain_1p00"];
    assert_eq!(
        mad_1p00, 0.0,
        "g=1.0 must be a byte-exact no-op vs the None baseline (got MAD {mad_1p00})"
    );

    // Invariant 2 — the knob is actually read on the wire: every g≠1 moves the image.
    for label in ["gain_1p50", "gain_1p75", "gain_0p50"] {
        assert!(
            mad_of[label] > 0.0,
            "{label} must change the render vs baseline (MAD {} — field not applied?)",
            mad_of[label]
        );
    }

    // Invariant 3 — the spike's core finding reproduced on MLX: early-emphasis (g=1.75, the early_ramp
    // 1.75→0.25) shifts the image MORE than the late-biased g=0.5. Matches the candle A/B on the wire.
    let (mad_hi, mad_lo) = (mad_of["gain_1p75"], mad_of["gain_0p50"]);
    assert!(
        mad_hi > mad_lo,
        "early-emphasis g=1.75 (MAD {mad_hi}) should move more than late-bias g=0.5 (MAD {mad_lo})"
    );
    eprintln!(
        "\nsc-11884 OK: g=1.0 no-op (MAD 0), g=1.75 MAD {mad_hi:.3} > g=0.5 MAD {mad_lo:.3} \
         (early-tap emphasis dominates — parity with the candle A/B)."
    );
}
