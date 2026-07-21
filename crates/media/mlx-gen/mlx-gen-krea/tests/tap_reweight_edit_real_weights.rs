//! sc-12009 — Mac-GPU A/B for the Krea "text style" tap-reweight gain (`text_style_gain`) on the
//! **image-edit** path (`krea_2_edit`, epic 10871). The edit path's positive context is an image-
//! *grounded* Qwen3-VL encode (`model::Krea::encode_contexts`'s `is_edit` branch), distinct from the
//! plain-text encode the txt2img A/B (`tap_reweight_real_weights.rs`, sc-11884) exercises. Because the
//! grounded encode returns the SAME `[b, n_tok, 12, hidden]` tap structure, `apply_tap_weights` is
//! shape-safe — this test proves it on the real wire. Loads the Raw base + the `krea2_identity_edit`
//! LoRA ONCE, then sweeps only the gain over a fixed source + seed, confirming:
//!   1. `g = 1.0` (and `None`) is a **no-op** — byte-identical pixels.
//!   2. `g = 1.75` moves the edit materially MORE than the late-biased `g = 0.5`.
//!
//! The gain applies to the POSITIVE grounded context only; the CFG-negative grounded context is left
//! untouched (guidance > 0), so the knob steers only the conditional prediction.
//!
//! Coherence is reported but NOT asserted: the source here is a deterministic synthetic gradient (the
//! A/B holds it fixed and only sweeps the gain), so an "edit" of it need not be a natural image — the
//! load-bearing claims are the no-op + monotonicity invariants, which are source-content-independent.
//!
//! `#[ignore]` — needs the real Krea 2 Raw snapshot + the identity-edit LoRA (defaults to the HF cache):
//! ```sh
//! cargo test -p mlx-gen-krea --release --test tap_reweight_edit_real_weights -- --ignored --nocapture
//! # or explicit: KREA_RAW_DIR=…/krea-2-raw-mlx/…/bf16 KREA_EDIT_LORA=…/krea2_identity_edit_v1_1.safetensors
//! ```

use mlx_gen::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec,
    Progress, WeightsSource,
};
use std::path::PathBuf;

fn hf_snapshot(model: &str) -> PathBuf {
    let snaps = std::path::PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)"))
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {model}: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn raw_dir() -> PathBuf {
    std::env::var("KREA_RAW_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--SceneWorks--krea-2-raw-mlx").join("bf16"))
}

fn edit_lora() -> PathBuf {
    std::env::var("KREA_EDIT_LORA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--conradlocke--krea2-identity-edit")
                .join("krea2_identity_edit_v1_1.safetensors")
        })
}

/// A deterministic RGB source stand-in — content is irrelevant, the A/B holds it fixed across every
/// render and only sweeps the gain.
fn fixed_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(1)) as u8);
            pixels.push((y * 255 / h.max(1)) as u8);
            pixels.push(((x + y) * 127 / (w + h).max(1)) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

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

fn is_coherent(px: &[u8], w: u32) -> bool {
    let (std, distinct, adj) = image_stats(px, w);
    std > 10.0 && distinct > 24 && adj < 60.0
}

fn mad(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "MAD needs same-size buffers");
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len().max(1) as f32
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn save(px: &[u8], w: u32, h: u32, name: &str) {
    let dir = std::path::Path::new("/tmp/krea_text_style_edit");
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(&path, px, w, h, image::ExtendedColorType::Rgb8).unwrap();
    eprintln!("  saved {}", path.display());
}

/// sc-12009 Mac-GPU A/B: the gain is read + applied on the real MLX image-edit wire (grounded
/// context), `g=1.0` is a byte-exact no-op, and early-emphasis (`g=1.75`) moves the edit more than
/// late-bias (`g=0.5`).
#[test]
#[ignore = "needs the real Krea 2 Raw snapshot + identity-edit LoRA (HF cache or KREA_RAW_DIR/KREA_EDIT_LORA)"]
fn text_style_gain_edit_sweep() {
    let spec = LoadSpec::new(WeightsSource::Dir(raw_dir())).with_adapters(vec![AdapterSpec::new(
        edit_lora(),
        1.0,
        AdapterKind::Lora,
    )]);
    let gen =
        mlx_gen_krea::load_edit(&spec).expect("load krea_2_edit engine (+ identity-edit LoRA)");

    let size = env_u32("KREA_TSG_SIZE", 768);
    let steps = env_u32("KREA_TSG_STEPS", 20);
    let source = fixed_image(size, size);
    let render = |gain: Option<f32>| -> Image {
        let req = GenerationRequest {
            prompt: "make the person smile warmly, keep their identity".into(),
            width: size,
            height: size,
            count: 1,
            seed: Some(0),
            steps: Some(steps),
            guidance: Some(3.5),
            text_style_gain: gain,
            conditioning: vec![Conditioning::Reference {
                image: source.clone(),
                strength: Some(1.0),
            }],
            ..Default::default()
        };
        let GenerationOutput::Images(mut imgs) = gen
            .generate(&req, &mut |_: Progress| {})
            .unwrap_or_else(|e| panic!("generate (gain={gain:?}): {e}"))
        else {
            panic!("expected images");
        };
        imgs.swap_remove(0)
    };

    let cases: [(&str, Option<f32>); 4] = [
        ("gain_none", None),
        ("gain_1p00", Some(1.0)),
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
        let img = render(gain);
        assert_eq!((img.width, img.height), (size, size), "output dims");
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let (mr, mg, mb) = channel_means(&img.pixels);
        let coh = is_coherent(&img.pixels, img.width);
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
        save(&img.pixels, img.width, img.height, label);
        if label == "gain_none" {
            baseline = Some(img.pixels.clone());
        }
    }

    // Invariant 1 — the no-op fast path: g=1.0 is byte-identical to the None baseline on the edit lane.
    let mad_1p00 = mad_of["gain_1p00"];
    assert_eq!(
        mad_1p00, 0.0,
        "g=1.0 must be a byte-exact no-op on the edit lane (got MAD {mad_1p00})"
    );

    // Invariant 2 — the knob is read on the grounded edit wire: g≠1 moves the render.
    for label in ["gain_1p75", "gain_0p50"] {
        assert!(
            mad_of[label] > 0.0,
            "{label} must change the edit render vs baseline (MAD {} — field not applied to the \
             grounded context?)",
            mad_of[label]
        );
    }

    // Invariant 3 — early-emphasis dominates on the edit lane too (parity with txt2img sc-11884).
    let (mad_hi, mad_lo) = (mad_of["gain_1p75"], mad_of["gain_0p50"]);
    assert!(
        mad_hi > mad_lo,
        "early-emphasis g=1.75 (MAD {mad_hi}) should move more than late-bias g=0.5 (MAD {mad_lo})"
    );
    eprintln!(
        "\nsc-12009 edit OK: g=1.0 no-op (MAD 0), g=1.75 MAD {mad_hi:.3} > g=0.5 MAD {mad_lo:.3} \
         (tap-reweight reaches the grounded image-edit lane)."
    );
}
