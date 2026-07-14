//! sc-12009 — Mac-GPU A/B for the Krea "text style" tap-reweight gain (`text_style_gain`) on the
//! **pose-control** path. The txt2img/img2img seam was covered by `tap_reweight_real_weights.rs`
//! (sc-11884); this extends the same knob to the `krea_2_turbo_control` lane, whose CFG-free encode
//! lives in `model_control.rs` (a separate seam from `model::Krea::encode_contexts`). Drives the real
//! `provider_registry().load("krea_2_turbo_control", ..)` → `generate` contract with a fixed pose +
//! seed, sweeping only the gain, and confirms:
//!   1. `g = 1.0` (and `None`) is a **no-op** — byte-identical pixels (a plain control render is
//!      unchanged, so existing recipes stay stable).
//!   2. `g = 1.75` moves the render materially MORE than the late-biased `g = 0.5` — the early-tap
//!      emphasis reaches the control lane too, on shape-identical taps.
//!
//! `#[ignore]` — needs the real Krea 2 Turbo control weights (defaults to the HF cache):
//! ```sh
//! cargo test -p mlx-gen-krea --release --test tap_reweight_control_real_weights -- --ignored --nocapture
//! # or point at explicit tiers:
//! KREA_CONTROL_DIR=…/krea-2-turbo-mlx/…/bf16 KREA_CONTROL_OVERLAY=…/control_step5000.safetensors \
//!   cargo test -p mlx-gen-krea --release --test tap_reweight_control_real_weights -- --ignored --nocapture
//! ```

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use std::path::PathBuf;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// First snapshot dir under an HF-cache `models--…` entry.
fn hf_snapshot(model: &str) -> PathBuf {
    let snaps = home()
        .join(".cache/huggingface/hub")
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {model}: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn base_dir() -> PathBuf {
    std::env::var("KREA_CONTROL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--SceneWorks--krea-2-turbo-mlx").join("bf16"))
}

fn overlay() -> PathBuf {
    std::env::var("KREA_CONTROL_OVERLAY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--SceneWorks--krea2-pose-controlnet-beta")
                .join("control_step5000.safetensors")
        })
}

/// A deterministic RGB pose stand-in — content is irrelevant, the A/B holds the pose + seed fixed
/// across every render and only sweeps the gain (mirrors `control_branch_quant_real_weights.rs`).
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

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) — a coherent natural image has a broad
/// histogram AND spatial smoothness; pure noise has a high adjacent Δ and flat std.
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

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(base_dir()))
        .with_control(WeightsSource::File(overlay()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_quant(Quant::Q4)
}

fn request(size: u32, gain: Option<f32>) -> GenerationRequest {
    GenerationRequest {
        prompt: "a person standing in a studio, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_TSG_STEPS", 8)),
        text_style_gain: gain,
        conditioning: vec![Conditioning::Control {
            image: fixed_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    }
}

fn save(img: &Image, name: &str) {
    let dir = std::path::Path::new("/tmp/krea_text_style_control");
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

/// sc-12009 Mac-GPU A/B: the gain is read + applied on the real MLX pose-control wire, `g=1.0` is a
/// byte-exact no-op, and early-emphasis (`g=1.75`) moves the render more than late-bias (`g=0.5`).
#[test]
#[ignore = "needs the real Krea 2 Turbo control weights (HF cache or KREA_CONTROL_DIR/OVERLAY)"]
fn text_style_gain_control_sweep() {
    let registry =
        mlx_gen_krea::provider_registry().expect("build explicit Krea provider registry");
    let size = env_u32("KREA_TSG_SIZE", 768);

    // (label, gain): None baseline; g=1.0 must hit the no-op fast path; g>1 early-emphasis, g<1 late.
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
        let model = registry
            .load("krea_2_turbo_control", &spec())
            .unwrap_or_else(|e| panic!("load krea_2_turbo_control: {e}"));
        let out = model
            .generate(&request(size, gain), &mut |_: Progress| {})
            .unwrap_or_else(|e| panic!("generate (gain={gain:?}): {e}"));
        let GenerationOutput::Images(mut imgs) = out else {
            panic!("expected images");
        };
        let img = imgs.swap_remove(0);
        assert_eq!((img.width, img.height), (size, size), "output dims");

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

    // Invariant 1 — the no-op fast path: g=1.0 is byte-identical to the None baseline.
    let mad_1p00 = mad_of["gain_1p00"];
    assert_eq!(
        mad_1p00, 0.0,
        "g=1.0 must be a byte-exact no-op on the control lane (got MAD {mad_1p00})"
    );

    // Invariant 2 — the knob is actually read on the control wire: g≠1 moves the render.
    for label in ["gain_1p75", "gain_0p50"] {
        assert!(
            mad_of[label] > 0.0,
            "{label} must change the control render vs baseline (MAD {} — field not applied?)",
            mad_of[label]
        );
    }

    // Invariant 3 — early-emphasis dominates on the control lane too (parity with txt2img sc-11884).
    let (mad_hi, mad_lo) = (mad_of["gain_1p75"], mad_of["gain_0p50"]);
    assert!(
        mad_hi > mad_lo,
        "early-emphasis g=1.75 (MAD {mad_hi}) should move more than late-bias g=0.5 (MAD {mad_lo})"
    );
    eprintln!(
        "\nsc-12009 control OK: g=1.0 no-op (MAD 0), g=1.75 MAD {mad_hi:.3} > g=0.5 MAD {mad_lo:.3} \
         (tap-reweight reaches the pose-control lane)."
    );
}
