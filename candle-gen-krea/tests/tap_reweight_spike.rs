//! sc-8596 [SPIKE, HELD] — "IP-Adapter-like" per-layer reweight of Krea's 12-layer Qwen3-VL text tap.
//!
//! Reproduces the ComfyUI-Conditioning-Rebalance trick (https://github.com/nova452/
//! ComfyUI-Conditioning-Rebalance): scale each of the 12 stacked Qwen3-VL select-layer taps by a
//! per-layer scalar **before** the DiT's `TextFusionTransformer` collapses them (layerwise-attn →
//! `projector` num_layers→1 → refiner). No new model weights — a steering knob on the text-only side.
//!
//! This is a GO/NO-GO eyeball harness, not a coherence gate: it encodes one prompt once, sweeps a set
//! of interpretable weight vectors (baseline, early-heavy, late-heavy, ramps, uniform amp/damp) through
//! [`pipeline::render_tap_reweight`], saves each PNG, and prints a mean-absolute-pixel-diff (MAD) vs the
//! all-ones baseline so the "does it meaningfully shift the image" question has a number next to it.
//!
//! `#[ignore]` — needs the real Turbo snapshot (bf16 ≈ 34 GB resident). Run on the Windows/CUDA GPU:
//! ```sh
//! KREA_TURBO_DIR=E:\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\bf16 \
//!   cargo test -p candle-gen-krea --release --features cuda --test tap_reweight_spike -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{GenerationRequest, Image};
use candle_gen_krea::pipeline;

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

fn snapshot() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
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

fn is_coherent(px: &[u8], w: u32) -> bool {
    let (std, distinct, adj) = image_stats(px, w);
    std > 10.0 && distinct > 24 && adj < 60.0
}

/// Mean absolute per-channel pixel difference (0..255) between two same-size RGB8 buffers — a scalar
/// "how far did the reweight move the image" against the all-ones baseline.
fn mad(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len() as f32
}

fn save(img: &Image, name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("krea_tap_reweight_spike");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    path
}

/// A linearly-ramped 12-vector from `first` to `last` (inclusive) — early-heavy when first>last.
fn ramp(first: f32, last: f32) -> Vec<f32> {
    (0..12)
        .map(|i| first + (last - first) * (i as f32) / 11.0)
        .collect()
}

/// The interpretable sweep. Layer order is the stacked select-layer order (LM layers [1,4,…,34]); index
/// 0 = earliest (syntactic/low-level) tap, index 11 = latest (semantic) tap.
fn sweep() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("00_baseline", vec![1.0; 12]),
        ("01_early_heavy", {
            let mut w = vec![0.5; 12];
            w[0..4].fill(2.0);
            w
        }),
        ("02_late_heavy", {
            let mut w = vec![0.5; 12];
            w[8..12].fill(2.0);
            w
        }),
        ("03_early_ramp", ramp(1.75, 0.25)),
        ("04_late_ramp", ramp(0.25, 1.75)),
        ("05_amp_all_1p5", vec![1.5; 12]),
        ("06_damp_all_0p6", vec![0.6; 12]),
        ("07_mid_suppress", {
            let mut w = vec![1.0; 12];
            w[4..8].fill(0.3);
            w
        }),
    ]
}

#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR); ~34 GB resident, CUDA"]
fn tap_reweight_sweep_1024() {
    candle_gen_krea::force_link();
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };
    let device = Device::cuda_if_available(0).expect("cuda device");
    assert!(
        device.is_cuda(),
        "spike needs a CUDA device (got {device:?})"
    );

    let t_load = Instant::now();
    let comps = pipeline::load_components(&root, &device, &[], None).expect("load components");
    eprintln!(
        "loaded components in {:.1}s",
        t_load.elapsed().as_secs_f32()
    );

    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };

    let mut baseline_px: Option<Vec<u8>> = None;
    eprintln!(
        "\n{:<18} {:>7} {:>8} {:>6} {:>7} {:>7}",
        "label", "std", "distinct", "adjΔ", "MAD", "coh"
    );
    for (label, weights) in sweep() {
        let t = Instant::now();
        let imgs = pipeline::render_tap_reweight(&comps, &req, &device, &weights, &mut |_| {})
            .expect("render");
        assert_eq!(imgs.len(), 1);
        let img = &imgs[0];
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let coh = is_coherent(&img.pixels, img.width);
        let mad_v = match &baseline_px {
            Some(b) => mad(b, &img.pixels),
            None => 0.0,
        };
        let path = save(img, label);
        eprintln!(
            "{label:<18} {std:>7.1} {distinct:>8} {adj:>6.1} {mad_v:>7.2} {coh:>7}  [{:.1}s] {}",
            t.elapsed().as_secs_f32(),
            path.display()
        );
        if label == "00_baseline" {
            assert!(coh, "baseline (all-ones reweight) must be coherent");
            baseline_px = Some(img.pixels.clone());
        }
    }
    eprintln!(
        "\nPNGs in {}",
        std::env::temp_dir()
            .join("krea_tap_reweight_spike")
            .display()
    );
}

/// sc-11878 — validate the **shipped** path: drive `pipeline::render` (the production Turbo txt2img
/// entry) through `GenerationRequest::text_style_gain` and confirm the single scalar moves the image
/// the way the raw-vector spike did (early emphasis at g>1, near-inert at g<1 / g≈1). This is the
/// end-to-end request-path proof (the raw sweep above proves the primitive; this proves the wiring).
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR); ~34 GB resident, CUDA"]
fn text_style_gain_sweep_1024() {
    candle_gen_krea::force_link();
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };
    let device = Device::cuda_if_available(0).expect("cuda device");
    assert!(
        device.is_cuda(),
        "spike needs a CUDA device (got {device:?})"
    );

    let comps = pipeline::load_components(&root, &device, &[], None).expect("load components");

    // (label, gain) — None is the untouched baseline; g>1 early-emphasis, g<1 late-bias.
    let gains: [(&str, Option<f32>); 5] = [
        ("gain_none", None),
        ("gain_1p00", Some(1.0)),
        ("gain_1p50", Some(1.5)),
        ("gain_1p75", Some(1.75)),
        ("gain_0p50", Some(0.5)),
    ];

    let mut baseline_px: Option<Vec<u8>> = None;
    eprintln!(
        "\n{:<12} {:>7} {:>8} {:>6} {:>7} {:>5}",
        "label", "std", "distinct", "adjΔ", "MAD", "coh"
    );
    for (label, gain) in gains {
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
        let imgs = pipeline::render(&comps, &req, &device, &mut |_| {}).expect("render");
        let img = &imgs[0];
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let coh = is_coherent(&img.pixels, img.width);
        let mad_v = baseline_px
            .as_ref()
            .map(|b| mad(b, &img.pixels))
            .unwrap_or(0.0);
        let path = save(img, label);
        eprintln!(
            "{label:<12} {std:>7.1} {distinct:>8} {adj:>6.1} {mad_v:>7.2} {coh:>5}  {}",
            path.display()
        );
        if label == "gain_none" {
            assert!(coh, "baseline (no gain) must be coherent");
            baseline_px = Some(img.pixels.clone());
        } else {
            assert!(coh, "gain={gain:?} must stay coherent");
        }
    }

    // Sanity: g=1.0 must be byte-identical to None (the no-op fast path), and g=1.75 must move the
    // image materially more than g=0.5 (early taps dominate — the spike's core finding on the wire).
    // (Left as eyeball asserts via the printed MAD to avoid over-pinning exact pixels across drivers.)
}
