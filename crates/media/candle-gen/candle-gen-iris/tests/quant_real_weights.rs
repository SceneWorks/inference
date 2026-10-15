//! sc-26266 real-weight gates for the `q8` / `q4` tiers against the pinned `speridlabs/iris-3b` +
//! `Qwen/Qwen3-VL-4B-Instruct` snapshots on the build's device (`cuda:0` under `--features cuda`).
//! `#[ignore]`d: they need ~21 GB of weights and a GPU. Inputs come from the environment, like
//! `real_weights`:
//!
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (generation backbone).
//! * `IRIS_TEXT_ENCODER_DIR` — the `Qwen/Qwen3-VL-4B-Instruct` snapshot.
//! * `IRIS_OUT` — a directory the tier renders write their PNGs into (render gate only).
//! * `IRIS_QUANT_STEPS` — solver steps of the render gate (default 30).
//!
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! quant_real_weights:: --test-threads 1`

use std::path::PathBuf;

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant, WeightsSource,
};
use candle_gen_iris::memory_strategy::generation_contract;
use candle_gen_iris::{load_backbone_quantized, IrisTextEncoder, MODEL_ID};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

const TIERS: [Option<Quant>; 3] = [None, Some(Quant::Q8), Some(Quant::Q4)];

fn spec(quant: Option<Quant>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
    spec.quantize = quant;
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
    );
    spec
}

/// `measured / dense` must fall in `[lo, hi]`.
fn assert_ratio(what: &str, measured: u64, dense: u64, lo: f64, hi: f64) {
    let ratio = measured as f64 / dense as f64;
    eprintln!(
        "[[IRIS_QUANT]] {what}: {:.3} GiB = {ratio:.3} x bf16",
        measured as f64 / f64::from(1 << 30)
    );
    assert!(
        (lo..=hi).contains(&ratio),
        "{what}: {ratio:.3} x bf16 is outside [{lo}, {hi}]"
    );
}

/// The resident weight bytes of each component at each tier, measured from the LOADED model (the
/// backbone's built tensors, the tower's weight census), one component resident at a time, and
/// never above the contract's pricing.
///
/// Declared bounds, from the release's tensor headers (sc-26266; the GGML Q8_0 / Q4_0 / Q4_K
/// blocks cost the MLX twin's 8.5 / 4.5 bits per weight): the backbone's projections are ~all of
/// its bytes, so it lands at ~0.53 x (q8) / ~0.28 x (q4) of bf16; the tower keeps its tied
/// embedding at bf16, so ~0.58 x / ~0.35 x; the pair ~0.56 x / ~0.32 x. The windows are the MLX
/// twin's and separate the tiers from each other and from a dense load.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and a GPU"]
fn real_tiers_shrink_the_resident_weights() {
    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let config = IrisConfig::from_dir(&backbone).unwrap();
    let tower = env_dir("IRIS_TEXT_ENCODER_DIR");
    let device = candle_gen::default_device().unwrap();

    let mut dit_bytes = Vec::new();
    let mut te_bytes = Vec::new();
    for quant in TIERS {
        let start = std::time::Instant::now();
        let (dit, _) = load_backbone_quantized(
            &backbone,
            &config,
            DType::BF16,
            &device,
            &[],
            IrisTask::Generation,
            MODEL_ID,
            quant,
        )
        .unwrap();
        assert_eq!(dit.is_quantized(), quant.is_some());
        eprintln!(
            "[[IRIS_QUANT]] backbone {quant:?} load {:.1}s",
            start.elapsed().as_secs_f32()
        );
        dit_bytes.push(dit.resident_weight_bytes());
        drop(dit);

        let start = std::time::Instant::now();
        let te =
            IrisTextEncoder::load_quantized(&tower, &config.text_encoder, &device, quant).unwrap();
        assert_eq!(te.is_quantized(), quant.is_some());
        eprintln!(
            "[[IRIS_QUANT]] text encoder {quant:?} load {:.1}s",
            start.elapsed().as_secs_f32()
        );
        te_bytes.push(te.resident_weight_bytes());
        drop(te);

        let priced = generation_contract(&spec(quant)).unwrap().asset_facts;
        let last = dit_bytes.len() - 1;
        assert!(
            dit_bytes[last] <= priced.transformer_bytes,
            "{quant:?}: backbone {} > priced {}",
            dit_bytes[last],
            priced.transformer_bytes
        );
        assert!(
            te_bytes[last] <= priced.conditioning_bytes,
            "{quant:?}: tower {} > priced {}",
            te_bytes[last],
            priced.conditioning_bytes
        );
    }
    assert_ratio("backbone q8", dit_bytes[1], dit_bytes[0], 0.45, 0.57);
    assert_ratio("backbone q4", dit_bytes[2], dit_bytes[0], 0.22, 0.31);
    assert_ratio("text encoder q8", te_bytes[1], te_bytes[0], 0.50, 0.60);
    assert_ratio("text encoder q4", te_bytes[2], te_bytes[0], 0.28, 0.37);
    let pair = |i: usize| dit_bytes[i] + te_bytes[i];
    assert_ratio("backbone + text encoder q8", pair(1), pair(0), 0.48, 0.58);
    assert_ratio("backbone + text encoder q4", pair(2), pair(0), 0.25, 0.34);
}

/// Luminance of an RGB8 image box-averaged to a `cells x cells` grid.
fn thumbnail(img: &Image, cells: u32) -> Vec<f64> {
    let (cw, ch) = (img.width / cells, img.height / cells);
    let mut out = vec![0f64; (cells * cells) as usize];
    for y in 0..cells * ch {
        for x in 0..cells * cw {
            let i = ((y * img.width + x) * 3) as usize;
            let p = &img.pixels[i..i + 3];
            let luma = 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
            out[((y / ch) * cells + x / cw) as usize] += luma / f64::from(cw * ch);
        }
    }
    out
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// The production path at 512² from a fixed seed at bf16, q8 and q4 (`load`, default residency,
/// CFG 3, `IRIS_QUANT_STEPS` steps): each render is coherent and the quantized renders stay within
/// the declared tolerance of the bf16 one. The three PNGs land in `IRIS_OUT` for a visual check.
///
/// **Declared tolerance** (sc-26266 AC; the MLX twin's pre-measurement declaration, recorded with
/// the first CUDA run's numbers in the story):
/// * coherent — mean RGB8 value in (5, 250) and luminance standard deviation ≥ 12 (not flat);
/// * composition — Pearson correlation of the 32 x 32 box-averaged luminance against bf16 ≥ 0.85
///   at q8 and ≥ 0.70 at q4;
/// * pixels — mean |Δ| against bf16 ≤ 12 / 255 at q8 and ≤ 28 / 255 at q4.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR), IRIS_OUT and a GPU"]
fn real_quantized_renders_track_the_bf16_render_at_512() {
    let steps: u32 = std::env::var("IRIS_QUANT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let out = env_dir("IRIS_OUT");
    let req = GenerationRequest {
        prompt: "a red fox sleeping in fresh snow, golden hour".into(),
        width: 512,
        height: 512,
        steps: Some(steps),
        seed: Some(26266),
        ..Default::default()
    };
    let mut renders = Vec::new();
    for quant in TIERS {
        let start = std::time::Instant::now();
        let generator = candle_gen_iris::load(&spec(quant)).unwrap();
        eprintln!(
            "[[IRIS_QUANT]] {quant:?} load {:.1}s",
            start.elapsed().as_secs_f32()
        );
        let start = std::time::Instant::now();
        let mut last = (0, 0);
        let GenerationOutput::Images(mut images) = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    last = (current, total);
                }
            })
            .unwrap()
        else {
            panic!("expected images")
        };
        eprintln!(
            "[[IRIS_QUANT]] {quant:?} 512² x {steps} steps: {:.1}s",
            start.elapsed().as_secs_f32()
        );
        assert_eq!(last, (steps, steps));
        drop(generator);
        let img = images.remove(0);
        let tier = quant.map_or("bf16".to_owned(), |q| format!("{q:?}").to_lowercase());
        let path = out.join(format!("iris_3b_candle_{tier}_512.png"));
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / img.pixels.len() as f64;
        let luma = thumbnail(&img, 512);
        let lm = luma.iter().sum::<f64>() / luma.len() as f64;
        let std = (luma.iter().map(|v| (v - lm).powi(2)).sum::<f64>() / luma.len() as f64).sqrt();
        eprintln!(
            "[[IRIS_QUANT]] {tier}: wrote {} (mean {mean:.1}, luma std {std:.1})",
            path.display()
        );
        assert!(
            mean > 5.0 && mean < 250.0,
            "{tier}: degenerate image (mean {mean})"
        );
        assert!(std >= 12.0, "{tier}: a flat image (luma std {std})");
        renders.push(img);
    }
    let reference = thumbnail(&renders[0], 32);
    for (tier, img, min_corr, max_mean) in [
        ("q8", &renders[1], 0.85, 12.0),
        ("q4", &renders[2], 0.70, 28.0),
    ] {
        let corr = pearson(&thumbnail(img, 32), &reference);
        let mean = img
            .pixels
            .iter()
            .zip(&renders[0].pixels)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .sum::<f64>()
            / img.pixels.len() as f64;
        eprintln!("[[IRIS_QUANT]] {tier} vs bf16: thumbnail corr {corr:.4}, mean |Δ| {mean:.2}");
        assert!(
            corr >= min_corr,
            "{tier}: composition corr {corr} < {min_corr}"
        );
        assert!(mean <= max_mean, "{tier}: mean |Δ| {mean} > {max_mean}");
    }
}
