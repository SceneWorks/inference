//! sc-17150 — **the per-tier quality datum, and the AdaLN width decision**, on real weights.
//!
//! `#[ignore]`d: needs the ~196 GB upstream snapshot for the shared components, a built tier's DiT,
//! and Metal.
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<upstream snapshot root> \
//! MINIMAX_H3_DIT=<tier>/transformer \
//! MINIMAX_H3_TIER=q4 \
//!   cargo test -p mlx-gen-minimax-h3 --test quant_tiers_real -- --ignored --nocapture \
//!   --test-threads=1
//! ```
//!
//! # Why the DiT arrives separately from the snapshot root
//!
//! Only the DiT is tiered. The text encoder, both VAEs and the tokenizer are dense in every tier
//! and shared as a co-requisite, so a tiered install is the upstream root **plus** one redirected
//! component — [`mlx_gen_minimax_h3::model::DIT_COMPONENT`]. That is exactly the shape this harness
//! stages, so what it exercises is the real install topology rather than a flattened copy of it.
//!
//! # The gate is relative max-abs-diff, and that choice is not incidental
//!
//! Seven quality checks in this epic were blind because they used a norm, a cosine or a checksum.
//! A cosine is **scale-invariant** — a tier that uniformly halved every activation would score 1.0.
//! A norm averages a local catastrophe away across 5376 channels. A checksum answers a question
//! nobody asked (are the bytes equal) when the honest question is *how far apart are they*. So
//! every comparison here is `max|a−b| / max|ref|`, which is bounded by nothing and hides nothing.
//!
//! # What "coherent" means here, operationally
//!
//! A tier is not judged by eye. [`frame_statistics`] extracts per-frame mean/stddev and inter-frame
//! motion, and a tier passes when its frames are structured (non-degenerate stddev), temporally
//! coherent (bounded inter-frame delta) and close to the bf16 reference under the relative
//! max-abs-diff above. Mage's uniform-Q4 failure (sc-15071) rendered a *tiled texture* — high
//! stddev, plausible checksum, obviously wrong — which is why structure alone is not the test.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource,
};

use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT};

/// The geometry every tier is measured at — the epic's reference point (124 frames = 5.1667 s, the
/// lattice floor) so the rows are comparable with sc-17152's bf16 sweep.
const FRAMES: u32 = 124;
const WIDTH: u32 = 576;
const HEIGHT: u32 = 320;

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn snapshot() -> PathBuf {
    PathBuf::from(env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<upstream snapshot root>"))
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Per-frame structure and inter-frame motion — the "is this a picture" evidence.
struct FrameStats {
    mean: f64,
    stddev: f64,
    motion: f64,
}

fn frame_statistics(frames: &[mlx_gen::gen_core::Image]) -> Vec<FrameStats> {
    let mut out = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        let n = f.pixels.len() as f64;
        let mean = f.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        let var = f
            .pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n;
        // Mean absolute inter-frame delta: 0 for a frozen clip, huge for per-frame noise.
        let motion = if i == 0 {
            0.0
        } else {
            let prev = &frames[i - 1].pixels;
            f.pixels
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .sum::<f64>()
                / n
        };
        out.push(FrameStats {
            mean,
            stddev: var.sqrt(),
            motion,
        });
    }
    out
}

/// `max|a−b| / max|ref|` over the whole clip. Never a norm, never a cosine (see the module docs).
fn relative_max_abs_diff(a: &[mlx_gen::gen_core::Image], b: &[mlx_gen::gen_core::Image]) -> f64 {
    let mut worst = 0.0f64;
    let mut denom = 0.0f64;
    for (fa, fb) in a.iter().zip(b.iter()) {
        for (&pa, &pb) in fa.pixels.iter().zip(fb.pixels.iter()) {
            worst = worst.max((f64::from(pa) - f64::from(pb)).abs());
            denom = denom.max(f64::from(pa));
        }
    }
    if denom == 0.0 {
        f64::INFINITY
    } else {
        worst / denom
    }
}

/// One tier's render: the frames, the wall clock and the MLX peak.
struct TierRender {
    frames: Vec<mlx_gen::gen_core::Image>,
    wall_s: f64,
    peak_bytes: u64,
    steps_seen: u32,
}

fn render_tier(dit: Option<&std::path::Path>, quant: Option<Quant>, steps: u32) -> TierRender {
    let root = snapshot();
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    if let Some(d) = dit {
        spec = spec.with_component(DIT_COMPONENT, WeightsSource::Dir(d.to_path_buf()));
    }
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let model = load(&spec).unwrap_or_else(|e| panic!("load tier {quant:?}: {e}"));

    let req = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, \
                 seagulls calling"
            .into(),
        width: WIDTH,
        height: HEIGHT,
        frames: Some(FRAMES),
        steps: Some(steps),
        seed: Some(17_150),
        cancel: CancelFlag::default(),
        ..Default::default()
    };
    model.validate(&req).expect("the tier point must validate");

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    let mut steps_seen = 0u32;
    let out = model
        .generate(&req, &mut |p| {
            if let Progress::Step { .. } = p {
                steps_seen += 1;
            }
        })
        .unwrap_or_else(|e| panic!("generate tier {quant:?}: {e}"));
    let wall_s = started.elapsed().as_secs_f64();
    let peak_bytes = mlx_rs::memory::get_peak_memory() as u64;

    let (frames, audio) = match out {
        GenerationOutput::Video { frames, audio, .. } => (frames, audio),
        other => panic!("expected Video, got {other:?}"),
    };
    // Evidence the render RAN rather than the `#[ignore]` falling through and printing a
    // fabricated row.
    assert_eq!(frames.len(), FRAMES as usize, "decoded frame count");
    assert!(
        audio.is_some(),
        "MiniMax-H3 always produces a soundtrack; its absence means the audio half never ran"
    );
    assert_eq!(steps_seen, steps, "every denoise step must report progress");

    TierRender {
        frames,
        wall_s,
        peak_bytes,
        steps_seen,
    }
}

/// **The per-tier quality datum.** Renders the identical geometry at the staged tier and reports
/// wall clock, MLX peak and frame structure. With `MINIMAX_H3_BF16_DIT` also set, it renders the
/// dense reference too and gates the tier against it on relative max-abs-diff.
#[test]
#[ignore = "needs MINIMAX_H3_SNAPSHOT + MINIMAX_H3_DIT (a built tier's transformer/) + Metal"]
fn tier_renders_coherently_and_is_close_to_bf16() {
    let dit = PathBuf::from(env("MINIMAX_H3_DIT").expect("MINIMAX_H3_DIT=<tier>/transformer"));
    let tier = env("MINIMAX_H3_TIER").unwrap_or_else(|| "unknown".into());
    let steps: u32 = env("MINIMAX_H3_TIER_STEPS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    };

    let r = render_tier(Some(&dit), quant, steps);
    let stats = frame_statistics(&r.frames);
    let mean_sd = stats.iter().map(|s| s.stddev).sum::<f64>() / stats.len() as f64;
    let mean_motion =
        stats.iter().skip(1).map(|s| s.motion).sum::<f64>() / (stats.len() - 1) as f64;

    println!(
        "\n  tier {tier}: {WIDTH}x{HEIGHT} / {FRAMES} frames / {} steps\n  \
         wall {:.1}s, MLX peak {:.2} GB\n  \
         frame mean {:.1}, stddev {:.1} (mean over frames), inter-frame motion {:.2}",
        r.steps_seen,
        r.wall_s,
        gb(r.peak_bytes),
        stats.iter().map(|s| s.mean).sum::<f64>() / stats.len() as f64,
        mean_sd,
        mean_motion
    );

    // Structure: a tier that collapsed to a flat field or to noise is not "coherent" whatever its
    // distance from the reference.
    assert!(
        mean_sd > 8.0,
        "tier {tier} frames are nearly flat (stddev {mean_sd:.2}) — the render collapsed"
    );
    assert!(
        mean_motion > 0.05,
        "tier {tier} is a frozen clip (inter-frame motion {mean_motion:.4})"
    );
    assert!(
        mean_motion < 60.0,
        "tier {tier} is per-frame noise, not video (inter-frame motion {mean_motion:.2})"
    );

    if let Some(bf16_dit) = env("MINIMAX_H3_BF16_DIT") {
        let reference = render_tier(Some(std::path::Path::new(&bf16_dit)), None, steps);
        let rel = relative_max_abs_diff(&reference.frames, &r.frames);
        println!(
            "  vs bf16: relative max-abs-diff {rel:.4} (bf16 wall {:.1}s, peak {:.2} GB)",
            reference.wall_s,
            gb(reference.peak_bytes)
        );
        // A packed tier is a different model, not a numerically-equal one, so this bound is a
        // sanity ceiling rather than a parity claim: it catches a tier that renders something
        // unrelated (Mage's tiled texture scored far past this), not one that renders the same
        // scene slightly differently.
        assert!(
            rel < 0.85,
            "tier {tier} is not rendering the reference scene: relative max-abs-diff {rel:.4}"
        );
    }
}
