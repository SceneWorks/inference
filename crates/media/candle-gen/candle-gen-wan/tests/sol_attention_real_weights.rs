//! SW-6137 (SW-F287 S12) — the opt-in Sol-Attn sparse self-attention engine option on real
//! Wan2.2 T2V-A14B weights: a fixed-seed short clip through the T2V loader with
//! `Wan14bEngineOptions::default()` (FlashAttention) and with `sol_attention: Some(top_k(0.25))`.
//!
//! - the default engine takes no Sol-Attn route at all (zero sol calls and zero sol refusals);
//! - the sol engine takes every denoise **self**-attention (two CFG forwards x 40 blocks per step)
//!   with zero refusals, and leaves the text cross-attention on FlashAttention;
//! - the sol clip's PSNR against the flash clip and the s/step of both engines are printed, and the
//!   first / last frames of each are saved for review. PSNR is **recorded, not gated**: Sol-Attn
//!   changes every self-attention output (~0.2 rel-L2 per call at top-k 0.25 on real Wan Q/K/V), and
//!   a 40-block x CFG denoise amplifies that into a different sample (the SW-6137 campaign: even the
//!   all-routed INT8 run is 16 dB from flash, the sparse runs 8-10 dB), so a pixel floor would be
//!   arbitrary. The quality call is the owner's, from the saved frames and the campaign strips;
//! - the sol clip is not degenerate: every frame keeps a pixel standard deviation of at least
//!   [`MIN_FRAME_STD`] (a black / constant / NaN-collapsed decode fails).
//!
//! ```sh
//! WAN14B_SNAPSHOT=<snapshot>/q4 WAN_SOL_ARTIFACT_DIR=<out> \
//!   cargo test -p candle-gen-wan --release --features cuda,flash-attn,model-inference-proof \
//!   --test integration sol_attention_real_weights:: -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::fused_attention::{attention_path_counts, AttentionPathCounts};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};
use candle_gen_wan::wan14b::{load_t2v_14b_with_options, Wan14bEngineOptions};

/// The smallest per-frame pixel standard deviation (RGB8) a non-degenerate frame keeps.
const MIN_FRAME_STD: f64 = 8.0;

const PROMPT: &str =
    "a red fox trotting through a snowy birch forest at golden hour, soft falling snow, cinematic";

/// 480x832, 17 frames (5 latent frames: 7,800 tokens), 6 steps.
const WIDTH: u32 = 832;
const HEIGHT: u32 = 480;
const FRAMES: u32 = 17;
const STEPS: u32 = 6;
const BLOCKS: u64 = 40;

fn snapshot() -> PathBuf {
    std::env::var("WAN14B_SNAPSHOT")
        .map(PathBuf::from)
        .expect("set WAN14B_SNAPSHOT to a Wan2.2 T2V-A14B snapshot tier (e.g. .../q4)")
}

fn artifact_dir() -> PathBuf {
    let dir = std::env::var("WAN_SOL_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("wan_sol_attention");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Clip {
    frames: Vec<Image>,
    s_per_step: f64,
    counts: AttentionPathCounts,
}

fn render(options: Wan14bEngineOptions) -> Clip {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let generator = load_t2v_14b_with_options(&spec, options).expect("load wan2_2_t2v_14b");
    let request = GenerationRequest {
        prompt: PROMPT.into(),
        width: WIDTH,
        height: HEIGHT,
        frames: Some(FRAMES),
        count: 1,
        seed: Some(42),
        steps: Some(STEPS),
        ..Default::default()
    };
    let mut marks = Vec::new();
    let before = attention_path_counts();
    let out = generator
        .generate(&request, &mut |progress| {
            if let Progress::Step { .. } = progress {
                marks.push(Instant::now());
            }
        })
        .expect("generate");
    let counts = attention_path_counts().since(&before);
    let GenerationOutput::Video { frames, .. } = out else {
        panic!("expected video");
    };
    assert_eq!(frames.len(), FRAMES as usize);
    let s_per_step = match marks.as_slice() {
        [first, .., last] => (*last - *first).as_secs_f64() / (marks.len() - 1) as f64,
        _ => f64::NAN,
    };
    drop(generator);
    Clip {
        frames,
        s_per_step,
        counts,
    }
}

fn save(frame: &Image, name: &str) {
    let path = artifact_dir().join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

/// Pixel standard deviation of one RGB8 frame.
fn pixel_std(frame: &Image) -> f64 {
    let n = frame.pixels.len() as f64;
    let mean = frame.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    (frame
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// PSNR (dB) over every frame of two RGB8 clips.
fn psnr(a: &[Image], b: &[Image]) -> f64 {
    let (mut se, mut n) = (0f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.pixels.len(), y.pixels.len());
        se += x
            .pixels
            .iter()
            .zip(&y.pixels)
            .map(|(&p, &q)| (p as f64 - q as f64).powi(2))
            .sum::<f64>();
        n += x.pixels.len();
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn wan_t2v_a14b_sol_attention_smoke_against_flash() {
    let flash = render(Wan14bEngineOptions::default());
    assert_eq!(flash.counts.sol, 0, "{:?}", flash.counts);
    assert_eq!(flash.counts.sol_refusals.total(), 0, "{:?}", flash.counts);
    assert!(flash.counts.flash > 0, "{:?}", flash.counts);
    let sol = render(Wan14bEngineOptions {
        sol_attention: Some(candle_gen::SolConfig::top_k(0.25)),
    });
    let self_attention = STEPS as u64 * 2 * BLOCKS;
    assert_eq!(sol.counts.sol, self_attention, "{:?}", sol.counts);
    assert_eq!(sol.counts.sol_refusals.total(), 0, "{:?}", sol.counts);
    assert_eq!(
        sol.counts.flash, self_attention,
        "cross attention: {:?}",
        sol.counts
    );
    let db = psnr(&sol.frames, &flash.frames);
    for (clip, name) in [(&flash, "flash"), (&sol, "sol_topk_0.25")] {
        save(&clip.frames[0], &format!("{name}_frame000"));
        save(clip.frames.last().unwrap(), &format!("{name}_frame016"));
    }
    eprintln!(
        "[wan sol] {WIDTH}x{HEIGHT}x{FRAMES} {STEPS} steps: flash {:.3} s/step, sol top-k 0.25 \
         {:.3} s/step, PSNR sol vs flash {db:.2} dB (recorded, not gated)",
        flash.s_per_step, sol.s_per_step
    );
    assert!(!db.is_nan(), "PSNR {db}");
    for (i, frame) in sol.frames.iter().enumerate() {
        let std = pixel_std(frame);
        assert!(
            std >= MIN_FRAME_STD,
            "sol frame {i} is degenerate: pixel std {std}"
        );
    }
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED sol_attention_real_weights::wan_t2v_a14b_sol_attention_smoke_against_flash"
    );
}
