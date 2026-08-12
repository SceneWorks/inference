//! **First light** (sc-17147): a real MiniMax-H3 `t2va` render on the published weights.
//!
//! `#[ignore]`d — it needs the whole ~196 GB snapshot and Metal. Point `MINIMAX_H3_SNAPSHOT` at the
//! snapshot root and run:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> SCENEWORKS_GPU_ID=mlx \
//!   cargo test -p mlx-gen-minimax-h3 --test first_light -- --ignored --nocapture
//! ```
//!
//! **A skipped run must not look like a passing one.** `common::snapshot()` asserts rather than
//! returning `None`, and every assertion below is on *evidence the render happened* — the decoded
//! frame count, a middle-frame pixel std inside a band, temporal coherence between adjacent frames,
//! and a non-silent soundtrack whose length matches the picture — not on "no panic".
//!
//! # What the two picture assertions each rule out
//!
//! Neither is sufficient alone, which is why both run:
//!
//! | assertion | rules out | does NOT rule out |
//! |---|---|---|
//! | middle-frame pixel std in `[0.02, 0.45]` | a constant / black / saturated frame — a stubbed decode, an all-zero load, a diverged denoise | **noise**: white noise has a perfectly healthy std |
//! | mean adjacent-frame difference well below that std | noise, and a per-frame-independent decode | a still image (which is *coherent*, just boring) |
//!
//! Together they say "the frames are structured **and** they are the same scene", which is what
//! "produces a coherent clip" means and what a pixel-std datum alone cannot say.

mod common;

use std::time::Instant;

use mlx_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

use mlx_gen_minimax_h3::model::{load, DEFAULT_STEPS};
use mlx_gen_minimax_h3::{AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE, SMALLEST_LEGAL_FRAMES};

use common::snapshot;

/// The gating canvas: the smallest sensible 32-aligned 16:9-ish frame. The spike measured this
/// geometry at ~21-24 s/step; peak barely moves with canvas, so a bigger one buys nothing but
/// wall-clock.
const WIDTH: u32 = 576;
const HEIGHT: u32 = 320;

/// Model evaluations. `MINIMAX_H3_FIRST_LIGHT_STEPS` lowers it for iteration; the receipt always
/// prints the value actually used, so a short run cannot be mistaken for the gating one.
fn steps() -> u32 {
    std::env::var("MINIMAX_H3_FIRST_LIGHT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_STEPS)
}

/// Mean absolute pixel value spread of one frame — the "is there a picture here" statistic.
fn frame_std(pixels: &[u8]) -> f64 {
    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
    let var = pixels
        .iter()
        .map(|&p| {
            let d = f64::from(p) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (var / (255.0 * 255.0)).sqrt()
}

/// Mean absolute difference between two frames, on the same `[0, 1]` scale as [`frame_std`].
fn frame_delta(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() as f64;
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum::<f64>()
        / n
        / 255.0
}

/// Root-mean-square of an interleaved PCM buffer.
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt()
}

/// **The render.** Prompt in, 124 frames of picture and 5.1667 s of stereo sound out.
#[test]
#[ignore = "needs the full MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; measured 52.81 GB peak / 655 s at 576x320 x 124 frames x 50 steps"]
fn first_light_renders_video_and_audio_at_the_smallest_geometry() {
    let root = snapshot();
    let evaluations = steps();

    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    let model = match load(&spec) {
        Ok(m) => m,
        Err(e) => panic!("load {}: {e}", root.display()),
    };

    let req = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, \
                 seagulls calling"
            .into(),
        width: WIDTH,
        height: HEIGHT,
        frames: Some(SMALLEST_LEGAL_FRAMES as u32),
        steps: Some(evaluations),
        seed: Some(17_147),
        cancel: CancelFlag::default(),
        ..Default::default()
    };
    model.validate(&req).expect("the request must validate");

    // MLX's peak counter is process-wide and cumulative; reset it so what this reports is THIS
    // render's peak and not whatever the test binary did before it.
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    let mut observed: Vec<u32> = Vec::new();
    let mut decoding = false;
    let out = model
        .generate(&req, &mut |p| match p {
            Progress::Step { current, total } => {
                assert!(current >= 1 && current <= total, "{current}/{total}");
                observed.push(current);
            }
            Progress::Decoding => decoding = true,
            Progress::Loading(_) => {}
        })
        .expect("real-weight render");
    let wall = started.elapsed();
    let peak_bytes = mlx_rs::memory::get_peak_memory();

    let (frames, fps, audio) = match out {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        other => panic!("expected a Video output, got {other:?}"),
    };
    let audio = audio.expect("MiniMax-H3 always produces a soundtrack");

    // --- the picture -------------------------------------------------------------------------
    assert_eq!(frames.len(), SMALLEST_LEGAL_FRAMES as usize);
    assert_eq!(fps, 24);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (WIDTH, HEIGHT), "frame {i} size");
        assert_eq!(
            f.pixels.len(),
            (WIDTH * HEIGHT * 3) as usize,
            "frame {i} is not RGB8"
        );
    }
    assert_eq!(
        observed,
        (1..=evaluations).collect::<Vec<_>>(),
        "one progress tick per evaluation, in order"
    );
    assert!(decoding, "the decode phase must be reported");

    let middle = &frames[frames.len() / 2];
    let spread = frame_std(&middle.pixels);
    // The band. The floor rejects a constant/black/stub frame; the ceiling rejects a saturated or
    // diverged one (a uniform-random frame sits at 0.289, so the ceiling alone would not catch
    // noise — the coherence check below is what does).
    assert!(
        (0.02..=0.45).contains(&spread),
        "middle-frame pixel std {spread:.4} is outside the [0.02, 0.45] band; the clip is \
         constant, saturated or diverged"
    );

    // Temporal coherence: consecutive frames of one scene differ by far less than the picture's own
    // spread. Independent per-frame noise would put these at the same order of magnitude.
    let mid = frames.len() / 2;
    let delta = frame_delta(&frames[mid].pixels, &frames[mid + 1].pixels);
    assert!(
        delta < spread * 0.5,
        "adjacent frames differ by {delta:.4} against a within-frame spread of {spread:.4}; this \
         is not one coherent scene"
    );
    // ...and the clip is not a still: something has to move over 124 frames.
    let span = frame_delta(&frames[0].pixels, &frames[frames.len() - 1].pixels);
    assert!(span > 1e-4, "nothing changes across the clip ({span:.6})");

    // --- the soundtrack ----------------------------------------------------------------------
    assert_eq!(audio.sample_rate, AUDIO_SAMPLE_RATE, "32 kHz");
    assert_eq!(audio.channels, AUDIO_OUTPUT_CHANNELS, "stereo");
    assert!(audio.stems.is_empty(), "this model emits a mix, not stems");
    let level = rms(&audio.samples);
    assert!(
        level > 1e-4,
        "the soundtrack is silent (rms {level:.3e}); the audio VAE did not really run"
    );
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "the soundtrack carries NaN/Inf"
    );

    let per_channel = audio.samples.len() / usize::from(audio.channels);
    let audio_seconds = per_channel as f64 / f64::from(audio.sample_rate);
    let video_seconds = f64::from(SMALLEST_LEGAL_FRAMES) / f64::from(fps);
    let residual = audio_seconds - video_seconds;
    // One audio token is 25 ms; the delivered track is fitted to the picture, so this is far
    // tighter than that — a single sample of rounding.
    assert!(
        residual.abs() < 0.025,
        "the soundtrack is {audio_seconds:.5} s against {video_seconds:.5} s of picture — \
         {residual:+.5} s, more than one 25 ms audio token"
    );

    println!(
        "FIRST LIGHT: {WIDTH}x{HEIGHT} / {} frames / {evaluations} steps in {:.1} s ({:.1} s per \
         step); peak {:.2} GB.\n  middle-frame pixel std {spread:.4} (band [0.02, 0.45]); adjacent \
         frames differ by {delta:.4}, first-to-last by {span:.4}.\n  audio: {} Hz, {} ch, {} \
         samples/ch = {audio_seconds:.5} s against {video_seconds:.5} s of picture ({residual:+.5} \
         s residual), rms {level:.4}.",
        frames.len(),
        wall.as_secs_f64(),
        wall.as_secs_f64() / f64::from(evaluations),
        peak_bytes as f64 / 1e9,
        audio.sample_rate,
        audio.channels,
        per_channel,
    );
    // The receipt has to be evidence, not decoration: a run that never reached the GPU cannot have
    // moved the peak counter past the weights it had to map.
    assert!(
        peak_bytes > 1_000_000_000,
        "peak MLX memory {peak_bytes} B is too small for a real 33 B forward"
    );
}

/// **Off-lattice geometry is refused at the contract boundary, on the real loaded model.**
///
/// The unit tests cover the arithmetic; this covers the wiring — that the geometry gate is reached
/// through `Generator::validate` on a genuinely loaded provider, so SceneWorks' upstream
/// normalization cannot route around it. Reads no weights beyond the load-time probes.
#[test]
#[ignore = "needs the full MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn a_loaded_model_refuses_off_lattice_geometry() {
    let root = snapshot();
    let model = match load(&LoadSpec::new(WeightsSource::Dir(root))) {
        Ok(m) => m,
        Err(e) => panic!("load: {e}"),
    };
    let base = GenerationRequest {
        prompt: "a coastal scene".into(),
        width: WIDTH,
        height: HEIGHT,
        frames: Some(SMALLEST_LEGAL_FRAMES as u32),
        ..Default::default()
    };
    model.validate(&base).expect("the lattice point validates");

    for (label, req) in [
        (
            "125 frames (17n+5 is the lattice, not 4k+1)",
            GenerationRequest {
                frames: Some(125),
                ..base.clone()
            },
        ),
        (
            "129 frames (a 4k+1 point)",
            GenerationRequest {
                frames: Some(129),
                ..base.clone()
            },
        ),
        (
            "a 16-aligned but not 32-aligned canvas",
            GenerationRequest {
                width: 592,
                ..base.clone()
            },
        ),
        (
            "a 400-frame request, past the 15 s ceiling",
            GenerationRequest {
                frames: Some(400),
                ..base.clone()
            },
        ),
    ] {
        let err = model
            .validate(&req)
            .expect_err(&format!("{label} must be refused"));
        let text = err.to_string();
        assert!(
            text.contains("17n + 5") || text.contains("multiple of 32") || text.contains("outside"),
            "{label}: unexpected error {text}"
        );
        println!("  refused {label}: {text}");
    }
}
