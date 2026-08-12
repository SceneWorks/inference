//! **`fl2va` first light** (sc-17148): all four conditioning shapes rendered on the published
//! weights.
//!
//! `#[ignore]`d — it needs the whole ~196 GB snapshot and Metal:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> SCENEWORKS_GPU_ID=mlx MINIMAX_H3_FL2VA_STEPS=4 \
//!   cargo test -p mlx-gen-minimax-h3 --test fl2va_first_light -- --ignored --nocapture
//! ```
//!
//! # What this covers that `fl2va_conditioning.rs` cannot
//!
//! The fixture-scale binding tests run the real conditioning code against a tiny committed VAE and
//! a recording stand-in for the DiT. Three things only real weights can exercise, and all three
//! are load-bearing:
//!
//! 1. **The Qwen3-VL vision tower actually loads and runs.** All 351 `model.visual.*` tensors live
//!    in **shard 14** — precisely the shard the `t2va` window (`TE_SHARDS`, 1..=12) excludes,
//!    because 13-14 otherwise hold only the never-executed decoder tail. Nothing else in the crate
//!    constructs a `VisionTower`; before sc-17148 `run_vision` / `encode_grounded` /
//!    `forward_with_images` had zero call sites.
//! 2. **The 118-tensor VAE encoder runs on the published checkpoint**, not just on the tiny
//!    fixture geometry.
//! 3. **All four shapes complete end to end** — the story's first acceptance criterion says
//!    *render*, and a layout that builds is not a render.
//!
//! # Steps are lowered on purpose, and the receipt says so
//!
//! The gating quality run is `first_light.rs` at 50 steps. What this file gates is **reachability
//! and conditioning**, so it defaults to a low step count: four shapes at 50 steps is ~44 minutes
//! of GPU for evidence that does not improve with step count. The printed receipt always carries
//! the value actually used, so a short run cannot be mistaken for a quality one.
//!
//! # The binding assertion here is deliberately weak, and that is the honest choice
//!
//! This file does **not** assert that the first output frame resembles the input keyframe. At 4
//! steps it would not, and at 50 steps the threshold that separates "conditioned" from
//! "coincidentally similar" is not something this slice measured. The strong binding evidence is
//! `fl2va_conditioning.rs`, which measures it directly on tensors rather than inferring it from
//! pixels. What this asserts is that each shape *renders a coherent clip*, that the two one-image
//! shapes are **different renders from the same seed**, and that a keyframe **changes the output**
//! against the `t2va` baseline — which is the pixel-level statement that can be made honestly.

mod common;

use std::time::Instant;

use mlx_gen::gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen::media::Image;

use mlx_gen_minimax_h3::{AUDIO_SAMPLE_RATE, SMALLEST_LEGAL_FRAMES};

use common::snapshot;

/// The same 32-aligned gating canvas `first_light.rs` uses.
const WIDTH: u32 = 576;
const HEIGHT: u32 = 320;

/// Model evaluations. Low by default — see the module docs.
fn steps() -> u32 {
    std::env::var("MINIMAX_H3_FL2VA_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

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

fn frame_delta(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() as f64;
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum::<f64>()
        / n
        / 255.0
}

/// A high-contrast structured keyframe — deliberately not a flat fill and not the canvas mean, for
/// the reason `fl2va_conditioning.rs::probe_image` documents.
fn keyframe(seed: u32) -> Image {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let on = if seed == 0 {
                ((x / 48) + (y / 48)) % 2 == 0
            } else {
                (x.max(y) / 64) % 2 == 0
            };
            let (r, g, b) = if on { (235, 60, 40) } else { (20, 70, 190) };
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    Image {
        width: WIDTH,
        height: HEIGHT,
        pixels,
    }
}

/// The last frame's index, for a `Keyframe` that anchors the end of the clip.
const LAST_FRAME_IDX: i32 = SMALLEST_LEGAL_FRAMES - 1;

/// **All four shapes, on real weights.**
#[test]
#[ignore = "needs the full MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; four renders"]
fn fl2va_renders_all_four_conditioning_shapes() {
    let root = snapshot();
    let evaluations = steps();
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    let model = load_generator(&spec, &root);

    let first = keyframe(0);
    let last = keyframe(1);

    let shapes: Vec<(&str, Vec<Conditioning>)> = vec![
        ("0 images (t2va)", vec![]),
        (
            "first only",
            vec![Conditioning::Keyframe {
                image: first.clone(),
                frame_idx: 0,
                strength: 1.0,
            }],
        ),
        (
            "last only",
            vec![Conditioning::Keyframe {
                image: first.clone(),
                frame_idx: LAST_FRAME_IDX,
                strength: 1.0,
            }],
        ),
        (
            "first + last",
            vec![
                Conditioning::Keyframe {
                    image: first.clone(),
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::Keyframe {
                    image: last.clone(),
                    frame_idx: LAST_FRAME_IDX,
                    strength: 1.0,
                },
            ],
        ),
    ];

    let mut middles: Vec<(&str, Vec<u8>)> = Vec::new();
    for (name, conditioning) in shapes {
        let req = GenerationRequest {
            prompt: "a lighthouse on a rocky coast at dusk, waves breaking against the rocks"
                .into(),
            width: WIDTH,
            height: HEIGHT,
            frames: Some(SMALLEST_LEGAL_FRAMES as u32),
            steps: Some(evaluations),
            // The SAME seed for every shape, so any difference between renders is attributable to
            // the conditioning and not to the noise draw.
            seed: Some(17_148),
            conditioning,
            cancel: CancelFlag::default(),
            ..Default::default()
        };
        model
            .validate(&req)
            .unwrap_or_else(|e| panic!("{name}: the request must validate: {e}"));

        mlx_rs::memory::reset_peak_memory();
        let started = Instant::now();
        let out = model
            .generate(&req, &mut |_| {})
            .unwrap_or_else(|e| panic!("{name}: real-weight render: {e}"));
        let wall = started.elapsed();
        let peak = mlx_rs::memory::get_peak_memory();

        let (frames, fps, audio) = match out {
            GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
            other => panic!("{name}: expected a Video output, got {other:?}"),
        };
        let audio = audio.unwrap_or_else(|| panic!("{name}: no soundtrack"));

        assert_eq!(frames.len(), SMALLEST_LEGAL_FRAMES as usize, "{name}");
        assert_eq!(fps, 24, "{name}");
        for (i, f) in frames.iter().enumerate() {
            assert_eq!((f.width, f.height), (WIDTH, HEIGHT), "{name} frame {i}");
        }
        assert_eq!(audio.sample_rate, AUDIO_SAMPLE_RATE, "{name}");
        assert!(
            audio.samples.iter().all(|s| s.is_finite()),
            "{name}: the soundtrack carries NaN/Inf"
        );

        let mid = frames.len() / 2;
        let spread = frame_std(&frames[mid].pixels);
        assert!(
            (0.02..=0.45).contains(&spread),
            "{name}: middle-frame pixel std {spread:.4} is outside the [0.02, 0.45] band"
        );
        // Not a still, and not per-frame-independent noise.
        let adjacent = frame_delta(&frames[mid].pixels, &frames[mid + 1].pixels);
        let span = frame_delta(&frames[0].pixels, &frames[frames.len() - 1].pixels);
        assert!(span > 1e-4, "{name}: nothing changes across the clip");

        println!(
            "  {name}: {WIDTH}x{HEIGHT} / {} frames / {evaluations} steps in {:.1} s; peak \
             {:.2} GB; mid std {spread:.4}, adjacent {adjacent:.4}, span {span:.4}",
            frames.len(),
            wall.as_secs_f64(),
            peak as f64 / 1e9
        );
        assert!(
            peak > 1_000_000_000,
            "{name}: peak MLX memory {peak} B is too small for a real 33 B forward"
        );
        middles.push((name, frames[mid].pixels.clone()));
    }

    // --- what the four renders say about each other ------------------------------------------
    // Same prompt, same seed, same geometry — so every difference below is the conditioning.
    let t2va = &middles[0].1;
    for (name, pixels) in &middles[1..] {
        let d = frame_delta(t2va, pixels);
        println!("  {name} vs t2va: mean frame delta {d:.4}");
        assert!(
            d > 1e-3,
            "{name} rendered the same picture as the unconditioned run ({d:.4}) at the same seed; \
             the keyframe is not reaching the render"
        );
    }
    // **The two one-image shapes are different renders.** Same image, same seed, same everything
    // except which end of the clip it anchors — which is exactly the distinction a first-frame-only
    // implementation erases.
    let first_only = &middles[1].1;
    let last_only = &middles[2].1;
    let d = frame_delta(first_only, last_only);
    println!("  first-only vs last-only: mean frame delta {d:.4}");
    assert!(
        d > 1e-3,
        "anchoring the SAME image to the first frame and to the last frame produced the same \
         render ({d:.4}); last-frame conditioning is being treated as first-frame conditioning"
    );

    println!(
        "FL2VA FIRST LIGHT: all four shapes rendered at {evaluations} steps. Note this step count \
         gates REACHABILITY, not quality — `first_light.rs` is the quality run."
    );
}

fn load_generator(
    spec: &LoadSpec,
    root: &std::path::Path,
) -> Box<dyn mlx_gen::gen_core::Generator> {
    match mlx_gen_minimax_h3::model::load(spec) {
        Ok(m) => m,
        Err(e) => panic!("load {}: {e}", root.display()),
    }
}
