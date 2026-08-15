//! sc-19120 — **the generate peak, attributed per stage**, on real weights.
//!
//! `#[ignore]`d: needs the upstream snapshot, a DiT tier, a text-encoder component and Metal.
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<upstream snapshot root> \
//! MINIMAX_H3_DIT=<tier>/transformer \
//! MINIMAX_H3_TE=<text_encoder dir> \
//!   cargo test -p mlx-gen-minimax-h3 --test te_tier_generate_stages -- --ignored --nocapture \
//!   --test-threads=1
//! ```
//!
//! # Why a process-wide peak was the wrong instrument
//!
//! Every H3 tier measured an identical ~53 GB generate peak, and that number was explained twice —
//! "activation-dominated", then "the DiT's" — before it was measured to be neither. The
//! conditioning stage runs **first** and is the tallest thing in the process, so `reset_peak_memory()`
//! before `generate` yields one number that no later stage can move. bf16 denoise residency is
//! 40.43 GB against that 53.07 GB mark: a **12.64 GB window** inside which any DiT-side transient
//! moves invisibly.
//!
//! This test resets the peak at every stage boundary the engine now emits, so each stage reports
//! its own high-water and none can be credited to another. The boundaries are
//! [`Progress::Loading(LoadPhase::TextEncoder)`], [`Progress::Loading(LoadPhase::Renderer)`], the
//! first [`Progress::Step`] and [`Progress::Decoding`] — public callback events, so what is
//! measured is the shipped `generate`, not a re-staged copy of it.
//!
//! The AdaLN precompute sits inside the renderer segment (it runs between the DiT map and step 0);
//! `tests/adaln_evict_real_weights.rs` separates it from the denoise directly.
//!
//! # sc-17151: the table became assertions
//!
//! As shipped by sc-19120 this file *printed* the per-stage attribution and asserted only that the
//! stage boundaries existed, the frame count matched and a soundtrack came back. A printed table is
//! not an assertion: every one of those passed on a build where the conditioning phase kept its
//! 53 GB text encoder through the denoise and the decode. Measured, with `release((te, w))` in
//! `model::encode_prompt` replaced by a leak: the process peak went 53.06 GB → 91.82 GB, the
//! conditioning stage closed holding 50.32 GB instead of 0.00 — and **all 481 of the crate's
//! non-ignored tests still passed**.
//!
//! The gates below are what fails on that. They are ratios of each stage's active-at-close against
//! that stage's own peak, so they hold for a quantized tier as well as for the dense components,
//! and both duration extremes are rendered because sc-17151's acceptance names both.
//! `tests/staged_residency.rs` gates the same property directly on the components, at full 66 GB
//! scale and without a render.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mlx_rs::memory::{get_active_memory, get_peak_memory, reset_peak_memory};

use mlx_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadPhase, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT, TEXT_ENCODER_COMPONENT};
use mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS;

/// The canvas and step count are the cheapest that still make each segment a real segment; quality
/// is not the question here, attribution is.
///
/// The duration axis is **derived** from the model's own [`LEGAL_FRAME_COUNTS`] rather than
/// tabulated: the smallest legal clip is 124 frames (the released checkpoint hardcodes a 5 s
/// minimum, so there is no shorter clip and no image lane) and the largest is 345. sc-17151's
/// acceptance is that the staged handoff holds at both ends, so both are rendered — and if a future
/// snapshot moves either bound, these follow it instead of quietly testing the old one.
const WIDTH: u32 = 384;
const HEIGHT: u32 = 224;
const STEPS: u32 = 4;

/// Peak a heavy stage must exceed before its ratios mean anything. MLX materializes a mapped tensor
/// on first use, so "the component is loaded" and "the component is resident" are different states
/// and a bare 66 GB load reads 33 KB; 1 GB is far below the smallest legal H3 stage and far above
/// anything a no-op could reach (sc-17151).
const FLOOR: usize = 1 << 30;

/// A stage may close holding at most `1/HANDOFF_RATIO` of its own peak. A stage that kept its
/// component closes at ~1/1 of it.
const HANDOFF_RATIO: usize = 4;

fn env(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

/// Close the stage named `open`, recording its peak, and open the next one.
fn rotate(
    stages: &mut BTreeMap<usize, (&'static str, usize, usize)>,
    open: &mut (usize, &'static str),
    next: &'static str,
) {
    let peak = get_peak_memory();
    let active = get_active_memory();
    stages.insert(open.0, (open.1, peak, active));
    *open = (open.0 + 1, next);
    reset_peak_memory();
}

/// The smallest legal clip — 124 frames, 5 s at the model's 24 fps.
#[test]
#[ignore = "needs the upstream snapshot, a DiT tier, a text-encoder component and Metal"]
fn every_stage_reports_its_own_high_water() {
    let frames = *LEGAL_FRAME_COUNTS.first().expect("legal frame counts") as u32;
    stage_the_render(frames);
}

/// **The same handoff at the largest supported duration** (sc-17151's third acceptance clause).
///
/// Duration moves the *activation* transient, not the component sizes, so a staging bug that the
/// short clip's headroom absorbs can still show here — the denoise stage is ~2.8x the token count
/// and the decode stage ~2.8x the frames. The gates are the identical ones; only the geometry
/// changes.
#[test]
#[ignore = "renders the LARGEST legal clip (345 frames) on the upstream snapshot and Metal"]
fn the_staged_handoff_holds_at_the_largest_supported_duration() {
    let frames = *LEGAL_FRAME_COUNTS.last().expect("legal frame counts") as u32;
    stage_the_render(frames);
}

fn stage_the_render(frames: u32) {
    let root = env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<upstream snapshot root>");
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    if let Some(dit) = env("MINIMAX_H3_DIT") {
        spec = spec.with_component(DIT_COMPONENT, WeightsSource::Dir(dit));
    }
    if let Some(te) = env("MINIMAX_H3_TE") {
        spec = spec.with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(te));
    }

    let generator = load(&spec).expect("load");

    let req = GenerationRequest {
        prompt: "a slow pan across a rainy street at night, neon reflections".into(),
        width: WIDTH,
        height: HEIGHT,
        frames: Some(frames),
        steps: Some(STEPS),
        seed: Some(7),
        cancel: CancelFlag::default(),
        ..Default::default()
    };

    // stage index → (name, peak, active-at-close)
    let mut stages: BTreeMap<usize, (&'static str, usize, usize)> = BTreeMap::new();
    let mut open = (0usize, "pre-generate");
    let mut first_step_seen = false;

    // The production drain, not a copy of it: one `clear_cache` reports success while buffers sit
    // in the allocator's cache, and a baseline taken over a dirty allocator shifts every stage's
    // reading below.
    mlx_gen::residency::drain_allocator_cache();
    reset_peak_memory();
    let started = std::time::Instant::now();

    let mut on_progress = |p: Progress| match p {
        Progress::Loading(LoadPhase::TextEncoder) => rotate(&mut stages, &mut open, "conditioning"),
        Progress::Loading(LoadPhase::Renderer) => {
            rotate(&mut stages, &mut open, "dit-load + adaln-precompute")
        }
        Progress::Step { .. } if !first_step_seen => {
            first_step_seen = true;
            rotate(&mut stages, &mut open, "denoise");
        }
        Progress::Decoding => rotate(&mut stages, &mut open, "decode"),
        _ => {}
    };

    let out = generator
        .generate(&req, &mut on_progress)
        .expect("generate");
    // Close the final stage.
    stages.insert(open.0, (open.1, get_peak_memory(), get_active_memory()));

    let elapsed = started.elapsed().as_secs_f64();
    println!("── generate, per stage ────────────────────────────────────────");
    println!("  text encoder   {:?}", env("MINIMAX_H3_TE"));
    println!("  dit            {:?}", env("MINIMAX_H3_DIT"));
    println!("  {WIDTH}x{HEIGHT} / {frames} frames / {STEPS} steps  in {elapsed:.1}s");
    println!("  {:<28} {:>10} {:>10}", "stage", "peak GB", "active GB");
    let mut process_peak = 0usize;
    let mut conditioning = 0usize;
    for (name, peak, active) in stages.values() {
        println!("  {name:<28} {:>10.2} {:>10.2}", gb(*peak), gb(*active));
        process_peak = process_peak.max(*peak);
        if *name == "conditioning" {
            conditioning = *peak;
        }
    }
    println!(
        "  {:<28} {:>10.2}",
        "process (max of stages)",
        gb(process_peak)
    );
    let (decoded, audio) = match out {
        GenerationOutput::Video { frames, audio, .. } => (frames, audio),
        other => panic!("expected Video, got {other:?}"),
    };
    println!(
        "  frames {} / soundtrack {}",
        decoded.len(),
        audio.is_some()
    );

    // Every stage must have been visited. A missing boundary means the engine stopped emitting one
    // and the attribution silently collapses back to a single number.
    let names: Vec<&str> = stages.values().map(|(n, ..)| *n).collect();
    for want in [
        "conditioning",
        "dit-load + adaln-precompute",
        "denoise",
        "decode",
    ] {
        assert!(names.contains(&want), "no {want} stage boundary: {names:?}");
    }
    assert!(conditioning > 0, "the conditioning stage reported no peak");
    // Evidence the render RAN rather than the `#[ignore]` falling through with a fabricated table.
    assert_eq!(decoded.len(), frames as usize, "decoded frame count");
    assert!(audio.is_some(), "H3 always produces a soundtrack");

    // --- the residency tripwire (sc-17151) ---------------------------------------------------
    // Everything above this line describes the table; nothing above it fails when a phase starts
    // holding the previous phase's weights, which is the defect the staging exists to prevent.
    let stage = |want: &str| -> (usize, usize) {
        stages
            .values()
            .find(|(name, ..)| *name == want)
            .map(|(_, peak, active)| (*peak, *active))
            .unwrap_or_else(|| panic!("no {want} stage"))
    };
    // Materialization guard, first: under lazy mmap a 66 GB map leaves peak at 33 KB, and every
    // ratio below would then be comparing noise to noise. Each heavy stage must have put real
    // bytes on the device.
    for want in ["conditioning", "dit-load + adaln-precompute", "denoise"] {
        let (peak, _) = stage(want);
        assert!(
            peak > FLOOR,
            "the {want} stage peaked at {:.2} GB, under the {:.2} GB floor a real forward on this \
             model cannot miss — MLX materializes lazily, so a stage that allocated nothing reads \
             as a pass on every other gate here",
            gb(peak),
            gb(FLOOR)
        );
    }

    // **The handoff gates.** Each heavy stage closes at the boundary *after* its own component was
    // released — `encode_prompt` forces the context, drops the 66.7 GB encoder and drains before
    // `Progress::Loading(Renderer)`; `release((model, video_rows, audio_rows))` runs before
    // `Progress::Decoding`. So each stage's active-at-close must be a small fraction of what it had
    // resident. A stage that carried its component into the next one closes at roughly its own
    // peak, which is the regression this whole file existed to *describe* and now fails on.
    //
    // The bound is a ratio against the stage's own peak rather than an absolute size, so it holds
    // for a quantized tier as well as for the dense bf16 components.
    for (name, keeps) in [("conditioning", "text encoder"), ("denoise", "DiT")] {
        let (peak, active) = stage(name);
        assert!(
            active * HANDOFF_RATIO < peak,
            "the {name} stage closed holding {:.2} GB against its own {:.2} GB peak — more than \
             1/{HANDOFF_RATIO} of what it had resident survived the boundary, so the {keeps} was \
             not released before the next phase loaded",
            gb(active),
            gb(peak)
        );
    }
}
