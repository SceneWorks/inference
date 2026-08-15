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

use std::collections::BTreeMap;
use std::path::PathBuf;

use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};

use mlx_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadPhase, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT, TEXT_ENCODER_COMPONENT};

/// The smallest render that still visits every stage. 124 frames is the model's own floor — the
/// released checkpoint hardcodes a 5 s minimum duration, so there is no shorter legal clip and no
/// image lane. The canvas and step count are then the cheapest that still make each segment a real
/// segment; quality is not the question here, attribution is.
const FRAMES: u32 = 124;
const WIDTH: u32 = 384;
const HEIGHT: u32 = 224;
const STEPS: u32 = 4;

fn env(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

/// Retry until *active* stops falling — one `clear_cache` reports success while buffers sit in the
/// allocator's cache.
fn drain() {
    let mut previous = usize::MAX;
    for _ in 0..8 {
        clear_cache();
        let active = get_active_memory();
        if active >= previous {
            break;
        }
        previous = active;
    }
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

#[test]
#[ignore = "needs the upstream snapshot, a DiT tier, a text-encoder component and Metal"]
fn every_stage_reports_its_own_high_water() {
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
        frames: Some(FRAMES),
        steps: Some(STEPS),
        seed: Some(7),
        cancel: CancelFlag::default(),
        ..Default::default()
    };

    // stage index → (name, peak, active-at-close)
    let mut stages: BTreeMap<usize, (&'static str, usize, usize)> = BTreeMap::new();
    let mut open = (0usize, "pre-generate");
    let mut first_step_seen = false;

    drain();
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
    println!("  {WIDTH}x{HEIGHT} / {FRAMES} frames / {STEPS} steps  in {elapsed:.1}s");
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
    let (frames, audio) = match out {
        GenerationOutput::Video { frames, audio, .. } => (frames, audio),
        other => panic!("expected Video, got {other:?}"),
    };
    println!("  frames {} / soundtrack {}", frames.len(), audio.is_some());

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
    assert_eq!(frames.len(), FRAMES as usize, "decoded frame count");
    assert!(audio.is_some(), "H3 always produces a soundtrack");
}
