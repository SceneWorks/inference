//! sc-18662 — **the rung-4 REQUEST peak, measured on the shipped `generate`**, on real weights.
//!
//! `#[ignore]`d: needs a staged DiT tier, a text-encoder component (or the upstream snapshot) and
//! Metal.
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root with vae/, audio_vae/, FL2VA/, tokenizer/> \
//! MINIMAX_H3_DIT=<tier>/transformer \
//! MINIMAX_H3_TE=<text_encoder dir> \
//!   cargo test -p mlx-gen-minimax-h3 --test streamed_generate_real -- --ignored --nocapture \
//!   --test-threads=1
//! ```
//!
//! # Why this file exists when `block_window_real.rs` already measured both arms
//!
//! The per-arm cells measure each *phase* in its own process window. The SceneWorks survey's
//! `requestPeak` axis asks a different question — the high-water of one whole request under the
//! rung-4 composition — and its own note says a static proxy will not do: rung 4 is only ever
//! selected in composition, and the phase envelope's `max(TE, DiT, decode)` arithmetic is a
//! prediction until a request actually runs through the deferred loaders. Until this file nothing
//! did: `generate_impl` built both transformers resident on every route, so the rung was
//! *declared* and its loaders *measured* while remaining unreachable from a request — the exact
//! declaration-is-not-reachability defect this epic keeps refusing.
//!
//! So this drives the **shipped `generate`** — not a re-staged copy of its phases — with the
//! streamed selection on the request (`stream_transformer_blocks`, `TransformerComponent::Both`,
//! window 1), attributes the peak per stage exactly the way `te_tier_generate_stages.rs` does, and
//! asserts on the max.
//!
//! # Peaks, not wall clock
//!
//! This Mac is also the `nax-macos` CI runner. Every asserted number is an MLX peak-memory figure
//! (a per-process allocator high-water mark); wall clock is printed for context and never asserted.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use mlx_rs::memory::{get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory};

use mlx_gen::gen_core::{
    CancelFlag, GenerationMemory, GenerationOutput, GenerationRequest, LoadPhase, LoadShape,
    LoadSpec, Progress, TransformerComponent, WeightsSource,
};
use mlx_gen_minimax_h3::memory_strategy::{
    CONDITIONING_STAGE_PEAK_BYTES, RUNG4_REQUEST_PEAK_Q4_BYTES, TRANSFORMER_WINDOW_SIZE,
};
use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT, TEXT_ENCODER_COMPONENT};
use mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS;

/// The same attribution canvas `te_tier_generate_stages.rs` uses, for the same reason: quality is
/// not the question, the high-water is, and peak barely moves with canvas on this model.
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

fn footprint() -> usize {
    get_active_memory() + get_cache_memory()
}

fn rotate(
    stages: &mut BTreeMap<usize, (&'static str, usize, usize)>,
    open: &mut (usize, &'static str),
    next: &'static str,
) {
    let peak = get_peak_memory();
    let held = footprint();
    stages.insert(open.0, (open.1, peak, held));
    *open = (open.0 + 1, next);
    reset_peak_memory();
}

/// The smallest legal clip, streamed end to end.
#[test]
#[ignore = "sc-18662: needs MINIMAX_H3_SNAPSHOT (+ optional staged components) and Metal"]
fn the_streamed_request_peak_is_the_declared_one() {
    let frames = *LEGAL_FRAME_COUNTS.first().expect("legal frame counts") as u32;
    let root = env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<snapshot root>");
    // The rung's sole shared prerequisite, asked for explicitly: `LoadSpec::new` defaults to
    // `EagerMaterialization`, on which `requested_transformer_window` refuses the request below —
    // that refusal has its own unit test in `model.rs`.
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_load_shape(LoadShape::DeferredMaterialization);
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
        // The rung-4 composition: staged residency plus the bounded transformer window, on the
        // only declared component arm and the only declared window size.
        memory: Some(GenerationMemory {
            stage_residency: true,
            stream_transformer_blocks: true,
            transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: Some(TransformerComponent::Both),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut stages: BTreeMap<usize, (&'static str, usize, usize)> = BTreeMap::new();
    let mut open = (0usize, "pre-generate");
    let mut first_step_seen = false;

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
        .expect("streamed generate");
    stages.insert(open.0, (open.1, get_peak_memory(), footprint()));

    let elapsed = started.elapsed().as_secs_f64();
    let dit_tier = env("MINIMAX_H3_DIT")
        .map(|d| common::describe_dit_tier(&d))
        .unwrap_or_else(|| "bf16".to_owned());
    // **`Err` PANICS rather than collapsing to "bf16" (sc-17153).** The previous
    // `.ok().flatten()` turned an unreadable tier marker into the dense label, which silently
    // routed the run out of the q4/q4 branch below — the request-peak constant then graded
    // nothing and the test still passed green with no message saying so. That is precisely the
    // failure `common::describe_dit_tier` panics on for the DiT side ("an unlabelled tier would
    // make every measured cell unattributable"); the two sides now behave the same way.
    // `None` (no marker present) is still a legitimate dense tier and is not an error.
    let te_tier = env("MINIMAX_H3_TE")
        .map(|d| match mlx_gen::quant::packed_quant_bits_at(&d) {
            Ok(Some(bits)) => format!("q{bits}"),
            Ok(None) => "bf16".to_owned(),
            Err(error) => panic!(
                "{}: cannot read the staged text encoder's quantization marker: {error}. An \
                 unlabelled tier would send this run's request peak to be graded against — or, \
                 worse, silently excluded from — a constant captured under different tier \
                 conditions.",
                d.display()
            ),
        })
        .unwrap_or_else(|| "bf16".to_owned());
    println!("── streamed generate (rung 4, window 1, Both), per stage ──────");
    println!("  dit {dit_tier} / te {te_tier}  {WIDTH}x{HEIGHT} / {frames} frames / {STEPS} steps  in {elapsed:.1}s");
    println!("  {:<28} {:>10} {:>14}", "stage", "peak GB", "act+cache GB");
    let mut request_peak = 0usize;
    for (name, peak, held) in stages.values() {
        println!("  {name:<28} {:>10.2} {:>14.2}", gb(*peak), gb(*held));
        request_peak = request_peak.max(*peak);
    }
    println!(
        "  {:<28} {:>10.2}",
        "request peak (max of stages)",
        gb(request_peak)
    );

    // Evidence the render RAN rather than the `#[ignore]` falling through.
    let (decoded, audio) = match out {
        GenerationOutput::Video { frames, audio, .. } => (frames, audio),
        other => panic!("expected Video, got {other:?}"),
    };
    assert_eq!(decoded.len(), frames as usize, "decoded frame count");
    assert!(audio.is_some(), "H3 always produces a soundtrack");
    let names: Vec<&str> = stages.values().map(|(n, ..)| *n).collect();
    for want in [
        "conditioning",
        "dit-load + adaln-precompute",
        "denoise",
        "decode",
    ] {
        assert!(names.contains(&want), "no {want} stage boundary: {names:?}");
    }

    // --- the request-peak claims -------------------------------------------------------------
    //
    // **1. The rung moves the REQUEST peak, not just its own phases.** The resident request's
    // high-water is the conditioning stage's mark (`CONDITIONING_STAGE_PEAK_BYTES`, 53.07 GB with
    // the dense encoder — the constant the resident arm of `block_window_real.rs` reproduces).
    // A streamed request whose peak were still a large fraction of that would mean some phase this
    // rung claims to bound is not bounded on the real path. A quarter is a deliberately coarse
    // discriminator: the streamed peak is expected near the decode floor (~6 GB), the resident
    // mark is ~53 GB, and nothing between 13 GB and 53 GB is a plausible healthy reading.
    assert!(
        (request_peak as u64) < CONDITIONING_STAGE_PEAK_BYTES / 4,
        "the streamed request peaked at {:.2} GB against a resident-request {:.2} GB — rung 4 is \
         not bounding the request on the shipped generate path",
        gb(request_peak),
        CONDITIONING_STAGE_PEAK_BYTES as f64 / 1e9,
    );

    // **2. Declared versus measured.** The survey's `requestPeak` row and the contract docs are
    // written from `RUNG4_REQUEST_PEAK_Q4_BYTES`; a drift between that constant and this harness
    // would leave the provider publishing a request peak it no longer reproduces.
    //
    // **sc-17153 (activity 20066, rider 1):** bounded absolutely, per constant, via
    // `common::assert_declared_peak_constant` rather than by a `|measured/declared − 1| < 0.25`
    // band. Note what the retired comment here got wrong, since it was load-bearing elsewhere: it
    // claimed this assertion caught the *pair-proportional* drift the block-window ratio bands
    // cancel. It never did — assertion 1 above bounds the request peak against
    // `CONDITIONING_STAGE_PEAK_BYTES`, a third constant, and reads neither member of a
    // resident/windowed pair. That class is now caught where it actually lives, by the per-constant
    // bounds in `tests/block_window_real.rs`.
    //
    // **The tier guard is `dit_tier` AND `te_tier` (sc-17153).** `RUNG4_REQUEST_PEAK_Q4_BYTES` was
    // captured with a q4 DiT *and* a **q4 packed** text encoder. `MINIMAX_H3_TE` is optional, and
    // when it is absent the run falls back to the dense ~62 GB upstream encoder while `dit_tier` is
    // still "q4" — which sent a dense-TE request peak into this branch to be compared against a
    // packed-TE constant. Both tiers are now required to match the constant's own conditions.
    if dit_tier == "q4" && te_tier == "q4" {
        let declared = RUNG4_REQUEST_PEAK_Q4_BYTES as f64;
        let measured = request_peak as f64;
        println!(
            "  declared {:.2} GB, measured {:.2} GB",
            declared / 1e9,
            measured / 1e9
        );
        common::assert_declared_peak_constant(
            "RUNG4_REQUEST_PEAK_Q4_BYTES",
            RUNG4_REQUEST_PEAK_Q4_BYTES,
            request_peak as u64,
        );
    } else {
        // **Say so (sc-17153).** This branch is legitimate — the constant was captured at q4/q4 and
        // grading a bf16 run against it would be wrong — but it is the only assertion this test
        // makes about `RUNG4_REQUEST_PEAK_Q4_BYTES`, so a silent skip reads exactly like a pass.
        println!(
            "  RUNG4_REQUEST_PEAK_Q4_BYTES NOT graded (dit={dit_tier}, te={te_tier}; the constant \
             is q4 DiT + q4 packed TE only). This run measured {:.2} GB and asserted nothing \
             against it.",
            gb(request_peak)
        );
    }

    common::write_rung4_request_evidence(
        &dit_tier,
        &te_tier,
        WIDTH,
        HEIGHT,
        frames,
        STEPS,
        &stages.values().copied().collect::<Vec<_>>(),
        request_peak,
    );
}
