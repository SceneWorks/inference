//! sc-17152 — **the measured duration cost curve** on the published MiniMax-H3 weights.
//!
//! `#[ignore]`d: needs the whole ~196 GB snapshot and Metal. Point `MINIMAX_H3_SNAPSHOT` at the
//! snapshot root and run:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> SCENEWORKS_GPU_ID=mlx \
//!   cargo test -p mlx-gen-minimax-h3 --test integration duration_sweep_real:: -- --ignored --nocapture \
//!   --test-threads=1
//! ```
//!
//! # What is measured and what is derived — stated up front
//!
//! The epic's duration claims came from the upstream marketing envelope. These replace them, and the
//! line between the two kinds of number is kept visible rather than blurred:
//!
//! * **measured**: wall-clock and MLX peak for a complete `generate` (load → encode → denoise →
//!   decode) at a stated canvas, frame count and step count, on the real weights;
//! * **measured**: the MLX peak, and each denoise step's wall time, for a complete `generate`
//!   (load → encode → denoise → decode) at a stated canvas, frame count and step count on the real
//!   weights;
//! * **projected**: any 50-step total, which is the fastest measured step × 50. Cost is linear in
//!   steps — [`the_cost_is_linear_in_steps_so_a_reduced_sweep_is_honest`] measures that rather than
//!   assuming it — so a reduced-step sweep is a legitimate way to get the *curve*, which is what the
//!   bound needs. Every such number is labelled a projection where it appears.
//!
//! Nothing here extrapolates across the duration axis: every rung in the table is its own render.
//!
//! # This is a shared machine, and the instrument is chosen for it
//!
//! Other agents build and render on this Mac. A wall-clock *total* is therefore not reproducible:
//! the same 576x320 / 124-frame geometry measured 13.1 s/step (the epic's quiet-machine datum),
//! 36.75 s/step and 85.84 s/step within one session. So the reported figure is the **fastest
//! denoise step** of the schedule ([`Point::fastest_step`]) — every step runs the identical
//! 50-block forward, and contention can only make one slower, so the minimum is a hardware lower
//! bound rather than a sample of a noisy distribution. Median and max are printed beside it and
//! their ratio is the contention band; a wide band devalues the row, and the reader can see it.
//!
//! ## The seven former wall-clock assertions, converted (sc-19556)
//!
//! sc-19505 removed `Instant`-fed assertions from the sibling tests; this file carried seven more,
//! all fed by `step_times.push(last.elapsed().as_secs_f64())`. sc-19556 converted them against a
//! **measured run** of all three owning tests on the packed q4 tier (2026-08-19, this Mac, Metal),
//! per the mapping recorded on the story. What each site asserts now:
//!
//! | old site | was | now asserts |
//! |---|---|---|
//! | `:424` | monotone `fastest_step` across 4 renders | pairwise monotone [`Point::denoise_peak`] — a longer clip packs a longer sequence, so the denoise transient must rise; a clock moves with host load, an allocation does not |
//! | `:455` | `step_growth > 1.0` | **deleted as subsumed**: `exponent > 0.8` with `seq_growth > 1` (asserted pairwise) already implies `step_growth > seq_growth^0.8 > 1` |
//! | `:459` | `exponent > 0.8` | **kept, documented** — the superlinear *cost* curve is the published claim and only a duration can carry it; the instrument is already the min-over-schedule form (`fastest_step` is a hardware lower bound), the same shape PR #626's review settled on |
//! | `:514` | `fastest_step() > 0.5` | every step's own MLX high-water clears the staged tier's denoise residency ([`denoise_resident_floor`]) — "the 33 B DiT was resident and live at every step", strictly stronger than any wall-clock floor |
//! | `:523` | `band < 6.0` (contention band) | per-step **peak flatness** within one render — identical shapes ⇒ identical allocation (measured byte-flat at q4: 13.474 GB × 7) |
//! | `:530` | `second_half < first_half * 3.0` | the same half-vs-half ratio on **`step_active`** — the direct reading of "a leak or a cache that accumulates with the step index" |
//! | `:610` | `fastest_step() > 0.5` | the same per-step residency floor as `:514`, at the default canvas |
//!
//! (Two further assertions read the *count* of timed steps rather than any duration — they gate
//! that the progress callback fired, and are not in this class.)
//!
//! The wall clock is still **measured and printed** everywhere it always was — the table's s/step
//! columns are the epic's published cost figures — but the only remaining duration *assertion* is
//! the exponent, which gates a genuinely-published superlinearity claim in its contention-tolerant
//! min-reduced form. Every other gate now reads the MLX allocator, which is a function of the
//! shapes the forward built rather than of how busy this Mac was.
//!
//! # The tier axis (sc-17150)
//!
//! The q4/q8 tiers now exist and load. `model::load` no longer refuses `spec.quantize`: it
//! **reconciles** it against the staged tier's `config.json` marker, because MiniMax-H3 never
//! quantizes at load (see `model::reconcile_tier`).
//!
//! This harness takes a snapshot root, not a tier — so a row measures **whatever tier that root
//! stages**: the flat upstream snapshot is bf16, and a split-layout root whose `transformer` /
//! `text_encoder` are packed tier directories measures that tier (that is how the sc-19556 q4
//! measured run was staged). The residency-floor assertions read the staged DiT's own
//! `quantization` marker ([`denoise_resident_floor`]) — the same authority `model::reconcile_tier`
//! trusts — so they gate correctly at every tier. The per-tier *quality* axis still lives in
//! `tests/quant_tiers_real.rs`, which stages each tier's DiT via `MINIMAX_H3_DIT` and renders the
//! identical geometry at each, so the two harnesses answer different questions and neither has to
//! guess the other's.
//!
//! # The 4 s row does not exist
//!
//! The lattice floor is 124 frames = **5.1667 s** and `MIN_DURATION_SECONDS` is a hardcoded model
//! property, so a 4 s render is not slow — it is unrepresentable.
//! [`the_four_second_row_is_structurally_unrenderable`] pins that rather than leaving a blank cell
//! that reads like a missing measurement.

use crate::common;

use std::time::{Duration, Instant};

use mlx_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

use mlx_gen_minimax_h3::cost::{packed_seq_len, DitSequenceCost};
use mlx_gen_minimax_h3::model::load;
use mlx_gen_minimax_h3::{
    align_frames_for_duration, resolve_geometry, JointGeometry, MiniMaxH3DitConfig,
    LEGAL_FRAME_COUNTS, MAX_DURATION_SECONDS, MINIMAX_H3_FPS, MIN_DURATION_SECONDS, PATCH_SIZE,
    SMALLEST_LEGAL_FRAMES,
};

use common::snapshot;

/// The gating canvas the whole epic's cost figures are quoted at — the smallest sensible 32-aligned
/// 16:9-ish frame, and what `first_light` measured 655.3 s / 52.81 GB at.
const SWEEP_WIDTH: u32 = 576;
const SWEEP_HEIGHT: u32 = 320;

/// Model evaluations per sweep point. Deliberately far below `DEFAULT_STEPS`: the *curve* is what
/// bounds the duration, cost is linear in steps, and 50 steps × 4 durations is over three hours of
/// wall-clock for numbers that are a multiplication away.
fn sweep_steps() -> u32 {
    std::env::var("MINIMAX_H3_SWEEP_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Pixel spread of one frame on a `[0, 1]` scale — the "is there a picture here" statistic.
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

/// One sweep point's receipt.
#[derive(Debug, Clone)]
struct Point {
    frames: i32,
    seq_len: i64,
    wall: Duration,
    /// Per-denoise-step wall times, in order.
    step_times: Vec<f64>,
    peak_bytes: u64,
    /// MLX **active** bytes sampled at each denoise step's progress tick, in order — the same
    /// schedule the `step_times` above index, so the two are directly comparable.
    step_active: Vec<u64>,
    /// MLX peak bytes attributable to each denoise step alone: the peak counter is reset at every
    /// tick, so entry `i` is the high-water of step `i` and nothing earlier.
    step_peaks: Vec<u64>,
    /// The peak standing at the FIRST progress tick — load + text encode + step 1. This is the
    /// ~53 GB conditioning mark that masks every later stage.
    conditioning_peak: u64,
    spread: f64,
}

impl Point {
    fn seconds(&self) -> f64 {
        f64::from(self.frames) / MINIMAX_H3_FPS
    }

    /// **The reported cost figure: the fastest denoise step.**
    ///
    /// This machine is a shared dev box and other agents' builds run on it, so a single wall-clock
    /// total is not reproducible — the sc-17152 debug sweep measured the same 124-frame geometry at
    /// 36.75 and 85.84 s/step in one session, and the epic's own quiet-machine datum is 13.1. A
    /// total divided by the step count inherits every one of those excursions plus the load I/O.
    ///
    /// The minimum over the schedule's steps is the right statistic instead, for a structural
    /// reason rather than a statistical one: **contention can only make a step slower.** Every step
    /// runs the identical 50-block forward on identical shapes, so the fastest one is the closest
    /// this machine gets to the uncontended cost and is a genuine *lower bound* on it. The median
    /// and max are reported alongside so the spread is visible rather than smoothed away — a wide
    /// one means the machine was busy, not that the model is unpredictable.
    fn fastest_step(&self) -> f64 {
        self.step_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    fn median_step(&self) -> f64 {
        let mut v = self.step_times.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    fn slowest_step(&self) -> f64 {
        self.step_times.iter().copied().fold(0.0, f64::max)
    }

    /// The largest per-step MLX peak, i.e. the denoise stage's own high-water with the conditioning
    /// mark subtracted out. Unlike a wall clock this is a property of the shapes the forward built,
    /// so it is reproducible under contention.
    fn denoise_peak(&self) -> u64 {
        self.step_peaks.iter().copied().max().unwrap_or(0)
    }
}

/// The denoise-stage residency floor for the tier staged at `root/transformer`, read from the
/// tier's own `config.json` `quantization` marker — the same authority
/// [`mlx_gen_minimax_h3::model::load`]'s `reconcile_tier` treats as decisive about what is staged.
///
/// * dense (no marker) → [`memory_strategy::DENOISE_RESIDENT_BF16_BYTES`] (40.43 GB, measured);
/// * any packed tier → [`memory_strategy::DENOISE_RESIDENT_Q4_BYTES`] (11.63 GB, measured). q8 has
///   no measured residency constant of its own yet, and gating a q8 run at the q4 floor is the
///   conservative direction: it still proves a resident, live, packed 33 B DiT and can never
///   false-red (q8's residency is strictly larger than q4's).
///
/// The sc-19556 measured run (q4, 576x320, 124 frames, 8 steps) put every per-step high-water at
/// 13.474 GB against this floor's 11.63 — the margin is the denoise activation transient, so the
/// floor holds by construction rather than by luck: a step cannot run without its weights resident,
/// and the transient only adds to them.
///
/// [`memory_strategy::DENOISE_RESIDENT_BF16_BYTES`]: mlx_gen_minimax_h3::memory_strategy::DENOISE_RESIDENT_BF16_BYTES
/// [`memory_strategy::DENOISE_RESIDENT_Q4_BYTES`]: mlx_gen_minimax_h3::memory_strategy::DENOISE_RESIDENT_Q4_BYTES
fn denoise_resident_floor(root: &std::path::Path) -> u64 {
    use mlx_gen_minimax_h3::memory_strategy::{
        DENOISE_RESIDENT_BF16_BYTES, DENOISE_RESIDENT_Q4_BYTES,
    };
    match mlx_gen::quant::packed_quant_bits_at(&root.join("transformer")) {
        Ok(Some(_)) => DENOISE_RESIDENT_Q4_BYTES,
        _ => DENOISE_RESIDENT_BF16_BYTES,
    }
}

/// Render one point and assert it really rendered.
fn render(root: &std::path::Path, width: u32, height: u32, frames: i32, steps: u32) -> Point {
    let model = match load(&LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))) {
        Ok(m) => m,
        Err(e) => panic!("load {}: {e}", root.display()),
    };
    let req = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, \
                 seagulls calling"
            .into(),
        width,
        height,
        frames: Some(frames as u32),
        steps: Some(steps),
        seed: Some(17_152),
        cancel: CancelFlag::default(),
        ..Default::default()
    };
    model.validate(&req).expect("the sweep point must validate");

    // The peak counter is process-wide and cumulative; reset it so this point's figure is this
    // point's and not the previous, longer render's.
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    let mut ticks = 0u32;
    // Per-step timing. The first interval spans load + text encode + DiT load as well as step 1, so
    // it is recorded and then dropped from the statistics — including it would charge the model's
    // per-step cost for ~140 GB of weight I/O.
    let mut last = Instant::now();
    let mut step_times: Vec<f64> = Vec::new();
    // sc-19556 — the clock-free half of the instrument. The MLX peak counter is reset at every
    // progress tick, so each entry of `step_peaks` is one denoise step's own high-water rather than
    // the running maximum, which the ~53 GB conditioning stage otherwise masks for the whole render.
    // `step_active` is the live (non-cache) residency at the tick. Both are functions of the shapes
    // the forward built, so unlike `step_times` they do not move with host load.
    let mut step_active: Vec<u64> = Vec::new();
    let mut step_peaks: Vec<u64> = Vec::new();
    let out = model
        .generate(&req, &mut |p| {
            if let Progress::Step { .. } = p {
                ticks += 1;
                step_times.push(last.elapsed().as_secs_f64());
                step_peaks.push(mlx_rs::memory::get_peak_memory() as u64);
                step_active.push(mlx_rs::memory::get_active_memory() as u64);
                mlx_rs::memory::reset_peak_memory();
                last = Instant::now();
            }
        })
        .unwrap_or_else(|e| panic!("{width}x{height} / {frames} frames / {steps} steps: {e}"));
    let wall = started.elapsed();
    let first_interval = step_times.first().copied().unwrap_or(0.0);
    // The first interval carries load + text encode as well as step 1, on every axis: drop it from
    // all three series together so they stay index-aligned with each other.
    let conditioning_peak = step_peaks.first().copied().unwrap_or(0);
    if step_times.len() > 1 {
        step_times.remove(0);
        step_peaks.remove(0);
        step_active.remove(0);
    }
    // Resetting the counter per tick split the render's high-water into segments; the maximum over
    // those segments (plus the decode-only tail) is the same number the un-instrumented harness
    // reported, so the peak column is unchanged by the instrumentation. The tail is decode-only:
    // the last step's compute is materialized by the synchronous per-step eval and sampled at its
    // own tick, so nothing of the denoise loop leaks past the final tick into the tail segment.
    let tail_peak = mlx_rs::memory::get_peak_memory() as u64;
    let peak_bytes = conditioning_peak
        .max(tail_peak)
        .max(step_peaks.iter().copied().max().unwrap_or(0));

    let (rendered, audio) = match out {
        GenerationOutput::Video { frames, audio, .. } => (frames, audio),
        other => panic!("expected a Video output, got {other:?}"),
    };
    // Evidence the render really happened, not just that nothing panicked: the frame count, one
    // structured middle frame, and a soundtrack. An `#[ignore]`d test that fell through would print
    // `ok` and a fabricated row of the table.
    assert_eq!(rendered.len(), frames as usize, "decoded frame count");
    let audio = audio.expect("MiniMax-H3 always produces a soundtrack");
    assert!(!audio.samples.is_empty(), "the soundtrack is empty");
    assert_eq!(ticks, steps, "one progress tick per evaluation");
    let spread = frame_std(&rendered[rendered.len() / 2].pixels);
    assert!(
        (0.005..=0.48).contains(&spread),
        "{frames} frames: middle-frame pixel std {spread:.4} — the clip is constant, saturated or \
         diverged. (The floor is looser than first_light's 0.02 because a {steps}-step render is \
         deliberately under-converged; it still rejects a black or stubbed decode.)"
    );
    assert!(
        peak_bytes > 1_000_000_000,
        "peak {peak_bytes} B is too small for a real 33 B forward — this measured nothing"
    );

    let geometry = resolve_geometry(width, height, frames).unwrap();
    // The text-token count is the tokenizer's; the sweep records the layout's own sequence length so
    // the cost model and the measurement are quoted against the same number.
    let seq_len = packed_seq_len(&geometry.joint, PATCH_SIZE, text_tokens(root), 0, 2).unwrap();
    assert!(
        !step_times.is_empty(),
        "no denoise step was timed — the progress callback never fired"
    );
    eprintln!(
        "    (first interval {first_interval:.1} s carries the ~140 GB load + text encode and is \
         excluded from the step statistics)"
    );
    eprintln!(
        "    [sc-19556 probe] conditioning peak {:.3} GB | per-step peaks {:?} GB | per-step \
         active {:?} GB",
        conditioning_peak as f64 / 1e9,
        step_peaks
            .iter()
            .map(|b| format!("{:.3}", *b as f64 / 1e9))
            .collect::<Vec<_>>(),
        step_active
            .iter()
            .map(|b| format!("{:.3}", *b as f64 / 1e9))
            .collect::<Vec<_>>(),
    );
    Point {
        frames,
        seq_len,
        wall,
        step_times,
        peak_bytes,
        step_active,
        step_peaks,
        conditioning_peak,
        spread,
    }
}

/// The prompt's real token count, from the shipped tokenizer — not a guess.
fn text_tokens(root: &std::path::Path) -> i32 {
    let tok = mlx_gen_minimax_h3::MiniMaxH3Tokenizer::from_snapshot(root).unwrap();
    let (ids, _) = tok
        .encode_prompt(
            "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, \
             seagulls calling",
        )
        .unwrap();
    ids.shape()[1]
}

/// **The cost table.** Every legal duration rung the story asks for, each its own render.
///
/// 5, 8, 12 and 15 s map onto the `17n + 5` lattice as 124, 192, 294 and 345 frames — 8 s is the
/// only one of the four that lands exactly, and 15 s lands on nothing at all (the lattice ceiling is
/// 14.375 s; see `the_declared_ceiling_has_no_lattice_point`).
#[test]
#[ignore = "sc-17152: needs the full MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~25 min at the default 6 steps"]
fn the_duration_cost_curve_at_the_gating_canvas() {
    let root = snapshot();
    let steps = sweep_steps();
    let width = env_u32("MINIMAX_H3_SWEEP_WIDTH", SWEEP_WIDTH);
    let height = env_u32("MINIMAX_H3_SWEEP_HEIGHT", SWEEP_HEIGHT);
    let cfg = MiniMaxH3DitConfig::default();

    // The four rungs the story's 4/5/8/12/15 s row set maps onto.
    //
    // Two of the five nominal durations are not requestable at all and are handled as facts rather
    // than as blank cells: **4 s** is below the lattice floor
    // (`the_four_second_row_is_structurally_unrenderable`) and **15 s** is above the lattice ceiling
    // since sc-17152 derived the advertised range from the lattice
    // (`the_declared_ceiling_has_no_lattice_point`). The nominal 5 s and 15 s rows are therefore
    // swept at the two rungs that actually bound the model — 5.1667 s and 14.375 s — and the
    // refusals are asserted here so the table's endpoints are not mistaken for the nominal ones.
    assert!(
        align_frames_for_duration(15.0).is_err(),
        "15 s must be refused, or this sweep's top row is not the model's ceiling"
    );
    assert!(
        align_frames_for_duration(4.0).is_err(),
        "4 s must be refused"
    );
    let requested = [
        f64::from(SMALLEST_LEGAL_FRAMES) as f32 / MINIMAX_H3_FPS as f32, // nominal 5 s
        8.0,
        12.0,
        MAX_DURATION_SECONDS, // nominal 15 s -> the lattice ceiling, 14.375 s
    ];
    let mut points: Vec<(f32, Point)> = Vec::new();
    for seconds in requested {
        let frames = align_frames_for_duration(seconds).unwrap();
        let p = render(&root, width, height, frames, steps);
        eprintln!(
            "[sweep] requested {seconds:>7.4} s -> {:>3} frames ({:.4} s) | seq {:>6} | steps \
             min {:>6.2} / med {:>6.2} / max {:>6.2} s | {:>7.1} s wall | peak {:>6.2} GB | \
             std {:.4}",
            p.frames,
            p.seconds(),
            p.seq_len,
            p.fastest_step(),
            p.median_step(),
            p.slowest_step(),
            p.wall.as_secs_f64(),
            p.peak_bytes as f64 / 1e9,
            p.spread,
        );
        points.push((seconds, p));
    }

    eprintln!("\n=== sc-17152 MEASURED duration cost curve ===");
    eprintln!(
        "canvas {width}x{height}, bf16 (the only tier that exists — q4/q8 pend sc-17150), \
         {steps} evaluations per point, seed 17152, MLX/Metal"
    );
    eprintln!(
        "{:>9} {:>7} {:>9} {:>8} {:>10} {:>10} {:>10} {:>10} {:>13}",
        "requested",
        "frames",
        "actual s",
        "seq",
        "min s/stp",
        "med s/stp",
        "max s/stp",
        "peak (GB)",
        "50-step proj"
    );
    for (seconds, p) in &points {
        eprintln!(
            "{seconds:>8.4}s {:>7} {:>8.4}s {:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.0}s",
            p.frames,
            p.seconds(),
            p.seq_len,
            p.fastest_step(),
            p.median_step(),
            p.slowest_step(),
            p.peak_bytes as f64 / 1e9,
            p.fastest_step() * 50.0,
        );
    }
    eprintln!(
        "\nMEASURED: the three s/step columns (per denoise step, load excluded) and peak.\n\
         PROJECTED: the 50-step column, = min s/step x 50. Cost is linear in steps \
         (`the_cost_is_linear_in_steps_so_a_reduced_sweep_is_honest`), so this is arithmetic on a \
         measurement — but it is not itself a measurement and a 50-step render also pays the load \
         once.\n\
         The min column is the reported cost: this is a SHARED dev machine and concurrent builds \
         inflate steps but cannot make one faster than the hardware allows, so the fastest step is \
         a lower bound and the min/max spread is the contention band."
    );
    for (_, p) in &points {
        let c = DitSequenceCost::new(p.seq_len, &cfg).unwrap();
        eprintln!(
            "  {:>3} frames: widest materialized tensor {} elements ({:.3}x i32::MAX, {:.2} GB at \
             bf16); a materializing attention would build {} ({:.0}x)",
            p.frames,
            c.widest_materialized_elements(),
            c.widest_materialized_elements() as f64 / mlx_gen_minimax_h3::MAX_WRITABLE_ELEMS as f64,
            c.widest_materialized_bytes(2) as f64 / 1e9,
            c.dense_score_elements,
            c.dense_score_elements as f64 / mlx_gen_minimax_h3::MAX_WRITABLE_ELEMS as f64,
        );
    }

    eprintln!("\n[sc-19556 probe] clock-free curve candidates");
    for (_, p) in &points {
        eprintln!(
            "  {:>3} frames | seq {:>6} | denoise peak {:>9.4} GB | cond peak {:>9.4} GB | \
             fastest step {:>7.3} s",
            p.frames,
            p.seq_len,
            p.denoise_peak() as f64 / 1e9,
            p.conditioning_peak as f64 / 1e9,
            p.fastest_step(),
        );
    }
    {
        let (f, l) = (&points[0].1, &points[3].1);
        let seq_growth = l.seq_len as f64 / f.seq_len as f64;
        let peak_growth = l.denoise_peak() as f64 / f.denoise_peak() as f64;
        eprintln!(
            "  denoise-peak growth {peak_growth:.4}x over a {seq_growth:.4}x sequence => S^{:.3}",
            peak_growth.ln() / seq_growth.ln()
        );
    }

    // --- what the table has to satisfy to be a curve rather than four numbers -------------------
    assert_eq!(points.len(), 4);
    let frames: Vec<i32> = points.iter().map(|(_, p)| p.frames).collect();
    assert_eq!(
        frames,
        vec![124, 192, 294, 345],
        "the four rungs 5/8/12/15 s align onto"
    );

    // **The curve must be monotonic — asserted on the denoise allocation, not on any clock**
    // (sc-19556). The question this gate asks is "were these really four different renders?", and
    // the per-step MLX high-water answers it better than the fastest step ever could: a longer
    // clip packs a longer sequence, so the denoise transient over the resident weights must rise
    // rung to rung, while four renders of the same geometry would land on the same peak to the
    // byte. A clock moves with host load — the debug run of this same sweep produced 343 / 282 /
    // 624 / 328 s totals, no relationship to duration at all, and even the per-step minimum only
    // bounds contention from one side — but an allocation is a function of the shapes the forward
    // built and of nothing else.
    //
    // The sc-19556 measured run is itself the receipt: the old `fastest_step` form of this gate
    // FALSE-REDDED during it (294 frames contended to 50.15 s/step min against 345 frames' 45.00,
    // concurrent cargo builds), while the denoise peaks separated cleanly and monotonically —
    // 13.425 / 14.300 / 15.729 / 16.443 GB across the four rungs, byte-flat within each rung.
    for pair in points.windows(2) {
        assert!(
            pair[1].1.denoise_peak() > pair[0].1.denoise_peak(),
            "{} frames' denoise high-water was {:.3} GB against {} frames' {:.3} GB — the curve \
             is not monotonic in the per-step allocation, so these are not four different renders \
             (a longer packed sequence must build a larger denoise transient)",
            pair[1].1.frames,
            pair[1].1.denoise_peak() as f64 / 1e9,
            pair[0].1.frames,
            pair[0].1.denoise_peak() as f64 / 1e9,
        );
        assert!(pair[1].1.seq_len > pair[0].1.seq_len);
    }

    // ...and it must grow FASTER than the sequence, because the attention is quadratic in it while
    // everything else is linear. A purely linear curve would mean the attention cost is not visible
    // at this canvas — which would make the whole cost-cliff premise wrong and is worth failing
    // over, since the duration bound is derived from it.
    let (first, last) = (&points[0].1, &points[3].1);
    let step_growth = last.fastest_step() / first.fastest_step();
    let seq_growth = last.seq_len as f64 / first.seq_len as f64;
    let exponent = step_growth.ln() / seq_growth.ln();
    eprintln!(
        "\nfastest step grew {step_growth:.2}x for a {seq_growth:.2}x longer sequence — S^{exponent:.2} \
         (purely linear would be S^1.00, purely quadratic S^2.00). The mix is what the cost model \
         predicts: the projections and feed-forward are O(S), the attention O(S^2), and at this \
         canvas the linear terms still dominate."
    );
    // sc-19556: the former `step_growth > 1.0` assertion is deleted as subsumed — with
    // `seq_growth > 1` (each pair's `seq_len` is asserted strictly increasing above) the exponent
    // bound below already implies `step_growth > seq_growth^0.8 > 1`, so a separate gate on the
    // ratio asserted nothing the exponent does not. The same subsumption pattern as PR #626's
    // candle-gen-svd deletion: what remains behind is stated, not nothing.
    //
    // The exponent gate itself **stays a duration on purpose** (AC1's second branch): "the
    // per-step cost grows superlinearly in the sequence" is the epic's published cost-curve claim,
    // and only a duration can carry it — the measured q4 run put the denoise-peak growth at
    // S^0.200 against the fastest step's S^1.15 (the allocation is linear in the transient and
    // dwarfed by the resident weights), so a peak-derived exponent cannot distinguish the
    // attention's quadratic term from a purely linear forward and the two instruments genuinely
    // answer different questions. The instrument is already the contention-tolerant min-reduced form PR #626's review
    // settled on: `fastest_step` is the minimum over a schedule of identical forwards, a hardware
    // lower bound that contention can only raise — and raising both endpoints of the sweep cannot
    // manufacture superlinearity that is not there. The margin (0.8) is the published one and is
    // not relaxed.
    assert!(
        exponent > 0.8,
        "the cost grew as S^{exponent:.2}, sub-linearly — that is not a real forward"
    );
}

/// **Cost is linear in steps** — the premise that makes a reduced-step sweep an honest measurement
/// of the curve, and the only thing licensing the table's 50-step projection column.
///
/// Demonstrated *directly* rather than by a two-point fit across step counts: one render, and every
/// denoise step after the first timed individually. Each step runs the identical 50-block forward
/// over the identical shapes, so if the per-step times sit in a tight band then `n` steps cost
/// `n × per_step` by construction — and the band width is simultaneously this machine's contention
/// measurement.
///
/// A two-point fit across two *renders* was tried first and is strictly worse here: it pays the
/// ~140 GB load twice, and it charges the difference between two contention regimes to the step
/// cost. It measured 36.75 s/step in one profile and 51.37 s/step in another against the epic's
/// quiet-machine 13.1 — three numbers for one quantity, none of them the model's.
#[test]
#[ignore = "sc-17152: needs the full MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~10 min"]
fn the_cost_is_linear_in_steps_so_a_reduced_sweep_is_honest() {
    let root = snapshot();
    let width = env_u32("MINIMAX_H3_SWEEP_WIDTH", SWEEP_WIDTH);
    let height = env_u32("MINIMAX_H3_SWEEP_HEIGHT", SWEEP_HEIGHT);
    let steps = 8u32;

    let p = render(&root, width, height, SMALLEST_LEGAL_FRAMES, steps);
    let times: Vec<String> = p.step_times.iter().map(|t| format!("{t:.2}")).collect();
    let mean = p.step_times.iter().sum::<f64>() / p.step_times.len() as f64;
    let band = p.slowest_step() / p.fastest_step();
    eprintln!(
        "[linearity] {width}x{height} / {} frames / {steps} steps: per-step [{}] s\n  \
         min {:.2} / median {:.2} / mean {mean:.2} / max {:.2} s  =>  contention band {band:.2}x\n  \
         wall {:.1} s total; the load + text encode is the {:.1} s the first interval carried",
        SMALLEST_LEGAL_FRAMES,
        times.join(", "),
        p.fastest_step(),
        p.median_step(),
        p.slowest_step(),
        p.wall.as_secs_f64(),
        p.wall.as_secs_f64() - p.step_times.iter().sum::<f64>(),
    );
    eprintln!(
        "  50-step PROJECTION at this geometry: {:.0} s from the fastest step, {:.0} s from the \
         median",
        p.fastest_step() * 50.0,
        p.median_step() * 50.0,
    );

    eprintln!(
        "[sc-19556 probe] per-step peaks (GB) {:?}\n  per-step active (GB) {:?}\n  \
         peak band {:.4}x, active band {:.4}x",
        p.step_peaks
            .iter()
            .map(|b| format!("{:.4}", *b as f64 / 1e9))
            .collect::<Vec<_>>(),
        p.step_active
            .iter()
            .map(|b| format!("{:.4}", *b as f64 / 1e9))
            .collect::<Vec<_>>(),
        p.step_peaks.iter().max().copied().unwrap_or(0) as f64
            / p.step_peaks.iter().min().copied().unwrap_or(1).max(1) as f64,
        p.step_active.iter().max().copied().unwrap_or(0) as f64
            / p.step_active.iter().min().copied().unwrap_or(1).max(1) as f64,
    );

    assert!(
        p.step_times.len() >= 4,
        "need at least four timed steps to speak about a band, got {}",
        p.step_times.len()
    );
    // sc-19556: `fastest_step() > 0.5` — an absolute duration constant, structurally the
    // `total.as_millis() >= 8` sc-19505 deleted as its headline defect — is replaced by a floor a
    // clock cannot fake: **every step's own MLX high-water must clear the staged tier's denoise
    // residency.** A step cannot run without the DiT's weights resident, so a per-step peak below
    // the floor means the step did not run a real 50-block forward — which is exactly what the
    // duration constant was reaching for, asserted on the mechanism instead of on the wall clock.
    // Measured (q4, this geometry): 13.474 GB per step against the 11.63 GB floor.
    let floor = denoise_resident_floor(&root);
    for (i, &pk) in p.step_peaks.iter().enumerate() {
        assert!(
            pk >= floor,
            "step {}'s own high-water was {:.3} GB — below the staged tier's denoise residency of \
             {:.3} GB, so the 33 B DiT was not resident and live at that step and this is not a \
             real 50-block forward",
            i + 1,
            pk as f64 / 1e9,
            floor as f64 / 1e9,
        );
    }
    // The steps must be the same size as each other — and allocation, not wall-clock, is where
    // "the same size" is actually observable on a shared machine. Every step runs the identical
    // 50-block forward over identical shapes, so each step's own high-water must be the same;
    // the sc-19556 measured run was byte-flat (13.474 GB x 7, band 1.0000x). The 1.05x allowance
    // is for allocator recycling nondeterminism, not for contention — host load cannot move an
    // allocation. This still rejects the shape that would invalidate the projection: a leak or a
    // per-step cache large enough to bend the 50-step cost projection bends this band first.
    let peak_band = p.step_peaks.iter().max().copied().unwrap_or(0) as f64
        / p.step_peaks.iter().min().copied().unwrap_or(1).max(1) as f64;
    assert!(
        peak_band < 1.05,
        "the per-step allocation band is {peak_band:.4}x (measured 1.0000x) — the steps are not \
         running the same shapes, so the table's 50-step projections are not supportable from \
         this run"
    );
    // ...and nothing may ACCUMULATE with the step index. This is the direct reading of the leak /
    // growing-cache shape the old `second_half < first_half * 3.0` wall-clock ratio stood in for:
    // the same half-vs-half comparison, on the live (non-cache) residency at each step's tick
    // instead of on durations. Measured: 11.645-11.797 GB, oscillating, no trend; 1.05x rejects
    // any accumulation above ~590 MB across the schedule while tolerating the measured 1.3%
    // buffer-recycling oscillation.
    // Means, not sums: 7 timed steps split 3/4, and a sum over unequal halves would compare 4
    // steps' residency against 3 and false-red on the size mismatch alone.
    let halves = p.step_active.split_at(p.step_active.len() / 2);
    let mean = |s: &[u64]| s.iter().sum::<u64>() as f64 / s.len().max(1) as f64;
    let (first_half, second_half) = (mean(halves.0), mean(halves.1));
    assert!(
        second_half < first_half * 1.05,
        "the second half of the schedule held {:.3} GB live per step against the first half's \
         {:.3} GB — residency is growing with the step index (a leak, or a cache that \
         accumulates), which no multiplication by 50 can be honest about",
        second_half / 1e9,
        first_half / 1e9,
    );

    // The peak must be essentially step-independent: it is load-bound, not schedule-bound (the
    // AdaLN cache is the only step-sized allocation and it is ~10 MB per timestep). A peak that
    // tracked the step count would mean the reduced-step sweep under-reports every peak in the
    // table above.
    let short = render(&root, width, height, SMALLEST_LEGAL_FRAMES, 2);
    let ratio = p.peak_bytes as f64 / short.peak_bytes as f64;
    eprintln!(
        "  peak 2 steps {:.2} GB, {steps} steps {:.2} GB ({ratio:.3}x) — load-bound, not \
         schedule-bound",
        short.peak_bytes as f64 / 1e9,
        p.peak_bytes as f64 / 1e9
    );
    assert!(
        ratio < 1.15,
        "peak grew {ratio:.2}x from 2 to {steps} steps — it is schedule-bound after all, and the \
         reduced-step sweep under-reports the peak column"
    );
}

/// **The default canvas** (sc-17152): the same lattice floor at 1344x768, the area the released
/// checkpoint actually generates at.
///
/// The gating-canvas sweep above measures the *duration* axis. This measures the **canvas** axis,
/// and it is the more violent of the two: 1344x768 carries **1008** rows per latent frame against
/// 576x320's 180, so the shortest legal clip at the default canvas already builds a longer sequence
/// (37 774 rows) than the *longest* clip at the gating one (19 530). Since the attention is
/// quadratic in the sequence, a duration bound quoted without a canvas is not a bound.
///
/// Only the lattice floor is rendered. The top of the default canvas is characterized instead by
/// `sequence_cost_real.rs`'s measured per-attention-call curve — 19.412 s per call at
/// `S = 104 030`, which is **13.5 h of attention alone** over 50 blocks x 50 steps. Spending that
/// to re-learn it end to end would be a day of machine time for a number already in hand.
/// `MINIMAX_H3_ANCHOR_FRAMES` raises the rung for anyone who wants to.
#[test]
#[ignore = "sc-17152: needs the full MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~15 min"]
fn the_default_canvas_is_bounded_by_area_not_by_duration() {
    let root = snapshot();
    let frames = env_u32("MINIMAX_H3_ANCHOR_FRAMES", SMALLEST_LEGAL_FRAMES as u32) as i32;
    let steps = env_u32("MINIMAX_H3_ANCHOR_STEPS", 3);
    let (width, height) = (1344u32, 768u32);
    assert_eq!(
        width * height,
        mlx_gen_minimax_h3::CANVAS_MAX_PIXELS,
        "the anchor must be the checkpoint's own canvas budget"
    );

    let p = render(&root, width, height, frames, steps);
    eprintln!(
        "\n=== sc-17152 MEASURED default-canvas anchor ===\n\
         {width}x{height} / {frames} frames ({:.4} s) / {steps} steps, bf16, seed 17152\n  \
         seq {} rows | steps min {:.2} / med {:.2} / max {:.2} s | peak {:.2} GB | wall {:.1} s\n  \
         50-step PROJECTION: {:.0} s ({:.2} h)",
        p.seconds(),
        p.seq_len,
        p.fastest_step(),
        p.median_step(),
        p.slowest_step(),
        p.peak_bytes as f64 / 1e9,
        p.wall.as_secs_f64(),
        p.fastest_step() * 50.0,
        p.fastest_step() * 50.0 / 3600.0,
    );

    // The point of the anchor: the SHORTEST clip at the default canvas is a longer sequence — and
    // therefore a costlier step — than the LONGEST clip at the gating canvas. If this ever stops
    // holding, the canvas is no longer the dominant term and the bound has to be re-derived.
    let gating_top = 19_530i64;
    assert!(
        p.seq_len > gating_top,
        "the default canvas at {frames} frames builds {} rows, not more than the gating canvas's \
         {gating_top} at 345 frames — the canvas is no longer the dominant term",
        p.seq_len
    );
    // sc-19556: the absolute duration floor (`fastest_step() > 0.5`) is replaced by the same
    // per-step residency floor as the linearity test — see the comment there. At the default
    // canvas the sequence is the longest this harness builds, so the activation transient over the
    // floor is at its widest; the floor is the invariant part.
    let floor = denoise_resident_floor(&root);
    for (i, &pk) in p.step_peaks.iter().enumerate() {
        assert!(
            pk >= floor,
            "step {}'s own high-water was {:.3} GB — below the staged tier's denoise residency of \
             {:.3} GB, so the 33 B DiT was not resident and live at that step and this is not a \
             real 50-block forward",
            i + 1,
            pk as f64 / 1e9,
            floor as f64 / 1e9,
        );
    }
    // The flat ~53 GB across the whole duration range belongs to the **conditioning** stage: the
    // dense text encoder runs before the DiT is mapped and sets the process high-water
    // (`mlx_gen_minimax_h3::memory_strategy::CONDITIONING_STAGE_PEAK_BYTES`), so every later stage
    // is masked. It is NOT an activation-dominated peak — that attribution was retracted
    // (sc-18659). At the default canvas the activation transient is 5.3x larger, so this is the
    // first place a denoise-side cost could out-grow that mask, and the measurement says whether
    // the peak column of the table above generalizes to the canvas axis or only to the duration one.
    eprintln!(
        "  peak vs the gating canvas's flat ~53.1 GB: {:+.2} GB ({:+.1}%)",
        (p.peak_bytes as f64 - 53_070_000_000.0) / 1e9,
        (p.peak_bytes as f64 / 53_070_000_000.0 - 1.0) * 100.0,
    );
    assert!(
        p.peak_bytes > 1_000_000_000,
        "peak {} B is too small for a real 33 B forward",
        p.peak_bytes
    );
}

/// **The 4 s row of the story's table is structurally unrenderable**, not slow and not missing.
///
/// Costs no weights: the refusal is at the geometry gate.
#[test]
fn the_four_second_row_is_structurally_unrenderable() {
    // 4 s is 96 frames. It is not on the lattice, and it is below the model's own floor.
    assert!(resolve_geometry(SWEEP_WIDTH, SWEEP_HEIGHT, 96).is_err());
    // ...and the nearest lattice point at or above 96 is 107, which is still below the floor.
    assert!(resolve_geometry(SWEEP_WIDTH, SWEEP_HEIGHT, 107).is_err());
    // Asking by duration is refused at the range check rather than aligned up into the range.
    let e = align_frames_for_duration(4.0).unwrap_err().to_string();
    assert!(e.contains("outside the model"), "{e}");
    // The floor really is 124 frames = 5.1667 s, and it is what the advertised floor is derived
    // from since sc-17152 — so the nominal 5 s row is unrenderable for the same reason 4 s is.
    assert_eq!(LEGAL_FRAME_COUNTS[0], SMALLEST_LEGAL_FRAMES);
    assert!((f64::from(SMALLEST_LEGAL_FRAMES) / MINIMAX_H3_FPS - 5.1667).abs() < 1e-3);
    assert!((f64::from(MIN_DURATION_SECONDS) - 5.16667).abs() < 1e-4);
    assert!(
        align_frames_for_duration(5.0).is_err(),
        "the nominal 5 s row"
    );
}

/// **The tier axis is real** (sc-17150), and this pins the three properties the sweep's
/// tier-comparability rests on. It replaces the "bf16 only until sc-17150" pin, whose whole job was
/// to fail on this day.
///
/// Costs no weights: every assertion is about the load contract, not about tensors.
#[test]
fn the_tier_axis_exists_and_never_quantizes_at_load() {
    // 1. Both packed tiers are advertised, so a caller can ask for one.
    let quants = mlx_gen_minimax_h3::descriptor()
        .capabilities
        .supported_quants;
    assert!(
        quants.contains(&mlx_gen::gen_core::Quant::Q4)
            && quants.contains(&mlx_gen::gen_core::Quant::Q8),
        "both packed tiers must be advertised, got {quants:?}"
    );

    // 2. A quantize request against a DENSE (unmarked) DiT is refused rather than served by
    //    quantizing 66_280_430_080 bytes at load — the install-time peak this model cannot afford.
    let dir = tempfile::tempdir().unwrap();
    let dit = dir.path().join("transformer");
    std::fs::create_dir_all(&dit).unwrap();
    // A dense component: a config with no `quantization` marker.
    std::fs::write(dit.join("config.json"), "{}").unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    spec.quantize = Some(mlx_gen::gen_core::Quant::Q4);
    let message = match load(&spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a quantized load against a dense DiT must not succeed"),
    };
    assert!(
        message.contains("does not quantize at load"),
        "the refusal must say the model never quantizes at load: {message}"
    );
    assert!(
        message.contains(mlx_gen_minimax_h3::model::DIT_COMPONENT),
        "the refusal must name the component to stage: {message}"
    );

    // 3. A tier that IS staged but disagrees with the request is a hard error, not a silent
    //    downgrade — an unmarked packed component and a dense one are indistinguishable to a
    //    loader that guesses, so the marker is authoritative and disagreement must be loud.
    std::fs::write(
        dit.join("config.json"),
        r#"{"quantization": {"bits": 8, "group_size": 64}}"#,
    )
    .unwrap();
    let message = match load(&spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("q4 requested against a staged q8 tier must not succeed"),
    };
    assert!(
        message.contains("disagrees"),
        "a tier mismatch must be reported as a disagreement: {message}"
    );
}

/// **The declared 15 s ceiling has no lattice point**, and this is the arithmetic the sc-17152
/// decision rests on. Costs no weights.
#[test]
fn the_declared_ceiling_has_no_lattice_point() {
    let top = LEGAL_FRAME_COUNTS[LEGAL_FRAME_COUNTS.len() - 1];
    assert_eq!(top, 345);
    let top_seconds = f64::from(top) / MINIMAX_H3_FPS;
    assert!((top_seconds - 14.375).abs() < 1e-9);
    // The next rung is past the declared ceiling.
    let next = top + 17;
    assert_eq!(next, 362);
    assert!((f64::from(next) / MINIMAX_H3_FPS - 15.0833).abs() < 1e-3);
    // ...so nothing sits in [14.375, 15.0].
    assert!(
        !LEGAL_FRAME_COUNTS
            .iter()
            .any(|&f| { (14.375..=15.0).contains(&(f64::from(f) / MINIMAX_H3_FPS)) && f != top }),
        "no legal frame count may sit strictly inside the gap"
    );
    // And the geometry of the top rung is derivable, i.e. 345 is a real renderable point.
    JointGeometry::new(top, 20, 36).unwrap();
}
