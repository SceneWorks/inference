//! sc-18661 — **rung 3 (`BoundedAttention`) measured inside a real MiniMax-H3 DiT forward.**
//!
//! ```sh
//! MINIMAX_H3_DIT=<tier>/transformer SCENEWORKS_GPU_ID=mlx \
//!   cargo test -p mlx-gen-minimax-h3 --test bounded_attention_real -- --ignored --nocapture \
//!   --test-threads=1
//! ```
//!
//! # Why this harness exists, and why the sibling one could not answer the question
//!
//! `tests/sequence_cost_real.rs` measured bounded attention **in isolation**: q/k/v materialized and
//! pinned alive, one call, no surrounding graph. At the shipped 19 574-row sequence that reported
//! bit-identical output at **+50.3 % peak and ~3x wall**, and its own doc says outright that the
//! figure "is deliberately not a prediction of the in-forward result — the whole MLX-side saving
//! comes from the graph cut, which only exists inside a real DiT forward".
//!
//! That is the gap this file closes. Here q/k/v are intermediates in a 50-block lazy graph, which is
//! the only configuration in which `AttentionBudget::eval_per_chunk` has anything to cut. Z-Image
//! measured **−1.7 %** on its denoise phase from exactly that mechanism (`mlx_gen::attention`), and a
//! magnitude never transfers between families — so it is measured here rather than inherited.
//!
//! # Both axes, because rung 3 has two and only one existed on MLX
//!
//! `gen_core::attention_budget` exposes `query_block_rows` **and** `head_chunks`, and says they
//! "share a score budget, but not a numerical contract". Until sc-18661 `mlx_gen::attention` honored
//! only the query-row axis, and `mlx_gen_sdxl::memory_strategy::ATTENTION_SUPPORT` records the
//! consequence in as many words: "a head-axis implementation would be bit-exact by construction".
//!
//! MiniMax-H3 is the family where that matters most — **56 heads**, against Z-Image's 30 — so a
//! verdict of "rung 3 does nothing here" taken from the query-row axis alone would be a verdict about
//! MLX's kernel coverage rather than about this model. `sdpa_head_budgeted_bhsd` (sc-18661) supplies
//! the missing axis, this harness measures both, and the prediction holds: the head arms below
//! reconstruct the unbounded forward at **exactly zero** relative max-abs at both durations.
//!
//! Three bounded arms per cell, not two. The third is the head axis at a budget derived from the
//! geometry (`B · Sq²`, one head per chunk with the query axis whole), because the shipped 64 Mi
//! operating point is **Z-Image's** and at 56 heads it stops fitting one head above `Sq = 8192` —
//! inside this family's legal duration range, so at 345 frames the 64 Mi head arm falls back to six
//! query blocks. Without the third arm the verdict would be about a budget rather than about the
//! rung.
//!
//! # What it needs, and what it deliberately does not
//!
//! **Only a DiT tier directory** — `MINIMAX_H3_DIT`, e.g. the `q4` transformer of the
//! `SceneWorks/minimax-h3-mlx` rehost. No text encoder, no VAE, no `MINIMAX_H3_SNAPSHOT`.
//!
//! That is not a shortcut, it is the correct instrument. The quantity under test is the **denoise
//! phase's activation peak** as a function of the attention plan. Conditioning and decode are
//! different phases with their own high-water marks (`memory_strategy`'s per-stage table), and a
//! whole-`generate` measurement would put a 14.68–52.80 GB conditioning mark *above* everything this
//! test is trying to resolve, exactly the process-wide-peak-attributed-to-a-component error the
//! provider's module docs open by warning about.
//!
//! The **weights are real**; the latents, the text rows and the timesteps are synthetic. Peak memory
//! does not read tensor values, and both arms of every comparison are driven from the identical
//! inputs, so synthetic drive is sound for a memory measurement and for an equivalence one. What it
//! could not support is an image-quality claim, and none is made.
//!
//! # The tier axis, stated honestly
//!
//! Every cell below is measured at whatever tier `MINIMAX_H3_DIT` names, and the tier is **recorded
//! in the evidence receipt** rather than assumed. The activation transient this rung acts on is
//! tier-independent in size — a packed tier's compute dtype is bf16 (`crate::quant::compute_dtype`),
//! so q/k/v and the attention output are bf16 at every tier — while the resident weight floor is not
//! (40.43 GB bf16 vs 11.63 GB q4). A tier therefore moves the *baseline* the delta is measured
//! against, and moves it in the direction that makes a saving **harder** to see, never easier. q4 is
//! consequently the tier on which an activation-side rung has its best chance, and a null result
//! there bounds the other two from above.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::{Array, Dtype};

use mlx_gen::attention::{AttentionBudget, AttentionChunkAxis};
use mlx_gen_minimax_h3::denoise::{
    adaln_schedule, JointSchedule, JointStep, JointVelocity, PackedLayout,
};
use mlx_gen_minimax_h3::dit::layers::attention_probe;
use mlx_gen_minimax_h3::dit::{AdaLnResidency, JointDit, MiniMaxH3Dit, TimestepSchedule};
use mlx_gen_minimax_h3::pipeline::{resolve_geometry, t2va_layout, PATCH_SIZE};

/// Prompt rows the packed sequence carries. The shipped conditioning length; it sets the text block
/// of every sequence below and is held constant so `frames` is the only moving axis.
const NUM_TEXT_TOKENS: i32 = 512;

/// The measured canvas. 576x320 is the geometry sc-17152's duration sweep used, so the sequence
/// lengths here line up with an already-published curve instead of introducing a third geometry.
const WIDTH: u32 = 576;
const HEIGHT: u32 = 320;

/// Model evaluations per arm. The attention plan is a per-forward property and every step runs the
/// identical 50-block stack, so two evaluations distinguish a first-call allocator artifact from the
/// steady state without paying for a full schedule.
const EVALS: usize = 2;

fn dit_dir() -> Option<PathBuf> {
    let raw = std::env::var("MINIMAX_H3_DIT").unwrap_or_default();
    (!raw.trim().is_empty()).then(|| PathBuf::from(raw.trim()))
}

/// Peak MLX bytes a closure moves, from a cleared cache and a reset counter.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, u64) {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let out = f();
    (out, mlx_rs::memory::get_peak_memory() as u64)
}

/// Deterministic `[shape]` bf16 drive, from a fixed key so both arms see identical inputs.
fn drive(shape: &[i32], seed: u64) -> Array {
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// One measured arm: peak bytes, fastest evaluation wall, the last step's video output, and what the
/// probe says the attention actually did.
struct Arm {
    peak: u64,
    fastest: f64,
    video: Array,
    head_chunks: usize,
    query_chunks: usize,
    seq: usize,
}

/// Everything one geometry needs, loaded once and reused across arms.
struct Rig {
    layout: PackedLayout,
    schedule: TimestepSchedule,
    joint: JointSchedule,
    context: Array,
    video_rows: Array,
    audio_rows: Array,
    seq: i32,
}

impl Rig {
    fn build(frames: i32, cfg: &mlx_gen_minimax_h3::MiniMaxH3DitConfig) -> Self {
        let geometry = resolve_geometry(WIDTH, HEIGHT, frames).expect("legal geometry");
        let layout = t2va_layout(&geometry, NUM_TEXT_TOKENS, PATCH_SIZE).expect("layout");
        // `num_inference_steps` counts the terminal sigma, so `EVALS + 1` requested steps is exactly
        // `EVALS` model evaluations.
        let joint = JointSchedule::new(EVALS + 1).expect("joint schedule");
        assert_eq!(joint.num_evals(), EVALS);
        let schedule = adaln_schedule(&joint).expect("adaln schedule");

        let seq = layout.seq_len();
        let video_features = cfg.in_channels * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2];
        let video_rows = drive(
            &[1, layout.video_indices().len() as i32, video_features],
            101,
        );
        let audio_rows = drive(
            &[
                1,
                layout.audio_indices().len() as i32,
                cfg.audio_in_channels,
            ],
            102,
        );
        let context = drive(&[1, NUM_TEXT_TOKENS, cfg.text_dim], 103);
        mlx_rs::transforms::eval([&video_rows, &audio_rows, &context]).unwrap();

        Self {
            layout,
            schedule,
            joint,
            context,
            video_rows,
            audio_rows,
            seq,
        }
    }

    /// Run `EVALS` model evaluations under one attention plan, measuring the whole window.
    ///
    /// The DiT is **reloaded per arm on purpose**: `JointDit::new` consumes the model and performs
    /// the AdaLN precompute-and-evict, so a shared instance would let one arm's evicted state and
    /// its 3.87 GB retained modulation table sit inside the next arm's measurement window.
    fn arm(&self, dir: &std::path::Path, budget: AttentionBudget, axis: AttentionChunkAxis) -> Arm {
        let dit = MiniMaxH3Dit::load_dir(dir, Dtype::Bfloat16).expect("dit loads");
        let mut model = JointDit::new(
            dit,
            self.layout.clone(),
            &self.context,
            self.schedule.clone(),
            AdaLnResidency::PrecomputeAndEvict,
        )
        .expect("joint dit");
        model.set_bounded_attention(budget, axis);

        // The load and the precompute sit OUTSIDE the window. They are the same work in every arm
        // and their high-water mark (12.66 GB at q4, per the provider's per-stage table) would sit
        // above the activation transient this test resolves and flatten every cell to one number.
        mlx_rs::memory::clear_cache();

        let ((video, fastest), peak) = peak_of(|| {
            let mut fastest = f64::MAX;
            let mut last: Option<Array> = None;
            for i in 0..EVALS {
                let adaln_indices = self
                    .schedule
                    .adaln_indices(i, self.layout.row_classes(), self.layout.token_tags())
                    .expect("adaln indices");
                let row_timesteps = PackedLayout::row_timesteps(
                    self.joint.video().timesteps()[i],
                    self.joint.audio().timesteps()[i],
                );
                attention_probe::reset();
                let started = Instant::now();
                let (v, a) = model
                    .forward(&JointStep {
                        index: i,
                        layout: &self.layout,
                        video_rows: &self.video_rows,
                        audio_rows: &self.audio_rows,
                        adaln_indices: &adaln_indices,
                        row_timesteps,
                    })
                    .expect("forward");
                // MLX is lazy: without this the timer measures graph construction and the peak
                // counter never sees the stack run at all.
                mlx_rs::transforms::eval([&v, &a]).unwrap();
                fastest = fastest.min(started.elapsed().as_secs_f64());
                last = Some(v);
            }
            (last.expect("at least one evaluation"), fastest)
        });

        Arm {
            peak,
            fastest,
            video,
            head_chunks: attention_probe::last_head_chunks(),
            query_chunks: attention_probe::last_query_chunks(),
            seq: attention_probe::last_seq(),
        }
    }
}

/// Peak-relative max-abs difference of `b` against `a`, both cast to f32 so the comparison is not
/// quantized to bf16's ~3 decimal digits and hidden.
///
/// Deliberately **not** a norm, a cosine or a checksum: all three are dominated by the bulk of an
/// already-agreeing tensor and would pass on a chunk boundary that dropped rows entirely.
fn relative_max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let max_abs: f32 = a
        .subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    let scale: f32 = a.abs().unwrap().max(None).unwrap().item();
    assert!(scale > 1e-3, "the reference arm is degenerate ({scale})");
    max_abs / scale
}

/// **The in-forward measurement, and the verdict input for rung 3's declaration.**
///
/// Three arms per duration — unbounded, head-chunked, query-row-chunked — at each end of the frame
/// lattice, so no cell is extrapolated across the duration axis (`frames` is an exact-match evidence
/// key; `mlx_fit_gate.rs:1381`).
///
/// The engagement premise is asserted **before** the peaks are compared, from the shared planner's
/// own arithmetic through the always-on `attention_probe`. Without it every comparison here is
/// vacuous: "chunked == unbounded" is trivially true when the chunking never engaged, and a peak
/// delta of 0 % would read as "the rung is free" rather than "the rung did not run".
#[test]
#[ignore = "sc-18661: needs MINIMAX_H3_DIT (a tier's transformer/) + Metal; loads the 50-block DiT once per arm"]
fn the_in_forward_graph_cut_is_measured_on_both_chunk_axes() {
    let Some(dir) = dit_dir() else {
        panic!("MINIMAX_H3_DIT=<tier>/transformer must point at a staged MiniMax-H3 DiT");
    };
    let tier = common::describe_dit_tier(&dir);
    let cfg = mlx_gen_minimax_h3::MiniMaxH3DitConfig::default();
    let heads = cfg.num_attention_heads;

    let mut rows = Vec::new();
    for frames in [124i32, 345] {
        let rig = Rig::build(frames, &cfg);
        eprintln!(
            "\n=== {tier} | {WIDTH}x{HEIGHT} | {frames} frames | packed seq {} | {heads} heads ===",
            rig.seq
        );

        let unbounded = rig.arm(&dir, AttentionBudget::UNBOUNDED, AttentionChunkAxis::Heads);
        assert_eq!(
            (unbounded.head_chunks, unbounded.query_chunks),
            (1, 1),
            "the unbounded arm must run one un-chunked fused call"
        );
        assert_eq!(
            unbounded.seq as i32, rig.seq,
            "the probe saw a different sequence than the layout declares"
        );

        // **Three bounded arms, because "the rung" is not one configuration.**
        //
        // `SHIPPED` is Z-Image's published 64 Mi operating point on each axis. `bit_exact_heads` is
        // the budget derived from *this* geometry — one head's score domain, `B·Sq²` — which is the
        // largest budget that still forces a head split and the smallest that keeps the query axis
        // whole inside it. It exists because 64 Mi does NOT stay bit-exact across this family's
        // envelope: a single head needs `Sq² <= budget`, so at 64 Mi the head axis runs out of
        // exactness above `Sq = 8192` — inside the legal duration range. Measuring only the shipped
        // budget would attribute that budget's behaviour to the rung.
        let one_head_scores = (rig.seq as u64).saturating_mul(rig.seq as u64);
        let bit_exact = AttentionBudget::from_score_elements(one_head_scores, true);

        let by_heads = rig.arm(
            &dir,
            AttentionBudget::CONSTRAINED,
            AttentionChunkAxis::Heads,
        );
        let by_rows = rig.arm(
            &dir,
            AttentionBudget::CONSTRAINED,
            AttentionChunkAxis::QueryRows,
        );
        let exact_heads = rig.arm(&dir, bit_exact, AttentionChunkAxis::Heads);

        // The premise. Every bounded arm must actually have split, on its own axis and not the
        // other's, or the comparisons below compare the fast path with itself.
        assert!(
            by_heads.head_chunks > 1,
            "the head axis did not split at seq {} with {heads} heads — every assertion below \
             would be vacuous",
            rig.seq
        );
        assert!(
            by_rows.query_chunks > 1,
            "the query-row axis did not split at seq {}",
            rig.seq
        );
        assert_eq!(
            by_rows.head_chunks, 1,
            "the query-row arm must not have split heads"
        );
        assert_eq!(
            (exact_heads.head_chunks, exact_heads.query_chunks),
            (heads as usize, 1),
            "the geometry-derived budget must split into exactly one head per chunk with the query \
             axis whole — that combination is the entire basis of the bit-exactness claim"
        );

        let head_rel = relative_max_abs(&unbounded.video, &by_heads.video);
        let row_rel = relative_max_abs(&unbounded.video, &by_rows.video);
        let exact_rel = relative_max_abs(&unbounded.video, &exact_heads.video);

        let report = |label: &str, arm: &Arm, rel: f32| {
            eprintln!(
                "  {label:<22} peak {:8.4} GB  fastest {:7.3} s  {:>3} head x {:>3} row  \
                 {:+6.2}% peak  {:+6.1}% wall  rel|Δ| {rel:.3e}",
                arm.peak as f64 / 1e9,
                arm.fastest,
                arm.head_chunks,
                arm.query_chunks,
                (arm.peak as f64 / unbounded.peak as f64 - 1.0) * 100.0,
                (arm.fastest / unbounded.fastest - 1.0) * 100.0
            );
        };
        eprintln!(
            "  {:<22} peak {:8.4} GB  fastest {:7.3} s    1 head x   1 row",
            "unbounded",
            unbounded.peak as f64 / 1e9,
            unbounded.fastest
        );
        report("heads @ 64Mi", &by_heads, head_rel);
        report("query rows @ 64Mi", &by_rows, row_rel);
        report("heads @ one-head", &exact_heads, exact_rel);

        // **The head axis is bit-exact when it does not fall back, and that is asserted rather than
        // described.** One head per chunk with the query axis whole preserves the query GEMM's `M`,
        // so MLX dispatches the identical Metal specialization; anything above zero here is a wiring
        // error, not ULP drift.
        assert_eq!(
            exact_rel, 0.0,
            "one-head-per-chunk attention moved the DiT output by {exact_rel:e} relative — the \
             configuration whose whole justification is bit identity is not bit-exact"
        );
        // The 64 Mi head arm inherits the query axis's weaker contract exactly when it had to fall
        // back to it, which is a property of the budget against the geometry rather than of the axis.
        if by_heads.query_chunks == 1 {
            assert_eq!(
                head_rel, 0.0,
                "the head axis did not fall back to query rows at seq {} yet still moved the \
                 output by {head_rel:e}",
                rig.seq
            );
        }
        // The query-row axis is only tolerance-equivalent (`mlx_gen::attention`'s documented
        // sc-2338 parity class), so it is bounded rather than pinned to zero.
        for (label, rel) in [("query rows", row_rel), ("heads @ 64Mi", head_rel)] {
            assert!(
                rel < 1e-2,
                "{label} moved the DiT output by {rel:e} relative — that is a wiring error, not \
                 ULP drift from narrowing the GEMM's M"
            );
        }

        rows.push(common::BoundedAttentionCell {
            tier: tier.clone(),
            width: WIDTH,
            height: HEIGHT,
            frames,
            seq: rig.seq,
            heads,
            unbounded_peak_bytes: unbounded.peak,
            head_chunked_peak_bytes: by_heads.peak,
            query_chunked_peak_bytes: by_rows.peak,
            bit_exact_head_peak_bytes: exact_heads.peak,
            bit_exact_head_budget: one_head_scores,
            head_chunks: by_heads.head_chunks,
            head_query_row_blocks: by_heads.query_chunks,
            query_row_blocks: by_rows.query_chunks,
            unbounded_fastest_eval_s: unbounded.fastest,
            head_chunked_fastest_eval_s: by_heads.fastest,
            query_chunked_fastest_eval_s: by_rows.fastest,
            bit_exact_head_fastest_eval_s: exact_heads.fastest,
            head_relative_max_abs: head_rel,
            query_relative_max_abs: row_rel,
            bit_exact_head_relative_max_abs: exact_rel,
        });
    }

    // Both duration cells must exist: a single-row table would be the exact "never a single number"
    // failure AC1 names, and would license extrapolating across an axis the fit gate exact-matches.
    assert_eq!(rows.len(), 2, "both duration cells must be measured");
    common::write_bounded_attention_evidence(&rows);

    // **The declared-versus-measured guard, and the one that would force a re-declaration.**
    //
    // `memory_strategy::RUNG3_MEASURED_PEAK_DELTAS` is what the contract's inapplicability reason is
    // built from, so a drift between the two would leave the provider publishing a justification its
    // own harness no longer reproduces. Two claims, and the second is the load-bearing one:
    //
    // 1. every declared cell is within a band of what was just measured — the numbers are current;
    // 2. every measured delta is **positive** — the rung costs peak. A future MLX bump that made
    //    chunking win here would turn this red, which is exactly right: the verdict would have to be
    //    re-taken rather than inherited.
    //
    // The band is generous (0.5 percentage points) because a peak counter on a shared machine is not
    // a bit-exact instrument, and because the claim under test is a **sign and an order of
    // magnitude**, not a third decimal place. It is still far tighter than the gap to a saving.
    const DELTA_BAND: f64 = 0.005;
    let declared_cells = mlx_gen_minimax_h3::memory_strategy::rung3_cells_for_tier(&tier);
    assert_eq!(
        declared_cells.len(),
        rows.len(),
        "tier {tier:?} has {} declared cells against {} measured — a tier measured but not declared \
         leaves the contract's inapplicability reason describing a subset of what was run",
        declared_cells.len(),
        rows.len()
    );
    for (cell, declared) in rows.iter().zip(declared_cells) {
        let (declared_frames, declared_seq, d_heads, d_rows, d_one_head) = declared;
        assert_eq!(
            (cell.frames, cell.seq),
            (declared_frames, declared_seq),
            "the declared cell order must match the measured one"
        );
        let base = cell.unbounded_peak_bytes as f64;
        for (label, bounded, declared_delta) in [
            ("heads @ 64Mi", cell.head_chunked_peak_bytes, d_heads),
            ("query rows @ 64Mi", cell.query_chunked_peak_bytes, d_rows),
            (
                "heads @ one-head",
                cell.bit_exact_head_peak_bytes,
                d_one_head,
            ),
        ] {
            let measured = bounded as f64 / base - 1.0;
            assert!(
                measured > 0.0,
                "{label} at {} frames SAVED peak ({:+.3}%) — rung 3 is declared \
                 StructurallyNotApplicable on the grounds that it cannot. Re-take the verdict.",
                cell.frames,
                measured * 100.0
            );
            assert!(
                (measured - declared_delta).abs() < DELTA_BAND,
                "{label} at {} frames measured {:+.3}% against the declared {:+.3}% — \
                 memory_strategy::RUNG3_MEASURED_PEAK_DELTAS is stale, and the contract's \
                 inapplicability reason is built from it",
                cell.frames,
                measured * 100.0,
                declared_delta * 100.0
            );
        }
    }
}
