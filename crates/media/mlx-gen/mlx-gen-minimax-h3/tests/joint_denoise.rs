//! Joint audio+video denoise parity (sc-17146) against `tools/dump_minimax_h3_av_denoise.py`.
//!
//! The golden is a whole **2-step** loop produced by the official diffusers `MiniMaxH3Scheduler` —
//! two instances loaded from the *published* `scheduler/` and `audio_scheduler/` configs, so the
//! 12.0 / 3.0 pair is read from the same bytes production reads rather than typed twice
//! (`src/layout.rs` rule 3).
//!
//! # What is gated, and on what
//!
//! `rel` (relative max-abs-diff) is the gate everywhere, never a norm, a checksum or cosine. Two
//! shipped defects and one near-miss in this epic all held cosine >= 0.99 or left norms unchanged.
//! Every mutation this file claims to catch has a **measured** control in the fixture's metadata,
//! asserted above `MUTATION_FLOOR`, so "this test would catch it" is evidence rather than a hope.
//!
//! # The indices are deliberately NOT compared
//!
//! The reference builds a per-step table of the timesteps actually present and sorts it ascending;
//! this port builds **one global table across the whole run** in first-appearance order, because
//! sc-17145's precompute-and-evict needs a single table. The index tensors are therefore not
//! comparable and agreeing on them would be the wrong contract. What is comparable — and what the
//! AdaLN row actually depends on — is the **resolved per-row timestep value**, which
//! `row_timesteps_resolve_to_the_reference_values` compares element by element.

mod common;

use std::time::Instant;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_gen_minimax_h3::denoise::{
    adaln_schedule, denoise_av, DenoiseModality, JointGeometry, JointSchedule, JointStep,
    JointVelocity, PackedLayout, RowClass, SigmaSchedule, NUM_ROW_CLASSES, TEXT_TAG,
};
use mlx_gen_minimax_h3::dit::positions::KeyframeAnchor;

use common::{assert_parity, cosine, l2_norm, rel, std_dev};

/// The committed joint-denoise fixture.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_denoise.safetensors"
);

/// The mlx-gen house parity tolerance.
const TOL: f32 = 1e-2;

/// A mutation must clear the numeric noise floor, or "the output moved" is just jitter.
const MUTATION_FLOOR: f32 = 1e-2;

/// The layout the fixture was dumped at — the shortest legal render (124 frames ⇒ 37 latent
/// frames, 207 stereo audio latents) on a 4×6 latent at patch `[1, 2, 2]`, with both keyframe
/// anchors.
const NUM_FRAMES: i32 = 124;
const LATENT_HEIGHT: i32 = 4;
const LATENT_WIDTH: i32 = 6;
const PATCH: [i32; 3] = [1, 2, 2];
const NUM_TEXT_TOKENS: usize = 5;
const AUDIO_CHANNELS: i32 = 2;
/// 3 requested steps is **2** model evaluations — the terminal sigma is inside the count.
const NUM_INFERENCE_STEPS: usize = 3;
const NUM_EVALS: usize = 2;

fn fixture() -> Weights {
    Weights::from_file(FIXTURE).unwrap()
}

fn layout() -> PackedLayout {
    PackedLayout::build(
        JointGeometry::new(NUM_FRAMES, LATENT_HEIGHT, LATENT_WIDTH).unwrap(),
        PATCH,
        &[TEXT_TAG; NUM_TEXT_TOKENS],
        AUDIO_CHANNELS,
        &[KeyframeAnchor::First, KeyframeAnchor::Last],
    )
    .unwrap()
}

/// `[rows, F]` fixture tensor → the `[1, rows, F]` the loop takes.
fn batched(f: &Weights, key: &str) -> Array {
    let t = f.require(key).unwrap();
    let s = t.shape();
    t.reshape(&[1, s[0], s[1]]).unwrap()
}

fn floats(f: &Weights, key: &str) -> Vec<f32> {
    f.require(key)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn ints(f: &Weights, key: &str) -> Vec<i32> {
    floats(f, key).into_iter().map(|v| v as i32).collect()
}

fn meta(f: &Weights, key: &str) -> String {
    f.metadata(key)
        .unwrap_or_else(|| {
            panic!(
                "fixture metadata is missing `{key}`; re-run \
                 tools/dump_minimax_h3_av_denoise.py against diffusers `main` (a fixture with no \
                 provenance cannot be shown to come from the converted-checkpoint path — sc-18740)"
            )
        })
        .to_string()
}

/// Replays the fixture's per-step velocities, counting calls and recording what it was handed.
///
/// The DiT's input/output projections are sc-17147's, so the loop is exercised against the
/// reference's own velocity table rather than a model — which makes this a test of exactly what
/// sc-17146 owns.
struct Replay {
    video: Vec<Array>,
    audio: Vec<Array>,
    calls: usize,
    batches: Vec<i32>,
    seen_video: Vec<Array>,
    seen_audio: Vec<Array>,
    seen_adaln: Vec<Array>,
    seen_row_timesteps: Vec<[f32; NUM_ROW_CLASSES]>,
}

impl Replay {
    fn from_fixture(f: &Weights) -> Self {
        Self {
            video: (0..NUM_EVALS)
                .map(|i| batched(f, &format!("in.video_velocity.{i}")))
                .collect(),
            audio: (0..NUM_EVALS)
                .map(|i| batched(f, &format!("in.audio_velocity.{i}")))
                .collect(),
            calls: 0,
            batches: vec![],
            seen_video: vec![],
            seen_audio: vec![],
            seen_adaln: vec![],
            seen_row_timesteps: vec![],
        }
    }
}

impl JointVelocity for Replay {
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
        self.calls += 1;
        self.batches.push(step.video_rows.shape()[0]);
        self.seen_video.push(step.video_rows.clone());
        self.seen_audio.push(step.audio_rows.clone());
        self.seen_adaln.push(step.adaln_indices.clone());
        self.seen_row_timesteps.push(step.row_timesteps);
        Ok((
            self.video[step.index].clone(),
            self.audio[step.index].clone(),
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------------------------

/// The fixture came from the converted-checkpoint path, and every mutation it claims to gate is
/// measurably above the noise floor.
#[test]
fn fixture_provenance_records_the_converted_path() {
    let f = fixture();
    assert_eq!(meta(&f, "provenance"), "converted-checkpoint");
    assert_eq!(meta(&f, "reference"), "diffusers.MiniMaxH3Scheduler");
    assert_eq!(
        meta(&f, "layout_reference"),
        "diffusers.MiniMaxH3PrepareLayoutStep.build_packed_sequence"
    );
    assert_eq!(
        meta(&f, "scheduler_source"),
        "published scheduler/ + audio_scheduler/ configs",
        "the shifts must be READ from the checkpoint, not typed into the generator"
    );
    // The shifts the published configs actually carry.
    assert_eq!(meta(&f, "video_sigma_shift"), "12.0");
    assert_eq!(meta(&f, "audio_sigma_shift"), "3.0");
    assert_eq!(meta(&f, "keyframe_noise_aug"), "0.999");
    assert_eq!(meta(&f, "num_evals"), "2");
    assert_eq!(meta(&f, "num_frames"), "124");
    assert_eq!(meta(&f, "story"), "sc-17146");

    for key in [
        "sigma_swap_rel",
        "one_shift_for_both_rel",
        "double_shift_rel",
        "velocity_sign_rel",
    ] {
        let v: f32 = meta(&f, key).parse().unwrap();
        assert!(
            v > MUTATION_FLOOR,
            "the generator measured {key} as only {v:.3e} — it would not be gateable"
        );
    }
    println!(
        "fixture provenance: {} {} (swap {} / both {} / double {} / sign {})",
        meta(&f, "reference"),
        meta(&f, "reference_version"),
        meta(&f, "sigma_swap_rel"),
        meta(&f, "one_shift_for_both_rel"),
        meta(&f, "double_shift_rel"),
        meta(&f, "velocity_sign_rel"),
    );
}

// ---------------------------------------------------------------------------------------------
// The two sigma schedules
// ---------------------------------------------------------------------------------------------

/// Both grids, both timestep lists, against the published shifts — **bitwise**.
///
/// Bitwise rather than within a tolerance because the port reproduces `torch.linspace`'s
/// halfway-split FMA exactly, and because [`mlx_gen_minimax_h3::dit::adaln::TimestepSchedule`]
/// resolves timesteps by bit pattern: a schedule that is merely close would key a different AdaLN
/// cache.
#[test]
fn the_two_sigma_grids_match_the_published_shifts() {
    let f = fixture();
    let joint = JointSchedule::new(NUM_INFERENCE_STEPS).unwrap();
    assert_eq!(joint.num_evals(), NUM_EVALS);

    for (modality, sigma_key, t_key) in [
        (
            DenoiseModality::Video,
            "out.video_sigmas",
            "out.video_timesteps",
        ),
        (
            DenoiseModality::Audio,
            "out.audio_sigmas",
            "out.audio_timesteps",
        ),
    ] {
        let s = joint.of(modality);
        let want_sigmas = floats(&f, sigma_key);
        let want_timesteps = floats(&f, t_key);
        assert_eq!(
            s.sigmas().len(),
            want_sigmas.len(),
            "{}: grid length",
            modality.name()
        );
        for (i, (got, want)) in s.sigmas().iter().zip(&want_sigmas).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{} sigma[{i}]: {got} != {want}",
                modality.name()
            );
        }
        for (i, (got, want)) in s.timesteps().iter().zip(&want_timesteps).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{} timestep[{i}]: {got} != {want}",
                modality.name()
            );
        }
        println!("  {} sigmas {:?}", modality.name(), s.sigmas());
    }

    // The two grids are materially different — which is what a swap would exchange.
    let peak = joint
        .video()
        .sigmas()
        .iter()
        .zip(joint.audio().sigmas())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let control: f32 = meta(&f, "sigma_swap_rel").parse().unwrap();
    assert!(peak > MUTATION_FLOOR, "grids differ by {peak:.3e}");
    println!("  |video sigma - audio sigma| peak {peak:.3e}, measured swap control {control:.3e}");
}

/// **The swap and one-shift-for-both mutations, at loop level.** Each produces a materially
/// different final latent, measured against the same control the generator recorded.
#[test]
fn swapped_or_shared_shifts_change_the_output() {
    let f = fixture();
    let want_video = batched(&f, "out.video_latents.1");
    let want_audio = batched(&f, "out.audio_latents.1");

    let correct = run(&f, JointSchedule::new(NUM_INFERENCE_STEPS).unwrap());
    assert_parity(&correct.0, &want_video, TOL, "video, correct shifts");
    assert_parity(&correct.1, &want_audio, TOL, "audio, correct shifts");

    // Swapped: video on 3.0, audio on 12.0.
    let swapped = run(
        &f,
        JointSchedule::with_shifts(NUM_INFERENCE_STEPS, 3.0, 12.0).unwrap(),
    );
    let (sv, _) = rel(&swapped.0, &want_video);
    let (sa, _) = rel(&swapped.1, &want_audio);
    assert!(
        sv.max(sa) > MUTATION_FLOOR,
        "a swapped pair of shifts must move the output, got {:.3e}",
        sv.max(sa)
    );

    // One shift for both: audio denoised on the video curve.
    let both = run(
        &f,
        JointSchedule::with_shifts(NUM_INFERENCE_STEPS, 12.0, 12.0).unwrap(),
    );
    let (bv, _) = rel(&both.0, &want_video);
    let (ba, _) = rel(&both.1, &want_audio);
    assert!(
        bv < 1e-6,
        "the video half is unchanged by the audio shift, got {bv:.3e}"
    );
    assert!(
        ba > MUTATION_FLOOR,
        "…while the audio half must move, got {ba:.3e}"
    );

    // Norm and cosine are reported, never gated: they are the metrics that miss this class.
    println!(
        "  swap rel {:.3e}/{:.3e}  both-audio rel {ba:.3e}  cosine {:.4}  |L2| {:.3} vs {:.3}",
        sv,
        sa,
        cosine(&both.1, &want_audio),
        l2_norm(&both.1),
        l2_norm(&want_audio)
    );
    assert!(
        std_dev(&want_video) > 1e-4,
        "the golden must not be constant"
    );
}

/// A double-applied shift is a different grid, and the generator measured how different.
#[test]
fn a_double_applied_shift_changes_the_grid() {
    let f = fixture();
    let once = SigmaSchedule::new(DenoiseModality::Video, NUM_INFERENCE_STEPS).unwrap();
    // The mutation: shift the already-shifted grid again.
    let twice: Vec<f32> = once
        .sigmas()
        .iter()
        .map(|&s| mlx_gen_minimax_h3::denoise::shift_sigma(s, 12.0))
        .collect();
    let peak = once
        .sigmas()
        .iter()
        .zip(&twice)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let control: f32 = meta(&f, "double_shift_rel").parse().unwrap();
    assert!(
        peak > MUTATION_FLOOR,
        "a doubled shift must be observable, got {peak:.3e} (generator measured {control:.3e})"
    );
    // ...and the endpoints cannot see it, which is why only a value comparison works.
    assert_eq!(once.sigmas()[0], twice[0]);
    assert_eq!(*once.sigmas().last().unwrap(), *twice.last().unwrap());
}

// ---------------------------------------------------------------------------------------------
// The packed layout
// ---------------------------------------------------------------------------------------------

/// Row order, the three index tensors, the tags and the rotary grid, against the reference's own
/// `build_packed_sequence`.
#[test]
fn the_packed_layout_matches_the_reference() {
    let f = fixture();
    let l = layout();

    assert_eq!(
        l.seq_len(),
        f.require("layout.position_ids").unwrap().shape()[0]
    );
    assert_eq!(l.video_indices(), ints(&f, "layout.video_indices"));
    assert_eq!(l.audio_indices(), ints(&f, "layout.audio_indices"));
    assert_eq!(l.text_indices(), ints(&f, "layout.text_indices"));
    assert_eq!(
        l.token_tags().as_slice::<i32>(),
        ints(&f, "layout.token_tags")
    );
    assert_parity(
        l.position_ids(),
        f.require("layout.position_ids").unwrap(),
        1e-5,
        "position_ids",
    );

    // `video_indices` skips the audio block — assert the discontiguity is real in the golden too,
    // so this is not vacuous on a regenerated fixture.
    let v = ints(&f, "layout.video_indices");
    let cond = l.num_condition_video_rows() as usize;
    assert_eq!(cond, 12);
    assert!(
        v[cond - 1] + 1 != v[cond],
        "the reference's own video_indices must straddle the audio block"
    );
    println!(
        "  seq_len {} = {} text + {cond} anchors + {} audio + {} video",
        l.seq_len(),
        l.num_text_tokens(),
        l.audio_indices().len(),
        l.generated_video_rows().len()
    );
}

/// **Text rows carry the video timestep**, and every other row class resolves to the reference's
/// own per-row value.
///
/// The port keys one global table in first-appearance order while the reference sorts a per-step
/// one, so the *indices* differ by construction; the resolved *values* must not.
#[test]
fn row_timesteps_resolve_to_the_reference_values() {
    let f = fixture();
    let l = layout();
    let joint = JointSchedule::new(NUM_INFERENCE_STEPS).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    let classes = l.row_classes().as_slice::<i32>().to_vec();

    for step in 0..NUM_EVALS {
        let want = floats(&f, &format!("out.row_timesteps.{step}"));
        let declared = adaln.step_timesteps(step).unwrap();
        assert_eq!(declared.len(), NUM_ROW_CLASSES);
        let got: Vec<f32> = classes.iter().map(|&c| declared[c as usize]).collect();
        assert_eq!(got.len(), want.len());
        for (row, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "step {step} row {row} (class {}): {g} != {w}",
                classes[row]
            );
        }

        // The named claim, stated separately so a regression reads clearly.
        let video_t = joint.video().timesteps()[step];
        for &row in l.text_indices() {
            assert_eq!(
                want[row as usize], video_t,
                "step {step}: text row {row} must sit at the video timestep, not 1.0"
            );
        }
        for &row in l.generated_audio_rows() {
            assert_eq!(want[row as usize], joint.audio().timesteps()[step]);
        }
        for &row in &l.video_indices()[..l.num_condition_video_rows() as usize] {
            assert_eq!(want[row as usize], video_t.max(0.999));
        }
    }

    // Pinning text at a clean 1.0 — what sc-17145's docs and the sc-17242 spike both describe —
    // would change exactly the text rows, and the generator counted them.
    let changed: usize = meta(&f, "text_rows_at_video_timestep").parse().unwrap();
    assert_eq!(changed, NUM_TEXT_TOKENS);
    let want0 = floats(&f, "out.row_timesteps.0");
    assert!(
        l.text_indices().iter().all(|&r| want0[r as usize] != 1.0),
        "no fl2va row sits at t = 1.0"
    );
}

/// The AdaLN gather stays in bounds at every step, and addresses more than one table block.
#[test]
fn the_adaln_indices_stay_in_bounds() {
    let f = fixture();
    let l = layout();
    let joint = JointSchedule::new(NUM_INFERENCE_STEPS).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    let mut model = Replay::from_fixture(&f);
    run_with(&l, &joint, &adaln, &f, &mut model);

    assert_eq!(model.seen_adaln.len(), NUM_EVALS);
    for (step, idx) in model.seen_adaln.iter().enumerate() {
        assert_eq!(idx.shape(), &[l.seq_len()]);
        let min: i32 = idx.min(None).unwrap().item();
        let max: i32 = idx.max(None).unwrap().item();
        assert!(min >= 0 && max < adaln.modulation_rows(), "step {step}");
        let distinct: std::collections::BTreeSet<i32> =
            idx.as_slice::<i32>().iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "step {step} addresses only one table row — the timestep axis would be inert"
        );
    }
    // The video timestep moves between the two steps, so the video rows' block must too.
    let a = model.seen_adaln[0].as_slice::<i32>().to_vec();
    let b = model.seen_adaln[1].as_slice::<i32>().to_vec();
    let row = l.generated_video_rows()[0] as usize;
    assert_ne!(a[row], b[row], "the video rows must re-address per step");
}

// ---------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------

fn run_with(
    l: &PackedLayout,
    joint: &JointSchedule,
    adaln: &mlx_gen_minimax_h3::dit::adaln::TimestepSchedule,
    f: &Weights,
    model: &mut Replay,
) -> (Array, Array) {
    denoise_av(
        model,
        l,
        joint,
        adaln,
        &batched(f, "in.video_latents"),
        &batched(f, "in.audio_latents"),
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap()
}

fn run(f: &Weights, joint: JointSchedule) -> (Array, Array) {
    let l = layout();
    let adaln = adaln_schedule(&joint).unwrap();
    let mut model = Replay::from_fixture(f);
    run_with(&l, &joint, &adaln, f, &mut model)
}

/// **The parity claim.** Both modalities, after each of the two steps.
#[test]
fn the_joint_loop_matches_the_reference_at_two_steps() {
    let f = fixture();
    let l = layout();
    let joint = JointSchedule::new(NUM_INFERENCE_STEPS).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    let mut model = Replay::from_fixture(&f);
    let (video, audio) = run_with(&l, &joint, &adaln, &f, &mut model);

    // Step 0's output is what the model was handed at step 1.
    assert_parity(
        &model.seen_video[1],
        &batched(&f, "out.video_latents.0"),
        TOL,
        "video after step 0",
    );
    assert_parity(
        &model.seen_audio[1],
        &batched(&f, "out.audio_latents.0"),
        TOL,
        "audio after step 0",
    );
    // ...and the return is step 1's.
    assert_parity(
        &video,
        &batched(&f, "out.video_latents.1"),
        TOL,
        "video after step 1",
    );
    assert_parity(
        &audio,
        &batched(&f, "out.audio_latents.1"),
        TOL,
        "audio after step 1",
    );

    let (vp, vm) = rel(&video, &batched(&f, "out.video_latents.1"));
    let (ap, am) = rel(&audio, &batched(&f, "out.audio_latents.1"));
    println!("  video rel peak {vp:.3e} mean {vm:.3e}; audio rel peak {ap:.3e} mean {am:.3e}");

    // The row timesteps the model was handed are this step's four classes.
    for step in 0..NUM_EVALS {
        assert_eq!(
            model.seen_row_timesteps[step][RowClass::Video as usize],
            joint.video().timesteps()[step]
        );
        assert_eq!(
            model.seen_row_timesteps[step][RowClass::Audio as usize],
            joint.audio().timesteps()[step]
        );
    }
}

/// **One forward per step, batch of one — the no-CFG pin.**
///
/// A reintroduced classifier-free-guidance path doubles either the call count (a second
/// unconditional pass) or the batch axis (a `cat([uncond, cond])`). Both are failures here.
#[test]
fn one_forward_per_step_and_no_cfg() {
    let f = fixture();
    let l = layout();
    let joint = JointSchedule::new(NUM_INFERENCE_STEPS).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    let mut model = Replay::from_fixture(&f);
    let mut steps = vec![];
    denoise_av(
        &mut model,
        &l,
        &joint,
        &adaln,
        &batched(&f, "in.video_latents"),
        &batched(&f, "in.audio_latents"),
        &CancelFlag::default(),
        &mut |i| steps.push(i),
    )
    .unwrap();

    assert_eq!(
        model.calls, NUM_EVALS,
        "exactly one transformer forward per step — the checkpoint is guidance-distilled"
    );
    assert_eq!(steps, vec![1, 2], "one progress tick per completed step");
    assert!(
        model.batches.iter().all(|&b| b == 1),
        "the packed sequence enters the transformer as a batch of ONE; a conditional/unconditional \
         pair would double it: {:?}",
        model.batches
    );
    for seen in model.seen_video.iter().chain(&model.seen_audio) {
        assert_eq!(seen.shape()[0], 1);
    }
}

/// The `fl2va` keyframe anchors are **bit-identical** after the loop: the scheduler writes only the
/// generated tail.
#[test]
fn the_conditioning_anchors_are_untouched() {
    let f = fixture();
    let l = layout();
    let input = batched(&f, "in.video_latents");
    let (video, _) = run(&f, JointSchedule::new(NUM_INFERENCE_STEPS).unwrap());

    let cond_rows = l.num_condition_video_rows();
    let features = input.shape()[2];
    let head = (cond_rows * features) as usize;
    let before = input.as_slice::<f32>();
    let after = video.as_slice::<f32>();
    assert_eq!(
        &before[..head],
        &after[..head],
        "the keyframe anchors must be bit-identical after the loop"
    );
    let (peak, _) = rel(
        &mlx_rs::ops::subtract(&video, &input).unwrap(),
        &Array::from_f32(0.0),
    );
    assert!(
        peak > 0.0,
        "…and the target rows must have moved: {peak:.3e}"
    );
}

// ---------------------------------------------------------------------------------------------
// Cancellation — measured, not asserted
// ---------------------------------------------------------------------------------------------

/// A synthetic model with real per-step compute, so a timer over the loop measures something.
struct Costly {
    w: Array,
    x: Array,
    matmuls: usize,
    calls: usize,
}

impl Costly {
    fn new(dim: i32, matmuls: usize) -> Self {
        let n = (dim * dim) as usize;
        let v: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 512.0).collect();
        Self {
            w: Array::from_slice(&v, &[dim, dim]),
            x: Array::from_slice(&v, &[dim, dim]),
            matmuls,
            calls: 0,
        }
    }
}

impl JointVelocity for Costly {
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
        self.calls += 1;
        // Real work, folded into the returned velocity so it cannot be optimized away and so it
        // lands in the same graph the loop evaluates.
        let mut acc = self.x.clone();
        for _ in 0..self.matmuls {
            acc = mlx_rs::ops::matmul(&acc, &self.w)?;
            acc = mlx_rs::ops::tanh(&acc)?;
        }
        let scale = acc.mean(None)?.reshape(&[1, 1, 1])?;
        Ok((
            mlx_rs::ops::multiply(step.video_rows, &scale)?,
            mlx_rs::ops::multiply(step.audio_rows, &scale)?,
        ))
    }
}

/// **The cancellation proof — measured.**
///
/// # What asserting `Err(Canceled)` would not prove
///
/// Nothing. Under MLX's lazy evaluation a loop with **no** `eval` returns `Canceled` *faster*, not
/// slower: it never computed anything to begin with. Cancel-latency alone is therefore blind to
/// exactly the defect it appears to test, and an earlier draft of this test — comparing when
/// `on_step(1)` fires against the mean cost of a step — **passed with the `eval` deleted**, because
/// in a debug build the per-step graph construction is itself a large fraction of a small model's
/// step. That false green is why this measures something else.
///
/// # The property that actually matters
///
/// A cancel can only be observed while control is inside the loop, because that is where the check
/// is. So the question is not "how fast does a cancel return" but **what fraction of the render's
/// compute happens inside the cancellable window**.
///
/// With the per-step `eval`, essentially all of it does, and the caller's first readback is free.
/// Without it, the loop returns in graph-construction time and the entire schedule lands in the
/// caller's next operation — a VAE decode, which has no cancel check — so the window in which a
/// cancel means anything shrinks to nothing while `Err(Canceled)` keeps being returned promptly.
///
/// Both numbers come from one run on one machine, so the assertion encodes no absolute duration.
#[test]
fn cancellation_is_observed_within_a_step() {
    const STEPS: usize = 6;
    const DIM: i32 = 1536;
    const MATMULS: usize = 8;

    let f = fixture();
    let l = layout();
    let joint = JointSchedule::new(STEPS + 1).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    assert_eq!(joint.num_evals(), STEPS);
    let video = batched(&f, "in.video_latents");
    let audio = batched(&f, "in.audio_latents");

    // Warm the kernels so the measured run is not timing compilation.
    let mut warm = Costly::new(DIM, MATMULS);
    let (wv, wa) = denoise_av(
        &mut warm,
        &l,
        &joint,
        &adaln,
        &video,
        &audio,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();
    mlx_rs::transforms::eval([&wv, &wa]).unwrap();

    // The measurement: time inside the loop (cancellable) against time the caller's first readback
    // still has to pay (not cancellable).
    let mut model = Costly::new(DIM, MATMULS);
    let start = Instant::now();
    let (v, a) = denoise_av(
        &mut model,
        &l,
        &joint,
        &adaln,
        &video,
        &audio,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();
    let in_loop = start.elapsed();
    mlx_rs::transforms::eval([&v, &a]).unwrap();
    let deferred = start.elapsed() - in_loop;
    let total = in_loop + deferred;

    println!(
        "  {STEPS} steps: {in_loop:?} inside the loop, {deferred:?} deferred to the caller's \
         readback ({:.1}% cancellable)",
        100.0 * in_loop.as_secs_f64() / total.as_secs_f64()
    );
    // A floor on the measurement itself, not on the property: below a few ms the split is timer
    // noise. Kept well under the ~20 ms this model costs so that a mutant — which is *faster*
    // overall, having skipped the per-step synchronization — still clears it and is judged by the
    // assertion below rather than accidentally reddening here.
    assert!(
        total.as_millis() >= 8,
        "the synthetic model is too cheap to measure on this machine ({total:?}); raise \
         DIM/MATMULS"
    );
    assert!(
        in_loop > deferred * 4,
        "only {in_loop:?} of the {total:?} render happened inside the loop — {deferred:?} was \
         deferred to the caller's first readback, where there is no cancel check. The per-step \
         `mlx_rs::transforms::eval` is what keeps the compute inside the cancellable window; \
         without it a cancel still returns promptly but stops nothing."
    );

    // ...and the cancel itself: tripped from the progress tick, the loop stops at the boundary,
    // the model is not called again, and the caller waits about one step rather than six.
    let cancel = CancelFlag::default();
    let trip = cancel.clone();
    let mut model = Costly::new(DIM, MATMULS);
    let start = Instant::now();
    let err = denoise_av(
        &mut model,
        &l,
        &joint,
        &adaln,
        &video,
        &audio,
        &cancel,
        &mut |i| {
            if i == 1 {
                trip.cancel();
            }
        },
    )
    .unwrap_err();
    let cancelled = start.elapsed();
    assert!(matches!(err, Error::Canceled), "{err}");
    assert_eq!(model.calls, 1, "the model must not run after the cancel");
    assert!(
        cancelled < total / 2,
        "a cancel at step 1 of {STEPS} must cost far less than the whole schedule ({cancelled:?} \
         vs {total:?})"
    );
    println!("  cancelled after 1 of {STEPS} steps in {cancelled:?} (whole schedule {total:?})");
}
