//! **Joint audio+video denoise parity**, candle side, against
//! `crates/media/mlx-gen/tools/dump_minimax_h3_av_denoise.py`.
//!
//! The golden is a whole **2-step** loop produced by the official diffusers `MiniMaxH3Scheduler` —
//! two instances loaded from the *published* `scheduler/` and `audio_scheduler/` configs, so the
//! 12.0 / 3.0 pair is read from the same bytes production reads rather than typed twice
//! (`src/layout.rs` rule 3).
//!
//! # What is gated, and on what
//!
//! `rel` (relative max-abs-diff) is the gate everywhere, never a norm, a checksum or cosine. Two
//! shipped defects and one near-miss in this family all held cosine ≥ 0.99 or left norms unchanged.
//! Every mutation this file claims to catch has a **measured** control in the fixture's metadata,
//! asserted above `MUTATION_FLOOR`, so "this test would catch it" is evidence rather than a hope.
//!
//! # The indices are deliberately NOT compared
//!
//! The reference builds a per-step table of the timesteps actually present and sorts it ascending;
//! this port builds **one global table across the whole run** in first-appearance order, because
//! the precompute-and-evict needs a single table. The index tensors are therefore not comparable
//! and agreeing on them would be the wrong contract. What is comparable — and what the AdaLN row
//! actually depends on — is the **resolved per-row timestep value**, which
//! `row_timesteps_resolve_to_the_reference_values` compares element by element.
//!
//! # The sigma grids are compared BITWISE
//!
//! Not within a tolerance: the port reproduces `torch.linspace`'s halfway-split FMA exactly, and
//! `TimestepSchedule` resolves timesteps by bit pattern, so a schedule that is merely close would
//! key a different AdaLN cache. This is also the test that proves the bitwise-linspace claim on
//! *this* lane rather than inheriting the MLX lane's verification.

use crate::common;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError, Result};

use candle_gen_minimax_h3::denoise::{
    adaln_schedule, denoise_av, DenoiseModality, JointGeometry, JointSchedule, JointStep,
    JointVelocity, PackedLayout, RowClass, SigmaSchedule, NUM_ROW_CLASSES, TEXT_TAG,
};
use candle_gen_minimax_h3::dit::adaln::TimestepSchedule;
use candle_gen_minimax_h3::dit::positions::KeyframeAnchor;

use common::{assert_parity, cosine, flat, l2_norm, rel, std_dev, Golden, DENOISE_FIXTURE};

/// **This lane's** bound. The loop is f32 arithmetic over the reference's own recorded velocities,
/// so its residual is round-off; MLX's 1e-2 is a Metal-precision number and is not inherited.
const TOL: f32 = 1e-5;

/// A mutation must clear the numeric noise floor, or "the output moved" is just jitter.
const MUTATION_FLOOR: f32 = 1e-2;

/// The layout the fixture was dumped at — the shortest legal render (124 frames ⇒ 37 latent
/// frames, 207 stereo audio latents) on a 4×6 latent at patch `[1, 2, 2]`, with both keyframe
/// anchors.
const NUM_FRAMES: usize = 124;
const LATENT_HEIGHT: usize = 4;
const LATENT_WIDTH: usize = 6;
const PATCH: [usize; 3] = [1, 2, 2];
const NUM_TEXT_TOKENS: usize = 5;
const AUDIO_CHANNELS: usize = 2;
/// 3 requested steps is **2** model evaluations — the terminal sigma is inside the count.
const NUM_INFERENCE_STEPS: usize = 3;
const NUM_EVALS: usize = 2;

fn dev() -> Device {
    Device::Cpu
}

fn fixture() -> Golden {
    Golden::load(DENOISE_FIXTURE)
}

fn layout() -> PackedLayout {
    PackedLayout::build(
        JointGeometry::new(NUM_FRAMES, LATENT_HEIGHT, LATENT_WIDTH).unwrap(),
        PATCH,
        &[TEXT_TAG; NUM_TEXT_TOKENS],
        AUDIO_CHANNELS,
        &[KeyframeAnchor::First, KeyframeAnchor::Last],
        &dev(),
    )
    .unwrap()
}

fn meta(f: &Golden, key: &str) -> String {
    f.meta(key)
        .unwrap_or_else(|| {
            panic!(
                "fixture metadata is missing `{key}`; re-run dump_minimax_h3_av_denoise.py against \
                 diffusers `main` (a fixture with no provenance cannot be shown to come from the \
                 converted-checkpoint path — sc-18740)"
            )
        })
        .to_string()
}

/// Replays the fixture's per-step velocities, counting calls and recording what it was handed.
///
/// Exercising the loop against the reference's own velocity table rather than a model is what makes
/// this a test of exactly what the denoise module owns.
struct Replay {
    video: Vec<Tensor>,
    audio: Vec<Tensor>,
    calls: usize,
    batches: Vec<usize>,
    seen_video: Vec<Tensor>,
    seen_audio: Vec<Tensor>,
    seen_adaln: Vec<Tensor>,
    seen_row_timesteps: Vec<[f32; NUM_ROW_CLASSES]>,
}

impl Replay {
    fn from_fixture(f: &Golden) -> Self {
        Self {
            video: (0..NUM_EVALS)
                .map(|i| f.batched(&format!("in.video_velocity.{i}")))
                .collect(),
            audio: (0..NUM_EVALS)
                .map(|i| f.batched(&format!("in.audio_velocity.{i}")))
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
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Tensor, Tensor)> {
        self.calls += 1;
        self.batches.push(step.video_rows.dims()[0]);
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

fn run_with(
    l: &PackedLayout,
    joint: &JointSchedule,
    adaln: &TimestepSchedule,
    f: &Golden,
    model: &mut Replay,
) -> (Tensor, Tensor) {
    denoise_av(
        model,
        l,
        joint,
        adaln,
        &f.batched("in.video_latents"),
        &f.batched("in.audio_latents"),
        &dev(),
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap()
}

fn run(f: &Golden, joint: JointSchedule) -> (Tensor, Tensor) {
    let l = layout();
    let adaln = adaln_schedule(&joint).unwrap();
    let mut model = Replay::from_fixture(f);
    run_with(&l, &joint, &adaln, f, &mut model)
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
        let want_sigmas = flat(&f.tensor(sigma_key));
        let want_timesteps = flat(&f.tensor(t_key));
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
                "{} sigma[{i}]: {got} != {want} — the bitwise torch.linspace reproduction has \
                 drifted, and every AdaLN cache key with it",
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
    let want_video = f.batched("out.video_latents.1");
    let want_audio = f.batched("out.audio_latents.1");

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
        "  swap rel {sv:.3e}/{sa:.3e}  both-audio rel {ba:.3e}  cosine {:.4}  |L2| {:.3} vs {:.3}",
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
        .map(|&s| candle_gen_minimax_h3::denoise::shift_sigma(s, 12.0))
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

    assert_eq!(l.seq_len(), f.shape("layout.position_ids")[0]);
    assert_eq!(l.video_indices(), f.u32_vec("layout.video_indices"));
    assert_eq!(l.audio_indices(), f.u32_vec("layout.audio_indices"));
    assert_eq!(l.text_indices(), f.u32_vec("layout.text_indices"));
    assert_eq!(l.token_tags(), f.u32_vec("layout.token_tags"));
    assert_parity(
        l.position_ids(),
        &f.tensor("layout.position_ids"),
        1e-5,
        "position_ids",
    );

    // `video_indices` skips the audio block — assert the discontiguity is real in the golden too,
    // so this is not vacuous on a regenerated fixture.
    let v = f.u32_vec("layout.video_indices");
    let cond = l.num_condition_video_rows();
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
    let classes = l.row_classes().to_vec();

    for step in 0..NUM_EVALS {
        let want = flat(&f.tensor(&format!("out.row_timesteps.{step}")));
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
        for &row in &l.video_indices()[..l.num_condition_video_rows()] {
            assert_eq!(want[row as usize], video_t.max(0.999));
        }
    }

    // Pinning text at a clean 1.0 — which sc-17145's original docs described — would change exactly
    // the text rows, and the generator counted them.
    let changed: usize = meta(&f, "text_rows_at_video_timestep").parse().unwrap();
    assert_eq!(changed, NUM_TEXT_TOKENS);
    let want0 = flat(&f.tensor("out.row_timesteps.0"));
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
        assert_eq!(idx.dims(), &[l.seq_len()]);
        let v = idx.to_vec1::<u32>().unwrap();
        assert!(
            (*v.iter().max().unwrap() as usize) < adaln.modulation_rows(),
            "step {step}"
        );
        let distinct: std::collections::BTreeSet<u32> = v.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "step {step} addresses only one table row — the timestep axis would be inert"
        );
    }
    // The video timestep moves between the two steps, so the video rows' block must too.
    let a = model.seen_adaln[0].to_vec1::<u32>().unwrap();
    let b = model.seen_adaln[1].to_vec1::<u32>().unwrap();
    let row = l.generated_video_rows()[0] as usize;
    assert_ne!(a[row], b[row], "the video rows must re-address per step");
}

// ---------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------

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
        &f.batched("out.video_latents.0"),
        TOL,
        "video after step 0",
    );
    assert_parity(
        &model.seen_audio[1],
        &f.batched("out.audio_latents.0"),
        TOL,
        "audio after step 0",
    );
    // ...and the return is step 1's.
    assert_parity(
        &video,
        &f.batched("out.video_latents.1"),
        TOL,
        "video after step 1",
    );
    assert_parity(
        &audio,
        &f.batched("out.audio_latents.1"),
        TOL,
        "audio after step 1",
    );

    let (vp, vm) = rel(&video, &f.batched("out.video_latents.1"));
    let (ap, am) = rel(&audio, &f.batched("out.audio_latents.1"));
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
        &f.batched("in.video_latents"),
        &f.batched("in.audio_latents"),
        &dev(),
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
        assert_eq!(seen.dims()[0], 1);
    }
}

/// The `fl2va` keyframe anchors are **bit-identical** after the loop: the scheduler writes only the
/// generated tail.
#[test]
fn the_conditioning_anchors_are_untouched() {
    let f = fixture();
    let l = layout();
    let input = f.batched("in.video_latents");
    let (video, _) = run(&f, JointSchedule::new(NUM_INFERENCE_STEPS).unwrap());

    let cond_rows = l.num_condition_video_rows();
    let features = input.dims()[2];
    let head = cond_rows * features;
    let before = flat(&input);
    let after = flat(&video);
    assert_eq!(
        &before[..head],
        &after[..head],
        "the keyframe anchors must be bit-identical after the loop"
    );
    assert_ne!(
        &before[head..],
        &after[head..],
        "…and the target rows must have moved"
    );
}

/// **Both modalities are emitted, and both really moved.** The acceptance criterion is "joint
/// denoise emits both modalities", so a loop that returned the audio unchanged — or returned the
/// video twice — must fail rather than pass on the video half alone.
#[test]
fn the_loop_emits_two_distinct_moved_modalities() {
    let f = fixture();
    let l = layout();
    let vin = f.batched("in.video_latents");
    let ain = f.batched("in.audio_latents");
    let (video, audio) = run(&f, JointSchedule::new(NUM_INFERENCE_STEPS).unwrap());

    assert_eq!(
        video.dims(),
        vin.dims(),
        "video keeps its row block's shape"
    );
    assert_eq!(
        audio.dims(),
        ain.dims(),
        "audio keeps its row block's shape"
    );
    assert_ne!(
        video.dims(),
        audio.dims(),
        "the two modalities have different widths, so returning one twice is detectable"
    );

    // Each modality moved away from its own input, past the noise floor.
    let (v_moved, _) = rel(&video, &vin);
    let (a_moved, _) = rel(&audio, &ain);
    println!("  video moved {v_moved:.3e} from its input; audio moved {a_moved:.3e}");
    assert!(v_moved > MUTATION_FLOOR, "the video half did not denoise");
    assert!(a_moved > MUTATION_FLOOR, "the audio half did not denoise");

    // ...and they are time-aligned by construction: one shared 40 Hz rotary clock, with the AV
    // residual inside the round's half-unit.
    let g = l.geometry();
    assert!(candle_gen_minimax_h3::denoise::rope_clocks_agree());
    assert!(
        (g.video_rope_span() - 5.0 / 3.0 * g.num_frames as f64).abs() < 1e-9,
        "the video clock must cover exactly {} frames",
        g.num_frames
    );
    println!(
        "  time alignment: video {} rotary units over {} latent frames, audio {} over {} latents; \
         drift {:.3} ms",
        g.video_rope_span(),
        g.num_latent_frames,
        g.audio_rope_span(),
        g.num_audio_latents,
        g.av_drift_seconds() * 1e3
    );
}

/// Cancellation returns the typed error and stops calling the model.
///
/// The MLX sibling additionally *times* this, because MLX's lazy graph makes a cancel that returns
/// promptly compatible with a loop that computed nothing. candle is eager on the CPU device, so the
/// arithmetic has already happened when `step` returns and there is no such gap to measure here —
/// the property the timing test was defending is instead structural on this lane, and on CUDA it is
/// held by the per-step `Device::synchronize` the loop performs (see `denoise`'s module docs).
/// Claiming a measured cancellation window on a device this suite does not run on would be the
/// overclaim; what is asserted is what can be shown.
#[test]
fn a_cancel_stops_the_loop_at_a_step_boundary() {
    let f = fixture();
    let l = layout();
    let joint = JointSchedule::new(6).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    let cancel = CancelFlag::default();
    let trip = cancel.clone();
    // A model that only serves 2 steps: if the loop ran past the cancel it would panic on the
    // missing velocity rather than quietly returning, so this cannot pass by accident.
    struct Two {
        calls: usize,
    }
    impl JointVelocity for Two {
        fn forward(&mut self, step: &JointStep<'_>) -> Result<(Tensor, Tensor)> {
            self.calls += 1;
            if self.calls > 2 {
                return Err(CandleError::Msg("the loop ran past the cancel".into()));
            }
            Ok((step.video_rows.clone(), step.audio_rows.clone()))
        }
    }
    let mut m = Two { calls: 0 };
    let err = denoise_av(
        &mut m,
        &l,
        &joint,
        &adaln,
        &f.batched("in.video_latents"),
        &f.batched("in.audio_latents"),
        &dev(),
        &cancel,
        &mut |i| {
            if i == 2 {
                trip.cancel();
            }
        },
    )
    .unwrap_err();
    assert!(matches!(err, CandleError::Canceled), "{err}");
    assert_eq!(m.calls, 2, "the model must not run after the cancel");
}
