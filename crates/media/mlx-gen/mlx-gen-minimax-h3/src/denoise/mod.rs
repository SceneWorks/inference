//! The **joint audio+video denoise loop** (sc-17146) — one packed sequence, one transformer pass
//! per step, two sigma schedules.
//!
//! ```text
//! ┌ per step i ────────────────────────────────────────────────────────────────┐
//! │  cancel check                          <- the seam, real only because of ↓ │
//! │  adaln_indices(i)                      <- bounds-checked; MLX gathers OOB  │
//! │  ONE model forward -> (v_video, v_audio)                                   │
//! │  video.step(i, tail)   sigma shift 12.0                                    │
//! │  audio.step(i, tail)   sigma shift  3.0                                    │
//! │  eval([video, audio])                  <- makes step i's compute happen    │
//! │  on_step(i + 1)                           INSIDE step i                    │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # What this module owns
//!
//! | | |
//! |---|---|
//! | [`geometry`] | frame / latent / audio-token counts and **the AV time alignment** |
//! | [`schedule`] | `MiniMaxH3Scheduler`, the two shifts, the reversed velocity sign |
//! | [`packing`] | row order, the three index tensors, tags, row classes, scatter/gather |
//! | this module | the loop, and the [`JointVelocity`] seam the DiT plugs into |
//!
//! The model itself is a **trait**, not a struct, because the DiT's 17 input/output projections —
//! `proj_in`, `audio_proj_in`, `context_embedder`, `time_embedder`, `norm_out`, `proj_out`,
//! `audio_proj_out` — are sc-17147's. The loop is complete and testable without them, and
//! sc-17147 implements [`JointVelocity`] over the block stack that already exists.
//!
//! # No CFG, and it is pinned rather than assumed
//!
//! The checkpoint is **guidance-distilled**. The reference has no guider component, no
//! `guidance_scale`, no `negative_prompt` and no unconditional pass; `hidden_states` enters the
//! transformer as a literal batch of one. Grepping the reference's whole `minimax_h3` package for
//! guidance vocabulary returns only prose asserting its absence.
//!
//! So [`JointVelocity::forward`] takes no guidance parameter and the loop calls it **once** per
//! step. `tests/joint_denoise.rs::one_forward_per_step_and_no_cfg` counts the calls and checks the
//! batch axis, so reintroducing a conditional/unconditional pair fails rather than silently
//! halving throughput.
//!
//! # Cancellation, and why the `eval` placement is the whole of it
//!
//! MLX is lazily evaluated. Without the per-step [`mlx_rs::transforms::eval`], every step's
//! `model.forward` would only *build graph*: the loop would run to completion in milliseconds, fire
//! every `on_step` before any arithmetic had happened, and then pay for the entire schedule in one
//! unbounded burst at the caller's first readback — where no cancel check exists at all. The
//! between-steps cancel check would be structurally present and practically inert.
//!
//! The `eval` is therefore not a memory optimization that happens to help; it is what makes the
//! step boundary a real point in time. `tests/joint_denoise.rs::cancellation_is_observed_within_a_step`
//! **measures** this — it compares the time at which `on_step(1)` fires against the measured mean
//! cost of a step, self-calibrating so the assertion does not encode this machine's speed — rather
//! than asserting that a cancel returns `Canceled`, which it does either way.

pub mod geometry;
pub mod packing;
pub mod schedule;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::{CancelFlag, Error, Result};

use crate::dit::adaln::TimestepSchedule;

pub use geometry::{
    align_num_frames, audio_latent_num_frames, av_grids_align_exactly, decoded_audio_samples,
    delivered_audio_samples, rope_clocks_agree, video_latent_num_frames, JointGeometry,
    AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLES_PER_LATENT, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK,
    LEGAL_FRAME_COUNTS, MAX_AV_DRIFT_SECONDS, MAX_DELIVERED_AV_RESIDUAL_SECONDS, MINIMAX_H3_FPS,
    ROPE_UNITS_PER_SECOND,
};
pub use packing::{PackedLayout, RowClass, AUDIO_TAG, NUM_ROW_CLASSES, TEXT_TAG, VIDEO_TAG};
pub use schedule::{
    shift_sigma, DenoiseModality, JointSchedule, SigmaSchedule, AUDIO_SIGMA_SHIFT,
    KEYFRAME_NOISE_AUG, MIN_INFERENCE_STEPS, REFERENCE_AUDIO_TIMESTEP, VIDEO_SIGMA_SHIFT,
};

/// Everything one model evaluation needs, besides the weights.
///
/// Deliberately carries **no** guidance scale, negative context or batch multiplier — see the
/// module docs on the guidance-distilled checkpoint.
#[derive(Debug)]
pub struct JointStep<'a> {
    /// Zero-based evaluation index within the schedule.
    pub index: usize,
    /// The packed sequence's structure — constant across steps.
    pub layout: &'a PackedLayout,
    /// `[1, video_indices.len(), video_features]`, **conditioning rows first**.
    pub video_rows: &'a Array,
    /// `[1, audio_indices.len(), audio_features]`.
    pub audio_rows: &'a Array,
    /// `[seq_len]` int32 rows of the precomputed AdaLN modulation table —
    /// `global_timestep_index · MODALITY_NUM + token_tag`, bounds-checked.
    pub adaln_indices: &'a Array,
    /// This step's timestep per [`RowClass`], in class order. For an
    /// [`crate::dit::adaln::AdaLnResidency::Resident`] load, this is what the timestep MLP consumes.
    pub row_timesteps: [f32; NUM_ROW_CLASSES],
}

/// The one shared transformer forward a joint step runs.
///
/// Implemented by sc-17147's DiT wrapper over the [`crate::dit`] block stack. The loop owns
/// scheduling, row bookkeeping and cancellation; the implementor owns only "packed rows in,
/// velocities out".
pub trait JointVelocity {
    /// Predict the **data-ward** velocity of every generated row.
    ///
    /// Returns `(video_velocity, audio_velocity)` shaped exactly like [`JointStep::video_rows`]
    /// and [`JointStep::audio_rows`] — including the conditioning rows, which the loop then
    /// discards rather than stepping.
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)>;
}

/// Build the global AdaLN modulation schedule a joint run needs.
///
/// One entry per [`RowClass`] per evaluation, which
/// [`crate::dit::adaln::TimestepSchedule`] then dedups into the table
/// [`crate::dit::adaln::AdaLnCache::precompute_and_evict`] projects. Every class is declared at
/// every step so the class indices are stable — see [`packing`]'s module docs.
pub fn adaln_schedule(schedule: &JointSchedule) -> Result<TimestepSchedule> {
    let steps: Vec<Vec<f32>> = (0..schedule.num_evals())
        .map(|i| {
            PackedLayout::row_timesteps(
                schedule.video().timesteps()[i],
                schedule.audio().timesteps()[i],
            )
            .to_vec()
        })
        .collect();
    TimestepSchedule::new(steps)
}

/// Step one modality's **generated tail**, leaving its conditioning rows untouched.
///
/// The reference writes `latents[num_condition_rows:] = scheduler.step(...)` in place, so the
/// `fl2va` anchors survive every step by construction. Reproduced here as a split-and-rejoin
/// because MLX arrays are values.
fn step_generated_tail(
    schedule: &SigmaSchedule,
    index: usize,
    latents: &Array,
    velocity: &Array,
    num_condition_rows: i32,
) -> Result<Array> {
    if latents.shape() != velocity.shape() {
        return Err(Error::Msg(format!(
            "minimax-h3 denoise: {} latents {:?} and velocity {:?} must be the same rows",
            schedule.modality().name(),
            latents.shape(),
            velocity.shape()
        )));
    }
    if num_condition_rows == 0 {
        return schedule.step(index, latents, velocity);
    }
    let rows = latents.shape()[1];
    if num_condition_rows < 0 || num_condition_rows >= rows {
        return Err(Error::Msg(format!(
            "minimax-h3 denoise: {num_condition_rows} conditioning rows in a {rows}-row {} block \
             leaves nothing to denoise",
            schedule.modality().name()
        )));
    }
    let head = crate::tensor::slice_axis(latents, 1, 0, num_condition_rows)?;
    let tail = crate::tensor::slice_axis(latents, 1, num_condition_rows, rows)?;
    let tail_velocity = crate::tensor::slice_axis(velocity, 1, num_condition_rows, rows)?;
    let stepped = schedule.step(index, &tail, &tail_velocity)?;
    Ok(concatenate_axis(&[head, stepped], 1)?)
}

/// **The joint dual-modality denoise loop.**
///
/// Runs [`JointSchedule::num_evals`] evaluations, each one model forward, stepping the video rows
/// on the 12.0-shift schedule and the audio rows on the 3.0-shift one. Returns the two latent row
/// blocks in the same shapes they arrived in — conditioning rows included and unchanged.
///
/// * `video` — `[1, video_indices.len(), video_features]`, conditioning rows first;
/// * `audio` — `[1, audio_indices.len(), audio_features]`;
/// * `cancel` — checked at every step boundary, which the per-step `eval` makes a real one;
/// * `on_step` — called with the 1-based completed step **after** that step's compute has landed.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av(
    model: &mut dyn JointVelocity,
    layout: &PackedLayout,
    schedule: &JointSchedule,
    adaln: &TimestepSchedule,
    video: &Array,
    audio: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<(Array, Array)> {
    if adaln.num_steps() != schedule.num_evals() {
        return Err(Error::Msg(format!(
            "minimax-h3 denoise: the AdaLN schedule covers {} steps but the sigma schedules drive \
             {}; build it with `adaln_schedule(schedule)`",
            adaln.num_steps(),
            schedule.num_evals()
        )));
    }
    if adaln.num_row_classes() != NUM_ROW_CLASSES {
        return Err(Error::Msg(format!(
            "minimax-h3 denoise: the AdaLN schedule declares {} row classes, expected \
             {NUM_ROW_CLASSES}",
            adaln.num_row_classes()
        )));
    }
    check_block(video, layout.video_indices().len(), "video")?;
    check_block(audio, layout.audio_indices().len(), "audio")?;

    let mut vlat = video.clone();
    let mut alat = audio.clone();
    for i in 0..schedule.num_evals() {
        // Honor the engine cancellation contract — check before each (minutes-long) step
        // (sc-5551). The `eval` at the bottom of the loop is what makes this seam real: without
        // it MLX's lazy graph defers every step's compute past the end of the loop and this check
        // could never observe a trip.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // The one place a stale step index is catchable — MLX gathers out of bounds silently.
        let adaln_indices = adaln.adaln_indices(i, layout.row_classes(), layout.token_tags())?;
        let row_timesteps = PackedLayout::row_timesteps(
            schedule.video().timesteps()[i],
            schedule.audio().timesteps()[i],
        );

        // ONE forward. The checkpoint is guidance-distilled: there is no unconditional pass.
        let (vvel, avel) = model.forward(&JointStep {
            index: i,
            layout,
            video_rows: &vlat,
            audio_rows: &alat,
            adaln_indices: &adaln_indices,
            row_timesteps,
        })?;

        vlat = step_generated_tail(
            schedule.video(),
            i,
            &vlat,
            &vvel,
            layout.num_condition_video_rows(),
        )?;
        alat = step_generated_tail(
            schedule.audio(),
            i,
            &alat,
            &avel,
            layout.num_condition_audio_rows(),
        )?;

        // Force this step's compute to happen INSIDE this step. See the module docs: this is the
        // whole of cancellation responsiveness, and it also bounds the graph's peak memory.
        mlx_rs::transforms::eval([&vlat, &alat])?;
        on_step(i + 1);
    }
    Ok((vlat, alat))
}

fn check_block(x: &Array, rows: usize, what: &str) -> Result<()> {
    let s = x.shape();
    if s.len() != 3 || s[0] != 1 || s[1] != rows as i32 {
        return Err(Error::Msg(format!(
            "minimax-h3 denoise: the {what} latents must be [1, {rows}, F], got {s:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::positions::KeyframeAnchor;

    /// A velocity model that returns a fixed multiple of its input and counts its calls.
    struct Counting {
        calls: usize,
        batches: Vec<i32>,
        scale: f32,
    }

    impl JointVelocity for Counting {
        fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
            self.calls += 1;
            self.batches.push(step.video_rows.shape()[0]);
            let k = Array::from_f32(self.scale);
            Ok((
                mlx_rs::ops::multiply(step.video_rows, &k)?,
                mlx_rs::ops::multiply(step.audio_rows, &k)?,
            ))
        }
    }

    /// The shortest legal render at a small canvas — see `packing::tests`.
    fn layout(anchors: &[KeyframeAnchor]) -> PackedLayout {
        let geometry = JointGeometry::new(124, 4, 6).unwrap();
        PackedLayout::build(geometry, [1, 2, 2], &[TEXT_TAG; 5], 2, anchors).unwrap()
    }

    fn rows(n: usize, f: i32) -> Array {
        let v: Vec<f32> = (0..n * f as usize)
            .map(|i| (i % 7) as f32 * 0.25 - 0.5)
            .collect();
        Array::from_slice(&v, &[1, n as i32, f])
    }

    /// **One forward per step.** A reintroduced CFG pass would double this.
    #[test]
    fn the_loop_runs_exactly_one_forward_per_step() {
        let l = layout(&[]);
        let s = JointSchedule::new(5).unwrap();
        let a = adaln_schedule(&s).unwrap();
        let mut m = Counting {
            calls: 0,
            batches: vec![],
            scale: 0.1,
        };
        let mut steps = vec![];
        denoise_av(
            &mut m,
            &l,
            &s,
            &a,
            &rows(l.video_indices().len(), 4),
            &rows(l.audio_indices().len(), 3),
            &CancelFlag::default(),
            &mut |i| steps.push(i),
        )
        .unwrap();
        assert_eq!(s.num_evals(), 4, "5 requested steps is 4 evaluations");
        assert_eq!(m.calls, 4, "exactly one forward per step — no CFG pair");
        assert_eq!(steps, vec![1, 2, 3, 4]);
        assert!(
            m.batches.iter().all(|&b| b == 1),
            "batch of one: {:?}",
            m.batches
        );
    }

    /// The conditioning anchors are never denoised — the scheduler writes only the generated tail.
    #[test]
    fn conditioning_rows_survive_every_step() {
        let l = layout(&[KeyframeAnchor::First, KeyframeAnchor::Last]);
        assert_eq!(l.num_condition_video_rows(), 12);
        let s = JointSchedule::new(4).unwrap();
        let a = adaln_schedule(&s).unwrap();
        let video = rows(l.video_indices().len(), 4);
        let audio = rows(l.audio_indices().len(), 3);
        let mut m = Counting {
            calls: 0,
            batches: vec![],
            scale: 0.7,
        };
        let (v, _) = denoise_av(
            &mut m,
            &l,
            &s,
            &a,
            &video,
            &audio,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap();

        let before = video.as_slice::<f32>();
        let after = v.as_slice::<f32>();
        let head = 12 * 4;
        assert_eq!(
            &before[..head],
            &after[..head],
            "the anchors must be untouched"
        );
        assert_ne!(
            &before[head..],
            &after[head..],
            "…and the target rows must have moved"
        );
    }

    /// Cancellation returns the typed error and stops calling the model.
    #[test]
    fn a_cancel_stops_the_loop_at_a_step_boundary() {
        let l = layout(&[]);
        let s = JointSchedule::new(6).unwrap();
        let a = adaln_schedule(&s).unwrap();
        let cancel = CancelFlag::default();
        let trip = cancel.clone();
        let mut m = Counting {
            calls: 0,
            batches: vec![],
            scale: 0.1,
        };
        let err = denoise_av(
            &mut m,
            &l,
            &s,
            &a,
            &rows(l.video_indices().len(), 4),
            &rows(l.audio_indices().len(), 3),
            &cancel,
            &mut |i| {
                if i == 2 {
                    trip.cancel();
                }
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::Canceled), "{err}");
        assert_eq!(m.calls, 2, "the model must not run after the cancel");
    }

    /// A mismatched AdaLN schedule is a typed error rather than a stale-index gather.
    #[test]
    fn a_mismatched_adaln_schedule_is_rejected() {
        let l = layout(&[]);
        let s = JointSchedule::new(5).unwrap();
        let wrong = adaln_schedule(&JointSchedule::new(9).unwrap()).unwrap();
        let mut m = Counting {
            calls: 0,
            batches: vec![],
            scale: 0.1,
        };
        let e = denoise_av(
            &mut m,
            &l,
            &s,
            &wrong,
            &rows(l.video_indices().len(), 4),
            &rows(l.audio_indices().len(), 3),
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("steps"), "{e}");
        assert_eq!(m.calls, 0, "rejected before any compute");

        // ...and a latent block of the wrong height too.
        let a = adaln_schedule(&s).unwrap();
        assert!(denoise_av(
            &mut m,
            &l,
            &s,
            &a,
            &rows(3, 4),
            &rows(l.audio_indices().len(), 3),
            &CancelFlag::default(),
            &mut |_| {},
        )
        .is_err());
    }

    /// The AdaLN schedule the loop drives has the four stable classes and dedups the constant ones.
    #[test]
    fn the_adaln_schedule_declares_four_stable_classes() {
        let s = JointSchedule::new(20).unwrap();
        let a = adaln_schedule(&s).unwrap();
        assert_eq!(a.num_steps(), s.num_evals());
        assert_eq!(a.num_row_classes(), NUM_ROW_CLASSES);
        // 19 video + 19 audio + 0.999 + 1.0, minus the sigma=1 coincidence at step 0.
        assert!(
            a.num_distinct_timesteps() < (a.num_steps() * NUM_ROW_CLASSES) as i32,
            "the constant classes must dedup"
        );
        // The reference-audio class is the same row at every step.
        let first = a
            .global_timestep_index(0, RowClass::ConditionAudio as i32)
            .unwrap();
        for step in 0..a.num_steps() {
            assert_eq!(
                a.global_timestep_index(step, RowClass::ConditionAudio as i32)
                    .unwrap(),
                first
            );
        }
    }
}
