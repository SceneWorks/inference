//! The **joint audio+video denoise loop** — one packed sequence, one transformer pass per step,
//! two sigma schedules.
//!
//! ```text
//! ┌ per step i ────────────────────────────────────────────────────────────────┐
//! │  cancel check                          <- the seam, real only because of ↓ │
//! │  adaln_indices(i)                      <- bounds-checked on the host       │
//! │  ONE model forward -> (v_video, v_audio)                                   │
//! │  video.step(i, tail)   sigma shift 12.0                                    │
//! │  audio.step(i, tail)   sigma shift  3.0                                    │
//! │  Device::synchronize()                 <- makes step i's compute LAND       │
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
//! The model is a **trait**, not a struct, so the loop is testable against the reference's own
//! recorded velocity table rather than against a model — which is what makes the committed
//! `av_denoise` golden a test of exactly what this module owns.
//!
//! # No CFG, and it is pinned rather than assumed
//!
//! The checkpoint is **guidance-distilled**. The reference has no guider component, no
//! `guidance_scale`, no `negative_prompt` and no unconditional pass; `hidden_states` enters the
//! transformer as a literal batch of one.
//!
//! So [`JointVelocity::forward`] takes no guidance parameter and the loop calls it **once** per
//! step. `tests/joint_denoise.rs::one_forward_per_step_and_no_cfg` counts the calls and checks the
//! batch axis, so reintroducing a conditional/unconditional pair fails rather than silently
//! halving throughput.
//!
//! # Cancellation, and why the synchronize is the whole of it — the candle version
//!
//! The MLX sibling places an `mlx_rs::transforms::eval` here and argues that without it the lazy
//! graph would defer every step's compute past the end of the loop, leaving the cancel check
//! structurally present and practically inert.
//!
//! candle is **eager**, so on the CPU device that argument does not apply: the arithmetic has
//! already happened by the time `step` returns. It does apply on **CUDA**, for a different reason —
//! candle launches its kernels on the device's stream *asynchronously*, so without a synchronize
//! the loop can run ahead of the GPU, fire every `on_step` early, and leave the queued work to land
//! in the caller's next operation (a VAE decode, which has no cancel check). The seam would again be
//! present and inert.
//!
//! [`denoise_av`] therefore synchronizes at the bottom of every step. The same call is the one that
//! returns evicted pages to the driver ([`crate::dit::adaln::release_device_memory`]), and
//! `candle_gen::block_window` already establishes the house position that a per-forward synchronize
//! is not optional. On the CPU device it is a cheap no-op, so the loop reads identically on both
//! backends rather than branching on the device.

pub mod geometry;
pub mod packing;
pub mod schedule;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError, Result};

use crate::dit::adaln::TimestepSchedule;

pub use geometry::{
    align_num_frames, audio_latent_num_frames, rope_clocks_agree, video_latent_num_frames,
    JointGeometry, AUDIO_LATENTS_PER_SECOND, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK,
    LEGAL_FRAME_COUNTS, MAX_AV_DRIFT_SECONDS, MINIMAX_H3_FPS, ROPE_UNITS_PER_SECOND,
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
    pub video_rows: &'a Tensor,
    /// `[1, audio_indices.len(), audio_features]`.
    pub audio_rows: &'a Tensor,
    /// `[seq_len]` u32 rows of the precomputed AdaLN modulation table —
    /// `global_timestep_index · MODALITY_NUM + token_tag`, bounds-checked.
    pub adaln_indices: &'a Tensor,
    /// This step's timestep per [`RowClass`], in class order. For an
    /// [`crate::dit::adaln::AdaLnResidency::Resident`] load, this is what the timestep MLP consumes.
    pub row_timesteps: [f32; NUM_ROW_CLASSES],
}

/// The one shared transformer forward a joint step runs.
///
/// The loop owns scheduling, row bookkeeping and cancellation; the implementor owns only "packed
/// rows in, velocities out".
pub trait JointVelocity {
    /// Predict the **data-ward** velocity of every generated row.
    ///
    /// Returns `(video_velocity, audio_velocity)` shaped exactly like [`JointStep::video_rows`]
    /// and [`JointStep::audio_rows`] — including the conditioning rows, which the loop then
    /// discards rather than stepping.
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Tensor, Tensor)>;
}

/// Build the global AdaLN modulation schedule a joint run needs.
///
/// One entry per [`RowClass`] per evaluation, which [`crate::dit::adaln::TimestepSchedule`] then
/// dedups into the table [`crate::dit::adaln::AdaLnCache::precompute_and_evict`] projects. Every
/// class is declared at every step so the class indices are stable — see [`packing`]'s module docs.
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
/// because candle tensors are values.
fn step_generated_tail(
    schedule: &SigmaSchedule,
    index: usize,
    latents: &Tensor,
    velocity: &Tensor,
    num_condition_rows: usize,
) -> Result<Tensor> {
    if latents.dims() != velocity.dims() {
        return Err(CandleError::Msg(format!(
            "minimax-h3 denoise: {} latents {:?} and velocity {:?} must be the same rows",
            schedule.modality().name(),
            latents.dims(),
            velocity.dims()
        )));
    }
    if num_condition_rows == 0 {
        return schedule.step(index, latents, velocity);
    }
    let rows = latents.dims()[1];
    if num_condition_rows >= rows {
        return Err(CandleError::Msg(format!(
            "minimax-h3 denoise: {num_condition_rows} conditioning rows in a {rows}-row {} block \
             leaves nothing to denoise",
            schedule.modality().name()
        )));
    }
    let head = latents.narrow(1, 0, num_condition_rows)?;
    let tail = latents.narrow(1, num_condition_rows, rows - num_condition_rows)?;
    let tail_velocity = velocity.narrow(1, num_condition_rows, rows - num_condition_rows)?;
    let stepped = schedule.step(index, &tail, &tail_velocity)?;
    // The head is re-joined at the stepped tail's dtype (f32, as the reference's latents are), so
    // the two halves of the block cannot silently end up at different precisions.
    let head = head.to_dtype(stepped.dtype())?;
    Ok(Tensor::cat(&[&head, &stepped], 1)?.contiguous()?)
}

/// **The joint dual-modality denoise loop.**
///
/// Runs [`JointSchedule::num_evals`] evaluations, each one model forward, stepping the video rows
/// on the 12.0-shift schedule and the audio rows on the 3.0-shift one. Returns the two latent row
/// blocks in the same shapes they arrived in — conditioning rows included and unchanged.
///
/// * `video` — `[1, video_indices.len(), video_features]`, conditioning rows first;
/// * `audio` — `[1, audio_indices.len(), audio_features]`;
/// * `cancel` — checked at every step boundary, which the per-step synchronize makes a real one;
/// * `on_step` — called with the 1-based completed step **after** that step's compute has landed.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av(
    model: &mut dyn JointVelocity,
    layout: &PackedLayout,
    schedule: &JointSchedule,
    adaln: &TimestepSchedule,
    video: &Tensor,
    audio: &Tensor,
    device: &Device,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<(Tensor, Tensor)> {
    if adaln.num_steps() != schedule.num_evals() {
        return Err(CandleError::Msg(format!(
            "minimax-h3 denoise: the AdaLN schedule covers {} steps but the sigma schedules drive \
             {}; build it with `adaln_schedule(schedule)`",
            adaln.num_steps(),
            schedule.num_evals()
        )));
    }
    if adaln.num_row_classes() != NUM_ROW_CLASSES {
        return Err(CandleError::Msg(format!(
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
        // (sc-5551). The synchronize at the bottom of the loop is what makes this seam real on
        // CUDA: without it candle's stream-async launches let the loop run ahead of the device and
        // the queued work lands in the caller's next operation, where no cancel check exists.
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // The one place a stale step index is catchable.
        let adaln_indices =
            adaln.adaln_indices(i, layout.row_classes(), layout.token_tags(), device)?;
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

        // Force this step's compute to LAND inside this step. See the module docs: on CUDA this is
        // the whole of cancellation responsiveness, and it also bounds the queued work. A no-op on
        // the CPU device, where the arithmetic has already happened.
        device.synchronize()?;
        on_step(i + 1);
    }
    Ok((vlat, alat))
}

fn check_block(x: &Tensor, rows: usize, what: &str) -> Result<()> {
    let s = x.dims();
    if s.len() != 3 || s[0] != 1 || s[1] != rows {
        return Err(CandleError::Msg(format!(
            "minimax-h3 denoise: the {what} latents must be [1, {rows}, F], got {s:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::positions::KeyframeAnchor;

    fn dev() -> Device {
        Device::Cpu
    }

    /// A velocity model that returns a fixed multiple of its input and counts its calls.
    struct Counting {
        calls: usize,
        batches: Vec<usize>,
        scale: f64,
    }

    impl JointVelocity for Counting {
        fn forward(&mut self, step: &JointStep<'_>) -> Result<(Tensor, Tensor)> {
            self.calls += 1;
            self.batches.push(step.video_rows.dims()[0]);
            Ok((
                (step.video_rows * self.scale)?,
                (step.audio_rows * self.scale)?,
            ))
        }
    }

    /// The shortest legal render at a small canvas — see `packing::tests`.
    fn layout(anchors: &[KeyframeAnchor]) -> PackedLayout {
        let geometry = JointGeometry::new(124, 4, 6).unwrap();
        PackedLayout::build(geometry, [1, 2, 2], &[TEXT_TAG; 5], 2, anchors, &dev()).unwrap()
    }

    fn rows(n: usize, f: usize) -> Tensor {
        let v: Vec<f32> = (0..n * f).map(|i| (i % 7) as f32 * 0.25 - 0.5).collect();
        Tensor::from_vec(v, (1, n, f), &dev()).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
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
            &dev(),
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
            &dev(),
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap();

        let before = flat(&video);
        let after = flat(&v);
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
            &dev(),
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
            &dev(),
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
            a.num_distinct_timesteps() < a.num_steps() * NUM_ROW_CLASSES,
            "the constant classes must dedup"
        );
        // The reference-audio class is the same row at every step.
        let first = a
            .global_timestep_index(0, RowClass::ConditionAudio as usize)
            .unwrap();
        for step in 0..a.num_steps() {
            assert_eq!(
                a.global_timestep_index(step, RowClass::ConditionAudio as usize)
                    .unwrap(),
                first
            );
        }
    }
}
