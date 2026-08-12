//! `MiniMaxH3Scheduler` — rectified-flow Euler, and the **two** sigma schedules a joint run steps
//! in lockstep.
//!
//! # Five ways this is not stock rectified flow
//!
//! The reference's own module docstring is unusually explicit that this is incompatible with a
//! `FlowMatchEulerDiscreteScheduler` config. Every difference is silently wrong rather than
//! loudly wrong — a port that copies any other flow-match family in-tree still produces
//! plausible-looking output.
//!
//! 1. **The velocity sign is reversed.** The transformer predicts a *data-ward* velocity, so
//!    `x0 = x_t + σ·v`, not diffusers' `x0 = x_t − σ·v`. See [`SigmaSchedule::step`].
//! 2. **`t = 1 − σ` on the unit interval, and `t = 1` is CLEAN** — the opposite direction from
//!    every other flow-match family, which exposes `σ · num_train_timesteps` on a 1000× scale.
//! 3. **The grid is `linspace(1, 0, N)` with the terminal zero included**, then shifted, then
//!    collapsed with `unique_consecutive`. So `N` requests **`N − 1` model evaluations**, and the
//!    collapse can make it fewer still. `N ≥ 2` is required.
//! 4. **Two schedules, one forward.** [`VIDEO_SIGMA_SHIFT`] 12.0 and [`AUDIO_SIGMA_SHIFT`] 3.0,
//!    stepped separately after **one** shared transformer pass.
//! 5. **The two σ in one step come from different sources**, deliberately — see below.
//!
//! # The two-source σ, which is the subtlest thing here
//!
//! ```text
//! denoised = x_t + (1 − t)·v      <- σ recovered from the TIMESTEP the model was conditioned on
//! r        = σ[i+1] / σ[i]        <- σ read from the stored GRID
//! x_next   = r·x_t + (1 − r)·denoised
//! ```
//!
//! Those are the same quantity in exact arithmetic and *not* in float32: below `σ = 0.5` the round
//! trip `1 − (1 − σ)` is inexact, and the reference keeps the two apart rather than reusing one.
//! [`SigmaSchedule::step`] reproduces that split; collapsing it is a real numeric divergence that
//! grows as the schedule approaches zero.
//!
//! # Each shift is bound to its modality by construction
//!
//! There is no "shift" parameter a caller can hand to the wrong schedule:
//! [`DenoiseModality::sigma_shift`] is the only source of the default, [`JointSchedule`] owns one
//! schedule per modality, and [`SigmaSchedule::modality`] travels with the grid so
//! [`JointSchedule::new_from_parts`] can reject a swapped pair. `tests/joint_denoise.rs` pins that
//! the distinction is measurable, not merely typed.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result};

/// Video sigma shift — `_minimax_h3.sigma_shift_scales.video`, and
/// `scheduler/scheduler_config.json`.
pub const VIDEO_SIGMA_SHIFT: f32 = 12.0;

/// Audio sigma shift — `_minimax_h3.sigma_shift_scales.audio`, and
/// `audio_scheduler/scheduler_config.json`.
pub const AUDIO_SIGMA_SHIFT: f32 = 3.0;

/// The `t` a **visual** conditioning anchor is pinned at: just short of clean.
///
/// `MiniMaxH3ModularPipeline.keyframe_noise_aug`, a hardcoded pipeline property with no config
/// backing — "the released model was trained with its anchors very slightly noised, so conditioning
/// on exactly `t = 1.0` is off-distribution". Applied as `max(video_t, 0.999)`, so once the video
/// schedule passes it the anchors follow the video timestep instead.
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;

/// The `t` a **reference audio** conditioning row is pinned at: exactly clean.
///
/// Only `ref2va` (sc-17157) has such rows; `t2va` / `fl2va` carry none.
pub const REFERENCE_AUDIO_TIMESTEP: f32 = 1.0;

/// The smallest `num_inference_steps` the scheduler accepts. The terminal zero is part of the
/// count, so 2 is one model evaluation.
pub const MIN_INFERENCE_STEPS: usize = 2;

/// Which modality a [`SigmaSchedule`] belongs to. **The shift is a property of this**, not a free
/// parameter, which is what makes a swap a type-level mistake rather than a numeric one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseModality {
    /// The 24-channel video latents.
    Video,
    /// The 32-channel audio latents.
    Audio,
}

impl DenoiseModality {
    /// The published sigma shift for this modality — 12.0 video, 3.0 audio.
    pub const fn sigma_shift(self) -> f32 {
        match self {
            Self::Video => VIDEO_SIGMA_SHIFT,
            Self::Audio => AUDIO_SIGMA_SHIFT,
        }
    }

    /// The packed sequence's per-row modality tag: video 0, audio 2. (Text is 1 and has no
    /// schedule — see [`crate::denoise::packing::RowClass`].)
    pub const fn token_tag(self) -> u32 {
        match self {
            Self::Video => 0,
            Self::Audio => 2,
        }
    }

    /// Human-readable name, for error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

/// The exponential sigma shift: `σ' = s·σ / (1 + (s − 1)·σ)`.
///
/// Fixed points at 0 and 1, so both schedules start at `σ = 1` (⇒ `t = 0`) and end at `σ = 0`
/// regardless of the shift, and diverge everywhere in between. Applying it twice is **not** the
/// same as applying it once with any single shift, and the arithmetic is observably different at
/// every interior point.
pub fn shift_sigma(sigma: f32, shift: f32) -> f32 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// `torch.linspace(1.0, 0.0, steps, dtype=torch.float32)`, **bitwise**.
///
/// Not `1 − i/(N−1)`: torch fills the two halves from opposite ends with a fused multiply-add,
/// `start + step·i` below the midpoint and `end − step·(N−1−i)` at or above it, with `step` itself
/// rounded to f32 first. The two disagree in the last ulp at most `N` — including `N = 4`, the
/// smallest schedule a 2-step parity fixture uses — and every downstream σ, `t` and
/// [`crate::dit::adaln::ScheduleKey`] is keyed on the exact bit pattern, so "close enough" would
/// make the port's schedule un-comparable with the reference's rather than merely imprecise.
///
/// Verified bitwise against `torch.linspace` for every `N` in `2..400` on the MLX lane, and against
/// the committed `av_denoise` fixture's own `out.video_sigmas` / `out.audio_sigmas` here.
fn descending_unit_linspace(steps: usize) -> Vec<f32> {
    debug_assert!(steps >= MIN_INFERENCE_STEPS);
    let (start, end) = (1.0f32, 0.0f32);
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|i| {
            if i < halfway {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            }
        })
        .collect()
}

/// Collapse runs of bitwise-equal adjacent values — `torch.unique_consecutive`.
fn unique_consecutive(values: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(values.len());
    for &v in values {
        if out.last().is_none_or(|&last| last.to_bits() != v.to_bits()) {
            out.push(v);
        }
    }
    out
}

/// One modality's sigma grid and the timesteps derived from it.
#[derive(Debug, Clone, PartialEq)]
pub struct SigmaSchedule {
    modality: DenoiseModality,
    shift: f32,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl SigmaSchedule {
    /// Build this modality's schedule at its **own** published shift.
    ///
    /// `num_inference_steps` counts the terminal `σ = 0`, so the model is evaluated
    /// [`Self::num_evals`] = `len(sigmas) − 1` times — one fewer than requested, and fewer still if
    /// the shift collapsed adjacent grid points.
    pub fn new(modality: DenoiseModality, num_inference_steps: usize) -> Result<Self> {
        Self::with_shift(modality, modality.sigma_shift(), num_inference_steps)
    }

    /// [`Self::new`] with an explicit shift — the reference's per-request `flow_shift` /
    /// `audio_flow_shift` override.
    ///
    /// The modality still travels with the schedule, so [`JointSchedule::new_from_parts`] can
    /// still reject a pair whose grids were built for the wrong streams.
    pub fn with_shift(
        modality: DenoiseModality,
        shift: f32,
        num_inference_steps: usize,
    ) -> Result<Self> {
        if !shift.is_finite() || shift <= 0.0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: the {} sigma shift must be finite and positive, got {shift}",
                modality.name()
            )));
        }
        if num_inference_steps < MIN_INFERENCE_STEPS {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: num_inference_steps must be >= {MIN_INFERENCE_STEPS} \
                 (the terminal sigma is part of the count, so {MIN_INFERENCE_STEPS} is one model \
                 evaluation), got {num_inference_steps}"
            )));
        }
        let base = descending_unit_linspace(num_inference_steps);
        let shifted: Vec<f32> = base.iter().map(|&b| shift_sigma(b, shift)).collect();
        let sigmas = unique_consecutive(&shifted);
        Self::from_sigmas(modality, shift, sigmas)
    }

    /// Adopt an explicit, already-shifted grid — the reference's `sigmas=` path, which applies
    /// **no** shift and **no** collapse.
    ///
    /// Requires at least two strictly decreasing values ending at exactly `0.0`, as the reference
    /// does.
    ///
    /// Finiteness is checked **first and separately** rather than folded into the ordering test: a
    /// `NaN` compares false against everything, so `w[1] >= w[0]` would wave it through, and
    /// [`crate::dit::adaln::TimestepSchedule`] keys on bit patterns where a `NaN` timestep can
    /// never be looked up again.
    pub fn from_sigmas(modality: DenoiseModality, shift: f32, sigmas: Vec<f32>) -> Result<Self> {
        if sigmas.len() < MIN_INFERENCE_STEPS
            || !sigmas.iter().all(|s| s.is_finite())
            || sigmas.windows(2).any(|w| w[1] >= w[0])
            || sigmas[sigmas.len() - 1] != 0.0
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: the {} sigma grid must hold at least \
                 {MIN_INFERENCE_STEPS} strictly decreasing values ending at 0.0, got {sigmas:?}",
                modality.name()
            )));
        }
        // t = 1 - sigma, over every sigma EXCEPT the terminal zero: the model is never evaluated
        // at sigma = 0, so there is no timestep for it.
        let timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|&s| 1.0 - s)
            .collect();
        Ok(Self {
            modality,
            shift,
            sigmas,
            timesteps,
        })
    }

    /// The modality this grid was built for.
    pub fn modality(&self) -> DenoiseModality {
        self.modality
    }

    /// The sigma shift this grid was built with.
    pub fn shift(&self) -> f32 {
        self.shift
    }

    /// The sigma grid, descending from 1.0 to a terminal 0.0.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// `t = 1 − σ` for every σ but the terminal zero — ascending, and never reaching 1.0.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Model evaluations this schedule drives: `len(sigmas) − 1`, **one fewer** than the requested
    /// `num_inference_steps`.
    pub fn num_evals(&self) -> usize {
        self.timesteps.len()
    }

    /// One Euler update on the rows this modality owns.
    ///
    /// `sample` is `x_t`, `velocity` is the transformer's **data-ward** prediction for the same
    /// rows. Both are read at float32; the result is float32, as the reference's latents are.
    ///
    /// ```text
    /// denoised = x_t + (1 − t)·v            (NOTE the +, and σ from the timestep)
    /// x_next   = r·x_t + (1 − r)·denoised   with r = σ[i+1]/σ[i] from the grid
    /// ```
    pub fn step(&self, index: usize, sample: &Tensor, velocity: &Tensor) -> Result<Tensor> {
        if index >= self.num_evals() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: {} step {index} is outside the schedule's {} evaluations",
                self.modality.name(),
                self.num_evals()
            )));
        }
        if sample.dims() != velocity.dims() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: {} sample {:?} and velocity {:?} must be the same rows",
                self.modality.name(),
                sample.dims(),
                velocity.dims()
            )));
        }
        let x = sample.to_dtype(DType::F32)?;
        let v = velocity.to_dtype(DType::F32)?;

        // sigma recovered from the TIMESTEP the model was conditioned on — deliberately not
        // `self.sigmas[index]`, which is a different float below sigma = 0.5.
        let sigma_from_timestep = 1.0 - self.timesteps[index];
        let denoised = (&x + (v * f64::from(sigma_from_timestep))?)?;

        // ...and the Euler ratio from the stored GRID.
        let ratio = self.sigmas[index + 1] / self.sigmas[index];
        Ok(((x * f64::from(ratio))? + (denoised * f64::from(1.0 - ratio))?)?)
    }
}

/// The two schedules a joint run steps in lockstep — video at 12.0, audio at 3.0.
///
/// Both are built at the same `num_inference_steps` and advanced by the same index `i`, but they
/// are *different grids*: at every interior step the video and audio rows of the one packed
/// sequence sit at different noise levels. That is the whole reason the sequence carries a per-row
/// time axis.
#[derive(Debug, Clone, PartialEq)]
pub struct JointSchedule {
    video: SigmaSchedule,
    audio: SigmaSchedule,
}

impl JointSchedule {
    /// Build both schedules at their own published shifts.
    pub fn new(num_inference_steps: usize) -> Result<Self> {
        Self::new_from_parts(
            SigmaSchedule::new(DenoiseModality::Video, num_inference_steps)?,
            SigmaSchedule::new(DenoiseModality::Audio, num_inference_steps)?,
        )
    }

    /// Build both at explicit shifts — the reference's `flow_shift` / `audio_flow_shift` overrides.
    pub fn with_shifts(
        num_inference_steps: usize,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
        Self::new_from_parts(
            SigmaSchedule::with_shift(DenoiseModality::Video, video_shift, num_inference_steps)?,
            SigmaSchedule::with_shift(DenoiseModality::Audio, audio_shift, num_inference_steps)?,
        )
    }

    /// Pair two pre-built schedules, **rejecting a swap**.
    ///
    /// Each grid carries the modality it was built for, so a caller that passed the audio schedule
    /// as the video one gets a typed error instead of a run whose picture is denoised on the
    /// soundtrack's curve. Also rejects a mismatched evaluation count: the reference `zip`s the two
    /// timestep lists, and a silent `zip` would truncate to the shorter one.
    pub fn new_from_parts(video: SigmaSchedule, audio: SigmaSchedule) -> Result<Self> {
        for (schedule, want) in [
            (&video, DenoiseModality::Video),
            (&audio, DenoiseModality::Audio),
        ] {
            if schedule.modality() != want {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 scheduler: the {} slot was given a schedule built for {} (shift \
                     {}); each sigma shift is bound to its own modality — 12.0 video, 3.0 audio",
                    want.name(),
                    schedule.modality().name(),
                    schedule.shift()
                )));
            }
        }
        if video.num_evals() != audio.num_evals() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: the video schedule drives {} evaluations and the audio one \
                 {}; both modalities step once per shared transformer pass",
                video.num_evals(),
                audio.num_evals()
            )));
        }
        Ok(Self { video, audio })
    }

    /// The video schedule — always the one built at [`DenoiseModality::Video`].
    pub fn video(&self) -> &SigmaSchedule {
        &self.video
    }

    /// The audio schedule — always the one built at [`DenoiseModality::Audio`].
    pub fn audio(&self) -> &SigmaSchedule {
        &self.audio
    }

    /// Model evaluations the joint loop runs — **one** transformer forward each.
    pub fn num_evals(&self) -> usize {
        self.video.num_evals()
    }

    /// This modality's schedule.
    pub fn of(&self, modality: DenoiseModality) -> &SigmaSchedule {
        match modality {
            DenoiseModality::Video => &self.video,
            DenoiseModality::Audio => &self.audio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The published shifts, and that they are reachable only through their own modality.
    #[test]
    fn each_shift_is_bound_to_its_modality() {
        assert_eq!(DenoiseModality::Video.sigma_shift(), 12.0);
        assert_eq!(DenoiseModality::Audio.sigma_shift(), 3.0);
        assert_eq!(VIDEO_SIGMA_SHIFT, 12.0);
        assert_eq!(AUDIO_SIGMA_SHIFT, 3.0);
        assert_eq!(DenoiseModality::Video.token_tag(), 0);
        assert_eq!(DenoiseModality::Audio.token_tag(), 2);

        let j = JointSchedule::new(8).unwrap();
        assert_eq!(j.video().shift(), 12.0);
        assert_eq!(j.audio().shift(), 3.0);
        assert_eq!(j.video().modality(), DenoiseModality::Video);
        assert_eq!(j.audio().modality(), DenoiseModality::Audio);
    }

    /// **The swap mutation.** Two claims: pairing the schedules the wrong way round is a typed
    /// error, and — because a type error alone would be satisfied by two identical grids — the two
    /// grids are measurably different, so a swap that slipped past the type would change the
    /// output.
    #[test]
    fn swapping_the_shifts_is_observable() {
        let video = SigmaSchedule::new(DenoiseModality::Video, 8).unwrap();
        let audio = SigmaSchedule::new(DenoiseModality::Audio, 8).unwrap();

        let e = JointSchedule::new_from_parts(audio.clone(), video.clone())
            .unwrap_err()
            .to_string();
        assert!(e.contains("bound to its own modality"), "{e}");

        // ...and the swap is numerically real, not just mislabelled. Relative max-abs-diff over the
        // interior of the grid, which is what a norm or a checksum would miss (both grids run 1→0).
        let (v, a) = (video.sigmas(), audio.sigmas());
        assert_eq!(v.len(), a.len());
        let peak = v
            .iter()
            .zip(a)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            peak > 0.2,
            "the two shifts must produce materially different grids, got peak |Δσ| {peak:.3e}"
        );
        // Both still start at 1 and end at 0 — which is exactly why the endpoints prove nothing.
        assert_eq!((v[0], a[0]), (1.0, 1.0));
        assert_eq!((v[v.len() - 1], a[a.len() - 1]), (0.0, 0.0));
    }

    /// **The one-shift-for-both mutation.** Building the audio grid at the video shift must be
    /// observably different from building it at its own.
    #[test]
    fn applying_one_shift_to_both_modalities_is_observable() {
        let correct = SigmaSchedule::new(DenoiseModality::Audio, 8).unwrap();
        let wrong =
            SigmaSchedule::with_shift(DenoiseModality::Audio, VIDEO_SIGMA_SHIFT, 8).unwrap();
        let peak = correct
            .sigmas()
            .iter()
            .zip(wrong.sigmas())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            peak > 0.2,
            "audio denoised on the video curve must differ measurably, got {peak:.3e}"
        );
        // The timesteps move with it — this is what the AdaLN table is keyed on.
        assert_ne!(correct.timesteps(), wrong.timesteps());
    }

    /// **The double-apply mutation.** Applying the shift twice is a different, still-monotone,
    /// still-1→0 grid — so only a value comparison can see it.
    #[test]
    fn double_applying_a_shift_is_observable() {
        let once = shift_sigma(0.5, 12.0);
        let twice = shift_sigma(once, 12.0);
        assert!((once - 12.0 * 0.5 / 6.5).abs() < 1e-6, "{once}");
        assert!(
            (twice - once).abs() > 0.05,
            "a doubled shift must move sigma, got {once} then {twice}"
        );
        // ...and the endpoints are fixed points of the shift, so they cannot detect it.
        for s in [12.0f32, 3.0] {
            assert_eq!(shift_sigma(1.0, s), 1.0);
            assert_eq!(shift_sigma(0.0, s), 0.0);
        }
    }

    /// `N` requests `N − 1` evaluations, and the terminal zero has no timestep. `t = 1 − σ`, and
    /// `t` never reaches 1.0 because the model is never evaluated at σ = 0.
    #[test]
    fn the_terminal_zero_is_inside_the_requested_step_count() {
        for n in [2usize, 3, 4, 8, 50] {
            let s = SigmaSchedule::new(DenoiseModality::Video, n).unwrap();
            assert_eq!(s.sigmas().len(), n, "no collapse is expected at n={n}");
            assert_eq!(s.num_evals(), n - 1, "n={n} drives one evaluation fewer");
            assert_eq!(s.timesteps().len(), n - 1);
            assert_eq!(s.sigmas()[n - 1], 0.0);
            assert_eq!(s.timesteps()[0], 0.0, "sigma 1 is t 0 — fully noisy");
            assert!(
                *s.timesteps().last().unwrap() < 1.0,
                "t = 1 is clean and is never evaluated"
            );
            for (t, sigma) in s.timesteps().iter().zip(s.sigmas()) {
                assert_eq!(*t, 1.0 - sigma);
            }
        }
        assert!(SigmaSchedule::new(DenoiseModality::Video, 1).is_err());
        assert!(SigmaSchedule::new(DenoiseModality::Video, 0).is_err());
    }

    /// `unique_consecutive` collapses bitwise-adjacent duplicates, and the grid stays strictly
    /// decreasing afterwards.
    #[test]
    fn duplicate_grid_points_are_collapsed() {
        assert_eq!(
            unique_consecutive(&[1.0, 1.0, 0.5, 0.5, 0.5, 0.0]),
            vec![1.0, 0.5, 0.0]
        );
        assert_eq!(unique_consecutive(&[1.0, 0.5, 1.0]), vec![1.0, 0.5, 1.0]);
        assert!(unique_consecutive(&[]).is_empty());
        // A collapse shortens the schedule rather than being an error: `num_evals` is derived from
        // the grid, never from the request.
        let s =
            SigmaSchedule::from_sigmas(DenoiseModality::Audio, 3.0, vec![1.0, 0.5, 0.0]).unwrap();
        assert_eq!(s.num_evals(), 2);
    }

    /// The explicit-sigmas path validates the reference's three conditions.
    #[test]
    fn an_explicit_grid_must_descend_to_zero() {
        SigmaSchedule::from_sigmas(DenoiseModality::Video, 12.0, vec![1.0, 0.4, 0.0]).unwrap();
        for bad in [
            vec![1.0],
            vec![1.0, 0.4],           // no terminal zero
            vec![1.0, 0.4, 0.4, 0.0], // not strictly decreasing
            vec![0.4, 1.0, 0.0],      // not descending
            vec![1.0, f32::NAN, 0.0], // a NaN compares false against every ordering test
            vec![1.0, f32::INFINITY, 0.0],
        ] {
            assert!(
                SigmaSchedule::from_sigmas(DenoiseModality::Video, 12.0, bad.clone()).is_err(),
                "{bad:?} must be rejected"
            );
        }
        assert!(SigmaSchedule::with_shift(DenoiseModality::Video, 0.0, 4).is_err());
        assert!(SigmaSchedule::with_shift(DenoiseModality::Video, f32::NAN, 4).is_err());
    }

    /// **The reversed velocity sign.** With `v = +1` and `x_t = 0`, a data-ward velocity moves the
    /// sample *up*; diffusers' convention would move it down. Checked at a step whose ratio is not
    /// 0 or 1, so the sign survives into the blend.
    #[test]
    fn the_velocity_sign_is_data_ward() {
        let dev = Device::Cpu;
        let s = SigmaSchedule::new(DenoiseModality::Video, 4).unwrap();
        let x = Tensor::from_vec(vec![0.0f32, 0.0], (2,), &dev).unwrap();
        let v = Tensor::from_vec(vec![1.0f32, -1.0], (2,), &dev).unwrap();
        let got = flat(&s.step(1, &x, &v).unwrap());
        // denoised = 0 + (1 - t)·v = sigma_from_t · v; x_next = (1-r)·denoised since x = 0.
        let sigma_from_t = 1.0 - s.timesteps()[1];
        let r = s.sigmas()[2] / s.sigmas()[1];
        let want = (1.0 - r) * sigma_from_t;
        assert!((got[0] - want).abs() < 1e-6, "{got:?} vs {want}");
        assert!(
            got[0] > 0.0,
            "a data-ward velocity must move the sample UP (x0 = x_t + sigma·v)"
        );
        assert!((got[1] + want).abs() < 1e-6, "and antisymmetric: {got:?}");
    }

    /// The two σ in one step come from different sources, and below σ = 0.5 they are different
    /// floats. Pinned so a "simplification" that reuses one is a failure.
    #[test]
    fn the_step_reads_sigma_from_two_sources() {
        let s = SigmaSchedule::new(DenoiseModality::Audio, 16).unwrap();
        let mut disagreements = 0;
        for i in 0..s.num_evals() {
            let from_grid = s.sigmas()[i];
            let from_timestep = 1.0 - s.timesteps()[i];
            if from_grid.to_bits() != from_timestep.to_bits() {
                disagreements += 1;
                assert!(
                    (from_grid - from_timestep).abs() < 1e-6,
                    "…but only by an ulp"
                );
            }
        }
        assert!(
            disagreements > 0,
            "the float32 round trip 1 - (1 - sigma) must be inexact somewhere in the grid, or the \
             reference's two-source split would be pointless"
        );
    }

    /// A terminal step lands exactly on the clean sample: `r = 0` makes `x_next = denoised`.
    #[test]
    fn the_last_step_returns_the_denoised_sample() {
        let dev = Device::Cpu;
        let s = SigmaSchedule::new(DenoiseModality::Video, 4).unwrap();
        let last = s.num_evals() - 1;
        assert_eq!(s.sigmas()[last + 1], 0.0);
        let x = Tensor::from_vec(vec![2.0f32], (1,), &dev).unwrap();
        let v = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let got = flat(&s.step(last, &x, &v).unwrap())[0];
        let want = 2.0 + (1.0 - s.timesteps()[last]) * 0.5;
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        assert!(s.step(last + 1, &x, &v).is_err(), "past the end");
        assert!(
            s.step(
                0,
                &x,
                &Tensor::from_vec(vec![1.0f32, 2.0], (2,), &dev).unwrap()
            )
            .is_err(),
            "mismatched rows"
        );
    }

    /// Both schedules run the same number of evaluations, and a mismatched pair is rejected rather
    /// than silently zipped to the shorter one.
    #[test]
    fn a_joint_schedule_steps_both_modalities_once_per_pass() {
        let j = JointSchedule::new(8).unwrap();
        assert_eq!(j.num_evals(), 7);
        assert_eq!(j.video().num_evals(), j.audio().num_evals());
        assert_eq!(j.of(DenoiseModality::Audio).shift(), 3.0);

        let short = SigmaSchedule::new(DenoiseModality::Audio, 4).unwrap();
        let long = SigmaSchedule::new(DenoiseModality::Video, 8).unwrap();
        let e = JointSchedule::new_from_parts(long, short)
            .unwrap_err()
            .to_string();
        assert!(e.contains("evaluations"), "{e}");
        assert!(JointSchedule::new(1).is_err());
    }

    /// The conditioning constants, and the `max(video_t, 0.999)` rule that lets the anchors follow
    /// the video schedule once it passes them.
    #[test]
    fn conditioning_timesteps_are_the_published_constants() {
        assert_eq!(KEYFRAME_NOISE_AUG, 0.999);
        assert_eq!(REFERENCE_AUDIO_TIMESTEP, 1.0);
        // A grid that reaches past 0.999 — the shift makes an ordinary step count stop well short
        // of it (a 2000-step run only reaches t = 0.994), so this is stated explicitly.
        let s = SigmaSchedule::from_sigmas(
            DenoiseModality::Video,
            VIDEO_SIGMA_SHIFT,
            vec![1.0, 0.5, 0.000_5, 0.0],
        )
        .unwrap();
        let anchored: Vec<f32> = s
            .timesteps()
            .iter()
            .map(|&t| t.max(KEYFRAME_NOISE_AUG))
            .collect();
        assert_eq!(anchored[0], KEYFRAME_NOISE_AUG, "pinned early");
        assert_eq!(anchored[1], KEYFRAME_NOISE_AUG, "still pinned at t = 0.5");
        assert!(
            *anchored.last().unwrap() > KEYFRAME_NOISE_AUG,
            "and follows the video timestep once it passes 0.999, got {:?}",
            anchored.last()
        );
    }
}
