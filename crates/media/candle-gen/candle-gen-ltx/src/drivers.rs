//! Typed ENG-6 `FamilyDriver` implementations and stream-targeted boundary transitions for
//! `candle-gen-ltx` (`PIPE-4`, `PIPE-6`, `PIPE-7`, `DEC-3`, `DEC-19`).
//!
//! All native Candle LTX denoising loops (`denoise_av_conditioned`, `denoise_video_dev_conditioned`,
//! `denoise_av_dev_conditioned`, `denoise_tokens_rf_ancestral`, and `dfr::denoise_dfr_tile`) execute
//! through these generic `FamilyDriver` structs and [`candle_gen::run_candle_driver_slices`],
//! preserving exact AP13 class-(a) math/f64-over-f32 precision, per-token timing, RNG stream draws,
//! and cancellation while enabling weights-free Tier 0 conformance testing over mocked forward seams.

use std::cell::RefCell;

use candle_gen::candle_core::{DType, Device, Result as CCoreResult, Tensor};
use candle_gen::driver::{
    DriverEvaluationKind, DriverRefusalReasons, DriverRunState, DriverStepInfo, DriverSupport,
    FamilyDriver, SliceInput, StepHooks, StreamId, StreamStepSigma,
};
use candle_gen::gen_core::multistream::{
    ResolvedSchedule, StreamTransition, TransitionBoundaryGuard,
};
use candle_gen::gen_core::{BoundaryState, Error as GenError, EstimateSource, Result as GenResult};
use candle_gen::{CandleError, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::conditioning::apply_denoise_mask;
use crate::dev_sampler;
use crate::params::GuiderParams;
use crate::pipeline::{flatten_audio_latent, unflatten_audio_latent};

/// Multi-stream latent state carried through [`FamilyDriver<CandleAvState>`] slices (`PIPE-4`,
/// `PIPE-6`).
#[derive(Clone, Debug)]
pub struct CandleAvState {
    pub video: Tensor,
    pub audio: Option<Tensor>,
}

impl CandleAvState {
    pub fn av(video: Tensor, audio: Tensor) -> Self {
        Self {
            video,
            audio: Some(audio),
        }
    }

    pub fn video_only(video: Tensor) -> Self {
        Self { video, audio: None }
    }
}

/// Borrowed per-token clean latent and denoise mask (`[B, S, C]` and `[B, S, 1]`).
#[derive(Clone, Copy)]
pub struct CandleVideoTokenMask<'a> {
    pub clean_latent: &'a Tensor,
    pub denoise_mask: &'a Tensor,
}

impl<'a> CandleVideoTokenMask<'a> {
    pub fn token_timesteps(&self, sigma: f32) -> CCoreResult<Tensor> {
        let (b, s, _one) = self.denoise_mask.dims3()?;
        self.denoise_mask.reshape((b, s))? * sigma as f64
    }

    pub fn pin_denoised(&self, denoised: &Tensor) -> CCoreResult<Tensor> {
        apply_denoise_mask(denoised, self.clean_latent, self.denoise_mask)
    }

    pub fn video_denoised(
        &self,
        x_sigma: &Tensor,
        vel: &Tensor,
        sigma: f32,
    ) -> CCoreResult<Tensor> {
        let raw = (x_sigma - (&vel.to_dtype(DType::F32)? * sigma as f64)?)?;
        self.pin_denoised(&raw)
    }
}

pub(crate) fn ltx_driver_support(driver_id: &'static str) -> DriverSupport {
    DriverSupport {
        driver_id,
        arbitrary_slice: true,
        carry_history: true,
        reset_history: true,
        renoise_from_clean: false,
        geometry_change: false,
        read_only_boundary_estimate: true,
        previous_estimate_capture: false,
        uses_global_rng: false,
        refusal_reasons: DriverRefusalReasons {
            renoise_from_clean:
                "LTX cross-stage re-noising is governed by explicit StreamTransition boundaries",
            geometry_change:
                "LTX geometry changes execute through learned upsampler transitions between slices",
            ..DriverRefusalReasons::default()
        },
    }
}

fn core_err(err: CandleError) -> GenError {
    err.into()
}

fn ccore_err(err: candle_gen::candle_core::Error) -> GenError {
    CandleError::from(err).into()
}

fn check_schedule_match<L>(
    driver_id: &'static str,
    schedule: &ResolvedSchedule,
    input: &SliceInput<'_, L>,
) -> GenResult<()> {
    if input.total_steps != schedule.num_steps() || input.span.end > schedule.num_steps() {
        return Err(GenError::Unsupported(format!(
            "{driver_id}: slice {:?} / total_steps={} disagrees with resolved schedule '{}' ({} steps)",
            input.span,
            input.total_steps,
            schedule.table_id,
            schedule.num_steps()
        )));
    }
    Ok(())
}

fn extract_state<L>(state: BoundaryState<L>) -> L {
    match state {
        BoundaryState::XSigma(l)
        | BoundaryState::CleanEstimate { latent: l, .. }
        | BoundaryState::RouteSpecific { state: l, .. } => l,
    }
}

fn make_step_info(
    pass: usize,
    phase: usize,
    total_outer_steps: usize,
    step: usize,
    sigma: f32,
    next_sigma: f32,
    kind: DriverEvaluationKind,
) -> DriverStepInfo {
    DriverStepInfo {
        pass,
        phase,
        outer_step: step,
        total_outer_steps,
        eval_in_step: 0,
        kind,
        sigma,
        next_sigma,
    }
}

fn make_stream_sigmas(has_audio: bool, sigma: f32, next_sigma: f32) -> Vec<StreamStepSigma> {
    if has_audio {
        vec![
            StreamStepSigma {
                stream: StreamId::Video,
                sigma,
                next_sigma,
            },
            StreamStepSigma {
                stream: StreamId::Audio,
                sigma,
                next_sigma,
            },
        ]
    } else {
        vec![StreamStepSigma {
            stream: StreamId::Video,
            sigma,
            next_sigma,
        }]
    }
}

fn audio_denoised(
    alat: &Tensor,
    avel: &Tensor,
    audio_frames: usize,
    sigma: f32,
) -> CCoreResult<Tensor> {
    let avel = unflatten_audio_latent(&avel.to_dtype(DType::F32)?, audio_frames)?;
    alat - (&avel * sigma as f64)?
}

fn euler_from_denoised(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f32,
    sigma_next: f32,
) -> CCoreResult<Tensor> {
    if sigma_next <= 0.0 {
        Ok(denoised.clone())
    } else {
        let step = (((x - denoised)? * sigma_next as f64)? / sigma as f64)?;
        denoised + step
    }
}

/// Apply an explicit [`StreamTransition`] guarded by [`TransitionBoundaryGuard`] between stages
/// (`PIPE-4`).
pub(crate) fn apply_candle_stream_transition(
    stream: StreamId,
    transition: &StreamTransition,
    clean_state: BoundaryState<Tensor>,
    op: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<BoundaryState<Tensor>> {
    use candle_gen::gen_core::multistream::StreamBoundarySemantics;
    let (_, expected_exit) = transition
        .kind
        .boundary_semantics(&transition.transition_id)?;
    TransitionBoundaryGuard::new(stream, clean_state)
        .apply(transition, |state| match state {
            BoundaryState::CleanEstimate { latent, .. } => {
                let next = op(latent).map_err(GenError::from)?;
                match expected_exit {
                    StreamBoundarySemantics::CleanEstimate => Ok(BoundaryState::CleanEstimate {
                        latent: next,
                        source: EstimateSource::EndpointEvaluation,
                    }),
                    StreamBoundarySemantics::NoisyXSigma => Ok(BoundaryState::XSigma(next)),
                    StreamBoundarySemantics::RouteSpecific => Err(GenError::Unsupported(format!(
                        "transition '{}': CleanEstimate entry cannot exit as RouteSpecific",
                        transition.transition_id
                    ))),
                }
            }
            BoundaryState::RouteSpecific { state, label } => {
                let next = op(state).map_err(GenError::from)?;
                Ok(BoundaryState::RouteSpecific { state: next, label })
            }
            BoundaryState::XSigma(x) => {
                let next = op(x).map_err(GenError::from)?;
                Ok(BoundaryState::XSigma(next))
            }
        })
        .map_err(Into::into)
}

/// Build a typed pre-load [`gen_core_ltx::LtxRenderPlan`] from a Candle request and loaded
/// route identity without reading or executing model weights (`DEC-3`, `DEC-13`, `MEM-3`).
pub fn render_plan_for_request(
    family: gen_core_ltx::LtxModelFamily,
    variant: crate::dev_sampler::TransformerVariant,
    req: &candle_gen::gen_core::GenerationRequest,
    resolved_frames: u32,
    fps: f32,
    has_temporal_upsampler: bool,
) -> Result<gen_core_ltx::LtxRenderPlan> {
    let plan_variant = match variant {
        crate::dev_sampler::TransformerVariant::Distilled => gen_core_ltx::LtxVariant::Distilled,
        crate::dev_sampler::TransformerVariant::Dev => gen_core_ltx::LtxVariant::Dev,
    };
    gen_core_ltx::LtxRenderPlan::from_generation_request(
        family,
        plan_variant,
        gen_core_ltx::MemoryBudgetDomainKey::DeviceVram,
        req,
        resolved_frames,
        fps,
        has_temporal_upsampler,
    )
    .map_err(Into::into)
}

// =================================================================================================
// Multi-Stream Distilled AV Token Driver (`denoise_av_conditioned`)
// =================================================================================================

/// Typed [`FamilyDriver<CandleAvState>`] for conditioned distilled AV token denoising
/// (`denoise_av_conditioned_with_scales`).
pub struct LtxCandleDistilledAvDriver<'a, F> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_mask: CandleVideoTokenMask<'a>,
    audio_frames: usize,
    forward: RefCell<F>,
}

impl<'a, F> LtxCandleDistilledAvDriver<'a, F>
where
    F: FnMut(&Tensor, &Tensor, &Tensor, f32, DriverEvaluationKind) -> Result<(Tensor, Tensor)>,
{
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_mask: CandleVideoTokenMask<'a>,
        audio_frames: usize,
        forward: F,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_mask,
            audio_frames,
            forward: RefCell::new(forward),
        }
    }

    fn evaluate_clean_x0(
        &self,
        cur: &CandleAvState,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<(Tensor, Tensor)> {
        let alat = cur.audio.as_ref().ok_or_else(|| {
            CandleError::Msg("LtxCandleDistilledAvDriver requires audio state".into())
        })?;
        let aflat = flatten_audio_latent(alat)?;
        let timesteps = self.video_mask.token_timesteps(sigma)?;
        let (vvel, avel) =
            (self.forward.borrow_mut())(&cur.video, &timesteps, &aflat, sigma, kind)?;
        let vden = self.video_mask.video_denoised(&cur.video, &vvel, sigma)?;
        let aden = audio_denoised(alat, &avel, self.audio_frames, sigma)?;
        Ok((vden, aden))
    }
}

impl<F> FamilyDriver<CandleAvState> for LtxCandleDistilledAvDriver<'_, F>
where
    F: FnMut(&Tensor, &Tensor, &Tensor, f32, DriverEvaluationKind) -> Result<(Tensor, Tensor)>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, CandleAvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, CandleAvState>,
    ) -> GenResult<BoundaryState<CandleAvState>> {
        self.support.validate_slice(&input, state)?;
        check_schedule_match(self.support.driver_id, &self.schedule, &input)?;
        state.enter_slice(&input)?;

        let SliceInput {
            pass,
            phase,
            span,
            total_steps,
            state: boundary_state,
            request_boundary_estimate,
            cancel,
            ..
        } = input;
        let mut cur = extract_state(boundary_state);
        if cur.audio.is_none() {
            return Err(GenError::Unsupported(format!(
                "{}: requires active audio stream",
                self.support.driver_id
            )));
        }

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (sigma, sigma_next) = self
                .schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing schedule window at step {step}")))?;
            let info = make_step_info(
                pass,
                phase,
                total_steps,
                step,
                sigma,
                sigma_next,
                DriverEvaluationKind::Trajectory,
            );
            let stream_sigmas = make_stream_sigmas(true, sigma, sigma_next);
            hooks.observe_streams(&info, &stream_sigmas, &cur)?;
            state.record_stream_step(&info, stream_sigmas);

            let (vden, aden) = self
                .evaluate_clean_x0(&cur, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            cur.video =
                euler_from_denoised(&cur.video, &vden, sigma, sigma_next).map_err(ccore_err)?;
            let alat = cur.audio.take().expect("audio checked");
            cur.audio =
                Some(euler_from_denoised(&alat, &aden, sigma, sigma_next).map_err(ccore_err)?);

            cur = hooks.post_step(&info, cur)?;
            hooks.materialize(&info, &cur)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(span.end);

        if request_boundary_estimate {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let boundary_sigma = self.schedule.sigmas()[span.end];
            if boundary_sigma > 0.0 {
                let (vden, aden) = self
                    .evaluate_clean_x0(&cur, boundary_sigma, DriverEvaluationKind::BoundaryEstimate)
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: CandleAvState::av(vden, aden),
                    source: EstimateSource::EndpointEvaluation,
                });
            }
            state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
            return Ok(BoundaryState::CleanEstimate {
                latent: cur,
                source: EstimateSource::PreviousEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(cur))
    }
}

// =================================================================================================
// Single-Stream Video Dev Guided Driver (`denoise_video_dev_conditioned`)
// =================================================================================================

/// Typed [`FamilyDriver<Tensor>`] for video-only Dev 3-branch guidance
/// (`denoise_video_dev_conditioned`).
pub struct LtxCandleVideoDevDriver<'a, F> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_mask: CandleVideoTokenMask<'a>,
    video_guider: GuiderParams,
    forward_step: RefCell<F>,
}

impl<'a, F> LtxCandleVideoDevDriver<'a, F>
where
    F: FnMut(&Tensor, &Tensor, f32, DriverEvaluationKind) -> Result<(Tensor, Tensor, Tensor)>,
{
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_mask: CandleVideoTokenMask<'a>,
        video_guider: GuiderParams,
        forward_step: F,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_mask,
            video_guider,
            forward_step: RefCell::new(forward_step),
        }
    }

    fn evaluate_guided_x0(
        &self,
        cur: &Tensor,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<Tensor> {
        let timesteps = self.video_mask.token_timesteps(sigma)?;
        let (cond_vel, neg_vel, stg_vel) =
            (self.forward_step.borrow_mut())(cur, &timesteps, sigma, kind)?;
        let cond = self.video_mask.video_denoised(cur, &cond_vel, sigma)?;
        let neg = self.video_mask.video_denoised(cur, &neg_vel, sigma)?;
        let stg = self.video_mask.video_denoised(cur, &stg_vel, sigma)?;
        Ok(dev_sampler::combine_guidance(
            &cond,
            &neg,
            &stg,
            &cond,
            self.video_guider,
        )?)
    }
}

impl<F> FamilyDriver<Tensor> for LtxCandleVideoDevDriver<'_, F>
where
    F: FnMut(&Tensor, &Tensor, f32, DriverEvaluationKind) -> Result<(Tensor, Tensor, Tensor)>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> GenResult<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        check_schedule_match(self.support.driver_id, &self.schedule, &input)?;
        state.enter_slice(&input)?;

        let SliceInput {
            pass,
            phase,
            span,
            total_steps,
            state: boundary_state,
            request_boundary_estimate,
            cancel,
            ..
        } = input;
        let mut cur = extract_state(boundary_state);

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (sigma, sigma_next) = self
                .schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing schedule window at step {step}")))?;
            let info = make_step_info(
                pass,
                phase,
                total_steps,
                step,
                sigma,
                sigma_next,
                DriverEvaluationKind::Trajectory,
            );
            let stream_sigmas = make_stream_sigmas(false, sigma, sigma_next);
            hooks.observe_streams(&info, &stream_sigmas, &cur)?;
            state.record_stream_step(&info, stream_sigmas);

            let video_denoised = self
                .evaluate_guided_x0(&cur, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            for _ in 0..3 {
                state.record_evaluation(DriverEvaluationKind::Trajectory);
            }
            cur =
                euler_from_denoised(&cur, &video_denoised, sigma, sigma_next).map_err(ccore_err)?;

            cur = hooks.post_step(&info, cur)?;
            hooks.materialize(&info, &cur)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(span.end);

        if request_boundary_estimate {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let boundary_sigma = self.schedule.sigmas()[span.end];
            if boundary_sigma > 0.0 {
                let vden = self
                    .evaluate_guided_x0(
                        &cur,
                        boundary_sigma,
                        DriverEvaluationKind::BoundaryEstimate,
                    )
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: vden,
                    source: EstimateSource::EndpointEvaluation,
                });
            }
            state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
            return Ok(BoundaryState::CleanEstimate {
                latent: cur,
                source: EstimateSource::PreviousEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(cur))
    }
}

// =================================================================================================
// Multi-Stream AV Dev 4-Pass Guided Driver (`denoise_av_dev_conditioned`)
// =================================================================================================

/// Raw 4-branch `(video_vel, audio_vel)` outputs for one Dev AV transition step.
pub struct CandleDevAvBranches {
    pub conditional: (Tensor, Tensor),
    pub negative: (Tensor, Tensor),
    pub perturbed: (Tensor, Tensor),
    pub isolated: (Tensor, Tensor),
}

/// Typed [`FamilyDriver<CandleAvState>`] for non-distilled 4-pass AV Dev guided Euler
/// (`denoise_av_dev_conditioned_with_scales`).
pub struct LtxCandleAvDevDriver<'a, F> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_mask: CandleVideoTokenMask<'a>,
    audio_frames: usize,
    video_guider: GuiderParams,
    audio_guider: GuiderParams,
    forward_branches: RefCell<F>,
}

impl<'a, F> LtxCandleAvDevDriver<'a, F>
where
    F: FnMut(&Tensor, &Tensor, &Tensor, f32, DriverEvaluationKind) -> Result<CandleDevAvBranches>,
{
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_mask: CandleVideoTokenMask<'a>,
        audio_frames: usize,
        video_guider: GuiderParams,
        audio_guider: GuiderParams,
        forward_branches: F,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_mask,
            audio_frames,
            video_guider,
            audio_guider,
            forward_branches: RefCell::new(forward_branches),
        }
    }

    fn evaluate_guided_x0(
        &self,
        cur: &CandleAvState,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<(Tensor, Tensor)> {
        let alat = cur
            .audio
            .as_ref()
            .ok_or_else(|| CandleError::Msg("LtxCandleAvDevDriver requires audio state".into()))?;
        let aflat = flatten_audio_latent(alat)?;
        let timesteps = self.video_mask.token_timesteps(sigma)?;
        let branches =
            (self.forward_branches.borrow_mut())(&cur.video, &timesteps, &aflat, sigma, kind)?;

        let conditional_video =
            self.video_mask
                .video_denoised(&cur.video, &branches.conditional.0, sigma)?;
        let negative_video =
            self.video_mask
                .video_denoised(&cur.video, &branches.negative.0, sigma)?;
        let perturbed_video =
            self.video_mask
                .video_denoised(&cur.video, &branches.perturbed.0, sigma)?;
        let isolated_video =
            self.video_mask
                .video_denoised(&cur.video, &branches.isolated.0, sigma)?;

        let conditional_audio =
            audio_denoised(alat, &branches.conditional.1, self.audio_frames, sigma)?;
        let negative_audio = audio_denoised(alat, &branches.negative.1, self.audio_frames, sigma)?;
        let perturbed_audio =
            audio_denoised(alat, &branches.perturbed.1, self.audio_frames, sigma)?;
        let isolated_audio = audio_denoised(alat, &branches.isolated.1, self.audio_frames, sigma)?;

        let video_denoised = dev_sampler::combine_guidance(
            &conditional_video,
            &negative_video,
            &perturbed_video,
            &isolated_video,
            self.video_guider,
        )?;
        let audio_denoised = dev_sampler::combine_guidance(
            &conditional_audio,
            &negative_audio,
            &perturbed_audio,
            &isolated_audio,
            self.audio_guider,
        )?;
        Ok((video_denoised, audio_denoised))
    }
}

impl<F> FamilyDriver<CandleAvState> for LtxCandleAvDevDriver<'_, F>
where
    F: FnMut(&Tensor, &Tensor, &Tensor, f32, DriverEvaluationKind) -> Result<CandleDevAvBranches>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, CandleAvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, CandleAvState>,
    ) -> GenResult<BoundaryState<CandleAvState>> {
        self.support.validate_slice(&input, state)?;
        check_schedule_match(self.support.driver_id, &self.schedule, &input)?;
        state.enter_slice(&input)?;

        let SliceInput {
            pass,
            phase,
            span,
            total_steps,
            state: boundary_state,
            request_boundary_estimate,
            cancel,
            ..
        } = input;
        let mut cur = extract_state(boundary_state);
        if cur.audio.is_none() {
            return Err(GenError::Unsupported(format!(
                "{}: requires active audio stream",
                self.support.driver_id
            )));
        }

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (sigma, sigma_next) = self
                .schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing schedule window at step {step}")))?;
            let info = make_step_info(
                pass,
                phase,
                total_steps,
                step,
                sigma,
                sigma_next,
                DriverEvaluationKind::Trajectory,
            );
            let stream_sigmas = make_stream_sigmas(true, sigma, sigma_next);
            hooks.observe_streams(&info, &stream_sigmas, &cur)?;
            state.record_stream_step(&info, stream_sigmas);

            let (video_denoised, audio_denoised) = self
                .evaluate_guided_x0(&cur, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            for _ in 0..4 {
                state.record_evaluation(DriverEvaluationKind::Trajectory);
            }
            cur.video = euler_from_denoised(&cur.video, &video_denoised, sigma, sigma_next)
                .map_err(ccore_err)?;
            let alat = cur.audio.take().expect("audio checked");
            cur.audio = Some(
                euler_from_denoised(&alat, &audio_denoised, sigma, sigma_next)
                    .map_err(ccore_err)?,
            );

            cur = hooks.post_step(&info, cur)?;
            hooks.materialize(&info, &cur)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(span.end);

        if request_boundary_estimate {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let boundary_sigma = self.schedule.sigmas()[span.end];
            if boundary_sigma > 0.0 {
                let (vden, aden) = self
                    .evaluate_guided_x0(
                        &cur,
                        boundary_sigma,
                        DriverEvaluationKind::BoundaryEstimate,
                    )
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: CandleAvState::av(vden, aden),
                    source: EstimateSource::EndpointEvaluation,
                });
            }
            state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
            return Ok(BoundaryState::CleanEstimate {
                latent: cur,
                source: EstimateSource::PreviousEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(cur))
    }
}

// =================================================================================================
// Multi-Stream RF Ancestral Driver (`denoise_tokens_rf_ancestral` & DFR tiles)
// =================================================================================================

/// Seeded noise generator seam for [`LtxCandleRfAncestralDriver`].
pub trait CandleAncestralNoiseDraw {
    fn draw_normal(
        &self,
        shape: &[usize],
        device: &Device,
        step: usize,
        modality: u64,
    ) -> Result<Tensor>;
}

/// Default production Candle CPU-seeded ChaCha noise generator matching `denoise_tokens_rf_ancestral`
/// bit-for-bit (`noise_seed.wrapping_add(2 * step as u64).wrapping_add(modality)`).
pub struct CandleStdRngNoise {
    pub noise_seed: u64,
}

impl CandleAncestralNoiseDraw for CandleStdRngNoise {
    fn draw_normal(
        &self,
        shape: &[usize],
        device: &Device,
        step: usize,
        modality: u64,
    ) -> Result<Tensor> {
        let effective_seed = self
            .noise_seed
            .wrapping_add(2 * step as u64)
            .wrapping_add(modality);
        candle_gen::execution::stream(
            &format!("ltx_ancestral_step_{step}_modality_{modality}"),
            effective_seed,
        );
        let mut rng = StdRng::seed_from_u64(effective_seed);
        let n: usize = shape.iter().product();
        let data = candle_gen::seeded_normal_vec(&mut rng, n);
        Ok(Tensor::from_vec(data, shape, device)?.to_dtype(DType::F32)?)
    }
}

/// Typed [`FamilyDriver<CandleAvState>`] for rectified-flow ancestral token denoising
/// (`denoise_tokens_rf_ancestral` and `dfr::denoise_dfr_tile`).
pub struct LtxCandleRfAncestralDriver<'a, F, N> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_mask: CandleVideoTokenMask<'a>,
    audio_frames: Option<usize>,
    eta: f32,
    s_noise: f32,
    forward: RefCell<F>,
    noise_draw: N,
}

impl<'a, F, N> LtxCandleRfAncestralDriver<'a, F, N>
where
    F: FnMut(
        &Tensor,
        &Tensor,
        Option<&Tensor>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Tensor, Option<Tensor>)>,
    N: CandleAncestralNoiseDraw,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_mask: CandleVideoTokenMask<'a>,
        audio_frames: Option<usize>,
        eta: f32,
        s_noise: f32,
        forward: F,
        noise_draw: N,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_mask,
            audio_frames,
            eta,
            s_noise,
            forward: RefCell::new(forward),
            noise_draw,
        }
    }

    fn evaluate_clean_x0(
        &self,
        cur: &CandleAvState,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let timesteps = self.video_mask.token_timesteps(sigma)?;
        let aflat = match (&cur.audio, self.audio_frames) {
            (Some(al), Some(_)) => Some(flatten_audio_latent(al)?),
            _ => None,
        };
        let (vvel, avel) =
            (self.forward.borrow_mut())(&cur.video, &timesteps, aflat.as_ref(), sigma, kind)?;
        let vden = self.video_mask.video_denoised(&cur.video, &vvel, sigma)?;
        let aden = match (&cur.audio, avel, self.audio_frames) {
            (Some(al), Some(av), Some(af)) => Some(audio_denoised(al, &av, af, sigma)?),
            _ => None,
        };
        Ok((vden, aden))
    }
}

impl<F, N> FamilyDriver<CandleAvState> for LtxCandleRfAncestralDriver<'_, F, N>
where
    F: FnMut(
        &Tensor,
        &Tensor,
        Option<&Tensor>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Tensor, Option<Tensor>)>,
    N: CandleAncestralNoiseDraw,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, CandleAvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, CandleAvState>,
    ) -> GenResult<BoundaryState<CandleAvState>> {
        self.support.validate_slice(&input, state)?;
        check_schedule_match(self.support.driver_id, &self.schedule, &input)?;
        state.enter_slice(&input)?;

        let SliceInput {
            pass,
            phase,
            span,
            total_steps,
            state: boundary_state,
            request_boundary_estimate,
            cancel,
            ..
        } = input;
        let mut cur = extract_state(boundary_state);
        let has_audio = cur.audio.is_some() && self.audio_frames.is_some();
        let device = cur.video.device().clone();

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (sigma, sigma_next) = self
                .schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing schedule window at step {step}")))?;
            let info = make_step_info(
                pass,
                phase,
                total_steps,
                step,
                sigma,
                sigma_next,
                DriverEvaluationKind::Trajectory,
            );
            let stream_sigmas = make_stream_sigmas(has_audio, sigma, sigma_next);
            hooks.observe_streams(&info, &stream_sigmas, &cur)?;
            state.record_stream_step(&info, stream_sigmas);

            let (vden, aden) = self
                .evaluate_clean_x0(&cur, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            if sigma_next <= 0.0 {
                cur.video = vden;
                if let Some(ad) = aden {
                    cur.audio = Some(ad);
                }
            } else {
                let coeffs = gen_core_ltx::ltx_dfr::RfAncestralCoeffs::new(
                    sigma,
                    sigma_next,
                    self.eta,
                    self.s_noise,
                )?;
                let ratio = coeffs.sigma_down_ratio as f64;
                let mut step_rf = |x: &Tensor, x0: &Tensor, modality: u64| -> Result<Tensor> {
                    let mut next = ((x * ratio)? + (x0 * (1.0 - ratio))?)?;
                    if self.eta > 0.0 {
                        next = (next * coeffs.alpha_ratio as f64)?;
                        if coeffs.renoise_coeff > 0.0 {
                            let _draw = state.next_draw_index();
                            let noise = self.noise_draw.draw_normal(
                                next.dims(),
                                &device,
                                step,
                                modality,
                            )?;
                            next = (next + (noise * coeffs.renoise_coeff as f64)?)?;
                        }
                    }
                    Ok(next)
                };
                let mut vnext = step_rf(&cur.video, &vden, 0).map_err(core_err)?;
                if self.eta > 0.0 {
                    vnext = self.video_mask.pin_denoised(&vnext).map_err(ccore_err)?;
                }
                cur.video = vnext;
                if let (Some(al), Some(ad)) = (&cur.audio, &aden) {
                    cur.audio = Some(step_rf(al, ad, 1).map_err(core_err)?);
                }
            }

            cur = hooks.post_step(&info, cur)?;
            hooks.materialize(&info, &cur)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(span.end);

        if request_boundary_estimate {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let boundary_sigma = self.schedule.sigmas()[span.end];
            if boundary_sigma > 0.0 {
                let (vden, aden) = self
                    .evaluate_clean_x0(&cur, boundary_sigma, DriverEvaluationKind::BoundaryEstimate)
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: CandleAvState {
                        video: vden,
                        audio: aden,
                    },
                    source: EstimateSource::EndpointEvaluation,
                });
            }
            state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
            return Ok(BoundaryState::CleanEstimate {
                latent: cur,
                source: EstimateSource::PreviousEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(cur))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::driver::conformance::{
        verify_driver_conformance, verify_multistream_driver_conformance, ConformanceCase,
        MultiStreamConformanceCase,
    };
    use candle_gen::driver::SliceSpec;
    use candle_gen::gen_core::multistream::{
        LearnedUpsamplerComponent, ScheduleSource, StreamTransitionKind,
    };
    use candle_gen::gen_core::{CancelFlag, Progress};
    use gen_core_ltx::{
        compute_audio_frames, LtxConditioningProfile, LtxMemoryFootprintSpec, LtxModelFamily,
        LtxOrchestrationRoute, LtxRenderPlan, LtxRenderPlanInput, LtxVariant, LTX_STAGE1_SIGMAS,
        LTX_STAGE2_SIGMAS, LTX_TEMPORAL_SIGMAS,
    };

    fn resolved(id: &str, sigmas: &[f32]) -> ResolvedSchedule {
        ScheduleSource::ExplicitTable {
            table_id: id.into(),
            sigmas: sigmas.to_vec(),
        }
        .resolve()
        .unwrap()
    }

    fn eq_f32_tensors(a: &Tensor, b: &Tensor) -> bool {
        if a.dims() != b.dims() {
            return false;
        }
        let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        va.len() == vb.len()
            && va
                .iter()
                .zip(vb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn eq_av(a: &CandleAvState, b: &CandleAvState) -> bool {
        eq_f32_tensors(&a.video, &b.video)
            && match (&a.audio, &b.audio) {
                (Some(x), Some(y)) => eq_f32_tensors(x, y),
                (None, None) => true,
                _ => false,
            }
    }

    #[test]
    fn candle_ltx_drivers_pass_tier0_and_multistream_conformance() {
        let dev = Device::Cpu;
        let clean: &'static Tensor = Box::leak(Box::new(
            Tensor::from_vec(vec![9.0_f32, 8.0], (1, 2, 1), &dev).unwrap(),
        ));
        let mask: &'static Tensor = Box::leak(Box::new(
            Tensor::from_vec(vec![0.0_f32, 1.0], (1, 2, 1), &dev).unwrap(),
        ));
        let s1 = resolved("ltx_stage1", &LTX_STAGE1_SIGMAS);
        let transition = StreamTransition::new(
            "stage1_to_stage2_video",
            vec![StreamId::Video],
            StreamTransitionKind::Composed(vec![
                StreamTransitionKind::LatentResample {
                    upsampler: Some(LearnedUpsamplerComponent {
                        name: "ltx_spatial_upsampler_x2".into(),
                        normalization: "vae".into(),
                        workspace_bytes: 1024,
                    }),
                    spatial_factor: (2, 2),
                    temporal_factor: 1,
                },
                StreamTransitionKind::Renoise {
                    sigma: LTX_STAGE2_SIGMAS[0],
                },
            ]),
        )
        .unwrap();

        let av_report = verify_multistream_driver_conformance(MultiStreamConformanceCase {
            base: ConformanceCase {
                total_steps: 8,
                split_at: 4,
                seed: 19,
                base_geometry: (64, 64),
                alt_geometry: (128, 128),
                make_driver: {
                    let s1 = s1.clone();
                    move || {
                        Box::new(LtxCandleDistilledAvDriver::new(
                            "candle_ltx_av_test",
                            s1.clone(),
                            CandleVideoTokenMask {
                                clean_latent: clean,
                                denoise_mask: mask,
                            },
                            2,
                            |vtok, vts, aflat, _sigma, _kind| {
                                let vv = vtok.broadcast_mul(&vts.reshape((1, 2, 1))?)?;
                                let av = aflat.affine(0.25, 0.0)?;
                                Ok((vv, av))
                            },
                        ))
                    }
                },
                make_initial: {
                    let dev = dev.clone();
                    move || {
                        CandleAvState::av(
                            Tensor::from_vec(vec![9.0_f32, 1.5], (1, 2, 1), &dev).unwrap(),
                            Tensor::from_vec(vec![0.5_f32; 8 * 2 * 16], (1, 8, 2, 16), &dev)
                                .unwrap(),
                        )
                    }
                },
                latents_equal: eq_av,
            },
            expected_stream_schedules: vec![
                (StreamId::Video, s1.clone()),
                (StreamId::Audio, s1.clone()),
            ],
            boundary_transition: Some(transition.clone()),
        })
        .unwrap();
        assert!(av_report.verified_transition_guard);
        assert!(av_report.base.verified_inert_split);

        // Video-only Dev driver conformance
        let guider = GuiderParams {
            cfg_scale: 3.0,
            stg_scale: 1.0,
            stg_blocks: &[28],
            rescale_scale: 0.7,
            modality_scale: 3.0,
        };
        let vdev_report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 5,
            base_geometry: (64, 64),
            alt_geometry: (128, 128),
            make_driver: {
                let sched = resolved("ltx_dev_slice", &LTX_STAGE1_SIGMAS[..5]);
                move || {
                    Box::new(LtxCandleVideoDevDriver::new(
                        "candle_ltx_video_dev_test",
                        sched.clone(),
                        CandleVideoTokenMask {
                            clean_latent: clean,
                            denoise_mask: mask,
                        },
                        guider,
                        |vtok, _vts, _sigma, _kind| {
                            Ok((
                                vtok.affine(0.5, 0.0)?,
                                vtok.affine(0.25, 0.0)?,
                                vtok.affine(0.125, 0.0)?,
                            ))
                        },
                    ))
                }
            },
            make_initial: {
                let dev = dev.clone();
                move || Tensor::from_vec(vec![9.0_f32, 1.0], (1, 2, 1), &dev).unwrap()
            },
            latents_equal: eq_f32_tensors,
        })
        .unwrap();
        assert!(vdev_report.verified_inert_split);
        assert!(vdev_report.verified_cancellation);

        // RF ancestral multi-stream conformance with explicit draw tracking
        let rf_sched = resolved("ltx_temporal", &LTX_TEMPORAL_SIGMAS);
        let rf_report = verify_multistream_driver_conformance(MultiStreamConformanceCase {
            base: ConformanceCase {
                total_steps: 4,
                split_at: 2,
                seed: 77,
                base_geometry: (64, 64),
                alt_geometry: (64, 64),
                make_driver: {
                    let rf_sched = rf_sched.clone();
                    move || {
                        Box::new(LtxCandleRfAncestralDriver::new(
                            "candle_ltx_rf_ancestral_test",
                            rf_sched.clone(),
                            CandleVideoTokenMask {
                                clean_latent: clean,
                                denoise_mask: mask,
                            },
                            Some(2),
                            0.5,
                            1.0,
                            |vtok, _vts, aflat, _sigma, _kind| {
                                let vv = vtok.affine(0.5, 0.0)?;
                                let av = aflat.map(|af| af.affine(0.25, 0.0)).transpose()?;
                                Ok((vv, av))
                            },
                            CandleStdRngNoise { noise_seed: 77 },
                        ))
                    }
                },
                make_initial: {
                    let dev = dev.clone();
                    move || {
                        CandleAvState::av(
                            Tensor::from_vec(vec![9.0_f32, 1.0], (1, 2, 1), &dev).unwrap(),
                            Tensor::from_vec(vec![2.0_f32; 8 * 2 * 16], (1, 8, 2, 16), &dev)
                                .unwrap(),
                        )
                    }
                },
                latents_equal: eq_av,
            },
            expected_stream_schedules: vec![
                (StreamId::Video, rf_sched.clone()),
                (StreamId::Audio, rf_sched),
            ],
            boundary_transition: Some(transition),
        })
        .unwrap();
        assert!(rf_report.base.verified_reset_preserves_draws);
        assert!(rf_report.base.unsplit_draws > 0);
    }

    #[test]
    fn candle_av_dev_4pass_and_per_token_timing_execute_via_runner() {
        let dev = Device::Cpu;
        let clean = Tensor::from_vec(vec![7.0_f32, -3.0], (1, 2, 1), &dev).unwrap();
        let mask = Tensor::from_vec(vec![0.0_f32, 1.0], (1, 2, 1), &dev).unwrap();
        let v_mask = CandleVideoTokenMask {
            clean_latent: &clean,
            denoise_mask: &mask,
        };
        let sched = resolved("ltx_stage2", &LTX_STAGE2_SIGMAS);
        let seen_video_ts = RefCell::new(Vec::<Vec<f32>>::new());
        let seen_audio_evals = RefCell::new(0usize);
        let guider_no_rescale = GuiderParams {
            cfg_scale: 3.0,
            stg_scale: 1.0,
            stg_blocks: &[28],
            rescale_scale: 0.0,
            modality_scale: 3.0,
        };

        let driver = LtxCandleAvDevDriver::new(
            "candle_av_dev_test",
            sched.clone(),
            v_mask,
            2,
            guider_no_rescale,
            guider_no_rescale,
            |vtok, vts, aflat, _sigma, _kind| {
                seen_video_ts
                    .borrow_mut()
                    .push(vts.flatten_all()?.to_vec1::<f32>()?);
                *seen_audio_evals.borrow_mut() += 1;
                Ok(CandleDevAvBranches {
                    conditional: (vtok.affine(0.5, 0.0)?, aflat.affine(0.5, 0.0)?),
                    negative: (vtok.affine(0.25, 0.0)?, aflat.affine(0.25, 0.0)?),
                    perturbed: (vtok.affine(0.125, 0.0)?, aflat.affine(0.125, 0.0)?),
                    isolated: (vtok.affine(0.3, 0.0)?, aflat.affine(0.3, 0.0)?),
                })
            },
        );
        let cancel = CancelFlag::new();
        let mut state = DriverRunState::new();
        let mut progress = Vec::<Progress>::new();
        let mut on_p = |p| progress.push(p);
        let mut hooks = StepHooks::new(&mut on_p);
        let out = candle_gen::run_candle_driver_slices(
            &driver,
            [SliceSpec::full(3)],
            BoundaryState::XSigma(CandleAvState::av(
                Tensor::from_vec(vec![7.0_f32, 1.5], (1, 2, 1), &dev).unwrap(),
                Tensor::from_vec(vec![1.0_f32; 8 * 2 * 16], (1, 8, 2, 16), &dev).unwrap(),
            )),
            3,
            1,
            &cancel,
            None,
            &mut state,
            &mut hooks,
            |_, s, _| Ok(s),
        )
        .unwrap();
        assert_eq!(state.internal_evaluations, 12);
        assert_eq!(state.completed_outer_steps, 3);
        assert_eq!(*seen_audio_evals.borrow(), 3);
        assert_eq!(
            seen_video_ts.borrow()[0],
            vec![0.0, LTX_STAGE2_SIGMAS[0]],
            "video timesteps must equal sigma * denoise_mask"
        );
        let BoundaryState::XSigma(final_av) = out else {
            panic!("expected XSigma");
        };
        let v_out = final_av
            .video
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            v_out[0], 7.0,
            "with rescale_scale=0.0, conditioned token 0 must remain strictly pinned"
        );

        // Also check LTX-2.5 Dev rescale_scale=0.7 reproduces HEAD's exact per-branch-masked
        // `combine_guidance` numerical step bit-for-bit.
        let guider_rescaled = GuiderParams {
            rescale_scale: 0.7,
            ..guider_no_rescale
        };
        let rescaled_driver = LtxCandleAvDevDriver::new(
            "candle_av_dev_rescaled_test",
            sched,
            v_mask,
            2,
            guider_rescaled,
            guider_rescaled,
            |vtok, _vts, aflat, _sigma, _kind| {
                Ok(CandleDevAvBranches {
                    conditional: (vtok.affine(0.5, 0.0)?, aflat.affine(0.5, 0.0)?),
                    negative: (vtok.affine(0.25, 0.0)?, aflat.affine(0.25, 0.0)?),
                    perturbed: (vtok.affine(0.125, 0.0)?, aflat.affine(0.125, 0.0)?),
                    isolated: (vtok.affine(0.3, 0.0)?, aflat.affine(0.3, 0.0)?),
                })
            },
        );
        let mut r_state = DriverRunState::new();
        let mut r_on_p = |_| {};
        let mut r_hooks = StepHooks::new(&mut r_on_p);
        let BoundaryState::XSigma(r_out) = candle_gen::run_candle_driver_slices(
            &rescaled_driver,
            [SliceSpec::full(3)],
            BoundaryState::XSigma(CandleAvState::av(
                Tensor::from_vec(vec![7.0_f32, 1.5], (1, 2, 1), &dev).unwrap(),
                Tensor::from_vec(vec![1.0_f32; 8 * 2 * 16], (1, 8, 2, 16), &dev).unwrap(),
            )),
            3,
            1,
            &cancel,
            None,
            &mut r_state,
            &mut r_hooks,
            |_, s, _| Ok(s),
        )
        .unwrap() else {
            panic!("expected XSigma");
        };
        assert_eq!(r_state.internal_evaluations, 12);

        // Compare bit-for-bit against the direct per-branch reference computation.
        let mut v_ref = Tensor::from_vec(vec![7.0_f32, 1.5], (1, 2, 1), &dev).unwrap();
        let mut a_ref = Tensor::from_vec(vec![1.0_f32; 8 * 2 * 16], (1, 8, 2, 16), &dev).unwrap();
        for win in LTX_STAGE2_SIGMAS.windows(2) {
            let (sigma, sigma_next) = (win[0], win[1]);
            let a_flat = flatten_audio_latent(&a_ref).unwrap();
            let vc = v_mask
                .video_denoised(&v_ref, &v_ref.affine(0.5, 0.0).unwrap(), sigma)
                .unwrap();
            let vn = v_mask
                .video_denoised(&v_ref, &v_ref.affine(0.25, 0.0).unwrap(), sigma)
                .unwrap();
            let vp = v_mask
                .video_denoised(&v_ref, &v_ref.affine(0.125, 0.0).unwrap(), sigma)
                .unwrap();
            let vi = v_mask
                .video_denoised(&v_ref, &v_ref.affine(0.3, 0.0).unwrap(), sigma)
                .unwrap();
            let v_den = dev_sampler::combine_guidance(&vc, &vn, &vp, &vi, guider_rescaled).unwrap();
            v_ref = euler_from_denoised(&v_ref, &v_den, sigma, sigma_next).unwrap();
            let ac = audio_denoised(&a_ref, &a_flat.affine(0.5, 0.0).unwrap(), 2, sigma).unwrap();
            let an = audio_denoised(&a_ref, &a_flat.affine(0.25, 0.0).unwrap(), 2, sigma).unwrap();
            let ap = audio_denoised(&a_ref, &a_flat.affine(0.125, 0.0).unwrap(), 2, sigma).unwrap();
            let ai = audio_denoised(&a_ref, &a_flat.affine(0.3, 0.0).unwrap(), 2, sigma).unwrap();
            let a_den = dev_sampler::combine_guidance(&ac, &an, &ap, &ai, guider_rescaled).unwrap();
            a_ref = euler_from_denoised(&a_ref, &a_den, sigma, sigma_next).unwrap();
        }
        assert_eq!(
            r_out.video.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            v_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            r_out
                .audio
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            a_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn candle_ltx_plan_and_double_noise_transition_guard_enforced_weights_free() {
        let dev = Device::Cpu;
        let plan = LtxRenderPlan::build(LtxRenderPlanInput {
            family: LtxModelFamily::Ltx25,
            variant: LtxVariant::Dev,
            route: LtxOrchestrationRoute::AvDfr,
            conditioning: LtxConditioningProfile::DfrGeneratedKeyframes {
                slot_count: 1,
                temporal_upsample_rounds: 1,
            },
            sampler: None,
            width: 512,
            height: 512,
            frames: 17,
            fps: 24.0,
            seed: 42,
            explicit_steps: None,
            has_temporal_upsampler: true,
            footprint: LtxMemoryFootprintSpec::default(),
        })
        .unwrap();
        assert_eq!(compute_audio_frames(17, 24.0), 18);
        assert_eq!(plan.stages.len(), 3);

        let noisy = BoundaryState::XSigma(Tensor::from_vec(vec![1.0_f32], 1, &dev).unwrap());
        let err = apply_candle_stream_transition(StreamId::Video, &plan.transitions[0], noisy, Ok)
            .unwrap_err();
        assert!(matches!(err, CandleError::Unsupported(_)));

        let req_ok = candle_gen::gen_core::GenerationRequest {
            prompt: "cinematic shot".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            num_generated_keyframes: Some(1),
            temporal_upsample_rounds: Some(1),
            ..Default::default()
        };
        let built = render_plan_for_request(
            LtxModelFamily::Ltx25,
            crate::dev_sampler::TransformerVariant::Dev,
            &req_ok,
            17,
            24.0,
            true,
        )
        .unwrap();
        assert_eq!(built.route, LtxOrchestrationRoute::AvDfr);

        let req_bad_curated_dev = candle_gen::gen_core::GenerationRequest {
            prompt: "bad".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            sampler: Some("heun".into()),
            ..Default::default()
        };
        assert!(matches!(
            render_plan_for_request(
                LtxModelFamily::Ltx25,
                crate::dev_sampler::TransformerVariant::Dev,
                &req_bad_curated_dev,
                17,
                24.0,
                true,
            ),
            Err(CandleError::Unsupported(_))
        ));
    }
}
