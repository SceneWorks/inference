//! Typed ENG-6 `FamilyDriver` implementations and stream-targeted boundary transitions for
//! `mlx-gen-ltx` (`PIPE-4`, `PIPE-6`, `PIPE-7`, `DEC-3`, `DEC-19`).
//!
//! All native MLX LTX denoising loops (`denoise`, `denoise_av`, `denoise_av_dev`,
//! `denoise_av_tokens`, `denoise_av_tokens_dev`, `denoise_tokens_rf_ancestral`, and the
//! LTX-2.5 trainer validation denoisers) execute through these generic `FamilyDriver` structs and
//! [`mlx_gen::run_mlx_driver_slices`], preserving exact dtype-preserving AP13 class-(a) math while
//! allowing weights-free Tier 0 conformance testing over mocked forward seams.

use std::cell::RefCell;

use mlx_gen::array::scalar;
use mlx_gen::driver::{
    DriverEvaluationKind, DriverRefusalReasons, DriverRunState, DriverStepInfo, DriverSupport,
    FamilyDriver, SliceInput, StepHooks, StreamId, StreamStepSigma,
};
use mlx_gen::gen_core::multistream::{ResolvedSchedule, StreamTransition, TransitionBoundaryGuard};
use mlx_gen::gen_core::{BoundaryState, Error as GenError, EstimateSource, Result as GenResult};
use mlx_gen::{Error, Result};
use mlx_rs::ops::{add, broadcast_to, multiply};
use mlx_rs::{Array, Dtype};

use crate::conditioning::{apply_denoise_mask, token_timesteps, I2vConditioning};
use crate::pipeline::euler_step;
use crate::transformer::to_denoised;

/// Multi-stream latent state carried through [`FamilyDriver<MlxAvState>`] slices (`PIPE-4`, `PIPE-6`).
#[derive(Clone, Debug)]
pub struct MlxAvState {
    pub video: Array,
    pub audio: Option<Array>,
}

impl MlxAvState {
    pub fn av(video: Array, audio: Array) -> Self {
        Self {
            video,
            audio: Some(audio),
        }
    }

    pub fn video_only(video: Array) -> Self {
        Self { video, audio: None }
    }
}

/// How the video stream is packed and conditioned inside LTX drivers.
#[derive(Clone, Copy)]
pub enum MlxVideoLayout<'a> {
    /// Grid `(B, 128, F, H, W)` flattened each step to `(B, F*H*W, 128)`, optional I2V mask.
    Grid {
        shape: [i32; 5],
        i2v: Option<&'a I2vConditioning>,
    },
    /// Already token-native `(B, S, 128)` with fixed clean latent and per-token denoise mask.
    Tokens {
        clean_latent: &'a Array,
        denoise_mask: &'a Array,
    },
}

/// Audio grid shape `(B, 8, T, 16)` when active alongside the video stream.
#[derive(Clone, Copy, Debug)]
pub struct MlxAudioLayout {
    pub shape: [i32; 4],
}

impl MlxAudioLayout {
    pub fn from_array(audio: &Array) -> Self {
        let s = audio.shape();
        Self {
            shape: [s[0], s[1], s[2], s[3]],
        }
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

fn core_err(err: Error) -> GenError {
    err.into()
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

fn flatten_video(layout: &MlxVideoLayout<'_>, video: &Array) -> Result<Array> {
    match layout {
        MlxVideoLayout::Grid { shape, .. } => Ok(video
            .reshape(&[shape[0], shape[1], -1])?
            .transpose_axes(&[0, 2, 1])?),
        MlxVideoLayout::Tokens { .. } => Ok(video.clone()),
    }
}

fn unflatten_video(layout: &MlxVideoLayout<'_>, velocity: Array) -> Result<Array> {
    match layout {
        MlxVideoLayout::Grid { shape, .. } => Ok(velocity
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[shape[0], shape[1], shape[2], shape[3], shape[4]])?),
        MlxVideoLayout::Tokens { .. } => Ok(velocity),
    }
}

fn video_timesteps(layout: &MlxVideoLayout<'_>, dt: Dtype, sigma: f32) -> Result<Array> {
    match layout {
        MlxVideoLayout::Grid { shape, i2v } => match i2v {
            Some(st) => st.token_timesteps(sigma, shape[3], shape[4]),
            None => {
                let num_tokens = shape[2] * shape[3] * shape[4];
                Ok(broadcast_to(
                    &scalar(sigma).as_dtype(dt)?,
                    &[shape[0], num_tokens],
                )?)
            }
        },
        MlxVideoLayout::Tokens { denoise_mask, .. } => token_timesteps(denoise_mask, dt, sigma),
    }
}

fn pin_video_denoised(layout: &MlxVideoLayout<'_>, denoised: Array) -> Result<Array> {
    match layout {
        MlxVideoLayout::Grid { i2v: Some(st), .. } => {
            apply_denoise_mask(&denoised, &st.clean_latent, &st.denoise_mask)
        }
        MlxVideoLayout::Grid { i2v: None, .. } => Ok(denoised),
        MlxVideoLayout::Tokens {
            clean_latent,
            denoise_mask,
        } => apply_denoise_mask(&denoised, clean_latent, denoise_mask),
    }
}

fn flatten_audio(
    layout: MlxAudioLayout,
    audio: &Array,
    dt: Dtype,
    sigma: f32,
) -> Result<(Array, Array)> {
    let [ab, ac, at, af] = layout.shape;
    let aflat = audio
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[ab, at, ac * af])?;
    let ats = broadcast_to(&scalar(sigma).as_dtype(dt)?, &[ab, at])?;
    Ok((aflat, ats))
}

fn unflatten_audio(layout: MlxAudioLayout, avel: Array) -> Result<Array> {
    let [ab, ac, at, af] = layout.shape;
    Ok(avel
        .reshape(&[ab, at, ac, af])?
        .transpose_axes(&[0, 2, 1, 3])?)
}

fn eval_av_state(state: &MlxAvState) -> Result<()> {
    match &state.audio {
        Some(audio) => Ok(mlx_rs::transforms::eval([&state.video, audio])?),
        None => Ok(mlx_rs::transforms::eval([&state.video])?),
    }
}

/// Apply an explicit [`StreamTransition`] guarded by [`TransitionBoundaryGuard`] between stages
/// (`PIPE-4`).
pub(crate) fn apply_mlx_stream_transition(
    stream: StreamId,
    transition: &StreamTransition,
    clean_state: BoundaryState<Array>,
    op: impl FnOnce(Array) -> Result<Array>,
) -> Result<BoundaryState<Array>> {
    use mlx_gen::gen_core::multistream::StreamBoundarySemantics;
    let (_, expected_exit) = transition
        .kind
        .boundary_semantics(&transition.transition_id)?;
    TransitionBoundaryGuard::new(stream, clean_state)
        .apply(transition, |state| match state {
            BoundaryState::CleanEstimate { latent, .. } => {
                let next = op(latent).map_err(core_err)?;
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
                let next = op(state).map_err(core_err)?;
                Ok(BoundaryState::RouteSpecific { state: next, label })
            }
            BoundaryState::XSigma(x) => {
                let next = op(x).map_err(core_err)?;
                Ok(BoundaryState::XSigma(next))
            }
        })
        .map_err(Into::into)
}

/// Build a typed pre-load [`gen_core_ltx::LtxRenderPlan`] from an MLX request and loaded route
/// identity without reading or executing model weights (`DEC-3`, `DEC-13`, `MEM-3`).
pub fn render_plan_for_request(
    family: gen_core_ltx::LtxModelFamily,
    variant: crate::dev_sampler::TransformerVariant,
    req: &mlx_gen::GenerationRequest,
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
        gen_core_ltx::MemoryBudgetDomainKey::UnifiedMemory,
        req,
        resolved_frames,
        fps,
        has_temporal_upsampler,
    )
    .map_err(Into::into)
}

// =================================================================================================
// Single-Stream Video Grid Distilled Driver (`pipeline::denoise`)
// =================================================================================================

/// Typed [`FamilyDriver<Array>`] for MLX video-only grid distilled Euler (`pipeline::denoise`).
pub struct LtxMlxVideoGridDriver<'a, F> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    layout: MlxVideoLayout<'a>,
    forward: RefCell<F>,
}

impl<'a, F> LtxMlxVideoGridDriver<'a, F>
where
    F: FnMut(&Array, &Array, f32, DriverEvaluationKind) -> Result<Array>,
{
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        shape: [i32; 5],
        i2v: Option<&'a I2vConditioning>,
        forward: F,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            layout: MlxVideoLayout::Grid { shape, i2v },
            forward: RefCell::new(forward),
        }
    }

    fn evaluate_clean_estimate(
        &self,
        latent: &Array,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<Array> {
        let dt = latent.dtype();
        let flat = flatten_video(&self.layout, latent)?;
        let ts = video_timesteps(&self.layout, dt, sigma)?;
        let vel_tokens = (self.forward.borrow_mut())(&flat, &ts, sigma, kind)?;
        let vel = unflatten_video(&self.layout, vel_tokens)?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let denoised = to_denoised(latent, &vel, &sig)?;
        pin_video_denoised(&self.layout, denoised)
    }
}

impl<F> FamilyDriver<Array> for LtxMlxVideoGridDriver<'_, F>
where
    F: FnMut(&Array, &Array, f32, DriverEvaluationKind) -> Result<Array>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> GenResult<BoundaryState<Array>> {
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
        let mut lat = extract_state(boundary_state);

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
            hooks.observe_streams(&info, &stream_sigmas, &lat)?;
            state.record_stream_step(&info, stream_sigmas);

            let denoised = self
                .evaluate_clean_estimate(&lat, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            lat = euler_step(&lat, &denoised, sigma, sigma_next).map_err(core_err)?;

            lat = hooks.post_step(&info, lat)?;
            mlx_rs::transforms::eval([&lat]).map_err(|e| GenError::Msg(e.to_string()))?;
            hooks.materialize(&info, &lat)?;
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
                let est = self
                    .evaluate_clean_estimate(
                        &lat,
                        boundary_sigma,
                        DriverEvaluationKind::BoundaryEstimate,
                    )
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: est,
                    source: EstimateSource::EndpointEvaluation,
                });
            }
            state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
            return Ok(BoundaryState::CleanEstimate {
                latent: lat,
                source: EstimateSource::PreviousEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(lat))
    }
}

// =================================================================================================
// Multi-Stream Distilled AV Driver (Grid & Tokens)
// =================================================================================================

/// Typed [`FamilyDriver<MlxAvState>`] for joint AV distilled Euler across grid and token-native
/// representations (`denoise_av`, `denoise_av_tokens`).
pub struct LtxMlxAvDistilledDriver<'a, F> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_layout: MlxVideoLayout<'a>,
    audio_layout: Option<MlxAudioLayout>,
    forward: RefCell<F>,
}

impl<'a, F> LtxMlxAvDistilledDriver<'a, F>
where
    F: FnMut(
        &Array,
        &Array,
        Option<(&Array, &Array)>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>)>,
{
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_layout: MlxVideoLayout<'a>,
        audio_layout: Option<MlxAudioLayout>,
        forward: F,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_layout,
            audio_layout,
            forward: RefCell::new(forward),
        }
    }

    fn evaluate_clean_estimate(
        &self,
        cur: &MlxAvState,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>)> {
        let dt = cur.video.dtype();
        let vflat = flatten_video(&self.video_layout, &cur.video)?;
        let vts = video_timesteps(&self.video_layout, dt, sigma)?;
        let audio_inputs = match (&cur.audio, self.audio_layout) {
            (Some(alat), Some(al)) => Some(flatten_audio(al, alat, dt, sigma)?),
            _ => None,
        };
        let audio_ref = audio_inputs.as_ref().map(|(aflat, ats)| (aflat, ats));
        let (vvel_tokens, avel_tokens) =
            (self.forward.borrow_mut())(&vflat, &vts, audio_ref, sigma, kind)?;
        let vvel = unflatten_video(&self.video_layout, vvel_tokens)?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let vden = pin_video_denoised(&self.video_layout, to_denoised(&cur.video, &vvel, &sig)?)?;
        let aden = match (&cur.audio, self.audio_layout, avel_tokens) {
            (Some(alat), Some(al), Some(av)) => {
                let avel = unflatten_audio(al, av)?;
                Some(to_denoised(alat, &avel, &sig)?)
            }
            _ => None,
        };
        Ok((vden, aden))
    }
}

impl<F> FamilyDriver<MlxAvState> for LtxMlxAvDistilledDriver<'_, F>
where
    F: FnMut(
        &Array,
        &Array,
        Option<(&Array, &Array)>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>)>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, MlxAvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, MlxAvState>,
    ) -> GenResult<BoundaryState<MlxAvState>> {
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
        let has_audio = cur.audio.is_some() && self.audio_layout.is_some();

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
                .evaluate_clean_estimate(&cur, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            cur.video = euler_step(&cur.video, &vden, sigma, sigma_next).map_err(core_err)?;
            if let (Some(alat), Some(ad)) = (&cur.audio, aden.as_ref()) {
                cur.audio = Some(euler_step(alat, ad, sigma, sigma_next).map_err(core_err)?);
            }

            cur = hooks.post_step(&info, cur)?;
            eval_av_state(&cur).map_err(core_err)?;
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
                    .evaluate_clean_estimate(
                        &cur,
                        boundary_sigma,
                        DriverEvaluationKind::BoundaryEstimate,
                    )
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: MlxAvState {
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

// =================================================================================================
// Multi-Stream Dev 4-Pass Guided Driver (Grid, Tokens, Validation Video / AV)
// =================================================================================================

/// Typed [`FamilyDriver<MlxAvState>`] for non-distilled 4-pass Dev guided Euler (`denoise_av_dev`,
/// `denoise_av_tokens_dev`, and trainer validation routes).
pub struct LtxMlxDevGuidedDriver<'a, F> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_layout: MlxVideoLayout<'a>,
    audio_layout: Option<MlxAudioLayout>,
    evaluator: RefCell<F>,
}

impl<'a, F> LtxMlxDevGuidedDriver<'a, F>
where
    F: FnMut(
        &MlxAvState,
        &Array,
        &Array,
        Option<(&Array, &Array)>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>, u64)>,
{
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_layout: MlxVideoLayout<'a>,
        audio_layout: Option<MlxAudioLayout>,
        evaluator: F,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_layout,
            audio_layout,
            evaluator: RefCell::new(evaluator),
        }
    }

    fn eval_step(
        &self,
        cur: &MlxAvState,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>, u64)> {
        let dt = cur.video.dtype();
        let vflat = flatten_video(&self.video_layout, &cur.video)?;
        let vts = video_timesteps(&self.video_layout, dt, sigma)?;
        let audio_inputs = match (&cur.audio, self.audio_layout) {
            (Some(alat), Some(al)) => Some(flatten_audio(al, alat, dt, sigma)?),
            _ => None,
        };
        let audio_ref = audio_inputs.as_ref().map(|(aflat, ats)| (aflat, ats));
        let (vden_raw, aden, evals) =
            (self.evaluator.borrow_mut())(cur, &vflat, &vts, audio_ref, sigma, kind)?;
        let vden = pin_video_denoised(&self.video_layout, vden_raw)?;
        Ok((vden, aden, evals))
    }
}

impl<F> FamilyDriver<MlxAvState> for LtxMlxDevGuidedDriver<'_, F>
where
    F: FnMut(
        &MlxAvState,
        &Array,
        &Array,
        Option<(&Array, &Array)>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>, u64)>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, MlxAvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, MlxAvState>,
    ) -> GenResult<BoundaryState<MlxAvState>> {
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
        let has_audio = cur.audio.is_some() && self.audio_layout.is_some();

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

            let (vden, aden, evals) = self
                .eval_step(&cur, sigma, DriverEvaluationKind::Trajectory)
                .map_err(core_err)?;
            for _ in 0..evals.max(1) {
                state.record_evaluation(DriverEvaluationKind::Trajectory);
            }

            cur.video = euler_step(&cur.video, &vden, sigma, sigma_next).map_err(core_err)?;
            if let (Some(alat), Some(ad)) = (&cur.audio, aden.as_ref()) {
                cur.audio = Some(euler_step(alat, ad, sigma, sigma_next).map_err(core_err)?);
            }

            cur = hooks.post_step(&info, cur)?;
            eval_av_state(&cur).map_err(core_err)?;
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
                let (vden, aden, _) = self
                    .eval_step(&cur, boundary_sigma, DriverEvaluationKind::BoundaryEstimate)
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: MlxAvState {
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

// =================================================================================================
// Multi-Stream RF Ancestral Driver (`denoise_tokens_rf_ancestral` & DFR tiles)
// =================================================================================================

/// Seeded noise generator seam for `LtxMlxRfAncestralDriver` so Tier 0 tests can inject
/// deterministic CPU noise or real `mlx_rs::random::key` + `normal` (`noise_seed + 2*step + modality`).
pub trait MlxAncestralNoiseDraw {
    fn draw_normal(&self, shape: &[i32], dt: Dtype, step: usize, modality: u64) -> Result<Array>;
}

/// Default production MLX explicit-key noise generator matching `denoise_tokens_rf_ancestral`
/// bit-for-bit (`noise_seed.wrapping_add(2 * step as u64).wrapping_add(modality)`).
pub struct MlxSeededKeyNoise {
    pub noise_seed: u64,
}

impl MlxAncestralNoiseDraw for MlxSeededKeyNoise {
    fn draw_normal(&self, shape: &[i32], dt: Dtype, step: usize, modality: u64) -> Result<Array> {
        let effective_seed = self
            .noise_seed
            .wrapping_add(2 * step as u64)
            .wrapping_add(modality);
        mlx_gen::video_execution::draw(
            0,
            &format!("ltx_dfr_ancestral_modality_{modality}"),
            effective_seed,
        );
        let key = mlx_rs::random::key(effective_seed)?;
        Ok(mlx_rs::random::normal::<f32>(shape, None, None, Some(&key))?.as_dtype(dt)?)
    }
}

/// Typed [`FamilyDriver<MlxAvState>`] executing token-native rectified-flow ancestral stepping
/// (`gen_core_ltx::ltx_dfr::RfAncestralCoeffs`).
pub struct LtxMlxRfAncestralDriver<'a, F, N = MlxSeededKeyNoise> {
    support: DriverSupport,
    schedule: ResolvedSchedule,
    video_clean_latent: &'a Array,
    video_denoise_mask: &'a Array,
    audio_layout: Option<MlxAudioLayout>,
    eta: f32,
    s_noise: f32,
    forward: RefCell<F>,
    noise_draw: N,
}

impl<'a, F, N> LtxMlxRfAncestralDriver<'a, F, N>
where
    F: FnMut(
        &Array,
        &Array,
        Option<(&Array, &Array)>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>)>,
    N: MlxAncestralNoiseDraw,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        driver_id: &'static str,
        schedule: ResolvedSchedule,
        video_clean_latent: &'a Array,
        video_denoise_mask: &'a Array,
        audio_layout: Option<MlxAudioLayout>,
        eta: f32,
        s_noise: f32,
        forward: F,
        noise_draw: N,
    ) -> Self {
        Self {
            support: ltx_driver_support(driver_id),
            schedule,
            video_clean_latent,
            video_denoise_mask,
            audio_layout,
            eta,
            s_noise,
            forward: RefCell::new(forward),
            noise_draw,
        }
    }

    fn video_layout(&self) -> MlxVideoLayout<'a> {
        MlxVideoLayout::Tokens {
            clean_latent: self.video_clean_latent,
            denoise_mask: self.video_denoise_mask,
        }
    }

    fn evaluate_clean_x0(
        &self,
        cur: &MlxAvState,
        sigma: f32,
        kind: DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>)> {
        let dt = cur.video.dtype();
        let vts = token_timesteps(self.video_denoise_mask, dt, sigma)?;
        let audio_inputs = match (&cur.audio, self.audio_layout) {
            (Some(alat), Some(al)) => Some(flatten_audio(al, alat, dt, sigma)?),
            _ => None,
        };
        let audio_ref = audio_inputs.as_ref().map(|(aflat, ats)| (aflat, ats));
        let (vvel, avel_tokens) =
            (self.forward.borrow_mut())(&cur.video, &vts, audio_ref, sigma, kind)?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let vden = apply_denoise_mask(
            &to_denoised(&cur.video, &vvel, &sig)?,
            self.video_clean_latent,
            self.video_denoise_mask,
        )?;
        let aden = match (&cur.audio, self.audio_layout, avel_tokens) {
            (Some(alat), Some(al), Some(av)) => {
                let avel = unflatten_audio(al, av)?;
                Some(to_denoised(alat, &avel, &sig)?)
            }
            _ => None,
        };
        Ok((vden, aden))
    }
}

impl<F, N> FamilyDriver<MlxAvState> for LtxMlxRfAncestralDriver<'_, F, N>
where
    F: FnMut(
        &Array,
        &Array,
        Option<(&Array, &Array)>,
        f32,
        DriverEvaluationKind,
    ) -> Result<(Array, Option<Array>)>,
    N: MlxAncestralNoiseDraw,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, MlxAvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, MlxAvState>,
    ) -> GenResult<BoundaryState<MlxAvState>> {
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
        let has_audio = cur.audio.is_some() && self.audio_layout.is_some();
        let dt = cur.video.dtype();

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
                (|| -> Result<()> {
                    let ratio = scalar(coeffs.sigma_down_ratio).as_dtype(dt)?;
                    let one_minus = scalar(1.0 - coeffs.sigma_down_ratio).as_dtype(dt)?;
                    let mut step_rf =
                        |x: &Array, x0: &Array, step_idx: usize, modality: u64| -> Result<Array> {
                            let mut next = add(&multiply(x, &ratio)?, &multiply(x0, &one_minus)?)?;
                            if self.eta > 0.0 {
                                let ar = scalar(coeffs.alpha_ratio).as_dtype(dt)?;
                                next = multiply(&next, &ar)?;
                                if coeffs.renoise_coeff > 0.0 {
                                    let _draw = state.next_draw_index();
                                    let noise = self.noise_draw.draw_normal(
                                        next.shape(),
                                        dt,
                                        step_idx,
                                        modality,
                                    )?;
                                    let rc = scalar(coeffs.renoise_coeff).as_dtype(dt)?;
                                    next = add(&next, &multiply(&noise, &rc)?)?;
                                }
                            }
                            Ok(next)
                        };
                    let mut vnext = step_rf(&cur.video, &vden, step, 0)?;
                    if self.eta > 0.0 {
                        vnext = apply_denoise_mask(
                            &vnext,
                            self.video_clean_latent,
                            self.video_denoise_mask,
                        )?;
                    }
                    cur.video = vnext;
                    if let (Some(al), Some(ad)) = (&cur.audio, &aden) {
                        cur.audio = Some(step_rf(al, ad, step, 1)?);
                    }
                    Ok(())
                })()
                .map_err(core_err)?;
            }

            cur = hooks.post_step(&info, cur)?;
            eval_av_state(&cur).map_err(core_err)?;
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
                let _ = self.video_layout();
                return Ok(BoundaryState::CleanEstimate {
                    latent: MlxAvState {
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
    use gen_core_ltx::{
        compute_audio_frames, LtxConditioningProfile, LtxMemoryFootprintSpec, LtxModelFamily,
        LtxOrchestrationRoute, LtxRenderPlan, LtxRenderPlanInput, LtxVariant, LTX_STAGE1_SIGMAS,
        LTX_STAGE2_SIGMAS, LTX_TEMPORAL_SIGMAS,
    };
    use mlx_gen::driver::conformance::{
        verify_driver_conformance, verify_multistream_driver_conformance, ConformanceCase,
        MultiStreamConformanceCase,
    };
    use mlx_gen::gen_core::multistream::{
        LearnedUpsamplerComponent, ScheduleSource, StreamTransitionKind,
    };
    use mlx_gen::gen_core::CancelFlag;

    fn resolved(id: &str, sigmas: &[f32]) -> ResolvedSchedule {
        ScheduleSource::ExplicitTable {
            table_id: id.into(),
            sigmas: sigmas.to_vec(),
        }
        .resolve()
        .unwrap()
    }

    fn eq_f32_arrays(a: &Array, b: &Array) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        mlx_rs::transforms::eval([a, b]).unwrap();
        let va = a.as_slice::<f32>();
        let vb = b.as_slice::<f32>();
        va.len() == vb.len()
            && va
                .iter()
                .zip(vb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn eq_av(a: &MlxAvState, b: &MlxAvState) -> bool {
        eq_f32_arrays(&a.video, &b.video)
            && match (&a.audio, &b.audio) {
                (Some(x), Some(y)) => eq_f32_arrays(x, y),
                (None, None) => true,
                _ => false,
            }
    }

    #[test]
    fn mlx_video_grid_driver_passes_full_tier0_conformance() {
        let sched = resolved("ltx_s1_test", &LTX_STAGE1_SIGMAS[..5]); // 4 steps
        let shape = [1, 4, 1, 2, 2];
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 42,
            base_geometry: (64, 64),
            alt_geometry: (128, 128),
            make_driver: {
                let sched = sched.clone();
                move || {
                    Box::new(LtxMlxVideoGridDriver::new(
                        "mlx_ltx_video_grid_test",
                        sched.clone(),
                        shape,
                        None,
                        |flat, ts, _sigma, _kind| {
                            // Mocked forward using `ts` directly (zero NN weights).
                            let tscale = ts.reshape(&[1, 4, 1])?;
                            Ok(multiply(flat, &tscale)?)
                        },
                    ))
                }
            },
            make_initial: || Array::from_slice(&[1.0_f32; 16], &[1, 4, 1, 2, 2]),
            latents_equal: eq_f32_arrays,
        })
        .unwrap();
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
        assert!(report.verified_boundary_estimate_readonly_or_refusal);
        assert!(report.verified_renoise_and_resample_guards);
    }

    #[test]
    fn mlx_av_distilled_and_rf_ancestral_drivers_pass_multistream_conformance() {
        let sched = resolved("ltx_stage1", &LTX_STAGE1_SIGMAS);
        let audio_layout = MlxAudioLayout {
            shape: [1, 2, 2, 2],
        };
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
                    let sched = sched.clone();
                    move || {
                        Box::new(LtxMlxAvDistilledDriver::new(
                            "mlx_ltx_av_grid_test",
                            sched.clone(),
                            MlxVideoLayout::Grid {
                                shape: [1, 2, 1, 2, 2],
                                i2v: None,
                            },
                            Some(audio_layout),
                            |vflat, vts, audio, _sigma, _kind| {
                                let vv = multiply(vflat, &vts.reshape(&[1, 4, 1])?)?;
                                let av = match audio {
                                    Some((af, _ats)) => Some(multiply(af, scalar(0.25))?),
                                    None => None,
                                };
                                Ok((vv, av))
                            },
                        ))
                    }
                },
                make_initial: || {
                    MlxAvState::av(
                        Array::from_slice(&[0.5_f32; 8], &[1, 2, 1, 2, 2]),
                        Array::from_slice(&[0.25_f32; 8], &[1, 2, 2, 2]),
                    )
                },
                latents_equal: eq_av,
            },
            expected_stream_schedules: vec![
                (StreamId::Video, sched.clone()),
                (StreamId::Audio, sched),
            ],
            boundary_transition: Some(transition.clone()),
        })
        .unwrap();
        assert!(av_report.verified_transition_guard);
        assert!(av_report.base.verified_inert_split);

        // Token-native RF ancestral driver with per-token video mask + uniform audio + explicit noise seed.
        let clean: &'static Array =
            Box::leak(Box::new(Array::from_slice(&[9.0_f32, 8.0], &[1, 2, 1])));
        let mask: &'static Array =
            Box::leak(Box::new(Array::from_slice(&[0.0_f32, 1.0], &[1, 2, 1])));
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
                        Box::new(LtxMlxRfAncestralDriver::new(
                            "mlx_ltx_rf_ancestral_test",
                            rf_sched.clone(),
                            clean,
                            mask,
                            Some(MlxAudioLayout {
                                shape: [1, 1, 2, 1],
                            }),
                            0.5,
                            1.0,
                            |vtok, _vts, audio, _sigma, _kind| {
                                let vv = multiply(vtok, scalar(0.5))?;
                                let av = match audio {
                                    Some((af, _)) => Some(multiply(af, scalar(0.25))?),
                                    None => None,
                                };
                                Ok((vv, av))
                            },
                            MlxSeededKeyNoise { noise_seed: 77 },
                        ))
                    }
                },
                make_initial: || {
                    MlxAvState::av(
                        Array::from_slice(&[9.0_f32, 1.0], &[1, 2, 1]),
                        Array::from_slice(&[2.0_f32; 2], &[1, 1, 2, 1]),
                    )
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
    fn mlx_per_token_timing_pins_video_tokens_while_audio_stays_uniform_sigma() {
        let clean = Array::from_slice(&[7.0_f32, -3.0], &[1, 2, 1]);
        // token 0 is pinned (denoise_mask = 0 => timestep 0), token 1 denoises (denoise_mask = 1 => sigma).
        let mask = Array::from_slice(&[0.0_f32, 1.0], &[1, 2, 1]);
        let sched = resolved("ltx_stage2", &LTX_STAGE2_SIGMAS);
        let seen_video_ts = std::cell::RefCell::new(Vec::<Vec<f32>>::new());
        let seen_audio_ts = std::cell::RefCell::new(Vec::<Vec<f32>>::new());

        let driver = LtxMlxAvDistilledDriver::new(
            "mlx_per_token_test",
            sched,
            MlxVideoLayout::Tokens {
                clean_latent: &clean,
                denoise_mask: &mask,
            },
            Some(MlxAudioLayout {
                shape: [1, 1, 2, 1],
            }),
            |vflat, vts, audio, _sigma, _kind| {
                let vts_dense = add(vts, &Array::zeros::<f32>(vts.shape())?)?;
                mlx_rs::transforms::eval([&vts_dense]).unwrap();
                seen_video_ts
                    .borrow_mut()
                    .push(vts_dense.as_slice::<f32>().to_vec());
                let (aflat, ats) = audio.expect("audio active");
                // `broadcast_to` yields a 0-stride array; force dense materialization before reading the slice.
                let ats_dense = add(ats, &Array::zeros::<f32>(ats.shape())?)?;
                mlx_rs::transforms::eval([&ats_dense]).unwrap();
                seen_audio_ts
                    .borrow_mut()
                    .push(ats_dense.as_slice::<f32>().to_vec());
                Ok((
                    multiply(vflat, scalar(0.5))?,
                    Some(multiply(aflat, scalar(0.5))?),
                ))
            },
        );
        let cancel = CancelFlag::new();
        let mut state = DriverRunState::new();
        let mut progress = Vec::new();
        let mut on_p = |p| progress.push(p);
        let mut hooks = StepHooks::new(&mut on_p);
        let out = mlx_gen::run_mlx_driver_slices(
            &driver,
            [mlx_gen::SliceSpec::full(3)],
            BoundaryState::XSigma(MlxAvState::av(
                Array::from_slice(&[7.0_f32, 1.5], &[1, 2, 1]),
                Array::from_slice(&[1.0_f32, -1.0], &[1, 1, 2, 1]),
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
        let BoundaryState::XSigma(final_av) = out else {
            panic!("expected XSigma");
        };
        mlx_rs::transforms::eval([&final_av.video]).unwrap();
        let v_out = final_av.video.as_slice::<f32>();
        assert_eq!(v_out[0], 7.0, "conditioned token 0 must remain pinned");
        assert_eq!(
            seen_video_ts.borrow()[0],
            vec![0.0, LTX_STAGE2_SIGMAS[0]],
            "video timesteps must equal sigma * denoise_mask"
        );
        assert_eq!(
            seen_audio_ts.borrow()[0],
            vec![LTX_STAGE2_SIGMAS[0], LTX_STAGE2_SIGMAS[0]],
            "audio timesteps must remain strictly uniform sigma"
        );
    }

    #[test]
    fn mlx_ltx_plan_and_double_noise_transition_guard_enforced_weights_free() {
        let plan = LtxRenderPlan::build(LtxRenderPlanInput {
            family: LtxModelFamily::Ltx25,
            variant: LtxVariant::Dev,
            route: LtxOrchestrationRoute::AvGrid,
            conditioning: LtxConditioningProfile::ReplaceLatentKeyframes { count: 1 },
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

        let noisy = BoundaryState::XSigma(Array::from_slice(&[1.0_f32], &[1]));
        let err = apply_mlx_stream_transition(StreamId::Video, &plan.transitions[0], noisy, Ok)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));

        let req_ok = mlx_gen::GenerationRequest {
            prompt: "cinematic drone shot".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            extensions: mlx_gen::gen_core::FamilyExtensions {
                ltx: Some(mlx_gen::gen_core::LtxExtension {
                    num_generated_keyframes: Some(1),
                    temporal_upsample_rounds: Some(1),
                    ..Default::default()
                }),
                ..Default::default()
            },
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

        let req_bad_23 = mlx_gen::GenerationRequest {
            prompt: "bad".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            extensions: mlx_gen::gen_core::FamilyExtensions {
                ltx: Some(mlx_gen::gen_core::LtxExtension {
                    temporal_upsample_rounds: Some(1),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            render_plan_for_request(
                LtxModelFamily::Ltx23,
                crate::dev_sampler::TransformerVariant::Distilled,
                &req_bad_23,
                17,
                24.0,
                false,
            ),
            Err(Error::Unsupported(_))
        ));
    }
}
