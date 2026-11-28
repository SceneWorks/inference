//! Typed ENG-6 `FamilyDriver` implementations and pre-load [`H3RenderPlan`] builders for
//! `mlx-gen-minimax-h3` (`E2 / AC1`, `DEC-3`, `PIPE-6`, `PIPE-7`, `MEM-3`).
//!
//! Covers all MiniMax-H3 routes on MLX:
//! * Joint AV video routes (`t2va`, `fl2va`, `ref2va`) via [`MlxH3JointDriver`] over packed
//!   `[text | video | audio]` sequences without CFG, recording paired `StreamId::Video` (default
//!   shift `12.0`) and `StreamId::Audio` (`3.0`) sigmas onto [`DriverStepInfo`] / [`DriverRunState`]
//!   while pinning conditioning prefixes untouched and honoring synchronous per-step `eval`.
//! * Image routes (`minimax_h3_image` T2I + Edit + PAG) via [`MlxH3ImageDriver`], preserving
//!   6-head `pdd_head_mix`, Euler/LCM/EulerAncestral via [`sampler_step`], optional PAG single
//!   stochastic draw per step, and streamed `window = 1` (`TransformerComponent::Both`).

use std::borrow::Cow;
use std::cell::RefCell;

use gen_core_h3::h3_plan::{H3BudgetDomainKey, H3EnvelopeSpec, H3RenderPlan};
use mlx_gen::driver::{
    DriverEvaluationKind, DriverRefusalReasons, DriverRunState, DriverStepInfo, DriverSupport,
    FamilyDriver, SliceInput, StepHooks, StreamId, StreamStepSigma,
};
use mlx_gen::gen_core::h3_image::{sampler_step, H3ImageSampler, SamplerStep};
use mlx_gen::gen_core::multistream::{ResolvedSchedule, ScheduleSource};
use mlx_gen::gen_core::{
    BoundaryState, Error as GenError, EstimateSource, PagResolved, Result as GenResult,
};
use mlx_gen::{Error, Result};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use crate::denoise::{
    check_block, step_generated_tail, JointSchedule, JointStep, JointVelocity, PackedLayout,
    SigmaSchedule, NUM_ROW_CLASSES,
};
use crate::dit::adaln::TimestepSchedule;
use crate::h3_image::dit::{velocity_euler_delta, ImageDenoiser};
use crate::h3_image::sampler::MlxStepOps;
use crate::tensor::slice_axis;

/// Paired audio+video latent rows carried through [`FamilyDriver<MlxH3AvState>`] slices.
#[derive(Clone, Debug)]
pub struct MlxH3AvState {
    pub video: Array,
    pub audio: Array,
}

impl MlxH3AvState {
    pub fn new(video: Array, audio: Array) -> Self {
        Self { video, audio }
    }
}

pub(crate) fn h3_driver_support(driver_id: &'static str) -> DriverSupport {
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
                "MiniMax-H3 runs a single trajectory without cross-stage re-noising",
            geometry_change: "MiniMax-H3 geometry is fixed pre-load inside H3RenderPlan",
            ..DriverRefusalReasons::default()
        },
    }
}

fn core_err(err: Error) -> GenError {
    err.into()
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

fn check_slice_bounds<L>(
    driver_id: &'static str,
    total_evals: usize,
    input: &SliceInput<'_, L>,
) -> GenResult<()> {
    if input.total_steps != total_evals || input.span.end > total_evals {
        return Err(GenError::Unsupported(format!(
            "{driver_id}: slice {:?} / total_steps={} disagrees with schedule evaluations={total_evals}",
            input.span, input.total_steps
        )));
    }
    Ok(())
}

fn resolve_explicit_schedule(table_id: String, sigmas: &[f32]) -> Result<ResolvedSchedule> {
    ScheduleSource::ExplicitTable {
        table_id,
        sigmas: sigmas.to_vec(),
    }
    .resolve()
    .map_err(Into::into)
}

/// Build and validate a pre-load video [`H3RenderPlan`] under [`H3BudgetDomainKey::UnifiedMemory`]
/// before reading any weights (`DEC-3`, `E2 / AC1`).
pub fn render_video_plan_for_request(req: &mlx_gen::GenerationRequest) -> Result<H3RenderPlan> {
    H3RenderPlan::from_video_generation_request(
        req,
        H3BudgetDomainKey::UnifiedMemory,
        H3EnvelopeSpec::m5_24gib(),
    )
    .map_err(Into::into)
}

/// Build and validate a pre-load video [`H3RenderPlan`] with already-resolved `(width, height,
/// frames)` geometry under [`H3BudgetDomainKey::UnifiedMemory`].
pub fn render_video_plan_for_request_geometry(
    req: &mlx_gen::GenerationRequest,
    width: u32,
    height: u32,
    frames: u32,
) -> Result<H3RenderPlan> {
    H3RenderPlan::from_video_request_with_geometry(
        req,
        width,
        height,
        frames,
        H3BudgetDomainKey::UnifiedMemory,
        H3EnvelopeSpec::m5_24gib(),
    )
    .map_err(Into::into)
}

/// Build and validate a pre-load image [`H3RenderPlan`] under [`H3BudgetDomainKey::UnifiedMemory`]
/// before reading any weights (`DEC-3`, `E2 / AC1`).
pub fn render_image_plan_for_request(req: &mlx_gen::GenerationRequest) -> Result<H3RenderPlan> {
    H3RenderPlan::from_image_generation_request(
        req,
        H3BudgetDomainKey::UnifiedMemory,
        H3EnvelopeSpec::m5_24gib(),
    )
    .map_err(Into::into)
}

// =================================================================================================
// Joint AV Video Driver (`t2va`, `fl2va`, `ref2va`)
// =================================================================================================

fn check_joint_schedules(adaln: &TimestepSchedule, schedule: &JointSchedule) -> Result<()> {
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
    Ok(())
}

fn clean_tail_prediction(
    schedule: &SigmaSchedule,
    index: usize,
    latents: &Array,
    velocity: &Array,
    num_condition_rows: i32,
) -> Result<Array> {
    let sigma = schedule.sigmas()[index];
    if num_condition_rows == 0 {
        let step = multiply(velocity, Array::from_f32(sigma))?;
        return Ok(add(latents, &step)?);
    }
    let rows = latents.shape()[1];
    let head = slice_axis(latents, 1, 0, num_condition_rows)?;
    let tail = slice_axis(latents, 1, num_condition_rows, rows)?;
    let tail_vel = slice_axis(velocity, 1, num_condition_rows, rows)?;
    let clean_tail = add(&tail, &multiply(&tail_vel, Array::from_f32(sigma))?)?;
    Ok(concatenate_axis(&[head, clean_tail], 1)?)
}

/// Typed [`FamilyDriver<MlxH3AvState>`] for MiniMax-H3 joint `[text | video | audio]` denoising
/// across `t2va`, `fl2va`, and `ref2va` (`E2 / AC1`).
pub struct MlxH3JointDriver<'a, M> {
    support: DriverSupport,
    model: RefCell<M>,
    layout: Cow<'a, PackedLayout>,
    schedule: Cow<'a, JointSchedule>,
    adaln: Cow<'a, TimestepSchedule>,
    video_schedule: ResolvedSchedule,
    audio_schedule: ResolvedSchedule,
}

impl<'a, M: JointVelocity> MlxH3JointDriver<'a, M> {
    fn from_cows(
        driver_id: &'static str,
        model: M,
        layout: Cow<'a, PackedLayout>,
        schedule: Cow<'a, JointSchedule>,
        adaln: Cow<'a, TimestepSchedule>,
    ) -> Result<Self> {
        check_joint_schedules(&adaln, &schedule)?;
        let video_schedule = resolve_explicit_schedule(
            format!("h3_video_shift_{:.1}", schedule.video().shift()),
            schedule.video().sigmas(),
        )?;
        let audio_schedule = resolve_explicit_schedule(
            format!("h3_audio_shift_{:.1}", schedule.audio().shift()),
            schedule.audio().sigmas(),
        )?;
        Ok(Self {
            support: h3_driver_support(driver_id),
            model: RefCell::new(model),
            layout,
            schedule,
            adaln,
            video_schedule,
            audio_schedule,
        })
    }

    /// Construct an owned `'static` driver (typically for conformance testing over mock models).
    pub fn owned(
        driver_id: &'static str,
        model: M,
        layout: PackedLayout,
        schedule: JointSchedule,
        adaln: TimestepSchedule,
    ) -> Result<Self> {
        Self::from_cows(
            driver_id,
            model,
            Cow::Owned(layout),
            Cow::Owned(schedule),
            Cow::Owned(adaln),
        )
    }

    /// Borrow `layout`, `schedule`, and `adaln` cleanly inside `denoise_av_with_observer`.
    pub fn borrowed(
        driver_id: &'static str,
        model: M,
        layout: &'a PackedLayout,
        schedule: &'a JointSchedule,
        adaln: &'a TimestepSchedule,
    ) -> Result<Self> {
        Self::from_cows(
            driver_id,
            model,
            Cow::Borrowed(layout),
            Cow::Borrowed(schedule),
            Cow::Borrowed(adaln),
        )
    }

    pub fn video_schedule(&self) -> &ResolvedSchedule {
        &self.video_schedule
    }

    pub fn audio_schedule(&self) -> &ResolvedSchedule {
        &self.audio_schedule
    }

    fn forward_at(&self, index: usize, cur: &MlxH3AvState) -> Result<(Array, Array)> {
        let adaln_indices =
            self.adaln
                .adaln_indices(index, self.layout.row_classes(), self.layout.token_tags())?;
        let row_timesteps = PackedLayout::row_timesteps(
            self.schedule.video().timesteps()[index],
            self.schedule.audio().timesteps()[index],
        );
        self.model.borrow_mut().forward(&JointStep {
            index,
            layout: &self.layout,
            video_rows: &cur.video,
            audio_rows: &cur.audio,
            adaln_indices: &adaln_indices,
            row_timesteps,
        })
    }
}

impl<M: JointVelocity> FamilyDriver<MlxH3AvState> for MlxH3JointDriver<'_, M> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, MlxH3AvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, MlxH3AvState>,
    ) -> GenResult<BoundaryState<MlxH3AvState>> {
        self.support.validate_slice(&input, state)?;
        let total_evals = self.schedule.num_evals();
        check_slice_bounds(self.support.driver_id, total_evals, &input)?;
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
        check_block(&cur.video, self.layout.video_indices().len(), "video").map_err(core_err)?;
        check_block(&cur.audio, self.layout.audio_indices().len(), "audio").map_err(core_err)?;

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (v_sigma, v_next) = self
                .video_schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing H3 video sigma window at {step}")))?;
            let (a_sigma, a_next) = self
                .audio_schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing H3 audio sigma window at {step}")))?;
            let info = make_step_info(
                pass,
                phase,
                total_steps,
                step,
                v_sigma,
                v_next,
                DriverEvaluationKind::Trajectory,
            );
            let stream_sigmas = vec![
                StreamStepSigma {
                    stream: StreamId::Video,
                    sigma: v_sigma,
                    next_sigma: v_next,
                },
                StreamStepSigma {
                    stream: StreamId::Audio,
                    sigma: a_sigma,
                    next_sigma: a_next,
                },
            ];
            hooks.observe_streams(&info, &stream_sigmas, &cur)?;
            state.record_stream_step(&info, stream_sigmas);

            let (vvel, avel) = self.forward_at(step, &cur).map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            cur.video = step_generated_tail(
                self.schedule.video(),
                step,
                &cur.video,
                &vvel,
                self.layout.num_condition_video_rows(),
            )
            .map_err(core_err)?;
            cur.audio = step_generated_tail(
                self.schedule.audio(),
                step,
                &cur.audio,
                &avel,
                self.layout.num_condition_audio_rows(),
            )
            .map_err(core_err)?;

            cur = hooks.post_step(&info, cur)?;
            mlx_rs::transforms::eval([&cur.video, &cur.audio])
                .map_err(Error::from)
                .map_err(core_err)?;
            hooks.materialize(&info, &cur)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(span.end);

        if request_boundary_estimate {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            if span.end < total_evals {
                let (vvel, avel) = self.forward_at(span.end, &cur).map_err(core_err)?;
                let v_clean = clean_tail_prediction(
                    self.schedule.video(),
                    span.end,
                    &cur.video,
                    &vvel,
                    self.layout.num_condition_video_rows(),
                )
                .map_err(core_err)?;
                let a_clean = clean_tail_prediction(
                    self.schedule.audio(),
                    span.end,
                    &cur.audio,
                    &avel,
                    self.layout.num_condition_audio_rows(),
                )
                .map_err(core_err)?;
                mlx_rs::transforms::eval([&v_clean, &a_clean])
                    .map_err(Error::from)
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: MlxH3AvState {
                        video: v_clean,
                        audio: a_clean,
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
// H3-Image Driver (`minimax_h3_image`: T2I + Edit + PAG)
// =================================================================================================

/// Seam abstracting MLX image velocity evaluation (both [`ImageDenoiser`] and weights-free mocks).
pub trait MlxH3ImageForward {
    fn sigmas(&self) -> &[f32];
    fn timesteps(&self) -> &[f32];
    fn num_reference_rows(&self) -> i32;
    fn forward_cond(&mut self, step: usize, image_rows: &Array) -> Result<Array>;
    fn forward_perturbed(
        &mut self,
        step: usize,
        image_rows: &Array,
        perturbed_layers: &[u32],
    ) -> Result<Array>;
}

impl MlxH3ImageForward for &ImageDenoiser {
    fn sigmas(&self) -> &[f32] {
        self.schedule().sigmas()
    }

    fn timesteps(&self) -> &[f32] {
        self.schedule().timesteps()
    }

    fn num_reference_rows(&self) -> i32 {
        self.layout().num_reference_rows()
    }

    fn forward_cond(&mut self, step: usize, image_rows: &Array) -> Result<Array> {
        self.velocity(step, image_rows)
    }

    fn forward_perturbed(
        &mut self,
        step: usize,
        image_rows: &Array,
        perturbed_layers: &[u32],
    ) -> Result<Array> {
        self.velocity_perturbed(step, image_rows, perturbed_layers)
    }
}

fn image_sigma_at(timesteps: &[f32], index: usize) -> f32 {
    timesteps.get(index).map_or(0.0, |&t| 1.0 - t)
}

fn sampler_draws_step_noise(sampler: &H3ImageSampler, sigma_next: f32) -> bool {
    sigma_next > 0.0
        && matches!(
            sampler,
            H3ImageSampler::Lcm { .. } | H3ImageSampler::EulerAncestral
        )
}

/// Shared single-step update over packed `[1, refs + gen, C]` image rows (`AP9`: invoked by both
/// `ImageDenoiser::step_with_pag` and `MlxH3ImageDriver::run_slice`).
pub(crate) fn step_packed_image_rows<M: MlxH3ImageForward>(
    model: &mut M,
    sampler: &H3ImageSampler,
    pag: Option<&PagResolved>,
    image_seed: u64,
    step: usize,
    image_rows: &Array,
    mut run_state: Option<&mut DriverRunState>,
) -> Result<Array> {
    let timesteps = model.timesteps().to_vec();
    let num_steps = timesteps.len();
    velocity_euler_delta(&timesteps, step)?;

    let refs = model.num_reference_rows();
    let all = image_rows.shape()[1];
    if refs < 0 || refs >= all {
        return Err(Error::Msg(format!(
            "minimax-h3-image denoise: {refs} reference rows of {all} leave no generated row"
        )));
    }
    let step_pag_scale = pag.map_or(0.0, |p| p.step_scale(step, num_steps));

    let v_cond = model.forward_cond(step, image_rows)?;
    if let Some(st) = run_state.as_deref_mut() {
        st.record_evaluation(DriverEvaluationKind::Trajectory);
    }
    let v_cond_gen = slice_axis(&v_cond.as_dtype(Dtype::Float32)?, 1, refs, all)?;
    let v_gen = if step_pag_scale > 0.0 {
        let layers = pag.map_or(&[][..], |p| p.step_layers(step, num_steps));
        mlx_rs::transforms::eval([&v_cond_gen])?;
        drop(v_cond);
        let v_perturbed = model.forward_perturbed(step, image_rows, layers)?;
        if let Some(st) = run_state.as_deref_mut() {
            st.record_evaluation(DriverEvaluationKind::Trajectory);
        }
        let v_perturbed_gen = slice_axis(&v_perturbed.as_dtype(Dtype::Float32)?, 1, refs, all)?;
        mlx_gen::gen_core::pag_apply(
            &mlx_gen::MlxLatentOps,
            &v_cond_gen,
            &v_cond_gen,
            &v_perturbed_gen,
            step_pag_scale,
        )?
    } else {
        v_cond_gen
    };

    let sigma = image_sigma_at(&timesteps, step);
    let sigma_next = image_sigma_at(&timesteps, step + 1);
    if let Some(st) = run_state {
        if sampler_draws_step_noise(sampler, sigma_next) {
            let _ = st.next_draw_index();
        }
    }

    let x = image_rows.as_dtype(Dtype::Float32)?;
    let generated = sampler_step(
        &mut MlxStepOps,
        sampler,
        SamplerStep {
            step,
            num_steps,
            sigma,
            sigma_next,
            image_seed,
        },
        &slice_axis(&x, 1, refs, all)?,
        &v_gen,
    )?;
    if refs == 0 {
        return Ok(generated);
    }
    Ok(concatenate_axis(
        &[slice_axis(&x, 1, 0, refs)?, generated],
        1,
    )?)
}

/// Typed [`FamilyDriver<Array>`] for `minimax_h3_image` (T2I + Edit + PAG).
pub struct MlxH3ImageDriver<'a, M> {
    support: DriverSupport,
    model: RefCell<M>,
    sampler: H3ImageSampler,
    pag: Option<Cow<'a, PagResolved>>,
    image_seed: u64,
    schedule: ResolvedSchedule,
}

impl<'a, M: MlxH3ImageForward> MlxH3ImageDriver<'a, M> {
    pub fn new(
        driver_id: &'static str,
        model: M,
        sampler: H3ImageSampler,
        pag: Option<Cow<'a, PagResolved>>,
        image_seed: u64,
    ) -> Result<Self> {
        let schedule = resolve_explicit_schedule("h3_image".into(), model.sigmas())?;
        Ok(Self {
            support: h3_driver_support(driver_id),
            model: RefCell::new(model),
            sampler,
            pag,
            image_seed,
            schedule,
        })
    }

    pub fn schedule(&self) -> &ResolvedSchedule {
        &self.schedule
    }
}

impl<M: MlxH3ImageForward> FamilyDriver<Array> for MlxH3ImageDriver<'_, M> {
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
        let num_steps = self.schedule.num_steps();
        check_slice_bounds(self.support.driver_id, num_steps, &input)?;
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
        let mut x = extract_state(boundary_state)
            .as_dtype(Dtype::Float32)
            .map_err(Error::from)
            .map_err(core_err)?;

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (sigma, sigma_next) = self
                .schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing H3 image sigma window at {step}")))?;
            let info = make_step_info(
                pass,
                phase,
                total_steps,
                step,
                sigma,
                sigma_next,
                DriverEvaluationKind::Trajectory,
            );
            let stream_sigmas = vec![StreamStepSigma {
                stream: StreamId::Video,
                sigma,
                next_sigma: sigma_next,
            }];
            hooks.observe_streams(&info, &stream_sigmas, &x)?;
            state.record_stream_step(&info, stream_sigmas);

            let mut model = self.model.borrow_mut();
            x = step_packed_image_rows(
                &mut *model,
                &self.sampler,
                self.pag.as_deref(),
                self.image_seed,
                step,
                &x,
                Some(state),
            )
            .map_err(core_err)?;
            drop(model);

            x = hooks.post_step(&info, x)?;
            mlx_rs::transforms::eval([&x])
                .map_err(Error::from)
                .map_err(core_err)?;
            hooks.materialize(&info, &x)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(span.end);

        if request_boundary_estimate {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            if span.end < num_steps {
                let boundary_sigma = self.schedule.sigmas()[span.end];
                let mut model = self.model.borrow_mut();
                let refs = model.num_reference_rows();
                let all = x.shape()[1];
                let v_cond = model.forward_cond(span.end, &x).map_err(core_err)?;
                let x_f32 = x
                    .as_dtype(Dtype::Float32)
                    .map_err(Error::from)
                    .map_err(core_err)?;
                let v_f32 = v_cond
                    .as_dtype(Dtype::Float32)
                    .map_err(Error::from)
                    .map_err(core_err)?;
                let v_gen = slice_axis(&v_f32, 1, refs, all).map_err(core_err)?;
                let x_gen = slice_axis(&x_f32, 1, refs, all).map_err(core_err)?;
                let x0_gen = add(
                    &x_gen,
                    &multiply(&v_gen, Array::from_f32(boundary_sigma))
                        .map_err(Error::from)
                        .map_err(core_err)?,
                )
                .map_err(Error::from)
                .map_err(core_err)?;
                let x0 = if refs > 0 {
                    let prefix = slice_axis(&x_f32, 1, 0, refs).map_err(core_err)?;
                    concatenate_axis(&[prefix, x0_gen], 1)
                        .map_err(Error::from)
                        .map_err(core_err)?
                } else {
                    x0_gen
                };
                mlx_rs::transforms::eval([&x0])
                    .map_err(Error::from)
                    .map_err(core_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: x0,
                    source: EstimateSource::EndpointEvaluation,
                });
            }
            state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
            return Ok(BoundaryState::CleanEstimate {
                latent: x,
                source: EstimateSource::PreviousEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise::{
        adaln_schedule, JointGeometry, AUDIO_SIGMA_SHIFT, TEXT_TAG, VIDEO_SIGMA_SHIFT,
    };
    use crate::dit::positions::KeyframeAnchor;
    use mlx_gen::driver::conformance::{
        verify_driver_conformance, verify_multistream_driver_conformance, ConformanceCase,
        MultiStreamConformanceCase,
    };
    use mlx_gen::gen_core::h3_image::{minimax_h3_image_pag_capability, H3ImageScheduler};
    use mlx_gen::gen_core::multistream::{StreamTransition, StreamTransitionKind};
    use mlx_gen::gen_core::{
        GenerationMemory, GenerationRequest, PagConfig, TransformerComponent, VedaExecutionReport,
        VedaRequest,
    };

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

    fn eq_av(a: &MlxH3AvState, b: &MlxH3AvState) -> bool {
        eq_f32_arrays(&a.video, &b.video) && eq_f32_arrays(&a.audio, &b.audio)
    }

    #[derive(Clone)]
    struct MockJoint {
        scale: f32,
        report: std::rc::Rc<RefCell<VedaExecutionReport>>,
        fallback_from_step: Option<usize>,
    }

    impl JointVelocity for MockJoint {
        fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
            let mut r = self.report.borrow_mut();
            if self.fallback_from_step.is_some_and(|s| step.index >= s) {
                r.dense_calls += 1;
                if r.fallback_reason.is_none() {
                    r.fallback_reason = Some("simulated dense fallback under pressure".into());
                }
            } else {
                r.sparse_calls += 1;
            }
            let k = Array::from_f32(self.scale + (step.index as f32) * 0.01);
            Ok((
                mlx_rs::ops::multiply(step.video_rows, &k)?,
                mlx_rs::ops::multiply(step.audio_rows, &k)?,
            ))
        }
    }

    #[test]
    fn mlx_h3_joint_driver_passes_multistream_conformance_across_t2va_and_fl2va() {
        let sched = JointSchedule::with_shifts(9, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT).unwrap();
        let adaln = adaln_schedule(&sched).unwrap();
        let g = JointGeometry::new(124, 4, 6).unwrap();
        let layout = PackedLayout::build(
            g,
            [1, 2, 2],
            &[TEXT_TAG; 4],
            2,
            &[KeyframeAnchor::First, KeyframeAnchor::Last],
        )
        .unwrap();
        let v_rows = layout.video_indices().len() as i32;
        let a_rows = layout.audio_indices().len() as i32;
        let video_resolved =
            resolve_explicit_schedule("h3_video_12".into(), sched.video().sigmas()).unwrap();
        let audio_resolved =
            resolve_explicit_schedule("h3_audio_3".into(), sched.audio().sigmas()).unwrap();
        assert_ne!(
            video_resolved.sigmas(),
            audio_resolved.sigmas(),
            "paired video/audio schedules must preserve distinct shifts 12.0 vs 3.0"
        );

        // `verify_multistream_driver_conformance` passes a single target boundary state when probing
        // the guard, so pass `StreamId::Video` to the harness and assert the paired `[Video, Audio]`
        // guard directly below.
        let transition = StreamTransition::new(
            "h3_no_renoise_boundary",
            vec![StreamId::Video],
            StreamTransitionKind::Renoise {
                sigma: sched.video().sigmas()[4],
            },
        )
        .unwrap();

        let report_sink = std::rc::Rc::new(RefCell::new(VedaExecutionReport {
            requested: true,
            ..Default::default()
        }));

        let rep = verify_multistream_driver_conformance(MultiStreamConformanceCase {
            base: ConformanceCase {
                total_steps: 8,
                split_at: 4,
                seed: 21,
                base_geometry: (1344, 768),
                alt_geometry: (960, 544),
                make_driver: {
                    let layout = layout.clone();
                    let sched = sched.clone();
                    let adaln = adaln.clone();
                    let sink = report_sink.clone();
                    move || {
                        Box::new(
                            MlxH3JointDriver::owned(
                                "mlx_h3_joint_test",
                                MockJoint {
                                    scale: 0.15,
                                    report: sink.clone(),
                                    fallback_from_step: Some(5),
                                },
                                layout.clone(),
                                sched.clone(),
                                adaln.clone(),
                            )
                            .unwrap(),
                        )
                    }
                },
                make_initial: move || {
                    let vv: Vec<f32> = (0..(v_rows as usize * 4))
                        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                        .collect();
                    let av: Vec<f32> = (0..(a_rows as usize * 3))
                        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
                        .collect();
                    MlxH3AvState::new(
                        Array::from_slice(&vv, &[1, v_rows, 4]),
                        Array::from_slice(&av, &[1, a_rows, 3]),
                    )
                },
                latents_equal: eq_av,
            },
            expected_stream_schedules: vec![
                (StreamId::Video, video_resolved),
                (StreamId::Audio, audio_resolved),
            ],
            boundary_transition: Some(transition),
        })
        .unwrap();
        assert!(rep.base.verified_inert_split);
        assert!(rep.base.verified_cancellation);
        assert!(rep.base.verified_boundary_estimate_readonly_or_refusal);
        assert!(rep.base.verified_renoise_and_resample_guards);
        assert!(rep.verified_transition_guard);

        let paired_transition = StreamTransition::new(
            "h3_paired_av_clean_boundary",
            vec![StreamId::Video, StreamId::Audio],
            StreamTransitionKind::Renoise {
                sigma: sched.video().sigmas()[4],
            },
        )
        .unwrap();
        let clean = BoundaryState::CleanEstimate {
            latent: (),
            source: EstimateSource::EndpointEvaluation,
        };
        let noisy = BoundaryState::XSigma(());
        paired_transition
            .validate_boundary_states(&[(StreamId::Video, &clean), (StreamId::Audio, &clean)])
            .unwrap();
        assert!(matches!(
            paired_transition
                .validate_boundary_states(&[(StreamId::Video, &clean), (StreamId::Audio, &noisy)]),
            Err(GenError::Unsupported(_))
        ));

        let vr = report_sink.borrow();
        assert!(vr.sparse_calls > 0);
        assert!(vr.dense_calls > 0);
        assert!(vr.fallback_reason.is_some());
    }

    struct MockImageVel {
        sigmas: Vec<f32>,
        timesteps: Vec<f32>,
        refs: i32,
        cond_val: f32,
        pert_val: f32,
    }

    impl MlxH3ImageForward for MockImageVel {
        fn sigmas(&self) -> &[f32] {
            &self.sigmas
        }

        fn timesteps(&self) -> &[f32] {
            &self.timesteps
        }

        fn num_reference_rows(&self) -> i32 {
            self.refs
        }

        fn forward_cond(&mut self, step: usize, image_rows: &Array) -> Result<Array> {
            let k = Array::from_f32(self.cond_val + step as f32 * 0.1);
            Ok(mlx_rs::ops::multiply(image_rows, &k)?)
        }

        fn forward_perturbed(
            &mut self,
            step: usize,
            image_rows: &Array,
            _perturbed_layers: &[u32],
        ) -> Result<Array> {
            let k = Array::from_f32(self.pert_val + step as f32 * 0.1);
            Ok(mlx_rs::ops::multiply(image_rows, &k)?)
        }
    }

    #[test]
    fn mlx_h3_image_driver_passes_conformance_across_samplers_and_edit_pag() {
        let sched = SigmaSchedule::h3_image(H3ImageScheduler::Normal).unwrap();
        let resolved = resolve_explicit_schedule("h3_image".into(), sched.sigmas()).unwrap();
        let cap = minimax_h3_image_pag_capability();
        let pag = cap
            .resolve(Some(&PagConfig {
                scale: 1.2,
                ..Default::default()
            }))
            .unwrap();

        for (label, refs, sampler, use_pag) in [
            ("euler_t2i", 0, H3ImageSampler::Euler, false),
            (
                "lcm_edit_pag",
                4,
                H3ImageSampler::Lcm {
                    s_noise: 1.1,
                    s_noise_end: 0.9,
                    noise_clip_std: 0.0,
                },
                true,
            ),
            ("ancestral_t2i", 0, H3ImageSampler::EulerAncestral, false),
        ] {
            let pag_opt = use_pag.then(|| Cow::Owned(pag.clone()));
            let rep = verify_multistream_driver_conformance(MultiStreamConformanceCase {
                base: ConformanceCase {
                    total_steps: 6,
                    split_at: 3,
                    seed: 77,
                    base_geometry: (2048, 2048),
                    alt_geometry: (1024, 1024),
                    make_driver: {
                        let sigmas = sched.sigmas().to_vec();
                        let timesteps = sched.timesteps().to_vec();
                        let pag_opt = pag_opt.clone();
                        move || {
                            Box::new(
                                MlxH3ImageDriver::new(
                                    "mlx_h3_image_test",
                                    MockImageVel {
                                        sigmas: sigmas.clone(),
                                        timesteps: timesteps.clone(),
                                        refs,
                                        cond_val: 0.4,
                                        pert_val: 0.2,
                                    },
                                    sampler,
                                    pag_opt.clone(),
                                    77,
                                )
                                .unwrap(),
                            )
                        }
                    },
                    make_initial: move || {
                        let total = refs + 4;
                        let vals: Vec<f32> = (0..(total as usize * 96))
                            .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
                            .collect();
                        Array::from_slice(&vals, &[1, total, 96])
                    },
                    latents_equal: eq_f32_arrays,
                },
                expected_stream_schedules: vec![(StreamId::Video, resolved.clone())],
                boundary_transition: None,
            })
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(rep.base.verified_inert_split, "{label}");
            assert!(rep.base.verified_cancellation, "{label}");
            if matches!(
                sampler,
                H3ImageSampler::Lcm { .. } | H3ImageSampler::EulerAncestral
            ) {
                assert_eq!(rep.base.unsplit_draws, 5, "{label}: 5 non-terminal draws");
                assert!(rep.base.verified_reset_preserves_draws, "{label}");
            } else {
                assert_eq!(rep.base.unsplit_draws, 0, "{label}");
            }

            // Full Tier 0 scalar conformance sanity.
            let scalar = verify_driver_conformance(ConformanceCase {
                total_steps: 6,
                split_at: 2,
                seed: 19,
                base_geometry: (2048, 2048),
                alt_geometry: (1024, 1024),
                make_driver: {
                    let sigmas = sched.sigmas().to_vec();
                    let timesteps = sched.timesteps().to_vec();
                    move || {
                        Box::new(
                            MlxH3ImageDriver::new(
                                "mlx_h3_image_scalar_test",
                                MockImageVel {
                                    sigmas: sigmas.clone(),
                                    timesteps: timesteps.clone(),
                                    refs,
                                    cond_val: 0.35,
                                    pert_val: 0.15,
                                },
                                sampler,
                                None,
                                19,
                            )
                            .unwrap(),
                        )
                    }
                },
                make_initial: move || {
                    Array::from_slice(&vec![0.5_f32; (refs + 4) as usize * 96], &[1, refs + 4, 96])
                },
                latents_equal: eq_f32_arrays,
            })
            .unwrap();
            assert!(scalar.verified_inert_split);
        }
    }

    #[test]
    fn mlx_preload_render_plan_validates_m5_envelope_before_weight_load() {
        let mut req = GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width: 1344,
            height: 768,
            frames: Some(345),
            steps: Some(8),
            extensions: mlx_gen::gen_core::FamilyExtensions {
                h3: Some(mlx_gen::gen_core::H3Extension {
                    veda: Some(VedaRequest {
                        enabled: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            memory: Some(GenerationMemory {
                stage_residency: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::Both),
                ..Default::default()
            }),
            ..Default::default()
        };
        let plan = render_video_plan_for_request(&req).unwrap();
        assert_eq!(plan.memory.domain, H3BudgetDomainKey::UnifiedMemory);
        assert!(plan.memory.fits_pool_budget);
        assert!(plan.memory.fits_total_envelope);

        // Invalid streamed window refused pre-load.
        req.memory.as_mut().unwrap().transformer_window_component = Some(TransformerComponent::Dit);
        assert!(render_video_plan_for_request(&req).is_err());

        // Image route plan validation.
        let img_req = GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width: 2048,
            height: 2048,
            ..Default::default()
        };
        let img_plan = render_image_plan_for_request(&img_req).unwrap();
        assert_eq!(img_plan.steps, 6);
    }
}
