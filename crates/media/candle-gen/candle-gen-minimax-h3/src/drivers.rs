//! Typed ENG-6 `FamilyDriver` implementations and pre-load [`H3RenderPlan`] builders for
//! `candle-gen-minimax-h3` (`E2 / AC1`, `DEC-3`, `PIPE-6`, `PIPE-7`, `MEM-3`).
//!
//! Covers all MiniMax-H3 routes on Candle:
//! * Joint AV video routes (`t2va`, `fl2va`, `ref2va`) via [`CandleH3JointDriver`] over packed
//!   `[text | video | audio]` sequences without CFG, preserving distinct paired `StreamId::Video`
//!   (`12.0` default / overridable) and `StreamId::Audio` (`3.0`) sigma schedules,
//!   `step_generated_tail` conditioning-row pinning, Veda sparse + dense fallback + cancellation,
//!   per-step `device.synchronize()`, and streamed window lifecycles (`block_stream::RequestGuard`,
//!   `prepare_streamed`, `AdaLnCache::precompute_and_evict`).
//! * Image routes (`minimax_h3_image` T2I + Edit + PAG) via [`CandleH3ImageDriver`], preserving
//!   6-head `pdd_head_mix`, Euler/LCM/EulerAncestral via [`sampler_step`], single stochastic draw
//!   per active step under PAG, reference prefix freezing, and streamed `window = 1`
//!   (`TransformerComponent::Both`).

use std::borrow::Cow;
use std::cell::RefCell;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::driver::{
    DriverEvaluationKind, DriverRefusalReasons, DriverRunState, DriverStepInfo, DriverSupport,
    FamilyDriver, SliceInput, StepHooks, StreamId, StreamStepSigma,
};
use candle_gen::gen_core::h3_image::{pdd_head_mix, sampler_step, H3ImageSampler, SamplerStep};
use candle_gen::gen_core::multistream::{ResolvedSchedule, ScheduleSource};
use candle_gen::gen_core::{
    BoundaryState, Error as GenError, EstimateSource, PagResolved, Result as GenResult,
};
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_plan::{H3BudgetDomainKey, H3EnvelopeSpec, H3RenderPlan};

use crate::denoise::{
    check_block, step_generated_tail, JointSchedule, JointStep, JointVelocity, PackedLayout,
    SigmaSchedule, NUM_ROW_CLASSES,
};
use crate::dit::adaln::TimestepSchedule;
use crate::image_pipeline::ImageVelocity;
use crate::image_sampler::CandleStepOps;

/// Paired audio+video latent rows carried through [`FamilyDriver<CandleH3AvState>`] slices.
#[derive(Clone, Debug)]
pub struct CandleH3AvState {
    pub video: Tensor,
    pub audio: Tensor,
}

impl CandleH3AvState {
    pub fn new(video: Tensor, audio: Tensor) -> Self {
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

fn core_err(err: CandleError) -> GenError {
    err.into()
}

fn ccore_err(err: candle_gen::candle_core::Error) -> GenError {
    CandleError::from(err).into()
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

/// Build and validate a pre-load video [`H3RenderPlan`] under [`H3BudgetDomainKey::DeviceVram`]
/// before reading any weights (`DEC-3`, `E2 / AC1`).
pub fn render_video_plan_for_request(
    req: &candle_gen::gen_core::GenerationRequest,
) -> candle_gen::gen_core::Result<H3RenderPlan> {
    H3RenderPlan::from_video_generation_request(
        req,
        H3BudgetDomainKey::DeviceVram,
        H3EnvelopeSpec::m5_24gib(),
    )
}

/// Build and validate a pre-load video [`H3RenderPlan`] with already-resolved `(width, height,
/// frames)` geometry under [`H3BudgetDomainKey::DeviceVram`].
pub fn render_video_plan_for_request_geometry(
    req: &candle_gen::gen_core::GenerationRequest,
    width: u32,
    height: u32,
    frames: u32,
) -> candle_gen::gen_core::Result<H3RenderPlan> {
    H3RenderPlan::from_video_request_with_geometry(
        req,
        width,
        height,
        frames,
        H3BudgetDomainKey::DeviceVram,
        H3EnvelopeSpec::m5_24gib(),
    )
}

/// Build and validate a pre-load image [`H3RenderPlan`] under [`H3BudgetDomainKey::DeviceVram`]
/// with resolved output `(width, height)` before reading any weights (`DEC-3`, `E2 / AC1`).
pub fn render_image_plan_for_request_size(
    req: &candle_gen::gen_core::GenerationRequest,
    width: u32,
    height: u32,
) -> candle_gen::gen_core::Result<H3RenderPlan> {
    H3RenderPlan::from_image_request_with_size(
        req,
        width,
        height,
        H3BudgetDomainKey::DeviceVram,
        H3EnvelopeSpec::m5_24gib(),
    )
}

// =================================================================================================
// Joint AV Video Driver (`t2va`, `fl2va`, `ref2va`)
// =================================================================================================

fn check_joint_schedules(adaln: &TimestepSchedule, schedule: &JointSchedule) -> Result<()> {
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
    Ok(())
}

fn clean_tail_prediction(
    schedule: &SigmaSchedule,
    index: usize,
    latents: &Tensor,
    velocity: &Tensor,
    num_condition_rows: usize,
) -> Result<Tensor> {
    let sigma = f64::from(schedule.sigmas()[index]);
    if num_condition_rows == 0 {
        let step = (velocity.to_dtype(DType::F32)? * sigma)?;
        return Ok((&latents.to_dtype(DType::F32)? + &step)?);
    }
    let rows = latents.dims()[1];
    let head = latents.narrow(1, 0, num_condition_rows)?;
    let tail = latents
        .narrow(1, num_condition_rows, rows - num_condition_rows)?
        .to_dtype(DType::F32)?;
    let tail_vel = velocity
        .narrow(1, num_condition_rows, rows - num_condition_rows)?
        .to_dtype(DType::F32)?;
    let clean_tail = (&tail + (&tail_vel * sigma)?)?;
    let head = head.to_dtype(clean_tail.dtype())?;
    Ok(Tensor::cat(&[&head, &clean_tail], 1)?.contiguous()?)
}

/// Typed [`FamilyDriver<CandleH3AvState>`] for MiniMax-H3 joint `[text | video | audio]` denoising
/// across `t2va`, `fl2va`, and `ref2va` (`E2 / AC1`).
pub struct CandleH3JointDriver<'a, M> {
    support: DriverSupport,
    model: RefCell<M>,
    layout: Cow<'a, PackedLayout>,
    schedule: Cow<'a, JointSchedule>,
    adaln: Cow<'a, TimestepSchedule>,
    device: Device,
    video_schedule: ResolvedSchedule,
    audio_schedule: ResolvedSchedule,
}

impl<'a, M: JointVelocity> CandleH3JointDriver<'a, M> {
    fn from_cows(
        driver_id: &'static str,
        model: M,
        layout: Cow<'a, PackedLayout>,
        schedule: Cow<'a, JointSchedule>,
        adaln: Cow<'a, TimestepSchedule>,
        device: Device,
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
            device,
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
        device: Device,
    ) -> Result<Self> {
        Self::from_cows(
            driver_id,
            model,
            Cow::Owned(layout),
            Cow::Owned(schedule),
            Cow::Owned(adaln),
            device,
        )
    }

    /// Borrow `layout`, `schedule`, and `adaln` cleanly inside `denoise_av_with_observer`.
    pub fn borrowed(
        driver_id: &'static str,
        model: M,
        layout: &'a PackedLayout,
        schedule: &'a JointSchedule,
        adaln: &'a TimestepSchedule,
        device: &Device,
    ) -> Result<Self> {
        Self::from_cows(
            driver_id,
            model,
            Cow::Borrowed(layout),
            Cow::Borrowed(schedule),
            Cow::Borrowed(adaln),
            device.clone(),
        )
    }

    pub fn video_schedule(&self) -> &ResolvedSchedule {
        &self.video_schedule
    }

    pub fn audio_schedule(&self) -> &ResolvedSchedule {
        &self.audio_schedule
    }

    fn forward_at(&self, index: usize, cur: &CandleH3AvState) -> Result<(Tensor, Tensor)> {
        let adaln_indices = self.adaln.adaln_indices(
            index,
            self.layout.row_classes(),
            self.layout.token_tags(),
            &self.device,
        )?;
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

impl<M: JointVelocity> FamilyDriver<CandleH3AvState> for CandleH3JointDriver<'_, M> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, CandleH3AvState>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, CandleH3AvState>,
    ) -> GenResult<BoundaryState<CandleH3AvState>> {
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

            let step_clock = std::time::Instant::now();
            tracing::info!(
                step = step + 1,
                total = total_evals,
                seq_len = self.layout.seq_len(),
                "minimax-h3: denoise step start"
            );

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
            self.device.synchronize().map_err(ccore_err)?;
            tracing::info!(
                step = step + 1,
                total = total_evals,
                elapsed_ms = step_clock.elapsed().as_millis() as u64,
                "minimax-h3: denoise step done"
            );
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
                self.device.synchronize().map_err(ccore_err)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                return Ok(BoundaryState::CleanEstimate {
                    latent: CandleH3AvState {
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

fn sampler_draws_step_noise(sampler: &H3ImageSampler, sigma_next: f32) -> bool {
    sigma_next > 0.0
        && matches!(
            sampler,
            H3ImageSampler::Lcm { .. } | H3ImageSampler::EulerAncestral
        )
}

/// Typed [`FamilyDriver<Tensor>`] for `minimax_h3_image` (T2I + Edit + PAG).
///
/// Carries generated image rows `[1, gen, F]` while prepending any immutable `condition` prefix
/// (`[1, refs, F]`) for model evaluation, preserving 6-head `pdd_head_mix`, Euler/LCM/Ancestral
/// updates via [`sampler_step`], and single stochastic noise draws under active PAG.
pub struct CandleH3ImageDriver<'a, M> {
    support: DriverSupport,
    model: RefCell<M>,
    sigma_schedule: Cow<'a, SigmaSchedule>,
    resolved_schedule: ResolvedSchedule,
    sampler: H3ImageSampler,
    pag: Option<Cow<'a, PagResolved>>,
    image_seed: u64,
    condition: Option<Tensor>,
    device: Device,
}

impl<'a, M: ImageVelocity> CandleH3ImageDriver<'a, M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        driver_id: &'static str,
        model: M,
        sigma_schedule: Cow<'a, SigmaSchedule>,
        sampler: H3ImageSampler,
        pag: Option<Cow<'a, PagResolved>>,
        image_seed: u64,
        condition: Option<Tensor>,
        device: Device,
    ) -> Result<Self> {
        let resolved_schedule =
            resolve_explicit_schedule("h3_image".into(), sigma_schedule.sigmas())?;
        Ok(Self {
            support: h3_driver_support(driver_id),
            model: RefCell::new(model),
            sigma_schedule,
            resolved_schedule,
            sampler,
            pag,
            image_seed,
            condition,
            device,
        })
    }

    pub fn schedule(&self) -> &ResolvedSchedule {
        &self.resolved_schedule
    }

    fn num_reference_rows(&self) -> usize {
        self.condition.as_ref().map_or(0, |c| c.dims()[1])
    }

    fn pack_rows(&self, generated: &Tensor) -> Result<Tensor> {
        match &self.condition {
            None => Ok(generated.clone()),
            Some(cond) => Ok(Tensor::cat(&[cond, generated], 1)?),
        }
    }
}

impl<M: ImageVelocity> FamilyDriver<Tensor> for CandleH3ImageDriver<'_, M> {
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
        let num_steps = self.resolved_schedule.num_steps();
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
            .to_dtype(DType::F32)
            .map_err(ccore_err)?;
        let num_reference_rows = self.num_reference_rows();
        let sigmas = self.sigma_schedule.sigmas();

        for step in span.clone() {
            if cancel.is_cancelled() {
                return Err(GenError::Canceled);
            }
            let (sigma, sigma_next) = self
                .resolved_schedule
                .step_window(step)
                .ok_or_else(|| GenError::Msg(format!("missing H3 image sigma window at {step}")))?;
            self.sigma_schedule.velocity_delta(step)?;
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

            let packed = self.pack_rows(&x).map_err(core_err)?;
            let heads = pdd_head_mix(sigmas, step)?;
            let velocity_cond = self
                .model
                .borrow_mut()
                .forward_heads(step, &heads, &packed)
                .map_err(core_err)?;
            if velocity_cond.dims() != packed.dims() {
                return Err(GenError::Msg(format!(
                    "minimax-h3-image denoise: step {step} returned a {:?} velocity for {:?} rows",
                    velocity_cond.dims(),
                    packed.dims()
                )));
            }
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            let v_cond_gen = velocity_cond
                .narrow(1, num_reference_rows, x.dims()[1])
                .map_err(ccore_err)?
                .to_dtype(DType::F32)
                .map_err(ccore_err)?;

            let step_pag_scale = self
                .pag
                .as_deref()
                .map_or(0.0, |p| p.step_scale(step, num_steps));
            let velocity = if step_pag_scale > 0.0 {
                let layers = self
                    .pag
                    .as_deref()
                    .map_or(&[][..], |p| p.step_layers(step, num_steps));
                let velocity_pert = self
                    .model
                    .borrow_mut()
                    .forward_perturbed_heads(step, &heads, &packed, layers)
                    .map_err(core_err)?;
                if velocity_pert.dims() != packed.dims() {
                    return Err(GenError::Msg(format!(
                        "minimax-h3-image denoise: step {step} perturbed pass returned a {:?} \
                         velocity for {:?} rows",
                        velocity_pert.dims(),
                        packed.dims()
                    )));
                }
                state.record_evaluation(DriverEvaluationKind::Trajectory);
                let v_pert_gen = velocity_pert
                    .narrow(1, num_reference_rows, x.dims()[1])
                    .map_err(ccore_err)?
                    .to_dtype(DType::F32)
                    .map_err(ccore_err)?;
                candle_gen::gen_core::pag_apply(
                    &candle_gen::CandleLatentOps,
                    &v_cond_gen,
                    &v_cond_gen,
                    &v_pert_gen,
                    step_pag_scale,
                )?
            } else {
                v_cond_gen
            };

            if sampler_draws_step_noise(&self.sampler, sigma_next) {
                let _ = state.next_draw_index();
            }
            let mut ops = CandleStepOps {
                device: &self.device,
            };
            x = sampler_step(
                &mut ops,
                &self.sampler,
                SamplerStep {
                    step,
                    num_steps,
                    sigma: sigmas[step],
                    sigma_next: sigmas[step + 1],
                    image_seed: self.image_seed,
                },
                &x,
                &velocity,
            )
            .map_err(core_err)?;

            x = hooks.post_step(&info, x)?;
            self.device.synchronize().map_err(ccore_err)?;
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
                let boundary_sigma = sigmas[span.end];
                let packed = self.pack_rows(&x).map_err(core_err)?;
                let heads = pdd_head_mix(sigmas, span.end)?;
                let velocity_cond = self
                    .model
                    .borrow_mut()
                    .forward_heads(span.end, &heads, &packed)
                    .map_err(core_err)?;
                if velocity_cond.dims() != packed.dims() {
                    return Err(GenError::Msg(format!(
                        "minimax-h3-image denoise: boundary {} returned a {:?} velocity for {:?} rows",
                        span.end,
                        velocity_cond.dims(),
                        packed.dims()
                    )));
                }
                let v_gen = velocity_cond
                    .narrow(1, num_reference_rows, x.dims()[1])
                    .map_err(ccore_err)?
                    .to_dtype(DType::F32)
                    .map_err(ccore_err)?;
                let scaled = (&v_gen * f64::from(boundary_sigma)).map_err(ccore_err)?;
                let x0 = (&x + &scaled).map_err(ccore_err)?;
                self.device.synchronize().map_err(ccore_err)?;
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
    use candle_gen::driver::conformance::{
        verify_driver_conformance, verify_multistream_driver_conformance, ConformanceCase,
        MultiStreamConformanceCase,
    };
    use candle_gen::gen_core::h3_image::{minimax_h3_image_pag_capability, H3ImageScheduler};
    use candle_gen::gen_core::h3_memory::window::{CudaWorkingSet, WindowAllocation};
    use candle_gen::gen_core::h3_memory::{NativeRows, NativeTaskAllocation, NativeTaskSelection};
    use candle_gen::gen_core::multistream::{StreamTransition, StreamTransitionKind};
    use candle_gen::gen_core::veda::workspace::Backend;
    use candle_gen::gen_core::{
        GenerationMemory, GenerationRequest, PagConfig, TransformerComponent, VedaBudget,
        VedaExecutionReport, VedaRequest,
    };
    use gen_core_h3::veda::workspace::for_request;
    use std::rc::Rc;

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

    fn eq_av(a: &CandleH3AvState, b: &CandleH3AvState) -> bool {
        eq_f32_tensors(&a.video, &b.video) && eq_f32_tensors(&a.audio, &b.audio)
    }

    struct MockJoint {
        scale: f64,
        report: Rc<RefCell<VedaExecutionReport>>,
        fallback_from_step: Option<usize>,
    }

    impl JointVelocity for MockJoint {
        fn forward(&mut self, step: &JointStep<'_>) -> Result<(Tensor, Tensor)> {
            let mut r = self.report.borrow_mut();
            if self.fallback_from_step.is_some_and(|s| step.index >= s) {
                r.dense_calls += 1;
                if r.fallback_reason.is_none() {
                    r.fallback_reason = Some("simulated dense fallback under pressure".into());
                }
            } else {
                r.sparse_calls += 1;
            }
            let k = self.scale + (step.index as f64) * 0.01;
            Ok(((step.video_rows * k)?, (step.audio_rows * k)?))
        }
    }

    #[test]
    fn candle_h3_joint_driver_passes_multistream_conformance_and_veda_fallback() {
        let dev = Device::Cpu;
        let sched = JointSchedule::with_shifts(9, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT).unwrap();
        let adaln = adaln_schedule(&sched).unwrap();
        let g = JointGeometry::new(124, 4, 6).unwrap();
        let layout = PackedLayout::build(
            g,
            [1, 2, 2],
            &[TEXT_TAG; 4],
            2,
            &[KeyframeAnchor::First, KeyframeAnchor::Last],
            &dev,
        )
        .unwrap();
        let v_rows = layout.video_indices().len();
        let a_rows = layout.audio_indices().len();
        let video_resolved =
            resolve_explicit_schedule("h3_video_12".into(), sched.video().sigmas()).unwrap();
        let audio_resolved =
            resolve_explicit_schedule("h3_audio_3".into(), sched.audio().sigmas()).unwrap();
        assert_ne!(video_resolved.sigmas(), audio_resolved.sigmas());

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
        let report_sink = Rc::new(RefCell::new(VedaExecutionReport {
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
                    let dev = dev.clone();
                    move || {
                        Box::new(
                            CandleH3JointDriver::owned(
                                "candle_h3_joint_test",
                                MockJoint {
                                    scale: 0.15,
                                    report: sink.clone(),
                                    fallback_from_step: Some(5),
                                },
                                layout.clone(),
                                sched.clone(),
                                adaln.clone(),
                                dev.clone(),
                            )
                            .unwrap(),
                        )
                    }
                },
                make_initial: {
                    let dev = dev.clone();
                    move || {
                        let vv: Vec<f32> = (0..(v_rows * 4))
                            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                            .collect();
                        let av: Vec<f32> = (0..(a_rows * 3))
                            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
                            .collect();
                        CandleH3AvState::new(
                            Tensor::from_vec(vv, (1, v_rows, 4), &dev).unwrap(),
                            Tensor::from_vec(av, (1, a_rows, 3), &dev).unwrap(),
                        )
                    }
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

    type HeadHistory = Rc<RefCell<Vec<Vec<(usize, f32)>>>>;

    struct MockImageVel {
        cond_val: f64,
        pert_val: f64,
        seen_heads: HeadHistory,
    }

    impl ImageVelocity for MockImageVel {
        fn forward(&mut self, _step: usize, image_rows: &Tensor) -> Result<Tensor> {
            Ok((image_rows * self.cond_val)?)
        }

        fn forward_heads(
            &mut self,
            step: usize,
            heads: &[(usize, f32)],
            image_rows: &Tensor,
        ) -> Result<Tensor> {
            self.seen_heads.borrow_mut().push(heads.to_vec());
            Ok((image_rows * (self.cond_val + step as f64 * 0.05))?)
        }

        fn forward_perturbed_heads(
            &mut self,
            step: usize,
            heads: &[(usize, f32)],
            image_rows: &Tensor,
            _perturbed_layers: &[u32],
        ) -> Result<Tensor> {
            self.seen_heads.borrow_mut().push(heads.to_vec());
            Ok((image_rows * (self.pert_val + step as f64 * 0.05))?)
        }
    }

    #[test]
    fn candle_h3_image_driver_passes_conformance_across_samplers_and_edit_pag() {
        let dev = Device::Cpu;
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
            ("euler_t2i", 0usize, H3ImageSampler::Euler, false),
            (
                "lcm_edit_pag_16ref",
                16,
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
            let condition = (refs > 0).then(|| {
                Tensor::from_vec(vec![0.999_f32; refs * 96], (1, refs, 96), &dev).unwrap()
            });
            let seen_heads = Rc::new(RefCell::new(Vec::new()));
            let rep = verify_multistream_driver_conformance(MultiStreamConformanceCase {
                base: ConformanceCase {
                    total_steps: 6,
                    split_at: 3,
                    seed: 77,
                    base_geometry: (2048, 2048),
                    alt_geometry: (1024, 1024),
                    make_driver: {
                        let sched = sched.clone();
                        let pag_opt = pag_opt.clone();
                        let condition = condition.clone();
                        let dev = dev.clone();
                        let seen_heads = seen_heads.clone();
                        move || {
                            Box::new(
                                CandleH3ImageDriver::new(
                                    "candle_h3_image_test",
                                    MockImageVel {
                                        cond_val: 0.4,
                                        pert_val: 0.2,
                                        seen_heads: seen_heads.clone(),
                                    },
                                    Cow::Owned(sched.clone()),
                                    sampler,
                                    pag_opt.clone(),
                                    77,
                                    condition.clone(),
                                    dev.clone(),
                                )
                                .unwrap(),
                            )
                        }
                    },
                    make_initial: {
                        let dev = dev.clone();
                        move || {
                            let vals: Vec<f32> = (0..(4 * 96))
                                .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
                                .collect();
                            Tensor::from_vec(vals, (1, 4, 96), &dev).unwrap()
                        }
                    },
                    latents_equal: eq_f32_tensors,
                },
                expected_stream_schedules: vec![(StreamId::Video, resolved.clone())],
                boundary_transition: None,
            })
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(rep.base.verified_inert_split, "{label}");
            assert!(rep.base.verified_cancellation, "{label}");
            assert!(
                !seen_heads.borrow().is_empty(),
                "{label}: pdd_head_mix must run"
            );
            if matches!(
                sampler,
                H3ImageSampler::Lcm { .. } | H3ImageSampler::EulerAncestral
            ) {
                assert_eq!(rep.base.unsplit_draws, 5, "{label}: 5 non-terminal draws");
                assert!(rep.base.verified_reset_preserves_draws, "{label}");
            } else {
                assert_eq!(rep.base.unsplit_draws, 0, "{label}");
            }

            let scalar = verify_driver_conformance(ConformanceCase {
                total_steps: 6,
                split_at: 2,
                seed: 19,
                base_geometry: (2048, 2048),
                alt_geometry: (1024, 1024),
                make_driver: {
                    let sched = sched.clone();
                    let condition = condition.clone();
                    let dev = dev.clone();
                    let seen_heads = seen_heads.clone();
                    move || {
                        Box::new(
                            CandleH3ImageDriver::new(
                                "candle_h3_image_scalar_test",
                                MockImageVel {
                                    cond_val: 0.35,
                                    pert_val: 0.15,
                                    seen_heads: seen_heads.clone(),
                                },
                                Cow::Owned(sched.clone()),
                                sampler,
                                None,
                                19,
                                condition.clone(),
                                dev.clone(),
                            )
                            .unwrap(),
                        )
                    }
                },
                make_initial: {
                    let dev = dev.clone();
                    move || Tensor::from_vec(vec![0.5_f32; 4 * 96], (1, 4, 96), &dev).unwrap()
                },
                latents_equal: eq_f32_tensors,
            })
            .unwrap();
            assert!(scalar.verified_inert_split);
        }
    }

    #[test]
    fn candle_preload_render_plan_validates_m5_envelope_and_cuda_window_fit_before_weight_load() {
        let mut req = GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width: 1344,
            height: 768,
            frames: Some(345),
            steps: Some(8),
            veda: Some(VedaRequest {
                enabled: true,
                generated: VedaBudget::SparsityPercent(70.0),
                ..Default::default()
            }),
            memory: Some(GenerationMemory {
                stage_residency: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::Both),
                ..Default::default()
            }),
            ..Default::default()
        };
        let m5 = render_video_plan_for_request(&req).unwrap();
        assert_eq!(m5.memory.domain, H3BudgetDomainKey::DeviceVram);
        assert!(m5.memory.fits_pool_budget);
        assert!(m5.memory.fits_total_envelope);

        let control = H3RenderPlan::from_video_generation_request(
            &req,
            H3BudgetDomainKey::DeviceVram,
            H3EnvelopeSpec::control_95gib(),
        )
        .unwrap();
        m5.assert_matched_policy(&control).unwrap();

        // Both streamed window sparse attention and dense fallback fit under the M5 24 GiB
        // envelope (`21 GiB` pool + `3 GiB` reserve) with `32 MiB` cuBLAS workspace.
        let profile = WindowAllocation::from_resident_tensors(
            &[
                ("token_refiner.proj.weight".into(), 64 * 1024 * 1024),
                (
                    "transformer_blocks.0.attn.to_q.weight".into(),
                    512 * 1024 * 1024,
                ),
                (
                    "transformer_blocks.0.adaln_proj.linear.weight".into(),
                    64 * 1024 * 1024,
                ),
                (
                    "transformer_blocks.1.attn.to_q.weight".into(),
                    512 * 1024 * 1024,
                ),
            ],
            &[],
            0,
            false,
        )
        .unwrap()
        .with_cuda_working_set(CudaWorkingSet {
            hidden: 5376,
            heads: 56,
            head_dim: 128,
            rotary_dim: 96,
            ffn_inner: 14336,
            element_bytes: 2,
            projection_decode_bytes: 256 * 1024 * 1024,
        })
        .unwrap();
        let task = NativeTaskAllocation {
            adapter: None,
            resident_bytes: 1_200_000_000,
            constructor_bytes: 0,
            patch_size: [1, 2, 2],
            reference_task: false,
            evicted_bytes: 0,
            table_bytes_per_timestep: 0,
            projections: Vec::new(),
            window: None,
        };
        let bound = for_request(&req, Backend::Cuda).unwrap().unwrap();
        let rows = NativeRows {
            task: NativeTaskSelection::Base,
            packed: 104_030,
            conditioning: 64,
            schedule: 18,
            visual_elements: 102 * 24 * 42 * 4,
            global: 1214,
        };
        let fit = m5
            .evaluate_cuda_window_fit(&profile, &task, rows, Some(&bound))
            .unwrap();
        assert!(fit.sparse_fits_pool);
        assert!(fit.dense_fallback_fits_pool);
        assert!(fit.fits_envelope);

        // Invalid streamed window component refused pre-load.
        req.memory.as_mut().unwrap().transformer_window_component = Some(TransformerComponent::Dit);
        assert!(render_video_plan_for_request(&req).is_err());

        // Image plan fits under DeviceVram domain.
        let img_req = GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width: 2048,
            height: 2048,
            ..Default::default()
        };
        let img_plan = render_image_plan_for_request_size(&img_req, 2048, 2048).unwrap();
        assert_eq!(img_plan.steps, 6);
    }
}
