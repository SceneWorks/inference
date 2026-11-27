//! `Pipeline.generate` for the distilled checkpoint (`src/flux3/pipeline.py` +
//! `sampling.denoise_flow_map`), the Candle twin of `mlx_gen_flux3::pipeline`: seeded
//! bf16-valued noise in the packed latent space → the flow-map Euler loop over
//! `timesteps(steps, alpha)` → the autoencoder decode → RGB8.
//!
//! The flow-map schedule, the target-time conditioning and the `dt · velocity` rounding are the
//! frozen source's, not FLUX.2's: the loop is written here against `gen_core_flux3` rather than
//! routed through the shared curated-sampler driver, whose sigma conventions do not express the
//! flow map's `(t, t_ref)` pair.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::{CandleError as Error, Result};
use gen_core_flux3::{latent_ids, scaled_image_ids, GenerationParams};
use rand::SeedableRng;

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::{Flux3Dit, SampleTokens};
use crate::nn::to;
use crate::references::ReferenceConditioning;
use crate::text_encoder::TextConditioning;

/// `torch.randn((h·w, C), generator=seed).to(bfloat16).float()`: the repo's launch-portable
/// seeded CPU normal ([`candle_gen::seed`]), rounded through bf16 and held as f32 (bf16 values,
/// f32 state across the Euler steps), moved to `device`. Not bit-compatible with torch's (or the
/// MLX twin's) RNG; parity is measured with injected noise.
pub fn noise(seed: u64, h: usize, w: usize, channels: usize, device: &Device) -> Result<Tensor> {
    candle_gen::execution::rng_scheme("candle-stdrng-box-muller-bf16-f32-v1");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let values = candle_gen::seed::seeded_normal_vec(&mut rng, h * w * channels);
    Ok(Tensor::from_vec(values, (h * w, channels), &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?
        .to_device(device)?)
}

fn apply_pag_candle(
    base: &Tensor,
    cond: &Tensor,
    pert: &Tensor,
    step_pag: f32,
    compute: DType,
) -> Result<Tensor> {
    let diff = to(
        &cond
            .to_dtype(DType::F32)?
            .sub(&pert.to_dtype(DType::F32)?)?,
        compute,
    )?;
    let scaled = to(
        &diff.to_dtype(DType::F32)?.affine(step_pag as f64, 0.0)?,
        compute,
    )?;
    to(
        &base
            .to_dtype(DType::F32)?
            .add(&scaled.to_dtype(DType::F32)?)?,
        compute,
    )
}

/// `denoise_flow_map(dit, packed, schedule)` for one text-to-image sample: from `noise`
/// (`[h·w, C]` f32) over `schedule` (the `steps + 1` points), `x ← x + dt · dit(x, t, t_ref)`
/// with `dt · velocity` multiplied at f32 and its PRODUCT rounded to the DiT's compute dtype
/// before the f32 add, as torch forms upstream's `packed.data["image"] + dt * dit(...)` (a
/// Python float times a bf16 tensor: f32 opmath, bf16 result; the state stays f32 because the
/// noise is `.float()`). `on_step(i)` fires after step `i` (1-based). Cancellation is checked
/// before every network evaluation and surfaces as [`Error::Canceled`] — the partial state is
/// dropped.
#[allow(clippy::too_many_arguments)]
pub fn denoise_flow_map(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_flow_map_with_references(dit, text, noise, image_ids, schedule, None, cancel, on_step)
}

/// Same frozen integrator with immutable AE conditioning shared by both CFG branches.
#[allow(clippy::too_many_arguments)]
pub fn denoise_flow_map_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    references: Option<&ReferenceConditioning>,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_flow_map_with_references_pag(
        dit, text, noise, image_ids, schedule, references, None, cancel, on_step,
    )
}

/// Forward-evaluation seam for [`Flux3FlowMapDriver`] (`SW-6968`, `PIPE-7`).
///
/// Generic over the predictor so weights-free Tier 0 tests can run
/// [`candle_gen::driver::conformance::verify_driver_conformance`] and verify FLUX.3's native
/// `(timestep, target_timestep)` + `dt`-rounded predictor semantics without loading weights.
pub trait Flux3FlowMapForward {
    /// Evaluate the DiT at `(timestep, target_timestep)` with optional PAG perturbation layers.
    fn forward_pag(
        &self,
        x: &Tensor,
        timestep: f32,
        target_timestep: Option<f32>,
        pag_layers: Option<&[u32]>,
    ) -> Result<Tensor>;
}

impl<F> Flux3FlowMapForward for F
where
    F: Fn(&Tensor, f32, Option<f32>, Option<&[u32]>) -> Result<Tensor>,
{
    fn forward_pag(
        &self,
        x: &Tensor,
        timestep: f32,
        target_timestep: Option<f32>,
        pag_layers: Option<&[u32]>,
    ) -> Result<Tensor> {
        self(x, timestep, target_timestep, pag_layers)
    }
}

pub(crate) fn flux3_driver_support(driver_id: &'static str) -> candle_gen::driver::DriverSupport {
    candle_gen::driver::DriverSupport {
        driver_id,
        arbitrary_slice: true,
        carry_history: true,
        reset_history: true,
        renoise_from_clean: false,
        geometry_change: false,
        read_only_boundary_estimate: false,
        previous_estimate_capture: false,
        uses_global_rng: false,
        refusal_reasons: candle_gen::driver::DriverRefusalReasons {
            renoise_from_clean:
                "FLUX.3 flow-map/base Euler driver does not re-noise from clean estimates",
            geometry_change:
                "FLUX.3 flow-map/base Euler driver does not support geometry-changing phase boundaries",
            read_only_boundary_estimate:
                "FLUX.3 flow-map/base Euler driver does not expose read-only boundary clean estimates",
            ..candle_gen::driver::DriverRefusalReasons::default()
        },
    }
}

/// Apply one FLUX.3 Euler update: `dt · velocity` runs at f32 opmath and only the product is
/// rounded to `compute`, then added into the f32 state `x` (`AP13 class (a)`).
fn flux3_euler_step_candle(
    x: &Tensor,
    velocity: &Tensor,
    dt: f32,
    compute: DType,
) -> Result<Tensor> {
    // torch: the Python-float `dt` is NOT rounded to bf16 first; the multiply runs at f32
    // opmath and only the product takes the velocity's dtype.
    let scaled = to(&to(velocity, DType::F32)?.affine(dt as f64, 0.0)?, compute)?;
    to(&x.add(&to(&scaled, DType::F32)?)?, DType::F32)
}

/// Typed [`candle_gen::driver::FamilyDriver`] for FLUX.3's native flow-map Euler integrator
/// (`SW-6968`, `PIPE-7`), preserving `(schedule[step], Some(schedule[step + 1]))`, optional PAG,
/// compute-dtype rounding on `dt · velocity`, and an `f32` running latent.
pub struct Flux3FlowMapDriver<F> {
    schedule: Vec<f32>,
    compute: DType,
    pag: Option<candle_gen::gen_core::PagResolved>,
    forward: F,
    support: candle_gen::driver::DriverSupport,
}

impl<F: Flux3FlowMapForward> Flux3FlowMapDriver<F> {
    pub fn new(
        driver_id: &'static str,
        schedule: &[f32],
        compute: DType,
        pag: Option<candle_gen::gen_core::PagResolved>,
        forward: F,
    ) -> Result<Self> {
        if schedule.len() < 2 {
            return Err(Error::Msg(format!(
                "flux3: a schedule needs at least two points, got {}",
                schedule.len()
            )));
        }
        Ok(Self {
            schedule: schedule.to_vec(),
            compute,
            pag,
            forward,
            support: flux3_driver_support(driver_id),
        })
    }

    pub fn total_steps(&self) -> usize {
        self.schedule.len() - 1
    }
}

impl<F: Flux3FlowMapForward> candle_gen::driver::FamilyDriver<Tensor> for Flux3FlowMapDriver<F> {
    fn support(&self) -> &candle_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: candle_gen::driver::SliceInput<'_, Tensor>,
        state: &mut candle_gen::driver::DriverRunState,
        hooks: &mut candle_gen::driver::StepHooks<'_, Tensor>,
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        let total_steps = self.total_steps();
        if input.total_steps != total_steps || input.span.end > total_steps {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with FLUX.3 flow-map schedule length {}",
                self.support.driver_id, input.span, input.total_steps, total_steps
            )));
        }
        state.enter_slice(&input)?;
        let init = match input.state {
            candle_gen::gen_core::BoundaryState::XSigma(x)
            | candle_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | candle_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let mut x = to(&init, DType::F32).map_err(candle_gen::gen_core::Error::from)?;

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            let t = self.schedule[step];
            let target = self.schedule[step + 1];
            let dt = target - t;
            let info = candle_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: candle_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: t,
                next_sigma: target,
            };
            hooks.observe(&info, &x)?;
            let cond = self
                .forward
                .forward_pag(&x, t, Some(target), None)
                .map_err(candle_gen::gen_core::Error::from)?;
            state.record_evaluation(candle_gen::driver::DriverEvaluationKind::Trajectory);
            let step_pag = self
                .pag
                .as_ref()
                .map_or(0.0, |p| p.step_scale(step, total_steps));
            let velocity = if step_pag > 0.0 {
                let layers = self
                    .pag
                    .as_ref()
                    .map_or(&[][..], |p| p.step_layers(step, total_steps));
                let pert = self
                    .forward
                    .forward_pag(&x, t, Some(target), Some(layers))
                    .map_err(candle_gen::gen_core::Error::from)?;
                state.record_evaluation(candle_gen::driver::DriverEvaluationKind::Trajectory);
                apply_pag_candle(&cond, &cond, &pert, step_pag, self.compute)
                    .map_err(candle_gen::gen_core::Error::from)?
            } else {
                cond
            };
            x = flux3_euler_step_candle(&x, &velocity, dt, self.compute)
                .map_err(candle_gen::gen_core::Error::from)?;
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(candle_gen::gen_core::BoundaryState::XSigma(x))
    }
}

/// Typed [`candle_gen::driver::FamilyDriver`] for FLUX.3's base CFG Euler loop
/// (`target_timestep: None`), sharing the same driver contract, `dt`-rounded Euler update,
/// cancellation, and progress reporting.
struct Flux3BaseDriver<F> {
    schedule: Vec<f32>,
    compute: DType,
    predict_velocity: F,
    support: candle_gen::driver::DriverSupport,
}

impl<F> Flux3BaseDriver<F>
where
    F: Fn(&Tensor, usize, usize, f32) -> Result<Tensor>,
{
    fn new(
        driver_id: &'static str,
        schedule: &[f32],
        compute: DType,
        predict_velocity: F,
    ) -> Result<Self> {
        if schedule.len() < 2 {
            return Err(Error::Msg("flux3: CFG schedule needs two points".into()));
        }
        Ok(Self {
            schedule: schedule.to_vec(),
            compute,
            predict_velocity,
            support: flux3_driver_support(driver_id),
        })
    }
}

impl<F> candle_gen::driver::FamilyDriver<Tensor> for Flux3BaseDriver<F>
where
    F: Fn(&Tensor, usize, usize, f32) -> Result<Tensor>,
{
    fn support(&self) -> &candle_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: candle_gen::driver::SliceInput<'_, Tensor>,
        state: &mut candle_gen::driver::DriverRunState,
        hooks: &mut candle_gen::driver::StepHooks<'_, Tensor>,
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        let total_steps = self.schedule.len() - 1;
        if input.total_steps != total_steps || input.span.end > total_steps {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with FLUX.3 CFG schedule length {}",
                self.support.driver_id, input.span, input.total_steps, total_steps
            )));
        }
        state.enter_slice(&input)?;
        let init = match input.state {
            candle_gen::gen_core::BoundaryState::XSigma(x)
            | candle_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | candle_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let mut x = to(&init, DType::F32).map_err(candle_gen::gen_core::Error::from)?;

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            let t = self.schedule[step];
            let target = self.schedule[step + 1];
            let dt = target - t;
            let info = candle_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: candle_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: t,
                next_sigma: target,
            };
            hooks.observe(&info, &x)?;
            let velocity = (self.predict_velocity)(&x, step, total_steps, t)
                .map_err(candle_gen::gen_core::Error::from)?;
            state.record_evaluation(candle_gen::driver::DriverEvaluationKind::Trajectory);
            x = flux3_euler_step_candle(&x, &velocity, dt, self.compute)
                .map_err(candle_gen::gen_core::Error::from)?;
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(candle_gen::gen_core::BoundaryState::XSigma(x))
    }
}

#[allow(clippy::too_many_arguments)]
fn denoise_flow_map_with_references_pag(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    references: Option<&ReferenceConditioning>,
    pag: Option<&candle_gen::gen_core::PagResolved>,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Tensor> {
    if !dit.is_flow_map() {
        return Err(Error::Unsupported(
            "flux3: denoise_flow_map needs a flow-map DiT (the base checkpoint samples with CFG, \
             sc-26239)"
                .into(),
        ));
    }
    let compute = dit.compute_dtype();
    let device = dit.device();
    let init = to(&noise.to_device(device)?, DType::F32)?;
    let text_tokens = text.tokens.to_device(device)?;
    let driver = Flux3FlowMapDriver::new(
        "flux3_flow_map_candle",
        schedule,
        compute,
        pag.cloned(),
        |x: &Tensor, t: f32, target: Option<f32>, pag_layers: Option<&[u32]>| {
            let sample = SampleTokens {
                text: &text_tokens,
                text_ids: &text.ids,
                image: x,
                image_ids,
                image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                timestep: t,
                target_timestep: target,
            };
            dit.forward_pag(std::slice::from_ref(&sample), pag_layers)
        },
    )?;
    let total_steps = driver.total_steps();
    let mut state = candle_gen::driver::DriverRunState::new();
    let mut on_progress = |progress: candle_gen::gen_core::Progress| {
        if let candle_gen::gen_core::Progress::Step { current, .. } = progress {
            on_step(current as usize);
        }
    };
    let mut hooks = candle_gen::driver::StepHooks::new(&mut on_progress);
    let out = candle_gen::driver::run_candle_driver_slices(
        &driver,
        [candle_gen::driver::SliceSpec::full(total_steps)],
        candle_gen::gen_core::BoundaryState::XSigma(init),
        total_steps,
        0,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    match out {
        candle_gen::gen_core::BoundaryState::XSigma(x)
        | candle_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | candle_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => Ok(x),
    }
}

/// Base checkpoint Euler CFG, preserving upstream's [unconditional, conditional] pack and
/// bf16 rounding after EACH velocity arithmetic operation, not just after the Euler product.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_cfg_with_references(
        dit, text, negative, noise, image_ids, schedule, guidance, None, cancel, on_step,
    )
}

/// Same frozen integrator with immutable AE conditioning shared by both CFG branches.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    references: Option<&ReferenceConditioning>,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_base_with_references_pag(
        dit,
        text,
        Some(negative),
        noise,
        image_ids,
        schedule,
        guidance,
        references,
        None,
        cancel,
        on_step,
    )
}

#[allow(clippy::too_many_arguments)]
fn denoise_base_with_references_pag(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: Option<&TextConditioning>,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    references: Option<&ReferenceConditioning>,
    pag: Option<&candle_gen::gen_core::PagResolved>,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Tensor> {
    if dit.is_flow_map() || !guidance.is_finite() || guidance < 0.0 {
        return Err(Error::Unsupported(
            "flux3: CFG needs a base DiT and finite non-negative guidance".into(),
        ));
    }
    let compute = dit.compute_dtype();
    let cfg_off_with_pag = gen_core_flux3::base_cfg_off_with_pag(guidance, pag);
    let init = noise.to_device(dit.device())?.to_dtype(DType::F32)?;
    let text_tokens = text.tokens.to_device(dit.device())?;
    let neg_tokens = negative
        .map(|n| n.tokens.to_device(dit.device()))
        .transpose()?;
    let driver = Flux3BaseDriver::new(
        "flux3_base_cfg_candle",
        schedule,
        compute,
        |x: &Tensor, step: usize, total_steps: usize, t: f32| {
            let step_pag = pag.map_or(0.0, |p| p.step_scale(step, total_steps));
            let layers = pag.map_or(&[][..], |p| p.step_layers(step, total_steps));
            if cfg_off_with_pag {
                let pos_sample = SampleTokens {
                    text: &text_tokens,
                    text_ids: &text.ids,
                    image: x,
                    image_ids,
                    image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                    timestep: t,
                    target_timestep: None,
                };
                let cond = dit.forward_pag(std::slice::from_ref(&pos_sample), None)?;
                if step_pag > 0.0 {
                    let pert = dit.forward_pag(std::slice::from_ref(&pos_sample), Some(layers))?;
                    apply_pag_candle(&cond, &cond, &pert, step_pag, compute)
                } else {
                    Ok(cond)
                }
            } else {
                let negative = negative.ok_or_else(|| {
                    Error::Unsupported(
                        "flux3: base CFG needs independently encoded negative conditioning".into(),
                    )
                })?;
                let neg_tokens = neg_tokens.as_ref().expect("encoded negative tokens");
                let samples = [
                    SampleTokens {
                        text: neg_tokens,
                        text_ids: &negative.ids,
                        image: x,
                        image_ids,
                        image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                        timestep: t,
                        target_timestep: None,
                    },
                    SampleTokens {
                        text: &text_tokens,
                        text_ids: &text.ids,
                        image: x,
                        image_ids,
                        image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                        timestep: t,
                        target_timestep: None,
                    },
                ];
                let prediction = dit.forward(&samples)?;
                let rows = x.dim(0)?;
                let u = prediction.narrow(0, 0, rows)?;
                let c = prediction.narrow(0, rows, rows)?;
                let difference = to(
                    &c.to_dtype(DType::F32)?.sub(&u.to_dtype(DType::F32)?)?,
                    compute,
                )?;
                let scaled = to(
                    &difference
                        .to_dtype(DType::F32)?
                        .affine(guidance as f64, 0.0)?,
                    compute,
                )?;
                let pred_cfg = to(
                    &u.to_dtype(DType::F32)?.add(&scaled.to_dtype(DType::F32)?)?,
                    compute,
                )?;
                if step_pag > 0.0 {
                    let pos_sample = SampleTokens {
                        text: &text_tokens,
                        text_ids: &text.ids,
                        image: x,
                        image_ids,
                        image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                        timestep: t,
                        target_timestep: None,
                    };
                    let pert = dit.forward_pag(std::slice::from_ref(&pos_sample), Some(layers))?;
                    apply_pag_candle(&pred_cfg, &c, &pert, step_pag, compute)
                } else {
                    Ok(pred_cfg)
                }
            }
        },
    )?;
    let total_steps = schedule.len() - 1;
    let mut state = candle_gen::driver::DriverRunState::new();
    let mut on_progress = |progress: candle_gen::gen_core::Progress| {
        if let candle_gen::gen_core::Progress::Step { current, .. } = progress {
            on_step(current as usize);
        }
    };
    let mut hooks = candle_gen::driver::StepHooks::new(&mut on_progress);
    let out = candle_gen::driver::run_candle_driver_slices(
        &driver,
        [candle_gen::driver::SliceSpec::full(total_steps)],
        candle_gen::gen_core::BoundaryState::XSigma(init),
        total_steps,
        0,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    match out {
        candle_gen::gen_core::BoundaryState::XSigma(x)
        | candle_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | candle_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => Ok(x),
    }
}

/// The image ids of a `(h, w)` token grid, scaled for the DiT (`scaled_ids(latent_ids(h, w))`).
pub fn image_ids(h: usize, w: usize, image_rope_scale: f32) -> Vec<[f32; 4]> {
    scaled_image_ids(&latent_ids(h, w), image_rope_scale)
}

/// The packed tokens of one image: noise from `seed` → [`denoise_flow_map`] over the request's
/// schedule. `on_step(i)` fires per Euler step.
pub fn denoise_one(
    dit: &Flux3Dit,
    text: &TextConditioning,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_one_conditioned(dit, text, None, params, seed, cancel, on_step)
}

/// Render tokens with an independently encoded negative caption for CFG.
pub fn denoise_one_conditioned(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: Option<&TextConditioning>,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_one_with_references(dit, text, negative, None, params, seed, cancel, on_step)
}

/// Generate fresh noise, while reference tokens remain pinned image_cond (not img2img).
#[allow(clippy::too_many_arguments)]
pub fn denoise_one_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: Option<&TextConditioning>,
    references: Option<&ReferenceConditioning>,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    let (h, w) = params.latent_shape()?;
    let channels = dit.config().image_in_channels;
    let ids = image_ids(h, w, dit.config().image_rope_scale as f32);
    let z = noise(seed, h, w, channels, dit.device())?;
    let schedule = params.schedule()?;
    let output = match params.guidance {
        Some(guidance) => denoise_base_with_references_pag(
            dit,
            text,
            negative,
            &z,
            &ids,
            &schedule,
            guidance,
            references,
            params.pag.as_ref(),
            cancel,
            on_step,
        ),
        None => denoise_flow_map_with_references_pag(
            dit,
            text,
            &z,
            &ids,
            &schedule,
            references,
            params.pag.as_ref(),
            cancel,
            on_step,
        ),
    }?;
    candle_gen::execution::schedule(
        if params.guidance.is_some() {
            "flux3-euler-cfg-v1"
        } else {
            "flux3-flow-map-v1"
        },
        &schedule,
        "native-timestep-and-target",
    );
    candle_gen::execution::runtime(dit.device(), dit.compute_dtype());
    candle_gen::execution::policy(
        "flux3.compute_dtype",
        format!("{:?}", dit.compute_dtype()).into(),
    );
    candle_gen::execution::policy(
        "flux3.reference_tokens",
        serde_json::json!(references.is_some()),
    );
    candle_gen::execution::policy(
        "flux3.pag",
        serde_json::to_value(&params.pag).unwrap_or(serde_json::Value::Null),
    );
    if let Some(guidance) = params.guidance {
        candle_gen::execution::f32_policy("flux3.guidance", guidance);
    }
    Ok(output)
}

/// Render one image: [`denoise_one`] → decode → RGB8 (the caller reports `Decoding`).
pub fn generate_one(
    dit: &Flux3Dit,
    ae: &Flux3AutoEncoder,
    text: &TextConditioning,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Image> {
    let (h, w) = params.latent_shape()?;
    if ae.latent_channels() != dit.config().image_in_channels {
        return Err(Error::Msg(format!(
            "flux3: the autoencoder packs {} channels per token, the DiT reads {}",
            ae.latent_channels(),
            dit.config().image_in_channels
        )));
    }
    let tokens = denoise_one(dit, text, params, seed, cancel, on_step)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    decode_tokens(ae, &tokens, h, w)
}

/// `to_pil(autoencoder.decode(tokens, ids))`: decode the packed tokens and convert the `[−1, 1]`
/// float image to RGB8 (`clamp(−1, 1)`, `(x + 1) · 127.5`, truncated to u8 — see
/// [`float_image_to_rgb8`]).
pub fn decode_tokens(ae: &Flux3AutoEncoder, tokens: &Tensor, h: usize, w: usize) -> Result<Image> {
    candle_gen::execution::component_runtime("autoencoder", tokens.device(), DType::F32);
    let image = ae.decode(tokens, h, w)?;
    float_image_to_rgb8(&image)
}

/// `to_pil`: `((image.clamp(−1, 1) + 1) · 127.5).to(uint8)` — torch's float→uint8 cast
/// **truncates**, so this is `floor`, not the nearest-even rounding of the shared
/// `candle_gen::round_rgb8`. `image` is `[1, 3, H, W]` f32 (NCHW).
pub fn float_image_to_rgb8(image: &Tensor) -> Result<Image> {
    let dims = image.dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(Error::Msg(format!(
            "flux3: a decoded image must be [1, 3, H, W], got {dims:?}"
        )));
    }
    let (height, width) = (dims[2] as u32, dims[3] as u32);
    let scaled = image
        .to_dtype(DType::F32)?
        .clamp(-1f32, 1f32)?
        .affine(127.5, 127.5)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .contiguous()?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let pixels: Vec<u8> = scaled.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    Ok(Image {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The noise holds bf16 values in an f32 tensor and depends only on the seed.
    #[test]
    fn noise_is_bf16_valued_f32_and_seeded() {
        let dev = Device::Cpu;
        let a = noise(7, 2, 3, 4, &dev).unwrap();
        let b = noise(7, 2, 3, 4, &dev).unwrap();
        let c = noise(8, 2, 3, 4, &dev).unwrap();
        assert_eq!(a.dims(), &[6, 4]);
        assert_eq!(a.dtype(), DType::F32);
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert_ne!(av, c.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        for v in &av {
            // A bf16 value has at most 8 significant bits: the low 16 bits of its f32 are zero.
            assert_eq!(v.to_bits() & 0xffff, 0, "{v} is not bf16-representable");
        }
        assert!(
            av.iter().any(|v| v.fract() != 0.0),
            "a normal draw, not integers"
        );
    }

    /// `to_pil` truncates (torch's float → uint8 cast), it does not round.
    #[test]
    fn rgb8_conversion_truncates_and_clamps() {
        let dev = Device::Cpu;
        let image = Tensor::new(
            &[
                -2.0f32, -1.0, -0.5, 0.0, 0.999, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &dev,
        )
        .unwrap()
        .reshape((1, 3, 2, 2))
        .unwrap();
        let rgb = float_image_to_rgb8(&image).unwrap();
        assert_eq!((rgb.width, rgb.height), (2, 2));
        // Pixel (y, x) channel c = image[c][y][x]; channel 0 of the four pixels:
        // -2 → clamp -1 → 0; -1 → 0; -0.5 → 63.75 → 63; 0 → 127.5 → 127.
        assert_eq!(rgb.pixels[0], 0);
        assert_eq!(rgb.pixels[3], 0);
        assert_eq!(rgb.pixels[6], 63);
        assert_eq!(rgb.pixels[9], 127);
        // channel 1: 0.999 → 254.87 → 254; 1 → 255; 1.5 → 255; 0 → 127.
        assert_eq!(rgb.pixels[1], 254);
        assert_eq!(rgb.pixels[4], 255);
        assert_eq!(rgb.pixels[7], 255);
        assert_eq!(rgb.pixels[10], 127);
        assert!(float_image_to_rgb8(&image.reshape((1, 2, 2, 3)).unwrap()).is_err());
    }

    #[test]
    fn base_cfg_off_shortcut_only_skips_negative_at_exact_guidance_one() {
        let cap = gen_core_flux3::flux3_pag_capability(false);
        let resolved = cap
            .resolve(Some(&candle_gen::gen_core::PagConfig {
                scale: 1.5,
                ..Default::default()
            }))
            .unwrap();
        for (guidance, expect_neg, expect_cfg_off) in [
            (0.0_f32, true, false),
            (0.5, true, false),
            (1.0, false, true),
            (3.5, true, false),
        ] {
            assert_eq!(
                gen_core_flux3::base_requires_negative_prompt(Some(guidance), Some(&resolved)),
                expect_neg,
                "guidance={guidance}"
            );
            assert_eq!(
                gen_core_flux3::base_cfg_off_with_pag(guidance, Some(&resolved)),
                expect_cfg_off,
                "guidance={guidance}"
            );
            assert!(
                gen_core_flux3::base_requires_negative_prompt(Some(guidance), None),
                "without PAG, base always encodes negative at guidance={guidance}"
            );
            assert!(!gen_core_flux3::base_cfg_off_with_pag(guidance, None));
        }
    }

    fn tensor_bit_equal(a: &Tensor, b: &Tensor) -> bool {
        let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn flux3_flow_map_driver_passes_conformance() {
        use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let schedule = gen_core_flux3::flow_map_schedule(4, gen_core_flux3::DEFAULT_ALPHA).unwrap();
        let sched_clone = schedule.clone();
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 19,
            base_geometry: (64, 64),
            alt_geometry: (128, 128),
            make_driver: move || {
                Box::new(
                    Flux3FlowMapDriver::new(
                        "flux3_flow_map_candle_test",
                        &sched_clone,
                        DType::BF16,
                        None,
                        |x: &Tensor, t: f32, target: Option<f32>, _pag: Option<&[u32]>| {
                            let target = target.expect("flow-map must pass Some(target_timestep)");
                            let coeff = 0.25 + 0.5 * f64::from(t) - 0.1 * f64::from(target);
                            to(&x.affine(coeff, 0.05)?, DType::BF16)
                        },
                    )
                    .unwrap(),
                )
            },
            make_initial: || {
                Tensor::new(&[0.5_f32, -0.75, 1.25, -0.25], &Device::Cpu)
                    .unwrap()
                    .reshape((2, 2))
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
            },
            latents_equal: tensor_bit_equal,
        })
        .expect("FLUX.3 Candle flow-map driver passes Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_reset_preserves_draws);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn flux3_flow_map_driver_preserves_target_timesteps_rounding_and_pag() {
        use candle_gen::driver::{DriverRunState, FamilyDriver, SliceInput, StepHooks};
        use candle_gen::gen_core::{BoundaryState, PagConfig, Progress};

        // Schedule with < 2 points is rejected.
        assert!(Flux3FlowMapDriver::new(
            "flux3_short",
            &[1.0],
            DType::BF16,
            None,
            |x: &Tensor, _: f32, _: Option<f32>, _: Option<&[u32]>| Ok(x.clone()),
        )
        .is_err());

        let schedule = gen_core_flux3::flow_map_schedule(4, gen_core_flux3::DEFAULT_ALPHA).unwrap();
        let pag = gen_core_flux3::flux3_pag_capability(true)
            .resolve(Some(&PagConfig {
                scale: 1.2,
                ..Default::default()
            }))
            .unwrap();
        let seen_calls =
            std::cell::RefCell::new(Vec::<(f32, Option<f32>, Option<Vec<u32>>)>::new());
        let driver = Flux3FlowMapDriver::new(
            "flux3_flow_map_candle_pag_test",
            &schedule,
            DType::BF16,
            Some(pag.clone()),
            |x: &Tensor, t: f32, target: Option<f32>, layers: Option<&[u32]>| {
                assert_eq!(x.dtype(), DType::F32, "running latent x must stay F32");
                seen_calls
                    .borrow_mut()
                    .push((t, target, layers.map(|l| l.to_vec())));
                let factor = if layers.is_some() { 0.15 } else { 0.45 };
                to(
                    &x.affine(
                        factor + 0.1 * f64::from(t),
                        f64::from(target.unwrap_or(0.0)) * 0.2,
                    )?,
                    DType::BF16,
                )
            },
        )
        .unwrap();

        let init = Tensor::new(&[0.5_f32, -0.75, 1.25, -0.25], &Device::Cpu)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        let cancel = CancelFlag::default();
        let mut state = DriverRunState::new();
        let mut ticks = Vec::new();
        let out = {
            let mut on_progress = |p: Progress| ticks.push(p);
            let mut hooks = StepHooks::new(&mut on_progress);
            driver
                .run_slice(
                    SliceInput::full_trajectory(init.clone(), 4, 7, &cancel),
                    &mut state,
                    &mut hooks,
                )
                .unwrap()
        };
        let BoundaryState::XSigma(driver_x) = out else {
            panic!("expected XSigma");
        };
        assert_eq!(driver_x.dtype(), DType::F32);

        // Reference calculation matching the frozen upstream formula bit-for-bit.
        let mut ref_x = to(&init, DType::F32).unwrap();
        let mut expected_calls = Vec::new();
        for step in 0..4 {
            let t = schedule[step];
            let target = schedule[step + 1];
            let dt = target - t;
            expected_calls.push((t, Some(target), None));
            let cond = to(
                &ref_x
                    .affine(0.45 + 0.1 * f64::from(t), f64::from(target) * 0.2)
                    .unwrap(),
                DType::BF16,
            )
            .unwrap();
            let step_pag = pag.step_scale(step, 4);
            let velocity = if step_pag > 0.0 {
                let layers = pag.step_layers(step, 4);
                expected_calls.push((t, Some(target), Some(layers.to_vec())));
                let pert = to(
                    &ref_x
                        .affine(0.15 + 0.1 * f64::from(t), f64::from(target) * 0.2)
                        .unwrap(),
                    DType::BF16,
                )
                .unwrap();
                apply_pag_candle(&cond, &cond, &pert, step_pag, DType::BF16).unwrap()
            } else {
                cond
            };
            let scaled = to(
                &to(&velocity, DType::F32)
                    .unwrap()
                    .affine(dt as f64, 0.0)
                    .unwrap(),
                DType::BF16,
            )
            .unwrap();
            ref_x = to(
                &ref_x.add(&to(&scaled, DType::F32).unwrap()).unwrap(),
                DType::F32,
            )
            .unwrap();
        }
        assert!(tensor_bit_equal(&driver_x, &ref_x));
        assert_eq!(*seen_calls.borrow(), expected_calls);
        assert_eq!(
            ticks,
            vec![
                Progress::Step {
                    current: 1,
                    total: 4
                },
                Progress::Step {
                    current: 2,
                    total: 4
                },
                Progress::Step {
                    current: 3,
                    total: 4
                },
                Progress::Step {
                    current: 4,
                    total: 4
                },
            ]
        );
    }

    #[test]
    fn flux3_base_driver_passes_conformance() {
        use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let schedule = gen_core_flux3::flow_map_schedule(4, gen_core_flux3::DEFAULT_ALPHA).unwrap();
        let sched_clone = schedule.clone();
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 23,
            base_geometry: (64, 64),
            alt_geometry: (128, 128),
            make_driver: move || {
                Box::new(
                    Flux3BaseDriver::new(
                        "flux3_base_cfg_candle_test",
                        &sched_clone,
                        DType::BF16,
                        |x: &Tensor, _step: usize, _total: usize, t: f32| {
                            to(&x.affine(0.3 + 0.2 * f64::from(t), 0.0)?, DType::BF16)
                        },
                    )
                    .unwrap(),
                )
            },
            make_initial: || {
                Tensor::new(&[0.25_f32, -0.5, 0.75, -1.0], &Device::Cpu)
                    .unwrap()
                    .reshape((2, 2))
                    .unwrap()
            },
            latents_equal: tensor_bit_equal,
        })
        .expect("FLUX.3 Candle base CFG driver passes Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
    }
}
