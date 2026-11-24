//! `Pipeline.generate` for the distilled checkpoint (`src/flux3/pipeline.py` +
//! `sampling.denoise_flow_map`): seeded bf16-valued noise in the packed latent space → the
//! flow-map Euler loop over `timesteps(steps, alpha)` → the autoencoder decode → RGB8.

use gen_core_flux3::{latent_ids, scaled_image_ids, GenerationParams};
use mlx_gen::{CancelFlag, Error, Image, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::{Flux3Dit, SampleTokens};
use crate::references::ReferenceConditioning;
use crate::text_encoder::TextConditioning;

/// `torch.randn((h·w, C), generator=seed).to(bfloat16).float()`: the repo's seeded MLX normal,
/// rounded through bf16 and held as f32 (bf16 values, f32 state across the Euler steps). Not
/// bit-compatible with torch's RNG; parity is measured with injected noise.
pub fn noise(seed: u64, h: usize, w: usize, channels: usize) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [(h * w) as i32, channels as i32];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?
        .as_dtype(Dtype::Bfloat16)?
        .as_dtype(Dtype::Float32)?)
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
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    denoise_flow_map_with_references(dit, text, noise, image_ids, schedule, None, cancel, on_step)
}

fn apply_pag_mlx(
    base: &Array,
    cond: &Array,
    pert: &Array,
    step_pag: f32,
    compute: Dtype,
) -> Result<Array> {
    let diff = cond
        .as_dtype(Dtype::Float32)?
        .subtract(&pert.as_dtype(Dtype::Float32)?)?
        .as_dtype(compute)?;
    let scaled = diff
        .as_dtype(Dtype::Float32)?
        .multiply(Array::from_f32(step_pag))?
        .as_dtype(compute)?;
    Ok(base
        .as_dtype(Dtype::Float32)?
        .add(&scaled.as_dtype(Dtype::Float32)?)?
        .as_dtype(compute)?)
}

/// Forward-evaluation seam for [`Flux3FlowMapDriver`] (`SW-6968`, `PIPE-7`).
///
/// Generic over the predictor so weights-free Tier 0 tests can run
/// [`mlx_gen::driver::conformance::verify_driver_conformance`] and verify FLUX.3's native
/// `(timestep, target_timestep)` + `dt`-rounded predictor semantics without loading weights.
pub trait Flux3FlowMapForward {
    /// Evaluate the DiT at `(timestep, target_timestep)` with optional PAG perturbation layers.
    fn forward_pag(
        &self,
        x: &Array,
        timestep: f32,
        target_timestep: Option<f32>,
        pag_layers: Option<&[u32]>,
    ) -> Result<Array>;
}

impl<F> Flux3FlowMapForward for F
where
    F: Fn(&Array, f32, Option<f32>, Option<&[u32]>) -> Result<Array>,
{
    fn forward_pag(
        &self,
        x: &Array,
        timestep: f32,
        target_timestep: Option<f32>,
        pag_layers: Option<&[u32]>,
    ) -> Result<Array> {
        self(x, timestep, target_timestep, pag_layers)
    }
}

fn flux3_driver_support(driver_id: &'static str) -> mlx_gen::driver::DriverSupport {
    mlx_gen::driver::DriverSupport {
        driver_id,
        arbitrary_slice: true,
        carry_history: true,
        reset_history: true,
        renoise_from_clean: false,
        geometry_change: false,
        read_only_boundary_estimate: false,
        previous_estimate_capture: false,
        uses_global_rng: false,
        refusal_reasons: mlx_gen::driver::DriverRefusalReasons {
            renoise_from_clean:
                "FLUX.3 flow-map/base Euler driver does not re-noise from clean estimates",
            geometry_change:
                "FLUX.3 flow-map/base Euler driver does not support geometry-changing phase boundaries",
            read_only_boundary_estimate:
                "FLUX.3 flow-map/base Euler driver does not expose read-only boundary clean estimates",
            ..mlx_gen::driver::DriverRefusalReasons::default()
        },
    }
}

/// Apply one FLUX.3 Euler update: `dt · velocity` runs at f32 opmath and only the product is
/// rounded to `compute`, then added into the f32 state `x` (`AP13 class (a)`).
fn flux3_euler_step_mlx(x: &Array, velocity: &Array, dt: f32, compute: Dtype) -> Result<Array> {
    // torch: the Python-float `dt` is NOT rounded to bf16 first; the multiply runs at f32
    // opmath and only the product takes the velocity's dtype.
    let scaled = velocity
        .as_dtype(Dtype::Float32)?
        .multiply(Array::from_f32(dt))?
        .as_dtype(compute)?;
    Ok(x.add(&scaled)?.as_dtype(Dtype::Float32)?)
}

/// Typed [`mlx_gen::driver::FamilyDriver`] for FLUX.3's native flow-map Euler integrator
/// (`SW-6968`, `PIPE-7`), preserving `(schedule[step], Some(schedule[step + 1]))`, optional PAG,
/// compute-dtype rounding on `dt · velocity`, and an `f32` running latent.
pub struct Flux3FlowMapDriver<F> {
    schedule: Vec<f32>,
    compute: Dtype,
    pag: Option<mlx_gen::gen_core::PagResolved>,
    forward: F,
    support: mlx_gen::driver::DriverSupport,
}

impl<F: Flux3FlowMapForward> Flux3FlowMapDriver<F> {
    pub fn new(
        driver_id: &'static str,
        schedule: &[f32],
        compute: Dtype,
        pag: Option<mlx_gen::gen_core::PagResolved>,
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

impl<F: Flux3FlowMapForward> mlx_gen::driver::FamilyDriver<Array> for Flux3FlowMapDriver<F> {
    fn support(&self) -> &mlx_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: mlx_gen::driver::SliceInput<'_, Array>,
        state: &mut mlx_gen::driver::DriverRunState,
        hooks: &mut mlx_gen::driver::StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        let total_steps = self.total_steps();
        if input.total_steps != total_steps || input.span.end > total_steps {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with FLUX.3 flow-map schedule length {}",
                self.support.driver_id, input.span, input.total_steps, total_steps
            )));
        }
        state.enter_slice(&input)?;
        let init = match input.state {
            mlx_gen::gen_core::BoundaryState::XSigma(x)
            | mlx_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | mlx_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let mut x = init
            .as_dtype(Dtype::Float32)
            .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let t = self.schedule[step];
            let target = self.schedule[step + 1];
            let dt = target - t;
            let info = mlx_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: mlx_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: t,
                next_sigma: target,
            };
            hooks.observe(&info, &x)?;
            let cond = self
                .forward
                .forward_pag(&x, t, Some(target), None)
                .map_err(mlx_gen::gen_core::Error::from)?;
            state.record_evaluation(mlx_gen::driver::DriverEvaluationKind::Trajectory);
            let step_pag = self
                .pag
                .as_ref()
                .map_or(0.0, |p| p.step_scale(step, total_steps));
            let velocity = if step_pag > 0.0 {
                eval([&cond]).map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
                let layers = self
                    .pag
                    .as_ref()
                    .map_or(&[][..], |p| p.step_layers(step, total_steps));
                let pert = self
                    .forward
                    .forward_pag(&x, t, Some(target), Some(layers))
                    .map_err(mlx_gen::gen_core::Error::from)?;
                state.record_evaluation(mlx_gen::driver::DriverEvaluationKind::Trajectory);
                apply_pag_mlx(&cond, &cond, &pert, step_pag, self.compute)
                    .map_err(mlx_gen::gen_core::Error::from)?
            } else {
                cond
            };
            x = flux3_euler_step_mlx(&x, &velocity, dt, self.compute)
                .map_err(mlx_gen::gen_core::Error::from)?;
            x = hooks.post_step(&info, x)?;
            eval([&x]).map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            hooks.materialize(&info, &x)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(mlx_gen::gen_core::BoundaryState::XSigma(x))
    }
}

/// Typed [`mlx_gen::driver::FamilyDriver`] for FLUX.3's base CFG Euler loop (`target_timestep: None`),
/// sharing the same driver contract, `dt`-rounded Euler update, cancellation, and progress reporting.
struct Flux3BaseDriver<F> {
    schedule: Vec<f32>,
    compute: Dtype,
    predict_velocity: F,
    support: mlx_gen::driver::DriverSupport,
}

impl<F> Flux3BaseDriver<F>
where
    F: Fn(&Array, usize, usize, f32) -> Result<Array>,
{
    fn new(
        driver_id: &'static str,
        schedule: &[f32],
        compute: Dtype,
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

impl<F> mlx_gen::driver::FamilyDriver<Array> for Flux3BaseDriver<F>
where
    F: Fn(&Array, usize, usize, f32) -> Result<Array>,
{
    fn support(&self) -> &mlx_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: mlx_gen::driver::SliceInput<'_, Array>,
        state: &mut mlx_gen::driver::DriverRunState,
        hooks: &mut mlx_gen::driver::StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        let total_steps = self.schedule.len() - 1;
        if input.total_steps != total_steps || input.span.end > total_steps {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with FLUX.3 CFG schedule length {}",
                self.support.driver_id, input.span, input.total_steps, total_steps
            )));
        }
        state.enter_slice(&input)?;
        let init = match input.state {
            mlx_gen::gen_core::BoundaryState::XSigma(x)
            | mlx_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | mlx_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let mut x = init
            .as_dtype(Dtype::Float32)
            .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let t = self.schedule[step];
            let target = self.schedule[step + 1];
            let dt = target - t;
            let info = mlx_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: mlx_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: t,
                next_sigma: target,
            };
            hooks.observe(&info, &x)?;
            let velocity = (self.predict_velocity)(&x, step, total_steps, t)
                .map_err(mlx_gen::gen_core::Error::from)?;
            state.record_evaluation(mlx_gen::driver::DriverEvaluationKind::Trajectory);
            x = flux3_euler_step_mlx(&x, &velocity, dt, self.compute)
                .map_err(mlx_gen::gen_core::Error::from)?;
            x = hooks.post_step(&info, x)?;
            eval([&x]).map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            hooks.materialize(&info, &x)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(mlx_gen::gen_core::BoundaryState::XSigma(x))
    }
}

#[allow(clippy::too_many_arguments)]
fn denoise_flow_map_with_references_pag(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    references: Option<&ReferenceConditioning>,
    pag: Option<&mlx_gen::gen_core::PagResolved>,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Array> {
    if !dit.is_flow_map() {
        return Err(Error::Unsupported(
            "flux3: denoise_flow_map needs a flow-map DiT (the base checkpoint samples with CFG, \
             sc-26239)"
                .into(),
        ));
    }
    let compute = dit.compute_dtype();
    let driver = Flux3FlowMapDriver::new(
        "flux3_flow_map_mlx",
        schedule,
        compute,
        pag.cloned(),
        |x: &Array, t: f32, target: Option<f32>, pag_layers: Option<&[u32]>| {
            let sample = SampleTokens {
                text: &text.tokens,
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
    let mut state = mlx_gen::driver::DriverRunState::new();
    let mut on_progress = |progress: mlx_gen::Progress| {
        if let mlx_gen::Progress::Step { current, .. } = progress {
            on_step(current as usize);
        }
    };
    let mut hooks = mlx_gen::driver::StepHooks::new(&mut on_progress);
    let out = mlx_gen::driver::run_mlx_driver_slices(
        &driver,
        [mlx_gen::driver::SliceSpec::full(total_steps)],
        mlx_gen::gen_core::BoundaryState::XSigma(noise.clone()),
        total_steps,
        0,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    match out {
        mlx_gen::gen_core::BoundaryState::XSigma(x)
        | mlx_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | mlx_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => Ok(x),
    }
}

/// Same frozen integrator with immutable AE conditioning shared by both CFG branches.
#[allow(clippy::too_many_arguments)]
pub fn denoise_flow_map_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    references: Option<&ReferenceConditioning>,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    denoise_flow_map_with_references_pag(
        dit, text, noise, image_ids, schedule, references, None, cancel, on_step,
    )
}

/// Base Euler CFG with independent unconditional/conditional samples in that order.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
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
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    references: Option<&ReferenceConditioning>,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
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
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    references: Option<&ReferenceConditioning>,
    pag: Option<&mlx_gen::gen_core::PagResolved>,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Array> {
    if dit.is_flow_map() || !guidance.is_finite() || guidance < 0.0 {
        return Err(Error::Unsupported(
            "flux3: CFG needs a base DiT and finite non-negative guidance".into(),
        ));
    }
    let compute = dit.compute_dtype();
    let cfg_off_with_pag = gen_core_flux3::base_cfg_off_with_pag(guidance, pag);
    let driver = Flux3BaseDriver::new(
        "flux3_base_cfg_mlx",
        schedule,
        compute,
        |x: &Array, step: usize, total_steps: usize, t: f32| {
            let step_pag = pag.map_or(0.0, |p| p.step_scale(step, total_steps));
            let layers = pag.map_or(&[][..], |p| p.step_layers(step, total_steps));
            if cfg_off_with_pag {
                let pos_sample = SampleTokens {
                    text: &text.tokens,
                    text_ids: &text.ids,
                    image: x,
                    image_ids,
                    image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                    timestep: t,
                    target_timestep: None,
                };
                let cond = dit.forward_pag(std::slice::from_ref(&pos_sample), None)?;
                if step_pag > 0.0 {
                    eval([&cond])?;
                    let pert = dit.forward_pag(std::slice::from_ref(&pos_sample), Some(layers))?;
                    apply_pag_mlx(&cond, &cond, &pert, step_pag, compute)
                } else {
                    Ok(cond)
                }
            } else {
                let negative = negative.ok_or_else(|| {
                    Error::Unsupported(
                        "flux3: base CFG needs independently encoded negative conditioning".into(),
                    )
                })?;
                let samples = [negative, text].map(|conditioning| SampleTokens {
                    text: &conditioning.tokens,
                    text_ids: &conditioning.ids,
                    image: x,
                    image_ids,
                    image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                    timestep: t,
                    target_timestep: None,
                });
                let prediction = dit.forward(&samples)?;
                let rows = prediction.split(2, 0)?;
                let difference = rows[1].subtract(&rows[0])?.as_dtype(compute)?;
                let scaled = difference
                    .as_dtype(Dtype::Float32)?
                    .multiply(Array::from_f32(guidance))?
                    .as_dtype(compute)?;
                let pred_cfg = rows[0].add(&scaled)?.as_dtype(compute)?;
                if step_pag > 0.0 {
                    let cond = rows[1].clone();
                    eval([&pred_cfg, &cond])?;
                    let pos_sample = SampleTokens {
                        text: &text.tokens,
                        text_ids: &text.ids,
                        image: x,
                        image_ids,
                        image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                        timestep: t,
                        target_timestep: None,
                    };
                    let pert = dit.forward_pag(std::slice::from_ref(&pos_sample), Some(layers))?;
                    apply_pag_mlx(&pred_cfg, &cond, &pert, step_pag, compute)
                } else {
                    Ok(pred_cfg)
                }
            }
        },
    )?;
    let total_steps = schedule.len() - 1;
    let mut state = mlx_gen::driver::DriverRunState::new();
    let mut on_progress = |progress: mlx_gen::Progress| {
        if let mlx_gen::Progress::Step { current, .. } = progress {
            on_step(current as usize);
        }
    };
    let mut hooks = mlx_gen::driver::StepHooks::new(&mut on_progress);
    let out = mlx_gen::driver::run_mlx_driver_slices(
        &driver,
        [mlx_gen::driver::SliceSpec::full(total_steps)],
        mlx_gen::gen_core::BoundaryState::XSigma(noise.clone()),
        total_steps,
        0,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    match out {
        mlx_gen::gen_core::BoundaryState::XSigma(x)
        | mlx_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | mlx_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => Ok(x),
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
) -> Result<Array> {
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
) -> Result<Array> {
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
) -> Result<Array> {
    let (h, w) = params.latent_shape()?;
    let channels = dit.config().image_in_channels;
    let ids = image_ids(h, w, dit.config().image_rope_scale as f32);
    let z = noise(seed, h, w, channels)?;
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
    mlx_gen::image_execution::sampled_native_with_rng(
        if params.guidance.is_some() {
            "flux3-euler-cfg-v1"
        } else {
            "flux3-flow-map-v1"
        },
        "native-timestep-and-target",
        output.dtype(),
        &schedule,
        seed,
        "mlx-keyed-normal-f32-v1;initial=output-seed;no-step-draws",
    );
    mlx_gen::image_execution::policy_debug("flux3.compute_dtype", &dit.compute_dtype());
    mlx_gen::image_execution::policy_debug("flux3.reference_tokens", &references.is_some());
    mlx_gen::image_execution::policy_debug("flux3.pag", &params.pag);
    if let Some(guidance) = params.guidance {
        mlx_gen::image_execution::scalar("flux3.guidance", guidance);
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
    if ae.latent_channels() as usize != dit.config().image_in_channels {
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
pub fn decode_tokens(ae: &Flux3AutoEncoder, tokens: &Array, h: usize, w: usize) -> Result<Image> {
    let image = ae.decode(tokens, h as i32, w as i32)?;
    float_image_to_rgb8(&image)
}

/// `to_pil`: `((image.clamp(−1, 1) + 1) · 127.5).to(uint8)` — torch's float→uint8 cast
/// **truncates**, so this is `floor`, not the round-half-up of the shared
/// `decoded_to_image`. `image` is `[1, H, W, 3]` f32 (NHWC).
pub fn float_image_to_rgb8(image: &Array) -> Result<Image> {
    let sh = image.shape();
    if sh.len() != 4 || sh[0] != 1 || sh[3] != 3 {
        return Err(Error::Msg(format!(
            "flux3: a decoded image must be [1, H, W, 3], got {sh:?}"
        )));
    }
    let (height, width) = (sh[1] as u32, sh[2] as u32);
    let scaled = mlx_rs::ops::clip(&image.as_dtype(Dtype::Float32)?, (-1.0f32, 1.0f32))?
        .add(Array::from_f32(1.0))?
        .multiply(Array::from_f32(127.5))?;
    // A decoded/transposed NHWC view may still have planar (NCHW) physical storage. Elementwise
    // arithmetic need not change its strides; materialize logical C-order before host readback,
    // just like mlx-gen's shared decoded_to_interleaved_u8 boundary.
    let flat = scaled.reshape(&[-1])?;
    eval([&flat])?;
    let values = flat.as_slice::<f32>();
    let pixels: Vec<u8> = values.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    Ok(Image {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mlx_bit_equal(a: &Array, b: &Array) -> bool {
        eval([a, b]).unwrap();
        let sa = a.as_slice::<f32>();
        let sb = b.as_slice::<f32>();
        sa.len() == sb.len() && sa.iter().zip(sb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn base_cfg_off_shortcut_only_skips_negative_at_exact_guidance_one() {
        let cap = gen_core_flux3::flux3_pag_capability(false);
        let resolved = cap
            .resolve(Some(&mlx_gen::gen_core::PagConfig {
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

    #[test]
    fn flux3_flow_map_driver_passes_conformance() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

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
                        "flux3_flow_map_mlx_test",
                        &sched_clone,
                        Dtype::Bfloat16,
                        None,
                        |x: &Array, t: f32, target: Option<f32>, _pag: Option<&[u32]>| {
                            let target = target.expect("flow-map must pass Some(target_timestep)");
                            let coeff = 0.25 + 0.5 * t - 0.1 * target;
                            Ok(x.multiply(Array::from_f32(coeff))?
                                .add(Array::from_f32(0.05))?
                                .as_dtype(Dtype::Bfloat16)?)
                        },
                    )
                    .unwrap(),
                )
            },
            make_initial: || {
                Array::from_slice(&[0.5_f32, -0.75, 1.25, -0.25], &[2, 2])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap()
            },
            latents_equal: mlx_bit_equal,
        })
        .expect("FLUX.3 MLX flow-map driver passes Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_reset_preserves_draws);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn flux3_flow_map_driver_preserves_target_timesteps_rounding_and_pag() {
        use mlx_gen::driver::{DriverRunState, FamilyDriver, SliceInput, StepHooks};
        use mlx_gen::gen_core::{BoundaryState, PagConfig, Progress};

        // Schedule with < 2 points is rejected.
        assert!(Flux3FlowMapDriver::new(
            "flux3_short",
            &[1.0],
            Dtype::Bfloat16,
            None,
            |x: &Array, _: f32, _: Option<f32>, _: Option<&[u32]>| Ok(x.clone()),
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
            "flux3_flow_map_mlx_pag_test",
            &schedule,
            Dtype::Bfloat16,
            Some(pag.clone()),
            |x: &Array, t: f32, target: Option<f32>, layers: Option<&[u32]>| {
                assert_eq!(x.dtype(), Dtype::Float32, "running latent x must stay F32");
                seen_calls
                    .borrow_mut()
                    .push((t, target, layers.map(|l| l.to_vec())));
                let factor = if layers.is_some() { 0.15 } else { 0.45 };
                Ok(x.multiply(Array::from_f32(factor + 0.1 * t))?
                    .add(Array::from_f32(target.unwrap_or(0.0) * 0.2))?
                    .as_dtype(Dtype::Bfloat16)?)
            },
        )
        .unwrap();

        let init = Array::from_slice(&[0.5_f32, -0.75, 1.25, -0.25], &[2, 2]);
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
        assert_eq!(driver_x.dtype(), Dtype::Float32);

        // Reference calculation matching the frozen upstream formula bit-for-bit.
        let mut ref_x = init.as_dtype(Dtype::Float32).unwrap();
        let mut expected_calls = Vec::new();
        for step in 0..4 {
            let t = schedule[step];
            let target = schedule[step + 1];
            let dt = target - t;
            expected_calls.push((t, Some(target), None));
            let cond = ref_x
                .multiply(Array::from_f32(0.45 + 0.1 * t))
                .unwrap()
                .add(Array::from_f32(target * 0.2))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let step_pag = pag.step_scale(step, 4);
            let velocity = if step_pag > 0.0 {
                let layers = pag.step_layers(step, 4);
                expected_calls.push((t, Some(target), Some(layers.to_vec())));
                let pert = ref_x
                    .multiply(Array::from_f32(0.15 + 0.1 * t))
                    .unwrap()
                    .add(Array::from_f32(target * 0.2))
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                apply_pag_mlx(&cond, &cond, &pert, step_pag, Dtype::Bfloat16).unwrap()
            } else {
                cond
            };
            let scaled = velocity
                .as_dtype(Dtype::Float32)
                .unwrap()
                .multiply(Array::from_f32(dt))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            ref_x = ref_x
                .add(&scaled)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            eval([&ref_x]).unwrap();
        }
        assert!(mlx_bit_equal(&driver_x, &ref_x));
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
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

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
                        "flux3_base_cfg_mlx_test",
                        &sched_clone,
                        Dtype::Bfloat16,
                        |x: &Array, _step: usize, _total: usize, t: f32| {
                            Ok(x.multiply(Array::from_f32(0.3 + 0.2 * t))?
                                .as_dtype(Dtype::Bfloat16)?)
                        },
                    )
                    .unwrap(),
                )
            },
            make_initial: || Array::from_slice(&[0.25_f32, -0.5, 0.75, -1.0], &[2, 2]),
            latents_equal: mlx_bit_equal,
        })
        .expect("FLUX.3 MLX base CFG driver passes Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
    }
}
