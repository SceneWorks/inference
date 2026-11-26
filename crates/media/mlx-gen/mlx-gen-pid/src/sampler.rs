//! The distilled few-step pixel-diffusion sampler — faithful port of
//! `pid_distill_model.py::_student_sample_loop` (+ `_velocity_to_x0`). The released students run the
//! **SDE / velocity-prediction** schedule (`student_t_list=[0.999,0.866,0.634,0.342,0.0]`,
//! `fm_timescale=1000`, cfg 1 — no classifier-free guidance). PiD denoises directly in high-res
//! **pixel** space: `noise`/`x` are `[B, 3, H, W]` at the *output* resolution, conditioned on the LQ
//! latent + caption + degrade σ.
//!
//! Per step `(t_cur, t_next)`: `v = net(x, t_cur·timescale, …)`, `x0 = x − t_cur·v`; then for an SDE
//! interior step `x = (1−t_next)·x0 + t_next·ε` (fresh noise), and the final `t_next=0` step takes
//! `x = x0`. Output is clamped to `[-1, 1]`.
//!
//! The step math is RNG-free and deterministic — [`Sampler::run`] takes the initial noise and the
//! per-step ε injected, so it parity-tests bit-for-bit against the torch loop. [`Sampler::sample`]
//! draws them from MLX's PRNG for production (cross-backend RNG does not match torch — a same-backend
//! decode, per the repo's full-trajectory chaos note).

use mlx_rs::ops::{add, clip, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use mlx_gen::gen_core::{self, BoundaryState, RenderRun};
use mlx_gen::{CancelFlag, Error, Result};

use crate::config::{SampleType, SamplerConfig};
use crate::lq::PidNet;
use crate::tiling::{forward_tiled, FeatherCache};

/// Prediction seam for [`PidSamplerDriver`], allowing production `PidNet` forwards (whole-image or
/// spatially tiled) and weights-free mocked forwards to execute identically under [`FamilyDriver`].
pub trait PidVelocityForward {
    fn predict(&self, x: &Array, t_scaled: &Array) -> Result<Array>;
}

impl<F> PidVelocityForward for F
where
    F: Fn(&Array, &Array) -> Result<Array>,
{
    fn predict(&self, x: &Array, t_scaled: &Array) -> Result<Array> {
        self(x, t_scaled)
    }
}

fn core_err(err: Error) -> gen_core::Error {
    err.into()
}

/// Typed [`FamilyDriver`] executing PiD's pixel-diffusion sampler schedule (`Sde` or `Ode`).
pub struct PidSamplerDriver<F> {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
    eps: Vec<Array>,
    forward: F,
    support: DriverSupport,
}

impl<F> PidSamplerDriver<F> {
    /// Construct a PiD inner sampler driver over explicit schedule parameters and pre-drawn ε tensors.
    pub fn new(t_list: Vec<f32>, timescale: f32, sde: bool, eps: Vec<Array>, forward: F) -> Self {
        Self {
            t_list,
            timescale,
            sde,
            eps,
            forward,
            support: DriverSupport {
                driver_id: "mlx_pid_sampler",
                arbitrary_slice: false,
                carry_history: false,
                reset_history: false,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: false,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons::default(),
            },
        }
    }

    /// Construct a [`PidSamplerDriver`] from a [`Sampler`] configuration and pre-drawn ε slice.
    pub fn from_sampler(sampler: &Sampler, eps: Vec<Array>, forward: F) -> Self {
        Self::new(
            sampler.t_list.clone(),
            sampler.timescale,
            sampler.sde,
            eps,
            forward,
        )
    }

    fn num_eps(&self) -> usize {
        if !self.sde {
            return 0;
        }
        (1..self.t_list.len())
            .filter(|&i| self.t_list[i] > 0.0)
            .count()
    }
}

impl<F: PidVelocityForward> FamilyDriver<Array> for PidSamplerDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> gen_core::Result<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        if input.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        // F-100: validate pre-drawn ε count up front so no interior SDE step panics OOB.
        let required_eps = self.num_eps();
        if self.eps.len() < required_eps {
            return Err(gen_core::Error::Msg(format!(
                "pid sampler: need {required_eps} eps draws for this schedule, got {}",
                self.eps.len()
            )));
        }
        state.enter_slice(&input)?;
        let mut x = match input.state {
            BoundaryState::XSigma(latent)
            | BoundaryState::CleanEstimate { latent, .. }
            | BoundaryState::RouteSpecific { state: latent, .. } => latent,
        };
        let b = x.shape()[0];
        for i in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let t_cur = self.t_list[i];
            let t_next = self.t_list[i + 1];
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: i,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: t_cur,
                next_sigma: t_next,
            };
            hooks.observe(&info, &x)?;
            let t_scaled = Array::from_slice(&vec![t_cur * self.timescale; b as usize], &[b]);
            let v = self.forward.predict(&x, &t_scaled).map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            if t_next > 0.0 {
                if self.sde {
                    let x0 = Sampler::velocity_to_x0(&x, &v, t_cur).map_err(core_err)?;
                    let ei = state.next_draw_index() as usize;
                    let e = self.eps.get(ei).ok_or_else(|| {
                        gen_core::Error::Msg(format!(
                            "pid sampler: missing eps draw at index {ei} (have {})",
                            self.eps.len()
                        ))
                    })?;
                    x = add(
                        &multiply(&x0, scalar(1.0 - t_next))
                            .map_err(Error::from)
                            .map_err(core_err)?,
                        &multiply(e, scalar(t_next))
                            .map_err(Error::from)
                            .map_err(core_err)?,
                    )
                    .map_err(Error::from)
                    .map_err(core_err)?;
                } else {
                    // ODE: x = x + (t_next − t_cur)·v (velocity prediction).
                    x = add(
                        &x,
                        &multiply(&v, scalar(t_next - t_cur))
                            .map_err(Error::from)
                            .map_err(core_err)?,
                    )
                    .map_err(Error::from)
                    .map_err(core_err)?;
                }
            } else {
                x = Sampler::velocity_to_x0(&x, &v, t_cur).map_err(core_err)?;
            }
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            // Materialize this step so a cancel between steps is actually observed (MLX is lazy),
            // and to bound transient peak memory (F-013): the 4-step graph no longer schedules at once.
            eval([&x]).map_err(Error::from).map_err(core_err)?;
            state.record_outer_step(i);
            hooks.emit_progress(&info);
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
        }
        let clipped = clip(&x, (-1.0, 1.0))
            .map_err(Error::from)
            .map_err(core_err)?;
        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(clipped))
    }
}

/// The distilled few-step sampler.
pub struct Sampler {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
    /// Per-decode cache of the tiled path's feather-weight tables (F-153) — deterministic across the
    /// 4 steps, so each tile shape's feather is built once. Lives here (one `Sampler` per decode) so it
    /// persists across the per-step `forward_tiled` calls without leaking across generations.
    feather_cache: FeatherCache,
}

impl Sampler {
    pub fn new(cfg: &SamplerConfig) -> Self {
        Self {
            t_list: cfg.student_t_list.clone(),
            timescale: cfg.fm_timescale,
            sde: cfg.sample_type == SampleType::Sde,
            feather_cache: FeatherCache::default(),
        }
    }

    pub(crate) fn t_list(&self) -> &[f32] {
        &self.t_list
    }

    pub(crate) fn timescale(&self) -> f32 {
        self.timescale
    }

    pub(crate) fn sample_type(&self) -> SampleType {
        if self.sde {
            SampleType::Sde
        } else {
            SampleType::Ode
        }
    }

    /// Host metadata from this actual sampler; no model execution is needed to test the receipt.
    pub(crate) fn observe_execution(&self, seed: u64) {
        mlx_gen::image_execution::scalar("pid_timescale", self.timescale);
        mlx_gen::image_execution::policy_debug("pid_epsilon_draws", &self.num_eps());
        mlx_gen::image_execution::policy_debug("pid_output_clamp", &"[-1,1]");
        mlx_gen::image_execution::policy_debug("pid_dtypes", &"network=bf16;noise=f32");
        mlx_gen::image_execution::sampled_decode(
            if self.sde { "pid_sde" } else { "pid_ode" }.into(),
            &self.t_list, seed,
            "mlx-split-key-normal-v1;noise=split(key(seed),2).0;epsilon=repeated-split(rest,2).0;full-image-noise-independent-of-tiling",
        );
    }

    /// Number of denoising steps (`len(t_list) − 1`).
    pub fn steps(&self) -> usize {
        self.t_list.len().saturating_sub(1)
    }

    /// Number of fresh-noise draws the SDE loop consumes (one per interior step with `t_next>0`).
    pub fn num_eps(&self) -> usize {
        if !self.sde {
            return 0;
        }
        (1..self.t_list.len())
            .filter(|&i| self.t_list[i] > 0.0)
            .count()
    }

    /// velocity-prediction `x0 = x − t·v`.
    fn velocity_to_x0(x: &Array, v: &Array, t: f32) -> Result<Array> {
        Ok(subtract(x, &multiply(v, scalar(t))?)?)
    }

    /// Execute [`PidSamplerDriver`] through [`run_mlx_driver_slices`], optionally hosted inside an
    /// active S10 [`RenderRun`] `Decode` stage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn drive_with_forward<F: Fn(&Array, &Array) -> Result<Array>>(
        &self,
        noise: &Array,
        eps: Vec<Array>,
        seed: u64,
        cancel: Option<&CancelFlag>,
        run: Option<&RenderRun<'_, '_>>,
        forward: F,
    ) -> Result<Array> {
        let total_steps = self.steps();
        let driver = PidSamplerDriver::from_sampler(self, eps, forward);
        let fallback_cancel = CancelFlag::new();
        let cancel = cancel.unwrap_or(&fallback_cancel);
        let sh = noise.shape();
        let geometry = (sh.len() >= 4).then(|| (sh[3] as u32, sh[2] as u32));
        let mut on_progress = |_| {};
        let mut hooks = StepHooks::new(&mut on_progress);
        let mut spec = SliceSpec::full(total_steps);
        spec.geometry = geometry;
        let out_state = match run {
            Some(render_run) => render_run.decode_phase_with_driver_with(
                driver.support(),
                total_steps,
                |driver_state| {
                    run_mlx_driver_slices(
                        &driver,
                        [spec],
                        BoundaryState::XSigma(noise.clone()),
                        total_steps,
                        seed,
                        cancel,
                        None,
                        driver_state,
                        &mut hooks,
                        |_, out, _| Ok(out),
                    )
                },
            )?,
            None => {
                let mut driver_state = DriverRunState::new();
                run_mlx_driver_slices(
                    &driver,
                    [spec],
                    BoundaryState::XSigma(noise.clone()),
                    total_steps,
                    seed,
                    cancel,
                    None,
                    &mut driver_state,
                    &mut hooks,
                    |_, out, _| Ok(out),
                )?
            }
        };
        let BoundaryState::XSigma(out) = out_state else {
            return Err(Error::Msg(
                "pid sampler driver returned non-XSigma state".into(),
            ));
        };
        Ok(out)
    }

    /// Deterministic loop with the initial `noise` and the per-step `eps` injected (one ε per SDE
    /// interior step, in order). `caption`/`lq_latent`/`sigma` condition the net every step.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.drive_with_forward(noise, eps.to_vec(), 0, cancel, None, |x, t_scaled| {
            net.forward(x, t_scaled, caption, lq_latent, sigma)
        })
    }

    /// Like [`Self::run`] but the per-step **velocity** forward is spatially tiled (sc-10087):
    /// [`crate::tiling::forward_tiled`] runs the net on overlapping `tile`-px pixel windows and
    /// feather-blends them, so the whole-image `PidNet::forward` peak (and its single long Metal command
    /// buffer) never materializes.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.drive_with_forward(noise, eps.to_vec(), 0, cancel, None, |x, t_scaled| {
            forward_tiled(
                net,
                x,
                t_scaled,
                caption,
                lq_latent,
                sigma,
                tile,
                overlap,
                &self.feather_cache,
            )
        })
    }

    /// Execute whole-image or spatially-tiled forward in an optional hosted [`RenderRun`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_in_render_run(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: Vec<Array>,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        seed: u64,
        tiling: Option<(i32, i32)>,
        cancel: Option<&CancelFlag>,
        run: &RenderRun<'_, '_>,
    ) -> Result<Array> {
        match tiling {
            Some((tile, overlap)) => {
                self.drive_with_forward(noise, eps, seed, cancel, Some(run), |x, t_scaled| {
                    forward_tiled(
                        net,
                        x,
                        t_scaled,
                        caption,
                        lq_latent,
                        sigma,
                        tile,
                        overlap,
                        &self.feather_cache,
                    )
                })
            }
            None => {
                let out =
                    self.drive_with_forward(noise, eps, seed, cancel, Some(run), |x, t_scaled| {
                        net.forward(x, t_scaled, caption, lq_latent, sigma)
                    })?;
                mlx_gen::image_execution::policy_debug("pid_forward_tiled", &false);
                Ok(out)
            }
        }
    }

    /// Draw the seeded initial noise + per-step ε from MLX's PRNG — the single source of truth for the
    /// RNG stream both [`Self::sample`] and [`Self::sample_tiled`] consume (F-155).
    pub(crate) fn draw_noise_eps(
        &self,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
    ) -> Result<(Array, Vec<Array>)> {
        let (k_noise, mut k_rest) = random::split(&random::key(seed)?, 2)?;
        let noise = random::normal::<f32>(&[b, 3, h, w], None, None, Some(&k_noise))?;
        let mut eps = Vec::with_capacity(self.num_eps());
        for _ in 0..self.num_eps() {
            let (k_e, k_n) = random::split(&k_rest, 2)?;
            eps.push(random::normal::<f32>(
                &[b, 3, h, w],
                None,
                None,
                Some(&k_e),
            )?);
            k_rest = k_n;
        }
        Ok((noise, eps))
    }

    /// Production entry: draw the initial noise + per-step ε from MLX's PRNG (seeded), then run the
    /// loop. Returns clamped pixels `[B, 3, H, W]`. `cancel` is threaded into [`Self::run`] (F-006).
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        net: &PidNet,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let (noise, eps) = self.draw_noise_eps(b, h, w, seed)?;
        let out = self.run(net, &noise, &eps, caption, lq_latent, sigma, cancel)?;
        mlx_gen::image_execution::policy_debug("pid_forward_tiled", &false);
        self.observe_execution(seed);
        Ok(out)
    }

    /// Tiled production entry (sc-10087): identical seeded noise/ε draw as [`Self::sample`], then
    /// [`Self::run_tiled`] with the spatial `tile` / `overlap`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tiled(
        &self,
        net: &PidNet,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let (noise, eps) = self.draw_noise_eps(b, h, w, seed)?;
        let out = self.run_tiled(
            net, &noise, &eps, caption, lq_latent, sigma, tile, overlap, cancel,
        )?;
        self.observe_execution(seed);
        Ok(out)
    }
}

#[cfg(test)]
mod execution_tests {
    use super::*;
    use mlx_gen::gen_core::generator::{ExecutionFact, ExecutionRandomness, NumericScheduleDigest};
    use mlx_gen::{Capabilities, GenerationOutput, Image, Modality, ModelDescriptor};

    #[test]
    fn completed_pid_observation_uses_sampler_schedule_and_shared_decoder_seed() {
        let descriptor = ModelDescriptor {
            id: "pid_probe",
            family: "probe",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        };
        for sample_type in [SampleType::Sde, SampleType::Ode] {
            let mut cfg = SamplerConfig::distill_4step();
            cfg.sample_type = sample_type;
            cfg.student_t_list = vec![0.9, 0.4, 0.0];
            cfg.fm_timescale = 321.0;
            let sampler = Sampler::new(&cfg);
            let report = mlx_gen::image_execution::complete_image_attempt(
                &descriptor,
                &CancelFlag::new(),
                || {
                    for seed in [9, 10] {
                        mlx_gen::image_execution::sampled_native(
                            "euler",
                            "Sigma",
                            mlx_rs::Dtype::Float32,
                            &[1.0, 0.0],
                            seed,
                        );
                    }
                    // Production shares one PiD decoder (and therefore its seed) across the count loop.
                    sampler.observe_execution(9);
                    sampler.observe_execution(9);
                    Ok(GenerationOutput::Images(vec![
                        Image {
                            width: 1,
                            height: 1,
                            pixels: vec![0; 3]
                        };
                        2
                    ]))
                },
            )
            .unwrap();
            let resolved = report.execution.resolved().unwrap();
            let ExecutionRandomness::Draws { draws, scheme } = &resolved.randomness else {
                panic!()
            };
            assert_eq!(
                draws
                    .iter()
                    .map(|draw| (draw.output_index, draw.seed))
                    .collect::<Vec<_>>(),
                vec![(0, Some(9)), (1, Some(10)), (0, Some(9)), (1, Some(9))]
            );
            assert!(scheme.contains("full-image-noise-independent-of-tiling"));
            assert_eq!(
                resolved.provider_choices["pid_timescale"],
                ExecutionFact::f32_bits(321.0).unwrap()
            );
            let ExecutionFact::Resolved { value: stages } =
                &resolved.provider_choices["auxiliary_decode"]
            else {
                panic!()
            };
            assert_eq!(
                stages[0]["schedule"]["sha256"],
                NumericScheduleDigest::f32_values(&cfg.student_t_list)
                    .unwrap()
                    .sha256
            );
            assert_eq!(
                stages[0]["sampling"],
                if sample_type == SampleType::Sde {
                    "pid_sde"
                } else {
                    "pid_ode"
                }
            );
        }
    }
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn make_constant(val: f32) -> Array {
        Array::from_slice(&[val; 3 * 2 * 2], &[1, 3, 2, 2])
    }

    fn eval_and_vec(a: &Array) -> Vec<f32> {
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    }

    #[test]
    fn pid_sampler_driver_passes_tier0_conformance_for_sde_and_ode() {
        for sample_type in [SampleType::Sde, SampleType::Ode] {
            let mut cfg = SamplerConfig::distill_4step();
            cfg.sample_type = sample_type;
            let eps = if sample_type == SampleType::Sde {
                vec![make_constant(0.1), make_constant(-0.2), make_constant(0.3)]
            } else {
                Vec::new()
            };
            let t_list = cfg.student_t_list.clone();
            let timescale = cfg.fm_timescale;
            let sde = cfg.sample_type == SampleType::Sde;

            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 4,
                split_at: 2,
                seed: 7,
                base_geometry: (64, 64),
                alt_geometry: (128, 128),
                make_driver: move || {
                    let t_list = t_list.clone();
                    let eps = eps.clone();
                    Box::new(PidSamplerDriver::new(
                        t_list,
                        timescale,
                        sde,
                        eps,
                        |x: &Array, t_scaled: &Array| {
                            let scale = Array::from_f32(t_scaled.as_slice::<f32>()[0] * 1e-4);
                            Ok(multiply(x, &scale)?)
                        },
                    ))
                },
                make_initial: || make_constant(0.5),
                latents_equal: |a, b| {
                    let va = eval_and_vec(a);
                    let vb = eval_and_vec(b);
                    va.len() == vb.len()
                        && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
                },
            })
            .expect("MLX PidSamplerDriver passes Tier 0 conformance");
            assert_eq!(report.driver_id, "mlx_pid_sampler");
            assert_eq!(report.total_steps, 4);
            assert!(report.verified_cancellation);
            assert_eq!(
                report.unsplit_draws,
                if sample_type == SampleType::Sde { 3 } else { 0 }
            );
        }
    }

    #[test]
    fn pid_sampler_driver_preserves_analytic_math_and_hosted_s10_run() {
        // Exact analytical verification without learned weights:
        // Constant velocity v = 0.25 with x_0 = 0.5, ODE steps [1.0, 0.5, 0.0]:
        //   step 0: x_1 = 0.5 + (0.5 - 1.0) * 0.25 = 0.375
        //   step 1: x0 = x_1 - 0.5 * 0.25 = 0.25 -> clamped [-1, 1] = 0.25
        let ode_driver = PidSamplerDriver::new(
            vec![1.0, 0.5, 0.0],
            1000.0,
            false,
            Vec::new(),
            |_x: &Array, _t: &Array| Ok(make_constant(0.25)),
        );
        let cancel = CancelFlag::new();
        let mut st = DriverRunState::new();
        let mut ticks = Vec::new();
        let mut on_progress = |p: mlx_gen::Progress| ticks.push(p);
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_mlx_driver_slices(
            &ode_driver,
            [SliceSpec::full(2)],
            BoundaryState::XSigma(make_constant(0.5)),
            2,
            11,
            &cancel,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let BoundaryState::XSigma(arr) = out else {
            panic!("expected XSigma");
        };
        let vals = eval_and_vec(&arr);
        assert!(vals.iter().all(|&v| (v - 0.25).abs() < 1e-6));
        assert_eq!(ticks.len(), 2);

        // Insufficient eps draws for SDE must refuse cleanly before running.
        let bad_sde = PidSamplerDriver::new(
            vec![0.999, 0.866, 0.634, 0.342, 0.0],
            1000.0,
            true,
            vec![make_constant(0.0)], // needs 3
            |_x: &Array, _t: &Array| Ok(make_constant(0.0)),
        );
        let mut st_bad = DriverRunState::new();
        let mut on_p = |_| {};
        let mut h_bad = StepHooks::new(&mut on_p);
        assert!(run_mlx_driver_slices(
            &bad_sde,
            [SliceSpec::full(4)],
            BoundaryState::XSigma(make_constant(0.0)),
            4,
            11,
            &cancel,
            None,
            &mut st_bad,
            &mut h_bad,
            |_, out, _| Ok(out),
        )
        .is_err());

        // Host PidSamplerDriver inside a real S10 Decode RenderPlan / RenderRun.
        let mut plan_req_cfg = SamplerConfig::distill_4step();
        plan_req_cfg.sample_type = SampleType::Sde;
        let decode_plan = crate::plan::PidDecodePlan::compile_for_fields(
            "pid",
            &crate::config::PidConfig::sr4x(),
            &plan_req_cfg,
            None,
            16,
            16,
            1,
            0.0,
            42,
            64.0,
            None,
        )
        .unwrap();
        let s10 = decode_plan.render_plan(64, 64, 1).unwrap();
        let sde_driver = PidSamplerDriver::new(
            plan_req_cfg.student_t_list.clone(),
            plan_req_cfg.fm_timescale,
            true,
            vec![make_constant(0.2), make_constant(-0.1), make_constant(0.05)],
            |_x: &Array, _t: &Array| Ok(make_constant(0.1)),
        );
        let run = RenderRun::new(&s10, &cancel, None);
        let out_hosted = run
            .decode_phase_with_driver_with(sde_driver.support(), 4, |driver_state| {
                let mut cb = |_| {};
                let mut hooks = StepHooks::new(&mut cb);
                run_mlx_driver_slices(
                    &sde_driver,
                    [SliceSpec::full(4)],
                    BoundaryState::XSigma(make_constant(0.5)),
                    4,
                    42,
                    &cancel,
                    None,
                    driver_state,
                    &mut hooks,
                    |_, out, _| Ok(out),
                )
            })
            .unwrap();
        let finished = run.finish_output::<_, Error>(out_hosted).unwrap();
        assert!(matches!(finished, BoundaryState::XSigma(_)));
        let execs = run.decode_driver_executions();
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].completed_outer_steps, 4);
        assert_eq!(execs[0].stochastic_draw_index, 3);
    }
}
