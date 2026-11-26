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
//! draws them from a seeded CPU `StdRng` (`candle_gen::seed`, launch-portable) for production —
//! cross-backend RNG does not match torch/MLX, so this is a same-backend decode.

use candle_gen::candle_core::Tensor;
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{self, BoundaryState, RenderRun};
use candle_gen::seed::seeded_normal_vec;
use candle_gen::{CandleError, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{SampleType, SamplerConfig};
use crate::lq::PidNet;
use crate::tiling::forward_tiled;

/// Prediction seam for [`PidSamplerDriver`], allowing production `PidNet` forwards (whole-image or
/// spatially tiled) and weights-free mocked forwards to execute identically under [`FamilyDriver`].
pub trait PidVelocityForward {
    fn predict(&self, x: &Tensor, t_scaled: &Tensor) -> Result<Tensor>;
}

impl<F> PidVelocityForward for F
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    fn predict(&self, x: &Tensor, t_scaled: &Tensor) -> Result<Tensor> {
        self(x, t_scaled)
    }
}

fn core_err(err: CandleError) -> gen_core::Error {
    err.into()
}

/// Typed [`FamilyDriver`] executing Candle PiD's pixel-diffusion sampler schedule (`Sde` or `Ode`).
pub struct PidSamplerDriver<F> {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
    eps: Vec<Tensor>,
    forward: F,
    support: DriverSupport,
}

impl<F> PidSamplerDriver<F> {
    /// Construct a PiD inner sampler driver over explicit schedule parameters and pre-drawn ε tensors.
    pub fn new(t_list: Vec<f32>, timescale: f32, sde: bool, eps: Vec<Tensor>, forward: F) -> Self {
        Self {
            t_list,
            timescale,
            sde,
            eps,
            forward,
            support: DriverSupport {
                driver_id: "candle_pid_sampler",
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
    pub fn from_sampler(sampler: &Sampler, eps: Vec<Tensor>, forward: F) -> Self {
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

impl<F: PidVelocityForward> FamilyDriver<Tensor> for PidSamplerDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> gen_core::Result<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        if input.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
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
        candle_gen::execution::runtime(x.device(), x.dtype());
        candle_gen::execution::schedule(
            if self.sde { "pid-sde" } else { "pid-ode" },
            &self.t_list,
            "pixel-velocity",
        );
        candle_gen::execution::policy(
            "pid_sampler",
            format!(
                "sde={};timescale_bits={};epsilon_count={};noise_shape={:?}",
                self.sde,
                self.timescale.to_bits(),
                self.eps.len(),
                x.dims()
            )
            .into(),
        );
        let b = x.dim(0).map_err(CandleError::from).map_err(core_err)?;
        let device = x.device().clone();
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
            let t_scaled = Tensor::from_vec(vec![t_cur * self.timescale; b], (b,), &device)
                .map_err(CandleError::from)
                .map_err(core_err)?;
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
                    let a = (&x0 * (1.0 - t_next) as f64)
                        .map_err(CandleError::from)
                        .map_err(core_err)?;
                    let b_term = (e * t_next as f64)
                        .map_err(CandleError::from)
                        .map_err(core_err)?;
                    x = (a + b_term).map_err(CandleError::from).map_err(core_err)?;
                } else {
                    let delta = (&v * (t_next - t_cur) as f64)
                        .map_err(CandleError::from)
                        .map_err(core_err)?;
                    x = (&x + &delta).map_err(CandleError::from).map_err(core_err)?;
                }
            } else {
                x = Sampler::velocity_to_x0(&x, &v, t_cur).map_err(core_err)?;
            }
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            state.record_outer_step(i);
            hooks.emit_progress(&info);
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
        }
        let clamped = x
            .clamp(-1.0f32, 1.0f32)
            .map_err(CandleError::from)
            .map_err(core_err)?;
        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(clamped))
    }
}

/// The distilled few-step sampler.
pub struct Sampler {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
}

impl Sampler {
    pub fn new(cfg: &SamplerConfig) -> Self {
        Self {
            t_list: cfg.student_t_list.clone(),
            timescale: cfg.fm_timescale,
            sde: cfg.sample_type == SampleType::Sde,
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
    fn velocity_to_x0(x: &Tensor, v: &Tensor, t: f32) -> Result<Tensor> {
        Ok((x - (v * t as f64)?)?)
    }

    /// Execute [`PidSamplerDriver`] through [`run_candle_driver_slices`], optionally hosted inside
    /// an active S10 [`RenderRun`] `Decode` stage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn drive_with_forward<F: Fn(&Tensor, &Tensor) -> Result<Tensor>>(
        &self,
        noise: &Tensor,
        eps: Vec<Tensor>,
        seed: u64,
        cancel: Option<&CancelFlag>,
        run: Option<&RenderRun<'_, '_>>,
        forward: F,
    ) -> Result<Tensor> {
        let total_steps = self.steps();
        let driver = PidSamplerDriver::from_sampler(self, eps, forward);
        let fallback_cancel = CancelFlag::new();
        let cancel = cancel.unwrap_or(&fallback_cancel);
        let dims = noise.dims();
        let geometry = (dims.len() >= 4).then(|| (dims[3] as u32, dims[2] as u32));
        let mut on_progress = |_| {};
        let mut hooks = StepHooks::new(&mut on_progress);
        let mut spec = SliceSpec::full(total_steps);
        spec.geometry = geometry;
        let out_state = match run {
            Some(render_run) => render_run.decode_phase_with_driver_with(
                driver.support(),
                total_steps,
                |driver_state| {
                    run_candle_driver_slices(
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
                run_candle_driver_slices(
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
            return Err(CandleError::Msg(
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
        noise: &Tensor,
        eps: &[Tensor],
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        candle_gen::execution::policy("pid_decode", "whole_image".into());
        self.drive_with_forward(noise, eps.to_vec(), 0, cancel, None, |x, t_scaled| {
            net.forward(x, t_scaled, caption, lq_latent, sigma)
        })
    }

    /// Like [`Self::run`] but the per-step **velocity** forward is spatially tiled (sc-10087):
    /// [`crate::tiling::forward_tiled`] runs the net on overlapping `tile`-px pixel windows and
    /// feather-blends them, so the whole-image `PidNet::forward` VRAM peak never materializes.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled(
        &self,
        net: &PidNet,
        noise: &Tensor,
        eps: &[Tensor],
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        candle_gen::execution::policy(
            "pid_decode",
            format!("tiled;edge={tile};overlap={overlap}").into(),
        );
        self.drive_with_forward(noise, eps.to_vec(), 0, cancel, None, |x, t_scaled| {
            forward_tiled(net, x, t_scaled, caption, lq_latent, sigma, tile, overlap)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_in_render_run(
        &self,
        net: &PidNet,
        noise: &Tensor,
        eps: Vec<Tensor>,
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        seed: u64,
        tiling: Option<(i32, i32)>,
        cancel: Option<&CancelFlag>,
        run: &RenderRun<'_, '_>,
    ) -> Result<Tensor> {
        match tiling {
            Some((tile, overlap)) => {
                candle_gen::execution::policy(
                    "pid_decode",
                    format!("tiled;edge={tile};overlap={overlap}").into(),
                );
                self.drive_with_forward(noise, eps, seed, cancel, Some(run), |x, t_scaled| {
                    forward_tiled(net, x, t_scaled, caption, lq_latent, sigma, tile, overlap)
                })
            }
            None => {
                candle_gen::execution::policy("pid_decode", "whole_image".into());
                self.drive_with_forward(noise, eps, seed, cancel, Some(run), |x, t_scaled| {
                    net.forward(x, t_scaled, caption, lq_latent, sigma)
                })
            }
        }
    }

    pub(crate) fn draw_noise_eps(
        &self,
        device: &candle_gen::candle_core::Device,
        b: usize,
        h: usize,
        w: usize,
        seed: u64,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        candle_gen::execution::stream("pid_initial_then_epsilon_continuing_stream", seed);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut draw = || -> Result<Tensor> {
            let v = seeded_normal_vec(&mut rng, b * 3 * h * w);
            Ok(Tensor::from_vec(v, (b, 3, h, w), device)?)
        };
        let noise = draw()?;
        let mut eps = Vec::with_capacity(self.num_eps());
        for _ in 0..self.num_eps() {
            eps.push(draw()?);
        }
        Ok((noise, eps))
    }

    /// Production entry: draw the initial noise + per-step ε from a seeded CPU `StdRng` (launch-
    /// portable), then run the loop. Returns clamped pixels `[B, 3, H, W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        net: &PidNet,
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        b: usize,
        h: usize,
        w: usize,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let (noise, eps) = self.draw_noise_eps(lq_latent.device(), b, h, w, seed)?;
        self.run(net, &noise, &eps, caption, lq_latent, sigma, cancel)
    }

    /// Tiled production entry (sc-10087): identical seeded noise/ε draw as [`Self::sample`], then
    /// [`Self::run_tiled`] with the spatial `tile` / `overlap`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tiled(
        &self,
        net: &PidNet,
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: &Tensor,
        b: usize,
        h: usize,
        w: usize,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let (noise, eps) = self.draw_noise_eps(lq_latent.device(), b, h, w, seed)?;
        self.run_tiled(
            net, &noise, &eps, caption, lq_latent, sigma, tile, overlap, cancel,
        )
    }
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn make_constant(val: f32) -> Tensor {
        Tensor::from_vec(vec![val; 3 * 2 * 2], (1, 3, 2, 2), &Device::Cpu).unwrap()
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
                        |x: &Tensor, t_scaled: &Tensor| {
                            let ts = t_scaled.to_vec1::<f32>()?[0] as f64 * 1e-4;
                            Ok((x * ts)?)
                        },
                    ))
                },
                make_initial: || make_constant(0.5),
                latents_equal: |a, b| {
                    let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    va.len() == vb.len()
                        && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
                },
            })
            .expect("Candle PidSamplerDriver passes Tier 0 conformance");
            assert_eq!(report.driver_id, "candle_pid_sampler");
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
        let ode_driver = PidSamplerDriver::new(
            vec![1.0, 0.5, 0.0],
            1000.0,
            false,
            Vec::new(),
            |_x: &Tensor, _t: &Tensor| Ok(make_constant(0.25)),
        );
        let cancel = CancelFlag::default();
        let mut st = DriverRunState::new();
        let mut ticks = Vec::new();
        let mut on_progress = |p: gen_core::Progress| ticks.push(p);
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_candle_driver_slices(
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
        let BoundaryState::XSigma(tensor) = out else {
            panic!("expected XSigma");
        };
        let vals = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|&v| (v - 0.25).abs() < 1e-6));
        assert_eq!(ticks.len(), 2);

        let bad_sde = PidSamplerDriver::new(
            vec![0.999, 0.866, 0.634, 0.342, 0.0],
            1000.0,
            true,
            vec![make_constant(0.0)],
            |_x: &Tensor, _t: &Tensor| Ok(make_constant(0.0)),
        );
        let mut st_bad = DriverRunState::new();
        let mut on_p = |_| {};
        let mut h_bad = StepHooks::new(&mut on_p);
        assert!(run_candle_driver_slices(
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

        let plan_req_cfg = SamplerConfig::distill_4step();
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
            |_x: &Tensor, _t: &Tensor| Ok(make_constant(0.1)),
        );
        let run = RenderRun::new(&s10, &cancel, None);
        let out_hosted = run
            .decode_phase_with_driver_with(sde_driver.support(), 4, |driver_state| {
                let mut cb = |_| {};
                let mut hooks = StepHooks::new(&mut cb);
                run_candle_driver_slices(
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
        let finished = run.finish_output::<_, CandleError>(out_hosted).unwrap();
        assert!(matches!(finished, BoundaryState::XSigma(_)));
        let execs = run.decode_driver_executions();
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].completed_outer_steps, 4);
        assert_eq!(execs[0].stochastic_draw_index, 3);
    }
}
