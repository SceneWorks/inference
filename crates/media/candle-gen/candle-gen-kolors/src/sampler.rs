//! Kolors scheduler — the candle port of `mlx-gen-kolors`'s `sampler.rs`: a faithful reproduction of
//! diffusers `EulerDiscreteScheduler` with the Kolors config (`scaled_linear` betas β₀=0.00085,
//! β₁=0.014, **num_train_timesteps=1100**, `timestep_spacing="leading"`, `steps_offset=1`,
//! `interpolation_type="linear"`, `final_sigmas_type="zero"`, epsilon prediction, `s_churn=0`).
//!
//! This is the non-ancestral Euler — **no per-step RNG** — so a denoise run is fully deterministic
//! given the initial latents (paired with the sc-3673 CPU-seeded noise, generation is a pure function
//! of `(seed, request)`). The schedule math is pure host f64/f32 (GPU-free, unit-tested below); the
//! pipeline applies it with candle tensor ops:
//!  - **leading** timesteps: `(arange(0,N)·(1100//N)).round()[::-1] + steps_offset`;
//!  - leading ⇒ `init_noise_sigma = (max_sigma² + 1)^0.5`;
//!  - `scale_model_input`: divide latents by `√(σ²+1)` before the UNet;
//!  - Euler step (ε-pred, γ=0): `x_next = x + ε·(σ_next − σ)`.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::sampling::{BetaCumprodPrecision, LeadingEulerSchedule};
use candle_gen::gen_core::{self, Progress};
use candle_gen::{CandleError, Result as CandleResult};

/// Kolors' `num_train_timesteps` — the length of the `scaled_linear` schedule the sampler interpolates
/// over. `num_steps` must lie in `1..=NUM_TRAIN_TIMESTEPS`.
pub use candle_gen::diffusion_schedule::KOLORS_TRAIN_STEPS as NUM_TRAIN_TIMESTEPS;

/// Kolors' EulerDiscrete (leading) sampler over the 1100-step `scaled_linear` schedule.
///
/// Schedule construction delegates to [`LeadingEulerSchedule`] (`PIPE-9`) with
/// [`BetaCumprodPrecision::F64`] for bit-exact parity with diffusers' f64 cumprod table.
#[derive(Clone, Debug)]
pub struct KolorsEulerSampler {
    schedule: LeadingEulerSchedule,
}

impl KolorsEulerSampler {
    /// Build the Kolors schedule for `num_steps` denoise steps. Errors on `num_steps == 0` (divide by
    /// zero) or `> NUM_TRAIN_TIMESTEPS` (every leading timestep would collapse to a single value).
    pub fn new(num_steps: usize) -> Result<Self, String> {
        let schedule = LeadingEulerSchedule::kolors(num_steps, BetaCumprodPrecision::F64)
            .map_err(|e| e.to_string())?;
        Ok(Self { schedule })
    }

    /// Build the full schedule, then keep only the strength-selected denoise tail. This mirrors
    /// diffusers' `KolorsImg2ImgPipeline.get_timesteps`: `floor(steps * strength)` effective steps,
    /// clamped to the original schedule. Strength zero retains only the terminal sigma and runs no
    /// denoise steps, so the deterministic VAE mean round-trips without injected noise.
    pub fn img2img(num_steps: usize, strength: f32) -> Result<Self, String> {
        let schedule =
            LeadingEulerSchedule::kolors_img2img(num_steps, strength, BetaCumprodPrecision::F64)
                .map_err(|e| e.to_string())?;
        Ok(Self { schedule })
    }

    /// The number of denoise steps.
    pub fn num_steps(&self) -> usize {
        self.schedule.num_steps()
    }

    /// The descending EDM σ table (length `num_steps + 1`, trailing `0.0`).
    pub fn sigmas(&self) -> &[f32] {
        self.schedule.sigmas()
    }

    /// The leading float timestep fed to the UNet time embedding at step `i`.
    pub fn timestep(&self, i: usize) -> f32 {
        if i == 0 {
            candle_gen::execution::schedule("kolors-euler", self.schedule.sigmas(), "epsilon");
            candle_gen::execution::policy(
                "kolors_timesteps",
                format!("{:?}", self.schedule.timesteps()).into(),
            );
        }
        self.schedule.timestep(i)
    }

    /// `init_noise_sigma` — the initial latents are `noise · init_noise_sigma`.
    pub fn init_noise_sigma(&self) -> f32 {
        self.schedule.init_noise_sigma()
    }

    /// Seed an img2img latent in raw VE sigma space: `x0 + noise * sigma_start`.
    pub fn add_noise(
        &self,
        x0: &Tensor,
        noise: &Tensor,
    ) -> candle_gen::candle_core::Result<Tensor> {
        if self.schedule.num_steps() == 0 {
            candle_gen::execution::without_schedule(
                "source_roundtrip",
                "strength selected zero denoise steps",
                "no denoiser evaluation",
            );
            candle_gen::execution::runtime(x0.device(), x0.dtype());
        }
        x0 + (noise * self.schedule.start_sigma() as f64)?
    }

    /// The `scale_model_input` divisor at step `i`: `√(σ_i² + 1)` (latents are divided by this before
    /// the UNet, putting them in the model's expected scale).
    pub fn scale_in(&self, i: usize) -> f32 {
        self.schedule.scale_in(i)
    }

    /// The Euler step delta at step `i`: `σ_next − σ_cur` (the latent update is `x += ε · dt`).
    pub fn step_dt(&self, i: usize) -> f32 {
        self.schedule.step_dt(i)
    }
}

/// Forward evaluation seam for Candle Kolors native leading-Euler denoising (`SW-6966`, `PIPE-7`).
///
/// Generic over model prediction so Tier 0 `DriverSupport` conformance tests can drive
/// [`KolorsLeadingEulerDriver`] without loading weights or executing a learned forward.
pub trait KolorsLeadingEulerForward {
    /// Predict the raw noise residual ε for the (already CFG-batched when `use_guide` is true)
    /// scaled model input `model_in` at leading `timestep` and 0-based `step_index`.
    fn predict_eps(
        &self,
        model_in: &Tensor,
        timestep: f64,
        step_index: usize,
    ) -> CandleResult<Tensor>;
}

enum KolorsSamplerOwner<'a> {
    Borrowed(&'a KolorsEulerSampler),
    Owned(KolorsEulerSampler),
}

impl KolorsSamplerOwner<'_> {
    fn as_sampler(&self) -> &KolorsEulerSampler {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s,
        }
    }
}

/// Typed [`candle_gen::driver::FamilyDriver`] for all Candle native Kolors leading-Euler routes
/// (txt2img/img2img resident + staged, ControlNet-pose resident + staged, IP-Adapter resident +
/// staged) (`SW-6966`, `PIPE-7`, `PIPE-8`).
pub struct KolorsLeadingEulerDriver<'a, F> {
    sampler: KolorsSamplerOwner<'a>,
    use_guide: bool,
    guidance: f32,
    forward: F,
    support: candle_gen::driver::DriverSupport,
}

impl<'a, F: KolorsLeadingEulerForward> KolorsLeadingEulerDriver<'a, F> {
    fn build_support(driver_id: &'static str) -> candle_gen::driver::DriverSupport {
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
                    "native Candle Kolors leading-Euler driver does not re-noise from clean estimates; use curated shared driver",
                geometry_change:
                    "native Candle Kolors leading-Euler driver does not support geometry-changing low-res start; use curated shared driver",
                read_only_boundary_estimate:
                    "native Candle Kolors leading-Euler driver does not expose read-only boundary clean estimates",
                ..candle_gen::driver::DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(
        driver_id: &'static str,
        sampler: &'a KolorsEulerSampler,
        use_guide: bool,
        guidance: f32,
        forward: F,
    ) -> Self {
        Self {
            sampler: KolorsSamplerOwner::Borrowed(sampler),
            use_guide,
            guidance,
            forward,
            support: Self::build_support(driver_id),
        }
    }

    pub fn new_owned(
        driver_id: &'static str,
        sampler: KolorsEulerSampler,
        use_guide: bool,
        guidance: f32,
        forward: F,
    ) -> KolorsLeadingEulerDriver<'static, F> {
        KolorsLeadingEulerDriver {
            sampler: KolorsSamplerOwner::Owned(sampler),
            use_guide,
            guidance,
            forward,
            support: KolorsLeadingEulerDriver::<'static, F>::build_support(driver_id),
        }
    }
}

impl<F: KolorsLeadingEulerForward> candle_gen::driver::FamilyDriver<Tensor>
    for KolorsLeadingEulerDriver<'_, F>
{
    fn support(&self) -> &candle_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: candle_gen::driver::SliceInput<'_, Tensor>,
        state: &mut candle_gen::driver::DriverRunState,
        hooks: &mut candle_gen::driver::StepHooks<'_, Tensor>,
    ) -> gen_core::Result<gen_core::BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        state.enter_slice(&input)?;
        let sampler = self.sampler.as_sampler();
        if input.span.end > sampler.num_steps() {
            return Err(gen_core::Error::Unsupported(format!(
                "kolors leading-euler driver '{}': slice span {:?} exceeds {} steps",
                self.support.driver_id,
                input.span,
                sampler.num_steps()
            )));
        }

        let mut latents = match input.state {
            gen_core::BoundaryState::XSigma(x)
            | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let sigmas = sampler.sigmas();
        for global_step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let info = candle_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: global_step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: candle_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: sigmas[global_step],
                next_sigma: sigmas[global_step + 1],
            };
            let scaled =
                (&latents / sampler.scale_in(global_step) as f64).map_err(CandleError::from)?;
            hooks.observe(&info, &scaled)?;
            let model_in = if self.use_guide {
                Tensor::cat(&[&scaled, &scaled], 0).map_err(CandleError::from)?
            } else {
                scaled
            };
            let timestep = sampler.timestep(global_step) as f64;
            let eps = self.forward.predict_eps(&model_in, timestep, global_step)?;
            state.record_evaluation(candle_gen::driver::DriverEvaluationKind::Trajectory);
            // Classifier-free guidance: row 0 = uncond, row 1 = cond (`PIPE-8`).
            let eps = if self.use_guide {
                let chunks = eps.chunk(2, 0).map_err(CandleError::from)?;
                gen_core::guidance::cfg(
                    &candle_gen::CandleLatentOps,
                    &chunks[1],
                    &chunks[0],
                    self.guidance,
                )?
            } else {
                eps
            };
            let step_delta =
                (eps * sampler.step_dt(global_step) as f64).map_err(CandleError::from)?;
            latents = (&latents + step_delta).map_err(CandleError::from)?;
            latents = hooks.post_step(&info, latents)?;
            hooks.materialize(&info, &latents)?;
            state.record_outer_step(global_step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(gen_core::BoundaryState::XSigma(latents))
    }
}

/// Shared execution helper driving [`KolorsLeadingEulerDriver`] through
/// [`candle_gen::run_candle_driver_slices`] across all six native Candle Kolors routes.
///
/// When `run` is `Some`, executes every compiled `denoise_spans()` slice through the shared
/// driver runner so inert splits (e.g. `RenderPlan::with_inert_split_at`) enter and finish every
/// compiled phase before `finish_image_denoise_phases()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_leading_euler<F: KolorsLeadingEulerForward>(
    sampler: &KolorsEulerSampler,
    latents: Tensor,
    use_guide: bool,
    guidance: f32,
    forward: F,
    cancel: &gen_core::runtime::CancelFlag,
    run: Option<&gen_core::RenderRun<'_, '_>>,
    on_progress: &mut dyn FnMut(Progress),
    preview: &gen_core::PreviewSink,
) -> CandleResult<Tensor> {
    if sampler.num_steps() == 0 && run.is_none() {
        return Ok(latents);
    }
    let preview_counter = candle_gen::preview::PreviewCounter::with_steps(sampler.num_steps());
    let driver = KolorsLeadingEulerDriver::new(
        "kolors_candle_leading_euler",
        sampler,
        use_guide,
        guidance,
        forward,
    );
    let slices = candle_gen::driver::SliceSpec::from_render_run(run, sampler.num_steps());
    let mut observe = |info: &candle_gen::driver::DriverStepInfo, scaled: &Tensor| {
        candle_gen::preview::emit_preview_at(preview, &preview_counter, info.outer_step, || {
            crate::preview::project_spatial_latents(scaled)
        });
        Ok(())
    };
    let mut state = candle_gen::driver::DriverRunState::new();
    let mut hooks = candle_gen::driver::StepHooks::new(on_progress);
    hooks.on_observe = Some(&mut observe);
    let out = candle_gen::run_candle_driver_slices(
        &driver,
        slices,
        gen_core::BoundaryState::XSigma(latents),
        sampler.num_steps(),
        0,
        cancel,
        run,
        &mut state,
        &mut hooks,
        |_phase, boundary, _state| Ok(boundary),
    )?;
    Ok(match out {
        gen_core::BoundaryState::XSigma(x)
        | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_step_counts() {
        assert!(KolorsEulerSampler::new(0).is_err());
        assert!(KolorsEulerSampler::new(NUM_TRAIN_TIMESTEPS + 1).is_err());
        assert!(KolorsEulerSampler::new(1).is_ok());
        assert!(KolorsEulerSampler::new(NUM_TRAIN_TIMESTEPS).is_ok());
    }

    #[test]
    fn leading_timesteps_descend_with_offset() {
        // N=50, step_ratio = 1100/50 = 22, leading timesteps = (j·22)[::-1] + 1.
        let s = KolorsEulerSampler::new(50).unwrap();
        assert_eq!(s.num_steps(), 50);
        // First (j=49): 49·22 + 1 = 1079; last (j=0): 0·22 + 1 = 1.
        assert!((s.timestep(0) - 1079.0).abs() < 1e-4, "{}", s.timestep(0));
        assert!((s.timestep(49) - 1.0).abs() < 1e-4, "{}", s.timestep(49));
        for i in 0..49 {
            assert!(s.timestep(i) > s.timestep(i + 1), "timesteps must descend");
        }
    }

    #[test]
    fn sigmas_descend_to_zero_and_scale_in_starts_above_one() {
        let s = KolorsEulerSampler::new(30).unwrap();
        // sigmas has length num_steps + 1, trailing 0.
        assert_eq!(s.sigmas().len(), 31);
        assert!(s.sigmas()[30].abs() < 1e-9, "final sigma is 0");
        for w in s.sigmas().windows(2) {
            assert!(w[0] >= w[1], "sigmas must be non-increasing");
        }
        // init_noise_sigma = sqrt(max_sigma^2 + 1) > max_sigma > 1 (the Karras sigmas exceed 1 near t=1100).
        assert!(s.init_noise_sigma() > 1.0);
        // scale_in at step 0 = sqrt(sigma_0^2 + 1) >= 1, and is 1 at the trailing zero sigma.
        assert!(s.scale_in(0) > 1.0);
    }

    #[test]
    fn euler_step_dt_is_negative() {
        // The schedule descends, so every Euler delta (σ_next − σ_cur) is ≤ 0.
        let s = KolorsEulerSampler::new(10).unwrap();
        for i in 0..s.num_steps() {
            assert!(s.step_dt(i) <= 0.0, "dt at {i} should be <= 0");
        }
    }

    #[test]
    fn img2img_strength_selects_the_schedule_tail_and_start_sigma() {
        let full = KolorsEulerSampler::new(10).unwrap();
        let half = KolorsEulerSampler::img2img(10, 0.5).unwrap();
        assert_eq!(half.num_steps(), 5);
        assert_eq!(half.schedule.timesteps(), &full.schedule.timesteps()[5..]);
        assert_eq!(half.sigmas(), &full.sigmas()[5..]);

        let zero = KolorsEulerSampler::img2img(10, 0.0).unwrap();
        assert_eq!(zero.num_steps(), 0);
        assert_eq!(zero.sigmas(), &[0.0]);

        let over = KolorsEulerSampler::img2img(10, 2.0).unwrap();
        assert_eq!(over.schedule.timesteps(), full.schedule.timesteps());
        assert_eq!(over.sigmas(), full.sigmas());
    }

    #[test]
    fn kolors_leading_euler_driver_passes_tier0_conformance_without_weights() {
        use candle_gen::candle_core::Device;
        use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        struct FakeLinearForward;
        impl KolorsLeadingEulerForward for FakeLinearForward {
            fn predict_eps(
                &self,
                model_in: &Tensor,
                timestep: f64,
                step_index: usize,
            ) -> CandleResult<Tensor> {
                // Verify [uncond, cond] batching (row 0 = uncond, row 1 = cond) when guided.
                let chunks = model_in.chunk(2, 0)?;
                let uncond = chunks[0].affine(0.03 + 0.0001 * timestep, 0.0)?;
                let cond =
                    chunks[1].affine(0.07 + 0.0001 * timestep + 0.005 * step_index as f64, 0.0)?;
                Ok(Tensor::cat(&[&uncond, &cond], 0)?)
            }
        }

        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 6,
            split_at: 3,
            seed: 19,
            base_geometry: (1024, 1024),
            alt_geometry: (512, 512),
            make_driver: || {
                let sampler = KolorsEulerSampler::new(6).unwrap();
                Box::new(KolorsLeadingEulerDriver::new_owned(
                    "kolors_candle_leading_euler_tier0",
                    sampler,
                    true,
                    5.0,
                    FakeLinearForward,
                ))
            },
            make_initial: || {
                Tensor::from_vec(vec![0.4f32, -0.3, 0.2, 0.8], (1, 4, 1, 1), &Device::Cpu).unwrap()
            },
            latents_equal: |a, b| {
                a.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    == b.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            },
        })
        .expect("KolorsLeadingEulerDriver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_carry_refusal_or_support);
        assert!(report.verified_reset_preserves_draws);
        assert!(report.verified_boundary_estimate_readonly_or_refusal);
        assert!(report.verified_renoise_and_resample_guards);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn native_kolors_planned_inert_split_finishes_every_compiled_phase() {
        use candle_gen::candle_core::Device;

        struct FakeLinearForward;
        impl KolorsLeadingEulerForward for FakeLinearForward {
            fn predict_eps(
                &self,
                model_in: &Tensor,
                timestep: f64,
                step_index: usize,
            ) -> CandleResult<Tensor> {
                let chunks = model_in.chunk(2, 0)?;
                let uncond = chunks[0].affine(0.03 + 0.0001 * timestep, 0.0)?;
                let cond =
                    chunks[1].affine(0.07 + 0.0001 * timestep + 0.005 * step_index as f64, 0.0)?;
                Ok(Tensor::cat(&[&uncond, &cond], 0)?)
            }
        }

        let sampler = KolorsEulerSampler::new(6).unwrap();
        let init =
            Tensor::from_vec(vec![0.4f32, -0.3, 0.2, 0.8], (1, 4, 1, 1), &Device::Cpu).unwrap();
        let cancel = gen_core::runtime::CancelFlag::new();
        let preview = gen_core::PreviewSink::default();

        let registry = crate::provider_registry().unwrap();
        let spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir("/tmp/kolors".into()));
        let req = gen_core::GenerationRequest {
            prompt: "a lantern over water".into(),
            width: 1024,
            height: 1024,
            steps: Some(6),
            guidance: Some(5.0),
            sampler: Some("euler_discrete".into()),
            seed: Some(19),
            ..Default::default()
        };
        let plan = registry
            .compile_render_plan(crate::MODEL_ID, &spec, &req)
            .expect("compile native kolors render plan")
            .expect("native kolors produces a RenderPlan")
            .with_inert_split_at(3)
            .expect("inert split at step 3");
        assert_eq!(plan.denoise_spans(), vec![0..3, 3..6]);

        let unsplit = run_leading_euler(
            &sampler,
            init.clone(),
            true,
            5.0,
            FakeLinearForward,
            &cancel,
            None,
            &mut |_| {},
            &preview,
        )
        .unwrap();

        let run = gen_core::RenderRun::new(&plan, &cancel, None);
        run.encode(|| Ok::<_, CandleError>(())).unwrap();
        run.begin_rendering().unwrap();
        run.reset_image_denoise_phases();
        let mut progress_steps = Vec::new();
        let split = run_leading_euler(
            &sampler,
            init,
            true,
            5.0,
            FakeLinearForward,
            &cancel,
            Some(&run),
            &mut |p| {
                if let Progress::Step { current, total } = p {
                    progress_steps.push((current, total));
                }
            },
            &preview,
        )
        .unwrap();
        run.finish_image_denoise_phases()
            .expect("planned inert split must complete every compiled denoise phase");
        run.finish_rendering();

        assert_eq!(
            progress_steps,
            vec![(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        );
        assert_eq!(
            split.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            unsplit.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "planned inert split must be bit-identical to unsplit native Kolors trajectory"
        );
    }
}
