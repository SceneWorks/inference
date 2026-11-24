//! Upstream's sampling loop (`inference.py::generate`), routed through the typed
//! [`FamilyDriver`] contract ([`Supra2EulerDriver`], `SW-6970`): uniform Euler integration of the
//! flow ODE from noise (`t = 0`) to data, with batched `2B` classifier-free guidance via
//! [`gen_core::guidance::cfg`] (`PIPE-8`) when an unconditional context is supplied.

use gen_core_checkpoint::supra2::{euler_dt, euler_timesteps};
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use mlx_gen::gen_core;
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_rs::ops::{add, concatenate_axis, multiply, split};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::dit::SupraDit;

/// A batch of text conditioning: context `[B, M, ctx_dim]` and its additive mask `[B, 1, 1, M]`.
#[derive(Clone, Copy)]
pub struct ContextBatch<'a> {
    pub context: &'a Array,
    pub mask: &'a Array,
}

/// Repeat one prompt's `[1, M, D]` context and 0/1 mask `n` times (upstream's
/// `[prompt] * n` / `.expand(n, …)`), returning the context and its additive mask.
pub fn broadcast(context: &Array, mask01: &[f32], n: usize) -> Result<(Array, Array)> {
    let ctx = concatenate_axis(&vec![context.clone(); n], 0)?;
    let mask = SupraDit::additive_mask(&mask01.repeat(n), n, mask01.len())?;
    Ok((ctx, mask))
}

/// Forward evaluation seam for MLX Supra2 uniform-Euler flow denoising (`SW-6970`, `PIPE-7`).
///
/// Generic over raw conditional + optional unconditional velocity prediction so weights-free
/// Tier 0 `DriverSupport` conformance tests can drive [`Supra2EulerDriver`] over mocked forwards
/// without loading model weights.
pub trait Supra2VelocityForward {
    /// Predict the conditional velocity and optional unconditional velocity `[B, C, S, S]` at flow
    /// time `t` and 0-based `step_index`. When `Some(vu)` is returned, [`Supra2EulerDriver`]
    /// combines `(vc, vu)` through shared [`gen_core::guidance::cfg`] (`PIPE-8`).
    fn predict_velocity(
        &self,
        z: &Array,
        t: f32,
        step_index: usize,
    ) -> Result<(Array, Option<Array>)>;
}

fn predict_velocity_with_dit(
    dit: &SupraDit,
    z: &Array,
    t: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
) -> Result<(Array, Option<Array>)> {
    let n = z.shape()[0] as usize;
    match uncond {
        Some(u) => {
            let z2 = concatenate_axis(&[z, z], 0)?;
            let tf = dit.timestep_features(&vec![t; 2 * n]);
            let ctx = concatenate_axis(&[cond.context, u.context], 0)?;
            let mask = concatenate_axis(&[cond.mask, u.mask], 0)?;
            let both = dit.forward(&z2, &tf, &ctx, &mask)?;
            let mut halves = split(&both, 2, 0)?;
            let vu = halves.remove(1);
            let vc = halves.remove(0);
            Ok((vc, Some(vu)))
        }
        None => {
            let tf = dit.timestep_features(&vec![t; n]);
            Ok((dit.forward(z, &tf, cond.context, cond.mask)?, None))
        }
    }
}

/// One velocity evaluation, upstream's loop body: with `uncond`, the conditional and unconditional
/// halves run as one `2B` batch and mix via [`gen_core::guidance::cfg`] (`v_u + g·(v_c − v_u)`);
/// without, a plain conditional pass.
pub fn velocity(
    dit: &SupraDit,
    z: &Array,
    t: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
    guidance: f32,
) -> Result<Array> {
    let (vc, vu) = predict_velocity_with_dit(dit, z, t, cond, uncond)?;
    match vu {
        Some(vu) => Ok(gen_core::guidance::cfg(
            &mlx_gen::MlxLatentOps,
            &vc,
            &vu,
            guidance,
        )?),
        None => Ok(vc),
    }
}

/// Typed [`FamilyDriver`] for MLX Supra2-IMG uniform-Euler rectified-flow denoising (`SW-6970`,
/// `PIPE-7`, `PIPE-8`).
pub struct Supra2EulerDriver<F> {
    timesteps: Vec<f32>,
    dt: f32,
    guidance: f32,
    forward: F,
    support: DriverSupport,
}

impl<F: Supra2VelocityForward> Supra2EulerDriver<F> {
    fn build_support(driver_id: &'static str) -> DriverSupport {
        DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons {
                renoise_from_clean:
                    "native MLX Supra2 Euler driver does not re-noise from clean boundary estimates",
                geometry_change:
                    "native MLX Supra2 Euler driver runs at fixed 256x256 geometry and refuses geometry changes",
                read_only_boundary_estimate:
                    "native MLX Supra2 Euler driver does not expose read-only boundary clean estimates",
                ..DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(steps: u32, guidance: f32, forward: F) -> Result<Self> {
        if steps == 0 {
            return Err(Error::Unsupported(
                "supra2: steps must be at least 1".into(),
            ));
        }
        Ok(Self {
            timesteps: euler_timesteps(steps),
            dt: euler_dt(steps),
            guidance,
            forward,
            support: Self::build_support("mlx_supra2_euler"),
        })
    }

    /// Number of uniform Euler steps in this schedule.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }
}

impl<F: Supra2VelocityForward> FamilyDriver<Array> for Supra2EulerDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> gen_core::Result<gen_core::BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        state.enter_slice(&input)?;
        if input.span.end > self.timesteps.len() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: slice span {:?} exceeds {} steps",
                self.support.driver_id,
                input.span,
                self.timesteps.len()
            )));
        }

        let mut z = match input.state {
            gen_core::BoundaryState::XSigma(x)
            | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let dt = Array::from_f32(self.dt);

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let t = self.timesteps[step];
            let next_t = if step + 1 < self.timesteps.len() {
                self.timesteps[step + 1]
            } else {
                1.0
            };
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: t,
                next_sigma: next_t,
            };
            hooks.observe(&info, &z)?;

            let (vc, vu) = self
                .forward
                .predict_velocity(&z, t, step)
                .map_err(gen_core::Error::from)?;
            let v = match vu {
                Some(vu) => {
                    gen_core::guidance::cfg(&mlx_gen::MlxLatentOps, &vc, &vu, self.guidance)?
                }
                None => vc,
            };
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            let step_delta =
                multiply(&dt, &v).map_err(|e| gen_core::Error::from(Error::from(e)))?;
            z = add(&z, &step_delta).map_err(|e| gen_core::Error::from(Error::from(e)))?;
            z = hooks.post_step(&info, z)?;
            hooks.materialize(&info, &z)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(gen_core::BoundaryState::XSigma(z))
    }
}

struct ProductionSupra2Forward<'a> {
    dit: &'a SupraDit,
    cond: ContextBatch<'a>,
    uncond: Option<ContextBatch<'a>>,
}

impl Supra2VelocityForward for ProductionSupra2Forward<'_> {
    fn predict_velocity(
        &self,
        z: &Array,
        t: f32,
        _step_index: usize,
    ) -> Result<(Array, Option<Array>)> {
        predict_velocity_with_dit(self.dit, z, t, self.cond, self.uncond)
    }
}

/// Integrate `z0` over `steps` uniform Euler steps (`z += dt · v`) via [`Supra2EulerDriver`].
/// CFG runs iff `uncond` is supplied — callers pass it exactly when
/// [`cfg_enabled`](gen_core_checkpoint::supra2::cfg_enabled) holds for `guidance`. Checks `cancel`
/// before every step and reports 1-based step progress.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &SupraDit,
    z0: Array,
    steps: u32,
    guidance: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let driver = Supra2EulerDriver::new(
        steps,
        guidance,
        ProductionSupra2Forward { dit, cond, uncond },
    )?;
    let total_steps = steps as usize;
    let mut state = DriverRunState::new();
    let mut on_materialize = |_info: &DriverStepInfo, latent: &Array| -> gen_core::Result<()> {
        eval([latent]).map_err(|err| gen_core::Error::from(Error::from(err)))?;
        Ok(())
    };
    let mut hooks = StepHooks::new(progress);
    hooks.on_materialize = Some(&mut on_materialize);
    let out = run_mlx_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        gen_core::BoundaryState::XSigma(z0),
        total_steps,
        0,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
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
    use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};
    use mlx_rs::ops::subtract;

    struct MockSupra2Forward {
        with_uncond: bool,
    }

    impl Supra2VelocityForward for MockSupra2Forward {
        fn predict_velocity(
            &self,
            z: &Array,
            t: f32,
            step_index: usize,
        ) -> Result<(Array, Option<Array>)> {
            let scale_c = Array::from_f32(0.2 + 0.1 * t + 0.01 * (step_index as f32));
            let vc = add(multiply(z, &scale_c)?, Array::from_f32(0.15))?;
            if !self.with_uncond {
                return Ok((vc, None));
            }
            let scale_u = Array::from_f32(-0.05 + 0.04 * t - 0.005 * (step_index as f32));
            let vu = add(multiply(z, &scale_u)?, Array::from_f32(-0.05))?;
            Ok((vc, Some(vu)))
        }
    }

    fn arrays_bit_equal(a: &Array, b: &Array) -> bool {
        let a = mlx_gen::array::contiguous(a).unwrap();
        let b = mlx_gen::array::contiguous(b).unwrap();
        eval([&a, &b]).unwrap();
        let sa = a.as_slice::<f32>();
        let sb = b.as_slice::<f32>();
        sa.len() == sb.len() && sa.iter().zip(sb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn supra2_euler_driver_cfg_passes_tier0_conformance() {
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 6,
            split_at: 3,
            seed: 13,
            base_geometry: (256, 256),
            alt_geometry: (512, 512),
            make_driver: || {
                Box::new(
                    Supra2EulerDriver::new(6, 3.0, MockSupra2Forward { with_uncond: true })
                        .unwrap(),
                )
            },
            make_initial: || Array::from_slice(&[0.25f32, -0.5, 0.75, -1.0], &[1, 1, 2, 2]),
            latents_equal: arrays_bit_equal,
        })
        .expect("MLX Supra2 CFG driver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_carry_refusal_or_support);
        assert!(report.verified_reset_preserves_draws);
        assert!(report.verified_boundary_estimate_readonly_or_refusal);
        assert!(report.verified_renoise_and_resample_guards);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn supra2_euler_driver_unguided_passes_tier0_conformance() {
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 29,
            base_geometry: (256, 256),
            alt_geometry: (512, 512),
            make_driver: || {
                Box::new(
                    Supra2EulerDriver::new(4, 1.0, MockSupra2Forward { with_uncond: false })
                        .unwrap(),
                )
            },
            make_initial: || Array::from_slice(&[-0.4f32, 0.6, 0.1, -0.8], &[1, 1, 2, 2]),
            latents_equal: arrays_bit_equal,
        })
        .expect("MLX Supra2 unguided driver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn supra2_euler_driver_preserves_euler_cfg_and_zero_step_refusal_bit_exact() {
        assert!(matches!(
            Supra2EulerDriver::new(0, 3.0, MockSupra2Forward { with_uncond: true }),
            Err(Error::Unsupported(_))
        ));

        let steps = 5u32;
        let guidance = 3.5f32;
        let z0 = Array::from_slice(&[0.5f32, -0.25, 0.75, -1.25], &[1, 1, 2, 2]);
        let mock = MockSupra2Forward { with_uncond: true };

        let dt = Array::from_f32(euler_dt(steps));
        let g = Array::from_f32(guidance);
        let mut expected = z0.clone();
        for (i, t) in euler_timesteps(steps).into_iter().enumerate() {
            let (vc, vu) = mock.predict_velocity(&expected, t, i).unwrap();
            let vu = vu.unwrap();
            let v = add(&vu, multiply(&g, subtract(&vc, &vu).unwrap()).unwrap()).unwrap();
            expected = add(&expected, multiply(&dt, &v).unwrap()).unwrap();
            eval([&expected]).unwrap();
        }

        let driver =
            Supra2EulerDriver::new(steps, guidance, MockSupra2Forward { with_uncond: true })
                .unwrap();
        let cancel = CancelFlag::new();
        let mut state = DriverRunState::new();
        let mut mat_count = 0usize;
        let mut on_progress = |_p: Progress| {};
        let mut on_materialize = |_info: &DriverStepInfo, latent: &Array| -> gen_core::Result<()> {
            mat_count += 1;
            eval([latent]).unwrap();
            Ok(())
        };
        let mut hooks = StepHooks::new(&mut on_progress);
        hooks.on_materialize = Some(&mut on_materialize);
        let out = run_mlx_driver_slices(
            &driver,
            [SliceSpec::full(steps as usize)],
            gen_core::BoundaryState::XSigma(z0),
            steps as usize,
            0,
            &cancel,
            None,
            &mut state,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let got = match out {
            gen_core::BoundaryState::XSigma(x) => x,
            _ => panic!("expected XSigma"),
        };
        assert_eq!(mat_count, steps as usize);
        assert!(
            arrays_bit_equal(&got, &expected),
            "MLX Supra2 driver must be bit-identical to pre-migration Euler+CFG loop"
        );
    }
}
