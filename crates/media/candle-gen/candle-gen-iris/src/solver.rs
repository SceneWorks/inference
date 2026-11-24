//! FlowDPM-Solver++ (`iris3b/flow/solver.py`) applied to Candle tensors. The schedule and every
//! scalar coefficient come from the backend-neutral plan ([`gen_core_iris::dpm_solver_plan`]); this
//! module only applies them, exactly as `mlx_gen_iris::solver` does. The integration state (`x`, the
//! `x0` history) is **f32** throughout, as upstream keeps it (`z.to(float32)`, `out.x.float()`).
//!
//! [`gen_core_iris::dpm_solver_plan`]: gen_core_iris::dpm_solver_plan

use std::cell::RefCell;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use candle_gen::gen_core::{self, BoundaryState, CancelFlag, EstimateSource};
use candle_gen::Result;
use gen_core_iris::{DpmStep, DpmUpdate, Prediction};

/// `x · c` with `c` an f32 scalar tensor — an elementwise f32 multiply, the same rounding as the
/// tensor-by-scalar products upstream and MLX perform.
fn scale(x: &Tensor, c: f32) -> Result<Tensor> {
    let c = Tensor::new(c, x.device())?;
    Ok(x.broadcast_mul(&c)?)
}

/// Typed `FamilyDriver` for Candle Iris pixel-space FlowDPM-Solver++ (`SW-6969`).
///
/// Carries the 2nd-order multistep `prev_x0` history inside request-owned [`DriverRunState`] so
/// inert `Continue + Carry` splits match unsplit trajectories bit-for-bit, and refuses
/// `Continue + Reset` on 2nd-order plans with a typed `DEC-13` refusal.
pub struct IrisFlowDpmDriver<F> {
    support: DriverSupport,
    plan: Vec<DpmStep>,
    prediction: Prediction,
    forward: RefCell<F>,
}

impl<F> IrisFlowDpmDriver<F>
where
    F: FnMut(&Tensor, &DpmStep) -> Result<Tensor>,
{
    /// Construct a Candle Iris FlowDPM-Solver++ driver over `plan` and `prediction`.
    pub fn new(plan: Vec<DpmStep>, prediction: Prediction, forward: F) -> Self {
        let first_order_only = plan
            .iter()
            .all(|step| matches!(step.update, DpmUpdate::First { .. }));
        Self {
            support: DriverSupport {
                driver_id: "candle_iris_flow_dpm",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: first_order_only,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: true,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    reset_history: "Iris FlowDPM-Solver++ 2nd-order multistep plan requires carrying prev_x0 history across slices",
                    renoise_from_clean: "Iris FlowDPM-Solver++ is a deterministic pixel-space ODE solver and refuses external Renoise transitions",
                    geometry_change: "Iris FlowDPM-Solver++ does not support mid-trajectory spatial resampling",
                    ..DriverRefusalReasons::default()
                },
            },
            plan,
            prediction,
            forward: RefCell::new(forward),
        }
    }

    fn pred_x0(&self, x: &Tensor, out: &Tensor, step: &DpmStep) -> gen_core::Result<Tensor> {
        match self.prediction {
            // v-prediction: x0 = x − s·v
            Prediction::Velocity => {
                let sv = scale(out, step.s_f32()).map_err(gen_core::Error::from)?;
                x.sub(&sv).map_err(|e| gen_core::Error::Msg(e.to_string()))
            }
            Prediction::Clean => Ok(out.clone()),
        }
    }
}

impl<F> FamilyDriver<Tensor> for IrisFlowDpmDriver<F>
where
    F: FnMut(&Tensor, &DpmStep) -> Result<Tensor>,
{
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
        if input.total_steps != self.plan.len() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: total_steps={} does not match FlowDPM plan len={}",
                self.support.driver_id,
                input.total_steps,
                self.plan.len()
            )));
        }
        state.enter_slice(&input)?;

        let initial = match input.state {
            BoundaryState::XSigma(t)
            | BoundaryState::CleanEstimate { latent: t, .. }
            | BoundaryState::RouteSpecific { state: t, .. } => t,
        };
        let mut x = initial
            .to_dtype(DType::F32)
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        let mut prev_x0 = state.take_history::<Tensor>();

        for i in input.span.clone() {
            if input.cancel.is_cancelled() {
                if let Some(prev) = prev_x0 {
                    state.set_history(prev);
                }
                return Err(gen_core::Error::Canceled);
            }
            let step = &self.plan[i];
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: i,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: step.s as f32,
                next_sigma: step.t as f32,
            };
            let out = (self.forward.borrow_mut())(&x, step)
                .map_err(gen_core::Error::from)?
                .to_dtype(DType::F32)
                .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            let x0 = self.pred_x0(&x, &out, step)?;
            x = match step.update {
                DpmUpdate::First { cx, c0 } => {
                    let lhs = scale(&x, cx).map_err(gen_core::Error::from)?;
                    let rhs = scale(&x0, c0).map_err(gen_core::Error::from)?;
                    lhs.sub(&rhs)
                        .map_err(|e| gen_core::Error::Msg(e.to_string()))?
                }
                DpmUpdate::Second { cx, c0, c1, r0 } => {
                    let prev = prev_x0.as_ref().ok_or_else(|| {
                        gen_core::Error::Msg(
                            "iris solver: a second-order step needs the previous x0".into(),
                        )
                    })?;
                    let d = x0
                        .sub(prev)
                        .and_then(|diff| diff.broadcast_div(&Tensor::new(r0, x0.device())?))
                        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
                    let t1 = scale(&x, cx).map_err(gen_core::Error::from)?;
                    let t2 = scale(&x0, c0).map_err(gen_core::Error::from)?;
                    let t3 = scale(&d, c1).map_err(gen_core::Error::from)?;
                    t1.sub(&t2)
                        .and_then(|v| v.sub(&t3))
                        .map_err(|e| gen_core::Error::Msg(e.to_string()))?
                }
            };
            hooks.observe(&info, &x0)?;
            x = hooks.post_step(&info, x)?;
            hooks.materialize(&info, &x)?;
            prev_x0 = Some(x0);
            state.record_outer_step(i);
            hooks.emit_progress(&info);
        }
        if let Some(prev) = prev_x0 {
            state.set_history(prev);
        }
        state.finish_slice(input.span.end);

        if input.request_boundary_estimate {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let est = if input.span.end < self.plan.len() {
                let step = &self.plan[input.span.end];
                let out = (self.forward.borrow_mut())(&x, step)
                    .map_err(gen_core::Error::from)?
                    .to_dtype(DType::F32)
                    .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                self.pred_x0(&x, &out, step)?
            } else {
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                x.clone()
            };
            return Ok(BoundaryState::CleanEstimate {
                latent: est,
                source: EstimateSource::EndpointEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(x))
    }
}

/// Integrate `z` along `plan`. `model_out(x, step)` returns the raw (CFG-combined) network output
/// at `step.s`, read as `prediction` (`_pred_x0`: velocity `x0 = x − s·out`, or the clean image
/// `x0 = out`); `on_step(i, x0)` fires after step `i` (1-based) completes with that step's predicted
/// clean image. Cancellation is checked before every network evaluation and surfaces as
/// [`Error::Canceled`] — the partial state is dropped, never returned.
pub fn sample(
    z: &Tensor,
    plan: &[DpmStep],
    prediction: Prediction,
    cancel: &CancelFlag,
    model_out: impl FnMut(&Tensor, &DpmStep) -> Result<Tensor>,
    mut on_step: impl FnMut(usize, &Tensor),
) -> Result<Tensor> {
    let x_init = z.to_dtype(DType::F32)?;
    let times: Vec<f64> = plan.iter().flat_map(|step| [step.s, step.t]).collect();
    candle_gen::execution::schedule_f64("iris-flow-dpm", &times, &format!("{prediction:?}"));
    candle_gen::execution::runtime(x_init.device(), x_init.dtype());
    candle_gen::execution::policy("iris_solver_coefficients", format!("{plan:?}").into());
    let total_steps = plan.len();
    let driver = IrisFlowDpmDriver::new(plan.to_vec(), prediction, model_out);
    let mut run_state = DriverRunState::new();
    let mut on_progress = |_: gen_core::Progress| {};
    let mut on_observe = |info: &DriverStepInfo, x0: &Tensor| {
        on_step(info.outer_step + 1, x0);
        Ok(())
    };
    let mut hooks = StepHooks::new(&mut on_progress);
    hooks.on_observe = Some(&mut on_observe);
    let out = run_candle_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(x_init),
        total_steps,
        0,
        cancel,
        None,
        &mut run_state,
        &mut hooks,
        |_, state, _| Ok(state),
    )?;
    Ok(match out {
        BoundaryState::XSigma(x)
        | BoundaryState::CleanEstimate { latent: x, .. }
        | BoundaryState::RouteSpecific { state: x, .. } => x,
    })
}

/// `out_uncond + scale · (out_cond − out_uncond)` in f32.
pub fn cfg_combine(out_uncond: &Tensor, out_cond: &Tensor, cfg_scale: f32) -> Result<Tensor> {
    let u = out_uncond.to_dtype(DType::F32)?;
    let c = out_cond.to_dtype(DType::F32)?;
    Ok(u.add(&scale(&c.sub(&u)?, cfg_scale)?)?)
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn tensors_bit_equal(a: &Tensor, b: &Tensor) -> bool {
        let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        va.len() == vb.len()
            && va
                .iter()
                .zip(vb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn candle_iris_flow_dpm_driver_passes_tier0_conformance_for_order2_and_order1() {
        for (order, expect_reset) in [(2usize, false), (1usize, true)] {
            let plan = gen_core_iris::dpm_solver_plan(4, order, 3.0).unwrap();
            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 4,
                split_at: 2,
                seed: 29,
                base_geometry: (512, 512),
                alt_geometry: (1024, 1024),
                make_driver: {
                    let plan = plan.clone();
                    move || {
                        Box::new(IrisFlowDpmDriver::new(
                            plan.clone(),
                            Prediction::Velocity,
                            |x, step| scale(x, 0.15 + 0.1 * step.s_f32()),
                        ))
                    }
                },
                make_initial: || {
                    Tensor::from_vec(
                        vec![
                            0.5f32, -0.25, 0.75, -0.5, 0.1, -0.3, 0.2, 0.4, -0.1, 0.3, -0.2, 0.6,
                        ],
                        (1, 3, 2, 2),
                        &Device::Cpu,
                    )
                    .unwrap()
                },
                latents_equal: tensors_bit_equal,
            })
            .expect("Candle IrisFlowDpmDriver must pass Tier 0 conformance");
            assert!(report.verified_inert_split);
            assert!(report.verified_carry_refusal_or_support);
            assert_eq!(report.verified_reset_preserves_draws, expect_reset);
            assert!(report.verified_boundary_estimate_readonly_or_refusal);
            assert!(report.verified_renoise_and_resample_guards);
            assert!(report.verified_cancellation);
            assert_eq!(report.unsplit_evaluations, 4);
        }
    }
}
