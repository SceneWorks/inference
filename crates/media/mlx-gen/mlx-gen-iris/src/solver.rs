//! FlowDPM-Solver++ (`iris3b/flow/solver.py`) applied to MLX tensors. The schedule and every
//! scalar coefficient come from the backend-neutral plan ([`gen_core_iris::dpm_solver_plan`]); this
//! module only applies them. The integration state (`x`, the `x0` history) is **f32** throughout,
//! as upstream keeps it (`z.to(float32)`, `out.x.float()`).

use std::cell::RefCell;

use gen_core_iris::{DpmStep, DpmUpdate, Prediction};
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use mlx_gen::gen_core::{BoundaryState, EstimateSource};
use mlx_gen::{CancelFlag, Error, MlxLatentOps, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Label of one solver update's iOS execution quantum.
pub(crate) const STEP_QUANTUM: &str = "Iris solver step";
/// Host-owned canonical plan and preview sigma payload, checked before allocation.
pub(crate) const PLAN_QUANTUM: &str = "Iris solver plan";
/// The final pixel-space clamp, evaluated before the solve returns and releases its text batches.
pub(crate) const FINALIZE_QUANTUM: &str = "Iris solver final clamp";

/// `clip(x, -1, 1)` is a maximum followed by a minimum: two f32 arrays and two f32 bounds.
pub(crate) fn final_clamp_scratch(elements: u64) -> u64 {
    2 * crate::retention::allocated(mlx_gen::ios_quantum::tensor_bytes(4, &[elements])) + 8
}

/// Typed `FamilyDriver` for Iris pixel-space FlowDPM-Solver++ (`SW-6969`).
///
/// Carries the 2nd-order multistep `prev_x0` history inside request-owned [`DriverRunState`] so
/// inert `Continue + Carry` splits match unsplit trajectories bit-for-bit, refuses `Continue + Reset`
/// on 2nd-order plans with a typed `DEC-13` error, and runs paired per-step `STEP_QUANTUM` iOS
/// execution barriers when constructed with diagnostics enabled.
pub struct IrisFlowDpmDriver<F> {
    support: DriverSupport,
    plan: Vec<DpmStep>,
    prediction: Prediction,
    emit_diagnostics: bool,
    forward: RefCell<F>,
}

/// Primary [`DriverSupport`] declaration for Iris pixel-space FlowDPM-Solver++.
pub(crate) fn iris_flow_dpm_driver_support(first_order_only: bool) -> DriverSupport {
    DriverSupport {
        driver_id: "iris_flow_dpm",
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
    }
}

impl<F> IrisFlowDpmDriver<F>
where
    F: FnMut(&Array, &DpmStep) -> Result<Array>,
{
    /// Construct an Iris FlowDPM-Solver++ driver over `plan` and `prediction`.
    pub fn new(plan: Vec<DpmStep>, prediction: Prediction, forward: F) -> Self {
        Self::with_diagnostics(plan, prediction, false, forward)
    }

    pub(crate) fn with_diagnostics(
        plan: Vec<DpmStep>,
        prediction: Prediction,
        emit_diagnostics: bool,
        forward: F,
    ) -> Self {
        let first_order_only = plan
            .iter()
            .all(|step| matches!(step.update, DpmUpdate::First { .. }));
        Self {
            support: iris_flow_dpm_driver_support(first_order_only),
            plan,
            prediction,
            emit_diagnostics,
            forward: RefCell::new(forward),
        }
    }

    fn pred_x0(&self, x: &Array, out: &Array, step: &DpmStep) -> mlx_gen::gen_core::Result<Array> {
        match self.prediction {
            Prediction::Velocity => x
                .subtract(
                    &out.multiply(Array::from_f32(step.s_f32()))
                        .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?,
                )
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string())),
            Prediction::Clean => Ok(out.clone()),
        }
    }
}

impl<F> FamilyDriver<Array> for IrisFlowDpmDriver<F>
where
    F: FnMut(&Array, &DpmStep) -> Result<Array>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps != self.plan.len() {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: total_steps={} does not match FlowDPM plan len={}",
                self.support.driver_id,
                input.total_steps,
                self.plan.len()
            )));
        }
        state.enter_slice(&input)?;

        let initial = match input.state {
            BoundaryState::XSigma(x)
            | BoundaryState::CleanEstimate { latent: x, .. }
            | BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let mut x = initial
            .as_dtype(Dtype::Float32)
            .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
        let mut prev_x0 = state.take_history::<Array>();

        for i in input.span.clone() {
            if input.cancel.is_cancelled() {
                if let Some(prev) = prev_x0 {
                    state.set_history(prev);
                }
                return Err(mlx_gen::gen_core::Error::Canceled);
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
            if self.emit_diagnostics {
                mlx_gen::ios_quantum::diagnostic_metadata(
                    crate::model::MODEL_ID,
                    "solver",
                    &[
                        ("step", i as u64),
                        ("steps", self.plan.len() as u64),
                        ("first", u64::from(i == 0)),
                        ("terminal", u64::from(i + 1 == self.plan.len())),
                        (
                            "update_order",
                            match step.update {
                                DpmUpdate::First { .. } => 1,
                                DpmUpdate::Second { .. } => 2,
                            },
                        ),
                        ("history_present", u64::from(prev_x0.is_some())),
                        (
                            "history_rows",
                            prev_x0
                                .as_ref()
                                .map_or(0, |previous| previous.shape()[0] as u64),
                        ),
                        ("state_rows", x.shape()[0] as u64),
                        ("state_f32", u64::from(x.dtype() == Dtype::Float32)),
                        // This is completed-loop evidence, not a native availability assertion on lazy state.
                        ("prior_completed_steps", i as u64),
                    ],
                );
                #[cfg(all(test, target_os = "macos"))]
                crate::anchor_tests::observe_solver_epoch(i, step, &x, prev_x0.as_ref());
            }
            let out = (self.forward.borrow_mut())(&x, step)?
                .as_dtype(Dtype::Float32)
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            if self.emit_diagnostics {
                mlx_gen::ios_quantum::before(
                    STEP_QUANTUM,
                    step_scratch(x.size() as u64, &step.update),
                )?;
            }
            let (next, x0) = step_update(&x, &out, step, self.prediction, prev_x0.as_ref())?;
            x = next;
            eval([&x, &x0]).map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            if self.emit_diagnostics {
                mlx_gen::ios_quantum::finish(STEP_QUANTUM, &[&x, &x0])?;
            }
            hooks.materialize(&info, &x)?;
            hooks.observe(&info, &x0)?;
            x = hooks.post_step(&info, x)?;
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
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let est = if input.span.end < self.plan.len() {
                let step = &self.plan[input.span.end];
                let out = (self.forward.borrow_mut())(&x, step)?
                    .as_dtype(Dtype::Float32)
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
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
/// at `step.s` in f32, read as `prediction` (`_pred_x0`: velocity `x0 = x − s·out`, or the clean
/// image `x0 = out`); `on_step(i, x0)` fires after step `i` (1-based) completes with that step's
/// predicted clean image. Cancellation is checked before every network evaluation and surfaces as
/// [`Error::Canceled`] — the partial state is dropped, never returned.
pub fn sample(
    z: &Array,
    plan: &[DpmStep],
    prediction: Prediction,
    cancel: &CancelFlag,
    model_out: impl FnMut(&Array, &DpmStep) -> Result<Array>,
    mut on_step: impl FnMut(usize, &Array),
) -> Result<Array> {
    let total_steps = plan.len();
    let driver = IrisFlowDpmDriver::with_diagnostics(plan.to_vec(), prediction, true, model_out);
    let mut run_state = DriverRunState::new();
    let mut on_progress = |_: mlx_gen::Progress| {};
    let mut on_observe = |info: &DriverStepInfo, x0: &Array| {
        on_step(info.outer_step + 1, x0);
        Ok(())
    };
    let mut hooks = StepHooks::new(&mut on_progress);
    hooks.on_observe = Some(&mut on_observe);
    let out = run_mlx_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(z.clone()),
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

/// iOS floor of one solver update over `state_elements` f32 elements, counted under MLX's
/// retention ([`crate::retention`]) from [`step_update`] together with the lazy CFG combine it
/// materializes, each `[B, C, H, W]` f32 buffer rounded as allocated: the combine's difference (its
/// scaled copy and the sum donate), `x0` (`s·out` and the difference donate), the update's `cx·x`
/// and `c0·x0` (the difference donates) — three states for a first-order step — and for a
/// second-order step the difference to the previous `x0` (its divided and scaled copies donate).
/// Plus the update's scalar coefficients (four, six; 4 B each).
pub(crate) fn step_scratch(state_elements: u64, update: &DpmUpdate) -> u64 {
    let (states, scalars) = match update {
        DpmUpdate::First { .. } => (3, 4),
        DpmUpdate::Second { .. } => (4, 6),
    };
    states * crate::retention::allocated(4 * state_elements) + 4 * scalars
}

/// One FlowDPM-Solver++ update of `x` from the network output `out` (both f32): returns the new
/// state and this step's predicted clean image `x0` (the body of one [`STEP_QUANTUM`]).
pub(crate) fn step_update(
    x: &Array,
    out: &Array,
    step: &DpmStep,
    prediction: Prediction,
    prev_x0: Option<&Array>,
) -> Result<(Array, Array)> {
    let x0 = match prediction {
        // v-prediction: x0 = x − s·v
        Prediction::Velocity => x.subtract(&out.multiply(Array::from_f32(step.s_f32()))?)?,
        Prediction::Clean => out.clone(),
    };
    let next = match step.update {
        DpmUpdate::First { cx, c0 } => x
            .multiply(Array::from_f32(cx))?
            .subtract(&x0.multiply(Array::from_f32(c0))?)?,
        DpmUpdate::Second { cx, c0, c1, r0 } => {
            let prev = prev_x0.ok_or_else(|| {
                Error::Msg("iris solver: a second-order step needs the previous x0".into())
            })?;
            let d = x0.subtract(prev)?.divide(Array::from_f32(r0))?;
            x.multiply(Array::from_f32(cx))?
                .subtract(&x0.multiply(Array::from_f32(c0))?)?
                .subtract(&d.multiply(Array::from_f32(c1))?)?
        }
    };
    Ok((next, x0))
}

/// `out_uncond + scale · (out_cond − out_uncond)` in f32 (`PIPE-8`).
pub fn cfg_combine(out_uncond: &Array, out_cond: &Array, scale: f32) -> Result<Array> {
    let u = out_uncond.as_dtype(Dtype::Float32)?;
    let c = out_cond.as_dtype(Dtype::Float32)?;
    Ok(mlx_gen::gen_core::guidance::cfg(
        &MlxLatentOps,
        &c,
        &u,
        scale,
    )?)
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn arrays_bit_equal(a: &Array, b: &Array) -> bool {
        a.eval().unwrap();
        b.eval().unwrap();
        a.as_slice::<f32>()
            .iter()
            .zip(b.as_slice::<f32>().iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn iris_flow_dpm_driver_passes_tier0_conformance_for_order2_and_order1() {
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
                            |x, step| {
                                let coeff = Array::from_f32(0.15 + 0.1 * step.s_f32());
                                Ok(x.multiply(&coeff)?)
                            },
                        ))
                    }
                },
                make_initial: || {
                    Array::from_slice(
                        &[
                            0.5f32, -0.25, 0.75, -0.5, 0.1, -0.3, 0.2, 0.4, -0.1, 0.3, -0.2, 0.6,
                        ],
                        &[1, 3, 2, 2],
                    )
                },
                latents_equal: arrays_bit_equal,
            })
            .expect("IrisFlowDpmDriver must pass Tier 0 conformance");
            assert!(report.verified_inert_split);
            assert!(report.verified_carry_refusal_or_support);
            assert_eq!(report.verified_reset_preserves_draws, expect_reset);
            assert!(report.verified_boundary_estimate_readonly_or_refusal);
            assert!(report.verified_renoise_and_resample_guards);
            assert!(report.verified_cancellation);
            assert_eq!(report.unsplit_evaluations, 4);
        }
    }

    #[cfg(not(target_os = "ios"))]
    #[test]
    fn iris_flow_dpm_driver_on_materialize_and_paired_quanta() {
        const CHILD: &str = "IRIS_FLOW_DPM_DRIVER_QUANTUM_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "solver::driver_conformance_tests::iris_flow_dpm_driver_on_materialize_and_paired_quanta",
                ])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(status.success(), "CPU quantum child test failed");
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());

        use mlx_gen::ios_quantum::{
            take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
        };
        use std::rc::Rc;

        let plan = gen_core_iris::dpm_solver_plan(4, 2, 3.0).unwrap();
        let initial = Array::from_slice(
            &[
                0.5f32, -0.25, 0.75, -0.5, 0.1, -0.3, 0.2, 0.4, -0.1, 0.3, -0.2, 0.6,
            ],
            &[1, 3, 2, 2],
        );
        let synthetic_forward = |x: &Array, step: &DpmStep| -> Result<Array> {
            let coeff = Array::from_f32(0.15 + 0.1 * step.s_f32());
            Ok(x.multiply(&coeff)?)
        };

        // 1. External caller attaching StepHooks::on_materialize to IrisFlowDpmDriver::new
        // must invoke on_materialize per step without leaking any unpaired STEP_QUANTUM events.
        let external_events = Rc::new(RefCell::new(Vec::new()));
        let sink = external_events.clone();
        let mut materialized_steps = Vec::new();
        let external_out = with_cpu_diagnostic_headroom(1 << 30, || {
            with_cpu_diagnostic_observer(
                move |event| sink.borrow_mut().push(event.clone()),
                || {
                    let driver = IrisFlowDpmDriver::new(
                        plan.clone(),
                        Prediction::Velocity,
                        synthetic_forward,
                    );
                    let mut run_state = DriverRunState::new();
                    let cancel = CancelFlag::new();
                    let mut on_progress = |_: mlx_gen::Progress| {};
                    let mut on_materialize = |info: &DriverStepInfo, _x: &Array| {
                        materialized_steps.push(info.outer_step);
                        Ok(())
                    };
                    let mut hooks = StepHooks::new(&mut on_progress);
                    hooks.on_materialize = Some(&mut on_materialize);
                    let out = run_mlx_driver_slices(
                        &driver,
                        [SliceSpec::full(plan.len())],
                        BoundaryState::XSigma(initial.clone()),
                        plan.len(),
                        0,
                        &cancel,
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
                },
            )
        })
        .expect("external driver with on_materialize must succeed");
        assert_eq!(materialized_steps, vec![0, 1, 2, 3]);
        assert!(
            external_events.borrow().is_empty(),
            "external IrisFlowDpmDriver::new + on_materialize must not leak STEP_QUANTUM events, got {:?}",
            external_events.borrow()
        );

        // 2. sample() enables diagnostics and emits strictly paired Checked/Synchronized
        // STEP_QUANTUM events per step, producing bit-identical output to the unguarded driver.
        let sample_events = Rc::new(RefCell::new(Vec::new()));
        let sink = sample_events.clone();
        let mut observed_steps = Vec::new();
        let sample_out = with_cpu_diagnostic_headroom(1 << 30, || {
            with_cpu_diagnostic_observer(
                move |event| sink.borrow_mut().push(event.clone()),
                || {
                    let cancel = CancelFlag::new();
                    sample(
                        &initial,
                        &plan,
                        Prediction::Velocity,
                        &cancel,
                        synthetic_forward,
                        |step_idx, _x0| observed_steps.push(step_idx),
                    )
                },
            )
        })
        .expect("sample() with CPU diagnostic headroom must succeed");
        assert_eq!(observed_steps, vec![1, 2, 3, 4]);
        assert!(
            arrays_bit_equal(&external_out, &sample_out),
            "sample() output must be bit-identical to external IrisFlowDpmDriver::new"
        );
        let events = sample_events.borrow();
        assert_eq!(
            events.len(),
            plan.len() * 2,
            "expected strictly paired Checked/Synchronized STEP_QUANTUM per step, got {:?}",
            *events
        );
        for (step_idx, pair) in events.chunks_exact(2).enumerate() {
            let expected_scratch = step_scratch(initial.size() as u64, &plan[step_idx].update);
            assert!(
                matches!(
                    &pair[0],
                    QuantumEvent::Checked { label, scratch }
                        if label == STEP_QUANTUM && *scratch == expected_scratch
                ),
                "step {step_idx}: expected Checked({STEP_QUANTUM}, {expected_scratch}), got {:?}",
                pair[0]
            );
            assert!(
                matches!(
                    &pair[1],
                    QuantumEvent::Synchronized { label, .. } if label == STEP_QUANTUM
                ),
                "step {step_idx}: expected Synchronized({STEP_QUANTUM}), got {:?}",
                pair[1]
            );
        }

        // 3. A refusal on zero headroom stops cleanly at STEP_QUANTUM.
        let _ = take_refusal();
        let refused = with_cpu_diagnostic_headroom(0, || {
            let cancel = CancelFlag::new();
            sample(
                &initial,
                &plan,
                Prediction::Velocity,
                &cancel,
                synthetic_forward,
                |_, _| {},
            )
        });
        assert!(refused.is_err());
        assert_eq!(
            take_refusal().expect("STEP_QUANTUM refusal recorded").label,
            STEP_QUANTUM
        );
    }
}
