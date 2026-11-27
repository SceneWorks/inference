//! Candle backend bindings for the typed [`FamilyDriver`] contract (`PIPE-7`, `PIPE-14`, `B3`).
//!
//! Exposes the shared [`gen_core::driver`] types plus [`run_candle_driver_slices`] so Candle
//! family drivers (SDXL ancestral/Lightning, Kolors leading-Euler, Wan UniPC, FLUX.3 flow-map,
//! Sana Sprint SCM, Boogu/MOSS DMD, Iris FlowDPM, Ideogram, Supra2) run under one typed slice
//! runner with request-owned [`DriverRunState`] and [`StepHooks`].

pub use gen_core::driver::{
    conformance, run_driver_slices, DriverEvaluationKind, DriverHistoryPolicy,
    DriverRefusalReasons, DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver,
    GlobalRngTrajectoryGate, GlobalRngTrajectoryLease, SliceInput, SliceSpec, SliceTransition,
    StepHooks, StreamObserveHook,
};
pub use gen_core::multistream::{MultiStreamStepInfo, StreamId, StreamStepSigma};

use crate::Result;
use gen_core::CancelFlag;

/// Drive a Candle [`FamilyDriver<L>`] across one or more planned slices, binding optional
/// [`gen_core::RenderRun`] phase scopes, cancellation, and owner-thread verification.
#[allow(clippy::too_many_arguments)]
pub fn run_candle_driver_slices<L, D: FamilyDriver<L> + ?Sized>(
    driver: &D,
    slices: impl IntoIterator<Item = SliceSpec>,
    initial_state: gen_core::BoundaryState<L>,
    total_steps: usize,
    seed: u64,
    cancel: &CancelFlag,
    run: Option<&gen_core::RenderRun<'_, '_>>,
    state: &mut DriverRunState,
    hooks: &mut StepHooks<'_, L>,
    on_boundary: impl FnMut(
        usize,
        gen_core::BoundaryState<L>,
        &mut DriverRunState,
    ) -> gen_core::Result<gen_core::BoundaryState<L>>,
) -> Result<gen_core::BoundaryState<L>> {
    run_driver_slices(
        driver,
        slices,
        initial_state,
        total_steps,
        seed,
        cancel,
        run,
        state,
        hooks,
        on_boundary,
    )
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    struct EchoDriver {
        support: DriverSupport,
    }

    impl FamilyDriver<Tensor> for EchoDriver {
        fn support(&self) -> &DriverSupport {
            &self.support
        }

        fn run_slice(
            &self,
            input: SliceInput<'_, Tensor>,
            state: &mut DriverRunState,
            hooks: &mut StepHooks<'_, Tensor>,
        ) -> gen_core::Result<gen_core::BoundaryState<Tensor>> {
            self.support.validate_slice(&input, state)?;
            state.enter_slice(&input)?;
            let mut x = match input.state {
                gen_core::BoundaryState::XSigma(t)
                | gen_core::BoundaryState::CleanEstimate { latent: t, .. }
                | gen_core::BoundaryState::RouteSpecific { state: t, .. } => t,
            };
            for step in input.span.clone() {
                if input.cancel.is_cancelled() {
                    return Err(gen_core::Error::Canceled);
                }
                let info = DriverStepInfo {
                    pass: input.pass,
                    phase: input.phase,
                    outer_step: step,
                    total_outer_steps: input.total_steps,
                    eval_in_step: 0,
                    kind: DriverEvaluationKind::Trajectory,
                    sigma: (input.total_steps - step) as f32,
                    next_sigma: (input.total_steps - step - 1) as f32,
                };
                hooks.observe(&info, &x)?;
                x = x
                    .affine(1.0, 0.25)
                    .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
                state.record_evaluation(DriverEvaluationKind::Trajectory);
                x = hooks.post_step(&info, x)?;
                hooks.materialize(&info, &x)?;
                state.record_outer_step(step);
                hooks.emit_progress(&info);
            }
            state.finish_slice(input.span.end);
            Ok(gen_core::BoundaryState::XSigma(x))
        }
    }

    #[test]
    fn candle_driver_slices_runs_weights_free_conformance() {
        let support = DriverSupport {
            driver_id: "candle_echo_test",
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons::default(),
        };
        let report = conformance::verify_driver_conformance(conformance::ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 7,
            base_geometry: (64, 64),
            alt_geometry: (128, 128),
            make_driver: || {
                Box::new(EchoDriver {
                    support: support.clone(),
                })
            },
            make_initial: || {
                Tensor::zeros((1, 4, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap()
            },
            latents_equal: |a, b| {
                let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("candle echo driver passes Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_cancellation);
    }
}
