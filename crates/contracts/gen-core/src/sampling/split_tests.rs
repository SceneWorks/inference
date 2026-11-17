//! Host-arithmetic conformance for an inert phase boundary. No learned module is constructed.

use super::model_sampling::FlowModelSampling;
use super::solvers::{SolverObservation, SolverObservationKind};
use super::{
    cfgpp_sampler_for, CpuLatentOps, EulerAncestral, LatentOps, Sampler, Solver, SolverRunState,
    TimestepConvention,
};
use crate::Result;
use std::cell::RefCell;

#[derive(Debug, Default, PartialEq, Eq)]
struct Trace {
    callbacks: Vec<(Vec<u32>, u32)>,
    events: Vec<TraceEvent>,
    draws: Vec<usize>,
}

type TraceEvent = (
    Solver,
    usize,
    SolverObservationKind,
    Vec<u32>,
    [bool; 3],
    bool,
    bool,
);

#[derive(Default)]
struct ObservedOps(RefCell<Trace>);

impl LatentOps for ObservedOps {
    type Latent = Vec<f32>;

    fn observe_solver(&self, event: SolverObservation<'_, Self::Latent>) {
        self.0.borrow_mut().events.push((
            event.solver,
            event.step,
            event.kind,
            event.current.iter().map(|v| v.to_bits()).collect(),
            event.history.map(|h| h.is_some()),
            event.cfg_pp,
            event.noise,
        ));
    }

    fn scale(&self, x: &Self::Latent, scale: f32) -> Result<Self::Latent> {
        CpuLatentOps.scale(x, scale)
    }

    fn add(&self, a: &Self::Latent, b: &Self::Latent) -> Result<Self::Latent> {
        CpuLatentOps.add(a, b)
    }

    fn sub(&self, a: &Self::Latent, b: &Self::Latent) -> Result<Self::Latent> {
        CpuLatentOps.sub(a, b)
    }

    fn axpy(&self, a: f32, x: &Self::Latent, b: f32, y: &Self::Latent) -> Result<Self::Latent> {
        CpuLatentOps.axpy(a, x, b, y)
    }

    fn randn_like(&self, x: &Self::Latent, seed: u64, step: usize) -> Result<Self::Latent> {
        self.0.borrow_mut().draws.push(step);
        CpuLatentOps.randn_like(x, seed, step)
    }
}

fn initial() -> Vec<f32> {
    vec![0.4, -0.2, 0.8, 1.1]
}

fn sigmas() -> [f32; 6] {
    [0.96, 0.77, 0.59, 0.34, 0.17, 0.0]
}

fn denoise(ops: &ObservedOps, x: &[f32], sigma: f32) -> Result<Vec<f32>> {
    ops.0
        .borrow_mut()
        .callbacks
        .push((x.iter().map(|v| v.to_bits()).collect(), sigma.to_bits()));
    Ok(x.iter().map(|v| v * 0.7 + 0.11 * sigma).collect())
}

fn cfgpp_denoise(ops: &ObservedOps, x: &[f32], sigma: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    ops.0
        .borrow_mut()
        .callbacks
        .push((x.iter().map(|v| v.to_bits()).collect(), sigma.to_bits()));
    Ok((
        x.iter().map(|v| v * 0.7 + 0.11 * sigma).collect(),
        x.iter().map(|v| v * 0.6 - 0.09 * sigma).collect(),
    ))
}

#[test]
fn every_curated_solver_keeps_exact_trace_across_every_inert_split() {
    let sigmas = sigmas();
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    for solver in Solver::ALL {
        let reference_ops = ObservedOps::default();
        let reference = solver
            .boxed::<ObservedOps>()
            .sample(
                &reference_ops,
                &ms,
                &mut |x, sigma| denoise(&reference_ops, x, sigma),
                initial(),
                &sigmas,
                104729,
            )
            .unwrap();
        let reference_trace = reference_ops.0.into_inner();

        for cut in 1..sigmas.len() - 1 {
            let ops = ObservedOps::default();
            let sampler = solver.boxed::<ObservedOps>();
            let mut state = SolverRunState::new(solver, &sigmas);
            let mut callback = |x: &Vec<f32>, sigma| denoise(&ops, x, sigma);
            let first = sampler
                .sample_span(
                    &ops,
                    &ms,
                    &mut callback,
                    initial(),
                    &sigmas,
                    104729,
                    0..cut,
                    &mut state,
                )
                .unwrap();
            assert_eq!(state.next_step(), cut, "{} first span", solver.name());
            let second = sampler
                .sample_span(
                    &ops,
                    &ms,
                    &mut callback,
                    first,
                    &sigmas,
                    104729,
                    cut..sigmas.len() - 1,
                    &mut state,
                )
                .unwrap();
            assert_eq!(state.next_step(), sigmas.len() - 1);
            assert_eq!(second, reference, "{} cut {cut} output", solver.name());
            assert_eq!(
                ops.0.into_inner(),
                reference_trace,
                "{} cut {cut} callbacks, history and draw addresses",
                solver.name()
            );
        }

        let ops = ObservedOps::default();
        let sampler = solver.boxed::<ObservedOps>();
        let mut state = SolverRunState::new(solver, &sigmas);
        let mut callback = |x: &Vec<f32>, sigma| denoise(&ops, x, sigma);
        let mut x = initial();
        for step in 0..sigmas.len() - 1 {
            x = sampler
                .sample_span(
                    &ops,
                    &ms,
                    &mut callback,
                    x,
                    &sigmas,
                    104729,
                    step..step + 1,
                    &mut state,
                )
                .unwrap();
        }
        assert_eq!(x, reference, "{} singleton spans", solver.name());
        assert_eq!(ops.0.into_inner(), reference_trace);
    }
}

#[test]
fn cfgpp_variants_keep_exact_trace_across_every_inert_split() {
    let sigmas = sigmas();
    for solver in [Solver::Euler, Solver::Ddim, Solver::Dpmpp2m] {
        let reference_ops = ObservedOps::default();
        let reference = cfgpp_sampler_for::<ObservedOps>(solver)
            .unwrap()
            .sample(
                &reference_ops,
                &mut |x, sigma| cfgpp_denoise(&reference_ops, x, sigma),
                initial(),
                &sigmas,
            )
            .unwrap();
        let reference_trace = reference_ops.0.into_inner();

        for cut in 1..sigmas.len() - 1 {
            let ops = ObservedOps::default();
            let sampler = cfgpp_sampler_for::<ObservedOps>(solver).unwrap();
            let mut state = SolverRunState::new_cfg_pp(solver, &sigmas);
            let mut callback = |x: &Vec<f32>, sigma| cfgpp_denoise(&ops, x, sigma);
            let first = sampler
                .sample_span(&ops, &mut callback, initial(), &sigmas, 0..cut, &mut state)
                .unwrap();
            let second = sampler
                .sample_span(
                    &ops,
                    &mut callback,
                    first,
                    &sigmas,
                    cut..sigmas.len() - 1,
                    &mut state,
                )
                .unwrap();
            assert_eq!(second, reference, "CFG++ {} cut {cut}", solver.name());
            assert_eq!(ops.0.into_inner(), reference_trace);
        }

        let ops = ObservedOps::default();
        let sampler = cfgpp_sampler_for::<ObservedOps>(solver).unwrap();
        let mut state = SolverRunState::new_cfg_pp(solver, &sigmas);
        let mut callback = |x: &Vec<f32>, sigma| cfgpp_denoise(&ops, x, sigma);
        let mut x = initial();
        for step in 0..sigmas.len() - 1 {
            x = sampler
                .sample_span(&ops, &mut callback, x, &sigmas, step..step + 1, &mut state)
                .unwrap();
        }
        assert_eq!(x, reference, "CFG++ {} singleton spans", solver.name());
        assert_eq!(ops.0.into_inner(), reference_trace);
    }
}

#[test]
fn carry_refuses_wrong_solver_schedule_or_position_without_advancing() {
    let sigmas = sigmas();
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let ops = ObservedOps::default();
    let mut state = SolverRunState::new(Solver::Euler, &sigmas);
    let sampler = Solver::Euler.boxed::<ObservedOps>();
    let mut callback = |x: &Vec<f32>, sigma| denoise(&ops, x, sigma);
    let first = sampler
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            initial(),
            &sigmas,
            13,
            0..2,
            &mut state,
        )
        .unwrap();
    assert_eq!(state.next_step(), 2);
    assert!(Solver::Heun
        .boxed::<ObservedOps>()
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first.clone(),
            &sigmas,
            13,
            2..3,
            &mut state
        )
        .is_err());
    let mut changed = sigmas;
    changed[3] = 0.33;
    assert!(sampler
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first.clone(),
            &changed,
            13,
            2..3,
            &mut state
        )
        .is_err());
    assert!(sampler
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first.clone(),
            &sigmas,
            14,
            2..3,
            &mut state
        )
        .is_err());
    let changed_convention = FlowModelSampling::new(TimestepConvention::OneMinusSigma);
    assert!(sampler
        .sample_span(
            &ops,
            &changed_convention,
            &mut callback,
            first.clone(),
            &sigmas,
            13,
            2..3,
            &mut state
        )
        .is_err());
    assert!(sampler
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first.clone(),
            &sigmas,
            13,
            3..4,
            &mut state
        )
        .is_err());
    assert_eq!(state.next_step(), 2);
    state.reset_history();
    assert_eq!(state.next_step(), 2, "Reset does not rewind draw positions");
    state.reset_for_solver(Solver::Heun);
    assert_eq!(state.next_step(), 2);
    Solver::Heun
        .boxed::<ObservedOps>()
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first,
            &sigmas,
            13,
            2..3,
            &mut state,
        )
        .unwrap();
    assert_eq!(state.next_step(), 3);
}

#[test]
fn er_sde_reset_restarts_correction_order_but_keeps_absolute_draws() {
    let sigmas = sigmas();
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let ops = ObservedOps::default();
    let sampler = Solver::ErSde.boxed::<ObservedOps>();
    let mut state = SolverRunState::new(Solver::ErSde, &sigmas);
    let mut callback = |x: &Vec<f32>, sigma| denoise(&ops, x, sigma);
    let first = sampler
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            initial(),
            &sigmas,
            7,
            0..2,
            &mut state,
        )
        .unwrap();
    state.reset_history();
    let _ = sampler
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first,
            &sigmas,
            7,
            2..sigmas.len() - 1,
            &mut state,
        )
        .unwrap();
    let trace = ops.0.into_inner();
    let reset_step = trace
        .events
        .iter()
        .find(|event| event.1 == 2)
        .expect("first step after Reset is observed");
    assert_eq!(reset_step.4, [false; 3]);
    assert!(trace.draws.iter().any(|&draw| draw >= 2));
    assert!(trace.draws.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn carry_refuses_stochastic_variant_change() {
    let sigmas = sigmas();
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let ops = ObservedOps::default();
    let mut state = SolverRunState::new(Solver::EulerAncestral, &sigmas);
    let mut callback = |x: &Vec<f32>, sigma| denoise(&ops, x, sigma);
    let first = EulerAncestral { eta: 1.0 }
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            initial(),
            &sigmas,
            7,
            0..1,
            &mut state,
        )
        .unwrap();
    assert!(EulerAncestral { eta: 0.5 }
        .sample_span(
            &ops,
            &ms,
            &mut callback,
            first,
            &sigmas,
            7,
            1..2,
            &mut state
        )
        .is_err());
    assert_eq!(state.next_step(), 1);
}
