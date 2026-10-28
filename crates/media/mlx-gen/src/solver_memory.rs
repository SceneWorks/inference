//! Concrete host-only logical allocation execution of the ACTUAL curated solver factory.
//! No tensors/numerical latent integration: scalar branch math stays in gen-core's solver.
//! Node edges mirror MlxLatentOps' lazy expressions; clone aliases and callback materialization
//! are explicit. This assumes NO donation/reuse and is not a physical allocation proof.
use crate::gen_core::sampling::{denoise, LatentOps, ModelSampling};
use crate::{Error, Result, Solver};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Clone, Debug, Default)]
pub struct SolverMemory {
    /// Maximum solver-owned bytes beside the callback's current x, at the SAME callback epoch.
    pub callback_outer_bytes: u64,
    /// All logical live nodes while constructing/materializing the lazy solver update.
    pub step_peak_bytes: u64,
    pub state_bytes: u64,
    pub final_bytes: u64,
    pub callbacks: usize,
}
#[derive(Debug)]
struct Node {
    elements: u64,
    width: u64,
    dependencies: RefCell<Vec<Rc<Node>>>,
}
#[derive(Clone, Debug)]
struct Token(Rc<Node>);
#[derive(Default)]
struct LogicalMlxOps {
    nodes: RefCell<Vec<Weak<Node>>>,
    peak: RefCell<u64>,
    // Scalar facts from SYMBOLIC execution only. Never exported as native capture observations.
    logical_events: RefCell<Vec<crate::gen_core::sampling::solvers::SolverEpoch>>,
}
impl LogicalMlxOps {
    fn active(&self) -> u64 {
        let mut nodes = self.nodes.borrow_mut();
        nodes.retain(|node| node.strong_count() != 0);
        nodes
            .iter()
            .filter_map(Weak::upgrade)
            .map(|n| n.elements.saturating_mul(n.width))
            .sum()
    }
    fn node(&self, elements: u64, width: u64, inputs: &[&Token]) -> Token {
        let token = Token(Rc::new(Node {
            elements,
            width,
            dependencies: RefCell::new(inputs.iter().map(|x| x.0.clone()).collect()),
        }));
        self.nodes.borrow_mut().push(Rc::downgrade(&token.0));
        let active = self.active();
        let mut peak = self.peak.borrow_mut();
        *peak = (*peak).max(active);
        token
    }
    fn scalar(&self, width: u64) -> Token {
        let scalar = self.node(1, 4, &[]); // array::scalar is f32
        if width == 4 {
            scalar
        } else {
            self.node(1, width, &[&scalar])
        }
    }
    fn materialize(node: &Rc<Node>) {
        let inputs = std::mem::take(&mut *node.dependencies.borrow_mut());
        for input in &inputs {
            Self::materialize(input);
        }
        // Evaluated children keep their own storage if a solver history/outer owner still holds it.
    }
    fn multiply(&self, x: &Token) -> Token {
        let scalar = self.scalar(x.0.width);
        self.node(x.0.elements, x.0.width, &[x, &scalar])
    }
}
impl LatentOps for LogicalMlxOps {
    type Latent = Token;
    fn scale(&self, x: &Token, scale: f32) -> crate::gen_core::Result<Token> {
        Ok(if scale == 1.0 {
            x.clone()
        } else {
            self.multiply(x)
        })
    }
    fn add(&self, a: &Token, b: &Token) -> crate::gen_core::Result<Token> {
        Ok(self.node(a.0.elements, a.0.width.max(b.0.width), &[a, b]))
    }
    fn sub(&self, a: &Token, b: &Token) -> crate::gen_core::Result<Token> {
        self.add(a, b)
    }
    fn axpy(&self, a: f32, x: &Token, _b: f32, y: &Token) -> crate::gen_core::Result<Token> {
        // EXACT MlxLatentOps graph: b*y always; a*x only when a != 1, then add.
        let by = self.multiply(y);
        if a == 1.0 {
            self.add(x, &by)
        } else {
            let ax = self.multiply(x);
            self.add(&ax, &by)
        }
    }
    fn randn_like(&self, x: &Token, _seed: u64, _step: usize) -> crate::gen_core::Result<Token> {
        Ok(self.node(x.0.elements, 4, &[])) // StepRng::normal is f32, irrespective of state dtype.
    }
    fn observe_solver(
        &self,
        event: crate::gen_core::sampling::solvers::SolverObservation<'_, Token>,
    ) {
        use crate::gen_core::sampling::solvers::{SolverEpoch, SolverObservationKind};
        let mut epochs = self.logical_events.borrow_mut();
        match event.kind {
            SolverObservationKind::Callback(evaluation) => epochs.push(SolverEpoch {
                step: event.step,
                sigma: event.sigma,
                next_sigma: event.next_sigma,
                evaluation,
                previous_denoised: event.history[0].is_some(),
                previous_derivative: event.history[1].is_some(),
                previous_sample: event.history[2].is_some(),
                outer_owners: event.outer.iter().filter(|owner| owner.is_some()).count() as u8,
                correction_order: event.correction_order,
                corrector_after_callback: false,
                state_aliases_denoised_after_step: false,
            }),
            SolverObservationKind::Correction if event.solver == Solver::UniPc => {
                for epoch in epochs.iter_mut().filter(|epoch| epoch.step == event.step) {
                    epoch.corrector_after_callback = true;
                }
            }
            SolverObservationKind::Update => {
                for epoch in epochs.iter_mut().filter(|epoch| epoch.step == event.step) {
                    epoch.state_aliases_denoised_after_step = event.terminal_alias;
                }
            }
            _ => {}
        }
    }
}

/// One POSSIBLE callback ownership family from bounded symbolic execution of the actual factory.
/// This is a coincident logical ledger, not observed execution or a physical backing certificate.
#[derive(Clone, Debug)]
pub struct SolverCallbackWindow {
    pub epoch: crate::gen_core::sampling::solvers::SolverEpoch,
    /// First actual MODEL callback in this symbolic solve, not merely step zero or a primary
    /// evaluation. A zero-sigma callback still consumes model parameters; a solve with no
    /// transitions has no callback marker at all.
    pub first_model_callback: bool,
    pub callback_outer_bytes: u64,
    pub state_bytes: u64,
    /// Actual symbolic c_in result held across the model callback. `alias` is logical Rc identity,
    /// not proof that native execution donates, shares or releases physical backing.
    pub scaled_input_bytes: u64,
    pub scaled_input_alias: bool,
    pub cfg_pp: bool,
}
impl SolverCallbackWindow {
    pub fn family(&self) -> String {
        let e = &self.epoch;
        use crate::gen_core::sampling::solvers::SolverEvaluation;
        let evaluation = match e.evaluation {
            SolverEvaluation::Primary => "primary",
            SolverEvaluation::HeunCorrector => "heun-corrector",
            SolverEvaluation::SdeMidpoint => "sde-midpoint",
        };
        format!(
            "{evaluation}/history-{}{}{}-outer-{}-stage-{}-correct-{}-terminal-{}-zero-{}-cfgpp-{}-scaled-alias-{}-first-model-{}",
            u8::from(e.previous_denoised),
            u8::from(e.previous_derivative),
            u8::from(e.previous_sample),
            e.outer_owners,
            e.correction_order,
            u8::from(e.corrector_after_callback),
            u8::from(e.next_sigma == 0.0),
            u8::from(e.sigma == 0.0),
            u8::from(self.cfg_pp),
            u8::from(self.scaled_input_alias),
            u8::from(self.first_model_callback),
        )
    }
}

/// Conservative finite family set, independent of the requested step/count values. Include
/// warm-up, steady history, terminal, repeated and zero-leading transitions. Extra possible
/// families are NOT evidence that the user's exact schedule executed any of them.
pub fn solver_callback_windows(
    solver: Solver,
    convention: crate::gen_core::sampling::TimestepConvention,
    elements: u64,
    initial_width: u64,
    prediction_width: u64,
) -> Result<Vec<SolverCallbackWindow>> {
    let ms = crate::gen_core::sampling::FlowModelSampling::new(convention);
    solver_callback_windows_for_model(
        solver,
        false,
        &ms,
        elements,
        initial_width,
        prediction_width,
    )
}

/// Bounded possible ownership families using the caller's REAL model sampling and actual ordinary
/// or CFG++ factory. This is symbolic source analysis only: no requested-N schedule, native arrays,
/// observed-capture facts, physical backing certificate or cross-model equivalence is produced.
pub fn solver_callback_windows_for_model(
    solver: Solver,
    cfg_pp: bool,
    ms: &dyn ModelSampling,
    elements: u64,
    initial_width: u64,
    prediction_width: u64,
) -> Result<Vec<SolverCallbackWindow>> {
    let n = solver.minimum_capture_transitions();
    let high = ms.sigma_max();
    let minimum = ms.sigma_min();
    let low = if minimum > 0.0 {
        minimum
    } else {
        high / n as f32
    };
    if !high.is_finite() || !low.is_finite() || high <= 0.0 || low > high {
        return Err(Error::Unsupported(
            "logical model sampling requires finite positive sigma bounds".into(),
        ));
    }
    let mut regular: Vec<f32> = (0..n)
        .map(|i| high + (low - high) * i as f32 / (n - 1) as f32)
        .collect();
    regular.push(0.0);
    let repeat = [high, high, 0.0];
    let mut windows = Vec::new();
    for sigmas in [regular.as_slice(), &[0.0, 0.0], repeat.as_slice()] {
        windows.extend(
            solver_memory_with_windows(
                solver,
                ms,
                sigmas,
                elements,
                initial_width,
                prediction_width,
                cfg_pp,
            )?
            .1,
        );
    }
    Ok(windows)
}

/// Per-output solver working set. `initial_width` and prediction width are ACTUAL architecture /
/// runtime dtype facts (2=bf16,4=f32), not model::DTYPE assumptions. Forward kernels, within-call
/// CFG retention, init/context owners and completed serial finals are provider-projected separately.
pub fn solver_memory(
    solver: Solver,
    ms: &dyn ModelSampling,
    sigmas: &[f32],
    elements: u64,
    initial_width: u64,
    prediction_width: u64,
) -> Result<SolverMemory> {
    Ok(solver_memory_with_windows(
        solver,
        ms,
        sigmas,
        elements,
        initial_width,
        prediction_width,
        false,
    )?
    .0)
}

fn record_callback_window(
    ops: &LogicalMlxOps,
    x: &Token,
    input: &Token,
    cfg_pp: bool,
    facts: &mut SolverMemory,
    windows: &mut Vec<SolverCallbackWindow>,
) -> crate::gen_core::Result<()> {
    let epoch = ops.logical_events.borrow().last().cloned().ok_or_else(|| {
        crate::gen_core::Error::Msg("symbolic factory emitted no callback ownership event".into())
    })?;
    let state_bytes = x.0.elements.saturating_mul(x.0.width);
    let scaled_input_bytes = input.0.elements.saturating_mul(input.0.width);
    let scaled_input_alias = Rc::ptr_eq(&x.0, &input.0);
    let callback_outer_bytes =
        ops.active()
            .saturating_sub(state_bytes)
            .saturating_sub(if scaled_input_alias {
                0
            } else {
                scaled_input_bytes
            });
    facts.callback_outer_bytes = facts.callback_outer_bytes.max(callback_outer_bytes);
    facts.state_bytes = facts.state_bytes.max(state_bytes);
    let first_model_callback = facts.callbacks == 0;
    facts.callbacks += 1;
    windows.push(SolverCallbackWindow {
        epoch,
        first_model_callback,
        callback_outer_bytes,
        state_bytes,
        scaled_input_bytes,
        scaled_input_alias,
        cfg_pp,
    });
    Ok(())
}

fn solver_memory_with_windows(
    solver: Solver,
    ms: &dyn ModelSampling,
    sigmas: &[f32],
    elements: u64,
    initial_width: u64,
    prediction_width: u64,
    cfg_pp: bool,
) -> Result<(SolverMemory, Vec<SolverCallbackWindow>)> {
    if !matches!(initial_width, 2 | 4) || !matches!(prediction_width, 2 | 4) {
        return Err(Error::Unsupported(
            "solver memory requires actual 16/32-bit widths".into(),
        ));
    }
    solver.logical_epochs(sigmas).map_err(Error::from)?; // exact supplied schedule validation only
    let mut windows = Vec::new();
    let ops = LogicalMlxOps::default();
    let initial = ops.node(elements, initial_width, &[]);
    let mut facts = SolverMemory::default();
    let final_state = if cfg_pp {
        let sampler = crate::gen_core::sampling::cfgpp_sampler_for::<LogicalMlxOps>(solver)
            .ok_or_else(|| Error::Unsupported("solver has no actual CFG++ factory".into()))?;
        let mut callback = |x: &Token, sigma: f32| -> crate::gen_core::Result<(Token, Token)> {
            LogicalMlxOps::materialize(&x.0);
            crate::gen_core::sampling::cfgpp_denoise(&ops, ms, x, sigma, |input, _time| {
                record_callback_window(&ops, x, input, true, &mut facts, &mut windows)?;
                Ok((
                    ops.node(input.0.elements, prediction_width, &[]),
                    ops.node(input.0.elements, prediction_width, &[]),
                ))
            })
        };
        sampler
            .sample(&ops, &mut callback, initial, sigmas)
            .map_err(Error::from)?
    } else {
        let mut callback = |x: &Token, sigma: f32| -> crate::gen_core::Result<Token> {
            LogicalMlxOps::materialize(&x.0);
            denoise(&ops, ms, x, sigma, |input, _time| {
                record_callback_window(&ops, x, input, false, &mut facts, &mut windows)?;
                Ok(ops.node(input.0.elements, prediction_width, &[]))
            })
        };
        solver
            .boxed::<LogicalMlxOps>()
            .sample(&ops, ms, &mut callback, initial, sigmas, 0)
            .map_err(Error::from)?
    };
    // Correction/terminal flags come from the ACTUAL selected factory branches during symbolic
    // execution, including CFG++ old_uncond retention. No ordinary-solver epoch policy is reused
    // as a CFG++ surrogate. These remain possible logical families, not native observations.
    let epochs = ops.logical_events.borrow();
    if epochs.len() != windows.len() {
        return Err(Error::Msg(
            "symbolic callback/event ownership count mismatch".into(),
        ));
    }
    for (window, epoch) in windows.iter_mut().zip(epochs.iter()) {
        window.epoch = epoch.clone();
    }
    facts.final_bytes = final_state.0.elements.saturating_mul(final_state.0.width);
    facts.state_bytes = facts.state_bytes.max(facts.final_bytes);
    facts.step_peak_bytes = *ops.peak.borrow();
    Ok((facts, windows))
}

#[cfg(test)]
mod callback_history_tests {
    use super::*;

    #[test]
    fn first_model_callback_is_not_repeated_for_a_corrector_or_later_primary() {
        let ms = crate::gen_core::sampling::FlowModelSampling::new(
            crate::gen_core::sampling::TimestepConvention::OneMinusSigma,
        );
        let (_, windows) =
            solver_memory_with_windows(Solver::Heun, &ms, &[1.0, 0.5, 0.0], 8, 2, 4, false)
                .unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| window.first_model_callback)
                .collect::<Vec<_>>(),
            [true, false, false],
        );
    }
}
