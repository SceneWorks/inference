//! CLOSED solver expression liveness interpreter under an explicit, verified
//! native retirement boundary. Selected ONLY by validated actual invocation
//! receipts; never a process/phone-fit certificate.
//!
//! Tokens mirror MlxLatentOps' source Array identities and lazy input edges.
//! Every reachable future output is conservatively charged before evaluation.
//! After the matching real eval+retirement succeeds, detach evaluated edges;
//! externally held histories/aliases remain registered via their actual Rc.
use super::*;
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

#[derive(Debug)]
struct Node {
    elements: u64,
    output_upper: u64,
    unevaluated_extra: u64,
    evaluated: Cell<bool>,
    inputs: RefCell<[Option<Rc<Node>>; 2]>,
}
#[derive(Clone, Debug)]
pub(super) struct LiveLatent(Rc<Node>);
pub(super) struct LiveOps {
    page: u64,
    backend: NativeBackend,
    analysis_budget: u64,
    nodes: RefCell<Vec<Weak<Node>>>,
    ambient: Cell<u64>,
    peak: Cell<u64>,
    created: Cell<u64>,
    solver_marks: Cell<Option<u64>>,
}
impl LiveOps {
    pub(super) fn new(page: u64, backend: NativeBackend, analysis_budget: u64) -> Result<Self> {
        NativeLedger::new(page, backend)?;
        Ok(Self {
            page,
            backend,
            analysis_budget,
            nodes: RefCell::new(Vec::new()),
            ambient: Cell::new(0),
            peak: Cell::new(0),
            created: Cell::new(0),
            solver_marks: Cell::new(Some(0)),
        })
    }
    pub(super) fn solver_marks(&self) -> Result<u64> {
        self.solver_marks
            .get()
            .ok_or_else(|| invalid("source solver mark count overflow"))
    }
    fn operand(&self, elements: u64) -> Result<u64> {
        let mut l = NativeLedger::new(self.page, self.backend)?;
        l.array(4, elements)?;
        Ok(l.total)
    }
    fn check_node_budget(&self) -> Result<()> {
        let mut nodes = self.nodes.borrow_mut();
        nodes.retain(|n| n.strong_count() != 0);
        // Requested analysis storage, not a physical heap statement. Node's
        // inputs have fixed inline arity; no hidden parent-vector growth.
        let rc = std::mem::size_of::<Node>() + 2 * std::mem::size_of::<usize>();
        let next = nodes
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("live analysis node count overflow"))?;
        let weak = std::mem::size_of::<Weak<Node>>();
        let bytes = next
            .checked_mul(rc)
            .and_then(|n| {
                nodes
                    .capacity()
                    .max(next)
                    .checked_mul(weak)
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("live analysis storage overflow"))?;
        if bytes > self.analysis_budget {
            return Err(invalid(
                "live expression analysis exceeds caller host byte budget",
            ));
        }
        if nodes.len() == nodes.capacity() {
            nodes
                .try_reserve_exact(1)
                .map_err(|_| invalid("live expression analysis allocation failed"))?;
        }
        Ok(())
    }
    fn node(
        &self,
        elements: u64,
        inputs: &[&LiveLatent],
        extra: u64,
        evaluated: bool,
    ) -> Result<LiveLatent> {
        if inputs.len() > 2 {
            return Err(invalid(
                "closed solver operation has excessive source arity",
            ));
        }
        let output_upper = self.operand(elements)?;
        self.check_node_budget()?;
        let mut parents = [None, None];
        for (i, parent) in inputs.iter().enumerate() {
            parents[i] = Some(parent.0.clone());
        }
        let token = LiveLatent(Rc::new(Node {
            elements,
            output_upper,
            unevaluated_extra: extra,
            evaluated: Cell::new(evaluated),
            inputs: RefCell::new(parents),
        }));
        self.nodes.borrow_mut().push(Rc::downgrade(&token.0));
        self.created.set(
            self.created
                .get()
                .checked_add(1)
                .ok_or_else(|| invalid("live source node count overflow"))?,
        );
        self.record_external_epoch(0)?;
        Ok(token)
    }
    pub(super) fn active_upper(&self) -> Result<u64> {
        let mut nodes = self.nodes.borrow_mut();
        nodes.retain(|n| n.strong_count() != 0);
        nodes
            .iter()
            .filter_map(Weak::upgrade)
            .try_fold(0, |total, node| {
                let current = if node.evaluated.get() {
                    node.output_upper
                } else {
                    sum(node.output_upper, node.unevaluated_extra)?
                };
                sum(total, current)
            })
    }
    pub(super) fn set_ambient(&self, bytes: u64) -> Result<()> {
        self.ambient.set(bytes);
        self.record_external_epoch(0)
    }
    /// Include the whole reached source allocation envelope of a model quantum
    /// beside CURRENT solver owners; never add independently maximized peaks.
    pub(super) fn record_external_epoch(&self, source_upper: u64) -> Result<()> {
        let here = sum(sum(self.ambient.get(), self.active_upper()?)?, source_upper)?;
        self.peak.set(self.peak.get().max(here));
        Ok(())
    }
    pub(super) fn peak(&self) -> u64 {
        self.peak.get()
    }
    pub(super) fn ambient(&self) -> u64 {
        self.ambient.get()
    }
    pub(super) fn fresh_materialized(&self, elements: u64) -> Result<LiveLatent> {
        self.node(elements, &[], 0, true)
    }
    pub(super) fn scalar(&self) -> Result<LiveLatent> {
        self.fresh_materialized(1)
    }
    pub(super) fn multiply_always(&self, x: &LiveLatent) -> Result<LiveLatent> {
        let scalar = self.scalar()?;
        self.node(x.0.elements, &[x, &scalar], 0, false)
    }
    pub(super) fn binary(&self, a: &LiveLatent, b: &LiveLatent) -> Result<LiveLatent> {
        if a.0.elements != b.0.elements {
            return Err(invalid("closed solver element extent mismatch"));
        }
        self.node(a.0.elements, &[a, b], 0, false)
    }
    pub(super) fn random(&self, elements: u64) -> Result<LiveLatent> {
        // Same audited F32 normal recipe as the cumulative native source walk.
        let mut l = NativeLedger::new(self.page, self.backend)?;
        l.array(4, 2)?;
        l.array(4, 4)?;
        l.repeated(4, 1, 6)?;
        l.repeated(4, elements, 8)?;
        let extra = l
            .total
            .checked_sub(self.operand(elements)?)
            .ok_or_else(|| invalid("random source envelope lost its output"))?;
        self.node(elements, &[], extra, false)
    }
    /// PRECONDITION: the source's matching eval and explicit native retirement
    /// fence BOTH succeeded. This symbolic method establishes no native success.
    /// Clearing only evaluated input edges mirrors transforms.cpp's detach;
    /// source histories and aliases held by real factory locals remain alive.
    pub(super) fn materialize_after_join(&self, token: &LiveLatent) {
        fn detach(node: &Rc<Node>) {
            if node.evaluated.replace(true) {
                return;
            }
            let inputs = std::mem::take(&mut *node.inputs.borrow_mut());
            for parent in inputs.into_iter().flatten() {
                detach(&parent);
            }
        }
        detach(&token.0);
    }
}
impl mlx_gen::gen_core::sampling::LatentOps for LiveOps {
    type Latent = LiveLatent;
    fn observe_solver(
        &self,
        _event: mlx_gen::gen_core::sampling::solvers::SolverObservation<'_, LiveLatent>,
    ) {
        // Count the actual generic solver's diagnostic calls; never emit planned
        // metadata as an observed execution or duplicate a solver-specific table.
        self.solver_marks.set(
            self.solver_marks
                .get()
                .and_then(|count| count.checked_add(1)),
        );
    }
    fn scale(&self, x: &LiveLatent, c: f32) -> mlx_gen::gen_core::Result<LiveLatent> {
        if c == 1.0 {
            Ok(x.clone())
        } else {
            self.multiply_always(x).map_err(Into::into)
        }
    }
    fn add(&self, a: &LiveLatent, b: &LiveLatent) -> mlx_gen::gen_core::Result<LiveLatent> {
        self.binary(a, b).map_err(Into::into)
    }
    fn sub(&self, a: &LiveLatent, b: &LiveLatent) -> mlx_gen::gen_core::Result<LiveLatent> {
        self.add(a, b)
    }
    fn axpy(
        &self,
        a: f32,
        x: &LiveLatent,
        _b: f32,
        y: &LiveLatent,
    ) -> mlx_gen::gen_core::Result<LiveLatent> {
        let by = self.multiply_always(y)?;
        if a == 1.0 {
            self.add(x, &by)
        } else {
            let ax = self.multiply_always(x)?;
            self.add(&ax, &by)
        }
    }
    fn randn_like(
        &self,
        x: &LiveLatent,
        _seed: u64,
        _step: usize,
    ) -> mlx_gen::gen_core::Result<LiveLatent> {
        self.random(x.0.elements).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ops() -> LiveOps {
        LiveOps::new(16_384, NativeBackend::Metal, 1 << 20).unwrap()
    }
    #[test]
    fn clones_alias_and_external_owners_survive_completed_join() {
        let ops = ops();
        let x = ops.fresh_materialized(128).unwrap();
        let alias = x.clone();
        let one = ops.active_upper().unwrap();
        assert!(
            one > 0,
            "real retained output must have positive native backing"
        );
        drop(x);
        assert_eq!(ops.active_upper().unwrap(), one);
        let next = ops.multiply_always(&alias).unwrap();
        let pending = ops.active_upper().unwrap();
        ops.materialize_after_join(&next);
        assert_eq!(
            ops.active_upper().unwrap(),
            2 * one,
            "external alias is still a live owner"
        );
        assert!(pending > 2 * one);
        drop(alias);
        assert_eq!(ops.active_upper().unwrap(), one);
        drop(next);
        assert_eq!(ops.active_upper().unwrap(), 0);
    }
    #[test]
    fn lazy_input_and_scalar_live_until_matching_join_not_host_drop() {
        let ops = ops();
        let x = ops.fresh_materialized(128).unwrap();
        let one = ops.active_upper().unwrap();
        let next = ops.multiply_always(&x).unwrap();
        drop(x);
        assert!(
            ops.active_upper().unwrap() > 2 * one,
            "lazy parents retain input and scalar"
        );
        ops.materialize_after_join(&next);
        assert_eq!(ops.active_upper().unwrap(), one);
    }
    #[test]
    fn random_workspace_never_retires_before_real_boundary_precondition() {
        let ops = ops();
        let noise = ops.random(128).unwrap();
        let before = ops.active_upper().unwrap();
        let one = ops.operand(128).unwrap();
        assert!(before > one * 8);
        ops.materialize_after_join(&noise);
        assert_eq!(ops.active_upper().unwrap(), one);
        assert!(ops.peak() >= before);
    }
    #[test]
    fn analysis_budget_is_caller_owned_not_a_step_cap() {
        let denied = LiveOps::new(16_384, NativeBackend::Cpu, 0).unwrap();
        assert!(denied.fresh_materialized(128).is_err());
        assert_eq!(denied.created.get(), 0);
        let allowed = ops();
        assert!(allowed.fresh_materialized(128).is_ok());
    }
    #[test]
    fn actual_solver_factories_keep_source_histories_and_final_owner() {
        use mlx_gen::gen_core::sampling::{denoise, FlowModelSampling, Solver, TimestepConvention};
        for solver in Solver::ALL {
            let ops = ops();
            let initial = ops.random(128).unwrap();
            let sigmas = mlx_gen::gen_core::sampling::build_flow_sigmas(5, 3.0f32.ln());
            let ms = FlowModelSampling::new(TimestepConvention::Sigma);
            let mut predict =
                |input: &LiveLatent, _t: f32| -> mlx_gen::gen_core::Result<LiveLatent> {
                    // Symbolic test boundary only, never a captured native receipt.
                    ops.materialize_after_join(input);
                    ops.fresh_materialized(input.0.elements).map_err(Into::into)
                };
            let mut callback =
                |x: &LiveLatent, sigma: f32| denoise(&ops, &ms, x, sigma, &mut predict);
            let result = solver
                .boxed::<LiveOps>()
                .sample(&ops, &ms, &mut callback, initial, &sigmas, 0)
                .unwrap();
            ops.materialize_after_join(&result);
            assert_eq!(
                ops.active_upper().unwrap(),
                ops.operand(128).unwrap(),
                "{solver:?}: factory-local histories must drop, final owner must not"
            );
            assert!(ops.peak() > ops.active_upper().unwrap());
        }
    }
}
