//! The callback `Sampler` trait + the first solver (Euler) of the unified framework (epic 7114, P1).
//!
//! Replaces the legacy precomputed-`StepCoeffs` design (`super`), which folds the whole step into one
//! affine `a_x·x + a_out·out + a_noise·ε` and so structurally cannot host a solver that needs more
//! than one model evaluation per step (Heun) or the previous steps' denoised history (DPM++ 2M,
//! UniPC). Here a [`Sampler`] drives a `denoise(x, σ) -> x0` callback as many times as its algorithm
//! needs, in ComfyUI / k-diffusion sigma space, advancing the latents through [`LatentOps`] only.
//!
//! The curated solver set (euler_ancestral, heun, dpmpp_2m, dpmpp_sde, uni_pc, lcm, ddim) lands on
//! this trait in sc-7117; this module ships the trait + plain [`Euler`] and proves the new path
//! reproduces the legacy [`super::FlowMatchPolicy`] within the epic's N1 parity gate.

use super::LatentOps;
use crate::{Error, Result};
use std::ops::Range;

/// A `denoise(x, σ) -> x0` callback the samplers drive — the engine's model wrapper (input scaling →
/// model forward → prediction-type recombination, see [`super::model_sampling::denoise`]). Boxed
/// `FnMut` form so the [`Sampler`] trait stays object-safe; `'a` is the closure's borrow.
pub type DenoiseFn<'a, L> =
    dyn FnMut(&<L as LatentOps>::Latent, f32) -> Result<<L as LatentOps>::Latent> + 'a;

/// How a phase transition handles multistep solver history (`PIPE-3`, `PIPE-5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseBoundary {
    /// Continue the same trajectory in the same geometry without re-noising.
    Carry,
    /// Clear solver history after a geometry change or re-noise while preserving absolute
    /// schedule and stochastic draw coordinates.
    Reset,
}

/// Captured primary guided clean estimate (`x0`) from outer step `K - 1` at `sigma_{K-1}`
/// under [`crate::EstimateSource::PreviousEvaluation`] (`SW-6940`: `FEAT-1`, `PIPE-4`, `E3`).
#[derive(Clone, Debug, PartialEq)]
pub struct CapturedPrimaryEstimate<T> {
    pub step: usize,
    pub sigma: f32,
    pub evaluation_kind: crate::EvaluationKind,
    pub solver: super::Solver,
    pub latent: T,
}

/// Numerical history for one request trajectory. The full pass schedule and the next absolute
/// step are retained across inert phase boundaries; resetting history leaves the step position
/// (and therefore stochastic draw addresses) unchanged.
pub struct SolverRunState<L: LatentOps> {
    pub(crate) solver: super::Solver,
    pub(crate) cfg_pp: bool,
    pub(crate) sigmas: Vec<f32>,
    pub(crate) next_step: usize,
    pub(crate) previous_denoised: Option<L::Latent>,
    pub(crate) previous_derivative: Option<L::Latent>,
    pub(crate) previous_sample: Option<L::Latent>,
    pub(crate) previous_interval: Option<f64>,
    pub(crate) history_steps: usize,
    pub(crate) capture_primary_at_step: Option<usize>,
    pub(crate) captured_primary_estimate: Option<CapturedPrimaryEstimate<L::Latent>>,
    seed: Option<u64>,
    sampling: Option<SamplingIdentity>,
    variant: Option<[u32; 3]>,
    pending_reset_reason: Option<&'static str>,
}

#[derive(PartialEq, Eq)]
struct SamplingIdentity {
    prediction: super::PredictionType,
    /// Captures the convention and coefficient contract at every pass node, bit for bit.
    nodes: Vec<[u32; 6]>,
}

impl SamplingIdentity {
    fn from_model(ms: &dyn super::ModelSampling, sigmas: &[f32]) -> Self {
        Self {
            prediction: ms.prediction(),
            nodes: sigmas
                .iter()
                .map(|&sigma| {
                    let (denoise_x, denoise_out) = ms.denoised_coeffs(sigma);
                    let (noise, clean) = ms.noise_scaling_coeffs(sigma);
                    [
                        ms.timestep(sigma).to_bits(),
                        ms.input_scale(sigma).to_bits(),
                        denoise_x.to_bits(),
                        denoise_out.to_bits(),
                        noise.to_bits(),
                        clean.to_bits(),
                    ]
                })
                .collect(),
        }
    }
}

impl<L: LatentOps> SolverRunState<L> {
    pub fn new(solver: super::Solver, sigmas: &[f32]) -> Self {
        Self {
            solver,
            cfg_pp: false,
            sigmas: sigmas.to_vec(),
            next_step: 0,
            previous_denoised: None,
            previous_derivative: None,
            previous_sample: None,
            previous_interval: None,
            history_steps: 0,
            capture_primary_at_step: None,
            captured_primary_estimate: None,
            seed: None,
            sampling: None,
            variant: None,
            pending_reset_reason: None,
        }
    }

    pub fn new_cfg_pp(solver: super::Solver, sigmas: &[f32]) -> Self {
        let mut state = Self::new(solver, sigmas);
        state.cfg_pp = true;
        state
    }

    pub fn next_step(&self) -> usize {
        self.next_step
    }

    /// Whether any multistep solver history is currently retained.
    pub fn has_history(&self) -> bool {
        self.previous_denoised.is_some()
            || self.previous_derivative.is_some()
            || self.previous_sample.is_some()
            || self.previous_interval.is_some()
            || self.history_steps > 0
    }

    /// Borrow the previous step's denoised (`x0`) estimate when retained by a multistep solver.
    pub fn previous_denoised(&self) -> Option<&L::Latent> {
        self.previous_denoised.as_ref()
    }

    /// Opt-in capture of the primary guided clean estimate (`GuidedPrimaryX0`) at outer step
    /// `target_step = K - 1` for [`crate::EstimateSource::PreviousEvaluation`] (`SW-6940`).
    ///
    /// Refuses CFG++ solver state because CFG++ carries dual `(guided, uncond)` re-noising
    /// rather than a single-`x0` callback trajectory.
    pub fn enable_primary_estimate_capture(&mut self, target_step: usize) -> Result<()> {
        if self.cfg_pp {
            return Err(Error::Unsupported(
                "low_res_start estimate_source=previous_evaluation is unsupported on CFG++ solver state".into(),
            ));
        }
        if target_step >= self.sigmas.len().saturating_sub(1) {
            return Err(Error::Unsupported(format!(
                "low_res_start primary estimate capture step {target_step} is out of range for schedule of {} steps",
                self.sigmas.len().saturating_sub(1)
            )));
        }
        self.capture_primary_at_step = Some(target_step);
        self.captured_primary_estimate = None;
        Ok(())
    }

    /// Configure [`SolverRunState`] for a [`crate::ResolvedLowResStartPlan`], enabling
    /// primary clean-estimate capture at step `K - 1` when [`crate::EstimateSource::PreviousEvaluation`]
    /// is selected and leaving [`crate::EstimateSource::EndpointEvaluation`] with zero extra
    /// retained buffers (`SW-6940`).
    pub fn configure_low_res_start(&mut self, plan: &crate::ResolvedLowResStartPlan) -> Result<()> {
        plan.estimate_source
            .validate_solver("low_res_start", self.solver, self.cfg_pp)?;
        match plan.estimate_source {
            crate::EstimateSource::EndpointEvaluation => {
                self.capture_primary_at_step = None;
                self.captured_primary_estimate = None;
                Ok(())
            }
            crate::EstimateSource::PreviousEvaluation => {
                let target_step = plan.k.checked_sub(1).ok_or_else(|| {
                    Error::Unsupported("low_res_start boundary k must be >= 1".into())
                })?;
                self.enable_primary_estimate_capture(target_step)
            }
        }
    }

    /// Target step index configured for primary clean-estimate capture (`Some(K - 1)` when
    /// [`crate::EstimateSource::PreviousEvaluation`] is active, `None` otherwise).
    pub fn capture_primary_at_step(&self) -> Option<usize> {
        self.capture_primary_at_step
    }

    /// Borrow the captured primary clean estimate from step `K - 1`, if currently retained.
    pub fn captured_primary_estimate(&self) -> Option<&CapturedPrimaryEstimate<L::Latent>> {
        self.captured_primary_estimate.as_ref()
    }

    /// Record the primary guided clean estimate `x0` at outer step `step` when opt-in capture
    /// targets this step (`SW-6940`). Secondary solver sub-evaluations (Heun corrector, DPM++ SDE
    /// midpoint) never call this method, so the primary `x0` at `sigma_{K-1}` is preserved while
    /// all sub-evaluations complete normally.
    #[inline]
    pub(crate) fn record_primary_estimate_if_targeted(
        &mut self,
        step: usize,
        sigma: f32,
        guided_x0: &L::Latent,
    ) {
        if self.capture_primary_at_step == Some(step) {
            self.captured_primary_estimate = Some(CapturedPrimaryEstimate {
                step,
                sigma,
                evaluation_kind: crate::EvaluationKind::GuidedPrimaryX0,
                solver: self.solver,
                latent: guided_x0.clone(),
            });
        }
    }

    /// Consume the captured primary guided clean estimate from outer step `expected_step = K - 1`
    /// at the phase boundary, refusing with a typed [`Error::Unsupported`] if capture was not
    /// enabled or if the solver/driver did not produce `GuidedPrimaryX0` at `expected_step` (`SW-6940`).
    pub fn take_captured_primary_estimate(
        &mut self,
        expected_step: usize,
    ) -> Result<CapturedPrimaryEstimate<L::Latent>> {
        self.capture_primary_at_step = None;
        let captured = self.captured_primary_estimate.take().ok_or_else(|| {
            Error::Unsupported(format!(
                "low_res_start estimate_source=previous_evaluation requires a captured GuidedPrimaryX0 at step {expected_step}, but none was recorded by solver '{}'",
                self.solver.name()
            ))
        })?;
        if captured.step != expected_step
            || captured.evaluation_kind != crate::EvaluationKind::GuidedPrimaryX0
        {
            return Err(Error::Unsupported(format!(
                "low_res_start captured primary estimate provenance mismatch: expected (step={expected_step}, GuidedPrimaryX0), got (step={}, {:?})",
                captured.step, captured.evaluation_kind
            )));
        }
        Ok(captured)
    }

    /// Record that a phase boundary changed the latent spatial geometry (`(from_w, from_h) -> (to_w, to_h)`).
    /// Continuing with `Carry` without calling [`Self::reset_history`] will refuse with a typed error.
    pub fn note_geometry_change(&mut self, from: (u32, u32), to: (u32, u32)) {
        if from != to {
            self.pending_reset_reason = Some("geometry change");
        }
    }

    /// Record that a phase boundary re-noised the latent at the transition node.
    /// Continuing with `Carry` without calling [`Self::reset_history`] will refuse with a typed error.
    pub fn note_renoise_transition(&mut self) {
        self.pending_reset_reason = Some("re-noise transition");
    }

    /// Apply a [`PhaseBoundary`] policy across a phase transition, refusing `PhaseBoundary::Carry`
    /// when a geometry change or re-noise occurred (`PIPE-3`, `PIPE-5`).
    pub fn apply_phase_boundary(
        &mut self,
        boundary: PhaseBoundary,
        geometry_changed_or_renoised: bool,
    ) -> Result<()> {
        if geometry_changed_or_renoised && self.pending_reset_reason.is_none() {
            self.pending_reset_reason = Some("geometry change or re-noise");
        }
        match boundary {
            PhaseBoundary::Carry => {
                if let Some(reason) = self.pending_reset_reason {
                    return Err(Error::Unsupported(format!(
                        "solver Carry across a {reason} boundary is incompatible; must reset solver history"
                    )));
                }
                Ok(())
            }
            PhaseBoundary::Reset => {
                self.reset_history();
                Ok(())
            }
        }
    }

    /// A geometry change, re-noise or explicit Reset clears solver history without rewinding the
    /// pass's RNG coordinates. The caller owns separate guidance and driver-cache invalidation.
    pub fn reset_history(&mut self) {
        self.previous_denoised = None;
        self.previous_derivative = None;
        self.previous_sample = None;
        self.previous_interval = None;
        self.history_steps = 0;
        self.capture_primary_at_step = None;
        self.captured_primary_estimate = None;
        self.pending_reset_reason = None;
    }

    /// Select a different solver only at an explicit Reset boundary. The pass step and bound seed
    /// remain absolute; the next span binds its own sampling convention.
    pub fn reset_for_solver(&mut self, solver: super::Solver) {
        self.reset_history();
        self.solver = solver;
        self.cfg_pp = false;
        self.sampling = None;
        self.variant = None;
    }

    /// Enter CFG++ at an explicit Reset boundary. Its guided/unconditional history is distinct
    /// from the standard solver history, while the pass step remains absolute.
    pub fn reset_for_cfg_pp(&mut self, solver: super::Solver) {
        self.reset_history();
        self.solver = solver;
        self.cfg_pp = true;
        self.sampling = None;
        self.variant = None;
    }

    pub(crate) fn check_span(
        &mut self,
        solver: super::Solver,
        ms: &dyn super::ModelSampling,
        sigmas: &[f32],
        seed: u64,
        variant: [u32; 3],
        span: &Range<usize>,
    ) -> Result<()> {
        if let Some(reason) = self.pending_reset_reason {
            return Err(Error::Unsupported(format!(
                "solver Carry across a {reason} boundary is incompatible; call reset_history() before the next span"
            )));
        }
        if self.cfg_pp
            || self.solver != solver
            || self.sigmas != sigmas
            || span.start != self.next_step
            || span.end < span.start
            || span.end > sigmas.len().saturating_sub(1)
            || self.seed.is_some_and(|bound| bound != seed)
            || self.variant.is_some_and(|bound| bound != variant)
        {
            return Err(Error::Unsupported(
                "solver Carry requires one implementation, unchanged pass schedule and contiguous absolute steps".into(),
            ));
        }
        let sampling = SamplingIdentity::from_model(ms, sigmas);
        if self
            .sampling
            .as_ref()
            .is_some_and(|bound| bound != &sampling)
        {
            return Err(Error::Unsupported(
                "solver Carry requires the same numerical sampling convention".into(),
            ));
        }
        self.seed = Some(seed);
        self.sampling = Some(sampling);
        self.variant = Some(variant);
        Ok(())
    }

    pub(crate) fn check_cfg_pp_span(
        &self,
        solver: super::Solver,
        sigmas: &[f32],
        span: &Range<usize>,
    ) -> Result<()> {
        if let Some(reason) = self.pending_reset_reason {
            return Err(Error::Unsupported(format!(
                "CFG++ Carry across a {reason} boundary is incompatible; call reset_history() before the next span"
            )));
        }
        if !self.cfg_pp
            || (self.solver != solver
                && !(solver == super::Solver::Euler && self.solver == super::Solver::Ddim))
            || self.sigmas != sigmas
            || span.start != self.next_step
            || span.end < span.start
            || span.end > sigmas.len().saturating_sub(1)
        {
            return Err(Error::Unsupported(
                "CFG++ Carry requires one implementation, unchanged pass schedule and contiguous absolute steps".into(),
            ));
        }
        Ok(())
    }
}

/// A denoise integrator over a sigma schedule. Implementors call `denoise(x, σ)` — the model wrapper
/// returning a denoised `x0` estimate — and advance the latents using only [`LatentOps`]. The trait
/// is object-safe (`denoise` is a `&mut` [`DenoiseFn`], the method has no generic params) so an
/// engine selects a solver by name into a `Box<dyn Sampler<Ops>>` and the denoise loop never knows
/// which is running.
///
/// Generic over the backend `L` rather than per-method so the box can be built once per request.
pub trait Sampler<L: LatentOps> {
    fn solver(&self) -> super::Solver;

    /// Integrate `x` from `sigmas[0]` down to `sigmas[last]` (the schedule carries a trailing `0.0`).
    ///
    /// - `ops`: backend tensor ops.
    /// - `ms`: the engine's prediction-type + noise-schedule contract. Most solvers ignore it (the
    ///   `denoise` closure already folds it in); the consistency re-noise (`lcm`) reads
    ///   [`super::ModelSampling::noise_scaling_coeffs`] so its between-step re-noise matches the model's
    ///   convention (the flow convex blend vs the VE additive form).
    /// - `denoise`: `denoise(x, σ) -> x0`; called at least once per step.
    /// - `x`: starting latents (already at `sigmas[0]`).
    /// - `sigmas`: descending schedule, length `num_steps + 1`, trailing `0.0`.
    /// - `seed`: request seed for stochastic solvers' per-step noise (deterministic solvers ignore it).
    fn sample(
        &self,
        ops: &L,
        ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        let mut state = SolverRunState::new(self.solver(), sigmas);
        self.sample_span(
            ops,
            ms,
            denoise,
            x,
            sigmas,
            seed,
            0..sigmas.len().saturating_sub(1),
            &mut state,
        )
    }

    /// Integrate a contiguous half-open range of **absolute pass steps**. Every phase supplies
    /// the same whole-pass schedule, so prior intervals and stochastic draw indices are stable.
    #[allow(clippy::too_many_arguments)]
    fn sample_span(
        &self,
        ops: &L,
        ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        x: L::Latent,
        sigmas: &[f32],
        seed: u64,
        span: Range<usize>,
        state: &mut SolverRunState<L>,
    ) -> Result<L::Latent>;
}

/// `to_d(x, σ, x0) = (x − x0) / σ` — the k-diffusion derivative (score-scaled) every ODE solver in
/// the curated set advances along. At `σ == 0` (a terminal node a solver should never integrate
/// *from*) the derivative is undefined; callers guard with [`is_terminal`].
pub(crate) fn to_d<L: LatentOps>(
    ops: &L,
    x: &L::Latent,
    sigma: f32,
    x0: &L::Latent,
) -> Result<L::Latent> {
    ops.scale(&ops.sub(x, x0)?, 1.0 / sigma)
}

/// Whether a schedule node is the terminal clean node (`σ == 0`), from which no step is integrated.
#[inline]
pub(crate) fn is_terminal(sigma: f32) -> bool {
    sigma == 0.0
}

/// Forward-Euler (1st-order) ODE solver — the k-diffusion `sample_euler` with no churn (`gamma = 0`).
///
/// Per step: `x0 = denoise(x, σ_i)`, `d = (x − x0)/σ_i`, `x ← x + d·(σ_{i+1} − σ_i)`. Driven over a
/// FLOW [`super::model_sampling::ModelSampling`] this is algebraically the legacy flow-match step
/// `x + v·(σ_{i+1} − σ_i)` (`d == v`); they differ only by the f32 rounding of the `to_d` round-trip,
/// which the N1 parity gate covers (the production Wan UniPC default already integrates through the
/// same `x0 = sample − v·σ` round-trip). See the `euler_matches_legacy_flow_match_within_eps` test.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euler;

impl<L: LatentOps> Sampler<L> for Euler {
    fn solver(&self) -> super::Solver {
        super::Solver::Euler
    }

    fn sample_span(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
        span: Range<usize>,
        state: &mut SolverRunState<L>,
    ) -> Result<L::Latent> {
        state.check_span(super::Solver::Euler, _ms, sigmas, _seed, [0; 3], &span)?;
        for i in span.clone() {
            let sigma = sigmas[i];
            ops.observe_solver(super::solvers::SolverObservation::callback(
                super::Solver::Euler,
                i,
                sigma,
                sigmas[i + 1],
                super::solvers::SolverEvaluation::Primary,
                &x,
            ));
            let x0 = denoise(&x, sigma)?;
            state.record_primary_estimate_if_targeted(i, sigma, &x0);
            if is_terminal(sigma) {
                // Degenerate leading 0 (no real schedule starts here): land on x0, no division.
                x = x0;
                ops.observe_solver(super::solvers::SolverObservation::update(
                    super::Solver::Euler,
                    i,
                    sigma,
                    sigmas[i + 1],
                    &x,
                    true,
                    false,
                ));
                continue;
            }
            let dt = sigmas[i + 1] - sigma;
            let d = to_d(ops, &x, sigma, &x0)?;
            x = ops.axpy(1.0, &x, dt, &d)?;
            ops.observe_solver(super::solvers::SolverObservation::update(
                super::Solver::Euler,
                i,
                sigma,
                sigmas[i + 1],
                &x,
                false,
                false,
            ));
        }
        state.next_step = span.end;
        Ok(x)
    }
}

/// Apply one legacy [`super::StepCoeffs`] through [`LatentOps`]:
/// `x_next = a_x·x + a_out·out + a_noise·ε`.
///
/// The bridge that lets the legacy coefficient policies — FlowMatch / Lightning / LCM / TCD, the
/// accel lineage that does NOT route through an `x0` callback — run on the unified backend layer
/// during the P3/P4 per-engine migration, bit-identically to mlx-gen's `apply_step`. The byte-parity
/// branch (`a_x == 1 && a_noise == 0 ⇒ x + out·a_out`, via `axpy(1.0, …)` whose `1.0·x` is an f32
/// identity) is preserved. `out` is the RAW (CFG-combined) model output; stochastic re-noise draws
/// from [`LatentOps::randn_like`] (the noise *source* is backend-specific by design — cross-backend
/// bitwise equality of the draw is not a goal, per `StepRng`).
pub fn apply_coeffs<L: LatentOps>(
    ops: &L,
    c: &super::StepCoeffs,
    x: &L::Latent,
    out: &L::Latent,
    seed: u64,
    step: usize,
) -> Result<L::Latent> {
    let acc = ops.axpy(c.a_x, x, c.a_out, out)?;
    if c.a_noise != 0.0 {
        let noise = ops.randn_like(&acc, seed, step)?;
        return ops.axpy(1.0, &acc, c.a_noise, &noise);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::model_sampling::{denoise, FlowModelSampling};
    use crate::sampling::{
        build_flow_sigmas, compute_mu, image_seq_len, AlphaSchedule, CpuLatentOps, FlowMatchPolicy,
        LcmPolicy, LightningPolicy, SamplerPolicy, TcdPolicy, TimestepConvention,
    };

    /// A deterministic, mild flow "model": velocity `v = 0.3·x + 0.1` (ignores the timestep). Stable
    /// enough that the legacy and unified trajectories stay comparable over a full schedule.
    fn stub_velocity(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 0.3 * v + 0.1).collect()
    }

    #[test]
    fn euler_single_flow_step_is_x_plus_v_dt() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        // One step from σ=0.8 to σ=0.5 over a constant velocity.
        let sigmas = [0.8_f32, 0.5];
        let v = vec![0.7_f32, -0.2];
        let x = vec![0.3_f32, 1.0];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let got = Euler
            .sample(&ops, &ms, &mut dn, x.clone(), &sigmas, 0)
            .unwrap();
        // Expected: x + v·(0.5−0.8) = x − 0.3·v.
        for ((g, &xi), &vi) in got.iter().zip(&x).zip(&v) {
            assert!((g - (xi + vi * (0.5 - 0.8))).abs() < 1e-5, "got {g}");
        }
    }

    #[test]
    fn euler_last_step_lands_on_x0() {
        // A schedule whose final node is 0 integrates exactly onto the denoised estimate.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = [0.5_f32, 0.0];
        let v = vec![0.4_f32, 0.4];
        let x = vec![0.2_f32, -0.6];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let got = Euler
            .sample(&ops, &ms, &mut dn, x.clone(), &sigmas, 0)
            .unwrap();
        // x0 at σ=0.5 = x − 0.5·v.
        for ((g, &xi), &vi) in got.iter().zip(&x).zip(&v) {
            assert!((g - (xi - 0.5 * vi)).abs() < 1e-5, "got {g}");
        }
    }

    /// THE byte-equivalence proof (sc-7115 acceptance): the new callback Euler over a FLOW
    /// ModelSampling reproduces the legacy `FlowMatchPolicy` (`x + v·(σ_{i+1}−σ_i)`, byte-parity
    /// branch) within the epic's N1 parity tolerance. Both paths feed the SAME (x, σ) to the SAME
    /// stub velocity model; the only difference is the unified path's `to_d` round-trip, an
    /// f32-cancellation that stays < 1e-5 over a full realistic shifted-flow schedule.
    #[test]
    fn euler_matches_legacy_flow_match_within_eps() {
        let ops = CpuLatentOps;
        // A realistic shifted-flow schedule (the FLUX/Qwen world): 8 steps, empirical μ.
        let mu = compute_mu(image_seq_len(1024, 1024), 8);
        let sigmas = build_flow_sigmas(8, mu);
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05, -0.4, 1.7];

        // Legacy path: FlowMatchPolicy + the apply_step byte-parity branch (x + out·a_out).
        let policy = FlowMatchPolicy::new(sigmas.clone(), TimestepConvention::Sigma);
        let mut legacy = x_init.clone();
        for i in 0..policy.num_steps() {
            let c = policy.coeffs(i); // c_in=1, timestep=σ_i, a_x=1, a_out=σ_{i+1}−σ_i, a_noise=0
            let out = stub_velocity(&legacy); // model input = c_in·x = x; timestep unused by the stub
            legacy = legacy
                .iter()
                .zip(&out)
                .map(|(&xi, &oi)| xi + oi * c.a_out)
                .collect();
        }

        // Unified path: callback Euler over a FLOW ModelSampling, same stub model.
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let mut dn =
            |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, _t| Ok(stub_velocity(xin)));
        let unified = Euler
            .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();

        assert_eq!(legacy.len(), unified.len());
        let max_abs = legacy
            .iter()
            .zip(&unified)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "unified Euler diverged from legacy FlowMatchPolicy by {max_abs:e} (> 1e-5)"
        );
    }

    #[test]
    fn euler_is_object_safe_boxed() {
        // Runtime solver selection: the trait must box (`Box<dyn Sampler<Ops>>`).
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sampler: Box<dyn Sampler<CpuLatentOps>> = Box::new(Euler);
        let v = vec![0.5_f32];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let out = sampler
            .sample(&ops, &ms, &mut dn, vec![0.1_f32], &[0.6_f32, 0.0], 0)
            .unwrap();
        assert_eq!(out.len(), 1);
    }

    /// `apply_coeffs` reproduces the legacy affine step (`a_x·x + a_out·out`) BIT-FOR-BIT for every
    /// policy family — the accel lineage the story names (FlowMatch / Lightning / LCM / TCD). This is
    /// the foundation-level "accel policies on the new interface, byte-equivalence proven": the
    /// per-engine callback samplers for `lcm`/`ddim` land in sc-7117 and the engine adoption in
    /// P3/P4, but the LatentOps layer already subsumes the legacy `apply_step` exactly here.
    #[test]
    fn apply_coeffs_reproduces_legacy_policy_steps_bit_exact() {
        let ops = CpuLatentOps;
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
        let x = vec![0.3_f32, -1.2, 2.5, 0.07];
        let out = vec![0.7_f32, 0.1, -0.4, 1.3];
        // Deterministic coeff sets (a_noise = 0) from each policy family.
        let policies: Vec<(&str, super::super::StepCoeffs)> = vec![
            (
                "flow",
                FlowMatchPolicy::new(vec![1.0, 0.75, 0.5, 0.25, 0.0], TimestepConvention::Sigma)
                    .coeffs(1),
            ),
            ("lightning", LightningPolicy::new(&sched, 1000, 4).coeffs(0)),
            (
                "lcm_last", // final step never re-noises -> deterministic
                LcmPolicy::new(sched.clone(), 1000, 50, 4).coeffs(3),
            ),
            (
                "tcd_eta0",
                TcdPolicy::new(sched.clone(), 1000, 50, 4, 0.0).coeffs(0),
            ),
        ];
        for (name, c) in policies {
            assert_eq!(c.a_noise, 0.0, "{name}: expected a deterministic coeff set");
            let got = apply_coeffs(&ops, &c, &x, &out, 0, 0).unwrap();
            // Legacy reference: a_x·x + a_out·out (the byte-parity branch is a_x==1 special-case;
            // both reduce to the same f32 arithmetic on the CPU backend).
            let want: Vec<f32> = x
                .iter()
                .zip(&out)
                .map(|(&xi, &oi)| c.a_x * xi + c.a_out * oi)
                .collect();
            assert_eq!(
                got, want,
                "{name}: apply_coeffs diverged from legacy affine step"
            );
        }
    }
}
