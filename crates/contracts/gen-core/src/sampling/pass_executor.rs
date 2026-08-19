//! The **chained denoise-pass executor** (epic 20414, sc-20418) — the model-agnostic runtime that
//! drives a [`ResolvedDenoisePlan`] through a provider's model **without reloading it**, over any
//! [`LatentOps`] backend.
//!
//! This module is the executable half of the [`crate::denoise_passes`] contract. The runtime
//! invariant is stated there once; this is where it is enforced:
//!
//! 1. **Pass 0 consumes the caller's initial latent verbatim** (the request's init noise for t2i, or
//!    the blended reference latent for img2img). That is what makes a one-pass plan bit-identical to
//!    the legacy single-trajectory render — see `one_pass_plan_matches_a_direct_solver_run` below.
//! 2. **Every later pass deterministically re-noises the current latent in latent space** with that
//!    pass's [resolved seed](crate::ResolvedDenoisePass::seed), through the model's own
//!    [`ModelSampling::noise_scaling_coeffs`] convention (the flow convex blend for rectified-flow
//!    models, the VE additive form for ε/EDM). No VAE decode/encode happens at a boundary — the
//!    executor never sees a decoder at all.
//! 3. **Each pass gets a fresh full schedule** from the host ([`DenoisePassHost::build_schedule`]),
//!    enters it at the terminal `steps + 1` segment selected by its `denoise` fraction
//!    (Comfy-style: the full schedule is built for `⌊steps / denoise⌋` steps and the last
//!    `steps + 1` nodes are kept — [`denoise_pass_schedule_steps`] / [`terminal_pass_segment`]),
//!    and is integrated by a **freshly boxed solver** — so multistep history
//!    ([`AbnorsettState`](super::AbnorsettState), UniPC/DPM++ buffers) structurally cannot leak
//!    across a pass boundary: each [`Sampler::sample`](super::Sampler::sample) call owns its own
//!    state.
//! 4. **Progress is aggregated by effective model evaluations** across the whole chain
//!    ([`Solver::model_evals`] per pass segment): one [`Progress::Step`] per model evaluation,
//!    `current` strictly `1..=total` over the chain, so a 2-pass `rk6_7s` + `euler` chain reports a
//!    single honest bar instead of restarting per pass.
//! 5. **Cancellation is checked at every evaluation and at every pass boundary**, surfacing as the
//!    typed [`Error::Canceled`]; [`DenoisePassHost::end_pass`] runs even when the pass errored, so a
//!    host's pass-local overrides (guidance, adapter weights) are always reverted and the provider
//!    stays reusable.
//!
//! ## Adoption
//!
//! A provider adopts the executor by implementing [`DenoisePassHost`] over its backend's
//! `LatentOps` — schedule building, the model forward (with any CFG combine inside), pass-local
//! override application/reversal, and an optional per-evaluation observation hook for previews. The
//! executor owns everything else. The gen-core-testkit `denoise_passes` module runs a
//! backend-neutral conformance suite over exactly this contract, so a non-Krea family (sc-20425)
//! adopts by supplying its schedule builder — the executor contract does not change per family.
//!
//! ## Extensibility note (deliberate)
//!
//! Per-pass resolution flows through [`ResolvedDenoisePass`] and the host hooks receive the whole
//! pass, so adding an *optional* per-pass field later (e.g. explicit `bong_tangent` slope/pivot
//! scheduler params, flagged during sc-20416 review) extends the pass struct and the host's
//! `build_schedule` reading of it — no executor signature changes.

use super::model_sampling::{denoise as gc_denoise, ModelSampling};
use super::solvers::Solver;
use super::LatentOps;
use crate::denoise_passes::{
    DenoisePassExecutionRecord, DenoisePlanExecution, ResolvedDenoisePass, ResolvedDenoisePlan,
};
use crate::error::Error;
use crate::runtime::{CancelFlag, Progress};
use crate::Result;

/// Domain-separation salt for the **between-pass re-noise** draw. Distinct from the per-step
/// stochastic-solver subkey salt (`0x9E37…`, which mixes the *pass seed* with the step index inside
/// a pass), so the boundary re-noise can never collide with an ancestral/SDE solver's per-step
/// draws. The constant is the fractional part of √2 in fixed point (the SHA-256 IV convention for
/// nothing-up-my-sleeve numbers).
pub const DENOISE_PASS_RENOISE_SALT: u64 = 0x6A09_E667_F3BC_C908;

/// The seed for a pass's boundary re-noise draw: splitmix64 over the pass seed xor
/// [`DENOISE_PASS_RENOISE_SALT`]. Pure and deterministic, so a replay re-noises identically.
#[inline]
#[must_use]
pub fn denoise_pass_renoise_seed(pass_seed: u64) -> u64 {
    let mut z = pass_seed ^ DENOISE_PASS_RENOISE_SALT;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The step count a pass's **full fresh schedule** is built for — ComfyUI `KSampler` denoise
/// semantics: `denoise ≥ ~1` builds the schedule for `steps` directly; a partial denoise builds the
/// schedule for `⌊steps / denoise⌋` total steps (so the executed tail has the same per-step noise
/// spacing a full run of that length would), and [`terminal_pass_segment`] keeps the last
/// `steps + 1` nodes.
#[must_use]
pub fn denoise_pass_schedule_steps(steps: u32, denoise: f32) -> usize {
    let steps = steps.max(1) as usize;
    // ComfyUI: `if denoise > 0.9999: sigmas(steps) else: sigmas(int(steps / denoise))`. The
    // division runs in f32 — the contract's own precision — so a decimal like `0.05` (whose f32 is
    // fractionally above the decimal value) still yields the intended `steps / denoise` (an f64
    // widening of the f32 would truncate `1 / 0.05` to 19).
    if denoise > 0.9999 {
        return steps;
    }
    // Validation bounds denoise to (0, 1], so the division is finite and >= steps.
    ((steps as f32 / denoise.max(f32::MIN_POSITIVE)) as usize).max(steps)
}

/// The terminal `steps + 1`-node segment of a pass's full schedule — the part the pass actually
/// integrates. A re-striding curated scheduler may return fewer nodes than requested; the segment is
/// then the whole schedule (the pass simply runs shorter, exactly as the single-pass path would).
#[must_use]
pub fn terminal_pass_segment(sigmas: &[f32], steps: u32) -> &[f32] {
    let keep = (steps.max(1) as usize + 1).min(sigmas.len());
    &sigmas[sigmas.len() - keep..]
}

/// One executor observation — the per-evaluation hook a host uses for previews. `chain_step` is the
/// 0-based **outer step index across the whole chain** (multi-eval solvers repeat it, exactly like
/// the per-driver preview counters expect), bounded by `chain_total_steps`.
pub struct PassObservation<'a, T> {
    /// Which pass is running.
    pub pass_index: usize,
    /// 0-based outer step index across the whole chain (pass offsets included).
    pub chain_step: usize,
    /// Total outer steps across every pass segment.
    pub chain_total_steps: usize,
    /// The σ this evaluation runs at.
    pub sigma: f32,
    /// The running latent (never the `c_in`-scaled model input).
    pub latent: &'a T,
}

/// What a provider supplies to run a denoise-pass chain — schedule math, the model forward, and the
/// pass-local override lifecycle. One host serves one generation (one image of a batch).
pub trait DenoisePassHost<L: LatentOps> {
    /// Build the **full fresh** σ schedule for this pass's resolved scheduler at `schedule_steps`
    /// steps (descending, trailing `0.0` — the family's ordinary schedule builder). Called once per
    /// pass, before any integration.
    fn build_schedule(
        &mut self,
        pass: &ResolvedDenoisePass,
        schedule_steps: usize,
    ) -> Result<Vec<f32>>;

    /// Apply this pass's pass-local state (guidance selection, adapter weight overrides). Called
    /// after the boundary re-noise and before the first evaluation of the pass.
    fn begin_pass(&mut self, _pass: &ResolvedDenoisePass) -> Result<()> {
        Ok(())
    }

    /// The model forward at `(x_in, timestep)` returning the RAW (already CFG-combined) output the
    /// model's prediction type expects — velocity for FLOW, ε for EPS. Any per-pass guidance
    /// combine lives inside this call.
    fn predict(
        &mut self,
        pass: &ResolvedDenoisePass,
        x: &L::Latent,
        timestep: f32,
    ) -> Result<L::Latent>;

    /// Revert this pass's pass-local state. Runs even when the pass errored or was cancelled, so
    /// the provider is never left with pass-local overrides applied (no poisoned state).
    fn end_pass(&mut self, _pass: &ResolvedDenoisePass) -> Result<()> {
        Ok(())
    }

    /// Per-evaluation observation of the running latent (previews). Failures must be swallowed by
    /// the host — observation is decorative and never fails a render.
    fn observe(&mut self, _observation: PassObservation<'_, L::Latent>) {}
}

/// A completed chain: the final latent (ready for ONE decode by the caller) plus the execution
/// record for upstream metadata.
#[derive(Debug)]
pub struct ExecutedDenoisePlan<T> {
    /// The latent the last pass produced. The executor never decodes — the caller owns the single
    /// VAE (or PiD) decode after the chain.
    pub latent: T,
    /// What actually ran, per pass ([`DenoisePlanExecution`]).
    pub execution: DenoisePlanExecution,
}

struct PreparedPass {
    segment: Vec<f32>,
    solver: Solver,
    schedule_steps: usize,
    schedule_len: usize,
    model_evals: usize,
}

/// Effective model evaluations of `solver` over `segment` — [`Solver::model_evals`] for the
/// ordinary trailing-zero segment; a truncated segment (no terminal `0.0`) has no cheap terminal
/// step, so every step costs the interior rate.
fn segment_model_evals(solver: Solver, segment: &[f32]) -> usize {
    let n = segment.len().saturating_sub(1);
    if n == 0 {
        return 0;
    }
    if segment.last().copied() == Some(0.0) {
        solver.model_evals(n)
    } else {
        match solver {
            Solver::Heun | Solver::DpmppSde => 2 * n,
            Solver::Rk67s => 7 * n,
            _ => n,
        }
    }
}

/// Execute a resolved denoise-pass plan over `initial` — the shared, model-agnostic chain runtime.
///
/// See the [module docs](self) for the invariant. `ms` is the model's prediction-type contract
/// (its `noise_scaling_coeffs` drives the boundary re-noise); `initial` is the request's initial
/// latent, consumed verbatim by pass 0. The caller decodes `ExecutedDenoisePlan::latent` exactly
/// once after the chain — the executor holds at most the running latent and one boundary noise
/// tensor at a time, and never a decoded image.
pub fn execute_denoise_plan<L: LatentOps + 'static>(
    ops: &L,
    ms: &dyn ModelSampling,
    plan: &ResolvedDenoisePlan,
    initial: L::Latent,
    host: &mut dyn DenoisePassHost<L>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<ExecutedDenoisePlan<L::Latent>> {
    if plan.passes.is_empty() {
        return Err(Error::Msg(
            "denoise-pass executor: a resolved plan must carry at least one pass".to_owned(),
        ));
    }
    // Prepare every pass first (host schedule math is cheap) so the chain-wide evaluation budget is
    // known before the first model forward — the progress denominator never changes mid-run.
    let mut prepared: Vec<PreparedPass> = Vec::with_capacity(plan.passes.len());
    for pass in &plan.passes {
        let schedule_steps = denoise_pass_schedule_steps(pass.steps, pass.denoise);
        let sigmas = host.build_schedule(pass, schedule_steps)?;
        if sigmas.len() < 2 {
            return Err(Error::Msg(format!(
                "denoise-pass executor: pass {} scheduler {:?} produced a degenerate schedule \
                 ({} node(s))",
                pass.index,
                pass.scheduler,
                sigmas.len()
            )));
        }
        let segment = terminal_pass_segment(&sigmas, pass.steps).to_vec();
        // N3: an id outside the curated registry (a family-native alias) integrates as Euler,
        // exactly as the single-pass drivers fall back.
        let solver = Solver::from_name(&pass.sampler).unwrap_or(Solver::Euler);
        prepared.push(PreparedPass {
            model_evals: segment_model_evals(solver, &segment),
            schedule_steps,
            schedule_len: sigmas.len(),
            segment,
            solver,
        });
    }
    let total_evals: usize = prepared.iter().map(|p| p.model_evals).sum();
    let chain_total_steps: usize = prepared
        .iter()
        .map(|p| p.segment.len().saturating_sub(1))
        .sum();
    let progress_total = (total_evals as u32).max(1);

    let mut latent = initial;
    let mut evals_done: usize = 0;
    let mut steps_before: usize = 0;
    let mut records: Vec<DenoisePassExecutionRecord> = Vec::with_capacity(plan.passes.len());

    for (pass, prep) in plan.passes.iter().zip(&prepared) {
        // Pass boundary: a safe cancellation point before any pass-local work.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let renoised = pass.index > 0;
        if renoised {
            // Deterministic latent-space re-noise at the segment's entry σ, through the model's own
            // noise convention: x = k_x0·x + k_noise·ε. Drawn on a dedicated seed domain so it can
            // never collide with a stochastic solver's per-step draws.
            let sigma_start = prep.segment[0];
            let (k_noise, k_x0) = ms.noise_scaling_coeffs(sigma_start);
            let noise = ops.randn_like(&latent, denoise_pass_renoise_seed(pass.seed), 0)?;
            latent = ops.axpy(k_x0, &latent, k_noise, &noise)?;
        }

        host.begin_pass(pass)?;
        let segment = prep.segment.as_slice();
        let seg_steps = segment.len().saturating_sub(1);
        let result = {
            // A freshly boxed solver per pass: multistep history cannot leak across the boundary.
            let sampler = prep.solver.boxed::<L>();
            let mut denoise_fn = |x: &L::Latent, sigma: f32| -> Result<L::Latent> {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                evals_done += 1;
                on_progress(Progress::Step {
                    current: (evals_done as u32).min(progress_total),
                    total: progress_total,
                });
                // Outer step within the pass: how many segment nodes were already descended past
                // (multi-eval solvers repeat a value; the host's preview counter dedups).
                let step_in_pass = segment
                    .iter()
                    .filter(|&&s| s > sigma)
                    .count()
                    .min(seg_steps.saturating_sub(1));
                host.observe(PassObservation {
                    pass_index: pass.index,
                    chain_step: steps_before + step_in_pass,
                    chain_total_steps,
                    sigma,
                    latent: x,
                });
                gc_denoise(ops, ms, x, sigma, |xin, t| host.predict(pass, xin, t))
            };
            sampler.sample(ops, ms, &mut denoise_fn, latent, segment, pass.seed)
        };
        // The pass-local overrides are reverted on EVERY exit path; the run error, when present,
        // stays the primary error.
        let ended = host.end_pass(pass);
        latent = result?;
        ended?;

        steps_before += seg_steps;
        records.push(DenoisePassExecutionRecord {
            requested: None,
            resolved: pass.clone(),
            schedule_steps: prep.schedule_steps,
            schedule_len: prep.schedule_len,
            effective_steps: seg_steps,
            model_evals: prep.model_evals,
            renoised,
        });
    }

    Ok(ExecutedDenoisePlan {
        latent,
        execution: DenoisePlanExecution {
            job_seed: plan.job_seed,
            passes: records,
            total_model_evals: total_evals,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise_passes::{
        denoise_pass_seed, resolve_denoise_plan, DenoiseDefaults, DenoisePass, DenoisePassContext,
    };
    use crate::generator::GenerationRequest;
    use crate::sampling::{
        build_flow_sigmas, compute_mu, image_seq_len, sampler_by_name, CpuLatentOps,
        FlowModelSampling, TimestepConvention,
    };

    fn ms() -> FlowModelSampling {
        FlowModelSampling::new(TimestepConvention::Sigma)
    }

    fn flow_sigmas(steps: usize) -> Vec<f32> {
        build_flow_sigmas(steps, compute_mu(image_seq_len(512, 512), steps))
    }

    /// A stub host over `CpuLatentOps`: the family schedule is the shared flow schedule, the model
    /// is the smooth linear velocity field `v = 0.25·x + 0.1`, and every lifecycle call is logged.
    struct StubHost {
        log: Vec<String>,
        fail_predict_at_eval: Option<usize>,
        evals: usize,
        observed: Vec<(usize, usize, f32)>,
        first_latent_of_pass: Vec<Vec<f32>>,
        seen_pass: Option<usize>,
    }

    impl StubHost {
        fn new() -> Self {
            Self {
                log: Vec::new(),
                fail_predict_at_eval: None,
                evals: 0,
                observed: Vec::new(),
                first_latent_of_pass: Vec::new(),
                seen_pass: None,
            }
        }
    }

    impl DenoisePassHost<CpuLatentOps> for StubHost {
        fn build_schedule(
            &mut self,
            pass: &ResolvedDenoisePass,
            schedule_steps: usize,
        ) -> Result<Vec<f32>> {
            self.log.push(format!("schedule[{}]", pass.index));
            Ok(flow_sigmas(schedule_steps))
        }
        fn begin_pass(&mut self, pass: &ResolvedDenoisePass) -> Result<()> {
            self.log.push(format!("begin[{}]", pass.index));
            Ok(())
        }
        fn predict(
            &mut self,
            _pass: &ResolvedDenoisePass,
            x: &Vec<f32>,
            _timestep: f32,
        ) -> Result<Vec<f32>> {
            self.evals += 1;
            if self.fail_predict_at_eval == Some(self.evals) {
                return Err(Error::Msg("stub predict fault".to_owned()));
            }
            Ok(x.iter().map(|v| 0.25 * v + 0.1).collect())
        }
        fn end_pass(&mut self, pass: &ResolvedDenoisePass) -> Result<()> {
            self.log.push(format!("end[{}]", pass.index));
            Ok(())
        }
        fn observe(&mut self, obs: PassObservation<'_, Vec<f32>>) {
            self.observed
                .push((obs.pass_index, obs.chain_step, obs.sigma));
            if self.seen_pass != Some(obs.pass_index) {
                self.seen_pass = Some(obs.pass_index);
                self.first_latent_of_pass.push(obs.latent.clone());
            }
        }
    }

    fn x_init() -> Vec<f32> {
        vec![0.3, -1.1, 2.0, 0.05]
    }

    fn plan_for(passes: Vec<DenoisePass>, job_seed: u64) -> ResolvedDenoisePlan {
        let req = GenerationRequest {
            denoise_passes: Some(passes),
            ..Default::default()
        };
        resolve_denoise_plan(
            &req,
            job_seed,
            &DenoiseDefaults::new(6, "euler", "normal"),
            &DenoisePassContext::registry_only(),
        )
        .expect("plan resolves")
    }

    fn pass(steps: u32, sampler: &str, denoise: f32) -> DenoisePass {
        DenoisePass {
            steps: Some(steps),
            sampler: Some(sampler.to_owned()),
            scheduler: Some("normal".to_owned()),
            denoise: Some(denoise),
            ..Default::default()
        }
    }

    fn run(
        plan: &ResolvedDenoisePlan,
        host: &mut StubHost,
    ) -> Result<ExecutedDenoisePlan<Vec<f32>>> {
        let cancel = CancelFlag::new();
        run_with(plan, host, &cancel, &mut |_| {})
    }

    fn run_with(
        plan: &ResolvedDenoisePlan,
        host: &mut StubHost,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<ExecutedDenoisePlan<Vec<f32>>> {
        execute_denoise_plan(
            &CpuLatentOps,
            &ms(),
            plan,
            x_init(),
            host,
            cancel,
            on_progress,
        )
    }

    // --- Comfy-style semantics ------------------------------------------------------------------

    #[test]
    fn schedule_expansion_and_terminal_segment_follow_comfy_semantics() {
        // denoise == 1.0: the schedule is built for `steps` directly.
        assert_eq!(denoise_pass_schedule_steps(4, 1.0), 4);
        // Partial denoise: int(steps / denoise) total, terminal steps+1 kept.
        assert_eq!(denoise_pass_schedule_steps(4, 0.5), 8);
        assert_eq!(denoise_pass_schedule_steps(10, 0.3), 33); // int(10 / 0.3)
        assert_eq!(denoise_pass_schedule_steps(1, 0.05), 20);
        // The f32-division contract: the same values Comfy's float math lands on.
        assert_eq!(denoise_pass_schedule_steps(2, 0.4), 5);

        let sigmas: Vec<f32> = (0..=8).map(|i| 1.0 - i as f32 / 8.0).collect();
        let seg = terminal_pass_segment(&sigmas, 4);
        assert_eq!(seg.len(), 5);
        assert_eq!(seg, &sigmas[4..]);
        assert_eq!(*seg.last().unwrap(), 0.0);
        // A re-strided (shorter) schedule degrades to the whole schedule, never panics.
        let short = [0.6_f32, 0.3, 0.0];
        assert_eq!(terminal_pass_segment(&short, 10), &short);
    }

    // --- AC1: one-pass equivalence --------------------------------------------------------------

    #[test]
    fn one_pass_plan_matches_a_direct_solver_run_bitwise() {
        for sampler in [
            "euler",
            "dpmpp_2m",
            "rk6_7s",
            "abnorsett_4m",
            "euler_ancestral",
        ] {
            let job_seed = 42_u64;
            let plan = plan_for(vec![pass(6, sampler, 1.0)], job_seed);
            let mut host = StubHost::new();
            let got = run(&plan, &mut host).expect("chain runs").latent;

            // The legacy single-pass shape: the same solver driven directly over the same schedule
            // with the same seed and the same denoise wrapper composition.
            let ops = CpuLatentOps;
            let m = ms();
            let sigmas = flow_sigmas(6);
            let solver = sampler_by_name::<CpuLatentOps>(sampler).unwrap();
            let mut dn = |x: &Vec<f32>, s: f32| {
                gc_denoise(&ops, &m, x, s, |xin, _t| {
                    Ok(xin.iter().map(|v| 0.25 * v + 0.1).collect())
                })
            };
            let want = solver
                .sample(&ops, &m, &mut dn, x_init(), &sigmas, job_seed)
                .unwrap();
            assert_eq!(got, want, "{sampler}: one-pass chain != direct run");
        }
    }

    // --- AC2: re-noise, resets, seed isolation ---------------------------------------------------

    #[test]
    fn later_passes_renoise_the_running_latent_with_the_pass_seed() {
        let job_seed = 7_u64;
        let plan = plan_for(vec![pass(4, "euler", 1.0), pass(3, "euler", 0.5)], job_seed);
        let mut host = StubHost::new();
        let out = run(&plan, &mut host).expect("chain runs");

        // Pass 1 ran alone first: reproduce its output.
        let one = plan_for(vec![pass(4, "euler", 1.0)], job_seed);
        let mut host_one = StubHost::new();
        let after_pass_1 = run(&one, &mut host_one).expect("pass 1 runs").latent;

        // The first latent pass 2 observed must be the deterministic re-noise of pass 1's output at
        // the segment entry σ — the exact flow convex blend with the pass-2 seed's dedicated
        // renoise stream. This pins BOTH that re-noise happens AND its exact convention.
        let ops = CpuLatentOps;
        let m = ms();
        let full = flow_sigmas(denoise_pass_schedule_steps(3, 0.5));
        let seg = terminal_pass_segment(&full, 3);
        let (k_noise, k_x0) = m.noise_scaling_coeffs(seg[0]);
        let pass2_seed = denoise_pass_seed(job_seed, 1);
        let noise = ops
            .randn_like(&after_pass_1, denoise_pass_renoise_seed(pass2_seed), 0)
            .unwrap();
        let expected_entry = ops.axpy(k_x0, &after_pass_1, k_noise, &noise).unwrap();
        assert_eq!(host.first_latent_of_pass.len(), 2);
        assert_eq!(host.first_latent_of_pass[1], expected_entry);
        assert_ne!(
            host.first_latent_of_pass[1], after_pass_1,
            "pass 2 must actually perturb the latent before denoising"
        );
        // And pass 2 then denoises: the final output differs from the perturbed entry.
        assert_ne!(out.latent, expected_entry);

        // Execution record agrees.
        assert!(!out.execution.passes[0].renoised);
        assert!(out.execution.passes[1].renoised);
    }

    #[test]
    fn renoise_seed_is_pass_isolated_and_the_chain_is_reproducible() {
        let plan = plan_for(vec![pass(3, "euler", 1.0), pass(3, "euler", 0.5)], 99);
        let a = run(&plan, &mut StubHost::new()).unwrap().latent;
        let b = run(&plan, &mut StubHost::new()).unwrap().latent;
        assert_eq!(a, b, "same request must reproduce identical output");

        // A different job seed shifts every pass seed, so the chain output differs.
        let other = plan_for(vec![pass(3, "euler", 1.0), pass(3, "euler", 0.5)], 100);
        let c = run(&other, &mut StubHost::new()).unwrap().latent;
        assert_ne!(a, c, "different job seed must produce different output");

        // Different pass indices draw different renoise noise even for the same job seed.
        let s1 = denoise_pass_renoise_seed(denoise_pass_seed(99, 1));
        let s2 = denoise_pass_renoise_seed(denoise_pass_seed(99, 2));
        assert_ne!(s1, s2);
        // And the renoise domain never equals the raw pass seed (solver subkey domain).
        assert_ne!(denoise_pass_renoise_seed(99), 99);
    }

    /// The multistep-state reset: a 2-pass abnorsett chain equals hand-running the solver with a
    /// FRESH state per segment (renoise applied between), and differs from continuing the pass-1
    /// history into pass 2 — proving history does not leak across the boundary.
    #[test]
    fn solver_history_resets_at_every_pass_boundary() {
        use crate::sampling::{Abnorsett4m, AbnorsettState};
        let job_seed = 5_u64;
        let plan = plan_for(
            vec![pass(5, "abnorsett_4m", 1.0), pass(4, "abnorsett_4m", 1.0)],
            job_seed,
        );
        let got = run(&plan, &mut StubHost::new()).unwrap().latent;

        let ops = CpuLatentOps;
        let m = ms();
        let mut dn = |x: &Vec<f32>, s: f32| {
            gc_denoise(&ops, &m, x, s, |xin, _t| {
                Ok(xin.iter().map(|v| 0.25_f32 * v + 0.1).collect())
            })
        };
        let seg_a = flow_sigmas(5);
        let seg_b = flow_sigmas(4);
        let renoise = |x: &Vec<f32>, pass_index: usize, sigma: f32| {
            let seed = denoise_pass_renoise_seed(denoise_pass_seed(job_seed, pass_index));
            let noise = ops.randn_like(x, seed, 0).unwrap();
            let (kn, kx) = m.noise_scaling_coeffs(sigma);
            ops.axpy(kx, x, kn, &noise).unwrap()
        };

        // Fresh state per pass == the executor.
        let mut fresh_a = AbnorsettState::default();
        let x_a = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_init(), &seg_a, &mut fresh_a)
            .unwrap();
        let entry_b = renoise(&x_a, 1, seg_b[0]);
        let mut fresh_b = AbnorsettState::default();
        let want = Abnorsett4m
            .sample_with_state(&ops, &mut dn, entry_b.clone(), &seg_b, &mut fresh_b)
            .unwrap();
        assert_eq!(got, want, "executor must equal fresh-state-per-segment");

        // Why fresh-per-pass is load-bearing: whenever λ keeps strictly increasing across a
        // boundary (the regime the solver's defensive self-clear does NOT cover), a carried
        // history IS a different trajectory. Demonstrated at the solver seam the executor drives.
        let seg_hi = [0.9_f32, 0.7, 0.5];
        let seg_lo = [0.4_f32, 0.3, 0.2];
        let mut carried = AbnorsettState::default();
        let mid = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_init(), &seg_hi, &mut carried)
            .unwrap();
        let with_history = Abnorsett4m
            .sample_with_state(&ops, &mut dn, mid.clone(), &seg_lo, &mut carried)
            .unwrap();
        let mut fresh = AbnorsettState::default();
        let without_history = Abnorsett4m
            .sample_with_state(&ops, &mut dn, mid, &seg_lo, &mut fresh)
            .unwrap();
        assert_ne!(
            with_history, without_history,
            "in the λ-increasing regime a carried history changes the trajectory — the reset the              executor performs per pass is observable, not decorative"
        );
    }

    // --- Progress by effective model evaluations -------------------------------------------------

    #[test]
    fn progress_counts_effective_model_evals_across_the_chain() {
        // rk6_7s (7n−6) then euler (n): the bar is ONE monotone 1..=total sweep.
        let plan = plan_for(vec![pass(3, "rk6_7s", 1.0), pass(4, "euler", 1.0)], 11);
        let mut host = StubHost::new();
        let mut events: Vec<(u32, u32)> = Vec::new();
        let cancel = CancelFlag::new();
        let out = run_with(&plan, &mut host, &cancel, &mut |p| {
            if let Progress::Step { current, total } = p {
                events.push((current, total));
            }
        })
        .unwrap();

        let want_total = Solver::Rk67s.model_evals(3) + Solver::Euler.model_evals(4);
        assert_eq!(out.execution.total_model_evals, want_total);
        assert_eq!(events.len(), want_total);
        assert!(events.iter().all(|&(_, t)| t as usize == want_total));
        let currents: Vec<u32> = events.iter().map(|&(c, _)| c).collect();
        assert_eq!(
            currents,
            (1..=want_total as u32).collect::<Vec<_>>(),
            "current must sweep exactly 1..=total, once per model evaluation"
        );
        // Pass/step detail is retained in the records.
        assert_eq!(out.execution.passes[0].model_evals, 7 * 3 - 6);
        assert_eq!(out.execution.passes[1].model_evals, 4);
        assert_eq!(out.execution.passes[0].effective_steps, 3);
        assert_eq!(out.execution.passes[1].effective_steps, 4);
    }

    #[test]
    fn observation_exposes_chain_global_outer_steps() {
        let plan = plan_for(vec![pass(2, "heun", 1.0), pass(3, "euler", 1.0)], 3);
        let mut host = StubHost::new();
        run(&plan, &mut host).unwrap();
        // heun over 2 steps: evals at outer steps 0,1 (interior twice), then terminal; euler pass
        // continues at chain steps 2,3,4. Dedup is the consumer's job; coverage is the executor's.
        let steps: Vec<usize> = host.observed.iter().map(|&(_, s, _)| s).collect();
        assert_eq!(steps.iter().max(), Some(&4));
        assert_eq!(steps.iter().min(), Some(&0));
        // Every chain step 0..=4 appears at least once.
        for want in 0..5 {
            assert!(steps.contains(&want), "chain step {want} never observed");
        }
        // Pass indices tag the observations.
        assert!(host.observed.iter().any(|&(p, _, _)| p == 0));
        assert!(host.observed.iter().any(|&(p, _, _)| p == 1));
    }

    // --- Cancellation + lifecycle ----------------------------------------------------------------

    #[test]
    fn cancellation_mid_chain_is_typed_and_reverts_the_pass() {
        let plan = plan_for(vec![pass(3, "euler", 1.0), pass(3, "euler", 1.0)], 1);
        let mut host = StubHost::new();
        let cancel = CancelFlag::new();
        let trip_after = 4_u32; // inside pass 2
        let mut seen = 0_u32;
        let err = run_with(&plan, &mut host, &cancel, &mut |p| {
            if let Progress::Step { .. } = p {
                seen += 1;
                if seen == trip_after {
                    cancel.cancel();
                }
            }
        })
        .expect_err("a tripped flag must cancel the chain");
        assert!(matches!(err, Error::Canceled));
        // end_pass ran for the cancelled pass — the host is never left with overrides applied.
        assert_eq!(
            host.log,
            vec![
                "schedule[0]",
                "schedule[1]",
                "begin[0]",
                "end[0]",
                "begin[1]",
                "end[1]",
            ]
        );
    }

    #[test]
    fn a_pre_tripped_flag_cancels_before_any_pass_work() {
        let plan = plan_for(vec![pass(3, "euler", 1.0)], 1);
        let mut host = StubHost::new();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let err = run_with(&plan, &mut host, &cancel, &mut |_| {}).expect_err("cancelled");
        assert!(matches!(err, Error::Canceled));
        assert!(
            !host.log.iter().any(|l| l.starts_with("begin")),
            "no pass may begin after a pre-tripped cancel: {:?}",
            host.log
        );
    }

    #[test]
    fn a_predict_fault_still_reverts_the_pass_and_is_primary() {
        let plan = plan_for(vec![pass(3, "euler", 1.0)], 1);
        let mut host = StubHost::new();
        host.fail_predict_at_eval = Some(2);
        let err = run(&plan, &mut host).expect_err("stub fault surfaces");
        assert!(err.to_string().contains("stub predict fault"));
        assert_eq!(host.log, vec!["schedule[0]", "begin[0]", "end[0]"]);
    }

    // --- Execution record ------------------------------------------------------------------------

    #[test]
    fn execution_record_carries_requested_and_resolved_values() {
        let requested = vec![pass(4, "euler", 1.0), pass(2, "dpmpp_2m", 0.4)];
        let plan = plan_for(requested.clone(), 21);
        let mut host = StubHost::new();
        let out = run(&plan, &mut host).unwrap();
        let execution = out.execution.with_requested(Some(&requested));
        assert_eq!(execution.job_seed, 21);
        assert_eq!(execution.passes.len(), 2);
        assert_eq!(execution.passes[0].requested, Some(requested[0].clone()));
        assert_eq!(execution.passes[1].requested, Some(requested[1].clone()));
        assert_eq!(execution.passes[1].resolved.sampler, "dpmpp_2m");
        assert_eq!(execution.passes[1].schedule_steps, 5); // int(2 / 0.4)
        assert_eq!(execution.passes[1].effective_steps, 2);
        assert_eq!(
            execution.total_model_evals,
            execution
                .passes
                .iter()
                .map(|p| p.model_evals)
                .sum::<usize>()
        );

        // JSON: seeds are strings, requested rides beside resolved.
        let json = execution.to_json();
        assert!(json["jobSeed"].is_string());
        assert!(json["passes"][0]["resolved"]["seed"].is_string());
        assert_eq!(json["passes"][1]["requested"]["sampler"], "dpmpp_2m");
        assert_eq!(json["passes"][1]["scheduleSteps"], 5);
        assert_eq!(json["passes"][1]["renoised"], true);
    }
}
