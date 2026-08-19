//! Backend-neutral **chained denoise-pass executor** conformance (epic 20414, sc-20418).
//!
//! The executor itself is shared host+`LatentOps` code in
//! [`gen_core::sampling::pass_executor`] — what differs per family is only the **schedule
//! builder** the family's [`DenoisePassHost`] supplies. This module therefore drives the real
//! executor over [`CpuLatentOps`] with a deterministic stub model and the *family's own* schedule
//! math, and checks the whole chain contract: one-pass equivalence, boundary re-noise, per-pass
//! schedule/solver-state reset, seed isolation, effective-model-evaluation progress accounting,
//! and cancellation/override-revert hygiene.
//!
//! A family adopts by handing its per-pass schedule builder to [`denoise_pass_conformance`]:
//!
//! ```ignore
//! // candle-gen-krea, Raw:
//! gen_core_testkit::denoise_passes::denoise_pass_conformance("candle krea_2_raw", &|pass, steps| {
//!     crate::pipeline::base_schedule(steps, 1024, 1024, Some(&pass.scheduler))
//! });
//! ```
//!
//! sc-20425 runs this same suite across the other provider families — the executor contract does
//! not change per family, which is the point of checking it here once.

use gen_core::sampling::{
    denoise as gc_denoise, denoise_pass_renoise_seed, denoise_pass_schedule_steps,
    execute_denoise_plan, sampler_by_name, terminal_pass_segment, CpuLatentOps, DenoisePassHost,
    ExecutedDenoisePlan, FlowModelSampling, LatentOps, ModelSampling, PassObservation, Solver,
    TimestepConvention,
};
use gen_core::{
    denoise_pass_seed, resolve_denoise_plan, CancelFlag, DenoiseDefaults, DenoisePass,
    DenoisePassContext, GenerationRequest, Progress, ResolvedDenoisePass, ResolvedDenoisePlan,
};

/// A family's per-pass schedule seam: `(resolved pass, schedule_steps) -> full σ schedule`.
///
/// `schedule_steps` is already Comfy-expanded by the executor
/// ([`denoise_pass_schedule_steps`]); the builder returns the family's **full fresh** schedule for
/// that step count under the pass's resolved scheduler id (descending, trailing `0.0`).
pub type PassScheduleBuilder<'a> = dyn Fn(&ResolvedDenoisePass, usize) -> Vec<f32> + 'a;

fn ms() -> FlowModelSampling {
    FlowModelSampling::new(TimestepConvention::Sigma)
}

fn x_init() -> Vec<f32> {
    vec![0.31, -1.07, 1.93, 0.06, -0.44, 0.72]
}

fn stub_velocity(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| 0.25 * v + 0.1).collect()
}

fn plan_for(passes: Vec<DenoisePass>, job_seed: u64) -> Result<ResolvedDenoisePlan, String> {
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
    .map_err(|e| format!("denoise-pass conformance: plan failed to resolve: {e}"))
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

/// The harness host: the family's schedule builder + the deterministic linear stub model, with the
/// full lifecycle logged so the checks can assert pairing/ordering.
struct HarnessHost<'a> {
    builder: &'a PassScheduleBuilder<'a>,
    log: Vec<String>,
    first_latent_of_pass: Vec<Vec<f32>>,
    seen_pass: Option<usize>,
    observations: Vec<(usize, usize, usize)>,
}

impl<'a> HarnessHost<'a> {
    fn new(builder: &'a PassScheduleBuilder<'a>) -> Self {
        Self {
            builder,
            log: Vec::new(),
            first_latent_of_pass: Vec::new(),
            seen_pass: None,
            observations: Vec::new(),
        }
    }
}

impl DenoisePassHost<CpuLatentOps> for HarnessHost<'_> {
    fn build_schedule(
        &mut self,
        pass: &ResolvedDenoisePass,
        schedule_steps: usize,
    ) -> gen_core::Result<Vec<f32>> {
        Ok((self.builder)(pass, schedule_steps))
    }
    fn begin_pass(&mut self, pass: &ResolvedDenoisePass) -> gen_core::Result<()> {
        self.log.push(format!("begin[{}]", pass.index));
        Ok(())
    }
    fn predict(
        &mut self,
        _pass: &ResolvedDenoisePass,
        x: &Vec<f32>,
        _timestep: f32,
    ) -> gen_core::Result<Vec<f32>> {
        Ok(stub_velocity(x))
    }
    fn end_pass(&mut self, pass: &ResolvedDenoisePass) -> gen_core::Result<()> {
        self.log.push(format!("end[{}]", pass.index));
        Ok(())
    }
    fn observe(&mut self, obs: PassObservation<'_, Vec<f32>>) {
        self.observations
            .push((obs.pass_index, obs.chain_step, obs.chain_total_steps));
        if self.seen_pass != Some(obs.pass_index) {
            self.seen_pass = Some(obs.pass_index);
            self.first_latent_of_pass.push(obs.latent.clone());
        }
    }
}

fn run_chain(
    builder: &PassScheduleBuilder<'_>,
    plan: &ResolvedDenoisePlan,
    on_progress: &mut dyn FnMut(Progress),
    cancel: &CancelFlag,
) -> Result<(ExecutedDenoisePlan<Vec<f32>>, HarnessHost<'static>), String> {
    // The host's lifetime is tied to `builder`; we transmute nothing — instead run inline and
    // return the pieces the checks read.
    let mut host = HarnessHost::new(builder);
    let out = execute_denoise_plan(
        &CpuLatentOps,
        &ms(),
        plan,
        x_init(),
        &mut host,
        cancel,
        on_progress,
    )
    .map_err(|e| format!("denoise-pass conformance: chain failed: {e}"))?;
    Ok((
        out,
        HarnessHost {
            builder: &NULL_BUILDER,
            log: host.log,
            first_latent_of_pass: host.first_latent_of_pass,
            seen_pass: host.seen_pass,
            observations: host.observations,
        },
    ))
}

/// An inert builder used only to give the returned harness state a `'static` carrier.
static NULL_BUILDER: fn(&ResolvedDenoisePass, usize) -> Vec<f32> = |_, _| Vec::new();

/// **Schedule shape.** The family builder, driven exactly as the executor drives it, returns a
/// usable schedule for every curated scheduler id at representative step counts: at least 2 nodes,
/// finite, non-increasing, trailing `0.0`.
pub fn check_pass_schedule_shape(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    for scheduler in gen_core::curated_scheduler_ids() {
        for steps in [1_u32, 4, 8] {
            let resolved = ResolvedDenoisePass {
                index: 0,
                steps,
                sampler: "euler".to_owned(),
                scheduler: scheduler.to_owned(),
                denoise: 1.0,
                guidance: None,
                seed: 7,
                adapters: Vec::new(),
            };
            let sigmas = builder(&resolved, denoise_pass_schedule_steps(steps, 1.0));
            if sigmas.len() < 2 {
                return Err(format!(
                    "pass-schedule shape[{scheduler} @ {steps}]: got {} node(s)",
                    sigmas.len()
                ));
            }
            if sigmas.iter().any(|s| !s.is_finite()) {
                return Err(format!(
                    "pass-schedule shape[{scheduler} @ {steps}]: non-finite σ in {sigmas:?}"
                ));
            }
            if sigmas.windows(2).any(|w| w[0] < w[1]) {
                return Err(format!(
                    "pass-schedule shape[{scheduler} @ {steps}]: not descending: {sigmas:?}"
                ));
            }
            if sigmas.last() != Some(&0.0) {
                return Err(format!(
                    "pass-schedule shape[{scheduler} @ {steps}]: no trailing 0.0: {sigmas:?}"
                ));
            }
        }
    }
    Ok(())
}

/// **AC1 — one-pass equivalence.** A one-pass plan through the executor is bit-identical to the
/// same solver driven directly over the family's own schedule with the same seed (the legacy
/// single-trajectory shape).
pub fn check_one_pass_equivalence(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    for sampler in [
        "euler",
        "dpmpp_2m",
        "rk6_7s",
        "abnorsett_4m",
        "euler_ancestral",
    ] {
        let job_seed = 42_u64;
        let plan = plan_for(vec![pass(6, sampler, 1.0)], job_seed)?;
        let (out, _) = run_chain(builder, &plan, &mut |_| {}, &CancelFlag::new())?;

        let ops = CpuLatentOps;
        let m = ms();
        let sigmas = builder(&plan.passes[0], 6);
        let solver = sampler_by_name::<CpuLatentOps>(sampler)
            .ok_or_else(|| format!("one-pass equivalence: unknown solver {sampler:?}"))?;
        let mut dn =
            |x: &Vec<f32>, s: f32| gc_denoise(&ops, &m, x, s, |xin, _t| Ok(stub_velocity(xin)));
        let want = solver
            .sample(&ops, &m, &mut dn, x_init(), &sigmas, job_seed)
            .map_err(|e| format!("one-pass equivalence: direct run failed: {e}"))?;
        if out.latent != want {
            return Err(format!(
                "one-pass equivalence[{sampler}]: executor output {:?} != direct solver run {:?}",
                out.latent, want
            ));
        }
    }
    Ok(())
}

/// **AC2 — boundary re-noise.** Pass 2 first observes exactly the deterministic re-noise of pass
/// 1's output at its segment-entry σ (the model's `noise_scaling_coeffs` blend with the pass-2
/// seed's dedicated renoise stream), which genuinely perturbs the latent before denoising resumes.
pub fn check_boundary_renoise(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    let job_seed = 7_u64;
    let plan = plan_for(vec![pass(4, "euler", 1.0), pass(3, "euler", 0.5)], job_seed)?;
    let (out, host) = run_chain(builder, &plan, &mut |_| {}, &CancelFlag::new())?;

    let one = plan_for(vec![pass(4, "euler", 1.0)], job_seed)?;
    let (only_pass_1, _) = run_chain(builder, &one, &mut |_| {}, &CancelFlag::new())?;
    let after_pass_1 = only_pass_1.latent;

    let ops = CpuLatentOps;
    let m = ms();
    let full = builder(&plan.passes[1], denoise_pass_schedule_steps(3, 0.5));
    let seg = terminal_pass_segment(&full, 3);
    let (k_noise, k_x0) = m.noise_scaling_coeffs(seg[0]);
    let pass2_seed = denoise_pass_seed(job_seed, 1);
    let noise = ops
        .randn_like(&after_pass_1, denoise_pass_renoise_seed(pass2_seed), 0)
        .map_err(|e| format!("boundary re-noise: randn failed: {e}"))?;
    let expected_entry = ops
        .axpy(k_x0, &after_pass_1, k_noise, &noise)
        .map_err(|e| format!("boundary re-noise: blend failed: {e}"))?;

    if host.first_latent_of_pass.len() != 2 {
        return Err(format!(
            "boundary re-noise: expected 2 observed passes, saw {}",
            host.first_latent_of_pass.len()
        ));
    }
    if host.first_latent_of_pass[1] != expected_entry {
        return Err(
            "boundary re-noise: pass 2's entry latent is not the deterministic \
                    noise_scaling blend of pass 1's output with the pass-2 renoise stream"
                .to_owned(),
        );
    }
    if host.first_latent_of_pass[1] == after_pass_1 {
        return Err(
            "boundary re-noise: pass 2 consumed pass 1's latent UNperturbed — no re-noise happened"
                .to_owned(),
        );
    }
    if out.latent == expected_entry {
        return Err("boundary re-noise: pass 2 never denoised its re-noised entry".to_owned());
    }
    if !out.execution.passes[1].renoised || out.execution.passes[0].renoised {
        return Err("boundary re-noise: execution record mis-reports the renoise flags".to_owned());
    }
    Ok(())
}

/// **AC2 — schedule + solver-state reset.** A 2-pass multistep (`abnorsett_4m`) chain equals
/// hand-running the solver with a FRESH state per pass segment over the family's own schedules
/// (re-noise applied at the boundary) — the executor's fresh-schedule / fresh-history invariant.
pub fn check_state_reset(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    use gen_core::sampling::{Abnorsett4m, AbnorsettState};
    let job_seed = 5_u64;
    let plan = plan_for(
        vec![pass(5, "abnorsett_4m", 1.0), pass(4, "abnorsett_4m", 1.0)],
        job_seed,
    )?;
    let (out, _) = run_chain(builder, &plan, &mut |_| {}, &CancelFlag::new())?;

    let ops = CpuLatentOps;
    let m = ms();
    let mut dn =
        |x: &Vec<f32>, s: f32| gc_denoise(&ops, &m, x, s, |xin, _t| Ok(stub_velocity(xin)));
    let seg_a_full = builder(&plan.passes[0], 5);
    let seg_a = terminal_pass_segment(&seg_a_full, 5).to_vec();
    let seg_b_full = builder(&plan.passes[1], 4);
    let seg_b = terminal_pass_segment(&seg_b_full, 4).to_vec();

    let mut fresh_a = AbnorsettState::default();
    let x_a = Abnorsett4m
        .sample_with_state(&ops, &mut dn, x_init(), &seg_a, &mut fresh_a)
        .map_err(|e| format!("state reset: hand-run pass 1 failed: {e}"))?;
    let renoise_seed = denoise_pass_renoise_seed(denoise_pass_seed(job_seed, 1));
    let noise = ops
        .randn_like(&x_a, renoise_seed, 0)
        .map_err(|e| format!("state reset: randn failed: {e}"))?;
    let (kn, kx) = m.noise_scaling_coeffs(seg_b[0]);
    let entry_b = ops
        .axpy(kx, &x_a, kn, &noise)
        .map_err(|e| format!("state reset: blend failed: {e}"))?;
    let mut fresh_b = AbnorsettState::default();
    let want = Abnorsett4m
        .sample_with_state(&ops, &mut dn, entry_b, &seg_b, &mut fresh_b)
        .map_err(|e| format!("state reset: hand-run pass 2 failed: {e}"))?;
    if out.latent != want {
        return Err(
            "state reset: chain output differs from fresh-state-per-segment — schedule or \
             multistep history leaked across the pass boundary"
                .to_owned(),
        );
    }
    Ok(())
}

/// **AC2 — seed isolation.** Same request ⇒ identical output; different job seed ⇒ different
/// output; and different pass indices draw from different renoise streams.
pub fn check_seed_isolation(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    let passes = || vec![pass(3, "euler", 1.0), pass(3, "euler", 0.5)];
    let plan = plan_for(passes(), 99)?;
    let (a, _) = run_chain(builder, &plan, &mut |_| {}, &CancelFlag::new())?;
    let (b, _) = run_chain(builder, &plan, &mut |_| {}, &CancelFlag::new())?;
    if a.latent != b.latent {
        return Err("seed isolation: the same request did not reproduce".to_owned());
    }
    let other = plan_for(passes(), 100)?;
    let (c, _) = run_chain(builder, &other, &mut |_| {}, &CancelFlag::new())?;
    if a.latent == c.latent {
        return Err("seed isolation: a different job seed reproduced the same output".to_owned());
    }
    if denoise_pass_seed(99, 1) == denoise_pass_seed(99, 2) {
        return Err("seed isolation: adjacent pass seeds collide".to_owned());
    }
    if denoise_pass_renoise_seed(denoise_pass_seed(99, 1))
        == denoise_pass_renoise_seed(denoise_pass_seed(99, 2))
    {
        return Err("seed isolation: adjacent renoise streams collide".to_owned());
    }
    Ok(())
}

/// **Progress by effective model evaluations.** One `Progress::Step` per model evaluation,
/// `current` sweeping exactly `1..=total` across the whole chain with
/// `total = Σ Solver::model_evals(pass steps)`, and the per-pass detail retained in the execution
/// record.
pub fn check_eval_accounting(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    let plan = plan_for(vec![pass(3, "rk6_7s", 1.0), pass(4, "euler", 1.0)], 11)?;
    let mut events: Vec<(u32, u32)> = Vec::new();
    let (out, host) = run_chain(
        builder,
        &plan,
        &mut |p| {
            if let Progress::Step { current, total } = p {
                events.push((current, total));
            }
        },
        &CancelFlag::new(),
    )?;
    let want_total = Solver::Rk67s.model_evals(3) + Solver::Euler.model_evals(4);
    if out.execution.total_model_evals != want_total {
        return Err(format!(
            "eval accounting: recorded total {} != Solver::model_evals sum {}",
            out.execution.total_model_evals, want_total
        ));
    }
    if events.len() != want_total {
        return Err(format!(
            "eval accounting: {} progress events != {} model evaluations",
            events.len(),
            want_total
        ));
    }
    let currents: Vec<u32> = events.iter().map(|&(c, _)| c).collect();
    if currents != (1..=want_total as u32).collect::<Vec<_>>() {
        return Err(format!(
            "eval accounting: current must sweep 1..={} once per evaluation, got {currents:?}",
            want_total
        ));
    }
    if events.iter().any(|&(_, t)| t as usize != want_total) {
        return Err("eval accounting: the total changed mid-chain".to_owned());
    }
    if out.execution.passes[0].model_evals != Solver::Rk67s.model_evals(3)
        || out.execution.passes[1].model_evals != Solver::Euler.model_evals(4)
    {
        return Err("eval accounting: per-pass model_evals mis-recorded".to_owned());
    }
    // Chain-global outer steps cover both passes.
    let max_step = host.observations.iter().map(|&(_, s, _)| s).max();
    if max_step != Some(6) {
        return Err(format!(
            "eval accounting: chain outer steps should reach 6 (3 + 4 − 1), got {max_step:?}"
        ));
    }
    Ok(())
}

/// **Cancellation + override-revert hygiene.** A flag tripped mid-chain surfaces as the typed
/// `Error::Canceled`; `begin_pass`/`end_pass` stay paired on every exit path, so pass-local
/// overrides are always reverted (no poisoned provider state).
pub fn check_cancellation(builder: &PassScheduleBuilder<'_>) -> Result<(), String> {
    let plan = plan_for(vec![pass(3, "euler", 1.0), pass(3, "euler", 1.0)], 1)?;
    let cancel = CancelFlag::new();
    let mut seen = 0_u32;
    let mut host = HarnessHost::new(builder);
    let err = execute_denoise_plan(
        &CpuLatentOps,
        &ms(),
        &plan,
        x_init(),
        &mut host,
        &cancel,
        &mut |p| {
            if matches!(p, Progress::Step { .. }) {
                seen += 1;
                if seen == 4 {
                    cancel.cancel();
                }
            }
        },
    )
    .err()
    .ok_or_else(|| "cancellation: a tripped flag did not stop the chain".to_owned())?;
    if !matches!(err, gen_core::Error::Canceled) {
        return Err(format!(
            "cancellation: expected the typed Error::Canceled, got {err:?}"
        ));
    }
    let begins = host.log.iter().filter(|l| l.starts_with("begin")).count();
    let ends = host.log.iter().filter(|l| l.starts_with("end")).count();
    if begins != ends {
        return Err(format!(
            "cancellation: begin/end unpaired after cancel ({begins} begins, {ends} ends): {:?}",
            host.log
        ));
    }

    // Pre-tripped: no pass may even begin.
    let cancel = CancelFlag::new();
    cancel.cancel();
    let mut host = HarnessHost::new(builder);
    let err = execute_denoise_plan(
        &CpuLatentOps,
        &ms(),
        &plan,
        x_init(),
        &mut host,
        &cancel,
        &mut |_| {},
    )
    .err()
    .ok_or_else(|| "cancellation: a pre-tripped flag did not stop the chain".to_owned())?;
    if !matches!(err, gen_core::Error::Canceled) {
        return Err(format!(
            "cancellation: pre-tripped flag must be the typed Error::Canceled, got {err:?}"
        ));
    }
    if host.log.iter().any(|l| l.starts_with("begin")) {
        return Err(format!(
            "cancellation: a pass began after a pre-tripped cancel: {:?}",
            host.log
        ));
    }
    Ok(())
}

/// Run every denoise-pass executor check against one family's schedule seam, panicking with the
/// aggregated failures. `family` only labels the panic (e.g. `"candle krea_2_raw"`).
pub fn denoise_pass_conformance(family: &str, builder: &PassScheduleBuilder<'_>) {
    let failures: Vec<String> = [
        check_pass_schedule_shape(builder),
        check_one_pass_equivalence(builder),
        check_boundary_renoise(builder),
        check_state_reset(builder),
        check_seed_isolation(builder),
        check_eval_accounting(builder),
        check_cancellation(builder),
    ]
    .into_iter()
    .filter_map(|r| r.err())
    .collect();
    if !failures.is_empty() {
        panic!(
            "denoise-pass executor conformance FAILED for {family} (gen-core {}):\n  - {}",
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::sampling::{schedule_sigmas, Scheduler};

    /// The shared gen-core schedule math itself passes the whole suite — the reference every
    /// family seam is compared against.
    fn gen_core_builder() -> impl Fn(&ResolvedDenoisePass, usize) -> Vec<f32> {
        |pass: &ResolvedDenoisePass, steps: usize| {
            let sched = Scheduler::from_name(&pass.scheduler).unwrap_or(Scheduler::Normal);
            schedule_sigmas(sched, &ms(), steps)
        }
    }

    #[test]
    fn shared_gen_core_math_passes_the_whole_suite() {
        denoise_pass_conformance("gen-core", &gen_core_builder());
    }

    /// Negative self-test: a NON-DETERMINISTIC schedule builder (different σ on a repeated call —
    /// the drift class that would silently break replays) is caught — the equivalence checks
    /// re-derive schedules independently, so the harness can actually fail.
    #[test]
    fn a_nondeterministic_builder_is_reported() {
        use std::cell::Cell;
        let calls = Cell::new(0_u32);
        let broken = move |pass: &ResolvedDenoisePass, steps: usize| {
            let sched = Scheduler::from_name(&pass.scheduler).unwrap_or(Scheduler::Normal);
            let mut s = schedule_sigmas(sched, &ms(), steps);
            calls.set(calls.get() + 1);
            if calls.get() > 1 {
                // Every later call drifts: interior σs shrink by 10%.
                let interior = s.len().saturating_sub(1);
                for v in s.iter_mut().take(interior) {
                    *v *= 0.9;
                }
            }
            s
        };
        let err = check_one_pass_equivalence(&broken).expect_err("a drifting builder must fail");
        assert!(err.contains("one-pass equivalence"), "{err}");
    }

    /// Negative self-test: a builder emitting a schedule with no trailing zero fails the shape
    /// check.
    #[test]
    fn a_builder_without_terminal_zero_is_reported() {
        let broken = |pass: &ResolvedDenoisePass, steps: usize| {
            let sched = Scheduler::from_name(&pass.scheduler).unwrap_or(Scheduler::Normal);
            let mut s = schedule_sigmas(sched, &ms(), steps);
            if let Some(last) = s.last_mut() {
                *last = 0.01;
            }
            s
        };
        let err = check_pass_schedule_shape(&broken).expect_err("no trailing zero must fail");
        assert!(err.contains("trailing 0.0"), "{err}");
    }
}
