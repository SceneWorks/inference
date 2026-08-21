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
//! A flow-cohort family adopts by handing its per-pass schedule builder to
//! [`denoise_pass_conformance`]:
//!
//! ```ignore
//! // candle-gen-krea, Raw:
//! gen_core_testkit::denoise_passes::denoise_pass_conformance("candle krea_2_raw", &|pass, steps| {
//!     crate::pipeline::base_schedule(steps, 1024, 1024, Some(&pass.scheduler))
//! });
//! ```
//!
//! # The model-sampling contract is a parameter, not a constant (sc-20425)
//!
//! The suite was written against [`FlowModelSampling`] alone, which quietly made it unable to
//! certify half its intended audience. **The boundary re-noise is the only part of the executor that
//! reads the model's prediction contract**, and the two cohorts disagree there:
//!
//! | cohort | `noise_scaling_coeffs(σ)` | boundary blend |
//! | --- | --- | --- |
//! | FLOW (`FlowModelSampling`) | `(σ, 1−σ)` | `x = σ·ε + (1−σ)·x₀` — convex interpolation |
//! | VE / discrete ε (`DiscreteModelSampling`, `EdmModelSampling`) | `(σ, 1)` | `x = x₀ + σ·ε` — x₀ at full scale |
//!
//! Feeding one the other's form is out-of-distribution, and at SDXL's σ range (up to ≈ 14.6) it is
//! not a subtle difference — it is the difference between a refinement pass and noise. A green
//! flow-only run therefore said nothing at all about SDXL, the epic's required latent-diffusion
//! acceptance family.
//!
//! [`denoise_pass_conformance_over`] takes the family's own [`ModelSampling`] and drives every check
//! through it, including a check ([`check_renoise_matches_model_type`]) that the blend which ran is
//! *this* cohort's and provably not the other's. [`denoise_pass_conformance`] is the flow-cohort
//! shorthand, unchanged for the families already adopting it.

use std::sync::{Arc, Mutex};

use gen_core::sampling::{
    denoise as gc_denoise, denoise_pass_renoise_seed, denoise_pass_schedule_steps,
    execute_denoise_plan, sampler_by_name, terminal_pass_segment, AlphaSchedule, CpuLatentOps,
    DenoisePassHost, DiscreteModelSampling, ExecutedDenoisePlan, FlowModelSampling, LatentOps,
    ModelSampling, PassObservation, Solver, TimestepConvention,
};
use gen_core::{
    denoise_pass_seed, resolve_denoise_plan, CancelFlag, DenoiseDefaults, DenoisePass,
    DenoisePassContext, DenoisePassReportSink, DenoisePlanExecution, GenerationRequest, Progress,
    ResolvedDenoisePass, ResolvedDenoisePlan,
};

/// A family's per-pass schedule seam: `(resolved pass, schedule_steps) -> full σ schedule`.
///
/// `schedule_steps` is already Comfy-expanded by the executor
/// ([`denoise_pass_schedule_steps`]); the builder returns the family's **full fresh** schedule for
/// that step count under the pass's resolved scheduler id (descending, trailing `0.0`).
pub type PassScheduleBuilder<'a> = dyn Fn(&ResolvedDenoisePass, usize) -> Vec<f32> + 'a;

/// The FLOW cohort's contract — the shorthand [`denoise_pass_conformance`] drives.
fn flow_ms() -> FlowModelSampling {
    FlowModelSampling::new(TimestepConvention::Sigma)
}

/// The SDXL/Kolors VE ε-prediction contract, ready to hand to
/// [`denoise_pass_conformance_over`].
///
/// Exposed because the alpha schedule is a mouthful (`scaled_linear(1000, 0.00085, 0.012)`) and an
/// adopter that re-derives it slightly differently would be certifying a *different* σ range than
/// the one it renders on. Families that build their own `DiscreteModelSampling` from their own
/// schedule pass that instead.
#[must_use]
pub fn sdxl_discrete_model_sampling() -> DiscreteModelSampling {
    DiscreteModelSampling::sdxl(&AlphaSchedule::scaled_linear(1000, 0.00085, 0.012))
}

/// The chain's initial latent, at a magnitude appropriate to the cohort.
///
/// A VE family's initial latent is the unit prior scaled by σ_max (≈ 14.6 on SDXL), and running the
/// harness on a unit-magnitude latent there would exercise the arithmetic at a scale the family
/// never sees. `FlowModelSampling::sigma_max()` is exactly `1.0`, so the flow cohort is bit-for-bit
/// unchanged by this.
fn x_init_for(ms: &dyn ModelSampling) -> Vec<f32> {
    let scale = ms.sigma_max().max(1.0);
    [0.31, -1.07, 1.93, 0.06, -0.44, 0.72]
        .iter()
        .map(|v| v * scale)
        .collect()
}

/// The deterministic stub model output — velocity for FLOW, ε for the discrete cohort.
///
/// One function serves both because [`gc_denoise`] converts whatever the model returns into `x₀`
/// through the caller's own [`ModelSampling::denoised_coeffs`]; the stub only has to be a fixed,
/// non-trivial function of the latent.
fn stub_model_output(x: &[f32]) -> Vec<f32> {
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
        Ok(stub_model_output(x))
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
    ms: &dyn ModelSampling,
    plan: &ResolvedDenoisePlan,
    on_progress: &mut dyn FnMut(Progress),
    cancel: &CancelFlag,
) -> Result<(ExecutedDenoisePlan<Vec<f32>>, HarnessHost<'static>), String> {
    // The host's lifetime is tied to `builder`; we transmute nothing — instead run inline and
    // return the pieces the checks read.
    let mut host = HarnessHost::new(builder);
    let out = execute_denoise_plan(
        &CpuLatentOps,
        ms,
        plan,
        x_init_for(ms),
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
pub fn check_one_pass_equivalence(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    for sampler in [
        "euler",
        "dpmpp_2m",
        "rk6_7s",
        "abnorsett_4m",
        "euler_ancestral",
    ] {
        let job_seed = 42_u64;
        let plan = plan_for(vec![pass(6, sampler, 1.0)], job_seed)?;
        let (out, _) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;

        let ops = CpuLatentOps;
        let sigmas = builder(&plan.passes[0], 6);
        let solver = sampler_by_name::<CpuLatentOps>(sampler)
            .ok_or_else(|| format!("one-pass equivalence: unknown solver {sampler:?}"))?;
        let mut dn =
            |x: &Vec<f32>, s: f32| gc_denoise(&ops, ms, x, s, |xin, _t| Ok(stub_model_output(xin)));
        let want = solver
            .sample(&ops, ms, &mut dn, x_init_for(ms), &sigmas, job_seed)
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
pub fn check_boundary_renoise(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    let job_seed = 7_u64;
    let plan = plan_for(vec![pass(4, "euler", 1.0), pass(3, "euler", 0.5)], job_seed)?;
    let (out, host) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;

    let one = plan_for(vec![pass(4, "euler", 1.0)], job_seed)?;
    let (only_pass_1, _) = run_chain(builder, ms, &one, &mut |_| {}, &CancelFlag::new())?;
    let after_pass_1 = only_pass_1.latent;

    let ops = CpuLatentOps;
    let full = builder(&plan.passes[1], denoise_pass_schedule_steps(3, 0.5));
    let seg = terminal_pass_segment(&full, 3);
    let (k_noise, k_x0) = ms.noise_scaling_coeffs(seg[0]);
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
pub fn check_state_reset(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    use gen_core::sampling::{Abnorsett4m, AbnorsettState};
    let job_seed = 5_u64;
    let plan = plan_for(
        vec![pass(5, "abnorsett_4m", 1.0), pass(4, "abnorsett_4m", 1.0)],
        job_seed,
    )?;
    let (out, _) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;

    let ops = CpuLatentOps;
    let mut dn =
        |x: &Vec<f32>, s: f32| gc_denoise(&ops, ms, x, s, |xin, _t| Ok(stub_model_output(xin)));
    let seg_a_full = builder(&plan.passes[0], 5);
    let seg_a = terminal_pass_segment(&seg_a_full, 5).to_vec();
    let seg_b_full = builder(&plan.passes[1], 4);
    let seg_b = terminal_pass_segment(&seg_b_full, 4).to_vec();

    let mut fresh_a = AbnorsettState::default();
    let x_a = Abnorsett4m
        .sample_with_state(&ops, &mut dn, x_init_for(ms), &seg_a, &mut fresh_a)
        .map_err(|e| format!("state reset: hand-run pass 1 failed: {e}"))?;
    let renoise_seed = denoise_pass_renoise_seed(denoise_pass_seed(job_seed, 1));
    let noise = ops
        .randn_like(&x_a, renoise_seed, 0)
        .map_err(|e| format!("state reset: randn failed: {e}"))?;
    let (kn, kx) = ms.noise_scaling_coeffs(seg_b[0]);
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
pub fn check_seed_isolation(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    let passes = || vec![pass(3, "euler", 1.0), pass(3, "euler", 0.5)];
    let plan = plan_for(passes(), 99)?;
    let (a, _) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;
    let (b, _) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;
    if a.latent != b.latent {
        return Err("seed isolation: the same request did not reproduce".to_owned());
    }
    let other = plan_for(passes(), 100)?;
    let (c, _) = run_chain(builder, ms, &other, &mut |_| {}, &CancelFlag::new())?;
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
pub fn check_eval_accounting(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    let plan = plan_for(vec![pass(3, "rk6_7s", 1.0), pass(4, "euler", 1.0)], 11)?;
    let mut events: Vec<(u32, u32)> = Vec::new();
    let (out, host) = run_chain(
        builder,
        ms,
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
pub fn check_cancellation(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    let plan = plan_for(vec![pass(3, "euler", 1.0), pass(3, "euler", 1.0)], 1)?;
    let cancel = CancelFlag::new();
    let mut seen = 0_u32;
    let mut host = HarnessHost::new(builder);
    let err = execute_denoise_plan(
        &CpuLatentOps,
        ms,
        &plan,
        x_init_for(ms),
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
        ms,
        &plan,
        x_init_for(ms),
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

/// **The boundary blend is this cohort's, and provably not the other's** (sc-20425).
///
/// [`check_boundary_renoise`] derives its expected entry latent from the same
/// [`ModelSampling::noise_scaling_coeffs`] the executor uses, so it would pass just as happily if
/// *both* sides were wrong about the model type — which is precisely how a flow-only harness came to
/// look like it certified SDXL. This check closes that by computing the **other** cohort's blend as
/// well and requiring the two to differ, so a green result is evidence about this family rather than
/// evidence that one formula was used twice.
///
/// FLOW is `σ·ε + (1−σ)·x₀`; VE / discrete ε / EDM is `x₀ + σ·ε`. They agree only where `1−σ = 1`,
/// i.e. σ = 0, which is never a segment entry, so the discrimination is always available.
pub fn check_renoise_matches_model_type(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
) -> Result<(), String> {
    let job_seed = 2024_u64;
    let plan = plan_for(vec![pass(4, "euler", 1.0), pass(3, "euler", 0.6)], job_seed)?;
    let (_, host) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;
    let one = plan_for(vec![pass(4, "euler", 1.0)], job_seed)?;
    let (only_pass_1, _) = run_chain(builder, ms, &one, &mut |_| {}, &CancelFlag::new())?;
    let after_pass_1 = only_pass_1.latent;

    let full = builder(&plan.passes[1], denoise_pass_schedule_steps(3, 0.6));
    let seg = terminal_pass_segment(&full, 3);
    let sigma_entry = seg[0];
    if !(sigma_entry.is_finite() && sigma_entry > 0.0) {
        return Err(format!(
            "renoise convention: pass 2 entered at σ = {sigma_entry}, which cannot discriminate the \
             two blends"
        ));
    }

    let ops = CpuLatentOps;
    let noise = ops
        .randn_like(
            &after_pass_1,
            denoise_pass_renoise_seed(denoise_pass_seed(job_seed, 1)),
            0,
        )
        .map_err(|e| format!("renoise convention: randn failed: {e}"))?;
    let blend = |k_x0: f32, k_noise: f32| {
        ops.axpy(k_x0, &after_pass_1, k_noise, &noise)
            .map_err(|e| format!("renoise convention: blend failed: {e}"))
    };

    let (k_noise, k_x0) = ms.noise_scaling_coeffs(sigma_entry);
    let mine = blend(k_x0, k_noise)?;
    // The other cohort's form at the same σ. Flow is the convex interpolation; everything else is
    // the VE default the trait itself documents.
    let (other_noise, other_x0) =
        if matches!(ms.prediction(), gen_core::sampling::PredictionType::Flow) {
            (sigma_entry, 1.0)
        } else {
            (sigma_entry, 1.0 - sigma_entry)
        };
    let theirs = blend(other_x0, other_noise)?;

    // At any σ > 0 the two forms are distinct — `(σ, 1)` vs `(σ, 1−σ)` — so if this model's own
    // coefficients equal the other cohort's, its DECLARATION and its blend disagree. That is a real
    // defect (a family copying a sibling's `noise_scaling_coeffs` alongside its own prediction type),
    // not a harness limitation, and it is reported as such rather than as "cannot discriminate".
    if mine == theirs {
        return Err(format!(
            "renoise convention: this model declares {:?} but its noise_scaling_coeffs at σ = \
             {sigma_entry} are the OTHER cohort's — the declaration and the blend disagree, so \
             every certification over it would be vacuous",
            ms.prediction()
        ));
    }
    let observed = host.first_latent_of_pass.get(1).ok_or_else(|| {
        "renoise convention: pass 2 was never observed, so no boundary latent exists".to_owned()
    })?;
    classify_boundary_latent(
        observed,
        &mine,
        &theirs,
        ms.prediction(),
        (k_x0, k_noise),
        sigma_entry,
    )
}

/// Which of the three things the observed boundary latent is: this cohort's blend (`Ok`), the other
/// cohort's, or neither.
///
/// Split out of [`check_renoise_matches_model_type`] so the two failing arms are **reachable from a
/// test**. They are EXECUTOR regression guards and deliberately unreachable from any
/// [`ModelSampling`] a family can write — the observation is the executor's own blend through the
/// same `ms`, so with a correct executor it always equals `mine`. They fire only if the executor
/// stops reading `noise_scaling_coeffs` and hardcodes a form, which is exactly the flow-only shape
/// this suite lived in before sc-20425 and which no test could then see. Keeping them separate keeps
/// the diagnosis specific: "it used the other cohort's formula" is a different bug from "it used
/// neither" (sc-20425 review MINOR 3).
fn classify_boundary_latent(
    observed: &[f32],
    mine: &[f32],
    theirs: &[f32],
    prediction: gen_core::sampling::PredictionType,
    (k_x0, k_noise): (f32, f32),
    sigma_entry: f32,
) -> Result<(), String> {
    if observed == theirs {
        return Err(format!(
            "renoise convention: the boundary used the OTHER cohort's blend (prediction type is \
             {prediction:?}, so it must be x = {k_x0}·x0 + {k_noise}·ε at σ = {sigma_entry})"
        ));
    }
    if observed != mine {
        return Err(format!(
            "renoise convention: the boundary latent is neither cohort's blend at σ = \
             {sigma_entry} — the executor is not reading {prediction:?}'s noise_scaling_coeffs"
        ));
    }
    Ok(())
}

/// Run every denoise-pass executor check against one family's schedule seam **and its own
/// [`ModelSampling`]**, panicking with the aggregated failures. `family` only labels the panic
/// (e.g. `"candle sdxl"`).
///
/// This is the entry point for any family outside the flow cohort — see the [module docs](self) for
/// why the model-sampling contract cannot be assumed.
pub fn denoise_pass_conformance_over(
    family: &str,
    ms: &dyn ModelSampling,
    builder: &PassScheduleBuilder<'_>,
) {
    let failures: Vec<String> = [
        check_pass_schedule_shape(builder),
        check_one_pass_equivalence(builder, ms),
        check_boundary_renoise(builder, ms),
        check_renoise_matches_model_type(builder, ms),
        check_state_reset(builder, ms),
        check_seed_isolation(builder, ms),
        check_eval_accounting(builder, ms),
        check_cancellation(builder, ms),
    ]
    .into_iter()
    .filter_map(|r| r.err())
    .collect();
    if !failures.is_empty() {
        panic!(
            "denoise-pass executor conformance FAILED for {family} over {:?} (gen-core {}):\n  - {}",
            ms.prediction(),
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

/// **The execution-record contract, checked against a family's own resolution ladder** (sc-20425).
///
/// A provider that runs a chain owes the caller exactly one
/// [`DenoisePlanExecution`] through [`GenerationRequest::emit_denoise_pass_report`] — the
/// requested AND resolved per-pass values plus the effective evaluation accounting — before any
/// image is returned. That record is the only thing a replay has to replay *from*, and four
/// separate adopters shipped binding it to `_execution` and dropping it, which no test anywhere
/// could see (the sc-20425 review's MAJOR 1).
///
/// This drives the REAL executor over the family's own schedule seam, `defaults` and `ctx`, so the
/// resolution ladder under test is the shipped one, then checks the record an adopter must
/// publish: one record, the request's passes stamped verbatim as `requested`, the resolved values
/// the ladder produced, the per-pass re-noise flags, and an evaluation total that is the sum of
/// the parts.
///
/// `req` must carry `denoise_passes`; an adopter test passes the same request shape its generator
/// accepts. Returns the record so a caller can assert family-specific details on top.
pub fn check_execution_record(
    builder: &PassScheduleBuilder<'_>,
    ms: &dyn ModelSampling,
    req: &GenerationRequest,
    job_seed: u64,
    defaults: &DenoiseDefaults,
    ctx: &DenoisePassContext<'_>,
) -> Result<DenoisePlanExecution, String> {
    let requested = req
        .denoise_passes
        .as_ref()
        .ok_or_else(|| "execution record: the request must carry `denoisePasses`".to_owned())?
        .clone();
    let plan = req
        .resolve_denoise_plan(job_seed, defaults, ctx)
        .map_err(|err| format!("execution record: the plan failed to resolve: {err}"))?;
    let (out, _) = run_chain(builder, ms, &plan, &mut |_| {}, &CancelFlag::new())?;

    // Emit through the shipped seam, over a request that is this one plus a capturing sink.
    let seen: Arc<Mutex<Vec<DenoisePlanExecution>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&seen);
    let reporting = GenerationRequest {
        denoise_passes: Some(requested.clone()),
        denoise_pass_report: DenoisePassReportSink::new(move |record| {
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(record)
        }),
        ..Default::default()
    };
    reporting.emit_denoise_pass_report(out.execution);

    let seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if seen.len() != 1 {
        return Err(format!(
            "execution record: expected exactly one record per generation, got {}",
            seen.len()
        ));
    }
    let record = seen[0].clone();
    if record.job_seed != job_seed {
        return Err(format!(
            "execution record: job seed {} != the request's {job_seed}",
            record.job_seed
        ));
    }
    if record.passes.len() != plan.passes.len() {
        return Err(format!(
            "execution record: {} pass records for a {}-pass plan",
            record.passes.len(),
            plan.passes.len()
        ));
    }
    for (index, entry) in record.passes.iter().enumerate() {
        if entry.requested.as_ref() != requested.get(index) {
            return Err(format!(
                "execution record: pass {index} requested {:?} != the request's {:?} — a \
                 consumer cannot tell an inherited default from an explicit choice",
                entry.requested,
                requested.get(index)
            ));
        }
        if entry.resolved != plan.passes[index] {
            return Err(format!(
                "execution record: pass {index} resolved {:?} != the plan's {:?}",
                entry.resolved, plan.passes[index]
            ));
        }
        if entry.renoised != (index > 0) {
            return Err(format!(
                "execution record: pass {index} renoised={} — only passes after the first \
                 re-noise",
                entry.renoised
            ));
        }
        if entry.effective_steps == 0 || entry.model_evals == 0 {
            return Err(format!(
                "execution record: pass {index} recorded {} step(s) / {} eval(s)",
                entry.effective_steps, entry.model_evals
            ));
        }
    }
    let summed: usize = record.passes.iter().map(|p| p.model_evals).sum();
    if record.total_model_evals != summed {
        return Err(format!(
            "execution record: total_model_evals {} != the per-pass sum {summed}",
            record.total_model_evals
        ));
    }
    Ok(record)
}
/// [`denoise_pass_conformance_over`] for the FLOW cohort — the shorthand every flow family uses.
pub fn denoise_pass_conformance(family: &str, builder: &PassScheduleBuilder<'_>) {
    denoise_pass_conformance_over(family, &flow_ms(), builder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::sampling::{schedule_sigmas, Scheduler};

    /// The shared gen-core schedule math itself passes the whole suite — the reference every
    /// family seam is compared against. Parameterized by the cohort's `ModelSampling`, exactly as a
    /// real family's builder is (`schedule_sigmas` reads the σ table off it).
    fn gen_core_builder<M: ModelSampling + Clone + 'static>(
        ms: M,
    ) -> impl Fn(&ResolvedDenoisePass, usize) -> Vec<f32> {
        move |pass: &ResolvedDenoisePass, steps: usize| {
            let sched = Scheduler::from_name(&pass.scheduler).unwrap_or(Scheduler::Normal);
            schedule_sigmas(sched, &ms, steps)
        }
    }

    #[test]
    fn shared_gen_core_math_passes_the_whole_suite() {
        denoise_pass_conformance("gen-core", &gen_core_builder(flow_ms()));
    }

    /// **The sc-20425 item-5 gap.** The suite was flow-only, so a green run said nothing about the
    /// VE / discrete ε cohort — SDXL and Kolors — whose boundary re-noise is a *different formula*.
    /// Running the identical suite over `DiscreteModelSampling` is what makes it able to certify
    /// them.
    #[test]
    fn the_whole_suite_also_runs_over_a_discrete_ve_model() {
        let ms = sdxl_discrete_model_sampling();
        assert!(
            ms.sigma_max() > 10.0,
            "the SDXL σ range is the point — a unit-σ stand-in would not exercise the VE blend"
        );
        denoise_pass_conformance_over("gen-core discrete", &ms, &gen_core_builder(ms.clone()));
    }

    /// A `ModelSampling` that **declares** the discrete ε cohort while blending like FLOW — the
    /// advertise/honor mismatch a family lands on by copying a sibling's `noise_scaling_coeffs`
    /// alongside its own prediction type. Everything but the declaration delegates to flow.
    struct MisdeclaredVe(FlowModelSampling);

    impl ModelSampling for MisdeclaredVe {
        fn prediction(&self) -> gen_core::sampling::PredictionType {
            gen_core::sampling::PredictionType::Eps
        }
        fn sigma_min(&self) -> f32 {
            self.0.sigma_min()
        }
        fn sigma_max(&self) -> f32 {
            self.0.sigma_max()
        }
        fn input_scale(&self, sigma: f32) -> f32 {
            self.0.input_scale(sigma)
        }
        fn denoised_coeffs(&self, sigma: f32) -> (f32, f32) {
            self.0.denoised_coeffs(sigma)
        }
        fn noise_scaling_coeffs(&self, sigma: f32) -> (f32, f32) {
            self.0.noise_scaling_coeffs(sigma)
        }
        fn timestep(&self, sigma: f32) -> f32 {
            self.0.timestep(sigma)
        }
        fn sigma(&self, timestep: f32) -> f32 {
            self.0.sigma(timestep)
        }
    }

    /// Negative self-test: a family whose declared prediction type disagrees with the blend it
    /// actually re-noises with is caught, and caught AS THAT — not as the generic "cannot
    /// discriminate" the earlier version returned, whose assertion matched every message this
    /// function can emit and so discriminated nothing (sc-20425 review MINOR 3).
    #[test]
    fn a_model_that_blends_unlike_its_declared_cohort_is_reported() {
        let lying = MisdeclaredVe(flow_ms());
        let err = check_renoise_matches_model_type(&gen_core_builder(flow_ms()), &lying)
            .expect_err("a declared-VE model blending like flow must fail");
        assert!(
            err.contains("the declaration and the blend disagree"),
            "expected the misdeclaration diagnosis, got: {err}"
        );
        assert!(
            err.contains("Eps"),
            "it must name the declared cohort: {err}"
        );
        // And it is NOT the σ-degenerate diagnosis, which would mean the harness gave up rather
        // than found something.
        assert!(!err.contains("cannot discriminate"), "{err}");
    }

    /// Negative self-test for the two EXECUTOR-regression arms, driven through the real classifier.
    ///
    /// No `ModelSampling` can trip them — the executor blends through the `ms` it was handed, so a
    /// correct one always lands on `mine` — which is why the comparison is split out and called
    /// directly here. Without this the arms were untested code the reviewer could only read as dead
    /// (sc-20425 review MINOR 3).
    #[test]
    fn a_boundary_latent_from_the_wrong_cohort_is_reported() {
        use gen_core::sampling::PredictionType;

        let mine = vec![1.0_f32, 2.0, 3.0];
        let theirs = vec![1.5_f32, 2.5, 3.5];
        let neither = vec![9.0_f32, 9.0, 9.0];
        let coeffs = (1.0_f32, 0.5_f32);

        // The healthy case: the executor blended through this model's own coefficients.
        classify_boundary_latent(&mine, &mine, &theirs, PredictionType::Eps, coeffs, 0.5)
            .expect("this cohort's blend is the pass condition");

        // The executor used the other cohort's formula.
        let err =
            classify_boundary_latent(&theirs, &mine, &theirs, PredictionType::Eps, coeffs, 0.5)
                .expect_err("the other cohort's blend must be reported");
        assert!(
            err.contains("the boundary used the OTHER cohort's blend"),
            "{err}"
        );
        assert!(
            err.contains("Eps"),
            "it must name the declared cohort: {err}"
        );

        // The executor used neither — a third formula entirely.
        let err =
            classify_boundary_latent(&neither, &mine, &theirs, PredictionType::Flow, coeffs, 0.5)
                .expect_err("an unrecognized blend must be reported");
        assert!(err.contains("neither cohort's blend"), "{err}");
        assert!(
            err.contains("not reading Flow's noise_scaling_coeffs"),
            "{err}"
        );
    }

    /// Negative self-test: a boundary that lands on σ = 0 cannot tell the two blends apart (they
    /// coincide there), so the check refuses to report a pass it did not earn.
    #[test]
    fn a_boundary_that_cannot_discriminate_the_blends_is_reported() {
        let ms = sdxl_discrete_model_sampling();
        let degenerate = {
            let inner = gen_core_builder(ms.clone());
            move |pass: &ResolvedDenoisePass, steps: usize| {
                if pass.index == 0 {
                    inner(pass, steps)
                } else {
                    vec![0.0, 0.0]
                }
            }
        };
        let err = check_renoise_matches_model_type(&degenerate, &ms)
            .expect_err("a σ = 0 boundary must not be reported as a pass");
        assert!(
            err.contains("cannot discriminate the"),
            "expected the σ-degenerate diagnosis, got: {err}"
        );
        // And it is NOT the misdeclaration diagnosis: this model is consistent, the boundary is not
        // informative.
        assert!(
            !err.contains("the declaration and the blend disagree"),
            "{err}"
        );
    }

    /// The two cohorts really do produce different chains from the same plan — the property the
    /// discrimination above relies on, asserted directly so a future change to
    /// `noise_scaling_coeffs` that collapsed them would be caught here rather than silently
    /// weakening every VE certification.
    #[test]
    fn the_two_cohorts_do_not_produce_the_same_chain() {
        let plan = plan_for(vec![pass(3, "euler", 1.0), pass(3, "euler", 0.5)], 4242)
            .expect("plan resolves");
        let flow = flow_ms();
        let discrete = sdxl_discrete_model_sampling();
        let (a, _) = run_chain(
            &gen_core_builder(flow),
            &flow,
            &plan,
            &mut |_| {},
            &CancelFlag::new(),
        )
        .expect("flow chain runs");
        let (b, _) = run_chain(
            &gen_core_builder(discrete.clone()),
            &discrete,
            &plan,
            &mut |_| {},
            &CancelFlag::new(),
        )
        .expect("discrete chain runs");
        assert_ne!(a.latent, b.latent);
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
            let mut s = schedule_sigmas(sched, &flow_ms(), steps);
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
        let err = check_one_pass_equivalence(&broken, &flow_ms())
            .expect_err("a drifting builder must fail");
        assert!(err.contains("one-pass equivalence"), "{err}");
    }

    /// Negative self-test: a builder emitting a schedule with no trailing zero fails the shape
    /// check.
    #[test]
    fn a_builder_without_terminal_zero_is_reported() {
        let broken = |pass: &ResolvedDenoisePass, steps: usize| {
            let sched = Scheduler::from_name(&pass.scheduler).unwrap_or(Scheduler::Normal);
            let mut s = schedule_sigmas(sched, &flow_ms(), steps);
            if let Some(last) = s.last_mut() {
                *last = 0.01;
            }
            s
        };
        let err = check_pass_schedule_shape(&broken).expect_err("no trailing zero must fail");
        assert!(err.contains("trailing 0.0"), "{err}");
    }
}
