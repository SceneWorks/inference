//! Minimal source profiling for the single bounded Krea acceptance invocation.
//! No production setting or preview cadence is changed. A proof-only scope
//! enables three matched six-update segments (18 TOTAL), never a second job.

#[cfg(feature = "model-inference-proof")]
use std::cell::{Cell, RefCell};

#[cfg(feature = "model-inference-proof")]
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static RECOVERY: Cell<bool> = const { Cell::new(false) };
    static RECOVERY_REPORT: RefCell<Option<serde_json::Value>> = const { RefCell::new(None) };
    static SEGMENT: Cell<usize> = const { Cell::new(0) };
    static ATTRIBUTION: Cell<bool> = const { Cell::new(false) };
    static TIMES: RefCell<Vec<(String,f64)>> = const { RefCell::new(Vec::new()) };
}

/// Proof-only original arithmetic selector; always false in production builds.
pub fn reference_math() -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        ACTIVE.get() && SEGMENT.get() == 0
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        false
    }
}

/// Whether the exact bounded acceptance invocation has opted in.
pub fn active() -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        ACTIVE.get()
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        false
    }
}

/// Opt into the bounded experiment on this synchronous thread. Production
/// callers cannot enable it without the manual proof feature. Nested scopes
/// are rejected, and all thread-local state is restored on errors/panics.
#[cfg(feature = "model-inference-proof")]
pub fn with_bounded_krea_profile<T>(f: impl FnOnce() -> T) -> T {
    assert!(!ACTIVE.replace(true), "nested bounded training profile");
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.set(false);
            RECOVERY.set(false);
            RECOVERY_REPORT.with(|report| *report.borrow_mut() = None);
            SEGMENT.set(0);
            ATTRIBUTION.set(false);
            TIMES.with(|t| t.borrow_mut().clear());
        }
    }
    let _reset = Reset;
    SEGMENT.set(0);
    f()
}

/// Explicit continuation of the failed M3 experiment. The eighteen-update default
/// remains separate; only six fresh candidate updates can enter this scope.
pub fn recovery_active() -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        RECOVERY.get()
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        false
    }
}

/// Immutable audited scalar observations, never executable artifact content.
#[cfg(feature = "model-inference-proof")]
pub fn krea_recovery_baseline() -> serde_json::Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../scripts/ci/krea-recovery-baseline.json"
    )))
    .expect("source-owned baseline JSON")
}

#[cfg(feature = "model-inference-proof")]
pub fn with_bounded_krea_recovery<T>(f: impl FnOnce() -> T) -> (T, Option<serde_json::Value>) {
    with_bounded_krea_profile(|| {
        RECOVERY.set(true);
        let result = f();
        let report = RECOVERY_REPORT.with(|report| report.borrow_mut().take());
        (result, report)
    })
}

/// Recovery has no saved shared adapter to restore on NO-GO. Preserve the actual
/// trained candidate for save/reload/apply, and expose adoption separately.
#[cfg(feature = "model-inference-proof")]
pub(crate) fn finish_krea_recovery(
    times: &[f64],
    losses: &[f32],
    candidate_dispatches: u64,
) -> candle_core::Result<()> {
    if !recovery_active()
        || times.len() != 4
        || losses.len() != 6
        || times.iter().any(|t| !t.is_finite() || *t <= 0.)
        || losses.iter().any(|l| !l.is_finite())
    {
        candle_core::bail!("incomplete six-update Krea recovery");
    }
    let baseline = krea_recovery_baseline();
    let shared = &baseline["reference_segments"][1]["steps"];
    let loss_parity = shared
        .as_array()
        .unwrap()
        .iter()
        .zip(losses)
        .all(|(old, new)| {
            let old: f64 = old["loss"].as_str().unwrap().parse().unwrap();
            (old - f64::from(*new)).abs() <= 0.01 * old.abs().max(1.)
        });
    let mean = |segment: usize| {
        baseline["reference_segments"][segment]["steps"]
            .as_array()
            .unwrap()
            .iter()
            .skip(2)
            .map(|step| step["seconds"].as_str().unwrap().parse::<f64>().unwrap())
            .sum::<f64>()
            / 4.
    };
    let reference = mean(0);
    let shared = mean(1);
    let candidate = times.iter().sum::<f64>() / 4.;
    let uncertainty = baseline["timing_mean_rounding_uncertainty_seconds"]
        .as_f64()
        .unwrap();
    let go_ambiguous = (candidate - shared).abs() <= uncertainty;
    let go = candidate_dispatches > 0 && loss_parity && candidate < shared - uncertainty;
    let target_ambiguous = (candidate - 0.75 * reference).abs() <= 0.75 * uncertainty;
    let target_met = go && !target_ambiguous && candidate <= 0.75 * (reference - uncertainty);
    let report = serde_json::json!({
        "schema_version": 1, "new_optimizer_updates": 6, "new_segment": 2,
        "steady_updates": 4, "prior_conservative_update_charge": 14,
        "hard_update_cap": 20, "remaining_updates_after_success": 0,
        "baseline": baseline, "reference_logged_mean_seconds": reference,
        "shared_logged_mean_seconds": shared, "candidate_mean_seconds": candidate,
        "candidate_steady_seconds": times, "candidate_losses": losses,
        "candidate_dispatches": candidate_dispatches, "loss_parity": loss_parity,
        "candidate_go": go, "go_rounding_boundary_ambiguous": go_ambiguous,
        "target_met": target_met, "target_rounding_boundary_ambiguous": target_ambiguous,
        "candidate_reduction_vs_original_logged_mean": 1. - candidate / reference,
        "saved_adapter_role": "actual-six-update-candidate-even-on-no-go",
        "shared_adapter_available": false,
        "claim_limit": "Cross-run recovery, not a new matched 18-update invocation. Original reference source/run/log identity retained; unchanged source/input closure qualification is mandatory. NO-GO never means a default Reference or saved shared-adapter success. Smoke completion does not imply attention adoption, the 25 percent target or convergence."
    });
    eprintln!("KREA_PROFILE_RECOVERY_RESULT {report}");
    RECOVERY_REPORT.with(|state| *state.borrow_mut() = Some(report));
    Ok(())
}

pub(crate) fn begin_step(segment: usize, logical_step: u32) {
    #[cfg(feature = "model-inference-proof")]
    {
        SEGMENT.set(segment);
        ATTRIBUTION.set(logical_step == 1);
    }
    #[cfg(not(feature = "model-inference-proof"))]
    let _ = (segment, logical_step);
}

/// Durable failure/budget boundary, called outside the measured step interval.
/// A start without a commit conservatively consumes one attempted update.
pub(crate) fn boundary(detail: &str) -> candle_core::Result<()> {
    #[cfg(feature = "model-inference-proof")]
    if active() || cohort_active() {
        use std::io::Write;
        let mut stderr = std::io::stderr().lock();
        let prefix = if active() {
            "KREA_PROFILE_BOUNDARY"
        } else {
            "COHORT_PROFILE_BOUNDARY"
        };
        writeln!(stderr, "{prefix} {detail}").map_err(candle_core::Error::wrap)?;
        stderr.flush().map_err(candle_core::Error::wrap)?;
    }
    let _ = detail;
    Ok(())
}

/// Synchronized wall span, enabled ONLY on the first (excluded warmup) update
/// of each segment. These are inclusive operation spans, not hardware counters
/// or independent kernel time. Steady timings have no per-operation barriers.
pub fn span<T>(
    name: &str,
    device: &candle_core::Device,
    f: impl FnOnce() -> candle_core::Result<T>,
) -> candle_core::Result<T> {
    #[cfg(feature = "model-inference-proof")]
    if active() && ATTRIBUTION.get() {
        device.synchronize()?;
        let start = std::time::Instant::now();
        let result = f()?;
        device.synchronize()?;
        TIMES.with(|t| {
            t.borrow_mut()
                .push((name.to_string(), start.elapsed().as_secs_f64()))
        });
        return Ok(result);
    }
    let _ = (name, device);
    f()
}

pub(crate) fn finish_attribution(segment: usize) {
    #[cfg(feature = "model-inference-proof")]
    if active() && ATTRIBUTION.replace(false) {
        let mut sums = std::collections::BTreeMap::<String, (usize, f64)>::new();
        TIMES.with(|t| {
            for (name, sec) in t.borrow_mut().drain(..) {
                let v = sums.entry(name).or_default();
                v.0 += 1;
                v.1 += sec;
            }
        });
        eprintln!("KREA_PROFILE_ATTRIBUTION segment={segment} synchronized_inclusive_spans={sums:?}; launches/GEMM/norm backward inferred from source; no CUDA hardware counters");
    }
    let _ = segment;
}

// sc-26857: observations for a single four-update cohort invocation. This scope
// is independent of Krea's matched 18-update experiment and never selects math.
#[cfg(feature = "model-inference-proof")]
#[derive(Default)]
struct CohortObservation {
    started: Option<std::time::Instant>,
    step_started: Option<std::time::Instant>,
    preparation_seconds: f64,
    updates: u32,
    steps: Vec<serde_json::Value>,
    gradients: Vec<candle_core::Tensor>,
    initial: std::collections::HashMap<candle_core::TensorId, candle_core::Tensor>,
    reuse: std::collections::BTreeMap<String, u64>,
    probes: std::collections::HashSet<String>,
}
#[cfg(feature = "model-inference-proof")]
thread_local! {
    static COHORT: RefCell<Option<CohortObservation>> = const { RefCell::new(None) };
}

pub fn cohort_active() -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        COHORT.with(|state| state.borrow().is_some())
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        false
    }
}

/// Existing manual proof cases opt in on the training thread. Four updates are
/// source-owned, and cancelled/partial/missing observer work cannot report PASS.
#[cfg(feature = "model-inference-proof")]
pub fn with_bounded_cohort<T>(output: &std::path::Path, f: impl FnOnce() -> T) -> T {
    assert!(!active() && !cohort_active(), "nested bounded proof scope");
    COHORT.with(|state| {
        *state.borrow_mut() = Some(CohortObservation {
            started: Some(std::time::Instant::now()),
            ..Default::default()
        })
    });
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            COHORT.with(|s| *s.borrow_mut() = None);
        }
    }
    let _reset = Reset;
    let result = f();
    COHORT.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().unwrap();
        assert_eq!(state.updates, 4, "one invocation must complete four actual optimizer updates");
        assert_eq!(state.steps.len(), 4, "all four bodies must finish");
        assert!(!state.reuse.is_empty(), "prepared mechanism consumption must be observed");
        let report = serde_json::json!({"optimizer_updates": state.updates,
            "preparation_including_load_cache_seconds": state.preparation_seconds,
            "steps": state.steps, "mechanism_consumption": state.reuse,
            "timing_scope": "synchronized forward/backward/update; first update warmup; diagnostic scalar validation excluded"});
        std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        eprintln!("COHORT_ACCEPTANCE_METRICS {report}");
    });
    result
}

/// Called at the consuming branch (cache hit / prepared geometry / actual pack),
/// never merely at preparation. IDs/residency belong to the owning cache checks.
pub fn cohort_reuse(mechanism: &str) {
    #[cfg(feature = "model-inference-proof")]
    COHORT.with(|state| {
        if let Some(state) = state.borrow_mut().as_mut() {
            *state.reuse.entry(mechanism.into()).or_default() += 1;
        }
    });
    let _ = mechanism;
}

/// Claim one bounded reference probe during the first excluded warmup only.
pub(crate) fn cohort_claim_probe(mechanism: &str) -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        COHORT.with(|state| {
            let mut state = state.borrow_mut();
            let Some(state) = state.as_mut() else {
                return false;
            };
            if state.updates != 0 || !state.steps.is_empty() {
                return false;
            }
            state.probes.insert(mechanism.into())
        })
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        let _ = mechanism;
        false
    }
}

pub fn cohort_begin_step(device: &candle_core::Device) -> candle_core::Result<()> {
    if !cohort_active() {
        return Ok(());
    }
    device.synchronize()?;
    #[cfg(feature = "model-inference-proof")]
    COHORT.with(|state| {
        let mut state = state.borrow_mut();
        let state = state.as_mut().unwrap();
        assert!(device.is_cuda(), "bounded cohort requires actual CUDA");
        if let Some(started) = state.started.take() {
            state.preparation_seconds = started.elapsed().as_secs_f64();
        }
        eprintln!("COHORT_UPDATE_START attempted_update={}", state.updates + 1);
        state.step_started = Some(std::time::Instant::now());
        crate::training_attention::take_training_attention_counts();
    });
    Ok(())
}

/// Capture factors before timing starts when a bespoke loop owns several experts.
/// The diagnostic copies are run-owned and included in observed device peak.
pub fn cohort_capture_initial(vars: &[candle_core::Var]) -> candle_core::Result<()> {
    #[cfg(feature = "model-inference-proof")]
    COHORT.with(|state| -> candle_core::Result<()> {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return Ok(());
        };
        for var in vars {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                state.initial.entry(var.as_tensor().id())
            {
                entry.insert(var.as_tensor().detach().copy()?);
            }
        }
        Ok(())
    })?;
    let _ = vars;
    Ok(())
}

/// Clone already-produced gradients at the actual optimizer boundary. Initial
/// factor copies happen only in excluded warmup (bespoke experts can pre-capture).
pub fn cohort_before_update(
    grads: &candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
) -> candle_core::Result<()> {
    #[cfg(feature = "model-inference-proof")]
    COHORT.with(|state| -> candle_core::Result<()> {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return Ok(());
        };
        if state.updates >= 20 {
            candle_core::bail!("bounded proof exceeded twenty optimizer updates");
        }
        for var in vars {
            let tensor = var.as_tensor();
            let grad = grads.get(tensor).ok_or_else(|| {
                candle_core::Error::Msg("missing reached adapter gradient".into())
            })?;
            state.gradients.push(grad.clone());
            if let std::collections::hash_map::Entry::Vacant(entry) =
                state.initial.entry(tensor.id())
            {
                entry.insert(tensor.detach().copy()?);
            }
        }
        Ok(())
    })?;
    let _ = (grads, vars);
    Ok(())
}

pub fn cohort_after_update() {
    #[cfg(feature = "model-inference-proof")]
    COHORT.with(|s| {
        if let Some(s) = s.borrow_mut().as_mut() {
            s.updates += 1;
        }
    });
}

pub fn cohort_end_step(
    device: &candle_core::Device,
    loss: f32,
    vars: &[candle_core::Var],
    owner: &str,
) -> candle_core::Result<()> {
    if !cohort_active() {
        return Ok(());
    }
    device.synchronize()?;
    #[cfg(feature = "model-inference-proof")]
    COHORT.with(|state| -> candle_core::Result<()> {
        let mut state = state.borrow_mut(); let state = state.as_mut().unwrap();
        let seconds = state.step_started.take().expect("step started").elapsed().as_secs_f64();
        // Durable count after synchronization and after closing the timed span.
        eprintln!("COHORT_UPDATE_COMMITTED optimizer_updates={}", state.updates);
        let attention = crate::training_attention::take_training_attention_counts();
        if !loss.is_finite() { candle_core::bail!("non-finite cohort loss"); }
        let mut nonzero_gradients = 0;
        let gradient_count = state.gradients.len();
        if gradient_count == 0 { candle_core::bail!("missing optimizer gradient witnesses"); }
        for grad in state.gradients.drain(..) {
            let norm = grad.to_dtype(candle_core::DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            if !norm.is_finite() { candle_core::bail!("non-finite cohort gradient"); }
            nonzero_gradients += usize::from(norm > 0.);
        }
        let mut changed = false;
        for var in vars {
            let start = state.initial.get(&var.as_tensor().id()).expect("initial factor");
            let delta = (var.as_tensor() - start)?.to_dtype(candle_core::DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            if !delta.is_finite() { candle_core::bail!("non-finite cohort adapter update"); }
            changed |= delta > 0.;
        }
        if !changed || nonzero_gradients == 0 { candle_core::bail!("cohort optimizer produced no actual adapter update"); }
        let warmup = state.steps.is_empty();
        state.steps.push(serde_json::json!({"update":state.updates,"owner":owner,"warmup":warmup,
            "seconds":seconds,"loss":loss,"finite_gradients":gradient_count,"nonzero_gradients":nonzero_gradients,
            "adapter_changed":changed,"attention_candidate":attention.0,"attention_reference":attention.1}));
        Ok(())
    })?;
    let _ = (loss, vars, owner);
    Ok(())
}

/// Require the exact mechanism branches named by a source-owned bounded case.
/// Counts describe consumption in this invocation; preparation alone cannot pass.
#[cfg(feature = "model-inference-proof")]
pub fn with_bounded_cohort_for<T>(
    output: &std::path::Path,
    required: &[&str],
    f: impl FnOnce() -> T,
) -> T {
    with_bounded_cohort(output, || {
        let result = f();
        COHORT.with(|state| {
            let state = state.borrow();
            let state = state.as_ref().unwrap();
            for mechanism in required {
                assert!(
                    state.reuse.get(*mechanism).is_some_and(|count| *count >= 2),
                    "bounded source case must repeatedly consume {mechanism}"
                );
            }
        });
        result
    })
}
