//! The approximation contract-to-mechanism bridges (trunk cache, token pruner) and their
//! trajectory guard.

use super::*;

/// The dense 5B's declared approximate-capability surface (sc-18322).
///
/// The candidate intervals are the mechanism's implemented operating points, not measured ones — the
/// domain shape carries no measurement claim, and the characterization binding (absent, and
/// unconstructible) is what refuses selection until one exists. Warmup is capped at 8 steps because the
/// native Wan trajectories run 20-50 steps and a warmup past a third of them leaves nothing to reuse.
pub(crate) fn approximation_surface() -> ApproximationSurface {
    ApproximationSurface::feature_cache(vec![2, 3, 4], 8).with_token_pruning(vec![2, 3, 4], 8)
}

/// The denoise feature cache a resolved plan asks for — the provider-side bridge from contract to
/// mechanism (sc-18322).
///
/// Deliberately two definitions rather than one with an inner `cfg`, so the production answer is
/// visibly and unconditionally `None`: [`TrunkCache`]'s only constructor is `#[cfg(test)]`, so a
/// production build cannot reach the uncharacterized path even if it somehow held a
/// [`ApproximationPlan::Approximate`]. That is the second of two independent locks — the first is the
/// contract refusing every approximate selection for want of a characterization artifact.
/// Both bridges first run [`refuse_inert_over_trajectory`], because the step count is only known here.
#[cfg(not(test))]
pub(super) fn trunk_cache(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
) -> Result<Option<TrunkCache>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    Ok(None)
}

#[cfg(test)]
pub(super) fn trunk_cache(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
) -> Result<Option<TrunkCache>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    Ok(TrunkCache::from_plan_for_test(plan))
}

/// The token pruner a resolved plan asks for — the second provider-side bridge, locked exactly like
/// [`trunk_cache`] (sc-18322). `total` is the generate's patchified token count, which the rotation
/// phases are built over.
#[cfg(not(test))]
pub(super) fn token_pruner(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
    _total: usize,
) -> Result<Option<TokenPruner>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    Ok(None)
}

#[cfg(test)]
pub(super) fn token_pruner(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
    total: usize,
) -> Result<Option<TokenPruner>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    TokenPruner::from_plan_for_test(plan, total)
}

/// Refuse an approximate policy that would engage on **no step** of this request's trajectory
/// (sc-18322).
///
/// The last way to encode *off* inside an `Approximate` plan. A warmup is gated against the declared
/// `max_warmup_steps` without knowing the step count — a request may leave `steps` to the model's default
/// — so a policy well inside its declared domain can still cover the whole trajectory and produce a
/// result bit-identical to the exact path. That is exactly the second-encoding-of-off defect that
/// rejecting a reuse interval of `1` exists to prevent, and letting it through here would make the
/// contract's own "an approximate plan changed the output" claim false.
///
/// This is the one place the step count and the policy are both in hand, so it is the one place the
/// refusal can be made. Unreachable today (no plan reaching a provider can be anything but `Exact`), and
/// it fails closed the day selection becomes possible.
pub(crate) fn refuse_inert_over_trajectory(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
) -> Result<()> {
    if let Some(policy) = plan.feature_cache() {
        if policy.is_inert_over(steps) {
            return Err(Error::Unsupported(format!(
                "{id}: the selected denoise feature cache (interval {} steps, warmup {}) reuses \
                 nothing over this request's {steps} steps, so it would produce the exact result \
                 through the approximate path. Lower the warmup, raise the step count, or leave \
                 approximation unset.",
                policy.reuse_interval.steps(),
                policy.warmup.steps()
            )));
        }
    }
    if let Some(policy) = plan.token_pruning() {
        if policy.is_inert_over(steps) {
            return Err(Error::Unsupported(format!(
                "{id}: the selected token pruning (drop stride {}, warmup {}) prunes nothing over \
                 this request's {steps} steps, so it would produce the exact result through the \
                 approximate path. Lower the warmup, raise the step count, or leave approximation \
                 unset.",
                policy.drop_stride.stride(),
                policy.warmup.steps()
            )));
        }
    }
    Ok(())
}
