//! Typed, admission-aware **approximate capabilities** — levers that make a render *cheaper by
//! changing its result* (sc-18322, epic 18304 P7).
//!
//! [`crate::execution_domains`] is this module's sibling and its deliberate opposite. Both declare
//! per provider on [`Capabilities`](crate::Capabilities), both are selected per request, and both
//! fail closed at
//! [`Capabilities::validate_request`](crate::Capabilities::validate_request) rather than letting a
//! provider silently ignore a selection. The difference is the *equivalence class*: every execution
//! domain is bit-identical or numerically equivalent by construction, so a planner may select one
//! freely. Nothing here is. A denoise feature cache reuses a transformer block's output from an
//! earlier step, so the trajectory it produces is a **different** trajectory — cheaper, and worse by
//! an amount only measurement can state.
//!
//! That single difference drives the two properties this module has and `execution_domains`
//! deliberately excludes.
//!
//! # 1. Byte-identical when off, and *off is absence*
//!
//! The `Default` of every type here is the provider's own exact path, and the off state is the
//! **absence** of a selection rather than a parameter value that happens to be inert:
//!
//! * [`GenerationRequest::approximation`](crate::GenerationRequest::approximation) is an `Option`
//!   whose `None` (the `Default`) is the pre-sc-18322 request, and
//!   [`ApproximationRequest::default()`] selects nothing.
//! * [`ApproximationPlan::Exact`] — the one value [`ApproximationSurface::resolve`] can return today
//!   — is a variant with **no fields**. There is no plan that carries an approximation configured to
//!   do nothing, so "off" cannot be expressed as a traversal of the approximate code path.
//! * [`CacheReuseInterval`] rejects `1` as hard as it rejects `0`. An interval of one step means
//!   "recompute every step", which is arithmetically the exact path — but reached *through* the cache,
//!   with its bookkeeping, its allocations and its evaluation schedule. Admitting it would create a
//!   second encoding of off whose byte-identity nobody could prove, so the type refuses it and the
//!   only way to be off is to not select.
//!
//! The consumer-side half of the guarantee is that a provider branches on
//! [`ApproximationPlan::is_exact`] exactly once, at the top of its denoise, and the `Exact` arm is
//! the untouched pre-existing code. That is what the family's tensor-level test proves: with the plan
//! `Exact`, every step of a multi-step run is **bit-identical** to an independent stateless call to
//! the same block stack — no cross-step state, no reordering, no perturbation.
//!
//! # 2. Selection is bound to a quality characterization, so today it is *structurally* impossible
//!
//! An approximate mechanism that ships selectable-but-unmeasured is a quality regression waiting for
//! a user to find. So the contract makes measurement a *precondition of selection*, not a convention:
//! an approximate selection is admitted only when the request carries a
//! [`CharacterizationRef`] **and** the provider's surface carries the
//! [`CharacterizationBinding`] that vouches for it.
//!
//! No quality-characterization artifact format ships yet — that is epic 18304's terminal measurement
//! campaign, which runs once at epic end. This module therefore defines the **seam** and nothing
//! more: an opaque reference ([`CharacterizationRef`] — an artifact id plus a digest, uninterpreted
//! here) and a validation hook ([`CharacterizationFamily::validate`]). The hook's input type
//! [`CharacterizationFamily`] is **uninhabited**, so no [`CharacterizationBinding::Bound`] value can
//! exist, so no provider can declare a binding, so *every* approximate selection is refused — not by
//! a runtime flag someone could flip, but because the admitting state has no inhabitant. The
//! campaign adds the variant and fills the hook; nothing else about the contract, the declarations or
//! the provider mechanism has to move.
//!
//! The resulting shipped state is the intended one: **declared, implemented, and opt-in-refused until
//! characterized.** The mechanism is real code on a real family, exercised by real tests, reachable
//! only through a `#[cfg(test)]` bypass in the consuming crate — never from a request.
//!
//! # Declaration is not reachability (and here, not selectability either)
//!
//! As in the ladder families and `execution_domains`, a provider may only declare a mechanism it
//! actually consumes, and the pairing is proven per family by a weights-free test. This module adds a
//! third rung to that ladder of claims, because an approximate mechanism has one more way to lie:
//!
//! | claim | proven by |
//! |---|---|
//! | *declared* | [`ApproximationSurface`] on the descriptor, coherent per [`ApproximationSurface::declaration_errors`] |
//! | *implemented* | the family's tensor-level tests — off is bit-identical, on genuinely reuses |
//! | *selectable* | **nothing, deliberately.** No binding exists, so every selection is refused by name |
//!
//! A consumer that reads a declared mechanism off a descriptor and concludes it can *use* it would be
//! wrong, which is why [`ApproximationSurface::is_selectable`] exists and answers `false` for every
//! provider today.

use std::num::NonZeroU32;

use crate::{Error, Result};

/// The smallest reuse interval that is actually an approximation.
///
/// One step apart means "recompute every step" — the exact path — so it is not an interval. See the
/// module docs: off is absence, never a parameter value.
const MIN_REUSE_INTERVAL: u32 = 2;

/// How many denoise steps apart a cached transformer feature is **recomputed**.
///
/// A denoise trajectory evaluates the same transformer stack at successive noise levels, and
/// consecutive steps produce highly similar block outputs. A feature cache exploits that: recompute
/// the stack on every `interval`-th step and reuse the retained output in between, trading a fraction
/// of the step's arithmetic for a trajectory that is close to — but not — the exact one.
///
/// `2` is the tightest interval and the most conservative approximation (every other step reused).
/// Larger intervals are cheaper and further from exact. `1` and `0` are both rejected: `1` is the
/// exact path expressed as a cache traversal (see the module docs), and `0` is not an interval at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheReuseInterval(NonZeroU32);

impl CacheReuseInterval {
    /// Recompute every other step, reuse the step in between — the tightest interval.
    pub const EVERY_OTHER_STEP: Self = Self(NonZeroU32::new(MIN_REUSE_INTERVAL).unwrap());

    /// An interval of `steps` denoise steps between recomputations.
    ///
    /// Rejects anything below [`MIN_REUSE_INTERVAL`] rather than clamping, because both rejected
    /// values mean something a caller must fix rather than something the provider can guess: `0` is
    /// malformed, and `1` is a request for the exact path, which is spelled by leaving the selection
    /// unset.
    pub fn new(steps: u32) -> Result<Self> {
        if steps < MIN_REUSE_INTERVAL {
            return Err(Error::Msg(format!(
                "cache reuse interval must be >= {MIN_REUSE_INTERVAL} steps (got {steps}); an \
                 interval of 1 recomputes every step, which is the exact path — leave the \
                 approximation unset for that"
            )));
        }
        Ok(Self(NonZeroU32::new(steps).expect("checked >= 2")))
    }

    /// Steps between recomputations.
    pub const fn steps(self) -> u32 {
        self.0.get()
    }

    /// Whether the zero-based denoise step `index`, in a run with `warmup` leading exact steps,
    /// **recomputes** the stack (as opposed to reusing the cache).
    ///
    /// The single decision point for every consumer, so a provider cannot get the phase wrong in one
    /// loop and right in another. Every step below `warmup` recomputes, and the interval then counts
    /// from the end of warmup — so step `warmup` itself always recomputes and primes the cache, which
    /// is what makes a reuse decision never read an empty cache.
    pub const fn recomputes_step(self, index: usize, warmup: CacheWarmupSteps) -> bool {
        let warmup = warmup.steps() as usize;
        if index < warmup {
            return true;
        }
        (index - warmup).is_multiple_of(self.steps() as usize)
    }
}

/// Leading denoise steps that always recompute, before any reuse begins.
///
/// The head of a trajectory is where composition and layout are decided, and it is where consecutive
/// block outputs differ *most* — so it is the worst place to reuse and the standard carve-out for
/// every published step-caching scheme. Zero (the `Default`) is legal and means reuse may begin at
/// the second step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheWarmupSteps(u32);

impl CacheWarmupSteps {
    /// No warmup — reuse may begin immediately after the first (priming) step.
    pub const NONE: Self = Self(0);

    /// `steps` leading exact steps.
    pub const fn new(steps: u32) -> Self {
        Self(steps)
    }

    /// Leading exact steps.
    pub const fn steps(self) -> u32 {
        self.0
    }
}

/// One request's selected denoise-feature-cache policy.
///
/// Deliberately not `Option`-per-field: an interval is what makes this a cache at all, so a policy
/// always has one, and a caller cannot express a half-configured cache whose meaning the provider
/// would have to invent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureCachePolicy {
    /// Steps between recomputations of the cached features.
    pub reuse_interval: CacheReuseInterval,
    /// Leading steps that always recompute.
    pub warmup: CacheWarmupSteps,
}

impl FeatureCachePolicy {
    /// A policy with no warmup.
    pub const fn new(reuse_interval: CacheReuseInterval) -> Self {
        Self {
            reuse_interval,
            warmup: CacheWarmupSteps::NONE,
        }
    }

    /// This policy with `warmup` leading exact steps.
    pub const fn with_warmup(mut self, warmup: CacheWarmupSteps) -> Self {
        self.warmup = warmup;
        self
    }

    /// Whether step `index` recomputes rather than reuses. Delegates to
    /// [`CacheReuseInterval::recomputes_step`] so there is exactly one phase decision in the
    /// workspace.
    pub const fn recomputes_step(&self, index: usize) -> bool {
        self.reuse_interval.recomputes_step(index, self.warmup)
    }
}

/// The production domain of a provider's denoise feature cache.
///
/// [`Unsupported`](Self::Unsupported) is the `Default`, so a provider gains a fail-closed refusal for
/// a mechanism it does not have without writing anything — the same shape as
/// [`ExecutionValueDomain`](crate::ExecutionValueDomain), and for the same reason.
///
/// The admitting variant is deliberately a closed candidate list rather than a range: an approximate
/// operating point is only meaningful with a measurement behind it, so "any interval >= 2" is a claim
/// no provider can honestly make. There is no `AtLeast` shape here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum FeatureCacheDomain {
    /// The provider has no denoise feature cache. Every selection is refused.
    #[default]
    Unsupported,
    /// Exactly these reuse intervals are implemented, with warmup up to `max_warmup_steps`.
    ///
    /// An empty interval list is a declaration error, not a silent `Unsupported` — see
    /// [`ApproximationSurface::declaration_errors`].
    Implemented {
        /// Reuse intervals (in steps) the provider's cache implements. Each must be
        /// >= [`MIN_REUSE_INTERVAL`].
        intervals: Vec<u32>,
        /// The largest warmup the provider's cache honours.
        max_warmup_steps: u32,
    },
}

impl FeatureCacheDomain {
    /// Whether the provider declares the mechanism at all.
    pub const fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Whether `policy` sits inside the declared domain. Always `false` for
    /// [`Unsupported`](Self::Unsupported).
    pub fn accepts(&self, policy: &FeatureCachePolicy) -> bool {
        match self {
            Self::Unsupported => false,
            Self::Implemented {
                intervals,
                max_warmup_steps,
            } => {
                intervals.contains(&policy.reuse_interval.steps())
                    && policy.warmup.steps() <= *max_warmup_steps
            }
        }
    }

    /// Human-readable domain for a refusal message.
    pub fn describe(&self) -> String {
        match self {
            Self::Unsupported => "unsupported".to_owned(),
            Self::Implemented {
                intervals,
                max_warmup_steps,
            } => format!("intervals {intervals:?}, warmup <= {max_warmup_steps}"),
        }
    }
}

/// The **artifact-family seam** for quality characterization — uninhabited today, on purpose.
///
/// Epic 18304's terminal measurement campaign owns what a quality-characterization artifact *is*: the
/// prompt corpus, the metrics, the reference trajectory, the verdict thresholds, the currency rule
/// that stales it. None of that exists yet and this module does not guess at it. What it does define
/// is where the answer plugs in:
///
/// * a variant per artifact family the campaign defines, and
/// * [`validate`](Self::validate), the hook that decides whether a caller's
///   [`CharacterizationRef`] is a current, sufficient characterization of the selected policy.
///
/// Having **no** variants is what makes the shipped state honest. [`CharacterizationBinding::Bound`]
/// carries one of these, so no binding value can be constructed, so no provider can declare that
/// selection is allowed, so [`ApproximationSurface::validate`] refuses every approximate selection —
/// by the shape of the types rather than by a check someone could delete. Adding the campaign's
/// variant is the whole of the future change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CharacterizationFamily {}

impl CharacterizationFamily {
    /// The validation hook: does `reference` characterize `policy` well enough to admit it?
    ///
    /// Unreachable today — the match has no arms because the type has no inhabitants — and that
    /// unreachability *is* the P7 deliverable's "refused until characterized" property. The campaign
    /// implements one arm per artifact family it defines.
    pub fn validate(
        &self,
        reference: &CharacterizationRef,
        policy: &FeatureCachePolicy,
    ) -> Result<()> {
        let _ = (reference, policy);
        match *self {}
    }

    /// Stable label for messages, for the same reason [`validate`](Self::validate) exists.
    pub fn label(&self) -> &'static str {
        match *self {}
    }
}

/// An **opaque reference** to a quality-characterization artifact, carried on
/// [`ApproximationRequest::characterization`].
///
/// Opaque is the point: gen-core stores an id and a digest and interprets neither. The id names the
/// artifact within whatever family the campaign defines; the digest is what makes a *stale* reference
/// distinguishable from a current one, since a characterization is only valid for the closure it was
/// measured against (the same currency problem the calibration records have). Both are checked for
/// non-emptiness and nothing else — a reference that is well-formed here is not thereby a valid
/// characterization, which is [`CharacterizationFamily::validate`]'s job.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CharacterizationRef {
    artifact_id: String,
    digest: String,
}

impl CharacterizationRef {
    /// A reference to artifact `artifact_id` at content digest `digest`.
    ///
    /// Both must be non-empty: an empty half is a malformed reference, and admitting one would let a
    /// caller satisfy the "a reference is present" gate with nothing at all.
    pub fn new(artifact_id: impl Into<String>, digest: impl Into<String>) -> Result<Self> {
        let artifact_id = artifact_id.into();
        let digest = digest.into();
        if artifact_id.trim().is_empty() {
            return Err(Error::Msg(
                "characterization reference requires a non-empty artifact id".to_owned(),
            ));
        }
        if digest.trim().is_empty() {
            return Err(Error::Msg(format!(
                "characterization reference {artifact_id} requires a non-empty content digest; a \
                 characterization is only valid for the closure it was measured against"
            )));
        }
        Ok(Self {
            artifact_id,
            digest,
        })
    }

    /// The artifact id, uninterpreted.
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// The content digest, uninterpreted.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// Whether a provider will accept a [`CharacterizationRef`] at all, and against which artifact
/// family.
///
/// `Default` is [`Unbound`](Self::Unbound) — the fail-closed state, and the only *constructible*
/// state, because [`Bound`](Self::Bound) carries an uninhabited
/// [`CharacterizationFamily`]. See that type for why that is deliberate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CharacterizationBinding {
    /// No artifact family is bound. Every approximate **selection** is refused, even one carrying a
    /// well-formed reference; declaration and implementation are unaffected.
    #[default]
    Unbound,
    /// Bound to an artifact family, whose [`validate`](CharacterizationFamily::validate) hook decides
    /// each reference. Unconstructible until the terminal campaign defines a family.
    Bound(CharacterizationFamily),
}

impl CharacterizationBinding {
    /// Whether any artifact family is bound. `false` for every provider today.
    pub const fn is_bound(&self) -> bool {
        matches!(self, Self::Bound(_))
    }

    /// The bound family, if any.
    pub const fn family(&self) -> Option<&CharacterizationFamily> {
        match self {
            Self::Unbound => None,
            Self::Bound(family) => Some(family),
        }
    }
}

/// One provider's declared **approximate-capability surface**, carried on
/// [`Capabilities::approximation`](crate::Capabilities::approximation).
///
/// `Default` declares nothing and binds nothing, which is the fail-closed state twice over: a
/// provider that has not opted a mechanism in refuses every request that selects it, and a provider
/// that has opted one in still refuses every *selection* until a characterization family exists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ApproximationSurface {
    /// Denoise-feature-cache policies this provider's denoise implements.
    pub denoise_feature_cache: FeatureCacheDomain,
    /// Which characterization artifact family, if any, this provider will accept a reference against.
    pub characterization: CharacterizationBinding,
}

impl ApproximationSurface {
    /// A surface declaring a denoise feature cache over `intervals` with warmup up to
    /// `max_warmup_steps`, and — as every provider must today — no characterization binding.
    pub fn feature_cache(intervals: Vec<u32>, max_warmup_steps: u32) -> Self {
        Self {
            denoise_feature_cache: FeatureCacheDomain::Implemented {
                intervals,
                max_warmup_steps,
            },
            characterization: CharacterizationBinding::Unbound,
        }
    }

    /// `true` when the provider declares no approximate mechanism at all — every selection is refused
    /// for the absent mechanism.
    pub const fn is_inert(&self) -> bool {
        !self.denoise_feature_cache.is_supported()
    }

    /// Whether an approximate selection could be **admitted** on this provider — declaration *and* a
    /// characterization binding.
    ///
    /// `false` for every provider today, and the distinction from
    /// [`is_inert`](Self::is_inert) is the whole point: a consumer reading a declared mechanism must
    /// not conclude it can select it. A declared-but-unselectable mechanism is not a defect, it is
    /// P7's shipped state.
    pub const fn is_selectable(&self) -> bool {
        !self.is_inert() && self.characterization.is_bound()
    }

    /// Gate one request's approximate selections against this surface, and resolve the plan the
    /// provider will execute.
    ///
    /// The **single** seam a provider reads: it cannot both forget the refusal and get a plan, because
    /// the plan is this function's return value. Refusals are the typed [`Error::Unsupported`] when
    /// they are capability gaps (absent mechanism, out-of-domain policy, absent binding) and
    /// [`Error::Msg`] when the request is malformed (a reference with nothing to characterize), the
    /// same split the rest of the shared floor uses (F-008).
    ///
    /// An absent or empty selection resolves to [`ApproximationPlan::Exact`] — the historical path,
    /// which is what makes this addition byte-for-byte inert for every existing request.
    pub fn resolve(
        &self,
        id: &str,
        request: Option<&ApproximationRequest>,
    ) -> Result<ApproximationPlan> {
        let Some(request) = request else {
            return Ok(ApproximationPlan::Exact);
        };
        let Some(policy) = request.denoise_feature_cache else {
            // A reference with nothing to characterize means the caller believes they selected a
            // mechanism and did not. Refusing is the only way they find out.
            if request.characterization.is_some() {
                return Err(Error::Msg(format!(
                    "{id}: a characterization reference was supplied with no approximate selection; \
                     select a mechanism or leave approximation unset"
                )));
            }
            return Ok(ApproximationPlan::Exact);
        };
        if !self.denoise_feature_cache.is_supported() {
            return Err(Error::Unsupported(format!(
                "{id}: a denoise feature cache was selected, but this provider declares no denoise \
                 feature cache mechanism. Leave approximation unset to run the provider's exact \
                 denoise."
            )));
        }
        if !self.denoise_feature_cache.accepts(&policy) {
            return Err(Error::Unsupported(format!(
                "{id}: denoise feature cache (interval {} steps, warmup {}) is outside the declared \
                 production domain {}. Select a declared policy, or leave approximation unset for \
                 the provider's exact denoise.",
                policy.reuse_interval.steps(),
                policy.warmup.steps(),
                self.denoise_feature_cache.describe()
            )));
        }
        // The characterization binding. Both refusals below are reachable today; the admitting path
        // past them is not, because `CharacterizationBinding::Bound` has no inhabitant.
        let Some(reference) = request.characterization.as_ref() else {
            return Err(Error::Unsupported(format!(
                "{id}: a denoise feature cache changes the output, so it may only be selected \
                 alongside a quality-characterization artifact reference. Supply \
                 approximation.characterization, or leave approximation unset for the provider's \
                 exact denoise."
            )));
        };
        let Some(family) = self.characterization.family() else {
            return Err(Error::Unsupported(format!(
                "{id}: characterization reference {} cannot be validated — this provider binds no \
                 quality-characterization artifact family, so its declared approximate mechanisms \
                 are implemented but not yet selectable. Leave approximation unset for the \
                 provider's exact denoise.",
                reference.artifact_id()
            )));
        };
        family.validate(reference, &policy)?;
        Ok(ApproximationPlan::FeatureCache(policy))
    }

    /// Static declaration coherence. An empty result means the surface is internally consistent; it
    /// does **not** mean any measurement exists, and it does **not** mean anything is selectable.
    ///
    /// The defects caught are the ones that would read as a supported mechanism while refusing every
    /// value: an empty interval list, an interval below [`MIN_REUSE_INTERVAL`] (which no
    /// [`CacheReuseInterval`] can ever equal, so it would be permanently unmatchable), and duplicates.
    pub fn declaration_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if let FeatureCacheDomain::Implemented { intervals, .. } = &self.denoise_feature_cache {
            if intervals.is_empty() {
                errors.push(
                    "denoise_feature_cache declares an empty interval list; use Unsupported to \
                     declare the mechanism absent"
                        .to_owned(),
                );
            }
            if intervals
                .iter()
                .any(|interval| *interval < MIN_REUSE_INTERVAL)
            {
                errors.push(format!(
                    "denoise_feature_cache declares an interval below {MIN_REUSE_INTERVAL}, which \
                     no CacheReuseInterval can hold"
                ));
            }
            let mut unique = intervals.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != intervals.len() {
                errors.push("denoise_feature_cache declares duplicate intervals".to_owned());
            }
        }
        errors
    }
}

/// One request's approximate selections, carried on
/// [`GenerationRequest::approximation`](crate::GenerationRequest::approximation).
///
/// A `Default`-able struct of `Option`s, like [`AudioParams`](crate::AudioParams): every field
/// unset is the exact path, and a later mechanism arrives as a further field without breaking
/// `ApproximationRequest { denoise_feature_cache: Some(..), ..Default::default() }` construction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ApproximationRequest {
    /// Select the provider's denoise feature cache with this policy. `None` (the `Default`) is the
    /// provider's exact denoise.
    pub denoise_feature_cache: Option<FeatureCachePolicy>,
    /// The quality-characterization artifact this selection is backed by. Required whenever any
    /// mechanism above is selected — see the module docs — and refused when supplied alone.
    pub characterization: Option<CharacterizationRef>,
}

impl ApproximationRequest {
    /// A request selecting `policy` and nothing else. Refused by every provider today for want of a
    /// characterization reference, which is the point: it is the shape a planner will eventually
    /// send, available now so the refusal is testable.
    pub fn feature_cache(policy: FeatureCachePolicy) -> Self {
        Self {
            denoise_feature_cache: Some(policy),
            characterization: None,
        }
    }

    /// Whether this selects no approximate mechanism (the exact path).
    pub const fn selects_nothing(&self) -> bool {
        self.denoise_feature_cache.is_none()
    }
}

/// What a provider will actually execute for one request — the resolved output of
/// [`ApproximationSurface::resolve`], and the only type a provider's denoise should branch on.
///
/// [`Exact`](Self::Exact) carries **no fields**, deliberately: there is no approximation configured
/// to do nothing, so the off state cannot be a traversal of the approximate code path. See the module
/// docs. `Default` is `Exact`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ApproximationPlan {
    /// The provider's exact denoise, byte-for-byte the pre-sc-18322 path.
    #[default]
    Exact,
    /// Run the denoise feature cache with this policy. Unreachable from a request today; a family's
    /// `#[cfg(test)]` bypass is the only producer.
    FeatureCache(FeatureCachePolicy),
}

impl ApproximationPlan {
    /// Whether this is the exact path. The one branch a provider needs.
    pub const fn is_exact(&self) -> bool {
        matches!(self, Self::Exact)
    }

    /// The feature-cache policy, if this plan selects one.
    pub const fn feature_cache(&self) -> Option<&FeatureCachePolicy> {
        match self {
            Self::Exact => None,
            Self::FeatureCache(policy) => Some(policy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> FeatureCachePolicy {
        FeatureCachePolicy::new(CacheReuseInterval::EVERY_OTHER_STEP)
    }

    fn declaring() -> ApproximationSurface {
        ApproximationSurface::feature_cache(vec![2, 3], 4)
    }

    #[test]
    fn one_is_rejected_as_hard_as_zero_because_off_is_absence() {
        // The type-level half of the byte-identical-when-off guarantee: an interval of 1 is
        // arithmetically the exact path, so admitting it would create a second encoding of "off"
        // that runs THROUGH the cache. Both rejections must be errors, not clamps.
        let zero = CacheReuseInterval::new(0).expect_err("0 is not an interval");
        let one = CacheReuseInterval::new(1).expect_err("1 is the exact path, not an interval");
        assert!(one.to_string().contains("exact path"), "{one}");
        assert!(one.to_string().contains("unset"), "remedy missing: {one}");
        assert!(zero.to_string().contains(">= 2"), "{zero}");
        assert_eq!(
            CacheReuseInterval::new(2).unwrap(),
            CacheReuseInterval::EVERY_OTHER_STEP
        );
        assert_eq!(CacheReuseInterval::new(7).unwrap().steps(), 7);
    }

    #[test]
    fn the_exact_plan_carries_no_configuration() {
        // Not a tautology to assert: it pins that nobody added an `Exact(FeatureCachePolicy)`-shaped
        // "off" that a provider could execute by walking the cache with inert parameters.
        assert!(ApproximationPlan::default().is_exact());
        assert_eq!(ApproximationPlan::default().feature_cache(), None);
        let approximate = ApproximationPlan::FeatureCache(policy());
        assert!(!approximate.is_exact());
        assert_eq!(approximate.feature_cache(), Some(&policy()));
    }

    #[test]
    fn warmup_steps_always_recompute_and_the_interval_counts_from_their_end() {
        let with_warmup = policy().with_warmup(CacheWarmupSteps::new(3));
        let recomputed: Vec<usize> = (0..9)
            .filter(|index| with_warmup.recomputes_step(*index))
            .collect();
        // 0,1,2 are warmup; 3 primes the cache; then every other step.
        assert_eq!(recomputed, vec![0, 1, 2, 3, 5, 7]);

        let no_warmup = policy();
        let recomputed: Vec<usize> = (0..6)
            .filter(|index| no_warmup.recomputes_step(*index))
            .collect();
        // Step 0 always recomputes, so a reuse decision never reads an empty cache.
        assert_eq!(recomputed, vec![0, 2, 4]);
        assert!(
            no_warmup.recomputes_step(0),
            "the first step must prime the cache"
        );

        let wide = FeatureCachePolicy::new(CacheReuseInterval::new(3).unwrap());
        let recomputed: Vec<usize> = (0..7).filter(|i| wide.recomputes_step(*i)).collect();
        assert_eq!(recomputed, vec![0, 3, 6]);
    }

    #[test]
    fn an_inert_surface_refuses_a_selection_for_the_absent_mechanism() {
        let surface = ApproximationSurface::default();
        assert!(surface.is_inert());
        assert!(!surface.is_selectable());

        let error = surface
            .resolve("prov", Some(&ApproximationRequest::feature_cache(policy())))
            .expect_err("an inert surface must refuse a feature cache");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let text = error.to_string();
        assert!(text.contains("prov"), "{text}");
        // The reason must be the ABSENT MECHANISM and not the missing characterization: the two have
        // different fixes (implement/declare the mechanism vs measure it), and reporting the wrong
        // one sends a caller down the wrong path. Pinning the wording also makes a mutation that
        // deletes this branch — leaving the characterization gate to refuse everything by accident —
        // a visible failure rather than a silent behaviour change.
        assert!(
            text.contains("declares no denoise feature cache mechanism"),
            "{text}"
        );
        assert!(text.contains("unset"), "remedy missing: {text}");
    }

    #[test]
    fn a_declared_mechanism_is_refused_for_want_of_a_characterization() {
        let surface = declaring();
        assert!(!surface.is_inert());
        // Declared and implemented, but NOT selectable — P7's shipped state.
        assert!(!surface.is_selectable());
        assert!(surface.declaration_errors().is_empty());

        let error = surface
            .resolve("prov", Some(&ApproximationRequest::feature_cache(policy())))
            .expect_err("a declared mechanism must still be refused until characterized");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let text = error.to_string();
        assert!(
            text.contains("quality-characterization artifact reference"),
            "the refusal must name the missing precondition: {text}"
        );
        assert!(text.contains("unset"), "remedy missing: {text}");
        assert!(
            !text.contains("declares no denoise feature cache mechanism"),
            "a DECLARED mechanism must not be refused as absent: {text}"
        );
    }

    #[test]
    fn a_characterized_selection_is_refused_because_no_family_is_bound() {
        // The structural gate: `CharacterizationBinding::Bound` carries an uninhabited
        // `CharacterizationFamily`, so no surface can ever reach the admitting branch. This test
        // pins that the refusal is the BINDING one — the last gate before admission — rather than
        // one of the earlier gates masking it.
        let surface = declaring();
        assert!(!surface.characterization.is_bound());
        assert_eq!(surface.characterization.family(), None);

        let request = ApproximationRequest {
            denoise_feature_cache: Some(policy()),
            characterization: Some(
                CharacterizationRef::new("wan-t2v-1.3b-cache-2026-08", "sha256:deadbeef").unwrap(),
            ),
        };
        let error = surface
            .resolve("prov", Some(&request))
            .expect_err("no family is bound, so admission is unreachable");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let text = error.to_string();
        assert!(text.contains("binds no"), "{text}");
        assert!(
            text.contains("wan-t2v-1.3b-cache-2026-08"),
            "the refusal must name the reference it could not validate: {text}"
        );
        assert!(
            text.contains("implemented but not yet selectable"),
            "the refusal must distinguish unimplemented from uncharacterized: {text}"
        );
    }

    #[test]
    fn an_out_of_domain_policy_is_refused_with_the_domain_named() {
        let surface = declaring();
        let off_grid = FeatureCachePolicy::new(CacheReuseInterval::new(5).unwrap());
        let text = surface
            .resolve("prov", Some(&ApproximationRequest::feature_cache(off_grid)))
            .expect_err("an undeclared interval must be refused")
            .to_string();
        assert!(text.contains("intervals [2, 3]"), "{text}");

        let over_warmup = policy().with_warmup(CacheWarmupSteps::new(5));
        let text = surface
            .resolve(
                "prov",
                Some(&ApproximationRequest::feature_cache(over_warmup)),
            )
            .expect_err("an over-cap warmup must be refused")
            .to_string();
        assert!(text.contains("warmup <= 4"), "{text}");

        // And the declared corners are NOT refused for the domain — they reach the characterization
        // gate instead, which is a different message. Without this the domain check could reject
        // everything and the test above would still pass.
        for interval in [2, 3] {
            let policy = FeatureCachePolicy::new(CacheReuseInterval::new(interval).unwrap())
                .with_warmup(CacheWarmupSteps::new(4));
            let text = surface
                .resolve("prov", Some(&ApproximationRequest::feature_cache(policy)))
                .expect_err("still uncharacterized")
                .to_string();
            assert!(
                text.contains("quality-characterization artifact reference"),
                "interval {interval} is declared and must pass the domain check: {text}"
            );
        }
    }

    #[test]
    fn an_absent_or_empty_selection_resolves_to_the_exact_plan() {
        // The request-side half of byte-identical-when-off, on both a declaring and an inert surface.
        for surface in [ApproximationSurface::default(), declaring()] {
            assert_eq!(
                surface.resolve("prov", None).expect("absent must resolve"),
                ApproximationPlan::Exact
            );
            let empty = ApproximationRequest::default();
            assert!(empty.selects_nothing());
            assert_eq!(
                surface
                    .resolve("prov", Some(&empty))
                    .expect("an empty selection must resolve"),
                ApproximationPlan::Exact
            );
        }
    }

    #[test]
    fn a_dangling_characterization_reference_is_malformed_not_a_capability_gap() {
        let surface = declaring();
        let dangling = ApproximationRequest {
            denoise_feature_cache: None,
            characterization: Some(CharacterizationRef::new("artifact", "sha256:0").unwrap()),
        };
        let error = surface
            .resolve("prov", Some(&dangling))
            .expect_err("a reference with nothing to characterize must be refused");
        assert!(
            matches!(error, Error::Msg(_)),
            "malformed request, not a capability gap: {error:?}"
        );
        assert!(
            error.to_string().contains("no approximate selection"),
            "{error}"
        );
    }

    #[test]
    fn a_reference_needs_both_halves() {
        assert!(CharacterizationRef::new("", "sha256:0").is_err());
        assert!(CharacterizationRef::new("   ", "sha256:0").is_err());
        let missing_digest =
            CharacterizationRef::new("artifact", " ").expect_err("a digest is required");
        assert!(
            missing_digest.to_string().contains("digest"),
            "{missing_digest}"
        );
        let reference = CharacterizationRef::new("artifact", "sha256:0").unwrap();
        assert_eq!(reference.artifact_id(), "artifact");
        assert_eq!(reference.digest(), "sha256:0");
    }

    #[test]
    fn empty_and_malformed_declarations_are_declaration_errors() {
        let empty = ApproximationSurface {
            denoise_feature_cache: FeatureCacheDomain::Implemented {
                intervals: Vec::new(),
                max_warmup_steps: 0,
            },
            characterization: CharacterizationBinding::Unbound,
        };
        assert!(
            empty
                .declaration_errors()
                .iter()
                .any(|error| error.contains("empty interval list")),
            "{:?}",
            empty.declaration_errors()
        );

        let malformed = ApproximationSurface::feature_cache(vec![1, 2, 2], 0);
        let errors = malformed.declaration_errors();
        assert!(
            errors.iter().any(|error| error.contains("below 2")),
            "{errors:?}"
        );
        assert!(
            errors.iter().any(|error| error.contains("duplicate")),
            "{errors:?}"
        );
    }

    #[test]
    fn an_unsupported_domain_accepts_nothing_and_a_declared_one_accepts_its_corners() {
        assert!(!FeatureCacheDomain::default().accepts(&policy()));
        let domain = FeatureCacheDomain::Implemented {
            intervals: vec![2, 4],
            max_warmup_steps: 3,
        };
        assert!(domain.accepts(&policy()));
        assert!(domain.accepts(&policy().with_warmup(CacheWarmupSteps::new(3))));
        assert!(!domain.accepts(&policy().with_warmup(CacheWarmupSteps::new(4))));
        assert!(
            !domain.accepts(&FeatureCachePolicy::new(
                CacheReuseInterval::new(3).unwrap()
            )),
            "3 is not declared"
        );
        assert!(domain.describe().contains("warmup <= 3"));
    }
}
