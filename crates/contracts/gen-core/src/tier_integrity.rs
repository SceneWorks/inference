//! **Tier integrity** (sc-15799) — the one shared rule for what precision a component may be
//! resident at, given the tier the user selected.
//!
//! > **No component is resident above the user's selected quant tier unless a declared, measured
//! > exception says otherwise.**
//!
//! Choosing q8 or q4 *is* a memory decision. If a user picks q4 because bf16 does not fit, carrying
//! any component at bf16 defeats the choice they made. The normative statement of the rule, the
//! catalog's declared exceptions, and the audit of every above-tier residency live in SceneWorks
//! `docs/memory-strategy-contract.md` ("Tier integrity"); this module is the executable half that
//! every provider and the worker must agree on.
//!
//! ## Branch quantization is not a lever — it is how packing works
//!
//! Packing the Krea pose-control branch to the base tier used to be the last rung of a provider-local
//! escalation ladder (the former branch-quantization step, sc-11748): a constrained machine engaged residency
//! flips and decode tiling first, clawing back single-digit GiB, while the branch sat resident at
//! **bf16** the entire time. The size of that unrequested precision is a shape fact, not a
//! measurement: the branch's projections are **3.30 B params ≈ 6.6 GB bf16**, and the packed
//! resident footprint is **~3.3 GB at q8 / ~1.7 GB at q4**
//! (`candle_gen_krea::control::ControlBranch::from_checkpoint_quantized`). So a q8 render that
//! carried a bf16 branch held **~3.3 GB** it never asked for, and a q4 render **~4.9 GB**. That is
//! the invariant above, implemented as an escalation step, and it is deleted. The branch's tier is
//! now a pure function of the base tier — [`control_branch_tier`] — decided once at load with no
//! reference to the device budget. The obsolete provider-local ladder was deleted by sc-15808.
//!
//! Those weight-side deltas are the only branch figures this module will quote. The catalog's
//! `candle.control.branchPackSaveGb` (q8 −8.4, q4 −10.2 GB) are **not** weight-side quantities —
//! each exceeds the entire 6.6 GB branch — so they cannot be subtracted from a peak to price a
//! packed branch. See the retraction on the `krea_2_turbo` catalog entry; sc-16013 owns the
//! re-measure.
//!
//! ## The one declared exception
//!
//! Packing the branch to **q4** is measured *"pose-locked; non-pose details drift"* (candle-gen
//! #483, sc-11743) — a **quality** finding, which is the whole basis of the exception. The control
//! residual is precision-sensitive and is the thing the user actually asked for, so a q4 base
//! **floors** its branch at q8 rather than following it down, forgoing the ~1.6 GB (3.3 → 1.7) a q4
//! branch would have freed. That is a declared, measured exception — not a hole in the rule, the
//! rule working. It is declared in the catalog on the `krea_2_turbo` entry
//! (`candle.control.branchTierByBaseTier` plus its `tierIntegrity` exception row) and enforced here.

use crate::runtime::Quant;

/// Position of a residency precision on the **fidelity ladder**, for answering "is this component
/// resident *above* the selected tier?" — `None` means dense (the provider's default dense dtype,
/// usually bf16) and is the most faithful.
///
/// [`Quant::Nvfp4`] is deliberately ranked **equal to** [`Quant::Q4`] rather than between q4 and q8.
/// It is a distinct numeric regime (E2M1 elements with FP8 block scales, ~4.5 effective
/// bits/weight — sc-11042 Option A), explicitly *not* an int4-affine equivalent, so "more faithful
/// than q4" is not a claim this ladder is entitled to make. Ranking them equal means neither is
/// ever reported as above the other, which is the only honest answer; both are unambiguously below
/// q8 and dense, which is what the invariant actually needs.
pub fn fidelity_rank(tier: Option<Quant>) -> u8 {
    match tier {
        None => 3,
        Some(Quant::Q8) => 2,
        Some(Quant::Q4) | Some(Quant::Nvfp4) => 1,
    }
}

/// Whether a component held at `resident` sits **above** the `selected` tier — the machine-checkable
/// form of the invariant. `true` is a tier-integrity violation unless a declared, measured exception
/// covers it.
pub fn is_above_selected_tier(resident: Option<Quant>, selected: Option<Quant>) -> bool {
    fidelity_rank(resident) > fidelity_rank(selected)
}

/// The tier the Krea pose-control branch is packed to, given the **base** DiT tier (sc-15799).
///
/// A pure function of the base tier: no device budget, no live-memory reading, no escalation. The
/// branch is a resident weight that cannot be re-packed mid-render, so this is decided once at load
/// — but that was never a reason for it to be *budget*-dependent, only for it to be early.
///
/// - dense base (`None`) ⇒ dense branch: it is already at the selected tier.
/// - `Q8` base ⇒ `Q8` branch: the branch follows the tier exactly, so the old q8 rung has no
///   remaining purpose.
/// - `Q4` base ⇒ `Q8` branch: the **declared, measured exception** (see the module docs). One tier
///   above the selection, deliberately, because a q4 control residual measurably drifts.
/// - `Nvfp4` base ⇒ `Q8` branch: no NVFP4 branch artifact exists and NVFP4 sits on the same rung of
///   [`fidelity_rank`] as q4, so it takes the same declared floor.
///
/// Adding a `Quant` variant is a deliberate compile error here: a new tier must decide what its
/// control branch does rather than inherit a wildcard.
pub fn control_branch_tier(base: Option<Quant>) -> Option<Quant> {
    match base {
        None => None,
        Some(Quant::Q8) => Some(Quant::Q8),
        Some(Quant::Q4) | Some(Quant::Nvfp4) => Some(Quant::Q8),
    }
}

/// Whether the control branch's tier for `base` is an above-tier **exception** (i.e. the declared
/// q4/NVFP4 → q8 floor) rather than a plain follow-the-tier packing. Exists so a caller can record
/// *which* case it took in telemetry without re-deriving the rule.
pub fn control_branch_tier_is_exception(base: Option<Quant>) -> bool {
    is_above_selected_tier(control_branch_tier(base), base)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The branch follows the base tier, and the ONLY case where it sits above the selection is the
    /// declared q4 floor. This is the invariant, asserted directly.
    #[test]
    fn branch_never_sits_above_the_base_tier_except_the_declared_q4_floor() {
        for base in [None, Some(Quant::Q8)] {
            assert_eq!(
                control_branch_tier(base),
                base,
                "{base:?} must follow exactly"
            );
            assert!(
                !control_branch_tier_is_exception(base),
                "{base:?} needs no exception"
            );
        }
        for base in [Some(Quant::Q4), Some(Quant::Nvfp4)] {
            assert_eq!(
                control_branch_tier(base),
                Some(Quant::Q8),
                "{base:?} floors its branch at q8"
            );
            assert!(
                control_branch_tier_is_exception(base),
                "{base:?} → q8 IS the declared exception and must report as one"
            );
        }
    }

    /// The branch is never dense (bf16) on a packed base — the defect this story exists to remove.
    /// A mutation that restored the old "keep the branch bf16 unless the budget is tight" behavior
    /// fails here.
    #[test]
    fn a_packed_base_never_carries_a_dense_branch() {
        for base in [Some(Quant::Q4), Some(Quant::Q8), Some(Quant::Nvfp4)] {
            assert!(
                control_branch_tier(base).is_some(),
                "{base:?} must not carry a dense branch"
            );
        }
    }

    /// Above-tier is strict, and NVFP4/q4 are not ordered against each other (neither direction is
    /// a violation, because the ladder cannot honestly rank them).
    #[test]
    fn above_tier_is_strict_and_nvfp4_is_unranked_against_q4() {
        assert!(is_above_selected_tier(None, Some(Quant::Q8)));
        assert!(is_above_selected_tier(Some(Quant::Q8), Some(Quant::Q4)));
        assert!(!is_above_selected_tier(Some(Quant::Q8), Some(Quant::Q8)));
        assert!(!is_above_selected_tier(Some(Quant::Q4), None));
        assert!(!is_above_selected_tier(Some(Quant::Nvfp4), Some(Quant::Q4)));
        assert!(!is_above_selected_tier(Some(Quant::Q4), Some(Quant::Nvfp4)));
    }
}
