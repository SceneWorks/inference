//! Fused-primitive policy and telemetry (epic sc-24128, story sc-24137).
//!
//! Moved to [`candle_layers::fused`] (`SW-6957` / `SW-F335`) and re-exported here so all internal
//! decoders and external consumers remain 100% compatible.

#[cfg(not(feature = "cuda"))]
pub use candle_layers::fused::note_not_attempted;
pub use candle_layers::fused::{
    fused_kernels_enabled, fused_policy_guard, fused_tally, note_fused, note_reference,
    set_fused_kernels, FusedPolicyGuard, FusedTally, FUSED_KERNELS_ENV, REASON_CUDA_FEATURE_OFF,
    REASON_DISABLED,
};
#[cfg(feature = "cuda")]
pub use candle_layers::fused::{outcome, try_fused};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_deltas_and_labels() {
        let start = fused_tally();
        assert_eq!(fused_tally().since(&start).label(), "none");
        note_fused();
        assert_eq!(fused_tally().since(&start).label(), "fused");
        note_reference("shape");
        let d = fused_tally().since(&start);
        assert_eq!(d.fused, 1);
        assert_eq!(d.reference, 1);
        assert_eq!(d.reference_reason, Some("shape"));
        assert_eq!(d.label(), "mixed");
        let later = fused_tally();
        note_reference("dtype");
        let d = fused_tally().since(&later);
        assert_eq!(
            d,
            FusedTally {
                fused: 0,
                reference: 1,
                reference_reason: Some("dtype")
            }
        );
        assert_eq!(d.label(), "reference");
        // A window with no reference run carries no reason even though the thread has one.
        let now = fused_tally();
        note_fused();
        assert_eq!(fused_tally().since(&now).reference_reason, None);
    }

    #[test]
    fn tally_is_per_thread() {
        let before = fused_tally();
        note_fused();
        let other = std::thread::spawn(|| {
            let start = fused_tally();
            note_reference("x");
            fused_tally().since(&start)
        })
        .join()
        .unwrap();
        assert_eq!(other.reference, 1);
        assert_eq!(fused_tally().since(&before).reference, 0);
    }

    #[test]
    fn runtime_override_beats_the_environment_and_can_be_cleared() {
        let _policy = fused_policy_guard(None);
        let from_env = fused_kernels_enabled();
        set_fused_kernels(Some(false));
        assert!(!fused_kernels_enabled());
        set_fused_kernels(Some(true));
        assert!(fused_kernels_enabled());
        set_fused_kernels(None);
        assert_eq!(fused_kernels_enabled(), from_env);
    }

    #[test]
    fn the_policy_guard_restores_the_switch_it_found() {
        // Every other policy change in this binary happens under a guard that restores, so the
        // value seen at each lock acquisition is the one the previous guard restored.
        let before = fused_policy_guard(None).found();
        {
            let _held = fused_policy_guard(Some(true));
            assert!(fused_kernels_enabled());
            set_fused_kernels(Some(false));
            assert!(!fused_kernels_enabled());
        }
        let after = fused_policy_guard(None);
        assert_eq!(
            after.found(),
            before,
            "dropping the guard restores the switch"
        );
    }

    #[test]
    fn reference_reason_names_the_switch_or_the_build() {
        let _policy = fused_policy_guard(Some(false));
        let start = fused_tally();
        #[cfg(feature = "cuda")]
        {
            assert!(!try_fused());
            let d = fused_tally().since(&start);
            assert_eq!(d.reference, 1);
            assert_eq!(d.reference_reason, Some(REASON_DISABLED));
        }
        #[cfg(not(feature = "cuda"))]
        {
            note_not_attempted();
            let d = fused_tally().since(&start);
            assert_eq!(d.reference, 1);
            assert_eq!(d.reference_reason, Some(REASON_CUDA_FEATURE_OFF));
        }
    }
}
