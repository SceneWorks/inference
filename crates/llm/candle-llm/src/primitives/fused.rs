//! Fused-primitive policy and telemetry (epic sc-24128, story sc-24137).
//!
//! The decode leaves in [`nn`](super::nn) and [`rope`](super::rope) each have two
//! implementations: candle's op chain (the **reference**, and the only path on CPU / non-CUDA
//! builds) and a single CUDA launch from `candle-quant-kernels::fused_decode` that is bit-identical
//! to it. Which one ran is never silent (epic E2): every call records itself here, per thread, and
//! [`DecodeRecord`](crate::decode::DecodeRecord) carries the per-request delta — how many leaves
//! ran fused, how many ran the reference and, for the latter, the last reason.
//!
//! **Switch.** The fused path is on by default in a `cuda` build; `CANDLE_LLM_FUSED_KERNELS=0`
//! (also `off`, `false`, `no`, `reference`) turns it off for the process, and
//! [`set_fused_kernels`] overrides the environment at runtime (the decode bench and the parity
//! tests flip it to compare the two paths in one process). Off means every leaf runs the reference
//! and reports [`REASON_DISABLED`].
//!
//! **Reasons.** A reference run's reason is a stable lower-case label: [`REASON_DISABLED`],
//! [`REASON_CUDA_FEATURE_OFF`] (built without `cuda`), the kernel's typed refusal
//! (`not_cuda`, `dtype`, `shape`, `rotary_dim`, … from `FusedRefusal::label`) or the cached
//! compile error's label (`nvrtc`, `compute_floor`, …).

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Environment switch: `0` / `off` / `false` / `no` / `reference` disable the fused path.
pub const FUSED_KERNELS_ENV: &str = "CANDLE_LLM_FUSED_KERNELS";

/// Reference-run reason: the switch is off.
pub const REASON_DISABLED: &str = "disabled";
/// Reference-run reason: this build has no `cuda` feature, so no fused path exists.
pub const REASON_CUDA_FEATURE_OFF: &str = "cuda_feature_off";

const POLICY_ENV: u8 = 0;
const POLICY_ON: u8 = 1;
const POLICY_OFF: u8 = 2;

static POLICY: AtomicU8 = AtomicU8::new(POLICY_ENV);

fn env_says_enabled() -> bool {
    static FROM_ENV: OnceLock<bool> = OnceLock::new();
    *FROM_ENV.get_or_init(|| {
        std::env::var(FUSED_KERNELS_ENV)
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "0" | "off" | "false" | "no" | "reference")
            })
            .unwrap_or(true)
    })
}

/// Whether the fused path may be tried at all (the switch; a `cuda` build is still required for
/// it to exist).
pub fn fused_kernels_enabled() -> bool {
    match POLICY.load(Ordering::Relaxed) {
        POLICY_ON => true,
        POLICY_OFF => false,
        _ => env_says_enabled(),
    }
}

/// Override the switch for the process: `Some(true)` / `Some(false)` force it, `None` returns to
/// the environment's setting.
pub fn set_fused_kernels(enabled: Option<bool>) {
    let policy = match enabled {
        Some(true) => POLICY_ON,
        Some(false) => POLICY_OFF,
        None => POLICY_ENV,
    };
    POLICY.store(policy, Ordering::Relaxed);
}

/// Serialises everything that writes or depends on the process-global switch. Test harnesses run
/// tests on parallel threads, so a test that flips the switch must hold this for as long as its
/// assertions depend on the policy.
static POLICY_LOCK: Mutex<()> = Mutex::new(());

/// Holds the process-wide switch lock; restores the switch it found when dropped. Returned by
/// [`fused_policy_guard`].
#[doc(hidden)]
#[must_use = "the policy is only held (and restored) while the guard is alive"]
pub struct FusedPolicyGuard {
    previous: u8,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for FusedPolicyGuard {
    fn drop(&mut self) {
        POLICY.store(self.previous, Ordering::Relaxed);
    }
}

/// Test seam: take the process-wide switch lock, apply `enabled` (as [`set_fused_kernels`]) and
/// hand back a guard that restores the previous switch when dropped. Every test that flips the
/// switch — or asserts a reason the switch could change — holds one, so parallel test threads
/// cannot race on the global. [`set_fused_kernels`] may still be called while it is held.
#[doc(hidden)]
pub fn fused_policy_guard(enabled: Option<bool>) -> FusedPolicyGuard {
    let lock = POLICY_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous = POLICY.load(Ordering::Relaxed);
    set_fused_kernels(enabled);
    FusedPolicyGuard {
        previous,
        _lock: lock,
    }
}

/// Per-thread counts of fused-vs-reference leaf runs (monotone; take deltas with
/// [`FusedTally::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedTally {
    /// Leaves served by a fused kernel launch.
    pub fused: u64,
    /// Leaves served by the reference op chain.
    pub reference: u64,
    /// Why the most recent reference run happened (`None` if none happened).
    pub reference_reason: Option<&'static str>,
}

impl FusedTally {
    /// The counts accumulated since `start` (the reason is the latest one).
    pub fn since(&self, start: &FusedTally) -> FusedTally {
        FusedTally {
            fused: self.fused.wrapping_sub(start.fused),
            reference: self.reference.wrapping_sub(start.reference),
            reference_reason: if self.reference != start.reference {
                self.reference_reason
            } else {
                None
            },
        }
    }

    /// Which path served the leaves: `fused` (only fused launches), `reference` (no fused
    /// launch), `mixed` (both), or `none` (no leaf ran).
    pub fn label(&self) -> &'static str {
        match (self.fused, self.reference) {
            (0, 0) => "none",
            (_, 0) => "fused",
            (0, _) => "reference",
            _ => "mixed",
        }
    }
}

thread_local! {
    static TALLY: Cell<FusedTally> = const { Cell::new(FusedTally { fused: 0, reference: 0, reference_reason: None }) };
}

/// This thread's monotone tally.
pub fn fused_tally() -> FusedTally {
    TALLY.with(Cell::get)
}

/// Record one leaf served by a fused launch.
#[inline]
pub fn note_fused() {
    TALLY.with(|c| {
        let mut t = c.get();
        t.fused = t.fused.wrapping_add(1);
        c.set(t);
    });
}

/// Record one leaf served by the reference chain, with why.
#[inline]
pub fn note_reference(reason: &'static str) {
    TALLY.with(|c| {
        let mut t = c.get();
        t.reference = t.reference.wrapping_add(1);
        t.reference_reason = Some(reason);
        c.set(t);
    });
}

/// Resolve a fused kernel's outcome: `Some(Ok)` when it ran (recorded as fused), `Some(Err)` for
/// a genuine failure inside candle (propagated — it is not an "unsupported input"), `None` when
/// the kernel refused the input or cannot be compiled here (recorded as a reference run with the
/// kernel's reason; the caller then runs the op chain).
#[cfg(feature = "cuda")]
pub(crate) fn outcome<T>(
    result: Result<T, candle_quant_kernels::FusedError>,
) -> Option<crate::error::Result<T>> {
    use candle_quant_kernels::FusedError;
    match result {
        Ok(v) => {
            note_fused();
            Some(Ok(v))
        }
        Err(FusedError::Candle(e)) => Some(Err(e.into())),
        Err(e @ (FusedError::Refused(_) | FusedError::Compile(_))) => {
            note_reference(e.label());
            None
        }
    }
}

/// True when the fused path should be tried (the switch is on); records [`REASON_DISABLED`]
/// as the reference reason when it is not.
#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn try_fused() -> bool {
    if fused_kernels_enabled() {
        true
    } else {
        note_reference(REASON_DISABLED);
        false
    }
}

/// Record that a leaf took the reference path because this build has no fused path at all.
#[cfg(not(feature = "cuda"))]
#[inline]
pub(crate) fn note_not_attempted() {
    note_reference(REASON_CUDA_FEATURE_OFF);
}

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
        let before = fused_policy_guard(None).previous;
        {
            let _held = fused_policy_guard(Some(true));
            assert!(fused_kernels_enabled());
            set_fused_kernels(Some(false));
            assert!(!fused_kernels_enabled());
        }
        let after = fused_policy_guard(None);
        assert_eq!(
            after.previous, before,
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
