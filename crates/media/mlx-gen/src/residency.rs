//! MLX adapter for gen-core's backend-neutral request-scoped residency state machine.
//!
//! Ownership, warm-cache transitions, phase order, cancellation, progress, and `StagedHeavy` live in
//! `gen_core::residency`. MLX contributes only its allocator-cache release hook and tensor-specific
//! materialization closures at call sites.

use crate::Error;

/// MLX release behavior for the shared residency driver.
pub struct MlxResidencyRuntime;

impl gen_core::ResidencyRuntime for MlxResidencyRuntime {
    type Error = Error;

    fn after_component_drop() {
        clear_allocator_cache();
    }
}

/// The one shared residency implementation, specialized only by MLX cleanup behavior.
pub type Residency<Text, Heavy> = gen_core::Residency<Text, Heavy, MlxResidencyRuntime>;

pub use gen_core::StagedHeavy;

/// Warn when staged execution over a dense quantized snapshot would re-quantize every request.
pub fn warn_sequential_requantize(model_id: &str, bits: i32) {
    eprintln!(
        "{model_id}: staged residency with Q{bits} over a dense snapshot re-quantizes the whole \
         model on EVERY generate (repeated compute; the dense transient makes the per-phase peak the \
         dense component size, not the packed one — shrinking the memory win). Point at a \
         pre-quantized Q{bits} snapshot to avoid both."
    );
}

#[cfg(test)]
static CLEAR_CACHE_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
fn clear_allocator_cache() {
    CLEAR_CACHE_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(not(test))]
fn clear_allocator_cache() {
    mlx_rs::memory::clear_cache();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancelFlag, Result};
    use std::sync::atomic::Ordering;

    fn reset_clear_cache_calls() {
        CLEAR_CACHE_CALLS.store(0, Ordering::SeqCst);
    }

    #[test]
    fn staged_run_flushes_after_both_component_release_points() {
        reset_clear_cache_calls();
        let residency = Residency::request_scoped(|_| Ok(2u8), |_, _| Ok(3u8));
        let output = residency
            .run_request_scoped(
                true,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text + 1),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
            .unwrap();
        assert_eq!(output, 6);
        assert_eq!(CLEAR_CACHE_CALLS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn warm_to_staged_flushes_eviction_text_and_heavy() {
        reset_clear_cache_calls();
        let residency = Residency::request_scoped(|_| Ok(2u8), |_, _| Ok(3u8));
        let run = |stage| -> Result<u8> {
            residency.run_request_scoped(
                stage,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
        };
        assert_eq!(run(false).unwrap(), 5);
        assert_eq!(run(true).unwrap(), 5);
        assert_eq!(CLEAR_CACHE_CALLS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn staged_error_still_flushes_the_loaded_phase() {
        reset_clear_cache_calls();
        let residency = Residency::<u8, u8>::request_scoped(|_| Ok(2), |_, _| Ok(3));
        let output: Result<()> = residency.run_request_scoped(
            true,
            false,
            &CancelFlag::new(),
            false,
            &mut |_| {},
            |_| -> Result<u8> { Err(Error::Msg("encode failed".into())) },
            |_| Ok(()),
            |_, _, _| Ok(()),
        );
        assert!(matches!(output, Err(Error::Msg(ref message)) if message == "encode failed"));
        assert_eq!(CLEAR_CACHE_CALLS.load(Ordering::SeqCst), 1);
    }
}
