//! Fused-versus-reference route tally (epic sc-24432 E3): the MLX primitives that have a fused
//! Metal route — the Prism Hadamard rotation ([`super::prism`]) and the Gated DeltaNet recurrence
//! ([`super::gated_delta`]) — count every run by the route that served it, on this thread, so a
//! request reports `fused` / `reference` / `mixed` / `none` from what actually ran
//! ([`DecodeReport::fused_primitives`](core_llm::DecodeReport::fused_primitives)) — the same
//! shared [`FusedTally`] Candle reports through (E8).
//!
//! A route is decided when the op is built (MLX is lazy), on the thread building the request's
//! graph, which is the thread the provider and the engine run on. The tally is monotone; a request
//! reports [`FusedTally::since`] its start.

use std::cell::Cell;

pub use core_llm::FusedTally;

/// Why a fused route did not run: the op was built for a CPU stream (a custom Metal kernel runs
/// only on the GPU).
pub const REASON_CPU_STREAM: &str = "cpu_stream";
/// Why a fused route did not run: the input's shape or dtype is outside the kernel's (a Prism
/// rotation block that is not a power of two up to the fused maximum, an unsupported dtype, an
/// empty input).
pub const REASON_SHAPE: &str = "shape";
/// Why a fused route did not run: a test forced the reference route (the parity oracle).
pub const REASON_FORCED_REFERENCE: &str = "forced_reference";
/// Why a fused route did not run: its process switch ([`crate::switches`]) — the MLX row of the
/// defaults table, the environment, or a runtime override — turned the kernel off (the label
/// Candle uses for its own switch).
pub const REASON_DISABLED: &str = "disabled";

thread_local! {
    static TALLY: Cell<FusedTally> = const {
        Cell::new(FusedTally { fused: 0, reference: 0, reference_reason: None })
    };
}

/// This thread's monotone tally.
pub fn fused_tally() -> FusedTally {
    TALLY.with(Cell::get)
}

/// Record one run served by a fused kernel.
#[inline]
pub(crate) fn note_fused() {
    TALLY.with(|c| {
        let mut t = c.get();
        t.fused = t.fused.wrapping_add(1);
        c.set(t);
    });
}

/// Record one run served by the reference route, with why.
#[inline]
pub(crate) fn note_reference(reason: &'static str) {
    TALLY.with(|c| {
        let mut t = c.get();
        t.reference = t.reference.wrapping_add(1);
        t.reference_reason = Some(reason);
        c.set(t);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tally_counts_each_route_on_this_thread() {
        let start = fused_tally();
        note_fused();
        note_reference(REASON_CPU_STREAM);
        let since = fused_tally().since(&start);
        assert_eq!((since.fused, since.reference), (1, 1));
        assert_eq!(since.label(), "mixed");
        assert_eq!(since.reference_reason, Some(REASON_CPU_STREAM));
        // Another thread's runs never reach this one's tally.
        std::thread::spawn(note_fused).join().unwrap();
        assert_eq!(fused_tally().since(&start).fused, 1);
    }
}
