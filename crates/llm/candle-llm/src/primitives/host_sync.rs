//! Host-synchronization accounting (epic sc-24128, story sc-24129).
//!
//! Every time the decode path pulls a tensor to the host — the sampler's device argmax, the
//! penalized-logits transfer, an MoE router read — the GPU pipeline drains. The fast-decode work
//! (unified speculation, CUDA graphs) is largely about removing those drains, so they are counted
//! where they happen and reported per request through
//! [`DecodeRecord`](crate::decode::DecodeRecord).
//!
//! The counter is **thread-local**: a request decodes on one thread, so the delta between
//! [`host_sync_count`] readings brackets exactly that request's transfers, and concurrent requests
//! on other threads never bleed into each other. It counts host transfers *this crate* issues; a
//! transfer Candle performs internally (e.g. a shape check) is not visible here and is not claimed.

//!
//! Story sc-24133 adds the sampler's own accounting beside it, on the same thread-local basis:
//! how many tokens were drawn on the device versus the host (and why the last host draw happened),
//! and how many **full logits rows** were copied to the host — the transfer the device sampler
//! exists to remove.

use std::cell::Cell;

use core_llm::{HostSampleReason, SamplerPath};

thread_local! {
    static HOST_SYNCS: Cell<u64> = const { Cell::new(0) };
    static LOGITS_TO_HOST: Cell<u64> = const { Cell::new(0) };
    static DEVICE_DRAWS: Cell<u64> = const { Cell::new(0) };
    static HOST_DRAWS: Cell<u64> = const { Cell::new(0) };
    static LAST_HOST_REASON: Cell<Option<HostSampleReason>> = const { Cell::new(None) };
}

/// Record one whole-row logits copy to the host on the current thread (a `to_vec1` of a vocabulary
/// row). It is also a host sync; callers note both.
#[inline]
pub fn note_logits_to_host() {
    LOGITS_TO_HOST.with(|c| c.set(c.get().wrapping_add(1)));
}

/// Record one sampling decision on the current thread: a token drawn on `path`, or (for
/// [`HostSampleReason::SpeculativeDistribution`]) a shaped distribution read on the host.
#[inline]
pub fn note_sampler_path(path: SamplerPath) {
    match path {
        SamplerPath::Device => DEVICE_DRAWS.with(|c| c.set(c.get().wrapping_add(1))),
        SamplerPath::Host(reason) => {
            HOST_DRAWS.with(|c| c.set(c.get().wrapping_add(1)));
            LAST_HOST_REASON.with(|c| c.set(Some(reason)));
        }
    }
}

/// The current thread's monotone sampler counters (take deltas).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerCounters {
    /// Tokens drawn on the device (greedy argmax or the device sampler).
    pub device_draws: u64,
    /// Host sampling decisions (draws, or speculative distribution reads).
    pub host_draws: u64,
    /// Whole logits rows copied to the host.
    pub logits_to_host: u64,
}

/// The current thread's sampler counters since it started.
pub fn sampler_counters() -> SamplerCounters {
    SamplerCounters {
        device_draws: DEVICE_DRAWS.with(Cell::get),
        host_draws: HOST_DRAWS.with(Cell::get),
        logits_to_host: LOGITS_TO_HOST.with(Cell::get),
    }
}

/// The reason of the most recent host sampling decision on this thread, if any.
pub fn last_host_reason() -> Option<HostSampleReason> {
    LAST_HOST_REASON.with(Cell::get)
}

/// Record one device→host transfer on the current thread.
#[inline]
pub fn note_host_sync() {
    HOST_SYNCS.with(|c| c.set(c.get().wrapping_add(1)));
}

/// Host transfers recorded on the current thread since it started (monotone; take deltas).
pub fn host_sync_count() -> u64 {
    HOST_SYNCS.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_on_this_thread_only() {
        let before = host_sync_count();
        note_host_sync();
        note_host_sync();
        assert_eq!(host_sync_count() - before, 2);
        let other = std::thread::spawn(|| {
            let start = host_sync_count();
            note_host_sync();
            host_sync_count() - start
        })
        .join()
        .unwrap();
        assert_eq!(other, 1);
        assert_eq!(
            host_sync_count() - before,
            2,
            "other thread's sync stays there"
        );
    }

    #[test]
    fn sampler_counters_track_paths_and_the_last_host_reason() {
        let before = sampler_counters();
        note_sampler_path(SamplerPath::Device);
        note_sampler_path(SamplerPath::Host(HostSampleReason::Penalty));
        note_logits_to_host();
        let after = sampler_counters();
        assert_eq!(after.device_draws - before.device_draws, 1);
        assert_eq!(after.host_draws - before.host_draws, 1);
        assert_eq!(after.logits_to_host - before.logits_to_host, 1);
        assert_eq!(last_host_reason(), Some(HostSampleReason::Penalty));
    }
}
