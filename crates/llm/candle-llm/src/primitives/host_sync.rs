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

use std::cell::Cell;

thread_local! {
    static HOST_SYNCS: Cell<u64> = const { Cell::new(0) };
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
}
