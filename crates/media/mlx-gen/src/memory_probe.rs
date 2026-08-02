//! Peak MLX **allocator footprint** sampling (`active + cache`).
//!
//! # Why `get_peak_memory` is not the whole residency picture
//!
//! MLX publishes two live counters: [`get_active_memory`](mlx_rs::memory::get_active_memory), the
//! bytes currently held by live arrays, and [`get_cache_memory`](mlx_rs::memory::get_cache_memory),
//! the bytes MLX has freed internally but retained for reuse rather than returning to the OS.
//! `get_peak_memory` is the high-water mark of the **first** only.
//!
//! An operating system that kills processes for using too much memory does not make that
//! distinction. Darwin's `phys_footprint` — the quantity iOS jetsam reads — counts both. A Z-Image
//! 1024² render on an iPhone held `active + cache` at a conserved **6068 MiB** across every sample,
//! the cache absorbing exactly what active released, and was killed with 4 MiB of headroom while
//! `get_peak_memory` reported 2901 MiB against a 6136 MiB cap. The peak was not wrong; it was
//! answering a different question than the one jetsam asks.
//!
//! # What this measures, and what it does not predict
//!
//! **This is not a portable process-footprint estimate or an admission verdict.** MLX sizes its cache
//! limit from the host's recommended working set, so on a 64 GB Mac the cache can grow almost without
//! bound. This probe then reports unconstrained allocator retention, not what the same workload will
//! retain in a differently capped process. The host values are also not monotone in obvious workload
//! parameters: a Z-Image 1024² decode measured 16002 MiB at a 512 px tile, 6488 MiB at 256 px, and
//! 43157 MiB at 640 px (reproducible to ±0.2%; different sizes hit different allocator size classes).
//!
//! A useful model for a cache-bounded target is:
//!
//! ```text
//! allocator footprint ≈ peak active + min(cache wanted, configured cache limit)
//! ```
//!
//! Therefore `get_peak_memory` remains the evidence/admission currency: it measures the irreducible
//! live working set. This probe answers a different question: whether cache bounding is required and
//! whether the configured reclaimer actually keeps allocator retention bounded. Calibration
//! harnesses may record it as auxiliary telemetry, but must not write it into
//! `MemoryEvidence::{predicted,observed}_peak_bytes` or compare an unconstrained host sample directly
//! with a target-device cap.
//!
//! The two MLX counters are separate API reads rather than one atomic snapshot. A transfer from
//! active storage into the reuse cache can occur between them, so one sample can transiently over- or
//! under-count. The running maximum is therefore an approximate retention diagnostic, not an exact
//! process measurement; it also excludes every non-MLX allocation in the process.
//!
//! # Why sampling, and not a read at the end
//!
//! The cache term moves fast and in the opposite direction to `active` — freeing a large array
//! lowers active and raises cache by the same amount within one allocation. Reading after the work
//! finishes therefore observes a quiet moment, not the peak, and reading only at phase boundaries
//! observes whichever moment the phase happened to end on. A background thread at a fixed interval
//! is the only way to catch an excursion that occurs *between* the points a harness thinks to look —
//! and on a memory-capped device the excursion that matters can be the one that ended the process.
//!
//! The measurements and kill attribution that motivated this shared probe were produced by the iOS
//! runtime work tracked in Shortcut epic 16774; sc-16784 is the backend-contract half of that work.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

/// A running maximum of MLX `active + cache`, sampled on a background thread.
///
/// Start it before the work being measured and call [`FootprintProbe::finish`] after. See the module
/// docs for the strict distinction between this allocator-local telemetry, process footprint, and
/// the live-allocation value used by memory-ladder evidence.
///
/// ```no_run
/// # use mlx_gen::memory_probe::FootprintProbe;
/// let probe = FootprintProbe::start_default();
/// // ... run a generation ...
/// let peak_bytes = probe.finish();
/// ```
pub struct FootprintProbe {
    /// Dropping this disconnects the channel the sampler waits on, waking it immediately.
    ///
    /// An `AtomicBool` checked around a `thread::sleep` is the obvious shape and is wrong here: the
    /// thread can only observe the flag *between* sleeps, so stopping a probe blocks for up to one
    /// full interval. That is invisible at 50 ms and a hang at any interval chosen to be lazy —
    /// [`FootprintProbe::finish`] joins, so the caller inherits the wait. A disconnect-driven
    /// `recv_timeout` makes the interval an upper bound on *sampling* rather than on *stopping*.
    stop: Option<Sender<()>>,
    peak_bytes: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// The default sampling interval: fast enough to catch a single VAE tile's transient (tens of ms on
/// a host), slow enough that the two atomic reads cost nothing beside the work being measured.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(50);

impl FootprintProbe {
    /// Start sampling at [`DEFAULT_INTERVAL`].
    pub fn start_default() -> Self {
        Self::start(DEFAULT_INTERVAL)
    }

    /// Start sampling at `interval`.
    ///
    /// The first sample is taken immediately, so a probe started and finished around a synchronous
    /// block still observes at least one value.
    pub fn start(interval: Duration) -> Self {
        Self::start_with_sampler(interval, || {
            (mlx_rs::memory::get_active_memory() as u64)
                .saturating_add(mlx_rs::memory::get_cache_memory() as u64)
        })
    }

    fn start_with_sampler<F>(interval: Duration, sample: F) -> Self
    where
        F: Fn() -> u64 + Send + 'static,
    {
        let (stop, rx) = channel::<()>();
        let (ready_tx, ready_rx) = sync_channel::<()>(0);
        let peak_bytes = Arc::new(AtomicU64::new(0));
        let handle = {
            let peak = Arc::clone(&peak_bytes);
            std::thread::spawn(move || {
                peak.fetch_max(sample(), Ordering::Relaxed);
                ready_tx
                    .send(())
                    .expect("footprint probe starter dropped before the first sample");
                loop {
                    match rx.recv_timeout(interval) {
                        Err(RecvTimeoutError::Timeout) => {}
                        // Disconnected (or signalled): take one FINAL sample after the stop rather than
                        // returning on it. A probe stopped immediately after the allocation it exists to
                        // catch would otherwise miss the very thing it was started for.
                        _ => {
                            peak.fetch_max(sample(), Ordering::Relaxed);
                            return;
                        }
                    }
                    peak.fetch_max(sample(), Ordering::Relaxed);
                }
            })
        };
        ready_rx
            .recv()
            .expect("footprint probe sampler panicked before the first sample");
        Self {
            stop: Some(stop),
            peak_bytes,
            handle: Some(handle),
        }
    }

    /// The peak `active + cache` observed so far, in bytes. Safe to call while sampling.
    pub fn peak_footprint_bytes(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Stop sampling and return the peak `active + cache` in bytes.
    pub fn finish(mut self) -> u64 {
        self.stop_and_join()
            .expect("footprint probe sampler panicked");
        self.peak_bytes.load(Ordering::Relaxed)
    }

    fn stop_and_join(&mut self) -> std::thread::Result<()> {
        drop(self.stop.take());
        if let Some(handle) = self.handle.take() {
            handle.join()?;
        }
        Ok(())
    }
}

impl Drop for FootprintProbe {
    /// Joins the sampler even when `finish` was never reached, so an error path cannot leave a
    /// thread polling the allocator for the rest of the process.
    fn drop(&mut self) {
        // `finish` propagates sampler panics. Drop must still join on early-return paths, but cannot
        // safely introduce a second panic while unwinding an unrelated error.
        let _ = self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe must observe a transient that exists only between caller-visible boundaries.
    #[test]
    fn observes_a_controlled_transient_that_is_gone_before_finish() {
        let current = Arc::new(AtomicU64::new(7));
        let sample = Arc::clone(&current);
        let (observed_tx, observed_rx) = sync_channel::<()>(1);
        let probe = FootprintProbe::start_with_sampler(Duration::from_millis(2), move || {
            let value = sample.load(Ordering::Relaxed);
            if value > 7 {
                let _ = observed_tx.try_send(());
            }
            value
        });
        current.store(64 * 1024 * 1024 + 7, Ordering::Relaxed);
        observed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("sampler did not acknowledge the controlled transient");
        current.store(7, Ordering::Relaxed);
        let peak = probe.finish();
        assert_eq!(
            peak,
            64 * 1024 * 1024 + 7,
            "probe missed or distorted the controlled transient"
        );
    }

    /// `finish` must take a final sample at stop time, and must not wait out the interval to do it.
    ///
    /// The hour-long interval is the point. It pins both halves at once: a probe that only sampled
    /// on its own schedule would report whatever it saw at construction and miss the allocation
    /// entirely, and a probe that checked a stop flag *between* sleeps would take an hour to join.
    /// The first version of this module did the latter, and this test hung rather than failing.
    #[test]
    fn final_sample_is_taken_at_stop_without_waiting_out_the_interval() {
        let started = std::time::Instant::now();
        let current = Arc::new(AtomicU64::new(11));
        let sample = Arc::clone(&current);
        let probe = FootprintProbe::start_with_sampler(Duration::from_secs(3600), move || {
            sample.load(Ordering::Relaxed)
        });
        current.store(64 * 1024 * 1024 + 11, Ordering::Relaxed);
        let peak = probe.finish();
        assert_eq!(peak, 64 * 1024 * 1024 + 11, "final sample was not taken");
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "finish() waited on the sampling interval instead of interrupting it"
        );
    }

    #[test]
    fn finish_propagates_sampler_panics() {
        let calls = Arc::new(AtomicU64::new(0));
        let sampler_calls = Arc::clone(&calls);
        let probe = FootprintProbe::start_with_sampler(Duration::from_secs(3600), move || {
            if sampler_calls.fetch_add(1, Ordering::Relaxed) == 0 {
                1
            } else {
                panic!("controlled sampler failure")
            }
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe.finish()));
        assert!(result.is_err(), "finish hid a sampler thread panic");
    }
}
