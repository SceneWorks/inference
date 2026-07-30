//! Peak **footprint** sampling — the allocator number that predicts a memory kill.
//!
//! # Why `get_peak_memory` is the wrong number to budget against
//!
//! MLX publishes two live counters: [`get_active_memory`](mlx_rs::memory::get_active_memory), the
//! bytes currently held by live arrays, and [`get_cache_memory`](mlx_rs::memory::get_cache_memory),
//! the bytes MLX has freed internally but retained for reuse rather than returning to the OS.
//! `get_peak_memory` is the high-water mark of the **first** only.
//!
//! An operating system that kills processes for using too much memory does not make that
//! distinction. Darwin's `phys_footprint` — the quantity iOS jetsam reads — counts both. So a
//! configuration can report a comfortable peak and still be at the kill threshold, and the gap is
//! not small: a Z-Image 1024² render measured `get_peak_memory` at 3102 MiB and a peak footprint of
//! **6488 MiB**, against a 6136 MiB per-process cap on the target device. It was killed on device
//! four times while the metric said it had 3 GB to spare.
//!
//! Worse for ranking, the two disagree in *order*. Of three configurations measured on the same
//! host, the one with the **lowest** `get_peak_memory` had the **highest** footprint and was the
//! only one the device refused:
//!
//! | configuration | `get_peak_memory` | peak footprint | device |
//! |---|---:|---:|---|
//! | SANA 1024, tile 128 | 3294 MiB | 3749 MiB | ran |
//! | SANA 512, tile 256 | 3453 MiB | 5995 MiB | ran |
//! | Z-Image 1024, tile 256 | **3102 MiB** | **6488 MiB** | killed |
//!
//! # Why sampling, and not a read at the end
//!
//! The cache term moves fast and in the opposite direction to `active` — freeing a large array
//! lowers active and raises cache by the same amount within one allocation. Reading after the work
//! finishes therefore observes a quiet moment, not the peak, and reading only at phase boundaries
//! observes whichever moment the phase happened to end on. A background thread at a fixed interval
//! is the only way to catch an excursion that occurs *between* the points a harness thinks to look —
//! and on a memory-capped device the excursion that matters is by definition the one that ended the
//! process.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

/// A running maximum of `active + cache`, sampled on a background thread.
///
/// Start it before the work being measured and call [`FootprintProbe::finish`] after. See the module
/// docs for why this is the number to budget against rather than `get_peak_memory`.
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
        let (stop, rx) = channel::<()>();
        let peak_bytes = Arc::new(AtomicU64::new(0));
        let handle = {
            let peak = Arc::clone(&peak_bytes);
            let sample = move |peak: &AtomicU64| {
                let footprint = mlx_rs::memory::get_active_memory() as u64
                    + mlx_rs::memory::get_cache_memory() as u64;
                peak.fetch_max(footprint, Ordering::Relaxed);
            };
            std::thread::spawn(move || loop {
                sample(&peak);
                match rx.recv_timeout(interval) {
                    Err(RecvTimeoutError::Timeout) => {}
                    // Disconnected (or signalled): take one FINAL sample after the stop rather than
                    // returning on it. A probe stopped immediately after the allocation it exists to
                    // catch would otherwise miss the very thing it was started for.
                    _ => {
                        sample(&peak);
                        return;
                    }
                }
            })
        };
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
        self.stop_and_join();
        self.peak_bytes.load(Ordering::Relaxed)
    }

    fn stop_and_join(&mut self) {
        drop(self.stop.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for FootprintProbe {
    /// Joins the sampler even when `finish` was never reached, so an error path cannot leave a
    /// thread polling the allocator for the rest of the process.
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe must observe a transient that exists only *between* the points a caller would
    /// think to measure — that is the entire reason it is a thread and not a pair of reads.
    #[test]
    fn observes_an_allocation_that_is_gone_before_finish() {
        let probe = FootprintProbe::start(Duration::from_millis(5));
        {
            // ~64 MiB, held briefly then dropped. `eval` forces materialization; without it the
            // lazy graph would never allocate and the test would pass vacuously.
            let big = mlx_rs::ops::zeros::<f32>(&[16 * 1024 * 1024]).unwrap();
            big.eval().unwrap();
            std::thread::sleep(Duration::from_millis(60));
        }
        mlx_rs::memory::clear_cache();
        let peak = probe.finish();
        assert!(
            peak >= 64 * 1024 * 1024,
            "probe missed a 64 MiB transient that was freed before finish: saw {peak} bytes"
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
        let probe = FootprintProbe::start(Duration::from_secs(3600));
        let big = mlx_rs::ops::zeros::<f32>(&[16 * 1024 * 1024]).unwrap();
        big.eval().unwrap();
        let peak = probe.finish();
        assert!(
            peak >= 64 * 1024 * 1024,
            "final sample was not taken: saw {peak} bytes"
        );
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "finish() waited on the sampling interval instead of interrupting it"
        );
    }
}
