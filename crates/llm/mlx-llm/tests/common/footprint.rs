//! The process's `phys_footprint` — what macOS charges it, MLX's buffer cache and host heap
//! included — and its **exact** peak since a reset (sc-24446).
//!
//! Sampling the footprint misses any peak shorter than the sampling interval (the 0.5 s memory
//! guard under-read one 81 GB load by 5 GiB). The kernel keeps the true maximum instead:
//! `ri_interval_max_phys_footprint` (`rusage_info_v4`) since the last
//! `proc_reset_footprint_interval`, and `ri_lifetime_max_phys_footprint` over the process.

extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn proc_reset_footprint_interval(pid: i32) -> i32;
}

/// `rusage_info_v4` as `u64` slots: a 16-byte UUID, then the counters.
fn rusage_v4() -> [u64; 64] {
    let mut info = [0u64; 64];
    // SAFETY: `info` outlives the call and is larger than `rusage_info_v4`.
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, info.as_mut_ptr()) };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    info
}

/// The current `ri_phys_footprint`.
pub fn current() -> u64 {
    rusage_v4()[9]
}

/// Restart the interval peak at the current footprint.
pub fn reset_peak() {
    // SAFETY: no pointers; resets this process's own interval counter.
    let rc = unsafe { proc_reset_footprint_interval(std::process::id() as i32) };
    assert_eq!(rc, 0, "proc_reset_footprint_interval failed");
}

/// `ri_interval_max_phys_footprint`: the exact peak since [`reset_peak`].
pub fn peak_since_reset() -> u64 {
    rusage_v4()[35]
}

/// `ri_lifetime_max_phys_footprint`: the exact peak over the process's life.
#[allow(dead_code)] // the guarded probe uses it
pub fn lifetime_peak() -> u64 {
    rusage_v4()[30]
}

/// The exact peak footprint growth over `run`: MLX's buffer cache is cleared first so a
/// previous step's freed buffers are not counted as this one's.
pub fn peak_growth(run: impl FnOnce()) -> u64 {
    mlx_rs::memory::clear_cache();
    settle();
    let before = current();
    reset_peak();
    run();
    peak_since_reset().saturating_sub(before)
}

/// Wait for buffers MLX has already released to leave the footprint. The Metal driver frees them
/// asynchronously — measured at 0.1–1.25 s after `clear_cache` — and a measurement started before
/// then reuses those pages instead of growing, under-reading what the measured work allocates.
pub fn settle() {
    let mut last = current();
    let start = std::time::Instant::now();
    let mut stable_since = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(3) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        let now = current();
        if now < last {
            stable_since = std::time::Instant::now();
        }
        last = now;
        if stable_since.elapsed() > std::time::Duration::from_millis(1500) {
            break;
        }
    }
}
