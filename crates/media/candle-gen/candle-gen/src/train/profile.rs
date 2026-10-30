//! Minimal source profiling for the single bounded Krea acceptance invocation.
//! No production setting or preview cadence is changed. A proof-only scope
//! enables three matched six-update segments (18 TOTAL), never a second job.

#[cfg(feature = "model-inference-proof")]
use std::cell::{Cell, RefCell};

#[cfg(feature = "model-inference-proof")]
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static SEGMENT: Cell<usize> = const { Cell::new(0) };
    static ATTRIBUTION: Cell<bool> = const { Cell::new(false) };
    static TIMES: RefCell<Vec<(String,f64)>> = const { RefCell::new(Vec::new()) };
}

/// Proof-only original arithmetic selector; always false in production builds.
pub fn reference_math() -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        ACTIVE.get() && SEGMENT.get() == 0
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        false
    }
}

/// Whether the exact bounded acceptance invocation has opted in.
pub fn active() -> bool {
    #[cfg(feature = "model-inference-proof")]
    {
        ACTIVE.get()
    }
    #[cfg(not(feature = "model-inference-proof"))]
    {
        false
    }
}

/// Opt into the bounded experiment on this synchronous thread. Production
/// callers cannot enable it without the manual proof feature. Nested scopes
/// are rejected, and all thread-local state is restored on errors/panics.
#[cfg(feature = "model-inference-proof")]
pub fn with_bounded_krea_profile<T>(f: impl FnOnce() -> T) -> T {
    assert!(!ACTIVE.replace(true), "nested bounded training profile");
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.set(false);
            SEGMENT.set(0);
            ATTRIBUTION.set(false);
            TIMES.with(|t| t.borrow_mut().clear());
        }
    }
    let _reset = Reset;
    SEGMENT.set(0);
    f()
}

pub(crate) fn begin_step(segment: usize, logical_step: u32) {
    #[cfg(feature = "model-inference-proof")]
    {
        SEGMENT.set(segment);
        ATTRIBUTION.set(logical_step == 1);
    }
    #[cfg(not(feature = "model-inference-proof"))]
    let _ = (segment, logical_step);
}

/// Synchronized wall span, enabled ONLY on the first (excluded warmup) update
/// of each segment. These are inclusive operation spans, not hardware counters
/// or independent kernel time. Steady timings have no per-operation barriers.
pub fn span<T>(
    name: &str,
    device: &candle_core::Device,
    f: impl FnOnce() -> candle_core::Result<T>,
) -> candle_core::Result<T> {
    #[cfg(feature = "model-inference-proof")]
    if active() && ATTRIBUTION.get() {
        device.synchronize()?;
        let start = std::time::Instant::now();
        let result = f()?;
        device.synchronize()?;
        TIMES.with(|t| {
            t.borrow_mut()
                .push((name.to_string(), start.elapsed().as_secs_f64()))
        });
        return Ok(result);
    }
    let _ = (name, device);
    f()
}

pub(crate) fn finish_attribution(segment: usize) {
    #[cfg(feature = "model-inference-proof")]
    if active() && ATTRIBUTION.replace(false) {
        let mut sums = std::collections::BTreeMap::<String, (usize, f64)>::new();
        TIMES.with(|t| {
            for (name, sec) in t.borrow_mut().drain(..) {
                let v = sums.entry(name).or_default();
                v.0 += 1;
                v.1 += sec;
            }
        });
        eprintln!("KREA_PROFILE_ATTRIBUTION segment={segment} synchronized_inclusive_spans={sums:?}; launches/GEMM/norm backward inferred from source; no CUDA hardware counters");
    }
    let _ = segment;
}
