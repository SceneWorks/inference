//! Request-scoped diagnostics for the cross-family MLX benchmark harness.
//!
//! The production generator contract deliberately stays tensor-neutral and does not expose
//! implementation details such as compiled-handle lifetime or provider-local memoization.  The
//! performance harness still needs those details to distinguish an optimization that really ran
//! from a requested toggle that silently fell back.  This module provides a narrow, opt-in seam:
//!
//! * [`begin_request`] installs a collector for the current synchronous render thread;
//! * hot-path [`record_compile`], [`record_cache`], [`record_fallback`], and [`record_toggle`]
//!   calls are no-ops when no collector is active;
//! * [`DiagnosticScope::finish`] returns stable, aggregated counters for that one request.
//!
//! MLX generation is synchronous and SceneWorks serializes work on the process-default Metal
//! device.  Thread-local ownership therefore gives each request an isolated collector without a
//! process-global lock or a cross-request toggle.  Nested scopes are rejected rather than merged.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::Instant;

/// P1 retained compiled-operation handles.
pub const RETAINED_COMPILATION: &str = "retained_compilation";
/// P3 exact elementwise/normalization/matmul epilogue fusions.
pub const EXACT_EPILOGUES: &str = "exact_epilogues";
/// P4 fused QK-normalization, RoPE/layout, and projection primitives.
pub const FUSED_ATTENTION_PRIMITIVES: &str = "fused_attention_primitives";
/// P5 indexed/scatter tiled-decode accumulation.
pub const INDEXED_DECODE_ACCUMULATOR: &str = "indexed_decode_accumulator";
/// P9 geometry-aware decode admission and policy.
pub const GEOMETRY_AWARE_DECODE: &str = "geometry_aware_decode";
/// Complete independently selectable optimization surface consumed by the P6 matrix.
pub const BENCHMARK_TOGGLES: [&str; 5] = [
    RETAINED_COMPILATION,
    EXACT_EPILOGUES,
    FUSED_ATTENTION_PRIMITIVES,
    INDEXED_DECODE_ACCUMULATOR,
    GEOMETRY_AWARE_DECODE,
];

/// How a compiled operation was obtained for one invocation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CompileDisposition {
    /// A fresh `compile(...)` handle was constructed at the call site and dropped after the call.
    OneShot,
    /// A retained handle did not yet contain a graph for the requested shape/signature.
    RetainedMiss,
    /// A retained handle reused an already-compiled shape/signature.
    RetainedHit,
}

/// Outcome of a provider-local cache lookup.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CacheDisposition {
    Hit,
    Miss,
    Bypass,
}

/// Whether an independently selectable benchmark optimization actually ran.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ToggleDisposition {
    Applied,
    Fallback,
    Unavailable,
}

/// Provider-neutral compute boundaries consumed by the P6 benchmark harness.
///
/// Providers emit these at the exact point where pre-denoise work ends and where denoise hands its
/// final latents to the decoder. They are deliberately separate from [`gen_core::Progress`]: the UI
/// progress contract has historically allowed providers to emit `Step` before or after a solver
/// evaluation, so it cannot also be a comparable timing boundary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BenchmarkPhaseBoundary {
    DenoiseStart,
    DecodeStart,
}

/// One provider-emitted phase boundary measured from the beginning of the diagnostic request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseBoundaryRecord {
    pub boundary: BenchmarkPhaseBoundary,
    pub elapsed_nanos: u64,
}

/// One aggregated diagnostic counter from a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiagnosticCounter {
    Compile {
        site: &'static str,
        disposition: CompileDisposition,
        count: u64,
    },
    Cache {
        site: &'static str,
        disposition: CacheDisposition,
        count: u64,
    },
    Fallback {
        site: &'static str,
        reason: &'static str,
        count: u64,
    },
    Toggle {
        toggle: &'static str,
        disposition: ToggleDisposition,
        count: u64,
    },
}

/// Completed diagnostics for exactly one benchmarked request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticReport {
    pub request_id: String,
    pub family: String,
    pub counters: Vec<DiagnosticCounter>,
    pub phase_boundaries: Vec<PhaseBoundaryRecord>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CounterKey {
    Compile(&'static str, CompileDisposition),
    Cache(&'static str, CacheDisposition),
    Fallback(&'static str, &'static str),
    Toggle(&'static str, ToggleDisposition),
}

struct Collector {
    request_id: String,
    family: String,
    requested_toggles: BTreeSet<&'static str>,
    counters: BTreeMap<CounterKey, u64>,
    started: Instant,
    phase_boundaries: Vec<PhaseBoundaryRecord>,
    phase_observer: Option<Box<dyn FnMut(BenchmarkPhaseBoundary)>>,
}

thread_local! {
    static COLLECTOR: RefCell<Option<Collector>> = const { RefCell::new(None) };
}

/// Returned when a benchmark tries to overlap two diagnostic requests on one render thread.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScopeAlreadyActive;

impl fmt::Display for ScopeAlreadyActive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an MLX diagnostic request is already active on this thread")
    }
}

impl std::error::Error for ScopeAlreadyActive {}

/// RAII owner for one request's diagnostic collector.
#[must_use = "finish the scope to retain its diagnostics; dropping it clears the request collector"]
#[derive(Debug)]
pub struct DiagnosticScope {
    active: bool,
}

/// Begin diagnostics for one synchronous render request.
pub fn begin_request(
    request_id: impl Into<String>,
    family: impl Into<String>,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles(request_id, family, &[])
}

/// Begin diagnostics and expose the requested benchmark toggles to provider code on this render
/// thread. Providers must call [`record_toggle`] after choosing their concrete path; merely finding
/// a name in [`toggle_requested`] is not an activation receipt.
pub fn begin_request_with_toggles(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles_and_phase_observer(request_id, family, requested_toggles, None)
}

/// Begin request diagnostics with a synchronous observer for provider-neutral phase boundaries.
///
/// The observer runs on the generation thread before the provider proceeds into the next phase.
/// P6 uses that stop-the-world seam to stop and join the old phase's memory sampler, reset MLX's
/// native active-memory high-water mark, and start the next sampler without misattributing work.
pub fn begin_request_with_phase_observer(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
    phase_observer: impl FnMut(BenchmarkPhaseBoundary) + 'static,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles_and_phase_observer(
        request_id,
        family,
        requested_toggles,
        Some(Box::new(phase_observer)),
    )
}

fn begin_request_with_toggles_and_phase_observer(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
    phase_observer: Option<Box<dyn FnMut(BenchmarkPhaseBoundary)>>,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    COLLECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(ScopeAlreadyActive);
        }
        *slot = Some(Collector {
            request_id: request_id.into(),
            family: family.into(),
            requested_toggles: requested_toggles.iter().copied().collect(),
            counters: BTreeMap::new(),
            started: Instant::now(),
            phase_boundaries: Vec::new(),
            phase_observer,
        });
        Ok(DiagnosticScope { active: true })
    })
}

/// Whether the active benchmark request selected `toggle`. Always `false` in production requests
/// that did not install a diagnostic scope.
pub fn toggle_requested(toggle: &str) -> bool {
    COLLECTOR.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|collector| collector.requested_toggles.contains(toggle))
    })
}

fn increment(key: CounterKey) {
    COLLECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(collector) = slot.as_mut() {
            *collector.counters.entry(key).or_insert(0) += 1;
        }
    });
}

/// Record one compiled-operation invocation. No-op outside an active request.
pub fn record_compile(site: &'static str, disposition: CompileDisposition) {
    increment(CounterKey::Compile(site, disposition));
}

/// Record one provider-local cache decision. No-op outside an active request.
pub fn record_cache(site: &'static str, disposition: CacheDisposition) {
    increment(CounterKey::Cache(site, disposition));
}

/// Record an explicit fallback and its stable reason. No-op outside an active request.
pub fn record_fallback(site: &'static str, reason: &'static str) {
    increment(CounterKey::Fallback(site, reason));
}

/// Record whether one benchmark toggle actually ran. A runner must not infer this from an
/// environment variable or from output timing alone.
pub fn record_toggle(toggle: &'static str, disposition: ToggleDisposition) {
    increment(CounterKey::Toggle(toggle, disposition));
}

/// Record and synchronously publish one provider-neutral phase boundary. No-op outside an active
/// request, so the production path pays only the request-local diagnostic branch.
pub fn record_phase_boundary(boundary: BenchmarkPhaseBoundary) {
    COLLECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(collector) = slot.as_mut() else {
            return;
        };
        let elapsed_nanos = collector.started.elapsed().as_nanos();
        collector.phase_boundaries.push(PhaseBoundaryRecord {
            boundary,
            elapsed_nanos: u64::try_from(elapsed_nanos).unwrap_or(u64::MAX),
        });
        if let Some(observer) = collector.phase_observer.as_mut() {
            observer(boundary);
        }
    });
}

impl DiagnosticScope {
    /// Finish this request and return its counters in deterministic key order.
    pub fn finish(mut self) -> DiagnosticReport {
        self.active = false;
        COLLECTOR.with(|slot| {
            let collector = slot
                .borrow_mut()
                .take()
                .expect("an active diagnostic scope owns one collector");
            let counters = collector
                .counters
                .into_iter()
                .map(|(key, count)| match key {
                    CounterKey::Compile(site, disposition) => DiagnosticCounter::Compile {
                        site,
                        disposition,
                        count,
                    },
                    CounterKey::Cache(site, disposition) => DiagnosticCounter::Cache {
                        site,
                        disposition,
                        count,
                    },
                    CounterKey::Fallback(site, reason) => DiagnosticCounter::Fallback {
                        site,
                        reason,
                        count,
                    },
                    CounterKey::Toggle(toggle, disposition) => DiagnosticCounter::Toggle {
                        toggle,
                        disposition,
                        count,
                    },
                })
                .collect();
            DiagnosticReport {
                request_id: collector.request_id,
                family: collector.family,
                counters,
                phase_boundaries: collector.phase_boundaries,
            }
        })
    }
}

impl Drop for DiagnosticScope {
    fn drop(&mut self) {
        if self.active {
            COLLECTOR.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_aggregated_and_sorted_per_request() {
        let scope = begin_request("case/0", "qwen_image").unwrap();
        record_compile("z", CompileDisposition::OneShot);
        record_compile("z", CompileDisposition::OneShot);
        record_cache("a", CacheDisposition::Miss);
        record_cache("a", CacheDisposition::Hit);
        record_fallback("decoder", "geometry_not_admitted");
        record_toggle("retained_compilation", ToggleDisposition::Unavailable);
        let report = scope.finish();

        assert_eq!(report.request_id, "case/0");
        assert_eq!(report.family, "qwen_image");
        assert_eq!(report.counters.len(), 5);
        assert!(report.counters.contains(&DiagnosticCounter::Compile {
            site: "z",
            disposition: CompileDisposition::OneShot,
            count: 2,
        }));
        assert!(report.counters.contains(&DiagnosticCounter::Cache {
            site: "a",
            disposition: CacheDisposition::Hit,
            count: 1,
        }));
    }

    #[test]
    fn nested_scope_is_rejected_and_drop_clears_the_slot() {
        let scope = begin_request("outer", "wan").unwrap();
        assert_eq!(
            begin_request("nested", "wan").unwrap_err(),
            ScopeAlreadyActive
        );
        drop(scope);
        let next = begin_request("next", "sdxl").unwrap();
        assert!(next.finish().counters.is_empty());
    }

    #[test]
    fn calls_outside_a_scope_do_not_leak_into_the_next_request() {
        record_compile("outside", CompileDisposition::OneShot);
        record_cache("outside", CacheDisposition::Hit);
        let scope = begin_request("clean", "wan").unwrap();
        assert!(scope.finish().counters.is_empty());
    }

    #[test]
    fn requested_toggles_are_request_local_but_not_implicit_receipts() {
        assert!(!toggle_requested("retained_compilation"));
        let scope = begin_request_with_toggles(
            "toggle",
            "wan",
            &["retained_compilation", "indexed_decode_accumulator"],
        )
        .unwrap();
        assert!(toggle_requested("retained_compilation"));
        assert!(!toggle_requested("geometry_aware_decode"));
        assert!(scope.finish().counters.is_empty());
        assert!(!toggle_requested("retained_compilation"));
    }

    #[test]
    fn phase_boundaries_are_explicit_ordered_and_synchronously_observed() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let observed = Rc::new(RefCell::new(Vec::new()));
        let observer = Rc::clone(&observed);
        let scope = begin_request_with_phase_observer("phase", "qwen_image", &[], move |phase| {
            observer.borrow_mut().push(phase);
        })
        .unwrap();
        record_phase_boundary(BenchmarkPhaseBoundary::DenoiseStart);
        record_phase_boundary(BenchmarkPhaseBoundary::DecodeStart);
        let report = scope.finish();

        assert_eq!(
            *observed.borrow(),
            [
                BenchmarkPhaseBoundary::DenoiseStart,
                BenchmarkPhaseBoundary::DecodeStart,
            ]
        );
        assert_eq!(report.phase_boundaries.len(), 2);
        assert_eq!(
            report.phase_boundaries[0].boundary,
            BenchmarkPhaseBoundary::DenoiseStart
        );
        assert_eq!(
            report.phase_boundaries[1].boundary,
            BenchmarkPhaseBoundary::DecodeStart
        );
        assert!(
            report.phase_boundaries[0].elapsed_nanos <= report.phase_boundaries[1].elapsed_nanos
        );
    }
}
