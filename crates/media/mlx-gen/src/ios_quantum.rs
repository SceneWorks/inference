//! Bounded execution quanta on a physical iOS device (every model in the iOS catalog).
//!
//! Before constructing each graph, compare architecture-derived tensor scratch plus an explicit
//! engineering reserve with CURRENT OS headroom. Materialize and synchronize each result, then
//! drain reusable buffers before the next quantum. This limits lazy graph/cast overlap; it is NOT
//! an allocator interception or an absolute jetsam guarantee. Driver allocations, OS pressure and
//! a single backend operation can exceed the accounting. Production artifact admission is unchanged.

use crate::{Error, Result};
use mlx_rs::Array;
#[path = "ios_quantum_budget.rs"]
mod budget;
#[path = "ios_quantum_kernels.rs"]
mod kernels;
#[cfg(not(target_os = "ios"))]
pub use budget::measured_peak_bytes;
pub use budget::{
    allocated_bytes, allocated_tensor_bytes, assert_tracks_measured, attention_block_bytes,
    check_headroom, conv_stage_bytes, elementwise_bytes, rope_table_bytes, sampler_step_bytes,
    scores_bytes, tensor_bytes, AttentionBlockGeometry, ConvStageGeometry, RatioReport, Scores,
    PROCESS_RESERVE_BYTES,
};
pub use kernels::{
    bytes, conv_channels, conv_sum, decoded_to_image_bytes, decoded_to_image_bytes_from_metadata,
    downsample_stage_bytes, downsample_tile_bytes, downsample_tile_bytes_from_metadata,
    dtype_bytes, element_bytes, elements, group_norm_buffers, norm_conv_bytes,
    norm_promotion_units, page_slack, resnet_bytes, resnet_bytes_from_metadata, sdpa_scores,
    single_conv_bytes, tile_bytes, tile_bytes_from_metadata, tiled_stage_bytes,
    tiled_stage_f32_accum, tiled_stage_f32_accum_from_metadata, vae_attention_bytes, Backend,
    ConvCall, ConvOperandMetadata, TileSource, GROUP_NORM_BUFFERS, PAGE_BYTES,
    VAE_ATTENTION_SCORE_ELEMENTS,
};

/// A typed record of the quantum that refused BEFORE constructing its graph. Error values cross
/// provider/contract seams that may stringify them, so the refusal is ALSO recorded here, where a
/// device capture reads it without parsing any message. Capacity evidence, never a cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantumRefusal {
    pub label: String,
    pub scratch_bytes: u64,
    pub reserve_bytes: u64,
    /// Live `os_proc_available_memory` at the check; `None` when the OS reported none.
    pub available_bytes: Option<u64>,
}

static LAST_REFUSAL: std::sync::Mutex<Option<QuantumRefusal>> = std::sync::Mutex::new(None);

fn record_refusal(label: &str, scratch: u64, reserve: u64, available: Option<u64>) {
    if let Ok(mut slot) = LAST_REFUSAL.lock() {
        *slot = Some(QuantumRefusal {
            label: label.to_string(),
            scratch_bytes: scratch,
            reserve_bytes: reserve,
            available_bytes: available,
        });
    }
}

/// Take (and clear) the most recent pre-construction quantum refusal in this process.
pub fn take_refusal() -> Option<QuantumRefusal> {
    LAST_REFUSAL.lock().ok().and_then(|mut slot| slot.take())
}

/// One observed quantum boundary in a CPU diagnostic fixture.
#[cfg(not(target_os = "ios"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantumEvent {
    /// A pre-construction headroom check passed for `scratch` bytes.
    Checked { label: String, scratch: u64 },
    /// A completed quantum synchronized MLX stream index `stream`.
    Synchronized { label: String, stream: i32 },
}

#[cfg(not(target_os = "ios"))]
type QuantumObserver = Box<dyn FnMut(&QuantumEvent)>;

#[cfg(not(target_os = "ios"))]
thread_local! {
    static CPU_FIXTURE_HEADROOM: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static CPU_FIXTURE_OBSERVER: std::cell::RefCell<Option<QuantumObserver>> = const { std::cell::RefCell::new(None) };
    static CPU_FIXTURE_SHORT_AT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Inside a CPU diagnostic fixture, make the FIRST check of `label` see live headroom one byte short
/// of its own price plus the reserve (every other check sees the fixture's headroom), so a test can
/// refuse any one quantum of a path — not only the first one whose price exceeds every earlier one.
/// Unavailable on iOS.
#[cfg(not(target_os = "ios"))]
pub fn with_cpu_diagnostic_short_at<T>(label: &str, body: impl FnOnce() -> Result<T>) -> Result<T> {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            CPU_FIXTURE_SHORT_AT.with(|s| *s.borrow_mut() = previous);
        }
    }
    let previous = CPU_FIXTURE_SHORT_AT.with(|s| s.borrow_mut().replace(label.to_string()));
    let _restore = Restore(previous);
    body()
}

/// The live headroom the check of `label` at `scratch` sees.
fn available_for(_label: &str, _scratch: u64) -> Option<u64> {
    #[cfg(not(target_os = "ios"))]
    {
        let short = CPU_FIXTURE_SHORT_AT.with(|s| {
            let mut slot = s.borrow_mut();
            if slot.as_deref() == Some(_label) {
                *slot = None;
                true
            } else {
                false
            }
        });
        if short {
            return Some(
                _scratch
                    .saturating_add(PROCESS_RESERVE_BYTES)
                    .saturating_sub(1),
            );
        }
    }
    available_bytes()
}

/// Observe checks/synchronizations of a CPU diagnostic fixture. Unavailable on iOS.
#[cfg(not(target_os = "ios"))]
pub fn with_cpu_diagnostic_observer<T>(
    observer: impl FnMut(&QuantumEvent) + 'static,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    struct Restore(Option<QuantumObserver>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            CPU_FIXTURE_OBSERVER.with(|s| *s.borrow_mut() = previous);
        }
    }
    let previous = CPU_FIXTURE_OBSERVER.with(|s| s.borrow_mut().replace(Box::new(observer)));
    let _restore = Restore(previous);
    body()
}

fn notify(_event: impl FnOnce() -> QuantumEventArg) {
    #[cfg(not(target_os = "ios"))]
    {
        let Some(mut observer) = CPU_FIXTURE_OBSERVER.with(|s| s.borrow_mut().take()) else {
            return;
        };
        let event = match _event() {
            QuantumEventArg::Checked(label, scratch) => QuantumEvent::Checked { label, scratch },
            QuantumEventArg::Synchronized(label, stream) => {
                QuantumEvent::Synchronized { label, stream }
            }
        };
        observer(&event);
        CPU_FIXTURE_OBSERVER.with(|s| {
            let mut slot = s.borrow_mut();
            if slot.is_none() {
                *slot = Some(observer);
            }
        });
    }
}

// Fields are only read by the host CPU-fixture observer; on iOS `notify` discards them.
#[cfg_attr(target_os = "ios", allow(dead_code))]
enum QuantumEventArg {
    Checked(String, u64),
    Synchronized(String, i32),
}

/// Exercise the exact physical-iOS barriers on tiny CPU models, with deterministic headroom.
/// Unavailable on iOS: a diagnostic fixture cannot replace a phone's real OS reading.
#[cfg(not(target_os = "ios"))]
pub fn with_cpu_diagnostic_headroom<T>(bytes: u64, body: impl FnOnce() -> Result<T>) -> Result<T> {
    if mlx_rs::Device::try_default()? != mlx_rs::Device::cpu() {
        return Err(Error::Unsupported(
            "quantum fixture requires an explicit CPU device".into(),
        ));
    }
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CPU_FIXTURE_HEADROOM.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(CPU_FIXTURE_HEADROOM.with(|s| s.replace(Some(bytes))));
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), body)
}

pub fn enabled() -> bool {
    #[cfg(not(target_os = "ios"))]
    if CPU_FIXTURE_HEADROOM.with(|s| s.get().is_some()) {
        return true;
    }
    cfg!(all(target_os = "ios", not(target_abi = "sim")))
}

/// Live remaining capacity, never interpreted as a measured total cap.
pub fn available_bytes() -> Option<u64> {
    #[cfg(not(target_os = "ios"))]
    if let Some(bytes) = CPU_FIXTURE_HEADROOM.with(|s| s.get()) {
        return Some(bytes);
    }
    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    {
        unsafe extern "C" {
            fn os_proc_available_memory() -> usize;
        }
        let bytes = unsafe { os_proc_available_memory() } as u64;
        (bytes > 0).then_some(bytes)
    }
    #[cfg(not(all(target_os = "ios", not(target_abi = "sim"))))]
    {
        None
    }
}

/// Actual Darwin VM page operand used by the pinned Metal allocator. Metadata only, never
/// proof that an existing backing was fresh or an absolute process-footprint certificate.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn native_metal_page_bytes() -> Option<u64> {
    unsafe extern "C" {
        static vm_page_size: usize;
    }
    let page = unsafe { vm_page_size } as u64;
    page.is_power_of_two().then_some(page)
}
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn native_metal_page_bytes() -> Option<u64> {
    None
}

pub fn fresh_metal_allocation_bytes(request_bytes: u64, page_bytes: u64) -> Option<u64> {
    if !page_bytes.is_power_of_two() {
        return None;
    }
    if request_bytes <= page_bytes {
        return Some(request_bytes);
    }
    request_bytes
        .checked_add(page_bytes - 1)?
        .checked_div(page_bytes)?
        .checked_mul(page_bytes)
}

// Private native-call seam only: production always uses pinned C bindings, never a provider hook.
#[derive(Clone, Copy)]
struct ScopeControls {
    set_limit: unsafe extern "C" fn(*mut usize, usize) -> i32,
    get_cached: unsafe extern "C" fn(*mut usize) -> i32,
    clear: unsafe extern "C" fn() -> i32,
    drain: fn() -> usize,
}
impl ScopeControls {
    const NATIVE: Self = Self {
        set_limit: mlx_sys::mlx_set_cache_limit,
        get_cached: mlx_sys::mlx_get_cache_memory,
        clear: mlx_sys::mlx_clear_cache,
        drain: crate::residency::drain_allocator_cache,
    };
    fn cached(self) -> Option<usize> {
        let mut bytes = 0;
        (unsafe { (self.get_cached)(&mut bytes) } == 0).then_some(bytes)
    }
    fn drain(self) -> Result<()> {
        (self.drain)();
        let status = unsafe { (self.clear)() };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Msg(format!(
                "native quantum cache clear failed with status {status}"
            )))
        }
    }
}
#[derive(Clone, Copy)]
struct ScopeRestore {
    previous: usize,
    controls: ScopeControls,
}

fn diagnostic_allocation_scope(
    requested_limit: usize,
    previous_return: Option<usize>,
    cached_bytes: Option<usize>,
    restoring: bool,
) {
    let Some(page_bytes) = native_metal_page_bytes() else {
        return;
    };
    if let (Some(previous), Some(cached)) = (previous_return, cached_bytes) {
        diagnostic_metadata(
            "mlx",
            "allocation-scope",
            &[
                ("schema", 1),
                ("page_bytes", page_bytes),
                ("cache_limit_requested_bytes", requested_limit as u64),
                ("setter_succeeded", 1),
                ("cache_query_succeeded", 1),
                ("setter_previous_bytes", previous as u64),
                ("cache_bytes", cached as u64),
                ("restoring", u64::from(restoring)),
            ],
        );
    } else {
        diagnostic_metadata(
            "mlx",
            "allocation-scope",
            &[
                ("schema", 1),
                ("page_bytes", page_bytes),
                ("cache_limit_requested_bytes", requested_limit as u64),
                ("setter_succeeded", u64::from(previous_return.is_some())),
                ("cache_query_succeeded", u64::from(cached_bytes.is_some())),
                ("restoring", u64::from(restoring)),
            ],
        );
    }
}

/// Cache lifetime matches the operation, including errors and early returns. Failed native
/// controls never become an inactive scope or fabricated previous/cache values.
pub struct Scope(Option<ScopeRestore>);
impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}
impl Scope {
    pub fn new() -> Self {
        Self::try_new().expect("native quantum scope setup failed")
    }
    pub fn try_new() -> Result<Self> {
        if enabled() {
            Self::try_new_enabled(ScopeControls::NATIVE)
        } else {
            Ok(Self(None))
        }
    }
    fn try_new_enabled(controls: ScopeControls) -> Result<Self> {
        let mut previous = 0;
        let status = unsafe { (controls.set_limit)(&mut previous, 0) };
        if status != 0 {
            diagnostic_allocation_scope(0, None, controls.cached(), false);
            return Err(Error::Msg(format!(
                "native quantum cache-zero setter failed with status {status}"
            )));
        }
        let scope = Self(Some(ScopeRestore { previous, controls }));
        controls.drain()?;
        let cached = controls.cached();
        diagnostic_allocation_scope(0, Some(previous), cached, false);
        if cached != Some(0) {
            return Err(Error::Msg(format!(
                "native quantum cache was not verifiably drained: {cached:?}"
            )));
        }
        Ok(scope)
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(ScopeRestore { previous, controls }) = self.0.take() {
            if let Err(error) = controls.drain() {
                eprintln!("MLX quantum cache cleanup failed: {error}");
            }
            let mut replaced = 0;
            let status = unsafe { (controls.set_limit)(&mut replaced, previous) };
            diagnostic_allocation_scope(
                previous,
                (status == 0).then_some(replaced),
                controls.cached(),
                true,
            );
            if status != 0 {
                eprintln!("MLX quantum cache policy restoration failed (native status {status}, requested limit {previous})");
            }
        }
    }
}

#[cfg(test)]
mod allocation_scope_tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    // Synthetic native-control faults exercise production setup/RAII, not backing physics.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fault {
        None,
        Setter,
        Query,
        Clear,
        Nonempty,
        Restore,
    }
    struct State {
        fault: Fault,
        limit: usize,
        cached: usize,
        setters: usize,
        restores: usize,
    }
    const PREVIOUS: usize = 123;
    thread_local! {
        static STATE: RefCell<State> = const { RefCell::new(State { fault: Fault::None, limit: PREVIOUS, cached: 7, setters: 0, restores: 0 }) };
    }
    fn reset(fault: Fault) {
        STATE.with(|state| {
            *state.borrow_mut() = State {
                fault,
                limit: PREVIOUS,
                cached: if fault == Fault::Clear { 0 } else { 7 },
                setters: 0,
                restores: 0,
            }
        });
    }
    unsafe extern "C" fn setter(previous: *mut usize, requested: usize) -> i32 {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.setters += 1;
            if requested != 0 {
                state.restores += 1;
            }
            if state.fault == Fault::Setter || (state.fault == Fault::Restore && requested != 0) {
                return 7;
            }
            unsafe {
                *previous = state.limit;
            }
            state.limit = requested;
            0
        })
    }
    unsafe extern "C" fn query(bytes: *mut usize) -> i32 {
        STATE.with(|state| {
            let state = state.borrow();
            if state.fault == Fault::Query {
                return 9;
            }
            unsafe {
                *bytes = state.cached;
            }
            0
        })
    }
    unsafe extern "C" fn clear() -> i32 {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if state.fault == Fault::Clear {
                return 11;
            }
            if state.fault != Fault::Nonempty {
                state.cached = 0;
            }
            0
        })
    }
    fn drain() -> usize {
        0
    }
    const CONTROLS: ScopeControls = ScopeControls {
        set_limit: setter,
        get_cached: query,
        clear,
        drain,
    };
    fn assert_closed_before_work(fault: Fault, expected_restores: usize) {
        reset(fault);
        let work = Cell::new(0);
        let result: Result<()> = (|| {
            let _scope = Scope::try_new_enabled(CONTROLS)?;
            work.set(work.get() + 1);
            Ok(())
        })();
        assert!(
            result.is_err(),
            "native control fault must return typed setup error"
        );
        assert_eq!(work.get(), 0, "no work after failed setup");
        STATE.with(|state| {
            let state = state.borrow();
            assert_eq!(
                state.limit, PREVIOUS,
                "restore known previous, not initialized zero"
            );
            assert_eq!(state.restores, expected_restores);
            if fault == Fault::Setter {
                assert_eq!(state.setters, 1);
            }
        });
    }
    #[test]
    fn setter_failure_is_closed_before_work_without_fake_previous_zero() {
        assert_closed_before_work(Fault::Setter, 0);
    }
    #[test]
    fn query_failure_is_closed_before_work_and_restores_previous() {
        assert_closed_before_work(Fault::Query, 1);
    }
    #[test]
    fn clear_failure_is_closed_before_work_and_restores_previous() {
        assert_closed_before_work(Fault::Clear, 1);
    }
    #[test]
    fn nonempty_cache_is_closed_before_work_and_restores_previous() {
        assert_closed_before_work(Fault::Nonempty, 1);
    }
    #[test]
    fn successful_scope_restores_the_actual_previous_limit() {
        reset(Fault::None);
        let scope = Scope::try_new_enabled(CONTROLS).unwrap();
        STATE.with(|state| assert_eq!(state.borrow().limit, 0));
        drop(scope);
        STATE.with(|state| {
            assert_eq!(state.borrow().limit, PREVIOUS);
            assert_eq!(state.borrow().restores, 1);
        });
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn restore_failure_is_honest_and_does_not_panic_during_unwind() {
        reset(Fault::Restore);
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        assert!(!set_quantum_listener(Some(Box::new(move |mark| {
            if let QuantumMark::Metadata {
                provider: "mlx",
                kind: "allocation-scope",
                fields,
            } = mark
            {
                sink.lock().unwrap().push(
                    fields
                        .iter()
                        .map(|&(key, value)| (key.to_owned(), value))
                        .collect::<std::collections::BTreeMap<_, _>>(),
                );
            }
        }))));
        struct Remove;
        impl Drop for Remove {
            fn drop(&mut self) {
                set_quantum_listener(None);
            }
        }
        let _remove = Remove;
        const BODY_PANIC: &str = "synthetic body failure while scope lives";
        let result = std::panic::catch_unwind(|| {
            let _scope = Scope::try_new_enabled(CONTROLS).unwrap();
            std::panic::panic_any(BODY_PANIC);
        });
        assert_eq!(
            result
                .expect_err("body panic preserved")
                .downcast_ref::<&str>()
                .copied(),
            Some(BODY_PANIC)
        );
        let seen = seen.lock().unwrap();
        let restore = seen.last().expect("actual restore metadata");
        assert_eq!(restore.get("restoring"), Some(&1));
        assert_eq!(restore.get("setter_succeeded"), Some(&0));
        assert!(
            !restore.contains_key("setter_previous_bytes") && !restore.contains_key("cache_bytes")
        );
        STATE.with(|state| {
            assert_eq!(state.borrow().limit, 0);
            assert_eq!(state.borrow().restores, 1);
        });
    }
    #[test]
    fn fresh_requests_use_the_supplied_page_boundary_not_a_machine_golden() {
        for page in [4096u64, 16384] {
            for (request, expected) in [
                (0, 0),
                (page - 1, page - 1),
                (page, page),
                (page + 1, 2 * page),
                (2 * page, 2 * page),
                (2 * page + 1, 3 * page),
            ] {
                assert_eq!(fresh_metal_allocation_bytes(request, page), Some(expected));
            }
        }
    }
    #[test]
    fn fresh_requests_reject_invalid_basis_and_overflow() {
        for page in [0u64, 3, u64::MAX] {
            assert_eq!(fresh_metal_allocation_bytes(1, page), None);
        }
        for page in [4096u64, 16384] {
            assert_eq!(fresh_metal_allocation_bytes(u64::MAX, page), None);
        }
    }
    #[test]
    fn fresh_requests_round_unique_allocations_before_summing() {
        for page in [4096u64, 16384] {
            let split = fresh_metal_allocation_bytes(page + 1, page).unwrap()
                + fresh_metal_allocation_bytes(page + 1, page).unwrap();
            let merged = fresh_metal_allocation_bytes(2 * page + 2, page).unwrap();
            assert_eq!(split, 4 * page);
            assert_eq!(merged, 3 * page);
            assert!(split > merged);
        }
    }
}

// ---- Production quantum marks and the admitted measured envelope (epic 16774 anchor law) ----

/// A quantum boundary as production code sees it. A device capture registers a listener to read
/// its kernel footprint at every boundary and attribute the lifetime peak to the phase each label
/// runs in (`ios_pricing::IosPhase`, via the provider's `ios_quantum_phase`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumMark<'a> {
    /// A pre-construction headroom check passed: the quantum `label` is about to run.
    Checked { label: &'a str, scratch: u64 },
    /// The quantum `label` completed and its stream was synchronized.
    Synchronized { label: &'a str },
    /// Opt-in actual execution metadata. Keys/tags are provider literals and scalar fields are
    /// borrowed only for the callback; never tensor references, clones, values or user text.
    Metadata {
        provider: &'a str,
        kind: &'a str,
        fields: &'a [(&'a str, u64)],
    },
}

/// Send bounded, stack-backed actual execution facts through the existing diagnostic listener.
/// Default is no-op; metadata never changes admission, scheduling or materialization.
pub fn diagnostic_metadata<'a>(provider: &'a str, kind: &'a str, fields: &'a [(&'a str, u64)]) {
    mark(QuantumMark::Metadata {
        provider,
        kind,
        fields,
    });
}

type Listener = Box<dyn Fn(QuantumMark<'_>) + Send + Sync>;
static LISTENER: std::sync::RwLock<Option<Listener>> = std::sync::RwLock::new(None);

/// Install (or with `None`, remove) the process-wide quantum listener. Returns whether one was
/// installed before. Only a device capture installs one; it never changes a decision.
pub fn set_quantum_listener(listener: Option<Listener>) -> bool {
    match LISTENER.write() {
        Ok(mut slot) => std::mem::replace(&mut *slot, listener).is_some(),
        Err(_) => false,
    }
}

fn mark(m: QuantumMark<'_>) {
    if let Ok(slot) = LISTENER.read() {
        if let Some(listener) = slot.as_ref() {
            listener(m);
        }
    }
}

/// The measured envelope an ADMITTED request runs under: the admission's demand (the anchor-derived
/// growth over the starting footprint) plus its stated margin, and the live headroom when it began.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Envelope {
    start_available: u64,
    budget: u64,
}

/// Diagnostic DEVICE observation only; this record grants no runtime calibration/discount.
/// One quantum formula's captured counts: the guard's priced scratch at
/// that invocation and the kernel-footprint growth the device observed across it (the lifetime
/// peak at the quantum's end minus the footprint at its check — exact: the peak rose inside it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantumCalibration {
    pub priced_bytes: u64,
    pub observed_bytes: u64,
}

static ENVELOPE: std::sync::Mutex<Option<Envelope>> = std::sync::Mutex::new(None);

/// An admitted, MEASURED envelope of process growth, open while this guard lives.
///
/// The envelope is the admission's demand (the anchor-measured or anchor-derived growth over the
/// footprint at admission) plus its stated margin, anchored at the live headroom when it opened —
/// i.e. it spans the admitted work's loads and every render the loaded model then serves. While the
/// process's growth since the envelope opened stays within it, a quantum's guard requires only the
/// envelope's REMAINING bytes of live headroom when that is less than its formula price (scratch +
/// [`PROCESS_RESERVE_BYTES`]): the measured device anchor, not the counted formula's conservative
/// scratch, bounds an admitted request. Once growth exceeds the envelope (the measurement no longer
/// describes this run) every guard is back on its formula — the runtime safety net. Without a live
/// headroom reading the envelope changes nothing. While one envelope is open another open is a
/// no-op (the outer envelope keeps bounding; past it, the formula does).
#[derive(Debug)]
pub struct AdmittedEnvelope {
    opened: bool,
}

impl Drop for AdmittedEnvelope {
    fn drop(&mut self) {
        if self.opened {
            if let Ok(mut slot) = ENVELOPE.lock() {
                *slot = None;
            }
        }
    }
}

/// Open an [`AdmittedEnvelope`] of the COMPLETE admitted bound (margin already included).
/// Legacy label/size observations are accepted for call compatibility but grant NO discount:
/// matching labels and smaller logical uppers do not prove transferable physical ratios.
pub fn open_admitted_envelope(
    budget_bytes: u64,
    _calibration: std::collections::BTreeMap<String, QuantumCalibration>,
    _margin_bytes: u64,
) -> AdmittedEnvelope {
    let opened = match (ENVELOPE.lock(), available_bytes()) {
        (Ok(mut slot), Some(start_available)) if slot.is_none() => {
            *slot = Some(Envelope {
                start_available,
                budget: budget_bytes,
            });
            true
        }
        _ => false,
    };
    AdmittedEnvelope { opened }
}

/// Run `body` inside an [`AdmittedEnvelope`] of `budget_bytes` with no per-quantum calibration.
pub fn with_admitted_envelope<T>(budget_bytes: u64, body: impl FnOnce() -> T) -> T {
    let _envelope = open_admitted_envelope(budget_bytes, Default::default(), 0);
    body()
}

/// Outside an admitted envelope, or past its validated complete bound: the raw guarded formula
/// plus engineering reserve. Inside it, only the complete envelope's remaining demand can be used
/// instead. Per-label observed/priced ratios are NEVER transferable proof (raw100/physical80 vs
/// raw50/physical45 is already a counterexample), even at or below the observed logical size.
fn requirement(_label: &str, available: Option<u64>, scratch: u64) -> (u64, u64) {
    let formula = (scratch, PROCESS_RESERVE_BYTES);
    let formula_total = scratch.saturating_add(PROCESS_RESERVE_BYTES);
    let (Some(available), Ok(slot)) = (available, ENVELOPE.lock()) else {
        return formula;
    };
    let Some(envelope) = slot.as_ref() else {
        return formula;
    };
    let consumed = envelope.start_available.saturating_sub(available);
    let Some(remaining) = envelope.budget.checked_sub(consumed) else {
        return formula;
    };
    if remaining < formula_total {
        (remaining, 0)
    } else {
        formula
    }
}

pub fn before(label: &str, scratch: u64) -> Result<()> {
    if enabled() {
        let available = available_for(label, scratch);
        let (required, reserve) = requirement(label, available, scratch);
        check_headroom(available, required, reserve).map_err(|e| {
            record_refusal(label, required, reserve, available);
            Error::Unsupported(format!("{label}: {e}; scratch {required} bytes + engineering reserve {reserve} bytes (not a process bound)"))
        })?;
        notify(|| QuantumEventArg::Checked(label.to_string(), scratch));
        mark(QuantumMark::Checked { label, scratch });
    }
    Ok(())
}

/// Explicit C API synchronization is needed in addition to eval: completed graph values must not
/// leave earlier command-buffer resources live across the next headroom check. Synchronize the
/// stream this quantum executed on: a caller-installed task-local stream, else the default.
fn synchronize(label: &str) -> Result<()> {
    let stream = mlx_rs::Stream::task_local_or_default();
    let status = unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()) };
    if status != 0 {
        return Err(Error::Msg(
            "iOS quantum stream synchronization failed".into(),
        ));
    }
    let index = stream.get_index().unwrap_or(-1);
    notify(|| QuantumEventArg::Synchronized(label.to_string(), index));
    mark(QuantumMark::Synchronized { label });
    Ok(())
}

pub fn finish(label: &str, arrays: &[&Array]) -> Result<()> {
    if enabled() {
        mlx_rs::transforms::eval(arrays.iter().copied())?;
        synchronize(label)?;
        crate::residency::drain_allocator_cache();
    }
    Ok(())
}

pub fn run(label: &str, scratch: u64, build: impl FnOnce() -> Result<Array>) -> Result<Array> {
    if !enabled() {
        return build();
    }
    let checked = std::cell::Cell::new(false);
    let available = available_for(label, scratch);
    let (required, reserve) = requirement(label, available, scratch);
    let result = budget::guarded(
        available,
        required,
        reserve,
        || {
            checked.set(true);
            notify(|| QuantumEventArg::Checked(label.to_string(), scratch));
            mark(QuantumMark::Checked { label, scratch });
            build()
        },
        |value| {
            mlx_rs::transforms::eval([value])?;
            synchronize(label)
        },
        || {
            crate::residency::drain_allocator_cache();
        },
    )
    .map_err(|error: Error| match error {
        Error::Msg(reason)
            if !checked.get()
                && (reason.starts_with("iOS quantum refused")
                    || reason == "iOS quantum accounting overflow") =>
        {
            record_refusal(label, required, reserve, available);
            Error::Unsupported(format!("{label}: {reason}"))
        }
        other => other,
    });
    result
}

/// Pre-load floor from the actual safetensors inventory, plus the largest tensor as read/copy
/// scratch. Called BEFORE model construction. Counts the entire VAE file even for txt2img.
/// Not a process upper bound: conversion or a backend operation can need more than this floor.
pub fn before_load(label: &str, components: &[std::path::PathBuf]) -> Result<()> {
    before_load_where(label, components, |_| true)
}

/// [`before_load`] over only the tensors `resident(name)` keeps: for a load that constructs just
/// part of a file up front (e.g. a rung-4 streamed DiT, whose block bodies are read per window by
/// that window's own priced quantum). Header-only; a no-op off device.
pub fn before_load_where(
    label: &str,
    components: &[std::path::PathBuf],
    resident: impl Fn(&str) -> bool,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    before(label, load_floor_bytes(components, resident)?)
}

/// Header-only load floor: the stored bytes of every kept tensor. Proven against MLX's measured
/// peak while a file is read (`cpu_quantum_materializes_before_return_refuses_before_build_and_
/// restores_cache`): the bytes land once, with no extra per-tensor read copy.
pub fn load_floor_bytes(
    components: &[std::path::PathBuf],
    resident: impl Fn(&str) -> bool,
) -> Result<u64> {
    let mut total = 0u64;
    for component in components {
        for header in crate::gen_core::weightsmeta::safetensors_path_tensor_headers(component)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            if resident(&header.name) {
                total = total
                    .checked_add(header.data_bytes)
                    .ok_or_else(|| Error::Unsupported("iOS load accounting overflow".into()))?;
            }
        }
    }
    Ok(total)
}

/// [`before_load`] for a component whose loader CASTS every tensor to a `bytes_per_element`-wide
/// dtype after reading it (e.g. an upcast to f32): the stored bytes AND the cast copies are live
/// together (proven against the measured peak of a read + cast). Header-only; a no-op off device.
pub fn before_load_cast(
    label: &str,
    components: &[std::path::PathBuf],
    bytes_per_element: u64,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    let overflow = || Error::Unsupported("iOS load accounting overflow".into());
    let mut total = 0u64;
    for component in components {
        for header in crate::gen_core::weightsmeta::safetensors_path_tensor_headers(component)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            let elements = header
                .shape
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d as u64))
                .ok_or_else(overflow)?;
            let cast = elements
                .checked_mul(bytes_per_element)
                .ok_or_else(overflow)?;
            total = total
                .checked_add(header.data_bytes)
                .and_then(|t| t.checked_add(cast))
                .ok_or_else(overflow)?;
        }
    }
    before(label, total)
}

#[cfg(all(test, target_os = "macos"))]
#[path = "../mlx-gen-iris/tests/common/metal_guard.rs"]
pub(crate) mod quantum_metal_guard;

#[cfg(all(test, target_os = "macos"))]
#[test]
fn cpu_quantum_materializes_before_return_refuses_before_build_and_restores_cache() {
    const CHILD: &str = "MLX_IOS_QUANTUM_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "ios_quantum::cpu_quantum_materializes_before_return_refuses_before_build_and_restores_cache", "--nocapture"])
            .env(CHILD,"1").status().unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(quantum_metal_guard::forbid_metal());
    assert!(quantum_metal_guard::metal_forbidden());
    fn available(value: &Array) -> bool {
        let mut result = false;
        assert_eq!(
            unsafe { mlx_sys::_mlx_array_is_available(&mut result, value.as_ptr()) },
            0
        );
        result
    }
    let a = Array::from_slice(&[1.0f32; 16], &[4, 4]);
    let lazy = mlx_rs::ops::add(&a, &a).unwrap();
    assert!(
        !available(&lazy),
        "fixture must contain a genuinely lazy operation"
    );
    let old_cache = mlx_rs::memory::set_cache_limit(1234567);
    with_cpu_diagnostic_headroom(1024 * 1024 * 1024, || {
        let _scope = Scope::new();
        assert_eq!(mlx_rs::memory::set_cache_limit(0), 0);
        let out = run("tiny lazy graph", 4096, || Ok(lazy))?;
        assert!(
            available(&out),
            "quantum returned its lazy graph rather than materializing it"
        );
        assert_eq!(out.as_slice::<f32>(), &[2.0f32; 16]);
        // Real mixed-dtype convolution: F16 storage must not hide an F32 promoted copy.
        let input = Array::ones::<f32>(&[1, 4, 4, 8])?;
        let weight = Array::ones::<f32>(&[40, 1, 1, 8])?.as_dtype(mlx_rs::Dtype::Float16)?;
        let promoted = run(
            "tiny F16 convolution",
            // Promoted F32 weights (40*8 elements) plus the f32 output, counted for this op.
            conv_stage_bytes(ConvStageGeometry {
                batch: 1,
                height: 4,
                width: 4,
                in_channels: 8,
                out_channels: 40,
                element_bytes: 4,
                in_live: 1,
                out_live: 1,
                weight_bytes: 40 * 8 * 4,
            }),
            || {
                Ok(mlx_rs::ops::conv2d(
                    &input,
                    &weight,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                )?)
            },
        )?;
        assert_eq!(promoted.dtype(), mlx_rs::Dtype::Float32);
        assert!(available(&promoted));
        Ok(())
    })
    .unwrap();
    assert_eq!(mlx_rs::memory::set_cache_limit(old_cache), 1234567);
    let constructed = std::cell::Cell::new(false);
    let refused = with_cpu_diagnostic_headroom(0, || {
        run("refused", 4096, || {
            constructed.set(true);
            Ok(a.clone())
        })
    });
    assert!(refused.is_err());
    assert!(!constructed.get());
    // The refusal is typed for a capture, independent of how the error string is propagated.
    let record = take_refusal().expect("a pre-construction refusal is recorded");
    assert_eq!(record.label, "refused");
    assert_eq!(record.scratch_bytes, 4096);
    assert_eq!(record.available_bytes, Some(0));
    assert_eq!(take_refusal(), None, "taking clears the record");
    let before_refused = with_cpu_diagnostic_headroom(1, || before("before refused", 8));
    assert!(before_refused.is_err());
    assert_eq!(
        take_refusal().map(|r| (r.label, r.available_bytes)),
        Some(("before refused".to_string(), Some(1)))
    );
    // The measurement every counted formula is proven against: two 256x256 f32 inputs built
    // inside the quantum plus their sum = three live 256 KiB tensors at the peak.
    let measured = measured_peak_bytes(|| {
        let a = Array::ones::<f32>(&[256, 256])?;
        let b = Array::ones::<f32>(&[256, 256])?;
        Ok(vec![mlx_rs::ops::add(&a, &b)?])
    })
    .unwrap();
    let three = 3 * tensor_bytes(4, &[256, 256]);
    assert_tracks_measured("ones + ones", three, measured, 1.5);
    // The measurement is STALLED, so it realizes the retention worst case whatever this host's
    // worker speed: a chain of eight 16 KiB matmuls (no donation), each read only by the next, is
    // encoded in full — every output allocated, every operand held by its queued task — before any
    // runs, so all eight products are live at once. The ops are tiny, so unstalled this host's
    // worker keeps up with the encoder and holds only four (measured) — the same 2x run-to-run
    // swing the CI runner showed on the Anima output quantum.
    let weight = Array::ones::<f32>(&[64, 64]).unwrap();
    let start = Array::ones::<f32>(&[64, 64]).unwrap();
    mlx_rs::transforms::eval([&weight, &start]).unwrap();
    let chain = measured_peak_bytes(|| {
        let mut x = start.clone();
        for _ in 0..8 {
            x = x.matmul(&weight)?;
        }
        Ok(vec![x])
    })
    .unwrap();
    assert_eq!(
        chain,
        8 * tensor_bytes(4, &[64, 64]),
        "a stalled measurement holds the whole encoded chain"
    );
    // The load floor is proven against MLX's peak while a safetensors file is read: the stored
    // bytes land once, with no extra per-tensor copy (no "largest tensor" scratch on top).
    {
        let dir = std::env::temp_dir().join(format!("ios-quantum-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("w.safetensors");
        let a = Array::ones::<f32>(&[256, 1024]).unwrap();
        let b = Array::ones::<f32>(&[512, 1024]).unwrap();
        let c = Array::ones::<f32>(&[1024, 1024]).unwrap();
        Array::save_safetensors(vec![("a", &a), ("b", &b), ("c", &c)], None, &file).unwrap();
        let measured = measured_peak_bytes(|| {
            let w = crate::weights::Weights::from_file(&file)?;
            Ok(["a", "b", "c"]
                .iter()
                .map(|k| w.require(k).unwrap().clone())
                .collect())
        })
        .unwrap();
        // Read + upcast: stored f16 bytes and the f32 copies are live together.
        let half = dir.join("h.safetensors");
        let h = [&a, &b, &c].map(|x| x.as_dtype(mlx_rs::Dtype::Float16).unwrap());
        Array::save_safetensors(vec![("a", &h[0]), ("b", &h[1]), ("c", &h[2])], None, &half)
            .unwrap();
        let cast_measured = measured_peak_bytes(|| {
            let w = crate::weights::Weights::from_file(&half)?;
            Ok(["a", "b", "c"]
                .iter()
                .map(|k| {
                    w.require(k)
                        .unwrap()
                        .as_dtype(mlx_rs::Dtype::Float32)
                        .unwrap()
                })
                .collect())
        })
        .unwrap();
        let stored_half = (256 + 512 + 1024) * 1024 * 2;
        assert_tracks_measured(
            "safetensors load + f32 cast",
            stored_half + 2 * stored_half,
            cast_measured,
            1.5,
        );
        assert_tracks_measured(
            "safetensors load",
            load_floor_bytes(std::slice::from_ref(&file), |_| true).unwrap(),
            measured,
            1.5,
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
    // A casting load prices stored bytes + the widened copies + the largest widened tensor.
    let dir = std::env::temp_dir().join(format!("ios-quantum-cast-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("vae.safetensors");
    let header = r#"{"w":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.resize(bytes.len() + 64, 0);
    std::fs::write(&file, bytes).unwrap();
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = seen.clone();
    with_cpu_diagnostic_headroom(1 << 30, || {
        with_cpu_diagnostic_observer(
            move |e| sink.borrow_mut().push(e.clone()),
            || {
                before_load("stored", std::slice::from_ref(&file))?;
                before_load_cast("cast", std::slice::from_ref(&file), 4)
            },
        )
    })
    .unwrap();
    assert_eq!(
        *seen.borrow(),
        [
            QuantumEvent::Checked {
                label: "stored".into(),
                scratch: 64
            },
            QuantumEvent::Checked {
                label: "cast".into(),
                scratch: 64 + 128
            },
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(all(test, target_os = "macos"))]
#[test]
fn cpu_quantum_synchronizes_the_installed_non_default_stream() {
    const CHILD: &str = "MLX_IOS_QUANTUM_STREAM_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ios_quantum::cpu_quantum_synchronizes_the_installed_non_default_stream",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(quantum_metal_guard::forbid_metal());
    let installed = mlx_rs::Stream::new_with_device(&mlx_rs::Device::cpu());
    let default_index = mlx_rs::Stream::new().get_index().unwrap();
    let installed_index = installed.get_index().unwrap();
    assert_ne!(
        installed_index, default_index,
        "fixture needs a genuinely non-default stream"
    );
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = seen.clone();
    with_cpu_diagnostic_headroom(1024 * 1024 * 1024, || {
        with_cpu_diagnostic_observer(
            move |event| sink.borrow_mut().push(event.clone()),
            || {
                mlx_rs::with_new_default_stream(installed.clone(), || {
                    let a = Array::from_slice(&[1.0f32; 4], &[4]);
                    run("non-default stream", 4096, || Ok(mlx_rs::ops::add(&a, &a)?))?;
                    finish("non-default finish", &[&a])
                })
            },
        )
    })
    .unwrap();
    let synchronized: Vec<i32> = seen
        .borrow()
        .iter()
        .filter_map(|event| match event {
            QuantumEvent::Synchronized { stream, .. } => Some(*stream),
            QuantumEvent::Checked { .. } => None,
        })
        .collect();
    assert_eq!(synchronized, [installed_index, installed_index]);
}

/// The admitted measured envelope (epic 16774 anchor law): inside it a guard needs only the
/// envelope's remaining bytes when its formula price is larger; a process grown past it is back on
/// the formula; and the production listener sees every check and synchronization in order.
#[cfg(all(test, target_os = "macos"))]
#[test]
fn an_admitted_envelope_replaces_formula_scratch_until_growth_exceeds_it() {
    const CHILD: &str = "MLX_IOS_QUANTUM_ENVELOPE_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ios_quantum::an_admitted_envelope_replaces_formula_scratch_until_growth_exceeds_it",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(quantum_metal_guard::forbid_metal());
    const MIB: u64 = 1 << 20;
    let headroom = 1024 * MIB;
    // Formula: 600 MiB scratch + 512 MiB reserve > 1024 MiB live headroom.
    let scratch = 600 * MIB;
    let marks = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let sink = std::sync::Arc::clone(&marks);
    assert!(!set_quantum_listener(Some(Box::new(move |m| {
        sink.lock().unwrap().push(match m {
            QuantumMark::Checked { label, .. } => format!("checked:{label}"),
            QuantumMark::Synchronized { label } => format!("synchronized:{label}"),
            QuantumMark::Metadata {
                provider,
                kind,
                fields,
            } => format!("metadata:{provider}:{kind}:{fields:?}"),
        })
    }))));
    with_cpu_diagnostic_headroom(headroom, || {
        assert!(
            before("formula", scratch).is_err(),
            "the formula alone refuses"
        );
        assert_eq!(take_refusal().unwrap().reserve_bytes, PROCESS_RESERVE_BYTES);
        // A measured envelope of 900 MiB, admitted against 1024 MiB: the guard now needs the
        // envelope's remaining 900 MiB, which the live headroom covers.
        with_admitted_envelope(900 * MIB, || before("enveloped", scratch))?;
        let ran = with_admitted_envelope(900 * MIB, || {
            run("enveloped run", scratch, || {
                Ok(mlx_rs::Array::from_slice(&[1.0f32, 2.0], &[2]))
            })
        })?;
        assert_eq!(ran.as_slice::<f32>(), &[1.0, 2.0]);
        // An envelope larger than live headroom (an unsound admission) still refuses.
        assert!(with_admitted_envelope(1100 * MIB, || before("unsound", scratch)).is_err());
        let refusal = take_refusal().unwrap();
        assert_eq!(
            (refusal.scratch_bytes, refusal.reserve_bytes),
            (1100 * MIB, 0)
        );
        // Growth beyond the envelope: the guard is back on its formula (the safety net).
        with_admitted_envelope(300 * MIB, || {
            with_cpu_diagnostic_headroom(headroom - 400 * MIB, || {
                assert!(before("past the envelope", 200 * MIB).is_err());
                Ok(())
            })
        })?;
        assert_eq!(take_refusal().unwrap().reserve_bytes, PROCESS_RESERVE_BYTES);
        // A cheap quantum whose formula is below the envelope's remaining keeps the formula.
        with_admitted_envelope(900 * MIB, || before("cheap", MIB))?;
        Ok(())
    })
    .unwrap();
    assert!(set_quantum_listener(None));
    assert_eq!(
        *marks.lock().unwrap(),
        [
            "checked:enveloped",
            "checked:enveloped run",
            "synchronized:enveloped run",
            "checked:cheap"
        ]
    );
}

/// Label/size ratios are diagnostic observations, not transferable physical bounds. Only the
/// complete already-admitted request envelope may lower a guard's raw formula requirement.
#[cfg(all(test, target_os = "macos"))]
#[test]
fn a_label_ratio_never_authorizes_changed_quantum_discount() {
    const CHILD: &str = "MLX_IOS_QUANTUM_CALIBRATION_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ios_quantum::a_label_ratio_never_authorizes_changed_quantum_discount",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(quantum_metal_guard::forbid_metal());
    const MIB: u64 = 1 << 20;
    let calibration = std::collections::BTreeMap::from([(
        "decode tile".to_string(),
        QuantumCalibration {
            priced_bytes: 1200 * MIB,
            observed_bytes: 600 * MIB,
        },
    )]);
    with_cpu_diagnostic_headroom(1024 * MIB, || {
        let envelope = open_admitted_envelope(5000 * MIB, calibration.clone(), 32 * MIB);
        // Even matching label and a smaller counted size cannot reuse the observed/priced ratio.
        assert_eq!(
            requirement("decode tile", Some(1024 * MIB), 600 * MIB),
            (600 * MIB, PROCESS_RESERVE_BYTES)
        );
        assert!(before("decode tile", 600 * MIB).is_err());
        assert_eq!(take_refusal().unwrap().reserve_bytes, PROCESS_RESERVE_BYTES);
        assert!(before("decode tile", 1200 * MIB).is_err());
        assert_eq!(take_refusal().unwrap().reserve_bytes, PROCESS_RESERVE_BYTES);
        assert!(before("denoise block", 600 * MIB).is_err());
        assert_eq!(take_refusal().unwrap().reserve_bytes, PROCESS_RESERVE_BYTES);
        drop(envelope);
        // Only the complete admitted envelope's remaining demand, with its margin already
        // included exactly once, may replace the more conservative per-quantum formula.
        let _complete = open_admitted_envelope(900 * MIB, calibration.clone(), 32 * MIB);
        assert_eq!(
            requirement("decode tile", Some(1024 * MIB), 600 * MIB),
            (900 * MIB, 0)
        );
        before("decode tile", 600 * MIB)?;
        Ok(())
    })
    .unwrap();
    // Past the complete admitted envelope, only the raw guarded formula applies.
    with_cpu_diagnostic_headroom(1024 * MIB, || {
        let _envelope = open_admitted_envelope(100 * MIB, calibration.clone(), 32 * MIB);
        with_cpu_diagnostic_headroom(800 * MIB, || {
            assert!(before("decode tile", 600 * MIB).is_err());
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(take_refusal().unwrap().reserve_bytes, PROCESS_RESERVE_BYTES);
}
