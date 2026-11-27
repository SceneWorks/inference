//! The **media fused-kernel path seam** (SW-6126, feature SW-F287): one switch layer and one
//! telemetry record per fused kernel, shared by every Candle media / audio kernel that has a
//! fused CUDA launch next to an op-chain reference.
//!
//! # Why here, and why beside — not instead of — the LLM seam
//!
//! The LLM decode leaves already have a seam, `candle_layers::fused` (`CANDLE_LLM_FUSED_KERNELS`):
//! one switch for all of them and a **per-thread, per-request** tally that `core_llm` reports on
//! every decode request. That shape is right for a decode step and stays exactly as it is. The
//! media lane needs something that seam does not give: a record **per kernel** (activation
//! quantizer, AdaLN, RoPE, VAE norm, …), each with its own runtime switch so a bench or a parity
//! test can flip one kernel and leave the rest alone, and **process-wide** counters a worker can
//! read once per job, like the shared attention router's `AttentionPathCounts` in candle-gen.
//!
//! It lives in `candle-quant-kernels` because that is the lowest crate every consumer shares:
//! the kernels themselves (the nvrtc seam, [`crate::nvrtc`]), `candle-layers`, `candle-gen` and
//! every `candle-gen-*` provider (re-exported as `candle_gen::fused_path`), and the audio crates.
//! The switch is the workspace's one switch implementation, `core_llm::switch::ProcessSwitch`.
//!
//! # The switch (three layers, most specific first)
//!
//! A kernel runs fused only when **both** of these are on:
//!
//! 1. the kernel's own switch — [`FusedKernelPath::set_enabled`] (runtime override), else the
//!    kernel's environment variable (named in its declaration), else on;
//! 2. the media-wide switch — [`set_media_fused_kernels`], else [`MEDIA_FUSED_KERNELS_ENV`]
//!    (`CANDLE_MEDIA_FUSED_KERNELS`), else on.
//!
//! A value of `0`, `off`, `false`, `no` or `reference` turns a variable off (the LLM seam's
//! words). Either switch off sends every call to the reference with [`PathReason::Disabled`].
//! Tests that flip a switch, or assert counts a flip could change, hold the switch's guard
//! ([`FusedKernelPath::guard`], [`media_fused_policy_guard`]), which serializes them and restores
//! the policy it found.
//!
//! # The telemetry
//!
//! Every call records exactly one outcome: **fused**, or **reference** with a typed
//! [`PathReason`] (stable lower-case labels in [`REFERENCE_REASONS`]). Two views are kept:
//!
//! - process-wide counters per kernel: [`FusedKernelPath::counts`] / [`FusedKernelPath::take_counts`],
//!   and every kernel that has recorded anything at once through [`fused_path_counts`] /
//!   [`take_fused_path_counts`] (a kernel registers itself on its first record — no link-time
//!   registration);
//! - this thread's counters per kernel, [`FusedKernelPath::thread_counts`], which parallel tests
//!   (and a caller that wants one request's delta) read with [`PathCounts::since`].
//!
//! A downgrade is never silent: it is counted, and a reason that means "the device cannot run the
//! kernel" (a cached nvrtc / load / compute-floor failure from the nvrtc seam) is also logged once
//! per kernel and reason at `warn`. Expected references (the switch is off, a CPU tensor, a dtype
//! or shape the kernel does not serve) are counted, not logged. A fused → reference downgrade
//! never changes backend: the reference runs on the input's own device.
//!
//! # Adding a kernel (the whole recipe)
//!
//! ```ignore
//! use candle_quant_kernels::fused_path::{Attempt, FusedKernelPath, PathReason};
//!
//! /// Declared once, next to the kernel.
//! pub static ADALN_PATH: FusedKernelPath =
//!     FusedKernelPath::new("adaln_modulate", "CANDLE_FUSED_ADALN");
//!
//! pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
//!     ADALN_PATH.dispatch(
//!         // cuda builds only call this; return `Attempt::Reference(reason)` for an input the
//!         // kernel does not serve (from a pure, CPU-testable `check_*` plan function) and for a
//!         // cached compile error (`PathReason::from(err)`); `Err` only for a genuine failure.
//!         || fused_modulate(x, shift, scale),
//!         || reference_modulate(x, shift, scale),
//!     )
//! }
//! ```
//!
//! [`FusedKernelPath::dispatch`] applies the switch, calls the fused attempt (never on a build
//! without `cuda`, which records [`PathReason::CudaFeatureOff`]), records the outcome and runs the
//! reference when the attempt declined. A `KernelSource` the kernel compiles must also be listed in
//! [`crate::NVRTC_SOURCES`], and a bench row added to candle-gen's `fused_kernels_bench` example.

use std::cell::RefCell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::{DType, Result};
use core_llm::switch::{ProcessSwitch, SwitchGuard};

use crate::nvrtc::KernelCompileError;

/// The media-wide environment switch: `0` / `off` / `false` / `no` / `reference` send every media
/// fused kernel to its reference.
pub const MEDIA_FUSED_KERNELS_ENV: &str = "CANDLE_MEDIA_FUSED_KERNELS";

fn env_value_enables(v: &str) -> bool {
    !matches!(v, "0" | "off" | "false" | "no" | "reference")
}

static MEDIA_SWITCH: ProcessSwitch =
    ProcessSwitch::new(MEDIA_FUSED_KERNELS_ENV, true, env_value_enables);

/// Whether the media-wide switch allows fused kernels.
pub fn media_fused_kernels_enabled() -> bool {
    MEDIA_SWITCH.enabled()
}

/// Override the media-wide switch: `Some(true)` / `Some(false)` force it, `None` returns to
/// [`MEDIA_FUSED_KERNELS_ENV`].
pub fn set_media_fused_kernels(enabled: Option<bool>) {
    MEDIA_SWITCH.set(enabled);
}

/// Test / bench seam: hold the media-wide switch's lock with `enabled` applied; the policy found is
/// restored when the guard drops.
pub fn media_fused_policy_guard(enabled: Option<bool>) -> SwitchGuard {
    MEDIA_SWITCH.guard(enabled)
}

/// Why a call ran the reference instead of the fused kernel. The label
/// ([`label`](Self::label)) is what the counters are keyed by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathReason {
    /// The kernel's switch or the media-wide switch is off.
    Disabled,
    /// This build has no `cuda` feature, so no fused path exists.
    CudaFeatureOff,
    /// The input is not on a CUDA device.
    NotCuda,
    /// The kernel does not serve this dtype.
    Dtype(DType),
    /// The kernel does not serve this shape (rank, extent, an empty tensor, …).
    Shape,
    /// The kernel does not serve this memory layout (strides, alignment, …).
    Layout,
    /// An extent or index exceeds what the kernel addresses (epic E6: no tensor beyond
    /// `i32::MAX` elements is launched).
    IndexOverflow,
    /// A gradient could flow through the call: training paths keep their composable ops (E9).
    Autograd,
    /// The kernel's source cannot be used on this device (cached by the nvrtc seam): below its
    /// compute-capability floor, an nvrtc / load failure, a missing symbol or a device query error.
    Compile(KernelCompileError),
}

/// Every reference reason's label, in counter-slot order ([`PathCounts::reference`] is indexed by
/// it). The compile-class labels are [`KernelCompileError::label`]'s.
pub const REFERENCE_REASONS: [&str; 13] = [
    "disabled",
    "cuda_feature_off",
    "not_cuda",
    "dtype",
    "shape",
    "layout",
    "index_overflow",
    "autograd",
    "compute_floor",
    "nvrtc",
    "load",
    "device",
    "missing_function",
];

/// Number of reference-reason counter slots.
pub const REFERENCE_REASON_COUNT: usize = REFERENCE_REASONS.len();

impl PathReason {
    /// Stable lower-case label (one of [`REFERENCE_REASONS`]).
    pub fn label(&self) -> &'static str {
        REFERENCE_REASONS[self.slot()]
    }

    fn slot(&self) -> usize {
        match self {
            Self::Disabled => 0,
            Self::CudaFeatureOff => 1,
            Self::NotCuda => 2,
            Self::Dtype(_) => 3,
            Self::Shape => 4,
            Self::Layout => 5,
            Self::IndexOverflow => 6,
            Self::Autograd => 7,
            Self::Compile(e) => match e {
                KernelCompileError::BelowComputeFloor { .. } => 8,
                KernelCompileError::Nvrtc { .. } => 9,
                KernelCompileError::Load { .. } => 10,
                KernelCompileError::Device { .. } => 11,
                KernelCompileError::MissingFunction { .. } => 12,
            },
        }
    }

    /// Whether this reason means the device cannot run the kernel at all (logged once at `warn`),
    /// rather than an input or a policy the reference is simply the path for.
    pub fn is_unexpected(&self) -> bool {
        matches!(self, Self::Compile(_))
    }
}

impl From<KernelCompileError> for PathReason {
    fn from(e: KernelCompileError) -> Self {
        Self::Compile(e)
    }
}

impl fmt::Display for PathReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dtype(d) => write!(f, "dtype {d:?} is not served"),
            Self::Compile(e) => write!(f, "{e}"),
            other => f.write_str(other.label()),
        }
    }
}

/// What a fused attempt produced: the result, or a typed reason to run the reference instead.
#[derive(Debug)]
pub enum Attempt<T> {
    /// The fused kernel ran.
    Fused(T),
    /// The kernel declined this call; the reference serves it.
    Reference(PathReason),
}

impl<T> Attempt<T> {
    /// Apply `f` to a fused result (a fallible post-step such as staging it for a GEMM); a
    /// declined attempt passes through with its reason.
    pub fn try_map<U>(self, f: impl FnOnce(T) -> Result<U>) -> Result<Attempt<U>> {
        match self {
            Self::Fused(v) => f(v).map(Attempt::Fused),
            Self::Reference(reason) => Ok(Attempt::Reference(reason)),
        }
    }
}

/// A snapshot of one kernel's path counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PathCounts {
    /// Calls served by the fused kernel.
    pub fused: u64,
    /// Calls served by the reference, per reason (indexed like [`REFERENCE_REASONS`]).
    pub reference: [u64; REFERENCE_REASON_COUNT],
}

impl PathCounts {
    /// Reference calls with the reason labelled `label` (0 for an unknown label).
    pub fn reference_for(&self, label: &str) -> u64 {
        REFERENCE_REASONS
            .iter()
            .position(|known| *known == label)
            .map_or(0, |slot| self.reference[slot])
    }

    /// Every reference call, whatever the reason.
    pub fn reference_total(&self) -> u64 {
        self.reference.iter().sum()
    }

    /// Every recorded call.
    pub fn total(&self) -> u64 {
        self.fused + self.reference_total()
    }

    /// The field-wise difference `self - earlier` (saturating), for a window between snapshots.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            fused: self.fused.saturating_sub(earlier.fused),
            reference: std::array::from_fn(|i| {
                self.reference[i].saturating_sub(earlier.reference[i])
            }),
        }
    }

    /// `none`, `fused`, `reference` or `mixed` — which paths ran in this window.
    pub fn label(&self) -> &'static str {
        match (self.fused > 0, self.reference_total() > 0) {
            (false, false) => "none",
            (true, false) => "fused",
            (false, true) => "reference",
            (true, true) => "mixed",
        }
    }

    /// The reference reasons with a nonzero count, `(label, count)`, in slot order.
    pub fn reference_reasons(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        REFERENCE_REASONS
            .iter()
            .zip(self.reference)
            .filter(|(_, n)| *n > 0)
            .map(|(label, n)| (*label, n))
    }

    fn add(&mut self, slot: Option<usize>) {
        match slot {
            None => self.fused = self.fused.wrapping_add(1),
            Some(s) => self.reference[s] = self.reference[s].wrapping_add(1),
        }
    }
}

impl fmt::Display for PathCounts {
    /// `fused=N reference=M (reason=n, …)`, the form a job log carries.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fused={} reference={}",
            self.fused,
            self.reference_total()
        )?;
        let mut reasons = self.reference_reasons().peekable();
        if reasons.peek().is_some() {
            f.write_str(" (")?;
            for (i, (label, n)) in reasons.enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{label}={n}")?;
            }
            f.write_str(")")?;
        }
        Ok(())
    }
}

/// One fused kernel's switch and path record. Declare one `static` per kernel (see the
/// [module docs](self)); every method that records takes `&'static self`.
pub struct FusedKernelPath {
    name: &'static str,
    switch: ProcessSwitch,
    fused: AtomicU64,
    reference: [AtomicU64; REFERENCE_REASON_COUNT],
    registered: AtomicBool,
    /// One bit per reason slot already logged at `warn`.
    warned: AtomicU32,
}

impl FusedKernelPath {
    /// A kernel record named `name` (stable; it labels telemetry and the bench row) whose own
    /// switch reads the environment variable `env`.
    pub const fn new(name: &'static str, env: &'static str) -> Self {
        Self {
            name,
            switch: ProcessSwitch::new(env, true, env_value_enables),
            fused: AtomicU64::new(0),
            reference: [const { AtomicU64::new(0) }; REFERENCE_REASON_COUNT],
            registered: AtomicBool::new(false),
            warned: AtomicU32::new(0),
        }
    }

    /// The kernel's telemetry name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The environment variable of the kernel's own switch.
    pub fn env(&self) -> &'static str {
        self.switch.env()
    }

    /// Whether a call may try the fused kernel: the kernel's switch and the media-wide switch are
    /// both on (a `cuda` build is still required for a fused path to exist).
    pub fn enabled(&self) -> bool {
        self.switch.enabled() && media_fused_kernels_enabled()
    }

    /// Override this kernel's switch: `Some(true)` / `Some(false)` force it, `None` returns to its
    /// environment variable. The media-wide switch still applies.
    pub fn set_enabled(&self, enabled: Option<bool>) {
        self.switch.set(enabled);
    }

    /// Test / bench seam: hold this kernel's switch lock with `enabled` applied; restores the
    /// policy it found when dropped.
    pub fn guard(&'static self, enabled: Option<bool>) -> SwitchGuard {
        self.switch.guard(enabled)
    }

    /// Record one call served by the fused kernel.
    pub fn record_fused(&'static self) {
        self.record(None);
    }

    /// Record one call served by the reference, with why.
    pub fn record_reference(&'static self, reason: &PathReason) {
        let slot = reason.slot();
        if reason.is_unexpected()
            && self.warned.fetch_or(1 << slot, Ordering::Relaxed) & (1 << slot) == 0
        {
            tracing::warn!(
                kernel = self.name,
                reason = reason.label(),
                "fused kernel unavailable on this device, running the reference op chain: {reason}"
            );
        }
        self.record(Some(slot));
    }

    fn record(&'static self, slot: Option<usize>) {
        if !self.registered.swap(true, Ordering::Relaxed) {
            registry()
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(self);
        }
        match slot {
            None => self.fused.fetch_add(1, Ordering::Relaxed),
            Some(s) => self.reference[s].fetch_add(1, Ordering::Relaxed),
        };
        THREAD_COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            let key = self as *const Self;
            match counts.iter_mut().find(|(k, _)| *k == key) {
                Some((_, c)) => c.add(slot),
                None => {
                    let mut c = PathCounts::default();
                    c.add(slot);
                    counts.push((key, c));
                }
            }
        });
    }

    /// The process-wide counters.
    pub fn counts(&self) -> PathCounts {
        PathCounts {
            fused: self.fused.load(Ordering::Relaxed),
            reference: std::array::from_fn(|i| self.reference[i].load(Ordering::Relaxed)),
        }
    }

    /// Drain the process-wide counters (each slot atomically; a racing call lands in this snapshot
    /// or the next, never lost). The per-thread counters are not affected.
    pub fn take_counts(&self) -> PathCounts {
        PathCounts {
            fused: self.fused.swap(0, Ordering::Relaxed),
            reference: std::array::from_fn(|i| self.reference[i].swap(0, Ordering::Relaxed)),
        }
    }

    /// This thread's counters for this kernel (monotone; take a window with
    /// [`PathCounts::since`]).
    pub fn thread_counts(&self) -> PathCounts {
        let key = self as *const Self;
        THREAD_COUNTS.with(|counts| {
            counts
                .borrow()
                .iter()
                .find(|(k, _)| *k == key)
                .map_or_else(PathCounts::default, |(_, c)| *c)
        })
    }

    /// Run one call through this kernel's path: the fused `attempt` when the switch allows it (and
    /// the build has `cuda`), else — or when the attempt declines — the `reference`. Exactly one
    /// outcome is recorded. A genuine error from either side is returned as is; it is not a
    /// downgrade.
    pub fn dispatch<T>(
        &'static self,
        attempt: impl FnOnce() -> Result<Attempt<T>>,
        reference: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !cfg!(feature = "cuda") {
            return self.run_reference(&PathReason::CudaFeatureOff, reference);
        }
        if !self.enabled() {
            return self.run_reference(&PathReason::Disabled, reference);
        }
        match attempt()? {
            Attempt::Fused(v) => {
                self.record_fused();
                Ok(v)
            }
            Attempt::Reference(reason) => self.run_reference(&reason, reference),
        }
    }

    /// Record `reason` and run the reference (for a caller that decides before any attempt).
    pub fn run_reference<T>(
        &'static self,
        reason: &PathReason,
        reference: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.record_reference(reason);
        reference()
    }
}

impl fmt::Debug for FusedKernelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedKernelPath")
            .field("name", &self.name)
            .field("env", &self.env())
            .field("enabled", &self.enabled())
            .field("counts", &self.counts())
            .finish()
    }
}

thread_local! {
    static THREAD_COUNTS: RefCell<Vec<(*const FusedKernelPath, PathCounts)>> =
        const { RefCell::new(Vec::new()) };
}

fn registry() -> &'static Mutex<Vec<&'static FusedKernelPath>> {
    static REGISTRY: Mutex<Vec<&'static FusedKernelPath>> = Mutex::new(Vec::new());
    &REGISTRY
}

/// Every kernel that has recorded at least one call in this process, with its process-wide
/// counters, in first-use order.
pub fn fused_path_counts() -> Vec<(&'static str, PathCounts)> {
    registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|path| (path.name, path.counts()))
        .collect()
}

/// [`fused_path_counts`], draining each kernel's process-wide counters (a worker's per-job read).
pub fn take_fused_path_counts() -> Vec<(&'static str, PathCounts)> {
    registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|path| (path.name, path.take_counts()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static UNDER_TEST: FusedKernelPath =
        FusedKernelPath::new("fused_path_test_kernel", "CANDLE_FUSED_PATH_TEST_UNSET");

    fn compile_error() -> KernelCompileError {
        KernelCompileError::Nvrtc {
            name: "k",
            message: "boom".into(),
        }
    }

    #[test]
    fn every_reason_has_a_distinct_slot_and_label() {
        let reasons = [
            PathReason::Disabled,
            PathReason::CudaFeatureOff,
            PathReason::NotCuda,
            PathReason::Dtype(DType::U8),
            PathReason::Shape,
            PathReason::Layout,
            PathReason::IndexOverflow,
            PathReason::Autograd,
            PathReason::Compile(KernelCompileError::BelowComputeFloor {
                name: "k",
                floor: (8, 9),
                found: (7, 5),
            }),
            PathReason::Compile(compile_error()),
            PathReason::Compile(KernelCompileError::Load {
                name: "k",
                message: String::new(),
            }),
            PathReason::Compile(KernelCompileError::Device {
                name: "k",
                message: String::new(),
            }),
            PathReason::Compile(KernelCompileError::MissingFunction {
                name: "k",
                function: "f".into(),
            }),
        ];
        let slots: Vec<usize> = reasons.iter().map(PathReason::slot).collect();
        assert_eq!(slots, (0..REFERENCE_REASON_COUNT).collect::<Vec<_>>());
        for reason in &reasons {
            if let PathReason::Compile(e) = reason {
                assert_eq!(reason.label(), e.label(), "compile labels are the seam's");
                assert!(reason.is_unexpected());
            } else {
                assert!(!reason.is_unexpected(), "{reason}");
            }
        }
    }

    #[test]
    fn counts_window_label_and_display() {
        let mut c = PathCounts::default();
        assert_eq!(c.label(), "none");
        c.add(None);
        c.add(Some(PathReason::Shape.slot()));
        c.add(Some(PathReason::Shape.slot()));
        assert_eq!(c.label(), "mixed");
        assert_eq!(c.reference_for("shape"), 2);
        assert_eq!(c.reference_for("nonsense"), 0);
        assert_eq!(c.total(), 3);
        assert_eq!(c.to_string(), "fused=1 reference=2 (shape=2)");
        let later = {
            let mut l = c;
            l.add(Some(PathReason::Disabled.slot()));
            l
        };
        let d = later.since(&c);
        assert_eq!(d.label(), "reference");
        assert_eq!(d.reference_for("disabled"), 1);
        assert_eq!(d.to_string(), "fused=0 reference=1 (disabled=1)");
    }

    /// The switch decides before any attempt; a declined attempt runs the reference and records
    /// the attempt's reason; a fused attempt never runs the reference; an error is not a downgrade.
    #[test]
    fn dispatch_records_exactly_one_outcome_per_call() {
        let _kernel = UNDER_TEST.guard(Some(true));
        let _media = media_fused_policy_guard(Some(true));
        let start = UNDER_TEST.thread_counts();
        let process_start = UNDER_TEST.counts();

        let got = UNDER_TEST
            .dispatch(
                || Ok(Attempt::Fused(1)),
                || {
                    // Without `cuda` the reference is the only path (and the attempt never runs).
                    if cfg!(feature = "cuda") {
                        panic!("the reference must not run");
                    }
                    Ok(1)
                },
            )
            .unwrap();
        assert_eq!(got, 1);
        let d = UNDER_TEST.thread_counts().since(&start);
        if cfg!(feature = "cuda") {
            assert_eq!((d.fused, d.reference_total()), (1, 0));
        } else {
            // No fused path exists without `cuda`: the attempt is never called.
            assert_eq!(d.reference_for("cuda_feature_off"), 1);
        }

        let mid = UNDER_TEST.thread_counts();
        let got = UNDER_TEST
            .dispatch(|| Ok(Attempt::Reference(PathReason::Shape)), || Ok(2))
            .unwrap();
        assert_eq!(got, 2);
        let d = UNDER_TEST.thread_counts().since(&mid);
        let expected = if cfg!(feature = "cuda") {
            "shape"
        } else {
            "cuda_feature_off"
        };
        assert_eq!((d.fused, d.reference_for(expected)), (0, 1));

        UNDER_TEST.set_enabled(Some(false));
        let off = UNDER_TEST.thread_counts();
        let got = UNDER_TEST
            .dispatch(
                || -> Result<Attempt<i32>> { panic!("a disabled kernel is never attempted") },
                || Ok(3),
            )
            .unwrap();
        assert_eq!(got, 3);
        let d = UNDER_TEST.thread_counts().since(&off);
        let expected = if cfg!(feature = "cuda") {
            "disabled"
        } else {
            "cuda_feature_off"
        };
        assert_eq!(d.reference_for(expected), 1);
        UNDER_TEST.set_enabled(Some(true));

        if cfg!(feature = "cuda") {
            let before = UNDER_TEST.thread_counts();
            let err = UNDER_TEST.dispatch(
                || -> Result<Attempt<i32>> { candle_core::bail!("genuine failure") },
                || panic!("an error is not a downgrade"),
            );
            assert!(err.is_err());
            assert_eq!(UNDER_TEST.thread_counts().since(&before).total(), 0);
        }

        // The process-wide view saw (at least) this thread's calls, and lists the kernel.
        assert!(UNDER_TEST.counts().since(&process_start).total() >= 3);
        assert!(fused_path_counts()
            .iter()
            .any(|(name, _)| *name == "fused_path_test_kernel"));
    }

    #[test]
    fn the_media_switch_gates_every_kernel() {
        let _kernel = UNDER_TEST.guard(Some(true));
        let _media = media_fused_policy_guard(Some(false));
        assert!(
            !UNDER_TEST.enabled(),
            "media-wide off wins over a kernel on"
        );
        set_media_fused_kernels(Some(true));
        assert!(UNDER_TEST.enabled());
        UNDER_TEST.set_enabled(Some(false));
        assert!(
            !UNDER_TEST.enabled(),
            "a kernel off wins over media-wide on"
        );
    }

    #[test]
    fn a_compile_reason_is_counted_under_the_seam_label() {
        let start = UNDER_TEST.thread_counts();
        UNDER_TEST.record_reference(&PathReason::from(compile_error()));
        UNDER_TEST.record_reference(&PathReason::from(compile_error()));
        let d = UNDER_TEST.thread_counts().since(&start);
        assert_eq!(d.reference_for("nvrtc"), 2);
    }

    #[test]
    fn thread_counts_are_per_thread() {
        let before = UNDER_TEST.thread_counts();
        let other = std::thread::spawn(|| {
            let start = UNDER_TEST.thread_counts();
            UNDER_TEST.record_reference(&PathReason::NotCuda);
            UNDER_TEST.thread_counts().since(&start)
        })
        .join()
        .unwrap();
        assert_eq!(other.reference_for("not_cuda"), 1);
        assert_eq!(
            UNDER_TEST
                .thread_counts()
                .since(&before)
                .reference_for("not_cuda"),
            0
        );
    }

    #[test]
    fn env_words_match_the_llm_seam() {
        for off in ["0", "off", "false", "no", "reference"] {
            assert!(!env_value_enables(off), "{off}");
        }
        for on in ["1", "on", "true", "fused", ""] {
            assert!(env_value_enables(on), "{on}");
        }
    }
}
