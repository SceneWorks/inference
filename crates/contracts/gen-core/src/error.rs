//! The **backend-neutral** crate error. gen-core cannot name `mlx_rs` (or candle) types, so device
//! failures arrive boxed in [`Error::Backend`] — each tensor backend lifts its own exception into
//! it (mlx-gen via `From<mlx_gen::Error>`, see `mlx-gen/src/error.rs`). The typed [`Error::Canceled`]
//! and [`Error::Unsupported`] variants are contract-load-bearing: the worker and the conformance
//! testkit distinguish cancellation and capability gaps from generic failure (epic 3720, D3).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Failure-isolation and GPU-cleanup classification (`PIPE-15`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// Request-scoped failure (validation refusal, cancellation, transient backend/shape fault).
    /// Safe to run GPU/request cleanup such as releasing allocator caches or stream temporaries.
    RequestLocal,
    /// The provider instance's internal state or cache is corrupted; retire that provider instance
    /// without running GPU/session cleanup across its state, while unrelated providers survive.
    ProviderPoisoned,
    /// Backend or device context is unrecoverably poisoned process-wide; retire the executor
    /// without touching GPU cleanup.
    ProcessPoisoned,
}

/// Anything that can go wrong across a gen-core contract call.
#[derive(Debug, Error)]
pub enum Error {
    /// A backend tensor/device operation failed (an MLX exception, a CUDA error, …), boxed because
    /// gen-core cannot name backend types. Construct via [`Error::backend`].
    #[error("backend op failed: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),

    /// A required tensor key was absent from a loaded checkpoint/adapter.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem error (model-dir traversal, safetensors open).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The request asked for something this engine/backend cannot do. Candle gating depends on this
    /// being typed — do **not** stringify it into [`Error::Msg`].
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Generation/training was cancelled via a `CancelFlag`. Typed so the worker and testkit can
    /// distinguish cancellation from failure; providers must check at step boundaries and return
    /// this (never a partial output).
    #[error("cancelled")]
    Canceled,

    /// The provider instance has entered an unrecoverable internal state and must be dropped/retired
    /// without running GPU cleanup across its corrupted locals; unrelated providers and the process
    /// survive (`PIPE-15`).
    #[error("provider poisoned: {0}")]
    ProviderPoisoned(String),

    /// The provider has identified an unrecoverable process-local backend state. The worker must
    /// retire its generator owner after this request; ordinary backend failures must not use it.
    /// A provider emitting this must contain or forget its own GPU-backed locals before return:
    /// the shared residency owner can protect only components it still owns at that boundary.
    #[error("process poisoned: {0}")]
    ProcessPoisoned(String),

    /// The requested output geometry was measured infeasible before render. `alternative` is present
    /// only when a current evidence record verifies that exact geometry; callers must never populate
    /// it from an estimate or tier heuristic.
    #[error(
        "geometry refused: {reason}; requested {requested_width}x{requested_height}; verified alternative: {alternative:?}"
    )]
    GeometryRefused {
        reason: String,
        requested_width: u32,
        requested_height: u32,
        alternative: Option<(u32, u32)>,
    },

    /// A contextual message (config/validation/adapter-shape errors).
    #[error("{0}")]
    Msg(String),
}

impl Error {
    /// Lift a concrete backend error into [`Error::Backend`]. Backends call this (or rely on their
    /// own `From` impl) at the seam where an `mlx_rs`/candle `Result` crosses into gen-core.
    ///
    /// F-127 tripwire: a typed [`Error::Canceled`] must NEVER be laundered through here — that would
    /// bury the cancellation contract inside an opaque `Backend` box and break the worker/testkit's
    /// `matches!(_, Error::Canceled)` check. A backend's `From` impl maps `Canceled` across 1:1
    /// *before* reaching this constructor, so a "cancelled"-stringified source landing here is a bug
    /// in that bridge. Caught in debug builds only (zero release cost).
    pub fn backend(e: impl std::error::Error + Send + Sync + 'static) -> Error {
        debug_assert_ne!(
            e.to_string(),
            "cancelled",
            "Error::backend received a cancellation — map it to Error::Canceled at the bridge, \
             don't bury it in Backend (F-127)"
        );
        Error::Backend(Box::new(e))
    }

    /// Classify this error for cleanup and failure isolation (`PIPE-15`).
    pub const fn error_class(&self) -> ErrorClass {
        match self {
            Self::ProviderPoisoned(_) => ErrorClass::ProviderPoisoned,
            Self::ProcessPoisoned(_) => ErrorClass::ProcessPoisoned,
            Self::Backend(_)
            | Self::MissingTensor(_)
            | Self::Io(_)
            | Self::Unsupported(_)
            | Self::Canceled
            | Self::GeometryRefused { .. }
            | Self::Msg(_) => ErrorClass::RequestLocal,
        }
    }

    /// Returns `true` only for [`ErrorClass::RequestLocal`] errors where GPU cleanup may run
    /// safely (`PIPE-15`).
    pub const fn allows_gpu_cleanup(&self) -> bool {
        matches!(self.error_class(), ErrorClass::RequestLocal)
    }
}

/// Disposition recorded by [`run_with_cleanup_policy_with_outcome`] (`PIPE-15`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CleanupPolicyOutcome {
    /// `work` succeeded; post-fault GPU cleanup was not invoked.
    Succeeded,
    /// `work` failed with [`ErrorClass::RequestLocal`]; GPU cleanup executed.
    CleanedRequestLocalFault,
    /// `work` failed with a poison error class; GPU cleanup was suppressed.
    SkippedPoisonedFault(ErrorClass),
}

/// Execute `work`, running `gpu_cleanup` ONLY when `work` fails with [`ErrorClass::RequestLocal`]
/// (`PIPE-15`).
///
/// Both [`ErrorClass::ProviderPoisoned`] and [`ErrorClass::ProcessPoisoned`] suppress `gpu_cleanup`
/// so corrupted provider locals or wedged device contexts are never touched during unwind.
pub fn run_with_cleanup_policy_with_outcome<T>(
    work: impl FnOnce() -> Result<T>,
    gpu_cleanup: impl FnOnce(),
) -> (Result<T>, CleanupPolicyOutcome) {
    match work() {
        Ok(value) => (Ok(value), CleanupPolicyOutcome::Succeeded),
        Err(err) => {
            let class = err.error_class();
            let outcome = if class == ErrorClass::RequestLocal {
                gpu_cleanup();
                CleanupPolicyOutcome::CleanedRequestLocalFault
            } else {
                CleanupPolicyOutcome::SkippedPoisonedFault(class)
            };
            (Err(err), outcome)
        }
    }
}

/// Execute `work`, running `gpu_cleanup` ONLY when `work` fails with [`ErrorClass::RequestLocal`]
/// (`PIPE-15`).
pub fn run_with_cleanup_policy<T>(
    work: impl FnOnce() -> Result<T>,
    gpu_cleanup: impl FnOnce(),
) -> Result<T> {
    run_with_cleanup_policy_with_outcome(work, gpu_cleanup).0
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}

impl From<core_llm::Error> for Error {
    fn from(error: core_llm::Error) -> Self {
        match error {
            core_llm::Error::Canceled => Self::Canceled,
            core_llm::Error::Unsupported(reason) => Self::Unsupported(reason),
            other => Self::Msg(other.to_string()),
        }
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_strings_are_stable() {
        assert_eq!(Error::Canceled.to_string(), "cancelled");
        assert_eq!(
            Error::Unsupported("q4".into()).to_string(),
            "unsupported: q4"
        );
        assert_eq!(
            Error::MissingTensor("unet.x".into()).to_string(),
            "missing tensor: unet.x"
        );
        assert!(Error::GeometryRefused {
            reason: "measured infeasible".into(),
            requested_width: 1024,
            requested_height: 768,
            alternative: Some((768, 512)),
        }
        .to_string()
        .contains("requested 1024x768"));
        assert_eq!(Error::Msg("boom".into()).to_string(), "boom");
        assert_eq!(
            Error::ProviderPoisoned("provider cache wedged".into()).to_string(),
            "provider poisoned: provider cache wedged"
        );
        assert_eq!(
            Error::ProcessPoisoned("device context lost".into()).to_string(),
            "process poisoned: device context lost"
        );
    }

    #[test]
    fn backend_wraps_a_source_error() {
        let io = std::io::Error::other("device lost");
        let e = Error::backend(io);
        assert!(matches!(e, Error::Backend(_)));
        assert_eq!(e.to_string(), "backend op failed: device lost");
    }

    #[test]
    fn str_and_string_lift_to_msg() {
        assert!(matches!(Error::from("x"), Error::Msg(_)));
        assert!(matches!(Error::from(String::from("y")), Error::Msg(_)));
    }

    #[test]
    fn error_class_and_cleanup_policy_run_gpu_cleanup_only_on_request_local_faults() {
        use std::cell::Cell;

        let request_local_variants = vec![
            Error::backend(std::io::Error::other("transient device stall")),
            Error::MissingTensor("k".into()),
            Error::Io(std::io::Error::other("io")),
            Error::Unsupported("bad combo".into()),
            Error::Canceled,
            Error::GeometryRefused {
                reason: "oof".into(),
                requested_width: 1024,
                requested_height: 1024,
                alternative: None,
            },
            Error::Msg("bad shape".into()),
        ];

        for fault in request_local_variants {
            assert_eq!(fault.error_class(), ErrorClass::RequestLocal);
            assert!(fault.allows_gpu_cleanup());
            let cleaned = Cell::new(false);
            let (res, outcome) = run_with_cleanup_policy_with_outcome(
                || Err::<(), Error>(fault),
                || cleaned.set(true),
            );
            assert!(res.is_err());
            assert!(cleaned.get(), "GPU cleanup must run on RequestLocal fault");
            assert_eq!(outcome, CleanupPolicyOutcome::CleanedRequestLocalFault);
        }

        let prov = Error::ProviderPoisoned("stale rope cache".into());
        assert_eq!(prov.error_class(), ErrorClass::ProviderPoisoned);
        assert!(!prov.allows_gpu_cleanup());
        let cleaned = Cell::new(false);
        let (res, outcome) =
            run_with_cleanup_policy_with_outcome(|| Err::<(), Error>(prov), || cleaned.set(true));
        assert!(matches!(res, Err(Error::ProviderPoisoned(_))));
        assert!(
            !cleaned.get(),
            "GPU cleanup must NOT run on ProviderPoisoned fault"
        );
        assert_eq!(
            outcome,
            CleanupPolicyOutcome::SkippedPoisonedFault(ErrorClass::ProviderPoisoned)
        );

        let proc = Error::ProcessPoisoned("metal context invalid".into());
        assert_eq!(proc.error_class(), ErrorClass::ProcessPoisoned);
        assert!(!proc.allows_gpu_cleanup());
        let cleaned = Cell::new(false);
        let res = run_with_cleanup_policy(|| Err::<(), Error>(proc), || cleaned.set(true));
        assert!(matches!(res, Err(Error::ProcessPoisoned(_))));
        assert!(
            !cleaned.get(),
            "GPU cleanup must NOT run on ProcessPoisoned fault"
        );

        let ok_cleaned = Cell::new(false);
        let (ok_res, ok_outcome) =
            run_with_cleanup_policy_with_outcome(|| Ok::<usize, Error>(7), || ok_cleaned.set(true));
        assert_eq!(ok_res.unwrap(), 7);
        assert!(!ok_cleaned.get());
        assert_eq!(ok_outcome, CleanupPolicyOutcome::Succeeded);
    }
}
