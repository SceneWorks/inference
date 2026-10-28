//! Typed iOS admission outcomes. A refusal is a decision with a kind a campaign can record without
//! parsing any message; a failure is a malformed request, unreadable source or provider error.

use mlx_gen::gen_core::Error;

/// Why admission refused, before any weight payload was read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefusalKind {
    /// Evidence exists, but its captured footprint exceeds the cap or the live budget.
    Capacity,
    /// No reviewed device row in the evidence table matches the request's key.
    NoEvidence,
    /// The provider does not serve this request mode (or the mode has no carrier on this route).
    Mode,
    /// The selection is not implemented, or execution does not implement it, or the provider's
    /// registered safety check rejected it for a reason other than capacity.
    Strategy,
    /// The supplied artifact's inventory/layout is outside the admitted shape (overrides, nested
    /// components, symlinked directories, an unknown route).
    Artifact,
}

impl RefusalKind {
    /// The stable label a campaign row records.
    pub fn label(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::NoEvidence => "no_evidence",
            Self::Mode => "mode",
            Self::Strategy => "strategy",
            Self::Artifact => "artifact",
        }
    }
}

/// A typed refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub reason: String,
}

/// The outcome of an iOS admission attempt that did not admit.
#[derive(Debug)]
pub enum AdmissionError {
    Refused(Refusal),
    Failed(Error),
}

impl AdmissionError {
    pub fn refused(kind: RefusalKind, reason: impl Into<String>) -> Self {
        Self::Refused(Refusal {
            kind,
            reason: reason.into(),
        })
    }

    /// The refusal kind, when this is a refusal.
    pub fn refusal_kind(&self) -> Option<RefusalKind> {
        match self {
            Self::Refused(refusal) => Some(refusal.kind),
            Self::Failed(_) => None,
        }
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(r) => write!(
                f,
                "iOS admission refused ({}): {}",
                r.kind.label(),
                r.reason
            ),
            Self::Failed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AdmissionError {}

impl From<Error> for AdmissionError {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

impl From<AdmissionError> for Error {
    fn from(error: AdmissionError) -> Self {
        match error {
            AdmissionError::Refused(_) => Error::Unsupported(error.to_string()),
            AdmissionError::Failed(error) => error,
        }
    }
}

pub type AdmissionResult<T> = std::result::Result<T, AdmissionError>;

/// An artifact-policy result: a typed `Unsupported` from an inventory/layout check is an artifact
/// refusal; any other error (I/O, malformed header) is a failure, never a refusal.
pub(crate) fn artifact<T>(result: mlx_gen::gen_core::Result<T>) -> AdmissionResult<T> {
    result.map_err(|error| match error {
        Error::Unsupported(reason) => AdmissionError::refused(RefusalKind::Artifact, reason),
        other => AdmissionError::Failed(other),
    })
}
