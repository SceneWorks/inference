//! The contract error type.
//!
//! Backend-neutral: it names no tensor library. The `Canceled` and `Unsupported` variants are kept
//! typed deliberately so consumers (and the conformance suite) can distinguish cancellation and a
//! capability gap from a generic failure — do not stringify those into [`Error::Msg`].

use thiserror::Error;

/// Errors surfaced across the contract.
#[derive(Debug, Error)]
pub enum Error {
    /// A backend (tensor engine, transport, …) operation failed. Boxed so the contract names no
    /// concrete backend type.
    #[error("backend error: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),

    /// The model could not be loaded (missing file, bad checkpoint, unreadable config).
    #[error("model load error: {0}")]
    Load(String),

    /// The request was invalid for this provider (out-of-bounds knob, unsupported field, …).
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// A requested capability is genuinely unsupported (keep typed; do not stringify).
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Generation was cancelled before any output (keep typed; do not stringify).
    #[error("cancelled")]
    Canceled,

    /// Filesystem / IO failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Anything else, with a human-readable message.
    #[error("{0}")]
    Msg(String),
}

/// Contract result alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Box a concrete backend error into [`Error::Backend`].
    pub fn backend<E: std::error::Error + Send + Sync + 'static>(e: E) -> Self {
        Error::Backend(Box::new(e))
    }
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
