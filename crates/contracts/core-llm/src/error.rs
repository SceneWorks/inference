//! The contract error type.
//!
//! Backend-neutral: it names no tensor library. The `Canceled` and `Unsupported` variants are kept
//! typed deliberately so consumers (and the conformance suite) can distinguish cancellation and a
//! capability gap from a generic failure — do not stringify those into [`Error::Msg`].

use thiserror::Error;

/// Exact, backend-neutral evidence for request admission that failed before native allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestResourceExhausted {
    /// Actual prompt tokens after template rendering and visual expansion.
    pub prompt_tokens: usize,
    /// Requested generation budget.
    pub max_new_tokens: u32,
    /// Provider-advertised architectural context limit.
    pub max_context_tokens: usize,
    /// Checked native workspace estimate for this request.
    pub required_bytes: u64,
    /// Fresh observed capacity after applying the operational cap.
    pub available_bytes: u64,
}

impl std::fmt::Display for RequestResourceExhausted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "request requires an estimated {} bytes of native workspace but only {} bytes are available; reduce prompt/media length or max_new_tokens",
            self.required_bytes, self.available_bytes
        )
    }
}

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

    /// An architecturally valid request exceeded the measured preallocation memory budget.
    #[error("invalid request: {0}")]
    RequestResourceExhausted(RequestResourceExhausted),

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
