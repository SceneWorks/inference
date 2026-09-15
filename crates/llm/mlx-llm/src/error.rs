//! Engine error type.
//!
//! `mlx-llm` owns its own error enum rather than reusing the gen-ai `gen_core::Error`: the whole
//! point of the inverted architecture is that this crate depends on nothing from the gen-ai side.
//! The backend-neutral `core-llm` contract (story 7154) will define its own error surface and
//! providers bridge into it at the trait boundary.

use thiserror::Error;

/// Errors surfaced by the engine and its tensor primitives.
#[derive(Debug, Error)]
pub enum Error {
    /// An MLX device operation failed (shape mismatch, allocation, command-buffer error, …).
    #[error("mlx op failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// A required weight tensor was absent from the loaded checkpoint.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem / checkpoint IO failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A model config (`config.json`) was malformed or missing a required field.
    #[error("invalid config: {0}")]
    Config(String),

    /// A requested capability is genuinely not supported (kept typed so callers can distinguish a
    /// capability gap from a generic failure — do not stringify into [`Error::Msg`]).
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Generation was cancelled before it could run. Kept typed so the conformance suite and any
    /// consumer can tell cancellation apart from a real error.
    #[error("cancelled")]
    Canceled,

    /// A freshly loaded weight buffer whose GPU view never converged on the bytes the CPU holds
    /// (sc-22414). Typed so a consumer can tell "the host's Metal mapping is lagging" from a
    /// generic failure; see `primitives::coherence`.
    #[error(
        "GPU view of `{name}` ({bytes} bytes) never matched the CPU view after {attempts} reads \
         (cpu checksum {cpu:#x}, gpu checksum {gpu:#x}): the Metal mapping of a freshly loaded \
         buffer is stale — retry the load once the host's write-back pressure subsides (sc-22414)"
    )]
    IncoherentLoad {
        /// The tensor key.
        name: String,
        /// Its byte size.
        bytes: usize,
        /// The CPU-stream checksum (the reference).
        cpu: u64,
        /// The last GPU-stream checksum observed.
        gpu: u64,
        /// GPU reads attempted.
        attempts: u32,
    },

    /// Anything else, with a human-readable message.
    #[error("{0}")]
    Msg(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

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
