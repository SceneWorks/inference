//! Engine error type.
//!
//! `candle-llm` owns its own error enum rather than reusing `core_llm::Error`: the engine and its
//! tensor primitives surface Candle errors, IO, and config problems internally; the provider bridges
//! into the backend-neutral contract error at the [`crate::provider`] trait boundary, preserving the
//! typed cancellation / capability / load variants.

use thiserror::Error;

/// Errors surfaced by the engine and its tensor primitives.
#[derive(Debug, Error)]
pub enum Error {
    /// A Candle tensor operation failed (shape mismatch, allocation, device error, …).
    #[error("candle op failed: {0}")]
    Candle(#[from] candle_core::Error),

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

    /// A cache was asked to roll back to position `n` but holds no checkpoint there (a recurrent
    /// state cannot be inverted, so it refuses rather than approximate). `have` lists the positions
    /// it could return to besides `0` and its current length. Typed so a speculative engine can
    /// tell "no checkpoint" apart from a real failure and fall back (e.g. restore a clone).
    #[error(
        "no checkpoint at position {n} (have {have:?}); the recurrent state cannot be rolled back \
         without one"
    )]
    RollbackUnavailable { n: i32, have: Vec<i32> },

    /// A preallocated (static) KV cache was asked for more positions than it can hold — either at
    /// construction (`requested` capacity past the model's `max_position_embeddings`, reported as
    /// `capacity`) or at a step (`requested` = the position the step would end at, past the
    /// buffer's `capacity`). Raised **before** any device write, so the cache is untouched and the
    /// request fails closed instead of tripping an allocator OOM mid-decode (E6). Typed so admission
    /// and the decode drivers can tell a capacity bound from a real failure.
    #[error(
        "static KV cache capacity exceeded: {requested} positions requested, capacity {capacity}"
    )]
    KvCapacityExceeded { requested: usize, capacity: usize },

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
