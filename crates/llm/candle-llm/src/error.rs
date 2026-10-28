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

    /// A requested hardware capability is unavailable on the load device — kept typed so the
    /// refusal carries *which* capability and why (sc-24135: NVFP4 below sm_120 or on CPU). Maps to
    /// the contract's `Unsupported`, never to a silent fallback.
    #[error("unsupported: {0}")]
    Nvfp4Refused(#[from] candle_quant_kernels::Nvfp4Refusal),

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

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability /
/// load variants (do not stringify those). The one bridge every Candle provider uses (sc-27300), so a
/// capability refusal or a capacity bound surfaces as the same contract error from every provider.
pub(crate) fn to_core(e: crate::Error) -> core_llm::Error {
    match e {
        crate::Error::Canceled => core_llm::Error::Canceled,
        crate::Error::Unsupported(m) => core_llm::Error::Unsupported(m),
        // The typed capability refusal (sc-24135) is the contract's `Unsupported`, verbatim.
        crate::Error::Nvfp4Refused(r) => core_llm::Error::Unsupported(r.to_string()),
        crate::Error::MissingTensor(m) => core_llm::Error::Load(format!("missing tensor: {m}")),
        crate::Error::Config(m) => core_llm::Error::Load(m),
        crate::Error::Io(e) => core_llm::Error::Io(e),
        // A static KV cache's capacity bound is a request-level refusal, never an opaque backend
        // failure. A request's decode maps it with the request's geometry
        // (`LlamaProvider::request_error`: `RequestResourceExhausted`); without one it is still
        // the request's fault.
        e @ crate::Error::KvCapacityExceeded { .. } => {
            core_llm::Error::InvalidRequest(e.to_string())
        }
        other => core_llm::Error::backend(other),
    }
}

/// The shared tensor-free config parser's two refusals keep their typed variants (sc-27255).
impl From<core_llm::model_config::ModelConfigError> for Error {
    fn from(e: core_llm::model_config::ModelConfigError) -> Self {
        match e {
            core_llm::model_config::ModelConfigError::Config(m) => Error::Config(m),
            core_llm::model_config::ModelConfigError::Unsupported(m) => Error::Unsupported(m),
        }
    }
}

/// A host image-preprocessing refusal (sc-27255) keeps its message.
impl From<core_llm::vision_preprocess::PreprocessError> for Error {
    fn from(e: core_llm::vision_preprocess::PreprocessError) -> Self {
        Error::Msg(e.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-27300: every provider's errors cross into the contract through this one bridge, so a
    /// capacity bound is a request refusal and a missing tensor names itself, whichever provider
    /// (LLaVA, StarVector, the Llama provider) raised it.
    #[test]
    fn the_shared_bridge_keeps_typed_refusals() {
        let capacity = to_core(Error::KvCapacityExceeded {
            requested: 9,
            capacity: 8,
        });
        assert!(
            matches!(capacity, core_llm::Error::InvalidRequest(_)),
            "{capacity:?}"
        );
        match to_core(Error::MissingTensor("lm_head.weight".into())) {
            core_llm::Error::Load(message) => assert_eq!(message, "missing tensor: lm_head.weight"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            to_core(Error::Canceled),
            core_llm::Error::Canceled
        ));
    }
}
