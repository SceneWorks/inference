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

    /// A constraint mask allowed no token of a logits row (story sc-24434): there is nothing the
    /// sampler may draw. Typed so a caller can tell a dead-ended constraint from a backend fault;
    /// the sampler never falls back to a forbidden id.
    #[error("the constraint mask allows no token of the {vocab}-entry logits row")]
    NoAllowedToken {
        /// The logits row's width.
        vocab: usize,
    },

    /// A forward would take a DeltaNet checkpoint window past the tokens it was armed to record
    /// (story sc-24435) — more than the speculative width admission priced. Refused before the
    /// forward runs, with the cache untouched; typed so the cap is never mistaken for another
    /// fault.
    #[error(
        "the DeltaNet checkpoint window holds {recorded} tokens and is armed for {max_tokens}: a \
         {requested}-token forward would exceed it"
    )]
    CheckpointWindowFull {
        /// Tokens the window has recorded.
        recorded: i32,
        /// Tokens the refused forward carried.
        requested: i32,
        /// The most the window records.
        max_tokens: i32,
    },
    /// A request's memory admission refused it mid-generation (a compressed KV cache's dense
    /// transition that the fresh memory budget cannot hold, sc-20682). Kept typed so the provider
    /// surfaces core-llm's `RequestResourceExhausted` rather than a backend failure.
    #[error("{0}")]
    ResourceExhausted(core_llm::RequestResourceExhausted),

    /// Anything else, with a human-readable message.
    #[error("{0}")]
    Msg(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability /
/// load / admission variants (do not stringify those). The one bridge every MLX provider uses
/// (sc-27300), so a mid-generation admission refusal surfaces as the same contract error from every
/// provider.
pub(crate) fn to_core(e: Error) -> core_llm::Error {
    match e {
        Error::Canceled => core_llm::Error::Canceled,
        Error::Unsupported(m) => core_llm::Error::Unsupported(m),
        Error::MissingTensor(m) => core_llm::Error::Load(format!("missing tensor: {m}")),
        Error::Config(m) => core_llm::Error::Load(m),
        Error::ResourceExhausted(evidence) => core_llm::Error::RequestResourceExhausted(evidence),
        Error::Io(e) => core_llm::Error::Io(e),
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
    /// missing tensor names itself and a refusal stays typed, whichever provider raised it.
    #[test]
    fn the_shared_bridge_keeps_typed_refusals() {
        match to_core(Error::MissingTensor("lm_head.weight".into())) {
            core_llm::Error::Load(message) => assert_eq!(message, "missing tensor: lm_head.weight"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            to_core(Error::Unsupported("x".into())),
            core_llm::Error::Unsupported(_)
        ));
        assert!(matches!(
            to_core(Error::Canceled),
            core_llm::Error::Canceled
        ));
    }
}
