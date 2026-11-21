//! Engine error type.
//!
//! Re-exported from [`mlx_layers::llm_projection`] so projections in `mlx-layers` and the LLM
//! decoders share one error and result identity (`SW-6957` / `SW-F335`); providers bridge into the
//! backend-neutral [`core_llm::Error`] at the trait boundary via `to_core`.

pub use mlx_layers::llm_projection::{Error, Result};

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
