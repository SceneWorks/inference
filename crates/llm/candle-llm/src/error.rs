//! Engine error type.
//!
//! Re-exported from [`candle_layers::llm_projection`] so projections in `candle-layers` and the
//! LLM decoders share one error and result identity (`SW-6957` / `SW-F335`); providers bridge into
//! the backend-neutral [`core_llm::Error`] at the [`crate::provider`] trait boundary via
//! `to_core`.

pub use candle_layers::llm_projection::{Error, Result};

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
