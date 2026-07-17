//! JSON-constrained decoding — delegated to `core-llm` (sc-12467).
//!
//! This module once held its own ~450-line incremental JSON-prefix validator (sc-6585); that
//! implementation was token-for-token identical to the copy `core-llm` ported for its constrained
//! decoding (`core_llm::constraint`), so the duplicate was collapsed and this module now re-exports
//! the single live implementation. The public path is unchanged: `gen_core::JsonState` and
//! `gen_core::json_constraint::JsonState` keep resolving, with identical construction
//! ([`JsonState::start`]), advancing ([`JsonState::advance`]), and stop gating
//! ([`JsonState::can_stop`]) semantics.
//!
//! Compatibility note for external consumers: `gen_core::JsonState` is now nominally
//! `core_llm::JsonState` (reachable here and via `gen_core::core_llm`). The state-machine grammar
//! is unchanged — the two copies were identical when collapsed.

pub use core_llm::JsonState;

#[cfg(test)]
mod tests {
    use super::*;

    /// A string that is COMPLETE valid JSON via the re-exported machine: valid prefix AND can_stop.
    fn is_complete(s: &str) -> bool {
        JsonState::start()
            .advance(s)
            .map(JsonState::can_stop)
            .unwrap_or(false)
    }

    /// Smoke-pins the re-export: `gen_core::json_constraint::JsonState` resolves and behaves like
    /// the historical gen-core machine (full grammar coverage lives in `core_llm::constraint`).
    #[test]
    fn reexported_state_machine_validates_prefixes() {
        assert!(is_complete("{\"a\": [1, true, null]}"));
        assert!(JsonState::start().advance("{\"unfinished").is_some());
        assert!(!is_complete("{\"unfinished"));
        assert!(JsonState::start().advance("{}trailing").is_none());
    }

    /// The gen-core-specific shape sc-6585 targeted: an Ideogram caption-style nested object.
    #[test]
    fn caption_shaped_object_round_trips() {
        let caption = "{\"high_level_description\": \"A red fox.\", \
             \"compositional_deconstruction\": {\"background\": \"snow\", \
             \"elements\": [{\"type\": \"obj\", \"desc\": \"fox\"}]}}";
        assert!(is_complete(caption));
        assert!(serde_json::from_str::<serde_json::Value>(caption).is_ok());
    }
}
