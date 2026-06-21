//! Link-time provider registry and id-based routing.
//!
//! Backends register a provider with [`inventory::submit!`]; consumers discover and load providers
//! by id without a central match statement (additive, like the mlx-gen registries). A registration
//! stores the descriptor constructor separately from `load`, so the registry can be introspected
//! cheaply without loading any weights.

use crate::capabilities::TextLlmDescriptor;
use crate::error::{Error, Result};
use crate::request::LoadSpec;
use crate::text_llm::TextLlm;

/// A registered provider: how to describe it (cheap) and how to load it (loads weights).
pub struct TextLlmRegistration {
    /// Build the provider's descriptor without loading weights.
    pub descriptor: fn() -> TextLlmDescriptor,
    /// Load an instance from a [`LoadSpec`].
    pub load: fn(&LoadSpec) -> Result<Box<dyn TextLlm>>,
}

inventory::collect!(TextLlmRegistration);

/// Iterate every registered provider (link-time collected).
pub fn textllms() -> impl Iterator<Item = &'static TextLlmRegistration> {
    inventory::iter::<TextLlmRegistration>.into_iter()
}

/// Look up a registered provider by its descriptor id.
pub fn find(id: &str) -> Option<&'static TextLlmRegistration> {
    textllms().find(|r| (r.descriptor)().id == id)
}

/// Load a provider by id. First-wins on duplicate ids (a `debug_assert!` flags the collision).
pub fn load_textllm(id: &str, spec: &LoadSpec) -> Result<Box<dyn TextLlm>> {
    let mut matches = textllms().filter(|r| (r.descriptor)().id == id);
    let reg = matches
        .next()
        .ok_or_else(|| Error::Msg(format!("no textllm registered for id '{id}'")))?;
    debug_assert!(
        matches.next().is_none(),
        "duplicate textllm id '{id}' registered (first-wins shadows the rest)"
    );
    (reg.load)(spec)
}
