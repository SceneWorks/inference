//! Provider identity and declared capabilities.

use crate::constraint::Constraint;
use crate::error::{Error, Result};
use crate::request::TextLlmRequest;

/// What a provider supports. Used both for honest advertisement and to validate requests up front.
#[derive(Clone, Debug, Default)]
pub struct TextLlmCapabilities {
    /// Maximum context length in tokens (`0` = unspecified / unbounded).
    pub max_context_tokens: usize,
    /// Maximum tokens that may be requested per generation (`0` = unspecified).
    pub max_new_tokens: u32,
    /// Whether a system turn is honored.
    pub supports_system_prompt: bool,
    /// Whether image (vision) content is accepted.
    pub supports_vision: bool,
    /// The output constraints this provider can enforce (empty = none).
    pub supported_constraints: Vec<Constraint>,
}

impl TextLlmCapabilities {
    /// Whether a given constraint is supported.
    pub fn supports_constraint(&self, c: Constraint) -> bool {
        self.supported_constraints.contains(&c)
    }

    /// Validate a request against these capabilities. Providers call this from
    /// [`TextLlm::validate`](crate::TextLlm::validate). Rejects (rather than silently ignoring)
    /// anything outside the declared surface.
    pub fn validate_request(&self, id: &str, req: &TextLlmRequest) -> Result<()> {
        let reject = |msg: String| Err(Error::InvalidRequest(format!("[{id}] {msg}")));

        if req.messages.is_empty() {
            return reject("request has no messages".into());
        }
        if req.messages.iter().all(|m| m.text_content().trim().is_empty() && !m.has_image()) {
            return reject("request has no non-empty content".into());
        }

        if !self.supports_system_prompt
            && req
                .messages
                .iter()
                .any(|m| m.role == crate::message::Role::System)
        {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support a system prompt"
            )));
        }

        if !self.supports_vision && req.has_image() {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support image (vision) input"
            )));
        }

        if let Some(c) = req.constraint {
            if !self.supports_constraint(c) {
                return Err(Error::Unsupported(format!(
                    "[{id}] provider does not support the {c:?} constraint"
                )));
            }
        }

        let s = &req.sampling;
        if !(0.0..=2.0).contains(&s.temperature) {
            return reject(format!("temperature {} out of [0, 2]", s.temperature));
        }
        if !(0.0..=1.0).contains(&s.top_p) {
            return reject(format!("top_p {} out of [0, 1]", s.top_p));
        }
        if req.max_new_tokens == 0 {
            return reject("max_new_tokens must be >= 1".into());
        }
        if self.max_new_tokens > 0 && req.max_new_tokens > self.max_new_tokens {
            return reject(format!(
                "max_new_tokens {} exceeds cap {}",
                req.max_new_tokens, self.max_new_tokens
            ));
        }
        Ok(())
    }
}

/// A provider's identity plus its capabilities.
#[derive(Clone, Debug)]
pub struct TextLlmDescriptor {
    /// Unique provider id used for registry routing (e.g. `"mlx-llama"`).
    pub id: String,
    /// Model family (e.g. `"llama"`, `"qwen3"`).
    pub family: String,
    /// Tensor backend tag (`"mlx"` | `"candle"`).
    pub backend: String,
    /// Declared capabilities.
    pub capabilities: TextLlmCapabilities,
}
