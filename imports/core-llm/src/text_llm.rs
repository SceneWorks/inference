//! The provider trait — the heart of the contract.

use crate::capabilities::TextLlmDescriptor;
use crate::error::Result;
use crate::output::{StreamEvent, TextLlmOutput};
use crate::request::TextLlmRequest;

/// A streaming, cancellable, multimodal (text + vision) text-LLM provider.
///
/// Implemented by backends (`mlx-llm`, `candle-llm`) and registered through
/// [`crate::registry`]. `&self`: a loaded provider is immutable and shareable.
///
/// Cancellation rides on [`TextLlmRequest::cancel`]: an already-cancelled request must return
/// [`Error::Canceled`](crate::Error::Canceled) before any inference; a mid-stream cancel stops
/// promptly and may return a partial output marked
/// [`FinishReason::Cancelled`](crate::FinishReason::Cancelled).
pub trait TextLlm {
    /// The provider's identity + declared capabilities (constructible without running inference).
    fn descriptor(&self) -> &TextLlmDescriptor;

    /// Cheap, pre-inference validation of a request against this provider's capabilities. Must
    /// reject (not silently ignore) anything outside the declared surface.
    fn validate(&self, req: &TextLlmRequest) -> Result<()>;

    /// Generate, streaming a [`StreamEvent`] per token through `on_event`, and returning the final
    /// [`TextLlmOutput`]. The terminal [`StreamEvent::Done`] carries the same finish reason + usage
    /// as the returned output.
    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TextLlmOutput>;

    /// Convenience: generate without observing the stream (collect the final output only).
    fn complete(&self, req: &TextLlmRequest) -> Result<TextLlmOutput> {
        self.generate(req, &mut |_| {})
    }
}
