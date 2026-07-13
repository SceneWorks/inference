//! Streaming events and the generation result.

/// Token accounting for a generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the (rendered + tokenized) prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub generated_tokens: u32,
}

impl Usage {
    /// Total tokens processed (prompt + generated).
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.generated_tokens
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop / EOS token (or a stop string) ended generation.
    Stop,
    /// The `max_new_tokens` budget was reached.
    Length,
    /// Cancellation tripped mid-stream (output may be partial).
    Cancelled,
    /// A content filter halted generation.
    ContentFilter,
}

/// Which channel a streamed token belongs to: the model's reasoning trace or its final answer.
///
/// Only reasoning ("thinking") models emit [`Thinking`](Channel::Content) spans; every token from a
/// non-thinking model — and every token outside a `<think>…</think>` block — is
/// [`Content`](Channel::Content). The provider classifies tokens with a
/// [`ThinkingSegmenter`](crate::ThinkingSegmenter).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Channel {
    /// Part of the final answer.
    #[default]
    Content,
    /// Part of the model's reasoning trace (the contents of a `<think>…</think>` block); the
    /// markers themselves are stripped and not emitted.
    Thinking,
}

/// An event emitted as a generation streams.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// A newly generated token and the incremental text it decodes to.
    Token {
        /// The token id.
        id: u32,
        /// The incremental decoded text contributed by this token (may be empty for a token that
        /// only completes a multi-byte character).
        text: String,
        /// 0-based index among generated tokens.
        index: usize,
        /// Whether this token is reasoning or final-answer text. Always
        /// [`Channel::Content`] for non-thinking providers.
        channel: Channel,
    },
    /// Terminal event: generation finished.
    Done {
        /// Why it stopped.
        finish_reason: FinishReason,
        /// Final usage.
        usage: Usage,
    },
}

/// The result of a generation (also recoverable by accumulating [`StreamEvent::Token`] text).
///
/// Not `Eq` (only `PartialEq`): a [`tool_calls`](Self::tool_calls) argument is a `serde_json::Value`,
/// which is `PartialEq` but not `Eq` (it can hold a float).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextLlmOutput {
    /// The final answer — reasoning markers (`<think>…</think>`) and tool-call blocks
    /// (`<tool_call>…</tool_call>`) excluded. Recoverable by accumulating the [`Channel::Content`]
    /// token deltas.
    pub text: String,
    /// The model's reasoning trace, when it produced one: the concatenated
    /// [`Channel::Thinking`] text (markers stripped). `None` for a non-thinking run, a no-think
    /// request, or a model with no reasoning mode. Mirrors OpenAI's `reasoning_content` vs
    /// `content` split.
    pub thinking: Option<String>,
    /// Tool / function calls the model emitted, parsed from its `<tool_call>` blocks (empty if none,
    /// or if the request offered no tools). The raw call markup is excluded from [`text`](Self::text)
    /// — it is structure, not answer content. Carry these back on an assistant
    /// [`Message::tool_calls`](crate::Message::tool_calls), paired with the tool result turn(s), to
    /// continue a multi-step tool exchange.
    pub tool_calls: Vec<crate::tool::ToolCall>,
    /// Token usage.
    pub usage: Usage,
    /// Why generation stopped (`None` only on a default-constructed value).
    pub finish_reason: Option<FinishReason>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_total() {
        let u = Usage {
            prompt_tokens: 10,
            generated_tokens: 5,
        };
        assert_eq!(u.total_tokens(), 15);
    }
}
