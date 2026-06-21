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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextLlmOutput {
    /// The full generated text.
    pub text: String,
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
