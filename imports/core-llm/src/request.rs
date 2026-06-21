//! The request model, sampling policy, and provider load spec.

use crate::cancel::CancelFlag;
use crate::constraint::Constraint;
use crate::message::Message;

/// Backend-neutral sampling policy. The backend's sampler consumes these knobs; `core-llm` owns the
/// policy so it is identical across MLX and Candle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// Softmax temperature; `<= 0` ⇒ greedy.
    pub temperature: f32,
    /// Nucleus (top-p) threshold in `(0, 1]`; `>= 1` disables it.
    pub top_p: f32,
    /// Keep only the `top_k` highest-logit tokens; `0` disables it.
    pub top_k: usize,
    /// CTRL/HF repetition penalty; `1.0` disables it.
    pub repetition_penalty: f32,
    /// History window the repetition penalty looks back over.
    pub repetition_context: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        // Mild defaults suitable for chat; callers override per request.
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        }
    }
}

impl Sampling {
    /// Deterministic greedy decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        }
    }

    /// Whether these knobs describe greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// A request to generate text.
///
/// Cancellation is in-band on [`TextLlmRequest::cancel`]; an already-cancelled request must error
/// before inference (see [`crate::cancel`]).
#[derive(Clone, Debug, Default)]
pub struct TextLlmRequest {
    /// The conversation so far (system / user / assistant / tool turns, text and images).
    pub messages: Vec<Message>,
    /// Sampling policy.
    pub sampling: Sampling,
    /// Maximum new tokens to generate.
    pub max_new_tokens: u32,
    /// RNG seed; `None` ⇒ a fresh per-call seed (non-reproducible). Greedy is seed-independent.
    pub seed: Option<u64>,
    /// Optional output constraint (e.g. valid JSON).
    pub constraint: Option<Constraint>,
    /// Extra stop strings (beyond the model's own EOS tokens).
    pub stop: Vec<String>,
    /// Cooperative cancellation handle.
    pub cancel: CancelFlag,
}

impl TextLlmRequest {
    /// A simple single-user-turn text request with greedy defaults aside from the given sampling.
    pub fn new(messages: Vec<Message>, max_new_tokens: u32) -> Self {
        Self {
            messages,
            max_new_tokens,
            ..Default::default()
        }
    }

    /// Whether any message carries image content (vision input).
    pub fn has_image(&self) -> bool {
        self.messages.iter().any(crate::message::Message::has_image)
    }
}

/// How a provider should load a model. Backend-neutral: the provider interprets `source` (a
/// snapshot directory path or a model id) and applies any load-time quantization.
#[derive(Clone, Debug, Default)]
pub struct LoadSpec {
    /// A snapshot directory path or a model identifier the provider understands.
    pub source: String,
    /// Optional load-time quantization.
    pub quantize: Option<Quantize>,
}

/// Load-time quantization request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantize {
    /// 4-bit group-wise affine.
    Q4,
    /// 8-bit group-wise affine.
    Q8,
}

impl LoadSpec {
    /// A dense (non-quantized) load from `source`.
    pub fn dense(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            quantize: None,
        }
    }
}
