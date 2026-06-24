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

/// Whether a model's reasoning ("thinking") mode is requested for a generation.
///
/// Reasoning models (e.g. Qwen3) gate an internal `<think>…</think>` chain on an `enable_thinking`
/// chat-template kwarg. This enum is the backend-neutral control: it maps 1:1 to the
/// `transformers` `chat_template_kwargs={"enable_thinking": …}` semantics via
/// [`enable_thinking_kwarg`](TextLlmRequest::enable_thinking_kwarg). A provider only honors it when
/// it advertises [`supports_thinking`](crate::TextLlmCapabilities::supports_thinking).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingMode {
    /// Use the model/template default — omit the kwarg entirely (the template decides).
    #[default]
    Auto,
    /// Request reasoning **on** (`enable_thinking=true`).
    Enabled,
    /// Request reasoning **off** — "no-think" (`enable_thinking=false`).
    Disabled,
}

impl ThinkingMode {
    /// The `enable_thinking` chat-template kwarg this mode maps to: `None` for [`Auto`](Self::Auto)
    /// (omit it, so the template's `is defined` test is false), else `Some(bool)`.
    pub fn enable_thinking_kwarg(self) -> Option<bool> {
        match self {
            ThinkingMode::Auto => None,
            ThinkingMode::Enabled => Some(true),
            ThinkingMode::Disabled => Some(false),
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
    /// Reasoning ("thinking") mode. Honored only by providers advertising
    /// [`supports_thinking`](crate::TextLlmCapabilities::supports_thinking); [`ThinkingMode::Auto`]
    /// (the default) leaves the model's template default in place.
    pub thinking: ThinkingMode,
    /// Tools / functions offered to the model (matching `transformers` `tools=`). Rendered into the
    /// prompt by the chat template and used to type-coerce the model's parsed tool calls. Honored only
    /// by providers advertising [`supports_tools`](crate::TextLlmCapabilities::supports_tools); a
    /// non-empty `tools` on a provider without that capability is rejected by
    /// [`validate`](crate::TextLlm::validate). Empty ⇒ no tool section, behavior unchanged.
    pub tools: Vec<crate::tool::ToolSpec>,
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

    /// The `enable_thinking` chat-template kwarg for this request's [`thinking`](Self::thinking)
    /// mode (`None` ⇒ omit it / use the template default). Feed into
    /// [`RenderOptions`](crate::template::RenderOptions).
    pub fn enable_thinking_kwarg(&self) -> Option<bool> {
        self.thinking.enable_thinking_kwarg()
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
