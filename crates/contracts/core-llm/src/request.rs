//! The request model, sampling policy, and provider load spec.

use crate::cancel::CancelFlag;
use crate::constraint::Constraint;
use crate::message::Message;

/// Qwen3.8 reasoning budgets accepted by its frozen official chat template.
///
/// The spellings deliberately match the template kwargs. Keeping this typed prevents callers from
/// passing a value the model would otherwise reject only while rendering the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// The template default and most thorough reasoning budget.
    XHigh,
    /// The template's middle reasoning budget (no extra budget instruction is injected).
    Medium,
    /// The brief, focused reasoning budget.
    Low,
}

impl ReasoningEffort {
    /// The exact `reasoning_effort` chat-template kwarg.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::XHigh => "xhigh",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }
}

/// Request policy for an in-checkpoint multi-token predictor (MTP).
///
/// MTP is an optional speculative decoder: the base autoregressive distribution remains valid
/// without it, while enabled runs use target-model verification to preserve that distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpMode {
    /// Use the ordinary autoregressive path. This is the compatibility-preserving default.
    #[default]
    Off,
    /// Use MTP when the loaded model advertises it, otherwise use ordinary autoregressive decode.
    Auto,
    /// Require MTP and propose at most `draft_tokens` tokens per target verification pass.
    Enabled {
        /// Number of speculative draft tokens. Must be within the provider's advertised limit.
        draft_tokens: u32,
    },
}

impl std::str::FromStr for ReasoningEffort {
    type Err = crate::Error;

    fn from_str(value: &str) -> crate::Result<Self> {
        match value {
            "xhigh" => Ok(Self::XHigh),
            "medium" => Ok(Self::Medium),
            "low" => Ok(Self::Low),
            unsupported => Err(crate::Error::InvalidRequest(format!(
                "unsupported reasoning_effort `{unsupported}`; expected xhigh, medium, or low"
            ))),
        }
    }
}

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
    /// OpenAI/HF presence penalty: subtract this value once from every token logit whose token has
    /// appeared in the prompt or generated history. `0.0` disables it. Unlike
    /// [`repetition_penalty`](Self::repetition_penalty), this is additive and independent of count.
    pub presence_penalty: f32,
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
            presence_penalty: 0.0,
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
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        }
    }

    /// Whether these knobs describe greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Whether a history-dependent logit penalty (repetition or presence) is requested. Penalties
    /// read the token history, so a backend applies them on the host.
    pub fn is_penalized(&self) -> bool {
        self.repetition_penalty != 1.0 || self.presence_penalty != 0.0
    }

    /// The backend-neutral sampler routing policy (epic sc-24128, story sc-24133): which path a
    /// backend should draw this request's tokens on, given whether a constraint (grammar / JSON
    /// mask) is active. Temperature, top-k and top-p are expressible on the accelerator; a
    /// constraint mask or a history penalty is not, so those requests take the host path with the
    /// reason named. A backend may still route a [`SamplerPath::Device`] request to the host for a
    /// backend reason ([`HostSampleReason::DeviceUnavailable`]) — never silently: the chosen path is
    /// what it reports.
    pub fn sampler_path(&self, constrained: bool) -> SamplerPath {
        if constrained {
            SamplerPath::Host(HostSampleReason::Constraint)
        } else if self.is_penalized() {
            SamplerPath::Host(HostSampleReason::Penalty)
        } else {
            SamplerPath::Device
        }
    }
}

/// Where a backend drew a token (epic sc-24128, story sc-24133). Telemetry, never a silent
/// downgrade: a request that ran on the host says so and says why.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplerPath {
    /// Sampled on the accelerator; only the chosen token id crosses to the host.
    Device,
    /// The logits row was copied to the host and sampled there.
    Host(HostSampleReason),
}

impl SamplerPath {
    /// `"device"` or `"host"`.
    pub fn label(&self) -> &'static str {
        match self {
            SamplerPath::Device => "device",
            SamplerPath::Host(_) => "host",
        }
    }

    /// Why the host path ran, or `None` on the device path.
    pub fn host_reason(&self) -> Option<HostSampleReason> {
        match self {
            SamplerPath::Device => None,
            SamplerPath::Host(reason) => Some(*reason),
        }
    }
}

impl std::fmt::Display for SamplerPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerPath::Device => f.write_str("device"),
            SamplerPath::Host(reason) => write!(f, "host:{}", reason.label()),
        }
    }
}

/// Why a token was sampled on the host rather than on the accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostSampleReason {
    /// A repetition or presence penalty reads the token history.
    Penalty,
    /// A grammar / JSON constraint mask is applied on the host.
    Constraint,
    /// The backend has no device sampler for this device (CPU, a build without the accelerator
    /// feature, a shape the device kernel does not serve, or a sampler kernel that failed to
    /// compile or load on this device).
    DeviceUnavailable,
    /// The temperature is positive but its reciprocal is not a usable finite scale (a subnormal
    /// temperature, `+inf`, or NaN): the device kernel cannot shape weights from it, so the host
    /// reference handles the request.
    DegenerateTemperature,
    /// Speculative acceptance needs the full shaped distribution on the host.
    SpeculativeDistribution,
    /// The caller forced the host reference sampler (parity checks and benchmarks).
    Reference,
}

impl HostSampleReason {
    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            HostSampleReason::Penalty => "penalty",
            HostSampleReason::Constraint => "constraint",
            HostSampleReason::DeviceUnavailable => "device_unavailable",
            HostSampleReason::DegenerateTemperature => "degenerate_temperature",
            HostSampleReason::SpeculativeDistribution => "speculative_distribution",
            HostSampleReason::Reference => "reference",
        }
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
    /// Optional Qwen `reasoning_effort` passed only to providers advertising
    /// [`supports_reasoning_effort`](crate::TextLlmCapabilities::supports_reasoning_effort).
    /// `None` omits the kwarg and preserves the model's own default (Qwen3.8 resolves that to `xhigh`).
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Qwen `preserve_thinking` control for retaining prior assistant reasoning during history
    /// rendering. Honored only by providers advertising
    /// [`supports_preserve_thinking`](crate::TextLlmCapabilities::supports_preserve_thinking).
    /// `None` omits the kwarg and preserves the model's default (Qwen3.8 defaults to `true`).
    pub preserve_thinking: Option<bool>,
    /// Optional in-checkpoint multi-token prediction policy. [`MtpMode::Off`] preserves the ordinary
    /// autoregressive path; explicit enablement is rejected unless the loaded model advertises MTP.
    pub mtp: MtpMode,
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
    /// A request over the given messages with the default sampling policy (temperature `0.7`, top-p
    /// `0.9`), which is stochastic rather than greedy.
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

    /// Whether any message carries video content.
    pub fn has_video(&self) -> bool {
        self.messages.iter().any(crate::message::Message::has_video)
    }

    /// Whether any message carries audio content.
    pub fn has_audio(&self) -> bool {
        self.messages.iter().any(crate::message::Message::has_audio)
    }

    /// The `enable_thinking` chat-template kwarg for this request's [`thinking`](Self::thinking)
    /// mode (`None` ⇒ omit it / use the template default). Feed into
    /// [`RenderOptions`](crate::template::RenderOptions).
    pub fn enable_thinking_kwarg(&self) -> Option<bool> {
        self.thinking.enable_thinking_kwarg()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sampler_path_policy_routes_penalties_and_constraints_to_host() {
        let mut s = Sampling {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            ..Sampling::default()
        };
        assert_eq!(s.sampler_path(false), SamplerPath::Device);
        assert_eq!(
            s.sampler_path(true),
            SamplerPath::Host(HostSampleReason::Constraint)
        );
        assert_eq!(Sampling::greedy().sampler_path(false), SamplerPath::Device);
        s.presence_penalty = 0.5;
        assert_eq!(
            s.sampler_path(false),
            SamplerPath::Host(HostSampleReason::Penalty)
        );
        s.presence_penalty = 0.0;
        s.repetition_penalty = 1.1;
        assert_eq!(
            s.sampler_path(false).to_string(),
            "host:penalty",
            "telemetry names the reason"
        );
        assert_eq!(
            s.sampler_path(true),
            SamplerPath::Host(HostSampleReason::Constraint),
            "a constraint outranks a penalty"
        );
        assert_eq!(SamplerPath::Device.to_string(), "device");
        assert_eq!(SamplerPath::Device.host_reason(), None);
        assert_eq!(
            SamplerPath::Host(HostSampleReason::DeviceUnavailable).label(),
            "host"
        );
        assert_eq!(
            SamplerPath::Host(HostSampleReason::DegenerateTemperature).to_string(),
            "host:degenerate_temperature"
        );
    }

    use super::*;

    #[test]
    fn new_uses_the_documented_default_sampling() {
        let request = TextLlmRequest::new(Vec::new(), 32);

        assert_eq!(request.sampling.temperature, 0.7);
        assert_eq!(request.sampling.top_p, 0.9);
        assert_eq!(request.sampling.presence_penalty, 0.0);
        assert!(!request.sampling.is_greedy());
        assert_eq!(request.reasoning_effort, None);
        assert_eq!(request.preserve_thinking, None);
        assert_eq!(request.mtp, MtpMode::Off);
    }

    #[test]
    fn reasoning_effort_accepts_only_the_frozen_qwen38_values() {
        use std::str::FromStr;

        assert_eq!(
            ReasoningEffort::from_str("xhigh").unwrap(),
            ReasoningEffort::XHigh
        );
        assert_eq!(
            ReasoningEffort::from_str("medium").unwrap(),
            ReasoningEffort::Medium
        );
        assert_eq!(
            ReasoningEffort::from_str("low").unwrap(),
            ReasoningEffort::Low
        );
        let err = ReasoningEffort::from_str("high").unwrap_err();
        assert!(matches!(err, crate::Error::InvalidRequest(_)));
        assert!(err.to_string().contains("expected xhigh, medium, or low"));
    }

    #[test]
    fn projector_association_is_explicit() {
        let spec = LoadSpec::dense("model.gguf").with_projector("projector.gguf");
        assert_eq!(spec.source, "model.gguf");
        assert_eq!(spec.projector_source.as_deref(), Some("projector.gguf"));
        assert!(spec.quantize.is_none());
    }
}

/// How a provider should load a model. Backend-neutral: the provider interprets `source` (a
/// snapshot directory path or a model id) and applies any load-time quantization.
#[derive(Clone, Debug, Default)]
pub struct LoadSpec {
    /// A snapshot directory path or a model identifier the provider understands.
    pub source: String,
    /// Optional explicitly-associated multimodal projector artifact. GGUF language files do not
    /// embed this association and a directory may contain several valid projectors, so providers
    /// must never guess between siblings. `None` keeps a separable model text-only.
    pub projector_source: Option<String>,
    /// Optional load-time **weight** quantization (the model projection weights).
    pub quantize: Option<Quantize>,
}

/// Load-time quantization request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantize {
    /// 4-bit group-wise affine.
    Q4,
    /// 8-bit group-wise affine.
    Q8,
    /// NVFP4 (sc-24135): E2M1 4-bit elements with one FP8-E4M3 scale per 16-element block and an
    /// FP32 per-tensor scale (~4.5 bits/weight), quantized **at load** and served by a native FP4
    /// GEMM. A hardware capability, not a storage format: a provider whose device cannot run it must
    /// refuse the load with [`Error::Unsupported`](crate::Error::Unsupported) naming the capability
    /// rather than fall back to another representation, and snapshot preparation never persists it.
    Nvfp4,
}

impl LoadSpec {
    /// A dense (non-quantized) load from `source`.
    pub fn dense(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            projector_source: None,
            quantize: None,
        }
    }

    /// Associate an exact multimodal projector artifact with this model load.
    pub fn with_projector(mut self, source: impl Into<String>) -> Self {
        self.projector_source = Some(source.into());
        self
    }
}
