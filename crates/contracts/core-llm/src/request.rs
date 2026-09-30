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

/// Legacy request policy for an in-checkpoint multi-token predictor (MTP) — the pre-sc-24432
/// form of [`Speculative`], still honoured through [`TextLlmRequest::mtp`]. It maps onto the
/// proposer-agnostic option one-to-one ([`From<MtpMode> for Speculative`](Speculative)): `Off` is
/// `off`, `Auto` is `auto`, and `Enabled { draft_tokens }` is `{proposer: mtp, depth}`.
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

/// A proposal source a speculative request can name (epic sc-24432 E4). The wire and label
/// spellings are `mtp`, `prompt_lookup` and `draft_model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeProposer {
    /// The checkpoint's own multi-token-prediction head.
    Mtp,
    /// Prompt lookup: drafts copied from the most recent earlier occurrence of the context's
    /// trailing n-gram ([`ngram_propose`](crate::speculative::ngram_propose)). Needs no extra
    /// weights, so every backend that runs the speculative engine can offer it.
    PromptLookup,
    /// A separate, smaller draft model sharing the target's vocabulary.
    DraftModel,
}

impl SpeculativeProposer {
    /// Every proposer, in declaration order; [`Speculative::Auto`] considers only `mtp` then
    /// `prompt_lookup` ([`resolve_speculative`](crate::resolve_speculative)).
    pub const ALL: [SpeculativeProposer; 3] = [
        SpeculativeProposer::Mtp,
        SpeculativeProposer::PromptLookup,
        SpeculativeProposer::DraftModel,
    ];

    /// The stable lower-case wire / evidence label (`mtp`, `prompt_lookup`, `draft_model`).
    pub const fn label(self) -> &'static str {
        match self {
            SpeculativeProposer::Mtp => "mtp",
            SpeculativeProposer::PromptLookup => "prompt_lookup",
            SpeculativeProposer::DraftModel => "draft_model",
        }
    }
}

impl std::fmt::Display for SpeculativeProposer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for SpeculativeProposer {
    type Err = crate::Error;

    fn from_str(value: &str) -> crate::Result<Self> {
        SpeculativeProposer::ALL
            .into_iter()
            .find(|p| p.label() == value)
            .ok_or_else(|| {
                crate::Error::InvalidRequest(format!(
                    "unknown speculative proposer `{value}`; expected mtp, prompt_lookup or \
                     draft_model"
                ))
            })
    }
}

/// The one proposer-agnostic speculative-decoding option (epic sc-24432 E4):
/// `off | auto | {proposer: mtp|prompt_lookup|draft_model, depth}`.
///
/// Speculation changes the speed, never the output: greedy output is the plain path's, and a
/// stochastic run keeps the target distribution through exact rejection sampling. What a request
/// actually ran is reported per generation in
/// [`DecodeReport`](crate::DecodeReport) (`proposer`, `draft_tokens`, `fallbacks`).
///
/// **Wire form.** Serializes as `"off"`, `"auto"` or `{"proposer": "<label>", "depth": N}`.
/// Deserialization also accepts the legacy `mtp` request shape (`{"mode": "off" | "auto"}`,
/// `{"mode": "enabled", "draft_tokens": N}`), so a consumer DTO can keep reading an old `mtp` field
/// with `#[serde(alias = "mtp")]` on its `speculative` field: `enabled` maps to
/// `{proposer: mtp, depth: draft_tokens}`, `off` / `auto` to themselves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Speculative {
    /// Plain token-at-a-time decoding.
    #[default]
    Off,
    /// The best proposer the loaded model advertises, at its recommended depth: MTP where the
    /// model has a head, else prompt lookup; plain decoding (with the reason named) when the
    /// backend offers neither ([`resolve_speculative`](crate::speculative::resolve_speculative)).
    Auto,
    /// Exactly this proposer, proposing up to `depth` tokens per target verification pass. A
    /// proposer the model does not advertise, a zero depth, or a depth above the advertised
    /// maximum is refused by [`TextLlmCapabilities::validate_request`](crate::TextLlmCapabilities::validate_request).
    Proposer {
        /// The proposal source.
        proposer: SpeculativeProposer,
        /// Draft tokens per verification pass (`>= 1`).
        depth: u32,
    },
}

impl Speculative {
    /// `{proposer, depth}`.
    pub const fn proposer(proposer: SpeculativeProposer, depth: u32) -> Self {
        Speculative::Proposer { proposer, depth }
    }
}

impl From<MtpMode> for Speculative {
    fn from(mode: MtpMode) -> Self {
        match mode {
            MtpMode::Off => Speculative::Off,
            MtpMode::Auto => Speculative::Auto,
            MtpMode::Enabled { draft_tokens } => {
                Speculative::proposer(SpeculativeProposer::Mtp, draft_tokens)
            }
        }
    }
}

impl serde::Serialize for Speculative {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        match self {
            Speculative::Off => serializer.serialize_str("off"),
            Speculative::Auto => serializer.serialize_str("auto"),
            Speculative::Proposer { proposer, depth } => {
                let mut s = serializer.serialize_struct("Speculative", 2)?;
                s.serialize_field("proposer", proposer)?;
                s.serialize_field("depth", depth)?;
                s.end()
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for Speculative {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Speculative::from_wire(&value).map_err(serde::de::Error::custom)
    }
}

impl Speculative {
    /// Parse one wire value (see the type docs), naming what was wrong on a refusal.
    fn from_wire(value: &serde_json::Value) -> std::result::Result<Self, String> {
        use serde_json::Value;
        const EXPECTED: &str = "expected \"off\", \"auto\", {\"proposer\": \"mtp\" | \
                                \"prompt_lookup\" | \"draft_model\", \"depth\": N} or the legacy \
                                mtp form {\"mode\": \"off\" | \"auto\" | \"enabled\", \
                                \"draft_tokens\": N}";
        let mode = |s: &str| match s {
            "off" => Ok(Speculative::Off),
            "auto" => Ok(Speculative::Auto),
            other => Err(format!("unknown speculative mode `{other}`; {EXPECTED}")),
        };
        let depth = |v: Option<&Value>, key: &str| -> std::result::Result<u32, String> {
            let v = v.ok_or_else(|| format!("speculative `{key}` is missing"))?;
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("speculative `{key}` must be an unsigned 32-bit integer"))
        };
        let only = |map: &serde_json::Map<String, Value>, keys: &[&str]| match map
            .keys()
            .find(|k| !keys.contains(&k.as_str()))
        {
            Some(extra) => Err(format!(
                "unexpected speculative field `{extra}`; {EXPECTED}"
            )),
            None => Ok(()),
        };
        match value {
            Value::String(s) => mode(s),
            Value::Object(map) if map.contains_key("proposer") => {
                only(map, &["proposer", "depth"])?;
                let proposer = map["proposer"]
                    .as_str()
                    .ok_or_else(|| "speculative `proposer` must be a string".to_string())?
                    .parse::<SpeculativeProposer>()
                    .map_err(|e| e.to_string())?;
                Ok(Speculative::proposer(
                    proposer,
                    depth(map.get("depth"), "depth")?,
                ))
            }
            Value::Object(map) if map.contains_key("mode") => {
                // The legacy `mtp` request shape.
                match map["mode"].as_str() {
                    Some("enabled") => {
                        only(map, &["mode", "draft_tokens"])?;
                        Ok(Speculative::proposer(
                            SpeculativeProposer::Mtp,
                            depth(map.get("draft_tokens"), "draft_tokens")?,
                        ))
                    }
                    Some(other) => {
                        only(map, &["mode"])?;
                        mode(other)
                    }
                    None => Err(format!("speculative `mode` must be a string; {EXPECTED}")),
                }
            }
            _ => Err(format!("invalid speculative option; {EXPECTED}")),
        }
    }
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
    /// appeared in the prompt or generated history. `0.0` disables it. Like
    /// [`repetition_penalty`](Self::repetition_penalty) it applies once per distinct id; it differs
    /// in being additive and in covering the full history rather than a recent window.
    pub presence_penalty: f32,
    /// CTRL/HF repetition penalty; `1.0` disables it. Multiplicative (a positive logit is divided by
    /// it, a negative one multiplied) and applied once per distinct id among the last
    /// [`repetition_context`](Self::repetition_context) tokens, however often the id occurs there.
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
    /// The speculative-decoding option (epic sc-24432 E4). `None` (the default) defers to the
    /// legacy [`mtp`](Self::mtp) field; `Some` is the request's explicit choice, including an
    /// explicit `Some(Speculative::Off)`. Read the effective option through
    /// [`speculative_mode`](Self::speculative_mode).
    pub speculative: Option<Speculative>,
    /// Legacy in-checkpoint multi-token prediction policy, kept so pre-sc-24432 callers still
    /// compile and behave: it maps onto [`speculative`](Self::speculative) through
    /// [`From<MtpMode> for Speculative`](Speculative) whenever that field is `None`. Setting both
    /// (a `Some` speculative option and a non-`Off` legacy mode) is refused by
    /// [`TextLlmCapabilities::validate_request`](crate::TextLlmCapabilities::validate_request)
    /// rather than silently preferring one.
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

    /// The effective speculative option: [`speculative`](Self::speculative) when set, else the
    /// legacy [`mtp`](Self::mtp) mapped onto it.
    pub fn speculative_mode(&self) -> Speculative {
        self.speculative.unwrap_or_else(|| self.mtp.into())
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
        assert_eq!(request.speculative, None);
        assert_eq!(request.speculative_mode(), Speculative::Off);
    }

    /// sc-24433 AC2: the proposer-agnostic option round-trips through its wire form, and the
    /// legacy `mtp` shape deserializes onto it — `enabled` as `{proposer: mtp, depth}`.
    #[test]
    fn speculative_option_round_trips_and_reads_the_legacy_mtp_shape() {
        let cases = [
            (Speculative::Off, serde_json::json!("off")),
            (Speculative::Auto, serde_json::json!("auto")),
            (
                Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
                serde_json::json!({"proposer": "prompt_lookup", "depth": 4}),
            ),
            (
                Speculative::proposer(SpeculativeProposer::Mtp, 3),
                serde_json::json!({"proposer": "mtp", "depth": 3}),
            ),
            (
                Speculative::proposer(SpeculativeProposer::DraftModel, 2),
                serde_json::json!({"proposer": "draft_model", "depth": 2}),
            ),
        ];
        for (option, wire) in cases {
            assert_eq!(serde_json::to_value(option).unwrap(), wire);
            assert_eq!(serde_json::from_value::<Speculative>(wire).unwrap(), option);
        }

        // The legacy `mtp` request shape (ChatWorks' `{"mode": …}` form).
        let legacy = |v| serde_json::from_value::<Speculative>(v).unwrap();
        assert_eq!(
            legacy(serde_json::json!({"mode": "enabled", "draft_tokens": 3})),
            Speculative::proposer(SpeculativeProposer::Mtp, 3)
        );
        assert_eq!(
            legacy(serde_json::json!({"mode": "auto"})),
            Speculative::Auto
        );
        assert_eq!(legacy(serde_json::json!({"mode": "off"})), Speculative::Off);

        // A consumer DTO keeps reading an old `mtp` field through a serde alias.
        #[derive(serde::Deserialize)]
        struct Dto {
            #[serde(default, alias = "mtp")]
            speculative: Option<Speculative>,
        }
        let dto: Dto =
            serde_json::from_str(r#"{"mtp": {"mode": "enabled", "draft_tokens": 3}}"#).unwrap();
        assert_eq!(
            dto.speculative,
            Some(Speculative::proposer(SpeculativeProposer::Mtp, 3))
        );
        let dto: Dto = serde_json::from_str(r#"{"speculative": "auto"}"#).unwrap();
        assert_eq!(dto.speculative, Some(Speculative::Auto));
        let dto: Dto = serde_json::from_str("{}").unwrap();
        assert_eq!(dto.speculative, None);

        // Refusals name what was wrong.
        for (bad, needle) in [
            (
                serde_json::json!("sometimes"),
                "unknown speculative mode `sometimes`",
            ),
            (
                serde_json::json!({"proposer": "medusa", "depth": 2}),
                "unknown speculative proposer `medusa`",
            ),
            (serde_json::json!({"proposer": "mtp"}), "`depth` is missing"),
            (
                serde_json::json!({"proposer": "mtp", "depth": -1}),
                "`depth` must be an unsigned",
            ),
            (
                serde_json::json!({"proposer": "mtp", "depth": 2, "k": 1}),
                "unexpected speculative field `k`",
            ),
            (
                serde_json::json!({"mode": "enabled"}),
                "`draft_tokens` is missing",
            ),
            (serde_json::json!(3), "invalid speculative option"),
        ] {
            let err = serde_json::from_value::<Speculative>(bad.clone())
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{bad}: {err}");
        }
    }

    /// The legacy `mtp` request field maps onto the option whenever `speculative` is unset; an
    /// explicit `speculative` (even `Off`) is the request's choice.
    #[test]
    fn the_legacy_mtp_field_maps_onto_the_speculative_option() {
        let mut request = TextLlmRequest::new(Vec::new(), 8);
        request.mtp = MtpMode::Enabled { draft_tokens: 3 };
        assert_eq!(
            request.speculative_mode(),
            Speculative::proposer(SpeculativeProposer::Mtp, 3)
        );
        request.mtp = MtpMode::Auto;
        assert_eq!(request.speculative_mode(), Speculative::Auto);
        request.mtp = MtpMode::Off;
        request.speculative = Some(Speculative::proposer(SpeculativeProposer::PromptLookup, 4));
        assert_eq!(
            request.speculative_mode(),
            Speculative::proposer(SpeculativeProposer::PromptLookup, 4)
        );
        request.speculative = Some(Speculative::Off);
        assert_eq!(request.speculative_mode(), Speculative::Off);
        assert_eq!(
            "prompt_lookup".parse::<SpeculativeProposer>().unwrap(),
            SpeculativeProposer::PromptLookup
        );
        assert_eq!(SpeculativeProposer::DraftModel.to_string(), "draft_model");
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
        assert!(
            spec.cuda_graphs.is_none(),
            "a dense load keeps the backend's graph default"
        );
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
    /// Load-time CUDA-graph decode policy (sc-24139): `Some(true)` / `Some(false)` turn the
    /// backend's CUDA-graph runner on / off for every generation on this loaded model; `None`
    /// keeps the backend's default (candle-llm: `CANDLE_LLM_CUDA_GRAPHS`, off unless set). It is a
    /// **load** option because a CUDA backend settles the model's stream at load — graph capture
    /// needs a dedicated stream, which the eager path does not use — so changing it means
    /// reloading. A hint, never a requirement: a backend that cannot capture graphs loads and
    /// decodes eagerly and names why — before the load in
    /// [`BackendCapabilities::cuda_graphs`](crate::BackendCapabilities::cuda_graphs), and per
    /// generation in [`DecodeReport::cuda_graphs`](crate::DecodeReport::cuda_graphs) where the
    /// backend reports one. Backends without CUDA ignore it.
    pub cuda_graphs: Option<bool>,
    /// Byte budget of the cross-turn prefix cache (story sc-24437): the KV (and, for a hybrid
    /// decoder, recurrent state) of earlier requests' prefixes, reused when a later prompt
    /// extends one. `None` asks for [`DEFAULT_PREFIX_CACHE_BYTES`](crate::DEFAULT_PREFIX_CACHE_BYTES);
    /// `Some(0)` turns the cache off. The load admits it: the settled budget is this clamped to
    /// the headroom the load's own admission leaves ([`prefix_cache_budget`](crate::prefix_cache_budget)).
    pub prefix_cache_bytes: Option<u64>,
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
            cuda_graphs: None,
            prefix_cache_bytes: None,
        }
    }

    /// Associate an exact multimodal projector artifact with this model load.
    pub fn with_projector(mut self, source: impl Into<String>) -> Self {
        self.projector_source = Some(source.into());
        self
    }
}
