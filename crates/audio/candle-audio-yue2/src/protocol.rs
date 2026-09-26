//! The YuE2 request protocol `yue2-native-v1` — a native port of upstream `yue2.protocol`, plus
//! the pre-generation checks of `yue2.sampling.generate_tokens` and `yue2.nar.song_chunks`
//! (sc-22990).
//!
//! * The vocabulary layout ([`EOD`], [`ABC_START`], [`MUSIC_START`], [`CODEC_OFFSET`], …) and the
//!   fixed [`CONTEXT`] of 24576 positions.
//! * [`Sampling`] — one AR phase's controls (temperature, top-p, top-k, repetition penalty and
//!   window, minimum and maximum new tokens), validated against upstream's limits. The ABC and
//!   semantic phases carry separate values ([`GenerationConfig`]).
//! * [`GenerationConfig`] — both phases' sampling plus the midpoint ODE step count. The ODE method
//!   (`midpoint`) and the context (24576) are fixed by the protocol and are not knobs.
//! * [`SongRequest`] — style, lyrics, chain-of-thought mode ([`CotMode`]), seed, optional external
//!   ABC, optional CFG scale and a filename-safe id, validated exactly as upstream.
//! * [`token_prefixes`] / [`negative_prefix`] — the exact positive and CFG-negative token prefixes
//!   the model is fed, assembled from the request text and the exact ABC token IDs.
//! * [`check_generation_budget`] and [`chunk_ranges`] — the explicit refusals upstream makes
//!   before generating (prefix + budget past the context; CFG without a negative) and before
//!   acoustic synthesis (a prefix that leaves no acoustic context). Nothing is truncated
//!   implicitly.
//!
//! JSON entry points ([`Sampling::with_json_overrides`], [`GenerationConfig::from_json`],
//! [`SongRequest::from_json`]) mirror upstream's keyword construction: an unknown key is an
//! error, integers must be JSON integers, and booleans are never numbers.

use serde_json::{Map, Number, Value};

use crate::tokenizer::{TokenizerError, Yue2TextTokenizer, ORDINARY_TOKENS};

/// `<|endoftext|>`: the first token of every prefix; ABC IDs must stay below it.
pub const EOD: u32 = ORDINARY_TOKENS;
/// `<abc>` — opens the symbolic plan.
pub const ABC_START: u32 = 151_847;
/// `</abc>` — closes the symbolic plan (the ABC phase's end token).
pub const ABC_END: u32 = 151_848;
/// Opens the semantic (codec) stream.
pub const MUSIC_START: u32 = 151_851;
/// Closes the semantic stream (the semantic phase's end token).
pub const MUSIC_END: u32 = 151_852;
/// First codec token ID; codec value `v` is token `CODEC_OFFSET + v`.
pub const CODEC_OFFSET: u32 = 151_853;
/// Number of codec values.
pub const CODEC_SIZE: u32 = 32_768;
/// Latent-stream start token.
pub const LATENT_START: u32 = 184_621;
/// Latent-stream end token.
pub const LATENT_END: u32 = 184_622;
/// Latent-stream padding token.
pub const LATENT_PAD: u32 = 184_623;
/// The model's full vocabulary.
pub const VOCAB_SIZE: u32 = 184_704;
/// The fixed context (positions) of the checkpoint.
pub const CONTEXT: usize = 24_576;
/// The protocol this module implements.
pub const PROTOCOL_VERSION: &str = "yue2-native-v1";
/// The only ODE method the protocol defines.
pub const ODE_METHOD: &str = "midpoint";
/// The default seed.
pub const DEFAULT_SEED: u64 = 831_001;
/// The default request id.
pub const DEFAULT_ID: &str = "song";

/// Every protocol refusal.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    /// A value is outside the protocol's limits or has the wrong type.
    #[error("invalid {field}: {detail}")]
    Invalid {
        /// The offending field.
        field: &'static str,
        /// What the protocol requires.
        detail: String,
    },
    /// A JSON object carries a key the protocol does not define.
    #[error("unknown {what} key `{key}`")]
    UnknownKey {
        /// Which object.
        what: &'static str,
        /// The key.
        key: String,
    },
    /// External ABC with `cot = off`, or empty / whitespace-only external ABC.
    #[error("external ABC requires nonempty text and cot=melody/full")]
    ExternalAbc,
    /// A token supplied as ABC is outside the ordinary text vocabulary.
    #[error("ABC IDs must remain inside the ordinary text vocabulary (got {0})")]
    AbcIdOutOfVocabulary(u32),
    /// `cot = off` carries no ABC, but ABC IDs were supplied.
    #[error("cot=off carries no symbolic plan, but {0} ABC IDs were supplied")]
    AbcIdsWithoutPlan(usize),
    /// A symbolic-mode negative prefix was requested without the positive branch's exact ABC IDs.
    #[error("symbolic CFG must retain the exact positive-branch ABC IDs")]
    NegativeWithoutAbc,
    /// CFG (`cfg_scale != 1`) without a negative prefix.
    #[error("CFG requires a negative prefix")]
    CfgWithoutNegative,
    /// A prefix plus the requested generation budget does not fit the context.
    #[error(
        "{branch} prefix ({prefix} tokens) + max_tokens ({max_tokens}) exceeds the {CONTEXT}-token \
         context; nothing is truncated implicitly"
    )]
    BudgetExceedsContext {
        /// `positive` or `negative`.
        branch: &'static str,
        /// Prefix length.
        prefix: usize,
        /// Requested budget.
        max_tokens: u64,
    },
    /// The prefix leaves no acoustic context (or there are no codec frames).
    #[error(
        "no acoustic context: {frames} codec frames after a {prefix_tokens}-token prefix in a \
         {context}-position context"
    )]
    ExhaustedAcousticContext {
        /// Codec frames to synthesize.
        frames: usize,
        /// Prefix length.
        prefix_tokens: usize,
        /// The context.
        context: usize,
    },
    /// Tokenizing the request failed.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
}

fn invalid(field: &'static str, detail: impl Into<String>) -> ProtocolError {
    ProtocolError::Invalid {
        field,
        detail: detail.into(),
    }
}

/// The chain-of-thought (symbolic planning) mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CotMode {
    /// No symbolic plan: codec tokens straight from the conditions.
    Off,
    /// A melody-only ABC plan (no chord symbols) before the codec tokens.
    Melody,
    /// A chord-annotated ABC plan before the codec tokens (the default).
    Full,
}

impl CotMode {
    /// Every mode.
    pub const ALL: [CotMode; 3] = [CotMode::Off, CotMode::Melody, CotMode::Full];

    /// The protocol spelling (`off` / `melody` / `full`).
    pub fn as_str(self) -> &'static str {
        match self {
            CotMode::Off => "off",
            CotMode::Melody => "melody",
            CotMode::Full => "full",
        }
    }

    /// Parse the protocol spelling (exact, case-sensitive).
    pub fn parse(text: &str) -> Result<Self, ProtocolError> {
        CotMode::ALL
            .into_iter()
            .find(|m| m.as_str() == text)
            .ok_or_else(|| invalid("cot", "cot must be off, melody or full"))
    }

    /// The mode's checkpoint-native instruction line.
    pub fn instruction(self) -> &'static str {
        match self {
            CotMode::Off => "Generate music with codec tokens from the given conditions.",
            CotMode::Melody => {
                "Generate a melody-only ABC transcription without chord symbols, then generate \
                 music with codec tokens from the given conditions."
            }
            CotMode::Full => {
                "Generate a chord-annotated ABC transcription, then generate music with codec \
                 tokens from the given conditions."
            }
        }
    }

    /// The default guidance scale when the request sets none (`1.01` without a plan, else `1`).
    pub fn default_guidance(self) -> f64 {
        if self == CotMode::Off {
            1.01
        } else {
            1.0
        }
    }
}

/// Explicit per-field overrides of a [`Sampling`]. Counts are signed so out-of-range input reaches
/// validation and is refused there, exactly as upstream refuses it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingOverrides {
    /// Softmax temperature, `0..=5` (`0` is greedy).
    pub temperature: Option<f64>,
    /// Nucleus mass, `(0, 1]`.
    pub top_p: Option<f64>,
    /// Top-k, `>= 1`.
    pub top_k: Option<i64>,
    /// Windowed repetition penalty, `> 0`.
    pub repetition_penalty: Option<f64>,
    /// Repetition-penalty window, `1..=100` tokens.
    pub penalty_window: Option<i64>,
    /// Tokens before the end token may be sampled, `0..=max_tokens`.
    pub min_tokens: Option<i64>,
    /// Maximum new tokens, `>= 1`.
    pub max_tokens: Option<i64>,
}

/// One AR phase's validated sampling controls (upstream `Sampling`). Only constructible through
/// validation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    temperature: f64,
    top_p: f64,
    top_k: u64,
    repetition_penalty: f64,
    penalty_window: u64,
    min_tokens: u64,
    max_tokens: u64,
}

impl Sampling {
    /// The protocol's field names, in upstream order.
    pub const FIELDS: [&'static str; 7] = [
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "penalty_window",
        "min_tokens",
        "max_tokens",
    ];

    /// Upstream's `Sampling()` default — the semantic phase's defaults.
    pub const fn semantic_default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
            penalty_window: 50,
            min_tokens: 200,
            max_tokens: 9000,
        }
    }

    /// The ABC (planning) phase's defaults.
    pub const fn abc_default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.005,
            penalty_window: 100,
            min_tokens: 32,
            max_tokens: 4096,
        }
    }

    /// `self` with `overrides` applied, validated as a whole.
    pub fn with_overrides(&self, overrides: &SamplingOverrides) -> Result<Self, ProtocolError> {
        let temperature = overrides.temperature.unwrap_or(self.temperature);
        let top_p = overrides.top_p.unwrap_or(self.top_p);
        let repetition_penalty = overrides
            .repetition_penalty
            .unwrap_or(self.repetition_penalty);
        let count = |value: Option<i64>, current: u64| value.map_or(current as i128, i128::from);
        let top_k = count(overrides.top_k, self.top_k);
        let penalty_window = count(overrides.penalty_window, self.penalty_window);
        let min_tokens = count(overrides.min_tokens, self.min_tokens);
        let max_tokens = count(overrides.max_tokens, self.max_tokens);

        if ![temperature, top_p, repetition_penalty]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(invalid("sampling", "sampling numbers must be finite"));
        }
        if !(0.0..=5.0).contains(&temperature) {
            return Err(invalid("temperature", "must be in [0, 5]"));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(invalid("top_p", "must be in (0, 1]"));
        }
        if top_k < 1 {
            return Err(invalid("top_k", "must be >= 1"));
        }
        if repetition_penalty <= 0.0 {
            return Err(invalid("repetition_penalty", "must be > 0"));
        }
        if !(1..=100).contains(&penalty_window) {
            return Err(invalid("penalty_window", "must be in [1, 100]"));
        }
        if !(0 <= min_tokens && min_tokens <= max_tokens) || max_tokens < 1 {
            return Err(invalid(
                "min_tokens/max_tokens",
                "require 0 <= min_tokens <= max_tokens and max_tokens >= 1",
            ));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k: top_k as u64,
            repetition_penalty,
            penalty_window: penalty_window as u64,
            min_tokens: min_tokens as u64,
            max_tokens: max_tokens as u64,
        })
    }

    /// `self` with a JSON object of overrides applied (upstream `resolve_sampling(dict, self)`):
    /// unknown keys are refused, counts must be JSON integers.
    pub fn with_json_overrides(&self, value: &Value) -> Result<Self, ProtocolError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("sampling", "must be an object of overrides"))?;
        let mut o = SamplingOverrides::default();
        for (key, v) in object {
            match key.as_str() {
                "temperature" => o.temperature = Some(json_float("temperature", v)?),
                "top_p" => o.top_p = Some(json_float("top_p", v)?),
                "top_k" => o.top_k = Some(json_count("top_k", v)?),
                "repetition_penalty" => {
                    o.repetition_penalty = Some(json_float("repetition_penalty", v)?)
                }
                "penalty_window" => o.penalty_window = Some(json_count("penalty_window", v)?),
                "min_tokens" => o.min_tokens = Some(json_count("min_tokens", v)?),
                "max_tokens" => o.max_tokens = Some(json_count("max_tokens", v)?),
                _ => {
                    return Err(ProtocolError::UnknownKey {
                        what: "sampling",
                        key: key.clone(),
                    })
                }
            }
        }
        self.with_overrides(&o)
    }

    /// The upstream `asdict` form.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "penalty_window": self.penalty_window,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
        })
    }

    /// Softmax temperature (`0` = greedy).
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    /// Nucleus mass.
    pub fn top_p(&self) -> f64 {
        self.top_p
    }
    /// Top-k.
    pub fn top_k(&self) -> u64 {
        self.top_k
    }
    /// Windowed repetition penalty.
    pub fn repetition_penalty(&self) -> f64 {
        self.repetition_penalty
    }
    /// Repetition-penalty window in tokens.
    pub fn penalty_window(&self) -> u64 {
        self.penalty_window
    }
    /// New tokens before the end token may be sampled.
    pub fn min_tokens(&self) -> u64 {
        self.min_tokens
    }
    /// Maximum new tokens.
    pub fn max_tokens(&self) -> u64 {
        self.max_tokens
    }
}

/// Both AR phases' sampling and the midpoint ODE step count (upstream `GenerationConfig`). The
/// context and ODE method are protocol constants ([`CONTEXT`], [`ODE_METHOD`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationConfig {
    abc: Sampling,
    semantic: Sampling,
    ode_steps: u64,
    version: String,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            abc: Sampling::abc_default(),
            semantic: Sampling::semantic_default(),
            ode_steps: 32,
            version: PROTOCOL_VERSION.to_string(),
        }
    }
}

impl GenerationConfig {
    /// A config with explicit phase sampling and a positive midpoint step count.
    pub fn new(abc: Sampling, semantic: Sampling, ode_steps: i64) -> Result<Self, ProtocolError> {
        if ode_steps < 1 {
            return Err(invalid(
                "ode_steps",
                "require a positive integer step count",
            ));
        }
        Ok(Self {
            abc,
            semantic,
            ode_steps: ode_steps as u64,
            version: PROTOCOL_VERSION.to_string(),
        })
    }

    /// Upstream `GenerationConfig.from_dict`: `abc` / `semantic` objects override the defaults
    /// field by field; `context` must be 24576 and `ode_method` `midpoint`.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("generation config", "must be an object"))?;
        let mut config = Self::default();
        let mut ode_steps = config.ode_steps as i64;
        for (key, v) in object {
            match key.as_str() {
                "abc" => config.abc = Sampling::abc_default().with_json_overrides(v)?,
                "semantic" => {
                    config.semantic = Sampling::semantic_default().with_json_overrides(v)?
                }
                "ode_steps" => ode_steps = json_count("ode_steps", v)?,
                "ode_method" => {
                    if v.as_str() != Some(ODE_METHOD) {
                        return Err(invalid("ode_method", "the protocol defines only midpoint"));
                    }
                }
                "context" => {
                    if json_count("context", v)? != CONTEXT as i64 {
                        return Err(invalid("context", format!("require context={CONTEXT}")));
                    }
                }
                "version" => {
                    config.version = v
                        .as_str()
                        .ok_or_else(|| invalid("version", "must be a string"))?
                        .to_string()
                }
                _ => {
                    return Err(ProtocolError::UnknownKey {
                        what: "generation config",
                        key: key.clone(),
                    })
                }
            }
        }
        let version = config.version;
        Ok(Self {
            version,
            ..Self::new(config.abc, config.semantic, ode_steps)?
        })
    }

    /// The upstream `asdict` form.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "abc": self.abc.to_json(),
            "semantic": self.semantic.to_json(),
            "ode_steps": self.ode_steps,
            "ode_method": ODE_METHOD,
            "context": CONTEXT,
            "version": self.version,
        })
    }

    /// The ABC (planning) phase's sampling.
    pub fn abc(&self) -> &Sampling {
        &self.abc
    }
    /// The semantic phase's sampling.
    pub fn semantic(&self) -> &Sampling {
        &self.semantic
    }
    /// Midpoint ODE steps.
    pub fn ode_steps(&self) -> u64 {
        self.ode_steps
    }
    /// The recorded protocol version string.
    pub fn version(&self) -> &str {
        &self.version
    }
}

/// The fields of a [`SongRequest`], before validation.
#[derive(Clone, Debug, PartialEq)]
pub struct SongRequestSpec {
    /// Genre, instruments, vocal character, language and tempo.
    pub style: String,
    /// The words to sing, with section tags such as `[Verse]`.
    pub lyrics: String,
    /// Planning mode (default [`CotMode::Full`]).
    pub cot: CotMode,
    /// Seed, `0..2^63` (default [`DEFAULT_SEED`]).
    pub seed: i128,
    /// External ABC score to use as the plan (requires `melody` / `full`).
    pub abc: Option<String>,
    /// CFG scale, finite in `[0, 20]`; `None` uses [`CotMode::default_guidance`].
    pub cfg_scale: Option<f64>,
    /// Filename-safe id (default [`DEFAULT_ID`]).
    pub id: String,
}

impl SongRequestSpec {
    /// A spec with upstream's defaults for everything but style and lyrics.
    pub fn new(style: impl Into<String>, lyrics: impl Into<String>) -> Self {
        Self {
            style: style.into(),
            lyrics: lyrics.into(),
            cot: CotMode::Full,
            seed: DEFAULT_SEED as i128,
            abc: None,
            cfg_scale: None,
            id: DEFAULT_ID.to_string(),
        }
    }
}

/// A validated song request (upstream `SongRequest`). Only constructible through validation; to
/// change the composition, derive a new request with [`SongRequest::with_abc`].
#[derive(Clone, Debug, PartialEq)]
pub struct SongRequest {
    style: String,
    lyrics: String,
    cot: CotMode,
    seed: u64,
    abc: Option<String>,
    cfg_scale: Option<f64>,
    id: String,
}

impl SongRequest {
    /// The protocol's field names, in upstream order.
    pub const FIELDS: [&'static str; 7] =
        ["style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id"];

    /// Validate `spec`.
    pub fn new(spec: SongRequestSpec) -> Result<Self, ProtocolError> {
        let SongRequestSpec {
            style,
            lyrics,
            cot,
            seed,
            abc,
            cfg_scale,
            id,
        } = spec;
        if !(0..1i128 << 63).contains(&seed) {
            return Err(invalid("seed", "seed must be an integer in [0, 2**63)"));
        }
        if !is_filename_safe_id(&id) {
            return Err(invalid("id", "id must be a filename-safe identifier"));
        }
        if let Some(abc) = &abc {
            if cot == CotMode::Off || abc.chars().all(python_isspace) {
                return Err(ProtocolError::ExternalAbc);
            }
        }
        if let Some(scale) = cfg_scale {
            if !(scale.is_finite() && (0.0..=20.0).contains(&scale)) {
                return Err(invalid(
                    "cfg_scale",
                    "cfg_scale must be finite and in [0,20]",
                ));
            }
        }
        Ok(Self {
            style,
            lyrics,
            cot,
            seed: seed as u64,
            abc,
            cfg_scale,
            id,
        })
    }

    /// Parse and validate upstream's `SongRequest(**data)` keyword form (also `plan.json`'s
    /// `request`). `style` and `lyrics` are required; unknown keys are refused.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("request", "must be an object"))?;
        let string = |field: &'static str, v: &Value| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| invalid(field, "must be a string"))
        };
        let mut style = None;
        let mut lyrics = None;
        let mut spec = SongRequestSpec::new("", "");
        for (key, v) in object {
            match key.as_str() {
                "style" => style = Some(string("style", v)?),
                "lyrics" => lyrics = Some(string("lyrics", v)?),
                "cot" => {
                    spec.cot = CotMode::parse(
                        v.as_str()
                            .ok_or_else(|| invalid("cot", "cot must be off, melody or full"))?,
                    )?
                }
                "seed" => spec.seed = json_integer("seed", v)?,
                "abc" => {
                    spec.abc = match v {
                        Value::Null => None,
                        Value::String(s) => Some(s.clone()),
                        _ => return Err(ProtocolError::ExternalAbc),
                    }
                }
                "cfg_scale" => {
                    spec.cfg_scale = match v {
                        Value::Null => None,
                        _ => Some(json_float("cfg_scale", v)?),
                    }
                }
                "id" => spec.id = string("id", v)?,
                _ => {
                    return Err(ProtocolError::UnknownKey {
                        what: "request",
                        key: key.clone(),
                    })
                }
            }
        }
        spec.style = style.ok_or_else(|| invalid("style", "style is required"))?;
        spec.lyrics = lyrics.ok_or_else(|| invalid("lyrics", "lyrics is required"))?;
        Self::new(spec)
    }

    /// The upstream `asdict` form (field order as upstream).
    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("style".into(), Value::String(self.style.clone()));
        object.insert("lyrics".into(), Value::String(self.lyrics.clone()));
        object.insert("cot".into(), Value::String(self.cot.as_str().into()));
        object.insert("seed".into(), Value::Number(self.seed.into()));
        object.insert(
            "abc".into(),
            self.abc.clone().map_or(Value::Null, Value::String),
        );
        object.insert(
            "cfg_scale".into(),
            self.cfg_scale
                .and_then(Number::from_f64)
                .map_or(Value::Null, Value::Number),
        );
        object.insert("id".into(), Value::String(self.id.clone()));
        Value::Object(object)
    }

    /// A **new** request that plans from `abc` (an edited copy of a score) instead of sampling
    /// one. The composition changed, so this is never the same request as `self`, and its plan
    /// is built fresh from the edited text ([`crate::plan::SymbolicPlan::prepare`]).
    pub fn with_abc(&self, abc: impl Into<String>) -> Result<Self, ProtocolError> {
        Self::new(SongRequestSpec {
            abc: Some(abc.into()),
            ..self.spec()
        })
    }

    /// The request's fields.
    pub fn spec(&self) -> SongRequestSpec {
        SongRequestSpec {
            style: self.style.clone(),
            lyrics: self.lyrics.clone(),
            cot: self.cot,
            seed: self.seed as i128,
            abc: self.abc.clone(),
            cfg_scale: self.cfg_scale,
            id: self.id.clone(),
        }
    }

    /// The effective CFG scale: the request's, or the mode default.
    pub fn guidance(&self) -> f64 {
        self.cfg_scale.unwrap_or(self.cot.default_guidance())
    }

    /// The prompt text the tokenizer encodes.
    pub fn text(&self) -> String {
        format!(
            "{}\n[Tags]\n{}\n[Lyrics]\n{}\n",
            self.cot.instruction(),
            self.style,
            self.lyrics
        )
    }

    /// Style.
    pub fn style(&self) -> &str {
        &self.style
    }
    /// Lyrics.
    pub fn lyrics(&self) -> &str {
        &self.lyrics
    }
    /// Planning mode.
    pub fn cot(&self) -> CotMode {
        self.cot
    }
    /// Seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }
    /// External ABC, if the request supplies one.
    pub fn abc(&self) -> Option<&str> {
        self.abc.as_deref()
    }
    /// Explicit CFG scale, if set.
    pub fn cfg_scale(&self) -> Option<f64> {
        self.cfg_scale
    }
    /// Filename-safe id.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Upstream `[A-Za-z0-9][A-Za-z0-9_.-]{0,179}` (full match), never `.` or `..`.
fn is_filename_safe_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 180
        && bytes[0].is_ascii_alphanumeric()
        && bytes[1..]
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// Python's `str.isspace` for one character: Unicode `White_Space` plus the four ASCII
/// information separators U+001C..U+001F, which Python counts (bidi class B/S) and Rust does not.
pub(crate) fn python_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn text_ids(tokenizer: &Yue2TextTokenizer, text: &str) -> Result<Vec<u32>, ProtocolError> {
    Ok(tokenizer.encode(text)?)
}

fn check_abc_ids(ids: &[u32]) -> Result<(), ProtocolError> {
    match ids.iter().find(|&&id| id >= EOD) {
        Some(&id) => Err(ProtocolError::AbcIdOutOfVocabulary(id)),
        None => Ok(()),
    }
}

/// The positive prefix the model is fed (upstream `token_prefixes`).
///
/// `[EOD] + encode(request.text())`, then:
/// * `cot = off`: `+ [ABC_START, ABC_END, MUSIC_START]` (`abc_ids` must be absent or empty);
/// * `abc_ids = None` and no external ABC: `+ [ABC_START]` — the ABC planner's prefix;
/// * otherwise `+ [ABC_START] + abc + [ABC_END, MUSIC_START]`, where `abc` is `abc_ids` or, when
///   absent, the encoded external ABC. ABC IDs must be ordinary tokens (`< EOD`).
pub fn token_prefixes(
    request: &SongRequest,
    tokenizer: &Yue2TextTokenizer,
    abc_ids: Option<&[u32]>,
) -> Result<Vec<u32>, ProtocolError> {
    let mut prefix = vec![EOD];
    prefix.extend(text_ids(tokenizer, &request.text())?);
    if request.cot == CotMode::Off {
        if let Some(ids) = abc_ids.filter(|ids| !ids.is_empty()) {
            return Err(ProtocolError::AbcIdsWithoutPlan(ids.len()));
        }
        prefix.extend([ABC_START, ABC_END, MUSIC_START]);
        return Ok(prefix);
    }
    let encoded;
    let abc = match (abc_ids, &request.abc) {
        (Some(ids), _) => ids,
        (None, None) => {
            prefix.push(ABC_START);
            return Ok(prefix);
        }
        (None, Some(text)) => {
            encoded = text_ids(tokenizer, text)?;
            &encoded
        }
    };
    check_abc_ids(abc)?;
    prefix.push(ABC_START);
    prefix.extend_from_slice(abc);
    prefix.extend([ABC_END, MUSIC_START]);
    Ok(prefix)
}

/// The CFG negative prefix (upstream `negative_prefix`): the mode instruction only (no tags, no
/// lyrics), then `[MUSIC_START]` for `cot = off`, or the positive branch's **exact** ABC IDs for a
/// symbolic mode (required).
pub fn negative_prefix(
    request: &SongRequest,
    tokenizer: &Yue2TextTokenizer,
    abc_ids: Option<&[u32]>,
) -> Result<Vec<u32>, ProtocolError> {
    let mut prefix = vec![EOD];
    prefix.extend(text_ids(tokenizer, request.cot.instruction())?);
    if request.cot == CotMode::Off {
        if let Some(ids) = abc_ids.filter(|ids| !ids.is_empty()) {
            return Err(ProtocolError::AbcIdsWithoutPlan(ids.len()));
        }
        prefix.push(MUSIC_START);
        return Ok(prefix);
    }
    let abc = abc_ids.ok_or(ProtocolError::NegativeWithoutAbc)?;
    check_abc_ids(abc)?;
    prefix.push(ABC_START);
    prefix.extend_from_slice(abc);
    prefix.extend([ABC_END, MUSIC_START]);
    Ok(prefix)
}

/// Upstream `generate_tokens`' refusals before prefill: the positive prefix plus `max_tokens`
/// must fit [`CONTEXT`]; `cfg_scale != 1` needs a negative prefix; a supplied negative prefix
/// plus `max_tokens` must fit too. Nothing is ever truncated to make a request fit.
pub fn check_generation_budget(
    prefix_tokens: usize,
    negative_tokens: Option<usize>,
    sampling: &Sampling,
    cfg_scale: f64,
) -> Result<(), ProtocolError> {
    let fits = |len: usize| (len as u128) + (sampling.max_tokens as u128) <= CONTEXT as u128;
    if !fits(prefix_tokens) {
        return Err(ProtocolError::BudgetExceedsContext {
            branch: "positive",
            prefix: prefix_tokens,
            max_tokens: sampling.max_tokens,
        });
    }
    if cfg_scale != 1.0 && negative_tokens.is_none() {
        return Err(ProtocolError::CfgWithoutNegative);
    }
    if let Some(len) = negative_tokens.filter(|&len| !fits(len)) {
        return Err(ProtocolError::BudgetExceedsContext {
            branch: "negative",
            prefix: len,
            max_tokens: sampling.max_tokens,
        });
    }
    Ok(())
}

/// The original acoustic chunks (upstream `chunk_ranges`, with `song_chunks`' context check):
/// `frames` codec frames split into half-open `[start, end)` ranges of
/// `min((context - prefix_tokens - 3) / 2, CONTEXT)` frames. No frames, or a prefix that leaves
/// no acoustic context, is [`ProtocolError::ExhaustedAcousticContext`].
pub fn chunk_ranges(
    frames: usize,
    prefix_tokens: usize,
    context: usize,
) -> Result<Vec<(usize, usize)>, ProtocolError> {
    if !(1..=CONTEXT).contains(&context) {
        return Err(invalid(
            "context",
            format!("context must be an integer in 1..{CONTEXT}"),
        ));
    }
    let size = (context as i128 - prefix_tokens as i128 - 3)
        .div_euclid(2)
        .min(CONTEXT as i128);
    if frames < 1 || size < 1 {
        return Err(ProtocolError::ExhaustedAcousticContext {
            frames,
            prefix_tokens,
            context,
        });
    }
    let size = size as usize;
    Ok((0..frames)
        .step_by(size)
        .map(|start| (start, (start + size).min(frames)))
        .collect())
}

/// A JSON integer (booleans and floats refused, like upstream's `type(x) is int`).
fn json_integer(field: &'static str, v: &Value) -> Result<i128, ProtocolError> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
            .ok_or_else(|| invalid(field, "must be an integer")),
        _ => Err(invalid(field, "must be an integer")),
    }
}

fn json_count(field: &'static str, v: &Value) -> Result<i64, ProtocolError> {
    let n = json_integer(field, v)?;
    i64::try_from(n).map_err(|_| invalid(field, "out of range"))
}

/// A JSON number (integer or float; booleans refused).
fn json_float(field: &'static str, v: &Value) -> Result<f64, ProtocolError> {
    v.as_number()
        .and_then(Number::as_f64)
        .ok_or_else(|| invalid(field, "must be a number"))
}

#[cfg(test)]
mod tests;
