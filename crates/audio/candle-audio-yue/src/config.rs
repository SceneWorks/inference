//! The engine's load- and render-time configuration: the six stage-1 variants, the LM tier, the
//! staged asset layout, and the per-render request carrying the full epic-R5 control set with the
//! reference pipeline's suggested defaults as *defaults* (every one overridable).

use std::path::{Path, PathBuf};

use candle_audio::gen_core::{self, AudioTrack, OutputLimiter, Quant};

/// Stage-1 checkpoint language.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Language {
    /// `YuE-s1-7B-anneal-en-*`.
    En,
    /// `YuE-s1-7B-anneal-zh-*`.
    Zh,
    /// `YuE-s1-7B-anneal-jp-kr-*` (Japanese and Korean).
    JpKr,
}

impl Language {
    /// The id fragment (`en` / `zh` / `jp_kr`).
    pub fn id(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Zh => "zh",
            Self::JpKr => "jp_kr",
        }
    }

    /// The upstream repository fragment (`en` / `zh` / `jp-kr`).
    pub fn repo_fragment(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Zh => "zh",
            Self::JpKr => "jp-kr",
        }
    }

    /// The request language codes this checkpoint sings (`AudioParams::language`).
    pub fn codes(self) -> &'static [&'static str] {
        match self {
            Self::En => &["en"],
            Self::Zh => &["zh"],
            Self::JpKr => &["ja", "ko"],
        }
    }
}

/// Stage-1 prompting mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Chain-of-thought checkpoints: lyrics + genre tags only.
    Cot,
    /// In-context-learning checkpoints: additionally conditioned on a reference clip (single-track
    /// mix, or dual-track vocal + instrumental) encoded through xcodec.
    Icl,
}

impl Mode {
    /// The id fragment (`cot` / `icl`).
    pub fn id(self) -> &'static str {
        match self {
            Self::Cot => "cot",
            Self::Icl => "icl",
        }
    }
}

/// One of the six stage-1 checkpoints (language × mode). Each is its own registered provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    /// Checkpoint language.
    pub language: Language,
    /// Prompting mode.
    pub mode: Mode,
}

impl Variant {
    /// Every stage-1 variant, in catalog order.
    pub const ALL: [Variant; 6] = [
        Variant::new(Language::En, Mode::Cot),
        Variant::new(Language::En, Mode::Icl),
        Variant::new(Language::Zh, Mode::Cot),
        Variant::new(Language::Zh, Mode::Icl),
        Variant::new(Language::JpKr, Mode::Cot),
        Variant::new(Language::JpKr, Mode::Icl),
    ];

    /// Construct a variant.
    pub const fn new(language: Language, mode: Mode) -> Self {
        Self { language, mode }
    }

    /// The registry id, e.g. `yue_en_cot`.
    pub fn id(self) -> &'static str {
        match (self.language, self.mode) {
            (Language::En, Mode::Cot) => "yue_en_cot",
            (Language::En, Mode::Icl) => "yue_en_icl",
            (Language::Zh, Mode::Cot) => "yue_zh_cot",
            (Language::Zh, Mode::Icl) => "yue_zh_icl",
            (Language::JpKr, Mode::Cot) => "yue_jp_kr_cot",
            (Language::JpKr, Mode::Icl) => "yue_jp_kr_icl",
        }
    }

    /// The upstream stage-1 repository, e.g. `m-a-p/YuE-s1-7B-anneal-en-cot`.
    pub fn stage1_repo(self) -> String {
        format!(
            "m-a-p/YuE-s1-7B-anneal-{}-{}",
            self.language.repo_fragment(),
            self.mode.id()
        )
    }
}

/// The LM weight tier for stage 1 and stage 2 (epic R2). xcodec and Vocos stay float32 at every tier
/// — the approved whole-pipeline carve-out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// The dense bf16 originals.
    Bf16,
    /// candle-llm `prepare_snapshot` q8.
    Q8,
    /// candle-llm `prepare_snapshot` q4.
    Q4,
}

/// Map `LoadSpec::quantize` onto the tier the load asserts. `None` means *the caller asserted
/// nothing* — the stage loaders detect the tier from the staged snapshot — and is deliberately not
/// [`Tier::Bf16`] (a positive assertion of denseness). NVFP4 is not a published YuE tier and is
/// refused by name rather than folded onto q4.
pub fn requested_tier(id: &str, quantize: Option<Quant>) -> gen_core::Result<Option<Tier>> {
    match quantize {
        None => Ok(None),
        Some(Quant::Q8) => Ok(Some(Tier::Q8)),
        Some(Quant::Q4) => Ok(Some(Tier::Q4)),
        Some(q @ Quant::Nvfp4) => Err(gen_core::Error::Unsupported(format!(
            "{id}: quantize={q:?} is not a YuE tier; the stage-1/stage-2 LMs ship bf16, q8 and q4 \
             (request Q8 or Q4, or leave quantize unset for the staged tier)"
        ))),
    }
}

/// The staged weight layout the engine loads from — every path caller-provisioned (epic 13657).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assets {
    /// The stage-1 snapshot (`LoadSpec::weights`): the 7B Llama shards + `tokenizer.model`.
    pub stage1: PathBuf,
    /// The stage-2 snapshot (component [`STAGE2_COMPONENT_ID`](crate::model::STAGE2_COMPONENT_ID)).
    pub stage2: PathBuf,
    /// The `xcodec_mini_infer` snapshot (component
    /// [`XCODEC_COMPONENT_ID`](crate::model::XCODEC_COMPONENT_ID)): codec checkpoint, semantic
    /// (HuBERT) branch, and the two Vocos decoders.
    pub xcodec: PathBuf,
}

impl Assets {
    /// The mm sentencepiece tokenizer shipped inside the stage-1 snapshot.
    pub fn tokenizer_model(&self) -> PathBuf {
        self.stage1.join("tokenizer.model")
    }

    /// The xcodec snapshot root.
    pub fn xcodec_root(&self) -> &Path {
        &self.xcodec
    }
}

/// Stage-1 classifier-free guidance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Guidance {
    /// No unconditional stream.
    Off,
    /// Guidance scale for the first segment and for every later one.
    On {
        /// Scale for segment 0.
        first: f32,
        /// Scale for segments 1..
        rest: f32,
    },
}

impl Guidance {
    /// The reference pipeline's schedule: 1.5 for segment 0, 1.2 after.
    pub const DEFAULT: Guidance = Guidance::On {
        first: 1.5,
        rest: 1.2,
    };

    /// The scale segment `index` runs at, or `None` when guidance is off.
    pub fn scale_for(self, index: usize) -> Option<f32> {
        match self {
            Self::Off => None,
            Self::On { first, rest } => Some(if index == 0 { first } else { rest }),
        }
    }
}

/// Stage-1 per-segment decode configuration. `Default` is the reference pipeline's suggested
/// configuration; every field is a knob.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeConfig {
    /// Token budget per segment (a segment also ends early on `<EOA>`).
    pub max_new_tokens: u32,
    /// Tokens a segment must produce before `<EOA>` may be sampled.
    pub min_new_tokens: u32,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Classifier-free guidance.
    pub guidance: Guidance,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 3_000,
            min_new_tokens: 100,
            repetition_penalty: 1.1,
            top_p: 0.93,
            temperature: 1.0,
            guidance: Guidance::DEFAULT,
        }
    }
}

/// The in-context reference clip(s) for an ICL render.
#[derive(Clone, Debug, PartialEq)]
pub enum IclTracks {
    /// One mixed reference clip.
    Single(AudioTrack),
    /// Separate vocal and instrumental reference clips.
    Dual {
        /// Vocal reference.
        vocals: AudioTrack,
        /// Instrumental reference.
        instrumental: AudioTrack,
    },
}

/// An ICL reference: the clip(s) plus the window of them the prompt uses.
#[derive(Clone, Debug, PartialEq)]
pub struct IclReference {
    /// The reference audio.
    pub tracks: IclTracks,
    /// Window start (seconds).
    pub start_secs: f32,
    /// Window end (seconds).
    pub end_secs: f32,
}

impl IclReference {
    /// The reference pipeline's default prompt window: the first 30 s.
    pub const DEFAULT_WINDOW: (f32, f32) = (0.0, 30.0);
}

/// Default segment count (the reference pipeline's `run_n_segments`).
pub const DEFAULT_SEGMENTS: u32 = 2;
/// Sampler seed used when a request carries none (the reference pipeline's default seed).
pub const DEFAULT_SEED: u64 = 42;

/// One render: the full R5 control set (tier is a load-time choice on [`crate::engine::YueEngine`]).
#[derive(Clone, Debug, PartialEq)]
pub struct YueRequest {
    /// Genre / style tags, e.g. `"inspiring female uplifting pop airy vocal"`.
    pub genres: String,
    /// Structured lyrics (`[verse]`, `[chorus]`, … sections).
    pub lyrics: String,
    /// How many lyric segments to render; the effective count is capped at the segments the
    /// lyrics hold (the reference pipeline's `min(n, len(lyrics))`).
    pub segments: u32,
    /// Stage-1 decode configuration.
    pub decode: DecodeConfig,
    /// Sampler seed.
    pub seed: u64,
    /// The ICL reference; required for ICL variants, refused for CoT ones.
    pub icl: Option<IclReference>,
    /// The output limiter (upstream `save_audio`: clamp by default, `--rescale` on request).
    pub limiter: OutputLimiter,
}

impl YueRequest {
    /// A request with every knob at its default.
    pub fn new(genres: impl Into<String>, lyrics: impl Into<String>) -> Self {
        Self {
            genres: genres.into(),
            lyrics: lyrics.into(),
            segments: DEFAULT_SEGMENTS,
            decode: DecodeConfig::default(),
            seed: DEFAULT_SEED,
            icl: None,
            limiter: OutputLimiter::Clamp,
        }
    }

    /// Check the request against `variant`, naming every problem.
    pub fn validate(&self, variant: Variant) -> gen_core::Result<()> {
        let id = variant.id();
        let msg = |m: String| Err(gen_core::Error::Msg(format!("{id}: {m}")));
        if self.lyrics.trim().is_empty() {
            return msg("lyrics must not be empty (YuE sings structured lyrics)".into());
        }
        if self.genres.trim().is_empty() {
            return msg("genre tags (the prompt) must not be empty".into());
        }
        if self.segments == 0 {
            return msg("segments must be >= 1".into());
        }
        let d = &self.decode;
        if d.max_new_tokens == 0 {
            return msg("max_new_tokens per segment must be >= 1".into());
        }
        if !(d.repetition_penalty.is_finite() && d.repetition_penalty > 0.0) {
            return msg(format!(
                "repetition_penalty must be finite and > 0, got {}",
                d.repetition_penalty
            ));
        }
        if !(d.top_p.is_finite() && d.top_p > 0.0 && d.top_p <= 1.0) {
            return msg(format!("top_p must be in (0, 1], got {}", d.top_p));
        }
        if !(d.temperature.is_finite() && d.temperature > 0.0) {
            return msg(format!(
                "temperature must be finite and > 0, got {}",
                d.temperature
            ));
        }
        if let Guidance::On { first, rest } = d.guidance {
            if !(first.is_finite() && rest.is_finite() && first > 0.0 && rest > 0.0) {
                return msg(format!(
                    "guidance scales must be finite and > 0, got {first}/{rest}"
                ));
            }
        }
        match (variant.mode, &self.icl) {
            (Mode::Cot, Some(_)) => Err(gen_core::Error::Unsupported(format!(
                "{id}: a chain-of-thought checkpoint takes no reference audio; use the matching \
                 `_icl` variant"
            ))),
            (Mode::Icl, None) => msg(
                "an in-context-learning checkpoint needs a reference clip (ReferenceAudio \
                 conditioning, single-track or with `vocals` + `instrumental` stems)"
                    .into(),
            ),
            (Mode::Icl, Some(icl)) => {
                let window_ok = icl.start_secs.is_finite()
                    && icl.start_secs >= 0.0
                    && icl.end_secs.is_finite()
                    && icl.end_secs > icl.start_secs;
                if !window_ok {
                    return msg(format!(
                        "reference window {}..{} s must satisfy 0 <= start < end",
                        icl.start_secs, icl.end_secs
                    ));
                }
                let tracks: Vec<&AudioTrack> = match &icl.tracks {
                    IclTracks::Single(t) => vec![t],
                    IclTracks::Dual {
                        vocals,
                        instrumental,
                    } => vec![vocals, instrumental],
                };
                if tracks
                    .iter()
                    .any(|t| t.samples.is_empty() || t.sample_rate == 0 || t.channels == 0)
                {
                    return msg("reference audio must be non-empty with a valid rate".into());
                }
                Ok(())
            }
            (Mode::Cot, None) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_variants_have_unique_ids_and_upstream_repos() {
        let ids: Vec<&str> = Variant::ALL.iter().map(|v| v.id()).collect();
        assert_eq!(
            ids,
            [
                "yue_en_cot",
                "yue_en_icl",
                "yue_zh_cot",
                "yue_zh_icl",
                "yue_jp_kr_cot",
                "yue_jp_kr_icl"
            ]
        );
        assert_eq!(
            Variant::new(Language::JpKr, Mode::Icl).stage1_repo(),
            "m-a-p/YuE-s1-7B-anneal-jp-kr-icl"
        );
    }

    #[test]
    fn defaults_are_the_reference_suggestions() {
        let r = YueRequest::new("pop", "[verse]\nla");
        assert_eq!(r.segments, 2);
        assert_eq!(r.seed, 42);
        assert_eq!(r.decode.max_new_tokens, 3_000);
        assert_eq!(r.decode.repetition_penalty, 1.1);
        assert_eq!(r.decode.top_p, 0.93);
        assert_eq!(r.decode.guidance.scale_for(0), Some(1.5));
        assert_eq!(r.decode.guidance.scale_for(3), Some(1.2));
        assert_eq!(Guidance::Off.scale_for(0), None);
    }

    #[test]
    fn tier_mapping_keeps_none_distinct_and_refuses_nvfp4() {
        assert_eq!(requested_tier("x", None).unwrap(), None);
        assert_eq!(
            requested_tier("x", Some(Quant::Q8)).unwrap(),
            Some(Tier::Q8)
        );
        assert_eq!(
            requested_tier("x", Some(Quant::Q4)).unwrap(),
            Some(Tier::Q4)
        );
        assert!(matches!(
            requested_tier("x", Some(Quant::Nvfp4)),
            Err(gen_core::Error::Unsupported(_))
        ));
    }

    #[test]
    fn validate_gates_mode_and_knobs() {
        let cot = Variant::new(Language::En, Mode::Cot);
        let icl = Variant::new(Language::En, Mode::Icl);
        let ok = YueRequest::new("pop", "[verse]\nla");
        assert!(ok.validate(cot).is_ok());
        assert!(ok.validate(icl).is_err(), "ICL needs a reference");

        let mut r = ok.clone();
        r.segments = 0;
        assert!(r.validate(cot).is_err());
        let mut r = ok.clone();
        r.decode.repetition_penalty = f32::NAN;
        assert!(r.validate(cot).is_err());
        let mut r = ok.clone();
        r.decode.max_new_tokens = 0;
        assert!(r.validate(cot).is_err());

        let track = AudioTrack {
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            channels: 1,
            stems: Vec::new(),
        };
        let mut r = ok.clone();
        r.icl = Some(IclReference {
            tracks: IclTracks::Single(track),
            start_secs: 0.0,
            end_secs: 30.0,
        });
        assert!(r.validate(icl).is_ok());
        assert!(matches!(
            r.validate(cot),
            Err(gen_core::Error::Unsupported(_))
        ));
        let mut bad = r.clone();
        bad.icl.as_mut().unwrap().end_secs = 0.0;
        assert!(bad.validate(icl).is_err());
    }
}
