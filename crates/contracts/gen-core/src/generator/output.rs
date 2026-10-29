//! What a generation returns: outputs, reports, artifacts and warnings, plus the output-shape knobs a request names.

use super::*;

/// What a [`Generator`] produced. The `Video` variant's `audio` is `Some` for LTX (always
/// audio) and `None` for Wan — no contract change needed across the two. The `Audio` variant
/// is a **pure** audio synthesis (TTS / music — `Modality::Audio`); audio attached to a video
/// stays on the `Video` variant.
#[derive(Clone, Debug)]
pub enum GenerationOutput {
    Images(Vec<Image>),
    /// Four-channel images with a **straight (un-premultiplied) alpha** channel (sc-24111) — the
    /// output of a request whose [`GenerationRequest::output_channels`] is
    /// [`OutputChannels::Rgba`], on a provider whose
    /// [`Capabilities::supports_alpha_output`] is `true`.
    ///
    /// A **separate variant**, not an `Option<Vec<u8>>` alpha plane on [`Images`](Self::Images),
    /// for the reason [`RgbaImage`] is a separate type: an RGB-only consumer must not be able to
    /// receive transparency by accident and silently drop it (a subject extracted onto a
    /// transparent background, flattened to whatever the decoder painted behind the alpha, is a
    /// wrong image the caller has no way to detect). Adding a variant makes every consumer's
    /// `match` name the case explicitly, which is the point.
    ///
    /// **Never produced unless asked for.** `OutputChannels::Rgb` is the `Default`, and the shared
    /// request floor refuses an RGBA request against a provider that does not advertise the
    /// capability, so no existing consumer can be handed this variant by an existing request.
    ImagesRgba(Vec<RgbaImage>),
    Video {
        frames: Vec<Image>,
        fps: u32,
        audio: Option<AudioTrack>,
    },
    Audio(AudioTrack),
}

/// The caption-overflow policy ([`GenerationRequest::caption_overflow`], epic sc-25678): what a
/// text encoder with a fixed caption budget does with a prompt that tokenizes past it. The three
/// values are upstream Iris-3B's `text_encoder.on_caption_overflow` vocabulary, verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CaptionOverflowPolicy {
    /// Truncate the caption to the budget and log a warning (upstream's release default).
    Warn,
    /// Refuse the request: a caption that does not fit is an error, never silently truncated.
    Error,
    /// Truncate the caption to the budget without logging.
    Silent,
}

impl CaptionOverflowPolicy {
    /// Parse upstream's lowercase spelling (`warn` / `error` / `silent`); anything else is `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "warn" => Self::Warn,
            "error" => Self::Error,
            "silent" => Self::Silent,
            _ => return None,
        })
    }

    /// The lowercase spelling (round-trips with [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Silent => "silent",
        }
    }
}

/// How many channels an image generation emits ([`GenerationRequest::output_channels`],
/// sc-24111).
///
/// [`Rgb`](Self::Rgb) is the `Default` and the historical behaviour, byte-for-byte: a provider
/// whose decoder is natively four-channel (Qwen-Image 2.1) composites its alpha over white and
/// emits [`GenerationOutput::Images`], exactly as it did before this field existed.
///
/// This is an **output-surface** selector, not a render mode. On Qwen-Image 2.1 — the only family
/// that reads it today — it changes nothing about conditioning, denoise or decode: upstream has no
/// transparency flag at all, always decodes four channels, and always returns a PIL `RGBA` image.
/// *Whether* a render is actually transparent is decided by the **prompt** (see the provider
/// crates' `UPSTREAM.md`). All this field decides is whether the caller receives the alpha the
/// decoder produced or a white composite of it — which is why `Rgba` on an ordinary opaque prompt
/// is legal and simply yields `A = 255` everywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OutputChannels {
    /// 8-bit RGB ([`GenerationOutput::Images`]) — the default, and what every provider emitted
    /// before sc-24111.
    #[default]
    Rgb,
    /// 8-bit RGBA with straight alpha ([`GenerationOutput::ImagesRgba`]). Gated by
    /// [`Capabilities::supports_alpha_output`].
    Rgba,
}

/// The HDR opt-in block ([`GenerationRequest::hdr`], sc-18790).
///
/// One field, not two: a colour space without a sink would render HDR and discard it, and a sink
/// without a colour space would leave the engine guessing how to interpret the samples. Bundling
/// them makes both of those unrepresentable, and makes `Option::is_some()` the single unambiguous
/// "is this an HDR render?" test.
#[derive(Clone, Debug)]
pub struct HdrRequest {
    /// The source **and** output colour space (upstream's single `--hdr` flag covers both): how
    /// to interpret EXR conditioning input, and which primaries / `colorSpace` tag the emitted
    /// EXR frames carry.
    pub color_space: HdrColorSpace,
    /// Where decoded HDR frames are delivered, streaming, as the render produces them. An inert
    /// sink is legal and means "render HDR but discard the frames" — useful for a caller that
    /// only wants the HDR *decode path* (float32 VAE, no 8-bit quantization) and reads the SDR
    /// proxy from [`GenerationOutput::Video`].
    pub sink: HdrFrameSink,
}

impl HdrRequest {
    /// The common case: stream frames in `color_space` to `sink`.
    pub fn new(color_space: HdrColorSpace, sink: HdrFrameSink) -> Self {
        Self { color_space, sink }
    }
}

/// The complete result of one generation (sc-22988), returned by
/// [`Generator::generate_with_report`]: the rendered output, the reproducibility record the render
/// published, and the non-fatal conditions the caller must see.
///
/// [`Generator::generate`] returns only [`Self::output`]; a caller that must know a render was
/// truncated, or that reads an artifacts-only render (a plan with no audio), calls
/// [`Generator::generate_with_report`] instead.
#[derive(Clone, Debug)]
pub struct GenerationReport {
    /// What was rendered. `None` only for an artifacts-only request ([`SongParams::plan_only`]),
    /// whose result is [`Self::artifacts`].
    pub output: Option<GenerationOutput>,
    /// The record published into [`AudioParams::artifacts`], when the request asked for one.
    pub artifacts: Option<ArtifactRecord>,
    /// Non-fatal conditions of this render (a truncated phase, a cover-check warning), in the
    /// order they were found. Empty when there is nothing to report.
    pub warnings: Vec<GenerationWarning>,
    /// Which prefix-KV-cache path the render ran and what its caches held (sc-25905). `None` for
    /// every provider that declares no
    /// [`ExecutionSurface::prefix_kv_cache`](crate::ExecutionSurface::prefix_kv_cache) domain;
    /// `Some` for one that does, on every render, whatever the request selected.
    pub prefix_kv_cache: Option<PrefixKvCacheReport>,
}

impl GenerationReport {
    /// A report carrying only `output`: what [`Generator::generate_with_report`] returns for a
    /// provider that keeps no record and reports no warnings.
    pub fn from_output(output: GenerationOutput) -> Self {
        Self {
            output: Some(output),
            artifacts: None,
            warnings: Vec::new(),
            prefix_kv_cache: None,
        }
    }
}

/// A published reproducibility record ([`GenerationReport::artifacts`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactRecord {
    /// The published directory.
    pub dir: std::path::PathBuf,
    /// The record's kind (YuE2: `song`, `plan` or `cached_decode`).
    pub kind: String,
    /// The record's identity (lower-case hex), as the record's own index states it.
    pub identity: String,
}

/// A non-fatal condition of a render ([`GenerationReport::warnings`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationWarning {
    /// A stable, machine-readable code (e.g. `semantic_truncated`).
    pub code: String,
    /// A human-readable detail.
    pub message: String,
}

/// The planning mode of a symbolic-plan song model ([`SongParams::planning`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SongPlanning {
    /// Plan melody and harmony.
    Full,
    /// Plan the melody only.
    Melody,
    /// No symbolic plan.
    Off,
}

/// Which published decoder a symbolic-plan song model decodes with ([`SongParams::decoder`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SongDecoder {
    /// The listening decoder (YuE2: `m-a-p/YuE2-Vae`).
    Standard,
    /// The evaluation decoder (YuE2: `m-a-p/YuE2-Vae-legacy`).
    Legacy,
}

/// A saved symbolic plan to restore ([`SongParams::plan`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SavedPlan {
    /// The plan directory.
    pub dir: std::path::PathBuf,
    /// The plan identity recorded when it was saved (lower-case hex), kept outside the plan
    /// directory; when present, a plan whose identity differs is refused.
    pub identity: Option<String>,
}

/// One autoregressive phase's token-sampling overrides ([`SongParams::score_sampling`] /
/// [`SongParams::semantic_sampling`]). `None` ⇒ the model's default for that field; the model
/// validates the resolved set as a whole.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenSampling {
    /// Softmax temperature (`0` = greedy).
    pub temperature: Option<f64>,
    /// Nucleus mass.
    pub top_p: Option<f64>,
    /// Top-k.
    pub top_k: Option<u32>,
    /// Windowed repetition penalty.
    pub repetition_penalty: Option<f64>,
    /// Repetition-penalty window, in tokens.
    pub penalty_window: Option<u32>,
    /// Output tokens before the end token may be sampled.
    pub min_tokens: Option<u32>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
}

/// Where a model publishes a render's reproducibility record ([`AudioParams::artifacts`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AudioArtifacts {
    /// The run directory. A fresh render requires it to be absent or empty; nothing appears there
    /// until the render has completed.
    pub dir: std::path::PathBuf,
    /// Reuse a completed record, or the verified completed stages of an interrupted one, whose
    /// recorded identities match this request exactly; a mismatched or corrupt record is refused,
    /// never overwritten.
    pub resume: bool,
}

/// The output limiter a model applies before returning audio ([`AudioParams::output_limiter`],
/// sc-19378). Gated by [`Capabilities::supports_output_limiter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputLimiter {
    /// Hard-clamp every sample to the model's limit (YuE: `±0.99`) — the reference default.
    Clamp,
    /// Scale the whole track by `min(limit / peak, 1)` so nothing clips (YuE's `--rescale`).
    Rescale,
}
