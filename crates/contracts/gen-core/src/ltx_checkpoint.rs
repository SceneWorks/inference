//! Backend-neutral **LTX checkpoint layout** resolution (sc-18757): which on-disk layout a bundle
//! declares, where each component lives, and which slice of `__metadata__["config"]` that component
//! is allowed to read.
//!
//! # The two layouts
//!
//! * **All-in-one (LTX-2.3).** One flat `.safetensors` whose `__metadata__["config"]` carries
//!   `transformer` **and** `vae` **and** `audio_vae` **and** `vocoder` together, plus the
//!   text-embedding projection. Everything the engine needs (bar the separately-provisioned Gemma-3
//!   text encoder) rides one file. SceneWorks additionally ships a *converted* MLX form of the same
//!   model — one `.safetensors` per component next to an `embedded_config.json` — which is still the
//!   2.3 layout as far as this module is concerned: its manifest declares `model_version: "2.3.0"`.
//!
//! * **Split (LTX-2.5).** One `.safetensors` per component, each carrying **only its own** config
//!   section. The 2.5 transformer's `config.vae`, `config.audio_vae` and `config.vocoder` are
//!   explicitly `null`; the video VAE file carries `config.vae`; the audio VAE file carries
//!   `config.audio_vae` + `config.vocoder`; the duration head carries `config.duration_head` (plus
//!   the transformer dims it projects from); a latent upsampler carries a **bare** config with no
//!   wrapper section at all.
//!
//! # Selection is keyed on `model_version`
//!
//! Every 2.5 file stamps `__metadata__["model_version"] = "2.5.0"`. [`layout_for_version`] keys off
//! that string and **nothing else** — not the file name, not which files happen to be present. A
//! 2.5 bundle that is missing its audio VAE is a 2.5 bundle with a missing component (a
//! message-bearing error naming the component and the paths searched), never a 2.3 bundle.
//!
//! # No cross-component config defaulting
//!
//! [`LtxResolvedComponent::config_section`] reads **only** the section its own component owns, from
//! **its own** file, and treats a JSON `null` exactly like an absent key. There is deliberately no
//! "if this file has no `vae` block, fall back to the 2.3 shape" path: a video VAE file without
//! `config.vae` is an error, because silently substituting the 2.3 structure would build the wrong
//! decoder against 2.5 weights.
//!
//! # Gemma version assertion
//!
//! LTX-2.5 transformers stamp `gemma_source_checkpoint = {"ltx_version": "2.5.0", "gemma_version":
//! "gemma4-12b-ltx-v1"}`. Upstream's `encoder_configurator._check_gemma_version` **raises** when the
//! text encoder's declared `gemma_version` disagrees; [`check_gemma_version`] ports that check,
//! including the "checkpoints at or above 2.4.0 must declare a `gemma_source_checkpoint` at all"
//! rule and the pre-2.4 "must be a Gemma 3 encoder" fallback.
//!
//! Reference: `Lightricks/LTX-2` @ `d1511477` — `packages/ltx-core/src/ltx_core/loader/helpers.py`
//! (`parse_model_version`), `loader/sft_loader.py` (metadata reading),
//! `text_encoders/gemma/encoders/encoder_configurator.py` (`_check_gemma_version`),
//! `text_encoders/gemma/gemma_assets.py` (`gemma_config` metadata key), the per-component
//! `model_configurator.py` files, and `ltx-pipelines/utils/model_paths.py` (`ModelPaths`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::weightsmeta::{is_hidden_file, safetensors_file_metadata};
use crate::{Error, Result};

// =================================================================================================
// model_version
// =================================================================================================

/// The first `model_version` whose checkpoints ship as **per-component files** rather than one
/// all-in-one checkpoint. Compared as a version tuple, so a `"2.5.1"` patch release stays split.
pub const SPLIT_LAYOUT_SINCE: &[u32] = &[2, 5, 0];

/// The first `model_version` that is **required** to declare `gemma_source_checkpoint`. Upstream
/// sets this deliberately below the 2.5 stamp so pre-release builds are covered too; 2.3 / Gemma-3
/// checkpoints stay under it and take the legacy `model_type == "gemma3"` check instead.
pub const GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE: &[u32] = &[2, 4, 0];

/// The `__metadata__` key carrying the JSON-encoded per-component config object.
pub const CONFIG_METADATA_KEY: &str = "config";
/// The `__metadata__` key carrying the checkpoint's declared model version.
pub const MODEL_VERSION_METADATA_KEY: &str = "model_version";
/// The `__metadata__` key carrying `{"ltx_version": …, "gemma_version": …}` on 2.4+ transformers.
pub const GEMMA_SOURCE_CHECKPOINT_METADATA_KEY: &str = "gemma_source_checkpoint";
/// The `__metadata__` key under which a packed single-file text encoder stores its HF Gemma config.
pub const GEMMA_CONFIG_METADATA_KEY: &str = "gemma_config";

/// Parse a checkpoint's `model_version` into comparable numeric components.
///
/// Port of upstream `parse_model_version`. Parsing stops at the first dot-separated component that
/// is not a plain integer, so pre-release tags are dropped (`"2.3.rc1"` → `[2, 3]`). An unset or
/// non-numeric version parses to the empty vector, which orders below every real version — callers
/// get their oldest fallback, matching upstream.
///
/// Comparison uses Rust slice ordering, which is lexicographic with a length tiebreak — the same
/// semantics as Python tuple comparison, so `[2, 3] < [2, 4, 0]` and `[2, 5] < [2, 5, 0]` both hold
/// exactly as upstream sees them.
pub fn parse_model_version(version: Option<&str>) -> Vec<u32> {
    let Some(version) = version else {
        return Vec::new();
    };
    let mut parts = Vec::new();
    for part in version.split('.') {
        // `str::parse` accepts a leading `+`, which `str.isdigit()` rejects; screen for digits first
        // so `"2.+3"` truncates the way upstream truncates it instead of parsing as 3.
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        match part.parse::<u32>() {
            Ok(n) => parts.push(n),
            // A component too large for u32 is not a version this engine can order; stop, as
            // upstream would for any other unparsable component.
            Err(_) => break,
        }
    }
    parts
}

/// Which on-disk layout a checkpoint of this version ships as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LtxCheckpointLayout {
    /// One flat checkpoint carrying every component's config and weights (LTX-2.3 and older).
    AllInOne,
    /// One `.safetensors` per component, each carrying only its own config (LTX-2.5+).
    Split,
}

impl LtxCheckpointLayout {
    /// A short, stable name for error messages and logs.
    pub fn id(self) -> &'static str {
        match self {
            LtxCheckpointLayout::AllInOne => "all-in-one",
            LtxCheckpointLayout::Split => "split-component",
        }
    }
}

/// The layout a parsed [`parse_model_version`] tuple implies.
///
/// This is the **only** layout discriminator: never the file name, never which files exist.
pub fn layout_for_version(version: &[u32]) -> LtxCheckpointLayout {
    if version >= SPLIT_LAYOUT_SINCE {
        LtxCheckpointLayout::Split
    } else {
        LtxCheckpointLayout::AllInOne
    }
}

/// The layout a raw `model_version` string implies. An absent/unparsable version is the oldest
/// layout ([`LtxCheckpointLayout::AllInOne`]), matching upstream's "callers get their oldest
/// fallback".
pub fn layout_for_declared_version(version: Option<&str>) -> LtxCheckpointLayout {
    layout_for_version(&parse_model_version(version))
}

// =================================================================================================
// Components
// =================================================================================================

/// Where inside a component file's `__metadata__` that component's own config lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LtxConfigRoot {
    /// Named sub-objects of `__metadata__["config"]` — e.g. `config.vae`. The first entry is the
    /// component's **primary** section ([`LtxResolvedComponent::config`]).
    Sections(&'static [&'static str]),
    /// `__metadata__["config"]` itself, with no wrapper section (the latent upsamplers).
    BareConfig,
    /// `__metadata__["gemma_config"]` — a packed single-file text encoder's HF config.
    GemmaConfig,
}

/// One independently-resolved LTX component.
///
/// The set mirrors upstream `ModelPaths` (transformer / text encoder / video VAE / audio VAE /
/// duration head) plus the two latent upsamplers upstream passes as separate `--*-upsampler-path`
/// flags. The video VAE is **two** variants rather than one slot because LTX-2.5 ships both a
/// convolutional (`CausalVideoAutoencoder`) and a diffusion (`CausalDiffusionVAE`) decoder as
/// separate files with different structures; collapsing them into one slot would force a silent
/// pick between two real, differently-shaped decoders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LtxComponent {
    /// The `AVTransformer3DModel` denoiser.
    Transformer,
    /// The Gemma text encoder — a packed single file (2.5) or an HF snapshot directory (2.3).
    TextEncoder,
    /// The convolutional video VAE (`CausalVideoAutoencoder`).
    ConvVideoVae,
    /// The diffusion video VAE (`CausalDiffusionVAE`).
    DiffusionVideoVae,
    /// The audio VAE **and** the vocoder — one file carrying both config sections.
    AudioVae,
    /// The `DurationHead` model patch.
    DurationHead,
    /// The spatial `LatentUpsampler`.
    SpatialUpsampler,
    /// The temporal `LatentUpsampler`.
    TemporalUpsampler,
}

impl LtxComponent {
    /// Every component, in resolution order.
    pub const ALL: &'static [LtxComponent] = &[
        LtxComponent::Transformer,
        LtxComponent::TextEncoder,
        LtxComponent::ConvVideoVae,
        LtxComponent::DiffusionVideoVae,
        LtxComponent::AudioVae,
        LtxComponent::DurationHead,
        LtxComponent::SpatialUpsampler,
        LtxComponent::TemporalUpsampler,
    ];

    /// The stable component id used in errors and in caller-facing component maps.
    pub fn id(self) -> &'static str {
        match self {
            LtxComponent::Transformer => "transformer",
            LtxComponent::TextEncoder => "text_encoder",
            LtxComponent::ConvVideoVae => "conv_video_vae",
            LtxComponent::DiffusionVideoVae => "diffusion_video_vae",
            LtxComponent::AudioVae => "audio_vae",
            LtxComponent::DurationHead => "duration_head",
            LtxComponent::SpatialUpsampler => "spatial_upsampler",
            LtxComponent::TemporalUpsampler => "temporal_upsampler",
        }
    }

    /// Human-readable description, for the missing-component error.
    pub fn describe(self) -> &'static str {
        match self {
            LtxComponent::Transformer => "the AVTransformer3DModel denoiser",
            LtxComponent::TextEncoder => "the Gemma text encoder",
            LtxComponent::ConvVideoVae => "the convolutional video VAE (CausalVideoAutoencoder)",
            LtxComponent::DiffusionVideoVae => "the diffusion video VAE (CausalDiffusionVAE)",
            LtxComponent::AudioVae => "the audio VAE + vocoder",
            LtxComponent::DurationHead => "the duration head",
            LtxComponent::SpatialUpsampler => "the spatial latent upsampler",
            LtxComponent::TemporalUpsampler => "the temporal latent upsampler",
        }
    }

    /// Where this component's own config lives inside its file's `__metadata__`.
    pub fn config_root(self) -> LtxConfigRoot {
        match self {
            // `scheduler` rides the transformer file but is optional — the section list carries only
            // what the component *must* have; optional siblings go through `optional_section`.
            LtxComponent::Transformer => LtxConfigRoot::Sections(&["transformer"]),
            LtxComponent::TextEncoder => LtxConfigRoot::GemmaConfig,
            LtxComponent::ConvVideoVae | LtxComponent::DiffusionVideoVae => {
                LtxConfigRoot::Sections(&["vae"])
            }
            LtxComponent::AudioVae => LtxConfigRoot::Sections(&["audio_vae", "vocoder"]),
            // The duration head projects from the transformer's `cross_attention_dim` /
            // `audio_cross_attention_dim`, so its file re-declares those dims alongside its own head
            // hyperparameters. Both are required.
            LtxComponent::DurationHead => {
                LtxConfigRoot::Sections(&["duration_head", "transformer"])
            }
            LtxComponent::SpatialUpsampler | LtxComponent::TemporalUpsampler => {
                LtxConfigRoot::BareConfig
            }
        }
    }

    /// The `_class_name` this component's config declares, when the family pins one.
    pub fn declared_class(self) -> Option<&'static str> {
        match self {
            LtxComponent::Transformer => Some(TRANSFORMER_CLASS),
            LtxComponent::ConvVideoVae => Some(CONV_VIDEO_VAE_CLASS),
            LtxComponent::DiffusionVideoVae => Some(DIFFUSION_VIDEO_VAE_CLASS),
            LtxComponent::SpatialUpsampler | LtxComponent::TemporalUpsampler => {
                Some(LATENT_UPSAMPLER_CLASS)
            }
            LtxComponent::TextEncoder | LtxComponent::AudioVae | LtxComponent::DurationHead => None,
        }
    }

    /// Parse a component id (the inverse of [`id`](Self::id)).
    pub fn from_id(id: &str) -> Option<LtxComponent> {
        LtxComponent::ALL.iter().copied().find(|c| c.id() == id)
    }
}

/// `config.transformer._class_name` for the LTX-2.x denoiser.
pub const TRANSFORMER_CLASS: &str = "AVTransformer3DModel";
/// `config.vae._class_name` for the convolutional video VAE.
pub const CONV_VIDEO_VAE_CLASS: &str = "CausalVideoAutoencoder";
/// `config.vae._class_name` for the diffusion video VAE.
pub const DIFFUSION_VIDEO_VAE_CLASS: &str = "CausalDiffusionVAE";
/// The bare-root `_class_name` a latent upsampler declares.
pub const LATENT_UPSAMPLER_CLASS: &str = "LatentUpsampler";
/// The `model_type` an LTX-2.5 Gemma 4 text encoder declares.
pub const GEMMA4_UNIFIED_MODEL_TYPE: &str = "gemma4_unified";
/// The `model_type` an LTX-2.3 Gemma 3 text encoder declares.
pub const GEMMA3_MODEL_TYPE: &str = "gemma3";

// =================================================================================================
// Per-file metadata
// =================================================================================================

/// The `gemma_source_checkpoint` block an LTX-2.4+ transformer stamps.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GemmaSourceCheckpoint {
    /// The LTX version the Gemma weights were captured against (`"2.5.0"`).
    pub ltx_version: Option<String>,
    /// The Gemma checkpoint identity the transformer was trained with (`"gemma4-12b-ltx-v1"`).
    pub gemma_version: Option<String>,
}

impl GemmaSourceCheckpoint {
    fn from_value(v: &Value) -> GemmaSourceCheckpoint {
        GemmaSourceCheckpoint {
            ltx_version: string_field(v, "ltx_version"),
            gemma_version: string_field(v, "gemma_version"),
        }
    }
}

/// One checkpoint file's parsed `__metadata__`.
///
/// Values are JSON string-encoded by convention: `config` and `gemma_source_checkpoint` parse as
/// JSON, `model_version` and `license` stay raw strings. This mirrors upstream
/// `SafetensorsModelStateDictLoader.metadata`, which JSON-parses each value and keeps the raw string
/// when parsing fails.
#[derive(Clone, Debug, Default)]
pub struct LtxCheckpointMetadata {
    raw: BTreeMap<String, String>,
    model_version: Option<String>,
    config: Option<Value>,
    gemma_config: Option<Value>,
    gemma_source_checkpoint: Option<GemmaSourceCheckpoint>,
}

impl LtxCheckpointMetadata {
    /// Parse a raw `__metadata__` map (the shape
    /// [`safetensors_file_metadata`] returns).
    ///
    /// `source` names the file in error messages. A `config` / `gemma_config` /
    /// `gemma_source_checkpoint` value that is present but not valid JSON is an **error**: those
    /// three are structural, and treating a malformed blob as "absent" would fall through to a
    /// missing-section or missing-version message that hides the real cause.
    pub fn from_raw(source: &Path, raw: BTreeMap<String, String>) -> Result<Self> {
        let parse = |key: &str| -> Result<Option<Value>> {
            let Some(text) = raw.get(key) else {
                return Ok(None);
            };
            let value: Value = serde_json::from_str(text).map_err(|e| {
                Error::Msg(format!(
                    "ltx: {} __metadata__[{key:?}] is not valid JSON: {e}",
                    source.display()
                ))
            })?;
            Ok(Some(value))
        };
        let config = parse(CONFIG_METADATA_KEY)?;
        let gemma_config = parse(GEMMA_CONFIG_METADATA_KEY)?;
        let gemma_source_checkpoint = parse(GEMMA_SOURCE_CHECKPOINT_METADATA_KEY)?
            .filter(|v| !v.is_null())
            .as_ref()
            .map(GemmaSourceCheckpoint::from_value);
        Ok(LtxCheckpointMetadata {
            model_version: raw.get(MODEL_VERSION_METADATA_KEY).cloned(),
            config: config.filter(|v| !v.is_null()),
            gemma_config: gemma_config.filter(|v| !v.is_null()),
            gemma_source_checkpoint,
            raw,
        })
    }

    /// Read one `.safetensors` file's `__metadata__` (header only — no tensor data is touched, so
    /// this is safe on a 44 GB transformer).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::from_raw(path, safetensors_file_metadata(path)?)
    }

    /// The declared `model_version` string, verbatim.
    pub fn model_version(&self) -> Option<&str> {
        self.model_version.as_deref()
    }

    /// The declared version parsed into comparable components.
    pub fn version(&self) -> Vec<u32> {
        parse_model_version(self.model_version())
    }

    /// The layout this file's declared version implies.
    pub fn layout(&self) -> LtxCheckpointLayout {
        layout_for_version(&self.version())
    }

    /// The whole parsed `config` object, if the file carries one.
    pub fn config(&self) -> Option<&Value> {
        self.config.as_ref()
    }

    /// A named `config.<key>` sub-object. A JSON `null` is treated exactly like an absent key —
    /// LTX-2.5's transformer sets `config.vae`, `config.audio_vae` and `config.vocoder` to `null`
    /// precisely to say "this file does not carry that component".
    pub fn section(&self, key: &str) -> Option<&Value> {
        self.config
            .as_ref()?
            .get(key)
            .filter(|value| !value.is_null())
    }

    /// The packed single-file text encoder's HF Gemma config.
    pub fn gemma_config(&self) -> Option<&Value> {
        self.gemma_config.as_ref()
    }

    /// The transformer's `gemma_source_checkpoint` assertion, if it declares one.
    pub fn gemma_source_checkpoint(&self) -> Option<&GemmaSourceCheckpoint> {
        self.gemma_source_checkpoint.as_ref()
    }

    /// The unparsed `__metadata__` map.
    pub fn raw(&self) -> &BTreeMap<String, String> {
        &self.raw
    }

    /// Which component this file *declares itself* to be, from its config content alone.
    ///
    /// Never keyed on the file name: the same `CausalDiffusionVAE` weights are called
    /// `ltx-2.5-video-vae-bf16.safetensors` upstream and something else in a rehost, and a fine-tune
    /// may be named anything at all. Returns `None` for a file that carries no recognizable LTX
    /// component config (a LoRA, an IC-LoRA, a stray tensor dump).
    pub fn classify(&self) -> Option<LtxComponent> {
        if self.gemma_config.is_some() {
            return Some(LtxComponent::TextEncoder);
        }
        // Ordered most-specific first: a duration-head file also carries `config.transformer` (the
        // dims it projects from), so it must be tested before the plain transformer.
        if self.section("duration_head").is_some() {
            return Some(LtxComponent::DurationHead);
        }
        if self.section("audio_vae").is_some() {
            return Some(LtxComponent::AudioVae);
        }
        if let Some(vae) = self.section("vae") {
            return Some(match string_field(vae, "_class_name").as_deref() {
                Some(DIFFUSION_VIDEO_VAE_CLASS) => LtxComponent::DiffusionVideoVae,
                // Upstream's `_vae_class_name_from_metadata` defaults an unstamped `vae` block to
                // the conv class, so an older extract without `_class_name` stays conv.
                _ => LtxComponent::ConvVideoVae,
            });
        }
        if self.section("transformer").is_some() {
            return Some(LtxComponent::Transformer);
        }
        // A latent upsampler's config is bare: no wrapper section, `_class_name: "LatentUpsampler"`,
        // and a `spatial_upsample` / `temporal_upsample` pair that says which axis it scales.
        let config = self.config.as_ref()?;
        if string_field(config, "_class_name").as_deref() == Some(LATENT_UPSAMPLER_CLASS) {
            // Upstream `LatentUpsamplerConfigurator` defaults are `spatial_upsample=True`,
            // `temporal_upsample=False`, so an unstamped upsampler is the spatial one.
            let temporal = config
                .get("temporal_upsample")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            return Some(if temporal {
                LtxComponent::TemporalUpsampler
            } else {
                LtxComponent::SpatialUpsampler
            });
        }
        None
    }
}

fn string_field(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(std::string::ToString::to_string)
}

// =================================================================================================
// Caption feature-extractor selection (config-driven)
// =================================================================================================

/// Which caption feature extractor a transformer's config selects.
///
/// Upstream `encoder_configurator._create_feature_extractor` picks between two shapes, and picks
/// **only** from config — never from a weight-key probe or a per-model constant:
///
/// * **V1** — a single `aggregate_embed` projection living in the transformer.
/// * **V2** — per-token RMS norm with dual (video + audio) aggregate embeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionFeatureVersion {
    V1,
    V2,
}

/// Upstream's `_V2_EXPECTED_CONFIG`, verbatim: the four keys whose presence selects V2 and the exact
/// value each must carry.
pub const CAPTION_V2_EXPECTED_CONFIG: [(&str, bool); 4] = [
    ("caption_proj_before_connector", true),
    ("caption_projection_first_linear", false),
    ("caption_proj_input_norm", false),
    ("caption_projection_second_linear", false),
];

/// The subset of [`CAPTION_V2_EXPECTED_CONFIG`] the shipped LTX-2.3 checkpoints actually declare.
///
/// Measured, not assumed: the `SceneWorks/ltx-2.3-mlx` q4/q8 tiers' `embedded_config.json` carries
/// `caption_projection_first_linear: false` and `caption_projection_second_linear: false` and
/// **omits** `caption_proj_before_connector` / `caption_proj_input_norm` entirely, while declaring
/// `text_encoder_norm_type: "per_token_rms"` — i.e. it is a V2 checkpoint that predates the two
/// newer keys. Upstream's strict rule would call that "partial V2" and raise.
pub const CAPTION_V2_LEGACY_KEYS: [&str; 2] = [
    "caption_projection_first_linear",
    "caption_projection_second_linear",
];

/// Select the caption feature extractor from a **transformer config section**.
///
/// Port of upstream `_create_feature_extractor`'s detection, plus one measured carve-out:
///
/// 1. **None** of the four [`CAPTION_V2_EXPECTED_CONFIG`] keys present → [`CaptionFeatureVersion::V1`]
///    (the projection lives in the transformer).
/// 2. **All four** present → [`CaptionFeatureVersion::V2`] iff every value matches; any disagreement
///    is a hard error naming the offending key, its actual value and the expected one. This is
///    upstream's `NotImplementedError("Unknown config: …")`.
/// 3. **Exactly the two [`CAPTION_V2_LEGACY_KEYS`], both `false`** → [`CaptionFeatureVersion::V2`].
///    The shipped LTX-2.3 tiers are this shape and they are genuinely V2; erroring here would refuse
///    to load a checkpoint that has always worked. Deliberately narrow: both keys must be present
///    **and** `false`, and neither newer key may appear.
/// 4. **Any other partial combination** → a hard error naming the missing keys, matching upstream's
///    `NotImplementedError("Partial V2 config — missing keys: …")`. Config drift must fail loudly
///    rather than silently choose an extractor.
///
/// LTX-2.5 declares all four keys, so it takes rule 2; LTX-2.3 takes rule 3.
pub fn caption_feature_version(transformer: &Value) -> Result<CaptionFeatureVersion> {
    let present: Vec<(&str, bool, Option<bool>)> = CAPTION_V2_EXPECTED_CONFIG
        .iter()
        .map(|(name, expected)| {
            (
                *name,
                *expected,
                transformer.get(*name).and_then(Value::as_bool),
            )
        })
        .collect();
    // A key that is present but not a bool is drift, not absence — treat it as present-and-wrong so
    // it reports through the value-mismatch path instead of silently reading as missing.
    let declared: Vec<&str> = CAPTION_V2_EXPECTED_CONFIG
        .iter()
        .filter(|(name, _)| transformer.get(*name).is_some_and(|v| !v.is_null()))
        .map(|(name, _)| *name)
        .collect();

    if declared.is_empty() {
        return Ok(CaptionFeatureVersion::V1);
    }

    if declared.len() == CAPTION_V2_EXPECTED_CONFIG.len() {
        let mismatched: Vec<String> = present
            .iter()
            .filter(|(_, expected, actual)| *actual != Some(*expected))
            .map(|(name, expected, actual)| match actual {
                Some(value) => format!("{name}={value} (expected {expected})"),
                None => format!("{name}=<non-boolean> (expected {expected})"),
            })
            .collect();
        if mismatched.is_empty() {
            return Ok(CaptionFeatureVersion::V2);
        }
        return Err(Error::Msg(format!(
            "ltx: unsupported caption-projection config: {}. Only the V2 shape upstream's \
             _V2_EXPECTED_CONFIG pins is implemented; this checkpoint's caption stack differs and \
             would be silently mis-built",
            mismatched.join(", ")
        )));
    }

    // Rule 3 — the measured LTX-2.3 shape: exactly the two legacy keys, both false.
    let legacy_only = declared.len() == CAPTION_V2_LEGACY_KEYS.len()
        && CAPTION_V2_LEGACY_KEYS
            .iter()
            .all(|key| declared.contains(key));
    if legacy_only {
        let both_false = CAPTION_V2_LEGACY_KEYS
            .iter()
            .all(|key| transformer.get(*key).and_then(Value::as_bool) == Some(false));
        if both_false {
            return Ok(CaptionFeatureVersion::V2);
        }
    }

    let missing: Vec<&str> = CAPTION_V2_EXPECTED_CONFIG
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !declared.contains(name))
        .collect();
    Err(Error::Msg(format!(
        "ltx: partial caption-projection config — declared [{}] but missing [{}]. The only accepted \
         partial shape is the shipped LTX-2.3 pair ({}) with both false; anything else is config \
         drift and must not silently select a feature extractor",
        declared.join(", "),
        missing.join(", "),
        CAPTION_V2_LEGACY_KEYS.join(", "),
    )))
}

// =================================================================================================
// Gemma text-encoder identity + the version assertion
// =================================================================================================

/// What a Gemma text encoder declares about itself: its `model_type` and (2.5+) its
/// `gemma_version`.
///
/// Read from an HF snapshot directory's `config.json` or from a packed single-file encoder's
/// `__metadata__["gemma_config"]` — the two layouts upstream `GemmaAssets.load` accepts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemmaEncoderIdentity {
    /// The path the identity was read from, for error messages.
    pub source: PathBuf,
    /// `model_type` — `"gemma3"` for LTX-2.3, `"gemma4_unified"` for LTX-2.5.
    pub model_type: Option<String>,
    /// `gemma_version` — the identity a 2.4+ checkpoint's `gemma_source_checkpoint` must match.
    pub gemma_version: Option<String>,
}

impl GemmaEncoderIdentity {
    /// Read the identity out of an already-parsed HF config object.
    pub fn from_config_value(source: impl Into<PathBuf>, config: &Value) -> Self {
        GemmaEncoderIdentity {
            source: source.into(),
            model_type: string_field(config, "model_type"),
            gemma_version: string_field(config, "gemma_version"),
        }
    }

    /// Read `<dir>/config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("config.json");
        if !path.exists() {
            return Err(Error::Msg(format!(
                "ltx: the Gemma text-encoder snapshot {} has no config.json — its model_type / \
                 gemma_version cannot be verified against the checkpoint",
                dir.display()
            )));
        }
        let text = std::fs::read_to_string(&path)?;
        let config: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse {}: {e}", path.display())))?;
        Ok(Self::from_config_value(dir, &config))
    }

    /// Read a packed single-file encoder's `__metadata__["gemma_config"]`.
    pub fn from_single_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let meta = LtxCheckpointMetadata::from_file(path)?;
        let config = meta.gemma_config().ok_or_else(|| {
            Error::Msg(format!(
                "ltx: the packed text encoder {} is missing __metadata__[{GEMMA_CONFIG_METADATA_KEY:?}] \
                 (the JSON-encoded HuggingFace Gemma config)",
                path.display()
            ))
        })?;
        Ok(Self::from_config_value(path, config))
    }

    /// Read whichever of the two layouts `path` is: a directory → `config.json`, a `.safetensors`
    /// file → its packed `gemma_config` metadata.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::from_dir(path);
        }
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            return Self::from_single_file(path);
        }
        Err(Error::Msg(format!(
            "ltx: the Gemma text-encoder path {} is neither a snapshot directory nor a \
             .safetensors file",
            path.display()
        )))
    }
}

/// How a text encoder satisfied [`check_gemma_version`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GemmaVersionCheck {
    /// The checkpoint declared a `gemma_source_checkpoint.gemma_version` and the encoder matched it.
    Matched(String),
    /// A pre-2.4 checkpoint paired with a `model_type == "gemma3"` encoder (the LTX-2.3 route).
    Gemma3Legacy,
    /// A 2.4+ checkpoint with no `gemma_source_checkpoint` **and** an encoder that declares no
    /// `gemma_version` either — upstream logs a warning and skips rather than failing, because
    /// there is nothing on either side to compare. Callers that want to surface that ambiguity have
    /// it typed here instead of buried in a log line.
    SkippedNoDeclaredVersion,
}

/// Verify the Gemma text encoder matches what the checkpoint expects.
///
/// Port of upstream `encoder_configurator._check_gemma_version`:
///
/// * A checkpoint that declares `gemma_source_checkpoint` (LTX-2.5 / Gemma 4) must match its
///   `gemma_version` against the encoder's — a mismatch is a **hard error**, never a warning.
/// * A checkpoint at or above [`GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE`] that declares **no**
///   `gemma_source_checkpoint` is itself an error, unless the encoder declares no `gemma_version`
///   either (then there is nothing to compare and the check is skipped).
/// * An older checkpoint must be paired with a Gemma 3 encoder.
pub fn check_gemma_version(
    checkpoint: &LtxCheckpointMetadata,
    encoder: &GemmaEncoderIdentity,
) -> Result<GemmaVersionCheck> {
    if let Some(source) = checkpoint.gemma_source_checkpoint() {
        let expected = source.gemma_version.as_deref();
        let actual = encoder.gemma_version.as_deref();
        if expected != actual {
            return Err(Error::Msg(format!(
                "ltx: Gemma version mismatch — the checkpoint's gemma_source_checkpoint expects \
                 gemma_version={expected:?}, but the Gemma config at {} declares \
                 gemma_version={actual:?}",
                encoder.source.display()
            )));
        }
        return Ok(GemmaVersionCheck::Matched(
            expected.unwrap_or_default().to_string(),
        ));
    }

    let version = checkpoint.version();
    if checkpoint.model_version().is_some()
        && version.as_slice() >= GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE
    {
        if encoder.gemma_version.is_none() {
            return Ok(GemmaVersionCheck::SkippedNoDeclaredVersion);
        }
        return Err(Error::Msg(format!(
            "ltx: the checkpoint declares model_version={:?} and so must declare \
             gemma_source_checkpoint.gemma_version, but none was found (the Gemma config at {} \
             declares gemma_version={:?})",
            checkpoint.model_version().unwrap_or_default(),
            encoder.source.display(),
            encoder.gemma_version.as_deref().unwrap_or_default(),
        )));
    }

    if encoder.model_type.as_deref() != Some(GEMMA3_MODEL_TYPE) {
        return Err(Error::Msg(format!(
            "ltx: the checkpoint has no gemma_source_checkpoint (model_version={:?}), so it \
             expects a Gemma 3 encoder (model_type={GEMMA3_MODEL_TYPE:?}) at {}, but that config \
             declares model_type={:?}",
            checkpoint.model_version().unwrap_or_default(),
            encoder.source.display(),
            encoder.model_type.as_deref().unwrap_or_default(),
        )));
    }
    Ok(GemmaVersionCheck::Gemma3Legacy)
}

// =================================================================================================
// Resolved components + the bundle
// =================================================================================================

/// One component, resolved to a concrete file with its own parsed metadata.
#[derive(Clone, Debug)]
pub struct LtxResolvedComponent {
    component: LtxComponent,
    path: PathBuf,
    metadata: LtxCheckpointMetadata,
}

impl LtxResolvedComponent {
    /// Read a component file directly (existence-checked, header-only metadata read).
    pub fn open(component: LtxComponent, path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            return Err(Error::Msg(format!(
                "ltx: the {} component ({}) path does not exist: {}",
                component.id(),
                component.describe(),
                path.display()
            )));
        }
        // A packed text encoder is a `.safetensors`; a 2.3 Gemma snapshot is a directory. Only the
        // former carries safetensors metadata.
        let metadata = if path.is_dir() {
            LtxCheckpointMetadata::default()
        } else {
            LtxCheckpointMetadata::from_file(&path)?
        };
        Ok(LtxResolvedComponent {
            component,
            path,
            metadata,
        })
    }

    /// Which component this is.
    pub fn component(&self) -> LtxComponent {
        self.component
    }

    /// The resolved file (or, for a Gemma snapshot text encoder, directory).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// This file's parsed `__metadata__`.
    pub fn metadata(&self) -> &LtxCheckpointMetadata {
        &self.metadata
    }

    /// This component's **primary** config: the first required section, or the bare config object
    /// for the latent upsamplers, or the packed HF config for the text encoder.
    ///
    /// Never falls back to another component's config or to a hardcoded default shape.
    pub fn config(&self) -> Result<&Value> {
        match self.component.config_root() {
            LtxConfigRoot::Sections(sections) => {
                let key = sections.first().copied().unwrap_or(CONFIG_METADATA_KEY);
                self.config_section(key)
            }
            LtxConfigRoot::BareConfig => {
                self.metadata.config().ok_or_else(|| self.missing_config())
            }
            LtxConfigRoot::GemmaConfig => self
                .metadata
                .gemma_config()
                .ok_or_else(|| self.missing_config()),
        }
    }

    /// A named `config.<key>` section of **this** file. Absent (or JSON `null`) is an error naming
    /// the component, the section and the file — the 2.5 transformer's `config.vae: null` is exactly
    /// this case, and must never silently resolve to the 2.3 VAE shape.
    pub fn config_section(&self, key: &str) -> Result<&Value> {
        self.metadata.section(key).ok_or_else(|| {
            Error::Msg(format!(
                "ltx: the {} component file {} carries no `config.{key}` section — {} must read its \
                 own config; no other component's config or built-in default is substituted",
                self.component.id(),
                self.path.display(),
                self.component.describe(),
            ))
        })
    }

    /// A named section that the component may legitimately lack (e.g. the transformer's
    /// `scheduler`).
    pub fn optional_section(&self, key: &str) -> Option<&Value> {
        self.metadata.section(key)
    }

    /// Every required section of this component, in declaration order. Errors on the first missing
    /// one.
    pub fn required_sections(&self) -> Result<Vec<(&'static str, &Value)>> {
        match self.component.config_root() {
            LtxConfigRoot::Sections(sections) => sections
                .iter()
                .map(|key| self.config_section(key).map(|value| (*key, value)))
                .collect(),
            LtxConfigRoot::BareConfig | LtxConfigRoot::GemmaConfig => {
                self.config().map(|value| vec![("config", value)])
            }
        }
    }

    fn missing_config(&self) -> Error {
        let key = match self.component.config_root() {
            LtxConfigRoot::GemmaConfig => GEMMA_CONFIG_METADATA_KEY,
            _ => CONFIG_METADATA_KEY,
        };
        Error::Msg(format!(
            "ltx: the {} component file {} carries no `__metadata__[{key:?}]` — {} must read its \
             own config",
            self.component.id(),
            self.path.display(),
            self.component.describe(),
        ))
    }
}

/// A resolved LTX bundle: which layout it declares, and where each component lives.
#[derive(Clone, Debug)]
pub struct LtxBundle {
    model_version: Option<String>,
    components: BTreeMap<LtxComponent, LtxResolvedComponent>,
    searched: Vec<PathBuf>,
}

impl LtxBundle {
    /// The declared `model_version` shared by every resolved component.
    pub fn model_version(&self) -> Option<&str> {
        self.model_version.as_deref()
    }

    /// The parsed version tuple.
    pub fn version(&self) -> Vec<u32> {
        parse_model_version(self.model_version())
    }

    /// The layout the declared version implies.
    pub fn layout(&self) -> LtxCheckpointLayout {
        layout_for_version(&self.version())
    }

    /// A resolved component, or `None` when the bundle does not carry it.
    pub fn get(&self, component: LtxComponent) -> Option<&LtxResolvedComponent> {
        self.components.get(&component)
    }

    /// A resolved component, or a message-bearing error naming the component **and** every path
    /// searched.
    pub fn require(&self, component: LtxComponent) -> Result<&LtxResolvedComponent> {
        self.components.get(&component).ok_or_else(|| {
            Error::Msg(format!(
                "ltx: missing component `{}` ({}) in the {} bundle{}; searched: {}",
                component.id(),
                component.describe(),
                self.layout().id(),
                match self.model_version() {
                    Some(v) => format!(" declaring model_version {v:?}"),
                    None => String::new(),
                },
                format_searched(&self.searched),
            ))
        })
    }

    /// This component's primary config (shorthand for `require(c)?.config()`).
    pub fn component_config(&self, component: LtxComponent) -> Result<&Value> {
        self.require(component)?.config()
    }

    /// Every resolved component, in [`LtxComponent::ALL`] order.
    pub fn components(&self) -> impl Iterator<Item = &LtxResolvedComponent> {
        self.components.values()
    }

    /// Every path examined while resolving this bundle.
    pub fn searched(&self) -> &[PathBuf] {
        &self.searched
    }

    /// Run the Gemma assertion for this bundle: the transformer's `gemma_source_checkpoint` against
    /// the provided encoder identity. Errors if the bundle has no transformer to assert from.
    pub fn check_gemma_version(&self, encoder: &GemmaEncoderIdentity) -> Result<GemmaVersionCheck> {
        let transformer = self.require(LtxComponent::Transformer)?;
        check_gemma_version(transformer.metadata(), encoder)
    }
}

fn format_searched(paths: &[PathBuf]) -> String {
    const MAX: usize = 12;
    if paths.is_empty() {
        return "(no candidate paths)".to_string();
    }
    let shown: Vec<String> = paths
        .iter()
        .take(MAX)
        .map(|p| p.display().to_string())
        .collect();
    if paths.len() > MAX {
        format!("{} (+{} more)", shown.join(", "), paths.len() - MAX)
    } else {
        shown.join(", ")
    }
}

/// Assemble an [`LtxBundle`] from explicitly-provisioned component paths.
///
/// This is the caller-driven route (SceneWorks stages every component path before load; there is no
/// environment side-channel and no HF-cache scan). [`discover_split_bundle`] is the directory-scan
/// convenience on top of it.
#[derive(Clone, Debug, Default)]
pub struct LtxBundleBuilder {
    entries: Vec<(LtxComponent, PathBuf)>,
    searched: Vec<PathBuf>,
}

impl LtxBundleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Provision one component. A later call for the same component replaces the earlier one.
    #[must_use]
    pub fn with_component(mut self, component: LtxComponent, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        self.entries.retain(|(c, _)| *c != component);
        self.entries.push((component, path));
        self
    }

    /// Record a path that was examined but did not become a component, so the missing-component
    /// error can name it.
    #[must_use]
    pub fn with_searched(mut self, path: impl Into<PathBuf>) -> Self {
        self.searched.push(path.into());
        self
    }

    /// Resolve and validate every provisioned component.
    ///
    /// Validation, in order:
    /// 1. every path exists (error names the component and the path);
    /// 2. each file's metadata parses;
    /// 3. each file's own config says it *is* the component it was provisioned as — a caller that
    ///    wires the audio VAE into the video VAE slot fails here rather than building the wrong
    ///    decoder;
    /// 4. every component agrees on `model_version` — a mixed bundle is refused, naming both files.
    pub fn build(self) -> Result<LtxBundle> {
        let mut searched = self.searched;
        let mut components = BTreeMap::new();
        let mut declared: Option<(String, PathBuf)> = None;

        for (component, path) in self.entries {
            searched.push(path.clone());
            let resolved = LtxResolvedComponent::open(component, path)?;

            // (3) self-declaration check. A directory-provisioned text encoder (the 2.3 Gemma
            // snapshot) carries no safetensors metadata at all, so there is nothing to check.
            if !resolved.path.is_dir() {
                if let Some(actual) = resolved.metadata.classify() {
                    if actual != component {
                        return Err(Error::Msg(format!(
                            "ltx: {} was provisioned as the `{}` component, but its config declares \
                             it is `{}` ({})",
                            resolved.path.display(),
                            component.id(),
                            actual.id(),
                            actual.describe(),
                        )));
                    }
                }
            }

            // (4) one version across the bundle.
            if let Some(version) = resolved.metadata.model_version() {
                match &declared {
                    Some((seen, seen_path)) if seen != version => {
                        return Err(Error::Msg(format!(
                            "ltx: mixed model_version in one bundle — {} declares {seen:?} but {} \
                             declares {version:?}; every component of a bundle must come from the \
                             same release",
                            seen_path.display(),
                            resolved.path.display(),
                        )));
                    }
                    Some(_) => {}
                    None => declared = Some((version.to_string(), resolved.path.clone())),
                }
            }

            components.insert(component, resolved);
        }

        Ok(LtxBundle {
            model_version: declared.map(|(v, _)| v),
            components,
            searched,
        })
    }
}

/// The manifest a SceneWorks-converted LTX-2.3 tree ships beside its per-component files. The
/// converter re-emits those files without `__metadata__`, so the manifest is their only version
/// declaration.
pub const SPLIT_MANIFEST_FILE: &str = "split_model.json";

/// The `model_version` a checkpoint location declares, or `None` when nothing there declares one.
///
/// `root` may be a single `.safetensors` file (the upstream all-in-one checkpoint) or a directory.
/// For a directory the order of authority is:
///
/// 1. `split_model.json`'s `model_version` — the SceneWorks-converted LTX-2.3 manifest;
/// 2. the first `.safetensors` in the tree (sorted, so the answer is deterministic) that stamps
///    `__metadata__["model_version"]`.
///
/// The result is what [`layout_for_declared_version`] keys on. It deliberately does not depend on
/// which components are present, so removing a component from a bundle can never change its layout.
pub fn declared_model_version(root: impl AsRef<Path>) -> Result<Option<String>> {
    let root = root.as_ref();
    if root.is_file() {
        return Ok(LtxCheckpointMetadata::from_file(root)?
            .model_version()
            .map(str::to_string));
    }
    if !root.is_dir() {
        return Ok(None);
    }
    let manifest = root.join(SPLIT_MANIFEST_FILE);
    if manifest.exists() {
        let text = std::fs::read_to_string(&manifest)?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse {}: {e}", manifest.display())))?;
        if let Some(version) = value
            .get(MODEL_VERSION_METADATA_KEY)
            .and_then(Value::as_str)
        {
            return Ok(Some(version.to_string()));
        }
    }
    let mut files = Vec::new();
    collect_safetensors(root, &mut files)?;
    files.sort();
    for path in files {
        if let Some(version) = LtxCheckpointMetadata::from_file(&path)?.model_version() {
            return Ok(Some(version.to_string()));
        }
    }
    Ok(None)
}

/// The layout a checkpoint location declares. An undeclared version is
/// [`LtxCheckpointLayout::AllInOne`] — the oldest layout, matching upstream's fallback — so every
/// pre-`model_version` LTX-2.3 tree stays on exactly the path it has always taken.
pub fn declared_layout(root: impl AsRef<Path>) -> Result<LtxCheckpointLayout> {
    Ok(layout_for_declared_version(
        declared_model_version(root)?.as_deref(),
    ))
}

/// Discover a split bundle by scanning `root` for `.safetensors` files and **classifying each by
/// its own metadata**.
///
/// Never keyed on file names: upstream's `vae/ltx-2.5-video-vae-conv-bf16.safetensors` and a
/// differently-named rehost of the same weights resolve identically. Files that carry no
/// recognizable LTX component config (LoRAs, IC-LoRAs, stray dumps) are skipped but still recorded
/// as searched, so a missing-component error names them.
///
/// Two files classifying as the same component is an **error** naming both — there is no "pick the
/// bigger one" tiebreak, because both are plausibly real and guessing would silently load the wrong
/// weights. When the caller has already provisioned that component explicitly, use
/// [`discover_split_bundle_skipping`] so the scan leaves the slot alone instead of tripping over an
/// ambiguity the caller has already resolved.
pub fn discover_split_bundle(root: impl AsRef<Path>) -> Result<LtxBundle> {
    discover_split_bundle_skipping(root, &[])
}

/// [`discover_split_bundle`], but the scan does not claim any component in `skip`.
///
/// Files that classify as a skipped component are still recorded as searched, so a later
/// missing-component error names them; they simply cannot fill (or contend for) that slot. This is
/// what lets an explicitly-provisioned component win over a directory that ships two plausible
/// candidates for it: without the skip, discovery would refuse the whole bundle as ambiguous before
/// the caller's own choice could be applied.
pub fn discover_split_bundle_skipping(
    root: impl AsRef<Path>,
    skip: &[LtxComponent],
) -> Result<LtxBundle> {
    let root = root.as_ref();
    if !root.is_dir() {
        return Err(Error::Msg(format!(
            "ltx: the bundle root {} is not a directory",
            root.display()
        )));
    }
    let mut files = Vec::new();
    collect_safetensors(root, &mut files)?;
    files.sort();

    let mut builder = LtxBundleBuilder::new();
    let mut claimed: BTreeMap<LtxComponent, PathBuf> = BTreeMap::new();
    for path in files {
        let metadata = LtxCheckpointMetadata::from_file(&path)?;
        let Some(component) = metadata.classify() else {
            builder = builder.with_searched(path);
            continue;
        };
        if skip.contains(&component) {
            builder = builder.with_searched(path);
            continue;
        }
        if let Some(existing) = claimed.get(&component) {
            return Err(Error::Msg(format!(
                "ltx: ambiguous `{}` component under {} — both {} and {} declare it; provision the \
                 component paths explicitly instead of relying on a directory scan",
                component.id(),
                root.display(),
                existing.display(),
                path.display(),
            )));
        }
        claimed.insert(component, path.clone());
        builder = builder.with_component(component, path);
    }
    builder.build()
}

fn collect_safetensors(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if is_hidden_file(&path) {
            continue;
        }
        // `file_type` does not follow symlinks, which is what we want for directories (an HF cache
        // snapshot links its blobs) while still admitting linked *files*.
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_safetensors(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(source: &str, entries: &[(&str, &str)]) -> LtxCheckpointMetadata {
        let raw = entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        LtxCheckpointMetadata::from_raw(Path::new(source), raw).expect("metadata parses")
    }

    /// The LTX-2.5 transformer file: `transformer` + `scheduler` only; every other component's
    /// section is explicitly `null`.
    fn ltx25_transformer_metadata() -> LtxCheckpointMetadata {
        meta(
            "/bundle/diffusion_models/transformer.safetensors",
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    GEMMA_SOURCE_CHECKPOINT_METADATA_KEY,
                    r#"{"ltx_version":"2.5.0","gemma_version":"gemma4-12b-ltx-v1"}"#,
                ),
                (
                    CONFIG_METADATA_KEY,
                    r#"{
                        "transformer": {"_class_name":"AVTransformer3DModel","num_layers":48},
                        "scheduler": {"_class_name":"RectifiedFlowScheduler"},
                        "vae": null,
                        "audio_vae": null,
                        "vocoder": null
                    }"#,
                ),
            ],
        )
    }

    // --- model_version ---------------------------------------------------------------------------

    #[test]
    fn parse_model_version_matches_upstream_truncation() {
        assert_eq!(parse_model_version(Some("2.5.0")), vec![2, 5, 0]);
        assert_eq!(parse_model_version(Some("2.3.0")), vec![2, 3, 0]);
        // Parsing stops at the first non-integer component (the pre-2.4 pre-release tags).
        assert_eq!(parse_model_version(Some("2.3.rc1")), vec![2, 3]);
        // Tags are not always dot-separated; upstream documents `"2.4-rc2"` → `(2,)`.
        assert_eq!(parse_model_version(Some("2.4-rc2")), vec![2]);
        assert_eq!(parse_model_version(Some("")), Vec::<u32>::new());
        assert_eq!(parse_model_version(None), Vec::<u32>::new());
        // A `+`-prefixed component is not `isdigit()` upstream, so it truncates rather than parsing.
        assert_eq!(parse_model_version(Some("2.+3")), vec![2]);
    }

    #[test]
    fn version_ordering_matches_python_tuple_semantics() {
        // (2, 3) < (2, 4, 0) — a shorter numeric prefix compares below a longer one sharing it.
        assert!(parse_model_version(Some("2.3")) < GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE.to_vec());
        // (2, 5) < (2, 5, 0) — the same rule, which is why a bare "2.5" is NOT yet the split layout.
        assert!(parse_model_version(Some("2.5")) < SPLIT_LAYOUT_SINCE.to_vec());
        assert!(parse_model_version(Some("2.5.0")) >= SPLIT_LAYOUT_SINCE.to_vec());
        assert!(parse_model_version(Some("2.6.0")) >= SPLIT_LAYOUT_SINCE.to_vec());
    }

    #[test]
    fn layout_is_keyed_on_version_never_on_names() {
        assert_eq!(
            layout_for_declared_version(Some("2.3.0")),
            LtxCheckpointLayout::AllInOne
        );
        assert_eq!(
            layout_for_declared_version(Some("2.5.0")),
            LtxCheckpointLayout::Split
        );
        assert_eq!(
            layout_for_declared_version(Some("2.5.1")),
            LtxCheckpointLayout::Split
        );
        // An undeclared version falls back to the OLDEST layout, matching upstream.
        assert_eq!(
            layout_for_declared_version(None),
            LtxCheckpointLayout::AllInOne
        );
    }

    // --- metadata parsing ------------------------------------------------------------------------

    #[test]
    fn null_sections_are_absent_not_empty() {
        let m = ltx25_transformer_metadata();
        assert!(m.section("transformer").is_some());
        assert!(m.section("scheduler").is_some());
        // The three the 2.5 transformer explicitly nulls out.
        assert!(m.section("vae").is_none());
        assert!(m.section("audio_vae").is_none());
        assert!(m.section("vocoder").is_none());
        assert_eq!(m.model_version(), Some("2.5.0"));
        assert_eq!(m.layout(), LtxCheckpointLayout::Split);
    }

    #[test]
    fn malformed_structural_metadata_is_an_error_not_a_silent_absence() {
        let raw: BTreeMap<String, String> =
            [(CONFIG_METADATA_KEY.to_string(), "{not json".to_string())]
                .into_iter()
                .collect();
        let err = LtxCheckpointMetadata::from_raw(Path::new("/x.safetensors"), raw)
            .expect_err("a malformed config blob must not parse as absent");
        let text = err.to_string();
        assert!(text.contains("not valid JSON"), "{text}");
        assert!(text.contains("/x.safetensors"), "{text}");
    }

    #[test]
    fn gemma_source_checkpoint_parses_both_fields() {
        let m = ltx25_transformer_metadata();
        let gsc = m.gemma_source_checkpoint().expect("declared");
        assert_eq!(gsc.ltx_version.as_deref(), Some("2.5.0"));
        assert_eq!(gsc.gemma_version.as_deref(), Some("gemma4-12b-ltx-v1"));
    }

    // --- classification --------------------------------------------------------------------------

    #[test]
    fn classification_reads_config_content_not_file_names() {
        assert_eq!(
            ltx25_transformer_metadata().classify(),
            Some(LtxComponent::Transformer)
        );
        // Deliberately mis-named files: the classification must follow the config, not the stem.
        let conv = meta(
            "/bundle/vae/totally-not-a-vae.safetensors",
            &[(
                CONFIG_METADATA_KEY,
                r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":128}}"#,
            )],
        );
        assert_eq!(conv.classify(), Some(LtxComponent::ConvVideoVae));
        let diff = meta(
            "/bundle/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
            &[(
                CONFIG_METADATA_KEY,
                r#"{"vae":{"_class_name":"CausalDiffusionVAE","decoder":{"head_dim":64}}}"#,
            )],
        );
        assert_eq!(diff.classify(), Some(LtxComponent::DiffusionVideoVae));
        let audio = meta(
            "/bundle/vae/a.safetensors",
            &[(
                CONFIG_METADATA_KEY,
                r#"{"audio_vae":{"model":{"params":{"ddconfig":{"ch":128}}}},"vocoder":{"vocoder":{}}}"#,
            )],
        );
        assert_eq!(audio.classify(), Some(LtxComponent::AudioVae));
        // The duration head also carries `config.transformer` — it must not classify as the DiT.
        let duration = meta(
            "/bundle/model_patches/d.safetensors",
            &[(
                CONFIG_METADATA_KEY,
                r#"{"transformer":{"cross_attention_dim":4096},"duration_head":{"num_queries":1}}"#,
            )],
        );
        assert_eq!(duration.classify(), Some(LtxComponent::DurationHead));
        let te = meta(
            "/bundle/text_encoders/te.safetensors",
            &[(
                GEMMA_CONFIG_METADATA_KEY,
                r#"{"model_type":"gemma4_unified","gemma_version":"gemma4-12b-ltx-v1"}"#,
            )],
        );
        assert_eq!(te.classify(), Some(LtxComponent::TextEncoder));
    }

    #[test]
    fn upsamplers_classify_off_the_bare_config_axis_flags() {
        let spatial = meta(
            "/bundle/latent_upscale_models/s.safetensors",
            &[(
                CONFIG_METADATA_KEY,
                r#"{"_class_name":"LatentUpsampler","spatial_upsample":true,"temporal_upsample":false,"spatial_scale":2.0}"#,
            )],
        );
        assert_eq!(spatial.classify(), Some(LtxComponent::SpatialUpsampler));
        let temporal = meta(
            "/bundle/latent_upscale_models/t.safetensors",
            &[(
                CONFIG_METADATA_KEY,
                r#"{"_class_name":"LatentUpsampler","spatial_upsample":false,"temporal_upsample":true}"#,
            )],
        );
        assert_eq!(temporal.classify(), Some(LtxComponent::TemporalUpsampler));
        // Upstream's configurator defaults `temporal_upsample=False`, so an unstamped upsampler is
        // spatial rather than unclassifiable.
        let bare = meta(
            "/bundle/latent_upscale_models/u.safetensors",
            &[(CONFIG_METADATA_KEY, r#"{"_class_name":"LatentUpsampler"}"#)],
        );
        assert_eq!(bare.classify(), Some(LtxComponent::SpatialUpsampler));
    }

    #[test]
    fn a_non_ltx_file_classifies_as_nothing() {
        let lora = meta(
            "/bundle/loras/x.safetensors",
            &[("networkType", "lokr"), ("rank", "32")],
        );
        assert_eq!(lora.classify(), None);
    }

    // --- the Gemma assertion ---------------------------------------------------------------------

    fn gemma(model_type: Option<&str>, gemma_version: Option<&str>) -> GemmaEncoderIdentity {
        GemmaEncoderIdentity {
            source: PathBuf::from("/te/gemma"),
            model_type: model_type.map(str::to_string),
            gemma_version: gemma_version.map(str::to_string),
        }
    }

    #[test]
    fn matching_gemma_version_passes() {
        let check = check_gemma_version(
            &ltx25_transformer_metadata(),
            &gemma(Some(GEMMA4_UNIFIED_MODEL_TYPE), Some("gemma4-12b-ltx-v1")),
        )
        .expect("matching versions");
        assert_eq!(
            check,
            GemmaVersionCheck::Matched("gemma4-12b-ltx-v1".to_string())
        );
    }

    #[test]
    fn a_2_5_bundle_with_a_gemma_3_encoder_is_a_hard_version_mismatch() {
        // The acceptance case: LTX-2.5 weights + the LTX-2.3 Gemma-3 encoder. Upstream RAISES here;
        // a warning-and-continue would silently produce garbage embeddings.
        let err = check_gemma_version(
            &ltx25_transformer_metadata(),
            &gemma(Some(GEMMA3_MODEL_TYPE), None),
        )
        .expect_err("a Gemma 3 encoder must not satisfy a 2.5 checkpoint");
        let text = err.to_string();
        assert!(text.contains("Gemma version mismatch"), "{text}");
        assert!(text.contains("gemma4-12b-ltx-v1"), "{text}");
        assert!(text.contains("/te/gemma"), "{text}");
    }

    #[test]
    fn a_wrong_gemma_4_generation_is_also_a_mismatch() {
        let err = check_gemma_version(
            &ltx25_transformer_metadata(),
            &gemma(Some(GEMMA4_UNIFIED_MODEL_TYPE), Some("gemma4-12b-ltx-v0")),
        )
        .expect_err("a different gemma_version must not pass");
        assert!(err.to_string().contains("gemma4-12b-ltx-v0"));
    }

    #[test]
    fn a_2_3_bundle_requires_a_gemma_3_encoder() {
        let ltx23 = meta(
            "/bundle/ltx-2.3-22b-distilled.safetensors",
            &[
                (MODEL_VERSION_METADATA_KEY, "2.3.0"),
                (CONFIG_METADATA_KEY, r#"{"transformer":{},"vae":{}}"#),
            ],
        );
        assert_eq!(
            check_gemma_version(&ltx23, &gemma(Some(GEMMA3_MODEL_TYPE), None)).unwrap(),
            GemmaVersionCheck::Gemma3Legacy
        );
        let err = check_gemma_version(&ltx23, &gemma(Some(GEMMA4_UNIFIED_MODEL_TYPE), None))
            .expect_err("a Gemma 4 encoder must not satisfy a 2.3 checkpoint");
        assert!(err.to_string().contains("gemma3"));
    }

    #[test]
    fn a_2_4_plus_checkpoint_must_declare_gemma_source_checkpoint() {
        // 2.4.0 is the floor: at or above it, an absent `gemma_source_checkpoint` is itself an error
        // whenever the encoder has a version to compare against.
        let undeclared = meta(
            "/bundle/t.safetensors",
            &[
                (MODEL_VERSION_METADATA_KEY, "2.4.0"),
                (CONFIG_METADATA_KEY, r#"{"transformer":{}}"#),
            ],
        );
        let err = check_gemma_version(&undeclared, &gemma(None, Some("gemma4-12b-ltx-v1")))
            .expect_err("2.4+ must declare gemma_source_checkpoint");
        assert!(err.to_string().contains("gemma_source_checkpoint"));
        // With nothing on either side, upstream warns and skips rather than failing.
        assert_eq!(
            check_gemma_version(&undeclared, &gemma(None, None)).unwrap(),
            GemmaVersionCheck::SkippedNoDeclaredVersion
        );
    }

    // --- per-component config isolation ----------------------------------------------------------

    #[test]
    fn a_component_reads_only_its_own_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transformer.safetensors");
        write_safetensors(
            &path,
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"transformer":{"num_layers":48},"vae":null}"#,
                ),
            ],
        );
        let resolved = LtxResolvedComponent::open(LtxComponent::Transformer, &path).unwrap();
        assert_eq!(resolved.config().unwrap()["num_layers"], 48);
        assert!(resolved.optional_section("scheduler").is_none());
    }

    #[test]
    fn an_absent_vae_section_is_an_error_not_a_2_3_shaped_default() {
        // The acceptance case: a video-VAE slot pointed at a file whose `config.vae` is absent must
        // ERROR. If it silently fell back to the 2.3 block structure the decoder would be built with
        // the wrong stage ladder against 2.5 weights and only fail (or worse, not fail) at decode.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vae.safetensors");
        write_safetensors(
            &path,
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (CONFIG_METADATA_KEY, r#"{"transformer":{"num_layers":48}}"#),
            ],
        );
        let resolved = LtxResolvedComponent::open(LtxComponent::ConvVideoVae, &path).unwrap();
        let err = resolved
            .config()
            .expect_err("an absent config.vae must not default");
        let text = err.to_string();
        assert!(text.contains("conv_video_vae"), "{text}");
        assert!(text.contains("config.vae"), "{text}");
        assert!(text.contains("no other component's config"), "{text}");
        assert!(text.contains(&path.display().to_string()), "{text}");
    }

    #[test]
    fn the_2_5_transformers_null_vae_does_not_satisfy_the_vae_slot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        write_safetensors(
            &path,
            &[(
                CONFIG_METADATA_KEY,
                r#"{"transformer":{"num_layers":48},"vae":null}"#,
            )],
        );
        let resolved = LtxResolvedComponent::open(LtxComponent::ConvVideoVae, &path).unwrap();
        assert!(resolved.config().is_err());
    }

    #[test]
    fn the_duration_head_requires_both_of_its_sections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.safetensors");
        write_safetensors(
            &path,
            &[(
                CONFIG_METADATA_KEY,
                r#"{"duration_head":{"num_queries":1}}"#,
            )],
        );
        let resolved = LtxResolvedComponent::open(LtxComponent::DurationHead, &path).unwrap();
        // Its primary section is present…
        assert!(resolved.config().is_ok());
        // …but the transformer dims it projects from are not, and that is an error.
        let err = resolved
            .required_sections()
            .expect_err("the duration head needs the transformer dims too");
        assert!(err.to_string().contains("config.transformer"));
    }

    // --- bundle assembly -------------------------------------------------------------------------

    fn write_safetensors(path: &Path, metadata: &[(&str, &str)]) {
        // A minimal, valid safetensors file: an 8-byte little-endian header length, the header JSON
        // (one real tensor plus `__metadata__`), then that tensor's bytes.
        let meta_json: String = metadata
            .iter()
            .map(|(k, v)| format!("{}:{}", serde_json::json!(k), serde_json::json!(v)))
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"__metadata__":{{{meta_json}}},"w":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_split_bundle(root: &Path) {
        write_safetensors(
            &root.join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    GEMMA_SOURCE_CHECKPOINT_METADATA_KEY,
                    r#"{"ltx_version":"2.5.0","gemma_version":"gemma4-12b-ltx-v1"}"#,
                ),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"transformer":{"_class_name":"AVTransformer3DModel","num_layers":48},"vae":null,"audio_vae":null,"vocoder":null}"#,
                ),
            ],
        );
        // Ground truth (sc-18756): the packed text encoder declares NO `model_version` — only
        // `format` + `gemma_config`.
        write_safetensors(
            &root.join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            &[
                ("format", "pt"),
                (
                    GEMMA_CONFIG_METADATA_KEY,
                    r#"{"model_type":"gemma4_unified","gemma_version":"gemma4-12b-ltx-v1"}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":128}}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-video-vae-bf16.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"vae":{"_class_name":"CausalDiffusionVAE","decoder":{"head_dim":64}}}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("vae/ltx-2.5-audio-vae-bf16.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"audio_vae":{"model":{"params":{"ddconfig":{"ch":128}}}},"vocoder":{"vocoder":{"resblock":"AMP1"}}}"#,
                ),
            ],
        );
        write_safetensors(
            &root.join("model_patches/ltx-2.5-duration-head-bf16.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"transformer":{"cross_attention_dim":4096,"audio_cross_attention_dim":2048},"duration_head":{"num_queries":1}}"#,
                ),
            ],
        );
        // Ground truth (sc-18756): the latent upsamplers declare NO `model_version` either, so a
        // bundle's version must come from whichever component does declare one.
        write_safetensors(
            &root.join(
                "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            ),
            &[(
                CONFIG_METADATA_KEY,
                r#"{"_class_name":"LatentUpsampler","spatial_upsample":true,"temporal_upsample":false}"#,
            )],
        );
        write_safetensors(
            &root.join(
                "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
            ),
            &[(
                CONFIG_METADATA_KEY,
                r#"{"_class_name":"LatentUpsampler","spatial_upsample":false,"temporal_upsample":true}"#,
            )],
        );
    }

    #[test]
    fn discovery_resolves_every_component_of_a_split_bundle() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        let bundle = discover_split_bundle(dir.path()).unwrap();
        assert_eq!(bundle.model_version(), Some("2.5.0"));
        assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
        for component in LtxComponent::ALL {
            let resolved = bundle
                .require(*component)
                .unwrap_or_else(|e| panic!("{}: {e}", component.id()));
            assert_eq!(resolved.component(), *component);
        }
        // Each component read its OWN config, from its OWN file.
        assert_eq!(
            bundle.component_config(LtxComponent::Transformer).unwrap()["num_layers"],
            48
        );
        assert_eq!(
            bundle.component_config(LtxComponent::ConvVideoVae).unwrap()["_class_name"],
            CONV_VIDEO_VAE_CLASS
        );
        assert_eq!(
            bundle
                .component_config(LtxComponent::DiffusionVideoVae)
                .unwrap()["_class_name"],
            DIFFUSION_VIDEO_VAE_CLASS
        );
        // The audio VAE file owns BOTH of its sections.
        let audio = bundle.require(LtxComponent::AudioVae).unwrap();
        assert_eq!(audio.required_sections().unwrap().len(), 2);
    }

    #[test]
    fn discovery_ignores_a_lora_but_still_records_it_as_searched() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        write_safetensors(
            &dir.path()
                .join("loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
            &[("networkType", "lora")],
        );
        let bundle = discover_split_bundle(dir.path()).unwrap();
        assert!(bundle
            .searched()
            .iter()
            .any(|p| p.ends_with("ltx-2.5-22b-distilled-lora-450-bf16.safetensors")));
        assert_eq!(bundle.components().count(), LtxComponent::ALL.len());
    }

    #[test]
    fn a_missing_component_error_names_the_component_and_the_paths_searched() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        std::fs::remove_file(
            dir.path()
                .join("model_patches/ltx-2.5-duration-head-bf16.safetensors"),
        )
        .unwrap();
        let bundle = discover_split_bundle(dir.path()).unwrap();
        let err = bundle
            .require(LtxComponent::DurationHead)
            .expect_err("the duration head is gone");
        let text = err.to_string();
        assert!(text.contains("duration_head"), "{text}");
        assert!(text.contains("the duration head"), "{text}");
        assert!(text.contains("2.5.0"), "{text}");
        // Every path the resolver actually looked at is named.
        assert!(
            text.contains("ltx-2.5-video-vae-conv-bf16.safetensors"),
            "{text}"
        );
    }

    #[test]
    fn a_missing_component_does_not_downgrade_the_bundle_to_the_older_layout() {
        // Selection is keyed on `model_version`, NOT on which files happen to be present: a 2.5
        // bundle missing its audio VAE is still a 2.5 bundle.
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        std::fs::remove_file(dir.path().join("vae/ltx-2.5-audio-vae-bf16.safetensors")).unwrap();
        let bundle = discover_split_bundle(dir.path()).unwrap();
        assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
        assert!(bundle.require(LtxComponent::AudioVae).is_err());
    }

    #[test]
    fn two_files_claiming_one_component_is_an_error_not_a_guess() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        write_safetensors(
            &dir.path().join("vae/a-second-conv-vae.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"vae":{"_class_name":"CausalVideoAutoencoder"}}"#,
                ),
            ],
        );
        let err = discover_split_bundle(dir.path()).expect_err("ambiguous conv VAE");
        assert!(err.to_string().contains("ambiguous `conv_video_vae`"));
    }

    #[test]
    fn skipping_a_component_lets_an_explicit_choice_survive_an_ambiguous_scan() {
        // A caller that has already provisioned a component must not be blocked by the scan finding
        // two candidates for that same slot — its own choice is the answer.
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        let second = dir.path().join("vae/a-second-conv-vae.safetensors");
        write_safetensors(
            &second,
            &[
                (MODEL_VERSION_METADATA_KEY, "2.5.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":64}}"#,
                ),
            ],
        );
        let discovered =
            discover_split_bundle_skipping(dir.path(), &[LtxComponent::ConvVideoVae]).unwrap();
        // The slot is empty — the scan declined to guess…
        assert!(discovered.get(LtxComponent::ConvVideoVae).is_none());
        // …but both candidates are still on the searched list, and every other component resolved.
        assert!(discovered.searched().iter().any(|p| p == &second));
        assert!(discovered.require(LtxComponent::Transformer).is_ok());
        // Layering the caller's explicit choice on top produces a complete bundle.
        let mut builder = LtxBundleBuilder::new();
        for resolved in discovered.components() {
            builder = builder.with_component(resolved.component(), resolved.path());
        }
        let bundle = builder
            .with_component(LtxComponent::ConvVideoVae, &second)
            .build()
            .unwrap();
        assert_eq!(
            bundle.component_config(LtxComponent::ConvVideoVae).unwrap()["latent_channels"],
            64
        );
    }

    #[test]
    fn a_component_provisioned_into_the_wrong_slot_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        let err = LtxBundleBuilder::new()
            .with_component(
                LtxComponent::ConvVideoVae,
                dir.path().join("vae/ltx-2.5-audio-vae-bf16.safetensors"),
            )
            .build()
            .expect_err("the audio VAE is not the video VAE");
        let text = err.to_string();
        assert!(
            text.contains("provisioned as the `conv_video_vae` component"),
            "{text}"
        );
        assert!(text.contains("`audio_vae`"), "{text}");
    }

    #[test]
    fn a_mixed_version_bundle_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        write_safetensors(
            &dir.path().join("stale/old-conv-vae.safetensors"),
            &[
                (MODEL_VERSION_METADATA_KEY, "2.3.0"),
                (
                    CONFIG_METADATA_KEY,
                    r#"{"vae":{"_class_name":"CausalVideoAutoencoder"}}"#,
                ),
            ],
        );
        let err = LtxBundleBuilder::new()
            .with_component(
                LtxComponent::Transformer,
                dir.path()
                    .join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            )
            .with_component(
                LtxComponent::ConvVideoVae,
                dir.path().join("stale/old-conv-vae.safetensors"),
            )
            .build()
            .expect_err("2.3 and 2.5 components must not mix");
        assert!(err.to_string().contains("mixed model_version"));
    }

    #[test]
    fn a_missing_provisioned_path_names_the_component() {
        let err = LtxBundleBuilder::new()
            .with_component(LtxComponent::AudioVae, "/nope/audio.safetensors")
            .build()
            .expect_err("the path does not exist");
        let text = err.to_string();
        assert!(text.contains("audio_vae"), "{text}");
        assert!(text.contains("/nope/audio.safetensors"), "{text}");
    }

    #[test]
    fn the_bundle_gemma_assertion_reads_the_transformers_stamp() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        let bundle = discover_split_bundle(dir.path()).unwrap();
        // The bundle's own packed text encoder satisfies the assertion…
        let te = GemmaEncoderIdentity::from_single_file(
            dir.path()
                .join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        )
        .unwrap();
        assert_eq!(te.model_type.as_deref(), Some(GEMMA4_UNIFIED_MODEL_TYPE));
        assert!(matches!(
            bundle.check_gemma_version(&te).unwrap(),
            GemmaVersionCheck::Matched(_)
        ));
        // …and an LTX-2.3 Gemma-3 snapshot directory does not.
        let gemma3 = dir.path().join("gemma-3-12b-it");
        std::fs::create_dir_all(&gemma3).unwrap();
        std::fs::write(
            gemma3.join("config.json"),
            r#"{"model_type":"gemma3","text_config":{"hidden_size":3840}}"#,
        )
        .unwrap();
        let legacy = GemmaEncoderIdentity::from_dir(&gemma3).unwrap();
        let err = bundle
            .check_gemma_version(&legacy)
            .expect_err("a Gemma 3 snapshot must not load a 2.5 bundle");
        assert!(err.to_string().contains("Gemma version mismatch"));
    }

    // --- declared version / layout ---------------------------------------------------------------

    #[test]
    fn a_converted_2_3_tree_declares_its_version_through_the_manifest() {
        // The SceneWorks converter re-emits per-component files with no `__metadata__` at all, so
        // the manifest is the only declaration in the tree.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(SPLIT_MANIFEST_FILE),
            r#"{"format":"split","model_version":"2.3.0","quantized":true}"#,
        )
        .unwrap();
        for name in ["transformer", "connector", "vae_decoder"] {
            write_safetensors(&dir.path().join(format!("{name}.safetensors")), &[]);
        }
        assert_eq!(
            declared_model_version(dir.path()).unwrap().as_deref(),
            Some("2.3.0")
        );
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn a_2_5_bundle_declares_its_version_through_component_metadata() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        assert_eq!(
            declared_model_version(dir.path()).unwrap().as_deref(),
            Some("2.5.0")
        );
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::Split
        );
    }

    #[test]
    fn a_single_file_checkpoint_declares_its_own_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ltx-2.3-22b-distilled.safetensors");
        write_safetensors(
            &path,
            &[
                (MODEL_VERSION_METADATA_KEY, "2.3.0"),
                (CONFIG_METADATA_KEY, r#"{"transformer":{},"vae":{}}"#),
            ],
        );
        assert_eq!(
            declared_model_version(&path).unwrap().as_deref(),
            Some("2.3.0")
        );
        assert_eq!(
            declared_layout(&path).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn an_undeclared_tree_stays_on_the_oldest_layout() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(&dir.path().join("mystery.safetensors"), &[]);
        assert_eq!(declared_model_version(dir.path()).unwrap(), None);
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::AllInOne
        );
    }

    #[test]
    fn removing_a_component_does_not_change_the_declared_layout() {
        let dir = tempfile::tempdir().unwrap();
        write_split_bundle(dir.path());
        std::fs::remove_file(dir.path().join("vae/ltx-2.5-audio-vae-bf16.safetensors")).unwrap();
        std::fs::remove_file(
            dir.path()
                .join("model_patches/ltx-2.5-duration-head-bf16.safetensors"),
        )
        .unwrap();
        assert_eq!(
            declared_layout(dir.path()).unwrap(),
            LtxCheckpointLayout::Split
        );
    }

    // --- caption feature-extractor selection ------------------------------------------------------

    #[test]
    fn no_v2_keys_selects_v1() {
        // Upstream: "V1: V2 config keys absent → projection lives in transformer".
        let v1 = serde_json::json!({"num_layers": 48, "caption_channels": 3840});
        assert_eq!(
            caption_feature_version(&v1).unwrap(),
            CaptionFeatureVersion::V1
        );
    }

    #[test]
    fn all_four_v2_keys_with_the_expected_values_select_v2() {
        let mut v2 = serde_json::Map::new();
        for (name, expected) in CAPTION_V2_EXPECTED_CONFIG {
            v2.insert(name.to_string(), Value::Bool(expected));
        }
        assert_eq!(
            caption_feature_version(&Value::Object(v2)).unwrap(),
            CaptionFeatureVersion::V2
        );
    }

    #[test]
    fn the_shipped_2_3_two_key_shape_selects_v2_through_the_carve_out() {
        // MEASURED: `SceneWorks/ltx-2.3-mlx` q4's `embedded_config.json` declares exactly these two
        // keys (both false), omits `caption_proj_before_connector` / `caption_proj_input_norm`, and
        // sets `text_encoder_norm_type: "per_token_rms"`. Upstream's strict rule calls that partial
        // V2 and raises — which would refuse a checkpoint that has always loaded.
        let legacy = serde_json::json!({
            "caption_projection_first_linear": false,
            "caption_projection_second_linear": false,
            "text_encoder_norm_type": "per_token_rms",
        });
        assert_eq!(
            caption_feature_version(&legacy).unwrap(),
            CaptionFeatureVersion::V2
        );
    }

    #[test]
    fn the_carve_out_is_narrow() {
        // Only the exact pair, only both-false. One key alone is drift.
        let one = serde_json::json!({"caption_projection_first_linear": false});
        let err = caption_feature_version(&one).expect_err("one key is not the legacy shape");
        assert!(
            err.to_string().contains("partial caption-projection"),
            "{err}"
        );

        // The right pair with a WRONG value is not the legacy shape either.
        let wrong_value = serde_json::json!({
            "caption_projection_first_linear": true,
            "caption_projection_second_linear": false,
        });
        assert!(caption_feature_version(&wrong_value).is_err());

        // Three of four is drift, and the message names what is missing.
        let three = serde_json::json!({
            "caption_proj_before_connector": true,
            "caption_projection_first_linear": false,
            "caption_projection_second_linear": false,
        });
        let err = caption_feature_version(&three).expect_err("three keys is partial");
        let text = err.to_string();
        assert!(text.contains("caption_proj_input_norm"), "{text}");
    }

    #[test]
    fn a_full_v2_block_with_a_drifted_value_is_a_hard_error() {
        let mut drifted = serde_json::Map::new();
        for (name, expected) in CAPTION_V2_EXPECTED_CONFIG {
            drifted.insert(name.to_string(), Value::Bool(expected));
        }
        drifted.insert(
            "caption_proj_before_connector".to_string(),
            Value::Bool(false),
        );
        let err = caption_feature_version(&Value::Object(drifted))
            .expect_err("a drifted value must not silently select V2");
        let text = err.to_string();
        assert!(
            text.contains("caption_proj_before_connector=false"),
            "{text}"
        );
        assert!(text.contains("expected true"), "{text}");
    }

    #[test]
    fn the_real_2_5_transformer_selects_v2_from_its_own_config() {
        // Ground truth: the shipped 2.5 transformers declare all four keys with the expected values.
        for name in [
            "ltx-2.5-22b-dev-transformer-bf16",
            "ltx-2.5-22b-distilled-transformer-bf16",
            "ltx-2.5-22b-distilled-transformer-nvfp4",
        ] {
            let meta = captured_metadata(&format!("diffusion_models/{name}.safetensors.json"));
            let transformer = meta.section("transformer").expect(name);
            assert_eq!(
                caption_feature_version(transformer).unwrap(),
                CaptionFeatureVersion::V2,
                "{name}"
            );
        }
    }

    // --- real captured LTX-2.5 headers -----------------------------------------------------------

    /// Load one of sc-18756's captured `__metadata__` dumps and rebuild the on-disk
    /// `__metadata__` map from it.
    ///
    /// The dumps store each value already JSON-**decoded** for readability; the real safetensors
    /// block stores every value as a string, so a decoded object is re-encoded and a decoded string
    /// is passed through verbatim (re-encoding `"2.5.0"` would yield `"\"2.5.0\""`, which is not what
    /// the file holds).
    ///
    /// This test module deliberately reaches into `docs/reference/` rather than copying the headers:
    /// binding the classifier to the captured evidence is the point, and a divergence between the
    /// two should fail loudly rather than drift behind a stale copy.
    fn captured_metadata(relative: &str) -> LtxCheckpointMetadata {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/reference/sc-18756-headers")
            .join(relative);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read captured header {}: {e}", path.display()));
        let dump: Value = serde_json::from_str(&text).expect("captured header parses");
        let block = dump
            .get("metadata")
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{relative} has no `metadata` block"));
        let raw: BTreeMap<String, String> = block
            .iter()
            .map(|(key, value)| {
                let encoded = match value {
                    Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).expect("re-encode"),
                };
                (key.clone(), encoded)
            })
            .collect();
        LtxCheckpointMetadata::from_raw(Path::new(relative), raw).expect("metadata parses")
    }

    #[test]
    fn the_real_2_5_transformer_headers_classify_and_stamp_as_documented() {
        // All five shipped transformer variants (dev/distilled × bf16/int8-convrot/nvfp4).
        for name in [
            "ltx-2.5-22b-dev-transformer-bf16",
            "ltx-2.5-22b-dev-transformer-comfy-int8-convrot",
            "ltx-2.5-22b-distilled-transformer-bf16",
            "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot",
            "ltx-2.5-22b-distilled-transformer-nvfp4",
        ] {
            let meta = captured_metadata(&format!("diffusion_models/{name}.safetensors.json"));
            assert_eq!(meta.model_version(), Some("2.5.0"), "{name}");
            assert_eq!(meta.layout(), LtxCheckpointLayout::Split, "{name}");
            assert_eq!(meta.classify(), Some(LtxComponent::Transformer), "{name}");
            // Its own section is present…
            assert_eq!(
                meta.section("transformer")
                    .and_then(|t| t.get("_class_name"))
                    .and_then(Value::as_str),
                Some(TRANSFORMER_CLASS),
                "{name}"
            );
            assert!(meta.section("scheduler").is_some(), "{name}");
            // …and the sections it no longer owns are simply NOT THERE. The shipped files omit the
            // keys outright rather than carrying them as `null`; `section` treats both the same, so
            // neither spelling can satisfy another component's slot.
            for key in ["vae", "audio_vae", "vocoder", "duration_head"] {
                assert!(meta.section(key).is_none(), "{name}: config.{key}");
            }
            // Every transformer stamps the Gemma assertion.
            let gsc = meta.gemma_source_checkpoint().expect(name);
            assert_eq!(gsc.ltx_version.as_deref(), Some("2.5.0"), "{name}");
            assert_eq!(
                gsc.gemma_version.as_deref(),
                Some("gemma4-12b-ltx-v1"),
                "{name}"
            );
        }
    }

    #[test]
    fn the_real_2_5_component_headers_classify_by_their_own_config() {
        let conv = captured_metadata("vae/ltx-2.5-video-vae-conv-bf16.safetensors.json");
        assert_eq!(conv.classify(), Some(LtxComponent::ConvVideoVae));
        assert_eq!(
            conv.section("vae").unwrap()["_class_name"],
            CONV_VIDEO_VAE_CLASS
        );

        // Named `...-video-vae-bf16` with no "diff" anywhere in the file name — only its config says
        // it is the diffusion decoder. This is why classification never keys on names.
        let diff = captured_metadata("vae/ltx-2.5-video-vae-bf16.safetensors.json");
        assert_eq!(diff.classify(), Some(LtxComponent::DiffusionVideoVae));
        assert_eq!(
            diff.section("vae").unwrap()["_class_name"],
            DIFFUSION_VIDEO_VAE_CLASS
        );

        // One file, both sections — the vocoder has no component of its own.
        let audio = captured_metadata("vae/ltx-2.5-audio-vae-bf16.safetensors.json");
        assert_eq!(audio.classify(), Some(LtxComponent::AudioVae));
        assert!(audio.section("audio_vae").is_some());
        assert!(audio.section("vocoder").is_some());
        assert_eq!(
            audio.section("audio_vae").unwrap()["model"]["params"]["ddconfig"]["z_channels"],
            8
        );

        // The duration head also carries `config.transformer` (the dims it projects from), so it
        // must be tested before the plain transformer — it is, and it classifies correctly.
        let head = captured_metadata("model_patches/ltx-2.5-duration-head-bf16.safetensors.json");
        assert_eq!(head.classify(), Some(LtxComponent::DurationHead));
        assert!(head.section("duration_head").is_some());
        assert!(head.section("transformer").is_some());
    }

    #[test]
    fn the_real_upsamplers_and_text_encoders_declare_no_model_version() {
        // Ground truth from sc-18756: the latent upsamplers and the packed text encoders carry NO
        // `model_version` at all. Resolution must not require one per component — the bundle's
        // version comes from whichever component declares it — and classification must still work.
        for (relative, expected) in [
            (
                "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors.json",
                LtxComponent::SpatialUpsampler,
            ),
            (
                "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors.json",
                LtxComponent::TemporalUpsampler,
            ),
        ] {
            let meta = captured_metadata(relative);
            assert_eq!(meta.model_version(), None, "{relative}");
            assert_eq!(meta.classify(), Some(expected), "{relative}");
            // The upsampler config is BARE — no wrapper section.
            assert_eq!(
                meta.config().unwrap()["_class_name"],
                LATENT_UPSAMPLER_CLASS,
                "{relative}"
            );
            assert!(meta.section("vae").is_none(), "{relative}");
        }

        for name in [
            "gemma4-12b-with-proj-ltx-2.5-bf16",
            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot",
        ] {
            let meta = captured_metadata(&format!("text_encoders/{name}.safetensors.json"));
            assert_eq!(meta.model_version(), None, "{name}");
            assert_eq!(meta.classify(), Some(LtxComponent::TextEncoder), "{name}");
            let identity = GemmaEncoderIdentity::from_config_value(
                name,
                meta.gemma_config().expect("packed TE config"),
            );
            assert_eq!(
                identity.model_type.as_deref(),
                Some(GEMMA4_UNIFIED_MODEL_TYPE),
                "{name}"
            );
            assert_eq!(
                identity.gemma_version.as_deref(),
                Some("gemma4-12b-ltx-v1"),
                "{name}"
            );
        }
    }

    #[test]
    fn the_real_transformer_and_text_encoder_satisfy_the_gemma_assertion() {
        let transformer = captured_metadata(
            "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors.json",
        );
        let te =
            captured_metadata("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors.json");
        let identity = GemmaEncoderIdentity::from_config_value(
            "gemma4-12b-with-proj-ltx-2.5-bf16",
            te.gemma_config().unwrap(),
        );
        assert_eq!(
            check_gemma_version(&transformer, &identity).unwrap(),
            GemmaVersionCheck::Matched("gemma4-12b-ltx-v1".to_string())
        );
        // The LTX-2.3 encoder against the same real transformer is the acceptance failure case.
        let gemma3 = GemmaEncoderIdentity {
            source: PathBuf::from("/models/gemma-3-12b-it"),
            model_type: Some(GEMMA3_MODEL_TYPE.to_string()),
            gemma_version: None,
        };
        let err = check_gemma_version(&transformer, &gemma3).expect_err("mismatch");
        assert!(err.to_string().contains("Gemma version mismatch"), "{err}");
    }

    #[test]
    fn the_real_distilled_lora_is_not_a_component() {
        // It ships inside the same bundle and stamps `model_version`, but carries no `config`, so a
        // directory scan must skip it rather than mistake it for a component.
        let lora = captured_metadata("loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors.json");
        assert_eq!(lora.model_version(), Some("2.5.0"));
        assert_eq!(lora.classify(), None);
    }

    #[test]
    fn component_ids_round_trip() {
        for component in LtxComponent::ALL {
            assert_eq!(LtxComponent::from_id(component.id()), Some(*component));
        }
        assert_eq!(LtxComponent::from_id("nope"), None);
    }
}
