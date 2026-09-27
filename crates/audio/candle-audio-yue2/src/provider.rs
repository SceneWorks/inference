//! The registered YuE2 provider (sc-22994): the [`gen_core::Generator`] adapter over
//! [`Yue2Engine`], its descriptor, the `LoadSpec` gate, the `GenerationRequest` mapping and the
//! registration the audio catalog composes.
//!
//! The provider id is [`PROVIDER_ID`] (`yue2`) — distinct from YuE1's six `yue_*` generators and
//! never an alias of them (epic E1). A registry `load("yue2", spec)` reaches
//! [`Yue2Engine::load`]; `generate` runs the native pipeline. No Python process is involved.
//!
//! # `LoadSpec`
//!
//! | field | YuE2 |
//! |---|---|
//! | `weights` | `Dir`: the `m-a-p/YuE2-3B` snapshot (holds `qwen.tiktoken`), a derived `q8` / `q4` tier snapshot of it ([`crate::tier`]), **or** a closure saved by [`crate::closure::save_closure`] (then no components) |
//! | `components["vae"]` | `Dir`: the `m-a-p/YuE2-Vae` snapshot (the standard decoder) |
//! | `components["vae_legacy"]` | `Dir`: the `m-a-p/YuE2-Vae-legacy` snapshot (the legacy decoder) |
//! | | a plain snapshot stages **at least one** of the two decoders ([`DECODER_COMPONENTS`]); every request's decoder is checked before any compute |
//! | `precision` | `Fp32` ⇒ F32; the default ⇒ BF16 on an accelerator, F32 on the CPU (no BF16 matmul) |
//! | `quantize` | `None` loads the staged tier (the original is `bf16`); `Q8` / `Q4` assert that tier — the `weights` directory must be that derived tier snapshot, anything else is refused. Exactly [`SUPPORTED_QUANTS`] (advertised as `supported_quants`, the audio lane's convention: the unquantized `bf16` load is `None`) is accepted; `Nvfp4` is refused ([`crate::precision`]) |
//! | `offload_policy` | `Sequential` ⇒ the AR-only weights move to host memory while the acoustic stage runs (upstream `offload_ar`) for every request that does not choose otherwise; `Resident` ⇒ they stay |
//! | `load_shape` | only the eager materialization YuE2 loads with; `DeferredMaterialization` is refused |
//!
//! The experimental FP8 AR mode ([`crate::fp8`]) is a native engine option
//! ([`crate::engine::ModelPrecision`]); `LoadSpec` has no FP8 value, so the registered provider
//! cannot request it (recorded owner decision `fp8_not_on_the_load_spec` in
//! [`crate::precision::OWNER_DECISIONS`]).
//!
//! Every snapshot is local and verified against its pins when loaded; a missing one is an explicit
//! cache-miss error, never a download.
//!
//! # Request mapping
//!
//! | request field | YuE2 control |
//! |---|---|
//! | `prompt` | `style` |
//! | `audio.lyrics` | `lyrics` (required unless restoring a plan or decoding cached latents) |
//! | `seed` | `seed` |
//! | `guidance` | `cfg_scale` (read as the shortest decimal of the `f32`, so `1.01` is `1.01`) |
//! | `steps` | midpoint ODE steps (`ode_steps`) |
//! | `audio.song.planning` | `cot` (`full` / `melody` / `off`) |
//! | `audio.song.score` | external ABC; it must parse in the native dialect ([`crate::cover::abc::parse`]), anything else is refused with the parse error |
//! | `audio.song.score_sampling` / `semantic_sampling` | the two AR phases' sampling overrides |
//! | `audio.song.plan` | restore an exact saved plan (its request must agree with any request field set) |
//! | `audio.song.cached_latents` | decode a completed run's verified latents (no generation fields) |
//! | `audio.song.decoder` | `Standard` (default) / `Legacy` |
//! | `audio.song.plan_only` | plan and publish the exact plan as a run of kind `plan` ([`Yue2Engine::plan_to`]); needs `audio.artifacts`; renders no audio |
//! | `audio.song.cover` | a zero-shot cover of a reviewed score ([`crate::cover::prepare_cover`]): `mode` melody (chord symbols removed, the melodies proved unchanged) or full, `keep`, source or translated lyrics. A cover refusal is an error; its warnings are reported; with artifacts, [`COVER_JSON`] (the [`CoverReport`]) is published inside the run under its digests |
//! | `audio.artifacts` | publish the run directory transactionally; `resume` reuses matching work |
//! | `memory` | the per-request memory controls below |
//!
//! Every other audio field (`target_duration`, `bpm`, `musical_key`, `voice`, `language`,
//! `repetition_penalty` — ambiguous between the two phases —, segment and limiter controls) is
//! refused rather than dropped. Recording transcription is not reachable here: a cover starts from
//! a reviewed score.
//!
//! # Memory controls (`GenerationRequest::memory`, sc-22988)
//!
//! None changes a result: each is recorded in the run's effective configuration and bound into no
//! stage identity. Absent ⇒ the engine's own (the `LoadSpec` above). Present
//! ([`memory_options`]):
//!
//! | field | YuE2 |
//! |---|---|
//! | `stage_residency` | `offload_ar`: the AR-only weights on the host while the acoustic stage runs; `false` keeps them resident, whatever the load chose |
//! | `chunk_attention` | bounded acoustic attention; `false` ⇒ the historical 256-row query tiles |
//! | `attention_chunk_size` | with `chunk_attention` only: at most this many score elements (`heads × rows × keys`) per attention call ([`QueryTile::ScoreElements`]); it must hold a row at the model's full context, `heads × positions` ([`Yue2Engine::attention_bounds`]) |
//! | `tile_vae_decode` | the halo/crop tiled decode (the production path) |
//! | `decode_tile_edge` | with `tile_vae_decode` only: the tile core in **latent frames**, `1..=1024` (one frame is 1920 samples, 40 ms). [`DecodeOptions::for_memory_budget_gib`]`(b)`[`.core_frames()`](DecodeOptions::core_frames) is the core for a decode memory budget of `b` GiB |
//! | anything else | `stream_transformer_blocks`, the transformer window, `decode_overlap` and calibration fault injection are refused as `Unsupported`; the execution domains are refused by the shared floor |
//!
//! # Results and progress
//!
//! [`Generator::generate_with_report`] is the primary entry point: the audio (none for
//! `plan_only`), the published record ([`gen_core::ArtifactRecord`]) and the warnings — a
//! truncated ABC or semantic phase (`abc_truncated` / `semantic_truncated`, [`TRUNCATION_CODES`])
//! and every cover warning under its own code. [`Generator::generate`] returns the audio only and
//! refuses `plan_only`.
//!
//! `Progress::Step` counts each sampled token of the plan and semantic stages against that
//! phase's `max_tokens`, then each midpoint step of the acoustic stage over all chunks (every stage
//! counts from 1 against its own total); `Progress::Decoding` marks the decoder's start.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device};
use candle_audio::gen_core::{
    self, ArtifactRecord, AudioParams, AudioTrack, Capabilities, GenerationMemory,
    GenerationOutput, GenerationReport, GenerationRequest, GenerationWarning, Generator, LoadShape,
    LoadSpec, MemoryStrategy, Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant,
    SongCover, SongCoverMode, SongCoverVoice, SongDecoder, SongPlanning, TokenSampling,
    WeightsSource,
};
use serde_json::{json, Value};

use crate::closure::{is_saved_closure, load_closure};
use crate::cover::abc::KeepVoice;
use crate::cover::{prepare_cover, CoverLyrics, CoverMode, CoverReport, CoverSpec};
use crate::decode::DecodeOptions;
use crate::engine::{
    EngineHooks, EngineObserver, EngineOptions, ModelPrecision, SongSettings, Stage, StageEvent,
    Yue2Engine,
};
use crate::inventory::{ComponentId, VaeVariant};
use crate::nar::{NarOptions, QueryTile};
use crate::plan::SymbolicPlan;
use crate::precision::Tier;
use crate::protocol::{
    CotMode, GenerationConfig, SamplingOverrides, SongRequest, SongRequestSpec, DEFAULT_ID,
    DEFAULT_SEED,
};
use crate::run::{verify_run, RunAttachment, RunOutcome, RunOutput, SongInput};
use crate::snapshot::SnapshotDirs;
use crate::vae::SAMPLE_RATE;

/// The registered provider id.
pub const PROVIDER_ID: &str = "yue2";
/// The model family.
pub const FAMILY: &str = "yue2";
/// The [`LoadSpec::components`] id of the standard decoder snapshot (`m-a-p/YuE2-Vae`).
pub const VAE_COMPONENT_ID: &str = "vae";
/// The [`LoadSpec::components`] id of the legacy decoder snapshot (`m-a-p/YuE2-Vae-legacy`).
pub const VAE_LEGACY_COMPONENT_ID: &str = "vae_legacy";
/// Components a plain-snapshot load requires unconditionally: none — but at least one of
/// [`DECODER_COMPONENTS`] must be staged.
pub const REQUIRED_COMPONENTS: &[&str] = &[];
/// The decoder components. A plain-snapshot load stages at least one; a request decodes only with
/// a staged one.
pub const DECODER_COMPONENTS: &[&str] = &[VAE_COMPONENT_ID, VAE_LEGACY_COMPONENT_ID];
/// Every component id the loader recognizes.
pub const KNOWN_COMPONENTS: &[&str] = &[VAE_COMPONENT_ID, VAE_LEGACY_COMPONENT_ID];
/// Output channels (stereo).
pub const CHANNELS: u16 = 2;
/// The per-request memory rungs YuE2 honours ([`Capabilities::request_memory_strategies`]; see
/// the [module docs](self#memory-controls-generationrequestmemory-sc-22988)).
pub const REQUEST_MEMORY_STRATEGIES: &[MemoryStrategy] = &[
    MemoryStrategy::StagedResidency,
    MemoryStrategy::BoundedDecode,
    MemoryStrategy::BoundedAttention,
];
/// The quantized tiers a `LoadSpec` may assert ([`Capabilities::supported_quants`]; the `bf16`
/// original is the unquantized load, `quantize: None`, as for YuE1). Exactly these are accepted.
pub const SUPPORTED_QUANTS: &[Quant] = &[Quant::Q4, Quant::Q8];
/// The record a cover publishes inside its run directory (under the run's digests).
pub const COVER_JSON: &str = "cover.json";
/// Schema of [`COVER_JSON`].
pub const COVER_SCHEMA: &str = "yue2-cover-v1";
/// The warning codes of a truncated ABC and semantic phase.
pub const TRUNCATION_CODES: [&str; 2] = ["abc_truncated", "semantic_truncated"];

/// The weights-free descriptor.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: REQUIRED_COMPONENTS,
        id: PROVIDER_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            max_count: 1,
            supports_guidance: true,
            audio_sample_rates: vec![SAMPLE_RATE],
            supports_symbolic_song: true,
            supports_audio_artifacts: true,
            supports_song_plan_only: true,
            supports_song_cover: true,
            supports_sequential_offload: true,
            request_memory_strategies: REQUEST_MEMORY_STRATEGIES,
            supported_quants: SUPPORTED_QUANTS,
            ..Default::default()
        },
    }
}

fn refuse(field: &str, why: &str) -> gen_core::Error {
    gen_core::Error::Unsupported(format!(
        "{PROVIDER_ID}: `{field}` is not a YuE2 control — {why}"
    ))
}

fn invalid(what: impl std::fmt::Display) -> gen_core::Error {
    gen_core::Error::Msg(format!("{PROVIDER_ID}: {what}"))
}

/// An `f32` request value as the `f64` its shortest decimal names (`1.01f32` ⇒ `1.01`).
fn decimal_f64(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(v as f64)
}

fn overrides(s: &TokenSampling) -> SamplingOverrides {
    SamplingOverrides {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k.map(i64::from),
        repetition_penalty: s.repetition_penalty,
        penalty_window: s.penalty_window.map(i64::from),
        min_tokens: s.min_tokens.map(i64::from),
        max_tokens: s.max_tokens.map(i64::from),
    }
}

fn decoder_of(d: Option<SongDecoder>) -> VaeVariant {
    match d {
        None | Some(SongDecoder::Standard) => VaeVariant::Standard,
        Some(SongDecoder::Legacy) => VaeVariant::Legacy,
    }
}

/// The first set field of `stray`, refused as not combinable with `with`.
fn refuse_stray(stray: &[(&str, bool)], with: &str, why: &str) -> gen_core::Result<()> {
    match stray.iter().find(|(_, set)| *set) {
        Some((field, _)) => Err(invalid(format!(
            "{field} cannot be combined with {with}: {why}"
        ))),
        None => Ok(()),
    }
}

/// What a request asks the engine to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Job {
    /// Generate from a request (a plain one, or a cover's).
    Generate(SongRequest),
    /// Generate from an exact saved plan.
    FromPlan {
        /// The plan directory.
        dir: PathBuf,
        /// The identity it must have, when the caller kept one.
        identity: Option<String>,
    },
    /// Decode a completed run's cached latents.
    DecodeCached(PathBuf),
}

/// A mapped request.
#[derive(Clone, Debug, PartialEq)]
pub struct MappedRequest {
    /// What to do.
    pub job: Job,
    /// Sampling, ODE steps, decoder and (once [`memory_options`] has run) memory controls.
    pub settings: SongSettings,
    /// Where to publish, when the request asks for artifacts (a cover's record attached).
    pub output: Option<RunOutput>,
    /// Plan and publish only ([`Yue2Engine::plan_to`]); always with a [`Job::Generate`] and an
    /// output.
    pub plan_only: bool,
    /// The cover's checks, when the request is a cover.
    pub cover: Option<CoverReport>,
}

/// Map a [`GenerationRequest`] onto the engine (see the [module docs](self)); `base` is the
/// engine's generation configuration the overrides apply to. The memory controls are mapped
/// separately ([`memory_options`]); `settings.options` is `None` here.
pub fn map_request(
    req: &GenerationRequest,
    base: &GenerationConfig,
) -> gen_core::Result<MappedRequest> {
    let audio = req.audio.clone().unwrap_or_default();
    let AudioParams {
        voice,
        language,
        target_duration,
        sample_rate: _,
        bpm,
        musical_key,
        lyrics,
        script,
        segments,
        max_new_tokens_per_segment,
        repetition_penalty,
        reference_region,
        output_limiter,
        song,
        artifacts,
    } = audio;
    let why_style = "describe it in the style (prompt)";
    for (field, set) in [
        ("audio.voice", voice.is_some()),
        ("audio.language", language.is_some()),
        ("audio.bpm", bpm.is_some()),
        ("audio.musical_key", musical_key.is_some()),
    ] {
        if set {
            return Err(refuse(field, why_style));
        }
    }
    if target_duration.is_some() {
        return Err(refuse(
            "audio.target_duration",
            "the song length follows the lyrics and the semantic token budget",
        ));
    }
    if repetition_penalty.is_some() {
        return Err(refuse(
            "audio.repetition_penalty",
            "set it per phase in audio.song.score_sampling / semantic_sampling",
        ));
    }
    if script.is_some()
        || segments.is_some()
        || max_new_tokens_per_segment.is_some()
        || reference_region.is_some()
        || output_limiter.is_some()
    {
        return Err(refuse(
            "audio.script/segments/max_new_tokens_per_segment/reference_region/output_limiter",
            "YuE2 renders one song in one pass without a reference clip or limiter",
        ));
    }
    if !req.conditioning.is_empty() {
        return Err(refuse(
            "conditioning",
            "YuE2 takes no reference audio (a cover plans from a reviewed score, audio.song.cover)",
        ));
    }
    let song = song.unwrap_or_default();
    let plan_only = song.plan_only;
    let output = artifacts.map(|a| RunOutput {
        dir: a.dir,
        resume: a.resume,
        attachments: Vec::new(),
    });
    if plan_only && output.is_none() {
        return Err(invalid(
            "audio.song.plan_only requires audio.artifacts (the plan is published there; nothing \
             else is returned)",
        ));
    }
    let decoder = decoder_of(song.decoder);

    // Decoding cached latents takes nothing that would regenerate them.
    if let Some(source) = song.cached_latents {
        refuse_stray(
            &[
                ("prompt", !req.prompt.is_empty()),
                ("audio.lyrics", lyrics.is_some()),
                ("seed", req.seed.is_some()),
                ("guidance", req.guidance.is_some()),
                ("steps", req.steps.is_some()),
                ("audio.song.planning", song.planning.is_some()),
                ("audio.song.score", song.score.is_some()),
                ("audio.song.plan", song.plan.is_some()),
                ("audio.song.score_sampling", song.score_sampling.is_some()),
                (
                    "audio.song.semantic_sampling",
                    song.semantic_sampling.is_some(),
                ),
                ("audio.song.plan_only", plan_only),
                ("audio.song.cover", song.cover.is_some()),
            ],
            "audio.song.cached_latents",
            "a cached decode re-renders the source run's latents and generates nothing",
        )?;
        return Ok(MappedRequest {
            job: Job::DecodeCached(source),
            settings: SongSettings {
                generation: base.clone(),
                decoder,
                options: None,
            },
            output,
            plan_only: false,
            cover: None,
        });
    }

    let abc = match &song.score_sampling {
        Some(s) => base.abc().with_overrides(&overrides(s)).map_err(invalid)?,
        None => *base.abc(),
    };
    let semantic = match &song.semantic_sampling {
        Some(s) => base
            .semantic()
            .with_overrides(&overrides(s))
            .map_err(invalid)?,
        None => *base.semantic(),
    };
    let steps = req.steps.map_or(base.ode_steps() as i64, i64::from);
    let generation = GenerationConfig::new(abc, semantic, steps).map_err(invalid)?;
    let settings = SongSettings {
        generation,
        decoder,
        options: None,
    };

    if let Some(plan) = song.plan {
        refuse_stray(
            &[
                ("seed", req.seed.is_some()),
                ("guidance", req.guidance.is_some()),
                ("audio.song.planning", song.planning.is_some()),
                ("audio.song.score", song.score.is_some()),
                ("audio.song.score_sampling", song.score_sampling.is_some()),
                ("audio.song.plan_only", plan_only),
                ("audio.song.cover", song.cover.is_some()),
            ],
            "audio.song.plan",
            "the restored plan fixes it (an edited plan is a new request)",
        )?;
        // `prompt` / `audio.lyrics`, when set, are checked against the restored plan's request
        // in `generate` (the plan is read there, not while mapping).
        return Ok(MappedRequest {
            job: Job::FromPlan {
                dir: plan.dir,
                identity: plan.identity,
            },
            settings,
            output,
            plan_only: false,
            cover: None,
        });
    }

    let Some(lyrics) = lyrics else {
        return Err(invalid("audio.lyrics is required"));
    };
    if let Some(cover) = song.cover {
        refuse_stray(
            &[
                ("audio.song.planning", song.planning.is_some()),
                ("audio.song.score", song.score.is_some()),
            ],
            "audio.song.cover",
            "the cover plans from its own reviewed score in its own mode",
        )?;
        return map_cover(req, cover, lyrics, settings, output, plan_only);
    }
    if let Some(score) = &song.score {
        // The native dialect is the only ABC YuE2 plans from reliably; an out-of-dialect score is
        // refused with the parser's reason instead of being planned from.
        crate::cover::abc::parse(score).map_err(|e| {
            invalid(format!(
                "audio.song.score is not in the native two-voice ABC dialect: {e}"
            ))
        })?;
    }
    let mut spec = SongRequestSpec::new(req.prompt.clone(), lyrics);
    spec.cot = match song.planning {
        None | Some(SongPlanning::Full) => CotMode::Full,
        Some(SongPlanning::Melody) => CotMode::Melody,
        Some(SongPlanning::Off) => CotMode::Off,
    };
    spec.seed = req.seed.unwrap_or(DEFAULT_SEED);
    spec.abc = song.score;
    spec.cfg_scale = req.guidance.map(decimal_f64);
    spec.id = DEFAULT_ID.to_string();
    let request = SongRequest::new(spec).map_err(invalid)?;
    Ok(MappedRequest {
        job: Job::Generate(request),
        settings,
        output,
        plan_only,
        cover: None,
    })
}

/// A cover request: [`prepare_cover`] (the native-dialect parse, chord removal with its invariant
/// proof for a melody cover, the harmony and melody checks, the lyric alignment) builds the
/// request; its report is published with the run.
fn map_cover(
    req: &GenerationRequest,
    cover: SongCover,
    lyrics: String,
    settings: SongSettings,
    output: Option<RunOutput>,
    plan_only: bool,
) -> gen_core::Result<MappedRequest> {
    let SongCover {
        mode,
        score,
        keep,
        translated_from,
    } = cover;
    let (mode, mode_name) = match mode {
        SongCoverMode::Melody => (CoverMode::Melody, "melody"),
        SongCoverMode::Full => (CoverMode::Full, "full"),
    };
    let (keep, keep_name) = match keep {
        None | Some(SongCoverVoice::Both) => (KeepVoice::Both, "both"),
        Some(SongCoverVoice::Vocal) => (KeepVoice::Vocal, "vocal"),
        Some(SongCoverVoice::Instrumental) => (KeepVoice::Ins, "instrumental"),
    };
    let translated = translated_from.is_some();
    let spec = CoverSpec {
        mode,
        score,
        keep,
        style: req.prompt.clone(),
        lyrics: CoverLyrics {
            lyrics,
            translated_from,
        },
        seed: req.seed.unwrap_or(DEFAULT_SEED),
        cfg_scale: req.guidance.map(decimal_f64),
        id: DEFAULT_ID.to_string(),
    };
    let prepared = prepare_cover(&spec).map_err(|e| invalid(format!("audio.song.cover: {e}")))?;
    let record = json!({
        "schema": COVER_SCHEMA,
        "mode": mode_name,
        "keep": keep_name,
        "translated": translated,
        "source_score_sha256": crate::durable::sha256_hex(spec.score.as_bytes()),
        "report": prepared.report.to_json(),
    });
    Ok(MappedRequest {
        job: Job::Generate(prepared.request),
        settings,
        output: output.map(|o| o.with_attachment(RunAttachment::json(COVER_JSON, &record))),
        plan_only,
        cover: Some(prepared.report),
    })
}

/// A request's memory controls over the engine's `base` (see the
/// [module docs](self#memory-controls-generationrequestmemory-sc-22988)): `None` without a
/// `memory` block; every field YuE2 does not honour is refused, never ignored. `attention` is
/// [`Yue2Engine::attention_bounds`].
pub fn memory_options(
    memory: Option<&GenerationMemory>,
    base: &EngineOptions,
    attention: (usize, usize),
) -> gen_core::Result<Option<EngineOptions>> {
    let Some(memory) = memory else {
        return Ok(None);
    };
    // Destructured without `..`: a new `GenerationMemory` field fails to compile here until it is
    // classified as honoured or refused.
    let GenerationMemory {
        stage_residency,
        tile_vae_decode,
        chunk_attention,
        stream_transformer_blocks,
        decode_tile_edge,
        decode_overlap,
        attention_chunk_size,
        transformer_window_size,
        transformer_window_component,
        graph_eval_cadence,
        ffn_chunk,
        cfg_batching,
        calibration_error_phase,
        calibration_fault_harness_authorized,
    } = *memory;
    for (field, set) in [
        (
            "memory.stream_transformer_blocks",
            stream_transformer_blocks,
        ),
        (
            "memory.transformer_window_size",
            transformer_window_size.is_some(),
        ),
        (
            "memory.transformer_window_component",
            transformer_window_component.is_some(),
        ),
        ("memory.decode_overlap", decode_overlap.is_some()),
        ("memory.graph_eval_cadence", graph_eval_cadence.is_some()),
        ("memory.ffn_chunk", ffn_chunk.is_some()),
        ("memory.cfg_batching", cfg_batching.is_some()),
        (
            "memory.calibration_error_phase",
            calibration_error_phase.is_some() || calibration_fault_harness_authorized,
        ),
    ] {
        if set {
            return Err(refuse(
                field,
                "YuE2 honours stage_residency, chunk_attention / attention_chunk_size and \
                 tile_vae_decode / decode_tile_edge only",
            ));
        }
    }
    if attention_chunk_size.is_some() && !chunk_attention {
        return Err(invalid(
            "memory.attention_chunk_size is read only with memory.chunk_attention",
        ));
    }
    if decode_tile_edge.is_some() && !tile_vae_decode {
        return Err(invalid(
            "memory.decode_tile_edge is read only with memory.tile_vae_decode",
        ));
    }
    let query_tile = match (chunk_attention, attention_chunk_size) {
        (false, _) => base.nar.query_tile,
        (true, None) => QueryTile::Upstream,
        (true, Some(elements)) => {
            let (heads, positions) = attention;
            let floor = heads.saturating_mul(positions);
            let elements = elements as usize;
            if elements < floor {
                return Err(invalid(format!(
                    "memory.attention_chunk_size {elements} cannot hold one query row at the \
                     model's full context ({heads} heads × {positions} keys = {floor} elements)"
                )));
            }
            QueryTile::ScoreElements(elements)
        }
    };
    let decode = match (tile_vae_decode, decode_tile_edge) {
        (false, _) | (true, None) => base.decode,
        (true, Some(core)) => DecodeOptions::tiled(core as usize)
            .map_err(|e| invalid(format!("memory.decode_tile_edge: {e}")))?,
    };
    Ok(Some(EngineOptions {
        nar: NarOptions {
            query_tile,
            offload_ar: stage_residency,
        },
        decode,
    }))
}

/// The engine's own memory controls for `spec`: the offload policy's AR offload, the historical
/// query tiles and the production decode tiling.
pub fn engine_options(spec: &LoadSpec) -> EngineOptions {
    EngineOptions {
        nar: NarOptions {
            offload_ar: spec.offload_policy == OffloadPolicy::Sequential,
            ..NarOptions::default()
        },
        decode: DecodeOptions::production(),
    }
}

/// A loaded YuE2 generator.
pub struct Yue2Generator {
    descriptor: ModelDescriptor,
    engine: Yue2Engine,
}

impl std::fmt::Debug for Yue2Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Yue2Generator")
            .field("engine", &self.engine)
            .finish()
    }
}

impl Yue2Generator {
    /// The engine behind this generator (the public native entry points: plan-only, stage-by-stage
    /// runs, closure export).
    pub fn engine(&self) -> &Yue2Engine {
        &self.engine
    }

    /// [`map_request`] plus this engine's [`memory_options`].
    pub fn map(&self, req: &GenerationRequest) -> gen_core::Result<MappedRequest> {
        let mut mapped = map_request(req, self.engine.generation_config())?;
        mapped.settings.options = memory_options(
            req.memory.as_ref(),
            self.engine.options(),
            self.engine.attention_bounds(),
        )?;
        Ok(mapped)
    }
}

/// Maps engine events onto the generator contract's progress: one `Step` per sampled token of the
/// plan and semantic stages (against the phase's budget), the acoustic stage's midpoint steps, and
/// `Decoding`.
struct ProgressBridge<'a> {
    on_progress: &'a mut dyn FnMut(Progress),
    /// `max_tokens` of the ABC and semantic phases.
    abc_total: u32,
    semantic_total: u32,
    /// Tokens sampled so far in the current AR stage.
    tokens: u32,
}

fn saturating_u32(v: impl TryInto<u32>) -> u32 {
    v.try_into().unwrap_or(u32::MAX)
}

impl EngineObserver for ProgressBridge<'_> {
    fn on_stage(&mut self, stage: Stage, event: StageEvent) {
        match (stage, event) {
            (Stage::Plan | Stage::Semantic, StageEvent::Started) => self.tokens = 0,
            (Stage::Decode, StageEvent::Started) => (self.on_progress)(Progress::Decoding),
            _ => {}
        }
    }

    fn on_token(&mut self, stage: Stage, _token: u32) {
        let total = match stage {
            Stage::Plan => self.abc_total,
            Stage::Semantic => self.semantic_total,
            Stage::Synthesis | Stage::Decode => return,
        };
        self.tokens = self.tokens.saturating_add(1);
        (self.on_progress)(Progress::Step {
            current: self.tokens.min(total),
            total,
        });
    }

    fn on_synthesis_progress(&mut self, completed: usize, total: usize) {
        (self.on_progress)(Progress::Step {
            current: saturating_u32(completed),
            total: saturating_u32(total),
        });
    }
}

/// The truncation warnings of a run's `truncated` record (`{"abc": …, "semantic": …}`).
fn truncation_warnings(abc: bool, semantic: bool) -> Vec<GenerationWarning> {
    let mut out = Vec::new();
    if abc {
        out.push(GenerationWarning {
            code: TRUNCATION_CODES[0].into(),
            message: "the ABC score phase hit its max_tokens budget before ABC_END: the plan is \
                      truncated"
                .into(),
        });
    }
    if semantic {
        out.push(GenerationWarning {
            code: TRUNCATION_CODES[1].into(),
            message: "the semantic phase hit its max_tokens budget before MUSIC_END: the song is \
                      truncated"
                .into(),
        });
    }
    out
}

fn recorded_truncation(result: &Value) -> Vec<GenerationWarning> {
    let flag = |key: &str| result.pointer(&format!("/truncated/{key}")) == Some(&json!(true));
    truncation_warnings(flag("abc"), flag("semantic"))
}

fn record_of(result: &Value, dir: &Path) -> gen_core::Result<ArtifactRecord> {
    let text = |key: &str| {
        result
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| invalid(format!("the published record has no `{key}`")))
    };
    Ok(ArtifactRecord {
        dir: dir.to_path_buf(),
        kind: text("kind")?,
        identity: text("identity")?,
    })
}

fn audio(samples: Vec<f32>) -> GenerationOutput {
    GenerationOutput::Audio(AudioTrack {
        samples,
        sample_rate: SAMPLE_RATE,
        channels: CHANNELS,
        stems: Vec::new(),
    })
}

impl Generator for Yue2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request_audio(PROVIDER_ID, req)?;
        let mapped = self.map(req)?;
        if mapped.plan_only {
            return Ok(());
        }
        // A decoder that is not provisioned is refused here, before any model compute.
        self.engine.check_decoder_available(mapped.settings.decoder)
    }

    /// The audio of [`Generator::generate_with_report`]; a `plan_only` request renders none and is
    /// refused before any compute.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let plan_only = req
            .audio
            .as_ref()
            .and_then(|a| a.song.as_ref())
            .is_some_and(|s| s.plan_only);
        if plan_only {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: audio.song.plan_only publishes a plan and renders no audio; call \
                 Generator::generate_with_report to receive its record"
            )));
        }
        self.generate_with_report(req, on_progress)?
            .output
            .ok_or_else(|| invalid("a song render returned no audio"))
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let mapped = self.map(req)?;
        let engine = &self.engine;
        let options = engine.options_for(&mapped.settings);
        let mut warnings: Vec<GenerationWarning> = mapped
            .cover
            .iter()
            .flat_map(|report| &report.warnings)
            .map(|w| GenerationWarning {
                code: w.code.to_string(),
                message: w.message.clone(),
            })
            .collect();
        let cancel = req.cancel.clone();
        let cancelled = move || cancel.is_cancelled();
        let generation = &mapped.settings.generation;
        let mut bridge = ProgressBridge {
            on_progress,
            abc_total: saturating_u32(generation.abc().max_tokens()),
            semantic_total: saturating_u32(generation.semantic().max_tokens()),
            tokens: 0,
        };
        let mut hooks = EngineHooks {
            cancelled: &cancelled,
            observer: &mut bridge,
        };
        let published = |outcome: &RunOutcome| record_of(&outcome.result, &outcome.dir);
        let (samples, artifacts) = match mapped.job {
            Job::DecodeCached(source) => {
                let outcome = engine.decode_cached_with(
                    &source,
                    mapped.settings.decoder,
                    &options.decode,
                    mapped.output.as_ref(),
                    &mut hooks,
                )?;
                // The source's truncation travels with its latents.
                warnings.extend(recorded_truncation(&outcome.result));
                let record = match mapped.output {
                    Some(_) => Some(published(&outcome)?),
                    None => None,
                };
                (Some(outcome.samples), record)
            }
            Job::Generate(request) if mapped.plan_only => {
                let output = mapped
                    .output
                    .as_ref()
                    .ok_or_else(|| invalid("audio.song.plan_only requires audio.artifacts"))?;
                let (plan, _, dir) =
                    engine.plan_to(&request, &mapped.settings.generation, output, &mut hooks)?;
                warnings.extend(truncation_warnings(plan.truncated(), false));
                let result = verify_run(&dir, None).map_err(gen_core::Error::from)?;
                (None, Some(record_of(&result, &dir)?))
            }
            job => {
                let input = match job {
                    Job::Generate(request) => SongInput::Request(request),
                    Job::FromPlan { dir, identity } => {
                        let plan = restore(engine, &dir, identity.as_deref())?;
                        let r = plan.request();
                        let lyrics = req.audio.as_ref().and_then(|a| a.lyrics.as_deref());
                        if (!req.prompt.is_empty() && req.prompt != r.style())
                            || lyrics.is_some_and(|l| l != r.lyrics())
                        {
                            return Err(invalid(
                                "prompt / audio.lyrics disagree with the restored plan's request \
                                 (an edited plan is a new request)",
                            ));
                        }
                        SongInput::Plan(plan)
                    }
                    Job::DecodeCached(_) => unreachable!("handled above"),
                };
                match &mapped.output {
                    Some(output) => {
                        let outcome =
                            engine.generate_to(&input, &mapped.settings, output, &mut hooks)?;
                        warnings.extend(recorded_truncation(&outcome.result));
                        let record = published(&outcome)?;
                        (Some(outcome.samples), Some(record))
                    }
                    None => {
                        let song = match input {
                            SongInput::Request(r) => {
                                engine.generate(&r, &mapped.settings, &mut hooks)?
                            }
                            SongInput::Plan(p) => {
                                engine.generate_from_plan(p, &mapped.settings, &mut hooks)?
                            }
                        };
                        warnings.extend(truncation_warnings(
                            song.semantic.plan.truncated(),
                            song.semantic.truncated,
                        ));
                        (Some(song.audio.samples().to_vec()), None)
                    }
                }
            }
        };
        Ok(GenerationReport {
            output: samples.map(audio),
            artifacts,
            warnings,
        })
    }
}

fn restore(
    engine: &Yue2Engine,
    dir: &std::path::Path,
    identity: Option<&str>,
) -> gen_core::Result<SymbolicPlan> {
    let plan = SymbolicPlan::restore(dir, engine.tokenizer()).map_err(invalid)?;
    if let Some(expected) = identity {
        let actual = plan.identity();
        if actual.to_string() != expected.to_ascii_lowercase() {
            return Err(invalid(format!(
                "the saved plan's identity {actual} is not the expected {expected}"
            )));
        }
    }
    Ok(plan)
}

/// The snapshot directories (and saved generation configuration) a [`LoadSpec`] names, and the
/// tier it asserts.
fn resolve_spec(
    spec: &LoadSpec,
) -> gen_core::Result<(SnapshotDirs, GenerationConfig, Option<Tier>)> {
    let id = PROVIDER_ID;
    let weights = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects the m-a-p/YuE2-3B snapshot directory (or a saved YuE2 closure), not \
                 the single file {}",
                p.display()
            )))
        }
    };
    // The accepted set is the advertised set: an unadvertised quant is refused here.
    if let Some(quant) = spec.quantize.filter(|q| !SUPPORTED_QUANTS.contains(q)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: quantize={quant:?} is not an advertised YuE2 tier (supported_quants: \
             {SUPPORTED_QUANTS:?}; the unquantized bf16 original is quantize: None)"
        )));
    }
    let tier = Tier::from_quant(spec.quantize)?;
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support LoRA/LoKr adapters"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support control/IP-adapter overlays"
        )));
    }
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} loads eagerly; load_shape {:?} is not supported",
            spec.load_shape
        )));
    }
    gen_core::reject_unknown_components(spec, KNOWN_COMPONENTS, id)?;
    if is_saved_closure(&weights) {
        if !spec.components.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: a saved closure carries its own decoders; stage no components beside it"
            )));
        }
        let saved = load_closure(&weights)?;
        return Ok((saved.dirs, saved.generation, tier));
    }
    let dir = |component: &str| -> gen_core::Result<Option<PathBuf>> {
        match spec.components.get(component) {
            None => Ok(None),
            Some(WeightsSource::Dir(p)) => Ok(Some(p.clone())),
            Some(WeightsSource::File(p)) => Err(gen_core::Error::Msg(format!(
                "{id}: component `{component}` must be a snapshot directory, got file {}",
                p.display()
            ))),
        }
    };
    let (standard, legacy) = (dir(VAE_COMPONENT_ID)?, dir(VAE_LEGACY_COMPONENT_ID)?);
    if standard.is_none() && legacy.is_none() {
        return Err(gen_core::Error::Msg(format!(
            "{id} needs at least one decoder snapshot: stage components[\"{VAE_COMPONENT_ID}\"] \
             (m-a-p/YuE2-Vae) and/or components[\"{VAE_LEGACY_COMPONENT_ID}\"] \
             (m-a-p/YuE2-Vae-legacy)"
        )));
    }
    let lm = ComponentId::Lm.component();
    let mut dirs = SnapshotDirs::new().with(lm.repo.id, weights);
    if let Some(p) = standard {
        dirs = dirs.with(ComponentId::VaeStandard.component().repo.id, p);
    }
    if let Some(p) = legacy {
        dirs = dirs.with(ComponentId::VaeLegacy.component().repo.id, p);
    }
    Ok((dirs, GenerationConfig::default(), tier))
}

fn device_and_dtype(spec: &LoadSpec) -> gen_core::Result<(Device, DType)> {
    let device = candle_audio::default_device().map_err(gen_core::Error::from)?;
    let dtype = match (spec.precision, &device) {
        (Precision::Fp32, _) | (_, Device::Cpu) => DType::F32,
        (Precision::Bf16, _) => DType::BF16,
    };
    Ok((device, dtype))
}

#[cfg(test)]
thread_local! {
    /// Unit tests: build the synthetic engine in place of the verified snapshot load, **after**
    /// the `LoadSpec` gate and through the same precision / tier gate — everything else on the
    /// registered path is production code.
    pub(crate) static SYNTHETIC_ENGINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Load the YuE2 generator from `spec` (the registered entry point).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

/// [`load`], returning the concrete generator (and so its [`Yue2Engine`]).
pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<Yue2Generator> {
    let (dirs, generation, tier) = resolve_spec(spec)?;
    let (device, dtype) = device_and_dtype(spec)?;
    let precision = ModelPrecision {
        tier,
        ..ModelPrecision::default()
    };
    let options = engine_options(spec);
    #[cfg(test)]
    if SYNTHETIC_ENGINE.with(|s| s.get()) {
        let _ = (generation, device, dtype);
        let weights = match &spec.weights {
            WeightsSource::Dir(p) | WeightsSource::File(p) => p.as_path(),
        };
        return Ok(Yue2Generator {
            descriptor: descriptor(),
            engine: Yue2Engine::synthetic_at(precision, options, weights)?
                .with_checked_decoders(dirs),
        });
    }
    let engine =
        Yue2Engine::load_with_precision(&dirs, dtype, &device, precision, generation, options)?;
    Ok(Yue2Generator {
        descriptor: descriptor(),
        engine,
    })
}

candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// The registration, in catalog order.
pub const REGISTRATIONS: [gen_core::registry::ModelRegistration; 1] = [REGISTRATION];

/// The licence rows of the components this provider loads (the generation closure) — what the
/// audio catalog publishes. The cover closure's rows ([`crate::license::COMPONENT_LICENSES`]) join
/// the catalog with the provider that loads them.
pub const PROVIDER_COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[
    crate::license::LICENSE_YUE2_3B,
    crate::license::LICENSE_QWEN_TIKTOKEN,
    crate::license::LICENSE_YUE2_VAE,
    crate::license::LICENSE_YUE2_VAE_LEGACY,
];

/// Provider → component mapping: the generation closure (MoT, tokenizer, both decoders). The
/// cover closure (SheetSage2 + MERT-v2-FullSong) is not loaded by this provider: a cover here plans
/// from a reviewed score, and transcription stays in the gated `candle-audio-sheetsage2` crate.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[gen_core::ProviderComponents {
    provider_id: PROVIDER_ID,
    components: &[
        "yue2_3b",
        "yue2_qwen_tiktoken",
        "yue2_vae",
        "yue2_vae_legacy",
    ],
}];

#[cfg(test)]
mod tests;
