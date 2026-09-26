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
//! | `weights` | `Dir`: the `m-a-p/YuE2-3B` snapshot (holds `qwen.tiktoken`), **or** a closure saved by [`crate::closure::save_closure`] (then no components) |
//! | `components["vae"]` | `Dir`: the `m-a-p/YuE2-Vae` snapshot (required with a plain snapshot) |
//! | `components["vae_legacy"]` | `Dir`: the `m-a-p/YuE2-Vae-legacy` snapshot (optional; needed for `decoder: Legacy`) |
//! | `precision` | `Fp32` ⇒ F32; the default ⇒ BF16 on an accelerator, F32 on the CPU (no BF16 matmul) |
//! | `quantize` | refused — precision tiers are a later slice (sc-22995) |
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
//! | `audio.song.score` | external ABC |
//! | `audio.song.score_sampling` / `semantic_sampling` | the two AR phases' sampling overrides |
//! | `audio.song.plan` | restore an exact saved plan (its request must agree with any request field set) |
//! | `audio.song.cached_latents` | decode a completed run's verified latents (no generation fields) |
//! | `audio.song.decoder` | `Standard` (default) / `Legacy` |
//! | `audio.artifacts` | publish the run directory transactionally; `resume` reuses matching work |
//!
//! Every other audio field (`target_duration`, `bpm`, `musical_key`, `voice`, `language`,
//! `repetition_penalty` — ambiguous between the two phases —, segment and limiter controls) is
//! refused rather than dropped.

use std::path::PathBuf;

use candle_audio::candle_core::{DType, Device};
use candle_audio::gen_core::{
    self, AudioParams, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, SongDecoder, SongPlanning,
    TokenSampling, WeightsSource,
};

use crate::closure::{is_saved_closure, load_closure};
use crate::engine::{EngineHooks, EngineObserver, EngineOptions, SongSettings, Stage, Yue2Engine};
use crate::inventory::{ComponentId, VaeVariant};
use crate::plan::SymbolicPlan;
use crate::protocol::{
    CotMode, GenerationConfig, SamplingOverrides, SongRequest, SongRequestSpec, DEFAULT_ID,
    DEFAULT_SEED,
};
use crate::run::{RunOutput, SongInput};
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
/// Components a plain-snapshot load requires.
pub const REQUIRED_COMPONENTS: &[&str] = &[VAE_COMPONENT_ID];
/// Every component id the loader recognizes.
pub const KNOWN_COMPONENTS: &[&str] = &[VAE_COMPONENT_ID, VAE_LEGACY_COMPONENT_ID];
/// Output channels (stereo).
pub const CHANNELS: u16 = 2;

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

/// What a request asks the engine to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Job {
    /// Generate from a request.
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
    /// Sampling, ODE steps and decoder.
    pub settings: SongSettings,
    /// Where to publish, when the request asks for artifacts.
    pub output: Option<RunOutput>,
}

/// Map a [`GenerationRequest`] onto the engine (see the [module docs](self)); `base` is the
/// engine's generation configuration the overrides apply to.
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
            "YuE2 generation takes no reference audio (covers are a separate slice)",
        ));
    }
    let song = song.unwrap_or_default();
    let output = artifacts.map(|a| RunOutput {
        dir: a.dir,
        resume: a.resume,
    });
    let decoder = decoder_of(song.decoder);

    // Decoding cached latents takes nothing that would regenerate them.
    if let Some(source) = song.cached_latents {
        let stray = [
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
        ];
        if let Some((field, _)) = stray.iter().find(|(_, set)| *set) {
            return Err(invalid(format!(
                "{field} cannot be combined with audio.song.cached_latents: a cached decode \
                 re-renders the source run's latents and generates nothing"
            )));
        }
        return Ok(MappedRequest {
            job: Job::DecodeCached(source),
            settings: SongSettings {
                generation: base.clone(),
                decoder,
            },
            output,
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
    };

    if let Some(plan) = song.plan {
        let stray = [
            ("seed", req.seed.is_some()),
            ("guidance", req.guidance.is_some()),
            ("audio.song.planning", song.planning.is_some()),
            ("audio.song.score", song.score.is_some()),
            ("audio.song.score_sampling", song.score_sampling.is_some()),
        ];
        if let Some((field, _)) = stray.iter().find(|(_, set)| *set) {
            return Err(invalid(format!(
                "{field} cannot be combined with audio.song.plan: the restored plan's request \
                 fixes it (an edited plan is a new request)"
            )));
        }
        // `prompt` / `audio.lyrics`, when set, are checked against the restored plan's request
        // in `generate` (the plan is read there, not while mapping).
        return Ok(MappedRequest {
            job: Job::FromPlan {
                dir: plan.dir,
                identity: plan.identity,
            },
            settings,
            output,
        });
    }

    let Some(lyrics) = lyrics else {
        return Err(invalid("audio.lyrics is required"));
    };
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
    })
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
}

/// Maps engine events onto the generator contract's progress.
struct ProgressBridge<'a> {
    on_progress: &'a mut dyn FnMut(Progress),
}

impl EngineObserver for ProgressBridge<'_> {
    fn on_stage(&mut self, stage: Stage, event: crate::engine::StageEvent) {
        if stage == Stage::Decode && event == crate::engine::StageEvent::Started {
            (self.on_progress)(Progress::Decoding);
        }
    }

    fn on_synthesis_progress(&mut self, completed: usize, total: usize) {
        (self.on_progress)(Progress::Step {
            current: u32::try_from(completed).unwrap_or(u32::MAX),
            total: u32::try_from(total).unwrap_or(u32::MAX),
        });
    }
}

impl Generator for Yue2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request_audio(PROVIDER_ID, req)?;
        map_request(req, self.engine.generation_config()).map(|_| ())
    }

    /// Progress: `Progress::Step` per acoustic midpoint step over all chunks, `Progress::Decoding`
    /// when the decoder starts.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let mapped = map_request(req, self.engine.generation_config())?;
        let cancel = req.cancel.clone();
        let cancelled = move || cancel.is_cancelled();
        let mut bridge = ProgressBridge { on_progress };
        let mut hooks = EngineHooks {
            cancelled: &cancelled,
            observer: &mut bridge,
        };
        let engine = &self.engine;
        let samples = match mapped.job {
            Job::DecodeCached(source) => {
                engine
                    .decode_cached(
                        &source,
                        mapped.settings.decoder,
                        mapped.output.as_ref(),
                        &mut hooks,
                    )?
                    .samples
            }
            job => {
                let input = match job {
                    Job::Generate(request) => SongInput::Request(request),
                    Job::FromPlan { dir, identity } => {
                        let plan = restore(engine, &dir, identity.as_deref())?;
                        let r = plan.request();
                        let audio = req.audio.as_ref();
                        if (!req.prompt.is_empty() && req.prompt != r.style())
                            || audio
                                .and_then(|a| a.lyrics.as_deref())
                                .is_some_and(|l| l != r.lyrics())
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
                        engine
                            .generate_to(&input, &mapped.settings, output, &mut hooks)?
                            .samples
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
                        song.audio.samples().to_vec()
                    }
                }
            }
        };
        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
            stems: Vec::new(),
        }))
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

/// The snapshot directories (and saved generation configuration) a [`LoadSpec`] names.
fn resolve_spec(spec: &LoadSpec) -> gen_core::Result<(SnapshotDirs, GenerationConfig)> {
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
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: quantized tiers are not available yet; load the BF16 checkpoint"
        )));
    }
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
    gen_core::reject_unknown_components(spec, KNOWN_COMPONENTS, id)?;
    if is_saved_closure(&weights) {
        if !spec.components.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: a saved closure carries its own decoders; stage no components beside it"
            )));
        }
        let saved = load_closure(&weights)?;
        return Ok((saved.dirs, saved.generation));
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
    gen_core::require_component(spec, VAE_COMPONENT_ID, id, "YuE2-Vae snapshot")?;
    let lm = ComponentId::Lm.component();
    let mut dirs = SnapshotDirs::new().with(lm.repo.id, weights);
    if let Some(p) = dir(VAE_COMPONENT_ID)? {
        dirs = dirs.with(ComponentId::VaeStandard.component().repo.id, p);
    }
    if let Some(p) = dir(VAE_LEGACY_COMPONENT_ID)? {
        dirs = dirs.with(ComponentId::VaeLegacy.component().repo.id, p);
    }
    Ok((dirs, GenerationConfig::default()))
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
    /// the `LoadSpec` gate — everything else on the registered path is production code.
    pub(crate) static SYNTHETIC_ENGINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Load the YuE2 generator from `spec` (the registered entry point).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

/// [`load`], returning the concrete generator (and so its [`Yue2Engine`]).
pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<Yue2Generator> {
    let (dirs, generation) = resolve_spec(spec)?;
    let (device, dtype) = device_and_dtype(spec)?;
    #[cfg(test)]
    if SYNTHETIC_ENGINE.with(|s| s.get()) {
        let _ = (dirs, generation, device, dtype);
        return Ok(Yue2Generator {
            descriptor: descriptor(),
            engine: Yue2Engine::synthetic(EngineOptions::default()),
        });
    }
    let engine = Yue2Engine::load(&dirs, dtype, &device, generation, EngineOptions::default())?;
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

/// Provider → component mapping: the generation closure (MoT, tokenizer, both decoders). The
/// cover closure (SheetSage2 + MERT-v2-FullSong) is not loaded by this provider.
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
