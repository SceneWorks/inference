//! `AceStepGenerator` — the [`gen_core::Generator`] implementation for **ACE-Step 1.5** on the
//! candle audio lane (sc-12842), plus its [`descriptor`]/[`load`] entry points and the explicit
//! registration constant wired into `candle-audio-catalog` under the id **`acestep_v15_turbo`** —
//! the audio lane's music/song (text + lyrics) provider.
//!
//! ## Snapshot layout
//!
//! [`load`] expects an `ACE-Step/acestep-v15-xl-turbo-diffusers`-shaped diffusers snapshot dir:
//!
//! ```text
//!   model_index.json                                        → pipeline identity
//!   scheduler/scheduler_config.json                         → flow-match shift default
//!   transformer/config.json + diffusion_pytorch_model-*.safetensors + index → the ~2B DiT
//!   condition_encoder/config.json + diffusion_pytorch_model.safetensors     → lyric/timbre encoder
//!   text_encoder/config.json + model.safetensors            → Qwen3-Embedding-0.6B
//!   tokenizer/tokenizer.json                                → the Qwen tokenizer
//!   vae/config.json + diffusion_pytorch_model.safetensors   → the stereo Oobleck VAE
//! ```
//!
//! ## Request mapping
//!
//! `prompt` is the style/genre/instrument/mood caption; [`gen_core::AudioParams`] carries `lyrics`
//! (structured with `[verse]`/`[chorus]` tags), `bpm`, `musical_key`, `target_duration`, and
//! `language` (`vocal_language`). `steps` (turbo default 8) and `scheduler_shift` (default 3.0) map
//! onto the flow-match sampler; `guidance` is ignored (the turbo checkpoint is guidance-distilled).
//! Progress is one `Step` per solver step plus `Decoding` before the VAE decode; cancellation is
//! checked before generate, at every solver step, between DiT blocks, AND inside the VAE decode.
//! Determinism: same request + seed ⇒ byte-identical samples. Output is a single stereo mix
//! (`stems` empty — ACE-Step 1.5 text-to-music renders a mixdown, not separated stems).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioEditMode, AudioTrack, Capabilities, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};
use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
use candle_audio::{AudioError, Result as AudioResult};

use crate::pipeline::{
    AceStepPipeline, EditParams, EditTask, PipelineProgress, SynthesisParams, DEFAULT_SECONDS,
    DEFAULT_STEPS,
};
use crate::scheduler::DEFAULT_SHIFT;
use crate::text::Metadata;

/// Registry id (the SceneWorks worker routes `payload.model` to this exact id).
pub const MODEL_ID: &str = "acestep_v15_turbo";

/// Hub pin: `ACE-Step/acestep-v15-xl-turbo-diffusers` at an immutable commit SHA (MIT weights +
/// code — ACE Studio / StepFun; commercial-OK, and it ships its own Oobleck VAE, so the
/// Stability-licensed DiffRhythm VAE trap does not apply).
pub const HUB_REPO: &str = "ACE-Step/acestep-v15-xl-turbo-diffusers";
pub const HUB_REVISION: &str = "200ba991ae448051e14b0183157e35c2d27c9fb0";

/// Native output sample rate (Hz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Output channels (stereo).
pub const CHANNELS: u16 = 2;

/// Longest clip the model synthesizes (the trained 10-minute window; CPU cost scales with it).
pub const MAX_DURATION_SECS: f32 = 600.0;

/// Solver-step ceiling — far above the turbo default 8 (users push base/sft to 30–60), a sanity
/// bound rather than a quality claim.
pub const MAX_STEPS: u32 = 200;

/// Prompt / lyric languages the model supports (the documented 50+; the advertised subset is the
/// set the model card names — the code is advisory, not a model switch).
pub const LANGUAGES: &[&str] = &["en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"];

/// ACE-Step's identity + capabilities — constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "acestep",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            // The turbo checkpoint is guidance-distilled: CFG is baked into the weights, so no
            // negative-prompt / guidance surface is advertised (an explicit value is a typed
            // Unsupported / ignored, never a second forward).
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // Prompted source-audio editing (sc-12847): the SAME turbo weights natively serve
            // ACE-Step's audio-to-audio task modes, so the edit capability rides this existing
            // generator via a new conditioning kind rather than a distinct provider id.
            conditioning: vec![ConditioningKind::AudioEdit],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 1,
            max_size: 4096,
            // One clip per request (GenerationOutput::Audio carries a single track).
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            // No voice/speaker surface — music, not TTS.
            audio_voices: vec![],
            audio_languages: LANGUAGES.to_vec(),
            // The edit modes the pinned diffusers checkpoint supports. Inpaint / Repaint / Extend
            // ride ACE-Step's `repaint` task (region regenerate + seamless stitch, sc-12847). Cover
            // (the `cover` task) is deliberately NOT advertised: it needs the audio
            // quantizer/detokenizer weights that this diffusers snapshot does not ship (tracked
            // follow-up) — so a Cover request is a typed Unsupported at the contract floor, honest
            // about the pinned weights rather than erroring mid-generate.
            audio_edit_modes: vec![
                AudioEditMode::Inpaint,
                AudioEditMode::Repaint,
                AudioEditMode::Extend,
            ],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
        },
    }
}

/// Capability-driven request validation, factored out for weightless unit tests.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt (the music style description) must not be empty"
        )));
    }
    let caps = &desc.capabilities;
    // Honor the advertised (audio-unused) size bounds so validate never accepts out-of-surface.
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width/height {}x{} outside the advertised {}..={} (unused by audio, but the \
             advertised surface is honored)",
            req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if let Some(steps) = req.steps {
        if steps > MAX_STEPS {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps {steps} above the {MAX_STEPS}-step ceiling"
            )));
        }
    }
    if let Some(s) = req.scheduler_shift {
        if s <= 0.0 {
            return Err(gen_core::Error::Msg(format!(
                "{id}: scheduler_shift (flow-match sigma shift) must be > 0, got {s}"
            )));
        }
    }
    if let Some(audio) = &req.audio {
        if let Some(bpm) = audio.bpm {
            if !bpm.is_finite() || bpm <= 0.0 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: audio.bpm must be finite and > 0, got {bpm}"
                )));
            }
        }
        if let Some(d) = audio.target_duration {
            // The model needs ≥ 1 latent frame (0.04 s at 25 fps); enforce a friendly floor.
            if d < 0.04 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: audio.target_duration {d}s below the 0.04 s floor (25 fps latents)"
                )));
            }
        }
        // musical_key / lyrics are free-form; the model accepts any text (empty lyrics ⇒
        // instrumental). No enum to gate.
    }
    // Prompted source-audio editing (sc-12847): the shared floor gates the edit *mode* against the
    // advertised surface and the region shape; here we add the checks that need the source clip —
    // native sample rate, and region/extend bounds against the clip duration.
    if let Some(edit) = req.audio_edit() {
        let src = edit.audio;
        if src.channels == 0 || src.samples.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: audio edit source clip is empty"
            )));
        }
        if src.sample_rate != SAMPLE_RATE {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: audio edit source must be {SAMPLE_RATE} Hz (the model's native rate), got {}",
                src.sample_rate
            )));
        }
        let src_secs = (src.samples.len() / src.channels as usize) as f32 / src.sample_rate as f32;
        match edit.mode {
            AudioEditMode::Extend => {
                let end = edit.region.and_then(|r| r.end_secs).ok_or_else(|| {
                    gen_core::Error::Msg(format!(
                        "{id}: extend requires region.end_secs (the new total length in seconds)"
                    ))
                })?;
                if end <= src_secs {
                    return Err(gen_core::Error::Msg(format!(
                        "{id}: extend length {end}s must exceed the source length {src_secs:.3}s"
                    )));
                }
            }
            AudioEditMode::Inpaint | AudioEditMode::Repaint => {
                let region = edit.region.ok_or_else(|| {
                    gen_core::Error::Msg(format!(
                        "{id}: {:?} editing requires a region (start/end seconds)",
                        edit.mode
                    ))
                })?;
                if region.start_secs >= src_secs {
                    return Err(gen_core::Error::Msg(format!(
                        "{id}: audio edit region start {}s is at/beyond the {src_secs:.3}s clip",
                        region.start_secs
                    )));
                }
                if let Some(end) = region.end_secs {
                    if end > src_secs + 1e-3 {
                        return Err(gen_core::Error::Msg(format!(
                            "{id}: audio edit region end {end}s is beyond the {src_secs:.3}s clip"
                        )));
                    }
                }
            }
            // Cover is not in the advertised `audio_edit_modes`, so the floor below rejects it as a
            // typed Unsupported; no clip-specific check needed here.
            AudioEditMode::Cover => {}
        }
    }
    caps.validate_request_audio(id, req)
}

/// A loaded (lazy) ACE-Step generator. The heavy pipeline (Qwen0.6B + condition encoder + ~2B DiT
/// + Oobleck VAE) is built on first use and cached; `load` does no file I/O beyond argument checks.
pub struct AceStepGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<AceStepPipeline>>>,
}

impl AceStepGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<AceStepPipeline>> {
        let mut guard = lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let device = candle_audio::default_device()?;
        let built = Arc::new(AceStepPipeline::from_snapshot(&self.root, &device)?);
        *guard = Some(built.clone());
        Ok(built)
    }
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Construct the (lazy) generator from a [`LoadSpec`].
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a diffusers snapshot directory (model_index.json + \
                 transformer/ + condition_encoder/ + text_encoder/ + tokenizer/ + vae/), not a \
                 single file"
            )));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support on-the-fly quantization"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support LoRA/LoKr adapters"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support control/IP-adapter overlays"
        )));
    }
    Ok(Box::new(AceStepGenerator {
        descriptor: descriptor(),
        root,
        pipeline: Mutex::new(None),
    }))
}

impl Generator for AceStepGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let audio = req.audio.clone().unwrap_or_default();
        let params = SynthesisParams {
            seconds: audio.target_duration.unwrap_or(DEFAULT_SECONDS),
            steps: req.steps.unwrap_or(DEFAULT_STEPS as u32) as usize,
            shift: req
                .scheduler_shift
                .map(|s| s as f64)
                .unwrap_or(DEFAULT_SHIFT),
            lyrics: audio.lyrics.clone().unwrap_or_default(),
            metadata: Metadata {
                bpm: audio.bpm,
                key: audio.musical_key.clone(),
                time_signature: None,
                vocal_language: audio.language.clone(),
            },
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
        };

        let pipeline = self.pipeline()?;
        let total = params.steps as u32;
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let mut progress = |p: PipelineProgress| match p {
            PipelineProgress::Step(k) => on_progress(Progress::Step {
                current: k as u32,
                total,
            }),
            PipelineProgress::Decoding => on_progress(Progress::Decoding),
        };

        // Prompted source-audio editing (sc-12847): a request carrying an `AudioEdit` conditioning
        // dispatches to the mask-conditioned edit path (regenerate the region + seamless stitch);
        // otherwise this is plain text-to-music synthesis.
        let samples = if let Some(edit) = req.audio_edit() {
            let task = match edit.mode {
                AudioEditMode::Inpaint => EditTask::Inpaint,
                AudioEditMode::Repaint => EditTask::Repaint,
                AudioEditMode::Extend => EditTask::Extend,
                // Defensive: `validate` already rejects Cover (unadvertised — the pinned diffusers
                // checkpoint ships no audio quantizer), but never reach the pipeline with it.
                AudioEditMode::Cover => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{}: cover editing is unavailable on the pinned checkpoint (no audio \
                         quantizer weights)",
                        self.descriptor.id
                    )));
                }
            };
            let (region_start_secs, region_end_secs) = edit
                .region
                .map(|r| (r.start_secs, r.end_secs))
                .unwrap_or((0.0, None));
            let source = edit.audio.samples.clone();
            let source_channels = edit.audio.channels as usize;
            let eparams = EditParams {
                task,
                region_start_secs,
                region_end_secs,
                base: params,
            };
            pipeline
                .edit(
                    &req.prompt,
                    &source,
                    source_channels,
                    &eparams,
                    &mut progress,
                    &probe,
                )
                .map_err(gen_core::Error::from)?
        } else {
            pipeline
                .synthesize(&req.prompt, &params, &mut progress, &probe)
                .map_err(gen_core::Error::from)?
        };

        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
            // ACE-Step 1.5 text-to-music emits a single stereo mix; stem separation is a distinct
            // audio-to-audio task (the reference "extract"/"lego" modes require input audio), so
            // no stems are produced here — the additive surface is left empty rather than faked.
            stems: Vec::new(),
        }))
    }
}

// Explicit catalog registration for `acestep_v15_turbo` (composed by `candle-audio-catalog`).
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Materialize the pinned ACE-Step snapshot through the audio lane's F-029 hub path: the
/// component configs, the sharded DiT safetensors (enumerated from its index), the
/// condition-encoder / text-encoder / VAE weights, the tokenizer, and the silence latent — all at
/// [`HUB_REVISION`]. Returns the snapshot dir as a [`WeightsSource::Dir`] ready for a [`LoadSpec`].
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let dir = pinned_snapshot_dir(HUB_REPO, HUB_REVISION, "model_index.json")?;
    for file in [
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "condition_encoder/config.json",
        "condition_encoder/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "silence_latent.pt",
    ] {
        hf_get_pinned(HUB_REPO, HUB_REVISION, file)?;
    }
    // DiT shards, enumerated from the index so a re-sharded upstream layout cannot silently skip a
    // file.
    for shard in dit_shards(HUB_REPO, HUB_REVISION)? {
        hf_get_pinned(HUB_REPO, HUB_REVISION, &format!("transformer/{shard}"))?;
    }
    Ok(dir)
}

/// The transformer safetensors shard filenames listed in
/// `transformer/diffusion_pytorch_model.safetensors.index.json`.
fn dit_shards(repo: &str, revision: &str) -> AudioResult<Vec<String>> {
    let index_path = hf_get_pinned(
        repo,
        revision,
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    )?;
    let text = std::fs::read_to_string(&index_path)
        .map_err(|e| AudioError::Msg(format!("read {}: {e}", index_path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| AudioError::Msg(format!("parse {}: {e}", index_path.display())))?;
    let mut shards: Vec<String> = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .map(|m| {
            m.values()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(AudioError::Msg(format!(
            "{repo}: transformer index lists no shards"
        )));
    }
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag};

    fn music_req(audio: AudioParams) -> GenerationRequest {
        GenerationRequest {
            prompt: "upbeat electronic dance track".into(),
            audio: Some(audio),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_music_surface() {
        let d = descriptor();
        assert_eq!(d.id, "acestep_v15_turbo");
        assert_eq!(d.family, "acestep");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [48_000]);
        assert_eq!(d.capabilities.max_audio_duration_secs, Some(600.0));
        assert!(
            d.capabilities.audio_voices.is_empty(),
            "music has no voices"
        );
        assert!(d.capabilities.audio_languages.contains(&"en"));
        assert!(d.capabilities.audio_languages.contains(&"zh"));
        assert!(
            !d.capabilities.supports_guidance,
            "turbo is guidance-distilled"
        );
        assert_eq!(d.capabilities.max_count, 1);
    }

    #[test]
    fn validate_gates_lyrics_bpm_key_and_sampling() {
        let d = descriptor();
        // A full in-surface music request passes (lyrics + bpm + key + language + duration).
        let ok = music_req(AudioParams {
            target_duration: Some(30.0),
            sample_rate: Some(48_000),
            language: Some("en".into()),
            bpm: Some(128.0),
            musical_key: Some("C minor".into()),
            lyrics: Some("[verse]\nhello world".into()),
            ..Default::default()
        });
        assert!(validate_request(&d, &ok).is_ok());

        // Out-of-surface values rejected.
        for bad in [
            AudioParams {
                target_duration: Some(MAX_DURATION_SECS + 1.0),
                ..Default::default()
            },
            AudioParams {
                sample_rate: Some(44_100),
                ..Default::default()
            },
            AudioParams {
                language: Some("xx".into()),
                ..Default::default()
            },
            AudioParams {
                voice: Some("af_heart".into()),
                ..Default::default()
            },
            AudioParams {
                bpm: Some(-5.0),
                ..Default::default()
            },
            AudioParams {
                bpm: Some(f32::NAN),
                ..Default::default()
            },
        ] {
            assert!(
                validate_request(&d, &music_req(bad.clone())).is_err(),
                "{bad:?} must reject"
            );
        }
        // Steps ceiling.
        let mut r = music_req(AudioParams::default());
        r.steps = Some(MAX_STEPS + 1);
        assert!(validate_request(&d, &r).is_err());
        // Empty prompt.
        let mut r = music_req(AudioParams::default());
        r.prompt = "   ".into();
        assert!(validate_request(&d, &r).is_err());
    }

    #[test]
    fn descriptor_advertises_the_edit_surface() {
        let d = descriptor();
        assert!(
            d.capabilities
                .conditioning
                .contains(&gen_core::ConditioningKind::AudioEdit),
            "advertises the AudioEdit conditioning kind"
        );
        // Exactly the checkpoint-supported modes, in order; Cover is absent (no audio quantizer).
        assert_eq!(
            d.capabilities.audio_edit_modes,
            vec![
                AudioEditMode::Inpaint,
                AudioEditMode::Repaint,
                AudioEditMode::Extend
            ]
        );
    }

    fn edit_track(secs: f32) -> AudioTrack {
        let frames = (secs * SAMPLE_RATE as f32) as usize;
        AudioTrack {
            samples: vec![0.0; frames * CHANNELS as usize],
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
            stems: Vec::new(),
        }
    }

    fn edit_req(mode: AudioEditMode, region: Option<gen_core::TimeRegion>) -> GenerationRequest {
        GenerationRequest {
            prompt: "energetic guitar solo".into(),
            conditioning: vec![gen_core::Conditioning::AudioEdit {
                audio: edit_track(12.0),
                mode,
                region,
                strength: None,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn validate_gates_the_edit_surface() {
        let d = descriptor();
        // A well-formed interior repaint (seconds 4–8 of a 12 s clip) passes.
        assert!(validate_request(
            &d,
            &edit_req(
                AudioEditMode::Repaint,
                Some(gen_core::TimeRegion {
                    start_secs: 4.0,
                    end_secs: Some(8.0),
                }),
            )
        )
        .is_ok());
        // Cover is unadvertised on the pinned checkpoint → typed Unsupported.
        assert!(matches!(
            validate_request(&d, &edit_req(AudioEditMode::Cover, None)).unwrap_err(),
            gen_core::Error::Unsupported(_)
        ));
        // A region past the clip is rejected.
        assert!(validate_request(
            &d,
            &edit_req(
                AudioEditMode::Repaint,
                Some(gen_core::TimeRegion {
                    start_secs: 4.0,
                    end_secs: Some(20.0),
                }),
            )
        )
        .is_err());
        // Extend without an end (the new total length) is rejected.
        assert!(validate_request(
            &d,
            &edit_req(
                AudioEditMode::Extend,
                Some(gen_core::TimeRegion {
                    start_secs: 12.0,
                    end_secs: None,
                }),
            )
        )
        .is_err());
        // Extend to a longer clip passes.
        assert!(validate_request(
            &d,
            &edit_req(
                AudioEditMode::Extend,
                Some(gen_core::TimeRegion {
                    start_secs: 12.0,
                    end_secs: Some(20.0),
                }),
            )
        )
        .is_ok());
        // A source at the wrong sample rate is rejected.
        let mut wrong = edit_req(
            AudioEditMode::Repaint,
            Some(gen_core::TimeRegion {
                start_secs: 1.0,
                end_secs: Some(2.0),
            }),
        );
        if let gen_core::Conditioning::AudioEdit { audio, .. } = &mut wrong.conditioning[0] {
            audio.sample_rate = 44_100;
        }
        assert!(matches!(
            validate_request(&d, &wrong).unwrap_err(),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn load_rejects_unsupported_spec_shapes() {
        let dir = std::env::temp_dir();
        let spec = LoadSpec::new(WeightsSource::File(dir.join("x.safetensors")));
        assert!(load(&spec).is_err());
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir = std::env::temp_dir().join("acestep-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = GenerationRequest {
            prompt: "lofi beats".into(),
            cancel: flag,
            ..Default::default()
        };
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }
}
