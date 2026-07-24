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

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioEditMode, AudioTrack, Capabilities, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};
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

/// The license of the pinned ACE-Step v1.5 XL Turbo weight checkpoint (sc-13332) — surfaced for
/// SceneWorks' end-product licenses page. MIT (permissive), verified against the
/// `ACE-Step/acestep-v15-xl-turbo-diffusers` model card. The bundled `text_encoder`
/// (Qwen3-Embedding-0.6B) is redistributed under Apache-2.0 — noted so the product surfaces the
/// full picture even though the primary weight license governs.
pub const WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "MIT",
        name: "MIT License",
        source_url: "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers",
        attribution: Some("ACE-Step v1.5 XL Turbo © ACE-Step — licensed under MIT"),
        commercial_use: true,
        restriction: Some(
            "Bundled text_encoder (Qwen3-Embedding-0.6B) is redistributed under Apache-2.0.",
        ),
    };

/// This provider's **composite** weight-license entry (keyed by [`MODEL_ID`], `component == None`)
/// for catalog aggregation — the at-a-glance effective license. All ACE-Step checkpoints (turbo
/// primary + the sft cover FSQ modules) are MIT, so the composite is MIT.
pub const WEIGHT_LICENSE_ENTRY: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: None,
        license: WEIGHT_LICENSE,
    };

/// Hub pin for the Cover checkpoint (sc-13251): `ACE-Step/acestep-v15-xl-sft-diffusers` at an
/// immutable commit SHA (MIT). The pinned turbo checkpoint ships no `audio_tokenizer` /
/// `audio_token_detokenizer`; this sibling checkpoint (same org, same 64-ch/25 Hz acoustic latent
/// space) does. The Cover restyle pulls the two FSQ component dirs AND the non-distilled
/// `transformer` (~7.8 GB) — the reference's actual cover DiT — reusing the already-loaded turbo
/// text-encoder / condition-encoder / VAE. Everything else (Inpaint/Repaint/Extend, text-to-music)
/// stays on the turbo DiT.
pub const SFT_HUB_REPO: &str = "ACE-Step/acestep-v15-xl-sft-diffusers";
pub const SFT_HUB_REVISION: &str = "4bf7b60a63b27144f539f980927eeb89f5f912b0";

/// The [`LoadSpec::components`] id under which the caller stages the sft Cover snapshot dir (epic
/// 13657/13678). **Optional / on-demand**, deliberately NOT a
/// [`ModelDescriptor::required_components`] id: only a Cover request needs it, so text2music and the
/// region edit modes (Inpaint/Repaint/Extend) load and run without it — mirroring LTX's optional
/// `uncensored_enhancer`. When provisioned it is a [`WeightsSource::Dir`] pointing at an
/// `ACE-Step/acestep-v15-xl-sft-diffusers` snapshot (`audio_tokenizer/`, `audio_token_detokenizer/`,
/// `transformer/`); a Cover request without it errors actionably at generate and never self-fetches.
pub const COVER_COMPONENT_ID: &str = "sft_cover";

/// License of the sft `audio_tokenizer` (FSQ) cover-conditioning checkpoint — MIT, verified against
/// the `acestep-v15-xl-sft-diffusers` model card.
pub const AUDIO_TOKENIZER_WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "MIT",
        name: "MIT License",
        source_url: "https://huggingface.co/ACE-Step/acestep-v15-xl-sft-diffusers",
        attribution: Some(
            "ACE-Step v1.5 XL SFT audio_tokenizer (FSQ) © ACE-Step — licensed under MIT",
        ),
        commercial_use: true,
        restriction: None,
    };

/// License of the sft `audio_token_detokenizer` cover-conditioning checkpoint — MIT.
pub const AUDIO_TOKEN_DETOKENIZER_WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "MIT",
        name: "MIT License",
        source_url: "https://huggingface.co/ACE-Step/acestep-v15-xl-sft-diffusers",
        attribution: Some(
            "ACE-Step v1.5 XL SFT audio_token_detokenizer © ACE-Step — licensed under MIT",
        ),
        commercial_use: true,
        restriction: None,
    };

/// Per-checkpoint attribution row for the sft `audio_tokenizer` (component of [`MODEL_ID`]).
pub const WEIGHT_LICENSE_ENTRY_AUDIO_TOKENIZER: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("audio_tokenizer"),
        license: AUDIO_TOKENIZER_WEIGHT_LICENSE,
    };

/// Per-checkpoint attribution row for the sft `audio_token_detokenizer` (component of [`MODEL_ID`]).
pub const WEIGHT_LICENSE_ENTRY_AUDIO_TOKEN_DETOKENIZER: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("audio_token_detokenizer"),
        license: AUDIO_TOKEN_DETOKENIZER_WEIGHT_LICENSE,
    };

/// License of the sft `transformer` — the non-distilled reference cover DiT (sc-13251) — MIT.
pub const SFT_TRANSFORMER_WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "MIT",
        name: "MIT License",
        source_url: "https://huggingface.co/ACE-Step/acestep-v15-xl-sft-diffusers",
        attribution: Some(
            "ACE-Step v1.5 XL SFT transformer (cover DiT) © ACE-Step — licensed under MIT",
        ),
        commercial_use: true,
        restriction: None,
    };

/// Per-checkpoint attribution row for the sft `transformer` cover DiT (component of [`MODEL_ID`]).
pub const WEIGHT_LICENSE_ENTRY_SFT_TRANSFORMER: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("transformer"),
        license: SFT_TRANSFORMER_WEIGHT_LICENSE,
    };

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
        // Cover's ~7.8 GB sft snapshot is an OPTIONAL, on-demand component ([`COVER_COMPONENT_ID`] =
        // `sft_cover`, read only for a Cover request), NOT a hard requirement — text2music + the
        // region edit modes load without it — so it is deliberately absent here (mirrors LTX's
        // optional `uncensored_enhancer`).
        required_components: &[],
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
            // Pure audio: no width/height. The descriptor sweep exempts Audio from the size floor
            // (sc-13314) and `validate_request_audio` skips the range, so these stay at the natural
            // unused 0 rather than a nominal placeholder bound.
            min_size: 0,
            max_size: 0,
            // One clip per request (GenerationOutput::Audio carries a single track).
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            // No voice/speaker surface — music, not TTS.
            audio_voices: vec![],
            audio_languages: LANGUAGES.to_vec(),
            // The edit modes the provider supports. Inpaint / Repaint / Extend ride ACE-Step's
            // `repaint` task (region regenerate + seamless stitch, sc-12847) on the turbo DiT. Cover
            // (sc-13251) is the whole-clip restyle: it pulls the FSQ `audio_tokenizer` /
            // `audio_token_detokenizer` AND the non-distilled `transformer` (the reference's actual
            // cover DiT) from the sibling sft checkpoint, then restyles the source from a new prompt —
            // the sft DiT preserves the source's melodic content the distilled turbo DiT does not.
            audio_edit_modes: vec![
                AudioEditMode::Inpaint,
                AudioEditMode::Repaint,
                AudioEditMode::Extend,
                AudioEditMode::Cover,
            ],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
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
    // Pure audio: width/height are unused, so the descriptor advertises no size bounds (sc-13314)
    // and the audio floor skips the size range entirely.
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
            // Cover (sc-13251) is a whole-clip restyle: no region is required (the source-empty
            // and native-rate checks above already gate it); the FSQ round-trip + turbo-DiT restyle
            // run over the entire clip.
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
    /// The optional sft Cover snapshot dir (the [`COVER_COMPONENT_ID`] component), staged by the caller
    /// in [`LoadSpec::components`] when Cover is provisioned. `None` ⇒ text2music + the region edit
    /// modes still load and run; a Cover request then errors actionably (epic 13657 — no self-fetch).
    sft_cover_root: Option<PathBuf>,
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

    /// The staged sft Cover snapshot dir (the [`COVER_COMPONENT_ID`] component), or a caller-actionable
    /// error naming the exact `with_component` call. This is the [`LoadSpec::components`] analogue of
    /// the removed `ACESTEP_SFT_SNAPSHOT` production env read: the Cover snapshot now flows through the
    /// component seam (epic 13657), so production code holds no env-var side channel.
    fn cover_snapshot_root(&self) -> gen_core::Result<&Path> {
        self.sft_cover_root.as_deref().ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID} Cover needs the sft cover checkpoint staged as the '{COVER_COMPONENT_ID}' \
                 component — provision an ACE-Step/acestep-v15-xl-sft-diffusers snapshot \
                 (audio_tokenizer/, audio_token_detokenizer/, transformer/) and pass it in \
                 LoadSpec::components (with_component(\"{COVER_COMPONENT_ID}\", WeightsSource::Dir(...))); \
                 inference does not self-fetch it"
            ))
        })
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
    // Cover's ~7.8 GB sft snapshot is an OPTIONAL, on-demand component (epic 13657/13678): only a
    // Cover request needs it, so it is deliberately NOT a `required_components` id — text2music and
    // the region edit modes load and run without it (mirroring LTX's optional `uncensored_enhancer`).
    // When provisioned it arrives as the `sft_cover` Dir component; a Cover request without it errors
    // at generate rather than self-fetching. Reject any other stray component key as a caller mistake.
    gen_core::reject_unknown_components(spec, &[COVER_COMPONENT_ID], MODEL_ID)?;
    let sft_cover_root = match spec.components.get(COVER_COMPONENT_ID) {
        Some(WeightsSource::Dir(p)) => Some(p.clone()),
        Some(WeightsSource::File(p)) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} component '{COVER_COMPONENT_ID}' must be the sft snapshot directory \
                 (audio_tokenizer/ + audio_token_detokenizer/ + transformer/), not the file {}",
                p.display()
            )));
        }
        None => None,
    };
    Ok(Box::new(AceStepGenerator {
        descriptor: descriptor(),
        root,
        sft_cover_root,
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
        // Fail fast on an unprovisioned Cover (epic 13657): a Cover request needs its sft snapshot
        // staged as the `sft_cover` component; error here — before the (heavy) base-pipeline load —
        // rather than after it, and never self-fetch.
        if let Some(edit) = req.audio_edit() {
            if edit.mode == AudioEditMode::Cover {
                self.cover_snapshot_root()?;
            }
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

        // Prompted source-audio editing: a request carrying an `AudioEdit` conditioning dispatches
        // to the edit path — the mask-conditioned region tasks (Inpaint/Repaint/Extend, sc-12847)
        // or the whole-clip Cover restyle (sc-13251); otherwise this is plain text-to-music.
        let samples = if let Some(edit) = req.audio_edit() {
            let source = edit.audio.samples.clone();
            let source_channels = edit.audio.channels as usize;
            match edit.mode {
                AudioEditMode::Cover => {
                    // Cover pulls the staged sft FSQ modules AND the non-distilled sft cover DiT (both
                    // lazily loaded + cached) from the caller-provisioned `sft_cover` component dir and
                    // restyles the whole clip from the new prompt. The component's presence was already
                    // checked above (fail-fast), so this resolves the path it staged.
                    let sft_root = self.cover_snapshot_root()?;
                    let (tok_w, tok_c, det_w, det_c) = cover_module_paths(sft_root);
                    let cover = pipeline
                        .cover_modules(&tok_w, &tok_c, &det_w, &det_c)
                        .map_err(gen_core::Error::from)?;
                    let dit_shards = cover_dit_shards(sft_root).map_err(gen_core::Error::from)?;
                    let cover_dit = pipeline
                        .cover_dit(&dit_shards)
                        .map_err(gen_core::Error::from)?;
                    pipeline
                        .cover(
                            &req.prompt,
                            &source,
                            source_channels,
                            &params,
                            &cover,
                            &cover_dit,
                            &mut progress,
                            &probe,
                        )
                        .map_err(gen_core::Error::from)?
                }
                region_mode => {
                    let task = match region_mode {
                        AudioEditMode::Inpaint => EditTask::Inpaint,
                        AudioEditMode::Repaint => EditTask::Repaint,
                        AudioEditMode::Extend => EditTask::Extend,
                        AudioEditMode::Cover => unreachable!("cover handled above"),
                    };
                    let (region_start_secs, region_end_secs) = edit
                        .region
                        .map(|r| (r.start_secs, r.end_secs))
                        .unwrap_or((0.0, None));
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
                }
            }
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

/// Relative paths of the Cover FSQ component files inside the sft snapshot.
const COVER_TOKENIZER_CONFIG: &str = "audio_tokenizer/config.json";
const COVER_TOKENIZER_WEIGHTS: &str = "audio_tokenizer/diffusion_pytorch_model.safetensors";
const COVER_DETOKENIZER_CONFIG: &str = "audio_token_detokenizer/config.json";
const COVER_DETOKENIZER_WEIGHTS: &str =
    "audio_token_detokenizer/diffusion_pytorch_model.safetensors";

/// Paths of the Cover FSQ module files (sc-13251) inside a staged sft snapshot `root` — the sft
/// `audio_tokenizer` + `audio_token_detokenizer` component files, returned as
/// `(tok_weights, tok_config, det_weights, det_config)`. Pure path joins: `root` is the
/// caller-provisioned [`COVER_COMPONENT_ID`] component dir (epic 13657); inference never self-fetches
/// it or derives an HF-cache location.
pub fn cover_module_paths(root: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    (
        root.join(COVER_TOKENIZER_WEIGHTS),
        root.join(COVER_TOKENIZER_CONFIG),
        root.join(COVER_DETOKENIZER_WEIGHTS),
        root.join(COVER_DETOKENIZER_CONFIG),
    )
}

/// Distinct, sorted shard filenames listed in a `*.safetensors.index.json` `weight_map`.
fn shard_names_from_index(index_path: &Path) -> AudioResult<Vec<String>> {
    let text = std::fs::read_to_string(index_path)
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
            "{}: index lists no shards",
            index_path.display()
        )));
    }
    Ok(shards)
}

/// The **sft** cover DiT transformer shard paths (sc-13251) — the non-distilled reference cover model,
/// ~7.8 GB — under a staged sft snapshot `root`, returned for lazy loading on the first Cover request.
/// `root` is the caller-provisioned [`COVER_COMPONENT_ID`] component dir (epic 13657); inference never
/// self-fetches it or derives an HF-cache location.
pub fn cover_dit_shards(root: &Path) -> AudioResult<Vec<PathBuf>> {
    let dir = root.join("transformer");
    let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
    Ok(shard_names_from_index(&index)?
        .into_iter()
        .map(|s| dir.join(s))
        .collect())
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
        // All supported modes, in order; Cover (sc-13251) is now advertised (FSQ round-trip).
        assert_eq!(
            d.capabilities.audio_edit_modes,
            vec![
                AudioEditMode::Inpaint,
                AudioEditMode::Repaint,
                AudioEditMode::Extend,
                AudioEditMode::Cover,
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
        // Cover (sc-13251) is advertised and needs no region — a well-formed source clip validates.
        assert!(validate_request(&d, &edit_req(AudioEditMode::Cover, None)).is_ok());
        // Cover still enforces the native sample rate on the source clip.
        let mut wrong_rate_cover = edit_req(AudioEditMode::Cover, None);
        if let gen_core::Conditioning::AudioEdit { audio, .. } =
            &mut wrong_rate_cover.conditioning[0]
        {
            audio.sample_rate = 44_100;
        }
        assert!(matches!(
            validate_request(&d, &wrong_rate_cover).unwrap_err(),
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

    #[test]
    fn load_captures_the_optional_sft_cover_component() {
        let dir = std::env::temp_dir();
        // No component ⇒ loads fine (text2music / region-edit path); the Cover snapshot stays absent.
        assert!(load(&LoadSpec::new(WeightsSource::Dir(dir.clone()))).is_ok());
        // A staged `sft_cover` Dir is accepted (path captured, no I/O at load).
        let staged = LoadSpec::new(WeightsSource::Dir(dir.clone()))
            .with_component(COVER_COMPONENT_ID, WeightsSource::Dir(dir.clone()));
        assert!(load(&staged).is_ok());
        // An unrecognized component key is a typed Unsupported (caller mistake), not silently ignored.
        let unknown = LoadSpec::new(WeightsSource::Dir(dir.clone()))
            .with_component("not_a_real_component", WeightsSource::Dir(dir.clone()));
        assert!(matches!(
            load(&unknown),
            Err(gen_core::Error::Unsupported(_))
        ));
        // `sft_cover` must be a directory (the snapshot), not a single file.
        let as_file = LoadSpec::new(WeightsSource::Dir(dir.clone())).with_component(
            COVER_COMPONENT_ID,
            WeightsSource::File(dir.join("x.safetensors")),
        );
        assert!(matches!(load(&as_file), Err(gen_core::Error::Msg(_))));
    }

    #[test]
    fn cover_request_without_the_sft_component_errors_before_any_weight_load() {
        // A Cover request on a generator with no `sft_cover` staged fails fast with an actionable
        // message (naming the component) — before the base pipeline is built, and never self-fetches.
        // `root` is an empty temp dir with no real weights, so reaching the base load would panic/err;
        // the fail-fast guard means we never get there.
        let dir = std::env::temp_dir().join("acestep-cover-no-component");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let err = g
            .generate(&edit_req(AudioEditMode::Cover, None), &mut |_| {})
            .unwrap_err();
        match err {
            gen_core::Error::Msg(m) => {
                assert!(m.contains(COVER_COMPONENT_ID), "names the component: {m}");
                assert!(
                    m.contains("does not self-fetch"),
                    "states no self-fetch: {m}"
                );
            }
            other => panic!("expected an actionable Msg naming the component, got {other:?}"),
        }
    }
}
