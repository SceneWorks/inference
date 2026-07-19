//! `MossSfxGenerator` — the [`gen_core::Generator`] implementation for **MOSS-SoundEffect v2.0**
//! on the candle audio lane (sc-12841), plus its [`descriptor`]/[`load`] entry points and the
//! explicit registration constant wired into `candle-audio-catalog` under the id
//! **`moss_sfx_v2`** — the audio lane's first diffusion (SFX / ambience) provider.
//!
//! ## Snapshot layout
//!
//! [`load`] expects an `OpenMOSS-Team/MOSS-SoundEffect-v2.0`-shaped diffusers snapshot dir:
//!
//! ```text
//!   model_index.json                                  → pipeline identity + output surface
//!   scheduler/scheduler_config.json                   → flow-match schedule
//!   transformer/config.json + diffusion_pytorch_model.safetensors   → the 1.3B audio DiT
//!   text_encoder/config.json + model-*.safetensors + index          → Qwen3-1.7B
//!   tokenizer/tokenizer.json                          → the Qwen tokenizer
//!   vae/vae_128d_48k.pth                              → the continuous DAC VAE
//! ```
//!
//! [`resolve_pinned_snapshot`] materializes exactly that layout through the audio lane's
//! pinned-SHA hub path (`candle_audio::hub`, F-029 — never a mutable ref).
//!
//! ## Request mapping
//!
//! `prompt` is the sound description (English or Chinese; text only — no G2P front-end).
//! [`gen_core::AudioParams::target_duration`] selects the output duration (default 10 s,
//! 0.1 s granularity, ≤ 30 s); `seed` / `steps` (default 100) / `guidance` (CFG scale,
//! default 4.0) / `scheduler_shift` (flow shift, default 5.0) / `negative_prompt` map onto the
//! reference sampler knobs. Progress is one `Step` per solver step plus `Decoding` before the
//! VAE decode; cancellation is checked before generate, at every solver step, between DiT
//! blocks, AND inside the DAC decode stages, returning the typed [`gen_core::Error::Canceled`].
//! Determinism: same request + seed ⇒ byte-identical samples.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, WeightsSource,
};
use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
use candle_audio::{AudioError, Result as AudioResult};

use crate::pipeline::{
    MossSfxPipeline, SynthesisParams, DEFAULT_CFG_SCALE, DEFAULT_SECONDS, DEFAULT_STEPS,
};

/// Registry id (the SceneWorks worker routes `payload.model` to this exact id).
pub const MODEL_ID: &str = "moss_sfx_v2";

/// Hub pin: `OpenMOSS-Team/MOSS-SoundEffect-v2.0` at an immutable commit SHA (F-029;
/// Apache-2.0 weights + code, no commercial restriction).
pub const HUB_REPO: &str = "OpenMOSS-Team/MOSS-SoundEffect-v2.0";
pub const HUB_REVISION: &str = "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d";

/// Native output sample rate (Hz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Longest clip the model synthesizes (the trained 30 s latent window).
pub const MAX_DURATION_SECS: f32 = 30.0;

/// Largest solver step count accepted — one step per training timestep; a finer ladder than
/// the 1000-timestep training grid adds cost without resolution.
pub const MAX_STEPS: u32 = 1000;

/// CFG guidance bounds: 1.0 turns guidance off (single forward per step); values far above the
/// reference default 4.0 over-saturate flow-matching CFG, so the advertised ceiling stays
/// generous-but-sane.
pub const GUIDANCE_RANGE: (f32, f32) = (1.0, 20.0);

/// Prompt languages the model was trained on (bilingual English / Chinese; free-text prompts —
/// the language code is advisory, not a model switch).
pub const LANGUAGES: &[&str] = &["en", "zh"];

/// MOSS-SoundEffect's identity + capabilities — constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "moss_soundeffect",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            // CFG with a real negative-prompt branch (the reference `negative_prompt` +
            // `cfg_scale` pair).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // The native flow-match Euler integrator is the only sampler; no selectable
            // sampler/scheduler surface is advertised (an explicit request is a typed
            // Unsupported via the shared floor).
            samplers: vec![],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            // Audio models skip the size floor (validate_request_audio); these bounds are the
            // audio-lane convention for a size-less descriptor.
            min_size: 1,
            max_size: 4096,
            // One clip per request (GenerationOutput::Audio carries a single track).
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            // No voice surface — this is SFX/ambience, not TTS; an explicit `audio.voice`
            // is rejected by the shared floor as Unsupported.
            audio_voices: vec![],
            audio_languages: LANGUAGES.to_vec(),
            audio_edit_modes: vec![],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            max_speakers: None,
        },
    }
}

/// Capability-driven request validation, factored out for weightless unit tests. Shared floor
/// checks ([`Capabilities::validate_request_audio`]) plus this model's own bounds.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt (the sound description) must not be empty"
        )));
    }
    // The audio floor deliberately skips the size checks (width/height are unused by a pure
    // audio model), but the descriptor still advertises bounds — honor them so validate never
    // accepts a request outside the advertised surface (validate-honesty).
    let caps = &desc.capabilities;
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
                "{id}: steps {steps} above the {MAX_STEPS}-step ceiling (the 1000-timestep \
                 flow-matching training grid)"
            )));
        }
    }
    if let Some(g) = req.guidance {
        if !(GUIDANCE_RANGE.0..=GUIDANCE_RANGE.1).contains(&g) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: guidance (CFG scale) {g} outside {:?} (1.0 disables CFG; the reference \
                 default is {DEFAULT_CFG_SCALE})",
                GUIDANCE_RANGE
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
        if let Some(d) = audio.target_duration {
            // The floor already enforces (0, 30]; the model additionally needs ≥ 0.1 s so the
            // 0.1 s duration rounding cannot collapse the request to zero output.
            if d < 0.1 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: audio.target_duration {d}s below the 0.1 s floor (duration is \
                     conditioned at 0.1 s granularity)"
                )));
            }
        }
    }
    caps.validate_request_audio(id, req)
}

/// A loaded (lazy) MOSS-SoundEffect generator. The heavy pipeline (Qwen3 + DiT + VAE, ~13 GB
/// resident in f32) is built on first use and cached; `load` does no file I/O beyond argument
/// checks (the sibling providers' lazy-load discipline).
pub struct MossSfxGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<MossSfxPipeline>>>,
}

impl MossSfxGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<MossSfxPipeline>> {
        let mut guard = lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let device = candle_audio::default_device()?;
        let built = Arc::new(MossSfxPipeline::from_snapshot(&self.root, &device)?);
        *guard = Some(built.clone());
        Ok(built)
    }
}

/// Recover a poisoned mutex (a prior panic mid-build leaves `None`/stale state, which the lazy
/// builder tolerates) — the audio twin of `candle_gen::lock_recover`.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Construct the (lazy) generator from a [`LoadSpec`]. `spec.weights` must be a snapshot
/// directory (see module docs); adapters/quantization/control overlays are rejected — refusing
/// is more honest than silently dropping.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a diffusers snapshot directory (model_index.json + \
                 transformer/ + text_encoder/ + tokenizer/ + vae/), not a single file"
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
    Ok(Box::new(MossSfxGenerator {
        descriptor: descriptor(),
        root,
        pipeline: Mutex::new(None),
    }))
}

impl Generator for MossSfxGenerator {
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
        // Pre-generate cancellation seam (sc-11128 class): consult the flag before ANY heavy
        // work — pipeline build, text encode, and denoise all come after this.
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let audio = req.audio.clone().unwrap_or_default();
        let params = SynthesisParams {
            seconds: audio.target_duration.unwrap_or(DEFAULT_SECONDS),
            steps: req.steps.unwrap_or(DEFAULT_STEPS as u32) as usize,
            cfg_scale: req.guidance.unwrap_or(DEFAULT_CFG_SCALE),
            sigma_shift: req.scheduler_shift.map(|s| s as f64),
            negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
        };

        let pipeline = self.pipeline()?;
        let total = params.steps as u32;
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let mut progress = |p: crate::pipeline::PipelineProgress| match p {
            crate::pipeline::PipelineProgress::Step(k) => on_progress(Progress::Step {
                current: k as u32,
                total,
            }),
            crate::pipeline::PipelineProgress::Decoding => on_progress(Progress::Decoding),
        };
        let samples = pipeline
            .synthesize(&req.prompt, &params, &mut progress, &probe)
            .map_err(gen_core::Error::from)?;

        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: 1,
            ..Default::default()
        }))
    }
}

// Explicit catalog registration for `moss_sfx_v2` (composed by `candle-audio-catalog`).
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Materialize the pinned MOSS-SoundEffect snapshot through the audio lane's F-029 hub path:
/// `model_index.json` (the snapshot-dir probe), the component configs, the DiT + text-encoder
/// safetensors (shards enumerated from the index), the tokenizer, and the VAE checkpoint — all
/// at [`HUB_REVISION`], landing in the ordinary HF cache. Returns the snapshot dir as a
/// [`WeightsSource::Dir`] ready for a [`LoadSpec`].
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let dir = pinned_snapshot_dir(HUB_REPO, HUB_REVISION, "model_index.json")?;
    for file in [
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        &format!("vae/{}", crate::vae::VAE_FILE),
    ] {
        hf_get_pinned(HUB_REPO, HUB_REVISION, file)?;
    }
    // Text-encoder shards, enumerated from the index so a re-sharded upstream layout cannot
    // silently skip a file.
    let index_path = hf_get_pinned(
        HUB_REPO,
        HUB_REVISION,
        "text_encoder/model.safetensors.index.json",
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
            "{HUB_REPO}: text_encoder/model.safetensors.index.json lists no shards"
        )));
    }
    for shard in shards {
        hf_get_pinned(HUB_REPO, HUB_REVISION, &format!("text_encoder/{shard}"))?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag};

    fn audio_req(audio: AudioParams) -> GenerationRequest {
        GenerationRequest {
            prompt: "glass shattering on a stone floor".into(),
            audio: Some(audio),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_sfx_surface() {
        let d = descriptor();
        assert_eq!(d.id, "moss_sfx_v2");
        assert_eq!(d.family, "moss_soundeffect");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [48_000]);
        assert_eq!(d.capabilities.max_audio_duration_secs, Some(30.0));
        assert!(d.capabilities.audio_voices.is_empty(), "SFX has no voices");
        assert_eq!(d.capabilities.audio_languages, ["en", "zh"]);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert_eq!(d.capabilities.max_count, 1);
    }

    #[test]
    fn validate_gates_the_sampling_surface() {
        let d = descriptor();
        // In-surface request passes (duration + rate + language + CFG knobs).
        let mut ok = audio_req(AudioParams {
            target_duration: Some(4.0),
            sample_rate: Some(48_000),
            language: Some("en".into()),
            ..Default::default()
        });
        ok.steps = Some(50);
        ok.guidance = Some(4.0);
        ok.negative_prompt = Some("muffled, low quality".into());
        ok.scheduler_shift = Some(5.0);
        assert!(validate_request(&d, &ok).is_ok());

        // Out-of-surface values are rejected.
        for bad_audio in [
            AudioParams {
                target_duration: Some(MAX_DURATION_SECS + 1.0),
                ..Default::default()
            },
            AudioParams {
                target_duration: Some(0.01), // below the 0.1 s conditioning granularity
                ..Default::default()
            },
            AudioParams {
                sample_rate: Some(44_100),
                ..Default::default()
            },
            AudioParams {
                voice: Some("af_heart".into()), // no voice surface on an SFX model
                ..Default::default()
            },
            AudioParams {
                language: Some("ja".into()),
                ..Default::default()
            },
        ] {
            assert!(
                validate_request(&d, &audio_req(bad_audio.clone())).is_err(),
                "{bad_audio:?} must be rejected"
            );
        }
        // Sampling knobs outside the advertised ranges are rejected.
        let mut r = audio_req(AudioParams::default());
        r.steps = Some(MAX_STEPS + 1);
        assert!(validate_request(&d, &r).is_err());
        let mut r = audio_req(AudioParams::default());
        r.guidance = Some(0.5);
        assert!(validate_request(&d, &r).is_err());
        let mut r = audio_req(AudioParams::default());
        r.guidance = Some(21.0);
        assert!(validate_request(&d, &r).is_err());
        let mut r = audio_req(AudioParams::default());
        r.scheduler_shift = Some(0.0);
        assert!(validate_request(&d, &r).is_err());
        // An explicit sampler name: no sampler surface is advertised → typed Unsupported.
        let mut r = audio_req(AudioParams::default());
        r.sampler = Some("euler".into());
        assert!(matches!(
            validate_request(&d, &r),
            Err(gen_core::Error::Unsupported(_))
        ));
        // Empty prompt is rejected.
        let mut r = audio_req(AudioParams::default());
        r.prompt = "  ".into();
        assert!(validate_request(&d, &r).is_err());
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
        let dir = std::env::temp_dir().join("moss-sfx-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = GenerationRequest {
            prompt: "rain".into(),
            cancel: flag,
            ..Default::default()
        };
        // The pre-generate seam fires before the pipeline build — typed Canceled, even though
        // this snapshot dir has no weights at all.
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }

    #[test]
    fn generate_on_a_missing_snapshot_fails_cleanly() {
        // A generator over an empty dir: generate must error (no weights), never panic.
        let dir = std::env::temp_dir().join("moss-sfx-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let req = GenerationRequest {
            prompt: "rain".into(),
            ..Default::default()
        };
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(!matches!(err, gen_core::Error::Canceled));
    }
}
