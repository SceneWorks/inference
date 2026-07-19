//! `ChatterboxGenerator` — the [`gen_core::Generator`] adapter for Chatterbox clone-TTS on the
//! candle audio lane (sc-13222), its [`descriptor`]/[`load`] entry points, the conditioning
//! mapping, and the [`REGISTRATION`] constant (id **`chatterbox_tts`**).
//!
//! ## Conditioning mapping (the two voice-conditioning paths)
//!
//! Chatterbox clones a voice from two distinct conditioning artifacts, and this provider maps the
//! gen-core conditioning inputs onto them faithfully:
//!
//! - [`Conditioning::VoiceEmbedding`] — the raw 256-d `chatterbox_ve` speaker vector (sc-12838).
//!   It drives **T3's** speaker conditioning directly (`T3CondEnc.spkr_enc`). It is *sufficient for
//!   T3* but **not** for S3Gen: S3Gen additionally needs a reference **mel**, reference **speech
//!   tokens**, and a **CAMPPlus x-vector**, none of which are recoverable from the 256-d vector —
//!   so a VoiceEmbedding-only request can drive the LM but not (once ported) the full S3Gen
//!   reference. This is the "Chatterbox needs MORE than the 256-d ve vector" case the story calls
//!   out.
//! - [`Conditioning::ReferenceAudio`] — the raw reference clip. The provider derives the 256-d
//!   speaker vector from it *inside the provider* via the merged `chatterbox_ve` embedder, and (once
//!   the S3Gen stack lands) will derive the prompt mel + prompt speech tokens + CAMPPlus x-vector
//!   from the same clip. ReferenceAudio is therefore the fuller conditioning path.
//!
//! ## Port status (honest partial — see [`crate::s3gen`])
//!
//! This slice ports the **T3 speech-token LM** (real `t3_cfg.safetensors` weights, CFG decode) and
//! the full provider/conditioning surface. The **S3Gen** token→waveform stack (four networks) is
//! not yet ported; the generator's `generate()` runs T3 to produce real speech tokens and then
//! returns a typed error at the S3Gen boundary rather than fabricate audio.

use std::sync::Mutex;

use candle_audio::candle_core::DType;
use candle_audio::gen_core::{
    self, AudioTrack, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Progress, VoiceEmbedder,
    WeightsSource,
};
use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
use candle_audio::Result as AudioResult;
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{GenerationDefaults, S3GenConfig, T3Config, S3GEN_SR};
use crate::t3::{strip_special_speech_tokens, T3Cond, T3};
use crate::text::EnTokenizer;

/// Registry id (the SceneWorks worker routes `payload.model` to this exact id).
pub const MODEL_ID: &str = "chatterbox_tts";

/// Hub pin: `ResembleAI/chatterbox` at the same immutable commit the `chatterbox_ve` sibling pins
/// (F-029; MIT weights — commercial use OK).
pub const HUB_REPO: &str = "ResembleAI/chatterbox";
pub const HUB_REVISION: &str = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";

/// The T3 LM checkpoint filename inside a snapshot.
pub const T3_WEIGHTS_FILE: &str = "t3_cfg.safetensors";
/// The text tokenizer filename inside a snapshot.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// Advertised language codes (the base English model).
pub const LANGUAGES: &[&str] = &["en", "en-us"];

/// Longest clip advertised (seconds).
pub const MAX_DURATION_SECS: f32 = 30.0;

/// Chatterbox clone-TTS identity + capabilities — constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "chatterbox",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // The two voice-cloning conditioning paths (see module docs).
            conditioning: vec![
                ConditioningKind::VoiceEmbedding,
                ConditioningKind::ReferenceAudio,
            ],
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
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![S3GEN_SR],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            // The voice is supplied by conditioning, not a named voice id.
            audio_voices: vec![],
            audio_languages: LANGUAGES.to_vec(),
            audio_edit_modes: vec![],
            supported_quants: &[],
            supports_kv_cache: true,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            max_speakers: None,
        },
    }
}

/// Capability-driven request validation (factored out for weightless unit tests): non-empty
/// prompt, exactly the advertised conditioning surface, at least one voice-conditioning input, and
/// the shared audio floor.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt (the text to speak) must not be empty"
        )));
    }
    // Pure audio: width/height are unused, so the descriptor advertises no size bounds (sc-13314)
    // and the audio floor skips the size range entirely.
    let caps = &desc.capabilities;
    caps.validate_request_audio(id, req)?;
    // A clone TTS needs a voice: exactly one of VoiceEmbedding / ReferenceAudio.
    let has_voice = req.conditioning.iter().any(|c| {
        matches!(
            c.kind(),
            ConditioningKind::VoiceEmbedding | ConditioningKind::ReferenceAudio
        )
    });
    if !has_voice {
        return Err(gen_core::Error::Msg(format!(
            "{id}: a voice is required — supply Conditioning::VoiceEmbedding (a chatterbox_ve \
             vector) or Conditioning::ReferenceAudio (a reference clip)"
        )));
    }
    Ok(())
}

/// A loaded (lazy) Chatterbox generator. The T3 LM (and, for ReferenceAudio conditioning, the
/// voice embedder) are built on first use.
pub struct ChatterboxGenerator {
    descriptor: ModelDescriptor,
    t3_config: T3Config,
    #[allow(dead_code)] // consumed once the S3Gen stack lands (see crate::s3gen).
    s3gen_config: S3GenConfig,
    root: std::path::PathBuf,
    t3: Mutex<Option<T3>>,
    tokenizer: Mutex<Option<EnTokenizer>>,
    embedder: Mutex<Option<Box<dyn VoiceEmbedder>>>,
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl ChatterboxGenerator {
    fn tokenizer_path(&self) -> std::path::PathBuf {
        self.root.join(TOKENIZER_FILE)
    }

    fn t3_path(&self) -> std::path::PathBuf {
        self.root.join(T3_WEIGHTS_FILE)
    }

    /// The 256-d speaker vector for a request's conditioning: a supplied [`Conditioning::VoiceEmbedding`]
    /// used directly, else derived from [`Conditioning::ReferenceAudio`] via the `chatterbox_ve`
    /// embedder inside the provider.
    fn speaker_embedding(&self, req: &GenerationRequest) -> gen_core::Result<Vec<f32>> {
        // Prefer an explicit voice embedding (the sc-12838 path).
        for c in &req.conditioning {
            if let Conditioning::VoiceEmbedding { embedding, .. } = c {
                if embedding.len() != self.t3_config.speaker_embed_size {
                    return Err(gen_core::Error::Msg(format!(
                        "{MODEL_ID}: VoiceEmbedding has {} dims, expected {} (a chatterbox_ve vector)",
                        embedding.len(),
                        self.t3_config.speaker_embed_size
                    )));
                }
                return Ok(embedding.clone());
            }
        }
        // Otherwise derive it from the reference clip through the merged voice embedder.
        for c in &req.conditioning {
            if let Conditioning::ReferenceAudio { audio, .. } = c {
                let emb = self.embed_reference(audio)?;
                return Ok(emb);
            }
        }
        Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: no voice conditioning present (validate() should have caught this)"
        )))
    }

    fn embed_reference(&self, audio: &AudioTrack) -> gen_core::Result<Vec<f32>> {
        let mut guard = lock_recover(&self.embedder);
        if guard.is_none() {
            let weights = candle_audio_chatterbox_ve::resolve_pinned_file().map_err(|e| {
                gen_core::Error::Msg(format!("{MODEL_ID}: resolve chatterbox_ve weights: {e}"))
            })?;
            let embedder = candle_audio_chatterbox_ve::load(&LoadSpec::new(weights))?;
            *guard = Some(embedder);
        }
        guard.as_ref().unwrap().embed(audio)
    }

    fn tokenizer(&self) -> gen_core::Result<()> {
        let mut guard = lock_recover(&self.tokenizer);
        if guard.is_none() {
            let tok = EnTokenizer::from_file(
                &self.tokenizer_path(),
                self.t3_config.start_text_token,
                self.t3_config.stop_text_token,
            )?;
            *guard = Some(tok);
        }
        Ok(())
    }

    fn ensure_t3(&self) -> gen_core::Result<()> {
        let mut guard = lock_recover(&self.t3);
        if guard.is_none() {
            let path = self.t3_path();
            if !path.is_file() {
                return Err(gen_core::Error::Msg(format!(
                    "{MODEL_ID}: T3 weights {} missing (resolve_pinned_snapshot materializes {T3_WEIGHTS_FILE})",
                    path.display()
                )));
            }
            let device = candle_audio::default_device()?;
            // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the shared idiom.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&path),
                    DType::F32,
                    &device,
                )
                .map_err(|e| {
                    gen_core::Error::Msg(format!("{MODEL_ID}: load {T3_WEIGHTS_FILE}: {e}"))
                })?
            };
            let t3 = T3::new(&self.t3_config, vb)
                .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: build T3: {e}")))?;
            *guard = Some(t3);
        }
        Ok(())
    }
}

impl ChatterboxGenerator {
    /// Run the **T3 stage** end-to-end: map the request conditioning to the T3 speaker vector,
    /// tokenize the prompt, and autoregressively decode speech tokens on the real Llama-520M LM.
    /// Returns `(raw_tokens, real_tokens)` where `real_tokens` has the special/BOS/EOS speech
    /// tokens stripped (the sequence S3Gen would consume). This is the ported half of the pipeline
    /// and the surface the conformance test asserts against directly.
    pub fn speech_tokens(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<(Vec<u32>, Vec<u32>)> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let defaults = GenerationDefaults::default();

        // 1. Conditioning → the T3 speaker vector (+ empty prompt tokens until s3tokenizer lands).
        let speaker_emb = self.speaker_embedding(req)?;
        let cond = T3Cond {
            speaker_emb,
            cond_prompt_speech_tokens: Vec::new(),
            emotion_adv: defaults.exaggeration,
        };

        // 2. Text → tokens.
        self.tokenizer()?;
        let text_tokens = {
            let guard = lock_recover(&self.tokenizer);
            guard.as_ref().unwrap().text_to_tokens(&req.prompt)?
        };

        // 3. T3 autoregressive decode (real weights) → speech tokens.
        self.ensure_t3()?;
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let mut rng = StdRng::seed_from_u64(seed);
        let max_new = self.t3_config.max_speech_tokens;
        let cancel = req.cancel.clone();
        let cancel_fn = move || cancel.is_cancelled();

        let mut guard = lock_recover(&self.t3);
        let t3 = guard.as_mut().unwrap();
        let mut on_step = |step: usize| {
            on_progress(Progress::Step {
                current: step as u32,
                total: max_new as u32,
            });
        };
        let out = t3
            .inference(
                &cond,
                &text_tokens,
                defaults.cfg_weight,
                defaults.temperature,
                defaults.top_p,
                defaults.min_p,
                defaults.repetition_penalty,
                max_new,
                &mut rng,
                &mut on_step,
                &cancel_fn,
            )
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: T3 decode: {e}")))?;
        match out {
            Some(raw) => {
                let real = strip_special_speech_tokens(&raw);
                Ok((raw, real))
            }
            None => Err(gen_core::Error::Canceled),
        }
    }
}

impl Generator for ChatterboxGenerator {
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
        // T3 stage (real weights) → speech tokens.
        let (raw_tokens, real_tokens) = self.speech_tokens(req, on_progress)?;

        // S3Gen token→waveform. Announce the decode phase, then hit the honest boundary: the T3
        // stage produced real speech tokens; the S3Gen stack is not yet ported (crate::s3gen).
        on_progress(Progress::Decoding);
        match crate::s3gen::decode(&real_tokens) {
            Ok(samples) => Ok(GenerationOutput::Audio(AudioTrack {
                samples,
                sample_rate: S3GEN_SR,
                channels: 1,
                ..Default::default()
            })),
            Err(e) => Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: T3 produced {} speech tokens ({} after dropping specials), but {e}",
                raw_tokens.len(),
                real_tokens.len()
            ))),
        }
    }
}

/// Construct the (lazy) Chatterbox generator from a [`LoadSpec`], returning the **concrete** type
/// (so callers — e.g. the conformance test — can reach [`ChatterboxGenerator::speech_tokens`]).
/// [`load`] wraps this behind the `dyn Generator` trait object.
pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<ChatterboxGenerator> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a snapshot directory ({T3_WEIGHTS_FILE} + {} + {TOKENIZER_FILE}), not a single file",
                crate::s3gen::S3GEN_WEIGHTS_FILE
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
    Ok(ChatterboxGenerator {
        descriptor: descriptor(),
        t3_config: T3Config::LLAMA_520M,
        s3gen_config: S3GenConfig::DEFAULT,
        root,
        t3: Mutex::new(None),
        tokenizer: Mutex::new(None),
        embedder: Mutex::new(None),
    })
}

/// Construct the (lazy) Chatterbox generator as a boxed [`Generator`] trait object.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

// Explicit registration for `chatterbox_tts` (see crate docs re: catalog wiring, which is gated on
// the S3Gen stack landing).
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Materialize the pinned Chatterbox snapshot through the audio lane's F-029 hub path: the T3
/// checkpoint, the S3Gen checkpoint, and the tokenizer, all at [`HUB_REVISION`]. Returns the
/// snapshot dir as a [`WeightsSource::Dir`].
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let dir = pinned_snapshot_dir(HUB_REPO, HUB_REVISION, T3_WEIGHTS_FILE)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, TOKENIZER_FILE)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, crate::s3gen::S3GEN_WEIGHTS_FILE)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag};

    fn req_with(conditioning: Vec<Conditioning>) -> GenerationRequest {
        GenerationRequest {
            prompt: "Hello there.".into(),
            audio: Some(AudioParams {
                language: Some("en".into()),
                sample_rate: Some(24_000),
                ..Default::default()
            }),
            conditioning,
            ..Default::default()
        }
    }

    fn ve_vec() -> Conditioning {
        Conditioning::VoiceEmbedding {
            embedding: vec![0.1; 256],
            strength: None,
        }
    }

    #[test]
    fn descriptor_advertises_the_clone_surface() {
        let d = descriptor();
        assert_eq!(d.id, "chatterbox_tts");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [24_000]);
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::VoiceEmbedding));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::ReferenceAudio));
        assert_eq!(d.capabilities.max_count, 1);
    }

    #[test]
    fn validate_requires_a_voice_and_nonempty_prompt() {
        let d = descriptor();
        // With a voice embedding + non-empty prompt: OK.
        assert!(validate_request(&d, &req_with(vec![ve_vec()])).is_ok());
        // No conditioning: rejected (a clone needs a voice).
        assert!(validate_request(&d, &req_with(vec![])).is_err());
        // Empty prompt: rejected.
        let mut r = req_with(vec![ve_vec()]);
        r.prompt = "   ".into();
        assert!(validate_request(&d, &r).is_err());
        // Unsupported sample rate: rejected by the audio floor.
        let mut r = req_with(vec![ve_vec()]);
        r.audio.as_mut().unwrap().sample_rate = Some(44_100);
        assert!(validate_request(&d, &r).is_err());
    }

    #[test]
    fn load_rejects_unsupported_spec_shapes() {
        let dir = std::env::temp_dir();
        assert!(load(&LoadSpec::new(WeightsSource::File(
            dir.join("x.safetensors")
        )))
        .is_err());
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir));
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir = std::env::temp_dir().join("chatterbox-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let mut req = req_with(vec![ve_vec()]);
        req.cancel = flag;
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }
}
