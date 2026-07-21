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
//!   speaker vector from it *inside the provider* via the merged `chatterbox_ve` embedder, and (in
//!   `generate()`) the S3Gen prompt mel + prompt speech tokens + CAMPPlus x-vector from the same
//!   clip ([`crate::s3gen::S3Gen`]). ReferenceAudio is therefore the fuller conditioning path — and
//!   the **only** one that can render a full clone WAV, since S3Gen's reference is not recoverable
//!   from a bare voice vector.
//!
//! ## Pipeline (sc-13239 — end-to-end)
//!
//! `generate()` runs the full clone: **T3** ([`crate::t3`]) decodes speech tokens from the text +
//! speaker conditioning, then **S3Gen** ([`crate::s3gen::S3Gen`]) renders those tokens into a 24 kHz
//! waveform in the reference voice (flow-matching token→mel + HiFTNet vocoder), and the **PerTh**
//! provenance watermark ([`crate::perth`]) is applied to the output — always, matching the reference
//! (no disable flag). A VoiceEmbedding-only request drives T3 but returns a typed error at S3Gen,
//! because the reference clip S3Gen needs is absent.

use std::sync::Mutex;

use candle_audio::candle_core::DType;
use candle_audio::gen_core::{
    self, reject_unknown_components, require_component, AudioTrack, Capabilities, Conditioning,
    ConditioningKind, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, VoiceEmbedder, WeightsSource,
};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{
    GenerationDefaults, T3Config, ENC_COND_LEN, S3GEN_SR, S3_SR, SPEECH_COND_PROMPT_LEN,
};
use crate::perth::PerthWatermarker;
use crate::s3gen::S3Gen;
use crate::s3tokenizer::S3Tokenizer;
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

/// The license of the pinned Chatterbox weight checkpoint (sc-13332) — surfaced for SceneWorks'
/// end-product licenses page. MIT (permissive), verified against the `ResembleAI/chatterbox`
/// model card. The clone TTS generator ships the same `ResembleAI/chatterbox` weights the
/// `chatterbox_ve` sibling does, keyed here by this provider's own [`MODEL_ID`].
pub const WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "MIT",
    name: "MIT License",
    source_url: "https://huggingface.co/ResembleAI/chatterbox",
    attribution: Some("Chatterbox © Resemble AI — licensed under MIT"),
    commercial_use: true,
    restriction: None,
};

/// This provider's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation.
pub const WEIGHT_LICENSE_ENTRY: gen_core::WeightLicenseEntry = gen_core::WeightLicenseEntry {
    provider_id: MODEL_ID,
    component: None,
    license: WEIGHT_LICENSE,
};

/// Advertised language codes (the base English model).
pub const LANGUAGES: &[&str] = &["en", "en-us"];

/// Longest clip advertised (seconds).
pub const MAX_DURATION_SECS: f32 = 30.0;

/// Component id (epic 13657) for the PerTh provenance watermarker weights
/// (`perth_implicit.safetensors`). The caller stages the resolved local path under this key in
/// [`LoadSpec::components`]; the generator no longer self-fetches it — `watermark()`
/// consumes the staged path lazily.
pub const COMPONENT_PERTH: &str = "perth";
/// Component id (epic 13657) for the `chatterbox_ve` voice-encoder weights (`ve.safetensors`). The
/// caller stages the resolved local path under this key in [`LoadSpec::components`]; the generator no
/// longer re-downloads it at render — `embed_reference()` consumes the staged
/// path lazily for `ReferenceAudio` conditioning.
pub const COMPONENT_VOICE_EMBEDDING: &str = "voice_embedding";
/// The named components a caller MUST stage in [`LoadSpec::components`] before [`load`] — validated
/// fail-fast at load via [`gen_core::require_component`], the descriptor advertising the same set.
pub const REQUIRED_COMPONENTS: &[&str] = &[COMPONENT_PERTH, COMPONENT_VOICE_EMBEDDING];

/// Chatterbox clone-TTS identity + capabilities — constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        required_components: REQUIRED_COMPONENTS,
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
    root: std::path::PathBuf,
    /// The caller-staged [`COMPONENT_PERTH`] weights (`perth_implicit.safetensors`), validated present
    /// at load (fail-fast) and consumed lazily by `watermark()` — no hub self-fetch (epic 13657).
    perth_weights: WeightsSource,
    /// The caller-staged [`COMPONENT_VOICE_EMBEDDING`] weights (`ve.safetensors`), validated present at
    /// load and consumed lazily by `embed_reference()` — no render-time re-download (epic 13657).
    voice_embedding_weights: WeightsSource,
    t3: Mutex<Option<T3>>,
    tokenizer: Mutex<Option<EnTokenizer>>,
    embedder: Mutex<Option<Box<dyn VoiceEmbedder>>>,
    /// The s3tokenizer (sc-13235), loaded lazily from the snapshot's `s3gen.safetensors` the first
    /// time a `ReferenceAudio` request needs the T3 conditioning prompt tokens.
    s3tokenizer: Mutex<Option<S3Tokenizer>>,
    /// The assembled S3Gen token→waveform stack (sc-13239), loaded lazily from `s3gen.safetensors`
    /// the first time `generate()` renders a clone.
    s3gen: Mutex<Option<S3Gen>>,
    /// The PerTh provenance watermarker (sc-13240/sc-13239), loaded lazily from the staged
    /// `perth_weights` component path.
    perth: Mutex<Option<PerthWatermarker>>,
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The `perth_implicit.safetensors` file from the staged [`COMPONENT_PERTH`] source: a
/// [`WeightsSource::File`] used directly, or `<dir>/perth_implicit.safetensors` for a
/// [`WeightsSource::Dir`]. ([`PerthWatermarker::from_safetensors`] itself turns a missing file into a
/// typed error.)
fn perth_component_file(src: &WeightsSource) -> std::path::PathBuf {
    match src {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(d) => d.join(crate::perth::PERTH_WEIGHTS_FILE),
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
            // Consume the caller-staged `voice_embedding` component path (validated present at load);
            // the chatterbox_ve `load` contract is unchanged — it still takes a `File` in `spec.weights`.
            let embedder = candle_audio_chatterbox_ve::load(&LoadSpec::new(
                self.voice_embedding_weights.clone(),
            ))?;
            *guard = Some(embedder);
        }
        guard.as_ref().unwrap().embed(audio)
    }

    /// The T3 conditioning prompt speech tokens for a reference clip (sc-13235): the s3tokenizer's
    /// 25 Hz codes over the first [`ENC_COND_LEN`] (6 s) of the clip, truncated to the T3
    /// `speech_cond_prompt_len` (150). Lazily loads the s3tokenizer from `s3gen.safetensors`.
    ///
    /// This is the port of the reference's `t3_cond_prompt_tokens` derivation — the prompt the
    /// Perceiver resampler consumes. It was empty in sc-13222 (weakening the voice conditioning);
    /// with the s3tokenizer ported it is filled from the reference clip.
    pub fn reference_speech_tokens(&self, audio: &AudioTrack) -> gen_core::Result<Vec<u32>> {
        let mut guard = lock_recover(&self.s3tokenizer);
        if guard.is_none() {
            let tok = S3Tokenizer::from_snapshot(&self.root)
                .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: load s3tokenizer: {e}")))?;
            *guard = Some(tok);
        }
        // Resample to 16 kHz first, THEN cap at ENC_COND_LEN — the cap is defined in 16 kHz
        // samples (6 s), so it must be applied post-resample, as the reference does.
        let wav16k = crate::s3tokenizer::resample_to_16k(&audio.samples, audio.sample_rate);
        let n = ENC_COND_LEN.min(wav16k.len());
        let codes = guard
            .as_ref()
            .unwrap()
            .encode(&wav16k[..n], S3_SR)
            .map_err(gen_core::Error::from)?;
        Ok(codes
            .into_iter()
            .take(SPEECH_COND_PROMPT_LEN)
            .map(|c| c as u32)
            .collect())
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
                    "{MODEL_ID}: T3 weights {} missing (the passed-in snapshot must supply {T3_WEIGHTS_FILE})",
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

    /// Lazily assemble the S3Gen token→waveform stack (s3tokenizer + CAMPPlus + flow + HiFTNet) from
    /// the snapshot's `s3gen.safetensors` (sc-13239).
    fn ensure_s3gen(&self) -> gen_core::Result<()> {
        let mut guard = lock_recover(&self.s3gen);
        if guard.is_none() {
            let s3gen = S3Gen::from_snapshot(&self.root)
                .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: load S3Gen: {e}")))?;
            *guard = Some(s3gen);
        }
        Ok(())
    }

    /// Apply the PerTh provenance watermark to a rendered 24 kHz clone (sc-13239). The watermarker is
    /// loaded lazily from the caller-staged [`COMPONENT_PERTH`] path (validated present at load — no
    /// hub self-fetch). The clone ALWAYS watermarks (the reference behavior — no disable flag), so a
    /// failure to load the staged weights is a typed error rather than a silently un-watermarked clone.
    fn watermark(&self, samples: &[f32]) -> gen_core::Result<Vec<f32>> {
        let mut guard = lock_recover(&self.perth);
        if guard.is_none() {
            let weights = perth_component_file(&self.perth_weights);
            let wm = PerthWatermarker::from_safetensors(&weights).map_err(|e| {
                gen_core::Error::Msg(format!("{MODEL_ID}: load PerTh watermarker: {e}"))
            })?;
            *guard = Some(wm);
        }
        guard
            .as_ref()
            .unwrap()
            .embed(samples, S3GEN_SR)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: apply PerTh watermark: {e}")))
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

        // 1. Conditioning → the T3 speaker vector, plus the s3tokenizer prompt tokens when a
        //    reference clip is present (sc-13235). A VoiceEmbedding-only request has no clip to
        //    tokenize, so the prompt stays empty — the reference's `cond_prompt_speech_emb is None`
        //    branch (a bare voice vector drives T3 without the Perceiver prompt).
        let speaker_emb = self.speaker_embedding(req)?;
        let cond_prompt_speech_tokens = match req.conditioning.iter().find_map(|c| match c {
            Conditioning::ReferenceAudio { audio, .. } => Some(audio),
            _ => None,
        }) {
            Some(audio) => self.reference_speech_tokens(audio)?,
            None => Vec::new(),
        };
        let cond = T3Cond {
            speaker_emb,
            cond_prompt_speech_tokens,
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
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }

        // S3Gen needs the reference CLIP: VoiceEmbedding alone conditions the T3 LM but cannot
        // supply S3Gen's reference mel / prompt tokens / speaker x-vector, so a full clone WAV
        // requires Conditioning::ReferenceAudio. This is the honest "Chatterbox needs MORE than the
        // 256-d ve vector" boundary — a VoiceEmbedding-only request drives T3 and stops here.
        let reference = req
            .conditioning
            .iter()
            .find_map(|c| match c {
                Conditioning::ReferenceAudio { audio, .. } => Some(audio),
                _ => None,
            })
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{MODEL_ID}: a full cloned WAV requires Conditioning::ReferenceAudio (the \
                     reference clip); VoiceEmbedding conditions the T3 LM but cannot supply S3Gen's \
                     reference mel / prompt tokens / speaker x-vector. T3 produced {} speech tokens \
                     ({} after dropping specials).",
                    raw_tokens.len(),
                    real_tokens.len()
                ))
            })?;

        // S3Gen token→waveform: derive the reference conditioning, run the flow-matching decoder,
        // and vocode with HiFTNet → a real 24 kHz cloned-voice waveform.
        on_progress(Progress::Decoding);
        self.ensure_s3gen()?;
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let cancel = req.cancel.clone();
        let should_cancel = move || cancel.is_cancelled();
        let rendered = {
            let guard = lock_recover(&self.s3gen);
            let s3gen = guard.as_ref().unwrap();
            s3gen
                .render(&real_tokens, reference, seed, on_progress, &should_cancel)
                .map_err(gen_core::Error::from)?
        };
        let samples = match rendered {
            Some(s) => s,
            None => return Err(gen_core::Error::Canceled),
        };

        // Provenance watermark — always applied (the reference behavior; no disable flag).
        let samples = self.watermark(&samples)?;

        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: S3GEN_SR,
            channels: 1,
            ..Default::default()
        }))
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
    // Named-component gate (epic 13657): reject any component key this model does not read, then
    // require BOTH co-requisites present — fail-fast at load with an actionable error, never a
    // mid-render hub fetch. The generator stores the staged paths and consumes them lazily.
    reject_unknown_components(spec, REQUIRED_COMPONENTS, MODEL_ID)?;
    let perth_weights = require_component(
        spec,
        COMPONENT_PERTH,
        MODEL_ID,
        "PerTh provenance watermarker",
    )?
    .clone();
    let voice_embedding_weights = require_component(
        spec,
        COMPONENT_VOICE_EMBEDDING,
        MODEL_ID,
        "chatterbox_ve voice encoder",
    )?
    .clone();
    Ok(ChatterboxGenerator {
        descriptor: descriptor(),
        t3_config: T3Config::LLAMA_520M,
        root,
        perth_weights,
        voice_embedding_weights,
        t3: Mutex::new(None),
        tokenizer: Mutex::new(None),
        embedder: Mutex::new(None),
        s3tokenizer: Mutex::new(None),
        s3gen: Mutex::new(None),
        perth: Mutex::new(None),
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

    /// A minimal spec that clears the load-time component gate with placeholder (never-read) paths —
    /// the components are validated present at load but only touched lazily during a render.
    fn spec_with_stub_components(dir: std::path::PathBuf) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(dir))
            .with_component(
                COMPONENT_PERTH,
                WeightsSource::File(std::path::PathBuf::from("unused-perth.safetensors")),
            )
            .with_component(
                COMPONENT_VOICE_EMBEDDING,
                WeightsSource::File(std::path::PathBuf::from("unused-ve.safetensors")),
            )
    }

    /// `Box<dyn Generator>` is not `Debug`, so `Result::unwrap_err` is unavailable — take the error
    /// out by matching (a passing load is a test failure here).
    fn load_err(spec: &LoadSpec) -> gen_core::Error {
        match load(spec) {
            Ok(_) => panic!("expected a load-time error, but load succeeded"),
            Err(e) => e,
        }
    }

    #[test]
    fn load_requires_both_components_fail_fast() {
        let dir = std::env::temp_dir();
        // Missing BOTH co-requisites → a load-time Msg error naming a missing component + the
        // actionable `with_component` fix (never a mid-render fetch).
        let err = load_err(&LoadSpec::new(WeightsSource::Dir(dir.clone())));
        assert!(matches!(err, gen_core::Error::Msg(_)));
        let msg = err.to_string();
        assert!(
            msg.contains(COMPONENT_PERTH),
            "error must name the missing component: {msg}"
        );
        assert!(
            msg.contains("with_component"),
            "error must be actionable: {msg}"
        );

        // Staging only `perth` → still fails at load, now naming the missing `voice_embedding`.
        let only_perth = LoadSpec::new(WeightsSource::Dir(dir.clone())).with_component(
            COMPONENT_PERTH,
            WeightsSource::File(dir.join("perth.safetensors")),
        );
        assert!(load_err(&only_perth)
            .to_string()
            .contains(COMPONENT_VOICE_EMBEDDING));

        // Both staged → load succeeds; the staged paths are not touched until a render.
        assert!(load(&spec_with_stub_components(dir.clone())).is_ok());

        // An unrecognized component key → typed `Unsupported`.
        let bogus = spec_with_stub_components(dir).with_component(
            "not_a_chatterbox_component",
            WeightsSource::File(std::path::PathBuf::from("x")),
        );
        assert!(matches!(load(&bogus), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir = std::env::temp_dir().join("chatterbox-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load(&spec_with_stub_components(dir)).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let mut req = req_with(vec![ve_vec()]);
        req.cancel = flag;
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }
}
