//! `MossTtsdGenerator` — the [`gen_core::Generator`] for **MOSS-TTSD** (multi-speaker dialogue TTS)
//! on the candle audio lane (sc-13360), plus its [`descriptor`]/[`load`] entry points, the pinned-SHA
//! hub paths (AR checkpoint + XY_Tokenizer codec), and the model-weight license.
//!
//! ## Honest partial (AR brain landed; XY_Tokenizer codec split off)
//!
//! MOSS-TTSD is an AR brain + a **from-scratch custom codec**, the same shape as the sibling
//! MOSS-streaming port (brain sc-13334 → codec sc-13392) which split. This slice lands the **AR
//! brain**: the Qwen3 backbone ([`crate::backbone`]) driving the delay-pattern multi-channel decode
//! ([`crate::decode`]) that emits real, in-range, deterministic 8-channel RVQ speech-codebook frames
//! on the real MOSS-TTSD-v0.5 weights (verified by the real-weights conformance test). The RVQ codec
//! — OpenMOSS's **XY_Tokenizer** (`OpenMOSS-Team/XY_Tokenizer_TTSD_V0`, a 2.1 GB raw-pickle codec
//! whose architecture lives only in the OpenMOSS reference code, *not* candle's Mimi/SNAC/DAC) — is a
//! large separate port. Since sc-13518 it is **landed**: [`crate::codec::XyTokenizerCodec`] decodes
//! the AR's delay-pattern frames into a 24 kHz waveform, so
//! [`generate`](MossTtsdGenerator::generate) renders a real [`gen_core::AudioTrack`] and this
//! generator is **registered** into `candle-audio-catalog`. [`MossTtsdGenerator::rvq_frames`] still
//! exposes the AR token stream the codec consumes (for the AR-stage conformance test).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::candle_core::DType;
use candle_audio::gen_core::{
    self, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, SpeechSegment, WeightsSource,
};
use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
use candle_audio::Result as AudioResult;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::backbone::Backbone;
use crate::codec::XyTokenizerCodec;
use crate::config::MossTtsdConfig;
use crate::decode::{build_prompt_grid, DecodeResult, Decoder, DEFAULT_SYSTEM_PROMPT};

/// Registry id — the id the catalog would carry once the codec lands and the provider ships.
pub const MODEL_ID: &str = "moss_ttsd_v05";

/// Hub pin: `OpenMOSS-Team/MOSS-TTSD-v0.5` at an immutable commit SHA (Apache-2.0 weights + code).
/// The **smallest runnable dialogue checkpoint** — a single ~4.1 GB `model.safetensors` (the v1.0 8B
/// `moss_tts_delay` model is the quality ceiling but a 4-shard ~16 GB bf16 / ~32 GB resident f32
/// stack; v0.5 `moss_ttsd`/`MossTTSDForCausalLM` is one shard and CPU-tractable).
pub const HUB_REPO: &str = "OpenMOSS-Team/MOSS-TTSD-v0.5";
pub const HUB_REVISION: &str = "8527b9136b6afefe2252ae597cecea2e80e7ebeb";

/// The XY_Tokenizer codec repo — the separate ~2.1 GB RVQ codec that decodes the AR's 8-codebook
/// frames into a 24 kHz waveform (ported in [`crate::codec`]). A single raw-pickle `xy_tokenizer.ckpt`
/// (no HF-standard safetensors layout); its decode-side tensors load through candle's pickle reader.
pub const CODEC_HUB_REPO: &str = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0";
pub const CODEC_HUB_REVISION: &str = "c83433728e698ed0698e88cb5096bc221fb8f8c5";

/// The single-file XY_Tokenizer checkpoint inside the codec repo (a `torch.load` pickle).
pub const CODEC_CHECKPOINT_FILE: &str = "xy_tokenizer.ckpt";

/// The license of the pinned MOSS-TTSD-v0.5 weight checkpoint (Apache-2.0, permissive) — surfaced for
/// SceneWorks' end-product licenses page. Verified against the `OpenMOSS-Team/MOSS-TTSD-v0.5` card.
pub const WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "Apache-2.0",
    name: "Apache License 2.0",
    source_url: "https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5",
    attribution: Some("MOSS-TTSD-v0.5 © OpenMOSS Team — licensed under Apache-2.0"),
    commercial_use: true,
    restriction: None,
};

/// This provider's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation.
pub const WEIGHT_LICENSE_ENTRY: gen_core::WeightLicenseEntry = gen_core::WeightLicenseEntry {
    provider_id: MODEL_ID,
    license: WEIGHT_LICENSE,
};

/// Native output sample rate of the XY_Tokenizer codec (Hz).
pub const SAMPLE_RATE: u32 = 24_000;

/// AR audio-frame rate: XY_Tokenizer runs at 12.5 Hz (24 kHz / the codec's 1920 `decoder_upsample_
/// rate`), so one AR delay-pattern frame becomes [`crate::codec::SAMPLES_PER_FRAME`] waveform
/// samples. Drives the duration↔frame budget and the emitted track's length.
pub const FRAME_RATE_HZ: f32 = 12.5;

/// Longest clip advertised.
pub const MAX_DURATION_SECS: f32 = 300.0;

/// Default clip length when a request does not set `audio.target_duration` (seconds).
pub const DEFAULT_SECONDS: f32 = 10.0;

/// Sampler seed used when a request carries no `seed` (the gen-core reproducibility law).
pub const DEFAULT_SAMPLING_SEED: u64 = 13_360;

/// The maximum distinct speakers MOSS-TTSD renders — the vocabulary carries dedicated `<speaker1>` /
/// `<speaker2>` turn tokens, so 2 speakers are honored end-to-end.
pub const MAX_SPEAKERS: u32 = 2;

/// The 20 languages MOSS-TTSD advertises in-band (no external G2P), per the model card.
pub const LANGUAGES: &[&str] = &[
    "zh", "en", "de", "es", "fr", "ja", "it", "he", "ko", "ru", "fa", "ar", "pl", "pt", "cs", "da",
    "sv", "hu", "el", "tr",
];

/// MOSS-TTSD's identity + capabilities — constructible without weights. `supports_multi_speaker` is
/// `true` (the model honors `[S1]`/`[S2]` turn labels; the AR brain models them and this is verified
/// at the token level), with `max_speakers = 2`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "moss_ttsd",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 0,
            max_size: 0,
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            audio_voices: vec![],
            audio_languages: LANGUAGES.to_vec(),
            audio_edit_modes: vec![],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: true,
            max_speakers: Some(MAX_SPEAKERS),
        },
    }
}

/// Capability-driven request validation, factored out for weightless unit tests.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    // A single-voice request needs a prompt; a multi-speaker request needs a non-empty script — but
    // one of the two must carry text.
    let has_script = req
        .audio
        .as_ref()
        .and_then(|a| a.script.as_ref())
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if req.prompt.trim().is_empty() && !has_script {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt or a multi-speaker audio.script (the text to speak) must not be empty"
        )));
    }
    desc.capabilities.validate_request_audio(id, req)
}

/// Convert a requested (or default) clip duration into an AR position budget (plus the delay tail).
fn position_budget(req: &GenerationRequest, channels: usize) -> usize {
    let secs = req
        .audio
        .as_ref()
        .and_then(|a| a.target_duration)
        .unwrap_or(DEFAULT_SECONDS);
    ((secs * FRAME_RATE_HZ).ceil() as usize).max(64) + channels
}

/// Render a request's text: a multi-speaker `audio.script` becomes `[S1]…[S2]…` dialogue (speakers
/// mapped to the two turn tags in first-seen order, alternating when unlabeled); otherwise the plain
/// `prompt`. The `[S1]`/`[S2]` tags are substituted for `<speaker1>`/`<speaker2>` downstream.
pub(crate) fn request_text(req: &GenerationRequest) -> String {
    if let Some(script) = req.audio.as_ref().and_then(|a| a.script.as_ref()) {
        if !script.is_empty() {
            return script_to_dialogue(script);
        }
    }
    req.prompt.clone()
}

/// Map speech segments onto MOSS-TTSD's two-speaker `[S1]`/`[S2]` turn format.
fn script_to_dialogue(script: &[SpeechSegment]) -> String {
    let mut speakers: Vec<String> = Vec::new();
    let mut out = String::new();
    for (i, seg) in script.iter().enumerate() {
        let tag = match &seg.speaker {
            Some(name) => {
                if let Some(idx) = speakers.iter().position(|s| s == name) {
                    idx
                } else {
                    speakers.push(name.clone());
                    speakers.len() - 1
                }
            }
            None => i,
        };
        // Only two turn tags exist; wrap around (validate() has already gated max_speakers).
        let n = (tag % MAX_SPEAKERS as usize) + 1;
        out.push_str(&format!("[S{n}]{}", seg.text.trim()));
    }
    out
}

/// The loaded AR stack (backbone + tokenizer + config), built lazily on first use.
struct Loaded {
    decoder: Decoder,
    tokenizer: Tokenizer,
}

impl Loaded {
    fn from_snapshot(root: &std::path::Path) -> gen_core::Result<Self> {
        let cfg = MossTtsdConfig::from_dir(root).map_err(gen_core::Error::from)?;
        let tok_path = root.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
            gen_core::Error::Msg(format!("{MODEL_ID}: load {}: {e}", tok_path.display()))
        })?;
        let weights = root.join(crate::prepare::MODEL_WEIGHTS);
        if !weights.is_file() {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: weights {} missing (resolve_pinned_snapshot materializes {})",
                weights.display(),
                crate::prepare::MODEL_WEIGHTS
            )));
        }
        let device = candle_audio::default_device()?;
        // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the shared idiom. The
        // BF16 checkpoint is loaded as F32 (CPU-friendly; the reference runs the AR head in f32).
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
                .map_err(|e| {
                    gen_core::Error::Msg(format!("{MODEL_ID}: mmap {}: {e}", weights.display()))
                })?
        };
        let backbone = Backbone::new(&cfg, vb)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: build backbone: {e}")))?;
        Ok(Self {
            decoder: Decoder { backbone, cfg },
            tokenizer,
        })
    }
}

/// A loaded (lazy) MOSS-TTSD generator.
pub struct MossTtsdGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    loaded: Mutex<Option<Arc<Loaded>>>,
    codec: Mutex<Option<Arc<XyTokenizerCodec>>>,
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl MossTtsdGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<Loaded>> {
        let mut guard = lock_recover(&self.loaded);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let built = Arc::new(Loaded::from_snapshot(&self.root)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    /// Load (once, lazily) the XY_Tokenizer codec decoder. The codec is a separate pinned raw-pickle
    /// checkpoint resolved through the audio lane's hub path (or the `MOSS_XY_TOKENIZER_SNAPSHOT`
    /// override); see [`resolve_pinned_codec_checkpoint`].
    fn codec(&self) -> gen_core::Result<Arc<XyTokenizerCodec>> {
        let mut guard = lock_recover(&self.codec);
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let ckpt = resolve_pinned_codec_checkpoint().map_err(gen_core::Error::from)?;
        let built = Arc::new(XyTokenizerCodec::load(&ckpt).map_err(gen_core::Error::from)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    /// Run the AR brain on real weights and return the emitted delay-pattern RVQ frames (each
    /// `channels` codebook tokens, un-shifted and trimmed). Exposed for the real-weights conformance
    /// test, which asserts on the token stream at the codec boundary ([`crate::decode`]).
    pub fn rvq_frames(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DecodeResult> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let pipeline = self.pipeline()?;
        let cfg = &pipeline.decoder.cfg;
        let text = request_text(req);
        let grid = build_prompt_grid(&pipeline.tokenizer, cfg, &text, DEFAULT_SYSTEM_PROMPT)
            .map_err(gen_core::Error::Msg)?;
        let budget = position_budget(req, cfg.channels);
        let total = budget as u32;
        let seed = req.seed.unwrap_or(DEFAULT_SAMPLING_SEED);
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let mut on_position = |step: usize| {
            on_progress(Progress::Step {
                current: (step as u32) + 1,
                total,
            });
        };
        let result = pipeline
            .decoder
            .run(grid, budget, seed, &probe, &mut on_position)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: AR decode: {e}")))?;
        match result {
            Some(r) => Ok(r),
            None => Err(gen_core::Error::Canceled),
        }
    }

    /// The full deterministic synthesis path: run the AR brain to delay-pattern RVQ frames, then
    /// decode them through the XY_Tokenizer codec ([`crate::codec`]) into one 24 kHz [`AudioTrack`].
    /// Progress spans the AR loop; a final [`Progress::Decoding`] marks the codec pass. Cancellation
    /// is honored in the AR loop and re-checked around the codec decode.
    fn synthesize(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<AudioTrack> {
        let result = self.rvq_frames(req, on_progress)?;
        let codec = self.codec()?;
        on_progress(Progress::Decoding);
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let samples = codec
            .decode_frames(&result.frames, &probe)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: XY_Tokenizer decode: {e}")))?
            .ok_or(gen_core::Error::Canceled)?;
        Ok(AudioTrack {
            samples,
            sample_rate: crate::codec::SAMPLE_RATE,
            channels: 1,
            stems: Vec::new(),
        })
    }
}

impl Generator for MossTtsdGenerator {
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
        let track = self.synthesize(req, on_progress)?;
        Ok(GenerationOutput::Audio(track))
    }
}

/// Construct the (lazy) generator, returning the **concrete** type (so the conformance test can reach
/// [`MossTtsdGenerator::rvq_frames`]). [`load`] wraps it behind `dyn Generator`.
pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<MossTtsdGenerator> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a snapshot directory (config.json + {} + tokenizer.json), not a \
                 single file",
                crate::prepare::MODEL_WEIGHTS
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
    Ok(MossTtsdGenerator {
        descriptor: descriptor(),
        root,
        loaded: Mutex::new(None),
        codec: Mutex::new(None),
    })
}

/// Construct the (lazy) generator as a boxed [`Generator`] trait object.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

// Explicit registration constant for `moss_ttsd_v05` (sc-13518): the XY_Tokenizer codec is ported,
// so this generator renders real 24 kHz multi-speaker audio and `candle-audio-catalog` registers it.
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Add the MOSS-TTSD multi-speaker dialogue-TTS generator to an explicit audio registry builder
/// (catalog composition), mirroring the sibling audio provider crates.
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(REGISTRATION)
}

/// Build this crate's own explicit provider catalog (its single-generator surface).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

/// Materialize the pinned XY_Tokenizer codec checkpoint (`xy_tokenizer.ckpt`) at
/// [`CODEC_HUB_REVISION`] through the audio lane's F-029 hub path, returning the `.ckpt` file path.
/// `MOSS_XY_TOKENIZER_SNAPSHOT` overrides with a local snapshot **dir** (the real-weights CI
/// convention — the file is joined) or a direct `.ckpt` file path (air-gapped runs).
pub fn resolve_pinned_codec_checkpoint() -> AudioResult<PathBuf> {
    if let Ok(p) = std::env::var("MOSS_XY_TOKENIZER_SNAPSHOT") {
        let path = PathBuf::from(p);
        return Ok(if path.is_dir() {
            path.join(CODEC_CHECKPOINT_FILE)
        } else {
            path
        });
    }
    hf_get_pinned(CODEC_HUB_REPO, CODEC_HUB_REVISION, CODEC_CHECKPOINT_FILE)
}

/// Materialize the pinned MOSS-TTSD-v0.5 AR snapshot through the audio lane's F-029 hub path:
/// `config.json` (the snapshot-dir probe), the single-file `model.safetensors`, and the Qwen
/// tokenizer — all at [`HUB_REVISION`]. Also warms the pinned XY_Tokenizer codec checkpoint (see
/// [`resolve_pinned_codec_checkpoint`]) so a single call fetches both models the provider needs to
/// render audio. Returns the AR snapshot dir.
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let dir = pinned_snapshot_dir(HUB_REPO, HUB_REVISION, "config.json")?;
    for file in [
        crate::prepare::MODEL_WEIGHTS,
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        hf_get_pinned(HUB_REPO, HUB_REVISION, file)?;
    }
    // Also warm the codec checkpoint — the provider needs both to render a waveform.
    let _ = resolve_pinned_codec_checkpoint()?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag, SpeechSegment};

    fn audio_req(audio: AudioParams) -> GenerationRequest {
        GenerationRequest {
            prompt: "Hello, this is a dialogue text to speech test.".into(),
            audio: Some(audio),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_multi_speaker_surface() {
        let d = descriptor();
        assert_eq!(d.id, "moss_ttsd_v05");
        assert_eq!(d.family, "moss_ttsd");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [24_000]);
        assert!(
            d.capabilities.supports_multi_speaker,
            "MOSS-TTSD is a dialogue model"
        );
        assert_eq!(d.capabilities.max_speakers, Some(2));
        assert_eq!(d.capabilities.max_count, 1);
        assert!(d.capabilities.audio_languages.contains(&"zh"));
        assert!(d.capabilities.audio_languages.contains(&"en"));
        assert_eq!(d.capabilities.audio_languages.len(), 20);
    }

    #[test]
    fn validate_accepts_multi_speaker_scripts_and_gates_the_surface() {
        let d = descriptor();
        // Single-voice request passes.
        let ok = audio_req(AudioParams {
            target_duration: Some(4.0),
            sample_rate: Some(24_000),
            language: Some("en".into()),
            ..Default::default()
        });
        assert!(validate_request(&d, &ok).is_ok());

        // A valid 2-speaker script is ACCEPTED (multi-speaker model).
        let ms = audio_req(AudioParams {
            script: Some(vec![
                SpeechSegment {
                    text: "Hello, how are you today?".into(),
                    speaker: Some("S1".into()),
                    ..Default::default()
                },
                SpeechSegment {
                    text: "I'm doing great, thanks for asking!".into(),
                    speaker: Some("S2".into()),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        });
        assert!(
            validate_request(&d, &ms).is_ok(),
            "a 2-speaker script must be accepted"
        );

        // Over max_speakers → rejected.
        let too_many = audio_req(AudioParams {
            script: Some(
                (0..3)
                    .map(|i| SpeechSegment {
                        text: format!("Line {i}."),
                        speaker: Some(format!("spk{i}")),
                        ..Default::default()
                    })
                    .collect(),
            ),
            ..Default::default()
        });
        assert!(
            validate_request(&d, &too_many).is_err(),
            "> max_speakers must be rejected"
        );

        // Unadvertised sample rate → typed Unsupported.
        let bad = audio_req(AudioParams {
            sample_rate: Some(44_100),
            ..Default::default()
        });
        assert!(matches!(
            validate_request(&d, &bad),
            Err(gen_core::Error::Unsupported(_))
        ));
    }

    #[test]
    fn script_maps_to_two_speaker_dialogue() {
        let script = vec![
            SpeechSegment {
                text: "Hello, how are you today?".into(),
                speaker: Some("alice".into()),
                ..Default::default()
            },
            SpeechSegment {
                text: "I'm doing great, thanks for asking!".into(),
                speaker: Some("bob".into()),
                ..Default::default()
            },
            SpeechSegment {
                text: "Glad to hear it.".into(),
                speaker: Some("alice".into()),
                ..Default::default()
            },
        ];
        let text = script_to_dialogue(&script);
        assert_eq!(
            text,
            "[S1]Hello, how are you today?[S2]I'm doing great, thanks for asking![S1]Glad to hear it."
        );
        // The `[S1]`/`[S2]` tags become the trained `<speaker1>`/`<speaker2>` tokens.
        let d = crate::decode::format_dialogue_text(&text);
        assert!(d.contains("<speaker1>") && d.contains("<speaker2>"));
        assert!(!d.contains("[S1]"));
    }

    #[test]
    fn generate_without_weights_errors_cleanly() {
        // generate() now renders real audio (the XY_Tokenizer codec is ported, sc-13518). With no
        // weights on disk it must surface a typed error while loading the AR snapshot — never panic,
        // never fabricate audio.
        let dir = std::env::temp_dir().join("moss-ttsd-no-weights");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let g = load_generator(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let req = audio_req(AudioParams {
            sample_rate: Some(24_000),
            ..Default::default()
        });
        assert!(
            g.generate(&req, &mut |_| {}).is_err(),
            "generate() must error (not panic) when the AR weights are missing"
        );
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled() {
        let dir = std::env::temp_dir().join("moss-ttsd-cancel");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load_generator(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = GenerationRequest {
            prompt: "hello".into(),
            cancel: flag,
            audio: Some(AudioParams::default()),
            ..Default::default()
        };
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
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
    fn weight_license_is_apache() {
        let lic = WEIGHT_LICENSE;
        assert_eq!(lic.spdx_id, "Apache-2.0");
        assert!(lic.commercial_use);
        assert_eq!(WEIGHT_LICENSE_ENTRY.provider_id, MODEL_ID);
    }
}
