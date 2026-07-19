//! `MossTtsRealtimeGenerator` — the [`gen_core::Generator`] for **MOSS-TTS-Realtime-1.7B** on the
//! candle audio lane (sc-13334), plus its [`descriptor`]/[`load`] entry points, the pinned-SHA hub
//! path, and the model-weight license.
//!
//! ## Honest partial (see [`crate`] docs)
//!
//! The AR brain — the Qwen3-1.7B backbone ([`crate::backbone`]) + the CSM-style local/depth
//! transformer ([`crate::local`]) — is ported and, on real weights, emits real 16-codebook RVQ
//! speech-token frames ([`crate::decode`]). Turning those frames into a 24 kHz waveform needs the
//! **MOSS-Audio-Tokenizer** codec (a separate ~7 GB RLFQ streaming codec), which is **not yet
//! ported**. So [`generate`](MossTtsRealtimeGenerator::generate) runs the AR loop to produce real
//! frames and then returns a typed error at the codec boundary rather than fabricate audio — and
//! this generator is deliberately **not registered** into `candle-audio-catalog`'s shipping surface
//! (registering an audio generator that cannot render audio would fail the gen-core audio
//! conformance suite and mis-advertise the lane). The [`REGISTRATION`] constant, the ordered-id
//! surface extension, and the bundle smokes land with the codec follow-up.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::candle_core::DType;
use candle_audio::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};
use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
use candle_audio::Result as AudioResult;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::backbone::Backbone;
use crate::config::MossTtsRealtimeConfig;
use crate::decode::{build_prompt_frames, Decoder};
use crate::local::LocalTransformer;

/// Registry id (the id the ordered-generator surface will carry once the codec lands).
pub const MODEL_ID: &str = "moss_tts_realtime";

/// Hub pin: `OpenMOSS-Team/MOSS-TTS-Realtime` at an immutable commit SHA (Apache-2.0 weights +
/// code). ~4.66 GB single-file `model.safetensors` (the AR backbone + local transformer; the codec
/// lives in a separate repo).
pub const HUB_REPO: &str = "OpenMOSS-Team/MOSS-TTS-Realtime";
pub const HUB_REVISION: &str = "6acbc7f161a0db71c291f2d0aaa9eee59334cab2";

/// The MOSS-Audio-Tokenizer codec repo — the separate model that decodes RVQ frames into a 24 kHz
/// waveform. Recorded here for the follow-up that ports it; not fetched by this crate.
pub const CODEC_HUB_REPO: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer";
pub const CODEC_HUB_REVISION: &str = "3cd226ba2947efa357ef453bcad111b6eafba782";

/// The license of the pinned MOSS-TTS-Realtime weight checkpoint (sc-13332) — surfaced for
/// SceneWorks' end-product licenses page. Apache-2.0 (permissive), verified against the
/// `OpenMOSS-Team/MOSS-TTS-Realtime` model card.
pub const WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "Apache-2.0",
    name: "Apache License 2.0",
    source_url: "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime",
    attribution: Some("MOSS-TTS-Realtime-1.7B © OpenMOSS Team — licensed under Apache-2.0"),
    commercial_use: true,
    restriction: None,
};

/// This provider's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation.
pub const WEIGHT_LICENSE_ENTRY: gen_core::WeightLicenseEntry = gen_core::WeightLicenseEntry {
    provider_id: MODEL_ID,
    license: WEIGHT_LICENSE,
};

/// Native output sample rate of the (not-yet-ported) codec (Hz).
pub const SAMPLE_RATE: u32 = 24_000;

/// The RVQ frame rate: 24 kHz / the codec's 1920 downsample = 12.5 frames/second.
pub const FRAME_RATE_HZ: f32 = 12.5;

/// Longest clip advertised (the trained 32 K context ≈ 40 minutes).
pub const MAX_DURATION_SECS: f32 = 2400.0;

/// Default clip length when a request does not set `audio.target_duration` (seconds).
pub const DEFAULT_SECONDS: f32 = 10.0;

/// Prompt languages advertised for the scaffold (the model card lists 20; the full set lands with
/// registration). English + Chinese are the primary verified pair.
pub const LANGUAGES: &[&str] = &["en", "zh"];

/// MOSS-TTS-Realtime's identity + capabilities — constructible without weights. `supports_streaming`
/// is `true`: this is the family's realtime/streaming model, and the AR loop emits one RVQ frame at
/// a time (the codec, once ported, decodes a block of frames into a streamed PCM chunk).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "moss_tts_realtime",
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
            supports_streaming: true,
            supports_multi_speaker: false,
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
            "{id}: prompt (the text to speak) must not be empty"
        )));
    }
    desc.capabilities.validate_request_audio(id, req)
}

/// Convert a requested (or default) clip duration into an AR frame budget.
fn frame_budget(req: &GenerationRequest) -> usize {
    let secs = req
        .audio
        .as_ref()
        .and_then(|a| a.target_duration)
        .unwrap_or(DEFAULT_SECONDS);
    ((secs * FRAME_RATE_HZ).ceil() as usize).max(1)
}

/// The loaded AR stack (backbone + local transformer + tokenizer), built lazily on first use.
struct Loaded {
    decoder: Decoder,
    tokenizer: Tokenizer,
}

impl Loaded {
    fn from_snapshot(root: &std::path::Path) -> gen_core::Result<Self> {
        let cfg = MossTtsRealtimeConfig::from_dir(root).map_err(gen_core::Error::from)?;
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
        // BF16 checkpoint is loaded as F32 (CPU-friendly, and the reference runs the AR head in f32).
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
                .map_err(|e| {
                    gen_core::Error::Msg(format!("{MODEL_ID}: mmap {}: {e}", weights.display()))
                })?
        };
        let backbone = Backbone::new(
            &cfg.language_config,
            cfg.rvq,
            cfg.audio_vocab_size,
            vb.clone(),
        )
        .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: build backbone: {e}")))?;
        let local = LocalTransformer::new(&cfg.local_config, vb.clone()).map_err(|e| {
            gen_core::Error::Msg(format!("{MODEL_ID}: build local transformer: {e}"))
        })?;
        Ok(Self {
            decoder: Decoder {
                backbone,
                local,
                cfg,
            },
            tokenizer,
        })
    }
}

/// A loaded (lazy) MOSS-TTS-Realtime generator.
pub struct MossTtsRealtimeGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    loaded: Mutex<Option<Arc<Loaded>>>,
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl MossTtsRealtimeGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<Loaded>> {
        let mut guard = lock_recover(&self.loaded);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let built = Arc::new(Loaded::from_snapshot(&self.root)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    /// Run the AR brain on real weights and return the emitted RVQ frames (each `rvq` codebook
    /// tokens). Exposed for the real-weights conformance test, which asserts on the token stream
    /// (the codec that would turn these into audio is not yet ported).
    pub fn rvq_frames(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<crate::decode::DecodeResult> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let pipeline = self.pipeline()?;
        let frames = build_prompt_frames(&pipeline.tokenizer, &pipeline.decoder.cfg, &req.prompt)
            .map_err(gen_core::Error::Msg)?;
        let budget = frame_budget(req);
        let total = budget as u32;
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let mut on_frame = |step: usize| {
            on_progress(Progress::Step {
                current: (step as u32) + 1,
                total,
            });
        };
        let result = pipeline
            .decoder
            .run(frames, budget, &probe, &mut on_frame)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: AR decode: {e}")))?;
        match result {
            Some(r) => Ok(r),
            None => Err(gen_core::Error::Canceled),
        }
    }
}

impl Generator for MossTtsRealtimeGenerator {
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
        // AR stage (real weights) → real RVQ frames.
        let result = self.rvq_frames(req, on_progress)?;
        // Codec boundary: announce decode, then hit the honest boundary — the AR brain produced
        // real frames; the MOSS-Audio-Tokenizer codec (RVQ → 24 kHz waveform) is not yet ported, so
        // refuse rather than fabricate audio.
        on_progress(Progress::Decoding);
        let n = result.frames.len();
        Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: AR brain produced {n} real {rvq}-codebook RVQ frame(s) (stop: {stop:?}), \
             but the MOSS-Audio-Tokenizer codec ({CODEC_HUB_REPO}, RVQ → {SAMPLE_RATE} Hz waveform) \
             is not yet ported — refusing to fabricate audio. This generator is intentionally \
             unregistered until the codec lands (see the sc-13334 follow-up).",
            rvq = result.frames.first().map(Vec::len).unwrap_or(0),
            stop = result.stop,
        )))
    }
}

/// Construct the (lazy) generator, returning the **concrete** type (so the conformance test can
/// reach [`MossTtsRealtimeGenerator::rvq_frames`]). [`load`] wraps it behind `dyn Generator`.
pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<MossTtsRealtimeGenerator> {
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
    Ok(MossTtsRealtimeGenerator {
        descriptor: descriptor(),
        root,
        loaded: Mutex::new(None),
    })
}

/// Construct the (lazy) generator as a boxed [`Generator`] trait object.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

// Explicit registration constant for `moss_tts_realtime`. NOTE: `candle-audio-catalog` does NOT
// call this yet — registration is gated on the MOSS-Audio-Tokenizer codec landing (see crate docs),
// exactly as `candle-audio-chatterbox` gates its registration on the S3Gen stack.
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Materialize the pinned MOSS-TTS-Realtime snapshot through the audio lane's F-029 hub path:
/// `config.json` (the snapshot-dir probe), the single-file `model.safetensors`, and the Qwen
/// tokenizer — all at [`HUB_REVISION`], landing in the ordinary HF cache.
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let dir = pinned_snapshot_dir(HUB_REPO, HUB_REVISION, "config.json")?;
    for file in [
        crate::prepare::MODEL_WEIGHTS,
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        hf_get_pinned(HUB_REPO, HUB_REVISION, file)?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag, SpeechSegment};

    fn audio_req(audio: AudioParams) -> GenerationRequest {
        GenerationRequest {
            prompt: "Hello, this is a streaming text to speech test.".into(),
            audio: Some(audio),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_streaming_tts_surface() {
        let d = descriptor();
        assert_eq!(d.id, "moss_tts_realtime");
        assert_eq!(d.family, "moss_tts_realtime");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [24_000]);
        assert!(
            d.capabilities.supports_streaming,
            "the realtime model streams"
        );
        assert!(!d.capabilities.supports_multi_speaker);
        assert_eq!(d.capabilities.max_count, 1);
        assert_eq!(d.capabilities.audio_languages, ["en", "zh"]);
    }

    #[test]
    fn validate_gates_the_request_surface() {
        let d = descriptor();
        // In-surface request passes.
        let ok = audio_req(AudioParams {
            target_duration: Some(4.0),
            sample_rate: Some(24_000),
            language: Some("en".into()),
            ..Default::default()
        });
        assert!(validate_request(&d, &ok).is_ok());

        // Empty prompt rejected.
        let mut r = audio_req(AudioParams::default());
        r.prompt = "  ".into();
        assert!(validate_request(&d, &r).is_err());

        // Unadvertised sample rate → typed Unsupported (shared floor).
        let bad = audio_req(AudioParams {
            sample_rate: Some(44_100),
            ..Default::default()
        });
        assert!(matches!(
            validate_request(&d, &bad),
            Err(gen_core::Error::Unsupported(_))
        ));

        // Unadvertised language → typed Unsupported.
        let bad = audio_req(AudioParams {
            language: Some("ja".into()),
            ..Default::default()
        });
        assert!(matches!(
            validate_request(&d, &bad),
            Err(gen_core::Error::Unsupported(_))
        ));

        // Duration above the advertised cap rejected.
        let bad = audio_req(AudioParams {
            target_duration: Some(MAX_DURATION_SECS + 1.0),
            ..Default::default()
        });
        assert!(validate_request(&d, &bad).is_err());

        // A multi-speaker script → typed Unsupported (we do not advertise multi-speaker).
        let bad = audio_req(AudioParams {
            script: Some(vec![
                SpeechSegment {
                    text: "one".into(),
                    speaker: Some("S1".into()),
                    ..Default::default()
                },
                SpeechSegment {
                    text: "two".into(),
                    speaker: Some("S2".into()),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        });
        assert!(matches!(
            validate_request(&d, &bad),
            Err(gen_core::Error::Unsupported(_))
        ));
    }

    #[test]
    fn frame_budget_tracks_duration() {
        let r = audio_req(AudioParams {
            target_duration: Some(4.0),
            ..Default::default()
        });
        // 4 s * 12.5 fps = 50 frames.
        assert_eq!(frame_budget(&r), 50);
        // Default when unset.
        let r = audio_req(AudioParams::default());
        assert_eq!(
            frame_budget(&r),
            (DEFAULT_SECONDS * FRAME_RATE_HZ).ceil() as usize
        );
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
        let dir = std::env::temp_dir().join("moss-tts-rt-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load_generator(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = GenerationRequest {
            prompt: "hello".into(),
            cancel: flag,
            ..Default::default()
        };
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }

    #[test]
    fn weight_license_is_apache() {
        let lic = WEIGHT_LICENSE;
        assert_eq!(lic.spdx_id, "Apache-2.0");
        assert!(lic.commercial_use, "Apache-2.0 permits commercial use");
        assert_eq!(WEIGHT_LICENSE_ENTRY.provider_id, MODEL_ID);
    }
}
