//! `MossTtsRealtimeGenerator` — the [`gen_core::Generator`] for **MOSS-TTS-Realtime-1.7B** on the
//! candle audio lane (sc-13334 + sc-13392), plus its [`descriptor`]/[`load`] entry points, the
//! pinned-SHA hub paths (AR + codec), and the model-weight license.
//!
//! ## Full streaming TTS (sc-13392)
//!
//! The AR brain — the Qwen3-1.7B backbone ([`crate::backbone`]) + the CSM-style local/depth
//! transformer ([`crate::local`]) — emits real 16-codebook RVQ speech-token frames
//! ([`crate::decode`]), and the ported **MOSS-Audio-Tokenizer** codec ([`crate::codec`]) turns those
//! into a 24 kHz waveform. [`generate`](MossTtsRealtimeGenerator::generate) and
//! [`generate_streaming`](MossTtsRealtimeGenerator::generate_streaming) share one deterministic
//! synthesis path (AR frames → incremental causal codec decode → PCM chunks), so the one-shot output
//! is byte-identical to the concatenated stream. This generator is **registered** into
//! `candle-audio-catalog`'s shipping surface.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::candle_core::DType;
use candle_audio::gen_core::{
    self, AudioChunk, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};
use candle_audio::hub::{hf_get_pinned, pinned_snapshot_dir};
use candle_audio::Result as AudioResult;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::backbone::Backbone;
use crate::codec::MossAudioCodec;
use crate::config::MossTtsRealtimeConfig;
use crate::decode::{build_prompt_frames, Decoder};
use crate::local::LocalTransformer;

/// Registry id — the id the catalog's ordered-generator surface carries.
pub const MODEL_ID: &str = "moss_tts_realtime";

/// Hub pin: `OpenMOSS-Team/MOSS-TTS-Realtime` at an immutable commit SHA (Apache-2.0 weights +
/// code). ~4.66 GB single-file `model.safetensors` (the AR backbone + local transformer; the codec
/// lives in a separate repo).
pub const HUB_REPO: &str = "OpenMOSS-Team/MOSS-TTS-Realtime";
pub const HUB_REVISION: &str = "6acbc7f161a0db71c291f2d0aaa9eee59334cab2";

/// The MOSS-Audio-Tokenizer codec repo — the separate ~7.1 GB model that decodes RVQ frames into a
/// 24 kHz waveform (ported in [`crate::codec`], resolved by [`resolve_pinned_codec_snapshot`]).
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

/// Native output sample rate of the codec (Hz).
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
/// a time (the codec decodes a block of frames into a streamed PCM chunk).
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

/// Default RVQ frames per streaming block (≈ 0.64 s at 12.5 fps). Sized down per request so a short
/// clip still yields ≥ 2 chunks (the streaming incrementality law).
const DEFAULT_FRAMES_PER_BLOCK: usize = 8;

/// The streaming block size for a clip of `total_frames`: at most [`DEFAULT_FRAMES_PER_BLOCK`], but
/// never more than half the clip, so a `≥ 2`-frame clip always produces `≥ 2` chunks.
fn frames_per_block(total_frames: usize) -> usize {
    DEFAULT_FRAMES_PER_BLOCK
        .min(total_frames.div_ceil(2))
        .max(1)
}

/// A loaded (lazy) MOSS-TTS-Realtime generator.
pub struct MossTtsRealtimeGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    loaded: Mutex<Option<Arc<Loaded>>>,
    codec: Mutex<Option<Arc<MossAudioCodec>>>,
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
    /// before it is handed to the codec ([`crate::codec`]).
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

    /// Load (once, lazily) the MOSS-Audio-Tokenizer codec decoder for `rvq` codebooks. The codec is
    /// a separate pinned snapshot resolved through the audio lane's hub path (or the
    /// `MOSS_AUDIO_TOKENIZER_SNAPSHOT` override); see [`resolve_pinned_codec_snapshot`].
    fn codec(&self, rvq: usize) -> gen_core::Result<Arc<MossAudioCodec>> {
        let mut guard = lock_recover(&self.codec);
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let dir = resolve_pinned_codec_snapshot().map_err(gen_core::Error::from)?;
        let built = Arc::new(MossAudioCodec::load(&dir, rvq).map_err(gen_core::Error::from)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    /// The single deterministic synthesis path shared by [`generate`](Self::generate) and
    /// [`generate_streaming`](Self::generate_streaming): run the AR brain to a full set of RVQ frames,
    /// then decode the codec **incrementally** over growing frame prefixes, emitting one
    /// [`AudioChunk`] per newly-revealed PCM block and assembling the identical full [`AudioTrack`].
    ///
    /// Because the codec decode graph is fully causal, decoding a growing prefix reproduces the
    /// earlier samples byte-for-byte — so the concatenated chunks equal the returned track exactly
    /// (the reassembly law), and the two entry points return byte-identical audio for the same
    /// request+seed (they call this one function). The first chunk is emitted after decoding only the
    /// first block, so first-chunk latency is strictly below the full synthesis time.
    fn synthesize(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<AudioTrack> {
        let result = self.rvq_frames(req, on_progress)?;
        on_progress(Progress::Decoding);
        let frames = result.frames;

        let pipeline = self.pipeline()?;
        let codec = self.codec(pipeline.decoder.cfg.rvq)?;
        let sample_rate = codec.sample_rate();
        let spf = codec.samples_per_frame();

        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();

        let total = frames.len();
        let total_samples = total * spf;
        let block = frames_per_block(total);

        let mut samples: Vec<f32> = Vec::with_capacity(total_samples);
        let mut index = 0usize;
        let mut end = block;
        while samples.len() < total_samples {
            let end_frames = end.min(total);
            let wav = match codec
                .decode_frames(&frames[..end_frames], &probe)
                .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: codec decode: {e}")))?
            {
                Some(w) => w,
                None => return Err(gen_core::Error::Canceled),
            };
            // The newly-revealed tail beyond what earlier prefixes already emitted.
            let tail = &wav[samples.len()..];
            if !tail.is_empty() {
                on_chunk(AudioChunk {
                    samples: tail.to_vec(),
                    sample_rate,
                    channels: 1,
                    index,
                });
                samples.extend_from_slice(tail);
                index += 1;
            }
            if end_frames >= total {
                break;
            }
            end += block;
        }

        Ok(AudioTrack {
            samples,
            sample_rate,
            channels: 1,
            stems: Vec::new(),
        })
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
        // Same deterministic path as generate_streaming, with the chunk sink discarded — so the
        // one-shot output is byte-identical to the concatenated stream.
        let track = self.synthesize(req, &mut |_| {}, on_progress)?;
        Ok(GenerationOutput::Audio(track))
    }

    fn generate_streaming(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let track = self.synthesize(req, on_chunk, on_progress)?;
        Ok(GenerationOutput::Audio(track))
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
        codec: Mutex::new(None),
    })
}

/// Construct the (lazy) generator as a boxed [`Generator`] trait object.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

// Explicit registration constant for `moss_tts_realtime` (sc-13392): the MOSS-Audio-Tokenizer codec
// is ported, so this generator renders real 24 kHz audio and `candle-audio-catalog` registers it.
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Add the MOSS-TTS-Realtime streaming-TTS generator to an explicit audio registry builder (catalog
/// composition), mirroring the sibling audio provider crates (e.g. `candle-audio-kokoro`).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(REGISTRATION)
}

/// Build this crate's own explicit provider catalog (its single-generator surface).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

/// The codec snapshot's sharded weight files (the 2-shard `model*.safetensors` + its index).
const CODEC_WEIGHT_FILES: &[&str] = &[
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

/// Materialize the pinned MOSS-TTS-Realtime AR snapshot through the audio lane's F-029 hub path:
/// `config.json` (the snapshot-dir probe), the single-file `model.safetensors`, and the Qwen
/// tokenizer — all at [`HUB_REVISION`]. Also materializes the pinned MOSS-Audio-Tokenizer codec
/// snapshot (see [`resolve_pinned_codec_snapshot`]) so a single call warms both models the provider
/// needs to render audio. Returns the AR snapshot dir.
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let dir = pinned_snapshot_dir(HUB_REPO, HUB_REVISION, "config.json")?;
    for file in [
        crate::prepare::MODEL_WEIGHTS,
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        hf_get_pinned(HUB_REPO, HUB_REVISION, file)?;
    }
    // Also warm the codec snapshot — the provider needs both to render a waveform.
    let _ = resolve_pinned_codec_snapshot()?;
    Ok(dir)
}

/// Materialize the pinned MOSS-Audio-Tokenizer codec snapshot (`config.json` + the sharded
/// `model*.safetensors` + its index) at [`CODEC_HUB_REVISION`] through the audio lane's F-029 hub
/// path, returning its snapshot directory. `MOSS_AUDIO_TOKENIZER_SNAPSHOT` overrides with a local
/// dir (for the real-weight test / air-gapped runs).
pub fn resolve_pinned_codec_snapshot() -> AudioResult<PathBuf> {
    if let Ok(dir) = std::env::var("MOSS_AUDIO_TOKENIZER_SNAPSHOT") {
        return Ok(PathBuf::from(dir));
    }
    let dir = pinned_snapshot_dir(CODEC_HUB_REPO, CODEC_HUB_REVISION, "config.json")?;
    for file in CODEC_WEIGHT_FILES {
        hf_get_pinned(CODEC_HUB_REPO, CODEC_HUB_REVISION, file)?;
    }
    match dir {
        WeightsSource::Dir(p) => Ok(p),
        other => Err(candle_audio::AudioError::Msg(format!(
            "{MODEL_ID}: expected a codec snapshot dir, got {other:?}"
        ))),
    }
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
