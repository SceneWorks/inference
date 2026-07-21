//! `MossTtsRealtimeGenerator` — the [`gen_core::Generator`] for **MOSS-TTS-Realtime-1.7B** on the
//! candle audio lane (sc-13334 + sc-13392), plus its [`descriptor`]/[`load`] entry points, the
//! pinned-SHA AR hub path, the MOSS-Audio-Tokenizer codec pin (staged as the passed-in
//! [`CODEC_COMPONENT_ID`] component, never self-fetched — epic 13657, sc-13662), and the
//! model-weight license.
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
/// 24 kHz waveform (ported in [`crate::codec`]).
///
/// **The codec is a passed-in component, not a self-fetch (epic 13657, sc-13662).** These pin
/// constants record *which* snapshot the consumer must stage under the [`CODEC_COMPONENT_ID`]
/// component of [`gen_core::LoadSpec::components`]; this crate never fetches it. Provisioning the pin
/// (fetch + snapshot assembly) is the consumer's job (SceneWorks, epic 13678).
pub const CODEC_HUB_REPO: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer";
pub const CODEC_HUB_REVISION: &str = "3cd226ba2947efa357ef453bcad111b6eafba782";

/// The [`ModelDescriptor::required_components`] id under which the caller stages the
/// MOSS-Audio-Tokenizer codec **snapshot directory** in [`gen_core::LoadSpec::components`]
/// (sc-13662). Validated at load via [`gen_core::require_component`]; the codec is then built lazily
/// from the staged directory.
pub const CODEC_COMPONENT_ID: &str = "codec";

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
    component: None,
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

/// Sampler seed used when a request carries no `seed` — keeps decoding deterministic (the gen-core
/// reproducibility law) while still using the reference's sampling (not greedy, which collapses).
pub const DEFAULT_SAMPLING_SEED: u64 = 13_392;

/// Prompt languages advertised for the scaffold (the model card lists 20; the full set lands with
/// registration). English + Chinese are the primary verified pair.
pub const LANGUAGES: &[&str] = &["en", "zh"];

/// MOSS-TTS-Realtime's identity + capabilities — constructible without weights. `supports_streaming`
/// is `true`: this is the family's realtime/streaming model, and the AR loop emits one RVQ frame at
/// a time (the codec decodes a block of frames into a streamed PCM chunk).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        required_components: &[CODEC_COMPONENT_ID],
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
                "{MODEL_ID}: weights {} missing (the passed-in snapshot must supply {})",
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

/// The streaming block size for a clip of `budget_frames` (the AR frame budget): at most
/// [`DEFAULT_FRAMES_PER_BLOCK`], but never more than half the budget, so a budget-reaching run of
/// `≥ 2` frames always streams `≥ 2` chunks.
fn frames_per_block(budget_frames: usize) -> usize {
    DEFAULT_FRAMES_PER_BLOCK
        .min(budget_frames.div_ceil(2))
        .max(1)
}

/// A loaded (lazy) MOSS-TTS-Realtime generator.
pub struct MossTtsRealtimeGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    /// The MOSS-Audio-Tokenizer codec snapshot directory, resolved at load from the passed-in
    /// [`CODEC_COMPONENT_ID`] component (sc-13662). The codec is built lazily from it on first
    /// synthesize; never a hub fetch.
    codec_dir: PathBuf,
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
        // Deterministic token sampling seeded by the request (a `None` seed maps to a fixed constant),
        // so the gen-core reproducibility law holds and generate/generate_streaming agree.
        let seed = req.seed.unwrap_or(DEFAULT_SAMPLING_SEED);
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let mut on_frame = |step: usize, _frame: &[u32]| {
            on_progress(Progress::Step {
                current: (step as u32) + 1,
                total,
            });
            Ok(())
        };
        let result = pipeline
            .decoder
            .run(frames, budget, seed, &probe, &mut on_frame)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: AR decode: {e}")))?;
        match result {
            Some(r) => Ok(r),
            None => Err(gen_core::Error::Canceled),
        }
    }

    /// Load (once, lazily) the MOSS-Audio-Tokenizer codec decoder for `rvq` codebooks from the
    /// snapshot directory the caller staged as the [`CODEC_COMPONENT_ID`] component at load
    /// (sc-13662) — never a hub fetch. The presence of the component was already validated in
    /// [`load_generator`]; here we build it from the stored directory on first synthesize.
    fn codec(&self, rvq: usize) -> gen_core::Result<Arc<MossAudioCodec>> {
        let mut guard = lock_recover(&self.codec);
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let built =
            Arc::new(MossAudioCodec::load(&self.codec_dir, rvq).map_err(gen_core::Error::from)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    /// The single deterministic synthesis path shared by [`generate`](Self::generate) and
    /// [`generate_streaming`](Self::generate_streaming): drive the AR brain and, **from inside the AR
    /// loop**, decode the codec block-wise over the growing RVQ-frame prefix — emitting one
    /// [`AudioChunk`] per newly-revealed PCM block *while later frames are still being generated*
    /// ([`crate::chunk::StreamingChunker`]).
    ///
    /// Because the codec decode graph is fully causal, decoding a growing prefix reproduces the
    /// earlier samples byte-for-byte — so the concatenated chunks equal the returned track exactly
    /// (the reassembly law), and the two entry points return byte-identical audio for the same
    /// request+seed (they call this one function). The first chunk is emitted after the first block
    /// of AR frames rather than after the whole track, so first-chunk latency is proportional to one
    /// block of AR frames, not the full synthesis time.
    ///
    /// The AR backbone runs a KV cache (sc-13417): the prompt is prefilled once and each emitted
    /// frame is a single-token step, so per-frame cost is O(1) amortized rather than O(seq). This is
    /// a pure latency optimization — the cached hidden states are byte-identical to the old
    /// full-recompute path, so the emitted frames and the streaming interleaving here are unchanged.
    fn synthesize(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<AudioTrack> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let pipeline = self.pipeline()?;
        let codec = self.codec(pipeline.decoder.cfg.rvq)?;

        let frames = build_prompt_frames(&pipeline.tokenizer, &pipeline.decoder.cfg, &req.prompt)
            .map_err(gen_core::Error::Msg)?;
        let budget = frame_budget(req);
        let total = budget as u32;
        // Deterministic token sampling seeded by the request (a `None` seed maps to a fixed constant),
        // so the gen-core reproducibility law holds and generate/generate_streaming agree.
        let seed = req.seed.unwrap_or(DEFAULT_SAMPLING_SEED);
        // Block sizing is driven by the frame budget (known up front, before the loop): at most
        // DEFAULT_FRAMES_PER_BLOCK and never more than half the budget, so a budget-reaching run
        // always streams >= 2 chunks.
        let block = frames_per_block(budget);

        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();

        let mut chunker = crate::chunk::StreamingChunker::new(codec.as_ref(), block);
        let mut canceled = false;
        let run = {
            // The AR loop hands each emitted frame here; the chunker decodes + streams block-wise.
            let mut on_frame =
                |step: usize, frame: &[u32]| -> candle_audio::candle_core::Result<()> {
                    on_progress(Progress::Step {
                        current: (step as u32) + 1,
                        total,
                    });
                    if chunker.push(frame.to_vec(), &probe, on_chunk)?.is_none() {
                        canceled = true;
                    }
                    Ok(())
                };
            pipeline
                .decoder
                .run(frames, budget, seed, &probe, &mut on_frame)
                .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: AR decode: {e}")))?
        };
        if canceled || run.is_none() {
            return Err(gen_core::Error::Canceled);
        }
        on_progress(Progress::Decoding);
        // Flush any remaining frames below a full block into a final chunk and take the full track.
        match chunker
            .finish(&probe, on_chunk)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: codec decode: {e}")))?
        {
            Some(track) => Ok(track),
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
    // The MOSS-Audio-Tokenizer codec is a passed-in component (epic 13657, sc-13662): validate it at
    // LOAD — an unknown component key is a typed-`Unsupported` caller mistake, and a missing `codec`
    // is a caller-actionable load error — so an unprovisioned codec fails here, never as a mid-render
    // hub fetch. The snapshot directory is stored; the codec is built lazily from it in synthesize.
    gen_core::reject_unknown_components(spec, &[CODEC_COMPONENT_ID], MODEL_ID)?;
    let codec_dir = match gen_core::require_component(
        spec,
        CODEC_COMPONENT_ID,
        MODEL_ID,
        "MOSS-Audio-Tokenizer codec",
    )? {
        // The codec is a sharded snapshot (config.json + model*.safetensors), so it needs a directory.
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_ID} codec component expects a snapshot directory (config.json + \
                 model*.safetensors), not a single file: {}",
                p.display()
            )));
        }
    };
    Ok(MossTtsRealtimeGenerator {
        descriptor: descriptor(),
        root,
        codec_dir,
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

    /// A `LoadSpec` with the required `codec` component staged as a directory (a placeholder path —
    /// the codec is built lazily, so `load` never touches it). Every load in these weightless tests
    /// goes through here so the sc-13662 component gate is satisfied.
    fn spec_with_codec(dir: PathBuf) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(dir)).with_component(
            CODEC_COMPONENT_ID,
            WeightsSource::Dir(PathBuf::from("/nonexistent/moss-audio-tokenizer")),
        )
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
        let mut spec = spec_with_codec(dir.clone());
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    /// Weights-free proof of the sc-13662 load-time codec gate: a fully-provisioned spec loads, but
    /// removing the required `codec` component (or adding an unknown one) fails at LOAD — the codec is
    /// never fetched mid-render (epic 13657). Driven through the real `load` by the shared testkit.
    #[test]
    fn missing_codec_component_fails_at_load() {
        let dir = std::env::temp_dir().join("moss-tts-rt-load-gate");
        std::fs::create_dir_all(&dir).unwrap();
        let base = spec_with_codec(dir);
        // The fully-provisioned spec loads (the codec is lazy — no directory read yet).
        assert!(
            load(&base).is_ok(),
            "a spec staging the required codec component must load"
        );
        gen_core_testkit::check_component_load_gate(load, &base, &[CODEC_COMPONENT_ID])
            .expect("missing / unknown codec component must be a load-time error");
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir = std::env::temp_dir().join("moss-tts-rt-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let g = load_generator(&spec_with_codec(dir)).unwrap();
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
