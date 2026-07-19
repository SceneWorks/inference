//! `WhisperTranscriber` — the [`gen_core::Transcriber`] implementation for **OpenAI Whisper**
//! automatic speech recognition on the candle audio lane (sc-12850), plus its [`descriptor`] /
//! [`load`] entry points and the explicit [`REGISTRATION`] wired into `candle-audio-catalog` under
//! the id `"whisper_base"` — the first real audio→text [`gen_core::Transcriber`].
//!
//! ## The reuse
//!
//! The Whisper encoder + decoder + log-mel front-end are candle's
//! ([`candle_transformers::models::whisper`]) at the workspace's pinned candle revision — reused
//! wholesale per the epic DoD, NOT re-ported. This crate owns only the gen-core adapter: the
//! [`crate::mel`] host front-end (downmix / resample / mel projection), the [`crate::decode`]
//! autoregressive policy (language/task/timestamp prompt, sampling, cancellation, timestamp parse),
//! and the pinned-SHA hub resolution below.
//!
//! ## Snapshot layout
//!
//! [`load`] expects an `openai/whisper-base` snapshot directory:
//!
//! ```text
//!   config.json         → the whisper Config (num_mel_bins / d_model / layers / vocab / suppress)
//!   tokenizer.json      → the BPE tokenizer (special <|sot|>/<|transcribe|>/<|lang|>/timestamp tokens)
//!   model.safetensors   → the encoder+decoder weights (model.encoder.* / model.decoder.*)
//! ```
//!
//! [`resolve_pinned_snapshot`] materializes exactly that layout through the audio lane's pinned-SHA
//! hub path (`candle_audio::hub`, F-029 — never the mutable `main` revision) and returns the
//! snapshot directory ready for a [`LoadSpec`].
//!
//! ## Checkpoint provenance
//!
//! `openai/whisper-base` (MIT) — the 74 M-param multilingual base checkpoint, chosen for a fast CPU
//! real-weights test. Pinned at commit [`HUB_REVISION`] (a config.json + tokenizer.json +
//! model.safetensors revision; 80-mel).

use std::path::PathBuf;
use std::sync::Mutex;

use candle_audio::candle_core::Tensor;
use candle_audio::gen_core::{
    self, LoadSpec, Progress, TimestampGranularity, TranscribeCapabilities, TranscribeRequest,
    TranscribeTask, Transcriber, TranscriberDescriptor, TranscriptOutput, WeightsSource,
};
use candle_audio::hub::hf_get_pinned;
use candle_audio::{AudioError, Result as AudioResult};
use candle_transformers::models::whisper::{model::Whisper, Config, DTYPE};
use tokenizers::Tokenizer;

use crate::decode::{WhisperDecoder, LANGUAGES};
use crate::mel;

/// Registry id (the SceneWorks worker routes a transcription request to this exact id).
pub const MODEL_ID: &str = "whisper_base";

/// Provider family for the Whisper transcriber.
pub const FAMILY: &str = "whisper";

/// Hub pin: `openai/whisper-base` (Apache-2.0) at an immutable commit SHA (F-029) — a revision
/// carrying config.json + tokenizer.json + model.safetensors (80-mel).
pub const HUB_REPO: &str = "openai/whisper-base";
pub const HUB_REVISION: &str = "e37978b90ca9030d5170a5c07aadb050351a65bb";

/// The license of the pinned Whisper weight checkpoint (sc-13332) — surfaced for SceneWorks'
/// end-product licenses page. Apache-2.0 (permissive), verified against the `openai/whisper-base`
/// model-card metadata (`license: apache-2.0`) — note this is the checkpoint's license, distinct
/// from the MIT license on OpenAI's Whisper *source* repository.
pub const WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "Apache-2.0",
        name: "Apache License 2.0",
        source_url: "https://huggingface.co/openai/whisper-base",
        attribution: Some("Whisper © OpenAI — licensed under Apache-2.0"),
        commercial_use: true,
        restriction: None,
    };

/// This provider's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation.
pub const WEIGHT_LICENSE_ENTRY: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        license: WEIGHT_LICENSE,
    };

/// The three files inside the pinned repo the provider resolves.
pub const CONFIG_FILE: &str = "config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The longest clip (seconds) this provider accepts in one request — a generous ceiling; the decode
/// loop chunks arbitrarily long audio into Whisper's fixed 30 s windows.
const MAX_AUDIO_SECONDS: f32 = 1800.0;

/// Whisper's identity + advertised capabilities — constructible without weights.
pub fn descriptor() -> TranscriberDescriptor {
    TranscriberDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        capabilities: TranscribeCapabilities {
            // The 99 Whisper language codes (a non-empty closed set validation checks a hint against;
            // `None` still auto-detects).
            languages: LANGUAGES.iter().map(|(code, _)| *code).collect(),
            supports_translate: true,
            // Segment timestamps come free from Whisper's timestamp tokens; word-level alignment
            // (cross-attention DTW) is a deliberate non-goal for this first provider.
            supports_segment_timestamps: true,
            supports_word_timestamps: false,
            max_audio_seconds: MAX_AUDIO_SECONDS,
            max_new_tokens: 448,
            mac_only: false,
        },
    }
}

/// Recover a poisoned mutex — the audio twin of `candle_gen::lock_recover`.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The cached loaded model: the (expensive) weights + tokenizer + config, built once on first
/// `transcribe` and reused across requests (the sibling providers' lazy-load discipline).
struct LoadedModel {
    model: Whisper,
    tokenizer: Tokenizer,
    config: Config,
}

/// A loaded (lazy) Whisper transcriber.
pub struct WhisperTranscriber {
    descriptor: TranscriberDescriptor,
    root: PathBuf,
    loaded: Mutex<Option<LoadedModel>>,
}

impl WhisperTranscriber {
    /// Load config.json / tokenizer.json / model.safetensors from the snapshot root into a
    /// [`LoadedModel`]. Called once, under the cache lock.
    fn build(&self) -> AudioResult<LoadedModel> {
        let device = candle_audio::default_device()?;
        let config_path = self.root.join(CONFIG_FILE);
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            AudioError::Msg(format!("whisper: reading {}: {e}", config_path.display()))
        })?;
        let config: Config = serde_json::from_str(&config_text)
            .map_err(|e| AudioError::Msg(format!("whisper: parsing {CONFIG_FILE}: {e}")))?;
        let tokenizer = Tokenizer::from_file(self.root.join(TOKENIZER_FILE))
            .map_err(|e| AudioError::Msg(format!("whisper: loading {TOKENIZER_FILE}: {e}")))?;
        let weights = self.root.join(WEIGHTS_FILE);
        // Safety: from_mmaped_safetensors mmaps a trusted, pinned-SHA snapshot file (F-029).
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)
                .map_err(AudioError::from)?
        };
        let model = Whisper::load(&vb, config.clone()).map_err(AudioError::from)?;
        Ok(LoadedModel {
            model,
            tokenizer,
            config,
        })
    }
}

impl Transcriber for WhisperTranscriber {
    fn descriptor(&self) -> &TranscriberDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TranscribeRequest) -> gen_core::Result<()> {
        self.descriptor.capabilities.validate_request(MODEL_ID, req)
    }

    fn transcribe(
        &self,
        req: &TranscribeRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<TranscriptOutput> {
        self.validate(req)?;
        // Pre-transcribe cancellation seam: consult the flag before ANY heavy work (weights load +
        // decode all come after this).
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }

        let device = candle_audio::default_device().map_err(gen_core::Error::from)?;
        let timestamps = !matches!(req.options.timestamps, TimestampGranularity::None);
        let translate = matches!(req.options.task, TranscribeTask::Translate);
        let temperature = req.sampling.temperature as f64;
        let seed = req.sampling.seed.unwrap_or_else(gen_core::default_seed);
        let max_new_tokens = req.sampling.max_new_tokens;

        let total = 3u32;
        on_progress(Progress::Step { current: 1, total });

        let mut guard = lock_recover(&self.loaded);
        if guard.is_none() {
            *guard = Some(self.build().map_err(gen_core::Error::from)?);
        }
        let loaded = guard.as_mut().expect("just populated");

        // Host front-end: downmix + resample to 16 kHz + log-mel (borrows config only).
        let (mel_vec, n_frames) = mel::track_to_mel(
            &req.audio.samples,
            req.audio.sample_rate,
            req.audio.channels,
            &loaded.config,
        )
        .map_err(gen_core::Error::from)?;
        if n_frames == 0 {
            return Err(gen_core::Error::Msg(
                "whisper: audio produced an empty mel spectrogram".to_owned(),
            ));
        }
        let mel = Tensor::from_vec(mel_vec, (1, loaded.config.num_mel_bins, n_frames), &device)
            .map_err(|e| gen_core::Error::from(AudioError::from(e)))?;
        on_progress(Progress::Step { current: 2, total });

        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }

        // Disjoint-field borrows: the mel above dropped the &config borrow, so the decoder can take
        // &mut model + &tokenizer.
        let mut decoder = WhisperDecoder::new(
            &mut loaded.model,
            &loaded.tokenizer,
            timestamps,
            device.clone(),
        )
        .map_err(gen_core::Error::from)?;

        // Language: an explicit hint (mapped to its <|lang|> token), else auto-detect.
        let (language_token, detected) = match &req.options.language {
            Some(code) => (
                Some(
                    decoder
                        .language_token(code)
                        .map_err(gen_core::Error::from)?,
                ),
                Some(code.clone()),
            ),
            None => {
                let token = decoder
                    .detect_language(&mel)
                    .map_err(gen_core::Error::from)?;
                (Some(token), detected_language_code(&decoder, token))
            }
        };

        on_progress(Progress::Decoding);
        let cancel = req.cancel.clone();
        let out = decoder
            .run(
                &mel,
                language_token,
                translate,
                timestamps,
                temperature,
                seed,
                max_new_tokens,
                &move || cancel.is_cancelled(),
            )
            .map_err(gen_core::Error::from)?;
        on_progress(Progress::Step { current: 3, total });

        Ok(TranscriptOutput {
            text: out.text,
            segments: out.segments,
            language: detected,
            generated_tokens: Some(out.tokens),
            finish_reason: Some(out.finish_reason),
        })
    }
}

/// Map a resolved `<|lang|>` token back to its language code for the output (best-effort).
fn detected_language_code(decoder: &WhisperDecoder, token: u32) -> Option<String> {
    LANGUAGES
        .iter()
        .find(|(code, _)| {
            decoder
                .language_token(code)
                .map(|t| t == token)
                .unwrap_or(false)
        })
        .map(|(code, _)| (*code).to_string())
}

/// Construct the (lazy) Whisper transcriber from a [`LoadSpec`]. `spec.weights` must be the snapshot
/// directory (see module docs); adapters/quantization/control overlays are rejected — refusing is
/// more honest than silently dropping.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Transcriber>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a snapshot directory ({CONFIG_FILE} + {TOKENIZER_FILE} + \
                 {WEIGHTS_FILE}), not a single file"
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
    Ok(Box::new(WhisperTranscriber {
        descriptor: descriptor(),
        root,
        loaded: Mutex::new(None),
    }))
}

// Explicit catalog registration for `whisper_base` (composed by `candle-audio-catalog`).
candle_audio::gen_core::register_transcriber! {
    pub const REGISTRATION = descriptor => load
}

/// Materialize the pinned `openai/whisper-base` snapshot through the audio lane's F-029 hub path:
/// config.json + tokenizer.json + model.safetensors at [`HUB_REVISION`], landing in the HF cache.
/// Returns the snapshot directory as a [`WeightsSource::Dir`] ready for a [`LoadSpec`].
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let cfg = hf_get_pinned(HUB_REPO, HUB_REVISION, CONFIG_FILE)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, TOKENIZER_FILE)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, WEIGHTS_FILE)?;
    let dir = cfg.parent().ok_or_else(|| {
        AudioError::Msg(format!(
            "{MODEL_ID}: resolved {CONFIG_FILE} path {} has no parent directory",
            cfg.display()
        ))
    })?;
    Ok(WeightsSource::Dir(dir.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioTrack, CancelFlag};

    fn clip(seconds: f32, rate: u32) -> AudioTrack {
        AudioTrack {
            samples: vec![0.01; (seconds * rate as f32) as usize],
            sample_rate: rate,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_asr_surface() {
        let d = descriptor();
        assert_eq!(d.id, "whisper_base");
        assert_eq!(d.family, "whisper");
        assert_eq!(d.backend, "candle");
        assert!(d.capabilities.supports_translate);
        assert!(d.capabilities.supports_segment_timestamps);
        assert!(!d.capabilities.supports_word_timestamps);
        assert!(d.capabilities.languages.contains(&"en"));
        assert_eq!(d.capabilities.languages.len(), 99);
    }

    #[test]
    fn validate_rejects_empty_and_overlong_audio() {
        let t = load(&LoadSpec::new(WeightsSource::Dir(std::env::temp_dir()))).unwrap();
        // Empty audio → rejected.
        assert!(t.validate(&TranscribeRequest::default()).is_err());
        // Over the max duration → rejected.
        let long = TranscribeRequest {
            audio: clip(MAX_AUDIO_SECONDS + 10.0, 16_000),
            ..Default::default()
        };
        assert!(t.validate(&long).is_err());
        // A normal short clip → ok.
        let ok = TranscribeRequest {
            audio: clip(3.0, 16_000),
            ..Default::default()
        };
        assert!(t.validate(&ok).is_ok());
    }

    #[test]
    fn load_rejects_unsupported_spec_shapes() {
        // A single file is not a snapshot dir.
        assert!(load(&LoadSpec::new(WeightsSource::File(
            std::env::temp_dir().join("model.safetensors")
        )))
        .is_err());
        // Quantization is refused (typed Unsupported).
        let mut spec = LoadSpec::new(WeightsSource::Dir(std::env::temp_dir()));
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir = std::env::temp_dir().join("whisper-missing-snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let t = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = TranscribeRequest {
            audio: clip(3.0, 16_000),
            cancel: flag,
            ..Default::default()
        };
        // The pre-trip fires before the (missing) snapshot is ever loaded.
        let err = t.transcribe(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }
}
