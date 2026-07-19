//! The `Transcriber` contract: speech/audio-to-text transcription for caller-owned workflows.
//!
//! Transcribers are the **audio sibling** of [`Captioner`](crate::caption::Captioner): both consume
//! media and produce text instead of synthesizing media, so ASR gets its own trait rather than
//! riding the [`Generator`](crate::generator::Generator). Where a captioner takes an
//! [`Image`](crate::media::Image) and returns a caption, a transcriber takes an
//! [`AudioTrack`] and returns a transcript (optionally with per-segment or
//! per-word timestamps). The request carries the audio, the language/task hints, and the same shape
//! of autoregressive sampling knobs a captioner exposes ([`CaptionSampling`](crate::caption::CaptionSampling)).
//! Backend-neutral and tensor-free, exactly like the captioner contract.

use crate::media::AudioTrack;
use crate::runtime::{CancelFlag, Progress};
use crate::{Error, Result};

/// A speech/audio-to-text transcription provider — the audio analog of
/// [`Captioner`](crate::caption::Captioner).
pub trait Transcriber {
    /// Stable identity + capability metadata, constructible without loading weights through the
    /// registry.
    fn descriptor(&self) -> &TranscriberDescriptor;

    /// Reject a request this transcriber cannot serve before running model inference.
    fn validate(&self, req: &TranscribeRequest) -> Result<()>;

    /// Transcribe one audio track. Long-running implementations should check
    /// [`TranscribeRequest::cancel`] and report progress through `on_progress`.
    fn transcribe(
        &self,
        req: &TranscribeRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<TranscriptOutput>;
}

/// A single audio-to-text transcription request (the audio twin of
/// [`CaptionRequest`](crate::caption::CaptionRequest)).
#[derive(Clone, Debug, Default)]
pub struct TranscribeRequest {
    /// The interleaved PCM audio to transcribe. The provider resamples/downmixes to its model's
    /// native rate at the edge — the contract stays tensor- and rate-agnostic.
    pub audio: AudioTrack,
    /// Language/task hints preserved from the caller-facing surface (the transcription analog of
    /// [`CaptionOptions`](crate::caption::CaptionOptions)).
    pub options: TranscribeOptions,
    /// Sampling controls for the autoregressive text decoder.
    pub sampling: TranscribeSampling,
    pub cancel: CancelFlag,
}

/// The task an ASR decoder performs — transcribe in the source language, or translate to English
/// (the Whisper task split; models that cannot translate advertise
/// [`supports_translate = false`](TranscribeCapabilities::supports_translate)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TranscribeTask {
    /// Emit text in the audio's own language.
    #[default]
    Transcribe,
    /// Emit an English translation of the audio.
    Translate,
}

/// How finely a transcript should be time-stamped. A model reports which granularities it can
/// serve through [`TranscribeCapabilities`]; requesting one it cannot serve is a typed
/// [`Error::Unsupported`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimestampGranularity {
    /// Plain text only — no timestamps.
    None,
    /// One `[start, end]` span per decoded segment (the natural Whisper timestamp-token output).
    #[default]
    Segment,
    /// Per-word `[start, end]` spans (requires cross-attention alignment; many models cannot).
    Word,
}

/// Transcription hints preserved from the caller-facing job contract (audio twin of
/// [`CaptionOptions`](crate::caption::CaptionOptions)).
#[derive(Clone, Debug, PartialEq)]
pub struct TranscribeOptions {
    /// BCP-47-ish language code hint (`"en"`, `"fr"`, …). `None` asks a multilingual model to
    /// auto-detect the spoken language; a model with a fixed language ignores it.
    pub language: Option<String>,
    /// Transcribe in-language, or translate to English.
    pub task: TranscribeTask,
    /// The timestamp granularity to emit.
    pub timestamps: TimestampGranularity,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: TranscribeTask::Transcribe,
            timestamps: TimestampGranularity::Segment,
        }
    }
}

/// Autoregressive sampling knobs for the transcription text decoder — the same shape a captioner
/// exposes ([`CaptionSampling`](crate::caption::CaptionSampling)), with ASR-appropriate defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranscribeSampling {
    /// Decoder temperature. `0.0` is greedy/deterministic — the ASR default, since transcription
    /// wants the single most-likely token rather than creative variety.
    pub temperature: f32,
    /// Nucleus-sampling mass for stochastic decoding (`temperature > 0`). `1.0` is a no-op.
    pub top_p: f32,
    /// Cap on tokens the decoder may emit per audio window. `0` is rejected by validation.
    pub max_new_tokens: u32,
    /// RNG seed for stochastic sampling (`temperature > 0`). `None` draws a fresh per-call seed via
    /// [`default_seed`](crate::generator::default_seed); pass `Some(seed)` to reproduce an exact
    /// transcript. (At `temperature == 0` decoding is greedy and the seed is unused.)
    pub seed: Option<u64>,
}

impl Default for TranscribeSampling {
    fn default() -> Self {
        Self {
            // Greedy by default: accuracy over variety is the ASR norm.
            temperature: 0.0,
            top_p: 1.0,
            max_new_tokens: 224,
            seed: None,
        }
    }
}

/// A transcription result (the audio twin of [`CaptionOutput`](crate::caption::CaptionOutput)):
/// the full text plus optional timestamped segments and the detected language.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptOutput {
    /// The full transcript (the concatenation of every segment's text).
    pub text: String,
    /// Time-stamped segments, when [`TranscribeOptions::timestamps`] requested them and the model
    /// emitted them. Empty for a plain-text transcription.
    pub segments: Vec<TranscriptSegment>,
    /// The language the model transcribed in, when it detects/reports one (`"en"`, `"fr"`, …).
    pub language: Option<String>,
    /// Total decoded token count, when the provider can report it.
    pub generated_tokens: Option<u32>,
    /// Why decoding stopped, when the provider can report it.
    pub finish_reason: Option<TranscribeFinishReason>,
}

/// One time-stamped transcript segment. `start`/`end` are seconds from the clip origin; `words` is
/// non-empty only when [`TimestampGranularity::Word`] was requested and served.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptSegment {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub words: Vec<TranscriptWord>,
}

/// One time-stamped word within a [`TranscriptSegment`] (`start`/`end` in seconds from the clip
/// origin).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Why transcription stopped, when the provider can report it (audio twin of
/// [`CaptionFinishReason`](crate::caption::CaptionFinishReason)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscribeFinishReason {
    /// The decoder emitted its end-of-transcript token.
    StopToken,
    /// The decoder hit [`TranscribeSampling::max_new_tokens`] first.
    MaxTokens,
    /// Cooperative cancellation tripped mid-decode.
    Cancelled,
}

/// A transcriber's stable identity + advertised capabilities (audio twin of
/// [`CaptionerDescriptor`](crate::caption::CaptionerDescriptor)).
#[derive(Clone, Debug)]
pub struct TranscriberDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this transcriber ("mlx" | "candle").
    pub backend: &'static str,
    pub capabilities: TranscribeCapabilities,
}

/// The shared transcription capability surface. Provider-specific constraints are layered on top by
/// each transcriber's own `validate`.
#[derive(Clone, Debug, Default)]
pub struct TranscribeCapabilities {
    /// Language codes this model can transcribe (empty = any / caller-unconstrained, e.g. a model
    /// that always auto-detects). A non-empty list is the closed set a `language` hint must fall in.
    pub languages: Vec<&'static str>,
    /// Whether the model can translate to English ([`TranscribeTask::Translate`]).
    pub supports_translate: bool,
    /// Whether the model emits per-segment timestamps ([`TimestampGranularity::Segment`]).
    pub supports_segment_timestamps: bool,
    /// Whether the model emits per-word timestamps ([`TimestampGranularity::Word`]).
    pub supports_word_timestamps: bool,
    /// The longest clip (seconds) this provider accepts in one request.
    pub max_audio_seconds: f32,
    /// The decoder's hard per-window token ceiling — the upper bound on
    /// [`TranscribeSampling::max_new_tokens`].
    pub max_new_tokens: u32,
    pub mac_only: bool,
}

impl TranscribeCapabilities {
    /// Reject request fields that exceed the advertised shared capability surface. Capability-gap
    /// rejections are typed [`Error::Unsupported`] (matching the captioner floor: candle gating
    /// depends on the typed variant); malformed values stay [`Error::Msg`].
    pub fn validate_request(&self, id: &str, req: &TranscribeRequest) -> Result<()> {
        // Footgun guard (mirrors CaptionCapabilities): a transcriber that leaves its bounds at the
        // `Default` 0 would reject every request. Catch the descriptor mistake in debug/test builds.
        debug_assert!(
            self.max_new_tokens > 0 && self.max_audio_seconds > 0.0,
            "{id}: TranscribeCapabilities bounds left at Default 0 (max_new_tokens={}, \
             max_audio_seconds={}) — descriptor forgot its bounds",
            self.max_new_tokens,
            self.max_audio_seconds
        );
        if req.audio.samples.is_empty() {
            return Err(Error::Msg(format!("{id}: audio track is empty")));
        }
        if req.audio.sample_rate == 0 {
            return Err(Error::Msg(format!("{id}: audio sample_rate is 0")));
        }
        if req.audio.channels == 0 {
            return Err(Error::Msg(format!("{id}: audio has 0 channels")));
        }
        // Duration = frames / rate, where frames = samples / channels.
        let frames = req.audio.samples.len() / req.audio.channels.max(1) as usize;
        let seconds = frames as f32 / req.audio.sample_rate as f32;
        if seconds > self.max_audio_seconds {
            return Err(Error::Msg(format!(
                "{id}: audio is {seconds:.1}s (> {:.1}s max)",
                self.max_audio_seconds
            )));
        }
        // Capability-gap rejections are typed `Error::Unsupported`.
        if req.options.task == TranscribeTask::Translate && !self.supports_translate {
            return Err(Error::Unsupported(format!(
                "{id}: translation is not supported"
            )));
        }
        match req.options.timestamps {
            TimestampGranularity::None => {}
            TimestampGranularity::Segment if !self.supports_segment_timestamps => {
                return Err(Error::Unsupported(format!(
                    "{id}: segment timestamps are not supported"
                )));
            }
            TimestampGranularity::Word if !self.supports_word_timestamps => {
                return Err(Error::Unsupported(format!(
                    "{id}: word timestamps are not supported"
                )));
            }
            _ => {}
        }
        if let Some(language) = &req.options.language {
            if !self.languages.is_empty() && !self.languages.contains(&language.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported language {language:?} (supported: {:?})",
                    self.languages
                )));
            }
        }
        // NaN-rejecting range checks (mirrors the captioner floor): `NAN < lo || NAN > hi` is
        // false for both, so a plain range test would let a NaN through to poison decode.
        if !(req.sampling.temperature >= 0.0 && req.sampling.temperature <= 2.0) {
            return Err(Error::Msg(format!(
                "{id}: temperature must be between 0 and 2"
            )));
        }
        if !(req.sampling.top_p >= 0.0 && req.sampling.top_p <= 1.0) {
            return Err(Error::Msg(format!("{id}: top_p must be between 0 and 1")));
        }
        if req.sampling.max_new_tokens == 0 || req.sampling.max_new_tokens > self.max_new_tokens {
            return Err(Error::Msg(format!(
                "{id}: max_new_tokens {} out of range 1..={}",
                req.sampling.max_new_tokens, self.max_new_tokens
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> TranscribeCapabilities {
        TranscribeCapabilities {
            languages: vec!["en", "fr"],
            supports_translate: true,
            supports_segment_timestamps: true,
            supports_word_timestamps: false,
            max_audio_seconds: 30.0,
            max_new_tokens: 448,
            mac_only: false,
        }
    }

    fn clip(seconds: f32, rate: u32) -> AudioTrack {
        AudioTrack {
            samples: vec![0.01; (seconds * rate as f32) as usize],
            sample_rate: rate,
            channels: 1,
            ..Default::default()
        }
    }

    fn base_req() -> TranscribeRequest {
        TranscribeRequest {
            audio: clip(3.0, 16_000),
            ..Default::default()
        }
    }

    #[test]
    fn transcribe_defaults_are_asr_shaped() {
        let req = TranscribeRequest::default();
        assert_eq!(req.options.task, TranscribeTask::Transcribe);
        assert_eq!(req.options.timestamps, TimestampGranularity::Segment);
        assert_eq!(req.sampling.temperature, 0.0);
        assert_eq!(req.sampling.top_p, 1.0);
        assert_eq!(req.sampling.max_new_tokens, 224);
    }

    #[test]
    fn validate_request_accepts_supported_surface() {
        let c = caps();
        assert!(c.validate_request("asr", &base_req()).is_ok());
        assert!(c
            .validate_request(
                "asr",
                &TranscribeRequest {
                    options: TranscribeOptions {
                        language: Some("fr".to_owned()),
                        task: TranscribeTask::Translate,
                        timestamps: TimestampGranularity::None,
                    },
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn validate_request_enforces_shared_surface() {
        let c = caps();
        let cases = [
            // empty audio
            TranscribeRequest {
                audio: AudioTrack::default(),
                ..base_req()
            },
            // zero sample rate
            TranscribeRequest {
                audio: AudioTrack {
                    samples: vec![0.0; 100],
                    sample_rate: 0,
                    channels: 1,
                    ..Default::default()
                },
                ..base_req()
            },
            // too long
            TranscribeRequest {
                audio: clip(31.0, 16_000),
                ..base_req()
            },
            // temperature out of range
            TranscribeRequest {
                sampling: TranscribeSampling {
                    temperature: 2.1,
                    ..Default::default()
                },
                ..base_req()
            },
            // top_p out of range
            TranscribeRequest {
                sampling: TranscribeSampling {
                    top_p: 1.1,
                    ..Default::default()
                },
                ..base_req()
            },
            // zero max_new_tokens
            TranscribeRequest {
                sampling: TranscribeSampling {
                    max_new_tokens: 0,
                    ..Default::default()
                },
                ..base_req()
            },
            // over the token ceiling
            TranscribeRequest {
                sampling: TranscribeSampling {
                    max_new_tokens: 449,
                    ..Default::default()
                },
                ..base_req()
            },
        ];
        for (i, req) in cases.iter().enumerate() {
            assert!(
                c.validate_request("asr", req).is_err(),
                "case {i} should have been rejected"
            );
        }
    }

    #[test]
    fn capability_gaps_are_typed_unsupported() {
        // A model that translates nothing, emits no timestamps, and speaks only English: each
        // out-of-surface ask is a typed `Unsupported`.
        let restrictive = TranscribeCapabilities {
            languages: vec!["en"],
            supports_translate: false,
            supports_segment_timestamps: false,
            supports_word_timestamps: false,
            ..caps()
        };
        let gap_cases = [
            TranscribeRequest {
                options: TranscribeOptions {
                    task: TranscribeTask::Translate,
                    ..Default::default()
                },
                ..base_req()
            },
            TranscribeRequest {
                options: TranscribeOptions {
                    timestamps: TimestampGranularity::Segment,
                    ..Default::default()
                },
                ..base_req()
            },
            TranscribeRequest {
                options: TranscribeOptions {
                    timestamps: TimestampGranularity::Word,
                    ..Default::default()
                },
                ..base_req()
            },
            TranscribeRequest {
                options: TranscribeOptions {
                    language: Some("de".to_owned()),
                    timestamps: TimestampGranularity::None,
                    ..Default::default()
                },
                ..base_req()
            },
        ];
        for (i, req) in gap_cases.iter().enumerate() {
            let err = restrictive.validate_request("asr", req).unwrap_err();
            assert!(
                matches!(err, Error::Unsupported(_)),
                "gap case {i} must be Unsupported, got {err:?}"
            );
        }
        // A malformed value (empty audio) stays `Msg`.
        let malformed = TranscribeRequest {
            audio: AudioTrack::default(),
            options: TranscribeOptions {
                timestamps: TimestampGranularity::None,
                ..Default::default()
            },
            ..base_req()
        };
        assert!(matches!(
            restrictive.validate_request("asr", &malformed).unwrap_err(),
            Error::Msg(_)
        ));
    }

    #[test]
    fn nan_sampling_params_are_rejected() {
        let c = caps();
        let bad_temp = TranscribeRequest {
            sampling: TranscribeSampling {
                temperature: f32::NAN,
                ..Default::default()
            },
            ..base_req()
        };
        assert!(c.validate_request("asr", &bad_temp).is_err());
        let bad_top_p = TranscribeRequest {
            sampling: TranscribeSampling {
                top_p: f32::NAN,
                ..Default::default()
            },
            ..base_req()
        };
        assert!(c.validate_request("asr", &bad_top_p).is_err());
    }
}
