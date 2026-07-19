//! `ChatterboxVoiceEmbedder` — the [`gen_core::VoiceEmbedder`] implementation for Chatterbox's
//! speaker encoder on the candle audio lane (sc-12844), plus its [`descriptor`]/[`load`] entry
//! points and the explicit [`REGISTRATION`] wired into `candle-audio-catalog` under the id
//! `"chatterbox_ve"` — the first real voice-identity provider (epic sc-12833), the audio sibling
//! of an ArcFace face embedder.
//!
//! ## Weights
//!
//! [`load`] expects a single-file `WeightsSource::File` pointing at Chatterbox's `ve.safetensors`
//! (≈5.7 MB). [`resolve_pinned_file`] materializes it through the audio lane's pinned-SHA hub
//! path (`candle_audio::hub`, F-029 — never the mutable `main` revision).
//!
//! ## Request mapping
//!
//! [`embed`](gen_core::VoiceEmbedder::embed) turns one reference [`gen_core::AudioTrack`] (any
//! rate; resampled to 16 kHz internally) into the raw 256-d speaker vector. Callers L2-normalize
//! for cosine similarity; a cloned-voice TTS generator (a future sc-12844 slice) feeds it raw
//! through [`Conditioning::VoiceEmbedding`](gen_core::generator::Conditioning::VoiceEmbedding).

use std::sync::{Arc, Mutex};

use candle_audio::candle_core::DType;
use candle_audio::gen_core::{
    self, AudioTrack, LoadSpec, VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding,
    WeightsSource,
};
use candle_audio::hub::hf_get_pinned;
use candle_audio::Result as AudioResult;
use candle_nn::VarBuilder;

use crate::config;
use crate::encoder::SpeakerEncoder;
use crate::frontend::wav_to_mel_frames;

/// Registry id (the SceneWorks worker routes a voice-embed request to this exact id).
pub const MODEL_ID: &str = "chatterbox_ve";

/// Provider family for voice-identity embedders.
pub const FAMILY: &str = "voice";

/// Hub pin: `ResembleAI/chatterbox` at an immutable commit SHA (F-029; MIT weights).
pub const HUB_REPO: &str = "ResembleAI/chatterbox";
pub const HUB_REVISION: &str = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";

/// The voice-encoder checkpoint (a single safetensors file inside the pinned repo).
pub const WEIGHTS_FILE: &str = "ve.safetensors";

/// Minimum reference-clip length (samples, at the source rate) `embed` will accept — one FFT
/// frame is meaningless as an identity cue, so a shorter clip is a typed error, not a silent
/// degenerate vector.
pub const MIN_REFERENCE_SAMPLES: usize = config::N_FFT;

/// Chatterbox voice-encoder identity + advertised shape — constructible without weights
/// (registry introspection).
pub fn descriptor() -> VoiceEmbedderDescriptor {
    VoiceEmbedderDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        embedding_dim: config::EMBED_DIM,
        mac_only: false,
    }
}

/// A loaded (lazy) Chatterbox voice embedder. The encoder weights are memory-mapped on first
/// `embed`, not at [`load`] — the sibling providers' lazy-load discipline.
pub struct ChatterboxVoiceEmbedder {
    descriptor: VoiceEmbedderDescriptor,
    weights: std::path::PathBuf,
    encoder: Mutex<Option<Arc<SpeakerEncoder>>>,
}

impl ChatterboxVoiceEmbedder {
    fn encoder(&self) -> gen_core::Result<Arc<SpeakerEncoder>> {
        let mut guard = lock_recover(&self.encoder);
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        if !self.weights.is_file() {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: voice-encoder weights {} missing (resolve_pinned_file materializes \
                 {WEIGHTS_FILE})",
                self.weights.display()
            )));
        }
        let device = candle_audio::default_device()?;
        // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the same idiom every
        // candle provider uses for its checkpoint.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.weights),
                DType::F32,
                &device,
            )
                .map_err(|e| {
                    gen_core::Error::Msg(format!("{MODEL_ID}: loading {WEIGHTS_FILE}: {e}"))
                })?
        };
        let built = Arc::new(
            SpeakerEncoder::new(vb, device)
                .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: building encoder: {e}")))?,
        );
        *guard = Some(built.clone());
        Ok(built)
    }
}

/// Recover a poisoned mutex — the audio twin of `candle_gen::lock_recover`.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl VoiceEmbedder for ChatterboxVoiceEmbedder {
    fn descriptor(&self) -> &VoiceEmbedderDescriptor {
        &self.descriptor
    }

    fn embed(&self, audio: &AudioTrack) -> gen_core::Result<VoiceEmbedding> {
        if audio.samples.len() < MIN_REFERENCE_SAMPLES {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: reference clip has {} samples (< {MIN_REFERENCE_SAMPLES}); too short \
                 to extract a speaker identity",
                audio.samples.len()
            )));
        }
        // Down-mix to mono if the caller handed an interleaved multi-channel clip.
        let mono = to_mono(&audio.samples, audio.channels);
        let mel = wav_to_mel_frames(&mono, audio.sample_rate);
        if mel.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: reference clip produced no analysis frames"
            )));
        }
        let embedding = self
            .encoder()?
            .embed_mel_frames(&mel)
            .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: encoding: {e}")))?;
        Ok(embedding)
    }
}

/// Interleaved `channels`-channel PCM → mono by averaging channels (a no-op for mono).
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Construct the (lazy) Chatterbox voice embedder from a [`LoadSpec`]. `spec.weights` must be the
/// single `ve.safetensors` file; quantization/adapters/control overlays are rejected — refusing
/// is more honest than silently dropping.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn VoiceEmbedder>> {
    let weights = match &spec.weights {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects the single {WEIGHTS_FILE} file, not a snapshot directory"
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
    Ok(Box::new(ChatterboxVoiceEmbedder {
        descriptor: descriptor(),
        weights,
        encoder: Mutex::new(None),
    }))
}

// Explicit catalog registration for `chatterbox_ve` (composed by `candle-audio-catalog`).
pub const REGISTRATION: gen_core::VoiceEmbedderRegistration =
    gen_core::VoiceEmbedderRegistration { descriptor, load };

/// Materialize the pinned `ve.safetensors` through the audio lane's F-029 hub path, landing in
/// the ordinary HF cache. Returns it as a [`WeightsSource::File`] ready for a [`LoadSpec`].
pub fn resolve_pinned_file() -> AudioResult<WeightsSource> {
    let path = hf_get_pinned(HUB_REPO, HUB_REVISION, WEIGHTS_FILE)?;
    Ok(WeightsSource::File(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_advertises_the_voice_surface() {
        let d = descriptor();
        assert_eq!(d.id, "chatterbox_ve");
        assert_eq!(d.family, "voice");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.embedding_dim, 256);
        assert!(!d.mac_only);
    }

    #[test]
    fn load_rejects_unsupported_spec_shapes() {
        let dir = std::env::temp_dir();
        // A snapshot dir is rejected (single-file provider).
        assert!(load(&LoadSpec::new(WeightsSource::Dir(dir.clone()))).is_err());
        // Quantization is rejected, typed Unsupported.
        let mut spec = LoadSpec::new(WeightsSource::File(dir.join("ve.safetensors")));
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn embed_rejects_a_too_short_clip() {
        let e = load(&LoadSpec::new(WeightsSource::File(
            std::env::temp_dir().join("ve.safetensors"),
        )))
        .unwrap();
        let clip = AudioTrack {
            samples: vec![0.0; 8],
            sample_rate: 16_000,
            channels: 1,
        };
        // Fails on the length gate before any weight I/O (weights need not exist).
        assert!(e.embed(&clip).is_err());
    }

    #[test]
    fn to_mono_averages_channels() {
        // Stereo [L,R,L,R] → mono average.
        let m = to_mono(&[1.0, 3.0, 2.0, 4.0], 2);
        assert_eq!(m, vec![2.0, 3.0]);
        // Mono passthrough.
        assert_eq!(to_mono(&[1.0, 2.0], 1), vec![1.0, 2.0]);
    }
}
