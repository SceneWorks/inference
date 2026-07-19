//! The `VoiceEmbedder` contract: audio→voice-identity embedding for the identity-preserving audio
//! pipelines (voice-cloning TTS), the audio sibling of [`FaceEmbedder`](crate::face::FaceEmbedder).
//!
//! Exactly the face→image identity path, one modality over: where a
//! [`FaceEmbedder`](crate::face::FaceEmbedder) turns a
//! reference *image* into an ArcFace identity vector that conditions InstantID / PuLID, a
//! `VoiceEmbedder` turns a reference *audio* clip into a speaker-identity vector that conditions a
//! cloned-voice TTS [`Generator`](crate::generator::Generator) — the voice analogue of
//! InstantID's `id_ante_embedding`, carried into generation by
//! [`Conditioning::VoiceEmbedding`](crate::generator::Conditioning::VoiceEmbedding).
//!
//! Backend-neutral like every other gen-core contract — host types only ([`AudioTrack`],
//! [`VoiceEmbedding`] = `Vec<f32>`), no `mlx_rs::Array` / candle `Tensor`. The real embedder (a
//! Chatterbox-style speaker encoder) lands in a `crates/audio` provider (sc-12844); this contract
//! is what it plugs into.

use crate::media::AudioTrack;
use crate::Result;

/// A raw (un-normalized) speaker-identity embedding — one vector per reference voice, of length
/// [`VoiceEmbedderDescriptor::embedding_dim`]. Host type only (`Vec<f32>`), exactly like the
/// [`FaceEmbedder`](crate::face::FaceEmbedder)'s ArcFace vector and the
/// [`ImageEmbedder`](crate::image_embed::ImageEmbedder)'s CLIP vector: callers L2-normalize for
/// cosine similarity, and the identity pipeline feeds it raw.
pub type VoiceEmbedding = Vec<f32>;

/// A voice-identity embedding provider (a speaker-encoder over reference audio).
///
/// `Send + Sync` like [`FaceEmbedder`](crate::face::FaceEmbedder) — the identity embedder it
/// mirrors — so a resolved embedder can be shared across the worker's threads.
pub trait VoiceEmbedder: Send + Sync {
    /// Stable identity + advertised shape, constructible without loading weights.
    fn descriptor(&self) -> &VoiceEmbedderDescriptor;

    /// Embed one reference voice clip into its raw (un-normalized) speaker vector of length
    /// [`VoiceEmbedderDescriptor::embedding_dim`]. Callers L2-normalize for cosine similarity; a
    /// cloned-voice TTS generator feeds it raw through
    /// [`Conditioning::VoiceEmbedding`](crate::generator::Conditioning::VoiceEmbedding).
    fn embed(&self, audio: &AudioTrack) -> Result<VoiceEmbedding>;
}

/// A voice embedder's stable identity + advertised shape. Mirrors
/// [`FaceEmbedderDescriptor`](crate::face::FaceEmbedderDescriptor) field-for-field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceEmbedderDescriptor {
    /// Stable id (e.g. `"chatterbox"`).
    pub id: &'static str,
    /// Provider family (`"voice"`).
    pub family: &'static str,
    /// Tensor backend that registered this embedder (`"mlx"` | `"candle"`); used by the worker's
    /// per-backend capability advertisement.
    pub backend: &'static str,
    /// Dimensionality of the returned [`VoiceEmbedding`].
    pub embedding_dim: usize,
    /// Whether this embedder only runs on macOS (an MLX implementation); a candle implementation
    /// sets this `false`.
    pub mac_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory embedder: the returned vector encodes the track's sample count, so
    /// `embed` is exercised without a tensor backend.
    struct StubVoiceEmbedder {
        descriptor: VoiceEmbedderDescriptor,
    }

    impl VoiceEmbedder for StubVoiceEmbedder {
        fn descriptor(&self) -> &VoiceEmbedderDescriptor {
            &self.descriptor
        }
        fn embed(&self, audio: &AudioTrack) -> Result<VoiceEmbedding> {
            Ok(vec![
                audio.samples.len() as f32;
                self.descriptor.embedding_dim
            ])
        }
    }

    fn descriptor() -> VoiceEmbedderDescriptor {
        VoiceEmbedderDescriptor {
            id: "stub",
            family: "voice",
            backend: "candle",
            embedding_dim: 256,
            mac_only: false,
        }
    }

    fn track(samples: usize) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: 24_000,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn embed_returns_the_raw_vector() {
        let e = StubVoiceEmbedder {
            descriptor: descriptor(),
        };
        let emb = e.embed(&track(3)).unwrap();
        assert_eq!(emb.len(), 256);
        assert_eq!(emb[0], 3.0);
    }

    #[test]
    fn descriptor_advertises_embedding_dim() {
        assert_eq!(descriptor().embedding_dim, 256);
        assert_eq!(descriptor().family, "voice");
        assert_eq!(descriptor().backend, "candle");
    }
}
