//! The `AudioEmbedder` contract: a single global audio embedding in a joint audio-text space
//! (CLAP-style) for retrieval / search / auto-tagging (P5, sc-12851).
//!
//! The audio parallel of [`ImageEmbedder`](crate::image_embed::ImageEmbedder): where an image
//! embedder maps a whole image into one CLIP-style vector, an `AudioEmbedder` maps a whole
//! [`AudioTrack`] into one vector in a space that a text query also lives in — so audio↔text
//! retrieval (find the clip that best matches "a dog barking") works by plain cosine similarity.
//!
//! This is the *semantic* audio embedder, deliberately distinct from
//! [`VoiceEmbedder`](crate::voice_embed::VoiceEmbedder) (sc-12838), which is a speaker-**identity**
//! vector: the same split as [`ImageEmbedder`](crate::image_embed::ImageEmbedder) (semantic) vs [`FaceEmbedder`](crate::face::FaceEmbedder)
//! (identity), one modality over.
//!
//! Backend-neutral like every other gen-core contract — host types only ([`AudioTrack`], `&str`,
//! `Vec<f32>`), no `mlx_rs::Array` / candle `Tensor`. The real embedder (a CLAP-class HTSAT audio
//! tower + RoBERTa text tower projected into one CLIP-style space) lands in a `crates/audio`
//! provider (`candle-audio-clap`); this contract is what it plugs into.
//!
//! ## Why text embedding lives on this trait (the joint-space decision)
//!
//! CLAP is *intrinsically joint*: one loaded checkpoint owns **both** an audio encoder and a text
//! encoder and projects both into a **single** shared space. Unlike the image path — where
//! [`ImageEmbedder`](crate::image_embed::ImageEmbedder) and [`TextEmbedder`](crate::text_embed::TextEmbedder) are *separate* traits and
//! *separate* registrations whose "same space" is only a convention enforced by matching the
//! [`space`](AudioEmbedderDescriptor::space) string — a CLAP provider gives you both vectors from
//! **one** object. Exposing [`embed_text`](AudioEmbedder::embed_text) as a companion method on the
//! same trait therefore makes the joint guarantee *structural*: you cannot accidentally rank audio
//! against a text vector from a different encoder, because both come from the same loaded model with
//! the same projection. The acceptance path (a text query ranking a set of audio clips by cosine)
//! needs exactly this — a text vector in the *same* space as the audio vectors — so the contract
//! exposes both, and the [`space`](AudioEmbedderDescriptor::space) field is still carried for
//! cross-provider comparability bookkeeping.

use crate::media::AudioTrack;
use crate::Result;

/// A semantic audio embedding provider that also embeds text into the *same* joint space
/// (a CLAP-style dual encoder).
///
/// No `Send`/`Sync` bound — matches [`ImageEmbedder`](crate::image_embed::ImageEmbedder) and
/// [`TextEmbedder`](crate::text_embed::TextEmbedder) (a candle provider holds candle `Tensor`s the
/// worker runs inside one blocking task), not the `Send + Sync`
/// [`VoiceEmbedder`](crate::voice_embed::VoiceEmbedder).
pub trait AudioEmbedder {
    /// Stable identity + advertised shape, constructible without loading weights.
    fn descriptor(&self) -> &AudioEmbedderDescriptor;

    /// Embed one audio clip into its vector of length
    /// [`AudioEmbedderDescriptor::embedding_dim`], in the joint audio-text space
    /// [`AudioEmbedderDescriptor::space`]. The returned vector is **L2-normalized** — CLAP's native
    /// retrieval feature — so cosine similarity is a plain dot product.
    fn embed(&self, audio: &AudioTrack) -> Result<Vec<f32>>;

    /// Embed one text string into a vector of the *same* length and *same* joint space as
    /// [`embed`](Self::embed), so `cosine(embed_text(query), embed(clip))` ranks clips by semantic
    /// match. Also **L2-normalized**.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of audio clips. The default maps [`embed`](Self::embed) over the slice; a
    /// provider can override with a batched forward. Order matches the input.
    fn embed_batch(&self, audios: &[AudioTrack]) -> Result<Vec<Vec<f32>>> {
        audios.iter().map(|audio| self.embed(audio)).collect()
    }

    /// Embed a batch of texts. The default maps [`embed_text`](Self::embed_text) over the slice; a
    /// provider can override with a batched forward. Order matches the input.
    fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed_text(text)).collect()
    }
}

/// An audio embedder's stable identity + advertised shape. Mirrors
/// [`ImageEmbedderDescriptor`](crate::image_embed::ImageEmbedderDescriptor) field-for-field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioEmbedderDescriptor {
    /// Stable id (e.g. `"clap_htsat_unfused"`).
    pub id: &'static str,
    /// Provider family (`"audio-embed"`).
    pub family: &'static str,
    /// Tensor backend that registered this embedder (`"mlx"` | `"candle"`); used by the worker's
    /// per-backend capability advertisement.
    pub backend: &'static str,
    /// Dimensionality of the returned embedding (512 for CLAP's projection space).
    pub embedding_dim: usize,
    /// The joint embedding-space identifier (e.g. `"clap-htsat-unfused"`). Both the audio and text
    /// vectors this provider returns live in this space; two vectors are only comparable when their
    /// `space` matches — it guards retrieval math against silently mixing vectors from different
    /// encoders.
    pub space: &'static str,
    /// Whether this embedder only runs on macOS (an MLX implementation); a candle implementation
    /// sets this `false`.
    pub mac_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory embedder: the audio vector encodes the track's sample count and the text
    /// vector its char count, so both `embed` paths are exercised without a tensor backend.
    struct StubAudioEmbedder {
        descriptor: AudioEmbedderDescriptor,
    }

    impl AudioEmbedder for StubAudioEmbedder {
        fn descriptor(&self) -> &AudioEmbedderDescriptor {
            &self.descriptor
        }
        fn embed(&self, audio: &AudioTrack) -> Result<Vec<f32>> {
            Ok(vec![
                audio.samples.len() as f32;
                self.descriptor.embedding_dim
            ])
        }
        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32; self.descriptor.embedding_dim])
        }
    }

    fn descriptor() -> AudioEmbedderDescriptor {
        AudioEmbedderDescriptor {
            id: "stub",
            family: "audio-embed",
            backend: "candle",
            embedding_dim: 512,
            space: "test-space",
            mac_only: false,
        }
    }

    fn track(samples: usize) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: 48_000,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn embed_audio_and_text_share_dim_and_space() {
        let e = StubAudioEmbedder {
            descriptor: descriptor(),
        };
        let a = e.embed(&track(3)).unwrap();
        let t = e.embed_text("clip").unwrap();
        assert_eq!(a.len(), 512);
        assert_eq!(
            t.len(),
            a.len(),
            "audio and text vectors share the joint dim"
        );
        assert_eq!(a[0], 3.0);
        assert_eq!(t[0], 4.0);
        assert_eq!(e.descriptor().space, "test-space");
    }

    #[test]
    fn default_batch_helpers_map_and_preserve_order() {
        let e = StubAudioEmbedder {
            descriptor: descriptor(),
        };
        let audio = e.embed_batch(&[track(1), track(2)]).unwrap();
        assert_eq!(audio.len(), 2);
        assert_eq!(audio[0][0], 1.0);
        assert_eq!(audio[1][0], 2.0);
        let text = e.embed_text_batch(&["a", "abcd"]).unwrap();
        assert_eq!(text.len(), 2);
        assert_eq!(text[0][0], 1.0);
        assert_eq!(text[1][0], 4.0);
    }
}
