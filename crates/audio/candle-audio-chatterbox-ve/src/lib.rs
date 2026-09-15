//! # candle-audio-chatterbox-ve
//!
//! **Chatterbox voice-encoder** provider for the SceneWorks Candle audio lane — the first real
//! [`gen_core::VoiceEmbedder`] (sc-12844, epic sc-12833). One candle implementation serves
//! `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the audio composition root
//! (`candle-audio-catalog`), per `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The port
//!
//! Resemble AI's Chatterbox (MIT) clones a voice zero-shot from a few seconds of reference audio.
//! Its pipeline splits cleanly into (1) a **speaker encoder** that maps the reference clip to a
//! speaker-identity vector and (2) an LM + S3Gen decoder that renders text in that voice. This
//! crate is a faithful candle port of component (1) — the `ve.safetensors` voice encoder — as the
//! standalone [`gen_core::VoiceEmbedder`] the sc-12838 contract was designed around: the vector
//! rides [`Conditioning::VoiceEmbedding`](gen_core::generator::Conditioning::VoiceEmbedding) into
//! a cloned-voice TTS [`Generator`](gen_core::Generator) (a later sc-12844 slice), exactly the way
//! an ArcFace face embedding rides into InstantID.
//!
//! The encoder is a GE2E-style speaker verifier (Resemblyzer lineage): a 40-channel 16 kHz mel
//! front-end (librosa Slaney mel, raw power — see [`frontend`]) feeding a 3-layer LSTM + a 256→256
//! projection ([`encoder`]), embeddings averaged over ~1.6 s partial utterances and L2-normalized.
//!
//! - [`frontend`] — the reference-audio → mel-frame preprocessing (resample, loudness-normalize,
//!   STFT, Slaney mel), self-contained host `f32` DSP,
//! - [`encoder`] — the candle LSTM + projection speaker encoder over `ve.safetensors`,
//! - [`model`] — the [`gen_core::VoiceEmbedder`] adapter registered under **`chatterbox_ve`**, its
//!   `descriptor`/`load` entry points; weights are passed in on the `gen_core::LoadSpec` (staged
//!   locally, never self-fetched, epic 13657).
//!
//! Weights are supplied as an explicit passed-in file: `ResembleAI/chatterbox` `ve.safetensors`
//! at an immutable commit, never a mutable ref. The single 5.7 MB `ve.safetensors` file needs no
//! bespoke snapshot preparer (unlike Kokoro's pickle / MOSS's diffusers layouts): it is loaded
//! directly as a `WeightsSource::File`.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod config;
pub mod encoder;
pub mod frontend;
pub mod model;

pub use encoder::{cosine_similarity, l2_normalize};
pub use model::{
    descriptor, load, ChatterboxVoiceEmbedder, HUB_REPO, HUB_REVISION, MODEL_ID, REGISTRATION,
    WEIGHTS_FILE,
};

/// This crate ships **no** component licence row of its own (sc-16663): the `chatterbox_ve` voice embedder loads the
/// same `ResembleAI/chatterbox` artifact the `chatterbox_tts` generator does, and one artifact is
/// one row. That row lives in `candle-audio-chatterbox`; this crate only maps its provider id onto
/// the same component key — duplicating the row is rejected by the table conformance checker.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[];

/// The provider→component mapping this crate contributes. A provider's effective terms are
/// **derived** from these components by [`gen_core::provider_terms`] — never hand-authored — so
/// they cannot drift from the component rows they summarize.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[gen_core::ProviderComponents {
    provider_id: model::MODEL_ID,
    components: &["chatterbox"],
}];

/// Add the Chatterbox voice embedder to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_voice_embedder(model::REGISTRATION)
}

/// Build the complete explicit Chatterbox-VE provider catalog (this crate's own surface).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_resolves_through_an_explicit_registry() {
        let registry = provider_registry().unwrap();
        let ids: Vec<String> = registry
            .voice_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["chatterbox_ve"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
