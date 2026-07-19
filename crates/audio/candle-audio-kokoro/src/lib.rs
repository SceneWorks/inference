//! # candle-audio-kokoro
//!
//! **Kokoro-82M** text-to-speech provider for the SceneWorks Candle audio lane — the first
//! real audio [`gen_core::Generator`] (sc-12836, epic sc-12833). One candle implementation
//! serves `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the audio composition
//! root (`candle-audio-catalog`), per `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The port
//!
//! Kokoro (hexgrad/Kokoro-82M, Apache-2.0 weights + reference code) is a StyleTTS2-derived
//! architecture: a PLBERT (ALBERT) phoneme encoder, a prosody predictor (duration / F0 /
//! energy heads over BiLSTM + AdaIN residual stacks), a text encoder, and an iSTFT-Net-style
//! vocoder (harmonic-plus-noise source, AdaIN Snake resblocks, magnitude/phase head +
//! inverse STFT). This crate is a faithful component port onto the workspace's pinned candle
//! revision — module by module against the reference `model.py` / `modules.py` /
//! `istftnet.py`:
//!
//! - [`albert`] — the shared-layer ALBERT encoder,
//! - [`text_encoder`] / [`predictor`] — the StyleTTS2 text/prosody stacks,
//! - [`decoder`] — the styled decoder + vocoder head (harmonic source and the tiny n_fft=20
//!   STFT pair run as host `f32` DSP),
//! - [`weights`] — the five-section pickle checkpoint (old-style weight-norm resolved at
//!   load) and the voice style-vector packs,
//! - [`g2p`] — pure-Rust phonemization (misaki-rs lexicons + the exact US/GB post-processing
//!   Kokoro was trained with; **no espeak, no Python**),
//! - [`pipeline`] / [`model`] — the assembled synthesis pipeline and the
//!   [`gen_core::Generator`] adapter registered under **`kokoro_82m`**,
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (a validated
//!   passthrough; Kokoro snapshots carry no tokenizer.json for the LLM preparer to demand).
//!
//! Weights resolve through the audio lane's pinned-SHA hub path
//! ([`model::resolve_pinned_snapshot`], F-029): `hexgrad/Kokoro-82M` at an immutable commit,
//! never a mutable ref.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod albert;
pub mod config;
pub mod decoder;
pub mod g2p;
pub mod model;
pub mod nn;
pub mod pipeline;
pub mod predictor;
pub mod prepare;
pub mod text_encoder;
pub mod weights;

pub use config::KokoroConfig;
pub use g2p::{EnglishVariant, KokoroG2p};
pub use model::{
    descriptor, load, resolve_pinned_snapshot, DEFAULT_VOICE, HUB_REPO, HUB_REVISION, LANGUAGES,
    MODEL_ID, REGISTRATION, VOICES,
};
pub use pipeline::KokoroPipeline;

pub use model::{WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY};

/// This crate's model-weight-license entries for catalog aggregation (sc-13332) — one row keyed by
/// [`MODEL_ID`]. The audio catalog concatenates every provider's slice into the model-licenses
/// manifest SceneWorks lists on its end-product licenses page.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];

/// Add the Kokoro generator to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build the complete explicit Kokoro provider catalog (this crate's own surface).
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
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["kokoro_82m"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
