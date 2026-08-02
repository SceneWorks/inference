//! # candle-audio-acestep
//!
//! **ACE-Step 1.5** text-to-music (+ lyrics) provider for the SceneWorks Candle audio lane — the
//! audio lane's music/song [`gen_core::Generator`] (sc-12842, epic sc-12833), following the MOSS
//! SFX diffusion provider (sc-12841). One candle implementation serves `runtime-cpu`,
//! `runtime-cuda`, and `runtime-macos` through `candle-audio-catalog`.
//!
//! ## The port
//!
//! ACE-Step 1.5 (ACE Studio + StepFun, MIT weights + code) is a flow-matching music foundation
//! model. This crate ports the diffusers `AceStepPipeline` (v0.39.0) text-to-music path natively
//! onto the workspace's pinned candle revision:
//!
//! - [`config`] — the diffusers-style snapshot configs (`model_index.json` + per-component),
//! - [`text`] — the prompt/metadata weave + lyric tokenization,
//! - [`qwen`] — the Qwen3-Embedding-0.6B text encoder (prompt hidden states + lyric token lookup),
//! - [`condition`] — the `AceStepConditionEncoder` (lyric + timbre encoders + fusion),
//! - [`dit`] — the ~2B `AceStepTransformer1DModel` (GQA + half-split RoPE + AdaLN-Zero + cross-attn,
//!   alternating sliding/full self-attention),
//! - [`scheduler`] — the flow-match shifted/turbo σ schedule + Euler update,
//! - [`vae`] — the stereo `AutoencoderOobleck` decoder (Snake, weight-norm folding),
//! - [`pipeline`] / [`model`] — the assembled synthesis pipeline and the [`gen_core::Generator`]
//!   adapter registered under **`acestep_v15_turbo`**,
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (validated passthrough).
//!
//! Weights are supplied as an explicit passed-in snapshot on the [`gen_core::LoadSpec`]:
//! `ACE-Step/acestep-v15-xl-turbo-diffusers` staged locally, never self-fetched (epic 13657). The
//! [`model::HUB_REPO`]@[`model::HUB_REVISION`] pin is retained as the provenance record of that
//! checkpoint. ACE-Step ships **its own** Oobleck VAE (`vae/`), so the Stability-licensed DiffRhythm
//! VAE is not pulled.
//!
//! ## Stems
//!
//! ACE-Step 1.5's text-to-music path renders a single **stereo mix** — separated stems
//! (vocals/drums/bass) are produced only by the reference's audio-to-audio editing tasks
//! (`extract`/`lego`/`complete`), which require input audio. This provider therefore emits the mix
//! and leaves [`gen_core::AudioTrack::stems`] empty (never faked); that field is the additive
//! carrier a future stem-emitting model would populate.
//!
//! ## Fidelity (sc-12842)
//!
//! `config` / `scheduler` / `text` / `qwen` are validated by offline unit tests. The acoustic core
//! (`dit` / `condition` / `vae`) is a structural port of the diffusers reference; the exact
//! condition-encoder fusion key layout, the sliding-window mask geometry, and the Oobleck decoder
//! channel/weight-norm storage are the points that need reference-activation validation against the
//! Python pipeline before the acoustic output is certified bit-faithful — which is what the
//! `#[ignore]`d real-weight conformance test exists to prove.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod condition;
pub mod config;
pub mod dit;
pub mod model;
pub mod pipeline;
pub mod prepare;
pub mod qwen;
pub mod scheduler;
pub mod text;
pub mod tokenizer;
pub mod vae;

pub use model::{
    cover_module_paths, descriptor, load, CHANNELS, COVER_COMPONENT_ID, HUB_REPO, HUB_REVISION,
    LANGUAGES, MAX_DURATION_SECS, MODEL_ID, REGISTRATION, SAMPLE_RATE,
};
pub use pipeline::{AceStepPipeline, CoverModules};

pub use model::{
    COMPONENT_KEY, COMPONENT_LICENSE, SFT_AUDIO_TOKENIZER_COMPONENT_LICENSE,
    SFT_AUDIO_TOKEN_DETOKENIZER_COMPONENT_LICENSE, SFT_HUB_REPO, SFT_HUB_REVISION,
    SFT_TRANSFORMER_COMPONENT_LICENSE, TEXT_ENCODER_COMPONENT_LICENSE,
};

/// This crate's schema-3 component licence rows (sc-16663) — one per loaded artifact: the turbo
/// primary, its bundled Qwen3-Embedding-0.6B `text_encoder`, and the three cover-only sft modules.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[
    model::COMPONENT_LICENSE,
    model::TEXT_ENCODER_COMPONENT_LICENSE,
    model::SFT_AUDIO_TOKENIZER_COMPONENT_LICENSE,
    model::SFT_AUDIO_TOKEN_DETOKENIZER_COMPONENT_LICENSE,
    model::SFT_TRANSFORMER_COMPONENT_LICENSE,
];

/// The provider→component mapping this crate contributes. The provider's effective terms are
/// **derived** from these five rows by [`gen_core::provider_terms`], replacing v2's hand-authored
/// composite row: that composite said "MIT" and carried the bundled Apache-2.0 encoder only as a
/// prose note, so the Apache-2.0 notice and flow-down duties never reached a joinable field.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[gen_core::ProviderComponents {
    provider_id: model::MODEL_ID,
    components: &[
        model::COMPONENT_KEY,
        model::TEXT_ENCODER_COMPONENT_KEY,
        model::SFT_AUDIO_TOKENIZER_COMPONENT_KEY,
        model::SFT_AUDIO_TOKEN_DETOKENIZER_COMPONENT_KEY,
        model::SFT_TRANSFORMER_COMPONENT_KEY,
    ],
}];

/// Add the ACE-Step generator to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build the complete explicit ACE-Step provider catalog (this crate's own surface).
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
        assert_eq!(ids, ["acestep_v15_turbo"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
