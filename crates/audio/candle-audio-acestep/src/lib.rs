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
//! Weights resolve through the audio lane's pinned-SHA hub path
//! ([`model::resolve_pinned_snapshot`]): `ACE-Step/acestep-v15-xl-turbo-diffusers` at an immutable
//! commit, never a mutable ref. ACE-Step ships **its own** Oobleck VAE (`vae/`), so the
//! Stability-licensed DiffRhythm VAE is not pulled.
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
    descriptor, load, resolve_cover_modules, resolve_pinned_snapshot, CHANNELS, HUB_REPO,
    HUB_REVISION, LANGUAGES, MAX_DURATION_SECS, MODEL_ID, REGISTRATION, SAMPLE_RATE,
};
pub use pipeline::{AceStepPipeline, CoverModules};

pub use model::{
    AUDIO_TOKENIZER_WEIGHT_LICENSE, AUDIO_TOKEN_DETOKENIZER_WEIGHT_LICENSE, SFT_HUB_REPO,
    SFT_HUB_REVISION, WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY,
};

/// This crate's model-weight-license entries for catalog aggregation (sc-13332, extended sc-13251).
/// The ACE-Step provider is assembled from multiple MIT checkpoints, so it contributes the composite
/// (effective-restriction) row keyed by [`MODEL_ID`] PLUS one per-checkpoint attribution row for
/// each cover-only sft component: the two FSQ modules (`audio_tokenizer`, `audio_token_detokenizer`)
/// and the non-distilled cover DiT (`transformer`). The audio catalog concatenates every provider's
/// slice into the model-licenses manifest SceneWorks lists on its end-product licenses page.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    model::WEIGHT_LICENSE_ENTRY,
    model::WEIGHT_LICENSE_ENTRY_AUDIO_TOKENIZER,
    model::WEIGHT_LICENSE_ENTRY_AUDIO_TOKEN_DETOKENIZER,
    model::WEIGHT_LICENSE_ENTRY_SFT_TRANSFORMER,
];

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
