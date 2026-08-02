//! # candle-audio-moss-sfx
//!
//! **MOSS-SoundEffect v2.0** SFX / ambience provider for the SceneWorks Candle audio lane —
//! the audio lane's first *diffusion* [`gen_core::Generator`] (sc-12841, epic sc-12833),
//! following the Kokoro TTS walking skeleton (sc-12836). One candle implementation serves
//! `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the audio composition root
//! (`candle-audio-catalog`), per `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The port
//!
//! MOSS-SoundEffect v2.0 (OpenMOSS-Team, Apache-2.0 weights + code) is a text-to-audio
//! flow-matching pipeline: a **Qwen3-1.7B** causal LM as text encoder (last hidden states), a
//! **1.3B Wan-style 1-D DiT** denoising 128-channel continuous latents at 50 frames/second,
//! and a **continuous DAC VAE** decoding those latents to 48 kHz mono waveforms — up to 30 s
//! with 0.1 s-granular duration control via a textual `" duration: Xs"` prompt suffix. This
//! crate is a faithful component port onto the workspace's pinned candle revision — module by
//! module against the reference `moss_soundeffect_v2` sources:
//!
//! - [`config`] — the diffusers-style snapshot configs (`model_index.json` + per-component),
//! - [`text`] — prompt cleaning, the duration suffix, and Qwen tokenization,
//! - [`qwen3`] — the stateless Qwen3 text encoder (post-final-norm hidden states),
//! - [`dit`] — the Wan-audio DiT (adaLN blocks, dim-wide q/k RMSNorm, interleaved 1-D RoPE),
//! - [`sampler`] — the flow-matching σ schedule (shift 5.0, `extra_one_step`) + Euler update,
//! - [`vae`] — the continuous DAC decoder (weight-norm resolution, Snake, odd-stride
//!   `output_padding`, final tanh),
//! - [`pipeline`] / [`model`] — the assembled synthesis pipeline and the
//!   [`gen_core::Generator`] adapter registered under **`moss_sfx_v2`**,
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (a validated
//!   passthrough; a diffusers-style snapshot has no top-level config/tokenizer for the LLM
//!   preparer to probe).
//!
//! Weights are supplied as an explicit passed-in snapshot on the `gen_core::LoadSpec`:
//! `OpenMOSS-Team/MOSS-SoundEffect-v2.0`, staged locally and never self-fetched (epic 13657). The
//! `HUB_REPO`@`HUB_REVISION` pin is retained as the provenance record of that checkpoint.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod config;
pub mod dit;
pub mod model;
pub mod pipeline;
pub mod prepare;
pub mod qwen3;
pub mod sampler;
pub mod text;
pub mod vae;

pub use model::{
    descriptor, load, HUB_REPO, HUB_REVISION, LANGUAGES, MAX_DURATION_SECS, MODEL_ID, REGISTRATION,
    SAMPLE_RATE,
};
pub use pipeline::MossSfxPipeline;

pub use model::{COMPONENT_KEY, COMPONENT_LICENSE};

/// This crate's schema-3 component licence rows — one per loaded artifact (sc-16663). The audio
/// catalog concatenates every provider crate's slice into the licence table it ships.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[model::COMPONENT_LICENSE];

/// The provider→component mapping this crate contributes. A provider's effective terms are
/// **derived** from these components by [`gen_core::provider_terms`] — never hand-authored — so
/// they cannot drift from the component rows they summarize.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[gen_core::ProviderComponents {
    provider_id: model::MODEL_ID,
    components: &[model::COMPONENT_KEY],
}];

/// Add the MOSS-SoundEffect generator to an explicit audio registry builder (catalog
/// composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build the complete explicit MOSS-SoundEffect provider catalog (this crate's own surface).
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
        assert_eq!(ids, ["moss_sfx_v2"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
