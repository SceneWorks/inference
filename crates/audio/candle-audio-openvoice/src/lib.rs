//! # candle-audio-openvoice
//!
//! **OpenVoice V2 tone-color voice conversion** provider for the SceneWorks Candle audio lane — the
//! first real [`gen_core::AudioTransform`] (sc-13223, epic sc-12833), which **releases the sc-12839
//! gate**. One candle implementation serves `runtime-cpu`, `runtime-cuda`, and `runtime-macos`
//! through the audio composition root (`candle-audio-catalog`), per
//! `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The port
//!
//! MyShell's OpenVoice V2 (MIT — commercial OK) converts *any* speaker's speech into a target
//! voice's timbre while preserving the source's content and prosody. Its converter is a
//! flow-based VITS-family model (`myshell-ai/OpenVoiceV2/converter`): a **tone-color reference
//! encoder** ([`reference_encoder`]) turns a reference clip's linear spectrogram into a speaker
//! embedding `g`, and the **converter** ([`converter`]) — a VITS posterior encoder (`enc_q`), a
//! four-flow normalizing flow, and a HiFi-GAN decoder (`dec`) — transfers timbre by running the
//! flow forward on the source's tone color and reverse on the target's. Under the checkpoint's
//! `zero_g = true`, the posterior encoder and decoder are conditioned on a zeroed `g`, so the
//! entire timbre transfer lives in the flow. This crate is a faithful component port onto the
//! workspace's pinned candle revision — module by module against the upstream `models.py` /
//! `modules.py` / `api.py`, **no Python at runtime**:
//!
//! - [`spectrogram`] — the exact `spectrogram_torch` linear front-end (center=False, reflect-pad,
//!   in-sqrt epsilon) + resampling, as self-contained host `f32` DSP with a small radix-2 FFT,
//! - [`reference_encoder`] — the `ref_enc` Conv2d+GRU tone-color extractor,
//! - [`converter`] — the `enc_q` / `flow` / `dec` VITS converter,
//! - [`weights`] — the single-section pickle checkpoint (`model` state dict; old-style weight-norm
//!   resolved at load),
//! - [`pipeline`] — the assembled load + `extract_se` + `voice_conversion` flow,
//! - [`model`] — the [`gen_core::AudioTransform`] adapter registered under **`openvoice_v2`**, its
//!   `descriptor`/`load` entry points, and the pinned-SHA hub resolution
//!   ([`model::resolve_pinned_snapshot`]),
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (a validated passthrough;
//!   OpenVoice converter snapshots carry no tokenizer.json for the LLM preparer to demand).
//!
//! The **target tone-color reference** is supplied per request through the additive
//! [`gen_core::AudioTransformRequest::target_reference`] field (sc-13223) — a reference-based
//! converter, unlike a weight-baked RVC-style single-target model.
//!
//! Weights resolve through the audio lane's pinned-SHA hub path
//! ([`model::resolve_pinned_snapshot`], F-029): `myshell-ai/OpenVoiceV2` at an immutable commit,
//! never a mutable ref.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod config;
pub mod converter;
pub mod model;
pub mod pipeline;
pub mod prepare;
pub mod reference_encoder;
pub mod spectrogram;
pub mod weights;

pub use model::{
    descriptor, load, resolve_pinned_snapshot, HUB_REPO, HUB_REVISION, MODEL_ID,
    OUTPUT_SAMPLE_RATE, REGISTRATION,
};
pub use pipeline::OpenVoicePipeline;

pub use model::{WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY};

/// This crate's model-weight-license entries for catalog aggregation (sc-13332) — one row keyed by
/// [`MODEL_ID`]. The audio catalog concatenates every provider's slice into the model-licenses
/// manifest SceneWorks lists on its end-product licenses page.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];

/// Add the OpenVoice V2 voice-conversion transform to an explicit audio registry builder (catalog
/// composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_audio_transform(model::REGISTRATION)
}

/// Build the complete explicit OpenVoice-V2 provider catalog (this crate's own surface).
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
            .audio_transforms()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["openvoice_v2"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
