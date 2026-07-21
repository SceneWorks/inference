//! # candle-audio-whisper
//!
//! **OpenAI Whisper** automatic-speech-recognition provider for the SceneWorks Candle audio lane —
//! the first real audio→text [`gen_core::Transcriber`] (sc-12850, epic sc-12833). A `Transcriber`
//! is the audio sibling of the [`gen_core::Captioner`]: both consume media and emit text rather
//! than synthesizing media, so ASR gets its own trait, not the [`gen_core::Generator`]. One candle
//! implementation serves `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the audio
//! composition root (`candle-audio-catalog`), per `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The reuse (not a re-port)
//!
//! The Whisper encoder + decoder + log-mel front-end are candle's
//! ([`candle_transformers::models::whisper`]) at the workspace's pinned candle revision — reused
//! wholesale per the epic DoD. This crate owns only the gen-core adapter:
//!
//! - [`mel`] — the host front-end (downmix / linear-resample to 16 kHz / Slaney mel projection over
//!   the bundled `melfilters.bytes`),
//! - [`decode`] — the autoregressive decode policy (the `<|sot|>`+language+task+timestamp prompt,
//!   greedy-or-temperature sampling honoring the request knobs, suppressed-token mask, cooperative
//!   cancellation, and the timestamp-token → [`gen_core::TranscriptSegment`] parse),
//! - [`model`] — the [`gen_core::Transcriber`] adapter registered under **`whisper_base`**; weights
//!   are passed in on the `gen_core::LoadSpec` (staged locally, never self-fetched, epic 13657),
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (a validated passthrough;
//!   Whisper snapshots describe an ASR arch the LLM preparer should not own).
//!
//! Weights resolve through the audio lane's pinned-SHA hub path: `openai/whisper-base`
//! (Apache-2.0 — the checkpoint's model-card license, distinct from the MIT license on OpenAI's
//! Whisper *source* repository) at an immutable commit, never a mutable ref. **No Python at
//! runtime.**

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod decode;
pub mod mel;
pub mod model;
pub mod prepare;

pub use model::{descriptor, load, HUB_REPO, HUB_REVISION, MODEL_ID, REGISTRATION};
pub use model::{WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY};

/// This crate's model-weight-license entries for catalog aggregation (sc-13332) — one row keyed by
/// [`MODEL_ID`]. The audio catalog concatenates every provider's slice into the model-licenses
/// manifest SceneWorks lists on its end-product licenses page.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];

/// Add the Whisper transcriber to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_transcriber(model::REGISTRATION)
}

/// Build the complete explicit Whisper provider catalog (this crate's own surface).
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
            .transcribers()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["whisper_base"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
