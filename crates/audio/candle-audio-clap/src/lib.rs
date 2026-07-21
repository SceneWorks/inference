//! # candle-audio-clap
//!
//! **LAION CLAP** semantic audio-text embedding provider for the SceneWorks Candle audio lane — the
//! first real [`gen_core::AudioEmbedder`] (sc-12851, epic sc-12833). An `AudioEmbedder` is the audio
//! parallel of the [`gen_core::ImageEmbedder`]: it maps a whole clip into one vector in a joint
//! audio-text (CLIP-style) space, and — because CLAP is intrinsically joint — also embeds a text
//! query into the *same* space, so cross-modal retrieval ("find the clip matching 'a dog barking'")
//! is a plain cosine similarity. This is deliberately the *semantic* embedder, distinct from the
//! identity [`gen_core::VoiceEmbedder`] (sc-12838).
//!
//! ## The port
//!
//! Unlike the Whisper transcriber (which reuses candle-transformers wholesale), candle-transformers
//! has no CLAP/HTSAT at the pinned revision, so both towers are **ported** on `candle-nn` from the
//! `laion/clap-htsat-unfused` (Apache-2.0) checkpoint, module-for-module against `transformers`
//! `modeling_clap.py`:
//!
//! - [`mel`] — the host front-end (downmix / linear-resample to 48 kHz / slaney log-mel over the
//!   STFT in [`candle_audio::dsp`]),
//! - [`audio`] — the HTSAT (Swin-transformer) audio tower: patch embed → 4 windowed-attention stages
//!   with relative-position bias, shifted-window masks, and patch merging → mean pool,
//! - [`text`] — the RoBERTa text tower + `[CLS]` tanh pooler,
//! - [`model`] — the [`gen_core::AudioEmbedder`] adapter registered under **`clap_htsat_unfused`**
//!   (both towers → `ClapProjectionLayer` → L2-normalized joint vector); weights are passed in on
//!   the `gen_core::LoadSpec` (staged locally, never self-fetched, epic 13657),
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (a validated passthrough).
//!
//! Weights are supplied as an explicit passed-in snapshot: `laion/clap-htsat-unfused`
//! (Apache-2.0) at an immutable commit, never a mutable ref. **No Python at runtime.**

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod audio;
pub mod config;
pub mod mel;
pub mod model;
pub mod prepare;
pub mod text;

pub use model::{descriptor, load, HUB_REPO, HUB_REVISION, MODEL_ID, REGISTRATION};
pub use model::{WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY};

/// This crate's model-weight-license entries for catalog aggregation (sc-13332) — one row keyed by
/// [`MODEL_ID`]. The audio catalog concatenates every provider's slice into the model-licenses
/// manifest SceneWorks lists on its end-product licenses page.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[model::WEIGHT_LICENSE_ENTRY];

/// Add the CLAP audio embedder to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_audio_embedder(model::REGISTRATION)
}

/// Build the complete explicit CLAP provider catalog (this crate's own surface).
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
            .audio_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["clap_htsat_unfused"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn descriptor_advertises_joint_space_and_dim() {
        let d = descriptor();
        assert_eq!(d.id, "clap_htsat_unfused");
        assert_eq!(d.family, "audio-embed");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.embedding_dim, 512);
        assert!(!d.mac_only);
        assert_eq!(d.space, "clap-htsat-unfused");
    }
}
