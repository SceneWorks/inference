//! Explicit, complete provider catalog for the SceneWorks **Candle audio lane** — the audio
//! composition root (sc-12835, `docs/architecture/audio-backend-strategy.md`).
//!
//! Audio generation is Candle-native on every platform: this one catalog supplies the audio
//! section of `runtime-cpu`, `runtime-cuda`, **and** `runtime-macos` (where it is the
//! sanctioned cross-backend seam beside the mlx media graph). Provider crates own their
//! registrations; this crate owns only composition and stable ordering, mirroring
//! `candle-gen-catalog`. It never touches the media catalogs — bundle inclusion of the audio
//! lane is a deliberate per-bundle edit through `runtime-catalog`'s `AudioLane`.
//!
//! The audio lane is **generators-only** (enforced by `runtime-catalog::validate_audio`):
//! audio providers implement the ordinary [`gen_core::Generator`] contract with
//! [`gen_core::Modality::Audio`] descriptors — no new trait, no linker discovery.

pub use candle_audio as audio;
pub use candle_audio::gen_core;
pub use candle_audio::gen_core::{ProviderRegistry, ProviderRegistryBuilder};

/// The single tensor backend every provider in this catalog registers under — `candle` on
/// every platform per the audio backend strategy. Bundles declare the same value as their
/// audio lane's backend; `runtime-catalog` validates every descriptor against it.
pub const AUDIO_BACKEND: &str = "candle";

/// Complete audio provider package surface owned by the Candle audio lane.
///
/// Empty at this release: the composition seam lands ahead of the first provider so every
/// bundle already resolves the audio lane through one root. The first shipped provider
/// (Kokoro TTS, sc-12836) adds its crate re-export here alongside its registration below —
/// the same two-line pattern `candle-gen-catalog::providers` uses.
pub mod providers {}

/// Add every provider shipped by the Candle audio lane to an explicit registry builder, in
/// stable catalog order. Registers nothing at this release (see [`providers`]); each audio
/// provider crate's `register_providers` call slots in here (sc-12836+).
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    registry
}

/// Build the complete explicit Candle audio provider catalog.
pub fn provider_registry() -> gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    /// The ordered audio id surface (the audio twin of candle-gen-catalog's
    /// `complete_catalog_has_stable_conforming_surface`). Pinned **empty** at this release —
    /// sc-12836+ extend this exact assertion with each shipped provider id, in catalog order.
    /// The generators-only / candle-backend / audio-modality sweeps are asserted here too so
    /// a provider that would fail bundle validation is caught in its own family first.
    #[test]
    fn complete_catalog_has_stable_conforming_surface() {
        let registry = super::provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();

        assert_eq!(generators, Vec::<String>::new());
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
        // The audio lane is generators-only; no other provider kind may ever register here.
        assert_eq!(registry.transforms().len(), 0);
        assert_eq!(registry.trainers().len(), 0);
        assert_eq!(registry.captioners().len(), 0);
        assert_eq!(registry.image_embedders().len(), 0);
        assert_eq!(registry.text_embedders().len(), 0);
        // Every audio generator is candle-backed and audio-modality (vacuous while empty;
        // load-bearing from the first registration onward).
        assert!(registry
            .generators()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
        assert!(registry
            .generators()
            .all(|r| matches!((r.descriptor)().modality, super::gen_core::Modality::Audio)));
    }
}
