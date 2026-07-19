//! Supported Apple-silicon runtime: explicit MLX media, LLM, and snapshot-preparer catalogs.

#[cfg(feature = "audio")]
pub use candle_audio_catalog::audio;
#[cfg(feature = "media")]
pub use mlx_gen_catalog::media;
pub use mlx_llm as llm;
pub use runtime_catalog::{core_llm, gen_core, RuntimeCatalog, RuntimeCatalogSnapshot};

/// The MLX backend crates this platform owns, re-exported from the media catalog
/// (available under the default `media` feature).
#[cfg(feature = "media")]
pub mod providers {
    pub use mlx_gen_catalog::providers::*;
}

/// Platform label for this bundle; matches `RuntimeCatalog::platform`.
pub const PLATFORM: &str = "macos";
/// The single tensor backend every media, LLM, and snapshot-preparer provider in this bundle uses.
pub const BACKEND: &str = "mlx";
/// The single tensor backend of this bundle's **audio lane** (sc-12901,
/// `docs/architecture/audio-backend-strategy.md`): audio generation is Candle-native on every
/// platform, so the mlx macOS bundle carries its audio generators on `candle` through the
/// catalog's dedicated audio section. This is the one sanctioned cross-backend seam — it does
/// not relax the mlx-only invariant on the media, LLM, or snapshot-preparer registries, and the
/// audio composition root (`candle-audio-catalog`, sc-12835) is owned by the audio lane, never
/// by `mlx-gen-catalog`. Shipped under the additive `audio` feature.
pub const AUDIO_BACKEND: &str = "candle";
/// Target triples this bundle is supported on.
pub const SUPPORTED_TARGET_TRIPLES: &[&str] = &["aarch64-apple-darwin"];
/// Native (non-Cargo) prerequisites required to build and run this bundle.
pub const NATIVE_PREREQUISITES: &[&str] = &["macOS 26.2+", "Xcode Metal toolchain"];

fn media_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    #[cfg(feature = "media")]
    {
        mlx_gen_catalog::provider_registry()
    }

    #[cfg(not(feature = "media"))]
    {
        gen_core::ProviderRegistryBuilder::new().build()
    }
}

/// The bundle's explicit audio lane (sc-12835): the complete Candle audio catalog from the audio
/// composition root — never `mlx-gen-catalog` — plus the lane's **candle** snapshot preparer
/// carried in the lane. The main preparer registry stays mlx-only (the single-backend invariant
/// is unchanged); without the lane's preparer, candle audio model snapshots could not be
/// prepared on macOS at all (audio-backend-strategy.md, "Consequences"). Since sc-12836 the lane
/// preparer is the catalog's composed `candle` registration — audio-shaped snapshots (Kokoro's
/// pickle layout) take the audio path; everything else delegates to `candle-llm`'s preparer
/// unchanged (candle-llm now arrives via the audio catalog, which owns the composition).
#[cfg(feature = "audio")]
fn audio_lane() -> runtime_catalog::AudioLane {
    runtime_catalog::AudioLane {
        backend: AUDIO_BACKEND,
        generators: candle_audio_catalog::provider_registry(),
        preparers: candle_audio_catalog::snapshot_preparer_registry(),
    }
}

/// Build the complete validated macOS runtime composition.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    #[cfg(feature = "audio")]
    {
        RuntimeCatalog::try_new_with_audio(
            PLATFORM,
            BACKEND,
            media_registry(),
            mlx_llm::text_registry(),
            mlx_llm::snapshot_preparer_registry(),
            audio_lane(),
        )
    }

    // Without the `audio` feature no audio lane is declared (media-only or LLM-only profiles).
    #[cfg(not(feature = "audio"))]
    {
        RuntimeCatalog::try_new(
            PLATFORM,
            BACKEND,
            media_registry(),
            mlx_llm::text_registry(),
            mlx_llm::snapshot_preparer_registry(),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke_catalog_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();
        assert_eq!(snapshot.platform, "macos");
        assert_eq!(snapshot.backend, "mlx");
        #[cfg(feature = "media")]
        assert!(snapshot.generator_ids.len() > 50);
        #[cfg(not(feature = "media"))]
        assert!(snapshot.generator_ids.is_empty());
        assert_eq!(snapshot.text_llm_ids, ["mlx-llama", "mlx-joycaption"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);
        // The audio lane is declared Candle-native on this mlx bundle (sc-12901) — the
        // sanctioned cross-backend seam. Its ordered id surface is the audio catalog's —
        // shipped generators kokoro_82m (sc-12836), moss_sfx_v2 (sc-12841), acestep_v15_turbo (sc-12842), plus the voice-cloning
        // identity embedder chatterbox_ve (sc-12844); later stories extend in catalog order. The
        // lane carries the composed candle preparer (sc-12835/sc-12836) while the main preparer
        // registry stays mlx-only.
        #[cfg(feature = "audio")]
        {
            assert_eq!(
                snapshot.audio_backend.as_deref(),
                Some(super::AUDIO_BACKEND)
            );
            assert_eq!(
                snapshot.audio_generator_ids,
                ["kokoro_82m", "moss_sfx_v2", "acestep_v15_turbo"]
            );
            assert_eq!(snapshot.audio_voice_embedder_ids, ["chatterbox_ve"]);
            assert_eq!(snapshot.audio_transform_ids, ["openvoice_v2"]);
            assert_eq!(snapshot.audio_transcriber_ids, ["whisper_base"]);
            assert_eq!(snapshot.audio_snapshot_preparer_backends, ["candle"]);
        }
        #[cfg(not(feature = "audio"))]
        {
            assert_eq!(snapshot.audio_backend, None);
            assert!(snapshot.audio_generator_ids.is_empty());
            assert!(snapshot.audio_voice_embedder_ids.is_empty());
            assert!(snapshot.audio_transform_ids.is_empty());
            assert!(snapshot.audio_transcriber_ids.is_empty());
        }
        #[cfg(feature = "media")]
        assert_eq!(mlx_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "macos");
    }

    /// The sc-12835 acceptance smoke on the sanctioned cross-backend seam: the complete mlx
    /// media catalog validates alongside a (test-only) dummy `candle` audio Generator registered
    /// through the audio composition root's builder — the exact seam a real provider crate's
    /// registration uses (sc-12836+) — and the dummy resolves in the bundle catalog with
    /// backend=candle and `Modality::Audio` while the lane carries the candle preparer.
    #[cfg(all(feature = "media", feature = "audio"))]
    #[test]
    fn dummy_audio_generator_resolves_through_the_bundle_audio_lane() {
        use super::gen_core;

        fn dummy_audio_descriptor() -> gen_core::ModelDescriptor {
            gen_core::ModelDescriptor {
                id: "dummy-audio",
                family: "test-audio",
                backend: super::AUDIO_BACKEND,
                modality: gen_core::Modality::Audio,
                capabilities: gen_core::Capabilities {
                    min_size: 1,
                    max_size: 4096,
                    max_count: 1,
                    ..Default::default()
                },
            }
        }
        fn dummy_audio_load(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
            Err(gen_core::Error::Msg("dummy audio provider".to_string()))
        }

        let audio =
            candle_audio_catalog::register_providers(gen_core::ProviderRegistryBuilder::new())
                .register_generator(gen_core::ModelRegistration {
                    descriptor: dummy_audio_descriptor,
                    load: dummy_audio_load,
                    footprint: None,
                })
                .build();

        let catalog = super::RuntimeCatalog::try_new_with_audio(
            super::PLATFORM,
            super::BACKEND,
            mlx_gen_catalog::provider_registry(),
            super::llm::text_registry(),
            super::llm::snapshot_preparer_registry(),
            runtime_catalog::AudioLane {
                backend: super::AUDIO_BACKEND,
                generators: audio,
                preparers: candle_audio_catalog::snapshot_preparer_registry(),
            },
        )
        .unwrap();

        assert_eq!(catalog.backend(), "mlx");
        assert_eq!(catalog.audio_backend(), Some("candle"));
        let descriptor = catalog
            .audio()
            .unwrap()
            .generators()
            .map(|r| (r.descriptor)())
            .find(|d| d.id == "dummy-audio")
            .expect("dummy audio generator resolves in the bundle catalog");
        assert_eq!(descriptor.backend, "candle");
        assert!(matches!(descriptor.modality, gen_core::Modality::Audio));
        let snapshot = catalog.snapshot();
        assert!(snapshot.generator_ids.len() > 50);
        assert_eq!(
            snapshot.audio_generator_ids,
            [
                "kokoro_82m",
                "moss_sfx_v2",
                "acestep_v15_turbo",
                "dummy-audio"
            ]
        );
        assert_eq!(snapshot.audio_snapshot_preparer_backends, ["candle"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);
    }
}
