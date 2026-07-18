//! Supported Apple-silicon runtime: explicit MLX media, LLM, and snapshot-preparer catalogs.

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
/// audio composition root is owned by the audio lane (sc-12835), never by `mlx-gen-catalog`.
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

/// The bundle's explicit audio registry. Empty until the Candle audio composition root lands
/// (sc-12835/12836) — its `register_providers` call slots in here, exactly like the media
/// catalog's, without touching `mlx-gen-catalog`.
#[cfg(feature = "media")]
fn audio_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    gen_core::ProviderRegistryBuilder::new().build()
}

/// Build the complete validated macOS runtime composition.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    #[cfg(feature = "media")]
    {
        RuntimeCatalog::try_new_with_audio(
            PLATFORM,
            BACKEND,
            media_registry(),
            mlx_llm::text_registry(),
            mlx_llm::snapshot_preparer_registry(),
            AUDIO_BACKEND,
            audio_registry(),
        )
    }

    // The LLM-only composition profile ships neither the media nor the audio graph.
    #[cfg(not(feature = "media"))]
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
        // The audio lane is declared Candle-native on this mlx bundle (sc-12901); its provider
        // surface stays empty until sc-12835/12836 register the real audio catalog.
        #[cfg(feature = "media")]
        {
            assert_eq!(snapshot.audio_backend.as_deref(), Some(super::AUDIO_BACKEND));
            assert!(snapshot.audio_generator_ids.is_empty());
        }
        #[cfg(not(feature = "media"))]
        assert_eq!(snapshot.audio_backend, None);
        #[cfg(feature = "media")]
        assert_eq!(mlx_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "macos");
    }

    /// The sc-12901 mechanics proof on the real bundle: the complete mlx media catalog validates
    /// alongside a `candle` audio generator carried in the dedicated audio section. Real audio
    /// providers replace this stub in sc-12835/12836; the stub's modality reuses an existing
    /// variant until `Modality::Audio` lands (sc-12834) — catalog validation does not inspect it.
    #[cfg(feature = "media")]
    #[test]
    fn mlx_bundle_validates_with_candle_audio_provider() {
        use super::gen_core;

        fn stub_audio_descriptor() -> gen_core::ModelDescriptor {
            gen_core::ModelDescriptor {
                id: "stub-candle-audio",
                family: "test-audio",
                backend: super::AUDIO_BACKEND,
                modality: gen_core::Modality::Image,
                capabilities: gen_core::Capabilities {
                    min_size: 1,
                    max_size: 4096,
                    max_count: 1,
                    ..Default::default()
                },
            }
        }
        fn stub_audio_load(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
            Err(gen_core::Error::Msg("stub audio provider".to_string()))
        }

        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(gen_core::ModelRegistration {
                descriptor: stub_audio_descriptor,
                load: stub_audio_load,
                footprint: None,
            })
            .build();

        let catalog = super::RuntimeCatalog::try_new_with_audio(
            super::PLATFORM,
            super::BACKEND,
            mlx_gen_catalog::provider_registry(),
            super::llm::text_registry(),
            super::llm::snapshot_preparer_registry(),
            super::AUDIO_BACKEND,
            audio,
        )
        .unwrap();

        assert_eq!(catalog.backend(), "mlx");
        assert_eq!(catalog.audio_backend(), Some("candle"));
        let snapshot = catalog.snapshot();
        assert!(snapshot.generator_ids.len() > 50);
        assert_eq!(snapshot.audio_generator_ids, ["stub-candle-audio"]);
    }
}
