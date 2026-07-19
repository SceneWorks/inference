//! Supported NVIDIA runtime: explicit Candle CUDA media, LLM, and snapshot-preparer catalogs.

#[cfg(feature = "audio")]
pub use candle_audio_catalog::audio;
#[cfg(feature = "media")]
pub use candle_gen_catalog::media;
pub use candle_llm as llm;
pub use runtime_catalog::{core_llm, gen_core, RuntimeCatalog, RuntimeCatalogSnapshot};

/// The Candle backend crates this platform owns, re-exported from the media catalog
/// (available under the default `media` feature).
#[cfg(feature = "media")]
pub mod providers {
    pub use candle_gen_catalog::providers::*;
}

/// The advanced quant tiers this CUDA runtime surfaces beyond affine `Q4`/`Q8` — the NVFP4 FP4
/// tensor-core tier (epic 11037, sc-11042 Option A) on consumer Blackwell `sm_120`. Re-exported from
/// the media catalog so a product/worker reads the served tier off the runtime bundle; empty when the
/// bundle is built without the media graph. See [`candle_gen_catalog::nvfp4_quant_tiers`].
#[cfg(feature = "media")]
pub use candle_gen_catalog::nvfp4_quant_tiers;

/// Platform label for this bundle; matches `RuntimeCatalog::platform`.
pub const PLATFORM: &str = "cuda";
/// The single tensor backend every media, LLM, and snapshot-preparer provider in this bundle uses.
pub const BACKEND: &str = "candle";
/// The single tensor backend of this bundle's audio lane (sc-12901,
/// `docs/architecture/audio-backend-strategy.md`). Audio is Candle-native on every platform, so
/// here it matches `BACKEND`; the lane is still composed through the catalog's dedicated audio
/// section, owned by the audio composition root (`candle-audio-catalog`, sc-12835) rather than
/// `candle-gen-catalog`, and shipped under the additive `audio` feature.
pub const AUDIO_BACKEND: &str = "candle";
/// Target triples this bundle is supported on.
pub const SUPPORTED_TARGET_TRIPLES: &[&str] =
    &["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"];
/// Native (non-Cargo) prerequisites required to build and run this bundle.
pub const NATIVE_PREREQUISITES: &[&str] = &["NVIDIA CUDA toolkit", "supported NVIDIA driver"];

fn media_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    #[cfg(feature = "media")]
    {
        candle_gen_catalog::provider_registry()
    }

    #[cfg(not(feature = "media"))]
    {
        gen_core::ProviderRegistryBuilder::new().build()
    }
}

/// The bundle's explicit audio lane (sc-12835): the complete Candle audio catalog from the audio
/// composition root, plus the lane's snapshot preparer carried **in the lane** so audio model
/// snapshots are prepared through the same registry shape on every platform. Since sc-12836 the
/// lane preparer is the catalog's composed `candle` registration — audio-shaped snapshots
/// (Kokoro's pickle layout) take the audio path; everything else delegates to `candle-llm`'s
/// preparer unchanged.
#[cfg(feature = "audio")]
fn audio_lane() -> runtime_catalog::AudioLane {
    runtime_catalog::AudioLane {
        backend: AUDIO_BACKEND,
        generators: candle_audio_catalog::provider_registry(),
        preparers: candle_audio_catalog::snapshot_preparer_registry(),
    }
}

/// Build the complete validated CUDA runtime composition.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    #[cfg(feature = "audio")]
    {
        RuntimeCatalog::try_new_with_audio(
            PLATFORM,
            BACKEND,
            media_registry(),
            candle_llm::text_registry(),
            candle_llm::snapshot_preparer_registry(),
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
            candle_llm::text_registry(),
            candle_llm::snapshot_preparer_registry(),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke_catalog_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();
        assert_eq!(snapshot.platform, "cuda");
        assert_eq!(snapshot.backend, "candle");
        #[cfg(feature = "media")]
        assert!(snapshot.generator_ids.len() > 40);
        #[cfg(not(feature = "media"))]
        assert!(snapshot.generator_ids.is_empty());
        assert_eq!(snapshot.text_llm_ids, ["candle-llama", "candle-llava"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["candle"]);
        // The audio lane is Candle-native (sc-12901) and matches this bundle's own backend. Its
        // ordered id surface is the audio catalog's — shipped generators kokoro_82m (sc-12836) and
        // moss_sfx_v2 (sc-12841), acestep_v15_turbo (sc-12842), plus the voice-cloning identity embedder chatterbox_ve
        // (sc-12844); later stories extend these exact assertions in catalog order. The lane
        // carries its own composed candle preparer (sc-12835/sc-12836).
        #[cfg(feature = "audio")]
        {
            assert_eq!(
                snapshot.audio_backend.as_deref(),
                Some(super::AUDIO_BACKEND)
            );
            assert_eq!(snapshot.audio_generator_ids, ["kokoro_82m", "moss_sfx_v2", "acestep_v15_turbo"]);
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
        assert_eq!(candle_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "cuda");
    }

    /// The sc-12835 acceptance smoke: a (test-only) dummy audio Generator registered through the
    /// audio composition root's builder — the exact seam a real provider crate's registration
    /// uses (sc-12836+) — resolves in the complete bundle catalog with backend=candle and
    /// `Modality::Audio`.
    #[cfg(feature = "audio")]
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
            super::media_registry(),
            super::llm::text_registry(),
            super::llm::snapshot_preparer_registry(),
            runtime_catalog::AudioLane {
                backend: super::AUDIO_BACKEND,
                generators: audio,
                preparers: super::llm::snapshot_preparer_registry(),
            },
        )
        .unwrap();

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
        assert_eq!(
            catalog.snapshot().audio_generator_ids,
            ["kokoro_82m", "moss_sfx_v2", "acestep_v15_turbo", "dummy-audio"]
        );
    }

    /// The CUDA bundle surfaces the NVFP4 FP4 tier (epic 11037, sc-11042 Option A). Pins the
    /// platform difference vs. the CPU/MLX runtimes, which do not (no FP4 hardware / no compute win).
    #[cfg(feature = "media")]
    #[test]
    fn cuda_bundle_surfaces_nvfp4_tier() {
        use super::gen_core::Quant;
        assert_eq!(super::nvfp4_quant_tiers(), &[Quant::Nvfp4]);
    }
}
