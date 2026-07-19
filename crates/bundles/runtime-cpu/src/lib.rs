//! Supported CPU runtime: explicit Candle media, LLM, and snapshot-preparer catalogs.

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

/// Platform label for this bundle; matches `RuntimeCatalog::platform`.
pub const PLATFORM: &str = "cpu";
/// The single tensor backend every media, LLM, and snapshot-preparer provider in this bundle uses.
pub const BACKEND: &str = "candle";
/// The single tensor backend of this bundle's audio lane (sc-12901,
/// `docs/architecture/audio-backend-strategy.md`). Audio is Candle-native on every platform, so
/// here it matches `BACKEND`; the lane is still composed through the catalog's dedicated audio
/// section, owned by the audio composition root (`candle-audio-catalog`, sc-12835) rather than
/// `candle-gen-catalog`, and shipped under the additive `audio` feature.
pub const AUDIO_BACKEND: &str = "candle";
/// Target triples this bundle is supported on.
pub const SUPPORTED_TARGET_TRIPLES: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
];
/// Native (non-Cargo) prerequisites required to build and run this bundle (none for CPU).
pub const NATIVE_PREREQUISITES: &[&str] = &[];

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

/// Build the complete validated CPU runtime composition.
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
        assert_eq!(snapshot.platform, "cpu");
        assert_eq!(snapshot.backend, "candle");
        #[cfg(feature = "media")]
        assert!(snapshot.generator_ids.len() > 40);
        #[cfg(not(feature = "media"))]
        assert!(snapshot.generator_ids.is_empty());
        assert_eq!(snapshot.text_llm_ids, ["candle-llama", "candle-llava"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["candle"]);
        // The audio lane is Candle-native (sc-12901) and matches this bundle's own backend. Its
        // ordered id surface is the audio catalog's — shipped providers: kokoro_82m (sc-12836)
        // and moss_sfx_v2 (sc-12841); later stories extend this exact assertion in catalog order.
        // The lane carries its own composed candle preparer (sc-12835/sc-12836).
        #[cfg(feature = "audio")]
        {
            assert_eq!(
                snapshot.audio_backend.as_deref(),
                Some(super::AUDIO_BACKEND)
            );
            assert_eq!(snapshot.audio_generator_ids, ["kokoro_82m", "moss_sfx_v2"]);
            assert_eq!(snapshot.audio_snapshot_preparer_backends, ["candle"]);
        }
        #[cfg(not(feature = "audio"))]
        {
            assert_eq!(snapshot.audio_backend, None);
            assert!(snapshot.audio_generator_ids.is_empty());
        }
        #[cfg(feature = "media")]
        assert_eq!(candle_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "cpu");
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
            ["kokoro_82m", "moss_sfx_v2", "dummy-audio"]
        );
    }

    /// The CPU candle bundle does **not** surface the NVFP4 FP4 tier (no FP4 compute win off Blackwell)
    /// — the pinned platform difference vs. the CUDA bundle (epic 11037, sc-11042 Option A).
    ///
    /// Asserted against the catalog's own compile-time answer rather than unconditionally, mirroring
    /// `nvfp4_tier_surface_is_cuda_only`: the tier is surfaced **iff** the catalog resolved with `cuda`.
    /// A bare `is_empty()` is the right claim for a supported CPU build but is not resolution-proof —
    /// Cargo feature unification enables `candle-gen-catalog/cuda` for *every* consumer in any graph
    /// that also resolves `runtime-cuda`, which would fail this test through no fault of the bundle.
    /// That combination is not a supported lane (CLAUDE.md: CPU/CUDA/MLX are mutually exclusive
    /// platform targets, not additive features), so the point is only to keep the assertion honest
    /// under either resolution instead of latently fragile. `cfg!(feature = "cuda")` cannot express
    /// this here — this bundle has no `cuda` feature, so the cfg would read as `false` and still fail.
    #[cfg(feature = "media")]
    #[test]
    fn cpu_bundle_does_not_surface_nvfp4_tier() {
        use super::gen_core::Quant;

        if candle_gen_catalog::SURFACES_NVFP4_TIER {
            // Unified CPU+CUDA resolution: the catalog compiled with FP4 support, so the tier is
            // surfaced through every consumer of it, this bundle included.
            assert_eq!(
                candle_gen_catalog::nvfp4_quant_tiers(),
                &[Quant::Nvfp4],
                "a cuda-resolved catalog must surface exactly the NVFP4 tier"
            );
        } else {
            assert!(
                candle_gen_catalog::nvfp4_quant_tiers().is_empty(),
                "the supported CPU-only resolution must NOT surface the NVFP4 tier"
            );
        }
    }
}
