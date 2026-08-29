//! Supported Apple-silicon runtime: explicit MLX media, LLM, and snapshot-preparer catalogs.

#[cfg(feature = "audio")]
pub use candle_audio_catalog::audio;
#[cfg(feature = "perf-bench")]
pub use mlx_gen_catalog::benchmark_toggle_capabilities;
#[cfg(feature = "media")]
pub use mlx_gen_catalog::media;
#[cfg(feature = "media")]
pub use mlx_gen_catalog::{vae_tiling, vae_tiling_unmodelled_reason};
pub use mlx_llm as llm;
pub use runtime_catalog::{
    core_llm, gen_core, memory_strategy, RuntimeCatalog, RuntimeCatalogSnapshot,
    VideoDecodeMemoryProfile,
};

/// Stable P6 workload/result schemas and fail-closed validation for the real-weight MLX harness.
#[cfg(feature = "perf-bench")]
pub mod perf_bench;

#[cfg(feature = "media")]
/// Resolve a provider-owned conservative VAE decode profile for contract-safe memory composition.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<VideoDecodeMemoryProfile> {
    mlx_gen_catalog::conservative_video_decode_memory_profile(provider_id, width, height, frames)
}

#[cfg(feature = "media")]
/// Resolve the load-exact provider numeric tier used by calibrated video memory admission.
pub fn resolved_video_memory_numeric_tier(
    provider_id: &str,
    spec: &gen_core::LoadSpec,
) -> gen_core::Result<Option<gen_core::MemoryNumericTier>> {
    mlx_gen_catalog::resolved_video_memory_numeric_tier(provider_id, spec)
}

#[cfg(feature = "media")]
/// Resolve the provider-owned profile for the exact selected bounded-decode carrier.
pub fn selected_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
    tile_edge: u32,
    overlap: u32,
) -> gen_core::Result<Option<VideoDecodeMemoryProfile>> {
    mlx_gen_catalog::selected_video_decode_memory_profile(
        provider_id,
        width,
        height,
        frames,
        tile_edge,
        overlap,
    )
}

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

/// Complete weights-free memory-contract surface for capability generation and reconciliation.
pub fn memory_contract_surface_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
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
    #[cfg(feature = "media")]
    #[test]
    fn bundle_exposes_narrow_selected_video_memory_apis() {
        let spec = super::gen_core::LoadSpec::new(super::gen_core::WeightsSource::Dir(
            "/nonexistent".into(),
        ));
        assert_eq!(
            super::resolved_video_memory_numeric_tier("unknown", &spec).unwrap(),
            None
        );
        assert_eq!(
            super::selected_video_decode_memory_profile("unknown", 480, 480, 1, 448, 64).unwrap(),
            None
        );
        assert!(super::resolved_video_memory_numeric_tier("bernini", &spec).is_err());
    }

    #[cfg(feature = "media")]
    #[test]
    fn bundle_exposes_engine_id_vae_geometry() {
        let tiling: super::gen_core::tiling::VaeTiling =
            super::vae_tiling("bernini").expect("modelled video id");
        assert_eq!(tiling.full_res_channels, 96);
        assert!(!tiling.causal_temporal);
        assert_eq!(
            super::conservative_video_decode_memory_profile("bernini", 64, 64, 9).map(|profile| (
                profile.working_set_bytes(),
                profile.resident_decoder_bytes_included(),
            )),
            Some((322_633_728, 0))
        );

        assert_eq!(super::vae_tiling("svd_xt"), None);
        assert_eq!(
            super::vae_tiling_unmodelled_reason("svd_xt"),
            Some(super::providers::svd::VAE_TILING_UNMODELLED_REASON),
            "the shipped MLX SVD route must remain explicitly unmodelled until its decoder has a \
             load-bearing spatial planner"
        );
    }

    #[test]
    fn smoke_catalog_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();
        assert_eq!(snapshot.platform, "macos");
        assert_eq!(snapshot.backend, "mlx");
        #[cfg(feature = "media")]
        assert!(snapshot.generator_ids.len() > 50);
        #[cfg(not(feature = "media"))]
        assert!(snapshot.generator_ids.is_empty());
        assert_eq!(
            snapshot.text_llm_ids,
            ["mlx-llama", "mlx-joycaption", "mlx-starvector-1b"]
        );
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);
        // The audio lane is declared Candle-native on this mlx bundle (sc-12901) — the
        // sanctioned cross-backend seam. Its ordered id surface is the audio catalog's —
        // shipped generators kokoro_82m (sc-12836), moss_sfx_v2 (sc-12841), acestep_v15_turbo
        // (sc-12842), moss_tts_realtime (sc-13392), chatterbox_tts (sc-13239),
        // mmaudio_small_16k (video->audio Foley, sc-12843) + mmaudio_large_44k (44.1 kHz,
        // sc-13441), stable_audio_3_small_music (text-to-music, sc-14543) +
        // stable_audio_3_small_sfx (text-to-SFX/Foley, sc-14544 — 44.1 kHz stereo, distinct from
        // the 48 kHz mono moss_sfx_v2) + stable_audio_3_medium (sc-14545 — 1.45B differential DiT
        // over SAME-L, both domains, 380 s) + the three pre-trained -base siblings
        // stable_audio_3_{small_music,small_sfx,medium}_base (sc-14546 — rectified_flow,
        // Euler/50/7.0 defaults), and moss_ttsd_v05
        // (multi-speaker dialogue TTS, sc-13518), plus the
        // voice-cloning identity embedder
        // chatterbox_ve (sc-12844); later stories extend in catalog order. The lane carries the
        // composed candle preparer (sc-12835/sc-12836) while the main preparer registry stays
        // mlx-only.
        #[cfg(feature = "audio")]
        {
            assert_eq!(
                snapshot.audio_backend.as_deref(),
                Some(super::AUDIO_BACKEND)
            );
            assert_eq!(
                snapshot.audio_generator_ids,
                [
                    "kokoro_82m",
                    "moss_sfx_v2",
                    "acestep_v15_turbo",
                    "stable_audio_3_small_music",
                    "stable_audio_3_small_sfx",
                    "stable_audio_3_medium",
                    "stable_audio_3_small_music_base",
                    "stable_audio_3_small_sfx_base",
                    "stable_audio_3_medium_base",
                    "moss_tts_realtime",
                    "chatterbox_tts",
                    "mmaudio_small_16k",
                    "mmaudio_large_44k",
                    "moss_ttsd_v05"
                ]
            );
            assert_eq!(snapshot.audio_voice_embedder_ids, ["chatterbox_ve"]);
            assert_eq!(snapshot.audio_transform_ids, ["openvoice_v2"]);
            assert_eq!(snapshot.audio_transcriber_ids, ["whisper_base"]);
            assert_eq!(snapshot.audio_embedder_ids, ["clap_htsat_unfused"]);
            assert_eq!(snapshot.audio_snapshot_preparer_backends, ["candle"]);
        }
        #[cfg(not(feature = "audio"))]
        {
            assert_eq!(snapshot.audio_backend, None);
            assert!(snapshot.audio_generator_ids.is_empty());
            assert!(snapshot.audio_voice_embedder_ids.is_empty());
            assert!(snapshot.audio_transform_ids.is_empty());
            assert!(snapshot.audio_transcriber_ids.is_empty());
            assert!(snapshot.audio_embedder_ids.is_empty());
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
                encoder_contract: None,
                denoiser_output_latent_space: None,
                control_kinds: None,
                required_components: &[],
                id: "dummy-audio",
                family: "test-audio",
                backend: super::AUDIO_BACKEND,
                modality: gen_core::Modality::Audio,
                capabilities: gen_core::Capabilities {
                    // Audio has no width/height — the sweep exempts Modality::Audio (sc-13314), so
                    // the bounds stay at the unused 0 like the real audio generators.
                    min_size: 0,
                    max_size: 0,
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
                "stable_audio_3_small_music",
                "stable_audio_3_small_sfx",
                "stable_audio_3_medium",
                "stable_audio_3_small_music_base",
                "stable_audio_3_small_sfx_base",
                "stable_audio_3_medium_base",
                "moss_tts_realtime",
                "chatterbox_tts",
                "mmaudio_small_16k",
                "mmaudio_large_44k",
                "moss_ttsd_v05",
                "dummy-audio"
            ]
        );
        assert_eq!(snapshot.audio_snapshot_preparer_backends, ["candle"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);
    }
}
