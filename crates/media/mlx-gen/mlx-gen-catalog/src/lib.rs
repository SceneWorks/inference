//! Explicit, complete provider catalog for the SceneWorks MLX media platform.
//!
//! Provider crates own their registrations; this top-level crate owns only platform composition and
//! stable ordering. Applications should construct one [`ProviderRegistry`] with [`provider_registry`]
//! and route all media loads through it.

pub use mlx_gen as media;
pub use mlx_gen::gen_core::{ProviderRegistry, ProviderRegistryBuilder};

pub mod licenses;

pub use licenses::{
    component_licenses, component_licenses_manifest_json, license_families, provider_components,
    MLX_MEDIA_PROVIDER_COMPONENTS,
};

/// Complete backend package surface owned by the macOS runtime.
///
/// Some modules are ordinary registry providers; `depth`, `face`, `instantid`, `pid`, `sam2`, and
/// `sam3` are intentionally bespoke utilities consumed through provider-specific APIs.
pub mod providers {
    pub use mlx_gen_anima as anima;
    pub use mlx_gen_bernini as bernini;
    pub use mlx_gen_boogu as boogu;
    pub use mlx_gen_chroma as chroma;
    pub use mlx_gen_clip as clip;
    pub use mlx_gen_depth as depth;
    pub use mlx_gen_face as face;
    pub use mlx_gen_flux as flux;
    pub use mlx_gen_flux2 as flux2;
    pub use mlx_gen_ideogram as ideogram;
    pub use mlx_gen_instantid as instantid;
    pub use mlx_gen_joycaption as joycaption;
    pub use mlx_gen_kolors as kolors;
    pub use mlx_gen_krea as krea;
    pub use mlx_gen_krea_realtime as krea_realtime;
    pub use mlx_gen_lens as lens;
    pub use mlx_gen_ltx as ltx;
    pub use mlx_gen_mage as mage;
    pub use mlx_gen_mochi as mochi;
    pub use mlx_gen_pid as pid;
    pub use mlx_gen_pulid as pulid;
    pub use mlx_gen_qwen_image as qwen_image;
    pub use mlx_gen_sam2 as sam2;
    pub use mlx_gen_sam3 as sam3;
    pub use mlx_gen_sana as sana;
    pub use mlx_gen_scail2 as scail2;
    pub use mlx_gen_sd3 as sd3;
    pub use mlx_gen_sdxl as sdxl;
    pub use mlx_gen_seedvr2 as seedvr2;
    pub use mlx_gen_sensenova as sensenova;
    pub use mlx_gen_svd as svd;
    pub use mlx_gen_wan as wan;
    pub use mlx_gen_z_image as z_image;
}

/// Platform-owned crates consumed through provider-specific APIs rather than the registry
/// `load(id, spec)` path (depth maps, face analysis, segmentation, the PiD latent decoder).
/// Listed here so their platform membership is as explicit as a registered generator.
pub const BESPOKE_UTILITY_CRATES: &[&str] = &["depth", "face", "instantid", "pid", "sam2", "sam3"];

/// Provider crates deliberately compiled but not yet composed into this platform registry.
///
/// Empty after sc-14041 registered the complete Mage-Flow RL provider. Future structure-only
/// crates must enter this list until their normal `Generator` load/generate path works.
pub const PENDING_REGISTRATION_CRATES: &[&str] = &[];

/// Add every provider shipped by the MLX media platform to an explicit registry builder.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    let registry = mlx_gen_anima::register_providers(registry);
    let registry = mlx_gen_bernini::register_providers(registry);
    let registry = mlx_gen_boogu::register_providers(registry);
    let registry = mlx_gen_chroma::register_providers(registry);
    let registry = mlx_gen_clip::register_providers(registry);
    let registry = mlx_gen_flux::register_providers(registry);
    let registry = mlx_gen_flux2::register_providers(registry);
    let registry = mlx_gen_ideogram::register_providers(registry);
    let registry = mlx_gen_joycaption::register_providers(registry);
    let registry = mlx_gen_kolors::register_providers(registry);
    let registry = mlx_gen_krea::register_providers(registry);
    let registry = mlx_gen_krea_realtime::register_providers(registry);
    let registry = mlx_gen_lens::register_providers(registry);
    let registry = mlx_gen_ltx::register_providers(registry);
    let registry = mlx_gen_mage::register_providers(registry);
    let registry = mlx_gen_mochi::register_providers(registry);
    let registry = mlx_gen_pulid::register_providers(registry);
    let registry = mlx_gen_qwen_image::register_providers(registry);
    let registry = mlx_gen_sana::register_providers(registry);
    let registry = mlx_gen_scail2::register_providers(registry);
    let registry = mlx_gen_sd3::register_providers(registry);
    let registry = mlx_gen_sdxl::register_providers(registry);
    let registry = mlx_gen_seedvr2::register_providers(registry);
    let registry = mlx_gen_sensenova::register_providers(registry);
    let registry = mlx_gen_svd::register_providers(registry);
    let registry = mlx_gen_wan::register_providers(registry);
    mlx_gen_z_image::register_providers(registry)
}

/// Why this platform refuses [`mlx_gen::Quant::Nvfp4`] — the reason reported by every rejected load.
///
/// NVFP4 (epic 11037, sc-11042 **Option A**) is a *distinct* quant tier: E2M1 4-bit elements with
/// FP8-E4M3 block scales, served by candle-gen's packed FP4 path on consumer Blackwell `sm_120`. MLX
/// has no FP4 hardware and no FP4 quantizer, so there is nothing here to serve it with.
pub const NVFP4_UNSUPPORTED_REASON: &str =
    "NVFP4 is a Blackwell/CUDA-only FP4 tier (E2M1 elements + FP8 block scales) with no MLX \
     quantizer; MLX would otherwise int4-affine quantize it, which is a different tier's numerics";

/// Build the complete explicit MLX media provider catalog.
///
/// The catalog declares NVFP4 unimplemented on this platform, so a [`LoadSpec`](mlx_gen::LoadSpec)
/// requesting it fails loudly here rather than reaching a provider (epic 11037 SC#5: *a quant tier is
/// a creative choice* — never silently substituted). This is **defense in depth**: no MLX catalog
/// surface offers the tier today, so nothing can request it, but the guard is what keeps that true
/// once a caller can pick tiers. It matters because the coercion would otherwise be *silent* rather
/// than a crash — every mlx-gen provider quantizes via `quantize(q.bits())`, and `Quant::Nvfp4.bits()`
/// is `4`, indistinguishable from `Q4` by the time it reaches the quantizer. Rejecting on the `Quant`
/// itself, at the one boundary every provider's load routes through, is the only place that
/// information still exists.
pub fn provider_registry() -> mlx_gen::gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new())
        .reject_quant(mlx_gen::Quant::Nvfp4, NVFP4_UNSUPPORTED_REASON)
        .build()
}

/// Resolve a provider-owned, load-exact numeric tier for calibrated MLX video admission. Today only
/// Wan TI2V-5B exposes this surface; every other route returns `None` rather than synthesizing a tier
/// from the requested quantization field.
pub fn resolved_video_memory_numeric_tier(
    provider_id: &str,
    spec: &media::LoadSpec,
) -> media::gen_core::Result<Option<media::gen_core::MemoryNumericTier>> {
    mlx_gen_wan::resolved_video_memory_numeric_tier(provider_id, spec)
}

/// Resolve the working-set profile for a worker-selected bounded-decode edge using the provider's
/// real live-budget planner. Unsupported provider ids return `Ok(None)`.
pub fn selected_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
    tile_edge: u32,
    overlap: u32,
) -> media::gen_core::Result<Option<media::VideoDecodeMemoryProfile>> {
    mlx_gen_wan::selected_video_decode_memory_profile(
        provider_id,
        width,
        height,
        frames,
        tile_edge,
        overlap,
    )
}

/// Resolve the load-bearing VAE geometry for a modelled MLX video generator.
///
/// Each provider owns its id-to-decoder assignment. SVD is intentionally unmodelled in this slice;
/// Mochi uses a different decode architecture and is outside the video-memory-ladder scope.
pub fn vae_tiling(provider_id: &str) -> Option<media::gen_core::tiling::VaeTiling> {
    mlx_gen_ltx::vae_tiling(provider_id)
        .or_else(|| mlx_gen_wan::vae_tiling(provider_id))
        .or_else(|| mlx_gen_bernini::vae_tiling(provider_id))
        .or_else(|| mlx_gen_scail2::vae_tiling(provider_id))
        .or_else(|| mlx_gen_krea_realtime::vae_tiling(provider_id))
}

/// Resolve a provider-owned conservative single-pass VAE decode memory profile.
///
/// This composes provider-owned calibrated VAE cost functions. For planner-backed routes it is the
/// planner's full-output case; MLX Bernini retains its established auto/explicit tiling policy and
/// reports the same Wan-z16 single-pass cost as a proven monotone upper bound instead. The result
/// excludes DiT/text-encoder weights. LTX's calibrated fixed term mixes decoder/base/runtime costs,
/// so it is retained in full and identifies zero substitutable decoder-resident bytes; Wan-family
/// profiles contain activation/accumulator work and likewise identify zero resident bytes. Use
/// [`media::VideoDecodeMemoryProfile::checked_composed_peak`] for checked composition and any declared
/// substitution. With LTX's zero attribution the whole mixed floor is deliberately preserved, even
/// though that may conservatively overlap a contract decoder charge. Unsupported ids, zero
/// dimensions, and arithmetic overflow return `None`.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<media::VideoDecodeMemoryProfile> {
    mlx_gen_ltx::conservative_video_decode_memory_profile(provider_id, width, height, frames)
        .or_else(|| {
            mlx_gen_wan::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
        .or_else(|| {
            mlx_gen_bernini::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
        .or_else(|| {
            mlx_gen_scail2::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
        .or_else(|| {
            mlx_gen_krea_realtime::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
}

#[cfg(test)]
mod tests {
    #[test]
    fn selected_video_memory_apis_do_not_expand_beyond_wan_ti2v_5b() {
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        for provider in [
            "unknown",
            mlx_gen_wan::MODEL_ID_T2V_14B,
            mlx_gen_wan::MODEL_ID_I2V_14B,
            mlx_gen_wan::MODEL_ID_VACE,
            mlx_gen_wan::MODEL_ID_VACE_FUN,
        ] {
            assert_eq!(
                super::resolved_video_memory_numeric_tier(provider, &spec).unwrap(),
                None
            );
            assert_eq!(
                super::selected_video_decode_memory_profile(provider, 480, 480, 1, 448, 64)
                    .unwrap(),
                None
            );
        }
    }

    #[test]
    fn modelled_video_provider_ids_have_typed_vae_assignments() {
        let registry = super::provider_registry().unwrap();
        let registered: Vec<&str> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        let expected = [
            (
                mlx_gen_ltx::MODEL_ID,
                mlx_gen_ltx::VAE_TILING,
                3_312_533_760,
            ),
            (
                mlx_gen_wan::MODEL_ID,
                mlx_gen_wan::WAN_Z48_VAE_TILING,
                126_812_160,
            ),
            (
                mlx_gen_wan::MODEL_ID_T2V_14B,
                mlx_gen_wan::WAN_Z16_VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_wan::MODEL_ID_I2V_14B,
                mlx_gen_wan::WAN_Z16_VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_wan::MODEL_ID_VACE,
                mlx_gen_wan::WAN_Z16_VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_wan::MODEL_ID_VACE_FUN,
                mlx_gen_wan::WAN_Z16_VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_bernini::pipeline::MODEL_ID,
                mlx_gen_bernini::VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_bernini::bernini::MODEL_ID,
                mlx_gen_bernini::VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_scail2::pipeline::MODEL_ID,
                mlx_gen_scail2::VAE_TILING,
                322_633_728,
            ),
            (
                mlx_gen_krea_realtime::MODEL_ID,
                mlx_gen_krea_realtime::VAE_TILING,
                322_633_728,
            ),
        ];

        for (provider_id, tiling, peak_bytes) in expected {
            assert!(
                registered.contains(&provider_id),
                "unregistered {provider_id}"
            );
            assert_eq!(
                super::vae_tiling(provider_id),
                Some(tiling),
                "{provider_id}"
            );
            assert_eq!(
                super::conservative_video_decode_memory_profile(provider_id, 64, 64, 9)
                    .map(|profile| profile.working_set_bytes()),
                Some(peak_bytes),
                "{provider_id}"
            );
        }
        assert_eq!(super::vae_tiling(mlx_gen_svd::MODEL_ID), None);
        assert_eq!(super::vae_tiling(mlx_gen_mochi::MODEL_ID), None);
        assert_eq!(super::vae_tiling("not_a_provider"), None);
        for provider_id in [
            mlx_gen_svd::MODEL_ID,
            mlx_gen_mochi::MODEL_ID,
            "not_a_provider",
        ] {
            assert_eq!(
                super::conservative_video_decode_memory_profile(provider_id, 64, 64, 9),
                None
            );
        }
        assert_eq!(
            super::conservative_video_decode_memory_profile(mlx_gen_ltx::MODEL_ID, 0, 64, 9),
            None
        );
        assert_eq!(
            super::conservative_video_decode_memory_profile(
                mlx_gen_wan::MODEL_ID_T2V_14B,
                u32::MAX,
                u32::MAX,
                u32::MAX,
            ),
            None
        );

        let ltx = super::conservative_video_decode_memory_profile(mlx_gen_ltx::MODEL_ID, 64, 64, 9)
            .unwrap();
        assert_eq!(ltx.resident_decoder_bytes_included(), 0);
        assert_eq!(
            ltx.checked_composed_peak(3_100_000_000, 3_100_000_000),
            Some(6_412_533_760)
        );
        assert_eq!(
            ltx.checked_composed_peak(3_500_000_000, 3_500_000_000),
            Some(6_812_533_760)
        );

        let wan = super::conservative_video_decode_memory_profile(
            mlx_gen_wan::MODEL_ID_T2V_14B,
            64,
            64,
            9,
        )
        .unwrap();
        assert_eq!(wan.resident_decoder_bytes_included(), 0);
        assert_eq!(
            wan.checked_composed_peak(1_000_000_000, 600_000_000),
            Some(1_322_633_728)
        );

        let z16_ids = [
            mlx_gen_wan::MODEL_ID_T2V_14B,
            mlx_gen_wan::MODEL_ID_I2V_14B,
            mlx_gen_wan::MODEL_ID_VACE,
            mlx_gen_wan::MODEL_ID_VACE_FUN,
            mlx_gen_bernini::pipeline::MODEL_ID,
            mlx_gen_bernini::bernini::MODEL_ID,
            mlx_gen_scail2::pipeline::MODEL_ID,
            mlx_gen_krea_realtime::MODEL_ID,
        ];
        for id in z16_ids {
            for (requested_frames, expected_bytes) in [
                (1, Some(107_544_576)),
                (9, Some(322_633_728)),
                (81, Some(2_258_436_096)),
                (u32::MAX, None),
            ] {
                assert_eq!(
                    super::conservative_video_decode_memory_profile(id, 64, 64, requested_frames,)
                        .map(|profile| profile.working_set_bytes()),
                    expected_bytes,
                    "{id} at {requested_frames} requested frames",
                );
            }
        }
    }

    const PREVIEW_PROVIDER_IDS: [&str; 38] = [
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
        "flux2_klein_9b",
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
        "flux2_dev",
        "flux2_dev_edit",
        "flux2_dev_control",
        "flux1_schnell",
        "flux1_dev",
        "flux1_dev_control",
        "ideogram_4",
        "ideogram_4_turbo",
        "krea_2_turbo",
        "krea_2_raw",
        "krea_2_edit",
        "krea_2_turbo_edit",
        "krea_2_turbo_control",
        "lens",
        "lens_turbo",
        "qwen_image",
        "qwen_image_edit",
        "sana_1600m",
        "sana_sprint_1600m",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sdxl",
        "kolors",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "pulid_flux",
        "z_image",
        "z_image_control",
        "z_image_turbo",
        "z_image_turbo_control",
    ];

    #[test]
    fn preview_capability_matches_every_wired_shipped_route_bidirectionally() {
        let registry = super::provider_registry().unwrap();
        let descriptors: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .collect();

        for id in PREVIEW_PROVIDER_IDS {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.id == id)
                .unwrap_or_else(|| panic!("preview allowlist contains unshipped provider {id}"));
            assert!(
                descriptor.capabilities.supports_preview,
                "wired preview provider {id} must advertise support"
            );
        }

        let advertising: std::collections::BTreeSet<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.capabilities.supports_preview)
            .map(|descriptor| descriptor.id)
            .collect();
        let expected: std::collections::BTreeSet<_> = PREVIEW_PROVIDER_IDS.into_iter().collect();
        assert_eq!(
            advertising, expected,
            "only providers with an actual PreviewSink denoise route may advertise support"
        );
    }

    #[test]
    fn temporal_svd_and_struct_only_instantid_stay_outside_preview_advertising() {
        let registry = super::provider_registry().unwrap();
        let descriptors: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .collect();
        let svd = descriptors
            .iter()
            .find(|descriptor| descriptor.id == "svd_xt")
            .expect("SVD remains a registered temporal generator");
        assert!(
            !svd.capabilities.supports_preview,
            "SVD temporal previews remain scoped to sc-16636"
        );
        assert!(
            descriptors
                .iter()
                .all(|descriptor| descriptor.id != "instantid"),
            "InstantID is a struct-only composition API and must not gain invented registration"
        );
    }

    #[test]
    fn measured_activation_anchors_are_provider_route_owned() {
        let registry = super::provider_registry().unwrap();

        assert_eq!(
            registry
                .activation_memory_bytes_1024("krea_2_turbo")
                .unwrap(),
            Some(8_235_599_791)
        );
        assert_eq!(
            registry.activation_memory_bytes_1024("qwen_image").unwrap(),
            Some(8_235_599_791)
        );
        assert_eq!(
            registry
                .activation_memory_bytes_1024("qwen_image_control")
                .unwrap(),
            None,
            "Qwen Control is a distinct unmeasured route"
        );
        assert_eq!(
            registry.activation_memory_bytes_1024("sdxl").unwrap(),
            Some(15_086_072_628)
        );
        assert_eq!(
            registry
                .activation_memory_bytes_1024("z_image_turbo")
                .unwrap(),
            Some(15_086_072_628)
        );
        for (id, expected) in [
            ("sana_sprint_1600m", 14_001_593_385),
            ("anima_base", 8_235_599_791),
            ("sensenova_u1_8b", 1_438_814_045),
            ("flux2_klein_9b", 15_107_547_464),
            ("flux1_dev", 15_096_810_046),
            ("mage_flow", 1_685_774_664),
        ] {
            assert_eq!(
                registry.activation_memory_bytes_1024(id).unwrap(),
                Some(expected),
                "{id} must publish its upward-rounded measured anchor"
            );
        }
        for id in [
            "sana_1600m",
            "anima_turbo",
            "sensenova_u1_8b_fast",
            "flux2_klein_9b_edit",
            "flux1_dev_control",
            "mage_flow_edit",
            "lens",
            "lens_turbo",
            "bernini_renderer",
            "bernini",
        ] {
            assert_eq!(
                registry.activation_memory_bytes_1024(id).unwrap(),
                None,
                "{id} is a distinct unmeasured route and must retain the consumer fallback"
            );
        }
    }

    #[test]
    fn every_registered_memory_strategy_rejects_cross_route_decode_geometry() {
        let registry = super::provider_registry().unwrap();
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);
    }

    #[test]
    fn cfg_capability_matrix_matches_the_registered_mlx_render_paths() {
        let registry = super::provider_registry().unwrap();
        let descriptor = |id: &str| {
            registry
                .generators()
                .find_map(|registration| {
                    let descriptor = (registration.descriptor)();
                    (descriptor.id == id).then_some(descriptor)
                })
                .unwrap_or_else(|| panic!("{id} missing from MLX catalog"))
        };

        for id in [
            "flux2_klein_9b",
            "flux2_klein_9b_edit",
            "flux2_klein_9b_kv_edit",
        ] {
            let capabilities = descriptor(id).capabilities;
            assert!(capabilities.supports_guidance, "{id}");
            assert!(capabilities.supports_negative_prompt, "{id}");
            assert!(!capabilities.supports_true_cfg, "{id}");
        }
        for id in ["sd3_5_large", "sd3_5_medium"] {
            let capabilities = descriptor(id).capabilities;
            assert!(capabilities.supports_guidance, "{id}");
            assert!(capabilities.supports_negative_prompt, "{id}");
            assert!(
                !capabilities.supports_true_cfg,
                "{id} does not consume request.true_cfg"
            );
        }
        let turbo = descriptor("sd3_5_large_turbo").capabilities;
        assert!(!turbo.supports_guidance);
        assert!(!turbo.supports_negative_prompt);
        assert!(!turbo.supports_true_cfg);

        for id in ["sensenova_u1_8b", "sensenova_u1_8b_fast"] {
            let capabilities = descriptor(id).capabilities;
            assert!(capabilities.supports_guidance, "{id}");
            assert!(!capabilities.supports_negative_prompt, "{id}");
            assert!(
                capabilities.supports_true_cfg,
                "{id} consumes request.true_cfg as reference-image guidance"
            );
            assert!(
                !capabilities.conditioning.is_empty(),
                "{id} must retain its reference-conditioned it2i surface"
            );
        }
    }

    #[test]
    fn complete_catalog_has_stable_conforming_surface() {
        let registry = super::provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let trainers: Vec<String> = registry
            .trainers()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let captioners: Vec<String> = registry
            .captioners()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let image_embedders: Vec<String> = registry
            .image_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let text_embedders: Vec<String> = registry
            .text_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();

        assert_eq!(registry.transforms().len(), 0);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
        assert!(registry
            .generators()
            .all(|r| (r.descriptor)().backend == "mlx"));
        assert!(registry
            .trainers()
            .all(|r| (r.descriptor)().backend == "mlx"));
        assert_eq!(
            generators,
            [
                "anima_base",
                "anima_aesthetic",
                "anima_turbo",
                "bernini_renderer",
                "bernini",
                "boogu_image",
                "boogu_image_turbo",
                "boogu_image_edit",
                "chroma1_hd",
                "chroma1_base",
                "chroma1_flash",
                "flux1_schnell",
                "flux1_dev",
                "flux1_dev_control",
                "flux2_klein_9b",
                "flux2_klein_9b_edit",
                "flux2_klein_9b_kv_edit",
                "flux2_dev",
                "flux2_dev_edit",
                "flux2_dev_control",
                "ideogram_4",
                "ideogram_4_turbo",
                "kolors",
                "krea_2_turbo",
                "krea_2_raw",
                "krea_2_edit",
                "krea_2_turbo_edit",
                "krea_2_turbo_control",
                "krea_realtime_14b",
                "lens_turbo",
                "lens",
                "ltx_2_3",
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo",
                "mochi_1",
                "pulid_flux",
                "qwen_image",
                "qwen_image_control",
                "qwen_image_edit",
                "sana_1600m",
                "sana_sprint_1600m",
                "scail2_14b",
                "sd3_5_large",
                "sd3_5_large_turbo",
                "sd3_5_medium",
                "sdxl",
                "seedvr2",
                "seedvr2_3b",
                "seedvr2_7b",
                "sensenova_u1_8b",
                "sensenova_u1_8b_fast",
                "svd_xt",
                "wan2_2_ti2v_5b",
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan_vace",
                "wan2_2_vace_fun_14b",
                "z_image_turbo",
                "z_image",
                "z_image_control",
                "z_image_turbo_control",
            ]
        );
        assert_eq!(
            trainers,
            [
                "anima_base",
                "anima_aesthetic",
                "anima_turbo",
                "kolors",
                "krea_2_raw",
                "lens",
                "ltx_2_3",
                "mage_flow_base",
                "sd3_5_large",
                "sd3_5_medium",
                "sdxl",
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan2_2_ti2v_5b",
                "z_image_turbo",
            ]
        );
        assert_eq!(
            captioners,
            ["fancyfeast/llama-joycaption-beta-one-hf-llava"]
        );
        assert_eq!(image_embedders, ["clip_vit_l14"]);
        assert_eq!(text_embedders, ["clip_vit_l14_text"]);

        // sc-16666: the licence mapping in [`crate::licenses`] is keyed off exactly these lists, so
        // this is where a surface change and a mapping change meet. All fifteen trainer ids are
        // also generator ids, which is why 65 + 15 + 1 + 2 registrations are 68 distinct ids.
        //
        // Registration is never conditioned on the mapping — 59 < 68 because nine ids load nothing
        // the shared checkpoint table covers, and they ship exactly as before. That gap is a hole in
        // our metadata for CI to report, and `licenses::tests` pins which nine and why.
        let distinct: std::collections::BTreeSet<&String> = generators
            .iter()
            .chain(&trainers)
            .chain(&captioners)
            .chain(&image_embedders)
            .chain(&text_embedders)
            .collect();
        assert_eq!(distinct.len(), 68);
        assert_eq!(super::MLX_MEDIA_PROVIDER_COMPONENTS.len(), 59);
    }

    /// Mage-Flow's base, turbo, and RL variants are registered on the shipped MLX platform surface
    /// (sc-14041). Pin both the registrations and the now-empty pending list so a later composition
    /// edit cannot silently revert the completed registration.
    #[test]
    fn mage_rl_is_on_the_shipped_platform_surface() {
        assert!(super::PENDING_REGISTRATION_CRATES.is_empty());
        let shipped: Vec<String> = super::provider_registry()
            .unwrap()
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert!(shipped.contains(&"mage_flow".to_string()));
        assert!(shipped.contains(&"mage_flow_base".to_string()));
        assert!(shipped.contains(&"mage_flow_turbo".to_string()));
    }

    /// Every shipped generator that advertises [`ConditioningKind::Control`] must also say **which**
    /// control kinds it admits, so a consumer can plan a control render weights-free instead of
    /// finding out when `generate` rejects the request after the weights are already resident.
    ///
    /// The exceptions are pinned by id rather than skipped by a predicate. That asymmetry is the
    /// gate: a new control branch that forgets to declare its policy fails here, and an id that
    /// gains one has to be removed from the list, so neither direction can rot quietly. Weights-free
    /// — descriptors only, no snapshot and no device.
    #[test]
    fn every_control_generator_declares_its_control_kinds() {
        use super::media::ConditioningKind;

        // These two advertise `Control` but are not `ControlBranch` implementors: they take a
        // caller-supplied ControlNet checkpoint through `LoadSpec::control`, so which signal is
        // admitted follows whichever checkpoint was loaded and there is no per-model fact to
        // advertise. Undeclared (`None`) is the honest answer — it reads as "control kind is
        // unchecked here", which is exactly true.
        //
        // Note `ConditioningKind::ControlClip` is a DIFFERENT kind (LTX / VACE video control
        // clips) and is not in scope here; only `Control` carries a ControlNet kind policy.
        const NO_BRANCH_POLICY: &[&str] = &["kolors", "sdxl"];

        let registry = super::provider_registry().unwrap();
        let control: Vec<_> = registry
            .generators()
            .map(|r| (r.descriptor)())
            .filter(|d| d.capabilities.accepts(ConditioningKind::Control))
            .collect();
        assert!(
            !control.is_empty(),
            "the catalog must ship control generators"
        );

        let undeclared: Vec<&str> = control
            .iter()
            .filter(|d| d.control_kinds.is_none())
            .map(|d| d.id)
            .filter(|id| !NO_BRANCH_POLICY.contains(id))
            .collect();
        assert!(
            undeclared.is_empty(),
            "these generators advertise Control but declare no control_kinds: {undeclared:?} — \
             declare the policy on the descriptor, or add the id to NO_BRANCH_POLICY with the \
             reason it has none"
        );

        let stale: Vec<&str> = control
            .iter()
            .filter(|d| d.control_kinds.is_some())
            .map(|d| d.id)
            .filter(|id| NO_BRANCH_POLICY.contains(id))
            .collect();
        assert!(
            stale.is_empty(),
            "these ids now declare control_kinds and must leave NO_BRANCH_POLICY: {stale:?}"
        );

        // Third direction: an id listed as an exception that does not advertise `Control` at all.
        // Without this the list silently accumulates dead entries — `ConditioningKind::ControlClip`
        // shares a prefix with `Control`, so a grep-derived list picks up video-clip providers that
        // were never in scope, and neither assertion above would notice.
        let ids: Vec<&str> = control.iter().map(|d| d.id).collect();
        let not_control: Vec<&&str> = NO_BRANCH_POLICY
            .iter()
            .filter(|id| !ids.contains(id))
            .collect();
        assert!(
            not_control.is_empty(),
            "these NO_BRANCH_POLICY ids do not advertise Control and must be removed: \
             {not_control:?} (shipped Control generators: {ids:?})"
        );
    }

    /// Defense in depth for epic 11037 SC#5: the MLX platform **rejects** the NVFP4 tier instead of
    /// silently int4-affine quantizing it (`Quant::Nvfp4.bits() == 4`, so a provider's
    /// `quantize(q.bits())` could not tell it from `Q4`). Weights-free — the guard fires at the
    /// catalog boundary, ahead of the provider's loader, so no snapshot is touched.
    #[test]
    fn mlx_catalog_rejects_nvfp4_quant_tier() {
        use super::media::{LoadSpec, Quant, WeightsSource};

        let registry = super::provider_registry().unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);

        let error = registry
            .load("flux1_dev", &spec)
            .err()
            .expect("NVFP4 must not reach an MLX provider")
            .to_string();
        assert!(error.contains("Nvfp4"), "{error}");
        assert!(error.contains(super::NVFP4_UNSUPPORTED_REASON), "{error}");
    }
}
