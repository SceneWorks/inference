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
    pub use mlx_gen_minimax_h3 as minimax_h3;
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

/// Explicit shared-optimization capability surface for P6's three benchmark providers.
///
/// Provider crates own these declarations and add a toggle only after concrete production call
/// sites exist. The benchmark independently requires a request-local `Applied` receipt, so this
/// availability preflight cannot fabricate execution.
pub fn benchmark_toggle_capabilities(provider_id: &str) -> Option<&'static [&'static str]> {
    match provider_id {
        mlx_gen_wan::MODEL_ID => Some(mlx_gen_wan::BENCHMARK_TOGGLE_CAPABILITIES),
        mlx_gen_qwen_image::MODEL_ID => Some(mlx_gen_qwen_image::BENCHMARK_TOGGLE_CAPABILITIES),
        "sdxl" => Some(mlx_gen_sdxl::BENCHMARK_TOGGLE_CAPABILITIES),
        _ => None,
    }
}

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
    let registry = mlx_gen_minimax_h3::register_providers(registry);
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
/// Each provider owns its id-to-decoder assignment. SVD is explicitly unmodelled: its MLX decode
/// path consumes temporal chunks but no `VaeTiling` or spatial planner, so copying the Candle
/// decoder's 256-channel geometry here would claim an enforcement seam that does not exist. See
/// [`vae_tiling_unmodelled_reason`]. Mochi uses a different decode architecture and is outside the
/// video-memory-ladder scope.
pub fn vae_tiling(provider_id: &str) -> Option<media::gen_core::tiling::VaeTiling> {
    mlx_gen_ltx::vae_tiling(provider_id)
        .or_else(|| mlx_gen_wan::vae_tiling(provider_id))
        .or_else(|| mlx_gen_bernini::vae_tiling(provider_id))
        .or_else(|| mlx_gen_scail2::vae_tiling(provider_id))
        .or_else(|| mlx_gen_krea_realtime::vae_tiling(provider_id))
}

/// Resolve a provider-owned reason that a registered MLX video route deliberately exposes no VAE
/// write-bound authority. `None` means either modelled, outside this surface, or unknown; callers
/// use [`vae_tiling`] for the actual modelled result.
pub fn vae_tiling_unmodelled_reason(provider_id: &str) -> Option<&'static str> {
    mlx_gen_svd::vae_tiling_unmodelled_reason(provider_id)
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
    fn benchmark_capabilities_are_explicit_for_every_p6_provider() {
        use std::collections::BTreeSet;

        let known: BTreeSet<_> = super::media::diagnostics::BENCHMARK_TOGGLES
            .iter()
            .copied()
            .collect();
        for provider in [mlx_gen_wan::MODEL_ID, mlx_gen_qwen_image::MODEL_ID, "sdxl"] {
            let declared = super::benchmark_toggle_capabilities(provider).unwrap_or_else(|| {
                panic!("{provider} must own an explicit P6 capability contract")
            });
            let unique: BTreeSet<_> = declared.iter().copied().collect();
            assert_eq!(unique.len(), declared.len(), "{provider} repeats a toggle");
            assert!(
                unique.is_subset(&known),
                "{provider} declares an unknown P6 toggle"
            );
        }
        assert!(super::benchmark_toggle_capabilities("not-a-provider").is_none());
    }

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
        assert_eq!(
            super::vae_tiling_unmodelled_reason(mlx_gen_svd::MODEL_ID),
            Some(mlx_gen_svd::VAE_TILING_UNMODELLED_REASON),
            "SVD's missing MLX geometry must be an explicit provider-owned result"
        );
        assert_eq!(super::vae_tiling(mlx_gen_mochi::MODEL_ID), None);
        assert_eq!(
            super::vae_tiling_unmodelled_reason(mlx_gen_mochi::MODEL_ID),
            None
        );
        assert_eq!(super::vae_tiling("not_a_provider"), None);
        assert_eq!(super::vae_tiling_unmodelled_reason("not_a_provider"), None);
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
    fn every_registered_generator_advertises_its_exact_latent_space() {
        use mlx_gen::gen_core::{
            LatentSpace, FLUX1_LATENT_SPACE, FLUX2_PACKED_LATENT_SPACE, LTX_VIDEO_LATENT_SPACE,
            MAGE_LATENT_SPACE, MOCHI_VIDEO_LATENT_SPACE, QWEN_KREA_Z16_LATENT_SPACE,
            SANA_LATENT_SPACE, SD3_LATENT_SPACE, SDXL_LATENT_SPACE, SEEDVR2_VIDEO_LATENT_SPACE,
            SVD_LATENT_SPACE, WAN_Z16_VIDEO_LATENT_SPACE, WAN_Z48_LATENT_SPACE,
        };

        fn expected(
            descriptor: &mlx_gen::gen_core::ModelDescriptor,
        ) -> Option<&'static LatentSpace> {
            match descriptor.family {
                "anima" | "qwen-image" | "krea_2" => Some(&QWEN_KREA_Z16_LATENT_SPACE),
                "krea_realtime" => Some(&WAN_Z16_VIDEO_LATENT_SPACE),
                "wan" if descriptor.id == "wan2_2_ti2v_5b" => Some(&WAN_Z48_LATENT_SPACE),
                "bernini" | "scail2" | "wan" => Some(&WAN_Z16_VIDEO_LATENT_SPACE),
                "flux" | "boogu" | "chroma" | "z-image" | "pulid" => Some(&FLUX1_LATENT_SPACE),
                "sd3" => Some(&SD3_LATENT_SPACE),
                "sdxl" | "kolors" => Some(&SDXL_LATENT_SPACE),
                "flux2" | "ideogram" | "lens" => Some(&FLUX2_PACKED_LATENT_SPACE),
                "ltx" => Some(&LTX_VIDEO_LATENT_SPACE),
                "mage_flow" => Some(&MAGE_LATENT_SPACE),
                "mochi" => Some(&MOCHI_VIDEO_LATENT_SPACE),
                "sana" => Some(&SANA_LATENT_SPACE),
                "seedvr2" => Some(&SEEDVR2_VIDEO_LATENT_SPACE),
                "svd" => Some(&SVD_LATENT_SPACE),
                // SenseNova's flow head emits RGB patches directly; there is no latent decoder seam.
                "sensenova-u1" => None,
                // MiniMax-H3's denoiser emits a 24-channel joint audio+video latent on the
                // 17-frame clip lattice (token-dropped, seam-blended dual decode — see the crate's
                // `chunking` module). No `LatentTemporalLaw` variant expresses that mapping and no
                // external decoder can consume it, so the descriptor deliberately advertises
                // nothing and fails closed against every decoder swap.
                "minimax_h3" => None,
                family => panic!(
                    "{} has unclassified latent lineage for registered family {family}",
                    descriptor.id
                ),
            }
        }

        let registry = super::provider_registry().unwrap();
        for registration in registry.generators() {
            let descriptor = (registration.descriptor)();
            assert_eq!(
                descriptor.denoiser_output_latent_space,
                expected(&descriptor),
                "{} must advertise the latent space its decoder consumes",
                descriptor.id
            );
        }
    }

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
        gen_core_testkit::memory_contract_surface_registry_conformance(&registry);
        assert_eq!(registry.memory_strategy_registrations().len(), 50);
        assert_eq!(registry.memory_contract_fixture_registrations().len(), 50);
        let resident_only: Vec<_> = registry
            .resident_only_memory_contract_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert!(resident_only.is_empty());
        let surfaces = registry.memory_contract_surfaces().unwrap();
        // 48 providers witness the complete 3-tier x 2-policy x 2-shape MLX surface (MiniMax-H3
        // joined them in the sc-17137 sync, and FLUX.2 Dev Control now has its exact fixture): these
        // providers publish every tier and materialization selector, even where a provider correctly
        // classifies a strategy as Missing. Two video providers publish narrower, truthful
        // inventories instead: LTX has no deferred/block-window loader, so it witnesses the eager
        // half; TI2V-5B admits only a BF16 Resident/Eager load, so it witnesses one selector per
        // tier. Spelling the sum out this way keeps a future provider's narrowing visible in the
        // diff rather than folded into a single total.
        assert_eq!(surfaces.len(), 48 * 12 + 6 + 3);
        assert!(surfaces.iter().all(|surface| !surface.composed));
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);
    }

    #[test]
    fn sc18457_deferred_provider_population_is_derived_from_typed_surfaces() {
        use mlx_gen::gen_core::{LoadShape, MemoryStrategy, MemoryStrategySupport};
        use std::collections::BTreeSet;

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        let expected: BTreeSet<_> = [
            "bf16:resident:deferred",
            "bf16:sequential:deferred",
            "q4:resident:deferred",
            "q4:sequential:deferred",
            "q8:resident:deferred",
            "q8:sequential:deferred",
        ]
        .into_iter()
        .collect();

        for provider in [
            "anima_base",
            "anima_aesthetic",
            "anima_turbo",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "kolors",
            "z_image",
        ] {
            let deferred: BTreeSet<_> = surfaces
                .iter()
                .filter(|surface| {
                    surface.contract.provider_id == provider
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                        && surface
                            .contract
                            .capability(MemoryStrategy::BoundedTransformerResidency)
                            .unwrap()
                            .support
                            == MemoryStrategySupport::Implemented
                })
                .map(|surface| surface.selector.id())
                .collect();
            assert_eq!(
                deferred, expected,
                "{provider} must derive every shipped resolved tier and both independent load policies"
            );
            assert!(surfaces
                .iter()
                .filter(|surface| {
                    surface.contract.provider_id == provider
                        && surface.selector.load_shape == LoadShape::EagerMaterialization
                })
                .all(|surface| surface
                    .contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support
                    == MemoryStrategySupport::Missing));
        }
    }

    /// SC-18610: all six registered Mage-Flow routes publish the complete engine ladder, rung 4
    /// included, on every shipped tier. Mage is request-scoped — it never reads
    /// `LoadSpec::offload_policy` — so unlike the FLUX/Krea families its rung 4 is reachable under
    /// both offload policies and is gated only by the deferred load shape.
    #[test]
    fn mage_routes_publish_the_full_rung_four_ladder_on_every_shipped_tier() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in mlx_gen_mage::model::MODEL_IDS {
            let provider: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider.len(), 12, "{provider_id}");

            let mut implemented = 0;
            for surface in provider {
                let expected = matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Bf16
                        | MemoryContractSurfaceTier::Q4
                        | MemoryContractSurfaceTier::Q8
                ) && surface.selector.load_shape
                    == LoadShape::DeferredMaterialization;
                assert_eq!(
                    surface
                        .contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .expect("complete Mage ladder")
                        .support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                for strategy in [
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                    MemoryStrategy::BoundedAttention,
                ] {
                    assert_eq!(
                        surface.contract.capability(strategy).unwrap().support,
                        MemoryStrategySupport::Implemented,
                        "{provider_id}: {} {strategy:?}",
                        surface.selector.id()
                    );
                }
                implemented += usize::from(expected);
                assert!(!surface.composed);
                assert_eq!(surface.contract.asset_facts, Default::default());
                assert_eq!(
                    surface.contract.calibration.as_ref().unwrap().fingerprint,
                    mlx_gen_mage::model::MEMORY_CALIBRATION_FINGERPRINT
                );
            }
            assert_eq!(
                implemented, 6,
                "{provider_id} publishes rung 4 on three tiers under both offload policies"
            );
        }
    }

    /// Every Mage route's declared rung 4 is executable through its own registered behavior: the
    /// scope opens and resolves into the request controls the pipeline reads. A route that only
    /// declared the rung would fail here.
    #[test]
    fn mage_behavior_inventory_executes_rung_four_for_every_route() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryMode, MemoryStrategy, OffloadPolicy,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in mlx_gen_mage::model::MODEL_IDS {
            let edit = provider_id.contains("_edit");
            let surface = surfaces
                .iter()
                .find(|surface| {
                    surface.contract.provider_id == provider_id
                        && surface.resolved_artifact_tier() == MemoryContractSurfaceTier::Q4
                        && surface.selector.offload_policy == OffloadPolicy::Sequential
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                })
                .unwrap_or_else(|| panic!("{provider_id} missing q4 sequential deferred surface"));
            let behavior = registry
                .memory_behavior_registrations()
                .find(|registration| registration.provider_id == provider_id)
                .unwrap_or_else(|| panic!("{provider_id} missing memory behavior"));
            let mut fixtures = (behavior.valid_fixtures)(
                &surface.spec,
                &surface.contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap();
            assert_eq!(fixtures.len(), 1, "{provider_id}");
            let fixture = &mut fixtures[0];
            assert_eq!(
                fixture.context.mode,
                if edit {
                    MemoryMode::Edit
                } else {
                    MemoryMode::TextToImage
                },
                "{provider_id}"
            );
            assert_eq!(
                fixture.context.geometry.reference_count,
                u32::from(edit),
                "{provider_id}"
            );
            assert!(!fixture.context.use_pid, "{provider_id}");
            assert_eq!(fixture.context.overlay, None, "{provider_id}");
            assert_eq!(
                fixture.request.image_reference_count(),
                u32::from(edit),
                "{provider_id}"
            );

            let mut scope =
                (behavior.begin_request)(&surface.spec, &surface.contract, &fixture.context)
                    .unwrap()
                    .unwrap_or_else(|| panic!("{provider_id} rung 4 must open a request scope"));
            scope.configure_request(&mut fixture.request).unwrap();
            let memory = fixture.request.memory.expect("configured request memory");
            assert!(memory.stream_transformer_blocks, "{provider_id}");
            assert!(memory.stage_residency, "{provider_id}");
            assert_eq!(memory.transformer_window_size, Some(1), "{provider_id}");
        }
    }

    #[test]
    fn krea_base_providers_publish_exact_prepacked_sequential_deferred_surfaces() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport,
            OffloadPolicy,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in [
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
        ] {
            let provider: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider.len(), 12, "{provider_id}");

            let mut implemented = 0;
            for surface in provider {
                let expected = matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Bf16
                        | MemoryContractSurfaceTier::Q4
                        | MemoryContractSurfaceTier::Q8
                ) && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization;
                assert_eq!(
                    surface
                        .contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .expect("complete Krea ladder")
                        .support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                implemented += usize::from(expected);
                assert!(!surface.composed);
                assert_eq!(surface.contract.asset_facts, Default::default());
                assert_eq!(
                    surface.contract.calibration.as_ref().unwrap().fingerprint,
                    mlx_gen_krea::block_memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                );
            }
            assert_eq!(implemented, 3, "{provider_id}");
        }
    }

    #[test]
    fn flux2_klein_providers_publish_exact_prepacked_sequential_deferred_surfaces() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport,
            OffloadPolicy,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in [
            "flux2_klein_9b",
            "flux2_klein_9b_edit",
            "flux2_klein_9b_kv_edit",
        ] {
            let provider: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider.len(), 12, "{provider_id}");

            let mut implemented = 0;
            for surface in provider {
                let expected = matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Bf16
                        | MemoryContractSurfaceTier::Q4
                        | MemoryContractSurfaceTier::Q8
                ) && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization;
                assert_eq!(
                    surface
                        .contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .expect("complete FLUX.2 Klein ladder")
                        .support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                implemented += usize::from(expected);
                assert!(!surface.composed);
                assert_eq!(surface.contract.asset_facts, Default::default());
                assert_eq!(
                    surface.contract.calibration.as_ref().unwrap().fingerprint,
                    format!(
                        "{}-{}",
                        mlx_gen_flux2::memory_strategy::KLEIN_STATIC_BEHAVIOR_FINGERPRINT,
                        provider_id.replace('_', "-")
                    )
                );
            }
            assert_eq!(implemented, 3, "{provider_id}");
        }
    }

    #[test]
    fn flux1_providers_publish_exact_declared_strategy_surfaces() {
        use mlx_gen::gen_core::{LoadShape, MemoryStrategy, MemoryStrategySupport, OffloadPolicy};

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in ["flux1_schnell", "flux1_dev", "flux1_dev_control"] {
            let provider: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider.len(), 12, "{provider_id}");

            let count = |strategy| {
                provider
                    .iter()
                    .filter(|surface| {
                        surface
                            .contract
                            .capability(strategy)
                            .expect("complete FLUX.1 ladder")
                            .support
                            == MemoryStrategySupport::Implemented
                    })
                    .count()
            };
            assert_eq!(count(MemoryStrategy::Resident), 12, "{provider_id}");
            assert_eq!(count(MemoryStrategy::StagedResidency), 6, "{provider_id}");
            let clean_base = provider_id != "flux1_dev_control";
            assert_eq!(
                count(MemoryStrategy::BoundedDecode),
                if clean_base { 12 } else { 0 },
                "{provider_id}"
            );
            assert_eq!(
                count(MemoryStrategy::BoundedAttention),
                if clean_base { 12 } else { 0 },
                "{provider_id}"
            );
            assert_eq!(
                count(MemoryStrategy::BoundedTransformerResidency),
                if clean_base { 3 } else { 0 },
                "{provider_id}"
            );
            assert!(provider.iter().all(|surface| {
                !surface.composed
                    && surface.contract.asset_facts == Default::default()
                    && surface
                        .contract
                        .calibration
                        .as_ref()
                        .is_some_and(|identity| {
                            identity.fingerprint.starts_with(
                                mlx_gen_flux::memory_strategy::STATIC_BEHAVIOR_FINGERPRINT,
                            )
                        })
            }));
            assert!(provider.iter().all(|surface| {
                let btr = surface
                    .contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap();
                btr.support != MemoryStrategySupport::Implemented
                    || (surface.selector.offload_policy == OffloadPolicy::Sequential
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization)
            }));
        }
    }

    #[test]
    fn flux1_behavior_inventory_is_single_phase_and_executable() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryMode, MemoryStrategy, OffloadPolicy,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in ["flux1_schnell", "flux1_dev", "flux1_dev_control"] {
            let surface = surfaces
                .iter()
                .find(|surface| {
                    surface.contract.provider_id == provider_id
                        && surface.resolved_artifact_tier() == MemoryContractSurfaceTier::Q4
                        && surface.selector.offload_policy == OffloadPolicy::Sequential
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                })
                .unwrap_or_else(|| panic!("{provider_id} missing q4 sequential deferred surface"));
            let strategy = if provider_id == "flux1_dev_control" {
                MemoryStrategy::StagedResidency
            } else {
                MemoryStrategy::BoundedTransformerResidency
            };
            let behavior = registry
                .memory_behavior_registrations()
                .find(|registration| registration.provider_id == provider_id)
                .unwrap();
            let fixtures =
                (behavior.valid_fixtures)(&surface.spec, &surface.contract, strategy).unwrap();
            assert_eq!(fixtures.len(), 1, "{provider_id}");
            let fixture = &fixtures[0];
            assert!(!fixture.context.has_phases, "{provider_id}");
            assert!(fixture.request.phases.is_none(), "{provider_id}");
            assert!(!fixture.context.use_pid, "{provider_id}");
            assert!(!fixture.request.use_pid, "{provider_id}");
            if provider_id == "flux1_dev_control" {
                assert_eq!(fixture.context.mode, MemoryMode::ImageToImage);
                assert_eq!(fixture.context.geometry.reference_count, 1);
                assert_eq!(fixture.context.overlay.as_deref(), Some("control"));
                assert!(matches!(
                    fixture.request.conditioning.as_slice(),
                    [mlx_gen::Conditioning::Control { .. }]
                ));
            } else {
                assert_eq!(fixture.context.mode, MemoryMode::TextToImage);
                assert_eq!(fixture.context.geometry.reference_count, 0);
                assert_eq!(fixture.context.overlay, None);
                assert!(fixture.request.conditioning.is_empty());
            }
            let mut scope =
                (behavior.begin_request)(&surface.spec, &surface.contract, &fixture.context)
                    .unwrap()
                    .expect("implemented FLUX.1 strategy must open a request scope");
            let mut request = fixture.request.clone();
            scope.configure_request(&mut request).unwrap();
            assert!(request.memory.is_some(), "{provider_id}");
        }
    }

    #[test]
    fn flux2_klein_behavior_inventory_is_exact_single_phase_and_executable() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryMode, MemoryStrategy, OffloadPolicy,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        let fixtures = |provider_id| {
            let surface = surfaces
                .iter()
                .find(|surface| {
                    surface.contract.provider_id == provider_id
                        && surface.resolved_artifact_tier() == MemoryContractSurfaceTier::Q4
                        && surface.selector.offload_policy == OffloadPolicy::Sequential
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                })
                .unwrap_or_else(|| panic!("{provider_id} missing q4 sequential deferred surface"));
            let behavior = registry
                .memory_behavior_registrations()
                .find(|registration| registration.provider_id == provider_id)
                .unwrap_or_else(|| panic!("{provider_id} missing memory behavior"));
            (behavior.valid_fixtures)(
                &surface.spec,
                &surface.contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap()
            .into_iter()
            .map(|fixture| {
                assert_eq!(fixture.context.overlay, None, "{provider_id}");
                assert!(!fixture.context.has_phases, "{provider_id}");
                assert!(fixture.request.phases.is_none(), "{provider_id}");
                assert!(!fixture.context.use_pid, "{provider_id}");
                let shape = match (
                    &fixture.context.mode,
                    fixture.context.geometry.reference_count,
                ) {
                    (MemoryMode::TextToImage, 0) if fixture.request.conditioning.is_empty() => 0,
                    (MemoryMode::ImageToImage, 1)
                        if matches!(
                            fixture.request.conditioning.as_slice(),
                            [mlx_gen::Conditioning::Reference {
                                strength: Some(_),
                                ..
                            }]
                        ) =>
                    {
                        1
                    }
                    (MemoryMode::Edit, 1)
                        if matches!(
                            fixture.request.conditioning.as_slice(),
                            [mlx_gen::Conditioning::Reference { strength: None, .. }]
                        ) =>
                    {
                        1
                    }
                    (MemoryMode::Edit, count @ 2..=8)
                        if matches!(
                            fixture.request.conditioning.as_slice(),
                            [mlx_gen::Conditioning::MultiReference { images }]
                                if images.len() == count as usize
                        ) =>
                    {
                        count
                    }
                    _ => panic!("{provider_id} published a non-executable behavior fixture"),
                };
                (
                    fixture.context.mode,
                    fixture.context.geometry.reference_count,
                    shape,
                )
            })
            .collect::<Vec<_>>()
        };

        assert_eq!(
            fixtures("flux2_klein_9b"),
            vec![
                (MemoryMode::TextToImage, 0, 0),
                (MemoryMode::ImageToImage, 1, 1),
            ]
        );
        let edit = (1..=8)
            .map(|references| (MemoryMode::Edit, references, references))
            .collect::<Vec<_>>();
        assert_eq!(fixtures("flux2_klein_9b_edit"), edit);
        assert_eq!(fixtures("flux2_klein_9b_kv_edit"), edit);
    }

    #[test]
    fn krea_base_behavior_inventory_preserves_typed_request_axes_without_overlays() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryMode, MemoryStrategy, OffloadPolicy,
        };

        let registry = super::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        let behavior = |provider_id| {
            registry
                .memory_behavior_registrations()
                .find(|registration| registration.provider_id == provider_id)
                .unwrap_or_else(|| panic!("{provider_id} missing memory behavior registration"))
        };
        let fixtures = |provider_id| {
            let surface = surfaces
                .iter()
                .find(|surface| {
                    surface.contract.provider_id == provider_id
                        && surface.resolved_artifact_tier() == MemoryContractSurfaceTier::Q4
                        && surface.selector.offload_policy == OffloadPolicy::Sequential
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                })
                .unwrap_or_else(|| panic!("{provider_id} missing q4 sequential deferred surface"));
            (behavior(provider_id).valid_fixtures)(
                &surface.spec,
                &surface.contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap()
            .into_iter()
            .map(|fixture| {
                assert_eq!(fixture.context.overlay, None, "{provider_id}");
                assert_eq!(
                    fixture.request.use_pid, fixture.context.use_pid,
                    "{provider_id}"
                );
                assert_eq!(
                    fixture.request.phases.is_some(),
                    fixture.context.has_phases,
                    "{provider_id}"
                );
                if let Some(phases) = &fixture.request.phases {
                    assert_eq!(phases.len(), 2, "{provider_id}");
                    assert!(phases.iter().all(|phase| phase.steps > 0), "{provider_id}");
                }
                let request_shape_matches = match (
                    &fixture.context.mode,
                    fixture.context.geometry.reference_count,
                ) {
                    (MemoryMode::TextToImage, 0) => fixture.request.conditioning.is_empty(),
                    (MemoryMode::ImageToImage, 1) => matches!(
                        fixture.request.conditioning.as_slice(),
                        [mlx_gen::Conditioning::Reference {
                            strength: Some(_),
                            ..
                        }]
                    ),
                    (MemoryMode::Edit, 1) => matches!(
                        fixture.request.conditioning.as_slice(),
                        [mlx_gen::Conditioning::Reference { strength: None, .. }]
                    ),
                    (MemoryMode::Edit, 2) => matches!(
                        fixture.request.conditioning.as_slice(),
                        [mlx_gen::Conditioning::MultiReference { images }] if images.len() == 2
                    ),
                    _ => false,
                };
                assert!(request_shape_matches, "{provider_id}");
                (
                    fixture.context.mode,
                    fixture.context.geometry.reference_count,
                    fixture.context.use_pid,
                    fixture.context.has_phases,
                )
            })
            .collect::<Vec<_>>()
        };

        let generation = vec![
            (MemoryMode::TextToImage, 0, false, false),
            (MemoryMode::TextToImage, 0, true, false),
            (MemoryMode::ImageToImage, 1, false, false),
            (MemoryMode::ImageToImage, 1, true, false),
        ];
        assert_eq!(fixtures("krea_2_turbo"), generation);

        let mut raw = generation;
        raw.push((MemoryMode::TextToImage, 0, false, true));
        assert_eq!(fixtures("krea_2_raw"), raw);

        let edit = vec![
            (MemoryMode::Edit, 1, false, false),
            (MemoryMode::Edit, 1, true, false),
            (MemoryMode::Edit, 2, false, false),
            (MemoryMode::Edit, 2, true, false),
        ];
        assert_eq!(fixtures("krea_2_edit"), edit);
        assert_eq!(fixtures("krea_2_turbo_edit"), edit);
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
                "minimax_h3",
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
        assert_eq!(distinct.len(), 69);
        assert_eq!(super::MLX_MEDIA_PROVIDER_COMPONENTS.len(), 60);
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
