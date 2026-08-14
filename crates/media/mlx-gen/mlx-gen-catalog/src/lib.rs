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
        assert_eq!(registry.memory_strategy_registrations().len(), 46);
        assert_eq!(registry.memory_contract_fixture_registrations().len(), 45);
        let resident_only: Vec<_> = registry
            .resident_only_memory_contract_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(resident_only, ["flux2_dev_control"]);
        let surfaces = registry.memory_contract_surfaces().unwrap();
        assert_eq!(surfaces.len(), 45 * 12);
        assert!(surfaces.iter().all(|surface| !surface.composed));
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
