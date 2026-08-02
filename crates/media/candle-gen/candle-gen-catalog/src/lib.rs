//! Explicit, complete provider catalog for the SceneWorks Candle media platform.
//!
//! Provider crates own their registrations; this top-level crate owns only platform composition and
//! stable ordering. Applications should construct one [`ProviderRegistry`] with [`provider_registry`]
//! and route all media loads through it.

pub use candle_gen as media;
pub use candle_gen::gen_core::{ProviderRegistry, ProviderRegistryBuilder};

pub mod licenses;

/// Complete backend package surface owned by the Candle runtimes.
///
/// Some modules are ordinary registry providers; `depth`, `face`, `instantid`, `pid`, `pulid`, and
/// `sam3` are intentionally bespoke utilities consumed through provider-specific APIs.
pub mod providers {
    pub use candle_gen_anima as anima;
    pub use candle_gen_bernini as bernini;
    pub use candle_gen_boogu as boogu;
    pub use candle_gen_chroma as chroma;
    pub use candle_gen_clip as clip;
    pub use candle_gen_depth as depth;
    pub use candle_gen_face as face;
    pub use candle_gen_flux as flux;
    pub use candle_gen_flux2 as flux2;
    pub use candle_gen_ideogram as ideogram;
    pub use candle_gen_instantid as instantid;
    pub use candle_gen_joycaption as joycaption;
    pub use candle_gen_kolors as kolors;
    pub use candle_gen_krea as krea;
    pub use candle_gen_lens as lens;
    pub use candle_gen_ltx as ltx;
    pub use candle_gen_mage as mage;
    pub use candle_gen_mochi as mochi;
    pub use candle_gen_pid as pid;
    pub use candle_gen_pulid as pulid;
    pub use candle_gen_qwen_image as qwen_image;
    pub use candle_gen_sam3 as sam3;
    pub use candle_gen_sana as sana;
    pub use candle_gen_scail2 as scail2;
    pub use candle_gen_sd3 as sd3;
    pub use candle_gen_sdxl as sdxl;
    pub use candle_gen_seedvr2 as seedvr2;
    pub use candle_gen_sensenova as sensenova;
    pub use candle_gen_svd as svd;
    pub use candle_gen_wan as wan;
    pub use candle_gen_z_image as z_image;
}

/// Platform-owned crates consumed through provider-specific APIs rather than the registry
/// `load(id, spec)` path (depth maps, face analysis, segmentation, identity conditioning,
/// the PiD latent decoder). Listed here so their platform membership is as explicit as a
/// registered generator. Note `pulid` is bespoke here, whereas MLX ships it as `pulid_flux`.
pub const BESPOKE_UTILITY_CRATES: &[&str] = &["depth", "face", "instantid", "pid", "pulid", "sam3"];

/// Add every provider shipped by the Candle media platform to an explicit registry builder.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    let registry = candle_gen_anima::register_providers(registry);
    let registry = candle_gen_bernini::register_providers(registry);
    let registry = candle_gen_boogu::register_providers(registry);
    let registry = candle_gen_chroma::register_providers(registry);
    let registry = candle_gen_clip::register_providers(registry);
    let registry = candle_gen_flux::register_providers(registry);
    let registry = candle_gen_flux2::register_providers(registry);
    let registry = candle_gen_ideogram::register_providers(registry);
    let registry = candle_gen_joycaption::register_providers(registry);
    let registry = candle_gen_kolors::register_providers(registry);
    let registry = candle_gen_krea::register_providers(registry);
    let registry = candle_gen_lens::register_providers(registry);
    let registry = candle_gen_ltx::register_providers(registry);
    let registry = candle_gen_mage::register_providers(registry);
    let registry = candle_gen_mochi::register_providers(registry);
    let registry = candle_gen_qwen_image::register_providers(registry);
    let registry = candle_gen_sana::register_providers(registry);
    let registry = candle_gen_scail2::register_providers(registry);
    let registry = candle_gen_sd3::register_providers(registry);
    let registry = candle_gen_sdxl::register_providers(registry);
    let registry = candle_gen_seedvr2::register_providers(registry);
    let registry = candle_gen_sensenova::register_providers(registry);
    let registry = candle_gen_svd::register_providers(registry);
    let registry = candle_gen_wan::register_providers(registry);
    candle_gen_z_image::register_providers(registry)
}

/// Build the complete explicit Candle media provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

// -------------------------------------------------------------------------------------------------
// Model-weight licence surface (sc-16667) — the Candle media half.
//
// DISCLOSURE ONLY. Nothing in this section blocks, gates, degrades or withholds anything, and
// nothing added here ever should. It exists so a consumer can SHOW a user which upstream artifacts a
// render touched and what those texts name; whether a given use is permitted is the consumer's
// evaluation of those facts against its own situation, which this crate knows nothing about.
//
// Three layers (see `gen_core::license`): the reviewed licence FAMILIES, one COMPONENT row per
// upstream checkpoint — shared with the MLX catalog, because a licence is a property of the
// checkpoint and both engines load the same ones — and the per-backend provider→component mapping in
// [`licenses`], whose term union is DERIVED and never hand-authored.
// -------------------------------------------------------------------------------------------------

/// The licence families every component row this catalog reaches resolves against — the reviewed
/// unit, shared with the audio and MLX catalogs.
pub fn license_families() -> &'static [media::gen_core::LicenseFamily] {
    media::gen_core::LICENSE_FAMILIES
}

/// The **shared** media checkpoint table: one row per upstream artifact whose licence has been read.
///
/// Deliberately not per-backend and deliberately not filtered to what this catalog reaches. It is
/// the same slice `mlx-gen-catalog` reads, so the two catalogs cannot drift into two different
/// answers about one upstream artifact; rows no registered Candle id loads are simply unreferenced
/// by [`provider_components`].
pub fn component_licenses() -> &'static [media::gen_core::ComponentLicense] {
    media::gen_core::MEDIA_COMPONENT_LICENSES
}

/// Which components each registered Candle provider id loads — the mapping
/// [`media::gen_core::provider_terms`] derives a provider's effective terms from.
///
/// Nine registered ids have no row because every component they load is a pinned hole in the shared
/// table; `licenses::tests` names which nine and why, deliberately as `#[cfg(test)]` data so no gate
/// can read it. Those ids still ship and still render — a missing disclosure never withholds a
/// provider, and registration is never conditioned on this mapping.
pub fn provider_components() -> &'static [media::gen_core::ProviderComponents] {
    licenses::PROVIDER_COMPONENTS
}

/// The model-licences manifest JSON at `schema_version` 3 for **this** catalog, in the same shape
/// the audio catalog emits and the release tooling ships beside the SPDX SBOM.
///
/// Not a committed artifact on its own: `release/model-weight-licenses.json` carries the audio lane
/// today, and merging the three catalogs' manifests into one file is sc-16664's job. Output is
/// deterministic, so a merge can compare byte-for-byte.
pub fn component_licenses_manifest_json() -> String {
    media::gen_core::component_licenses_manifest_json(
        license_families(),
        component_licenses(),
        provider_components(),
    )
}

/// The **advanced** quant tiers this Candle catalog surfaces beyond the universal group-wise affine
/// `Q4`/`Q8` (which every provider advertises via `Capabilities::supported_quants`).
///
/// The NVFP4 FP4 tensor-core tier ([`media::gen_core::Quant::Nvfp4`], epic 11037, sc-11042 **Option A**
/// — a *distinct* creative-choice tier, never an auto-swap of `q4`) is surfaced **only** when the
/// catalog is compiled with the `cuda` feature, i.e. on the consumer Blackwell (`sm_120`) runtime where
/// the FP4 cores exist and the sc-11039 cuBLASLt FP4 GEMM / [`media::quant::Nvfp4Linear`] serve it
/// natively packed (epic 11037 SC#6). The CPU candle bundle (dequant→bf16 fallback, no FP4 compute win)
/// and the MLX/macOS runtime (no FP4 hardware, a separate `mlx-gen-catalog`) do **not** surface it — a
/// deliberate, pinned platform difference (CONTRIBUTING: pin catalog-surface differences rather than
/// paper over them; see `nvfp4_tier_surface_is_cuda_only`).
///
/// This is the inference-repo registration point per the 2026-07-14 epic replan: an NVFP4 tier reaches
/// the SceneWorks worker only through this catalog, shipped by runtime-tag (sc-12006); the worker-side
/// tier-select that *requests* it is deferred to the post-tag phase.
pub fn nvfp4_quant_tiers() -> &'static [media::gen_core::Quant] {
    #[cfg(feature = "cuda")]
    {
        &[media::gen_core::Quant::Nvfp4]
    }
    #[cfg(not(feature = "cuda"))]
    {
        &[]
    }
}

/// Whether **this compilation** of the catalog surfaces the NVFP4 tier — i.e. whether it resolved with
/// the `cuda` feature. Equivalent to `!nvfp4_quant_tiers().is_empty()`, exposed as a `const` because a
/// *dependent* crate cannot ask that question with `cfg!`.
///
/// `cfg!(feature = "cuda")` only ever reads the features of the crate it is written in, so a bundle
/// such as `runtime-cpu` — which has no `cuda` feature of its own — cannot mirror the rule pinned by
/// `nvfp4_tier_surface_is_cuda_only` by writing the same `cfg`. It has to read the catalog's resolved
/// answer, and this is it. That matters because Cargo **feature unification** makes the answer a
/// property of the *resolved graph*, not of the bundle: any build that pulls in `runtime-cpu` and
/// `runtime-cuda` together enables `candle-gen-catalog/cuda` once, for every consumer, so a CPU bundle
/// can legitimately observe the tier surfaced. That resolution is not a supported lane (CLAUDE.md: CPU,
/// CUDA, and MLX are mutually exclusive platform targets, never additive features), but a bundle-side
/// assertion should stay honest under it rather than fail spuriously.
pub const SURFACES_NVFP4_TIER: bool = cfg!(feature = "cuda");

#[cfg(test)]
mod tests {
    #[test]
    fn every_registered_memory_strategy_rejects_cross_route_decode_geometry() {
        let registry = super::provider_registry().unwrap();
        let spec = candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
            "/nonexistent".into(),
        ));
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);
    }

    #[test]
    fn cfg_capability_matrix_matches_the_registered_candle_render_paths() {
        let registry = super::provider_registry().unwrap();
        let descriptor = |id: &str| {
            registry
                .generators()
                .find_map(|registration| {
                    let descriptor = (registration.descriptor)();
                    (descriptor.id == id).then_some(descriptor)
                })
                .unwrap_or_else(|| panic!("{id} missing from Candle catalog"))
        };

        let klein = descriptor("flux2_klein_9b").capabilities;
        assert!(klein.supports_guidance);
        assert!(klein.supports_negative_prompt);
        assert!(!klein.supports_true_cfg);

        for id in ["sd3_5_large", "sd3_5_medium"] {
            let capabilities = descriptor(id).capabilities;
            assert!(capabilities.supports_guidance, "{id}");
            assert!(capabilities.supports_negative_prompt, "{id}");
            assert!(!capabilities.supports_true_cfg, "{id}");
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
                !capabilities.supports_true_cfg,
                "{id} has no Candle reference/image-CFG path"
            );
            assert!(
                capabilities.conditioning.is_empty(),
                "{id} is Candle txt2img only"
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
            .all(|r| (r.descriptor)().backend == "candle"));
        assert!(registry
            .trainers()
            .all(|r| (r.descriptor)().backend == "candle"));
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
                "flux2_klein_9b",
                "flux2_dev",
                "ideogram_4",
                "ideogram_4_turbo",
                "kolors",
                "krea_2_turbo",
                "krea_2_raw",
                "krea_2_edit",
                "lens_turbo",
                "lens",
                "ltx_2_3_distilled",
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo",
                "mochi_1",
                "qwen_image",
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
                "z_image_turbo",
                "z_image",
            ]
        );
        assert_eq!(
            trainers,
            [
                "krea_2_raw",
                "krea_2_control",
                "lens",
                "ltx_2_3",
                "sdxl",
                "wan2_2_t2v_14b",
                "z_image_turbo",
            ]
        );
        assert_eq!(
            captioners,
            ["fancyfeast/llama-joycaption-beta-one-hf-llava"]
        );
        assert_eq!(image_embedders, ["clip_vit_l14"]);
        assert_eq!(text_embedders, ["clip_vit_l14_text"]);

        // sc-16667: the pinned surface and the model-weight licence mapping move together — this is
        // where a surface change and a mapping change meet. Five of the seven trainer ids are also
        // generator ids, which is why 51 + 7 + 1 + 2 registrations are 56 distinct ids.
        //
        // Registration is never conditioned on the mapping: 47 < 56 because nine ids load nothing
        // the shared checkpoint table covers, and they ship exactly as before. That gap is a hole in
        // our metadata for CI to report, and `licenses::tests` pins which nine and why — as
        // `#[cfg(test)]` data, so no gate can read it and suppress them.
        let distinct: std::collections::BTreeSet<&String> = generators
            .iter()
            .chain(&trainers)
            .chain(&captioners)
            .chain(&image_embedders)
            .chain(&text_embedders)
            .collect();
        assert_eq!(distinct.len(), 56);
        assert_eq!(super::provider_components().len(), 47);
    }

    /// The manifest emitter runs on **this** catalog's three slices, and its output is
    /// deterministic.
    ///
    /// Deliberately cheap. Byte-stability of the shared emitter was settled in sc-16663 (#406,
    /// #411) and the audio lane gates its bytes against a committed file; there is no committed
    /// Candle media manifest to compare against, because merging the three catalogs into one
    /// release artifact is sc-16664's job. So this pins only what is this crate's to pin: the
    /// function is reachable, emits the schema-3 shape with all three layers present, and returns
    /// the same bytes call to call.
    #[test]
    fn component_licenses_manifest_json_is_well_formed_and_stable() {
        let generated = super::component_licenses_manifest_json();
        assert!(!generated.is_empty());

        let parsed: serde_json::Value =
            serde_json::from_str(&generated).expect("manifest is valid JSON");
        assert_eq!(parsed["schema_version"], 3);
        assert_eq!(parsed["kind"], "model-weight-licenses");
        // A consumer reads one document and finds all three layers in it.
        for section in ["families", "components", "providers"] {
            assert!(
                parsed[section]
                    .as_array()
                    .is_some_and(|rows| !rows.is_empty()),
                "manifest section {section:?} is missing or empty"
            );
        }
        assert_eq!(
            parsed["providers"].as_array().expect("providers").len(),
            super::provider_components().len()
        );

        assert_eq!(
            super::component_licenses_manifest_json(),
            generated,
            "the emitter must be deterministic — a merged release artifact compares bytes"
        );
    }

    #[test]
    fn mage_is_shipped_with_truthful_quant_surface() {
        let registry = super::provider_registry().expect("catalog");
        for id in [
            "mage_flow",
            "mage_flow_base",
            "mage_flow_turbo",
            "mage_flow_edit",
            "mage_flow_edit_base",
            "mage_flow_edit_turbo",
        ] {
            let descriptor = registry
                .generators()
                .map(|registration| (registration.descriptor)())
                .find(|descriptor| descriptor.id == id)
                .unwrap_or_else(|| panic!("{id} missing from Candle catalog"));
            assert_eq!(descriptor.backend, "candle");
            assert!(!descriptor.capabilities.mac_only);
            assert_eq!(
                descriptor.capabilities.supported_quants,
                &[
                    super::media::gen_core::Quant::Q4,
                    super::media::gen_core::Quant::Q8
                ]
            );
        }
    }

    /// Pin the NVFP4 tier's catalog surface (epic 11037, sc-11042 Option A): the FP4 tier is exposed
    /// **only** under the `cuda` feature (consumer Blackwell `sm_120`); the CPU candle bundle surfaces
    /// no advanced tier. The MLX/macOS runtime uses a separate `mlx-gen-catalog` with no such surface —
    /// the third leg of the pinned platform difference.
    #[test]
    fn nvfp4_tier_surface_is_cuda_only() {
        #[cfg(feature = "cuda")]
        assert_eq!(
            super::nvfp4_quant_tiers(),
            &[super::media::gen_core::Quant::Nvfp4],
            "NVFP4 must be surfaced on the cuda catalog"
        );
        #[cfg(not(feature = "cuda"))]
        assert!(
            super::nvfp4_quant_tiers().is_empty(),
            "NVFP4 must NOT be surfaced on a non-cuda (CPU) candle catalog"
        );

        // `SURFACES_NVFP4_TIER` is what dependent bundles read instead of a `cfg!` they cannot write;
        // it must never drift from the tier list itself.
        assert_eq!(
            super::SURFACES_NVFP4_TIER,
            !super::nvfp4_quant_tiers().is_empty(),
            "SURFACES_NVFP4_TIER must agree with nvfp4_quant_tiers()"
        );
    }

    #[test]
    fn krea_cuda_memory_contract_is_not_exposed_by_the_cpu_catalog() {
        let registry = super::provider_registry().expect("catalog");
        let spec = super::media::gen_core::LoadSpec::new(
            super::media::gen_core::WeightsSource::Dir("/nonexistent".into()),
        );
        let contract = registry
            .memory_strategy_contract("krea_2_turbo", &spec)
            .expect("known Krea generator");
        #[cfg(feature = "cuda")]
        assert!(
            contract.is_some(),
            "CUDA catalog must expose the Krea CUDA contract"
        );
        #[cfg(not(feature = "cuda"))]
        assert!(
            contract.is_none(),
            "CPU catalog must leave Krea on its compatibility default"
        );
    }
}
