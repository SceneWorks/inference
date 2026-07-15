//! Explicit, complete provider catalog for the SceneWorks MLX media platform.
//!
//! Provider crates own their registrations; this top-level crate owns only platform composition and
//! stable ordering. Applications should construct one [`ProviderRegistry`] with [`provider_registry`]
//! and route all media loads through it.

pub use mlx_gen as media;
pub use mlx_gen::gen_core::{ProviderRegistry, ProviderRegistryBuilder};

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
    pub use mlx_gen_lens as lens;
    pub use mlx_gen_ltx as ltx;
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
    let registry = mlx_gen_lens::register_providers(registry);
    let registry = mlx_gen_ltx::register_providers(registry);
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
                "lens_turbo",
                "lens",
                "ltx_2_3",
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
