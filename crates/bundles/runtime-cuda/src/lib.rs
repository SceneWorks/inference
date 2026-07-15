//! Supported NVIDIA runtime: explicit Candle CUDA media, LLM, and snapshot-preparer catalogs.

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
/// The single tensor backend every provider in this bundle uses.
pub const BACKEND: &str = "candle";
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

/// Build the complete validated CUDA runtime composition.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        media_registry(),
        candle_llm::text_registry(),
        candle_llm::snapshot_preparer_registry(),
    )
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
        #[cfg(feature = "media")]
        assert_eq!(candle_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "cuda");
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
