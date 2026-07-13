//! Supported NVIDIA runtime: explicit Candle CUDA media, LLM, and snapshot-preparer catalogs.

pub use candle_gen_catalog::media;
pub use candle_llm as llm;
pub use runtime_catalog::{core_llm, gen_core, RuntimeCatalog, RuntimeCatalogSnapshot};

pub mod providers {
    pub use candle_gen_catalog::providers::*;
}

pub const PLATFORM: &str = "cuda";
pub const BACKEND: &str = "candle";
pub const SUPPORTED_TARGET_TRIPLES: &[&str] =
    &["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"];
pub const NATIVE_PREREQUISITES: &[&str] = &["NVIDIA CUDA toolkit", "supported NVIDIA driver"];

/// Build the complete validated CUDA runtime composition.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        candle_gen_catalog::provider_registry(),
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
        assert!(snapshot.generator_ids.len() > 40);
        assert_eq!(snapshot.text_llm_ids, ["candle-llama", "candle-llava"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["candle"]);
        assert_eq!(candle_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "cuda");
    }
}
