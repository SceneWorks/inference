//! Supported Apple-silicon runtime: explicit MLX media, LLM, and snapshot-preparer catalogs.

pub use mlx_gen_catalog::media;
pub use mlx_llm as llm;
pub use runtime_catalog::{core_llm, gen_core, RuntimeCatalog, RuntimeCatalogSnapshot};

pub mod providers {
    pub use mlx_gen_catalog::providers::*;
}

pub const PLATFORM: &str = "macos";
pub const BACKEND: &str = "mlx";
pub const SUPPORTED_TARGET_TRIPLES: &[&str] = &["aarch64-apple-darwin"];
pub const NATIVE_PREREQUISITES: &[&str] = &["macOS 26.2+", "Xcode Metal toolchain"];

/// Build the complete validated macOS runtime composition.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        mlx_gen_catalog::provider_registry(),
        mlx_llm::text_registry(),
        mlx_llm::snapshot_preparer_registry(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke_catalog_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();
        assert_eq!(snapshot.platform, "macos");
        assert_eq!(snapshot.backend, "mlx");
        assert!(snapshot.generator_ids.len() > 50);
        assert_eq!(snapshot.text_llm_ids, ["mlx-llama", "mlx-joycaption"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);
        assert_eq!(mlx_gen_catalog::BESPOKE_UTILITY_CRATES.len(), 6);
        assert_eq!(snapshot.to_json()["platform"], "macos");
    }
}
