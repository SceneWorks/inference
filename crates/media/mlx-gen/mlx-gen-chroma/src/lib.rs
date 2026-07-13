//! # mlx-gen-chroma
//!
//! Chroma provider crate for [`mlx-gen`](mlx_gen) (epic 3531). Chroma (`chroma1_hd` / `chroma1_base`
//! / `chroma1_flash`, family `chroma`) is a FLUX.1-schnell-derived DiT: the FLUX MMDiT skeleton with
//! a distilled-guidance **Approximator** replacing the FLUX modulation stack, **T5-XXL-only**
//! conditioning (no CLIP / no pooled), MMDiT attention masking, and **true CFG**.
//!
//! Reuses `mlx-gen-flux` for the T5-XXL encoder, the AutoencoderKL VAE loader, and the
//! pack/unpack/sigma helpers; the Chroma DiT is ported fresh.

pub mod adapters;
pub mod beta;
pub mod config;
pub mod convert;
pub mod loader;
pub mod model;
pub mod quant;
pub mod text;
pub mod transformer;

pub use adapters::apply_chroma_adapters;
pub use config::{
    ChromaTransformerConfig, ChromaVariant, CHROMA1_BASE_ID, CHROMA1_FLASH_ID, CHROMA1_HD_ID,
    DEFAULT_SAMPLER, MAX_SEQUENCE_LENGTH,
};
pub use model::{
    descriptor_base, descriptor_flash, descriptor_hd, load_base, load_chroma, load_flash, load_hd,
    Chroma,
};
pub use text::{encode_prompt, t5_key_mask, transformer_text_mask};
pub use transformer::ChromaTransformer;

/// Add all MLX Chroma providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::HD_REGISTRATION)
        .register_generator(model::BASE_REGISTRATION)
        .register_generator(model::FLASH_REGISTRATION)
}

/// Build the complete explicit MLX Chroma provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["chroma1_hd", "chroma1_base", "chroma1_flash"]);
    }
}
