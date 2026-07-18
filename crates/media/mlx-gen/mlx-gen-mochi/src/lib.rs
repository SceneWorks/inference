//! # mlx-gen-mochi
//!
//! Native-Rust / MLX inference for **Mochi 1** (`genmo/mochi-1-preview`, Apache-2.0) — a
//! T5-XXL-conditioned MMDiT text-to-video model with an asymmetric 3-D causal-conv VAE (6× temporal,
//! 8× spatial). This crate ships the complete provider surface:
//!
//! - [`text_encoder`] and [`tokenizer`] provide masked T5-XXL conditioning;
//! - [`transformer`], [`scheduler`], and [`pipeline`] implement the MMDiT denoise path;
//! - [`vae`] decodes the asymmetric causal latent representation;
//! - [`convert`] assembles and quantizes snapshots; and
//! - [`model`] owns loading, the descriptor, and explicit provider registration.
//!
//! These public modules and the re-exports below are the source of truth for the supported surface.

pub mod config;
pub mod convert;
pub mod model;
pub mod pipeline;
pub mod positions;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

pub use config::{MochiConfig, MochiSplitModel, MochiVaeConfig};
pub use convert::{
    convert_and_assemble, quantize_transformer_map, stage_shared_components, MochiConvertOpts,
    MOCHI_QUANT_SUFFIXES,
};
pub use model::{descriptor, load, Mochi, MODEL_ID, SIZE_MULTIPLE};
pub use pipeline::{decode_to_frames, denoise, frames_to_images};
pub use positions::get_positions;
pub use rope::MochiRope;
pub use scheduler::{cfg_combine, linear_quadratic_schedule, MochiScheduler};
pub use text_encoder::{encode_prompt, load_t5_encoder, MochiTextConditioning};
pub use tokenizer::{load_tokenizer, load_tokenizer_with_max_len, MochiTokenizer};
pub use transformer::{
    load_transformer_weights, MochiAttention, MochiDitConfig, MochiLinear, MochiQuant,
    MochiTransformer3DModel, MochiTransformerBlock,
};
pub use vae::{load_vae_decoder, MochiVaeDecoder, DEFAULT_DECODE_CHUNK_FRAMES};

/// Add the MLX Mochi 1 generator to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build the complete explicit MLX Mochi 1 provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(generators, ["mochi_1"]);
    }

    #[test]
    fn registered_descriptor_conforms() {
        let registry = super::provider_registry().unwrap();
        assert!(
            registry.descriptor_conformance_errors().is_empty(),
            "mochi descriptor conformance violations: {:?}",
            registry.descriptor_conformance_errors()
        );
    }
}
