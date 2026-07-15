//! # mlx-gen-mochi
//!
//! Native-Rust / MLX inference edges for **Mochi 1** (`genmo/mochi-1-preview`, Apache-2.0) — a
//! T5-XXL-conditioned MMDiT text-to-video model with an asymmetric 3-D causal-conv VAE (6× temporal,
//! 8× spatial). This crate (story A2) scaffolds the model's **I/O edges**:
//!
//!  - the **text encoder** — the reused [`mlx_gen_flux::T5TextEncoder`] run *with* the tokenizer
//!    padding mask (Mochi's `_get_t5_prompt_embeds`), plus the vendored t5-v1.1-xxl tokenizer;
//!  - the **AsymmVAE decoder** — a faithful port of `AutoencoderKLMochi`'s decode path (attention-free
//!    mid-blocks, `MochiUpBlock3D` depth-to-space unpatchify, `CogVideoXCausalConv3d` replicate-pad,
//!    per-frame chunked GroupNorm), gated by the A1 real-weight goldens.
//!
//! The DiT transformer itself lands in a later story (A3/A4); this crate deliberately stops at the
//! edges so each component is parity-gated in isolation against the A1 goldens.

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
    convert_and_assemble, stage_shared_components, MochiConvertOpts, MOCHI_QUANT_SUFFIXES,
};
pub use model::{descriptor, load, Mochi, MODEL_ID};
pub use pipeline::{decode_to_frames, denoise, frames_to_images};
pub use positions::get_positions;
pub use rope::MochiRope;
pub use scheduler::{cfg_combine, linear_quadratic_schedule, MochiScheduler};
pub use text_encoder::{encode_prompt, load_t5_encoder, MochiTextConditioning};
pub use tokenizer::{load_tokenizer, load_tokenizer_with_max_len};
pub use transformer::{
    load_transformer_weights, MochiAttention, MochiDitConfig, MochiQuant, MochiTransformer3DModel,
    MochiTransformerBlock,
};
pub use vae::{load_vae_decoder, MochiVaeDecoder};

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
