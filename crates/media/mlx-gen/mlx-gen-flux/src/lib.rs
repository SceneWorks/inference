//! # mlx-gen-flux
//!
//! FLUX.1 provider crate for [`mlx-gen`](mlx_gen). This crate establishes the provider
//! boundary for FLUX.1-schnell and FLUX.1-dev: registry ids, fork-derived variant
//! configuration, tokenizer loading contracts, text encoders, MMDiT transformer, VAE loading, and
//! the base txt2img generation path.

pub mod adapters;
pub mod config;
pub mod control_transformer;
pub mod convert;
pub mod image_encoder;
pub mod ip_adapter;
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod model_control;
pub mod pipeline;
pub mod preview;
pub mod quant;
pub mod text_encoder;
pub mod transformer;

pub use adapters::apply_flux_adapters;
pub use config::{
    FluxTokenizerKind, FluxVariant, DEFAULT_GUIDANCE, DEFAULT_HEIGHT, DEFAULT_WIDTH,
    FLUX1_DEV_CONTROL_ID, FLUX1_DEV_ID, FLUX1_SCHNELL_ID,
};
pub use control_transformer::{FluxControlNet, FluxControlNetConfig, FluxControlTransformer};
pub use image_encoder::FluxIpImageEncoder;
pub use ip_adapter::{FluxIpAdapter, FluxIpInjector};
pub use loader::{
    load_clip_encoder, load_clip_tokenizer, load_control_transformer_dev, load_t5_encoder,
    load_t5_tokenizer, load_transformer, load_vae, load_vae_from_weights,
};
pub use model::{
    descriptor_dev, descriptor_for, descriptor_schnell, load_dev, load_schnell, Flux1,
    SIZE_MULTIPLE,
};
pub use model_control::{descriptor_dev_control, load_dev_control, Flux1DevControl};
pub use pipeline::{
    build_linear_sigmas, create_noise, image_seq_len, pack_latents, unpack_latents,
};
pub use text_encoder::{ClipTextEncoder, FluxTextEncoders, T5Sublayer, T5TextEncoder};
pub use transformer::{FluxTransformer, FluxTransformerConfig};

/// sc-16209 Apple-Silicon warm sweep: FLUX.1 Dev bf16 peaked below 14.06 GiB at 1024².
pub const DEV_ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: FLUX1_DEV_ID,
        anchor: mlx_gen::ActivationMemoryAnchor {
            bytes_1024: 15_096_810_046,
        },
    };

/// Add all MLX FLUX.1 providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::SCHNELL_REGISTRATION)
        .register_memory_strategy(model::SCHNELL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: FLUX1_SCHNELL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(FLUX1_SCHNELL_ID, spec)
            },
        })
        .register_memory_behavior(model::SCHNELL_MEMORY_BEHAVIOR)
        .register_generator(model::DEV_REGISTRATION)
        .register_memory_strategy(model::DEV_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: FLUX1_DEV_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(FLUX1_DEV_ID, spec)
            },
        })
        .register_memory_behavior(model::DEV_MEMORY_BEHAVIOR)
        .register_activation_memory(DEV_ACTIVATION_MEMORY_REGISTRATION)
        .register_generator(model_control::DEV_CONTROL_REGISTRATION)
        .register_memory_strategy(model_control::DEV_CONTROL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: FLUX1_DEV_CONTROL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(FLUX1_DEV_CONTROL_ID, spec)
            },
        })
        .register_memory_behavior(model_control::DEV_CONTROL_MEMORY_BEHAVIOR)
}

/// Build the complete explicit MLX FLUX.1 provider catalog.
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

        assert_eq!(
            explicit,
            ["flux1_schnell", "flux1_dev", "flux1_dev_control"]
        );
    }
}
