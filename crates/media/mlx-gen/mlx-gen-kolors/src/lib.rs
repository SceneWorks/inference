//! Kolors provider for mlx-gen — a bilingual (Chinese/English) SDXL-family T2I model.
//!
//! Kolors keeps the SDXL U-Net + SDXL VAE but swaps dual-CLIP conditioning for a **ChatGLM3-6B**
//! text encoder (penultimate hidden state = context, last-token last-layer state = pooled). This
//! crate is built up across epic 3090:
//!
//!  - [`chatglm3`] — the ChatGLM3-6B encoder-only forward (sc-3091).
//!  - [`tokenizer`] — the ChatGLM3 SentencePiece tokenizer (sc-3092).
//!  - [`unet`] — the SDXL U-Net + ChatGLM3 context-projection wiring (sc-3093).
//!  - the T2I / img2img pipelines (sc-3094/3095), quant (sc-3096), ControlNet / IP-Adapter-Plus
//!    (sc-3097/98).

pub mod chatglm3;
pub mod convert;
pub mod ip_adapter;
pub mod model;
pub mod registry;
pub mod sampler;
pub mod tokenizer;
pub mod training;
pub mod unet;

pub use model::Kolors;
pub use registry::{descriptor, KolorsGenerator, MODEL_ID};
pub use training::{load_trainer, KolorsTrainer};

/// Add the MLX Kolors generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(crate::registry::REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX Kolors provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit_generators, ["kolors"]);
        assert_eq!(explicit_trainers, ["kolors"]);
    }
}
