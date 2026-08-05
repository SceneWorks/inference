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

pub(crate) mod block_stream;
pub mod chatglm3;
pub mod convert;
pub mod ip_adapter;
pub mod memory_strategy;
pub mod model;
pub mod registry;
pub mod sampler;
pub mod tokenizer;
pub mod training;
pub mod unet;

pub use model::Kolors;
pub use registry::{descriptor, KolorsGenerator, MODEL_ID, SIZE_MULTIPLE};
pub use training::{load_trainer, KolorsTrainer};

/// Add the MLX Kolors generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(crate::registry::REGISTRATION)
        .register_memory_strategy(crate::registry::MEMORY_REGISTRATION)
        .register_memory_behavior(crate::registry::MEMORY_BEHAVIOR_REGISTRATION)
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

    /// Weights-free behavioral oracle for the shared memory ladder (SC-15521).
    ///
    /// This is the check that makes the declaration non-vacuous without weights: for **every**
    /// declared rung it builds the provider's own representative selection, drives the whole request
    /// scope through it (`configure_request` → phases → `configure_decode` / `configure_attention` /
    /// `materialize_transformer_window` → `finish`), and proves the safety check is not blind to an
    /// impossible budget.
    #[test]
    fn shared_ladder_registrations_pass_the_weights_free_behavior_oracle() {
        let registry = super::provider_registry().unwrap();
        for shape in [
            mlx_gen::LoadShape::DeferredMaterialization,
            mlx_gen::LoadShape::EagerMaterialization,
        ] {
            let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
                .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
                .with_load_shape(shape);
            gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(
                &registry, &spec,
            );
        }
    }
}
