//! # mlx-gen-ideogram
//!
//! The **Ideogram 4.0** provider crate for [`mlx-gen`](mlx_gen) (epic 4725). Ideogram 4 is a
//! flow-matching text-to-image model whose useful prompt contract is a structured **JSON
//! caption** (handled SceneWorks-side); the engine consumes that caption as a plain string.
//!
//! Architecture (from the `ideogram-ai/ideogram-4-fp8` checkpoint, sc-5984):
//! * **Text encoder** — `Qwen3-VL-8B-Instruct` (text path), hidden states from 13 layers
//!   (`config::EXTRACTED_LAYERS`) concatenated to 53248 features. Mirrors the `mlx-gen-flux2`
//!   Qwen3 blocks + a multi-layer capture hook.
//! * **Transformer** — single-stream 34-layer `Ideogram4Transformer2DModel` (AdaLN-modulated
//!   SwiGLU, fused QKV + per-head QK-norm, 3D MRoPE), instantiated **twice**
//!   (conditional + unconditional) for asymmetric CFG — or **once** with the ostris TurboTime LoRA
//!   for the CFG-free few-step `ideogram_4_turbo` variant ([`model::load_turbo`], issue #488).
//! * **VAE** — `AutoencoderKLFlux2` (the FLUX.2 VAE) → reuse `mlx-gen-flux2::Flux2Vae`.
//! * **Scheduler** — `FlowMatchEulerDiscreteScheduler` → reuse the core flow-match schedule.
//!
//! Weights are provisioned offline by `tools/convert_ideogram4_to_mlx.py` (fp8 weight-only →
//! bf16 MLX safetensors). Runtime is pure Rust/MLX.
//!
//! Slice status: engine **complete** and explicitly registered — converter (sc-5984), text encoder
//! (sc-5985), transformer (sc-5986), VAE (sc-5987), native tokenizer + `generate` pipeline +
//! [`Generator`](mlx_gen::Generator) registry registration under id `"ideogram_4"` (sc-5988, see
//! [`model`]). Follow-ons: Q4/Q8 quantization (sc-5989) and the gated turnkey publish (sc-5990).

pub mod adapters;
pub mod config;
pub mod convert;
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;

/// Packed (pre-quantized) weight loading — internal; the [`convert`] consume side.
mod quant;

pub use adapters::apply_ideogram_adapters;
pub use config::{
    Ideogram4DitConfig, Ideogram4TextEncoderConfig, DEFAULT_GUIDANCE, DEFAULT_HEIGHT,
    DEFAULT_STEPS, DEFAULT_TURBO_STEPS, DEFAULT_WIDTH, EXTRACTED_LAYERS, IDEOGRAM_4_FP8_REPO,
    IDEOGRAM_4_ID, IDEOGRAM_4_TURBO_ID, RES_MULTIPLE, TURBO_LORA_FILE, TURBO_LORA_SCALE,
};
pub use loader::{
    load_text_encoder, load_tokenizer, load_transformer, load_unconditional_transformer, load_vae,
};
pub use model::{
    descriptor, descriptor_turbo, load, load_turbo, Ideogram4, MODEL_ID, MODEL_ID_TURBO,
};
pub use pipeline::{EditInit, Ideogram4Pipeline};
pub use scheduler::{make_step_intervals, LogitNormalSchedule};
pub use text_encoder::Ideogram4TextEncoder;
pub use transformer::Ideogram4Transformer;

/// Add all MLX Ideogram providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::QUALITY_REGISTRATION)
        .register_memory_strategy(model::QUALITY_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: MODEL_ID,
            contract: |spec| memory_strategy::weights_free_memory_strategy_contract(MODEL_ID, spec),
        })
        .register_memory_behavior(model::QUALITY_MEMORY_BEHAVIOR)
        .register_generator(model::TURBO_REGISTRATION)
        .register_memory_strategy(model::TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: MODEL_ID_TURBO,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(MODEL_ID_TURBO, spec)
            },
        })
        .register_memory_behavior(model::TURBO_MEMORY_BEHAVIOR)
}

/// Build the complete explicit MLX Ideogram provider catalog.
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

        assert_eq!(explicit, ["ideogram_4", "ideogram_4_turbo"]);
        let preview_ids: Vec<_> = registry
            .generators()
            .filter_map(|registration| {
                let descriptor = (registration.descriptor)();
                descriptor
                    .capabilities
                    .supports_preview
                    .then_some(descriptor.id)
            })
            .collect();
        assert_eq!(
            preview_ids, explicit,
            "every and only Ideogram 4 route previews"
        );
    }

    /// sc-22732: both routes now carry a memory-strategy registration, a weights-free contract
    /// fixture and a behavior seam, so the shared conformance walk grades the published ladder.
    #[test]
    fn memory_strategy_registrations_pass_the_weights_free_conformance_walk() {
        let registry = super::provider_registry().unwrap();
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(&registry, &spec);
        gen_core_testkit::memory_strategy::memory_contract_surface_registry_conformance(&registry);
    }
}
