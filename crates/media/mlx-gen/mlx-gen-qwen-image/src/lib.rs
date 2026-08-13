//! # mlx-gen-qwen-image
//!
//! The **Qwen-Image** (+ Qwen-Image-Edit) provider crate for [`mlx-gen`](mlx_gen). Depends only on
//! the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator` contract, and the
//! explicit registry). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! Ported from the frozen Python mflux fork (`~/repos/mflux/src/mflux/models/qwen/`) and
//! parity-proven against it on real bf16 weights. Shipped: **Qwen-Image T2I** (`qwen_image`,
//! sc-2348) and **Qwen-Image-Edit** (`qwen_image_edit`, sc-2465) — the causal-Conv3d VAE, the
//! Qwen2.5-VL text encoder, the 60-layer dual-stream MMDiT, the Qwen2-VL image processor +
//! Qwen2.5-VL vision transformer + reference-latent conditioning (Edit), and transformer-only
//! Q4/Q8 quantization (sc-2565; the fork keeps the text encoder + VAE dense). Also wired: LoRA/LoKr
//! consumption (sc-2528), multi-image Edit (sc-2529), T2I img2img (sc-2530), and few-step
//! **Lightning** acceleration — the `lightning` sampler ([`sampler::lightning`], sc-2909): the
//! official lightx2v recipe (static flow-match shift 3.0, CFG-off single forward) for both T2I and
//! Edit, requiring the matching distillation LoRA via `spec.adapters`.

pub mod adapters;
mod block_stream;
pub mod control_transformer;
pub mod convert;
pub mod image_processor;
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod model_control;
pub mod model_edit;
pub mod pipeline;
pub mod preview;
pub mod quant;
pub mod sampler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vl_tokenizer;

pub use adapters::apply_qwen_adapters;
pub use control_transformer::{QwenFunControlBranch, QwenFunControlConfig};
pub use image_processor::{ImageInput, ProcessedImage, QwenImageProcessor};
pub use loader::{
    load_controlnet, load_text_encoder, load_tokenizer, load_transformer, load_transformer_edit,
    load_vae, load_vision_encoder, load_vision_language_encoder,
};
pub use model::{descriptor, load, QwenImage, MODEL_ID, SIZE_MULTIPLE};
pub use model_control::QwenImageControl;
pub use model_edit::QwenImageEdit;
pub use pipeline::{
    add_noise_by_interpolation, compute_guided_noise, create_noise, decoded_to_image,
    denoise_control_with_progress, denoise_edit_with_progress, denoise_with_progress,
    encode_init_latents, init_time_step, pack_latents, preprocess_init_image, qwen_scheduler,
    unpack_latents,
};
pub use sampler::{lightning, FlowMatchSampler, LIGHTNING_SHIFT};
pub use text_encoder::{QwenTextEncoder, QwenTextEncoderConfig};
pub use transformer::{QwenTransformer, QwenTransformerConfig};
pub use vae::QwenVae;
pub use vl_tokenizer::{
    encode_reference_latents, preprocess_edit_image, tokenize_edit, tokenize_edit_text, EditImage,
    EditInputs,
};

/// Shared-optimization toggles whose production call sites this provider can actually execute.
/// Availability never substitutes for the request-local `Applied` receipt required by P6.
pub const BENCHMARK_TOGGLE_CAPABILITIES: &[&str] = &[mlx_gen::diagnostics::RETAINED_COMPILATION];

/// sc-16195 Apple-Silicon warm sweep: base Qwen-Image q8 peaked at 7.661 GiB at 1024².
/// Rounded upward to 7.67 GiB and applies across weight tiers because activations stay bf16.
/// Control/Edit are distinct unmeasured routes.
pub const ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: MODEL_ID,
        anchor: mlx_gen::ActivationMemoryAnchor {
            bytes_1024: 8_235_599_791,
        },
    };

/// Add all MLX Qwen-Image generators to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_activation_memory(ACTIVATION_MEMORY_REGISTRATION)
        .register_generator(model_control::REGISTRATION)
        .register_generator(model_edit::REGISTRATION)
        .register_memory_strategy(model::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: model::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(model::MODEL_ID, spec)
            },
        })
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_strategy(model_control::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: model_control::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(
                    model_control::MODEL_ID,
                    spec,
                )
            },
        })
        .register_memory_behavior(model_control::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_strategy(model_edit::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: model_edit::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(model_edit::MODEL_ID, spec)
            },
        })
        .register_memory_behavior(model_edit::MEMORY_BEHAVIOR_REGISTRATION)
}

/// Build the complete explicit MLX Qwen-Image provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    fn write_minimal_safetensors(path: &std::path::Path) {
        let mut header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).unwrap();
    }

    fn snapshot(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        let root = tmp.path().join("qwen-registry");
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        root
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let descriptors: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .collect();
        let explicit: Vec<_> = descriptors.iter().map(|descriptor| descriptor.id).collect();

        assert_eq!(
            explicit,
            ["qwen_image", "qwen_image_control", "qwen_image_edit"]
        );
        let preview_support: std::collections::BTreeMap<_, _> = descriptors
            .iter()
            .map(|descriptor| (descriptor.id, descriptor.capabilities.supports_preview))
            .collect();
        assert_eq!(
            preview_support,
            std::collections::BTreeMap::from([
                ("qwen_image", true),
                ("qwen_image_control", false),
                ("qwen_image_edit", true),
            ])
        );
    }

    #[test]
    fn every_variant_resolves_its_memory_strategy_contract() {
        let tmp = tempfile::tempdir().unwrap();
        use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, WeightsSource};

        let registry = super::provider_registry().unwrap();
        let root = snapshot(&tmp);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        for id in ["qwen_image", "qwen_image_control", "qwen_image_edit"] {
            let contract = registry
                .memory_strategy_contract(id, &spec)
                .unwrap()
                .unwrap_or_else(|| panic!("{id} must register a memory-strategy contract"));
            assert_eq!(contract.provider_id, id);
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                super::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
            );
        }
        std::fs::remove_dir_all(root).ok();
    }
}
