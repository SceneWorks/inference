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

pub const TOKENIZER_CONTRACT: mlx_gen::gen_core::EncoderTokenizerContract =
    mlx_gen::gen_core::EncoderTokenizerContract {
        family: "qwen2_5_vl",
        binding: mlx_gen::gen_core::EncoderTokenizerBinding::RetainBase,
        artifact_candidates: &["tokenizer/tokenizer.json"],
        required_tokens: &[
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_endoftext",
                literal: "<|endoftext|>",
                id: 151_643,
                config_field: Some("bos_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_im_start",
                literal: "<|im_start|>",
                id: 151_644,
                config_field: None,
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_im_end",
                literal: "<|im_end|>",
                id: 151_645,
                config_field: Some("eos_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_vision_start",
                literal: "<|vision_start|>",
                id: 151_652,
                config_field: Some("vision_start_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_vision_end",
                literal: "<|vision_end|>",
                id: 151_653,
                config_field: Some("vision_end_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_image_pad",
                literal: "<|image_pad|>",
                id: 151_655,
                config_field: Some("image_token_id"),
            },
        ],
    };
pub const PROMPT_EXECUTIONS: &[mlx_gen::gen_core::EncoderPromptExecutionContract] = &[
    mlx_gen::gen_core::EncoderPromptExecutionContract {
        purpose: "qwen_image_t2i",
        template: mlx_gen::gen_core::EncoderPromptTemplate::QwenImage,
        add_special_tokens: true,
        length: mlx_gen::gen_core::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 1058 },
        padding: mlx_gen::gen_core::EncoderPromptPadding::None,
        prefix_trim: 34,
    },
    mlx_gen::gen_core::EncoderPromptExecutionContract {
        purpose: "qwen_image_edit",
        template: mlx_gen::gen_core::EncoderPromptTemplate::QwenImageEdit,
        add_special_tokens: true,
        length: mlx_gen::gen_core::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 1058 },
        padding: mlx_gen::gen_core::EncoderPromptPadding::None,
        prefix_trim: 64,
    },
];

pub const ENCODER_CONTRACT: mlx_gen::gen_core::EncoderContract =
    mlx_gen::gen_core::EncoderContract {
        architecture: "qwen2_5_vl_text",
        hidden_size: 3584,
        intermediate_size: 18_944,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: 4,
        head_dim: 128,
        vocab_size: 152_064,
        output_width: 3584,
        loaded_hidden_layers: 28,
        requires_final_norm: true,
        requires_lm_head: false,
        hidden_activation: "silu",
        attention_dropout: mlx_gen::gen_core::EncoderConfigFloat::new(0.0),
        rms_norm_eps: mlx_gen::gen_core::EncoderConfigFloat::new(1e-6),
        qk_norm_eps: None,
        rope_theta: mlx_gen::gen_core::EncoderConfigFloat::new(1_000_000.0),
        max_position_embeddings: 128_000,
        attention_bias: mlx_gen::gen_core::EncoderConfigBool::Optional(true),
        tie_word_embeddings: mlx_gen::gen_core::EncoderConfigBool::Required(false),
        tokenizer: TOKENIZER_CONTRACT,
        prompt_executions: PROMPT_EXECUTIONS,
        bos_token_id: Some(151_643),
        eos_token_id: Some(151_645),
        image_token_id: Some(151_655),
        vision_start_token_id: Some(151_652),
        vision_end_token_id: Some(151_653),
        mrope_section: &[16, 24, 24],
        mrope_interleaved: None,
        selected_hidden_layers: &[28],
        packing: None,
        dense_storage_dtype_probe: None,
    };

pub const VISION_ENCODER_CONTRACT: mlx_gen::gen_core::VisionEncoderContract =
    mlx_gen::gen_core::VisionEncoderContract {
        architecture: mlx_gen::gen_core::VisionEncoderArchitecture::Qwen2_5Vl,
        hidden_size: 1280,
        intermediate_size: 3420,
        num_hidden_layers: 32,
        num_attention_heads: 16,
        output_width: 3584,
        hidden_activation: "silu",
        rope_theta: mlx_gen::gen_core::EncoderConfigFloat::new(10_000.0),
        normalization_eps: mlx_gen::gen_core::EncoderConfigFloat::new(1e-6),
        patch_size: 14,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        in_channels: 3,
        num_position_embeddings: None,
        deepstack_visual_indexes: &[],
        window_size: Some(112),
        full_attention_block_indexes: &[7, 15, 23, 31],
    };

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
pub const BENCHMARK_TOGGLE_CAPABILITIES: &[&str] = &[
    mlx_gen::diagnostics::RETAINED_COMPILATION,
    mlx_gen::diagnostics::EXACT_EPILOGUES,
];

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
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: model::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(model::MODEL_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: model::MODEL_ID,
                contract: |surface| {
                    memory_strategy::weights_free_memory_surface_contract(model::MODEL_ID, surface)
                },
            },
        )
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_strategy(model_control::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
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
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: model_edit::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(model_edit::MODEL_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: model_edit::MODEL_ID,
                contract: |surface| {
                    memory_strategy::weights_free_memory_surface_contract(
                        model_edit::MODEL_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model_edit::MEMORY_BEHAVIOR_REGISTRATION)
}

/// Build the complete explicit MLX Qwen-Image provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    /// One line per selector: the whole ladder's per-surface disposition, in rung order 0..4.
    /// `I` = Implemented, `S` = StructurallyNotApplicable, `-` = Missing.
    fn ladder_lines(provider_id: &str) -> Vec<String> {
        use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport};

        let registry = super::provider_registry().unwrap();
        registry
            .memory_contract_surfaces()
            .unwrap()
            .iter()
            .filter(|surface| surface.contract.provider_id == provider_id)
            .map(|surface| {
                let ladder: String = MemoryStrategy::ALL
                    .iter()
                    .map(
                        |strategy| match surface.contract.capability(*strategy).unwrap().support {
                            MemoryStrategySupport::Implemented => 'I',
                            MemoryStrategySupport::StructurallyNotApplicable { .. } => 'S',
                            MemoryStrategySupport::Missing => '-',
                        },
                    )
                    .collect();
                format!("{} {ladder}", surface.selector.id())
            })
            .collect()
    }

    /// sc-21510: rung 4 is tier-agnostic on the base and edit routes. The selector names an
    /// already-resolved artifact tier, so Q4/Q8 deferred-materialization surfaces stream exactly as
    /// BF16 does. Every other (surface, rung) disposition is unchanged, and Control stays excluded
    /// from rungs 3 and 4 at every tier because its five-block branch is unbounded.
    #[test]
    fn published_ladder_surface_is_pinned_per_selector() {
        assert_eq!(
            ladder_lines(super::model::MODEL_ID),
            [
                "bf16:resident:eager I-II-",
                "bf16:resident:deferred I-II-",
                "bf16:sequential:eager IIII-",
                "bf16:sequential:deferred IIIII",
                "q4:resident:eager I-II-",
                "q4:resident:deferred I-II-",
                "q4:sequential:eager IIII-",
                "q4:sequential:deferred IIIII",
                "q8:resident:eager I-II-",
                "q8:resident:deferred I-II-",
                "q8:sequential:eager IIII-",
                "q8:sequential:deferred IIIII",
            ]
        );
        assert_eq!(
            ladder_lines(super::model_edit::MODEL_ID),
            ladder_lines(super::model::MODEL_ID)
        );
        assert_eq!(
            ladder_lines(super::model_control::MODEL_ID),
            [
                "bf16:resident:eager I-I--",
                "bf16:resident:deferred I-I--",
                "bf16:sequential:eager III--",
                "bf16:sequential:deferred III--",
                "q4:resident:eager I-I--",
                "q4:resident:deferred I-I--",
                "q4:sequential:eager III--",
                "q4:sequential:deferred III--",
                "q8:resident:eager I-I--",
                "q8:resident:deferred I-I--",
                "q8:sequential:eager III--",
                "q8:sequential:deferred III--",
            ]
        );
    }

    #[test]
    fn qwen_authored_attention_bias_must_match_the_biasful_runtime() {
        let tmp = tempfile::tempdir().unwrap();
        let encoder = tmp.path().join("encoder");
        gen_core_testkit::write_encoder_contract_fixture(&encoder, super::ENCODER_CONTRACT)
            .unwrap();
        super::ENCODER_CONTRACT
            .validate_source(&mlx_gen::WeightsSource::Dir(encoder.clone()))
            .expect("omission must select the biasful runtime behavior");
        let config_path = encoder.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["attention_bias"] = serde_json::json!(false);
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let error = super::ENCODER_CONTRACT
            .validate_source(&mlx_gen::WeightsSource::Dir(encoder))
            .unwrap_err()
            .to_string();
        assert!(error.contains("attention_bias"), "{error}");
        assert!(error.contains("expected true"), "{error}");
    }

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
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &root.join("text_encoder"),
            super::ENCODER_CONTRACT,
            super::VISION_ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
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

    #[test]
    fn registry_footprints_price_only_route_materialized_conditioning() {
        use mlx_gen::{LoadSpec, WeightsSource};

        let tmp = tempfile::tempdir().unwrap();
        let registry = super::provider_registry().unwrap();
        let root = snapshot(&tmp);
        let base_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let base = registry
            .footprint("qwen_image", &base_spec)
            .unwrap()
            .unwrap();
        let control = registry
            .footprint("qwen_image_control", &base_spec)
            .unwrap()
            .unwrap();
        let edit = registry
            .footprint("qwen_image_edit", &base_spec)
            .unwrap()
            .unwrap();
        assert_eq!(base.text_encoder, control.text_encoder);
        assert!(edit.text_encoder > base.text_encoder);

        let language_only = tmp.path().join("alternate-language");
        gen_core_testkit::write_encoder_contract_fixture(&language_only, super::ENCODER_CONTRACT)
            .unwrap();
        let complete = tmp.path().join("alternate-complete");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &complete.join("text_encoder"),
            super::ENCODER_CONTRACT,
            super::VISION_ENCODER_CONTRACT,
        )
        .unwrap();

        let language_spec = base_spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(language_only));
        let complete_spec = base_spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(complete));
        for id in ["qwen_image", "qwen_image_control", "qwen_image_edit"] {
            let language = registry.footprint(id, &language_spec).unwrap().unwrap();
            let multimodal = registry.footprint(id, &complete_spec).unwrap().unwrap();
            assert_eq!(
                language.text_encoder, multimodal.text_encoder,
                "{id}: ignored alternate visual tensors must not be priced"
            );
        }
        let selected_t2i = registry
            .footprint("qwen_image", &language_spec)
            .unwrap()
            .unwrap();
        let selected_edit = registry
            .footprint("qwen_image_edit", &language_spec)
            .unwrap()
            .unwrap();
        assert_eq!(
            selected_edit.text_encoder - selected_t2i.text_encoder,
            edit.text_encoder - base.text_encoder,
            "edit must always add the exact builtin vision side once"
        );
    }
}
