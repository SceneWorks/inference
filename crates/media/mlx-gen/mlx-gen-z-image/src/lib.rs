//! # mlx-gen-z-image
//!
//! The **Z-Image** (Tongyi Z-Image-turbo) provider crate for [`mlx-gen`](mlx_gen). Depends only
//! on the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator` contract, and
//! the explicit registry). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! Ported & parity-proven against the frozen Python mflux fork (tolerance 1e-2 — Metal runs
//! fp32 matmul in reduced precision) and validated end-to-end on real bf16 weights (sc-2352):
//! the Qwen text encoder (prompt → `cap_feats`), the flow-match Euler scheduler, the DiT
//! transformer (block, context block, timestep / RoPE embedders, final layer, full forward),
//! and the VAE encoder + decoder. [`load`] assembles the model from a snapshot
//! directory and [`ZImageTurbo::generate`](model::ZImageTurbo) runs the full prompt→image
//! pipeline, including img2img (VAE-encode an init image + noise blend, sc-2533) and whole-model
//! Q4/Q8 quantization (sc-2532).

pub mod adapters;
pub mod attention;
// Ladder rung 4 (SC-15754): the family half of bounded transformer residency — how a Z-Image block is
// rebuilt from its snapshot, quantized and adapted like its resident twin. The window lifecycle itself
// is the shared `mlx_gen::block_residency` (SC-15750), never re-implemented here.
mod block_stream;
mod comfyui;
pub mod context_block;
pub mod control_transformer;
pub mod control_transformer_block;
pub mod convert;
pub mod feed_forward;
pub mod final_layer;
// Shared memory-strategy contract adoption (SC-15449) + the SC-15615 rung-3 finding.
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod model_base;
pub mod model_base_control;
pub mod model_control;
pub mod pipeline;
pub mod preview;
pub mod quant;
pub mod rope_embedder;
pub mod text_encoder;
pub mod timestep_embedder;
pub mod training;
pub mod transformer;
pub mod transformer_block;
pub mod vae;

pub const TOKENIZER_CONTRACT: mlx_gen::gen_core::EncoderTokenizerContract =
    mlx_gen::gen_core::EncoderTokenizerContract {
        family: "qwen3",
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
        ],
    };
pub const PROMPT_EXECUTIONS: &[mlx_gen::gen_core::EncoderPromptExecutionContract] = &[
    mlx_gen::gen_core::EncoderPromptExecutionContract {
        purpose: "z_image_prompt",
        template: mlx_gen::gen_core::EncoderPromptTemplate::QwenInstruct,
        add_special_tokens: true,
        length: mlx_gen::gen_core::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 512 },
        padding: mlx_gen::gen_core::EncoderPromptPadding::RightToMax {
            pad_token_id: 151_643,
        },
        prefix_trim: 0,
    },
    mlx_gen::gen_core::EncoderPromptExecutionContract {
        purpose: "z_image_empty_negative",
        template: mlx_gen::gen_core::EncoderPromptTemplate::QwenInstruct,
        add_special_tokens: true,
        length: mlx_gen::gen_core::EncoderPromptLengthPolicy::Unbounded,
        padding: mlx_gen::gen_core::EncoderPromptPadding::None,
        prefix_trim: 0,
    },
];

pub const ENCODER_CONTRACT: mlx_gen::gen_core::EncoderContract =
    mlx_gen::gen_core::EncoderContract {
        architecture: "qwen3",
        hidden_size: 2560,
        intermediate_size: 9728,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 151_936,
        output_width: 2560,
        loaded_hidden_layers: 36,
        requires_final_norm: false,
        requires_lm_head: false,
        hidden_activation: "silu",
        attention_dropout: mlx_gen::gen_core::EncoderConfigFloat::new(0.0),
        rms_norm_eps: mlx_gen::gen_core::EncoderConfigFloat::new(1e-6),
        qk_norm_eps: Some(mlx_gen::gen_core::EncoderConfigFloat::new(1e-5)),
        rope_theta: mlx_gen::gen_core::EncoderConfigFloat::new(1_000_000.0),
        max_position_embeddings: 40_960,
        attention_bias: Some(false),
        tie_word_embeddings: Some(true),
        tokenizer: TOKENIZER_CONTRACT,
        prompt_executions: PROMPT_EXECUTIONS,
        bos_token_id: Some(151_643),
        eos_token_id: Some(151_645),
        image_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
        mrope_section: &[],
        mrope_interleaved: None,
        selected_hidden_layers: &[35],
        packing: Some(mlx_gen::gen_core::EncoderPackingContract {
            group_size: 64,
            pack_embedding: true,
            pack_lm_head: false,
            supports_file: true,
        }),
        dense_storage_dtype_probe: None,
    };

pub use adapters::apply_z_image_adapters;
pub use context_block::ZImageContextBlock;
pub use control_transformer::{ZImageControlTransformer, CONTROL_IN_DIM};
pub use control_transformer_block::ZImageControlBlock;
pub use final_layer::FinalLayer;
pub use loader::{
    load_control_transformer, load_text_encoder, load_text_encoder_streamable, load_tokenizer,
    load_transformer, load_vae,
};
pub use model::{
    descriptor, load, load_from_comfyui_checkpoint, load_from_comfyui_components, ZImageTurbo,
    MODEL_ID, SIZE_MULTIPLE,
};
// The base (`z_image`, sc-8320) and control (`z_image_turbo_control`) variants each register
// publish their own registration constants; their `descriptor`/`load`/`MODEL_ID` items share the names of the
// turbo model's, so reach them through their module paths (consumers use the registry ids
// `"z_image"` / `"z_image_turbo_control"`). The base reuses the identical `ZImageTransformer` — only
// the scheduler shift (6.0 vs 3.0), default steps (50 vs 4), and the CFG path differ.
pub use model_base::ZImage;
// The base control variant (`z_image_control`, sc-8251) publishes its own registration constant; its
// `descriptor`/`load`/`MODEL_ID` share the names of the turbo control model's, so reach them through
// the module path (consumers use the registry id `"z_image_control"`). It reuses the identical
// `ZImageControlTransformer` as the turbo control variant — only the base descriptor (CFG) + the base
// control repo differ.
pub use model_base_control::ZImageControl;
pub use model_control::ZImageTurboControl;
pub use pipeline::{
    add_noise_by_interpolation, create_noise, decoded_to_image, denoise, denoise_cfg_with_progress,
    denoise_cfg_with_progress_and_preview, denoise_control_cfg_with_progress,
    denoise_control_cfg_with_progress_and_preview, denoise_control_with_progress,
    denoise_control_with_progress_and_preview, denoise_with_progress,
    denoise_with_progress_and_preview, encode_control_context, encode_init_latents, init_time_step,
    pack_latents, preprocess_init_image, slice_valid, unpack_latents,
};
pub use rope_embedder::RopeEmbedder;
pub use timestep_embedder::TimestepEmbedder;
pub use training::{LoraTarget, ZImageTurboTrainer};
pub use transformer::{ZImageTransformer, ZImageTransformerConfig};
pub use transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};

/// Add every Z-Image MLX provider to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_activation_memory(model::ACTIVATION_MEMORY_REGISTRATION)
        .register_generator(model_base::REGISTRATION)
        .register_generator(model_base_control::REGISTRATION)
        .register_generator(model_control::REGISTRATION)
        .register_memory_strategy(model::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: model::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(model::MODEL_ID, spec)
            },
        })
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_strategy(model_base::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: model_base::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(model_base::MODEL_ID, spec)
            },
        })
        .register_memory_behavior(model_base::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_strategy(model_base_control::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: model_base_control::MODEL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(
                    model_base_control::MODEL_ID,
                    spec,
                )
            },
        })
        .register_memory_behavior(model_base_control::MEMORY_BEHAVIOR_REGISTRATION)
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
        .register_trainer(training::REGISTRATION)
}

/// Build the complete explicit Z-Image MLX provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

// sc-2963 compiled-glue toggle (rollout of the Wan sc-2957 template): when on, the DiT's fusable
// elementwise *glue* — the SwiGLU FFN activation (`silu(h1)·h3`), the gated residuals
// (`x+gate·norm(out)`), the complex RoPE rotation, and the control-branch hint injection
// (`x+hint·scale`) — runs through `mx.compile` so MLX fuses each chain into a single Metal kernel.
// The toggle + its RAII [`CompileGlueGuard`] are hoisted into core (F-104); re-export core's so the
// process-global is shared with the FLUX family rather than each crate hand-rolling its own
// `AtomicBool`. **Bit-exact** to the eager form; **enabled by the production denoise loops** (turbo +
// control, [`pipeline`]); left **off by default** so the reference-parity gates run eager. The
// mixed-precision dtype flow (base bf16, f32 `control_context`, sc-2720) is preserved unchanged.
pub(crate) use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};

#[cfg(test)]
mod compile_glue_guard_tests {
    use super::{compile_glue, set_compile_glue, CompileGlueGuard};

    // Single-threaded test runner (`.cargo/config.toml` RUST_TEST_THREADS=1) makes the
    // process-global `COMPILE_GLUE` safe to assert on, matching the existing `set_compile_glue`
    // A/B tests in feed_forward / control_transformer.
    #[test]
    fn guard_enables_then_restores_prior_value() {
        // Prior off → on within scope → restored off on drop (the doc's "eager by default" intent).
        set_compile_glue(false);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue(), "guard enables compiled glue for its scope");
        }
        assert!(!compile_glue(), "guard restores the prior (off) on drop");

        // Restores the *prior* value, not a hardcoded false: prior on stays on after drop.
        set_compile_glue(true);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue());
        }
        assert!(compile_glue(), "guard restores the prior (on) on drop");

        // Leave the global eager, as the reference-parity gates expect.
        set_compile_glue(false);
    }
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

    fn snapshot(tmp: &tempfile::TempDir, tag: &str) -> std::path::PathBuf {
        let root = tmp.path().join(format!("z-image-{tag}"));
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
        let mut explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        explicit_generators.sort();
        assert_eq!(
            explicit_generators,
            [
                "z_image",
                "z_image_control",
                "z_image_turbo",
                "z_image_turbo_control"
            ]
        );

        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(explicit_trainers, ["z_image_turbo"]);
    }

    /// The four `register_memory_strategy` calls are what makes the SC-15449 contract resolvable before
    /// weights load. Without this test, dropping any one of them is green — every other contract test
    /// builds the contract directly instead of going through the registry.
    #[test]
    fn every_variant_resolves_its_memory_strategy_contract_through_the_registry() {
        let tmp = tempfile::tempdir().unwrap();
        use mlx_gen::gen_core::{LoadSpec, MemoryStrategy, MemoryStrategySupport, WeightsSource};

        let registry = super::provider_registry().unwrap();
        // SC-15998: rung 4 is declared per load — a re-openable snapshot dir with deferred
        // materialization, independent from phase residency.
        // The registry must hand back the same contract the direct builder produces for that load.
        let root = snapshot(&tmp, "registry-memory");
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        for id in [
            "z_image_turbo",
            "z_image",
            "z_image_turbo_control",
            "z_image_control",
        ] {
            let contract = registry
                .memory_strategy_contract(id, &spec)
                .unwrap()
                .unwrap_or_else(|| panic!("{id} must register a memory-strategy contract"));
            assert_eq!(contract.provider_id, id);
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                super::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
                "{id}"
            );
            // The registry-resolved contract must be the same one the direct builder produces —
            // every rung Implemented on a snapshot load (SC-15510 / SC-15754), each with its
            // recorded parameter domain.
            for strategy in MemoryStrategy::ALL {
                assert!(
                    matches!(
                        contract.capability(strategy).map(|c| &c.support),
                        Some(MemoryStrategySupport::Implemented)
                    ),
                    "{id}: {strategy:?}"
                );
            }
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .parameters
                    .attention_chunk_sizes,
                vec![super::memory_strategy::ATTENTION_CHUNK_SIZE],
                "{id}"
            );
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .parameters
                    .transformer_window_sizes,
                super::memory_strategy::TRANSFORMER_WINDOW_SIZES.to_vec(),
                "{id}"
            );
            assert!(
                contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .parameters
                    .decode_tile_edges
                    .len()
                    > 1,
                "{id}: the decode ladder must be sweepable, not a single point"
            );
        }
        std::fs::remove_dir_all(root).ok();
    }
}
