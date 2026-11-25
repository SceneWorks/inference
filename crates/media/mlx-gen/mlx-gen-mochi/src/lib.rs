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
    registry
        .register_generator(model::REGISTRATION)
        .register_video_plan(mlx_gen::gen_core::VideoPlanRegistration {
            provider_id: MODEL_ID,
            compile: model::compile_video_plan,
        })
}

/// Build the complete explicit MLX Mochi 1 provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn registered_video_plan_reads_shared_codec_metadata_before_weights() {
        use mlx_gen::gen_core::{GenerationRequest, LoadSpec, WeightsSource};
        let root = tempfile::tempdir().unwrap();
        let tier = root.path().join("bf16");
        std::fs::create_dir_all(tier.join("transformer")).unwrap();
        std::fs::write(tier.join("split_model.json"), r#"{"quantized":false}"#).unwrap();
        std::fs::create_dir(root.path().join("text_encoder")).unwrap();
        std::fs::create_dir(root.path().join("vae")).unwrap();
        // Non-default metadata detects accidental hard-coding of the real 6x8 checkpoint.
        std::fs::write(
            root.path().join("vae/config.json"),
            r#"{"temporal_expansions":[1,1,3],"spatial_expansions":[1,2,2]}"#,
        )
        .unwrap();
        let req = GenerationRequest {
            prompt: "kitten".into(),
            width: 64,
            height: 48,
            frames: Some(7),
            ..Default::default()
        };
        let spec = LoadSpec::new(WeightsSource::Dir(tier));
        let registry = super::provider_registry().unwrap();
        let plan = registry
            .compile_video_plan(super::MODEL_ID, &spec, &req)
            .unwrap()
            .unwrap();
        let phase = &plan.layout().phases[0];
        assert_eq!(
            plan.minimum_denoise_live_bytes(plan.layout().domain)
                .unwrap(),
            16_891_904
        );
        assert_eq!(
            (phase.latent_frames, phase.latent_bytes),
            (3, 3 * 12 * 16 * 12 * 4)
        );
        assert_eq!((phase.decoded_frames, phase.decode_chunk_frames), (9, 3));
        assert_eq!(plan.layout().output.frames, 7);
        // All seven cropped f32 frames are live beside their concatenation. The two
        // contiguous RGB copies (one array and the returned vectors) are a smaller peak.
        let rgb_frame = 64 * 48 * 3;
        assert_eq!(
            plan.minimum_decode_live_bytes(mlx_gen::gen_core::MemoryBudgetDomain::UnifiedMemory)
                .unwrap(),
            14 * 4 * rgb_frame
        );
        let invalid = GenerationRequest {
            frames: Some(8),
            ..req
        };
        assert!(registry
            .compile_video_plan(super::MODEL_ID, &spec, &invalid)
            .is_err());
    }

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

#[cfg(test)]
pub(crate) mod model_inference_device;
