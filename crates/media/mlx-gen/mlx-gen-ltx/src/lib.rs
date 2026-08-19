//! # mlx-gen-ltx
//!
//! LTX-2.3 **video** (text-to-video) provider crate for [`mlx_gen`]. Port of the
//! `mlx-video-with-audio` package's LTX video path (`generate_av.py`, `models/ltx/*`,
//! `models/ltx/video_vae/*`) onto Rust + `mlx-rs`.
//!
//! **Scope:** the full **AudioVideo** path (`generate_av.py`, sc-2684) — synchronized audio+video;
//! `generate()` runs the joint dual-modality denoise and returns video frames + an audio track. Built
//! on the sc-2679 video core + **single-image I2V** (sc-2685) + checkpoint-driven **Q4/Q8** quant
//! (sc-2686) + **LoRA in generate** (sc-2687 — forward-time residuals + per-pass strength over the
//! full video+audio+cross-modal surface; see [`adapters`]). LoKr (sc-2393) is a sibling story.
//!
//! This crate publishes `ltx_2_3` through its explicit provider registry.
//!
//! ## Status (S0–S6 complete)
//! The full text-to-video path is wired and pixel-parity vs the reference `generate_av.py`: Gemma-3
//! tokenizer (byte-exact) → [`LtxTextEncoder`] (Gemma backbone + connector) → seeded noise → the
//! 2-stage distilled denoise ([`pipeline`]: stage-1 half-res → 2× [`upsampler::LatentUpsampler`] →
//! re-noise → stage-2 full-res over the 48-layer [`transformer::LtxDiT`]) → [`vae::LtxVideoVae`]
//! decode → uint8 frames. Built on SPLIT 3-D RoPE (double-precision), an f32 position grid, the
//! distilled sigma schedules, and the legacy dtype-preserving Euler step.
//!
//! The distilled stage-1 sampler is chaos-sensitive, so e2e pixel-parity requires a **bit-exact
//! per-forward DiT** (sc-2842 — the adaLN timestep table must be built in MLX f32, not host f64). Two
//! shipped precisions, both gated bit-exact vs their reference golden: [`transformer::Precision::quant_f32`]
//! (f32 activations × quantized weights — the quality target) and [`transformer::Precision::quant_bf16`]
//! (the reference's native bf16 activations — the production-speed path). The quant geometry (**Q4**/Q8)
//! rides on the checkpoint's `split_model.json` (sc-2686). **I2V** single-image conditioning (sc-2685)
//! is wired into the same 2-stage path ([`conditioning`] + [`pipeline::generate_i2v_latents`], gated
//! bit-exact by `tests/i2v_parity.rs`). **LoRA in generate** (sc-2687) is wired (forward-time residuals
//! + per-pass strength; see [`adapters`]). LoKr (sc-2393) is a sibling.

/// Force a logically contiguous copy at public output boundaries. Host reads return the physical
/// buffer, so transposed arrays must be materialized before exposure.
pub(crate) fn contiguous(x: &mlx_rs::Array) -> mlx_gen::Result<mlx_rs::Array> {
    mlx_gen::array::contiguous(x)
}

pub mod adapters;
pub mod audio_vae;
pub mod bundle;
pub mod conditioning;
pub mod config;
pub mod connector;
pub mod convert;
pub mod diff_vae;
pub mod enhance;
pub mod gemma;
pub mod image_crf;
pub mod memory_strategy;
pub mod model;
pub mod params;
pub mod pipeline;
pub mod positions;
pub mod rope;
pub mod text_encoder;
pub mod tokenizer;
pub mod training;
pub mod transformer;
pub mod upsampler;
pub mod vae;
pub mod vocoder;

pub use adapters::{apply_ltx_adapters, LtxLoraReport};
pub use audio_vae::AudioDecoder;
pub use bundle::{
    assert_gemma_version, declared_layout, declared_model_version, resolve_split_bundle,
    split_component_ids, text_encoder_identity,
};
pub use conditioning::{
    append_keyframe_clip, apply_conditioning, apply_denoise_mask, apply_keyframes,
    keyframe_append_positions, patchify_grid, unpatchify_grid, I2vConditioning, Keyframe,
    VideoTokenState,
};
pub use config::{AudioVaeConfig, LatentLogVar, LtxConfig, LtxVaeConfig, RopeType, VaeBlock};
pub use connector::Connector;
pub use convert::{convert_and_assemble, convert_vae_components, LtxConvertOpts};
pub use diff_vae::{
    DiffVaeTiling, ModelOutputType, NaDiffusionDecoder, NaDiffusionDecoderConfig,
    DIFFUSION_DECODER_COMPONENT,
};
pub use enhance::{clean_response, EnhanceConfig, SampleParams};
pub use image_crf::{condition_image_for_checkpoint, default_image_recompress};
pub use model::{apply_replacement_mask, descriptor, load, Ltx, MODEL_ID, SIZE_MULTIPLE};
pub use params::{
    resolve_generation_params, GuiderParams, LtxGenerationParams, LTX_2_3_PARAMS, LTX_2_5_PARAMS,
};
pub use pipeline::{
    decode_audio_track, decode_to_frames, denoise, denoise_av, denoise_av_tokens,
    generate_av_latents, generate_av_latents_iclora, generate_i2v_latents, generate_t2v,
    generate_t2v_latents, preprocess_conditioning_image, renoise, to_uint8_frames, StageClip,
    StageKeyframe, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
pub use text_encoder::LtxTextEncoder;
// Tiling moved to `mlx_gen` core (shared with the Wan VAE — sc-2808). Re-export the module + config
// so `mlx_gen_ltx::tiling::*` / `mlx_gen_ltx::TilingConfig` keep resolving for existing callers.
pub use config::{VocoderConfig, VocoderGenConfig};
pub use mlx_gen::tiling::{self, TilingConfig};
pub use tokenizer::{Ltx25Tokenizer, LtxTokenizer};
pub use training::{load_trainer, LtxTrainer};
pub use transformer::{to_denoised, AvDiT, LtxDiT, Precision, VideoBlock};
pub use upsampler::{upsample_latents, LatentUpsampler};
pub use vae::LtxVideoVae;
/// Provider-facing LTX geometry, derived from the decoder implementation.
pub const VAE_TILING: mlx_gen::tiling::VaeTiling = LtxVideoVae::VAE_TILING;
// Re-exported as `VocoderGenerator`, not `Generator`: this is the HiFi-GAN vocoder struct, and a
// bare `Generator` at the crate root would shadow the core `mlx_gen::Generator` *trait* name — an
// accidental `use mlx_gen_ltx::Generator` would then compile and mean the wrong thing (F-059).
pub use vocoder::{Generator as VocoderGenerator, LtxVocoder, VocoderWithBwe};

#[cfg(test)]
mod vae_tiling_assignment_tests {
    #[test]
    fn provider_id_resolves_to_the_concrete_decoder_geometry() {
        assert_eq!(super::VAE_TILING, mlx_gen::tiling::VaeTiling::LTX);
        assert_eq!(super::VAE_TILING, super::LtxVideoVae::VAE_TILING);
        assert_eq!(super::vae_tiling(super::MODEL_ID), Some(super::VAE_TILING));
        assert_eq!(super::vae_tiling("not_ltx"), None);
    }
}

// sc-2963 (rollout of the Wan sc-2957 template): when on, the AvDiT's fusable elementwise *glue* —
// adaLN affine (`x·(1+scale)+shift`), the gated residuals (`x+out·gate`), the **tanh-GELU FFN
// activation**, and the split (rotate-halves) RoPE rotation — runs through `mx.compile` so MLX fuses
// each chain into a single Metal kernel. The big quantized GEMMs / SDPA / `mx.fast` norms stay eager.
//
// This is the same machine that gave Wan **+14%/step**. **Bit-exact** to the eager form. **Enabled by
// the production denoise loops** ([`pipeline::denoise`] / [`pipeline::denoise_av`]); **off by default**
// so the reference-parity gates run eager. Dtype-preserving — f32 / bf16 / quantized paths flow through
// unchanged.
//
// The toggle + its RAII [`CompileGlueGuard`] are hoisted into core (F-104); re-export core's so the
// request/thread-local setting is shared with the FLUX family.
pub(crate) use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};

/// Add the MLX LTX generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: MODEL_ID,
            contract: memory_strategy::weights_free_memory_strategy_contract,
            surface_specs: memory_strategy::memory_contract_surface_specs,
        })
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR)
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX LTX provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

/// Resolve the load-bearing VAE geometry for an MLX LTX generator id.
pub fn vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    (provider_id == MODEL_ID).then_some(VAE_TILING)
}

/// Resolve the provider-owned conservative VAE decode working-set peak for an LTX generator id.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<mlx_gen::VideoDecodeMemoryProfile> {
    vae_tiling(provider_id)?;
    pipeline::conservative_video_decode_memory_profile(width, height, frames)
}

#[cfg(test)]
mod explicit_registry_tests {
    use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport, ProviderRegistryBuilder};
    use mlx_gen::{LoadSpec, WeightsSource};

    fn write_named_safetensors(
        path: &std::path::Path,
        tensor_name: &str,
        dtype: &str,
        elements: usize,
        bytes_per_element: usize,
    ) {
        let payload_bytes = elements.checked_mul(bytes_per_element).unwrap();
        let mut tensors = serde_json::Map::new();
        tensors.insert(
            tensor_name.to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": [elements],
                "data_offsets": [0, payload_bytes]
            }),
        );
        let mut header = serde_json::to_vec(&tensors).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + payload_bytes, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_safetensors(
        path: &std::path::Path,
        dtype: &str,
        elements: usize,
        bytes_per_element: usize,
    ) {
        write_named_safetensors(path, "weight", dtype, elements, bytes_per_element);
    }

    fn production_contract_spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("q8");
        let gemma = tmp.path().join("gemma");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::write(
            root.join("split_model.json"),
            r#"{"quantized":true,"quantization_bits":8,"quantization_group_size":64}"#,
        )
        .unwrap();
        for (name, dtype, elements, bytes_per_element) in [
            ("connector.safetensors", "F32", 11, 4),
            ("transformer.safetensors", "F32", 13, 4),
            ("upsampler.safetensors", "BF16", 17, 2),
            ("vae_decoder.safetensors", "BF16", 19, 2),
            ("audio_vae.safetensors", "BF16", 23, 2),
            ("vocoder.safetensors", "BF16", 29, 2),
        ] {
            write_safetensors(&root.join(name), dtype, elements, bytes_per_element);
        }
        write_safetensors(&gemma.join("model.safetensors"), "F32", 31, 4);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.text_encoder = Some(WeightsSource::Dir(gemma));
        spec
    }

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

        assert_eq!(explicit_generators, ["ltx_2_3"]);
        assert_eq!(explicit_trainers, ["ltx_2_3"]);
    }

    #[test]
    fn registered_contract_is_present_and_removal_changes_resolution_to_none() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = production_contract_spec(&tmp);
        let registry = super::provider_registry().unwrap();
        let contract = registry
            .memory_strategy_contract(super::MODEL_ID, &spec)
            .unwrap()
            .expect("LTX production registry must resolve its memory contract");
        assert!(matches!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        // Gemma + connector are retained as bf16, transformer + upsampler preserve storage, and all
        // three decode components are unconditionally materialized as f32.
        assert_eq!(contract.asset_facts.conditioning_bytes, 31 * 2 + 11 * 2);
        assert_eq!(contract.asset_facts.transformer_bytes, 13 * 4 + 17 * 2);
        assert_eq!(contract.asset_facts.decoder_bytes, 19 * 4 + 23 * 4 + 29 * 4);
        assert_eq!(
            contract.asset_facts.base_bytes,
            31 * 2 + 11 * 2 + 13 * 4 + 17 * 2 + 19 * 4 + 23 * 4 + 29 * 4
        );
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            super::memory_strategy::CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            super::memory_strategy::resolved_numeric_tier(&spec)
                .unwrap()
                .quant,
            Some(mlx_gen::Quant::Q8)
        );
        assert_eq!(
            super::memory_strategy::route_overlay(&spec),
            None,
            "canonical dense-bf16 Gemma is the base route, not an overlay"
        );
        let mut mismatched_tier = spec.clone();
        mismatched_tier.quantize = Some(mlx_gen::Quant::Q4);
        assert!(registry
            .memory_strategy_contract(super::MODEL_ID, &mismatched_tier)
            .is_err());

        // Negative mutation: the identical executable provider without its memory registration
        // must resolve `None`. This guards the registry row from becoming decorative fixture-only
        // metadata, the failure SC-19109 is repairing.
        let without_contract = ProviderRegistryBuilder::new()
            .register_generator(super::model::REGISTRATION)
            .register_trainer(super::training::TRAINER_REGISTRATION)
            .build()
            .unwrap();
        assert!(without_contract
            .memory_strategy_contract(super::MODEL_ID, &spec)
            .unwrap()
            .is_none());
    }

    #[test]
    fn quantized_gemma_route_does_not_borrow_the_canonical_bf16_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = production_contract_spec(&tmp);
        let WeightsSource::Dir(root) = &spec.weights else {
            panic!("fixture must use a model directory");
        };
        let Some(WeightsSource::Dir(gemma)) = spec.text_encoder.as_ref() else {
            panic!("fixture must use a Gemma directory");
        };
        let split = super::config::SplitModel::from_model_dir(root).unwrap();
        let quant = super::gemma::GemmaQuant { group: 64, bits: 8 };

        assert!(
            super::memory_strategy::contract_for_loaded(&spec, &split, gemma, Some(quant))
                .unwrap()
                .is_none()
        );

        std::fs::write(
            gemma.join("config.json"),
            r#"{"quantization":{"mode":"affine","group_size":64,"bits":8}}"#,
        )
        .unwrap();
        let error = super::provider_registry()
            .unwrap()
            .memory_strategy_contract(super::MODEL_ID, &spec)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires canonical dense-bf16 Gemma"));
    }

    #[test]
    fn dense_delta_adapters_keep_loading_but_do_not_publish_underpriced_contracts() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = production_contract_spec(&tmp);
        let WeightsSource::Dir(root) = &spec.weights else {
            panic!("fixture must use a model directory");
        };
        let Some(WeightsSource::Dir(gemma)) = spec.text_encoder.as_ref() else {
            panic!("fixture must use a Gemma directory");
        };
        let split = super::config::SplitModel::from_model_dir(root).unwrap();
        let adapter = tmp.path().join("third-party-lokr.safetensors");
        write_named_safetensors(
            &adapter,
            "transformer_blocks.0.attn.to_q.lokr_w1",
            "BF16",
            8,
            2,
        );
        spec.adapters.push(mlx_gen::AdapterSpec::new(
            adapter,
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));

        assert!(
            super::memory_strategy::contract_for_loaded(&spec, &split, gemma, None)
                .unwrap()
                .is_none()
        );
        let error = super::provider_registry()
            .unwrap()
            .memory_strategy_contract(super::MODEL_ID, &spec)
            .unwrap_err()
            .to_string();
        assert!(error.contains("LoKr/LoHa routes"));
    }

    #[test]
    fn weights_free_registry_behavior_is_executable_for_each_implemented_rung() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-fixture".into()))
            .with_quant(mlx_gen::Quant::Q8);
        gen_core_testkit::memory_strategy_registry_conformance(
            &super::provider_registry().unwrap(),
            &spec,
        );
    }
}

#[cfg(test)]
mod compile_glue_guard_tests {
    use super::{compile_glue, set_compile_glue, CompileGlueGuard};

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

        // Leave this thread eager, as the reference-parity gates expect.
        set_compile_glue(false);
    }
}
