//! # mlx-gen-qwen-image-2-1
//!
//! The **Qwen-Image 2.1** provider crate for [`mlx-gen`](mlx_gen) (sc-24108): a native MLX port of
//! the frozen upstream `QwenImage21Pipeline` — Qwen3-VL-8B text conditioning, the 32-layer
//! single-stream block-causal DiT, the resolution-shifted flow-match Euler schedule, and the
//! 64-channel / 16× **RGBA** autoencoder — registered as the distinct engine id
//! [`MODEL_ID`] (`qwen_image_2_1`). It is **not** a checkpoint alias of the 2512
//! `mlx-gen-qwen-image` crate: the DiT (single-stream, shared modulation, block-causal joint
//! sequence, unpatched 64-ch latents), the text tower (Qwen3-VL, last-layer pre-norm hidden
//! states) and the VAE (4-channel in/out, 16× spatial, learned-shortcut-free residual up/down)
//! are all new architectures.
//!
//! ## Frozen upstream
//!
//! Every numeric leaf mirrors the pinned sources recorded in [`UPSTREAM_HF_REVISION`],
//! [`UPSTREAM_DIFFUSERS_REVISION`] and [`UPSTREAM_GITHUB_REVISION`] (see `UPSTREAM.md`); the
//! committed parity fixtures under `tests/fixtures/` were produced by `tools/dump_qwen21_*.py`
//! from those exact revisions on miniature configs. The weights are distributed under the
//! **Qwen Research License** (research/evaluation only) — [`UPSTREAM_LICENSE_NOTICE`] is the
//! attribution the licence requires every derived bundle to carry (`NOTICE`).
//!
//! ## What ships in this story
//!
//! Text-to-image at the seven upstream presets (any 32-px-multiple size in range), steps, seed,
//! true-CFG guidance with a negative prompt, seeded determinism, progress + cancellation through
//! the shared `run_flow_sampler` contract, Resident/Sequential residency, and load-time Q4/Q8 of
//! the DiT. The VAE decodes RGBA ([`QwenImage21Vae::decode_rgba`]); the emitted [`Image`] is RGB
//! **composited over white** ([`pipeline::rgba_to_rgb_over_white`]) until gen-core grows an RGBA
//! output surface (sc-24111). The joint-sequence layout ([`transformer::JointLayout`]) already
//! models condition-image blocks so the reference/edit path (a later story) appends segments
//! rather than restructuring attention.
//!
//! [`Image`]: mlx_gen::Image
//! [`QwenImage21Vae::decode_rgba`]: crate::vae::QwenImage21Vae::decode_rgba

pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

/// Hugging Face repository the production snapshot layout is frozen from.
pub const UPSTREAM_HF_REPO: &str = "Qwen/Qwen-Image-2.1";
/// Pinned `Qwen/Qwen-Image-2.1` revision (weights, configs, tokenizer, scheduler config).
pub const UPSTREAM_HF_REVISION: &str = "790c92633540aa0cb11d9abf19eb46d861714758";
/// GitHub repository carrying the upstream presets, defaults and prompt-rewrite tooling.
pub const UPSTREAM_GITHUB_REPO: &str = "QwenLM/Qwen-Image-2.1";
/// Pinned `QwenLM/Qwen-Image-2.1` commit (README presets / 40-step default / RGBA prompt form).
pub const UPSTREAM_GITHUB_REVISION: &str = "fb7ae1d1f9611cd91524d03c53c5246b36ac8577";
/// Pinned `huggingface/diffusers` commit whose `QwenImage21Pipeline`,
/// `QwenImage21Transformer2DModel`, `AutoencoderKLQwenImage21` and
/// `FlowMatchEulerDiscreteScheduler` are the numeric reference for every port here.
pub const UPSTREAM_DIFFUSERS_REVISION: &str = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65";
/// Licence the pinned weights are distributed under.
pub const UPSTREAM_LICENSE: &str = "Qwen Research License Agreement (research/evaluation only)";
/// The attribution notice the Qwen Research License §3(c) requires in every distributed copy.
pub const UPSTREAM_LICENSE_NOTICE: &str = "Qwen is licensed under the Qwen RESEARCH LICENSE \
    AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights \
    Reserved.";

pub use config::{
    SchedulerConfig, SizePreset, TextEncoderConfig, TransformerConfig, VaeConfig, DEFAULT_STEPS,
    DEFAULT_TRUE_CFG, MAX_REFERENCE_IMAGES, PRESETS, SIZE_MULTIPLE, SYSTEM_PROMPT,
    VAE_SCALE_FACTOR,
};
pub use loader::{
    load_scheduler_config, load_text_encoder, load_tokenizer, load_transformer, load_vae,
};
pub use model::{descriptor, load, QwenImage21, MODEL_ID};
pub use pipeline::{
    create_noise, decode_rgb, denoise, encode_prompt, pack_latents, rgba_to_rgb_over_white,
    unpack_latents, DenoiseInputs,
};
pub use text_encoder::{
    prompt_template, system_prefix, system_prompt_drop_count, QwenImage21TextEncoder,
};
pub use transformer::{JointLayout, QwenImage21Transformer, Segment};
pub use vae::QwenImage21Vae;

/// Add the MLX Qwen-Image 2.1 generator to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build the complete explicit MLX Qwen-Image 2.1 provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    #[test]
    fn frozen_revisions_are_full_shas() {
        for sha in [
            super::UPSTREAM_HF_REVISION,
            super::UPSTREAM_GITHUB_REVISION,
            super::UPSTREAM_DIFFUSERS_REVISION,
        ] {
            assert_eq!(sha.len(), 40);
            assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
        }
        assert!(super::UPSTREAM_LICENSE_NOTICE.contains("Qwen RESEARCH LICENSE AGREEMENT"));
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let ids: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, ["qwen_image_2_1"]);
        assert!(registry.descriptor_conformance_errors().is_empty());
    }
}
