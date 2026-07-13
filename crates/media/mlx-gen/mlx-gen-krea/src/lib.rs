//! # mlx-gen-krea
//!
//! The **Krea 2** provider crate for [`mlx-gen`](mlx_gen). Krea 2 is Krea AI's first from-scratch
//! foundation image model (released 2026-06-22). Two surfaces share **one architecture**:
//! - **Krea 2 Turbo** (`krea_2_turbo`) — the user-facing text-to-image model (TDM-distilled few-step,
//!   CFG-free, 8 steps, up to 2048²),
//! - **Krea 2 Raw** (`krea_2_raw`) — the undistilled base. Both a **generation model** (full-CFG
//!   text-to-image: real guidance + a user negative prompt, 52 steps, resolution-dynamic mu — epic
//!   9992) AND the **LoRA-training base** (LoRAs train on Raw and apply at Turbo inference, the Lens /
//!   Z-Image precedent — epic 7565 P3). One id, both roles (generator + trainer registries).
//!
//! ## Architecture (verified against the real `krea/Krea-2-Turbo` configs + safetensors index)
//! - **DiT** — `Krea2Transformer2DModel`, a **dense single-stream** rectified-flow / v-param
//!   transformer: text + image tokens concatenated through 28 gated single-stream `transformer_blocks`
//!   (hidden 6144, GQA 48Q/12KV, head_dim 128, SwiGLU 16384, 3-axis RoPE `[32,48,48]`), a
//!   `DoubleSharedModulation` (one shared 6-factor `time_mod_proj` + per-block `scale_shift_table`),
//!   and a `text_fusion` (`TextFusionTransformer`) front-end that aggregates the 12 selected Qwen3-VL
//!   hidden layers (2 layerwise cross-layer-axis blocks → learned `projector` 12→1 → 2 token-axis
//!   refiner blocks).
//! - **Text encoder** — `Qwen3-VL-4B-Instruct` (`Qwen3VLModel`): the pipeline stacks the
//!   `text_encoder_select_layers` `[2,5,…,35]` hidden states and feeds them to the DiT's `text_fusion`.
//! - **VAE** — `AutoencoderKLQwenImage` (z_dim 16, per-channel `latents_mean`/`latents_std` de-norm) —
//!   direct reuse of `mlx-gen-qwen-image`'s `QwenVae`.
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler`, v-param, dynamic exponential time-shift; Turbo
//!   fixes mu 1.15 / 8 steps / CFG 0.
//!
//! ## Slice plan (epic 7565 P1 — complete)
//! The provider scaffold, the `krea_2_turbo` registration, the architecture-validated [`model::load`],
//! and the offline Q4/Q8 converter ([`convert`]) landed in sc-7567; the single-stream DiT in sc-7568
//! ([`transformer`], reusing `mlx-gen-boogu`'s 3-axis-RoPE single-stream + refiner blocks); the
//! Qwen3-VL-4B text encoder + layer-stack in sc-7569 ([`text_encoder`], reusing `mlx-gen-ideogram`'s
//! encoder); the VAE + rectified-flow sampler in sc-7570 ([`vae`], reusing `mlx-gen-qwen-image`'s
//! `QwenVae`; [`schedule`], the exponential-mu flow-match schedule over the core
//! [`mlx_gen::FlowMatchSampler`]); and the end-to-end Turbo t2i [`pipeline`] in sc-7571 — the runnable
//! `krea_2_turbo` engine ([`model::Krea::generate`]). P2+ extends to the worker/web surfaces, the Raw
//! LoRA-training base, and the candle backend.

pub mod config;
pub mod control;
pub mod convert;
pub mod loader;
pub mod model;
pub mod model_control;
pub mod pipeline;
mod quant;
pub mod schedule;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vae;

pub use config::Krea2Config;
pub use control::Krea2ControlBranch;
pub use loader::{load_text_encoder, load_transformer};
pub use model::{descriptor, load, load_raw, raw_descriptor, Krea, KREA_2_RAW_ID, KREA_2_TURBO_ID};
pub use model_control::{KreaTurboControl, KREA_2_TURBO_CONTROL_ID};
pub use pipeline::{KreaHeavy, KreaPipeline, KreaText, TurboOptions};
pub use schedule::{krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
pub use text_encoder::{KreaTeConfig, KreaTextEncoder, KreaTokenizer};
pub use training::{load_trainer, KreaRawTrainer, KREA_2_RAW_TRAINER_ID};
pub use transformer::Krea2Transformer;
pub use vae::{load_vae, QwenVae};

/// Add all MLX Krea generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::TURBO_REGISTRATION)
        .register_generator(model::RAW_REGISTRATION)
        .register_generator(model::EDIT_REGISTRATION)
        .register_generator(model::TURBO_EDIT_REGISTRATION)
        .register_generator(model_control::CONTROL_REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX Krea provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_matches_inventory_compatibility_catalog() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let mut compatibility_generators: Vec<String> = mlx_gen::gen_core::registry::generators()
            .filter_map(|registration| {
                let descriptor = (registration.descriptor)();
                (descriptor.family == "krea_2" && descriptor.backend == "mlx")
                    .then(|| descriptor.id.to_string())
            })
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let mut compatibility_trainers: Vec<String> = mlx_gen::gen_core::registry::trainers()
            .filter_map(|registration| {
                let descriptor = (registration.descriptor)();
                (descriptor.family == "krea_2" && descriptor.backend == "mlx")
                    .then(|| descriptor.id.to_string())
            })
            .collect();
        let mut sorted_explicit_generators = explicit_generators.clone();
        sorted_explicit_generators.sort();
        compatibility_generators.sort();
        let mut sorted_explicit_trainers = explicit_trainers.clone();
        sorted_explicit_trainers.sort();
        compatibility_trainers.sort();

        assert_eq!(sorted_explicit_generators, compatibility_generators);
        assert_eq!(
            explicit_generators,
            [
                "krea_2_turbo",
                "krea_2_raw",
                "krea_2_edit",
                "krea_2_turbo_edit",
                "krea_2_turbo_control",
            ]
        );
        assert_eq!(sorted_explicit_trainers, compatibility_trainers);
        assert_eq!(explicit_trainers, ["krea_2_raw"]);
    }
}
