//! # candle-gen-flux3
//!
//! The **FLUX 3 Image Klein** provider crate for [`candle-gen`](candle_gen) (epic sc-26233; story
//! sc-26235 is the native Candle/CUDA generation of the distilled text-to-image route
//! `flux3_klein`) — the Candle (Windows/Linux CUDA) sibling of `mlx-gen-flux3` (sc-26234) and a
//! native port of the frozen upstream `black-forest-labs/flux-3-image-klein` inference code: the
//! Qwen3-VL-4B text encoder (eight stacked hidden states, a pad-to-80 window), the
//! 16×-downsampling autoencoder (8× VAE + 2×2 patchify, BatchNorm-stat latents), the per-mode +
//! shared single-stream DiT with 4-axis RoPE and target-time (flow-map) conditioning, and the
//! flow-map Euler sampler. No Python runtime path.
//!
//! The backend-neutral half (identity, resources, `config.json`, text window, schedule, id and
//! RoPE tables, request params, typed refusals) is [`gen_core::flux3`], shared with the MLX twin
//! and the SceneWorks worker; the two providers accept the same generation request contract. The
//! tensor work is here.
//!
//! ## Resources
//!
//! `LoadSpec::weights` is the package directory (`config.json` + `model.safetensors`);
//! `components["text_encoder"]` and `components["autoencoder"]` optionally override the bundled
//! `text_encoder/` snapshot and `ae.safetensors`. See `UPSTREAM.md` for the pinned revisions and
//! the deliberate differences from the MLX twin. The parity tests read the **same** committed
//! fixtures the MLX twin reads (`crates/media/mlx-gen/mlx-gen-flux3/tests/fixtures/`, produced by
//! `crates/media/mlx-gen/tools/dump_flux3_*.py` from the frozen source), so both backends are
//! held to one numeric reference.
//!
//! ## Deliberate differences from the MLX twin
//!
//! * `backend = "candle"`, `mac_only = false`; the autoencoder runs NCHW (candle's conv layout).
//! * Noise is drawn from the shared launch-portable CPU `StdRng` (`candle_gen::seed`), rounded
//!   through bf16 as upstream's `torch.randn(...).to(bfloat16).float()`, not MLX's RNG: a seed
//!   reproduces within a backend but not across the two (upstream's `torch.randn` matches
//!   neither; parity is measured with injected noise on both).
//! * On the Candle CPU backend (no half-precision GEMM) the bf16 policy runs each GEMM in f32
//!   over bf16 operands and rounds its result to bf16, and the Qwen3-VL tower runs in f32
//!   (`candle_llm`'s CPU rule); on CUDA both are the real bf16 kernels.
//!
//! [`gen_core::flux3`]: candle_gen::gen_core::flux3

pub mod adapters;
pub mod autoencoder;
pub mod convert;
pub mod dit;
pub mod memory_strategy;
pub mod model;
pub mod nn;
pub mod pipeline;
pub mod pup;
pub mod references;
pub mod text_encoder;
pub mod training;

pub use autoencoder::Flux3AutoEncoder;
pub use candle_gen::gen_core::flux3::{
    AUTOENCODER_COMPONENT, DISTILLED_MODEL_ID, DISTILLED_WEIGHTS_REPO, DISTILLED_WEIGHTS_REVISION,
    FAMILY, TEXT_ENCODER_COMPONENT, TEXT_ENCODER_UPSTREAM_REPO, TEXT_ENCODER_UPSTREAM_REVISION,
    UPSTREAM_CODE_REVISION,
};
pub use dit::{Flux3Dit, SampleTokens};
pub use model::{
    compute_dtype, descriptor, descriptor_base, load, load_autoencoder, load_base, load_dit,
    Flux3Klein, MODEL_ID,
};
pub use pipeline::{
    decode_tokens, denoise_flow_map, denoise_one, float_image_to_rgb8, generate_one, image_ids,
    noise,
};
pub use text_encoder::{Flux3TextEncoder, TextConditioning};

/// Add the Candle distilled FLUX 3 generator to an explicit media registry builder, with its
/// memory-strategy contract, weights-free contract fixture and behavior seam.
///
/// Deliberately registers **no** `ActivationMemoryRegistration`: that carrier publishes only real
/// on-device measurements, and the epic's terminal campaign owns the anchors.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    register_memory_contract_surfaces(
        registry
            .register_generator(model::REGISTRATION)
            .register_generator(model::BASE_REGISTRATION),
    )
}

/// The shared-ladder registrations of the route: strategy, weights-free contract fixture and
/// behavior. Split out the way the sibling Candle providers do, so a catalog can append the
/// contract surface to a registry that already carries the generator.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::BASE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::BASE_MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::BASE_MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR_REGISTRATION)
}

/// Build the complete explicit Candle FLUX 3 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    /// This crate's `UPSTREAM.md` and the MLX twin's (the shared coverage table) both record the
    /// pinned revisions this crate is accountable to.
    #[test]
    fn upstream_md_records_the_pinned_revisions() {
        let ours = include_str!("../UPSTREAM.md");
        let twin = include_str!("../../../mlx-gen/mlx-gen-flux3/UPSTREAM.md");
        for sha in [
            super::UPSTREAM_CODE_REVISION,
            super::DISTILLED_WEIGHTS_REVISION,
            super::TEXT_ENCODER_UPSTREAM_REVISION,
        ] {
            assert!(ours.contains(sha), "UPSTREAM.md must record {sha}");
            assert!(
                twin.contains(sha),
                "the MLX twin's UPSTREAM.md must record {sha}"
            );
        }
        assert!(ours.contains("candle-gen-flux3"));
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let ids: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, ["flux3_klein", "flux3_klein_base"]);
        assert!(registry.descriptor_conformance_errors().is_empty());
        assert_eq!(registry.memory_strategy_registrations().len(), 2);
        assert_eq!(registry.memory_contract_fixture_registrations().len(), 2);
    }
}
