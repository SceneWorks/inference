//! Native Supra2-IMG text-to-image on MLX (epic sc-24212, story sc-24213).
//!
//! `SupraLabs/Supra2-IMG` — a 104.1M-parameter rectified-flow DiT conditioned on Flan-T5-Base and
//! decoded by SD VAE ft-MSE — at 256×256. The backend-neutral half (geometry, revision pins, the
//! install-time `.pt` → bundle converter, tokenization, schedule, noise, quantization) lives in
//! [`gen_core::supra2`](mlx_gen::gen_core::supra2) and is shared with the Candle port.
//!
//! Load contract: `LoadSpec::weights` is the converted bundle directory
//! ([`convert_pinned_checkpoint`](mlx_gen::gen_core::supra2::convert_pinned_checkpoint) output);
//! `LoadSpec::components` carries `text_encoder` (the pinned `google/flan-t5-base` snapshot) and
//! `vae` (the pinned `stabilityai/sd-vae-ft-mse` snapshot). Nothing is downloaded.

pub mod dit;
pub mod memory_strategy;
pub mod model;
pub mod sampling;

pub use dit::SupraDit;
pub use model::{descriptor, load, Supra2Image, Supra2Output, Supra2Sources, TextConditioning};

/// Compose the Supra2 generator and its memory surfaces into an explicit registry.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    use mlx_gen::gen_core::MemoryContractFixtureRegistration;
    registry
        .register_generator(model::REGISTRATION)
        .register_memory_strategy(model::MEMORY)
        .register_memory_contract_fixture(MemoryContractFixtureRegistration {
            provider_id: mlx_gen::gen_core::supra2::PROVIDER_ID,
            contract: memory_strategy::fixture,
            surface_specs: memory_strategy::surface_specs,
        })
}
