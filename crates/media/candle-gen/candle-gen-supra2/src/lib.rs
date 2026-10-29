//! # candle-gen-supra2
//!
//! Native Supra2-IMG text-to-image on Candle (Windows/Linux CUDA; CPU) — epic sc-24212, story
//! sc-24214. The Candle sibling of `mlx-gen-supra2` (sc-24213), registered under the **same** engine
//! id [`PROVIDER_ID`](gen_core_checkpoint::supra2::PROVIDER_ID) (`supra2_img`) and loading the
//! **same** inputs.
//!
//! `SupraLabs/Supra2-IMG` — a 104.1M-parameter rectified-flow DiT conditioned on Flan-T5-Base and
//! decoded by SD VAE ft-MSE — at 256×256. The backend-neutral half (geometry, revision pins, the
//! install-time `.pt` → bundle converter, component resolution, tokenization, schedule, noise, pixel
//! quantization, memory facts) lives in [`gen_core_checkpoint::supra2`] and is
//! shared with MLX, so both backends integrate the identical seeded `z0` for the same seed.
//!
//! Load contract (identical to MLX): `LoadSpec::weights` is the converted bundle directory
//! ([`convert_pinned_checkpoint`](gen_core_checkpoint::supra2::convert_pinned_checkpoint) output);
//! `LoadSpec::components` carries `text_encoder` (the pinned `google/flan-t5-base` snapshot) and
//! `vae` (the pinned `stabilityai/sd-vae-ft-mse` snapshot). Nothing is downloaded.
//!
//! Reuse: Flan-T5 runs on candle's T5 v1.1 encoder (`candle_transformers::models::t5`) and the VAE on
//! `candle-gen-sdxl`'s ported AutoencoderKL decoder (which reads the checkpoint's legacy attention
//! keys directly). The DiT ([`SupraDit`]) is a port of the frozen upstream module.
//!
//! Parity fixtures are the MLX twin's: the weights-free tiny DiT fixture
//! (`crates/media/mlx-gen/mlx-gen-supra2/tests/fixtures/tiny_dit_parity.safetensors`) and the
//! committed compact real-weight golden derived from the same torch CPU reference dump
//! (`tests/fixtures/supra2_real_golden.safetensors`, see `tests/real_weights.rs`).

pub mod dit;
pub mod memory_strategy;
pub mod model;
pub mod sampling;

pub use dit::SupraDit;
pub use model::{descriptor, encode_tokens, load, Supra2Image, Supra2Output, TextConditioning};

/// Compose the Supra2 generator, its memory strategy and its resident-only contract witness into
/// an explicit registry. The loader is resident-only (no quantization, staging or deferred
/// materialization), so — like the other resident-only Candle routes — it publishes a typed
/// [`ResidentOnlyMemoryContractRegistration`](candle_gen::gen_core::ResidentOnlyMemoryContractRegistration)
/// rather than optimized contract surfaces.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    use candle_gen::gen_core::ResidentOnlyMemoryContractRegistration;
    registry
        .register_generator(model::REGISTRATION)
        .register_memory_strategy(model::MEMORY)
        .register_resident_only_memory_contract(ResidentOnlyMemoryContractRegistration {
            provider_id: gen_core_checkpoint::supra2::PROVIDER_ID,
            contract: memory_strategy::witness,
            surface_specs: memory_strategy::witness_specs,
        })
}

/// Build the explicit Candle Supra2 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let ids: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, ["supra2_img"]);
        assert!(registry.descriptor_conformance_errors().is_empty());
        assert_eq!(registry.memory_strategy_registrations().len(), 1);
        assert_eq!(
            registry
                .resident_only_memory_contract_registrations()
                .map(|r| r.provider_id)
                .collect::<Vec<_>>(),
            ["supra2_img"]
        );
    }
}
