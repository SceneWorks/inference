//! **MiniMax-H3-Image** (`MiniMaxAI/MiniMax-H3-Image`) on MLX: the `minimax_h3_image` generator —
//! `text2image` and `edit` with 1-16 reference images — built from this crate's H3 layers.
//!
//! The image model is the MiniMax-H3 transformer without its audio stream, distilled to a fixed
//! six-step sampler with one output head per step, over one packed `[text | image]` sequence whose
//! image is a single latent frame. It shares H3's text encoder (byte-identical Qwen3-VL-32B,
//! layer-50 tap), H3's DiT layers, H3's velocity-Euler scheduler, and SW-7089's 2-D image VAE.
//! Host-side request math (sizes, canvas, sigmas, refusals, prompt truncation) is
//! [`mlx_gen::gen_core::h3_image`].
//!
//! | | |
//! |---|---|
//! | [`layout`] | the `text2image` / `edit` packed sequences: positions, tags |
//! | [`dit`] | [`dit::MiniMaxH3ImageDit`] (644 tensors) and the per-step [`dit::ImageDenoiser`] |
//! | [`model`] | the generator: descriptor, load, validate, staged generate |
//! | [`tier`] | q8 / q4 DiT tier resolution; the text encoder is H3's own tier artifact |
//! | [`memory`] | the shared-ladder contract: staged residency + bounded decode |

pub mod dit;
pub mod layout;
pub mod memory;
pub mod model;
pub mod tier;

pub use dit::{
    image_row_classes, velocity_euler_delta, ImageDenoiser, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT,
    PUBLISHED_IMAGE_DIT_TENSORS,
};
pub use layout::ImageLayout;
pub use model::{
    reference_condition_rows, reference_noise, reference_vae_pixels, resize_reference,
    MiniMaxH3Image, IMAGE_MODEL_ID,
};

/// Add the MLX MiniMax-H3-Image generator and its three memory registrations to an explicit media
/// registry builder. Separate from [`crate::register_providers`] so H3 video's catalog surface is
/// unchanged; the media catalog composes both.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_memory_strategy(memory::IMAGE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory::IMAGE_MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory::IMAGE_MEMORY_BEHAVIOR)
}

#[cfg(test)]
mod registry_tests {
    /// One generator at its own id, with its memory seams wired (the builder refuses a memory
    /// registration with no generator, and a fixture/behavior with no memory registration).
    #[test]
    fn the_image_registry_builds_with_one_generator() {
        let registry = super::register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new())
            .build()
            .expect("registry");
        let ids: Vec<&str> = registry.generators().map(|g| (g.descriptor)().id).collect();
        assert_eq!(ids, vec![super::IMAGE_MODEL_ID]);
        assert_eq!(registry.memory_strategy_registrations().len(), 1);
    }
}
