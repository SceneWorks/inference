//! # mlx-gen-svd
//!
//! Stable Video Diffusion (img2vid-xt) image-to-video provider for mlx-gen (epic 3040, sc-3054).
//! A from-arch port of `stabilityai/stable-video-diffusion-img2vid-xt`:
//! `UNetSpatioTemporalConditionModel` + `AutoencoderKLTemporalDecoder` + the ViT-H
//! `CLIPVisionModelWithProjection` image encoder + the EDM `EulerDiscreteScheduler`, wired through the
//! epic-3018 video runtime (frames → mp4 by the consuming app).
//!
//! Built as slices (mirroring the SDXL port): **S0** config + EDM scheduler (this commit); S1 VAE
//! (2D encoder reuse + temporal decoder); S2 image encoder; S3 UNet; S4 pipeline + provider + e2e
//! parity vs diffusers `StableVideoDiffusionPipeline`. Reuses `mlx-gen-sdxl`'s 2D VAE encoder +
//! CLIP-vision encoder + conv/attn patterns where the spatial parts match.

pub mod config;
pub mod embeddings;
pub mod image_encoder;
pub mod memory_strategy;
pub mod model;
pub mod pipeline;
pub mod preprocess;
pub mod scheduler;
pub mod transformer;
pub mod unet;
pub mod vae;

pub use config::{ImageEncoderConfig, SchedulerConfig, UnetConfig, VaeConfig};
pub use image_encoder::SvdImageEncoder;
pub use model::{descriptor, load, Svd, MODEL_ID, SIZE_ALIGN, VAE_SCALE};
pub use pipeline::{SvdParams, SvdPipeline};
pub use preprocess::resize_with_antialiasing_unit;
pub use scheduler::{euler_step, scale_model_input, v_pred_denoised, EdmSchedule};
pub use transformer::TransformerSpatioTemporal;
pub use unet::SvdUnet;
pub use vae::SvdVae;

/// Why the MLX SVD route has no provider-owned [`mlx_gen::gen_core::tiling::VaeTiling`].
///
/// This is an explicit capability result, not a missing catalog row. The MLX decoder only bounds
/// an invocation by splitting the clip into temporal `decode_chunk_size` windows; unlike the
/// Candle decoder, [`SvdVae::decode`] neither consumes a `VaeTiling` nor reaches the shared spatial
/// tiling planner. Publishing the architectural 256-channel intermediate as a write-bound
/// authority here would therefore advertise a bound that no MLX decode path enforces. Admission
/// must keep `decode_cap_modelled = false` on this lane until a load-bearing MLX planner exists.
pub const VAE_TILING_UNMODELLED_REASON: &str =
    "mlx svd temporally chunks decode calls but has no load-bearing VaeTiling spatial planner";

/// Resolve the explicit unmodelled result for the MLX SVD provider id.
pub fn vae_tiling_unmodelled_reason(provider_id: &str) -> Option<&'static str> {
    (provider_id == MODEL_ID).then_some(VAE_TILING_UNMODELLED_REASON)
}

/// Add the MLX SVD provider to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
}

/// Build the complete explicit MLX SVD provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(explicit, ["svd_xt"]);
        assert_eq!(
            super::vae_tiling_unmodelled_reason(super::MODEL_ID),
            Some(super::VAE_TILING_UNMODELLED_REASON)
        );
        assert_eq!(super::vae_tiling_unmodelled_reason("not_svd"), None);
    }
}
