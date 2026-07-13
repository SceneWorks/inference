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
pub mod model;
pub mod pipeline;
pub mod preprocess;
pub mod scheduler;
pub mod transformer;
pub mod unet;
pub mod vae;

pub use config::{ImageEncoderConfig, SchedulerConfig, UnetConfig, VaeConfig};
pub use image_encoder::SvdImageEncoder;
pub use model::{descriptor, load, Svd, MODEL_ID};
pub use pipeline::{SvdParams, SvdPipeline};
pub use preprocess::resize_with_antialiasing_unit;
pub use scheduler::{euler_step, scale_model_input, v_pred_denoised, EdmSchedule};
pub use transformer::TransformerSpatioTemporal;
pub use unet::SvdUnet;
pub use vae::SvdVae;

/// Add the MLX SVD provider to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
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
    }
}
