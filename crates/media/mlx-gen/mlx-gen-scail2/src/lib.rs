//! zai-org **SCAIL-2** — native MLX provider (epic 5439, sc-5442).
//!
//! SCAIL-2 is an end-to-end controlled **character-animation / motion-transfer** model: a reference
//! image + driving video (+ color-coded segmentation masks) → an animated or identity-replaced video.
//! The backbone is **Wan2.1-14B I2V** (dense), so it reuses the [`mlx_gen_wan`] foundation (DiT blocks,
//! z16 VAE, UMT5, 3-axis RoPE, UniPC/flow schedulers) with three SCAIL-2-specific deltas:
//!
//!   1. **packed-token conditioning** — reference + driving (pose) + 28-channel color-coded masks are
//!      patch-embedded (three Conv3d stems; the mask/pose embeds are *added* to the latent embeds) and
//!      concatenated with the noisy target on the token axis (Bernini-family packed conditioning, not
//!      VACE). Only the target tokens are kept from the prediction.
//!   2. **per-source RoPE shifts** — the base 3-axis Wan RoPE with integer (T,H,W) position shifts per
//!      chunk; `replace_flag` flips the reference H-shift (animation vs. cross-identity replacement),
//!      and the pose chunk is spatially frequency-downsampled.
//!   3. **CLIP image cross-attention** — the reference image is encoded by an open-CLIP XLM-RoBERTa
//!      ViT-H/14 visual tower and injected via Wan-I2V image cross-attention (`k_img`/`v_img`).
//!
//! Weights: the turnkey `SceneWorks/scail2-mlx` snapshot (converted bf16 DiT + stock Wan2.1 VAE / UMT5
//! / CLIP). Plain single-scale CFG; macOS-only.
//!
//! Status (sc-5443): the registration + capability surface, the [`model::Scail2Dit`] DiT forward, the
//! per-chunk [`rope::ScailRope`], the CLIP/VAE/mask preprocessing, and the live [`generate()`] denoise
//! loop all land here (each parity-gated against upstream on tiny seeded fixtures). Real-weight 40-layer
//! + end-to-end parity is sc-5446; Q4/Q8 load-time quant is sc-5445.

pub mod clip;
pub mod config;
pub mod convert;
pub mod generate;
pub mod lora;
pub mod model;
pub mod pipeline;
pub mod preprocess;
pub mod resize;
pub mod rope;

/// The single VAE implementation used by SCAIL-2.
pub type ProviderVae = mlx_gen_wan::WanVae;
/// SCAIL-2's provider-facing geometry, derived from its concrete VAE assignment.
pub const VAE_TILING: mlx_gen::tiling::VaeTiling = ProviderVae::VAE_TILING;

/// Resolve SCAIL-2 VAE geometry by registered generator id.
pub fn vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    (provider_id == pipeline::MODEL_ID).then_some(VAE_TILING)
}

/// Resolve SCAIL-2's provider-owned conservative VAE decode working-set peak.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<mlx_gen::VideoDecodeMemoryProfile> {
    vae_tiling(provider_id)?;
    mlx_gen_wan::conservative_video_decode_memory_profile_for_vae(VAE_TILING, width, height, frames)
}

pub use clip::{ClipVisionConfig, ScailClip};
pub use config::Scail2Config;
pub use convert::{quantize_scail2_dit, quantize_scail2_transformer};
pub use generate::{generate, CharacterRef, Scail2Job};
pub use lora::{has_diff_patch_keys, merge_diff_patch_adapters, DiffPatchReport};
pub use model::{Scail2Dit, Scail2Inputs};
pub use preprocess::extract_and_compress_mask_to_latent;
pub use resize::{clip_preprocess, downsample_half, interpolate, Interp};
pub use rope::ScailRope;

/// Add the MLX Scail2 provider to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(pipeline::REGISTRATION)
}

/// Build the complete explicit MLX Scail2 provider catalog.
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
        assert_eq!(explicit, ["scail2_14b"]);
    }

    #[test]
    fn provider_id_is_bound_to_the_wan_z16_geometry() {
        assert_eq!(super::VAE_TILING, super::ProviderVae::VAE_TILING);
        assert_eq!(super::VAE_TILING, mlx_gen::tiling::VaeTiling::WAN);
        assert_eq!(super::preprocess::TEMPORAL_STRIDE, 4);
        assert_eq!(super::generate::DIM_ALIGN, 32);
        assert_eq!(
            super::vae_tiling(super::pipeline::MODEL_ID),
            Some(super::VAE_TILING)
        );
        assert_eq!(
            super::conservative_video_decode_memory_profile(super::pipeline::MODEL_ID, 64, 64, 9)
                .map(|profile| (
                    profile.working_set_bytes(),
                    profile.resident_decoder_bytes_included()
                )),
            Some((322_633_728, 0))
        );
        assert_eq!(super::vae_tiling("not_scail2"), None);
    }
}
