//! # candle-gen-scail2
//!
//! zai-org **SCAIL-2** — the candle (Windows/CUDA + Linux/NVIDIA) sibling of `mlx-gen-scail2` (epic
//! 6563, the CUDA port of the MLX product epic 5439).
//!
//! SCAIL-2 is an end-to-end controlled **character-animation / motion-transfer** model: a reference
//! image + driving video (+ color-coded segmentation masks) → an animated or identity-replaced video.
//! The backbone is **Wan2.1-14B I2V** (dense), so it reuses the [`candle_gen_wan`] foundation (z16 VAE,
//! UMT5, the flow/UniPC scheduler, the base 3-axis RoPE apply) with three SCAIL-2-specific deltas:
//!
//!   1. **packed-token conditioning** — reference + driving (pose) + 28-channel color-coded masks are
//!      patch-embedded (three Conv3d stems; the mask/pose embeds are *added* to the latent embeds) and
//!      concatenated with the noisy target on the token axis (Bernini-family packed conditioning, not
//!      VACE). Only the target tokens are kept from the prediction.
//!   2. **per-source RoPE shifts** ([`rope::ScailRope`]) — the base 3-axis Wan RoPE with integer
//!      (T,H,W) position shifts per chunk; `replace_flag` flips the reference H-shift (animation vs.
//!      cross-identity replacement), and the pose chunk is spatially frequency-downsampled.
//!   3. **CLIP image cross-attention** — the reference image is encoded by an open-CLIP XLM-RoBERTa
//!      ViT-H/14 visual tower ([`clip::ScailClip`]) and injected via Wan-I2V image cross-attention
//!      (`k_img`/`v_img`).
//!
//! Plain single-scale CFG; f32 DiT compute (bf16 overflows to NaN at high token length); temporal-tiled
//! VAE decode for high-res clips. `backend = "candle"`, `mac_only = false`.
//!
//! ## Status
//! The engine (sc-6836/sc-7078) is GPU-validated: the per-chunk [`rope::ScailRope`], the open-CLIP
//! [`clip::ScailClip`] image encoder, the 28-channel
//! [`preprocess::extract_and_compress_mask_to_latent`] mask build, the PyTorch-faithful [`resize`]
//! kernels, the [`model`] DiT forward, the `generate` denoise pipeline, and the provider
//! registration. Inference adapters — LoRA / LoKr / LoHa, the lightx2v lightning diff-patch, and the
//! Bias-Aware DPO refinement LoRA — fold into the dense DiT via [`adapters::merge_adapters`] (sc-6838).

mod common;

pub mod adapters;
pub mod clip;
pub mod config;
pub mod generate;
pub mod memory_strategy;
pub mod model;
pub mod pipeline;
pub mod preprocess;
pub mod resize;
pub mod rope;

/// The single VAE implementation used by SCAIL-2.
pub type ProviderVae = candle_gen_wan::vae16::WanVae16;
/// SCAIL-2's provider-facing geometry, derived from its concrete VAE assignment.
pub const VAE_TILING: candle_gen::gen_core::tiling::VaeTiling = ProviderVae::VAE_TILING;

/// Resolve SCAIL-2 VAE geometry by registered generator id.
pub fn vae_tiling(provider_id: &str) -> Option<candle_gen::gen_core::tiling::VaeTiling> {
    (provider_id == MODEL_ID).then_some(VAE_TILING)
}

/// Resolve SCAIL-2's provider-owned conservative VAE decode working-set peak.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<candle_gen::VideoDecodeMemoryProfile> {
    vae_tiling(provider_id)?;
    candle_gen::VideoDecodeMemoryProfile::new(
        candle_gen_wan::conservative_video_decode_peak_bytes_for_vae(
            VAE_TILING, width, height, frames,
        )?,
        0,
    )
}

pub use adapters::{has_diff_patch_keys, merge_adapters, MergeReport};
pub use clip::{ClipVisionConfig, ScailClip};
pub use config::Scail2Config;
pub use generate::{generate, CharacterRef, Components, Scail2Job};
pub use model::{Scail2Dit, Scail2Inputs};
pub use pipeline::{
    descriptor, load, snapshot_layout, SnapshotLayout, MODEL_ID, SHARED_TIER_FILES,
};
pub use preprocess::extract_and_compress_mask_to_latent;
pub use resize::{clip_preprocess, downsample_half, interpolate, Interp};
pub use rope::ScailRope;

/// Add the Candle Scail2 provider to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(pipeline::REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
}

/// Build the complete explicit Candle Scail2 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
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
    fn provider_id_is_bound_to_the_causal_wan_z16_geometry() {
        assert_eq!(super::VAE_TILING, super::ProviderVae::VAE_TILING);
        assert_eq!(super::VAE_TILING.full_res_channels, 96);
        assert_eq!(super::preprocess::TEMPORAL_STRIDE, 4);
        assert_eq!(super::generate::DIM_ALIGN, 32);
        let mapped = super::vae_tiling(super::MODEL_ID).unwrap();
        assert!(mapped.causal_temporal);
        assert_eq!(mapped, super::VAE_TILING);
        assert_eq!(
            super::conservative_video_decode_memory_profile(super::MODEL_ID, 64, 64, 9).map(
                |profile| (
                    profile.working_set_bytes(),
                    profile.resident_decoder_bytes_included(),
                )
            ),
            Some((265_830_400, 0))
        );
        assert_eq!(super::vae_tiling("not_scail2"), None);
    }
}
