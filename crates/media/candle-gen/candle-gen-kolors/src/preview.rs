//! Kolors' per-step latent preview seam (epic 16948, sc-16954; the MLX original is epic 16624).
//!
//! Kolors carries **no fit of its own**. It denoises in the SDXL four-channel `AutoencoderKL` latent
//! space, so this module owns only the wiring and defers every coefficient to
//! [`candle_gen_sdxl::preview`]. That deferral is the point: a copy of the constants here could drift
//! from the donor, and `preview::tests::the_fit_is_the_shared_sdxl_one` fails if one is introduced.
//!
//! ## The reuse is grounded in tensor bytes, not in a matching Rust type
//!
//! Kolors and SDXL are one latent space because they ship **one VAE file**. `vae/…fp16.safetensors`,
//! SHA-256 `bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68`, 167,335,342 bytes, is
//! byte-identical across `Kwai-Kolors/Kolors-diffusers`, `stabilityai/stable-diffusion-xl-base-1.0`
//! and every shipped tier (`bf16`/`q8`/`q4`) of `SceneWorks/kolors-mlx` and `SceneWorks/sdxl-base-mlx`
//! — the MLX packer mirrors the VAE dense rather than packing it, so no tier has its own copy. Both
//! `vae/config.json`s declare `latent_channels: 4` and `scaling_factor: 0.13025`, the two numbers that
//! define the space, and `crate::pipeline::sdxl_vae_config` is documented as being SDXL's config.
//!
//! That hash is the one `mlx-gen-sdxl/src/preview.rs` already cites as its Kolors grounding, so the
//! candle claim and the MLX claim rest on the same bytes. Pinned by `tests/preview_real_weights.rs`,
//! which hashes the file Kolors actually loads rather than trusting this comment.
//!
//! Note the one real asymmetry with SDXL, recorded rather than glossed: **Kolors decodes with the
//! snapshot's own VAE** (`crate::pipeline` builds `AutoEncoderKL` straight from the snapshot's `vae/`
//! dir at f32), whereas candle SDXL decodes through the caller-staged `madebyollin/sdxl-vae-fp16-fix`.
//! Kolors is therefore the *closer* match to the epic-16624 fit corpus, not the looser one.
//!
//! ## The latent shape and convention at the emission point — verified, not assumed
//!
//! `crate::common::initial_noise` builds `[1, 4, H/8, W/8]` and every lane keeps that rank-4 spatial
//! layout to the last step, so there is no unpack to write. Batch is always 1: `req.count` is served
//! sequentially through `candle_gen::for_each_image_seed`, and CFG is fused only *inside* the predict
//! closure (`Tensor::cat` on entry, `chunk(2, 0)` plus the guidance combine before returning), so no
//! unconditional half ever reaches a preview.
//!
//! The *convention* differs per lane, and the projector is chosen from what the lane feeds its UNet
//! rather than by assumption:
//!
//! * the **curated** lanes hand the driver a raw k-diffusion VE σ-space latent, so they project
//!   through [`candle_gen_sdxl::preview::ve_hook`], which applies `1/√(σ²+1)` first;
//! * the **native leading-Euler** lanes hold a latent they divide by
//!   `KolorsEulerSampler::scale_in` to build their model input, so they emit that
//!   same quotient — bound to the lane's own coefficient, not to a second opinion about it.
//!
//! See `candle_gen_sdxl::preview` for why the ε/DDPM cohort needs the correction at all when the
//! flow-match families wired earlier in this epic do not.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::PreviewHook;
use candle_gen::Result;

/// The SDXL-family latent channel count the reused fit is defined over, re-exported from the crate
/// that owns the constants so Kolors cannot drift from it by restating a number.
pub use candle_gen_sdxl::preview::PREVIEW_LATENT_CHANNELS;

/// Project a Kolors latent that is **already in the fit's domain** — the native leading-Euler lanes'
/// `latents / scale_in(i)`. Straight through to the shared SDXL seam; Kolors adds nothing.
pub fn project_spatial_latents(latents: &Tensor) -> Result<Image> {
    candle_gen_sdxl::preview::project_spatial_latents(latents)
}

/// The preview hook the **curated** lanes hand `candle_gen::run_curated_sampler` (directly in
/// `crate::pipeline`, or through `candle_gen_sdxl::denoise_curated` for the control / IP providers).
///
/// Build it per image: the driver starts a fresh counter per call, so a batched route that reused one
/// hook across seeds would find every position already emitted from the second image on.
pub(crate) fn ve_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    candle_gen_sdxl::preview::ve_hook(sink)
}

#[cfg(test)]
mod tests {
    use candle_gen::candle_core::{DType, Device};

    use super::*;

    /// Kolors must project **identically** to the SDXL seam, pixel for pixel. A copy of the
    /// coefficients in this crate could not pass this row without being kept in perfect sync, which is
    /// the same thing as not having one.
    #[test]
    fn the_fit_is_the_shared_sdxl_one() {
        let latents = Tensor::from_vec(
            (0..4 * 3 * 5)
                .map(|i| (i % 11) as f32 * 0.1 - 0.5)
                .collect::<Vec<f32>>(),
            (1, 4, 3, 5),
            &Device::Cpu,
        )
        .unwrap();
        let ours = project_spatial_latents(&latents).unwrap();
        let theirs = candle_gen_sdxl::preview::project_spatial_latents(&latents).unwrap();
        assert_eq!(ours.pixels, theirs.pixels);
        assert_eq!((ours.width, ours.height), (5, 3));
    }

    /// This crate defines no coefficients of its own — the channel count is re-exported, never
    /// restated.
    #[test]
    fn the_channel_count_comes_from_the_donor_crate() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 4);
        assert_eq!(
            PREVIEW_LATENT_CHANNELS,
            candle_gen_sdxl::preview::PREVIEW_LATENT_CHANNELS
        );
    }

    #[test]
    fn projection_rejects_a_non_kolors_layout() {
        let bad = Tensor::zeros((1, 16, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let error = project_spatial_latents(&bad).unwrap_err().to_string();
        assert!(error.contains("[1, 4, h, w]"), "{error}");
    }
}
