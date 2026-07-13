//! SD3.5 16-channel VAE wiring (sc-7876, epic 7982).
//!
//! SD3.5 uses a 16-channel diffusers `AutoencoderKL` with the SAME module structure as Z-Image's
//! (`block_out_channels = [128, 256, 512, 512]`, /8 spatial, group-norm 32, diffusers weight
//! naming). Only the scale/shift constants differ:
//!  - `scaling_factor ≈ 1.5305`, `shift_factor ≈ 0.0609`.
//!
//! So C1 **reuses** the candle-transformers `z_image::vae::AutoEncoderKL` — which already takes a
//! parameterized [`VaeConfig`] with `scaling_factor`/`shift_factor` — rather than re-porting the VAE.
//! The encode/decode direction is confirmed against diffusers:
//!  - encode: `latent = (z - shift_factor) * scaling_factor` (`AutoEncoderKL::encode`);
//!  - decode: `z = latent / scaling_factor + shift_factor` (`AutoEncoderKL::decode`).
//!
//! This module just provides the SD3.5 [`Sd3VaeConfig`] preset + a thin loader so the pipeline (C2)
//! and any test can construct the VAE with the right constants. We re-export the reused VAE type so
//! downstream code does not reach into candle-transformers directly.

pub use candle_transformers::models::z_image::vae::{
    AutoEncoderKL, Encoder as VaeEncoder, VaeConfig,
};

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::Image;

/// SD3.5 latent channel count (the DiT `in_channels` and the VAE `latent_channels`).
pub const LATENT_CHANNELS: usize = 16;

/// SD3.5 VAE spatial downscale (image /8 per side; the `[128,256,512,512]` AutoencoderKL has 3
/// downsamplers).
pub const SPATIAL_SCALE: u32 = 8;

/// The VAE **encoder**'s dtype for the img2img / `Reference` path (sc-11784): f32. Candle's
/// `AutoEncoderKL::encode` samples the diagonal-gaussian via the *device* RNG (not launch-portable —
/// breaks the sc-3673 deterministic-seed contract), so the img2img init runs the raw [`VaeEncoder`]
/// and takes the distribution **mean** deterministically in f32, then casts the latent to the compute
/// dtype (bf16). Mirrors `candle-gen-z-image`'s `common::ENC_DTYPE`.
pub const ENC_DTYPE: DType = DType::F32;

/// SD3.5 `scaling_factor` (diffusers `vae/config.json`).
pub const SCALING_FACTOR: f64 = 1.5305;

/// SD3.5 `shift_factor` (diffusers `vae/config.json`).
pub const SHIFT_FACTOR: f64 = 0.0609;

/// Build the SD3.5 [`VaeConfig`] preset — the Z-Image VAE geometry with SD3.5's scale/shift.
pub fn sd3_vae_config() -> VaeConfig {
    VaeConfig {
        scaling_factor: SCALING_FACTOR,
        shift_factor: SHIFT_FACTOR,
        ..VaeConfig::z_image()
    }
}

/// Construct the SD3.5 16-channel `AutoEncoderKL` from a diffusers `vae/` VarBuilder.
pub fn load_vae(vb: VarBuilder) -> Result<AutoEncoderKL> {
    AutoEncoderKL::new(&sd3_vae_config(), vb)
}

/// Construct the raw SD3.5 VAE **encoder** (the down path only) from a diffusers `vae/` VarBuilder
/// (sc-11784). Used by the img2img / `Reference` path to VAE-encode the reference deterministically
/// (the distribution mean, not a sampled draw — see [`ENC_DTYPE`]). The full decode `AutoEncoderKL`
/// carries an encoder too, but it is private and its `encode` samples via the device RNG, so — exactly
/// as `candle-gen-z-image` does — the raw `Encoder` is run here. The `encoder.` prefix matches the
/// diffusers `AutoencoderKL` weight layout.
pub fn load_vae_encoder(vb: VarBuilder) -> Result<VaeEncoder> {
    VaeEncoder::new(&sd3_vae_config(), vb.pp("encoder"))
}

/// Preprocess a source [`Image`] into a `[1, 3, H, W]` f32 tensor in `[-1, 1]` (NCHW) at the render
/// size (sc-11784). LANCZOS-resizes only when the source is off-size (the base img2img resize policy —
/// the worker pre-fits to the render size, so this is usually a no-op), then maps `[0,255] → [-1,1]`.
/// Mirrors `candle-gen-z-image`'s `common::preprocess_image` with `ResizeIfNeeded`.
pub fn preprocess_reference(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> candle_gen::Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(candle_gen::CandleError::Msg(format!(
            "sd3 img2img: image buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    // LANCZOS-resize only when off-size (ResizeIfNeeded); already-sized sources pass through verbatim.
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw)? // HWC f32 [0,255]
    };
    // [0,255] → [-1,1], HWC → CHW.
    let mut data = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            for c in 0..3 {
                data[c * rh * rw + y * rw + x] = resized[(y * rw + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, rh, rw), device)?.to_dtype(ENC_DTYPE)?)
}

/// Deterministic VAE-encode **mean** of a preprocessed `[1, 3, H, W]` f32 image → the init latent
/// `(1, 16, H/8, W/8)` at `out_dtype` (sc-11784): run the raw [`VaeEncoder`], take the distribution
/// mean (the first 16 of the 32 moment channels — NOT a sampled draw, for launch-portable
/// determinism), and map to latent space as `(mean − shift_factor) · scaling_factor` (the diffusers
/// `AutoEncoderKL::encode` direction). Mirrors `candle-gen-z-image`'s `common::encode_mean`.
pub fn encode_mean(
    encoder: &VaeEncoder,
    img: &Tensor,
    out_dtype: DType,
) -> candle_gen::Result<Tensor> {
    let moments = img.apply(encoder)?; // (1, 32, H/8, W/8) — [mean | logvar]
    let mean = moments.chunk(2, 1)?[0].clone(); // (1, 16, H/8, W/8)
    let latents = ((mean - SHIFT_FACTOR)? * SCALING_FACTOR)?;
    Ok(latents.to_dtype(out_dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd3_vae_config_uses_sd35_constants() {
        let c = sd3_vae_config();
        assert_eq!(c.latent_channels, 16);
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert!((c.scaling_factor - 1.5305).abs() < 1e-9);
        assert!((c.shift_factor - 0.0609).abs() < 1e-9);
        // The reused Z-Image VAE preset uses different constants — confirm we actually overrode them.
        assert_ne!(c.scaling_factor, VaeConfig::z_image().scaling_factor);
        assert_ne!(c.shift_factor, VaeConfig::z_image().shift_factor);
    }

    #[test]
    fn latent_geometry_constants() {
        assert_eq!(LATENT_CHANNELS, 16);
        assert_eq!(SPATIAL_SCALE, 8);
    }

    /// The VAE builds + round-trips a latent on CPU (tiny image), exercising the encode/decode
    /// direction with the SD3.5 config. Uses random weights so the structural wiring is what's
    /// validated (not pixel quality).
    #[test]
    fn vae_builds_and_decodes_on_cpu() {
        use candle_gen::candle_core::{DType, Device, Tensor};
        use candle_gen::candle_nn::{VarBuilder, VarMap};

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let vae = load_vae(vb).unwrap();
        // A tiny 16-ch latent /8: a 16x16 image -> 2x2 latent.
        let latent = Tensor::randn(0f32, 1f32, (1, LATENT_CHANNELS, 2, 2), &dev).unwrap();
        let decoded = vae.decode(&latent).unwrap();
        // Decode -> [B, 3, H, W] at the upscaled resolution (2*8 = 16).
        assert_eq!(decoded.dims(), &[1, 3, 16, 16]);
    }
}
