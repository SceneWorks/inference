//! The Qwen-Image 2.1 text-to-image pipeline body — the port of `QwenImage21Pipeline.__call__`
//! for the text-only path: packed noise → resolution-shifted flow-match Euler denoise (optional
//! true CFG) → unpack → latent denormalisation → RGBA decode → RGB emission.
//!
//! Progress, cancellation, the per-step boundary and the curated-sampler axis all come from the
//! shared [`run_flow_sampler`] contract; this module only supplies the velocity closure and the
//! Qwen-Image 2.1 latent layout (unpatched `[1, h·w, 64]`, `TimestepConvention::Sigma`).

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{CancelFlag, Image, Progress};
use candle_gen::run_flow_sampler;
use candle_gen::{CandleError as Error, Result};

use crate::config::VAE_SCALE_FACTOR;
use crate::text_encoder::QwenImage21TextEncoder;
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// `_pack_latents`: 2.1 consumes latents unpatched, so packing is a plain spatial flatten —
/// `[B, C, h, w]` → `[B, h·w, C]`.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let dims = latents.dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: pack_latents expects [B, C, h, w], got {dims:?}"
        )));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    Ok(latents.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?)
}

/// `_unpack_latents`: `[B, h·w, C]` → `[B, C, h, w]` for a `width × height` image.
pub fn unpack_latents(latents: &Tensor, width: u32, height: u32) -> Result<Tensor> {
    let (h, w) = latent_grid(width, height);
    let dims = latents.dims().to_vec();
    if dims.len() != 3 || dims[1] != h * w {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: unpack_latents expects [B, {}, C] for {width}x{height}, got {dims:?}",
            h * w
        )));
    }
    let (b, c) = (dims[0], dims[2]);
    Ok(latents.transpose(1, 2)?.reshape((b, c, h, w))?.contiguous()?)
}

/// The latent grid `(h, w)` of a `width × height` request: `2 · (size // 32)` per side.
pub fn latent_grid(width: u32, height: u32) -> (usize, usize) {
    let multiple = VAE_SCALE_FACTOR * 2;
    (
        2 * (height / multiple) as usize,
        2 * (width / multiple) as usize,
    )
}

/// Seeded packed Gaussian noise `[1, h·w, channels]` (f32) for a `width × height` request.
/// Drawn on the CPU through the shared launch-portable seeded normal draw, so a seed reproduces
/// bit-for-bit on every candle backend (and differs from upstream's torch stream — seed parity
/// across frameworks is not a goal, exactly as on the MLX twin).
pub fn create_noise(
    seed: u64,
    width: u32,
    height: u32,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    let (h, w) = latent_grid(width, height);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let values = candle_gen::seeded_normal_vec(&mut rng, h * w * channels);
    Ok(Tensor::from_vec(values, (1, h * w, channels), &Device::Cpu)?.to_device(device)?)
}

/// Prompt → conditioning `[1, L, hidden]` (f32) through the loaded tower; `drop` is
/// [`crate::system_prompt_drop_count`] for the loaded tokenizer.
pub fn encode_prompt(
    text_encoder: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    prompt: &str,
    drop: usize,
) -> Result<Tensor> {
    text_encoder.encode_prompt(tokenizer, prompt, drop)
}

/// Everything one denoise run needs.
pub struct DenoiseInputs<'a> {
    pub transformer: &'a QwenImage21Transformer,
    /// Descending schedule, trailing `0.0` (see [`crate::scheduler`]).
    pub sigmas: &'a [f32],
    /// Initial packed latents `[1, h·w, C]`.
    pub latents: Tensor,
    pub prompt_embeds: &'a Tensor,
    /// The negative branch; `Some` enables true CFG with `true_cfg_scale`.
    pub negative_embeds: Option<&'a Tensor>,
    pub true_cfg_scale: f32,
    pub width: u32,
    pub height: u32,
    /// Curated solver name (`None` = Euler, the native path).
    pub sampler: Option<&'a str>,
    pub seed: u64,
    pub cancel: &'a CancelFlag,
}

/// The denoise loop: every step runs the transformer over the full joint sequence and, with a
/// negative branch, applies `neg + s·(pos − neg)`. Returns the final packed latents (f32).
pub fn denoise(inputs: DenoiseInputs<'_>, on_progress: &mut dyn FnMut(Progress)) -> Result<Tensor> {
    let (h, w) = latent_grid(inputs.width, inputs.height);
    let transformer = inputs.transformer;
    let pos = inputs.prompt_embeds;
    let neg = inputs.negative_embeds;
    let scale = inputs.true_cfg_scale as f64;
    let predict = |latents: &Tensor, sigma: f32| -> Result<Tensor> {
        let velocity = transformer.forward(latents, pos, sigma, h, w)?;
        match neg {
            Some(neg) => {
                let uncond = transformer.forward(latents, neg, sigma, h, w)?;
                // `neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)`
                let diff = ((&velocity - &uncond)? * scale)?;
                Ok((&uncond + diff)?)
            }
            None => Ok(velocity),
        }
    };
    Ok(run_flow_sampler(
        inputs.sampler,
        TimestepConvention::Sigma,
        inputs.sigmas,
        inputs.latents,
        inputs.seed,
        inputs.cancel,
        on_progress,
        None,
        predict,
    )?)
}

/// Alpha-composite an RGBA NCHW tensor in `[-1, 1]` over white → RGB NCHW in `[-1, 1]` (f32).
/// The chosen RGB emission for the current gen-core image surface: a transparent region reads as
/// white, exactly as a viewer would show the RGBA result on a white page, rather than exposing
/// whatever the decoder painted behind the alpha.
pub fn rgba_to_rgb_over_white(rgba: &Tensor) -> Result<Tensor> {
    let dims = rgba.dims().to_vec();
    if dims.len() != 4 || dims[1] != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: rgba_to_rgb_over_white expects [B, 4, H, W], got {dims:?}"
        )));
    }
    let x = rgba.to_dtype(DType::F32)?;
    let rgb01 = ((x.i((.., 0..3, .., ..))? * 0.5)? + 0.5)?;
    let a01 = ((x.i((.., 3..4, .., ..))? * 0.5)? + 0.5)?;
    let out01 = (rgb01.broadcast_mul(&a01)? + (1.0 - &a01)?)?;
    Ok(((out01 * 2.0)? - 1.0)?)
}

/// Final packed latents → RGB8 [`Image`]: unpack, denormalise (`z·std + mean`), decode RGBA,
/// composite over white, quantise.
pub fn decode_rgb(
    vae: &QwenImage21Vae,
    latents: &Tensor,
    width: u32,
    height: u32,
) -> Result<Image> {
    let unpacked = unpack_latents(latents, width, height)?;
    let rgba = vae.decode_rgba(&vae.denormalize(&unpacked)?)?;
    decoded_to_image(&rgba_to_rgb_over_white(&rgba)?)
}

/// RGB NCHW in `[-1, 1]` → the gen-core RGB8 [`Image`], with diffusers-compatible
/// nearest-even rounding ([`candle_gen::round_rgb8`]).
pub fn decoded_to_image(rgb: &Tensor) -> Result<Image> {
    let dims = rgb.dims().to_vec();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: decoded_to_image expects [1, 3, H, W], got {dims:?}"
        )));
    }
    let (height, width) = (dims[2], dims[3]);
    let scaled = ((rgb.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let pixels = candle_gen::round_rgb8(&scaled)?
        .i(0)?
        .to_device(&Device::Cpu)?
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<u8>()?;
    Ok(Image {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let dev = Device::Cpu;
        // A 32x64 request is a 2x4 latent grid; two channels.
        let x = Tensor::from_vec(
            (0..2 * 2 * 4).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 2, 2, 4),
            &dev,
        )
        .unwrap();
        let packed = pack_latents(&x).unwrap();
        assert_eq!(packed.dims(), &[1, 8, 2]);
        // token (row 1, col 2) = index 6, channel 1 → original x[0, 1, 1, 2] = 8 + 4 + 2 = 14.
        assert_eq!(
            packed.i((0, 6, 1)).unwrap().to_scalar::<f32>().unwrap(),
            14.0
        );
        let back = unpack_latents(&packed, 64, 32).unwrap();
        assert_eq!(back.dims(), &[1, 2, 2, 4]);
        assert_eq!(
            back.i((0, 1, 1, 2)).unwrap().to_scalar::<f32>().unwrap(),
            14.0
        );
        assert!(
            unpack_latents(&packed, 64, 64).is_err(),
            "token count must match the grid"
        );
    }

    #[test]
    fn noise_is_seeded_and_shaped() {
        let dev = Device::Cpu;
        let a = create_noise(7, 64, 32, 8, &dev).unwrap();
        let b = create_noise(7, 64, 32, 8, &dev).unwrap();
        let c = create_noise(8, 64, 32, 8, &dev).unwrap();
        assert_eq!(a.dims(), &[1, 8, 8]);
        let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(host(&a), host(&b));
        assert_ne!(host(&a), host(&c));
    }

    #[test]
    fn transparent_pixels_composite_to_white() {
        let dev = Device::Cpu;
        // Two pixels: an opaque red and a fully transparent black.
        let rgba = Tensor::from_vec(
            vec![1.0f32, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
            (1, 4, 1, 2),
            &dev,
        )
        .unwrap();
        let rgb = rgba_to_rgb_over_white(&rgba).unwrap();
        let v = rgb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    }
}
