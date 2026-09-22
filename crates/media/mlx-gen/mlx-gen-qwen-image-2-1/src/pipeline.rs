//! The Qwen-Image 2.1 text-to-image pipeline body — the port of `QwenImage21Pipeline.__call__`
//! for the text-only path: packed noise → resolution-shifted flow-match Euler denoise (optional
//! true CFG) → unpack → latent denormalisation → RGBA decode → RGB emission.
//!
//! Progress, cancellation, the per-step `eval` boundary and the curated-sampler axis all come from
//! the shared [`run_flow_sampler`] contract; this module only supplies the velocity closure and the
//! Qwen-Image 2.1 latent layout (unpatched `[1, h·w, 64]`, `TimestepConvention::Sigma`).

use mlx_gen::gen_core::sampling::TimestepConvention;
use mlx_gen::image::decoded_to_image;
use mlx_gen::sampler::run_flow_sampler;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{CancelFlag, Error, GenerationRequest, Image, Progress, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use crate::config::VAE_SCALE_FACTOR;
use crate::text_encoder::QwenImage21TextEncoder;
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// `_pack_latents`: 2.1 consumes latents unpatched, so packing is a plain spatial flatten —
/// `[1, C, h, w]` → `[1, h·w, C]`.
pub fn pack_latents(latents: &Array) -> Result<Array> {
    let shape = latents.shape().to_vec();
    if shape.len() != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: pack_latents expects [B, C, h, w], got {shape:?}"
        )));
    }
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    Ok(latents
        .reshape(&[b, c, h * w])?
        .transpose_axes(&[0, 2, 1])?)
}

/// `_unpack_latents`: `[1, h·w, C]` → `[1, C, h, w]` for a `width × height` image.
pub fn unpack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    let (h, w) = latent_grid(width, height);
    let shape = latents.shape().to_vec();
    if shape.len() != 3 || shape[1] != h * w {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: unpack_latents expects [B, {}, C] for {width}x{height}, got {shape:?}",
            h * w
        )));
    }
    let (b, c) = (shape[0], shape[2]);
    Ok(latents.transpose_axes(&[0, 2, 1])?.reshape(&[b, c, h, w])?)
}

/// The latent grid `(h, w)` of a `width × height` request: `2 · (size // 32)` per side.
pub fn latent_grid(width: u32, height: u32) -> (i32, i32) {
    let multiple = VAE_SCALE_FACTOR * 2;
    (
        (2 * (height / multiple)) as i32,
        (2 * (width / multiple)) as i32,
    )
}

/// Seeded packed Gaussian noise `[1, h·w, channels]` (f32) for a `width × height` request.
/// Seeded through MLX's own RNG, so a seed reproduces bit-for-bit on the engine and differs from
/// upstream's torch stream (seed parity across frameworks is not a goal).
pub fn create_noise(seed: u64, width: u32, height: u32, channels: usize) -> Result<Array> {
    let key = random::key(seed)?;
    let (h, w) = latent_grid(width, height);
    let shape = [1, h * w, channels as i32];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Prompt → conditioning `[1, L, hidden]` (f32) through the loaded tower; `drop` is
/// [`crate::system_prompt_drop_count`] for the loaded tokenizer.
pub fn encode_prompt(
    text_encoder: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    prompt: &str,
    drop: usize,
) -> Result<Array> {
    text_encoder.encode_prompt(tokenizer, prompt, drop)
}

/// Everything one denoise run needs.
pub struct DenoiseInputs<'a> {
    pub transformer: &'a QwenImage21Transformer,
    /// Descending schedule, trailing `0.0` (see [`crate::scheduler`]).
    pub sigmas: &'a [f32],
    /// Initial packed latents `[1, h·w, C]`.
    pub latents: Array,
    pub prompt_embeds: &'a Array,
    /// The negative branch; `Some` enables true CFG with `true_cfg_scale`.
    pub negative_embeds: Option<&'a Array>,
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
pub fn denoise(inputs: DenoiseInputs<'_>, on_progress: &mut dyn FnMut(Progress)) -> Result<Array> {
    let (h, w) = latent_grid(inputs.width, inputs.height);
    let (h, w) = (h as usize, w as usize);
    let transformer = inputs.transformer;
    let pos = inputs.prompt_embeds;
    let neg = inputs.negative_embeds;
    let scale = inputs.true_cfg_scale;
    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
        let velocity = transformer.forward(latents, pos, sigma, h, w)?;
        match neg {
            Some(neg) => {
                let uncond = transformer.forward(latents, neg, sigma, h, w)?;
                // `neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)`
                let diff = subtract(&velocity, &uncond)?;
                Ok(add(&uncond, &multiply(&diff, Array::from_f32(scale))?)?)
            }
            None => Ok(velocity),
        }
    };
    run_flow_sampler(
        inputs.sampler,
        TimestepConvention::Sigma,
        inputs.sigmas,
        inputs.latents,
        inputs.seed,
        inputs.cancel,
        on_progress,
        predict,
    )
}

/// Alpha-composite an RGBA NCHW tensor in `[-1, 1]` over white → RGB NCHW in `[-1, 1]` (f32).
/// The chosen RGB emission for the current gen-core image surface: a transparent region reads as
/// white, exactly as a viewer would show the RGBA result on a white page, rather than exposing
/// whatever the decoder painted behind the alpha.
pub fn rgba_to_rgb_over_white(rgba: &Array) -> Result<Array> {
    let shape = rgba.shape().to_vec();
    if shape.len() != 4 || shape[1] != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: rgba_to_rgb_over_white expects [B, 4, H, W], got {shape:?}"
        )));
    }
    let half = Array::from_f32(0.5);
    let x = rgba.as_dtype(Dtype::Float32)?;
    let rgb01 = add(&multiply(x.index((.., 0..3, .., ..)), &half)?, &half)?;
    let a01 = add(&multiply(x.index((.., 3..4, .., ..)), &half)?, &half)?;
    let one = Array::from_f32(1.0);
    let out01 = add(&multiply(&rgb01, &a01)?, &subtract(&one, &a01)?)?;
    Ok(subtract(&multiply(&out01, Array::from_f32(2.0))?, &one)?)
}

/// Default bounded-decode tile geometry (output pixels) when a request asks for tiling without
/// naming one — the sibling image VAEs' 512 px / 64 px.
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;

/// The bounded-decode tiling a request selects: `GenerationMemory::tile_vae_decode` with its
/// optional edge/overlap (output pixels), else `None` (single-pass decode).
pub fn decode_tiling(req: &GenerationRequest) -> Option<TilingConfig> {
    req.memory
        .filter(|memory| memory.tile_vae_decode)
        .map(|memory| {
            TilingConfig::spatial_only(
                memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE) as i32,
                memory.decode_overlap.unwrap_or(DECODE_OVERLAP) as i32,
            )
        })
}

/// Final packed latents → RGB8 [`Image`]: unpack, denormalise (`z·std + mean`), decode RGBA
/// (tiled when `tiling` is given), composite over white, quantise.
pub fn decode_rgb(
    vae: &QwenImage21Vae,
    latents: &Array,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Image> {
    let unpacked = unpack_latents(latents, width, height)?;
    let vae_space = vae.denormalize(&unpacked)?;
    let rgba = match tiling {
        Some(cfg) => vae.decode_rgba_tiled(&vae_space, cfg, cancel)?,
        None => vae.decode_rgba(&vae_space)?,
    };
    let rgb = rgba_to_rgb_over_white(&rgba)?;
    decoded_to_image(&rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        // A 32x64 request is a 2x4 latent grid; two channels.
        let x = Array::from_slice(
            &(0..2 * 2 * 4).map(|v| v as f32).collect::<Vec<_>>(),
            &[1, 2, 2, 4],
        );
        let packed = pack_latents(&x).unwrap();
        assert_eq!(packed.shape(), &[1, 8, 2]);
        // token (row 1, col 2) = index 6, channel 1 → original x[0, 1, 1, 2] = 8 + 4 + 2 = 14.
        assert_eq!(packed.index((0, 6, 1)).item::<f32>(), 14.0);
        let back = unpack_latents(&packed, 64, 32).unwrap();
        assert_eq!(back.shape(), &[1, 2, 2, 4]);
        assert_eq!(back.index((0, 1, 1, 2)).item::<f32>(), 14.0);
        assert!(
            unpack_latents(&packed, 64, 64).is_err(),
            "token count must match the grid"
        );
    }

    #[test]
    fn noise_is_seeded_and_shaped() {
        let a = create_noise(7, 64, 32, 8).unwrap();
        let b = create_noise(7, 64, 32, 8).unwrap();
        let c = create_noise(8, 64, 32, 8).unwrap();
        assert_eq!(a.shape(), &[1, 8, 8]);
        assert!(a.all_close(&b, None, None, None).unwrap().item::<bool>());
        assert!(!a.all_close(&c, None, None, None).unwrap().item::<bool>());
    }

    #[test]
    fn transparent_pixels_composite_to_white() {
        // Two pixels: an opaque red and a fully transparent black.
        let rgba = Array::from_slice(
            &[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
            &[1, 4, 1, 2],
        );
        let rgb = rgba_to_rgb_over_white(&rgba).unwrap();
        let v: Vec<f32> = rgb.as_slice().to_vec();
        assert_eq!(v, vec![1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    }
}
