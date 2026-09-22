//! The Qwen-Image 2.1 text-to-image pipeline body — the port of `QwenImage21Pipeline.__call__`
//! for the text-only path: packed noise → resolution-shifted flow-match Euler denoise (optional
//! true CFG) → unpack → latent denormalisation → RGBA decode → RGB emission.
//!
//! Progress, cancellation, the per-step `eval` boundary and the curated-sampler axis all come from
//! the shared [`run_flow_sampler`] contract; this module only supplies the velocity closure and the
//! Qwen-Image 2.1 latent layout (unpatched `[1, h·w, 64]`, `TimestepConvention::Sigma`).

use mlx_gen::gen_core::sampling::TimestepConvention;
use mlx_gen::image::{decoded_to_image, decoded_to_rgba_image};
use mlx_gen::sampler::run_flow_sampler;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{CancelFlag, Error, GenerationRequest, Image, Progress, Result, RgbaImage};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use crate::config::VAE_SCALE_FACTOR;
use crate::reference::PreparedReference;
use crate::text_encoder::QwenImage21TextEncoder;
use crate::transformer::{JointLayout, QwenImage21Transformer, Segment};
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

/// VAE-encode every prepared reference, in order, into the denoiser-space packed latents the
/// joint sequence prepends to the noise — `prepare_latents`' `images is not None` branch:
/// `_encode_vae_image` with `sample_mode="argmax"` (the posterior **mode**, not a sample), the
/// same `(z − mean) / std` normalisation the target latents use, then the unpatched
/// `[1, (h/16)·(w/16), 64]` flatten.
pub fn encode_references(
    vae: &QwenImage21Vae,
    references: &[PreparedReference],
) -> Result<Vec<Array>> {
    references
        .iter()
        .map(|reference| {
            let mode = vae.encode_mode(&reference.vae_input)?;
            pack_latents(&vae.normalize(&mode)?)
        })
        .collect()
}

/// The joint text/image layout for a conditioned request — the Rust form of upstream's
/// `image_pad_mask` expansion (`repeat_interleave(img_mask, where(img_mask, 4, 1))`) plus the
/// appended target block.
///
/// `image_pad_mask` marks the vision slots in the **VLM** sequence; each slot stands for a 2×2
/// group of latent tokens, so a run of `n` slots belonging to reference `k` becomes one
/// [`Segment::Image`] of that reference's `(h/16, w/16)` grid. Block boundaries come from the
/// per-reference slot counts, not from runs of `true`: two adjacent references with no text
/// between them stay two blocks, exactly as `build_token_metadata` insists.
pub fn joint_layout(
    image_pad_mask: &[bool],
    references: &[PreparedReference],
    width: u32,
    height: u32,
) -> Result<JointLayout> {
    let slots = image_pad_mask.iter().filter(|m| **m).count();
    let expected: usize = references.iter().map(PreparedReference::vision_slots).sum();
    if slots != expected {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: the prompt reserved {slots} vision slots but the {} reference images \
             need {expected}",
            references.len()
        )));
    }
    let mut segments: Vec<Segment> = Vec::with_capacity(2 * references.len() + 2);
    let mut text_run = 0usize;
    let mut cursor = 0usize;
    let mut next_reference = 0usize;
    while cursor < image_pad_mask.len() {
        if !image_pad_mask[cursor] {
            text_run += 1;
            cursor += 1;
            continue;
        }
        if text_run > 0 {
            segments.push(Segment::Text { len: text_run });
            text_run = 0;
        }
        let reference = references.get(next_reference).ok_or_else(|| {
            Error::Msg(
                "qwen_image_2_1: the prompt carries more vision-slot runs than reference images"
                    .into(),
            )
        })?;
        let take = reference.vision_slots();
        for offset in 0..take {
            if !image_pad_mask
                .get(cursor + offset)
                .copied()
                .unwrap_or(false)
            {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1: reference image {next_reference} needs {take} contiguous \
                     vision slots but the prompt breaks the run after {offset}"
                )));
            }
        }
        let (h, w) = reference.latent_grid();
        segments.push(Segment::Image {
            height: h,
            width: w,
        });
        cursor += take;
        next_reference += 1;
    }
    if next_reference != references.len() {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: only {next_reference} of {} reference images found a vision-slot run \
             in the prompt",
            references.len()
        )));
    }
    if text_run > 0 {
        segments.push(Segment::Text { len: text_run });
    }
    let (h, w) = latent_grid(width, height);
    segments.push(Segment::Image {
        height: h as usize,
        width: w as usize,
    });
    Ok(JointLayout { segments })
}

/// The text rows of a conditioning tensor — everything the DiT does **not** overwrite with a
/// condition latent. `txt_in` is row-wise, so selecting the rows before the projection is exactly
/// upstream's "project everything, then scatter the latents over the image rows".
pub fn text_rows(hidden: &Array, image_pad_mask: &[bool]) -> Result<Array> {
    if image_pad_mask.iter().all(|m| !*m) {
        return Ok(hidden.clone());
    }
    let keep: Vec<i32> = image_pad_mask
        .iter()
        .enumerate()
        .filter(|(_, m)| !**m)
        .map(|(i, _)| i as i32)
        .collect();
    let index = Array::from_slice(&keep, &[keep.len() as i32]);
    Ok(hidden.take_axis(&index, 1)?)
}

/// The reference conditioning one denoise run carries: the packed condition latents (in order)
/// and the joint layout of each branch.
pub struct ReferenceConditioning<'a> {
    /// Packed `[1, tokens, 64]` latents, condition images first, in request order.
    pub latents: &'a [Array],
    /// Layout of the positive branch.
    pub layout: &'a JointLayout,
    /// Layout of the negative branch — a different prompt is a different text length, so true CFG
    /// needs its own.
    pub negative_layout: Option<&'a JointLayout>,
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
    /// Ordered reference conditioning, or `None` for the text-to-image route.
    pub references: Option<ReferenceConditioning<'a>>,
}

/// The denoise loop: every step runs the transformer over the full joint sequence and, with a
/// negative branch, applies `neg + s·(pos − neg)`. Returns the final packed latents (f32).
///
/// With [`DenoiseInputs::references`] set, the condition latents are prepended to the target
/// latents each step (`latent_model_input = cat([*reference_latents, latents], dim=1)`) and the
/// transformer attends over the full interleaved layout; the velocity it returns is already
/// sliced to the target block.
pub fn denoise(inputs: DenoiseInputs<'_>, on_progress: &mut dyn FnMut(Progress)) -> Result<Array> {
    let (h, w) = latent_grid(inputs.width, inputs.height);
    let (h, w) = (h as usize, w as usize);
    let transformer = inputs.transformer;
    let pos = inputs.prompt_embeds;
    let neg = inputs.negative_embeds;
    let scale = inputs.true_cfg_scale;
    let references = inputs.references.as_ref();
    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
        let branch = |text: &Array, layout: Option<&JointLayout>| -> Result<Array> {
            match (references, layout) {
                (Some(conditioning), Some(layout)) => {
                    let mut images: Vec<&Array> = conditioning.latents.iter().collect();
                    images.push(latents);
                    transformer.forward_joint(text, &images, sigma, layout)
                }
                _ => transformer.forward(latents, text, sigma, h, w),
            }
        };
        let velocity = branch(pos, references.map(|c| c.layout))?;
        match neg {
            Some(neg) => {
                let uncond = branch(neg, references.and_then(|c| c.negative_layout))?;
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

/// Final packed latents → the VAE's four-channel decode, NCHW `[1, 4, H, W]` in `[-1, 1]`:
/// unpack, denormalise (`z·std + mean`), decode RGBA (tiled when `tiling` is given).
///
/// The single decode both emissions are built from. There is only ever ONE denoise and ONE decode
/// per image: upstream has no transparency flag and always decodes four channels (see
/// `UPSTREAM.md`), so [`decode_rgb`] and [`decode_rgba`] differ **only** in what they do with the
/// alpha the decoder already produced.
fn decode_to_rgba_tensor(
    vae: &QwenImage21Vae,
    latents: &Array,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Array> {
    let unpacked = unpack_latents(latents, width, height)?;
    let vae_space = vae.denormalize(&unpacked)?;
    match tiling {
        Some(cfg) => vae.decode_rgba_tiled(&vae_space, cfg, cancel),
        None => vae.decode_rgba(&vae_space),
    }
}

/// Final packed latents → RGB8 [`Image`]: `decode_to_rgba_tensor`, composite over white,
/// quantise. The `OutputChannels::Rgb` emission (the default), byte-for-byte unchanged by
/// sc-24111.
pub fn decode_rgb(
    vae: &QwenImage21Vae,
    latents: &Array,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Image> {
    let rgba = decode_to_rgba_tensor(vae, latents, width, height, tiling, cancel)?;
    let rgb = rgba_to_rgb_over_white(&rgba)?;
    decoded_to_image(&rgb)
}

/// Final packed latents → RGBA8 [`RgbaImage`] with **straight (un-premultiplied)** alpha
/// (sc-24111): `decode_to_rgba_tensor`, quantise. The `OutputChannels::Rgba` emission.
///
/// No compositing anywhere on this path — the alpha the VAE decoded reaches the caller, which is
/// the whole point of the opt-in. Quantisation is upstream's
/// `VaeImageProcessor.postprocess`: `clip(x·0.5 + 0.5, 0, 1)` per channel, `(v·255).round()`,
/// `uint8`, interleaved RGBA (see [`mlx_gen::image::decoded_to_rgba_image`]).
pub fn decode_rgba(
    vae: &QwenImage21Vae,
    latents: &Array,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<RgbaImage> {
    let rgba = decode_to_rgba_tensor(vae, latents, width, height, tiling, cancel)?;
    decoded_to_rgba_image(&rgba)
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
