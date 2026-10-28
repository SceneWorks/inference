//! Ordered reference-image conditioning — the host side of `QwenImage21Pipeline.__call__`'s
//! condition-image path (sc-24110).
//!
//! Upstream has **one** pipeline: "text to image" is `image=None`, and "edit" / "multi-reference
//! composition" / "local editing" are the same call with one to ten ordered condition images.
//! There is no mask tensor, no `strength`, no inpaint and no pixel preservation — see the
//! *Reference conditioning and local editing* section of `UPSTREAM.md`, which this module
//! implements literally.
//!
//! Per reference, in request order:
//!
//! 1. [`calculate_dimensions`] derives that image's own `(w', h')` from
//!    `output_resolution² · (w/h)`, each side rounded (half-to-even) onto the 32-px grid.
//! 2. **One** PIL-LANCZOS resize to `(w', h')` feeds both consumers (diffusers'
//!    `VaeImageProcessor` defaults to `resample="lanczos"`, and upstream calls `.resize(...)` for
//!    the vision tower and `.preprocess(...)` for the VAE on the same source at the same size).
//! 3. The **vision** copy goes through the snapshot's Qwen3-VL processor geometry
//!    ([`candle_llm::image::Qwen35ImageProcessor`]) into `pixel_values` + `grid_thw`.
//! 4. The **VAE** copy becomes RGBA NCHW in `[-1, 1]`, carrying **all four channels**.
//!
//! Upstream converts every condition image to `RGBA` up front (`img.convert("RGBA")`) and then
//! splits the two consumers deliberately: `_get_qwen_prompt_embeds` composites the RGBA copy over
//! **white** before the Qwen3-VL processor sees it ("the checkpoint was trained with the alpha
//! composited over white for the vision encoder"), while `image_processor.preprocess` hands the
//! VAE all four channels. Both of those are reproduced here literally (sc-24111): a
//! [`Conditioning::ReferenceRgba`](candle_gen::gen_core::Conditioning::ReferenceRgba) reference's alpha reaches the VAE encode intact and is
//! flattened over white for the vision tower only.
//!
//! An ordinary RGB [`Conditioning::Reference`](candle_gen::gen_core::Conditioning::Reference) is the `A = 255` special case of exactly that
//! path — the widening upstream's `convert("RGBA")` performs — so the white composite is the
//! identity and the VAE alpha is the constant `+1.0`. The RGB reference route is therefore
//! byte-for-byte what it was before transparent references existed, which `rgb_reference_is_the_
//! opaque_rgba_case` asserts.
//!
//! The two grids have to agree: `(h'/32) · (w'/32)` merged vision patches × 4 latent tokens per
//! slot must equal the `(h'/16) · (w'/16)` VAE tokens. They do whenever the processor's own
//! `smart_resize` is a no-op on a 32-aligned side inside its pixel budget — which is why
//! [`crate::config::VisionConfig::output_resolution`] derives the fit from that budget. A
//! reference `smart_resize` still rebinds is refused with a typed error rather than mis-binding
//! the blocks.

use candle_core::{Device, Tensor};
use candle_gen::gen_core::imageops::resize_lanczos_rgba_u8;
use candle_gen::gen_core::RgbaImage;
use candle_gen::{CandleError as Error, Result};

pub use candle_gen::gen_core::qwen_image_2_1::reference::*;

use crate::config::{VisionConfig, IMAGE_TOKENS_PER_SLOT, VAE_SCALE_FACTOR};

/// One reference, preprocessed for both consumers.
#[derive(Debug)]
pub struct PreparedReference {
    /// Qwen3-VL `pixel_values` `[grid_h·grid_w, C·T·P·P]`, f32.
    pub pixel_values: Tensor,
    /// The single `[1, grid_h, grid_w]` vision grid, in patch units.
    pub grid_thw: [i32; 3],
    /// RGBA NCHW `[1, 4, h', w']` in `[-1, 1]`, f32 — the VAE's input.
    pub vae_input: Tensor,
    /// The resize target `(w', h')` in pixels.
    pub size: (u32, u32),
}

impl PreparedReference {
    /// The reference's latent-token grid `(h'/16, w'/16)` — its [`crate::transformer::Segment`].
    pub fn latent_grid(&self) -> (usize, usize) {
        let scale = VAE_SCALE_FACTOR as usize;
        (self.size.1 as usize / scale, self.size.0 as usize / scale)
    }

    /// Latent tokens this reference contributes to the joint sequence.
    pub fn latent_tokens(&self) -> usize {
        let (h, w) = self.latent_grid();
        h * w
    }

    /// Vision slots (`<|image_pad|>` tokens) the processor emits for it — one per merged block,
    /// i.e. exactly a quarter of [`Self::latent_tokens`].
    pub fn vision_slots(&self) -> usize {
        self.latent_tokens() / IMAGE_TOKENS_PER_SLOT
    }
}

/// The fitted `(w', h')` of a `size = (width, height)` reference, **refused** when the Qwen3-VL
/// processor's `smart_resize` would rebind that fit (its vision slots would then no longer cover
/// its VAE latents 4:1). Pure geometry — no pixel is read — so a caller holding only an image
/// header (the edit trainer's preflight, sc-24162) refuses exactly what [`prepare_reference`]
/// refuses, before any weight loads. `index` only names the offender. The MLX twin's function of
/// the same name.
pub fn reference_fit(size: (u32, u32), index: usize, vision: &VisionConfig) -> Result<(u32, u32)> {
    let (rw, rh) = reference_target_size(size, vision.output_resolution())?;
    let (sh, sw) = vision
        .processor
        .smart_resize(rh as usize, rw as usize)
        .map_err(from_llm)?;
    if (sh, sw) != (rh as usize, rw as usize) {
        let patch = vision.processor.patch_size.max(1);
        return Err(reference_rebind_refusal(
            index,
            size,
            (rw, rh),
            (sw / patch, sh / patch),
            (rw as usize / patch, rh as usize / patch),
            vision.output_resolution(),
            (vision.processor.min_pixels, vision.processor.max_pixels),
        )
        .into());
    }
    Ok((rw, rh))
}

/// Preprocess one reference: resize once (LANCZOS), then fan out to the vision processor and the
/// VAE, both on `device`. `index` only names the offender in error messages.
pub fn prepare_reference(
    image: &RgbaImage,
    index: usize,
    vision: &VisionConfig,
    device: &Device,
) -> Result<PreparedReference> {
    validate_reference_image(image, index)?;
    let (rw, rh) = reference_fit((image.width, image.height), index, vision)?;
    let (src_h, src_w) = (image.height as usize, image.width as usize);
    let (dst_h, dst_w) = (rh as usize, rw as usize);

    // One resize on the u8 **RGBA** source (PIL LANCZOS, `VaeImageProcessor`'s default resample),
    // matching upstream's order: convert to RGBA first, resize all four channels together, split
    // the consumers afterwards. The resampler already clips to `[0, 255]` integers, so the cast is
    // exact.
    let resized: Vec<u8> = if (src_h, src_w) == (dst_h, dst_w) {
        image.pixels.clone()
    } else {
        resize_lanczos_rgba_u8(&image.pixels, src_h, src_w, dst_h, dst_w)
            .map_err(|e| Error::Msg(format!("qwen_image_2_1: reference image {index}: {e}")))?
            .into_iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect()
    };

    // The VISION copy is flattened over white — upstream's `white.paste(img, mask=A)` in
    // `_get_qwen_prompt_embeds`, which the checkpoint was trained with. Only this copy; the VAE
    // below still reads all four channels. For an opaque reference this is the identity.
    let vision_rgb = RgbaImage {
        width: rw,
        height: rh,
        pixels: resized.clone(),
    }
    .to_rgb_over_white()
    .map_err(|e| Error::Msg(format!("qwen_image_2_1: reference image {index}: {e}")))?;

    let (pixel_values, grids) = vision
        .processor
        .preprocess(&vision_rgb.pixels, dst_w, dst_h, device)
        .map_err(from_llm)?;
    let grid_thw = *grids.first().ok_or_else(|| {
        Error::Msg(format!(
            "qwen_image_2_1: the Qwen3-VL processor returned no grid for reference image {index}"
        ))
    })?;

    // The two grids must describe the same picture: `smart_resize` must have left the 32-aligned
    // sides alone. It does not when the fitted reference falls outside the processor's budget.
    // `reference_fit` above already refused a rebinding fit from the geometry alone; this re-checks
    // the grid the processor actually produced, so the two can never disagree silently.
    let patch = vision.processor.patch_size as i32;
    let (want_h, want_w) = ((rh as i32) / patch, (rw as i32) / patch);
    if grid_thw != [1, want_h, want_w] {
        return Err(reference_rebind_refusal(
            index,
            (image.width, image.height),
            (rw, rh),
            (grid_thw[2].max(0) as usize, grid_thw[1].max(0) as usize),
            (want_w.max(0) as usize, want_h.max(0) as usize),
            vision.output_resolution(),
            (vision.processor.min_pixels, vision.processor.max_pixels),
        )
        .into());
    }

    Ok(PreparedReference {
        pixel_values,
        grid_thw,
        vae_input: rgba8_to_nchw(&resized, dst_w, dst_h, device)?,
        size: (rw, rh),
    })
}

/// Preprocess every reference, in order.
pub fn prepare_references(
    images: &[RgbaImage],
    vision: &VisionConfig,
    device: &Device,
) -> Result<Vec<PreparedReference>> {
    validate_reference_count(images.len())?;
    images
        .iter()
        .enumerate()
        .map(|(i, image)| prepare_reference(image, i, vision, device))
        .collect()
}

/// RGBA8 HWC → RGBA NCHW `[1, 4, h, w]` in `[-1, 1]` — `VaeImageProcessor.preprocess`'s
/// `2x − 1`, applied to **all four channels** exactly as upstream hands them to the VAE.
///
/// The alpha is normalised by the same `2x − 1` as the colour (it is just a fourth plane to
/// `preprocess`), which is what makes the decode side's `x·0.5 + 0.5` its exact inverse. An opaque
/// reference (`A = 255`) yields the constant `+1.0` plane the RGB route always produced.
fn rgba8_to_nchw(pixels: &[u8], width: usize, height: usize, device: &Device) -> Result<Tensor> {
    let plane = width * height;
    let mut out = vec![0.0f32; 4 * plane];
    for (c, chunk) in out.chunks_mut(plane).enumerate() {
        for (i, v) in chunk.iter_mut().enumerate() {
            *v = f32::from(pixels[i * 4 + c]) / 255.0 * 2.0 - 1.0;
        }
    }
    Ok(Tensor::from_vec(out, (1, 4, height, width), device)?)
}

fn from_llm(e: candle_llm::Error) -> Error {
    match e {
        candle_llm::Error::Candle(c) => Error::Candle(c),
        other => Error::Msg(format!("qwen_image_2_1 vision: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use candle_gen::gen_core::Image;

    /// A tensor's f32 samples, flattened — the exactness gates below compare bit patterns, so
    /// nothing here rounds or tolerates.
    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn image(width: u32, height: u32) -> Image {
        Image {
            width,
            height,
            pixels: (0..(width as usize * height as usize * 3))
                .map(|i| (i % 251) as u8)
                .collect(),
        }
    }

    /// The same deterministic picture as [`image`], widened the way upstream's
    /// `img.convert("RGBA")` widens an opaque source (`A = 255`).
    fn opaque(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_rgb(&image(width, height)).unwrap()
    }

    /// A genuinely transparent reference: the same colours, with a horizontal alpha ramp so the
    /// alpha is neither constant nor separable from position.
    fn transparent(width: u32, height: u32) -> RgbaImage {
        let mut rgba = opaque(width, height);
        for (i, px) in rgba.pixels.chunks_exact_mut(4).enumerate() {
            px[3] = ((i % width as usize) * 255 / width.max(2) as usize) as u8;
        }
        rgba
    }

    fn production_vision() -> VisionConfig {
        let json = serde_json::json!({
            "image_token_id": 151655,
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        });
        let processor = serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "size": { "shortest_edge": 65536, "longest_edge": 16777216 }
        });
        VisionConfig::from_value(&json, Some(&processor)).unwrap()
    }

    #[test]
    fn the_released_budget_puts_the_fit_at_upstreams_1024() {
        assert_eq!(production_vision().output_resolution(), 1024);
    }

    #[test]
    fn a_reference_binds_four_latent_tokens_per_vision_slot() {
        let vision = production_vision();
        let device = Device::Cpu;
        let prepared = prepare_reference(&opaque(640, 480), 0, &vision, &device).unwrap();
        assert_eq!(prepared.latent_tokens(), prepared.vision_slots() * 4);
        assert_eq!(
            prepared.pixel_values.dims()[0],
            prepared.vision_slots() * 4,
            "one patch row per unmerged patch"
        );
        let (h, w) = prepared.latent_grid();
        assert_eq!(h, prepared.size.1 as usize / 16);
        assert_eq!(w, prepared.size.0 as usize / 16);
        assert_eq!(prepared.vae_input.dims(), &[1, 4, h * 16, w * 16]);
    }

    #[test]
    fn malformed_references_are_refused_before_any_tensor_work() {
        let vision = production_vision();
        let device = Device::Cpu;
        // Built directly: a zero side is exactly what `prepare_reference` must refuse, so it
        // cannot come through the `opaque` helper (whose own widening rejects it first).
        let zero = RgbaImage {
            width: 0,
            height: 8,
            pixels: Vec::new(),
        };
        let err = prepare_reference(&zero, 3, &vision, &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reference image 3"), "{err}");
        let mut short = opaque(8, 8);
        short.pixels.truncate(3);
        let err = prepare_reference(&short, 1, &vision, &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bytes"), "{err}");
    }

    /// The RGB reference route is the **opaque special case** of the RGBA one (sc-24111).
    ///
    /// Two claims, both load-bearing for "sc-24111 changed nothing for an RGB reference":
    ///
    /// * the vision copy is byte-identical to the plain three-channel LANCZOS resize the crate
    ///   performed before transparent references existed — i.e. widening to RGBA, resampling four
    ///   channels and compositing over white round-trips exactly;
    /// * the VAE input's alpha plane is the constant `+1.0` the old `rgb8_to_rgba_nchw` wrote.
    #[test]
    fn rgb_reference_is_the_opaque_rgba_case() {
        let vision = production_vision();
        let device = Device::Cpu;
        let src = image(640, 480);
        let prepared = prepare_reference(&opaque(640, 480), 0, &vision, &device).unwrap();
        let (rw, rh) = prepared.size;

        // The pre-sc-24111 path: resize the three-channel buffer directly.
        let want: Vec<u8> = candle_gen::gen_core::imageops::resize_lanczos_u8(
            &src.pixels,
            src.height as usize,
            src.width as usize,
            rh as usize,
            rw as usize,
        )
        .unwrap()
        .into_iter()
        .map(|v| v.clamp(0.0, 255.0) as u8)
        .collect();
        let (got, _) = vision
            .processor
            .preprocess(&want, rw as usize, rh as usize, &device)
            .unwrap();
        assert_eq!(
            host(&prepared.pixel_values),
            host(&got),
            "an opaque reference's vision copy must be byte-identical to the 3-channel resize"
        );

        // Alpha plane: `[1, 4, h, w]`, channel 3, every sample exactly +1.0.
        let alpha = host(&prepared.vae_input.i((0, 3, .., ..)).unwrap());
        assert!(
            alpha.iter().all(|&a| a == 1.0),
            "an opaque reference's VAE alpha must be the constant +1.0"
        );
    }

    /// A **transparent** reference is fed to the two consumers differently, exactly as upstream
    /// feeds it (sc-24111): the VAE encode sees the real alpha, the Qwen3-VL vision tower sees the
    /// reference composited over white.
    #[test]
    fn a_transparent_reference_reaches_the_vae_and_is_whitened_for_the_vision_tower() {
        let vision = production_vision();
        let device = Device::Cpu;
        let prepared = prepare_reference(&transparent(640, 480), 0, &vision, &device).unwrap();
        let (rw, rh) = prepared.size;

        // The VAE alpha is NOT flattened away — it is a real ramp in `[-1, 1]`.
        let alpha = host(&prepared.vae_input.i((0, 3, .., ..)).unwrap());
        let min = alpha.iter().copied().fold(f32::INFINITY, f32::min);
        let max = alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min < -0.5 && max > 0.5,
            "the VAE must receive the reference's real alpha, got the range [{min}, {max}]"
        );

        // The vision copy is the white composite of the SAME resized RGBA — not a channel drop,
        // and not the un-composited colour. Rebuild both and prove the processor got the former.
        let resized: Vec<u8> = {
            let src = transparent(640, 480);
            resize_lanczos_rgba_u8(
                &src.pixels,
                src.height as usize,
                src.width as usize,
                rh as usize,
                rw as usize,
            )
            .unwrap()
            .into_iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect()
        };
        let whitened = RgbaImage {
            width: rw,
            height: rh,
            pixels: resized.clone(),
        }
        .to_rgb_over_white()
        .unwrap();
        let dropped: Vec<u8> = resized
            .chunks_exact(4)
            .flat_map(|px| px[..3].to_vec())
            .collect();
        assert_ne!(
            whitened.pixels, dropped,
            "the fixture must actually exercise the composite (a ramp alpha changes the colour)"
        );
        let (want, _) = vision
            .processor
            .preprocess(&whitened.pixels, rw as usize, rh as usize, &device)
            .unwrap();
        assert_eq!(
            host(&prepared.pixel_values),
            host(&want),
            "the vision tower must read the reference composited over white"
        );
    }
}
