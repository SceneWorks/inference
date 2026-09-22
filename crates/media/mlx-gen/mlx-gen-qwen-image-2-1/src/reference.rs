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
//!    ([`mlx_llm::image::Qwen35ImageProcessor`]) into `pixel_values` + `grid_thw`.
//! 4. The **VAE** copy becomes RGBA NCHW in `[-1, 1]`. gen-core's [`Image`] is RGB8, so the alpha
//!    upstream's `img.convert("RGBA")` synthesises for an opaque source is the constant `1.0`, and
//!    upstream's "flatten RGBA over white for the vision tower" step is then the identity — which
//!    is what lets one resized buffer serve both.
//!
//! The two grids have to agree: `(h'/32) · (w'/32)` merged vision patches × 4 latent tokens per
//! slot must equal the `(h'/16) · (w'/16)` VAE tokens. They do whenever the processor's own
//! `smart_resize` is a no-op on a 32-aligned side inside its pixel budget — which is why
//! [`crate::config::VisionConfig::output_resolution`] derives the fit from that budget. A
//! reference `smart_resize` still rebinds is refused with a typed error rather than mis-binding
//! the blocks.

use mlx_gen::gen_core::imageops::checked_image_buffer_len;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{Conditioning, Error, GenerationRequest, Image, Result};
use mlx_rs::Array;

use crate::config::{
    VisionConfig, IMAGE_TOKENS_PER_SLOT, MAX_REFERENCE_IMAGES, SIZE_MULTIPLE, VAE_SCALE_FACTOR,
};

/// `calculate_dimensions(target_area, ratio)` — `w = round(sqrt(A·r)/32)·32`,
/// `h = round((w/r)/32)·32`. Python's `round` is half-to-even, which [`f64::round_ties_even`]
/// reproduces exactly. Never returns a zero side.
pub fn calculate_dimensions(target_area: f64, ratio: f64) -> (u32, u32) {
    let multiple = SIZE_MULTIPLE as f64;
    let width = (target_area * ratio).sqrt();
    let height = width / ratio;
    let snap = |v: f64| -> u32 {
        if !v.is_finite() {
            return SIZE_MULTIPLE;
        }
        ((v / multiple).round_ties_even() * multiple).max(multiple) as u32
    };
    (snap(width), snap(height))
}

/// One reference, preprocessed for both consumers.
#[derive(Debug)]
pub struct PreparedReference {
    /// Qwen3-VL `pixel_values` `[grid_h·grid_w, C·T·P·P]`, f32.
    pub pixel_values: Array,
    /// The single `[1, grid_h, grid_w]` vision grid, in patch units.
    pub grid_thw: [i32; 3],
    /// RGBA NCHW `[1, 4, h', w']` in `[-1, 1]`, f32 — the VAE's input.
    pub vae_input: Array,
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

/// Reject a reference whose buffer disagrees with its declared size, or whose size is zero.
/// `index` only names the offender in the message.
fn validate_image(image: &Image, index: usize) -> Result<()> {
    if image.width == 0 || image.height == 0 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: reference image {index} is {}x{}; a reference must have both sides \
             above zero",
            image.width, image.height
        )));
    }
    let expected = checked_image_buffer_len(image.width as usize, image.height as usize, 3)
        .ok_or_else(|| {
            Error::Msg(format!(
                "qwen_image_2_1: reference image {index} is {}x{}, which overflows an RGB \
                     buffer length",
                image.width, image.height
            ))
        })?;
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: reference image {index} carries {} bytes, not the {expected} an RGB8 \
             {}x{} image needs",
            image.pixels.len(),
            image.width,
            image.height
        )));
    }
    Ok(())
}

/// The per-reference resize target upstream derives for `image` at `output_resolution`.
pub fn reference_target_size(image: &Image, output_resolution: u32) -> Result<(u32, u32)> {
    validate_image(image, 0)?;
    let area = f64::from(output_resolution) * f64::from(output_resolution);
    Ok(calculate_dimensions(
        area,
        f64::from(image.width) / f64::from(image.height),
    ))
}

/// Upstream's `height`/`width` fallback when the caller pins neither: the **last** reference's
/// aspect ratio at `output_resolution`.
///
/// SceneWorks always sends an explicit size, so this is not on the render path; it is the
/// documented derivation a caller can use to fill the request, and the ordering tests assert that
/// it reads the *last* reference rather than the first.
pub fn reference_derived_size(references: &[Image], output_resolution: u32) -> Result<(u32, u32)> {
    let last = references.last().ok_or_else(|| {
        Error::Msg("qwen_image_2_1: no reference images to derive the output size from".into())
    })?;
    reference_target_size(last, output_resolution)
}

/// Preprocess one reference: resize once (LANCZOS), then fan out to the vision processor and the
/// VAE. `index` only names the offender in error messages.
pub fn prepare_reference(
    image: &Image,
    index: usize,
    vision: &VisionConfig,
) -> Result<PreparedReference> {
    validate_image(image, index)?;
    let output_resolution = vision.output_resolution();
    let (rw, rh) = reference_target_size(image, output_resolution)?;
    let (src_h, src_w) = (image.height as usize, image.width as usize);
    let (dst_h, dst_w) = (rh as usize, rw as usize);

    // One resize on the u8 RGB source (PIL LANCZOS, `VaeImageProcessor`'s default resample). The
    // resampler already clips to `[0, 255]` integers, so the cast is exact.
    let resized: Vec<u8> = if (src_h, src_w) == (dst_h, dst_w) {
        image.pixels.clone()
    } else {
        resize_lanczos_u8(&image.pixels, src_h, src_w, dst_h, dst_w)?
            .into_iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect()
    };

    let (pixel_values, grids) = vision
        .processor
        .preprocess(&resized, dst_w, dst_h)
        .map_err(from_llm)?;
    let grid_thw = *grids.first().ok_or_else(|| {
        Error::Msg(format!(
            "qwen_image_2_1: the Qwen3-VL processor returned no grid for reference image {index}"
        ))
    })?;

    // The two grids must describe the same picture: `smart_resize` must have left the 32-aligned
    // sides alone. It does not when the fitted reference falls outside the processor's budget.
    let patch = vision.processor.patch_size as i32;
    let (want_h, want_w) = ((rh as i32) / patch, (rw as i32) / patch);
    if grid_thw != [1, want_h, want_w] {
        return Err(Error::Unsupported(format!(
            "qwen_image_2_1: reference image {index} ({}x{}) fits to {rw}x{rh}, but the Qwen3-VL \
             processor's smart_resize rebinds it to a {}x{} patch grid instead of {want_w}x{want_h}, \
             so its vision slots would no longer cover its VAE latents 4:1. Supply a reference whose \
             {output_resolution}-px fit stays inside the processor's [{}, {}] pixel budget.",
            image.width,
            image.height,
            grid_thw[2],
            grid_thw[1],
            vision.processor.min_pixels,
            vision.processor.max_pixels,
        )));
    }

    Ok(PreparedReference {
        pixel_values,
        grid_thw,
        vae_input: rgb8_to_rgba_nchw(&resized, dst_w, dst_h),
        size: (rw, rh),
    })
}

/// Preprocess every reference, in order.
pub fn prepare_references(
    images: &[Image],
    vision: &VisionConfig,
) -> Result<Vec<PreparedReference>> {
    validate_reference_count(images.len())?;
    images
        .iter()
        .enumerate()
        .map(|(i, image)| prepare_reference(image, i, vision))
        .collect()
}

/// RGB8 HWC → RGBA NCHW `[1, 4, h, w]` in `[-1, 1]`, alpha constant `1.0` (upstream's
/// `img.convert("RGBA")` on an opaque source, then `VaeImageProcessor`'s `2x − 1`).
fn rgb8_to_rgba_nchw(pixels: &[u8], width: usize, height: usize) -> Array {
    let plane = width * height;
    let mut out = vec![1.0f32; 4 * plane];
    for (c, chunk) in out.chunks_mut(plane).enumerate().take(3) {
        for (i, v) in chunk.iter_mut().enumerate() {
            *v = f32::from(pixels[i * 3 + c]) / 255.0 * 2.0 - 1.0;
        }
    }
    Array::from_slice(&out, &[1, 4, height as i32, width as i32])
}

fn from_llm(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Unsupported(m) => Error::Unsupported(m),
        mlx_llm::Error::MissingTensor(k) => Error::MissingTensor(k),
        other => Error::Msg(format!("qwen_image_2_1 vision: {other}")),
    }
}

/// The ordered reference list a request carries, flattened from every
/// [`Conditioning::Reference`] / [`Conditioning::MultiReference`] **in request order**.
///
/// Returns `Ok(vec![])` for a request with no conditioning at all — that is the text-to-image
/// route, which stays exactly as it was. Every other shape is a typed refusal:
///
/// * [`Conditioning::Mask`] — upstream exposes no mask tensor; the message names the workaround.
/// * a per-reference `strength` other than `1.0` — condition images carry no strength upstream.
/// * an empty `MultiReference`, or a conditioning list that yields zero images.
/// * more than [`MAX_REFERENCE_IMAGES`].
/// * any other conditioning kind — [`Error::Unsupported`], as the capability floor would.
pub fn collect_references(req: &GenerationRequest) -> Result<Vec<Image>> {
    if req.conditioning.is_empty() {
        return Ok(Vec::new());
    }
    let mut images: Vec<Image> = Vec::new();
    for (slot, conditioning) in req.conditioning.iter().enumerate() {
        match conditioning {
            Conditioning::Reference { image, strength } => {
                if let Some(strength) = strength {
                    if (strength - 1.0).abs() > f32::EPSILON {
                        return Err(Error::Unsupported(format!(
                            "qwen_image_2_1: conditioning slot {slot} sets reference strength \
                             {strength}, but Qwen-Image 2.1 conditions on a reference at full \
                             weight — upstream's condition images have no strength. Drop the field \
                             (or send 1.0)."
                        )));
                    }
                }
                images.push(image.clone());
            }
            Conditioning::MultiReference { images: refs } => {
                if refs.is_empty() {
                    return Err(Error::Msg(format!(
                        "qwen_image_2_1: conditioning slot {slot} is an empty MultiReference; send \
                         one to {MAX_REFERENCE_IMAGES} reference images, or no conditioning at all \
                         for text-to-image"
                    )));
                }
                images.extend(refs.iter().cloned());
            }
            Conditioning::Mask { .. } => {
                return Err(Error::Unsupported(format!(
                    "qwen_image_2_1: conditioning slot {slot} is a Mask, and Qwen-Image 2.1 has no \
                     mask input — upstream's pipeline takes only an ordered list of condition \
                     images and performs no inpainting. For a local edit, either draw the \
                     annotation (a circle or paint stroke) into the reference image itself, or pass \
                     the mask as an ordinary extra reference and name it in the prompt (\"use the \
                     second image as the mask\")."
                )));
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "qwen_image_2_1: conditioning slot {slot} is {:?}, which this route does not \
                     accept; it takes Reference and MultiReference only",
                    other.kind()
                )));
            }
        }
    }
    validate_reference_count(images.len())?;
    Ok(images)
}

/// The 1..=[`MAX_REFERENCE_IMAGES`] bound, as its own entry point so the boundary and the refusal
/// can be exercised without building a whole request.
pub fn validate_reference_count(count: usize) -> Result<()> {
    if count == 0 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: the request carries conditioning but no reference images; send one to \
             {MAX_REFERENCE_IMAGES}, or no conditioning at all for text-to-image"
        )));
    }
    if count > MAX_REFERENCE_IMAGES {
        return Err(Error::Unsupported(format!(
            "qwen_image_2_1: {count} reference images were supplied; upstream composes at most \
             {MAX_REFERENCE_IMAGES}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32) -> Image {
        Image {
            width,
            height,
            pixels: (0..(width as usize * height as usize * 3))
                .map(|i| (i % 251) as u8)
                .collect(),
        }
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
    fn calculate_dimensions_snaps_to_the_32_grid() {
        assert_eq!(calculate_dimensions(1024.0 * 1024.0, 1.0), (1024, 1024));
        // 16:9 — w = sqrt(1024²·16/9) = 1365.33 → 1376; h = 1376/(16/9) = 774 → 768.
        assert_eq!(
            calculate_dimensions(1024.0 * 1024.0, 16.0 / 9.0),
            (1376, 768)
        );
        // 9:16 is the transpose.
        assert_eq!(
            calculate_dimensions(1024.0 * 1024.0, 9.0 / 16.0),
            (768, 1376)
        );
        // A degenerate ratio still lands on the grid rather than on zero.
        let (w, h) = calculate_dimensions(1024.0 * 1024.0, 1.0 / 400.0);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w >= 32 && h >= 32, "{w}x{h}");
    }

    #[test]
    fn a_reference_binds_four_latent_tokens_per_vision_slot() {
        let vision = production_vision();
        let prepared = prepare_reference(&image(640, 480), 0, &vision).unwrap();
        assert_eq!(prepared.latent_tokens(), prepared.vision_slots() * 4);
        assert_eq!(
            prepared.pixel_values.shape()[0] as usize,
            prepared.vision_slots() * 4,
            "one patch row per unmerged patch"
        );
        let (h, w) = prepared.latent_grid();
        assert_eq!(h, prepared.size.1 as usize / 16);
        assert_eq!(w, prepared.size.0 as usize / 16);
        assert_eq!(
            prepared.vae_input.shape(),
            &[1, 4, h as i32 * 16, w as i32 * 16]
        );
    }

    #[test]
    fn malformed_references_are_refused_before_any_tensor_work() {
        let vision = production_vision();
        let err = prepare_reference(&image(0, 8), 3, &vision)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reference image 3"), "{err}");
        let mut short = image(8, 8);
        short.pixels.truncate(3);
        let err = prepare_reference(&short, 1, &vision)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bytes"), "{err}");
    }

    #[test]
    fn the_derived_size_reads_the_last_reference() {
        let refs = vec![image(64, 128), image(128, 64)];
        let (w, h) = reference_derived_size(&refs, 1024).unwrap();
        assert!(w > h, "the last reference is landscape, got {w}x{h}");
        assert!(reference_derived_size(&[], 1024).is_err());
    }

    #[test]
    fn reference_counts_hold_the_one_to_ten_window() {
        assert!(validate_reference_count(1).is_ok());
        assert!(validate_reference_count(MAX_REFERENCE_IMAGES).is_ok());
        let err = validate_reference_count(0).unwrap_err().to_string();
        assert!(err.contains("no reference images"), "{err}");
        let err = validate_reference_count(MAX_REFERENCE_IMAGES + 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at most 10"), "{err}");
    }

    #[test]
    fn conditioning_is_flattened_in_request_order_and_masks_are_refused() {
        let mut req = GenerationRequest {
            prompt: "compose".into(),
            ..Default::default()
        };
        assert!(collect_references(&req).unwrap().is_empty());

        req.conditioning = vec![
            Conditioning::Reference {
                image: image(8, 16),
                strength: None,
            },
            Conditioning::MultiReference {
                images: vec![image(16, 8), image(32, 8)],
            },
        ];
        let refs = collect_references(&req).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!((refs[0].width, refs[0].height), (8, 16));
        assert_eq!((refs[2].width, refs[2].height), (32, 8));

        req.conditioning = vec![Conditioning::Mask { image: image(8, 8) }];
        let err = collect_references(&req).unwrap_err().to_string();
        assert!(err.contains("no mask input"), "{err}");
        assert!(err.contains("extra reference"), "{err}");

        req.conditioning = vec![Conditioning::Reference {
            image: image(8, 8),
            strength: Some(0.5),
        }];
        let err = collect_references(&req).unwrap_err().to_string();
        assert!(err.contains("strength"), "{err}");

        req.conditioning = vec![Conditioning::MultiReference { images: vec![] }];
        let err = collect_references(&req).unwrap_err().to_string();
        assert!(err.contains("empty MultiReference"), "{err}");

        req.conditioning = (0..11)
            .map(|_| Conditioning::Reference {
                image: image(8, 8),
                strength: None,
            })
            .collect();
        let err = collect_references(&req).unwrap_err().to_string();
        assert!(err.contains("at most 10"), "{err}");
    }
}
