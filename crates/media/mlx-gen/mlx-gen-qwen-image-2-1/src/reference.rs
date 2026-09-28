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
//! 4. The **VAE** copy becomes RGBA NCHW in `[-1, 1]`, carrying **all four channels**.
//!
//! Upstream converts every condition image to `RGBA` up front (`img.convert("RGBA")`) and then
//! splits the two consumers deliberately: `_get_qwen_prompt_embeds` composites the RGBA copy over
//! **white** before the Qwen3-VL processor sees it ("the checkpoint was trained with the alpha
//! composited over white for the vision encoder"), while `image_processor.preprocess` hands the
//! VAE all four channels. Both of those are reproduced here literally (sc-24111): a
//! [`Conditioning::ReferenceRgba`] reference's alpha reaches the VAE encode intact and is
//! flattened over white for the vision tower only.
//!
//! An ordinary RGB [`Conditioning::Reference`] is the `A = 255` special case of exactly that
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

use mlx_gen::gen_core::imageops::checked_image_buffer_len;
use mlx_gen::image::resize_lanczos_rgba_u8;
use mlx_gen::{Conditioning, Error, GenerationRequest, Image, Result, RgbaImage};
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
fn validate_image(image: &RgbaImage, index: usize) -> Result<()> {
    if image.width == 0 || image.height == 0 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: reference image {index} is {}x{}; a reference must have both sides \
             above zero",
            image.width, image.height
        )));
    }
    let expected = checked_image_buffer_len(image.width as usize, image.height as usize, 4)
        .ok_or_else(|| {
            Error::Msg(format!(
                "qwen_image_2_1: reference image {index} is {}x{}, which overflows an RGBA \
                     buffer length",
                image.width, image.height
            ))
        })?;
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: reference image {index} carries {} bytes, not the {expected} an RGBA8 \
             {}x{} image needs",
            image.pixels.len(),
            image.width,
            image.height
        )));
    }
    Ok(())
}

/// The per-reference resize target upstream derives for a `(width, height)` at
/// `output_resolution`.
///
/// **Carrier-free on purpose**: this is pure geometry over the two numbers upstream's
/// `calculate_dimensions` actually reads, so an RGB caller, an RGBA caller and a caller holding
/// only a size all reach it without converting anything. Taking `&RgbaImage` here would have
/// forced every RGB caller to widen a whole image — allocating `w·h·4` bytes — to ask a question
/// about its aspect ratio.
///
/// Rejects a zero side (the ratio is undefined); the buffer/length check belongs to
/// [`prepare_reference`], which is the entry point that actually reads pixels.
pub fn reference_target_size(size: (u32, u32), output_resolution: u32) -> Result<(u32, u32)> {
    let (width, height) = size;
    if width == 0 || height == 0 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: a reference is {width}x{height}; both sides must be above zero"
        )));
    }
    let area = f64::from(output_resolution) * f64::from(output_resolution);
    Ok(calculate_dimensions(
        area,
        f64::from(width) / f64::from(height),
    ))
}

/// Upstream's `height`/`width` fallback when the caller pins neither: the **last** reference's
/// aspect ratio at `output_resolution`.
///
/// SceneWorks always sends an explicit size, so this is not on the render path; it is the
/// documented derivation a caller can use to fill the request, and the ordering tests assert that
/// it reads the *last* reference rather than the first. Takes sizes rather than images for the
/// reason [`reference_target_size`] does.
pub fn reference_derived_size(
    reference_sizes: &[(u32, u32)],
    output_resolution: u32,
) -> Result<(u32, u32)> {
    let last = reference_sizes.last().copied().ok_or_else(|| {
        Error::Msg("qwen_image_2_1: no reference images to derive the output size from".into())
    })?;
    reference_target_size(last, output_resolution)
}

/// Preprocess one reference: resize once (LANCZOS), then fan out to the vision processor and the
/// VAE. `index` only names the offender in error messages.
pub fn prepare_reference(
    image: &RgbaImage,
    index: usize,
    vision: &VisionConfig,
) -> Result<PreparedReference> {
    validate_image(image, index)?;
    let output_resolution = vision.output_resolution();
    let (rw, rh) = reference_target_size((image.width, image.height), output_resolution)?;
    let (src_h, src_w) = (image.height as usize, image.width as usize);
    let (dst_h, dst_w) = (rh as usize, rw as usize);

    // One resize on the u8 **RGBA** source (PIL LANCZOS, `VaeImageProcessor`'s default resample),
    // matching upstream's order: convert to RGBA first, resize all four channels together, split
    // the consumers afterwards. The resampler already clips to `[0, 255]` integers, so the cast is
    // exact.
    let resized: Vec<u8> = if (src_h, src_w) == (dst_h, dst_w) {
        image.pixels.clone()
    } else {
        resize_lanczos_rgba_u8(&image.pixels, src_h, src_w, dst_h, dst_w)?
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
    .to_rgb_over_white()?;

    let (pixel_values, grids) = vision
        .processor
        .preprocess(&vision_rgb.pixels, dst_w, dst_h)
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
        vae_input: rgba8_to_nchw(&resized, dst_w, dst_h),
        size: (rw, rh),
    })
}

/// Preprocess every reference, in order.
pub fn prepare_references(
    images: &[RgbaImage],
    vision: &VisionConfig,
) -> Result<Vec<PreparedReference>> {
    validate_reference_count(images.len())?;
    images
        .iter()
        .enumerate()
        .map(|(i, image)| prepare_reference(image, i, vision))
        .collect()
}

/// RGBA8 HWC → RGBA NCHW `[1, 4, h, w]` in `[-1, 1]` — `VaeImageProcessor.preprocess`'s
/// `2x − 1`, applied to **all four channels** exactly as upstream hands them to the VAE.
///
/// The alpha is normalised by the same `2x − 1` as the colour (it is just a fourth plane to
/// `preprocess`), which is what makes the decode side's `x·0.5 + 0.5` its exact inverse. An opaque
/// reference (`A = 255`) yields the constant `+1.0` plane the RGB route always produced.
fn rgba8_to_nchw(pixels: &[u8], width: usize, height: usize) -> Array {
    let plane = width * height;
    let mut out = vec![0.0f32; 4 * plane];
    for (c, chunk) in out.chunks_mut(plane).enumerate() {
        for (i, v) in chunk.iter_mut().enumerate() {
            *v = f32::from(pixels[i * 4 + c]) / 255.0 * 2.0 - 1.0;
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
/// [`Conditioning::Reference`] / [`Conditioning::ReferenceRgba`] / [`Conditioning::MultiReference`]
/// **in request order**, every entry widened to RGBA.
///
/// RGB entries are widened with `A = 255` — upstream's `img.convert("RGBA")`, which it applies to
/// every condition image whatever its mode — so the three carriers land in one ordered list and
/// may be freely interleaved. An [`Conditioning::ReferenceRgba`] entry keeps the alpha it came
/// with (sc-24111).
///
/// Returns `Ok(vec![])` for a request with no conditioning at all — that is the text-to-image
/// route, which stays exactly as it was. Every other shape is a typed refusal:
///
/// * [`Conditioning::Mask`] — upstream exposes no mask tensor; the message names the workaround.
/// * a per-reference `strength` other than `1.0` — condition images carry no strength upstream.
/// * an empty `MultiReference`, or a conditioning list that yields zero images.
/// * more than [`MAX_REFERENCE_IMAGES`].
/// * any other conditioning kind — [`Error::Unsupported`], as the capability floor would.
pub fn collect_references(req: &GenerationRequest) -> Result<Vec<RgbaImage>> {
    if req.conditioning.is_empty() {
        return Ok(Vec::new());
    }
    let mut images: Vec<RgbaImage> = Vec::new();
    for (slot, conditioning) in req.conditioning.iter().enumerate() {
        match conditioning {
            Conditioning::Reference { image, strength } => {
                reject_reference_strength(slot, *strength)?;
                images.push(widen_rgb(image, slot)?);
            }
            // A reference that carries its own alpha (sc-24111) — a transparent layer being
            // edited, or a previously extracted subject. Same ordering and strength rules as the
            // RGB carrier; the alpha survives to the VAE encode (see `prepare_reference`).
            Conditioning::ReferenceRgba { image, strength } => {
                reject_reference_strength(slot, *strength)?;
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
                for image in refs {
                    images.push(widen_rgb(image, slot)?);
                }
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
                     accept; it takes Reference, ReferenceRgba and MultiReference only",
                    other.kind()
                )));
            }
        }
    }
    validate_reference_count(images.len())?;
    Ok(images)
}

/// Upstream's `img.convert("RGBA")`: widen an RGB reference to RGBA with a fully opaque alpha.
/// `slot` names the offender when the caller's buffer disagrees with its declared size.
fn widen_rgb(image: &Image, slot: usize) -> Result<RgbaImage> {
    RgbaImage::from_rgb(image).map_err(|e| {
        Error::Msg(format!(
            "qwen_image_2_1: conditioning slot {slot} carries a malformed reference image: {e}"
        ))
    })
}

/// Condition images carry no strength upstream; accept only an unset or exactly-`1.0` value.
/// Shared by the RGB and RGBA reference carriers so the two cannot drift.
fn reject_reference_strength(slot: usize, strength: Option<f32>) -> Result<()> {
    if let Some(strength) = strength {
        if (strength - 1.0).abs() > f32::EPSILON {
            return Err(Error::Unsupported(format!(
                "qwen_image_2_1: conditioning slot {slot} sets reference strength {strength}, but \
                 Qwen-Image 2.1 conditions on a reference at full weight — upstream's condition \
                 images have no strength. Drop the field (or send 1.0)."
            )));
        }
    }
    Ok(())
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
        // Python's `round` is half-to-EVEN, and the fit lands on a 32-grid tie often enough that
        // the difference is reachable, so pin two ratios whose width sits exactly on `.5`:
        //   r = 1.031494140625 → w_raw = 1024·1.015625 = 1040, 1040/32 = 32.5 → ties-even 32
        //     (1024 px); round-half-away would give 33 (1056 px).
        //   r = 25/16 at the tiny fixture's 64-px fit → w_raw = 80, 80/32 = 2.5 → ties-even 2
        //     (64 px); round-half-away would give 3 (96 px), which is also outside the tiny
        //     processor's pixel budget — the `aspect` fixture case carries that one end to end.
        assert_eq!(
            calculate_dimensions(1024.0 * 1024.0, 1.031_494_140_625),
            (1024, 1024)
        );
        assert_eq!(calculate_dimensions(64.0 * 64.0, 25.0 / 16.0), (64, 64));
        // The `aspect` fixture's two non-square fits: they land exactly on the tiny processor's
        // 4096-px ceiling and give heterogeneous, transposed latent blocks.
        assert_eq!(calculate_dimensions(64.0 * 64.0, 4.0), (128, 32));
        assert_eq!(calculate_dimensions(64.0 * 64.0, 0.25), (32, 128));
        // A degenerate ratio still lands on the grid rather than on zero.
        let (w, h) = calculate_dimensions(1024.0 * 1024.0, 1.0 / 400.0);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w >= 32 && h >= 32, "{w}x{h}");
    }

    #[test]
    fn a_reference_binds_four_latent_tokens_per_vision_slot() {
        let vision = production_vision();
        let prepared = prepare_reference(&opaque(640, 480), 0, &vision).unwrap();
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
        // Built directly: a zero side is exactly what `prepare_reference` must refuse, so it
        // cannot come through the `opaque` helper (whose own widening rejects it first).
        let zero = RgbaImage {
            width: 0,
            height: 8,
            pixels: Vec::new(),
        };
        let err = prepare_reference(&zero, 3, &vision)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reference image 3"), "{err}");
        let mut short = opaque(8, 8);
        short.pixels.truncate(3);
        let err = prepare_reference(&short, 1, &vision)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bytes"), "{err}");
    }

    #[test]
    fn the_derived_size_reads_the_last_reference() {
        // Sizes, not images: the helper is carrier-free, so the claim (it reads the LAST entry)
        // is stated without building pixel buffers at all.
        let refs = [(64, 128), (128, 64)];
        let (w, h) = reference_derived_size(&refs, 1024).unwrap();
        assert!(w > h, "the last reference is landscape, got {w}x{h}");
        assert!(reference_derived_size(&[], 1024).is_err());
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
        let src = image(640, 480);
        let prepared = prepare_reference(&opaque(640, 480), 0, &vision).unwrap();
        let (rw, rh) = prepared.size;

        // The pre-sc-24111 path: resize the three-channel buffer directly.
        let want: Vec<u8> = mlx_gen::image::resize_lanczos_u8(
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
            .preprocess(&want, rw as usize, rh as usize)
            .unwrap();
        assert!(
            prepared
                .pixel_values
                .all_close(&got, Some(0.0), Some(0.0), None)
                .unwrap()
                .item::<bool>(),
            "an opaque reference's vision copy must be byte-identical to the 3-channel resize"
        );

        // Alpha plane: `[1, 4, h, w]`, channel 3, every sample exactly +1.0.
        let plane = mlx_rs::ops::indexing::IndexOp::index(&prepared.vae_input, (0, 3, .., ..));
        let alpha: Vec<f32> = plane
            .flatten(None, None)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
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
        let prepared = prepare_reference(&transparent(640, 480), 0, &vision).unwrap();
        let (rw, rh) = prepared.size;

        // The VAE alpha is NOT flattened away — it is a real ramp in `[-1, 1]`.
        let plane = mlx_rs::ops::indexing::IndexOp::index(&prepared.vae_input, (0, 3, .., ..));
        let alpha: Vec<f32> = plane
            .flatten(None, None)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
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
            mlx_gen::image::resize_lanczos_rgba_u8(
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
            .preprocess(&whitened.pixels, rw as usize, rh as usize)
            .unwrap();
        assert!(
            prepared
                .pixel_values
                .all_close(&want, Some(0.0), Some(0.0), None)
                .unwrap()
                .item::<bool>(),
            "the vision tower must read the reference composited over white"
        );
    }

    /// RGB and RGBA references share ONE ordered list, in request order (sc-24111). Ordering is
    /// semantic for this model, so an interleaved request must not be silently regrouped.
    #[test]
    fn rgb_and_rgba_references_interleave_in_request_order() {
        let rgb = image(48, 48);
        let rgba = transparent(64, 64);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: rgb.clone(),
                    strength: None,
                },
                Conditioning::ReferenceRgba {
                    image: rgba.clone(),
                    strength: Some(1.0),
                },
                Conditioning::MultiReference {
                    images: vec![rgb.clone()],
                },
            ],
            ..Default::default()
        };
        let refs = collect_references(&req).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0], RgbaImage::from_rgb(&rgb).unwrap());
        assert_eq!(refs[1], rgba, "the RGBA reference keeps its own alpha");
        assert_eq!(refs[2], RgbaImage::from_rgb(&rgb).unwrap());
        assert!(refs[0].is_opaque() && refs[2].is_opaque());
        assert!(!refs[1].is_opaque());
    }

    /// A transparent reference carries no strength either — the same refusal as the RGB carrier,
    /// from the one shared check.
    #[test]
    fn a_transparent_reference_still_carries_no_strength() {
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            conditioning: vec![Conditioning::ReferenceRgba {
                image: transparent(64, 64),
                strength: Some(0.4),
            }],
            ..Default::default()
        };
        let err = collect_references(&req).unwrap_err().to_string();
        assert!(err.contains("full weight"), "{err}");
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
