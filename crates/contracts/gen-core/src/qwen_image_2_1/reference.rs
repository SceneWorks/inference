//! Ordered reference-image conditioning, the backend-neutral half — the host side of
//! `QwenImage21Pipeline.__call__`'s condition-image path (sc-24110), shared by both engines
//! (hoisted from their mirrored `reference.rs` copies, sc-27329).
//!
//! This module owns what reads no pixel and no tensor: the per-reference fit geometry
//! ([`calculate_dimensions`], [`reference_target_size`]), the request flattening and every refusal
//! ([`collect_references`], [`validate_reference_count`], [`validate_reference_image`],
//! [`reference_rebind_refusal`]). Each engine's `reference` module re-exports it and keeps the
//! resize → Qwen3-VL processor → VAE-input fan-out on its own tensors; see that module (and the
//! *Reference conditioning and local editing* section of each engine's `UPSTREAM.md`) for the
//! pipeline in full.

use super::config::{MAX_REFERENCE_IMAGES, SIZE_MULTIPLE};
use crate::imageops::checked_image_buffer_len;
use crate::{Conditioning, Error, GenerationRequest, Image, Result, RgbaImage};

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

/// Reject a reference whose buffer disagrees with its declared size, or whose size is zero.
/// `index` only names the offender in the message.
pub fn validate_reference_image(image: &RgbaImage, index: usize) -> Result<()> {
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
/// [`validate_reference_image`], which each engine's `prepare_reference` runs before it reads
/// pixels.
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

/// The typed refusal for a reference whose fit the Qwen3-VL processor's `smart_resize` rebinds:
/// its vision slots would then no longer cover its VAE latents 4:1. `size` is the source
/// `(width, height)`, `fit` the derived `(w', h')`, `got` / `want` are `(width, height)` patch
/// grids, and `budget` the processor's `[min_pixels, max_pixels]`. Shared so both engines' geometry
/// preflight (`reference_fit`) and post-processor grid check raise one text.
pub fn reference_rebind_refusal(
    index: usize,
    size: (u32, u32),
    (rw, rh): (u32, u32),
    got: (usize, usize),
    want: (usize, usize),
    output_resolution: u32,
    budget: (usize, usize),
) -> Error {
    Error::Unsupported(format!(
        "qwen_image_2_1: reference image {index} ({}x{}) fits to {rw}x{rh}, but the Qwen3-VL \
         processor's smart_resize rebinds it to a {}x{} patch grid instead of {}x{}, so its \
         vision slots would no longer cover its VAE latents 4:1. Supply a reference whose \
         {output_resolution}-px fit stays inside the processor's [{}, {}] pixel budget.",
        size.0, size.1, got.0, got.1, want.0, want.1, budget.0, budget.1,
    ))
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
///
/// Every one of these is the typed [`Error::Unsupported`] (or `Msg`) both engines raise with the
/// identical message text: the worker and the gen-core-testkit validate-honesty check key off the
/// variant to tell "refused by design" from "broke" (sc-24114).
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
            // RGB carrier; the alpha survives to the VAE encode (see each engine's `prepare_reference`).
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
    fn the_derived_size_reads_the_last_reference() {
        // Sizes, not images: the helper is carrier-free, so the claim (it reads the LAST entry)
        // is stated without building pixel buffers at all.
        let refs = [(64, 128), (128, 64)];
        let (w, h) = reference_derived_size(&refs, 1024).unwrap();
        assert!(w > h, "the last reference is landscape, got {w}x{h}");
        assert!(reference_derived_size(&[], 1024).is_err());
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
