//! Image-conditioning CRF re-compression (sc-18759).
//!
//! Upstream re-compresses every I2V conditioning image at a checkpoint-generation-specific H.264
//! CRF before VAE encode, matching the compression the model was trained against
//! (`ImageConditioner.resolve_crf` + `media_io/decode.py::preprocess`/`encode_single_frame`,
//! reference `d1511477`). [`condition_image_for_checkpoint`] is the plumbing seam: it resolves the
//! CRF for a `model_version` via [`crate::params::resolve_generation_params`] exactly the way
//! upstream's `ImageConditioner.default_image_crf` does, then hands the image to an injected
//! `recompress` step before [`crate::pipeline::preprocess_conditioning_image`] prepares it for the
//! VAE. Injecting the recompressor is what makes the resolved CRF value provable by
//! instrumentation (a test spy) rather than only by reading the `LTX_2_5_PARAMS` constant.
//!
//! [`default_image_recompress`] is the production recompressor. It is a real, working lossy
//! re-encode (JPEG quality mapped from CRF) — **not** a bit- or curve-matched port of upstream's
//! `libx264` H.264 pass. True H.264 CRF parity needs a video-codec dependency (shell to `ffmpeg`,
//! or a native encoder) available on macOS *and* the Windows/Linux CUDA runners candle-gen-ltx
//! targets; that cross-platform dependency choice is out of this story's scope and is called out
//! to Michael rather than decided here.

use mlx_gen::{Error, Image, Result};

use crate::params::resolve_generation_params;
use crate::pipeline::preprocess_conditioning_image;

/// Recompress an I2V conditioning image at its resolved CRF, then prepare it for VAE encode.
///
/// * `model_version` — the loaded checkpoint's declared version (e.g. `"2.5.0"`); resolved through
///   [`resolve_generation_params`] when `requested_crf` is `None`.
/// * `requested_crf` — mirrors upstream's `ImageConditioningInput.crf`: `None` resolves to the
///   checkpoint generation's `default_image_crf`; `Some(0)` means "no recompression" (upstream's
///   `crf == 0` no-op); any other explicit value is honored as-is, overriding the checkpoint
///   default.
/// * `recompress` — the actual re-encode step, injected so production and tests can supply (or
///   spy on) it independently of the CRF-resolution logic above.
pub fn condition_image_for_checkpoint(
    image: &Image,
    target_width: u32,
    target_height: u32,
    model_version: &str,
    requested_crf: Option<u8>,
    recompress: &mut dyn FnMut(&Image, u8) -> Result<Image>,
) -> Result<mlx_rs::Array> {
    let crf = match requested_crf {
        Some(crf) => crf,
        None => resolve_generation_params(model_version)?.default_image_crf,
    };
    let recompressed;
    let image = if crf == 0 {
        image
    } else {
        recompressed = recompress(image, crf)?;
        &recompressed
    };
    preprocess_conditioning_image(image, target_width, target_height)
}

/// Production CRF recompressor (see module docs for the H.264-parity caveat). Maps the CRF's
/// `0..=51` libx264 range (0 = lossless, 51 = worst) onto a JPEG quality (`1..=100`, inverted) and
/// round-trips the image through a JPEG encode/decode, applying real lossy degradation that scales
/// with the resolved CRF rather than silently skipping re-compression.
pub fn default_image_recompress(image: &Image, crf: u8) -> Result<Image> {
    use image::codecs::jpeg::JpegEncoder;
    use image::{ExtendedColorType, ImageEncoder};

    let quality = 100u32.saturating_sub(u32::from(crf) * 2).clamp(1, 100) as u8;
    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, quality)
        .write_image(
            &image.pixels,
            image.width,
            image.height,
            ExtendedColorType::Rgb8,
        )
        .map_err(|e| Error::Msg(format!("ltx: CRF re-compress encode (crf={crf}): {e}")))?;
    let decoded = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
        .map_err(|e| Error::Msg(format!("ltx: CRF re-compress decode (crf={crf}): {e}")))?
        .into_rgb8();
    Ok(Image {
        width: image.width,
        height: image.height,
        pixels: decoded.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32, rgb: [u8; 3]) -> Image {
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            pixels.extend_from_slice(&rgb);
        }
        Image {
            width: w,
            height: h,
            pixels,
        }
    }

    /// sc-18759 acceptance: "The image-conditioning CRF used for a 2.5 i2v render is 18, proven by
    /// instrumentation rather than by reading the constant." A spy `recompress` records the CRF it
    /// was actually invoked with; the assertion is against that recorded call, not against
    /// `LTX_2_5_PARAMS.default_image_crf` directly.
    #[test]
    fn resolved_crf_for_2_5_reaches_the_conditioner_via_instrumentation() {
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut seen_crf: Option<u8> = None;
        let mut spy = |img: &Image, crf: u8| -> Result<Image> {
            seen_crf = Some(crf);
            Ok(img.clone())
        };
        let _ = condition_image_for_checkpoint(&image, 4, 4, "2.5.0", None, &mut spy).unwrap();
        assert_eq!(seen_crf, Some(18));
    }

    /// sc-18759 acceptance: "2.3 renders keep CRF 33." Same instrumentation, 2.3 checkpoint.
    #[test]
    fn resolved_crf_for_2_3_reaches_the_conditioner_via_instrumentation() {
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut seen_crf: Option<u8> = None;
        let mut spy = |img: &Image, crf: u8| -> Result<Image> {
            seen_crf = Some(crf);
            Ok(img.clone())
        };
        let _ = condition_image_for_checkpoint(&image, 4, 4, "2.3.0", None, &mut spy).unwrap();
        assert_eq!(seen_crf, Some(33));
    }

    /// An explicit `Some(crf)` overrides the checkpoint default (matches upstream's
    /// `ImageConditioningInput.crf` semantics) — proven by the same instrumentation.
    #[test]
    fn explicit_crf_override_reaches_the_conditioner_and_skips_resolution() {
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut seen_crf: Option<u8> = None;
        let mut spy = |img: &Image, crf: u8| -> Result<Image> {
            seen_crf = Some(crf);
            Ok(img.clone())
        };
        // A bogus model_version would error if resolution were reached at all -- proves the
        // explicit override short-circuits resolve_generation_params.
        let _ = condition_image_for_checkpoint(&image, 4, 4, "9.9.9", Some(40), &mut spy).unwrap();
        assert_eq!(seen_crf, Some(40));
    }

    /// `crf == 0` is "no recompression" (upstream's no-op) — the spy must NOT be called.
    #[test]
    fn zero_crf_skips_recompression_entirely() {
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut called = false;
        let mut spy = |img: &Image, _crf: u8| -> Result<Image> {
            called = true;
            Ok(img.clone())
        };
        let _ = condition_image_for_checkpoint(&image, 4, 4, "2.5.0", Some(0), &mut spy).unwrap();
        assert!(!called, "crf=0 must skip the recompress hook entirely");
    }

    /// An unrecognized model_version with no explicit crf override propagates
    /// `resolve_generation_params`'s error rather than silently defaulting.
    #[test]
    fn unresolvable_version_without_override_errors() {
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut spy = |img: &Image, _crf: u8| -> Result<Image> { Ok(img.clone()) };
        assert!(condition_image_for_checkpoint(&image, 4, 4, "2.6.0", None, &mut spy).is_err());
    }

    /// The production recompressor actually degrades pixels (real recompression, not a no-op
    /// pass-through) and round-trips to a same-sized RGB8 buffer.
    #[test]
    fn default_recompress_is_real_lossy_work_not_a_noop() {
        // A gradient (not a flat color) so JPEG's DCT quantization has something to degrade.
        let w = 16;
        let h = 16;
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.extend_from_slice(&[(x * 16) as u8, (y * 16) as u8, 128]);
            }
        }
        let image = Image {
            width: w,
            height: h,
            pixels,
        };
        let low_quality = default_image_recompress(&image, 51).unwrap();
        assert_eq!(low_quality.width, w);
        assert_eq!(low_quality.height, h);
        assert_eq!(low_quality.pixels.len(), image.pixels.len());
        assert_ne!(
            low_quality.pixels, image.pixels,
            "CRF 51 (worst quality) must actually change pixels"
        );
    }
}
