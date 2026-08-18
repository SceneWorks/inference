//! Image-conditioning CRF re-compression (sc-18759) — the candle sibling of
//! `mlx_gen_ltx::image_crf`. See that crate's module docs for the full upstream reference and the
//! H.264-parity caveat; kept in lockstep here, minus the `Device` threading candle's `Tensor` needs.

use candle_gen::candle_core::{Device, Error, Result, Tensor};
use candle_gen::gen_core::Image;

use crate::conditioning::preprocess_conditioning_image;
use crate::params::resolve_generation_params;

/// Recompress an I2V conditioning image at its resolved CRF, then prepare it for VAE encode. See
/// `mlx_gen_ltx::image_crf::condition_image_for_checkpoint` for the full parameter contract (this
/// is its candle twin, threading a `Device` through to [`preprocess_conditioning_image`]).
pub fn condition_image_for_checkpoint(
    image: &Image,
    target_width: u32,
    target_height: u32,
    model_version: &str,
    requested_crf: Option<u8>,
    device: &Device,
    recompress: &mut dyn FnMut(&Image, u8) -> Result<Image>,
) -> Result<Tensor> {
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
    preprocess_conditioning_image(image, target_width, target_height, device)
}

/// Production CRF recompressor — numerically identical stand-in to
/// `mlx_gen_ltx::image_crf::default_image_recompress` (JPEG quality mapped from CRF; **not** a
/// bit-/curve-matched port of upstream's `libx264` H.264 pass — see that function's doc comment
/// for the full caveat, which applies here unchanged).
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
    /// was actually invoked with.
    #[test]
    fn resolved_crf_for_2_5_reaches_the_conditioner_via_instrumentation() {
        let device = Device::Cpu;
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut seen_crf: Option<u8> = None;
        let mut spy = |img: &Image, crf: u8| -> Result<Image> {
            seen_crf = Some(crf);
            Ok(img.clone())
        };
        let _ =
            condition_image_for_checkpoint(&image, 4, 4, "2.5.0", None, &device, &mut spy).unwrap();
        assert_eq!(seen_crf, Some(18));
    }

    /// sc-18759 acceptance: "2.3 renders keep CRF 33."
    #[test]
    fn resolved_crf_for_2_3_reaches_the_conditioner_via_instrumentation() {
        let device = Device::Cpu;
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut seen_crf: Option<u8> = None;
        let mut spy = |img: &Image, crf: u8| -> Result<Image> {
            seen_crf = Some(crf);
            Ok(img.clone())
        };
        let _ =
            condition_image_for_checkpoint(&image, 4, 4, "2.3.0", None, &device, &mut spy).unwrap();
        assert_eq!(seen_crf, Some(33));
    }

    /// `crf == 0` is "no recompression" -- the spy must NOT be called.
    #[test]
    fn zero_crf_skips_recompression_entirely() {
        let device = Device::Cpu;
        let image = solid_image(4, 4, [10, 20, 30]);
        let mut called = false;
        let mut spy = |img: &Image, _crf: u8| -> Result<Image> {
            called = true;
            Ok(img.clone())
        };
        let _ = condition_image_for_checkpoint(&image, 4, 4, "2.5.0", Some(0), &device, &mut spy)
            .unwrap();
        assert!(!called, "crf=0 must skip the recompress hook entirely");
    }

    /// The production recompressor actually degrades pixels (real recompression, not a no-op).
    #[test]
    fn default_recompress_is_real_lossy_work_not_a_noop() {
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
        assert_ne!(
            low_quality.pixels, image.pixels,
            "CRF 51 (worst quality) must actually change pixels"
        );
    }
}
