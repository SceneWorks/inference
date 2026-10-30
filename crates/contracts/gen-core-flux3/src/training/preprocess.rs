//! Frozen training pixel recipe, deliberately separate from inference reference
//! preparation. Intake is the explicitly documented S3 dataset-oriented RGB/RGBA
//! boundary, not a universal raw-PIL/datasets-version claim.
use super::super::image_input::{decode_reference_file, DecodedReference};
use super::rng::NativeRng;
use gen_core::imageops::{
    checked_image_buffer_len, resize_bilinear_u8, resize_lanczos_rgba_u8, resize_lanczos_u8,
};
use gen_core::{Error, Image, Result};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 training pixels: {detail}"))
}
#[derive(Clone, Copy, Debug)]
pub struct PixelOptions {
    pub width: u32,
    pub height: u32,
    pub center_crop: bool,
    pub random_flip: bool,
    pub seed: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub resized_width: u32,
    pub resized_height: u32,
    pub crop_x: u32,
    pub crop_y: u32,
    pub flip: bool,
    pub reference_prepass: Option<(u32, u32)>,
}
pub struct PreparedPair {
    pub target: Image,
    pub reference: Option<Image>,
    pub geometry: Geometry,
}
fn validate(width: u32, height: u32, pixels: usize, channels: usize) -> Result<()> {
    let needed = checked_image_buffer_len(width as usize, height as usize, channels)
        .ok_or_else(|| invalid("pixel shape overflows"))?;
    if width == 0 || height == 0 || pixels != needed {
        return Err(invalid("invalid/empty pixel buffer"));
    }
    Ok(())
}
fn rgb(decoded: DecodedReference) -> Result<Image> {
    match decoded {
        DecodedReference::Rgb(image) => {
            validate(image.width, image.height, image.pixels.len(), 3)?;
            Ok(image)
        }
        DecodedReference::Rgba(image) => {
            validate(image.width, image.height, image.pixels.len(), 4)?;
            Ok(Image {
                width: image.width,
                height: image.height,
                pixels: image
                    .pixels
                    .chunks_exact(4)
                    .flat_map(|pixel| pixel[..3].iter().copied())
                    .collect(),
            })
        }
    }
}
fn crop_rgb(image: &Image, x: u32, y: u32, width: u32, height: u32, flip: bool) -> Result<Image> {
    validate(image.width, image.height, image.pixels.len(), 3)?;
    if x.checked_add(width).is_none_or(|n| n > image.width)
        || y.checked_add(height).is_none_or(|n| n > image.height)
    {
        return Err(invalid("crop lies outside resized pixels"));
    }
    let length = checked_image_buffer_len(width as usize, height as usize, 3)
        .ok_or_else(|| invalid("crop shape overflows"))?;
    let mut pixels = vec![0u8; length];
    for row in 0..height as usize {
        for column in 0..width as usize {
            let source_x = x as usize
                + if flip {
                    width as usize - 1 - column
                } else {
                    column
                };
            let source = ((y as usize + row) * image.width as usize + source_x) * 3;
            let destination = (row * width as usize + column) * 3;
            pixels[destination..destination + 3].copy_from_slice(&image.pixels[source..source + 3]);
        }
    }
    Ok(Image {
        width,
        height,
        pixels,
    })
}
fn resize_rgb(image: &Image, width: u32, height: u32) -> Result<Image> {
    validate(image.width, image.height, image.pixels.len(), 3)?;
    checked_image_buffer_len(width as usize, height as usize, 3)
        .ok_or_else(|| invalid("resized shape overflows"))?;
    let pixels = resize_bilinear_u8(
        &image.pixels,
        image.height as usize,
        image.width as usize,
        height as usize,
        width as usize,
    )?
    .into_iter()
    .map(|v| v as u8)
    .collect();
    Ok(Image {
        width,
        height,
        pixels,
    })
}

/// `_img2img.py` conditioning prepass: optional >1MP Lanczos, effective grid16
/// PURE center crop, RGB, F32 normalization and source conditional byte roundtrip.
/// In particular all-bright inputs skip inverse normalization; do not "fix" it.
pub fn conditioning_prepass(mut decoded: DecodedReference) -> Result<Image> {
    let (mut width, mut height) = match &decoded {
        DecodedReference::Rgb(i) => (i.width, i.height),
        DecodedReference::Rgba(i) => (i.width, i.height),
    };
    let area = width as u64 * height as u64;
    if area > 1024 * 1024 {
        let scale = ((1024.0 * 1024.0) / (area as f64)).sqrt();
        let out_width = (width as f64 * scale) as u32;
        let out_height = (height as f64 * scale) as u32;
        if out_width == 0 || out_height == 0 {
            return Err(invalid(
                "source area resize produced an empty conditioning image",
            ));
        }
        decoded = match decoded {
            DecodedReference::Rgb(image) => {
                validate(width, height, image.pixels.len(), 3)?;
                DecodedReference::Rgb(Image {
                    width: out_width,
                    height: out_height,
                    pixels: resize_lanczos_u8(
                        &image.pixels,
                        height as usize,
                        width as usize,
                        out_height as usize,
                        out_width as usize,
                    )?
                    .into_iter()
                    .map(|v| v as u8)
                    .collect(),
                })
            }
            DecodedReference::Rgba(image) => {
                validate(width, height, image.pixels.len(), 4)?;
                DecodedReference::Rgba(gen_core::RgbaImage {
                    width: out_width,
                    height: out_height,
                    pixels: resize_lanczos_rgba_u8(
                        &image.pixels,
                        height as usize,
                        width as usize,
                        out_height as usize,
                        out_width as usize,
                    )?
                    .into_iter()
                    .map(|v| v as u8)
                    .collect(),
                })
            }
        };
        width = out_width;
        height = out_height;
    }
    // Script floor8 followed by Flux2ImageProcessor's default floor16 is floor16.
    let target_width = width / 16 * 16;
    let target_height = height / 16 * 16;
    if target_width == 0 || target_height == 0 {
        return Err(invalid(
            "conditioning image must retain at least a 16x16 source processor grid",
        ));
    }
    let image = rgb(decoded)?;
    let mut image = crop_rgb(
        &image,
        (width - target_width) / 2,
        (height - target_height) / 2,
        target_width,
        target_height,
        false,
    )?;
    let values = image
        .pixels
        .iter()
        .map(|p| 2.0 * (*p as f32 / 255.0) - 1.0)
        .collect::<Vec<_>>();
    let inverse = values.iter().any(|value| *value < 0.0);
    for (pixel, value) in image.pixels.iter_mut().zip(values) {
        let unit = if inverse { (value + 1.0) / 2.0 } else { value };
        *pixel = (unit.clamp(0.0, 1.0) * 255.0) as u8;
    }
    Ok(image)
}

pub fn prepare_pair(
    target: DecodedReference,
    reference: Option<DecodedReference>,
    options: PixelOptions,
    item_index: u64,
) -> Result<PreparedPair> {
    if options.width == 0
        || options.height == 0
        || !options.width.is_multiple_of(16)
        || !options.height.is_multiple_of(16)
    {
        return Err(invalid(
            "training bucket sides must be positive multiples of16",
        ));
    }
    let target = rgb(target)?;
    let reference = reference.map(conditioning_prepass).transpose()?;
    let reference_prepass = reference.as_ref().map(|image| (image.width, image.height));
    let scale = (options.height as f64 / target.height as f64)
        .max(options.width as f64 / target.width as f64);
    let resized_width = (target.width as f64 * scale).round_ties_even() as u32;
    let resized_height = (target.height as f64 * scale).round_ties_even() as u32;
    if resized_width < options.width || resized_height < options.height {
        return Err(invalid("source resize cannot cover the requested bucket"));
    }
    let (crop_x, crop_y) = if options.center_crop {
        (
            ((resized_width - options.width) as f64 / 2.0).round_ties_even() as u32,
            ((resized_height - options.height) as f64 / 2.0).round_ties_even() as u32,
        )
    } else {
        let mut rng = NativeRng::keyed(options.seed, "cache-random-crop", item_index);
        let y = rng.bounded((resized_height - options.height) as usize + 1)? as u32;
        let x = rng.bounded((resized_width - options.width) as usize + 1)? as u32;
        (x, y)
    };
    let flip = options.random_flip
        && NativeRng::keyed(options.seed, "cache-horizontal-flip", item_index).uniform() < 0.5;
    let target = resize_rgb(&target, resized_width, resized_height)?;
    let target = crop_rgb(&target, crop_x, crop_y, options.width, options.height, flip)?;
    let reference = reference
        .map(|image| -> Result<Image> {
            // The frozen pair transform warps BOTH to the TARGET-derived resized size,
            // then applies the SAME crop coordinates and flip, not inference letterboxing.
            crop_rgb(
                &resize_rgb(&image, resized_width, resized_height)?,
                crop_x,
                crop_y,
                options.width,
                options.height,
                flip,
            )
        })
        .transpose()?;
    Ok(PreparedPair {
        target,
        reference,
        geometry: Geometry {
            resized_width,
            resized_height,
            crop_x,
            crop_y,
            flip,
            reference_prepass,
        },
    })
}

pub fn decode_pair(
    target: &std::path::Path,
    reference: Option<&std::path::Path>,
    options: PixelOptions,
    item_index: u64,
) -> Result<PreparedPair> {
    prepare_pair(
        decode_reference_file(target)?,
        reference.map(decode_reference_file).transpose()?,
        options,
        item_index,
    )
}
/// Torch TF.to_tensor then normalize([0.5],[0.5]); keep F32 operation order.
pub fn normalized_hwc(image: &Image) -> Result<Vec<f32>> {
    validate(image.width, image.height, image.pixels.len(), 3)?;
    Ok(image
        .pixels
        .iter()
        .map(|p| ((*p as f32 / 255.0) - 0.5) / 0.5)
        .collect())
}
