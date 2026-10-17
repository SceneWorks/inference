//! Source-mode-preserving owned-file input shared by FLUX 3 generation and prompt upsampling.
//!
//! Frozen BFL45e57c9 pipeline.prepare_reference and pup.policy_reference use different RGB/alpha
//! forks. PNG EXPAND's decoded colour type cannot stand in for Pillow's ORIGINAL mode. This
//! helper retains the mode policy while decoding EXIF and normalizing samples, before either
//! consumer resizes/composites. No tensor, generation-request, reference-plan, or model dependency.
use crate::{Error, Image, Result, RgbaImage};

/// Pixel-equivalent normalized input. RGBA retains hidden RGB and alpha; RGB means the frozen
/// original mode has no alpha-composite policy (including RGB/L PNG tRNS and grayscale PNG16).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodedReference {
    Rgb(Image),
    Rgba(RgbaImage),
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Unsupported(format!("flux3 image input: {}", message.into()))
}

/// Decode only a caller-provisioned/authorized local asset. This path is NOT an untrusted API
/// request carrier. Apply EXIF once, preserve alpha until the AE/PUP fork, and follow frozen
/// Pillow's common PNG8/PNG16 conversion rules rather than generic normalized-depth conversion.
pub fn decode_reference_file(path: &std::path::Path) -> Result<DecodedReference> {
    use image::ImageDecoder;
    use std::io::{Read, Seek};
    let mut file = std::fs::File::open(path).map_err(|e| {
        invalid(format!(
            "cannot open owned reference {}: {e}",
            path.display()
        ))
    })?;
    // PNG EXPAND changes RGB/L+tRNS into RGBA/LA. Pillow keeps their original RGB/L mode,
    // whose transparency the frozen PUP policy intentionally ignores. Keep IHDR from the SAME
    // file handle, not a second path read; decoded colour_type alone cannot recover this intent.
    let mut header = Vec::with_capacity(26);
    file.by_ref()
        .take(26)
        .read_to_end(&mut header)
        .map_err(|e| {
            invalid(format!(
                "cannot read reference header {}: {e}",
                path.display()
            ))
        })?;
    file.rewind().map_err(|e| {
        invalid(format!(
            "cannot rewind owned reference {}: {e}",
            path.display()
        ))
    })?;
    let png = if header.len() == 26
        && header.starts_with(b"\x89PNG\r\n\x1a\n")
        && &header[12..16] == b"IHDR"
    {
        Some((header[24], header[25]))
    } else {
        None
    };
    let reader = image::ImageReader::new(std::io::BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| {
            invalid(format!(
                "cannot identify owned reference {}: {e}",
                path.display()
            ))
        })?;
    let mut decoder = reader.into_decoder().map_err(|e| {
        invalid(format!(
            "cannot decode owned reference {}: {e}",
            path.display()
        ))
    })?;
    let orientation = decoder.orientation().map_err(|e| {
        invalid(format!(
            "invalid reference orientation {}: {e}",
            path.display()
        ))
    })?;
    let mut decoded = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| invalid(format!("invalid reference image {}: {e}", path.display())))?;
    decoded.apply_orientation(orientation);
    let (width, height) = (decoded.width(), decoded.height());
    let rgb = |pixels| {
        DecodedReference::Rgb(Image {
            width,
            height,
            pixels,
        })
    };
    let rgba = |pixels| {
        DecodedReference::Rgba(RgbaImage {
            width,
            height,
            pixels,
        })
    };
    match png {
        // Pillow opens grayscale PNG16 as I;16: convert("RGB") saturates integer values at 255,
        // it does NOT rescale 0..65535. tRNS does not turn that original mode into alpha.
        Some((16, 0)) => {
            return Ok(rgb(decoded
                .to_luma16()
                .into_raw()
                .into_iter()
                .flat_map(|v| [v.min(255) as u8; 3])
                .collect()))
        }
        // Pillow's colour PNG16 decoder exposes the high byte in RGB/RGBA (LA16 opens RGBA).
        // Explicit shifts avoid imagecrate's generic normalized-channel conversion/rounding.
        Some((16, 2)) => {
            return Ok(rgb(decoded
                .to_rgb16()
                .into_raw()
                .into_iter()
                .map(|v| (v >> 8) as u8)
                .collect()))
        }
        Some((16, 4 | 6)) => {
            return Ok(rgba(
                decoded
                    .to_rgba16()
                    .into_raw()
                    .into_iter()
                    .map(|v| (v >> 8) as u8)
                    .collect(),
            ))
        }
        _ => {}
    }
    if !matches!(
        decoded.color(),
        image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::Rgb8
            | image::ColorType::Rgba8
    ) {
        return Err(invalid(format!("reference {} uses unsupported {:?} pixels; export an 8-bit RGB/RGBA image or a supported PNG",path.display(),decoded.color())));
    }
    let keep_alpha = decoded.color().has_alpha() && !matches!(png, Some((_, 0 | 2)));
    Ok(if keep_alpha {
        rgba(decoded.to_rgba8().into_raw())
    } else {
        rgb(decoded.to_rgb8().into_raw())
    })
}
