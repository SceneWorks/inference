//! OpenEXR read/write for the HDR lane (sc-18790).
//!
//! The byte-level counterpart of [`crate::hdr`]: that module owns the colour transforms, this one
//! moves [`HdrFrame`]s to and from `.exr` bytes with the header attributes that make the pixels
//! *interpretable* — `chromaticities` (which primaries the RGB triples are in) and `colorSpace`
//! (which transfer encoding they carry).
//!
//! Those two attributes are the whole reason this is a hand-written wrapper rather than a
//! one-line call to the crate's `write_rgb_file` convenience: an untagged scene-linear EXR is
//! indistinguishable from an ACEScct log EXR by inspection, and a compositor that guesses wrong
//! produces a silently wrong grade. Upstream (`ltx_pipelines…media_io.exr.save_exr_tensor`) tags
//! both, and so do we.
//!
//! # Interop
//!
//! Reading accepts any single-part scanline **or tiled** EXR the `exr` crate supports (ZIP, PIZ,
//! RLE, DWA, …) and any sample type, because conditioning plates come from third-party tools.
//! Alpha and extra channels are dropped; a single-channel (luminance) EXR is broadcast to three,
//! matching upstream's reader. Values are returned **unmodified** — unbounded scene-linear, so
//! highlights above `1.0` survive.

use std::io::Cursor;

use exr::meta::attribute::{AttributeValue, Chromaticities, Text};
use exr::prelude::*;

use crate::hdr::Primaries;
use crate::media::HdrFrame;
use crate::Error;

/// The EXR header attribute naming the transfer encoding (e.g. `"sRGB"`, `"ACEScg"`,
/// `"ACEScct"`). Not a standard OpenEXR attribute, but the one upstream writes and the one
/// downstream tooling reads to tell scene-linear from log.
pub const EXR_COLOR_SPACE_ATTRIBUTE: &str = "colorSpace";

/// An EXR frame plus the header attributes that say how to interpret its samples.
///
/// The tags are `Option` because a third-party file may carry neither; a reader that silently
/// substituted a default would be asserting a colour space it does not know.
#[derive(Clone, Debug, PartialEq)]
pub struct ExrImage {
    /// The pixels, unmodified (unbounded).
    pub frame: HdrFrame,
    /// `chromaticities`, in EXR order: `R.x R.y G.x G.y B.x B.y W.x W.y`.
    pub chromaticities: Option<[f32; 8]>,
    /// The [`EXR_COLOR_SPACE_ATTRIBUTE`] value, when present.
    pub color_space_tag: Option<String>,
}

fn exr_err(context: &str, e: impl std::fmt::Display) -> Error {
    Error::Msg(format!("{context}: {e}"))
}

/// Encode one [`HdrFrame`] as OpenEXR bytes, tagged with `primaries` and `color_space_tag`.
///
/// `half` selects the sample type: `true` writes 16-bit float (upstream's default — half the
/// bytes, and ~3 decimal digits is plenty for scene-linear light), `false` writes 32-bit float
/// for a lossless round-trip. Compression is ZIP (16-scanline blocks), matching upstream's
/// `spec.attribute("compression", "zip")`.
///
/// Note the precision consequence of `half = true`: a value only round-trips to `f16` precision,
/// so an exact write→read equality check must pass `half = false`.
pub fn write_rgb_exr(
    frame: &HdrFrame,
    primaries: Primaries,
    color_space_tag: &str,
    half: bool,
) -> crate::Result<Vec<u8>> {
    frame.validate()?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let rgb = &frame.rgb;

    let c = primaries.exr_chromaticities();
    let chromaticities = Chromaticities {
        red: Vec2(c[0], c[1]),
        green: Vec2(c[2], c[3]),
        blue: Vec2(c[4], c[5]),
        white: Vec2(c[6], c[7]),
    };
    let tag_key = Text::from(EXR_COLOR_SPACE_ATTRIBUTE);
    let tag_value = AttributeValue::Text(Text::from(color_space_tag));

    let encoding = Encoding {
        compression: Compression::ZIP16,
        blocks: Blocks::ScanLines,
        line_order: LineOrder::Increasing,
    };
    let layer_attributes = LayerAttributes::default();

    let mut out = Cursor::new(Vec::<u8>::new());
    // The two arms differ only in sample type, which `SpecificChannels` encodes in the closure's
    // return type — so the generic image type differs and the arms cannot be merged.
    if half {
        let channels = SpecificChannels::rgb(|p: Vec2<usize>| {
            let i = (p.y() * w + p.x()) * 3;
            (
                f16::from_f32(rgb[i]),
                f16::from_f32(rgb[i + 1]),
                f16::from_f32(rgb[i + 2]),
            )
        });
        let mut image = Image::from_layer(Layer::new((w, h), layer_attributes, encoding, channels));
        image.attributes.chromaticities = Some(chromaticities);
        image.attributes.other.insert(tag_key, tag_value);
        image
            .write()
            .to_buffered(&mut out)
            .map_err(|e| exr_err("write_rgb_exr (half)", e))?;
    } else {
        let channels = SpecificChannels::rgb(|p: Vec2<usize>| {
            let i = (p.y() * w + p.x()) * 3;
            (rgb[i], rgb[i + 1], rgb[i + 2])
        });
        let mut image = Image::from_layer(Layer::new((w, h), layer_attributes, encoding, channels));
        image.attributes.chromaticities = Some(chromaticities);
        image.attributes.other.insert(tag_key, tag_value);
        image
            .write()
            .to_buffered(&mut out)
            .map_err(|e| exr_err("write_rgb_exr (float)", e))?;
    }
    Ok(out.into_inner())
}

/// Decode OpenEXR bytes into an [`ExrImage`].
///
/// Reads the largest resolution level of the first valid layer. Alpha is read but discarded
/// (scene-linear conditioning has no use for it); a file with no `B`/`G` channel — a luminance-
/// only render pass — is broadcast to three channels rather than rejected.
pub fn read_rgb_exr(bytes: &[u8]) -> crate::Result<ExrImage> {
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .specific_channels()
        .required("R")
        // A luminance-only EXR has no G/B; default them to the red sample so the frame comes back
        // grey rather than red-tinted (upstream broadcasts a 1-channel file to three).
        .optional("G", f32::NAN)
        .optional("B", f32::NAN)
        .collect_pixels(
            // `set_pixel` receives only the buffer and a position, so carry the row stride
            // alongside the samples rather than trying to recover it from the buffer length.
            |resolution, _| {
                (
                    resolution.width(),
                    vec![(0f32, 0f32, 0f32); resolution.width() * resolution.height()],
                )
            },
            |(width, pixels): &mut (usize, Vec<(f32, f32, f32)>),
             position: Vec2<usize>,
             (r, g, b): (f32, f32, f32)| {
                pixels[position.y() * *width + position.x()] = (r, g, b);
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .map_err(|e| exr_err("read_rgb_exr", e))?;

    let size = image.layer_data.size;
    let (w, h) = (size.width(), size.height());
    let mut out = Vec::with_capacity(w * h * 3);
    for (r, g, b) in image.layer_data.channel_data.pixels.1.iter() {
        // A missing G/B arrives as NaN (the `optional` default) — broadcast red into it.
        out.push(*r);
        out.push(if g.is_nan() { *r } else { *g });
        out.push(if b.is_nan() { *r } else { *b });
    }

    let frame = HdrFrame {
        width: w as u32,
        height: h as u32,
        rgb: out,
    };
    frame.validate()?;

    let chromaticities = image.attributes.chromaticities.map(|c| {
        [
            c.red.x(),
            c.red.y(),
            c.green.x(),
            c.green.y(),
            c.blue.x(),
            c.blue.y(),
            c.white.x(),
            c.white.y(),
        ]
    });
    let color_space_tag = image
        .attributes
        .other
        .get(&Text::from(EXR_COLOR_SPACE_ATTRIBUTE))
        .and_then(|v| match v {
            AttributeValue::Text(t) => Some(t.to_string()),
            _ => None,
        });

    Ok(ExrImage {
        frame,
        chromaticities,
        color_space_tag,
    })
}
