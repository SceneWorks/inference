//! Ordered AE reference preparation from frozen `pipeline.py:129..176` and `generate.py:65..77`.
//! Generation references are NOT Qwen vision inputs and never initialize the noisy output state.
//! PUP has a separate multimodal preprocessing policy; do not reuse its white composite/JPEG.
use super::{DOWNSAMPLING_RATIO, MAX_SIDE, MIN_SIDE};
use crate::{Error, Image, Result};

pub const MAX_REFERENCES: usize = 10;
pub const MATCH_MAX_PIXELS: u64 = 4 * 1024 * 1024;
pub const MULTI_REFERENCE_MAX_PIXELS: u64 = 1024 * 1024;

/// Original optional dimensions, before a consumer inserts its ordinary T2I defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReferenceOutputSize {
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Pixel-coordinate correspondence. Source coordinates refer to the EXIF-oriented RGB input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceGeometry {
    pub source_size: (u32, u32),
    pub fitted_size: (u32, u32),
    /// Kept rectangle in the fitted (resized source) image; no deduplication/reordering.
    pub source_region: [u32; 4],
    /// Placement of that rectangle on the conditioning canvas.
    pub canvas_region: [u32; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputGeometry {
    pub canvas: (u32, u32),
    /// Upstream crops a padded single reference's output only if neither side was specified.
    pub crop: Option<[u32; 4]>,
}

/// Allocation-free geometry/token contract for admission before native components are loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferencePlan {
    pub geometry: Vec<ReferenceGeometry>,
    pub canvases: Vec<(u32, u32)>,
    pub output: OutputGeometry,
    /// Sum of H/16 * W/16 AE tokens; base packs this context in BOTH CFG samples.
    pub conditioning_tokens: u64,
}

pub struct PreparedReferences {
    pub images: Vec<Image>,
    pub geometry: Vec<ReferenceGeometry>,
    pub output: OutputGeometry,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Unsupported(format!("flux3 references: {}", message.into()))
}

/// The fitted size and padded canvas, including the source's iterative padded-area budget.
pub fn reference_canvas(width: u32, height: u32) -> Result<((u32, u32), (u32, u32))> {
    if width == 0 || height == 0 {
        return Err(invalid("empty image dimensions"));
    }
    let mut scale = 1.0_f64
        .min((MATCH_MAX_PIXELS as f64 / (width as f64 * height as f64)).sqrt())
        .min(MAX_SIDE as f64 / width.max(height) as f64);
    loop {
        let fit = (
            (width as f64 * scale).round_ties_even().max(1.0) as u32,
            (height as f64 * scale).round_ties_even().max(1.0) as u32,
        );
        let canvas = (
            fit.0
                .div_ceil(DOWNSAMPLING_RATIO)
                .saturating_mul(DOWNSAMPLING_RATIO)
                .clamp(MIN_SIDE, MAX_SIDE),
            fit.1
                .div_ceil(DOWNSAMPLING_RATIO)
                .saturating_mul(DOWNSAMPLING_RATIO)
                .clamp(MIN_SIDE, MAX_SIDE),
        );
        if canvas.0 as u64 * canvas.1 as u64 <= MATCH_MAX_PIXELS {
            return Ok((fit, canvas));
        }
        scale *= 0.995;
    }
}

fn geometry(width: u32, height: u32, multi: bool) -> Result<(ReferenceGeometry, (u32, u32))> {
    if width == 0 || height == 0 {
        return Err(invalid("empty image dimensions"));
    }
    if multi {
        let fit = if width as u64 * height as u64 > MULTI_REFERENCE_MAX_PIXELS {
            let scale = (MULTI_REFERENCE_MAX_PIXELS as f64 / (width as f64 * height as f64)).sqrt();
            (
                (width as f64 * scale) as u32,
                (height as f64 * scale) as u32,
            )
        } else {
            (width, height)
        };
        let (w, h) = (fit.0 / 16 * 16, fit.1 / 16 * 16);
        if w == 0 || h == 0 {
            return Err(invalid(format!("multi-reference {width}x{height} becomes empty after the upstream 16px crop; use a less extreme aspect ratio and sides >=16")));
        }
        let (left, top) = ((fit.0 - w) / 2, (fit.1 - h) / 2);
        Ok((
            ReferenceGeometry {
                source_size: (width, height),
                fitted_size: fit,
                source_region: [left, top, left + w, top + h],
                canvas_region: [0, 0, w, h],
            },
            (w, h),
        ))
    } else {
        let (fit, canvas) = reference_canvas(width, height)?;
        let (left, top) = ((canvas.0 - fit.0) / 2, (canvas.1 - fit.1) / 2);
        Ok((
            ReferenceGeometry {
                source_size: (width, height),
                fitted_size: fit,
                source_region: [0, 0, fit.0, fit.1],
                canvas_region: [left, top, left + fit.0, top + fit.1],
            },
            canvas,
        ))
    }
}

fn buffer_valid(image: &Image, index: usize) -> Result<()> {
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|n| n.checked_mul(3));
    if image.width == 0 || image.height == 0 || expected != Some(image.pixels.len()) {
        return Err(invalid(format!(
            "reference {} has invalid RGB dimensions/buffer ({}x{}, {} bytes)",
            index + 1,
            image.width,
            image.height,
            image.pixels.len()
        )));
    }
    Ok(())
}

/// Source-derived canvas and reference-context cost. Dimensions are EXIF-oriented, before resize.
pub fn plan(dimensions: &[(u32, u32)], size: ReferenceOutputSize) -> Result<ReferencePlan> {
    if !(1..=MAX_REFERENCES).contains(&dimensions.len()) {
        return Err(invalid(format!(
            "expected 1–{MAX_REFERENCES} ordered references, got {}",
            dimensions.len()
        )));
    }
    let multi = dimensions.len() > 1;
    let plans = dimensions
        .iter()
        .map(|&(w, h)| geometry(w, h, multi))
        .collect::<Result<Vec<_>>>()?;
    let first = plans[0].1;
    let canvas = (
        size.width.unwrap_or(first.0),
        size.height.unwrap_or(first.1),
    );
    if canvas.0 < MIN_SIDE
        || canvas.1 < MIN_SIDE
        || canvas.0 > MAX_SIDE
        || canvas.1 > MAX_SIDE
        || !canvas.0.is_multiple_of(16)
        || !canvas.1.is_multiple_of(16)
    {
        return Err(invalid(format!("output canvas {}x{} must have 64–4096px sides on the 16px grid; specify output dimensions for a smaller multi-reference canvas",canvas.0,canvas.1)));
    }
    let crop = if !multi
        && size.width.is_none()
        && size.height.is_none()
        && plans[0].0.fitted_size != first
    {
        Some(plans[0].0.canvas_region)
    } else {
        None
    };
    let conditioning_tokens = plans
        .iter()
        .map(|(_, (w, h))| (*w as u64 / 16) * (*h as u64 / 16))
        .sum();
    let (geometry, canvases) = plans.into_iter().unzip();
    Ok(ReferencePlan {
        geometry,
        canvases,
        output: OutputGeometry { canvas, crop },
        conditioning_tokens,
    })
}

/// Validate the entire set before allocating/resampling conditioning images. The caller resolves
/// owned assets and EXIF orientation before supplying RGB. PUP intentionally sees the originals.
pub fn prepare(images: &[Image], size: ReferenceOutputSize) -> Result<PreparedReferences> {
    prepare_borrowed(&images.iter().collect::<Vec<_>>(), size)
}

/// Borrowed counterpart avoids cloning all original images when preparing a public request.
pub fn prepare_borrowed(
    images: &[&Image],
    size: ReferenceOutputSize,
) -> Result<PreparedReferences> {
    // Count first, then every buffer and geometry, before performing any resize.
    if !(1..=MAX_REFERENCES).contains(&images.len()) {
        return Err(invalid("expected 1–10 ordered references"));
    }
    for (index, image) in images.iter().enumerate() {
        buffer_valid(image, index)?;
    }
    let plan = plan(
        &images
            .iter()
            .map(|i| (i.width, i.height))
            .collect::<Vec<_>>(),
        size,
    )?;
    let multi = images.len() > 1;
    let mut prepared = Vec::with_capacity(images.len());
    let mut geometries = Vec::with_capacity(images.len());
    for (image, (g, (w, h))) in images
        .iter()
        .zip(plan.geometry.into_iter().zip(plan.canvases))
    {
        let fit = g.fitted_size;
        let resized = if fit == (image.width, image.height) {
            image.pixels.clone()
        } else {
            crate::imageops::resize_lanczos_u8(
                &image.pixels,
                image.height as usize,
                image.width as usize,
                fit.1 as usize,
                fit.0 as usize,
            )?
            .into_iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect()
        };
        let mut pixels = Vec::with_capacity(w as usize * h as usize * 3);
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = if multi {
                    (x + g.source_region[0], y + g.source_region[1])
                } else {
                    let symmetric = fit.0.min(fit.1) <= 16;
                    (
                        pad_index(x as i64 - g.canvas_region[0] as i64, fit.0, symmetric),
                        pad_index(y as i64 - g.canvas_region[1] as i64, fit.1, symmetric),
                    )
                };
                let offset = (sy as usize * fit.0 as usize + sx as usize) * 3;
                pixels.extend_from_slice(&resized[offset..offset + 3]);
            }
        }
        prepared.push(Image {
            width: w,
            height: h,
            pixels,
        });
        geometries.push(g);
    }
    Ok(PreparedReferences {
        images: prepared,
        geometry: geometries,
        output: plan.output,
    })
}

/// Decode a caller-resolved OWNED asset file, applying EXIF once before the AE/PUP fork.
/// This is not an API path carrier: SceneWorks must resolve/authorize asset IDs in its store;
/// never expose this path argument as an untrusted generation-request field.
/// Preserve alpha/hidden RGB until request_images (AE) or upsampler_input (PUP) chooses its policy.
pub fn decode_reference_file(path: &std::path::Path) -> Result<crate::Conditioning> {
    use super::image_input::{decode_reference_file as decode_input, DecodedReference};
    Ok(match decode_input(path)? {
        DecodedReference::Rgb(image) => crate::Conditioning::Reference {
            image,
            strength: None,
        },
        DecodedReference::Rgba(image) => crate::Conditioning::ReferenceRgba {
            image,
            strength: None,
        },
    })
}

enum Input<'a> {
    Rgb(&'a Image),
    Rgba(&'a crate::RgbaImage),
}

impl Input<'_> {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Rgb(i) => (i.width, i.height),
            Self::Rgba(i) => (i.width, i.height),
        }
    }
}

// Preserve original RGBA until the AE/PUP fork. Validation/planning do not allocate image copies.
fn ordered_inputs(req: &crate::GenerationRequest) -> Result<Vec<Input<'_>>> {
    use crate::Conditioning;
    if req.strength.is_some() {
        return Err(invalid(
            "img2img strength is unsupported; remove strength to use reference conditioning",
        ));
    }
    let count: usize = req
        .conditioning
        .iter()
        .map(|c| match c {
            Conditioning::MultiReference { images } => images.len(),
            _ => 1,
        })
        .sum();
    if count > MAX_REFERENCES {
        return Err(invalid(format!(
            "at most {MAX_REFERENCES} ordered references, got {count}"
        )));
    }
    let mut inputs = Vec::with_capacity(count);
    for item in &req.conditioning {
        match item {
            Conditioning::Reference {image,strength:None}=>{buffer_valid(image,inputs.len())?;inputs.push(Input::Rgb(image));},
            Conditioning::ReferenceRgba {image,strength:None}=>{
                let expected=(image.width as usize).checked_mul(image.height as usize).and_then(|n|n.checked_mul(4));
                if image.width==0 || image.height==0 || expected!=Some(image.pixels.len()) {
                    return Err(invalid(format!("reference {} has invalid RGBA dimensions/buffer ({}x{}, {} bytes)",inputs.len()+1,image.width,image.height,image.pixels.len())));
                }
                inputs.push(Input::Rgba(image));
            },
            Conditioning::MultiReference {images} if !images.is_empty()=>{
                for image in images {buffer_valid(image,inputs.len())?;inputs.push(Input::Rgb(image));}
            },
            Conditioning::Reference {strength:Some(_),..}|Conditioning::ReferenceRgba {strength:Some(_),..}=>return Err(invalid("per-reference strength is unsupported; remove it to use reference conditioning")),
            _=>return Err(invalid("only ordered Reference/ReferenceRgba/MultiReference conditioning is supported; remove unsupported conditioning")),
        }
    }
    if inputs.is_empty() && req.flux3_reference_output.is_some() {
        return Err(invalid("reference-output intent requires 1–10 references"));
    }
    Ok(inputs)
}

/// Ordered AE RGB inputs: Pillow `convert("RGB")` drops alpha and preserves hidden RGB, it does
/// NOT premultiply or composite white. RGB buffers stay borrowed; RGBA converts only at this fork.
pub fn request_images(req: &crate::GenerationRequest) -> Result<Vec<std::borrow::Cow<'_, Image>>> {
    use std::borrow::Cow;
    Ok(ordered_inputs(req)?
        .into_iter()
        .map(|input| match input {
            Input::Rgb(image) => Cow::Borrowed(image),
            Input::Rgba(image) => Cow::Owned(Image {
                width: image.width,
                height: image.height,
                pixels: image
                    .pixels
                    .chunks_exact(4)
                    .flat_map(|p| p[..3].iter().copied())
                    .collect(),
            }),
        })
        .collect())
}

/// Validate original RGB/RGBA buffers and derive context cost without converting pixels/GPU work.
/// The consumer must use this before admission, not after accepting a default-size estimate.
pub fn request_plan(req: &crate::GenerationRequest) -> Result<Option<ReferencePlan>> {
    let inputs = ordered_inputs(req)?;
    if inputs.is_empty() {
        return Ok(None);
    }
    plan(
        &inputs.iter().map(Input::dimensions).collect::<Vec<_>>(),
        request_output(req),
    )
    .map(Some)
}

/// Existing PUP consumes ORIGINAL ordered RGB/RGBA, never AE RGB-converted/resized canvases.
/// Its own alpha-over-white/JPEG policy is intentionally separate. Text-only behavior is unchanged.
pub fn upsampler_input(req: &crate::GenerationRequest) -> Result<super::pup::UpsamplerInput> {
    use super::pup::{ReferenceImage, UpsampleMode, UpsamplerInput};
    let inputs = ordered_inputs(req)?;
    let mode = UpsampleMode::infer(inputs.len(), super::pup::detect(&req.prompt).is_some());
    Ok(UpsamplerInput {
        prompt: req.prompt.clone(),
        references: inputs
            .into_iter()
            .map(|input| match input {
                Input::Rgb(image) => ReferenceImage::Rgb8 {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels.clone(),
                },
                Input::Rgba(image) => ReferenceImage::Rgba8 {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels.clone(),
                },
            })
            .collect(),
        boxes: None,
        mode,
        bypass: !req.enhance_prompt,
    })
}

/// Resolve the request's optional intent. Legacy callers with no hint keep explicit dimensions.
pub fn request_output(req: &crate::GenerationRequest) -> ReferenceOutputSize {
    req.flux3_reference_output.unwrap_or(ReferenceOutputSize {
        width: Some(req.width),
        height: Some(req.height),
    })
}

/// Consistent weights-free memory-route guard; edit/image-to-image names both denote immutable
/// reference conditioning, never denoise-strength initialization. Context demand is caller-priced.
pub fn validate_memory_route(mode: &crate::MemoryMode, count: u32) -> Result<()> {
    match mode {
        crate::MemoryMode::TextToImage if count==0 => Ok(()),
        crate::MemoryMode::ImageToImage | crate::MemoryMode::Edit if (1..=MAX_REFERENCES as u32).contains(&count) => Ok(()),
        _ => Err(invalid(format!("mode {} requires zero references for text_to_image or 1–10 for image_to_image/edit; got {count}",mode.as_key()))),
    }
}

/// Frozen `encode_references`: spatial axes scale by 0.5, ordered references occupy negative l.
pub fn conditioning_ids(h: usize, w: usize, index: usize) -> Result<Vec<[f32; 4]>> {
    if index >= MAX_REFERENCES || h == 0 || w == 0 {
        return Err(invalid("invalid reference token grid or ordinal"));
    }
    let mut ids = super::scaled_image_ids(&super::latent_ids(h, w), super::IMAGE_ROPE_SCALE);
    for id in &mut ids {
        id[3] = -((index + 1) as f32);
    }
    Ok(ids)
}

/// Normalize only resolved canvas dimensions, preserving the original typed intent for crop and
/// provenance. GenerationParams::resolve uses the same request_plan, so raw defaults cannot win.
pub fn normalize_request(req: &mut crate::GenerationRequest) -> Result<Option<ReferencePlan>> {
    let plan = request_plan(req)?;
    if let Some(plan) = &plan {
        (req.width, req.height) = plan.output.canvas;
    }
    Ok(plan)
}

/// Guard the family-specific output intent before an existing generic scope sees numeric sizes.
/// This verifies geometry/count, NOT the caller's token/AE peak budget: workers must price actual
/// owned-reference shapes before admission and cache lookup, never a user-forged numeric hint.
pub fn guard_memory_scope<'a>(
    inner: Box<dyn crate::MemoryRequestScope + 'a>,
    admitted: crate::MemoryGeometry,
) -> Box<dyn crate::MemoryRequestScope + 'a> {
    Box::new(ReferenceScope { inner, admitted })
}

struct ReferenceScope<'a> {
    inner: Box<dyn crate::MemoryRequestScope + 'a>,
    admitted: crate::MemoryGeometry,
}

impl crate::MemoryRequestScope for ReferenceScope<'_> {
    fn configure_request(&mut self, req: &mut crate::GenerationRequest) -> Result<()> {
        if let Some(plan) = request_plan(req)? {
            if plan.output.canvas != (self.admitted.width, self.admitted.height)
                || plan.geometry.len() != self.admitted.reference_count as usize
            {
                return Err(invalid(format!("resolved reference canvas {}x{} and {} references do not match admitted {}x{} and {}; resolve owned references before memory admission",plan.output.canvas.0,plan.output.canvas.1,plan.geometry.len(),self.admitted.width,self.admitted.height,self.admitted.reference_count)));
            }
        }
        normalize_request(req)?;
        self.inner.configure_request(req)
    }
    fn enter_phase(&mut self, phase: crate::MemoryPhase) -> Result<()> {
        self.inner.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: crate::MemoryPhase) -> Result<()> {
        self.inner.leave_phase(phase)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: crate::MemoryGeometry,
    ) -> Result<()> {
        self.inner.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, size: u32) -> Result<()> {
        self.inner.configure_attention(size)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> Result<()> {
        self.inner.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: crate::MemoryRunOutcome) -> Result<()> {
        self.inner.finish(outcome)
    }
}

// numpy.pad's repeated reflect/symmetric folding, including pads larger than a tiny input.
fn pad_index(index: i64, length: u32, symmetric: bool) -> u32 {
    if length == 1 {
        return 0;
    }
    let n = length as i64;
    let period = if symmetric { 2 * n } else { 2 * (n - 1) };
    let v = index.rem_euclid(period);
    if v < n {
        v as u32
    } else if symmetric {
        (2 * n - 1 - v) as u32
    } else {
        (2 * n - 2 - v) as u32
    }
}

/// Crop the decoded canvas only after the native renderer has finished, preserving RGB order.
pub fn crop_output(image: Image, geometry: &OutputGeometry) -> Result<Image> {
    buffer_valid(&image, 0)?;
    if (image.width, image.height) != geometry.canvas {
        return Err(invalid(
            "decoded canvas disagrees with planned output geometry",
        ));
    }
    let Some([left, top, right, bottom]) = geometry.crop else {
        return Ok(image);
    };
    if left >= right || top >= bottom || right > image.width || bottom > image.height {
        return Err(invalid("invalid output crop"));
    }
    let mut pixels = Vec::with_capacity((right - left) as usize * (bottom - top) as usize * 3);
    for y in top..bottom {
        let start = (y as usize * image.width as usize + left as usize) * 3;
        pixels.extend_from_slice(&image.pixels[start..start + (right - left) as usize * 3]);
    }
    Ok(Image {
        width: right - left,
        height: bottom - top,
        pixels,
    })
}
