//! **Keyframe geometry and pixel preparation** for `fl2va` — resolving the canvas a keyframe
//! implies, fitting the keyframes onto it, and normalizing them for the video VAE. The candle
//! sibling of `mlx_gen_minimax_h3::keyframe` (sc-17148 on the MLX lane, ported here under sc-17156).
//!
//! # A keyframe BINDS the canvas; a reference does not
//!
//! This is the fact that most distinguishes `fl2va` from `ref2va`. With a keyframe and no explicit
//! size, `height`/`width` come from **the first keyframe's aspect ratio** via
//! [`resolve_canvas_size`]; with no keyframe at all the model falls back to 16:9. A `ref2va`
//! reference image does *not* bind geometry and always defaults to 16:9. So the same image passed
//! as a keyframe and as a reference produces two different canvases, and there is no
//! `aspect_ratio` parameter anywhere to reconcile them with.
//!
//! # The first keyframe is STRETCHED; a second one is COVER-CROPPED
//!
//! [`KeyframeFit`] is not a style choice, it is positional. The geometry anchor — `keyframes[0]`,
//! which is `last_image` when only a last frame was passed — is resized straight onto the canvas
//! with **no aspect preservation**: the reference calls PIL's plain `resize((W, H), LANCZOS)`.
//! Letterboxing it instead would put bars into the conditioning signal.
//!
//! Every *later* keyframe is cover-cropped, because the canvas is already fixed by then and
//! stretching a second image with a different aspect would distort the two anchors inconsistently.
//!
//! # Rounding is banker's rounding
//!
//! [`resolve_canvas_size`] and the cover-crop arithmetic both go through Python's `round`, which is
//! **half-to-even**. Rust's `f64::round` is half-away-from-zero, and the two disagree on exactly
//! the inputs a 32-multiple canvas keeps landing on. [`round_half_to_even`] is the reference's.
//!
//! # What is NOT bit-exact, and where the parity boundary is drawn
//!
//! The resampling filter is the one place this port cannot claim exactness. The reference uses
//! **PIL's** Lanczos; this crate uses the `image` crate's `Lanczos3`. Both are a=3 windowed-sinc
//! with support scaled on downscale, and they agree closely, but they are not the same code and
//! nothing here claims they produce identical bytes.
//!
//! The parity boundary is therefore drawn **after** the resize: `tests/video_vae_encode_parity.rs`
//! gates the VAE encode on already-sized pixels, and the tests in this module gate the *policy* —
//! which canvas is chosen, which keyframe is stretched, which is cropped, and how the pixels are
//! normalized. Conflating the two would let a resampling difference of a few LSBs read as a
//! conditioning defect, or worse, hide one.
//!
//! # candle vs MLX: one divergence, and it is not the tensor layout
//!
//! Everything above is backend-neutral integer/byte arithmetic and is a line-for-line port. The
//! only signature difference is [`keyframe_to_vae_pixels`]'s explicit `device`: candle places a
//! tensor at construction where mlx has an implicit default device. That is the same shape every
//! other candle pixel-preprocessor in this workspace takes — `candle-gen-sdxl`'s
//! `preprocess_clip_image_sized`, for instance.
//!
//! The **layout does not diverge**, which is worth stating because the sibling module docs make it
//! look as though it should: [`crate::vae_encoder`] runs its stack in NCTHW where the MLX lane runs
//! NDHWC. That difference is *internal* to the encoder — the MLX `VideoEncoder3d::forward`
//! transposes on entry and back on exit, so both lanes' **public** encode boundary is NCTHW and
//! both lanes' `keyframe_to_vae_pixels` returns the identical `[1, 3, 1, H, W]`. See
//! [`keyframe_to_vae_pixels`].

use image::imageops::FilterType;
use image::RgbImage;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, Result};

use crate::dit::positions::KeyframeAnchor;
use crate::pipeline::{
    CANVAS_MAX_PIXELS, CANVAS_SHORT_EDGE, PIXEL_MEAN, PIXEL_STD, SPATIAL_STRIDE,
};

/// Narrowest aspect ratio MiniMax-H3 accepts — 1:4.
pub const MIN_ASPECT_RATIO: f64 = 1.0 / 4.0;

/// Widest aspect ratio MiniMax-H3 accepts — 4:1.
pub const MAX_ASPECT_RATIO: f64 = 4.0;

/// Python's `round`: half-to-even, **not** half-away-from-zero.
///
/// `round(2.5)` is `2`, not `3`. Rust's `f64::round` is the other convention, and the difference
/// bites exactly where a canvas divides evenly into half a stride — which, on a 32-multiple
/// lattice, is a case the arithmetic reaches constantly rather than rarely.
pub fn round_half_to_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Resolve the canvas a keyframe of ratio `aspect_width : aspect_height` implies.
///
/// Returns `(height, width)`, matching the reference's return order. The algorithm:
///
/// 1. reject a ratio outside `1:4 … 4:1` — **rejected, not clamped**;
/// 2. lay the short edge at `short_edge` and derive the long one from the ratio;
/// 3. if the area exceeds `max_pixels`, scale both edges by `sqrt(max_pixels / area)`;
/// 4. round **each axis independently** to a multiple of `canvas_multiple`, floored at one
///    multiple.
///
/// Step 4 happens *after* the area cap, so the final area can land slightly **above** the
/// pre-rounding budget and the delivered ratio drifts from the requested one — 4:1 resolves to
/// 512x2016, which is 3.938:1. That is the reference's behaviour and not a rounding bug to fix.
pub fn resolve_canvas_size(
    aspect_width: f64,
    aspect_height: f64,
    canvas_multiple: i32,
    short_edge: i32,
    max_pixels: i64,
) -> Result<(i32, i32)> {
    if aspect_width <= 0.0
        || aspect_height <= 0.0
        || !aspect_width.is_finite()
        || !aspect_height.is_finite()
    {
        return Err(CandleError::Msg(format!(
            "minimax-h3 canvas: the aspect ratio must be positive, got \
             {aspect_width}:{aspect_height}"
        )));
    }
    if canvas_multiple <= 0 || short_edge <= 0 || max_pixels <= 0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 canvas: multiple/short_edge/max_pixels must be positive, got \
             {canvas_multiple}/{short_edge}/{max_pixels}"
        )));
    }
    let ratio = aspect_width / aspect_height;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 canvas: MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got \
             {aspect_width}:{aspect_height} ({ratio})"
        )));
    }
    let short = f64::from(short_edge);
    let (mut width, mut height) = if ratio >= 1.0 {
        (short * ratio, short)
    } else {
        (short, short / ratio)
    };
    let area = width * height;
    let budget = max_pixels as f64;
    if area > budget {
        let scale = (budget / area).sqrt();
        width *= scale;
        height *= scale;
    }
    let m = f64::from(canvas_multiple);
    let snap = |v: f64| -> i32 { (round_half_to_even(v / m) * m).max(m) as i32 };
    Ok((snap(height), snap(width)))
}

/// [`resolve_canvas_size`] at the shipped MiniMax-H3 constants — stride 32, short edge 768, area
/// capped at `768 · 1344`.
pub fn resolve_keyframe_canvas(width: u32, height: u32) -> Result<(u32, u32)> {
    let (h, w) = resolve_canvas_size(
        f64::from(width),
        f64::from(height),
        SPATIAL_STRIDE as i32,
        CANVAS_SHORT_EDGE as i32,
        CANVAS_MAX_PIXELS as i64,
    )?;
    Ok((w as u32, h as u32))
}

/// How one keyframe is fitted onto an already-resolved canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeFit {
    /// **The geometry anchor** — resized straight onto the canvas with no aspect preservation.
    /// The canvas came from this image's own ratio, so the stretch is usually a no-op or close to
    /// it; it is a genuine distortion only when the caller supplied an explicit size.
    Stretch,
    /// A follower — scaled to cover the canvas, then centre-cropped. Used for every keyframe after
    /// the first, whose aspect the canvas was not derived from.
    CoverCrop,
}

impl KeyframeFit {
    /// The fit for the keyframe at `index` in packed order. Index 0 stretches; the rest crop.
    pub fn for_index(index: usize) -> Self {
        if index == 0 {
            Self::Stretch
        } else {
            Self::CoverCrop
        }
    }
}

/// Which anchors a request supplies, in the packed order the reference builds them in.
///
/// The reference filters a fixed `(("first", image), ("last", last_image))` pair by presence, so
/// only four shapes exist and `last`-only is one of them — **not** a special case bolted onto a
/// first-frame path. See [`KeyframeAnchor`].
pub fn anchors_for(has_first: bool, has_last: bool) -> Vec<KeyframeAnchor> {
    let mut v = Vec::with_capacity(2);
    if has_first {
        v.push(KeyframeAnchor::First);
    }
    if has_last {
        v.push(KeyframeAnchor::Last);
    }
    v
}

/// Fit one keyframe onto a `width x height` canvas.
///
/// See the module docs for the filter caveat: this is the `image` crate's `Lanczos3`, not PIL's.
pub fn fit_keyframe(image: &Image, width: u32, height: u32, fit: KeyframeFit) -> Result<Image> {
    if width == 0 || height == 0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 keyframe: canvas must be positive, got {width}x{height}"
        )));
    }
    if image.width == 0 || image.height == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 keyframe: a keyframe must have a positive extent".into(),
        ));
    }
    let expected = image.width as usize * image.height as usize * 3;
    if image.pixels.len() != expected {
        return Err(CandleError::Msg(format!(
            "minimax-h3 keyframe: a {}x{} RGB8 image is {expected} bytes, got {}",
            image.width,
            image.height,
            image.pixels.len()
        )));
    }
    if image.width == width && image.height == height {
        return Ok(image.clone());
    }
    let pixels = image.pixels.clone();
    let rgb = RgbImage::from_raw(image.width, image.height, pixels).ok_or_else(|| {
        CandleError::Msg("minimax-h3 keyframe: the pixel buffer is not a valid RGB8 image".into())
    })?;

    let fitted = match fit {
        KeyframeFit::Stretch => image::imageops::resize(&rgb, width, height, FilterType::Lanczos3),
        KeyframeFit::CoverCrop => {
            // MiniMax's own arithmetic: `round()` sizing (not floor) and `(src - dst) // 2`
            // centring (not `dst//2 - src//2`). `VaeImageProcessor`'s own `resize_mode="crop"`
            // disagrees with it on roughly half of sampled aspect ratios, so it must not be
            // substituted.
            let scale = (f64::from(width) / f64::from(image.width))
                .max(f64::from(height) / f64::from(image.height));
            let rw = (round_half_to_even(f64::from(image.width) * scale) as u32).max(width);
            let rh = (round_half_to_even(f64::from(image.height) * scale) as u32).max(height);
            let resized = image::imageops::resize(&rgb, rw, rh, FilterType::Lanczos3);
            let left = rw.saturating_sub(width) / 2;
            let top = rh.saturating_sub(height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
    };
    Ok(Image {
        width,
        height,
        pixels: fitted.into_raw(),
    })
}

/// Fit every keyframe onto the canvas in packed order — index 0 stretched, the rest cover-cropped.
pub fn fit_keyframes(images: &[&Image], width: u32, height: u32) -> Result<Vec<Image>> {
    images
        .iter()
        .enumerate()
        .map(|(i, img)| fit_keyframe(img, width, height, KeyframeFit::for_index(i)))
        .collect()
}

/// Normalize a canvas-sized keyframe into the video VAE's pixel space: `x/255`, then per-channel
/// **ImageNet** mean/std. Returns NCTHW `[1, 3, 1, H, W]` f32 on `device` — the single-frame shape
/// [`crate::vae::MiniMaxH3VideoVae::encode`] short-circuits on.
///
/// # The layout is NOT a candle/MLX divergence
///
/// Worth stating explicitly, because [`crate::vae_encoder`]'s module docs make it look as though it
/// should be: that stack runs NCTHW here and NDHWC on MLX. The transpose is **internal** — the MLX
/// `VideoEncoder3d::forward` converts on entry and back on exit — so *both* lanes document their
/// `forward` and their `MiniMaxH3VideoVae::encode` as `[B, 3, T, H, W]`, and `mlx-gen-minimax-h3`'s
/// `keyframe_to_vae_pixels` returns the same `[1, 3, 1, H, W]` this one does. Emitting NTHWC here to
/// "match candle's native conv layout" would be wrong twice over: it is not this crate's boundary
/// convention, and it would silently reinterpret H as T at a 5-D shape check that cannot tell the
/// two apart.
///
/// The dtype is **f32**, which every encode entry point (`encode`, `encode_clip_untiled`,
/// `encode_clip_tiled_2d`) re-casts to the VAE's own `dtype` on the way in, so this does not have to
/// know what the weights were loaded at.
///
/// # Two deliberate differences from the MLX sibling
///
/// * the explicit `device` — candle places a tensor at construction where mlx has an implicit
///   default device. Same shape as every other candle pixel-preprocessor in the workspace.
/// * the normalization runs **on the host**, with the HWC → CHW permutation folded into the write
///   index, so the result reaches `device` as one contiguous upload rather than an upload followed
///   by a `permute` + `contiguous` + three elementwise kernels. The arithmetic is identical: a true
///   f32 divide by 255 per element, not an `affine` by the (inexact) reciprocal that `Tensor / f64`
///   would compile to.
///
/// Note this is a *different* normalization from the one the Qwen3-VL vision tower applies to the
/// same image (`0.5/0.5` onto `[-1, 1]`, after its own bicubic resize to 16-px patches). A keyframe
/// really is preprocessed twice, two different ways, for the two paths it feeds.
pub fn keyframe_to_vae_pixels(image: &Image, device: &Device) -> Result<Tensor> {
    let (w, h) = (image.width as usize, image.height as usize);
    let expected = w * h * 3;
    if image.pixels.len() != expected {
        return Err(CandleError::Msg(format!(
            "minimax-h3 keyframe: a {w}x{h} RGB8 image is {expected} bytes, got {}",
            image.pixels.len()
        )));
    }
    if w == 0 || h == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 keyframe: cannot normalize an empty image".into(),
        ));
    }
    // HWC -> CHW in the write index; the singleton batch and temporal axes are the constructed
    // shape, so no reshape is needed afterwards.
    let plane = h * w;
    let mut chw = vec![0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = f32::from(image.pixels[(y * w + x) * 3 + c]) / 255.0;
                chw[c * plane + y * w + x] = (v - PIXEL_MEAN[c]) / PIXEL_STD[c];
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, 1, h, w), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::DType;

    fn solid(width: u32, height: u32, rgb: [u8; 3]) -> Image {
        Image {
            width,
            height,
            pixels: (0..width as usize * height as usize)
                .flat_map(|_| rgb)
                .collect(),
        }
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// **The canvas table, measured off the running reference.** Every row is a
    /// `resolve_canvas_size` result, not a derivation.
    #[test]
    fn the_canvas_table_matches_the_reference() {
        let canvas = |w: f64, h: f64| {
            resolve_canvas_size(
                w,
                h,
                SPATIAL_STRIDE as i32,
                CANVAS_SHORT_EDGE as i32,
                CANVAS_MAX_PIXELS as i64,
            )
            .unwrap()
        };
        // (height, width)
        assert_eq!(canvas(16.0, 9.0), (768, 1344));
        assert_eq!(canvas(9.0, 16.0), (1344, 768));
        assert_eq!(canvas(1.0, 1.0), (768, 768));
        assert_eq!(canvas(1920.0, 1080.0), (768, 1344));
        assert_eq!(canvas(1024.0, 768.0), (768, 1024));
        assert_eq!(canvas(4.0, 1.0), (512, 2016));

        // Every result is on the stride-32 lattice, in BOTH axes.
        let stride = SPATIAL_STRIDE as i32;
        for (w, h) in [
            (16.0, 9.0),
            (9.0, 16.0),
            (1.0, 1.0),
            (4.0, 1.0),
            (1.0, 4.0),
            (832.0, 1216.0),
            (3.0, 2.0),
            (21.0, 9.0),
        ] {
            let (ch, cw) = canvas(w, h);
            assert_eq!(ch % stride, 0, "{w}:{h} height {ch}");
            assert_eq!(cw % stride, 0, "{w}:{h} width {cw}");
            assert!(ch > 0 && cw > 0);
        }
    }

    /// Ratios outside 1:4 … 4:1 are **rejected**, not clamped — the same posture
    /// [`crate::pipeline::resolve_geometry`] takes for frame counts.
    #[test]
    fn out_of_range_ratios_are_rejected_not_clamped() {
        let canvas = |w: f64, h: f64| {
            resolve_canvas_size(
                w,
                h,
                SPATIAL_STRIDE as i32,
                CANVAS_SHORT_EDGE as i32,
                CANVAS_MAX_PIXELS as i64,
            )
        };
        for (w, h) in [(5.0, 1.0), (1.0, 5.0), (100.0, 1.0), (1.0, 100.0)] {
            let e = canvas(w, h).unwrap_err().to_string();
            assert!(
                e.contains("aspect ratio"),
                "{w}:{h} must be rejected, got: {e}"
            );
        }
        // The boundary ratios themselves are legal.
        canvas(4.0, 1.0).unwrap();
        canvas(1.0, 4.0).unwrap();
        // Degenerate inputs are typed errors.
        assert!(canvas(0.0, 1.0).is_err());
        assert!(canvas(1.0, 0.0).is_err());
        assert!(canvas(-1.0, 1.0).is_err());
    }

    /// The area cap is applied **before** rounding, so the delivered ratio drifts and the final
    /// area can exceed the pre-rounding budget. Pinned because it looks like a bug.
    #[test]
    fn rounding_happens_after_the_area_cap() {
        let (h, w) = resolve_canvas_size(
            4.0,
            1.0,
            SPATIAL_STRIDE as i32,
            CANVAS_SHORT_EDGE as i32,
            CANVAS_MAX_PIXELS as i64,
        )
        .unwrap();
        assert_eq!((h, w), (512, 2016));
        let delivered = f64::from(w) / f64::from(h);
        assert!(
            (delivered - 3.9375).abs() < 1e-9,
            "4:1 requested delivers {delivered}, not 4.0"
        );
        let budget = CANVAS_MAX_PIXELS as i64;
        assert!(
            i64::from(w) * i64::from(h) <= budget,
            "this one happens to land on the budget exactly"
        );
    }

    /// Python's `round` is half-to-even and Rust's is not.
    #[test]
    fn rounding_is_bankers() {
        assert_eq!(round_half_to_even(0.5), 0.0);
        assert_eq!(round_half_to_even(1.5), 2.0);
        assert_eq!(round_half_to_even(2.5), 2.0);
        assert_eq!(round_half_to_even(3.5), 4.0);
        assert_eq!(round_half_to_even(-0.5), 0.0);
        assert_eq!(round_half_to_even(-1.5), -2.0);
        assert_eq!(round_half_to_even(-2.5), -2.0);
        // Non-halves are unaffected, so this cannot silently change ordinary arithmetic.
        for x in [0.4, 0.6, 1.2, -1.2, 7.49, 7.51] {
            assert_eq!(round_half_to_even(x), x.round(), "{x}");
        }
        // …and it really differs from `f64::round` on the halves.
        assert_ne!(round_half_to_even(2.5), 2.5f64.round());
    }

    /// All four conditioning shapes come out of one ordered filter, and `last`-only is one of
    /// them rather than a special case.
    #[test]
    fn all_four_anchor_shapes_come_from_one_filter() {
        assert_eq!(anchors_for(false, false), vec![]);
        assert_eq!(anchors_for(true, false), vec![KeyframeAnchor::First]);
        assert_eq!(anchors_for(false, true), vec![KeyframeAnchor::Last]);
        assert_eq!(
            anchors_for(true, true),
            vec![KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        // The order is packed order: first always precedes last.
        let both = anchors_for(true, true);
        assert_eq!(both[0], KeyframeAnchor::First);
        assert_eq!(both[1], KeyframeAnchor::Last);
    }

    /// Index 0 stretches, later ones cover-crop — and the two really do differ on a
    /// non-matching aspect.
    #[test]
    fn the_first_keyframe_is_stretched_and_the_rest_are_cropped() {
        assert_eq!(KeyframeFit::for_index(0), KeyframeFit::Stretch);
        assert_eq!(KeyframeFit::for_index(1), KeyframeFit::CoverCrop);
        assert_eq!(KeyframeFit::for_index(7), KeyframeFit::CoverCrop);

        // A 2:1 source onto a 1:1 canvas: stretching squashes the whole width in, cropping keeps
        // the centre. Build a source whose left and right halves differ so the two are
        // distinguishable.
        let mut src = solid(64, 32, [0, 0, 0]);
        for y in 0..32usize {
            for x in 0..64usize {
                let i = (y * 64 + x) * 3;
                let v = if x < 32 { 0u8 } else { 255u8 };
                src.pixels[i] = v;
                src.pixels[i + 1] = v;
                src.pixels[i + 2] = v;
            }
        }
        let stretched = fit_keyframe(&src, 32, 32, KeyframeFit::Stretch).unwrap();
        let cropped = fit_keyframe(&src, 32, 32, KeyframeFit::CoverCrop).unwrap();
        assert_eq!(stretched.width, 32);
        assert_eq!(stretched.height, 32);
        assert_eq!(cropped.width, 32);
        assert_eq!(cropped.height, 32);
        assert_ne!(
            stretched.pixels, cropped.pixels,
            "stretch and cover-crop must not coincide on a 2:1 source"
        );
        // The stretch keeps BOTH halves (dark left, bright right); the crop keeps the seam only.
        let row = 16 * 32 * 3;
        assert!(stretched.pixels[row] < 64, "stretched left edge stays dark");
        assert!(
            stretched.pixels[row + 31 * 3] > 192,
            "stretched right edge stays bright"
        );
    }

    /// A keyframe already at canvas size is passed through untouched — no resampling at all.
    #[test]
    fn an_exact_size_keyframe_is_not_resampled() {
        let src = solid(64, 48, [7, 9, 11]);
        for fit in [KeyframeFit::Stretch, KeyframeFit::CoverCrop] {
            let out = fit_keyframe(&src, 64, 48, fit).unwrap();
            assert_eq!(out.pixels, src.pixels, "{fit:?} must be a no-op at size");
        }
        // Malformed inputs are typed errors.
        assert!(fit_keyframe(&src, 0, 48, KeyframeFit::Stretch).is_err());
        let ragged = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 5],
        };
        assert!(fit_keyframe(&ragged, 8, 8, KeyframeFit::Stretch).is_err());
    }

    /// Pixel normalization is `x/255` then ImageNet mean/std, in NCTHW `[1, 3, 1, H, W]` f32 — and
    /// it is **not** the `[-1, 1]` convention the vision tower uses on the same image.
    ///
    /// The shape assertion is the load-bearing one: it is the layout
    /// [`crate::vae_encoder::VideoEncoder3d::forward`] documents on **both** backends (see
    /// [`keyframe_to_vae_pixels`]), so an NTHWC port fails here rather than at a 5-D shape check
    /// that cannot tell H from T.
    #[test]
    fn pixels_are_imagenet_normalized_into_a_single_frame() {
        let img = solid(2, 3, [255, 0, 128]);
        let x = keyframe_to_vae_pixels(&img, &Device::Cpu).unwrap();
        assert_eq!(x.dims(), &[1, 3, 1, 3, 2], "[B, C, T=1, H, W]");
        assert_eq!(x.dtype(), DType::F32);
        let v = flat(&x);
        // Channel-major: 6 red entries, then 6 green, then 6 blue.
        let want_r = (1.0 - PIXEL_MEAN[0]) / PIXEL_STD[0];
        let want_g = (0.0 - PIXEL_MEAN[1]) / PIXEL_STD[1];
        let want_b = (128.0 / 255.0 - PIXEL_MEAN[2]) / PIXEL_STD[2];
        assert!((v[0] - want_r).abs() < 1e-5, "R {} != {want_r}", v[0]);
        assert!((v[6] - want_g).abs() < 1e-5, "G {} != {want_g}", v[6]);
        assert!((v[12] - want_b).abs() < 1e-5, "B {} != {want_b}", v[12]);
        // A `[-1, 1]` normalization would put white at exactly 1.0; ImageNet's does not.
        assert!(
            (want_r - 1.0).abs() > 0.5,
            "ImageNet normalization must not coincide with the 0.5/0.5 one"
        );

        let ragged = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 3],
        };
        assert!(keyframe_to_vae_pixels(&ragged, &Device::Cpu).is_err());
    }

    /// The canvas really is derived from the keyframe: two keyframes of different aspect give
    /// two different canvases, which is what "a keyframe binds the canvas" means operationally.
    #[test]
    fn the_keyframe_binds_the_canvas() {
        let landscape = resolve_keyframe_canvas(1920, 1080).unwrap();
        let portrait = resolve_keyframe_canvas(1080, 1920).unwrap();
        let square = resolve_keyframe_canvas(512, 512).unwrap();
        assert_eq!(landscape, (1344, 768));
        assert_eq!(portrait, (768, 1344));
        assert_eq!(square, (768, 768));
        assert_ne!(landscape, portrait);
        // …and all three are legal render canvases, i.e. they survive `resolve_geometry`.
        for (w, h) in [landscape, portrait, square] {
            crate::pipeline::resolve_geometry(w, h, crate::pipeline::SMALLEST_LEGAL_FRAMES)
                .unwrap_or_else(|e| panic!("{w}x{h} must be a legal canvas: {e}"));
        }
    }
}
