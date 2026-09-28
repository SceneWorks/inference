//! PIL-compatible image resampling and host-side mask/geometry — the shared, model-agnostic image
//! math used by the provider crates' img2img / edit / control preprocessing (e.g. the fork's
//! `scale_to_dimensions` and the Qwen2-VL image processor). Pure host code (no tensors), so it lives
//! in gen-core and every backend reuses one copy.
//!
//! `resize_u8` bit-matches PIL's `ImagingResample` 8-bit path: float filter coefficients
//! quantized to `PRECISION_BITS` fixed-point, an integer multiply-accumulate seeded with the
//! rounding bias, then `clip8` (`>>PRECISION_BITS` + clamp). Reproducing PIL's *fixed-point*
//! arithmetic (not just "a bicubic") is what gives the edit/img2img conditioning images
//! pixel-parity with the frozen Python fork — an f64-coefficient resampler diverges ±1–2 ULP at
//! gradient cliffs (sc-2465: 24% e2e px>8).
//!
//! (The VAE-decoded-tensor → [`Image`] denormalize step, `decoded_to_image`, is **not** here — it
//! operates on a backend tensor and stays in `mlx_gen::image`.)

use crate::media::Image;
use crate::Error;

/// Returns the number of scalar elements in an interleaved image buffer without overflowing.
///
/// Zero-sized images remain representable as a zero-length buffer; validation of whether zero
/// dimensions are permitted belongs to the caller's image contract.
#[inline]
pub fn checked_image_buffer_len(width: usize, height: usize, channels: usize) -> Option<usize> {
    width.checked_mul(height)?.checked_mul(channels)
}

/// PIL `bicubic_filter` (Keys cubic, a = -0.5), support 2.0.
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// PIL `Image.BILINEAR` filter: a triangle of support radius 1.0.
fn triangle(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// Normalized sinc, `sin(πx)/(πx)`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// PIL `lanczos_filter` (a = 3): `sinc(x)·sinc(x/3)`, support 3.0.
fn lanczos3(x: f64) -> f64 {
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Per-output-pixel resampling coefficients for a 1-D axis resize, matching PIL's
/// `precompute_coeffs`: antialias by scaling the filter support when downscaling, clamp the
/// window to the input bounds, and renormalize the (possibly truncated) weights to sum to 1.
/// `support_radius` is the filter's base support (2.0 bicubic, 3.0 lanczos).
fn precompute_coeffs(
    in_size: usize,
    out_size: usize,
    support_radius: f64,
    filter: &dyn Fn(f64) -> f64,
) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support_radius * filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(in_size as i64) as usize;
        let mut weights = Vec::with_capacity(xmax - xmin);
        let mut total = 0.0;
        for x in xmin..xmax {
            let w = filter((x as f64 - center + 0.5) / filterscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        out.push((xmin, weights));
    }
    out
}

/// PIL's `PRECISION_BITS` for the 8-bit resample path (`32 - 8 - 2`): filter coefficients are
/// quantized to this many fractional bits and the convolution is accumulated in integers.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// PIL `clip8` for the resample accumulator (which already carries the `1<<(PRECISION_BITS-1)`
/// rounding bias): shift down by `PRECISION_BITS` and clamp to `[0,255]`.
#[inline]
fn clip8(acc: i64) -> f32 {
    if acc <= 0 {
        return 0.0;
    }
    let v = acc >> PRECISION_BITS;
    if v >= 255 {
        255.0
    } else {
        v as f32
    }
}

/// Quantize PIL float coefficients to fixed-point integers — `normalize_coeffs_8bpc`: round half
/// away from zero at `1<<PRECISION_BITS` (matches C's `(int)(±0.5 + w·2^PRECISION_BITS)`).
fn quantize_coeffs(coeffs: &[(usize, Vec<f64>)]) -> Vec<(usize, Vec<i64>)> {
    let scale = (1i64 << PRECISION_BITS) as f64;
    coeffs
        .iter()
        .map(|(xmin, w)| {
            let ik = w
                .iter()
                .map(|&c| {
                    if c < 0.0 {
                        (c * scale - 0.5) as i64
                    } else {
                        (c * scale + 0.5) as i64
                    }
                })
                .collect();
            (*xmin, ik)
        })
        .collect()
}

/// Two-pass (horizontal then vertical) separable resize of a uint8 HWC image, bit-matching PIL's
/// `ImagingResample` 8-bit path: float coefficients quantized to `PRECISION_BITS` fixed-point, an
/// integer multiply-accumulate seeded with the rounding bias, then `clip8` (`>>PRECISION_BITS` +
/// clamp) between/after passes. Returns f32 HWC with integer-valued samples in `[0, 255]`.
///
/// `c` is the interleaved channel count — 3 (RGB) for every caller that predates sc-24111, 4
/// (RGBA) for [`resize_lanczos_rgba_u8`]. The two resampling passes accumulate **per channel and
/// independently of the others**, so resizing a 4-channel buffer produces byte-identical R/G/B
/// planes to resizing the same image's 3-channel buffer; widening is therefore free of any
/// numerical consequence for the existing callers, and an all-255 alpha plane resamples to all-255
/// (the quantized coefficients of each output sample sum to exactly one unit).
// Eight parameters, one over the lint's threshold: `src`/`in_h`/`in_w`/`out_h`/`out_w` are the
// image, `c` the channel count (sc-24111), and `support_radius`/`filter` the kernel. Bundling any
// of them into a struct would be ceremony around a private, single-module helper with three
// call sites, all of which spell the kernel pair as literals.
#[allow(clippy::too_many_arguments)]
fn resize_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    c: usize,
    support_radius: f64,
    filter: &dyn Fn(f64) -> f64,
) -> crate::Result<Vec<f32>> {
    // Reject zero/degenerate dims up front (F-008/L-E): a 0 source edge makes the buffer guard below
    // vacuous (`in_h*in_w*c == 0`) and yields a silent uniform-black output instead of a rejection,
    // and a 0 target edge later divides by zero in `precompute_coeffs`. The shipped envelope never
    // hits this (validate_request bounds size), but a request-supplied mask/conditioning image can —
    // so this is a typed Err (matching `union_masks` in this file), not a panic (F-055).
    if !(in_h > 0 && in_w > 0 && out_h > 0 && out_w > 0) {
        return Err(Error::Msg(format!(
            "resize_u8: zero dimension — {in_w}×{in_h} → {out_w}×{out_h} (all edges must be > 0)"
        )));
    }
    // The inner loops index `src[(y*in_w + xmin + k)*c + ch]` trusting the caller's `in_h`/`in_w`. A
    // buffer inconsistent with those dims (e.g. a request-supplied conditioning image whose
    // `pixels.len()` doesn't match `width*height*3`) would otherwise panic deep in the loop with an
    // opaque out-of-bounds index. Fail fast at the top with a clear message (F-007). All three public
    // entry points funnel through here, and this is a no-op for every well-formed image.
    if src.len() < in_h * in_w * c {
        return Err(Error::Msg(format!(
            "resize_u8: pixel buffer too small — {} bytes for a {in_w}×{in_h} {c}-channel \
             image (need {})",
            src.len(),
            in_h * in_w * c
        )));
    }
    let bias = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: (in_h, in_w) -> (in_h, out_w).
    let hcoeffs = quantize_coeffs(&precompute_coeffs(in_w, out_w, support_radius, filter));
    let mut horiz = vec![0f32; in_h * out_w * c];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hcoeffs.iter().enumerate() {
            for ch in 0..c {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * c + ch] as i64 * wk;
                }
                horiz[(y * out_w + xx) * c + ch] = clip8(acc);
            }
        }
    }

    // Vertical pass: (in_h, out_w) -> (out_h, out_w). Reads the integer-valued horiz samples.
    let vcoeffs = quantize_coeffs(&precompute_coeffs(in_h, out_h, support_radius, filter));
    let mut out = vec![0f32; out_h * out_w * c];
    for (yy, (ymin, w)) in vcoeffs.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..c {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * c + ch] as i64 * wk;
                }
                out[(yy * out_w + x) * c + ch] = clip8(acc);
            }
        }
    }
    Ok(out)
}

/// PIL `Image.BICUBIC` resize of a uint8 RGB HWC image. Returns f32 HWC, integer-valued `[0,255]`.
/// `Err` (never a panic) on a zero dimension or a `src` buffer smaller than `in_h·in_w·3` (F-055).
pub fn resize_bicubic_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    resize_u8(src, in_h, in_w, out_h, out_w, 3, 2.0, &cubic)
}

/// PIL `Image.BILINEAR` resize of a uint8 RGB HWC image (SAM2's preprocessing filter). Returns
/// f32 HWC, integer-valued `[0,255]`.
pub fn resize_bilinear_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    resize_u8(src, in_h, in_w, out_h, out_w, 3, 1.0, &triangle)
}

/// PIL `Image.LANCZOS` resize of a uint8 RGB HWC image (the fork's `scale_to_dimensions`). Returns
/// f32 HWC, integer-valued `[0,255]`.
pub fn resize_lanczos_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    resize_u8(src, in_h, in_w, out_h, out_w, 3, 3.0, &lanczos3)
}

/// PIL `Image.LANCZOS` resize of a uint8 **RGBA** HWC image — the four-channel sibling of
/// [`resize_lanczos_u8`] (sc-24111). Returns f32 HWC, integer-valued `[0,255]`, `out_h·out_w·4`,
/// **straight (un-premultiplied)** alpha in and out.
///
/// This is upstream's own order of operations for a transparent condition image: diffusers
/// converts to `RGBA` **first** and resizes all four channels together
/// (`image_processor.resize(img, …)` on the RGBA PIL image), and only then flattens a copy over
/// white for the vision tower.
///
/// ## The resample runs in PREMULTIPLIED space
///
/// `PIL.Image.resize` does not resample an `RGBA` image's four bands straight. It special-cases
/// `LA`/`RGBA` for every filter but `NEAREST`:
///
/// ```text
/// if self.mode in ["LA", "RGBA"] and resample != Resampling.NEAREST:
///     im = self.convert({"LA": "La", "RGBA": "RGBa"}[self.mode])
///     im = im.resize(size, resample, box)
///     return im.convert(self.mode)
/// ```
///
/// so the real pipeline is **premultiply → resample the four premultiplied bands (with `clip8`
/// between passes, exactly as [`resize_lanczos_u8`] does) → un-premultiply**. That is not a
/// rounding detail: on a soft matte edge, resampling straight colour lets the colour of
/// nearly-transparent pixels bleed into visible ones at full weight, which is the classic dark
/// (or, here, arbitrary) halo. Reproducing it is what makes a transparent reference's fitted
/// pixels match upstream's rather than merely resemble them.
///
/// Both conversions use PIL's integer rules, verified exhaustively over all 256×256
/// (channel, alpha) pairs against Pillow itself:
///
/// * premultiply `c' = (c·a + 127) / 255` (round-half-up — `floor(c·a/255)` is off by one);
/// * un-premultiply `c = min(255, c'·255 / a)` for `a > 0`, and `c = c'` for `a = 0` (which only
///   arises for a fully transparent pixel, where the stored colour is already `0` on legal input).
///
/// For a fully opaque image (`a = 255` everywhere) both conversions are the identity and the
/// alpha band resamples to a constant 255, so this is **byte-identical** to widening an RGB image
/// and calling [`resize_lanczos_u8`] on it — which is what keeps the ordinary RGB reference path
/// unchanged.
///
/// `Err` (never a panic) on a zero dimension or a `src` buffer smaller than `in_h·in_w·4`.
pub fn resize_lanczos_rgba_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    if src.len() < in_h * in_w * 4 {
        return Err(Error::Msg(format!(
            "resize_lanczos_rgba_u8: pixel buffer too small — {} bytes for a {in_w}×{in_h} RGBA \
             image (need {})",
            src.len(),
            in_h * in_w * 4
        )));
    }
    // RGBA -> RGBa.
    let mut premultiplied = vec![0u8; in_h * in_w * 4];
    for (dst, px) in premultiplied
        .chunks_exact_mut(4)
        .zip(src.chunks_exact(4).take(in_h * in_w))
    {
        let a = u32::from(px[3]);
        for ch in 0..3 {
            dst[ch] = ((u32::from(px[ch]) * a + 127) / 255) as u8;
        }
        dst[3] = px[3];
    }

    let resampled = resize_u8(&premultiplied, in_h, in_w, out_h, out_w, 4, 3.0, &lanczos3)?;

    // RGBa -> RGBA.
    let mut out = vec![0f32; out_h * out_w * 4];
    for (dst, px) in out.chunks_exact_mut(4).zip(resampled.chunks_exact(4)) {
        // `resize_u8` applies `clip8`, so every sample is already an integer in [0, 255].
        let a = px[3] as u32;
        for ch in 0..3 {
            let c = px[ch] as u32;
            // NOT a `checked_div`: the zero-alpha branch does not fall back to a neutral value,
            // it passes the stored channel through UNCHANGED, which is what Pillow does (verified
            // exhaustively over all 256x256 premultiplied pairs). `checked_div(...).unwrap_or(0)`
            // would be a different, wrong rule.
            #[allow(clippy::manual_checked_ops)]
            {
                dst[ch] = if a == 0 {
                    c as f32
                } else {
                    (c * 255 / a).min(255) as f32
                };
            }
        }
        dst[3] = px[3];
    }
    Ok(out)
}

/// PIL `Image.LANCZOS` resize of an **unbounded f32** RGB HWC image — the HDR counterpart of
/// [`resize_lanczos_u8`] (sc-18790).
///
/// Shares the uint8 path's filter-coefficient precomputation, so the sampling geometry is identical, but
/// accumulates in `f64` and **never quantizes or clips**. That difference is the whole point: the
/// uint8 path's fixed-point accumulator and `clip8` saturate at 255, which would crush every
/// scene-linear highlight above diffuse white to the same value. HDR conditioning must preserve
/// them, so this path stays unbounded — negative lobes from the Lanczos kernel included, which
/// the caller's colour transform clamps at the point it actually matters.
///
/// `Err` (never a panic) on a zero dimension or a `src` buffer smaller than `in_h·in_w·3`.
pub fn resize_lanczos_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    resize_f32(src, in_h, in_w, out_h, out_w, 3.0, &lanczos3)
}

/// PIL `Image.BILINEAR` resize of an unbounded f32 RGB HWC image. See [`resize_lanczos_f32`].
pub fn resize_bilinear_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    resize_f32(src, in_h, in_w, out_h, out_w, 1.0, &triangle)
}

/// Two-pass separable resize of an unbounded f32 RGB HWC image. Assumes 3 channels.
fn resize_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    support_radius: f64,
    filter: &dyn Fn(f64) -> f64,
) -> crate::Result<Vec<f32>> {
    let c = 3usize;
    if !(in_h > 0 && in_w > 0 && out_h > 0 && out_w > 0) {
        return Err(Error::Msg(format!(
            "resize_f32: zero dimension — {in_w}×{in_h} → {out_w}×{out_h} (all edges must be > 0)"
        )));
    }
    if src.len() < in_h * in_w * c {
        return Err(Error::Msg(format!(
            "resize_f32: pixel buffer too small — {} samples for a {in_w}×{in_h} RGB image (need {})",
            src.len(),
            in_h * in_w * c
        )));
    }

    // Horizontal pass: (in_h, in_w) -> (in_h, out_w).
    let hcoeffs = precompute_coeffs(in_w, out_w, support_radius, filter);
    let mut horiz = vec![0f32; in_h * out_w * c];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hcoeffs.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0f64;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * c + ch] as f64 * wk;
                }
                horiz[(y * out_w + xx) * c + ch] = acc as f32;
            }
        }
    }

    // Vertical pass: (in_h, out_w) -> (out_h, out_w).
    let vcoeffs = precompute_coeffs(in_h, out_h, support_radius, filter);
    let mut out = vec![0f32; out_h * out_w * c];
    for (yy, (ymin, w)) in vcoeffs.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..c {
                let mut acc = 0f64;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * c + ch] as f64 * wk;
                }
                out[(yy * out_w + x) * c + ch] = acc as f32;
            }
        }
    }
    Ok(out)
}

/// Nearest-neighbour resize of a uint8 HWC image (`C = len / (in_h·in_w)`), torch
/// `F.interpolate(mode="nearest")`: each destination samples source index `floor(dst · in/out)`.
/// Unlike the windowed filters above it introduces **no** intermediate values, so it's the right
/// resampler for masks / label maps where interpolation would create spurious grays that flip a
/// downstream binarize threshold. Returns f32 HWC, integer-valued `[0,255]`.
pub fn resize_nearest_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> crate::Result<Vec<f32>> {
    // Fail fast on a zero/degenerate dimension (F-008): `c = src.len() / (in_h*in_w)` divides by zero
    // when a source edge is 0, `(in_h - 1)` / `(in_w - 1)` underflow `usize`, and a 0 target edge
    // divides by zero in the index map. Reachable from a request-supplied mask/conditioning image
    // (e.g. inpaint mask) — `validate_request`'s min-size does not cover conditioning images — so turn
    // the opaque arithmetic panic into a typed Err (matching `union_masks`), not a panic (F-055).
    if !(in_h > 0 && in_w > 0 && out_h > 0 && out_w > 0) {
        return Err(Error::Msg(format!(
            "resize_nearest_u8: zero dimension — {in_w}×{in_h} → {out_w}×{out_h} (all edges must be > 0)"
        )));
    }
    let c = src.len() / (in_h * in_w);
    let mut out = vec![0f32; out_h * out_w * c];
    for oy in 0..out_h {
        let sy = ((oy * in_h) / out_h).min(in_h - 1);
        for ox in 0..out_w {
            let sx = ((ox * in_w) / out_w).min(in_w - 1);
            for ch in 0..c {
                out[(oy * out_w + ox) * c + ch] = src[(sy * in_w + sx) * c + ch] as f32;
            }
        }
    }
    Ok(out)
}

/// Round-half-to-even (Python `round`), for pixel-geometry parity with the worker's `_contain_box`
/// (Rust's `f64::round` is half-away-from-zero, which differs at exact `.5`). Positive inputs only.
fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// Where a `src_w`×`src_h` image lands when **contained** (long edge fits) and centered in a
/// `width`×`height` box: `(new_w, new_h, left, top)`. Mirrors the worker's `_contain_box` (Python
/// `round` = half-to-even) so the kept rect and a padded source line up exactly.
pub fn contain_box(src_w: u32, src_h: u32, width: u32, height: u32) -> (u32, u32, i32, i32) {
    // Preconditions (the caller's `validate_request` enforces both; dims come from the bounded request
    // size, L-E): the source edges are non-zero — else the `width/src_w` ratio divides by zero — and
    // every edge is `<= i32::MAX`, since the `as i32` casts below wrap on a larger value. These are
    // `debug_assert`s rather than a hard error because this is pure host geometry on already-validated
    // sizes; making the precondition explicit catches an out-of-envelope caller in debug/test.
    debug_assert!(
        src_w > 0 && src_h > 0,
        "contain_box: zero source dimension {src_w}×{src_h}"
    );
    debug_assert!(
        width <= i32::MAX as u32 && height <= i32::MAX as u32,
        "contain_box: target {width}×{height} exceeds i32::MAX (the `as i32` casts would wrap)"
    );
    let ratio = (width as f64 / src_w as f64).min(height as f64 / src_h as f64);
    let new_w = round_half_even(src_w as f64 * ratio).max(1) as u32;
    let new_h = round_half_even(src_h as f64 * ratio).max(1) as u32;
    let left = (width as i32 - new_w as i32) / 2;
    let top = (height as i32 - new_h as i32) / 2;
    (new_w, new_h, left, top)
}

/// Outpaint inpaint mask (the worker's `outpaint_border_mask`): an RGB8 grayscale mask —
/// **white (255) = the padded border to GENERATE, black (0) = the centered source rect to KEEP**
/// (inpaint convention: white = repaint). Geometry matches a "pad"/contain fit so the mask aligns
/// with the padded source. Pure host-side op; the engine consumes it as a `Conditioning::Mask`.
///
/// The worker's optional gaussian **feather** is intentionally omitted: the inpaint pipeline
/// binarizes the mask (`do_binarize`), and a symmetric blur's 0.5 crossing stays on the original
/// edge, so after the 8× latent downsample the feather is a no-op (it only rounds corners
/// sub-latent-pixel). Callers that want the seam softened should feather post-decode, not here.
pub fn outpaint_border_mask(src_w: u32, src_h: u32, width: u32, height: u32) -> Image {
    let (w, h) = (width.max(1), height.max(1));
    let (new_w, new_h, left, top) = contain_box(src_w, src_h, w, h);
    let mut pixels = vec![255u8; (w * h * 3) as usize]; // white = generate
    for y in 0..new_h as i32 {
        let cy = top + y;
        if cy < 0 || cy >= h as i32 {
            continue;
        }
        for x in 0..new_w as i32 {
            let cx = left + x;
            if cx < 0 || cx >= w as i32 {
                continue;
            }
            let idx = ((cy as u32 * w + cx as u32) * 3) as usize;
            pixels[idx] = 0; // black = keep
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Per-pixel max ("white wins" — PIL `ImageChops.lighter`) of two equal-size RGB8 masks. Unions a
/// user edit region with a generated outpaint border.
pub fn union_masks(a: &Image, b: &Image) -> crate::Result<Image> {
    // For well-formed Images the `pixels.len()` check is implied by the dimension check; it is kept
    // to also guard the malformed case (a buffer whose length ≠ width·height·channels) so the
    // element-wise `zip` below can't silently truncate to the shorter buffer.
    if (a.width, a.height) != (b.width, b.height) || a.pixels.len() != b.pixels.len() {
        return Err(Error::Msg(format!(
            "union_masks: size mismatch {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        )));
    }
    let pixels = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| x.max(y))
        .collect();
    Ok(Image {
        width: a.width,
        height: a.height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::checked_image_buffer_len;

    #[test]
    fn checked_image_buffer_len_handles_valid_zero_and_overflow_dimensions() {
        assert_eq!(checked_image_buffer_len(640, 480, 3), Some(921_600));
        assert_eq!(checked_image_buffer_len(0, usize::MAX, 4), Some(0));
        assert_eq!(checked_image_buffer_len(usize::MAX, 2, 1), None);
        assert_eq!(checked_image_buffer_len(usize::MAX / 2 + 1, 1, 2), None);
    }

    use super::*;

    #[test]
    fn resize_nearest_introduces_no_intermediate_values() {
        // F-075: nearest copies source samples (`floor(dst·in/out)`), never blends — so a mask can't
        // gain grays that flip a binarize. 1×2 [0,255] → 1×4 replicates each pixel; 1×4 → 1×2 picks
        // the floor-sampled source indices (0 and 2).
        assert_eq!(
            resize_nearest_u8(&[0u8, 255], 1, 2, 1, 4).unwrap(),
            vec![0.0, 0.0, 255.0, 255.0]
        );
        assert_eq!(
            resize_nearest_u8(&[10u8, 20, 30, 40], 1, 4, 1, 2).unwrap(),
            vec![10.0, 30.0]
        );
    }

    #[test]
    fn resize_accepts_correctly_sized_buffer() {
        // A well-formed 2×2 RGB buffer (12 bytes) resizes without erroring (the F-007 guard is a
        // no-op for valid inputs).
        let src = vec![0u8; 2 * 2 * 3];
        let out = resize_bicubic_u8(&src, 2, 2, 4, 4).unwrap();
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn resize_rejects_undersized_buffer() {
        // F-007/F-055: claiming a 4×4 image from an 8-byte buffer must fail fast with a typed Err,
        // not an opaque out-of-bounds index deep in the resample loop (and not a panic).
        let src = vec![0u8; 8];
        let err = resize_bilinear_u8(&src, 4, 4, 2, 2).unwrap_err();
        assert!(err.to_string().contains("pixel buffer too small"));
    }

    #[test]
    fn resize_nearest_rejects_zero_source_dim() {
        // F-008/F-055: a 0-width source would make `c = len / (in_h*in_w)` divide by zero. Typed Err.
        let err = resize_nearest_u8(&[], 4, 0, 4, 4).unwrap_err();
        assert!(err.to_string().contains("zero dimension"));
    }

    #[test]
    fn resize_nearest_rejects_zero_target_dim() {
        // F-008/F-055: a 0 target edge divides by zero in the index map. Typed Err.
        let src = vec![0u8; 4 * 4 * 3];
        let err = resize_nearest_u8(&src, 4, 4, 0, 4).unwrap_err();
        assert!(err.to_string().contains("zero dimension"));
    }

    #[test]
    fn resize_windowed_rejects_zero_source_dim() {
        // L-E/F-055: the windowed path's buffer guard is vacuous when a source edge is 0
        // (`in_h*in_w*c == 0`); the dims guard returns a typed Err instead of silent uniform-black.
        let err = resize_bicubic_u8(&[], 0, 4, 4, 4).unwrap_err();
        assert!(err.to_string().contains("zero dimension"));
    }

    #[test]
    #[should_panic(expected = "exceeds i32::MAX")]
    fn contain_box_rejects_oversized_target_in_debug() {
        // L-E: a target edge above i32::MAX would wrap the `as i32` centering math. The debug_assert
        // makes the precondition explicit (this test runs in debug, like all `cargo test`).
        let _ = contain_box(100, 100, i32::MAX as u32 + 1, 200);
    }

    #[test]
    fn outpaint_border_mask_keeps_centered_source() {
        // A 50×100 source contained in a 200×200 canvas: long edge (100) fits → ratio 2.0 →
        // 100×200 kept rect, centered at left=50, top=0. White border L/R, black center column.
        let m = outpaint_border_mask(50, 100, 200, 200);
        assert_eq!((m.width, m.height), (200, 200));
        let px = |x: u32, y: u32| m.pixels[((y * 200 + x) * 3) as usize];
        assert_eq!(px(0, 100), 255, "left border = generate");
        assert_eq!(px(199, 100), 255, "right border = generate");
        assert_eq!(px(100, 100), 0, "center = keep");
        assert_eq!(px(50, 100), 0, "kept rect starts at left=50");
        assert_eq!(px(49, 100), 255, "just outside kept rect = generate");
    }

    #[test]
    fn union_masks_white_wins() {
        let a = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };
        let b = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 0, 0, 0],
        };
        let u = union_masks(&a, &b).unwrap();
        assert_eq!(u.pixels, vec![255, 255, 255, 0, 0, 0]);
    }
}
