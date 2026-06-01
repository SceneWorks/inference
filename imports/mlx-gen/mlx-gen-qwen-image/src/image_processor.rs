//! Qwen2-VL image processor — port of the fork's hand-rolled `QwenImageProcessor`
//! (`models/qwen/tokenizer/qwen_image_processor.py`), used by Qwen-Image-Edit's reference
//! flow. Pipeline: `smart_resize` → PIL-compatible BICUBIC resize → `/255` → CLIP normalize
//! → temporal-repeat → 9-D patchify → `(N, 1176)` pixel_values + `(1, 3)` grid_thw.
//!
//! Parity (tests/qwen_image_processor.rs): no-resize and upscale are **bit-exact**; the
//! antialiased downscale path matches PIL `Image.BICUBIC` to a measured max of 1/255 (one
//! uint8 quantization level — PIL's fixed-point resampler isn't bit-reproduced, but the
//! Keys kernel, antialias support scaling, and clip8 rounding are). That's well below any
//! meaningful threshold for vision-encoder input, so the float impl stands.

use mlx_rs::Array;

use mlx_gen::Result;

pub const OPENAI_CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const OPENAI_CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// RGB uint8 image in HWC layout.
pub struct ImageInput<'a> {
    pub data: &'a [u8],
    pub height: usize,
    pub width: usize,
}

/// Patchified output ready for the vision encoder.
pub struct ProcessedImage {
    /// `(grid_t·grid_h·grid_w, channel·temporal·patch·patch)` = `(N, 1176)`, f32.
    pub pixel_values: Array,
    /// `(1, 3)` int32: `[grid_t, grid_h, grid_w]`.
    pub grid_thw: Array,
}

#[derive(Debug, Clone)]
pub struct QwenImageProcessor {
    pub min_pixels: i64,
    pub max_pixels: i64,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for QwenImageProcessor {
    fn default() -> Self {
        Self {
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: OPENAI_CLIP_MEAN,
            image_std: OPENAI_CLIP_STD,
        }
    }
}

impl QwenImageProcessor {
    /// Integer target dims: round each side to a multiple of `patch_size*merge_size`,
    /// then clamp the pixel count into `[min_pixels, max_pixels]`. Uses Python's round-half-
    /// to-even (`round_ties_even`) so `.5` cases match the fork exactly.
    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = (self.patch_size * self.merge_size) as f64;
        let (h, w) = (height as f64, width as f64);
        let (minp, maxp) = (self.min_pixels as f64, self.max_pixels as f64);

        let mut h_bar = (h / factor).round_ties_even() * factor;
        let mut w_bar = (w / factor).round_ties_even() * factor;
        if h_bar * w_bar > maxp {
            let beta = ((h * w) / maxp).sqrt();
            h_bar = factor.max((h / beta / factor).floor() * factor);
            w_bar = factor.max((w / beta / factor).floor() * factor);
        } else if h_bar * w_bar < minp {
            let beta = (minp / (h * w)).sqrt();
            h_bar = (h * beta / factor).ceil() * factor;
            w_bar = (w * beta / factor).ceil() * factor;
        }
        (h_bar as usize, w_bar as usize)
    }

    pub fn preprocess(&self, image: ImageInput) -> Result<ProcessedImage> {
        let (rh, rw) = self.smart_resize(image.height, image.width);

        // Resize on the uint8 image (matching PIL), then convert to normalized f32 CHW.
        let resized: Vec<f32> = if (image.height, image.width) == (rh, rw) {
            image.data.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(image.data, image.height, image.width, rh, rw)
        };

        // /255, CLIP-normalize, and lay out as CHW; then duplicate across temporal_patch_size
        // (a single frame is repeated, mirroring the fork's `np.repeat` of the last frame).
        let (c, t) = (3usize, self.temporal_patch_size);
        let plane = rh * rw;
        let mut chw = vec![0f32; t * c * plane];
        for ch in 0..c {
            let (mean, std) = (self.image_mean[ch], self.image_std[ch]);
            for y in 0..rh {
                for x in 0..rw {
                    let v = (resized[(y * rw + x) * c + ch] / 255.0 - mean) / std;
                    let chw_idx = ch * plane + y * rw + x;
                    for frame in 0..t {
                        chw[frame * c * plane + chw_idx] = v;
                    }
                }
            }
        }

        // Patchify: (grid_t, temporal, channel, gh/m, m, patch, gw/m, m, patch)
        //   -> transpose (0,3,6,4,7,2,1,5,8) -> (grid_t·grid_h·grid_w, channel·temporal·patch²).
        let p = self.patch_size as i32;
        let m = self.merge_size as i32;
        let (gh, gw) = ((rh / self.patch_size) as i32, (rw / self.patch_size) as i32);
        let nine = Array::from_slice(&chw, &[1, t as i32, c as i32, gh / m, m, p, gw / m, m, p]);
        let patched = nine
            .transpose_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8])?
            .reshape(&[gh * gw, (c as i32) * (t as i32) * p * p])?;

        let grid_thw = Array::from_slice(&[1i32, gh, gw], &[1, 3]);
        Ok(ProcessedImage {
            pixel_values: patched,
            grid_thw,
        })
    }
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
fn resize_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    support_radius: f64,
    filter: &dyn Fn(f64) -> f64,
) -> Vec<f32> {
    let c = 3usize;
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
    out
}

/// PIL `Image.BICUBIC` resize of a uint8 HWC image (used by the Qwen2-VL processor + the edit
/// condition resize).
pub(crate) fn resize_bicubic_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    resize_u8(src, in_h, in_w, out_h, out_w, 2.0, &cubic)
}

/// PIL `Image.LANCZOS` resize of a uint8 HWC image (used by `scale_to_dimensions` for the
/// Qwen-Image-Edit reference VAE-encode path).
pub(crate) fn resize_lanczos_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    resize_u8(src, in_h, in_w, out_h, out_w, 3.0, &lanczos3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_reference_cases() {
        let p = QwenImageProcessor::default();
        assert_eq!(p.smart_resize(56, 84), (56, 84)); // already aligned -> no-op
        assert_eq!(p.smart_resize(200, 150), (196, 140)); // downscale
        assert_eq!(p.smart_resize(20, 20), (56, 56)); // upscale to min_pixels
    }

    /// `resize_u8` must be **bit-identical** to PIL `Image.BICUBIC` (the fixed-point integer path),
    /// not merely close — this is what gives the Qwen-Image-Edit conditioning image pixel-parity with
    /// the fork (sc-2465: an f64-coefficient resampler diverged ±1–2 ULP at gradient cliffs → 24% e2e
    /// px>8). Golden via `tools/dump_pil_resize_golden.py` (gitignored, like the other goldens).
    #[test]
    #[ignore = "needs tools/golden/pil_resize_golden.safetensors (run tools/dump_pil_resize_golden.py)"]
    fn resize_bicubic_matches_pil_512_to_384() {
        let g = mlx_gen::weights::Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pil_resize_golden.safetensors"
        ))
        .unwrap();
        // Sawtooth: `(x+y)%256`, ×2, ×3 — sharp 255→0 cliffs where bicubic implementations diverge.
        let mut saw = Vec::with_capacity(512 * 512 * 3);
        let mut smo = Vec::with_capacity(512 * 512 * 3);
        for y in 0..512u32 {
            for x in 0..512u32 {
                let b = (y + x) % 256;
                saw.push(b as u8);
                saw.push(((b * 2) % 256) as u8);
                saw.push(((b * 3) % 256) as u8);
                let v = ((x + y) / 4).min(255) as u8;
                smo.push(v);
                smo.push(v);
                smo.push(v);
            }
        }
        let cmp = |got: &[f32], pil: &[i32]| -> (usize, i32) {
            assert_eq!(got.len(), pil.len(), "len");
            got.iter()
                .zip(pil)
                .fold((0usize, 0i32), |(n, m), (&gv, &pv)| {
                    let d = (gv as i32 - pv).abs();
                    (n + (d != 0) as usize, m.max(d))
                })
        };
        let (saw_diff, saw_max) = cmp(
            &resize_bicubic_u8(&saw, 512, 512, 384, 384),
            g.require("pil384").unwrap().as_slice::<i32>(),
        );
        let (smo_diff, smo_max) = cmp(
            &resize_bicubic_u8(&smo, 512, 512, 384, 384),
            g.require("pil384_smooth").unwrap().as_slice::<i32>(),
        );
        println!("vs PIL 512->384: sawtooth {saw_diff} diff (max {saw_max}), smooth {smo_diff} diff (max {smo_max})");
        assert_eq!(
            saw_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the cliff gradient"
        );
        assert_eq!(
            smo_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the smooth ramp"
        );
    }
}
