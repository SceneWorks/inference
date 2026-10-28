//! Host-side image preprocessing for the vision paths, shared by the MLX and Candle engines
//! (sc-27255; the MLX 7157 preprocessor and its Candle 7262 port were line-for-line copies).
//!
//! [`resize_bicubic_u8`] reproduces PIL's `Image.BICUBIC` resize **bit-exactly** — PIL's
//! *fixed-point* arithmetic, not merely "a bicubic", is what avoids the ±1–2 ULP divergence at
//! gradient cliffs. [`SiglipImageProcessor`] resizes to the model's square input and normalizes;
//! [`Qwen35ImageProcessor`] smart-resizes and patchifies (images and videos). Everything here is
//! tensor-free: each backend wraps the host values as its own tensor layout.

/// A preprocessing refusal (a zero dimension, a pixel buffer that does not match its size, frames
/// that disagree); each backend surfaces the message as its own error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreprocessError(pub String);

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PreprocessError {}

/// Result alias for the preprocessing seam.
pub type Result<T> = std::result::Result<T, PreprocessError>;

/// Patchified vision rows: `rows` rows of `features` f32 values, row-major (`values.len() ==
/// rows * features`).
#[derive(Clone, Debug, PartialEq)]
pub struct PatchRows {
    /// The row-major values.
    pub values: Vec<f32>,
    /// Number of patch rows.
    pub rows: usize,
    /// Values per row (`C·T·P·P`).
    pub features: usize,
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

/// Per-output-pixel 1-D resampling coefficients, matching PIL `precompute_coeffs`: antialias by
/// scaling the filter support when downscaling, clamp the window to the input bounds, renormalize.
fn precompute_coeffs(
    in_size: usize,
    out_size: usize,
    support_radius: f64,
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
            let w = cubic((x as f64 - center + 0.5) / filterscale);
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

/// PIL `PRECISION_BITS` for the 8-bit resample path (`32 - 8 - 2`).
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// PIL `clip8` for the resample accumulator (which carries the `1<<(PRECISION_BITS-1)` rounding
/// bias): shift down by `PRECISION_BITS` and clamp to `[0,255]`.
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

/// Quantize PIL float coefficients to fixed-point ints (`normalize_coeffs_8bpc`: round half away
/// from zero at `1<<PRECISION_BITS`).
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

/// PIL `Image.BICUBIC` resize of a uint8 RGB HWC image (two separable passes, fixed-point). Returns
/// f32 HWC, integer-valued in `[0, 255]`. Assumes 3 channels.
pub fn resize_bicubic_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>> {
    const C: usize = 3;
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(PreprocessError(format!(
            "resize: zero dimension {in_w}x{in_h} -> {out_w}x{out_h}"
        )));
    }
    if src.len() < in_h * in_w * C {
        return Err(PreprocessError(format!(
            "resize: pixel buffer too small ({} bytes for {in_w}x{in_h} RGB, need {})",
            src.len(),
            in_h * in_w * C
        )));
    }
    let bias = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: (in_h, in_w) -> (in_h, out_w).
    let hcoeffs = quantize_coeffs(&precompute_coeffs(in_w, out_w, 2.0));
    let mut horiz = vec![0f32; in_h * out_w * C];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hcoeffs.iter().enumerate() {
            for ch in 0..C {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * C + ch] as i64 * wk;
                }
                horiz[(y * out_w + xx) * C + ch] = clip8(acc);
            }
        }
    }

    // Vertical pass: (in_h, out_w) -> (out_h, out_w).
    let vcoeffs = quantize_coeffs(&precompute_coeffs(in_h, out_h, 2.0));
    let mut out = vec![0f32; out_h * out_w * C];
    for (yy, (ymin, w)) in vcoeffs.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..C {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * C + ch] as i64 * wk;
                }
                out[(yy * out_w + x) * C + ch] = clip8(acc);
            }
        }
    }
    Ok(out)
}

/// RGB uint8 → SigLIP input: resize (BICUBIC) to a square `size`, rescale to `[0,1]`, and normalize
/// per channel.
#[derive(Clone, Debug)]
pub struct SiglipImageProcessor {
    /// Target square edge (384 for so400m-patch14-384).
    pub size: usize,
    /// Per-channel mean (SigLIP: 0.5).
    pub mean: [f32; 3],
    /// Per-channel std (SigLIP: 0.5).
    pub std: [f32; 3],
}

impl Default for SiglipImageProcessor {
    fn default() -> Self {
        Self {
            size: 384,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

impl SiglipImageProcessor {
    /// Resize raw interleaved RGB8 `pixels` (`width*height*3` bytes) to the square `size` and
    /// normalize: `size*size*3` f32 values in HWC (interleaved) order. Each backend lays them out
    /// as its vision tower's tensor (MLX NHWC, Candle NCHW).
    pub fn normalized_hwc(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(PreprocessError(format!(
                "siglip preprocess: expected {expected} RGB bytes for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        let resized: Vec<f32> = if width == self.size && height == self.size {
            pixels.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(pixels, height, width, self.size, self.size)?
        };
        let mut normalized = Vec::with_capacity(self.size * self.size * 3);
        for px in resized.chunks_exact(3) {
            for (ch, &v) in px.iter().enumerate() {
                normalized.push((v / 255.0 - self.mean[ch]) / self.std[ch]);
            }
        }
        Ok(normalized)
    }
}

/// RGB uint8 → Qwen3.6 vision input: dynamic-resolution **smart-resize** to a patch-grid multiple
/// (BICUBIC), rescale + per-channel normalize, then patchify into the `Conv3d` temporal/patch layout
/// the backend `Qwen35VisionModel` consumes (`Qwen2VLImageProcessorFast` with the qwen3.6
/// settings — patch 16, merge 2, temporal 2, mean/std 0.5). Returns the `pixel_values`
/// `[total_patches, C·T·P·P]` (in merge-block order, the inner feature `(C, T, P_row, P_col)`
/// flattened with the single frame replicated `T` times) plus the per-image `grid_thw` (`[t, h, w]`
/// in patch units).
///
/// The resize reuses [`resize_bicubic_u8`] (PIL fixed-point BICUBIC). The reference fast processor
/// resizes with torchvision BICUBIC; the two agree to ≤ 2/255 (mean ~0.004/255) — negligible against
/// the encoder's own f32-GEMM noise — so this is faithful without re-deriving torchvision's kernel.
#[derive(Clone, Debug)]
pub struct Qwen35ImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    /// Smart-resize pixel bounds (`size.shortest_edge` / `size.longest_edge`).
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// Video-path smart-resize pixel bounds (`video_preprocessor_config.json`
    /// `size.shortest_edge` / `size.longest_edge`), distinct from the image bounds.
    pub video_min_pixels: usize,
    pub video_max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Qwen35ImageProcessor {
    /// The qwen3.6 `preprocessor_config.json` geometry.
    fn default() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 65536,          // image size.shortest_edge (256²)
            max_pixels: 16777216,       // image size.longest_edge (4096²)
            video_min_pixels: 4096,     // video_preprocessor_config size.shortest_edge
            video_max_pixels: 25165824, // video_preprocessor_config size.longest_edge
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

impl Qwen35ImageProcessor {
    /// Smart-resize: round each dimension to a multiple of `patch_size·merge_size` while keeping the
    /// total pixel count within `[min_pixels, max_pixels]` and the aspect ratio as close as possible
    /// (`Qwen2VL.smart_resize`). The initial round is **half-to-even** (Python `round`), which the
    /// min/max-pixels branches then override via floor/ceil. Uses the image processor's pixel bounds.
    pub fn smart_resize(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        self.smart_resize_with(height, width, 1, self.min_pixels, self.max_pixels)
    }

    /// The general smart-resize with explicit pixel bounds and a temporal multiplier. The pixel budget
    /// is over `temporal · h_bar · w_bar` (the video reference folds the temporal axis into the
    /// budget; for images `temporal = 1` this reduces to the standard 2-D smart-resize). `min_pixels`
    /// / `max_pixels` come from the image vs video preprocessor config.
    pub fn smart_resize_with(
        &self,
        height: usize,
        width: usize,
        temporal: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Result<(usize, usize)> {
        let factor = self.patch_size * self.merge_size;
        let (hi, lo) = (height.max(width), height.min(width));
        if lo == 0 {
            return Err(PreprocessError(format!(
                "smart_resize: zero dimension {width}x{height}"
            )));
        }
        if hi as f64 / lo as f64 > 200.0 {
            return Err(PreprocessError(format!(
                "smart_resize: aspect ratio {} exceeds 200",
                hi as f64 / lo as f64
            )));
        }
        let round_factor =
            |x: usize| -> usize { (x as f64 / factor as f64).round_ties_even() as usize * factor };
        let (mut hb, mut wb) = (round_factor(height), round_factor(width));
        let (hw, fac) = ((height * width) as f64, factor as f64);
        let t = temporal.max(1) as f64;
        let budget = t * (hb * wb) as f64;
        // Pixel budget includes the temporal factor (video reference); `beta` divides out `temporal`.
        if budget > max_pixels as f64 {
            let beta = (t * hw / max_pixels as f64).sqrt();
            hb = factor.max((height as f64 / beta / fac).floor() as usize * factor);
            wb = factor.max((width as f64 / beta / fac).floor() as usize * factor);
        } else if budget < min_pixels as f64 {
            let beta = (min_pixels as f64 / (t * hw)).sqrt();
            hb = (height as f64 * beta / fac).ceil() as usize * factor;
            wb = (width as f64 * beta / fac).ceil() as usize * factor;
        }
        Ok((hb, wb))
    }

    /// Preprocess raw interleaved RGB8 `pixels` (`width*height*3` bytes) → `(pixel_values, grid_thw)`
    /// on the host; each backend wraps the rows as its tensor.
    pub fn image_patches(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<(PatchRows, Vec<[i32; 3]>)> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(PreprocessError(format!(
                "qwen3.6 preprocess: expected {expected} RGB bytes for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        let (rh, rw) = self.smart_resize(height, width)?;
        // HWC f32 in [0,255]: either the raw pixels (no-op resize) or the BICUBIC resample.
        let resized: Vec<f32> = if rh == height && rw == width {
            pixels.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(pixels, height, width, rh, rw)?
        };

        let (p, m, t) = (self.patch_size, self.merge_size, self.temporal_patch_size);
        let (grid_h, grid_w) = (rh / p, rw / p);
        let feat = 3 * t * p * p;
        let mut out = Vec::with_capacity(grid_h * grid_w * feat);
        // Merge-block patch order (bh, bw, ih, iw) — the order the encoder's position ids / pos-embed
        // gather assume; inner feature (channel, temporal, patch_row, patch_col), temporal replicated.
        let norm = |row: usize, col: usize, c: usize| -> f32 {
            (resized[(row * rw + col) * 3 + c] / 255.0 - self.mean[c]) / self.std[c]
        };
        for bh in 0..grid_h / m {
            for bw in 0..grid_w / m {
                for ih in 0..m {
                    for iw in 0..m {
                        let (gh, gw) = (bh * m + ih, bw * m + iw);
                        for c in 0..3 {
                            for _ in 0..t {
                                for pr in 0..p {
                                    for pc in 0..p {
                                        out.push(norm(gh * p + pr, gw * p + pc, c));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((
            PatchRows {
                values: out,
                rows: grid_h * grid_w,
                features: feat,
            },
            vec![[1, grid_h as i32, grid_w as i32]],
        ))
    }

    /// Preprocess a sampled **video** — a list of already-decoded RGB8 frames (each
    /// `width*height*3` bytes, all the same size) — into the `(pixel_values_videos, video_grid_thw)`
    /// the backend `Qwen35VisionModel` consumes, mirroring `Qwen3VLVideoProcessor._preprocess`.
    ///
    /// **Temporal patching (the video-vs-image difference).** The vision encoder folds
    /// `temporal_patch_size` (2) consecutive frames into one temporal patch, so the frame count is
    /// padded up to a multiple of `temporal_patch_size` (the reference repeats the last frame) and
    /// `grid_t = padded_frames / temporal_patch_size`. Each row's inner feature is
    /// `(channel, temporal, patch_row, patch_col)` flattened — but where an *image* replicates its
    /// single frame across the temporal axis, a video carries the **two distinct frames** of the
    /// temporal patch. The spatial layout (`smart_resize`, merge-block patch order) is identical to
    /// [`Self::image_patches`].
    ///
    /// Returns `pixel_values_videos` `[grid_t·grid_h·grid_w, C·T·P·P]` plus the single
    /// `video_grid_thw` `[grid_t, grid_h, grid_w]` (patch units). One video → one grid entry; the
    /// per-frame timestamp tokens are rendered separately by the provider (Text–Timestamp Alignment).
    pub fn video_patches(&self, frames: &[(&[u8], usize, usize)]) -> Result<(PatchRows, [i32; 3])> {
        if frames.is_empty() {
            return Err(PreprocessError(
                "qwen3.6 video preprocess: no frames".into(),
            ));
        }
        let (_, w0, h0) = frames[0];
        for (i, &(px, w, h)) in frames.iter().enumerate() {
            if w != w0 || h != h0 {
                return Err(PreprocessError(format!(
                    "qwen3.6 video preprocess: frame {i} is {w}x{h}, expected {w0}x{h0} (all frames must match)"
                )));
            }
            if px.len() != w * h * 3 {
                return Err(PreprocessError(format!(
                    "qwen3.6 video preprocess: frame {i} expected {} RGB bytes for {w}x{h}, got {}",
                    w * h * 3,
                    px.len()
                )));
            }
        }

        let (p, m, t) = (self.patch_size, self.merge_size, self.temporal_patch_size);
        // Video smart-resize uses the video pixel bounds and folds the (padded) frame count into the
        // pixel budget (`t_bar = ceil(n / temporal) * temporal`), matching `Qwen3VLVideoProcessor`.
        let t_bar = frames.len().div_ceil(t) * t;
        let (rh, rw) =
            self.smart_resize_with(h0, w0, t_bar, self.video_min_pixels, self.video_max_pixels)?;
        let (grid_h, grid_w) = (rh / p, rw / p);

        // Resize every frame to the common `(rh, rw)` (HWC f32 in [0, 255]).
        let mut resized_frames: Vec<Vec<f32>> = Vec::with_capacity(frames.len());
        for &(px, w, h) in frames {
            let r: Vec<f32> = if rh == h && rw == w {
                px.iter().map(|&v| v as f32).collect()
            } else {
                resize_bicubic_u8(px, h, w, rh, rw)?
            };
            resized_frames.push(r);
        }

        // Pad the frame count up to a multiple of `temporal_patch_size` by repeating the last frame
        // (matching `Qwen3VLVideoProcessor`), then `grid_t = padded / temporal_patch_size`.
        let mut n_frames = resized_frames.len();
        while !n_frames.is_multiple_of(t) {
            let last = resized_frames.last().unwrap().clone();
            resized_frames.push(last);
            n_frames += 1;
        }
        let grid_t = n_frames / t;

        let feat = 3 * t * p * p;
        let mut out = Vec::with_capacity(grid_t * grid_h * grid_w * feat);
        // Merge-block patch order (gt, bh, bw, ih, iw) with the inner feature
        // `(channel, temporal, patch_row, patch_col)` — temporal now indexes the two **distinct**
        // frames of the temporal patch (`gt*t + tt`), not a replicated single frame.
        for gt in 0..grid_t {
            for bh in 0..grid_h / m {
                for bw in 0..grid_w / m {
                    for ih in 0..m {
                        for iw in 0..m {
                            let (gh, gw) = (bh * m + ih, bw * m + iw);
                            for c in 0..3 {
                                for tt in 0..t {
                                    let frame = &resized_frames[gt * t + tt];
                                    for pr in 0..p {
                                        for pc in 0..p {
                                            let row = gh * p + pr;
                                            let col = gw * p + pc;
                                            let v = (frame[(row * rw + col) * 3 + c] / 255.0
                                                - self.mean[c])
                                                / self.std[c];
                                            out.push(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((
            PatchRows {
                values: out,
                rows: grid_t * grid_h * grid_w,
                features: feat,
            },
            [grid_t as i32, grid_h as i32, grid_w as i32],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_uniform_image_is_uniform() {
        // A solid color resized stays that exact color everywhere (renormalized weights sum to 1).
        let src = [200u8, 100, 50].repeat(64 * 48); // height 48, width 64
        let out = resize_bicubic_u8(&src, 48, 64, 384, 384).unwrap();
        assert_eq!(out.len(), 384 * 384 * 3);
        for px in out.chunks_exact(3) {
            assert_eq!(px, &[200.0, 100.0, 50.0]);
        }
    }

    #[test]
    fn resize_rejects_zero_and_undersized() {
        assert!(resize_bicubic_u8(&[0u8; 12], 0, 2, 4, 4).is_err());
        assert!(resize_bicubic_u8(&[0u8; 3], 2, 2, 4, 4).is_err());
    }

    #[test]
    fn siglip_normalizes_to_minus_one_one_in_hwc_order() {
        let pixels = [0u8, 128, 255].repeat(4 * 4);
        let processor = SiglipImageProcessor {
            size: 4,
            ..SiglipImageProcessor::default()
        };
        let v = processor.normalized_hwc(&pixels, 4, 4).unwrap();
        assert_eq!(v.len(), 4 * 4 * 3);
        assert_eq!(v[0], -1.0);
        assert!((v[1] - 0.003_921_628).abs() < 1e-6);
        assert_eq!(v[2], 1.0);
        assert!(processor.normalized_hwc(&[0u8; 3], 2, 2).is_err());
    }

    #[test]
    fn video_patches_pad_odd_frame_counts_and_reject_mismatched_frames() {
        let processor = Qwen35ImageProcessor::default();
        let frame = [128u8, 128, 128].repeat(64 * 64);
        let frames: Vec<(&[u8], usize, usize)> = vec![(&frame, 64, 64); 3];
        let (rows, grid) = processor.video_patches(&frames).unwrap();
        assert_eq!(grid, [2, 4, 4], "3 frames pad to 4 -> grid_t = 2");
        assert_eq!(rows.rows, 2 * 4 * 4);
        assert_eq!(rows.features, 3 * 2 * 16 * 16);
        assert_eq!(rows.values.len(), rows.rows * rows.features);
        let small = [0u8, 0, 0].repeat(16 * 16);
        let mismatched: Vec<(&[u8], usize, usize)> = vec![(&frame, 64, 64), (&small, 16, 16)];
        assert!(processor.video_patches(&mismatched).is_err());
        assert!(processor.video_patches(&[]).is_err());
    }
}
