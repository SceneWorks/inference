//! Real-ESRGAN x2/x4: padded CHW tiles, ORT execution and validated RGB assembly.
use crate::session::ort_err;
use crate::{Error, Result};
use gen_core::CancelFlag;
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use std::path::Path;
const TILE_SIZE: usize = 512;
const TILE_PAD: usize = 16;
const MAX_UPSCALE_TARGET_DIMENSION: u32 = 8192;
const MAX_UPSCALE_TARGET_PIXELS: u64 =
    MAX_UPSCALE_TARGET_DIMENSION as u64 * MAX_UPSCALE_TARGET_DIMENSION as u64;
const CANCEL_MESSAGE: &str = "Image upscale canceled by user.";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// `upscalers.py:tile_slices` — single tile if the image fits, else a row-major grid.
fn tile_slices(w: usize, h: usize, tile: usize) -> Vec<Tile> {
    if tile == 0 || tile >= w.max(h) {
        return vec![Tile {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        }];
    }
    let mut out = Vec::new();
    let mut y0 = 0;
    while y0 < h {
        let mut x0 = 0;
        while x0 < w {
            out.push(Tile {
                x0,
                y0,
                x1: (x0 + tile).min(w),
                y1: (y0 + tile).min(h),
            });
            x0 += tile;
        }
        y0 += tile;
    }
    out
}

/// CHW f32 `[0,1]` for the crop region `[x0,x1) × [y0,y1)` of an RGB image.
fn crop_to_chw(
    img: &RgbImage,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> (Vec<f32>, usize, usize) {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut data = vec![0.0f32; 3 * ch * cw];
    for c in 0..3 {
        let plane = c * ch * cw;
        for yy in 0..ch {
            for xx in 0..cw {
                let p = img.get_pixel((x0 + xx) as u32, (y0 + yy) as u32);
                data[plane + yy * cw + xx] = p[c] as f32 / 255.0;
            }
        }
    }
    (data, cw, ch)
}

/// Real-ESRGAN's x2 model begins with `pixel_unshuffle(x, 2)` (see the RRDBNet port in
/// `scripts/spikes/sc3489_export_reference.py:88`), which reshapes the spatial dims
/// `(H, W) -> (H/2, 2, W/2, 2)` and therefore **requires even input height and width**. A
/// library/dataset image (or edge tile) with an odd dimension — e.g. 528×323 — otherwise blows
/// up the onnx `Reshape` node (`input_shape % requested_shape != 0`). Pad the CHW crop up to
/// even dims by **replicating** the last column/row, so no dark seam bleeds into the readable
/// region through the conv receptive field. The extra `factor` output px on the padded edge fall
/// outside the inner (unpadded) region the caller copies back, so they are simply discarded.
/// Harmless for x4 (its RRDBNet has no `pixel_unshuffle`, so odd dims already work) — applied
/// unconditionally to keep the call site simple.
fn pad_chw_even(data: Vec<f32>, cw: usize, ch: usize) -> (Vec<f32>, usize, usize) {
    let pw = cw + (cw & 1);
    let ph = ch + (ch & 1);
    if pw == cw && ph == ch {
        return (data, cw, ch);
    }
    let mut out = vec![0.0f32; 3 * ph * pw];
    for c in 0..3 {
        let src_plane = c * ch * cw;
        let dst_plane = c * ph * pw;
        for yy in 0..ph {
            let sy = yy.min(ch - 1);
            for xx in 0..pw {
                let sx = xx.min(cw - 1);
                out[dst_plane + yy * pw + xx] = data[src_plane + sy * cw + sx];
            }
        }
    }
    (out, pw, ph)
}

pub fn validate_upscale_target_dimensions(width: u32, height: u32) -> Result<()> {
    let pixels = u64::from(width) * u64::from(height);
    if width > MAX_UPSCALE_TARGET_DIMENSION
        || height > MAX_UPSCALE_TARGET_DIMENSION
        || pixels > MAX_UPSCALE_TARGET_PIXELS
    {
        return Err(Error::InvalidPayload(format!(
            "Upscale target {width}x{height} exceeds the {MAX_UPSCALE_TARGET_DIMENSION}px side / {MAX_UPSCALE_TARGET_PIXELS} pixel limit."
        )));
    }
    Ok(())
}

pub struct Upscaler {
    session: Session,
    /// The session's actual execution device — the hardware EP ([`crate::session::ACCEL_DEVICE`]: CoreML on macOS,
    /// CUDA off-Mac) when it initialised, else `"cpu"` after the fallback. Surfaced on the upscale
    /// result's `rawAdapterSettings` for observability (sc-8923).
    pub device: &'static str,
}

impl Upscaler {
    /// Finalize optional ORT profiling configured by the caller's loader.
    /// No profiling is enabled by the engine or by ordinary execution.
    pub fn finish_profiling(&mut self) -> Result<Vec<std::path::PathBuf>> {
        Ok(vec![self.session.end_profiling().map_err(ort_err)?.into()])
    }

    /// Load the session, preferring the hardware EP (CoreML on Mac / CUDA off-Mac) and
    /// falling back to CPU if the provider can't initialise (mirrors `pose_jobs::Detector::load`).
    pub fn load(path: &Path, load: &mut dyn crate::session::SessionLoader) -> Result<Self> {
        let ([session], device) = crate::session::load_sessions([path], "Real-ESRGAN", load)?;
        Ok(Self { session, device })
    }

    /// Tiled x`factor` upscale of one RGB image → an upscaled RGB image. Tiling +
    /// crop/place is a verbatim port of `upscalers.py:_run_tiled`.
    pub fn upscale(
        &mut self,
        img: &RgbImage,
        factor: usize,
        cancel: &CancelFlag,
    ) -> Result<RgbImage> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let ow = w
            .checked_mul(factor)
            .ok_or_else(|| Error::InvalidPayload("upscale width overflow".to_owned()))?;
        let oh = h
            .checked_mul(factor)
            .ok_or_else(|| Error::InvalidPayload("upscale height overflow".to_owned()))?;
        let ow_u32 = u32::try_from(ow)
            .map_err(|_| Error::InvalidPayload("upscale width overflow".to_owned()))?;
        let oh_u32 = u32::try_from(oh)
            .map_err(|_| Error::InvalidPayload("upscale height overflow".to_owned()))?;
        validate_upscale_target_dimensions(ow_u32, oh_u32)?;
        let mut output = vec![0u8; ow * oh * 3];
        for tl in tile_slices(w, h, TILE_SIZE) {
            if cancel.is_cancelled() {
                return Err(Error::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let cx0 = tl.x0.saturating_sub(TILE_PAD);
            let cy0 = tl.y0.saturating_sub(TILE_PAD);
            let cx1 = (tl.x1 + TILE_PAD).min(w);
            let cy1 = (tl.y1 + TILE_PAD).min(h);
            let (data, cw, ch) = crop_to_chw(img, cx0, cy0, cx1, cy1);
            // Pad to even dims for the x2 model's `pixel_unshuffle(2)` (odd W/H else fails the
            // onnx Reshape, e.g. a 323px-wide image); padded edge px are discarded below.
            let (data, cw, ch) = pad_chw_even(data, cw, ch);

            let tensor =
                Tensor::from_array((vec![1i64, 3, ch as i64, cw as i64], data)).map_err(ort_err)?;
            let outputs = self.session.run(ort::inputs![tensor]).map_err(ort_err)?;
            let (oshape, odata) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
            // Validate the reported output shape/length against the `(1, 3, ch*factor,
            // cw*factor)` contract before indexing `odata` (sc-11175/F-010) so a mis-scaled
            // pinned export can't slice out of bounds and panic inside `spawn_blocking`.
            let (och, ocw) = validate_upscale_output(oshape, odata.len(), cw, ch, factor)?;

            // inner (unpadded) region within the upscaled crop → destination
            let ix0 = (tl.x0 - cx0) * factor;
            let iy0 = (tl.y0 - cy0) * factor;
            let iw = (tl.x1 - tl.x0) * factor;
            let ih = (tl.y1 - tl.y0) * factor;
            let (dst_x0, dst_y0) = (tl.x0 * factor, tl.y0 * factor);
            for yy in 0..ih {
                for xx in 0..iw {
                    let (sy, sx) = (iy0 + yy, ix0 + xx);
                    let di = ((dst_y0 + yy) * ow + dst_x0 + xx) * 3;
                    for c in 0..3 {
                        let v = odata[c * och * ocw + sy * ocw + sx];
                        output[di + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
        RgbImage::from_raw(ow as u32, oh as u32, output)
            .ok_or_else(|| Error::InvalidPayload("upscale buffer size mismatch".to_owned()))
    }
}

/// Validate the Real-ESRGAN ONNX output shape/length before indexing `odata` (sc-11175/F-010).
///
/// The generic `SCENEWORKS_REALESRGAN_ONNX` pin serves both the x2 and x4 exports, so an
/// operator can pin an x2 export and run a 4× job (or any mismatched export). The output
/// dims would then be half (or otherwise not `input × factor`) the assumed size and the
/// `odata[c * och * ocw + sy * ocw + sx]` reads below would slice out of bounds → panic
/// inside `spawn_blocking`, killing the job as a join error and poisoning the `UPSCALERS`
/// lock for every subsequent job. Validate the `(1, 3, ch·factor, cw·factor)` contract and
/// return a `Error::Engine` with expected-vs-actual dims instead — mirroring the
/// sc-8904/F-102 sibling decode-path guards in `person_jobs::decode` / `pose_jobs`.
///
/// `cw`/`ch` are the (even-padded) per-tile model *input* width/height; a valid export
/// returns `(1, 3, ch·factor, cw·factor)`. Returns the validated `(och, ocw)`.
fn validate_upscale_output(
    oshape: &[i64],
    odata_len: usize,
    cw: usize,
    ch: usize,
    factor: usize,
) -> Result<(usize, usize)> {
    if oshape.len() != 4 {
        return Err(Error::Engine(format!(
            "real-esrgan output rank {} != 4 (expected (1, 3, H·factor, W·factor)); is the \
             pinned SCENEWORKS_REALESRGAN_ONNX export the right model?",
            oshape.len()
        )));
    }
    let och = oshape[2].max(0) as usize;
    let ocw = oshape[3].max(0) as usize;
    let exp_h = ch.saturating_mul(factor);
    let exp_w = cw.saturating_mul(factor);
    if och != exp_h || ocw != exp_w {
        return Err(Error::Engine(format!(
            "real-esrgan output dims {och}×{ocw} != expected {exp_h}×{exp_w} \
             (input {ch}×{cw} × factor {factor}); is the pinned SCENEWORKS_REALESRGAN_ONNX \
             export the wrong scale (e.g. an x2 export run as a {factor}× job)?"
        )));
    }
    let needed = 3usize.saturating_mul(och).saturating_mul(ocw);
    if odata_len < needed {
        return Err(Error::Engine(format!(
            "real-esrgan output has {odata_len} values but the (3 × {och} × {ocw}) contract \
             needs {needed}"
        )));
    }
    Ok((och, ocw))
}

#[cfg(test)]
#[path = "upscale_tests.rs"]
mod tests;
