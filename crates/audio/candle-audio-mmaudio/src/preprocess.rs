//! RGB-frame preprocessing faithful to Synchformer's own transform (MMAudio runtime path).
//!
//! Steps, per frame: resize the shorter edge to [`config::IMG_SIZE`] (CatmullRom, an approximation
//! of the reference's torchvision bicubic — see `resize_center_crop`), center-crop to
//! `IMG_SIZE × IMG_SIZE`, scale to `[0,1]`, then normalize with mean/std = 0.5 → `[-1, 1]`
//! (`DATA.MEAN` / `DATA.STD`, **not** ImageNet stats). Frames are then windowed into overlapping
//! 16-frame segments (`sync_step_size = 8`, 50% overlap) and stacked into the encoder's input
//! tensor `(S, C=3, T=16, H=224, W=224)` — i.e. already in the post-permute layout
//! `MotionFormer.forward` consumes.

use candle_audio::candle_core::{Device, Tensor};
use candle_nn::ops;
use image::imageops::FilterType;
use image::RgbImage;

use crate::config;
use crate::{AudioError, Result};

/// Resize a frame's shorter edge to `IMG_SIZE`, then center-crop `IMG_SIZE × IMG_SIZE`.
///
/// The reference sync transform (`eval_utils.py`: `v2.Resize(224, interpolation=BICUBIC)`) uses
/// torchvision bicubic. The `image` crate has no bicubic filter, so we use `CatmullRom`, its
/// closest analogue. This **approximates** the reference — it does not bit-match torchvision's
/// a=-0.75 bicubic kernel — so conditioning features on non-224 input can differ slightly.
fn resize_center_crop(frame: &RgbImage) -> RgbImage {
    let (w, h) = frame.dimensions();
    let target = config::IMG_SIZE as u32;
    // Shorter-edge → target, preserving aspect ratio.
    let (nw, nh) = if w <= h {
        (
            target,
            ((h as f32) * (target as f32) / (w as f32)).round() as u32,
        )
    } else {
        (
            ((w as f32) * (target as f32) / (h as f32)).round() as u32,
            target,
        )
    };
    // CatmullRom approximates the reference's torchvision bicubic (see fn doc).
    let resized = image::imageops::resize(
        frame,
        nw.max(target),
        nh.max(target),
        FilterType::CatmullRom,
    );
    // Center crop.
    let (rw, rh) = resized.dimensions();
    let x0 = (rw - target) / 2;
    let y0 = (rh - target) / 2;
    image::imageops::crop_imm(&resized, x0, y0, target, target).to_image()
}

/// Flatten one preprocessed frame into a channel-major `[C, H, W]` f32 buffer, normalized to
/// `[-1, 1]`. Channel-major so a stack of frames reshapes directly into `(T, C, H, W)`.
fn frame_to_chw(frame: &RgbImage) -> Vec<f32> {
    let cropped = resize_center_crop(frame);
    let hw = config::IMG_SIZE * config::IMG_SIZE;
    let mut out = vec![0f32; config::IN_CHANS * hw];
    for (i, px) in cropped.pixels().enumerate() {
        // pixels() iterates row-major (y outer, x inner) → position i within a channel plane.
        for c in 0..config::IN_CHANS {
            let v = (px[c] as f32) / 255.0;
            out[c * hw + i] = (v - config::NORM_MEAN) / config::NORM_STD;
        }
    }
    out
}

/// Number of overlapping 16-frame segments a clip of `n_frames` yields (`(n-16)/8 + 1`), or 0 if
/// fewer than one full segment.
pub fn num_segments(n_frames: usize) -> usize {
    if n_frames < config::NUM_FRAMES {
        0
    } else {
        (n_frames - config::NUM_FRAMES) / config::SYNC_STEP_SIZE + 1
    }
}

/// Window RGB frames into the encoder input tensor `(S, C=3, T=16, H=224, W=224)`.
///
/// Requires at least [`config::NUM_FRAMES`] frames. Frames are assumed sampled at
/// [`config::SYNC_FRAME_RATE`] fps (the caller is responsible for the temporal resample — that is a
/// container/decoder concern outside this deterministic encoder); this fn does the spatial
/// transform + normalization + overlapping-window packing.
pub fn frames_to_segments(frames: &[RgbImage], device: &Device) -> Result<Tensor> {
    let n = frames.len();
    let s = num_segments(n);
    if s == 0 {
        return Err(AudioError::Msg(format!(
            "synchformer: need at least {} frames for one segment, got {n}",
            config::NUM_FRAMES
        )));
    }
    // Preprocess each frame once (frames are shared across overlapping windows).
    let per_frame: Vec<Vec<f32>> = frames.iter().map(frame_to_chw).collect();
    let chw = config::IN_CHANS * config::IMG_SIZE * config::IMG_SIZE;

    // Build (S, T, C, H, W) then permute T<->C to (S, C, T, H, W).
    let mut buf: Vec<f32> = Vec::with_capacity(s * config::NUM_FRAMES * chw);
    for seg in 0..s {
        let start = seg * config::SYNC_STEP_SIZE;
        for t in 0..config::NUM_FRAMES {
            buf.extend_from_slice(&per_frame[start + t]);
        }
    }
    // (S, T, C, H, W)
    let stw = Tensor::from_vec(
        buf,
        (
            s,
            config::NUM_FRAMES,
            config::IN_CHANS,
            config::IMG_SIZE,
            config::IMG_SIZE,
        ),
        device,
    )?;
    // → (S, C, T, H, W): the layout MMAudio's `permute(0,1,3,2,4,5)` produces per segment.
    let scthw = stw.permute((0, 2, 1, 3, 4))?.contiguous()?;
    Ok(scthw)
}

/// Softmax helper re-export point kept local so the encoder and preprocess share one import site.
pub(crate) fn softmax_last(x: &Tensor) -> Result<Tensor> {
    Ok(ops::softmax_last_dim(x)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(rgb))
    }

    #[test]
    fn num_segments_matches_overlap_formula() {
        assert_eq!(num_segments(15), 0, "fewer than 16 frames = no segment");
        assert_eq!(num_segments(16), 1);
        assert_eq!(num_segments(24), 2); // (24-16)/8 + 1
        assert_eq!(num_segments(200), 24, "8s @ 25fps -> 24 segments");
    }

    #[test]
    fn frames_to_segments_shape_range_and_determinism() {
        let dev = Device::Cpu;
        // 16 non-square frames of varying color to exercise resize+crop.
        let frames: Vec<RgbImage> = (0..16)
            .map(|i| solid(320, 240, [(i * 10) as u8, 40, 200]))
            .collect();
        let t1 = frames_to_segments(&frames, &dev).expect("segments");
        assert_eq!(
            t1.dims(),
            &[
                1,
                config::IN_CHANS,
                config::NUM_FRAMES,
                config::IMG_SIZE,
                config::IMG_SIZE
            ]
        );
        let v = t1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|x| (-1.0001..=1.0001).contains(x)),
            "normalized to [-1,1]"
        );
        assert!(v.iter().all(|x| x.is_finite()));
        // Deterministic.
        let t2 = frames_to_segments(&frames, &dev).unwrap();
        assert_eq!(v, t2.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }

    #[test]
    fn too_few_frames_errors() {
        let dev = Device::Cpu;
        let frames: Vec<RgbImage> = (0..8).map(|_| solid(224, 224, [10, 20, 30])).collect();
        assert!(frames_to_segments(&frames, &dev).is_err());
    }
}
