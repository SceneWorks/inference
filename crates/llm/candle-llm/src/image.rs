//! Host-side image preprocessing for the vision path (story 7262 — the Candle port of mlx-llm's
//! 7157 preprocessor).
//!
//! The preprocessing itself — the PIL-exact fixed-point BICUBIC [`resize_bicubic_u8`], the SigLIP
//! square resize + normalize, the Qwen3.6 smart-resize + patchify — is tensor-free and lives once
//! in [`core_llm::vision_preprocess`], shared with MLX (sc-27255). This module only lays the host
//! values out as the Candle tensors the vision towers consume: **NCHW** `[1, 3, size, size]` for
//! the [`SiglipVisionTower`](crate::models::SiglipVisionTower) (Candle's `conv2d` layout),
//! `[rows, C·T·P·P]` for the [`Qwen35VisionModel`](crate::models::Qwen35VisionModel).

use candle_core::{Device, Tensor};

use crate::error::Result;

pub use core_llm::vision_preprocess::{
    resize_bicubic_u8, PatchRows, Qwen35ImageProcessor, SiglipImageProcessor,
};

/// The SigLIP processor's Candle tensor.
pub trait SiglipPreprocess {
    /// Preprocess raw interleaved RGB8 `pixels` (`width*height*3` bytes) into the SigLIP NCHW
    /// tensor `[1, 3, size, size]` on `device`.
    fn preprocess(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<Tensor>;
}

impl SiglipPreprocess for SiglipImageProcessor {
    fn preprocess(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let hwc = self.normalized_hwc(pixels, width, height)?;
        // Transpose the normalized HWC values to CHW so the result is row-major NCHW.
        let n = self.size * self.size;
        let mut chw = vec![0f32; 3 * n];
        for (i, px) in hwc.chunks_exact(3).enumerate() {
            for (ch, &v) in px.iter().enumerate() {
                chw[ch * n + i] = v;
            }
        }
        Ok(Tensor::from_vec(chw, (1, 3, self.size, self.size), device)?)
    }
}

/// The Qwen3.6 processor's Candle tensors.
pub trait Qwen35Preprocess {
    /// Preprocess raw interleaved RGB8 `pixels` (`width*height*3` bytes) → `(pixel_values,
    /// grid_thw)` on `device` ([`Qwen35ImageProcessor::image_patches`]).
    fn preprocess(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<(Tensor, Vec<[i32; 3]>)>;

    /// Preprocess a sampled video's decoded RGB8 frames → `(pixel_values_videos,
    /// video_grid_thw)` on `device` ([`Qwen35ImageProcessor::video_patches`]).
    fn preprocess_video(
        &self,
        frames: &[(&[u8], usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, [i32; 3])>;
}

fn patch_tensor(rows: PatchRows, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        rows.values,
        (rows.rows, rows.features),
        device,
    )?)
}

impl Qwen35Preprocess for Qwen35ImageProcessor {
    fn preprocess(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<(Tensor, Vec<[i32; 3]>)> {
        let (rows, grid) = self.image_patches(pixels, width, height)?;
        Ok((patch_tensor(rows, device)?, grid))
    }

    fn preprocess_video(
        &self,
        frames: &[(&[u8], usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, [i32; 3])> {
        let (rows, grid) = self.video_patches(frames)?;
        Ok((patch_tensor(rows, device)?, grid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_f32(a: &Tensor) -> Vec<f32> {
        a.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn preprocess_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "models/testdata/qwen35_preprocess_oracle.json"
        ))
        .unwrap()
    }

    fn arr_f32(j: &serde_json::Value, path: &[&str], k: &str) -> Vec<f32> {
        let mut v = j;
        for p in path {
            v = &v[p];
        }
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn arr_u8(j: &serde_json::Value, path: &[&str], k: &str) -> Vec<u8> {
        let mut v = j;
        for p in path {
            v = &v[p];
        }
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as u8)
            .collect()
    }

    /// `smart_resize` is pure integer math and must reproduce the reference `Qwen2VL.smart_resize`
    /// bit-exact under the production pixel bounds — including the half-to-even round (the 2576→2560
    /// case diverges from round-half-away) and the min/max-pixels rescale branches.
    #[test]
    fn smart_resize_matches_reference() {
        let j = preprocess_oracle();
        let proc = Qwen35ImageProcessor::default(); // production bounds
        for case in j["smart_cases"].as_array().unwrap() {
            let (h, w) = (
                case["h"].as_u64().unwrap() as usize,
                case["w"].as_u64().unwrap() as usize,
            );
            let (rh, rw) = proc.smart_resize(h, w).unwrap();
            assert_eq!(
                (rh, rw),
                (
                    case["rh"].as_u64().unwrap() as usize,
                    case["rw"].as_u64().unwrap() as usize
                ),
                "smart_resize({h},{w})"
            );
        }
    }

    /// `preprocess_video` produces the right `video_grid_thw` (frames folded `temporal_patch_size`
    /// per temporal patch) and feature dim, and the two **distinct** frames of a temporal patch land
    /// at the two temporal slots of each row (unlike an image, which replicates one frame).
    #[test]
    fn preprocess_video_grid_and_temporal_layout() {
        let proc = Qwen35ImageProcessor::default(); // patch 16, merge 2, temporal 2
                                                    // 64x64 frames → grid_h = grid_w = 4 (64/16). With the video pixel budget (t_bar=2,
                                                    // 2·64·64 = 8192 ∈ [4096, max]) the resize is a no-op, so the grid is exact. Two distinct
                                                    // solid frames so the temporal slots are distinguishable.
        let frame0 = [10u8, 20, 30].repeat(64 * 64); // first frame color
        let frame1 = [200u8, 210, 220].repeat(64 * 64); // second frame color
        let frames: Vec<(&[u8], usize, usize)> = vec![(&frame0, 64, 64), (&frame1, 64, 64)];
        let (pixels, grid) = proc.preprocess_video(&frames, &Device::Cpu).unwrap();

        // grid_t = ceil(2 / temporal_patch_size=2) = 1; grid_h = grid_w = 4.
        assert_eq!(grid, [1, 4, 4], "video grid [grid_t, grid_h, grid_w]");
        let feat = 3 * proc.temporal_patch_size * proc.patch_size * proc.patch_size; // C·T·P·P
        assert_eq!(
            pixels.dims(),
            &[(grid[0] * grid[1] * grid[2]) as usize, feat]
        );

        // Inner feature layout per row is (channel, temporal, patch_row, patch_col). The first
        // `P*P` values are channel-0 / temporal-0 (frame 0); the next `P*P` are channel-0 /
        // temporal-1 (frame 1). Frame 0's red = (10/255 - 0.5)/0.5; frame 1's red differs — proving
        // the two distinct frames occupy the temporal axis (an image would replicate one frame here).
        let v = host_f32(&pixels);
        let p2 = proc.patch_size * proc.patch_size;
        let want0 = (10.0f32 / 255.0 - 0.5) / 0.5;
        let want1 = (200.0f32 / 255.0 - 0.5) / 0.5;
        assert!((v[0] - want0).abs() < 1e-4, "temporal slot 0 = frame 0 red");
        assert!(
            (v[p2] - want1).abs() < 1e-4,
            "temporal slot 1 = frame 1 red (distinct frame)"
        );
    }

    /// A frame count not divisible by `temporal_patch_size` is padded by repeating the last frame
    /// (matching `Qwen3VLVideoProcessor`), so `grid_t = ceil(n / temporal_patch_size)`.
    #[test]
    fn preprocess_video_pads_odd_frame_count() {
        let proc = Qwen35ImageProcessor::default();
        let f = [128u8, 128, 128].repeat(64 * 64);
        let frames: Vec<(&[u8], usize, usize)> = vec![(&f, 64, 64), (&f, 64, 64), (&f, 64, 64)];
        let (pixels, grid) = proc.preprocess_video(&frames, &Device::Cpu).unwrap();
        assert_eq!(grid, [2, 4, 4], "3 frames pad to 4 → grid_t = 2");
        let feat = 3 * proc.temporal_patch_size * proc.patch_size * proc.patch_size;
        assert_eq!(pixels.dims()[0], (grid[0] * grid[1] * grid[2]) as usize);
        assert_eq!(pixels.dims()[1], feat);
    }

    #[test]
    fn preprocess_video_rejects_mismatched_frames() {
        let proc = Qwen35ImageProcessor::default();
        let f0 = [0u8, 0, 0].repeat(32 * 32);
        let f1 = [0u8, 0, 0].repeat(16 * 16);
        let frames: Vec<(&[u8], usize, usize)> = vec![(&f0, 32, 32), (&f1, 16, 16)];
        assert!(proc.preprocess_video(&frames, &Device::Cpu).is_err());
        assert!(proc.preprocess_video(&[], &Device::Cpu).is_err());
    }

    /// Patchify layout vs the reference, isolated from the resampler: an already-aligned image (dims a
    /// multiple of patch·merge, pixel count within bounds) is **not** resized, so `pixel_values` is
    /// pure rescale+normalize+reshape. A tight tolerance pins the merge-block patch order, the
    /// (C,T,P,P) inner layout, and the temporal replication.
    #[test]
    fn patchify_layout_matches_reference() {
        let j = preprocess_oracle();
        let proc = Qwen35ImageProcessor {
            min_pixels: j["params"]["min_pixels_small"].as_u64().unwrap() as usize,
            ..Default::default()
        };
        let (h, w) = (
            j["exact"]["h"].as_u64().unwrap() as usize,
            j["exact"]["w"].as_u64().unwrap() as usize,
        );
        let img = arr_u8(&j, &["exact"], "image_u8");
        let (pv, grid) = proc.preprocess(&img, w, h, &Device::Cpu).unwrap();
        assert_eq!(grid, vec![[1, (h / 16) as i32, (w / 16) as i32]]);
        let got = host_f32(&pv);
        let exp = arr_f32(&j, &["exact"], "pixel_values");
        assert_eq!(got.len(), exp.len());
        let md = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "patchify layout vs reference: max abs diff {md}");
    }

    /// End-to-end (genuine resize): the full pipeline (PIL-bicubic `resize_bicubic_u8` + normalize +
    /// patchify) vs the reference fast processor (torchvision bicubic). Matches within the measured
    /// resampler divergence (~2/255 = ~0.016 on the [-1,1] normalized scale) — the one
    /// implementation-defined step; the structural layout is pinned exactly by the test above.
    #[test]
    fn end_to_end_within_resampler_tolerance() {
        let j = preprocess_oracle();
        let proc = Qwen35ImageProcessor {
            min_pixels: j["params"]["min_pixels_small"].as_u64().unwrap() as usize,
            ..Default::default()
        };
        let (h, w) = (
            j["e2e"]["h"].as_u64().unwrap() as usize,
            j["e2e"]["w"].as_u64().unwrap() as usize,
        );
        let img = arr_u8(&j, &["e2e"], "image_u8");
        let (pv, grid) = proc.preprocess(&img, w, h, &Device::Cpu).unwrap();
        assert_eq!(grid, vec![[1, 4, 4]]); // 50x70 -> 64x64 -> grid 4x4
        let got = host_f32(&pv);
        let exp = arr_f32(&j, &["e2e"], "pixel_values");
        let md = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 0.02,
            "e2e vs reference (resampler tol): max abs diff {md}"
        );
    }

    #[test]
    fn preprocess_normalizes_to_minus_one_one() {
        // 384x384 so no resize; mean/std 0.5 maps 0->-1, 255->1, 128->~0.004. NCHW: channel-major.
        let pixels = [0u8, 128, 255].repeat(384 * 384);
        let out = SiglipImageProcessor::default()
            .preprocess(&pixels, 384, 384, &Device::Cpu)
            .unwrap();
        assert_eq!(out.dims(), &[1, 3, 384, 384]);
        let v = host_f32(&out);
        let n = 384 * 384;
        // All of channel 0 came from the 0-valued R bytes -> -1; channel 2 from 255 -> 1.
        assert_eq!(v[0], -1.0);
        assert!((v[n] - 0.003_921_628).abs() < 1e-6); // channel 1 (128)
        assert_eq!(v[2 * n], 1.0); // channel 2 (255)
    }

    #[test]
    fn preprocess_rejects_bad_buffer() {
        assert!(SiglipImageProcessor::default()
            .preprocess(&[0u8; 3], 2, 2, &Device::Cpu)
            .is_err());
    }

    #[test]
    fn resize_uniform_image_is_uniform() {
        // A solid color resized stays that exact color everywhere (renormalized weights sum to 1).
        let src = [200u8, 100, 50].repeat(64 * 48); // width=64, height=48
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
}
