//! Dataset preparation for training (sc-5165) — the candle twin of `mlx_gen::train::dataset`.
//!
//! Unlike the MLX harness (which operates on a pre-decoded `Image` so the core crate stays
//! decode-free), candle-gen links the `image` crate, so this module owns the full file → tensor path:
//! decode → center-crop to a square → resize to the bucketed edge (Lanczos, matching the Python
//! kernel's `Image.LANCZOS`) → normalize to `[-1, 1]` as a VAE-input tensor `[1, 3, edge, edge]`
//! (RGB, channel-first). Resolution is bucketed to a multiple of 32 (the latent grid must tile
//! cleanly), mirroring the Python `bucket_resolution`.

use std::path::Path;

use candle_core::{Device, Tensor};
use image::imageops::FilterType;

use crate::{CandleError, Result};

/// Floor `resolution` to a multiple of 32 (the training bucket). `0` → the `512` default; otherwise
/// `(res/32)*32` with a 32-px floor so a tiny nonzero input never collapses to 0. Mirrors the Python
/// `bucket_resolution` (and the MLX twin).
pub fn bucket_resolution(resolution: u32) -> u32 {
    if resolution == 0 {
        return 512;
    }
    ((resolution / 32) * 32).max(32)
}

/// The bucketed training edges of `config` (epic 2123 sc-2127) — one per
/// [`TrainingConfig::training_buckets`](gen_core::TrainingConfig::training_buckets) entry, in order,
/// each floored by [`bucket_resolution`]. With no `resolution_buckets` this is exactly
/// `[bucket_resolution(config.resolution)]`, the single edge every trainer used before buckets
/// existed. A trainer caches each item once per edge (item-major) and walks the cache through a
/// [`gen_core::BucketSchedule`].
pub fn bucket_edges(config: &gen_core::TrainingConfig) -> Vec<u32> {
    config
        .training_buckets()
        .iter()
        .map(|b| bucket_resolution(b.resolution))
        .collect()
}

/// A decoded, center-cropped square training image (sc-2127) — decode a dataset image ONCE with
/// [`decode_square`], then build one tensor per bucket edge with [`square_image_tensor`].
pub type SquareImage = image::RgbImage;

/// Decode `path` and center-crop it to its largest centered square — the per-item half of
/// [`load_image_tensor`], done once per item however many bucket edges it is trained at.
pub fn decode_square(path: &Path) -> Result<SquareImage> {
    let img = image::open(path)
        .map_err(|e| CandleError::Msg(format!("open image {}: {e}", path.display())))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let side = w.min(h);
    let (x0, y0) = ((w - side) / 2, (h - side) / 2);
    Ok(image::imageops::crop_imm(&img, x0, y0, side, side).to_image())
}

/// Resize a [`decode_square`] image to `edge`×`edge` (Lanczos) and return the VAE-input tensor
/// `[1, 3, edge, edge]` (RGB, channel-first) normalized to `[-1, 1]` on `device` — the per-edge half
/// of [`load_image_tensor`].
pub fn square_image_tensor(square: &SquareImage, edge: u32, device: &Device) -> Result<Tensor> {
    let resized = image::imageops::resize(square, edge, edge, FilterType::Lanczos3);
    let (ew, eh) = (edge as usize, edge as usize);
    let mut data = vec![0f32; 3 * eh * ew];
    for (x, y, px) in resized.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for c in 0..3 {
            // channel-first [3, H, W]; RGB; [-1, 1].
            data[c * eh * ew + y * ew + x] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, eh, ew), &Device::Cpu)?.to_device(device)?)
}

/// Decode `path`, center-crop to its largest centered square, resize to `edge`×`edge` (Lanczos), and
/// return a VAE-input tensor `[1, 3, edge, edge]` (RGB, channel-first) normalized to `[-1, 1]` on
/// `device` — exactly the Python `_load_training_image` + `_image_to_tensor` pipeline
/// (`array/127.5 - 1.0`, `permute(2,0,1).unsqueeze(0)`). Equal to
/// [`square_image_tensor`]`(`[`decode_square`]`(path)?, edge, device)`.
pub fn load_image_tensor(path: &Path, edge: u32, device: &Device) -> Result<Tensor> {
    square_image_tensor(&decode_square(path)?, edge, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_once_then_per_edge_tensors_match_the_one_shot_loader() {
        // sc-2127: the split (decode + crop once, resize per bucket edge) is the same pixels as the
        // one-shot loader at every edge.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.png");
        let img =
            image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([x as u8, y as u8, (x ^ y) as u8]));
        img.save(&path).unwrap();
        let square = decode_square(&path).unwrap();
        assert_eq!(square.dimensions(), (64, 64));
        // Center crop of a 96-wide image keeps columns 16..80.
        assert_eq!(square.get_pixel(0, 0).0, [16, 0, 16]);
        assert_eq!(square.get_pixel(63, 5).0, [79, 5, 79 ^ 5]);
        for edge in [32u32, 64] {
            let split = square_image_tensor(&square, edge, &Device::Cpu).unwrap();
            let once = load_image_tensor(&path, edge, &Device::Cpu).unwrap();
            assert_eq!(split.dims(), &[1, 3, edge as usize, edge as usize]);
            let a: Vec<f32> = split.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = once.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(a, b, "edge {edge}");
        }
    }

    #[test]
    fn bucket_edges_floor_every_bucket_and_default_to_the_legacy_edge() {
        let mut cfg = gen_core::TrainingConfig {
            resolution: 1000,
            ..Default::default()
        };
        assert_eq!(bucket_edges(&cfg), vec![992]);
        cfg.resolution_buckets = vec![
            gen_core::ResolutionBucket {
                resolution: 512,
                repeats: 16,
            },
            gen_core::ResolutionBucket {
                resolution: 770,
                repeats: 4,
            },
            gen_core::ResolutionBucket {
                resolution: 1024,
                repeats: 1,
            },
        ];
        assert_eq!(bucket_edges(&cfg), vec![512, 768, 1024]);
    }

    #[test]
    fn bucket_floors_to_multiple_of_32() {
        assert_eq!(bucket_resolution(1024), 1024);
        assert_eq!(bucket_resolution(1000), 992);
        assert_eq!(bucket_resolution(1023), 992);
        assert_eq!(bucket_resolution(0), 512);
        assert_eq!(bucket_resolution(16), 32);
        assert_eq!(bucket_resolution(512), 512);
    }
}
