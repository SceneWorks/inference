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

use gen_core_train::canvas::{training_canvas, TrainingCanvas};

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
    let img = gen_core::__model_input_reader!(
        image::open(path),
        "crates/media/candle-gen/candle-gen/src/train/dataset.rs::decode_square::image::open@2345",
        Reference,
        path,
        |__model_input_path| image::open(__model_input_path)
    )
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
    rgb_to_tensor(&resized, device)
}

/// Decode `path` whole (RGB8) — the per-item half of the aspect-preserving canvas path: decoded
/// once, then cut and resized onto each bucket edge's [`TrainingCanvas`] by [`canvas_image_tensor`].
pub fn decode_rgb(path: &Path) -> Result<image::RgbImage> {
    Ok(gen_core::__model_input_reader!(
        image::open(path),
        "crates/media/candle-gen/candle-gen/src/train/dataset.rs::decode_rgb::image::open@2345",
        Reference,
        path,
        |__model_input_path| image::open(__model_input_path)
    )
    .map_err(|e| CandleError::Msg(format!("open image {}: {e}", path.display())))?
    .to_rgb8())
}

/// The training canvas of a decoded `image` at bucket `edge` for a family whose canvas sides are
/// multiples of `stride` and at most `max_side` ([`gen_core_train::canvas::training_canvas`]).
pub fn image_canvas(
    image: &image::RgbImage,
    edge: u32,
    stride: u32,
    max_side: Option<u32>,
) -> TrainingCanvas {
    training_canvas(image.dimensions(), edge, stride, max_side)
}

/// Cut `image` to `canvas.crop`, resize it to the canvas (Lanczos) and return the VAE-input tensor
/// `[1, 3, height, width]` (RGB, channel-first) normalized to `[-1, 1]` on `device` — the per-edge
/// half of the aspect-preserving canvas path (the whole frame, never centre-cropped to a square).
pub fn canvas_image_tensor(
    image: &image::RgbImage,
    canvas: &TrainingCanvas,
    device: &Device,
) -> Result<Tensor> {
    let c = canvas.crop;
    let full = (c.x, c.y, c.width, c.height) == (0, 0, image.width(), image.height());
    let resized = if full {
        image::imageops::resize(image, canvas.width, canvas.height, FilterType::Lanczos3)
    } else {
        let cut = image::imageops::crop_imm(image, c.x, c.y, c.width, c.height).to_image();
        image::imageops::resize(&cut, canvas.width, canvas.height, FilterType::Lanczos3)
    };
    rgb_to_tensor(&resized, device)
}

/// An RGB8 image → `[1, 3, H, W]` channel-first `[-1, 1]` on `device`.
fn rgb_to_tensor(img: &image::RgbImage, device: &Device) -> Result<Tensor> {
    let (ew, eh) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 3 * eh * ew];
    for (x, y, px) in img.enumerate_pixels() {
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
    fn a_portrait_trains_whole_at_its_own_aspect() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.png");
        let img = image::RgbImage::from_fn(60, 90, |x, y| image::Rgb([x as u8, y as u8, 7]));
        img.save(&path).unwrap();
        let rgb = decode_rgb(&path).unwrap();
        assert_eq!(rgb.dimensions(), (60, 90));
        // Edge 64 (area 4096 < 5400): scaled to ~52×78 at 2:3, stride-floored to 48×64.
        let canvas = image_canvas(&rgb, 64, 16, None);
        assert_eq!(
            canvas.crop,
            gen_core_train::subject_mask::CropBox::full(60, 90)
        );
        assert_eq!(canvas.size(), (48, 64));
        let t = canvas_image_tensor(&rgb, &canvas, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 64, 48]);
        // A strip beyond 4:1 is centre-cropped first; the tensor follows the crop's canvas.
        let strip = image::RgbImage::from_fn(200, 20, |x, _| image::Rgb([x as u8, 0, 0]));
        let canvas = image_canvas(&strip, 64, 16, None);
        assert_eq!((canvas.crop.x, canvas.crop.width), (60, 80));
        let t = canvas_image_tensor(&strip, &canvas, &Device::Cpu).unwrap();
        assert_eq!(
            t.dims(),
            &[1, 3, canvas.height as usize, canvas.width as usize]
        );
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
