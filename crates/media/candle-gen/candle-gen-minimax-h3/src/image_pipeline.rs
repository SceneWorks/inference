//! MiniMax-H3-Image **text-to-image** host pipeline on candle (SW-7091): the `[text | image]`
//! packed layout, the seeded initial latent, the fixed six-step velocity-form sampler, and the
//! decode-side pixel handling.
//!
//! Every piece is H3's own, at `T = 1`, rather than a family-local copy (E29):
//!
//! * the rotary grid is [`crate::dit::positions`] — text rows at `t = row`, the image's rows at
//!   `t = num_text_tokens` over [`frame_grid`] (`build_image_packed_sequence`,
//!   `modular_pipelines/minimax_h3_image/before_denoise.py`);
//! * the latent is drawn `[1, 24, 1, H, W]` and packed with
//!   [`crate::pipeline::patchify_video_latents`] (`patchify_video_latents`, same file);
//! * the sigma grid and the step coefficient are `gen_core_h3`'s
//!   [`SigmaSchedule::h3_image_distilled`] and [`SigmaSchedule::velocity_delta`] — the request
//!   contract is `gen_core::h3_image`;
//! * the DiT is [`crate::dit::ImageDit`], the H3 block stack with per-step output heads.
//!
//! # The sampler is the velocity form, not H3 video's two-source form
//!
//! `scheduler/scheduler_config.json` sets `euler_update_form = "velocity"`, so every step is
//! `x ← x + (σᵢ − σᵢ₊₁) · v` in float32, over **only** the generated rows (there are no reference
//! rows on `text2image`). The model is evaluated once per step — the checkpoint is
//! guidance-distilled — and step `i` projects through `pdd_proj_out.{i}`.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::seed::seeded_normal_vec;
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_schedule::SigmaSchedule;
use rand::SeedableRng;

use crate::config::LATENT_CHANNELS;
use crate::denoise::{TEXT_TAG, VIDEO_TAG};
use crate::dit::positions::{frame_grid, text_position_ids};
use crate::pipeline::{patchify_video_latents, unpatchify_video_rows, PIXEL_MEAN, PIXEL_STD};

/// The modality tag an image row carries — `gen_core::h3_image::IMAGE_TAG`, MiniMax-H3's video
/// slot of the AdaLN table.
pub const IMAGE_TAG: u32 = candle_gen::gen_core::h3_image::IMAGE_TAG as u32;
const _: () = assert!(IMAGE_TAG == VIDEO_TAG);
const _: () = assert!(candle_gen::gen_core::h3_image::TEXT_TAG as u32 == TEXT_TAG);

/// The packed `[text | generated image]` sequence of one `text2image` request.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageLayout {
    num_text_tokens: usize,
    latent_height: usize,
    latent_width: usize,
    patch: [usize; 3],
    position_ids: Vec<[f64; 3]>,
    token_tags: Vec<u32>,
    text_indices: Vec<u32>,
    image_indices: Vec<u32>,
}

impl ImageLayout {
    /// `build_image_packed_sequence(text_token_tags, [], latent_h, latent_w, patch, image_tag)`:
    /// `num_text_tokens` text rows first (rotary `t = 0..L`, `h = w = 0`, tag text), then the
    /// generated image as ONE rotary frame at `t = L` over the aspect-normalized `(h, w)` grid
    /// (tag image).
    pub fn text_to_image(
        num_text_tokens: usize,
        latent_height: usize,
        latent_width: usize,
        patch: [usize; 3],
    ) -> Result<Self> {
        if num_text_tokens == 0 {
            return Err(CandleError::Msg(
                "minimax-h3-image layout: the prompt encoded to no tokens".into(),
            ));
        }
        if patch[0] != 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image layout: an image is one latent frame, so the temporal patch must \
                 be 1, got {patch:?}"
            )));
        }
        let (grid, _) = frame_grid(latent_height, latent_width, patch[1], patch[2])?;
        let mut position_ids = text_position_ids(num_text_tokens)?;
        let t = num_text_tokens as f64;
        position_ids.extend(grid.iter().map(|&[h, w]| [t, h, w]));
        let seq = position_ids.len();
        let mut token_tags = vec![TEXT_TAG; num_text_tokens];
        token_tags.resize(seq, IMAGE_TAG);
        Ok(Self {
            num_text_tokens,
            latent_height,
            latent_width,
            patch,
            position_ids,
            token_tags,
            text_indices: (0..num_text_tokens as u32).collect(),
            image_indices: (num_text_tokens as u32..seq as u32).collect(),
        })
    }

    /// Text rows.
    pub fn num_text_tokens(&self) -> usize {
        self.num_text_tokens
    }

    /// Image rows: `(latent_height / ph) · (latent_width / pw)`.
    pub fn num_image_rows(&self) -> usize {
        self.image_indices.len()
    }

    /// Text + image rows.
    pub fn seq_len(&self) -> usize {
        self.position_ids.len()
    }

    /// The latent grid.
    pub fn latent_hw(&self) -> (usize, usize) {
        (self.latent_height, self.latent_width)
    }

    /// The DiT patch.
    pub fn patch(&self) -> [usize; 3] {
        self.patch
    }

    /// The `(t, h, w)` rotary coordinate of every row.
    pub fn position_ids(&self) -> &[[f64; 3]] {
        &self.position_ids
    }

    /// The per-row modality tag (text 1, image 0).
    pub fn token_tags(&self) -> &[u32] {
        &self.token_tags
    }

    /// Sequence positions of the text rows.
    pub fn text_indices(&self) -> &[u32] {
        &self.text_indices
    }

    /// Sequence positions of the image rows.
    pub fn image_indices(&self) -> &[u32] {
        &self.image_indices
    }
}

/// The latent grid of a pixel canvas — `gen_core::h3_image::latent_grid` (the VAE's 16x spatial
/// compression of a 32-multiple canvas), as `usize` for the tensor shapes.
pub fn latent_grid(canvas_width: u32, canvas_height: u32) -> Result<(usize, usize)> {
    let (h, w) = candle_gen::gen_core::h3_image::latent_grid(canvas_width, canvas_height)
        .map_err(|e| CandleError::Msg(e.to_string()))?;
    Ok((h as usize, w as usize))
}

/// The seeded initial latent of the generated image, packed into rows: drawn as
/// `[1, 24, 1, H, W]` float32 on the host stream (the same `StdRng` + `seeded_normal_vec` draw
/// H3 video uses), then patchified — `[1, rows, 96]`.
pub fn initial_image_rows(
    latent_height: usize,
    latent_width: usize,
    patch: [usize; 3],
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let shape = (1usize, LATENT_CHANNELS, 1usize, latent_height, latent_width);
    let n = LATENT_CHANNELS * latent_height * latent_width;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let latents = Tensor::from_vec(seeded_normal_vec(&mut rng, n), shape, device)?;
    patchify_video_latents(&latents, patch)
}

/// The packed image rows back to the VAE's `[1, 24, H, W]` latent.
pub fn unpack_image_rows(rows: &Tensor, layout: &ImageLayout) -> Result<Tensor> {
    let (h, w) = layout.latent_hw();
    let latents = unpatchify_video_rows(rows, LATENT_CHANNELS, 1, h, w, layout.patch())?;
    Ok(latents.squeeze(2)?)
}

/// The velocity model one denoise step evaluates: `(step, image_rows) → velocity`.
pub trait ImageVelocity {
    /// Evaluate the model at sampling step `step` (which selects the output head).
    fn forward(&mut self, step: usize, image_rows: &Tensor) -> Result<Tensor>;
}

/// **The six-step loop**: one forward per step, `x ← x + (σᵢ − σᵢ₊₁) · v` in float32.
///
/// Cancellation is checked before every step and the device synchronized after it, so a cancel
/// lands within one step and `on_step(completed)` reports work the GPU has actually done.
pub fn denoise_image(
    model: &mut dyn ImageVelocity,
    schedule: &SigmaSchedule,
    initial_rows: &Tensor,
    device: &Device,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Tensor> {
    let mut x = initial_rows.to_dtype(DType::F32)?;
    for step in 0..schedule.num_evals() {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let delta = schedule
            .velocity_delta(step)
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let velocity = model.forward(step, &x)?;
        if velocity.dims() != x.dims() {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image denoise: step {step} returned a {:?} velocity for {:?} rows",
                velocity.dims(),
                x.dims()
            )));
        }
        x = (x + (velocity.to_dtype(DType::F32)? * f64::from(delta))?)?;
        device.synchronize()?;
        on_step(step + 1);
    }
    Ok(x)
}

/// The decoded, ImageNet-normalized `[1, 3, H, W]` RGB to the delivered image, the released
/// pipeline's way (`decoders.py`): revert the normalization, clamp to `[0, 1]`, quantize to 8 bits
/// **by truncation** (`(image * 255).clamp(0, 255).to(uint8)`), then — when the canvas differs
/// from the requested size — resize the 8-bit image to `width x height` with a Lanczos filter.
pub fn decoded_to_image(decoded: &Tensor, width: u32, height: u32) -> Result<Image> {
    let s = decoded.dims();
    if s.len() != 4 || s[0] != 1 || s[1] != 3 {
        return Err(CandleError::Msg(format!(
            "minimax-h3-image decode: expected [1, 3, H, W], got {s:?}"
        )));
    }
    let (h, w) = (s[2], s[3]);
    let dev = decoded.device();
    let mean = Tensor::from_vec(PIXEL_MEAN.to_vec(), (1, 3, 1, 1), dev)?;
    let std = Tensor::from_vec(PIXEL_STD.to_vec(), (1, 3, 1, 1), dev)?;
    let pixels = decoded
        .to_dtype(DType::F32)?
        .broadcast_mul(&std)?
        .broadcast_add(&mean)?
        .clamp(0f32, 1f32)?;
    // Truncation, not rounding: `.to(torch.uint8)` of a non-negative float drops the fraction.
    let bytes = (pixels * 255.0)?
        .clamp(0f32, 255f32)?
        .floor()?
        .to_dtype(DType::U8)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .contiguous()?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;
    let image = Image {
        width: w as u32,
        height: h as u32,
        pixels: bytes,
    };
    if (image.width, image.height) == (width, height) {
        return Ok(image);
    }
    resize_lanczos(&image, width, height)
}

/// Resize an 8-bit RGB image with a Lanczos (a = 3) filter — the `VaeImageProcessor.resize` the
/// released pipeline applies when the canvas is larger than the request. `image`'s Lanczos3 is the
/// same windowed sinc as PIL's `LANCZOS`, not a byte-identical implementation of it.
fn resize_lanczos(image: &Image, width: u32, height: u32) -> Result<Image> {
    let rgb = image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .ok_or_else(|| {
            CandleError::Msg("minimax-h3-image decode: pixel buffer does not match its size".into())
        })?;
    let resized =
        image::imageops::resize(&rgb, width, height, image::imageops::FilterType::Lanczos3);
    Ok(Image {
        width,
        height,
        pixels: resized.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The t2i layout is `[text | image]`: text at `t = row` with `h = w = 0` and tag 1, the image
    /// as one frame at `t = L` over the aspect-normalized grid with tag 0, and the index tensors
    /// partition the sequence.
    #[test]
    fn the_text_to_image_layout_is_text_then_one_image_frame() {
        let l = ImageLayout::text_to_image(3, 4, 8, [1, 2, 2]).unwrap();
        assert_eq!(l.num_text_tokens(), 3);
        assert_eq!(l.num_image_rows(), 2 * 4);
        assert_eq!(l.seq_len(), 11);
        assert_eq!(l.text_indices(), &[0, 1, 2]);
        assert_eq!(l.image_indices(), &(3..11).collect::<Vec<u32>>()[..]);
        assert_eq!(&l.token_tags()[..3], &[1, 1, 1]);
        assert!(l.token_tags()[3..].iter().all(|&t| t == 0));
        assert_eq!(l.position_ids()[1], [1.0, 0.0, 0.0]);
        assert!(l.position_ids()[3..].iter().all(|p| p[0] == 3.0));
        // 4x8 latent: sqrt_area = sqrt(32); the h axis spans less than the w axis.
        let (grid, _) = frame_grid(4, 8, 2, 2).unwrap();
        for (row, g) in l.position_ids()[3..].iter().zip(&grid) {
            assert_eq!([row[1], row[2]], *g);
        }
        assert!(ImageLayout::text_to_image(0, 4, 4, [1, 2, 2]).is_err());
        assert!(ImageLayout::text_to_image(2, 4, 4, [2, 2, 2]).is_err());
        assert!(ImageLayout::text_to_image(2, 3, 4, [1, 2, 2]).is_err());
    }

    /// The canvas → latent arithmetic, and that a 2048² request is a 128² latent of 64² rows.
    #[test]
    fn the_latent_grid_is_the_canvas_over_sixteen() {
        assert_eq!(latent_grid(2048, 2048).unwrap(), (128, 128));
        assert_eq!(latent_grid(512, 1024).unwrap(), (64, 32));
        assert!(latent_grid(2048, 2040).is_err());
        assert!(latent_grid(0, 2048).is_err());
        let l = ImageLayout::text_to_image(10, 128, 128, [1, 2, 2]).unwrap();
        assert_eq!(l.num_image_rows(), 64 * 64);
    }

    /// The initial rows are the patchified `[1, 24, 1, H, W]` draw, and the round trip through
    /// [`unpack_image_rows`] recovers the latent.
    #[test]
    fn the_initial_latent_round_trips_through_the_packed_rows() {
        let d = Device::Cpu;
        let rows = initial_image_rows(4, 6, [1, 2, 2], 7, &d).unwrap();
        assert_eq!(rows.dims(), &[1, 6, 96]);
        let again = initial_image_rows(4, 6, [1, 2, 2], 7, &d).unwrap();
        assert_eq!(
            rows.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            again.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the draw is a pure function of the seed"
        );
        let l = ImageLayout::text_to_image(1, 4, 6, [1, 2, 2]).unwrap();
        let latents = unpack_image_rows(&rows, &l).unwrap();
        assert_eq!(latents.dims(), &[1, 24, 4, 6]);
        let repacked = patchify_video_latents(&latents.unsqueeze(2).unwrap(), [1, 2, 2]).unwrap();
        assert_eq!(
            repacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            rows.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    struct Recorder {
        steps: Vec<usize>,
    }

    impl ImageVelocity for Recorder {
        fn forward(&mut self, step: usize, rows: &Tensor) -> Result<Tensor> {
            self.steps.push(step);
            Ok(Tensor::ones_like(rows)?)
        }
    }

    /// The loop evaluates each of the six steps once, in order (so step `i` reaches head `i`), and
    /// a unit velocity moves the rows by exactly the whole grid, Σ(σᵢ − σᵢ₊₁) = 1. A host-side
    /// recorder stands in for the DiT; no model runs.
    #[test]
    fn the_loop_runs_each_step_once_and_steps_by_the_velocity_form() {
        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image_distilled().unwrap();
        let x0 = Tensor::zeros((1, 2, 96), DType::F32, &d).unwrap();
        let mut model = Recorder { steps: Vec::new() };
        let mut seen = Vec::new();
        let out = denoise_image(
            &mut model,
            &schedule,
            &x0,
            &d,
            &CancelFlag::default(),
            &mut |n| seen.push(n),
        )
        .unwrap();
        assert_eq!(model.steps, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(seen, vec![1, 2, 3, 4, 5, 6]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-6), "{:?}", &v[..4]);

        let cancel = CancelFlag::default();
        cancel.cancel();
        let mut model = Recorder { steps: Vec::new() };
        assert!(matches!(
            denoise_image(&mut model, &schedule, &x0, &d, &cancel, &mut |_| {}),
            Err(CandleError::Canceled)
        ));
        assert!(model.steps.is_empty());
    }

    /// Delivery: ImageNet revert, clamp, truncating 8-bit quantize, and a resize only when the
    /// canvas differs from the request.
    #[test]
    fn decoded_pixels_are_reverted_truncated_and_resized_back() {
        let d = Device::Cpu;
        // Normalized value that reverts to exactly 0.5 in every channel: (0.5 - mean) / std.
        let vals: Vec<f32> = (0..3)
            .flat_map(|c| std::iter::repeat_n((0.5 - PIXEL_MEAN[c]) / PIXEL_STD[c], 4 * 6))
            .collect();
        let decoded = Tensor::from_vec(vals, (1, 3, 4, 6), &d).unwrap();
        let img = decoded_to_image(&decoded, 6, 4).unwrap();
        assert_eq!((img.width, img.height), (6, 4));
        assert_eq!(img.pixels.len(), 6 * 4 * 3);
        // 0.5 * 255 = 127.5 -> truncated to 127 (rounding would give 128).
        assert!(
            img.pixels.iter().all(|&p| p == 127),
            "{:?}",
            &img.pixels[..6]
        );
        let small = decoded_to_image(&decoded, 3, 2).unwrap();
        assert_eq!((small.width, small.height), (3, 2));
        assert_eq!(small.pixels.len(), 3 * 2 * 3);
    }
}
