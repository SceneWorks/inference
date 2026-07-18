//! Mochi 1 text-to-video pipeline — the flow-match **true-CFG** denoise loop + AsymmVAE decode.
//!
//! [`denoise`] runs `MochiPipeline`'s sampling loop: at each step the seeded latent is doubled into a
//! `[neg, pos]` CFG batch, the AsymmDiT predicts the velocity for both branches, true CFG recombines
//! them (`uncond + g·(cond − uncond)`), and a 1st-order FlowMatchEuler step advances the latent.
//! `req.cancel` is honored at the top of every step and each step forces an `mlx_rs::eval`, so
//! cancellation lands promptly and the streamed [`Progress::Step`] reflects real (materialized) work
//! rather than a lazily-deferred graph.
//!
//! The 3-D RoPE / positions for the visual grid are built **inside** [`MochiTransformer3DModel::forward`]
//! from the geometry (`f`/`ph`/`pw`), so the pipeline only supplies the latent + conditioning.

use mlx_rs::ops::{add, concatenate_axis, divide, maximum, minimum, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::{CancelFlag, Error, Image, Progress, Result};

use crate::scheduler::{cfg_combine, MochiScheduler};
use crate::transformer::MochiTransformer3DModel;
use crate::vae::{MochiVaeDecoder, DEFAULT_DECODE_CHUNK_FRAMES};

/// `[v]` as a length-1 `Array` (small scalar helper for the uint8 conversion).
fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy: host reads (`as_slice`) return the *physical* buffer, so an
/// array left strided by the `(F,H,W,C)` transpose reads scrambled. sc-12748: delegates to the shared
/// int64-safe [`mlx_gen::array::contiguous`] (Mochi's RGB output stays under `i32::MAX` at its 848×480
/// design point, but this keeps every video model on one materialization helper).
fn contiguous(x: &Array) -> Result<Array> {
    mlx_gen::array::contiguous(x)
}

/// N-step 1st-order FlowMatchEuler **true-CFG** denoise (Mochi is not distilled).
///
/// `init_latents [1, C, F, H, W]` are the seeded init noise (`init_noise_sigma = 1`, used unscaled).
/// `enc [2, L, 4096]` / `enc_mask [2, L]` carry the raw T5 conditioning for the **[neg, pos]** CFG
/// batch. `guidance` is the CFG scale; `shift` the flow-match resolution shift (Mochi = 1.0). Returns
/// the post-loop latents `[1, C, F, H, W]`.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &MochiTransformer3DModel,
    init_latents: &Array,
    enc: &Array,
    enc_mask: &Array,
    steps: usize,
    guidance: f32,
    shift: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut sched = MochiScheduler::new();
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    let mut latents = init_latents.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Cooperative cancel at the top of every step (the per-step `eval` below makes it land).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // latent_model_input = cat([latents, latents]) → [2, C, F, H, W] (the two CFG branches).
        let lmi = concatenate_axis(&[&latents, &latents], 0)?;
        // Same model timestep for both branches (`t.expand(batch)`).
        let timestep = Array::from_slice(&[t, t], &[2]);
        // AsymmDiT velocity for [neg, pos]; CFG recombine runs in f32 (matches the reference).
        let noise_pred = transformer
            .forward(&lmi, enc, &timestep, enc_mask)?
            .as_dtype(Dtype::Float32)?;
        let velocity = cfg_combine(&noise_pred, guidance)?; // [1, C, F, H, W]
        latents = sched.step(&velocity, &latents)?;
        // Force per-step materialization so the next cancel check is meaningful and progress is real.
        mlx_rs::transforms::eval([&latents])?;
        on_progress(Progress::Step {
            current: (i + 1) as u32,
            total: steps as u32,
        });
    }
    Ok(latents)
}

/// De-normalize + AsymmVAE-decode the final latents into `(F, H, W, 3)` uint8 frames.
///
/// The decode is **chunked** ([`MochiVaeDecoder::decode_chunked`]): an untiled decode holds the
/// 128-channel `block_out` stage at full output resolution for the whole clip, which is what gated
/// Mochi to 96 GB-class Macs (sc-12291). Chunking is exact here — not a blended tiling — so the
/// decoded video is unchanged. `cancel` is honored before the decode and once per chunk.
pub fn decode_to_frames(
    vae: &MochiVaeDecoder,
    latents: &Array,
    cancel: &CancelFlag,
) -> Result<Array> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // Drop whatever the denoise loop left in MLX's caching allocator before the decode's own peak
    // (the scail2 precedent — retained cache otherwise stacks on top of the decode high-water).
    mlx_rs::memory::clear_cache();
    let video = vae.decode_chunked(latents, DEFAULT_DECODE_CHUNK_FRAMES, Some(cancel))?;
    to_uint8_frames(&video)
}

/// `(1, 3, F, H, W)` video in ~[-1, 1] → `(F, H, W, 3)` uint8. Mirrors the reference
/// `((x + 1) / 2).clamp(0, 1)·255` with a truncating cast (saturating at 255).
pub fn to_uint8_frames(video: &Array) -> Result<Array> {
    let sh = video.shape(); // (1, 3, F, H, W)
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "mochi to_uint8_frames: batch size must be 1, got {}",
            sh[0]
        )));
    }
    let (c, f, h, w) = (sh[1], sh[2], sh[3], sh[4]);
    let dt = video.dtype();
    let chw = video
        .reshape(&[c, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // (F, H, W, 3)
    let half = divide(
        &add(&chw, &scalar(1.0).as_dtype(dt)?)?,
        &scalar(2.0).as_dtype(dt)?,
    )?;
    let clipped = minimum(
        &maximum(&half, &scalar(0.0).as_dtype(dt)?)?,
        &scalar(1.0).as_dtype(dt)?,
    )?;
    let scaled = multiply(&clipped, &scalar(255.0).as_dtype(dt)?)?;
    contiguous(&scaled.as_dtype(Dtype::Uint8)?)
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame.
pub fn frames_to_images(frames: &Array) -> Result<Vec<Image>> {
    let sh = frames.shape(); // (F, H, W, 3)
    if sh.len() != 4 || sh[3] != 3 {
        return Err(Error::Msg(format!(
            "mochi frames_to_images: expected (F, H, W, 3) uint8, got {sh:?}"
        )));
    }
    let (fr, h, w) = (sh[0] as usize, sh[1] as u32, sh[2] as u32);
    let owned = frames.as_dtype(Dtype::Uint8)?;
    let data = owned.as_slice::<u8>();
    let per = (h as usize) * (w as usize) * 3;
    Ok((0..fr)
        .map(|i| Image {
            width: w,
            height: h,
            pixels: data[i * per..(i + 1) * per].to_vec(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `to_uint8_frames` clips to [0, 255] and lays frames out row-major `(F, H, W, 3)`.
    #[test]
    fn to_uint8_frames_clips_and_scales() {
        // 1 frame, 1×2 pixels, 3 channels: values chosen to hit clip low / mid / high.
        // pixel0 = (-2, -1, 0) → ((x+1)/2) = (-0.5, 0, 0.5) → clip → (0, 0, 127)
        // pixel1 = ( 1,  2, 3) → (1, 1.5, 2)   → clip → (255, 255, 255)
        let v = Array::from_slice(&[-2.0f32, 1.0, -1.0, 2.0, 0.0, 3.0], &[1, 3, 1, 1, 2]);
        let out = to_uint8_frames(&v).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 3]);
        let px: Vec<u8> = out.as_slice::<u8>().to_vec();
        assert_eq!(px, vec![0, 0, 127, 255, 255, 255]);
    }

    /// `frames_to_images` splits `(F, H, W, 3)` into per-frame RGB8 `Image`s.
    #[test]
    fn frames_to_images_splits_per_frame() {
        // 2 frames, 1×1, 3 channels.
        let f = Array::from_slice(&[10u8, 20, 30, 40, 50, 60], &[2, 1, 1, 3]);
        let imgs = frames_to_images(&f).unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].width, 1);
        assert_eq!(imgs[0].height, 1);
        assert_eq!(imgs[0].pixels, vec![10, 20, 30]);
        assert_eq!(imgs[1].pixels, vec![40, 50, 60]);
    }
}
