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

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::{CancelFlag, Error, Image, Progress, Result};

use crate::scheduler::{cfg_combine, MochiScheduler};
use crate::transformer::MochiTransformer3DModel;
use crate::vae::{MochiVaeDecoder, DEFAULT_DECODE_CHUNK_FRAMES};

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
/// `((x + 1) / 2).clamp(0, 1)·255` + round. The body is the shared
/// [`mlx_gen::image::video_bcfhw_to_u8_frames`] (identical to LTX's; sc-27319).
pub fn to_uint8_frames(video: &Array) -> Result<Array> {
    mlx_gen::image::video_bcfhw_to_u8_frames(video, "mochi to_uint8_frames")
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame, via the shared shape-checked, logical-order
/// [`mlx_gen::image::u8_frames_to_images`] (sc-27319).
pub fn frames_to_images(frames: &Array) -> Result<Vec<Image>> {
    mlx_gen::image::u8_frames_to_images(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `to_uint8_frames` clips to [0, 255] and lays frames out row-major `(F, H, W, 3)`.
    #[test]
    fn to_uint8_frames_clips_and_scales() {
        // 1 frame, 1×2 pixels, 3 channels: values chosen to hit clip low / mid / high.
        // pixel0 = (-2, -1, 0) → ((x+1)/2) = (-0.5, 0, 0.5) → nearest-even → (0, 0, 128)
        // pixel1 = ( 1,  2, 3) → (1, 1.5, 2)   → clip → (255, 255, 255)
        let v = Array::from_slice(&[-2.0f32, 1.0, -1.0, 2.0, 0.0, 3.0], &[1, 3, 1, 1, 2]);
        let out = to_uint8_frames(&v).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 3]);
        let px: Vec<u8> = out.as_slice::<u8>().to_vec();
        assert_eq!(px, vec![0, 0, 128, 255, 255, 255]);
    }
}
