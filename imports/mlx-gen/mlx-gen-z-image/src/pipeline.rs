//! Z-Image sampling-pipeline helpers: seeded latent creation, latent unpacking, and the
//! decoded-tensor → [`Image`] conversion — ports of the fork's `ZImageLatentCreator` +
//! `ImageUtil`. The denoise loop that composes these with the transformer
//! ([`crate::transformer`]), scheduler ([`mlx_gen::FlowMatchEuler`]) and VAE ([`crate::vae`])
//! lands once `load()` assembles the model from weights (+ the text encoder).

use mlx_gen::{CancelFlag, Error, FlowMatchEuler, Image, Progress, Result};
use mlx_rs::ops::{add, maximum, minimum, multiply, round};
use mlx_rs::{random, Array};

use crate::ZImageTransformer;

/// Z-Image latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Seeded txt2img latent noise — shape `[16, 1, height/8, width/8]`, f32. Port of
/// `ZImageLatentCreator.create_noise` (`mx.random.normal` with `key(seed)`). The fork casts to
/// the model precision (bf16) when the latents enter the loop; this returns the raw f32 sample
/// so seeded-RNG parity can be checked directly.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        LATENT_CHANNELS,
        1,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Port of `ZImageLatentCreator.unpack_latents`: `[C, 1, H, W]` → `[1, C, H, W]` (add a batch
/// axis, drop the singleton temporal axis) before VAE decode.
pub fn unpack_latents(latents: &Array) -> Result<Array> {
    Ok(latents.expand_dims(0)?.squeeze_axes(&[2])?)
}

/// `cap_feats = encoder_out[0, :num_valid, :]` — drop the batch axis and the padded tail. The
/// text encoder returns `[1, seq, hidden]` (padded to max length); the DiT consumes only the
/// valid caption tokens. (mlx-rs has no slice op, so this is a range-gather.)
pub fn slice_valid(encoder_out: &Array, num_valid: i32) -> Result<Array> {
    let sh = encoder_out.shape();
    let (s, h) = (sh[1], sh[2]);
    let flat = encoder_out.reshape(&[s, h])?;
    let idx = Array::from_slice(&(0..num_valid).collect::<Vec<i32>>(), &[num_valid]);
    Ok(flat.take_axis(&idx, 0)?)
}

/// Flow-match Euler denoise loop with progress + cooperative cancellation: each step predicts the
/// velocity with the DiT and takes an Euler step, emitting a [`Progress::Step`] and checking
/// `cancel` between steps. `latents` is the seeded init (see [`create_noise`]); `cap_feats` is the
/// text-encoder conditioning. Returns the final latents (pre-VAE).
///
/// Mirrors the fork's loop: `timestep = 1 - sigma[t]` (the transformer applies its own
/// `t_scale`), `latents += (sigma[t+1] - sigma[t]) * velocity`.
pub fn denoise_with_progress(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut latents = latents;
    let total = scheduler.num_steps() as u32;
    for t in 0..scheduler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let velocity = transformer.forward(&latents, scheduler.timestep(t), cap_feats)?;
        latents = scheduler.step(&latents, &velocity, t)?;
        on_progress(Progress::Step {
            current: t as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// [`denoise_with_progress`] with no progress callback and no cancellation — the bare loop used
/// by the stage-wise parity tests. Composes the parity-proven transformer + scheduler; full-weights
/// numeric parity is the real-hardware E2E (sc-2352).
pub fn denoise(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
) -> Result<Array> {
    denoise_with_progress(
        transformer,
        scheduler,
        latents,
        cap_feats,
        &CancelFlag::default(),
        &mut |_| {},
    )
}

/// Decoded VAE tensor → RGB8 [`Image`]. Mirrors the fork's `ImageUtil`: denormalize
/// `clip(x/2 + 0.5, 0, 1)`, drop the temporal axis if 5-D, `NCHW → NHWC`, then
/// `(x*255).round()` as `uint8`, taking the first batch element.
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let half = scalar(0.5);
    // denormalize: clip(x*0.5 + 0.5, 0, 1)
    let x = add(&multiply(decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    // drop the singleton temporal axis if present (5-D → 4-D)
    let x = if x.shape().len() == 5 {
        x.squeeze_axes(&[2])?
    } else {
        x
    };
    // NCHW → NHWC
    let x = x.transpose_axes(&[0, 2, 3, 1])?;
    // (x*255).round() to integer pixel values.
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;

    let sh = x.shape();
    let (h, w, c) = (sh[1] as u32, sh[2] as u32, sh[3] as u32);
    let n = (h * w * c) as usize;
    // `transpose_axes` yields a strided view; a raw `as_slice` would read physical (pre-transpose)
    // order. `reshape` re-materializes in C-order, so the slice is logical NHWC. Take batch 0.
    let total: i32 = sh.iter().product();
    let flat = x.reshape(&[total])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w,
        height: h,
        pixels,
    })
}
