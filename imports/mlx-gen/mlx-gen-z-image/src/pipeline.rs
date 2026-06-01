//! Z-Image sampling-pipeline helpers: seeded latent creation, latent unpacking, and the
//! decoded-tensor → [`Image`] conversion — ports of the fork's `ZImageLatentCreator` +
//! `ImageUtil`. The denoise loop that composes these with the transformer
//! ([`crate::transformer`]), scheduler ([`mlx_gen::FlowMatchEuler`]) and VAE ([`crate::vae`])
//! lands once `load()` assembles the model from weights (+ the text encoder).

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{CancelFlag, Error, FlowMatchEuler, Image, Progress, Result};
use mlx_rs::ops::{add, maximum, minimum, multiply, round};
use mlx_rs::{random, Array};

use crate::vae::Vae;
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
/// `start_step` is the first schedule index to run — `0` for txt2img, `init_time_step` for img2img
/// (the fork's `range(init_time_step, num_steps)`). Progress is reported over the steps actually
/// run (`total = num_steps - start_step`).
///
/// Mirrors the fork's loop: `timestep = 1 - sigma[t]` (the transformer applies its own
/// `t_scale`), `latents += (sigma[t+1] - sigma[t]) * velocity`.
pub fn denoise_with_progress(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
    start_step: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut latents = latents;
    let total = (scheduler.num_steps() - start_step) as u32;
    for t in start_step..scheduler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let velocity = transformer.forward(&latents, scheduler.timestep(t), cap_feats)?;
        latents = scheduler.step(&latents, &velocity, t)?;
        on_progress(Progress::Step {
            current: (t - start_step) as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// [`denoise_with_progress`] from step 0, with no progress callback and no cancellation — the bare
/// loop used by the stage-wise parity tests. Composes the parity-proven transformer + scheduler;
/// full-weights numeric parity is the real-hardware E2E (sc-2352).
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
        0,
        &CancelFlag::default(),
        &mut |_| {},
    )
}

/// Resolve the img2img start step (the fork's `Config.init_time_step`): for a reference image with
/// `strength` in `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise `0` (pure txt2img).
/// Higher strength → later start → fewer denoise steps → output stays closer to the init image
/// (the fork's convention).
pub fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// img2img init image → packed clean latents `[16, 1, H/8, W/8]` (f32). Port of the fork's
/// `LatentCreator.encode_image` ∘ `ZImageLatentCreator.pack_latents`: PIL-LANCZOS scale to the
/// target dims, normalize `[0,255] → [-1,1]` as NCHW, VAE-encode (mean → latent space), pack.
pub fn encode_init_latents(
    vae: &Vae,
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, target_width, target_height)?;
    let encoded = vae.encode(&image_nchw)?; // [1, 16, H/8, W/8]
    pack_latents(&encoded)
}

/// Scale an RGB8 init image to `target` dims with PIL LANCZOS (the fork's `scale_to_dimensions`,
/// a no-op when already sized), normalize `[0,255] → [-1,1]`, and lay out as NCHW `[1, 3, H, W]`
/// f32 — the input the VAE encoder expects.
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "init image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // PIL LANCZOS on the uint8 image (no-op when already at target size), matching the fork.
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    // /255 then [-1,1], as NHWC, then transpose to NCHW (the fork's `to_array` convention).
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let nhwc = Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]);
    Ok(nhwc.transpose_axes(&[0, 3, 1, 2])?)
}

/// Port of `ZImageLatentCreator.pack_latents`: VAE-encoder latent `[1, C, H/8, W/8]` (or a 5-D
/// `[1, C, 1, H/8, W/8]`) → `[C, 1, H/8, W/8]`, matching the seeded-noise layout so the two can be
/// blended.
pub fn pack_latents(encoded: &Array) -> Result<Array> {
    let sh = encoded.shape();
    let e = if sh.len() == 5 {
        encoded.reshape(&[sh[0], sh[1], sh[3], sh[4]])? // drop temporal axis
    } else {
        encoded.clone()
    };
    Ok(e.expand_dims(2)?.squeeze_axes(&[0])?)
}

/// Port of `LatentCreator.add_noise_by_interpolation`: `(1 - sigma) * clean + sigma * noise`. The
/// img2img blend that seeds the denoise loop at `sigma = sigmas[init_time_step]`.
pub fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    Ok(add(&multiply(clean, one_minus)?, &multiply(noise, s)?)?)
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
