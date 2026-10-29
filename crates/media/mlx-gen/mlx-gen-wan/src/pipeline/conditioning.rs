//! I2V / TI2V image preprocessing, masks and the keyframe latent.

use super::*;

/// `[d0, d1, ...]` → `[1, d0, d1, ...]` (prepend a batch axis).
pub(super) fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

// ===========================================================================================
// I2V-14B channel-concat conditioning (port of `generate_wan.py`'s `is_i2v_channel_concat` setup)
// ===========================================================================================

/// Python `round()` — round half to **even** (banker's rounding), matching `round(img.width * scale)`
/// in the reference's image preprocessing. (Rust `f64::round` rounds half away from zero, which would
/// differ on exact `.5` derived sizes.)
pub(super) fn py_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    // Round up on frac > 0.5, or on an exact tie (frac == 0.5) when `floor` is odd (→ even).
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    (if round_up { floor + 1.0 } else { floor }) as usize
}

/// Preprocess an I2V conditioning image to `[3, height, width]` f32 in `[-1, 1]` (CHW), matching the
/// reference's inline pipeline: **cover-fit** LANCZOS resize (`scale = max(W/iw, H/ih)`, new dims
/// `round(·)`), **center-crop** to the target, then `px/255·2 − 1`. The resize is the core PIL-exact
/// fixed-point integer LANCZOS ([`resize_lanczos_u8`]), so it's bit-identical to PIL's `Image.LANCZOS`.
pub fn preprocess_i2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len()
        != mlx_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(Error::Msg(format!(
            "i2v image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit: scale so the image covers the target, then round to integer dims (PIL `round`).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = py_round(iw as f64 * scale).max(tw);
    let nh = py_round(ih as f64 * scale).max(th);
    let resized: Vec<f32> = if (nh, nw) == (ih, iw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, nh, nw)?
    };
    // Center-crop the (integer-valued) resized HWC buffer to (th, tw), then normalize → CHW [-1,1].
    let x1 = (nw - tw) / 2;
    let y1 = (nh - th) / 2;
    let mut chw = vec![0f32; 3 * th * tw];
    let plane = th * tw;
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Array::from_slice(&chw, &[3, th as i32, tw as i32]))
}

/// The I2V-14B 4-channel temporal mask `[4, T_lat, h_lat, w_lat]` (f32): `1.0` for the first latent
/// temporal frame (all 4 channels, all spatial), `0.0` elsewhere. The reference builds this via a
/// `ones`/`zeros` → `repeat(first,4)` → `reshape(·,T_lat,4,·,·)` → `transpose` dance over the
/// `[1, F, h_lat, w_lat]` per-frame mask (first frame 1, rest 0); the result is exactly this pattern
/// (the per-frame mask collapses to "the first 4 of `F+3` temporal slots", which is latent frame 0).
pub(super) fn build_i2v_mask(t_lat: usize, h_lat: usize, w_lat: usize) -> Array {
    let plane = h_lat * w_lat;
    let mut data = vec![0f32; 4 * t_lat * plane];
    for c in 0..4 {
        let base = c * t_lat * plane; // temporal index 0 of channel c
        for p in 0..plane {
            data[base + p] = 1.0;
        }
    }
    Array::from_slice(&data, &[4, t_lat as i32, h_lat as i32, w_lat as i32])
}

/// Build the I2V-14B channel-concat conditioning `y = [mask(4), z_video(16)]` → `[20, T_lat, h_lat,
/// w_lat]` (f32). Port of `generate_wan.py`'s `is_i2v_channel_concat` branch: a conditioning video
/// (first frame = the preprocessed image, the remaining `frames−1` zero) is encoded by the 2.1 z16
/// `WanVae` → `z_video [16, T_lat, …]`, and concatenated under the temporal mask. `vae` must carry
/// encoder weights. The result is `Some(y)` fed to [`denoise_moe`].
pub fn build_i2v_y(
    vae: &WanVae,
    image: &Image,
    frames: usize,
    height: u32,
    width: u32,
    vae_stride: (usize, usize, usize),
) -> Result<Array> {
    let (h, w) = (height as i32, width as i32);
    // `frames == 0` would make both the `frames − 1` zero-pad count (negative i32) and the `t_lat`
    // subtraction (usize underflow) bogus; reject it up front (F-007).
    let frames_minus_1 = frames
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("wan build_i2v_y: frames must be >= 1".to_string()))?;
    // Conditioning video [3, F, H, W]: first frame = image, rest zeros.
    let first = preprocess_i2v_image(image, width, height)?.reshape(&[3, 1, h, w])?;
    let rest = Array::zeros::<f32>(&[3, frames_minus_1 as i32, h, w])?;
    let video = concatenate_axis(&[&first, &rest], 1)?; // [3, F, H, W]

    // VAE-encode → [1, 16, T_lat, h_lat, w_lat], drop the batch axis → [16, T_lat, h_lat, w_lat].
    let z_video = vae.encode(&video.reshape(&[1, 3, frames as i32, h, w])?)?;
    let z_video = z_video.reshape(&z_video.shape()[1..])?;

    let t_lat = frames_minus_1 / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    let mask = build_i2v_mask(t_lat, h_lat, w_lat);

    Ok(concatenate_axis(&[&mask, &z_video], 0)?)
}

// ===========================================================================================
// TI2V-5B mask-blend conditioning (port of `generate_wan.py`'s `is_i2v_mask_blend` setup + i2v_utils)
// ===========================================================================================

/// Preprocess a TI2V conditioning image to **channels-last** `[1, 1, height, width, 3]` f32 in
/// `[-1, 1]` (batch + temporal dims), the layout the z48 [`Wan22Vae::encode`] consumes. Reuses the
/// PIL-exact cover-fit LANCZOS + center-crop pipeline of [`preprocess_i2v_image`] (which returns CHW),
/// then moves channels last + adds the batch/temporal axes. Mirrors `i2v_utils.preprocess_image`.
pub fn preprocess_ti2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let chw = preprocess_i2v_image(image, width, height)?; // [3, H, W]
    Ok(chw
        .transpose_axes(&[1, 2, 0])?
        .expand_dims(0)?
        .expand_dims(0)?) // [1, 1, H, W, 3]
}

/// Build the TI2V-5B mask-blend tensors (port of `i2v_utils.build_i2v_mask`, generalized to the
/// Wan-native multi-keyframe pins of epic 3040 and to per-pin **strength**, sc-19571):
///  - `mask` `[z, T_lat, h_lat, w_lat]` (f32): `1 − strength` at each pinned latent temporal frame
///    (all channels/spatial), `1.0` elsewhere. `strength = 1.0` fully freezes the frame (mask `0`,
///    the historical hard pin); `strength = 0.0` leaves it free to denoise; anything between is a
///    partial pin, which is the whole point of the knob.
///  - `mask_tokens` `[1, L]` (f32): the channel-0 mask subsampled to the patch grid (`1 − strength`
///    for each pinned frame's tokens, `1.0` for the rest), `L` = the DiT patch-token count
///    `(T_lat/pt)·(h_lat/ph)·(w_lat/pw)`. Token order is temporal-slowest (matching
///    [`crate::patchify::patchify`]).
///
/// `pins` are `(latent_frame, strength)` pairs: first_last_frame is `[(0, s0), (t_lat-1, s1)]`, a
/// single `Reference` image is `[(0, s)]`. Latent indices `>= t_lat` are ignored (the caller
/// validates); the same weight drives both the clean-latent blend and the per-token diffusion
/// timestep, so one number controls the whole pin. **Byte-for-byte the same construction as
/// candle's `candle_gen_wan::pipeline::build_ti2v_mask`** — the two lanes must not diverge here.
pub fn build_ti2v_mask(
    pins: &[(usize, f32)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch_size: (usize, usize, usize),
) -> (Array, Array) {
    let plane = h_lat * w_lat;
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        for &(t, strength) in pins.iter().filter(|&&(t, _)| t < t_lat) {
            let base = (c * t_lat + t) * plane;
            mask[base..base + plane].fill(1.0 - strength);
        }
    }
    let mask = Array::from_slice(
        &mask,
        &[z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32],
    );

    let (pt, ph, pw) = patch_size;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tok = vec![1f32; tg * hg * wg];
    for &(t, strength) in pins {
        let tg_idx = t / pt;
        if tg_idx < tg {
            tok[(tg_idx * hg * wg)..((tg_idx + 1) * hg * wg)].fill(1.0 - strength);
        }
    }
    let mask_tokens = Array::from_slice(&tok, &[1, (tg * hg * wg) as i32]);
    (mask, mask_tokens)
}

/// Scatter per-keyframe latents into a single `[z, T_lat, h, w]` clean latent for the multi-keyframe
/// mask-blend (epic 3040): each `(z_k, idx)` (with `z_k` shaped `[z, 1, h, w]`) is placed at temporal
/// frame `idx`; every other frame is zeros (those frames have mask `1`, so the zero is never read). The
/// resulting latent feeds [`ti2v_blend_init`] + [`denoise_ti2v`] as the `z_img` per-frame conditioning.
pub fn build_ti2v_keyframe_z(
    frames: &[(Array, usize)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
) -> Result<Array> {
    let zero = Array::zeros::<f32>(&[z_dim as i32, 1, h_lat as i32, w_lat as i32])?;
    let mut slices: Vec<Array> = (0..t_lat).map(|_| zero.clone()).collect();
    for (z_k, idx) in frames {
        if *idx < t_lat {
            slices[*idx] = z_k.clone();
        }
    }
    let refs: Vec<&Array> = slices.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Blend the encoded image latent with the initial noise for the TI2V start: `latents = (1−mask)·
/// z_img + mask·noise` (port of `generate_wan.py`'s `is_i2v_mask_blend` init). `z_img` is `[z,1,h,w]`
/// (broadcasts over the noise's `T_lat`), `mask`/`noise` are `[z,T_lat,h,w]`.
pub fn ti2v_blend_init(z_img: &Array, mask: &Array, noise: &Array) -> Result<Array> {
    let one_minus_mask = subtract(scalar(1.0), mask)?;
    Ok(add(
        &multiply(&one_minus_mask, z_img)?,
        &multiply(mask, noise)?,
    )?)
}
