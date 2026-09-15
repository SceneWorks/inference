//! Pipeline glue for Wan T2V/TI2V: latent geometry, exact image preprocessing, Reference/keyframe
//! masks, deterministic CPU-seeded noise, classifier-free guidance, and frame conversion.

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::gen_core::Image;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::Ti2vProviderVae;

fn py_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    let up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    (if up { floor + 1.0 } else { floor }) as usize
}

/// PIL-exact cover-fit LANCZOS + center crop, returned as `[1,3,1,H,W]` f32 in `[-1,1]` for the
/// z48 encoder. This is the Candle form of mlx-gen-wan's `preprocess_ti2v_image`.
pub fn preprocess_ti2v_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> candle_gen::Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    let expected =
        candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX);
    if image.pixels.len() != expected {
        return Err(candle_gen::CandleError::Msg(format!(
            "wan TI2V image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = py_round(iw as f64 * scale).max(tw);
    let nh = py_round(ih as f64 * scale).max(th);
    let resized = if (nw, nh) == (iw, ih) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        candle_gen::gen_core::imageops::resize_lanczos_u8(&image.pixels, ih, iw, nh, nw)?
    };
    let (x0, y0) = ((nw - tw) / 2, (nh - th) / 2);
    let plane = th * tw;
    let mut chw = vec![0f32; 3 * plane];
    for y in 0..th {
        for x in 0..tw {
            let source = ((y0 + y) * nw + x0 + x) * 3;
            for c in 0..3 {
                chw[c * plane + y * tw + x] = 2.0 * resized[source + c] / 255.0 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, 1, th, tw), device)?)
}

/// Latent dims `(t_lat, h_lat, w_lat)` for `frames × height × width`.
pub fn latent_dims(frames: u32, width: u32, height: u32) -> (usize, usize, usize) {
    let temporal_scale = Ti2vProviderVae::VAE_TILING.temporal_scale as u32;
    let spatial_scale = Ti2vProviderVae::VAE_TILING.spatial_scale as u32;
    let t_lat = (frames - 1) / temporal_scale + 1;
    let h_lat = height / spatial_scale;
    let w_lat = width / spatial_scale;
    (t_lat as usize, h_lat as usize, w_lat as usize)
}

/// Deterministic N(0,1) latent noise `[1, 48, t_lat, h_lat, w_lat]` (f32) — CPU `StdRng` (ChaCha),
/// launch-portable per seed.
pub fn create_noise(
    seed: u64,
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    device: &Device,
) -> Result<Tensor> {
    let n = z_dim * t_lat * h_lat * w_lat;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Tensor::from_vec(data, (1, z_dim, t_lat, h_lat, w_lat), device)
}

/// Classifier-free guidance: `uncond + g·(cond − uncond)`.
pub fn cfg(cond: &Tensor, uncond: &Tensor, guidance: f64) -> Result<Tensor> {
    uncond + (cond - uncond)?.affine(guidance, 0.0)?
}

/// Build the latent and token masks for Wan TI2V Reference/keyframe conditioning. Each `(frame,
/// strength)` produces `1-strength` in both masks, so the same weight controls clean-latent blending
/// and the per-token diffusion timestep. The token mask follows the DiT patch-grid order.
pub fn build_ti2v_mask(
    pins: &[(usize, f32)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch: (usize, usize, usize),
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let plane = h_lat * w_lat;
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        for &(t, strength) in pins.iter().filter(|&&(t, _)| t < t_lat) {
            mask[((c * t_lat + t) * plane)..((c * t_lat + t + 1) * plane)].fill(1.0 - strength);
        }
    }
    let mask = Tensor::from_vec(mask, (1, z_dim, t_lat, h_lat, w_lat), device)?;

    let (pt, ph, pw) = patch;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tokens = vec![1f32; tg * hg * wg];
    for &(t, strength) in pins {
        let token_t = t / pt;
        if token_t < tg {
            tokens[(token_t * hg * wg)..((token_t + 1) * hg * wg)].fill(1.0 - strength);
        }
    }
    let tokens = Tensor::from_vec(tokens, (1, tg * hg * wg), device)?;
    Ok((mask, tokens))
}

/// Scatter independently encoded `[1,z,1,h,w]` keyframes into one clean `[1,z,T,h,w]` latent.
pub fn build_ti2v_keyframe_z(
    frames: &[(Tensor, usize)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    device: &Device,
) -> Result<Tensor> {
    let zero = Tensor::zeros(
        (1, z_dim, 1, h_lat, w_lat),
        candle_gen::candle_core::DType::F32,
        device,
    )?;
    let mut slices = (0..t_lat).map(|_| zero.clone()).collect::<Vec<_>>();
    for (latent, index) in frames {
        if *index < t_lat {
            slices[*index] = latent.clone();
        }
    }
    Tensor::cat(&slices.iter().collect::<Vec<_>>(), 2)
}

/// Initial and per-step TI2V blend: `(1-mask)·clean + mask·sample`.
pub fn ti2v_blend(clean: &Tensor, mask: &Tensor, sample: &Tensor) -> Result<Tensor> {
    let frozen = clean.broadcast_mul(&mask.affine(-1.0, 1.0)?)?;
    frozen + mask.broadcast_mul(sample)?
}

/// Decoded video `[1, 3, T, H, W]` in `[-1, 1]` → one RGB8 [`Image`] per frame.
pub fn frames_to_images(decoded: &Tensor) -> Result<Vec<Image>> {
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let u8s = candle_gen::round_rgb8(&scaled)?.to_device(&Device::Cpu)?;
    let (_b, c, t, h, w) = u8s.dims5()?;
    let frames = u8s.squeeze(0)?; // [3,T,H,W]
    let mut out = Vec::with_capacity(t);
    for ti in 0..t {
        let frame = frames.narrow(1, ti, 1)?.squeeze(1)?; // [3,H,W]
        let pixels = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        debug_assert_eq!(c, 3);
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod ti2v_tests {
    use super::*;
    use candle_gen::candle_core::DType;

    #[test]
    fn mask_and_keyframe_scatter_pin_first_and_last() -> Result<()> {
        let dev = Device::Cpu;
        let (mask, tokens) = build_ti2v_mask(&[(0, 1.0), (2, 0.25)], 1, 3, 2, 2, (1, 2, 2), &dev)?;
        assert_eq!(
            mask.flatten_all()?.to_vec1::<f32>()?,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75]
        );
        assert_eq!(
            tokens.flatten_all()?.to_vec1::<f32>()?,
            vec![0.0, 1.0, 0.75]
        );

        let first = Tensor::full(2f32, (1, 1, 1, 2, 2), &dev)?;
        let last = Tensor::full(7f32, (1, 1, 1, 2, 2), &dev)?;
        let clean = build_ti2v_keyframe_z(&[(first, 0), (last, 2)], 1, 3, 2, 2, &dev)?;
        let noise = Tensor::ones(clean.shape(), DType::F32, &dev)?;
        let blended = ti2v_blend(&clean, &mask, &noise)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            blended,
            vec![2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5]
        );
        Ok(())
    }
}
