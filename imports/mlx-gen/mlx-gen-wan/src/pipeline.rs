//! S4 — the dense **T2V generation pipeline**: the denoise loop + CFG + VAE decode + frame
//! assembly that turns a prompt into video frames. Port of `generate_wan.py`'s `generate_video`
//! (the single-model, non-MoE path).
//!
//! This is **reusable machinery**, not a model: the dense loop here is exactly what each expert of
//! the Wan2.2-A14B dual-expert MoE runs (S5 adds the per-step boundary swap on top) and what the 5B
//! runs (sc-2680, with its z48 VAE). The concrete `Generator::generate` wiring for a runnable model
//! lands with those slices; S4 delivers + parity-gates the shared loop.
//!
//! Shapes are channels-first **`[C, F, H, W]`** (no batch dim) throughout — matching
//! [`WanTransformer::forward`] (one sample per call) and [`WanScheduler::step`]. CFG runs cond +
//! uncond as two B=1 forwards (bit-identical to the reference's batched B=2, since attention never
//! mixes batch elements — see the `forward` docs).

use mlx_rs::ops::{add, maximum, minimum, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

use crate::scheduler::{make_scheduler, SolverKind};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Align a pixel dimension **down** to a multiple of `patch · vae_stride` (the reference rounds the
/// requested size to the nearest valid grid; sub-tile requests are rejected by `validate`).
pub fn align_dim(value: u32, patch: usize, stride: usize) -> u32 {
    let align = (patch * stride) as u32;
    (value / align) * align
}

/// Latent shape `[z_dim, t_lat, h_lat, w_lat]` for a `frames × H × W` request.
/// `t_lat = (frames − 1) / vae_stride_t + 1`; spatial divide by the vae stride.
pub fn latent_shape(
    frames: usize,
    height: u32,
    width: u32,
    z_dim: usize,
    vae_stride: (usize, usize, usize),
) -> [i32; 4] {
    let t_lat = (frames - 1) / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    [z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32]
}

/// Transformer sequence length: `ceil(h_lat · w_lat / (patch_h · patch_w) · t_lat)`.
pub fn seq_len(latent: [i32; 4], patch_size: (usize, usize, usize)) -> usize {
    let (_z, t_lat, h_lat, w_lat) = (latent[0], latent[1], latent[2], latent[3]);
    let per_frame = (h_lat as usize * w_lat as usize) as f64 / (patch_size.1 * patch_size.2) as f64;
    (per_frame * t_lat as f64).ceil() as usize
}

/// Classifier-free guidance combine: `uncond + gs·(cond − uncond)`.
fn cfg_combine(cond: &Array, uncond: &Array, gs: f32) -> Result<Array> {
    Ok(add(
        uncond,
        &multiply(&subtract(cond, uncond)?, scalar(gs))?,
    )?)
}

/// The dense denoise loop. `ctx_cond`/`ctx_uncond` are [`WanTransformer::embed_text`] outputs
/// (`[1, text_len, dim]`, bf16); pass `ctx_uncond = None` for the CFG-disabled B=1 fast path.
/// `init_noise` is `[C, F, H, W]` f32. Returns the denoised latents `[out_dim, F, H, W]` (f32).
///
/// `on_step(i)` is called after each completed step (progress reporting).
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        let pred = match ctx_uncond {
            Some(uncond_ctx) => {
                let cond = transformer.forward(&latents, t, ctx_cond)?;
                let uncond = transformer.forward(&latents, t, uncond_ctx)?;
                cfg_combine(&cond, &uncond, guidance)?
            }
            None => transformer.forward(&latents, t, ctx_cond)?,
        };
        latents = sched.step(&pred, &latents)?;
        // Force evaluation each step to bound the lazy graph's peak memory (the reference's
        // per-step `mx.eval(latents)`).
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// Decode denoised latents `[C, F, H, W]` → an RGB video tensor `[F_out, H_out, W_out, 3]` of
/// `uint8` (the reference's `(video + 1)/2 · 255`, clamped). Uses the Wan 2.1 z16 VAE (S2).
pub fn decode_to_frames(vae: &WanVae, latents: &Array) -> Result<Array> {
    // WanVae::decode expects/returns a leading batch dim: [1, 3, F, H, W] in [-1, 1].
    let video = vae.decode(&latents.reshape(&prepend1(latents.shape()))?)?;
    // [1,3,F,H,W] → [F,H,W,3]
    let sh = video.shape(); // [1,3,F,H,W]
    let (f, h, w) = (sh[2], sh[3], sh[4]);
    let chw = video
        .reshape(&[3, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // [F,H,W,3]
                                         // [-1,1] → [0,255] uint8
    let scaled = multiply(&add(&chw, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(clamped.as_dtype(mlx_rs::Dtype::Uint8)?)
}

/// `[d0, d1, ...]` → `[1, d0, d1, ...]` (prepend a batch axis).
fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_dim_rounds_down_to_tile() {
        // patch 2 × vae_stride 8 = 16-px grid.
        assert_eq!(align_dim(1280, 2, 8), 1280);
        assert_eq!(align_dim(1281, 2, 8), 1280);
        assert_eq!(align_dim(1295, 2, 8), 1280);
        assert_eq!(align_dim(1296, 2, 8), 1296);
    }

    #[test]
    fn latent_shape_and_seq_len_match_reference_formulas() {
        // 49 frames, 512×512, z16, stride (4,8,8), patch (1,2,2).
        let ls = latent_shape(49, 512, 512, 16, (4, 8, 8));
        assert_eq!(ls, [16, 13, 64, 64]); // (49-1)/4+1=13, 512/8=64
        let sl = seq_len(ls, (1, 2, 2));
        // ceil(64*64/(2*2) * 13) = 1024 * 13 = 13312
        assert_eq!(sl, 13312);
    }

    #[test]
    fn cfg_combine_is_uncond_plus_gs_delta() {
        let cond = Array::from_slice(&[2.0f32, 4.0], &[2]);
        let uncond = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let got = cfg_combine(&cond, &uncond, 3.0).unwrap();
        // 1 + 3*(2-1) = 4 ; 1 + 3*(4-1) = 10
        assert_eq!(got.as_slice::<f32>(), &[4.0, 10.0]);
    }
}
