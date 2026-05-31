//! Sinusoidal timestep projection. Port of the fork's `QwenTimesteps` (proj_dim 256, scale 1000,
//! max_period 10000, `flip_sin_to_cos`, downscale_freq_shift 0). Stateless.

use mlx_rs::ops::{concatenate_axis, multiply};
use mlx_rs::Array;

use mlx_gen::Result;

/// `timesteps`: `[B]` f32 → `[B, proj_dim]`. `flip_sin_to_cos` means the output is `[cos | sin]`.
pub fn timestep_proj(timesteps: &Array, proj_dim: i32, scale: f32) -> Result<Array> {
    let half = (proj_dim / 2) as usize;
    let max_period = 10000f32;
    // freq[k] = exp(-ln(max_period) * k / half) = max_period^(-k/half)
    let freqs: Vec<f32> = (0..half)
        .map(|k| (-(max_period.ln()) * k as f32 / half as f32).exp())
        .collect();
    let freq = Array::from_slice(&freqs, &[1, half as i32]);
    let b = timesteps.shape()[0];
    let emb = multiply(&timesteps.reshape(&[b, 1])?, &freq)?; // [B, half]
    let emb = multiply(&emb, Array::from_slice(&[scale], &[1]))?;
    let cos = emb.cos()?;
    let sin = emb.sin()?;
    Ok(concatenate_axis(&[&cos, &sin], 1)?) // flip_sin_to_cos → [cos, sin]
}
