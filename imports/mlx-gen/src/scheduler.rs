//! Flow-match Euler discrete scheduler — the sampler shared by the mlx-gen DiT families
//! (Z-Image, FLUX, Qwen). Port of the Python mflux fork's `FlowMatchEulerDiscreteScheduler`
//! (`models/common/schedulers/flow_match_euler_discrete_scheduler.py`).
//!
//! The schedule is a `linspace(1, 1/n, n)` run through an exponential **time-shift** whose
//! `mu` is fit empirically from the latent sequence length (the fork's `requires_sigma_shift`
//! path), with a trailing `0` appended to mark the final step. Each denoise step is the Euler
//! update `x_{t+1} = x_t + (sigma[t+1] - sigma[t]) * v`, where `v` is the model's (already
//! sign-flipped) velocity prediction.

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use crate::Result;

/// A flow-match Euler denoising schedule.
pub struct FlowMatchEuler {
    /// Denoising sigmas, length `num_steps + 1` (the trailing `0.0` marks the final step).
    pub sigmas: Vec<f32>,
}

impl FlowMatchEuler {
    /// Build the schedule for `num_steps` with an explicit time-shift `mu`.
    pub fn new(num_steps: usize, mu: f32) -> Self {
        Self {
            sigmas: build_sigmas(num_steps, mu),
        }
    }

    /// Build the schedule for an image of `width`×`height`, computing the resolution-dependent
    /// `mu` from the latent sequence length (the fork's `requires_sigma_shift` path).
    pub fn for_image(num_steps: usize, width: u32, height: u32) -> Self {
        let seq_len = image_seq_len(width, height);
        Self::new(num_steps, compute_mu(seq_len, num_steps))
    }

    /// Number of denoising steps (loop iterations).
    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// The transformer timestep at step `t`: `1 - sigma[t]` (in `[0, 1]`; the model applies its
    /// own `t_scale`).
    pub fn timestep(&self, t: usize) -> f32 {
        1.0 - self.sigmas[t]
    }

    /// One Euler step: `x_{t+1} = x_t + (sigma[t+1] - sigma[t]) * velocity`.
    pub fn step(&self, latents: &Array, velocity: &Array, t: usize) -> Result<Array> {
        let dt = self.sigmas[t + 1] - self.sigmas[t];
        Ok(add(
            latents,
            &multiply(velocity, Array::from_slice(&[dt], &[1]))?,
        )?)
    }
}

/// Latent sequence length used for the empirical `mu` fit: `(height/16) * (width/16)`.
pub fn image_seq_len(width: u32, height: u32) -> usize {
    ((height / 16) * (width / 16)) as usize
}

/// Port of the fork's `_compute_empirical_mu`: a piecewise-linear fit of the time-shift `mu`
/// from the latent sequence length and step count.
//  Constants mirror the fork's Python float64 literals verbatim (8.73809524e-05 / 1.89833333 /
//  0.00016927 / 0.45666666) for parity auditing; f32 rounds the extra digits harmlessly.
#[allow(clippy::excessive_precision)]
pub fn compute_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    let (a1, b1) = (8.738_095_24e-5_f32, 1.898_333_33_f32);
    let (a2, b2) = (0.000_169_27_f32, 0.456_666_66_f32);
    let seq = image_seq_len as f32;
    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }
    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f32 + b
}

/// `exp(mu) / (exp(mu) + (1/t - 1))` — the fork's `_time_shift_exponential_array` at
/// `sigma_power = 1`.
fn time_shift_exponential(mu: f32, t: f32) -> f32 {
    let e = mu.exp();
    e / (e + (1.0 / t - 1.0))
}

fn build_sigmas(num_steps: usize, mu: f32) -> Vec<f32> {
    let n = num_steps.max(1);
    let (start, end) = (1.0_f32, 1.0_f32 / n as f32);
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, 1.0/n, n)
            let t = if n == 1 {
                start
            } else {
                start + (end - start) * (i as f32) / ((n - 1) as f32)
            };
            time_shift_exponential(mu, t)
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_shape_and_endpoints() {
        let s = FlowMatchEuler::for_image(4, 1024, 1024);
        assert_eq!(s.sigmas.len(), 5); // num_steps + 1
        assert_eq!(s.num_steps(), 4);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        // sigmas strictly decreasing.
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
        // timestep is 1 - sigma.
        assert!((s.timestep(0) - (1.0 - s.sigmas[0])).abs() < 1e-6);
    }

    #[test]
    fn seq_len_matches_definition() {
        assert_eq!(image_seq_len(1024, 1024), 4096);
        assert_eq!(image_seq_len(256, 256), 256);
        assert_eq!(image_seq_len(1280, 1280), 6400);
    }

    #[test]
    fn mu_large_seq_branch() {
        // > 4300 uses the linear-in-seq_len branch (independent of num_steps).
        let a = compute_mu(6400, 4);
        let b = compute_mu(6400, 8);
        assert!((a - b).abs() < 1e-6);
    }
}
