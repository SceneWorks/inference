//! LTX distilled flow-match schedule + the **legacy dtype-preserving Euler** step.
//!
//! The 2.3 distilled / unified path uses two fixed sigma lists (not the resolution-dependent
//! `mu`-shift of the image families' [`mlx_gen::FlowMatchEuler`]). `generate_av.py`'s
//! `build_stage_sigma_schedules` returns these defaults for the canonical 11-step app run; the
//! token-dependent `ltx2_schedule` / `linear_quadratic_schedule` branches are **disabled** on the
//! 2.3 MLX weights (they produce garbled output) and are deliberately **not** ported here — they
//! remain a noted-future path (sc-2679 S6), not silently enabled.
//!
//! The step matches `generate.py::denoise` exactly (velocity → denoised → Euler), written in the
//! reference's dtype-preserving form: every scalar is folded in at the latents' dtype so no f32
//! promotion sneaks in. Algebraically this is the same flow-match Euler update
//! `x_{t+1} = x_t + (σ_{t+1} − σ_t)·v` — but the explicit `denoised` intermediate is preserved
//! because the I2V sibling masks on it (`apply_denoise_mask`).

use mlx_rs::ops::{add, divide, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

/// Stage-1 distilled schedule (8 steps): half-resolution generation.
pub const DEFAULT_STAGE_1_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];

/// Stage-2 distilled schedule (3 steps): full-resolution refine. The first sigma (0.909375) is the
/// re-noise level applied to the upsampled stage-1 latents before stage 2.
pub const DEFAULT_STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// A scalar `Array` at the same dtype as `like` (dtype-preserving, like the reference's
/// `mx.array(sigma, dtype=dtype)`).
fn scalar_like(value: f32, like: &Array) -> Result<Array> {
    Ok(Array::from_slice(&[value], &[1]).as_dtype(like.dtype())?)
}

/// Velocity → denoised: `x_0 = x_t − σ·v` (port of `utils.to_denoised`, scalar σ).
pub fn to_denoised(noisy: &Array, velocity: &Array, sigma: f32) -> Result<Array> {
    let s = scalar_like(sigma, velocity)?;
    Ok(subtract(noisy, &multiply(&s, velocity)?)?)
}

/// One legacy Euler step from `(noisy, denoised)`:
/// `x_{t+1} = denoised + σ_next·(noisy − denoised)/σ` when `σ_next > 0`, else `denoised`.
/// `noisy` is the pre-step latent (the RHS `latents` in `generate.py::denoise`).
pub fn euler_step(noisy: &Array, denoised: &Array, sigma: f32, sigma_next: f32) -> Result<Array> {
    if sigma_next > 0.0 {
        let s_next = scalar_like(sigma_next, noisy)?;
        let s = scalar_like(sigma, noisy)?;
        let diff = subtract(noisy, denoised)?;
        let scaled = divide(&multiply(&s_next, &diff)?, &s)?;
        Ok(add(denoised, &scaled)?)
    } else {
        Ok(denoised.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_are_exact() {
        assert_eq!(DEFAULT_STAGE_1_SIGMAS.len(), 9); // 8 steps
        assert_eq!(DEFAULT_STAGE_2_SIGMAS.len(), 4); // 3 steps
        assert_eq!(DEFAULT_STAGE_1_SIGMAS[0], 1.0);
        assert_eq!(*DEFAULT_STAGE_1_SIGMAS.last().unwrap(), 0.0);
        assert_eq!(DEFAULT_STAGE_2_SIGMAS[0], 0.909375);
        assert_eq!(*DEFAULT_STAGE_2_SIGMAS.last().unwrap(), 0.0);
        // The stage boundary: stage-2 starts at stage-1's σ index 5 (the 0.909375 anchor).
        assert_eq!(DEFAULT_STAGE_1_SIGMAS[5], DEFAULT_STAGE_2_SIGMAS[0]);
    }

    #[test]
    fn euler_equals_flow_match_form() {
        // x_next = denoised + σn·(x−denoised)/σ  ==  x + (σn−σ)·v  where denoised = x − σ·v.
        let x = Array::from_slice(&[1.0f32, 2.0, -3.0, 0.5], &[4]);
        let v = Array::from_slice(&[0.1f32, -0.2, 0.3, 1.0], &[4]);
        let (sigma, sigma_next) = (0.725f32, 0.421875f32);
        let denoised = to_denoised(&x, &v, sigma).unwrap();
        let got = euler_step(&x, &denoised, sigma, sigma_next).unwrap();
        // expected = x + (σn − σ)·v
        let dt = sigma_next - sigma;
        let expected: Vec<f32> = x
            .as_slice::<f32>()
            .iter()
            .zip(v.as_slice::<f32>())
            .map(|(xi, vi)| xi + dt * vi)
            .collect();
        for (g, e) in got.as_slice::<f32>().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "euler mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn final_step_is_denoised() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let v = Array::from_slice(&[0.5f32, -0.5], &[2]);
        let sigma = 0.421875f32;
        let denoised = to_denoised(&x, &v, sigma).unwrap();
        let got = euler_step(&x, &denoised, sigma, 0.0).unwrap();
        for (g, d) in got.as_slice::<f32>().iter().zip(denoised.as_slice::<f32>()) {
            assert_eq!(g, d);
        }
    }
}
