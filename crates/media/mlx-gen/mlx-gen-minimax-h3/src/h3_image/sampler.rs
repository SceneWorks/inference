//! The MLX tensor operations behind the shared H3-Image sampler step
//! ([`mlx_gen::gen_core::h3_image::sampler_step`]). The step arithmetic (Euler and LCM) lives in
//! gen-core; this file only supplies float32 MLX ops and the seeded noise source.
//!
//! **Noise.** Step `i` of an image seeded `image_seed` draws its LCM noise with
//! `mlx_rs::random::normal` under `mlx_rs::random::key(lcm_noise_seed(image_seed, i))`, shaped like
//! the generated rows, so a seed and sampler render the same image on every MLX run.

use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::h3_image::H3ImageStepOps;
use mlx_gen::{Error, Result};

/// [`H3ImageStepOps`] over float32 MLX arrays.
#[derive(Debug, Default, Clone, Copy)]
pub struct MlxStepOps;

impl H3ImageStepOps for MlxStepOps {
    type Tensor = Array;
    type Error = Error;

    fn add_scaled(&mut self, x: &Array, y: &Array, b: f32) -> Result<Array> {
        Ok(mlx_rs::ops::add(
            x,
            &mlx_rs::ops::multiply(y, Array::from_f32(b))?,
        )?)
    }

    fn scale(&mut self, x: &Array, a: f32) -> Result<Array> {
        Ok(mlx_rs::ops::multiply(x, Array::from_f32(a))?)
    }

    fn gaussian_like(&mut self, like: &Array, seed: u64) -> Result<Array> {
        let key = mlx_rs::random::key(seed)?;
        Ok(mlx_rs::random::normal::<f32>(
            like.shape(),
            None,
            None,
            Some(&key),
        )?)
    }

    fn std(&mut self, x: &Array) -> Result<f32> {
        let s = mlx_rs::ops::std(&x.as_dtype(Dtype::Float32)?, None, 1)?;
        Ok(s.item::<f32>())
    }

    fn clamp(&mut self, x: &Array, lo: f32, hi: f32) -> Result<Array> {
        Ok(mlx_rs::ops::clip(x, (lo, hi))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::h3_image::{lcm_noise_seed, sampler_step, H3ImageSampler, SamplerStep};

    fn host(a: &Array) -> Vec<f32> {
        a.as_slice::<f32>().to_vec()
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{got:?} vs {want:?}");
        }
    }

    const AT: SamplerStep = SamplerStep {
        step: 0,
        num_steps: 6,
        sigma: 0.75,
        sigma_next: 0.5,
        image_seed: 7,
    };

    /// Euler on tiny arrays: `x + (σ − σ')·v`.
    #[test]
    fn mlx_euler_step_matches_hand_values() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]);
        let v = Array::from_slice(&[0.5f32, -1.0], &[1, 1, 2]);
        let out = sampler_step(&mut MlxStepOps, &H3ImageSampler::Euler, AT, &x, &v).unwrap();
        close(&host(&out), &[1.125, 1.75]);
    }

    /// LCM on tiny arrays reproduces `0.5·(x + 0.75·v) + 0.5·1.1·ε` with ε the step's seeded draw,
    /// and the same seed gives the same result.
    #[test]
    fn mlx_lcm_step_matches_hand_values_and_is_seeded() {
        let x = Array::from_slice(&[1.0f32, 2.0, 0.0, -1.0], &[1, 2, 2]);
        let v = Array::from_slice(&[0.5f32, -1.0, 2.0, 0.0], &[1, 2, 2]);
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let out = host(&sampler_step(&mut MlxStepOps, &lcm, AT, &x, &v).unwrap());
        let eps = host(&MlxStepOps.gaussian_like(&x, lcm_noise_seed(7, 0)).unwrap());
        let x0 = [1.375f32, 1.25, 1.5, -1.0];
        let want: Vec<f32> = x0
            .iter()
            .zip(&eps)
            .map(|(x0, e)| 0.5 * x0 + 0.5 * 1.1 * e)
            .collect();
        close(&out, &want);
        let again = host(&sampler_step(&mut MlxStepOps, &lcm, AT, &x, &v).unwrap());
        assert_eq!(out, again);
        let other = SamplerStep {
            image_seed: 8,
            ..AT
        };
        assert_ne!(
            out,
            host(&sampler_step(&mut MlxStepOps, &lcm, other, &x, &v).unwrap())
        );
    }

    /// The clip bounds the draw by `noise_clip_std` sample standard deviations.
    #[test]
    fn mlx_clip_and_std_match_hand_values() {
        let e = Array::from_slice(&[3.0f32, -3.0, 1.0, -1.0], &[1, 4]);
        let s = MlxStepOps.std(&e).unwrap();
        assert!((s - (20.0f32 / 3.0).sqrt()).abs() < 1e-5, "{s}");
        let c = host(&MlxStepOps.clamp(&e, -2.0, 2.0).unwrap());
        assert_eq!(c, vec![2.0, -2.0, 1.0, -1.0]);
        let x = Array::from_slice(&[0.0f32; 64], &[1, 64]);
        let clipped = H3ImageSampler::Lcm {
            s_noise: 1.0,
            s_noise_end: 1.0,
            noise_clip_std: 0.5,
        };
        let out = host(&sampler_step(&mut MlxStepOps, &clipped, AT, &x, &x).unwrap());
        let eps = MlxStepOps.gaussian_like(&x, lcm_noise_seed(7, 0)).unwrap();
        let bound = 0.5 * 0.5 * MlxStepOps.std(&eps).unwrap();
        assert!(out.iter().all(|v| v.abs() <= bound + 1e-6));
        assert!(out.iter().any(|v| (v.abs() - bound).abs() < 1e-6));
    }
}
