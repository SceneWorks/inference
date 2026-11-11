//! The candle tensor operations behind the shared H3-Image sampler step
//! ([`candle_gen::gen_core::h3_image::sampler_step`]). The step arithmetic (Euler and LCM) lives in
//! gen-core; this file only supplies float32 candle ops and the seeded noise source.
//!
//! **Noise.** Step `i` of an image seeded `image_seed` draws its LCM noise on the host, the way the
//! initial latent is drawn: `StdRng::seed_from_u64(lcm_noise_seed(image_seed, i))` through
//! [`seeded_normal_vec`], shaped like the generated rows, then moved to the device. The draw is
//! therefore the same on CPU, CUDA and Metal, and a seed and sampler render the same image on
//! every run.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image::H3ImageStepOps;
use candle_gen::seed::seeded_normal_vec;
use candle_gen::{CandleError, Result};
use rand::SeedableRng;

/// [`H3ImageStepOps`] over float32 candle tensors on `device`.
#[derive(Debug, Clone, Copy)]
pub struct CandleStepOps<'a> {
    pub device: &'a Device,
}

impl H3ImageStepOps for CandleStepOps<'_> {
    type Tensor = Tensor;
    type Error = CandleError;

    fn add_scaled(&mut self, x: &Tensor, y: &Tensor, b: f32) -> Result<Tensor> {
        Ok((x + (y * f64::from(b))?)?)
    }

    fn scale(&mut self, x: &Tensor, a: f32) -> Result<Tensor> {
        Ok((x * f64::from(a))?)
    }

    fn gaussian_like(&mut self, like: &Tensor, seed: u64) -> Result<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Ok(Tensor::from_vec(
            seeded_normal_vec(&mut rng, like.elem_count()),
            like.dims(),
            self.device,
        )?)
    }

    fn std(&mut self, x: &Tensor) -> Result<f32> {
        let n = x.elem_count();
        if n < 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image sampler: the standard deviation of {n} element(s) is undefined"
            )));
        }
        let x = x.to_dtype(DType::F32)?.flatten_all()?;
        let centered = x.broadcast_sub(&x.mean_all()?)?;
        let ss = centered.sqr()?.sum_all()?.to_scalar::<f32>()?;
        Ok((ss / (n - 1) as f32).sqrt())
    }

    fn clamp(&mut self, x: &Tensor, lo: f32, hi: f32) -> Result<Tensor> {
        Ok(x.clamp(lo, hi)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::h3_image::{
        lcm_noise_seed, sampler_step, H3ImageSampler, SamplerStep,
    };

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
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

    #[test]
    fn candle_euler_step_matches_hand_values() {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &d).unwrap();
        let v = Tensor::from_vec(vec![0.5f32, -1.0], (1, 1, 2), &d).unwrap();
        let mut ops = CandleStepOps { device: &d };
        let out = sampler_step(&mut ops, &H3ImageSampler::Euler, AT, &x, &v).unwrap();
        close(&host(&out), &[1.125, 1.75]);
    }

    /// LCM on tiny tensors reproduces `0.5·(x + 0.75·v) + 0.5·1.1·ε` with ε the step's seeded
    /// host draw; the same seed gives the same result, another seed a different one.
    #[test]
    fn candle_lcm_step_matches_hand_values_and_is_seeded() {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 0.0, -1.0], (1, 2, 2), &d).unwrap();
        let v = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.0], (1, 2, 2), &d).unwrap();
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let mut ops = CandleStepOps { device: &d };
        let out = host(&sampler_step(&mut ops, &lcm, AT, &x, &v).unwrap());
        let mut rng = rand::rngs::StdRng::seed_from_u64(lcm_noise_seed(7, 0));
        let eps = seeded_normal_vec(&mut rng, 4);
        let x0 = [1.375f32, 1.25, 1.5, -1.0];
        let want: Vec<f32> = x0
            .iter()
            .zip(&eps)
            .map(|(x0, e)| 0.5 * x0 + 0.5 * 1.1 * e)
            .collect();
        close(&out, &want);
        assert_eq!(
            out,
            host(&sampler_step(&mut ops, &lcm, AT, &x, &v).unwrap())
        );
        let other = SamplerStep {
            image_seed: 8,
            ..AT
        };
        assert_ne!(
            out,
            host(&sampler_step(&mut ops, &lcm, other, &x, &v).unwrap())
        );
    }

    #[test]
    fn candle_clip_and_std_match_hand_values() {
        let d = Device::Cpu;
        let mut ops = CandleStepOps { device: &d };
        let e = Tensor::from_vec(vec![3.0f32, -3.0, 1.0, -1.0], (1, 4), &d).unwrap();
        let s = ops.std(&e).unwrap();
        assert!((s - (20.0f32 / 3.0).sqrt()).abs() < 1e-5, "{s}");
        assert_eq!(
            host(&ops.clamp(&e, -2.0, 2.0).unwrap()),
            vec![2.0, -2.0, 1.0, -1.0]
        );
        let zero = Tensor::zeros((1, 64), DType::F32, &d).unwrap();
        let clipped = H3ImageSampler::Lcm {
            s_noise: 1.0,
            s_noise_end: 1.0,
            noise_clip_std: 0.5,
        };
        let out = host(&sampler_step(&mut ops, &clipped, AT, &zero, &zero).unwrap());
        let eps = ops.gaussian_like(&zero, lcm_noise_seed(7, 0)).unwrap();
        let bound = 0.5 * 0.5 * ops.std(&eps).unwrap();
        assert!(out.iter().all(|v| v.abs() <= bound + 1e-6));
        assert!(out.iter().any(|v| (v.abs() - bound).abs() < 1e-6));
    }
}
