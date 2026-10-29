//! `MiniMaxH3Scheduler` — rectified-flow Euler over the **two** sigma schedules a joint run steps
//! in lockstep.
//!
//! The grids, the shifts and the Euler coefficients are backend-neutral and live in
//! [`candle_gen::gen_core::h3_schedule`] (sc-27256), shared with the MLX twin; this module
//! re-exports them so `schedule::*` paths stay stable and adds [`SigmaScheduleStep`], the update
//! on this crate's tensors.

pub use candle_gen::gen_core::h3_schedule::*;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result};

/// One Euler update on the rows a [`SigmaSchedule`]'s modality owns.
///
/// `sample` is `x_t`, `velocity` is the transformer's **data-ward** prediction for the same rows.
/// Both are read at float32; the result is float32, as the reference's latents are. The two
/// coefficients come from [`SigmaSchedule::euler_coefficients`]:
///
/// ```text
/// denoised = x_t + (1 − t)·v            (NOTE the +, and σ from the timestep)
/// x_next   = r·x_t + (1 − r)·denoised   with r = σ[i+1]/σ[i] from the grid
/// ```
pub trait SigmaScheduleStep {
    /// Apply step `index` to `sample` with `velocity`.
    fn step(&self, index: usize, sample: &Tensor, velocity: &Tensor) -> Result<Tensor>;
}

impl SigmaScheduleStep for SigmaSchedule {
    fn step(&self, index: usize, sample: &Tensor, velocity: &Tensor) -> Result<Tensor> {
        let (sigma_from_timestep, ratio) = self.euler_coefficients(index)?;
        if sample.dims() != velocity.dims() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 scheduler: {} sample {:?} and velocity {:?} must be the same rows",
                self.modality().name(),
                sample.dims(),
                velocity.dims()
            )));
        }
        let x = sample.to_dtype(DType::F32)?;
        let v = velocity.to_dtype(DType::F32)?;
        let denoised = (&x + (v * f64::from(sigma_from_timestep))?)?;
        Ok(((x * f64::from(ratio))? + (denoised * f64::from(1.0 - ratio))?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// **The reversed velocity sign.** With `v = +1` and `x_t = 0`, a data-ward velocity moves the
    /// sample *up*; diffusers' convention would move it down. Checked at a step whose ratio is not
    /// 0 or 1, so the sign survives into the blend.
    #[test]
    fn the_velocity_sign_is_data_ward() {
        let dev = Device::Cpu;
        let s = SigmaSchedule::new(DenoiseModality::Video, 4).unwrap();
        let x = Tensor::from_vec(vec![0.0f32, 0.0], (2,), &dev).unwrap();
        let v = Tensor::from_vec(vec![1.0f32, -1.0], (2,), &dev).unwrap();
        let got = flat(&s.step(1, &x, &v).unwrap());
        // denoised = 0 + (1 - t)·v = sigma_from_t · v; x_next = (1-r)·denoised since x = 0.
        let sigma_from_t = 1.0 - s.timesteps()[1];
        let r = s.sigmas()[2] / s.sigmas()[1];
        let want = (1.0 - r) * sigma_from_t;
        assert!((got[0] - want).abs() < 1e-6, "{got:?} vs {want}");
        assert!(
            got[0] > 0.0,
            "a data-ward velocity must move the sample UP (x0 = x_t + sigma·v)"
        );
        assert!((got[1] + want).abs() < 1e-6, "and antisymmetric: {got:?}");
    }

    /// A terminal step lands exactly on the clean sample: `r = 0` makes `x_next = denoised`.
    #[test]
    fn the_last_step_returns_the_denoised_sample() {
        let dev = Device::Cpu;
        let s = SigmaSchedule::new(DenoiseModality::Video, 4).unwrap();
        let last = s.num_evals() - 1;
        assert_eq!(s.sigmas()[last + 1], 0.0);
        let x = Tensor::from_vec(vec![2.0f32], (1,), &dev).unwrap();
        let v = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let got = flat(&s.step(last, &x, &v).unwrap())[0];
        let want = 2.0 + (1.0 - s.timesteps()[last]) * 0.5;
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        assert!(s.step(last + 1, &x, &v).is_err(), "past the end");
        assert!(
            s.step(
                0,
                &x,
                &Tensor::from_vec(vec![1.0f32, 2.0], (2,), &dev).unwrap()
            )
            .is_err(),
            "mismatched rows"
        );
    }
}
