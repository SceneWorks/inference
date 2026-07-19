//! The ACE-Step 1.5 flow-matching schedule (sc-12842) — a faithful port of the reference
//! `AceStepPipeline._get_timestep_schedule()` + the `FlowMatchEulerDiscreteScheduler` Euler
//! update it drives.
//!
//! The reference computes its own sigma ladder and hands it to the scheduler via
//! `set_timesteps(sigmas=…)`:
//!
//! ```text
//!   t = linspace(1.0, 0.0, num_inference_steps + 1)          # steps+1 points, 1 … 0
//!   if shift != 1.0:  t = shift·t / (1 + (shift − 1)·t)      # the classic flow shift
//!   return t[:-1]                                            # drop the terminal 0
//! ```
//!
//! The scheduler then appends a terminal `σ = 0` for the final Euler step. ACE-Step feeds the DiT
//! the timestep in `[0, 1]` directly (its scheduler is configured `num_train_timesteps = 1`), so
//! — unlike the MOSS SFX port — the DiT conditioning timestep is `σ_k` itself, not `σ_k · 1000`.
//! The Euler update is `x ← x + v·(σ_{k+1} − σ_k)`.

use candle_audio::candle_core::{Result as CandleResult, Tensor};

/// Turbo's guidance-distilled default shift.
pub const DEFAULT_SHIFT: f64 = 3.0;

/// The precomputed σ ladder for one denoise run: `steps` shifted sigmas plus the terminal `0.0`.
#[derive(Debug, Clone)]
pub struct FlowMatchSchedule {
    sigmas: Vec<f64>,
}

impl FlowMatchSchedule {
    /// Build the schedule for `steps` inference steps with the given flow `shift`
    /// (`shift = 1.0` is the identity map `σ_k = s_k`).
    pub fn new(steps: usize, shift: f64) -> Self {
        let steps = steps.max(1);
        let mut sigmas: Vec<f64> = (0..steps)
            .map(|i| {
                // linspace(1, 0, steps+1)[i] = 1 − i/steps.
                let s = 1.0 - (i as f64) / (steps as f64);
                if (shift - 1.0).abs() < f64::EPSILON {
                    s
                } else {
                    shift * s / (1.0 + (shift - 1.0) * s)
                }
            })
            .collect();
        sigmas.push(0.0); // terminal boundary the scheduler appends.
        Self { sigmas }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// σ at step `k` (`k ≤ steps`; `σ_steps = 0`).
    pub fn sigma(&self, k: usize) -> f64 {
        self.sigmas[k]
    }

    /// The DiT conditioning timestep at step `k` — `σ_k` itself (ACE-Step's `[0, 1]` frame).
    pub fn timestep(&self, k: usize) -> f64 {
        self.sigmas[k]
    }

    /// One Euler flow-match update: `x + v·(σ_{k+1} − σ_k)`.
    pub fn step(&self, v: &Tensor, sample: &Tensor, k: usize) -> CandleResult<Tensor> {
        let dt = self.sigmas[k + 1] - self.sigmas[k];
        sample + v.affine(dt, 0.0)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device, Tensor};

    #[test]
    fn schedule_matches_the_reference_shape() {
        // steps=4, shift=1 → s_k = [1, 0.75, 0.5, 0.25], terminal 0.
        let s = FlowMatchSchedule::new(4, 1.0);
        assert_eq!(s.num_steps(), 4);
        let expect = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (k, want) in expect.iter().enumerate() {
            assert!(
                (s.sigma(k) - want).abs() < 1e-12,
                "σ_{k}={} want {want}",
                s.sigma(k)
            );
        }
    }

    #[test]
    fn shift_bends_the_ladder_upward() {
        let base = FlowMatchSchedule::new(8, DEFAULT_SHIFT);
        let flat = FlowMatchSchedule::new(8, 1.0);
        // The shift pushes intermediate sigmas up toward 1 (more time spent at high noise).
        assert!((base.sigma(0) - 1.0).abs() < 1e-12);
        assert!(base.sigma(4) > flat.sigma(4));
        // Both start at 1 and end at 0.
        assert_eq!(base.sigma(base.num_steps()), 0.0);
        // shift = 3, s = 0.5 → 3·0.5/(1+2·0.5) = 1.5/2 = 0.75.
        let s = FlowMatchSchedule::new(2, 3.0);
        assert!((s.sigma(1) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn euler_step_integrates_velocity_and_lands_on_zero() {
        let s = FlowMatchSchedule::new(2, 1.0);
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 2, 3), DType::F32, &dev).unwrap();
        let v = Tensor::full(2.0f32, (1, 2, 3), &dev).unwrap();
        let dt0 = s.sigma(1) - s.sigma(0);
        let x1 = s.step(&v, &x, 0).unwrap();
        let got = x1.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((got as f64 - (1.0 + 2.0 * dt0)).abs() < 1e-6);
        assert_eq!(s.sigma(s.num_steps()), 0.0);
    }
}
