//! The MOSS flow-matching schedule (sc-12841) — a faithful port of the reference
//! `FlowMatchScheduler` in its shipped configuration (`shift=5.0`, `sigma_min=0.0`,
//! `extra_one_step=true`, `num_train_timesteps=1000`) with the Euler update the reference
//! pipeline integrates with.
//!
//! Schedule: `s_k = linspace(1, σ_min, steps+1)[k]` for `k = 0..steps` (the `extra_one_step`
//! shape drops the terminal element, so exactly `steps` sigmas), then the classic shift
//! `σ_k = shift·s_k / (1 + (shift−1)·s_k)`. The DiT timestep at step `k` is
//! `σ_k · num_train_timesteps`. The Euler update is `x ← x + v·(σ_{k+1} − σ_k)` with the final
//! step jumping straight to the boundary `σ_steps := 0` (`FlowMatchScheduler.step`'s
//! `to_final` branch — with `extra_one_step` the stored schedule has no terminal element, so
//! the last transition always takes it).

use candle_audio::candle_core::{Result as CandleResult, Tensor};

use crate::config::SchedulerConfig;

/// The precomputed σ ladder for one denoise run.
#[derive(Debug, Clone)]
pub struct FlowMatchSchedule {
    /// `steps + 1` sigmas: the `steps` shifted schedule points plus the terminal `0.0`.
    sigmas: Vec<f64>,
    num_train_timesteps: u32,
}

impl FlowMatchSchedule {
    /// Build the schedule for `steps` inference steps. `shift` overrides the config's shift
    /// when supplied (the request's `scheduler_shift` knob — the reference `sigma_shift`
    /// argument).
    pub fn new(cfg: &SchedulerConfig, steps: usize, shift: Option<f64>) -> Self {
        let steps = steps.max(1);
        let shift = shift.unwrap_or(cfg.shift);
        let (start, end) = (1.0f64, cfg.sigma_min);
        // linspace(start, end, steps+1)[:steps] — the reference `extra_one_step=true` shape.
        // (extra_one_step=false would be linspace(start, end, steps); the pinned scheduler
        // config ships true, and `SchedulerConfig::extra_one_step` is honored here.)
        let denom = if cfg.extra_one_step {
            steps as f64
        } else {
            (steps as f64 - 1.0).max(1.0)
        };
        let mut sigmas: Vec<f64> = (0..steps)
            .map(|k| {
                let s = start + (end - start) * (k as f64) / denom;
                shift * s / (1.0 + (shift - 1.0) * s)
            })
            .collect();
        sigmas.push(0.0); // terminal boundary (`to_final`)
        Self {
            sigmas,
            num_train_timesteps: cfg.num_train_timesteps,
        }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// σ at step `k` (`k ≤ steps`; `σ_steps = 0`).
    pub fn sigma(&self, k: usize) -> f64 {
        self.sigmas[k]
    }

    /// The DiT conditioning timestep at step `k` (`σ_k · num_train_timesteps`).
    pub fn timestep(&self, k: usize) -> f64 {
        self.sigmas[k] * self.num_train_timesteps as f64
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
    use candle_audio::candle_core::{Device, Tensor};

    fn cfg() -> SchedulerConfig {
        serde_json::from_str(
            r#"{"shift": 5.0, "sigma_min": 0.0, "extra_one_step": true,
                "num_train_timesteps": 1000}"#,
        )
        .unwrap()
    }

    #[test]
    fn schedule_matches_the_reference_shape() {
        // Reference: sigmas = linspace(1, 0, N+1)[:-1] then σ = 5s/(1+4s); timesteps = σ·1000.
        let s = FlowMatchSchedule::new(&cfg(), 4, None);
        assert_eq!(s.num_steps(), 4);
        // s_k = [1.0, 0.75, 0.5, 0.25] → σ_k = [1.0, 0.9375, 0.83333…, 0.625], terminal 0.
        let expect = [1.0, 0.9375, 5.0 * 0.5 / 3.0, 0.625, 0.0];
        for (k, want) in expect.iter().enumerate() {
            assert!(
                (s.sigma(k) - want).abs() < 1e-12,
                "σ_{k} = {} (want {want})",
                s.sigma(k)
            );
        }
        assert!((s.timestep(0) - 1000.0).abs() < 1e-9);
        assert!((s.timestep(1) - 937.5).abs() < 1e-9);
    }

    #[test]
    fn shift_override_reshapes_the_ladder() {
        let base = FlowMatchSchedule::new(&cfg(), 4, None);
        let flat = FlowMatchSchedule::new(&cfg(), 4, Some(1.0));
        // shift=1 is the identity map: σ_k = s_k exactly.
        assert!((flat.sigma(1) - 0.75).abs() < 1e-12);
        assert!(base.sigma(1) > flat.sigma(1));
    }

    #[test]
    fn euler_step_integrates_the_velocity_and_lands_on_zero() {
        let s = FlowMatchSchedule::new(&cfg(), 2, None);
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 2, 3), candle_audio::candle_core::DType::F32, &dev).unwrap();
        let v = Tensor::full(2.0f32, (1, 2, 3), &dev).unwrap();
        // Step 0: dt = σ_1 − σ_0.
        let dt0 = s.sigma(1) - s.sigma(0);
        let x1 = s.step(&v, &x, 0).unwrap();
        let got = x1.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((got as f64 - (1.0 + 2.0 * dt0)).abs() < 1e-6);
        // The last transition ends on the terminal σ = 0 boundary.
        assert_eq!(s.sigma(s.num_steps()), 0.0);
    }

    #[test]
    fn single_step_schedule_is_one_full_jump() {
        let s = FlowMatchSchedule::new(&cfg(), 1, None);
        assert_eq!(s.num_steps(), 1);
        assert!((s.sigma(0) - 1.0).abs() < 1e-12);
        assert_eq!(s.sigma(1), 0.0);
    }
}
