//! Mochi flow-match scheduler + CFG — port of `MochiPipeline`'s `linear_quadratic_schedule`, the
//! `FlowMatchEulerDiscreteScheduler` set-up (`shift`, `invert_sigmas`) and the denoise-loop CFG combine.
//!
//! The sigma schedule is Genmo's **linear-quadratic** curve (linear for the first half of the steps,
//! quadratic after), which the scheduler then **inverts** — the one branch `invert_sigmas=True` guards,
//! used only by Mochi: `sigmas ← 1 − sigmas`, `timesteps ← 1000·sigmas`, and a terminal `1.0` is
//! appended so the last Euler step has a `sigma_next`. The denoise is 1st-order Euler in the model
//! velocity — `x_next = x + (σ_next − σ_cur)·v`. CFG is true classifier-free guidance over the
//! `[neg, pos]` batch: `uncond + g·(cond − uncond)`.
//!
//! All arithmetic mirrors the reference precision: the schedule is built in f64, cast per-value to f32
//! (the reference stores `sigmas`/`timesteps` as float32), and the Euler `dt` is widened back to f64.

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

use candle_gen::gen_core::render::video::VideoStep;
pub use gen_core_family::mochi::linear_quadratic_schedule;
use gen_core_family::mochi::{inverted_sigmas, MochiEuler};

/// 1st-order Euler flow-match scheduler with Mochi's inverted-sigma schedule.
pub struct MochiScheduler {
    /// `num_steps + 1` sigmas (trailing terminal `1.0`), f32-valued.
    sigmas: Vec<f32>,
    /// `num_steps` model timesteps `1000·(1 − sigma_pre)`, f32-valued.
    timesteps: Vec<f32>,
    step_index: usize,
}

impl MochiScheduler {
    /// New scheduler (`num_train_timesteps = 1000`, `threshold_noise = 0.025` — the Mochi config).
    pub fn new() -> Self {
        Self {
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: 0,
        }
    }

    /// Build the inverted schedule for `num_steps` with the given resolution `shift` (Mochi config
    /// `shift = 1.0`, i.e. no shift). Mirrors `set_timesteps(sigmas=linear_quadratic_schedule(...))`
    /// followed by the `invert_sigmas` branch.
    pub fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = inverted_sigmas(num_steps, shift);
        self.timesteps = self.sigmas[..num_steps]
            .iter()
            .copied()
            .map(MochiEuler::timestep)
            .collect();
        self.step_index = 0;
    }

    /// The `num_steps + 1` sigmas (trailing terminal `1.0`).
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// The `num_steps` model timesteps fed to the transformer (`1000·(1 − sigma_pre)`).
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// One Euler step: `x_next = x + (σ_next − σ_cur)·v`. `model_output` is the post-CFG velocity.
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let i = self.step_index;
        if i + 1 >= self.sigmas.len() {
            return Err(CandleError::Msg(format!(
                "mochi scheduler: step {i} out of range for {} sigmas — call set_timesteps and run \
                 exactly {} step(s)",
                self.sigmas.len(),
                self.sigmas.len().saturating_sub(1)
            )));
        }
        let step = VideoStep {
            index: i,
            sigma: self.sigmas[i],
            next_sigma: self.sigmas[i + 1],
        };
        let output = advance(step, &self.sigmas, model_output, sample)?;
        self.step_index += 1;
        Ok(output)
    }
}

pub(crate) fn advance(
    step: VideoStep,
    sigmas: &[f32],
    model_output: &Tensor,
    sample: &Tensor,
) -> Result<Tensor> {
    if step.index == 0 {
        let nodes: Vec<_> = sigmas.iter().copied().map(f64::from).collect();
        candle_gen::execution::schedule_f64("mochi_flow_euler_v1", &nodes, "inverse_flow_velocity");
        candle_gen::execution::runtime(sample.device(), sample.dtype());
    }
    Ok((sample + model_output.affine(MochiEuler::delta(step), 0.0)?)?)
}

impl Default for MochiScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// True classifier-free guidance over a `[neg, pos]`-ordered batch: split `noise_pred [2, ...]` into
/// `(uncond, cond)` and return `uncond + guidance·(cond − uncond)`. Mirrors the pipeline's
/// `noise_pred_uncond + guidance_scale·(noise_pred_text − noise_pred_uncond)`.
pub fn cfg_combine(noise_pred: &Tensor, guidance: f32) -> Result<Tensor> {
    let b = noise_pred.dim(0)?;
    if b != 2 {
        return Err(CandleError::Msg(format!(
            "mochi cfg_combine: expected a [neg, pos] batch of 2, got batch {b}"
        )));
    }
    let uncond = noise_pred.narrow(0, 0, 1)?;
    let cond = noise_pred.narrow(0, 1, 1)?;
    let delta = (cond - &uncond)?;
    Ok((uncond + delta.affine(guidance as f64, 0.0)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{Device, Tensor};

    #[test]
    fn linear_quadratic_two_steps_matches_reference() {
        // linear_quadratic_schedule(2, 0.025) = [1.0, 0.975] (verified against diffusers).
        let s = linear_quadratic_schedule(2, 0.025);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 0.975).abs() < 1e-12);
    }

    #[test]
    fn inverted_sigma_schedule_matches_reference() {
        // Reference (FlowMatchEulerDiscreteScheduler, invert_sigmas, shift=1) for STEPS=2:
        //   sigmas    = [0.0, 0.024999976, 1.0]
        //   timesteps = [0.0, 24.999977]
        let mut sch = MochiScheduler::new();
        sch.set_timesteps(2, 1.0);
        let sig = sch.sigmas();
        let ts = sch.timesteps();
        assert_eq!(sig.len(), 3);
        assert_eq!(ts.len(), 2);
        assert!(sig[0].abs() < 1e-9, "sigma0 {}", sig[0]);
        assert!((sig[1] - 0.024_999_976).abs() < 1e-6, "sigma1 {}", sig[1]);
        assert!((sig[2] - 1.0).abs() < 1e-9, "terminal {}", sig[2]);
        assert!(ts[0].abs() < 1e-4, "t0 {}", ts[0]);
        assert!((ts[1] - 24.999_977).abs() < 1e-2, "t1 {}", ts[1]);
    }

    #[test]
    fn one_step_starts_at_zero_and_reaches_the_terminal_one() {
        let mut scheduler = MochiScheduler::new();
        scheduler.set_timesteps(1, 1.0);
        assert_eq!(scheduler.sigmas(), [0.0, 1.0]);
        assert_eq!(scheduler.timesteps(), [0.0]);
    }

    #[test]
    fn euler_step_advances_by_dt() {
        let dev = Device::Cpu;
        let mut sch = MochiScheduler::new();
        sch.set_timesteps(2, 1.0);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let v = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (1, 4), &dev).unwrap();
        let out = sch.step(&v, &x).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dt = 0.024_999_976_f32;
        for (g, base) in got.iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((g - (base + dt)).abs() < 1e-5, "{g} vs {}", base + dt);
        }
        // Step 1 ok; a third step must error (only 2 steps configured, 3 sigmas).
        let _ = sch.step(&v, &out).unwrap();
        assert!(sch.step(&v, &x).is_err(), "step 2 must be out of range");
    }

    #[test]
    fn cfg_combine_is_true_guidance() {
        let dev = Device::Cpu;
        // batch [neg, pos]; uncond=[1,2], cond=[5,8].
        let np = Tensor::from_vec(vec![1.0f32, 2.0, 5.0, 8.0], (2, 2), &dev).unwrap();
        let out = cfg_combine(&np, 4.5).unwrap();
        assert_eq!(out.dims(), &[1, 2]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // [1 + 4.5·(5−1), 2 + 4.5·(8−2)] = [19.0, 29.0].
        assert!((got[0] - 19.0).abs() < 1e-4, "{}", got[0]);
        assert!((got[1] - 29.0).abs() < 1e-4, "{}", got[1]);
    }
}
