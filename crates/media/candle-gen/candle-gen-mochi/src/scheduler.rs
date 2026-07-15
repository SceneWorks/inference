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
//! **Why not `candle_gen::run_curated_sampler`:** Mochi's inverted schedule *ascends* from `σ = 0`
//! (`sigmas = [0, 0.025, …, 1.0]`), and its step-0 (`σ = 0`, `dt = 0.025`) is a **real** step. The
//! curated flow solvers integrate a *descending* VE schedule and treat `σ == 0` as terminal — the plain
//! `Euler` short-circuits a leading zero to `x0` ("no real schedule starts here") and `to_d` divides by
//! `σ`. Forcing Mochi through them would drop step-0's `0.025·v(t=0)` contribution and/or divide by
//! zero, so the loop is run directly (mirroring `mlx-gen-mochi`), with the CFG recombine still done
//! inside a `predict`-style closure and cooperative cancel/progress plumbed by hand in the pipeline.
//!
//! All arithmetic mirrors the reference precision: the schedule is built in f64, cast per-value to f32
//! (the reference stores `sigmas`/`timesteps` as float32), and the Euler `dt` is widened back to f64.

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

/// Genmo's linear-quadratic sigma schedule (`linear_quadratic_schedule`, `linear_steps = num_steps/2`).
///
/// Returns `num_steps` values in `[0, 1]`, descending (`sigma[0] = 1.0`). This is the **pre-inversion**
/// schedule; [`MochiScheduler::set_timesteps`] applies the `invert_sigmas` transform on top.
pub fn linear_quadratic_schedule(num_steps: usize, threshold_noise: f64) -> Vec<f64> {
    let n = num_steps as f64;
    let linear_steps = num_steps / 2;
    let ls = linear_steps as f64;

    // Linear segment: sigma_i = i·threshold / linear_steps for i in 0..linear_steps.
    let mut sched: Vec<f64> = (0..linear_steps)
        .map(|i| i as f64 * threshold_noise / ls)
        .collect();

    // Quadratic segment for i in linear_steps..num_steps.
    let threshold_noise_step_diff = ls - threshold_noise * n;
    let quadratic_steps = (num_steps - linear_steps) as f64;
    let quadratic_coef = threshold_noise_step_diff / (ls * quadratic_steps * quadratic_steps);
    let linear_coef =
        threshold_noise / ls - 2.0 * threshold_noise_step_diff / (quadratic_steps * quadratic_steps);
    let konst = quadratic_coef * (ls * ls);
    for i in linear_steps..num_steps {
        let i = i as f64;
        sched.push(quadratic_coef * i * i + linear_coef * i + konst);
    }

    // sigma_schedule = 1 − x.
    sched.iter().map(|x| 1.0 - x).collect()
}

/// 1st-order Euler flow-match scheduler with Mochi's inverted-sigma schedule.
pub struct MochiScheduler {
    num_train_timesteps: usize,
    threshold_noise: f64,
    /// `num_steps + 1` sigmas (trailing terminal `1.0`), f32-valued.
    sigmas: Vec<f32>,
    /// `num_steps` model timesteps `1000·(1 − sigma_pre)`, f32-valued.
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
}

impl MochiScheduler {
    /// New scheduler (`num_train_timesteps = 1000`, `threshold_noise = 0.025` — the Mochi config).
    pub fn new() -> Self {
        Self {
            num_train_timesteps: 1000,
            threshold_noise: 0.025,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
        }
    }

    /// Build the inverted schedule for `num_steps` with the given resolution `shift` (Mochi config
    /// `shift = 1.0`, i.e. no shift). Mirrors `set_timesteps(sigmas=linear_quadratic_schedule(...))`
    /// followed by the `invert_sigmas` branch.
    pub fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        let pre = linear_quadratic_schedule(num_steps, self.threshold_noise);
        // Reference casts the custom sigmas to float32 before the transforms.
        let mut sig: Vec<f32> = pre.iter().map(|&x| x as f32).collect();
        // Resolution shift: sigma ← shift·sigma / (1 + (shift−1)·sigma) (identity at shift = 1).
        if shift != 1.0 {
            for s in sig.iter_mut() {
                *s = shift * *s / (1.0 + (shift - 1.0) * *s);
            }
        }
        // invert_sigmas: sigma ← 1 − sigma; timesteps ← 1000·sigma; append terminal 1.0.
        let inv: Vec<f32> = sig.iter().map(|&s| 1.0 - s).collect();
        self.timesteps = inv
            .iter()
            .map(|&s| s * self.num_train_timesteps as f32)
            .collect();
        let mut sigmas = inv;
        sigmas.push(1.0);
        self.sigmas_f64 = sigmas.iter().map(|&s| s as f64).collect();
        self.sigmas = sigmas;
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
        if i + 1 >= self.sigmas_f64.len() {
            return Err(CandleError::Msg(format!(
                "mochi scheduler: step {i} out of range for {} sigmas — call set_timesteps and run \
                 exactly {} step(s)",
                self.sigmas_f64.len(),
                self.sigmas_f64.len().saturating_sub(1)
            )));
        }
        let dt = self.sigmas_f64[i + 1] - self.sigmas_f64[i];
        let x_next = (sample + model_output.affine(dt, 0.0)?)?;
        self.step_index += 1;
        Ok(x_next)
    }
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
