//! The curated callback-solver library (epic 7114, P1, sc-7117): the integration-method half of the
//! unified framework. Each solver is a [`Sampler`] impl driving a `denoise(x, σ) -> x0` callback over
//! a sigma schedule, advancing the latents through [`LatentOps`] only — backend-neutral, prediction-
//! type-agnostic (the [`super::model_sampling::ModelSampling`] already mapped the raw output to `x0`).
//!
//! Faithful ports of ComfyUI / k-diffusion in the unified VE-sigma space (`λ = −ln σ`, `sigma_fn(t) =
//! e^{−t}`): `euler` (in [`super::unified`]), `euler_ancestral`, `heun`, `dpmpp_2m`, `dpmpp_sde`,
//! `uni_pc`, `lcm`, `ddim` (decision 2). The multistep solvers (`dpmpp_2m`, `uni_pc`) lift the structure
//! of Wan's flow-mode `dpmpp_2m`/`uni_pc` (`mlx-gen-wan/src/scheduler.rs`) generalised to VE space —
//! where each reduces to plain Euler at first order, so a constant-velocity field integrates EXACTLY
//! (the per-solver coherence test).
//!
//! The high-order pair `rk6_7s` / `abnorsett_4m` (epic 20414, sc-20417) is instead derived directly
//! from the published numerical-methods literature (Butcher's 1964 6(7) Runge–Kutta tableau;
//! Nørsett's exponential Adams–Bashforth coefficients) — RES4LYF supplies only the *naming*
//! convention, never code — and validated against independently generated float64 goldens
//! (`tests/fixtures/rk6_abnorsett_golden.json`) plus exact order-condition / quadrature-exactness
//! tests.

use super::unified::{is_terminal, to_d, DenoiseFn, Euler, Sampler};
use super::LatentOps;
use crate::Result;

// =================================================================================================
// Scalar helpers (host f64) — the VE-sigma `λ = −ln σ` time and the ancestral noise split.
// =================================================================================================

/// `λ(σ) = −ln σ` — the VE half-log-SNR "time" the DPM/UniPC multistep solvers integrate in.
#[inline]
fn lambda(sigma: f32) -> f64 {
    -(sigma.max(1e-12) as f64).ln()
}

/// `σ(λ) = e^{−λ}` — inverse of [`lambda`].
#[inline]
fn sigma_of(l: f64) -> f64 {
    (-l).exp()
}

/// k-diffusion `get_ancestral_step`: split the `σ_from -> σ_to` step into a deterministic descent to
/// `σ_down` plus `σ_up` of fresh noise, scaled by `eta`. `σ_to == 0` -> `(0, 0)` (no noise on the
/// terminal step).
fn ancestral_step(sigma_from: f32, sigma_to: f32, eta: f32) -> (f32, f32) {
    if sigma_to <= 0.0 {
        return (0.0, 0.0);
    }
    let (sf, st) = (sigma_from as f64, sigma_to as f64);
    let su = (eta as f64 * (st * st * (sf * sf - st * st) / (sf * sf)).max(0.0).sqrt()).min(st);
    let sd = (st * st - su * su).max(0.0).sqrt();
    (sd as f32, su as f32)
}

// =================================================================================================
// euler_ancestral — k-diffusion `sample_euler_ancestral` (1st order, stochastic).
// =================================================================================================

/// Ancestral Euler: a forward-Euler descent to `σ_down` plus `σ_up` of fresh noise each step
/// (`eta = 1`). The added noise makes it non-deterministic-in-σ but seed-reproducible.
#[derive(Clone, Copy, Debug)]
pub struct EulerAncestral {
    /// Stochasticity scale (k-diffusion default `1.0`).
    pub eta: f32,
}

impl Default for EulerAncestral {
    fn default() -> Self {
        Self { eta: 1.0 }
    }
}

impl<L: LatentOps> Sampler<L> for EulerAncestral {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            let (sd, su) = ancestral_step(sigma, sigmas[i + 1], self.eta);
            let d = to_d(ops, &x, sigma, &x0)?;
            x = ops.axpy(1.0, &x, sd - sigma, &d)?;
            if su > 0.0 {
                let noise = ops.randn_like(&x, seed, i)?;
                x = ops.axpy(1.0, &x, su, &noise)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// heun — k-diffusion `sample_heun` (2nd order, deterministic; 2 model evals/step).
// =================================================================================================

/// Heun's method: an Euler predictor to `σ_{i+1}`, a second model eval there, then a step with the
/// averaged derivative `(d + d')/2`. The terminal step (`σ_{i+1} = 0`) falls back to plain Euler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Heun;

impl<L: LatentOps> Sampler<L> for Heun {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            let d = to_d(ops, &x, sigma, &x0)?;
            let dt = s_next - sigma;
            if s_next == 0.0 {
                x = ops.axpy(1.0, &x, dt, &d)?;
            } else {
                let x2 = ops.axpy(1.0, &x, dt, &d)?;
                let x0_2 = denoise(&x2, s_next)?;
                let d2 = to_d(ops, &x2, s_next, &x0_2)?;
                let d_prime = ops.axpy(0.5, &d, 0.5, &d2)?;
                x = ops.axpy(1.0, &x, dt, &d_prime)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// dpmpp_2m — k-diffusion `sample_dpmpp_2m` (DPM-Solver++(2M), 2nd-order multistep).
// =================================================================================================

/// DPM-Solver++(2M): a multistep solver that reuses the previous step's denoised estimate for a
/// 2nd-order update with a single model eval per step. Falls back to 1st order on the first and
/// terminal steps. Wan's flow-mode `dpmpp_2m` is the structural reference (generalised to VE space).
#[derive(Clone, Copy, Debug, Default)]
pub struct Dpmpp2m;

impl<L: LatentOps> Sampler<L> for Dpmpp2m {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        let mut old_x0: Option<L::Latent> = None;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                // Degenerate leading σ==0 (no real schedule starts here): land on x0, avoid the
                // λ(0)/coeff_x = s_next/σ division. Mirrors every sibling solver's leading guard.
                x = x0.clone();
                old_x0 = Some(x0);
                continue;
            }
            if s_next == 0.0 {
                // Terminal: land on the denoised estimate (the 1st-order limit).
                x = x0.clone();
                old_x0 = Some(x0);
                continue;
            }
            let t = lambda(sigma);
            let t_next = lambda(s_next);
            let h = t_next - t;
            let coeff_x = s_next / sigma; // sigma_fn(t_next)/sigma_fn(t)
            let coeff_x0 = (-(-h).exp_m1()) as f32; // −expm1(−h)
            let denoised_d = match &old_x0 {
                None => x0.clone(),
                Some(old) => {
                    let h_last = t - lambda(sigmas[i - 1]);
                    let r = h_last / h;
                    // denoised_d = (1 + 1/(2r))·x0 − (1/(2r))·old_x0
                    let a = 1.0 + 1.0 / (2.0 * r);
                    let b = -1.0 / (2.0 * r);
                    ops.axpy(a as f32, &x0, b as f32, old)?
                }
            };
            x = ops.axpy(coeff_x, &x, coeff_x0, &denoised_d)?;
            old_x0 = Some(x0);
        }
        Ok(x)
    }
}

// =================================================================================================
// dpmpp_sde — k-diffusion `sample_dpmpp_sde` (stochastic midpoint, 2nd order; 2 evals/step).
// =================================================================================================

/// DPM-Solver++ SDE (midpoint, `r = 1/2`): a stochastic 2nd-order solver — a noised half-step, a
/// second model eval there, then a noised full step using the midpoint denoised. Seed-reproducible.
#[derive(Clone, Copy, Debug)]
pub struct DpmppSde {
    /// Stochasticity scale (k-diffusion default `1.0`).
    pub eta: f32,
}

impl Default for DpmppSde {
    fn default() -> Self {
        Self { eta: 1.0 }
    }
}

impl<L: LatentOps> Sampler<L> for DpmppSde {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        const R: f64 = 0.5;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            if s_next == 0.0 {
                // Terminal: plain Euler onto the denoised estimate.
                let d = to_d(ops, &x, sigma, &x0)?;
                x = ops.axpy(1.0, &x, s_next - sigma, &d)?;
                continue;
            }
            let t = lambda(sigma);
            let t_next = lambda(s_next);
            let h = t_next - t;
            let s = t + h * R; // midpoint time
            let sigma_s = sigma_of(s) as f32;

            // Half-step to the midpoint (noised).
            let (sd1, su1) = ancestral_step(sigma, sigma_s, self.eta);
            let s_lam = lambda(sd1);
            let coeff_x = (sigma_of(s_lam) / sigma_of(t)) as f32; // sd1/sigma
            let coeff_x0 = -(t - s_lam).exp_m1() as f32;
            let mut x2 = ops.axpy(coeff_x, &x, coeff_x0, &x0)?;
            if su1 > 0.0 {
                let noise1 = ops.randn_like(&x2, seed, 2 * i)?;
                x2 = ops.axpy(1.0, &x2, su1, &noise1)?;
            }
            let x0_2 = denoise(&x2, sigma_s)?;

            // Full step using the midpoint denoised (fac = 1/(2r) = 1 -> denoised_d = x0_2), noised.
            let (sd2, su2) = ancestral_step(sigma, s_next, self.eta);
            let t_next_lam = lambda(sd2);
            let coeff_x = (sigma_of(t_next_lam) / sigma_of(t)) as f32; // sd2/sigma
            let coeff_x0 = -(t - t_next_lam).exp_m1() as f32;
            x = ops.axpy(coeff_x, &x, coeff_x0, &x0_2)?;
            if su2 > 0.0 {
                let noise2 = ops.randn_like(&x, seed, 2 * i + 1)?;
                x = ops.axpy(1.0, &x, su2, &noise2)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// uni_pc — UniPC predictor-corrector (order 2, bh2). Wan's flow-mode uni_pc generalised to VE space.
// =================================================================================================

/// UniPC (order 2): a multistep predictor-corrector. Each step refines the previous prediction with
/// the fresh denoised estimate (the corrector, a trapezoidal blend), then predicts the next sample
/// with a 2nd-order extrapolation. At order 1 / constant denoised it reduces to Euler. Wan's flow-mode
/// `uni_pc` is the structural reference (generalised to VE `λ = −ln σ`, `α = 1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct UniPc;

impl<L: LatentOps> Sampler<L> for UniPc {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        let mut prev_x0: Option<L::Latent> = None; // x0_{i-1}
        let mut last_sample: Option<L::Latent> = None; // corrected x_{i-1}
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;

            // Corrector: refine x_i (predicted at step i-1) using the fresh x0_i. Order-2 corrector
            // is the trapezoidal blend 0.5·(x0_{i-1} + x0_i).
            if let (Some(prev), Some(ls)) = (&prev_x0, &last_sample) {
                let s_prev = sigmas[i - 1];
                let phi = (sigma / s_prev) - 1.0; // expm1(−h_corr)
                let blend = ops.axpy(0.5, prev, 0.5, &x0)?; // 0.5·(x0_{i-1}+x0_i)
                x = ops.axpy(sigma / s_prev, ls, -phi, &blend)?;
            }

            if s_next == 0.0 {
                // Terminal: land on the (corrected-then-)denoised estimate.
                x = x0;
                break;
            }

            // Predictor: σ_i -> σ_{i+1}, 2nd order when history is available.
            let t = lambda(sigma);
            let t_next = lambda(s_next);
            let h = t_next - t;
            let phi = (s_next / sigma) - 1.0; // expm1(−h)
            let mut x_next = ops.axpy(s_next / sigma, &x, -phi, &x0)?;
            if let Some(prev) = &prev_x0 {
                let r = (lambda(sigmas[i - 1]) - t) / h; // (λ_{i-1} − λ_i)/h
                let d1 = ops.axpy((1.0 / r) as f32, prev, (-1.0 / r) as f32, &x0)?; // (x0_{i-1}−x0_i)/r
                x_next = ops.axpy(1.0, &x_next, -phi * 0.5, &d1)?;
            }

            prev_x0 = Some(x0);
            last_sample = Some(x);
            x = x_next;
        }
        Ok(x)
    }
}

// =================================================================================================
// lcm — ComfyUI `sample_lcm` (consistency: x <- denoised, re-noise between steps).
// =================================================================================================

/// Latent Consistency Model sampler (ComfyUI `sample_lcm`): each step jumps straight to the denoised
/// estimate, then re-noises to `σ_{i+1}`. ~2–8 steps. (Distinct from the legacy diffusers-faithful
/// `LcmPolicy` accel path, which the engines keep for distilled LoRAs; this is the user-selectable
/// curated sampler.)
#[derive(Clone, Copy, Debug, Default)]
pub struct Lcm;

impl<L: LatentOps> Sampler<L> for Lcm {
    fn sample(
        &self,
        ops: &L,
        ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let x0 = denoise(&x, sigmas[i])?;
            x = x0;
            let s_next = sigmas[i + 1];
            if s_next > 0.0 {
                let noise = ops.randn_like(&x, seed, i)?;
                // Re-noise through the model's own noise_scaling (ComfyUI `sample_lcm`): `x = k_x0·x0 +
                // k_noise·noise`. VE/EDM/DDPM keep x0 at full scale (k_x0 = 1, k_noise = σ); FLOW uses the
                // convex blend (k_x0 = 1−σ, k_noise = σ) so a flow-distilled student is re-noised in its
                // training regime instead of the OOD VE form (sc-7491).
                let (k_noise, k_x0) = ms.noise_scaling_coeffs(s_next);
                x = ops.axpy(k_x0, &x, k_noise, &noise)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// ddim — DDIM (η = 0): the deterministic x0-interpolation step.
// =================================================================================================

/// DDIM (η = 0): the deterministic update `x_{i+1} = (σ_{i+1}/σ_i)·x_i + (1 − σ_{i+1}/σ_i)·x0`. In the
/// unified VE-sigma space this coincides with Euler, but is exposed as a named curated sampler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ddim;

impl<L: LatentOps> Sampler<L> for Ddim {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            let ratio = s_next / sigma;
            x = ops.axpy(ratio, &x, 1.0 - ratio, &x0)?;
        }
        Ok(x)
    }
}

// =================================================================================================
// er_sde — ComfyUI / k-diffusion `sample_er_sde` (Extended Reverse-Time SDE, ER-SDE-Solver-3).
// =================================================================================================

/// The ER-SDE noise scaler `φ(λ) = λ·(e^{λ^{0.3}} + 10)` (ComfyUI `default_er_sde_noise_scaler`). In
/// the unified VE-sigma space the ER-SDE "λ" IS the schedule σ (`er_lambda_t = σ_t/α_t`, `α = 1`), so
/// this is evaluated at σ.
#[inline]
fn er_sde_noise_scaler(l: f64) -> f64 {
    l * (l.powf(0.3).exp() + 10.0)
}

/// Extended Reverse-Time SDE solver (ER-SDE-Solver-3): a faithful port of ComfyUI / k-diffusion
/// `sample_er_sde` reduced to the unified VE-sigma space (`α = 1`, so `er_lambda_t = σ_t` and the
/// per-step `alpha_t` factors drop out). A stochastic multistep solver — a 1st-order (Euler-like)
/// blend `x ← r·x + (1−r)·x0` with `r = φ(σ_{i+1})/φ(σ_i)`, then 2nd- and 3rd-order corrections built
/// from finite-difference denoised derivatives and a 200-point σ-integral of `1/φ`, then fresh
/// per-step Gaussian noise. Anima's recommended sampler. Reference:
/// <https://github.com/QinpengCui/ER-SDE-Solver> (arXiv:2309.06169) via ComfyUI's `sample_er_sde`.
#[derive(Clone, Copy, Debug)]
pub struct ErSde {
    /// Fresh-noise scale (ComfyUI default `1.0`; `0` makes the solver deterministic).
    pub s_noise: f32,
    /// Highest correction order to use (ComfyUI `max_stage`, default `3` = ER-SDE-Solver-3).
    pub max_stage: u8,
}

impl Default for ErSde {
    fn default() -> Self {
        Self {
            s_noise: 1.0,
            max_stage: 3,
        }
    }
}

impl<L: LatentOps> Sampler<L> for ErSde {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        const NUM_INTEGRATION_POINTS: usize = 200;
        let max_stage = self.max_stage.max(1) as usize;
        let mut old_denoised: Option<L::Latent> = None;
        let mut old_denoised_d: Option<L::Latent> = None;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) || s_next == 0.0 {
                // Terminal (or degenerate leading σ==0): land on the denoised estimate, no noise.
                x = x0.clone();
                old_denoised = Some(x0);
                continue;
            }
            let stage_used = max_stage.min(i + 1);
            // VE-sigma space: er_lambda == σ, α == 1 (so r_alpha == 1 and the α_t prefactors drop out).
            let lam_s = sigma as f64;
            let lam_t = s_next as f64;
            let phi_s = er_sde_noise_scaler(lam_s);
            let phi_t = er_sde_noise_scaler(lam_t);
            let r = phi_t / phi_s;

            // Stage 1 (Euler): x = r·x + (1 − r)·x0.
            let mut x_next = ops.axpy(r as f32, &x, (1.0 - r) as f32, &x0)?;

            if stage_used >= 2 {
                let dt = lam_t - lam_s; // < 0 (σ descends)
                let step = -dt / NUM_INTEGRATION_POINTS as f64; // > 0
                                                                // 200-point Riemann integrals over [σ_{i+1}, σ_i): λ_pos[k] = σ_{i+1} + k·step.
                let (mut s_int, mut s_u_int) = (0.0_f64, 0.0_f64);
                for k in 0..NUM_INTEGRATION_POINTS {
                    let lp = lam_t + k as f64 * step;
                    let phi = er_sde_noise_scaler(lp);
                    s_int += 1.0 / phi;
                    s_u_int += (lp - lam_s) / phi;
                }
                s_int *= step;
                s_u_int *= step;

                // Stage 2: denoised_d = (x0 − old_x0)/(σ_i − σ_{i-1}); x += (dt + s·φ(σ_{i+1}))·denoised_d.
                let inv_d = 1.0 / (lam_s - sigmas[i - 1] as f64);
                let old = old_denoised
                    .as_ref()
                    .expect("stage_used>=2 implies i>=1, so old_denoised is set");
                let denoised_d = ops.axpy(inv_d as f32, &x0, -inv_d as f32, old)?;
                let c2 = (dt + s_int * phi_t) as f32;
                x_next = ops.axpy(1.0, &x_next, c2, &denoised_d)?;

                if stage_used >= 3 {
                    // Stage 3: denoised_u = (denoised_d − old_denoised_d)/((σ_i − σ_{i-2})/2);
                    //          x += (dt²/2 + s_u·φ(σ_{i+1}))·denoised_u.
                    let inv_u = 1.0 / ((lam_s - sigmas[i - 2] as f64) / 2.0);
                    let old_d = old_denoised_d
                        .as_ref()
                        .expect("stage_used>=3 implies i>=2, so old_denoised_d is set");
                    let denoised_u = ops.axpy(inv_u as f32, &denoised_d, -inv_u as f32, old_d)?;
                    let c3 = (dt * dt / 2.0 + s_u_int * phi_t) as f32;
                    x_next = ops.axpy(1.0, &x_next, c3, &denoised_u)?;
                }
                old_denoised_d = Some(denoised_d);
            }

            // Fresh per-step noise: x += s_noise·√max(σ_{i+1}² − σ_i²·r², 0)·ε (nan_to_num guard).
            x = x_next;
            if self.s_noise > 0.0 {
                let var = lam_t * lam_t - lam_s * lam_s * r * r;
                let noise_coeff = self.s_noise as f64 * var.max(0.0).sqrt();
                if noise_coeff != 0.0 {
                    let noise = ops.randn_like(&x, seed, i)?;
                    x = ops.axpy(1.0, &x, noise_coeff as f32, &noise)?;
                }
            }
            old_denoised = Some(x0);
        }
        Ok(x)
    }
}

// =================================================================================================
// dpmpp_2m_sde — ComfyUI / k-diffusion `sample_dpmpp_2m_sde` (DPM-Solver++(2M) SDE, midpoint).
// =================================================================================================

/// DPM-Solver++(2M) SDE (midpoint variant): a faithful port of ComfyUI / k-diffusion
/// `sample_dpmpp_2m_sde` reduced to the unified VE-sigma space (`α_t = 1`). A stochastic 2nd-order
/// multistep solver — like [`Dpmpp2m`] it reuses the previous step's denoised estimate for the
/// 2nd-order term, but adds an `η`-scaled SDE contraction plus fresh Gaussian noise each step. Anima's
/// creative alternative (`dpmpp_2m_sde_gpu`). Falls back to 1st order on the first step and to a plain
/// denoise on the terminal step. Reference: ComfyUI `sample_dpmpp_2m_sde` (`solver_type='midpoint'`).
#[derive(Clone, Copy, Debug)]
pub struct Dpmpp2mSde {
    /// SDE stochasticity scale (ComfyUI default `1.0`).
    pub eta: f32,
    /// Fresh-noise scale (ComfyUI default `1.0`).
    pub s_noise: f32,
}

impl Default for Dpmpp2mSde {
    fn default() -> Self {
        Self {
            eta: 1.0,
            s_noise: 1.0,
        }
    }
}

impl<L: LatentOps> Sampler<L> for Dpmpp2mSde {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        let eta = self.eta as f64;
        let mut old_denoised: Option<L::Latent> = None;
        let mut h_last: Option<f64> = None;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) || s_next == 0.0 {
                // Terminal (or degenerate leading σ==0): land on the denoised estimate, no noise.
                x = x0.clone();
                old_denoised = Some(x0);
                continue;
            }
            // VE-sigma space: λ = −ln σ, α_t = 1.
            let h = lambda(s_next) - lambda(sigma); // > 0
            let h_eta = h * (eta + 1.0);
            // x = (σ_{i+1}/σ_i)·e^{−ηh}·x + (1 − e^{−(η+1)h})·x0.
            let coeff_x = (s_next as f64 / sigma as f64 * (-eta * h).exp()) as f32;
            let coeff_d = (-(-h_eta).exp_m1()) as f32; // 1 − e^{−h_eta}
            let mut x_next = ops.axpy(coeff_x, &x, coeff_d, &x0)?;
            // Midpoint 2nd-order correction once a previous denoised estimate exists.
            if let (Some(old), Some(hl)) = (&old_denoised, h_last) {
                let r = hl / h; // h_last / h
                let c = (0.5 * (-(-h_eta).exp_m1()) / r) as f32;
                let diff = ops.axpy(1.0, &x0, -1.0, old)?; // x0 − old_x0
                x_next = ops.axpy(1.0, &x_next, c, &diff)?;
            }
            // Fresh per-step noise: x += σ_{i+1}·√(1 − e^{−2ηh})·s_noise·ε.
            x = x_next;
            if eta > 0.0 && self.s_noise > 0.0 {
                let noise_coeff =
                    s_next as f64 * (-(-2.0 * eta * h).exp_m1()).sqrt() * self.s_noise as f64;
                if noise_coeff != 0.0 {
                    let noise = ops.randn_like(&x, seed, i)?;
                    x = ops.axpy(1.0, &x, noise_coeff as f32, &noise)?;
                }
            }
            old_denoised = Some(x0);
            h_last = Some(h);
        }
        Ok(x)
    }
}

// =================================================================================================
// rk6_7s — 6th-order, 7-stage explicit Runge-Kutta (Butcher 1964). RES4LYF-style naming (sc-20417).
// =================================================================================================

/// Butcher's classical 6th-order, 7-stage explicit Runge–Kutta tableau — the `c` (abscissa) column.
///
/// Source: J. C. Butcher, *On Runge-Kutta processes of high order*, J. Austral. Math. Soc. 4 (1964),
/// pp. 179–194 (the 7-stage order-6 process; seven stages is the minimum for order 6 — the Butcher
/// barrier). The exact rational tableau is widely reproduced in the numerical-ODE literature (e.g.
/// Butcher, *Numerical Methods for Ordinary Differential Equations*).
///
/// **Naming ambiguity, documented (sc-20417):** the id `rk6_7s` follows the RES4LYF ComfyUI
/// extension's `<order>_<stages>` sampler-family naming, which does not pin one specific published
/// 6(7) tableau. This implementation deliberately commits to Butcher's classical 1964 tableau (a
/// published, exactly-rational choice) and encodes that choice in the regression goldens
/// (`tests/fixtures/rk6_abnorsett_golden.json`); it is a from-the-literature derivation, not a port
/// of RES4LYF code. The `rk6_7s_tableau_satisfies_all_order_conditions_through_order_6` test
/// verifies every rooted-tree order condition up to order 6 against these constants, so a single
/// mistyped coefficient fails the suite.
const RK6_7S_C: [f64; 7] = [
    0.0,
    1.0 / 3.0,
    2.0 / 3.0,
    1.0 / 3.0,
    1.0 / 2.0,
    1.0 / 2.0,
    1.0,
];

/// The strictly-lower-triangular `a` matrix of the Butcher (1964) 6(7) tableau (row `i` holds
/// `a_i1..a_i6`; unused upper entries are `0`). Row sums equal [`RK6_7S_C`] (verified in tests).
const RK6_7S_A: [[f64; 6]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 12.0, 1.0 / 3.0, -1.0 / 12.0, 0.0, 0.0, 0.0],
    [-1.0 / 16.0, 9.0 / 8.0, -3.0 / 16.0, -3.0 / 8.0, 0.0, 0.0],
    [0.0, 9.0 / 8.0, -3.0 / 8.0, -3.0 / 4.0, 1.0 / 2.0, 0.0],
    [
        9.0 / 44.0,
        -9.0 / 11.0,
        63.0 / 44.0,
        18.0 / 11.0,
        0.0,
        -16.0 / 11.0,
    ],
];

/// The quadrature weights `b` of the Butcher (1964) 6(7) tableau.
const RK6_7S_B: [f64; 7] = [
    11.0 / 120.0,
    0.0,
    27.0 / 40.0,
    27.0 / 40.0,
    -4.0 / 15.0,
    -4.0 / 15.0,
    11.0 / 120.0,
];

/// `rk6_7s`: a 6th-order, 7-stage explicit Runge–Kutta solver over the k-diffusion probability-flow
/// ODE `dx/dσ = (x − x0(x,σ))/σ` (Butcher's classical 1964 tableau — see [`RK6_7S_C`] for
/// provenance and the RES4LYF naming ambiguity). Deterministic; **~7 model evaluations per nominal
/// step** (exactly 7 on every interior step; the terminal step to `σ = 0` is a single evaluation that
/// lands on the denoised estimate, because the k-diffusion derivative `(x − x0)/σ` is undefined at
/// `σ = 0`, so no stage may be evaluated there — the same terminal fallback [`Heun`] uses). Total
/// over a trailing-zero schedule of `n` steps: `7·(n−1) + 1` (see
/// [`Solver::model_evals`]).
///
/// Stateless across steps (no multistep history): a chained-pass executor (sc-20418) needs no reset
/// between passes for this solver. Each stage routes through the `denoise` callback, so per-eval
/// cancellation and progress hooks fire once per model evaluation, exactly like [`Heun`]'s second
/// stage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rk67s;

impl<L: LatentOps> Sampler<L> for Rk67s {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            if is_terminal(sigma) || s_next == 0.0 {
                // Terminal (or degenerate leading σ==0): a single evaluation landing on the denoised
                // estimate — the exact algebra of an Euler step to σ=0 (`x + (0−σ)·(x−x0)/σ = x0`),
                // avoiding any stage at σ=0 where `to_d` is undefined.
                x = denoise(&x, sigma)?;
                continue;
            }
            let h = s_next as f64 - sigma as f64; // < 0 (σ descends); exact f64 difference
            let mut k: Vec<L::Latent> = Vec::with_capacity(7);
            for j in 0..7 {
                // Stage latent X_j = x + h·Σ_{l<j} a_jl·k_l (X_1 = x: row 0 of `a` is empty).
                let mut xj = x.clone();
                for (l, kl) in k.iter().enumerate() {
                    let w = h * RK6_7S_A[j][l];
                    if w != 0.0 {
                        xj = ops.axpy(1.0, &xj, w as f32, kl)?;
                    }
                }
                // Stage abscissa σ + c_j·h ∈ [σ_{i+1}, σ_i] — strictly positive on interior steps
                // (c_7 = 1 evaluates exactly at σ_{i+1}).
                let sj = (sigma as f64 + RK6_7S_C[j] * h) as f32;
                let x0j = denoise(&xj, sj)?;
                k.push(to_d(ops, &xj, sj, &x0j)?);
            }
            // x ← x + h·Σ_j b_j·k_j (b_2 = 0 is skipped).
            for (kj, &bj) in k.iter().zip(RK6_7S_B.iter()) {
                let w = h * bj;
                if w != 0.0 {
                    x = ops.axpy(1.0, &x, w as f32, kj)?;
                }
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// abnorsett_4m — 4th-order exponential Adams–Bashforth (Nørsett-type) multistep (sc-20417).
// =================================================================================================

/// The full order of [`Abnorsett4m`]: up to 4 history points (the current denoised estimate plus the
/// three most recent previous steps') feed each step once the startup ramp completes.
pub const ABNORSETT_ORDER: usize = 4;

/// `I_m(h) = ∫_0^h e^{τ−h}·τ^m dτ` for `m = 0..k−1` — the exponential-kernel moment integrals the
/// [`Abnorsett4m`] quadrature weights are built from. Computed by the integration-by-parts recursion
/// `I_0 = 1 − e^{−h}`, `I_m = h^m − m·I_{m−1}` for `h ≥ 1`; for `h < 1` (where the recursion's
/// subtraction cancels) by the equivalent series `I_m = h^{m+1}·m!·Σ_{i≥0} (−h)^i/(m+1+i)!`. These are the
/// `φ`-function combinations of the exponential-integrator literature (`I_0 = h·φ_1(−h)`, and
/// generally `I_m = h^{m+1}·m!·φ_{m+1}(−h)`).
fn exp_kernel_integrals(h: f64, k: usize) -> Vec<f64> {
    debug_assert!(h > 0.0 && h.is_finite(), "exp_kernel_integrals: h = {h}");
    let mut out = Vec::with_capacity(k);
    if h < 1.0 {
        // Series branch: for h < 1 the recursion computes I_m ≈ h^{m+1}/(m+1) as a difference of
        // O(h^m) terms, losing ~m·log10(1/h) digits; the alternating series is cancellation-free
        // (terms shrink by ≥ h/(m+2) each; 20 terms bound the truncation below f64 ulp at h → 1).
        let mut m_fact = 1.0; // m!
        for m in 0..k {
            if m > 0 {
                m_fact *= m as f64;
            }
            let mut sum = 0.0;
            let mut num = 1.0; // (−h)^i
            let mut denom = (1..=m as u64 + 1).product::<u64>() as f64; // (m+1+i)! running
            for i in 0..20u32 {
                sum += num / denom;
                num *= -h;
                denom *= (m as f64) + 2.0 + i as f64;
            }
            out.push(h.powi(m as i32 + 1) * m_fact * sum);
        }
    } else {
        let mut prev = -(-h).exp_m1(); // I_0 = 1 − e^{−h}, computed cancellation-free
        out.push(prev);
        for m in 1..k {
            let cur = h.powi(m as i32) - m as f64 * prev;
            out.push(cur);
            prev = cur;
        }
    }
    out
}

/// The variable-step **exponential Adams–Bashforth (Nørsett-type) weights** `w_j` such that
///
/// ```text
/// x_{n+1} = e^{−h}·x_n + Σ_{j=0}^{k−1} w_j·x0_{n−j}
/// ```
///
/// exactly integrates `dx/dλ = −x + g(λ)` (the probability-flow ODE in the half-log-SNR time
/// `λ = −ln σ`, where `g` is the denoised estimate `x̂0` along the trajectory) whenever `g` is a
/// polynomial of degree `< k` — i.e. `g` is replaced by the Lagrange polynomial through the `k`
/// history nodes and the resulting integral `∫_0^h e^{τ−h}·ℓ_j(τ) dτ` is evaluated in closed form
/// via [`exp_kernel_integrals`].
///
/// `deltas[j]` is the history node offset `λ_{n−j} − λ_n` (so `deltas[0] == 0.0` and the rest are
/// strictly negative and distinct); `h = λ_{n+1} − λ_n > 0` is the current step. On a **uniform** λ
/// grid these reduce to the classical coefficients of Nørsett's A-stable modification of the
/// Adams–Bashforth methods (S. P. Nørsett, *An A-stable modification of the Adams-Bashforth
/// methods*, Lecture Notes in Mathematics 109, Springer, 1969; see also Hochbruck & Ostermann,
/// *Exponential integrators*, Acta Numerica 19 (2010), §2 — the exponential Adams methods), whose
/// order-2 form `w_0 = I_0 + I_1/h`, `w_1 = −I_1/h` is pinned in tests. The weights sum to
/// `I_0 = 1 − e^{−h}` (interpolation reproduces constants), so the `k = 1` step is exactly the
/// [`Ddim`] / exponential-Euler update.
fn exp_ab_weights(deltas: &[f64], h: f64) -> Vec<f64> {
    let k = deltas.len();
    let ints = exp_kernel_integrals(h, k);
    (0..k)
        .map(|j| {
            // Expand ℓ_j(τ) = Π_{l≠j} (τ − δ_l)/(δ_j − δ_l) into monomial coefficients of τ.
            let mut coef = vec![1.0_f64];
            let mut denom = 1.0_f64;
            for (l, &dl) in deltas.iter().enumerate() {
                if l == j {
                    continue;
                }
                denom *= deltas[j] - dl;
                let mut next = vec![0.0_f64; coef.len() + 1];
                for (m, &cm) in coef.iter().enumerate() {
                    next[m + 1] += cm;
                    next[m] -= cm * dl;
                }
                coef = next;
            }
            coef.iter()
                .zip(&ints)
                .map(|(&cm, &im)| cm * im)
                .sum::<f64>()
                / denom
        })
        .collect()
}

/// The pass-local multistep state of [`Abnorsett4m`]: the `(λ, x0)` history the solver builds while
/// integrating one pass, capped at [`ABNORSETT_ORDER`] entries (most recent last).
///
/// **Reset contract (sc-20417 → sc-20418):** the history is only meaningful along one continuous
/// trajectory. A chained-pass executor that re-noises latents between passes MUST call
/// [`reset`](Self::reset) (or supply a fresh state) before the next pass —
/// [`Abnorsett4m::sample_with_state`] makes the state explicit for exactly that seam, and the plain
/// [`Sampler::sample`] entry point constructs a fresh state per call, so one `sample` call is always
/// one self-contained pass. As defense in depth the solver also clears the history itself whenever
/// λ stops strictly increasing (a restarted or degenerate schedule), but an executor must not rely
/// on that in place of the explicit reset.
#[derive(Clone, Debug)]
pub struct AbnorsettState<T> {
    /// `(λ = −ln σ, x0)` per completed evaluation, oldest first, at most [`ABNORSETT_ORDER`] entries.
    history: Vec<(f64, T)>,
}

impl<T> AbnorsettState<T> {
    /// A fresh, empty state (the solver ramps up from order 1).
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(ABNORSETT_ORDER),
        }
    }

    /// Drop all history: the next step runs at order 1 again, exactly as a fresh state would.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// How many history entries are currently held (`0..=`[`ABNORSETT_ORDER`]). The order of the
    /// next interior step is `min(depth + 1, ABNORSETT_ORDER)` — `depth` counts entries from
    /// *previous* steps; the current step's own evaluation joins before the weights are computed.
    pub fn depth(&self) -> usize {
        self.history.len()
    }
}

impl<T> Default for AbnorsettState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// `abnorsett_4m`: a 4th-order **exponential Adams–Bashforth (Nørsett-family) multistep** solver.
///
/// Integrates the probability-flow ODE in half-log-SNR time `λ = −ln σ`, where it is the semilinear
/// `dx/dλ = −x + x̂0(λ)`: the linear part is solved exactly (`e^{−h}`, which is the DPM-Solver++
/// `σ_next/σ` contraction) and the denoised estimate `x̂0` is extrapolated by the Lagrange
/// polynomial through up to [`ABNORSETT_ORDER`] history points, integrated exactly against the
/// exponential kernel ([`exp_ab_weights`] — Nørsett's exponential Adams–Bashforth coefficients,
/// generalised to this schedule's non-uniform λ grid). Derived from the published literature (see
/// [`exp_ab_weights`] for the citations); the RES4LYF id `abnorsett_4m` names the family + order,
/// and this implementation is a from-the-derivation build, not a port of RES4LYF code.
///
/// **Startup ramp (short-schedule behavior):** a multistep method has no history at step 0. The
/// solver runs step `n` at order `min(n+1, 4)` — order 1 (the exact-linear-part exponential-Euler
/// step, identical to [`Ddim`]) on the first step, then 2, 3, and 4 from the fourth step on. A
/// schedule shorter than 4 steps simply never reaches full order; the 4-step reference recipe runs
/// orders 1, 2, 3 and then the terminal step. **The terminal step to `σ = 0`** is `h = ∞` in λ, where
/// polynomial extrapolation diverges, so it is taken at order 1 — which in the `h → ∞` limit lands
/// exactly on the final denoised estimate (the same terminal handling as [`Dpmpp2m`] / [`UniPc`]).
///
/// **Evaluation accounting:** exactly **1 model evaluation per nominal step**, startup steps
/// included (the ramp lowers order, it never adds evaluations) — `n` evaluations over an `n`-step
/// schedule ([`Solver::model_evals`]). Deterministic (ignores the seed).
#[derive(Clone, Copy, Debug, Default)]
pub struct Abnorsett4m;

impl Abnorsett4m {
    /// [`Sampler::sample`] with the multistep state made explicit — the seam the chained-pass
    /// executor (sc-20418) drives: keep one [`AbnorsettState`] per pass and
    /// [`reset`](AbnorsettState::reset) it between passes (or hand in a fresh one). Passing a fresh
    /// state is byte-identical to the plain [`Sampler::sample`] entry point.
    pub fn sample_with_state<L: LatentOps>(
        &self,
        ops: &L,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        state: &mut AbnorsettState<L::Latent>,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                // Degenerate leading σ==0 (no real schedule starts here): land on x0; λ(0) is not a
                // usable history node.
                x = x0;
                continue;
            }
            let lam = lambda(sigma);
            // History hygiene: λ must be strictly increasing along one trajectory. A non-increasing
            // λ means a restarted or degenerate schedule — drop the stale history (defense in depth;
            // the explicit contract is `AbnorsettState::reset` between passes).
            if state.history.last().is_some_and(|(l, _)| *l >= lam) {
                state.history.clear();
            }
            state.history.push((lam, x0.clone()));
            if state.history.len() > ABNORSETT_ORDER {
                state.history.remove(0);
            }
            if s_next == 0.0 {
                // Terminal: the order-1 step in the h → ∞ limit is exactly the denoised estimate.
                x = x0;
                continue;
            }
            let h = lambda(s_next) - lam; // > 0 (σ descends)
            let hist = &state.history; // oldest first; last = current step's x0
                                       // deltas[j] = λ_{n−j} − λ_n: 0 for the current node, strictly negative for older ones.
            let deltas: Vec<f64> = hist.iter().rev().map(|(l, _)| l - lam).collect();
            let w = exp_ab_weights(&deltas, h);
            // x ← e^{−h}·x + Σ_j w_j·x0_{n−j}.
            let mut next = ops.scale(&x, (-h).exp() as f32)?;
            for (j, wj) in w.iter().enumerate() {
                let (_, x0j) = &hist[hist.len() - 1 - j];
                next = ops.axpy(1.0, &next, *wj as f32, x0j)?;
            }
            x = next;
        }
        Ok(x)
    }
}

impl<L: LatentOps> Sampler<L> for Abnorsett4m {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        // One `sample` call is one pass: the multistep state is constructed fresh here, never shared
        // across calls (the pass-local contract sc-20418's executor relies on).
        let mut state = AbnorsettState::new();
        self.sample_with_state(ops, denoise, x, sigmas, &mut state)
    }
}

// =================================================================================================
// Registry — name <-> solver, the per-request selection seam the worker/engine drives (sc-7127).
// =================================================================================================

/// The curated sampler vocabulary (epic 7114 decision 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    Euler,
    EulerAncestral,
    Heun,
    Dpmpp2m,
    DpmppSde,
    UniPc,
    Lcm,
    Ddim,
    ErSde,
    Dpmpp2mSde,
    Rk67s,
    Abnorsett4m,
}

impl Solver {
    /// Parse the canonical lowercase name (the UI / recipe vocabulary). Unknown -> `None` (callers
    /// fall back to the model default + emit an event, N3).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "euler" => Self::Euler,
            "euler_ancestral" => Self::EulerAncestral,
            "heun" => Self::Heun,
            "dpmpp_2m" => Self::Dpmpp2m,
            "dpmpp_sde" => Self::DpmppSde,
            "uni_pc" => Self::UniPc,
            "lcm" => Self::Lcm,
            "ddim" => Self::Ddim,
            "er_sde" => Self::ErSde,
            "dpmpp_2m_sde" => Self::Dpmpp2mSde,
            "rk6_7s" => Self::Rk67s,
            "abnorsett_4m" => Self::Abnorsett4m,
            _ => return None,
        })
    }

    /// The canonical lowercase name (round-trips with [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::EulerAncestral => "euler_ancestral",
            Self::Heun => "heun",
            Self::Dpmpp2m => "dpmpp_2m",
            Self::DpmppSde => "dpmpp_sde",
            Self::UniPc => "uni_pc",
            Self::Lcm => "lcm",
            Self::Ddim => "ddim",
            Self::ErSde => "er_sde",
            Self::Dpmpp2mSde => "dpmpp_2m_sde",
            Self::Rk67s => "rk6_7s",
            Self::Abnorsett4m => "abnorsett_4m",
        }
    }

    /// Whether the solver draws fresh per-step noise (needs a request seed for reproducibility).
    pub fn is_stochastic(self) -> bool {
        matches!(
            self,
            Self::EulerAncestral | Self::DpmppSde | Self::Lcm | Self::ErSde | Self::Dpmpp2mSde
        )
    }

    /// Every curated solver, in menu order.
    pub const ALL: [Solver; 12] = [
        Self::Euler,
        Self::EulerAncestral,
        Self::Heun,
        Self::Dpmpp2m,
        Self::DpmppSde,
        Self::UniPc,
        Self::Lcm,
        Self::Ddim,
        Self::ErSde,
        Self::Dpmpp2mSde,
        Self::Rk67s,
        Self::Abnorsett4m,
    ];

    /// Exact model-evaluation count over a standard trailing-zero schedule of `num_steps` nominal
    /// steps (`sigmas = [σ_0 > … > σ_{n−1} > 0.0]`, length `num_steps + 1`) — the nominal-step vs
    /// model-evaluation accounting surface (sc-20417). Progress/cancellation hooks routed through
    /// the `denoise` callback fire exactly this many times (pinned per solver by the
    /// `model_evals_matches_actual_denoise_calls_for_every_solver` test):
    ///
    /// - 1 evaluation per step (`n` total): `euler`, `euler_ancestral`, `dpmpp_2m`, `uni_pc`, `lcm`,
    ///   `ddim`, `er_sde`, `dpmpp_2m_sde`, and `abnorsett_4m` (its multistep startup ramp lowers
    ///   *order*, never adds evaluations).
    /// - 2 evaluations per interior step, 1 on the terminal step (`2n − 1` total): `heun`,
    ///   `dpmpp_sde`.
    /// - 7 evaluations per interior step, 1 on the terminal step (`7n − 6` total): `rk6_7s`.
    ///
    /// A truncated schedule (no trailing `0.0`, e.g. a PiD early-stop capture) has no terminal step:
    /// every step is an interior step, so `heun`/`dpmpp_sde` cost `2n` and `rk6_7s` costs `7n` there.
    pub fn model_evals(self, num_steps: usize) -> usize {
        let n = num_steps;
        if n == 0 {
            return 0;
        }
        match self {
            Self::Euler
            | Self::EulerAncestral
            | Self::Dpmpp2m
            | Self::UniPc
            | Self::Lcm
            | Self::Ddim
            | Self::ErSde
            | Self::Dpmpp2mSde
            | Self::Abnorsett4m => n,
            Self::Heun | Self::DpmppSde => 2 * n - 1,
            Self::Rk67s => 7 * n - 6,
        }
    }

    /// Box the matching [`Sampler`] for a backend `L`.
    pub fn boxed<L: LatentOps + 'static>(self) -> Box<dyn Sampler<L>> {
        match self {
            Self::Euler => Box::new(Euler),
            Self::EulerAncestral => Box::new(EulerAncestral::default()),
            Self::Heun => Box::new(Heun),
            Self::Dpmpp2m => Box::new(Dpmpp2m),
            Self::DpmppSde => Box::new(DpmppSde::default()),
            Self::UniPc => Box::new(UniPc),
            Self::Lcm => Box::new(Lcm),
            Self::Ddim => Box::new(Ddim),
            Self::ErSde => Box::new(ErSde::default()),
            Self::Dpmpp2mSde => Box::new(Dpmpp2mSde::default()),
            Self::Rk67s => Box::new(Rk67s),
            Self::Abnorsett4m => Box::new(Abnorsett4m),
        }
    }
}

/// Build the [`Sampler`] for a canonical solver name, or `None` if unknown (N3 fallback).
pub fn sampler_by_name<L: LatentOps + 'static>(name: &str) -> Option<Box<dyn Sampler<L>>> {
    Solver::from_name(name).map(Solver::boxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::model_sampling::{denoise, FlowModelSampling};
    use crate::sampling::{
        build_flow_sigmas, compute_mu, image_seq_len, CpuLatentOps, TimestepConvention,
    };

    fn flow_sigmas(steps: usize) -> Vec<f32> {
        build_flow_sigmas(steps, compute_mu(image_seq_len(1024, 1024), steps))
    }

    /// Drive a solver over a FLOW model with a CONSTANT velocity `v` (independent of x and t). The
    /// rectified-flow ODE `dx/dσ = v` is linear, so every *consistent* deterministic solver must land
    /// EXACTLY on `x_init + v·(σ_final − σ_init) = x_init − v·σ_0` (σ_final = 0). This catches any
    /// coefficient sign/scale bug.
    fn run_const_velocity<S: Sampler<CpuLatentOps>>(
        solver: &S,
        steps: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(steps);
        let v = vec![0.37_f32, -0.12, 0.8, -0.5];
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let got = solver
            .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();
        // Exact: x_init − v·σ_0.
        let want: Vec<f32> = x_init
            .iter()
            .zip(&v)
            .map(|(&xi, &vi)| xi - vi * sigmas[0])
            .collect();
        (got, want)
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        let max = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max < tol, "{label}: max abs diff {max:e} (tol {tol:e})");
    }

    // =============================================================================================
    // Independent Python golden (sc-10519, adversarial-review fix). The er_sde / dpmpp_2m_sde
    // trajectories below are pinned against numbers produced by ComfyUI's OWN `sample_er_sde` /
    // `sample_dpmpp_2m_sde` (vendored verbatim, driven in VE space, s_noise=0), NOT by re-reading
    // this Rust — so a sign/coefficient error in a Stage-2/3 or midpoint term is caught. Fixture +
    // generator + the verbatim ComfyUI source live in `gen-core/tests/fixtures/`.
    // =============================================================================================

    /// f32-appropriate trajectory tolerance. The golden is float64; the Rust port carries latents in
    /// f32, yet the measured max drift over the whole 6-step schedule is ~1.2e-7 (a single f32 ULP at
    /// O(1) — pure CPU scalar math, so the agreement is essentially exact). `1e-5` keeps ~80x headroom
    /// for cross-platform f32 rounding while staying two orders TIGHTER than a Metal-grade 1e-3 fudge.
    /// A real sign/coefficient bug moves a step by O(1e-2..1e0), far above this (verified by mutation).
    const GOLDEN_TOL: f32 = 1e-5;

    /// The toy denoiser mirrored bit-for-bit from `gen-core/tests/fixtures/gen_sde_solver_golden.py`:
    /// `denoised = 0.5·x + 0.25·sin(1.3·x) + 0.2·σ·cos(0.7·x) − 0.1·σ`. A NON-constant field in both
    /// `x` and `σ`, so the higher-order terms genuinely engage. Computed in f64 (matching the float64
    /// reference) then demoted, isolating the SOLVER's f32 arithmetic as the only source of drift.
    fn toy_denoised(x: &[f32], sigma: f32) -> Vec<f32> {
        let sig = sigma as f64;
        x.iter()
            .map(|&xj| {
                let xj = xj as f64;
                (0.5 * xj + 0.25 * (1.3 * xj).sin() + 0.2 * sig * (0.7 * xj).cos() - 0.1 * sig)
                    as f32
            })
            .collect()
    }

    fn load_golden() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sde_solver_golden.json"
        );
        let text = std::fs::read_to_string(path).expect("read sde_solver_golden.json fixture");
        serde_json::from_str(&text).expect("parse sde_solver_golden.json")
    }

    fn json_vec(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn json_rows(v: &serde_json::Value) -> Vec<Vec<f32>> {
        v.as_array().unwrap().iter().map(json_vec).collect()
    }

    /// Drive `solver` over the golden schedule with a recording denoiser, returning the full
    /// trajectory `[x_0, x_1, …, x_N]` — the pre-step latent captured at each `denoise` call plus the
    /// final returned latent, exactly mirroring the Python generator's capture.
    fn run_recorded_trajectory<S: Sampler<CpuLatentOps>>(
        solver: &S,
        x_init: &[f32],
        sigmas: &[f32],
    ) -> Vec<Vec<f32>> {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let mut recorded: Vec<Vec<f32>> = Vec::new();
        let final_x = {
            let mut dn = |xx: &Vec<f32>, s: f32| -> Result<Vec<f32>> {
                recorded.push(xx.clone());
                Ok(toy_denoised(xx, s))
            };
            solver
                .sample(&ops, &ms, &mut dn, x_init.to_vec(), sigmas, 0)
                .unwrap()
        };
        recorded.push(final_x);
        recorded
    }

    #[test]
    fn deterministic_solvers_integrate_constant_velocity_exactly() {
        // Euler (from unified) + the deterministic curated solvers must all hit the exact solution.
        let (g, w) = run_const_velocity(&Euler, 12);
        assert_close(&g, &w, 1e-4, "euler");
        let (g, w) = run_const_velocity(&Heun, 12);
        assert_close(&g, &w, 1e-4, "heun");
        let (g, w) = run_const_velocity(&Dpmpp2m, 12);
        assert_close(&g, &w, 1e-4, "dpmpp_2m");
        let (g, w) = run_const_velocity(&UniPc, 12);
        assert_close(&g, &w, 1e-4, "uni_pc");
        let (g, w) = run_const_velocity(&Ddim, 12);
        assert_close(&g, &w, 1e-4, "ddim");
        // sc-20417: the high-order pair must also integrate the linear constant-velocity ODE
        // exactly (rk6_7s because every stage stays on the exact line — the row-sum condition;
        // abnorsett_4m because a constant velocity makes the denoised estimate x̂0 CONSTANT along
        // the trajectory, and the exponential-AB weights reproduce constants exactly: Σw_j = I_0).
        let (g, w) = run_const_velocity(&Rk67s, 12);
        assert_close(&g, &w, 1e-4, "rk6_7s");
        let (g, w) = run_const_velocity(&Abnorsett4m, 12);
        assert_close(&g, &w, 1e-4, "abnorsett_4m");
    }

    #[test]
    fn stochastic_solvers_are_finite_and_seed_reproducible() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(10);
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05, 0.9, -0.3];
        let run = |solver: &dyn Sampler<CpuLatentOps>, seed: u64| {
            let mut dn = |xx: &Vec<f32>, s: f32| {
                denoise(&ops, &ms, xx, s, |xin, _t| {
                    Ok(xin.iter().map(|&v| 0.2 * v + 0.05).collect())
                })
            };
            solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, seed)
                .unwrap()
        };
        for solver in [
            &EulerAncestral::default() as &dyn Sampler<CpuLatentOps>,
            &DpmppSde::default(),
            &Lcm,
            &ErSde::default(),
            &Dpmpp2mSde::default(),
        ] {
            let a = run(solver, 7);
            let b = run(solver, 7);
            let c = run(solver, 8);
            assert!(a.iter().all(|v| v.is_finite()), "non-finite output");
            assert_eq!(a, b, "same seed must reproduce");
            assert_ne!(a, c, "different seed must differ");
        }
    }

    /// Both SDE solvers must inject NO noise on the terminal step (`σ_next == 0`): a single-step
    /// schedule `[σ, 0]` lands EXACTLY on the denoised estimate, seed-independently (the invariant the
    /// story pins — a stochastic solver must degrade to a plain denoise at the clean node).
    #[test]
    fn sde_solvers_add_no_noise_on_terminal_step() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = [0.5_f32, 0.0];
        let x_init = vec![0.2_f32, -0.6, 1.1];
        let v = vec![0.4_f32, 0.4, -0.2];
        // Flow denoise: x0 at σ=0.5 = x − 0.5·v.
        let want: Vec<f32> = x_init
            .iter()
            .zip(&v)
            .map(|(&xi, &vi)| xi - 0.5 * vi)
            .collect();
        for solver in [
            &ErSde::default() as &dyn Sampler<CpuLatentOps>,
            &Dpmpp2mSde::default(),
        ] {
            let run = |seed: u64| {
                let mut dn =
                    |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
                solver
                    .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, seed)
                    .unwrap()
            };
            let a = run(1);
            let b = run(999);
            assert_eq!(
                a, b,
                "terminal step must not inject noise (seed-independent)"
            );
            assert_close(&a, &want, 1e-6, "terminal step lands on x0");
        }
    }

    /// er_sde reproduces ComfyUI's `sample_er_sde` step-by-step over a 6-step VE schedule (deterministic,
    /// `s_noise=0`) — reaching Stage-3 (`i>=2`) so the 200-point Riemann integrals and the Stage-2/3
    /// finite-difference corrections are all exercised against the INDEPENDENT float64 golden, not a
    /// re-transcription of this Rust. Catches a sign/coefficient error anywhere in the higher-order path.
    #[test]
    fn er_sde_matches_python_golden_trajectory() {
        let g = load_golden();
        let sigmas = json_vec(&g["sigmas"]);
        let x_init = json_vec(&g["x_init"]);
        let want = json_rows(&g["er_sde"]["trajectory"]);
        let solver = ErSde {
            s_noise: 0.0,
            max_stage: 3,
        };
        let got = run_recorded_trajectory(&solver, &x_init, &sigmas);
        assert_eq!(got.len(), want.len(), "er_sde golden trajectory length");
        assert_eq!(
            got.len(),
            7,
            "6 steps + trailing terminal => 7 latent snapshots"
        );
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_close(a, b, GOLDEN_TOL, &format!("er_sde golden step {k}"));
        }
    }

    /// dpmpp_2m_sde reproduces ComfyUI's `sample_dpmpp_2m_sde` (`solver_type='midpoint'`, `η=1`)
    /// step-by-step over the same schedule (deterministic, `s_noise=0`) — exercising the midpoint
    /// 2nd-order correction against the INDEPENDENT float64 golden.
    #[test]
    fn dpmpp_2m_sde_matches_python_golden_trajectory() {
        let g = load_golden();
        let sigmas = json_vec(&g["sigmas"]);
        let x_init = json_vec(&g["x_init"]);
        let want = json_rows(&g["dpmpp_2m_sde"]["trajectory"]);
        let solver = Dpmpp2mSde {
            eta: 1.0,
            s_noise: 0.0,
        };
        let got = run_recorded_trajectory(&solver, &x_init, &sigmas);
        assert_eq!(
            got.len(),
            want.len(),
            "dpmpp_2m_sde golden trajectory length"
        );
        assert_eq!(
            got.len(),
            7,
            "6 steps + trailing terminal => 7 latent snapshots"
        );
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_close(a, b, GOLDEN_TOL, &format!("dpmpp_2m_sde golden step {k}"));
        }
    }

    /// er_sde's first step (`i==0`, Stage-1) pinned against the INDEPENDENT golden value (`s_noise=0`),
    /// then the noise term proven wired by its EXACT linearity in `s_noise` — the noise vector is drawn
    /// on the (s_noise-independent) deterministic `x_next`, so `out(s_noise) − det = s_noise·(coeff·ε)`,
    /// i.e. doubling `s_noise` must exactly double the deviation. No hand-transcription of the coeff.
    #[test]
    fn er_sde_first_step_matches_python_golden_and_scales_noise() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let g = load_golden();
        let sigmas = json_vec(&g["sigmas"]);
        let x_init = json_vec(&g["x_init"]);
        let want_x1 = json_rows(&g["er_sde"]["trajectory"])[1].clone();
        let two = [sigmas[0], sigmas[1]];

        let run = |s_noise: f32| {
            let mut dn = |xx: &Vec<f32>, s: f32| Ok(toy_denoised(xx, s));
            ErSde {
                s_noise,
                max_stage: 3,
            }
            .sample(&ops, &ms, &mut dn, x_init.clone(), &two, 3)
            .unwrap()
        };
        // Deterministic first step == the independent golden (non-tautological value check at i=0).
        let det = run(0.0);
        assert_close(&det, &want_x1, GOLDEN_TOL, "er_sde first step golden");
        // Noise term is present and scales linearly with s_noise.
        let o1 = run(1.0);
        let o2 = run(2.0);
        assert_ne!(o1, det, "er_sde must inject noise when s_noise>0");
        let d1: Vec<f32> = o1.iter().zip(&det).map(|(&a, &b)| a - b).collect();
        let d2: Vec<f32> = o2.iter().zip(&det).map(|(&a, &b)| a - b).collect();
        let twice: Vec<f32> = d1.iter().map(|&v| 2.0 * v).collect();
        assert_close(&d2, &twice, 1e-6, "er_sde noise term linear in s_noise");
    }

    /// dpmpp_2m_sde's first step (`i==0`, no history) pinned against the INDEPENDENT golden value
    /// (`η=1`, `s_noise=0`), then the SDE noise term proven wired by its exact linearity in `s_noise`.
    #[test]
    fn dpmpp_2m_sde_first_step_matches_python_golden_and_scales_noise() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let g = load_golden();
        let sigmas = json_vec(&g["sigmas"]);
        let x_init = json_vec(&g["x_init"]);
        let want_x1 = json_rows(&g["dpmpp_2m_sde"]["trajectory"])[1].clone();
        let two = [sigmas[0], sigmas[1]];

        let run = |s_noise: f32| {
            let mut dn = |xx: &Vec<f32>, s: f32| Ok(toy_denoised(xx, s));
            Dpmpp2mSde { eta: 1.0, s_noise }
                .sample(&ops, &ms, &mut dn, x_init.clone(), &two, 5)
                .unwrap()
        };
        let det = run(0.0);
        assert_close(&det, &want_x1, GOLDEN_TOL, "dpmpp_2m_sde first step golden");
        let o1 = run(1.0);
        let o2 = run(2.0);
        assert_ne!(o1, det, "dpmpp_2m_sde must inject noise when s_noise>0");
        let d1: Vec<f32> = o1.iter().zip(&det).map(|(&a, &b)| a - b).collect();
        let d2: Vec<f32> = o2.iter().zip(&det).map(|(&a, &b)| a - b).collect();
        let twice: Vec<f32> = d1.iter().map(|&v| 2.0 * v).collect();
        assert_close(
            &d2,
            &twice,
            1e-6,
            "dpmpp_2m_sde noise term linear in s_noise",
        );
    }

    #[test]
    fn ddim_equals_euler_on_flow() {
        // In VE-sigma space DDIM(η=0) coincides with Euler — verify on a non-trivial velocity field.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(8);
        let x_init = vec![0.4_f32, -0.7, 1.3];
        let mut model = |xx: &Vec<f32>, _t: f32| -> Result<Vec<f32>> {
            Ok(xx.iter().map(|&v| 0.3 * v + 0.1).collect())
        };
        let mut dn_e = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, &mut model);
        let euler = Euler
            .sample(&ops, &ms, &mut dn_e, x_init.clone(), &sigmas, 0)
            .unwrap();
        let mut dn_d = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, &mut model);
        let ddim = Ddim
            .sample(&ops, &ms, &mut dn_d, x_init.clone(), &sigmas, 0)
            .unwrap();
        assert_close(&euler, &ddim, 1e-5, "ddim_vs_euler");
    }

    #[test]
    fn second_order_solvers_track_euler_on_smooth_field() {
        // dpmpp_2m / uni_pc / heun should produce sane (finite, close-to-Euler) trajectories on a
        // smooth field — not identical (they're higher order) but in the same neighbourhood.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(16);
        let x_init = vec![0.4_f32, -0.7, 1.3, 0.2, -0.9];
        let model = |xx: &Vec<f32>, t: f32| -> Vec<f32> {
            xx.iter().map(|&v| 0.25 * v + 0.1 * (t).sin()).collect()
        };
        let mut dn_e =
            |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, t| Ok(model(xin, t)));
        let euler = Euler
            .sample(&ops, &ms, &mut dn_e, x_init.clone(), &sigmas, 0)
            .unwrap();
        for solver in [&Heun as &dyn Sampler<CpuLatentOps>, &Dpmpp2m, &UniPc] {
            let mut dn =
                |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, t| Ok(model(xin, t)));
            let out = solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
                .unwrap();
            assert!(out.iter().all(|v| v.is_finite()));
            assert_close(&out, &euler, 0.5, "2nd-order near euler");
        }
    }

    #[test]
    fn registry_round_trips_and_boxes_all() {
        for s in Solver::ALL {
            assert_eq!(Solver::from_name(s.name()), Some(s));
            let _boxed: Box<dyn Sampler<CpuLatentOps>> = s.boxed();
        }
        assert_eq!(
            Solver::ALL.len(),
            12,
            "curated solver set is 12 (added rk6_7s + abnorsett_4m, sc-20417)"
        );
        assert!(sampler_by_name::<CpuLatentOps>("euler").is_some());
        assert!(sampler_by_name::<CpuLatentOps>("er_sde").is_some());
        assert!(sampler_by_name::<CpuLatentOps>("dpmpp_2m_sde").is_some());
        assert!(sampler_by_name::<CpuLatentOps>("rk6_7s").is_some());
        assert!(sampler_by_name::<CpuLatentOps>("abnorsett_4m").is_some());
        assert!(sampler_by_name::<CpuLatentOps>("nope").is_none());
        assert!(Solver::Lcm.is_stochastic());
        assert!(!Solver::Heun.is_stochastic());
        // Both new SDE solvers draw fresh per-step noise -> stochastic.
        assert!(Solver::ErSde.is_stochastic());
        assert!(Solver::Dpmpp2mSde.is_stochastic());
        // The sc-20417 high-order pair is deterministic (never draws noise).
        assert!(!Solver::Rk67s.is_stochastic());
        assert!(!Solver::Abnorsett4m.is_stochastic());
        // dpmpp_2m_sde must NOT be confused with the distinct dpmpp_sde solver.
        assert_ne!(Solver::Dpmpp2mSde, Solver::DpmppSde);
    }

    // =============================================================================================
    // sc-20417: rk6_7s + abnorsett_4m — coefficient provenance, closed-form, golden, state tests.
    // =============================================================================================

    /// A rooted tree, for the Butcher order-condition check (subtree forest; a leaf has none).
    #[derive(Clone)]
    struct Tree(Vec<Tree>);

    /// All ordered compositions of `total` into positive parts (order-condition duplicates from
    /// reordered forests are harmless — the same condition is just checked more than once).
    fn compositions(total: usize) -> Vec<Vec<usize>> {
        if total == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for first in 1..=total {
            for rest in compositions(total - first) {
                let mut v = vec![first];
                v.extend(rest);
                out.push(v);
            }
        }
        out
    }

    /// Every rooted tree of order `n` (as ordered trees; reorderings duplicate conditions, which is
    /// fine for a checker).
    fn trees_of_order(n: usize) -> Vec<Tree> {
        if n == 1 {
            return vec![Tree(Vec::new())];
        }
        let mut out = Vec::new();
        for comp in compositions(n - 1) {
            let mut forests: Vec<Vec<Tree>> = vec![Vec::new()];
            for &part in &comp {
                let pool = trees_of_order(part);
                let mut next = Vec::new();
                for pre in &forests {
                    for t in &pool {
                        let mut v = pre.clone();
                        v.push(t.clone());
                        next.push(v);
                    }
                }
                forests = next;
            }
            out.extend(forests.into_iter().map(Tree));
        }
        out
    }

    fn tree_order(t: &Tree) -> usize {
        1 + t.0.iter().map(tree_order).sum::<usize>()
    }

    /// Butcher's density `γ(t) = |t| · Π_k γ(t_k)`.
    fn tree_gamma(t: &Tree) -> f64 {
        tree_order(t) as f64 * t.0.iter().map(tree_gamma).product::<f64>()
    }

    /// The per-stage elementary weights `g_i(t) = Π_k (Σ_j a_ij · g_j(t_k))`, `g_i(leaf) = 1`.
    fn elementary_weights(t: &Tree) -> [f64; 7] {
        let mut g = [1.0_f64; 7];
        for sub in &t.0 {
            let gs = elementary_weights(sub);
            for (i, gi) in g.iter_mut().enumerate() {
                let acc: f64 = (0..i).map(|j| RK6_7S_A[i][j] * gs[j]).sum();
                *gi *= acc;
            }
        }
        g
    }

    /// THE tableau-provenance gate: a Runge–Kutta method has order p iff `Σ_i b_i·g_i(t) = 1/γ(t)`
    /// for EVERY rooted tree `t` with `|t| ≤ p` (Butcher's theorem). Verifies all order conditions
    /// through order 6 against the encoded [`RK6_7S_A`]/[`RK6_7S_B`]/[`RK6_7S_C`] constants — a
    /// single mistyped coefficient fails some condition by O(1e-2..1), far above the f64 tolerance.
    /// Also proves the method is NOT order 7 (the checker has teeth) and that the row sums equal `c`
    /// (the stage-consistency condition the constant-x0 exactness test relies on).
    #[test]
    fn rk6_7s_tableau_satisfies_all_order_conditions_through_order_6() {
        let mut checked = 0usize;
        for order in 1..=6 {
            for t in trees_of_order(order) {
                let g = elementary_weights(&t);
                let phi: f64 = RK6_7S_B.iter().zip(&g).map(|(&b, &gi)| b * gi).sum();
                let want = 1.0 / tree_gamma(&t);
                assert!(
                    (phi - want).abs() < 1e-13,
                    "order-{order} condition violated: Phi = {phi}, want {want}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 37,
            "expected >= 37 conditions, checked {checked}"
        );
        // Teeth: at least one order-7 condition must FAIL (the tableau is order 6 exactly).
        let mut any_order7_fails = false;
        for t in trees_of_order(7) {
            let g = elementary_weights(&t);
            let phi: f64 = RK6_7S_B.iter().zip(&g).map(|(&b, &gi)| b * gi).sum();
            if (phi - 1.0 / tree_gamma(&t)).abs() > 1e-6 {
                any_order7_fails = true;
                break;
            }
        }
        assert!(
            any_order7_fails,
            "checker has no teeth: order 7 all passed?"
        );
        // Row sums equal c (stage consistency).
        for (row, &ci) in RK6_7S_A.iter().zip(&RK6_7S_C) {
            let sum: f64 = row.iter().sum();
            assert!((sum - ci).abs() < 1e-15, "row sum {sum} != c {ci}");
        }
        // b sums to 1 (the order-1 condition, doubly pinned).
        assert!((RK6_7S_B.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    /// Simpson quadrature of `∫_0^h e^{τ−h}·τ^m dτ` — an INDEPENDENT evaluation of the exponential
    /// moments (no shared code with [`exp_kernel_integrals`]' recursion/series).
    fn simpson_exp_moment(h: f64, m: usize) -> f64 {
        let n = 20_000usize; // even
        let dt = h / n as f64;
        let f = |t: f64| (t - h).exp() * if m == 0 { 1.0 } else { t.powi(m as i32) };
        let mut s = f(0.0) + f(h);
        for i in 1..n {
            let w = if i % 2 == 1 { 4.0 } else { 2.0 };
            s += w * f(i as f64 * dt);
        }
        s * dt / 3.0
    }

    /// The defining property of the Nørsett / exponential-AB weights: for every polynomial `q` of
    /// degree `< k`, `Σ_j w_j·q(δ_j) = ∫_0^h e^{τ−h}·q(τ) dτ`. Checked on monomials over
    /// non-uniform grids and a spread of step sizes (including the small-h Taylor branch) against
    /// independent Simpson quadrature — this uniquely determines the weights, so it validates both
    /// the Lagrange expansion and the moment recursion with no self-reference.
    #[test]
    fn abnorsett_weights_integrate_polynomials_exactly() {
        let grids: [&[f64]; 4] = [
            &[0.0],
            &[0.0, -0.4],
            &[0.0, -0.35, -0.8],
            &[0.0, -0.3, -0.75, -1.3],
        ];
        for deltas in grids {
            for &h in &[5e-4_f64, 0.05, 0.5, 2.0] {
                let w = exp_ab_weights(deltas, h);
                assert_eq!(w.len(), deltas.len());
                for m in 0..deltas.len() {
                    let got: f64 = w
                        .iter()
                        .zip(deltas)
                        .map(|(&wj, &d)| wj * if m == 0 { 1.0 } else { d.powi(m as i32) })
                        .sum();
                    let want = simpson_exp_moment(h, m);
                    assert!(
                        (got - want).abs() < 1e-10 * want.abs().max(1e-3),
                        "k={} h={h} m={m}: got {got} want {want}",
                        deltas.len()
                    );
                }
            }
        }
    }

    /// On a uniform λ grid the k=2 weights reduce to the published Nørsett order-2 closed form
    /// `w_0 = I_0 + I_1/h`, `w_1 = −I_1/h` (the `γ_1` coefficient of the exponential
    /// Adams–Bashforth backward-difference form, Hochbruck & Ostermann §2), and the k=1 step is the
    /// exponential-Euler / DDIM update `w_0 = I_0 = 1 − e^{−h} = 1 − σ_next/σ`.
    #[test]
    fn abnorsett_uniform_grid_reduces_to_norsett_closed_forms() {
        let h = 0.37_f64;
        let i0 = -(-h).exp_m1();
        let i1 = h - i0;
        let w1 = exp_ab_weights(&[0.0], h);
        assert!((w1[0] - i0).abs() < 1e-14, "k=1: {} vs {}", w1[0], i0);
        let w2 = exp_ab_weights(&[0.0, -h], h);
        assert!((w2[0] - (i0 + i1 / h)).abs() < 1e-14, "k=2 w0");
        assert!((w2[1] - (-i1 / h)).abs() < 1e-14, "k=2 w1");
        // Any-order weights reproduce constants: Σw_j = I_0.
        let w4 = exp_ab_weights(&[0.0, -0.3, -0.7, -1.2], h);
        assert!((w4.iter().sum::<f64>() - i0).abs() < 1e-12);
    }

    /// Constant-x0 model: the flow ODE `dx/dσ = (x − C)/σ` has the exact straight-line solution
    /// `x(σ) = C + (x_init − C)·σ/σ_0`. rk6_7s is exact here because the row-sum condition keeps
    /// every stage on that line; abnorsett_4m is exact because `x̂0` is constant along the
    /// trajectory (weights reproduce constants). Truncated schedule (no terminal node) so the
    /// interior math itself is what lands on the closed form.
    #[test]
    fn rk6_and_abnorsett_land_on_constant_x0_closed_form() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        // 6 uniform-ratio steps from 1.0 down to 0.2 (all positive; no terminal step).
        let n = 6usize;
        let ratio = (0.2_f64 / 1.0).powf(1.0 / n as f64);
        let sigmas: Vec<f32> = (0..=n).map(|i| ratio.powi(i as i32) as f32).collect();
        let c0 = [0.4_f32, -0.9, 1.6, 0.02];
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05];
        let want: Vec<f32> = x_init
            .iter()
            .zip(&c0)
            .map(|(&xi, &ci)| ci + (xi - ci) * (sigmas[n] / sigmas[0]))
            .collect();
        for (name, solver) in [
            ("rk6_7s", &Rk67s as &dyn Sampler<CpuLatentOps>),
            ("abnorsett_4m", &Abnorsett4m),
        ] {
            let mut dn = |_xx: &Vec<f32>, _s: f32| Ok(c0.to_vec());
            let got = solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
                .unwrap();
            assert_close(&got, &want, 2e-5, name);
        }
    }

    /// Linear model `x̂0 = a·x`: the ODE `dx/dλ = −(1−a)·x` has the closed form
    /// `x(σ_end) = x_init·(σ_end/σ_0)^{1−a}` — non-polynomial, so it separates the orders: rk6_7s
    /// and abnorsett_4m must sit essentially on the answer while first-order Euler carries a
    /// visible O(h) error on the same schedule.
    #[test]
    fn rk6_and_abnorsett_track_linear_model_closed_form_tighter_than_euler() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let a = 0.4_f32;
        let n = 12usize;
        let ratio = (0.25_f64).powf(1.0 / n as f64);
        let sigmas: Vec<f32> = (0..=n).map(|i| ratio.powi(i as i32) as f32).collect();
        let x_init = vec![0.8_f32, -1.3, 2.1];
        let factor = (sigmas[n] as f64 / sigmas[0] as f64).powf((1.0 - a) as f64) as f32;
        let want: Vec<f32> = x_init.iter().map(|&v| v * factor).collect();
        let run = |solver: &dyn Sampler<CpuLatentOps>| {
            let mut dn =
                |xx: &Vec<f32>, _s: f32| Ok(xx.iter().map(|&v| a * v).collect::<Vec<f32>>());
            solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
                .unwrap()
        };
        let max_err = |got: &[f32]| {
            got.iter()
                .zip(&want)
                .map(|(&g, &w)| (g - w).abs() / w.abs())
                .fold(0.0_f32, f32::max)
        };
        let e_rk6 = max_err(&run(&Rk67s));
        let e_ab = max_err(&run(&Abnorsett4m));
        let e_euler = max_err(&run(&Euler));
        assert!(e_rk6 < 1e-4, "rk6_7s rel err {e_rk6:e}");
        assert!(e_ab < 2e-3, "abnorsett_4m rel err {e_ab:e}");
        assert!(
            e_euler > 5e-3,
            "euler rel err {e_euler:e} suspiciously small"
        );
        assert!(e_rk6 < e_euler / 10.0 && e_ab < e_euler / 2.0);
    }

    /// f32-appropriate tolerance for the rk6/abnorsett float64 goldens. rk6_7s takes ~5 f32 axpys
    /// per stage × 64 evaluations over the 10-step recipe; the accumulated rounding drift against
    /// the float64 reference stays well under 1e-5 at O(1) latents (measured ~1e-6), while a wrong
    /// tableau/weight coefficient displaces a trajectory by O(1e-2..1).
    const RK_GOLDEN_TOL: f32 = 2e-5;

    fn load_rk6_golden() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/rk6_abnorsett_golden.json"
        );
        let text = std::fs::read_to_string(path).expect("read rk6_abnorsett_golden.json fixture");
        serde_json::from_str(&text).expect("parse rk6_abnorsett_golden.json")
    }

    /// Drive one golden case: the recorded latent at EVERY denoise call (for rk6_7s that pins each
    /// of the 7 stage latents per step, not just the outer trajectory) plus the final latent, against
    /// the independently generated float64 fixture (`gen_rk6_abnorsett_golden.py` — a from-the-
    /// literature Python implementation, not a re-transcription of this Rust).
    fn assert_golden_case<S: Sampler<CpuLatentOps>>(solver: &S, key: &str, snapshots: usize) {
        let g = load_rk6_golden();
        let case = &g[key];
        let sigmas = json_vec(&case["sigmas"]);
        let x_init = json_vec(&case["x_init"]);
        let want = json_rows(&case["trajectory"]);
        let got = run_recorded_trajectory(solver, &x_init, &sigmas);
        assert_eq!(got.len(), want.len(), "{key}: golden trajectory length");
        assert_eq!(got.len(), snapshots, "{key}: expected snapshot count");
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_close(a, b, RK_GOLDEN_TOL, &format!("{key} snapshot {k}"));
        }
    }

    /// AC2: the exact 10-step rk6_7s reference-recipe sequence, pinned stage-by-stage against the
    /// independent float64 golden. 9 interior steps × 7 stages + 1 terminal eval = 64 recorded
    /// latents, + the final latent = 65 snapshots.
    #[test]
    fn rk6_7s_matches_independent_golden_10_step_recipe() {
        assert_golden_case(&Rk67s, "rk6_7s_10step", 65);
    }

    /// AC2: the exact 4-step abnorsett_4m reference-recipe sequence (startup ramp orders 1, 2, 3,
    /// then the terminal step) against the independent float64 golden. 4 evals + final = 5.
    #[test]
    fn abnorsett_4m_matches_independent_golden_4_step_recipe() {
        assert_golden_case(&Abnorsett4m, "abnorsett_4m_4step", 5);
    }

    /// An 8-step abnorsett_4m golden so the FULL-order (k = 4) interior steps genuinely engage —
    /// the 4-step recipe's last step is terminal, so order 4 never fires there. 8 evals + final = 9.
    #[test]
    fn abnorsett_4m_matches_independent_golden_8_step_full_order() {
        assert_golden_case(&Abnorsett4m, "abnorsett_4m_8step", 9);
    }

    /// Determinism (AC1): both solvers are noise-free — identical output run-to-run AND across
    /// seeds.
    #[test]
    fn rk6_and_abnorsett_are_deterministic_and_seed_independent() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(9);
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05];
        for solver in [&Rk67s as &dyn Sampler<CpuLatentOps>, &Abnorsett4m] {
            let run = |seed: u64| {
                let mut dn = |xx: &Vec<f32>, s: f32| Ok(toy_denoised(xx, s));
                solver
                    .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, seed)
                    .unwrap()
            };
            assert_eq!(run(7), run(7), "same run must be bitwise identical");
            assert_eq!(run(7), run(8), "deterministic solver must ignore the seed");
        }
    }

    /// First/last-step handling (AC1): a single-step `[σ, 0]` schedule lands EXACTLY on the
    /// denoised estimate (one eval — no σ=0 stage is ever evaluated), and a degenerate leading
    /// σ==0 node neither panics nor divides by zero.
    #[test]
    fn rk6_and_abnorsett_terminal_and_degenerate_leading_steps() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let x_init = vec![0.2_f32, -0.6, 1.1];
        for (name, solver) in [
            ("rk6_7s", &Rk67s as &dyn Sampler<CpuLatentOps>),
            ("abnorsett_4m", &Abnorsett4m),
        ] {
            // Terminal: [σ, 0] → exactly x0(x_init, σ), with exactly one model eval.
            let evals = std::cell::Cell::new(0usize);
            let mut dn = |xx: &Vec<f32>, s: f32| {
                evals.set(evals.get() + 1);
                Ok(toy_denoised(xx, s))
            };
            let got = solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &[0.5_f32, 0.0], 0)
                .unwrap();
            assert_eq!(
                got,
                toy_denoised(&x_init, 0.5),
                "{name}: terminal lands on x0"
            );
            assert_eq!(evals.get(), 1, "{name}: terminal step is a single eval");
            // Degenerate leading 0 (no real schedule starts here): finite, lands on x0 of x0.
            let mut dn2 = |xx: &Vec<f32>, s: f32| Ok(toy_denoised(xx, s));
            let got2 = solver
                .sample(&ops, &ms, &mut dn2, x_init.clone(), &[0.0_f32, 0.0], 0)
                .unwrap();
            assert!(
                got2.iter().all(|v| v.is_finite()),
                "{name}: degenerate leading 0"
            );
        }
    }

    /// AC4: `Solver::model_evals` matches the ACTUAL number of `denoise` calls (= per-eval
    /// progress/cancellation hook firings) for EVERY curated solver — pinning the existing solvers'
    /// counts as a regression guard and the new pair's accounting (rk6_7s ~7/step, abnorsett_4m
    /// 1/step including its startup ramp).
    #[test]
    fn model_evals_matches_actual_denoise_calls_for_every_solver() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let steps = 8usize;
        let sigmas = flow_sigmas(steps); // trailing 0.0
        for s in Solver::ALL {
            let count = std::cell::Cell::new(0usize);
            let mut dn = |xx: &Vec<f32>, sig: f32| {
                count.set(count.get() + 1);
                denoise(&ops, &ms, xx, sig, |xin, _t| {
                    Ok(xin.iter().map(|&v| 0.2 * v + 0.05).collect())
                })
            };
            s.boxed::<CpuLatentOps>()
                .sample(&ops, &ms, &mut dn, vec![0.3_f32, -0.7, 1.1], &sigmas, 3)
                .unwrap();
            assert_eq!(
                count.get(),
                s.model_evals(steps),
                "{}: model_evals({steps}) out of sync with the actual denoise-call count",
                s.name()
            );
        }
        // The story's reference recipe, spelled out: 10-step rk6_7s = 7·9 + 1 = 64 evals;
        // 4-step abnorsett_4m = 4 evals (startup ramp adds order, never evaluations).
        assert_eq!(Solver::Rk67s.model_evals(10), 64);
        assert_eq!(Solver::Abnorsett4m.model_evals(4), 4);
        assert_eq!(Solver::Heun.model_evals(8), 15);
        assert_eq!(Solver::Euler.model_evals(0), 0);
        assert_eq!(Solver::Rk67s.model_evals(1), 1); // single-step schedule: terminal only
    }

    /// AC1 history reset + the sc-20418 pass-local contract: `Sampler::sample` equals
    /// `sample_with_state` over a fresh state bitwise; a completed pass leaves history behind;
    /// `reset()` restores fresh-state behavior exactly; and stale history genuinely feeds the math
    /// (a λ-continuing second pass differs with vs without the carried state).
    #[test]
    fn abnorsett_state_is_pass_local_and_reset_reproduces() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        // Two λ-continuing truncated segments (no terminal node, so history always builds).
        let seg_a = [1.0_f32, 0.8, 0.65, 0.5];
        let seg_b = [0.4_f32, 0.3, 0.22, 0.15];
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05];
        let mut dn = |xx: &Vec<f32>, s: f32| Ok(toy_denoised(xx, s));

        // Plain `sample` == `sample_with_state` over a fresh state, bitwise.
        let via_trait = Sampler::<CpuLatentOps>::sample(
            &Abnorsett4m,
            &ops,
            &ms,
            &mut dn,
            x_init.clone(),
            &seg_a,
            0,
        )
        .unwrap();
        let mut state = AbnorsettState::new();
        assert_eq!(state.depth(), 0);
        let x_a = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_init.clone(), &seg_a, &mut state)
            .unwrap();
        assert_eq!(via_trait, x_a, "fresh explicit state == trait entry point");
        assert_eq!(state.depth(), 3, "a 3-step pass leaves 3 history entries");

        // Stale history matters: continuing into seg_b (λ still increasing) with the carried state
        // runs at higher startup order than a fresh state — outputs must differ.
        let mut dirty = state.clone();
        let y_dirty = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_a.clone(), &seg_b, &mut dirty)
            .unwrap();
        let mut fresh = AbnorsettState::new();
        let y_fresh = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_a.clone(), &seg_b, &mut fresh)
            .unwrap();
        assert_ne!(
            y_dirty, y_fresh,
            "carried multistep history must actually change the integration"
        );

        // The reset contract: reset() then re-run reproduces the first pass bitwise.
        state.reset();
        assert_eq!(state.depth(), 0);
        let x_a2 = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_init.clone(), &seg_a, &mut state)
            .unwrap();
        assert_eq!(x_a, x_a2, "reset + re-run must be bitwise identical");
    }

    /// Defense in depth behind the explicit reset: a RESTARTED schedule (λ drops back) clears the
    /// stale history automatically, so even an executor that forgot to reset never extrapolates
    /// across a re-noise boundary with history from a previous pass.
    #[test]
    fn abnorsett_self_clears_history_on_non_increasing_lambda() {
        let ops = CpuLatentOps;
        let seg = [1.0_f32, 0.8, 0.65, 0.5];
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05];
        let mut dn = |xx: &Vec<f32>, s: f32| Ok(toy_denoised(xx, s));
        let mut state = AbnorsettState::new();
        let x_a = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_init.clone(), &seg, &mut state)
            .unwrap();
        // Re-run the SAME schedule without reset: λ restarts below the carried history, the guard
        // clears it, and the pass reproduces the fresh-state result exactly.
        let x_b = Abnorsett4m
            .sample_with_state(&ops, &mut dn, x_init.clone(), &seg, &mut state)
            .unwrap();
        assert_eq!(
            x_a, x_b,
            "restart guard must behave exactly like a fresh state"
        );
    }

    /// The exponential-moment recursion and its small-h Taylor branch agree with independent
    /// Simpson quadrature across the branch point.
    #[test]
    fn exp_kernel_integrals_match_quadrature_on_both_branches() {
        for &h in &[1e-5_f64, 5e-4, 1e-3, 2e-3, 0.1, 1.0, 3.0] {
            let ints = exp_kernel_integrals(h, 4);
            for (m, &got) in ints.iter().enumerate() {
                let want = simpson_exp_moment(h, m);
                assert!(
                    (got - want).abs() < 1e-12 * want.abs().max(1e-6),
                    "h={h} m={m}: got {got:e} want {want:e}"
                );
            }
        }
    }
}
