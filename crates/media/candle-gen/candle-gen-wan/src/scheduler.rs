//! Flow-match sampling for Wan: the **`UniPCMultistepScheduler`** (the checkpoint default —
//! `solver_order=2`, `solver_type="bh2"`, `predict_x0=true`, `use_flow_sigmas`, `final_sigmas_type="zero"`,
//! `lower_order_final`), the flow-SNR **DPM-Solver++(2M)** multistep (sc-27309, ported from
//! `mlx-gen-wan`'s `FlowDpmpp2m`), and a plain flow-match **Euler** fallback. UniPC is ported from
//! diffusers `scheduling_unipc_multistep.py`.
//!
//! Sigmas: `σ_k = flow_shift·s / (1 + (flow_shift−1)·s)` over `s = linspace(1, 1/N, steps+1)[:-1]`,
//! with a tiny `σ_0 -= 1e-6` to keep `log(1−σ_0)` finite, then a terminal `0`. The DiT timestep at
//! step `i` is `σ_i · N`. Sample math runs in f32 (model output is the velocity `v`).

use candle_gen::candle_core::{Result, Tensor};

use crate::config::{FLOW_SHIFT, NUM_TRAIN_TIMESTEPS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    UniPC,
    Euler,
    /// DPM-Solver++(2M) in Wan's own flow log-SNR `λ = log((1−σ)/σ)` — NOT gen-core's VE-space
    /// `dpmpp_2m` (`λ = −ln σ`), which would diverge from the MLX twin.
    Dpmpp2m,
}

impl Sampler {
    /// Map a curated sampler name (epic 7114) to Wan's native integrator. The curated `uni_pc` and the
    /// legacy `unipc` spelling both select Wan's OWN native diffusers FLOW-SNR UniPC multistep — NOT
    /// gen-core's VE-space `uni_pc` solver, which would diverge from Wan's parity. `euler` is flow Euler;
    /// the curated `dpmpp_2m` and legacy `dpmpp2m` select the native flow-SNR DPM-Solver++(2M) (the
    /// MLX twin's menu, sc-27309). Any unknown name falls back to the UniPC default (N3 — never
    /// hard-fail over a sampling knob).
    pub fn parse(name: Option<&str>) -> Self {
        match name {
            Some("euler") => Sampler::Euler,
            Some("dpmpp_2m" | "dpmpp2m") => Sampler::Dpmpp2m,
            _ => Sampler::UniPC, // "uni_pc" (curated) / "unipc" (legacy) / default / anything else
        }
    }
}

const SOLVER_ORDER: usize = 2;

/// Flow log-SNR `λ(σ) = log((1−σ)/σ)`, with ±∞ at the σ=1 / σ=0 boundaries (matching `torch.log`).
fn log_snr(sigma: f64) -> f64 {
    if sigma >= 1.0 {
        return f64::NEG_INFINITY;
    }
    if sigma <= 0.0 {
        return f64::INFINITY;
    }
    ((1.0 - sigma) / sigma).ln()
}

/// The Wan flow-match σ schedule (f64): `σ_k = shift·s/(1+(shift−1)·s)` over
/// `s = linspace(1, 1/N, steps+1)[:-1]`, with the `σ_0 -= 1e-6` guard (keeps `log(1−σ_0)` finite for
/// UniPC) and a terminal `0`. Shared by the native [`FlowScheduler`] and the curated solver fold-in.
fn flow_sigmas_f64(steps: usize, shift: f64) -> Vec<f64> {
    let n = NUM_TRAIN_TIMESTEPS as f64;
    let mut sigmas: Vec<f64> = (0..steps)
        .map(|k| {
            let s = 1.0 + (1.0 / n - 1.0) * (k as f64) / (steps as f64); // linspace(1, 1/N, steps+1)[k]
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect();
    if (sigmas[0] - 1.0).abs() < 1e-6 {
        sigmas[0] -= 1e-6; // avoid log(1 - σ_0) = log(0)
    }
    sigmas.push(0.0);
    sigmas
}

/// The Wan flow-match σ schedule as `f32` (descending, length `steps + 1`, trailing `0.0`) — the native
/// schedule the unified curated solver fold-in (epic 7114 P4, sc-7124) integrates over via
/// [`candle_gen::run_flow_sampler`]. The gen-core-only solvers (euler_ancestral / heun / dpmpp_sde /
/// ddim) run over THIS schedule. The curated `uni_pc` name (sc-7296) maps to Wan's OWN native UniPC
/// (a diffusers FLOW-SNR multistep, λ = log((1−σ)/σ)); gen-core's VE-space `uni_pc`/`dpmpp_2m`
/// (λ = −ln σ) are deliberately NOT routed through the fold-in — they would diverge from Wan's parity.
pub fn flow_sigmas(steps: usize, shift: f64) -> Vec<f32> {
    flow_sigmas_f64(steps, shift)
        .iter()
        .map(|&s| s as f32)
        .collect()
}

pub struct FlowScheduler {
    sampler: Sampler,
    sigmas: Vec<f64>, // len = steps + 1 (terminal 0)
    // UniPC multistep state:
    model_outputs: Vec<Option<Tensor>>, // converted x0 preds, [older, newest]
    last_sample: Option<Tensor>,
    this_order: usize,
    lower_order_nums: usize,
    step_index: usize,
    // DPM-Solver++(2M) state: the previous step's x0 prediction.
    prev_x0: Option<Tensor>,
}

impl FlowScheduler {
    pub fn new(sampler: Sampler, steps: usize, shift: f64) -> Self {
        let sigmas = flow_sigmas_f64(steps, shift);
        Self {
            sampler,
            sigmas,
            model_outputs: vec![None; SOLVER_ORDER],
            last_sample: None,
            this_order: 1,
            lower_order_nums: 0,
            step_index: 0,
            prev_x0: None,
        }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// DiT timestep at step `i` (`σ_i · num_train_timesteps`).
    pub fn timestep(&self, i: usize) -> f64 {
        self.sigmas[i] * NUM_TRAIN_TIMESTEPS as f64
    }

    /// Advance one step given the model velocity `v` and current `sample` → next sample.
    pub fn step(&mut self, v: &Tensor, sample: &Tensor) -> Result<Tensor> {
        match self.sampler {
            Sampler::Euler => {
                let dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index];
                let next = (sample + v.affine(dt, 0.0)?)?;
                self.step_index += 1;
                Ok(next)
            }
            Sampler::UniPC => self.unipc_step(v, sample),
            Sampler::Dpmpp2m => self.dpmpp_2m_step(v, sample),
        }
    }

    /// DPM-Solver++(2M) over the flow log-SNR, with `lower_order_final`: first order on the first step,
    /// and on the last step of a short (< 15 step) schedule; the terminal `σ = 0` step returns the x0
    /// prediction. Coefficients in f64 — the same recurrence as `mlx-gen-wan`'s `FlowDpmpp2m`.
    fn dpmpp_2m_step(&mut self, v: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let i = self.step_index;
        let (sigma_cur, sigma_next) = (self.sigmas[i], self.sigmas[i + 1]);
        // convert_model_output (flow predict_x0): x0 = sample − σ·v.
        let x0 = (sample - v.affine(sigma_cur, 0.0)?)?;
        let num_steps = self.num_steps();
        let first_order = i + 1 == num_steps && num_steps < 15;
        let next = match &self.prev_x0 {
            _ if sigma_next == 0.0 => x0.clone(),
            Some(prev_x0) if !first_order => {
                let lambda_prev = log_snr(self.sigmas[i - 1]);
                let lambda_cur = log_snr(sigma_cur);
                let h = log_snr(sigma_next) - lambda_cur;
                let r0 = (lambda_cur - lambda_prev) / h;
                let alpha_e = (1.0 - sigma_next) * (-h).exp_m1();
                // D1 = (x0 − prev_x0)/r0; x_next = (σ_next/σ_cur)·x − α·e·x0 − ½·α·e·D1.
                let d1 = (&x0 - prev_x0)?.affine(1.0 / r0, 0.0)?;
                ((sample.affine(sigma_next / sigma_cur, 0.0)? - x0.affine(alpha_e, 0.0)?)?
                    - d1.affine(0.5 * alpha_e, 0.0)?)?
            }
            _ => {
                let h = log_snr(sigma_next) - log_snr(sigma_cur);
                let alpha_e = (1.0 - sigma_next) * (-h).exp_m1();
                (sample.affine(sigma_next / sigma_cur, 0.0)? - x0.affine(alpha_e, 0.0)?)?
            }
        };
        self.prev_x0 = Some(x0);
        self.step_index += 1;
        Ok(next)
    }

    fn unipc_step(&mut self, v: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let i = self.step_index;
        let sigma = self.sigmas[i];
        // convert_model_output (flow predict_x0): x0 = sample - σ·v.
        let x0 = (sample - v.affine(sigma, 0.0)?)?;

        let use_corrector = i > 0 && self.last_sample.is_some();
        let mut sample = sample.clone();
        if use_corrector {
            let last = self.last_sample.clone().unwrap();
            sample = self.uni_c(&x0, &last, &sample, self.this_order)?;
        }

        // Shift the model-output history and append the new x0.
        self.model_outputs[0] = self.model_outputs[1].take();
        self.model_outputs[1] = Some(x0);

        // lower_order_final order schedule.
        let this_order = SOLVER_ORDER.min(self.num_steps() - i);
        self.this_order = this_order.min(self.lower_order_nums + 1);

        self.last_sample = Some(sample.clone());
        let prev = self.uni_p(&sample, self.this_order)?;

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(prev)
    }

    /// UniP predictor (B(h)=bh2, predict_x0). `order ∈ {1,2}` with analytic ρ=0.5.
    fn uni_p(&self, sample: &Tensor, order: usize) -> Result<Tensor> {
        let i = self.step_index;
        let m0 = self.model_outputs[1].as_ref().unwrap();
        let (s_t, s_s0) = (self.sigmas[i + 1], self.sigmas[i]);
        let (a_t, a_s0) = (1.0 - s_t, 1.0 - s_s0);
        let h = (a_t.ln() - s_t.ln()) - (a_s0.ln() - s_s0.ln());
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();

        // x_t_ = (σ_t/σ_s0)·x − α_t·h_phi_1·m0.
        let xt_ = (sample.affine(s_t / s_s0, 0.0)? - m0.affine(a_t * h_phi_1, 0.0)?)?;
        if order >= 2 {
            let m_prev = self.model_outputs[0].as_ref().unwrap();
            let s_si = self.sigmas[i - 1];
            let lam_si = (1.0 - s_si).ln() - s_si.ln();
            let rk = (lam_si - (a_s0.ln() - s_s0.ln())) / h;
            let d1 = (m_prev - m0)?.affine(1.0 / rk, 0.0)?;
            // ρ_p = 0.5 ⇒ pred_res = 0.5·D1.
            xt_ - d1.affine(a_t * b_h * 0.5, 0.0)?
        } else {
            Ok(xt_)
        }
    }

    /// UniC corrector (B(h)=bh2, predict_x0). `order ∈ {1,2}`.
    fn uni_c(
        &self,
        this_x0: &Tensor,
        last_sample: &Tensor,
        _this_sample: &Tensor,
        order: usize,
    ) -> Result<Tensor> {
        let i = self.step_index;
        let m0 = self.model_outputs[1].as_ref().unwrap(); // previous step's x0
        let (s_t, s_s0) = (self.sigmas[i], self.sigmas[i - 1]);
        let (a_t, a_s0) = (1.0 - s_t, 1.0 - s_s0);
        let lam_s0 = a_s0.ln() - s_s0.ln();
        let h = (a_t.ln() - s_t.ln()) - lam_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();

        let xt_ = (last_sample.affine(s_t / s_s0, 0.0)? - m0.affine(a_t * h_phi_1, 0.0)?)?;
        let d1_t = (this_x0 - m0)?;
        if order == 1 {
            // ρ_c = 0.5.
            xt_ - d1_t.affine(a_t * b_h * 0.5, 0.0)?
        } else {
            let m_prev = self.model_outputs[0].as_ref().unwrap();
            let s_si = self.sigmas[i - 2];
            let lam_si = (1.0 - s_si).ln() - s_si.ln();
            let rk = (lam_si - lam_s0) / h;
            let d1 = (m_prev - m0)?.affine(1.0 / rk, 0.0)?;
            // b coefficients (same recurrence as the predictor).
            let h_phi_k0 = h_phi_1 / hh - 1.0;
            let b0 = h_phi_k0 / b_h;
            let h_phi_k1 = h_phi_k0 / hh - 0.5;
            let b1 = h_phi_k1 * 2.0 / b_h;
            // Solve [[1,1],[rk,1]] · ρ = [b0, b1].
            let det = 1.0 - rk;
            let r0 = (b0 - b1) / det;
            let r1 = (b1 - rk * b0) / det;
            let corr = (d1.affine(r0, 0.0)? + d1_t.affine(r1, 0.0)?)?;
            xt_ - corr.affine(a_t * b_h, 0.0)?
        }
    }
}

/// Default flow-shift unless the request overrides it.
pub fn flow_shift(req_shift: Option<f32>) -> f64 {
    req_shift.map(|s| s as f64).unwrap_or(FLOW_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::{FlowScheduler, Sampler};
    use candle_gen::candle_core::{Device, Tensor};

    #[test]
    fn dpmpp_2m_parses_from_the_curated_and_legacy_names() {
        assert_eq!(Sampler::parse(Some("dpmpp_2m")), Sampler::Dpmpp2m);
        assert_eq!(Sampler::parse(Some("dpmpp2m")), Sampler::Dpmpp2m);
    }

    /// On a straight rectified-flow path (`x_σ = (1−σ)·x0 + σ·ε`, constant velocity `ε − x0`) every
    /// DPM-Solver++(2M) update is exact — the first-order step lands on `(1−σ_next)·x0 + σ_next·ε`
    /// and the second-order correction `D1` vanishes — so each step must track the analytic path and
    /// the terminal step must return `x0`. Pure scheduler arithmetic on host tensors.
    #[test]
    fn dpmpp_2m_tracks_a_straight_flow_path_exactly() {
        let dev = Device::Cpu;
        let x0 = Tensor::new(&[0.25f32, -1.5, 3.0], &dev).unwrap();
        let eps = Tensor::new(&[1.0f32, 0.5, -2.0], &dev).unwrap();
        let v = (&eps - &x0).unwrap();
        for steps in [4usize, 20] {
            let mut sched = FlowScheduler::new(Sampler::Dpmpp2m, steps, 5.0);
            let path = |sigma: f64| {
                (x0.affine(1.0 - sigma, 0.0).unwrap() + eps.affine(sigma, 0.0).unwrap()).unwrap()
            };
            let mut x = path(sched.sigmas[0]);
            for i in 0..steps {
                x = sched.step(&v, &x).unwrap();
                let expected = path(sched.sigmas[i + 1]);
                let err = (&x - &expected)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(
                    err < 1e-5,
                    "steps {steps}, step {i}: off the flow path by {err}"
                );
            }
        }
    }

    #[test]
    fn parse_maps_curated_and_legacy_names() {
        // Curated `uni_pc` (sc-7296) and the legacy `unipc` spelling both select Wan's native UniPC;
        // the default (None) is UniPC; `euler` is flow Euler; unknown falls back to UniPC (N3).
        assert_eq!(Sampler::parse(Some("uni_pc")), Sampler::UniPC);
        assert_eq!(Sampler::parse(Some("unipc")), Sampler::UniPC);
        assert_eq!(Sampler::parse(None), Sampler::UniPC);
        assert_eq!(Sampler::parse(Some("euler")), Sampler::Euler);
        assert_eq!(Sampler::parse(Some("something_else")), Sampler::UniPC);
    }
}
