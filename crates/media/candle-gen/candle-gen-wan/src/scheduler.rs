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

/// The Wan flow-match σ schedule (f64): delegates to
/// [`candle_gen::gen_core::sampling::wan_candle_sigmas_f64`] (`PIPE-9`), preserving the
/// `σ_0 -= 1e-6` guard (keeps `log(1−σ_0)` finite for UniPC) and terminal `0`.
pub(crate) fn flow_sigmas_f64(steps: usize, shift: f64) -> Vec<f64> {
    candle_gen::gen_core::sampling::wan_candle_sigmas_f64(steps, shift, NUM_TRAIN_TIMESTEPS)
}

/// The Wan flow-match σ schedule as `f32` (descending, length `steps + 1`, trailing `0.0`) — the native
/// schedule the unified curated solver fold-in (epic 7114 P4, sc-7124) integrates over via
/// [`candle_gen::run_flow_sampler`]. The gen-core-only solvers (euler_ancestral / heun / dpmpp_sde /
/// ddim) run over THIS schedule. The curated `uni_pc` name (sc-7296) maps to Wan's OWN native UniPC
/// (a diffusers FLOW-SNR multistep, λ = log((1−σ)/σ)); gen-core's VE-space `uni_pc`/`dpmpp_2m`
/// (λ = −ln σ) are deliberately NOT routed through the fold-in — they would diverge from Wan's parity.
pub fn flow_sigmas(steps: usize, shift: f64) -> Vec<f32> {
    candle_gen::gen_core::sampling::wan_candle_sigmas(steps, shift, NUM_TRAIN_TIMESTEPS)
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
        Self::new_at_step(sampler, steps, shift, 0)
    }

    pub(crate) fn new_at_step(
        sampler: Sampler,
        steps: usize,
        shift: f64,
        start_step: usize,
    ) -> Self {
        let sigmas = flow_sigmas_f64(steps, shift);
        Self {
            sampler,
            sigmas,
            model_outputs: vec![None; SOLVER_ORDER],
            last_sample: None,
            this_order: 1,
            lower_order_nums: 0,
            step_index: start_step,
            prev_x0: None,
        }
    }

    /// Native f64 nodes, including the terminal zero, used by this scheduler instance.
    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub(crate) fn step_index(&self) -> usize {
        self.step_index
    }

    /// DiT timestep at step `i` (`σ_i · num_train_timesteps`).
    pub fn timestep(&self, i: usize) -> f64 {
        self.sigmas[i] * NUM_TRAIN_TIMESTEPS as f64
    }

    /// Advance one step given the model velocity `v` and current `sample` → next sample.
    pub fn step(&mut self, v: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if self.step_index == 0 {
            let sampler = match self.sampler {
                Sampler::Euler => "wan_flow_euler_v1",
                Sampler::UniPC => "wan_flow_unipc_bh2_order2_v1",
                Sampler::Dpmpp2m => "wan_flow_dpmpp_2m_v1",
            };
            candle_gen::execution::schedule_f64(sampler, &self.sigmas, "flow_velocity");
            candle_gen::execution::runtime(sample.device(), sample.dtype());
        }
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

// =====================================================================================
// Typed FamilyDriver for Candle Wan Native Flow-SNR Solvers (SW-6968, PIPE-3, PIPE-7)
// =====================================================================================

/// Forward-evaluation seam for [`WanNativeDriver`] (`SW-6968`, `PIPE-7`).
///
/// Generic over the forward predictor so weights-free Tier 0 tests can run
/// [`candle_gen::driver::conformance::verify_driver_conformance`] and expert-crossing `Carry`
/// trajectory parity checks over mocked forwards without loading any model weights.
pub trait WanNativeForward {
    /// Predict the flow velocity `v` for `latents` at absolute outer `step` and `timestep` (`σ·N`).
    fn predict_velocity(
        &self,
        latents: &Tensor,
        step: usize,
        timestep: f64,
    ) -> candle_gen::Result<Tensor>;
}

impl<F> WanNativeForward for F
where
    F: Fn(&Tensor, usize, f64) -> candle_gen::Result<Tensor>,
{
    fn predict_velocity(
        &self,
        latents: &Tensor,
        step: usize,
        timestep: f64,
    ) -> candle_gen::Result<Tensor> {
        self(latents, step, timestep)
    }
}

/// Request-owned Candle Wan multistep solver history stored inside
/// [`candle_gen::driver::DriverRunState`].
struct WanCandleHistory {
    sampler: Sampler,
    total_steps: usize,
    sigmas_bits: Vec<u64>,
    expected_next_step: usize,
    scheduler: FlowScheduler,
}

/// Typed [`candle_gen::driver::FamilyDriver`] for Candle Wan's native flow-SNR solvers (`UniPC`,
/// `Dpmpp2m`, `Euler`) across TI2V-5B, A14B MoE (resident + staged swap), and VACE / VACE-Fun
/// routes (`SW-6968`, `PIPE-3`, `PIPE-7`).
pub struct WanNativeDriver<F> {
    sampler: Sampler,
    num_steps: usize,
    shift: f64,
    sigmas: Vec<f64>,
    sigmas_bits: Vec<u64>,
    forward: F,
    support: candle_gen::driver::DriverSupport,
}

impl<F: WanNativeForward> WanNativeDriver<F> {
    fn build_support(driver_id: &'static str) -> candle_gen::driver::DriverSupport {
        candle_gen::driver::DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: candle_gen::driver::DriverRefusalReasons {
                renoise_from_clean:
                    "native Candle Wan flow-SNR driver does not re-noise from clean estimates",
                geometry_change:
                    "native Candle Wan flow-SNR driver does not support geometry-changing phase boundaries",
                read_only_boundary_estimate:
                    "native Candle Wan flow-SNR driver does not expose read-only boundary clean estimates",
                ..candle_gen::driver::DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(
        driver_id: &'static str,
        sampler: Sampler,
        num_steps: usize,
        shift: f64,
        forward: F,
    ) -> Self {
        let sigmas = flow_sigmas_f64(num_steps, shift);
        let sigmas_bits = sigmas.iter().map(|s| s.to_bits()).collect();
        Self {
            sampler,
            num_steps,
            shift,
            sigmas,
            sigmas_bits,
            forward,
            support: Self::build_support(driver_id),
        }
    }

    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }
}

impl<F: WanNativeForward> candle_gen::driver::FamilyDriver<Tensor> for WanNativeDriver<F> {
    fn support(&self) -> &candle_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: candle_gen::driver::SliceInput<'_, Tensor>,
        state: &mut candle_gen::driver::DriverRunState,
        hooks: &mut candle_gen::driver::StepHooks<'_, Tensor>,
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps != self.num_steps || input.span.end > self.num_steps {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Wan schedule length {}",
                self.support.driver_id, input.span, input.total_steps, self.num_steps
            )));
        }

        // Validate Carry compatibility BEFORE mutating `state` (`PIPE-3`).
        let carrying = input.transition == candle_gen::driver::SliceTransition::Continue
            && input.history == candle_gen::driver::DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<WanCandleHistory>() else {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry requested without prior Wan solver history in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.sampler != self.sampler {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible solver change ({:?} -> {:?})",
                    self.support.driver_id, existing.sampler, self.sampler
                )));
            }
            if existing.total_steps != input.total_steps || existing.sigmas_bits != self.sigmas_bits
            {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible Wan sigma schedule",
                    self.support.driver_id
                )));
            }
            if existing.expected_next_step != input.span.start
                || existing.scheduler.step_index() != input.span.start
            {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused at non-contiguous step {} (history at {})",
                    self.support.driver_id,
                    input.span.start,
                    existing.scheduler.step_index()
                )));
            }
        }

        state.enter_slice(&input)?;
        if !carrying {
            state.set_history(WanCandleHistory {
                sampler: self.sampler,
                total_steps: input.total_steps,
                sigmas_bits: self.sigmas_bits.clone(),
                expected_next_step: input.span.start,
                scheduler: FlowScheduler::new_at_step(
                    self.sampler,
                    self.num_steps,
                    self.shift,
                    input.span.start,
                ),
            });
        }

        let mut latents = match input.state {
            candle_gen::gen_core::BoundaryState::XSigma(x)
            | candle_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | candle_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            let info = candle_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: candle_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: self.sigmas[step] as f32,
                next_sigma: self.sigmas[step + 1] as f32,
            };
            hooks.observe(&info, &latents)?;
            let timestep = self.sigmas[step] * NUM_TRAIN_TIMESTEPS as f64;
            let velocity = self
                .forward
                .predict_velocity(&latents, step, timestep)
                .map_err(candle_gen::gen_core::Error::from)?;
            state.record_evaluation(candle_gen::driver::DriverEvaluationKind::Trajectory);
            if input.cancel.is_cancelled() {
                return Err(candle_gen::gen_core::Error::Canceled);
            }
            latents = {
                let history = state
                    .history_mut::<WanCandleHistory>()
                    .expect("wan candle solver history initialized on slice entry");
                let next = history
                    .scheduler
                    .step(&velocity, &latents)
                    .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))?;
                history.expected_next_step = step + 1;
                next
            };
            latents = hooks.post_step(&info, latents)?;
            hooks.materialize(&info, &latents)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(candle_gen::gen_core::BoundaryState::XSigma(latents))
    }
}

#[cfg(test)]
mod tests {
    use super::{FlowScheduler, Sampler, WanNativeDriver};
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

    fn tensor_bit_equal(a: &Tensor, b: &Tensor) -> bool {
        let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn mock_expert_velocity(
        x: &Tensor,
        step: usize,
        timestep: f64,
        slope: f64,
        bias: f64,
    ) -> candle_gen::Result<Tensor> {
        let scale = slope + 0.00025 * timestep + 0.01 * (step as f64);
        let offset = bias - 0.0001 * timestep;
        Ok(x.affine(scale, offset)?)
    }

    #[test]
    fn wan_candle_driver_passes_conformance() {
        use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        for sampler in [Sampler::UniPC, Sampler::Dpmpp2m, Sampler::Euler] {
            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 6,
                split_at: 3,
                seed: 42,
                base_geometry: (64, 64),
                alt_geometry: (128, 128),
                make_driver: move || {
                    Box::new(WanNativeDriver::new(
                        "wan_native_candle_test",
                        sampler,
                        6,
                        5.0,
                        |x: &Tensor, step: usize, t: f64| {
                            mock_expert_velocity(x, step, t, 0.35, -0.1)
                        },
                    ))
                },
                make_initial: || {
                    Tensor::new(
                        &[0.4_f32, -0.8, 1.2, -0.3, 0.7, 0.1, -1.1, 0.9],
                        &Device::Cpu,
                    )
                    .unwrap()
                    .reshape((1, 2, 1, 2, 2))
                    .unwrap()
                },
                latents_equal: tensor_bit_equal,
            })
            .unwrap_or_else(|e| panic!("conformance failed for {sampler:?}: {e}"));
            assert!(report.verified_inert_split, "{sampler:?}");
            assert!(report.verified_reset_preserves_draws, "{sampler:?}");
            assert!(report.verified_cancellation, "{sampler:?}");
        }
    }

    #[test]
    fn wan_candle_unipc_expert_crossing_carry_matches_legacy_bit_for_bit() {
        use candle_gen::driver::{
            DriverHistoryPolicy, DriverRunState, FamilyDriver, SliceInput, SliceTransition,
            StepHooks,
        };
        use candle_gen::gen_core::{BoundaryState, CancelFlag, Progress};

        let steps = 8usize;
        let shift = 5.0_f64;
        let init = Tensor::new(
            &[0.25_f32, -0.75, 1.1, -0.4, 0.6, -0.2, 0.9, -1.3],
            &Device::Cpu,
        )
        .unwrap()
        .reshape((1, 2, 1, 2, 2))
        .unwrap();
        let cancel = CancelFlag::default();

        // Legacy single-FlowScheduler reference across a high->low expert crossing.
        let mut legacy_sched = FlowScheduler::new(Sampler::UniPC, steps, shift);
        let crossing = (0..steps)
            .find(|&i| legacy_sched.timestep(i) < 875.0)
            .unwrap_or(steps);
        assert!(crossing > 0 && crossing < steps, "crossing={crossing}");

        let mut legacy_latents = init.clone();
        for i in 0..steps {
            let t = legacy_sched.timestep(i);
            let v = if i < crossing {
                mock_expert_velocity(&legacy_latents, i, t, 0.45, 0.12).unwrap()
            } else {
                mock_expert_velocity(&legacy_latents, i, t, 0.22, -0.07).unwrap()
            };
            legacy_latents = legacy_sched.step(&v, &legacy_latents).unwrap();
        }

        // Typed FamilyDriver with two distinct expert drivers sharing one DriverRunState via Carry.
        let high_driver = WanNativeDriver::new(
            "wan_candle_moe_high",
            Sampler::UniPC,
            steps,
            shift,
            |x: &Tensor, step: usize, t: f64| mock_expert_velocity(x, step, t, 0.45, 0.12),
        );
        let low_driver = WanNativeDriver::new(
            "wan_candle_moe_low",
            Sampler::UniPC,
            steps,
            shift,
            |x: &Tensor, step: usize, t: f64| mock_expert_velocity(x, step, t, 0.22, -0.07),
        );

        let mut state = DriverRunState::new();
        let mut ticks = Vec::new();
        let high_out = {
            let mut on_progress = |p: Progress| ticks.push(p);
            let mut hooks = StepHooks::new(&mut on_progress);
            high_driver
                .run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: 0..crossing,
                        total_steps: steps,
                        state: BoundaryState::XSigma(init),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: None,
                        seed: 7,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut state,
                    &mut hooks,
                )
                .unwrap()
        };
        let low_out = {
            let mut on_progress = |p: Progress| ticks.push(p);
            let mut hooks = StepHooks::new(&mut on_progress);
            low_driver
                .run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: crossing..steps,
                        total_steps: steps,
                        state: high_out,
                        transition: SliceTransition::Continue,
                        history: DriverHistoryPolicy::Carry,
                        geometry: None,
                        seed: 7,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut state,
                    &mut hooks,
                )
                .unwrap()
        };
        let BoundaryState::XSigma(carried_latents) = low_out else {
            panic!("expected XSigma boundary state");
        };
        assert!(
            tensor_bit_equal(&carried_latents, &legacy_latents),
            "Carry across expert crossing must match legacy single-FlowScheduler UniPC bit-for-bit"
        );
        let expected_ticks: Vec<Progress> = (1..=steps)
            .map(|i| Progress::Step {
                current: i as u32,
                total: steps as u32,
            })
            .collect();
        assert_eq!(ticks, expected_ticks);
        assert_eq!(state.completed_outer_steps, steps);
        assert_eq!(state.history_resets, 0);
    }

    #[test]
    fn wan_candle_carry_refuses_incompatible_and_reset_is_rng_independent() {
        use candle_gen::driver::{
            DriverHistoryPolicy, DriverRunState, FamilyDriver, SliceInput, SliceTransition,
            StepHooks,
        };
        use candle_gen::gen_core::{BoundaryState, CancelFlag, Error as CoreError, Progress};

        let steps = 6usize;
        let split = 3usize;
        let cancel = CancelFlag::default();
        let init = Tensor::new(&[0.5_f32, -0.5, 0.9, -0.2], &Device::Cpu).unwrap();

        let unipc = WanNativeDriver::new(
            "wan_candle_unipc",
            Sampler::UniPC,
            steps,
            5.0,
            |x: &Tensor, step: usize, t: f64| mock_expert_velocity(x, step, t, 0.4, 0.1),
        );
        let dpmpp = WanNativeDriver::new(
            "wan_candle_dpmpp",
            Sampler::Dpmpp2m,
            steps,
            5.0,
            |x: &Tensor, step: usize, t: f64| mock_expert_velocity(x, step, t, 0.4, 0.1),
        );
        let diff_shift = WanNativeDriver::new(
            "wan_candle_shift",
            Sampler::UniPC,
            steps,
            3.0,
            |x: &Tensor, step: usize, t: f64| mock_expert_velocity(x, step, t, 0.4, 0.1),
        );

        // Missing prior history refuses Carry.
        {
            let mut empty_state = DriverRunState::new();
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let res = unipc.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: split..steps,
                    total_steps: steps,
                    state: BoundaryState::XSigma(init.clone()),
                    transition: SliceTransition::Continue,
                    history: DriverHistoryPolicy::Carry,
                    geometry: None,
                    seed: 1,
                    cancel: &cancel,
                    request_boundary_estimate: false,
                },
                &mut empty_state,
                &mut hooks,
            );
            assert!(matches!(res, Err(CoreError::Unsupported(_))));
        }

        // Run first half with UniPC.
        let mut state = DriverRunState::new();
        state.next_draw_index();
        state.next_draw_index();
        let draws_before = state.stochastic_draw_index;
        let mid = {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            unipc
                .run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: 0..split,
                        total_steps: steps,
                        state: BoundaryState::XSigma(init.clone()),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: None,
                        seed: 1,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut state,
                    &mut hooks,
                )
                .unwrap()
        };
        let BoundaryState::XSigma(mid_tensor) = mid else {
            panic!("expected XSigma");
        };

        // Incompatible solver refuses Carry.
        {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let res = dpmpp.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: split..steps,
                    total_steps: steps,
                    state: BoundaryState::XSigma(mid_tensor.clone()),
                    transition: SliceTransition::Continue,
                    history: DriverHistoryPolicy::Carry,
                    geometry: None,
                    seed: 1,
                    cancel: &cancel,
                    request_boundary_estimate: false,
                },
                &mut state,
                &mut hooks,
            );
            assert!(matches!(res, Err(CoreError::Unsupported(_))));
        }

        // Incompatible shift refuses Carry.
        {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let res = diff_shift.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: split..steps,
                    total_steps: steps,
                    state: BoundaryState::XSigma(mid_tensor.clone()),
                    transition: SliceTransition::Continue,
                    history: DriverHistoryPolicy::Carry,
                    geometry: None,
                    seed: 1,
                    cancel: &cancel,
                    request_boundary_estimate: false,
                },
                &mut state,
                &mut hooks,
            );
            assert!(matches!(res, Err(CoreError::Unsupported(_))));
        }

        // Continue + Reset clears UniPC history without altering stochastic_draw_index,
        // and produces a distinct trajectory from Carry.
        let mut carried_state = DriverRunState::new();
        let carried = {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let m = unipc
                .run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 0,
                        span: 0..split,
                        total_steps: steps,
                        state: BoundaryState::XSigma(init),
                        transition: SliceTransition::Start,
                        history: DriverHistoryPolicy::Reset,
                        geometry: None,
                        seed: 1,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut carried_state,
                    &mut hooks,
                )
                .unwrap();
            unipc
                .run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: split..steps,
                        total_steps: steps,
                        state: m,
                        transition: SliceTransition::Continue,
                        history: DriverHistoryPolicy::Carry,
                        geometry: None,
                        seed: 1,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut carried_state,
                    &mut hooks,
                )
                .unwrap()
        };
        let reset_out = {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            unipc
                .run_slice(
                    SliceInput {
                        pass: 0,
                        phase: 1,
                        span: split..steps,
                        total_steps: steps,
                        state: BoundaryState::XSigma(mid_tensor),
                        transition: SliceTransition::Continue,
                        history: DriverHistoryPolicy::Reset,
                        geometry: None,
                        seed: 1,
                        cancel: &cancel,
                        request_boundary_estimate: false,
                    },
                    &mut state,
                    &mut hooks,
                )
                .unwrap()
        };
        assert_eq!(state.stochastic_draw_index, draws_before);
        assert_eq!(state.history_resets, 1);
        let (BoundaryState::XSigma(c_t), BoundaryState::XSigma(r_t)) = (carried, reset_out) else {
            panic!("expected XSigma");
        };
        assert!(
            !tensor_bit_equal(&c_t, &r_t),
            "Reset must actually clear UniPC multistep history"
        );
    }

    #[test]
    fn wan_candle_driver_post_step_hook_applies_ti2v_blend() {
        use candle_gen::driver::{DriverRunState, FamilyDriver, SliceInput, StepHooks};
        use candle_gen::gen_core::{BoundaryState, CancelFlag, Progress};

        let steps = 4usize;
        let cancel = CancelFlag::default();
        let dev = Device::Cpu;
        let clean = Tensor::new(&[3.0_f32, -2.0, 0.0, 0.0], &dev).unwrap();
        let mask = Tensor::new(&[0.0_f32, 0.0, 1.0, 1.0], &dev).unwrap();
        let init = Tensor::new(&[3.0_f32, -2.0, 0.5, -0.5], &dev).unwrap();
        let driver = WanNativeDriver::new(
            "wan_candle_ti2v_hook",
            Sampler::UniPC,
            steps,
            5.0,
            |x: &Tensor, step: usize, t: f64| mock_expert_velocity(x, step, t, 0.3, 0.05),
        );
        let mut state = DriverRunState::new();
        let mut noop = |_: Progress| {};
        let mut blend = |_info: &candle_gen::driver::DriverStepInfo, latents: Tensor| {
            crate::pipeline::ti2v_blend(&clean, &mask, &latents)
                .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))
        };
        let mut hooks = StepHooks::new(&mut noop);
        hooks.on_post_step = Some(&mut blend);
        let out = driver
            .run_slice(
                SliceInput::full_trajectory(init, steps, 1, &cancel),
                &mut state,
                &mut hooks,
            )
            .unwrap();
        let BoundaryState::XSigma(t) = out else {
            panic!("expected XSigma");
        };
        let vals = t.to_vec1::<f32>().unwrap();
        assert_eq!(vals[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(vals[1].to_bits(), (-2.0_f32).to_bits());
        assert!(vals[2].is_finite() && vals[3].is_finite());
    }
}

#[cfg(test)]
mod execution_identity_tests {
    use super::*;
    fn output() -> candle_gen::gen_core::GenerationOutput {
        candle_gen::gen_core::GenerationOutput::Video {
            frames: vec![candle_gen::gen_core::Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            }],
            fps: 24,
            audio: None,
        }
    }

    fn report(
        name: Option<&str>,
        shift: f64,
    ) -> candle_gen::gen_core::generator::ExecutionIdentity {
        let attempt = candle_gen::execution::Attempt::begin();
        let device = candle_gen::candle_core::Device::Cpu;
        let noise = crate::pipeline::create_noise(71, 1, 1, 1, 1, &device).unwrap();
        let mut scheduler = FlowScheduler::new(Sampler::parse(name), 3, shift);
        let velocity = noise.zeros_like().unwrap();
        // Raw scheduler arithmetic only: no learned model is constructed or executed.
        let _ = scheduler.step(&velocity, &noise).unwrap();
        candle_gen::execution::policy("test_decode", serde_json::json!("no_model_fixture"));
        attempt.finish_image(&crate::descriptor(), &output())
    }

    #[test]
    fn resolved_report_uses_native_fallback_and_exact_schedule() {
        let default = report(None, 5.0);
        assert_eq!(default, report(Some("unknown"), 5.0));
        assert_eq!(default, report(Some("unipc"), 5.0));
        assert_ne!(default, report(Some("euler"), 5.0));
        assert_ne!(default, report(None, 3.0));
        let resolved = default.resolved().expect("observed native scheduler");
        resolved.validate(crate::MODEL_ID, "candle", 1).unwrap();
        let candle_gen::gen_core::generator::ExecutionFact::Resolved { value } = &resolved.schedule
        else {
            panic!("missing schedule")
        };
        assert_eq!(value[0]["schedule"]["element"], "f64_le_bits");
        assert_eq!(value[0]["schedule"]["count"], 4);
    }

    #[test]
    fn scheduler_construction_and_failed_attempt_do_not_publish_stale_facts() {
        let _completed = report(None, 5.0);
        let attempt = candle_gen::execution::Attempt::begin();
        let _unused = FlowScheduler::new(Sampler::Euler, 5, 7.0);
        assert!(matches!(
            attempt.finish_image(&crate::descriptor(), &output()),
            candle_gen::gen_core::generator::ExecutionIdentity::Unavailable { .. }
        ));
        {
            let _failed = candle_gen::execution::Attempt::begin();
            candle_gen::execution::draw("abandoned", 99, 0);
        }
        let retry = report(None, 5.0);
        let candle_gen::gen_core::generator::ExecutionRandomness::Draws { draws, .. } =
            &retry.resolved().unwrap().randomness
        else {
            panic!("draws")
        };
        assert!(draws.iter().all(|draw| draw.seed == Some(71)));
    }
}
