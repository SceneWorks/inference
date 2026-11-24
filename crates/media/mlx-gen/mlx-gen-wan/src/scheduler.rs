//! Wan2.2 flow-match schedulers — port of `models/wan/scheduler.py`.
//!
//! Three rectified-flow solvers share one **shifted-sigma schedule** with **integer timesteps**
//! (`(σ·1000).trunc`): [`FlowMatchEuler`] (1st-order), [`FlowDpmpp2m`] (DPM-Solver++(2M)), and
//! [`FlowUniPC`] (the **default** — order-2 predictor-corrector). This is *not* the core mflux
//! `FlowMatchEuler` (`linspace(1,1/n,n)` + exponential time-shift); Wan builds
//! `linspace(σ_max, σ_min, n+1)[:-1]` from the unshifted training bounds and applies the algebraic
//! shift `σ' = shift·σ / (1 + (shift−1)·σ)` once, then appends a terminal `0.0`.
//!
//! Coefficient math runs in **f64** (the reference's Python-float / numpy path); only the final
//! array combinations run in the latent dtype (f32 for the 5B T2V/TI2V path). `_sigmas_float` in the
//! reference is `sigmas.tolist()` on a float32 array, i.e. f64 values that equal the f32 sigmas
//! exactly — mirrored here by storing f32 sigmas and widening to f64 for scalar math.

use mlx_rs::ops::{add, subtract};
use mlx_rs::Array;

// `x · f64`, the weak scalar rounded to the array's f32 dtype, is core's `fscale` (sc-27448).
use mlx_gen::array::fscale;
use mlx_gen::{Error, Result};

/// Which flow-match solver to use. `UniPC` is the Wan2.2 default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    Euler,
    Dpmpp2m,
    UniPC,
}

impl SolverKind {
    /// Resolve a sampler name (CLI/string) to a native solver. Accepts the curated gen-core vocabulary
    /// (`uni_pc`/`dpmpp_2m`, epic 7114 sc-7296) and the legacy spellings (`unipc`/`dpmpp2m`/`dpm++`/…);
    /// `uni_pc`/`unipc` and any unknown name fall back to UniPC, matching `generate_wan.py`'s default.
    pub fn from_name(name: &str) -> SolverKind {
        match name.to_ascii_lowercase().as_str() {
            "euler" => SolverKind::Euler,
            "dpmpp_2m" | "dpmpp2m" | "dpm++" | "dpmpp" | "dpm++2m" => SolverKind::Dpmpp2m,
            // `uni_pc` (curated) / `unipc` (legacy) and anything unknown → UniPC (the Wan default).
            _ => SolverKind::UniPC,
        }
    }
}

/// The common scheduler interface used by the denoise loop.
pub trait WanScheduler {
    /// Build the schedule for `num_steps` with time-`shift`.
    fn set_timesteps(&mut self, num_steps: usize, shift: f32);
    /// Sigmas, length `num_steps + 1` (trailing `0.0`).
    fn sigmas(&self) -> &[f32];
    /// Integer-valued model timesteps, length `num_steps` (`(σ·num_train).trunc`).
    fn timesteps(&self) -> &[f32];
    /// One denoise step; advances the internal step index. `model_output` is the velocity `v`.
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array>;
}

/// Guard the per-step `sigmas_f64[step_index(+1)]` indexing every `step` impl performs (F-019).
/// `WanScheduler` is public, so a caller that miscounts steps — over-steps past `num_steps`, or steps
/// before `set_timesteps` builds the schedule — must get a typed `Error`, not an out-of-bounds panic
/// (and, for UniPC, a `num_steps - i` underflow). A built schedule has `num_steps + 1` sigmas
/// (trailing terminal `0.0`), so a valid step needs `step_index + 1 < sigmas.len()`.
fn ensure_step_in_range(step_index: usize, sigmas_len: usize) -> Result<()> {
    if step_index + 1 >= sigmas_len {
        return Err(Error::Msg(format!(
            "WanScheduler::step over-stepped: step_index {step_index} but the schedule has {} \
             step(s) ({sigmas_len} sigmas) — call set_timesteps and run exactly that many steps",
            sigmas_len.saturating_sub(1)
        )));
    }
    Ok(())
}

/// Construct a boxed scheduler for `kind`.
pub fn make_scheduler(kind: SolverKind, num_train_timesteps: usize) -> Box<dyn WanScheduler> {
    match kind {
        SolverKind::Euler => Box::new(FlowMatchEuler::new(num_train_timesteps)),
        SolverKind::Dpmpp2m => Box::new(FlowDpmpp2m::new(num_train_timesteps)),
        SolverKind::UniPC => Box::new(FlowUniPC::new(num_train_timesteps, 2)),
    }
}

/// Shifted-sigma schedule (`_compute_sigmas`): `num_steps + 1` f32 sigmas, trailing `0.0`.
///
/// Delegates to [`mlx_gen::gen_core::sampling::wan_mlx_sigmas`] (`PIPE-9`):
/// `σ_max = (N−1)/N`, `σ_min = 0`, `linspace(σ_max, σ_min, n+1)[:-1]`, algebraic shift, trailing `0.0`.
pub fn compute_sigmas(num_steps: usize, shift: f32, num_train_timesteps: usize) -> Vec<f32> {
    mlx_gen::gen_core::sampling::wan_mlx_sigmas(num_steps, shift, num_train_timesteps)
}

/// Integer-valued model timesteps from sigmas: `(σ[:-1] · num_train).trunc` (`PIPE-9`).
pub fn integer_timesteps(sigmas: &[f32], num_train_timesteps: usize) -> Vec<f32> {
    mlx_gen::gen_core::sampling::wan_mlx_timesteps(sigmas, num_train_timesteps)
}

// =====================================================================================
// Euler
// =====================================================================================

/// 1st-order Euler flow-match scheduler: `x_next = x + (σ_next − σ_cur)·v`.
pub struct FlowMatchEuler {
    num_train_timesteps: usize,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
}

impl FlowMatchEuler {
    pub fn new(num_train_timesteps: usize) -> Self {
        Self {
            num_train_timesteps,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
        }
    }
}

impl WanScheduler for FlowMatchEuler {
    fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = integer_timesteps(&self.sigmas, self.num_train_timesteps);
        self.sigmas_f64 = self.sigmas.iter().map(|&s| s as f64).collect();
        self.step_index = 0;
    }
    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let i = self.step_index;
        ensure_step_in_range(i, self.sigmas_f64.len())?;
        let dt = self.sigmas_f64[i + 1] - self.sigmas_f64[i];
        let x_next = add(sample, &fscale(model_output, dt)?)?;
        self.step_index += 1;
        Ok(x_next)
    }
}

// =====================================================================================
// DPM-Solver++(2M)
// =====================================================================================

/// log-SNR `λ(σ) = log((1−σ)/σ)`, with ±∞ at the σ=1 / σ=0 boundaries (matching `torch.log`).
fn log_snr(sigma: f64) -> f64 {
    if sigma >= 1.0 {
        return f64::NEG_INFINITY;
    }
    if sigma <= 0.0 {
        return f64::INFINITY;
    }
    ((1.0 - sigma) / sigma).ln()
}

/// DPM-Solver++(2M): 2nd-order multistep, falls back to 1st order on the first/last step.
pub struct FlowDpmpp2m {
    num_train_timesteps: usize,
    lower_order_final: bool,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
    num_steps: usize,
    prev_x0: Option<Array>,
}

impl FlowDpmpp2m {
    pub fn new(num_train_timesteps: usize) -> Self {
        Self {
            num_train_timesteps,
            lower_order_final: true,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
            num_steps: 0,
            prev_x0: None,
        }
    }
}

impl WanScheduler for FlowDpmpp2m {
    fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = integer_timesteps(&self.sigmas, self.num_train_timesteps);
        self.sigmas_f64 = self.sigmas.iter().map(|&s| s as f64).collect();
        self.step_index = 0;
        self.num_steps = num_steps;
        self.prev_x0 = None;
    }
    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let i = self.step_index;
        ensure_step_in_range(i, self.sigmas_f64.len())?;
        let s = &self.sigmas_f64;
        let sigma_cur = s[i];
        let sigma_next = s[i + 1];

        // x0 = sample − σ_cur · v
        let x0 = subtract(sample, &fscale(model_output, sigma_cur)?)?;

        let use_first_order = self.prev_x0.is_none()
            || (self.lower_order_final && i == self.num_steps - 1 && self.num_steps < 15);

        let x_next = if use_first_order || sigma_next == 0.0 {
            if sigma_next == 0.0 {
                x0.clone()
            } else {
                let lambda_cur = log_snr(sigma_cur);
                let lambda_next = log_snr(sigma_next);
                let h = lambda_next - lambda_cur;
                let alpha_next = 1.0 - sigma_next;
                let coeff_x = sigma_next / sigma_cur;
                let coeff_x0 = alpha_next * (-h).exp_m1();
                subtract(&fscale(sample, coeff_x)?, &fscale(&x0, coeff_x0)?)?
            }
        } else {
            let sigma_prev = s[i - 1];
            let lambda_prev = log_snr(sigma_prev);
            let lambda_cur = log_snr(sigma_cur);
            let lambda_next = log_snr(sigma_next);
            let h = lambda_next - lambda_cur;
            let h_0 = lambda_cur - lambda_prev;
            let r0 = h_0 / h;
            let alpha_next = 1.0 - sigma_next;
            let exp_neg_h_m1 = (-h).exp_m1(); // expm1(-h)

            // D0 = x0 ; D1 = (1/r0)·(x0 − prev_x0)
            let prev_x0 = self.prev_x0.as_ref().expect("2nd order requires history");
            let d1 = fscale(&subtract(&x0, prev_x0)?, 1.0 / r0)?;
            // x_next = (σ_next/σ_cur)·sample − (α·e)·D0 − 0.5·(α·e)·D1
            let term0 = fscale(sample, sigma_next / sigma_cur)?;
            let term1 = fscale(&x0, alpha_next * exp_neg_h_m1)?;
            let term2 = fscale(&d1, 0.5 * alpha_next * exp_neg_h_m1)?;
            subtract(&subtract(&term0, &term1)?, &term2)?
        };

        self.prev_x0 = Some(x0);
        self.step_index += 1;
        Ok(x_next)
    }
}

// =====================================================================================
// UniPC (default)
// =====================================================================================

/// UniPC predictor-corrector (B(h)=expm1(−h) / bh2 variant), `solver_order` (=2 for all Wan
/// configs). The order-2 corrector solves a 2×2 system via `solve_linear` (the reference's
/// `np.linalg.solve`), assembled with the f64 factorial / `h_phi` recurrence.
pub struct FlowUniPC {
    num_train_timesteps: usize,
    solver_order: usize,
    lower_order_final: bool,
    use_corrector: bool,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
    num_steps: usize,
    lower_order_nums: usize,
    model_outputs: Vec<Option<Array>>, // x0 history, newest-last
    last_sample: Option<Array>,
    this_order: usize,
}

impl FlowUniPC {
    pub fn new(num_train_timesteps: usize, solver_order: usize) -> Self {
        Self {
            num_train_timesteps,
            solver_order,
            lower_order_final: true,
            use_corrector: true,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
            num_steps: 0,
            lower_order_nums: 0,
            model_outputs: vec![None; solver_order],
            last_sample: None,
            this_order: 1,
        }
    }

    /// Predictor `multistep_uni_p_bh_update`: predict `x_next` from x0 history. Returns the base
    /// prediction plus the order-≥2 correction (analytic ρ=[0.5] when effective_order ≤ 2).
    fn uni_p_bh2(&self, x0: &Array, sample: &Array, order: usize) -> Result<Array> {
        let i = self.step_index;
        let s = &self.sigmas_f64;
        let sigma_s0 = s[i];
        let sigma_t = s[i + 1];
        if sigma_t == 0.0 {
            return Ok(x0.clone());
        }
        let lambda_s0 = log_snr(sigma_s0);
        let lambda_t = log_snr(sigma_t);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let alpha_t = 1.0 - sigma_t;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;

        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .expect("history newest must be set before predict");
        // Base: (σ_t/σ_s0)·sample − (α_t·h_phi_1)·m0
        let mut x_t = subtract(
            &fscale(sample, sigma_t / sigma_s0)?,
            &fscale(m0, alpha_t * h_phi_1)?,
        )?;

        if order >= 2 {
            let mut rks: Vec<f64> = Vec::new();
            let mut d1s: Vec<Array> = Vec::new();
            for k in 1..order {
                let si_idx = i as isize - k as isize;
                let hist = self
                    .model_outputs
                    .get(self.solver_order.wrapping_sub(k + 1));
                let mk = match (si_idx >= 0, hist) {
                    (true, Some(Some(mk))) => mk,
                    _ => break,
                };
                let sigma_sk = s[si_idx as usize];
                let lambda_sk = log_snr(sigma_sk);
                let rk = (lambda_sk - lambda_s0) / h;
                if rk.is_infinite() {
                    break;
                }
                rks.push(rk);
                d1s.push(fscale(&subtract(mk, m0)?, 1.0 / rk)?);
            }

            if !d1s.is_empty() {
                let effective_order = d1s.len() + 1;
                let rhos_p: Vec<f64> = if effective_order <= 2 {
                    vec![0.5]
                } else {
                    let (r, b) = build_rb(&rks, h_phi_1, hh, b_h, 1, effective_order)?;
                    solve_linear(&r, &b)?
                };
                // pred_res = Σ ρ·D1 ; x_t -= (α_t·B_h)·pred_res. (rhos_p.len() == d1s.len().)
                let mut pred_res = fscale(&d1s[0], rhos_p[0])?;
                for (rho, d1) in rhos_p.iter().zip(d1s.iter()).skip(1) {
                    pred_res = add(&pred_res, &fscale(d1, *rho)?)?;
                }
                x_t = subtract(&x_t, &fscale(&pred_res, alpha_t * b_h)?)?;
            }
        }
        Ok(x_t)
    }

    /// Corrector `multistep_uni_c_bh_update`: refine the current sample using the freshly-computed
    /// model output. effective_order==1 → ρ=[0.5]; else solve the (effective_order)² system.
    fn uni_c_bh2(
        &self,
        model_x0: &Array,
        last_sample: &Array,
        this_sample: &Array,
        order: usize,
    ) -> Result<Array> {
        let i = self.step_index;
        let s = &self.sigmas_f64;
        let sigma_s0 = s[i - 1];
        let sigma_t = s[i];
        if sigma_t == 0.0 {
            return Ok(this_sample.clone());
        }
        let lambda_s0 = log_snr(sigma_s0);
        let lambda_t = log_snr(sigma_t);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let alpha_t = 1.0 - sigma_t;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;

        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .expect("history newest must be set before correct");
        // Re-derive base from last_sample.
        let x_t_ = subtract(
            &fscale(last_sample, sigma_t / sigma_s0)?,
            &fscale(m0, alpha_t * h_phi_1)?,
        )?;
        let d1_t = subtract(model_x0, m0)?;

        let mut rks: Vec<f64> = Vec::new();
        let mut d1s: Vec<Array> = Vec::new();
        for k in 1..order {
            let si_idx = i as isize - (k as isize + 1);
            let hist = self
                .model_outputs
                .get(self.solver_order.wrapping_sub(k + 1));
            let mk = match (si_idx >= 0, hist) {
                (true, Some(Some(mk))) => mk,
                _ => break,
            };
            let sigma_sk = s[si_idx as usize];
            let lambda_sk = log_snr(sigma_sk);
            let rk = (lambda_sk - lambda_s0) / h;
            if rk.is_infinite() {
                break;
            }
            rks.push(rk);
            d1s.push(fscale(&subtract(mk, m0)?, 1.0 / rk)?);
        }
        rks.push(1.0);
        let effective_order = rks.len(); // = d1s.len() + 1

        let rhos_c: Vec<f64> = if effective_order == 1 {
            vec![0.5]
        } else {
            let (r, b) = build_rb(&rks, h_phi_1, hh, b_h, 1, effective_order + 1)?;
            solve_linear(&r, &b)?
        };

        // corr = Σ ρ_k·D1_k + ρ_last·D1_t ; x_t = x_t_ − (α_t·B_h)·corr
        let mut corr = fscale(&d1_t, *rhos_c.last().unwrap())?;
        for (k_idx, d1) in d1s.iter().enumerate() {
            corr = add(&corr, &fscale(d1, rhos_c[k_idx])?)?;
        }
        Ok(subtract(&x_t_, &fscale(&corr, alpha_t * b_h)?)?)
    }
}

impl WanScheduler for FlowUniPC {
    fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = integer_timesteps(&self.sigmas, self.num_train_timesteps);
        self.sigmas_f64 = self.sigmas.iter().map(|&s| s as f64).collect();
        self.step_index = 0;
        self.num_steps = num_steps;
        self.lower_order_nums = 0;
        self.model_outputs = vec![None; self.solver_order];
        self.last_sample = None;
        self.this_order = 1;
    }
    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let i = self.step_index;
        ensure_step_in_range(i, self.sigmas_f64.len())?;
        // x0 = sample − σ[i]·v
        let x0 = subtract(sample, &fscale(model_output, self.sigmas_f64[i])?)?;

        // 1. Corrector (uses last step's `this_order`).
        let mut sample_cur = sample.clone();
        let use_corrector = self.use_corrector && i > 0 && self.last_sample.is_some();
        if use_corrector {
            let last = self.last_sample.as_ref().unwrap().clone();
            sample_cur = self.uni_c_bh2(&x0, &last, &sample_cur, self.this_order)?;
        }

        // 2. Shift x0 history (newest-last), append current x0.
        for k in 0..self.solver_order - 1 {
            self.model_outputs[k] = self.model_outputs[k + 1].clone();
        }
        let last_idx = self.solver_order - 1;
        self.model_outputs[last_idx] = Some(x0.clone());

        // 3. Prediction order.
        let this_order = if self.lower_order_final {
            self.solver_order.min(self.num_steps - i)
        } else {
            self.solver_order
        };
        self.this_order = this_order.min(self.lower_order_nums + 1);

        // 4. Predict next.
        self.last_sample = Some(sample_cur.clone());
        let x_next = self.uni_p_bh2(&x0, &sample_cur, self.this_order)?;

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(x_next)
    }
}

/// Build the (R, b) system for the UniPC ρ solve. `rks` are the rk values (Vandermonde base);
/// rows `j = 1..j_end` give `R_row = rks.^(j−1)` and `b = h_phi_k·factorial_i / B_h`, advancing the
/// factorial / `h_phi_k` recurrence in f64. The predictor passes `j_end = effective_order`; the
/// corrector passes `j_end = effective_order + 1`.
fn build_rb(
    rks: &[f64],
    h_phi_1: f64,
    hh: f64,
    b_h: f64,
    j_start: usize,
    j_end: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    // `hh = λ_{s0} − λ_t`; equal consecutive sigmas (a degenerate/single-step schedule) make it 0,
    // so `h_phi_1 / hh` and the `h_phi_k / hh` recurrence below become ±Inf/NaN and would silently
    // corrupt the predictor coefficients (solve_linear's near-zero-pivot guard only fires *after*
    // NaN has entered the matrix). Reject the degenerate step here instead (F-008).
    if hh.abs() < f64::EPSILON {
        return Err(Error::Msg(format!(
            "wan scheduler: degenerate schedule (equal consecutive sigmas, hh={hh:.3e}) in build_rb"
        )));
    }
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let mut factorial_i = 1.0_f64;
    let mut r_rows: Vec<Vec<f64>> = Vec::new();
    let mut b_vals: Vec<f64> = Vec::new();
    for j in j_start..j_end {
        r_rows.push(rks.iter().map(|&rk| rk.powi((j - 1) as i32)).collect());
        b_vals.push(h_phi_k * factorial_i / b_h);
        factorial_i *= (j + 1) as f64;
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
    }
    Ok((r_rows, b_vals))
}

/// Solve `R·x = b` for a small dense square system via Gaussian elimination with partial pivoting
/// (f64) — the host-side stand-in for `np.linalg.solve` (LAPACK LU). For the order-2 systems Wan
/// actually builds this matches LAPACK to f64 round-off. A (near-)singular system returns `Err`
/// rather than silently propagating NaN/Inf into the latents, mirroring how `np.linalg.solve`
/// raises `LinAlgError` (F-020); reachable only via a custom `solver_order > 2` with degenerate `rks`.
// Index loops read clearer than iterator adapters for a matrix solve.
#[allow(clippy::needless_range_loop)]
fn solve_linear(r: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    let mut a: Vec<Vec<f64>> = r.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        a.swap(col, piv);
        x.swap(col, piv);
        let diag = a[col][col];
        // The largest-magnitude pivot in this column is ~0 ⇒ the system is (near-)singular; dividing
        // by it (here and in back-substitution) would yield NaN/Inf that silently corrupts the
        // trajectory. Error instead, like `np.linalg.solve` (F-020).
        if diag.abs() < 1e-12 {
            return Err(Error::Msg(format!(
                "wan scheduler: singular {n}x{n} system in solve_linear (near-zero pivot \
                 {diag:.3e} at column {col}) — degenerate rks for the configured solver_order"
            )));
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            if factor != 0.0 {
                for c in col..n {
                    a[row][c] -= factor * a[col][c];
                }
                x[row] -= factor * x[col];
            }
        }
    }
    // Back-substitution (each `a[col][col]` is a pivot already checked non-zero above).
    for col in (0..n).rev() {
        let mut sum = x[col];
        for c in (col + 1)..n {
            sum -= a[col][c] * x[c];
        }
        x[col] = sum / a[col][col];
    }
    Ok(x)
}

// =====================================================================================
// Typed FamilyDriver for Wan Native Flow-SNR Solvers (SW-6968, PIPE-3, PIPE-7)
// =====================================================================================

/// Forward-evaluation seam for [`WanNativeDriver`] (`SW-6968`, `PIPE-7`).
///
/// Generic over the forward predictor so weights-free Tier 0 tests can run
/// [`mlx_gen::driver::conformance::verify_driver_conformance`] and expert-crossing `Carry`
/// trajectory parity checks over mocked forwards without loading any model weights.
pub trait WanNativeForward {
    /// Predict the flow velocity `v` for `latents` at absolute outer `step` and integer `timestep`.
    fn predict_velocity(&self, latents: &Array, step: usize, timestep: f32) -> Result<Array>;
}

impl<F> WanNativeForward for F
where
    F: Fn(&Array, usize, f32) -> Result<Array>,
{
    fn predict_velocity(&self, latents: &Array, step: usize, timestep: f32) -> Result<Array> {
        self(latents, step, timestep)
    }
}

/// Owner-local adapter lifting an `FnMut` predictor (e.g. `denoise_approx`'s mutable `TrunkCache`
/// / `TokenPruner`) into [`WanNativeForward`].
pub struct WanMutForward<F>(std::cell::RefCell<F>);

impl<F> WanMutForward<F>
where
    F: FnMut(&Array, usize, f32) -> Result<Array>,
{
    pub fn new(forward: F) -> Self {
        Self(std::cell::RefCell::new(forward))
    }
}

impl<F> WanNativeForward for WanMutForward<F>
where
    F: FnMut(&Array, usize, f32) -> Result<Array>,
{
    fn predict_velocity(&self, latents: &Array, step: usize, timestep: f32) -> Result<Array> {
        (self.0.borrow_mut())(latents, step, timestep)
    }
}

enum WanSolverState {
    Euler(FlowMatchEuler),
    Dpmpp2m(FlowDpmpp2m),
    UniPC(FlowUniPC),
}

impl WanSolverState {
    fn new_at_step(
        kind: SolverKind,
        num_train_timesteps: usize,
        num_steps: usize,
        shift: f32,
        start_step: usize,
    ) -> Self {
        match kind {
            SolverKind::Euler => {
                let mut s = FlowMatchEuler::new(num_train_timesteps);
                s.set_timesteps(num_steps, shift);
                s.step_index = start_step;
                Self::Euler(s)
            }
            SolverKind::Dpmpp2m => {
                let mut s = FlowDpmpp2m::new(num_train_timesteps);
                s.set_timesteps(num_steps, shift);
                s.step_index = start_step;
                Self::Dpmpp2m(s)
            }
            SolverKind::UniPC => {
                let mut s = FlowUniPC::new(num_train_timesteps, 2);
                s.set_timesteps(num_steps, shift);
                s.step_index = start_step;
                Self::UniPC(s)
            }
        }
    }

    fn step_index(&self) -> usize {
        match self {
            Self::Euler(s) => s.step_index,
            Self::Dpmpp2m(s) => s.step_index,
            Self::UniPC(s) => s.step_index,
        }
    }

    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        match self {
            Self::Euler(s) => s.step(model_output, sample),
            Self::Dpmpp2m(s) => s.step(model_output, sample),
            Self::UniPC(s) => s.step(model_output, sample),
        }
    }
}

/// Request-owned Wan native multistep solver history stored inside [`mlx_gen::driver::DriverRunState`].
///
/// Carried across compatible expert-crossing slices (`0..crossing` high expert -> `crossing..steps`
/// low expert, including resident MoE, staged expert swap, and VACE-Fun) via
/// [`mlx_gen::driver::DriverHistoryPolicy::Carry`], and cleanly refused if solver kind, train-step
/// convention, step count, or shifted sigma schedule disagrees (`PIPE-3`, `PIPE-7`).
struct WanSolverHistory {
    kind: SolverKind,
    num_train_timesteps: usize,
    total_steps: usize,
    sigmas_bits: Vec<u32>,
    expected_next_step: usize,
    solver: WanSolverState,
}

/// Typed [`mlx_gen::driver::FamilyDriver`] for Wan's native flow-SNR solvers (`UniPC`,
/// `Dpmpp2m`, `Euler`) across dense T2V, TI2V mask-blend, A14B MoE (resident + staged swap),
/// and VACE / VACE-Fun routes (`SW-6968`, `PIPE-3`, `PIPE-7`).
pub struct WanNativeDriver<F> {
    kind: SolverKind,
    num_train_timesteps: usize,
    num_steps: usize,
    shift: f32,
    sigmas: Vec<f32>,
    sigmas_bits: Vec<u32>,
    timesteps: Vec<f32>,
    forward: F,
    support: mlx_gen::driver::DriverSupport,
}

impl<F: WanNativeForward> WanNativeDriver<F> {
    fn build_support(driver_id: &'static str) -> mlx_gen::driver::DriverSupport {
        mlx_gen::driver::DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: mlx_gen::driver::DriverRefusalReasons {
                renoise_from_clean:
                    "native MLX Wan flow-SNR driver does not re-noise from clean estimates",
                geometry_change:
                    "native MLX Wan flow-SNR driver does not support geometry-changing phase boundaries",
                read_only_boundary_estimate:
                    "native MLX Wan flow-SNR driver does not expose read-only boundary clean estimates",
                ..mlx_gen::driver::DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(
        driver_id: &'static str,
        kind: SolverKind,
        num_train_timesteps: usize,
        num_steps: usize,
        shift: f32,
        forward: F,
    ) -> Self {
        let sigmas = compute_sigmas(num_steps, shift, num_train_timesteps);
        let timesteps = integer_timesteps(&sigmas, num_train_timesteps);
        let sigmas_bits = sigmas.iter().map(|s| s.to_bits()).collect();
        Self {
            kind,
            num_train_timesteps,
            num_steps,
            shift,
            sigmas,
            sigmas_bits,
            timesteps,
            forward,
            support: Self::build_support(driver_id),
        }
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
}

impl<F: WanNativeForward> mlx_gen::driver::FamilyDriver<Array> for WanNativeDriver<F> {
    fn support(&self) -> &mlx_gen::driver::DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: mlx_gen::driver::SliceInput<'_, Array>,
        state: &mut mlx_gen::driver::DriverRunState,
        hooks: &mut mlx_gen::driver::StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps != self.num_steps || input.span.end > self.num_steps {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Wan schedule length {}",
                self.support.driver_id, input.span, input.total_steps, self.num_steps
            )));
        }

        // Validate Carry compatibility BEFORE mutating `state` (`PIPE-3`).
        let carrying = input.transition == mlx_gen::driver::SliceTransition::Continue
            && input.history == mlx_gen::driver::DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<WanSolverHistory>() else {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry requested without prior Wan solver history in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.kind != self.kind {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible solver change ({:?} -> {:?})",
                    self.support.driver_id, existing.kind, self.kind
                )));
            }
            if existing.num_train_timesteps != self.num_train_timesteps
                || existing.total_steps != input.total_steps
                || existing.sigmas_bits != self.sigmas_bits
            {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible Wan schedule or train-timestep convention",
                    self.support.driver_id
                )));
            }
            if existing.expected_next_step != input.span.start
                || existing.solver.step_index() != input.span.start
            {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused at non-contiguous step {} (history at {})",
                    self.support.driver_id,
                    input.span.start,
                    existing.solver.step_index()
                )));
            }
        }

        state.enter_slice(&input)?;
        if !carrying {
            state.set_history(WanSolverHistory {
                kind: self.kind,
                num_train_timesteps: self.num_train_timesteps,
                total_steps: input.total_steps,
                sigmas_bits: self.sigmas_bits.clone(),
                expected_next_step: input.span.start,
                solver: WanSolverState::new_at_step(
                    self.kind,
                    self.num_train_timesteps,
                    self.num_steps,
                    self.shift,
                    input.span.start,
                ),
            });
        }

        let mut latents = match input.state {
            mlx_gen::gen_core::BoundaryState::XSigma(x)
            | mlx_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | mlx_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let info = mlx_gen::driver::DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: mlx_gen::driver::DriverEvaluationKind::Trajectory,
                sigma: self.sigmas[step],
                next_sigma: self.sigmas[step + 1],
            };
            hooks.observe(&info, &latents)?;
            let t = self.timesteps[step];
            let pred = self
                .forward
                .predict_velocity(&latents, step, t)
                .map_err(mlx_gen::gen_core::Error::from)?;
            state.record_evaluation(mlx_gen::driver::DriverEvaluationKind::Trajectory);
            latents = {
                let history = state
                    .history_mut::<WanSolverHistory>()
                    .expect("wan solver history initialized on slice entry");
                let next = history
                    .solver
                    .step(&pred, &latents)
                    .map_err(mlx_gen::gen_core::Error::from)?;
                history.expected_next_step = step + 1;
                next
            };
            latents = hooks.post_step(&info, latents)?;
            // Force per-step evaluation to bound MLX's lazy graph and keep cancellation responsive.
            mlx_rs::transforms::eval([&latents])
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            hooks.materialize(&info, &latents)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(mlx_gen::gen_core::BoundaryState::XSigma(latents))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::multiply;

    #[test]
    fn ensure_step_in_range_guards_over_and_under_run() {
        // F-019: a built schedule has num_steps+1 sigmas; a valid step needs step_index+1 < len.
        assert!(ensure_step_in_range(0, 0).is_err()); // empty (set_timesteps not called)
        assert!(ensure_step_in_range(0, 3).is_ok()); // 2-step schedule, first step
        assert!(ensure_step_in_range(1, 3).is_ok()); // last valid step
        assert!(ensure_step_in_range(2, 3).is_err()); // over-stepped
    }

    #[test]
    fn step_errors_instead_of_panicking_when_over_stepped() {
        // F-019: driving a scheduler past its step count returns a typed error, not an OOB panic.
        let mut sched = make_scheduler(SolverKind::Euler, 1000);
        sched.set_timesteps(2, 5.0);
        let x = Array::from_slice(&[0.5f32, -0.5], &[1, 2]);
        let v = Array::from_slice(&[0.1f32, 0.2], &[1, 2]);
        assert!(sched.step(&v, &x).is_ok()); // step 0
        assert!(sched.step(&v, &x).is_ok()); // step 1 (last)
        let err = sched.step(&v, &x).unwrap_err().to_string(); // step 2 → over-stepped
        assert!(err.contains("over-stepped"), "got: {err}");

        // Stepping before set_timesteps (empty schedule) also errors rather than panicking.
        let mut fresh = make_scheduler(SolverKind::Euler, 1000);
        assert!(fresh.step(&v, &x).is_err());
    }

    #[test]
    fn from_name_maps_advertised_samplers_and_defaults_to_unipc() {
        // F-021 / sc-7296: the single sampler→solver mapping the production model entries share. The
        // advertised set (validate_request) is now the curated `uni_pc`/`euler`/`dpmpp_2m`, with the
        // legacy `unipc`/`dpmpp2m` spellings kept as aliases; an unset/unknown name → UniPC.
        assert_eq!(SolverKind::from_name("euler"), SolverKind::Euler);
        // Curated gen-core vocabulary (the advertised names since sc-7296).
        assert_eq!(SolverKind::from_name("uni_pc"), SolverKind::UniPC);
        assert_eq!(SolverKind::from_name("dpmpp_2m"), SolverKind::Dpmpp2m);
        // Legacy spellings stay valid (old recipes).
        assert_eq!(SolverKind::from_name("unipc"), SolverKind::UniPC);
        assert_eq!(SolverKind::from_name("dpmpp2m"), SolverKind::Dpmpp2m);
        assert_eq!(SolverKind::from_name("dpm++"), SolverKind::Dpmpp2m);
        assert_eq!(SolverKind::from_name(""), SolverKind::UniPC); // the unset-sampler sentinel
        assert_eq!(SolverKind::from_name("nope"), SolverKind::UniPC);
        // Case-insensitive (the model entries lower-case via the same path).
        assert_eq!(SolverKind::from_name("Euler"), SolverKind::Euler);
        assert_eq!(SolverKind::from_name("UNI_PC"), SolverKind::UniPC);
        assert_eq!(SolverKind::from_name("DPMPP_2M"), SolverKind::Dpmpp2m);
    }

    #[test]
    fn sigma_schedule_endpoints_and_shape() {
        let s = compute_sigmas(40, 5.0, 1000);
        assert_eq!(s.len(), 41);
        assert_eq!(*s.last().unwrap(), 0.0);
        // First sigma = shift·σ_max/(1+(shift−1)·σ_max), σ_max = 0.999, shift = 5.
        let sm = 0.999_f64;
        let want0 = (5.0 * sm / (1.0 + 4.0 * sm)) as f32;
        assert!((s[0] - want0).abs() < 1e-6, "s0 {} want {}", s[0], want0);
        // Strictly decreasing.
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn integer_timesteps_truncate() {
        let s = compute_sigmas(40, 5.0, 1000);
        let t = integer_timesteps(&s, 1000);
        assert_eq!(t.len(), 40);
        // Each timestep is trunc(sigma*1000) and integer-valued.
        for (ts, sig) in t.iter().zip(s.iter()) {
            assert_eq!(*ts, (sig * 1000.0).trunc());
            assert_eq!(*ts, ts.trunc());
        }
        // First timestep with shift=5, σ0≈0.9998 → ~999.
        assert!(t[0] >= 995.0 && t[0] <= 999.0, "t0 = {}", t[0]);
    }

    #[test]
    fn euler_step_matches_formula() {
        let mut sched = FlowMatchEuler::new(1000);
        sched.set_timesteps(4, 5.0);
        let sample = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let v = Array::from_slice(&[0.5_f32, 0.5, 0.5], &[3]);
        let out = sched.step(&v, &sample).unwrap();
        let dt = sched.sigmas[1] - sched.sigmas[0];
        let got = out.as_slice::<f32>();
        assert!((got[0] - (1.0 + dt * 0.5)).abs() < 1e-6);
        assert!((got[2] - (3.0 + dt * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn solve_linear_2x2() {
        // [[2,1],[1,3]] x = [3,5] → x = [0.8, 1.4].
        let r = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![3.0, 5.0];
        let x = solve_linear(&r, &b).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn solve_linear_singular_errs() {
        // F-020: a singular system (row 2 = 2 × row 1) errors instead of returning NaN/Inf.
        let r = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        match solve_linear(&r, &b) {
            Ok(x) => panic!("singular system must error, got {x:?}"),
            Err(e) => assert!(e.to_string().contains("singular"), "got: {e}"),
        }
    }

    #[test]
    fn unipc_runs_full_trajectory() {
        // Smoke: UniPC default order-2 over 8 steps with constant velocity does not panic and stays
        // finite (exercises corrector + 2×2 solve path).
        let mut sched = FlowUniPC::new(1000, 2);
        sched.set_timesteps(8, 5.0);
        let mut x = Array::from_slice(&[0.3_f32; 16], &[16]);
        for _ in 0..8 {
            let v = multiply(&x, Array::from_f32(0.1)).unwrap();
            x = sched.step(&v, &x).unwrap();
        }
        let xs = x.as_slice::<f32>();
        assert!(xs.iter().all(|v| v.is_finite()));
    }

    fn mlx_bit_equal(a: &Array, b: &Array) -> bool {
        mlx_rs::transforms::eval([a, b]).unwrap();
        let sa = a.as_slice::<f32>();
        let sb = b.as_slice::<f32>();
        sa.len() == sb.len() && sa.iter().zip(sb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn mock_expert_velocity(
        x: &Array,
        step: usize,
        timestep: f32,
        slope: f32,
        bias: f32,
    ) -> Result<Array> {
        // Deterministic, weights-free non-trivial velocity depending on latent, step, and timestep
        // so multistep UniPC history (`model_outputs`, `last_sample`, `this_order`) is exercised.
        let scale = slope + 0.00025 * timestep + 0.01 * (step as f32);
        let offset = bias - 0.0001 * timestep;
        Ok(add(
            &multiply(x, Array::from_f32(scale))?,
            Array::from_f32(offset),
        )?)
    }

    #[test]
    fn wan_native_driver_passes_conformance() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        for kind in [SolverKind::UniPC, SolverKind::Dpmpp2m, SolverKind::Euler] {
            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 6,
                split_at: 3,
                seed: 42,
                base_geometry: (64, 64),
                alt_geometry: (128, 128),
                make_driver: move || {
                    Box::new(WanNativeDriver::new(
                        "wan_native_mlx_test",
                        kind,
                        1000,
                        6,
                        5.0,
                        |x: &Array, step: usize, t: f32| {
                            mock_expert_velocity(x, step, t, 0.35, -0.1)
                        },
                    ))
                },
                make_initial: || {
                    Array::from_slice(&[0.4_f32, -0.8, 1.2, -0.3, 0.7, 0.1, -1.1, 0.9], &[2, 2, 2])
                },
                latents_equal: mlx_bit_equal,
            })
            .unwrap_or_else(|e| panic!("conformance failed for {kind:?}: {e}"));
            assert!(report.verified_inert_split, "{kind:?}");
            assert!(report.verified_reset_preserves_draws, "{kind:?}");
            assert!(report.verified_cancellation, "{kind:?}");
        }
    }

    #[test]
    fn wan_native_unipc_expert_crossing_carry_matches_legacy_bit_for_bit() {
        use mlx_gen::driver::{
            DriverHistoryPolicy, DriverRunState, FamilyDriver, SliceInput, SliceTransition,
            StepHooks,
        };
        use mlx_gen::gen_core::{BoundaryState, CancelFlag, Progress};

        let steps = 8usize;
        let shift = 5.0_f32;
        let num_train = 1000usize;
        let init = Array::from_slice(
            &[0.25_f32, -0.75, 1.1, -0.4, 0.6, -0.2, 0.9, -1.3],
            &[2, 2, 2],
        );
        let cancel = CancelFlag::default();

        // Legacy single-scheduler reference across a high->low expert crossing.
        let mut legacy_sched = FlowUniPC::new(num_train, 2);
        legacy_sched.set_timesteps(steps, shift);
        let timesteps = legacy_sched.timesteps().to_vec();
        let crossing = timesteps.iter().position(|&t| t < 875.0).unwrap_or(steps);
        assert!(crossing > 0 && crossing < steps, "crossing={crossing}");

        let mut legacy_latents = init.clone();
        for (i, &t) in timesteps.iter().enumerate() {
            let v = if i < crossing {
                mock_expert_velocity(&legacy_latents, i, t, 0.45, 0.12).unwrap()
            } else {
                mock_expert_velocity(&legacy_latents, i, t, 0.22, -0.07).unwrap()
            };
            legacy_latents = legacy_sched.step(&v, &legacy_latents).unwrap();
            mlx_rs::transforms::eval([&legacy_latents]).unwrap();
        }

        // Typed FamilyDriver with two distinct expert drivers sharing one DriverRunState via Carry.
        let high_driver = WanNativeDriver::new(
            "wan_moe_high",
            SolverKind::UniPC,
            num_train,
            steps,
            shift,
            |x: &Array, step: usize, t: f32| mock_expert_velocity(x, step, t, 0.45, 0.12),
        );
        let low_driver = WanNativeDriver::new(
            "wan_moe_low",
            SolverKind::UniPC,
            num_train,
            steps,
            shift,
            |x: &Array, step: usize, t: f32| mock_expert_velocity(x, step, t, 0.22, -0.07),
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
            mlx_bit_equal(&carried_latents, &legacy_latents),
            "Carry across expert crossing must match legacy single-scheduler UniPC bit-for-bit"
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
    fn wan_native_carry_refuses_incompatible_and_reset_clears_history_independently_of_rng() {
        use mlx_gen::driver::{
            DriverHistoryPolicy, DriverRunState, FamilyDriver, SliceInput, SliceTransition,
            StepHooks,
        };
        use mlx_gen::gen_core::{BoundaryState, CancelFlag, Error as CoreError, Progress};

        let steps = 6usize;
        let split = 3usize;
        let cancel = CancelFlag::default();
        let init = Array::from_slice(&[0.5_f32, -0.5, 0.9, -0.2], &[4]);

        let unipc = WanNativeDriver::new(
            "wan_unipc",
            SolverKind::UniPC,
            1000,
            steps,
            5.0,
            |x: &Array, step: usize, t: f32| mock_expert_velocity(x, step, t, 0.4, 0.1),
        );
        let dpmpp = WanNativeDriver::new(
            "wan_dpmpp",
            SolverKind::Dpmpp2m,
            1000,
            steps,
            5.0,
            |x: &Array, step: usize, t: f32| mock_expert_velocity(x, step, t, 0.4, 0.1),
        );
        let diff_shift = WanNativeDriver::new(
            "wan_shift",
            SolverKind::UniPC,
            1000,
            steps,
            3.0,
            |x: &Array, step: usize, t: f32| mock_expert_velocity(x, step, t, 0.4, 0.1),
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
        let BoundaryState::XSigma(mid_arr) = mid else {
            panic!("expected XSigma");
        };

        // Incompatible solver (UniPC -> Dpmpp2m) refuses Carry.
        {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let res = dpmpp.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: split..steps,
                    total_steps: steps,
                    state: BoundaryState::XSigma(mid_arr.clone()),
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

        // Incompatible shift (5.0 -> 3.0) refuses Carry.
        {
            let mut noop = |_: Progress| {};
            let mut hooks = StepHooks::new(&mut noop);
            let res = diff_shift.run_slice(
                SliceInput {
                    pass: 0,
                    phase: 1,
                    span: split..steps,
                    total_steps: steps,
                    state: BoundaryState::XSigma(mid_arr.clone()),
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
        // and diverges from Carry (proving history was actually cleared).
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
                        state: BoundaryState::XSigma(mid_arr),
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
        let (BoundaryState::XSigma(c_arr), BoundaryState::XSigma(r_arr)) = (carried, reset_out)
        else {
            panic!("expected XSigma");
        };
        assert!(
            !mlx_bit_equal(&c_arr, &r_arr),
            "Reset must actually clear UniPC multistep history"
        );
    }

    #[test]
    fn wan_native_driver_post_step_hook_applies_ti2v_blend() {
        use mlx_gen::driver::{DriverRunState, FamilyDriver, SliceInput, StepHooks};
        use mlx_gen::gen_core::{BoundaryState, CancelFlag, Progress};

        let steps = 4usize;
        let cancel = CancelFlag::default();
        let mask = Array::from_slice(&[0.0_f32, 0.0, 1.0, 1.0], &[4]);
        let frozen = Array::from_slice(&[3.0_f32, -2.0, 0.0, 0.0], &[4]);
        let init = Array::from_slice(&[3.0_f32, -2.0, 0.5, -0.5], &[4]);
        let driver = WanNativeDriver::new(
            "wan_ti2v_hook",
            SolverKind::UniPC,
            1000,
            steps,
            5.0,
            |x: &Array, step: usize, t: f32| mock_expert_velocity(x, step, t, 0.3, 0.05),
        );
        let mut state = DriverRunState::new();
        let mut noop = |_: Progress| {};
        let mut blend = |_info: &mlx_gen::driver::DriverStepInfo, latents: Array| {
            let masked = multiply(&mask, &latents)
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            add(&frozen, &masked).map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))
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
        let BoundaryState::XSigma(arr) = out else {
            panic!("expected XSigma");
        };
        mlx_rs::transforms::eval([&arr]).unwrap();
        let vals = arr.as_slice::<f32>();
        assert_eq!(vals[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(vals[1].to_bits(), (-2.0_f32).to_bits());
        assert!(vals[2].is_finite() && vals[3].is_finite());
    }
}
