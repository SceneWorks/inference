//! Krea Realtime 14B **Self-Forcing few-step** flow-matching scheduler (sc-8437, S4).
//!
//! Despite the config's UniPC-flavoured naming, Krea Realtime's per-block denoise is **not** UniPC
//! (the S1 audit): it is the bespoke few-step flow-matching renoise loop **adapted from** the reference
//! (`krea-ai/realtime-video utils/scheduler.py` + `before_denoise.py::WanRTSetTimestepsStep` +
//! `denoise.py`, mirrored by Self-Forcing's `FlowMatchScheduler`). A short `denoising_step_list`
//! (`[1000, 937, 833, 625, 0]` for the shipped 14B) is run over one autoregressive chunk; each step
//! takes an Euler `x0` estimate and renoises to the next step, with **CFG off** (a single batch-1
//! forward per step).
//!
//! The schedule itself is the same **algebraic time-shift** the base Wan sampler uses
//! ([`mlx_gen_wan::compute_sigmas`]), built here as the reference's full-resolution table so a
//! `denoising_step_list` timestep maps to a sigma exactly as the reference does:
//!
//! ```text
//! sigmas    = linspace(1.0, 0.0, 1001)[:-1]                  # 1000 unshifted, 1.0 … 0.001
//! sigmas    = shift * sigmas / (1 + (shift - 1) * sigmas)    # σ' = shift·σ / (1 + (shift-1)·σ)
//! timesteps = sigmas * 1000                                  # the model-timestep table
//! # sigma_at_timestep(t) = sigmas[argmin(|timesteps - t|)]   # the reference's argmin lookup
//! x0        = latents - sigma_t * noise_pred                 # Euler x0 estimate (velocity = noise_pred)
//! renoise   = (1 - sigma_next) * x0 + sigma_next * randn     # to the next step's noise level
//! ```
//!
//! Note the terminal `0` maps (via argmin over the `[:-1]`-truncated table) to the smallest tabled
//! sigma `≈ 0.00498`, **not** exactly `0` — this is the reference's behaviour, not a bug: the final
//! `x0` is `latents - 0.00498·noise_pred` (essentially the clean latent). The coefficient math runs
//! in `f64` (the reference's Python-float / numpy path); only the final tensor combinations run in the
//! latent dtype (f32), mirroring [`mlx_gen_wan::scheduler`].
//!
//! The sigma schedule, the time-shift, and the Euler/renoise formulas are **re-derived** published
//! flow-matching algebra (mathematical facts), written independently in Rust and validated against
//! hand-computed literals — an architecture/formula reimplementation, **not** a line-for-line
//! transcription of the reference's source structure.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// Flow-matching training horizon (the reference's `num_train_timesteps`) — the resolution of the
/// full sigma/timestep table a `denoising_step_list` timestep is looked up against.
pub const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// Multiply a latent-dtype array by an `f64` scalar (rounded to the array's dtype), mirroring
/// [`mlx_gen_wan::scheduler`]'s `fscale` — the weak scalar adopts the array dtype.
fn fscale(a: &Array, c: f64) -> Result<Array> {
    Ok(multiply(a, Array::from_f32(c as f32))?)
}

/// The Euler flow-matching `x0` estimate: `x0 = latents − σ_t · v`, where `v` is the model velocity
/// (`noise_pred`) and `σ_t` is the current step's sigma. Exposed as a pure, testable unit.
pub fn euler_x0(latents: &Array, velocity: &Array, sigma_t: f64) -> Result<Array> {
    Ok(subtract(latents, &fscale(velocity, sigma_t)?)?)
}

/// Renoise an `x0` estimate to the next step's noise level:
/// `sample = (1 − σ_next) · x0 + σ_next · ε`, with `ε` fresh Gaussian noise. Exposed as a pure,
/// testable unit. A swapped interpolation coefficient (`σ_next · x0 + (1−σ_next) · ε`) is a distinct
/// result the unit test discriminates.
pub fn renoise_step(x0: &Array, eps: &Array, sigma_next: f64) -> Result<Array> {
    let kept = fscale(x0, 1.0 - sigma_next)?;
    let added = fscale(eps, sigma_next)?;
    Ok(add(&kept, &added)?)
}

/// Build the full shifted-sigma table for `shift` at `num_train` resolution: unshifted
/// `linspace(1.0, 0.0, num_train + 1)[:-1]` (`u_i = 1 − i/num_train`, `i ∈ 0..num_train`), the algebraic
/// time-shift `σ' = shift·σ / (1 + (shift − 1)·σ)`, and `timesteps = σ' · num_train`. Returns
/// `(sigmas, timesteps)`, both length `num_train`, monotonically decreasing (`sigmas[0] = 1.0`,
/// `timesteps[0] = num_train`).
fn full_shifted_table(shift: f64, num_train: usize) -> (Vec<f64>, Vec<f64>) {
    let n = num_train as f64;
    let mut sigmas = Vec::with_capacity(num_train);
    let mut timesteps = Vec::with_capacity(num_train);
    for i in 0..num_train {
        let u = 1.0 - (i as f64) / n; // linspace(1.0, 0.0, num_train+1)[:-1][i]
        let s = shift * u / (1.0 + (shift - 1.0) * u);
        sigmas.push(s);
        timesteps.push(s * n);
    }
    (sigmas, timesteps)
}

/// The v2v **denoise-strength** step timesteps — the port of `v2v.py::get_denoising_schedule`:
///
/// ```text
/// raw  = linspace(strength·1000, 0, steps).to(int64)          # descending, ends at 0 (steps ≥ 2)
/// list = zero_padded_timesteps[1000 − raw]                    # re-index the warped model-timestep table
/// ```
///
/// where `warped_timesteps` is the full shifted model-timestep table (`sigmas·1000`, length
/// [`NUM_TRAIN_TIMESTEPS`], `timesteps[0] = 1000`) and `zero_padded_timesteps` is that table with a
/// terminal `0` appended at index [`NUM_TRAIN_TIMESTEPS`] (the reference's
/// `torch.cat((scheduler.timesteps, [0]))`). The returned per-step timesteps are truncated to integers
/// (the reference stores `int64` timesteps), descending, with a terminal `0` for `steps ≥ 2`.
///
/// `strength = 1` reproduces the shipped few-step list (`[1000, 937, 833, 625, 0]` at `steps = 5`);
/// `strength → 0` collapses the whole list toward `0` (keep the source clip — a 0-strength v2v renoises
/// the source only to the smallest tabled sigma and denoises at `t = 0`, so the output tracks the
/// source). Mirrors the reference so a lower strength preserves more of the VAE-encoded source.
fn strength_step_timesteps(warped_timesteps: &[f64], strength: f64, steps: usize) -> Vec<f64> {
    let n = steps.max(1);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // linspace(strength·1000, 0, n)[i]: descending from strength·1000 to 0.
        let frac = if n == 1 {
            0.0
        } else {
            i as f64 / (n - 1) as f64
        };
        let raw = (strength * NUM_TRAIN_TIMESTEPS as f64 * (1.0 - frac)).trunc() as i64;
        // 1000 − raw indexes the zero-padded warped table (index NUM_TRAIN_TIMESTEPS ⇒ the appended 0).
        let idx = (NUM_TRAIN_TIMESTEPS as i64 - raw).clamp(0, NUM_TRAIN_TIMESTEPS as i64) as usize;
        let t = warped_timesteps.get(idx).copied().unwrap_or(0.0); // idx == NUM_TRAIN_TIMESTEPS ⇒ 0
        out.push(t.trunc());
    }
    out
}

/// Resolve the per-step denoising timesteps (descending, terminal `0`).
///
/// * `None` → the config's `default_list` verbatim (`[1000, 937, 833, 625, 0]` for the shipped 14B).
/// * `Some(n)` → an `n`-entry list rebuilt from the same shifted schedule: the first `n − 1`
///   shifted timesteps of `linspace(1.0, 0.0, n)[:-1]` (truncated to integers, as the reference stores
///   `int64` timesteps) followed by the terminal `0`. `Some(5)` reproduces the shipped list exactly
///   (`linspace(1,0,5)[:-1] = [1, .75, .5, .25]` → `[1000, 937, 833, 625]` → `+ [0]`).
fn resolve_step_timesteps(
    shift: f64,
    default_list: &[u32],
    steps: Option<usize>,
) -> Result<Vec<f64>> {
    match steps {
        None => Ok(default_list.iter().map(|&t| t as f64).collect()),
        Some(0) => Err(Error::Msg(
            "krea few-step: steps override must be >= 1".into(),
        )),
        Some(n) => {
            let mut out = Vec::with_capacity(n);
            if n >= 2 {
                let denom = (n - 1) as f64; // linspace(1.0, 0.0, n) step base
                for i in 0..(n - 1) {
                    let u = 1.0 - (i as f64) / denom;
                    let s = shift * u / (1.0 + (shift - 1.0) * u);
                    out.push((s * NUM_TRAIN_TIMESTEPS as f64).trunc());
                }
            }
            out.push(0.0); // terminal clean step
            Ok(out)
        }
    }
}

/// The Self-Forcing few-step schedule for one autoregressive chunk: the full shifted-sigma/timestep
/// table (for the timestep → sigma lookup) plus the resolved per-step `denoising_step_list`.
///
/// Construct with [`FewStepSchedule::new`]; drive a chunk by iterating [`step_timesteps`](Self::step_timesteps),
/// taking [`euler_x0`] with [`sigma_at_timestep`](Self::sigma_at_timestep)`(t)` each step and
/// [`renoise_step`] to `sigma_at_timestep(next_t)` between steps.
#[derive(Clone, Debug)]
pub struct FewStepSchedule {
    shift: f64,
    /// Full shifted-sigma table, `sigmas[i]` for `u_i = 1 − i/1000`. Length [`NUM_TRAIN_TIMESTEPS`].
    sigmas: Vec<f64>,
    /// `timesteps[i] = sigmas[i] · 1000`. Length [`NUM_TRAIN_TIMESTEPS`].
    timesteps: Vec<f64>,
    /// The resolved per-step model timesteps (descending, terminal `0`).
    step_timesteps: Vec<f64>,
}

impl FewStepSchedule {
    /// Build the schedule for flow-match `shift` (5.0 for Krea Realtime) and resolve the per-step
    /// denoising timesteps: `steps = None` uses `default_list` (the config's `denoising_step_list`)
    /// verbatim; `steps = Some(n)` rebuilds an `n`-entry list from the same shifted schedule
    /// (`Some(5)` reproduces the shipped `[1000, 937, 833, 625, 0]`). See
    /// [`step_timesteps`](Self::step_timesteps).
    pub fn new(shift: f64, default_list: &[u32], steps: Option<usize>) -> Result<Self> {
        if shift <= 0.0 || shift.is_nan() {
            return Err(Error::Msg(format!(
                "krea few-step: timestep_shift must be > 0, got {shift}"
            )));
        }
        let (sigmas, timesteps) = full_shifted_table(shift, NUM_TRAIN_TIMESTEPS);
        let step_timesteps = resolve_step_timesteps(shift, default_list, steps)?;
        Ok(Self {
            shift,
            sigmas,
            timesteps,
            step_timesteps,
        })
    }

    /// The v2v **denoise-strength** schedule (sc-8440 S7) — the reference `v2v.py::get_denoising_schedule`.
    /// Builds the same shifted-sigma table as [`new`](Self::new) but resolves the per-step timesteps so
    /// the schedule's **max** timestep is `strength·1000` (warped), descending to `0` over `steps`
    /// forwards. `strength = 1` reproduces the shipped few-step list; `strength → 0` collapses the list
    /// toward `0` (keep the source). The v2v init noise level is
    /// [`sigma_at_timestep`](Self::sigma_at_timestep) of the first (max) step timestep, so a lower
    /// strength renoises the VAE-encoded source less and preserves more of it. `strength` must be finite
    /// and in `[0, 1]`; `steps ≥ 1`.
    pub fn for_strength(shift: f64, strength: f64, steps: usize) -> Result<Self> {
        if shift <= 0.0 || shift.is_nan() {
            return Err(Error::Msg(format!(
                "krea few-step: timestep_shift must be > 0, got {shift}"
            )));
        }
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(Error::Msg(format!(
                "krea v2v: denoise strength must be finite in [0, 1], got {strength}"
            )));
        }
        if steps == 0 {
            return Err(Error::Msg("krea v2v: denoise steps must be >= 1".into()));
        }
        let (sigmas, timesteps) = full_shifted_table(shift, NUM_TRAIN_TIMESTEPS);
        let step_timesteps = strength_step_timesteps(&timesteps, strength, steps);
        Ok(Self {
            shift,
            sigmas,
            timesteps,
            step_timesteps,
        })
    }

    /// The per-step model timesteps to run (descending, terminal `0`), one forward per entry.
    pub fn step_timesteps(&self) -> &[f64] {
        &self.step_timesteps
    }

    /// Number of denoising forwards per chunk (`= step_timesteps().len()`).
    pub fn num_steps(&self) -> usize {
        self.step_timesteps.len()
    }

    /// The flow-match shift this schedule was built with.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// The full shifted-sigma table (length [`NUM_TRAIN_TIMESTEPS`]).
    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// The full model-timestep table `sigmas · 1000` (length [`NUM_TRAIN_TIMESTEPS`]).
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// The sigma for a model timestep `t`, via the reference's `argmin(|timesteps − t|)` lookup over
    /// the full table. `t = 1000` → `1.0` (exact); `t = 0` → the smallest tabled sigma `≈ 0.00498`
    /// (the `[:-1]` table has no exact `0`).
    pub fn sigma_at_timestep(&self, t: f64) -> f64 {
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (i, &ts) in self.timesteps.iter().enumerate() {
            let d = (ts - t).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        self.sigmas[best_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// The full table's endpoints + a hand-computed interior entry (independent literals — a wrong
    /// shift formula fails). `timesteps[i] = sigmas[i]·1000`.
    #[test]
    fn full_table_shift5_matches_hand_computed() {
        let (sigmas, timesteps) = full_shifted_table(5.0, NUM_TRAIN_TIMESTEPS);
        assert_eq!(sigmas.len(), 1000);
        assert_eq!(timesteps.len(), 1000);

        // i=0: u=1.0 → σ = 5·1/(1+4·1) = 1.0 → t = 1000.
        assert!((sigmas[0] - 1.0).abs() < 1e-12, "sigma[0] = {}", sigmas[0]);
        assert!((timesteps[0] - 1000.0).abs() < 1e-9);

        // i=1: u=0.999 → σ = 5·0.999/(1+4·0.999) = 4.995/4.996.
        let want1 = 5.0 * 0.999 / (1.0 + 4.0 * 0.999);
        assert!(
            (sigmas[1] - want1).abs() < 1e-12,
            "sigma[1] = {}",
            sigmas[1]
        );
        assert!((timesteps[1] - want1 * 1000.0).abs() < 1e-9);

        // i=999 (last): u=0.001 → σ = 5·0.001/(1+4·0.001) = 0.005/1.004 ≈ 0.00498.
        let want_last = 5.0 * 0.001 / (1.0 + 4.0 * 0.001);
        assert!(
            (sigmas[999] - want_last).abs() < 1e-12,
            "sigma[999] = {}",
            sigmas[999]
        );
        assert!((want_last - 0.004_980_079_68).abs() < 1e-9); // pin the literal

        // Strictly decreasing.
        assert!(sigmas.windows(2).all(|w| w[0] > w[1]));
        assert!(timesteps.windows(2).all(|w| w[0] > w[1]));
    }

    /// `sigma_at_timestep` for the shipped `[1000, 937, 833, 625, 0]` schedule against hand-derived
    /// table entries. The integer timesteps sit ≈ on tabled entries, so their sigma ≈ t/1000, except
    /// the terminal `0` → the smallest tabled sigma. A wrong shift or a broken argmin fails.
    #[test]
    fn sigma_at_timestep_shift5_matches_hand_computed() {
        let s = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();

        // t=1000 sits exactly on the table head → σ = 1.0.
        assert!((s.sigma_at_timestep(1000.0) - 1.0).abs() < 1e-12);

        // t=937: the closest unshifted linspace entry is u=0.748 (i=252, timestep 936.87 vs u=0.749's
        // 937.19). σ = 5·0.748/(1+4·0.748).
        let want937 = 5.0 * 0.748 / (1.0 + 4.0 * 0.748);
        assert!(
            (s.sigma_at_timestep(937.0) - want937).abs() < 1e-9,
            "σ(937) = {} want {}",
            s.sigma_at_timestep(937.0),
            want937
        );
        // …and it is within one table step of the naive t/1000.
        assert!((s.sigma_at_timestep(937.0) - 0.937).abs() < 1e-3);

        // t=625 sits on u=0.25 → σ = 5·0.25/(1+4·0.25) = 1.25/2 = 0.625 (exact tabled entry).
        assert!((s.sigma_at_timestep(625.0) - 0.625).abs() < 1e-6);

        // Terminal t=0 → the smallest tabled sigma (u=0.001), NOT 0.
        let want0 = 5.0 * 0.001 / (1.0 + 4.0 * 0.001);
        assert!((s.sigma_at_timestep(0.0) - want0).abs() < 1e-9);
        assert!(s.sigma_at_timestep(0.0) > 0.0);
    }

    /// The default `denoising_step_list` (None) and the `Some(5)` rebuild both yield the shipped list.
    #[test]
    fn step_timesteps_default_and_override_agree() {
        let default = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        assert_eq!(
            default.step_timesteps(),
            &[1000.0, 937.0, 833.0, 625.0, 0.0]
        );
        assert_eq!(default.num_steps(), 5);

        // Some(5) reconstructs the schedule from scratch → identical to the shipped list.
        let rebuilt = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], Some(5)).unwrap();
        assert_eq!(
            rebuilt.step_timesteps(),
            &[1000.0, 937.0, 833.0, 625.0, 0.0],
            "Some(5) must reproduce the shipped few-step list"
        );

        // A different override count → a different-length list, still descending with a terminal 0.
        let three = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], Some(3)).unwrap();
        assert_eq!(three.num_steps(), 3);
        assert_eq!(*three.step_timesteps().last().unwrap(), 0.0);
        assert!(three.step_timesteps()[0] > three.step_timesteps()[1]);
        // Some(3): linspace(1,0,3)[:-1] = [1.0, 0.5] → [1000, 833] → + [0].
        assert_eq!(three.step_timesteps(), &[1000.0, 833.0, 0.0]);

        assert!(FewStepSchedule::new(5.0, &[1000, 0], Some(0)).is_err());
    }

    /// The Euler `x0` step on a small tensor with hand-derived expected values.
    #[test]
    fn euler_x0_matches_formula() {
        // x0 = latents − σ·v.  σ = 0.5, latents = [1, 2, 3], v = [0.4, 0.2, 1.0].
        let latents = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let v = Array::from_slice(&[0.4f32, 0.2, 1.0], &[3]);
        let out = euler_x0(&latents, &v, 0.5).unwrap();
        let got = out.as_slice::<f32>();
        // [1 − .5·.4, 2 − .5·.2, 3 − .5·1] = [0.8, 1.9, 2.5].
        assert!((got[0] - 0.8).abs() < 1e-6, "{got:?}");
        assert!((got[1] - 1.9).abs() < 1e-6, "{got:?}");
        assert!((got[2] - 2.5).abs() < 1e-6, "{got:?}");
    }

    /// The renoise step on a small tensor with hand-derived expected values; a swapped coefficient
    /// would give the wrong answer.
    #[test]
    fn renoise_step_matches_formula() {
        // sample = (1 − σ)·x0 + σ·ε.  σ = 0.25, x0 = [4, 8], ε = [0, 4].
        let x0 = Array::from_slice(&[4.0f32, 8.0], &[2]);
        let eps = Array::from_slice(&[0.0f32, 4.0], &[2]);
        let out = renoise_step(&x0, &eps, 0.25).unwrap();
        let got = out.as_slice::<f32>();
        // [.75·4 + .25·0, .75·8 + .25·4] = [3.0, 7.0].
        assert!((got[0] - 3.0).abs() < 1e-6, "{got:?}");
        assert!((got[1] - 7.0).abs() < 1e-6, "{got:?}");
        // The swapped coefficient would give [.25·4 = 1.0, .25·8 + .75·4 = 5.0] — distinct.
        assert!((got[0] - 1.0).abs() > 1e-3);
    }

    /// v2v strength schedule (`for_strength`): `strength = 1` reproduces the shipped few-step list
    /// exactly (`get_denoising_schedule(warped, 1.0, 5) == [1000, 937, 833, 625, 0]`), so a full-strength
    /// v2v is identical to the t2v denoise from max noise.
    #[test]
    fn strength_one_reproduces_shipped_list() {
        let s = FewStepSchedule::for_strength(5.0, 1.0, 5).unwrap();
        assert_eq!(
            s.step_timesteps(),
            &[1000.0, 937.0, 833.0, 625.0, 0.0],
            "strength=1 reproduces the shipped few-step list (get_denoising_schedule at strength 1)"
        );
    }

    /// The strength lever moves the schedule's **max** timestep (and thus the init noise level): a lower
    /// strength starts closer to clean and ends at `0`, so more of the source survives. `strength = 0`
    /// collapses the whole list to `0` (keep the source); the init sigma is the smallest tabled sigma.
    #[test]
    fn strength_scales_the_schedule_head_and_init_sigma() {
        let full = FewStepSchedule::for_strength(5.0, 1.0, 5).unwrap();
        let half = FewStepSchedule::for_strength(5.0, 0.5, 5).unwrap();
        let zero = FewStepSchedule::for_strength(5.0, 0.0, 5).unwrap();

        // The first (max) step timestep tracks strength·1000 (warped): 1 → 1000, 0.5 → ~833, 0 → 0.
        assert_eq!(full.step_timesteps()[0], 1000.0);
        // strength 0.5: raw[0] = 500 → 1000−500 = 500 → warped timesteps[500] = 833.
        assert_eq!(half.step_timesteps()[0], 833.0);
        assert_eq!(zero.step_timesteps(), &[0.0, 0.0, 0.0, 0.0, 0.0]);

        // The init noise level (sigma at the max step) shrinks with strength.
        let sig_full = full.sigma_at_timestep(full.step_timesteps()[0]);
        let sig_half = half.sigma_at_timestep(half.step_timesteps()[0]);
        let sig_zero = zero.sigma_at_timestep(zero.step_timesteps()[0]);
        assert!(
            (sig_full - 1.0).abs() < 1e-9,
            "strength 1 ⇒ init sigma 1.0 (pure noise)"
        );
        assert!(
            sig_half < sig_full && sig_half > sig_zero,
            "strength 0.5 init sigma is intermediate"
        );
        // strength 0 ⇒ the smallest tabled sigma (source barely noised, near-clean denoise).
        let want0 = 5.0 * 0.001 / (1.0 + 4.0 * 0.001);
        assert!((sig_zero - want0).abs() < 1e-9);

        // Every list is descending with a terminal 0 (for steps ≥ 2).
        for s in [&full, &half] {
            assert_eq!(*s.step_timesteps().last().unwrap(), 0.0);
            assert!(s.step_timesteps().windows(2).all(|w| w[0] >= w[1]));
        }
    }

    /// `for_strength` rejects an out-of-range or non-finite strength and a zero step count.
    #[test]
    fn for_strength_validates_inputs() {
        assert!(FewStepSchedule::for_strength(5.0, 1.5, 5).is_err());
        assert!(FewStepSchedule::for_strength(5.0, -0.1, 5).is_err());
        assert!(FewStepSchedule::for_strength(5.0, f64::NAN, 5).is_err());
        assert!(FewStepSchedule::for_strength(5.0, 0.5, 0).is_err());
        assert!(FewStepSchedule::for_strength(0.0, 0.5, 5).is_err());
    }

    #[test]
    fn scalar_ops_preserve_f32_dtype() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let b = Array::from_slice(&[0.0f32, 0.0], &[2]);
        assert_eq!(euler_x0(&a, &b, 0.3).unwrap().dtype(), Dtype::Float32);
        assert_eq!(renoise_step(&a, &b, 0.3).unwrap().dtype(), Dtype::Float32);
    }
}
