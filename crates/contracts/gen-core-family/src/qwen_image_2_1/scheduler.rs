//! The Qwen-Image 2.1 flow-match schedule, backend-neutral and shared by both engines (hoisted from
//! their mirrored `scheduler.rs` copies, sc-27329) — a port of what `QwenImage21Pipeline.__call__` asks the
//! frozen `FlowMatchEulerDiscreteScheduler` for:
//!
//! 1. `sigmas = linspace(1, 1/N, N)` (float64, as numpy builds it);
//! 2. `mu = calculate_shift(tokens, base_image_seq_len, max_image_seq_len, base_shift, max_shift)`
//!    — a linear fit in the **latent token count** `(h/16)·(w/16)`, extrapolated past
//!    `max_image_seq_len` (the 2K presets sit at ~16.4k tokens against a 8192 knee);
//! 3. the exponential time shift `σ' = e^μ / (e^μ + (1/σ − 1))`;
//! 4. the terminal stretch `σ'' = 1 − (1 − σ') / ((1 − σ'_last) / (1 − shift_terminal))`, so the
//!    last sigma lands exactly on `shift_terminal` (0.02);
//! 5. a trailing `0.0`, and `timesteps = σ · num_train_timesteps`.
//!
//! The transformer is fed the raw sigma (`timestep / 1000` in the pipeline), i.e.
//! [`TimestepConvention::Sigma`](gen_core::sampling::TimestepConvention::Sigma), and the
//! Euler update is the shared `x ← x + v·(σ_{i+1} − σ_i)`.
//!
//! All arithmetic is carried in `f64` and rounded to `f32` at the end, matching numpy → torch
//! float32; each engine's `tests/scheduler_parity.rs` pins every preset against the upstream table.
//!
//! The distilled Turbo checkpoint (sc-25900) uses none of steps 1–4: its snapshot saves the sigma
//! list outright and [`resample_saved_sigmas`] is its whole schedule.

use super::config::{SchedulerConfig, VAE_SCALE_FACTOR};
use gen_core::{Error, Result};

/// Latent tokens for a `width × height` request — one per 16×16 pixel tile.
pub fn image_tokens(width: u32, height: u32) -> usize {
    ((height / VAE_SCALE_FACTOR) * (width / VAE_SCALE_FACTOR)) as usize
}

/// `calculate_shift`: the resolution-dependent time-shift `mu` for `tokens` latent tokens.
pub fn mu_for_tokens(cfg: &SchedulerConfig, tokens: usize) -> f64 {
    let m = (cfg.max_shift as f64 - cfg.base_shift as f64)
        / (cfg.max_image_seq_len as f64 - cfg.base_image_seq_len as f64);
    let b = cfg.base_shift as f64 - m * cfg.base_image_seq_len as f64;
    tokens as f64 * m + b
}

/// The descending sigma schedule for `steps` denoising steps over `tokens` latent tokens: length
/// `steps + 1` with a trailing `0.0`. `steps < 2` is refused — the terminal stretch divides by
/// `1 − σ'_last`, which is zero at a single step (upstream produces `NaN` there).
pub fn sigmas(cfg: &SchedulerConfig, steps: usize, tokens: usize) -> Result<Vec<f32>> {
    if steps < 2 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: steps must be >= 2 (got {steps}); the terminal-sigma stretch is undefined at one step"
        )));
    }
    let n = steps;
    let mu = mu_for_tokens(cfg, tokens);
    // numpy `linspace(1.0, 1/n, n)`: start + i * (stop - start) / (n - 1).
    let (start, stop) = (1.0_f64, 1.0_f64 / n as f64);
    let step = (stop - start) / (n as f64 - 1.0);
    let mut s: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    // numpy pins the last linspace sample to `stop` exactly.
    s[n - 1] = stop;
    // `_time_shift_exponential(mu, 1.0, t)`: exp(mu) / (exp(mu) + (1/t - 1) ** 1.0).
    let e = mu.exp();
    for t in s.iter_mut() {
        *t = e / (e + (1.0 / *t - 1.0));
    }
    if let Some(terminal) = cfg.shift_terminal {
        // `stretch_shift_to_terminal`: one_minus_z = 1 - t; scale = one_minus_z[-1] / (1 - terminal);
        // stretched = 1 - one_minus_z / scale.
        let scale = (1.0 - s[n - 1]) / (1.0 - terminal as f64);
        for t in s.iter_mut() {
            *t = 1.0 - (1.0 - *t) / scale;
        }
    }
    let mut out: Vec<f32> = s.into_iter().map(|t| t as f32).collect();
    out.push(0.0);
    Ok(out)
}

/// The schedule for a `width × height` request (`sigmas` over [`image_tokens`]).
pub fn sigmas_for_image(
    cfg: &SchedulerConfig,
    steps: usize,
    width: u32,
    height: u32,
) -> Result<Vec<f32>> {
    sigmas(cfg, steps, image_tokens(width, height))
}

/// The schedule of a **distilled** checkpoint for `steps` denoising steps, from the sampling
/// sigmas its snapshot saves (`saved`: descending, terminal `0` excluded — see
/// [`super::config::SavedSigmaSchedule`]). Length `steps + 1`, trailing `0.0`. No `mu` shift and no
/// terminal stretch: the saved values are the schedule.
///
/// The saved curve is a piecewise-linear function of progress, with knots `s_0 … s_{K−1}` and
/// `s_K = 0` at `u_j = j / K` (`K = saved.len()`); step `i` of `N` reads it at `u = i / N`:
///
/// ```text
/// t = i·K        j = t div N        r = t mod N
/// σ_i = s_j + (s_{j+1} − s_j) · (r / N)
/// ```
///
/// The knot index and the fraction come from **integer** arithmetic, so `r = 0` lands exactly on
/// a knot: `steps == K` returns the saved list bit for bit (the upstream schedule), and a multiple
/// of `K` passes through every saved sigma. Everything else is `f64`, rounded to `f32` at the end.
/// Any other `steps` is a SceneWorks resample, not an upstream schedule — upstream ignores
/// `num_inference_steps` when the pipeline saves sigmas (`UPSTREAM.md`, *Deliberate divergences*).
pub fn resample_saved_sigmas(saved: &[f64], steps: usize) -> Result<Vec<f32>> {
    if saved.is_empty() {
        return Err(Error::Msg(
            "qwen_image_2_1_turbo: the snapshot saves no sampling sigmas to resample".into(),
        ));
    }
    if steps == 0 {
        return Err(Error::Msg(
            "qwen_image_2_1_turbo: steps must be >= 1 (got 0)".into(),
        ));
    }
    let (k, n) = (saved.len(), steps);
    // `s_K`, the terminal knot.
    let knot = |j: usize| if j == k { 0.0 } else { saved[j] };
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = i * k;
        let (j, r) = (t / n, t % n);
        let sigma = knot(j) + (knot(j + 1) - knot(j)) * (r as f64 / n as f64);
        out.push(sigma as f32);
    }
    out.push(0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Qwen/Qwen-Image-2.1-Turbo` @ `UPSTREAM_TURBO_HF_REVISION`, `model_index.json`, verbatim.
    const TURBO_SAVED: [f64; 8] = [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
    ];

    /// `resample_saved_sigmas(&TURBO_SAVED, 12)`, each value the shortest decimal that rounds to
    /// the expected `f32`.
    const TWELVE_STEP_RESAMPLE: [f32; 13] = [
        1.0, 0.98563533, 0.970362, 0.95418, 0.9358107, 0.9161107, 0.89508, 0.861792, 0.79827667,
        0.704534, 0.5112233, 0.27637867, 0.0,
    ];

    /// At the saved length the resample **is** the saved list plus the terminal 0 — bit for bit,
    /// with no shift and no stretch.
    ///
    /// *Mutation that reds this:* reading the knot one segment late (`knot(j + 1)` as the base),
    /// or passing the result through the base checkpoint's `mu` shift or terminal stretch.
    #[test]
    fn the_saved_length_reproduces_the_saved_sigmas_exactly() {
        let got = resample_saved_sigmas(&TURBO_SAVED, TURBO_SAVED.len()).unwrap();
        let mut want: Vec<f32> = TURBO_SAVED.iter().map(|s| *s as f32).collect();
        want.push(0.0);
        assert_eq!(got, want);
        // A multiple of K passes through every saved sigma at the even indices.
        let doubled = resample_saved_sigmas(&TURBO_SAVED, 16).unwrap();
        let through: Vec<f32> = doubled.iter().step_by(2).copied().collect();
        assert_eq!(through, want);
    }

    /// The 12-step resample of the released curve. The expected values are the definition
    /// evaluated independently (CPython `float`, i.e. IEEE-754 binary64, then `numpy.float32`):
    /// `u = i/12` falls two thirds of the way along a segment, one third along the next, then on
    /// a knot, repeating. The candle twin pins the same twelve numbers.
    #[test]
    fn twelve_steps_resample_the_saved_curve() {
        let got = resample_saved_sigmas(&TURBO_SAVED, 12).unwrap();
        assert_eq!(got, TWELVE_STEP_RESAMPLE);
    }

    /// Any step count from one up keeps the curve's shape: starts on the first saved sigma (1.0),
    /// strictly descends, ends on the appended 0.
    #[test]
    fn every_step_count_starts_at_one_descends_strictly_and_ends_at_zero() {
        for steps in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 16, 20, 30, 40, 50, 100] {
            let s = resample_saved_sigmas(&TURBO_SAVED, steps).unwrap();
            assert_eq!(s.len(), steps + 1, "{steps} steps");
            assert_eq!(s[0], 1.0, "{steps} steps");
            assert_eq!(s[steps], 0.0, "{steps} steps");
            assert!(
                s.windows(2).all(|w| w[0] > w[1]),
                "{steps} steps is not strictly descending: {s:?}"
            );
        }
        assert_eq!(resample_saved_sigmas(&TURBO_SAVED, 1).unwrap(), [1.0, 0.0]);
    }

    #[test]
    fn resampling_holds_its_shape_at_every_step_count() {
        for steps in 1..=40 {
            let s = resample_saved_sigmas(&TURBO_SAVED, steps).unwrap();
            assert_eq!(s.len(), steps + 1, "{steps}");
            assert_eq!(s[0], 1.0, "{steps}: starts on the first saved sigma");
            assert_eq!(s[steps], 0.0, "{steps}");
            assert!(s.windows(2).all(|w| w[0] > w[1]), "{steps}: {s:?}");
        }
        // One step is the whole interval; a doubled count passes through every saved sigma.
        assert_eq!(resample_saved_sigmas(&TURBO_SAVED, 1).unwrap(), [1.0, 0.0]);
        let doubled = resample_saved_sigmas(&TURBO_SAVED, 16).unwrap();
        for (j, sigma) in TURBO_SAVED.iter().enumerate() {
            assert_eq!(doubled[2 * j], *sigma as f32, "knot {j}");
        }
        assert!(resample_saved_sigmas(&TURBO_SAVED, 0).is_err());
        assert!(resample_saved_sigmas(&[], 8).is_err());
    }

    #[test]
    fn no_saved_sigmas_and_zero_steps_are_refused() {
        let err = resample_saved_sigmas(&[], 8).unwrap_err().to_string();
        assert!(err.contains("saves no sampling sigmas"), "{err}");
        let err = resample_saved_sigmas(&TURBO_SAVED, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 1"), "{err}");
    }

    #[test]
    fn schedule_shape_and_endpoints() {
        let cfg = SchedulerConfig::production();
        let s = sigmas_for_image(&cfg, 40, 2048, 2048).unwrap();
        assert_eq!(s.len(), 41);
        assert_eq!(s[0], 1.0);
        assert!((s[39] - 0.02).abs() < 1e-6, "terminal {}", s[39]);
        assert_eq!(s[40], 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        // The 2K presets extrapolate past the 8192-token knee.
        assert!((mu_for_tokens(&cfg, 16384) - 1.312903).abs() < 1e-5);
    }

    #[test]
    fn one_step_is_refused_not_nan() {
        let err = sigmas(&SchedulerConfig::production(), 1, 4096)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
    }
}
