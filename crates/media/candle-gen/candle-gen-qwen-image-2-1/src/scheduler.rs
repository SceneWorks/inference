//! The Qwen-Image 2.1 flow-match schedule — a port of what `QwenImage21Pipeline.__call__` asks the
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
//! [`TimestepConvention::Sigma`](candle_gen::gen_core::sampling::TimestepConvention::Sigma), and the
//! Euler update is the shared `x ← x + v·(σ_{i+1} − σ_i)`.
//!
//! All arithmetic is carried in `f64` and rounded to `f32` at the end, matching numpy → torch
//! float32; `tests/scheduler_parity.rs` pins every preset against the upstream table.

use candle_gen::{CandleError as Error, Result};

use crate::config::{SchedulerConfig, VAE_SCALE_FACTOR};

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

/// The Turbo schedule (sc-25901) for `steps` denoising steps over the snapshot's saved
/// `sample_sigmas`: length `steps + 1`, trailing `0.0`. No `mu` shift and no terminal stretch —
/// the distilled checkpoint is sampled on the sigmas it was distilled over.
///
/// * `steps == saved.len()` (what a request omitting `steps` gets): exactly the saved list, then
///   `0.0` — upstream's `set_timesteps(sigmas=config.sample_sigmas)` under a static scheduler.
/// * any other `steps = N >= 1`: the saved curve **resampled** — a SceneWorks choice, not
///   upstream's (see `UPSTREAM.md`, "Deliberate divergences"). The knots are the `K` saved sigmas
///   followed by `0.0`, at positions `u_j = j / K` for `j = 0..=K`; `σ_i` is the piecewise-linear
///   interpolation of the knots at `u = i / N` for `i = 0..N`.
///
/// The knot index and the in-segment fraction come from the **integer** product `i · K` (`j = ⌊i·K
/// / N⌋`, fraction `(i·K mod N) / N`), so a sample that lands on a knot reads it back bit-for-bit
/// and `N = K` reproduces the saved list exactly rather than to a rounding error. Arithmetic is
/// `f64`, rounded to `f32` at the end, like [`sigmas`].
pub fn saved_sigmas(saved: &[f64], steps: usize) -> Result<Vec<f32>> {
    let k = saved.len();
    if k == 0 {
        return Err(Error::Msg(format!(
            "{}: the snapshot's `sample_sigmas` is empty",
            crate::TURBO_MODEL_ID
        )));
    }
    if steps == 0 {
        return Err(Error::Msg(format!(
            "{}: steps must be >= 1 (got 0)",
            crate::TURBO_MODEL_ID
        )));
    }
    let knot = |j: usize| saved.get(j).copied().unwrap_or(0.0);
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            let (j, rem) = ((i * k) / steps, (i * k) % steps);
            let fraction = rem as f64 / steps as f64;
            (knot(j) + fraction * (knot(j + 1) - knot(j))) as f32
        })
        .collect();
    out.push(0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// `Qwen/Qwen-Image-2.1-Turbo`'s `model_index.json` `sample_sigmas`, verbatim.
    const TURBO_SAVED: [f64; 8] = [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
    ];

    /// The saved step count is the saved list itself — every entry bit-for-bit — plus the `0`.
    ///
    /// *Mutation that reds this:* applying any shift or stretch to the saved sigmas.
    #[test]
    fn the_saved_step_count_is_the_saved_list_then_zero() {
        let got = saved_sigmas(&TURBO_SAVED, TURBO_SAVED.len()).unwrap();
        let mut want: Vec<f32> = TURBO_SAVED.iter().map(|s| *s as f32).collect();
        want.push(0.0);
        assert_eq!(got, want);
    }

    /// Twelve steps over the eight saved sigmas, pinned to literals computed from the algorithm in
    /// [`saved_sigmas`]'s doc (numpy, f64, rounded to f32) — the cross-backend comparison point.
    #[test]
    fn twelve_steps_resample_the_saved_curve() {
        let got = saved_sigmas(&TURBO_SAVED, 12).unwrap();
        let want: [f32; 13] = [
            1.0,
            0.985_635_34,
            0.970_362,
            0.954_18,
            0.935_810_7,
            0.916_110_7,
            0.895_08,
            0.861_792,
            0.798_276_66,
            0.704_534,
            0.511_223_3,
            0.276_378_66,
            0.0,
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn resampling_holds_its_shape_at_every_step_count() {
        for steps in 1..=40 {
            let s = saved_sigmas(&TURBO_SAVED, steps).unwrap();
            assert_eq!(s.len(), steps + 1, "{steps}");
            assert_eq!(s[0], 1.0, "{steps}: starts on the first saved sigma");
            assert_eq!(s[steps], 0.0, "{steps}");
            assert!(s.windows(2).all(|w| w[0] > w[1]), "{steps}: {s:?}");
        }
        // One step is the whole interval; a doubled count passes through every saved sigma.
        assert_eq!(saved_sigmas(&TURBO_SAVED, 1).unwrap(), [1.0, 0.0]);
        let doubled = saved_sigmas(&TURBO_SAVED, 16).unwrap();
        for (j, sigma) in TURBO_SAVED.iter().enumerate() {
            assert_eq!(doubled[2 * j], *sigma as f32, "knot {j}");
        }
        assert!(saved_sigmas(&TURBO_SAVED, 0).is_err());
        assert!(saved_sigmas(&[], 8).is_err());
    }

    #[test]
    fn one_step_is_refused_not_nan() {
        let err = sigmas(&SchedulerConfig::production(), 1, 4096)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
    }
}
