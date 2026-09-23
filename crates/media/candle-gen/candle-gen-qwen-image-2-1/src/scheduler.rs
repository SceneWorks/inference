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

    #[test]
    fn one_step_is_refused_not_nan() {
        let err = sigmas(&SchedulerConfig::production(), 1, 4096)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
    }
}
