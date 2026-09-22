//! Scheduler parity vs the frozen `FlowMatchEulerDiscreteScheduler`
//! (`crates/media/mlx-gen/tools/dump_qwen21_scheduler.py`): every upstream preset at 40 steps, the
//! 1:1 preset at 8, and the tiny e2e geometries — sigmas (N + 1, trailing 0), timesteps and `mu`.
//!
//! The fixture is the SAME committed `qwen21_scheduler.safetensors` the MLX twin validates against,
//! so both backends' schedules are pinned to one table.
//!
//! Tolerance: **1e-6 absolute** on sigmas and on `mu` (both sides are f64 rounded to f32), and 1e-3
//! absolute on `sigma · num_train_timesteps` vs the upstream timestep table.

use candle_gen_qwen_image_2_1::scheduler::{image_tokens, mu_for_tokens, sigmas};
use candle_gen_qwen_image_2_1::SchedulerConfig;

use crate::common::{host_f32, Fixture};

const CASES: [&str; 11] = [
    "preset_1x1",
    "preset_4x3",
    "preset_3x4",
    "preset_3x2",
    "preset_2x3",
    "preset_16x9",
    "preset_9x16",
    "preset_1x1_8",
    "tiny_32x32_3",
    "tiny_64x32_2",
    "tiny_1024x1024_2",
];

#[test]
fn every_preset_schedule_matches_upstream() {
    let w = Fixture::open("qwen21_scheduler.safetensors");
    let cfg = SchedulerConfig::production();
    for case in CASES {
        let width = w.meta_usize(&format!("{case}/width")) as u32;
        let height = w.meta_usize(&format!("{case}/height")) as u32;
        let steps = w.meta_usize(&format!("{case}/steps"));
        let tokens = image_tokens(width, height);
        let want_sigmas = host_f32(&w.tensor(&format!("{case}/sigmas")));
        let want_timesteps = host_f32(&w.tensor(&format!("{case}/timesteps")));
        let want_mu = host_f32(&w.tensor(&format!("{case}/mu")))[0];

        let got = sigmas(&cfg, steps, tokens).unwrap();
        assert_eq!(got.len(), want_sigmas.len(), "{case}");
        assert_eq!(got.len(), steps + 1, "{case}");
        let mut worst = 0f32;
        for (i, (g, s)) in got.iter().zip(&want_sigmas).enumerate() {
            let d = (g - s).abs();
            worst = worst.max(d);
            assert!(d <= 1e-6, "{case}: sigma[{i}] {g} vs upstream {s}");
        }
        for (i, t) in want_timesteps.iter().enumerate() {
            assert!(
                (got[i] * cfg.num_train_timesteps as f32 - t).abs() <= 1e-3,
                "{case}: timestep[{i}] {} vs upstream {t}",
                got[i] * 1000.0
            );
        }
        let mu = mu_for_tokens(&cfg, tokens) as f32;
        assert!((mu - want_mu).abs() <= 1e-6, "{case}: mu {mu} vs {want_mu}");
        eprintln!(
            "{case}: {width}x{height} steps={steps} tokens={tokens} mu={mu:.6} max|Δσ|={worst:.2e}"
        );
    }
}

#[test]
fn the_seven_presets_are_all_covered_by_the_table() {
    let w = Fixture::open("qwen21_scheduler.safetensors");
    for preset in candle_gen_qwen_image_2_1::PRESETS {
        let case = format!("preset_{}", preset.ratio.replace(':', "x"));
        assert_eq!(w.meta_usize(&format!("{case}/width")), preset.width as usize);
        assert_eq!(
            w.meta_usize(&format!("{case}/height")),
            preset.height as usize
        );
        assert_eq!(w.meta_usize(&format!("{case}/steps")), 40);
    }
}

#[test]
fn one_step_is_refused_not_nan() {
    let err = sigmas(&SchedulerConfig::production(), 1, 4096)
        .unwrap_err()
        .to_string();
    assert!(err.contains("steps must be >= 2"), "{err}");
}
