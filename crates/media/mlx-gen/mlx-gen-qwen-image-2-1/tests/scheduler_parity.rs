//! Scheduler parity vs the frozen `FlowMatchEulerDiscreteScheduler` (`tools/dump_qwen21_scheduler.py`):
//! every upstream preset at 40 steps, the 1:1 preset at 8, and the tiny e2e geometries — sigmas
//! (N + 1, trailing 0), timesteps and `mu`.
//!
//! Tolerance: **1e-6 absolute** on sigmas/timesteps-÷1000 (both sides are f64 rounded to f32) and
//! on `mu`.

use mlx_gen_qwen_image_2_1::scheduler::{image_tokens, mu_for_tokens, sigmas};
use mlx_gen_qwen_image_2_1::SchedulerConfig;

use crate::common::{fixture, host_f32, meta_usize};

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
    let w = fixture("qwen21_scheduler.safetensors");
    let cfg = SchedulerConfig::production();
    for case in CASES {
        let width = meta_usize(&w, &format!("{case}/width")) as u32;
        let height = meta_usize(&w, &format!("{case}/height")) as u32;
        let steps = meta_usize(&w, &format!("{case}/steps"));
        let tokens = image_tokens(width, height);
        let want_sigmas = host_f32(w.require(&format!("{case}/sigmas")).unwrap());
        let want_timesteps = host_f32(w.require(&format!("{case}/timesteps")).unwrap());
        let want_mu = host_f32(w.require(&format!("{case}/mu")).unwrap())[0];

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
    let w = fixture("qwen21_scheduler.safetensors");
    for preset in mlx_gen_qwen_image_2_1::PRESETS {
        let case = format!("preset_{}", preset.ratio.replace(':', "x"));
        assert_eq!(
            meta_usize(&w, &format!("{case}/width")),
            preset.width as usize
        );
        assert_eq!(
            meta_usize(&w, &format!("{case}/height")),
            preset.height as usize
        );
        assert_eq!(meta_usize(&w, &format!("{case}/steps")), 40);
    }
}
