//! sc-5114 — Lens schedule + CFG parity vs the vendor `LensImagePipeline` (diffusers
//! `FlowMatchEulerDiscreteScheduler`).
//!
//! **Weight-free** (CPU, no GPU): checks, against `scripts/dump_lens_schedule_golden.py`, that for
//! both the Turbo (4-step) and base (20-step) counts the Rust [`candle_gen_lens::schedule`]
//! reproduces (a) the shifted sigmas and (b) the norm-rescaled CFG — both near-bit (f32). The
//! per-step Euler integration is exercised by the unified `candle_gen::run_flow_sampler` tests
//! (epic 7114). Gated on the golden file existing (it needs the lens-venv to dump):
//!   LENS_SCHEDULE_GOLDENS — lens_schedule_golden.safetensors (default: .scratch/lens-schedule-goldens/…)
//! Run:  cargo test -p candle-gen-lens --test integration schedule_parity:: -- --nocapture

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::testkit::{tensor_max_abs_diff_host, tensor_peak_rel_host};
use candle_gen_lens::schedule::{cfg_rescale, lens_sigmas};

const LATENT: usize = 64; // 64×64 = 4096 seq_len (matches the dump's SEQ_LEN)

#[test]
#[ignore = "needs the schedule goldens (LENS_SCHEDULE_GOLDENS; run scripts/dump_lens_schedule_golden.py)"]
fn lens_schedule_matches_reference() -> Result<()> {
    let goldens_path = std::env::var("LENS_SCHEDULE_GOLDENS").unwrap_or_else(|_| {
        ".scratch/lens-schedule-goldens/lens_schedule_golden.safetensors".to_string()
    });
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_schedule_golden.py)"
        );
        return Ok(());
    }
    let dev = Device::Cpu;
    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &dev)?;

    for n in [4usize, 20] {
        let sigmas = lens_sigmas(n, LATENT, LATENT);

        // (a) shifted sigmas (n+1, trailing 0).
        let got_sigmas = Tensor::from_slice(&sigmas, sigmas.len(), &dev)?;
        let d_sig = tensor_max_abs_diff_host(&got_sigmas, &g[&format!("sigmas_{n}")]);

        eprintln!("n={n}: sigmas Δ={d_sig:.3e}");
        assert!(d_sig < 1e-5, "n={n} sigmas Δ {d_sig:.3e}");
    }

    // (b) norm-rescaled CFG (guidance 5.0).
    let got_cfg = cfg_rescale(&g["cfg_cond"], &g["cfg_uncond"], 5.0)?;
    let d_cfg = tensor_peak_rel_host(&got_cfg, &g["cfg_out"], Some(1e-12));
    eprintln!("cfg: peak_rel={d_cfg:.3e}");
    assert!(d_cfg < 1e-4, "cfg peak_rel {d_cfg:.3e}");
    eprintln!("ALL PASS");
    Ok(())
}
