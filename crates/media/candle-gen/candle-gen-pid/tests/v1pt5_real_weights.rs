//! sc-12143 gate: load the REAL PiD **v1.5** flux student in candle and (1) run a forward, (2) match the
//! reference torch `PidNet.forward` on identical inputs. Candle runs **f32** throughout (vs the mlx
//! port's bf16 weights), so parity against the f32 reference is tight.
//!
//! `#[ignore]`d — needs the converted v1.5 flux safetensors + the reference forward dump (both produced
//! in sc-12141/sc-12142):
//!
//! ```text
//! PID_V1PT5_CKPT=/path/pid_flux_2kto4k_v1pt5.safetensors \
//! PID_V1PT5_REF_DUMP=/path/ref_v1pt5_flux_forward.safetensors \
//!   cargo test -p candle-gen-pid --test v1pt5_real_weights -- --ignored --nocapture
//! ```

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;
use candle_gen_pid::config::PidConfig;
use candle_gen_pid::lq::PidNet;

fn env_path(var: &str) -> std::path::PathBuf {
    std::env::var(var)
        .unwrap_or_else(|_| panic!("set {var} to the required safetensors path"))
        .into()
}

/// The v1.5 config for the backbone under test (`PID_V1PT5_BACKBONE`, default `flux`). flux/qwenimage
/// are 16-ch / 8×; flux2 feeds the packed 128-ch / 16× latent (un-patchified in-adapter).
fn v1pt5_cfg() -> PidConfig {
    let mut cfg = PidConfig::sr4x_v1pt5();
    if std::env::var("PID_V1PT5_BACKBONE").as_deref() == Ok("flux2") {
        cfg.lq_latent_channels = 128;
        cfg.latent_spatial_down_factor = 16;
    } else {
        cfg.lq_latent_channels = 16;
        cfg.latent_spatial_down_factor = 8;
    }
    cfg
}

fn max_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
#[ignore = "needs the converted PiD v1.5 flux safetensors (PID_V1PT5_CKPT)"]
fn v1pt5_flux_loads_and_forwards() {
    let dev = Device::Cpu;
    let w = Weights::from_file(&env_path("PID_V1PT5_CKPT"), &dev, DType::F32).unwrap();
    // Building the v1.5 net REQUIRES the v1.5-only keys (`lq_proj.pit_head`, top-level `pit_lq_gate`),
    // so a clean build already proves the new modules are present + wired.
    let net = PidNet::from_weights(&w, "", &v1pt5_cfg()).unwrap();

    // Output pixels [1,3,64,64] (pH=pW=4 over patch 16); LQ latent [1,16,2,2] (upsample 4·8/16 = 2).
    let x = Tensor::randn(0f32, 1., (1, 3, 64, 64), &dev).unwrap();
    let t = Tensor::new(&[500.0f32], &dev).unwrap();
    let y = Tensor::randn(0f32, 1., (1, 8, 2304), &dev).unwrap();
    let lq = Tensor::randn(0f32, 1., (1, 16, 2, 2), &dev).unwrap();
    let sigma = Tensor::new(&[0.0f32], &dev).unwrap();

    let out = net.forward(&x, &t, &y, &lq, &sigma).unwrap();
    assert_eq!(out.dims(), &[1, 3, 64, 64], "v1.5 forward output shape");
    let peak = max_abs(&out);
    assert!(
        peak.is_finite() && peak > 1e-6,
        "v1.5 forward degenerate (peak={peak})"
    );
    eprintln!("candle v1.5 flux forward OK — output [1,3,64,64], peak|·|={peak:.4}");
}

#[test]
#[ignore = "needs PID_V1PT5_CKPT + PID_V1PT5_REF_DUMP (reference forward dump)"]
fn v1pt5_flux_forward_matches_reference() {
    let dev = Device::Cpu;
    let w = Weights::from_file(&env_path("PID_V1PT5_CKPT"), &dev, DType::F32).unwrap();
    let net = PidNet::from_weights(&w, "", &v1pt5_cfg()).unwrap();

    let d = Weights::from_file(&env_path("PID_V1PT5_REF_DUMP"), &dev, DType::F32).unwrap();
    let get = |k: &str| d.require(k).unwrap();
    let (x, t, y) = (get("x"), get("t"), get("y"));
    let (lq, sigma, ref_out) = (get("lq_latent"), get("sigma"), get("out"));

    let out = net.forward(&x, &t, &y, &lq, &sigma).unwrap();
    assert_eq!(out.dims(), ref_out.dims(), "shape vs reference");

    let max_abs_diff = max_abs(&(out - &ref_out).unwrap());
    let ref_peak = max_abs(&ref_out);
    let peak_rel = max_abs_diff / ref_peak;
    eprintln!(
        "candle v1.5 flux forward parity: max|Δ|={max_abs_diff:.4e}  ref-peak={ref_peak:.4}  peak-rel={peak_rel:.4e}"
    );
    // Candle runs f32 vs the f32 reference, so this is a near-exact match (f32 matmul-order noise only),
    // much tighter than the mlx bf16 floor (1.6e-2). Bound at 5e-3; a wiring bug would be orders larger.
    assert!(
        peak_rel < 5e-3,
        "candle v1.5 forward diverges from reference: peak-rel={peak_rel} (max|Δ|={max_abs_diff})"
    );
}
