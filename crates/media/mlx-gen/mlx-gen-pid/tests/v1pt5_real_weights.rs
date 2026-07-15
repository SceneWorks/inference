//! sc-12142 structural gate: load the REAL PiD **v1.5** flux student and run a forward, proving the
//! v1.5 loader + the new modules (wider 1024 trunk, per-token scalar gate, **replicate** conv padding,
//! **PiT** injection via `pit_head` + `pit_lq_gate`, 2048 RoPE ref) consume the released checkpoint and
//! produce a finite image of the right shape.
//!
//! `#[ignore]`d — needs the converted v1.5 safetensors (sc-12141 converted the `nvidia/PiD`
//! `PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step` checkpoint):
//!
//! ```text
//! PID_V1PT5_CKPT=/path/pid_flux_2kto4k_v1pt5.safetensors \
//!   cargo test -p mlx-gen-pid --test v1pt5_real_weights -- --ignored --nocapture
//! ```
//!
//! This is the STRUCTURAL gate (correct topology consumed, finite output). Bit-level numeric parity vs
//! the reference torch decode is sc-12142's golden step (needs the reference pipeline stood up).

use mlx_gen::weights::Weights;
use mlx_gen_pid::{PidConfig, PidNet};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{random, Array, Dtype};

fn ckpt_path() -> String {
    std::env::var("PID_V1PT5_CKPT")
        .expect("set PID_V1PT5_CKPT to the converted v1.5 flux safetensors")
}

/// The flux latent-space v1.5 config: the shared v1.5 topology + flux's 16-ch / 8× LQ geometry (exactly
/// what `PidEngine::load` builds after sniffing v1.5 — see `engine.rs`).
fn flux_v1pt5_cfg() -> PidConfig {
    let mut cfg = PidConfig::sr4x_v1pt5();
    cfg.lq_latent_channels = 16;
    cfg.latent_spatial_down_factor = 8;
    cfg
}

#[test]
#[ignore = "needs the converted PiD v1.5 flux safetensors (PID_V1PT5_CKPT)"]
fn v1pt5_flux_loads_and_forwards() {
    let w = Weights::from_file(ckpt_path()).unwrap();
    let cfg = flux_v1pt5_cfg();

    // Building the v1.5 net REQUIRES the v1.5-only keys: `lq_proj.pit_head` and the top-level
    // `pit_lq_gate` (pit_lq_inject=true → `.transpose()?` errors if absent). So a clean build already
    // proves those modules are present + wired. The 7 scalar gates + 1024-wide replicate-padded conv
    // trunk load through the same path.
    let net = PidNet::from_weights(&w, "", &cfg).unwrap();
    assert_eq!(net.lq_latent_channels(), 16);

    // Small internally-consistent decode geometry: output pixels [1,3,64,64] (pH=pW=4 over patch 16),
    // LQ latent [1,16,2,2] (upsample_ratio = sr·lsdf/patch = 4·8/16 = 2 → 2·2 = 4 = pH).
    let key = random::key(7).unwrap();
    let x = random::normal::<f32>(&[1, 3, 64, 64], None, None, Some(&key)).unwrap();
    let t = Array::from_slice(&[500.0f32], &[1]);
    let y = random::normal::<f32>(&[1, 8, 2304], None, None, Some(&key)).unwrap();
    let lq_latent = random::normal::<f32>(&[1, 16, 2, 2], None, None, Some(&key)).unwrap();
    let sigma = Array::from_slice(&[0.0f32], &[1]);

    let out = net.forward(&x, &t, &y, &lq_latent, &sigma).unwrap();
    assert_eq!(out.shape(), &[1, 3, 64, 64], "v1.5 forward output shape");

    // Finite + non-degenerate: a broken PiT injection / padding / gate would NaN or collapse to 0.
    let peak = max(abs(&out).unwrap(), None).unwrap().item::<f32>();
    assert!(
        peak.is_finite(),
        "v1.5 forward produced non-finite output (peak={peak})"
    );
    assert!(peak > 1e-6, "v1.5 forward collapsed to ~0 (peak={peak})");
    eprintln!("v1.5 flux forward OK — output [1,3,64,64], peak|·|={peak:.4}");
}

/// NUMERIC GOLDEN (sc-12142): the mlx v1.5 flux `PidNet::forward` matches the reference torch
/// `PidNet.forward` (nv-tlabs/PiD) on identical inputs, within the cross-precision floor (mlx runs
/// bf16 weights / f32 activations vs the f32 reference — the repo's parity philosophy is "no quality
/// regression," not bit-exact). The reference dump is produced by `scratchpad/.../ref_forward_dump.py`,
/// which `strict`-loads the real checkpoint into the reference `PidNet` (so its config is verified to
/// exactly match the released topology) and forwards fixed-seed inputs.
///
/// ```text
/// PID_V1PT5_CKPT=/path/pid_flux_2kto4k_v1pt5.safetensors \
/// PID_V1PT5_REF_DUMP=/path/ref_v1pt5_flux_forward.safetensors \
///   cargo test -p mlx-gen-pid --test v1pt5_real_weights -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs PID_V1PT5_CKPT + PID_V1PT5_REF_DUMP (reference forward dump)"]
fn v1pt5_flux_forward_matches_reference() {
    let w = Weights::from_file(ckpt_path()).unwrap();
    let net = PidNet::from_weights(&w, "", &flux_v1pt5_cfg()).unwrap();

    let dump = std::env::var("PID_V1PT5_REF_DUMP")
        .expect("set PID_V1PT5_REF_DUMP to the reference forward-dump safetensors");
    let d = Weights::from_file(&dump).unwrap();
    let f32 = |k: &str| d.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();
    let (x, t, y) = (f32("x"), f32("t"), f32("y"));
    let (lq_latent, sigma, ref_out) = (f32("lq_latent"), f32("sigma"), f32("out"));

    let out = net
        .forward(&x, &t, &y, &lq_latent, &sigma)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    assert_eq!(out.shape(), ref_out.shape(), "shape vs reference");

    let diff = subtract(&out, &ref_out).unwrap();
    let max_abs_diff = max(abs(&diff).unwrap(), None).unwrap().item::<f32>();
    let ref_peak = max(abs(&ref_out).unwrap(), None).unwrap().item::<f32>();
    let peak_rel = max_abs_diff / ref_peak;
    eprintln!(
        "v1.5 flux forward parity: max|Δ|={max_abs_diff:.4e}  ref-peak={ref_peak:.4}  peak-rel={peak_rel:.4e}"
    );
    // Cross-precision floor: mlx (bf16 weights) vs f32 reference over a 14-block MMDiT + 2 PiT + LQ
    // forward. The repo's own LQ-projection parity gate is 2e-2 peak-rel; a full forward accumulates a
    // bit more, so allow 3e-2. A wiring bug (wrong padding / gate / PiT injection) would be orders larger.
    assert!(
        peak_rel < 3e-2,
        "v1.5 forward diverges from reference: peak-rel={peak_rel} (max|Δ|={max_abs_diff})"
    );
}
