//! sc-7843 component 5: the Gemma-2 caption-encoder decoder matches HF `Gemma2Model`. Fixture =
//! `tools/dump_pid_gemma.py` (tiny Gemma-2, eager attention so the logit soft-cap path runs). Gates
//! the full decoder forward: embed×√hidden, rotate_half RoPE, GQA, soft-capped attention, gelu-tanh
//! MLP, the norm-sandwich, final RMSNorm.

use mlx_gen::weights::Weights;
use mlx_gen_pid::{Gemma2, Gemma2Config};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

#[cfg(target_os = "macos")]
#[path = "../../mlx-gen-iris/tests/common/metal_guard.rs"]
mod quantum_metal_guard;

#[cfg(target_os = "macos")]
#[test]
fn gemma_quantum_cpu_matches_lazy_and_refuses_low_headroom() {
    const CHILD: &str = "GEMMA_QUANTUM_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "gemma_parity::gemma_quantum_cpu_matches_lazy_and_refuses_low_headroom",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    assert!(quantum_metal_guard::forbid_metal());
    assert!(quantum_metal_guard::metal_forbidden());
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let w = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gemma2_tiny.safetensors"
    ))
    .unwrap();
    let model = Gemma2::from_weights(&w, "", &tiny_cfg()).unwrap();
    let ids = w.require("__io__.ids").unwrap();
    let lazy = model.forward(ids, None).unwrap();
    lazy.eval().unwrap();
    let guarded = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1024 * 1024 * 1024, || {
        model.forward(ids, None)
    })
    .unwrap();
    assert_eq!(
        max_abs(&subtract(&guarded, &lazy).unwrap()),
        0.0,
        "barriers change lifetime, not Gemma arithmetic"
    );
    let refused =
        mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(0, || model.forward(ids, None));
    assert!(refused
        .unwrap_err()
        .to_string()
        .contains("before construction"));
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

fn tiny_cfg() -> Gemma2Config {
    Gemma2Config {
        hidden_size: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        attn_softcap: 50.0,
        query_pre_attn_scalar: 8.0,
        rms_eps: 1e-6,
    }
}

#[test]
fn gemma2_decoder_matches() {
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let w = Weights::from_file(format!("{dir}/gemma2_tiny.safetensors")).unwrap();

    let ids = w.require("__io__.ids").unwrap().clone(); // [1,6] i32
    let golden = w.require("__io__.last_hidden").unwrap().clone();

    let model = Gemma2::from_weights(&w, "", &tiny_cfg()).unwrap();
    let got = model.forward(&ids, None).unwrap();
    assert_eq!(got.shape(), golden.shape(), "last_hidden shape");

    let d = max_abs(&subtract(&got, &golden).unwrap());
    let scale = max_abs(&golden);
    let rel = d / scale;
    eprintln!("gemma2 decoder: max|Δ|={d:.3e} scale={scale:.3e} peak-rel={rel:.3e}");
    assert!(
        rel < 2e-2,
        "gemma2 last_hidden peak-rel={rel} (max|Δ|={d}) — divergence in the decoder forward"
    );
}
