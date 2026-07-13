//! SCAIL-2 **real-weight 40-layer** DiT forward parity gate (sc-5446).
//!
//! The tiny-seeded [`dit_parity`](../dit_parity.rs) proves the algorithm on a 2-layer random model;
//! this proves the **real 40-layer** `dit.safetensors` loads + forwards correctly. It loads the real
//! snapshot DiT via [`Scail2Dit::from_weights`] (f32 compute) and compares one forward per mode
//! (animation / cross-identity replacement) against the upstream `SCAIL2Model.forward` fp32 reference.
//! Both sides start from the SAME bf16 weights upcast to f32, so the residual vs. the tiny gate's
//! 0.9999996 is MLX Metal matmul reduced precision accumulated over 40 real layers.
//!
//! Generate the fp32 reference on the Mac (torch venv; ~100 GB peak — run when memory is free):
//!
//! ```text
//! SCAIL2_SNAPSHOT_DIR=~/.cache/scail2-mlx-convert \
//! SCAIL2_REAL_PARITY_DIR=~/.cache/scail2-parity-real \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_dit_realweight_fixtures.py
//! ```
//!
//! `#[ignore]` (needs the ~31 GB snapshot + the locally-generated fixtures). Run on macOS with
//! `cargo test -p mlx-gen-scail2 --test dit_real_parity -- --ignored --nocapture`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{Scail2Config, Scail2Dit, Scail2Inputs};
use mlx_rs::{Array, Dtype};

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn snapshot_dir() -> PathBuf {
    std::env::var("SCAIL2_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(".cache/scail2-mlx-convert"))
}

fn real_parity_dir() -> PathBuf {
    std::env::var("SCAIL2_REAL_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(".cache/scail2-parity-real"))
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn compare(a: &Array, b: &Array) -> (f32, f32) {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    let (mut dot, mut na, mut nb, mut max_abs) = (0f64, 0f64, 0f64, 0f32);
    for (x, y) in va.iter().zip(vb.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    ((dot / (na.sqrt() * nb.sqrt())) as f32, max_abs)
}

#[test]
#[ignore = "real ~31 GB DiT + locally-generated fixtures (see module doc); run with --ignored on macOS"]
fn dit_real_parity() {
    let snap = snapshot_dir();
    let pdir = real_parity_dir();
    let dit_path = snap.join("dit.safetensors");
    assert!(
        dit_path.exists() && pdir.join("real_anim/io.safetensors").exists(),
        "missing real DiT weights / fixtures — generate with \
         `~/mlx-flux-venv/bin/python _vendor/scail2/_gen_dit_realweight_fixtures.py` ({} / {})",
        dit_path.display(),
        pdir.display(),
    );

    let cfg = Scail2Config::from_model_dir(&snap).unwrap();
    let w = Weights::from_file(&dit_path).unwrap();
    let dit = Scail2Dit::from_weights(&w, &cfg).unwrap();
    assert_eq!(dit.num_blocks(), 40, "real SCAIL-2 14B is 40 layers");

    let mut worst = 1.0f32;
    for (name, replace_flag) in [("real_anim", false), ("real_replace", true)] {
        let io = Weights::from_file(pdir.join(name).join("io.safetensors")).unwrap();
        let get = |k: &str| io.require(k).unwrap();
        let inputs = Scail2Inputs {
            x: get("x"),
            ref_latent: get("ref_latent"),
            ref_masks: get("ref_masks"),
            pose_latent: get("pose_latent"),
            driving_masks: get("driving_masks"),
            history_mask: None,
            additional_ref_latent: None,
            additional_ref_masks: None,
            clip_fea: get("clip_fea"),
            context: get("context"),
            t: 500.0,
            replace_flag,
        };
        let out = dit.forward(&inputs).unwrap();
        let reference = get("output");
        assert_eq!(out.shape(), reference.shape(), "[{name}] output shape");
        let (cos, max_abs) = compare(&out, reference);
        println!("[{name:13}] real 40-layer cosine {cos:.7}  max|Δ| {max_abs:.5}");
        worst = worst.min(cos);
        assert!(
            cos > 0.99,
            "[{name}] cosine {cos} below 0.99 (max|Δ| {max_abs})"
        );
    }
    println!("real 40-layer DiT worst-case cosine: {worst:.7}");
}
