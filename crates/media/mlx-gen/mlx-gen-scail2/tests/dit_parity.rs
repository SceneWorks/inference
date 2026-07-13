//! SCAIL-2 DiT forward parity gate (sc-5442 / sc-5446).
//!
//! Compares [`mlx_gen_scail2::Scail2Dit::forward`] (f32) against the upstream `SCAIL2Model.forward`
//! reference on a tiny-seeded model (real head_dim 128 lane split, 2 layers) across four conditioning
//! cases: animation, cross-identity replacement (`replace_flag`), multi-reference (`additional_ref`),
//! and clean-history. The reference fixtures are generated on the Mac by:
//!
//! ```text
//! SCAIL2_PARITY_DIR=~/.cache/scail2-parity \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_dit_parity_fixtures.py
//! ```
//!
//! `#[ignore]` because it needs those locally-generated fixtures (the mandatory CLIP `img_emb` makes
//! a committed fixture ~16 MB). Run with `cargo test -p mlx-gen-scail2 -- --ignored`. The
//! lane-split / config / registration unit tests carry CI.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{Scail2Config, Scail2Dit, Scail2Inputs};
use mlx_rs::{Array, Dtype};

fn parity_dir() -> PathBuf {
    std::env::var("SCAIL2_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/scail2-parity")
        })
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// (cosine similarity, max abs diff) between two same-shape tensors.
fn compare(a: &Array, b: &Array) -> (f32, f32) {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut max_abs = 0f32;
    for (x, y) in va.iter().zip(vb.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    ((dot / (na.sqrt() * nb.sqrt())) as f32, max_abs)
}

#[test]
#[ignore = "needs locally-generated fixtures (see module doc); run with --ignored on macOS"]
fn dit_parity_tiny_seeded() {
    let dir = parity_dir();
    let model_path = dir.join("model.safetensors");
    assert!(
        model_path.exists(),
        "missing fixtures at {} — generate with \
         `SCAIL2_PARITY_DIR={} ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_dit_parity_fixtures.py`",
        dir.display(),
        dir.display(),
    );

    let cfg = Scail2Config::from_model_dir(&dir).unwrap();
    let w = Weights::from_file(&model_path).unwrap();
    let dit = Scail2Dit::from_weights(&w, &cfg).unwrap();
    assert_eq!(dit.num_blocks(), cfg.wan.num_layers);

    let cases = ["base_anim", "base_replace", "addref", "history"];
    let mut worst_cos = 1.0f32;
    for name in cases {
        let cdir = dir.join(name);
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(cdir.join("case.json")).unwrap())
                .unwrap();
        let t = meta["t"].as_f64().unwrap() as f32;
        let replace_flag = meta["replace_flag"].as_bool().unwrap();
        let has_history = meta["has_history"].as_bool().unwrap();
        let addref = meta["addref"].as_i64().unwrap();

        let io = Weights::from_file(cdir.join("io.safetensors")).unwrap();
        let get = |k: &str| io.require(k).unwrap();
        let history = if has_history {
            Some(get("history_mask"))
        } else {
            None
        };
        let (add_lat, add_mask) = if addref > 0 {
            (
                Some(get("additional_ref_latent")),
                Some(get("additional_ref_masks")),
            )
        } else {
            (None, None)
        };

        let inputs = Scail2Inputs {
            x: get("x"),
            ref_latent: get("ref_latent"),
            ref_masks: get("ref_masks"),
            pose_latent: get("pose_latent"),
            driving_masks: get("driving_masks"),
            history_mask: history,
            additional_ref_latent: add_lat,
            additional_ref_masks: add_mask,
            clip_fea: get("clip_fea"),
            context: get("context"),
            t,
            replace_flag,
        };

        let out = dit.forward(&inputs).unwrap();
        let reference = get("output");
        assert_eq!(out.shape(), reference.shape(), "[{name}] output shape");
        let (cos, max_abs) = compare(&out, reference);
        println!("[{name:13}] cosine {cos:.7}  max|Δ| {max_abs:.5}");
        worst_cos = worst_cos.min(cos);
        assert!(
            cos > 0.999,
            "[{name}] cosine {cos} below 0.999 (max|Δ| {max_abs})"
        );
    }
    println!(
        "worst-case cosine across {} cases: {worst_cos:.7}",
        cases.len()
    );
}
