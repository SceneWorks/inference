//! sc-5986 — real-weight parity for the Ideogram 4 single-stream DiT against the upstream torch
//! `Ideogram4Transformer` (f32, independent graph). Exercises the interleaved 3D MRoPE, the AdaLN
//! sandwich-norm blocks, per-head q/k RMSNorm, the segment mask, the `[text;image]` token
//! composition, and the affine-less final layer.
//!
//! `#[ignore]` — needs the converted snapshot + the golden (`tools/dump_ideogram4_dit_golden.py`):
//!   CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-ideogram --test integration dit_parity:: -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::testing::{cosine_f32_device, env_path_or_home};
use mlx_gen::weights::Weights;
use mlx_gen_ideogram::load_transformer;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ideogram4_dit.safetensors"
);

fn snapshot_dir() -> PathBuf {
    env_path_or_home("IDEOGRAM4_MLX", ".cache/ideogram4-mlx-convert")
}

#[test]
#[ignore = "needs converted weights + generated golden"]
fn dit_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("golden — run tools/dump_ideogram4_dit_golden.py");
    let dit = load_transformer(&snapshot_dir()).expect("load converted transformer");
    let out = dit
        .forward(
            g.require("llm_features").unwrap(),
            g.require("x").unwrap(),
            g.require("t").unwrap(),
            g.require("position_ids").unwrap(),
            g.require("segment_ids").unwrap(),
            g.require("indicator").unwrap(),
        )
        .unwrap();
    let want = g.require("golden").unwrap();
    assert_eq!(out.shape(), want.shape(), "velocity shape");

    let c = cosine_f32_device(&out, want);
    println!("Ideogram 4 DiT parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "DiT parity cosine {c} too low — structural mismatch"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED dit_parity::dit_matches_reference");
}
