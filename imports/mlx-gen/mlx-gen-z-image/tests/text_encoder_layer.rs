//! sc-2344: Z-Image text-encoder sub-module parity vs the fork (tiny random config).
//!
//! Fixture `tests/fixtures/text_encoder_layer.safetensors` ← `tools/dump_z_image_text_encoder_layer.py`.
//! Covers (1) the HF half-split RoPE cos/sin and (2) a full `EncoderLayer` forward (GQA +
//! q_norm/k_norm + RoPE + causal SDPA + SwiGLU MLP + pre-norm residuals). 1e-2 tolerance —
//! Metal runs fp32 matmul in reduced precision.

use mlx_gen::weights::Weights;
use mlx_gen_z_image::text_encoder::{EncoderLayer, TextRope};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/text_encoder_layer.safetensors"
);

fn close(a: &Array, b: &Array) -> bool {
    all_close(a, b, 1e-2, 1e-2, false).unwrap().item::<bool>()
}

/// cfg = "H,NH,NKV,HD,INTER,SEQ,EPS,THETA"
fn cfg(w: &Weights) -> (i32, i32, i32, i32, i32, i32, f32, f32) {
    let p: Vec<&str> = w.metadata("cfg").unwrap().split(',').collect();
    (
        p[0].parse().unwrap(),
        p[1].parse().unwrap(),
        p[2].parse().unwrap(),
        p[3].parse().unwrap(),
        p[4].parse().unwrap(),
        p[5].parse().unwrap(),
        p[6].parse().unwrap(),
        p[7].parse().unwrap(),
    )
}

#[test]
fn rope_cos_sin_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let (_h, _nh, _nkv, hd, _i, seq, _eps, theta) = cfg(&w);

    let (cos, sin) = TextRope::new(hd, theta).forward(seq).unwrap();
    assert!(close(&cos, w.require("cos").unwrap()), "RoPE cos diverged");
    assert!(close(&sin, w.require("sin").unwrap()), "RoPE sin diverged");
}

#[test]
fn encoder_layer_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let (_h, nh, nkv, hd, _i, _seq, eps, _theta) = cfg(&w);

    // Flat fixture keys (empty prefix).
    let layer = EncoderLayer::from_weights(&w, "", nh, nkv, hd, eps).unwrap();
    let out = layer
        .forward(
            w.require("in").unwrap(),
            w.require("cos").unwrap(),
            w.require("sin").unwrap(),
            w.require("mask").unwrap(),
        )
        .unwrap();

    assert!(
        close(&out, w.require("out").unwrap()),
        "EncoderLayer forward diverged from the fork"
    );
}
