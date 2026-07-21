//! sc-2348 slice 2: Qwen2.5-VL text-encoder parity vs the frozen fork.
//!
//! `#[ignore]`d — loads the **real** `Qwen/Qwen-Image` `text_encoder/` weights from the HF cache
//! (the on-disk `model.*` layout maps onto the Rust tree under the `"model"` prefix) and the local
//! golden from `tools/dump_qwen_text_encoder_golden.py` (gitignored: fixed inputs + the fork's f32
//! encoder hidden states + drop-34 prompt embeds). Run:
//!   cargo test -p mlx-gen-qwen-image --release --test text_encoder_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::{QwenTextEncoder, QwenTextEncoderConfig};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_text_encoder_golden.safetensors"
);

/// Locate the Qwen-Image snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    let p = std::env::var("QWEN_IMAGE_SNAPSHOT").unwrap_or_else(|_| panic!("set QWEN_IMAGE_SNAPSHOT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

fn load_encoder() -> QwenTextEncoder {
    let w = Weights::from_dir(snapshot().join("text_encoder")).unwrap();
    QwenTextEncoder::from_weights(&w, "model", &QwenTextEncoderConfig::qwen_image()).unwrap()
}

#[test]
#[ignore = "needs real Qwen-Image text-encoder weights + local golden"]
fn text_encoder_hidden_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let enc = load_encoder();
    let out = enc
        .forward(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("hidden_states").unwrap();
    assert_eq!(out.shape(), want.shape(), "hidden_states shape");
    let (peak, mean) = rel_errors(&out, want);
    println!("text-encoder hidden: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    assert!(
        mean < 1e-3,
        "text-encoder hidden mean-rel regressed: {mean:.3e}"
    );
    assert!(
        peak < 2e-3,
        "text-encoder hidden peak-rel regressed: {peak:.3e}"
    );
}

#[test]
#[ignore = "needs real Qwen-Image text-encoder weights + local golden"]
fn text_encoder_prompt_embeds_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let enc = load_encoder();
    let out = enc
        .encode(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("prompt_embeds").unwrap();
    assert_eq!(out.shape(), want.shape(), "prompt_embeds shape");
    let (peak, mean) = rel_errors(&out, want);
    println!("text-encoder prompt_embeds: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    assert!(mean < 1e-3, "prompt_embeds mean-rel regressed: {mean:.3e}");
    assert!(peak < 2e-3, "prompt_embeds peak-rel regressed: {peak:.3e}");
}
