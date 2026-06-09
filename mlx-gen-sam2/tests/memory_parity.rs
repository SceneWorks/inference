//! sc-3713 — SAM2 memory layer (memory encoder + memory attention) parity vs the MLX-native
//! reference (`avbiswas/sam2-mlx`, the impl this crate ports).
//!
//! Golden: `tools/dump_sam2_memory_golden.py` (reference forward passes on fixed random fixtures —
//! a 3-frame memory bank + 2 object pointers for the attention path). Both run MLX Metal, so parity
//! is near-bit. Validates: the mask downsampler, the depthwise-conv ConvNeXt fuser, the 64-feature
//! sinusoidal position encoding (memory encoder), and the interleaved axial RoPE self/cross
//! attention with key-repeat + object-pointer RoPE exclusion (memory attention).
//!
//! Run:
//!   PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python tools/dump_sam2_memory_golden.py --size large
//!   cargo test -p mlx-gen-sam2 --release --test memory_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{MemoryAttention, MemoryEncoder};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sam2_memory_golden_large.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_sam2_memory_golden.py --size large first.")
    })
}

fn flat(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.reshape(&[n])
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// `(cosine, max-rel, mean-rel)` of `got` vs `want`.
fn metrics(got: &Array, want: &Array) -> (f32, f32, f32) {
    let a = flat(got);
    let b = flat(want);
    assert_eq!(
        a.len(),
        b.len(),
        "shape {:?} vs {:?}",
        got.shape(),
        want.shape()
    );
    let peak_ref = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_ref: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_diff: f64 = a.iter().zip(&b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
    (cos, max_diff / peak_ref, (sum_diff / sum_ref) as f32)
}

#[test]
#[ignore = "needs local golden from tools/dump_sam2_memory_golden.py --size large"]
fn memory_encoder_matches_mlx_reference_large() {
    let g = golden();
    let enc = MemoryEncoder::from_weights(&g, "memory_encoder").unwrap();
    let out = enc
        .forward(
            g.require("mem_pix_feat").unwrap(),
            g.require("mem_masks").unwrap(),
            true,
        )
        .unwrap();

    assert_eq!(out.vision_features.shape(), &[1, 64, 64, 64]);
    let (cos, peak, mean) = metrics(&out.vision_features, g.require("mem_vis_features").unwrap());
    println!("memory_encoder features: cos {cos:.7} peak-rel {peak:.3e} mean-rel {mean:.3e}");
    assert!(cos > 0.9999, "features cosine {cos:.7}");
    assert!(mean < 2e-3, "features mean-rel {mean:.3e}");

    let (cos_p, _, mean_p) = metrics(&out.vision_pos_enc, g.require("mem_vis_pos").unwrap());
    println!("memory_encoder pos_enc: cos {cos_p:.7} mean-rel {mean_p:.3e}");
    assert!(cos_p > 0.99999, "pos_enc cosine {cos_p:.7}");
}

#[test]
#[ignore = "needs local golden from tools/dump_sam2_memory_golden.py --size large"]
fn memory_attention_matches_mlx_reference_large() {
    let g = golden();
    let attn = MemoryAttention::from_weights(&g, "memory_attention").unwrap();
    let num_obj = flat(g.require("ma_num_obj").unwrap())[0] as i32;

    let out = attn
        .forward(
            g.require("ma_curr").unwrap(),
            g.require("ma_curr_pos").unwrap(),
            g.require("ma_mem").unwrap(),
            g.require("ma_mem_pos").unwrap(),
            num_obj,
        )
        .unwrap();

    assert_eq!(out.shape(), &[64 * 64, 1, 256]);
    let (cos, peak, mean) = metrics(&out, g.require("ma_out").unwrap());
    println!("memory_attention out: cos {cos:.7} peak-rel {peak:.3e} mean-rel {mean:.3e} (num_obj {num_obj})");
    assert!(cos > 0.999, "attention cosine {cos:.7}");
    assert!(mean < 5e-3, "attention mean-rel {mean:.3e}");
}
