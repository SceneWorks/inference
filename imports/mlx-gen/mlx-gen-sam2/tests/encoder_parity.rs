//! sc-3705 — SAM2 image-encoder (Hiera trunk + FPN neck) parity vs the MLX-native reference.
//!
//! Golden: `tools/dump_sam2_encoder_golden.py` (runs `avbiswas/sam2-mlx`'s `Sam2ImageEncoder` — the
//! impl this crate ports — on a fixed input, bundling encoder weights + input + reference outputs).
//! Both sides run the *same* MLX Metal kernels (Python reference vs this Rust port), so parity is
//! near-bit, not the looser cross-backend (torch↔MLX) floor: cosine ≈ 1, mean-rel ≪ 1e-3.
//!
//! Run:
//!   PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python tools/dump_sam2_encoder_golden.py --size large
//!   cargo test -p mlx-gen-sam2 --release --test encoder_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ImageEncoder, Sam2ImageEncoderConfig};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sam2_encoder_golden_large.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_sam2_encoder_golden.py --size large first.")
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

/// (peak relative error, mean relative error) of `got` vs reference `want`.
fn rel_errors(got: &Array, want: &Array) -> (f32, f32) {
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
    (max_diff / peak_ref, (sum_diff / sum_ref) as f32)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let a = flat(got);
    let b = flat(want);
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn assert_parity(got: &Array, want: &Array, label: &str) {
    assert_eq!(
        got.shape(),
        want.shape(),
        "{label} shape {:?} vs {:?}",
        got.shape(),
        want.shape()
    );
    let (peak, mean) = rel_errors(got, want);
    let cos = cosine(got, want);
    println!(
        "{label}: shape {:?} cos {cos:.7} peak-rel {peak:.3e} mean-rel {mean:.3e}",
        got.shape()
    );
    assert!(cos > 0.9999, "{label} cosine {cos:.7}");
    assert!(mean < 1e-3, "{label} mean-rel {mean:.3e}");
}

#[test]
#[ignore = "needs local golden from tools/dump_sam2_encoder_golden.py --size large"]
fn encoder_matches_mlx_reference_large() {
    let g = golden();
    let enc = Sam2ImageEncoder::from_weights(&g, &Sam2ImageEncoderConfig::large()).unwrap();
    let out = enc.forward(g.require("enc_in").unwrap()).unwrap();

    // backbone_fpn: 3 levels, fine→coarse.
    assert_eq!(out.backbone_fpn.len(), 3, "expected 3 FPN levels");
    for i in 0..3 {
        assert_parity(
            &out.backbone_fpn[i],
            g.require(&format!("ref_backbone_fpn_{i}")).unwrap(),
            &format!("backbone_fpn_{i}"),
        );
        assert_parity(
            &out.vision_pos_enc[i],
            g.require(&format!("ref_pos_{i}")).unwrap(),
            &format!("vision_pos_enc_{i}"),
        );
    }
    assert_parity(
        &out.vision_features,
        g.require("ref_vision_features").unwrap(),
        "vision_features",
    );
}
