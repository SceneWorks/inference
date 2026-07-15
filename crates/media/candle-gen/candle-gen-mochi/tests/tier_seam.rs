//! **Weightless** CI-green gate for the Mochi 1 quant-ingest seam (A6, sc-11990) — the candle twin of
//! candle-wan's `candle_tier_build` round-trip and the non-ignored half of the MLX A6 `quant_parity`.
//! No model weights: synthetic tensors packed with the shared MLX-affine packer, on the **real** Mochi
//! DiT key layout, loaded back through [`QLinear::linear_detect`] (the exact seam
//! `crate::transformer` wires). Proves (1) the packed-detect fires on a `.scales` sibling and the
//! dequant-on-forward reproduces the dense weight within Q4/Q8 quant error (cosine > 0.997), (2) a leaf
//! with no `.scales` stays dense, and (3) the dense arm's `forward_upcast` (bf16 weight, f32 activation)
//! is **byte-identical** to the pre-seam `nn::linear_nb` path (the parity regime the goldens were blessed
//! in). The real-tier CUDA forward-parity gate is the `#[ignore]`d `tier_parity.rs`.

use std::collections::HashMap;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::quant::{pack_mlx_affine, DenseLinear, QLinear, MLX_GROUP_SIZE};

/// Deterministic well-conditioned `[out, in]` weight (seeded, not `randn` — portable FP so the cosine
/// parity doesn't flap around the Q4 loss floor on CI, per candle-wan's note).
fn dense(out_dim: usize, in_dim: usize, seed: f32) -> Tensor {
    let data: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i as f32 + seed) * 0.013).sin() * 1.3)
        .collect();
    Tensor::from_vec(data, (out_dim, in_dim), &Device::Cpu).unwrap()
}

/// Deterministic `[b, s, in]` activation stream.
fn acts(b: usize, s: usize, in_dim: usize, seed: f32) -> Tensor {
    let data: Vec<f32> = (0..b * s * in_dim)
        .map(|i| ((i as f32 + seed) * 0.017).cos() * 0.9)
        .collect();
    Tensor::from_vec(data, (b, s, in_dim), &Device::Cpu).unwrap()
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let a = a.to_vec1::<f32>().unwrap();
    let b = b.to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.to_dtype(DType::F32)
        .unwrap()
        .sub(&b.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// Pack a synthetic dense weight on the real Mochi DiT key, load it back through the seam, and check the
/// packed-detect fires and the dequant-on-forward is a bounded (cosine > 0.997) perturbation of dense.
fn packed_linear_round_trips(bits: usize) -> Result<()> {
    let dev = Device::Cpu;
    // `to_q`-shaped: in = out = a multiple of the group 64.
    let (out_dim, in_dim) = (192usize, 256usize);
    let base = "transformer_blocks.0.attn1.to_q";
    let w = dense(out_dim, in_dim, 7.0);
    let (wq, scales, biases) = pack_mlx_affine(&w, bits, MLX_GROUP_SIZE)?;

    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert(format!("{base}.weight"), wq);
    map.insert(format!("{base}.scales"), scales);
    map.insert(format!("{base}.biases"), biases);
    // A dense sibling (no `.scales`) — must take the dense arm.
    map.insert(
        "transformer_blocks.0.attn1.norm_q.weight".into(),
        dense(1, 128, 3.0),
    );

    let tmp = std::env::temp_dir().join(format!(
        "sc11990_seam_q{bits}_{}.safetensors",
        std::process::id()
    ));
    candle_gen::candle_core::safetensors::save(&map, &tmp)?;
    // SAFETY: freshly written, single-reader for the test.
    let st = unsafe { MmapedSafetensors::new(&tmp)? };
    let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

    // (1) packed-detect fires.
    let packed = QLinear::linear_detect(in_dim, out_dim, &vb, base, false)?;
    assert!(
        packed.is_quantized(),
        "Q{bits}: `.scales` under {base} must fire the packed path"
    );

    // (2) the dequant-on-forward is a bounded perturbation of the dense weight.
    let dense_lin = Linear::new(w, None);
    let x = acts(2, 5, in_dim, 11.0);
    let cos = cosine(&packed.forward(&x)?, &dense_lin.forward(&x)?);
    assert!(
        cos > 0.997,
        "Q{bits}: packed-load forward cosine {cos:.6} vs dense too low"
    );

    std::fs::remove_file(&tmp).ok();
    Ok(())
}

#[test]
fn q4_packed_linear_round_trips_through_the_seam() -> Result<()> {
    packed_linear_round_trips(4)
}

#[test]
fn q8_packed_linear_round_trips_through_the_seam() -> Result<()> {
    packed_linear_round_trips(8)
}

/// A leaf with **no** `.scales` sibling loads dense through the seam (`linear_detect` dense arm) — the
/// current raw diffusers checkpoint keeps every DiT Linear dense, byte-identical to before.
#[test]
fn dense_leaf_stays_dense() -> Result<()> {
    let dev = Device::Cpu;
    let (out_dim, in_dim) = (64usize, 128usize);
    let w = dense(out_dim, in_dim, 1.0);
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("transformer_blocks.0.attn1.to_q.weight".into(), w);

    let tmp =
        std::env::temp_dir().join(format!("sc11990_dense_{}.safetensors", std::process::id()));
    candle_gen::candle_core::safetensors::save(&map, &tmp)?;
    // SAFETY: freshly written, single-reader.
    let st = unsafe { MmapedSafetensors::new(&tmp)? };
    let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

    let lin = QLinear::linear_detect(
        in_dim,
        out_dim,
        &vb,
        "transformer_blocks.0.attn1.to_q",
        false,
    )?;
    assert!(!lin.is_quantized(), "no `.scales` ⇒ dense arm");

    std::fs::remove_file(&tmp).ok();
    Ok(())
}

/// **The byte-identical dense-path pin.** The seam swaps the DiT attn/ff projections from the raw
/// `nn::linear_nb` (flatten → upcast bf16 weight → matmul) to `QLinear::Dense::forward_upcast`. On a
/// bf16-stored weight with f32 activations (the exact Mochi parity regime) the two must be **bit-for-bit
/// equal** — otherwise the dense goldens would drift. A rank-3 `[B, S, in]` activation (the attention
/// case) exercises candle's batched-broadcast `Linear::forward` vs the flattened `linear_nb`.
#[test]
fn dense_forward_upcast_is_byte_identical_to_linear_nb() -> Result<()> {
    let (out_dim, in_dim) = (96usize, 128usize);
    let w_bf16 = dense(out_dim, in_dim, 5.0).to_dtype(DType::BF16)?;
    let x = acts(2, 7, in_dim, 9.0); // f32, rank-3

    // Pre-seam path (`nn::linear_nb` returns the crate's `CandleError`; bridge it into this test's
    // `candle_core::Result`).
    let want = candle_gen_mochi::nn::linear_nb(&x, &w_bf16)
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    // Seam path (dense arm), bias-less.
    let lin = QLinear::from_dense(DenseLinear::Linear(Linear::new(w_bf16, None)));
    let got = lin.forward_upcast(&x)?;

    assert_eq!(got.dims(), want.dims());
    let d = max_abs_diff(&got, &want);
    assert_eq!(
        d, 0.0,
        "dense forward_upcast must be byte-identical to linear_nb (max|Δ|={d:.3e})"
    );
    Ok(())
}
