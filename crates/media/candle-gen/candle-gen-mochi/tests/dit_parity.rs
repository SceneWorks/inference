//! **Real-weight CUDA** full-`MochiTransformer3DModel` parity for Mochi 1 (A5, sc-11989) — the candle
//! twin of `mlx-gen-mochi`'s ignored `dit_parity` gate. Gated on `feature = "cuda"` + `#[ignore]`d.
//! Feeds the raw whole-transformer inputs from `mochi_dit_golden.safetensors` and checks the predicted
//! velocity `noise_pred [2, 12, 2, 8, 8]` (**pre-CFG**, both `[neg, pos]` branches) reproduces the
//! golden. The random-weight shape/determinism gate lives in `transformer::tests`.
//!
//! Windows run:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p candle-gen-mochi --features cuda --test dit_parity -- --ignored --nocapture`
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::Weights;
use candle_gen_mochi::{
    load_transformer_var_builder, MochiDitConfig, MochiTransformer3DModel, DIT_DTYPE,
};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/mochi_dit_golden.safetensors"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn max_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}
fn mean_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// `max|got − want| / max|want|` — peak relative error.
fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let got = got.to_dtype(DType::F32).unwrap();
    let want = want.to_dtype(DType::F32).unwrap();
    max_abs(&(&got - &want).unwrap()) / max_abs(&want).max(1e-12)
}

/// `mean|got − want| / mean|want|` — mean relative error.
fn mean_rel(got: &Tensor, want: &Tensor) -> f32 {
    let got = got.to_dtype(DType::F32).unwrap();
    let want = want.to_dtype(DType::F32).unwrap();
    mean_abs(&(&got - &want).unwrap()) / mean_abs(&want).max(1e-12)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (bf16 DiT shards) + tools/golden/mochi_dit_golden.safetensors (CUDA)"]
fn transformer_forward_matches_golden() {
    let device = candle_gen::default_device().unwrap();
    let root = snapshot_dir();
    let g = Weights::from_file(Path::new(GOLDEN), &device, DType::F32).expect("dit golden");
    let cfg = MochiDitConfig::default();

    let vb = load_transformer_var_builder(&root, DIT_DTYPE, &device).expect("load DiT weights");
    let model = MochiTransformer3DModel::new(vb, &cfg, &device).expect("build DiT model");

    let hidden = g.require("hidden_states").unwrap(); // [2, 12, 2, 8, 8]
    let enc = g.require("encoder_hidden_states").unwrap(); // [2, 256, 4096] (raw T5)
    let timestep = g.require("timestep").unwrap(); // [2]
    let enc_mask = g.require("encoder_attention_mask").unwrap(); // [2, 256]

    let got = model
        .forward(&hidden, &enc, &timestep, &enc_mask)
        .expect("DiT forward");
    let want = g.require("noise_pred").unwrap();
    assert_eq!(got.dims(), want.dims(), "noise_pred shape");

    let pr = peak_rel(&got, &want);
    let mr = mean_rel(&got, &want);
    eprintln!("DIT noise_pred peak_rel: {pr:.3e}  mean_rel: {mr:.3e}");

    // Cross-impl f32-activation vs the reference's bf16 over the full 48-block AsymmDiT. peak_rel is
    // dominated by the tail of a deep bf16 stack; mean_rel is the aggregate signal (the MLX bars).
    assert!(pr < 1.0e-1, "noise_pred peak_rel {pr:.3e} too high");
    assert!(mr < 3.0e-2, "noise_pred mean_rel {mr:.3e} too high");
}
