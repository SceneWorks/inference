//! **Real-tier CUDA** per-tier forward parity for the Mochi 1 quant-ingest (A6, sc-11990) — the candle
//! twin of `mlx-gen-mochi`'s `#[ignore]`d `quant_parity` real-tier residual. Gated on the `cuda`
//! feature and `#[ignore]`d (needs the `SceneWorks/mochi-1-mlx` tier tree and the
//! `mochi_dit_golden.safetensors`; a 10B DiT forward per tier is GPU-only). Loads each q4/q8/bf16 tier
//! dir through the **packed-detect seam**
//! (`transformer::QLinear::linear_detect` fires on the tier's `.scales` siblings) and checks the
//! predicted velocity `noise_pred [2, 12, 2, 8, 8]` (**pre-CFG**, both `[neg, pos]` branches) against the
//! **bf16** golden, at per-tier tolerances consistent with the MLX A6 results.
//!
//! Windows run (author-only here — do NOT run on the Mac):
//! ```sh
//! # MOCHI_MLX_TIER_DIR = the SceneWorks/mochi-1-mlx download root (holds q4/ q8/ bf16/ + shared
//! # text_encoder/ vae/ tokenizer/ siblings). Or leave unset to use ~/mochi-tiers.
//! set MOCHI_MLX_TIER_DIR=C:\models\mochi-1-mlx
//! cargo test -p candle-gen-mochi --features cuda --test tier_parity -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::Weights;
use candle_gen_mochi::tier::MochiTierPaths;
use candle_gen_mochi::{
    load_transformer_var_builder, MochiDitConfig, MochiTransformer3DModel, Pipeline, DIT_DTYPE,
};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/mochi_dit_golden.safetensors"
);

/// The `SceneWorks/mochi-1-mlx` download root (holds `q4/ q8/ bf16/` + shared components), or
/// `~/mochi-tiers`.
fn tier_root() -> PathBuf {
    if let Ok(d) = std::env::var("MOCHI_MLX_TIER_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("mochi-tiers")
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

/// Load the `<root>/<tier>` tier dir's transformer through the packed-detect seam and check the
/// predicted velocity against the bf16 golden at `(peak_bar, mean_bar)`. `expect_bits` asserts the tier
/// manifest (q4 ⇒ 4, q8 ⇒ 8, bf16 ⇒ dense/`None`).
fn tier_forward_parity(tier: &str, expect_bits: Option<i32>, peak_bar: f32, mean_bar: f32) {
    let device = candle_gen::default_device().unwrap();
    let tier_dir = tier_root().join(tier);
    if !tier_dir.join("transformer").is_dir() {
        eprintln!(
            "skip: no {tier} tier at {} (set MOCHI_MLX_TIER_DIR)",
            tier_dir.display()
        );
        return;
    }

    // Exercise the tier resolver: detect + group-64 validation + manifest bits.
    let paths = MochiTierPaths::detect(&tier_dir).expect("tier detected (split_model.json)");
    paths.validate_group_size().expect("group must be 64");
    assert_eq!(
        paths.manifest_bits().expect("manifest bits"),
        expect_bits,
        "{tier}: unexpected manifest bits"
    );

    let g = Weights::from_file(Path::new(GOLDEN), &device, DType::F32).expect("dit golden");
    let cfg = MochiDitConfig::default();
    // The packed attn/ff Linears self-detect on their `.scales` siblings; the dense bf16 tier loads dense.
    let vb = load_transformer_var_builder(&tier_dir, DIT_DTYPE, &device).expect("load tier DiT");
    let model = MochiTransformer3DModel::new(vb, &cfg, &device).expect("build tier DiT");

    let hidden = g.require("hidden_states").unwrap(); // [2, 12, 2, 8, 8]
    let enc = g.require("encoder_hidden_states").unwrap(); // [2, 256, 4096] raw T5
    let timestep = g.require("timestep").unwrap(); // [2]
    let enc_mask = g.require("encoder_attention_mask").unwrap(); // [2, 256]
    let got = model
        .forward(&hidden, &enc, &timestep, &enc_mask)
        .expect("tier DiT forward");
    let want = g.require("noise_pred").unwrap();
    assert_eq!(got.dims(), want.dims(), "noise_pred shape");
    assert!(
        got.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "{tier}: non-finite velocity (a structural break, not quant loss)"
    );

    let pr = peak_rel(&got, &want);
    let mr = mean_rel(&got, &want);
    eprintln!("[{tier} tier] noise_pred vs bf16 golden — peak_rel={pr:.3e} mean_rel={mr:.3e}");
    assert!(
        pr < peak_bar,
        "{tier}: peak_rel {pr:.3e} exceeds {peak_bar:.3e}"
    );
    assert!(
        mr < mean_bar,
        "{tier}: mean_rel {mr:.3e} exceeds {mean_bar:.3e}"
    );
}

/// **bf16 tier = dense**: the tier repacks the transformer bf16 with no quantization, so the parity vs
/// the golden matches the dense `dit_parity` bars (peak < 1.0e-1, mean < 3.0e-2).
#[test]
#[ignore = "needs the mochi-1-mlx bf16 tier (MOCHI_MLX_TIER_DIR) + the CUDA golden"]
fn bf16_tier_forward_matches_golden() {
    tier_forward_parity("bf16", None, 1.0e-1, 3.0e-2);
}

/// **q8 tier ≈ dense**: the Q8 packed attn/ff Linears add only the accepted `Q8_0` double-quant (Phase-2
/// dequant compat ~0.4% per weight), so parity stays close to dense (~5.6e-2, MLX A6) — bars with margin.
#[test]
#[ignore = "needs the mochi-1-mlx q8 tier (MOCHI_MLX_TIER_DIR) + the CUDA golden"]
fn q8_tier_forward_matches_golden() {
    tier_forward_parity("q8", Some(8), 1.2e-1, 6.0e-2);
}

/// **q4 tier is lossy-bounded**: Q4 on a 10B DiT is genuinely lossy — a bounded perturbation of the bf16
/// velocity (~1.1e-1 mean / ~1.9e-1 peak, MLX A6), NOT bit-exact. Bars record that bound (with margin);
/// a structural break (wrong predicate / scale / transpose / NaN) is orders of magnitude larger.
#[test]
#[ignore = "needs the mochi-1-mlx q4 tier (MOCHI_MLX_TIER_DIR) + the CUDA golden"]
fn q4_tier_forward_matches_golden() {
    tier_forward_parity("q4", Some(4), 2.0e-1, 1.2e-1);
}

/// End-to-end **load** of the q4 tier through `Pipeline::load_components`: exercises the tier resolver
/// (packed DiT from the tier dir + the shared T5-XXL / AsymmVAE resolved from the tier dir's parent) on
/// the real tree. Heavy (materializes the shared T5), so `#[ignore]`d; no denoise.
#[test]
#[ignore = "loads the whole q4 tier (packed DiT + shared T5/VAE) — needs the mochi-1-mlx tree"]
fn q4_tier_loads_components_end_to_end() {
    let device = candle_gen::default_device().unwrap();
    let tier_dir = tier_root().join("q4");
    if !tier_dir.join("transformer").is_dir() {
        eprintln!(
            "skip: no q4 tier at {} (MOCHI_MLX_TIER_DIR)",
            tier_dir.display()
        );
        return;
    }
    let _components = Pipeline::new(&tier_dir, &device)
        .load_components()
        .unwrap_or_else(|e| panic!("load q4 tier components from {}: {e}", tier_dir.display()));
    eprintln!(
        "OK: q4 tier loaded end-to-end (packed DiT + shared T5/VAE) from {}",
        tier_dir.display()
    );
}
