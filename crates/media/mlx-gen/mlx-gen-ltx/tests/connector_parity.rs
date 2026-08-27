//! S1 connector parity vs the reference `Embeddings1DConnector` (sc-2679 S1; re-derived sc-21663).
//!
//! `#[ignore]`d: needs the real 2.3 `connector.safetensors` (~6.3 GB; the `SceneWorks/ltx-2.3-mlx`
//! bf16 tier carries the identical tensors). The committed golden
//! (`tests/fixtures/ltx_connector_golden.safetensors`, from `tools/dump_ltx_connector_golden.py`)
//! holds the reference f32 input/mask/output — **ltx_core semantics** (`2·sigmoid` gate,
//! tanh-GELU, f32-quantized RoPE indices) since sc-21663; this test loads the SAME connector
//! weights and checks the Rust `Connector` reproduces the video/audio embeddings.
//!
//! # Bars (sc-21663) — derivation from the measured per-op decomposition
//!
//! The audio widening is NOT an amplification story and NOT an absorbed defect. The full per-op
//! decomposition (port vs the patched-MLX oracle, identical inputs, all f32, audio connector on
//! this fixture) measured:
//!
//! ```text
//! rope tables               6.0e-8      (1 f32 ULP — bit-matched construction)
//! b0 q/k-norm, v, sdpa,
//!   gate, to_out, ff        1.0e-4 … 7.5e-4   (GEMM/kernel accumulation differences: the pmetal
//!                                              fork's Metal kernels vs the wheel's — measurably
//!                                              ~1e-4-class per op, NOT 1e-7; torch-CPU vs MLX
//!                                              GEMMs sit in the same class)
//! block_0 … block_6         ~5e-4 FLAT        (per-op deviations do NOT compound per layer)
//! block_7                   3.3e-3
//! final rms_norm            8.086e-2          (the entire jump happens AT the renormalization)
//! ```
//!
//! The mechanism of the final jump is the row-norm dynamic range, not any op: at the last block
//! the audio rows' RMS spans `1.74 … 473` (272x; video: 134x), and the closing per-row RMS-norm
//! rescales every row to unit size — converting absolute-scale kernel noise into large
//! *relative* error on the near-cancelled low-norm register rows (the worst row is a register
//! row; `corr(log rowRMS, log rowErr) = -0.62`). No implementation pair escapes this: stock torch
//! ltx_core differs from the patched-MLX oracle by `6.4e-2` audio-global on this same fixture,
//! and even the oracle's own rope-table precision (f64 vs f32-quantized indices, everything else
//! identical) moves its audio output by `1.2e-2`. The old 5e-3 audio bar was attainable only
//! because the σ-gated (bugged) connector produced a different register-row norm distribution.
//!
//! Bars, from the measured values: video keeps the historical `5e-3` global (measured `2.756e-3`);
//! audio splits into valid rows `< 3e-2` (measured `1.639e-2` — the rows carrying prompt
//! information) and global `< 1.2e-1` (measured `8.086e-2`, register rows).
//!
//! Reconciliation with the LTX-2.5 outputs gate (`ltx_2_5_te_connector_inputs.rs`: video
//! `9.107e-3` / audio `1.141e-2` against a torch-f32 oracle): the register-row norm distribution
//! is a property of the weights AND the inputs — the 2.5 checkpoint's audio connector on the real
//! prompt cancels its register rows far less than the 2.3 checkpoint does on this fixture's
//! random features, so the video/audio gap here does not contradict the near-parity there.
//!
//! Run: `LTX_EROS_DIR=… cargo test -p mlx-gen-ltx --test integration connector_parity:: -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::connector::Connector;
use mlx_gen_ltx::transformer::Precision;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_connector_golden.safetensors"
);

fn eros_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_EROS_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_eros")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// [`peak_rel`] over token rows `[lo, hi)`, denominator taken over the same slice — so the audio
/// valid-row bar cannot hide behind (or be drowned by) the register rows (see the module docs).
fn peak_rel_rows(got: &Array, want: &Array, lo: i32, hi: i32) -> f32 {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    let g = got.take_axis(idx.clone(), 1).expect("slice got");
    let w = want.take_axis(idx, 1).expect("slice want");
    peak_rel(&g, &w)
}

#[test]
#[ignore = "needs eros connector.safetensors (~6.3 GB)"]
fn connector_matches_reference() {
    let dir = eros_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let w = Weights::from_file(dir.join("connector.safetensors")).expect("connector.safetensors");
    let conn = Connector::from_weights(
        &w,
        "video_embeddings_connector.",
        &cfg,
        // f32 activations (the isolated bit-exact gate) at the checkpoint's quant geometry; the
        // 2.3 connector ships dense, so every Linear takes the dense arm.
        Precision::quant_f32(split.bits, split.group),
    )
    .expect("build");

    let g = Weights::from_file(GOLDEN).expect("golden");
    let features = g.require("features").unwrap();
    let mask01 = g.require("mask01").unwrap();
    let want = g.require("video_embeddings").unwrap();

    let got = conn.forward(features, mask01).expect("forward");
    assert_eq!(got.shape(), want.shape());
    let pr = peak_rel(&got, want);
    eprintln!("connector peak_rel = {pr:.3e}");
    // f32 Rust vs f32 reference (both f64 rope → f32, f32 sdpa) → tight.
    assert!(pr < 5e-3, "connector peak_rel {pr:.3e} too high");
}

#[test]
#[ignore = "needs eros connector.safetensors (~6.3 GB)"]
fn audio_connector_matches_reference() {
    // sc-2684: the audio connector is the same architecture at audio dims (32 × 64 = 2048).
    let dir = eros_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let w = Weights::from_file(dir.join("connector.safetensors")).expect("connector.safetensors");
    let conn = Connector::from_weights_dims(
        &w,
        "audio_embeddings_connector.",
        cfg.connector_num_layers,
        cfg.audio_connector_num_attention_heads,
        cfg.audio_connector_attention_head_dim,
        cfg.positional_embedding_theta,
        cfg.connector_positional_embedding_max_pos,
        cfg.connector_ff_bias,
        Precision::quant_f32(split.bits, split.group),
    )
    .expect("build audio connector");

    let g = Weights::from_file(GOLDEN).expect("golden");
    let features = g.require("audio_features").unwrap();
    let mask01 = g.require("mask01").unwrap();
    let want = g.require("audio_embeddings").unwrap();

    let got = conn.forward(features, mask01).expect("forward");
    assert_eq!(got.shape(), want.shape());
    // The connector reorders its input, so the valid rows are the PREFIX (`0..nv`).
    let nv = mlx_rs::ops::sum(mask01, None).unwrap().item::<i32>();
    let pr = peak_rel(&got, want);
    let pr_valid = peak_rel_rows(&got, want, 0, nv);
    eprintln!("audio connector peak_rel = {pr:.3e} (valid rows {pr_valid:.3e})");
    // Split bars — measured cross-implementation floors of the final-norm row-renormalization on
    // this fixture's register-row norm distribution; see the module docs for the per-op
    // decomposition (valid floor 1.64e-2, register/global floor 8.09e-2).
    assert!(
        pr_valid < 3e-2,
        "audio connector valid-row peak_rel {pr_valid:.3e} too high"
    );
    assert!(pr < 1.2e-1, "audio connector peak_rel {pr:.3e} too high");
}
