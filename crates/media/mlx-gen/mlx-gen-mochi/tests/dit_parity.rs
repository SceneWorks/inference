//! A3 full-MochiTransformer3DModel parity for Mochi 1 (sc-11987) vs the A1 real-weight golden.
//!
//! An **`#[ignore]`d** real-weight test that loads the whole AsymmDiT from `$MOCHI_SNAPSHOT`, feeds
//! the raw whole-transformer inputs from `mochi_dit_golden.safetensors` (`hidden_states`,
//! `encoder_hidden_states` = pre-caption-proj T5 embeds, `timestep`, `encoder_attention_mask`) and
//! checks the predicted velocity `noise_pred [2, 12, 2, 8, 8]` (**pre-CFG**, both `[neg, pos]`
//! branches) reproduces the golden. The committed random-weight shape/determinism gate lives in
//! `transformer::tests::full_model_forward_shapes_and_determinism`.
//!
//! Run the real-weight gate:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p mlx-gen-mochi --test dit_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, mean, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mochi::{load_transformer_weights, MochiDitConfig, MochiTransformer3DModel};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_dit_golden.safetensors"
);

fn snapshot_dir() -> std::path::PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// `max|got − want| / max|want|` — peak relative error.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    max_abs(&subtract(&got, &want).unwrap()) / max_abs(&want).max(1e-12)
}

/// `mean|got − want| / mean|want|` — mean relative error (the LTX dit_parity companion metric).
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let num = mean(&abs(&subtract(&got, &want).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let den = mean(&abs(&want).unwrap(), None).unwrap().item::<f32>();
    num / den.max(1e-12)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (bf16 DiT shards) + tools/golden/mochi_dit_golden.safetensors"]
fn transformer_forward_matches_golden() {
    let root = snapshot_dir();
    let g = Weights::from_file(GOLDEN).expect("dit golden");
    let cfg = MochiDitConfig::default();

    let w = load_transformer_weights(&root).expect("load DiT transformer weights");
    let model =
        MochiTransformer3DModel::from_weights(&w, &cfg, Dtype::Float32).expect("build DiT model");

    let hidden = g.require("hidden_states").unwrap(); // [2, 12, 2, 8, 8]
    let enc = g.require("encoder_hidden_states").unwrap(); // [2, 256, 4096] (raw T5)
    let timestep = g.require("timestep").unwrap(); // [2]
    let enc_mask = g.require("encoder_attention_mask").unwrap(); // [2, 256]

    let got = model
        .forward(hidden, enc, timestep, enc_mask)
        .expect("DiT forward");
    let want = g.require("noise_pred").unwrap();
    assert_eq!(got.shape(), want.shape(), "noise_pred shape");

    let pr = peak_rel(&got, want);
    let mr = mean_rel(&got, want);
    eprintln!("DIT noise_pred peak_rel: {pr:.3e}  mean_rel: {mr:.3e}");

    // Cross-impl f32-vs-bf16 over the full 48-block AsymmDiT (patchify + time/caption embed + attention
    // pool + 48 joint-attention blocks + norm_out). The residual reflects the reference's accumulated
    // bf16 rounding, not a structural delta — a real bug diverges orders of magnitude. peak_rel is
    // dominated by the tail of a deep bf16 stack; mean_rel is the aggregate signal (the LTX precedent).
    assert!(pr < 1.0e-1, "noise_pred peak_rel {pr:.3e} too high");
    assert!(mr < 3.0e-2, "noise_pred mean_rel {mr:.3e} too high");
}
