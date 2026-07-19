//! **Real-weight CUDA** masked-T5 parity for Mochi 1 (A5, sc-11989) — the candle twin of
//! `mlx-gen-mochi`'s ignored `te_parity` gate. Gated on `feature = "cuda"` (compiles only on the
//! Windows/CUDA lane) and `#[ignore]`d (needs `$MOCHI_SNAPSHOT` + the gitignored A1 golden). Runs the
//! full masked encode and checks `prompt_embeds` (masked real-token `peak_rel`) + both attention masks
//! reproduce `mochi_te_golden.safetensors`.
//!
//! Windows run:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p candle-gen-mochi --features cuda --test te_parity -- --ignored --nocapture`
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::Weights;
use candle_gen_mochi::{encode_prompt, load_tokenizer, MochiT5};

/// The exact prompt the A1 dump harness blessed the golden with.
const PROMPT: &str = "A calico kitten batting a ball of red yarn across a sunlit wooden floor.";
/// The A1 harness used an empty negative prompt.
const NEGATIVE: &str = "";

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/mochi_te_golden.safetensors"
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

/// `max|got − want| / max|want|` — peak relative error.
fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let got = got.to_dtype(DType::F32).unwrap();
    let want = want.to_dtype(DType::F32).unwrap();
    max_abs(&(&got - &want).unwrap()) / max_abs(&want).max(1e-12)
}

/// Peak relative error over the **real (unmasked) token rows only** — zero out padded positions in
/// both tensors via `mask [1, L]`. Padded rows carry near-zero signal and are masked out downstream,
/// so a full-tensor `peak_rel` is dominated by padding-noise (worst on the empty negative).
fn masked_peak_rel(got: &Tensor, want: &Tensor, mask: &Tensor) -> f32 {
    let (got, want) = masked_pair(got, want, mask);
    peak_rel(&got, &want)
}

/// Zero out padded token rows in both tensors via `mask [1, L]`.
fn masked_pair(got: &Tensor, want: &Tensor, mask: &Tensor) -> (Tensor, Tensor) {
    let l = mask.dim(1).unwrap();
    let m = mask
        .to_dtype(DType::F32)
        .unwrap()
        .reshape((1, l, 1))
        .unwrap();
    let got = got.to_dtype(DType::F32).unwrap().broadcast_mul(&m).unwrap();
    let want = want
        .to_dtype(DType::F32)
        .unwrap()
        .broadcast_mul(&m)
        .unwrap();
    (got, want)
}

/// Mean relative error over the **real (unmasked)** token rows: `mean|got − want| / max|want|`,
/// averaged over the real elements only (padded rows contribute neither error nor denominator).
fn masked_mean_rel(got: &Tensor, want: &Tensor, mask: &Tensor) -> f32 {
    let (got, want) = masked_pair(got, want, mask);
    let n_real = mask
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        * want.dim(2).unwrap() as f32;
    let sum_abs = (&got - &want)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    (sum_abs / n_real.max(1.0)) / max_abs(&want).max(1e-12)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (T5-XXL shards) + tools/golden/mochi_te_golden.safetensors (CUDA)"]
fn t5_encode_matches_golden() {
    let device = candle_gen::default_device().unwrap();
    let root = snapshot_dir();
    let tok = load_tokenizer().unwrap();
    // The production regime: bf16 weights, f32 activations (MLX's T5 regime; see
    // `MochiT5::load_reference_regime`).
    let t5 =
        MochiT5::load_reference_regime(&root.join("text_encoder"), &device).expect("load T5-XXL");
    let g = Weights::from_file(Path::new(GOLDEN), &device, DType::F32).expect("te golden");

    let pos = encode_prompt(&tok, &t5, PROMPT, &device).expect("encode prompt");
    let neg = encode_prompt(&tok, &t5, NEGATIVE, &device).expect("encode negative");

    // Masks are exact 0/1 — must match bit-for-bit.
    let pos_mask_rel = peak_rel(
        &pos.prompt_attention_mask,
        &g.require("prompt_attention_mask").unwrap(),
    );
    let neg_mask_rel = peak_rel(
        &neg.prompt_attention_mask,
        &g.require("negative_prompt_attention_mask").unwrap(),
    );
    eprintln!("TE masks: prompt {pos_mask_rel:.3e}  negative {neg_mask_rel:.3e}");
    assert_eq!(pos_mask_rel, 0.0, "prompt_attention_mask must be exact");
    assert_eq!(
        neg_mask_rel, 0.0,
        "negative_prompt_attention_mask must be exact"
    );

    // Embeds: cross-BACKEND (candle-CUDA vs a golden blessed on the diffusers reference, itself
    // reproduced under MLX-Metal), over 24 T5 blocks. We are in the reference's exact weight regime
    // (the fp32 shards are bf16-trained, so bf16-resident weights are bit-identical to the golden's)
    // with f32 activations, so what remains is kernel/device noise — not a code delta.
    //
    // BOTH bars are load-bearing, and each catches what the other misses — measured, not assumed:
    //
    //                            peak_rel    mean_rel     verdict
    //   this regime (bf16 w/f32 acts)  6.044e-2    6.007e-4    pass
    //   all-bf16 (the A5 dtype bug)    1.444e-1    8.751e-4    FAIL on peak only
    //
    //  - `peak_rel < 8e-2` DISCRIMINATES the dtype regime. Note the all-bf16 bug slips *under* the
    //    3e-3 mean bar (8.75e-4) — mean_rel alone would have shipped it. 8e-2 sits in the real gap
    //    between 6.04e-2 (correct) and 1.444e-1 (broken), with margin on both sides; it is NOT the
    //    6e-2 MLX bar nudged to accommodate this run.
    //  - `mean_rel < 3e-3` catches *systematic* drift (wrong mask, transposed weight, wrong bucket)
    //    that a peak bar could miss, and is the signal that says this encoder is correct: 100× inside,
    //    on 90,112 real elements.
    //
    // Why not the 6e-2 MLX bar: it was blessed under MLX-Metal, and this is candle-CUDA. We are in the
    // reference's exact weight regime, so the 6.04e-2 is cross-backend kernel noise decided by ONE
    // element in 90,112 — a bare peak bar at 6e-2 has no cross-backend headroom and flips on any
    // unrelated kernel change. See sc-11989 #12112/#12115/#12116/#12119.
    let pos_rel = masked_peak_rel(
        &pos.prompt_embeds,
        &g.require("prompt_embeds").unwrap(),
        &g.require("prompt_attention_mask").unwrap(),
    );
    let neg_rel = masked_peak_rel(
        &neg.prompt_embeds,
        &g.require("negative_prompt_embeds").unwrap(),
        &g.require("negative_prompt_attention_mask").unwrap(),
    );
    let pos_mean = masked_mean_rel(
        &pos.prompt_embeds,
        &g.require("prompt_embeds").unwrap(),
        &g.require("prompt_attention_mask").unwrap(),
    );
    let neg_mean = masked_mean_rel(
        &neg.prompt_embeds,
        &g.require("negative_prompt_embeds").unwrap(),
        &g.require("negative_prompt_attention_mask").unwrap(),
    );
    eprintln!("TE embeds real-token peak_rel: prompt {pos_rel:.3e}  negative {neg_rel:.3e}");
    eprintln!("TE embeds real-token mean_rel: prompt {pos_mean:.3e}  negative {neg_mean:.3e}");

    // PRIMARY: systematic-error gate.
    assert!(
        pos_mean < 3e-3,
        "prompt_embeds real-token mean_rel {pos_mean:.3e} too high — systematic encoder delta"
    );
    assert!(
        neg_mean < 3e-3,
        "negative_prompt_embeds real-token mean_rel {neg_mean:.3e} too high — systematic encoder delta"
    );
    // COARSE guard: gross single-element blowup.
    assert!(
        pos_rel < 8e-2,
        "prompt_embeds real-token peak_rel {pos_rel:.3e} too high"
    );
    assert!(
        neg_rel < 8e-2,
        "negative_prompt_embeds real-token peak_rel {neg_rel:.3e} too high"
    );
}
