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
use candle_gen_mochi::{encode_prompt, load_tokenizer, MochiT5, DIT_DTYPE};

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
    peak_rel(&got, &want)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (T5-XXL shards) + tools/golden/mochi_te_golden.safetensors (CUDA)"]
fn t5_encode_matches_golden() {
    let device = candle_gen::default_device().unwrap();
    let root = snapshot_dir();
    let tok = load_tokenizer().unwrap();
    // Load the T5 at the production dtype (bf16); the 6e-2 bar absorbs the reference's bf16 rounding.
    let t5 = MochiT5::load(&root.join("text_encoder"), DIT_DTYPE, &device).expect("load T5-XXL");
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

    // Embeds: cross-impl (candle bf16 vs the reference's bf16, over 24 T5 blocks). Real-token peak_rel
    // (padded rows zeroed via the attention mask) is the meaningful gate; 6e-2 is the LTX/Chroma-family
    // cross-bf16 T5 precedent (the MLX te_parity bar).
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
    eprintln!("TE embeds real-token peak_rel: prompt {pos_rel:.3e}  negative {neg_rel:.3e}");
    assert!(
        pos_rel < 6e-2,
        "prompt_embeds real-token peak_rel {pos_rel:.3e} too high"
    );
    assert!(
        neg_rel < 6e-2,
        "negative_prompt_embeds real-token peak_rel {neg_rel:.3e} too high"
    );
}
