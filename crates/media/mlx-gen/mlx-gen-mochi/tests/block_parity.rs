//! A3 MochiTransformerBlock parity for Mochi 1 (sc-11987) vs the A1 real-weight golden.
//!
//! Two tiers:
//!  - a **committed, non-ignored** CI-green test that exercises the block's forward on synthetic
//!    random weights (shape + determinism) — covered by the crate's `transformer::tests` unit tests,
//!    so this file adds only the real-weight gate below.
//!  - an **`#[ignore]`d** real-weight test that loads block 0's weights from `$MOCHI_SNAPSHOT`, feeds
//!    the 6 `block_in.*` tensors from `mochi_dit_block_golden.safetensors` directly (using the
//!    golden's captured `image_rotary_emb` so the block is isolated from RoPE construction), and
//!    checks `block_out.0` (visual) and `block_out.1` (text, **valid rows only**) reproduce the golden.
//!
//! The text output's padded rows differ by construction — the reference gathers only valid text
//! tokens and zero-pads the rest, while the joint-mask forward attends every text-query row — so the
//! `block_out.1` gate masks padded positions via `encoder_attention_mask` (the `te_parity` precedent).
//!
//! Run the real-weight gate:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p mlx-gen-mochi --test block_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mochi::{load_transformer_weights, MochiDitConfig, MochiRope, MochiTransformerBlock};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_dit_block_golden.safetensors"
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

/// Peak relative error over the **valid (non-padded) text rows only** — zero out padded positions via
/// `mask [B, St]` before comparing. Padded text-query rows are zeroed by the reference (gather-valid)
/// but attended by the joint-mask forward, so a full-tensor compare is dominated by those by-design
/// differences. This is the meaningful gate for `block_out.1`.
fn masked_peak_rel(got: &Array, want: &Array, mask: &Array) -> f32 {
    let sh = mask.shape();
    let (b, l) = (sh[0], sh[1]);
    let m = mask
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[b, l, 1])
        .unwrap();
    let got = multiply(got.as_dtype(Dtype::Float32).unwrap(), &m).unwrap();
    let want = multiply(want.as_dtype(Dtype::Float32).unwrap(), &m).unwrap();
    peak_rel(&got, &want)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (bf16 DiT shards) + tools/golden/mochi_dit_block_golden.safetensors"]
fn block0_forward_matches_golden() {
    let root = snapshot_dir();
    let g = Weights::from_file(GOLDEN).expect("dit_block golden");
    let cfg = MochiDitConfig::default();

    // Block 0 is a normal (non-final) block. Load its weights as f32.
    let w = load_transformer_weights(&root).expect("load DiT transformer weights");
    let block = MochiTransformerBlock::from_weights(
        &w,
        "transformer_blocks.0",
        &cfg,
        false,
        Dtype::Float32,
    )
    .expect("build block 0");

    // Golden inputs.
    let hidden = g.require("block_in.hidden_states").unwrap(); // [2, 32, 3072]
    let enc = g.require("block_in.encoder_hidden_states").unwrap(); // [2, 256, 1536]
    let temb = g.require("block_in.temb").unwrap(); // [2, 3072]
    let enc_mask = g.require("block_in.encoder_attention_mask").unwrap(); // [2, 256]
    let rope = MochiRope::from_parts(
        g.require("block_in.image_rotary_emb.0").unwrap().clone(),
        g.require("block_in.image_rotary_emb.1").unwrap().clone(),
    );

    let (hidden_out, enc_out) = block
        .forward(hidden, enc, temb, &rope, enc_mask)
        .expect("block forward");

    // Visual stream (block_out.0): all rows are meaningful.
    let vis_rel = peak_rel(&hidden_out, g.require("block_out.0").unwrap());
    // Text stream (block_out.1): valid rows only (padded rows differ by construction).
    let txt_rel = masked_peak_rel(&enc_out, g.require("block_out.1").unwrap(), enc_mask);
    // Full-tensor text rel, reported for transparency (inflated by the padded rows).
    let txt_full = peak_rel(&enc_out, g.require("block_out.1").unwrap());

    eprintln!("BLOCK block_out.0 (visual) peak_rel:        {vis_rel:.3e}");
    eprintln!("BLOCK block_out.1 (text, valid rows):       {txt_rel:.3e}");
    eprintln!("BLOCK block_out.1 (text, full incl padded): {txt_full:.3e}");

    // Cross-impl f32-vs-bf16 over one MMDiT block (joint attention + two gated residuals + SwiGLU).
    // The residual reflects the reference's bf16 rounding, not a structural delta — a real bug (wrong
    // mask, wrong norm/gate, wrong RoPE wiring) diverges orders of magnitude. 5e-2 is the T5-family
    // cross-bf16 precedent (te_parity 6e-2).
    assert!(
        vis_rel < 5e-2,
        "block_out.0 visual peak_rel {vis_rel:.3e} too high"
    );
    assert!(
        txt_rel < 5e-2,
        "block_out.1 text valid-row peak_rel {txt_rel:.3e} too high"
    );
}
