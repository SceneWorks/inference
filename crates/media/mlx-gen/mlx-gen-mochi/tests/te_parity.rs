//! A2 text-encoder parity for Mochi 1 (sc-11985) vs the A1 real-weight golden.
//!
//! Two tiers:
//!  - a **committed, non-ignored** CI-green test that exercises the vendored tokenizer (shape,
//!    determinism, EOS+pad structure) with no model weights — the parts this crate owns. The reused
//!    [`mlx_gen_flux::T5TextEncoder`] is fixed to the t5-v1.1-xxl geometry (24 blocks / 4096-dim), so a
//!    "tiny random-weight" forward isn't instantiable; the forward is gated by the ignored test below.
//!  - an **`#[ignore]`d** real-weight test that runs the full masked encode from `$MOCHI_SNAPSHOT` and
//!    checks `prompt_embeds` + both attention masks reproduce `mochi_te_golden.safetensors`.
//!
//! Run the real-weight gate:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p mlx-gen-mochi --test te_parity -- --ignored --nocapture`

use mlx_gen::tokenizer::to_arrays;
use mlx_gen::weights::Weights;
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen_mochi::{encode_prompt, load_t5_encoder, load_tokenizer};

/// The exact prompt the A1 dump harness (`tools/dump_mochi_golden.py`) blessed the golden with.
const PROMPT: &str = "A calico kitten batting a ball of red yarn across a sunlit wooden floor.";
/// The A1 harness used an empty negative prompt.
const NEGATIVE: &str = "";

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_te_golden.safetensors"
);

fn snapshot_dir() -> std::path::PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

/// `max|got − want| / max|want|` — the repo's peak relative error (see LTX `te_parity`).
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let diff = abs(subtract(&got, &want).unwrap()).unwrap();
    let denom = max(abs(&want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// Peak relative error over the **real (unmasked) token rows only** — zero out padded positions in
/// both tensors via `mask [1, L]` before comparing. Padded rows carry near-zero signal and are masked
/// out in the DiT downstream, so a full-tensor `peak_rel` is dominated by padding-noise / tiny-denom
/// artifacts (worst on the empty negative prompt). This is the parity metric that reflects the tokens
/// that actually condition generation.
fn masked_peak_rel(got: &Array, want: &Array, mask: &Array) -> f32 {
    use mlx_rs::ops::multiply;
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    // mask [1, L] -> [1, L, 1] to broadcast over the 4096 feature dim.
    let l = *mask.shape().last().unwrap();
    let m = mask
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[1, l, 1])
        .unwrap();
    let got = multiply(&got, &m).unwrap();
    let want = multiply(&want, &m).unwrap();
    peak_rel(&got, &want)
}

// ---------------------------------------------------------------- CI-green (no weights)

#[test]
fn tokenizer_pads_to_max_len_and_is_deterministic() {
    let tok = load_tokenizer().unwrap();

    let a = tok.tokenize(PROMPT).unwrap();
    let b = tok.tokenize(PROMPT).unwrap();
    // Determinism: identical ids + mask across calls.
    assert_eq!(a.ids, b.ids, "tokenization must be deterministic");
    assert_eq!(a.mask, b.mask);

    // Padded to max_length = 256.
    assert_eq!(a.ids.len(), 256, "padded to max_sequence_length");

    let (input_ids, _) = to_arrays(&a);
    assert_eq!(input_ids.shape(), &[1, 256]);

    // Real tokens are a contiguous non-pad prefix ending in the EOS `</s>` (id 1), then pad (id 0).
    let first_pad = a.ids.iter().position(|&id| id == 0).expect("some padding");
    assert!(first_pad >= 2, "content + EOS present");
    assert_eq!(a.ids[first_pad - 1], 1, "content ends with EOS </s>=1");
    assert!(
        a.ids[first_pad..].iter().all(|&id| id == 0),
        "tail is all pad after the first pad"
    );
}

#[test]
fn empty_prompt_is_eos_then_pad() {
    let tok = load_tokenizer().unwrap();
    let out = tok.tokenize(NEGATIVE).unwrap();
    assert_eq!(out.ids.len(), 256);
    // T5 encodes "" (add_special_tokens) to just the EOS, then pads.
    assert_eq!(out.ids[0], 1, "empty prompt starts with EOS </s>=1");
    assert!(out.ids[1..].iter().all(|&id| id == 0), "then all pad");
}

#[test]
fn long_prompt_reserves_the_final_slot_for_eos() {
    let tok = load_tokenizer().unwrap();
    let prompt = "hello ".repeat(512);
    let out = tok.tokenize(&prompt).unwrap();

    assert_eq!(out.ids.len(), 256);
    assert_eq!(out.ids[255], 1, "final token must be EOS");
    assert!(
        out.mask.iter().all(|&m| m == 1),
        "truncated row has no padding"
    );
}

// ------------------------------------------------------------- real-weight golden gate

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (fp32 T5-XXL shards, ~20 GB) + tools/golden/mochi_te_golden.safetensors"]
fn t5_encode_matches_golden() {
    let root = snapshot_dir();
    let tok = load_tokenizer().unwrap();
    let t5 = load_t5_encoder(&root).expect("load T5-XXL encoder");
    let g = Weights::from_file(GOLDEN).expect("te golden");

    let pos = encode_prompt(&tok, &t5, PROMPT).expect("encode prompt");
    let neg = encode_prompt(&tok, &t5, NEGATIVE).expect("encode negative");

    // Masks are exact 0/1 — must match bit-for-bit.
    let pos_mask_rel = peak_rel(
        &pos.prompt_attention_mask,
        g.require("prompt_attention_mask").unwrap(),
    );
    let neg_mask_rel = peak_rel(
        &neg.prompt_attention_mask,
        g.require("negative_prompt_attention_mask").unwrap(),
    );
    eprintln!("TE masks: prompt {pos_mask_rel:.3e}  negative {neg_mask_rel:.3e}");
    assert_eq!(pos_mask_rel, 0.0, "prompt_attention_mask must be exact");
    assert_eq!(
        neg_mask_rel, 0.0,
        "negative_prompt_attention_mask must be exact"
    );

    // Embeds: cross-impl (MLX f32 activations vs the reference's bf16, over 24 T5 blocks). The residual
    // reflects the reference's accumulated bf16 rounding, not a structural delta — a real bug (wrong
    // mask, wrong norm, wrong layer wiring) diverges orders of magnitude. Start tight; the 6e-2 bar is
    // the LTX/Chroma-family cross-bf16 T5 precedent.
    // Full-tensor peak_rel (reported for transparency — inflated on the empty negative by padding).
    let pos_full = peak_rel(&pos.prompt_embeds, g.require("prompt_embeds").unwrap());
    let neg_full = peak_rel(
        &neg.prompt_embeds,
        g.require("negative_prompt_embeds").unwrap(),
    );
    // Real-token peak_rel (the meaningful gate — padded rows zeroed via the attention mask).
    let pos_rel = masked_peak_rel(
        &pos.prompt_embeds,
        g.require("prompt_embeds").unwrap(),
        g.require("prompt_attention_mask").unwrap(),
    );
    let neg_rel = masked_peak_rel(
        &neg.prompt_embeds,
        g.require("negative_prompt_embeds").unwrap(),
        g.require("negative_prompt_attention_mask").unwrap(),
    );
    eprintln!("TE embeds peak_rel (full):        prompt {pos_full:.3e}  negative {neg_full:.3e}");
    eprintln!("TE embeds peak_rel (real tokens): prompt {pos_rel:.3e}  negative {neg_rel:.3e}");
    // Cross-impl (MLX f32 activations vs the reference's bf16, over 24 T5 blocks); the residual reflects
    // the reference's accumulated bf16 rounding, not a structural delta (a real bug — wrong mask, norm,
    // or layer wiring — diverges orders of magnitude, and the masks above are bit-exact). 6e-2 is the
    // LTX/Chroma-family cross-bf16 T5 precedent.
    assert!(
        pos_rel < 6e-2,
        "prompt_embeds real-token peak_rel {pos_rel:.3e} too high"
    );
    assert!(
        neg_rel < 6e-2,
        "negative_prompt_embeds real-token peak_rel {neg_rel:.3e} too high"
    );
}
