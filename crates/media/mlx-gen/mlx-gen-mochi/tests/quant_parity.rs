//! A6 Mochi 1 **Q4/Q8 packed-load parity** (sc-11990).
//!
//! Two layers of gate:
//!
//! 1. **Bit-exact vs the reference `mx.quantize` golden** (non-ignored, CI-green) — a single
//!    `to_q`-shaped Linear slice. The committed fixture `tests/fixtures/mochi_quant_slice.safetensors`
//!    (from `tools/dump_mochi_quant_fixtures.py`) carries the bf16 weight `w`, the f32 activations `x`,
//!    the packed `q{4,8}.{wq,scales,biases}`, and the reference forward `q{4,8}.y`. Both packed-load
//!    routes reproduce `y` **bit-for-bit** — (a) *convert-then-load*: [`quantize_transformer_map`]
//!    packs the same bf16 `w` (the packing `convert.rs` applies), then [`MochiLinear::load`] consumes
//!    it; (b) *consume-prequantized*: [`MochiLinear::load`] reads the dumped packs off a `Weights`.
//!    Same MLX `quantize` on the same bf16 weight ⇒ byte-identical scales ⇒ deterministic
//!    `quantized_matmul` ⇒ bit-exact. Needs only MLX + the ~0.1 MB fixture (no real weights).
//!
//! 2. **Real-tier transformer-forward residual** (`#[ignore]`d) — loads a built `q4`/`q8` tier dir's
//!    transformer (the consume-prequantized path on the whole 10B AsymmDiT) and reports the residual of
//!    the predicted velocity vs the **bf16** `mochi_dit_golden.safetensors`. Q4 on a 10B DiT is lossy —
//!    this **records** the actual residual (it does not pretend Q4 is bit-exact to bf16). Run after
//!    building the tiers: `MOCHI_Q4_DIR=~/mochi-tiers/q4 MOCHI_Q8_DIR=~/mochi-tiers/q8 cargo test -p
//!    mlx-gen-mochi --test quant_parity -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::ops::{abs, max, mean, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mochi::{
    load_transformer_weights, quantize_transformer_map, MochiDitConfig, MochiLinear, MochiQuant,
    MochiSplitModel, MochiTransformer3DModel,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mochi_quant_slice.safetensors"
);
const DIT_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_dit_golden.safetensors"
);
const GROUP: i32 = 64;
/// A real quant-target key so [`quantize_transformer_map`]'s predicate fires on the slice.
const SLICE_KEY: &str = "transformer_blocks.0.attn1.to_q";

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// `max|got − want|` over f32-cast arrays — `0.0` means bit-exact.
fn max_abs_diff(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    max_abs(&subtract(&got, &want).unwrap())
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    max_abs_diff(got, want) / max_abs(&want.as_dtype(Dtype::Float32).unwrap()).max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let num = mean(abs(subtract(&got, &want).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let den = mean(abs(&want).unwrap(), None).unwrap().item::<f32>();
    num / den.max(1e-12)
}

/// Build a `Weights` holding a single Linear's packed parts under `prefix`.
fn packed_weights(prefix: &str, wq: Array, scales: Array, biases: Array) -> Weights {
    let mut w = Weights::empty();
    w.insert(format!("{prefix}.weight"), wq);
    w.insert(format!("{prefix}.scales"), scales);
    w.insert(format!("{prefix}.biases"), biases);
    w
}

/// The bit-exact fixture gate for one bit-width, covering both packed-load routes.
fn slice_matches_reference(bits: i32) {
    let f = Weights::from_file(FIXTURE).expect("quant slice fixture");
    let x = f.require("x").expect("x").clone(); // [B, in] f32
    let w = f.require("w").expect("w").clone(); // [out, in] bf16
    let y_ref = f.require(&format!("q{bits}.y")).expect("y").clone();
    let quant = Some(MochiQuant { bits, group: GROUP });

    // Route A — convert-then-load: pack the bf16 weight via convert's map quantizer, then consume it.
    let mut dense: HashMap<String, Array> = HashMap::new();
    dense.insert(format!("{SLICE_KEY}.weight"), w.clone());
    let packed = quantize_transformer_map(dense, bits, GROUP).expect("quantize_transformer_map");
    let mut wa = Weights::empty();
    for (k, v) in &packed {
        wa.insert(k.clone(), v.clone());
    }
    let lin_a =
        MochiLinear::load(&wa, SLICE_KEY, false, quant, Dtype::Float32).expect("load route-A");
    let y_a = lin_a.forward(&x).expect("forward route-A");
    let d_a = max_abs_diff(&y_a, &y_ref);

    // Route B — consume-prequantized: read the dumped packs directly.
    let wb = packed_weights(
        SLICE_KEY,
        f.require(&format!("q{bits}.wq")).unwrap().clone(),
        f.require(&format!("q{bits}.scales")).unwrap().clone(),
        f.require(&format!("q{bits}.biases")).unwrap().clone(),
    );
    let lin_b =
        MochiLinear::load(&wb, SLICE_KEY, false, quant, Dtype::Float32).expect("load route-B");
    let y_b = lin_b.forward(&x).expect("forward route-B");
    let d_b = max_abs_diff(&y_b, &y_ref);

    eprintln!(
        "[Q{bits}] slice packed-forward max|Δ| vs reference: route-A(convert)={d_a:.3e} \
         route-B(consume)={d_b:.3e}; route-A vs route-B={:.3e}",
        max_abs_diff(&y_a, &y_b)
    );

    // Same MLX quantize on the same bf16 weight + deterministic quantized_matmul ⇒ bit-exact. A
    // predicate/scale/transpose bug is O(1e-1+); flag any real divergence loudly.
    assert_eq!(d_a, 0.0, "Q{bits} convert-then-load route not bit-exact");
    assert_eq!(d_b, 0.0, "Q{bits} consume-prequantized route not bit-exact");
    assert_eq!(
        max_abs_diff(&y_a, &y_b),
        0.0,
        "Q{bits} the two routes disagree"
    );
}

#[test]
fn q4_slice_forward_matches_reference() {
    slice_matches_reference(4);
}

#[test]
fn q8_slice_forward_matches_reference() {
    slice_matches_reference(8);
}

// --------------------------------------------------------------- real-tier residual (#[ignore]d)

fn tier_dir(bits: i32) -> PathBuf {
    if let Ok(d) = std::env::var(format!("MOCHI_Q{bits}_DIR")) {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").unwrap()).join(format!("mochi-tiers/q{bits}"))
}

/// Load a built tier dir's transformer (consume-prequantized) and report the predicted-velocity
/// residual vs the **bf16** DiT golden. Records the (lossy) Q4/Q8 residual on the real 10B AsymmDiT.
fn tier_transformer_residual(bits: i32) {
    let dir = tier_dir(bits);
    if !dir.join("transformer").is_dir() {
        eprintln!(
            "skip: {} has no transformer/ (build the tiers first — MOCHI_Q{bits}_DIR)",
            dir.display()
        );
        return;
    }
    let split = MochiSplitModel::from_model_dir(&dir).expect("read split_model.json");
    assert!(split.quantized, "tier {} must be quantized", dir.display());
    assert_eq!(split.bits, bits, "tier bits");
    let cfg = MochiDitConfig {
        quantization: Some(MochiQuant {
            bits: split.bits,
            group: split.group,
        }),
        ..Default::default()
    };

    let w = load_transformer_weights(&dir).expect("load tier transformer");
    let model =
        MochiTransformer3DModel::from_weights(&w, &cfg, Dtype::Float32).expect("build tier DiT");

    let g = Weights::from_file(DIT_GOLDEN).expect("dit golden");
    let got = model
        .forward(
            g.require("hidden_states").unwrap(),
            g.require("encoder_hidden_states").unwrap(),
            g.require("timestep").unwrap(),
            g.require("encoder_attention_mask").unwrap(),
        )
        .expect("tier DiT forward");
    let want = g.require("noise_pred").unwrap();
    assert_eq!(got.shape(), want.shape(), "noise_pred shape");
    let pr = peak_rel(&got, want);
    let mr = mean_rel(&got, want);
    eprintln!("[Q{bits} tier] noise_pred vs bf16 golden — peak_rel={pr:.3e} mean_rel={mr:.3e}");
    // Sanity floor only: a quantized 10B DiT stays a bounded perturbation of the bf16 velocity; a
    // structural break (wrong predicate / transpose / NaN) is orders of magnitude larger. The exact
    // residual is RECORDED (Q4 is lossy), never asserted bit-exact against bf16.
    assert!(got.as_slice::<f32>().iter().all(|v| v.is_finite()));
    assert!(
        mr < 0.5,
        "Q{bits} tier mean_rel {mr:.3e} implies a structural break, not just quant loss"
    );
}

#[test]
#[ignore = "needs a built q4 tier dir (MOCHI_Q4_DIR) + tools/golden/mochi_dit_golden.safetensors"]
fn q4_tier_transformer_residual() {
    tier_transformer_residual(4);
}

#[test]
#[ignore = "needs a built q8 tier dir (MOCHI_Q8_DIR) + tools/golden/mochi_dit_golden.safetensors"]
fn q8_tier_transformer_residual() {
    tier_transformer_residual(8);
}
