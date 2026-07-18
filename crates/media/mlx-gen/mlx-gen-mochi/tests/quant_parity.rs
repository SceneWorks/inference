//! A6 Mochi 1 **Q4/Q8 packed-load parity** (sc-11990).
//!
//! Two layers of gate:
//!
//! 1. **Packed-load parity vs the reference `mx.quantize` golden** (non-ignored, CI-green) — a
//!    single `to_q`-shaped Linear slice. The committed fixture
//!    `tests/fixtures/mochi_quant_slice.safetensors` (from `tools/dump_mochi_quant_fixtures.py`)
//!    carries the bf16 weight `w`, the f32 activations `x`, the packed `q{4,8}.{wq,scales,biases}`,
//!    and the reference forward `q{4,8}.y`. Two packed-load routes exercise the loader — (a)
//!    *convert-then-load*: [`quantize_transformer_map`] packs the same bf16 `w` (the packing
//!    `convert.rs` applies), then [`MochiLinear::load`] consumes it; (b) *consume-prequantized*:
//!    [`MochiLinear::load`] reads the dumped packs off a `Weights`. Needs only MLX + the ~0.1 MB
//!    fixture (no real weights). This gate has **two comparisons** with deliberately different
//!    strictness:
//!
//!    * **routes-agree — BIT-EXACT** (`max|y_a − y_b| == 0.0`): both routes run the *same*
//!      `quantized_matmul` on the *same* platform off byte-identical packs, so they must be
//!      bit-for-bit identical on every platform. This is the real packing/scale/transpose
//!      tripwire and stays asserted exact.
//!    * **vs-golden — TIGHT ULP TOLERANCE** ([`MOCHI_QUANT_GOLDEN_ULP_TOL`], relative): under
//!      the MLX 0.32.0 pin (epic 12742) `quantized_matmul`'s f32 output is **NAX-path-dependent**
//!      — the self-hosted Apple-matrix-unit "NAX" path (macOS 26.2, deployment-target 26.2) and
//!      the hosted non-NAX path (macOS 15, deployment-target 15.0 — what PR CI runs) differ by
//!      ~1–2 ULP-f32 (Q4 abs 1.31e-6, Q8 9.54e-7 on this fixture; ≤2 ULP relative). The packs
//!      (`wq`/`scales`/`biases`) are byte-identical across 0.31.2→0.32.0 and across NAX/non-NAX —
//!      only the accumulation drifted, and only on the NAX path (MLX #3631/#3632/#3810). On
//!      0.31.2 both Metal paths were bit-identical so one golden served both; 0.32.0's NAX quant
//!      fixes broke that tie. The committed golden is the **non-NAX (CI-matching) reference**,
//!      which is unchanged from the 0.31.2 dump — so CI reproduces it ~exactly while the
//!      self-hosted NAX runner sits ~1–2 ULP away. A predicate/scale/transpose bug is O(1e-1),
//!      orders of magnitude above this ULP floor, so it is still caught loudly.
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

/// Relative tolerance for the packed-forward **vs the committed golden** comparison (see the
/// module doc, gate 1). Under the MLX 0.32.0 pin `quantized_matmul`'s f32 accumulation is
/// NAX-path-dependent: the self-hosted Apple-matrix-unit ("NAX", dt26.2) path and the hosted
/// non-NAX (dt15.0, PR-CI) path differ by ~1–2 ULP-f32. On this fixture the measured gap is
/// **≤2 ULP relative** — Q4 `1.311e-6 / max|y|≈5.503 = 2.383e-7` (= 2.00 ULP), Q8
/// `9.537e-7 / 5.662 = 1.684e-7` (= 1.41 ULP). The committed golden is the non-NAX (CI) value,
/// so CI reproduces it ~exactly and the self-hosted NAX runner lands one drift away; a single
/// bound has to cover the larger of the two. `16 ULP` gives ~8× headroom over the worst (Q4,
/// 2 ULP) gap while staying ~5e4× below a real predicate/scale/transpose bug (O(1e-1) relative),
/// so that class of regression is still caught loudly. Applied **relative** via [`peak_rel`],
/// mirroring [`mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL`] (the 0.32.0 compiled-glue f32 gate). The
/// two packed-load routes still agree **bit-exact** with each other — only the vs-golden compare
/// uses this tolerance.
const MOCHI_QUANT_GOLDEN_ULP_TOL: f32 = 16.0 * f32::EPSILON;

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

/// The fixture gate for one bit-width, covering both packed-load routes: the two routes must agree
/// **bit-exact** with each other, and each must match the committed golden within a tight ULP
/// tolerance (the golden is the non-NAX/CI value; the self-hosted NAX path drifts ~1–2 ULP under
/// MLX 0.32.0). See the module doc and [`MOCHI_QUANT_GOLDEN_ULP_TOL`].
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
    let rel_a = peak_rel(&y_a, &y_ref);

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
    let rel_b = peak_rel(&y_b, &y_ref);

    eprintln!(
        "[Q{bits}] slice packed-forward vs reference: route-A(convert) abs={d_a:.3e} rel={rel_a:.3e} \
         route-B(consume) abs={d_b:.3e} rel={rel_b:.3e}; route-A vs route-B={:.3e} \
         (vs-golden tol={MOCHI_QUANT_GOLDEN_ULP_TOL:.3e})",
        max_abs_diff(&y_a, &y_b)
    );

    // vs-golden: TIGHT ULP tolerance. Under MLX 0.32.0 `quantized_matmul`'s f32 output is
    // NAX-path-dependent — the self-hosted NAX (dt26.2) path and the hosted non-NAX (dt15.0, CI)
    // path differ ~1–2 ULP. The committed golden is the non-NAX (CI) value, so CI reproduces it
    // ~exactly while this NAX runner sits one drift (≤2 ULP rel) away; both stay under the bound.
    // A predicate/scale/transpose bug is O(1e-1) — orders above this floor — so it is still caught.
    assert!(
        rel_a <= MOCHI_QUANT_GOLDEN_ULP_TOL,
        "Q{bits} convert-then-load route rel {rel_a:.3e} exceeds vs-golden tol \
         {MOCHI_QUANT_GOLDEN_ULP_TOL:.3e} (NAX/non-NAX drift is ≤2 ULP; this is larger)"
    );
    assert!(
        rel_b <= MOCHI_QUANT_GOLDEN_ULP_TOL,
        "Q{bits} consume-prequantized route rel {rel_b:.3e} exceeds vs-golden tol \
         {MOCHI_QUANT_GOLDEN_ULP_TOL:.3e} (NAX/non-NAX drift is ≤2 ULP; this is larger)"
    );
    // routes-agree: BIT-EXACT. Both routes run the same `quantized_matmul` on the same platform
    // off byte-identical packs — they must be bit-for-bit identical everywhere. The real packing/
    // scale/transpose tripwire; stays asserted exact on NAX and non-NAX alike.
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

// -------------------------------------------------------------- full tier load (#[ignore]d, heavy)

/// End-to-end load of a built `q4` tier through the public [`mlx_gen_mochi::load`] seam: exercises the
/// `split_model.json` manifest read, the `spec.quantize` assert-against-manifest, the packed-DiT
/// consume path, AND the shared T5/VAE resolution from the tier dir's parent. Heavy (materializes the
/// shared fp32 T5-XXL); `#[ignore]`d.
#[test]
#[ignore = "loads the whole q4 tier (packed DiT + shared T5/VAE) — needs a built tier tree"]
fn q4_tier_loads_end_to_end() {
    let dir = tier_dir(4);
    if !dir.join("transformer").is_dir() {
        eprintln!("skip: no built q4 tier at {} (MOCHI_Q4_DIR)", dir.display());
        return;
    }
    // `.with_quant(Q4)` also validates the assert-against-manifest matches the tier's bits.
    let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(dir.clone()))
        .with_quant(mlx_gen::Quant::Q4);
    let _model = mlx_gen_mochi::load(&spec)
        .unwrap_or_else(|e| panic!("load q4 tier {} failed: {e}", dir.display()));
    eprintln!(
        "OK: q4 tier loaded end-to-end (packed DiT + shared T5/VAE) from {}",
        dir.display()
    );
}

/// The `spec.quantize` assertion is checked before any heavy load, so a bits mismatch errors fast:
/// asking for Q8 against the `q4` tier must be a hard error (never a silent wrong-tier run).
#[test]
#[ignore = "needs a built q4 tier dir (MOCHI_Q4_DIR)"]
fn tier_quant_bits_mismatch_errors() {
    let dir = tier_dir(4);
    if !dir.join("split_model.json").is_file() {
        eprintln!("skip: no built q4 tier at {} (MOCHI_Q4_DIR)", dir.display());
        return;
    }
    let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(dir.clone()))
        .with_quant(mlx_gen::Quant::Q8);
    assert!(
        mlx_gen_mochi::load(&spec).is_err(),
        "Q8 against the q4 tier must error (assert-against-manifest)"
    );
}
