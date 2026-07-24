//! Shared helpers for the sc-14040 NR-MMDiT parity tests: golden-bundle discovery, the checkpoint
//! snapshot override, and the error metric the gates are stated in.
//!
//! Included per test binary via `mod common;`. `#![allow(dead_code)]` because no single test
//! binary uses every helper.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;

/// `crates/media/mlx-gen/tools/golden` — the directory the real-weights goldens live in. Its
/// contents are **gitignored** (see `crates/media/mlx-gen/.gitignore`), regenerable with
/// `tools/dump_mage_flow_golden.py --stage all` under `MAGE_DEVICE=cpu` (mandatory — sc-14250:
/// MPS dumps are silently corrupt), and validated by `tools/verify_mage_flow_golden.py`.
pub const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden");

/// The one-block golden: `block_in.*` (the real step-0 inputs of `transformer_blocks[0]`, msrope
/// table included) and `block_out.{0,1}` = `(encoder_hidden_states, hidden_states)`.
pub const BLOCK_GOLDEN: &str = "mage_flow_dit_block_golden.safetensors";

/// The whole-stack golden: `dit_in.*` (the real step-0 inputs of `MageFlow.forward`) plus
/// `img_shapes` and `dit_out`, the 12-block velocity.
pub const STACK_GOLDEN: &str = "mage_flow_dit_golden.safetensors";

/// Load a golden bundle, or `None` when the (gitignored) directory has not been populated.
pub fn golden(name: &str) -> Option<Weights> {
    let path = Path::new(GOLDEN_DIR).join(name);
    if !path.exists() {
        return None;
    }
    Some(Weights::from_file(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display())))
}

/// [`golden`] but panicking with a runnable instruction — for the `#[ignore]`d gates.
pub fn require_golden(name: &str) -> Weights {
    golden(name).unwrap_or_else(|| {
        panic!(
            "missing {name}. Regenerate with:\n  \
             MAGE_DEVICE=cpu <ref-venv>/bin/python crates/media/mlx-gen/tools/\
             dump_mage_flow_golden.py --stage all\n\
             (or copy the bundle into {GOLDEN_DIR}; it is gitignored and must never be committed)"
        )
    })
}

/// The `microsoft/Mage-Flow` snapshot directory, from `MAGE_SNAPSHOT`.
///
/// A **passed-in path**, never derived: this repository resolves no HF cache location of its own
/// (the epic-13657 boundary `scripts/check-workspace.py` enforces). Accepts either the repo root
/// or the `transformer/` subdirectory and returns the `transformer/` directory.
pub fn transformer_dir() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var("MAGE_SNAPSHOT").ok()?);
    let candidate = if root.join("config.json").exists() {
        root
    } else {
        root.join("transformer")
    };
    candidate.join("config.json").exists().then_some(candidate)
}

/// [`transformer_dir`] but panicking with a clear message — for the `#[ignore]`d gates.
pub fn require_transformer_dir() -> PathBuf {
    transformer_dir().expect(
        "set MAGE_SNAPSHOT to a microsoft/Mage-Flow snapshot directory (the repo root, or its \
         `transformer/` subdirectory)",
    )
}

/// `(max_abs, max_rel, mean_rel)` against reference `want`:
/// `max|a−b|`, `max|a−b| / peak|b|` and `mean|a−b| / mean|b|`.
pub fn error(got: &Array, want: &Array) -> (f32, f32, f32) {
    assert_eq!(got.shape(), want.shape(), "shape mismatch");
    let n = want.shape().iter().product::<i32>();
    let got = got.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let want = want
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap();
    let (xs, ys) = (got.as_slice::<f32>(), want.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mean_abs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_abs = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_abs, max_abs / peak, mean_diff / mean_abs)
}

/// `max|x|` over a tensor — the denominator behind the peak-relative metric, and the anchor for
/// the bf16 ULP arithmetic the block-parity gate states its result in.
pub fn peak_abs(x: &Array) -> f32 {
    let n = x.shape().iter().product::<i32>();
    x.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .fold(0f32, |m, &v| m.max(v.abs()))
}

/// The bf16 unit-in-last-place at `value`: bf16 carries **8** significand bits (1 implicit + 7
/// stored), so a value in `[2ᵉ, 2ᵉ⁺¹)` is representable to `2ᵉ⁻⁷`.
///
/// Exists so "the block agrees to within one bf16 rounding step" is an executable assertion rather
/// than exponent arithmetic done by hand in a comment — which is exactly how the first revision of
/// that comment got it wrong by 2×.
pub fn bf16_ulp_at(value: f32) -> f32 {
    const SIGNIFICAND_BITS: i32 = 8;
    assert!(value > 0.0, "ULP is only defined for a positive magnitude");
    ((value.log2().floor() as i32 - (SIGNIFICAND_BITS - 1)) as f32).exp2()
}

/// Read an integer golden tensor (`*_cu_seqlens`, `img_shapes`) as `i32`.
pub fn ints(w: &Weights, key: &str) -> Vec<i32> {
    let t = w.require(key).unwrap_or_else(|e| panic!("{key}: {e}"));
    t.as_dtype(Dtype::Int32)
        .unwrap()
        .reshape(&[t.shape().iter().product::<i32>()])
        .unwrap()
        .as_slice::<i32>()
        .to_vec()
}
