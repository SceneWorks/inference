//! sc-2963 invariant (rollout of the Wan sc-2957 template): the **compiled elementwise glue**
//! ([`set_compile_glue(true)`]) produces a forward that matches the eager forward. `mx.compile`
//! fuses the SwiGLU activation, the gated residuals, the complex RoPE rotation, and (control only) the
//! hint injection into single kernels. Gated on the committed tiny synthetic models — the **base**
//! DiT forward AND the **control** DiT forward (control ON, so `add_hint` is exercised) — in CI, no
//! real checkpoint.
//!
//! sc-12747 (epic 12742, MLX 0.31.2 → 0.32.0): the fused forward was **bit-identical** to eager
//! (`max|Δ|=0`) on every Metal path under the 0.31.2 pin. Under 0.32.0, `mx.compile`'s shapeless
//! fusion (MLX #3672) rounds the whole-forward *f32* result ~1–2.5 ULP differently from eager **only
//! on the non-NAX path** — macOS 15 / deployment-target 15.0, which is exactly what hosted PR CI
//! builds. The self-hosted NAX (macOS 26.2 / dt26.2) runner stays bit-identical at `0.0`. So this
//! gate asserts the compiled forward within a tight peak-**relative** ULP tolerance
//! ([`COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL`]), not bit-exact; the output **shape** stays asserted
//! exact. A real fusion/packing regression is O(1e-1), orders of magnitude above the ULP floor, so
//! it is still caught loudly. (flux2/wan whole-forward compile-parity happen to stay bit-identical
//! on non-NAX and keep their `== 0.0` gate; only z-image's deeper stack accumulates past 0.)

use mlx_gen::weights::Weights;
use mlx_gen_z_image::{
    set_compile_glue, ZImageControlTransformer, ZImageTransformer, ZImageTransformerConfig,
};
use mlx_rs::Array;

/// Peak-**relative** tolerance for the whole-forward compiled-vs-eager gates below (see the module
/// doc). Under the MLX 0.32.0 pin `mx.compile`'s shapeless fusion drifts the *f32* result ~1 ULP
/// per fused op **only on the non-NAX / dt15.0 (hosted CI) path**; a whole DiT forward accumulates
/// that over every fused SwiGLU / gated-residual / complex-RoPE / (control) hint-injection kernel
/// across all layers, so the peak-relative divergence lands a few ULP above the op-level
/// [`mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL`] (4 ULP). Measured on the non-NAX metallib: base
/// `rel=2.96e-7` (2.48 ULP; `max|Δ|` 4.77e-7 / peak|eager| 1.61), control `rel=2.36e-7` (1.98 ULP;
/// 3.58e-7 / 1.51). `16 ULP` (= the mochi quant vs-golden bound) gives ~6.5× headroom over the
/// worst (base, 2.48 ULP) while staying ~5e4× below an O(1e-1) real fusion/packing bug, so that
/// class of regression is still caught loudly. Applied **relative** via [`mlx_gen::nn::max_rel_diff`];
/// the NAX / dt26.2 path passes trivially (`0.0 ≤ tol`).
const COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL: f32 = 16.0 * f32::EPSILON;

fn max_abs(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn base_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 96,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

fn control_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        dim: 64,
        n_layers: 4,
        n_refiner_layers: 2,
        axes_dims: vec![8, 4, 4],
        ..base_cfg()
    }
}

#[test]
fn compiled_glue_bit_identical_to_eager_base() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/z_transformer.safetensors"
    );
    let w = Weights::from_file(path).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", base_cfg()).unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();

    set_compile_glue(false);
    let eager = model.forward(x, 0.7, cap).unwrap();
    set_compile_glue(true);
    let compiled = model.forward(x, 0.7, cap).unwrap();
    set_compile_glue(false);

    // Shape stays bit-exact (structural guarantee); only the numeric magnitude gate is loosened.
    assert_eq!(
        compiled.shape(),
        eager.shape(),
        "Z-Image base compiled/eager shape"
    );
    let d = max_abs(&compiled, &eager);
    let rel = mlx_gen::nn::max_rel_diff(&compiled, &eager);
    println!("[z-image base compiled vs eager] max|Δ|={d:.3e} rel={rel:.3e}");
    // 0.32.0 non-NAX (hosted CI, dt15.0) shapeless-compile drift is ~2.5 ULP relative; NAX = 0.0.
    assert!(
        rel <= COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL,
        "Z-Image base compiled glue rel|Δ|={rel:e} exceeds {:e} \
         (0.32.0 non-NAX drift is ≤~2.5 ULP; this is larger)",
        COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL
    );
}

#[test]
fn compiled_glue_bit_identical_to_eager_control() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/z_control_transformer.safetensors"
    );
    let w = Weights::from_file(path).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg()).unwrap();
    let model = ZImageControlTransformer::from_weights(base, &w, "w").unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let cc = w.require("in.control_context").unwrap();

    // Control ON (scale 1.0) so the control branch + `add_hint` are exercised.
    set_compile_glue(false);
    let eager = model.forward(x, 0.7, cap, Some(cc), 1.0).unwrap();
    set_compile_glue(true);
    let compiled = model.forward(x, 0.7, cap, Some(cc), 1.0).unwrap();
    set_compile_glue(false);

    // Shape stays bit-exact (structural guarantee); only the numeric magnitude gate is loosened.
    assert_eq!(
        compiled.shape(),
        eager.shape(),
        "Z-Image control compiled/eager shape"
    );
    let d = max_abs(&compiled, &eager);
    let rel = mlx_gen::nn::max_rel_diff(&compiled, &eager);
    println!("[z-image control compiled vs eager] max|Δ|={d:.3e} rel={rel:.3e}");
    // 0.32.0 non-NAX (hosted CI, dt15.0) shapeless-compile drift is ~2 ULP relative; NAX = 0.0.
    assert!(
        rel <= COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL,
        "Z-Image control compiled glue rel|Δ|={rel:e} exceeds {:e} \
         (0.32.0 non-NAX drift is ≤~2.5 ULP; this is larger)",
        COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL
    );
}
