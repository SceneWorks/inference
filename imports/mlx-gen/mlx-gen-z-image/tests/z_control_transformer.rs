//! sc-2349 / sc-2257: full Z-Image **ControlNet** DiT forward parity vs the fork.
//! Fixture `tests/fixtures/z_control_transformer.safetensors` ← `tools/dump_z_control_transformer.py`
//! (tiny synthetic model: dim=64, 4 heads, 2 refiner + 4 main layers, in_ch=4, patch=2, with the
//! control projections perturbed off zero-init so the control branch is actually active).
//! Tol 1e-2 — Metal fp32 across a 40+-matmul control forward.

use mlx_gen::weights::Weights;
use mlx_gen_z_image::{ZImageControlTransformer, ZImageTransformer, ZImageTransformerConfig};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_control_transformer.safetensors"
);

fn small_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 64,
        n_layers: 4,
        n_refiner_layers: 2,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 4, 4],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

#[test]
fn control_forward_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", small_cfg()).unwrap();
    let model = ZImageControlTransformer::from_weights(base, &w, "w").unwrap();

    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let cc = w.require("in.control_context").unwrap();

    // 1) Control ON (scale 1.0) reproduces the fork's control output.
    let y_ctrl = model.forward(x, 0.7, cap, Some(cc), 1.0).unwrap();
    let want_ctrl = w.require("out.y_ctrl").unwrap();
    assert_eq!(y_ctrl.shape(), want_ctrl.shape(), "control output shape");
    assert!(
        all_close(&y_ctrl, want_ctrl, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "control-on forward diverged from the fork (max|Δ| {:.3e})",
        max_abs_diff(&y_ctrl, want_ctrl)
    );

    // 2) control_context = None delegates to the base path and matches the fork's base output.
    let y_none = model.forward(x, 0.7, cap, None, 1.0).unwrap();
    let want_none = w.require("out.y_none").unwrap();
    assert!(
        all_close(&y_none, want_none, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "control-off (None) forward diverged from the fork's base output (max|Δ| {:.3e})",
        max_abs_diff(&y_none, want_none)
    );

    // 3) Self-consistency: control_context_scale = 0 contributes zero, so it equals the None path
    //    (same Rust forward, so near-exact). This is the fork's own parity gate.
    let y_scale0 = model.forward(x, 0.7, cap, Some(cc), 0.0).unwrap();
    assert!(
        all_close(&y_scale0, &y_none, 1e-4, 1e-4, false)
            .unwrap()
            .item::<bool>(),
        "control scale=0 is not inert: it differs from the base path (max|Δ| {:.3e})",
        max_abs_diff(&y_scale0, &y_none)
    );

    // 4) Guard against a no-op control bug: control-on must meaningfully differ from control-off.
    let delta = max_abs_diff(&y_ctrl, &y_none);
    assert!(
        delta > 1e-2,
        "control branch appears inert (max|Δ| {delta:.3e}); a no-op would pass gates 1-3 vacuously"
    );
    println!(
        "✓ control transformer matches fork: ctrl shape {:?}; on-vs-off max|Δ| {delta:.4}",
        y_ctrl.shape()
    );
}
