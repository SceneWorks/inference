//! SC-15615 — DiT-level bounded attention: the MLX twin of `candle_gen_z_image`'s
//! `attention_query_chunking_matches_the_unbounded_forward` (`packed_dit.rs`).
//!
//! Weight-free: reuses the tiny synthetic fixtures the parity tests already ship
//! (`z_transformer.safetensors`, `z_control_transformer.safetensors`), so this runs in ordinary CI
//! rather than only on a Mac with a 35 GB snapshot.
//!
//! ## Why a tripped cancel flag is the assertion
//!
//! An equivalence test alone — "chunked output == unbounded output" — is a **false green**: it passes
//! trivially if the plan never reaches the attention, or if chunking never engages. Deleting the whole
//! rung would leave it green.
//!
//! So each equivalence check is paired with a plan whose cancel flag is **already tripped**. Between
//! chunks, `mlx_gen::attention::sdpa_budgeted_bhsd` checks that flag and returns [`Error::Canceled`];
//! the unbounded fast path has no chunk boundary and never consults it. A forward that returns `Ok`
//! under a tripped-cancel bounded plan therefore proves one of: the plan was dropped somewhere in the
//! call chain, or the budget did not chunk. Either is the regression this file exists to catch.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error};
use mlx_gen_z_image::{ZImageControlTransformer, ZImageTransformer, ZImageTransformerConfig};
use mlx_rs::{Array, Dtype};

const BASE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_transformer.safetensors"
);
const CONTROL_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_control_transformer.safetensors"
);

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

/// A budget that chunks these fixtures' unified stacks into several blocks. The fixtures are tiny
/// (dim 64-96, 4 heads, a few dozen tokens), so the production 64 Mi budget would never engage —
/// this picks a budget in the same *shape* (a score-element cap) sized for the fixture.
fn chunking_budget() -> AttentionBudget {
    // 4 heads x ~64 keys x 8 query rows = ~2048 scores per chunk.
    AttentionBudget::from_score_elements(2048, true)
}

fn peak_relative_delta(a: &Array, b: &Array) -> f32 {
    let n: i32 = b.shape().iter().product();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    xs.iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

/// MLX's Metal matmul is reduced-precision (the sc-2338 class this crate already tolerates at 1e-2
/// across a whole forward), and chunking changes which kernel specialization runs per block, so this
/// is the numerical-equivalence bound over a whole DiT forward.
const TOLERANCE: f32 = 1e-3;

#[test]
fn base_dit_chunked_forward_matches_the_unbounded_forward() {
    let w = Weights::from_file(BASE_FIXTURE).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", base_cfg()).unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();

    let unbounded = model.forward(x, 0.7, cap).unwrap();
    let chunked = model
        .forward_budgeted(x, 0.7, cap, AttentionPlan::budgeted(chunking_budget()))
        .unwrap();
    assert_eq!(chunked.shape(), unbounded.shape(), "output shape");
    let delta = peak_relative_delta(&chunked, &unbounded);
    assert!(
        delta < TOLERANCE,
        "bounded attention changed the DiT velocity: peak-relative delta {delta:e}"
    );

    // The anti-false-green half: the plan must actually reach an attention that actually chunks.
    let cancel = CancelFlag::default();
    cancel.cancel();
    let plan = AttentionPlan::budgeted(chunking_budget()).with_cancel(&cancel);
    match model.forward_budgeted(x, 0.7, cap, plan) {
        Err(Error::Canceled) => {}
        other => panic!(
            "a tripped cancel under a CHUNKING plan must abort the forward between chunks — got \
             {:?}. Either the plan is not threaded to every attention, or the budget did not chunk, \
             which would make the equivalence assertion above vacuous.",
            other.map(|v| v.shape().to_vec())
        ),
    }

    // ...and the unbounded fast path has no chunk boundary, so a tripped flag cannot affect it.
    let plan = AttentionPlan::UNBOUNDED.with_cancel(&cancel);
    assert!(
        model.forward_budgeted(x, 0.7, cap, plan).is_ok(),
        "a tripped cancel must not affect the unbounded (default) forward"
    );
}

#[test]
fn control_dit_chunked_forward_matches_the_unbounded_forward() {
    let w = Weights::from_file(CONTROL_FIXTURE).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg()).unwrap();
    let model = ZImageControlTransformer::from_weights(base, &w, "w").unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let cc = w.require("in.control_context").unwrap();
    let scale = 1.0;

    let unbounded = model.forward(x, 0.7, cap, Some(cc), scale).unwrap();
    let chunked = model
        .forward_budgeted(
            x,
            0.7,
            cap,
            Some(cc),
            scale,
            AttentionPlan::budgeted(chunking_budget()),
        )
        .unwrap();
    assert_eq!(chunked.shape(), unbounded.shape(), "output shape");
    let delta = peak_relative_delta(&chunked, &unbounded);
    assert!(
        delta < TOLERANCE,
        "bounded attention changed the control DiT velocity: peak-relative delta {delta:e}"
    );

    // The control route has TWO attention stacks — the base DiT's and the parallel control branch's.
    // A tripped cancel proves the plan reached a chunking attention on this route as well.
    let cancel = CancelFlag::default();
    cancel.cancel();
    let plan = AttentionPlan::budgeted(chunking_budget()).with_cancel(&cancel);
    match model.forward_budgeted(x, 0.7, cap, Some(cc), scale, plan) {
        Err(Error::Canceled) => {}
        other => panic!(
            "a tripped cancel under a CHUNKING plan must abort the control forward — got {:?}",
            other.map(|v| v.shape().to_vec())
        ),
    }

    // The no-control short-circuit delegates to the base DiT, and must carry the plan with it.
    let plan = AttentionPlan::budgeted(chunking_budget()).with_cancel(&cancel);
    match model.forward_budgeted(x, 0.7, cap, None, scale, plan) {
        Err(Error::Canceled) => {}
        other => panic!(
            "the control transformer's no-control short-circuit dropped the attention plan — got \
             {:?}",
            other.map(|v| v.shape().to_vec())
        ),
    }
}

/// The default forward is byte-identical to the pre-SC-15615 one: `forward` and an explicitly
/// UNBOUNDED `forward_budgeted` must agree **exactly**, not within a tolerance.
#[test]
fn the_default_forward_is_exactly_the_unbounded_forward() {
    let w = Weights::from_file(BASE_FIXTURE).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", base_cfg()).unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();

    let implicit = model.forward(x, 0.7, cap).unwrap();
    let explicit = model
        .forward_budgeted(x, 0.7, cap, AttentionPlan::UNBOUNDED)
        .unwrap();
    assert_eq!(peak_relative_delta(&implicit, &explicit), 0.0);
}
