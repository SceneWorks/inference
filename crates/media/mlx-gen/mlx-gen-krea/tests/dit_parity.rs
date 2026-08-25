//! sc-7568 — committed-fixture parity for the Krea 2 single-stream DiT against the Krea-published
//! reference (`github.com/krea-ai/krea-2` `mmdit.py` `SingleStreamDiT`), at tiny dims.
//!
//! Block-level + full-DiT-forward parity (the story AC). The fixtures are produced by
//! `tools/dump_krea_dit_golden.py` (random seeded weights, remapped to the diffusers checkpoint keys)
//! and committed under `tests/fixtures/` — so these run by default. Tolerance 1e-2 matches the spike +
//! the rest of the repo: MLX runs fp32 matmul in reduced precision on Metal (~3–4 sig figs).

use mlx_gen::weights::Weights;
use mlx_gen_krea::transformer::block::{SingleStreamBlock, TextFusionTransformer};
use mlx_gen_krea::transformer::rope::RopeTables;
use mlx_gen_krea::Krea2Transformer;
use mlx_rs::ops::{all_close, multiply, sqrt, subtract, sum};
use mlx_rs::{Array, Dtype};

use crate::common;

use common::{
    tiny_dit_config, SHARED_FIXTURE_DIT_AXES_DIMS_ROPE, SHARED_FIXTURE_DIT_EPS,
    SHARED_FIXTURE_DIT_HEADS, SHARED_FIXTURE_DIT_HEAD_DIM, SHARED_FIXTURE_DIT_HIDDEN,
    SHARED_FIXTURE_DIT_KV_HEADS, SHARED_FIXTURE_DIT_NUM_LAYERWISE_TEXT_BLOCKS,
    SHARED_FIXTURE_DIT_NUM_REFINER_TEXT_BLOCKS, SHARED_FIXTURE_DIT_ROPE_THETA,
    SHARED_FIXTURE_DIT_TXT_HEADS, SHARED_FIXTURE_ROPE_CAP_LEN, SHARED_FIXTURE_ROPE_GRID_H,
    SHARED_FIXTURE_ROPE_GRID_W,
};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

fn load(name: &str) -> Weights {
    Weights::from_file(format!("{FIX}{name}")).unwrap_or_else(|e| {
        panic!("load fixture {name} (run tools/dump_krea_dit_golden.py): {e}");
    })
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::ops::max(mlx_rs::ops::abs(subtract(&a, &b).unwrap()).unwrap(), false)
        .unwrap()
        .item::<f32>()
}

/// The #1 parity risk localized: the 3-axis interleaved RoPE table for the DiT's joint positions
/// (`cap_len` text `(0,0,0)` + an `ht×wt` grid `(0,row,col)`) must match the reference cos/sin exactly.
#[test]
fn rope_matches_reference() {
    let g = load("rope_golden.safetensors");
    // meta = [n_tok, ht, wt, ax0, ax1, ax2] (see the dump); theta fixed at 1000.
    let (cap, ht, wt) = (
        SHARED_FIXTURE_ROPE_CAP_LEN,
        SHARED_FIXTURE_ROPE_GRID_H,
        SHARED_FIXTURE_ROPE_GRID_W,
    );
    let (cos, sin) = RopeTables::build_t2i(
        cap,
        ht,
        wt,
        SHARED_FIXTURE_DIT_AXES_DIMS_ROPE,
        SHARED_FIXTURE_DIT_ROPE_THETA as f64,
    )
    .joint();

    let want_cos = g.require("cos").unwrap();
    let want_sin = g.require("sin").unwrap();
    assert_eq!(cos.shape(), want_cos.shape(), "cos shape");
    assert!(
        all_close(&cos, want_cos, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "rope cos diverged (max abs {:e})",
        max_abs_diff(&cos, want_cos)
    );
    assert!(
        all_close(&sin, want_sin, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "rope sin diverged (max abs {:e})",
        max_abs_diff(&sin, want_sin)
    );
}

/// One `SingleStreamBlock`: DoubleSharedModulation (6-factor pre/post), the sigmoid-gated GQA
/// attention with interleaved RoPE, and the SwiGLU FFN.
#[test]
fn single_block_matches_reference() {
    let w = load("single_block_golden.safetensors");
    let blk = SingleStreamBlock::from_weights(
        &w,
        "blk",
        SHARED_FIXTURE_DIT_HEADS,
        SHARED_FIXTURE_DIT_KV_HEADS,
        SHARED_FIXTURE_DIT_HEAD_DIM,
        SHARED_FIXTURE_DIT_HIDDEN,
        SHARED_FIXTURE_DIT_EPS,
    )
    .unwrap();
    let y = blk
        .forward(
            w.require("in.x").unwrap(),
            w.require("in.tvec").unwrap(),
            w.require("in.cos").unwrap(),
            w.require("in.sin").unwrap(),
        )
        .unwrap();
    let want = w.require("out.y").unwrap();
    assert_eq!(y.shape(), want.shape());
    let c = cosine(&y, want);
    println!(
        "single_block parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&y, want)
    );
    assert!(
        all_close(&y, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "single block diverged beyond 1e-2 (cosine {c:.7})"
    );
}

/// The `TextFusionTransformer`: layer-axis aggregation (attention across the stacked layers) →
/// `projector` 12→1 collapse → token-axis refiner blocks.
#[test]
fn text_fusion_matches_reference() {
    let w = load("text_fusion_golden.safetensors");
    let tf = TextFusionTransformer::from_weights(
        &w,
        SHARED_FIXTURE_DIT_NUM_LAYERWISE_TEXT_BLOCKS,
        SHARED_FIXTURE_DIT_NUM_REFINER_TEXT_BLOCKS,
        SHARED_FIXTURE_DIT_TXT_HEADS,
        SHARED_FIXTURE_DIT_TXT_HEADS,
        SHARED_FIXTURE_DIT_HEAD_DIM,
        SHARED_FIXTURE_DIT_EPS,
    )
    .unwrap();
    let y = tf.forward(w.require("in.x").unwrap()).unwrap();
    let want = w.require("out.y").unwrap();
    assert_eq!(y.shape(), want.shape());
    let c = cosine(&y, want);
    println!(
        "text_fusion parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&y, want)
    );
    assert!(
        all_close(&y, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "text_fusion diverged beyond 1e-2 (cosine {c:.7})"
    );
}

/// **sc-20644 — the adapter's declared logical shapes are the shapes the real module tree loads.**
///
/// `dit_golden.safetensors` is the committed reference dump: every tensor is at the shape the
/// reference `SingleStreamDiT` actually constructs for `tiny_dit_config()` (the same bytes
/// `dit_matches_reference` loads and runs). Checking `logical_shape` against that header — not
/// against a shape this test invents — is what makes the declaration trustworthy for an MXFP8
/// unpad. `in.*` / `out.*` are the golden's forward inputs, not model weights.
#[test]
fn declared_logical_shapes_match_the_reference_dit_header() {
    use mlx_gen::gen_core::LogicalKeyMapping;

    let w = load("dit_golden.safetensors");
    let cfg = tiny_dit_config();
    let mapping = mlx_gen_krea::KreaNativeToDiffusersMapping::for_config(&cfg);

    let mut checked = 0usize;
    for key in w.keys() {
        if key.starts_with("in.") || key.starts_with("out.") {
            continue;
        }
        let want: Vec<usize> = w
            .require(key)
            .unwrap()
            .shape()
            .iter()
            .map(|d| *d as usize)
            .collect();
        assert_eq!(
            mapping.logical_shape(key),
            Some(want),
            "declared logical shape disagrees with the reference dump for `{key}`"
        );
        checked += 1;
    }
    // The golden holds the whole module tree; a silently shrunken fixture must not pass vacuously.
    assert_eq!(
        checked,
        mlx_gen_krea::convert::expected_transformer_keys(&cfg).len(),
        "the golden must cover every transformer key"
    );
}

/// Full `SingleStreamDiT` forward: img patch-embed, the custom timestep embedding + shared modulation,
/// text fusion + `txt_in`, the joint single-stream stack under 3-axis RoPE, the final layer, and
/// unpatchify — end to end vs the reference velocity.
#[test]
fn dit_matches_reference() {
    let w = load("dit_golden.safetensors");
    let cfg = tiny_dit_config();
    cfg.validate().unwrap();
    let dit = Krea2Transformer::from_weights(&w, &cfg).unwrap();
    let velocity = dit
        .forward(
            w.require("in.latent").unwrap(),
            w.require("in.timestep").unwrap(),
            w.require("in.context").unwrap(),
            None,
        )
        .unwrap();
    let want = w.require("out.velocity").unwrap();
    assert_eq!(velocity.shape(), want.shape(), "velocity shape");
    let c = cosine(&velocity, want);
    println!(
        "full-DiT parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&velocity, want)
    );
    assert!(c > 0.999, "full-DiT cosine {c:.7} <= 0.999");
    assert!(
        all_close(&velocity, want, 2e-2, 2e-2, false)
            .unwrap()
            .item::<bool>(),
        "full-DiT velocity diverged beyond 2e-2 (cosine {c:.7})"
    );
}
