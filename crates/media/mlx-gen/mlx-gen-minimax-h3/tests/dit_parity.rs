//! sc-17144: DiT block + 3D MM-RoPE parity — against the **official diffusers**
//! `MiniMaxH3Transformer3DModel`, i.e. the converted checkpoint production loads.
//!
//! Fixture `tests/fixtures/dit_block.safetensors` ← `tools/dump_minimax_h3_dit.py`.
//!
//! # What this file has to be paranoid about
//!
//! Two merged ports in this crate shipped functionally wrong and passed their parity suites
//! (sc-18740). The DiT carries **both** of the transforms that caused it — the gated-FFN half-swap
//! and a fused-QKV reorder — and adds two more silent-error surfaces of its own: the AdaLN row
//! addressing and the MM-RoPE axis split. Every one of them is shape-identical when wrong. So:
//!
//! * the golden comes from the **converted-checkpoint** class, and
//!   `fixture_provenance_records_the_converted_path` fails a regeneration that reverts to a
//!   reference-module dump;
//! * the fixture carries the pre-conversion `src.` tensors, round-tripped through the **official**
//!   conversion functions in the generator, so `published_qkv_is_contiguous_thirds_…` and
//!   `published_ffn_projection_is_value_then_gate` assert against real evidence;
//! * every anti-swap assertion has a matching mutation test that performs the swap and **measures**
//!   how far it moves the output, printing the margin so it stays auditable;
//! * `rel` (relative max-abs-diff) is the gate, never a norm or a checksum. A half-swap leaves the
//!   output norm essentially unchanged and cosine at 0.7-0.8 — cosine is scale-invariant and blind
//!   to it.
//!
//! # The noise floor bounds what this suite can gate
//!
//! Tolerance 1e-2 peak-relative, the mlx-gen house value. Everything here is f32 and MLX runs f32
//! matmul in reduced precision on Metal. `parity_residuals_bound_the_mutation_floor` measures and
//! prints the residual of every stage on each run; the deepest (a full block) sits around 4e-3, so
//! `MUTATION_FLOOR` at 1e-2 is only ~2.4× above it — and every mutation this file performs lands at
//! 1e-1 or above, i.e. ≥24× the floor. That margin is measured on every run rather than assumed,
//! because a suite whose mutations sat near its own noise floor would be reporting jitter.

mod common;

use common::{
    assert_parity, cosine, dit_fixture_config, l2_norm, rel, std_dev, DIT_FIXTURE, DIT_LAYOUT,
};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::layers::DitAttention;
use mlx_gen_minimax_h3::dit::positions::{self, KeyframeAnchor};
use mlx_gen_minimax_h3::dit::qkv;
use mlx_gen_minimax_h3::{
    split_fused_qkv, DitBlock, GatedFfnLayout, MiniMaxH3DitConfig, MmRope, TokenRefiner,
    TokenRefinerBlock, MODALITY_NUM, PUBLISHED_GATED_FFN_LAYOUT,
};

const TOL: f32 = 1e-2;

/// Mutation checks must clear the numeric noise floor, or "the output moved" is just
/// reduced-precision jitter.
///
/// The measured residuals are rope 1e-7, adaln 8e-4, refiner 2e-3, attention 3e-3, full block
/// 4e-3, so this sits ~2.4× over the deepest one — a narrower margin than the usual 10×, which is
/// why `parity_residuals_bound_the_mutation_floor` re-measures the whole ladder on every run and
/// fails if it ever closes. The mutations themselves all land at ≥1e-1, i.e. ≥24× the residual,
/// and every probe prints its measured value.
const MUTATION_FLOOR: f32 = 1e-2;

fn fixture() -> Weights {
    Weights::from_file(DIT_FIXTURE).unwrap()
}

/// Fixture tensors minus the reference-side extras — i.e. exactly the model weights in the
/// published naming.
fn model_weights() -> Weights {
    let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    w
}

fn rope_of(cfg: &MiniMaxH3DitConfig) -> MmRope {
    MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap()
}

/// `[seq_len]` int32 indices from a float-stored fixture tensor.
fn indices(f: &Weights, key: &str) -> Array {
    f.require(key).unwrap().as_dtype(Dtype::Int32).unwrap()
}

fn block(w: &mut Weights, cfg: &MiniMaxH3DitConfig, index: i32) -> DitBlock {
    DitBlock::from_weights(
        w,
        &format!("transformer_blocks.{index}"),
        cfg,
        Dtype::Float32,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------------------------
// MM-RoPE
// ---------------------------------------------------------------------------------------------

/// The rotary tables themselves, before any block math — `[seq_len, 2·3·rope_freq_dim]`.
#[test]
fn mm_rope_tables_match_the_reference() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let rope = rope_of(&cfg);
    assert_eq!(rope.rotary_dim(), cfg.rotary_dim());
    assert!(
        rope.rotary_dim() < cfg.attention_head_dim,
        "the fixture must exercise the PARTIAL rotary path"
    );

    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    assert_parity(
        &tables.cos,
        f.require("out.rope_cos").unwrap(),
        TOL,
        "rope cos",
    );
    assert_parity(
        &tables.sin,
        f.require("out.rope_sin").unwrap(),
        TOL,
        "rope sin",
    );
}

/// **The `(t, h, w)` axis split, against reference bytes.** Rebuilding the tables from a grid whose
/// three axes have been permuted must NOT reproduce the golden — the strongest available evidence
/// that the concatenation order is `t, h, w` rather than any other permutation, all of which are
/// shape-identical.
#[test]
fn permuting_the_rope_axes_breaks_parity() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let rope = rope_of(&cfg);
    let ids = f.require("layout.position_ids").unwrap();
    let want = f.require("out.rope_sin").unwrap();

    // Baseline agrees.
    let (base, _) = rel(&rope.tables(ids).unwrap().sin, want);
    assert!(base < TOL, "baseline rope must match first ({base:.3e})");

    // (t, h, w) -> (h, w, t) and (t, w, h): both are legal grids and neither is this model.
    for (label, order) in [("h,w,t", [1i32, 2, 0]), ("t,w,h", [0, 2, 1])] {
        let idx = Array::from_slice(&order, &[3]);
        let permuted = ids.take_axis(&idx, 1).unwrap();
        let got = rope.tables(&permuted).unwrap().sin;
        let (peak, _) = rel(&got, want);
        println!("  rope axes {label}: peak rel {peak:.3e}");
        assert!(
            peak > MUTATION_FLOOR,
            "permuting the rope axes to ({label}) reproduced the golden ({peak:.3e}); the axis \
             order is not reaching the tables"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Position conventions — the highest-risk detail in the port
// ---------------------------------------------------------------------------------------------

/// **The audio-token position convention, pinned against the reference's own
/// `build_packed_sequence` output.**
///
/// The whole `[text | audio | video]` grid is rebuilt from `crate::dit::positions` and compared
/// row for row against the fixture's `layout.position_ids`, which the generator obtained by calling
/// `MiniMaxH3PrepareLayoutStep.build_packed_sequence` — not by describing it.
///
/// The audio rows are the reason this test exists: they carry **no height**, ride the video's clock
/// **channel-major**, and their only spatial identity is the two **extremes of the video's width
/// grid**. Zero-everywhere and reuse-the-video-grid are both shape-identical guesses.
#[test]
fn the_packed_position_grid_matches_the_reference() {
    let f = fixture();
    let l = &DIT_LAYOUT;
    let cfg = dit_fixture_config();
    let [_, patch_h, patch_w] = cfg.patch_size;

    let (frame, width) =
        positions::frame_grid(l.latent_height, l.latent_width, patch_h, patch_w).unwrap();

    let mut rows = positions::text_position_ids(l.num_text_tokens).unwrap();
    rows.extend(
        positions::audio_position_ids(
            l.num_text_tokens,
            l.num_audio_latents,
            l.audio_channels,
            &width,
        )
        .unwrap(),
    );
    rows.extend(
        positions::video_position_ids(l.num_text_tokens, l.num_latent_frames, &frame).unwrap(),
    );

    let got = positions::to_array(&rows).unwrap();
    let want = f.require("layout.position_ids").unwrap();
    assert_eq!(got.shape(), want.shape(), "packed sequence length");
    // Exact, not approximate: these are coordinates, and the reference computes them in f64 and
    // narrows once. A 1e-2 relative gate would let a wrong grid through.
    let g: Vec<f32> = got.as_slice::<f32>().to_vec();
    let w: Vec<f32> = want.as_slice::<f32>().to_vec();
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "position row {}, axis {}: got {a}, reference {b}",
            i / 3,
            i % 3
        );
    }

    // ...and the audio rows are specifically the ones this pins.
    let audio_start = (l.num_text_tokens * 3) as usize;
    let audio_rows = (l.num_audio_latents * l.audio_channels) as usize;
    for row in 0..audio_rows {
        let base = audio_start + row * 3;
        assert_eq!(w[base + 1], 0.0, "audio row {row} must carry no height");
        let want_w = if row < l.num_audio_latents as usize {
            width[0] as f32
        } else {
            *width.last().unwrap() as f32
        };
        assert!(
            (w[base + 2] - want_w).abs() < 1e-5,
            "audio row {row} must sit at a width-grid extreme"
        );
    }
    println!(
        "  audio rows: t={:?} w-extremes=({:.4}, {:.4}); interior width points {:?} are unused by \
         audio",
        (0..audio_rows)
            .map(|r| w[audio_start + r * 3])
            .collect::<Vec<_>>(),
        width[0],
        width.last().unwrap(),
        &width[1..width.len() - 1],
    );
}

/// A plausible wrong audio convention — zero width, or the video's full grid — must NOT reproduce
/// the reference rows. Without this, "we wrote down a convention" and "we wrote down *the*
/// convention" are indistinguishable.
#[test]
fn a_plausible_wrong_audio_convention_is_distinguishable() {
    let f = fixture();
    let l = &DIT_LAYOUT;
    let cfg = dit_fixture_config();
    let [_, patch_h, patch_w] = cfg.patch_size;
    let (_, width) =
        positions::frame_grid(l.latent_height, l.latent_width, patch_h, patch_w).unwrap();

    let correct = positions::audio_position_ids(
        l.num_text_tokens,
        l.num_audio_latents,
        l.audio_channels,
        &width,
    )
    .unwrap();

    // (a) w = 0 everywhere.
    let zeroed: Vec<[f64; 3]> = correct.iter().map(|r| [r[0], r[1], 0.0]).collect();
    assert_ne!(zeroed, correct, "audio width is not zero");

    // (b) latent-major instead of channel-major: t would be [5,5,6,6,7,7] rather than [5,6,7,5,6,7].
    let latent_major: Vec<[f64; 3]> = (0..correct.len())
        .map(|i| {
            let latent = i / l.audio_channels as usize;
            [
                l.num_text_tokens as f64 + latent as f64,
                correct[i][1],
                correct[i][2],
            ]
        })
        .collect();
    assert_ne!(
        latent_major, correct,
        "audio rows are channel-major, not latent-major"
    );

    // (c) both interior and extreme width points — i.e. reusing the video grid.
    assert!(
        width.len() > 2,
        "the fixture's width grid must have interior points for this to discriminate"
    );
    let interior = width[1] as f32;
    let audio_start = (l.num_text_tokens * 3) as usize;
    let ids: Vec<f32> = f
        .require("layout.position_ids")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    for row in 0..(l.num_audio_latents * l.audio_channels) as usize {
        assert!(
            (ids[audio_start + row * 3 + 2] - interior).abs() > 1e-5,
            "audio row {row} landed on an INTERIOR width point; it must be pinned to an extreme"
        );
    }
}

/// **The `fl2va` keyframe conditioning rows**, against the reference's own `build_packed_sequence`
/// with `keyframe_anchors = ("first", "last")`.
///
/// A keyframe block is one frame's worth of rows at ONE constant time carrying the full spatial
/// grid — a literal frame pinned to an end of the clip's clock. The `"last"` anchor is the detail
/// worth pinning: it is the time of the FINAL generated latent frame, which is a sum over the
/// non-uniform `5/3 × (1,4,4,4,4)` cycle rather than `num_latent_frames × a constant`.
#[test]
fn the_keyframe_anchor_rows_match_the_reference() {
    let f = fixture();
    let l = &DIT_LAYOUT;
    let cfg = dit_fixture_config();
    let [_, patch_h, patch_w] = cfg.patch_size;
    let (frame, _) =
        positions::frame_grid(l.latent_height, l.latent_width, patch_h, patch_w).unwrap();

    let mut rows = positions::keyframe_position_ids(
        KeyframeAnchor::First,
        l.num_text_tokens,
        l.num_latent_frames,
        &frame,
    )
    .unwrap();
    rows.extend(
        positions::keyframe_position_ids(
            KeyframeAnchor::Last,
            l.num_text_tokens,
            l.num_latent_frames,
            &frame,
        )
        .unwrap(),
    );

    let want = f.require("layout.keyframe_position_ids").unwrap();
    let got = positions::to_array(&rows).unwrap();
    assert_eq!(got.shape(), want.shape(), "two blocks of one frame's rows");
    let g: Vec<f32> = got.as_slice::<f32>().to_vec();
    let w: Vec<f32> = want.as_slice::<f32>().to_vec();
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "keyframe row {}, axis {}: got {a}, reference {b}",
            i / 3,
            i % 3
        );
    }

    // The two anchors must be genuinely different times, and `last` must NOT be a uniform ramp.
    let per_block = frame.len();
    let (first_t, last_t) = (w[0], w[per_block * 3]);
    assert!(last_t > first_t, "the two anchors collapsed to one time");
    let naive = first_t + (l.num_latent_frames - 1) as f32 * positions::ROPE_FRAME_RESCALE as f32;
    assert!(
        (last_t - naive).abs() > 1e-3,
        "a uniform 5/3 ramp reproduced the `last` anchor ({last_t} vs {naive}); the non-uniform \
         (1,4,4,4,4) cycle is not reaching it"
    );
    println!(
        "  keyframe anchors: first={first_t:.4} last={last_t:.4} (uniform-ramp guess {naive:.4})"
    );
}

// ---------------------------------------------------------------------------------------------
// Block-level parity
// ---------------------------------------------------------------------------------------------

/// The six AdaLN modulation tables, `[num_timesteps · MODALITY_NUM, hidden_size]` each.
///
/// This is where the `view(-1, 6·hidden)`-before-`chunk(6)` row layout is pinned: a port that
/// chunked the `[T, 18·hidden]` projection directly produces six tensors of the same total size
/// with different contents.
#[test]
fn adaln_modulation_matches_the_reference() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let b = block(&mut w, &cfg, 0);

    let temb = f.require("in.temb").unwrap();
    let m = b.modulation(temb).unwrap();
    let steps = temb.shape()[0];
    for (name, got) in [
        ("shift_msa", &m.shift_msa),
        ("scale_msa", &m.scale_msa),
        ("gate_msa", &m.gate_msa),
        ("shift_mlp", &m.shift_mlp),
        ("scale_mlp", &m.scale_mlp),
        ("gate_mlp", &m.gate_mlp),
    ] {
        assert_eq!(
            got.shape(),
            &[steps * MODALITY_NUM, cfg.hidden_size],
            "{name}: one row per (timestep, modality)"
        );
        assert_parity(
            got,
            f.require(&format!("out.modulation.{name}")).unwrap(),
            TOL,
            name,
        );
    }
}

/// One `MiniMaxH3TransformerBlock`: pre-norm + AdaLN modulation, qk-normed MM-RoPE attention,
/// gated residual, SwiGLU feed-forward, gated residual.
#[test]
fn transformer_block_matches_the_reference() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let b = block(&mut w, &cfg, 0);

    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let got = b
        .forward_with_temb(
            f.require("in.block.hidden").unwrap(),
            f.require("in.temb").unwrap(),
            &indices(&f, "layout.adaln_indices"),
            &rope,
            &tables,
        )
        .unwrap();
    assert_parity(
        &got,
        f.require("out.block.hidden").unwrap(),
        TOL,
        "dit block",
    );
}

/// Bare attention, so a qk-norm or rotary error localizes instead of only appearing folded into
/// the block's two gated residuals. Both configurations are covered: with the rotary (the block's)
/// and without it (the refiner's).
#[test]
fn attention_matches_the_reference_with_and_without_the_rotary() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let attn =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
            .unwrap();

    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.attn.hidden").unwrap();

    let with_rope = attn.forward(x, Some((&rope, &tables))).unwrap();
    assert_parity(
        &with_rope,
        f.require("out.attn.hidden").unwrap(),
        TOL,
        "attn + rope",
    );

    let without = attn.forward(x, None).unwrap();
    assert_parity(
        &without,
        f.require("out.attn.norope").unwrap(),
        TOL,
        "attn, no rope",
    );

    // ...and the two must genuinely differ, or "the refiner applies no rotary" would be untestable.
    let (peak, _) = rel(&with_rope, &without);
    println!("  rotary on/off: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "applying the rotary changed nothing ({peak:.3e}); it is not reaching attention"
    );
}

// ---------------------------------------------------------------------------------------------
// The 2-layer token refiner — covered, not stubbed
// ---------------------------------------------------------------------------------------------

/// Both refiner blocks and the `final_norm`, against the reference.
#[test]
fn token_refiner_matches_the_reference() {
    let f = fixture();
    let cfg = dit_fixture_config();

    let mut w = model_weights();
    let first = TokenRefinerBlock::from_weights(
        &mut w,
        "token_refiner.refiner_blocks.0",
        &cfg,
        Dtype::Float32,
    )
    .unwrap();
    let x = f.require("in.refiner.hidden").unwrap();
    assert_parity(
        &first.forward(x).unwrap(),
        f.require("out.refiner.block0").unwrap(),
        TOL,
        "refiner block 0",
    );

    let mut w = model_weights();
    let refiner =
        TokenRefiner::from_weights(&mut w, "token_refiner", &cfg, Dtype::Float32).unwrap();
    assert_eq!(refiner.num_layers(), 2, "the refiner has two blocks");
    let got = refiner.forward(x).unwrap();
    assert_parity(
        &got,
        f.require("out.refiner.hidden").unwrap(),
        TOL,
        "token refiner",
    );

    // Not a stub: the stack must move its input, and the second block must contribute on top of
    // the first.
    let (vs_input, _) = rel(&got, x);
    let (vs_block0, _) = rel(&got, f.require("out.refiner.block0").unwrap());
    println!("  refiner vs input: {vs_input:.3e}; full stack vs block 0 only: {vs_block0:.3e}");
    assert!(
        vs_input > MUTATION_FLOOR,
        "the refiner is an identity ({vs_input:.3e})"
    );
    assert!(
        vs_block0 > MUTATION_FLOOR,
        "block 1 and final_norm contribute nothing ({vs_block0:.3e}); a 1-layer refiner would pass"
    );
}

/// Every refiner tensor must be load-bearing. The refiner is 21 of the 638 published tensors and
/// the easiest thing in this port to leave half-wired; the lightx2v turbo LoRAs (sc-18724) target
/// 24 tensors here, and a partially-wired refiner would silently drop them.
#[test]
fn every_refiner_weight_is_load_bearing() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let x = f.require("in.refiner.hidden").unwrap();

    let mut w = model_weights();
    let baseline = TokenRefiner::from_weights(&mut w, "token_refiner", &cfg, Dtype::Float32)
        .unwrap()
        .forward(x)
        .unwrap();

    for key in TokenRefiner::names("token_refiner", &cfg) {
        let mut w = model_weights();
        let original = w.require(&key).unwrap().clone();
        let bumped = mlx_rs::ops::add(
            mlx_rs::ops::multiply(&original, Array::from_f32(1.5)).unwrap(),
            Array::from_f32(0.5),
        )
        .unwrap();
        w.insert(&key, bumped);
        let mutated = TokenRefiner::from_weights(&mut w, "token_refiner", &cfg, Dtype::Float32)
            .unwrap()
            .forward(x)
            .unwrap();
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e}");
        assert!(
            peak > MUTATION_FLOOR,
            "perturbing {key} moved the refiner by only {peak:.3e}; it is not wired in"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// qk-norm: placement, shape and eps — tested by BEHAVIOUR
// ---------------------------------------------------------------------------------------------

/// **qk-norm placement.** `norm_q` / `norm_k` run per head, after the unflatten and **before** the
/// rotary.
///
/// The video VAE's qk-norm is non-affine and leaves no tensors at all, so "there is a
/// `norm_q.weight` in the index" proves nothing about whether a port applies it, where, or over
/// what axis. This builds the **rope-first** variant by hand from the fixture's own tensors and
/// requires it to miss the golden, while the port's norm-first order hits it.
///
/// The two orders are genuinely distinguishable only because the norm weight is non-constant: the
/// rotary mixes channels, so a per-channel scale does not commute with it.
#[test]
fn qk_norm_runs_per_head_before_the_rotary() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let attn =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
            .unwrap();
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.attn.hidden").unwrap();
    let want = f.require("out.attn.hidden").unwrap();

    // The port hits the golden.
    let (base, _) = rel(&attn.forward(x, Some((&rope, &tables))).unwrap(), want);
    assert!(
        base < TOL,
        "baseline attention must match first ({base:.3e})"
    );

    // The norm weight must be non-constant, or neither of the variants below can discriminate.
    let nq = f
        .require("transformer_blocks.0.attn.norm_q.weight")
        .unwrap();
    assert_eq!(nq.shape(), &[cfg.attention_head_dim], "qk-norm is per HEAD");
    assert!(
        std_dev(nq) > 1e-2,
        "norm_q is ~constant ({:.3e}); this test could not tell the two orders apart",
        std_dev(nq)
    );

    // Hand-built variants over the same weights: `rope_first` and `no_norm`.
    let (b, seq) = (x.shape()[0], x.shape()[1]);
    let shape = [b, seq, cfg.num_attention_heads, cfg.attention_head_dim];
    let proj = |name: &str| -> Array {
        let t = f
            .require(&format!("transformer_blocks.0.attn.{name}.weight"))
            .unwrap();
        mlx_rs::ops::matmul(x, t.t())
            .unwrap()
            .reshape(&shape)
            .unwrap()
    };
    let norm =
        |t: &Array, weight: &Array| mlx_rs::fast::rms_norm(t, weight, cfg.qk_norm_eps).unwrap();
    let nk = f
        .require("transformer_blocks.0.attn.norm_k.weight")
        .unwrap();
    let (q, k, v) = (proj("to_q"), proj("to_k"), proj("to_v"));

    let finish = |q: Array, k: Array| -> Array {
        let t = |a: &Array| a.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let out = mlx_rs::fast::scaled_dot_product_attention(
            t(&q),
            t(&k),
            t(&v),
            1.0 / (cfg.attention_head_dim as f32).sqrt(),
            None,
            None,
        )
        .unwrap();
        let flat = out
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap()
            .reshape(&[b, seq, cfg.inner_dim()])
            .unwrap();
        let wo = f
            .require("transformer_blocks.0.attn.to_out.0.weight")
            .unwrap();
        mlx_rs::ops::matmul(&flat, wo.t()).unwrap()
    };

    let rope_first = finish(
        norm(&rope.apply(&q, &tables).unwrap(), nq),
        norm(&rope.apply(&k, &tables).unwrap(), nk),
    );
    let no_norm = finish(
        rope.apply(&q, &tables).unwrap(),
        rope.apply(&k, &tables).unwrap(),
    );

    for (label, variant) in [
        ("rope-then-norm", &rope_first),
        ("no qk-norm at all", &no_norm),
    ] {
        let (peak, _) = rel(variant, want);
        let cos = cosine(variant, want);
        println!(
            "  {label}: peak rel {peak:.3e} cosine {cos:.4} ||variant||={:.4} ||golden||={:.4}",
            l2_norm(variant),
            l2_norm(want)
        );
        assert!(
            peak > MUTATION_FLOOR,
            "the `{label}` variant still reproduced the golden ({peak:.3e}); this suite cannot \
             pin the qk-norm"
        );
    }
}

/// **All three epsilons must reach their norms** — `qk_norm_eps`, `norm_eps` and `final_norm_eps`.
///
/// Gated at `eps = 1.0` rather than at a near-miss: with unit-RMS inputs the difference between
/// 1e-5 and 1e-4 is far under this suite's ~4e-3 reduced-precision floor, so a near-miss test would
/// be a false green. What IS provable is that each field is *wired* — a port that hard-coded any
/// epsilon, or crossed two of them, fails here.
///
/// Crossing them matters even though the shipped config sets all three to 1e-5: they are separate
/// knobs (`refiner.rs` deliberately reads `final_norm_eps` for `final_norm` and `norm_eps` for the
/// block norms), so a variant checkpoint that diverged would otherwise run silently wrong.
#[test]
fn every_norm_epsilon_is_wired_through() {
    let f = fixture();
    let cfg = dit_fixture_config();
    assert_eq!(cfg.qk_norm_eps, 1e-5, "the published value");
    assert_eq!(cfg.norm_eps, 1e-5);
    assert_eq!(cfg.final_norm_eps, 1e-5);

    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let attn_in = f.require("in.attn.hidden").unwrap();
    let refiner_in = f.require("in.refiner.hidden").unwrap();
    let block_in = f.require("in.block.hidden").unwrap();
    let temb = f.require("in.temb").unwrap();
    let idx = indices(&f, "layout.adaln_indices");

    // Each epsilon is probed through the module that actually reads it.
    let attn_out = |c: &MiniMaxH3DitConfig| {
        let mut w = model_weights();
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", c, Dtype::Float32)
            .unwrap()
            .forward(attn_in, Some((&rope, &tables)))
            .unwrap()
    };
    let block_out = |c: &MiniMaxH3DitConfig| {
        let mut w = model_weights();
        block(&mut w, c, 0)
            .forward_with_temb(block_in, temb, &idx, &rope, &tables)
            .unwrap()
    };
    let refiner_out = |c: &MiniMaxH3DitConfig| {
        let mut w = model_weights();
        TokenRefiner::from_weights(&mut w, "token_refiner", c, Dtype::Float32)
            .unwrap()
            .forward(refiner_in)
            .unwrap()
    };

    /// `(label, how to make the epsilon loud, which module reads it)`.
    type EpsProbe<'a> = (
        &'a str,
        fn(&mut MiniMaxH3DitConfig),
        &'a dyn Fn(&MiniMaxH3DitConfig) -> Array,
    );
    let probes: [EpsProbe; 3] = [
        ("qk_norm_eps", |c| c.qk_norm_eps = 1.0, &attn_out),
        ("norm_eps", |c| c.norm_eps = 1.0, &block_out),
        ("final_norm_eps", |c| c.final_norm_eps = 1.0, &refiner_out),
    ];
    for (name, mutate, run) in probes {
        let baseline = run(&cfg);
        let mut loud = cfg.clone();
        mutate(&mut loud);
        let (peak, _) = rel(&run(&loud), &baseline);
        println!("  {name} 1e-5 -> 1.0: peak rel {peak:.3e}");
        assert!(
            peak > MUTATION_FLOOR,
            "changing {name} moved its module by only {peak:.3e}; the config field is inert"
        );
    }
}

/// A qk-norm sized to the 7168-wide projection instead of to a 128-wide head is the plausible wrong
/// shape, and it would broadcast silently. Rejected at load.
#[test]
fn a_projection_sized_qk_norm_is_rejected() {
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    w.insert(
        "transformer_blocks.0.attn.norm_q.weight",
        Array::ones::<f32>(&[cfg.inner_dim()]).unwrap(),
    );
    let e = DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
        .unwrap_err()
        .to_string();
    assert!(e.contains("norm_q"), "unexpected error: {e}");
}

// ---------------------------------------------------------------------------------------------
// Rule 1 — the gated-FFN layout contract
// ---------------------------------------------------------------------------------------------

/// Swap the two row halves of a `[2·inner, dim]` fused gated projection.
fn swap_halves(t: &Array) -> Array {
    let rows = t.shape()[0];
    let parts = t.split_axis(&[rows / 2], 0).unwrap();
    mlx_rs::ops::concatenate_axis(&[parts[1].clone(), parts[0].clone()], 0).unwrap()
}

/// **The DiT carries the same `[value | gate]` layout the video VAE does.** The fixture holds both
/// forms: `…ff.net.0.proj.weight` as the converted checkpoint holds it, and
/// `src.…mlp.fc1.weight` as the original MiniMax modules do (verified in the generator by running
/// the OFFICIAL `convert_transformer_key` forward on it and requiring the published tensor back
/// byte-for-byte).
///
/// The `assert_ne!` is the half that matters: a regeneration through a pure rename makes the two
/// forms equal and fails here rather than quietly passing everything else.
#[test]
fn published_ffn_projection_is_value_then_gate() {
    let f = fixture();
    let cfg = dit_fixture_config();
    assert_eq!(
        PUBLISHED_GATED_FFN_LAYOUT,
        GatedFfnLayout::ValueFirst,
        "the crate's layout contract"
    );

    let mut prefixes: Vec<String> = (0..cfg.num_layers)
        .map(|i| format!("transformer_blocks.{i}"))
        .collect();
    prefixes
        .extend((0..cfg.num_refiner_layers).map(|i| format!("token_refiner.refiner_blocks.{i}")));

    for prefix in prefixes {
        let published = f
            .require(&format!("{prefix}.ff.net.0.proj.weight"))
            .unwrap();
        let source = f.require(&format!("src.{prefix}.mlp.fc1.weight")).unwrap();
        assert_eq!(
            published.shape(),
            source.shape(),
            "shapes are identical — that is the hazard"
        );
        assert_eq!(
            published.as_slice::<f32>(),
            swap_halves(source).as_slice::<f32>(),
            "{prefix}: the published projection must be `mlp.fc1` with its halves SWAPPED"
        );
        assert_ne!(
            published.as_slice::<f32>(),
            source.as_slice::<f32>(),
            "{prefix}.ff.net.0.proj is byte-equal to the pre-conversion `mlp.fc1` — this fixture \
             was dumped through a pure rename with no half-swap, so it tests the port against a \
             layout production never loads (sc-18740). Re-run tools/dump_minimax_h3_dit.py."
        );
    }
}

/// **Mutation gate for the half-swap.** Handing the loader the source layout must break parity —
/// and the printed norms and cosine show why a magnitude, std or checksum assertion of any
/// tolerance would not.
#[test]
fn reading_the_ffn_halves_gate_first_breaks_parity() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.block.hidden").unwrap();
    let temb = f.require("in.temb").unwrap();
    let idx = indices(&f, "layout.adaln_indices");
    let want = f.require("out.block.hidden").unwrap();

    let mut w = model_weights();
    let baseline = block(&mut w, &cfg, 0)
        .forward_with_temb(x, temb, &idx, &rope, &tables)
        .unwrap();
    let (base_peak, _) = rel(&baseline, want);

    let mut w = model_weights();
    let key = "transformer_blocks.0.ff.net.0.proj.weight";
    let swapped = swap_halves(w.require(key).unwrap());
    w.insert(key, swapped);
    let mutated = block(&mut w, &cfg, 0)
        .forward_with_temb(x, temb, &idx, &rope, &tables)
        .unwrap();

    let (peak, mean) = rel(&mutated, &baseline);
    let cos = cosine(&mutated, &baseline);
    println!(
        "FFN half-swap: rel-max-abs={peak:.3e} rel-mean={mean:.3e} cosine={cos:.4} \
         ||correct||={:.4} ||swapped||={:.4}  (parity residual with the correct layout: \
         {base_peak:.3e})",
        l2_norm(&baseline),
        l2_norm(&mutated),
    );
    assert!(
        peak > MUTATION_FLOOR,
        "swapping the gate and value halves moved the block by only {peak:.3e}"
    );
    let (vs_ref, _) = rel(&mutated, want);
    assert!(
        vs_ref > TOL,
        "the source-layout block still matches the reference within {TOL:.0e} ({vs_ref:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Rule 2 — the fused-QKV transform is NOT the video VAE's
// ---------------------------------------------------------------------------------------------

/// **Which transform produced the published `to_q`/`to_k`/`to_v`.**
///
/// The fixture carries the raw per-head-interleaved fused tensor and its reordered
/// `[q_all; k_all; v_all]` form, both round-tripped through the official conversion functions in
/// the generator. This asserts, against those bytes:
///
/// 1. contiguous thirds of the **reordered** tensor reproduce the published split — the DiT's
///    transform;
/// 2. the video VAE's interleaved gather applied to the **raw** tensor reproduces it too — the two
///    routes compose to the same thing, which is what the conversion script claims;
/// 3. **crossing them does not** — contiguous thirds of the raw rows, and an interleaved gather of
///    already-reordered rows, are both shape-identical and wrong.
///
/// Point 3 is the whole test. Without it, "we picked a transform" and "we picked the right one"
/// are indistinguishable.
#[test]
fn published_qkv_is_contiguous_thirds_of_the_reordered_fused_projection() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let (heads, head_dim) = (cfg.num_attention_heads, cfg.attention_head_dim);

    let mut prefixes: Vec<String> = (0..cfg.num_layers)
        .map(|i| format!("transformer_blocks.{i}"))
        .collect();
    prefixes
        .extend((0..cfg.num_refiner_layers).map(|i| format!("token_refiner.refiner_blocks.{i}")));

    for prefix in prefixes {
        let raw = f
            .require(&format!("src.{prefix}.attn.qkv_proj.weight"))
            .unwrap();
        let reordered = f
            .require(&format!("src.{prefix}.attn.qkv_proj.reordered"))
            .unwrap();
        let published: Vec<Array> = ["to_q", "to_k", "to_v"]
            .iter()
            .map(|n| {
                f.require(&format!("{prefix}.attn.{n}.weight"))
                    .unwrap()
                    .clone()
            })
            .collect();

        // The reorder itself.
        assert_eq!(
            qkv::reorder_interleaved(raw, heads, head_dim)
                .unwrap()
                .as_slice::<f32>(),
            reordered.as_slice::<f32>(),
            "{prefix}: reorder_interleaved must reproduce the reference's in-memory layout"
        );

        // 1 & 2 — both correct routes agree with the published tensors.
        let thirds = qkv::split_thirds(reordered, heads, head_dim).unwrap();
        let gather = split_fused_qkv(raw, heads, head_dim).unwrap();
        for i in 0..3 {
            assert_eq!(
                thirds[i].as_slice::<f32>(),
                published[i].as_slice::<f32>(),
                "{prefix}: contiguous thirds of the REORDERED tensor is the DiT's transform"
            );
            assert_eq!(
                gather[i].as_slice::<f32>(),
                published[i].as_slice::<f32>(),
                "{prefix}: the composed gather must agree"
            );
        }

        // 3 — the crossed applications are shape-identical and wrong.
        let wrong_thirds = qkv::split_thirds(raw, heads, head_dim).unwrap();
        let wrong_gather = split_fused_qkv(reordered, heads, head_dim).unwrap();
        for i in 0..3 {
            assert_eq!(wrong_thirds[i].shape(), published[i].shape());
            assert_ne!(
                wrong_thirds[i].as_slice::<f32>(),
                published[i].as_slice::<f32>(),
                "{prefix}: contiguous thirds of the RAW rows must not equal the published split — \
                 a fixture dumped from raw shards needs the interleaved gather instead"
            );
            assert_ne!(
                wrong_gather[i].as_slice::<f32>(),
                published[i].as_slice::<f32>(),
                "{prefix}: the video VAE's interleaved gather applied to an ALREADY-REORDERED \
                 state_dict must not equal the published split"
            );
        }
    }
}

/// Feeding the attention a QKV set produced by the *wrong* transform must break parity — the
/// numeric consequence of the contract above, measured rather than assumed.
#[test]
fn the_wrong_qkv_transform_breaks_parity() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let (heads, head_dim) = (cfg.num_attention_heads, cfg.attention_head_dim);
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.attn.hidden").unwrap();
    let want = f.require("out.attn.hidden").unwrap();

    let mut w = model_weights();
    let baseline =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
            .unwrap()
            .forward(x, Some((&rope, &tables)))
            .unwrap();
    let (base, _) = rel(&baseline, want);
    assert!(base < TOL, "baseline must match first ({base:.3e})");

    // Contiguous thirds of the RAW rows — the transform a state_dict fixture needs, applied to
    // raw-shard bytes.
    let raw = f
        .require("src.transformer_blocks.0.attn.qkv_proj.weight")
        .unwrap();
    let wrong = qkv::split_thirds(raw, heads, head_dim).unwrap();
    let mut w = model_weights();
    for (part, name) in wrong.iter().zip(["to_q", "to_k", "to_v"]) {
        w.insert(
            format!("transformer_blocks.0.attn.{name}.weight"),
            part.clone(),
        );
    }
    let mutated =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
            .unwrap()
            .forward(x, Some((&rope, &tables)))
            .unwrap();
    let (peak, _) = rel(&mutated, &baseline);
    let cos = cosine(&mutated, &baseline);
    println!("  wrong qkv transform: peak rel {peak:.3e} cosine {cos:.4}");
    assert!(
        peak > MUTATION_FLOOR,
        "the wrong QKV partition moved attention by only {peak:.3e}"
    );
}

// ---------------------------------------------------------------------------------------------
// The AdaLN row addressing
// ---------------------------------------------------------------------------------------------

/// **`adaln_indices = timestep_index · MODALITY_NUM + tag`.** The modality term must be present:
/// dropping it (addressing by timestep alone) is a legal index vector that selects the wrong rows.
#[test]
fn the_modality_term_of_the_adaln_index_is_load_bearing() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.block.hidden").unwrap();
    let temb = f.require("in.temb").unwrap();
    let want = f.require("out.block.hidden").unwrap();

    let mut w = model_weights();
    let b = block(&mut w, &cfg, 0);

    // The fixture's tags must actually vary, or nothing below discriminates.
    let tags: Vec<f32> = f
        .require("layout.token_tags")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let distinct: std::collections::BTreeSet<i32> = tags.iter().map(|&t| t as i32).collect();
    assert_eq!(
        distinct.len(),
        3,
        "the golden must carry all three modalities"
    );
    let steps: Vec<f32> = f
        .require("layout.timestep_indices")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(
        steps.iter().any(|&s| s != steps[0]),
        "the golden must carry more than one timestep"
    );

    let correct = indices(&f, "layout.adaln_indices");
    let (base, _) = rel(
        &b.forward_with_temb(x, temb, &correct, &rope, &tables)
            .unwrap(),
        want,
    );
    assert!(base < TOL, "baseline must match first ({base:.3e})");

    // Timestep only — every row of a step now reads the video modality's parameters.
    let by_step: Vec<i32> = steps.iter().map(|&s| s as i32 * MODALITY_NUM).collect();
    // Modality only — every row reads timestep 0's parameters.
    let by_tag: Vec<i32> = tags.iter().map(|&t| t as i32).collect();
    for (label, idx) in [("timestep only", by_step), ("modality only", by_tag)] {
        let idx = Array::from_slice(&idx, &[idx.len() as i32]);
        let got = b.forward_with_temb(x, temb, &idx, &rope, &tables).unwrap();
        let (peak, _) = rel(&got, want);
        println!("  adaln index `{label}`: peak rel {peak:.3e}");
        assert!(
            peak > MUTATION_FLOOR,
            "addressing the AdaLN table by `{label}` reproduced the golden ({peak:.3e})"
        );
    }
}

/// **The `view(-1, 6·hidden)`-before-`chunk(6)` row layout.** Chunking the `[T, 18·hidden]`
/// projection into six `[T, 3·hidden]` pieces instead yields six tensors of the same total size and
/// different contents; this reconstructs that wrong reading from the fixture's own weights and
/// shows the six tables differ.
#[test]
fn the_adaln_table_is_reshaped_before_it_is_chunked() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let b = block(&mut w, &cfg, 0);
    let temb = f.require("in.temb").unwrap();
    let m = b.modulation(temb).unwrap();

    // The raw projection, `[T, 6·3·hidden]`.
    let weight = f
        .require("transformer_blocks.0.adaln_proj.linear.weight")
        .unwrap();
    let bias = f
        .require("transformer_blocks.0.adaln_proj.linear.bias")
        .unwrap();
    let activated = mlx_rs::ops::multiply(temb, mlx_rs::ops::sigmoid(temb).unwrap()).unwrap();
    let projected = mlx_gen::nn::linear(&activated, weight, bias).unwrap();
    let steps = temb.shape()[0];
    assert_eq!(projected.shape(), &[steps, cfg.adaln_out_features()]);

    // The WRONG reading: chunk the column axis into 6 first, then reshape each piece.
    let width = MODALITY_NUM * cfg.hidden_size;
    let wrong_shift = projected
        .split_axis(&[width, 2 * width, 3 * width, 4 * width, 5 * width], 1)
        .unwrap()[0]
        .reshape(&[steps * MODALITY_NUM, cfg.hidden_size])
        .unwrap();
    assert_eq!(wrong_shift.shape(), m.shift_msa.shape(), "shape-identical");
    let (peak, _) = rel(&wrong_shift, &m.shift_msa);
    println!("  adaln chunk-before-reshape: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "chunking before the reshape produced the same shift_msa ({peak:.3e}); the row layout is \
         not pinned"
    );
}

// ---------------------------------------------------------------------------------------------
// Mutation + false-green guards
// ---------------------------------------------------------------------------------------------

/// Every one of a block's 12 tensors must move the output. A tensor that can be changed without
/// changing the result is not wired into the graph and the parity test is not covering it.
#[test]
fn every_block_weight_is_load_bearing() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.block.hidden").unwrap();
    let temb = f.require("in.temb").unwrap();
    let idx = indices(&f, "layout.adaln_indices");

    let mut w = model_weights();
    let baseline = block(&mut w, &cfg, 0)
        .forward_with_temb(x, temb, &idx, &rope, &tables)
        .unwrap();

    let names = DitBlock::names("transformer_blocks.0");
    assert_eq!(names.len(), 12, "the published block ships 12 tensors");
    for key in names {
        let mut w = model_weights();
        let original = w.require(&key).unwrap().clone();
        // Scale AND shift: the shift makes an all-zero tensor observable, the scale keeps the
        // perturbation proportionate for tensors that are already large.
        let bumped = mlx_rs::ops::add(
            mlx_rs::ops::multiply(&original, Array::from_f32(1.5)).unwrap(),
            Array::from_f32(0.5),
        )
        .unwrap();
        w.insert(&key, bumped);
        let mutated = block(&mut w, &cfg, 0)
            .forward_with_temb(x, temb, &idx, &rope, &tables)
            .unwrap();
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e}");
        assert!(
            peak > MUTATION_FLOOR,
            "perturbing {key} moved the block by only {peak:.3e}; it is either not wired into the \
             graph or no longer observable here"
        );
    }
}

/// Measure and print the parity residual of every stage, and require the whole ladder to sit well
/// under [`MUTATION_FLOOR`].
///
/// This is the test that makes the rest of the file's margins auditable rather than implied: a
/// mutation gate is only meaningful relative to the numeric floor it is measured against, and that
/// floor is a property of the hardware and dtype, not of the port. If MLX's reduced-precision f32
/// matmul ever drifts up to meet the floor, this fails here instead of quietly turning every
/// mutation test into a coin flip.
#[test]
fn parity_residuals_bound_the_mutation_floor() {
    let f = fixture();
    let cfg = dit_fixture_config();
    let rope = rope_of(&cfg);
    let ids = f.require("layout.position_ids").unwrap();
    let tables = rope.tables(ids).unwrap();
    let idx = indices(&f, "layout.adaln_indices");
    let temb = f.require("in.temb").unwrap();

    let mut w = model_weights();
    let attn =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
            .unwrap();
    let mut w = model_weights();
    let b = block(&mut w, &cfg, 0);
    let mut w = model_weights();
    let refiner =
        TokenRefiner::from_weights(&mut w, "token_refiner", &cfg, Dtype::Float32).unwrap();

    let attn_in = f.require("in.attn.hidden").unwrap();
    let stages: [(&str, Array, &Array); 5] = [
        (
            "rope cos",
            tables.cos.clone(),
            f.require("out.rope_cos").unwrap(),
        ),
        (
            "attention",
            attn.forward(attn_in, Some((&rope, &tables))).unwrap(),
            f.require("out.attn.hidden").unwrap(),
        ),
        (
            "adaln modulation",
            b.modulation(temb).unwrap().gate_msa,
            f.require("out.modulation.gate_msa").unwrap(),
        ),
        (
            "token refiner",
            refiner
                .forward(f.require("in.refiner.hidden").unwrap())
                .unwrap(),
            f.require("out.refiner.hidden").unwrap(),
        ),
        (
            "transformer block",
            b.forward_with_temb(
                f.require("in.block.hidden").unwrap(),
                temb,
                &idx,
                &rope,
                &tables,
            )
            .unwrap(),
            f.require("out.block.hidden").unwrap(),
        ),
    ];

    let mut worst = 0.0f32;
    for (name, got, want) in &stages {
        let (peak, mean) = rel(got, want);
        println!("  {name:<18} peak rel {peak:.3e}  mean rel {mean:.3e}");
        worst = worst.max(peak);
    }
    println!(
        "  worst residual {worst:.3e}; mutation floor {MUTATION_FLOOR:.1e} ({:.1}x above it)",
        MUTATION_FLOOR / worst
    );
    assert!(worst < TOL, "parity residual {worst:.3e} exceeds {TOL:.0e}");
    assert!(
        MUTATION_FLOOR > 2.0 * worst,
        "the mutation floor {MUTATION_FLOOR:.1e} is not clear of the {worst:.3e} numeric residual; \
         every mutation gate in this file would be measuring jitter"
    );
}

/// The fixture must record which reference path produced it, so a regeneration that reverts to the
/// reference-module method fails rather than passing (sc-18740's Rule 3).
#[test]
fn fixture_provenance_records_the_converted_path() {
    let f = fixture();
    let meta = |k: &str| {
        f.metadata(k)
            .unwrap_or_else(|| {
                panic!(
                    "fixture metadata is missing `{k}`; re-run tools/dump_minimax_h3_dit.py against \
                     diffusers `main` (a fixture with no provenance cannot be shown to come from \
                     the converted-checkpoint path — sc-18740)"
                )
            })
            .to_string()
    };
    assert_eq!(meta("provenance"), "converted-checkpoint");
    assert_eq!(meta("reference"), "diffusers.MiniMaxH3Transformer3DModel");
    assert_eq!(meta("gated_ffn_layout"), "value_first");
    assert_eq!(meta("qkv_transform"), "contiguous_thirds_of_reordered");
    let swap: f32 = meta("ffn_half_swap_rel").parse().unwrap();
    assert!(
        swap > MUTATION_FLOOR,
        "the generator measured the half-swap as only {swap:.3e} — it would not be gateable"
    );
    println!(
        "fixture provenance: {} {} (half-swap {swap:.3e})",
        meta("reference"),
        meta("reference_version")
    );
}

/// A golden whose expected output is near-constant proves nothing, and neither does one whose
/// modulation parameters are at their zero initialization — the reference's `_init_weights` would
/// leave the gates at zero, making every block an exact identity that a stub would reproduce.
#[test]
fn fixture_is_not_degenerate() {
    let f = fixture();
    for key in [
        "out.block.hidden",
        "out.attn.hidden",
        "out.refiner.hidden",
        "out.rope_sin",
        "out.modulation.gate_msa",
        "out.modulation.scale_mlp",
    ] {
        let t = f.require(key).unwrap();
        assert!(
            std_dev(t) > 1e-3,
            "{key} is ~constant ({:.3e}); a constant golden is a false green",
            std_dev(t)
        );
    }
    // The gates specifically must not be zero, or the residual branches are inert.
    for key in ["out.modulation.gate_msa", "out.modulation.gate_mlp"] {
        let max: f32 = f
            .require(key)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(
            max > 1e-3,
            "{key} is ~zero ({max:.3e}); the block would be an identity"
        );
    }
    // The block must actually move its input.
    let (peak, _) = rel(
        f.require("out.block.hidden").unwrap(),
        f.require("in.block.hidden").unwrap(),
    );
    assert!(
        peak > MUTATION_FLOOR,
        "the golden block is an identity ({peak:.3e})"
    );
}

/// The declared tensor names must be exactly the fixture's block + refiner tensor set, and every
/// one must be consumed. A silently unmapped tensor still produces plausible output.
#[test]
fn weight_mapping_is_exhaustive_over_the_block_stack() {
    let cfg = dit_fixture_config();

    let mut declared: std::collections::BTreeSet<String> = (0..cfg.num_layers)
        .flat_map(|i| DitBlock::names(&format!("transformer_blocks.{i}")))
        .collect();
    declared.extend(TokenRefiner::names("token_refiner", &cfg));

    let mut w = model_weights();
    let published: std::collections::BTreeSet<String> = w
        .keys()
        .filter(|k| k.starts_with("transformer_blocks.") || k.starts_with("token_refiner."))
        .map(str::to_string)
        .collect();
    assert_eq!(
        declared, published,
        "declared block-stack tensor names differ from the checkpoint's"
    );

    // ...and loading really does consume all of them.
    for i in 0..cfg.num_layers {
        let _ = block(&mut w, &cfg, i);
    }
    let _ = TokenRefiner::from_weights(&mut w, "token_refiner", &cfg, Dtype::Float32).unwrap();
    w.remove_accessed();
    let leftover: Vec<&str> = w
        .keys()
        .filter(|k| k.starts_with("transformer_blocks.") || k.starts_with("token_refiner."))
        .collect();
    assert!(leftover.is_empty(), "never read: {leftover:?}");
}
