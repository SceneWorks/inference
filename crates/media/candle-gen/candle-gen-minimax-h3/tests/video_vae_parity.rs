//! sc-17154: candle video-VAE decode parity — against the **official diffusers**
//! `AutoencoderKLMiniMaxH3`, i.e. the converted checkpoint production actually loads.
//!
//! Fixture `tests/fixtures/video_vae_decode.safetensors` ← the MLX lane's
//! `tools/dump_minimax_h3_video_vae.py`, copied byte-for-byte (`cross_backend.rs` asserts that).
//!
//! # Why the reference class matters (sc-18740)
//!
//! This fixture was originally dumped from the MiniMax reference modules shipped inside the
//! snapshot (`FL2VA/video_vae/*.py`) with a **pure rename** onto the published key names. That made
//! the MLX suite a false green: the official conversion swaps the two halves of every gated FFN
//! projection, so the fixture carried the *source* layout under *published* names, the loader read
//! it the source way, they agreed, and the shipped 36-layer decoder was wrong on real weights by
//! 0.86-0.99 relative max-abs-diff per block. See [`candle_gen_minimax_h3::layout`].
//!
//! `fixture_provenance_records_the_converted_path` and
//! `published_ffn_projection_is_value_then_gate` below make a silent revert to the old method fail
//! in this lane too, and `reading_the_ffn_halves_gate_first_breaks_parity` proves the assertion is
//! not vacuous by actually performing the mutation and measuring how far the decode moves.
//!
//! # Tolerance, and what it can resolve
//!
//! **1e-5 peak-relative**, deliberately three orders TIGHTER than the MLX sibling's 1e-2 house
//! value. That is not a stricter standard applied to the same hardware — it is the same standard
//! applied to a much quieter one: MLX pays for Metal's reduced-precision f32 matmul and measures a
//! ~1e-3 residual, while this lane runs f32 on the CPU and measures **2.1e-7 … 3.7e-7** across
//! every golden in this file. Keeping 1e-2 here would have left a gate five orders above its own
//! noise floor, which is a gate in name only.
//!
//! So this suite resolves roughly **30× above f32 round-off**, and
//! `an_eps_displacement_the_size_of_a_misplaced_rms_epsilon_is_detected` measures the one
//! sub-percent divergence class that matters most for this crate — an RMSNorm epsilon applied
//! outside the square root rather than inside — rather than asserting it is covered.
//!
//! A structural error — wrong QKV split, a different rotary fraction, LayerNorm instead of
//! RMSNorm, missing register tokens, a mis-planned chunk seam, the sc-18740 gated-FFN half-swap —
//! diverges by orders of magnitude, which the mutation tests at the bottom confirm by measuring and
//! PRINTING how far each perturbation actually moves the decode.

mod common;

use std::collections::BTreeSet;

use common::{
    assert_parity, cosine, fixture_config, flat, l2_norm, rel, std_dev, weights, Golden, FIXTURE,
};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_minimax_h3::blocks::TransformerBlock;
use candle_gen_minimax_h3::{
    split_fused_qkv, swap_gated_halves, GatedFfnLayout, MiniMaxH3VaeConfig, MiniMaxH3VideoVae,
    Rope3d, ViT3dDecoder, PUBLISHED_GATED_FFN_LAYOUT,
};

/// Peak-relative parity bound. Observed residual across this file is 2.1e-7 … 3.7e-7 (candle CPU
/// f32), so this sits ~30× above the measured floor. See the module docs for why it is three orders
/// tighter than the MLX lane's.
const TOL: f32 = 1e-5;

/// Mutation checks must clear the numeric noise floor by a wide margin, or "the output moved"
/// would just be reduced-precision jitter. 1e-2 matches the MLX lane's floor even though candle's
/// CPU-f32 residual is smaller, so the two suites gate the same defects at the same bar.
const MUTATION_FLOOR: f32 = 1e-2;

/// The reference-side extras: pre-conversion tensors and the recorded activations/statistics.
const NON_MODEL_PREFIXES: [&str; 4] = ["src.", "in.", "out.", "const."];

fn fixture() -> Golden {
    Golden::load(FIXTURE)
}

/// Exactly the model weights in the published root naming — what the loader consumes for real
/// weights.
fn model_map(f: &Golden) -> std::collections::HashMap<String, Tensor> {
    f.model_map(&NON_MODEL_PREFIXES)
}

fn vae_from(map: std::collections::HashMap<String, Tensor>, token_drop: i32) -> MiniMaxH3VideoVae {
    MiniMaxH3VideoVae::from_weights(
        &weights(map),
        &fixture_config(token_drop),
        &Device::Cpu,
        DType::F32,
    )
    .expect("build the video VAE")
}

fn vae(f: &Golden, token_drop: i32) -> MiniMaxH3VideoVae {
    vae_from(model_map(f), token_drop)
}

// ---------------------------------------------------------------------------------------------
// Layer-by-layer parity
// ---------------------------------------------------------------------------------------------

/// One transformer block: RMSNorm pre-norms, non-affine q/k RMSNorm, 3-D partial RoPE, SwiGLU
/// FFN and the two scaled residuals.
#[test]
fn transformer_block_matches_the_reference() {
    let f = fixture();
    let cfg = fixture_config(3);
    let w = weights(model_map(&f));
    let block =
        TransformerBlock::from_weights(&w, "decoder.transformer_blocks.0", &cfg, DType::F32)
            .expect("load block 0");

    let rope = Rope3d::new(cfg.rope_apply_dim(), cfg.rope_theta).expect("rope");
    let ids = f.tensor("in.block.ids");
    let tables = rope.tables(&ids).expect("rope tables");

    // The rotary tables themselves, before any block math.
    assert_parity(
        &tables.cos,
        &f.tensor("out.block.rope_cos"),
        TOL,
        "rope cos",
    );
    assert_parity(
        &tables.sin,
        &f.tensor("out.block.rope_sin"),
        TOL,
        "rope sin",
    );

    let got = block
        .forward(&f.tensor("in.block.hidden"), &rope, &tables)
        .expect("block forward");
    assert_parity(
        &got,
        &f.tensor("out.block.hidden"),
        TOL,
        "transformer block",
    );
}

/// The whole ViT decoder: token packing, `proj_in`, register tokens + zero CLS token, zeroed
/// suffix position ids, 2 blocks, `norm_out`, `proj_out`, suffix truncation and patch unpacking.
#[test]
fn vit_decoder_matches_the_reference() {
    let f = fixture();
    let w = weights(model_map(&f));
    let decoder = ViT3dDecoder::from_weights(&w, "decoder", &fixture_config(3), DType::F32)
        .expect("load decoder");
    let got = decoder
        .forward(&f.tensor("in.vit.latent"))
        .expect("vit forward");
    // 5 latent frames × 4, 3×2 spatial × 2 -> [1, 3, 20, 6, 8].
    assert_eq!(got.dims(), &[1, 3, 20, 6, 8]);
    assert_parity(&got, &f.tensor("out.vit.video"), TOL, "ViT3DDecoder");
}

/// `decode` = `post_quant_conv` (1×1×1 Conv3d, applied here as a pointwise linear) then the ViT.
#[test]
fn decode_clip_matches_the_reference() {
    let f = fixture();
    let got = vae(&f, 3)
        .decode_clip(&f.tensor("in.decode.latent"))
        .expect("decode_clip");
    assert_parity(
        &got,
        &f.tensor("out.decode.video"),
        TOL,
        "post_quant_conv + decoder",
    );
}

// ---------------------------------------------------------------------------------------------
// Temporal chunking — the highest-risk part of the port
// ---------------------------------------------------------------------------------------------

/// Token counts straddling the `clip_length`-17 chunk boundary with the production
/// `token_drop = 3`: 5 and 9 need repeat padding, 7 is a single chunk, 12 and 17 exercise the
/// cross-faded seam across 2 and 3 chunks. Each also exercises the 24-entry per-channel
/// de-normalization, since `decode` applies it before chunking.
#[test]
fn temporal_decode_matches_the_reference() {
    let f = fixture();
    let vae = vae(&f, 3);
    for (tokens, frames) in [(5, 17), (7, 22), (9, 30), (12, 39), (17, 56)] {
        let latent = f.tensor(&format!("in.temporal{tokens}.latent"));
        let want = f.tensor(&format!("out.temporal{tokens}.video"));
        let got = vae.decode(&latent).expect("decode");
        assert_eq!(
            got.dims(),
            &[1, 3, frames, 6, 8],
            "{tokens} tokens should decode to {frames} frames"
        );
        assert_parity(&got, &want, TOL, &format!("decode_temporal({tokens})"));
    }
}

/// `token_drop = 0` — the two-pass alignment path: no overlap, one split per chunk, chunks abut
/// with no cross-fade.
#[test]
fn token_drop_zero_two_pass_matches_the_reference() {
    let f = fixture();
    let vae = vae(&f, 0);
    assert_eq!(vae.geometry().token_overlap, 0);
    assert_eq!(vae.geometry().frame_overlap, 0);
    assert_eq!(vae.geometry().split_count(), 1);
    for (tokens, frames) in [(5, 17), (10, 34)] {
        let latent = f.tensor(&format!("in.drop0_temporal{tokens}.latent"));
        let want = f.tensor(&format!("out.drop0_temporal{tokens}.video"));
        let got = vae.decode(&latent).expect("decode");
        assert_eq!(got.dims(), &[1, 3, frames, 6, 8]);
        assert_parity(
            &got,
            &want,
            TOL,
            &format!("drop0 decode_temporal({tokens})"),
        );
    }
}

/// The two drop settings must actually produce DIFFERENT decodes from the same latent — proof
/// that `token_drop` is wired through rather than being an inert config field.
#[test]
fn token_drop_changes_the_decode() {
    let f = fixture();
    let latent = f.tensor("in.temporal5.latent");
    let a = vae(&f, 3).decode(&latent).expect("drop3");
    let b = vae(&f, 0).decode(&latent).expect("drop0");
    assert_eq!(a.dims(), b.dims(), "both yield 17 frames here");
    let (peak, _) = rel(&a, &b);
    println!("  token_drop 3 vs 0: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "token_drop=3 and token_drop=0 decoded identically (peak rel {peak:.3e}); the chunk \
         plan is not reaching the tensor path"
    );
}

// ---------------------------------------------------------------------------------------------
// Latent de-normalization
// ---------------------------------------------------------------------------------------------

/// The per-channel statistics must be APPLIED, not assumed unit. Decoding a latent directly
/// (skipping de-normalization) must differ from decoding it properly.
#[test]
fn latent_denormalization_is_applied_per_channel() {
    let f = fixture();
    let vae = vae(&f, 3);
    let latent = f.tensor("in.temporal7.latent");

    let denorm = vae.denormalize(&latent).expect("denormalize");
    let (peak, _) = rel(&denorm, &latent);
    assert!(
        peak > MUTATION_FLOOR,
        "de-normalization was a no-op (peak rel {peak:.3e})"
    );

    // Channel 0: mean 0.858090, std 1.222377. Check the closed form on one element.
    let l = flat(&latent);
    let d = flat(&denorm);
    let cfg = MiniMaxH3VaeConfig::default();
    let expect = l[0] * cfg.latents_std[0] + cfg.latents_mean[0];
    assert!(
        (d[0] - expect).abs() < 1e-5,
        "expected z·std + mean = {expect}, got {}",
        d[0]
    );

    // Skipping the de-normalization changes the decode.
    let proper = vae.decode(&latent).expect("decode");
    let skipped = vae.decode_temporal(&latent).expect("decode_temporal");
    let (peak, _) = rel(&proper, &skipped);
    assert!(
        peak > MUTATION_FLOOR,
        "decode ignored latents_mean/std (peak rel {peak:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Weight mapping — layout Rule 2
// ---------------------------------------------------------------------------------------------

/// The published checkpoint ships the reference's fused `to_qkv` already split into
/// `to_q`/`to_k`/`to_v`. The fixture carries BOTH forms straight from the reference module, so
/// this asserts the head-interleaved split rule against real reference data rather than against a
/// comment. A contiguous `chunk(3, dim=0)` fails here.
#[test]
fn fused_qkv_split_reproduces_the_published_split() {
    let f = fixture();
    let cfg = fixture_config(3);
    for block in 0..cfg.num_layers {
        for suffix in ["weight", "bias"] {
            let fused = f.tensor(&format!(
                "src.decoder.transformer_blocks.{block}.attn.to_qkv.{suffix}"
            ));
            let [q, k, v] =
                split_fused_qkv(&fused, cfg.num_heads, cfg.head_dim).expect("split fused qkv");
            for (part, name) in [(&q, "to_q"), (&k, "to_k"), (&v, "to_v")] {
                let want = f.tensor(&format!(
                    "decoder.transformer_blocks.{block}.attn.{name}.{suffix}"
                ));
                assert_eq!(
                    flat(part),
                    flat(&want),
                    "block {block} {name}.{suffix}: interleaved split mismatch"
                );
            }
        }
    }
}

/// Crossing the two QKV transforms is **shape-identical and wrong**: a contiguous thirds split of
/// the same fused tensor produces a different `to_q` with the same dims.
///
/// This is what turns "we picked a transform" into "we picked a distinguishable one" — the VAE form
/// is the per-head interleaved gather, and the DiT's `[q_all; k_all; v_all]` in-memory form is not
/// what this component holds (layout Rule 2).
#[test]
fn a_contiguous_thirds_split_is_shape_identical_and_different() {
    let f = fixture();
    let cfg = fixture_config(3);
    let fused = f.tensor("src.decoder.transformer_blocks.0.attn.to_qkv.weight");
    let [q, _, _] = split_fused_qkv(&fused, cfg.num_heads, cfg.head_dim).expect("interleaved");
    let rows = fused.dims()[0] / 3;
    let contiguous = fused.narrow(0, 0, rows).expect("thirds").contiguous().ok();
    let contiguous = contiguous.expect("contiguous thirds");
    assert_eq!(
        q.dims(),
        contiguous.dims(),
        "the two transforms are shape-identical — that is the hazard"
    );
    let (peak, _) = rel(&contiguous, &q);
    println!("  contiguous-thirds vs per-head-interleaved to_q: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the contiguous split reproduced the interleaved one ({peak:.3e}); this fixture cannot \
         distinguish the two QKV transforms"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-18740 — the gated-FFN layout contract (Rule 1)
// ---------------------------------------------------------------------------------------------

/// **The assertion that kills the false green.** The committed fixture must be in the CONVERTED
/// (`[value | gate]`) layout, not the source (`[gate | value]`) one.
///
/// The fixture carries both forms: `decoder.…ff.net.0.proj.*` as the official
/// `AutoencoderKLMiniMaxH3` holds it, and `src.decoder.…ff.w1.*` as the MiniMax reference holds it.
/// This asserts the published tensor is the source tensor with its halves *swapped* — and,
/// critically, that it is NOT the source tensor verbatim.
///
/// That second half is the whole point. Regenerating this fixture with a pure `"ff.w1" ->
/// "ff.net.0.proj"` rename — which is exactly how sc-18740 shipped — makes the two forms equal and
/// fails here, instead of quietly passing every other test in this file.
#[test]
fn published_ffn_projection_is_value_then_gate() {
    let f = fixture();
    let cfg = fixture_config(3);
    assert_eq!(
        PUBLISHED_GATED_FFN_LAYOUT,
        GatedFfnLayout::ValueFirst,
        "the crate's layout contract"
    );

    for block in 0..cfg.num_layers {
        for suffix in ["weight", "bias"] {
            let published = f.tensor(&format!(
                "decoder.transformer_blocks.{block}.ff.net.0.proj.{suffix}"
            ));
            let source = f.tensor(&format!(
                "src.decoder.transformer_blocks.{block}.ff.w1.{suffix}"
            ));
            assert_eq!(
                published.dims(),
                source.dims(),
                "shapes are identical — that is the hazard"
            );

            assert_eq!(
                flat(&published),
                flat(&swap_gated_halves(&source).expect("swap")),
                "block {block} ff.net.0.proj.{suffix}: the published tensor must be the source \
                 tensor with its halves SWAPPED"
            );
            assert_ne!(
                flat(&published),
                flat(&source),
                "block {block} ff.net.0.proj.{suffix} is byte-equal to the pre-conversion \
                 `ff.w1` — this fixture was dumped through a pure rename with no half-swap, so it \
                 tests the port against a layout production never loads (sc-18740). Re-run \
                 tools/dump_minimax_h3_video_vae.py against diffusers `main`."
            );
        }
    }
}

/// **Mutation gate for the half-swap specifically.** Swapping the halves of every published
/// `ff.net.0.proj` — i.e. handing the loader the source layout, which is what production was
/// effectively doing before sc-18740 — must break parity.
///
/// Gated on the **relative max-abs-diff**, and it prints the L2 norms and the cosine alongside so
/// the reason the old gates were blind stays visible in the test output rather than only in a story
/// comment: the norms barely move, and the cosine stays well away from zero because `silu(a)·b` and
/// `silu(b)·a` share sign structure. A `norm`, `std` or checksum assertion of any tolerance would
/// pass here.
#[test]
fn reading_the_ffn_halves_gate_first_breaks_parity() {
    let f = fixture();
    let cfg = fixture_config(3);
    let latent = f.tensor("in.temporal7.latent");
    let want = f.tensor("out.temporal7.video");

    let baseline = vae(&f, 3).decode(&latent).expect("baseline decode");
    let (base_peak, _) = rel(&baseline, &want);

    let mut map = model_map(&f);
    for block in 0..cfg.num_layers {
        for suffix in ["weight", "bias"] {
            let key = format!("decoder.transformer_blocks.{block}.ff.net.0.proj.{suffix}");
            let swapped = swap_gated_halves(&map[&key]).expect("swap");
            map.insert(key, swapped);
        }
    }
    let mutated = vae_from(map, 3).decode(&latent).expect("mutated decode");

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
        "swapping the gate and value halves moved the decode by only {peak:.3e}; this suite \
         cannot gate the sc-18740 defect"
    );
    // ...and the mutated decode must fail the actual parity gate, not merely differ.
    let (vs_ref, _) = rel(&mutated, &want);
    assert!(
        vs_ref > TOL,
        "the source-layout decode still matches the reference within {TOL:.0e} ({vs_ref:.3e}) — \
         the fixture is not in the converted layout"
    );
}

/// The fixture must record which reference path produced it. A regeneration that reverts to
/// running the MiniMax reference modules writes different provenance (or none) and fails here.
///
/// candle's own safetensors loader drops `__metadata__` entirely, which is why `common::Golden`
/// parses the container header by hand — without that, this gate could not exist in this lane.
#[test]
fn fixture_provenance_records_the_converted_path() {
    let f = fixture();
    let meta = |k: &str| {
        f.meta(k)
            .unwrap_or_else(|| {
                panic!(
                    "fixture metadata is missing `{k}`; re-run tools/dump_minimax_h3_video_vae.py \
                     against diffusers `main` (a fixture with no provenance is one that cannot be \
                     shown to come from the converted-checkpoint path — sc-18740)"
                )
            })
            .to_string()
    };
    assert_eq!(meta("provenance"), "converted-checkpoint");
    assert_eq!(meta("reference"), "diffusers.AutoencoderKLMiniMaxH3");
    assert_eq!(meta("gated_ffn_layout"), "value_first");
    // The generator's own conversion cross-check and half-swap negative control, carried forward
    // so their results are auditable from the committed artifact.
    let cross: f32 = meta("conversion_cross_check_rel")
        .parse()
        .expect("a number");
    assert!(
        cross < 1e-5,
        "the generator's inverse-conversion cross-check was {cross:.3e}"
    );
    let swap: f32 = meta("ffn_half_swap_rel").parse().expect("a number");
    assert!(
        swap > MUTATION_FLOOR,
        "the generator measured the half-swap as only {swap:.3e} — it would not be gateable"
    );
    println!(
        "fixture provenance: {} {} (cross-check {cross:.3e}, half-swap {swap:.3e})",
        meta("reference"),
        meta("reference_version"),
    );
}

/// Every model tensor must be consumed, and the declared name list must be exactly the fixture's
/// tensor set. A silently unmapped tensor is the failure mode this crate's key mapping most
/// plausibly has, and it would still produce plausible-looking output.
///
/// Set equality is the whole proof in both directions: `from_weights` `require`s every declared
/// name (so a declared-but-absent key is a load error), and a fixture key outside the declared set
/// would be a tensor nothing reads.
#[test]
fn weight_mapping_is_exhaustive() {
    let f = fixture();
    let cfg = fixture_config(3);
    let present: BTreeSet<String> = model_map(&f).keys().cloned().collect();
    let declared: BTreeSet<String> = MiniMaxH3VideoVae::tensor_names(&cfg).into_iter().collect();
    assert_eq!(
        declared, present,
        "declared tensor names differ from the checkpoint's"
    );
    // ...and the load really does require all of them.
    let _vae = vae(&f, 3);
    let mut short = model_map(&f);
    short.remove("decoder.transformer_blocks.1.ff.net.2.bias");
    assert!(
        MiniMaxH3VideoVae::from_weights(&weights(short), &cfg, &Device::Cpu, DType::F32).is_err(),
        "dropping a declared tensor must be a load error, not a silent default"
    );
}

// ---------------------------------------------------------------------------------------------
// False-green guards
// ---------------------------------------------------------------------------------------------

/// The reference initializes `scale1`/`scale2` to ZERO, which makes every block an exact identity
/// — a golden dumped that way would pass against a port with no attention and no FFN at all. The
/// fixture re-randomizes them; assert here that they are non-zero and that the expected outputs
/// are not near-constant.
#[test]
fn fixture_is_not_degenerate() {
    let f = fixture();
    for key in [
        "decoder.transformer_blocks.0.scale1",
        "decoder.transformer_blocks.0.scale2",
        "decoder.transformer_blocks.1.scale1",
        "decoder.transformer_blocks.1.scale2",
    ] {
        let max = flat(&f.tensor(key))
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max > 1e-3,
            "{key} is ~zero ({max:.3e}); the residual branches would be inert and the golden \
             would pass against a stub"
        );
    }
    // The register tokens must not be zeros either, or their concatenation would be untestable.
    assert!(
        std_dev(&f.tensor("decoder.register_tokens")) > 1e-3,
        "register_tokens are ~constant"
    );

    for key in ["out.vit.video", "out.temporal7.video", "out.block.hidden"] {
        assert!(
            std_dev(&f.tensor(key)) > 1e-3,
            "{key} is ~constant; a constant golden is a false green"
        );
    }
}

/// Mutation check: perturbing any single weight must move the decode. If a tensor can be changed
/// without changing the output, it is not actually wired into the graph and the parity test is
/// not covering it.
#[test]
fn every_weight_is_load_bearing() {
    let f = fixture();
    let latent = f.tensor("in.temporal7.latent");
    let baseline = vae(&f, 3).decode(&latent).expect("baseline");

    // One representative of every distinct tensor role in the decode path, each with the floor its
    // MEASURED sensitivity supports. Sensitivity is not uniform and pretending it is would either
    // fail honestly-wired tensors or drop the floor into the noise for everything else.
    //
    // `decoder.register_tokens` is the one genuine outlier. That is intrinsic, not a wiring defect:
    // the registers are 4 of 65 attention keys and their own output rows are discarded before
    // `proj_out`, so they reach the result only through the attention mixture — and because softmax
    // saturates, a LARGER perturbation moves the output *less*, not more. It is covered
    // independently by `register_tokens_participate_and_their_shape_is_checked`, which zeroes the
    // tensor outright.
    let probes: [(&str, f32); 23] = [
        ("post_quant_conv.weight", MUTATION_FLOOR),
        ("post_quant_conv.bias", MUTATION_FLOOR),
        ("decoder.proj_in.weight", MUTATION_FLOOR),
        ("decoder.proj_in.bias", MUTATION_FLOOR),
        ("decoder.register_tokens", 3e-3),
        ("decoder.transformer_blocks.0.norm1.weight", MUTATION_FLOOR),
        (
            "decoder.transformer_blocks.0.attn.to_q.weight",
            MUTATION_FLOOR,
        ),
        (
            "decoder.transformer_blocks.0.attn.to_k.weight",
            MUTATION_FLOOR,
        ),
        (
            "decoder.transformer_blocks.0.attn.to_v.weight",
            MUTATION_FLOOR,
        ),
        (
            "decoder.transformer_blocks.0.attn.to_v.bias",
            MUTATION_FLOOR,
        ),
        (
            "decoder.transformer_blocks.0.attn.to_out.0.weight",
            MUTATION_FLOOR,
        ),
        (
            "decoder.transformer_blocks.0.attn.to_out.0.bias",
            MUTATION_FLOOR,
        ),
        ("decoder.transformer_blocks.0.scale1", MUTATION_FLOOR),
        ("decoder.transformer_blocks.0.norm2.weight", MUTATION_FLOOR),
        (
            "decoder.transformer_blocks.0.ff.net.0.proj.weight",
            MUTATION_FLOOR,
        ),
        (
            "decoder.transformer_blocks.0.ff.net.2.weight",
            MUTATION_FLOOR,
        ),
        ("decoder.transformer_blocks.0.scale2", MUTATION_FLOOR),
        // The LAST block's query projection is the second genuine outlier: a perturbation there
        // passes through one attention mixture and one `scale2` residual before `proj_out`, where
        // block 0's also propagates through block 1. Block 0's `to_q` covers the projection's
        // wiring at the full floor.
        ("decoder.transformer_blocks.1.attn.to_q.weight", 5e-3),
        ("decoder.transformer_blocks.1.scale1", MUTATION_FLOOR),
        ("decoder.norm_out.weight", MUTATION_FLOOR),
        ("decoder.norm_out.bias", MUTATION_FLOOR),
        ("decoder.proj_out.weight", MUTATION_FLOOR),
        ("decoder.proj_out.bias", MUTATION_FLOOR),
    ];

    for (key, floor) in probes {
        let mut map = model_map(&f);
        let original = map[key].clone();
        // Scale AND shift: the shift makes an all-zero tensor observable, the scale keeps the
        // perturbation proportionate for tensors that are already large.
        let bumped = ((original * 1.5).expect("scale") + 0.5).expect("shift");
        map.insert(key.to_string(), bumped);
        let mutated = vae_from(map, 3).decode(&latent).expect("mutated decode");
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e} (floor {floor:.1e})");
        assert!(
            peak > floor,
            "perturbing {key} moved the decode by only {peak:.3e}, under its {floor:.1e} floor — \
             it is either not wired into the graph or no longer observable here"
        );
    }
}

/// The suffix tokens must sit at the rotary ORIGIN (zero position ids). If they inherited the
/// patch grid's ids instead, the decode changes — pin that the implementation takes the zero-id
/// path by checking the register tokens influence the output at all, and that a decoder built
/// with a different register-token count is rejected.
#[test]
fn register_tokens_participate_and_their_shape_is_checked() {
    let f = fixture();
    let cfg = fixture_config(3);
    let latent = f.tensor("in.vit.latent");

    let baseline = ViT3dDecoder::from_weights(&weights(model_map(&f)), "decoder", &cfg, DType::F32)
        .expect("decoder")
        .forward(&latent)
        .expect("forward");

    let mut map = model_map(&f);
    map.insert(
        "decoder.register_tokens".to_string(),
        Tensor::zeros(
            (1, cfg.num_register_tokens, cfg.dim()),
            DType::F32,
            &Device::Cpu,
        )
        .expect("zeros"),
    );
    let without = ViT3dDecoder::from_weights(&weights(map), "decoder", &cfg, DType::F32)
        .expect("decoder")
        .forward(&latent)
        .expect("forward");
    let (peak, _) = rel(&without, &baseline);
    println!("  zeroing register_tokens: peak rel {peak:.3e}");
    assert!(peak > 1e-3, "register tokens do not affect the decode");

    // A config that disagrees with the checkpoint's register-token count must fail loudly.
    let mut bad = cfg.clone();
    bad.num_register_tokens = 2;
    assert!(
        ViT3dDecoder::from_weights(&weights(model_map(&f)), "decoder", &bad, DType::F32).is_err()
    );
}

/// **How small a numeric divergence this fixture gate can actually resolve**, measured rather than
/// argued.
///
/// The failure mode this is written for is an RMSNorm epsilon applied OUTSIDE the square root
/// (`x / (sqrt(ms) + eps)`) instead of inside it (`x / sqrt(ms + eps)`), which is a sub-percent
/// divergence that no cosine, norm or checksum assertion can see — and which the cross-backend
/// comparison in `cross_backend.rs` explicitly cannot detect either, because its floor is four
/// orders larger.
///
/// It is not directly injectable through the public API, so it is emulated by the displacement it
/// produces. On unit-RMS activations, `sqrt(ms + eps) ≈ sqrt(ms) + eps/(2·sqrt(ms))`, so applying
/// `eps` outside instead of inside is equivalent to using an epsilon about `2·eps` too large.
/// `norm_eps = 3e-5` therefore displaces the norms by the same ~2e-5 an eps misplacement would at
/// the shipped `1e-5`.
///
/// **The measured answer is that this fixture gate does NOT resolve it.** The displacement moves
/// the decode by ~5.9e-6, which is 16× above the parity residual — so the epsilon demonstrably
/// reaches the norms and is not inert — but still under the 1e-5 bound. Tightening the bound to
/// cover it would put it ~8× above f32 round-off, which is not a margin a golden shared with a
/// different BLAS on a different CPU can be relied on to hold.
///
/// That class is therefore gated **structurally** instead, by
/// `candle_gen_minimax_h3::nn`'s `rms_norm_puts_epsilon_inside_the_square_root`, which pins the
/// formulation against a hand-computed closed form and rejects the outside-the-root variant
/// outright. This test exists to keep the numeric half of that statement a measurement: it asserts
/// only that the displacement is visible above the noise, and prints where it lands relative to the
/// gate.
#[test]
fn an_eps_displacement_the_size_of_a_misplaced_rms_epsilon_is_visible_but_under_the_gate() {
    let f = fixture();
    let want = f.tensor("out.vit.video");

    let baseline = ViT3dDecoder::from_weights(
        &weights(model_map(&f)),
        "decoder",
        &fixture_config(3),
        DType::F32,
    )
    .expect("decoder")
    .forward(&f.tensor("in.vit.latent"))
    .expect("forward");
    let (residual, _) = rel(&baseline, &want);

    let mut cfg = fixture_config(3);
    cfg.norm_eps = 3e-5;
    let got = ViT3dDecoder::from_weights(&weights(model_map(&f)), "decoder", &cfg, DType::F32)
        .expect("decoder")
        .forward(&f.tensor("in.vit.latent"))
        .expect("forward");
    let (peak, mean) = rel(&got, &want);
    println!(
        "  norm_eps 3e-5 instead of 1e-5 (an eps-outside-sqrt-sized displacement): peak rel \
         {peak:.3e} (mean {mean:.3e}) vs parity residual {residual:.3e} and gate {TOL:.0e} — \
         {:.0}x the residual, {:.2}x the gate",
        peak / residual,
        peak / TOL
    );
    assert!(
        peak > residual * 5.0,
        "the epsilon displacement ({peak:.3e}) is not distinguishable from the parity residual \
         ({residual:.3e}); `norm_eps` is not reaching the norms at all"
    );
}

/// A config claiming a different rotary fraction must NOT reproduce the golden — the strongest
/// available evidence that `rope_dim_ratio = 0.75` is genuinely load-bearing rather than a constant
/// nobody reads.
#[test]
fn a_different_rotary_fraction_breaks_parity() {
    let f = fixture();
    let mut cfg = fixture_config(3);
    // 16 · 0.375 = 6 rotated dims instead of 12 — legal for the 3-axis split, but a different
    // model. (A ratio of 1.0 would rotate all 16, which `validate` rejects: 16 is not a
    // multiple of 6, so the shipped 0.75 is not freely adjustable.)
    cfg.rope_dim_ratio = 0.375;
    assert_eq!(cfg.rope_apply_dim(), 6);

    let got = ViT3dDecoder::from_weights(&weights(model_map(&f)), "decoder", &cfg, DType::F32)
        .expect("decoder")
        .forward(&f.tensor("in.vit.latent"))
        .expect("forward");
    let (peak, _) = rel(&got, &f.tensor("out.vit.video"));
    println!("  rope_dim_ratio 0.375 instead of 0.75: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "a different rotary fraction reproduced the golden (peak rel {peak:.3e}); the ratio is \
         not reaching the rotary"
    );
}

// ---------------------------------------------------------------------------------------------
// Spatial tiling (sc-18786)
// ---------------------------------------------------------------------------------------------
//
// `AutoencoderKLMiniMaxH3` ships with `use_tiling = True` and upstream is explicit that **the
// released frames are the blended-tile ones, so disabling tiling changes the output**. sc-17154
// decoded the whole canvas in one pass, which is the reference only below one 256 px tile.
//
// These are the *committed* gates — they run on every CI pass, without a snapshot. They cannot
// prove the tiled numbers are right (only real weights against diffusers can, see
// `real_weights.rs::real_weight_tiled_decode_matches_the_official_diffusers_vae`); what they prove
// is that tiling is inert where the committed fixtures live, and that the new path is nevertheless
// reachable and distinguishable rather than untested by construction.

/// The fixture canvas is 6x8 px (a 3x4 latent at the fixture's 2x ratio) — far below one 256 px
/// tile. So `decode_clip` with the shipped defaults must be **bit**-identical to the untiled path,
/// which is what keeps every reference-backed decode assertion in this file valid across the
/// tiling change.
#[test]
fn shipped_tiling_is_on_by_default_and_bit_inert_below_one_tile() {
    let f = fixture();
    let vae = vae(&f, 3);
    let t = vae.tiling();
    assert!(t.enabled, "the shipped VAE tiles by default");
    assert_eq!((t.tile_height, t.tile_width), (256, 256));
    assert_eq!((t.overlap_height, t.overlap_width), (64, 64));

    let latent = f.tensor("in.decode.latent");
    assert_eq!(latent.dims()[3..], [3, 4], "fixture latent is 3x4");

    let tiled = vae.decode_clip(&latent).unwrap();
    let untiled = vae.decode_clip_untiled(&latent).unwrap();
    assert_eq!(tiled.dims(), untiled.dims());
    assert_eq!(
        flat(&tiled),
        flat(&untiled),
        "below one tile the tiled path must be BIT-identical"
    );
}

/// The plan the fixture canvas takes at a shrunk tile, against the reference's own `_split_tiles`.
///
/// 6 px of height at a 4 px tile is 2 rows; 8 px of width is **3** columns, not 2 — two 4 px tiles
/// at a 2 px overlap span only 6 px. The grid is deliberately non-square so a transposed plan is
/// not accidentally correct.
#[test]
fn the_fixture_canvas_tiles_into_the_reference_grid() {
    use candle_gen_minimax_h3::spatial_tiling::TilePlan;
    let rows = TilePlan::split(6, 4, 2, 2).unwrap();
    assert_eq!(rows.starts, vec![0, 2]);
    assert_eq!(rows.lengths, vec![4, 4]);
    assert_eq!(rows.overlaps, vec![2]);

    let cols = TilePlan::split(8, 4, 2, 2).unwrap();
    assert_eq!(
        cols.starts,
        vec![0, 2, 4],
        "8 px at a 4/2 tile is three columns"
    );
    assert_eq!(cols.lengths, vec![4, 4, 4]);
    assert_eq!(cols.overlaps, vec![2, 2]);
    assert_eq!(cols.coverage(), 8);
}

/// **The tiled path is reachable and it changes the answer.** Without this, the tiling code could
/// be a no-op — or never entered at all — and every other assertion in this file would still pass.
/// That is exactly how sc-18740 shipped a broken FFN layout.
///
/// The decoder is a ViT whose attention is global over the whole clip, so a tile sees a strictly
/// smaller context than the full canvas and cannot agree with it.
#[test]
fn a_tile_smaller_than_the_canvas_changes_the_decode() {
    let f = fixture();
    let vae = vae(&f, 3);
    let latent = f.tensor("in.decode.latent");

    let untiled = vae.decode_clip_untiled(&latent).unwrap();
    let tiled = vae.decode_clip_tiled(&latent, 4, 4, 2, 2).unwrap();
    assert_eq!(
        tiled.dims(),
        untiled.dims(),
        "tiling must not change the shape"
    );

    let (peak, mean) = rel(&tiled, &untiled);
    println!("SPATIAL TILING 4/2 vs untiled: rel-max-abs={peak:.3e} rel-mean-abs={mean:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "a 2x3 tile grid moved the decode by only {peak:.3e}; the tiled path is not being taken"
    );
}

/// The `enable_tiling` / `disable_tiling` knobs select the paths they name.
#[test]
fn the_tiling_knobs_select_the_paths_they_name() {
    use candle_gen_minimax_h3::spatial_tiling::SpatialTiling;
    let f = fixture();
    let latent = f.tensor("in.decode.latent");
    let base = vae(&f, 3);
    let untiled = flat(&base.decode_clip_untiled(&latent).unwrap());
    let tiled = flat(&base.decode_clip_tiled(&latent, 4, 4, 2, 2).unwrap());
    assert_ne!(tiled, untiled, "the two reference paths must differ");

    let mut on = vae(&f, 3);
    on.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    assert_eq!(flat(&on.decode_clip(&latent).unwrap()), tiled);

    let mut off = vae(&f, 3);
    off.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    off.disable_tiling();
    assert!(!off.tiling().enabled);
    assert_eq!(flat(&off.decode_clip(&latent).unwrap()), untiled);

    let built = vae(&f, 3).with_tiling(SpatialTiling::disabled());
    assert_eq!(flat(&built.decode_clip(&latent).unwrap()), untiled);
    let built_on = vae(&f, 3).with_tiling(SpatialTiling::square(4, 2));
    assert_eq!(flat(&built_on.decode_clip(&latent).unwrap()), tiled);
}

/// **The latent-alignment guard is reachable from the public API**, not merely declared on
/// `TilePlan::split`.
///
/// `enable_tiling` takes pixel tile sizes, and `decode_clip_tiled` indexes the latent with
/// `start / ratio` and `length / ratio`. At the fixture's ratio of 2 an odd tile or overlap would
/// truncate every tile and stitch it with overlaps derived from the un-truncated size — a
/// wrong-sized video and no error at all. Both clauses are probed separately so removing either
/// one alone fails here, and the aligned geometry is decoded to show the guard is not simply
/// rejecting everything.
#[test]
fn a_tile_geometry_the_latent_grid_cannot_express_is_refused() {
    let f = fixture();
    let latent = f.tensor("in.decode.latent");
    assert_eq!(
        fixture_config(3).patch_size,
        2,
        "this test's odd tile sizes are only misaligned at ratio 2"
    );

    // A misaligned TILE, at an aligned overlap.
    let err = vae(&f, 3)
        .decode_clip_tiled(&latent, 3, 4, 2, 2)
        .expect_err("a 3 px tile is 1.5 latent cells and must be refused");
    let msg = err.to_string();
    assert!(msg.contains("2x spatial compression ratio"), "{msg}");
    assert!(
        msg.contains("silently truncated"),
        "the error must say WHY: {msg}"
    );

    // A misaligned OVERLAP, at an aligned tile.
    assert!(
        vae(&f, 3).decode_clip_tiled(&latent, 4, 4, 1, 2).is_err(),
        "a 1 px overlap is half a latent cell and must be refused"
    );

    // …and it reaches through `enable_tiling` / `decode_clip`, which is the surface a caller uses.
    let mut misaligned = vae(&f, 3);
    misaligned.enable_tiling(Some(3), Some(3), Some(2), Some(2));
    assert!(misaligned.decode_clip(&latent).is_err());

    // The aligned geometry still decodes, so the guard rejects the misalignment and nothing else.
    assert!(vae(&f, 3).decode_clip_tiled(&latent, 4, 4, 2, 2).is_ok());
}

/// Tiling composes with the temporal chunking rather than replacing it: `decode` still produces the
/// planned frame count at a canvas that tiles spatially.
#[test]
fn spatial_tiling_composes_with_the_temporal_chunk_plan() {
    let f = fixture();
    let mut vae = vae(&f, 3);
    vae.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    let latent = f.tensor("in.temporal12.latent");
    let want = f.tensor("out.temporal12.video");
    let got = vae.decode(&latent).unwrap();
    assert_eq!(
        got.dims(),
        want.dims(),
        "the spatial grid must not disturb the temporal frame plan"
    );
    let (peak, _) = rel(&got, &want);
    println!("TILED multi-chunk decode vs untiled golden: rel-max-abs={peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the spatial grid was not applied under chunking"
    );
}
