//! sc-17140: video-VAE decode parity — against the **official diffusers**
//! `AutoencoderKLMiniMaxH3`, i.e. the converted checkpoint production actually loads.
//!
//! Fixture `tests/fixtures/video_vae_decode.safetensors` ← `tools/dump_minimax_h3_video_vae.py`.
//!
//! # Why the reference class changed (sc-18740)
//!
//! This fixture was originally dumped from the MiniMax reference modules shipped inside the
//! snapshot (`FL2VA/video_vae/*.py`) with a **pure rename** onto the published key names. That made
//! the whole file a false green: the official conversion swaps the two halves of every gated FFN
//! projection, so the fixture carried the *source* layout under *published* names, the loader read
//! it the source way, they agreed, and the shipped 36-layer decoder was wrong on real weights by
//! 0.86-0.99 relative max-abs-diff per block. See [`mlx_gen_minimax_h3::layout`].
//!
//! The generator now runs `AutoencoderKLMiniMaxH3` and additionally proves the conversion by
//! loading the inverse-converted weights back into the reference and asserting both decode
//! identically. `fixture_provenance_records_the_converted_path` and
//! `published_ffn_projection_is_value_then_gate` below make a silent revert to the old method fail.
//!
//! Tolerance 1e-2 peak-relative, the mlx-gen house value. Everything here is f32 and MLX runs f32
//! matmul in reduced precision on Metal, so the observed residual is ~1e-3 — a 10x margin, and the
//! floor on what these tests can prove. A structural error — wrong QKV split, full instead of
//! partial rotary, LayerNorm instead of RMSNorm, missing register tokens, a mis-planned chunk seam
//! — diverges by orders of magnitude, which the mutation tests at the bottom confirm by measuring
//! and PRINTING how far a perturbed weight actually moves the decode.

mod common;

use common::{assert_parity, cosine, fixture_config, l2_norm, rel, std_dev, FIXTURE};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::blocks::TransformerBlock;
use mlx_gen_minimax_h3::{
    split_fused_qkv, GatedFfnLayout, MiniMaxH3VaeConfig, MiniMaxH3VideoVae, Rope3d, ViT3dDecoder,
    PUBLISHED_GATED_FFN_LAYOUT,
};

const TOL: f32 = 1e-2;

/// Mutation checks must clear the numeric noise floor by a wide margin, or "the output moved"
/// would just be reduced-precision jitter. Observed parity residual is ~1e-3 (MLX runs f32 matmul
/// in reduced precision on Metal), so 1e-2 is 10x above it. The test prints every probe so the
/// real margin stays auditable rather than implied.
///
/// Individual probes may declare a lower floor where their measured sensitivity is intrinsically
/// bounded; see the probe table for the one case and why.
const MUTATION_FLOOR: f32 = 1e-2;

/// Fixture tensors minus the reference-side extras (`src.` fused QKV, `in.`/`out.` activations,
/// `const.` statistics) — i.e. exactly the model weights in the published root naming.
fn model_weights() -> Weights {
    let mut w = Weights::from_file(FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "const."] {
        w.remove_prefix(prefix);
    }
    w
}

fn fixture() -> Weights {
    Weights::from_file(FIXTURE).unwrap()
}

fn vae(token_drop: i32) -> MiniMaxH3VideoVae {
    let mut w = model_weights();
    MiniMaxH3VideoVae::from_weights(&mut w, &fixture_config(token_drop), Dtype::Float32).unwrap()
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
    let mut w = model_weights();
    let block = TransformerBlock::from_weights(
        &mut w,
        "decoder.transformer_blocks.0",
        &cfg,
        Dtype::Float32,
    )
    .unwrap();

    let rope = Rope3d::new(cfg.rope_apply_dim(), cfg.rope_theta).unwrap();
    let ids = f.require("in.block.ids").unwrap();
    let tables = rope.tables(ids).unwrap();

    // The rotary tables themselves, before any block math.
    assert_parity(
        &tables.cos,
        f.require("out.block.rope_cos").unwrap(),
        TOL,
        "rope cos",
    );
    assert_parity(
        &tables.sin,
        f.require("out.block.rope_sin").unwrap(),
        TOL,
        "rope sin",
    );

    let got = block
        .forward(f.require("in.block.hidden").unwrap(), &rope, &tables)
        .unwrap();
    assert_parity(
        &got,
        f.require("out.block.hidden").unwrap(),
        TOL,
        "transformer block",
    );
}

/// The whole ViT decoder: token packing, `proj_in`, register tokens + zero CLS token, zeroed
/// suffix position ids, 2 blocks, `norm_out`, `proj_out`, suffix truncation and patch unpacking.
#[test]
fn vit_decoder_matches_the_reference() {
    let f = fixture();
    let mut w = model_weights();
    let decoder =
        ViT3dDecoder::from_weights(&mut w, "decoder", &fixture_config(3), Dtype::Float32).unwrap();
    let got = decoder
        .forward(f.require("in.vit.latent").unwrap())
        .unwrap();
    let want = f.require("out.vit.video").unwrap();
    // 5 latent frames × patch_size_t, a 3×4 latent × patch_size spatially. Derived from the
    // config rather than restated, so a fixture-geometry change shows up as a parity failure
    // rather than as a shape literal nobody updated.
    let cfg = fixture_config(3);
    assert_eq!(
        got.shape(),
        &[
            1,
            3,
            5 * cfg.patch_size_t,
            3 * cfg.patch_size,
            4 * cfg.patch_size
        ]
    );
    assert_parity(&got, want, TOL, "ViT3DDecoder");
}

/// `decode` = `post_quant_conv` (1×1×1 Conv3d, applied here as a pointwise linear) then the ViT.
#[test]
fn decode_clip_matches_the_reference() {
    let f = fixture();
    let got = vae(3)
        .decode_clip(f.require("in.decode.latent").unwrap())
        .unwrap();
    assert_parity(
        &got,
        f.require("out.decode.video").unwrap(),
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
    let cfg = fixture_config(3);
    let vae = vae(3);
    for (tokens, frames) in [(5, 17), (7, 22), (9, 30), (12, 39), (17, 56)] {
        let latent = f.require(&format!("in.temporal{tokens}.latent")).unwrap();
        let want = f.require(&format!("out.temporal{tokens}.video")).unwrap();
        let got = vae.decode(latent).unwrap();
        assert_eq!(
            got.shape(),
            &[1, 3, frames, 3 * cfg.patch_size, 4 * cfg.patch_size],
            "{tokens} tokens should decode to {frames} frames"
        );
        assert_parity(&got, want, TOL, &format!("decode_temporal({tokens})"));
    }
}

/// `token_drop = 0` — the two-pass alignment path: no overlap, one split per chunk, chunks abut
/// with no cross-fade.
#[test]
fn token_drop_zero_two_pass_matches_the_reference() {
    let f = fixture();
    let cfg = fixture_config(0);
    let vae = vae(0);
    assert_eq!(vae.geometry().token_overlap, 0);
    assert_eq!(vae.geometry().frame_overlap, 0);
    assert_eq!(vae.geometry().split_count(), 1);
    for (tokens, frames) in [(5, 17), (10, 34)] {
        let latent = f
            .require(&format!("in.drop0_temporal{tokens}.latent"))
            .unwrap();
        let want = f
            .require(&format!("out.drop0_temporal{tokens}.video"))
            .unwrap();
        let got = vae.decode(latent).unwrap();
        assert_eq!(
            got.shape(),
            &[1, 3, frames, 3 * cfg.patch_size, 4 * cfg.patch_size]
        );
        assert_parity(&got, want, TOL, &format!("drop0 decode_temporal({tokens})"));
    }
}

/// The two drop settings must actually produce DIFFERENT decodes from the same latent — proof
/// that `token_drop` is wired through rather than being an inert config field.
#[test]
fn token_drop_changes_the_decode() {
    let f = fixture();
    let latent = f.require("in.temporal5.latent").unwrap();
    let a = vae(3).decode(latent).unwrap();
    let b = vae(0).decode(latent).unwrap();
    assert_eq!(a.shape(), b.shape(), "both yield 17 frames here");
    let (peak, _) = rel(&a, &b);
    assert!(
        peak > 1e-2,
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
    let vae = vae(3);
    let latent = f.require("in.temporal7.latent").unwrap();

    let denorm = vae.denormalize(latent).unwrap();
    let (peak, _) = rel(&denorm, latent);
    assert!(
        peak > 1e-2,
        "de-normalization was a no-op (peak rel {peak:.3e})"
    );

    // Channel 0: mean 0.858090, std 1.222377. Check the closed form on one element.
    let l: Vec<f32> = latent.as_slice::<f32>().to_vec();
    let d: Vec<f32> = denorm.as_slice::<f32>().to_vec();
    let cfg = MiniMaxH3VaeConfig::default();
    let expect = l[0] * cfg.latents_std[0] + cfg.latents_mean[0];
    assert!(
        (d[0] - expect).abs() < 1e-5,
        "expected z·std + mean = {expect}, got {}",
        d[0]
    );

    // Skipping the de-normalization changes the decode.
    let proper = vae.decode(latent).unwrap();
    let skipped = vae.decode_temporal(latent).unwrap();
    let (peak, _) = rel(&proper, &skipped);
    assert!(
        peak > 1e-2,
        "decode ignored latents_mean/std (peak rel {peak:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Weight mapping
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
            let fused = f
                .require(&format!(
                    "src.decoder.transformer_blocks.{block}.attn.to_qkv.{suffix}"
                ))
                .unwrap();
            let [q, k, v] = split_fused_qkv(fused, cfg.num_heads, cfg.head_dim).unwrap();
            for (part, name) in [(&q, "to_q"), (&k, "to_k"), (&v, "to_v")] {
                let want = f
                    .require(&format!(
                        "decoder.transformer_blocks.{block}.attn.{name}.{suffix}"
                    ))
                    .unwrap();
                assert_eq!(
                    part.as_slice::<f32>(),
                    want.as_slice::<f32>(),
                    "block {block} {name}.{suffix}: interleaved split mismatch"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// sc-18740 — the gated-FFN layout contract
// ---------------------------------------------------------------------------------------------

/// Swap the two row halves of a `[2·inner, dim]` (or `[2·inner]`) fused gated projection.
fn swap_halves(t: &Array) -> Array {
    let rows = t.shape()[0];
    let parts = t.split_axis(&[rows / 2], 0).unwrap();
    mlx_rs::ops::concatenate_axis(&[parts[1].clone(), parts[0].clone()], 0).unwrap()
}

/// **The assertion that kills the false green.** The committed fixture must be in the CONVERTED
/// (`[value | gate]`) layout, not the source (`[gate | value]`) one.
///
/// The fixture carries both forms: `decoder.…ff.net.0.proj.*` as the official
/// `AutoencoderKLMiniMaxH3` holds it, and `src.decoder.…ff.w1.*` as the MiniMax reference holds it
/// (verified in the generator by loading the inverse conversion back into the reference and
/// getting a bit-identical decode). This asserts the published tensor is the source tensor with its
/// halves *swapped* — and, critically, that it is NOT the source tensor verbatim.
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
            let published = f
                .require(&format!(
                    "decoder.transformer_blocks.{block}.ff.net.0.proj.{suffix}"
                ))
                .unwrap();
            let source = f
                .require(&format!(
                    "src.decoder.transformer_blocks.{block}.ff.w1.{suffix}"
                ))
                .unwrap();
            assert_eq!(
                published.shape(),
                source.shape(),
                "shapes are identical — that is the hazard"
            );

            assert_eq!(
                published.as_slice::<f32>(),
                swap_halves(source).as_slice::<f32>(),
                "block {block} ff.net.0.proj.{suffix}: the published tensor must be the source \
                 tensor with its halves SWAPPED"
            );
            assert_ne!(
                published.as_slice::<f32>(),
                source.as_slice::<f32>(),
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
/// effectively doing — must break parity.
///
/// Gated on the **relative max-abs-diff**, and it prints the L2 norms and the cosine alongside so
/// the reason the old gates were blind stays visible in the test output rather than only in a
/// story comment: the norms barely move, and the cosine stays well away from zero because
/// `silu(a)·b` and `silu(b)·a` share sign structure. A `norm`, `std` or checksum assertion of any
/// tolerance would pass here.
#[test]
fn reading_the_ffn_halves_gate_first_breaks_parity() {
    let f = fixture();
    let cfg = fixture_config(3);
    let latent = f.require("in.temporal7.latent").unwrap();
    let want = f.require("out.temporal7.video").unwrap();

    let baseline = vae(3).decode(latent).unwrap();
    let (base_peak, _) = rel(&baseline, want);

    let mut w = model_weights();
    for block in 0..cfg.num_layers {
        for suffix in ["weight", "bias"] {
            let key = format!("decoder.transformer_blocks.{block}.ff.net.0.proj.{suffix}");
            let swapped = swap_halves(w.require(&key).unwrap());
            w.insert(&key, swapped);
        }
    }
    let mutated = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32)
        .unwrap()
        .decode(latent)
        .unwrap();

    let (peak, mean) = rel(&mutated, &baseline);
    let cos = cosine(&mutated, &baseline);
    println!(
        "FFN half-swap: rel-max-abs={peak:.3e} rel-mean={mean:.3e} cosine={cos:.4} \
         ||correct||={:.4} ||swapped||={:.4}  (parity residual with the correct layout: {base_peak:.3e})",
        l2_norm(&baseline),
        l2_norm(&mutated),
    );

    assert!(
        peak > 1e-2,
        "swapping the gate and value halves moved the decode by only {peak:.3e}; this suite \
         cannot gate the sc-18740 defect"
    );
    // ...and the mutated decode must fail the actual parity gate, not merely differ.
    let (vs_ref, _) = rel(&mutated, want);
    assert!(
        vs_ref > TOL,
        "the source-layout decode still matches the reference within {TOL:.0e} ({vs_ref:.3e}) — \
         the fixture is not in the converted layout"
    );
}

/// The fixture must record which reference path produced it. A regeneration that reverts to
/// running the MiniMax reference modules writes different provenance (or none) and fails here.
#[test]
fn fixture_provenance_records_the_converted_path() {
    let f = fixture();
    let meta = |k: &str| {
        f.metadata(k)
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
    let cross: f32 = meta("conversion_cross_check_rel").parse().unwrap();
    assert!(
        cross < 1e-5,
        "the generator's inverse-conversion cross-check was {cross:.3e}"
    );
    let swap: f32 = meta("ffn_half_swap_rel").parse().unwrap();
    assert!(
        swap > 1e-2,
        "the generator measured the half-swap as only {swap:.3e} — it would not be gateable"
    );
    println!(
        "fixture provenance: {} {} (cross-check {cross:.3e}, half-swap {swap:.3e})",
        meta("reference"),
        meta("reference_version"),
    );
}

/// Every model tensor must be consumed. A silently unmapped tensor is the failure mode this
/// crate's key mapping most plausibly has, and it would still produce plausible-looking output.
#[test]
fn weight_mapping_is_exhaustive() {
    let cfg = fixture_config(3);
    let mut w = model_weights();
    let before: std::collections::BTreeSet<String> = w.keys().map(str::to_string).collect();

    let _vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();
    w.remove_accessed();
    let leftover: Vec<&str> = w.keys().collect();
    assert!(
        leftover.is_empty(),
        "these checkpoint tensors were never read: {leftover:?}"
    );

    // ...and conversely, the declared name list is exactly the fixture's tensor set.
    let declared: std::collections::BTreeSet<String> =
        MiniMaxH3VideoVae::tensor_names(&cfg).into_iter().collect();
    assert_eq!(
        declared, before,
        "declared tensor names differ from the checkpoint's"
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
        let t = f.require(key).unwrap();
        let max: f32 = t.abs().unwrap().max(None).unwrap().item();
        assert!(
            max > 1e-3,
            "{key} is ~zero ({max:.3e}); the residual branches would be inert and the golden \
             would pass against a stub"
        );
    }
    // The register tokens must not be zeros either, or their concatenation would be untestable.
    let reg = f.require("decoder.register_tokens").unwrap();
    assert!(std_dev(reg) > 1e-3, "register_tokens are ~constant");

    for key in ["out.vit.video", "out.temporal7.video", "out.block.hidden"] {
        let t = f.require(key).unwrap();
        assert!(
            std_dev(t) > 1e-3,
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
    let cfg = fixture_config(3);
    let latent = f.require("in.temporal7.latent").unwrap();
    let baseline = vae(3).decode(latent).unwrap();

    // One representative of every distinct tensor role in the decode path, each with the floor its
    // MEASURED sensitivity supports. Sensitivity is not uniform and pretending it is would either
    // fail honestly-wired tensors or drop the floor into the noise for everything else.
    //
    // `decoder.register_tokens` is the one genuine outlier at ~6e-3. That is intrinsic, not a
    // wiring defect: the registers are 4 of 65 attention keys and their own output rows are
    // discarded before `proj_out`, so they reach the result only through the attention mixture —
    // and because softmax saturates, a LARGER perturbation moves the output *less*, not more
    // (measured: 1.5x+0.5 -> 6.1e-3, 8x+2 -> 4.5e-3). It still sits ~6x over the ~1e-3 noise
    // floor, and `register_tokens_participate_and_their_shape_is_checked` covers it independently
    // by zeroing the tensor outright.
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
        // The LAST block's query projection is the second genuine outlier, measured at ~8.7e-3 on
        // the sc-18740 fixture draw (block 0's equivalent is 1.5e-2). That is intrinsic to being
        // last: a perturbation there passes through one attention mixture and one `scale2`
        // residual before `proj_out`, where block 0's also propagates through block 1. It still
        // sits ~9x over the ~1e-3 parity residual this suite measures, and block 0's `to_q` covers
        // the projection's wiring at the full floor.
        ("decoder.transformer_blocks.1.attn.to_q.weight", 5e-3),
        ("decoder.transformer_blocks.1.scale1", MUTATION_FLOOR),
        ("decoder.norm_out.weight", MUTATION_FLOOR),
        ("decoder.norm_out.bias", MUTATION_FLOOR),
        ("decoder.proj_out.weight", MUTATION_FLOOR),
        ("decoder.proj_out.bias", MUTATION_FLOOR),
    ];

    for (key, floor) in probes {
        let mut w = model_weights();
        let original = w.require(key).unwrap().clone();
        // Scale AND shift: the shift makes an all-zero tensor observable, the scale keeps the
        // perturbation proportionate for tensors that are already large.
        let bumped = mlx_rs::ops::add(
            mlx_rs::ops::multiply(&original, Array::from_f32(1.5)).unwrap(),
            Array::from_f32(0.5),
        )
        .unwrap();
        w.insert(key, bumped);
        let mutated = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32)
            .unwrap()
            .decode(latent)
            .unwrap();
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
    let latent = f.require("in.vit.latent").unwrap();

    let mut w = model_weights();
    let baseline = ViT3dDecoder::from_weights(&mut w, "decoder", &cfg, Dtype::Float32)
        .unwrap()
        .forward(latent)
        .unwrap();

    let mut w = model_weights();
    let zeroed = Array::zeros::<f32>(&[1, cfg.num_register_tokens, cfg.dim()]).unwrap();
    w.insert("decoder.register_tokens", zeroed);
    let without = ViT3dDecoder::from_weights(&mut w, "decoder", &cfg, Dtype::Float32)
        .unwrap()
        .forward(latent)
        .unwrap();
    let (peak, _) = rel(&without, &baseline);
    assert!(peak > 1e-3, "register tokens do not affect the decode");

    // A config that disagrees with the checkpoint's register-token count must fail loudly.
    let mut bad = cfg.clone();
    bad.num_register_tokens = 2;
    let mut w = model_weights();
    assert!(ViT3dDecoder::from_weights(&mut w, "decoder", &bad, Dtype::Float32).is_err());
}

/// A config claiming full (non-partial) rotary must NOT reproduce the golden — the strongest
/// available evidence that `rope_dim_ratio = 0.75` is genuinely load-bearing rather than a
/// constant nobody reads.
#[test]
fn full_rotary_breaks_parity() {
    let f = fixture();
    let mut cfg = fixture_config(3);
    // 16 · 0.375 = 6 rotated dims instead of 12 — legal for the 3-axis split, but a different
    // model. (A ratio of 1.0 would rotate all 16, which `validate` rejects: 16 is not a
    // multiple of 6, so the shipped 0.75 is not freely adjustable.)
    cfg.rope_dim_ratio = 0.375;
    assert_eq!(cfg.rope_apply_dim(), 6);

    let mut w = model_weights();
    let got = ViT3dDecoder::from_weights(&mut w, "decoder", &cfg, Dtype::Float32)
        .unwrap()
        .forward(f.require("in.vit.latent").unwrap())
        .unwrap();
    let (peak, _) = rel(&got, f.require("out.vit.video").unwrap());
    assert!(
        peak > 1e-2,
        "full rotary reproduced the partial-rotary golden (peak rel {peak:.3e}); the ratio is \
         not reaching the rotary"
    );
}
