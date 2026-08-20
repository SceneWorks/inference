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
    // 5 latent frames × 4, 3×2 spatial × 2 -> [1, 3, 20, 6, 8].
    assert_eq!(got.shape(), &[1, 3, 20, 6, 8]);
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
    let vae = vae(3);
    for (tokens, frames) in [(5, 17), (7, 22), (9, 30), (12, 39), (17, 56)] {
        let latent = f.require(&format!("in.temporal{tokens}.latent")).unwrap();
        let want = f.require(&format!("out.temporal{tokens}.video")).unwrap();
        let got = vae.decode(latent).unwrap();
        assert_eq!(
            got.shape(),
            &[1, 3, frames, 6, 8],
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
        assert_eq!(got.shape(), &[1, 3, frames, 6, 8]);
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

    // ...and conversely, the declared name list is exactly the fixture's tensor set — **for the
    // decode half**. `tensor_names` covers the whole published `vae/` component since sc-17148
    // added the encoder, but this fixture deliberately carries only the decode half: its bytes are
    // shared verbatim with `candle-gen-minimax-h3`, whose `cross_backend.rs` digests them, so
    // adding tensors here would break that crate's gate. `video_vae_encode_parity.rs` holds the
    // encode half against `video_vae_encode.safetensors`, and `real_weights.rs` asserts the union
    // is exactly the published 703.
    let is_encode_half = |k: &str| k.starts_with("encoder.") || k.starts_with("quant_conv.");
    assert!(
        !before.iter().any(|k| is_encode_half(k)),
        "the decode fixture must not carry encode-half tensors"
    );
    let declared: std::collections::BTreeSet<String> = MiniMaxH3VideoVae::tensor_names(&cfg)
        .into_iter()
        .filter(|k| !is_encode_half(k))
        .collect();
    assert_eq!(
        declared, before,
        "declared decode-half tensor names differ from the checkpoint's"
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

// ---------------------------------------------------------------------------------------------
// Spatial tiling (sc-18786)
// ---------------------------------------------------------------------------------------------
//
// `AutoencoderKLMiniMaxH3` ships with `use_tiling = True` and upstream is explicit that **the
// released frames are the blended-tile ones, so disabling tiling changes the output**. sc-17140
// decoded the whole canvas in one pass, which is the reference only below one 256 px tile.
//
// These are the *committed* gates — they run on every CI pass, without a snapshot. They cannot
// prove the tiled numbers are right (only real weights against diffusers can, see
// `real_weights.rs::real_weight_tiled_decode_matches_the_official_diffusers_vae`); what they prove
// is the two halves of the sc-18740 lesson: that tiling is inert where the committed fixtures live,
// and that the new code path is nevertheless *reachable and distinguishable* rather than untested
// by construction.

/// The fixture canvas is 6x8 px (a 3x4 latent at the fixture's 2x ratio) — three orders of
/// magnitude below one 256 px tile. So `decode_clip` with the shipped defaults must be **bit**
/// identical to the untiled path, which is what keeps every reference-backed decode assertion in
/// this file (and its byte-identical candle twin) valid across the tiling change.
#[test]
fn shipped_tiling_is_on_by_default_and_bit_inert_below_one_tile() {
    let vae = vae(3);
    let t = vae.tiling();
    assert!(t.enabled, "the shipped VAE tiles by default");
    assert_eq!((t.tile_height, t.tile_width), (256, 256));
    assert_eq!((t.overlap_height, t.overlap_width), (64, 64));

    let f = fixture();
    let latent = f.require("in.decode.latent").unwrap();
    assert_eq!(latent.shape()[3..], [3, 4], "fixture latent is 3x4");

    let tiled = vae.decode_clip(latent).unwrap();
    let untiled = vae.decode_clip_untiled(latent).unwrap();
    assert_eq!(tiled.shape(), untiled.shape());
    let (peak, _) = rel(&tiled, &untiled);
    assert_eq!(
        peak, 0.0,
        "below one tile the tiled path must be BIT-identical, got rel-max-abs {peak:.3e}"
    );
}

/// The plan the fixture canvas takes at a shrunk tile, against the reference's own `_split_tiles`.
///
/// 6 px of height at a 4 px tile is 2 rows; 8 px of width is **3** columns, not 2 — two 4 px tiles
/// at a 2 px overlap span only 6 px. The grid is deliberately non-square so a transposed plan is
/// not accidentally correct.
#[test]
fn the_fixture_canvas_tiles_into_the_reference_grid() {
    use mlx_gen_minimax_h3::spatial_tiling::TilePlan;
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
/// smaller context than the full canvas and cannot agree with it. The measured separation is
/// printed so the margin over the ~1e-3 numeric noise floor stays auditable.
#[test]
fn a_tile_smaller_than_the_canvas_changes_the_decode() {
    let vae = vae(3);
    let f = fixture();
    let latent = f.require("in.decode.latent").unwrap();

    let untiled = vae.decode_clip_untiled(latent).unwrap();
    let tiled = vae.decode_clip_tiled(latent, 4, 4, 2, 2).unwrap();
    assert_eq!(
        tiled.shape(),
        untiled.shape(),
        "tiling must not change the shape"
    );

    let (peak, mean) = rel(&tiled, &untiled);
    println!("SPATIAL TILING 4/2 vs untiled: rel-max-abs={peak:.3e} rel-mean-abs={mean:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "a 2x3 tile grid moved the decode by only {peak:.3e}; the tiled path is not being taken"
    );
}

/// The `enable_tiling` / `disable_tiling` knobs select the paths they name — the reference exposes
/// both and the conditioning encode half reads the same fields.
#[test]
fn the_tiling_knobs_select_the_paths_they_name() {
    use mlx_gen_minimax_h3::spatial_tiling::SpatialTiling;
    let f = fixture();
    let latent = f.require("in.decode.latent").unwrap();
    let base = vae(3);
    let untiled = base.decode_clip_untiled(latent).unwrap();
    let tiled = base.decode_clip_tiled(latent, 4, 4, 2, 2).unwrap();

    // `enable_tiling` at the shrunk geometry routes `decode_clip` to the tiled path, exactly.
    let mut on = vae(3);
    on.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    assert_eq!(rel(&on.decode_clip(latent).unwrap(), &tiled).0, 0.0);

    // `disable_tiling` routes it to the untiled path even at a geometry that would tile.
    let mut off = vae(3);
    off.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    off.disable_tiling();
    assert!(!off.tiling().enabled);
    assert_eq!(rel(&off.decode_clip(latent).unwrap(), &untiled).0, 0.0);

    // …and the builder form agrees with the setters.
    let built = vae(3).with_tiling(SpatialTiling::disabled());
    assert_eq!(rel(&built.decode_clip(latent).unwrap(), &untiled).0, 0.0);
    let built_on = vae(3).with_tiling(SpatialTiling::square(4, 2));
    assert_eq!(rel(&built_on.decode_clip(latent).unwrap(), &tiled).0, 0.0);
}

/// Tiling composes with the temporal chunking rather than replacing it: `decode` still produces the
/// planned frame count, at a canvas that tiles spatially.
#[test]
fn spatial_tiling_composes_with_the_temporal_chunk_plan() {
    let f = fixture();
    let mut vae = vae(3);
    vae.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    // 12 tokens spans multiple temporal chunks AND now tiles spatially in both axes.
    let latent = f.require("in.temporal12.latent").unwrap();
    let want = f.require("out.temporal12.video").unwrap();
    let got = vae.decode(latent).unwrap();
    assert_eq!(
        got.shape(),
        want.shape(),
        "the spatial grid must not disturb the temporal frame plan"
    );
    // The values legitimately differ from the untiled golden — that is the point — but the frame
    // plan, which is what the temporal path owns, must be untouched.
    let (peak, _) = rel(&got, want);
    println!("TILED multi-chunk decode vs untiled golden: rel-max-abs={peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the spatial grid was not applied under chunking"
    );
}

// ---------------------------------------------------------------------------------------------
// Rung 2 — bounded decode (sc-18660)
// ---------------------------------------------------------------------------------------------

/// Per-frame relative max-abs between two `[B, C, T, H, W]` videos.
///
/// Gated on **relative max-abs-diff** only. Norm, cosine and checksum were each blind to a real
/// defect in this family and are deliberately absent from every assertion below.
fn per_frame_rel(a: &Array, b: &Array) -> Vec<f32> {
    let frames = a.shape()[2];
    (0..frames)
        .map(|t| {
            let idx = Array::from_slice(&[t], &[1]);
            let fa = a.take_axis(&idx, 2).unwrap();
            let fb = b.take_axis(&idx, 2).unwrap();
            rel(&fa, &fb).0
        })
        .collect()
}

/// The temporal derivative `v[t+1] - v[t]` — the field a *cross-frame* discrepancy lives in.
///
/// A per-frame metric can be satisfied by a video whose every frame is individually plausible
/// while the motion between them is wrong, which is precisely the tile-starvation failure mode the
/// story names: the corruption is visible across frames, not within one.
fn temporal_delta(v: &Array) -> Array {
    let t = v.shape()[2];
    let head = mlx_gen_minimax_h3::tensor::slice_axis(v, 2, 1, t).unwrap();
    let tail = mlx_gen_minimax_h3::tensor::slice_axis(v, 2, 0, t - 1).unwrap();
    mlx_rs::ops::subtract(&head, &tail).unwrap()
}

/// **The bounded stitch is bit-identical to the full-grid stitch, through the real decoder.**
///
/// `spatial_tiling::tests::bounded_stitch_matches_the_full_grid_stitch` proves the equality on
/// synthetic tiles; this proves the *decode path* takes it, by decoding the same grid by hand and
/// stitching it the old way. If `decode_clip_tiled` ever streams a different grid — a transposed
/// push order, a dropped tile, a strip captured after the blend instead of before — this separates
/// while every parity assertion in this file stays green.
#[test]
fn the_bounded_stitch_reproduces_the_full_grid_decode_exactly() {
    use mlx_gen_minimax_h3::spatial_tiling::{stitch_tiles, TilePlan};
    let f = fixture();
    let vae = vae(3);
    let cfg = fixture_config(3);
    let latent = f.require("in.decode.latent").unwrap();
    let (ratio, tile, overlap) = (cfg.patch_size, 4, 2);

    // Rebuild the grid the way the pre-rung-2 code did: decode every tile, hold them all, stitch.
    let s = latent.shape();
    let rows = TilePlan::split(s[3] * ratio, tile, overlap, ratio).unwrap();
    let cols = TilePlan::split(s[4] * ratio, tile, overlap, ratio).unwrap();
    assert!(
        rows.len() > 1 && cols.len() > 1,
        "the comparison is vacuous on a single-tile grid: got {}x{}",
        rows.len(),
        cols.len()
    );
    let mut grid = Vec::with_capacity(rows.len());
    for (i, &y) in rows.starts.iter().enumerate() {
        let mut row = Vec::with_capacity(cols.len());
        for (j, &x) in cols.starts.iter().enumerate() {
            let (y0, x0) = (y / ratio, x / ratio);
            let t =
                mlx_gen_minimax_h3::tensor::slice_axis(latent, 3, y0, y0 + rows.lengths[i] / ratio)
                    .unwrap();
            let t = mlx_gen_minimax_h3::tensor::slice_axis(&t, 4, x0, x0 + cols.lengths[j] / ratio)
                .unwrap();
            row.push(vae.decode_clip_untiled(&t).unwrap());
        }
        grid.push(row);
    }
    let full = stitch_tiles(&grid, &rows.overlaps, &cols.overlaps).unwrap();
    let bounded = vae
        .decode_clip_tiled(latent, tile, tile, overlap, overlap)
        .unwrap();

    assert_eq!(bounded.shape(), full.shape());
    let (peak, _) = rel(&bounded, &full);
    assert_eq!(
        peak, 0.0,
        "the bounded stitch must be BIT-identical to the full-grid stitch, got {peak:.3e}"
    );
}

/// **A starved tile overlap corrupts the decode, and it does so across every frame** — the AC2
/// correctness assertion, which no memory number can stand in for.
///
/// A zero overlap abuts the tiles with no cross-fade. Two tolerances are stated and both are
/// violated by a wide margin:
///
/// * **per-frame** — `PER_FRAME_TOL`, the same 1e-2 mutation floor the rest of this file uses,
///   ~10x the ~1e-3 reduced-precision noise. **Every** frame must exceed it, not merely the worst:
///   a corruption confined to one frame would be a different (and much more visible) defect.
/// * **cross-frame** — `CROSS_FRAME_TOL`, on the temporal derivative. This is the tolerance that
///   makes the guard specific to the failure mode the story names, and it is the one a per-frame
///   spot check cannot reach.
///
/// The overlap is the *only* thing that moves: same weights, same tile size, same latent, same
/// temporal plan. And the run is over `in.temporal12.latent`, which spans multiple temporal chunks,
/// so the frame sequence under test is a real one rather than a single clip.
#[test]
fn a_starved_tile_overlap_corrupts_the_decode_across_frames() {
    /// Set from **this probe's own measured minimum**, not inherited from [`MUTATION_FLOOR`].
    ///
    /// Measured per-frame separation at this fixture geometry is 7.238e-3 … 3.324e-2 over 39
    /// frames. The *worst* frame clears the 1e-2 house mutation floor comfortably, but the least
    /// affected frame does not — so requiring 1e-2 of **every** frame would be a bound this probe
    /// cannot show, and lowering the claim to "the worst frame moved" would let a corruption
    /// confined to one frame pass. 5e-3 is ~1.45x below the measured minimum and ~5x above the
    /// ~1e-3 reduced-precision noise floor this file documents.
    const PER_FRAME_TOL: f32 = 5e-3;
    /// The temporal-derivative bound, at the house mutation floor. Measured cross-frame separation
    /// is **2.336e-2**, so the margin is ~2.3x — real, and deliberately not overstated. It is the
    /// narrowest of the three margins here, which is expected: a seam that is stationary in the
    /// canvas perturbs consecutive frames similarly, so much of the error cancels in the
    /// difference. That it survives the cancellation at all is the signal.
    const CROSS_FRAME_TOL: f32 = MUTATION_FLOOR;

    let f = fixture();
    let latent = f.require("in.temporal12.latent").unwrap();

    let mut reference = vae(3);
    reference.enable_tiling(Some(4), Some(4), Some(2), Some(2));
    let mut starved = vae(3);
    starved.enable_tiling(Some(4), Some(4), Some(0), Some(0));
    assert_eq!(
        starved.tiling().overlap_height,
        0,
        "this probe is only meaningful at a zero overlap"
    );

    let good = reference.decode(latent).unwrap();
    let bad = starved.decode(latent).unwrap();
    assert_eq!(
        good.shape(),
        bad.shape(),
        "starvation must corrupt the picture, not the frame plan — a shape change would be caught \
         by assertions that cannot see the corruption itself"
    );
    assert!(
        good.shape()[2] > 2,
        "a cross-frame metric needs more than two frames, got {}",
        good.shape()[2]
    );

    let frames = per_frame_rel(&bad, &good);
    let worst = frames.iter().copied().fold(0.0f32, f32::max);
    let least = frames.iter().copied().fold(f32::INFINITY, f32::min);
    println!(
        "STARVED OVERLAP per-frame rel-max-abs: worst={worst:.3e} least={least:.3e} over {} frames",
        frames.len()
    );
    assert!(
        least > PER_FRAME_TOL,
        "frame {} moved by only {least:.3e}; a starved seam must corrupt EVERY frame, not one",
        frames.iter().position(|&v| v == least).unwrap()
    );

    let (cross, _) = rel(&temporal_delta(&bad), &temporal_delta(&good));
    println!("STARVED OVERLAP cross-frame (temporal-derivative) rel-max-abs: {cross:.3e}");
    assert!(
        cross > CROSS_FRAME_TOL,
        "the temporal derivative moved by only {cross:.3e}; the cross-frame guard is inert"
    );

    // Non-vacuity: the SAME two metrics report zero when nothing is starved, so a green run above
    // is the metric firing rather than the metric being unable to report a small number.
    let again = reference.decode(latent).unwrap();
    assert_eq!(
        per_frame_rel(&again, &good)
            .iter()
            .copied()
            .fold(0.0f32, f32::max),
        0.0,
        "the per-frame metric must report 0.0 for an identical decode"
    );
    assert_eq!(
        rel(&temporal_delta(&again), &temporal_delta(&good)).0,
        0.0,
        "the cross-frame metric must report 0.0 for an identical decode"
    );
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
    let latent = f.require("in.decode.latent").unwrap();
    assert_eq!(
        fixture_config(3).patch_size,
        2,
        "this test's odd tile sizes are only misaligned at ratio 2"
    );

    // A misaligned TILE, at an aligned overlap.
    let err = vae(3)
        .decode_clip_tiled(latent, 3, 4, 2, 2)
        .expect_err("a 3 px tile is 1.5 latent cells and must be refused");
    let msg = err.to_string();
    assert!(msg.contains("2x spatial compression ratio"), "{msg}");
    assert!(
        msg.contains("silently truncated"),
        "the error must say WHY: {msg}"
    );

    // A misaligned OVERLAP, at an aligned tile. Asserted by MESSAGE for the same reason as the tile
    // clause above (sc-19488): a 1 px overlap that reached the decode would fail anyway, downstream
    // and for an unrelated reason, so `is_err()` alone is satisfied with the guard deleted and the
    // probe is inert. `TilePlan::split` checks tile and overlap in ONE arm, so the message is shared
    // — what distinguishes this clause from the tile clause is the geometry it passes, not the text.
    let err = vae(3)
        .decode_clip_tiled(latent, 4, 4, 1, 2)
        .expect_err("a 1 px overlap is half a latent cell and must be refused");
    let msg = err.to_string();
    assert!(msg.contains("2x spatial compression ratio"), "{msg}");
    assert!(
        msg.contains("silently truncated"),
        "the error must say WHY: {msg}"
    );

    // …and it reaches through `enable_tiling` / `decode_clip`, which is the surface a caller uses.
    let mut misaligned = vae(3);
    misaligned.enable_tiling(Some(3), Some(3), Some(2), Some(2));
    let msg = misaligned
        .decode_clip(latent)
        .expect_err("the guard must reach the public tiling surface")
        .to_string();
    assert!(
        msg.contains("2x spatial compression ratio"),
        "reached through enable_tiling/decode_clip this must still be the ALIGNMENT guard, not a \
         downstream shape fault: {msg}"
    );

    // The aligned geometry still decodes, so the guard rejects the misalignment and nothing else.
    assert!(vae(3).decode_clip_tiled(latent, 4, 4, 2, 2).is_ok());
}
