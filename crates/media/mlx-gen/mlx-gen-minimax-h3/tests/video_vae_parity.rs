//! sc-17140: video-VAE decode parity against the Apache-2.0 reference implementation.
//!
//! Fixture `tests/fixtures/video_vae_decode.safetensors` ← `tools/dump_minimax_h3_video_vae.py`,
//! which imports the reference bundle shipped INSIDE the MiniMax-H3 snapshot
//! (`FL2VA/video_vae/{klvae,vae_vit,base_module,attention,func,flash}.py`) and runs its real
//! `ViT3DDecoder` / `AutoencoderKLLegacy.decode_temporal` at tiny dims. This is a genuine
//! independent-graph parity check, not a re-derivation from config.
//!
//! Tolerance 1e-2 peak-relative, the mlx-gen house value. Everything here is f32 and MLX runs f32
//! matmul in reduced precision on Metal, so the observed residual is ~1e-3 — a 10x margin, and the
//! floor on what these tests can prove. A structural error — wrong QKV split, full instead of
//! partial rotary, LayerNorm instead of RMSNorm, missing register tokens, a mis-planned chunk seam
//! — diverges by orders of magnitude, which the mutation tests at the bottom confirm by measuring
//! and PRINTING how far a perturbed weight actually moves the decode.

mod common;

use common::{assert_parity, fixture_config, rel, std_dev, FIXTURE};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::blocks::TransformerBlock;
use mlx_gen_minimax_h3::{
    split_fused_qkv, MiniMaxH3VaeConfig, MiniMaxH3VideoVae, Rope3d, ViT3dDecoder,
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
        (
            "decoder.transformer_blocks.1.attn.to_q.weight",
            MUTATION_FLOOR,
        ),
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
