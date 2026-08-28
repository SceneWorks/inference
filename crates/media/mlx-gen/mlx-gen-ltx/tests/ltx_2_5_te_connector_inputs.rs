//! LTX-2.5 text encoder → **connector inputs**, on real tier weights (sc-18770).
//!
//! The "connector input" is the feature-extractor output that enters the `Embeddings1DConnector`:
//! `video_features` `(1, L, 4096)` and `audio_features` `(1, L, 2048)`, post-projection and
//! post-norm. `LtxTextEncoder::encode_av_with_features` returns the 2.3 pair and
//! [`Ltx25TextEncoder::encode_av_with_features`] returns the 2.5 pair, deliberately in the same
//! shape so the two are comparable.
//!
//! # The two fixtures this file reads
//!
//! * `tests/fixtures/ltx25_te_connector_golden.safetensors` — the **numeric oracle**, dumped from
//!   the upstream reference (Lightricks/LTX-2 v1.2.0) on the real *unquantized*
//!   `gemma4-12b-with-proj` encoder and the real 2.5 DiT connectors by
//!   `tools/dump_ltx25_te_connector_golden.py`. Backend-neutral: it carries its own `input_ids` and
//!   `mask01`, so this test and its candle twin consume one tokenization and one oracle.
//! * `tests/fixtures/ltx_connector_golden.safetensors` — the LTX-2.3 / eros fixture, read **only**
//!   for its connector-input geometry, so the contract this file enforces is the reference's rather
//!   than this file's opinion. Never a numeric oracle for 2.5.
//!
//! Run:
//! `LTX25_TIER_DIR=/Volumes/Models/scratch-tiers-sc18775/tiers cargo test -p mlx-gen-ltx --release
//!  --test integration -- ltx_2_5_te_connector_inputs:: --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract, sum};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use mlx_gen::gen_core::ltx_checkpoint::LtxCheckpointMetadata;
use mlx_gen::gen_core::OffloadPolicy;
use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::gemma4_te::Ltx25TextEncoder;
use mlx_gen_ltx::tokenizer::Ltx25Tokenizer;
use mlx_gen_ltx::transformer::Precision;

/// The committed 2.3 reference fixture. Consumed **by path** for its connector-input geometry; never
/// re-recorded, and never used as a numeric oracle for 2.5 (see the module docs).
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_connector_golden.safetensors"
);

/// The LTX-**2.5** connector-input golden — the numeric oracle (see the module docs).
const GOLDEN_25: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx25_te_connector_golden.safetensors"
);

const PROMPT: &str = "A slow dolly shot across a rain-slicked street at night, neon reflections.";

/// The reference fixture's sequence length, and not an arbitrary choice: the connector prepends 128
/// learnable registers and refuses a sequence shorter than that count
/// (`ltx connector: sequence length N is smaller than the register count 128`, measured on the real
/// q8 tier at 64). 256 is what `tools/dump_ltx_connector_golden.py` dumped, so the geometry checked
/// here is the geometry the reference was recorded at.
const MAX_LEN: usize = 256;

/// The tier root. A missing env var is a hard failure, not a skip: `#[ignore]` is the only opt-out,
/// so a test that runs must actually run.
fn tier_dir() -> std::path::PathBuf {
    let root = std::env::var("LTX25_TIER_DIR").unwrap_or_else(|_| {
        panic!(
            "set LTX25_TIER_DIR to the built LTX-2.5 tier root (the directory holding q4/q8/bf16)"
        )
    });
    std::path::PathBuf::from(root).join("q8")
}

/// The tier the **numeric** gate runs on: `bf16`, i.e. the dense one.
///
/// Not a convenience — it is what makes the comparison meaningful. The oracle is an f32 forward
/// over the unquantized upstream checkpoint, so the port must be given the closest thing it has:
/// dense bf16 weights, bf16 activations. That is exactly the shape of comparison the 2.3 TE
/// sibling makes (LTX-2.3 ships its connector dense), which is why its `1.5e-2` / `6e-2` bars
/// transfer. Running this against the `q8` tier instead would fold the tier's own quantization
/// error into a text-encoder correctness gate — a *tier-quality* question, which
/// `mlx-llm`'s `ltx_2_5_te_tier_quality.rs` owns and this file must not silently answer.
fn golden_tier_dir() -> std::path::PathBuf {
    let root = std::env::var("LTX25_TIER_DIR").unwrap_or_else(|_| {
        panic!(
            "set LTX25_TIER_DIR to the built LTX-2.5 tier root (the directory holding q4/q8/bf16)"
        )
    });
    std::path::PathBuf::from(root).join("bf16")
}

/// `(video_dim, audio_dim)` read off the committed reference fixture rather than hard-coded, so the
/// contract this test enforces is the reference's, not this file's opinion. The sequence length is
/// not returned but *asserted* against [`MAX_LEN`], so the two cannot drift apart silently.
fn golden_connector_dims() -> (i32, i32) {
    let g = Weights::from_file(GOLDEN).expect("committed connector golden");
    let features = g.require("features").expect("features");
    let audio = g.require("audio_features").expect("audio_features");
    assert_eq!(
        features.shape()[1] as usize,
        MAX_LEN,
        "MAX_LEN must track the reference fixture's sequence length"
    );
    (features.shape()[2], audio.shape()[2])
}

/// `(checkpoint, te_path, connector weights, config, precision)` — everything both constructors
/// need, read off the tier once.
fn tier_inputs(
    dir: &std::path::Path,
) -> (
    LtxCheckpointMetadata,
    std::path::PathBuf,
    Weights,
    LtxConfig,
    Precision,
) {
    let te_path = dir.join("text_encoder.safetensors");
    let cfg = LtxConfig::from_model_dir(dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(dir).expect("split_model.json");
    let checkpoint = LtxCheckpointMetadata::from_file(dir.join("transformer.safetensors"))
        .expect("transformer metadata");
    let connector_w =
        Weights::from_file(dir.join("connector.safetensors")).expect("connector.safetensors");
    (
        checkpoint,
        te_path,
        connector_w,
        cfg,
        // bf16 activations at the tier's declared quant geometry — the q8 tier packs both the
        // connector and the aggregate projections, so the quantized arm is the one exercised.
        Precision::quant_bf16(split.bits, split.group),
    )
}

fn encoder(dir: &std::path::Path) -> (Ltx25TextEncoder, Ltx25Tokenizer) {
    let (checkpoint, te_path, connector_w, cfg, prec) = tier_inputs(dir);
    let te = Ltx25TextEncoder::from_packed_av(
        &checkpoint,
        &te_path,
        &connector_w,
        &cfg,
        prec,
        OffloadPolicy::Resident,
    )
    .expect("build the LTX-2.5 text encoder");
    let tok = Ltx25Tokenizer::from_packed_te_file(&te_path).expect("packed tokenizer");
    (te, tok)
}

fn finite(name: &str, t: &Array) {
    eval([t]).expect("eval");
    let hi = max(abs(t).unwrap(), None).unwrap().item::<f32>();
    assert!(hi.is_finite(), "{name} contains a non-finite value");
    // A head that failed to bind, or a stack that never got past the embedding, produces an
    // all-zero tensor that every shape assertion above would still pass.
    assert!(
        hi > 0.0,
        "{name} is identically zero — the head produced nothing"
    );
}

#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 tiers (LTX25_TIER_DIR); one ~14 GB encoder load"]
fn connector_inputs_match_the_reference_geometry_and_are_well_formed() {
    let dir = tier_dir();
    let (video_dim, audio_dim) = golden_connector_dims();
    let (te, tok) = encoder(&dir);

    let (input_ids, mask) = tok.encode(PROMPT, MAX_LEN).expect("tokenize");
    let (vf, af, ve, ae) = te
        .encode_av_with_features(&input_ids, &mask)
        .expect("encode_av_with_features");

    // The connector inputs carry the reference's dims, at the tier's own sequence length.
    assert_eq!(
        vf.shape(),
        &[1, MAX_LEN as i32, video_dim],
        "video connector input geometry"
    );
    assert_eq!(
        af.shape(),
        &[1, MAX_LEN as i32, audio_dim],
        "audio connector input geometry"
    );
    finite("video_features", &vf);
    finite("audio_features", &af);

    // ...and the connector accepts them, returning at the same dims. This is the seam the AC is
    // about: the 2.5 encoder's output is what the 2.5 connector consumes.
    assert_eq!(ve.shape(), &[1, MAX_LEN as i32, video_dim]);
    assert_eq!(ae.shape(), &[1, MAX_LEN as i32, audio_dim]);
    finite("video_embeddings", &ve);
    finite("audio_embeddings", &ae);

    eprintln!(
        "ltx_2_5 connector inputs: video {:?} audio {:?}",
        vf.shape(),
        af.shape()
    );
}

/// `max|got - want| / max|want|` — the peak-relative error the 2.3 TE sibling (`te_parity.rs`)
/// asserts on, and the same bars are used below.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(mlx_rs::Dtype::Float32).expect("f32");
    let diff = abs(subtract(&got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// [`peak_rel`] restricted to token rows `[lo, hi)`, with the denominator taken over the same
/// slice.
///
/// The global peak denominator is `max|want|` over the *whole* tensor, which the register /
/// pad region can dominate; a mismatch confined to the valid rows would then be divided by a
/// number that has nothing to do with it. This asks the question again where it matters.
fn peak_rel_rows(got: &Array, want: &Array, lo: i32, hi: i32) -> f32 {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    let g = got.take_axis(idx.clone(), 1).expect("slice got");
    let w = want.take_axis(idx, 1).expect("slice want");
    peak_rel(&g, &w)
}

/// The four tensors both numeric tests compare, run once off the golden's own tokenization.
fn against_the_golden() -> (Weights, i32, Array, Array, Array, Array) {
    let dir = golden_tier_dir();
    let g = Weights::from_file(GOLDEN_25).expect("the LTX-2.5 connector-input golden");
    let input_ids = g.require("input_ids").expect("input_ids").clone();
    let mask01 = g.require("mask01").expect("mask01").clone();
    assert_eq!(
        input_ids.shape(),
        &[1, MAX_LEN as i32],
        "the golden's tokenization must be the geometry this file runs at"
    );
    // Feed the golden's own ids/mask rather than re-tokenizing: the oracle and the port must be
    // answering the same question, and a tokenizer drift would otherwise read as a numeric one.
    let nv = sum(&mask01, None).unwrap().item::<i32>();
    assert!(
        nv > 0 && nv < MAX_LEN as i32,
        "the golden must carry a real pad run, got {nv} valid of {MAX_LEN}"
    );

    let (te, _) = encoder(&dir);
    let (vf, af, ve, ae) = te
        .encode_av_with_features(&input_ids, &mask01)
        .expect("encode_av_with_features");
    (g, nv, vf, af, ve, ae)
}

/// **The numeric gate on the text encoder.** `video_features` / `audio_features` — the connector
/// *inputs*, which is what this story is about — must reproduce the upstream reference.
///
/// This is the whole Gemma 4 stack under test: 49 hidden states under the causal+padding mask, the
/// per-token-RMS V2 extractor, and both `text_embedding_projection.*_aggregate_embed` heads.
///
/// The bar is the 2.3 TE sibling's `1.5e-2` peak-relative (`te_parity.rs`), not the diffvae
/// golden's 2e-3 — this compares a bf16 port against an f32 reference. Measured on the bf16 tier:
/// `video_features 2.282e-3`, `audio_features 1.432e-3`, i.e. ~6× inside the bar.
///
/// Each bar is asserted **twice**: once globally, and once over the valid-token rows alone (the
/// suffix, since the features keep the tokenizer's left-padded order), so a mismatch confined to
/// the prompt cannot hide behind a peak denominator set by the pad region.
///
/// **This test is what the padding-mask fix is proved against.** With
/// `masked_hidden_states_in_order` reverted to `AttnMask::Causal` it goes RED at
/// `video_features 8.212e-1`; with the fix it is `2.282e-3`.
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 bf16 tier (LTX25_TIER_DIR); one ~24 GB encoder load"]
fn connector_inputs_match_the_2_5_reference_golden() {
    let (g, nv, vf, af, _, _) = against_the_golden();
    let want_vf = g.require("video_features").expect("video_features");
    let want_af = g.require("audio_features").expect("audio_features");

    let (pr_vf, pr_af) = (peak_rel(&vf, want_vf), peak_rel(&af, want_af));
    let lo = MAX_LEN as i32 - nv;
    let v_vf = peak_rel_rows(&vf, want_vf, lo, MAX_LEN as i32);
    let v_af = peak_rel_rows(&af, want_af, lo, MAX_LEN as i32);
    eprintln!(
        "ltx_2_5 connector INPUTS vs reference: video_features {pr_vf:.3e} (valid {v_vf:.3e}) \
         audio_features {pr_af:.3e} (valid {v_af:.3e})"
    );

    assert!(pr_vf < 1.5e-2, "video_features peak_rel {pr_vf:.3e}");
    assert!(pr_af < 1.5e-2, "audio_features peak_rel {pr_af:.3e}");
    assert!(
        v_vf < 1.5e-2,
        "video_features valid-row peak_rel {v_vf:.3e}"
    );
    assert!(
        v_af < 1.5e-2,
        "audio_features valid-row peak_rel {v_af:.3e}"
    );
}

/// The same gate one stage further on — GREEN since sc-21663 fixed the `crate::connector` defects
/// this golden surfaced.
///
/// The defect this test found (it was committed RED at `video 1.275e0 / audio 1.771e0`): the
/// connector — code **shared with LTX-2.3** — had only ever been pinned against `mlx_video`'s
/// `Embeddings1DConnector`, which disagrees with the canonical `ltx_core` (the training stack) in
/// two semantics (the per-head gate is `2·sigmoid`, not `sigmoid`; the FFN GELU is
/// tanh-approximate, not erf) plus the RoPE table's f32 index quantization. This golden was the
/// first comparison against upstream's own implementation. sc-21663 fixed all three (both
/// backends), re-derived the 2.3 golden from the correct authority, and moved the connector's
/// activations to f32 — the connector's closing per-row RMS-norm rescales rows whose magnitudes
/// span >100x, so bf16 activation rounding alone reached `6.094e-2` against this bar on the same
/// run (see `connector.rs`'s dtype doc for both measured comparisons and the mechanism).
///
/// Measured after the fix on the bf16 (dense) tier:
///
/// ```text
/// video_embeddings  9.107e-3  (valid rows 2.224e-3)     bar 6e-2
/// audio_embeddings  1.141e-2  (valid rows 2.178e-3)     bar 6e-2
/// ```
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 bf16 tier (LTX25_TIER_DIR); one ~24 GB encoder load"]
fn connector_outputs_match_the_2_5_reference_golden() {
    let (g, nv, _, _, ve, ae) = against_the_golden();
    let want_ve = g.require("video_embeddings").expect("video_embeddings");
    let want_ae = g.require("audio_embeddings").expect("audio_embeddings");

    let (pr_ve, pr_ae) = (peak_rel(&ve, want_ve), peak_rel(&ae, want_ae));
    // The connector reorders its input, so here the valid rows are the PREFIX.
    let v_ve = peak_rel_rows(&ve, want_ve, 0, nv);
    let v_ae = peak_rel_rows(&ae, want_ae, 0, nv);
    eprintln!(
        "ltx_2_5 connector OUTPUTS vs reference: video_emb {pr_ve:.3e} (valid {v_ve:.3e}) \
         audio_emb {pr_ae:.3e} (valid {v_ae:.3e})"
    );

    assert!(pr_ve < 6e-2, "video_embeddings peak_rel {pr_ve:.3e}");
    assert!(pr_ae < 6e-2, "audio_embeddings peak_rel {pr_ae:.3e}");
    assert!(
        v_ve < 6e-2,
        "video_embeddings valid-row peak_rel {v_ve:.3e}"
    );
    assert!(
        v_ae < 6e-2,
        "audio_embeddings valid-row peak_rel {v_ae:.3e}"
    );
}

/// The video-only constructor is reachable and genuinely leaves the audio head absent.
///
/// Before the `from_packed_video` split, `audio` was `Some` on every constructed encoder and both
/// `ok_or_else` arms in `encode_av*` were dead code. This is the test that makes them live: build
/// video-only, check the video path still works, and check the AV path refuses by name rather than
/// panicking or silently returning the video features twice.
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 tiers (LTX25_TIER_DIR); one ~14 GB encoder load"]
fn the_video_only_constructor_omits_the_audio_head() {
    let dir = tier_dir();
    let (video_dim, _) = golden_connector_dims();
    let (checkpoint, te_path, connector_w, cfg, prec) = tier_inputs(&dir);
    let te = Ltx25TextEncoder::from_packed_video(
        &checkpoint,
        &te_path,
        &connector_w,
        &cfg,
        prec,
        OffloadPolicy::Resident,
    )
    .expect("build the video-only LTX-2.5 text encoder");
    let tok = Ltx25Tokenizer::from_packed_te_file(&te_path).expect("packed tokenizer");

    let (input_ids, mask) = tok.encode(PROMPT, MAX_LEN).expect("tokenize");
    let (vf, ve) = te
        .encode_with_features(&input_ids, &mask)
        .expect("the video path must work without an audio head");
    assert_eq!(vf.shape(), &[1, MAX_LEN as i32, video_dim]);
    assert_eq!(ve.shape(), &[1, MAX_LEN as i32, video_dim]);
    finite("video_features", &vf);
    finite("video_embeddings", &ve);

    let err = match te.encode_av_with_features(&input_ids, &mask) {
        Ok(_) => panic!("the AV path must refuse a video-only encoder"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("from_packed_video"),
        "the refusal must name the constructor that produced this encoder: {err}"
    );
}

/// The mask half of the contract: padded positions must carry **no token-dependent** information.
///
/// Note what this is NOT. The extractor zeroes the padded rows of `normed`, but the aggregate
/// projection then adds its bias, so a pad row of the connector input equals the bias — not zero.
/// Asserting "pad rows are zero" would therefore fail against a correct encoder, and asserting
/// "pad rows are small" would pass against a broken one. The property that actually holds, and the
/// one that matters, is that **every pad row is identical to every other pad row**: whatever
/// constant the bias contributes, no pad position may vary with the prompt. Dropping the mask makes
/// pad rows vary, which this catches; keeping it makes them constant.
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 tiers (LTX25_TIER_DIR); one ~14 GB encoder load"]
fn padded_positions_carry_no_token_dependent_conditioning() {
    let dir = tier_dir();
    let (te, tok) = encoder(&dir);

    let (input_ids, mask) = tok.encode(PROMPT, MAX_LEN).expect("tokenize");
    let mask_host = mask.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    eval([&mask_host]).expect("eval");
    let mask_v = mask_host.as_slice::<f32>().to_vec();
    let pads = mask_v.iter().filter(|m| **m == 0.0).count();
    assert!(
        pads >= 2,
        "the prompt must be short enough to leave at least two padded positions, got {pads}"
    );
    // Left-padding: the pad run is the prefix (`gen_core`'s tokenizer policy).
    assert!(
        mask_v[..pads].iter().all(|m| *m == 0.0),
        "the tokenizer must left-pad, so the pad run is the prefix"
    );

    let (vf, _, _, _) = te
        .encode_av_with_features(&input_ids, &mask)
        .expect("encode_av_with_features");
    let vf_host = vf.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    eval([&vf_host]).expect("eval");
    let v = vf_host.as_slice::<f32>().to_vec();

    let dim = vf.shape()[2] as usize;
    let row = |t: usize| &v[t * dim..(t + 1) * dim];

    // Every pad row equals the first pad row.
    let first = row(0);
    let mut worst = 0f32;
    for t in 1..pads {
        for (a, b) in row(t).iter().zip(first.iter()) {
            worst = worst.max((a - b).abs());
        }
    }
    eprintln!("pad-row spread across {pads} padded positions: {worst:.3e}");
    assert!(
        worst < 1e-2,
        "padded connector-input rows must not vary with the prompt (spread {worst:.3e}) — the \
         feature extractor's mask was dropped"
    );

    // ...and the valid rows must differ from that constant, or the assertion above would also pass
    // on an encoder that emitted the bias everywhere.
    let last = row(MAX_LEN - 1);
    let signal = last
        .iter()
        .zip(first.iter())
        .fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
    eprintln!("valid-vs-pad row separation: {signal:.3e}");
    assert!(
        signal > 1e-2,
        "a valid position must differ from the padded constant (separation {signal:.3e}) — the \
         encoder produced no conditioning at all"
    );
}
