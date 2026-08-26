//! LTX-2.5 text encoder → **connector inputs**, on real tier weights (sc-18770).
//!
//! The "connector input" is the feature-extractor output that enters the `Embeddings1DConnector`:
//! `video_features` `(1, L, 4096)` and `audio_features` `(1, L, 2048)`, post-projection and
//! post-norm. `LtxTextEncoder::encode_av_with_features` returns the 2.3 pair and
//! [`Ltx25TextEncoder::encode_av_with_features`] returns the 2.5 pair, deliberately in the same
//! shape so the two are comparable.
//!
//! # What this file can and cannot gate — read before adding a numeric assertion
//!
//! The committed fixture `tests/fixtures/ltx_connector_golden.safetensors` holds the reference
//! connector inputs (`features`, `audio_features`) **for LTX-2.3 / eros**, dumped from the PyTorch
//! reference by `tools/dump_ltx_connector_golden.py`. It is the fixture `connector_parity.rs` and
//! its candle twin consume, and it is what pins the connector itself on both backends.
//!
//! There is **no LTX-2.5 connector-input golden in the repository.** The 2.5 text encoder is a
//! different model with different weights, so its connector inputs are numerically unrelated to the
//! 2.3 fixture — asserting equality against it would be meaningless, and manufacturing a 2.5 golden
//! here would mean re-recording an oracle from our own implementation, which is exactly the
//! circularity a golden exists to prevent. Producing a genuine 2.5 golden needs the gated upstream
//! reference and belongs to the epic's terminal measurement campaign (sc-18783).
//!
//! So this file gates what is honestly checkable on real weights: the connector-input **contract**
//! (geometry, dtype, finiteness, non-degeneracy, mask semantics) read off the committed fixture
//! rather than hard-coded, and the fact that the produced features are actually accepted by the 2.5
//! connector. The numeric oracle gate is named above so nobody mistakes this for one.
//!
//! Run:
//! `LTX25_TIER_DIR=/Volumes/Models/scratch-tiers-sc18775/tiers cargo test -p mlx-gen-ltx --release
//!  --test integration -- ltx_2_5_te_connector_inputs:: --ignored --nocapture`

use mlx_rs::ops::{abs, max};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use mlx_gen::gen_core::ltx_checkpoint::LtxCheckpointMetadata;
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

/// `(seq, video_dim, audio_dim)` read off the committed reference fixture rather than hard-coded, so
/// the contract this test enforces is the reference's, not this file's opinion. Also pins [`MAX_LEN`]
/// to the fixture's own sequence length, so the two cannot drift apart silently.
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

fn encoder(dir: &std::path::Path) -> (Ltx25TextEncoder, Ltx25Tokenizer) {
    let te_path = dir.join("text_encoder.safetensors");
    let cfg = LtxConfig::from_model_dir(dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(dir).expect("split_model.json");
    let checkpoint = LtxCheckpointMetadata::from_file(dir.join("transformer.safetensors"))
        .expect("transformer metadata");
    let connector_w =
        Weights::from_file(dir.join("connector.safetensors")).expect("connector.safetensors");

    let te = Ltx25TextEncoder::from_packed_av(
        &checkpoint,
        &te_path,
        &connector_w,
        &cfg,
        // bf16 activations at the tier's declared quant geometry — the q8 tier packs both the
        // connector and the aggregate projections, so the quantized arm is the one exercised.
        Precision::quant_bf16(split.bits, split.group),
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
