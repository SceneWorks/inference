//! LTX-2.5 text encoder → **connector inputs**, on real tier weights (sc-18770) — the candle twin
//! of `mlx-gen-ltx`'s `tests/ltx_2_5_te_connector_inputs.rs`.
//!
//! `#![cfg(feature = "cuda")]`: candle's plain CPU backend has no bf16 matmul, the same reason
//! `tests/conformance.rs` and `tests/te_parity.rs` are cuda-only on this backend.
//!
//! # The two fixtures this file reads
//!
//! Both live in `mlx-gen-ltx/tests/fixtures/` and are reached across the backend boundary exactly
//! as `connector_parity.rs` reaches them — one file, one oracle, both backends.
//!
//! * `ltx25_te_connector_golden.safetensors` — the **numeric oracle**, dumped from the upstream
//!   reference (Lightricks/LTX-2 v1.2.0) on the real *unquantized* `gemma4-12b-with-proj` encoder
//!   and the real 2.5 DiT connectors by `mlx-gen/tools/dump_ltx25_te_connector_golden.py`. It
//!   carries its own `input_ids` / `mask01`, so this test does not re-tokenize.
//! * `ltx_connector_golden.safetensors` — the LTX-2.3 / eros fixture, read **only** for its
//!   connector-input geometry. Never a numeric oracle for 2.5.
//!
//! Run:
//! `LTX25_TIER_DIR=<tiers> cargo test -p candle-gen-ltx --features cuda --release --test integration
//!  -- ltx_2_5_te_connector_inputs:: --ignored --nocapture`

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use candle_gen::candle_core::{safetensors, DType, Device, Tensor};
use candle_gen::gen_core::ltx_checkpoint::{CaptionFeatureVersion, LtxCheckpointMetadata};
use candle_gen_ltx::config::{AvConfig, ConnectorConfig};
use candle_gen_ltx::gemma4_te::Ltx25TextEncoder;
use candle_gen_ltx::tier::TierPaths;
use candle_gen_ltx::tokenizer::Ltx25Tokenizer;

/// The committed 2.3 reference fixture, reached across the backend boundary exactly as
/// `connector_parity.rs` reaches it. Consumed **by path** for its geometry; never re-recorded, and
/// never used as a numeric oracle for 2.5 (see the module docs).
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors"
);

/// The LTX-**2.5** connector-input golden — the numeric oracle (see the module docs).
const GOLDEN_25: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx25_te_connector_golden.safetensors"
);

const PROMPT: &str = "A slow dolly shot across a rain-slicked street at night, neon reflections.";

/// The reference fixture's sequence length, and not an arbitrary choice: the connector prepends 128
/// learnable registers and refuses a shorter sequence (measured on the real q8 tier at 64). 256 is
/// what `tools/dump_ltx_connector_golden.py` dumped, so the geometry checked here is the geometry
/// the reference was recorded at — and the same length the MLX twin uses.
const MAX_LEN: usize = 256;

/// The tier root. A missing env var is a hard failure, not a skip: `#[ignore]` is the only opt-out.
fn tier_dir() -> std::path::PathBuf {
    let root = std::env::var("LTX25_TIER_DIR").unwrap_or_else(|_| {
        panic!(
            "set LTX25_TIER_DIR to the built LTX-2.5 tier root (the directory holding q4/q8/bf16)"
        )
    });
    std::path::PathBuf::from(root).join("q8")
}

/// `(video_dim, audio_dim)` read off the committed reference fixture rather than hard-coded, so the
/// contract this test enforces is the reference's — and demonstrably the same one the MLX twin
/// enforces, since both read the same file.
fn golden_connector_dims(device: &Device) -> (usize, usize) {
    let g = safetensors::load(GOLDEN, device).expect("committed connector golden");
    let (_, seq, video) = g["features"].dims3().expect("features rank 3");
    let audio = g["audio_features"]
        .dims3()
        .expect("audio_features rank 3")
        .2;
    assert_eq!(
        seq, MAX_LEN,
        "MAX_LEN must track the reference fixture's sequence length"
    );
    (video, audio)
}

fn max_abs(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("host copy")
        .into_iter()
        .fold(0f32, |a, v| a.max(v.abs()))
}

fn finite(name: &str, t: &Tensor) {
    let hi = max_abs(t);
    assert!(hi.is_finite(), "{name} contains a non-finite value");
    // A head that failed to bind, or a stack that never got past the embedding, produces an
    // all-zero tensor that every shape assertion would still pass.
    assert!(
        hi > 0.0,
        "{name} is identically zero — the head produced nothing"
    );
}

/// `(checkpoint, te_path, connector varbuilder, av config, video connector config, audio connector
/// config)` — everything both constructors need, read off the tier once.
type TierInputs = (
    LtxCheckpointMetadata,
    std::path::PathBuf,
    candle_gen::candle_nn::VarBuilder<'static>,
    AvConfig,
    ConnectorConfig,
    ConnectorConfig,
);

fn tier_inputs(dir: &std::path::Path, device: &Device) -> TierInputs {
    let te_path = dir.join("text_encoder.safetensors");
    let checkpoint = LtxCheckpointMetadata::from_file(dir.join("transformer.safetensors"))
        .expect("transformer metadata");
    let paths = TierPaths::detect(dir, None).expect("LTX25_TIER_DIR/q8 must be a tier directory");
    let root = paths
        .connector_vb(DType::BF16, device)
        .expect("connector varbuilder")
        .pp("model.diffusion_model");

    // Every config comes off THIS checkpoint's own `config.transformer` section — the candle
    // equivalent of the MLX twin's `LtxConfig::from_model_dir`. Handing a 2.5 tier the 2.3
    // constants would leave `require_v2_version`'s per-checkpoint wiring, the
    // `caption_feature_version` detection, and `connector_ff_bias` unexercised on this backend,
    // which is exactly the cross-backend asymmetry this file exists to rule out.
    let transformer = checkpoint
        .section("transformer")
        .expect("the 2.5 transformer must carry a `config.transformer` section");
    let av_cfg = AvConfig::from_transformer_config(transformer).expect("AvConfig from checkpoint");
    let conn_cfg = ConnectorConfig::from_transformer_config(transformer);
    let audio_conn_cfg = ConnectorConfig::audio_from_transformer_config(transformer);

    // The V2 selection must have been *detected* off the checkpoint, not assumed: this is the value
    // `require_v2_version` is fed, and a V1 answer here would mean the 2.5 path was running V2 math
    // against a V1-shaped config.
    assert_eq!(
        av_cfg.caption_feature_version,
        CaptionFeatureVersion::V2,
        "the 2.5 checkpoint's own config must resolve to the V2 caption feature extractor"
    );

    (checkpoint, te_path, root, av_cfg, conn_cfg, audio_conn_cfg)
}

fn encoder(dir: &std::path::Path, device: &Device) -> (Ltx25TextEncoder, Ltx25Tokenizer) {
    let (checkpoint, te_path, root, av_cfg, conn_cfg, audio_conn_cfg) = tier_inputs(dir, device);
    let te = Ltx25TextEncoder::from_packed_av(
        &checkpoint,
        &te_path,
        root.clone(),
        root,
        &av_cfg,
        &conn_cfg,
        &audio_conn_cfg,
    )
    .expect("build the LTX-2.5 text encoder");

    let tok = Ltx25Tokenizer::from_packed_te_file(&te_path).expect("packed tokenizer");
    (te, tok)
}

#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 tiers (LTX25_TIER_DIR) and a CUDA GPU"]
fn connector_inputs_match_the_reference_geometry_and_are_well_formed() {
    let device = Device::new_cuda(0).expect("cuda device");
    let dir = tier_dir();
    let (video_dim, audio_dim) = golden_connector_dims(&device);
    let (te, tok) = encoder(&dir, &device);

    let (input_ids, mask01) = tok.encode(PROMPT, MAX_LEN, &device).expect("tokenize");
    let (vf, af, ve, ae) = te
        .encode_both_with_features(&input_ids, &mask01)
        .expect("encode_both_with_features");

    assert_eq!(
        vf.dims3().expect("rank 3"),
        (1, MAX_LEN, video_dim),
        "video connector input geometry"
    );
    assert_eq!(
        af.dims3().expect("rank 3"),
        (1, MAX_LEN, audio_dim),
        "audio connector input geometry"
    );
    finite("video_features", &vf);
    finite("audio_features", &af);

    assert_eq!(ve.dims3().expect("rank 3"), (1, MAX_LEN, video_dim));
    assert_eq!(ae.dims3().expect("rank 3"), (1, MAX_LEN, audio_dim));
    finite("video_embeddings", &ve);
    finite("audio_embeddings", &ae);

    eprintln!(
        "ltx_2_5 connector inputs (candle): video {:?} audio {:?}",
        vf.dims3().unwrap(),
        af.dims3().unwrap()
    );
}

/// `max|got - want| / max|want|` — the peak-relative error the 2.3 TE sibling (`te_parity.rs`)
/// asserts on, and the same bars are used below.
fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let g = got
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("host copy");
    let w = want
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("host copy");
    assert_eq!(g.len(), w.len(), "peak_rel over mismatched shapes");
    let denom = w.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-12);
    g.iter()
        .zip(&w)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()))
        / denom
}

/// [`peak_rel`] restricted to token rows `[lo, hi)`, with the denominator taken over the same
/// slice.
///
/// The global peak denominator is `max|want|` over the *whole* tensor, which the register / pad
/// region can dominate; a mismatch confined to the valid rows would then be divided by a number
/// that has nothing to do with it. This asks the question again where it matters.
fn peak_rel_rows(got: &Tensor, want: &Tensor, lo: usize, hi: usize) -> f32 {
    let g = got.narrow(1, lo, hi - lo).expect("slice got");
    let w = want.narrow(1, lo, hi - lo).expect("slice want");
    peak_rel(&g, &w)
}

/// The four tensors both numeric tests compare, run once off the golden's own tokenization.
///
/// The **bf16 (dense) tier**, not `q8`: the oracle is an f32 forward over the unquantized upstream
/// checkpoint, so the port must be given dense bf16 weights or the tier's own quantization error
/// is folded into a text-encoder correctness gate. Same choice the MLX twin makes.
fn against_the_golden(device: &Device) -> (HashMap<String, Tensor>, usize, [Tensor; 4]) {
    let root = std::env::var("LTX25_TIER_DIR").expect("LTX25_TIER_DIR");
    let dir = std::path::PathBuf::from(root).join("bf16");
    let g = safetensors::load(GOLDEN_25, device).expect("the LTX-2.5 connector-input golden");
    let input_ids = g["input_ids"]
        .to_dtype(DType::U32)
        .expect("input_ids as u32");
    assert_eq!(
        input_ids.dims2().expect("rank 2"),
        (1, MAX_LEN),
        "the golden's tokenization must be the geometry this file runs at"
    );
    // Feed the golden's own ids/mask rather than re-tokenizing: the oracle and the port must be
    // answering the same question, and a tokenizer drift would otherwise read as a numeric one.
    let mask01: Vec<u32> = g["mask01"]
        .to_dtype(DType::U32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<u32>())
        .expect("mask01");
    let nv = mask01.iter().filter(|&&m| m != 0).count();
    assert!(
        nv > 0 && nv < MAX_LEN,
        "the golden must carry a real pad run, got {nv} valid of {MAX_LEN}"
    );

    let (te, _) = encoder(&dir, device);
    let (vf, af, ve, ae) = te
        .encode_both_with_features(&input_ids, &mask01)
        .expect("encode_both_with_features");
    (g, nv, [vf, af, ve, ae])
}

/// **The numeric gate on the text encoder**, asserted in exactly the terms the MLX twin asserts,
/// against exactly the same file: `video_features` / `audio_features`, the connector *inputs*,
/// which is what this story is about.
///
/// The bar is the 2.3 TE sibling's `1.5e-2` peak-relative (`te_parity.rs`) — a bf16 port against
/// an f32 reference. Asserted twice, globally and over the valid-token rows alone (the suffix,
/// since the features keep the tokenizer's left-padded order), so a mismatch confined to the
/// prompt cannot hide behind a peak denominator set by the pad region.
///
/// The MLX twin measured `2.282e-3` / `1.432e-3` here; this backend's number is CUDA-lane
/// evidence.
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 bf16 tier (LTX25_TIER_DIR) and a CUDA GPU"]
fn connector_inputs_match_the_2_5_reference_golden() {
    let device = Device::new_cuda(0).expect("cuda device");
    let (g, nv, [vf, af, _, _]) = against_the_golden(&device);
    let (want_vf, want_af) = (&g["video_features"], &g["audio_features"]);

    let (pr_vf, pr_af) = (peak_rel(&vf, want_vf), peak_rel(&af, want_af));
    let lo = MAX_LEN - nv;
    let v_vf = peak_rel_rows(&vf, want_vf, lo, MAX_LEN);
    let v_af = peak_rel_rows(&af, want_af, lo, MAX_LEN);
    eprintln!(
        "ltx_2_5 connector INPUTS vs reference (candle): video_features {pr_vf:.3e} (valid \
         {v_vf:.3e}) audio_features {pr_af:.3e} (valid {v_af:.3e})"
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

/// The same gate one stage further on — and **it is RED on MLX today**, on a defect this golden
/// found. See the MLX twin's `connector_outputs_match_the_2_5_reference_golden` for the full
/// analysis; in short, `video_embeddings 1.275e0` / `audio_embeddings 1.771e0` against a `6e-2`
/// bar on the dense bf16 tier, while the connector *inputs* on the same run reproduce to
/// `2.282e-3`. The defect is in the connector — shared with LTX-2.3 and pinned only against
/// `mlx_video`'s `Embeddings1DConnector`, never against `ltx_core`'s — not in this adapter.
///
/// Kept as a real assertion rather than a comment so the day it is fixed, it turns green on its
/// own, on both backends.
#[test]
#[ignore = "sc-18770: RED — records a connector defect the 2.5 golden surfaced; see the doc \
            comment. Needs the bf16 tier (LTX25_TIER_DIR) and a CUDA GPU."]
fn connector_outputs_match_the_2_5_reference_golden() {
    let device = Device::new_cuda(0).expect("cuda device");
    let (g, nv, [_, _, ve, ae]) = against_the_golden(&device);
    let (want_ve, want_ae) = (&g["video_embeddings"], &g["audio_embeddings"]);

    let (pr_ve, pr_ae) = (peak_rel(&ve, want_ve), peak_rel(&ae, want_ae));
    // The connector reorders its input, so here the valid rows are the PREFIX.
    let v_ve = peak_rel_rows(&ve, want_ve, 0, nv);
    let v_ae = peak_rel_rows(&ae, want_ae, 0, nv);
    eprintln!(
        "ltx_2_5 connector OUTPUTS vs reference (candle): video_emb {pr_ve:.3e} (valid \
         {v_ve:.3e}) audio_emb {pr_ae:.3e} (valid {v_ae:.3e})"
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

/// The video-only constructor is reachable and genuinely leaves the audio head absent — the candle
/// twin of the MLX file's `the_video_only_constructor_omits_the_audio_head`.
///
/// Before the `from_packed_video` split, `audio` was `Some` on every constructed encoder and both
/// `ok_or_else` arms in `encode_both*` were dead code.
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 tiers (LTX25_TIER_DIR) and a CUDA GPU"]
fn the_video_only_constructor_omits_the_audio_head() {
    let device = Device::new_cuda(0).expect("cuda device");
    let dir = tier_dir();
    let (video_dim, _) = golden_connector_dims(&device);
    let (checkpoint, te_path, root, av_cfg, conn_cfg, _) = tier_inputs(&dir, &device);

    let te = Ltx25TextEncoder::from_packed_video(
        &checkpoint,
        &te_path,
        root.clone(),
        root,
        &av_cfg,
        &conn_cfg,
    )
    .expect("build the video-only LTX-2.5 text encoder");
    let tok = Ltx25Tokenizer::from_packed_te_file(&te_path).expect("packed tokenizer");

    let (input_ids, mask01) = tok.encode(PROMPT, MAX_LEN, &device).expect("tokenize");
    let (vf, ve) = te
        .encode_with_features(&input_ids, &mask01)
        .expect("the video path must work without an audio head");
    assert_eq!(vf.dims3().expect("rank 3"), (1, MAX_LEN, video_dim));
    assert_eq!(ve.dims3().expect("rank 3"), (1, MAX_LEN, video_dim));
    finite("video_features", &vf);
    finite("video_embeddings", &ve);

    let err = match te.encode_both_with_features(&input_ids, &mask01) {
        Ok(_) => panic!("the AV path must refuse a video-only encoder"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("from_packed_video"),
        "the refusal must name the constructor that produced this encoder: {err}"
    );
}

/// The mask half of the contract, asserted in the same terms as the MLX twin.
///
/// The extractor zeroes the padded rows of `normed`, but the aggregate projection then adds its
/// bias, so a pad row equals the bias — not zero. The property that actually holds is that every
/// pad row is identical to every other: whatever constant the bias contributes, no pad position may
/// vary with the prompt.
#[test]
#[ignore = "sc-18770: needs the built LTX-2.5 tiers (LTX25_TIER_DIR) and a CUDA GPU"]
fn padded_positions_carry_no_token_dependent_conditioning() {
    let device = Device::new_cuda(0).expect("cuda device");
    let dir = tier_dir();
    let (te, tok) = encoder(&dir, &device);

    let (input_ids, mask01) = tok.encode(PROMPT, MAX_LEN, &device).expect("tokenize");
    let pads = mask01.iter().filter(|&&m| m == 0).count();
    assert!(
        pads >= 2,
        "the prompt must be short enough to leave at least two padded positions, got {pads}"
    );
    assert!(
        mask01[..pads].iter().all(|&m| m == 0),
        "the tokenizer must left-pad, so the pad run is the prefix"
    );

    let (vf, _, _, _) = te
        .encode_both_with_features(&input_ids, &mask01)
        .expect("encode_both_with_features");
    let dim = vf.dims3().expect("rank 3").2;
    let v = vf
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("host copy");
    let row = |t: usize| &v[t * dim..(t + 1) * dim];

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
