//! S1 full text-encoder parity vs the reference (sc-2679 S1) — end-to-end.
//!
//! `#[ignore]`d: needs the `gemma-3-12b-it-bf16` shards (~24 GB) + a converted BASE
//! `ltx_2_3_base_q8/connector.safetensors` + the golden from `tools/dump_ltx_te_golden.py`
//! (gitignored). Runs the Rust `LtxTextEncoder` in bf16 (Gemma → feature extractor → connector)
//! and checks `video_features` + `video_embeddings` reproduce the reference.
//!
//! Run: `cargo test -p mlx-gen-ltx --test integration te_parity:: -- --ignored --nocapture`

use mlx_gen::testing::peak_rel_native;
use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::gemma::GemmaConfig;
use mlx_gen_ltx::text_encoder::LtxTextEncoder;
use mlx_gen_ltx::transformer::Precision;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ltx_te_golden.safetensors"
);

fn gemma_dir() -> std::path::PathBuf {
    let d = std::env::var("LTX_GEMMA_DIR").unwrap_or_else(|_| panic!("set LTX_GEMMA_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    d.into()
}

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

#[test]
#[ignore = "needs gemma-3-12b-it-bf16 (~24 GB) + ltx_2_3_base_q8 + tools/golden/ltx_te_golden.safetensors"]
fn full_text_encoder_matches_reference() {
    let base = base_dir();
    let cfg = LtxConfig::from_model_dir(&base).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&base).expect("split_model.json");
    let gemma_w = Weights::from_dir(gemma_dir()).expect("gemma shards");
    let conn_w =
        Weights::from_file(base.join("connector.safetensors")).expect("connector.safetensors");
    let te = LtxTextEncoder::from_weights(
        &gemma_w,
        &conn_w,
        GemmaConfig::gemma_3_12b(),
        None, // dense bf16 Gemma (the default snapshot)
        &cfg,
        // bf16 activations at the checkpoint's declared quant geometry. The 2.3 connector ships
        // dense, so every Linear here takes the dense arm regardless of the bits/group carried.
        Precision::quant_bf16(split.bits, split.group),
    )
    .expect("build TE");

    let g = Weights::from_file(GOLDEN).expect("te golden");
    let input_ids = g.require("input_ids").unwrap();
    let attention_mask = g.require("attention_mask").unwrap();

    let (feats, emb) = te
        .encode_with_features(input_ids, attention_mask)
        .expect("encode");

    let pr_feats = peak_rel_native(&feats, g.require("video_features").unwrap(), Some(1e-12));
    let pr_emb = peak_rel_native(&emb, g.require("video_embeddings").unwrap(), Some(1e-12));
    eprintln!("TE: video_features peak_rel {pr_feats:.3e}  video_embeddings peak_rel {pr_emb:.3e}");
    // `video_features` (after the per-token-RMS feature extractor) gates feat-extract + Gemma
    // tightly — the RMS-norm contracts the Gemma bf16 cross-build delta (~9e-3) rather than
    // amplifying it.
    assert!(
        pr_feats < 1.5e-2,
        "video_features peak_rel {pr_feats:.3e} too high"
    );
    // `video_embeddings` is looser: the 8-layer connector amplifies that bf16 cross-build delta
    // (pmetal mlx 0.31.1 vs the reference's wheel 0.31.0) ~5×. NOT a bug — the connector is proven
    // correct (bit-exact in f32 via connector_parity; f32-vs-bf16-except-SDPA agree to ~3e-3 on
    // identical input). The real end-to-end gate is the S6 video px>8.
    assert!(
        pr_emb < 6e-2,
        "video_embeddings peak_rel {pr_emb:.3e} too high"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED te_parity::full_text_encoder_matches_reference");
}

#[test]
#[ignore = "needs gemma-3-12b-it-bf16 (~24 GB) + ltx_2_3_base_q8 + tools/golden/ltx_te_golden.safetensors"]
fn full_text_encoder_av_matches_reference() {
    let base = base_dir();
    let cfg = LtxConfig::from_model_dir(&base).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&base).expect("split_model.json");
    let gemma_w = Weights::from_dir(gemma_dir()).expect("gemma shards");
    let conn_w =
        Weights::from_file(base.join("connector.safetensors")).expect("connector.safetensors");
    let te = LtxTextEncoder::from_weights_av(
        &gemma_w,
        &conn_w,
        GemmaConfig::gemma_3_12b(),
        None, // dense bf16 Gemma (the default snapshot)
        &cfg,
        // bf16 activations at the checkpoint's declared quant geometry. The 2.3 connector ships
        // dense, so every Linear here takes the dense arm regardless of the bits/group carried.
        Precision::quant_bf16(split.bits, split.group),
    )
    .expect("build AV TE");

    let g = Weights::from_file(GOLDEN).expect("te golden");
    let input_ids = g.require("input_ids").unwrap();
    let attention_mask = g.require("attention_mask").unwrap();

    let (vf, af, ve, ae) = te
        .encode_av_with_features(input_ids, attention_mask)
        .expect("encode_av");

    // Video half identical to the video-only gate (shared normed_hidden).
    let pr_vf = peak_rel_native(&vf, g.require("video_features").unwrap(), Some(1e-12));
    let pr_ve = peak_rel_native(&ve, g.require("video_embeddings").unwrap(), Some(1e-12));
    // Audio half: same per-token-RMS feature path → tight on features, looser through the connector.
    let pr_af = peak_rel_native(&af, g.require("audio_features").unwrap(), Some(1e-12));
    let pr_ae = peak_rel_native(&ae, g.require("audio_embeddings").unwrap(), Some(1e-12));
    eprintln!(
        "TE/AV: video_features {pr_vf:.3e} video_emb {pr_ve:.3e} | audio_features {pr_af:.3e} audio_emb {pr_ae:.3e}"
    );
    assert!(pr_vf < 1.5e-2, "video_features peak_rel {pr_vf:.3e}");
    assert!(pr_ve < 6e-2, "video_embeddings peak_rel {pr_ve:.3e}");
    assert!(pr_af < 1.5e-2, "audio_features peak_rel {pr_af:.3e}");
    assert!(pr_ae < 6e-2, "audio_embeddings peak_rel {pr_ae:.3e}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED te_parity::full_text_encoder_av_matches_reference");
}
