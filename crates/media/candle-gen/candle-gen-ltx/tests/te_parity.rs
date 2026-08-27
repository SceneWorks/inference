//! S1 full text-encoder golden parity vs the reference (sc-18763) — the candle twin of
//! `mlx-gen-ltx`'s `tests/te_parity.rs`.
//!
//! Closes the gap `connector_parity.rs` left (sc-18763 coordinator review point 4): that test
//! feeds a REFERENCE `features` tensor into candle's connector and checks the connector's own
//! output — it never verifies candle's OWN Gemma-forward + feature-extractor math actually
//! reproduces the reference `video_features` (the connector INPUT, post-projection, post-norm).
//! This test runs the full `LtxTextEncoder::encode_with_features` / `encode_both_with_features`
//! path and checks both `video_features` (the connector input) and `video_embeddings` (the
//! connector output) against the golden, mirroring mlx's `te_parity.rs` exactly.
//!
//! `#[ignore]`d: needs the Gemma-3-12B shards (~13-24 GB) via the candle CUDA tier under
//! `LTX_BASE_DIR` (or `LTX_GEMMA_DIR` to override the Gemma location), a CUDA GPU, and the golden
//! from `tools/dump_ltx_te_golden.py` — the SAME (gitignored, not committed) golden mlx's own
//! `te_parity.rs` consumes, at the same path. Regenerate it with that dump tool before running
//! either backend's gate.
//!
//! Run:
//! `LTX_BASE_DIR=<snapshot>/q8 cargo test -p candle-gen-ltx --features cuda --test integration --
//! te_parity:: --ignored --nocapture`

#![cfg(feature = "cuda")]

// Test functions return `candle_gen::Result` (not `candle_core::Result`/`CoreResult`), matching
// `tests/vae_encode_parity.rs`: `TierPaths::gemma_vb`/`connector_vb` return `candle_gen::Result`
// (`CandleError`, the crate's rich error type) — `CandleError: From<candle_core::Error>` lets `?`
// widen a `candle_core` error INTO `candle_gen::Result`, never the reverse (that direction would
// need `From<CandleError> for candle_core::Error`, which the orphan rule forbids). A test fn
// declared `candle_core::Result` that calls a `TierPaths` accessor fails to compile with "? could
// not convert the error" — this is exactly that failure mode, fixed here.
use candle_gen::candle_core::{safetensors, DType, Device, Result as CoreResult, Tensor};
use candle_gen_ltx::config::{ConnectorConfig, GemmaConfig};
use candle_gen_ltx::text_encoder::LtxTextEncoder;
use candle_gen_ltx::tier::TierPaths;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/ltx_te_golden.safetensors"
);

fn peak_rel(got: &Tensor, want: &Tensor) -> CoreResult<f32> {
    let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let want = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), want.len());
    let mut diff = 0f32;
    let mut mag = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        diff = diff.max((g - w).abs());
        mag = mag.max(w.abs());
    }
    Ok(diff / mag.max(1e-12))
}

fn tier_paths() -> TierPaths {
    let base = std::env::var("LTX_BASE_DIR").expect("set LTX_BASE_DIR to the q4/q8 tier directory");
    let gemma_override = std::env::var("LTX_GEMMA_DIR")
        .ok()
        .map(std::path::PathBuf::from);
    TierPaths::detect(std::path::Path::new(&base), gemma_override.as_deref())
        .expect("LTX_BASE_DIR must contain quantize_config.json and transformer.safetensors")
}

/// The golden's `attention_mask` (stored int) → candle's `mask01: &[u32]`.
fn mask01_from_golden(attention_mask: &Tensor) -> CoreResult<Vec<u32>> {
    attention_mask
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()
}

#[test]
#[ignore = "needs Gemma-3-12B shards (candle CUDA tier), a CUDA GPU, and tools/golden/ltx_te_golden.safetensors"]
fn full_text_encoder_matches_reference() -> candle_gen::Result<()> {
    let device = Device::new_cuda(0)?;
    let paths = tier_paths();
    let gemma_vb = paths.gemma_vb(DType::BF16, &device)?;
    // In the split candle CUDA tier, `text_embedding_projection.*` sits under
    // `model.diffusion_model.` inside the connector file (unlike the single dense LTX-2.3
    // checkpoint, where it's at that file's bare top level) — `lib.rs`'s own tiered `new_av` call
    // site passes the SAME `.pp("model.diffusion_model")`-rooted builder for both `proj_vb` and
    // `dit_vb`; mirrored here.
    let root = paths
        .connector_vb(DType::BF16, &device)?
        .pp("model.diffusion_model");
    let te = LtxTextEncoder::new(
        gemma_vb,
        root.clone(),
        root,
        &GemmaConfig::gemma_3_12b(),
        &ConnectorConfig::ltx_2_3(),
    )?;

    let golden = safetensors::load(GOLDEN, &device)?;
    let input_ids = golden["input_ids"].to_dtype(DType::U32)?;
    let mask01 = mask01_from_golden(&golden["attention_mask"])?;

    let (feats, emb) = te.encode_with_features(&input_ids, &mask01)?;

    let pr_feats = peak_rel(&feats, &golden["video_features"])?;
    let pr_emb = peak_rel(&emb, &golden["video_embeddings"])?;
    eprintln!(
        "candle TE: video_features peak_rel {pr_feats:.3e}  video_embeddings peak_rel {pr_emb:.3e}"
    );
    // Same tolerance discipline as the mlx gate (`tests/te_parity.rs`): `video_features` (the
    // connector input) is gated tightly, `video_embeddings` looser through the 8-layer connector.
    assert!(
        pr_feats < 1.5e-2,
        "video_features peak_rel {pr_feats:.3e} too high"
    );
    assert!(
        pr_emb < 6e-2,
        "video_embeddings peak_rel {pr_emb:.3e} too high"
    );
    Ok(())
}

#[test]
#[ignore = "needs Gemma-3-12B shards (candle CUDA tier), a CUDA GPU, and tools/golden/ltx_te_golden.safetensors"]
fn full_text_encoder_av_matches_reference() -> candle_gen::Result<()> {
    let device = Device::new_cuda(0)?;
    let paths = tier_paths();
    let gemma_vb = paths.gemma_vb(DType::BF16, &device)?;
    let root = paths
        .connector_vb(DType::BF16, &device)?
        .pp("model.diffusion_model");
    let te = LtxTextEncoder::new_av(
        gemma_vb,
        root.clone(),
        root,
        &GemmaConfig::gemma_3_12b(),
        &ConnectorConfig::ltx_2_3(),
        &ConnectorConfig::ltx_2_3_audio(),
    )?;

    let golden = safetensors::load(GOLDEN, &device)?;
    let input_ids = golden["input_ids"].to_dtype(DType::U32)?;
    let mask01 = mask01_from_golden(&golden["attention_mask"])?;

    let (vf, af, ve, ae) = te.encode_both_with_features(&input_ids, &mask01)?;

    let pr_vf = peak_rel(&vf, &golden["video_features"])?;
    let pr_ve = peak_rel(&ve, &golden["video_embeddings"])?;
    let pr_af = peak_rel(&af, &golden["audio_features"])?;
    let pr_ae = peak_rel(&ae, &golden["audio_embeddings"])?;
    eprintln!(
        "candle TE/AV: video_features {pr_vf:.3e} video_emb {pr_ve:.3e} | audio_features {pr_af:.3e} audio_emb {pr_ae:.3e}"
    );
    assert!(pr_vf < 1.5e-2, "video_features peak_rel {pr_vf:.3e}");
    assert!(pr_ve < 6e-2, "video_embeddings peak_rel {pr_ve:.3e}");
    assert!(pr_af < 1.5e-2, "audio_features peak_rel {pr_af:.3e}");
    assert!(pr_ae < 6e-2, "audio_embeddings peak_rel {pr_ae:.3e}");
    Ok(())
}
