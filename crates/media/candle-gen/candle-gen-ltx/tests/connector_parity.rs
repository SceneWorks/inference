//! S1 connector golden parity vs the reference `Embeddings1DConnector` (sc-18763; re-derived
//! sc-21663) — the candle twin of `mlx-gen-ltx`'s `tests/connector_parity.rs`.
//!
//! `#[ignore]`d: needs the real 2.3 connector weights, in the candle CUDA tier layout under
//! `LTX_BASE_DIR`, plus a CUDA GPU — candle's plain CPU backend has no bf16 matmul, the same
//! reason `tests/conformance.rs` is `cuda`-only.
//!
//! The golden holds `features` / `mask01` / `video_embeddings` plus the audio equivalents. It is
//! the SAME committed fixture `mlx-gen-ltx`'s own `connector_parity.rs` consumes. Since sc-21663
//! its oracle is **ltx_core semantics executed via the patched mlx_video module in f32 on MLX**
//! (see `tools/dump_ltx_connector_golden.py` for why the oracle is same-framework with the mlx
//! port, and the torch cross-check it ships). The two backends are NOT numerically identical
//! against it: the mlx port compares f32-vs-f32 on near-identical Metal kernels, while this crate
//! runs **bf16 activations** (f32 attention) on CUDA kernels — so its bars must budget both the
//! bf16-activation penalty and an unmeasured CUDA-vs-Metal kernel delta.
//!
//! # Bars (sc-21663) — provisional derived values, to be replaced by CUDA measurements
//!
//! No CUDA hardware was reachable when the golden was re-derived, so these bars are DERIVED
//! bounds, not measured floors. Components (all measured on the mlx side, same fixture):
//!
//! * f32 kernel-pair floor vs this oracle: video `2.756e-3` global; audio `1.639e-2` valid /
//!   `8.086e-2` global (register rows — the closing per-row RMS-norm rescales rows spanning a
//!   272x norm range, converting kernel noise into relative error on near-cancelled register
//!   rows; see the mlx twin's decomposition).
//! * bf16-activation penalty, measured on the LTX-2.5 video golden: global `5.568e-2` bf16 vs
//!   `1.288e-2` f32 (isolated connector); valid rows stayed under `7e-3` even at bf16.
//!
//! Bars = floor ⊕ bf16 penalty, ×~2 margin for the unmeasured CUDA kernel pair: video valid
//! `3e-2` / global `1.5e-1`; audio valid `5e-2` / global `2.5e-1`. **The first CUDA run must
//! record the printed peak_rels on the story/epic and tighten these to measured-floor × ~1.5** —
//! and it is also the decision point for mirroring the mlx side's f32 activations (see
//! `src/connector.rs`'s bf16-deficiency note).
//!
//! Run:
//! `LTX_BASE_DIR=<snapshot>/q8 cargo test -p candle-gen-ltx --features cuda --test
//! connector_parity -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{safetensors, DType, Device, Result as CoreResult, Tensor};
use candle_gen_ltx::config::ConnectorConfig;
use candle_gen_ltx::connector::Connector;
use candle_gen_ltx::tier::TierPaths;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors"
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

/// [`peak_rel`] over token rows `[lo, hi)` of a `[1, seq, dim]` pair, denominator over the same
/// slice — the valid-row bar must not hide behind (or be drowned by) the register rows.
fn peak_rel_rows(got: &Tensor, want: &Tensor, lo: usize, hi: usize) -> CoreResult<f32> {
    peak_rel(&got.narrow(1, lo, hi - lo)?, &want.narrow(1, lo, hi - lo)?)
}

/// The golden's `mask01` (stored int) → the connector's `nv` (valid, non-padding token count).
fn nv_from_mask01(mask01: &Tensor) -> CoreResult<usize> {
    let mask01 = mask01
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    Ok(mask01.iter().filter(|&&m| m != 0).count())
}

/// Open the eros connector weights via the candle CUDA-tier loader (the production path,
/// `TierPaths::connector_vb` — the same remapper `lib.rs`'s `new_av` call site uses), rooted the
/// same way: `model.diffusion_model.` (the connector's `video_embeddings_connector.*` /
/// `audio_embeddings_connector.*` prefixes sit under that).
///
/// Returns `candle_gen::Result` (not `candle_core::Result`/`CoreResult`) because
/// `TierPaths::connector_vb` itself returns `candle_gen::Result` (`CandleError`, the crate's own
/// rich error type) — `CandleError: From<candle_core::Error>` lets `?` widen a `candle_core`
/// error INTO `candle_gen::Result` here, but not the reverse (that direction would need
/// `From<CandleError> for candle_core::Error`, which the orphan rule forbids since neither type is
/// local to this crate). Matches `tests/vae_encode_parity.rs`'s exact pattern.
fn connector_root(
    device: &Device,
) -> candle_gen::Result<candle_gen::candle_nn::VarBuilder<'static>> {
    let base = std::env::var("LTX_BASE_DIR").expect("set LTX_BASE_DIR to the q4/q8 tier directory");
    let paths = TierPaths::detect(std::path::Path::new(&base), None)
        .expect("LTX_BASE_DIR must contain quantize_config.json and transformer.safetensors");
    Ok(paths
        .connector_vb(DType::BF16, device)?
        .pp("model.diffusion_model"))
}

#[test]
#[ignore = "needs real eros connector weights (candle CUDA tier) and a CUDA GPU"]
fn video_connector_matches_reference() -> candle_gen::Result<()> {
    let device = Device::new_cuda(0)?;
    let root = connector_root(&device)?;
    let conn = Connector::new(root, &ConnectorConfig::ltx_2_3())?;

    let golden = safetensors::load(GOLDEN, &device)?;
    let features = golden["features"].to_dtype(DType::F32)?;
    let mask01 = &golden["mask01"];
    let want = &golden["video_embeddings"];
    let nv = nv_from_mask01(mask01)?;

    let got = conn.forward(&features, nv)?;
    assert_eq!(got.shape(), want.shape());
    // The connector reorders its input, so the valid rows are the PREFIX (`0..nv`).
    let pr = peak_rel(&got, want)?;
    let pr_valid = peak_rel_rows(&got, want, 0, nv)?;
    eprintln!("candle video connector peak_rel = {pr:.3e} (valid rows {pr_valid:.3e})");
    // Provisional derived bars (sc-21663) — see the module docs; the first CUDA run records the
    // printed values and tightens these.
    assert!(
        pr_valid < 3e-2,
        "video connector valid-row peak_rel {pr_valid:.3e} too high"
    );
    assert!(pr < 1.5e-1, "video connector peak_rel {pr:.3e} too high");
    Ok(())
}

#[test]
#[ignore = "needs real eros connector weights (candle CUDA tier) and a CUDA GPU"]
fn audio_connector_matches_reference() -> candle_gen::Result<()> {
    // sc-2684 / sc-5495: the audio connector is the same architecture at audio dims (32×64=2048).
    let device = Device::new_cuda(0)?;
    let root = connector_root(&device)?;
    let conn = Connector::new_with_prefix(
        root,
        &ConnectorConfig::ltx_2_3_audio(),
        "audio_embeddings_connector",
    )?;

    let golden = safetensors::load(GOLDEN, &device)?;
    let features = golden["audio_features"].to_dtype(DType::F32)?;
    let mask01 = &golden["mask01"];
    let want = &golden["audio_embeddings"];
    let nv = nv_from_mask01(mask01)?;

    let got = conn.forward(&features, nv)?;
    assert_eq!(got.shape(), want.shape());
    let pr = peak_rel(&got, want)?;
    let pr_valid = peak_rel_rows(&got, want, 0, nv)?;
    eprintln!("candle audio connector peak_rel = {pr:.3e} (valid rows {pr_valid:.3e})");
    // Provisional derived bars (sc-21663) — see the module docs; the first CUDA run records the
    // printed values and tightens these.
    assert!(
        pr_valid < 5e-2,
        "audio connector valid-row peak_rel {pr_valid:.3e} too high"
    );
    assert!(pr < 2.5e-1, "audio connector peak_rel {pr:.3e} too high");
    Ok(())
}
