//! S1 connector golden parity vs the reference `Embeddings1DConnector` (sc-18763) — the candle
//! twin of `mlx-gen-ltx`'s `tests/connector_parity.rs`.
//!
//! `#[ignore]`d: needs the real eros connector weights, in the candle CUDA tier layout under
//! `LTX_BASE_DIR`, plus a CUDA GPU — candle's plain CPU backend has no bf16 matmul, the same
//! reason `tests/conformance.rs` is `cuda`-only.
//!
//! The golden holds `features` / `mask01` / `video_embeddings` plus the audio equivalents. It is
//! the SAME committed fixture `mlx-gen-ltx`'s own `connector_parity.rs` consumes, dumped once from
//! the PyTorch reference and checked into `mlx-gen-ltx/tests/fixtures/`. `tests/vae_encode_parity.rs`
//! establishes this cross-backend golden-reuse convention already; it applies here too because the
//! connector's math is identical on both backends given the same feature-extractor output. This is
//! the acceptance-criterion golden-parity gate for the connector input (post-projection,
//! post-norm) against the reference, on the candle side.
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
    let pr = peak_rel(&got, want)?;
    eprintln!("candle video connector peak_rel = {pr:.3e}");
    // bf16 (candle always runs the connector at bf16, vs the mlx gate's f32 build) — looser than
    // the mlx `connector_parity.rs` 5e-3, matching `te_parity.rs`'s bf16-through-connector budget.
    assert!(pr < 6e-2, "video connector peak_rel {pr:.3e} too high");
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
    eprintln!("candle audio connector peak_rel = {pr:.3e}");
    assert!(pr < 6e-2, "audio connector peak_rel {pr:.3e} too high");
    Ok(())
}
