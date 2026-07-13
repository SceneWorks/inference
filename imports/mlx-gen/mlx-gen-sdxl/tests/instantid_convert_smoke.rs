//! sc-3112: InstantID weight-conversion smoke test.
//!
//! `#[ignore]`d — needs the converter output (`tools/convert_instantid.py` →
//! `tools/golden/instantid/ip-adapter.safetensors`) and the InstantID `ControlNetModel` snapshot.
//! Run:
//!   cargo test -p mlx-gen-sdxl --release --test instantid_convert_smoke -- --ignored --nocapture
//!
//! Proves the converted tensors load cleanly with the right shapes into all three consumers
//! (the acceptance for sc-3112), without a deep golden:
//!   1. `image_proj.*` → the face Resampler (`ResamplerConfig::instantid_face()`).
//!   2. `ip_adapter.*` → the 70 decoupled-cross-attn K/V pairs (`load_ip_kv_pairs`), all 2048-in.
//!   3. The IdentityNet `ControlNetModel` → `ControlNet::from_weights(.., &UNetConfig::sdxl_base())`
//!      (loads directly — it is a stock diffusers SDXL ControlNet; no conversion).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::config::UNetConfig;
use mlx_gen_sdxl::{load_ip_kv_pairs, ControlNet, Resampler, ResamplerConfig};
use mlx_rs::Dtype;

const IP_ADAPTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/instantid/ip-adapter.safetensors"
);

/// The InstantID snapshot dir (override with `INSTANTID_SNAPSHOT`).
fn instantid_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("INSTANTID_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--InstantX--InstantID/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for InstantX/InstantID")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs convert_instantid.py output + InstantID ControlNetModel weights"]
fn instantid_converted_weights_load() {
    let mut ipa = Weights::from_file(IP_ADAPTER)
        .unwrap_or_else(|e| panic!("load {IP_ADAPTER:?} (run tools/convert_instantid.py): {e}"));
    ipa.cast_all(Dtype::Float32).unwrap();

    // 1. Resampler (image_proj.*).
    let resampler =
        Resampler::from_weights(&ipa, "image_proj", &ResamplerConfig::instantid_face()).unwrap();
    assert_eq!(resampler.output_dim(), 2048, "Resampler output dim");

    // 2. Decoupled-cross-attn K/V pairs (ip_adapter.*): 70 SDXL cross-attn layers, all 2048-in.
    let kv = load_ip_kv_pairs(&ipa).unwrap();
    assert_eq!(
        kv.len(),
        70,
        "expected 70 ip_adapter K/V pairs, got {}",
        kv.len()
    );
    for (i, (k, v)) in kv.iter().enumerate() {
        assert_eq!(k.shape(), v.shape(), "K/V shape mismatch at pair {i}");
        assert_eq!(
            *k.shape().last().unwrap(),
            2048,
            "K/V pair {i} input dim != cross_attention_dim 2048: {:?}",
            k.shape()
        );
    }

    // 3. IdentityNet — a stock diffusers SDXL ControlNet, loads directly (no conversion).
    let cn_path = instantid_snapshot().join("ControlNetModel/diffusion_pytorch_model.safetensors");
    let mut cn_w = Weights::from_file(&cn_path).unwrap_or_else(|e| panic!("load {cn_path:?}: {e}"));
    cn_w.cast_all(Dtype::Float32).unwrap();
    let _identitynet = ControlNet::from_weights(&cn_w, &UNetConfig::sdxl_base())
        .expect("IdentityNet must load via the stock SDXL ControlNet loader");

    println!(
        "[instantid convert] OK: Resampler + 70 K/V pairs + IdentityNet ControlNet all loaded"
    );
}
