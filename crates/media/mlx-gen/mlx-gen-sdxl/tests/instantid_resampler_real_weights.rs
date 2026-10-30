//! sc-3110: InstantID face Resampler parity vs torch (f32).
//!
//! `#[ignore]`d — needs the golden from `tools/dump_instantid_resampler_golden.py` (which bundles the
//! f32 `image_proj.*` weights from `InstantX/InstantID` `ip-adapter.bin`, so no separate weights file
//! is required). Run with:
//!   cargo test -p mlx-gen-sdxl --release --test integration instantid_resampler_real_weights:: -- --ignored --nocapture
//!
//! InstantID's `image_proj_model` is the *same* Tencent `Resampler` already ported for the SDXL
//! IP-Adapter (sc-3059); this only validates it under `ResamplerConfig::instantid_face()`
//! (embedding_dim=512) on a fixed 512-d ArcFace embedding → 16 face tokens × 2048.

use mlx_gen::testing::peak_rel_host;
use mlx_gen::weights::Weights;
use mlx_gen_sdxl::ip_adapter::{Resampler, ResamplerConfig};
use mlx_rs::Dtype;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/instantid_resampler_golden.safetensors"
);

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "needs the instantid_resampler golden (tools/dump_instantid_resampler_golden.py)"]
fn instantid_face_resampler_matches_torch() {
    let mut g = Weights::from_file(GOLDEN).unwrap_or_else(|e| panic!("load {GOLDEN:?}: {e}"));
    g.cast_all(Dtype::Float32).unwrap();

    let embed = g.require("arcface_embed").unwrap(); // [1, 1, 512]
    let tok_golden = g.require("face_tokens").unwrap(); // [1, 16, 2048]
    assert_eq!(embed.shape(), [1, 1, 512], "arcface_embed shape");
    assert_eq!(tok_golden.shape(), [1, 16, 2048], "face_tokens shape");

    // The golden bundles the `image_proj.*` weights, so this is the same source the dump ran on.
    let resampler =
        Resampler::from_weights(&g, "image_proj", &ResamplerConfig::instantid_face()).unwrap();

    let tokens = (
        crate::common::record_model_inference_device(),
        resampler.forward(embed),
    )
        .1
        .unwrap();
    assert_eq!(tokens.shape(), [1, 16, 2048], "output token shape");

    let rel = peak_rel_host(&tokens, tok_golden, None);
    println!("[instantid resampler] face tokens peak_rel = {rel:.3e}");

    // f32 vs f32 cross-backend (torch CPU vs MLX Metal). `norm_out` renormalizes the output, so this
    // lands bit-close — the SDXL IP-Adapter Resampler hits 4.9e-4 on the same architecture; gate at
    // 1e-3 (a wrong port — e.g. a misordered residual or a transposed fused split — diverges orders
    // of magnitude past this).
    assert!(rel < 1e-3, "InstantID Resampler diverged: {rel:.3e}");

    println!("\nMODEL_INFERENCE_BODY_COMPLETED instantid_resampler_real_weights::instantid_face_resampler_matches_torch");
}
