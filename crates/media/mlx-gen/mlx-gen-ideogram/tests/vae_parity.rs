//! sc-5987 — real-weight parity for the Ideogram 4 VAE decode. Ideogram's `AutoencoderKLFlux2`
//! weights load into the reused `mlx-gen-flux2::Flux2Vae`; this checks the decode matches the
//! diffusers `AutoencoderKLFlux2` reference on the same latent.
//!
//! `#[ignore]` — needs the converted snapshot + the golden (`tools/dump_ideogram4_vae_golden.py`):
//!   CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-ideogram --test integration vae_parity:: -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::testing::{cosine_f32_device, env_path_or_home};
use mlx_gen::weights::Weights;
use mlx_gen_ideogram::load_vae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ideogram4_vae.safetensors"
);

fn snapshot_dir() -> PathBuf {
    env_path_or_home("IDEOGRAM4_MLX", ".cache/ideogram4-mlx-convert")
}

#[test]
#[ignore = "needs converted weights + generated golden"]
fn vae_decode_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("golden — run tools/dump_ideogram4_vae_golden.py");
    let vae = load_vae(&snapshot_dir()).expect("load converted VAE");
    let out = vae.decode(g.require("z").unwrap()).unwrap();
    let want = g.require("golden").unwrap();
    assert_eq!(out.shape(), want.shape(), "decoded image shape (NHWC)");

    let c = cosine_f32_device(&out, want);
    println!("Ideogram 4 VAE decode parity cosine = {c:.7}");
    assert!(c > 0.999, "VAE decode parity cosine {c} too low");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED vae_parity::vae_decode_matches_reference");
}
