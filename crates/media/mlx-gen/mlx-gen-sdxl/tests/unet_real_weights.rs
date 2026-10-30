//! sc-2400 S3: SDXL U-Net single-forward parity vs the vendored Apple reference (f32).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from `tools/dump_sdxl_unet_golden.py`.
//! Run with:
//!   cargo test -p mlx-gen-sdxl --release --test integration unet_real_weights:: -- --ignored --nocapture
//!
//! Feeds the golden's exact inputs (latents, timestep, dual-CLIP conditioning, pooled, the hardcoded
//! `[512,512,0,0,512,512]` micro-conditioning) and checks the predicted eps. Reference + Rust both
//! f32, so the tolerance is tight — this isolates the entire U-Net forward (down/mid/up, cross-attn,
//! the time + text_time embeddings, the skip connections).

use crate::common;

use mlx_gen::testing::peak_rel_host;
use mlx_gen::weights::Weights;
use mlx_gen_sdxl::load_unet;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_unet_golden.safetensors"
);

use common::snapshot;

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "needs the real SDXL snapshot + U-Net golden"]
fn unet_single_forward_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let timestep: f32 = g.metadata("timestep").unwrap().parse().unwrap();
    let unet = load_unet(&snapshot()).unwrap();

    let eps = (
        crate::common::record_model_inference_device(),
        unet.forward(
            g.require("latents").unwrap(),
            timestep,
            g.require("conditioning").unwrap(),
            g.require("pooled").unwrap(),
            g.require("time_ids").unwrap(),
        ),
    )
        .1
        .unwrap();

    let golden = g.require("eps").unwrap();
    assert_eq!(eps.shape(), golden.shape(), "eps shape");
    let pr = peak_rel_host(&eps, golden, None);
    let mean_rel = {
        let n = golden.shape().iter().product::<i32>();
        let (a, b) = (eps.reshape(&[n]).unwrap(), golden.reshape(&[n]).unwrap());
        let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        let mabs = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32 / mabs
    };
    println!(
        "unet eps {:?}: peak_rel={pr:.3e} mean_rel={mean_rel:.3e}",
        eps.shape()
    );
    assert!(pr < 5e-3, "U-Net forward diverged: peak_rel {pr:.3e}");
    println!("✓ SDXL U-Net single forward matches the vendored reference (f32)");

    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED unet_real_weights::unet_single_forward_matches_vendored"
    );
}
