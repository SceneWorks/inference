//! Full public-pipeline parity gate for sc-14041.
//!
//! The committed bundle defaults to a fast 256²/4-step capture. Re-dump and run at the story
//! geometry with `MAGE_DEVICE=cpu MAGE_H=1024 MAGE_W=1024 MAGE_STEPS=20` for the release witness.

mod common;

use common::{error, require_golden};
use mlx_gen_mage::{GsKey, MageFlowPipeline};

const E2E_GOLDEN: &str = "mage_flow_e2e_golden.safetensors";
const PROMPT: &str = "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug";

#[test]
#[ignore = "needs MAGE_SNAPSHOT, Metal, and mage_flow_e2e_golden.safetensors"]
fn public_pipeline_matches_the_torch_render() {
    let root = std::env::var("MAGE_SNAPSHOT").expect("set MAGE_SNAPSHOT to microsoft/Mage-Flow");
    let golden = require_golden(E2E_GOLDEN);
    let geometry = golden.require("geometry").unwrap().as_slice::<i32>();
    let (height, width, steps) = (geometry[0] as u32, geometry[1] as u32, geometry[2] as usize);
    let cfg = golden.require("cfg").unwrap().as_slice::<f32>()[0];
    let seed = golden.require("seed").unwrap().as_slice::<i64>()[0];
    let key = GsKey::from_u64(golden.require("gs_key").unwrap().as_slice::<i64>()[0] as u64);

    let pipeline = MageFlowPipeline::load(root).unwrap();
    let got = pipeline
        .generate(PROMPT, " ", height, width, steps, cfg, seed, &key, false)
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap()
        .reshape(&[height as i32, width as i32, 3])
        .unwrap();
    mlx_rs::transforms::eval([&got]).unwrap();
    let want = golden.require("image_u8").unwrap();
    let (max_abs, _peak_rel, mean_rel) = error(&got, want);
    println!("Mage-Flow public pipeline: max_abs={max_abs:.3}, mean_rel={mean_rel:.5}");
    assert!(max_abs <= 64.0, "render max pixel error {max_abs}");
    assert!(mean_rel <= 0.08, "render mean-relative error {mean_rel}");
}
