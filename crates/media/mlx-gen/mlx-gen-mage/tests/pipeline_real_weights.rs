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
    let trace = pipeline
        .generate_trace(
            PROMPT,
            " ",
            height,
            width,
            steps,
            cfg,
            seed,
            &key,
            false,
            &mut |_| {},
        )
        .unwrap();
    let got = &trace.image_u8;
    mlx_rs::transforms::eval([
        got,
        &trace.final_tokens,
        &trace.final_latent,
        &trace.trajectories[0],
        &trace.trajectories[1],
    ])
    .unwrap();

    // These boundaries precede the chaos-amplifying VAE/image conversion. The port runs bf16
    // activations while the oracle captures torch bf16→f32, so 2% mean-relative / 0.5 peak is a
    // deliberately tight bf16 accumulation bound across four Euler steps, not a pixel heuristic.
    for (label, actual, expected) in [
        (
            "traj_step0",
            &trace.trajectories[0],
            golden.require("traj_step0").unwrap(),
        ),
        (
            "traj_step1",
            &trace.trajectories[1],
            golden.require("traj_step1").unwrap(),
        ),
        (
            "final_tokens",
            &trace.final_tokens,
            golden.require("final_tokens").unwrap(),
        ),
        (
            "final_latent",
            &trace.final_latent,
            golden.require("final_latent").unwrap(),
        ),
    ] {
        let (max_abs, _, mean_rel) = error(actual, expected);
        println!("{label}: max_abs={max_abs:.5}, mean_rel={mean_rel:.6}");
        assert!(max_abs <= 0.5, "{label} peak error {max_abs}");
        assert!(mean_rel <= 0.02, "{label} mean-relative error {mean_rel}");
    }
    let want = golden.require("image_u8").unwrap();
    let (max_abs, _peak_rel, mean_rel) = error(got, want);
    println!("Mage-Flow public pipeline: max_abs={max_abs:.3}, mean_rel={mean_rel:.5}");
    assert!(max_abs <= 64.0, "render max pixel error {max_abs}");
    assert!(mean_rel <= 0.08, "render mean-relative error {mean_rel}");
}
