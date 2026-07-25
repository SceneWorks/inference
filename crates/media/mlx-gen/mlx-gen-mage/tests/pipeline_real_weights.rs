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

    // The oracle is torch-bf16 on CPU; the port is MLX-bf16 on Metal. The isolated step-zero DiT
    // gate measures 2.0741% mean-relative between those backends. CFG=5 then feeds that residual
    // back through four nonlinear forwards, so final-token equality is not a valid gate.
    //
    // Mutation measurements against this exact golden (256², 4 steps):
    // faithful:       step1 .008032, final .156555, image .04088 mean-relative
    // swapped CFG:    step1 .112686, final 1.311762, image .80928
    // GS key + 1:     step0 .994330, step1 1.012730, final 1.161042, image .42842
    // pre-add bf16:   step1 .008032, final .155926, image .04053
    //
    // Thus the early trajectory and image mean discriminate semantic mistakes, while the
    // scheduler cast-order mutation is explicitly NOT separable through this cross-backend
    // four-step oracle. Source fidelity for that operation is covered by the unit-level scheduler
    // contract, not by pretending the chaotic final latent supplies a tighter answer.
    let mut failures = Vec::new();
    for (label, actual, expected, max_abs_gate, mean_rel_gate) in [
        (
            "traj_step0",
            &trace.trajectories[0],
            golden.require("traj_step0").unwrap(),
            1.0e-6,
            1.0e-6,
        ),
        (
            "traj_step1",
            &trace.trajectories[1],
            golden.require("traj_step1").unwrap(),
            0.5,
            0.03,
        ),
        (
            "final_tokens",
            &trace.final_tokens,
            golden.require("final_tokens").unwrap(),
            8.0,
            0.30,
        ),
        (
            "final_latent",
            &trace.final_latent,
            golden.require("final_latent").unwrap(),
            8.0,
            0.30,
        ),
    ] {
        let (max_abs, _, mean_rel) = error(actual, expected);
        println!("{label}: max_abs={max_abs:.5}, mean_rel={mean_rel:.6}");
        if max_abs > max_abs_gate || mean_rel > mean_rel_gate {
            failures.push(format!(
                "{label}: max_abs={max_abs:.5} (gate {max_abs_gate}), \
                 mean_rel={mean_rel:.6} (gate {mean_rel_gate})"
            ));
        }
    }
    let want = golden.require("image_u8").unwrap();
    let (max_abs, _peak_rel, mean_rel) = error(got, want);
    println!("Mage-Flow public pipeline: max_abs={max_abs:.3}, mean_rel={mean_rel:.5}");
    // Peak pixel error is not stable under the VAE's nonlinear amplification (faithful 163,
    // wrong cast-order 156). Mean-relative image error does discriminate the semantic controls.
    if mean_rel > 0.10 {
        failures.push(format!(
            "image_u8: max_abs={max_abs:.3}, mean_rel={mean_rel:.5} (gate 0.10)"
        ));
    }
    assert!(
        failures.is_empty(),
        "public-pipeline parity failures:\n{}",
        failures.join("\n")
    );
}
