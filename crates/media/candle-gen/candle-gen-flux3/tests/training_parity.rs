//! Synthetic tiny native autograd vs independent CPU PyTorch oracle. NO actual BFL weights.
use crate::common::{fixture, host_f32, ids_of, tiny_config, tiny_package};
use candle_gen::candle_core::{safetensors, DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::flux3::training::lora_targets;
use candle_gen::train::optim::TrainOptimizer;
use candle_gen_flux3::{training::TrainingGraph, Flux3Dit, SampleTokens};
use std::collections::BTreeMap;

struct ScalarAdamState {
    weights: Vec<f32>,
    first: Vec<f64>,
    second: Vec<f64>,
}

struct ScalarAdam {
    state: BTreeMap<String, ScalarAdamState>,
}
impl ScalarAdam {
    fn new(graph: &TrainingGraph) -> Self {
        let mut state = BTreeMap::new();
        for factor in &graph.factors {
            for (suffix, var) in [
                ("lora_A.weight", &factor.down),
                ("lora_B.weight", &factor.up),
            ] {
                let weights = host_f32(var.as_tensor());
                let n = weights.len();
                state.insert(
                    format!("{}.{}", factor.target.native, suffix),
                    ScalarAdamState {
                        weights,
                        first: vec![0.0; n],
                        second: vec![0.0; n],
                    },
                );
            }
        }
        Self { state }
    }
    fn step(
        &mut self,
        graph: &TrainingGraph,
        grads: &candle_gen::candle_core::backprop::GradStore,
        step: i32,
    ) {
        let lr = 0.0003f32 as f64;
        let decay = 0.0001f32 as f64;
        for factor in &graph.factors {
            for (suffix, var) in [
                ("lora_A.weight", &factor.down),
                ("lora_B.weight", &factor.up),
            ] {
                let state = self
                    .state
                    .get_mut(&format!("{}.{}", factor.target.native, suffix))
                    .unwrap();
                for (i, gradient) in host_f32(grads.get(var.as_tensor()).unwrap())
                    .into_iter()
                    .enumerate()
                {
                    let g = gradient as f64;
                    state.first[i] = 0.9 * state.first[i] + 0.1 * g;
                    state.second[i] = 0.999 * state.second[i] + 0.001 * g * g;
                    let corrected_m = state.first[i] / (1.0 - 0.9f64.powi(step));
                    let corrected_v = state.second[i] / (1.0 - 0.999f64.powi(step));
                    state.weights[i] = (state.weights[i] as f64 * (1.0 - lr * decay)
                        - lr * corrected_m / (corrected_v.sqrt() + 1e-8))
                        as f32;
                }
            }
        }
    }
}

fn numeric(label: &str, got: &Tensor, expected: &Tensor, absolute: f32, relative: f32) {
    assert_eq!(got.dims(), expected.dims(), "{label}");
    let (got, expected) = (host_f32(got), host_f32(expected));
    let peak = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let error = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(got.iter().all(|v| v.is_finite()), "{label}: nonfinite");
    let worst = got
        .iter()
        .zip(&expected)
        .enumerate()
        .max_by(|(_, (a, b)), (_, (c, d))| ((*a - *b).abs()).total_cmp(&((*c - *d).abs())))
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        error <= absolute + relative * peak,
        "{label}: absolute error {error}, expected peak {peak}, index {worst}, actual {}, reference {}", got[worst], expected[worst]
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn all_six_projection_families_and_earlier_blocks_match_two_adamw_steps() {
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.dit.flow_map = false;
    let weights = safetensors::load(tiny_package().join("model.safetensors"), &device)
        .unwrap()
        .into_iter()
        .filter(|(k, _)| !k.starts_with("ref_time_in."))
        .collect();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let dit = Flux3Dit::from_var_builder(&vb, &config.dit, DType::F32, &device).unwrap();
    let golden = fixture("flux3_training_golden.safetensors");
    let initial = lora_targets(&config.dit, &[])
        .unwrap()
        .into_iter()
        .map(|t| {
            let a = golden
                .require(&format!("initial/{}.lora_A.weight", t.native))
                .clone();
            (t.native, a)
        })
        .collect::<BTreeMap<_, _>>();
    let graph = TrainingGraph::new(dit, 3, 1.5, &[], initial).unwrap();
    assert_eq!(
        graph.factors.len(),
        54,
        "independent Q/K/V A/B at every mode/shared block"
    );
    let mut optimizer = TrainOptimizer::from_config("adamw", graph.vars(), 0.0003, 0.0001).unwrap();
    // Near-zero gradients make Adam's epsilon-conditioned update sensitive to
    // backend reduction rounding. Keep the graph's gradients/loss/velocity against
    // PyTorch, and independently verify the optimizer on the ACTUAL native grads.
    // The companion test pins native optimizer factors against identical PyTorch grads.
    let mut scalar = ScalarAdam::new(&graph);
    let (clean, noise) = (golden.require("clean"), golden.require("noise"));
    let target = noise.sub(clean).unwrap();
    let text_ids = ids_of(golden.require("text_ids"));
    let image_ids = ids_of(golden.require("image_ids"));
    let reference_ids = ids_of(golden.require("reference_ids"));
    for step in 0..2 {
        let sigma = host_f32(golden.require(&format!("step{step}/sigma")))[0];
        let image =
            ((clean * (1.0 - sigma) as f64).unwrap() + (noise * sigma as f64).unwrap()).unwrap();
        let samples = [SampleTokens {
            text: golden.require("text"),
            text_ids: &text_ids,
            image: &image,
            image_ids: &image_ids,
            image_cond: Some((golden.require("reference"), &reference_ids)),
            timestep: sigma,
            target_timestep: None,
        }];
        let prediction = graph.dit.forward_training(&samples).unwrap();
        numeric(
            "velocity",
            &prediction,
            golden.require(&format!("step{step}/prediction")),
            2e-7,
            1e-4,
        );
        let loss = graph.loss(&samples, &target, 1.0).unwrap();
        numeric(
            "loss",
            &loss.reshape(1).unwrap(),
            golden.require(&format!("step{step}/loss")),
            2e-7,
            1e-5,
        );
        let grads = loss.backward().unwrap();
        for factor in &graph.factors {
            for (suffix, var) in [
                ("lora_A.weight", &factor.down),
                ("lora_B.weight", &factor.up),
            ] {
                let name = format!("{}.{}", factor.target.native, suffix);
                let grad = grads
                    .get(var.as_tensor())
                    .unwrap_or_else(|| panic!("detached {name}"));
                numeric(
                    &format!("gradient {name}"),
                    grad,
                    golden.require(&format!("step{step}/gradient/{name}")),
                    1e-11,
                    2e-3,
                );
                if step == 1 {
                    assert!(
                        host_f32(grad).iter().any(|g| g.abs() > 0.0),
                        "zero gradient {name}"
                    );
                }
            }
        }
        scalar.step(&graph, &grads, step + 1);
        optimizer.step(&grads).unwrap();
        for factor in &graph.factors {
            for (suffix, var) in [
                ("lora_A.weight", &factor.down),
                ("lora_B.weight", &factor.up),
            ] {
                let name = format!("{}.{}", factor.target.native, suffix);
                numeric(
                    &format!("updated {name}"),
                    var.as_tensor(),
                    &Tensor::from_vec(scalar.state[&name].weights.clone(), var.shape(), &device)
                        .unwrap(),
                    2e-9,
                    1e-5,
                );
            }
        }
    }
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training_parity::all_six_projection_families_and_earlier_blocks_match_two_adamw_steps");
}

#[test]
fn shared_adamw_matches_torch_on_identical_source_gradients() {
    use candle_gen::candle_core::{backprop::GradStore, Var};
    let golden = fixture("flux3_training_golden.safetensors");
    let mut config = tiny_config();
    config.dit.flow_map = false;
    let mut named = Vec::new();
    for projection in lora_targets(&config.dit, &[]).unwrap() {
        for suffix in ["lora_A.weight", "lora_B.weight"] {
            let name = format!("{}.{}", projection.native, suffix);
            let var = Var::from_tensor(golden.require(&format!("initial/{name}"))).unwrap();
            named.push((name, var));
        }
    }
    let mut optimizer = TrainOptimizer::from_config(
        "adamw",
        named.iter().map(|(_, v)| v.clone()).collect(),
        0.0003,
        0.0001,
    )
    .unwrap();
    for step in 0..2 {
        let mut grads = GradStore::default();
        for (name, var) in &named {
            grads.insert(
                var.as_tensor(),
                golden
                    .require(&format!("step{step}/gradient/{name}"))
                    .clone(),
            );
        }
        optimizer.step(&grads).unwrap();
        for (name, var) in &named {
            numeric(
                &format!("identical-gradient AdamW {name}"),
                var.as_tensor(),
                golden.require(&format!("step{step}/factor/{name}")),
                2e-9,
                1e-5,
            );
        }
    }
}
