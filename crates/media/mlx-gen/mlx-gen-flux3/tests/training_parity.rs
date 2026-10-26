//! Synthetic CPU PyTorch oracle; executed only on supported MLX hosts.
use crate::common::{cpu_process, fixture, host_f32, on_cpu, tiny_base_config, tiny_package};
use mlx_gen::gen_core::flux3::training::lora_targets;
use mlx_gen::{train::lora::LoraParams, weights::Weights};
use mlx_gen_flux3::{
    training::{optim::SourceAdamW, TrainingGraph},
    Flux3Dit, SampleTokens,
};
use mlx_rs::{Array, Dtype};
use std::collections::BTreeMap;

struct ScalarState {
    weights: Vec<f32>,
    first: Vec<f64>,
    second: Vec<f64>,
}
struct ScalarAdam {
    state: BTreeMap<String, ScalarState>,
}
impl ScalarAdam {
    fn new(params: &LoraParams) -> Self {
        Self {
            state: params
                .iter()
                .map(|(key, array)| {
                    let weights = host_f32(array);
                    let n = weights.len();
                    (
                        key.to_string(),
                        ScalarState {
                            weights,
                            first: vec![0.0; n],
                            second: vec![0.0; n],
                        },
                    )
                })
                .collect(),
        }
    }
    fn step(&mut self, grads: &LoraParams, step: i32) {
        let lr = 0.0003f32 as f64;
        let decay = 0.0001f32 as f64;
        for (name, gradient) in grads {
            let state = self.state.get_mut(name.as_ref()).unwrap();
            for (i, gradient) in host_f32(gradient).into_iter().enumerate() {
                let g = gradient as f64;
                state.first[i] = 0.9 * state.first[i] + 0.1 * g;
                state.second[i] = 0.999 * state.second[i] + 0.001 * g * g;
                let m = state.first[i] / (1.0 - 0.9f64.powi(step));
                let v = state.second[i] / (1.0 - 0.999f64.powi(step));
                state.weights[i] = (state.weights[i] as f64 * (1.0 - lr * decay)
                    - lr * m / (v.sqrt() + 1e-8)) as f32;
            }
        }
    }
}

fn ids_of(array: &Array) -> Vec<[f32; 4]> {
    host_f32(array)
        .chunks_exact(4)
        .map(|v| [v[0], v[1], v[2], v[3]])
        .collect()
}

fn numeric(label: &str, got: &Array, expected: &Array, absolute: f32, relative: f32) {
    assert_eq!(got.shape(), expected.shape(), "{label}");
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
    assert!(
        error <= absolute + relative * peak,
        "{label}: error {error}, peak {peak}"
    );
}

#[test]
fn all_six_projection_families_match_two_adamw_steps() {
    cpu_process(
        "training_parity::all_six_projection_families_match_two_adamw_steps",
        || {
            on_cpu(|| {
                let config = tiny_base_config();
                let mut weights =
                    Weights::from_file(tiny_package().join("model.safetensors")).unwrap();
                let remove = weights
                    .keys()
                    .filter(|key| key.starts_with("ref_time_in."))
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                for key in remove {
                    weights.remove(&key);
                }
                let dit = Flux3Dit::from_weights(&weights, &config.dit, Dtype::Float32).unwrap();
                let mut graph = TrainingGraph::new(dit, 3, 1.5, &[]).unwrap();
                let golden = fixture("flux3_training_golden.safetensors");
                let mut params = LoraParams::new();
                for target in lora_targets(&config.dit, &[]).unwrap() {
                    for suffix in ["lora_A.weight", "lora_B.weight"] {
                        let name = format!("{}.{}", target.native, suffix);
                        params.insert(
                            name.clone().into(),
                            golden.require(&format!("initial/{name}")).unwrap().clone(),
                        );
                    }
                }
                assert_eq!(params.len(), 108);
                let mut optimizer = SourceAdamW::new(0.0003, 0.0001).unwrap();
                let mut scalar = ScalarAdam::new(&params);
                let get = |key: &str| golden.require(key).unwrap();
                let clean = get("clean");
                let noise = get("noise");
                let target = noise.subtract(clean).unwrap();
                let text_ids = ids_of(get("text_ids"));
                let image_ids = ids_of(get("image_ids"));
                let reference_ids = ids_of(get("reference_ids"));
                for step in 0..2 {
                    let sigma = host_f32(get(&format!("step{step}/sigma")))[0];
                    let image = clean
                        .multiply(Array::from_f32(1.0 - sigma))
                        .unwrap()
                        .add(noise.multiply(Array::from_f32(sigma)).unwrap())
                        .unwrap();
                    let samples = [SampleTokens {
                        text: get("text"),
                        text_ids: &text_ids,
                        image: &image,
                        image_ids: &image_ids,
                        image_cond: Some((get("reference"), &reference_ids)),
                        timestep: sigma,
                        target_timestep: None,
                    }];
                    let (loss, grads) = graph
                        .loss_and_grads(&params, &samples, &target, 1.0)
                        .unwrap();
                    numeric(
                        "loss",
                        &loss.reshape(&[1]).unwrap(),
                        get(&format!("step{step}/loss")),
                        2e-7,
                        1e-5,
                    );
                    for (name, gradient) in &grads {
                        numeric(
                            &format!("gradient {name}"),
                            gradient,
                            get(&format!("step{step}/gradient/{name}")),
                            1e-11,
                            2e-3,
                        );
                        if step == 1 {
                            assert!(
                                host_f32(gradient).iter().any(|g| g.abs() > 0.0),
                                "zero gradient {name}"
                            );
                        }
                    }
                    assert_eq!(grads.len(), 108);
                    scalar.step(&grads, step + 1);
                    optimizer.step(&mut params, &grads).unwrap();
                    for (name, factor) in &params {
                        numeric(
                            &format!("updated {name}"),
                            factor,
                            &Array::from_slice(
                                &scalar.state[name.as_ref()].weights,
                                factor.shape(),
                            ),
                            2e-9,
                            1e-5,
                        );
                    }
                }
            })
        },
    );
}

#[test]
fn corrected_adamw_matches_identical_torch_gradients() {
    cpu_process(
        "training_parity::corrected_adamw_matches_identical_torch_gradients",
        || {
            on_cpu(|| {
                let golden = fixture("flux3_training_golden.safetensors");
                let config = tiny_base_config();
                let mut params = LoraParams::new();
                for projection in lora_targets(&config.dit, &[]).unwrap() {
                    for suffix in ["lora_A.weight", "lora_B.weight"] {
                        let name = format!("{}.{}", projection.native, suffix);
                        params.insert(
                            name.clone().into(),
                            golden.require(&format!("initial/{name}")).unwrap().clone(),
                        );
                    }
                }
                let mut optimizer = SourceAdamW::new(0.0003, 0.0001).unwrap();
                for step in 0..2 {
                    let grads: LoraParams = params
                        .keys()
                        .map(|name| {
                            (
                                name.clone(),
                                golden
                                    .require(&format!("step{step}/gradient/{name}"))
                                    .unwrap()
                                    .clone(),
                            )
                        })
                        .collect();
                    optimizer.step(&mut params, &grads).unwrap();
                    for (name, factor) in &params {
                        numeric(
                            &format!("identical-gradient AdamW {name}"),
                            factor,
                            golden
                                .require(&format!("step{step}/factor/{name}"))
                                .unwrap(),
                            2e-9,
                            1e-5,
                        );
                    }
                }
            })
        },
    );
}
