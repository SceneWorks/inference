//! Native training step vs the frozen upstream oracle (`tools/dump_iris_train.py`, sc-25685): the
//! rectified-flow loss (v- and x-prediction), and two accumulated, clipped, EMA-tracked optimizer
//! steps for full training and LoRA / LoKr adapters under AdamW and Dion Muon, on the miniature
//! backbone. FP32 on the MLX CPU stream (upstream's fp32 CPU path); Muon's Newton–Schulz is bf16 on
//! both sides.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_gen::gen_core::iris::train::{
    adapter_param_route, full_param_route, FlowObjective, MuonAdjustLr, OptimizerKind,
    OptimizerPlan, Prediction, TimestepSampler, TrainSchedule,
};
use mlx_gen::weights::Weights;
use mlx_gen_iris::train::model::{
    flow_loss, AdapterKind, AdapterTarget, StepBatch, TrainModel, Trainable,
};
use mlx_gen_iris::train::optim::{newton_schulz, IrisOptimizer, Params};
use mlx_gen_iris::train::Window;
use mlx_rs::{Array, Dtype};

use crate::common::{
    assert_close, fixture, host_f32, host_i32, on_cpu, tiny_backbone, tiny_config,
};

const GOLDEN: &str = "iris_train_golden.safetensors";

fn golden() -> Weights {
    fixture(GOLDEN)
}

fn objective(prediction: Prediction) -> FlowObjective {
    FlowObjective {
        num_train_timesteps: 1000,
        shift: 4.0,
        base_shift: 4.0,
        shift_law: "none".into(),
        shift_base_tokens: 256,
        sampler: TimestepSampler::LogitNormal {
            mean: 0.0,
            std: 1.0,
        },
        prediction,
        x_pred_sigma_min: 0.05,
    }
}

fn plan(kind: OptimizerKind) -> OptimizerPlan {
    OptimizerPlan {
        kind,
        base_lr: 1e-3,
        lr: 1e-3,
        betas: (0.9, 0.95),
        weight_decay: 0.01,
        muon_momentum: 0.95,
        muon_nesterov: true,
        muon_adjust_lr: MuonAdjustLr::RmsNorm,
        auto_lr: "none".into(),
        base_batch_size: 256,
    }
}

fn base_f32() -> HashMap<String, Array> {
    Weights::from_file(tiny_backbone().join("model.safetensors"))
        .unwrap()
        .into_tensors()
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Float32).unwrap()))
        .collect()
}

fn batch(g: &Weights, i: usize) -> StepBatch {
    let get = |k: &str| g.require(&format!("mb{i}.{k}")).unwrap().clone();
    let sched = TrainSchedule::new(1000, 4.0);
    let idx = host_i32(&get("t_idx"));
    let sig: Vec<f32> = idx.iter().map(|&t| sched.sigmas[t as usize]).collect();
    let tm: Vec<f32> = idx.iter().map(|&t| sched.model_times[t as usize]).collect();
    let mask = host_i32(&get("y_mask"));
    let t_len = mask.len() / 2;
    StepBatch {
        x0: get("x0"),
        noise: get("noise"),
        sigma: Array::from_slice(&sig, &[2, 1, 1, 1]),
        t: Array::from_slice(&tm, &[2]),
        states: get("y"),
        masks: mask.chunks(t_len).map(|c| c.to_vec()).collect(),
    }
}

fn full_model() -> (TrainModel, Params) {
    let params: Params = base_f32()
        .into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v))
        .collect();
    (
        TrainModel {
            cfg: tiny_config().model,
            compute: Dtype::Float32,
            trainable: Trainable::Full,
        },
        params,
    )
}

/// `Trainer.run`'s accumulate → clip → EMA → step sequence, twice; returns (losses, norms, ema).
fn run_steps(
    model: &TrainModel,
    params: &mut Params,
    opt: &mut IrisOptimizer,
    g: &Weights,
) -> (Vec<f32>, Vec<f32>, Params) {
    let obj = objective(Prediction::Velocity);
    let mut ema = params.clone();
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    // The trainer's own accumulation window (grad_accum = 2, clip 0.5, EMA 0.9, lr 1e-3).
    let mut window = Window::new(2);
    for _ in 0..2 {
        for i in 0..2 {
            losses.push(window.micro(model, params, &batch(g, i), &obj, 2).unwrap());
        }
        let norm = window
            .update(params, Some((&mut ema, 0.9)), opt, 0.5, 1e-3)
            .unwrap()
            .expect("the window holds gradients");
        norms.push(norm as f32);
    }
    (losses, norms, ema)
}

/// Compare an **update** (a parameter or EMA delta from its starting value) by its relative L2
/// error `‖Δgot − Δwant‖ / ‖Δwant‖`. A per-element bound is the wrong gauge for an update: AdamW's
/// `m/√v` normalises a near-zero gradient to a full ±lr step, so f32 summation-order noise in such
/// a gradient flips a handful of elements by O(lr) while the update as a whole agrees to ~1e-3; and
/// Muon's bf16 Newton–Schulz rounds differently on the two sides.
fn assert_update(name: &str, got: &Array, want: &Array, rel_tol: f32) {
    let (g, w) = (host_f32(got), host_f32(want));
    assert_eq!(g.len(), w.len(), "{name}: size");
    let num: f32 = g
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let den: f32 = w.iter().map(|b| b * b).sum::<f32>().sqrt();
    let rel = num / den.max(1e-30);
    eprintln!("{name}: rel L2 {rel:.3e} (‖Δ‖ {den:.3e})");
    assert!(den > 0.0, "{name}: the oracle update is zero");
    assert!(
        rel <= rel_tol,
        "{name}: rel L2 {rel:.3e} exceeds {rel_tol:.0e}"
    );
}

fn assert_vec(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let g = Array::from_slice(got, &[got.len() as i32]);
    let w = Array::from_slice(want, &[want.len() as i32]);
    assert_close(name, &g, &w, tol);
}

#[test]
fn flow_loss_matches_rectified_flow() {
    on_cpu(|| {
        let g = golden();
        let (model, params) = full_model();
        let dit = model.dit(&params).unwrap();
        for (pred, key) in [
            (Prediction::Velocity, "loss_v"),
            (Prediction::Clean, "loss_x"),
        ] {
            let got = flow_loss(&dit, &batch(&g, 0), &objective(pred)).unwrap();
            assert_close(
                key,
                &got.reshape(&[1]).unwrap(),
                g.require(key).unwrap(),
                1e-5,
            );
        }
    });
}

#[test]
fn newton_schulz_matches_dion() {
    on_cpu(|| {
        let g = golden();
        for name in ["tall", "wide", "square"] {
            let input = g.require(&format!("ns.{name}.in")).unwrap();
            let got = newton_schulz(input, 1e-8).unwrap();
            assert_eq!(got.dtype(), Dtype::Bfloat16);
            // bf16 quintic on both sides; differing bf16 rounding of the GEMMs only.
            assert_close(
                &format!("ns.{name}"),
                &got.as_dtype(Dtype::Float32).unwrap(),
                g.require(&format!("ns.{name}.out")).unwrap(),
                3e-2,
            );
        }
    });
}

fn full_case(kind: OptimizerKind, tag: &str, rel_tol: f32) {
    on_cpu(|| {
        let g = golden();
        let (model, mut params) = full_model();
        let init = params.clone();
        let cfg = model.cfg.clone();
        let mut opt = IrisOptimizer::new(&plan(kind), &params, |k, nd| {
            full_param_route(k, nd, kind, &cfg)
        })
        .unwrap();
        let (losses, norms, ema) = run_steps(&model, &mut params, &mut opt, &g);
        let want = |k: &str| g.require(&format!("full_{tag}.{k}")).unwrap().clone();
        assert_vec(
            &format!("{tag} losses"),
            &losses,
            &host_f32(&want("losses")),
            1e-4,
        );
        assert_vec(
            &format!("{tag} grad norms"),
            &norms,
            &host_f32(&want("grad_norms")),
            1e-4,
        );
        let order: Vec<String> = g
            .metadata("full_order")
            .unwrap()
            .split(',')
            .map(str::to_string)
            .collect();
        let norm = |a: &Array| -> f32 { host_f32(a).iter().map(|v| v * v).sum::<f32>().sqrt() };
        let got_delta: Vec<f32> = order
            .iter()
            .map(|k| norm(&params[k.as_str()].subtract(&init[k.as_str()]).unwrap()))
            .collect();
        let got_ema: Vec<f32> = order
            .iter()
            .map(|k| norm(&ema[k.as_str()].subtract(&init[k.as_str()]).unwrap()))
            .collect();
        // The update size of EVERY backbone tensor, then the full update of a subset covering each
        // route (Muon plain / split-3 / split-6, boundary AdamW, vectors, the position table).
        let v = |x: &[f32]| Array::from_slice(x, &[x.len() as i32]);
        assert_update(
            &format!("{tag} per-tensor |Δθ|"),
            &v(&got_delta),
            &v(&host_f32(&want("delta_norms"))),
            rel_tol,
        );
        assert_update(
            &format!("{tag} per-tensor |Δema|"),
            &v(&got_ema),
            &v(&host_f32(&want("ema_delta_norms"))),
            rel_tol,
        );
        // Every tensor moved (observable parameter updates).
        assert!(
            got_delta.iter().all(|&d| d > 0.0),
            "{tag}: a parameter did not move"
        );
        for k in g
            .keys()
            .filter_map(|k| k.strip_prefix(&format!("full_{tag}.delta.")))
            .map(str::to_string)
            .collect::<Vec<_>>()
        {
            let got = params[k.as_str()].subtract(&init[k.as_str()]).unwrap();
            let want = g.require(&format!("full_{tag}.delta.{k}")).unwrap();
            if kind == OptimizerKind::Muon && k.starts_with("modulation_cores") {
                // A shared adaLN core's gradient is Σ outer(∂mod, silu(t_emb)) — rank ≤ the number
                // of distinct timesteps in the window (here 4 in a 384x64 matrix). Newton–Schulz
                // drives every singular value toward 1, so the bf16 noise floor of the null space
                // is amplified ~10³x on both sides and the two orthogonalizations differ in that
                // subspace by construction. What the port must get right — the six independently
                // orthogonalized row blocks and their per-block lr scales — shows in each block's
                // update norm (a wrong scale is a 2.45x error); the direction is bounded loosely.
                let gp = got.split(6, 0).unwrap();
                let wp = want.split(6, 0).unwrap();
                for i in 0..6 {
                    let n = |a: &Array| host_f32(a).iter().map(|x| x * x).sum::<f32>().sqrt();
                    let (a, b) = (n(&gp[i]), n(&wp[i]));
                    eprintln!("{tag} {k} block {i}: |Δ| {a:.4e} vs {b:.4e}");
                    assert!((a - b).abs() <= 1e-2 * b, "{k} block {i}: |Δ| {a} vs {b}");
                }
                assert_update(&format!("{tag} Δ{k}"), &got, want, 0.25);
                let got = ema[k.as_str()].subtract(&init[k.as_str()]).unwrap();
                assert_update(
                    &format!("{tag} Δema {k}"),
                    &got,
                    g.require(&format!("full_{tag}.ema.{k}")).unwrap(),
                    0.25,
                );
                continue;
            }
            assert_update(&format!("{tag} Δ{k}"), &got, want, rel_tol);
            let got = ema[k.as_str()].subtract(&init[k.as_str()]).unwrap();
            assert_update(
                &format!("{tag} Δema {k}"),
                &got,
                g.require(&format!("full_{tag}.ema.{k}")).unwrap(),
                rel_tol,
            );
        }
    });
}

#[test]
fn full_training_adamw_matches_upstream() {
    full_case(OptimizerKind::AdamW, "adamw", 2e-2);
}

#[test]
fn full_training_muon_matches_upstream() {
    full_case(OptimizerKind::Muon, "muon", 5e-2);
}

fn adapter_case(kind_name: &str, opt_kind: OptimizerKind, tol: f32) {
    on_cpu(|| {
        let g = golden();
        let targets: Vec<String> = g
            .metadata("targets")
            .unwrap()
            .split(',')
            .map(str::to_string)
            .collect();
        let base = base_f32();
        let kind = if kind_name == "lora" {
            AdapterKind::Lora {
                rank: 2,
                alpha: 4.0,
            }
        } else {
            AdapterKind::Lokr {
                rank: 2,
                alpha: 4.0,
                factor: -1,
            }
        };
        let targets: Vec<AdapterTarget> = targets
            .iter()
            .map(|p| {
                let w = &base[&format!("{p}.weight")];
                AdapterTarget {
                    path: p.clone(),
                    out_f: w.shape()[0],
                    in_f: w.shape()[1],
                }
            })
            .collect();
        let prefix = format!("{kind_name}.init.");
        let mut params: Params = g
            .keys()
            .filter_map(|k| k.strip_prefix(&prefix))
            .map(|k| {
                (
                    Rc::from(k),
                    g.require(&format!("{prefix}{k}")).unwrap().clone(),
                )
            })
            .collect();
        let expected: usize = targets.iter().map(|t| t.factor_keys(kind).len()).sum();
        assert_eq!(
            params.len(),
            expected,
            "the native factor layout matches the oracle's"
        );
        let owner: HashMap<String, String> = targets
            .iter()
            .flat_map(|t| t.factor_keys(kind).into_iter().map(|k| (k, t.path.clone())))
            .collect();
        let model = TrainModel {
            cfg: tiny_config().model,
            compute: Dtype::Float32,
            trainable: Trainable::Adapter {
                kind,
                targets,
                base,
            },
        };
        let mut opt = IrisOptimizer::new(&plan(opt_kind), &params, |k, nd| {
            adapter_param_route(&owner[k], nd, opt_kind)
        })
        .unwrap();
        let (losses, norms, ema) = run_steps(&model, &mut params, &mut opt, &g);
        let tag = format!("{kind_name}_{}", opt_kind.as_str());
        assert_vec(
            &format!("{tag} losses"),
            &losses,
            &host_f32(g.require(&format!("{tag}.losses")).unwrap()),
            1e-4,
        );
        assert_vec(
            &format!("{tag} grad norms"),
            &norms,
            &host_f32(g.require(&format!("{tag}.grad_norms")).unwrap()),
            1e-4,
        );
        for (k, p) in &params {
            let init = g.require(&format!("{prefix}{k}")).unwrap();
            let want = g
                .require(&format!("{tag}.final.{k}"))
                .unwrap()
                .subtract(init)
                .unwrap();
            assert_update(
                &format!("{tag} Δ{k}"),
                &p.subtract(init).unwrap(),
                &want,
                tol,
            );
            let want = g
                .require(&format!("{tag}.ema.{k}"))
                .unwrap()
                .subtract(init)
                .unwrap();
            assert_update(
                &format!("{tag} Δema {k}"),
                &ema[k].subtract(init).unwrap(),
                &want,
                tol,
            );
        }
    });
}

#[test]
fn lora_adamw_matches_oracle() {
    adapter_case("lora", OptimizerKind::AdamW, 2e-2);
}

#[test]
fn lora_muon_matches_oracle() {
    adapter_case("lora", OptimizerKind::Muon, 5e-2);
}

#[test]
fn lokr_adamw_matches_oracle() {
    adapter_case("lokr", OptimizerKind::AdamW, 2e-3);
}

#[test]
fn lokr_muon_matches_oracle() {
    adapter_case("lokr", OptimizerKind::Muon, 6e-2);
}
