//! The Candle training step vs the frozen upstream oracle (`tools/dump_iris_train.py`, sc-25686) —
//! the MLX twin's `train_parity`, on the **same** committed fixture: the rectified-flow loss (v- and
//! x-prediction), Dion's bf16 Newton–Schulz, and two accumulated, clipped, EMA-tracked optimizer
//! steps for full training and LoRA / LoKr adapters under AdamW and Dion Muon, on the miniature
//! backbone. FP32 on the Candle CPU backend (upstream's fp32 CPU path); Muon's Newton–Schulz is
//! bf16 on both sides. Every tolerance is printed beside its measured value.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle_gen::candle_core::{DType, Tensor, TensorId, Var};
use candle_gen::gen_core::iris::train::{
    adapter_param_route, full_param_route, FlowObjective, MuonAdjustLr, OptimizerKind,
    OptimizerPlan, Prediction, TimestepSampler, TrainSchedule,
};
use candle_gen_iris::train::model::{
    flow_loss, AdapterKind, AdapterTarget, StepBatch, TrainModel, Trainable,
};
use candle_gen_iris::train::optim::{
    detached_copy, newton_schulz, snapshot, IrisOptimizer, Params, Tensors,
};
use candle_gen_iris::train::Window;

use crate::common::{
    assert_close, cpu, fixture, host_f32, host_i32, tiny_backbone, tiny_config, Fixture,
};

const GOLDEN: &str = "iris_train_golden.safetensors";

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

fn base_f32() -> HashMap<String, Tensor> {
    candle_gen::candle_core::safetensors::load(tiny_backbone().join("model.safetensors"), &cpu())
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, v.to_dtype(DType::F32).unwrap()))
        .collect()
}

fn batch(g: &Fixture, i: usize) -> StepBatch {
    let get = |k: &str| {
        g.require(&format!("mb{i}.{k}"))
            .to_dtype(DType::F32)
            .unwrap()
    };
    let sched = TrainSchedule::new(1000, 4.0);
    let idx = host_i32(g.require(&format!("mb{i}.t_idx")));
    let sig: Vec<f32> = idx.iter().map(|&t| sched.sigmas[t as usize]).collect();
    let tm: Vec<f32> = idx.iter().map(|&t| sched.model_times[t as usize]).collect();
    let mask = host_i32(g.require(&format!("mb{i}.y_mask")));
    let t_len = mask.len() / 2;
    StepBatch {
        x0: get("x0"),
        noise: get("noise"),
        sigma: Tensor::from_vec(sig, (2, 1, 1, 1), &cpu()).unwrap(),
        t: Tensor::from_vec(tm, 2, &cpu()).unwrap(),
        states: get("y"),
        masks: mask.chunks(t_len).map(|c| c.to_vec()).collect(),
    }
}

fn vars(m: HashMap<String, Tensor>) -> Params {
    m.into_iter()
        .map(|(k, v)| (k, Var::from_tensor(&v).unwrap()))
        .collect()
}

fn full_model() -> (TrainModel, Params) {
    (
        TrainModel {
            cfg: tiny_config().model,
            compute: DType::F32,
            device: cpu(),
            trainable: Trainable::Full,
        },
        vars(base_f32()),
    )
}

/// `Trainer.run`'s accumulate → clip → EMA → step sequence, twice; returns (losses, norms, ema).
fn run_steps(
    model: &TrainModel,
    params: &Params,
    opt: &mut IrisOptimizer,
    g: &Fixture,
) -> (Vec<f32>, Vec<f32>, Tensors) {
    let obj = objective(Prediction::Velocity);
    let mut ema = detached_copy(params).unwrap();
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

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compare an **update** (a parameter or EMA delta from its starting value) by its relative L2
/// error `‖Δgot − Δwant‖ / ‖Δwant‖` — the MLX twin's gauge (an elementwise bound is the wrong one:
/// AdamW's `m/√v` turns f32 noise in a near-zero gradient into an O(lr) element flip; Muon's bf16
/// Newton–Schulz rounds differently on the two sides). Returns the measured error.
fn assert_update(name: &str, got: &[f32], want: &[f32], rel_tol: f32) -> f32 {
    assert_eq!(got.len(), want.len(), "{name}: size");
    let diff: Vec<f32> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    let den = norm(want);
    let rel = norm(&diff) / den.max(1e-30);
    eprintln!("{name}: rel L2 {rel:.3e} (‖Δ‖ {den:.3e}, tol {rel_tol:.0e})");
    assert!(den > 0.0, "{name}: the oracle update is zero");
    assert!(
        rel <= rel_tol,
        "{name}: rel L2 {rel:.3e} exceeds {rel_tol:.0e}"
    );
    rel
}

fn delta(a: &Tensor, b: &Tensor) -> Vec<f32> {
    host_f32(
        &a.to_dtype(DType::F32)
            .unwrap()
            .sub(&b.to_dtype(DType::F32).unwrap())
            .unwrap(),
    )
}

fn assert_vec(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let g = Tensor::new(got, &cpu()).unwrap();
    let w = Tensor::new(want, &cpu()).unwrap();
    assert_close(name, &g, &w, tol);
}

#[test]
fn flow_loss_matches_rectified_flow() {
    let g = fixture(GOLDEN);
    let (model, params) = full_model();
    let snapshot: Tensors = params
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect();
    let dit = model.dit(&snapshot).unwrap();
    for (pred, key) in [
        (Prediction::Velocity, "loss_v"),
        (Prediction::Clean, "loss_x"),
    ] {
        let got = flow_loss(&dit, &batch(&g, 0), &objective(pred)).unwrap();
        assert_close(key, &got.reshape(1).unwrap(), g.require(key), 1e-6);
    }
    // The autograd path (composable RMSNorm / softmax over `Var` masters) computes the same loss.
    let tracked = model
        .dit(&candle_gen_iris::train::optim::snapshot(&params))
        .unwrap();
    let got = flow_loss(&tracked, &batch(&g, 0), &objective(Prediction::Velocity)).unwrap();
    assert!(
        got.track_op(),
        "the training forward is on the autograd tape"
    );
    assert_close(
        "loss_v (autograd)",
        &got.reshape(1).unwrap(),
        g.require("loss_v"),
        1e-6,
    );
}

#[test]
fn newton_schulz_matches_dion() {
    let g = fixture(GOLDEN);
    for name in ["tall", "wide", "square"] {
        let input = g.require(&format!("ns.{name}.in"));
        let got = newton_schulz(input, 1e-8).unwrap();
        assert_eq!(got.dtype(), DType::BF16);
        // bf16 quintic on both sides; differing bf16 rounding of the GEMMs only.
        assert_close(
            &format!("ns.{name}"),
            &got.to_dtype(DType::F32).unwrap(),
            g.require(&format!("ns.{name}.out")),
            3e-2,
        );
    }
}

fn full_case(kind: OptimizerKind, tag: &str, rel_tol: f32) {
    let g = fixture(GOLDEN);
    let (model, params) = full_model();
    let init = detached_copy(&params).unwrap();
    let cfg = model.cfg.clone();
    let mut opt = IrisOptimizer::new(&plan(kind), &params, |k, nd| {
        full_param_route(k, nd, kind, &cfg)
    })
    .unwrap();
    let (losses, norms, ema) = run_steps(&model, &params, &mut opt, &g);
    let want = |k: &str| host_f32(g.require(&format!("full_{tag}.{k}")));
    assert_vec(&format!("{tag} losses"), &losses, &want("losses"), 1e-5);
    assert_vec(
        &format!("{tag} grad norms"),
        &norms,
        &want("grad_norms"),
        1e-5,
    );
    let order: Vec<String> = g
        .meta("full_order")
        .split(',')
        .map(str::to_string)
        .collect();
    let got_delta: Vec<f32> = order
        .iter()
        .map(|k| norm(&delta(params[k].as_tensor(), &init[k])))
        .collect();
    let got_ema: Vec<f32> = order
        .iter()
        .map(|k| norm(&delta(&ema[k], &init[k])))
        .collect();
    // The update size of EVERY backbone tensor, then the full update of a subset covering each
    // route (Muon plain / split-3 / split-6, boundary AdamW, vectors, the position table).
    assert_update(
        &format!("{tag} per-tensor |Δθ|"),
        &got_delta,
        &want("delta_norms"),
        rel_tol,
    );
    assert_update(
        &format!("{tag} per-tensor |Δema|"),
        &got_ema,
        &want("ema_delta_norms"),
        rel_tol,
    );
    assert!(
        got_delta.iter().all(|&d| d > 0.0),
        "{tag}: a parameter did not move"
    );
    let prefix = format!("full_{tag}.delta.");
    let keys: Vec<String> = g
        .tensors
        .keys()
        .filter_map(|k| k.strip_prefix(&prefix))
        .map(str::to_string)
        .collect();
    assert!(!keys.is_empty());
    for k in keys {
        let got = delta(params[&k].as_tensor(), &init[&k]);
        let want_d = host_f32(g.require(&format!("full_{tag}.delta.{k}")));
        let got_e = delta(&ema[&k], &init[&k]);
        let want_e = host_f32(g.require(&format!("full_{tag}.ema.{k}")));
        if kind == OptimizerKind::Muon && k.starts_with("modulation_cores") {
            // A shared adaLN core's gradient is Σ outer(∂mod, silu(t_emb)) — rank ≤ the number of
            // distinct timesteps in the window (4 here). Newton–Schulz drives every singular value
            // toward 1, so the bf16 noise floor of the null space is amplified on both sides and
            // the two orthogonalizations differ in that subspace by construction (the MLX twin's
            // finding). What the port must get right — the six independently orthogonalized row
            // blocks and their per-block lr scales — shows in each block's update norm (a wrong
            // scale is a 2.45x error); the direction is bounded at the MLX twin's 0.25.
            let rows = params[&k].as_tensor().dim(0).unwrap();
            let block = got.len() / 6;
            assert_eq!(rows % 6, 0);
            for i in 0..6 {
                let (a, b) = (
                    norm(&got[i * block..(i + 1) * block]),
                    norm(&want_d[i * block..(i + 1) * block]),
                );
                eprintln!("{tag} {k} block {i}: |Δ| {a:.4e} vs {b:.4e}");
                assert!((a - b).abs() <= 1e-2 * b, "{k} block {i}: |Δ| {a} vs {b}");
            }
            assert_update(&format!("{tag} Δ{k}"), &got, &want_d, 0.25);
            assert_update(&format!("{tag} Δema {k}"), &got_e, &want_e, 0.25);
            continue;
        }
        assert_update(&format!("{tag} Δ{k}"), &got, &want_d, rel_tol);
        assert_update(&format!("{tag} Δema {k}"), &got_e, &want_e, rel_tol);
    }
}

#[test]
fn full_training_adamw_matches_upstream() {
    full_case(OptimizerKind::AdamW, "adamw", 2e-2);
}

#[test]
fn full_training_muon_matches_upstream() {
    full_case(OptimizerKind::Muon, "muon", 5e-2);
}

/// The adapter run on the miniature backbone: the model, its factor `Var`s at the oracle's
/// initial values (and those values), and each factor key's owning target path.
fn adapter_model(
    g: &Fixture,
    kind_name: &str,
) -> (
    TrainModel,
    Params,
    BTreeMap<String, Tensor>,
    HashMap<String, String>,
) {
    let targets: Vec<String> = g.meta("targets").split(',').map(str::to_string).collect();
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
            let (out_f, in_f) = base[&format!("{p}.weight")].dims2().unwrap();
            AdapterTarget {
                path: p.clone(),
                out_f,
                in_f,
            }
        })
        .collect();
    let prefix = format!("{kind_name}.init.");
    let init: BTreeMap<String, Tensor> = g
        .tensors
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(&prefix)
                .map(|k| (k.to_string(), v.to_dtype(DType::F32).unwrap()))
        })
        .collect();
    let params: Params = init
        .iter()
        .map(|(k, v)| (k.clone(), Var::from_tensor(v).unwrap()))
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
        compute: DType::F32,
        device: cpu(),
        trainable: Trainable::Adapter {
            kind,
            targets,
            base,
        },
    };
    (model, params, init, owner)
}

fn adapter_case(kind_name: &str, opt_kind: OptimizerKind, tol: f32) {
    let g = fixture(GOLDEN);
    let (model, params, init, owner) = adapter_model(&g, kind_name);
    let mut opt = IrisOptimizer::new(&plan(opt_kind), &params, |k, nd| {
        adapter_param_route(&owner[k], nd, opt_kind)
    })
    .unwrap();
    let (losses, norms, ema) = run_steps(&model, &params, &mut opt, &g);
    let tag = format!("{kind_name}_{}", opt_kind.as_str());
    assert_vec(
        &format!("{tag} losses"),
        &losses,
        &host_f32(g.require(&format!("{tag}.losses"))),
        1e-5,
    );
    assert_vec(
        &format!("{tag} grad norms"),
        &norms,
        &host_f32(g.require(&format!("{tag}.grad_norms"))),
        1e-5,
    );
    for (k, p) in &params {
        let start = &init[k];
        assert_update(
            &format!("{tag} Δ{k}"),
            &delta(p.as_tensor(), start),
            &delta(g.require(&format!("{tag}.final.{k}")), start),
            tol,
        );
        assert_update(
            &format!("{tag} Δema {k}"),
            &delta(&ema[k], start),
            &delta(g.require(&format!("{tag}.ema.{k}")), start),
            tol,
        );
    }
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

/// The step's `GradStore` holds a gradient for every trainable `Var` and for nothing else: the
/// frozen base weights (and the step's data) of an adapter run, which sit as matmul / binary-op
/// operands of the tracked graph, get no gradient computed or stored (the vendored candle-core
/// patch, sc-25686) — upstream candle stored a full `[in, out]` weight gradient per frozen Linear.
fn assert_store_is_exactly_the_trainables(
    tag: &str,
    model: &TrainModel,
    params: &Params,
    g: &Fixture,
) {
    let dit = model.dit(&snapshot(params)).unwrap();
    let loss = flow_loss(&dit, &batch(g, 0), &objective(Prediction::Velocity)).unwrap();
    let store = loss.backward().unwrap();
    let vars: HashSet<TensorId> = params.values().map(|v| v.as_tensor().id()).collect();
    let stored: HashSet<TensorId> = store.get_ids().copied().collect();
    let extra = stored.difference(&vars).count();
    assert_eq!(
        extra, 0,
        "{tag}: {extra} gradient(s) stored for non-trainable tensors"
    );
    assert_eq!(stored, vars, "{tag}: every trainable tensor has a gradient");
}

#[test]
fn backward_stores_gradients_only_for_trainable_tensors() {
    let g = fixture(GOLDEN);
    for kind_name in ["lora", "lokr"] {
        let (model, params, _, _) = adapter_model(&g, kind_name);
        assert_store_is_exactly_the_trainables(kind_name, &model, &params, &g);
    }
    let (model, params) = full_model();
    assert_store_is_exactly_the_trainables("full", &model, &params, &g);
}
