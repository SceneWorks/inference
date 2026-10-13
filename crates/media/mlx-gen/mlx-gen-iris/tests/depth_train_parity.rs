//! The depth trainer's numerics vs the torch oracle (`tools/dump_iris_depth_train.py`): the frozen
//! upstream `IrisDepth` on the miniature export (`tests/fixtures/tiny-depth`), the paper's
//! normalization and Eq. 5 loss, one `clip_grad_norm_(1.0)` + `torch.optim.AdamW` step — FP32 on the
//! MLX CPU stream.
//!
//! Stages and tolerances:
//! * normalized targets (`gen_core` vs numpy): 1e-5 absolute;
//! * prediction: 1e-4 of peak (summation order, the `depth_parity` gate);
//! * loss: 1e-4 relative; per-tensor gradient norms and the selected gradients: 1e-3 of the
//!   tensor's peak (backward through the whole backbone; measured margins printed);
//! * the AdamW step: on every element whose oracle gradient is above 1e-3 of its tensor's peak the
//!   update matches to 1e-6 (≈ 1e-3·lr); the first Adam step is `lr·sign(g)` there, so elements whose
//!   gradient is at rounding level may legitimately flip sign — those are bounded by `2·lr`.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_gen::gen_core::iris::downstream::{EmptyPrompt, TaskExport};
use mlx_gen::gen_core::iris::train::depth::{
    normalize_log_depth, DepthBatch, DepthLossPlan, MIN_VALID_PIXELS_FLOOR,
};
use mlx_gen::gen_core::iris::train::{
    full_param_route, MuonAdjustLr, OptimizerKind, OptimizerPlan,
};
use mlx_gen::gen_core::iris::IrisTask;
use mlx_gen_iris::train::depth::{load_export_tensors, DepthModel, DepthStep, DepthWindow};
use mlx_gen_iris::train::optim::{IrisOptimizer, Params};
use mlx_rs::{Array, Dtype};

use crate::common::{fixture, fixtures, host_f32, on_cpu};

fn golden_batch(g: &mlx_gen::weights::Weights) -> DepthBatch {
    let depth = host_f32(g.require("depth").unwrap());
    let mask = host_f32(g.require("mask").unwrap());
    let rgb = host_f32(g.require("rgb").unwrap());
    let want = host_f32(g.require("target").unwrap());
    let (b, h, w) = (2usize, 16usize, 16usize);
    let mut target = Vec::new();
    let mut valid = Vec::new();
    let mut counts = Vec::new();
    for i in 0..b {
        let d = &depth[i * h * w..(i + 1) * h * w];
        let v: Vec<bool> = mask[i * h * w..(i + 1) * h * w]
            .iter()
            .map(|&m| m > 0.5)
            .collect();
        let t = normalize_log_depth(d, &v, &DepthLossPlan::default(), MIN_VALID_PIXELS_FLOOR)
            .expect("the oracle rows are normalizable");
        target.extend(t);
        valid.extend(v.iter().map(|&x| if x { 1.0f32 } else { 0.0 }));
        counts.push(v.iter().filter(|&&x| x).count());
    }
    let max = target
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("target: max|Δ| = {max:.3e}");
    assert!(max <= 1e-5, "normalized target differs from numpy by {max}");
    DepthBatch {
        batch: b,
        height: h,
        width: w,
        rgb,
        target,
        valid,
        valid_count: counts,
    }
}

fn peak_rel(got: &[f32], want: &[f32]) -> f32 {
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / peak
}

fn setup() -> (DepthModel, Params) {
    let dir = fixtures().join("tiny-depth");
    let export = TaskExport::from_dir(&dir, IrisTask::Depth, "test").unwrap();
    let prompt = EmptyPrompt::read(&export.empty_prompt_path(), &export.config).unwrap();
    let params: Params = load_export_tensors(&dir)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v))
        .collect();
    (
        DepthModel::new(export.config, Dtype::Float32, &prompt),
        params,
    )
}

#[test]
fn one_step_matches_the_torch_oracle() {
    crate::common::cpu_process(
        "depth_train_parity::one_step_matches_the_torch_oracle",
        one_step_matches_the_torch_oracle_body,
    );
}

fn one_step_matches_the_torch_oracle_body() {
    on_cpu(|| {
        let g = fixture("iris_depth_train_golden.safetensors");
        let batch = golden_batch(&g);
        let loss_plan = DepthLossPlan::default();
        let (model, mut params) = setup();
        let step = DepthStep::new(&batch, &loss_plan);

        let pred = model.predict(&params, &step.rgb).unwrap();
        let rel = peak_rel(&host_f32(&pred), &host_f32(g.require("pred0").unwrap()));
        eprintln!("pred0: {rel:.3e} of peak");
        assert!(rel <= 1e-4);

        let mut window = DepthWindow::new(1);
        let loss = window.micro(&model, &params, &step, &loss_plan).unwrap();
        let want = host_f32(g.require("loss0").unwrap())[0];
        eprintln!("loss0: {loss} vs {want}");
        assert!(((loss - want) / want).abs() <= 1e-4);

        // gradients: the window's accumulator is private, so take them from one more evaluation
        let (_, grads) =
            mlx_gen_iris::train::depth::loss_and_grads(&model, &params, &step, &loss_plan, 1.0)
                .unwrap();
        let keys: Vec<String> =
            serde_json::from_str(g.metadata("grad_norm_keys").unwrap()).unwrap();
        assert_eq!(
            keys.len(),
            params.len(),
            "the oracle and the trainer train the same tensors"
        );
        let norms = host_f32(g.require("grad_norms").unwrap());
        for (k, want) in keys.iter().zip(&norms) {
            let got = host_f32(&grads[k.as_str()]);
            let n = got.iter().map(|v| v * v).sum::<f32>().sqrt();
            let tol = 1e-3 * want.max(1e-6);
            assert!((n - want).abs() <= tol.max(1e-7), "{k}: |g| {n} vs {want}");
        }
        let selected: Vec<String> = serde_json::from_str(g.metadata("selected").unwrap()).unwrap();
        let mut want_grads: HashMap<&str, Vec<f32>> = HashMap::new();
        for k in &selected {
            let want = host_f32(g.require(&format!("grad/{k}")).unwrap());
            let rel = peak_rel(&host_f32(&grads[k.as_str()]), &want);
            eprintln!("grad {k}: {rel:.3e} of peak");
            assert!(rel <= 1e-3, "{k}");
            want_grads.insert(k.as_str(), want);
        }
        // The widened channel never sees a non-zero input: its columns get exactly zero gradient.
        let s = host_f32(&grads["s_embedder.proj.weight"]);
        let cols = 4 * 4 * 4;
        assert!(s
            .chunks(cols)
            .all(|row| row[3 * 16..].iter().all(|&v| v == 0.0)));

        let plan = OptimizerPlan {
            kind: OptimizerKind::AdamW,
            base_lr: 1e-3,
            lr: 1e-3,
            betas: (0.9, 0.999),
            weight_decay: 0.01,
            muon_momentum: 0.95,
            muon_nesterov: true,
            muon_adjust_lr: MuonAdjustLr::RmsNorm,
            auto_lr: "none".into(),
            base_batch_size: 256,
        };
        let cfg = model.config.model.clone();
        let mut opt = IrisOptimizer::new(&plan, &params, |k, nd| {
            full_param_route(k, nd, OptimizerKind::AdamW, &cfg)
        })
        .unwrap();
        let before: HashMap<String, Vec<f32>> = selected
            .iter()
            .map(|k| (k.clone(), host_f32(&params[k.as_str()])))
            .collect();
        let norm = window
            .update(&mut params, None, &mut opt, 1.0, 1e-3)
            .unwrap()
            .unwrap();
        let want_norm = host_f32(g.require("total_norm").unwrap())[0] as f64;
        eprintln!("pre-clip norm {norm} vs {want_norm}");
        assert!(((norm - want_norm) / want_norm).abs() <= 1e-3);
        for k in &selected {
            let got = host_f32(&params[k.as_str()]);
            let want = host_f32(g.require(&format!("step1/{k}")).unwrap());
            let gw = &want_grads[k.as_str()];
            let gpeak = gw.iter().fold(0f32, |m, v| m.max(v.abs()));
            let (mut tight, mut loose) = (0f32, 0f32);
            for i in 0..got.len() {
                let d = ((got[i] - before[k][i]) - (want[i] - before[k][i])).abs();
                if gw[i].abs() > 1e-3 * gpeak {
                    tight = tight.max(d);
                } else {
                    loose = loose.max(d);
                }
            }
            eprintln!(
                "step1 {k}: |Δ| {tight:.3e} (well-conditioned) / {loose:.3e} (rounding-level g)"
            );
            assert!(tight <= 1e-6, "{k}: {tight}");
            assert!(loose <= 2.1e-3, "{k}: {loose}");
        }
        let after = model.predict(&params, &step.rgb).unwrap();
        let l1 = mlx_gen_iris::train::depth::depth_loss(&after, &step, loss_plan.gradient_weight)
            .unwrap()
            .item::<f32>();
        let want1 = host_f32(g.require("loss1").unwrap())[0];
        eprintln!("loss1: {l1} vs {want1}");
        assert!(((l1 - want1) / want1).abs() <= 1e-3);
    });
}

/// Eq. 5 on tensors equals the contract's host reference (both reductions, masked rows).
#[test]
fn tensor_loss_matches_the_host_reference() {
    crate::common::cpu_process(
        "depth_train_parity::tensor_loss_matches_the_host_reference",
        tensor_loss_matches_the_host_reference_body,
    );
}

fn tensor_loss_matches_the_host_reference_body() {
    on_cpu(|| {
        let g = fixture("iris_depth_train_golden.safetensors");
        let mut batch = golden_batch(&g);
        let pred = host_f32(g.require("pred0").unwrap());
        for reduction in [
            mlx_gen::gen_core::iris::train::depth::LossReduction::ValidPixels,
            mlx_gen::gen_core::iris::train::depth::LossReduction::PerImage,
        ] {
            for weight in [5.0, 0.0, 1.5] {
                let plan = DepthLossPlan {
                    reduction,
                    gradient_weight: weight,
                    ..DepthLossPlan::default()
                };
                let step = DepthStep::new(&batch, &plan);
                let p = Array::from_slice(&pred, &[2, 1, 16, 16]);
                let got = mlx_gen_iris::train::depth::depth_loss(&p, &step, weight)
                    .unwrap()
                    .item::<f32>() as f64;
                let want =
                    mlx_gen::gen_core::iris::train::depth::depth_loss_host(&pred, &batch, &plan);
                assert!(
                    ((got - want) / want).abs() <= 1e-5,
                    "{reduction:?} λ={weight}: {got} vs {want}"
                );
            }
            // a row without support contributes nothing
            batch.valid[256..].iter_mut().for_each(|v| *v = 0.0);
            batch.valid_count[1] = 0;
        }
    });
}

/// sc-25690: the depth trainer's activation checkpointing (`gradient_checkpointing`) — the
/// generation trainer's trunk / pixel regions (MLX's checkpoint transform) around the widened
/// backbone, the reducer and Eq. 5 downstream of them. f32: the gradients are the plain graph's;
/// bf16: the same loss bit for bit and gradients within the declared bf16 bound.
#[test]
fn activation_checkpointing_gradients_equal_the_plain_graph() {
    crate::common::cpu_process(
        "depth_train_parity::activation_checkpointing_gradients_equal_the_plain_graph",
        activation_checkpointing_gradients_equal_the_plain_graph_body,
    );
}

fn activation_checkpointing_gradients_equal_the_plain_graph_body() {
    use mlx_gen::gen_core::iris::train::ac::AcPolicy;
    use mlx_gen_iris::train::depth::loss_and_grads;
    use mlx_gen_iris::train::model::checkpointed_regions;
    on_cpu(|| {
        let g = fixture("iris_depth_train_golden.safetensors");
        let batch = golden_batch(&g);
        let loss_plan = DepthLossPlan::default();
        let (_, params) = setup();
        let step = DepthStep::new(&batch, &loss_plan);
        let dir = fixtures().join("tiny-depth");
        let export = TaskExport::from_dir(&dir, IrisTask::Depth, "test").unwrap();
        let prompt = EmptyPrompt::read(&export.empty_prompt_path(), &export.config).unwrap();
        let case = |compute: Dtype, ac: AcPolicy| {
            let mut model = DepthModel::new(export.config.clone(), compute, &prompt);
            model.ac = ac;
            let before = checkpointed_regions();
            let (loss, grads) = loss_and_grads(&model, &params, &step, &loss_plan, 1.0).unwrap();
            (loss, grads, checkpointed_regions() - before)
        };
        let cfg = &export.config.model;
        let regions = cfg.depth + cfg.pixel.depth;
        let (f_loss, f_none, entered) = case(Dtype::Float32, AcPolicy::None);
        assert_eq!(entered, 0, "none checkpoints no region");
        let (f_loss_ac, f_full, entered) = case(Dtype::Float32, AcPolicy::Full);
        assert_eq!(
            entered, regions,
            "full checkpoints every trunk and pixel block"
        );
        assert_eq!(f_loss.to_bits(), f_loss_ac.to_bits(), "f32 loss");
        assert_eq!(
            f_full.len(),
            params.len(),
            "every trained tensor has a gradient"
        );
        crate::common::assert_ac_f32_equal("depth", 1e-6, &f_none, &f_full);
        let (b_loss, b_none, _) = case(Dtype::Bfloat16, AcPolicy::None);
        let (b_loss_ac, b_full, entered) = case(Dtype::Bfloat16, AcPolicy::Full);
        assert_eq!(entered, regions);
        assert_eq!(b_loss.to_bits(), b_loss_ac.to_bits(), "bf16 loss");
        crate::common::assert_ac_bf16_noise_only("depth", &f_none, &b_none, &b_full);
    });
}
