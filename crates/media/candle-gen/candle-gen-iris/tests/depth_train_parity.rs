//! The Candle depth trainer's numerics vs the torch oracle (`tools/dump_iris_depth_train.py`, the
//! MLX twin's fixture — one reference for both backends, so MLX ↔ Candle parity of the one-step
//! update is transitive through it): the frozen upstream `IrisDepth` on the miniature export, the
//! paper's normalization and Eq. 5, one `clip_grad_norm_(1.0)` + `torch.optim.AdamW` step — FP32 on
//! the Candle CPU device. Tolerances as the MLX twin's `depth_train_parity`.

use std::collections::HashMap;

use candle_gen::candle_core::{Tensor, Var};
use candle_gen::gen_core::iris::downstream::{EmptyPrompt, TaskExport};
use candle_gen::gen_core::iris::train::depth::{
    depth_loss_host, normalize_log_depth, DepthBatch, DepthLossPlan, LossReduction,
    MIN_VALID_PIXELS_FLOOR,
};
use candle_gen::gen_core::iris::train::{
    full_param_route, MuonAdjustLr, OptimizerKind, OptimizerPlan,
};
use candle_gen::gen_core::iris::IrisTask;
use candle_gen_iris::train::depth::{
    depth_loss, load_export_tensors, loss_and_grads, DepthModel, DepthStep, DepthWindow,
};
use candle_gen_iris::train::optim::{IrisOptimizer, Params};

use crate::common::{cpu, fixture, fixtures, host_f32, Fixture};

fn golden_batch(g: &Fixture) -> DepthBatch {
    let depth = host_f32(g.require("depth"));
    let mask = host_f32(g.require("mask"));
    let rgb = host_f32(g.require("rgb"));
    let want = host_f32(g.require("target"));
    let (b, h, w) = (2usize, 16usize, 16usize);
    let (mut target, mut valid, mut counts) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..b {
        let d = &depth[i * h * w..(i + 1) * h * w];
        let v: Vec<bool> = mask[i * h * w..(i + 1) * h * w]
            .iter()
            .map(|&m| m > 0.5)
            .collect();
        target.extend(
            normalize_log_depth(d, &v, &DepthLossPlan::default(), MIN_VALID_PIXELS_FLOOR).unwrap(),
        );
        valid.extend(v.iter().map(|&x| if x { 1.0f32 } else { 0.0 }));
        counts.push(v.iter().filter(|&&x| x).count());
    }
    let max = target
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
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

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn one_step_matches_the_torch_oracle() {
    let g = fixture("iris_depth_train_golden.safetensors");
    let batch = golden_batch(&g);
    let loss_plan = DepthLossPlan::default();
    let device = cpu();
    let dir = fixtures().join("tiny-depth");
    let export = TaskExport::from_dir(&dir, IrisTask::Depth, "test").unwrap();
    let prompt = EmptyPrompt::read(&export.empty_prompt_path(), &export.config).unwrap();
    let params: Params = load_export_tensors(&dir, &device)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, Var::from_tensor(&v).unwrap()))
        .collect();
    let model = DepthModel::new(
        export.config,
        candle_gen::candle_core::DType::F32,
        &prompt,
        &device,
    )
    .unwrap();
    let step = DepthStep::new(&batch, &loss_plan, &device).unwrap();

    let snap = candle_gen_iris::train::optim::snapshot(&params);
    let pred = model.predict(&snap, &step.rgb).unwrap();
    let rel = peak_rel(&host_f32(&pred), &host_f32(g.require("pred0")));
    eprintln!("pred0: {rel:.3e} of peak");
    assert!(rel <= 1e-4);

    let (loss, grads) = loss_and_grads(&model, &params, &step, &loss_plan, 1.0).unwrap();
    let want = host_f32(g.require("loss0"))[0];
    eprintln!("loss0: {loss} vs {want}");
    assert!(((loss - want) / want).abs() <= 1e-4);
    let keys: Vec<String> = serde_json::from_str(g.meta("grad_norm_keys")).unwrap();
    assert_eq!(keys.len(), params.len());
    let norms = host_f32(g.require("grad_norms"));
    for (k, want) in keys.iter().zip(&norms) {
        let n = host_f32(&grads[k])
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (n - want).abs() <= (1e-3 * want).max(1e-7),
            "{k}: |g| {n} vs {want}"
        );
    }
    let selected: Vec<String> = serde_json::from_str(g.meta("selected")).unwrap();
    let mut want_grads: HashMap<String, Vec<f32>> = HashMap::new();
    for k in &selected {
        let want = host_f32(g.require(&format!("grad/{k}")));
        let rel = peak_rel(&host_f32(&grads[k]), &want);
        eprintln!("grad {k}: {rel:.3e} of peak");
        assert!(rel <= 1e-3, "{k}");
        want_grads.insert(k.clone(), want);
    }

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
        .map(|k| (k.clone(), host_f32(params[k].as_tensor())))
        .collect();
    let mut window = DepthWindow::new(1);
    window.micro(&model, &params, &step, &loss_plan).unwrap();
    let norm = window
        .update(&params, None, &mut opt, 1.0, 1e-3)
        .unwrap()
        .unwrap();
    let want_norm = host_f32(g.require("total_norm"))[0] as f64;
    eprintln!("pre-clip norm {norm} vs {want_norm}");
    assert!(((norm - want_norm) / want_norm).abs() <= 1e-3);
    for k in &selected {
        let got = host_f32(params[k].as_tensor());
        let want = host_f32(g.require(&format!("step1/{k}")));
        let gw = &want_grads[k];
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
        eprintln!("step1 {k}: |Δ| {tight:.3e} / {loose:.3e}");
        assert!(tight <= 1e-6, "{k}: {tight}");
        assert!(loose <= 2.1e-3, "{k}: {loose}");
    }
    let after = model
        .predict(&candle_gen_iris::train::optim::snapshot(&params), &step.rgb)
        .unwrap();
    let l1 = depth_loss(&after, &step, loss_plan.gradient_weight)
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    let want1 = host_f32(g.require("loss1"))[0];
    eprintln!("loss1: {l1} vs {want1}");
    assert!(((l1 - want1) / want1).abs() <= 1e-3);

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED depth_train_parity::one_step_matches_the_torch_oracle"
    );
}

#[test]
fn tensor_loss_matches_the_host_reference() {
    let g = fixture("iris_depth_train_golden.safetensors");
    let mut batch = golden_batch(&g);
    let pred = host_f32(g.require("pred0"));
    let device = cpu();
    for reduction in [LossReduction::ValidPixels, LossReduction::PerImage] {
        for weight in [5.0, 0.0, 1.5] {
            let plan = DepthLossPlan {
                reduction,
                gradient_weight: weight,
                ..DepthLossPlan::default()
            };
            let step = DepthStep::new(&batch, &plan, &device).unwrap();
            let p = Tensor::from_slice(&pred, (2, 1, 16, 16), &device).unwrap();
            let got = depth_loss(&p, &step, weight)
                .unwrap()
                .to_vec0::<f32>()
                .unwrap() as f64;
            let want = depth_loss_host(&pred, &batch, &plan);
            assert!(
                ((got - want) / want).abs() <= 1e-5,
                "{reduction:?} λ={weight}: {got} vs {want}"
            );
        }
        batch.valid[256..].iter_mut().for_each(|v| *v = 0.0);
        batch.valid_count[1] = 0;
    }
}

/// sc-25690: the depth trainer's activation checkpointing (`gradient_checkpointing`) — the
/// generation trainer's trunk / pixel regions around the widened backbone. f32: the gradients are
/// the plain graph's; bf16: the same loss bit for bit and gradients within the declared bf16 bound.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn activation_checkpointing_gradients_equal_the_plain_graph() {
    use candle_gen::candle_core::DType;
    use candle_gen::gen_core::iris::train::ac::AcPolicy;
    use candle_gen_iris::train::model::checkpointed_regions;
    let g = fixture("iris_depth_train_golden.safetensors");
    let batch = golden_batch(&g);
    let loss_plan = DepthLossPlan::default();
    let device = cpu();
    let dir = fixtures().join("tiny-depth");
    let export = TaskExport::from_dir(&dir, IrisTask::Depth, "test").unwrap();
    let prompt = EmptyPrompt::read(&export.empty_prompt_path(), &export.config).unwrap();
    let params: Params = load_export_tensors(&dir, &device)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, Var::from_tensor(&v).unwrap()))
        .collect();
    let step = DepthStep::new(&batch, &loss_plan, &device).unwrap();
    let case = |compute: DType, ac: AcPolicy| {
        let mut model = DepthModel::new(export.config.clone(), compute, &prompt, &device).unwrap();
        model.ac = ac;
        let before = checkpointed_regions();
        let out = loss_and_grads(&model, &params, &step, &loss_plan, 1.0).unwrap();
        (out.0, out.1, checkpointed_regions() - before)
    };
    let (f_loss, f_none, entered) = case(DType::F32, AcPolicy::None);
    assert_eq!(entered, 0, "none checkpoints no region");
    let (f_loss_ac, f_full, entered) = case(DType::F32, AcPolicy::Full);
    let cfg = &export.config.model;
    assert_eq!(
        entered,
        cfg.depth + cfg.pixel.depth,
        "full checkpoints every trunk and pixel block"
    );
    assert_eq!(f_loss.to_bits(), f_loss_ac.to_bits(), "f32 loss");
    crate::common::assert_ac_f32_equal("depth", 1e-6, &f_none, &f_full);
    let (b_loss, b_none, _) = case(DType::BF16, AcPolicy::None);
    let (b_loss_ac, b_full, _) = case(DType::BF16, AcPolicy::Full);
    assert_eq!(b_loss.to_bits(), b_loss_ac.to_bits(), "bf16 loss");
    crate::common::assert_ac_bf16_noise_only("depth", &f_none, &b_none, &b_full);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_parity::activation_checkpointing_gradients_equal_the_plain_graph");
}
