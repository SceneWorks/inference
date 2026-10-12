//! sc-25689 — the curricula surface against the frozen upstream oracle on the **Candle** backend
//! (the MLX twin's `curricula_parity`, on the same committed fixture
//! `iris_curricula_golden.safetensors`): activation checkpointing's gradients (the segmented
//! recompute equals the plain graph and upstream), a non-square bucket step under both RoPE laws,
//! the REPA / iREPA teacher tokens, losses and gradients (three teacher/student grid relations),
//! the register teacher's antialiased position table, and the frozen-grid holdout loss. The
//! backend-neutral shape / sampler arithmetic is checked once, by the MLX twin and gen-core. FP32
//! on the Candle CPU backend.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Tensor, Var};
use candle_gen::gen_core::iris::train::ac::AcPolicy;
use candle_gen::gen_core::iris::train::holdout::{grid_noise, grid_timestep};
use candle_gen::gen_core::iris::train::repa::{
    RepaPlan, RepaVariant, TeacherArch, TeacherFfn, PROJECTOR_PREFIX,
};
use candle_gen::gen_core::iris::train::{
    FlowObjective, Prediction, TimestepSampler, TrainSchedule,
};
use candle_gen_iris::train::model::{
    checkpointed_regions, flow_loss_per_sample, loss_and_grads_ext, RepaTargets, StepBatch,
    TrainExtras, TrainModel, Trainable,
};
use candle_gen_iris::train::optim::{Params, Tensors};
use candle_gen_iris::train::teacher::{spatial_normalize, Teacher};
use serde_json::Value as Json;

use crate::common::{
    assert_close, cpu, fixture, host_f32, host_i32, tiny_backbone, tiny_config, Fixture,
};

const GOLDEN: &str = "iris_curricula_golden.safetensors";
const TRAIN: &str = "iris_train_golden.safetensors";
const SUBSET: [&str; 10] = [
    "blocks.0.attn.q_proj_x.weight",
    "blocks.2.mlp.w1.weight",
    "modulation_cores.adaln_img.weight",
    "y_embedder.layer_blocks.0.attn.qkv.weight",
    "pixel_blocks.0.attn.qkv.weight",
    "s_embedder.proj.weight",
    "blocks.1.adaln_txt.bias",
    "y_pos_embedding",
    "final_layer.linear.weight",
    "blocks.0.norm_x1.weight",
];

fn meta(g: &Fixture, k: &str) -> Json {
    serde_json::from_str(g.meta(k)).unwrap()
}

fn f32t(t: &Tensor) -> Tensor {
    t.to_dtype(DType::F32).unwrap()
}

fn objective() -> FlowObjective {
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
        prediction: Prediction::Velocity,
        x_pred_sigma_min: 0.05,
    }
}

fn step_batch(x0: Tensor, noise: Tensor, t_idx: &[i32], y: Tensor, mask: &[i32]) -> StepBatch {
    let sched = TrainSchedule::new(1000, 4.0);
    let b = t_idx.len();
    let sig: Vec<f32> = t_idx.iter().map(|&t| sched.sigmas[t as usize]).collect();
    let tm: Vec<f32> = t_idx
        .iter()
        .map(|&t| sched.model_times[t as usize])
        .collect();
    let t_len = mask.len() / b;
    StepBatch {
        x0: f32t(&x0),
        noise: f32t(&noise),
        sigma: Tensor::from_vec(sig, (b, 1, 1, 1), &cpu()).unwrap(),
        t: Tensor::from_vec(tm, b, &cpu()).unwrap(),
        states: f32t(&y),
        masks: mask.chunks(t_len).map(|c| c.to_vec()).collect(),
    }
}

/// Micro-batch 0 of the training oracle.
fn mb0() -> StepBatch {
    let g = fixture(TRAIN);
    let get = |k: &str| g.require(&format!("mb0.{k}")).clone();
    step_batch(
        get("x0"),
        get("noise"),
        &host_i32(&get("t_idx")),
        get("y"),
        &host_i32(&get("y_mask")),
    )
}

fn base_tensors() -> HashMap<String, Tensor> {
    candle_gen::candle_core::safetensors::load(tiny_backbone().join("model.safetensors"), &cpu())
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, f32t(&v)))
        .collect()
}

fn base_params() -> Params {
    base_tensors()
        .into_iter()
        .map(|(k, v)| (k, Var::from_tensor(&v).unwrap()))
        .collect()
}

fn full_model(rope_aspect: &str) -> TrainModel {
    let mut cfg = tiny_config().model;
    cfg.rope_aspect = rope_aspect.into();
    TrainModel {
        cfg,
        compute: DType::F32,
        device: cpu(),
        trainable: Trainable::Full,
    }
}

fn json_pair(v: &Json) -> (usize, usize) {
    (
        v[0].as_u64().unwrap() as usize,
        v[1].as_u64().unwrap() as usize,
    )
}

fn grads_under(model: &TrainModel, params: &Params, ac: AcPolicy) -> Tensors {
    let extras = TrainExtras { repa: None, ac };
    loss_and_grads_ext(model, params, &mb0(), &objective(), &extras, None, 1.0)
        .unwrap()
        .2
}

/// Activation checkpointing (tape-free region forwards replayed on the tape by the segmented
/// backward) leaves every gradient equal to the plain graph's — and to upstream's.
#[test]
fn activation_checkpointing_gradients_equal_the_plain_graph() {
    let g = fixture(GOLDEN);
    let model = full_model("isotropic");
    let params = base_params();
    let none = grads_under(&model, &params, AcPolicy::None);
    for n in SUBSET {
        assert_close(
            &format!("ac none vs torch: {n}"),
            &none[n],
            &f32t(g.require(&format!("ac.none.grad.{n}"))),
            1e-4,
        );
    }
    // `none` enters no checkpointed region; `full` / `selective_op` enter every region (the same
    // set natively); `selective_layer` with stride 2 a strict, non-empty subset. (Stride 1 is the
    // `full` region set — not a separate case.)
    let before = checkpointed_regions();
    let none_again = grads_under(&model, &params, AcPolicy::None);
    assert_eq!(
        checkpointed_regions() - before,
        0,
        "none checkpoints no region"
    );
    assert_eq!(none_again.len(), none.len());
    let mut regions = Vec::new();
    for ac in [
        AcPolicy::Full,
        AcPolicy::SelectiveOp,
        AcPolicy::SelectiveLayer { every: 2 },
    ] {
        let before = checkpointed_regions();
        let got = grads_under(&model, &params, ac);
        regions.push(checkpointed_regions() - before);
        assert_eq!(got.len(), none.len());
        let mut worst = 0f32;
        for (k, v) in &none {
            let d = host_f32(&got[k])
                .iter()
                .zip(host_f32(v))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            worst = worst.max(d);
        }
        eprintln!("{}: max |Δgrad| vs none = {worst:.3e}", ac.name());
        assert!(worst <= 1e-6, "{}: {worst}", ac.name());
    }
    eprintln!("checkpointed regions (full, selective_op, selective_layer/2): {regions:?}");
    // one region per trunk block and per pixel block; stride 2 keeps blocks 0, 2, … of each stack
    let (trunk, pix) = (model.cfg.depth, model.cfg.pixel.depth);
    let full = trunk + pix;
    let strided = trunk.div_ceil(2) + pix.div_ceil(2);
    assert!(strided > 0 && strided < full);
    assert_eq!(
        regions,
        vec![full, full, strided],
        "regions entered per policy"
    );
}

#[test]
fn non_square_bucket_steps_match_upstream_under_both_rope_laws() {
    let g = fixture(GOLDEN);
    let get = |k: &str| g.require(&format!("ns.{k}")).clone();
    let batch = step_batch(
        get("x0"),
        get("noise"),
        &host_i32(&get("t_idx")),
        get("y"),
        &host_i32(&get("y_mask")),
    );
    for aspect in ["isotropic", "square"] {
        let model = full_model(aspect);
        let (flow, _, grads) = loss_and_grads_ext(
            &model,
            &base_params(),
            &batch,
            &objective(),
            &TrainExtras::default(),
            None,
            1.0,
        )
        .unwrap();
        assert_close(
            &format!("{aspect} loss"),
            &Tensor::new(&[flow], &cpu()).unwrap(),
            &f32t(&get(&format!("{aspect}.loss"))),
            1e-5,
        );
        for n in SUBSET {
            assert_close(
                &format!("{aspect} grad {n}"),
                &grads[n],
                &f32t(&get(&format!("{aspect}.grad.{n}"))),
                1e-4,
            );
        }
    }
}

fn tiny_arch(registers: usize) -> TeacherArch {
    TeacherArch {
        entrypoint: "miniature".into(),
        embed_dim: 32,
        depth: 2,
        num_heads: 2,
        ffn: TeacherFfn::Mlp,
        ffn_hidden: 128,
        patch_size: 14,
        pos_grid: 37,
        num_register_tokens: registers,
        interpolate_antialias: registers > 0,
        interpolate_offset: if registers > 0 { 0.0 } else { 0.1 },
    }
}

fn tiny_teacher(g: &Fixture, name: &str, registers: usize) -> Teacher {
    let prefix = format!("{name}.");
    let tensors: HashMap<String, Tensor> = g
        .tensors
        .iter()
        .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|s| (s.to_string(), f32t(v))))
        .collect();
    Teacher::from_tensors(tiny_arch(registers), tensors).unwrap()
}

#[test]
fn repa_and_irepa_teacher_tokens_losses_and_gradients_match_upstream() {
    let g = fixture(GOLDEN);
    let teacher = tiny_teacher(&g, "teacher", 0);
    let batch = mb0();
    let weight: f64 = g.meta("repa.weight").parse().unwrap();
    let layer: usize = g.meta("repa.layer").parse().unwrap();
    for case in ["repa_t2", "irepa_match", "repa_t6"] {
        let m = meta(&g, &format!("repa.{case}"));
        let variant = RepaVariant::parse(m["variant"].as_str().unwrap()).unwrap();
        let plan = RepaPlan {
            weight,
            variant,
            spatial_norm_gamma: m
                .get("spatial_norm_gamma")
                .and_then(Json::as_f64)
                .unwrap_or(1.0),
            teacher: tiny_arch(0),
            teacher_dim: 32,
            proj_hidden_dim: 24,
            teacher_image_size: m["teacher_image_size"].as_u64().unwrap() as usize,
            teacher_match_student: m["teacher_match_student"].as_bool().unwrap(),
            layer,
            student_dim: 64,
            teacher_weights: None,
        };
        let student_grid = (4, 4);
        let teacher_grid = plan.teacher_grid(student_grid);
        assert_eq!(json_pair(&m["grid"]), teacher_grid, "{case} teacher grid");
        let tokens = teacher.tokens(&batch.x0, teacher_grid).unwrap();
        assert_close(
            &format!("{case} teacher tokens"),
            &tokens,
            &f32t(g.require(&format!("repa.{case}.teacher_tokens"))),
            1e-4,
        );
        let target = if variant == RepaVariant::Irepa {
            let n = spatial_normalize(&tokens, plan.spatial_norm_gamma).unwrap();
            assert_close(
                &format!("{case} spatial normalization"),
                &n,
                &f32t(g.require(&format!("repa.{case}.teacher_normalized"))),
                1e-4,
            );
            n
        } else {
            tokens
        };
        let mut params = base_params();
        let proj_prefix = format!("repa.{case}.projector.");
        for (k, v) in &g.tensors {
            if let Some(rest) = k.strip_prefix(&proj_prefix) {
                params.insert(
                    format!("{PROJECTOR_PREFIX}{rest}"),
                    Var::from_tensor(&f32t(v)).unwrap(),
                );
            }
        }
        let extras = TrainExtras {
            repa: Some(plan.clone()),
            ac: AcPolicy::None,
        };
        let targets = RepaTargets {
            teacher: target,
            student_grid,
            teacher_grid,
        };
        let (flow, repa, grads) = loss_and_grads_ext(
            &full_model("isotropic"),
            &params,
            &batch,
            &objective(),
            &extras,
            Some(&targets),
            1.0,
        )
        .unwrap();
        let scalar = |v: f32| Tensor::new(&[v], &cpu()).unwrap();
        assert_close(
            &format!("{case} flow"),
            &scalar(flow),
            &f32t(g.require(&format!("repa.{case}.flow"))),
            1e-5,
        );
        assert_close(
            &format!("{case} repa"),
            &scalar(repa.unwrap()),
            &f32t(g.require(&format!("repa.{case}.repa"))),
            1e-5,
        );
        let grad_prefix = format!("repa.{case}.grad.projector.");
        let mut seen = 0;
        for (k, v) in &g.tensors {
            if let Some(rest) = k.strip_prefix(&grad_prefix) {
                let native = format!("{PROJECTOR_PREFIX}{rest}");
                assert_close(
                    &format!("{case} grad {native}"),
                    &grads[&native],
                    &f32t(v),
                    1e-4,
                );
                seen += 1;
            }
        }
        assert!(seen >= 2, "{case}: projector gradients compared");
        for n in SUBSET {
            assert_close(
                &format!("{case} grad {n}"),
                &grads[n],
                &f32t(g.require(&format!("repa.{case}.grad.{n}"))),
                1e-4,
            );
        }
        // The REPA capture inside checkpointed regions: the same losses and gradients.
        if case == "repa_t2" {
            for ac in [AcPolicy::Full, AcPolicy::SelectiveLayer { every: 2 }] {
                let before = checkpointed_regions();
                let (f2, r2, g2) = loss_and_grads_ext(
                    &full_model("isotropic"),
                    &params,
                    &batch,
                    &objective(),
                    &TrainExtras {
                        repa: Some(plan.clone()),
                        ac,
                    },
                    Some(&targets),
                    1.0,
                )
                .unwrap();
                let (trunk, pix) = (tiny_config().model.depth, tiny_config().model.pixel.depth);
                let want = match ac {
                    AcPolicy::Full => trunk + pix,
                    _ => trunk.div_ceil(2) + pix.div_ceil(2),
                };
                assert_eq!(
                    checkpointed_regions() - before,
                    want,
                    "{}: regions entered",
                    ac.name()
                );
                assert!(
                    (f2 - flow).abs() <= 1e-6,
                    "{}: flow {f2} vs {flow}",
                    ac.name()
                );
                let (ra, rb) = (r2.unwrap(), repa.unwrap());
                assert!((ra - rb).abs() <= 1e-6, "{}: repa {ra} vs {rb}", ac.name());
                assert_eq!(g2.len(), grads.len());
                let mut worst = 0f32;
                for (k, v) in &grads {
                    let d = host_f32(&g2[k])
                        .iter()
                        .zip(host_f32(v))
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    worst = worst.max(d);
                }
                eprintln!("repa + {}: max |Δgrad| vs none = {worst:.3e}", ac.name());
                assert!(worst <= 1e-6, "repa + {}: {worst}", ac.name());
            }
        }
    }
}

#[test]
fn the_register_teacher_resizes_its_position_table_with_antialias() {
    let g = fixture(GOLDEN);
    let teacher = tiny_teacher(&g, "teacher_reg", 4);
    let got = teacher.tokens(&mb0().x0, (3, 3)).unwrap();
    assert_close(
        "register teacher tokens (3x3)",
        &got,
        &f32t(g.require("teacher_reg.tokens_3x3")),
        1e-4,
    );
}

#[test]
fn the_frozen_holdout_grid_matches_upstreams_val_loss() {
    let g = fixture(GOLDEN);
    let get = |k: &str| g.require(&format!("val.{k}")).clone();
    let t_idx = host_i32(&get("t_idx"));
    let want_noise = host_f32(&get("noise"));
    let numel = want_noise.len() / t_idx.len();
    for (k, &t) in t_idx.iter().enumerate() {
        assert_eq!(grid_timestep(k, 1000), t as usize);
        let n = grid_noise(k, numel);
        let max = n
            .iter()
            .zip(&want_noise[k * numel..(k + 1) * numel])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max <= 1e-6, "holdout noise {k}: {max}");
    }
    let batch = step_batch(
        get("x0"),
        get("noise"),
        &t_idx,
        get("y"),
        &host_i32(&get("y_mask")),
    );
    let model = full_model("isotropic");
    let dit = model.dit(&base_tensors().into_iter().collect()).unwrap();
    let got = flow_loss_per_sample(&dit, &batch, &objective()).unwrap();
    assert_close(
        "holdout per-sample loss",
        &got,
        &f32t(&get("per_sample")),
        1e-5,
    );
}

/// The release teacher (`facebook/dinov2-base` @ the pinned revision) on real weights:
/// `IRIS_REPA_TEACHER_DIR` (the CUDA real-weights lane).
#[test]
#[ignore = "needs the dinov2-base teacher resource (IRIS_REPA_TEACHER_DIR)"]
fn the_release_teacher_matches_upstream_on_real_weights() {
    let Ok(dir) = std::env::var("IRIS_REPA_TEACHER_DIR") else {
        panic!("set IRIS_REPA_TEACHER_DIR to the facebook/dinov2-base snapshot");
    };
    let g = fixture("iris_repa_teacher_real.safetensors");
    let arch = TeacherArch::from_entrypoint("dinov2_vitb14").unwrap();
    let device = crate::common::device();
    let teacher = Teacher::load(std::path::Path::new(&dir), arch, &device).unwrap();
    let img = f32t(g.require("image")).to_device(&device).unwrap();
    // Peak-relative. The 4×4 case (a 37² position table resampled to 4², twelve fp32 blocks) is
    // the sensitive one: measured max |Δ| / peak 4.2e-4 on the CPU (MLX CPU 1.9e-4), 1.2e-3 on
    // CUDA (mean |Δ| 9.9e-5 there — accumulation order and the device's transcendental kernels,
    // not a systematic offset), so the device bound is 2e-3.
    let tol = if device.is_cpu() { 1e-3 } else { 2e-3 };
    for (name, grid) in [("g16", (16, 16)), ("g4_match", (4, 4))] {
        let got = teacher
            .tokens(&img, grid)
            .unwrap()
            .to_device(&cpu())
            .unwrap();
        assert_close(
            &format!("release teacher {name}"),
            &got,
            &f32t(g.require(&format!("{name}.tokens"))),
            tol,
        );
    }
}
