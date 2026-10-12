//! sc-25689 — the curricula surface against the frozen upstream oracle
//! (`tools/dump_iris_train.py`'s curricula section → `iris_curricula_golden.safetensors`): the
//! shape policies and the covering resize, the ranged walk / shape batcher plans and resume
//! states, activation checkpointing's gradients (equal to the plain graph's, and to upstream's),
//! a non-square bucket step under both RoPE laws, the REPA / iREPA teacher tokens, losses and
//! gradients (three teacher/student grid relations), the register teacher's antialiased position
//! table, and the frozen-grid holdout loss. FP32 on the MLX CPU stream.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_gen::gen_core::iris::train::ac::AcPolicy;
use mlx_gen::gen_core::iris::train::holdout::{grid_noise, grid_timestep};
use mlx_gen::gen_core::iris::train::repa::{
    RepaPlan, RepaVariant, TeacherArch, TeacherFfn, PROJECTOR_PREFIX,
};
use mlx_gen::gen_core::iris::train::shapes::{
    preprocess_to_shape, RangedSampler, ShapeBatchSampler, ShapePolicy,
};
use mlx_gen::gen_core::iris::train::{FlowObjective, Prediction, TimestepSampler, TrainSchedule};
use mlx_gen::weights::Weights;
use mlx_gen_iris::train::model::{
    checkpointed_regions, flow_loss_per_sample, loss_and_grads_ext, RepaTargets, StepBatch,
    TrainExtras, TrainModel, Trainable,
};
use mlx_gen_iris::train::optim::Params;
use mlx_gen_iris::train::teacher::{spatial_normalize, Teacher};
use mlx_rs::{Array, Dtype};
use serde_json::Value as Json;

use crate::common::{
    assert_close, fixture, host_f32, host_i32, on_cpu, tiny_backbone, tiny_config,
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

fn meta(g: &Weights, k: &str) -> Json {
    serde_json::from_str(g.metadata(k).unwrap_or_else(|| panic!("metadata {k}"))).unwrap()
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

fn step_batch(x0: Array, noise: Array, t_idx: &[i32], y: Array, mask: &[i32]) -> StepBatch {
    let sched = TrainSchedule::new(1000, 4.0);
    let b = t_idx.len();
    let sig: Vec<f32> = t_idx.iter().map(|&t| sched.sigmas[t as usize]).collect();
    let tm: Vec<f32> = t_idx
        .iter()
        .map(|&t| sched.model_times[t as usize])
        .collect();
    let t_len = mask.len() / b;
    StepBatch {
        x0,
        noise,
        sigma: Array::from_slice(&sig, &[b as i32, 1, 1, 1]),
        t: Array::from_slice(&tm, &[b as i32]),
        states: y,
        masks: mask.chunks(t_len).map(|c| c.to_vec()).collect(),
    }
}

/// Micro-batch 0 of the training oracle.
fn mb0() -> StepBatch {
    let g = fixture(TRAIN);
    let get = |k: &str| g.require(&format!("mb0.{k}")).unwrap().clone();
    step_batch(
        get("x0"),
        get("noise"),
        &host_i32(&get("t_idx")),
        get("y"),
        &host_i32(&get("y_mask")),
    )
}

fn base_params() -> Params {
    Weights::from_file(tiny_backbone().join("model.safetensors"))
        .unwrap()
        .into_tensors()
        .into_iter()
        .map(|(k, v)| (Rc::from(k.as_str()), v.as_dtype(Dtype::Float32).unwrap()))
        .collect()
}

fn full_model(rope_aspect: &str) -> TrainModel {
    let mut cfg = tiny_config().model;
    cfg.rope_aspect = rope_aspect.into();
    TrainModel {
        cfg,
        compute: Dtype::Float32,
        trainable: Trainable::Full,
    }
}

#[test]
fn shape_policies_and_the_covering_resize_match_upstream() {
    crate::common::cpu_process(
        "curricula_parity::shape_policies_and_the_covering_resize_match_upstream",
        shape_policies_and_the_covering_resize_match_upstream_body,
    );
}

fn shape_policies_and_the_covering_resize_match_upstream_body() {
    let g = fixture(GOLDEN);
    let native: [(usize, usize); 8] = [
        (1000, 1000),
        (300, 1200),
        (2000, 500),
        (600, 300),
        (480, 640),
        (31, 47),
        (1024, 256),
        (700, 701),
    ];
    let check = |key: &str, p: &ShapePolicy| {
        let want = meta(&g, key);
        for (i, &(h, w)) in native.iter().enumerate() {
            let s = p.target(h, w);
            assert_eq!(
                json_pair(&want[i]),
                (s.height, s.width),
                "{key} target of {h}x{w}"
            );
        }
    };
    check(
        "bucket.shared21-512",
        &ShapePolicy::build("bucket", 512, 16, "shared21-512", 32, 4.0).unwrap(),
    );
    check(
        "bucket.shared21-1024",
        &ShapePolicy::build("bucket", 1024, 16, "shared21-1024", 32, 4.0).unwrap(),
    );
    check(
        "area.256.16.32.4",
        &ShapePolicy::build("area", 256, 16, "shared21-512", 32, 4.0).unwrap(),
    );
    let small = ShapePolicy::build("area", 32, 4, "shared21-512", 8, 4.0).unwrap();
    check("area.32.4.8.4", &small);
    let rgb: Vec<u8> = host_f32(g.require("shape.rgb").unwrap())
        .iter()
        .map(|&v| v as u8)
        .collect();
    let img = preprocess_to_shape(&rgb, 47, 31, &small).unwrap();
    assert_eq!(
        json_pair(&meta(&g, "shape.target")),
        (img.height, img.width)
    );
    let got = Array::from_slice(&img.chw, &[3, img.height as i32, img.width as i32]);
    // PIL's fixed-point bicubic is reproduced exactly; the [−1, 1] map is the same f32 ops.
    assert_close(
        "covering resize + crop",
        &got,
        g.require("shape.chw").unwrap(),
        1e-6,
    );
}

fn json_pair(v: &Json) -> (usize, usize) {
    (
        v[0].as_u64().unwrap() as usize,
        v[1].as_u64().unwrap() as usize,
    )
}

fn json_ranges(v: &Json) -> Vec<(usize, usize)> {
    v.as_array().unwrap().iter().map(json_pair).collect()
}

#[test]
fn ranged_walk_and_shape_batches_match_upstream() {
    crate::common::cpu_process(
        "curricula_parity::ranged_walk_and_shape_batches_match_upstream",
        ranged_walk_and_shape_batches_match_upstream_body,
    );
}

fn ranged_walk_and_shape_batches_match_upstream_body() {
    let g = fixture(GOLDEN);
    let want = meta(&g, "ranged.1300.w2");
    for r in 0..2 {
        let s = RangedSampler::new(1300, r, 2).unwrap();
        assert_eq!(s.ranges(), json_ranges(&want[r]), "rank {r}");
    }
    let state = RangedSampler::new(640, 0, 2)
        .unwrap()
        .resume_state(3, 4, None, None)
        .unwrap();
    assert_eq!(
        state,
        meta(&g, "ranged.640.w2.state"),
        "the resume state, key for key"
    );
    let w4 = meta(&g, "ranged.640.w2_to_w4");
    for r in 0..4 {
        let mut s = RangedSampler::new(640, r, 4).unwrap();
        s.load_resume_state(&state).unwrap();
        assert_eq!(s.ranges(), json_ranges(&w4[r]), "resharded rank {r}");
    }
    let keys: Vec<Option<String>> = (0..30)
        .map(|i| Some(if i % 3 == 0 { "a" } else { "b" }.to_string()))
        .collect();
    let mut sb = ShapeBatchSampler::new(RangedSampler::new(30, 0, 1).unwrap(), 2, true).unwrap();
    sb.set_epoch(1);
    let plan: Vec<Vec<usize>> = sb.batches(&keys);
    let want: Vec<Vec<usize>> = serde_json::from_value(meta(&g, "shape_batches.30.b2")).unwrap();
    assert_eq!(plan, want);
    sb.set_epoch(1);
    sb.set_start_batches(0, 2, &keys).unwrap();
    let sstate = sb.resume_state(4, 2, Some(8), &keys).unwrap();
    assert_eq!(sstate, meta(&g, "shape_batches.state"));
    let mut sb3 = ShapeBatchSampler::new(RangedSampler::new(30, 0, 1).unwrap(), 3, true).unwrap();
    sb3.set_epoch(1);
    sb3.load_resume_state(&sstate, &keys).unwrap();
    let want: Vec<Vec<usize>> =
        serde_json::from_value(meta(&g, "shape_batches.regroup_b3")).unwrap();
    assert_eq!(
        sb3.batches(&keys),
        want,
        "a micro-batch change regroups the suffix"
    );
}

fn grads_under(model: &TrainModel, params: &Params, ac: AcPolicy) -> Params {
    let extras = TrainExtras { repa: None, ac };
    loss_and_grads_ext(model, params, &mb0(), &objective(), &extras, None, 1.0)
        .unwrap()
        .2
}

/// Activation checkpointing (MLX's checkpoint transform per region) leaves every gradient equal to
/// the plain graph's — and to upstream's — for the full model.
#[test]
fn activation_checkpointing_gradients_equal_the_plain_graph() {
    crate::common::cpu_process(
        "curricula_parity::activation_checkpointing_gradients_equal_the_plain_graph",
        activation_checkpointing_gradients_equal_the_plain_graph_body,
    );
}

fn activation_checkpointing_gradients_equal_the_plain_graph_body() {
    on_cpu(|| {
        let g = fixture(GOLDEN);
        let model = full_model("isotropic");
        let params = base_params();
        let none = grads_under(&model, &params, AcPolicy::None);
        for n in SUBSET {
            assert_close(
                &format!("ac none vs torch: {n}"),
                &none[n],
                g.require(&format!("ac.none.grad.{n}")).unwrap(),
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
    });
}

#[test]
fn non_square_bucket_steps_match_upstream_under_both_rope_laws() {
    crate::common::cpu_process(
        "curricula_parity::non_square_bucket_steps_match_upstream_under_both_rope_laws",
        non_square_bucket_steps_match_upstream_under_both_rope_laws_body,
    );
}

fn non_square_bucket_steps_match_upstream_under_both_rope_laws_body() {
    on_cpu(|| {
        let g = fixture(GOLDEN);
        let get = |k: &str| g.require(&format!("ns.{k}")).unwrap().clone();
        let batch = step_batch(
            get("x0"),
            get("noise"),
            &host_i32(&get("t_idx")),
            get("y"),
            &host_i32(&get("y_mask")),
        );
        for aspect in ["isotropic", "square"] {
            let model = full_model(aspect);
            let extras = TrainExtras::default();
            let (flow, _, grads) = loss_and_grads_ext(
                &model,
                &base_params(),
                &batch,
                &objective(),
                &extras,
                None,
                1.0,
            )
            .unwrap();
            assert_close(
                &format!("{aspect} loss"),
                &Array::from_slice(&[flow], &[1]),
                &get(&format!("{aspect}.loss")),
                1e-5,
            );
            for n in SUBSET {
                assert_close(
                    &format!("{aspect} grad {n}"),
                    &grads[n],
                    &get(&format!("{aspect}.grad.{n}")),
                    1e-4,
                );
            }
        }
    });
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

fn tiny_teacher(g: &Weights, name: &str, registers: usize) -> Teacher {
    let prefix = format!("{name}.");
    let tensors: HashMap<String, Array> = g
        .keys()
        .filter_map(|k| {
            k.strip_prefix(&prefix)
                .map(|s| (s.to_string(), g.require(k).unwrap().clone()))
        })
        .collect();
    Teacher::from_tensors(tiny_arch(registers), tensors).unwrap()
}

#[test]
fn repa_and_irepa_teacher_tokens_losses_and_gradients_match_upstream() {
    crate::common::cpu_process(
        "curricula_parity::repa_and_irepa_teacher_tokens_losses_and_gradients_match_upstream",
        repa_and_irepa_teacher_tokens_losses_and_gradients_match_upstream_body,
    );
}

fn repa_and_irepa_teacher_tokens_losses_and_gradients_match_upstream_body() {
    on_cpu(|| {
        let g = fixture(GOLDEN);
        let teacher = tiny_teacher(&g, "teacher", 0);
        let batch = mb0();
        let weight: f64 = g.metadata("repa.weight").unwrap().parse().unwrap();
        let layer: usize = g.metadata("repa.layer").unwrap().parse().unwrap();
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
            let want_t = g.require(&format!("repa.{case}.teacher_tokens")).unwrap();
            assert_close(&format!("{case} teacher tokens"), &tokens, want_t, 1e-4);
            let target = if variant == RepaVariant::Irepa {
                let n = spatial_normalize(&tokens, plan.spatial_norm_gamma).unwrap();
                assert_close(
                    &format!("{case} spatial normalization"),
                    &n,
                    g.require(&format!("repa.{case}.teacher_normalized"))
                        .unwrap(),
                    1e-4,
                );
                n
            } else {
                tokens
            };
            let mut params = base_params();
            let proj_prefix = format!("repa.{case}.projector.");
            for k in g.keys() {
                if let Some(rest) = k.strip_prefix(&proj_prefix) {
                    params.insert(
                        Rc::from(format!("{PROJECTOR_PREFIX}{rest}").as_str()),
                        g.require(k).unwrap().clone(),
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
            let scalar = |v: f32| Array::from_slice(&[v], &[1]);
            assert_close(
                &format!("{case} flow"),
                &scalar(flow),
                g.require(&format!("repa.{case}.flow")).unwrap(),
                1e-5,
            );
            assert_close(
                &format!("{case} repa"),
                &scalar(repa.unwrap()),
                g.require(&format!("repa.{case}.repa")).unwrap(),
                1e-5,
            );
            let grad_prefix = format!("repa.{case}.grad.projector.");
            let mut seen = 0;
            for k in g.keys() {
                if let Some(rest) = k.strip_prefix(&grad_prefix) {
                    let native = format!("{PROJECTOR_PREFIX}{rest}");
                    assert_close(
                        &format!("{case} grad {native}"),
                        &grads[native.as_str()],
                        g.require(k).unwrap(),
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
                    g.require(&format!("repa.{case}.grad.{n}")).unwrap(),
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
    });
}

#[test]
fn the_register_teacher_resizes_its_position_table_with_antialias() {
    crate::common::cpu_process(
        "curricula_parity::the_register_teacher_resizes_its_position_table_with_antialias",
        the_register_teacher_resizes_its_position_table_with_antialias_body,
    );
}

fn the_register_teacher_resizes_its_position_table_with_antialias_body() {
    on_cpu(|| {
        let g = fixture(GOLDEN);
        let teacher = tiny_teacher(&g, "teacher_reg", 4);
        let got = teacher.tokens(&mb0().x0, (3, 3)).unwrap();
        assert_close(
            "register teacher tokens (3x3)",
            &got,
            g.require("teacher_reg.tokens_3x3").unwrap(),
            1e-4,
        );
    });
}

#[test]
fn the_frozen_holdout_grid_matches_upstreams_val_loss() {
    crate::common::cpu_process(
        "curricula_parity::the_frozen_holdout_grid_matches_upstreams_val_loss",
        the_frozen_holdout_grid_matches_upstreams_val_loss_body,
    );
}

fn the_frozen_holdout_grid_matches_upstreams_val_loss_body() {
    on_cpu(|| {
        let g = fixture(GOLDEN);
        let get = |k: &str| g.require(&format!("val.{k}")).unwrap().clone();
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
        let dit = model.dit(&base_params()).unwrap();
        let got = flow_loss_per_sample(&dit, &batch, &objective()).unwrap();
        assert_close("holdout per-sample loss", &got, &get("per_sample"), 1e-5);
    });
}

/// The release teacher (`facebook/dinov2-base` @ the pinned revision, == the hub
/// `dinov2_vitb14_pretrain.pth`, checked by the oracle) on real weights: `IRIS_REPA_TEACHER_DIR`.
#[test]
#[ignore = "needs the dinov2-base teacher resource (IRIS_REPA_TEACHER_DIR)"]
fn the_release_teacher_matches_upstream_on_real_weights() {
    // FP32 parity (the CPU stream's f32 GEMM), in a CPU-default child: no op reaches the GPU.
    crate::common::cpu_process(
        "curricula_parity::the_release_teacher_matches_upstream_on_real_weights",
        release_teacher_body,
    );
}

fn release_teacher_body() {
    let Ok(dir) = std::env::var("IRIS_REPA_TEACHER_DIR") else {
        panic!("set IRIS_REPA_TEACHER_DIR to the facebook/dinov2-base snapshot");
    };
    on_cpu(|| {
        let g = fixture("iris_repa_teacher_real.safetensors");
        let arch = TeacherArch::from_entrypoint("dinov2_vitb14").unwrap();
        let teacher = Teacher::load(std::path::Path::new(&dir), arch).unwrap();
        let img = g.require("image").unwrap();
        for (name, grid) in [("g16", (16, 16)), ("g4_match", (4, 4))] {
            let got = teacher.tokens(img, grid).unwrap();
            assert_close(
                &format!("release teacher {name}"),
                &got,
                g.require(&format!("{name}.tokens")).unwrap(),
                1e-3,
            );
        }
    });
}
