//! sc-25689 — the curricula surface end to end on the miniature snapshot through the catalog's
//! `load_trainer`, on the **Candle** trainer (the MLX twin's `curricula_lifecycle`, case for
//! case): the official stage presets as editable configs (stage 1 at 256 with REPA on a
//! provisioned teacher → stage 2 at 512 continuing as a `new_phase`, lineage stamped and checked),
//! the `area` shape policy's per-shape batches, activation checkpointing leaving the run
//! unchanged, holdout validation metrics, fp16 (a CUDA-only precision on Candle: trained on a CUDA
//! build, refused typed on CPU) and `resume_optimizer: core` starting a new projector cold. On the
//! build's device (CPU here).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::iris::train::repa::{
    TeacherArch, TEACHER_COMPONENT, TEACHER_WEIGHTS_FILE,
};
use candle_gen::gen_core::iris::train::upstream::stage_preset_config;
use candle_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, read_checkpoint_state, HostRng, OPTIONS_KEY,
};
use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use candle_gen::gen_core::{
    LoadSpec, LrSchedule, Trainer, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};
use serde_json::{json, Value as Json};

use crate::common::{cpu, device, tiny_backbone, tiny_text_encoder};

const ID: &str = "iris_3b";

/// The MLX twin runs each case in a CPU-default child process; Candle's device is explicit.
fn cpu_process(_test: &str, body: fn()) {
    body()
}

fn on_cpu<T>(f: impl FnOnce() -> T) -> T {
    f()
}

fn trainer_with(teacher: Option<&Path>) -> Box<dyn Trainer> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    if let Some(t) = teacher {
        spec.components.insert(
            TEACHER_COMPONENT.into(),
            WeightsSource::Dir(t.to_path_buf()),
        );
    }
    candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(ID, &spec)
        .expect("the tiny snapshot loads as a trainer")
}

const CAPTIONS: [&str; 5] = [
    "a red fox in the snow",
    "golden hour",
    "a fox , snow",
    "snow , golden fox",
    "a golden snow fox",
];

/// Five images of mixed aspect ratios; the last is marked `split: holdout`.
fn dataset(dir: &Path) -> Vec<TrainingItem> {
    let sizes = [(24u32, 16u32), (16, 24), (20, 20), (32, 16), (18, 18)];
    CAPTIONS
        .iter()
        .zip(sizes)
        .enumerate()
        .map(|(i, (c, (w, h)))| {
            let img = image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([
                    (x * 9 + i as u32 * 40) as u8,
                    (y * 7 + 30) as u8,
                    ((x + y) * 5 + i as u32 * 70) as u8,
                ])
            });
            let path = dir.join(format!("img{i}.png"));
            img.save(&path).unwrap();
            let mut item = TrainingItem::captioned(path, c.to_string());
            if i == CAPTIONS.len() - 1 {
                item.model_options.insert("split".into(), json!("holdout"));
            }
            item
        })
        .collect()
}

/// A provisioned `dinov2_vits14` teacher resource with random weights (the real teacher's parity
/// is `curricula_parity`'s; this exercises the resource + training path). The MLX twin writes the
/// same values from the same seed.
fn random_teacher(dir: &Path) -> PathBuf {
    let arch = TeacherArch::from_entrypoint("dinov2_vits14").unwrap();
    let mut rng = HostRng::new(7);
    let tensors: Vec<(String, Tensor)> = arch
        .tensor_shapes()
        .into_iter()
        .map(|(k, shape)| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = rng.normals_f32(n).iter().map(|x| x * 0.02).collect();
            let a = Tensor::from_vec(v, shape, &cpu())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            (k, a)
        })
        .collect();
    std::fs::create_dir_all(dir).unwrap();
    candle_gen::candle_core::safetensors::save(
        &tensors.into_iter().collect(),
        dir.join(TEACHER_WEIGHTS_FILE),
    )
    .unwrap();
    dir.to_path_buf()
}

fn base() -> TrainingConfig {
    TrainingConfig {
        learning_rate: 1e-3,
        steps: 2,
        batch_size: 2,
        resolution: 16,
        save_every: 1,
        seed: 9,
        train_dtype: "f32".into(),
        lr_scheduler: LrSchedule::Constant,
        full_finetune: true,
        optimizer: "adamw".into(),
        ..Default::default()
    }
}

fn opts(cfg: &mut TrainingConfig) -> &mut serde_json::Map<String, Json> {
    cfg.model_options
        .entry(OPTIONS_KEY)
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .unwrap()
}

/// An official preset edited for the miniature backbone: its architecture / text-encoder pins and
/// the cluster topology (8-way hybrid FSDP, 1024 global batch, torch.compile) removed, the
/// resolution and horizon shrunk.
fn tiny_preset(id: &str, resolution: usize) -> TrainingConfig {
    let mut cfg = stage_preset_config(id, base()).unwrap();
    cfg.resolution = resolution as u32;
    cfg.batch_size = 2;
    cfg.steps = 2;
    cfg.save_every = 1;
    cfg.learning_rate = 1e-3;
    cfg.train_dtype = "f32".into();
    let o = opts(&mut cfg);
    for k in [
        "architecture",
        "text_encoder",
        "dist",
        "expected_global_batch",
        "perf",
    ] {
        o.remove(k);
    }
    o.insert("log_every".into(), json!(1));
    cfg
}

fn request(items: &[TrainingItem], cfg: TrainingConfig, out: &Path) -> TrainingRequest {
    TrainingRequest {
        items: items.to_vec(),
        config: cfg,
        output_dir: out.to_path_buf(),
        file_name: "curricula.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

fn train(t: &mut dyn Trainer, req: &TrainingRequest) -> candle_gen::gen_core::Result<u32> {
    on_cpu(|| t.train(req, &mut |_| {}).map(|o| o.steps))
}

fn metrics(out: &Path) -> Vec<Json> {
    let p = checkpoint_root(out, "curricula.safetensors").join("metrics.jsonl");
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn latest(out: &Path) -> PathBuf {
    latest_checkpoint(&checkpoint_root(out, "curricula.safetensors")).unwrap()
}

#[test]
fn stage_presets_chain_256_to_512_with_repa_and_a_new_phase() {
    cpu_process(
        "curricula_lifecycle::stage_presets_chain_256_to_512_with_repa_and_a_new_phase",
        stage_presets_chain_256_to_512_with_repa_and_a_new_phase_body,
    );
}

fn stage_presets_chain_256_to_512_with_repa_and_a_new_phase_body() {
    let (data, out, tdir) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = dataset(data.path());
    let teacher = random_teacher(&tdir.path().join("dinov2_vits14"));

    // Stage 1: the 256 preset (fixed square, square RoPE, REPA 0.5) at the miniature's 16.
    let mut s1 = tiny_preset("stage1_256", 16);
    {
        let o = opts(&mut s1);
        assert_eq!(o["stage"], json!("stage1_256"));
        let repa = o.get_mut("repa").unwrap().as_object_mut().unwrap();
        assert_eq!(
            repa["weight"],
            json!(0.5),
            "the preset carries stage 1's REPA"
        );
        repa.insert("teacher".into(), json!("dinov2_vits14"));
        repa.insert("teacher_dim".into(), json!(384));
        repa.insert("teacher_image_size".into(), json!(56));
        repa.insert("proj_hidden_dim".into(), json!(32));
        repa.insert("layer".into(), json!(1));
    }
    let mut t = trainer_with(Some(&teacher));
    let out1 = out.path().join("s1");
    assert_eq!(train(t.as_mut(), &request(&items, s1, &out1)).unwrap(), 2);
    let ck1 = latest(&out1);
    let st1 = read_checkpoint_state(&ck1).unwrap();
    assert_eq!(st1.stage.as_deref(), Some("stage1_256"));
    assert!(st1.repa_projector.is_some(), "stage 1 trained a projector");
    assert!(ck1.join("repa_projector.safetensors").is_file());
    let m1 = metrics(&out1);
    assert!(
        m1.iter()
            .any(|r| r["kind"] == "step" && r.get("loss/repa_raw").is_some()),
        "REPA losses are logged: {m1:?}"
    );

    // Stage 2: the 512 preset (bucket → `area` at the miniature's 32, isotropic RoPE, no REPA)
    // continuing stage 1's checkpoint as a new phase.
    let mut s2 = tiny_preset("stage2_512", 32);
    // Each aspect is its own `area` shape here, so batches of 1; `max_steps` is the absolute
    // horizon (upstream's 365k → 530k), so stage 2 runs to step 4.
    s2.batch_size = 1;
    s2.steps = 4;
    {
        let o = opts(&mut s2);
        assert_eq!(o["parent_stage"], json!("stage1_256"));
        assert_eq!(o["resume_data_policy"], json!("new_phase"));
        o.insert("shape_policy".into(), json!("area"));
        o.insert("shape_align".into(), json!(8));
    }
    opts(&mut s2).insert("resume_from".into(), json!(ck1));
    let out2 = out.path().join("s2");
    let mut t = trainer_with(None);
    // Single-process (ddp) stage 2 drops stage 1's projector: a full optimizer resume refuses the
    // changed routing, as upstream's replicated load does; `core` resumes without it (an FSDP2
    // cluster run drops it as upstream's sharded load does).
    let e = train(t.as_mut(), &request(&items, s2.clone(), &out2))
        .expect_err("a full ddp resume refuses the dropped projector");
    assert!(e.to_string().contains("resume_optimizer: core"), "{e}");
    opts(&mut s2).insert("resume_optimizer".into(), json!("core"));
    assert_eq!(
        train(t.as_mut(), &request(&items, s2.clone(), &out2)).unwrap(),
        4
    );
    let st2 = read_checkpoint_state(&latest(&out2)).unwrap();
    assert_eq!(
        (st2.stage.as_deref(), st2.parent_stage.as_deref()),
        (Some("stage2_512"), Some("stage1_256"))
    );
    assert_eq!(st2.step, 4, "the step count continues across the phase");
    assert!(st2.repa_projector.is_none(), "stage 2 drops the projector");

    // Lineage: stage 3 does not continue from stage 1.
    let mut s3 = tiny_preset("stage3_1024", 32);
    s3.batch_size = 1;
    s3.steps = 6;
    opts(&mut s3).insert("shape_policy".into(), json!("area"));
    opts(&mut s3).insert("shape_align".into(), json!(8));
    opts(&mut s3).insert("resume_from".into(), json!(ck1));
    let e = trainer_with(None)
        .validate(&request(&items, s3.clone(), &out.path().join("s3")))
        .err()
        .or_else(|| {
            train(
                trainer_with(None).as_mut(),
                &request(&items, s3, &out.path().join("s3")),
            )
            .err()
        })
        .expect("stage 3 refuses a stage 1 parent");
    assert!(e.to_string().contains("stage"), "{e}");
}

#[test]
fn area_shape_batches_train_per_shape_and_resume_exactly() {
    cpu_process(
        "curricula_lifecycle::area_shape_batches_train_per_shape_and_resume_exactly",
        area_shape_batches_train_per_shape_and_resume_exactly_body,
    );
}

fn area_shape_batches_train_per_shape_and_resume_exactly_body() {
    let (data, a, b) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = dataset(data.path());
    let mut cfg = base();
    cfg.steps = 4;
    cfg.batch_size = 1;
    cfg.save_every = 2;
    opts(&mut cfg).extend(
        json!({"shape_policy": "area", "shape_align": 8, "shape_max_ratio": 4.0,
               "ema_decay": 0.9})
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut t = trainer_with(None);
    assert_eq!(
        train(t.as_mut(), &request(&items, cfg.clone(), a.path())).unwrap(),
        4
    );
    // Interrupted at 2, then resumed (the shape batcher's regroup history replays).
    let req = request(&items, cfg.clone(), b.path());
    let flag = req.cancel.clone();
    on_cpu(|| {
        trainer_with(None)
            .train(&req, &mut |p| {
                if let candle_gen::gen_core::TrainingProgress::Training { step: 2, .. } = p {
                    flag.cancel();
                }
            })
            .unwrap()
    });
    cfg.resume = true;
    assert_eq!(
        train(t.as_mut(), &request(&items, cfg, b.path())).unwrap(),
        4
    );
    let (sa, sb) = (
        read_checkpoint_state(&latest(a.path())).unwrap(),
        read_checkpoint_state(&latest(b.path())).unwrap(),
    );
    assert_eq!(sa, sb, "the resumed area run equals the uninterrupted one");
    assert!(
        sa.data_position.unwrap().to_string().contains("shape"),
        "a shape-batch cursor"
    );
}

#[test]
fn activation_checkpointing_leaves_the_run_unchanged() {
    cpu_process(
        "curricula_lifecycle::activation_checkpointing_leaves_the_run_unchanged",
        activation_checkpointing_leaves_the_run_unchanged_body,
    );
}

fn activation_checkpointing_leaves_the_run_unchanged_body() {
    let (data, a, b) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = dataset(data.path());
    let cfg = base();
    let mut ac = cfg.clone();
    opts(&mut ac).insert("activation_checkpointing".into(), json!("selective_layer"));
    opts(&mut ac).insert("ac_selective_every".into(), json!(2));
    let mut t = trainer_with(None);
    train(t.as_mut(), &request(&items, cfg, a.path())).unwrap();
    train(t.as_mut(), &request(&items, ac, b.path())).unwrap();
    let file = |d: &Path| d.join("curricula").join("model.safetensors");
    let load = |p: PathBuf| candle_gen::candle_core::safetensors::load(p, &cpu()).unwrap();
    let (x, y) = (load(file(a.path())), load(file(b.path())));
    let mut worst = 0f32;
    for (k, v) in &x {
        let d = v
            .to_dtype(DType::F32)
            .unwrap()
            .sub(&y[k].to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        worst = worst.max(d);
    }
    eprintln!("activation checkpointing vs plain: max |Δ| = {worst:.3e}");
    assert!(worst <= 1e-6, "{worst}");
}

#[test]
fn holdout_validation_writes_the_banded_metrics() {
    cpu_process(
        "curricula_lifecycle::holdout_validation_writes_the_banded_metrics",
        holdout_validation_writes_the_banded_metrics_body,
    );
}

fn holdout_validation_writes_the_banded_metrics_body() {
    let (data, out) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let items = dataset(data.path());
    let mut cfg = base();
    opts(&mut cfg).insert("val_every_steps".into(), json!(1));
    opts(&mut cfg).insert("val_samples".into(), json!(1));
    opts(&mut cfg).insert("log_every".into(), json!(1));
    let mut t = trainer_with(None);
    train(t.as_mut(), &request(&items, cfg, out.path())).unwrap();
    let m = metrics(out.path());
    let vals: Vec<&Json> = m.iter().filter(|r| r["kind"] == "validation").collect();
    assert_eq!(vals.len(), 2, "one validation per step: {m:?}");
    for k in [
        "val/loss",
        "val/loss_t0",
        "val/loss_t1",
        "val/loss_t2",
        "val/loss_t3",
    ] {
        assert!(vals[0][k].is_number(), "{k}: {:?}", vals[0]);
    }
    // The holdout item never trains: the frozen grid's single sample sits in band 0 (k = 0 → t
    // index 0), and the overall mean is that sample's loss.
    assert_eq!(vals[0]["val/loss"], vals[0]["val/loss_t0"]);
    let steps = m.iter().filter(|r| r["kind"] == "step").count();
    assert_eq!(steps, 2);
}

#[test]
fn fp16_trains_with_the_dynamic_loss_scaler() {
    cpu_process(
        "curricula_lifecycle::fp16_trains_with_the_dynamic_loss_scaler",
        fp16_trains_with_the_dynamic_loss_scaler_body,
    );
}

fn fp16_trains_with_the_dynamic_loss_scaler_body() {
    let (data, out) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let items = dataset(data.path());
    let mut cfg = base();
    cfg.train_dtype = "fp16".into();
    opts(&mut cfg).insert("log_every".into(), json!(1));
    let mut t = trainer_with(None);
    if device().is_cpu() {
        // Candle's CPU backend has no half-precision GEMM: a typed refusal up front.
        let e = t
            .validate(&request(&items, cfg, out.path()))
            .expect_err("fp16 on CPU is refused");
        assert!(
            matches!(e, candle_gen::gen_core::Error::Unsupported(_)),
            "{e}"
        );
        return;
    }
    assert_eq!(
        train(t.as_mut(), &request(&items, cfg, out.path())).unwrap(),
        2
    );
    let m = metrics(out.path());
    assert!(
        m.iter()
            .any(|r| r["kind"] == "step" && r.get("loss_scale").is_some()),
        "{m:?}"
    );
    let st = read_checkpoint_state(&latest(out.path())).unwrap();
    assert!(st.grad_scaler.is_some(), "the scaler state is checkpointed");
}

#[test]
fn resume_optimizer_core_starts_a_new_projector_cold() {
    cpu_process(
        "curricula_lifecycle::resume_optimizer_core_starts_a_new_projector_cold",
        resume_optimizer_core_starts_a_new_projector_cold_body,
    );
}

fn resume_optimizer_core_starts_a_new_projector_cold_body() {
    let (data, out, tdir) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = dataset(data.path());
    let teacher = random_teacher(&tdir.path().join("t"));
    let mut cfg = base();
    let mut t = trainer_with(Some(&teacher));
    train(t.as_mut(), &request(&items, cfg.clone(), out.path())).unwrap();
    cfg.steps = 4;
    cfg.resume = true;
    opts(&mut cfg).insert(
        "repa".into(),
        json!({"weight": 0.5, "teacher": "dinov2_vits14", "teacher_dim": 384,
               "teacher_image_size": 56, "proj_hidden_dim": 32, "layer": 1}),
    );
    let e = train(t.as_mut(), &request(&items, cfg.clone(), out.path()))
        .expect_err("a full optimizer resume refuses the new projector");
    assert!(e.to_string().contains("projector"), "{e}");
    opts(&mut cfg).insert("resume_optimizer".into(), json!("core"));
    assert_eq!(
        train(t.as_mut(), &request(&items, cfg, out.path())).unwrap(),
        4
    );
    let st = read_checkpoint_state(&latest(out.path())).unwrap();
    assert!(st.repa_projector.is_some());
}

#[test]
fn a_pre_curricula_checkpoint_resumes_exactly() {
    cpu_process(
        "curricula_lifecycle::a_pre_curricula_checkpoint_resumes_exactly",
        a_pre_curricula_checkpoint_resumes_exactly_body,
    );
}

/// A checkpoint written before sc-25689 (`state.json` without `data_position` / `stage` / `dist` /
/// `repa_projector`, its data identity carrying `batch=…;accum=…`) resumes **exactly**: the legacy
/// cursor (the saved micro-batch's `batches_consumed`) continues the walk, and the resumed run
/// equals the uninterrupted one.
fn a_pre_curricula_checkpoint_resumes_exactly_body() {
    let (data, a, b) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = dataset(data.path());
    let mut cfg = base();
    cfg.steps = 4;
    cfg.batch_size = 1;
    cfg.save_every = 1;
    opts(&mut cfg).insert("ema_decay".into(), json!(0.9));
    let mut t = trainer_with(None);
    assert_eq!(
        train(t.as_mut(), &request(&items, cfg.clone(), a.path())).unwrap(),
        4
    );
    let req = request(&items, cfg.clone(), b.path());
    let flag = req.cancel.clone();
    on_cpu(|| {
        trainer_with(None)
            .train(&req, &mut |p| {
                if let candle_gen::gen_core::TrainingProgress::Training { step: 1, .. } = p {
                    flag.cancel();
                }
            })
            .unwrap()
    });
    // Rewrite the interrupted checkpoint's state.json into the pre-sc-25689 shape.
    let state_path = latest(b.path()).join("state.json");
    let mut v: Json = serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    let o = v.as_object_mut().unwrap();
    for k in [
        "data_position",
        "stage",
        "parent_stage",
        "dist",
        "repa_projector",
        "grad_scaler",
    ] {
        o.remove(k);
    }
    let id = o["data_identity"].as_str().unwrap().to_string();
    assert!(
        id.starts_with("image_size=") && !id.contains("batch="),
        "{id}"
    );
    let legacy = id.replacen(";seed=", ";batch=1;accum=1;seed=", 1);
    o.insert("data_identity".into(), json!(legacy));
    std::fs::write(&state_path, serde_json::to_vec(&v).unwrap()).unwrap();
    let old = read_checkpoint_state(&latest(b.path())).unwrap();
    assert!(old.data_position.is_none() && old.legacy_batch() == Some((1, 1)));
    assert_eq!(old.step, 1);

    cfg.resume = true;
    assert_eq!(
        train(t.as_mut(), &request(&items, cfg, b.path())).unwrap(),
        4
    );
    let (la, lb) = (latest(a.path()), latest(b.path()));
    let (mut sa, mut sb) = (
        read_checkpoint_state(&la).unwrap(),
        read_checkpoint_state(&lb).unwrap(),
    );
    // The cursors are equivalent, not equal: a resumed walk records its re-split plan (upstream's
    // `load_resume_state` does the same) — both have walked the whole epoch.
    let left = |s: &Json| {
        let covered: u64 = s["ranges_by_rank"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r[1].as_u64().unwrap() - r[0].as_u64().unwrap())
            .sum();
        covered - s["samples_per_rank"].as_u64().unwrap()
    };
    let (pa, pb) = (
        sa.data_position.take().unwrap(),
        sb.data_position.take().unwrap(),
    );
    assert_eq!((left(&pa), left(&pb)), (0, 0), "{pa} / {pb}");
    assert_eq!(sa, sb, "the resumed legacy checkpoint continues exactly");
    for f in [
        "trainable.safetensors",
        "ema.safetensors",
        "optimizer.safetensors",
    ] {
        assert_eq!(
            std::fs::read(la.join(f)).unwrap(),
            std::fs::read(lb.join(f)).unwrap(),
            "{f}: the resumed run's state equals the uninterrupted one's"
        );
    }
}
