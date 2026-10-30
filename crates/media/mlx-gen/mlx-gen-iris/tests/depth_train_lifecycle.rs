//! The MLX depth trainer (`iris_3b_depth`, sc-25687) end to end on the miniature depth export
//! (`tests/fixtures/tiny-depth`) and the miniature generation parent, through the explicit
//! catalog's `load_trainer`: the gen-core trainer conformance checks, cancel → durable checkpoint →
//! resume reproducing the uninterrupted run bit for bit, the invalid-example policy, a bounded
//! overfit with a held-out check, previews, parent init (empty prompt from a file or encoded once),
//! and the export: `depth/` layout + task stamps, loading in the standalone `iris_3b_depth`
//! estimator with the depth moving toward the target, refused by the generation and restoration
//! loaders and the generation trainer.

use std::path::{Path, PathBuf};

use gen_core_iris::depth::{DepthRequest, DepthResolution};
use gen_core_iris::downstream::{weights_metadata, EmptyPrompt, TaskExport};
use gen_core_iris::train::depth::{
    load_depth_sample, normalize_log_depth, DepthLossPlan, EMPTY_PROMPT_COMPONENT, ITEM_DEPTH_KEY,
};
use gen_core_iris::train::{
    checkpoint_root, latest_checkpoint, read_checkpoint_state, OPTIONS_KEY,
};
use gen_core_iris::{IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{
    Error as CoreError, Trainer, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
};
use mlx_gen::{Error, Image, LoadSpec, Precision, WeightsSource};
use mlx_rs::{Array, Dtype};
use serde_json::json;

use crate::common::{fixtures, on_cpu, tiny_backbone, tiny_text_encoder};

const ID: &str = "iris_3b_depth";

fn export_dir() -> PathBuf {
    fixtures().join("tiny-depth")
}

fn trainer_for(spec: &LoadSpec) -> Box<dyn Trainer> {
    mlx_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(ID, spec)
        .expect("the depth trainer loads")
}

fn trainer() -> Box<dyn Trainer> {
    trainer_for(&LoadSpec::new(WeightsSource::Dir(export_dir())))
}

pub fn write_npy(path: &Path, h: usize, w: usize, f: impl Fn(usize, usize) -> f32) {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}").into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    for y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(&f(y, x).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// One aligned example: an RGB image whose colours follow the depth, its depth `.npy` and a mask
/// with a few holes.
fn example(dir: &Path, i: usize, w: usize, h: usize, kind: usize) -> TrainingItem {
    let depth = move |y: usize, x: usize| -> f32 {
        match kind {
            0 => 1.0 + 0.4 * x as f32 + 0.1 * y as f32,
            1 => 2.0 + (x as f32 / 3.0).sin() + 0.3 * y as f32,
            2 => 8.0 - 0.3 * x as f32 + 0.2 * (y as f32 / 2.0).cos(),
            _ => 3.0,
        }
    };
    let img = image::RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let d = depth(y as usize, x as usize);
        image::Rgb([
            (d * 25.0).min(255.0) as u8,
            (255.0 - d * 20.0).max(0.0) as u8,
            ((x * 13 + y * 7) % 256) as u8,
        ])
    });
    let rgb = dir.join(format!("rgb{i}.png"));
    img.save(&rgb).unwrap();
    let dpath = dir.join(format!("depth{i}.npy"));
    write_npy(&dpath, h, w, depth);
    let mpath = dir.join(format!("mask{i}.npy"));
    write_npy(&mpath, h, w, |y, x| {
        if (x * 7 + y * 3) % 11 == 0 {
            0.0
        } else {
            1.0
        }
    });
    let mut item = TrainingItem::captioned(rgb, String::new());
    item.model_options.insert(
        ITEM_DEPTH_KEY.into(),
        json!({"path": dpath.to_str().unwrap(), "mask_path": mpath.to_str().unwrap()}),
    );
    item
}

fn dataset(dir: &Path) -> Vec<TrainingItem> {
    (0..3).map(|i| example(dir, i, 16, 16, i)).collect()
}

fn config() -> TrainingConfig {
    TrainingConfig {
        learning_rate: 1e-3,
        steps: 4,
        batch_size: 2,
        resolution: 16,
        save_every: 2,
        seed: 9,
        train_dtype: "f32".into(),
        full_finetune: true,
        loss_type: "l1".into(),
        weight_decay: 0.01,
        ..Default::default()
    }
}

fn options(cfg: &mut TrainingConfig, v: serde_json::Value) {
    cfg.model_options.insert(OPTIONS_KEY.into(), v);
}

fn request(items: &[TrainingItem], cfg: TrainingConfig, out: &Path, name: &str) -> TrainingRequest {
    TrainingRequest {
        items: items.to_vec(),
        config: cfg,
        output_dir: out.to_path_buf(),
        file_name: name.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

fn run_with(
    t: &mut dyn Trainer,
    items: &[TrainingItem],
    cfg: TrainingConfig,
    out: &Path,
    name: &str,
    cancel_at: Option<u32>,
) -> (u32, Vec<f32>) {
    let req = request(items, cfg, out, name);
    let flag = req.cancel.clone();
    let mut losses = Vec::new();
    let res = t
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                losses.push(loss);
                if Some(step) == cancel_at {
                    flag.cancel();
                }
            }
        })
        .unwrap();
    (res.steps, losses)
}

fn run(
    items: &[TrainingItem],
    cfg: TrainingConfig,
    out: &Path,
    name: &str,
    cancel_at: Option<u32>,
) -> u32 {
    run_with(trainer().as_mut(), items, cfg, out, name, cancel_at).0
}

fn bytes_of(path: &Path) -> Vec<(String, Vec<f32>)> {
    let mut v: Vec<(String, Vec<f32>)> = Array::load_safetensors(path)
        .unwrap()
        .into_iter()
        .map(|(k, a)| {
            let a = a.as_dtype(Dtype::Float32).unwrap();
            let n: i32 = a.shape().iter().product();
            (k, a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec())
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn gen_core_trainer_conformance_checks() {
    crate::common::cpu_process(
        "depth_train_lifecycle::gen_core_trainer_conformance_checks",
        || {
            gen_core_trainer_conformance_checks_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::gen_core_trainer_conformance_checks");
        },
    );
}

fn gen_core_trainer_conformance_checks_body() {
    use gen_core_testkit::trainer as kit;
    let data = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let mut profile = kit::TrainerProfile::cheap(items[..2].to_vec(), out.path().into());
    profile.config.full_finetune = true;
    profile.config.loss_type = "l1".into();
    profile.config.resolution = 16;
    // `check_trainer_validate` assumes an adapter trainer (it probes LoRA / LoKr acceptance), which a
    // full-fine-tune-only trainer cannot satisfy by construction; every other check runs.
    kit::check_trainer_progress(trainer().as_mut(), &profile).unwrap();
    kit::check_trainer_cancellation(&trainer, &profile).unwrap();
    kit::check_trainer_technique_refusal(&trainer, &profile).unwrap();
    kit::check_trainer_train_floors(&trainer, &profile).unwrap();
    let registry = mlx_gen_iris::provider_registry().unwrap();
    kit::check_trainer_registry(&registry, trainer().as_ref()).unwrap();
    // validate: the cheap request is accepted, an empty dataset and an adapter request refused.
    let t = trainer();
    let mut req = request(
        &profile.items,
        profile.config.clone(),
        out.path(),
        "v.safetensors",
    );
    t.validate(&req).unwrap();
    req.config.full_finetune = false;
    assert!(matches!(t.validate(&req), Err(CoreError::Unsupported(_))));
    req.config.full_finetune = true;
    // sc-25690: activation checkpointing (`gradient_checkpointing`) is a trained option of this
    // task on both backends.
    req.config.gradient_checkpointing = true;
    t.validate(&req).unwrap();
    req.config.gradient_checkpointing = false;
    req.items.clear();
    assert!(t.validate(&req).is_err());
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn cancel_and_resume_reproduce_the_uninterrupted_run() {
    crate::common::cpu_process(
        "depth_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run",
        || {
            cancel_and_resume_reproduce_the_uninterrupted_run_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run");
        },
    );
}

fn cancel_and_resume_reproduce_the_uninterrupted_run_body() {
    cancel_and_resume_under(false);
}

/// sc-25690: the same lifecycle with the step's trunk / pixel blocks activation-checkpointed —
/// cancel → durable checkpoint → resume reproduces the uninterrupted checkpointed run bit for bit
/// (weights, EMA, optimizer, scheduler / data position, export).
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn cancel_and_resume_reproduce_the_uninterrupted_run_under_activation_checkpointing() {
    crate::common::cpu_process(
        "depth_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run_under_activation_checkpointing", || { cancel_and_resume_under(true); eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run_under_activation_checkpointing"); },
    );
}

fn cancel_and_resume_under(ac: bool) {
    use mlx_gen_iris::train::model::checkpointed_regions;
    let entered = checkpointed_regions();
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let name = "depth.safetensors";
        let mut cfg = config();
        cfg.gradient_checkpointing = ac;
        cfg.batch_size = 1;
        cfg.gradient_accumulation = 2;
        cfg.steps = 8;
        cfg.save_every = 4;
        options(
            &mut cfg,
            json!({"crop": "random", "ema_enabled": true, "ema_decay": 0.9, "depth_size": [8, 12]}),
        );
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        assert_eq!(run(&items, cfg.clone(), a.path(), name, None), 8);
        assert_eq!(run(&items, cfg.clone(), b.path(), name, Some(4)), 4);
        let root_b = checkpoint_root(b.path(), name);
        assert_eq!(
            read_checkpoint_state(&latest_checkpoint(&root_b).unwrap())
                .unwrap()
                .step,
            2
        );
        let mut resumed = cfg.clone();
        resumed.resume = true;
        // The resumed run itself is checkpointed: a resume that dropped the policy would still
        // reproduce the weights (the gradients are the plain graph's), so count its own regions.
        let before_resume = checkpointed_regions();
        assert_eq!(run(&items, resumed, b.path(), name, None), 8);
        assert_eq!(
            checkpointed_regions() > before_resume,
            ac,
            "the resumed run enters regions exactly when it is checkpointed"
        );
        let (la, lb) = (
            latest_checkpoint(&checkpoint_root(a.path(), name)).unwrap(),
            latest_checkpoint(&root_b).unwrap(),
        );
        for f in [
            "trainable.safetensors",
            "ema.safetensors",
            "optimizer.safetensors",
        ] {
            assert_eq!(bytes_of(&la.join(f)), bytes_of(&lb.join(f)), "{f}");
        }
        let (sa, sb) = (
            read_checkpoint_state(&la).unwrap(),
            read_checkpoint_state(&lb).unwrap(),
        );
        assert_eq!(sa.task, IrisTask::Depth);
        assert_eq!((sa.step, sa.epoch, sa.batches_consumed), (4, 3, 2));
        assert_eq!(sa, sb);
        let model = |d: &Path| d.join("depth").join("model.safetensors");
        assert_eq!(bytes_of(&model(a.path())), bytes_of(&model(b.path())));
        // The generation trainer refuses the depth checkpoint as a resume source.
        let mut gen_spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
        gen_spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(tiny_text_encoder()),
        );
        let mut gen_cfg = TrainingConfig {
            steps: 1,
            resolution: 16,
            train_dtype: "f32".into(),
            ..Default::default()
        };
        options(&mut gen_cfg, json!({"resume_from": lb.to_str().unwrap()}));
        let mut captioned = items.clone();
        for it in &mut captioned {
            it.caption = "a fox".into();
            it.model_options.clear();
        }
        let err = mlx_gen_iris::provider_registry()
            .unwrap()
            .load_trainer("iris_3b", &gen_spec)
            .unwrap()
            .train(
                &request(&captioned, gen_cfg, b.path(), "g.safetensors"),
                &mut |_| {},
            )
            .unwrap_err();
        assert!(err.to_string().contains("depth task"), "{err}");
    });
    assert_eq!(
        checkpointed_regions() > entered,
        ac,
        "regions are entered exactly when the run is checkpointed"
    );
}

/// sc-25690: a checkpointed depth run is the plain run (f32) — the trainer wires
/// `gradient_checkpointing` into its step, the export moves by rounding only, keeps the release's
/// layout and stamps, and loads in the estimator.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn activation_checkpointing_leaves_the_run_unchanged() {
    crate::common::cpu_process(
        "depth_train_lifecycle::activation_checkpointing_leaves_the_run_unchanged",
        || {
            activation_checkpointing_leaves_the_run_unchanged_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::activation_checkpointing_leaves_the_run_unchanged");
        },
    );
}

fn activation_checkpointing_leaves_the_run_unchanged_body() {
    use mlx_gen_iris::train::model::checkpointed_regions;
    let (data, a, b) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = dataset(data.path());
    let export = |ac: bool, out: &Path| {
        let mut cfg = config();
        cfg.save_every = 0;
        cfg.gradient_checkpointing = ac;
        let before = checkpointed_regions();
        let res = trainer()
            .train(&request(&items, cfg, out, "depth.safetensors"), &mut |_| {})
            .unwrap();
        assert_eq!(res.steps, 4);
        assert_eq!(checkpointed_regions() > before, ac);
        res.adapter_path
    };
    let (plain, checkpointed) = (export(false, a.path()), export(true, b.path()));
    let (tp, tc) = (bytes_of(&plain), bytes_of(&checkpointed));
    assert_eq!(tp.len(), tc.len());
    let mut worst = 0f32;
    for ((k, x), (k2, y)) in tp.iter().zip(&tc) {
        assert_eq!(k, k2);
        for (p, q) in x.iter().zip(y) {
            worst = worst.max((p - q).abs());
        }
    }
    eprintln!("depth: activation checkpointing vs plain: max |Δ| = {worst:.3e}");
    assert!(worst <= 1e-6, "{worst}");
    // The policy is no part of the export's identity, and the checkpointed export is a depth model.
    let meta = |p: &Path| weights_metadata(p).unwrap().unwrap();
    assert_eq!(meta(&plain), meta(&checkpointed));
    let dir = checkpointed.parent().unwrap();
    TaskExport::from_dir(dir, IrisTask::Depth, ID).unwrap();
    let values = estimate(dir, &rgb_image(&items[0]));
    assert!(values.iter().all(|v| v.is_finite()));
}

/// `invalid_examples`: `reject` refuses before any step naming the item; `skip` trains on the rest.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn invalid_examples_follow_the_policy() {
    crate::common::cpu_process(
        "depth_train_lifecycle::invalid_examples_follow_the_policy",
        || {
            invalid_examples_follow_the_policy_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::invalid_examples_follow_the_policy");
        },
    );
}

fn invalid_examples_follow_the_policy_body() {
    let data = tempfile::tempdir().unwrap();
    let mut items = dataset(data.path());
    items.push(example(data.path(), 3, 16, 16, 3)); // a constant plane: no normalization
    let out = tempfile::tempdir().unwrap();
    let req = request(&items, config(), out.path(), "r.safetensors");
    let mut trained = false;
    let err = trainer()
        .train(&req, &mut |p| {
            trained |= matches!(p, TrainingProgress::Training { .. });
        })
        .unwrap_err();
    assert!(!trained);
    assert!(err.to_string().contains("rgb3.png"), "{err}");
    let mut cfg = config();
    cfg.steps = 2;
    options(&mut cfg, json!({"invalid_examples": "skip"}));
    assert_eq!(run(&items, cfg, out.path(), "s.safetensors", None), 2);
}

fn estimate(dir: &Path, image: &Image) -> Vec<f32> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
    spec.precision = Precision::Fp32;
    let est = mlx_gen_iris::depth::load(&spec).unwrap();
    let req = DepthRequest {
        image: image.clone(),
        resolution: DepthResolution::Native,
        cancel: Default::default(),
    };
    est.estimate(&req, &mut |_| {}).unwrap().map.values
}

fn rgb_image(item: &TrainingItem) -> Image {
    let img = mlx_gen::gen_core::__model_input_reader!(
        image::open(&item.image_path),
        "crates/media/mlx-gen/mlx-gen-iris/tests/depth_train_lifecycle.rs::rgb_image/image::open",
        Reference,
        &item.image_path,
        |__model_input_path| image::open(__model_input_path)
    )
    .unwrap()
    .to_rgb8();
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

/// Mean |prediction − normalized target| over the valid pixels.
fn target_error(pred: &[f32], item: &TrainingItem) -> f64 {
    let s = load_depth_sample(item).unwrap();
    let t = normalize_log_depth(&s.depth, &s.valid, &DepthLossPlan::default(), 2).unwrap();
    let (mut sum, mut n) = (0f64, 0usize);
    for i in 0..t.len() {
        if s.valid[i] {
            sum += (pred[i] - t[i]).abs() as f64;
            n += 1;
        }
    }
    sum / n as f64
}

/// A bounded overfit on two aligned examples: the loss drops, the held-out example stays finite and
/// sane, and the exported model — loaded by the standalone `iris_3b_depth` estimator unchanged —
/// predicts depth closer to the training targets than the starting export.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn overfit_export_and_load_in_the_estimator() {
    crate::common::cpu_process(
        "depth_train_lifecycle::overfit_export_and_load_in_the_estimator",
        || {
            overfit_export_and_load_in_the_estimator_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::overfit_export_and_load_in_the_estimator");
        },
    );
}

fn overfit_export_and_load_in_the_estimator_body() {
    let data = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let out = tempfile::tempdir().unwrap();
    let mut cfg = config();
    cfg.steps = 80;
    cfg.save_every = 0;
    cfg.sample_every = 40;
    options(&mut cfg, json!({"preview_items": 1, "hflip_prob": 0.0}));
    let req = request(&items[..2], cfg, out.path(), "overfit.safetensors");
    let mut losses = Vec::new();
    let mut previews = Vec::new();
    let output = trainer()
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { loss, .. } => losses.push(loss),
            TrainingProgress::Sample { image, prompt, .. } => previews.push((prompt, image)),
            _ => {}
        })
        .unwrap();
    let first = losses[..4].iter().sum::<f32>() / 4.0;
    let last = losses[losses.len() - 4..].iter().sum::<f32>() / 4.0;
    eprintln!("loss {first:.4} -> {last:.4}");
    assert!(last < 0.5 * first, "the loss must drop: {first} -> {last}");
    assert_eq!(previews.len(), 2);
    assert_eq!((previews[0].1.width, previews[0].1.height), (16, 16));
    assert!(previews[0].0.ends_with("rgb0.png"));

    // The export: the `depth/` layout, the task stamps, the release's key layout.
    let dir = out.path().join("overfit");
    assert_eq!(output.adapter_path, dir.join("model.safetensors"));
    let export = TaskExport::from_dir(&dir, IrisTask::Depth, ID).unwrap();
    assert_eq!(export.settings.name, "depth");
    let meta = weights_metadata(&dir.join("model.safetensors"))
        .unwrap()
        .unwrap();
    assert_eq!(meta["irisTask"], "depth");
    assert_eq!(meta["irisArtifact"], "full_model");
    assert_eq!(meta["irisWeights"], "raw");
    assert_eq!(meta["irisTrainedSteps"], "80");
    assert_eq!(meta["format"], "pt");
    assert!(!meta["irisBaseIdentity"].is_empty());
    let keys: Vec<String> = bytes_of(&dir.join("model.safetensors"))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let release: Vec<String> = bytes_of(&export_dir().join("model.safetensors"))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(
        keys, release,
        "the export carries exactly the release's tensors"
    );
    let prompt = |d: &Path| {
        let c = TaskExport::from_dir(d, IrisTask::Depth, ID).unwrap();
        EmptyPrompt::read(&c.empty_prompt_path(), &c.config).unwrap()
    };
    assert_eq!(prompt(&dir), prompt(&export_dir()));

    // Estimator: closer to the trained targets than the starting model; the held-out item sane.
    for item in &items[..2] {
        let img = rgb_image(item);
        let before = target_error(&estimate(&export_dir(), &img), item);
        let after = target_error(&estimate(&dir, &img), item);
        eprintln!(
            "{}: |pred − target| {before:.4} -> {after:.4}",
            item.image_path.display()
        );
        assert!(after < 0.5 * before);
    }
    let held = estimate(&dir, &rgb_image(&items[2]));
    assert!(held.iter().all(|v| v.is_finite()));
    let held_err = target_error(&held, &items[2]);
    eprintln!("held-out |pred − target| {held_err:.4}");
    assert!(held_err < 3.0);

    // The other loaders refuse the depth export by its task identity.
    let mut gen_spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    gen_spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    assert!(matches!(
        mlx_gen_iris::load(&gen_spec),
        Err(Error::Unsupported(_))
    ));
    let registry = mlx_gen_iris::provider_registry().unwrap();
    assert!(matches!(
        registry.load_trainer("iris_3b", &gen_spec),
        Err(CoreError::Unsupported(_))
    ));
    let restore = registry.load_transform(
        "iris_3b_restore",
        &LoadSpec::new(WeightsSource::Dir(dir.clone())),
    );
    assert!(matches!(restore, Err(CoreError::Unsupported(_))));
    // ...and a mislabelled stamp is refused by the depth loader itself.
    assert!(TaskExport::from_dir(&dir, IrisTask::Restoration, "x").is_err());
}

/// From the generation parent: the empty prompt comes from a staged file (equal to the export's)
/// or is encoded once by the text encoder; the run trains and its export loads in the estimator.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn parent_init_from_a_file_or_the_text_encoder() {
    crate::common::cpu_process(
        "depth_train_lifecycle::parent_init_from_a_file_or_the_text_encoder",
        || {
            parent_init_from_a_file_or_the_text_encoder_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::parent_init_from_a_file_or_the_text_encoder");
        },
    );
}

fn parent_init_from_a_file_or_the_text_encoder_body() {
    let data = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let mut cfg = config();
    cfg.steps = 2;
    cfg.save_every = 0;
    for (component, source) in [
        (
            EMPTY_PROMPT_COMPONENT,
            WeightsSource::File(export_dir().join("empty_prompt.safetensors")),
        ),
        (
            TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir(tiny_text_encoder()),
        ),
    ] {
        let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
        spec.components.insert(component.into(), source);
        let out = tempfile::tempdir().unwrap();
        let mut t = trainer_for(&spec);
        let (steps, _) = run_with(
            t.as_mut(),
            &items,
            cfg.clone(),
            out.path(),
            "p.safetensors",
            None,
        );
        assert_eq!(steps, 2);
        let dir = out.path().join("p");
        let values = estimate(&dir, &rgb_image(&items[0]));
        assert!(values.iter().all(|v| v.is_finite()), "{component}");
        // The export keeps the parent's backbone shape, widened.
        let tensors = bytes_of(&dir.join("model.safetensors"));
        let s = tensors
            .iter()
            .find(|(k, _)| k == "pixel.s_embedder.proj.weight")
            .unwrap();
        assert_eq!(s.1.len(), 64 * 4 * 4 * 4);
    }
    // A parent without conditioning, or a depth export with an extra text encoder, is refused.
    let registry = mlx_gen_iris::provider_registry().unwrap();
    assert!(registry
        .load_trainer(ID, &LoadSpec::new(WeightsSource::Dir(tiny_backbone())))
        .is_err());
    let mut spec = LoadSpec::new(WeightsSource::Dir(export_dir()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    assert!(matches!(
        registry.load_trainer(ID, &spec),
        Err(CoreError::Unsupported(_))
    ));
}
