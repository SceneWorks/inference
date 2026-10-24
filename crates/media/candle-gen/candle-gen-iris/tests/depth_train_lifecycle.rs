//! The Candle depth trainer (`iris_3b_depth`, sc-25687) end to end on the miniature depth export
//! and generation parent — the twin of the MLX `depth_train_lifecycle`: conformance checks, bit-exact
//! cancel / resume, the invalid-example policy, a bounded overfit with a held-out check, previews,
//! parent init, and the export loading in the standalone `iris_3b_depth` estimator (depth moving
//! toward the target) while the generation and restoration loaders refuse it.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::iris::depth::{DepthRequest, DepthResolution};
use candle_gen::gen_core::iris::downstream::{weights_metadata, EmptyPrompt, TaskExport};
use candle_gen::gen_core::iris::train::depth::{
    load_depth_sample, normalize_log_depth, DepthLossPlan, EMPTY_PROMPT_COMPONENT, ITEM_DEPTH_KEY,
};
use candle_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, read_checkpoint_state, OPTIONS_KEY,
};
use candle_gen::gen_core::iris::{IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    Error as CoreError, Image, LoadSpec, Precision, Trainer, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use serde_json::json;

use crate::common::{fixtures, tiny_backbone, tiny_text_encoder};

const ID: &str = "iris_3b_depth";

fn export_dir() -> PathBuf {
    fixtures().join("tiny-depth")
}

fn trainer_for(spec: &LoadSpec) -> Box<dyn Trainer> {
    candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(ID, spec)
        .expect("the depth trainer loads")
}

fn trainer() -> Box<dyn Trainer> {
    trainer_for(&LoadSpec::new(WeightsSource::Dir(export_dir())))
}

fn write_npy(path: &Path, h: usize, w: usize, f: impl Fn(usize, usize) -> f32) {
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
) -> u32 {
    let req = request(items, cfg, out, name);
    let flag = req.cancel.clone();
    let res = t
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, .. } = p {
                if Some(step) == cancel_at {
                    flag.cancel();
                }
            }
        })
        .unwrap();
    res.steps
}

fn run(
    items: &[TrainingItem],
    cfg: TrainingConfig,
    out: &Path,
    name: &str,
    cancel_at: Option<u32>,
) -> u32 {
    run_with(trainer().as_mut(), items, cfg, out, name, cancel_at)
}

fn tensors_of(path: &Path) -> Vec<(String, Vec<f32>)> {
    let mut v: Vec<(String, Vec<f32>)> =
        candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
            .unwrap()
            .into_iter()
            .map(|(k, t)| {
                let t = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
                (k, t.to_vec1::<f32>().unwrap())
            })
            .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn gen_core_trainer_conformance_checks() {
    use gen_core_testkit::trainer as kit;
    let data = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let mut profile = kit::TrainerProfile::cheap(items[..2].to_vec(), out.path().into());
    profile.config.full_finetune = true;
    profile.config.loss_type = "l1".into();
    profile.config.resolution = 16;
    profile.config.train_dtype = "f32".into();
    // `check_trainer_validate` probes LoRA / LoKr acceptance (an adapter trainer's contract), which a
    // full-fine-tune-only trainer cannot satisfy by construction; every other check runs.
    kit::check_trainer_progress(trainer().as_mut(), &profile).unwrap();
    kit::check_trainer_cancellation(&trainer, &profile).unwrap();
    kit::check_trainer_technique_refusal(&trainer, &profile).unwrap();
    kit::check_trainer_train_floors(&trainer, &profile).unwrap();
    let registry = candle_gen_iris::provider_registry().unwrap();
    kit::check_trainer_registry(&registry, trainer().as_ref()).unwrap();
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
    req.items.clear();
    assert!(t.validate(&req).is_err());

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::gen_core_trainer_conformance_checks");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn cancel_and_resume_reproduce_the_uninterrupted_run() {
    let data = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let name = "depth.safetensors";
    let mut cfg = config();
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
    assert_eq!(run(&items, resumed, b.path(), name, None), 8);
    let (la, lb) = (
        latest_checkpoint(&checkpoint_root(a.path(), name)).unwrap(),
        latest_checkpoint(&root_b).unwrap(),
    );
    for f in [
        "trainable.safetensors",
        "ema.safetensors",
        "optimizer.safetensors",
    ] {
        assert_eq!(tensors_of(&la.join(f)), tensors_of(&lb.join(f)), "{f}");
    }
    let (sa, sb) = (
        read_checkpoint_state(&la).unwrap(),
        read_checkpoint_state(&lb).unwrap(),
    );
    assert_eq!(sa.task, IrisTask::Depth);
    assert_eq!((sa.step, sa.epoch, sa.batches_consumed), (4, 3, 2));
    assert_eq!(sa, sb);
    let model = |d: &Path| d.join("depth").join("model.safetensors");
    assert_eq!(tensors_of(&model(a.path())), tensors_of(&model(b.path())));

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn invalid_examples_follow_the_policy() {
    let data = tempfile::tempdir().unwrap();
    let mut items = dataset(data.path());
    items.push(example(data.path(), 3, 16, 16, 3));
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::invalid_examples_follow_the_policy");
}

fn estimate(dir: &Path, image: &Image) -> Vec<f32> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
    spec.precision = Precision::Fp32;
    let est = candle_gen_iris::depth::load(&spec).unwrap();
    let req = DepthRequest {
        image: image.clone(),
        resolution: DepthResolution::Native,
        cancel: Default::default(),
    };
    est.estimate(&req, &mut |_| {}).unwrap().map.values
}

fn rgb_image(item: &TrainingItem) -> Image {
    let img = candle_gen::gen_core::__model_input_reader!(
        image::open(&item.image_path),
        "crates/media/candle-gen/candle-gen-iris/tests/depth_train_lifecycle.rs::rgb_image/image::open",
        Reference,
        &item.image_path,
        |__model_input_path| image::open(__model_input_path)
    ).unwrap().to_rgb8();
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

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

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn overfit_export_and_load_in_the_estimator() {
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

    let dir = out.path().join("overfit");
    assert_eq!(output.adapter_path, dir.join("model.safetensors"));
    let export = TaskExport::from_dir(&dir, IrisTask::Depth, ID).unwrap();
    assert_eq!(export.settings.name, "depth");
    let meta = weights_metadata(&dir.join("model.safetensors"))
        .unwrap()
        .unwrap();
    assert_eq!(meta["irisTask"], "depth");
    assert_eq!(meta["irisArtifact"], "full_model");
    assert_eq!(meta["irisTrainedSteps"], "80");
    let keys: Vec<String> = tensors_of(&dir.join("model.safetensors"))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let release: Vec<String> = tensors_of(&export_dir().join("model.safetensors"))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(keys, release);
    let prompt = |d: &Path| {
        let c = TaskExport::from_dir(d, IrisTask::Depth, ID).unwrap();
        EmptyPrompt::read(&c.empty_prompt_path(), &c.config).unwrap()
    };
    assert_eq!(prompt(&dir), prompt(&export_dir()));

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
    assert!(target_error(&held, &items[2]) < 3.0);

    let mut gen_spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    gen_spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    assert!(matches!(
        candle_gen_iris::load(&gen_spec),
        Err(CoreError::Unsupported(_))
    ));
    let registry = candle_gen_iris::provider_registry().unwrap();
    assert!(matches!(
        registry.load_trainer("iris_3b", &gen_spec),
        Err(CoreError::Unsupported(_))
    ));
    assert!(matches!(
        registry.load_transform(
            "iris_3b_restore",
            &LoadSpec::new(WeightsSource::Dir(dir.clone()))
        ),
        Err(CoreError::Unsupported(_))
    ));

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::overfit_export_and_load_in_the_estimator");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn parent_init_from_a_file_or_the_text_encoder() {
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
        assert_eq!(
            run_with(
                t.as_mut(),
                &items,
                cfg.clone(),
                out.path(),
                "p.safetensors",
                None
            ),
            2
        );
        let dir = out.path().join("p");
        assert!(estimate(&dir, &rgb_image(&items[0]))
            .iter()
            .all(|v| v.is_finite()));
    }
    let registry = candle_gen_iris::provider_registry().unwrap();
    assert!(registry
        .load_trainer(ID, &LoadSpec::new(WeightsSource::Dir(tiny_backbone())))
        .is_err());

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::parent_init_from_a_file_or_the_text_encoder");
}

/// sc-25690: a checkpointed depth run is the plain run (f32): the trainer wires
/// `gradient_checkpointing` into its step and the export moves by rounding only.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn activation_checkpointing_leaves_the_run_unchanged() {
    use candle_gen_iris::train::model::checkpointed_regions;
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
        assert_eq!(checkpointed_regions() > before, ac);
        tensors_of(&res.adapter_path)
    };
    let (plain, checkpointed) = (export(false, a.path()), export(true, b.path()));
    assert_eq!(plain.len(), checkpointed.len());
    let mut worst = 0f32;
    for ((k, x), (k2, y)) in plain.iter().zip(&checkpointed) {
        assert_eq!(k, k2);
        for (p, q) in x.iter().zip(y) {
            worst = worst.max((p - q).abs());
        }
    }
    eprintln!("depth: activation checkpointing vs plain: max |Δ| = {worst:.3e}");
    assert!(worst <= 1e-5, "{worst}");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_lifecycle::activation_checkpointing_leaves_the_run_unchanged");
}
