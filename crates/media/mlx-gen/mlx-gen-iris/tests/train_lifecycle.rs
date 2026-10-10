//! The Iris generation trainer end to end on the miniature snapshot, through the explicit catalog's
//! `load_trainer` (sc-25685): the gen-core trainer conformance suite, cancel → durable checkpoint →
//! resume reproducing the uninterrupted run bit for bit (full + Muon, LoRA + AdamW with
//! accumulation), retention, the resume identity checks, cached vs on-the-fly conditioning, random
//! init + x-prediction, previews from the in-progress state, and the exported artifacts loading back
//! (the full model through the inference provider; the adapter through the provider's own adapter
//! loader, rendering the trainer's own preview up to residual-vs-merged rounding).

use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, list_checkpoints, read_checkpoint_state, AdapterMetadata,
    OPTIONS_KEY,
};
use mlx_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{
    LrSchedule, NetworkType, Trainer, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_iris::train::render_preview_with;
use mlx_gen_iris::{load_backbone_with_adapters, IrisDiT, IrisTextEncoder};
use mlx_rs::{Array, Dtype};
use serde_json::json;

use crate::common::{on_cpu, tiny_backbone, tiny_config, tiny_text_encoder};

const ID: &str = "iris_3b";

fn spec(backbone: &Path) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(backbone.to_path_buf()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    spec
}

fn trainer() -> Box<dyn Trainer> {
    mlx_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(ID, &spec(&tiny_backbone()))
        .expect("the tiny snapshot loads as a trainer")
}

/// Three small RGB images of different aspect ratios + captions from the tiny tokenizer's vocab.
fn dataset(dir: &Path) -> Vec<TrainingItem> {
    let captions = ["a red fox in the snow", "golden hour", "a fox , snow"];
    let sizes = [(24u32, 20u32), (20, 28), (16, 16)];
    captions
        .iter()
        .zip(sizes)
        .enumerate()
        .map(|(i, (caption, (w, h)))| {
            let img = image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([
                    (x * 9 + i as u32 * 40) as u8,
                    (y * 7 + 30) as u8,
                    ((x + y) * 5 + i as u32 * 70) as u8,
                ])
            });
            let path = dir.join(format!("img{i}.png"));
            img.save(&path).unwrap();
            TrainingItem::captioned(path, caption.to_string())
        })
        .collect()
}

fn config() -> TrainingConfig {
    TrainingConfig {
        rank: 2,
        alpha: 4.0,
        learning_rate: 1e-3,
        steps: 4,
        batch_size: 2,
        resolution: 16,
        save_every: 2,
        seed: 9,
        train_dtype: "f32".into(),
        lr_scheduler: LrSchedule::Cosine,
        lr_warmup_steps: 1,
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

fn tensors(path: &Path) -> HashMap<String, Vec<f32>> {
    Array::load_safetensors(path)
        .unwrap()
        .into_iter()
        .map(|(k, v)| {
            let v = v.as_dtype(Dtype::Float32).unwrap();
            let n: i32 = v.shape().iter().product();
            (k, v.reshape(&[n]).unwrap().as_slice::<f32>().to_vec())
        })
        .collect()
}

fn assert_same_files(a: &Path, b: &Path) {
    let (ta, tb) = (tensors(a), tensors(b));
    assert_eq!(ta.len(), tb.len(), "{} vs {}", a.display(), b.display());
    for (k, va) in &ta {
        assert_eq!(
            Some(va),
            tb.get(k),
            "{k} differs between {} and {}",
            a.display(),
            b.display()
        );
    }
}

#[test]
fn gen_core_trainer_conformance() {
    let data = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let profile =
        gen_core_testkit::trainer::TrainerProfile::cheap(items[..2].to_vec(), out.path().into());
    gen_core_testkit::trainer::trainer_conformance(trainer, &profile);
    let registry = mlx_gen_iris::provider_registry().unwrap();
    gen_core_testkit::trainer::check_trainer_registry(&registry, trainer().as_ref()).unwrap();
}

/// Train `cfg` to completion in `out`, cancelling right after optimizer step `cancel_at` when set.
fn run(
    items: &[TrainingItem],
    cfg: TrainingConfig,
    out: &Path,
    name: &str,
    cancel_at: Option<u32>,
) -> u32 {
    let req = request(items, cfg, out, name);
    let flag = req.cancel.clone();
    let mut t = trainer();
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

fn resume_case(mut cfg: TrainingConfig, name: &str) {
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        assert_eq!(run(&items, cfg.clone(), a.path(), name, None), 4);
        // Interrupted after step 2 (a published checkpoint), then resumed to 4.
        assert_eq!(run(&items, cfg.clone(), b.path(), name, Some(2)), 2);
        let root_b = checkpoint_root(b.path(), name);
        assert_eq!(
            read_checkpoint_state(&latest_checkpoint(&root_b).unwrap())
                .unwrap()
                .step,
            2
        );
        cfg.resume = true;
        assert_eq!(run(&items, cfg, b.path(), name, None), 4);
        let (la, lb) = (
            latest_checkpoint(&checkpoint_root(a.path(), name)).unwrap(),
            latest_checkpoint(&root_b).unwrap(),
        );
        for f in [
            "trainable.safetensors",
            "ema.safetensors",
            "optimizer.safetensors",
        ] {
            assert_same_files(&la.join(f), &lb.join(f));
        }
        let (sa, sb) = (
            read_checkpoint_state(&la).unwrap(),
            read_checkpoint_state(&lb).unwrap(),
        );
        assert_eq!(
            (sa.step, sa.epoch, sa.batches_consumed, sa.scheduler_step),
            (4, 2, 2, 4)
        );
        assert_eq!(
            sa, sb,
            "state.json of the resumed run equals the uninterrupted one"
        );
        // The exported artifacts agree too.
        let artifact = |dir: &Path| -> PathBuf {
            let p = dir.join(name);
            if p.is_file() {
                p
            } else {
                dir.join(name.trim_end_matches(".safetensors"))
                    .join("model.safetensors")
            }
        };
        assert_same_files(&artifact(a.path()), &artifact(b.path()));
    });
}

#[test]
fn full_muon_cancel_and_resume_reproduce_the_uninterrupted_run() {
    let mut cfg = config();
    cfg.full_finetune = true;
    cfg.optimizer = "muon".into();
    cfg.weight_decay = 0.01;
    options(&mut cfg, json!({"text_dropout": 0.5, "ema_decay": 0.9}));
    resume_case(cfg, "full.safetensors");
}

#[test]
fn lora_accumulated_cancel_and_resume_reproduce_the_uninterrupted_run() {
    let mut cfg = config();
    cfg.batch_size = 1;
    cfg.gradient_accumulation = 2;
    cfg.network_type = NetworkType::Lora;
    // 3 items, batch 1, accum 2: windows straddle the epoch boundary (accelerate's global count).
    options(
        &mut cfg,
        json!({"text_dropout": 0.5, "ema_decay": 0.9, "caption_fields": ["caption"]}),
    );
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let name = "lora.safetensors";
        assert_eq!(run(&items, cfg.clone(), a.path(), name, None), 4);
        assert_eq!(run(&items, cfg.clone(), b.path(), name, Some(2)), 2);
        let mut resumed = cfg.clone();
        resumed.resume = true;
        assert_eq!(run(&items, resumed, b.path(), name, None), 4);
        let (la, lb) = (
            latest_checkpoint(&checkpoint_root(a.path(), name)).unwrap(),
            latest_checkpoint(&checkpoint_root(b.path(), name)).unwrap(),
        );
        for f in [
            "trainable.safetensors",
            "ema.safetensors",
            "optimizer.safetensors",
        ] {
            assert_same_files(&la.join(f), &lb.join(f));
        }
        // 8 micro-batches over 3 items: epoch 3, position 2.
        let s = read_checkpoint_state(&la).unwrap();
        assert_eq!((s.step, s.epoch, s.batches_consumed), (4, 3, 2));
        assert_same_files(&a.path().join(name), &b.path().join(name));
    });
}

#[test]
fn resume_identity_retention_and_new_phase() {
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let out = tempfile::tempdir().unwrap();
        let name = "keep.safetensors";
        let mut cfg = config();
        cfg.save_every = 1;
        options(
            &mut cfg,
            json!({"keep_last_checkpoints": 1, "milestone_steps": [2]}),
        );
        assert_eq!(run(&items, cfg.clone(), out.path(), name, None), 4);
        let root = checkpoint_root(out.path(), name);
        let steps: Vec<u64> = list_checkpoints(&root)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        assert_eq!(steps, [2, 4], "newest one kept plus the milestone");

        // An exact resume on a different dataset is refused...
        let mut changed = items.clone();
        changed[0].caption = "snow".into();
        let mut resume = cfg.clone();
        resume.resume = true;
        resume.steps = 6;
        let req = request(&changed, resume.clone(), out.path(), name);
        let err = trainer().train(&req, &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("dataset differs"), "{err}");
        // ...while a new phase keeps the trained state and starts the new data at epoch 1.
        options(
            &mut resume,
            json!({"keep_last_checkpoints": 1, "milestone_steps": [2], "resume_data_policy": "new_phase"}),
        );
        assert_eq!(run(&changed, resume, out.path(), name, None), 6);
        let s = read_checkpoint_state(&latest_checkpoint(&root).unwrap()).unwrap();
        assert_eq!((s.step, s.epoch, s.batches_consumed), (6, 1, 2));
        // A different adapter shape is never resumed into.
        let mut other = cfg.clone();
        other.resume = true;
        other.rank = 4;
        other.steps = 8;
        let req = request(&items, other, out.path(), name);
        let err = trainer().train(&req, &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("different state"), "{err}");
    });
}

#[test]
fn cached_and_on_the_fly_conditioning_train_identically() {
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let mut cfg = config();
        cfg.steps = 2;
        options(
            &mut cfg,
            json!({"text_conditioning": "on_the_fly", "text_dropout": 0.3}),
        );
        run(&items, cfg.clone(), a.path(), "c.safetensors", None);
        options(
            &mut cfg,
            json!({"text_conditioning": "cached", "text_dropout": 0.3}),
        );
        run(&items, cfg, b.path(), "c.safetensors", None);
        assert_same_files(
            &a.path().join("c.safetensors"),
            &b.path().join("c.safetensors"),
        );
    });
}

/// The adapter artifact, merged into the backbone with the documented schema, renders exactly
/// the image the trainer previewed from its in-progress raw weights at the last step.
fn adapter_round_trip(network: NetworkType) {
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let out = tempfile::tempdir().unwrap();
        let mut cfg = config();
        cfg.steps = 2;
        cfg.network_type = network;
        // A large rate so two steps visibly move the render (the comparison below needs an effect).
        cfg.learning_rate = 5e-2;
        cfg.sample_every = 2;
        cfg.sample_prompts = vec!["a red fox".into()];
        cfg.sample_steps = 3;
        cfg.sample_guidance_scale = 2.0;
        options(
            &mut cfg,
            json!({"export_weights": "raw", "preview_weights": "raw"}),
        );
        let req = request(&items, cfg, out.path(), "style.safetensors");
        let mut previews = Vec::new();
        let res = trainer()
            .train(&req, &mut |p| {
                if let TrainingProgress::Sample {
                    step,
                    image,
                    prompt,
                    ..
                } = p
                {
                    previews.push((step, prompt, image));
                }
            })
            .unwrap();
        // Upstream renders at step 1 and every `sample_every`.
        assert_eq!(previews.iter().map(|p| p.0).collect::<Vec<_>>(), [1, 2]);
        let (_, prompt, image) = previews.pop().unwrap();
        assert_eq!((image.width, image.height), (16, 16));

        let (_, meta) = Array::load_safetensors_with_metadata(&res.adapter_path).unwrap();
        let meta = AdapterMetadata::from_map(&meta.into_iter().collect()).unwrap();
        assert_eq!(meta.steps, 2);
        assert!(!meta.targets.is_empty());

        // Load through the provider's adapter path (S3's strict loader: identity stamp checked,
        // forward-time residuals) and render with the same sampler, prompt and seed.
        let cfg = tiny_config();
        let kind = match network {
            NetworkType::Lora => AdapterKind::Lora,
            NetworkType::Lokr => AdapterKind::Lokr,
        };
        let te = IrisTextEncoder::load(&tiny_text_encoder(), &cfg.text_encoder).unwrap();
        let (c, u) = (te.encode(&prompt).unwrap(), te.encode("").unwrap());
        let render = |adapters: &[AdapterSpec]| {
            let (dit, reports) = load_backbone_with_adapters(
                &tiny_backbone(),
                &cfg,
                Dtype::Float32,
                adapters,
                IrisTask::Generation,
                ID,
            )
            .unwrap();
            assert_eq!(reports.len(), adapters.len());
            render_preview_with(
                &dit,
                3,
                2.0,
                cfg.flow.shift,
                1000,
                mlx_gen::gen_core::iris::train::Prediction::Velocity,
                16,
                (&c.states, &c.mask),
                Some((&u.states, &u.mask)),
                9,
            )
            .unwrap()
        };
        let loaded = render(&[AdapterSpec::new(res.adapter_path.clone(), 1.0, kind)]);
        let bare = render(&[]);
        let max_diff = |a: &[u8], b: &[u8]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max()
                .unwrap()
        };
        let (to_preview, effect) = (
            max_diff(&loaded.pixels, &image.pixels),
            max_diff(&bare.pixels, &image.pixels),
        );
        eprintln!(
            "{network:?}: |provider − preview| max {to_preview}, adapter effect max {effect}"
        );
        // The provider applies the factors as residuals (LoKr's factors reconstructed in bf16), the
        // trainer merges them in f32: a rounding-level difference, far below the adapter's effect.
        assert!(
            effect >= 10,
            "the trained adapter must visibly change the render ({effect})"
        );
        assert!(
            to_preview <= 2,
            "provider-loaded adapter vs preview: {to_preview}"
        );
    });
}

#[test]
fn lora_adapter_exports_and_loads_through_the_provider() {
    adapter_round_trip(NetworkType::Lora);
}

#[test]
fn lokr_adapter_exports_and_loads_through_the_provider() {
    adapter_round_trip(NetworkType::Lokr);
}

#[test]
fn exported_full_model_loads_and_generates_in_the_provider() {
    let data = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let out = tempfile::tempdir().unwrap();
    let mut cfg = config();
    cfg.steps = 2;
    cfg.full_finetune = true;
    cfg.train_dtype = "bf16".into();
    options(&mut cfg, json!({"ema_decay": 0.5, "export_dtype": "bf16"}));
    let req = request(&items, cfg, out.path(), "tuned.safetensors");
    let res = trainer().train(&req, &mut |_| {}).unwrap();
    let dir = res.adapter_path.parent().unwrap().to_path_buf();
    assert_eq!(dir, out.path().join("tuned"));
    let (_, meta) = Array::load_safetensors_with_metadata(&res.adapter_path).unwrap();
    assert_eq!(meta["irisArtifact"], "full_model");
    assert_eq!(meta["irisWeights"], "ema");
    assert_eq!(IrisConfig::from_dir(&dir).unwrap(), tiny_config());
    let g = mlx_gen_iris::provider_registry()
        .unwrap()
        .load(ID, &spec(&dir))
        .unwrap();
    let image = match g
        .generate(
            &GenerationRequest {
                prompt: "a red fox".into(),
                width: 16,
                height: 16,
                steps: Some(2),
                seed: Some(1),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap()
    {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => panic!("expected an image"),
    };
    assert_eq!((image.width, image.height), (16, 16));
}

#[test]
fn random_init_x_prediction_trains_and_its_export_records_the_objective() {
    let data = tempfile::tempdir().unwrap();
    let items = dataset(data.path());
    let out = tempfile::tempdir().unwrap();
    let mut cfg = config();
    cfg.steps = 2;
    cfg.full_finetune = true;
    cfg.train_dtype = "bf16".into();
    options(
        &mut cfg,
        json!({"init": "random", "prediction": "x", "ema_enabled": false}),
    );
    let req = request(&items, cfg, out.path(), "scratch.safetensors");
    let losses = Cell::new(0usize);
    let res = trainer()
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { loss, .. } = p {
                assert!(loss.is_finite());
                losses.set(losses.get() + 1);
            }
        })
        .unwrap();
    assert_eq!((res.steps, losses.get()), (2, 2));
    let dir = res.adapter_path.parent().unwrap();
    let text = std::fs::read_to_string(dir.join("config.yaml")).unwrap();
    assert!(text.contains("prediction: x"), "{text}");
    // Random init: the zero-initialised head moved off zero after two steps.
    let w = tensors(&res.adapter_path);
    assert!(w["final_layer.linear.weight"].iter().any(|v| *v != 0.0));
    // The exported x-prediction backbone loads in the provider, which reads `prediction: x` from
    // its config: its render equals the trainer's own clean-image sampler on the same weights,
    // and differs from treating the output as a velocity.
    let g = mlx_gen_iris::provider_registry()
        .unwrap()
        .load(ID, &spec(dir))
        .expect("the provider serves an x-prediction backbone");
    let provider = match g
        .generate(
            &GenerationRequest {
                prompt: "a red fox".into(),
                width: 16,
                height: 16,
                steps: Some(3),
                guidance: Some(2.0),
                seed: Some(4),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap()
    {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => panic!("expected an image"),
    };
    let cfg = IrisConfig::from_dir(dir).unwrap();
    let dit = mlx_gen_iris::load_backbone(dir, &cfg, Dtype::Bfloat16).unwrap();
    let te = IrisTextEncoder::load(&tiny_text_encoder(), &cfg.text_encoder).unwrap();
    let (c, u) = (te.encode("a red fox").unwrap(), te.encode("").unwrap());
    let ours = |p| {
        render_preview_with(
            &dit,
            3,
            2.0,
            cfg.flow.shift,
            1000,
            p,
            16,
            (&c.states, &c.mask),
            Some((&u.states, &u.mask)),
            4,
        )
        .unwrap()
    };
    use mlx_gen::gen_core::iris::train::Prediction;
    assert_eq!(ours(Prediction::Clean).pixels, provider.pixels);
    assert_ne!(ours(Prediction::Velocity).pixels, provider.pixels);
}

/// The generation trainer needs its frozen text encoder: a load without the `text_encoder`
/// component (or with a file in its place) is a load-time error naming it.
#[test]
fn a_trainer_load_without_the_text_encoder_is_refused() {
    let registry = mlx_gen_iris::provider_registry().unwrap();
    let bare = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    let err = registry.load_trainer(ID, &bare).err().expect("refused");
    assert!(err.to_string().contains(TEXT_ENCODER_COMPONENT), "{err}");
    let mut file = bare.clone();
    file.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::File(tiny_text_encoder().join("model.safetensors")),
    );
    let err = registry.load_trainer(ID, &file).err().expect("refused");
    assert!(err.to_string().contains(TEXT_ENCODER_COMPONENT), "{err}");
}

/// `text_dropout` substitutes the CFG null for the dropped rows: at probability 1 every row is the
/// null, so two datasets differing only in their captions train identically — and at 0 they do not.
#[test]
fn caption_dropout_substitutes_the_null_conditioning() {
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let mut recaptioned = items.clone();
        for it in &mut recaptioned {
            it.caption = "snow".into();
        }
        let train = |items: &[TrainingItem], dropout: f64| -> HashMap<String, Vec<f32>> {
            let out = tempfile::tempdir().unwrap();
            let mut cfg = config();
            cfg.steps = 1;
            cfg.lr_warmup_steps = 0;
            options(
                &mut cfg,
                json!({"text_dropout": dropout, "ema_enabled": false}),
            );
            run(items, cfg, out.path(), "d.safetensors", None);
            tensors(&out.path().join("d.safetensors"))
        };
        assert_eq!(train(&items, 1.0), train(&recaptioned, 1.0));
        assert_ne!(train(&items, 0.0), train(&recaptioned, 0.0));
    });
}

/// `train/lr.py`'s warmup ramps from exactly 0: the first optimizer step of a warmed-up run runs at
/// lr 0 and leaves the (zero-decay) weights untouched; without warmup the same step moves them.
#[test]
fn warmup_starts_the_schedule_at_zero() {
    on_cpu(|| {
        let data = tempfile::tempdir().unwrap();
        let items = dataset(data.path());
        let out = tempfile::tempdir().unwrap();
        let mut cfg = config();
        cfg.steps = 1;
        cfg.full_finetune = true;
        cfg.lr_warmup_steps = 3;
        options(&mut cfg, json!({"ema_enabled": false}));
        run(&items, cfg.clone(), out.path(), "w.safetensors", None);
        let base = tensors(&tiny_backbone().join("model.safetensors"));
        let trained = tensors(&out.path().join("w").join("model.safetensors"));
        assert_eq!(trained, base, "lr factor 0 at scheduler step 0");
        cfg.lr_warmup_steps = 0;
        let out = tempfile::tempdir().unwrap();
        run(&items, cfg, out.path(), "w.safetensors", None);
        assert_ne!(
            tensors(&out.path().join("w").join("model.safetensors")),
            base
        );
    });
}

/// The preview renderer is the provider's sampler: on the same backbone, prompt, steps, CFG and
/// seed it reproduces the catalog generator's image exactly.
#[test]
fn preview_renderer_matches_the_provider() {
    let cfg = tiny_config();
    let g = mlx_gen_iris::provider_registry()
        .unwrap()
        .load(ID, &spec(&tiny_backbone()))
        .unwrap();
    let want = match g
        .generate(
            &GenerationRequest {
                prompt: "a red fox".into(),
                width: 16,
                height: 16,
                steps: Some(4),
                guidance: Some(2.0),
                seed: Some(5),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap()
    {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => panic!("expected an image"),
    };
    let map: HashMap<String, Array> = Weights::from_file(tiny_backbone().join("model.safetensors"))
        .unwrap()
        .into_tensors()
        .into_iter()
        .map(|(k, v)| {
            let v = mlx_gen_iris::train::model::provider_dtype(&k, &v, Dtype::Bfloat16).unwrap();
            (k, v)
        })
        .collect();
    let dit = IrisDiT::from_weights(&Weights::from_map(map), &cfg.model, Dtype::Bfloat16).unwrap();
    let te = IrisTextEncoder::load(&tiny_text_encoder(), &cfg.text_encoder).unwrap();
    let (c, u) = (te.encode("a red fox").unwrap(), te.encode("").unwrap());
    let got = render_preview_with(
        &dit,
        4,
        2.0,
        cfg.flow.shift,
        1000,
        mlx_gen::gen_core::iris::train::Prediction::Velocity,
        16,
        (&c.states, &c.mask),
        Some((&u.states, &u.mask)),
        5,
    )
    .unwrap();
    assert_eq!(got.pixels, want.pixels);
}
