//! The Candle Iris restoration trainer end to end on miniature networks (sc-25688) — the twin of
//! `mlx-gen-iris/tests/restoration_train_lifecycle.rs`: the miniature `upscaler/` export as the
//! generator, the parity fixture's miniature timm ConvNeXt as the critic trunk, LPIPS-VGG at
//! reduced widths on formula weights, on the Candle CPU device.
//!
//! * every published loss term is active: zeroing any one weight changes the generator update, and
//!   the update moves with each term;
//! * synthetic second-order degradation and aligned pairs both train; overfitting a few pairs
//!   improves PSNR and LPIPS of the restored output;
//! * cancel → durable checkpoint → resume reproduces the uninterrupted run bit for bit (generator,
//!   EMA, generator optimizer, critic head + spectral buffers, critic optimizer, data position);
//! * the export is the exact `upscaler/` layout, stamped `irisTask=restoration`, loads in the
//!   `iris_3b_restore` transform with no training-only dependency and restores an image; the depth
//!   and generation loaders refuse it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::iris::downstream::{TaskExport, EMPTY_PROMPT_FILE};
use candle_gen::gen_core::iris::restoration::RestorationResources;
use candle_gen::gen_core::iris::restoration_train::{
    self as contract, ConvNextConfig, RestoreTrainPlan, RestoreTrainResources, CKPT_CRITIC,
    CKPT_CRITIC_OPTIMIZER, ITEM_DEGRADED_KEY, OPTIONS_KEY,
};
use candle_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, CKPT_EMA, CKPT_OPTIMIZER, CKPT_TRAINABLE,
};
use candle_gen::gen_core::iris::{GenerationResources, IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    Image, InputSizing, LoadSpec, Precision, TargetSize, Transform, TransformRequest, WeightsSource,
};
use candle_gen::gen_core::{
    LrSchedule, NetworkType, Trainer, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest,
};
use candle_gen_iris::train::critic::ConvNextTrunk;
use candle_gen_iris::train::lpips::Lpips;
use candle_gen_iris::train::optim::snapshot;
use candle_gen_iris::train::restore::{generator_grads, FrozenNets, IrisRestoreTrainer, PairBatch};
use candle_gen_iris::IrisRestorer;
use serde_json::json;

use crate::common::{fixture, fixtures, host_f32};

const LPIPS_WIDTHS: [usize; 5] = [4, 8, 8, 16, 16];
const HEAD_WIDTH: usize = 12;

fn export_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/upscaler")
}

fn mini_trunk_cfg() -> ConvNextConfig {
    ConvNextConfig {
        depths: vec![1, 1, 2, 1],
        dims: vec![8, 16, 24, 32],
        norm_eps: 1e-5,
        embed_dim: 16,
    }
}

fn nets(compute: DType, device: &Device) -> candle_gen::Result<FrozenNets> {
    let g = fixture("iris_restoration_train_golden.safetensors");
    let map: HashMap<String, Tensor> = g
        .tensors
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("critic/w/")
                .map(|s| (s.to_string(), v.clone()))
        })
        .collect();
    Ok(FrozenNets {
        lpips: Lpips::formula(LPIPS_WIDTHS, device)?,
        trunk: ConvNextTrunk::from_map(&map, &mini_trunk_cfg(), compute, device)?,
        trunk_cfg: mini_trunk_cfg(),
        head_width: HEAD_WIDTH,
    })
}

fn resources() -> RestoreTrainResources {
    RestoreTrainResources {
        export: RestorationResources::from_dir(&export_dir(), contract::TRAINER_ID).unwrap(),
        vgg16: PathBuf::new(),
        lpips_lin: PathBuf::new(),
        critic_trunk: PathBuf::new(),
        parent: None,
    }
}

fn trainer() -> IrisRestoreTrainer {
    IrisRestoreTrainer::with_networks(resources(), Box::new(nets), Device::Cpu)
}

fn pattern(w: u32, h: u32, i: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        let f = |a: f32| (127.5 + 120.0 * a.sin()) as u8;
        image::Rgb([
            f(x as f32 * 0.31 + i as f32),
            f(y as f32 * 0.23 + (x / 7) as f32 + 2.0 * i as f32),
            f((x + y) as f32 * 0.17 + 3.0 * i as f32),
        ])
    })
}

/// Clean images (synthetic mode).
fn clean_items(dir: &Path, n: u32) -> Vec<TrainingItem> {
    (0..n)
        .map(|i| {
            let p = dir.join(format!("clean{i}.png"));
            pattern(80 + 8 * i, 72, i).save(&p).unwrap();
            TrainingItem::captioned(p, String::new())
        })
        .collect()
}

/// Aligned ×2 pairs: the degraded image is the 2×2 box average of the clean one.
fn pair_items(dir: &Path, n: u32) -> Vec<TrainingItem> {
    (0..n)
        .map(|i| {
            let gt = pattern(64, 64, i);
            let lq = image::RgbImage::from_fn(32, 32, |x, y| {
                let mut s = [0u32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = gt.get_pixel(2 * x + dx, 2 * y + dy);
                    for c in 0..3 {
                        s[c] += p[c] as u32;
                    }
                }
                image::Rgb([(s[0] / 4) as u8, (s[1] / 4) as u8, (s[2] / 4) as u8])
            });
            let (pg, pl) = (
                dir.join(format!("gt{i}.png")),
                dir.join(format!("lq{i}.png")),
            );
            gt.save(&pg).unwrap();
            lq.save(&pl).unwrap();
            let mut item = TrainingItem::captioned(pg, String::new());
            item.model_options
                .insert(ITEM_DEGRADED_KEY.into(), json!(pl.to_str().unwrap()));
            item
        })
        .collect()
}

/// A miniature degradation that fits a 64px crop (the default needs ≥ ~150 px).
fn mini_degradation() -> serde_json::Value {
    json!({
        "stage2_scale": 2,
        "resize_range": [0.8, 1.2],
        "resize_range2": [0.8, 1.2],
    })
}

fn config(steps: u32, options: serde_json::Value) -> TrainingConfig {
    let mut cfg = TrainingConfig {
        learning_rate: 2e-4,
        steps,
        batch_size: 1,
        resolution: 64,
        save_every: 2,
        seed: 11,
        train_dtype: "f32".into(),
        optimizer: "adamw".into(),
        weight_decay: 0.01,
        lr_scheduler: LrSchedule::Constant,
        full_finetune: true,
        ..Default::default()
    };
    let mut options = options;
    // f32 everywhere: the miniature parity of the whole run (Candle's CPU device has no bf16 GEMM).
    options["critic_dtype"] = json!("fp32");
    cfg.model_options.insert(OPTIONS_KEY.into(), options);
    cfg
}

fn request(items: &[TrainingItem], cfg: TrainingConfig, out: &Path) -> TrainingRequest {
    TrainingRequest {
        items: items.to_vec(),
        config: cfg,
        output_dir: out.to_path_buf(),
        file_name: "restorer.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

fn tensors(path: &Path) -> HashMap<String, Vec<f32>> {
    candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, host_f32(&v)))
        .collect()
}

fn assert_same(a: &Path, b: &Path) {
    let (ta, tb) = (tensors(a), tensors(b));
    assert_eq!(ta.len(), tb.len(), "{}", a.display());
    for (k, v) in &ta {
        assert_eq!(
            Some(v),
            tb.get(k),
            "{k} differs: {} vs {}",
            a.display(),
            b.display()
        );
    }
}

#[test]
fn every_published_loss_term_moves_the_generator_update() {
    let tmp = tempfile::tempdir().unwrap();
    let items = clean_items(tmp.path(), 1);
    let pair = |opts: serde_json::Value| -> Vec<f32> {
        let req = request(&items, config(2, opts), tmp.path());
        let res = resources();
        let plan = RestoreTrainPlan::resolve(&req, &res.export, &[], false).unwrap();
        let pair = contract::training_pair(&items[0], &plan, 1, 0).unwrap();
        let batch = PairBatch::from_pairs(&[pair], &Device::Cpu).unwrap();
        let (model, params, frozen, head, sn) = candle_gen_iris::train::restore::miniature_parts(
            &res,
            &plan,
            Box::new(nets),
            &Device::Cpu,
        )
        .unwrap();
        let (_, grads) = generator_grads(
            &model,
            &params,
            &frozen,
            &plan,
            &snapshot(&head),
            &sn,
            &batch,
            1.0,
        )
        .unwrap();
        let mut keys: Vec<_> = grads.keys().cloned().collect();
        keys.sort();
        keys.iter().flat_map(|k| host_f32(&grads[k])).collect()
    };
    let deg = mini_degradation();
    let all = pair(json!({"degradation": deg}));
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let full = norm(&all);
    assert!(full.is_finite() && full > 0.0, "generator gradient {full}");
    // Each term alone (the others weighted 0): the exact contribution zeroing it removes.
    let terms = ["loss_l2", "loss_lpips", "loss_gan"];
    let mut sum = vec![0f32; all.len()];
    for term in terms {
        let mut o = json!({"degradation": deg});
        for other in terms {
            if other != term {
                o[other] = json!(0.0);
            }
        }
        let only = pair(o);
        let n = norm(&only);
        eprintln!("{term}: |grad_term| / |grad_all| = {:.3e}", n / full);
        assert!(
            n.is_finite() && n > 1e-6 * full,
            "{term} contributes no gradient ({n:e})"
        );
        for (s, v) in sum.iter_mut().zip(&only) {
            *s += v;
        }
        // zeroing the term changes the update by its contribution
        let mut z = json!({"degradation": deg});
        z[term] = json!(0.0);
        let without = pair(z);
        assert_ne!(
            without, all,
            "zeroing {term} must change the generator update"
        );
    }
    // The update is the sum of the three terms (Eq. 6 is a weighted sum).
    let resid: Vec<f32> = sum.iter().zip(&all).map(|(a, b)| a - b).collect();
    let rel = norm(&resid) / full;
    eprintln!("|Σ terms − all| / |all| = {rel:.3e}");
    assert!(
        rel < 1e-2,
        "the generator gradient is not the sum of its terms ({rel:e})"
    );
}

fn psnr_lpips(trainer_dir: &Path, items: &[TrainingItem], export: &Path) -> (f64, f64) {
    let dev = Device::Cpu;
    let res = RestorationResources::from_dir(export, "test").unwrap();
    let ck = candle_gen_iris::nn::Checkpoint::from_tensors(
        candle_gen::candle_core::safetensors::load(res.export.weights_path(), &dev).unwrap(),
        &dev,
    );
    let dit = candle_gen_iris::IrisDiT::from_checkpoint(&ck, &res.config().model, DType::F32, &dev)
        .unwrap();
    let p = res.empty_prompt().unwrap();
    let states = Tensor::from_vec(p.embeddings.clone(), p.shape.to_vec(), &dev).unwrap();
    let lp = Lpips::formula(LPIPS_WIDTHS, &dev).unwrap();
    let mut req = request(items, config(2, json!({"crop": "center"})), trainer_dir);
    req.config.save_every = 0;
    let plan = RestoreTrainPlan::resolve(&req, &resources().export, &[], false).unwrap();
    let (mut psnr, mut lpips) = (0.0, 0.0);
    for (i, item) in items.iter().enumerate() {
        let pair = contract::training_pair(item, &plan, 1, i).unwrap();
        let b = PairBatch::from_pairs(&[pair], &dev).unwrap();
        let t = Tensor::new(&[res.settings.model_time(res.config())], &dev).unwrap();
        let v = dit
            .forward(
                &b.lq,
                &t,
                &candle_gen_iris::TextBatch {
                    states: &states,
                    mask: std::slice::from_ref(&p.mask),
                },
            )
            .unwrap();
        let x = b.lq.sub(&(v * 0.5).unwrap()).unwrap();
        let mse = x
            .sub(&b.gt)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap() as f64;
        psnr += 10.0 * (4.0 / mse).log10();
        lpips += host_f32(&lp.distance(&x, &b.gt).unwrap())[0] as f64;
    }
    (psnr / items.len() as f64, lpips / items.len() as f64)
}

#[test]
fn overfitting_a_few_pairs_improves_psnr_and_lpips() {
    let tmp = tempfile::tempdir().unwrap();
    let items = pair_items(tmp.path(), 2);
    let (p0, l0) = psnr_lpips(tmp.path(), &items, &export_dir());
    let mut cfg = config(
        80,
        json!({"crop": "center", "ema_enabled": false, "loss_gan": 0.05}),
    );
    cfg.save_every = 0;
    cfg.learning_rate = 1e-3;
    let req = request(&items, cfg, tmp.path());
    let out = trainer().train(&req, &mut |_| {}).unwrap();
    let export = out.adapter_path.parent().unwrap().to_path_buf();
    let (p1, l1) = psnr_lpips(tmp.path(), &items, &export);
    eprintln!("PSNR {p0:.2} → {p1:.2} dB; LPIPS {l0:.4} → {l1:.4}");
    assert!(p1 > p0 + 1.0, "PSNR must improve: {p0} → {p1}");
    assert!(l1 < l0, "LPIPS must improve: {l0} → {l1}");
}

#[test]
fn cancel_and_resume_reproduce_the_uninterrupted_run_bit_for_bit() {
    let tmp = tempfile::tempdir().unwrap();
    let items = clean_items(tmp.path(), 3);
    let opts = json!({"degradation": mini_degradation(), "ema_decay": 0.9});
    // uninterrupted: 4 steps (16 micro-steps with accumulation 2)
    let mut cfg = config(16, opts.clone());
    cfg.gradient_accumulation = 2;
    cfg.save_every = 8;
    let a = tmp.path().join("a");
    trainer()
        .train(&request(&items, cfg.clone(), &a), &mut |_| {})
        .unwrap();
    // interrupted after the first checkpoint, then resumed
    let b = tmp.path().join("b");
    let req = request(&items, cfg.clone(), &b);
    let cancel = req.cancel.clone();
    let out = trainer()
        .train(&req, &mut |p| {
            if let TrainingProgress::Checkpoint { step: 8 } = p {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!(
        out.steps, 8,
        "the cancelled run stops at the step boundary after the cancel"
    );
    let mut cfg2 = cfg.clone();
    cfg2.resume = true;
    trainer()
        .train(&request(&items, cfg2, &b), &mut |_| {})
        .unwrap();
    let ca = latest_checkpoint(&checkpoint_root(&a, "restorer.safetensors")).unwrap();
    let cb = latest_checkpoint(&checkpoint_root(&b, "restorer.safetensors")).unwrap();
    for f in [
        CKPT_TRAINABLE,
        CKPT_EMA,
        CKPT_OPTIMIZER,
        CKPT_CRITIC,
        CKPT_CRITIC_OPTIMIZER,
    ] {
        assert_same(&ca.join(f), &cb.join(f));
    }
    assert_same(
        &a.join("restorer/model.safetensors"),
        &b.join("restorer/model.safetensors"),
    );
    let sa = contract::read_checkpoint_state(&ca).unwrap();
    let sb = contract::read_checkpoint_state(&cb).unwrap();
    assert_eq!(
        (sa.step, sa.epoch, sa.batches_consumed),
        (sb.step, sb.epoch, sb.batches_consumed)
    );
    assert_eq!(sa.losses, sb.losses);
    // a generation checkpoint dir / a different recipe refuses to resume
    let mut other = cfg.clone();
    other.resume = true;
    other.full_finetune = false;
    other.rank = 2;
    other.alpha = 2.0;
    assert!(trainer()
        .train(&request(&items, other, &b), &mut |_| {})
        .is_err());
}

#[test]
fn the_export_is_a_restoration_model_the_transform_runs_and_other_tasks_refuse() {
    let tmp = tempfile::tempdir().unwrap();
    let items = clean_items(tmp.path(), 2);
    let mut cfg = config(4, json!({"degradation": mini_degradation()}));
    cfg.full_finetune = false;
    cfg.network_type = NetworkType::Lora;
    cfg.rank = 2;
    cfg.alpha = 2.0;
    cfg.sample_every = 2;
    let req = request(&items, cfg, tmp.path());
    let mut previews = 0;
    let out = trainer()
        .train(&req, &mut |p| {
            if let TrainingProgress::Sample { image, .. } = p {
                assert_eq!((image.width, image.height), (64, 64));
                previews += 1;
            }
        })
        .unwrap();
    assert_eq!(previews, 2);
    let dir = out.adapter_path.parent().unwrap().to_path_buf();
    // exact upscaler/ layout: config.yaml + empty prompt copied verbatim, model.safetensors
    for f in ["config.yaml", EMPTY_PROMPT_FILE] {
        assert_eq!(
            std::fs::read(dir.join(f)).unwrap(),
            std::fs::read(export_dir().join(f)).unwrap(),
            "{f} keeps the export identity"
        );
    }
    let bytes = std::fs::read(&out.adapter_path).unwrap();
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let meta = header.metadata().clone().unwrap();
    assert_eq!(meta["irisTask"], "restoration");
    assert_eq!(meta["irisArtifact"], "full_model");
    assert_eq!(meta["irisWeights"], "ema");
    // the merged LoRA changed the base
    let base = tensors(&export_dir().join("model.safetensors"));
    let merged = tensors(&out.adapter_path);
    assert_eq!(base.len(), merged.len());
    assert!(base.iter().any(|(k, v)| merged[k] != *v));
    // loads in iris_3b_restore with no training-only dependency and restores
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    let restorer = IrisRestorer::load_on(&spec, &Device::Cpu).unwrap();
    let img = Image {
        width: 20,
        height: 16,
        pixels: (0..20 * 16 * 3).map(|i| (i * 7 % 251) as u8).collect(),
    };
    let treq = TransformRequest {
        image: img,
        target: TargetSize::Scale(2.0),
        input_sizing: InputSizing::Original,
        ..Default::default()
    };
    let restored = restorer.apply(&treq, &mut |_| {}).unwrap();
    assert_eq!((restored.width, restored.height), (40, 32));
    // other tasks refuse it
    assert!(matches!(
        TaskExport::from_dir(&dir, IrisTask::Depth, "iris_3b_depth"),
        Err(candle_gen::gen_core::Error::Unsupported(_))
    ));
    let mut gen = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    gen.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(crate::common::tiny_text_encoder()),
    );
    assert!(matches!(
        GenerationResources::from_spec(&gen, "iris_3b"),
        Err(candle_gen::gen_core::Error::Unsupported(_))
    ));
}

/// sc-25690: the restoration trainer's activation checkpointing (`gradient_checkpointing`) — the
/// generation trainer's trunk / pixel regions around the generator in the generator step, the
/// critic and LPIPS terms downstream of them. f32: the generator gradients are the plain graph's;
/// bf16 generator (f32 critic, as the CPU backend requires): the same loss terms bit for bit and
/// gradients within the declared bf16 bound.
#[test]
fn activation_checkpointing_gradients_equal_the_plain_graph() {
    use candle_gen_iris::train::model::checkpointed_regions;
    let tmp = tempfile::tempdir().unwrap();
    let items = clean_items(tmp.path(), 1);
    let case = |dtype: &str, ac: bool| {
        let mut cfg = config(2, json!({"degradation": mini_degradation()}));
        cfg.train_dtype = dtype.into();
        cfg.gradient_checkpointing = ac;
        let req = request(&items, cfg, tmp.path());
        let res = resources();
        let plan = RestoreTrainPlan::resolve(&req, &res.export, &[], false).unwrap();
        assert_eq!(plan.ac.active(), ac);
        let pair = contract::training_pair(&items[0], &plan, 1, 0).unwrap();
        let batch = PairBatch::from_pairs(&[pair], &Device::Cpu).unwrap();
        let (model, params, frozen, head, sn) = candle_gen_iris::train::restore::miniature_parts(
            &res,
            &plan,
            Box::new(nets),
            &Device::Cpu,
        )
        .unwrap();
        let before = checkpointed_regions();
        let (terms, grads) = generator_grads(
            &model,
            &params,
            &frozen,
            &plan,
            &snapshot(&head),
            &sn,
            &batch,
            1.0,
        )
        .unwrap();
        let entered = checkpointed_regions() - before;
        assert_eq!(entered > 0, ac, "regions entered: {entered}");
        (terms, grads)
    };
    let (f_terms, f_none) = case("f32", false);
    let (f_terms_ac, f_full) = case("f32", true);
    assert_eq!(
        f_terms.map(f64::to_bits),
        f_terms_ac.map(f64::to_bits),
        "f32 loss terms"
    );
    // 1e-5, not the other tasks' 1e-6: the adversarial term is O(10²) here and its gradient to
    // the text embedder is a difference of large partials (measured: 3.8e-6 with every term,
    // 1.6e-8 with the L2 term alone).
    crate::common::assert_ac_f32_equal("restoration", 1e-5, &f_none, &f_full);
    let (b_terms, b_none) = case("bf16", false);
    let (b_terms_ac, b_full) = case("bf16", true);
    assert_eq!(
        b_terms.map(f64::to_bits),
        b_terms_ac.map(f64::to_bits),
        "bf16 loss terms"
    );
    crate::common::assert_ac_bf16_noise_only("restoration", &f_none, &b_none, &b_full);
}

/// sc-25690: a checkpointed restoration run is the plain run (f32): the trainer wires
/// `gradient_checkpointing` into its generator step and the export moves by rounding only.
#[test]
fn activation_checkpointing_leaves_the_run_unchanged() {
    use candle_gen_iris::train::model::checkpointed_regions;
    let (data, a, b) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let items = clean_items(data.path(), 1);
    let run = |ac: bool, out: &Path| {
        let mut cfg = config(2, json!({"degradation": mini_degradation()}));
        cfg.save_every = 0;
        cfg.gradient_checkpointing = ac;
        let before = checkpointed_regions();
        let res = trainer()
            .train(&request(&items, cfg, out), &mut |_| {})
            .unwrap();
        assert_eq!(checkpointed_regions() > before, ac);
        tensors(&res.adapter_path)
    };
    let (plain, checkpointed) = (run(false, a.path()), run(true, b.path()));
    assert_eq!(plain.len(), checkpointed.len());
    let mut worst = 0f32;
    for (k, v) in &plain {
        for (x, y) in v.iter().zip(&checkpointed[k]) {
            worst = worst.max((x - y).abs());
        }
    }
    eprintln!("restoration: activation checkpointing vs plain: max |Δ| = {worst:.3e}");
    assert!(worst <= 1e-5, "{worst}");
}
