//! The MLX Iris restoration trainer end to end on miniature networks (sc-25688): the miniature
//! `upscaler/` export as the generator, the miniature timm ConvNeXt of the parity fixture as the
//! critic trunk, LPIPS-VGG at reduced widths on formula weights. Each case runs in a CPU-default child
//! process (`common::cpu_process`, sc-25689: no op reaches the GPU, which the dev Mac's other work
//! owns; run-to-run deterministic since the fork's CPU-conv border fix — see
//! `restoration_train_parity`). The Metal production path is the dispatch-only real-weight lane's.
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

use mlx_gen::gen_core::iris::downstream::{TaskExport, EMPTY_PROMPT_FILE};
use mlx_gen::gen_core::iris::restoration::RestorationResources;
use mlx_gen::gen_core::iris::restoration_train::{
    self as contract, ConvNextConfig, RestoreTrainPlan, RestoreTrainResources, CKPT_CRITIC,
    CKPT_CRITIC_OPTIMIZER, ITEM_DEGRADED_KEY, OPTIONS_KEY,
};
use mlx_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, CKPT_EMA, CKPT_OPTIMIZER, CKPT_TRAINABLE,
};
use mlx_gen::gen_core::iris::{GenerationResources, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{
    LrSchedule, NetworkType, Trainer, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    Image, InputSizing, LoadSpec, Precision, TargetSize, Transform, TransformRequest, WeightsSource,
};
use mlx_gen_iris::train::critic::ConvNextTrunk;
use mlx_gen_iris::train::lpips::Lpips;
use mlx_gen_iris::train::restore::{generator_grads, FrozenNets, IrisRestoreTrainer, PairBatch};
use mlx_gen_iris::IrisRestorer;
use mlx_rs::{Array, Dtype};
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

fn nets(compute: Dtype) -> mlx_gen::Result<FrozenNets> {
    let g = fixture("iris_restoration_train_golden.safetensors");
    let map: HashMap<String, Array> = g
        .keys()
        .filter_map(|k| {
            k.strip_prefix("critic/w/")
                .map(|s| (s.to_string(), g.get(k).unwrap().clone()))
        })
        .collect();
    Ok(FrozenNets {
        lpips: Lpips::formula(LPIPS_WIDTHS)?,
        trunk: ConvNextTrunk::from_map(&map, &mini_trunk_cfg(), compute)?,
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
    IrisRestoreTrainer::with_networks(resources(), Box::new(nets))
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
    Array::load_safetensors(path)
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
    crate::common::cpu_process(
        "restoration_train_lifecycle::every_published_loss_term_moves_the_generator_update",
        every_published_loss_term_moves_the_generator_update_body,
    );
}

fn every_published_loss_term_moves_the_generator_update_body() {
    let tmp = tempfile::tempdir().unwrap();
    let items = clean_items(tmp.path(), 1);
    let pair = |opts: serde_json::Value| -> Vec<f32> {
        let req = request(&items, config(2, opts), tmp.path());
        let res = resources();
        let plan = RestoreTrainPlan::resolve(&req, &res.export, &[], false).unwrap();
        let pair = contract::training_pair(&items[0], &plan, 1, 0).unwrap();
        let batch = PairBatch::from_pairs(&[pair]);
        let (model, params, frozen, head, sn) =
            mlx_gen_iris::train::restore::miniature_parts(&res, &plan, Box::new(nets)).unwrap();
        let (_, grads) =
            generator_grads(&model, &params, &frozen, &plan, &head, &sn, &batch, 1.0).unwrap();
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
    // Restore each pair's degraded input with the given export directory's weights.
    let res = RestorationResources::from_dir(export, "test").unwrap();
    let _ = trainer_dir;
    let w = Weights::from_file(res.export.weights_path()).unwrap();
    let dit = mlx_gen_iris::IrisDiT::from_weights(&w, &res.config().model, Dtype::Float32).unwrap();
    let p = res.empty_prompt().unwrap();
    let states = Array::from_slice(&p.embeddings, &[1, 10, 12, 32]);
    let lp = Lpips::formula(LPIPS_WIDTHS).unwrap();
    let mut req = request(items, config(2, json!({"crop": "center"})), trainer_dir);
    req.config.save_every = 0;
    let plan = RestoreTrainPlan::resolve(&req, &resources().export, &[], false).unwrap();
    let (mut psnr, mut lpips) = (0.0, 0.0);
    for (i, item) in items.iter().enumerate() {
        let pair = contract::training_pair(item, &plan, 1, i).unwrap();
        let b = PairBatch::from_pairs(&[pair]);
        let t = Array::from_slice(&[res.settings.model_time(res.config())], &[1]);
        let v = dit
            .forward(
                &b.lq,
                &t,
                &mlx_gen_iris::TextBatch {
                    states: &states,
                    mask: std::slice::from_ref(&p.mask),
                },
            )
            .unwrap();
        let x =
            b.lq.subtract(v.multiply(Array::from_f32(0.5)).unwrap())
                .unwrap();
        let mse = host_f32(
            &x.subtract(&b.gt)
                .unwrap()
                .square()
                .unwrap()
                .mean(None)
                .unwrap(),
        )[0] as f64;
        // [-1, 1] range: peak-to-peak 2
        psnr += 10.0 * (4.0 / mse).log10();
        lpips += host_f32(&lp.distance(&x, &b.gt).unwrap())[0] as f64;
    }
    (psnr / items.len() as f64, lpips / items.len() as f64)
}

#[test]
fn overfitting_a_few_pairs_improves_psnr_and_lpips() {
    crate::common::cpu_process(
        "restoration_train_lifecycle::overfitting_a_few_pairs_improves_psnr_and_lpips",
        overfitting_a_few_pairs_improves_psnr_and_lpips_body,
    );
}

fn overfitting_a_few_pairs_improves_psnr_and_lpips_body() {
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
    crate::common::cpu_process("restoration_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run_bit_for_bit", cancel_and_resume_reproduce_the_uninterrupted_run_bit_for_bit_body);
}

fn cancel_and_resume_reproduce_the_uninterrupted_run_bit_for_bit_body() {
    cancel_and_resume_under(false);
}

/// sc-25690: the same lifecycle with the generator step's trunk / pixel blocks
/// activation-checkpointed — cancel → durable checkpoint → resume reproduces the uninterrupted
/// checkpointed run bit for bit (generator, EMA, both optimizers, critic head, data position,
/// export).
#[test]
fn cancel_and_resume_reproduce_the_uninterrupted_run_under_activation_checkpointing() {
    crate::common::cpu_process("restoration_train_lifecycle::cancel_and_resume_reproduce_the_uninterrupted_run_under_activation_checkpointing", || cancel_and_resume_under(true));
}

fn cancel_and_resume_under(ac: bool) {
    use mlx_gen_iris::train::model::checkpointed_regions;
    let entered = checkpointed_regions();
    let tmp = tempfile::tempdir().unwrap();
    let items = clean_items(tmp.path(), 3);
    let opts = json!({"degradation": mini_degradation(), "ema_decay": 0.9});
    // uninterrupted: 4 steps (16 micro-steps with accumulation 2)
    let mut cfg = config(16, opts.clone());
    cfg.gradient_checkpointing = ac;
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
    assert_eq!(
        checkpointed_regions() > entered,
        ac,
        "regions are entered exactly when the run is checkpointed"
    );
}

/// sc-25690: the restoration trainer's activation checkpointing (`gradient_checkpointing`) — the
/// generation trainer's trunk / pixel regions (MLX's checkpoint transform) around the generator in
/// the generator step, the L2 / LPIPS / critic terms downstream of them. f32: the generator
/// gradients are the plain graph's; bf16 generator (f32 critic, as the miniature runs): the same
/// loss terms bit for bit and gradients within the declared bf16 bound.
#[test]
fn activation_checkpointing_gradients_equal_the_plain_graph() {
    crate::common::cpu_process(
        "restoration_train_lifecycle::activation_checkpointing_gradients_equal_the_plain_graph",
        activation_checkpointing_gradients_equal_the_plain_graph_body,
    );
}

fn activation_checkpointing_gradients_equal_the_plain_graph_body() {
    use mlx_gen_iris::train::model::checkpointed_regions;
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
        let batch = PairBatch::from_pairs(&[pair]);
        let (model, params, frozen, head, sn) =
            mlx_gen_iris::train::restore::miniature_parts(&res, &plan, Box::new(nets)).unwrap();
        let before = checkpointed_regions();
        let (terms, grads) =
            generator_grads(&model, &params, &frozen, &plan, &head, &sn, &batch, 1.0).unwrap();
        let entered = checkpointed_regions() - before;
        let regions = model.cfg.depth + model.cfg.pixel.depth;
        assert_eq!(
            entered,
            if ac { regions } else { 0 },
            "full checkpoints every trunk and pixel block, none no region"
        );
        assert_eq!(
            grads.len(),
            params.len(),
            "every trained tensor has a gradient"
        );
        (terms, grads)
    };
    let (f_terms, f_none) = case("f32", false);
    let (f_terms_ac, f_full) = case("f32", true);
    assert_eq!(
        f_terms.map(f64::to_bits),
        f_terms_ac.map(f64::to_bits),
        "f32 loss terms"
    );
    // The generation trainer's 1e-6 holds here too — the Candle twin needs 1e-5 for this task (its
    // replay reaches the O(10²) adversarial term's gradient in another summation order). MLX's
    // checkpoint transform re-runs the block's own graph: measured 0 (bit-equal) on the CPU stream.
    crate::common::assert_ac_f32_equal("restoration", 1e-6, &f_none, &f_full);
    let (b_terms, b_none) = case("bf16", false);
    let (b_terms_ac, b_full) = case("bf16", true);
    assert_eq!(
        b_terms.map(f64::to_bits),
        b_terms_ac.map(f64::to_bits),
        "bf16 loss terms"
    );
    crate::common::assert_ac_bf16_noise_only("restoration", &f_none, &b_none, &b_full);
}

/// sc-25690: a checkpointed restoration run is the plain run (f32), for the full fine-tune and for
/// an adapter (a region then rebuilds its block from the frozen base plus the traced factors): the
/// trainer wires `gradient_checkpointing` into its generator step, the export moves by rounding
/// only, keeps its stamps and loads in the `iris_3b_restore` transform.
#[test]
fn activation_checkpointing_leaves_the_run_unchanged() {
    crate::common::cpu_process(
        "restoration_train_lifecycle::activation_checkpointing_leaves_the_run_unchanged",
        activation_checkpointing_leaves_the_run_unchanged_body,
    );
}

fn activation_checkpointing_leaves_the_run_unchanged_body() {
    use mlx_gen_iris::train::model::checkpointed_regions;
    let data = tempfile::tempdir().unwrap();
    let items = clean_items(data.path(), 1);
    for lora in [false, true] {
        let run = |ac: bool| {
            let out = tempfile::tempdir().unwrap();
            let mut cfg = config(4, json!({"degradation": mini_degradation()}));
            cfg.save_every = 0;
            cfg.gradient_checkpointing = ac;
            if lora {
                cfg.full_finetune = false;
                cfg.network_type = NetworkType::Lora;
                cfg.rank = 2;
                cfg.alpha = 2.0;
            }
            let before = checkpointed_regions();
            let res = trainer()
                .train(&request(&items, cfg, out.path()), &mut |_| {})
                .unwrap();
            assert_eq!(res.steps, 4);
            assert_eq!(checkpointed_regions() > before, ac, "lora={lora}");
            let w = Weights::from_file(&res.adapter_path).unwrap();
            assert_eq!(w.metadata("irisTask"), Some("restoration"));
            assert_eq!(w.metadata("irisArtifact"), Some("full_model"));
            let mut spec = LoadSpec::new(WeightsSource::Dir(
                res.adapter_path.parent().unwrap().to_path_buf(),
            ));
            spec.precision = Precision::Fp32;
            IrisRestorer::load(&spec).unwrap();
            tensors(&res.adapter_path)
        };
        let (plain, checkpointed) = (run(false), run(true));
        assert_eq!(plain.len(), checkpointed.len());
        let base = tensors(&export_dir().join("model.safetensors"));
        assert!(
            base.iter().any(|(k, v)| checkpointed[k] != *v),
            "lora={lora}: the checkpointed run trained"
        );
        let mut worst = 0f32;
        for (k, v) in &plain {
            for (x, y) in v.iter().zip(&checkpointed[k]) {
                worst = worst.max((x - y).abs());
            }
        }
        eprintln!(
            "restoration (lora={lora}): activation checkpointing vs plain: max |Δ| = {worst:.3e}"
        );
        assert!(worst <= 1e-6, "lora={lora}: {worst}");
    }
}

#[test]
fn the_export_is_a_restoration_model_the_transform_runs_and_other_tasks_refuse() {
    crate::common::cpu_process("restoration_train_lifecycle::the_export_is_a_restoration_model_the_transform_runs_and_other_tasks_refuse", the_export_is_a_restoration_model_the_transform_runs_and_other_tasks_refuse_body);
}

fn the_export_is_a_restoration_model_the_transform_runs_and_other_tasks_refuse_body() {
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
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("irisTask"), Some("restoration"));
    assert_eq!(w.metadata("irisArtifact"), Some("full_model"));
    assert_eq!(w.metadata("irisWeights"), Some("ema"));
    // the merged LoRA changed the base
    let base = tensors(&export_dir().join("model.safetensors"));
    let merged = tensors(&out.adapter_path);
    assert_eq!(base.len(), merged.len());
    assert!(base.iter().any(|(k, v)| merged[k] != *v));
    // loads in iris_3b_restore with no training-only dependency and restores
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    let restorer = IrisRestorer::load(&spec).unwrap();
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
        Err(mlx_gen::gen_core::Error::Unsupported(_))
    ));
    let mut gen = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    gen.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(crate::common::tiny_text_encoder()),
    );
    assert!(matches!(
        GenerationResources::from_spec(&gen, "iris_3b"),
        Err(mlx_gen::gen_core::Error::Unsupported(_))
    ));
    // sc-25690: activation checkpointing (`gradient_checkpointing`) is a trained option of this
    // task on both backends, for the full fine-tune and the adapters.
    let mut checkpointed = config(2, json!({"degradation": mini_degradation()}));
    checkpointed.gradient_checkpointing = true;
    trainer()
        .validate(&request(&items, checkpointed.clone(), tmp.path()))
        .unwrap();
    checkpointed.full_finetune = false;
    checkpointed.network_type = NetworkType::Lora;
    checkpointed.rank = 2;
    checkpointed.alpha = 2.0;
    trainer()
        .validate(&request(&items, checkpointed, tmp.path()))
        .unwrap();
}
