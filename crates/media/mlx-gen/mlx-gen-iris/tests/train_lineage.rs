//! sc-25690 — the generation trainer's checkpoint lineage and data cursor on the miniature
//! snapshot (the Candle twin's `train_lineage`, case for case), each case in a CPU child process
//! (`cpu_process`):
//!
//! * a cancel observed at the first batch of a new epoch, in a run that itself resumed mid-epoch,
//!   saves the data position of the pass its last step belongs to: resuming reproduces the
//!   uninterrupted run bit for bit;
//! * a fresh run into an output directory holding another run's checkpoints starts its own
//!   lineage (the old checkpoints move aside, never deleted): retention keeps the new run's, and
//!   a later resume continues the new run, not the old one;
//! * a full-model export replaces its directory whole (config + weights published together).

use std::cell::Cell;
use std::collections::HashMap;
use std::path::Path;

use mlx_gen::gen_core::iris::train::lineage::prior_checkpoint_root;
use mlx_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, list_checkpoints, read_checkpoint_state, OPTIONS_KEY,
};
use mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use mlx_gen::gen_core::{
    LrSchedule, Trainer, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
};
use mlx_gen::{LoadSpec, WeightsSource};
use mlx_rs::{Array, Dtype};
use serde_json::json;

use crate::common::{cpu_process, tiny_backbone, tiny_text_encoder};

fn trainer() -> Box<dyn Trainer> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    mlx_gen_iris::provider_registry()
        .unwrap()
        .load_trainer("iris_3b", &spec)
        .expect("the tiny snapshot loads as a trainer")
}

fn dataset(dir: &Path) -> Vec<TrainingItem> {
    let captions = ["a red fox in the snow", "golden hour", "a fox , snow"];
    captions
        .iter()
        .enumerate()
        .map(|(i, caption)| {
            let img = image::RgbImage::from_fn(16 + 4 * i as u32, 16, |x, y| {
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

/// LoRA + AdamW, one item per step: 3 steps per epoch over the 3 items.
fn config(steps: u32) -> TrainingConfig {
    let mut cfg = TrainingConfig {
        rank: 2,
        alpha: 4.0,
        learning_rate: 1e-3,
        steps,
        batch_size: 1,
        resolution: 16,
        save_every: 0,
        seed: 9,
        train_dtype: "f32".into(),
        lr_scheduler: LrSchedule::Cosine,
        lr_warmup_steps: 1,
        ..Default::default()
    };
    cfg.model_options.insert(
        OPTIONS_KEY.into(),
        json!({"text_dropout": 0.5, "ema_decay": 0.9}),
    );
    cfg
}

/// Train `cfg` in `out`, cancelling right after the `Training` event of step `cancel_at`; returns
/// the steps the output reports and how many `Training` events this invocation emitted.
fn run(
    items: &[TrainingItem],
    cfg: TrainingConfig,
    out: &Path,
    name: &str,
    cancel_at: Option<u32>,
) -> (u32, u32) {
    let req = TrainingRequest {
        items: items.to_vec(),
        config: cfg,
        output_dir: out.to_path_buf(),
        file_name: name.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let flag = req.cancel.clone();
    let events = Cell::new(0u32);
    let res = trainer()
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, .. } = p {
                events.set(events.get() + 1);
                if Some(step) == cancel_at {
                    flag.cancel();
                }
            }
        })
        .unwrap();
    (res.steps, events.get())
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

fn steps_in(root: &Path) -> Vec<u64> {
    list_checkpoints(root).into_iter().map(|(s, _)| s).collect()
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_cancel_at_a_new_epochs_first_batch_after_a_resume_resumes_exactly() {
    cpu_process(
        "train_lineage::a_cancel_at_a_new_epochs_first_batch_after_a_resume_resumes_exactly",
        || {
            let data = tempfile::tempdir().unwrap();
            let items = dataset(data.path());
            let name = "cursor.safetensors";
            let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            assert_eq!(run(&items, config(6), a.path(), name, None).0, 6);
            // Cancelled mid-epoch 1 (after step 1): the resumed pass skips one batch.
            assert_eq!(run(&items, config(6), b.path(), name, Some(1)).0, 1);
            let mut resumed = config(6);
            resumed.resume = true;
            // The resumed pass walks steps 2 and 3 (positions 1, 2); the cancel is observed at
            // the first batch of epoch 2, after `begin_epoch` reset the walker.
            assert_eq!(
                run(&items, resumed.clone(), b.path(), name, Some(3)),
                (3, 2)
            );
            let root_b = checkpoint_root(b.path(), name);
            let s = read_checkpoint_state(&latest_checkpoint(&root_b).unwrap()).unwrap();
            assert_eq!((s.step, s.epoch, s.batches_consumed), (3, 1, 3));
            assert_eq!(run(&items, resumed, b.path(), name, None), (6, 3));
            let (la, lb) = (
                latest_checkpoint(&checkpoint_root(a.path(), name)).unwrap(),
                latest_checkpoint(&root_b).unwrap(),
            );
            for f in [
                "trainable.safetensors",
                "ema.safetensors",
                "optimizer.safetensors",
            ] {
                assert_eq!(
                    tensors(&la.join(f)),
                    tensors(&lb.join(f)),
                    "{f}: the twice-resumed run equals the uninterrupted one"
                );
            }
            assert_eq!(
                read_checkpoint_state(&la).unwrap(),
                read_checkpoint_state(&lb).unwrap()
            );
            assert_eq!(tensors(&a.path().join(name)), tensors(&b.path().join(name)));
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED train_lineage::a_cancel_at_a_new_epochs_first_batch_after_a_resume_resumes_exactly");
        },
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_fresh_run_starts_its_own_lineage_beside_another_runs_checkpoints() {
    cpu_process(
        "train_lineage::a_fresh_run_starts_its_own_lineage_beside_another_runs_checkpoints",
        || {
            let data = tempfile::tempdir().unwrap();
            let items = dataset(data.path());
            let out = tempfile::tempdir().unwrap();
            let name = "lineage.safetensors";
            let keep = |steps: u32| {
                let mut cfg = config(steps);
                cfg.save_every = 1;
                cfg.model_options
                    .insert(OPTIONS_KEY.into(), json!({"keep_last_checkpoints": 2}));
                cfg
            };
            let root = checkpoint_root(out.path(), name);
            run(&items, keep(4), out.path(), name, None);
            assert_eq!(steps_in(&root), [3, 4]);
            // A fresh (non-resume) run of 2 steps into the same output directory.
            run(&items, keep(2), out.path(), name, None);
            assert_eq!(
                steps_in(&root),
                [1, 2],
                "retention keeps the new run's checkpoints"
            );
            let prior = prior_checkpoint_root(out.path(), name);
            assert_eq!(
                steps_in(&prior),
                [3, 4],
                "the old run's checkpoints are kept aside"
            );
            // Resuming continues the new run (2 more steps), not the old run's step 4.
            let mut resume = keep(4);
            resume.resume = true;
            assert_eq!(run(&items, resume, out.path(), name, None), (4, 2));
            assert_eq!(steps_in(&root), [3, 4]);
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED train_lineage::a_fresh_run_starts_its_own_lineage_beside_another_runs_checkpoints");
        },
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_full_model_export_replaces_its_directory_whole() {
    cpu_process(
        "train_lineage::a_full_model_export_replaces_its_directory_whole",
        || {
            let data = tempfile::tempdir().unwrap();
            let items = dataset(data.path());
            let out = tempfile::tempdir().unwrap();
            let mut cfg = config(1);
            cfg.full_finetune = true;
            cfg.optimizer = "adamw".into();
            let dir = out.path().join("full");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("stale.bin"), b"an older export's leftover").unwrap();
            run(&items, cfg, out.path(), "full.safetensors", None);
            let mut files: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            files.sort();
            assert_eq!(files, ["config.yaml", "model.safetensors"]);
            let staged: Vec<String> = std::fs::read_dir(out.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| n.contains(".export-"))
                .collect();
            assert!(
                staged.is_empty(),
                "no staging directory is left: {staged:?}"
            );
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED train_lineage::a_full_model_export_replaces_its_directory_whole");
        },
    );
}

/// sc-25690: a resume source refused at validation leaves the output's checkpoint root as it was
/// (retirement runs only after the source was validated and loaded).
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_refused_resume_leaves_the_checkpoint_root_untouched() {
    cpu_process(
        "train_lineage::a_refused_resume_leaves_the_checkpoint_root_untouched",
        || {
            let data = tempfile::tempdir().unwrap();
            let items = dataset(data.path());
            let (other, out) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            let name = "refused.safetensors";
            let mut cfg = config(2);
            cfg.save_every = 1;
            run(&items, cfg.clone(), other.path(), name, None);
            run(&items, cfg.clone(), out.path(), name, None);
            let root = checkpoint_root(out.path(), name);
            assert_eq!(steps_in(&root), [1, 2]);
            // An external resume source of another adapter shape: refused at validation.
            let mut wrong = cfg.clone();
            wrong.rank = 4;
            wrong.model_options.insert(
                OPTIONS_KEY.into(),
                json!({"resume_from": checkpoint_root(other.path(), name)}),
            );
            let req = TrainingRequest {
                items: items.clone(),
                config: wrong,
                output_dir: out.path().to_path_buf(),
                file_name: name.into(),
                trigger_words: Vec::new(),
                cancel: Default::default(),
            };
            let err = trainer().train(&req, &mut |_| {}).unwrap_err();
            assert!(err.to_string().contains("different state"), "{err}");
            assert_eq!(
                steps_in(&root),
                [1, 2],
                "the refused run moved nothing aside"
            );
            assert!(!prior_checkpoint_root(out.path(), name).exists());
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED train_lineage::a_refused_resume_leaves_the_checkpoint_root_untouched");
        },
    );
}
