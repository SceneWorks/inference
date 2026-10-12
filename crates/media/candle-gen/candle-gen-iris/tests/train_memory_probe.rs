//! sc-25690 (epic 25678 terminal measurement) — the Candle twin of the MLX probe: ONE bounded
//! real-weight training run of an Iris task at a chosen square size and activation-checkpointing
//! policy on the build's device (`cuda:0` under `--features cuda`; select the card with
//! `CUDA_VISIBLE_DEVICES`), its device VRAM sampled from BEFORE the load by `VramProbe` (report-only) (the probe itself asserts only that the run trains finitely and exports). Nothing is
//! gated on the number; the runner records the guard's peak as the measured floor of that shape.
//!
//! Inputs (env):
//! * `IRIS_PROBE_TASK` — `generation` | `depth` | `restoration`.
//! * `IRIS_PROBE_SIZE` — the square training size (px), e.g. 1024.
//! * `IRIS_PROBE_AC` — `none` (default) | `full` (generation only: the depth and restoration
//!   trainers refuse activation checkpointing).
//! * `IRIS_PROBE_STEPS` — optimizer steps (default 2).
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (generation backbone + `depth/`);
//!   `IRIS_TEXT_ENCODER_DIR` (generation); `IRIS_UPSCALER_DIR`, `IRIS_LPIPS_VGG16_DIR`,
//!   `IRIS_LPIPS_LIN_DIR`, `IRIS_CRITIC_TRUNK_DIR` (restoration).
//!
//! The shape is the product's FULL fine-tune at batch 1: generation with SceneWorks' Iris target
//! defaults (AdamW, bf16 autocast over f32 masters, EMA on, cached text conditioning), depth's
//! published direct-regression recipe, restoration's paper recipe (`recipe_training_config`:
//! generator + critic AdamW, LPIPS, EMA).

use std::path::{Path, PathBuf};

use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use candle_gen::gen_core::{
    CancelFlag, LoadSpec, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

/// A deterministic, non-constant RGB image of `size`² (a soft two-axis gradient with a checker).
fn write_rgb(path: &Path, size: u32, seed: u32) {
    image::RgbImage::from_fn(size, size, |x, y| {
        let c = ((x / 32 + y / 32 + seed) % 2) as u8 * 40;
        image::Rgb([
            ((x * 255) / size) as u8 / 2 + c,
            ((y * 255) / size) as u8 / 2 + c,
            (((x + y) * 255) / (2 * size)) as u8 / 2 + 60,
        ])
    })
    .save(path)
    .unwrap();
}

fn write_npy(path: &Path, h: u32, w: u32, f: impl Fn(u32, u32) -> f32) {
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

#[test]
#[ignore = "needs the real Iris weights for IRIS_PROBE_TASK and a CUDA device"]
fn real_training_memory_probe() {
    let task = env_or("IRIS_PROBE_TASK", "generation");
    let size: u32 = env_or("IRIS_PROBE_SIZE", "256").parse().unwrap();
    let ac = env_or("IRIS_PROBE_AC", "none");
    let steps: u32 = env_or("IRIS_PROBE_STEPS", "2").parse().unwrap();
    let work = tempfile::tempdir().unwrap();
    let data = work.path().join("data");
    std::fs::create_dir_all(&data).unwrap();

    let (engine, spec, items, config) = match task.as_str() {
        "generation" => {
            use candle_gen::gen_core::iris::train::OPTIONS_KEY;
            let items = (0..2)
                .map(|i| {
                    let p = data.join(format!("g{i}.png"));
                    write_rgb(&p, size, i);
                    TrainingItem::captioned(
                        p,
                        format!("a soft gradient with a checker pattern {i}"),
                    )
                })
                .collect::<Vec<_>>();
            let mut config = TrainingConfig {
                learning_rate: 1e-5,
                steps,
                batch_size: 1,
                resolution: size,
                save_every: 0,
                seed: 3,
                full_finetune: true,
                train_dtype: "bf16".into(),
                optimizer: "adamw".into(),
                ..Default::default()
            };
            config.model_options.insert(
                OPTIONS_KEY.into(),
                serde_json::json!({
                    "init": "weights", "text_conditioning": "cached", "ema_enabled": true,
                    "ema_decay": 0.9999, "gradient_clip": 0.5, "betas": [0.9, 0.95],
                    "activation_checkpointing": ac,
                }),
            );
            let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
            spec.components.insert(
                TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
            );
            ("iris_3b", spec, items, config)
        }
        "depth" => {
            use candle_gen::gen_core::iris::train::depth::ITEM_DEPTH_KEY;
            use candle_gen::gen_core::iris::train::OPTIONS_KEY;
            assert_eq!(
                ac, "none",
                "the depth trainer refuses activation checkpointing"
            );
            let rgb = data.join("d.png");
            write_rgb(&rgb, size, 0);
            let depth = data.join("d.npy");
            write_npy(&depth, size, size, |y, _| {
                1.0 + 9.0 * (1.0 - y as f32 / (size - 1) as f32)
            });
            let mut item = TrainingItem::captioned(rgb, String::new());
            item.model_options.insert(
                ITEM_DEPTH_KEY.into(),
                serde_json::json!({"path": depth.to_str().unwrap()}),
            );
            let mut config = TrainingConfig {
                learning_rate: 3e-5,
                weight_decay: 0.01,
                steps,
                batch_size: 1,
                resolution: size,
                save_every: 0,
                seed: 3,
                full_finetune: true,
                loss_type: "l1".into(),
                train_dtype: "bf16".into(),
                ..Default::default()
            };
            config
                .model_options
                .insert(OPTIONS_KEY.into(), serde_json::json!({"hflip_prob": 0.0}));
            let spec = LoadSpec::new(WeightsSource::Dir(
                env_dir("IRIS_WEIGHTS_DIR").join("depth"),
            ));
            ("iris_3b_depth", spec, vec![item], config)
        }
        "restoration" => {
            use candle_gen::gen_core::iris::restoration_train::{
                recipe_training_config, CRITIC_TRUNK_COMPONENT, LPIPS_LIN_COMPONENT, OPTIONS_KEY,
                VGG16_COMPONENT,
            };
            assert_eq!(
                ac, "none",
                "the restoration trainer refuses activation checkpointing"
            );
            let items = (0..2)
                .map(|i| {
                    let p = data.join(format!("r{i}.png"));
                    // Larger than the crop so the trainer's random crop has room.
                    write_rgb(&p, size + 64, i);
                    TrainingItem::captioned(p, String::new())
                })
                .collect::<Vec<_>>();
            let mut config = recipe_training_config();
            assert!(config.full_finetune);
            config.steps = steps * 2; // generator + critic micro-steps per optimizer step
            config.batch_size = 1;
            config.resolution = size;
            config.save_every = 0;
            config.seed = 5;
            config.train_dtype = "bf16".into();
            config
                .model_options
                .insert(OPTIONS_KEY.into(), serde_json::json!({"ema_decay": 0.999}));
            let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_UPSCALER_DIR")));
            for (component, var) in [
                (VGG16_COMPONENT, "IRIS_LPIPS_VGG16_DIR"),
                (LPIPS_LIN_COMPONENT, "IRIS_LPIPS_LIN_DIR"),
                (CRITIC_TRUNK_COMPONENT, "IRIS_CRITIC_TRUNK_DIR"),
            ] {
                spec.components
                    .insert(component.into(), WeightsSource::Dir(env_dir(var)));
            }
            ("iris_3b_restore", spec, items, config)
        }
        other => panic!("IRIS_PROBE_TASK {other:?}: generation | depth | restoration"),
    };

    eprintln!(
        "[[IRIS_MEMORY_PROBE]] backend=candle task={task} engine={engine} size={size} ac={ac} \
         steps={} batch=1 full_finetune=true optimizer={} dtype={}",
        config.steps, config.optimizer, config.train_dtype
    );
    // Sampled from before the load, so the reported peak is the whole run's device delta.
    let probe_vram = used_mib(probe_gpu()).is_some();
    let mut probe = probe_vram.then(VramProbe::start_rendered);
    let phase = probe.as_ref().map(VramProbe::phase);
    let start = std::time::Instant::now();
    let mut trainer = candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(engine, &spec)
        .unwrap();
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] loaded in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    let req = TrainingRequest {
        items,
        config,
        output_dir: work.path().join("train"),
        file_name: "probe.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    };
    let mut last = std::time::Instant::now();
    let res = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, total, loss } = p {
                eprintln!(
                    "[[IRIS_MEMORY_PROBE]] step {step}/{total} loss {loss:.4} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                assert!(loss.is_finite(), "non-finite loss at step {step}");
                last = std::time::Instant::now();
            }
        })
        .unwrap();
    drop(trainer);
    if let (Some(p), Some(phase)) = (probe.as_mut(), phase) {
        p.end_load(phase);
    }
    match &probe {
        Some(p) => eprintln!("[[IRIS_MEMORY_PROBE]] vram {}", p.report()),
        None => eprintln!("[[IRIS_MEMORY_PROBE]] vram: nvidia-smi unavailable, not measured"),
    }
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] done: {} steps, final loss {:.4}, total {:.1}s → {}",
        res.steps,
        res.final_loss,
        start.elapsed().as_secs_f32(),
        res.adapter_path.display()
    );
    assert!(res.final_loss.is_finite());
}
