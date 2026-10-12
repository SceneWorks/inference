//! sc-25690 (epic 25678 terminal measurement) — the Candle twin of the MLX probe: ONE bounded
//! real-weight training run of an Iris task at a chosen square size and activation-checkpointing
//! policy on the build's device (`cuda:0` under `--features cuda`; select the card with
//! `CUDA_VISIBLE_DEVICES`). Report-only: the probe asserts that its instruments are trustworthy and
//! that the run trains finitely and exports; nothing is gated on a number.
//!
//! Three quantities are printed, all in decimal GB (1e9 bytes), for the whole run (context + load +
//! train + export):
//! * `used_high_gb` — the driver pool's `USED_MEM_HIGH`: the continuous high-water of concurrently
//!   LIVE bytes. This is what the shape needs.
//! * `reserved_high_gb` — the pool's `RESERVED_MEM_HIGH`: live + cached-free bytes the pool held from
//!   the driver, the unit `nvidia-smi` and an admission gate read.
//! * `device_overall_peak_gb` — `VramProbe`'s polled device-level `nvidia-smi memory.used` over the
//!   pre-load baseline. It misses sub-poll transients and saturates at the card's size, so it is the
//!   cross-check, not the answer (`candle_gen::cuda_mempool` module docs).
//!
//! The counters are printed when the run fails too (an out-of-memory row then carries how far the
//! pool got), and the runner classifies the row (`scripts/ci/iris_memory_probe_row.py`).
//!
//! Inputs (env) — all REQUIRED, so a misspelled variable cannot silently measure another shape:
//! * `IRIS_PROBE_TASK` — `generation` | `depth` | `restoration`.
//! * `IRIS_PROBE_SIZE` — the square training size (px), e.g. 1024.
//! * `IRIS_PROBE_AC` — `none` | `full` (generation only: the depth and restoration trainers refuse
//!   activation checkpointing).
//! * `IRIS_PROBE_STEPS` — optimizer steps.
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
use candle_gen::testkit::VramProbe;

/// The selected card must be idle before the load: a resident tenant contaminates a device-level
/// peak. The CUDA box's free card idles at 0.2-0.9 GB (WDDM desktop residency).
const MAX_IDLE_BASELINE_GB: f64 = 2.0;

/// Trap 2 of `candle_gen::cuda_mempool`: a high-water mark zeroed while the pool still holds pages
/// snaps straight back to them. The pool of a fresh process must hold (next to) nothing.
#[cfg(feature = "cuda")]
const MAX_QUIESCED_POOL_BYTES: u64 = 64 << 20;

fn env_req(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(env_req(name))
}

/// The device's default stream-ordered pool — the one candle allocates every tensor from — with its
/// high-water marks zeroed on a quiesced pool.
#[cfg(feature = "cuda")]
struct PoolMeter {
    pool: candle_gen::cuda_mempool::MemPool,
    // Keeps the device's primary context retained (and current on this thread) for the reads.
    _device: candle_gen::candle_core::Device,
}

#[cfg(feature = "cuda")]
impl PoolMeter {
    fn start() -> Self {
        let device = candle_gen::default_device().expect("the CUDA device");
        let pool = candle_gen::cuda_mempool::MemPool::device_default(0)
            .expect("the CUDA default memory pool");
        let used = pool.used().expect("USED_MEM_CURRENT");
        let reserved = pool.reserved().expect("RESERVED_MEM_CURRENT");
        eprintln!(
            "[[IRIS_MEMORY_PROBE]] cuda_pool_at_reset used_bytes={used} reserved_bytes={reserved}"
        );
        assert!(
            reserved <= MAX_QUIESCED_POOL_BYTES,
            "the pool already holds {reserved} bytes: its high-water marks cannot be reset"
        );
        assert!(pool.reset_high_water(), "reset the pool high-water marks");
        Self {
            pool,
            _device: device,
        }
    }

    /// `(USED_MEM_HIGH, RESERVED_MEM_HIGH)` in bytes.
    fn highs(&self) -> (u64, u64) {
        (
            self.pool.used_high().expect("USED_MEM_HIGH"),
            self.pool.reserved_high().expect("RESERVED_MEM_HIGH"),
        )
    }

    /// A completed run allocated through this pool, and a pool never reserves less than is live:
    /// anything else means the counters read some other pool (trap 1) or were never reset.
    fn assert_measured(&self) {
        let (live, reserved) = self.highs();
        assert!(
            live > 0 && reserved >= live,
            "the pool counters are not a measurement: USED_MEM_HIGH {live}, RESERVED_MEM_HIGH {reserved}"
        );
    }
}

#[cfg(not(feature = "cuda"))]
struct PoolMeter;

#[cfg(not(feature = "cuda"))]
impl PoolMeter {
    fn start() -> Self {
        panic!("the Candle memory probe measures a CUDA device: build with --features cuda")
    }

    fn highs(&self) -> (u64, u64) {
        panic!("the Candle memory probe measures a CUDA device: build with --features cuda")
    }

    fn assert_measured(&self) {
        panic!("the Candle memory probe measures a CUDA device: build with --features cuda")
    }
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
    let task = env_req("IRIS_PROBE_TASK");
    let size: u32 = env_req("IRIS_PROBE_SIZE")
        .parse()
        .expect("IRIS_PROBE_SIZE: a pixel count");
    let ac = env_req("IRIS_PROBE_AC");
    assert!(
        matches!(ac.as_str(), "none" | "full"),
        "IRIS_PROBE_AC {ac:?}: none | full"
    );
    let steps: u32 = env_req("IRIS_PROBE_STEPS")
        .parse()
        .expect("IRIS_PROBE_STEPS: a step count");
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
    // The device baseline is read BEFORE the context exists (a failed `nvidia-smi` query panics), so
    // the polled peak is the whole run's device delta: context + load + train + export.
    let mut probe = VramProbe::start_rendered().assert_idle(MAX_IDLE_BASELINE_GB);
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] cuda_baseline baseline_gb={:.2}",
        probe.report().baseline_gb
    );
    let phase = probe.phase();
    let pool = PoolMeter::start();
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
    // The counters are printed whether or not the run survives: an out-of-memory row reports how far
    // the pool got instead of nothing.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.train(&req, &mut |p| {
            if let TrainingProgress::Training { step, total, loss } = p {
                eprintln!(
                    "[[IRIS_MEMORY_PROBE]] step {step}/{total} loss {loss:.4} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                assert!(loss.is_finite(), "non-finite loss at step {step}");
                last = std::time::Instant::now();
            }
        })
    }));
    if let Ok(Err(error)) = &outcome {
        eprintln!("[[IRIS_MEMORY_PROBE]] train failed: {error:?}");
    }
    drop(trainer);
    // One phase over the whole run, so only the overall peak means anything.
    probe.end_gen(phase);
    let vram = probe.report();
    let (used_high, reserved_high) = pool.highs();
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] cuda_memory used_high_gb={:.2} reserved_high_gb={:.2} \
         device_overall_peak_gb={:.2} baseline_gb={:.2}",
        used_high as f64 / 1e9,
        reserved_high as f64 / 1e9,
        vram.peak_gb,
        vram.baseline_gb
    );
    let res = match outcome {
        Ok(res) => res.unwrap(),
        Err(panic) => std::panic::resume_unwind(panic),
    };
    vram.assert_trustworthy(MAX_IDLE_BASELINE_GB);
    pool.assert_measured();
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] done: {} steps, final loss {:.4}, total {:.1}s → {}",
        res.steps,
        res.final_loss,
        start.elapsed().as_secs_f32(),
        res.adapter_path.display()
    );
    assert!(res.final_loss.is_finite());
}
