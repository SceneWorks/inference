//! sc-25690 (epic 25678 terminal measurement) — ONE bounded real-weight training run of an Iris
//! task at a chosen square size and activation-checkpointing policy. Report-only: the probe asserts
//! that its counters are a measurement and that the run trains finitely and exports; nothing is
//! gated on a number.
//!
//! Two different quantities come out of a run, both in decimal GB (1e9 bytes):
//! * `active_peak_gb` — MLX's own `get_peak_memory`, reset before the load: the high-water of bytes
//!   held by LIVE arrays. This is what the shape needs, and it does not depend on the buffer cache.
//! * the process `phys_footprint` the external guard polls
//!   (`scripts/ci/iris_restore_footprint_guard.py`) — active bytes PLUS MLX's buffer cache
//!   (freed-but-retained Metal buffers) plus the process's non-MLX memory. The trainer does not clear
//!   the cache per step, so under the default cache limit the footprint climbs to MLX's allocator GC
//!   limit (0.95 x the device's recommended working set) and sits there: an allocator ceiling, not
//!   the shape's peak. `IRIS_PROBE_MLX_CACHE=none` sets the cache limit to 0 from this test (the
//!   trainer is untouched), which makes the footprint track the active bytes instead.
//!
//! The limits are printed before the load and the counters at every step (a guard stop kills the
//! process, so the last step line is then the lower bound) and at the end, whether or not the run
//! survives. The runner classifies the row (`scripts/ci/iris_memory_probe_row.py`).
//!
//! Inputs (env) — all REQUIRED, so a misspelled variable cannot silently measure another shape:
//! * `IRIS_PROBE_TASK` — `generation` | `depth` | `restoration`.
//! * `IRIS_PROBE_SIZE` — the square training size (px), e.g. 1024.
//! * `IRIS_PROBE_AC` — `none` | `full` (generation only: the depth and restoration trainers refuse
//!   activation checkpointing).
//! * `IRIS_PROBE_STEPS` — optimizer steps.
//! * `IRIS_PROBE_MLX_CACHE` — `default` (MLX's own cache limit, as the product runs) | `none` (cache
//!   limit 0).
//! * `IRIS_PROBE_MLX_LIMIT_GB` — `default` (this machine's own allocator limits) | a decimal-GB
//!   figure that becomes BOTH MLX's memory limit and its cache limit, to emulate a smaller Mac's
//!   allocator. MLX derives `gc_limit = min(memory_limit, 0.95 x this device's recommended working
//!   set)` inside `set_memory_limit` and offers no way to set it alone, so the figure to pass is the
//!   smaller machine's GC limit; its (higher) memory limit is then emulated conservatively low. This
//!   emulates the allocator's limits only — not a smaller machine's real memory pressure. Needs
//!   `IRIS_PROBE_MLX_CACHE=default`. The wired limit is left at MLX's default (0) as on any Mac.
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (generation backbone + `depth/`);
//!   `IRIS_TEXT_ENCODER_DIR` (generation); `IRIS_UPSCALER_DIR`, `IRIS_LPIPS_VGG16_DIR`,
//!   `IRIS_LPIPS_LIN_DIR`, `IRIS_CRITIC_TRUNK_DIR` (restoration).
//!
//! The shape is the product's FULL fine-tune at batch 1: generation with SceneWorks' Iris target
//! defaults (AdamW, bf16 autocast over f32 masters, EMA on, cached text conditioning), depth's
//! published direct-regression recipe, restoration's paper recipe (`recipe_training_config`:
//! generator + critic AdamW, LPIPS, EMA).

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use mlx_gen::gen_core::{
    CancelFlag, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
};
use mlx_gen::{LoadSpec, WeightsSource};

use mlx_rs::memory::{
    get_active_memory, get_cache_memory, get_memory_limit, get_peak_memory, reset_peak_memory,
    set_cache_limit, set_memory_limit,
};

fn env_req(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(env_req(name))
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
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
#[ignore = "needs the real Iris weights for IRIS_PROBE_TASK and the GPU; run under an external memory guard"]
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
    let cache = env_req("IRIS_PROBE_MLX_CACHE");
    assert!(
        matches!(cache.as_str(), "default" | "none"),
        "IRIS_PROBE_MLX_CACHE {cache:?}: default | none"
    );
    let limit = env_req("IRIS_PROBE_MLX_LIMIT_GB");
    let emulated_limit: Option<usize> = (limit != "default").then(|| {
        let limit_gb: f64 = limit
            .parse()
            .expect("IRIS_PROBE_MLX_LIMIT_GB: default | a decimal-GB figure");
        assert!(
            (1.0..=1024.0).contains(&limit_gb),
            "IRIS_PROBE_MLX_LIMIT_GB {limit_gb}: 1..=1024 GB"
        );
        assert_eq!(
            cache, "default",
            "an emulated allocator limit needs IRIS_PROBE_MLX_CACHE=default"
        );
        (limit_gb * 1e9) as usize
    });
    let work = tempfile::tempdir().unwrap();
    let data = work.path().join("data");
    std::fs::create_dir_all(&data).unwrap();

    let (engine, spec, items, config) = match task.as_str() {
        "generation" => {
            use mlx_gen::gen_core::iris::train::OPTIONS_KEY;
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
            use mlx_gen::gen_core::iris::train::depth::ITEM_DEPTH_KEY;
            use mlx_gen::gen_core::iris::train::OPTIONS_KEY;
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
            use mlx_gen::gen_core::iris::restoration_train::{
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
        "[[IRIS_MEMORY_PROBE]] backend=mlx task={task} engine={engine} size={size} ac={ac} \
         steps={} batch=1 full_finetune=true optimizer={} dtype={}",
        config.steps, config.optimizer, config.train_dtype
    );
    // Once, on this thread, before any weight is read. The cache limit can only be read by setting
    // it; `default` puts MLX's own value straight back.
    let default_cache_limit = set_cache_limit(0);
    let cache_limit = if cache == "none" {
        0
    } else {
        let cache_limit = emulated_limit.unwrap_or(default_cache_limit);
        set_cache_limit(cache_limit);
        cache_limit
    };
    if let Some(bytes) = emulated_limit {
        set_memory_limit(bytes);
    }
    let memory_limit = get_memory_limit();
    assert!(
        emulated_limit.is_none_or(|bytes| bytes == memory_limit),
        "the emulated memory limit did not take: MLX reports {memory_limit} bytes"
    );
    let recommended = mlx_gen::memory::recommended_working_set_bytes()
        .expect("Metal's recommended working set (MLX device_info)") as usize;
    // MLX's metal allocator: `gc_limit = min(0.95 * max_recommended_working_set_size, block_limit)`,
    // and `block_limit` is what `get_memory_limit` returns. Above it a malloc releases cached buffers
    // first, so an `active + cache` footprint cannot settle higher.
    let gc_limit = ((recommended as f64 * 0.95) as usize).min(memory_limit);
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] mlx_limits cache_mode={cache} limit_mode={limit} cache_limit_gb={:.2} gc_limit_gb={:.2} \
         memory_limit_gb={:.2} recommended_working_set_gb={:.2}",
        gb(cache_limit),
        gb(gc_limit),
        gb(memory_limit),
        gb(recommended)
    );
    reset_peak_memory();
    let start = std::time::Instant::now();
    let mut trainer = mlx_gen_iris::provider_registry()
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
    // The counters are printed whether or not the run survives.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.train(&req, &mut |p| {
            if let TrainingProgress::Training { step, total, loss } = p {
                eprintln!(
                    "[[IRIS_MEMORY_PROBE]] step {step}/{total} loss {loss:.4} ({:.1}s) \
                     active_peak_gb={:.2} active_gb={:.2} cache_gb={:.2}",
                    last.elapsed().as_secs_f32(),
                    gb(get_peak_memory()),
                    gb(get_active_memory()),
                    gb(get_cache_memory())
                );
                assert!(loss.is_finite(), "non-finite loss at step {step}");
                last = std::time::Instant::now();
            }
        })
    }));
    if let Ok(Err(error)) = &outcome {
        eprintln!("[[IRIS_MEMORY_PROBE]] train failed: {error:?}");
    }
    // The cache is read before the trainer is dropped: the drop moves its live buffers into it.
    let (active_peak, cache_end) = (get_peak_memory(), get_cache_memory());
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] mlx_memory active_peak_gb={:.2} cache_end_gb={:.2}",
        gb(active_peak),
        gb(cache_end)
    );
    drop(trainer);
    let res = match outcome {
        Ok(res) => res.unwrap(),
        Err(panic) => std::panic::resume_unwind(panic),
    };
    assert!(
        active_peak > 0,
        "MLX reported a zero active peak: not a measurement"
    );
    eprintln!(
        "[[IRIS_MEMORY_PROBE]] done: {} steps, final loss {:.4}, total {:.1}s → {}",
        res.steps,
        res.final_loss,
        start.elapsed().as_secs_f32(),
        res.adapter_path.display()
    );
    assert!(res.final_loss.is_finite());
}
