//! The bounded real-weight **training peak** measurement on CUDA — `#[ignore]`d and `cuda`-only.
//!
//! The trainer's preflight prices a step from structural counts that sc-24163 calibrated against the
//! driver's live high-water. Some of those counts describe what candle's backward materializes, so
//! a change to the backward (the vendored `backprop.rs` that stops computing gradients for frozen
//! operands, sc-25686) moves the real peak and leaves the counts stale. This test re-measures it:
//! per cell, the preflight's derived peak (read from its refusal under a 1 KiB budget) beside the
//! driver's `CU_MEMPOOL_ATTR_USED_MEM_HIGH` over a 10-step run of the same request with the budget
//! disabled. It asserts the estimate covers the measured peak and prints both, so the log is the
//! evidence for any recalibration of the counts in `src/training.rs`.
//!
//! The cells are sc-24163's A1 shapes (rank 16, adamw8bit, bf16, previews on, the same captions and
//! instructions, 768² inputs) so the unit test pinning those measurements derives the identical
//! shape, plus one small **dense** cell: a non-checkpointed step backpropagates every block in one
//! `GradStore`, so it is where per-projection backward residue accumulates 32-fold.
//!
//! Needs the pinned `Qwen/Qwen-Image-2.1` snapshot at `CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT` and an
//! idle CUDA card. `QWEN21_TRAIN_CELL=<label>` runs one cell (one process per cell keeps each
//! high-water to its own run); unset runs them all. Dispatched by the `qwen-image-2-1-train`
//! profile of `.github/workflows/real-weights.yml`.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::cuda_mempool::MemPool;
use candle_gen::gen_core::train::{
    TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{CancelFlag, LoadSpec, NetworkType, Precision, Trainer, WeightsSource};
use candle_gen_qwen_image_2_1::training::QwenImage21Trainer;

const STEPS: u32 = 10;
const SUBJECTS: [&str; 8] = [
    "a cat",
    "a house",
    "a tree",
    "a car",
    "a mountain",
    "a coffee cup",
    "a bicycle",
    "a robot",
];
const EDIT_INSTRUCTION: &str =
    "zxq invert: invert the colours of image 1 and posterize them, using the key in image 2";

struct Cell {
    label: &'static str,
    edit: bool,
    network: NetworkType,
    resolution: u32,
    checkpointed: bool,
}

const CELLS: [Cell; 6] = [
    Cell {
        label: "t2i-lora-256-dense",
        edit: false,
        network: NetworkType::Lora,
        resolution: 256,
        checkpointed: false,
    },
    Cell {
        label: "t2i-lora-768-ckpt",
        edit: false,
        network: NetworkType::Lora,
        resolution: 768,
        checkpointed: true,
    },
    Cell {
        label: "t2i-lokr-768-ckpt",
        edit: false,
        network: NetworkType::Lokr,
        resolution: 768,
        checkpointed: true,
    },
    Cell {
        label: "t2i-lora-1024-ckpt",
        edit: false,
        network: NetworkType::Lora,
        resolution: 1024,
        checkpointed: true,
    },
    Cell {
        label: "edit-lokr-768-ckpt-1ref",
        edit: true,
        network: NetworkType::Lokr,
        resolution: 768,
        checkpointed: true,
    },
    Cell {
        label: "edit-lora-768-ckpt-1ref",
        edit: true,
        network: NetworkType::Lora,
        resolution: 768,
        checkpointed: true,
    },
];

fn snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT to the pinned snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

/// A 768² RGB PNG at `path`: a flat ground with `seed`-placed bars, so items differ. Pixel content
/// does not move the footprint; the geometry and the captions do.
fn write_image(path: &Path, seed: u32, ground: [u8; 3]) {
    let mut img = image::RgbImage::from_pixel(768, 768, image::Rgb(ground));
    for k in 0..6u32 {
        let x0 = (seed * 97 + k * 131) % 600;
        let y0 = (seed * 53 + k * 71) % 600;
        let colour = image::Rgb([
            (k * 40) as u8,
            (255 - k * 30) as u8,
            (seed * 29 % 256) as u8,
        ]);
        for y in y0..y0 + 150 {
            for x in x0..x0 + 150 {
                img.put_pixel(x, y, colour);
            }
        }
    }
    img.save(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// sc-24163's A1 datasets: eight captioned style images, or six one-reference edit pairs.
fn dataset(dir: &Path, edit: bool) -> Vec<TrainingItem> {
    std::fs::create_dir_all(dir).unwrap();
    if edit {
        (0..6u32)
            .map(|i| {
                let target = dir.join(format!("tgt_{i}.png"));
                let source = dir.join(format!("src_{i}.png"));
                write_image(&target, 2000 + i, [40, 40, 40]);
                write_image(&source, 3000 + i, [200, 200, 200]);
                TrainingItem::edit_pair(target, EDIT_INSTRUCTION.into(), vec![source])
            })
            .collect()
    } else {
        SUBJECTS
            .iter()
            .enumerate()
            .map(|(i, subject)| {
                let path = dir.join(format!("style_{i}.png"));
                write_image(&path, 1000 + i as u32, [250, 220, 30]);
                TrainingItem::captioned(
                    path,
                    format!(
                        "zxq style, {subject} made of concentric teal and orange rings on yellow"
                    ),
                )
            })
            .collect()
    }
}

fn request(cell: &Cell, root: &Path) -> TrainingRequest {
    let sample_prompt = if cell.edit {
        EDIT_INSTRUCTION
    } else {
        "zxq style, a lighthouse made of concentric teal and orange rings on yellow"
    };
    TrainingRequest {
        items: dataset(&root.join("data"), cell.edit),
        config: TrainingConfig {
            rank: 16,
            alpha: 16.0,
            learning_rate: 4e-4,
            steps: STEPS,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: cell.checkpointed,
            train_dtype: "bf16".into(),
            resolution: cell.resolution,
            save_every: 50,
            seed: 42,
            optimizer: "adamw8bit".into(),
            weight_decay: 1e-4,
            network_type: cell.network,
            timestep_type: "sigmoid".into(),
            timestep_bias: "balanced".into(),
            loss_type: "mse".into(),
            sample_every: 100,
            sample_prompts: vec![sample_prompt.into()],
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            ..Default::default()
        },
        output_dir: root.join("out"),
        file_name: "peak.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    }
}

fn load(budget: u64) -> QwenImage21Trainer {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.precision = Precision::Bf16;
    let mut trainer = QwenImage21Trainer::load(&spec).expect("load the qwen_image_2_1 trainer");
    trainer.set_memory_budget_override(Some(budget));
    trainer
}

/// The preflight's derived peak and train-stage figures in GiB, from its refusal message.
fn derived_gib(message: &str) -> (f64, f64) {
    let after = |marker: &str| -> f64 {
        let rest = &message[message
            .find(marker)
            .unwrap_or_else(|| panic!("no {marker:?} in the refusal: {message}"))
            + marker.len()..];
        let end = rest.find(' ').unwrap_or(rest.len());
        rest[..end]
            .parse()
            .unwrap_or_else(|e| panic!("{marker:?} in {message}: {e}"))
    };
    (after("derived peak memory is ~"), after("train ~"))
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[test]
#[ignore]
fn training_peak_is_within_the_preflight_estimate() {
    let only = std::env::var("QWEN21_TRAIN_CELL").ok();
    let cells: Vec<&Cell> = CELLS
        .iter()
        .filter(|c| only.as_deref().is_none_or(|o| o == c.label))
        .collect();
    assert!(
        !cells.is_empty(),
        "QWEN21_TRAIN_CELL={only:?} names no cell"
    );
    let pool = MemPool::device_default(0).expect("the CUDA default memory pool");
    for cell in cells {
        let root = tempfile::tempdir().unwrap();
        let req = request(cell, root.path());

        // The derived figure: the preflight refuses before loading anything heavy.
        let Err(refusal) = load(1 << 10).train(&req, &mut |_| {}) else {
            panic!("{}: a 1 KiB budget must refuse", cell.label);
        };
        let (derived_peak, derived_train) = derived_gib(&refusal.to_string());

        // The measurement: the same request with the budget disabled.
        let mut trainer = load(u64::MAX);
        assert!(pool.reset_high_water(), "reset the pool high-water");
        let t0 = Instant::now();
        let mut steps = 0u32;
        trainer
            .train(&req, &mut |p| {
                if let TrainingProgress::Training { step, .. } = p {
                    steps = step;
                }
            })
            .unwrap_or_else(|e| panic!("{}: training failed: {e}", cell.label));
        let used_high = pool.used_high().expect("USED_MEM_HIGH");
        let reserved_high = pool.reserved_high().expect("RESERVED_MEM_HIGH");
        assert_eq!(steps, STEPS, "{}: ran every step", cell.label);
        println!(
            "QWEN21_TRAIN_PEAK cell={} derivedPeakGiB={derived_peak:.1} derivedTrainGiB={derived_train:.1} \
             usedHighBytes={used_high} usedHighGiB={:.2} reservedHighBytes={reserved_high} \
             derivedOverMeasured={:.3} wallSeconds={:.1}",
            cell.label,
            gib(used_high),
            derived_peak / gib(used_high),
            t0.elapsed().as_secs_f64()
        );
        // The preflight's one-decimal figure may round down by up to 0.05 GiB.
        assert!(
            gib(used_high) <= derived_peak + 0.05,
            "{}: measured {:.2} GiB over the preflight's {derived_peak:.1} GiB",
            cell.label,
            gib(used_high)
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_peak_real_weights::training_peak_is_within_the_preflight_estimate");
}

/// sc-26857: one source-owned invocation, four total updates and one saved-adapter sample.
/// This sibling preserves the historical peak matrix and its admission recipe.
#[cfg(feature = "model-inference-proof")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn qwen21_bounded_acceleration_acceptance() {
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest};
    use candle_gen::testkit::PeakSampler;
    use candle_gen::train::profile::with_bounded_cohort_for;
    use candle_gen_qwen_image_2_1::{adapters, quant::Tier};

    let root =
        PathBuf::from(std::env::var("MODEL_PROOF_OUTPUT_DIR").expect("executor-owned output"))
            .join("cohort_acceptance");
    std::fs::create_dir(&root).expect("fresh acceptance directory");
    let data = root.join("data");
    std::fs::create_dir(&data).unwrap();
    let target = data.join("target.png");
    write_image(&target, 7, [250, 220, 30]);
    let req = TrainingRequest {
        items: vec![TrainingItem::captioned(
            target,
            "zxq style, teal and orange rings".into(),
        )],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.,
            learning_rate: 1e-3,
            steps: 4,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: true,
            train_dtype: "bf16".into(),
            resolution: if candle_gen::train::comparison::requested_stage().is_some() {
                512
            } else {
                64
            },
            save_every: 0,
            sample_every: 0,
            sample_prompts: vec![],
            seed: 7,
            optimizer: "adamw".into(),
            weight_decay: 0.,
            lr_warmup_steps: 0,
            network_type: NetworkType::Lora,
            ..Default::default()
        },
        output_dir: root.join("out"),
        file_name: "adapter.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    if let Some(stage) = candle_gen::train::comparison::requested_stage() {
        use candle_gen::train::comparison::{self, ComparisonCadence, ComparisonStage};
        assert_ne!(
            stage,
            ComparisonStage::MinusReferenceOwner,
            "Qwen uses the composable budgeted reference"
        );
        comparison::with_stage(
            &root.join("comparison_metrics.json"),
            "qwen_image_2_1",
            ComparisonCadence::Ordinary,
            || {
                let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
                spec.precision = Precision::Bf16;
                let mut trainer =
                    QwenImage21Trainer::load(&spec).expect("fresh Qwen comparison trainer");
                let output = trainer
                    .train(&req, &mut |_| {})
                    .expect("four matched updates");
                assert_eq!(output.steps, 4);
                assert!(output.final_loss.is_finite());
            },
        );
        println!("\nMODEL_INFERENCE_BODY_COMPLETED train_peak_real_weights::qwen21_bounded_acceleration_acceptance");
        return;
    }
    let sampler = PeakSampler::start_rendered();
    let output = with_bounded_cohort_for(
        &root.join("metrics.json"),
        &[
            "qwen21_frozen_initial_text",
            "qwen21_joint_geometry",
            "qwen21_frozen_qkv_pack",
            "qwen21_shared_rms",
        ],
        || {
            let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
            spec.precision = Precision::Bf16;
            let mut trainer = QwenImage21Trainer::load(&spec).expect("admitted real Qwen trainer");
            let mut observed = Vec::new();
            let output = trainer
                .train(&req, &mut |progress| {
                    if let TrainingProgress::Training { step, loss, .. } = progress {
                        assert!(loss.is_finite());
                        observed.push(step);
                    }
                })
                .expect("four real updates complete under native admission");
            assert_eq!(observed, [1, 2, 3, 4]);
            assert_eq!(output.steps, 4);
            output
        },
    );
    // Reopen the exact native export and validate its production target/shape contract.
    let bytes = std::fs::read(&output.adapter_path).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(
        parsed.len(),
        32 * 7 * 3,
        "two factors plus alpha per default target"
    );
    let mut nonzero = false;
    for (name, view) in parsed.tensors() {
        assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}");
        for value in view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        {
            assert!(value.is_finite(), "{name}");
            if name.ends_with(".lora_B.weight") {
                nonzero |= value != 0.;
            }
        }
    }
    assert!(
        nonzero,
        "saved trained up factors must carry an actual update"
    );
    let adapter = AdapterSpec::new(output.adapter_path.clone(), 1., AdapterKind::Lora);
    adapters::preflight(&snapshot(), std::slice::from_ref(&adapter), Tier::Bf16).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![adapter]);
    let generator =
        candle_gen_qwen_image_2_1::load(&spec).expect("strict native adapter reload/apply");
    let request = GenerationRequest {
        prompt: "zxq style, teal and orange rings".into(),
        width: 64,
        height: 64,
        steps: Some(2),
        seed: Some(7),
        ..Default::default()
    };
    let GenerationOutput::Images(images) = generator
        .generate(&request, &mut |_| {})
        .expect("one adapted generated sample")
    else {
        panic!("RGB image expected")
    };
    assert_eq!(images.len(), 1);
    let image = &images[0];
    assert_eq!(
        (image.width, image.height, image.pixels.len()),
        (64, 64, 64 * 64 * 3)
    );
    image::save_buffer(
        root.join("adapter_sample.png"),
        &image.pixels,
        image.width,
        image.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    let peak = sampler.stop();
    assert!(peak > 0, "physical device sampler must observe usage");
    std::fs::write(
        root.join("peak_device_bytes.txt"),
        (peak * 1024 * 1024).to_string(),
    )
    .unwrap();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_peak_real_weights::qwen21_bounded_acceleration_acceptance");
}
