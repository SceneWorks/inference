//! Real-weight evidence of the restoration trainer on the build's device (`cuda:0` under
//! `--features cuda`, sc-25688):
//!
//! * the pinned training-only networks against the committed upstream reference
//!   (`iris_restoration_train_real.safetensors`, `tools/dump_iris_restoration_train_realweight.py`):
//!   LPIPS-VGG (the five taps and the distances) and the OpenCLIP ConvNeXt-XXL critic trunk (the
//!   three maps and the pooled embedding HYPIR's critic reads), FP32;
//! * bounded train → export → restore runs of the full recipe (pixel + LPIPS + adversarial, the
//!   critic on the real trunk) through the catalog trainer on the real restoration export, 256²
//!   crop, batch 1, two steps (four micro-batches: G, D, G, D), bf16, EMA, AdamW weight decay 0.01,
//!   second-order degradation of the repository photo's crops → the `upscaler/` export → loaded by
//!   `iris_3b_restore` (no training-only component) → a ×4 restoration of a small image, beside the
//!   base export's: the paper's FULL fine-tune (`recipe_training_config`) and HYPIR's LoRA arm
//!   (rank 8, merged). Writes the PNGs, losses and per-step timings into `IRIS_OUT/<case>` (the export itself stays in
//!   a temp dir).
//!
//! Inputs: `IRIS_UPSCALER_DIR`, the pinned snapshot directories `IRIS_LPIPS_VGG16_DIR`,
//! `IRIS_LPIPS_LIN_DIR`, `IRIS_CRITIC_TRUNK_DIR` (`gen_core::iris::restoration_train`'s
//! `TRAINING_RESOURCES`), `IRIS_OUT`.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::restoration_train::{
    recipe_training_config, ConvNextConfig, TrainingResource, CRITIC_TRUNK_COMPONENT,
    CRITIC_TRUNK_RESOURCE, LPIPS_LIN_COMPONENT, LPIPS_LIN_RESOURCE, OPTIONS_KEY, VGG16_COMPONENT,
    VGG16_RESOURCE,
};
use candle_gen::gen_core::{
    CancelFlag, Image, InputSizing, LoadSpec, TargetSize, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, Transform, TransformRequest, WeightsSource,
};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};
use candle_gen_iris::train::critic::ConvNextTrunk;
use candle_gen_iris::train::lpips::Lpips;
use candle_gen_iris::IrisRestorer;

use crate::common::{fixture, host_f32};
use candle_gen::testkit::save_image_png;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

/// A pinned training resource's file inside its snapshot directory (`<var>`).
fn resource(var: &str, res: &TrainingResource) -> PathBuf {
    env_path(var).join(res.file)
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let d = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!("[[IRIS_RESTORE_TRAIN]] {name}: max|Δ|={d:.3e} peak={peak:.3e}");
    assert!(
        d <= tol * peak,
        "{name}: max|Δ|={d:.3e} > {tol:e}·{peak:.3e}"
    );
}

#[test]
#[ignore = "needs the pinned training resources (IRIS_LPIPS_VGG16_DIR, IRIS_LPIPS_LIN_DIR, IRIS_CRITIC_TRUNK_DIR)"]
fn real_lpips_and_critic_trunk_match_the_reference() {
    let device = candle_gen::default_device().unwrap();
    let g = fixture("iris_restoration_train_real.safetensors");
    let net = Lpips::load(
        &resource("IRIS_LPIPS_VGG16_DIR", &VGG16_RESOURCE),
        &resource("IRIS_LPIPS_LIN_DIR", &LPIPS_LIN_RESOURCE),
        &device,
    )
    .unwrap();
    let (in0, in1) = (
        g.require("lpips/in0").to_device(&device).unwrap(),
        g.require("lpips/in1").to_device(&device).unwrap(),
    );
    for (k, tap) in net.layer_distances(&in0, &in1).unwrap().iter().enumerate() {
        close(
            &format!("lpips tap{k}"),
            &host_f32(tap),
            &host_f32(g.require(&format!("lpips/tap{k}"))),
            1e-3,
        );
    }
    close(
        "lpips dist",
        &host_f32(&net.distance(&in0, &in1).unwrap()),
        &host_f32(g.require("lpips/dist")),
        1e-3,
    );
    drop(net);
    let trunk = ConvNextTrunk::load(
        &resource("IRIS_CRITIC_TRUNK_DIR", &CRITIC_TRUNK_RESOURCE),
        &ConvNextConfig::xxlarge(),
        DType::F32,
        &device,
    )
    .unwrap();
    let (maps, emb) = trunk
        .features(&g.require("clip/x").to_device(&device).unwrap())
        .unwrap();
    for (i, m) in maps.iter().enumerate() {
        close(
            &format!("clip feat{i}"),
            &host_f32(m),
            &host_f32(g.require(&format!("clip/feat{i}"))),
            2e-3,
        );
    }
    close(
        "clip feat3 (pooled)",
        &host_f32(&emb),
        &host_f32(g.require("clip/feat3")),
        2e-3,
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_real_weights::real_lpips_and_critic_trunk_match_the_reference");
}

/// Three crops of the repository photo (the trainer re-crops each to 256² and degrades it).
fn dataset(out: &Path) -> Vec<TrainingItem> {
    let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg");
    let rgb = candle_gen::gen_core::__model_input_reader!(
        image::open(photo),
        "crates/media/candle-gen/candle-gen-iris/tests/restoration_train_real_weights.rs::dataset/image::open",
        Reference,
        photo,
        |__model_input_path| image::open(__model_input_path)
    )
        .expect("the repo photo decodes")
        .to_rgb8();
    let (w, h) = rgb.dimensions();
    let side = w.min(h);
    let dir = out.join("restore-train-data");
    std::fs::create_dir_all(&dir).unwrap();
    [
        (0, 0),
        ((w - side) / 2, (h - side) / 2),
        (w - side, h - side),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (x, y))| {
        let p = dir.join(format!("crop{i}.png"));
        image::imageops::crop_imm(&rgb, x, y, side, side)
            .to_image()
            .save(&p)
            .unwrap();
        TrainingItem::captioned(p, String::new())
    })
    .collect()
}

/// The ×4 restoration case: the repository photo downscaled to 64x48 (the input) and the same
/// photo at 256x192 (the reference a restoration is scored against).
fn restore_case() -> (Image, image::RgbImage) {
    let photo = candle_gen::gen_core::__model_input_reader!(
        image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg"),
    ),
        "crates/media/candle-gen/candle-gen-iris/tests/restoration_train_real_weights.rs::restore_case/image::open",
        Reference,
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg"),
        |__model_input_path| image::open(__model_input_path)
    )
    .unwrap();
    let lq = photo
        .resize_exact(64, 48, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let gt = photo
        .resize_exact(256, 192, image::imageops::FilterType::Lanczos3)
        .to_rgb8();
    (
        Image {
            width: 64,
            height: 48,
            pixels: lq.into_raw(),
        },
        gt,
    )
}

/// Restore the case with the `upscaler/`-layout export in `export` — through the restoration
/// transform alone, no training-only component.
fn restore(export: &Path, input: &Image) -> Image {
    let restorer =
        IrisRestorer::load(&LoadSpec::new(WeightsSource::Dir(export.to_path_buf()))).unwrap();
    let t = std::time::Instant::now();
    let restored = restorer
        .apply(
            &TransformRequest {
                image: input.clone(),
                target: TargetSize::Scale(4.0),
                input_sizing: InputSizing::Original,
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!((restored.width, restored.height), (256, 192));
    eprintln!(
        "[[IRIS_RESTORE_TRAIN]] restored 64x48 → 256x192 with {} in {:.1}s",
        export.display(),
        t.elapsed().as_secs_f32()
    );
    restored
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64.powi(2) / mse.max(1e-12)).log10()
}

/// One bounded train → export → restore run of the full recipe (pixel + LPIPS + adversarial, the
/// critic on the real trunk) through the catalog trainer on the real restoration export: 256²
/// crops, batch 1, two steps (four micro-batches: G, D, G, D), bf16, EMA (decay 0.5 so two steps
/// show), second-order degradation of the repository photo's crops. The base export and the trained
/// export each restore the same ×4 case; the PSNR against the 256x192 reference before → after and
/// the largest pixel change are reported (two steps are not expected to improve PSNR; the export
/// must differ from the base — a run whose export restores identically trained nothing).
fn train_export_and_restore(case: &str, mut config: TrainingConfig) {
    let out = env_path("IRIS_OUT").join(case);
    std::fs::create_dir_all(&out).unwrap();
    let items = dataset(&out);
    let (input, gt) = restore_case();
    let base = env_path("IRIS_UPSCALER_DIR");
    let before = restore(&base, &input);
    save_image_png(&out.join("restore_base_x4.png"), &before);

    let probe_vram =
        std::env::var("IRIS_VRAM_PROBE").is_ok_and(|v| v == "1") && used_mib(probe_gpu()).is_some();
    let mut probe = probe_vram.then(VramProbe::start_rendered);
    config
        .model_options
        .insert(OPTIONS_KEY.into(), serde_json::json!({"ema_decay": 0.5}));
    let mut spec = LoadSpec::new(WeightsSource::Dir(base));
    // Snapshot directories: the trainer finds each pinned file inside its component directory.
    for (component, var) in [
        (VGG16_COMPONENT, "IRIS_LPIPS_VGG16_DIR"),
        (LPIPS_LIN_COMPONENT, "IRIS_LPIPS_LIN_DIR"),
        (CRITIC_TRUNK_COMPONENT, "IRIS_CRITIC_TRUNK_DIR"),
    ] {
        spec.components
            .insert(component.into(), WeightsSource::Dir(env_path(var)));
    }
    let mut trainer = candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer("iris_3b_restore", &spec)
        .unwrap();
    assert_eq!(trainer.descriptor().backend, "candle");
    eprintln!(
        "[[IRIS_RESTORE_TRAIN]] {case}: full_finetune={} rank={} lr={:e} weight_decay={}",
        config.full_finetune, config.rank, config.learning_rate, config.weight_decay
    );
    // The export (12 GB f32) and checkpoints live in a temp dir dropped with the test; only the
    // PNGs and logs go to the evidence directory.
    let work = tempfile::tempdir().unwrap();
    let req = TrainingRequest {
        items,
        config,
        output_dir: work.path().join("restore-train"),
        file_name: "restorer.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    };
    let start = std::time::Instant::now();
    let phase = probe.as_ref().map(VramProbe::phase);
    let mut last = start;
    let res = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { step, total, loss } => {
                eprintln!(
                    "[[IRIS_RESTORE_TRAIN]] {case}: step {step}/{total} generator loss {loss:.4} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                assert!(loss.is_finite());
                last = std::time::Instant::now();
            }
            other => eprintln!(
                "[[IRIS_RESTORE_TRAIN]] {case}: {other:?} ({:.1}s)",
                start.elapsed().as_secs_f32()
            ),
        })
        .unwrap();
    drop(trainer);
    if let (Some(p), Some(ph)) = (probe.as_mut(), phase) {
        p.end_load(ph);
    }
    eprintln!(
        "[[IRIS_RESTORE_TRAIN]] {case}: train total {:.1}s → {} (final generator loss {:.4})",
        start.elapsed().as_secs_f32(),
        res.adapter_path.display(),
        res.final_loss
    );
    assert_eq!(res.steps, 4);
    let export = res.adapter_path.parent().unwrap().to_path_buf();
    let after = restore(&export, &input);
    save_image_png(&out.join("restore_trained_x4.png"), &after);
    let moved = before
        .pixels
        .iter()
        .zip(&after.pixels)
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap();
    eprintln!(
        "[[IRIS_RESTORE_TRAIN]] {case}: restore PSNR vs the 256x192 reference {:.2} dB (base) → \
         {:.2} dB (trained); max |Δpixel| base → trained {moved}",
        psnr(&before.pixels, gt.as_raw()),
        psnr(&after.pixels, gt.as_raw())
    );
    if let Some(p) = &probe {
        eprintln!("[[IRIS_RESTORE_TRAIN]] {case}: vram {}", p.report());
    }
    assert!(
        moved > 0,
        "{case}: the trained export restores exactly as the base did"
    );
    drop(work);
}

/// HYPIR's LoRA arm (rank 8, the merged adapter) at the recipe's lr 1e-5 / weight decay 0.01.
#[test]
#[ignore = "needs the real restoration export (IRIS_UPSCALER_DIR), the training resources and IRIS_OUT"]
fn real_restoration_trains_exports_and_restores() {
    train_export_and_restore(
        "lora",
        TrainingConfig {
            rank: 8,
            alpha: 8.0,
            learning_rate: 1e-5,
            weight_decay: 0.01,
            steps: 4,
            batch_size: 1,
            resolution: 256,
            save_every: 2,
            seed: 5,
            train_dtype: "bf16".into(),
            ..Default::default()
        },
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_real_weights::real_restoration_trains_exports_and_restores");
}

/// The paper's recipe — a FULL fine-tune of the 3B generator (`recipe_training_config`: AdamW,
/// lr 1e-5, weight decay 0.01 for generator and critic, constant LR) — at a 256² crop, batch 1,
/// two steps. No intermediate checkpoint (each would write the f32 weights, EMA and AdamW state).
#[test]
#[ignore = "needs the real restoration export (IRIS_UPSCALER_DIR), the training resources, IRIS_OUT and ~100 GB"]
fn real_restoration_full_finetune_trains_exports_and_restores() {
    let mut config = recipe_training_config();
    assert!(config.full_finetune);
    config.steps = 4;
    config.batch_size = 1;
    config.resolution = 256;
    config.save_every = 0;
    config.seed = 5;
    config.train_dtype = "bf16".into();
    train_export_and_restore("full", config);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_real_weights::real_restoration_full_finetune_trains_exports_and_restores");
}
