//! Real-weight evidence of the MLX restoration trainer (sc-25688) — the twin of the Candle
//! `restoration_train_real_weights.rs`, on the default (Metal) stream:
//!
//! * the pinned training-only networks against the committed upstream reference
//!   (`iris_restoration_train_real.safetensors`, `tools/dump_iris_restoration_train_realweight.py`):
//!   LPIPS-VGG (the five taps and the distances) and the OpenCLIP ConvNeXt-XXL critic trunk (the
//!   three maps and the pooled embedding HYPIR's critic reads), FP32;
//! * one bounded train → export → restore run of the full recipe (pixel + LPIPS + adversarial, the
//!   critic on the real trunk) through the catalog trainer: a LoRA on the real restoration export
//!   at a 256² crop, batch 1, two steps (four micro-batches: G, D, G, D), bf16, EMA, second-order
//!   degradation of the repository photo's crops → the merged `upscaler/` export → loaded by
//!   `iris_3b_restore` (no training-only component) → a ×4 restoration of a small image. Writes the
//!   PNG + per-step timings into `IRIS_OUT`; VRAM is report-only (`IRIS_VRAM_PROBE=1`).
//!
//! Inputs: `IRIS_UPSCALER_DIR`, the pinned snapshot directories `IRIS_LPIPS_VGG16_DIR`,
//! `IRIS_LPIPS_LIN_DIR`, `IRIS_CRITIC_TRUNK_DIR` (`gen_core::iris::restoration_train`'s
//! `TRAINING_RESOURCES`), `IRIS_OUT`.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::restoration_train::{
    ConvNextConfig, TrainingResource, CRITIC_TRUNK_COMPONENT, CRITIC_TRUNK_RESOURCE,
    LPIPS_LIN_COMPONENT, LPIPS_LIN_RESOURCE, OPTIONS_KEY, VGG16_COMPONENT, VGG16_RESOURCE,
};
use mlx_gen::gen_core::{
    CancelFlag, Image, InputSizing, LoadSpec, TargetSize, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, Transform, TransformRequest, WeightsSource,
};
use mlx_gen_iris::train::critic::ConvNextTrunk;
use mlx_gen_iris::train::lpips::Lpips;
use mlx_gen_iris::IrisRestorer;
use mlx_rs::Dtype;

use crate::common::{fixture, host_f32};

fn req<'a>(g: &'a mlx_gen::weights::Weights, k: &str) -> &'a mlx_rs::Array {
    g.require(k).unwrap()
}

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
    let g = fixture("iris_restoration_train_real.safetensors");
    let net = Lpips::load(
        &resource("IRIS_LPIPS_VGG16_DIR", &VGG16_RESOURCE),
        &resource("IRIS_LPIPS_LIN_DIR", &LPIPS_LIN_RESOURCE),
    )
    .unwrap();
    let (in0, in1) = (req(&g, "lpips/in0").clone(), req(&g, "lpips/in1").clone());
    for (k, tap) in net.layer_distances(&in0, &in1).unwrap().iter().enumerate() {
        close(
            &format!("lpips tap{k}"),
            &host_f32(tap),
            &host_f32(req(&g, &format!("lpips/tap{k}"))),
            1e-3,
        );
    }
    close(
        "lpips dist",
        &host_f32(&net.distance(&in0, &in1).unwrap()),
        &host_f32(req(&g, "lpips/dist")),
        1e-3,
    );
    drop(net);
    let trunk = ConvNextTrunk::load(
        &resource("IRIS_CRITIC_TRUNK_DIR", &CRITIC_TRUNK_RESOURCE),
        &ConvNextConfig::xxlarge(),
        Dtype::Float32,
    )
    .unwrap();
    let (maps, emb) = trunk.features(req(&g, "clip/x")).unwrap();
    // Metal's f32 GEMM / conv accumulate at reduced precision (TF32-class): through 37 ConvNeXt
    // blocks that measured 2.8e-3 of peak on the stage-2 map (nax-macos-2, 2026-10-11); the Candle
    // twin holds 2e-3 on CUDA and ~2e-6 on CPU.
    const METAL: f32 = 6e-3;
    for (i, m) in maps.iter().enumerate() {
        let m = &m.transpose_axes(&[0, 3, 1, 2]).unwrap();
        close(
            &format!("clip feat{i}"),
            &host_f32(m),
            &host_f32(req(&g, &format!("clip/feat{i}"))),
            METAL,
        );
    }
    close(
        "clip feat3 (pooled)",
        &host_f32(&emb),
        &host_f32(req(&g, "clip/feat3")),
        METAL,
    );
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .unwrap()
        .save(path)
        .unwrap();
}

/// Three crops of the repository photo (the trainer re-crops each to 256² and degrades it).
fn dataset(out: &Path) -> Vec<TrainingItem> {
    let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg");
    let rgb = image::open(photo)
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

#[test]
#[ignore = "needs the real restoration export (IRIS_UPSCALER_DIR), the training resources and IRIS_OUT"]
fn real_restoration_trains_exports_and_restores() {
    let out = env_path("IRIS_OUT");
    let items = dataset(&out);
    let mut config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-5,
        steps: 4,
        batch_size: 1,
        resolution: 256,
        save_every: 2,
        seed: 5,
        train_dtype: "bf16".into(),
        ..Default::default()
    };
    config
        .model_options
        .insert(OPTIONS_KEY.into(), serde_json::json!({"ema_decay": 0.5}));
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_path("IRIS_UPSCALER_DIR")));
    // Snapshot directories: the trainer finds each pinned file inside its component directory.
    for (component, var) in [
        (VGG16_COMPONENT, "IRIS_LPIPS_VGG16_DIR"),
        (LPIPS_LIN_COMPONENT, "IRIS_LPIPS_LIN_DIR"),
        (CRITIC_TRUNK_COMPONENT, "IRIS_CRITIC_TRUNK_DIR"),
    ] {
        spec.components
            .insert(component.into(), WeightsSource::Dir(env_path(var)));
    }
    let mut trainer = mlx_gen_iris::provider_registry()
        .unwrap()
        .load_trainer("iris_3b_restore", &spec)
        .unwrap();
    assert_eq!(trainer.descriptor().backend, "mlx");
    let req = TrainingRequest {
        items,
        config,
        output_dir: out.join("restore-train"),
        file_name: "restorer.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    };
    let start = std::time::Instant::now();
    let mut last = start;
    let res = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { step, total, loss } => {
                eprintln!(
                    "[[IRIS_RESTORE_TRAIN]] step {step}/{total} generator loss {loss:.4} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                assert!(loss.is_finite());
                last = std::time::Instant::now();
            }
            other => eprintln!(
                "[[IRIS_RESTORE_TRAIN]] {other:?} ({:.1}s)",
                start.elapsed().as_secs_f32()
            ),
        })
        .unwrap();
    drop(trainer);
    eprintln!(
        "[[IRIS_RESTORE_TRAIN]] train total {:.1}s → {}",
        start.elapsed().as_secs_f32(),
        res.adapter_path.display()
    );
    assert_eq!(res.steps, 4);
    let export = res.adapter_path.parent().unwrap().to_path_buf();
    // The export runs in the restoration transform with no training-only component.
    let restorer = IrisRestorer::load(&LoadSpec::new(WeightsSource::Dir(export))).unwrap();
    let photo = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg"),
    )
    .unwrap()
    .resize_exact(64, 48, image::imageops::FilterType::Triangle)
    .to_rgb8();
    let input = Image {
        width: 64,
        height: 48,
        pixels: photo.into_raw(),
    };
    let t = std::time::Instant::now();
    let restored = restorer
        .apply(
            &TransformRequest {
                image: input,
                target: TargetSize::Scale(4.0),
                input_sizing: InputSizing::Original,
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!((restored.width, restored.height), (256, 192));
    save_png(&restored, &out.join("restore_trained_x4.png"));
    eprintln!(
        "[[IRIS_RESTORE_TRAIN]] restored 64x48 → 256x192 in {:.1}s",
        t.elapsed().as_secs_f32()
    );
}
