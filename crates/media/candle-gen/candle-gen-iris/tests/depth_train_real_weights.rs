//! A bounded real-weight **depth full fine-tune** on the build's device (`cuda:0` under
//! `--features cuda`; sc-25687) — the Candle twin of the MLX test: the pinned `depth/` export
//! (~3B parameters) trained for a few AdamW steps (bf16 mixed precision, f32 masters, batch 1, a
//! small square size) on one aligned example — a center crop of the repository photo
//! (`crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg`) with a synthetic depth ramp (near at
//! the bottom, far at the top) — then the exported `depth/` directory is loaded by the standalone
//! `iris_3b_depth` estimator unchanged and its prediction must have moved **toward** the target
//! (the mean |prediction − normalized target| drops below the release export's). `#[ignore]`d: it
//! needs the 12 GB export; the export (12 GB) goes to a temp directory. With
//! `IRIS_VRAM_PROBE=1` the train phase's VRAM peak is reported (report-only).
//!
//! Inputs: `IRIS_WEIGHTS_DIR` (the snapshot root), optional `IRIS_OUT` (logs / previews),
//! `IRIS_DEPTH_TRAIN_SIZE` (default 128), `IRIS_DEPTH_TRAIN_STEPS` (default 4).
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! depth_train_real_weights:: --test-threads 1`

use std::path::{Path, PathBuf};

use candle_gen::gen_core::iris::depth::{
    colorize_inferno, DepthMap, DepthRequest, DepthResolution,
};
use candle_gen::gen_core::iris::train::depth::{
    load_depth_sample, normalize_log_depth, DepthLossPlan, ITEM_DEPTH_KEY,
};
use candle_gen::gen_core::iris::train::OPTIONS_KEY;
use candle_gen::gen_core::{
    CancelFlag, Image, LoadSpec, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn write_npy(path: &Path, h: usize, w: usize, f: impl Fn(usize, usize) -> f32) {
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

/// The photo's center square resized to `size`, and a depth ramp 1 m (bottom) → 10 m (top).
fn example(dir: &Path, size: usize) -> (TrainingItem, Image) {
    let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg");
    let rgb = candle_gen::gen_core::__model_input_reader!(
        image::open(photo),
        "crates/media/candle-gen/candle-gen-iris/tests/depth_train_real_weights.rs::example/image::open",
        Reference,
        photo,
        |__model_input_path| image::open(__model_input_path)
    )
        .expect("the repo photo decodes")
        .to_rgb8();
    let (w, h) = rgb.dimensions();
    let side = w.min(h);
    let crop =
        image::imageops::crop_imm(&rgb, (w - side) / 2, (h - side) / 2, side, side).to_image();
    let img = image::imageops::resize(
        &crop,
        size as u32,
        size as u32,
        image::imageops::FilterType::CatmullRom,
    );
    std::fs::create_dir_all(dir).unwrap();
    let rgb_path = dir.join("dog.png");
    img.save(&rgb_path).unwrap();
    let depth_path = dir.join("dog_depth.npy");
    write_npy(&depth_path, size, size, |y, _| {
        1.0 + 9.0 * (1.0 - y as f32 / (size - 1) as f32)
    });
    let mut item = TrainingItem::captioned(rgb_path, String::new());
    item.model_options.insert(
        ITEM_DEPTH_KEY.into(),
        serde_json::json!({"path": depth_path.to_str().unwrap()}),
    );
    let image = Image {
        width: size as u32,
        height: size as u32,
        pixels: img.into_raw(),
    };
    (item, image)
}

fn estimate(dir: &Path, image: &Image) -> Vec<f32> {
    let start = std::time::Instant::now();
    let est = candle_gen_iris::depth::load(&LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())))
        .unwrap();
    let req = DepthRequest {
        image: image.clone(),
        resolution: DepthResolution::Native,
        cancel: Default::default(),
    };
    let values = est.estimate(&req, &mut |_| {}).unwrap().map.values;
    eprintln!(
        "[[IRIS_DEPTH_TRAIN_CANDLE]] estimate from {} ({:.1}s)",
        dir.display(),
        start.elapsed().as_secs_f32()
    );
    values
}

fn mean_error(pred: &[f32], target: &[f32]) -> f64 {
    pred.iter()
        .zip(target)
        .map(|(p, t)| (p - t).abs() as f64)
        .sum::<f64>()
        / pred.len() as f64
}

fn save_preview(values: &[f32], size: usize, path: &Path) {
    let img = colorize_inferno(&DepthMap {
        width: size as u32,
        height: size as u32,
        values: values.to_vec(),
    });
    image::RgbImage::from_raw(img.width, img.height, img.pixels)
        .unwrap()
        .save(path)
        .unwrap();
}

#[test]
#[ignore = "needs the real depth export (IRIS_WEIGHTS_DIR)"]
fn real_depth_full_finetune_trains_exports_and_estimates() {
    let release = env_dir("IRIS_WEIGHTS_DIR").join("depth");
    let size = env_usize("IRIS_DEPTH_TRAIN_SIZE", 128);
    let steps = env_usize("IRIS_DEPTH_TRAIN_STEPS", 4);
    let work = tempfile::tempdir().unwrap();
    let out = std::env::var("IRIS_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| work.path().join("evidence"));
    std::fs::create_dir_all(&out).unwrap();
    let (item, image) = example(&work.path().join("data"), size);
    let sample = load_depth_sample(&item).unwrap();
    let target =
        normalize_log_depth(&sample.depth, &sample.valid, &DepthLossPlan::default(), 2).unwrap();

    let before = estimate(&release, &image);
    save_preview(&before, size, &out.join("depth_before.png"));
    let err_before = mean_error(&before, &target);

    let mut config = TrainingConfig {
        learning_rate: 3e-5,
        weight_decay: 0.01,
        steps: steps as u32,
        batch_size: 1,
        resolution: size as u32,
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
    let mut trainer = candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(
            "iris_3b_depth",
            &LoadSpec::new(WeightsSource::Dir(release.clone())),
        )
        .unwrap();
    let req = TrainingRequest {
        items: vec![item],
        config,
        output_dir: work.path().join("train"),
        file_name: "dog_depth.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    };
    let probe_vram =
        std::env::var("IRIS_VRAM_PROBE").is_ok_and(|v| v == "1") && used_mib(probe_gpu()).is_some();
    let mut probe = probe_vram.then(VramProbe::start_rendered);
    let train_phase = probe.as_ref().map(VramProbe::phase);
    let start = std::time::Instant::now();
    let mut last = start;
    let mut losses = Vec::new();
    let res = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { step, total, loss } => {
                eprintln!(
                    "[[IRIS_DEPTH_TRAIN_CANDLE]] step {step}/{total} loss {loss:.4} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                losses.push(loss);
                last = std::time::Instant::now();
            }
            other => eprintln!(
                "[[IRIS_DEPTH_TRAIN_CANDLE]] {other:?} ({:.1}s)",
                start.elapsed().as_secs_f32()
            ),
        })
        .unwrap();
    drop(trainer);
    if let (Some(p), Some(phase)) = (probe.as_mut(), train_phase) {
        p.end_load(phase);
    }
    if let Some(p) = &probe {
        eprintln!("[[IRIS_DEPTH_TRAIN_CANDLE]] train vram {}", p.report());
    }
    eprintln!(
        "[[IRIS_DEPTH_TRAIN_CANDLE]] {size}x{size}, {steps} steps, train+export {:.1}s → {}",
        start.elapsed().as_secs_f32(),
        res.adapter_path.display()
    );
    assert!(losses.iter().all(|l| l.is_finite()));
    let export = res.adapter_path.parent().unwrap().to_path_buf();
    let after = estimate(&export, &image);
    save_preview(&after, size, &out.join("depth_after.png"));
    let err_after = mean_error(&after, &target);
    eprintln!(
        "[[IRIS_DEPTH_TRAIN_CANDLE]] mean |pred − target|: release {err_before:.4} → trained \
         {err_after:.4}; losses {losses:?}"
    );
    assert!(
        err_after < err_before,
        "the trained export must move toward the target"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED depth_train_real_weights::real_depth_full_finetune_trains_exports_and_estimates");
}
