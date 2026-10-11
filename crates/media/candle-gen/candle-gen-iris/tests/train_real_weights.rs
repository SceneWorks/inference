//! A bounded real-weight LoRA run on the build's device (`cuda:0` under `--features cuda`, sc-25686)
//! — the Candle twin of the MLX `real_lora_trains_exports_and_renders`: a small dataset at 256²,
//! batch 1, four AdamW steps through the catalog trainer (bf16 mixed precision, cached Qwen3-VL
//! conditioning, EMA, a checkpoint at step 2, a preview from the in-progress EMA weights) → the
//! adapter artifact → loaded onto the real backbone through the Candle inference route
//! (`load_backbone_with_adapters`) → a render beside the unadapted base render. The adapted render
//! must equal the trainer's own preview (training and inference install the same residual) and
//! differ from the base. Writes PNGs + logs into `IRIS_OUT` and reports time and (with
//! `IRIS_VRAM_PROBE=1`) the train / render VRAM peaks — report-only, nothing is gated on them.
//!
//! Inputs: `IRIS_WEIGHTS_DIR`, `IRIS_TEXT_ENCODER_DIR`, `IRIS_OUT`, and optionally
//! `IRIS_TRAIN_DATA` (a directory of PNGs); without it the dataset is four crops/flips of the
//! repository photo `crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg`, so the CUDA dispatch
//! lane needs no extra input.
//!
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! train_real_weights:: --test-threads 1`

use std::path::{Path, PathBuf};

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::train::{AdapterMetadata, Prediction, OPTIONS_KEY};
use candle_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, Image, LoadSpec, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};
use candle_gen_iris::train::render_preview_with;
use candle_gen_iris::{load_backbone_with_adapters, IrisTextEncoder};

const CAPTION: &str = "a photo of a dog sitting on grass";
const PROMPT: &str = "a photo of a dog sitting on grass in the snow";
const SIZE: usize = 256;
const SEED: u64 = 3;

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .unwrap()
        .save(path)
        .unwrap();
}

/// `IRIS_TRAIN_DATA`'s PNGs, or four crops / flips of the repository photo written under `out`.
fn dataset(out: &Path) -> Vec<TrainingItem> {
    let mut paths: Vec<PathBuf> = match std::env::var("IRIS_TRAIN_DATA") {
        Ok(dir) => std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "png"))
            .collect(),
        Err(_) => {
            let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg");
            let rgb = image::open(photo)
                .expect("the repo photo decodes")
                .to_rgb8();
            let (w, h) = rgb.dimensions();
            let side = w.min(h);
            let dir = out.join("train-data");
            std::fs::create_dir_all(&dir).unwrap();
            let crops = [
                (0, 0),
                ((w - side) / 2, (h - side) / 2),
                (w - side, h - side),
            ];
            let mut paths = Vec::new();
            for (i, (x, y)) in crops.into_iter().enumerate() {
                let crop = image::imageops::crop_imm(&rgb, x, y, side, side).to_image();
                let p = dir.join(format!("crop{i}.png"));
                crop.save(&p).unwrap();
                paths.push(p);
                if i == 1 {
                    let p = dir.join("crop1_flip.png");
                    image::imageops::flip_horizontal(&crop).save(&p).unwrap();
                    paths.push(p);
                }
            }
            paths
        }
    };
    paths.sort();
    assert!(!paths.is_empty());
    paths
        .into_iter()
        .map(|p| TrainingItem::captioned(p, CAPTION.into()))
        .collect()
}

#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and IRIS_OUT"]
fn real_lora_trains_exports_and_renders() {
    let weights = env_dir("IRIS_WEIGHTS_DIR");
    let te_dir = env_dir("IRIS_TEXT_ENCODER_DIR");
    let out = env_dir("IRIS_OUT");
    let items = dataset(&out);
    let probe_vram =
        std::env::var("IRIS_VRAM_PROBE").is_ok_and(|v| v == "1") && used_mib(probe_gpu()).is_some();
    let mut probe = probe_vram.then(VramProbe::start_rendered);
    let mut config = TrainingConfig {
        rank: 16,
        alpha: 16.0,
        learning_rate: 1e-4,
        steps: 4,
        batch_size: 1,
        resolution: SIZE as u32,
        save_every: 2,
        seed: SEED,
        sample_every: 4,
        sample_prompts: vec![PROMPT.into()],
        sample_steps: 20,
        sample_guidance_scale: 3.0,
        ..Default::default()
    };
    config.model_options.insert(
        OPTIONS_KEY.into(),
        serde_json::json!({"text_conditioning": "cached", "ema_decay": 0.5,
                           "preview_weights": "ema", "export_weights": "ema"}),
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights.clone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(te_dir.clone()),
    );
    let mut trainer = candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer("iris_3b", &spec)
        .unwrap();
    assert_eq!(trainer.descriptor().backend, "candle");
    let req = TrainingRequest {
        items,
        config,
        output_dir: out.join("train"),
        file_name: "dog_lora.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    };
    let start = std::time::Instant::now();
    let train_phase = probe.as_ref().map(VramProbe::phase);
    let mut last = start;
    let mut preview: Option<Image> = None;
    let res = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { step, total, loss } => {
                eprintln!(
                    "[[IRIS_CANDLE_TRAIN]] step {step}/{total} loss {loss:.4} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                assert!(loss.is_finite());
                last = std::time::Instant::now();
            }
            TrainingProgress::Sample { step, image, .. } => {
                save_png(&image, &out.join(format!("train_preview_step{step}.png")));
                eprintln!(
                    "[[IRIS_CANDLE_TRAIN]] preview at step {step} ({:.1}s)",
                    last.elapsed().as_secs_f32()
                );
                preview = Some(image);
                last = std::time::Instant::now();
            }
            other => eprintln!(
                "[[IRIS_CANDLE_TRAIN]] {other:?} ({:.1}s)",
                start.elapsed().as_secs_f32()
            ),
        })
        .unwrap();
    drop(trainer);
    if let (Some(p), Some(phase)) = (probe.as_mut(), train_phase) {
        p.end_load(phase);
    }
    eprintln!(
        "[[IRIS_CANDLE_TRAIN]] device={} train total {:.1}s → {}",
        if cfg!(feature = "cuda") {
            "cuda:0"
        } else {
            "cpu"
        },
        start.elapsed().as_secs_f32(),
        res.adapter_path.display()
    );
    assert_eq!(res.steps, 4);
    let preview = preview.expect("one preview at step 4");
    let bytes = std::fs::read(&res.adapter_path).unwrap();
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let meta = AdapterMetadata::from_map(
        &header
            .metadata()
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect(),
    )
    .unwrap();
    eprintln!(
        "[[IRIS_CANDLE_TRAIN]] adapter: {} targets, rank {}, {} bytes",
        meta.targets.len(),
        meta.rank,
        bytes.len()
    );

    // Render the base and the adapted backbone through the Candle inference route.
    let device = candle_gen::default_device().unwrap();
    let cfg = IrisConfig::from_dir(&weights).unwrap();
    let te = IrisTextEncoder::load(&te_dir, &cfg.text_encoder, &device).unwrap();
    let (c, u) = (te.encode(PROMPT).unwrap(), te.encode("").unwrap());
    drop(te);
    let render_phase = probe.as_ref().map(VramProbe::phase);
    let render = |adapters: &[AdapterSpec]| {
        let (dit, reports) = load_backbone_with_adapters(
            &weights,
            &cfg,
            DType::BF16,
            &device,
            adapters,
            IrisTask::Generation,
            "iris_3b",
        )
        .unwrap();
        assert_eq!(reports.len(), adapters.len());
        render_preview_with(
            &dit,
            20,
            3.0,
            cfg.flow.shift,
            1000,
            Prediction::Velocity,
            SIZE,
            (&c.states, &c.mask),
            Some((&u.states, &u.mask)),
            SEED,
        )
        .unwrap()
    };
    let t = std::time::Instant::now();
    let before = render(&[]);
    save_png(&before, &out.join("base_256.png"));
    let after = render(&[AdapterSpec::new(
        res.adapter_path.clone(),
        1.0,
        AdapterKind::Lora,
    )]);
    save_png(&after, &out.join("lora_adapted_256.png"));
    if let (Some(p), Some(phase)) = (probe.as_mut(), render_phase) {
        p.end_gen(phase);
    }
    let mean_abs = |a: &Image, b: &Image| {
        a.pixels
            .iter()
            .zip(&b.pixels)
            .map(|(x, y)| (*x as f32 - *y as f32).abs())
            .sum::<f32>()
            / a.pixels.len() as f32
    };
    let max_abs = |a: &Image, b: &Image| {
        a.pixels
            .iter()
            .zip(&b.pixels)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap()
    };
    eprintln!(
        "[[IRIS_CANDLE_TRAIN]] renders {:.1}s; mean |base − adapted| = {:.2}; |preview − provider| \
         max {} mean {:.3}",
        t.elapsed().as_secs_f32(),
        mean_abs(&before, &after),
        max_abs(&preview, &after),
        mean_abs(&preview, &after)
    );
    if let Some(p) = &probe {
        // `load` = the training run, `gen` = the two renders (report-only).
        eprintln!("[[IRIS_CANDLE_TRAIN]] vram {}", p.report());
    }
    assert!(
        mean_abs(&before, &after) > 0.0,
        "the adapter changed nothing"
    );
    assert!(
        max_abs(&preview, &after) <= 2,
        "the provider-loaded adapter must render the trainer's (EMA) preview"
    );
}
