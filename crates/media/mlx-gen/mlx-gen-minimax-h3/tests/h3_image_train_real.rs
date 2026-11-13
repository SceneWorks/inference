//! SW-7097: the MiniMax-H3-Image LoRA trainer on real weights (MLX), over the q4 tier with the
//! frozen DiT streamed one block at a time and gradient checkpointing on — the trainer's default
//! configuration at its 2048² default edge.
//!
//! * `h3_image_train_t2i_saves_reproduces_and_resumes_q4` — AC1 text-to-image and AC2: a
//!   three-step run is admitted by the training memory estimate, trains, and saves an adapter in
//!   the PEFT layout the H3-Image LoRA install reads; the registered generator loading that file
//!   reproduces the trainer's own final-step sample (mean |Δ| ≤ [`SAMPLE_MEAN_TOL`] and at most
//!   [`SAMPLE_OUTLIER_FRACTION`] of channels off by more than [`SAMPLE_OUTLIER_LEVELS`]); and a
//!   run resumed from the step-2 checkpoint makes the same step-3 update as the uninterrupted run
//!   (every factor within [`RESUME_MAX_ABS`]).
//! * `h3_image_train_edit_saves_and_reproduces_q4` — AC1 edit: a two-reference edit-pair run,
//!   saved with the edit marker, reproduced the same way.
//!
//! `#[ignore]`d: needs `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`), the converted q4
//! DiT tier (`MINIMAX_H3_IMAGE_TIERS_ROOT/q4`) and MiniMax-H3's `text_encoder/`
//! (`MINIMAX_H3_SNAPSHOT`). The `minimax-h3-image-train` real-weights profile runs it on `mlx-minimax-h3-image-train`. The dataset is drawn procedurally into a scratch directory;
//! nothing is downloaded. The t2i adapter is copied to `MINIMAX_H3_IMAGE_TRAINED_LORA_OUT` for the
//! Candle load gate.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, LoadShape, LoadSpec, Quant, Trainer,
    TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::h3_image::IMAGE_DIT_COMPONENT;
use mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT;
use mlx_gen_minimax_h3::training::{preview_request, MiniMaxH3ImageTrainer};

/// The trainer's sample and the generator's render of the saved adapter run the same code on the
/// same inputs; these bound the only freedom left (Metal kernel scheduling) in 8-bit levels.
const SAMPLE_MEAN_TOL: f64 = 0.5;
const SAMPLE_OUTLIER_LEVELS: u8 = 8;
const SAMPLE_OUTLIER_FRACTION: f64 = 1e-3;
/// A resumed step-3 update restores the step-2 factors and optimizer state exactly and draws the
/// same step; the factors may differ by at most this (f32).
const RESUME_MAX_ABS: f32 = 1e-6;

fn env_dir(name: &str) -> PathBuf {
    let raw = std::env::var(name).unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "{name} must be set; this test is #[ignore]d and asserts rather than skips"
    );
    PathBuf::from(raw)
}

/// The q4 trainer spec: the snapshot, the converted q4 DiT, and MiniMax-H3's text encoder (with
/// its vision tower, which an edit reads), streamed a layer at a time.
fn q4_spec() -> LoadSpec {
    let root = env_dir("MINIMAX_H3_IMAGE_SNAPSHOT");
    let dit = env_dir("MINIMAX_H3_IMAGE_TIERS_ROOT")
        .join("q4")
        .join(IMAGE_DIT_COMPONENT);
    let te = env_dir("MINIMAX_H3_SNAPSHOT").join("text_encoder");
    for probe in [dit.join("config.json"), te.join("config.json")] {
        assert!(probe.is_file(), "missing {}", probe.display());
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.quantize = Some(Quant::Q4);
    spec.components
        .insert(IMAGE_DIT_COMPONENT.to_owned(), WeightsSource::Dir(dit));
    spec.components
        .insert(TEXT_ENCODER_COMPONENT.to_owned(), WeightsSource::Dir(te));
    spec
}

fn scratch() -> tempfile::TempDir {
    match std::env::var("RUNNER_TEMP") {
        Ok(dir) if !dir.is_empty() => tempfile::tempdir_in(dir).expect("scratch in RUNNER_TEMP"),
        _ => tempfile::tempdir().expect("scratch"),
    }
}

/// A procedural RGB image: a smooth two-colour gradient with a solid disc, varied by `seed`.
fn draw(path: &Path, width: u32, height: u32, seed: u32) {
    let (cx, cy) = (
        width as f32 * (0.3 + 0.1 * (seed % 4) as f32),
        height as f32 * 0.5,
    );
    let r = width.min(height) as f32 * 0.2;
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let inside = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() < r;
            let g = x as f32 / width as f32;
            let px = if inside {
                [220, 40 + (seed * 30 % 120) as u8, 40]
            } else {
                [
                    (30.0 + 160.0 * g) as u8,
                    (60.0 + 100.0 * (1.0 - g)) as u8,
                    (120 + seed * 20 % 100) as u8,
                ]
            };
            pixels.extend_from_slice(&px);
        }
    }
    image::save_buffer(path, &pixels, width, height, image::ColorType::Rgb8).unwrap();
}

fn decode(path: &Path) -> Image {
    let img = image::open(path).unwrap().to_rgb8();
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

fn config(steps: u32) -> TrainingConfig {
    TrainingConfig {
        rank: 16,
        alpha: 16.0,
        learning_rate: 1e-4,
        steps,
        save_every: 2,
        seed: 7097,
        sample_every: steps,
        sample_prompts: vec!["a red disc on a gradient, sks style".into()],
        ..gen_core_train::h3_image::default_training_config()
    }
}

fn request(items: Vec<TrainingItem>, cfg: TrainingConfig, out: &Path) -> TrainingRequest {
    TrainingRequest {
        items,
        config: cfg,
        output_dir: out.to_path_buf(),
        file_name: "h3_image_lora.safetensors".into(),
        trigger_words: vec![],
        cancel: Default::default(),
    }
}

/// Train `req`, returning the output and every sample image the run reported (by step).
fn train(spec: &LoadSpec, req: &TrainingRequest) -> (PathBuf, f32, Vec<(u32, Image)>) {
    let mut trainer = MiniMaxH3ImageTrainer::load(spec).expect("load the trainer");
    trainer.validate(req).expect("the request validates");
    let mut samples = Vec::new();
    let started = std::time::Instant::now();
    let output = trainer
        .train(req, &mut |p| match p {
            TrainingProgress::Training { step, loss, .. } => println!(
                "[sw-7097 mlx] step {step} loss {loss:.5} ({:.0} s)",
                started.elapsed().as_secs_f64()
            ),
            TrainingProgress::Sample { step, image, .. } => samples.push((step, image)),
            _ => {}
        })
        .expect("training completes");
    assert!(output.final_loss.is_finite(), "{}", output.final_loss);
    (output.adapter_path, output.final_loss, samples)
}

/// The saved file is the PEFT layout the H3-Image install reads: a down/up pair and an alpha for
/// every block leaf, the reload stamps, and at least one trained (non-zero) up factor.
fn check_layout(path: &Path, edit: bool) -> usize {
    let w = Weights::from_file(path).unwrap();
    let downs: Vec<String> = w
        .keys()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .map(String::from)
        .collect();
    assert_eq!(downs.len(), 50 * 6, "every block leaf");
    for down in &downs {
        let stem = down.trim_end_matches(".lora_A.weight");
        assert!(stem.starts_with("transformer_blocks."), "{stem}");
        assert!(w.get(&format!("{stem}.lora_B.weight")).is_some(), "{stem}");
        assert!(w.get(&format!("{stem}.alpha")).is_some(), "{stem}");
    }
    assert_eq!(w.metadata("networkType"), Some("lora"));
    assert_eq!(w.metadata("baseModel"), Some("minimax_h3_image"));
    assert_eq!(w.metadata("trainingMode"), edit.then_some("edit"));
    let up = w
        .get("transformer_blocks.0.attn.to_q.lora_B.weight")
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>();
    assert!(up > 0.0, "the up factor never moved off its zero init");
    downs.len()
}

/// Render `prompt` at `seed` through the registered generator with the saved adapter.
fn render_saved(
    spec: &LoadSpec,
    adapter: &Path,
    prompt: &str,
    seed: u64,
    size: (u32, u32),
    refs: &[Image],
) -> Image {
    let mut spec = spec
        .clone()
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.adapters = vec![AdapterSpec::new(
        adapter.to_path_buf(),
        1.0,
        AdapterKind::Lora,
    )];
    let registry = mlx_gen_minimax_h3::h3_image::register_providers(
        mlx_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    let generator = (registry
        .generators()
        .find(|g| (g.descriptor)().id == "minimax_h3_image")
        .unwrap()
        .load)(&spec)
    .unwrap();
    let req = preview_request(prompt, seed, size, refs, Default::default());
    generator.validate(&req).unwrap();
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut images) => images.remove(0),
        _ => panic!("minimax_h3_image must return images"),
    }
}

/// AC2's reproduction bound between the trainer's sample and the saved adapter's render.
fn check_reproduction(label: &str, sample: &Image, render: &Image) {
    assert_eq!((sample.width, sample.height), (render.width, render.height));
    let n = sample.pixels.len() as f64;
    let mut sum = 0f64;
    let mut outliers = 0usize;
    for (&a, &b) in sample.pixels.iter().zip(&render.pixels) {
        let d = a.abs_diff(b);
        sum += f64::from(d);
        outliers += usize::from(d > SAMPLE_OUTLIER_LEVELS);
    }
    let mean = sum / n;
    let frac = outliers as f64 / n;
    println!("[sw-7097 mlx] {label}: sample vs saved-adapter render mean |Δ| {mean:.4}, outliers {frac:.2e}");
    assert!(mean <= SAMPLE_MEAN_TOL, "{label}: mean |Δ| {mean}");
    assert!(
        frac <= SAMPLE_OUTLIER_FRACTION,
        "{label}: outlier fraction {frac}"
    );
}

/// The step-`step` sample seed of prompt 0 (the trainer loop's mix).
fn sample_seed(seed: u64, step: u32) -> u64 {
    seed.wrapping_add(u64::from(step))
        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
}

#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT, the q4 DiT tier and MINIMAX_H3_SNAPSHOT's text encoder; trains 2048² twice and renders once"]
fn h3_image_train_t2i_saves_reproduces_and_resumes_q4() {
    let spec = q4_spec();
    let data = scratch();
    let items: Vec<TrainingItem> = (0..2u32)
        .map(|i| {
            let path = data.path().join(format!("item{i}.png"));
            draw(&path, 2048, 2048, i);
            TrainingItem::captioned(
                path,
                format!("a red disc on a gradient, sks style, variant {i}"),
            )
        })
        .collect();

    // A: uninterrupted, three steps, the step-2 checkpoint kept.
    let a_dir = data.path().join("a");
    let cfg = config(3);
    let req_a = request(items.clone(), cfg.clone(), &a_dir);
    let (a_path, _, samples) = train(&spec, &req_a);
    let targets = check_layout(&a_path, false);
    let (step, sample) = samples.last().expect("a final-step sample");
    assert_eq!(*step, 3);

    // AC2: the generator loading the saved adapter reproduces the trainer's own sample.
    let render = render_saved(
        &spec,
        &a_path,
        &cfg.sample_prompts[0],
        sample_seed(cfg.seed, 3),
        (2048, 2048),
        &[],
    );
    check_reproduction("t2i", sample, &render);

    // AC2: resume from A's step-2 checkpoint in a fresh directory and take step 3.
    let b_dir = data.path().join("b");
    std::fs::create_dir_all(&b_dir).unwrap();
    for entry in std::fs::read_dir(&a_dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains("-step000002") {
            std::fs::copy(entry.path(), b_dir.join(&name)).unwrap();
        }
    }
    let req_b = request(
        items,
        TrainingConfig {
            resume: true,
            ..cfg
        },
        &b_dir,
    );
    let (b_path, _, _) = train(&spec, &req_b);
    let (wa, wb) = (
        Weights::from_file(&a_path).unwrap(),
        Weights::from_file(&b_path).unwrap(),
    );
    let mut worst = 0f32;
    for key in wa.keys() {
        let d = wa
            .get(key)
            .unwrap()
            .subtract(wb.get(key).expect("same keys"))
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        worst = worst.max(d);
    }
    println!("[sw-7097 mlx] resumed step-3 update vs uninterrupted: max |Δ| {worst:.3e} over {targets} targets");
    assert!(worst <= RESUME_MAX_ABS, "resume diverged: {worst}");

    if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_TRAINED_LORA_OUT") {
        std::fs::create_dir_all(&out).unwrap();
        std::fs::copy(&a_path, Path::new(&out).join("h3_image_lora.safetensors")).unwrap();
        let png = Path::new(&out).join("trainer-sample-step3.png");
        image::save_buffer(
            &png,
            &sample.pixels,
            sample.width,
            sample.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
    mlx_gen_minimax_h3_train_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_train_real::h3_image_train_t2i_saves_reproduces_and_resumes_q4");
}

#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT, the q4 DiT tier and MINIMAX_H3_SNAPSHOT's text encoder; trains a 2048² edit pair and renders once"]
fn h3_image_train_edit_saves_and_reproduces_q4() {
    let spec = q4_spec();
    let data = scratch();
    let target = data.path().join("target.png");
    draw(&target, 2048, 2048, 5);
    let refs: Vec<PathBuf> = (0..2u32)
        .map(|i| {
            let p = data.path().join(format!("ref{i}.png"));
            draw(&p, 1024, 1024, i + 1);
            p
        })
        .collect();
    let items = vec![TrainingItem::edit_pair(
        target,
        "Combine Picture 1 and Picture 2 into one scene".into(),
        refs.clone(),
    )];
    let out = data.path().join("edit");
    let cfg = TrainingConfig {
        save_every: 0,
        ..config(2)
    };
    let req = request(items, cfg.clone(), &out);
    let (path, _, samples) = train(&spec, &req);
    check_layout(&path, true);
    let (step, sample) = samples.last().expect("a final-step sample");
    assert_eq!(*step, 2);
    let references: Vec<Image> = refs.iter().map(|p| decode(p)).collect();
    let render = render_saved(
        &spec,
        &path,
        &cfg.sample_prompts[0],
        sample_seed(cfg.seed, 2),
        (2048, 2048),
        &references,
    );
    check_reproduction("edit", sample, &render);
    mlx_gen_minimax_h3_train_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_train_real::h3_image_train_edit_saves_and_reproduces_q4");
}

fn mlx_gen_minimax_h3_train_device() {
    crate::model_inference_device::record_model_inference_device();
}
