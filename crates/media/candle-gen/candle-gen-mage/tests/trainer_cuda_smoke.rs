//! Explicit real-CUDA acceptance entry points for Mage adapter and full-base training.
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

fn run(kind: NetworkType, full: bool) {
    let snapshot = PathBuf::from(
        std::env::var("MAGE_BASE_TRAIN_SNAPSHOT").expect("set MAGE_BASE_TRAIN_SNAPSHOT"),
    );
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([210, 160, 30]))
        .save(&input)
        .unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone()));
    let mut trainer = candle_gen_mage::provider_registry()
        .unwrap()
        .load_trainer("mage_flow_base", &spec)
        .unwrap();
    let output_dir = temp.path().join("out");
    let output = trainer
        .train(
            &TrainingRequest {
                items: vec![TrainingItem::captioned(input, "a gold tile".into())],
                config: TrainingConfig {
                    network_type: kind,
                    full_finetune: full,
                    rank: if full { 0 } else { 2 },
                    alpha: 2.0,
                    learning_rate: if full { 1e-6 } else { 1e-3 },
                    train_dtype: if full { "f32" } else { "bf16" }.into(),
                    steps: 1,
                    resolution: 64,
                    save_every: 0,
                    ..Default::default()
                },
                output_dir: output_dir.clone(),
                file_name: "adapter.safetensors".into(),
                trigger_words: vec![],
                cancel: CancelFlag::new(),
            },
            &mut |_| {},
        )
        .unwrap();
    assert!(output.final_loss.is_finite());
    let tensors =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    assert!(!tensors.is_empty());
    if full {
        let config = std::fs::read_to_string(snapshot.join("transformer/config.json")).unwrap();
        let cfg = candle_gen_mage::MageConfig::from_json(&config).unwrap();
        let _reloaded = candle_gen_mage::MageTransformer::load(
            &output_dir,
            &cfg,
            &candle_gen::default_device().unwrap(),
        )
        .expect("full output reloads through production Mage loader");
        let spec = LoadSpec::new(WeightsSource::Dir(output_dir))
            .with_component(
                candle_gen_mage::COMPONENT_TEXT_ENCODER,
                WeightsSource::Dir(snapshot.join("text_encoder")),
            )
            .with_component(
                candle_gen_mage::COMPONENT_VAE,
                WeightsSource::Dir(snapshot.join("vae")),
            );
        let generator = candle_gen_mage::load_finetuned(candle_gen_mage::MageVariant::Base, &spec)
            .expect("trainer output loads through the public fine-tuned generation seam");
        let request = GenerationRequest {
            prompt: "a gold tile".into(),
            width: 512,
            height: 512,
            steps: Some(1),
            seed: Some(7),
            ..Default::default()
        };
        let GenerationOutput::Images(images) = generator
            .generate(&request, &mut |_| {})
            .expect("fine-tuned model renders through the production generator")
        else {
            panic!("expected image output")
        };
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].pixels.len(), 512 * 512 * 3);
    } else {
        let trained_suffix = match kind {
            NetworkType::Lora => ".lora_B.weight",
            NetworkType::Lokr => ".lokr_w1",
        };
        assert!(tensors.iter().any(|(name, tensor)| {
            name.ends_with(trained_suffix)
                && tensor
                    .to_dtype(candle_gen::candle_core::DType::F32)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    > 0.0
        }));
        let request = GenerationRequest {
            prompt: "a gold tile".into(),
            width: 512,
            height: 512,
            steps: Some(1),
            seed: Some(7),
            ..Default::default()
        };
        let render = |spec: LoadSpec| {
            let generator = candle_gen_mage::provider_registry()
                .unwrap()
                .load("mage_flow_base", &spec)
                .unwrap();
            let GenerationOutput::Images(images) =
                generator.generate(&request, &mut |_| {}).unwrap()
            else {
                panic!("expected image output")
            };
            images[0].pixels.clone()
        };
        let base = render(LoadSpec::new(WeightsSource::Dir(snapshot.clone())));
        let runtime_kind = match kind {
            NetworkType::Lora => AdapterKind::Lora,
            NetworkType::Lokr => AdapterKind::Lokr,
        };
        let adapted = render(
            LoadSpec::new(WeightsSource::Dir(snapshot)).with_adapters(vec![AdapterSpec::new(
                output.adapter_path,
                1.0,
                runtime_kind,
            )]),
        );
        assert_ne!(
            base, adapted,
            "production-installed adapter must change output"
        );
    }
}

#[test]
#[ignore = "needs MAGE_BASE_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn mage_base_lora_real_cuda() {
    run(NetworkType::Lora, false);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::mage_base_lora_real_cuda");
}

#[test]
#[ignore = "needs MAGE_BASE_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn mage_base_lokr_real_cuda() {
    run(NetworkType::Lokr, false);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::mage_base_lokr_real_cuda");
}

#[test]
#[ignore = "needs MAGE_BASE_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn mage_base_full_real_cuda() {
    run(NetworkType::Lora, true);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::mage_base_full_real_cuda");
}

#[cfg(feature = "model-inference-proof")]
fn bounded_root() -> std::path::PathBuf {
    let root = std::path::PathBuf::from(
        std::env::var_os("MODEL_PROOF_OUTPUT_DIR").expect("manual executor output directory"),
    )
    .join("bespoke_acceptance");
    std::fs::create_dir_all(&root).unwrap();
    root
}
#[cfg(feature = "model-inference-proof")]
fn bounded_peak(sampler: candle_gen::testkit::PeakSampler, root: &std::path::Path) {
    let peak = sampler.stop();
    assert!(
        peak > 0,
        "selected physical CUDA device peak must be observed"
    );
    std::fs::write(
        root.join("peak_device_bytes.txt"),
        (peak * 1024 * 1024).to_string(),
    )
    .unwrap();
}

/// One source-owned invocation: four TOTAL updates (one warmup, three timed),
/// one config/seed/device, then ONE production-generated saved-adapter sample.
#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn mage_bounded_acceleration_acceptance() {
    let root = bounded_root();
    let snapshot = PathBuf::from(
        std::env::var_os("MAGE_BASE_TRAIN_SNAPSHOT").expect("pinned installed snapshot"),
    );
    let input = root.join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([210, 160, 30]))
        .save(&input)
        .unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone()))
        .with_component(
            candle_gen_mage::COMPONENT_TEXT_ENCODER,
            WeightsSource::Dir(PathBuf::from(
                std::env::var_os("MAGE_TRAIN_TEXT_ENCODER").expect("pinned shared encoder"),
            )),
        )
        .with_component(
            candle_gen_mage::COMPONENT_VAE,
            WeightsSource::Dir(PathBuf::from(
                std::env::var_os("MAGE_TRAIN_VAE").expect("pinned shared VAE"),
            )),
        );
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &root.join("metrics.json"),
        &[
            "mage:geometry-storage-hit",
            "mage:frozen-caption-base-hit-live-residual",
            "mage:frozen-projection-pack",
        ],
        || {
            let mut trainer = candle_gen_mage::provider_registry()
                .unwrap()
                .load_trainer("mage_flow_base", &spec)
                .unwrap();
            let req = TrainingRequest {
                items: vec![TrainingItem::captioned(input, "a gold tile".into())],
                config: TrainingConfig {
                    rank: 2,
                    alpha: 2.,
                    learning_rate: 1e-3,
                    steps: 4,
                    resolution: 64,
                    batch_size: 1,
                    gradient_accumulation: 1,
                    lr_warmup_steps: 0,
                    save_every: 0,
                    seed: 7,
                    train_dtype: "bf16".into(),
                    gradient_checkpointing: false,
                    ..Default::default()
                },
                output_dir: root.join("out"),
                file_name: "adapter.safetensors".into(),
                trigger_words: vec![],
                cancel: CancelFlag::new(),
            };
            let output = trainer.train(&req, &mut |_| {}).unwrap();
            assert_eq!(output.steps, 4);
            assert!(output.final_loss.is_finite());
            output
        },
    );
    bounded_peak(sampler, &root);
    let reopened =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    assert!(!reopened.is_empty());
    drop(reopened);
    let generator = candle_gen_mage::provider_registry()
        .unwrap()
        .load(
            "mage_flow_base",
            &spec.with_adapters(vec![AdapterSpec::new(
                output.adapter_path,
                1.,
                AdapterKind::Lora,
            )]),
        )
        .unwrap();
    let GenerationOutput::Images(images) = generator
        .generate(
            &GenerationRequest {
                prompt: "a gold tile".into(),
                width: 512,
                height: 512,
                steps: Some(1),
                seed: Some(7),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap()
    else {
        panic!("expected image");
    };
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].pixels.len(), 512 * 512 * 3);
    assert!(images[0].pixels.iter().any(|p| *p != 0));
    candle_gen::testkit::save_image_png(&root.join("adapter_sample.png"), &images[0]);
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::mage_bounded_acceleration_acceptance"
    );
}
