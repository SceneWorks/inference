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
/// The first excluded warmup checks cached/fallback prediction, loss and all
/// optimizer-owned gradients too, with zero reference optimizer updates.
#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn mage_bounded_acceleration_acceptance() {
    let root = bounded_root();
    let comparison = candle_gen::train::comparison::requested_stage().is_some();
    let cadence = candle_gen::train::comparison::ComparisonCadence::Ordinary;
    let steps = if comparison { cadence.updates() } else { 4 };
    // One source-owned representative workload, never a resolution grid. The
    // controller must admit the worst legacy arm before executing any stage.
    let resolution = if comparison { 512 } else { 64 };
    let snapshot = PathBuf::from(
        std::env::var_os("MAGE_BASE_TRAIN_SNAPSHOT").expect("pinned installed snapshot"),
    );
    let input = root.join("input.png");
    image::RgbImage::from_pixel(
        resolution as u32,
        resolution as u32,
        image::Rgb([210, 160, 30]),
    )
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
    let config = TrainingConfig {
        rank: 2,
        alpha: 2.,
        learning_rate: 1e-3,
        steps,
        resolution,
        batch_size: 1,
        gradient_accumulation: 1,
        lr_warmup_steps: 0,
        save_every: 0,
        seed: 7,
        train_dtype: "bf16".into(),
        gradient_checkpointing: false,
        ..Default::default()
    };
    let items = vec![TrainingItem::captioned(input, "a gold tile".into())];
    let sampler = (!comparison).then(candle_gen::testkit::PeakSampler::start_rendered);
    // A new child owns a fresh native trainer/Run, initial parameters, native
    // optimizer, EMA/data cursors and (restoration) critic/SN state. No quality
    // render or adapter reload is repeated by this timing-only supplement.
    let run = || {
        if comparison {
            comparison_initialization(&config, &items, &spec);
        }
        let mut trainer = candle_gen_mage::provider_registry()
            .unwrap()
            .load_trainer("mage_flow_base", &spec)
            .unwrap();
        let req = TrainingRequest {
            items,
            config,
            output_dir: root.join("out"),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        let output = trainer.train(&req, &mut |_| {}).unwrap();
        assert_eq!(output.steps, steps);
        assert!(output.final_loss.is_finite());
        output
    };
    let output = if comparison {
        candle_gen::train::comparison::with_stage(
            &root.join("comparison_metrics.json"),
            "mage-adapter",
            cadence,
            run,
        )
    } else {
        candle_gen::train::profile::with_bounded_cohort_for(
            &root.join("metrics.json"),
            &[
                "mage:geometry-storage-hit",
                "mage:frozen-caption-base-hit-live-residual",
                "mage:frozen-projection-pack",
            ],
            run,
        )
    };
    if comparison {
        println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::mage_bounded_acceleration_acceptance");
        return;
    }
    bounded_peak(sampler.expect("ordinary fixture sampler"), &root);
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

#[cfg(feature = "model-inference-proof")]
fn comparison_input_sha256(path: &std::path::Path) -> String {
    use sha2::Digest;
    format!(
        "{:x}",
        sha2::Sha256::digest(std::fs::read(path).expect("comparison input"))
    )
}

#[cfg(feature = "model-inference-proof")]
fn comparison_initialization(
    config: &candle_gen::gen_core::TrainingConfig,
    items: &[candle_gen::gen_core::TrainingItem],
    spec: &candle_gen::gen_core::LoadSpec,
) {
    let data = items.iter().map(|item| {
        let sidecars = item.model_options.iter().map(|(key,value)| {
            let value = if let Some(path) = value.as_str().filter(|path| std::path::Path::new(path).is_file()) {
                serde_json::json!({"file_sha256":comparison_input_sha256(std::path::Path::new(path))})
            } else { value.clone() };
            (key.clone(),value)
        }).collect::<serde_json::Map<String,serde_json::Value>>();
        serde_json::json!({"image_sha256":comparison_input_sha256(&item.image_path),
            "caption":item.caption,"sidecars":sidecars,
            "references":item.reference_image_paths.iter().map(|path|comparison_input_sha256(path)).collect::<Vec<_>>(),
            "control":item.control_image_path.as_deref().map(comparison_input_sha256),
            "subject_mask":item.subject_mask_path.as_deref().map(comparison_input_sha256)})
    }).collect::<Vec<_>>();
    candle_gen::train::comparison::record_initialization(&serde_json::json!({
        "training_config":format!("{config:?}"),"source_spec":format!("{spec:?}"),
        "seed":config.seed,"resolution":config.resolution,"batch_size":config.batch_size,
        "gradient_accumulation":config.gradient_accumulation,"data":data,
        "factory":"fresh native trainer in one comparison child; no resumed mutable run",
        "claim_limit":"content/configuration witnesses; not a claim of whole checkpoint bytes or tensor equality from source flags"
    }));
}
