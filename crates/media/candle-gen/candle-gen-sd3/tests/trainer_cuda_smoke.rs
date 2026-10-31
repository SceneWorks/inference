//! Explicit real-CUDA acceptance entry points for both SD3.5 training bases and adapter kinds.
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

/// Training accepts upstream diffusers snapshots; generation admits only the exact shipped
/// SceneWorks turnkey. Keep their source identities separate rather than relabelling the trainer
/// snapshot as a production artifact. The production receipt checks the route, revision, tier,
/// inventory and physical headers before any optimizer update can be consumed.
fn sample_load_spec(id: &str) -> LoadSpec {
    let (route, sample_env) = match id {
        "sd3_5_medium" => (
            candle_gen_sd3::memory_strategy::Sd35Route::Medium,
            "SD35_MEDIUM_SAMPLE_SNAPSHOT",
        ),
        "sd3_5_large" => (
            candle_gen_sd3::memory_strategy::Sd35Route::Large,
            "SD35_LARGE_SAMPLE_SNAPSHOT",
        ),
        other => panic!("unsupported SD3.5 training sample route {other}"),
    };
    let snapshot =
        PathBuf::from(std::env::var_os(sample_env).unwrap_or_else(|| panic!("set {sample_env}")));
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot)).with_resolved_route(route.provider_id());
    candle_gen_sd3::memory_strategy::Sd35LoadReceipt::capture(route, &spec)
        .expect("sample must satisfy the production route contract before training");
    spec
}

fn run(id: &str, env: &str, kind: NetworkType) {
    let snapshot = PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env}")));
    let sample_spec = sample_load_spec(id);
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([60, 180, 90]))
        .save(&input)
        .unwrap();
    let mut trainer = candle_gen_sd3::provider_registry()
        .unwrap()
        .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
        .unwrap();
    let output = trainer
        .train(
            &TrainingRequest {
                items: vec![TrainingItem::captioned(input, "a green tile".into())],
                config: TrainingConfig {
                    network_type: kind,
                    rank: 2,
                    alpha: 2.0,
                    learning_rate: 1e-2,
                    steps: 1,
                    resolution: 64,
                    save_every: 0,
                    ..Default::default()
                },
                output_dir: temp.path().join("out"),
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
        prompt: "a green tile".into(),
        width: 256,
        height: 256,
        steps: Some(1),
        seed: Some(7),
        ..Default::default()
    };
    let render = |spec: LoadSpec| {
        let generator = candle_gen_sd3::provider_registry()
            .unwrap()
            .load(id, &spec)
            .unwrap();
        let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
        else {
            panic!("expected image output")
        };
        images[0].pixels.clone()
    };
    let base = render(sample_spec.clone());
    let runtime_kind = match kind {
        NetworkType::Lora => AdapterKind::Lora,
        NetworkType::Lokr => AdapterKind::Lokr,
    };
    let adapted = render(sample_spec.with_adapters(vec![AdapterSpec::new(
        output.adapter_path,
        1.0,
        runtime_kind,
    )]));
    assert_ne!(
        base, adapted,
        "production-installed adapter must change output"
    );
}

macro_rules! smoke {
    ($name:ident, $id:literal, $env:literal, $kind:expr) => {
        #[test]
        #[ignore = "needs real SD3.5 weights and explicitly scheduled CUDA"]
        fn $name() {
            run($id, $env, $kind);
                    #[cfg(feature = "model-inference-proof")]
            match stringify!($name) {
                "sd35_large_lora_real_cuda" => println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::sd35_large_lora_real_cuda"),
                "sd35_large_lokr_real_cuda" => println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::sd35_large_lokr_real_cuda"),
                "sd35_medium_lora_real_cuda" => println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::sd35_medium_lora_real_cuda"),
                "sd35_medium_lokr_real_cuda" => println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::sd35_medium_lokr_real_cuda"),
                _ => panic!("undeclared generated model-proof case"),
            };
}
    };
}

smoke!(
    sd35_large_lora_real_cuda,
    "sd3_5_large",
    "SD35_LARGE_TRAIN_SNAPSHOT",
    NetworkType::Lora
);
smoke!(
    sd35_large_lokr_real_cuda,
    "sd3_5_large",
    "SD35_LARGE_TRAIN_SNAPSHOT",
    NetworkType::Lokr
);
smoke!(
    sd35_medium_lora_real_cuda,
    "sd3_5_medium",
    "SD35_MEDIUM_TRAIN_SNAPSHOT",
    NetworkType::Lora
);
smoke!(
    sd35_medium_lokr_real_cuda,
    "sd3_5_medium",
    "SD35_MEDIUM_TRAIN_SNAPSHOT",
    NetworkType::Lokr
);

#[cfg(feature = "model-inference-proof")]
fn bounded_output_root() -> PathBuf {
    let root = PathBuf::from(
        std::env::var_os("MODEL_PROOF_OUTPUT_DIR")
            .expect("manual model-proof executor must own the retained output directory"),
    );
    let path = root.join("cohort_acceptance");
    std::fs::create_dir_all(&path).unwrap();
    path
}
#[cfg(feature = "model-inference-proof")]
fn bounded_peak(sampler: candle_gen::testkit::PeakSampler, dir: &std::path::Path) {
    let peak = sampler.stop();
    assert!(peak > 0, "must observe selected physical CUDA device");
    std::fs::write(
        dir.join("peak_device_bytes.txt"),
        (peak * 1024 * 1024).to_string(),
    )
    .unwrap();
}

#[cfg(feature = "model-inference-proof")]
fn bounded_run(id: &str, env: &str, kind: NetworkType) {
    if let Some(stage) = candle_gen::train::comparison::requested_stage() {
        use candle_gen::train::comparison::{self, ComparisonCadence, ComparisonStage};
        assert_ne!(
            stage,
            ComparisonStage::MinusReferenceOwner,
            "this trainer uses the composable reference attention"
        );
        assert_eq!(kind, NetworkType::Lora);
        let root = bounded_output_root();
        comparison::with_stage(
            &root.join("comparison_metrics.json"),
            id,
            ComparisonCadence::Ordinary,
            || {
                let snapshot =
                    PathBuf::from(std::env::var(env).expect("existing installed model snapshot"));
                let input = root.join("comparison_input.png");
                image::RgbImage::from_pixel(64, 64, image::Rgb([60, 180, 90]))
                    .save(&input)
                    .unwrap();
                let mut trainer = candle_gen_sd3::provider_registry()
                    .unwrap()
                    .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot)))
                    .unwrap();
                let output = trainer
                    .train(
                        &TrainingRequest {
                            items: vec![TrainingItem::captioned(input, "a green tile".into())],
                            config: TrainingConfig {
                                network_type: NetworkType::Lora,
                                rank: 2,
                                alpha: 2.,
                                learning_rate: 1e-2,
                                steps: 4,
                                seed: 7,
                                train_dtype: "bf16".into(),
                                resolution: 512,
                                batch_size: 1,
                                gradient_accumulation: 1,
                                lr_warmup_steps: 0,
                                save_every: 0,
                                sample_every: 0,
                                resume: false,
                                gradient_checkpointing: false,
                                ..Default::default()
                            },
                            output_dir: root.join("comparison_out"),
                            file_name: "comparison.safetensors".into(),
                            trigger_words: vec![],
                            cancel: CancelFlag::new(),
                        },
                        &mut |_| {},
                    )
                    .expect("four fresh comparison updates");
                assert_eq!(output.steps, 4);
                assert!(output.final_loss.is_finite());
                assert!(output.adapter_path.is_file());
            },
        );
        return;
    }

    let snapshot = PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env}")));
    let sample_spec = sample_load_spec(id);
    let retained = bounded_output_root();
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let temp = &retained;
    let input = temp.as_path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([60, 180, 90]))
        .save(&input)
        .unwrap();
    let mut trainer = candle_gen_sd3::provider_registry()
        .unwrap()
        .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
        .unwrap();
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &retained.join("metrics.json"),
        &[
            "sd3:differentiable-QK-RMS",
            "sd3:frozen-context-storage-hit",
            "sd3:actual-frozen-QKV-pack",
        ],
        || {
            trainer
                .train(
                    &TrainingRequest {
                        items: vec![TrainingItem::captioned(input, "a green tile".into())],
                        config: TrainingConfig {
                            network_type: kind,
                            rank: 2,
                            alpha: 2.0,
                            learning_rate: 1e-2,
                            steps: 4,
                            seed: 7,
                            train_dtype: "bf16".into(),
                            resolution: 64,
                            save_every: 0,
                            ..Default::default()
                        },
                        output_dir: temp.as_path().join("out"),
                        file_name: "adapter.safetensors".into(),
                        trigger_words: vec![],
                        cancel: CancelFlag::new(),
                    },
                    &mut |_| {},
                )
                .unwrap()
        },
    );
    bounded_peak(sampler, &retained);
    assert_eq!(output.steps, 4);
    assert!(output.final_loss.is_finite());
    let tensors =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    assert!(!tensors.is_empty());
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
        prompt: "a green tile".into(),
        width: 256,
        height: 256,
        steps: Some(1),
        seed: Some(7),
        ..Default::default()
    };
    let render = |spec: LoadSpec| {
        let generator = candle_gen_sd3::provider_registry()
            .unwrap()
            .load(id, &spec)
            .unwrap();
        let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
        else {
            panic!("expected image output")
        };
        assert_eq!(images.len(), 1);
        assert_eq!((images[0].width, images[0].height), (256, 256));
        candle_gen::testkit::save_image_png(&retained.join("adapter_sample.png"), &images[0]);
        images[0].pixels.clone()
    };
    let runtime_kind = match kind {
        NetworkType::Lora => AdapterKind::Lora,
        NetworkType::Lokr => AdapterKind::Lokr,
    };
    let adapted = render(sample_spec.with_adapters(vec![AdapterSpec::new(
        output.adapter_path,
        1.0,
        runtime_kind,
    )]));
    assert!(adapted.iter().any(|value| *value != 0));
}
#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn sd35_medium_bounded_acceleration_acceptance() {
    bounded_run(
        "sd3_5_medium",
        "SD35_MEDIUM_TRAIN_SNAPSHOT",
        NetworkType::Lora,
    );
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::sd35_medium_bounded_acceleration_acceptance");
}
