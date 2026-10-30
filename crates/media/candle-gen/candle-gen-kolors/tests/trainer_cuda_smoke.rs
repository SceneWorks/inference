//! Explicit real-CUDA acceptance entry points for Kolors LoRA and LoKr training.
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

fn run(kind: NetworkType) {
    let snapshot =
        PathBuf::from(std::env::var("KOLORS_TRAIN_SNAPSHOT").expect("set KOLORS_TRAIN_SNAPSHOT"));
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([190, 70, 40]))
        .save(&input)
        .unwrap();
    let mut trainer = candle_gen_kolors::provider_registry()
        .unwrap()
        .load_trainer(
            "kolors",
            &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
        )
        .unwrap();
    let request = TrainingRequest {
        items: vec![TrainingItem::captioned(input, "a red tile".into())],
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
    };
    let output = trainer.train(&request, &mut |_| {}).unwrap();
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
        prompt: "a red tile".into(),
        width: 512,
        height: 512,
        steps: Some(1),
        seed: Some(7),
        ..Default::default()
    };
    let render = |spec: LoadSpec| {
        let generator = candle_gen_kolors::provider_registry()
            .unwrap()
            .load("kolors", &spec)
            .unwrap();
        let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
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

#[test]
#[ignore = "needs KOLORS_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn kolors_lora_real_cuda() {
    run(NetworkType::Lora);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::kolors_lora_real_cuda");
}

#[test]
#[ignore = "needs KOLORS_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn kolors_lokr_real_cuda() {
    run(NetworkType::Lokr);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::kolors_lokr_real_cuda");
}

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
fn bounded_run(kind: NetworkType) {
    let snapshot =
        PathBuf::from(std::env::var("KOLORS_TRAIN_SNAPSHOT").expect("set KOLORS_TRAIN_SNAPSHOT"));
    let retained = bounded_output_root();
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let temp = &retained;
    let input = temp.as_path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([190, 70, 40]))
        .save(&input)
        .unwrap();
    let mut trainer = candle_gen_kolors::provider_registry()
        .unwrap()
        .load_trainer(
            "kolors",
            &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
        )
        .unwrap();
    let request = TrainingRequest {
        items: vec![TrainingItem::captioned(input, "a red tile".into())],
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
    };
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &retained.join("metrics.json"),
        &[
            "sdxl-or-kolors:differentiable-LayerNorm",
            "sdxl-or-kolors:frozen-cross-base-KV-storage-hit",
            "sdxl:actual-frozen-projection-pack",
        ],
        || trainer.train(&request, &mut |_| {}).unwrap(),
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
        prompt: "a red tile".into(),
        width: 512,
        height: 512,
        steps: Some(1),
        seed: Some(7),
        ..Default::default()
    };
    let render = |spec: LoadSpec| {
        let generator = candle_gen_kolors::provider_registry()
            .unwrap()
            .load("kolors", &spec)
            .unwrap();
        let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
        else {
            panic!("expected image output")
        };
        assert_eq!(images.len(), 1);
        assert_eq!((images[0].width, images[0].height), (512, 512));
        candle_gen::testkit::save_image_png(&retained.join("adapter_sample.png"), &images[0]);
        images[0].pixels.clone()
    };
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
    assert!(adapted.iter().any(|value| *value != 0));
}
#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn kolors_bounded_acceleration_acceptance() {
    bounded_run(NetworkType::Lora);
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_cuda_smoke::kolors_bounded_acceleration_acceptance");
}
