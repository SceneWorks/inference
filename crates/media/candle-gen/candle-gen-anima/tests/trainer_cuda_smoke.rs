//! Explicit real-CUDA acceptance entry points for the new Anima base trainer. These tests perform a
//! functional one-step run and verify a finite loss plus a non-empty reloadable adapter artifact.
use std::path::{Path, PathBuf};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

fn image(path: &Path) {
    image::RgbImage::from_pixel(64, 64, image::Rgb([80, 140, 210]))
        .save(path)
        .unwrap();
}

fn run(kind: NetworkType) {
    let snapshot =
        PathBuf::from(std::env::var("ANIMA_TRAIN_SNAPSHOT").expect("set ANIMA_TRAIN_SNAPSHOT"));
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.png");
    image(&input);
    let mut trainer = candle_gen_anima::provider_registry()
        .unwrap()
        .load_trainer(
            "anima_base",
            &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
        )
        .unwrap();
    let request = TrainingRequest {
        items: vec![TrainingItem::captioned(input, "a blue tile".into())],
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
        prompt: "a blue tile".into(),
        width: 512,
        height: 512,
        steps: Some(1),
        seed: Some(7),
        ..Default::default()
    };
    let render = |spec: LoadSpec| {
        let generator = candle_gen_anima::provider_registry()
            .unwrap()
            .load("anima_base", &spec)
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
#[ignore = "needs ANIMA_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn anima_base_lora_real_cuda() {
    run(NetworkType::Lora);
}

#[test]
#[ignore = "needs ANIMA_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn anima_base_lokr_real_cuda() {
    run(NetworkType::Lokr);
}
