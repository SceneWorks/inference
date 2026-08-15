//! Explicit real-CUDA acceptance entry points for both SD3.5 training bases and adapter kinds.
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

fn run(id: &str, env: &str, kind: NetworkType) {
    let snapshot = PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env}")));
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

macro_rules! smoke {
    ($name:ident, $id:literal, $env:literal, $kind:expr) => {
        #[test]
        #[ignore = "needs real SD3.5 weights and explicitly scheduled CUDA"]
        fn $name() {
            run($id, $env, $kind);
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
