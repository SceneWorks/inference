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
}

#[test]
#[ignore = "needs MAGE_BASE_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn mage_base_lokr_real_cuda() {
    run(NetworkType::Lokr, false);
}

#[test]
#[ignore = "needs MAGE_BASE_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn mage_base_full_real_cuda() {
    run(NetworkType::Lora, true);
}
