//! Real-CUDA acceptance entry points for the two newly admitted single-DiT Wan trainers.
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image,
    LoadSpec, MoeExpert, NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

fn run(id: &str, env: &str) {
    let snapshot = PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env}")));
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([120, 90, 220]))
        .save(&input)
        .unwrap();
    let mut trainer = candle_gen_wan::provider_registry()
        .unwrap()
        .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
        .unwrap();
    let output = trainer
        .train(
            &TrainingRequest {
                items: vec![TrainingItem::captioned(input, "a violet tile".into())],
                config: TrainingConfig {
                    network_type: NetworkType::Lora,
                    rank: 2,
                    alpha: 2.0,
                    learning_rate: 1e-2,
                    gradient_checkpointing: true,
                    // A14B alternates high/low experts, so two steps are the minimum truthful smoke.
                    steps: if id == "wan2_2_i2v_14b" { 2 } else { 1 },
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
    let high = output.adapter_path;
    let low = high.with_file_name("adapter.low_noise.safetensors");
    let paths = if id == "wan2_2_i2v_14b" {
        vec![high.clone(), low.clone()]
    } else {
        vec![high.clone()]
    };
    for path in &paths {
        let tensors = candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
        assert!(!tensors.is_empty(), "{} is empty", path.display());
        assert!(tensors.iter().any(|(name, tensor)| {
            name.ends_with(".lora_B.weight")
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
    }

    let conditioning = (id == "wan2_2_i2v_14b")
        .then(|| Conditioning::Reference {
            image: Image {
                width: 256,
                height: 256,
                pixels: vec![128; 256 * 256 * 3],
            },
            strength: None,
        })
        .into_iter()
        .collect();
    let request = GenerationRequest {
        prompt: "a violet tile".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(2),
        seed: Some(7),
        conditioning,
        ..Default::default()
    };
    let render = |spec: LoadSpec| {
        let generator = candle_gen_wan::provider_registry()
            .unwrap()
            .load(id, &spec)
            .unwrap();
        let GenerationOutput::Video { frames, .. } =
            generator.generate(&request, &mut |_| {}).unwrap()
        else {
            panic!("expected video output")
        };
        frames[0].pixels.clone()
    };
    let base = render(LoadSpec::new(WeightsSource::Dir(snapshot.clone())));
    let adapters = if id == "wan2_2_i2v_14b" {
        vec![
            AdapterSpec::new(high, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
            AdapterSpec::new(low, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
        ]
    } else {
        vec![AdapterSpec::new(high, 1.0, AdapterKind::Lora)]
    };
    let adapted = render(LoadSpec::new(WeightsSource::Dir(snapshot)).with_adapters(adapters));
    assert_ne!(
        base, adapted,
        "production-installed adapter must change output"
    );
}

#[test]
#[ignore = "needs WAN_I2V_14B_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn wan_i2v_14b_lora_real_cuda() {
    run("wan2_2_i2v_14b", "WAN_I2V_14B_TRAIN_SNAPSHOT");
}

#[test]
#[ignore = "needs WAN_TI2V_5B_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn wan_ti2v_5b_lora_real_cuda() {
    run("wan2_2_ti2v_5b", "WAN_TI2V_5B_TRAIN_SNAPSHOT");
}
