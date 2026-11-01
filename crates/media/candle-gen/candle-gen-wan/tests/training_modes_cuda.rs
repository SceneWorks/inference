//! Real-CUDA acceptance entry points for the two newly admitted single-DiT Wan trainers.
use std::path::PathBuf;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image,
    LoadSpec, MoeExpert, NetworkType, TrainingConfig, TrainingItem, TrainingRequest, WeightsSource,
};

fn run(id: &str, env: &str) {
    // TI2V-5B's descriptor floor is 480 px per side. Use that shared smoke geometry for both
    // production reloads so the request is valid on the stricter provider, and keep the I2V
    // reference exactly aligned with it. One frame is the provider's minimum valid 1 mod 4 lattice;
    // two denoise steps are required here so the dual-expert I2V smoke exercises high and low.
    const GENERATION_EDGE: u32 = 480;
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
                width: GENERATION_EDGE,
                height: GENERATION_EDGE,
                pixels: vec![128; GENERATION_EDGE as usize * GENERATION_EDGE as usize * 3],
            },
            strength: None,
        })
        .into_iter()
        .collect();
    let request = GenerationRequest {
        prompt: "a violet tile".into(),
        width: GENERATION_EDGE,
        height: GENERATION_EDGE,
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training_modes_cuda::wan_i2v_14b_lora_real_cuda");
}

#[test]
#[ignore = "needs WAN_TI2V_5B_TRAIN_SNAPSHOT and explicitly scheduled CUDA"]
fn wan_ti2v_5b_lora_real_cuda() {
    run("wan2_2_ti2v_5b", "WAN_TI2V_5B_TRAIN_SNAPSHOT");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training_modes_cuda::wan_ti2v_5b_lora_real_cuda");
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
fn bounded_run(id: &str, env: &str) {
    if let Some(stage) = candle_gen::train::comparison::requested_stage() {
        use candle_gen::train::comparison::{self, ComparisonCadence, ComparisonStage};
        assert_ne!(
            stage,
            ComparisonStage::MinusReferenceOwner,
            "this trainer uses the composable reference attention"
        );
        assert_eq!(id, "wan2_2_ti2v_5b");
        let root = bounded_output_root();
        comparison::with_stage(
            &root.join("comparison_metrics.json"),
            id,
            ComparisonCadence::Ordinary,
            || {
                let snapshot =
                    PathBuf::from(std::env::var(env).expect("existing installed model snapshot"));
                let input = root.join("comparison_input.png");
                image::RgbImage::from_pixel(64, 64, image::Rgb([120, 90, 220]))
                    .save(&input)
                    .unwrap();
                let mut trainer = candle_gen_wan::provider_registry()
                    .unwrap()
                    .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot)))
                    .unwrap();
                let output = trainer
                    .train(
                        &TrainingRequest {
                            items: vec![TrainingItem::captioned(input, "a violet tile".into())],
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
                                gradient_checkpointing: true,
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

    // TI2V-5B's descriptor floor is 480 px per side. Use that shared smoke geometry for both
    // production reloads so the request is valid on the stricter provider, and keep the I2V
    // reference exactly aligned with it. One frame is the provider's minimum valid 1 mod 4 lattice;
    // two denoise steps are required here so the dual-expert I2V smoke exercises high and low.
    const GENERATION_EDGE: u32 = 480;
    let snapshot = PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env}")));
    let retained = bounded_output_root();
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let temp = &retained;
    let input = temp.as_path().join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([120, 90, 220]))
        .save(&input)
        .unwrap();
    let mut trainer = candle_gen_wan::provider_registry()
        .unwrap()
        .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
        .unwrap();
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &retained.join("metrics.json"),
        &[
            "wan:shared-differentiable-RMS",
            "wan:expert-owned-caption-storage-hit",
            "wan:actual-frozen-projection-pack",
        ],
        || {
            trainer
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
                width: GENERATION_EDGE,
                height: GENERATION_EDGE,
                pixels: vec![128; GENERATION_EDGE as usize * GENERATION_EDGE as usize * 3],
            },
            strength: None,
        })
        .into_iter()
        .collect();
    let request = GenerationRequest {
        prompt: "a violet tile".into(),
        width: GENERATION_EDGE,
        height: GENERATION_EDGE,
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
        assert_eq!(frames.len(), 1);
        assert_eq!(
            (frames[0].width, frames[0].height),
            (GENERATION_EDGE, GENERATION_EDGE)
        );
        candle_gen::testkit::save_image_png(&retained.join("adapter_sample.png"), &frames[0]);
        frames[0].pixels.clone()
    };
    let adapters = if id == "wan2_2_i2v_14b" {
        vec![
            AdapterSpec::new(high, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
            AdapterSpec::new(low, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
        ]
    } else {
        vec![AdapterSpec::new(high, 1.0, AdapterKind::Lora)]
    };
    let adapted = render(LoadSpec::new(WeightsSource::Dir(snapshot)).with_adapters(adapters));
    assert!(adapted.iter().any(|value| *value != 0));
}
#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn wan_ti2v_5b_bounded_acceleration_acceptance() {
    bounded_run("wan2_2_ti2v_5b", "WAN_TI2V_5B_TRAIN_SNAPSHOT");
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training_modes_cuda::wan_ti2v_5b_bounded_acceleration_acceptance");
}
