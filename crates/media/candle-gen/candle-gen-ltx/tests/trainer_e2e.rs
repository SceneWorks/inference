//! Real-weight Candle LTX-2.3 trainer round-trip (sc-13868).
//!
//! The ignored CUDA test uses the explicit provider registry for the complete
//! prepare -> cache -> train -> save -> inference-load -> render path. It trains one f32,
//! gradient-checkpointed LoRA step against the packed q4 tier, verifies the saved PEFT factors are
//! non-zero, then renders the distilled generator with and without the adapter at the same seed and
//! proves the adapter changes inference output.
//!
//! ```text
//! set LTX_TRAINING_TIER=E:\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\<hash>\q4
//! set LTX_GEMMA_DIR=E:\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\<hash>\gemma
//! cargo test -p candle-gen-ltx --features cuda --release --test integration trainer_e2e:: -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};

fn required_dir(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var(name)
            .unwrap_or_else(|_| panic!("set {name} to the required snapshot directory")),
    );
    assert!(
        path.is_dir(),
        "{name} is not a directory: {}",
        path.display()
    );
    path
}

fn load_spec(adapters: Vec<AdapterSpec>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(required_dir("LTX_TRAINING_TIER")))
        .with_adapters(adapters);
    spec.text_encoder = Some(WeightsSource::Dir(required_dir("LTX_GEMMA_DIR")));
    spec
}

fn make_item(dir: &Path) -> TrainingItem {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut image = image::RgbImage::new(64, 64);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgb([190, 45, 35]);
    }
    let image_path = dir.join("red_swatch.png");
    image.save(&image_path).unwrap();
    TrainingItem::captioned(image_path, "a solid red colour swatch".to_string())
}

fn render(adapter: Option<&Path>) -> Vec<u8> {
    let adapters = adapter
        .map(|path| {
            vec![AdapterSpec::new(
                path.to_path_buf(),
                16.0,
                AdapterKind::Lora,
            )]
        })
        .unwrap_or_default();
    let registry = candle_gen_ltx::provider_registry().expect("LTX registry");
    let generator = registry
        .load(candle_gen_ltx::config::MODEL_ID, &load_spec(adapters))
        .expect("load distilled LTX generator");
    let request = GenerationRequest {
        prompt: "a solid red colour swatch".to_string(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(candle_gen_ltx::config::NATIVE_STEPS),
        seed: Some(11),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("render LTX frame");
    let GenerationOutput::Video { frames, .. } = output else {
        panic!("expected LTX video output");
    };
    assert_eq!(frames.len(), 1);
    assert_eq!((frames[0].width, frames[0].height), (256, 256));
    assert_eq!(frames[0].pixels.len(), 256 * 256 * 3);
    assert!(
        frames[0].pixels.iter().any(|&value| value != 0),
        "rendered frame must not be all black"
    );
    frames.into_iter().next().unwrap().pixels
}

#[test]
#[ignore = "needs SceneWorks/ltx-2.3-mlx q4 + Gemma weights and a CUDA GPU"]
fn ltx_trainer_round_trips_through_inference_and_changes_render() {
    assert_eq!(candle_gen_ltx::config::TRAINER_ID, "ltx_2_3");
    assert_eq!(candle_gen_ltx::config::MODEL_ID, "ltx_2_3_distilled");

    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let item = make_item(&tmp.join("data"));
    let output_dir = tmp.join("out");
    let mut trainer = candle_gen_ltx::provider_registry()
        .unwrap()
        .load_trainer(candle_gen_ltx::config::TRAINER_ID, &load_spec(Vec::new()))
        .expect("registered LTX trainer");
    let request = TrainingRequest {
        items: vec![item],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-3,
            steps: 1,
            resolution: 64,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            train_dtype: "f32".to_string(),
            gradient_checkpointing: true,
            ..Default::default()
        },
        output_dir,
        file_name: "ltx_one_step.safetensors".to_string(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::new(),
    };

    let mut cached = Vec::new();
    let mut trained = Vec::new();
    let output = trainer
        .train(&request, &mut |progress| match progress {
            TrainingProgress::Caching { current, total } => cached.push((current, total)),
            TrainingProgress::Training {
                step, total, loss, ..
            } => trained.push((step, total, loss)),
            _ => {}
        })
        .expect("one-step LTX training");
    assert_eq!(cached, [(1, 1)]);
    assert_eq!(trained.len(), 1);
    assert_eq!((trained[0].0, trained[0].1), (1, 1));
    assert!(trained[0].2.is_finite());
    assert_eq!(output.steps, 1);
    assert!(output.adapter_path.is_file());

    let tensors =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    let a_count = tensors
        .keys()
        .filter(|key| key.ends_with(".lora_A.weight"))
        .count();
    let b_factors: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.ends_with(".lora_B.weight"))
        .collect();
    assert!(a_count > 0);
    assert_eq!(a_count, b_factors.len());
    assert!(
        b_factors.iter().any(|(_, tensor)| {
            tensor
                .to_dtype(DType::F32)
                .and_then(|tensor| tensor.abs())
                .and_then(|tensor| tensor.max_all())
                .and_then(|tensor| tensor.to_scalar::<f32>())
                .is_ok_and(|value| value > 0.0)
        }),
        "one optimizer step must update at least one LoRA B factor"
    );

    let base = render(None);
    let adapted = render(Some(&output.adapter_path));
    assert_ne!(
        base, adapted,
        "the trained adapter must have an observable inference effect"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::ltx_trainer_round_trips_through_inference_and_changes_render");
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
fn bounded_peak(sampler: candle_gen::testkit::PeakSampler, dir: &Path) {
    let peak = sampler.stop();
    assert!(peak > 0, "must observe selected physical CUDA device");
    std::fs::write(
        dir.join("peak_device_bytes.txt"),
        (peak * 1024 * 1024).to_string(),
    )
    .unwrap();
}
#[cfg(feature = "model-inference-proof")]
fn bounded_render(adapter: Option<&Path>) -> Vec<u8> {
    let adapters = adapter
        .map(|path| {
            vec![AdapterSpec::new(
                path.to_path_buf(),
                16.0,
                AdapterKind::Lora,
            )]
        })
        .unwrap_or_default();
    let registry = candle_gen_ltx::provider_registry().expect("LTX registry");
    let generator = registry
        .load(candle_gen_ltx::config::MODEL_ID, &load_spec(adapters))
        .expect("load distilled LTX generator");
    let request = GenerationRequest {
        prompt: "a solid red colour swatch".to_string(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(candle_gen_ltx::config::NATIVE_STEPS),
        seed: Some(7),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("render LTX frame");
    let GenerationOutput::Video { frames, .. } = output else {
        panic!("expected LTX video output");
    };
    assert_eq!(frames.len(), 1);
    assert_eq!((frames[0].width, frames[0].height), (256, 256));
    assert_eq!(frames[0].pixels.len(), 256 * 256 * 3);
    assert!(
        frames[0].pixels.iter().any(|&value| value != 0),
        "rendered frame must not be all black"
    );
    candle_gen::testkit::save_image_png(
        &bounded_output_root().join("adapter_sample.png"),
        &frames[0],
    );
    frames.into_iter().next().unwrap().pixels
}
#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn ltx23_bounded_acceleration_acceptance() {
    assert_eq!(candle_gen_ltx::config::TRAINER_ID, "ltx_2_3");
    assert_eq!(candle_gen_ltx::config::MODEL_ID, "ltx_2_3_distilled");

    let tmp = bounded_output_root();
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let item = make_item(&tmp.join("data"));
    let output_dir = tmp.join("out");
    let mut trainer = candle_gen_ltx::provider_registry()
        .unwrap()
        .load_trainer(candle_gen_ltx::config::TRAINER_ID, &load_spec(Vec::new()))
        .expect("registered LTX trainer");
    let request = TrainingRequest {
        items: vec![item],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-3,
            steps: 4,
            resolution: 64,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            train_dtype: "f32".to_string(),
            gradient_checkpointing: true,
            ..Default::default()
        },
        output_dir,
        file_name: "ltx23_bounded.safetensors".to_string(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::new(),
    };

    let mut cached = Vec::new();
    let mut trained = Vec::new();
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &tmp.join("metrics.json"),
        &[
            "ltx:shared-differentiable-RMS",
            "ltx23:rope-storage-hit",
            "ltx23:native-incompatible-projection-fallback",
        ],
        || {
            trainer
                .train(&request, &mut |progress| match progress {
                    TrainingProgress::Caching { current, total } => cached.push((current, total)),
                    TrainingProgress::Training {
                        step, total, loss, ..
                    } => trained.push((step, total, loss)),
                    _ => {}
                })
                .expect("bounded LTX training")
        },
    );
    bounded_peak(sampler, &tmp);
    assert_eq!(cached, [(1, 1)]);
    assert_eq!(trained.len(), 4);
    assert_eq!((trained[3].0, trained[3].1), (4, 4));
    assert!(trained.iter().all(|row| row.2.is_finite()));
    assert_eq!(output.steps, 4);
    assert!(output.adapter_path.is_file());

    let tensors =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    let a_count = tensors
        .keys()
        .filter(|key| key.ends_with(".lora_A.weight"))
        .count();
    let b_factors: Vec<_> = tensors
        .iter()
        .filter(|(key, _)| key.ends_with(".lora_B.weight"))
        .collect();
    assert!(a_count > 0);
    assert_eq!(a_count, b_factors.len());
    assert!(
        b_factors.iter().any(|(_, tensor)| {
            tensor
                .to_dtype(DType::F32)
                .and_then(|tensor| tensor.abs())
                .and_then(|tensor| tensor.max_all())
                .and_then(|tensor| tensor.to_scalar::<f32>())
                .is_ok_and(|value| value > 0.0)
        }),
        "one optimizer step must update at least one LoRA B factor"
    );

    bounded_render(Some(&output.adapter_path));

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::ltx23_bounded_acceleration_acceptance");
}

/// The target data is an existing real LTX2.3/q8 joint denoise, with seed20
/// synthetic conditioning (the committed producer), NOT an LTX2.5 encoder claim.
/// Only latent outputs are repackaged; no zero/dummy model tensors are admitted.
#[cfg(feature = "model-inference-proof")]
fn real_ltx25_prepared_item(root: &Path) -> TrainingItem {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx_av_e2e_golden.safetensors");
    let bytes = candle_gen::gen_core::__model_input_reader!(
        std::fs::read(&golden),
        "candle-gen-ltx::trainer_e2e::real_ltx25_prepared_item/real-ltx23-target-data",
        TensorInput,
        &golden,
        |actual_path| std::fs::read(actual_path)
    )
    .expect("committed real AV denoise fixture");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let video = tensors.tensor("video_latents").unwrap();
    let audio = tensors.tensor("audio_latents").unwrap();
    assert_eq!(video.shape(), [1, 128, 2, 8, 8]);
    assert_eq!(audio.shape(), [1, 8, 9, 16]);
    for tensor in [&video, &audio] {
        assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
        let values: Vec<f32> = tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > 1e-8));
    }
    // 2 latent frames -> 9 output frames; production audio mapping gives 9
    // latent audio frames at24fps. Native prepared reader validates this again.
    let metadata = std::collections::HashMap::from([
        ("schemaVersion".into(), "ltx-prepared-v1".into()),
        ("videoShape".into(), "[1,128,2,8,8]".into()),
        ("audioShape".into(), "[1,8,9,16]".into()),
        ("fps".into(), "24".into()),
        ("targetDataProvenance".into(), "committed mlx-gen-ltx/tests/fixtures/ltx_av_e2e_golden.safetensors; producer mlx-gen/tools/dump_ltx_av_e2e_golden.py; real LTX2.3 q8 joint denoise; seed20 synthetic conditioning;9frames24fps; trainer=LTX2.5 Dev/q4".into()),
    ]);
    let path = root.join("real_ltx23_targets_for_ltx25.safetensors");
    safetensors::serialize_to_file(
        [("video_latents", video), ("audio_latents", audio)],
        Some(metadata.clone()),
        &path,
    )
    .unwrap();
    std::fs::write(
        root.join("target_data_provenance.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let mut item = TrainingItem::captioned(path.clone(), "a solid colour swatch".into());
    item.model_options.insert(
        "ltxPreparedBundlePath".into(),
        serde_json::json!(path.to_str().unwrap()),
    );
    item
}

#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn ltx25_bounded_acceleration_acceptance() {
    let root = bounded_output_root();
    let item = real_ltx25_prepared_item(&root);
    let tier = required_dir("LTX25_TRAINING_TIER");
    let spec = LoadSpec::new(WeightsSource::Dir(tier)).with_quant(candle_gen::gen_core::Quant::Q4);
    let mut config = TrainingConfig {
        rank: 4,
        alpha: 4.0,
        learning_rate: 1e-3,
        steps: 4,
        resolution: 256,
        save_every: 0,
        seed: 7,
        network_type: NetworkType::Lora,
        train_dtype: "f32".into(),
        gradient_checkpointing: false,
        gradient_accumulation: 1,
        ..Default::default()
    };
    config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    let request = TrainingRequest {
        items: vec![item],
        config,
        output_dir: root.join("out"),
        file_name: "ltx25_bounded.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &root.join("metrics.json"),
        &[
            "ltx25:differentiable-QK-RMS",
            "ltx25:exact-grid-rope-storage-hit",
            "ltx25:native-incompatible-projection-fallback",
            "ltx:bounded-packed-frozen-input-vjp",
            "ltx:packed-frozen-vjp-reference-parity",
        ],
        || {
            let mut trainer = candle_gen_ltx::provider_registry()
                .unwrap()
                .load_trainer(candle_gen_ltx::MODEL_25_ID, &spec)
                .expect("strict real Dev/q4 trainer");
            trainer
                .train(&request, &mut |_| {})
                .expect("four joint AV optimizer updates")
        },
    );
    bounded_peak(sampler, &root);
    assert_eq!(output.steps, 4);
    assert!(output.final_loss.is_finite());
    let factors =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    assert!(factors.keys().any(|key| key.ends_with(".lora_A.weight")));
    assert!(factors
        .iter()
        .any(|(key, tensor)| key.ends_with(".lora_B.weight")
            && tensor
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 0.));
    // Production reopening + strict adapter installation. Native Dev30 is the
    // admitted schedule; training's four-update budget does not count denoise.
    let generator = candle_gen_ltx::provider_registry()
        .unwrap()
        .load(
            candle_gen_ltx::MODEL_25_ID,
            &spec.with_adapters(vec![AdapterSpec::new(
                output.adapter_path,
                1.,
                AdapterKind::Lora,
            )]),
        )
        .expect("reload and apply saved AV adapter to Dev generator");
    let request = GenerationRequest {
        prompt: "a solid colour swatch".into(),
        width: 256,
        height: 256,
        frames: Some(9),
        steps: Some(30),
        seed: Some(7),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("one adapted native Dev sample");
    let GenerationOutput::Video { frames, .. } = output else {
        panic!("expected video");
    };
    assert_eq!(frames.len(), 9);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.width, frame.height), (256, 256));
        assert_eq!(frame.pixels.len(), 256 * 256 * 3);
        assert!(frame.pixels.iter().any(|pixel| *pixel != 0));
        candle_gen::testkit::save_image_png(
            &root.join(format!("adapter_sample_frame_{index:02}.png")),
            frame,
        );
    }
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::ltx25_bounded_acceleration_acceptance");
}
