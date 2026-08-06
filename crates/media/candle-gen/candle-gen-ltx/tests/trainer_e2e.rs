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
//! cargo test -p candle-gen-ltx --features cuda --release --test trainer_e2e -- --ignored --nocapture
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
}
