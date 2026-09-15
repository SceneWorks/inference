//! Real-weight gen-core trainer conformance for the Candle LTX-2.3 LoRA trainer (sc-13868).
//!
//! This drives the actual provider through descriptor honesty, registry discovery, progress
//! monotonicity, and typed pre-step cancellation. It is ignored and CUDA-gated because the progress
//! and mid-cache cancellation checks use the real packed tier, Gemma encoder, and VAE:
//!
//! ```text
//! set LTX_TRAINING_TIER=E:\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\<hash>\q4
//! set LTX_GEMMA_DIR=E:\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\<hash>\gemma
//! cargo test -p candle-gen-ltx --features cuda --release --test integration trainer_conformance:: -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{LoadSpec, TrainingItem, WeightsSource};
use gen_core_testkit::TrainerProfile;

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

fn load_spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(required_dir("LTX_TRAINING_TIER")));
    spec.text_encoder = Some(WeightsSource::Dir(required_dir("LTX_GEMMA_DIR")));
    spec
}

fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    [[200u8, 40, 40], [40, 80, 200]]
        .iter()
        .enumerate()
        .map(|(index, color)| {
            let mut image = image::RgbImage::new(64, 64);
            for pixel in image.pixels_mut() {
                *pixel = image::Rgb(*color);
            }
            let image_path = dir.join(format!("swatch_{index}.png"));
            image.save(&image_path).unwrap();
            TrainingItem::captioned(image_path, format!("a solid colour swatch number {index}"))
        })
        .collect()
}

#[test]
#[ignore = "needs SceneWorks/ltx-2.3-mlx q4 + Gemma weights and a CUDA GPU"]
fn ltx_trainer_satisfies_gen_core_contract() {
    assert_eq!(candle_gen_ltx::config::TRAINER_ID, "ltx_2_3");
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let items = make_dataset(&tmp.join("data"));
    let mut profile = TrainerProfile::cheap(items, tmp.join("out"));
    profile.config.train_dtype = "f32".to_string();
    profile.config.gradient_checkpointing = true;

    let registry = candle_gen_ltx::provider_registry().expect("LTX registry");
    let trainer = registry
        .load_trainer(candle_gen_ltx::config::TRAINER_ID, &load_spec())
        .expect("load registered LTX trainer");
    gen_core_testkit::check_trainer_registry(&registry, trainer.as_ref())
        .expect("trainer descriptor id is discoverable in the LTX registry");

    gen_core_testkit::trainer_conformance(
        || {
            candle_gen_ltx::provider_registry()
                .unwrap()
                .load_trainer(candle_gen_ltx::config::TRAINER_ID, &load_spec())
                .expect("load registered LTX trainer")
        },
        &profile,
    );
}
