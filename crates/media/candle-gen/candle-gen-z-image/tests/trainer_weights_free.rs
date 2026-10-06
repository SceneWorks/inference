//! Weights-free gen-core **Trainer contract** conformance for the candle `z_image_turbo` trainer
//! (epic 2123 AT1) — runs on every CPU lane, beside the real-weight `trainer_conformance` suite.
//!
//! The trainer is loaded through the crate's registered `load_trainer` (the production path), which
//! is lazy: it only records the snapshot directory. `validate` and the `train` refusal floors run
//! before any weight is read, so an empty per-process snapshot dir is enough to drive
//! `check_trainer_validate` and `check_trainer_technique_refusal`.
//!
//! It also pins that `train`, called directly, runs every `validate` floor (epic 2123 E3) — not
//! only the technique one.

use std::path::Path;

use candle_gen::gen_core::{LoadSpec, Trainer, TrainingItem, WeightsSource};
use gen_core_testkit::TrainerProfile;

/// Two small swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    std::fs::create_dir_all(dir).unwrap();
    [[200u8, 40, 40], [40, 80, 200]]
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let path = dir.join(format!("img{i}.png"));
            image::RgbImage::from_pixel(32, 32, image::Rgb(*color))
                .save(&path)
                .unwrap();
            TrainingItem::captioned(path, format!("a solid colour swatch number {i}"))
        })
        .collect()
}

fn load_trainer(snapshot: &Path) -> Box<dyn Trainer> {
    candle_gen_z_image::provider_registry()
        .unwrap()
        .load_trainer(
            candle_gen_z_image::MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .expect("load the z_image_turbo trainer")
}

#[test]
fn z_image_turbo_trainer_validates_and_refuses_without_weights() {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("z_image_trainer_wf_{}_", std::process::id()))
        .tempdir()
        .unwrap();
    let snapshot = tmp.path().join("snapshot");
    std::fs::create_dir_all(&snapshot).unwrap();
    let profile = TrainerProfile::cheap(
        make_dataset(&tmp.path().join("data")),
        tmp.path().join("out"),
    );

    gen_core_testkit::check_trainer_validate(load_trainer(&snapshot).as_ref(), &profile).unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&|| load_trainer(&snapshot), &profile)
        .unwrap();
    gen_core_testkit::check_trainer_train_floors(&|| load_trainer(&snapshot), &profile).unwrap();
}
