//! Weights-free gen-core **Trainer contract** conformance for the candle `mage_flow_base` trainer
//! (epic 2123 AT1) — runs on every CPU lane.
//!
//! The trainer is loaded through the crate's registered `load_trainer` (the production path), which
//! only resolves the component directories and probes their `config.json` for a packed tier (absent
//! ⇒ dense). `validate` and the `train` refusal floors run before any weight is read, so an empty
//! per-process snapshot dir drives `check_trainer_validate` and `check_trainer_technique_refusal`.

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

/// A per-process temp root (CI shares `$TMPDIR` across processes).
fn temp_root(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("{tag}_trainer_wf_{}_", std::process::id()))
        .tempdir()
        .unwrap()
}

fn load_trainer(snapshot: &Path) -> Box<dyn Trainer> {
    candle_gen_mage::provider_registry()
        .unwrap()
        .load_trainer(
            candle_gen_mage::config::BASE_MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .expect("load the mage_flow_base trainer")
}

#[test]
fn mage_trainer_validates_and_refuses_without_weights() {
    let tmp = temp_root("mage");
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
