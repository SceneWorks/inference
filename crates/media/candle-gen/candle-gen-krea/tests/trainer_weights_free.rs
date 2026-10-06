//! Weights-free gen-core **Trainer contract** conformance for the candle Krea trainers — the
//! `krea_2_raw` LoRA trainer and the `krea_2_control` ControlNet-branch trainer (epic 2123 AT1) —
//! runs on every CPU lane.
//!
//! Both are loaded through the crate's registered `load_trainer` (the production path), which is
//! lazy: it only records the snapshot directory. `validate` and the `train` refusal floors run before
//! any weight is read, so an empty per-process snapshot dir drives the testkit checks. Both also pin
//! that `train`, called directly, runs every `validate` floor (epic 2123 E3).

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

/// The control trainer's id (its constant is crate-private).
const KREA_2_CONTROL_ID: &str = "krea_2_control";

fn load_trainer(id: &str, snapshot: &Path) -> Box<dyn Trainer> {
    candle_gen_krea::provider_registry()
        .unwrap()
        .load_trainer(
            id,
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .unwrap_or_else(|e| panic!("load the {id} trainer: {e}"))
}

#[test]
fn krea_trainer_validates_and_refuses_without_weights() {
    let tmp = temp_root("krea");
    let snapshot = tmp.path().join("snapshot");
    std::fs::create_dir_all(&snapshot).unwrap();
    let profile = TrainerProfile::cheap(
        make_dataset(&tmp.path().join("data")),
        tmp.path().join("out"),
    );
    let id = candle_gen_krea::KREA_2_RAW_ID;
    gen_core_testkit::check_trainer_validate(load_trainer(id, &snapshot).as_ref(), &profile)
        .unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&|| load_trainer(id, &snapshot), &profile)
        .unwrap();
    gen_core_testkit::check_trainer_train_floors(&|| load_trainer(id, &snapshot), &profile)
        .unwrap();
}

/// The control trainer trains a ControlNet branch; the testkit's validate check builds its
/// positive request as a control request (control type + a conditioning image on every item) for a
/// trainer advertising `supports_control` and neither adapter kind.
#[test]
fn krea_control_trainer_refuses_without_weights() {
    let tmp = temp_root("krea_control");
    let snapshot = tmp.path().join("snapshot");
    std::fs::create_dir_all(&snapshot).unwrap();
    let mut items = make_dataset(&tmp.path().join("data"));
    for item in &mut items {
        item.control_image_path = Some(item.image_path.clone());
    }
    let mut profile = TrainerProfile::cheap(items, tmp.path().join("out"));
    profile.config.control_type = Some("pose".to_owned());
    let make = || load_trainer(KREA_2_CONTROL_ID, &snapshot);
    gen_core_testkit::check_trainer_validate(make().as_ref(), &profile).unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&make, &profile).unwrap();
    gen_core_testkit::check_trainer_train_floors(&make, &profile).unwrap();
}
