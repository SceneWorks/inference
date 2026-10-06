//! Weights-free gen-core **Trainer contract** conformance for the candle Wan 2.2 trainers — all three
//! registered ids (`wan2_2_t2v_14b`, `wan2_2_i2v_14b`, `wan2_2_ti2v_5b`) (epic 2123 AT1) — runs on
//! every CPU lane, beside the real-weight `trainer_conformance` suite.
//!
//! Each trainer is loaded through the crate's registered `load_trainer` (the production path), which
//! only probes the transformer component(s) for a packed tier (absent ⇒ dense) and records the
//! snapshot root. `validate` and the `train` refusal floors run before any weight is read, so an
//! empty per-process snapshot dir drives `check_trainer_validate` and
//! `check_trainer_technique_refusal`.

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

fn load_trainer(id: &str, snapshot: &Path) -> Box<dyn Trainer> {
    candle_gen_wan::provider_registry()
        .unwrap()
        .load_trainer(
            id,
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .unwrap_or_else(|e| panic!("load the {id} trainer: {e}"))
}

#[test]
fn wan_trainers_validate_and_refuse_without_weights() {
    let tmp = temp_root("wan");
    let snapshot = tmp.path().join("snapshot");
    std::fs::create_dir_all(&snapshot).unwrap();
    let profile = TrainerProfile::cheap(
        make_dataset(&tmp.path().join("data")),
        tmp.path().join("out"),
    );
    for id in [
        candle_gen_wan::config::MODEL_ID_T2V_14B,
        candle_gen_wan::config::MODEL_ID_I2V_14B,
        candle_gen_wan::config::MODEL_ID,
    ] {
        gen_core_testkit::check_trainer_validate(load_trainer(id, &snapshot).as_ref(), &profile)
            .unwrap();
        gen_core_testkit::check_trainer_technique_refusal(
            &|| load_trainer(id, &snapshot),
            &profile,
        )
        .unwrap();
        gen_core_testkit::check_trainer_train_floors(&|| load_trainer(id, &snapshot), &profile)
            .unwrap();
    }
}
