//! gen-core **Trainer contract** conformance for the candle `qwen_image_2_1` trainer (sc-24160),
//! on the committed miniature snapshot — so it runs on every CPU lane, not behind real weights.
//!
//! Drives the registered trainer through the backend-neutral checks: validate honesty (shared
//! control / full-fine-tune / instruction-edit floors — with the trainer's reference cap > 0 since
//! sc-24162, the edit floor's check is that one reference over the cap is refused — empty dataset,
//! unknown knobs), a completed run's `TrainingProgress` monotonicity, and typed cancellation before
//! any step. Run twice: over a captioned (text-to-image) dataset and over an instruction-edit one.

use std::path::Path;

use candle_gen::gen_core::{LoadSpec, Trainer, TrainingItem, WeightsSource};
use gen_core_testkit::TrainerProfile;

use crate::common::tiny_snapshot;

/// Two small swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    std::fs::create_dir_all(dir).unwrap();
    [[200u8, 40, 40], [40, 80, 200]]
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let mut img = image::RgbImage::new(48, 48);
            for px in img.pixels_mut() {
                *px = image::Rgb(*color);
            }
            let path = dir.join(format!("img{i}.png"));
            img.save(&path).unwrap();
            TrainingItem::captioned(path, format!("a solid colour swatch number {i}"))
        })
        .collect()
}

/// [`make_dataset`] as instruction-edit pairs: each swatch is the target of an edit whose single
/// reference is the other swatch (sc-24162).
fn make_edit_dataset(dir: &Path) -> Vec<TrainingItem> {
    let swatches = make_dataset(dir);
    swatches
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let source = swatches[(i + 1) % swatches.len()].image_path.clone();
            TrainingItem::edit_pair(
                item.image_path.clone(),
                format!("recolour the swatch to colour number {i}"),
                vec![source],
            )
        })
        .collect()
}

fn load_trainer(snapshot: &Path) -> Box<dyn Trainer> {
    candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load_trainer(
            "qwen_image_2_1",
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .expect("load the qwen_image_2_1 trainer")
}

fn run_conformance(tag: &str, make: fn(&Path) -> Vec<TrainingItem>) {
    let tmp = tempfile::Builder::new()
        .prefix(&format!(
            "qwen21_trainer_conformance_{tag}_{}_",
            std::process::id()
        ))
        .tempdir()
        .unwrap();
    let items = make(&tmp.path().join("data"));
    let mut profile = TrainerProfile::cheap(items, tmp.path().join("out"));
    // The miniature DiT's widths are tiny; rank 4 keeps the factors below every projection.
    profile.config.rank = 4;
    profile.config.alpha = 4.0;
    let snapshot = tiny_snapshot();
    assert!(
        load_trainer(&snapshot).descriptor().max_reference_images > 0,
        "the candle 2.1 trainer advertises the render path's reference cap"
    );
    gen_core_testkit::trainer_conformance(|| load_trainer(&snapshot), &profile);
}

#[test]
fn qwen_image_2_1_trainer_satisfies_gen_core_contract() {
    run_conformance("t2i", make_dataset);
}

/// sc-24162: the same contract over an instruction-edit dataset, with the trainer's cap > 0 — the
/// completed run, its progress and its cancellation go through the edit path.
#[test]
fn qwen_image_2_1_edit_trainer_satisfies_gen_core_contract() {
    run_conformance("edit", make_edit_dataset);
}
