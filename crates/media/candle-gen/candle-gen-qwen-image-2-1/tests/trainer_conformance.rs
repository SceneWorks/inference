//! gen-core **Trainer contract** conformance for the candle `qwen_image_2_1` trainer (sc-24160),
//! on the committed miniature snapshot — so it runs on every CPU lane, not behind real weights.
//!
//! Drives the registered trainer through the backend-neutral checks: validate honesty (shared
//! control / full-fine-tune / instruction-edit floors, empty dataset, unknown knobs), a completed
//! run's `TrainingProgress` monotonicity, and typed cancellation before any step.

use std::path::Path;

use candle_gen::gen_core::{LoadSpec, TrainingItem, WeightsSource};
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

#[test]
fn qwen_image_2_1_trainer_satisfies_gen_core_contract() {
    let tmp = tempfile::Builder::new()
        .prefix(&format!(
            "qwen21_trainer_conformance_{}_",
            std::process::id()
        ))
        .tempdir()
        .unwrap();
    let items = make_dataset(&tmp.path().join("data"));
    let mut profile = TrainerProfile::cheap(items, tmp.path().join("out"));
    // The miniature DiT's widths are tiny; rank 4 keeps the factors below every projection.
    profile.config.rank = 4;
    profile.config.alpha = 4.0;
    let snapshot = tiny_snapshot();
    gen_core_testkit::trainer_conformance(
        || {
            candle_gen_qwen_image_2_1::provider_registry()
                .unwrap()
                .load_trainer(
                    "qwen_image_2_1",
                    &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
                )
                .expect("load the qwen_image_2_1 trainer")
        },
        &profile,
    );
}
