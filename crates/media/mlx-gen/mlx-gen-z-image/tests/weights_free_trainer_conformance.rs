//! Weights-free gen-core **trainer conformance** (sc-2124 AT1) for every trainer the `mlx-gen-z-image`
//! registry registers. Each one is loaded through `provider_registry().load_trainer(id, ..)` — the
//! worker's path — from an EMPTY snapshot directory, so the shared validate-honesty and
//! undeclared-technique-refusal probes run without weights: the lazy loader defers every weight read
//! to a `train` that has cleared the refusal floors, and any earlier read fails on the empty dir.

use std::path::{Path, PathBuf};

use gen_core_testkit::trainer::{
    check_trainer_technique_refusal, check_trainer_train_floors, check_trainer_validate,
    TrainerProfile,
};
use mlx_gen::{LoadSpec, Trainer, TrainingItem, WeightsSource};

/// A deterministic 64×64 RGB PNG (a diagonal colour ramp) written under `dir`.
fn write_image(dir: &Path, name: &str, phase: u32) -> PathBuf {
    let img = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([
            ((x * 4 + phase) % 256) as u8,
            ((y * 4 + 2 * phase) % 256) as u8,
            (((x + y) * 2) % 256) as u8,
        ])
    });
    let path = dir.join(name);
    img.save(&path).expect("write the dataset image");
    path
}

fn dataset(dir: &Path) -> Vec<TrainingItem> {
    vec![
        TrainingItem::captioned(write_image(dir, "a.png", 0), "a red swatch".into()),
        TrainingItem::captioned(write_image(dir, "b.png", 40), "a blue ramp".into()),
    ]
}

/// *Mutations that red this:* reading weights at construction (the registry load fails on the empty
/// snapshot); a `validate` that needs the loaded base for a default-target request, or a `train` that
/// loads before its refusal floors (the load error is not the typed `Unsupported` refusal).
#[test]
fn every_registered_trainer_validates_and_refuses_techniques_without_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join("snapshot");
    std::fs::create_dir_all(&snapshot).unwrap();
    let profile = TrainerProfile::cheap(dataset(tmp.path()), tmp.path().join("out"));
    let registry = mlx_gen_z_image::provider_registry().unwrap();
    let ids: Vec<&str> = registry
        .trainers()
        .map(|registration| (registration.descriptor)().id)
        .collect();
    assert!(!ids.is_empty(), "the registry registers no trainer");
    for id in ids {
        let make = || -> Box<dyn Trainer> {
            registry
                .load_trainer(id, &LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
                .unwrap_or_else(|e| panic!("{id}: the weights-free trainer load failed: {e}"))
        };
        check_trainer_validate(make().as_ref(), &profile).unwrap_or_else(|e| panic!("{e}"));
        check_trainer_technique_refusal(&make, &profile).unwrap_or_else(|e| panic!("{e}"));
        // Every validate floor runs at `train` too (E3): a full fine-tune / control / edit
        // request sent straight to `train` is refused before any progress (sc-2124).
        check_trainer_train_floors(&make, &profile).unwrap_or_else(|e| panic!("{e}"));
    }
}
