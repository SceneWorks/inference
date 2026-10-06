//! Weights-free gen-core **trainer conformance** (sc-2124 AT1) for every trainer the `mlx-gen-sdxl`
//! registry registers. Each one is loaded through `provider_registry().load_trainer(id, ..)` — the
//! worker's path — from an EMPTY snapshot directory, so the shared validate-honesty and
//! undeclared-technique-refusal probes run without weights: the lazy loader defers every weight read
//! to a `train` that has cleared the refusal floors, and any earlier read fails on the empty dir.

use std::path::{Path, PathBuf};

use gen_core_testkit::trainer::{
    check_trainer_technique_refusal, check_trainer_validate, TrainerProfile,
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
    let registry = mlx_gen_sdxl::provider_registry().unwrap();
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
    }
}

/// sc-2124 (epic 2123 E3): a request only a NON-technique floor refuses — a control-branch request on
/// this LoRA-only trainer — sent straight to `train` (no `validate`) is refused with the floor's typed
/// `Unsupported` before any progress event and before the base loads: `train` runs every validate
/// floor, not just the technique one. (Non-vacuous even where the trainer declares every probed
/// technique, which leaves `check_trainer_technique_refusal` nothing to refuse.)
///
/// *Mutations that red this:* `train` loading before its floors, or running only the technique
/// floor (the request then reaches the weight load, whose error is not the control refusal).
#[test]
fn train_refuses_a_non_technique_floor_before_any_progress() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join("snapshot");
    std::fs::create_dir_all(&snapshot).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    let profile = TrainerProfile::cheap(dataset(tmp.path()), tmp.path().join("out"));
    let req = mlx_gen::TrainingRequest {
        items: profile.items,
        config: mlx_gen::TrainingConfig {
            control_type: Some("pose".into()),
            ..profile.config
        },
        output_dir: profile.output_dir,
        file_name: profile.file_name,
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let registry = mlx_gen_sdxl::provider_registry().unwrap();
    for registration in registry.trainers() {
        let id = (registration.descriptor)().id;
        let mut trainer = registry
            .load_trainer(id, &spec)
            .unwrap_or_else(|e| panic!("{id}: the weights-free trainer load failed: {e}"));
        let mut events = 0;
        let result = trainer.train(&req, &mut |_| events += 1);
        assert!(
            matches!(result, Err(mlx_gen::gen_core::Error::Unsupported(_))),
            "{id}: a control-branch train() must be the typed control refusal, got {:?}",
            result.map(|out| out.steps)
        );
        assert_eq!(
            events, 0,
            "{id}: the refusal must precede every progress event"
        );
    }
}
