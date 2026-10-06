//! Weights-free gen-core **trainer conformance** (sc-2124 AT1) for every trainer the `mlx-gen-ltx`
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

/// A tiny LTX-2.5 prepared pack (one T2V video latent + matching audio latent) at `path` — the
/// shape-only contract `validate_ltx25_training_request` reads (no weights).
fn write_tiny_prepared_pack(path: &Path) {
    let total_bytes = 512 + 8 * 16 * std::mem::size_of::<f32>();
    let header = serde_json::json!({
        "__metadata__": {
            "schemaVersion": "ltx-prepared-v1",
            "videoShape": "[1,128,1,1,1]",
            "audioShape": "[1,8,1,16]",
            "fps": "24"
        },
        "video_latents": {"dtype": "F32", "shape": [1, 128, 1, 1, 1], "data_offsets": [0, 512]},
        "audio_latents": {"dtype": "F32", "shape": [1, 8, 1, 16], "data_offsets": [512, total_bytes]}
    });
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(bytes.len() + total_bytes, 0);
    std::fs::write(path, bytes).unwrap();
}

/// LTX-2.3 and LTX-2.5 (both registrations, each with its own descriptor) through the shared probes.
/// LTX-2.3 is handed an existing (empty) Gemma-3 dir — construction still checks the override exists
/// (sc-9989) — and LTX-2.5 a T2V workflow over tiny prepared packs, the dataset its `validate` takes.
///
/// *Mutations that red this:* reading weights at construction (the registry load fails on the empty
/// snapshot); a `train` that loads before its refusal floors (the load error is not the typed
/// `Unsupported` refusal).
#[test]
fn both_ltx_trainers_validate_and_refuse_techniques_without_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let snapshot = tmp.path().join("snapshot");
    let gemma = tmp.path().join("gemma");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(&gemma).unwrap();
    let out = tmp.path().join("out");

    let profile_23 = TrainerProfile::cheap(dataset(tmp.path()), out.clone());
    let mut profile_25 = TrainerProfile::cheap(dataset(tmp.path()), out);
    profile_25
        .config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    for (i, item) in profile_25.items.iter_mut().enumerate() {
        let pack = tmp.path().join(format!("prepared_{i}.safetensors"));
        write_tiny_prepared_pack(&pack);
        item.model_options
            .insert("ltxPreparedBundlePath".into(), serde_json::json!(pack));
    }

    let registry = mlx_gen_ltx::provider_registry().unwrap();
    let ids: Vec<&str> = registry
        .trainers()
        .map(|registration| (registration.descriptor)().id)
        .collect();
    assert!(
        ids.contains(&mlx_gen_ltx::MODEL_ID) && ids.contains(&mlx_gen_ltx::MODEL_25_ID),
        "both LTX trainers must be registered, got {ids:?}"
    );
    for id in ids {
        let profile = if id == mlx_gen_ltx::MODEL_25_ID {
            &profile_25
        } else {
            &profile_23
        };
        let make = || -> Box<dyn Trainer> {
            let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone()));
            spec.text_encoder = Some(WeightsSource::Dir(gemma.clone()));
            registry
                .load_trainer(id, &spec)
                .unwrap_or_else(|e| panic!("{id}: the weights-free trainer load failed: {e}"))
        };
        check_trainer_validate(make().as_ref(), profile).unwrap_or_else(|e| panic!("{e}"));
        check_trainer_technique_refusal(&make, profile).unwrap_or_else(|e| panic!("{e}"));
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
    let gemma = tmp.path().join("gemma");
    std::fs::create_dir_all(&gemma).unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    spec.text_encoder = Some(WeightsSource::Dir(gemma));
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
    let registry = mlx_gen_ltx::provider_registry().unwrap();
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
