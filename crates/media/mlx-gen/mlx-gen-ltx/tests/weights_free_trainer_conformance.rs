//! Weights-free gen-core **trainer conformance** (sc-2124 AT1) for every trainer the `mlx-gen-ltx`
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
        // Every validate floor runs at `train` too (E3): a full fine-tune / control / edit
        // request sent straight to `train` is refused before any progress (sc-2124).
        check_trainer_train_floors(&make, profile).unwrap_or_else(|e| panic!("{e}"));
    }
}
