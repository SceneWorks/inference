//! Weights-free gen-core **Trainer contract** conformance for the candle `sdxl` trainer (epic 2123
//! AT1) — runs on every CPU lane.
//!
//! The trainer is loaded through the crate's registered `load_trainer` (the production path), which
//! only checks that the three passed-in components are named on the spec; their paths are read
//! lazily at `train`. `validate` and the `train` refusal floors run before any weight is read, so a
//! per-process snapshot dir with named (never-read) component dirs drives `check_trainer_validate`
//! and `check_trainer_technique_refusal`. It also pins that `train`, called directly, runs every
//! `validate` floor (epic 2123 E3).

use std::path::Path;

use candle_gen::gen_core::{
    Error, LoadSpec, Trainer, TrainingItem, TrainingRequest, WeightsSource,
};
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

fn load_trainer(root: &Path) -> Box<dyn Trainer> {
    let dir = |name: &str| WeightsSource::Dir(root.join(name));
    let spec = LoadSpec::new(dir("snapshot"))
        .with_component("tokenizer_clip_l", dir("tokenizer_clip_l"))
        .with_component("tokenizer_clip_bigg", dir("tokenizer_clip_bigg"))
        .with_component("vae_fp16_fix", dir("vae_fp16_fix"));
    candle_gen_sdxl::provider_registry()
        .unwrap()
        .load_trainer(candle_gen_sdxl::MODEL_ID, &spec)
        .expect("load the sdxl trainer")
}

#[test]
fn sdxl_trainer_validates_and_refuses_without_weights() {
    let tmp = temp_root("sdxl");
    let root = tmp.path().to_path_buf();
    std::fs::create_dir_all(root.join("snapshot")).unwrap();
    let profile = TrainerProfile::cheap(make_dataset(&root.join("data")), root.join("out"));
    gen_core_testkit::check_trainer_validate(load_trainer(&root).as_ref(), &profile).unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&|| load_trainer(&root), &profile).unwrap();
    check_train_runs_validate_floors(&|| load_trainer(&root), &profile);
}

/// Epic 2123 E3: `train` called directly (skipping `validate`) refuses a request that only a
/// non-technique `validate` floor catches — full fine-tune / control branch when not advertised, instruction edit — with a typed `Unsupported` before any
/// progress event, so nothing is loaded or cached.
fn check_train_runs_validate_floors(make: &dyn Fn() -> Box<dyn Trainer>, profile: &TrainerProfile) {
    let base = TrainingRequest {
        items: profile.items.clone(),
        config: profile.config.clone(),
        output_dir: profile.output_dir.clone(),
        file_name: profile.file_name.clone(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let desc = *make().descriptor();
    let mut probes = Vec::new();
    if !desc.supports_full_finetune {
        let mut full = base.clone();
        full.config.full_finetune = true;
        probes.push(("full_finetune", full));
    }
    if !desc.supports_control {
        let mut control = base.clone();
        control.config.control_type = Some("pose".to_owned());
        probes.push(("control_type", control));
    }
    let mut edit = base;
    for item in &mut edit.items {
        item.reference_image_paths = vec![item.image_path.clone()];
    }
    probes.push(("edit", edit));
    for (floor, req) in probes {
        let mut events = 0;
        let result = make().train(&req, &mut |_| events += 1);
        assert!(
            matches!(result, Err(Error::Unsupported(_))),
            "{floor}: train() must refuse with a typed Unsupported, got {:?}",
            result.err()
        );
        assert_eq!(
            events, 0,
            "{floor}: train() emitted progress before refusing"
        );
    }
}
