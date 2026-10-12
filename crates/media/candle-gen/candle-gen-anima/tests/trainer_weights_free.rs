//! Weights-free gen-core **Trainer contract** conformance for the candle `anima_base` trainer (epic
//! 2123 AT1) — runs on every CPU lane.
//!
//! The trainer is loaded through the crate's registered `load_trainer` (the production path), which
//! reads only the DiT file's safetensors *header* to refuse a packed tier, then records the source.
//! A per-process snapshot whose DiT is a one-tensor dense placeholder therefore loads it, and
//! `validate` and the `train` refusal floors run before any weight is read — enough to drive
//! `check_trainer_validate` and `check_trainer_technique_refusal`.

use std::path::{Path, PathBuf};

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

/// A snapshot whose `split_files/diffusion_models/<base DiT>` is a valid one-tensor dense
/// safetensors file — all the loader's packed-tier probe reads.
fn placeholder_snapshot(root: &Path) -> PathBuf {
    let dit = root
        .join("split_files")
        .join("diffusion_models")
        .join(candle_gen_anima::Variant::Base.dit_filename());
    std::fs::create_dir_all(dit.parent().unwrap()).unwrap();
    let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&0_f32.to_le_bytes());
    std::fs::write(&dit, bytes).unwrap();
    root.to_path_buf()
}

fn load_trainer(snapshot: &Path) -> Box<dyn Trainer> {
    candle_gen_anima::provider_registry()
        .unwrap()
        .load_trainer(
            candle_gen_anima::Variant::Base.id(),
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .expect("load the anima_base trainer")
}

#[test]
fn anima_trainer_validates_and_refuses_without_weights() {
    let tmp = temp_root("anima");
    let snapshot = placeholder_snapshot(&tmp.path().join("snapshot"));
    let mut profile = TrainerProfile::cheap(
        make_dataset(&tmp.path().join("data")),
        tmp.path().join("out"),
    );
    // The dense base trains at bf16; `validate` refuses any other train dtype.
    profile.config.train_dtype = "bf16".to_owned();
    gen_core_testkit::check_trainer_validate(load_trainer(&snapshot).as_ref(), &profile).unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&|| load_trainer(&snapshot), &profile)
        .unwrap();
    gen_core_testkit::check_trainer_train_floors(&|| load_trainer(&snapshot), &profile).unwrap();
}
