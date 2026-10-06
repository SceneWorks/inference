//! Weights-free gen-core **Trainer contract** conformance for the candle LTX trainers — the
//! `ltx_2_3` and the LTX-2.5 (`ltx_2_5_distilled`) registrations (epic 2123 AT1) — runs on every CPU
//! lane, beside the real-weight `trainer_conformance` suite.
//!
//! Both are loaded through the crate's registered `load_trainer` (the production path):
//!
//! * `ltx_2_3` is lazy — it only records the tier directory — so an empty per-process dir loads it.
//! * LTX-2.5 resolves and validates its split bundle at load: the Gemma-version pairing, the dev
//!   transformer identity and the whole `split_model.json` q4 tier, all from safetensors *headers*.
//!   It loads from the structurally exact synthetic q4 tier of `packed_tier_validate`, with the
//!   transformer and text encoder stamped with the identity metadata a real bundle carries. Its
//!   requests carry a prepared latent pack per item (header-only too) and the `t2v_lora` workflow.
//!
//! `validate` and the `train` refusal floors run before any weight is read, so these drive
//! `check_trainer_validate` and `check_trainer_technique_refusal`. The LTX-2.5 test also pins the
//! refusal order (epic 2123 E3): the specific LTX-2.5 subject-mask reason wins over the generic
//! technique floor's, from both `validate` and `train`.

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    Error, LoadSpec, SubjectMaskLoss, Trainer, TrainingItem, TrainingRequest, WeightsSource,
};
use gen_core_testkit::TrainerProfile;

use crate::packed_tier_validate::ltx25_fixture::BundleSpec;

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

fn load_trainer(id: &str, weights: &Path) -> Box<dyn Trainer> {
    candle_gen_ltx::provider_registry()
        .unwrap()
        .load_trainer(
            id,
            &LoadSpec::new(WeightsSource::Dir(weights.to_path_buf())),
        )
        .unwrap_or_else(|e| panic!("load the {id} trainer: {e}"))
}

#[test]
fn ltx_2_3_trainer_validates_and_refuses_without_weights() {
    let tmp = temp_root("ltx23");
    let tier = tmp.path().join("q4");
    std::fs::create_dir_all(&tier).unwrap();
    let mut profile = TrainerProfile::cheap(
        make_dataset(&tmp.path().join("data")),
        tmp.path().join("out"),
    );
    // As the real-weight suite: LTX LoRA training is f32-only.
    profile.config.train_dtype = "f32".to_owned();
    profile.config.gradient_checkpointing = true;
    let id = candle_gen_ltx::config::TRAINER_ID;
    gen_core_testkit::check_trainer_validate(load_trainer(id, &tier).as_ref(), &profile).unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&|| load_trainer(id, &tier), &profile)
        .unwrap();
    gen_core_testkit::check_trainer_train_floors(&|| load_trainer(id, &tier), &profile).unwrap();
}

/// Merge `entries` into the `__metadata__` of the safetensors file at `path`, keeping its tensors.
fn stamp_metadata(path: &Path, entries: &[(&str, &str)]) {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let metadata = header["__metadata__"].as_object_mut().unwrap();
    for (key, value) in entries {
        metadata.insert((*key).to_owned(), serde_json::Value::from(*value));
    }
    let text = serde_json::to_string(&header).unwrap();
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(&bytes[8 + header_len..]);
    std::fs::write(path, out).unwrap();
}

/// The synthetic q4 LTX-2.5 tier, with the transformer stamped as the `dev` variant naming its Gemma
/// source and the text encoder declaring that Gemma — what bundle discovery, the Gemma-version check
/// and the trainer's dev/q4 gate read. Returns the tier directory (the trainer's weights root).
fn ltx25_dev_q4_tier(root: &Path) -> PathBuf {
    const GEMMA: &str = "gemma4-12b-ltx-v1";
    let tier = BundleSpec::q4().build(root);
    stamp_metadata(
        &tier.join("transformer.safetensors"),
        &[
            (
                "config",
                r#"{"transformer":{"_class_name":"AVTransformer3DModel"}}"#,
            ),
            ("variant", "dev"),
            (
                "gemma_source_checkpoint",
                &format!(r#"{{"ltx_version":"2.5.0","gemma_version":"{GEMMA}"}}"#),
            ),
        ],
    );
    stamp_metadata(
        &tier.join("text_encoder.safetensors"),
        &[(
            "gemma_config",
            &format!(r#"{{"model_type":"gemma4_unified","gemma_version":"{GEMMA}"}}"#),
        )],
    );
    tier
}

/// A header-valid prepared latent pack (`ltx-prepared-v1`: one video + one audio latent).
fn write_prepared_pack(path: &Path) {
    let video_bytes = 128 * 4;
    let audio_bytes = 8 * 16 * 4;
    let header = serde_json::json!({
        "__metadata__": {
            "schemaVersion": "ltx-prepared-v1",
            "videoShape": "[1,128,1,1,1]",
            "audioShape": "[1,8,1,16]",
            "fps": "24",
        },
        "video_latents": {
            "dtype": "F32", "shape": [1, 128, 1, 1, 1], "data_offsets": [0, video_bytes],
        },
        "audio_latents": {
            "dtype": "F32", "shape": [1, 8, 1, 16],
            "data_offsets": [video_bytes, video_bytes + audio_bytes],
        },
    })
    .to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + video_bytes + audio_bytes, 0);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn ltx_2_5_trainer_validates_and_refuses_without_weights() {
    let tmp = temp_root("ltx25");
    let tier = ltx25_dev_q4_tier(&tmp.path().join("bundle"));
    let pack = tmp.path().join("prepared.safetensors");
    write_prepared_pack(&pack);
    let mut items = make_dataset(&tmp.path().join("data"));
    for item in &mut items {
        item.model_options.insert(
            "ltxPreparedBundlePath".to_owned(),
            serde_json::Value::from(pack.to_string_lossy().into_owned()),
        );
    }
    let mut profile = TrainerProfile::cheap(items, tmp.path().join("out"));
    profile.config.train_dtype = "f32".to_owned();
    profile.config.model_options.insert(
        "ltxWorkflow".to_owned(),
        serde_json::Value::from("t2v_lora"),
    );
    let id = candle_gen_ltx::MODEL_25_ID;
    let make = || load_trainer(id, &tier);
    gen_core_testkit::check_trainer_validate(make().as_ref(), &profile).unwrap();
    gen_core_testkit::check_trainer_technique_refusal(&make, &profile).unwrap();

    // E3 refusal order: subject-masked loss is undeclared on LTX-2.5, so the generic technique
    // floor would refuse it too — the LTX-2.5 reason (no decodable image aligned to the prepared
    // latent) must be the one the caller sees, from `validate` and from a direct `train`.
    let mut masked = TrainingRequest {
        items: profile.items.clone(),
        config: profile.config.clone(),
        output_dir: profile.output_dir.clone(),
        file_name: profile.file_name.clone(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    masked.config.subject_mask_loss = Some(SubjectMaskLoss {
        background_weight: 0.1,
        subject_weight: 1.0,
    });
    for item in &mut masked.items {
        item.subject_mask_path = Some(item.image_path.clone());
    }
    let specific = |result: Result<(), Error>, seam: &str| match result {
        Err(Error::Unsupported(m)) if m.contains("prepared latent bundle") => {}
        other => panic!("{seam}: expected the LTX-2.5 subject-mask refusal, got {other:?}"),
    };
    specific(make().validate(&masked), "validate");
    let mut events = 0;
    let trained = make().train(&masked, &mut |_| events += 1).map(|_| ());
    specific(trained, "train");
    assert_eq!(events, 0, "train() emitted progress before refusing");
}
