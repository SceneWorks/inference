//! Closure export refusals and `pipeline.json` parsing. The copy of real pinned files is exercised
//! by the `#[ignore]`d real-weight engine test (CI has no weights).

use candle_audio::gen_core::{LoadSpec, WeightsSource};

use super::*;
use crate::provider::VAE_COMPONENT_ID;

fn pipeline(dir: &Path, repos: Value) {
    std::fs::create_dir_all(dir).unwrap();
    let metadata = json!({
        "format": CLOSURE_FORMAT,
        "repositories": repos,
        "decoders": ["standard", "legacy"],
        "generation_config": GenerationConfig::default().to_json(),
    });
    std::fs::write(
        dir.join(PIPELINE_JSON),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
}

#[test]
fn saving_needs_every_snapshot_and_leaves_nothing_behind_when_one_is_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("closure");
    let err = save_closure(
        &SnapshotDirs::new(),
        &[VaeVariant::Standard],
        &GenerationConfig::default(),
        &dest,
    )
    .unwrap_err();
    assert!(err.to_string().contains("offline cache miss"), "{err}");
    assert!(!dest.exists());
    assert!(!partial_dir(&dest).exists(), "the working copy is removed");

    let err = save_closure(
        &SnapshotDirs::new(),
        &[],
        &GenerationConfig::default(),
        &dest,
    )
    .unwrap_err();
    assert!(matches!(err, RunError::Invalid(_)));

    std::fs::create_dir(&dest).unwrap();
    std::fs::write(dest.join("keep"), b"x").unwrap();
    let err = save_closure(
        &SnapshotDirs::new(),
        &[VaeVariant::Standard],
        &GenerationConfig::default(),
        &dest,
    )
    .unwrap_err();
    assert!(matches!(err, RunError::Exists(_)), "{err}");
    assert_eq!(std::fs::read(dest.join("keep")).unwrap(), b"x");
}

#[test]
fn a_saved_closure_maps_back_to_its_snapshot_directories() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("closure");
    pipeline(
        &dir,
        json!({
            "m-a-p/YuE2-3B": "YuE2-3B",
            "m-a-p/YuE2-Vae": "YuE2-Vae",
            "m-a-p/YuE2-Vae-legacy": "YuE2-Vae-legacy",
        }),
    );
    for sub in ["YuE2-3B", "YuE2-Vae", "YuE2-Vae-legacy"] {
        std::fs::create_dir(dir.join(sub)).unwrap();
    }
    let saved = load_closure(&dir).unwrap();
    assert_eq!(saved.decoders, [VaeVariant::Standard, VaeVariant::Legacy]);
    assert_eq!(saved.generation, GenerationConfig::default());
    for id in [
        ComponentId::Lm,
        ComponentId::QwenTiktoken,
        ComponentId::VaeStandard,
        ComponentId::VaeLegacy,
    ] {
        let repo = &id.component().repo;
        assert_eq!(
            saved.dirs.snapshot_dir(repo).unwrap(),
            dir.join(repo_dir_name(repo))
        );
    }
    // The saved closure's files are still verified at load: these directories are empty.
    let err = Yue2Engine::load_saved(&dir, DType::F32, &Device::Cpu, EngineOptions::default())
        .unwrap_err();
    assert!(err.to_string().contains("YuE2"), "{err}");

    // Through the provider: a closure directory is a complete `weights`, with no components.
    let alone = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    assert!(
        crate::provider::load(&alone).is_err(),
        "verified (and refused) at load, not before"
    );
    let with_components = alone.with_component(VAE_COMPONENT_ID, WeightsSource::Dir(dir.clone()));
    let err = crate::provider::load(&with_components).err().unwrap();
    assert!(err.to_string().contains("saved closure"), "{err}");
}

#[test]
fn a_closure_cannot_point_outside_itself() {
    let tmp = tempfile::tempdir().unwrap();
    for bad in ["../elsewhere", "/abs", "a/b", ""] {
        let dir = tmp.path().join(format!("c{}", bad.len()));
        pipeline(&dir, json!({ "m-a-p/YuE2-3B": bad }));
        assert!(
            matches!(load_closure(&dir), Err(RunError::Corrupt { .. })),
            "{bad:?}"
        );
    }
    let dir = tmp.path().join("wrong-format");
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join(PIPELINE_JSON), br#"{"format": "other"}"#).unwrap();
    assert!(matches!(load_closure(&dir), Err(RunError::Corrupt { .. })));
    assert!(!is_saved_closure(tmp.path()));
    assert!(is_saved_closure(&dir));
}
