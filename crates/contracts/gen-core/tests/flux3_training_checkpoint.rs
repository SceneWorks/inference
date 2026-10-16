//! Checkpoint transport tests use synthetic bytes and identities, never model weights.
use gen_core::flux3::training::checkpoint::{publish_bytes, TrainingCursor};
use gen_core::{CancelFlag, Error};

#[test]
fn cursor_roundtrips_pending_accumulation_rng_and_sampler_position() {
    let cursor = TrainingCursor {
        micro_step: 7,
        update_step: 3,
        accumulation_count: 1,
        sampler_epoch: 2,
        sampler_cursor: 3,
        rng_state: vec![4, 7, 9, 12],
    };
    cursor.validate(20, 2, 5).unwrap();
    let metadata = cursor.to_metadata().unwrap();
    assert_eq!(TrainingCursor::from_metadata(&metadata).unwrap(), cursor);
    let mut changed = cursor.clone();
    changed.accumulation_count = 0;
    assert!(changed.validate(20, 2, 5).is_err());
    let mut changed = cursor.clone();
    changed.rng_state.clear();
    assert!(changed.validate(20, 2, 5).is_err());
    let mut changed = cursor.clone();
    changed.sampler_cursor = 6;
    assert!(changed.validate(20, 2, 5).is_err());
    let mut json: serde_json::Value = serde_json::from_str(&metadata).unwrap();
    json["unknown_cursor"] = 1.into();
    assert!(TrainingCursor::from_metadata(&json.to_string()).is_err());
    let tail = TrainingCursor {
        micro_step: 7,
        update_step: 4,
        accumulation_count: 0,
        ..cursor
    };
    tail.validate(7, 2, 5).unwrap();
    assert!(
        tail.validate(20, 2, 5).is_err(),
        "partial-tail update is valid only at run completion"
    );
}

#[test]
fn publication_is_owned_atomic_non_overwriting_and_cancellation_aware() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancelFlag::new();
    let output = publish_bytes(
        dir.path(),
        "checkpoint-000007.safetensors",
        b"synthetic-first",
        &cancel,
    )
    .unwrap();
    assert_eq!(std::fs::read(&output).unwrap(), b"synthetic-first");
    assert!(publish_bytes(
        dir.path(),
        "checkpoint-000007.safetensors",
        b"must-not-overwrite",
        &cancel
    )
    .is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"synthetic-first");
    for name in [
        "../outside",
        "sub/file",
        "sub\\file",
        "C:\\outside",
        ".",
        "..",
        "",
    ] {
        assert!(
            publish_bytes(dir.path(), name, b"synthetic", &cancel).is_err(),
            "accepted {name}"
        );
    }
    cancel.cancel();
    assert!(matches!(
        publish_bytes(dir.path(), "canceled.safetensors", b"synthetic", &cancel),
        Err(Error::Canceled)
    ));
    let names = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(
        names.len(),
        1,
        "failed/canceled publication must not retain partial files"
    );
}
