//! Checkpoint transport tests use synthetic bytes and identities, never model weights.
use gen_core::flux3::training::checkpoint::{publish_bytes, TrainingCursor};
use gen_core::{CancelFlag, Error};

#[test]
fn cursor_roundtrips_pending_accumulation_rng_and_sampler_position() {
    let cursor = TrainingCursor {
        micro_step: 7,
        update_step: 3,
        accumulation_count: 1,
        sampler_epoch: 1,
        sampler_cursor: 3,
        rng_state: vec![4, 7, 9, 12],
        last_loss_bits: None,
    };
    cursor.validate(20, 2, 4).unwrap();
    let metadata = cursor.to_metadata().unwrap();
    assert_eq!(TrainingCursor::from_metadata(&metadata).unwrap(), cursor);
    let mut measured = cursor.clone();
    measured.last_loss_bits = Some(0.25f32.to_bits());
    assert_eq!(measured.last_loss().unwrap(), Some(0.25));
    assert_eq!(
        TrainingCursor::from_metadata(&measured.to_metadata().unwrap()).unwrap(),
        measured
    );
    measured.last_loss_bits = Some(f32::NAN.to_bits());
    assert!(measured.to_metadata().is_err());
    let mut changed = cursor.clone();
    changed.accumulation_count = 0;
    assert!(changed.validate(20, 2, 4).is_err());
    let mut changed = cursor.clone();
    changed.rng_state.clear();
    assert!(changed.validate(20, 2, 4).is_err());
    let mut changed = cursor.clone();
    changed.sampler_cursor = 6;
    assert!(changed.validate(20, 2, 4).is_err());
    let mut json: serde_json::Value = serde_json::from_str(&metadata).unwrap();
    json["unknown_cursor"] = 1.into();
    assert!(TrainingCursor::from_metadata(&json.to_string()).is_err());
    let tail = TrainingCursor {
        micro_step: 7,
        update_step: 4,
        accumulation_count: 0,
        ..cursor
    };
    tail.validate(7, 2, 4).unwrap();
    assert!(
        tail.validate(20, 2, 4).is_err(),
        "partial-tail update is valid only at run completion"
    );
}

#[test]
fn source_epoch_tail_flush_and_ready_full_windows_remain_reproducible() {
    let cursor = TrainingCursor {
        micro_step: 3,
        update_step: 2,
        accumulation_count: 0,
        sampler_epoch: 0,
        sampler_cursor: 3,
        rng_state: vec![1, 2],
        last_loss_bits: None,
    };
    cursor.validate(6, 2, 3).unwrap();
    let pending = TrainingCursor {
        update_step: 1,
        accumulation_count: 1,
        ..cursor.clone()
    };
    pending.validate(6, 2, 3).unwrap();
    let dropped = TrainingCursor {
        update_step: 1,
        ..cursor.clone()
    };
    assert!(
        dropped.validate(6, 2, 3).is_err(),
        "an epoch-tail gradient must not be silently dropped"
    );
    let next_epoch = TrainingCursor {
        sampler_epoch: 1,
        sampler_cursor: 0,
        ..cursor.clone()
    };
    next_epoch.validate(6, 2, 3).unwrap();
    let ready = TrainingCursor {
        micro_step: 2,
        update_step: 0,
        accumulation_count: 2,
        sampler_epoch: 0,
        sampler_cursor: 2,
        ..cursor
    };
    ready.validate(6, 2, 3).unwrap();
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
