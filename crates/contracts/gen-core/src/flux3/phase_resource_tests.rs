use super::*;
use std::sync::{Arc, Mutex, Weak};

struct Resource {
    marker: Arc<()>,
    events: Arc<Mutex<Vec<&'static str>>>,
}
impl Drop for Resource {
    fn drop(&mut self) {
        self.events.lock().unwrap().push("drop-owner");
    }
}
fn resource(events: &Arc<Mutex<Vec<&'static str>>>) -> (Resource, Weak<()>) {
    let marker = Arc::new(());
    let weak = Arc::downgrade(&marker);
    (
        Resource {
            marker,
            events: events.clone(),
        },
        weak,
    )
}

#[test]
fn exclusive_work_fences_and_really_drops_before_the_next_phase() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let (owner, weak) = resource(&events);
    let cancel = CancelFlag::new();
    let value = slot
        .with_request(
            true,
            true,
            &cancel,
            || {
                events.lock().unwrap().push("load");
                Ok(owner)
            },
            |model| {
                assert!(model.is_some());
                events.lock().unwrap().push("caption");
                Ok("exact caption".to_owned())
            },
            |model| {
                assert_eq!(Arc::strong_count(&model.marker), 1);
                events.lock().unwrap().push("owned-fence");
                Ok(())
            },
            || events.lock().unwrap().push("after-drop"),
        )
        .unwrap();
    events.lock().unwrap().push("image-phase");
    assert_eq!(value, "exact caption");
    assert!(!slot.is_loaded().unwrap());
    assert!(weak.upgrade().is_none());
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            "load",
            "caption",
            "owned-fence",
            "drop-owner",
            "after-drop",
            "image-phase"
        ]
    );
}

#[test]
fn bypass_never_loads_and_evicts_a_prior_resident_owner() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let cancel = CancelFlag::new();
    let (owner, weak) = resource(&events);
    slot.with_request(
        true,
        false,
        &cancel,
        || Ok(owner),
        |_| Ok(()),
        |_| panic!("resident success must not add a release fence"),
        || panic!("not released"),
    )
    .unwrap();
    assert!(slot.is_loaded().unwrap());
    assert!(weak.upgrade().is_some());
    let original = "  unchanged\nsource prompt  ";
    let value = slot
        .with_request(
            false,
            false,
            &cancel,
            || panic!("bypass must never load"),
            |owner| {
                assert!(owner.is_none());
                Ok(original)
            },
            |_| {
                events.lock().unwrap().push("owned-fence");
                Ok(())
            },
            || events.lock().unwrap().push("after-drop"),
        )
        .unwrap();
    assert_eq!(value, original);
    assert!(!slot.is_loaded().unwrap());
    assert!(weak.upgrade().is_none());
    assert_eq!(
        *events.lock().unwrap(),
        vec!["owned-fence", "drop-owner", "after-drop"]
    );
}

#[test]
fn fence_failure_keeps_actual_owner_and_preserves_primary_cancel_or_error() {
    for cancelled in [false, true] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let slot = PhaseResourceSlot::default();
        let cancel = CancelFlag::new();
        let (owner, weak) = resource(&events);
        let result: Result<()> = slot.with_request(
            true,
            true,
            &cancel,
            || Ok(owner),
            |_| {
                if cancelled {
                    Err(Error::Canceled)
                } else {
                    Err(Error::Msg("primary error".into()))
                }
            },
            |_| {
                events.lock().unwrap().push("failed-fence");
                Err(Error::Unsupported("injected unfenced backend".into()))
            },
            || panic!("an unfenced owner must not be reported released"),
        );
        if cancelled {
            assert!(matches!(result, Err(Error::Canceled)));
        } else {
            assert!(matches!(result,Err(Error::Msg(message))if message=="primary error"));
        }
        assert!(slot.is_loaded().unwrap());
        assert!(weak.upgrade().is_some());
        assert_eq!(*events.lock().unwrap(), vec!["failed-fence"]);
        slot.release_cached(
            |_| {
                events.lock().unwrap().push("retry-fence");
                Ok(())
            },
            || events.lock().unwrap().push("after-drop"),
        )
        .unwrap();
        assert!(!slot.is_loaded().unwrap());
        assert!(weak.upgrade().is_none());
        assert_eq!(
            *events.lock().unwrap(),
            vec!["failed-fence", "retry-fence", "drop-owner", "after-drop"]
        );
    }
}

#[test]
fn successful_work_with_unsupported_fence_is_not_successful_eviction() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let cancel = CancelFlag::new();
    let (owner, weak) = resource(&events);
    let result = slot.with_request(
        true,
        true,
        &cancel,
        || Ok(owner),
        |_| Ok(17),
        |_| Err(Error::Unsupported("no owned fence".into())),
        || panic!("must retain"),
    );
    assert!(matches!(result, Err(Error::Unsupported(_))));
    assert!(slot.is_loaded().unwrap());
    assert!(weak.upgrade().is_some());
    assert!(events.lock().unwrap().is_empty());
}

#[test]
fn cancellation_during_load_fences_and_drops_without_caption_or_image_work() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let cancel = CancelFlag::new();
    let (owner, weak) = resource(&events);
    let result: Result<()> = slot.with_request(
        true,
        true,
        &cancel,
        || {
            cancel.cancel();
            Ok(owner)
        },
        |_| panic!("cancelled before caption"),
        |_| {
            events.lock().unwrap().push("owned-fence");
            Ok(())
        },
        || events.lock().unwrap().push("after-drop"),
    );
    assert!(matches!(result, Err(Error::Canceled)));
    assert!(weak.upgrade().is_none());
    assert!(!slot.is_loaded().unwrap());
    assert_eq!(
        *events.lock().unwrap(),
        vec!["owned-fence", "drop-owner", "after-drop"]
    );
}

#[test]
fn cancellation_after_successful_caption_releases_even_resident_policy() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let cancel = CancelFlag::new();
    let (owner, weak) = resource(&events);
    let result = slot.with_request(
        true,
        false,
        &cancel,
        || Ok(owner),
        |_| {
            cancel.cancel();
            Ok("must not publish")
        },
        |_| {
            events.lock().unwrap().push("owned-fence");
            Ok(())
        },
        || events.lock().unwrap().push("after-drop"),
    );
    assert!(matches!(result, Err(Error::Canceled)));
    assert!(weak.upgrade().is_none());
    assert!(!slot.is_loaded().unwrap());
    assert_eq!(
        *events.lock().unwrap(),
        vec!["owned-fence", "drop-owner", "after-drop"]
    );
}

#[test]
fn warm_resident_to_exclusive_drops_old_owner_before_new_load() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let cancel = CancelFlag::new();
    let (old, old_weak) = resource(&events);
    slot.with_request(
        true,
        false,
        &cancel,
        || Ok(old),
        |_| Ok(()),
        |_| panic!("resident kept"),
        || panic!("resident kept"),
    )
    .unwrap();
    let (new, new_weak) = resource(&events);
    slot.with_request(
        true,
        true,
        &cancel,
        || {
            assert!(old_weak.upgrade().is_none());
            events.lock().unwrap().push("new-load");
            Ok(new)
        },
        |_| {
            events.lock().unwrap().push("caption");
            Ok(())
        },
        |_| {
            events.lock().unwrap().push("owned-fence");
            Ok(())
        },
        || events.lock().unwrap().push("after-drop"),
    )
    .unwrap();
    assert!(new_weak.upgrade().is_none());
    assert!(!slot.is_loaded().unwrap());
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            "owned-fence",
            "drop-owner",
            "after-drop",
            "new-load",
            "caption",
            "owned-fence",
            "drop-owner",
            "after-drop"
        ]
    );
}

/// A panic inside the caption phase poisons the ownership lock; the slot recovers, and the owner
/// the panicking phase left behind is fenced and dropped before the next request reuses anything.
///
/// *Mutations that red this:* surfacing the poison as a permanent error, or recovering without
/// marking the surviving owner `pending_release`.
#[test]
fn a_panicking_phase_poisons_nothing_and_its_owner_is_released_first() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slot = PhaseResourceSlot::default();
    let (owner, weak) = resource(&events);
    let cancel = CancelFlag::new();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = slot.with_request(
            true,
            false,
            &cancel,
            || Ok(owner),
            |_| -> Result<()> { panic!("caption phase panicked") },
            |_| Ok(()),
            || {},
        );
    }));
    assert!(panicked.is_err());
    assert!(
        slot.is_loaded().unwrap(),
        "the panicking phase left its owner"
    );
    assert!(
        slot.cleanup_pending().unwrap(),
        "recovery marks it for release"
    );
    events.lock().unwrap().clear();
    let (next, _) = resource(&events);
    let value = slot
        .with_request(
            true,
            false,
            &cancel,
            || {
                events.lock().unwrap().push("load");
                Ok(next)
            },
            |_| Ok(7),
            |_| {
                events.lock().unwrap().push("fence");
                Ok(())
            },
            || {},
        )
        .unwrap();
    assert_eq!(value, 7);
    assert!(
        weak.upgrade().is_none(),
        "the surviving owner was really dropped"
    );
    assert_eq!(*events.lock().unwrap(), vec!["fence", "drop-owner", "load"]);
}
