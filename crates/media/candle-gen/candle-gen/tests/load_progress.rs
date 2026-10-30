//! File I/O and tensor copies only: no learned module forward or inference.
use candle_core::{
    safetensors::{self, Load, MmapedSafetensors},
    Device, Tensor,
};
use gen_core::{LoadActivity, LoadActivityContext, LoadActivityObserver, LoadActivityPhase};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn fixture(path: &Path) -> u64 {
    let tensors: HashMap<_, _> = [
        (
            "a",
            Tensor::from_vec(vec![1u8; 41], 41, &Device::Cpu).unwrap(),
        ),
        (
            "b",
            Tensor::from_vec(vec![2u8; 73], 73, &Device::Cpu).unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    safetensors::save(&tensors, path).unwrap();
    std::fs::metadata(path).unwrap().len()
}
fn max_bytes(events: &[Option<LoadActivity>]) -> u64 {
    events
        .iter()
        .flatten()
        .map(|v| v.bytes_read)
        .max()
        .unwrap_or(0)
}
#[test]
fn safetensors_buffered_and_mapped_loads_count_consumed_file_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let component = dir.path().join("text_encoder");
    std::fs::create_dir(&component).unwrap();
    let path = component.join("model.safetensors");
    let size = fixture(&path);
    let events = Arc::new(Mutex::new(Vec::new()));
    let out = events.clone();
    let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
    let loaded = safetensors::load(&path, &Device::Cpu).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(max_bytes(&events.lock().unwrap()), size);
    assert_eq!(events.lock().unwrap().last(), Some(&None));
    events.lock().unwrap().clear();
    // SAFETY: fixture is immutable for the lifetime of the mapping.
    let mapped = unsafe { MmapedSafetensors::new(&path) }.unwrap();
    let header = max_bytes(&events.lock().unwrap());
    assert!(header > 8 && header < size, "mapping is not payload I/O");
    let _unused_view = mapped.get("a").unwrap();
    assert_eq!(
        max_bytes(&events.lock().unwrap()),
        header,
        "view lookup doesn't consume payload"
    );
    for (_, view) in mapped.tensors() {
        view.load(&Device::Cpu).unwrap();
    }
    let reports = events.lock().unwrap();
    assert_eq!(max_bytes(&reports), size);
    assert!(reports
        .iter()
        .flatten()
        .all(|v| v.component == "text encoder"
            && v.phase == LoadActivityPhase::Loading
            && v.total_bytes == Some(size)));
    assert_eq!(reports.last(), Some(&None));
}
#[test]
fn explicit_shards_aggregate_and_repeated_views_do_not_count_twice() {
    let dir = tempfile::tempdir().unwrap();
    let component = dir.path().join("transformer");
    std::fs::create_dir(&component).unwrap();
    let a = component.join("first.safetensors");
    let b = component.join("second.safetensors");
    let size = fixture(&a) + fixture(&b);
    let events = Arc::new(Mutex::new(Vec::new()));
    let out = events.clone();
    let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
    // SAFETY: fixtures remain immutable while mapped. Direct tensors() covers both files even
    // though duplicate key routing would intentionally select only the later shard.
    let mapped = unsafe { MmapedSafetensors::multi(&[a, b]) }.unwrap();
    for (_, view) in mapped.tensors() {
        view.load(&Device::Cpu).unwrap();
    }
    let max = events
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .map(|v| v.component_bytes_read)
        .max()
        .unwrap();
    assert_eq!(max, size);
    mapped.load("a", &Device::Cpu).unwrap();
    let reports = events.lock().unwrap();
    assert_eq!(
        reports
            .iter()
            .flatten()
            .map(|v| v.component_bytes_read)
            .max(),
        Some(size)
    );
    assert!(reports
        .iter()
        .flatten()
        .all(|v| v.component_total_bytes == Some(size)));
    assert!(LoadActivityContext::capture().is_observed());
}

#[test]
fn cached_mapped_reader_routes_consumption_to_the_evaluating_request_and_retains_slots() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("selected.safetensors");
    let size = fixture(&path);
    let a = Arc::new(Mutex::new(Vec::new()));
    let out = a.clone();
    let mapped = {
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let context = LoadActivityContext::capture()
            .with_components([(path.clone(), "auxiliary tower".into())]);
        let _attribution = context.install();
        // SAFETY: fixture is immutable while any mapping or view exists.
        unsafe { MmapedSafetensors::new(&path) }.unwrap()
    };
    let header = max_bytes(&a.lock().unwrap());
    let prior_a = a.lock().unwrap().len();
    let b = Arc::new(Mutex::new(Vec::new()));
    let out = b.clone();
    let context = {
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        LoadActivityContext::capture()
    };
    std::thread::spawn(move || {
        let _observer = context.install();
        for (_, view) in mapped.tensors() {
            view.load(&Device::Cpu).unwrap();
        }
    })
    .join()
    .unwrap();
    assert_eq!(
        a.lock().unwrap().len(),
        prior_a,
        "B's evaluation cannot post into A"
    );
    let reports = b.lock().unwrap();
    assert_eq!(
        max_bytes(&reports),
        size - header,
        "B only consumed payload; A already parsed the header"
    );
    assert!(reports
        .iter()
        .flatten()
        .all(|v| v.component == "auxiliary tower"));
    assert_eq!(reports.last(), Some(&None));
}
