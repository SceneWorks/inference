//! Shared per-reader load activity bridge used by MLX language and media weights.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use load_progress::{LoadActivityContext, LoadActivityFile, LoadActivityPhase};
use mlx_rs::load_progress::{LoadProgress, LoadProgressEvent};
use mlx_rs::{Array, Stream};

/// Register only the selected shards, excluding alternative checkpoints in the same directory.
pub fn register_files(paths: &[PathBuf]) {
    LoadActivityContext::capture().register_files(paths.iter().filter_map(|path| {
        std::fs::metadata(path)
            .ok()
            .map(|metadata| (path.clone(), metadata.len()))
    }));
}

type ReaderState = std::sync::Arc<Mutex<(u64, usize, Option<LoadActivityFile>)>>;

fn callback(context: LoadActivityContext, path: PathBuf, state: ReaderState) -> LoadProgress {
    LoadProgress::new(move |event| {
        let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
        match event {
            LoadProgressEvent::Started { total_bytes } => {
                if state.1 == 0 {
                    let progress = context.file_with_bytes(
                        LoadActivityPhase::Loading,
                        &path,
                        Some(total_bytes),
                        state.0,
                    );
                    state.2 = Some(progress);
                }
                state.1 += 1;
            }
            LoadProgressEvent::BytesRead { bytes, .. } => {
                state.0 = state.0.saturating_add(bytes);
                if let Some(progress) = &state.2 {
                    progress.update(state.0);
                }
            }
            LoadProgressEvent::Idle => {
                state.1 = state.1.saturating_sub(1);
                if state.1 == 0 {
                    if let Some(progress) = state.2.take() {
                        progress.finish();
                    }
                }
            }
        }
    })
}

/// Retain source attribution at open, then capture the executing request synchronously at each
/// lazy Load scheduling boundary. I/O threads use that immutable token even after TLS changes.
pub fn for_file(path: &Path) -> LoadProgress {
    use std::sync::Arc;
    let origin = LoadActivityContext::capture();
    origin.reset_file(path);
    let path = path.to_owned();
    let state: ReaderState = Arc::new(Mutex::new((0, 0, None)));
    let states = Mutex::new(HashMap::from([(origin.request_id(), state.clone())]));
    callback(origin.clone(), path.clone(), state).with_evaluation_capture(move || {
        let current = LoadActivityContext::capture();
        let context = if current.is_observed() {
            current.with_sources_from(&origin)
        } else {
            origin.clone()
        };
        // Only numeric counters remain in this reader's map after an idle interval. The captured
        // per-evaluation callback owns its request; the cache does not retain every later job.
        let state = states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(context.request_id())
            .or_insert_with(|| {
                context.reset_file(&path);
                Arc::new(Mutex::new((0, 0, None)))
            })
            .clone();
        callback(context, path.clone(), state)
    })
}

/// Safetensors metadata and lazy payloads always use MLX's CPU reader stream.
#[allow(clippy::type_complexity)]
pub fn load_with_metadata(
    path: impl AsRef<Path>,
) -> Result<(HashMap<String, Array>, HashMap<String, String>), mlx_rs::error::IoError> {
    let path = path.as_ref();
    Array::load_safetensors_with_metadata_and_progress_device(path, for_file(path), Stream::cpu())
}

pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, mlx_rs::error::IoError> {
    load_with_metadata(path).map(|(arrays, _)| arrays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use load_progress::{LoadActivity, LoadActivityObserver};
    use std::io::Write;
    use std::sync::Arc;

    fn fixture(dir: &Path) -> PathBuf {
        let path = dir.join("model.safetensors");
        let header = br#"{"a":{"dtype":"U8","shape":[1024],"data_offsets":[0,1024]},"b":{"dtype":"U8","shape":[1024],"data_offsets":[1024,2048]}}"#;
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&vec![1; 2048]).unwrap();
        path
    }
    fn observe() -> (LoadActivityObserver, Arc<Mutex<Vec<Option<LoadActivity>>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let guard = LoadActivityObserver::new(move |event| output.lock().unwrap().push(event));
        (guard, events)
    }
    fn bytes(events: &Arc<Mutex<Vec<Option<LoadActivity>>>>) -> u64 {
        events
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .map(|e| e.bytes_read)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn same_reader_keeps_separate_evaluation_requests_and_origin_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture(dir.path());
        let (origin, a) = observe();
        let arrays = load(&path).unwrap();
        let header_bytes = bytes(&a);
        assert_eq!(header_bytes, std::fs::metadata(path).unwrap().len() - 2048);
        drop(origin);
        let (job_b, b) = observe();
        mlx_rs::transforms::eval_pending_loads([&arrays["a"]]).unwrap();
        assert_eq!(bytes(&b), 1024);
        assert_eq!(bytes(&a), header_bytes);
        drop(job_b);
        let (job_c, c) = observe();
        mlx_rs::transforms::eval_pending_loads([&arrays["b"]]).unwrap();
        assert_eq!(bytes(&c), 1024);
        assert_eq!(bytes(&b), 1024);
        assert_eq!(bytes(&a), header_bytes);
        drop(job_c);

        let (origin, a) = observe();
        let arrays = load(fixture(dir.path())).unwrap();
        drop(origin);
        mlx_rs::transforms::eval_pending_loads(arrays.values()).unwrap();
        assert_eq!(
            bytes(&a),
            header_bytes + 2048,
            "without an executing observer, retain origin"
        );
        assert_eq!(a.lock().unwrap().last(), Some(&None));
    }

    #[test]
    fn unobserved_creation_still_binds_evaluating_job_and_accumulates_all_leaves() {
        let dir = tempfile::tempdir().unwrap();
        let arrays = load(fixture(dir.path())).unwrap();
        let (_job, events) = observe();
        mlx_rs::transforms::eval_pending_loads([&arrays["a"]]).unwrap();
        assert_eq!(bytes(&events), 1024);
        mlx_rs::transforms::eval_pending_loads([&arrays["b"]]).unwrap();
        assert_eq!(bytes(&events), 2048);
        assert_eq!(events.lock().unwrap().last(), Some(&None));
    }

    #[test]
    fn rebound_read_failure_clears_only_executing_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture(dir.path());
        let (origin, a) = observe();
        let arrays = load(&path).unwrap();
        let initial = a.lock().unwrap().len();
        drop(origin);
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(8)
            .unwrap();
        let (_job, b) = observe();
        assert!(mlx_rs::transforms::eval_pending_loads(arrays.values()).is_err());
        assert_eq!(a.lock().unwrap().len(), initial);
        assert_eq!(bytes(&b), 0);
        assert_eq!(b.lock().unwrap().last(), Some(&None));
    }
    #[test]
    fn overlapping_lazy_evaluations_keep_distinct_request_tokens() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        let dir = tempfile::tempdir().unwrap();
        let arrays = load(fixture(dir.path())).unwrap();
        let (started, received) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let resume = Mutex::new(resume);
        let first = AtomicBool::new(true);
        let events_b = Arc::new(Mutex::new(Vec::new()));
        let output = events_b.clone();
        let _job_b = LoadActivityObserver::new(move |event| {
            output.lock().unwrap().push(event.clone());
            if event.is_some() && first.swap(false, Ordering::Relaxed) {
                // Test-only rendezvous keeps B's read pending while C schedules another leaf.
                started.send(()).unwrap();
                resume.lock().unwrap().recv().unwrap();
            }
        });
        mlx_rs::transforms::async_eval([&arrays["a"]]).unwrap();
        received
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("B read must begin");
        let (_job_c, events_c) = observe();
        mlx_rs::transforms::async_eval([&arrays["b"]]).unwrap();
        release.send(()).unwrap();
        mlx_rs::transforms::eval(arrays.values()).unwrap();
        assert_eq!(bytes(&events_b), 1024);
        assert_eq!(bytes(&events_c), 1024);
        assert_eq!(events_b.lock().unwrap().last(), Some(&None));
        assert_eq!(events_c.lock().unwrap().last(), Some(&None));
    }
}
