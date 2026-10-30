//! Request-local observations from file readers. Callbacks enqueue work; they must not block I/O.
//! Captured contexts travel with lazy reads, never through a process-global current request.
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoadActivityPhase {
    Checking,
    Verifying,
    Loading,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadActivity {
    pub phase: LoadActivityPhase,
    pub component: String,
    pub file: PathBuf,
    /// Logical source bytes consumed, including headers; mapping an address range is not a read.
    pub bytes_read: u64,
    pub total_bytes: Option<u64>,
    /// Present only when the caller supplied the complete selected component file inventory.
    pub component_bytes_read: u64,
    pub component_total_bytes: Option<u64>,
    pub files_checked: usize,
    pub total_files: Option<usize>,
}

type Observer = Arc<dyn Fn(Option<LoadActivity>) + Send + Sync>;
#[derive(Default)]
struct Request {
    observer: Option<Observer>,
    id: u64,
    next: u64,
    active: u64,
    activities: HashMap<u64, (u64, LoadActivity)>,
    files: HashMap<PathBuf, (String, u64)>,
    consumed: HashMap<(LoadActivityPhase, PathBuf), u64>,
}

impl Request {
    fn aggregate(&self, activity: &mut LoadActivity) {
        if self.files.contains_key(&activity.file) {
            let (bytes, total) = self
                .files
                .iter()
                .filter(|(_, (component, _))| *component == activity.component)
                .fold((0u64, 0u64), |(b, t), (path, (_, size))| {
                    (
                        b.saturating_add(
                            self.consumed
                                .get(&(activity.phase, path.clone()))
                                .copied()
                                .unwrap_or(0),
                        ),
                        t.saturating_add(*size),
                    )
                });
            activity.component_bytes_read = bytes;
            activity.component_total_bytes = Some(total);
        }
    }
}

#[derive(Clone, Default)]
pub struct LoadActivityContext {
    request: Arc<Mutex<Request>>,
    emission: Arc<Mutex<()>>,
    components: Arc<Vec<(PathBuf, String)>>,
}
thread_local! { static CONTEXT: RefCell<LoadActivityContext> = RefCell::default(); }

impl LoadActivityContext {
    /// Stable, process-unique request identity without retaining its observer in reader counters.
    pub fn request_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);
        let mut request = self.request.lock().unwrap_or_else(|e| e.into_inner());
        if request.id == 0 {
            request.id = NEXT_REQUEST.fetch_add(1, Ordering::Relaxed);
        }
        request.id
    }
    pub fn capture() -> Self {
        CONTEXT.with(|v| v.borrow().clone())
    }
    pub fn is_observed(&self) -> bool {
        self.request
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .observer
            .is_some()
    }
    pub fn install(&self) -> LoadActivityObserver {
        let previous = CONTEXT.with(|v| v.replace(self.clone()));
        LoadActivityObserver {
            previous,
            _thread_bound: PhantomData,
        }
    }
    /// Bind exact file paths or directory roots to neutral LoadSpec slots. Longest match wins.
    pub fn with_components(&self, components: impl IntoIterator<Item = (PathBuf, String)>) -> Self {
        let mut roots = self.components.as_ref().clone();
        roots.extend(components.into_iter().map(|(p, c)| (absolute(&p), c)));
        Self {
            request: self.request.clone(),
            emission: self.emission.clone(),
            components: Arc::new(roots),
        }
    }
    /// Use a retained reader's source attribution and selected files with this request's callback.
    /// Cached mappings can therefore serve a new request without routing events to an old job.
    pub fn with_sources_from(&self, source: &Self) -> Self {
        let context = self.with_components(source.components.iter().cloned());
        if !Arc::ptr_eq(&self.request, &source.request) {
            let files = source
                .request
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .files
                .iter()
                .map(|(path, (_, bytes))| (path.clone(), *bytes))
                .collect::<Vec<_>>();
            context.register_files(files);
        }
        context
    }
    pub fn component(&self, path: &Path) -> String {
        let path = absolute(path);
        if let Some((_, component)) = self
            .components
            .iter()
            .filter(|(root, _)| path.starts_with(root))
            .max_by_key(|(root, _)| root.components().count())
        {
            return component.replace('_', " ");
        }
        let parts: Vec<_> = path
            .components()
            .map(|v| v.as_os_str().to_string_lossy().into_owned())
            .collect();
        if let Some(i) = parts.iter().rposition(|v| v == "snapshots") {
            if parts.len() > i + 3 {
                return parts[i + 2].replace('_', " ");
            }
        }
        // Diffusers component directories, including numbered/novel component names, need no
        // provider table. Arbitrary standalone checkpoints fall back to the filename.
        let parent = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|v| v.to_str());
        if let Some(p) = parent.filter(|v| {
            *v == "vae"
                || *v == "transformer"
                || v.contains("encoder")
                || v.contains("decoder")
                || *v == "unet"
        }) {
            return p.replace('_', " ");
        }
        path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
    /// Register a complete selected file set, not an unfiltered directory listing.
    pub fn register_files(&self, files: impl IntoIterator<Item = (PathBuf, u64)>) {
        let files = files
            .into_iter()
            .map(|(p, size)| {
                let p = absolute(&p);
                let component = self.component(&p);
                (p, (component, size))
            })
            .collect::<Vec<_>>();
        self.request
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .extend(files);
    }
    /// Begin a new read lifetime of a path (interval updates in a lazy reader stay cumulative).
    pub fn reset_file(&self, path: impl AsRef<Path>) {
        let path = absolute(path.as_ref());
        self.request
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .consumed
            .retain(|(_, p), _| p != &path);
    }
    pub fn report(&self, activity: Option<LoadActivity>) {
        let _emission = self.emission.lock().unwrap_or_else(|e| e.into_inner());
        let observer = self
            .request
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .observer
            .clone();
        if let Some(observe) = observer {
            observe(activity);
        }
    }
    pub fn file(
        &self,
        phase: LoadActivityPhase,
        path: impl AsRef<Path>,
        total_bytes: Option<u64>,
    ) -> LoadActivityFile {
        self.file_with_bytes(phase, path, total_bytes, 0)
    }
    /// Resume an interval of an existing reader without briefly publishing a false zero.
    pub fn file_with_bytes(
        &self,
        phase: LoadActivityPhase,
        path: impl AsRef<Path>,
        total_bytes: Option<u64>,
        bytes_read: u64,
    ) -> LoadActivityFile {
        let file = absolute(path.as_ref());
        let id = {
            let mut r = self.request.lock().unwrap_or_else(|e| e.into_inner());
            r.next += 1;
            r.next
        };
        let activity = LoadActivity {
            phase,
            component: self.component(&file),
            file,
            bytes_read,
            total_bytes,
            component_bytes_read: 0,
            component_total_bytes: None,
            files_checked: 0,
            total_files: None,
        };
        let handle = LoadActivityFile {
            inner: Arc::new(FileState {
                context: self.clone(),
                id,
                state: Mutex::new(Some(activity)),
            }),
        };
        handle.update(bytes_read);
        handle
    }
}
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Thread-bound installation. Drop restores the enclosing request, including during unwinding.
#[must_use]
pub struct LoadActivityObserver {
    previous: LoadActivityContext,
    _thread_bound: PhantomData<Rc<()>>,
}
impl LoadActivityObserver {
    pub fn new(observe: impl Fn(Option<LoadActivity>) + Send + Sync + 'static) -> Self {
        LoadActivityContext {
            request: Arc::new(Mutex::new(Request {
                observer: Some(Arc::new(observe)),
                ..Request::default()
            })),
            emission: Arc::default(),
            components: Arc::default(),
        }
        .install()
    }
}
impl Drop for LoadActivityObserver {
    fn drop(&mut self) {
        CONTEXT.with(|v| {
            v.replace(std::mem::take(&mut self.previous));
        });
    }
}

struct FileState {
    context: LoadActivityContext,
    id: u64,
    state: Mutex<Option<LoadActivity>>,
}
impl FileState {
    fn finish(&self) {
        let _emission = self
            .context
            .emission
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .is_none()
        {
            return;
        }
        let (observer, activity) = {
            let mut r = self
                .context
                .request
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            r.activities.remove(&self.id);
            if r.active != self.id {
                return;
            }
            let mut next = r
                .activities
                .iter()
                .max_by_key(|(_, (order, _))| order)
                .map(|(id, (_, activity))| (*id, activity.clone()));
            if let Some((_, activity)) = &mut next {
                r.aggregate(activity);
            }
            r.active = next.as_ref().map(|(id, _)| *id).unwrap_or(0);
            (r.observer.clone(), next.map(|(_, activity)| activity))
        };
        if let Some(observe) = observer {
            observe(activity);
        }
    }
}
impl Drop for FileState {
    fn drop(&mut self) {
        self.finish();
    }
}
/// Share with lazy/background callbacks. Last-drop and explicit finish both end the phase.
#[derive(Clone)]
pub struct LoadActivityFile {
    inner: Arc<FileState>,
}
impl LoadActivityFile {
    pub fn update(&self, bytes_read: u64) {
        self.change(|v| v.bytes_read = bytes_read);
    }
    pub fn add_bytes(&self, bytes: u64) {
        self.change(|v| v.bytes_read = v.bytes_read.saturating_add(bytes));
    }
    pub fn checked(&self, completed: usize, total: Option<usize>) {
        self.change(|v| {
            v.files_checked = completed;
            v.total_files = total;
        });
    }
    fn change(&self, change: impl FnOnce(&mut LoadActivity)) {
        let _emission = self
            .inner
            .context
            .emission
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let activity = {
            let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            let Some(activity) = state.as_mut() else {
                return;
            };
            change(activity);
            activity.clone()
        };
        let mut activity = activity;
        let observer = {
            let mut r = self
                .inner
                .context
                .request
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            r.active = self.inner.id;
            let consumed = r
                .consumed
                .entry((activity.phase, activity.file.clone()))
                .or_default();
            *consumed = (*consumed).max(activity.bytes_read);
            r.aggregate(&mut activity);
            r.next += 1;
            let order = r.next;
            r.activities
                .insert(self.inner.id, (order, activity.clone()));
            r.observer.clone()
        };
        if let Some(observe) = observer {
            observe(Some(activity));
        }
    }
    pub fn finish(&self) {
        self.inner.finish();
    }
}
/// Synchronous file-read scope; use a cloned LoadActivityFile for asynchronous/lazy work.
pub type LoadActivityScope = LoadActivityFile;
impl LoadActivityFile {
    pub fn file(
        phase: LoadActivityPhase,
        path: impl AsRef<Path>,
        total_bytes: Option<u64>,
    ) -> Self {
        LoadActivityContext::capture().file(phase, path, total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn panic_ends_file_activity_and_restores_the_enclosing_request() {
        let outer = Arc::new(Mutex::new(Vec::new()));
        let out = outer.clone();
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let inner = Arc::new(Mutex::new(Vec::new()));
        let out = inner.clone();
        let result = std::panic::catch_unwind(move || {
            let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
            let _file = LoadActivityScope::file(LoadActivityPhase::Verifying, "inner", Some(10));
            panic!("source reader failed");
        });
        assert!(result.is_err());
        assert_eq!(inner.lock().unwrap().last(), Some(&None));
        let file = LoadActivityScope::file(LoadActivityPhase::Loading, "outer", Some(10));
        file.update(10);
        file.finish();
        let events = outer.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert!(events
            .iter()
            .flatten()
            .all(|v| v.phase == LoadActivityPhase::Loading));
        assert_eq!(events.last(), Some(&None));
    }

    #[test]
    fn captured_context_isolated_from_other_requests_and_unwinds() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let out = events.clone();
        let context = {
            let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
            LoadActivityContext::capture()
        };
        let _other = LoadActivityObserver::new(|_| panic!("request leaked"));
        std::thread::spawn(move || {
            let _installed = context.install();
            let file = LoadActivityScope::file(
                LoadActivityPhase::Loading,
                "text_encoder/model.safetensors",
                Some(100),
            );
            file.update(100);
        })
        .join()
        .unwrap();
        let events = events.lock().unwrap();
        assert_eq!(events[0].as_ref().unwrap().component, "text encoder");
        assert_eq!(events[1].as_ref().unwrap().bytes_read, 100);
        assert_eq!(events.last(), Some(&None));
    }
    #[test]
    fn restoring_a_shard_uses_the_latest_component_total_and_request_ids_are_stable() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let out = events.clone();
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let context = LoadActivityContext::capture()
            .with_components([(PathBuf::from("component"), "tower".into())]);
        assert_eq!(context.request_id(), context.clone().request_id());
        assert_ne!(
            context.request_id(),
            LoadActivityContext::default().request_id()
        );
        context.register_files([
            (PathBuf::from("component/a"), 100),
            (PathBuf::from("component/b"), 100),
        ]);
        let a = context.file(LoadActivityPhase::Loading, "component/a", Some(100));
        a.update(27);
        let b = context.file(LoadActivityPhase::Loading, "component/b", Some(100));
        b.update(50);
        b.finish();
        let events = events.lock().unwrap();
        let restored = events.last().unwrap().as_ref().unwrap();
        assert_eq!(restored.component_bytes_read, 77);
        assert_eq!(restored.component_total_bytes, Some(200));
        drop(events);
        a.finish();
    }

    #[test]
    fn ending_latest_parallel_read_restores_still_running_source() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let out = events.clone();
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let a = LoadActivityScope::file(LoadActivityPhase::Loading, "a", Some(100));
        a.update(27);
        let b = LoadActivityScope::file(LoadActivityPhase::Loading, "b", Some(100));
        b.finish();
        assert_eq!(
            events
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .as_ref()
                .unwrap()
                .bytes_read,
            27
        );
        a.finish();
        assert_eq!(events.lock().unwrap().last(), Some(&None));
    }

    #[test]
    fn completed_older_read_cannot_clear_newer_activity() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let out = events.clone();
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let a = LoadActivityScope::file(LoadActivityPhase::Loading, "a", None);
        let b = LoadActivityScope::file(LoadActivityPhase::Loading, "b", None);
        a.finish();
        assert!(events.lock().unwrap().last().unwrap().is_some());
        b.finish();
        assert_eq!(events.lock().unwrap().last(), Some(&None));
    }
}
