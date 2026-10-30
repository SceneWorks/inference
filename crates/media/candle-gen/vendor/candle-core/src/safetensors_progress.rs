//! Logical bytes consumed from mapped safetensors. Mapping alone never reports payload reads.
//! Tracks the ranges used by Candle's shared tensor conversion paths, including CPU and GPU copy.
use load_progress::{LoadActivityContext, LoadActivityFile, LoadActivityPhase};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

pub(super) struct MappedRead {
    path: PathBuf,
    start: usize,
    len: usize,
    context: LoadActivityContext,
    ranges: Mutex<HashMap<LoadActivityPhase, (BTreeMap<usize, usize>, u64)>>,
}
fn mappings() -> &'static Mutex<Vec<Weak<MappedRead>>> {
    static MAPPINGS: OnceLock<Mutex<Vec<Weak<MappedRead>>>> = OnceLock::new();
    MAPPINGS.get_or_init(Mutex::default)
}
impl MappedRead {
    pub(super) fn new(path: &Path, bytes: &[u8]) -> Arc<Self> {
        let context = LoadActivityContext::capture();
        context.reset_file(path);
        let read = Arc::new(Self {
            path: path.to_path_buf(),
            start: bytes.as_ptr() as usize,
            len: bytes.len(),
            context,
            ranges: Mutex::default(),
        });
        let mut mappings = mappings().lock().unwrap_or_else(|e| e.into_inner());
        mappings.retain(|v| v.strong_count() > 0);
        mappings.push(Arc::downgrade(&read));
        read
    }
    fn consumed(&self, phase: LoadActivityPhase) -> u64 {
        self.ranges
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&phase)
            .map(|v| v.1)
            .unwrap_or(0)
    }
    fn add(&self, phase: LoadActivityPhase, mut start: usize, mut end: usize) -> u64 {
        let mut phases = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
        let state = phases.entry(phase).or_default();
        if let Some((&a, &b)) = state.0.range(..=start).next_back() {
            if b >= start {
                start = a;
                end = end.max(b);
                state.0.remove(&a);
                state.1 -= (b - a) as u64;
            }
        }
        while let Some((&a, &b)) = state.0.range(start..).next() {
            if a > end {
                break;
            }
            end = end.max(b);
            state.0.remove(&a);
            state.1 -= (b - a) as u64;
        }
        state.0.insert(start, end);
        state.1 += (end - start) as u64;
        state.1
    }
    fn begin(self: &Arc<Self>, phase: LoadActivityPhase, start: usize, end: usize) -> Consumption {
        // A cached mapping reused by a later request belongs to that request's current observer.
        // Background materialization with no installed observer retains its originating context.
        let current = LoadActivityContext::capture();
        let context = if current.is_observed() {
            current.with_sources_from(&self.context)
        } else {
            self.context.clone()
        };
        let activity = context.file(phase, &self.path, Some(self.len as u64));
        activity.update(self.consumed(phase));
        Consumption {
            phase,
            read: self.clone(),
            start,
            end,
            activity,
        }
    }
    pub(super) fn header(self: &Arc<Self>, data: &[u8]) -> Consumption {
        let len = data
            .get(..8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .unwrap_or(0);
        let end = usize::try_from(len)
            .unwrap_or(usize::MAX)
            .saturating_add(8)
            .min(data.len());
        self.begin(LoadActivityPhase::Loading, 0, end)
    }
}
pub(super) struct Consumption {
    phase: LoadActivityPhase,
    read: Arc<MappedRead>,
    start: usize,
    end: usize,
    activity: LoadActivityFile,
}
impl Consumption {
    pub(super) fn complete(&self) {
        self.activity
            .update(self.read.add(self.phase, self.start, self.end));
    }
}
impl Drop for Consumption {
    fn drop(&mut self) {
        self.activity.finish();
    }
}

pub(super) fn consuming(data: &[u8]) -> Option<Consumption> {
    consuming_phase(data, LoadActivityPhase::Loading)
}
pub(super) fn consuming_phase(data: &[u8], phase: LoadActivityPhase) -> Option<Consumption> {
    if data.is_empty() {
        return None;
    }
    let start = data.as_ptr() as usize;
    let read = mappings()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter_map(Weak::upgrade)
        .find(|read| {
            start >= read.start
                && start.saturating_add(data.len()) <= read.start.saturating_add(read.len)
        });
    read.map(|read| {
        let offset = start - read.start;
        read.begin(phase, offset, offset + data.len())
    })
}

pub(super) fn read_file(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let total = file.metadata()?.len();
    let context = LoadActivityContext::capture();
    context.reset_file(path);
    let activity = context.file(LoadActivityPhase::Loading, path, Some(total));
    let capacity = usize::try_from(total)
        .map_err(|_| std::io::Error::other("file size exceeds address space"))?;
    let mut data = Vec::with_capacity(capacity);
    let mut chunk = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..count]);
        activity.update(data.len() as u64);
    }
    Ok(data)
}
