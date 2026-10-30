//! Shared logical source-byte accounting for seekable readers and mapped-file slices.
//!
//! A source retains the originating request across lazy reads. Operations delimit actual I/O;
//! dropping an operation clears its phase even on an error. Seeks and overlapping/repeated reads
//! never inflate the unique source-byte count.
use crate::{LoadActivityContext, LoadActivityFile, LoadActivityPhase};
use std::collections::BTreeMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Ranges {
    spans: BTreeMap<u64, u64>,
    bytes: u64,
    active: Option<(u64, LoadActivityFile)>,
    next: u64,
}

/// One opened source, including its captured request and cumulative consumed ranges.
#[derive(Clone)]
pub struct SourceProgress {
    context: LoadActivityContext,
    path: PathBuf,
    total: Option<u64>,
    ranges: Arc<Mutex<Ranges>>,
}
impl std::fmt::Debug for SourceProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceProgress")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}
impl SourceProgress {
    pub fn new(path: impl AsRef<Path>, total: Option<u64>) -> Self {
        Self::with_context(LoadActivityContext::capture(), path, total)
    }
    pub fn with_context(
        context: LoadActivityContext,
        path: impl AsRef<Path>,
        total: Option<u64>,
    ) -> Self {
        let path = path.as_ref().to_path_buf();
        context.reset_file(&path);
        Self {
            context,
            path,
            total,
            ranges: Arc::default(),
        }
    }
    pub fn operation(&self) -> ReadOperation {
        let activity = self
            .context
            .file(LoadActivityPhase::Loading, &self.path, self.total);
        let (bytes, id) = {
            let mut r = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
            r.next += 1;
            let id = r.next;
            r.active = Some((id, activity.clone()));
            (r.bytes, id)
        };
        activity.update(bytes);
        ReadOperation {
            source: self.clone(),
            activity,
            id,
        }
    }
    pub fn reader<R: Read + Seek>(&self, mut inner: R) -> io::Result<ProgressReader<R>> {
        let position = inner.stream_position()?;
        Ok(ProgressReader {
            inner,
            position,
            source: self.clone(),
            operation: None,
        })
    }
}

impl SourceProgress {
    fn record(&self, activity: &LoadActivityFile, start: u64, len: u64) {
        if len == 0 || !self.context.is_observed() {
            return;
        }
        let mut end = start.saturating_add(len);
        if let Some(total) = self.total {
            end = end.min(total);
        }
        if start >= end {
            return;
        }
        let bytes = {
            let mut r = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
            let mut start = start;
            if let Some((&s, &e)) = r.spans.range(..=start).next_back() {
                if e >= start {
                    start = s;
                    end = end.max(e);
                }
            }
            loop {
                let next = r.spans.range(start..).next().map(|(&s, &e)| (s, e));
                match next {
                    Some((s, e)) if s <= end => {
                        r.spans.remove(&s);
                        r.bytes -= e - s;
                        end = end.max(e);
                    }
                    _ => break,
                }
            }
            r.spans.insert(start, end);
            r.bytes += end - start;
            r.bytes
        };
        activity.update(bytes);
    }
}

/// An actual synchronous read interval. Last drop clears the loading phase.
pub struct ReadOperation {
    source: SourceProgress,
    activity: LoadActivityFile,
    id: u64,
}
impl ReadOperation {
    /// Record a successfully consumed slice of a mapped source (never the mmap size itself).
    pub fn record(&self, start: u64, len: u64) {
        self.source.record(&self.activity, start, len);
    }
}
impl Drop for ReadOperation {
    fn drop(&mut self) {
        {
            let mut r = self.source.ranges.lock().unwrap_or_else(|e| e.into_inner());
            if r.active.as_ref().is_some_and(|(id, _)| *id == self.id) {
                r.active = None;
            }
        }
        self.activity.finish();
    }
}

/// An underlying reader which counts successful bytes, including partial reads before an error.
/// Call `begin`/`finish` around each logical archive operation when retaining it for lazy reads.
pub struct ProgressReader<R> {
    inner: R,
    position: u64,
    source: SourceProgress,
    operation: Option<ReadOperation>,
}
impl<R> ProgressReader<R> {
    pub fn begin(&mut self) {
        self.operation = Some(self.source.operation());
    }
    pub fn finish(&mut self) {
        self.operation = None;
    }
}
impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // A transient operation also permits persistent ZIP readers: each actual read is observed,
        // while parsing/compute between archive calls never leaves a stale loading phase.
        let active = self
            .source
            .ranges
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active
            .as_ref()
            .map(|(_, h)| h.clone());
        let transient = active.is_none().then(|| self.source.operation());
        let result = self.inner.read(buf);
        if let Ok(n) = result {
            if let Some(operation) = transient.as_ref() {
                operation.record(self.position, n as u64);
            } else if let Some(activity) = active {
                // Borrow the existing operation's handle without ending its lifetime.
                self.source.record(&activity, self.position, n as u64);
            }
            self.position = self.position.saturating_add(n as u64);
        }
        result
    }
}
impl<R: Seek> Seek for ProgressReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let position = self.inner.seek(pos)?;
        self.position = position;
        Ok(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoadActivityObserver;
    #[test]
    fn reads_count_unique_ranges_not_seeks_and_clear_on_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
        let source = SourceProgress::new("transformer/model.gguf", Some(16));
        let mut reader = source.reader(io::Cursor::new(vec![0; 16])).unwrap();
        reader.begin();
        let mut buf = [0; 4];
        reader.read_exact(&mut buf).unwrap();
        reader.seek(SeekFrom::Start(8)).unwrap();
        reader.read_exact(&mut buf).unwrap();
        reader.seek(SeekFrom::Start(2)).unwrap();
        reader.read_exact(&mut buf).unwrap();
        reader.finish();
        let events = events.lock().unwrap();
        let counts: Vec<_> = events
            .iter()
            .filter_map(|v| v.as_ref().map(|v| v.bytes_read))
            .collect();
        assert_eq!(counts.last(), Some(&10));
        assert!(counts.windows(2).all(|v| v[0] <= v[1]));
        assert_eq!(events.last(), Some(&None));
    }
    #[test]
    fn lazy_source_keeps_request_and_error_clears_phase() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let source = {
            let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
            SourceProgress::new("vae/model.pth", Some(2))
        };
        let _other = LoadActivityObserver::new(|_| panic!("wrong request"));
        std::thread::spawn(move || {
            let mut reader = source.reader(io::Cursor::new(vec![0; 2])).unwrap();
            // read_exact consumes two bytes before UnexpectedEof; both are accounted for.
            assert!(reader.read_exact(&mut [0; 4]).is_err());
        })
        .join()
        .unwrap();
        let events = events.lock().unwrap();
        assert!(events.iter().flatten().any(|v| v.bytes_read == 2));
        assert_eq!(events.last(), Some(&None));
    }
}
