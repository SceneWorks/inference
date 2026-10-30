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
    published: u64,
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
    emission: Arc<Mutex<()>>,
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
        let path =
            std::path::absolute(path.as_ref()).unwrap_or_else(|_| path.as_ref().to_path_buf());
        context.reset_file(&path);
        Self {
            context,
            path,
            total,
            ranges: Arc::default(),
            emission: Arc::default(),
        }
    }
    pub fn operation(&self) -> ReadOperation {
        let _emission = self.emission.lock().unwrap_or_else(|e| e.into_inner());
        let bytes = self.ranges.lock().unwrap_or_else(|e| e.into_inner()).bytes;
        let activity =
            self.context
                .file_with_bytes(LoadActivityPhase::Loading, &self.path, self.total, bytes);
        let id = {
            let mut r = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
            r.next += 1;
            let id = r.next;
            r.active = Some((id, activity.clone()));
            r.published = bytes;
            id
        };
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
        let _emission = self.emission.lock().unwrap_or_else(|e| e.into_inner());
        self.record_inner(activity, start, len);
    }
    fn record_read(&self, start: u64, len: u64) {
        if len == 0 {
            return;
        }
        let _emission = self.emission.lock().unwrap_or_else(|e| e.into_inner());
        let (active, bytes) = {
            let r = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
            (r.active.as_ref().map(|(_, h)| h.clone()), r.bytes)
        };
        let transient = active.is_none();
        let activity = active.unwrap_or_else(|| {
            self.context
                .file_with_bytes(LoadActivityPhase::Loading, &self.path, self.total, bytes)
        });
        self.record_inner(&activity, start, len);
        if transient {
            let bytes = self.ranges.lock().unwrap_or_else(|e| e.into_inner()).bytes;
            activity.update(bytes);
            activity.finish();
        }
    }
    fn record_inner(&self, activity: &LoadActivityFile, start: u64, len: u64) {
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
            // Primitive header parsers can perform millions of tiny reads. Keep exact interval
            // accounting but bound observer traffic; operation completion flushes the remainder.
            if r.bytes.saturating_sub(r.published) < 64 * 1024 {
                return;
            }
            r.published = r.bytes;
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
        let _emission = self
            .source
            .emission
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let flush = {
            let mut r = self.source.ranges.lock().unwrap_or_else(|e| e.into_inner());
            if r.active.as_ref().is_some_and(|(id, _)| *id == self.id) {
                r.active = None;
            }
            // A newer overlapping operation may already have finished and cleared `active`.
            // The remaining operation still owns its actual tail reads and must flush them.
            // Preserve a different live owner while publishing this source's cumulative extent.
            let changed = r.bytes != r.published;
            r.published = r.bytes;
            changed.then_some(r.bytes)
        };
        if let Some(bytes) = flush {
            self.activity.update(bytes);
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
    pub fn get_ref(&self) -> &R {
        &self.inner
    }
    pub fn begin(&mut self) {
        self.operation = Some(self.source.operation());
    }
    pub fn finish(&mut self) {
        self.operation = None;
    }
}
impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let result = self.inner.read(buf);
        if let Ok(n) = result {
            self.source.record_read(self.position, n as u64);
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
    fn overlapping_same_source_operations_flush_and_restore_cumulative_bytes() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let out = events.clone();
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let source = SourceProgress::new("shared.weights", Some(128));
        let older = source.operation();
        older.record(0, 16);
        let newer = source.operation();
        newer.record(16, 32);
        drop(newer);
        assert_eq!(
            events
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .as_ref()
                .unwrap()
                .bytes_read,
            48,
            "restoring the older live operation must retain the newer consumed extent"
        );
        older.record(48, 16);
        drop(older);
        let events = events.lock().unwrap();
        let counts: Vec<_> = events.iter().flatten().map(|v| v.bytes_read).collect();
        assert!(
            counts.windows(2).all(|v| v[0] <= v[1]),
            "overlap regressed: {counts:?}"
        );
        assert_eq!(
            counts.last(),
            Some(&64),
            "last surviving operation must flush its actual tail bytes"
        );
        assert_eq!(events.last(), Some(&None));
    }

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
        reader.begin();
        reader.seek(SeekFrom::Start(0)).unwrap();
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
    #[test]
    fn primitive_reads_are_coalesced_and_completion_flushes_exact_bytes() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
        let source = SourceProgress::new("header.gguf", Some(1000));
        {
            let operation = source.operation();
            for offset in 0..1000 {
                operation.record(offset, 1);
            }
            assert_eq!(events.lock().unwrap().len(), 1);
        }
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[1].as_ref().unwrap().bytes_read, 1000);
        assert_eq!(events.last(), Some(&None));
    }
}
