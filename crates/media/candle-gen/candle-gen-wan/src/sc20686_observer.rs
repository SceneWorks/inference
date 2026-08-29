//! Optional SC-20686 campaign observer. `None` is the production default.
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
#[derive(Clone, Debug, PartialEq)]
pub struct CacheEvent {
    pub phase: &'static str,
    pub attention: &'static str,
    pub persistent_bytes: u64,
    pub transient_bytes: u64,
    pub elapsed_ms: u64,
    pub reused: u64,
    pub peak_bytes: u64,
    pub at_ns: u128,
}
pub trait CacheObserver {
    fn record(&mut self, event: CacheEvent);
}
/// Product-owned identity captured after Wan model loading; campaign callers must not populate
/// evidence fields from CLI claims.
#[derive(Clone, Debug)]
pub struct CampaignContext {
    pub source_ref: String,
    pub snapshot_sha256: String,
    pub snapshot_bytes: u64,
    pub variant: String,
    pub geometry_json: String,
}
pub struct JsonlObserver(File);
impl CacheObserver for JsonlObserver {
    fn record(&mut self, event: CacheEvent) {
        let line = format!("{{\"phase\":\"{}\",\"attention\":\"{}\",\"persistent_bytes\":{},\"transient_bytes\":{},\"peak_bytes\":{},\"reused\":{},\"elapsed_ms\":{},\"at_ns\":{}}}\n", event.phase, event.attention, event.persistent_bytes, event.transient_bytes, event.peak_bytes, event.reused, event.elapsed_ms, event.at_ns);
        let _ = self.0.write_all(line.as_bytes());
        let _ = self.0.flush();
    }
}
pub fn install_jsonl(path: impl AsRef<Path>) -> io::Result<Scope> {
    let path = path.as_ref();
    let file = if path == Path::new("-") { File::create("/dev/stdout")? } else { File::create(path)? };
    Ok(install(Box::new(JsonlObserver(file))))
}
pub fn install_jsonl_with_context(path: impl AsRef<Path>, context: CampaignContext) -> io::Result<Scope> {
    let scope = install_jsonl(path)?;
    observe("metadata", context.snapshot_bytes, 0, 0);
    Ok(scope)
}
thread_local! { static ACTIVE: RefCell<Option<Box<dyn CacheObserver>>> = RefCell::new(None); }
pub struct Scope {
    started: Instant,
}
pub fn install(observer: Box<dyn CacheObserver>) -> Scope {
    ACTIVE.with(|slot| *slot.borrow_mut() = Some(observer));
    Scope {
        started: Instant::now(),
    }
}
pub fn observe(phase: &'static str, persistent_bytes: u64, transient_bytes: u64, reused: u64) {
    ACTIVE.with(|slot| {
        if let Some(observer) = slot.borrow_mut().as_mut() {
            observer.record(CacheEvent {
                phase,
                attention: "cross",
                persistent_bytes,
                transient_bytes,
                elapsed_ms: 0,
                reused,
                peak_bytes: persistent_bytes.saturating_add(transient_bytes),
                at_ns: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos(),
            });
        }
    });
}
pub fn observe_cancelled() { observe("cancelled", 0, 0, 0); }
impl Drop for Scope {
    fn drop(&mut self) {
        observe("released", 0, 0, 0);
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        let _ = self.started;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Sink(std::rc::Rc<std::cell::RefCell<Vec<CacheEvent>>>);
    impl CacheObserver for Sink {
        fn record(&mut self, e: CacheEvent) {
            self.0.borrow_mut().push(e);
        }
    }
    #[test]
    fn captures_cross_lifecycle_only() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let _scope = install(Box::new(Sink(out.clone())));
        observe("cross-kv-created", 128, 0, 1);
        observe("cross-kv-read", 128, 64, 2);
        let rows = out.borrow();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|e| e.attention == "cross"));
        assert_eq!(rows[1].transient_bytes, 64);
    }
}
