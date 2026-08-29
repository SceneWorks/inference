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
    pub operation: &'static str,
    pub tensor_shape: String,
    pub dtype: String,
    pub mask: String,
    pub rope: String,
    pub context: Option<CampaignContext>,
}
pub trait CacheObserver {
    fn record(&mut self, event: CacheEvent);
}
/// Product-owned identity captured after Wan model loading; campaign callers must not populate
/// evidence fields from CLI claims.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        let line = format!("{{\"phase\":\"{}\",\"attention\":\"{}\",\"operation\":\"{}\",\"tensor_shape\":\"{}\",\"dtype\":\"{}\",\"mask\":\"{}\",\"rope\":\"{}\",\"persistent_bytes\":{},\"transient_bytes\":{},\"peak_bytes\":{},\"reused\":{},\"elapsed_ms\":{},\"at_ns\":{}}}\n", event.phase, event.attention, event.operation, event.tensor_shape, event.dtype, event.mask, event.rope, event.persistent_bytes, event.transient_bytes, event.peak_bytes, event.reused, event.elapsed_ms, event.at_ns);
        let _ = self.0.write_all(line.as_bytes());
        let _ = self.0.flush();
    }
}
pub fn install_jsonl(path: impl AsRef<Path>) -> io::Result<Scope> {
    let path = path.as_ref();
    let file = if path == Path::new("-") {
        File::create("/dev/stdout")?
    } else {
        File::create(path)?
    };
    Ok(install(Box::new(JsonlObserver(file))))
}
pub fn install_jsonl_with_context(
    path: impl AsRef<Path>,
    context: CampaignContext,
) -> io::Result<Scope> {
    let path = path.as_ref();
    let file = if path == Path::new("-") {
        File::create("/dev/stdout")?
    } else {
        File::create(path)?
    };
    let snapshot_bytes = context.snapshot_bytes;
    let scope = install_with_context(Box::new(JsonlObserver(file)), context);
    observe("metadata", snapshot_bytes, 0, 0);
    Ok(scope)
}
thread_local! { static ACTIVE: RefCell<Option<Box<dyn CacheObserver>>> = RefCell::new(None); }
thread_local! { static CONTEXT: RefCell<Option<CampaignContext>> = RefCell::new(None); }
pub struct Scope {
    started: Instant,
}
pub fn install(observer: Box<dyn CacheObserver>) -> Scope {
    install_with_context_inner(observer, None)
}

pub fn install_with_context(observer: Box<dyn CacheObserver>, context: CampaignContext) -> Scope {
    install_with_context_inner(observer, Some(context))
}

fn install_with_context_inner(
    observer: Box<dyn CacheObserver>,
    context: Option<CampaignContext>,
) -> Scope {
    ACTIVE.with(|slot| *slot.borrow_mut() = Some(observer));
    CONTEXT.with(|slot| *slot.borrow_mut() = context);
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
                at_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
                operation: "observe",
                tensor_shape: String::new(),
                dtype: String::new(),
                mask: String::new(),
                rope: String::new(),
                context: CONTEXT.with(|ctx| ctx.borrow().clone()),
            });
        }
    });
}
pub fn observe_tensor(
    phase: &'static str,
    operation: &'static str,
    persistent_bytes: u64,
    transient_bytes: u64,
    reused: u64,
    tensor_shape: impl Into<String>,
    dtype: impl Into<String>,
    mask: impl Into<String>,
    rope: impl Into<String>,
) {
    let shape = tensor_shape.into();
    let dtype = dtype.into();
    let mask = mask.into();
    let rope = rope.into();
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
                at_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
                operation,
                tensor_shape: shape,
                dtype,
                mask,
                rope,
                context: CONTEXT.with(|ctx| ctx.borrow().clone()),
            });
        }
    });
}
pub fn observe_cancelled() {
    observe("cancelled", 0, 0, 0);
}
impl Drop for Scope {
    fn drop(&mut self) {
        observe("released", 0, 0, 0);
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        CONTEXT.with(|slot| *slot.borrow_mut() = None);
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

    #[test]
    fn context_and_tensor_metadata_are_opt_in_and_bound_to_events() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let context = CampaignContext {
            source_ref: "snapshot".into(),
            snapshot_sha256: "a".repeat(64),
            snapshot_bytes: 4096,
            variant: "t2v".into(),
            geometry_json: "{\"layers\":1}".into(),
        };
        let _scope = install_with_context(Box::new(Sink(out.clone())), context);
        observe_tensor(
            "cross-kv-prepared",
            "prepare",
            128,
            0,
            1,
            "[1,8,64]",
            "BF16",
            "none",
            "applied",
        );
        let row = out.borrow().last().cloned().unwrap();
        assert_eq!(row.operation, "prepare");
        assert_eq!(row.tensor_shape, "[1,8,64]");
        assert_eq!(row.context.unwrap().snapshot_bytes, 4096);
    }
}
