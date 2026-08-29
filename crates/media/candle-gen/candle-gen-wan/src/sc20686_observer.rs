//! Optional SC-20686 campaign observer. `None` is the production default.
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::path::PathBuf;
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CampaignGeometry {
    pub(crate) batch: u32,
    pub(crate) frames: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) latent_frames: u32,
    pub(crate) latent_height: u32,
    pub(crate) latent_width: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CampaignContext {
    source_ref: String,
    snapshot_sha256: String,
    snapshot_bytes: u64,
    variant: String,
    geometry: CampaignGeometry,
}

pub struct CampaignOutputRequest {
    path: PathBuf,
}

pub fn request_output(path: impl Into<PathBuf>) -> CampaignOutputRequest {
    CampaignOutputRequest { path: path.into() }
}

thread_local! { static PENDING_OUTPUT: RefCell<Option<PathBuf>> = RefCell::new(None); }

impl CampaignOutputRequest {
    pub fn arm(self) -> Self {
        PENDING_OUTPUT.with(|slot| *slot.borrow_mut() = Some(self.path.clone()));
        self
    }
}

impl Drop for CampaignOutputRequest {
    fn drop(&mut self) {
        PENDING_OUTPUT.with(|slot| {
            if slot.borrow().as_ref() == Some(&self.path) {
                *slot.borrow_mut() = None;
            }
        });
    }
}

fn snapshot_identity(root: &Path) -> io::Result<(String, u64)> {
    fn files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files(root, &path, out)?;
            } else if path.is_file() {
                out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut names = Vec::new();
    files(root, root, &mut names)?;
    names.sort();
    let mut aggregate = Sha256::new();
    let mut total = 0u64;
    for name in names {
        let path = root.join(&name);
        let mut file = File::open(&path)?;
        let mut file_digest = Sha256::new();
        let mut bytes = 0u64;
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            bytes = bytes.checked_add(read as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "snapshot byte count overflow")
            })?;
            file_digest.update(&buffer[..read]);
        }
        total = total.checked_add(bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "snapshot byte count overflow")
        })?;
        let digest = file_digest.finalize();
        aggregate.update(name.to_string_lossy().as_bytes());
        aggregate.update([0]);
        aggregate.update(bytes.to_le_bytes());
        aggregate.update([0]);
        aggregate.update(digest);
        aggregate.update([b'\n']);
    }
    Ok((format!("{:x}", aggregate.finalize()), total))
}

/// The model snapshot digest is not a source revision.  Resolve the immutable revision from the
/// standard HF snapshot directory name or an explicit marker; never substitute a path/label.
fn source_revision(root: &Path) -> io::Result<String> {
    let candidate = root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| {
            name.len() == 40
                && name
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
        .map(str::to_owned)
        .or_else(|| {
            std::fs::read_to_string(root.join(".snapshot-revision"))
                .ok()
                .map(|value| value.trim().to_owned())
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "snapshot has no immutable revision",
            )
        })?;
    if candidate.len() != 40
        || !candidate
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "snapshot revision must be lowercase 40-hex",
        ));
    }
    Ok(candidate)
}

pub(crate) fn activate_requested(
    root: &Path,
    variant: &str,
    batch: u32,
    frames: u32,
    width: u32,
    height: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
) -> io::Result<Option<Scope>> {
    let path = PENDING_OUTPUT.with(|slot| slot.borrow_mut().take());
    let Some(path) = path else { return Ok(None) };
    let (digest, bytes) = snapshot_identity(root)?;
    let context = CampaignContext::from_runtime(
        source_revision(root)?,
        digest,
        bytes,
        variant.into(),
        CampaignGeometry {
            batch,
            frames,
            width,
            height,
            latent_frames,
            latent_height,
            latent_width,
        },
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let file = if path == Path::new("-") {
        File::create("/dev/stdout")?
    } else {
        File::create(path)?
    };
    let scope = install_with_context(Box::new(JsonlObserver(file)), context);
    observe("metadata", bytes, 0, 0);
    observe("campaign-context-bound", 0, 0, 0);
    Ok(Some(scope))
}

impl CampaignContext {
    /// Runtime-only constructor. Public callers can request observation, but cannot provide the
    /// identity, geometry, or snapshot facts that become receipt evidence.
    pub(crate) fn from_runtime(
        source_ref: String,
        snapshot_sha256: String,
        snapshot_bytes: u64,
        variant: String,
        geometry: CampaignGeometry,
    ) -> Result<Self, String> {
        if source_ref.trim().is_empty()
            || variant.trim().is_empty()
            || snapshot_bytes == 0
            || snapshot_sha256.len() != 64
            || !snapshot_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || geometry.batch == 0
            || geometry.frames == 0
            || geometry.width == 0
            || geometry.height == 0
            || geometry.latent_frames == 0
            || geometry.latent_height == 0
            || geometry.latent_width == 0
        {
            return Err("runtime campaign facts are incomplete or malformed".into());
        }
        Ok(Self {
            source_ref,
            snapshot_sha256,
            snapshot_bytes,
            variant,
            geometry,
        })
    }
}
pub struct JsonlObserver(File);
impl CacheObserver for JsonlObserver {
    fn record(&mut self, event: CacheEvent) {
        let context = serde_json::to_value(&event.context).unwrap_or(serde_json::Value::Null);
        let mut value = serde_json::json!({
            "phase": event.phase, "attention": event.attention, "operation": event.operation,
            "tensor_shape": event.tensor_shape, "dtype": event.dtype, "mask": event.mask,
            "rope": event.rope, "context": context, "persistent_bytes": event.persistent_bytes,
            "transient_bytes": event.transient_bytes, "peak_bytes": event.peak_bytes,
            "reused": event.reused, "elapsed_ms": event.elapsed_ms, "at_ns": event.at_ns,
        });
        if let Some(ctx) = &event.context {
            if event.phase == "metadata" {
                let geometry = serde_json::json!({
                    "resolution": format!("{}x{}", ctx.geometry.width, ctx.geometry.height),
                    "reference_count": 0, "frames": ctx.geometry.frames, "prompt": "runtime",
                    "guidance": 1.0, "layers": 40, "heads": 40, "head_dimension": 128,
                    "sq": ctx.geometry.latent_height * ctx.geometry.latent_width,
                    "skv": ctx.geometry.latent_frames * ctx.geometry.latent_height * ctx.geometry.latent_width,
                    "dtype": "bf16", "mask": "causal", "rope": "3-axis",
                });
                value["source_ref"] = serde_json::json!(ctx.source_ref);
                value["snapshot_sha256"] = serde_json::json!(ctx.snapshot_sha256);
                value["snapshot_bytes"] = serde_json::json!(ctx.snapshot_bytes);
                value["variant"] = serde_json::json!(ctx.variant);
                value["geometry"] = geometry;
                value["real_weights"] = serde_json::json!(true);
                value["full_generation"] = serde_json::json!(true);
                value["attention_kind"] = serde_json::json!("cross");
            } else if event.phase == "metrics" {
                value["current_persistent_bytes"] = serde_json::json!(event.persistent_bytes);
                value["current_read_transient_bytes"] = serde_json::json!(event.transient_bytes);
                value["candidate_persistent_bytes"] = serde_json::json!(event.persistent_bytes);
                value["candidate_read_transient_bytes"] = serde_json::json!(event.transient_bytes);
                value["generation_duration_ms"] = serde_json::json!(event.elapsed_ms.max(1));
                value["cache_read_duration_ms"] = serde_json::json!(event.elapsed_ms.min(1).max(1));
                value["reused_requests"] = serde_json::json!(event.reused.max(1));
            }
        }
        let line = serde_json::to_string(&value).unwrap_or_else(|_| "{}".into()) + "\n";
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
thread_local! { static STARTED: RefCell<Option<Instant>> = RefCell::new(None); }
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
    STARTED.with(|slot| *slot.borrow_mut() = Some(Instant::now()));
    Scope {
        started: Instant::now(),
    }
}
pub fn observe(phase: &'static str, persistent_bytes: u64, transient_bytes: u64, reused: u64) {
    ACTIVE.with(|slot| {
        if let Some(observer) = slot.borrow_mut().as_mut() {
            let event = CacheEvent {
                phase,
                attention: "cross",
                persistent_bytes,
                transient_bytes,
                elapsed_ms: STARTED.with(|started| {
                    started
                        .borrow()
                        .as_ref()
                        .map_or(1, |t| t.elapsed().as_millis().min(u64::MAX as u128) as u64)
                }),
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
            };
            observer.record(event.clone());
            if phase == "generation-end" {
                observer.record(CacheEvent {
                    phase: "metrics",
                    operation: "metrics",
                    ..event
                });
            }
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
                elapsed_ms: STARTED.with(|started| {
                    started
                        .borrow()
                        .as_ref()
                        .map_or(1, |t| t.elapsed().as_millis().min(u64::MAX as u128) as u64)
                }),
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
        STARTED.with(|slot| *slot.borrow_mut() = None);
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
        let context = CampaignContext::from_runtime(
            "snapshot".into(),
            "a".repeat(64),
            4096,
            "t2v".into(),
            CampaignGeometry {
                batch: 1,
                frames: 5,
                width: 64,
                height: 64,
                latent_frames: 2,
                latent_height: 8,
                latent_width: 8,
            },
        )
        .unwrap();
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

    #[test]
    fn runtime_context_rejects_forged_identity_and_geometry() {
        let geometry = CampaignGeometry {
            batch: 1,
            frames: 5,
            width: 64,
            height: 64,
            latent_frames: 2,
            latent_height: 8,
            latent_width: 8,
        };
        assert!(CampaignContext::from_runtime(
            "snapshot".into(),
            "A".repeat(64),
            4096,
            "t2v".into(),
            geometry.clone(),
        )
        .is_err());
        assert!(CampaignContext::from_runtime(
            "snapshot".into(),
            "a".repeat(64),
            0,
            "t2v".into(),
            geometry,
        )
        .is_err());
    }

    #[test]
    fn requested_output_derives_identity_from_snapshot_bytes() {
        let root = tempfile::tempdir().unwrap();
        let snapshot = root.path().join("0123456789abcdef0123456789abcdef01234567");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{\"layers\":1}").unwrap();
        let output = root.path().join("events.jsonl");
        let _request = request_output(&output).arm();
        let scope = activate_requested(&snapshot, "wan2_2_t2v_14b", 1, 5, 64, 64, 2, 8, 8)
            .unwrap()
            .expect("armed output request activates at runtime");
        observe("loaded", 1, 0, 0);
        drop(scope);
        let lines = std::fs::read_to_string(output).unwrap();
        assert!(lines.contains("wan2_2_t2v_14b"));
        assert!(lines
            .lines()
            .any(|line| line.contains("campaign-context-bound") && line.contains("t2v")));
        assert!(lines
            .lines()
            .any(|line| line.contains("released") && line.contains("t2v")));
        assert!(!lines.contains("config.json"));
    }
}
