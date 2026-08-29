//! Optional SC-20686 campaign observer. `None` is the production default.
use candle_gen::gen_core::runtime::CancelFlag;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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
    pub sample_kind: &'static str,
}
pub trait CacheObserver {
    fn record(&mut self, event: CacheEvent);
}

/// The complete source-owned arm plan. It is inert until a caller explicitly arms an output
/// request; keeping it here prevents individual examples from silently omitting a route or arm.
pub const CAMPAIGN_ROUTES: [&str; 5] = [
    "wan2_2_ti2v_5b",
    "wan2_2_t2v_14b",
    "wan2_2_i2v_14b",
    "wan_vace",
    "wan2_2_vace_fun_14b",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CampaignArm {
    Normal,
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CampaignCoordinate {
    pub variant: &'static str,
    pub arm: CampaignArm,
}

pub fn campaign_plan() -> [CampaignCoordinate; 10] {
    let mut plan = [CampaignCoordinate {
        variant: CAMPAIGN_ROUTES[0],
        arm: CampaignArm::Normal,
    }; 10];
    let mut index = 0;
    for variant in CAMPAIGN_ROUTES {
        plan[index] = CampaignCoordinate {
            variant,
            arm: CampaignArm::Normal,
        };
        plan[index + 1] = CampaignCoordinate {
            variant,
            arm: CampaignArm::Cancel,
        };
        index += 2;
    }
    plan
}

/// Checked projection for the campaign's explicit block-quant contract. The producer supplies the
/// measured dense element count; no caller-provided byte estimate is accepted.
pub fn checked_compressed_bytes(
    elements: u64,
    bits_per_element: u8,
    block_bytes: u64,
) -> Option<u64> {
    if elements == 0 || bits_per_element == 0 || bits_per_element > 32 || block_bytes == 0 {
        return None;
    }
    let bits = elements.checked_mul(u64::from(bits_per_element))?;
    let payload = bits.checked_add(7)?.checked_div(8)?;
    let blocks = payload
        .checked_add(block_bytes - 1)?
        .checked_div(block_bytes)?;
    blocks.checked_mul(block_bytes)
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
    pub(crate) prompt_sha256: String,
    pub(crate) guidance: String,
    pub(crate) reference_count: u32,
    pub(crate) layers: u32,
    pub(crate) heads: u32,
    pub(crate) head_dimension: u32,
    pub(crate) sq: u64,
    pub(crate) skv: u64,
    pub(crate) dtype: String,
    pub(crate) mask: String,
    pub(crate) rope: String,
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
    cancellation: bool,
}
static LAST_CAMPAIGN_CANCELLATION: AtomicBool = AtomicBool::new(false);

/// Whether the most recent armed campaign was cancelled by the product after its first live read.
/// This survives scope teardown so CLI wrappers can map only that expected error to exit 0.
pub fn campaign_cancelled() -> bool {
    LAST_CAMPAIGN_CANCELLATION.load(Ordering::Relaxed)
}

pub fn request_output(path: impl Into<PathBuf>) -> CampaignOutputRequest {
    CampaignOutputRequest {
        path: path.into(),
        cancellation: false,
    }
}

thread_local! { static PENDING_OUTPUT: RefCell<Option<PathBuf>> = RefCell::new(None); }
thread_local! { static PENDING_CANCELLATION: RefCell<bool> = RefCell::new(false); }

impl CampaignOutputRequest {
    pub fn arm(self) -> Self {
        PENDING_OUTPUT.with(|slot| *slot.borrow_mut() = Some(self.path.clone()));
        PENDING_CANCELLATION.with(|slot| *slot.borrow_mut() = self.cancellation);
        self
    }

    /// Arms the deliberate cancellation campaign. This is opt-in and remains inert for ordinary
    /// generation requests.
    pub fn arm_cancellation(mut self) -> Self {
        self.cancellation = true;
        self.arm()
    }
}

impl Drop for CampaignOutputRequest {
    fn drop(&mut self) {
        PENDING_OUTPUT.with(|slot| {
            if slot.borrow().as_ref() == Some(&self.path) {
                *slot.borrow_mut() = None;
                PENDING_CANCELLATION.with(|cancel| *cancel.borrow_mut() = false);
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
                if path.file_name().is_some_and(|name| name == ".git") {
                    continue;
                }
                files(root, &path, out)?;
            } else if path.is_file()
                && !path
                    .components()
                    .any(|component| component.as_os_str() == ".git")
            {
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
    cancel: &CancelFlag,
    variant: &str,
    batch: u32,
    frames: u32,
    width: u32,
    height: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    prompt: &str,
    guidance: Option<f32>,
    reference_count: u32,
) -> io::Result<Option<Scope>> {
    let path = PENDING_OUTPUT.with(|slot| slot.borrow_mut().take());
    let cancellation = PENDING_CANCELLATION.with(|slot| *slot.borrow());
    let Some(path) = path else { return Ok(None) };
    if !cfg!(test) && backend_peak_bytes().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SC-20686 requires a live backend allocator high-water mark",
        ));
    }
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
            prompt_sha256: format!("{:x}", Sha256::digest(prompt.as_bytes())),
            guidance: guidance.map_or_else(|| "none".into(), |value| value.to_string()),
            reference_count,
            // These are resolved from the live prepared arrays at the ownership hook. Until that
            // hook binds them, activation is intentionally incomplete rather than fabricated.
            layers: 0,
            heads: 0,
            head_dimension: 0,
            sq: 0,
            skv: 0,
            dtype: String::new(),
            mask: "none".into(),
            rope: "none".into(),
        },
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let file = if path == Path::new("-") {
        File::create("/dev/stdout")?
    } else {
        File::create(path)?
    };
    let scope = install_with_context(Box::new(JsonlObserver(file)), context);
    // Metadata is emitted exactly once, after bind_cross_kv_geometry has populated the
    // product-owned attention geometry. Never publish the zero-valued activation placeholder.
    CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = cancellation);
    LAST_CAMPAIGN_CANCELLATION.store(false, Ordering::Relaxed);
    CAMPAIGN_HANDLE.with(|slot| *slot.borrow_mut() = Some(cancel.clone()));
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
        if source_ref.len() != 40
            || !source_ref
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
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
            "sample_kind": event.sample_kind,
        });
        if let Some(ctx) = &event.context {
            if event.phase == "metadata" {
                value["source_ref"] = serde_json::json!(ctx.source_ref);
                value["snapshot_sha256"] = serde_json::json!(ctx.snapshot_sha256);
                value["snapshot_bytes"] = serde_json::json!(ctx.snapshot_bytes);
                value["variant"] = serde_json::json!(ctx.variant);
                value["geometry"] = serde_json::json!({
                    "batch": ctx.geometry.batch,
                    "resolution": format!("{}x{}", ctx.geometry.width, ctx.geometry.height),
                    "reference_count": ctx.geometry.reference_count,
                    "frames": ctx.geometry.frames,
                    "latent_frames": ctx.geometry.latent_frames,
                    "latent_height": ctx.geometry.latent_height,
                    "latent_width": ctx.geometry.latent_width,
                    "prompt": ctx.geometry.prompt_sha256,
                    "guidance": ctx.geometry.guidance,
                    "layers": ctx.geometry.layers,
                    "heads": ctx.geometry.heads,
                    "head_dimension": ctx.geometry.head_dimension,
                    "sq": ctx.geometry.sq,
                    "skv": ctx.geometry.skv,
                    "dtype": ctx.geometry.dtype,
                    "mask": ctx.geometry.mask,
                    "rope": ctx.geometry.rope,
                });
                value["cancellation_armed"] = serde_json::json!(event.reused == 1);
                if event.reused == 1 {
                    value["cancellation_arm_id"] =
                        serde_json::json!(format!("{}:{}", ctx.source_ref, ctx.variant));
                }
                value["real_weights"] = serde_json::json!(true);
                value["full_generation"] = serde_json::json!(event.reused != 1);
                value["attention_kind"] = serde_json::json!("cross");
            }
        }
        if event.phase == "metrics" {
            let current = CURRENT_PERSISTENT.with(|slot| *slot.borrow());
            let transient = CURRENT_READ_TRANSIENT.with(|slot| *slot.borrow());
            let reused = REUSED_REQUESTS.with(|slot| *slot.borrow());
            let generation_ms = GENERATION_DURATION_MS.with(|slot| *slot.borrow());
            let read_ms = CACHE_READ_DURATION_MS.with(|slot| *slot.borrow());
            // This is a checked geometry-derived projection for the qualified 2-bit format; it is
            // intentionally emitted separately from measured current allocation.
            let candidate = event
                .context
                .as_ref()
                .and_then(|ctx| {
                    let g = &ctx.geometry;
                    let elements = u64::from(g.batch)
                        .checked_mul(u64::from(g.layers))?
                        .checked_mul(u64::from(g.heads))?
                        .checked_mul(g.skv)?
                        .checked_mul(u64::from(g.head_dimension))?
                        .checked_mul(2)?;
                    checked_compressed_bytes(elements, 2, 64)
                })
                .unwrap_or(0);
            value["current_persistent_bytes"] = serde_json::json!(current);
            value["current_read_transient_bytes"] = serde_json::json!(transient);
            value["candidate_persistent_bytes"] = serde_json::json!(candidate);
            value["candidate_read_transient_bytes"] = serde_json::json!(transient);
            value["generation_duration_ms"] = serde_json::json!(GENERATION_DURATION_PRECISE_MS
                .with(|slot| *slot.borrow())
                .max(generation_ms as f64)
                .max(0.001));
            value["cache_read_duration_ms"] = serde_json::json!(CACHE_READ_DURATION_PRECISE_MS
                .with(|slot| *slot.borrow())
                .max(read_ms as f64)
                .max(0.001));
            value["reused_requests"] = serde_json::json!(reused);
            value["real_weights"] = serde_json::json!(true);
            value["full_generation"] =
                serde_json::json!(!CAMPAIGN_CANCEL.with(|slot| *slot.borrow()));
            value["attention_kind"] = serde_json::json!("cross");
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
thread_local! { static METADATA_EMITTED: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static START_EVENT_EMITTED: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static PENDING_START: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static LIVE_READ_SEEN: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static CAMPAIGN_CANCEL: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static CAMPAIGN_HANDLE: RefCell<Option<CancelFlag>> = const { RefCell::new(None) }; }
thread_local! { static CURRENT_PERSISTENT: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static CURRENT_READ_TRANSIENT: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static REUSED_REQUESTS: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static GENERATION_DURATION_MS: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static CACHE_READ_DURATION_MS: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static GENERATION_DURATION_PRECISE_MS: RefCell<f64> = const { RefCell::new(0.0) }; }
thread_local! { static CACHE_READ_DURATION_PRECISE_MS: RefCell<f64> = const { RefCell::new(0.0) }; }
thread_local! { static CANCEL_TRIGGERED: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static TERMINAL_EMITTED: RefCell<bool> = const { RefCell::new(false) }; }

/// Read the backend's continuous high-water mark. A campaign receipt is never allowed to use the
/// attributed byte sum as a substitute for a process/device measurement.
#[cfg(feature = "cuda")]
fn backend_peak_bytes() -> Option<u64> {
    candle_gen::cuda_mempool::MemPool::device_default(0)?.reserved_high()
}

#[cfg(not(feature = "cuda"))]
fn backend_peak_bytes() -> Option<u64> {
    None
}
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
    METADATA_EMITTED.with(|slot| *slot.borrow_mut() = false);
    START_EVENT_EMITTED.with(|slot| *slot.borrow_mut() = false);
    PENDING_START.with(|slot| *slot.borrow_mut() = false);
    LIVE_READ_SEEN.with(|slot| *slot.borrow_mut() = false);
    CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = false);
    CAMPAIGN_HANDLE.with(|slot| *slot.borrow_mut() = None);
    CURRENT_PERSISTENT.with(|slot| *slot.borrow_mut() = 0);
    CURRENT_READ_TRANSIENT.with(|slot| *slot.borrow_mut() = 0);
    REUSED_REQUESTS.with(|slot| *slot.borrow_mut() = 0);
    GENERATION_DURATION_MS.with(|slot| *slot.borrow_mut() = 0);
    CACHE_READ_DURATION_MS.with(|slot| *slot.borrow_mut() = 0);
    GENERATION_DURATION_PRECISE_MS.with(|slot| *slot.borrow_mut() = 0.0);
    CACHE_READ_DURATION_PRECISE_MS.with(|slot| *slot.borrow_mut() = 0.0);
    CANCEL_TRIGGERED.with(|slot| *slot.borrow_mut() = false);
    TERMINAL_EMITTED.with(|slot| *slot.borrow_mut() = false);
    Scope {
        started: Instant::now(),
    }
}
pub fn observe(phase: &'static str, persistent_bytes: u64, transient_bytes: u64, reused: u64) {
    observe_timed(phase, persistent_bytes, transient_bytes, reused, None);
}

/// Record a producer-measured operation duration. `None` retains the lifecycle elapsed time.
pub fn observe_timed(
    phase: &'static str,
    persistent_bytes: u64,
    transient_bytes: u64,
    reused: u64,
    measured: Option<Instant>,
) {
    if phase == "generation-start" {
        if START_EVENT_EMITTED.with(|slot| *slot.borrow()) {
            return;
        }
        let metadata = METADATA_EMITTED.with(|slot| *slot.borrow());
        if !metadata {
            PENDING_START.with(|slot| *slot.borrow_mut() = true);
            return;
        }
        START_EVENT_EMITTED.with(|slot| *slot.borrow_mut() = true);
    }
    if phase == "cross-kv-read" {
        LIVE_READ_SEEN.with(|slot| *slot.borrow_mut() = true);
        CURRENT_READ_TRANSIENT.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).max(transient_bytes);
        });
        REUSED_REQUESTS.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = value.saturating_add(reused);
        });
        CACHE_READ_DURATION_MS.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).max(measured.map_or(0, |t| t.elapsed().as_millis() as u64));
        });
        CACHE_READ_DURATION_PRECISE_MS.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value)
                .max(measured.map_or(0.001, |t| (t.elapsed().as_secs_f64() * 1000.0).max(0.001)));
        });
    }
    if phase == "cross-kv-created" {
        CURRENT_PERSISTENT.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).max(persistent_bytes);
        });
    }
    if phase == "generation-end" {
        if TERMINAL_EMITTED.with(|slot| *slot.borrow()) {
            return;
        }
        GENERATION_DURATION_MS.with(|slot| *slot.borrow_mut() = elapsed_ms_for(measured));
        GENERATION_DURATION_PRECISE_MS
            .with(|slot| *slot.borrow_mut() = elapsed_precise_ms_for(measured).max(0.001));
        TERMINAL_EMITTED.with(|slot| *slot.borrow_mut() = true);
    }
    if phase == "metadata" {
        let ready = CONTEXT.with(|slot| {
            slot.borrow().as_ref().is_some_and(|context| {
                let g = &context.geometry;
                g.layers != 0
                    && g.heads != 0
                    && g.head_dimension != 0
                    && g.sq != 0
                    && g.skv != 0
                    && !g.dtype.is_empty()
            })
        });
        if !ready || METADATA_EMITTED.with(|slot| *slot.borrow()) {
            return;
        }
        METADATA_EMITTED.with(|slot| *slot.borrow_mut() = true);
    }
    ACTIVE.with(|slot| {
        if let Some(observer) = slot.borrow_mut().as_mut() {
            let context = CONTEXT.with(|ctx| ctx.borrow().clone());
            let measured_peak = backend_peak_bytes();
            if context.is_some() && measured_peak.is_none() && !cfg!(test) {
                return;
            }
            let event = CacheEvent {
                phase,
                attention: "cross",
                persistent_bytes,
                transient_bytes,
                elapsed_ms: measured.map_or_else(
                    || {
                        STARTED.with(|started| {
                            started
                                .borrow()
                                .as_ref()
                                .map_or(1, |t| t.elapsed().as_millis().min(u64::MAX as u128) as u64)
                        })
                    },
                    |t| t.elapsed().as_millis().min(u64::MAX as u128) as u64,
                ),
                reused,
                peak_bytes: measured_peak
                    .unwrap_or_else(|| persistent_bytes.saturating_add(transient_bytes)),
                at_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
                operation: "observe",
                tensor_shape: String::new(),
                dtype: String::new(),
                mask: String::new(),
                rope: String::new(),
                context,
                sample_kind: "allocator",
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
    if phase == "cross-kv-read"
        && CAMPAIGN_CANCEL.with(|slot| *slot.borrow())
        && !CANCEL_TRIGGERED.with(|slot| *slot.borrow())
    {
        CANCEL_TRIGGERED.with(|slot| *slot.borrow_mut() = true);
        LAST_CAMPAIGN_CANCELLATION.store(true, Ordering::Relaxed);
        CAMPAIGN_HANDLE.with(|slot| {
            if let Some(cancel) = slot.borrow().as_ref() {
                cancel.cancel();
            }
        });
        observe_cancelled();
    }
    if phase == "metadata" && PENDING_START.with(|slot| *slot.borrow()) {
        PENDING_START.with(|slot| *slot.borrow_mut() = false);
        observe("generation-start", 0, 0, 0);
    }
}

fn elapsed_ms_for(measured: Option<Instant>) -> u64 {
    measured.map_or_else(
        || {
            STARTED.with(|started| {
                started
                    .borrow()
                    .as_ref()
                    .map_or(1, |t| t.elapsed().as_millis() as u64)
            })
        },
        |t| t.elapsed().as_millis() as u64,
    )
}
fn elapsed_precise_ms_for(measured: Option<Instant>) -> f64 {
    measured.map_or_else(
        || {
            STARTED.with(|started| {
                started
                    .borrow()
                    .as_ref()
                    .map_or(0.001, |t| (t.elapsed().as_secs_f64() * 1000.0).max(0.001))
            })
        },
        |t| (t.elapsed().as_secs_f64() * 1000.0).max(0.001),
    )
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
            let context = CONTEXT.with(|ctx| ctx.borrow().clone());
            let measured_peak = backend_peak_bytes();
            if context.is_some() && measured_peak.is_none() && !cfg!(test) {
                return;
            }
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
                peak_bytes: measured_peak
                    .unwrap_or_else(|| persistent_bytes.saturating_add(transient_bytes)),
                at_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
                operation,
                tensor_shape: shape,
                dtype,
                mask,
                rope,
                context,
                sample_kind: "allocator",
            });
        }
    });
}
pub fn observe_cancelled() {
    // A deliberate cancellation arm is only meaningful after a product-owned live K/V read.
    // This prevents preflight/request validation failures from masquerading as cancellation runs.
    if CAMPAIGN_CANCEL.with(|slot| *slot.borrow())
        && LIVE_READ_SEEN.with(|slot| *slot.borrow())
        && !TERMINAL_EMITTED.with(|slot| *slot.borrow())
    {
        TERMINAL_EMITTED.with(|slot| *slot.borrow_mut() = true);
        let elapsed = elapsed_precise_ms_for(None).max(0.001);
        GENERATION_DURATION_PRECISE_MS.with(|slot| *slot.borrow_mut() = elapsed);
        GENERATION_DURATION_MS.with(|slot| *slot.borrow_mut() = elapsed.ceil() as u64);
        observe("cancelled", 0, 0, 0);
        observe("metrics", 0, 0, 0);
    }
}

/// Bind geometry from the live prepared K/V arrays. This is deliberately producer-only; callers
/// cannot supply these values through the campaign request.
pub(crate) fn bind_cross_kv_geometry(
    layers: u32,
    heads: u32,
    head_dimension: u32,
    sq: u64,
    skv: u64,
    dtype: impl Into<String>,
) {
    let bound = CONTEXT.with(|slot| {
        if let Some(context) = slot.borrow_mut().as_mut() {
            if layers != 0 {
                context.geometry.layers = layers;
            }
            if heads != 0 {
                context.geometry.heads = heads;
            }
            if head_dimension != 0 {
                context.geometry.head_dimension = head_dimension;
            }
            if sq != 0 {
                context.geometry.sq = sq;
            }
            if skv != 0 {
                context.geometry.skv = skv;
            }
            let dtype = dtype.into();
            if !dtype.is_empty() {
                context.geometry.dtype = dtype;
            }
            true
        } else {
            false
        }
    });
    if bound {
        let cancellation = CAMPAIGN_CANCEL.with(|slot| *slot.borrow());
        observe("metadata", 0, 0, u64::from(cancellation));
        if !START_EVENT_EMITTED.with(|slot| *slot.borrow())
            && !PENDING_START.with(|slot| *slot.borrow())
        {
            observe("generation-start", 0, 0, 0);
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if !TERMINAL_EMITTED.with(|slot| *slot.borrow()) {
            observe("generation-end", 0, 0, 0);
        }
        observe("invalidated", 0, 0, 0);
        observe("released", 0, 0, 0);
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        CONTEXT.with(|slot| *slot.borrow_mut() = None);
        STARTED.with(|slot| *slot.borrow_mut() = None);
        METADATA_EMITTED.with(|slot| *slot.borrow_mut() = false);
        START_EVENT_EMITTED.with(|slot| *slot.borrow_mut() = false);
        PENDING_START.with(|slot| *slot.borrow_mut() = false);
        LIVE_READ_SEEN.with(|slot| *slot.borrow_mut() = false);
        CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = false);
        CAMPAIGN_HANDLE.with(|slot| *slot.borrow_mut() = None);
        CANCEL_TRIGGERED.with(|slot| *slot.borrow_mut() = false);
        TERMINAL_EMITTED.with(|slot| *slot.borrow_mut() = false);
        GENERATION_DURATION_PRECISE_MS.with(|slot| *slot.borrow_mut() = 0.0);
        CACHE_READ_DURATION_PRECISE_MS.with(|slot| *slot.borrow_mut() = 0.0);
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
            "b".repeat(40),
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
                prompt_sha256: "a".repeat(64),
                guidance: "none".into(),
                reference_count: 0,
                layers: 1,
                heads: 1,
                head_dimension: 1,
                sq: 64,
                skv: 128,
                dtype: "bf16".into(),
                mask: "none".into(),
                rope: "none".into(),
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
            prompt_sha256: "c".repeat(64),
            guidance: "1".into(),
            reference_count: 0,
            layers: 1,
            heads: 1,
            head_dimension: 1,
            sq: 1,
            skv: 1,
            dtype: "BF16".into(),
            mask: "none".into(),
            rope: "none".into(),
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
    fn compressed_projection_is_checked_and_block_aligned() {
        assert_eq!(checked_compressed_bytes(1024, 4, 64), Some(512));
        assert_eq!(checked_compressed_bytes(0, 4, 64), None);
        assert_eq!(checked_compressed_bytes(u64::MAX, 32, 64), None);
    }

    #[test]
    fn campaign_plan_covers_each_route_once_per_arm() {
        let plan = campaign_plan();
        assert_eq!(plan.len(), CAMPAIGN_ROUTES.len() * 2);
        for route in CAMPAIGN_ROUTES {
            assert_eq!(
                plan.iter()
                    .filter(|item| item.variant == route && item.arm == CampaignArm::Normal)
                    .count(),
                1
            );
            assert_eq!(
                plan.iter()
                    .filter(|item| item.variant == route && item.arm == CampaignArm::Cancel)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn lifecycle_deduplicates_start_and_triggers_cancel_once_after_read() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let scope = install(Box::new(Sink(out.clone())));
        CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = true);
        METADATA_EMITTED.with(|slot| *slot.borrow_mut() = true);
        observe("generation-start", 0, 0, 0);
        observe("generation-start", 0, 0, 0);
        observe("cross-kv-read", 1, 2, 1);
        observe("cross-kv-read", 1, 2, 1);
        drop(scope);
        let rows = out.borrow();
        assert_eq!(
            rows.iter()
                .filter(|event| event.phase == "generation-start")
                .count(),
            1
        );
        assert_eq!(
            rows.iter()
                .filter(|event| event.phase == "cancelled")
                .count(),
            1
        );
        assert_eq!(
            rows.iter()
                .filter(|event| event.phase == "generation-end")
                .count(),
            0
        );
        assert_eq!(
            rows.iter()
                .filter(|event| event.phase == "invalidated")
                .count(),
            1
        );
        assert_eq!(
            rows.iter()
                .filter(|event| event.phase == "released")
                .count(),
            1
        );
        let read = rows
            .iter()
            .position(|event| event.phase == "cross-kv-read")
            .unwrap();
        let cancelled = rows
            .iter()
            .position(|event| event.phase == "cancelled")
            .unwrap();
        assert!(read < cancelled);
    }

    #[test]
    fn requested_output_derives_identity_from_snapshot_bytes() {
        let root = tempfile::tempdir().unwrap();
        let snapshot = root.path().join("0123456789abcdef0123456789abcdef01234567");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{\"layers\":1}").unwrap();
        let output = root.path().join("events.jsonl");
        let _request = request_output(&output).arm();
        let scope = activate_requested(
            &snapshot,
            &CancelFlag::default(),
            "wan2_2_t2v_14b",
            1,
            5,
            64,
            64,
            2,
            8,
            8,
            "prompt",
            Some(1.0),
            0,
        )
        .unwrap()
        .expect("armed output request activates at runtime");
        bind_cross_kv_geometry(1, 8, 64, 1, 2, "bf16");
        observe("loaded", 1, 0, 0);
        drop(scope);
        let lines = std::fs::read_to_string(output).unwrap();
        assert!(lines.contains("wan2_2_t2v_14b"));
        assert_eq!(
            lines
                .lines()
                .filter(|line| line.contains("\"phase\":\"metadata\""))
                .count(),
            1
        );
        assert!(lines
            .lines()
            .any(|line| line.contains("\"attention_kind\":\"cross\"")));
        assert!(lines
            .lines()
            .any(|line| line.contains("released") && line.contains("t2v")));
        assert!(!lines.contains("config.json"));
    }
}
