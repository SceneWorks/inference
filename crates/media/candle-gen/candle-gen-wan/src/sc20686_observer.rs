//! Optional SC-20686 campaign observer. `None` is the production default.
use candle_gen::gen_core::runtime::CancelFlag;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
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
    pub sample_kind: &'static str,
    pub cache_id: u64,
    pub candidate_persistent_bytes: u64,
    pub allocator_before_bytes: u64,
    pub allocator_after_bytes: u64,
    pub allocator_high_bytes: u64,
    pub allocator_reserved_bytes: u64,
    pub allocator_measurement_available: bool,
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

/// Exact SC-20675 v2 payload projection for one live `[B,H,Skv,D]` K/V cache.
/// Keys group on the token axis and retain an f32 dense pending tail; values group on the
/// channel axis. Every completed group carries f16 scale and zero metadata.
pub fn checked_packed_group_affine_kv_bytes(
    batch: u64,
    heads: u64,
    tokens: u64,
    width: u64,
    group_size: u64,
) -> Option<u64> {
    if batch == 0 || heads == 0 || tokens == 0 || width == 0 || group_size == 0 {
        return None;
    }
    let rows = batch.checked_mul(heads)?;
    let complete_groups = tokens.checked_div(group_size)?;
    let pending_tokens = tokens.checked_rem(group_size)?;

    let key_codes_per_group = group_size
        .checked_mul(width)?
        .checked_add(3)?
        .checked_div(4)?;
    let key_codes = rows
        .checked_mul(complete_groups)?
        .checked_mul(key_codes_per_group)?;
    let key_metadata = rows
        .checked_mul(complete_groups)?
        .checked_mul(width)?
        .checked_mul(4)?;
    let key_pending = rows
        .checked_mul(pending_tokens)?
        .checked_mul(width)?
        .checked_mul(4)?;

    let value_rows = rows.checked_mul(tokens)?;
    let value_codes_per_row = width.checked_add(3)?.checked_div(4)?;
    let value_groups_per_row = width.checked_add(group_size - 1)?.checked_div(group_size)?;
    let value_codes = value_rows.checked_mul(value_codes_per_row)?;
    let value_metadata = value_rows
        .checked_mul(value_groups_per_row)?
        .checked_mul(4)?;
    key_codes
        .checked_add(key_metadata)?
        .checked_add(key_pending)?
        .checked_add(value_codes)?
        .checked_add(value_metadata)
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
    /// Set only after a product-owned transformer has successfully projected live snapshot-backed
    /// tensors. Campaign activation and CLI arguments cannot assert this fact.
    real_weights: bool,
}

pub struct CampaignOutputRequest {
    path: PathBuf,
    cancellation: bool,
}
/// Whether the most recent armed campaign was cancelled by the product after its first live read.
/// This survives scope teardown so CLI wrappers can map only that expected error to exit 0.
pub fn campaign_cancelled() -> bool {
    LAST_CAMPAIGN_CANCELLATION.with(Cell::get)
}

pub fn request_output(path: impl Into<PathBuf>) -> CampaignOutputRequest {
    CampaignOutputRequest {
        path: path.into(),
        cancellation: false,
    }
}

thread_local! { static PENDING_OUTPUT: RefCell<Option<PathBuf>> = RefCell::new(None); }
thread_local! { static PENDING_CANCELLATION: RefCell<bool> = RefCell::new(false); }
thread_local! { static LAST_CAMPAIGN_CANCELLATION: Cell<bool> = const { Cell::new(false) }; }

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
    let writer: Box<dyn Write> = if path == Path::new("-") {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(path)?)
    };
    let scope = install_with_context(Box::new(JsonlObserver(writer)), context);
    // Metadata is emitted exactly once, after bind_cross_kv_geometry has populated the
    // product-owned attention geometry. Never publish the zero-valued activation placeholder.
    CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = cancellation);
    LAST_CAMPAIGN_CANCELLATION.with(|slot| slot.set(false));
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
            real_weights: false,
        })
    }
}
pub struct JsonlObserver(Box<dyn Write>);
impl CacheObserver for JsonlObserver {
    fn record(&mut self, event: CacheEvent) {
        let context = serde_json::to_value(&event.context).unwrap_or(serde_json::Value::Null);
        let mut value = serde_json::json!({
            "phase": event.phase, "attention": event.attention, "operation": event.operation,
            "tensor_shape": event.tensor_shape, "dtype": event.dtype, "mask": event.mask,
            "rope": event.rope, "context": context, "persistent_bytes": event.persistent_bytes,
            "transient_bytes": event.transient_bytes, "peak_bytes": event.peak_bytes,
            "reused": event.reused, "elapsed_ms": event.elapsed_ms, "at_ns": event.at_ns,
            "sample_kind": event.sample_kind, "cache_id": event.cache_id,
            "candidate_persistent_bytes": event.candidate_persistent_bytes,
            "allocator_before_bytes": event.allocator_before_bytes,
            "allocator_after_bytes": event.allocator_after_bytes,
            "allocator_high_bytes": event.allocator_high_bytes,
            "allocator_reserved_bytes": event.allocator_reserved_bytes,
            "allocator_measurement_available": event.allocator_measurement_available,
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
                value["real_weights"] = serde_json::json!(ctx.real_weights);
                value["full_generation"] = serde_json::json!(event.reused != 1);
                value["attention_kind"] = serde_json::json!("cross");
            }
        }
        if event.phase == "metrics" {
            let current = PEAK_PERSISTENT.with(|slot| *slot.borrow());
            let transient = CURRENT_READ_TRANSIENT.with(|slot| *slot.borrow());
            let reused = REUSED_REQUESTS.with(|slot| *slot.borrow());
            let generation_ms = GENERATION_DURATION_MS.with(|slot| *slot.borrow());
            let read_ms = CACHE_READ_DURATION_MS.with(|slot| *slot.borrow());
            let candidate = PEAK_CANDIDATE.with(|slot| *slot.borrow());
            let active_min =
                CACHE_STATES.with(|slot| slot.borrow().values().map(|state| state.reads).min());
            let completed_min =
                COMPLETED_CACHE_READS.with(|slot| slot.borrow().iter().copied().min());
            let minimum_cache_reads = active_min
                .into_iter()
                .chain(completed_min)
                .min()
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
            value["minimum_cache_reads"] = serde_json::json!(minimum_cache_reads);
            value["real_weights"] = serde_json::json!(event
                .context
                .as_ref()
                .is_some_and(|context| context.real_weights));
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
    let writer: Box<dyn Write> = if path == Path::new("-") {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(path)?)
    };
    Ok(install(Box::new(JsonlObserver(writer))))
}
pub fn install_jsonl_with_context(
    path: impl AsRef<Path>,
    context: CampaignContext,
) -> io::Result<Scope> {
    let path = path.as_ref();
    let writer: Box<dyn Write> = if path == Path::new("-") {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(path)?)
    };
    let snapshot_bytes = context.snapshot_bytes;
    let scope = install_with_context(Box::new(JsonlObserver(writer)), context);
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
thread_local! { static LIVE_PERSISTENT: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static PEAK_PERSISTENT: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static LIVE_CANDIDATE: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static PEAK_CANDIDATE: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static NEXT_CACHE_ID: Cell<u64> = const { Cell::new(1) }; }
thread_local! { static CACHE_STATES: RefCell<BTreeMap<u64, CacheAccounting>> = const { RefCell::new(BTreeMap::new()) }; }
thread_local! { static COMPLETED_CACHE_READS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) }; }
thread_local! { static CURRENT_READ_TRANSIENT: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static REUSED_REQUESTS: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static GENERATION_DURATION_MS: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static CACHE_READ_DURATION_MS: RefCell<u64> = const { RefCell::new(0) }; }
thread_local! { static GENERATION_DURATION_PRECISE_MS: RefCell<f64> = const { RefCell::new(0.0) }; }
thread_local! { static CACHE_READ_DURATION_PRECISE_MS: RefCell<f64> = const { RefCell::new(0.0) }; }
thread_local! { static CANCEL_TRIGGERED: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static TERMINAL_EMITTED: RefCell<bool> = const { RefCell::new(false) }; }

/// The single observer-presence probe used by product code to make the entire campaign path inert.
pub(crate) fn campaign_active() -> bool {
    ACTIVE.with(|slot| slot.borrow().is_some())
}

#[derive(Clone, Copy, Debug, Default)]
struct CacheAccounting {
    dense_bytes: u64,
    candidate_bytes: u64,
    reads: u64,
}

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

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveAllocatorWindow {
    used_before: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ActiveAllocatorMeasurement {
    pub(crate) used_before: u64,
    pub(crate) used_after: u64,
    pub(crate) used_high: u64,
    pub(crate) reserved_after: u64,
    pub(crate) available: bool,
}

impl ActiveAllocatorMeasurement {
    pub(crate) fn transient_bytes(self) -> u64 {
        self.used_high.saturating_sub(self.used_before)
    }
}

/// Reset only the active-byte high-water for one serialized product operation. The whole-run
/// RESERVED high-water remains untouched and continues to supply the admission-domain peak.
#[cfg(feature = "cuda")]
pub(crate) fn begin_active_allocator_window() -> Option<ActiveAllocatorWindow> {
    let pool = candle_gen::cuda_mempool::MemPool::device_default(0)?;
    let used_before = pool.used()?;
    if !pool.reset_used_high_water() || pool.used_high()? < used_before {
        return None;
    }
    Some(ActiveAllocatorWindow { used_before })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn begin_active_allocator_window() -> Option<ActiveAllocatorWindow> {
    None
}

#[cfg(feature = "cuda")]
pub(crate) fn finish_active_allocator_window(
    window: Option<ActiveAllocatorWindow>,
) -> ActiveAllocatorMeasurement {
    let Some(window) = window else {
        return ActiveAllocatorMeasurement::default();
    };
    let Some(pool) = candle_gen::cuda_mempool::MemPool::device_default(0) else {
        return ActiveAllocatorMeasurement::default();
    };
    let (Some(used_after), Some(used_high), Some(reserved_after)) =
        (pool.used(), pool.used_high(), pool.reserved())
    else {
        return ActiveAllocatorMeasurement::default();
    };
    ActiveAllocatorMeasurement {
        used_before: window.used_before,
        used_after,
        used_high,
        reserved_after,
        available: used_high >= window.used_before && reserved_after >= used_after,
    }
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn finish_active_allocator_window(
    _window: Option<ActiveAllocatorWindow>,
) -> ActiveAllocatorMeasurement {
    ActiveAllocatorMeasurement::default()
}

#[cfg(feature = "cuda")]
fn allocator_remnant() -> ActiveAllocatorMeasurement {
    let Some(pool) = candle_gen::cuda_mempool::MemPool::device_default(0) else {
        return ActiveAllocatorMeasurement::default();
    };
    let (Some(used), Some(reserved)) = (pool.used(), pool.reserved()) else {
        return ActiveAllocatorMeasurement::default();
    };
    ActiveAllocatorMeasurement {
        used_before: used,
        used_after: used,
        used_high: used,
        reserved_after: reserved,
        available: reserved >= used,
    }
}

#[cfg(not(feature = "cuda"))]
fn allocator_remnant() -> ActiveAllocatorMeasurement {
    ActiveAllocatorMeasurement::default()
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
    LIVE_PERSISTENT.with(|slot| *slot.borrow_mut() = 0);
    PEAK_PERSISTENT.with(|slot| *slot.borrow_mut() = 0);
    LIVE_CANDIDATE.with(|slot| *slot.borrow_mut() = 0);
    PEAK_CANDIDATE.with(|slot| *slot.borrow_mut() = 0);
    NEXT_CACHE_ID.with(|slot| slot.set(1));
    CACHE_STATES.with(|slot| slot.borrow_mut().clear());
    COMPLETED_CACHE_READS.with(|slot| slot.borrow_mut().clear());
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
    observe_timed_for_cache(
        phase,
        persistent_bytes,
        transient_bytes,
        reused,
        measured,
        0,
        0,
        0,
        0,
        0,
        0,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn observe_timed_for_cache(
    phase: &'static str,
    persistent_bytes: u64,
    transient_bytes: u64,
    reused: u64,
    measured: Option<Instant>,
    cache_id: u64,
    candidate_persistent_bytes: u64,
    allocator_before_bytes: u64,
    allocator_after_bytes: u64,
    allocator_high_bytes: u64,
    allocator_reserved_bytes: u64,
    allocator_measurement_available: bool,
) {
    // Outside an explicitly installed campaign, this must be inert before touching any of the
    // accounting thread-locals or allocating event strings/maps.
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
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
        if cache_id != 0 {
            CACHE_STATES.with(|slot| {
                if let Some(state) = slot.borrow_mut().get_mut(&cache_id) {
                    state.reads = state.reads.saturating_add(1);
                }
            });
        }
        CACHE_READ_DURATION_MS.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).saturating_add(
                measured.map_or(0, |t| t.elapsed().as_millis().min(u64::MAX as u128) as u64),
            );
        });
        CACHE_READ_DURATION_PRECISE_MS.with(|slot| {
            let mut value = slot.borrow_mut();
            *value += measured.map_or(0.001, |t| (t.elapsed().as_secs_f64() * 1000.0).max(0.001));
        });
    }
    if phase == "cross-kv-created" {
        let live = LIVE_PERSISTENT.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).saturating_add(persistent_bytes);
            *value
        });
        PEAK_PERSISTENT.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).max(live);
        });
        if cache_id != 0 {
            CACHE_STATES.with(|slot| {
                slot.borrow_mut().insert(
                    cache_id,
                    CacheAccounting {
                        dense_bytes: persistent_bytes,
                        candidate_bytes: candidate_persistent_bytes,
                        reads: 0,
                    },
                );
            });
            let candidate_live = LIVE_CANDIDATE.with(|slot| {
                let mut value = slot.borrow_mut();
                *value = value.saturating_add(candidate_persistent_bytes);
                *value
            });
            PEAK_CANDIDATE.with(|slot| {
                let mut value = slot.borrow_mut();
                *value = (*value).max(candidate_live);
            });
        }
    }
    if phase == "cross-kv-released" {
        LIVE_PERSISTENT.with(|slot| {
            let mut value = slot.borrow_mut();
            *value = (*value).saturating_sub(persistent_bytes);
        });
        if cache_id != 0 {
            LIVE_CANDIDATE.with(|slot| {
                let mut value = slot.borrow_mut();
                *value = value.saturating_sub(candidate_persistent_bytes);
            });
        }
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
            let peak_bytes = measured_peak.unwrap_or_else(|| {
                if context.is_some() && !cfg!(test) {
                    // Activation already refuses an absent counter. If it disappears mid-run,
                    // preserve an explicit invalid zero sample so the adapter rejects the receipt.
                    0
                } else {
                    persistent_bytes.saturating_add(transient_bytes)
                }
            });
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
                peak_bytes,
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
                cache_id,
                candidate_persistent_bytes,
                allocator_before_bytes,
                allocator_after_bytes,
                allocator_high_bytes,
                allocator_reserved_bytes,
                allocator_measurement_available,
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
        CAMPAIGN_HANDLE.with(|slot| {
            if let Some(cancel) = slot.borrow().as_ref() {
                cancel.cancel();
            }
        });
    }
    if phase == "metadata" && PENDING_START.with(|slot| *slot.borrow()) {
        PENDING_START.with(|slot| *slot.borrow_mut() = false);
        observe("generation-start", 0, 0, 0);
    }
}

/// Register one product-owned prepared cache and return its run-local identity.
pub(crate) fn register_cache(dense_bytes: u64, candidate_bytes: u64, measured: Instant) -> u64 {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return 0;
    }
    let cache_id = NEXT_CACHE_ID.with(|slot| {
        let value = slot.get();
        slot.set(value.saturating_add(1));
        value
    });
    observe_timed_for_cache(
        "cross-kv-created",
        dense_bytes,
        0,
        0,
        Some(measured),
        cache_id,
        candidate_bytes,
        0,
        0,
        0,
        0,
        false,
    );
    cache_id
}

/// Record one read against the exact created cache identity. `transient_bytes` is a physical
/// allocator delta, never a tensor element-count surrogate.
pub(crate) fn record_cache_read(
    cache_id: u64,
    allocator: ActiveAllocatorMeasurement,
    measured: Instant,
) {
    if cache_id == 0 || !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
    observe_timed_for_cache(
        "cross-kv-read",
        0,
        allocator.transient_bytes(),
        1,
        Some(measured),
        cache_id,
        0,
        allocator.used_before,
        allocator.used_after,
        allocator.used_high,
        allocator.reserved_after,
        allocator.available,
    );
}

pub(crate) fn release_cache(cache_id: u64) {
    if cache_id == 0 || !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
    let state = CACHE_STATES.with(|slot| slot.borrow_mut().remove(&cache_id));
    if let Some(state) = state {
        let allocator = allocator_remnant();
        COMPLETED_CACHE_READS.with(|slot| slot.borrow_mut().push(state.reads));
        observe_timed_for_cache(
            "cross-kv-released",
            state.dense_bytes,
            0,
            0,
            None,
            cache_id,
            state.candidate_bytes,
            allocator.used_before,
            allocator.used_after,
            allocator.used_high,
            allocator.reserved_after,
            allocator.available,
        );
    }
}

fn observe_release_remnant() {
    let allocator = allocator_remnant();
    observe_timed_for_cache(
        "released",
        0,
        0,
        0,
        None,
        0,
        0,
        allocator.used_before,
        allocator.used_after,
        allocator.used_high,
        allocator.reserved_after,
        allocator.available,
    );
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
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
    let shape = tensor_shape.into();
    let dtype = dtype.into();
    let mask = mask.into();
    let rope = rope.into();
    ACTIVE.with(|slot| {
        if let Some(observer) = slot.borrow_mut().as_mut() {
            let context = CONTEXT.with(|ctx| ctx.borrow().clone());
            let measured_peak = backend_peak_bytes();
            let peak_bytes = measured_peak.unwrap_or_else(|| {
                if context.is_some() && !cfg!(test) {
                    0
                } else {
                    persistent_bytes.saturating_add(transient_bytes)
                }
            });
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
                peak_bytes,
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
                cache_id: 0,
                candidate_persistent_bytes: 0,
                allocator_before_bytes: 0,
                allocator_after_bytes: 0,
                allocator_high_bytes: 0,
                allocator_reserved_bytes: 0,
                allocator_measurement_available: false,
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
        LAST_CAMPAIGN_CANCELLATION.with(|slot| slot.set(true));
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
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
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
        try_emit_bound_metadata();
    }
}

/// Confirm that live model-owned weights completed a K/V projection. Geometry binding alone is
/// insufficient: it happens before projection and cannot prove that a snapshot-backed model loaded
/// successfully. Tests deliberately remain non-promotable even when their synthetic tensors project.
pub(crate) fn confirm_real_weight_projection() {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
    let bound = CONTEXT.with(|slot| {
        if let Some(context) = slot.borrow_mut().as_mut() {
            context.real_weights = !cfg!(test);
            true
        } else {
            false
        }
    });
    if !bound {
        return;
    }
    try_emit_bound_metadata();
}

fn try_emit_bound_metadata() {
    let cancellation = CAMPAIGN_CANCEL.with(|slot| *slot.borrow());
    observe("metadata", 0, 0, u64::from(cancellation));
    if METADATA_EMITTED.with(|slot| *slot.borrow())
        && !START_EVENT_EMITTED.with(|slot| *slot.borrow())
        && !PENDING_START.with(|slot| *slot.borrow())
    {
        observe("generation-start", 0, 0, 0);
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        let unconfirmed_cancel = CAMPAIGN_CANCEL.with(|slot| *slot.borrow())
            && CANCEL_TRIGGERED.with(|slot| *slot.borrow())
            && !LAST_CAMPAIGN_CANCELLATION.with(Cell::get);
        if !TERMINAL_EMITTED.with(|slot| *slot.borrow()) && !unconfirmed_cancel {
            observe("generation-end", 0, 0, 0);
        }
        observe("invalidated", 0, 0, 0);
        observe_release_remnant();
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
        CACHE_STATES.with(|slot| slot.borrow_mut().clear());
        COMPLETED_CACHE_READS.with(|slot| slot.borrow_mut().clear());
        LIVE_PERSISTENT.with(|slot| *slot.borrow_mut() = 0);
        PEAK_PERSISTENT.with(|slot| *slot.borrow_mut() = 0);
        LIVE_CANDIDATE.with(|slot| *slot.borrow_mut() = 0);
        PEAK_CANDIDATE.with(|slot| *slot.borrow_mut() = 0);
        CACHE_STATES.with(|slot| slot.borrow_mut().clear());
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
    fn accounting_tracks_simultaneous_residency_and_sums_all_reads() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let _scope = install(Box::new(Sink(out)));
        observe("cross-kv-created", 100, 0, 1);
        observe("cross-kv-created", 150, 0, 1);
        observe("cross-kv-released", 100, 0, 0);
        observe("cross-kv-created", 50, 0, 1);
        observe_timed("cross-kv-read", 0, 32, 1, None);
        observe_timed("cross-kv-read", 0, 64, 1, None);

        assert_eq!(LIVE_PERSISTENT.with(|slot| *slot.borrow()), 200);
        assert_eq!(PEAK_PERSISTENT.with(|slot| *slot.borrow()), 250);
        assert_eq!(CURRENT_READ_TRANSIENT.with(|slot| *slot.borrow()), 64);
        assert_eq!(REUSED_REQUESTS.with(|slot| *slot.borrow()), 2);
        assert_eq!(
            CACHE_READ_DURATION_PRECISE_MS.with(|slot| *slot.borrow()),
            0.002
        );
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
    fn packed_projection_includes_metadata_and_dense_pending_key_tail() {
        // B=1,H=2,Skv=65,D=64: keys have one complete group and one dense f32
        // pending token; values have 130 channel-grouped rows.
        assert_eq!(
            checked_packed_group_affine_kv_bytes(1, 2, 65, 64, 32),
            Some(6_704)
        );
        assert_eq!(checked_packed_group_affine_kv_bytes(1, 2, 0, 64, 32), None);
        assert_eq!(
            checked_packed_group_affine_kv_bytes(u64::MAX, 2, 65, 64, 32),
            None
        );
    }

    #[test]
    fn reuse_is_the_minimum_reads_of_each_created_cache() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let _scope = install(Box::new(Sink(out.clone())));
        let first = register_cache(100, 50, Instant::now());
        let second = register_cache(200, 80, Instant::now());
        let sample = |before, after, high| ActiveAllocatorMeasurement {
            used_before: before,
            used_after: after,
            used_high: high,
            reserved_after: high,
            available: true,
        };
        record_cache_read(first, sample(1000, 1016, 1032), Instant::now());
        record_cache_read(first, sample(1032, 1048, 1064), Instant::now());
        record_cache_read(second, sample(1064, 1100, 1128), Instant::now());
        assert_eq!(
            CACHE_STATES.with(|slot| slot.borrow().values().map(|state| state.reads).min()),
            Some(1)
        );
        record_cache_read(second, sample(1128, 1160, 1192), Instant::now());
        assert_eq!(
            CACHE_STATES.with(|slot| slot.borrow().values().map(|state| state.reads).min()),
            Some(2)
        );
        assert_eq!(PEAK_PERSISTENT.with(|slot| *slot.borrow()), 300);
        assert_eq!(PEAK_CANDIDATE.with(|slot| *slot.borrow()), 130);
        release_cache(first);
        assert!(!CACHE_STATES.with(|slot| slot.borrow().contains_key(&first)));
        assert_eq!(
            COMPLETED_CACHE_READS.with(|slot| slot.borrow().clone()),
            vec![2]
        );
        assert_eq!(LIVE_PERSISTENT.with(|slot| *slot.borrow()), 200);
        assert_eq!(LIVE_CANDIDATE.with(|slot| *slot.borrow()), 80);
        let rows = out.borrow();
        assert!(rows.iter().any(|event| event.cache_id == first));
        assert!(rows.iter().any(|event| {
            event.cache_id == second
                && event.allocator_before_bytes == 1_128
                && event.allocator_after_bytes == 1_160
                && event.allocator_high_bytes == 1_192
                && event.allocator_measurement_available
        }));
    }

    #[test]
    fn observer_off_skips_string_conversion_and_accounting() {
        struct CountInto(std::rc::Rc<Cell<u32>>);
        impl From<CountInto> for String {
            fn from(value: CountInto) -> Self {
                value.0.set(value.0.get() + 1);
                "constructed".into()
            }
        }
        let conversions = std::rc::Rc::new(Cell::new(0));
        observe_tensor(
            "disabled",
            "disabled",
            1,
            2,
            3,
            CountInto(conversions.clone()),
            CountInto(conversions.clone()),
            CountInto(conversions.clone()),
            CountInto(conversions.clone()),
        );
        assert_eq!(conversions.get(), 0);
        assert_eq!(LIVE_PERSISTENT.with(|slot| *slot.borrow()), 0);
        assert_eq!(CURRENT_READ_TRANSIENT.with(|slot| *slot.borrow()), 0);
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
        LAST_CAMPAIGN_CANCELLATION.with(|slot| slot.set(false));
        CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = true);
        METADATA_EMITTED.with(|slot| *slot.borrow_mut() = true);
        observe("generation-start", 0, 0, 0);
        observe("generation-start", 0, 0, 0);
        observe("cross-kv-read", 1, 2, 1);
        observe("cross-kv-read", 1, 2, 1);
        assert!(!campaign_cancelled());
        observe_cancelled();
        assert!(campaign_cancelled());
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
    fn cancellation_trigger_without_product_error_never_claims_a_terminal() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let scope = install(Box::new(Sink(out.clone())));
        LAST_CAMPAIGN_CANCELLATION.with(|slot| slot.set(false));
        CAMPAIGN_CANCEL.with(|slot| *slot.borrow_mut() = true);
        METADATA_EMITTED.with(|slot| *slot.borrow_mut() = true);
        observe("generation-start", 0, 0, 0);
        observe("cross-kv-read", 1, 2, 1);
        drop(scope);
        let rows = out.borrow();
        assert!(!campaign_cancelled());
        assert!(!rows
            .iter()
            .any(|event| { matches!(event.phase, "cancelled" | "generation-end" | "metrics") }));
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
        confirm_real_weight_projection();
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
        assert!(lines.contains("\"real_weights\":false"));
        assert!(lines
            .lines()
            .any(|line| line.contains("released") && line.contains("t2v")));
        assert!(!lines.contains("config.json"));
    }
}
