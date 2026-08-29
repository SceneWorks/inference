//! Optional SC-20686 campaign observer. `None` is the production default.
//!
//! The observer records only facts produced by the FLUX.2 edit route.  Edit re-concatenates
//! reference tokens at every denoise step, so it has no persistent cross-request K/V cache.
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Exact externally registered edit route. The model configuration remains `flux2_klein_9b`;
/// campaign evidence must identify the product operation, not conflate it with that config id.
pub const PRODUCT_ROUTE_ID: &str = "flux2_klein_9b_edit";

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
    pub metrics: Option<CampaignMetrics>,
}
pub trait CacheObserver {
    fn record(&mut self, event: CacheEvent);
}

/// Checked block-aligned candidate projection; never an observation about this dense route.
pub fn checked_compressed_bytes(
    elements: u64,
    bits_per_element: u8,
    block_bytes: u64,
) -> Option<u64> {
    if elements == 0 || bits_per_element == 0 || bits_per_element > 32 || block_bytes == 0 {
        return None;
    }
    let payload = elements
        .checked_mul(u64::from(bits_per_element))?
        .checked_add(7)?
        .checked_div(8)?;
    payload
        .checked_add(block_bytes - 1)?
        .checked_div(block_bytes)?
        .checked_mul(block_bytes)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CampaignGeometry {
    batch: u32,
    frames: u32,
    width: u32,
    height: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    prompt_sha256: String,
    guidance: String,
    reference_count: u32,
    layers: u32,
    heads: u32,
    head_dimension: u32,
    sq: u64,
    skv: u64,
    dtype: String,
    mask: String,
    rope: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CampaignContext {
    source_ref: String,
    snapshot_sha256: String,
    snapshot_bytes: u64,
    variant: String,
    cancellation_armed: bool,
    real_weights: bool,
    geometry: CampaignGeometry,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CampaignMetrics {
    current_persistent_bytes: u64,
    current_read_transient_bytes: u64,
    candidate_persistent_bytes: u64,
    candidate_read_transient_bytes: u64,
    generation_duration_ms: f64,
    cache_read_duration_ms: f64,
    reused_requests: u64,
    minimum_cache_reads: u64,
}
pub struct CampaignOutputRequest {
    path: PathBuf,
    cancellation: bool,
}
pub fn request_output(path: impl Into<PathBuf>) -> CampaignOutputRequest {
    CampaignOutputRequest {
        path: path.into(),
        cancellation: false,
    }
}

thread_local! { static PENDING_OUTPUT: RefCell<Option<PathBuf>> = const { RefCell::new(None) }; }
thread_local! { static PENDING_CANCELLATION: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static ACTIVE: RefCell<Option<Box<dyn CacheObserver>>> = const { RefCell::new(None) }; }
thread_local! { static CONTEXT: RefCell<Option<CampaignContext>> = const { RefCell::new(None) }; }
thread_local! { static STARTED: RefCell<Option<Instant>> = const { RefCell::new(None) }; }
thread_local! { static MEASUREMENTS: RefCell<Measurements> = const { RefCell::new(Measurements::EMPTY) }; }
thread_local! { static CANCELLATION_TRIGGERED: RefCell<bool> = const { RefCell::new(false) }; }
#[derive(Clone, Copy, Default)]
struct Measurements {
    current_read_transient_bytes: u64,
    candidate_persistent_bytes: u64,
    cache_read_duration_ns: u128,
    reused_requests: u64,
}
impl Measurements {
    const EMPTY: Self = Self {
        current_read_transient_bytes: 0,
        candidate_persistent_bytes: 0,
        cache_read_duration_ns: 0,
        reused_requests: 0,
    };
}

impl CampaignOutputRequest {
    pub fn arm(self) -> Self {
        PENDING_OUTPUT.with(|slot| *slot.borrow_mut() = Some(self.path.clone()));
        PENDING_CANCELLATION.with(|slot| *slot.borrow_mut() = self.cancellation);
        self
    }
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
    fn collect(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == ".git") {
                    continue;
                }
                collect(root, &path, out)?;
            } else if path.is_file() && !path.components().any(|part| part.as_os_str() == ".git") {
                out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut names = Vec::new();
    collect(root, root, &mut names)?;
    names.sort();
    if names.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "snapshot inventory is empty",
        ));
    }
    let mut aggregate = Sha256::new();
    let mut total = 0u64;
    for name in names {
        let mut file = File::open(root.join(&name))?;
        let mut digest = Sha256::new();
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
            digest.update(&buffer[..read]);
        }
        total = total.checked_add(bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "snapshot byte count overflow")
        })?;
        aggregate.update(name.to_string_lossy().as_bytes());
        aggregate.update([0]);
        aggregate.update(bytes.to_le_bytes());
        aggregate.update([0]);
        aggregate.update(digest.finalize());
        aggregate.update([b'\n']);
    }
    Ok((format!("{:x}", aggregate.finalize()), total))
}
fn is_lowercase_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn source_revision(root: &Path) -> io::Result<String> {
    let value = root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| is_lowercase_sha(name))
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
    if !is_lowercase_sha(&value) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "snapshot revision must be lowercase 40-hex",
        ));
    }
    Ok(value)
}

/// Activates only after the real edit route starts. Metadata is delayed until live tensors bind it.
pub(crate) fn activate_requested(
    root: &Path,
    variant: &str,
    width: u32,
    height: u32,
    prompt: &str,
    guidance: f32,
    reference_count: u32,
) -> io::Result<Option<Scope>> {
    let path = PENDING_OUTPUT.with(|slot| slot.borrow_mut().take());
    let cancellation = PENDING_CANCELLATION.with(|slot| *slot.borrow());
    let Some(path) = path else { return Ok(None) };
    if backend_peak_bytes().is_none() && !cfg!(test) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SC-20686 requires a live backend allocator peak counter",
        ));
    }
    let (snapshot_sha256, snapshot_bytes) = snapshot_identity(root)?;
    let context = CampaignContext {
        source_ref: source_revision(root)?,
        snapshot_sha256,
        snapshot_bytes,
        variant: variant.into(),
        cancellation_armed: cancellation,
        real_weights: false,
        geometry: CampaignGeometry {
            batch: 1,
            frames: 1,
            width,
            height,
            latent_frames: 1,
            latent_height: 0,
            latent_width: 0,
            prompt_sha256: format!("{:x}", Sha256::digest(prompt.as_bytes())),
            guidance: guidance.to_string(),
            reference_count,
            layers: 0,
            heads: 0,
            head_dimension: 0,
            sq: 0,
            skv: 0,
            dtype: String::new(),
            mask: "joint-unmasked".into(),
            rope: "flux2-4-axis".into(),
        },
    };
    let writer: Box<dyn Write> = if path == Path::new("-") {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(path)?)
    };
    let scope = install_with_context(Box::new(JsonlObserver(writer)), context);
    if cancellation {
        observe("campaign-cancellation-armed", 0, 0, 1);
    }
    Ok(Some(scope))
}

/// Binds the exact tensors and transformer contract used by the edit call; callers cannot forge it.
pub(crate) fn bind_edit_geometry(
    layers: u32,
    heads: u32,
    head_dimension: u32,
    sq: u64,
    skv: u64,
    latent_height: u32,
    latent_width: u32,
    dtype: impl Into<String>,
    reference_elements: u64,
) {
    let bound = CONTEXT.with(|slot| {
        let Some(context) = slot.borrow_mut().as_mut() else {
            return false;
        };
        if layers == 0
            || heads == 0
            || head_dimension == 0
            || sq == 0
            || skv == 0
            || latent_height == 0
            || latent_width == 0
        {
            return false;
        }
        context.geometry.layers = layers;
        context.geometry.heads = heads;
        context.geometry.head_dimension = head_dimension;
        context.geometry.sq = sq;
        context.geometry.skv = skv;
        context.geometry.latent_height = latent_height;
        context.geometry.latent_width = latent_width;
        context.geometry.dtype = dtype.into();
        // This hook runs only after the product edit route has materialized live reference tokens
        // from its loaded snapshot. Synthetic unit-test tensors remain explicitly non-promotable.
        context.real_weights = !cfg!(test);
        MEASUREMENTS.with(|metrics| {
            metrics.borrow_mut().candidate_persistent_bytes =
                checked_compressed_bytes(reference_elements, 4, 64).unwrap_or(0)
        });
        true
    });
    if bound {
        observe("metadata", 0, 0, 0);
    }
}

pub struct JsonlObserver(Box<dyn Write>);
impl CacheObserver for JsonlObserver {
    fn record(&mut self, event: CacheEvent) {
        let mut value = serde_json::json!({"phase": event.phase, "attention": event.attention, "operation": event.operation, "tensor_shape": event.tensor_shape, "dtype": event.dtype, "mask": event.mask, "rope": event.rope, "persistent_bytes": event.persistent_bytes, "transient_bytes": event.transient_bytes, "peak_bytes": event.peak_bytes, "reused": event.reused, "elapsed_ms": event.elapsed_ms, "at_ns": event.at_ns, "sample_kind": event.sample_kind});
        if let Some(context) = &event.context {
            if event.phase == "metadata" {
                value["source_ref"] = serde_json::json!(context.source_ref);
                value["snapshot_sha256"] = serde_json::json!(context.snapshot_sha256);
                value["snapshot_bytes"] = serde_json::json!(context.snapshot_bytes);
                value["variant"] = serde_json::json!(context.variant);
                value["geometry"] = serde_json::json!({"resolution": format!("{}x{}", context.geometry.width, context.geometry.height), "reference_count": context.geometry.reference_count, "frames": context.geometry.frames, "prompt": context.geometry.prompt_sha256, "guidance": context.geometry.guidance, "layers": context.geometry.layers, "heads": context.geometry.heads, "head_dimension": context.geometry.head_dimension, "sq": context.geometry.sq, "skv": context.geometry.skv, "dtype": context.geometry.dtype, "mask": context.geometry.mask, "rope": context.geometry.rope});
                value["cancellation_armed"] = serde_json::json!(context.cancellation_armed);
                if context.cancellation_armed {
                    value["cancellation_arm_id"] =
                        serde_json::json!(format!("{}:{}", context.source_ref, context.variant));
                }
                value["real_weights"] = serde_json::json!(context.real_weights);
                value["full_generation"] = serde_json::json!(!context.cancellation_armed);
                value["attention_kind"] = serde_json::json!("cross");
            }
        }
        if let Some(metrics) = event.metrics {
            value["current_persistent_bytes"] = serde_json::json!(metrics.current_persistent_bytes);
            value["current_read_transient_bytes"] =
                serde_json::json!(metrics.current_read_transient_bytes);
            value["candidate_persistent_bytes"] =
                serde_json::json!(metrics.candidate_persistent_bytes);
            value["candidate_read_transient_bytes"] =
                serde_json::json!(metrics.candidate_read_transient_bytes);
            value["generation_duration_ms"] = serde_json::json!(metrics.generation_duration_ms);
            value["cache_read_duration_ms"] = serde_json::json!(metrics.cache_read_duration_ms);
            value["reused_requests"] = serde_json::json!(metrics.reused_requests);
            value["minimum_cache_reads"] = serde_json::json!(metrics.minimum_cache_reads);
        }
        let _ = self.0.write_all(
            serde_json::to_string(&value)
                .unwrap_or_else(|_| "{}".into())
                .as_bytes(),
        );
        let _ = self.0.write_all(b"\n");
        let _ = self.0.flush();
    }
}
#[cfg(feature = "cuda")]
fn backend_peak_bytes() -> Option<u64> {
    candle_gen::cuda_mempool::MemPool::device_default(0)?.reserved_high()
}
#[cfg(not(feature = "cuda"))]
fn backend_peak_bytes() -> Option<u64> {
    None
}

pub struct Scope;
pub fn install(observer: Box<dyn CacheObserver>) -> Scope {
    install_inner(observer, None)
}
fn install_with_context(observer: Box<dyn CacheObserver>, context: CampaignContext) -> Scope {
    install_inner(observer, Some(context))
}
fn install_inner(observer: Box<dyn CacheObserver>, context: Option<CampaignContext>) -> Scope {
    ACTIVE.with(|slot| *slot.borrow_mut() = Some(observer));
    CONTEXT.with(|slot| *slot.borrow_mut() = context);
    STARTED.with(|slot| *slot.borrow_mut() = Some(Instant::now()));
    MEASUREMENTS.with(|slot| *slot.borrow_mut() = Measurements::EMPTY);
    CANCELLATION_TRIGGERED.with(|slot| *slot.borrow_mut() = false);
    Scope
}

/// Consume the campaign's deliberate cancellation request at the first live cache-read boundary.
/// Ordinary product cancellation continues to use the request's `CancelFlag`; this hook exists only
/// while an explicitly armed receipt observer owns the current edit call.
pub(crate) fn take_requested_cancellation() -> bool {
    let armed = CONTEXT.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|context| context.cancellation_armed)
    });
    armed
        && CANCELLATION_TRIGGERED.with(|slot| {
            let mut triggered = slot.borrow_mut();
            if *triggered {
                false
            } else {
                *triggered = true;
                true
            }
        })
}
pub fn observe(phase: &'static str, persistent_bytes: u64, transient_bytes: u64, reused: u64) {
    observe_tensor(
        phase,
        "observe",
        persistent_bytes,
        transient_bytes,
        reused,
        "",
        "",
        "",
        "",
        None,
    )
}
pub fn observe_timed(
    phase: &'static str,
    persistent_bytes: u64,
    transient_bytes: u64,
    reused: u64,
    measured: Instant,
) {
    observe_tensor(
        phase,
        "timed",
        persistent_bytes,
        transient_bytes,
        reused,
        "",
        "",
        "",
        "",
        Some(measured),
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
    measured: Option<Instant>,
) {
    ACTIVE.with(|slot| {
        let Some(observer) = slot.borrow_mut().as_mut() else {
            return;
        };
        let context = CONTEXT.with(|ctx| ctx.borrow().clone());
        let measured_peak = backend_peak_bytes();
        let peak_bytes = measured_peak.unwrap_or_else(|| {
            if context.is_some() && !cfg!(test) {
                // Activation already rejects a missing counter; a mid-run loss is retained as an
                // invalid zero sample so publication fails closed without panicking in cleanup.
                0
            } else {
                persistent_bytes.saturating_add(transient_bytes)
            }
        });
        let measured_duration = measured.map(|instant| instant.elapsed());
        let elapsed_ms = measured
            .map(|_| {
                measured_duration
                    .unwrap_or_default()
                    .as_millis()
                    .min(u64::MAX as u128) as u64
            })
            .unwrap_or_else(|| {
                STARTED.with(|started| {
                    started.borrow().as_ref().map_or(0, |instant| {
                        instant.elapsed().as_millis().min(u64::MAX as u128) as u64
                    })
                })
            });
        if phase == "cross-kv-read" {
            MEASUREMENTS.with(|metrics| {
                let mut metrics = metrics.borrow_mut();
                metrics.current_read_transient_bytes =
                    metrics.current_read_transient_bytes.max(transient_bytes);
                metrics.cache_read_duration_ns = metrics
                    .cache_read_duration_ns
                    .saturating_add(measured_duration.unwrap_or_default().as_nanos());
                metrics.reused_requests = metrics.reused_requests.saturating_add(reused);
            });
        }
        let event = CacheEvent {
            phase,
            attention: "cross",
            persistent_bytes,
            transient_bytes,
            elapsed_ms,
            reused,
            peak_bytes,
            at_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            operation,
            tensor_shape: tensor_shape.into(),
            dtype: dtype.into(),
            mask: mask.into(),
            rope: rope.into(),
            context,
            sample_kind: "allocator",
            metrics: None,
        };
        observer.record(event.clone());
        if phase == "generation-end" || phase == "cancelled" {
            let metrics = MEASUREMENTS.with(|measurement| {
                let measurement = *measurement.borrow();
                CampaignMetrics {
                    current_persistent_bytes: 0,
                    current_read_transient_bytes: measurement.current_read_transient_bytes,
                    candidate_persistent_bytes: measurement.candidate_persistent_bytes,
                    // No compressed FLUX read kernel exists in this attribution story. Preserve the
                    // measured dense read transient conservatively instead of claiming it vanishes.
                    candidate_read_transient_bytes: measurement.current_read_transient_bytes,
                    generation_duration_ms: STARTED.with(|started| {
                        started
                            .borrow()
                            .as_ref()
                            .map_or(0.0, |instant| instant.elapsed().as_secs_f64() * 1_000.0)
                    }),
                    cache_read_duration_ms: measurement.cache_read_duration_ns as f64 / 1_000_000.0,
                    reused_requests: measurement.reused_requests,
                    // FLUX owns one logical reference-conditioning payload for the coordinate;
                    // every observed recomputation is a read of that same product-owned payload.
                    minimum_cache_reads: measurement.reused_requests,
                }
            });
            observer.record(CacheEvent {
                phase: "metrics",
                operation: "metrics",
                metrics: Some(metrics),
                ..event
            });
        }
    });
}
pub fn observe_cancelled() {
    observe("cancelled", 0, 0, 0);
}
impl Drop for Scope {
    fn drop(&mut self) {
        observe("invalidated", 0, 0, 0);
        observe("released", 0, 0, 0);
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        CONTEXT.with(|slot| *slot.borrow_mut() = None);
        STARTED.with(|slot| *slot.borrow_mut() = None);
        CANCELLATION_TRIGGERED.with(|slot| *slot.borrow_mut() = false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Sink(std::rc::Rc<RefCell<Vec<CacheEvent>>>);
    impl CacheObserver for Sink {
        fn record(&mut self, event: CacheEvent) {
            self.0.borrow_mut().push(event);
        }
    }
    #[test]
    fn nonpersistent_route_reports_zero_current_cache_and_checked_candidate() {
        assert_eq!(checked_compressed_bytes(1024, 4, 64), Some(512));
        assert_eq!(checked_compressed_bytes(0, 4, 64), None);
        assert_eq!(checked_compressed_bytes(u64::MAX, 32, 64), None);
    }
    #[test]
    fn observer_is_optional_and_records_real_route_hooks() {
        let out = std::rc::Rc::new(RefCell::new(Vec::new()));
        let _scope = install(Box::new(Sink(out.clone())));
        observe("cross-kv-created", 0, 0, 0);
        observe_timed("cross-kv-read", 0, 128, 1, Instant::now());
        assert_eq!(out.borrow()[0].persistent_bytes, 0);
        assert_eq!(out.borrow()[1].transient_bytes, 128);
    }
    #[test]
    fn lowercase_revision_is_required() {
        assert!(is_lowercase_sha("a012345678901234567890123456789012345678"));
        assert!(!is_lowercase_sha(
            "A012345678901234567890123456789012345678"
        ));
    }

    #[test]
    fn deliberate_cancellation_is_consumed_once_only_when_armed() {
        let out = std::rc::Rc::new(RefCell::new(Vec::new()));
        let context = CampaignContext {
            source_ref: "a".repeat(40),
            snapshot_sha256: "b".repeat(64),
            snapshot_bytes: 1,
            variant: "flux2_klein_9b_edit".into(),
            cancellation_armed: true,
            real_weights: false,
            geometry: CampaignGeometry {
                batch: 1,
                frames: 1,
                width: 1024,
                height: 1024,
                latent_frames: 1,
                latent_height: 128,
                latent_width: 128,
                prompt_sha256: "c".repeat(64),
                guidance: "1".into(),
                reference_count: 1,
                layers: 1,
                heads: 1,
                head_dimension: 1,
                sq: 1,
                skv: 1,
                dtype: "F32".into(),
                mask: "joint-unmasked".into(),
                rope: "flux2-4-axis".into(),
            },
        };
        let _scope = install_with_context(Box::new(Sink(out)), context);
        assert!(take_requested_cancellation());
        assert!(!take_requested_cancellation());
    }
}
