//! Optional SC-20686 campaign observer. `None` is the production default.
//!
//! The observer records only facts produced by the FLUX.2 edit route.  Edit re-concatenates
//! reference tokens at every denoise step, so it has no persistent cross-request K/V cache.
use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::runtime::CancelFlag;
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
    pub allocator_before_bytes: u64,
    pub allocator_after_bytes: u64,
    pub allocator_high_bytes: u64,
    pub allocator_reserved_bytes: u64,
    pub allocator_measurement_available: bool,
}
pub trait CacheObserver {
    fn record(&mut self, event: CacheEvent);
}

/// Exact SC-20675 v2 projection for one `[B,H,Skv,D]` K/V payload: 2-bit codes,
/// group-32 f16 scale/zero metadata, and a dense f32 pending key tail.
pub fn checked_packed_group_affine_kv_bytes(
    batch: u64,
    heads: u64,
    tokens: u64,
    width: u64,
) -> Option<u64> {
    const GROUP: u64 = 32;
    if batch == 0 || heads == 0 || tokens == 0 || width == 0 {
        return None;
    }
    let rows = batch.checked_mul(heads)?;
    let complete_groups = tokens.checked_div(GROUP)?;
    let pending_tokens = tokens.checked_rem(GROUP)?;
    let key_codes = rows
        .checked_mul(complete_groups)?
        .checked_mul(GROUP.checked_mul(width)?.checked_add(3)?.checked_div(4)?)?;
    let key_metadata = rows
        .checked_mul(complete_groups)?
        .checked_mul(width)?
        .checked_mul(4)?;
    let key_pending = rows
        .checked_mul(pending_tokens)?
        .checked_mul(width)?
        .checked_mul(4)?;
    let value_rows = rows.checked_mul(tokens)?;
    let value_codes = value_rows.checked_mul(width.checked_add(3)?.checked_div(4)?)?;
    let value_metadata = value_rows
        .checked_mul(width.checked_add(GROUP - 1)?.checked_div(GROUP)?)?
        .checked_mul(4)?;
    key_codes
        .checked_add(key_metadata)?
        .checked_add(key_pending)?
        .checked_add(value_codes)?
        .checked_add(value_metadata)
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
    joint_attention_context_duration_ms: f64,
    reference_runtime_attribution_available: bool,
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
thread_local! { static CAMPAIGN_HANDLE: RefCell<Option<CancelFlag>> = const { RefCell::new(None) }; }
thread_local! { static METADATA_EMITTED: RefCell<bool> = const { RefCell::new(false) }; }
thread_local! { static START_EMITTED: RefCell<bool> = const { RefCell::new(false) }; }
#[derive(Clone, Default)]
struct Measurements {
    current_read_transient_bytes: u64,
    candidate_persistent_bytes: u64,
    candidate_read_transient_bytes: u64,
    joint_attention_context_duration_ns: u128,
    reused_requests: u64,
    projection_count: u64,
}
impl Measurements {
    const EMPTY: Self = Self {
        current_read_transient_bytes: 0,
        candidate_persistent_bytes: 0,
        candidate_read_transient_bytes: 0,
        joint_attention_context_duration_ns: 0,
        reused_requests: 0,
        projection_count: 0,
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn activate_requested(
    root: &Path,
    cancel: &CancelFlag,
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
    CAMPAIGN_HANDLE.with(|slot| *slot.borrow_mut() = Some(cancel.clone()));
    if cancellation {
        observe("campaign-cancellation-armed", 0, 0, 1);
    }
    Ok(Some(scope))
}

/// Binds the exact tensors and transformer contract used by the edit call; callers cannot forge it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_edit_geometry(
    layers: u32,
    batch: u32,
    heads: u32,
    head_dimension: u32,
    target_sq: u64,
    reference_skv: u64,
    latent_height: u32,
    latent_width: u32,
    dtype: impl Into<String>,
) {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
    let bound = CONTEXT.with(|slot| {
        let mut context_slot = slot.borrow_mut();
        let Some(context) = context_slot.as_mut() else {
            return false;
        };
        if layers == 0
            || heads == 0
            || head_dimension == 0
            || batch == 0
            || target_sq == 0
            || reference_skv == 0
            || latent_height == 0
            || latent_width == 0
        {
            return false;
        }
        context.geometry.layers = layers;
        context.geometry.batch = batch;
        context.geometry.heads = heads;
        context.geometry.head_dimension = head_dimension;
        context.geometry.sq = target_sq;
        context.geometry.skv = reference_skv;
        context.geometry.latent_height = latent_height;
        context.geometry.latent_width = latent_width;
        context.geometry.dtype = dtype.into();
        true
    });
    let _ = bound;
}

pub struct JsonlObserver(Box<dyn Write>);
impl CacheObserver for JsonlObserver {
    fn record(&mut self, event: CacheEvent) {
        let mut value = serde_json::json!({"phase": event.phase, "attention": event.attention, "operation": event.operation, "tensor_shape": event.tensor_shape, "dtype": event.dtype, "mask": event.mask, "rope": event.rope, "persistent_bytes": event.persistent_bytes, "transient_bytes": event.transient_bytes, "peak_bytes": event.peak_bytes, "reused": event.reused, "elapsed_ms": event.elapsed_ms, "at_ns": event.at_ns, "sample_kind": event.sample_kind, "allocator_before_bytes": event.allocator_before_bytes, "allocator_after_bytes": event.allocator_after_bytes, "allocator_high_bytes": event.allocator_high_bytes, "allocator_reserved_bytes": event.allocator_reserved_bytes, "allocator_measurement_available": event.allocator_measurement_available});
        if let Some(context) = &event.context {
            if event.phase == "metadata" {
                value["source_ref"] = serde_json::json!(context.source_ref);
                value["snapshot_sha256"] = serde_json::json!(context.snapshot_sha256);
                value["snapshot_bytes"] = serde_json::json!(context.snapshot_bytes);
                value["variant"] = serde_json::json!(context.variant);
                value["geometry"] = serde_json::json!({"batch": context.geometry.batch, "resolution": format!("{}x{}", context.geometry.width, context.geometry.height), "reference_count": context.geometry.reference_count, "frames": context.geometry.frames, "prompt": context.geometry.prompt_sha256, "guidance": context.geometry.guidance, "layers": context.geometry.layers, "heads": context.geometry.heads, "head_dimension": context.geometry.head_dimension, "sq": context.geometry.sq, "skv": context.geometry.skv, "dtype": context.geometry.dtype, "mask": context.geometry.mask, "rope": context.geometry.rope});
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
            value["joint_attention_context_duration_ms"] =
                serde_json::json!(metrics.joint_attention_context_duration_ms);
            value["reference_runtime_attribution_available"] =
                serde_json::json!(metrics.reference_runtime_attribution_available);
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

#[derive(Clone, Copy, Debug)]
struct ActiveAllocatorWindow {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    used_before: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ActiveAllocatorMeasurement {
    used_before: u64,
    used_after: u64,
    used_high: u64,
    reserved_after: u64,
    available: bool,
}

#[cfg(feature = "cuda")]
fn begin_active_allocator_window() -> Option<ActiveAllocatorWindow> {
    let pool = candle_gen::cuda_mempool::MemPool::device_default(0)?;
    let used_before = pool.used()?;
    if !pool.reset_used_high_water() || pool.used_high()? < used_before {
        return None;
    }
    Some(ActiveAllocatorWindow { used_before })
}

#[cfg(not(feature = "cuda"))]
fn begin_active_allocator_window() -> Option<ActiveAllocatorWindow> {
    None
}

#[cfg(feature = "cuda")]
fn finish_active_allocator_window(
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
fn finish_active_allocator_window(
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
    METADATA_EMITTED.with(|slot| *slot.borrow_mut() = false);
    START_EMITTED.with(|slot| *slot.borrow_mut() = false);
    Scope
}

pub(crate) struct FluxReferenceKv {
    dense_bytes: u64,
    shape: String,
    dtype: String,
}

pub(crate) struct FluxKvRead {
    started: Instant,
    dense_bytes: u64,
    shape: String,
    dtype: String,
    allocator_window: Option<ActiveAllocatorWindow>,
}

/// Narrow the actual image `to_k`/`to_v` outputs at the frozen target/reference token boundary.
/// Q and text K/V projections are not arguments and therefore cannot enter the byte attribution.
pub(crate) fn record_flux_kv_created(
    image_k: &Tensor,
    image_v: &Tensor,
) -> Option<FluxReferenceKv> {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return None;
    }
    let started = Instant::now();
    let (batch, heads, image_tokens, width) = image_k.dims4().ok()?;
    if image_v.dims4().ok()? != (batch, heads, image_tokens, width) {
        return None;
    }
    let (target_tokens, reference_tokens, layers) = CONTEXT.with(|slot| {
        let context = slot.borrow();
        let geometry = &context.as_ref()?.geometry;
        Some((
            geometry.sq as usize,
            geometry.skv as usize,
            u64::from(geometry.layers),
        ))
    })?;
    if image_tokens != target_tokens.checked_add(reference_tokens)? || layers == 0 {
        return None;
    }
    let reference_k = image_k.narrow(2, target_tokens, reference_tokens).ok()?;
    let reference_v = image_v.narrow(2, target_tokens, reference_tokens).ok()?;
    let exact_shape = (batch, heads, reference_tokens, width);
    if reference_k.dims4().ok()? != exact_shape
        || reference_v.dims4().ok()? != exact_shape
        || reference_k.dtype() != reference_v.dtype()
    {
        return None;
    }
    let dense_bytes = [&reference_k, &reference_v]
        .iter()
        .try_fold(0u64, |total, tensor| {
            total.checked_add(
                (tensor.elem_count() as u64).checked_mul(tensor.dtype().size_in_bytes() as u64)?,
            )
        })?;
    let candidate_total = checked_packed_group_affine_kv_bytes(
        batch as u64,
        heads as u64,
        reference_tokens as u64,
        width as u64,
    )?
    .checked_mul(layers)?;
    CONTEXT.with(|slot| {
        if let Some(context) = slot.borrow_mut().as_mut() {
            context.real_weights = !cfg!(test);
        }
    });
    if !METADATA_EMITTED.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), true)) {
        observe("metadata", 0, 0, 0);
    }
    if !START_EMITTED.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), true)) {
        observe("generation-start", 0, 0, 0);
    }
    MEASUREMENTS.with(|metrics| {
        let mut metrics = metrics.borrow_mut();
        metrics.candidate_persistent_bytes = candidate_total;
        metrics.projection_count = metrics.projection_count.saturating_add(1);
    });
    let shape = format!(
        "reference_k={:?};reference_v={:?}",
        reference_k.dims(),
        reference_v.dims()
    );
    let dtype = format!("{:?}", reference_k.dtype());
    observe_tensor(
        "cross-kv-created",
        "DoubleAttention::to_k/to_v(reference-slice)",
        0,
        dense_bytes,
        0,
        shape.clone(),
        dtype.clone(),
        "joint-unmasked",
        "flux2-4-axis",
        Some(started),
    );
    Some(FluxReferenceKv {
        dense_bytes,
        shape,
        dtype,
    })
}

/// Begin the exact product attention read after all unrelated Q/text projection and RoPE work.
pub(crate) fn begin_flux_kv_read(reference: Option<FluxReferenceKv>) -> Option<FluxKvRead> {
    let reference = reference?;
    Some(FluxKvRead {
        started: Instant::now(),
        dense_bytes: reference.dense_bytes,
        shape: reference.shape,
        dtype: reference.dtype,
        allocator_window: begin_active_allocator_window(),
    })
}

/// Finish the product attention read. The active allocator high-water is retained as exact
/// evidence, but the shared joint-attention workspace is not credited as reference-cache savings.
/// The elapsed duration covers fused text + target + reference attention and is therefore sealed
/// only as non-attributable context. With no packed reader wired, the candidate must materialize
/// the same exact dense reference K/V.
pub(crate) fn record_flux_kv_read(measurement: Option<FluxKvRead>) {
    let Some(measurement) = measurement else {
        return;
    };
    let allocator = finish_active_allocator_window(measurement.allocator_window);
    let current_transient = measurement.dense_bytes;
    MEASUREMENTS.with(|metrics| {
        let mut metrics = metrics.borrow_mut();
        metrics.current_read_transient_bytes =
            metrics.current_read_transient_bytes.max(current_transient);
        metrics.candidate_read_transient_bytes = metrics
            .candidate_read_transient_bytes
            .max(measurement.dense_bytes);
        metrics.joint_attention_context_duration_ns = metrics
            .joint_attention_context_duration_ns
            .saturating_add(measurement.started.elapsed().as_nanos());
        let layers = CONTEXT.with(|slot| {
            slot.borrow()
                .as_ref()
                .map_or(0, |context| u64::from(context.geometry.layers))
        });
        metrics.reused_requests = metrics.projection_count.checked_div(layers).unwrap_or(0);
    });
    observe_tensor_with_allocator(
        "cross-kv-read",
        "DoubleAttention::attention(joint-context-non-attributable)",
        0,
        current_transient,
        1,
        measurement.shape,
        measurement.dtype,
        "joint-unmasked",
        "flux2-4-axis",
        Some(measurement.started),
        allocator,
    );
    let armed = CONTEXT.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|context| context.cancellation_armed)
    });
    if armed
        && !CANCELLATION_TRIGGERED.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), true))
    {
        CAMPAIGN_HANDLE.with(|slot| {
            if let Some(cancel) = slot.borrow().as_ref() {
                cancel.cancel();
            }
        });
    }
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
#[allow(clippy::too_many_arguments)]
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
    observe_tensor_with_allocator(
        phase,
        operation,
        persistent_bytes,
        transient_bytes,
        reused,
        tensor_shape,
        dtype,
        mask,
        rope,
        measured,
        ActiveAllocatorMeasurement::default(),
    );
}

#[allow(clippy::too_many_arguments)]
fn observe_tensor_with_allocator(
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
    allocator: ActiveAllocatorMeasurement,
) {
    if !ACTIVE.with(|slot| slot.borrow().is_some()) {
        return;
    }
    ACTIVE.with(|slot| {
        let mut observer_slot = slot.borrow_mut();
        let Some(observer) = observer_slot.as_mut() else {
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
        if phase == "cross-kv-read"
            && operation != "DoubleAttention::attention(joint-context-non-attributable)"
        {
            MEASUREMENTS.with(|metrics| {
                let mut metrics = metrics.borrow_mut();
                metrics.current_read_transient_bytes =
                    metrics.current_read_transient_bytes.max(transient_bytes);
                metrics.joint_attention_context_duration_ns = metrics
                    .joint_attention_context_duration_ns
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
            allocator_before_bytes: allocator.used_before,
            allocator_after_bytes: allocator.used_after,
            allocator_high_bytes: allocator.used_high,
            allocator_reserved_bytes: allocator.reserved_after,
            allocator_measurement_available: allocator.available,
        };
        observer.record(event.clone());
        if phase == "generation-end" || phase == "cancelled" {
            let metrics = MEASUREMENTS.with(|measurement| {
                let measurement = measurement.borrow().clone();
                CampaignMetrics {
                    current_persistent_bytes: 0,
                    current_read_transient_bytes: measurement.current_read_transient_bytes,
                    candidate_persistent_bytes: measurement.candidate_persistent_bytes,
                    candidate_read_transient_bytes: measurement.candidate_read_transient_bytes,
                    generation_duration_ms: STARTED.with(|started| {
                        started
                            .borrow()
                            .as_ref()
                            .map_or(0.0, |instant| instant.elapsed().as_secs_f64() * 1_000.0)
                    }),
                    // FLUX performs one fused joint attention over text, target image, and
                    // reference image tokens. Its duration is useful execution context but cannot
                    // be isolated as reference-K/V runtime, so the attributable duration is zero.
                    cache_read_duration_ms: 0.0,
                    joint_attention_context_duration_ms: measurement
                        .joint_attention_context_duration_ns
                        as f64
                        / 1_000_000.0,
                    reference_runtime_attribution_available: false,
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

fn observe_release_remnant() {
    observe_tensor_with_allocator(
        "released",
        "release-remnant",
        0,
        0,
        0,
        "",
        "",
        "",
        "",
        None,
        allocator_remnant(),
    );
}
pub fn observe_cancelled() {
    observe("cancelled", 0, 0, 0);
}
impl Drop for Scope {
    fn drop(&mut self) {
        observe("invalidated", 0, 0, 0);
        observe_release_remnant();
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        CONTEXT.with(|slot| *slot.borrow_mut() = None);
        STARTED.with(|slot| *slot.borrow_mut() = None);
        CANCELLATION_TRIGGERED.with(|slot| *slot.borrow_mut() = false);
        CAMPAIGN_HANDLE.with(|slot| *slot.borrow_mut() = None);
        METADATA_EMITTED.with(|slot| *slot.borrow_mut() = false);
        START_EMITTED.with(|slot| *slot.borrow_mut() = false);
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
    fn observer_off_projection_is_inert_and_group32_projection_is_exact() {
        assert!(begin_flux_kv_read(None).is_none());
        assert_eq!(MEASUREMENTS.with(|slot| slot.borrow().projection_count), 0);
        assert_eq!(
            checked_packed_group_affine_kv_bytes(1, 2, 65, 64),
            Some(6_704)
        );
    }

    #[test]
    fn reference_slice_excludes_target_and_unrelated_projections() {
        let out = std::rc::Rc::new(RefCell::new(Vec::new()));
        let context = CampaignContext {
            source_ref: "a".repeat(40),
            snapshot_sha256: "b".repeat(64),
            snapshot_bytes: 1,
            variant: PRODUCT_ROUTE_ID.into(),
            cancellation_armed: false,
            real_weights: false,
            geometry: CampaignGeometry {
                batch: 1,
                frames: 1,
                width: 32,
                height: 32,
                latent_frames: 1,
                latent_height: 1,
                latent_width: 1,
                prompt_sha256: "c".repeat(64),
                guidance: "1".into(),
                reference_count: 1,
                layers: 2,
                heads: 2,
                head_dimension: 4,
                sq: 5,
                skv: 3,
                dtype: "F32".into(),
                mask: "joint-unmasked".into(),
                rope: "flux2-4-axis".into(),
            },
        };
        let _scope = install_with_context(Box::new(Sink(out.clone())), context);
        let image_k = Tensor::zeros(
            (1, 2, 8, 4),
            candle_gen::candle_core::DType::F32,
            &candle_gen::candle_core::Device::Cpu,
        )
        .unwrap();
        let image_v = Tensor::zeros(
            (1, 2, 8, 4),
            candle_gen::candle_core::DType::F32,
            &candle_gen::candle_core::Device::Cpu,
        )
        .unwrap();
        let reference = record_flux_kv_created(&image_k, &image_v).expect("reference slice");
        assert_eq!(reference.dense_bytes, 2 * 2 * 3 * 4 * 4);
        let created = out
            .borrow()
            .iter()
            .find(|event| event.phase == "cross-kv-created")
            .cloned()
            .unwrap();
        assert_eq!(created.transient_bytes, reference.dense_bytes);
        assert!(created.tensor_shape.contains("[1, 2, 3, 4]"));
        assert!(!created.tensor_shape.contains("[1, 2, 8, 4]"));
        assert_eq!(
            created.operation,
            "DoubleAttention::to_k/to_v(reference-slice)"
        );
    }
}
