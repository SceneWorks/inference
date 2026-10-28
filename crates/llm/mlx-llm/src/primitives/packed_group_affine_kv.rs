//! Physically packed group-affine KV representation (SC-20675) and its SC-20676 decoder route.
//!
//! Codes are [`PackedCodeBits`] wide (2-bit: four per byte, the SC-20673 qualified candidate;
//! 4-bit: two per byte, the KIVI-style candidate; 8-bit: one per byte) and each group has an f16
//! scale and zero. The
//! code width is a property of the cache and of its reader; a snapshot records it and a reader or
//! snapshot of another width is refused. Two stores share one device layout:
//!
//! * The host CPU reference ([`PackedGroupAffineKvCache::append`]) quantizes on the CPU, owns the
//!   deterministic snapshot/lifecycle contract, and uploads only its new packed deltas into a
//!   block-preallocated device mirror when a reader dispatches.
//! * The decoder route ([`PackedGroupAffineKvCache::append_device`], driven by
//!   [`DenseFallbackPackedDecoderCache`]) keeps K/V only on the device: every completed 32-token
//!   group is quantized on the GPU (bit-identically to the CPU reference) and written in place;
//!   the incomplete group stays in a bounded dense residual that the reader takes as a separate
//!   input. Nothing is read back or evaluated per layer, and no host copy exists.
//!
//! A dense reader is an explicit, instrumented fallback; neither store retains a dense mirror.

use std::any::Any;
use std::convert::TryInto;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::primitives::kv_cache::{
    CacheRoute, CompressedCacheStorage, ContiguousKvCache, KvCache, PackedAttentionMask,
    PackedCacheEvidence, KV_BLOCK_TOKENS,
};
use crate::primitives::packed_metal::{
    quantize_group_affine_flush, PackedAttentionArgs, PackedKernelSelection,
    PagedPackedAttentionArgs,
};
use half::f16;
use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
use mlx_rs::ops::{add, concatenate_axis, floor_divide, multiply, remainder, zeros_dtype};
use mlx_rs::{Array, Dtype};

const MAGIC: &[u8; 8] = b"SW20675\0";
const VERSION: u32 = 2;
/// SC-20673 qualified the `b=2,g=32` group-affine candidate; `b=4` and `b=8` share the group. K groups
/// span tokens and V groups span channels, but both use this same quantization group width.
pub const PACKED_METAL_QUANT_GROUP_SIZE: usize = 32;

/// Code width of a packed group-affine cache. Codes of one width are packed little-end-first into
/// bytes (`8 / bits` per byte); a group's `2^bits` levels span `[min, max]` as
/// `zero + scale · code` with `scale = max((max − min) / (2^bits − 1), ε)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PackedCodeBits {
    /// SC-20673 qualified `b=2` (four codes per byte).
    #[default]
    Two,
    /// KIVI-style `b=4` (two codes per byte).
    Four,
    /// `b=8` (one code per byte).
    Eight,
}

impl PackedCodeBits {
    pub const ALL: [Self; 3] = [Self::Two, Self::Four, Self::Eight];

    pub const fn bits(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Supported widths only; any other value is refused.
    pub fn from_bits(bits: u8) -> Result<Self> {
        match bits {
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            other => Err(Error::Config(format!(
                "packed group-affine code width {other} is unsupported (expected 2, 4, or 8)"
            ))),
        }
    }

    /// Independent codes packed into one byte. A packing property, not the affine group.
    pub const fn codes_per_byte(self) -> usize {
        8 / self.bits() as usize
    }

    /// Largest code (`2^bits − 1`).
    pub const fn max_code(self) -> u8 {
        ((1u16 << self.bits()) - 1) as u8
    }

    /// Quantization step divisor `2^bits − 1`, exactly as the GPU quantizer uses it.
    pub fn levels(self) -> f32 {
        f32::from(self.max_code())
    }

    /// Bytes holding `codes` codes.
    pub const fn code_bytes(self, codes: usize) -> usize {
        codes.div_ceil(self.codes_per_byte())
    }

    fn shift(self, index: usize) -> usize {
        (index % self.codes_per_byte()) * usize::from(self.bits())
    }

    /// OR `code` into slot `index` of a zero-initialized packed byte run.
    fn pack(self, bytes: &mut [u8], index: usize, code: u8) {
        bytes[index / self.codes_per_byte()] |= code << self.shift(index);
    }

    fn unpack(self, bytes: &[u8], index: usize) -> u8 {
        (bytes[index / self.codes_per_byte()] >> self.shift(index)) & self.max_code()
    }

    /// CPU reference quantization of one group's value against the group's STORED (f16) zero and
    /// scale: `clamp(round((x − zero) / scale), 0, 2^bits − 1)`, so the dequantized
    /// `zero + scale · code` is the nearest level of exactly what the reader sees.
    fn quantize(self, value: f32, zero: f16, scale: f16) -> u8 {
        ((value - zero.to_f32()) / scale.to_f32())
            .round()
            .clamp(0.0, self.levels()) as u8
    }

    /// The stored f16 `(zero, scale)` of a group spanning `[min, max]` (the GPU quantizer's
    /// `sc20676_affine`, bit for bit). The zero is `min` rounded to f16 and stepped down one f16
    /// ulp if it rounded above `min`, so every value is at or above it; the scale is
    /// `(max − zero) / (2^bits − 1)` floored at the smallest normal f16 (no subnormal scales, and a
    /// zero-range group is exact at code 0) and stepped up one ulp if rounding left `max` beyond
    /// the top level. Codes therefore never clamp, and the round-trip error of every value is at
    /// most half the stored scale, however large `|min|` is relative to the range.
    fn affine(self, min: f32, max: f32) -> (f16, f16) {
        let mut zero = f16::from_f32(min);
        if zero.to_f32() > min {
            let down = f16_next_down(zero);
            if down.is_finite() {
                zero = down;
            }
        }
        let range = max - zero.to_f32();
        let mut scale = f16::from_f32((range / self.levels()).max(f16::MIN_POSITIVE.to_f32()));
        if range / scale.to_f32() > self.levels() {
            let up = f16::from_bits(scale.to_bits() + 1);
            if up.is_finite() {
                scale = up;
            }
        }
        (zero, scale)
    }
}

/// The largest f16 below `value` (finite, non-NaN input).
fn f16_next_down(value: f16) -> f16 {
    let bits = value.to_bits();
    f16::from_bits(if bits & 0x7fff == 0 {
        0x8001
    } else if bits & 0x8000 == 0 {
        bits - 1
    } else {
        bits + 1
    })
}

/// MLX exposes array extents as signed `i32`s, while packed-cache storage uses `usize`.
/// Keep that conversion at the array boundary so a large host-side cache can never wrap into an
/// invalid MLX shape.
fn mlx_shape(shape: [usize; 4]) -> Result<[i32; 4]> {
    shape
        .map(|dimension| {
            i32::try_from(dimension)
                .map_err(|_| Error::Config("packed KV dimension exceeds MLX i32 range".into()))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .and_then(|dimensions| {
            dimensions
                .try_into()
                .map_err(|_| Error::Msg("internal packed KV MLX shape conversion failed".into()))
        })
}

/// Input tensors are owned by MLX and therefore arrive with signed extents. Reject malformed
/// extents before comparing them with cache-owned `usize` geometry.
pub(super) fn packed_input_shape(shape: &[i32], tensor: &str) -> Result<[usize; 4]> {
    let [batch, heads, tokens, width] = shape else {
        return Err(Error::Unsupported(format!(
            "{tensor} tensor must have rank four"
        )));
    };
    [*batch, *heads, *tokens, *width]
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                Error::Unsupported(format!("{tensor} tensor has a negative MLX dimension"))
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .and_then(|dimensions| {
            dimensions
                .try_into()
                .map_err(|_| Error::Msg("internal packed KV input shape conversion failed".into()))
        })
}

pub(crate) fn packed_metal_head_dimension_supported(head_dimension: usize) -> bool {
    matches!(head_dimension, 64 | 128 | 256)
}

pub(crate) fn packed_metal_cache_geometry_supported(
    head_dimension: usize,
    group_size: usize,
) -> bool {
    packed_metal_head_dimension_supported(head_dimension)
        && group_size == PACKED_METAL_QUANT_GROUP_SIZE
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationMetadata {
    pub identity: String,
    pub version: u32,
    pub group_size: usize,
    pub bits: u8,
    pub batch: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub logical_len: usize,
    pub capacity: usize,
    pub absolute_offset: usize,
    pub host_allocated_payload_bytes: usize,
    pub retained_device_packed_logical_bytes: usize,
    /// Total cache-attributable packed bytes: host vector capacity plus the allocated device arrays
    /// (block capacity and dense residuals). Backend allocator overhead and shared pools require
    /// process-level measurement.
    pub allocated_bytes: usize,
    pub key_grouping: &'static str,
    pub value_grouping: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseFallbackEvent {
    pub operation: String,
    pub reason: String,
    pub logical_len: usize,
    pub allocated_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackedDispatchTelemetry {
    /// Reader calls which passed cache preflight and began packed argument staging. This and the
    /// failure/JIT fields are physical-attempt evidence and remain monotonic across rollback.
    pub dispatch_attempts: u64,
    pub failed_dispatches: u64,
    pub compile_jit_attempts: u64,
    /// True once a reader output has evaluated successfully. Logical model-step rollback cannot
    /// un-warm the retained pipeline, so this state is deliberately not transactional.
    pub kernel_warmed: bool,
    /// Total wall time spent in physical reader attempts, including failed attempts.
    pub attempted_elapsed_ms: f64,
    /// Accepted reader calls classified by the physical pipeline state at attempt start. These
    /// counters are transactional at the decoder whole-step boundary.
    pub cold_dispatches: u64,
    pub steady_dispatches: u64,
    /// End-to-end first successful packed dispatch latency, including any host-mirror upload, lazy
    /// pipeline compilation, execution, and synchronization (a cold dispatch is always evaluated).
    /// This is deliberately not presented as compile-only time because MLX does not expose that
    /// boundary independently.
    pub cold_elapsed_ms: f64,
    /// Sum of the latency of successful dispatches after the first. On the decoder route a steady
    /// dispatch is left lazy (it joins the caller's single per-token evaluation), so this is the
    /// host-side dispatch time; GPU execution is attributed by the caller's end-to-end timing.
    pub steady_elapsed_ms: f64,
    /// Cumulative packed payload written into cache-owned device storage for physical attempts,
    /// including attempts that later fail: host upload for the CPU reference cache, on-device
    /// quantize-on-append (codes, metadata, and dense residual rows) for the decoder route.
    pub uploaded_packed_bytes: u64,
    /// Subset of `uploaded_packed_bytes` belonging to accepted calls in committed model steps.
    /// Whole-step rollback restores this counter while leaving physical write evidence intact.
    pub accepted_uploaded_packed_bytes: u64,
    /// Live-extent payload (codes, metadata, dense residual rows) of the cache-owned device arrays;
    /// their allocated block capacity is reported separately as physical bytes. Backend allocator
    /// overhead and shared pools remain process-level receipt measurements.
    pub retained_device_packed_logical_bytes: u64,
    /// Largest packed argument payload presented to one kernel dispatch. This is not allocator
    /// usage; the sealed process harness measures the actual transient high-water mark.
    pub peak_packed_argument_logical_bytes: u64,
    /// Conservative cache-attributable high-water mark: live extents of every layer plus any
    /// rollback-held residuals during a dispatch, and pre- and post-growth arrays coexisting
    /// during a block growth. Process-wide allocator receipts remain authoritative, but this value
    /// prevents retained bytes from being mislabeled as peak.
    pub peak_packed_transient_logical_bytes: u64,
}

/// A retained backend object which owns the compiled reader for one cache identity.
///
/// SC-20675 intentionally does not manufacture a Metal object from descriptive strings.  The
/// SC-20676 supplies its retained pipeline/argument state through this small object-safe boundary,
/// so a cache cannot outlive (or be rebound to) another cache's reader.
pub trait RetainedPackedKernel: fmt::Debug {
    fn cache_identity(&self) -> &str;
    fn backend(&self) -> &str;
    /// Heap/device bytes retained exclusively by this compiled object, if the backend can report
    /// them.  The storage accounting includes this value and never calls it payload bytes.
    fn retained_host_bytes_estimate(&self) -> usize;
    fn dispatch(&self, args: &PackedAttentionArgs<'_>) -> Result<Array>;
    /// The kernel [`Self::dispatch`] runs for `args` and why, recorded per accepted call so
    /// receipts name the path actually taken. Readers that cannot report one return `None`, and a
    /// receipt whose fused calls are not all attributed is refused.
    fn kernel_selection(&self, _args: &PackedAttentionArgs<'_>) -> Option<PackedKernelSelection> {
        None
    }
    /// Fused attention over an SC-20680 page pool, reading the pages in place through the page
    /// table. Readers without a paged layout refuse, and the paged cache then serves the call
    /// through its explicit dense gather fallback.
    fn dispatch_paged(&self, _args: &PagedPackedAttentionArgs<'_>) -> Result<Array> {
        Err(Error::Unsupported(
            "this packed reader has no paged-layout kernels".into(),
        ))
    }
    /// The kernel [`Self::dispatch_paged`] runs for `args` and why (`None` when unreported).
    fn paged_kernel_selection(
        &self,
        _args: &PagedPackedAttentionArgs<'_>,
    ) -> Option<PackedKernelSelection> {
        None
    }
    /// Code width this reader was built to read. A cache of another width refuses to bind it.
    /// Readers that predate the 4-bit representation read the qualified 2-bit layout.
    fn code_bits(&self) -> PackedCodeBits {
        PackedCodeBits::Two
    }
}

/// Lifetime-owned, type-erased compiled-kernel slot.  `Arc` keeps the real backend object alive
/// across cache clones/snapshots without relying on mutable device/queue/context labels.
#[derive(Clone)]
pub struct CompiledKernelHandle {
    inner: Arc<dyn RetainedPackedKernel>,
    warmed: Arc<AtomicBool>,
    /// The paged-layout kernels are separate pipelines compiled on their own first dispatch, so
    /// their warm state is tracked apart from the contiguous kernels'.
    paged_warmed: Arc<AtomicBool>,
}

impl CompiledKernelHandle {
    pub fn new(inner: Arc<dyn RetainedPackedKernel>) -> Self {
        Self {
            inner,
            warmed: Arc::new(AtomicBool::new(false)),
            paged_warmed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cache_identity(&self) -> &str {
        self.inner.cache_identity()
    }

    pub fn backend(&self) -> &str {
        self.inner.backend()
    }

    pub fn retained_host_bytes_estimate(&self) -> usize {
        self.inner.retained_host_bytes_estimate()
    }

    /// Code width the retained reader reads (see [`RetainedPackedKernel::code_bits`]).
    pub fn code_bits(&self) -> PackedCodeBits {
        self.inner.code_bits()
    }

    pub(crate) fn is_warmed(&self) -> bool {
        self.warmed.load(Ordering::Acquire)
    }

    pub(crate) fn mark_warmed(&self) {
        self.warmed.store(true, Ordering::Release);
    }

    /// Whether the reader's paged kernels have produced an output (see [`Self::dispatch_paged`]).
    pub(crate) fn is_paged_warmed(&self) -> bool {
        self.paged_warmed.load(Ordering::Acquire)
    }

    pub(crate) fn mark_paged_warmed(&self) {
        self.paged_warmed.store(true, Ordering::Release);
    }

    /// The retained reader's paged dispatch (see [`RetainedPackedKernel::dispatch_paged`]).
    pub(crate) fn dispatch_paged(&self, args: &PagedPackedAttentionArgs<'_>) -> Result<Array> {
        self.inner.dispatch_paged(args)
    }

    pub(crate) fn paged_kernel_selection(
        &self,
        args: &PagedPackedAttentionArgs<'_>,
    ) -> Option<PackedKernelSelection> {
        self.inner.paged_kernel_selection(args)
    }
}

impl fmt::Debug for CompiledKernelHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledKernelHandle")
            .field("cache_identity", &self.cache_identity())
            .field("backend", &self.backend())
            .field("code_bits", &self.code_bits().bits())
            .field(
                "retained_host_bytes_estimate",
                &self.retained_host_bytes_estimate(),
            )
            .finish_non_exhaustive()
    }
}

/// Minimal opaque carrier for a backend-owned compiled object. It can retain an arbitrary concrete
/// handle, but makes no claim that the supplied handle can dispatch packed attention. The host
/// estimate deliberately excludes globally shared MLX/Metal allocations.
pub struct OpaqueCompiledKernel {
    cache_identity: String,
    backend: String,
    retained_host_bytes_estimate: usize,
    _keep_alive: Arc<dyn Any + Send + Sync>,
}

impl fmt::Debug for OpaqueCompiledKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueCompiledKernel")
            .field("cache_identity", &self.cache_identity)
            .field("backend", &self.backend)
            .field(
                "retained_host_bytes_estimate",
                &self.retained_host_bytes_estimate,
            )
            .finish_non_exhaustive()
    }
}

impl OpaqueCompiledKernel {
    pub fn new(
        cache_identity: impl Into<String>,
        backend: impl Into<String>,
        retained_host_bytes_estimate: usize,
        keep_alive: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            cache_identity: cache_identity.into(),
            backend: backend.into(),
            retained_host_bytes_estimate,
            _keep_alive: keep_alive,
        }
    }
}

impl RetainedPackedKernel for OpaqueCompiledKernel {
    fn cache_identity(&self) -> &str {
        &self.cache_identity
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn retained_host_bytes_estimate(&self) -> usize {
        self.retained_host_bytes_estimate
    }

    fn dispatch(&self, _args: &PackedAttentionArgs<'_>) -> Result<Array> {
        Err(Error::Unsupported(
            "opaque packed reader cannot dispatch".into(),
        ))
    }
}

/// Explicit, opt-in decoder construction request.  This is deliberately an override rather than
/// an ambient environment switch: normal decoders construct the current contiguous cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedCacheRequest {
    pub enabled: bool,
    pub backend: String,
    pub identity: String,
    pub layers: usize,
    pub batch: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub group_size: usize,
    /// Code width of the staged packed cache; the bound reader must read the same width.
    pub bits: PackedCodeBits,
    pub query_length: usize,
    pub has_mask: bool,
}

impl PackedCacheRequest {
    pub fn disabled(layers: usize) -> Self {
        Self {
            enabled: false,
            backend: "disabled".into(),
            identity: "sc-20675-default-dense".into(),
            layers,
            batch: 0,
            kv_heads: 0,
            head_dimension: 0,
            group_size: 0,
            bits: PackedCodeBits::Two,
            query_length: 0,
            has_mask: false,
        }
    }
}

/// Result of choosing a decoder cache before any K/V update.  The route is retained separately so
/// callers can emit the fallback diagnostic without relying on a downcast.
pub struct DecoderCacheSelection {
    route: CacheRoute,
    cache: Box<dyn KvCache>,
}

impl DecoderCacheSelection {
    pub fn route(&self) -> &CacheRoute {
        &self.route
    }

    pub fn into_cache(self) -> Box<dyn KvCache> {
        self.cache
    }
}

/// Accepted-dispatch counters restored by a whole-step rollback.
#[derive(Clone, Debug, PartialEq)]
struct AcceptedDispatchSnapshot {
    direct_dispatches: usize,
    kernel_paths: Vec<(PackedKernelSelection, u64)>,
    cold_dispatches: u64,
    steady_dispatches: u64,
    cold_elapsed_ms: f64,
    steady_elapsed_ms: f64,
    accepted_uploaded_packed_bytes: u64,
}

#[derive(Clone, Debug)]
struct PendingPackedStep {
    original_len: usize,
    next_layer: usize,
    step: usize,
    /// Per-layer rollback points (counts and residual arrays only; history is append-only).
    marks: Vec<Option<DeviceLayerMark>>,
    accepted_dispatch: AcceptedDispatchSnapshot,
    /// Bytes this step wrote into device storage; accepted only when the step commits.
    written_bytes: u64,
    /// A layer of this step attended its fresh K/V through dense SDPA (an empty cache's first
    /// multi-row step), so no output reads that layer's store: the commit evaluates the store.
    store_unread: bool,
}

/// Decoder-facing owner for the device-resident packed store and the retained fused reader. A model
/// step is a whole-step transaction across layers: each layer appends lazily into the device store
/// and dispatches the reader lazily (only a cold dispatch is evaluated, to surface a JIT fault
/// inside the transaction); a failure rolls every layer back to its marked extents. A step whose
/// attention ran on dense SDPA has no output that reads the store, so its commit evaluates the
/// store: left lazy, it would keep the step's dense K/V alive until the first packed read. Unsupported
/// semantics transition the exact evaluated history to dense before the caller's ordinary update.
#[derive(Debug)]
pub struct DenseFallbackPackedDecoderCache {
    dense: ContiguousKvCache,
    staged: PackedGroupAffineKvCache,
    pending_step: Option<PendingPackedStep>,
    packed_layer_dtypes: Vec<Option<(mlx_rs::Dtype, mlx_rs::Dtype)>>,
    dense_active: bool,
    reason: String,
    /// The armed speculation window (see [`KvCache::begin_speculation`]); consumed by the next
    /// truncate.
    speculation: Option<ContiguousSpeculation>,
    transition_admission: Option<DenseTransitionAdmission>,
}

/// Exact speculative rollback on the contiguous compressed cache (sc-20681): armed at `base`, it
/// keeps each layer's unquantized rows from its group-aligned quantized extent at arming to the
/// logical end, so a truncate back to `len >= base` rebuilds the residual from those rows instead
/// of re-staging a cut group from its codes. Every group completed below `len` keeps its codes:
/// they were quantized from exactly these rows, as appending only the kept rows would have.
#[derive(Clone, Debug)]
struct ContiguousSpeculation {
    base: usize,
    layers: Vec<SpeculationRows>,
}

/// One layer's speculation window: the unquantized rows `start..start + rows`.
#[derive(Clone, Debug, Default)]
struct SpeculationRows {
    start: usize,
    rows: usize,
    /// `[B, H, rows, D]` keys and values; `None` while `rows == 0`.
    kv: Option<(Array, Array)>,
}

impl SpeculationRows {
    fn extend(&mut self, keys: &Array, values: &Array) -> Result<()> {
        let step = usize::try_from(keys.shape()[2])
            .map_err(|_| Error::Config("negative speculation step".into()))?;
        self.kv = Some(match self.kv.take() {
            None => (keys.clone(), values.clone()),
            Some((k, v)) => (
                concatenate_axis(&[&k, keys], 2)?,
                concatenate_axis(&[&v, values], 2)?,
            ),
        });
        self.rows += step;
        Ok(())
    }

    /// Drop the rows past logical position `len` (a rolled-back step's).
    fn cut_to(&mut self, len: usize) -> Result<()> {
        let rows = len.saturating_sub(self.start).min(self.rows);
        if rows == self.rows {
            return Ok(());
        }
        self.kv = match self.kv.take() {
            Some((k, v)) if rows > 0 => Some((live_rows(&k, rows)?, live_rows(&v, rows)?)),
            _ => None,
        };
        self.rows = rows;
        Ok(())
    }

    fn bytes(&self) -> u64 {
        self.kv
            .as_ref()
            .map_or(0, |(k, v)| array_bytes(k).saturating_add(array_bytes(v)))
    }
}

/// Admission of an observable dense transition (sc-20682). A product generation admitted at the
/// compressed price has not paid for a dense reconstruction of its history, so before the
/// transition allocates anything the cache calls this with the bytes it is about to allocate
/// ([`DenseFallbackPackedDecoderCache::dense_reconstruction_bytes`]); an `Err` (the product's
/// typed resource refusal) aborts the transition and fails the step instead of oversubscribing
/// memory. Without one (the campaign arms, which run under their own supervisor) the transition
/// is unconditional.
pub struct DenseTransitionAdmission(Box<dyn Fn(u64) -> Result<()>>);

impl DenseTransitionAdmission {
    pub fn new(admit: impl Fn(u64) -> Result<()> + 'static) -> Self {
        Self(Box::new(admit))
    }

    /// Admit a transition about to allocate `bytes`.
    pub fn admit(&self, bytes: u64) -> Result<()> {
        (self.0)(bytes)
    }
}

impl fmt::Debug for DenseTransitionAdmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DenseTransitionAdmission")
    }
}

impl DenseFallbackPackedDecoderCache {
    pub fn staged_representation(&self) -> RepresentationMetadata {
        self.staged.representation()
    }

    pub fn fallback_events(&self) -> &[DenseFallbackEvent] {
        self.staged.fallback_events()
    }

    /// Pending device snapshots are rollback-only state. They contribute to the telemetry's
    /// logical transient high-water, never to cache-resident representation evidence.
    fn pending_device_snapshot_overhead(&self) -> u64 {
        self.pending_step.as_ref().map_or(0, |pending| {
            PackedGroupAffineKvCache::device_mark_overhead(&pending.marks, pending.next_layer)
        })
    }

    /// Immutable receipt evidence at the same public object used by `CausalLm`.
    pub fn model_evidence(&self) -> PackedCacheEvidence {
        let telemetry = self.staged.dispatch_telemetry();
        let representation = self.staged.representation();
        let (code_bytes, metadata_bytes) = self.staged.retained_device_component_bytes();
        // Residual arrays a pending step keeps for rollback are counted with the metadata.
        let pending_metadata_bytes = self.pending_device_snapshot_overhead();
        // A failed host-size conversion becomes zero, which the sealed SC-20676 reducer rejects
        // rather than accepting an inexact physical-representation claim.
        let retained_device_code_bytes = u64::try_from(code_bytes).unwrap_or_default();
        let retained_device_metadata_bytes = u64::try_from(metadata_bytes)
            .unwrap_or_default()
            .saturating_add(pending_metadata_bytes);
        PackedCacheEvidence {
            representation_identity: representation.identity,
            representation_version: representation.version,
            bits: representation.bits,
            quantization_group_size: representation.group_size,
            accepted_direct_calls: self.staged.direct_dispatches(),
            kernel_paths: crate::primitives::kv_cache::PackedKernelPathEvidence::sorted(
                self.staged.kernel_paths(),
            ),
            full_cache_dequantizations: self.staged.full_cache_dequantizations(),
            dispatch_attempts: telemetry.dispatch_attempts,
            failed_dispatches: telemetry.failed_dispatches,
            compile_jit_attempts: telemetry.compile_jit_attempts,
            kernel_warmed: telemetry.kernel_warmed,
            attempted_elapsed_ms: telemetry.attempted_elapsed_ms,
            cold_dispatches: telemetry.cold_dispatches,
            steady_dispatches: telemetry.steady_dispatches,
            cold_elapsed_ms: telemetry.cold_elapsed_ms,
            steady_elapsed_ms: telemetry.steady_elapsed_ms,
            uploaded_packed_bytes: telemetry.uploaded_packed_bytes,
            accepted_uploaded_packed_bytes: telemetry.accepted_uploaded_packed_bytes,
            retained_device_code_bytes,
            retained_device_metadata_bytes,
            retained_device_packed_logical_bytes: retained_device_code_bytes
                .saturating_add(retained_device_metadata_bytes),
            peak_packed_argument_logical_bytes: telemetry.peak_packed_argument_logical_bytes,
            peak_packed_transient_logical_bytes: telemetry.peak_packed_transient_logical_bytes,
            dense_active: self.dense_active,
            fallback_reasons: self
                .staged
                .fallback_events()
                .iter()
                .map(|event| (event.operation.clone(), event.reason.clone()))
                .collect(),
            dense_gather_calls: Vec::new(),
        }
    }

    /// Attach only a reader whose identity matches the staged representation.
    pub fn bind_compiled_handle(&mut self, handle: CompiledKernelHandle) -> Result<()> {
        self.staged.bind_compiled_handle(handle)
    }

    /// Admit every later dense transition through `admission` before it allocates.
    pub fn set_dense_transition_admission(&mut self, admission: DenseTransitionAdmission) {
        self.transition_admission = Some(admission);
    }

    /// Bytes the dense transition (`transition_to_dense`) allocates for the resident history:
    /// every layer's dense K/V at the dense cache's block-rounded capacity, plus the largest
    /// layer's reconstruction transient (its Float32 dequantization beside the cast K/V it copies
    /// into the dense cache).
    pub fn dense_reconstruction_bytes(&self) -> u64 {
        let rows = self.staged.batch.saturating_mul(self.staged.kv_heads);
        let block = usize::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).unwrap_or(256);
        let mut resident = 0_u64;
        let mut transient = 0_u64;
        for layer in 0..self.staged.layers() {
            let tokens = self.staged.layer_tokens(layer);
            if tokens == 0 {
                continue;
            }
            let (key_dtype, value_dtype) = self
                .packed_layer_dtypes
                .get(layer)
                .and_then(|dtypes| *dtypes)
                .unwrap_or((Dtype::Float32, Dtype::Float32));
            let element = dtype_bytes(key_dtype).saturating_add(dtype_bytes(value_dtype));
            let elements = rows.saturating_mul(self.staged.head_dimension);
            let capacity = tokens.div_ceil(block).saturating_mul(block);
            resident = resident.saturating_add(
                u64::try_from(capacity.saturating_mul(elements).saturating_mul(element))
                    .unwrap_or(u64::MAX),
            );
            let layer_transient = tokens
                .saturating_mul(elements)
                .saturating_mul((2 * dtype_bytes(Dtype::Float32)).saturating_add(element));
            transient = transient.max(u64::try_from(layer_transient).unwrap_or(u64::MAX));
        }
        resident.saturating_add(transient)
    }

    fn restore_pending(&mut self) -> Result<bool> {
        let Some(pending) = self.pending_step.take() else {
            return Ok(false);
        };
        self.staged
            .restore_device_marks(pending.marks, pending.original_len);
        self.staged
            .restore_accepted_dispatch(pending.accepted_dispatch);
        if pending.original_len == 0 {
            self.packed_layer_dtypes.fill(None);
        }
        if let Some(speculation) = self.speculation.as_mut() {
            for window in &mut speculation.layers {
                window.cut_to(pending.original_len)?;
            }
        }
        Ok(true)
    }

    fn rollback_pending(&mut self, operation: &str, reason: impl Into<String>) -> Result<()> {
        if !self.restore_pending()? {
            return Ok(());
        }
        self.staged.dense_read_fallback(operation, reason);
        Ok(())
    }

    /// Materialize the exact packed values which have already participated in attention into a
    /// fresh dense cache. A partially completed model step is preserved per layer: layers already
    /// dispatched contain `original_len + step`, while later layers still contain `original_len`.
    /// The fresh cache is published only after every reconstructed MLX array evaluates.
    fn transition_to_dense(&mut self, operation: &str, reason: impl Into<String>) -> Result<()> {
        if self.dense_active {
            return Ok(());
        }
        // Admitted even with no resident history to rebuild (a first-step fault rolled back to
        // an empty cache): the rest of the generation still runs dense, which the request's
        // compressed admission did not price (sc-20682).
        if let Some(admission) = &self.transition_admission {
            admission.admit(self.dense_reconstruction_bytes())?;
        }
        let mut dense = ContiguousKvCache::new(self.staged.layers());
        // The replacement keeps recording exactly when its predecessor did (campaign only).
        if self.dense.records_events() {
            dense.record_events();
        }
        let mut reconstructed = false;
        for layer_index in 0..self.staged.layers() {
            if self.staged.layer_tokens(layer_index) == 0 {
                continue;
            }
            reconstructed = true;
            let (key_dtype, value_dtype) = self
                .packed_layer_dtypes
                .get(layer_index)
                .and_then(|value| *value)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "packed layer {layer_index} has resident values without dtype evidence"
                    ))
                })?;
            // Rebuild exactly the values the reader consumed: dequantized groups/rows plus the
            // dense residual, reconstructed on the device for device-resident layers.
            let (keys, values) =
                self.staged
                    .dense_layer_arrays(layer_index, key_dtype, value_dtype)?;
            keys.eval()?;
            values.eval()?;
            dense.update(layer_index, &keys, &values)?;
        }

        let reason = reason.into();
        if reconstructed {
            self.staged.record_dense_dequantization();
        }
        self.staged.dense_read_fallback(operation, reason.clone());
        self.dense = dense;
        self.pending_step = None;
        self.speculation = None;
        self.staged.handle = None;
        self.staged.clear();
        self.dense_active = true;
        self.reason = reason;
        Ok(())
    }

    fn decline_before_packed_mutation(
        &mut self,
        operation: &str,
        reason: impl Into<String>,
    ) -> Result<Option<Array>> {
        let reason = reason.into();
        self.transition_to_dense(operation, reason)?;
        Ok(None)
    }

    /// Install-time decoder seam for the retained reader.  The dense cache remains the
    /// compatibility implementation until this explicit opt-in succeeds; thereafter callers can
    /// dispatch attention directly from the staged packed cache without asking `update` for a
    /// reconstructed full K/V tensor.
    pub fn dispatch_packed(
        &mut self,
        layer: usize,
        query: &mlx_rs::Array,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> Result<mlx_rs::Array> {
        self.staged.dispatch_packed(layer, query, mask)
    }
}

impl KvCache for DenseFallbackPackedDecoderCache {
    fn preflight_packed(&self, query_length: usize, mask: bool) -> CacheRoute {
        let reason = if self.dense_active {
            Some(self.reason.clone())
        } else if self.staged.compiled_handle().is_none() {
            Some("no retained compiled packed reader".into())
        } else if query_length == 0 {
            Some("packed attention requires a non-empty query".into())
        } else if mask {
            Some("additive attention masks require dense fallback".into())
        } else {
            None
        };
        reason.map_or(CacheRoute::ExperimentalPacked, |reason| {
            CacheRoute::DenseFallback { reason }
        })
    }

    fn update(
        &mut self,
        layer: usize,
        keys: &mlx_rs::Array,
        values: &mlx_rs::Array,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        if !self.dense_active {
            self.transition_to_dense("update", "decoder layer requires dense attention")?;
        }
        self.dense.update(layer, keys, values)
    }

    fn offset(&self) -> i32 {
        if self.dense_active {
            self.dense.offset()
        } else {
            i32::try_from(self.staged.logical_len())
                .expect("packed cache length is bounded to the MLX i32 range")
        }
    }

    fn batch_size(&self) -> i32 {
        if self.dense_active {
            self.dense.batch_size()
        } else {
            i32::try_from(self.staged.batch_size())
                .expect("packed cache batch is validated for the MLX i32 range")
        }
    }

    fn try_packed_attention(
        &mut self,
        layer: usize,
        query: &mlx_rs::Array,
        keys: &mlx_rs::Array,
        values: &mlx_rs::Array,
        mask: crate::primitives::kv_cache::PackedAttentionMask,
        scale: f32,
        retained_for_sharing: bool,
    ) -> Result<Option<mlx_rs::Array>> {
        let outcome = (|| -> Result<Option<mlx_rs::Array>> {
            if self.dense_active {
                return Ok(None);
            }
            if self.staged.logical_len() != 0 && self.staged.compiled_handle().is_none() {
                return self.decline_before_packed_mutation(
                    "missing packed reader",
                    "packed reader became unavailable",
                );
            }
            if self.staged.compiled_handle().is_none() {
                return Ok(None);
            }
            if retained_for_sharing {
                return self.decline_before_packed_mutation(
                    "shared-kv attention",
                    "the retained packed reader cannot publish K/V for a sharing tail",
                );
            }

            // Validate the complete accepted surface before converting K/V, opening a transaction,
            // appending host storage, or synchronizing any MLX device array.
            let q_shape = query.shape();
            let k_shape = keys.shape();
            let v_shape = values.shape();
            let supported_dtype = |dtype| {
                matches!(
                    dtype,
                    mlx_rs::Dtype::Float16 | mlx_rs::Dtype::Bfloat16 | mlx_rs::Dtype::Float32
                )
            };
            let shape_reason = if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
                Some("query, key, and value tensors must all have rank four")
            } else if layer >= self.staged.layers() {
                Some("layer index is out of range")
            } else if q_shape
                .iter()
                .chain(k_shape.iter())
                .chain(v_shape.iter())
                .any(|&dimension| dimension < 0)
            {
                Some("query, key, and value tensors must not have negative MLX dimensions")
            } else {
                None
            };
            if let Some(reason) = shape_reason {
                return self.decline_before_packed_mutation("geometry/dtype", reason);
            }
            let q_shape = packed_input_shape(q_shape, "query")?;
            let k_shape = packed_input_shape(k_shape, "key")?;
            let v_shape = packed_input_shape(v_shape, "value")?;
            let shape_reason = if q_shape[0] != self.staged.batch
                || k_shape[0] != self.staged.batch
                || v_shape[0] != self.staged.batch
            {
                Some("query, key, and value batch dimensions must match the packed cache")
            } else if k_shape[1] != self.staged.kv_heads
                || v_shape[1] != self.staged.kv_heads
                || q_shape[1] == 0
                || q_shape[1] % self.staged.kv_heads != 0
            {
                Some("query heads must map evenly onto the packed K/V heads")
            } else if !packed_metal_head_dimension_supported(self.staged.head_dimension)
                || q_shape[3] != self.staged.head_dimension
                || k_shape[3] != self.staged.head_dimension
                || v_shape[3] != self.staged.head_dimension
            {
                Some("packed Metal head dimension must be 64, 128, or 256 and match all tensors")
            } else if q_shape[2] == 0 || k_shape[2] == 0 || v_shape[2] != k_shape[2] {
                Some("packed attention requires a non-empty, equal K/V token step")
            } else if q_shape[2] != k_shape[2] {
                Some("packed causal queries must cover exactly the newly appended K/V step")
            } else if !supported_dtype(query.dtype())
                || !supported_dtype(keys.dtype())
                || !supported_dtype(values.dtype())
            {
                Some(
                    "packed attention accepts only f16, bf16, or f32 query, key, and value tensors",
                )
            } else if self.packed_layer_dtypes[layer].is_some_and(|(key_dtype, value_dtype)| {
                key_dtype != keys.dtype() || value_dtype != values.dtype()
            }) {
                Some("packed K/V dtype must remain stable for each resident layer")
            } else {
                None
            };
            if let Some(reason) = shape_reason {
                return self.decline_before_packed_mutation("geometry/dtype", reason);
            }

            let step = k_shape[2];
            let (original_len, expected_layer, expected_step) = self
                .pending_step
                .as_ref()
                .map_or((self.staged.logical_len(), 0, step), |pending| {
                    (pending.original_len, pending.next_layer, pending.step)
                });
            if (self.pending_step.is_none() && layer != 0)
                || expected_layer != layer
                || expected_step != step
            {
                return Err(Error::Config(
                    "packed step layer order or token length mismatch".into(),
                ));
            }
            let expected_cache_len = original_len
                .checked_add(step)
                .ok_or_else(|| Error::Config("packed cache length overflow".into()))?;
            if q_shape[2] > expected_cache_len {
                return self.decline_before_packed_mutation(
                    "query/cache length",
                    "query length exceeds the post-append packed cache length",
                );
            }

            let expected_scale = (q_shape[3] as f32).powf(-0.5);
            if scale.to_bits() != expected_scale.to_bits() {
                return self.decline_before_packed_mutation(
                    "attention scale",
                    "attention scale does not match inverse square-root head dimension",
                );
            }
            let packed_mask = match mask {
                crate::primitives::kv_cache::PackedAttentionMask::None => {
                    crate::primitives::packed_metal::PackedMask::None
                }
                crate::primitives::kv_cache::PackedAttentionMask::Causal => {
                    crate::primitives::packed_metal::PackedMask::Causal
                }
                crate::primitives::kv_cache::PackedAttentionMask::SlidingWindow(window)
                    if window > 0 && i32::try_from(window).is_ok() =>
                {
                    crate::primitives::packed_metal::PackedMask::SlidingWindow(window)
                }
                crate::primitives::kv_cache::PackedAttentionMask::SlidingWindow(_) => {
                    return self.decline_before_packed_mutation(
                        "sliding-window mask",
                        "sliding window must be in 1..=i32::MAX",
                    );
                }
                crate::primitives::kv_cache::PackedAttentionMask::Additive => {
                    return self.decline_before_packed_mutation(
                        "additive mask",
                        "additive masks require dense attention",
                    );
                }
            };

            let key_dtype = keys.dtype();
            let value_dtype = values.dtype();
            // An empty cache's first multi-row step has no history: its queries attend only to the
            // step's own fresh K/V, which dense SDPA does exactly (the same call the dense decoder
            // makes) without an O(S_q·S_kv) vector-kernel pass. Nothing is reconstructed; the fresh
            // K/V are still appended packed. Steps within the fused vector-kernel row limit use the
            // packed reader, whose cost there equals dense.
            let first_step_dense_mask = (original_len == 0
                && step > crate::primitives::attention::SDPA_MAX_FUSED_QLEN as usize)
                .then_some(packed_mask)
                .and_then(|mask| match mask {
                    crate::primitives::packed_metal::PackedMask::None => {
                        Some(crate::primitives::attention::AttnMask::None)
                    }
                    crate::primitives::packed_metal::PackedMask::Causal => {
                        Some(crate::primitives::attention::AttnMask::Causal)
                    }
                    crate::primitives::packed_metal::PackedMask::SlidingWindow(window)
                        if window >= step =>
                    {
                        Some(crate::primitives::attention::AttnMask::Causal)
                    }
                    _ => None,
                });

            // Mutate the device store in place and retain only per-layer counts and residual
            // arrays for rollback: history past the marked counts is append-only.
            if layer == 0 {
                self.pending_step = Some(PendingPackedStep {
                    original_len,
                    next_layer: 0,
                    step,
                    marks: self.staged.device_marks(),
                    accepted_dispatch: self.staged.accepted_dispatch_snapshot(),
                    written_bytes: 0,
                    store_unread: false,
                });
            }
            let (written, replaced) = self.staged.append_device_replacing(layer, keys, values)?;
            if let Some(window) = self
                .speculation
                .as_mut()
                .and_then(|speculation| speculation.layers.get_mut(layer))
            {
                window.extend(keys, values)?;
            }
            if let Some(pending) = self.pending_step.as_mut() {
                pending.written_bytes = pending.written_bytes.saturating_add(written);
                if let Some(Some(mark)) = pending.marks.get_mut(layer) {
                    mark.residuals = mark.residuals.take().or(replaced);
                }
            }
            let reader_warmed = self
                .staged
                .compiled_handle()
                .is_some_and(CompiledKernelHandle::is_warmed);
            let output = if let Some(dense_mask) = first_step_dense_mask {
                if let Some(pending) = self.pending_step.as_mut() {
                    pending.store_unread = true;
                }
                let output =
                    crate::primitives::attention::sdpa(query, keys, values, scale, dense_mask)?;
                // Evaluate this layer's appended store and its attention output before the next
                // layer is built (sc-20682). Left lazy until the last layer, the whole step is one
                // graph and every layer's dense prompt K/V stays resident beside the packed store:
                // measured 1.88x the request's admission estimate on a KV-dominant decoder, above
                // the same request run dense. Evaluated per layer, only the current layer's dense
                // K/V coexist with the packed history. Inside the transaction, so a quantizer
                // fault still rolls the step back.
                self.staged.evaluate_device_layer(layer, &output)?;
                output
            } else {
                let snapshot_overhead = self.pending_device_snapshot_overhead();
                match self.staged.dispatch_layer(
                    layer,
                    query,
                    packed_mask,
                    snapshot_overhead,
                    false,
                ) {
                    Ok(output) => output,
                    Err(Error::Canceled) => return Err(Error::Canceled),
                    Err(error) if layer == 0 && original_len == 0 => {
                        // Before any packed output is published, a device/JIT fault can still
                        // select the ordinary dense path result-equivalently for this whole step.
                        self.rollback_pending("dispatch-fault", error.to_string())?;
                        self.staged.handle = None;
                        self.reason = format!("packed dispatch unavailable: {error}");
                        return Ok(None);
                    }
                    Err(error) if layer == 0 && !reader_warmed => {
                        // The reader has never produced an output (its cold compile failed) but
                        // history is resident: roll the step back and hand the exact packed history
                        // to the dense cache through the observable reconstruction transition.
                        let reason = format!("packed dispatch unavailable: {error}");
                        self.rollback_pending("dispatch-fault", error.to_string())?;
                        self.transition_to_dense("dispatch-fault", reason)?;
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                }
            };
            self.packed_layer_dtypes[layer] = Some((key_dtype, value_dtype));
            if layer + 1 == self.staged.layers() {
                if self.staged.logical_len() != expected_cache_len {
                    return Err(Error::Msg(
                        "packed whole-step transaction did not commit".into(),
                    ));
                }
                // The dense-SDPA output does not depend on the store, so evaluating the caller's
                // logits leaves the store a lazy graph over this step's dense K/V: the dense K/V
                // stay resident and the packed bytes the cache reports are not, until the first
                // packed read (SC-20671 4-bit prefill attribution floor; the first decode step's
                // peak then held both). Evaluate inside the transaction, so a quantizer fault
                // rolls the step back instead of surfacing at a later unrelated evaluation.
                if self
                    .pending_step
                    .as_ref()
                    .is_some_and(|pending| pending.store_unread)
                {
                    self.staged.evaluate_device_store()?;
                }
                if let Some(pending) = self.pending_step.take() {
                    self.staged.telemetry.accepted_uploaded_packed_bytes = self
                        .staged
                        .telemetry
                        .accepted_uploaded_packed_bytes
                        .saturating_add(pending.written_bytes);
                }
            } else {
                self.pending_step
                    .as_mut()
                    .expect("installed above")
                    .next_layer += 1;
            }
            Ok(Some(output))
        })();

        match outcome {
            Err(Error::Canceled) if self.pending_step.is_some() => {
                self.restore_pending()?;
                Err(Error::Canceled)
            }
            Err(error) if self.pending_step.is_some() => {
                self.rollback_pending("packed-transaction", error.to_string())?;
                Err(error)
            }
            other => other,
        }
    }

    /// Import a reused dense prefix (e.g. a prefix-cache hit) by quantize-on-append into the
    /// device-resident packed representation. This is an ordinary append, not a reconstruction:
    /// no packed history exists yet. Declines (`Ok(false)`, nothing mutated) unless the packed route
    /// is live and empty and every layer's K/V matches the cache geometry; the caller then keeps
    /// its dense seed.
    fn import_prefix(&mut self, layers: &[(mlx_rs::Array, mlx_rs::Array)]) -> Result<bool> {
        if self.dense_active
            || self.pending_step.is_some()
            || self.staged.compiled_handle().is_none()
            || self.staged.logical_len() != 0
            || layers.len() != self.staged.layers()
        {
            return Ok(false);
        }
        let float = |dtype| {
            matches!(
                dtype,
                mlx_rs::Dtype::Float16 | mlx_rs::Dtype::Bfloat16 | mlx_rs::Dtype::Float32
            )
        };
        let expected = |array: &mlx_rs::Array, step: Option<usize>| {
            packed_input_shape(array.shape(), "prefix").is_ok_and(|shape| {
                shape[0] == self.staged.batch
                    && shape[1] == self.staged.kv_heads
                    && shape[2] != 0
                    && step.is_none_or(|step| shape[2] == step)
                    && shape[3] == self.staged.head_dimension
            }) && float(array.dtype())
        };
        let Some(step) = layers
            .first()
            .and_then(|(keys, _)| usize::try_from(keys.shape().get(2).copied()?).ok())
        else {
            return Ok(false);
        };
        if !layers
            .iter()
            .all(|(keys, values)| expected(keys, Some(step)) && expected(values, Some(step)))
        {
            return Ok(false);
        }
        let mut written = 0u64;
        for (layer, (keys, values)) in layers.iter().enumerate() {
            match self.staged.append_device(layer, keys, values) {
                Ok(bytes) => {
                    written = written.saturating_add(bytes);
                    if let Some(window) = self
                        .speculation
                        .as_mut()
                        .and_then(|speculation| speculation.layers.get_mut(layer))
                    {
                        window.extend(keys, values)?;
                    }
                }
                Err(error) => {
                    self.staged.clear();
                    self.packed_layer_dtypes.fill(None);
                    return Err(error);
                }
            }
            self.packed_layer_dtypes[layer] = Some((keys.dtype(), values.dtype()));
        }
        if self.staged.logical_len() != step {
            self.staged.clear();
            self.packed_layer_dtypes.fill(None);
            return Err(Error::Msg("packed prefix import did not commit".into()));
        }
        self.staged.telemetry.accepted_uploaded_packed_bytes = self
            .staged
            .telemetry
            .accepted_uploaded_packed_bytes
            .saturating_add(written);
        Ok(true)
    }

    fn num_layers(&self) -> usize {
        self.dense.num_layers()
    }

    fn prepare_dense_fallback(&mut self, operation: &str, reason: &str) -> Result<()> {
        self.transition_to_dense(operation, reason)
    }

    /// Arm exact speculative rollback (sc-20681): keep every layer's unquantized rows from its
    /// group-aligned quantized extent on, so the next truncate back to at least this position
    /// restores the residual from them (see `ContiguousSpeculation`). The history stays
    /// compressed. Only a host-staged (CPU reference) store, which keeps no device residual to
    /// window, moves to the recorded dense fallback.
    fn begin_speculation(&mut self) -> Result<()> {
        if self.dense_active {
            return self.dense.begin_speculation();
        }
        if self.pending_step.is_some() {
            return Err(Error::Config(
                "packed speculation armed inside an open step".into(),
            ));
        }
        match self.staged.speculation_rows()? {
            Some(layers) => {
                self.speculation = Some(ContiguousSpeculation {
                    base: self.staged.logical_len(),
                    layers,
                });
                Ok(())
            }
            None => self.transition_to_dense(
                "speculation",
                "a host-staged packed store keeps no unquantized residual for an exact \
                 speculative rollback",
            ),
        }
    }

    fn packed_evidence(&self) -> Option<PackedCacheEvidence> {
        Some(self.model_evidence())
    }

    /// Campaign-only physical ownership of the live packed representation (SC-20676 compressed
    /// rows). Device bytes are the sizes of the MLX arrays actually retained by the cache (block
    /// capacity included; the bounded dense K/V residual is counted with the metadata) and host
    /// bytes are any host staging payload (none on the device-resident decoder route), so a receipt
    /// never substitutes bit accounting for storage. `None` while the dense fallback owns the
    /// history or nothing is resident.
    fn compressed_storage(&self) -> Result<Option<CompressedCacheStorage>> {
        if self.dense_active || self.staged.logical_len() == 0 {
            return Ok(None);
        }
        let (mut device_code_bytes, mut device_metadata_bytes) = (0_u64, 0_u64);
        for device in self.staged.device_layers.iter().flatten() {
            let (codes, metadata) = device.physical_component_bytes();
            device_code_bytes = device_code_bytes
                .checked_add(codes)
                .ok_or_else(|| Error::Msg("packed device code bytes overflow".into()))?;
            device_metadata_bytes = device_metadata_bytes
                .checked_add(metadata)
                .ok_or_else(|| Error::Msg("packed device metadata bytes overflow".into()))?;
        }
        // An armed speculation window's unquantized rows are resident too.
        device_metadata_bytes = device_metadata_bytes.saturating_add(
            self.speculation
                .iter()
                .flat_map(|speculation| &speculation.layers)
                .map(SpeculationRows::bytes)
                .sum::<u64>(),
        );
        let element_bytes = match self.packed_layer_dtypes.iter().flatten().next() {
            Some((mlx_rs::Dtype::Float32, _)) => 4,
            Some((mlx_rs::Dtype::Float16 | mlx_rs::Dtype::Bfloat16, _)) => 2,
            Some((dtype, _)) => {
                return Err(Error::Msg(format!(
                    "packed cache retained unsupported K dtype {dtype:?}"
                )))
            }
            None => {
                return Err(Error::Msg(
                    "packed cache has resident values without dtype evidence".into(),
                ))
            }
        };
        Ok(Some(CompressedCacheStorage {
            device_code_bytes,
            device_metadata_bytes,
            host_payload_bytes: u64::try_from(self.staged.host_allocated_payload_bytes())
                .map_err(|_| Error::Msg("packed host payload bytes overflow u64".into()))?,
            tokens: u64::try_from(self.staged.logical_len())
                .map_err(|_| Error::Msg("packed cache length overflows u64".into()))?,
            element_bytes,
        }))
    }

    /// The dense cache that owns history after an explicit transition (and records its own
    /// allocation/release events). Campaign observation reads it whether or not it is active so a
    /// reset's release is never missed.
    fn compressed_dense_fallback(&self) -> Option<&ContiguousKvCache> {
        Some(&self.dense)
    }

    fn record_events(&mut self) {
        self.dense.record_events()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        if self.dense_active {
            return self.dense.retain_sequences(keep);
        }
        if self.pending_step.is_some() {
            return Err(Error::Unsupported(
                "retain_sequences cannot mutate a pending packed step".into(),
            ));
        }
        if self.staged.logical_len() != 0 {
            self.transition_to_dense(
                "retain_sequences",
                "batch compaction requires exact dense reconstruction",
            )?;
            return self.dense.retain_sequences(keep);
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if self.dense_active {
            self.speculation = None;
            return self.dense.truncate(len);
        }
        if self.pending_step.is_some() {
            self.rollback_pending("truncate", "discarded pending packed step")?;
        }
        let len = usize::try_from(len).map_err(|_| {
            Error::Config("packed cache truncate length must not be negative".into())
        })?;
        match self.speculation.take() {
            Some(speculation) if len >= speculation.base && len < self.staged.logical_len() => self
                .staged
                .trim_into_speculation(len, &speculation.layers)?,
            _ => self.staged.trim(len)?,
        }
        if len == 0 {
            self.packed_layer_dtypes.fill(None);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.dense.reset()?;
        self.speculation = None;
        if let Some(pending) = self.pending_step.take() {
            self.staged
                .restore_device_marks(pending.marks, pending.original_len);
            self.staged
                .restore_accepted_dispatch(pending.accepted_dispatch);
            self.staged
                .dense_read_fallback("reset", "discarded pending packed step");
        }
        self.staged.clear();
        self.packed_layer_dtypes.fill(None);
        self.dense_active = self.staged.compiled_handle().is_none();
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn dense_selection(layers: usize, reason: impl Into<String>) -> DecoderCacheSelection {
    DecoderCacheSelection {
        route: CacheRoute::DenseFallback {
            reason: reason.into(),
        },
        cache: Box::new(ContiguousKvCache::new(layers)),
    }
}

/// Construct the cache which a decoder will use, before the first append.  Disabled overrides and
/// every unsupported geometry return the established contiguous implementation without allocating
/// a packed cache.  A supported override stages the packed format but still deterministically uses
/// dense until a retained SC-20676 reader is bound; this prevents a metadata-only path from being
/// mistaken for compressed-domain execution.
pub fn select_decoder_cache(request: PackedCacheRequest) -> DecoderCacheSelection {
    if !request.enabled {
        return dense_selection(
            request.layers,
            "experimental packed cache override is disabled",
        );
    }
    if request.backend != "mlx-metal" {
        return dense_selection(request.layers, "packed cache requires mlx-metal backend");
    }
    if request.has_mask {
        return dense_selection(
            request.layers,
            "additive attention masks require dense fallback",
        );
    }
    if request.layers == 0
        || request.batch == 0
        || request.kv_heads == 0
        || request.query_length == 0
    {
        return dense_selection(
            request.layers,
            "packed cache geometry is unsupported before allocation",
        );
    }
    if !packed_metal_head_dimension_supported(request.head_dimension) {
        return dense_selection(
            request.layers,
            "packed Metal head dimension must be 64, 128, or 256",
        );
    }
    if !packed_metal_cache_geometry_supported(request.head_dimension, request.group_size) {
        return dense_selection(
            request.layers,
            format!("packed Metal quantization group size must be {PACKED_METAL_QUANT_GROUP_SIZE}"),
        );
    }
    let staged = match PackedGroupAffineKvCache::with_bits(
        request.identity,
        request.layers,
        request.batch,
        request.kv_heads,
        request.head_dimension,
        request.group_size,
        request.bits,
    ) {
        Ok(cache) => cache,
        Err(error) => {
            return dense_selection(request.layers, format!("packed cache rejected: {error}"))
        }
    };
    let reason = "packed storage has no retained fused reader; use dense cache".to_string();
    DecoderCacheSelection {
        route: CacheRoute::DenseFallback {
            reason: reason.clone(),
        },
        cache: Box::new(DenseFallbackPackedDecoderCache {
            dense: ContiguousKvCache::new(request.layers),
            staged,
            pending_step: None,
            packed_layer_dtypes: vec![None; request.layers],
            dense_active: false,
            reason,
            speculation: None,
            transition_admission: None,
        }),
    }
}

/// Explicit experimental construction route.  The retained reader is supplied by the model
/// after its backend/device capability probe; binding happens before the returned cache can see an
/// update.  The legacy factory above intentionally remains dense by default.
pub fn select_decoder_cache_with_reader(
    request: PackedCacheRequest,
    handle: CompiledKernelHandle,
) -> DecoderCacheSelection {
    let mut selection = select_decoder_cache(request.clone());
    let Some(cache) = selection
        .cache
        .as_any_mut()
        .downcast_mut::<DenseFallbackPackedDecoderCache>()
    else {
        return selection;
    };
    if let Err(error) = cache.bind_compiled_handle(handle) {
        selection.route = CacheRoute::DenseFallback {
            reason: format!("packed reader rejected before mutation: {error}"),
        };
        return selection;
    }
    selection.route = cache.preflight_packed(request.query_length, request.has_mask);
    selection
}

/// One fused-reader parity case: `history` tokens resident, `query_len` query rows at the end of
/// the KV range, attended with `mask`.
struct ParityCase {
    history: usize,
    query_len: usize,
    mask: PackedAttentionMask,
}

/// Fused packed-reader parity against an independent fp32 dequantize-then-attend oracle, through
/// the decoder's production data path: K/V quantized on the device by the append path (completed
/// 32-token groups packed, the incomplete group in the dense residual), dispatched by the retained
/// reader, and compared with host-fp32 attention (`attention_f32_masked`) over the values the
/// device representation holds, reconstructed by MLX dequantization independently of the kernel.
/// Cases cover multiple K groups plus a residual tail at decode (77 tokens), a causal prefill chunk
/// (`S_q = 5`) over that history, and a history long enough for the split-KV heuristic to choose
/// several splits. The cache is built at the reader's [`CompiledKernelHandle::code_bits`], so a
/// 4-bit reader is checked on 4-bit codes. Returns every per-element absolute error.
pub fn group_affine_kernel_fp32_parity_errors(
    reader: &CompiledKernelHandle,
) -> std::result::Result<Vec<f64>, String> {
    group_affine_kernel_fp32_parity_errors_at(reader, 128)
}

/// [`group_affine_kernel_fp32_parity_errors`] at `head_dimension` (a multiple of the quantization
/// group), so a model's gate runs the kernel at the width it actually decodes with.
pub fn group_affine_kernel_fp32_parity_errors_at(
    reader: &CompiledKernelHandle,
    head_dimension: usize,
) -> std::result::Result<Vec<f64>, String> {
    if head_dimension == 0 || !head_dimension.is_multiple_of(PACKED_METAL_QUANT_GROUP_SIZE) {
        return Err(format!(
            "packed kernel parity head dimension {head_dimension} is not a multiple of the quantization group"
        ));
    }
    const SPLIT_HISTORY: usize = 525;
    // One batch row, one query position, two query heads per KV head: two threadgroup rows.
    if crate::primitives::packed_metal::packed_kv_split_count(2, SPLIT_HISTORY, 1) < 2 {
        return Err("packed kernel parity split case no longer selects split-KV".into());
    }
    let cases = [
        ParityCase {
            history: 77,
            query_len: 1,
            mask: PackedAttentionMask::None,
        },
        ParityCase {
            history: 82,
            query_len: 5,
            mask: PackedAttentionMask::Causal,
        },
        ParityCase {
            history: SPLIT_HISTORY,
            query_len: 1,
            mask: PackedAttentionMask::Causal,
        },
    ];
    let mut errors = Vec::new();
    for case in cases {
        errors.extend(group_affine_parity_case(reader, &case, head_dimension)?);
    }
    mlx_rs::memory::clear_cache();
    Ok(errors)
}

fn group_affine_parity_case(
    reader: &CompiledKernelHandle,
    case: &ParityCase,
    width: usize,
) -> std::result::Result<Vec<f64>, String> {
    const QUERY_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    let tokens = case.history;
    let query = (0..QUERY_HEADS * case.query_len * width)
        .map(|index| (index as i32 % 17 - 8) as f32 * 0.05)
        .collect::<Vec<_>>();
    // A slow per-token ramp on K makes score maxima differ across KV splits, so the split-KV
    // rescaling is observable (the periodic base pattern alone would give every split equal maxima).
    let keys = (0..KV_HEADS * tokens * width)
        .map(|index| {
            let token = (index / width) % tokens;
            (index as i32 % 29 - 14) as f32 * 0.05 + (token % 97) as f32 * 0.004
        })
        .collect::<Vec<_>>();
    let values = (0..KV_HEADS * tokens * width)
        .map(|index| (index as i32 % 23 - 11) as f32 * 0.05)
        .collect::<Vec<_>>();
    let shape = [1, KV_HEADS as i32, tokens as i32, width as i32];
    let mut cache = PackedGroupAffineKvCache::with_bits(
        reader.cache_identity(),
        1,
        1,
        KV_HEADS,
        width,
        PACKED_METAL_QUANT_GROUP_SIZE,
        reader.code_bits(),
    )
    .map_err(|e| format!("packed kernel parity cache: {e}"))?;
    cache
        .append_device(
            0,
            &Array::from_slice(&keys, &shape),
            &Array::from_slice(&values, &shape),
        )
        .map_err(|e| format!("packed kernel parity append: {e}"))?;
    cache
        .bind_compiled_handle(reader.clone())
        .map_err(|e| format!("packed kernel parity reader: {e}"))?;
    let (resident, dequantized_keys, dequantized_values) = cache
        .evaluated_dense_layer(0)
        .map_err(|e| format!("packed kernel parity readback: {e}"))?;
    if resident != tokens {
        return Err("packed kernel parity readback has the wrong length".into());
    }
    let q = Array::from_slice(
        &query,
        &[1, QUERY_HEADS as i32, case.query_len as i32, width as i32],
    );
    let mask = match case.mask {
        PackedAttentionMask::None => crate::primitives::packed_metal::PackedMask::None,
        PackedAttentionMask::Causal => crate::primitives::packed_metal::PackedMask::Causal,
        PackedAttentionMask::SlidingWindow(window) => {
            crate::primitives::packed_metal::PackedMask::SlidingWindow(window)
        }
        PackedAttentionMask::Additive => {
            return Err("packed kernel parity has no additive case".into())
        }
    };
    let output = cache
        .dispatch_packed(0, &q, mask)
        .map_err(|e| format!("packed kernel parity dispatch: {e}"))?
        .as_dtype(Dtype::Float32)
        .map_err(|e| format!("packed kernel parity dtype: {e}"))?;
    output
        .eval()
        .map_err(|e| format!("packed kernel parity evaluation: {e}"))?;
    if cache.direct_dispatches() != 1 || cache.full_cache_dequantizations() != 0 {
        return Err("packed kernel parity did not run exactly one fused dispatch".into());
    }
    let expected = crate::primitives::packed_attention::attention_f32_masked(
        crate::primitives::packed_attention::PackedAttentionShape {
            batch: 1,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            query_len: case.query_len,
            kv_len: tokens,
            head_dim: width,
        },
        &query,
        |_, head, token, channel| dequantized_keys[(head * tokens + token) * width + channel],
        |_, head, token, channel| dequantized_values[(head * tokens + token) * width + channel],
        (width as f32).powf(-0.5),
        case.mask,
    )
    .map_err(|e| format!("packed kernel parity oracle: {e}"))?;
    let errors = output
        .as_slice::<f32>()
        .iter()
        .zip(expected)
        .map(|(actual, expected)| f64::from((*actual - expected).abs()))
        .collect::<Vec<_>>();
    if errors.len() != QUERY_HEADS * case.query_len * width {
        return Err("packed kernel parity output has the wrong shape".into());
    }
    Ok(errors)
}

#[derive(Clone, Debug)]
struct PackedTensor {
    bits: PackedCodeBits,
    rows: usize,
    width: usize,
    groups: usize,
    codes: Vec<u8>,
    scales: Vec<f16>,
    zeros: Vec<f16>,
}

impl PackedTensor {
    fn new(
        bits: PackedCodeBits,
        rows: usize,
        width: usize,
        group_size: usize,
        capacity_rows: usize,
    ) -> Self {
        let groups = width.div_ceil(group_size);
        Self {
            bits,
            rows,
            width,
            groups,
            codes: Vec::with_capacity(capacity_rows * bits.code_bytes(width)),
            scales: Vec::with_capacity(capacity_rows * groups),
            zeros: Vec::with_capacity(capacity_rows * groups),
        }
    }

    fn append(&mut self, values: &[f32], group_size: usize) -> Result<()> {
        if values.len() != self.width {
            return Err(Error::Config("packed KV row width mismatch".into()));
        }
        let code_start = self.codes.len();
        let bits = self.bits;
        self.codes
            .resize(code_start + bits.code_bytes(self.width), 0);
        for group in 0..self.groups {
            let start = group * group_size;
            let end = (start + group_size).min(self.width);
            let slice = &values[start..end];
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let (zero, scale) = bits.affine(min, max);
            self.scales.push(scale);
            self.zeros.push(zero);
            for (i, value) in slice.iter().copied().enumerate() {
                bits.pack(
                    &mut self.codes[code_start..],
                    start + i,
                    bits.quantize(value, zero, scale),
                );
            }
        }
        self.rows += 1;
        Ok(())
    }

    fn row(&self, row: usize, group_size: usize) -> Result<Vec<f32>> {
        if row >= self.rows {
            return Err(Error::Config("packed KV row out of range".into()));
        }
        let mut out = Vec::with_capacity(self.width);
        let row_codes = &self.codes[row * self.bits.code_bytes(self.width)..];
        for i in 0..self.width {
            let code = self.bits.unpack(row_codes, i);
            let group = i / group_size;
            out.push(
                self.zeros[row * self.groups + group].to_f32()
                    + self.scales[row * self.groups + group].to_f32() * code as f32,
            );
        }
        Ok(out)
    }

    fn truncate(&mut self, rows: usize) {
        self.rows = rows;
        self.codes.truncate(rows * self.bits.code_bytes(self.width));
        self.scales.truncate(rows * self.groups);
        self.zeros.truncate(rows * self.groups);
    }

    fn bytes(&self) -> usize {
        self.codes.len() + (self.scales.len() + self.zeros.len()) * std::mem::size_of::<f16>()
    }
    fn allocated_bytes(&self) -> usize {
        self.codes.capacity()
            + (self.scales.capacity() + self.zeros.capacity()) * std::mem::size_of::<f16>()
    }

    /// [`Self::reserve_rows`] for an untrusted row count (a restored snapshot's capacity): `Err`
    /// on overflow or a failed allocation instead of a panic.
    fn try_reserve_rows(&mut self, rows: usize) -> Result<()> {
        let refused = || Error::Config("snapshot capacity cannot be reserved".into());
        let codes = rows
            .checked_mul(self.bits.code_bytes(self.width))
            .ok_or_else(refused)?;
        let metadata = rows.checked_mul(self.groups).ok_or_else(refused)?;
        self.codes
            .try_reserve(codes.saturating_sub(self.codes.len()))
            .map_err(|_| refused())?;
        self.scales
            .try_reserve(metadata.saturating_sub(self.scales.len()))
            .map_err(|_| refused())?;
        self.zeros
            .try_reserve(metadata.saturating_sub(self.zeros.len()))
            .map_err(|_| refused())
    }

    fn reserve_rows(&mut self, rows: usize) {
        self.codes.reserve(
            rows.saturating_mul(self.bits.code_bytes(self.width))
                .saturating_sub(self.codes.len()),
        );
        self.scales.reserve(
            rows.saturating_mul(self.groups)
                .saturating_sub(self.scales.len()),
        );
        self.zeros.reserve(
            rows.saturating_mul(self.groups)
                .saturating_sub(self.zeros.len()),
        );
    }
}

/// Key storage groups tokens per channel (`[B,H,ceil(S/group),D]`).  Only an
/// incomplete final token group is staged densely; every completed group is
/// physically `bits`-wide codes plus f16 scale/zero metadata.
#[derive(Clone, Debug)]
struct TokenGroupKeyTensor {
    bits: PackedCodeBits,
    rows: usize,
    width: usize,
    group_size: usize,
    complete_tokens: usize,
    pending_tokens: usize,
    codes: Vec<u8>,
    scales: Vec<f16>,
    zeros: Vec<f16>,
    // `[pending_token, row, channel]`, bounded to `group_size - 1` tokens.
    pending: Vec<f32>,
}

impl TokenGroupKeyTensor {
    fn new(
        bits: PackedCodeBits,
        rows: usize,
        width: usize,
        group_size: usize,
        capacity_tokens: usize,
    ) -> Self {
        let groups = capacity_tokens.div_ceil(group_size);
        Self {
            bits,
            rows,
            width,
            group_size,
            complete_tokens: 0,
            pending_tokens: 0,
            codes: Vec::with_capacity(rows * groups * bits.code_bytes(group_size * width)),
            scales: Vec::with_capacity(rows * groups * width),
            zeros: Vec::with_capacity(rows * groups * width),
            pending: Vec::with_capacity(rows * group_size.saturating_sub(1) * width),
        }
    }

    fn logical_tokens(&self) -> usize {
        self.complete_tokens + self.pending_tokens
    }

    fn complete_groups(&self) -> usize {
        self.complete_tokens / self.group_size
    }

    fn code_bytes_per_group(&self) -> usize {
        self.bits.code_bytes(self.group_size * self.width)
    }

    fn append(&mut self, values: &[f32], step: usize) -> Result<()> {
        if values.len() != self.rows * step * self.width {
            return Err(Error::Config("packed key append shape mismatch".into()));
        }
        for token in 0..step {
            for row in 0..self.rows {
                let start = (row * step + token) * self.width;
                self.pending
                    .extend_from_slice(&values[start..start + self.width]);
            }
            self.pending_tokens += 1;
            if self.pending_tokens == self.group_size {
                self.flush_pending_group()?;
            }
        }
        Ok(())
    }

    fn flush_pending_group(&mut self) -> Result<()> {
        if self.pending_tokens != self.group_size {
            return Err(Error::Config(
                "incomplete key token group cannot be quantized".into(),
            ));
        }
        let code_start = self.codes.len();
        self.codes
            .resize(code_start + self.rows * self.code_bytes_per_group(), 0);
        let code_bytes_per_group = self.code_bytes_per_group();
        let group_index = self.complete_groups();
        let bits = self.bits;
        for row in 0..self.rows {
            for channel in 0..self.width {
                let min = (0..self.group_size)
                    .map(|token| self.pending[(token * self.rows + row) * self.width + channel])
                    .fold(f32::INFINITY, f32::min);
                let max = (0..self.group_size)
                    .map(|token| self.pending[(token * self.rows + row) * self.width + channel])
                    .fold(f32::NEG_INFINITY, f32::max);
                let (zero, scale) = bits.affine(min, max);
                self.scales.push(scale);
                self.zeros.push(zero);
                for token in 0..self.group_size {
                    let value = self.pending[(token * self.rows + row) * self.width + channel];
                    bits.pack(
                        &mut self.codes[code_start + row * code_bytes_per_group..],
                        token * self.width + channel,
                        bits.quantize(value, zero, scale),
                    );
                }
            }
        }
        debug_assert_eq!(
            self.scales.len(),
            (group_index + 1) * self.rows * self.width
        );
        self.complete_tokens += self.group_size;
        self.pending_tokens = 0;
        self.pending.clear();
        Ok(())
    }

    fn truncate(&mut self, tokens: usize) -> Result<()> {
        if tokens > self.logical_tokens() {
            return Err(Error::Config(
                "packed key truncate exceeds logical length".into(),
            ));
        }
        let complete_tokens = tokens / self.group_size * self.group_size;
        let complete_groups = complete_tokens / self.group_size;
        let keep_pending = tokens - complete_tokens;
        let pending = if keep_pending == 0 {
            Vec::new()
        } else if complete_tokens == self.complete_tokens {
            self.pending[..keep_pending * self.rows * self.width].to_vec()
        } else {
            // A rollback may cut a completed group. Re-stage its retained prefix from the
            // already-quantized representation; no discarded dense mirror is retained.
            let mut retained = Vec::with_capacity(keep_pending * self.rows * self.width);
            for token in 0..keep_pending {
                for row in 0..self.rows {
                    retained.extend(self.row(complete_tokens + token, row)?);
                }
            }
            retained
        };
        self.codes
            .truncate(complete_groups * self.rows * self.code_bytes_per_group());
        self.scales
            .truncate(complete_groups * self.rows * self.width);
        self.zeros
            .truncate(complete_groups * self.rows * self.width);
        if keep_pending == 0 {
            self.pending.clear();
        } else {
            self.pending = pending;
        }
        self.complete_tokens = complete_tokens;
        self.pending_tokens = keep_pending;
        Ok(())
    }

    fn row(&self, token: usize, row: usize) -> Result<Vec<f32>> {
        if token >= self.logical_tokens() || row >= self.rows {
            return Err(Error::Config("packed key row out of range".into()));
        }
        if token >= self.complete_tokens {
            let pending_token = token - self.complete_tokens;
            let start = (pending_token * self.rows + row) * self.width;
            return Ok(self.pending[start..start + self.width].to_vec());
        }
        let group = token / self.group_size;
        let local_token = token % self.group_size;
        let code_base =
            group * self.rows * self.code_bytes_per_group() + row * self.code_bytes_per_group();
        let metadata_base = (group * self.rows + row) * self.width;
        let mut out = Vec::with_capacity(self.width);
        for channel in 0..self.width {
            let code = self
                .bits
                .unpack(&self.codes[code_base..], local_token * self.width + channel);
            out.push(
                self.zeros[metadata_base + channel].to_f32()
                    + self.scales[metadata_base + channel].to_f32() * code as f32,
            );
        }
        Ok(out)
    }

    fn logical_bytes(&self) -> usize {
        self.codes.len()
            + (self.scales.len() + self.zeros.len()) * std::mem::size_of::<f16>()
            + self.pending.len() * std::mem::size_of::<f32>()
    }

    fn allocated_bytes(&self) -> usize {
        self.codes.capacity()
            + (self.scales.capacity() + self.zeros.capacity()) * std::mem::size_of::<f16>()
            + self.pending.capacity() * std::mem::size_of::<f32>()
    }

    /// [`Self::reserve_tokens`] for an untrusted token count (a restored snapshot's capacity):
    /// `Err` on overflow or a failed allocation instead of a panic.
    fn try_reserve_tokens(&mut self, tokens: usize) -> Result<()> {
        let refused = || Error::Config("snapshot capacity cannot be reserved".into());
        let groups = self
            .rows
            .checked_mul(tokens.div_ceil(self.group_size))
            .ok_or_else(refused)?;
        let codes = groups
            .checked_mul(self.code_bytes_per_group())
            .ok_or_else(refused)?;
        let metadata = groups.checked_mul(self.width).ok_or_else(refused)?;
        self.codes
            .try_reserve(codes.saturating_sub(self.codes.len()))
            .map_err(|_| refused())?;
        self.scales
            .try_reserve(metadata.saturating_sub(self.scales.len()))
            .map_err(|_| refused())?;
        self.zeros
            .try_reserve(metadata.saturating_sub(self.zeros.len()))
            .map_err(|_| refused())
    }

    fn reserve_tokens(&mut self, tokens: usize) {
        let groups = tokens.div_ceil(self.group_size);
        self.codes.reserve(
            self.rows
                .saturating_mul(groups)
                .saturating_mul(self.code_bytes_per_group())
                .saturating_sub(self.codes.len()),
        );
        self.scales.reserve(
            self.rows
                .saturating_mul(groups)
                .saturating_mul(self.width)
                .saturating_sub(self.scales.len()),
        );
        self.zeros.reserve(
            self.rows
                .saturating_mul(groups)
                .saturating_mul(self.width)
                .saturating_sub(self.zeros.len()),
        );
    }
}

#[derive(Clone, Debug)]
struct LayerStorage {
    keys: TokenGroupKeyTensor,
    values: PackedTensor,
}

/// Geometry of one layer's packed arrays.
#[derive(Clone, Copy, Debug)]
pub(super) struct PackedGeometry {
    pub(super) batch: usize,
    pub(super) heads: usize,
    pub(super) dim: usize,
    pub(super) group: usize,
    pub(super) bits: PackedCodeBits,
}

impl PackedGeometry {
    fn rows(self) -> usize {
        self.batch * self.heads
    }
    fn key_words(self) -> usize {
        self.bits.code_bytes(self.group * self.dim)
    }
    fn value_words(self) -> usize {
        self.bits.code_bytes(self.dim)
    }
    fn value_groups(self) -> usize {
        self.dim.div_ceil(self.group)
    }
    /// Growth granularity of the packed arrays: [`KV_BLOCK_TOKENS`] rounded up to whole K groups.
    fn block_tokens(self) -> usize {
        usize::try_from(KV_BLOCK_TOKENS)
            .unwrap_or(256)
            .div_ceil(self.group)
            * self.group
    }
    pub(super) fn shape(self, tokens: usize, width: usize) -> Result<[i32; 4]> {
        mlx_shape([self.batch, self.heads, tokens, width])
    }
}

pub(super) fn array_bytes(array: &Array) -> u64 {
    // `usize` always fits in `u64` on supported targets.
    (array.size() as u64).saturating_mul(array.item_size() as u64)
}

pub(super) fn dtype_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Float32 => 4,
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        _ => 4,
    }
}

/// Write `delta` into `array[.., .., start..start + delta_len, ..]` in place. MLX's slice update
/// donates the buffer when the cache holds the only reference, so no history is copied.
pub(super) fn write_rows(array: &mut Array, start: usize, delta: &Array) -> Result<()> {
    let start = i32::try_from(start)
        .map_err(|_| Error::Config("packed KV row offset exceeds MLX i32 range".into()))?;
    let end = start
        .checked_add(delta.shape()[2])
        .ok_or_else(|| Error::Config("packed KV row end exceeds MLX i32 range".into()))?;
    array.try_index_mut((.., .., start..end, ..), delta)?;
    Ok(())
}

pub(super) fn rows_range(array: &Array, start: usize, end: usize) -> Result<Array> {
    let bound = |value: usize| {
        i32::try_from(value)
            .map_err(|_| Error::Config("packed KV row range exceeds MLX i32 range".into()))
    };
    Ok(array.try_index((.., .., bound(start)?..bound(end)?, ..))?)
}

pub(super) fn live_rows(array: &Array, rows: usize) -> Result<Array> {
    rows_range(array, 0, rows)
}

/// An owned, row-contiguous copy, so a host readback of a strided view is valid. `x · 1` (bit-exact)
/// releases the source buffer but keeps a permuted-dense input's strides; [`contiguous`] then
/// makes it row-major (a no-op share when it already is).
///
/// [`contiguous`]: crate::primitives::nn::contiguous
fn materialized(array: &Array) -> Result<Array> {
    let one = Array::from_slice(&[1i32], &[1]).as_dtype(array.dtype())?;
    crate::primitives::nn::contiguous(&multiply(array, &one)?)
}

/// Unpack `8 / bits` codes per byte along the last axis, low bits first: `[..., W]` Uint8 →
/// `[..., W · 8 / bits]` Float32. Integer place-value arithmetic, independent of the kernels'
/// shift-and-mask decoding.
fn unpack_codes(codes: &Array, bits: PackedCodeBits) -> Result<Array> {
    let per_byte = bits.codes_per_byte();
    let mut shape = codes.shape().to_vec();
    let last = shape.len() - 1;
    shape[last] *= i32::try_from(per_byte)
        .map_err(|_| Error::Msg("internal packed code width conversion failed".into()))?;
    let widened = codes.as_dtype(Dtype::Uint32)?.expand_dims(-1)?;
    let levels = u32::from(bits.max_code()) + 1;
    let place = (0..per_byte)
        .map(|k| levels.pow(k as u32))
        .collect::<Vec<_>>();
    let shifted = floor_divide(&widened, Array::from_slice(&place, &[per_byte as i32]))?;
    let digits = remainder(&shifted, Array::from_slice(&[levels], &[1]))?;
    Ok(digits.reshape(&shape)?.as_dtype(Dtype::Float32)?)
}

/// `zero + scale · code` for the first `groups` K token groups: `[B,H,groups·G,D]` Float32. The
/// separate multiply and add match the CPU reference's rounding.
pub(super) fn dequantize_key_groups(
    geometry: PackedGeometry,
    codes: &Array,
    scales: &Array,
    zeros: &Array,
) -> Result<Array> {
    let codes = unpack_codes(codes, geometry.bits)?;
    let (b, h, g, group, d) = (
        codes.shape()[0],
        codes.shape()[1],
        codes.shape()[2],
        i32::try_from(geometry.group)
            .map_err(|_| Error::Config("packed group exceeds MLX i32 range".into()))?,
        i32::try_from(geometry.dim)
            .map_err(|_| Error::Config("packed width exceeds MLX i32 range".into()))?,
    );
    let codes = codes.reshape(&[b, h, g, group, d])?;
    let scales = scales.as_dtype(Dtype::Float32)?.expand_dims(3)?;
    let zeros = zeros.as_dtype(Dtype::Float32)?.expand_dims(3)?;
    Ok(add(&zeros, &multiply(&scales, &codes)?)?.reshape(&[b, h, g * group, d])?)
}

/// `zero + scale · code` for the first `tokens` V rows: `[B,H,tokens,D]` Float32.
pub(super) fn dequantize_value_rows(
    geometry: PackedGeometry,
    codes: &Array,
    scales: &Array,
    zeros: &Array,
) -> Result<Array> {
    let codes = unpack_codes(codes, geometry.bits)?;
    let (b, h, t) = (codes.shape()[0], codes.shape()[1], codes.shape()[2]);
    let (groups, group) = (
        i32::try_from(geometry.value_groups())
            .map_err(|_| Error::Config("packed value groups exceed MLX i32 range".into()))?,
        i32::try_from(geometry.group)
            .map_err(|_| Error::Config("packed group exceeds MLX i32 range".into()))?,
    );
    let d = i32::try_from(geometry.dim)
        .map_err(|_| Error::Config("packed width exceeds MLX i32 range".into()))?;
    // Channel groups are whole for every accepted width; the CPU reference permits a ragged last
    // group only for arbitrary host-only widths, which never reach the device.
    let codes = codes
        .try_index((.., .., .., ..d))?
        .reshape(&[b, h, t, groups, group])?;
    let scales = scales.as_dtype(Dtype::Float32)?.expand_dims(4)?;
    let zeros = zeros.as_dtype(Dtype::Float32)?.expand_dims(4)?;
    Ok(add(&zeros, &multiply(&scales, &codes)?)?.reshape(&[b, h, t, d])?)
}

/// Device-resident packed K/V for one layer: block-preallocated packed arrays written in place plus
/// a bounded dense residual for K and for V. Live extents are counts, never array shapes, so an
/// append never copies history; the packed arrays grow by whole blocks (one copy per block).
///
/// Two owners use this layout. The decoder route (`authoritative`) holds its K/V only here: an
/// append quantizes every completed 32-token group on the GPU and keeps the incomplete group in the
/// dense residuals, so no host copy exists. The CPU reference cache (a mirror) uploads its
/// host-quantized deltas into the same arrays; its V is quantized per token on the host, so its
/// value residual is empty and its key residual is the host's pending group.
#[derive(Clone, Debug)]
struct DevicePackedLayer {
    authoritative: bool,
    bits: PackedCodeBits,
    capacity_tokens: usize,
    /// Leading tokens whose K is quantized (a multiple of the group size).
    key_packed_tokens: usize,
    /// Leading tokens whose V is quantized.
    value_packed_tokens: usize,
    key_tail_rows: usize,
    value_tail_rows: usize,
    key_codes: Array,
    key_scales: Array,
    key_zeros: Array,
    value_codes: Array,
    value_scales: Array,
    value_zeros: Array,
    key_tail: Array,
    value_tail: Array,
}

/// A layer's `(key, value)` dense residual arrays.
type Residuals = (Array, Array);

/// Rollback point of one device layer. The packed arrays and the residuals are written only past
/// these counts, so the counts alone restore them; a residual array is held only once an append
/// replaced it (a group flush), which keeps every in-place residual write donatable.
#[derive(Clone, Debug)]
struct DeviceLayerMark {
    key_packed_tokens: usize,
    value_packed_tokens: usize,
    key_tail_rows: usize,
    value_tail_rows: usize,
    residuals: Option<Residuals>,
}

/// An evaluated, unpublished reader state for one layer (see
/// [`PackedGroupAffineKvCache::staged_reader_arguments`]).
pub(crate) struct StagedReaderLayer(DevicePackedLayer);

impl StagedReaderLayer {
    pub(crate) fn args<'a>(
        &'a self,
        query: &'a Array,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> PackedAttentionArgs<'a> {
        self.0.args(query, mask)
    }
}

impl DevicePackedLayer {
    fn empty(
        geometry: PackedGeometry,
        capacity_tokens: usize,
        authoritative: bool,
        key_tail: (usize, Dtype),
        value_tail: (usize, Dtype),
    ) -> Result<Self> {
        let groups = capacity_tokens / geometry.group;
        let metadata = |tokens, width| -> Result<Array> {
            Ok(zeros_dtype(
                &geometry.shape(tokens, width)?,
                Dtype::Float16,
            )?)
        };
        Ok(Self {
            authoritative,
            bits: geometry.bits,
            capacity_tokens,
            key_packed_tokens: 0,
            value_packed_tokens: 0,
            key_tail_rows: 0,
            value_tail_rows: 0,
            key_codes: zeros_dtype(&geometry.shape(groups, geometry.key_words())?, Dtype::Uint8)?,
            key_scales: metadata(groups, geometry.dim)?,
            key_zeros: metadata(groups, geometry.dim)?,
            value_codes: zeros_dtype(
                &geometry.shape(capacity_tokens, geometry.value_words())?,
                Dtype::Uint8,
            )?,
            value_scales: metadata(capacity_tokens, geometry.value_groups())?,
            value_zeros: metadata(capacity_tokens, geometry.value_groups())?,
            key_tail: zeros_dtype(&geometry.shape(key_tail.0, geometry.dim)?, key_tail.1)?,
            value_tail: zeros_dtype(&geometry.shape(value_tail.0, geometry.dim)?, value_tail.1)?,
        })
    }

    fn kv_tokens(&self) -> usize {
        self.key_packed_tokens + self.key_tail_rows
    }

    /// Grow the packed arrays by whole blocks so `tokens` quantized tokens fit. Returns the bytes of
    /// the pre-growth arrays that coexisted with their successors (zero when no growth happened).
    fn ensure_packed_capacity(&mut self, geometry: PackedGeometry, tokens: usize) -> Result<u64> {
        if tokens <= self.capacity_tokens {
            return Ok(0);
        }
        let capacity = tokens.div_ceil(geometry.block_tokens()) * geometry.block_tokens();
        let extra = capacity - self.capacity_tokens;
        let before = self.physical_packed_bytes();
        let extend = |array: &Array, rows: usize| -> Result<Array> {
            let mut shape = array.shape().to_vec();
            shape[2] = i32::try_from(rows)
                .map_err(|_| Error::Config("packed KV capacity exceeds MLX i32 range".into()))?;
            Ok(concatenate_axis(
                &[array, &zeros_dtype(&shape, array.dtype())?],
                2,
            )?)
        };
        let extra_groups = extra / geometry.group;
        self.key_codes = extend(&self.key_codes, extra_groups)?;
        self.key_scales = extend(&self.key_scales, extra_groups)?;
        self.key_zeros = extend(&self.key_zeros, extra_groups)?;
        self.value_codes = extend(&self.value_codes, extra)?;
        self.value_scales = extend(&self.value_scales, extra)?;
        self.value_zeros = extend(&self.value_zeros, extra)?;
        self.capacity_tokens = capacity;
        Ok(before)
    }

    /// Counts-only rollback point (residual arrays are captured if an append replaces them).
    fn mark(&self) -> DeviceLayerMark {
        DeviceLayerMark {
            key_packed_tokens: self.key_packed_tokens,
            value_packed_tokens: self.value_packed_tokens,
            key_tail_rows: self.key_tail_rows,
            value_tail_rows: self.value_tail_rows,
            residuals: None,
        }
    }

    /// Rollback point that also holds the current residual arrays (host-mirror dispatch, whose
    /// sync replaces the key residual unconditionally).
    fn mark_with_residuals(&self) -> DeviceLayerMark {
        DeviceLayerMark {
            residuals: Some((self.key_tail.clone(), self.value_tail.clone())),
            ..self.mark()
        }
    }

    fn restore(&mut self, mark: DeviceLayerMark) {
        self.key_packed_tokens = mark.key_packed_tokens;
        self.value_packed_tokens = mark.value_packed_tokens;
        self.key_tail_rows = mark.key_tail_rows;
        self.value_tail_rows = mark.value_tail_rows;
        if let Some((key_tail, value_tail)) = mark.residuals {
            self.key_tail = key_tail;
            self.value_tail = value_tail;
        }
    }

    /// Bytes of the live extents: `(codes, metadata + dense residual)`.
    fn logical_component_bytes(&self, geometry: PackedGeometry) -> (usize, usize) {
        let rows = geometry.rows();
        let groups = self.key_packed_tokens / geometry.group;
        let half = std::mem::size_of::<f16>();
        let codes = rows.saturating_mul(
            groups
                .saturating_mul(geometry.key_words())
                .saturating_add(self.value_packed_tokens * geometry.value_words()),
        );
        let metadata = rows.saturating_mul(
            groups
                .saturating_mul(geometry.dim * 2 * half)
                .saturating_add(self.value_packed_tokens * geometry.value_groups() * 2 * half)
                .saturating_add(
                    self.key_tail_rows * geometry.dim * dtype_bytes(self.key_tail.dtype()),
                )
                .saturating_add(
                    self.value_tail_rows * geometry.dim * dtype_bytes(self.value_tail.dtype()),
                ),
        );
        (codes, metadata)
    }

    fn physical_packed_bytes(&self) -> u64 {
        [
            &self.key_codes,
            &self.key_scales,
            &self.key_zeros,
            &self.value_codes,
            &self.value_scales,
            &self.value_zeros,
        ]
        .into_iter()
        .map(array_bytes)
        .fold(0, u64::saturating_add)
    }

    /// Bytes of the arrays actually retained: `(codes, metadata + dense residual)`.
    fn physical_component_bytes(&self) -> (u64, u64) {
        let codes = array_bytes(&self.key_codes).saturating_add(array_bytes(&self.value_codes));
        let metadata = [
            &self.key_scales,
            &self.key_zeros,
            &self.value_scales,
            &self.value_zeros,
            &self.key_tail,
            &self.value_tail,
        ]
        .into_iter()
        .map(array_bytes)
        .fold(0, u64::saturating_add);
        (codes, metadata)
    }

    fn args<'a>(
        &'a self,
        query: &'a Array,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> PackedAttentionArgs<'a> {
        PackedAttentionArgs {
            query,
            key_codes: &self.key_codes,
            key_scales: &self.key_scales,
            key_zeros: &self.key_zeros,
            key_tail: &self.key_tail,
            value_codes: &self.value_codes,
            value_scales: &self.value_scales,
            value_zeros: &self.value_zeros,
            value_tail: &self.value_tail,
            key_packed_tokens: self.key_packed_tokens,
            value_packed_tokens: self.value_packed_tokens,
            kv_tokens: self.kv_tokens(),
            code_bits: self.bits,
            mask,
        }
    }

    /// The exact K/V the reader consumes, reconstructed densely as Float32 `[B,H,T,D]`. Used only
    /// by an observable dense transition, trim of a completed group, and test/oracle readback.
    fn dense_f32(&self, geometry: PackedGeometry) -> Result<(Array, Array)> {
        let mut keys = Vec::with_capacity(2);
        if self.key_packed_tokens > 0 {
            let groups = self.key_packed_tokens / geometry.group;
            keys.push(dequantize_key_groups(
                geometry,
                &live_rows(&self.key_codes, groups)?,
                &live_rows(&self.key_scales, groups)?,
                &live_rows(&self.key_zeros, groups)?,
            )?);
        }
        if self.key_tail_rows > 0 {
            keys.push(live_rows(&self.key_tail, self.key_tail_rows)?.as_dtype(Dtype::Float32)?);
        }
        let mut values = Vec::with_capacity(2);
        if self.value_packed_tokens > 0 {
            let tokens = self.value_packed_tokens;
            values.push(dequantize_value_rows(
                geometry,
                &live_rows(&self.value_codes, tokens)?,
                &live_rows(&self.value_scales, tokens)?,
                &live_rows(&self.value_zeros, tokens)?,
            )?);
        }
        if self.value_tail_rows > 0 {
            values
                .push(live_rows(&self.value_tail, self.value_tail_rows)?.as_dtype(Dtype::Float32)?);
        }
        let join = |parts: Vec<Array>| -> Result<Array> {
            match parts.as_slice() {
                [] => Err(Error::Config("packed layer is empty".into())),
                [one] => Ok(one.clone()),
                many => Ok(concatenate_axis(&many.iter().collect::<Vec<_>>(), 2)?),
            }
        };
        Ok((join(keys)?, join(values)?))
    }

    /// Trim an authoritative layer to `len` tokens. Cutting into a completed group re-stages the
    /// retained prefix of that group from its quantized values (as the CPU reference does); no
    /// discarded dense copy exists to restore from.
    fn trim_authoritative(&mut self, geometry: PackedGeometry, len: usize) -> Result<()> {
        if len >= self.kv_tokens() {
            return Ok(());
        }
        if len >= self.key_packed_tokens {
            self.key_tail_rows = len - self.key_packed_tokens;
            self.value_tail_rows = len - self.value_packed_tokens;
            return Ok(());
        }
        let keep = len / geometry.group * geometry.group;
        let rows = len - keep;
        if rows > 0 {
            let group = keep / geometry.group;
            let keys = dequantize_key_groups(
                geometry,
                &rows_range(&self.key_codes, group, group + 1)?,
                &rows_range(&self.key_scales, group, group + 1)?,
                &rows_range(&self.key_zeros, group, group + 1)?,
            )?;
            let values = dequantize_value_rows(
                geometry,
                &rows_range(&self.value_codes, keep, len)?,
                &rows_range(&self.value_scales, keep, len)?,
                &rows_range(&self.value_zeros, keep, len)?,
            )?;
            self.key_tail = padded_residual(
                geometry,
                &live_rows(&keys, rows)?.as_dtype(self.key_tail.dtype())?,
                rows,
            )?;
            self.value_tail =
                padded_residual(geometry, &values.as_dtype(self.value_tail.dtype())?, rows)?;
        }
        self.key_packed_tokens = keep;
        self.value_packed_tokens = keep;
        self.key_tail_rows = rows;
        self.value_tail_rows = rows;
        Ok(())
    }
}

/// A `[B,H,G,D]` residual whose first `rows` rows are `rows_array` and the rest zero.
pub(super) fn padded_residual(
    geometry: PackedGeometry,
    rows_array: &Array,
    rows: usize,
) -> Result<Array> {
    if rows >= geometry.group {
        return Ok(rows_array.clone());
    }
    let padding = zeros_dtype(
        &geometry.shape(geometry.group - rows, geometry.dim)?,
        rows_array.dtype(),
    )?;
    Ok(concatenate_axis(&[rows_array, &padding], 2)?)
}

fn row_major_outer_rows<T: Copy>(
    source: &[T],
    outer: usize,
    rows: usize,
    payload: usize,
) -> Result<Vec<T>> {
    if source.len() != outer.saturating_mul(rows).saturating_mul(payload) {
        return Err(Error::Config("packed storage layout mismatch".into()));
    }
    let mut reordered = Vec::with_capacity(source.len());
    for row in 0..rows {
        for item in 0..outer {
            let start = (item * rows + row) * payload;
            reordered.extend_from_slice(&source[start..start + payload]);
        }
    }
    Ok(reordered)
}

#[derive(Clone, Debug)]
pub struct PackedGroupAffineKvCache {
    identity: String,
    group_size: usize,
    bits: PackedCodeBits,
    batch: usize,
    kv_heads: usize,
    head_dimension: usize,
    capacity: usize,
    logical_len: usize,
    absolute_offset: usize,
    layers: Vec<Option<LayerStorage>>,
    device_layers: Vec<Option<DevicePackedLayer>>,
    fallback_events: Vec<DenseFallbackEvent>,
    cancelled: bool,
    /// No reader is installed until the explicit SC-20676 model route binds one for this identity.
    handle: Option<CompiledKernelHandle>,
    /// Counters are cache-owned so a successful packed dispatch cannot be confused with a
    /// dense reconstruction performed by a caller.
    direct_dispatches: usize,
    full_cache_dequantizations: usize,
    /// Accepted calls per reported kernel path (transactional with `direct_dispatches`).
    kernel_paths: Vec<(PackedKernelSelection, u64)>,
    telemetry: PackedDispatchTelemetry,
}

impl PackedGroupAffineKvCache {
    /// A cache of the qualified 2-bit representation.
    pub fn new(
        identity: impl Into<String>,
        layers: usize,
        batch: usize,
        kv_heads: usize,
        head_dimension: usize,
        group_size: usize,
    ) -> Result<Self> {
        Self::with_bits(
            identity,
            layers,
            batch,
            kv_heads,
            head_dimension,
            group_size,
            PackedCodeBits::Two,
        )
    }

    /// A cache whose codes are `bits` wide.
    pub fn with_bits(
        identity: impl Into<String>,
        layers: usize,
        batch: usize,
        kv_heads: usize,
        head_dimension: usize,
        group_size: usize,
        bits: PackedCodeBits,
    ) -> Result<Self> {
        if group_size == 0 || head_dimension == 0 || batch == 0 || kv_heads == 0 || layers == 0 {
            return Err(Error::Config("invalid packed KV shape".into()));
        }
        let _ = mlx_shape([batch, kv_heads, 1, head_dimension])?;
        Ok(Self {
            identity: identity.into(),
            group_size,
            bits,
            batch,
            kv_heads,
            head_dimension,
            capacity: 0,
            logical_len: 0,
            absolute_offset: 0,
            layers: vec![None; layers],
            device_layers: vec![None; layers],
            fallback_events: Vec::new(),
            cancelled: false,
            handle: None,
            direct_dispatches: 0,
            full_cache_dequantizations: 0,
            kernel_paths: Vec::new(),
            telemetry: PackedDispatchTelemetry::default(),
        })
    }

    fn rows(&self) -> usize {
        self.batch * self.kv_heads
    }
    fn row_width(&self) -> usize {
        self.head_dimension
    }
    fn validate_step(&self, values: &[f32], step: usize) -> Result<()> {
        if self.cancelled {
            return Err(Error::Canceled);
        }
        if step == 0 || values.len() != step * self.rows() * self.row_width() {
            return Err(Error::Config("packed KV append shape mismatch".into()));
        }
        Ok(())
    }
    fn ensure_layer(&mut self, layer: usize) -> Result<&mut LayerStorage> {
        let cap = self.capacity.max(1);
        let width = self.row_width();
        let group_size = self.group_size;
        let bits = self.bits;
        let rows = self.rows();
        let slot = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| Error::Config("layer out of range".into()))?;
        if slot.is_none() {
            *slot = Some(LayerStorage {
                keys: TokenGroupKeyTensor::new(bits, rows, width, group_size, cap),
                values: PackedTensor::new(
                    bits,
                    self.logical_len * rows,
                    width,
                    group_size,
                    rows * cap,
                ),
            });
        }
        Ok(slot.as_mut().expect("initialized layer"))
    }
    fn grow(&mut self, required: usize) {
        while self.capacity < required {
            self.capacity = self.capacity.max(1) * 2;
        }
        self.reserve_storage_for_capacity();
    }

    fn reserve_storage_for_capacity(&mut self) {
        let capacity = self.capacity;
        let rows = self.rows();
        for layer in self.layers.iter_mut().flatten() {
            layer.keys.reserve_tokens(capacity);
            layer.values.reserve_rows(rows.saturating_mul(capacity));
        }
    }

    /// Append a contiguous `[batch, kv_heads, step, head_dimension]` slice. Input remains
    /// batch/head-major as required by `KvCache`; keys stage token-axis groups and values pack
    /// each token's channel groups without copying completed historical codes.
    pub fn append(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        step: usize,
    ) -> Result<()> {
        self.validate_step(keys, step)?;
        self.validate_step(values, step)?;
        if layer >= self.layers.len() {
            return Err(Error::Config("layer out of range".into()));
        }
        if self.device_resident() {
            return Err(Error::Config(
                "a device-resident packed cache takes only device appends".into(),
            ));
        }
        let next_logical_len = self
            .logical_len
            .checked_add(step)
            .ok_or_else(|| Error::Config("packed cache length overflow".into()))?;
        i32::try_from(next_logical_len)
            .map_err(|_| Error::Config("packed cache length exceeds MLX i32 range".into()))?;
        self.grow(next_logical_len);
        let rows = self.rows();
        let width = self.row_width();
        let group = self.group_size;
        if self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .is_some_and(|l| l.keys.logical_tokens() != self.logical_len)
        {
            return Err(Error::Msg(
                "layer append is ahead of the atomic commit".into(),
            ));
        }
        {
            let store = self.ensure_layer(layer)?;
            store.keys.append(keys, step)?;
            for token in 0..step {
                for row in 0..rows {
                    let index = (row * step + token) * width;
                    store.values.append(&values[index..index + width], group)?;
                }
            }
        }
        if self.layers.iter().all(Option::is_some)
            && self
                .layers
                .iter()
                .flatten()
                .all(|l| l.keys.logical_tokens() == next_logical_len)
        {
            self.logical_len = next_logical_len;
        }
        Ok(())
    }

    /// Stage every layer against a clone and commit only after all layer shapes/quantization pass.
    /// The caller owns the layer order; no partially appended representation is observable.
    pub fn append_all_layers(&mut self, updates: &[(&[f32], &[f32])], step: usize) -> Result<()> {
        if updates.len() != self.layers.len() {
            return Err(Error::Config("atomic append layer count mismatch".into()));
        }
        let mut staged = self.clone();
        for (layer, (keys, values)) in updates.iter().enumerate() {
            staged.append(layer, keys, values, step)?;
        }
        if staged.logical_len != self.logical_len + step {
            return Err(Error::Msg("atomic append did not commit all layers".into()));
        }
        *self = staged;
        Ok(())
    }

    pub fn trim(&mut self, len: usize) -> Result<()> {
        if len > self.logical_len {
            return Err(Error::Config("trim exceeds logical length".into()));
        }
        let rows = self.rows();
        let geometry = self.geometry();
        for layer in self.layers.iter_mut().flatten() {
            layer.keys.truncate(len)?;
            layer.values.truncate(len * rows);
        }
        let complete = len / self.group_size * self.group_size;
        for device in self.device_layers.iter_mut().flatten() {
            if device.authoritative {
                device.trim_authoritative(geometry, len)?;
            } else {
                // A host mirror keeps only the prefix still identical to the host representation;
                // the next dispatch uploads the rest (including any re-staged pending group).
                device.key_packed_tokens = device.key_packed_tokens.min(complete);
                device.value_packed_tokens = device.value_packed_tokens.min(len);
                device.key_tail_rows = 0;
            }
        }
        self.logical_len = len;
        self.update_retained_telemetry();
        Ok(())
    }
    pub fn rollback(&mut self, len: usize) -> Result<()> {
        self.trim(len)
    }

    /// Per layer, the speculation window an exact rollback needs (sc-20681): the authoritative
    /// device layer's group-aligned quantized extent and its live residual rows. `None` for a
    /// host-staged store (or a host mirror), which keeps no device residual to window.
    fn speculation_rows(&self) -> Result<Option<Vec<SpeculationRows>>> {
        if self.layers.iter().any(Option::is_some) {
            return Ok(None);
        }
        let mut windows = Vec::with_capacity(self.device_layers.len());
        for device in &self.device_layers {
            windows.push(match device {
                None => SpeculationRows::default(),
                Some(device) if !device.authoritative => return Ok(None),
                Some(device) => {
                    let rows = device.key_tail_rows;
                    SpeculationRows {
                        start: device.key_packed_tokens,
                        rows,
                        kv: (rows > 0)
                            .then(|| -> Result<(Array, Array)> {
                                Ok((
                                    live_rows(&device.key_tail, rows)?,
                                    live_rows(&device.value_tail, rows)?,
                                ))
                            })
                            .transpose()?,
                    }
                }
            });
        }
        Ok(Some(windows))
    }

    /// Truncate to `len` inside the speculation `windows` (sc-20681): every group completed below
    /// `len` keeps its codes (they came from exactly the windows' rows), and each residual is
    /// rebuilt from the windows' unquantized rows — never re-staged from codes. Nothing is mutated
    /// unless every layer's window covers the cut.
    fn trim_into_speculation(&mut self, len: usize, windows: &[SpeculationRows]) -> Result<()> {
        if len > self.logical_len || windows.len() != self.device_layers.len() {
            return Err(Error::Config(
                "speculative truncate outside the cache's extent".into(),
            ));
        }
        let geometry = self.geometry();
        let keep = len / self.group_size * self.group_size;
        let rows = len - keep;
        let mut tails = Vec::with_capacity(windows.len());
        for (device, window) in self.device_layers.iter().zip(windows) {
            let Some(device) = device.as_ref().filter(|device| device.authoritative) else {
                tails.push(None);
                continue;
            };
            if keep < window.start || len > window.start + window.rows {
                return Err(Error::Msg(
                    "speculative truncate outside the armed window".into(),
                ));
            }
            let tail = match window.kv.as_ref() {
                Some((keys, values)) if rows > 0 => {
                    let (from, to) = (keep - window.start, len - window.start);
                    (
                        padded_residual(geometry, &rows_range(keys, from, to)?, rows)?,
                        padded_residual(geometry, &rows_range(values, from, to)?, rows)?,
                    )
                }
                _ => (device.key_tail.clone(), device.value_tail.clone()),
            };
            tails.push(Some(tail));
        }
        for (device, tail) in self.device_layers.iter_mut().zip(tails) {
            if let (Some(device), Some((key_tail, value_tail))) = (device.as_mut(), tail) {
                device.key_packed_tokens = keep;
                device.value_packed_tokens = keep;
                device.key_tail_rows = rows;
                device.value_tail_rows = rows;
                device.key_tail = key_tail;
                device.value_tail = value_tail;
            }
        }
        self.logical_len = len;
        self.update_retained_telemetry();
        Ok(())
    }
    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = None);
        self.device_layers.iter_mut().for_each(|l| *l = None);
        self.logical_len = 0;
        self.capacity = 0;
        self.cancelled = false;
        self.telemetry.retained_device_packed_logical_bytes = 0;
    }
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.layers.iter_mut().for_each(|layer| *layer = None);
        self.device_layers
            .iter_mut()
            .for_each(|layer| *layer = None);
        self.logical_len = 0;
        self.capacity = 0;
        self.telemetry.retained_device_packed_logical_bytes = 0;
    }
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }
    pub fn batch_size(&self) -> usize {
        self.batch
    }
    /// Token capacity: host vector capacity for the CPU reference, or the device block capacity
    /// of the decoder's device-resident layers.
    pub fn allocated_len(&self) -> usize {
        if self.device_resident() {
            return self
                .device_layers
                .iter()
                .flatten()
                .map(|device| device.capacity_tokens)
                .max()
                .unwrap_or(0);
        }
        self.capacity
    }
    pub fn absolute_offset(&self) -> usize {
        self.absolute_offset
    }
    pub fn set_absolute_offset(&mut self, offset: usize) {
        self.absolute_offset = offset;
    }
    pub fn layers(&self) -> usize {
        self.layers.len()
    }
    /// Bytes logically occupied by codes, quantization metadata, and an incomplete dense key
    /// group.  This deliberately excludes unused vector capacity and all structural allocations.
    pub fn logical_stored_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.logical_bytes() + l.values.bytes())
            .sum()
    }

    /// Per-component logical bytes of one resident layer, read from the live host vectors:
    /// `[key codes, key scale+zero, pending dense key tail, value codes, value scale+zero]`.
    pub(crate) fn layer_component_bytes(&self, layer: usize) -> Option<[usize; 5]> {
        let storage = self.layers.get(layer)?.as_ref()?;
        let half = std::mem::size_of::<f16>();
        Some([
            storage.keys.codes.len(),
            (storage.keys.scales.len() + storage.keys.zeros.len()) * half,
            storage.keys.pending.len() * std::mem::size_of::<f32>(),
            storage.values.codes.len(),
            (storage.values.scales.len() + storage.values.zeros.len()) * half,
        ])
    }

    /// Actual allocated payload capacity for codes, metadata, and the pending key tail.
    pub fn host_allocated_payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.allocated_bytes() + l.values.allocated_bytes())
            .sum()
    }

    /// Cache-attributable packed bytes retained across host vectors and the allocated MLX arrays
    /// (block capacity and dense residuals included).
    pub fn allocated_payload_bytes(&self) -> usize {
        self.host_allocated_payload_bytes().saturating_add(
            usize::try_from(self.retained_device_physical_bytes()).unwrap_or(usize::MAX),
        )
    }

    /// Compatibility name for callers which need physical packed payload capacity.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_payload_bytes()
    }

    /// Backwards-compatible spelling for physical payload capacity.
    pub fn allocated_vec_bytes(&self) -> usize {
        self.host_allocated_payload_bytes()
    }

    /// A conservative process-visible estimate: payload capacity plus the cache object, backing
    /// `layers`/fallback vectors, identity string capacity, and any per-cache retained reader.
    /// It excludes allocator bookkeeping, shared backend pools, and memory owned outside the
    /// cache, which must be measured by the process-level receipt harness.
    pub fn process_visible_bytes_estimate(&self) -> usize {
        self.allocated_payload_bytes()
            + std::mem::size_of_val(self)
            + self.layers.capacity() * std::mem::size_of::<Option<LayerStorage>>()
            + self.device_layers.capacity() * std::mem::size_of::<Option<DevicePackedLayer>>()
            + self.fallback_events.capacity() * std::mem::size_of::<DenseFallbackEvent>()
            + self.identity.capacity()
            + self
                .fallback_events
                .iter()
                .map(|event| event.operation.capacity() + event.reason.capacity())
                .sum::<usize>()
            + self
                .handle
                .as_ref()
                .map_or(0, CompiledKernelHandle::retained_host_bytes_estimate)
    }

    /// Dense fp16 K+V payload for the same resident layers and logical token length.  It is a
    /// comparison baseline only: dense allocator capacity and process-wide backend pools are not
    /// attributed to this packed-cache estimate.
    pub fn dense_fp16_equivalent_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .count()
            .saturating_mul(self.rows())
            .saturating_mul(self.logical_len)
            .saturating_mul(self.head_dimension)
            .saturating_mul(2) // K and V
            .saturating_mul(std::mem::size_of::<f16>())
    }
    pub fn representation(&self) -> RepresentationMetadata {
        RepresentationMetadata {
            identity: self.identity.clone(),
            version: VERSION,
            group_size: self.group_size,
            bits: self.bits.bits(),
            batch: self.batch,
            kv_heads: self.kv_heads,
            head_dimension: self.head_dimension,
            logical_len: self.logical_len,
            capacity: self.allocated_len(),
            absolute_offset: self.absolute_offset,
            host_allocated_payload_bytes: self.host_allocated_payload_bytes(),
            retained_device_packed_logical_bytes: self.retained_device_packed_logical_bytes(),
            allocated_bytes: self.allocated_payload_bytes(),
            key_grouping: "token-axis groups [B,H,ceil(S/group_size),D]",
            value_grouping: "channel-axis groups [B,H,S,ceil(D/group_size)]",
        }
    }
    pub fn read_row(&self, layer: usize, token: usize, row: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if token >= self.logical_len || row >= self.rows() {
            return Err(Error::Config("packed KV read index out of bounds".into()));
        }
        let Some(l) = self.layers.get(layer).and_then(Option::as_ref) else {
            // Device-resident layers keep no host copy; read back the reconstructed layer.
            let (tokens, keys, values) = self.evaluated_dense_layer(layer)?;
            let start = (row * tokens + token) * self.head_dimension;
            let end = start + self.head_dimension;
            return Ok((keys[start..end].to_vec(), values[start..end].to_vec()));
        };
        let index = token * self.rows() + row;
        Ok((
            l.keys.row(token, row)?,
            l.values.row(index, self.group_size)?,
        ))
    }
    pub fn dense_read_fallback(
        &mut self,
        operation: impl Into<String>,
        reason: impl Into<String>,
    ) -> DenseFallbackEvent {
        let event = DenseFallbackEvent {
            operation: operation.into(),
            reason: reason.into(),
            logical_len: self.logical_len,
            allocated_bytes: self.allocated_payload_bytes(),
        };
        self.fallback_events.push(event.clone());
        event
    }
    pub fn fallback_events(&self) -> &[DenseFallbackEvent] {
        &self.fallback_events
    }
    pub fn no_dense_mirror(&self) -> bool {
        true
    }
    /// Code width of this cache's packed representation.
    pub fn code_bits(&self) -> PackedCodeBits {
        self.bits
    }
    pub fn compiled_handle(&self) -> Option<&CompiledKernelHandle> {
        self.handle.as_ref()
    }
    pub fn bind_compiled_handle(&mut self, handle: CompiledKernelHandle) -> Result<()> {
        if handle.cache_identity() != self.identity {
            return Err(Error::Config("compiled handle identity mismatch".into()));
        }
        if handle.backend() != "mlx-metal" {
            return Err(Error::Config("compiled handle backend mismatch".into()));
        }
        if handle.code_bits() != self.bits {
            return Err(Error::Config(format!(
                "compiled handle reads {}-bit codes but the cache stores {}-bit codes",
                handle.code_bits().bits(),
                self.bits.bits()
            )));
        }
        if !packed_metal_cache_geometry_supported(self.head_dimension, self.group_size) {
            return Err(Error::Config(
                format!(
                    "SC-20676 retained Metal reader requires head dimension 64, 128, or 256 and packed quantization group size {PACKED_METAL_QUANT_GROUP_SIZE}"
                ),
            ));
        }
        self.handle = Some(handle);
        Ok(())
    }

    fn geometry(&self) -> PackedGeometry {
        PackedGeometry {
            batch: self.batch,
            heads: self.kv_heads,
            dim: self.head_dimension,
            group: self.group_size,
            bits: self.bits,
        }
    }

    /// True once the decoder route owns any layer on the device (no host copy exists).
    pub fn device_resident(&self) -> bool {
        self.device_layers
            .iter()
            .flatten()
            .any(|layer| layer.authoritative)
    }

    /// Tokens resident for `layer`, whichever store owns it.
    fn layer_tokens(&self, layer: usize) -> usize {
        if let Some(storage) = self.layers.get(layer).and_then(Option::as_ref) {
            return storage.keys.logical_tokens();
        }
        self.device_layers
            .get(layer)
            .and_then(Option::as_ref)
            .filter(|device| device.authoritative)
            .map_or(0, DevicePackedLayer::kv_tokens)
    }

    fn update_retained_telemetry(&mut self) {
        self.telemetry.retained_device_packed_logical_bytes =
            self.retained_device_packed_logical_bytes() as u64;
    }

    /// Device-resident append for the decoder route. `keys`/`values` are `[B, H, step, D]` MLX
    /// arrays and are never read back: every completed quantization group of `residual ++ fresh`
    /// is quantized on the GPU and written in place into the block-preallocated packed arrays, and
    /// the incomplete remainder stays in the bounded dense residual. Nothing is evaluated here, so
    /// the append joins the caller's lazy per-token graph. Returns the bytes written into device
    /// storage.
    pub fn append_device(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<u64> {
        self.append_device_replacing(layer, keys, values)
            .map(|(written, _)| written)
    }

    /// [`Self::append_device`], also returning the residual arrays a group flush replaced (the
    /// only state a whole-step rollback cannot rebuild from counts).
    fn append_device_replacing(
        &mut self,
        layer: usize,
        keys: &Array,
        values: &Array,
    ) -> Result<(u64, Option<Residuals>)> {
        if self.cancelled {
            return Err(Error::Canceled);
        }
        if self.layers.iter().any(Option::is_some) {
            return Err(Error::Config(
                "a host-staged packed cache cannot take device appends".into(),
            ));
        }
        if !packed_metal_cache_geometry_supported(self.head_dimension, self.group_size) {
            return Err(Error::Unsupported(format!(
                "device-resident packed KV requires head dimension 64, 128, or 256 and group size {PACKED_METAL_QUANT_GROUP_SIZE}"
            )));
        }
        if layer >= self.layers.len() {
            return Err(Error::Config("layer out of range".into()));
        }
        let key_shape = packed_input_shape(keys.shape(), "key")?;
        let step = key_shape[2];
        if step == 0
            || key_shape != [self.batch, self.kv_heads, step, self.head_dimension]
            || packed_input_shape(values.shape(), "value")? != key_shape
        {
            return Err(Error::Config("packed KV append shape mismatch".into()));
        }
        let float = |dtype| matches!(dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32);
        if !float(keys.dtype()) || !float(values.dtype()) {
            return Err(Error::Config(
                "packed KV append requires f16, bf16, or f32 K/V".into(),
            ));
        }
        let next_len = self
            .logical_len
            .checked_add(step)
            .ok_or_else(|| Error::Config("packed cache length overflow".into()))?;
        i32::try_from(next_len)
            .map_err(|_| Error::Config("packed cache length exceeds MLX i32 range".into()))?;
        if self.layer_tokens(layer) != self.logical_len {
            return Err(Error::Msg(
                "layer append is ahead of the atomic commit".into(),
            ));
        }
        let geometry = self.geometry();
        let group = self.group_size;
        let mut device = match self.device_layers[layer].take() {
            Some(device) if device.authoritative => device,
            _ => DevicePackedLayer::empty(
                geometry,
                geometry.block_tokens(),
                true,
                (group, keys.dtype()),
                (group, values.dtype()),
            )?,
        };
        let outcome = (|| -> Result<(u64, u64, Option<Residuals>)> {
            if device.key_tail.dtype() != keys.dtype()
                || device.value_tail.dtype() != values.dtype()
            {
                return Err(Error::Config(
                    "packed K/V dtype must remain stable for each resident layer".into(),
                ));
            }
            let residual = device.key_tail_rows;
            let total = residual + step;
            let groups = total / group;
            let remainder = total % group;
            let rows = geometry.rows();
            let (key_elem, value_elem) = (dtype_bytes(keys.dtype()), dtype_bytes(values.dtype()));
            if groups == 0 {
                write_rows(&mut device.key_tail, residual, keys)?;
                write_rows(&mut device.value_tail, residual, values)?;
                device.key_tail_rows = total;
                device.value_tail_rows = total;
                let written = rows * step * self.head_dimension * (key_elem + value_elem);
                return Ok((written as u64, 0, None));
            }
            let flushed = groups * group;
            let coexisting =
                device.ensure_packed_capacity(geometry, device.value_packed_tokens + flushed)?;
            let [key_codes, key_scales, key_zeros, value_codes, value_scales, value_zeros] =
                quantize_group_affine_flush(
                    &device.key_tail,
                    &device.value_tail,
                    residual,
                    keys,
                    values,
                    groups,
                    self.bits,
                )?;
            let replacement = if remainder > 0 {
                // The flush consumed the whole residual, so the remainder is the fresh suffix.
                let from = step - remainder;
                Some((
                    padded_residual(geometry, &rows_range(keys, from, step)?, remainder)?,
                    padded_residual(geometry, &rows_range(values, from, step)?, remainder)?,
                ))
            } else {
                None
            };
            let first_group = device.key_packed_tokens / group;
            write_rows(&mut device.key_codes, first_group, &key_codes)?;
            write_rows(&mut device.key_scales, first_group, &key_scales)?;
            write_rows(&mut device.key_zeros, first_group, &key_zeros)?;
            let first_row = device.value_packed_tokens;
            write_rows(&mut device.value_codes, first_row, &value_codes)?;
            write_rows(&mut device.value_scales, first_row, &value_scales)?;
            write_rows(&mut device.value_zeros, first_row, &value_zeros)?;
            let replaced = replacement.map(|(key_tail, value_tail)| {
                (
                    std::mem::replace(&mut device.key_tail, key_tail),
                    std::mem::replace(&mut device.value_tail, value_tail),
                )
            });
            device.key_packed_tokens += flushed;
            device.value_packed_tokens += flushed;
            device.key_tail_rows = remainder;
            device.value_tail_rows = remainder;
            let written = [
                &key_codes,
                &key_scales,
                &key_zeros,
                &value_codes,
                &value_scales,
                &value_zeros,
            ]
            .into_iter()
            .map(array_bytes)
            .fold(0, u64::saturating_add)
            .saturating_add(
                (rows * remainder * self.head_dimension * (key_elem + value_elem)) as u64,
            );
            Ok((written, coexisting, replaced))
        })();
        self.device_layers[layer] = Some(device);
        let (written, coexisting, replaced) = outcome?;
        self.telemetry.uploaded_packed_bytes =
            self.telemetry.uploaded_packed_bytes.saturating_add(written);
        self.update_retained_telemetry();
        if coexisting > 0 {
            // Block growth briefly holds the pre-growth arrays beside their successors.
            self.telemetry.peak_packed_transient_logical_bytes =
                self.telemetry.peak_packed_transient_logical_bytes.max(
                    self.retained_device_physical_bytes()
                        .saturating_add(coexisting),
                );
        }
        if self.device_layers.iter().all(|device| {
            device
                .as_ref()
                .is_some_and(|device| device.authoritative && device.kv_tokens() == next_len)
        }) {
            self.logical_len = next_len;
        }
        Ok((written, replaced))
    }

    /// Rollback points for every device-resident layer (decoder whole-step transactions).
    fn device_marks(&self) -> Vec<Option<DeviceLayerMark>> {
        self.device_layers
            .iter()
            .map(|device| {
                device
                    .as_ref()
                    .filter(|device| device.authoritative)
                    .map(DevicePackedLayer::mark)
            })
            .collect()
    }

    /// Evaluate every resident device layer's store (packed extents, metadata, and residuals), so
    /// the arrays the cache accounts are materialized and no lazy append retains its dense input.
    /// Evaluate one device layer's store together with `with` (the attention output that layer
    /// returns), so the step's dense K/V that feed them are released before the next layer.
    fn evaluate_device_layer(&self, layer: usize, with: &Array) -> Result<()> {
        let Some(device) = self.device_layers.get(layer).and_then(Option::as_ref) else {
            return Err(Error::Msg(format!(
                "packed layer {layer} has no device store"
            )));
        };
        mlx_rs::transforms::eval([
            with,
            &device.key_codes,
            &device.key_scales,
            &device.key_zeros,
            &device.key_tail,
            &device.value_codes,
            &device.value_scales,
            &device.value_zeros,
            &device.value_tail,
        ])?;
        Ok(())
    }

    fn evaluate_device_store(&self) -> Result<()> {
        mlx_rs::transforms::eval(self.device_layers.iter().flatten().flat_map(|device| {
            [
                &device.key_codes,
                &device.key_scales,
                &device.key_zeros,
                &device.key_tail,
                &device.value_codes,
                &device.value_scales,
                &device.value_zeros,
                &device.value_tail,
            ]
        }))?;
        Ok(())
    }

    /// Return every device-resident layer to `marks` and the logical length to `len`. Packed arrays
    /// are append-only past the marked counts, so nothing but counts and residuals is restored.
    fn restore_device_marks(&mut self, marks: Vec<Option<DeviceLayerMark>>, len: usize) {
        for (slot, mark) in self.device_layers.iter_mut().zip(marks) {
            match (slot.as_mut(), mark) {
                (Some(device), Some(mark)) => device.restore(mark),
                (_, None) => *slot = None,
                (None, Some(_)) => {}
            }
        }
        self.logical_len = len;
        self.update_retained_telemetry();
    }

    /// Bytes of residual arrays that `marks` keep alive beside their replacements for the first
    /// `layers` layers (rollback-only state).
    fn device_mark_overhead(marks: &[Option<DeviceLayerMark>], layers: usize) -> u64 {
        marks
            .iter()
            .take(layers)
            .flatten()
            .filter_map(|mark| mark.residuals.as_ref())
            .map(|(key_tail, value_tail)| {
                array_bytes(key_tail).saturating_add(array_bytes(value_tail))
            })
            .fold(0, u64::saturating_add)
    }

    /// Upload the host reference's deltas into the resident device mirror in place: only K groups
    /// and V rows past the mirror's extents, plus the bounded pending K group. Returns the bytes
    /// uploaded from the host.
    fn sync_host_mirror(&self, layer: usize, mirror: &mut DevicePackedLayer) -> Result<u64> {
        let storage = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Config("packed layer is not resident".into()))?;
        let geometry = self.geometry();
        let (rows, group, dim) = (geometry.rows(), self.group_size, self.head_dimension);
        let tokens = storage.keys.logical_tokens();
        let complete = storage.keys.complete_tokens;
        mirror.ensure_packed_capacity(geometry, tokens)?;
        let half = std::mem::size_of::<f16>();
        let mut uploaded = 0usize;
        let (old_groups, new_groups) = (mirror.key_packed_tokens / group, complete / group);
        if old_groups < new_groups {
            let n = new_groups - old_groups;
            let words = geometry.key_words();
            let codes = row_major_outer_rows(
                &storage.keys.codes[old_groups * rows * words..new_groups * rows * words],
                n,
                rows,
                words,
            )?;
            let metadata = old_groups * rows * dim..new_groups * rows * dim;
            let scales =
                row_major_outer_rows(&storage.keys.scales[metadata.clone()], n, rows, dim)?;
            let zeros = row_major_outer_rows(&storage.keys.zeros[metadata], n, rows, dim)?;
            let code_shape = geometry.shape(n, words)?;
            let metadata_shape = geometry.shape(n, dim)?;
            write_rows(
                &mut mirror.key_codes,
                old_groups,
                &Array::from_slice(&codes, &code_shape),
            )?;
            write_rows(
                &mut mirror.key_scales,
                old_groups,
                &Array::from_slice(&scales, &metadata_shape),
            )?;
            write_rows(
                &mut mirror.key_zeros,
                old_groups,
                &Array::from_slice(&zeros, &metadata_shape),
            )?;
            uploaded += n * rows * (words + dim * 2 * half);
        }
        let old_rows = mirror.value_packed_tokens;
        if old_rows < tokens {
            let n = tokens - old_rows;
            let (words, groups) = (geometry.value_words(), geometry.value_groups());
            let codes = row_major_outer_rows(
                &storage.values.codes[old_rows * rows * words..tokens * rows * words],
                n,
                rows,
                words,
            )?;
            let metadata = old_rows * rows * groups..tokens * rows * groups;
            let scales =
                row_major_outer_rows(&storage.values.scales[metadata.clone()], n, rows, groups)?;
            let zeros = row_major_outer_rows(&storage.values.zeros[metadata], n, rows, groups)?;
            let code_shape = geometry.shape(n, words)?;
            let metadata_shape = geometry.shape(n, groups)?;
            write_rows(
                &mut mirror.value_codes,
                old_rows,
                &Array::from_slice(&codes, &code_shape),
            )?;
            write_rows(
                &mut mirror.value_scales,
                old_rows,
                &Array::from_slice(&scales, &metadata_shape),
            )?;
            write_rows(
                &mut mirror.value_zeros,
                old_rows,
                &Array::from_slice(&zeros, &metadata_shape),
            )?;
            uploaded += n * rows * (words + groups * 2 * half);
        }
        let pending = storage.keys.pending_tokens;
        if pending > 0 {
            // The host pending group is `[token, row, channel]`; the residual is row-major.
            let mut tail = vec![0.0f32; rows * group * dim];
            for row in 0..rows {
                for token in 0..pending {
                    let source = (token * rows + row) * dim;
                    let target = (row * group + token) * dim;
                    tail[target..target + dim]
                        .copy_from_slice(&storage.keys.pending[source..source + dim]);
                }
            }
            mirror.key_tail = Array::from_slice(&tail, &geometry.shape(group, dim)?);
            uploaded += pending * rows * dim * std::mem::size_of::<f32>();
        }
        mirror.key_packed_tokens = complete;
        mirror.value_packed_tokens = tokens;
        mirror.key_tail_rows = pending;
        mirror.value_tail_rows = 0;
        Ok(uploaded as u64)
    }

    fn empty_host_mirror(&self) -> Result<DevicePackedLayer> {
        let geometry = self.geometry();
        DevicePackedLayer::empty(
            geometry,
            geometry.block_tokens(),
            false,
            (self.group_size, Dtype::Float32),
            (1, Dtype::Float32),
        )
    }

    /// Logical bytes of the live extents retained by the device layers.
    pub fn retained_device_packed_logical_bytes(&self) -> usize {
        let (codes, metadata) = self.retained_device_component_bytes();
        codes.saturating_add(metadata)
    }

    /// Live-extent `(codes, metadata + dense residual)` bytes retained by the device layers.
    pub fn retained_device_component_bytes(&self) -> (usize, usize) {
        let geometry = self.geometry();
        self.device_layers
            .iter()
            .flatten()
            .fold((0, 0), |(codes, metadata), device| {
                let (next_codes, next_metadata) = device.logical_component_bytes(geometry);
                (
                    codes.saturating_add(next_codes),
                    metadata.saturating_add(next_metadata),
                )
            })
    }

    /// Bytes of the device arrays actually allocated (block capacity and residual buffers).
    pub fn retained_device_physical_bytes(&self) -> u64 {
        self.device_layers
            .iter()
            .flatten()
            .map(|device| {
                let (codes, metadata) = device.physical_component_bytes();
                codes.saturating_add(metadata)
            })
            .fold(0, u64::saturating_add)
    }

    /// Dense values numerically equal to what the packed reader consumes for `layer`, row-major
    /// `[batch·head, token, channel]`. Used only by an observable dense transition and by oracles;
    /// successful packed dispatches never call it.
    pub(crate) fn evaluated_dense_layer(
        &self,
        layer: usize,
    ) -> Result<(usize, Vec<f32>, Vec<f32>)> {
        if let Some(storage) = self.layers.get(layer).and_then(Option::as_ref) {
            let tokens = storage.keys.logical_tokens();
            let rows = self.rows();
            let mut keys = Vec::with_capacity(rows * tokens * self.head_dimension);
            let mut values = Vec::with_capacity(rows * tokens * self.head_dimension);
            for row in 0..rows {
                for token in 0..tokens {
                    keys.extend(storage.keys.row(token, row)?);
                    values.extend(storage.values.row(token * rows + row, self.group_size)?);
                }
            }
            return Ok((tokens, keys, values));
        }
        let device = self
            .device_layers
            .get(layer)
            .and_then(Option::as_ref)
            .filter(|device| device.authoritative)
            .ok_or_else(|| Error::Config("packed layer is not resident".into()))?;
        let (keys, values) = device.dense_f32(self.geometry())?;
        let (keys, values) = (materialized(&keys)?, materialized(&values)?);
        keys.eval()?;
        values.eval()?;
        Ok((
            device.kv_tokens(),
            keys.as_slice::<f32>().to_vec(),
            values.as_slice::<f32>().to_vec(),
        ))
    }

    /// Dense `[B,H,T,D]` K/V arrays equal to what the packed reader consumes, cast to the layer's
    /// dtypes. Device-resident layers are reconstructed on the GPU without a host round trip.
    fn dense_layer_arrays(
        &self,
        layer: usize,
        key_dtype: Dtype,
        value_dtype: Dtype,
    ) -> Result<(Array, Array)> {
        if let Some(device) = self
            .device_layers
            .get(layer)
            .and_then(Option::as_ref)
            .filter(|device| device.authoritative)
        {
            let (keys, values) = device.dense_f32(self.geometry())?;
            return Ok((keys.as_dtype(key_dtype)?, values.as_dtype(value_dtype)?));
        }
        let (tokens, keys, values) = self.evaluated_dense_layer(layer)?;
        let shape = mlx_shape([self.batch, self.kv_heads, tokens, self.head_dimension])?;
        Ok((
            Array::from_slice(&keys, &shape).as_dtype(key_dtype)?,
            Array::from_slice(&values, &shape).as_dtype(value_dtype)?,
        ))
    }

    fn record_dispatch_evidence(&mut self, uploaded: u64, argument: u64, transient: u64) {
        self.telemetry.uploaded_packed_bytes = self
            .telemetry
            .uploaded_packed_bytes
            .saturating_add(uploaded);
        self.telemetry.peak_packed_argument_logical_bytes = self
            .telemetry
            .peak_packed_argument_logical_bytes
            .max(argument);
        self.telemetry.peak_packed_transient_logical_bytes = self
            .telemetry
            .peak_packed_transient_logical_bytes
            .max(transient);
    }

    /// The reader state `dispatch_packed` would dispatch `layer` against — a host-reference cache's
    /// mirror with its deltas and bounded pending K group uploaded, or a device-resident layer
    /// as is — evaluated but not published (no device state or telemetry changes). SC-20677 times
    /// the retained reader on it separately from the per-dispatch mirror sync.
    pub(crate) fn staged_reader_arguments(&self, layer: usize) -> Result<StagedReaderLayer> {
        let previous = self.device_layers.get(layer).and_then(Option::as_ref);
        let device = if self.layers.get(layer).and_then(Option::as_ref).is_some() {
            let mut mirror = match previous {
                Some(mirror) if !mirror.authoritative => mirror.clone(),
                _ => self.empty_host_mirror()?,
            };
            self.sync_host_mirror(layer, &mut mirror)?;
            mirror
        } else {
            previous
                .cloned()
                .ok_or_else(|| Error::Config("packed layer is not resident".into()))?
        };
        for array in [
            &device.key_codes,
            &device.key_scales,
            &device.key_zeros,
            &device.key_tail,
            &device.value_codes,
            &device.value_scales,
            &device.value_zeros,
            &device.value_tail,
        ] {
            array.eval()?;
        }
        Ok(StagedReaderLayer(device))
    }

    /// Internal source-test seam: upload the host reference into its device mirror and publish it,
    /// returning the six live packed extents the reader would receive.
    #[cfg(test)]
    fn packed_mlx_arguments(
        &mut self,
        layer: usize,
    ) -> Result<(Array, Array, Array, Array, Array, Array)> {
        let mut mirror = match self.device_layers[layer].take() {
            Some(mirror) if !mirror.authoritative => mirror,
            _ => self.empty_host_mirror()?,
        };
        let uploaded = self.sync_host_mirror(layer, &mut mirror)?;
        let (codes, metadata) = mirror.logical_component_bytes(self.geometry());
        let groups = mirror.key_packed_tokens / self.group_size;
        let tokens = mirror.value_packed_tokens;
        let arguments = (
            materialized(&live_rows(&mirror.key_codes, groups)?)?,
            materialized(&live_rows(&mirror.key_scales, groups)?)?,
            materialized(&live_rows(&mirror.key_zeros, groups)?)?,
            materialized(&live_rows(&mirror.value_codes, tokens)?)?,
            materialized(&live_rows(&mirror.value_scales, tokens)?)?,
            materialized(&live_rows(&mirror.value_zeros, tokens)?)?,
        );
        self.device_layers[layer] = Some(mirror);
        self.update_retained_telemetry();
        let argument = (codes + metadata) as u64;
        self.record_dispatch_evidence(
            uploaded,
            argument,
            self.retained_device_packed_logical_bytes() as u64 + argument,
        );
        Ok(arguments)
    }

    /// Run the retained reader on one layer and evaluate its output. A host-reference cache first
    /// uploads its deltas into the resident device mirror; the mirror's extents advance only after
    /// the output evaluates, so a failed attempt leaves the prior mirror intact.
    pub fn dispatch_packed(
        &mut self,
        layer: usize,
        query: &Array,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> Result<Array> {
        self.dispatch_layer(layer, query, mask, 0, true)
    }

    /// Shared dispatch. `evaluate = false` leaves a steady (already-compiled) dispatch lazy so it
    /// joins the caller's per-token evaluation; a cold dispatch is always evaluated here so a JIT
    /// fault surfaces inside the cache's transaction.
    fn dispatch_layer(
        &mut self,
        layer: usize,
        query: &Array,
        mask: crate::primitives::packed_metal::PackedMask,
        additional_transient_bytes: u64,
        evaluate: bool,
    ) -> Result<Array> {
        let handle = self
            .handle
            .clone()
            .ok_or_else(|| Error::Unsupported("no retained packed reader".into()))?;
        if layer >= self.layers.len() || self.layer_tokens(layer) == 0 {
            return Err(Error::Config(
                "cannot dispatch an empty packed cache".into(),
            ));
        }
        let dense_dequantizations_before = self.full_cache_dequantizations;
        let started = std::time::Instant::now();
        self.telemetry.dispatch_attempts += 1;
        let host = self.layers[layer].is_some();
        let previous = self.device_layers[layer].take();
        let (device, mark, uploaded) = if host {
            let existed = previous
                .as_ref()
                .is_some_and(|mirror| !mirror.authoritative);
            let mut mirror = match previous {
                Some(mirror) if !mirror.authoritative => mirror,
                _ => match self.empty_host_mirror() {
                    Ok(mirror) => mirror,
                    Err(error) => {
                        self.telemetry.failed_dispatches += 1;
                        return Err(error);
                    }
                },
            };
            let mark = existed.then(|| mirror.mark_with_residuals());
            match self.sync_host_mirror(layer, &mut mirror) {
                Ok(uploaded) => (mirror, mark, uploaded),
                Err(error) => {
                    if let Some(mark) = mark {
                        mirror.restore(mark);
                        self.device_layers[layer] = Some(mirror);
                    }
                    self.telemetry.failed_dispatches += 1;
                    self.telemetry.attempted_elapsed_ms += started.elapsed().as_secs_f64() * 1000.0;
                    return Err(error);
                }
            }
        } else {
            let device =
                previous.ok_or_else(|| Error::Config("packed layer is not resident".into()))?;
            (device, None, 0)
        };
        let (codes, metadata) = device.logical_component_bytes(self.geometry());
        let argument = (codes + metadata) as u64;
        let others = self.retained_device_packed_logical_bytes() as u64;
        self.record_dispatch_evidence(
            uploaded,
            argument,
            others
                .saturating_add(argument)
                .saturating_add(additional_transient_bytes),
        );
        // Compilation belongs to the retained handle, not to any one cache that borrows it. A new
        // cache over an already-exercised handle must therefore classify its first dispatch as
        // steady rather than inventing another cold/JIT event.
        let was_warmed = handle.is_warmed();
        self.telemetry.kernel_warmed = was_warmed;
        if !was_warmed {
            self.telemetry.compile_jit_attempts += 1;
        }
        let args = device.args(query, mask);
        let selection = handle.inner.kernel_selection(&args);
        let result = handle.inner.dispatch(&args).and_then(|output| {
            if evaluate || !was_warmed {
                output.eval()?;
            }
            Ok(output)
        });
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.attempted_elapsed_ms += elapsed_ms;
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.telemetry.failed_dispatches += 1;
                let mut device = device;
                match (host, mark) {
                    (true, Some(mark)) => {
                        device.restore(mark);
                        self.device_layers[layer] = Some(device);
                    }
                    (true, None) => {}
                    (false, _) => self.device_layers[layer] = Some(device),
                }
                self.update_retained_telemetry();
                return Err(error);
            }
        };
        handle.mark_warmed();
        self.telemetry.kernel_warmed = true;
        self.device_layers[layer] = Some(device);
        self.update_retained_telemetry();
        debug_assert_eq!(
            self.full_cache_dequantizations, dense_dequantizations_before,
            "accepted fused packed dispatch must not reconstruct the full cache"
        );
        self.direct_dispatches += 1;
        if let Some(selection) = selection {
            match self
                .kernel_paths
                .iter_mut()
                .find(|(recorded, _)| *recorded == selection)
            {
                Some((_, calls)) => *calls += 1,
                None => self.kernel_paths.push((selection, 1)),
            }
        }
        self.telemetry.accepted_uploaded_packed_bytes = self
            .telemetry
            .accepted_uploaded_packed_bytes
            .saturating_add(uploaded);
        if was_warmed {
            self.telemetry.steady_dispatches += 1;
            self.telemetry.steady_elapsed_ms += elapsed_ms;
        } else {
            self.telemetry.cold_dispatches += 1;
            self.telemetry.cold_elapsed_ms = elapsed_ms;
        }
        Ok(output)
    }

    pub fn direct_dispatches(&self) -> usize {
        self.direct_dispatches
    }
    /// Accepted calls per kernel path the reader reported, in first-use order.
    pub fn kernel_paths(&self) -> &[(PackedKernelSelection, u64)] {
        &self.kernel_paths
    }
    pub fn full_cache_dequantizations(&self) -> usize {
        self.full_cache_dequantizations
    }
    pub fn record_dense_dequantization(&mut self) {
        self.full_cache_dequantizations += 1;
    }
    pub fn dispatch_telemetry(&self) -> PackedDispatchTelemetry {
        self.telemetry
    }
    fn accepted_dispatch_snapshot(&self) -> AcceptedDispatchSnapshot {
        AcceptedDispatchSnapshot {
            direct_dispatches: self.direct_dispatches,
            kernel_paths: self.kernel_paths.clone(),
            cold_dispatches: self.telemetry.cold_dispatches,
            steady_dispatches: self.telemetry.steady_dispatches,
            cold_elapsed_ms: self.telemetry.cold_elapsed_ms,
            steady_elapsed_ms: self.telemetry.steady_elapsed_ms,
            accepted_uploaded_packed_bytes: self.telemetry.accepted_uploaded_packed_bytes,
        }
    }
    fn restore_accepted_dispatch(&mut self, snapshot: AcceptedDispatchSnapshot) {
        self.direct_dispatches = snapshot.direct_dispatches;
        self.kernel_paths = snapshot.kernel_paths;
        self.telemetry.cold_dispatches = snapshot.cold_dispatches;
        self.telemetry.steady_dispatches = snapshot.steady_dispatches;
        self.telemetry.cold_elapsed_ms = snapshot.cold_elapsed_ms;
        self.telemetry.steady_elapsed_ms = snapshot.steady_elapsed_ms;
        self.telemetry.accepted_uploaded_packed_bytes = snapshot.accepted_uploaded_packed_bytes;
    }
    pub fn preflight(
        &mut self,
        backend: &str,
        query_length: usize,
        mask: bool,
    ) -> crate::primitives::CacheRoute {
        if backend != "mlx-metal" || query_length == 0 || mask || self.handle.is_none() {
            let reason = if backend != "mlx-metal" {
                "unsupported backend"
            } else if query_length == 0 {
                "empty query"
            } else if mask {
                "mask requires dense fallback"
            } else {
                "no retained compiled packed reader"
            };
            self.dense_read_fallback("preflight", reason);
            crate::primitives::CacheRoute::DenseFallback {
                reason: reason.into(),
            }
        } else {
            crate::primitives::CacheRoute::ExperimentalPacked
        }
    }

    /// Capability-aware preflight used by the decoder seam.  Causal and bounded sliding masks
    /// are handled by the retained reader; arbitrary additive masks remain a dense fallback and
    /// are reported before any cache mutation.
    pub fn preflight_mask(
        &mut self,
        backend: &str,
        query_length: usize,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> crate::primitives::CacheRoute {
        if matches!(
            mask,
            crate::primitives::packed_metal::PackedMask::AdditiveUnsupported
        ) {
            self.dense_read_fallback("preflight", "additive mask requires dense fallback");
            return crate::primitives::CacheRoute::DenseFallback {
                reason: "additive mask requires dense fallback".into(),
            };
        }
        self.preflight(backend, query_length, false)
    }

    /// Versioned, identity-bound snapshot. Restore is all-or-nothing and rejects mismatched shape,
    /// quantization, or identity before installing any state. A device-resident decoder cache is
    /// refused rather than silently re-quantized: this format has no dense value residual.
    pub fn save(&self) -> Result<Vec<u8>> {
        if self.device_resident() {
            return Err(Error::Unsupported(
                "device-resident packed caches carry a dense value residual the SC-20675 snapshot format cannot represent".into(),
            ));
        }
        let mut out = MAGIC.to_vec();
        out.extend(VERSION.to_le_bytes());
        out.extend((self.identity.len() as u32).to_le_bytes());
        out.extend(self.identity.as_bytes());
        for n in [
            self.group_size,
            self.batch,
            self.kv_heads,
            self.head_dimension,
            self.capacity,
            self.logical_len,
            self.absolute_offset,
            self.layers.len(),
        ] {
            out.extend((n as u64).to_le_bytes());
        }
        out.push(self.bits.bits());
        for layer in &self.layers {
            out.push(layer.is_some() as u8);
            if let Some(layer) = layer {
                let keys = &layer.keys;
                out.extend((keys.complete_tokens as u64).to_le_bytes());
                out.extend((keys.pending_tokens as u64).to_le_bytes());
                out.extend((keys.codes.len() as u64).to_le_bytes());
                out.extend(&keys.codes);
                for values in [&keys.scales, &keys.zeros] {
                    out.extend((values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend(value.to_bits().to_le_bytes());
                    }
                }
                out.extend((keys.pending.len() as u64).to_le_bytes());
                for value in &keys.pending {
                    out.extend(value.to_bits().to_le_bytes());
                }

                let values = &layer.values;
                out.extend((values.rows as u64).to_le_bytes());
                out.extend((values.codes.len() as u64).to_le_bytes());
                out.extend(&values.codes);
                for metadata in [&values.scales, &values.zeros] {
                    out.extend((metadata.len() as u64).to_le_bytes());
                    for value in metadata {
                        out.extend(value.to_bits().to_le_bytes());
                    }
                }
            }
        }
        out.extend(checksum(&out).to_le_bytes());
        Ok(out)
    }

    pub fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 8
            || u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap())
                != checksum(&bytes[..bytes.len() - 8])
        {
            return Err(Error::Config("snapshot checksum mismatch".into()));
        }
        let bytes = &bytes[..bytes.len() - 8];
        let mut p = 0;
        let take = |p: &mut usize, n: usize| -> Result<&[u8]> {
            let end = p
                .checked_add(n)
                .ok_or_else(|| Error::Config("snapshot overflow".into()))?;
            let s = bytes
                .get(*p..end)
                .ok_or_else(|| Error::Config("truncated snapshot".into()))?;
            *p = end;
            Ok(s)
        };
        if take(&mut p, 8)? != MAGIC {
            return Err(Error::Config("snapshot magic mismatch".into()));
        }
        if u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) != VERSION {
            return Err(Error::Config("snapshot version mismatch".into()));
        }
        let id_len = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        if take(&mut p, id_len)? != self.identity.as_bytes() {
            return Err(Error::Config("snapshot identity mismatch".into()));
        }
        let mut nums = [0u64; 8];
        for n in &mut nums {
            *n = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
        }
        if nums.iter().any(|&value| value > usize::MAX as u64)
            || nums[0] as usize != self.group_size
            || nums[1] as usize != self.batch
            || nums[2] as usize != self.kv_heads
            || nums[3] as usize != self.head_dimension
            || nums[7] as usize != self.layers.len()
            || take(&mut p, 1)?[0] != self.bits.bits()
        {
            return Err(Error::Config(
                "snapshot quantization or shape mismatch".into(),
            ));
        }
        // The capacity is a reservation the cache's own growth produced: zero, or a power of two
        // (`grow` doubles from one) holding at least the declared logical length. Anything else is
        // refused. A cache trimmed after growing legitimately carries more than its length needs,
        // so the reservation restored is bounded by what growth produces for the restored length
        // — `grow` from empty yields the next power of two at or above it, and at least two —
        // never the header's own figure (sc-20688 review: a crafted capacity must not reach an
        // unbounded or overcommitted reservation).
        let declared = nums[4] as usize;
        if nums[5] > nums[4] || (declared != 0 && !declared.is_power_of_two()) {
            return Err(Error::Config(
                "snapshot capacity/logical bounds mismatch".into(),
            ));
        }
        let capacity = match nums[5] as usize {
            0 => 0,
            len => declared.min(len.next_power_of_two().max(2)),
        };
        let mut restored = Vec::with_capacity(self.layers.len());
        let rows = self.rows();
        for _ in 0..self.layers.len() {
            let present_byte = take(&mut p, 1)?[0];
            if present_byte > 1 {
                return Err(Error::Config(
                    "snapshot layer presence flag mismatch".into(),
                ));
            }
            let present = present_byte != 0;
            if !present {
                if nums[5] != 0 {
                    return Err(Error::Config("snapshot omits a resident layer".into()));
                }
                restored.push(None);
                continue;
            }
            let complete_tokens = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let pending_tokens = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            if complete_tokens > usize::MAX as u64
                || pending_tokens > usize::MAX as u64
                || !(complete_tokens as usize).is_multiple_of(self.group_size)
                || complete_tokens.checked_add(pending_tokens) != Some(nums[5])
                || pending_tokens as usize >= self.group_size
            {
                return Err(Error::Config(
                    "snapshot key token-group shape mismatch".into(),
                ));
            }
            let complete_tokens = complete_tokens as usize;
            let pending_tokens = pending_tokens as usize;
            let key_groups = complete_tokens / self.group_size;
            let key_code_len = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_key_codes = key_groups
                .checked_mul(rows)
                .and_then(|value| {
                    value.checked_mul(self.bits.code_bytes(self.group_size * self.head_dimension))
                })
                .ok_or_else(|| Error::Config("snapshot key code length overflow".into()))?;
            if key_code_len != expected_key_codes as u64 {
                return Err(Error::Config("snapshot key code shape mismatch".into()));
            }
            let key_codes = take(&mut p, expected_key_codes)?.to_vec();
            let expected_key_metadata = key_groups
                .checked_mul(rows)
                .and_then(|value| value.checked_mul(self.head_dimension))
                .ok_or_else(|| Error::Config("snapshot key metadata overflow".into()))?;
            let mut key_metadata = Vec::new();
            for _ in 0..2 {
                let count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
                if count != expected_key_metadata as u64 {
                    return Err(Error::Config(
                        "snapshot key scale/zero count mismatch".into(),
                    ));
                }
                let mut values = Vec::with_capacity(expected_key_metadata);
                for _ in 0..expected_key_metadata {
                    values.push(f16::from_bits(u16::from_le_bytes(
                        take(&mut p, 2)?.try_into().unwrap(),
                    )));
                }
                key_metadata.push(values);
            }
            let pending_count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_pending = pending_tokens
                .checked_mul(rows)
                .and_then(|value| value.checked_mul(self.head_dimension))
                .ok_or_else(|| Error::Config("snapshot key pending overflow".into()))?;
            if pending_count != expected_pending as u64 {
                return Err(Error::Config("snapshot key pending count mismatch".into()));
            }
            let mut pending = Vec::with_capacity(expected_pending);
            for _ in 0..expected_pending {
                pending.push(f32::from_bits(u32::from_le_bytes(
                    take(&mut p, 4)?.try_into().unwrap(),
                )));
            }

            let value_rows = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_rows = (nums[5] as usize)
                .checked_mul(rows)
                .ok_or_else(|| Error::Config("snapshot value row overflow".into()))?;
            let value_code_len = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
            let expected_value_codes = expected_rows
                .checked_mul(self.bits.code_bytes(self.head_dimension))
                .ok_or_else(|| Error::Config("snapshot value code overflow".into()))?;
            if value_rows != expected_rows as u64 || value_code_len != expected_value_codes as u64 {
                return Err(Error::Config("snapshot value code shape mismatch".into()));
            }
            let value_codes = take(&mut p, expected_value_codes)?.to_vec();
            let expected_value_metadata = expected_rows
                .checked_mul(self.head_dimension.div_ceil(self.group_size))
                .ok_or_else(|| Error::Config("snapshot value metadata overflow".into()))?;
            let mut value_metadata = Vec::new();
            for _ in 0..2 {
                let count = u64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
                if count != expected_value_metadata as u64 {
                    return Err(Error::Config(
                        "snapshot value scale/zero count mismatch".into(),
                    ));
                }
                let mut values = Vec::with_capacity(expected_value_metadata);
                for _ in 0..expected_value_metadata {
                    values.push(f16::from_bits(u16::from_le_bytes(
                        take(&mut p, 2)?.try_into().unwrap(),
                    )));
                }
                value_metadata.push(values);
            }
            restored.push(Some(LayerStorage {
                keys: TokenGroupKeyTensor {
                    bits: self.bits,
                    rows,
                    width: self.head_dimension,
                    group_size: self.group_size,
                    complete_tokens,
                    pending_tokens,
                    codes: key_codes,
                    scales: key_metadata.remove(0),
                    zeros: key_metadata.remove(0),
                    pending,
                },
                values: PackedTensor {
                    bits: self.bits,
                    rows: expected_rows,
                    width: self.head_dimension,
                    groups: self.head_dimension.div_ceil(self.group_size),
                    codes: value_codes,
                    scales: value_metadata.remove(0),
                    zeros: value_metadata.remove(0),
                },
            }));
        }
        if p != bytes.len() {
            return Err(Error::Config("snapshot trailing bytes".into()));
        }
        // Reserve the declared capacity before installing anything: a capacity no allocation can
        // hold is an `Err`, and the cache is left as it was.
        let capacity_rows = rows
            .checked_mul(capacity)
            .ok_or_else(|| Error::Config("snapshot capacity cannot be reserved".into()))?;
        for layer in restored.iter_mut().flatten() {
            layer.keys.try_reserve_tokens(capacity)?;
            layer.values.try_reserve_rows(capacity_rows)?;
        }
        self.capacity = capacity;
        self.logical_len = nums[5] as usize;
        self.absolute_offset = nums[6] as usize;
        self.layers = restored;
        self.device_layers = vec![None; self.layers.len()];
        self.telemetry.retained_device_packed_logical_bytes = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
