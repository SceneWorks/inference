//! Physically packed group-affine KV representation (SC-20675) and its SC-20676 decoder route.
//!
//! Codes are [`PackedCodeBits`] wide (2-bit: four per byte, the SC-20673 qualified candidate;
//! 4-bit: two per byte, the KIVI-style candidate) and each group has an f16 scale and zero. The
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
};
use half::f16;
use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
use mlx_rs::ops::{add, concatenate_axis, floor_divide, multiply, remainder, zeros_dtype};
use mlx_rs::{Array, Dtype};

const MAGIC: &[u8; 8] = b"SW20675\0";
const VERSION: u32 = 2;
/// SC-20673 qualified the `b=2,g=32` group-affine candidate; `b=4,g=32` shares the group. K groups
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
}

impl PackedCodeBits {
    pub const ALL: [Self; 2] = [Self::Two, Self::Four];

    pub const fn bits(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
        }
    }

    /// Supported widths only; any other value is refused.
    pub fn from_bits(bits: u8) -> Result<Self> {
        match bits {
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            other => Err(Error::Config(format!(
                "packed group-affine code width {other} is unsupported (expected 2 or 4)"
            ))),
        }
    }

    /// Independent codes packed into one byte. A packing property, not the affine group.
    pub const fn codes_per_byte(self) -> usize {
        8 / self.bits() as usize
    }

    /// Largest code (`2^bits − 1`).
    pub const fn max_code(self) -> u8 {
        (1u8 << self.bits()) - 1
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

    /// CPU reference quantization of one group's value: `clamp(round((x − min) / scale), 0,
    /// 2^bits − 1)`.
    fn quantize(self, value: f32, min: f32, scale: f32) -> u8 {
        ((value - min) / scale).round().clamp(0.0, self.levels()) as u8
    }

    /// CPU reference group scale.
    fn scale(self, min: f32, max: f32) -> f32 {
        ((max - min) / self.levels()).max(f32::EPSILON)
    }
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
fn packed_input_shape(shape: &[i32], tensor: &str) -> Result<[usize; 4]> {
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
}

impl CompiledKernelHandle {
    pub fn new(inner: Arc<dyn RetainedPackedKernel>) -> Self {
        Self {
            inner,
            warmed: Arc::new(AtomicBool::new(false)),
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

    fn is_warmed(&self) -> bool {
        self.warmed.load(Ordering::Acquire)
    }

    fn mark_warmed(&self) {
        self.warmed.store(true, Ordering::Release);
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
}

/// Decoder-facing owner for the device-resident packed store and the retained fused reader. A model
/// step is a whole-step transaction across layers: each layer appends lazily into the device store
/// and dispatches the reader lazily (only a cold dispatch is evaluated, to surface a JIT fault
/// inside the transaction); a failure rolls every layer back to its marked extents. Unsupported
/// semantics transition the exact evaluated history to dense before the caller's ordinary update.
#[derive(Debug)]
pub struct DenseFallbackPackedDecoderCache {
    dense: ContiguousKvCache,
    staged: PackedGroupAffineKvCache,
    pending_step: Option<PendingPackedStep>,
    packed_layer_dtypes: Vec<Option<(mlx_rs::Dtype, mlx_rs::Dtype)>>,
    dense_active: bool,
    reason: String,
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
        }
    }

    /// Attach only a reader whose identity matches the staged representation.
    pub fn bind_compiled_handle(&mut self, handle: CompiledKernelHandle) -> Result<()> {
        self.staged.bind_compiled_handle(handle)
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
        let mut dense = ContiguousKvCache::new(self.staged.layers());
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
                });
            }
            let (written, replaced) = self.staged.append_device_replacing(layer, keys, values)?;
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
                crate::primitives::attention::sdpa(query, keys, values, scale, dense_mask)?
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
                Ok(bytes) => written = written.saturating_add(bytes),
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
            return self.dense.truncate(len);
        }
        if self.pending_step.is_some() {
            self.rollback_pending("truncate", "discarded pending packed step")?;
        }
        let len = usize::try_from(len).map_err(|_| {
            Error::Config("packed cache truncate length must not be negative".into())
        })?;
        self.staged.trim(len)?;
        if len == 0 {
            self.packed_layer_dtypes.fill(None);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.dense.reset()?;
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
            let scale = bits.scale(min, max);
            self.scales.push(f16::from_f32(scale));
            self.zeros.push(f16::from_f32(min));
            for (i, value) in slice.iter().copied().enumerate() {
                bits.pack(
                    &mut self.codes[code_start..],
                    start + i,
                    bits.quantize(value, min, scale),
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
                let scale = bits.scale(min, max);
                self.scales.push(f16::from_f32(scale));
                self.zeros.push(f16::from_f32(min));
                for token in 0..self.group_size {
                    let value = self.pending[(token * self.rows + row) * self.width + channel];
                    bits.pack(
                        &mut self.codes[code_start + row * code_bytes_per_group..],
                        token * self.width + channel,
                        bits.quantize(value, min, scale),
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
struct PackedGeometry {
    batch: usize,
    heads: usize,
    dim: usize,
    group: usize,
    bits: PackedCodeBits,
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
    fn shape(self, tokens: usize, width: usize) -> Result<[i32; 4]> {
        mlx_shape([self.batch, self.heads, tokens, width])
    }
}

fn array_bytes(array: &Array) -> u64 {
    // `usize` always fits in `u64` on supported targets.
    (array.size() as u64).saturating_mul(array.item_size() as u64)
}

fn dtype_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Float32 => 4,
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        _ => 4,
    }
}

/// Write `delta` into `array[.., .., start..start + delta_len, ..]` in place. MLX's slice update
/// donates the buffer when the cache holds the only reference, so no history is copied.
fn write_rows(array: &mut Array, start: usize, delta: &Array) -> Result<()> {
    let start = i32::try_from(start)
        .map_err(|_| Error::Config("packed KV row offset exceeds MLX i32 range".into()))?;
    let end = start
        .checked_add(delta.shape()[2])
        .ok_or_else(|| Error::Config("packed KV row end exceeds MLX i32 range".into()))?;
    array.try_index_mut((.., .., start..end, ..), delta)?;
    Ok(())
}

fn rows_range(array: &Array, start: usize, end: usize) -> Result<Array> {
    let bound = |value: usize| {
        i32::try_from(value)
            .map_err(|_| Error::Config("packed KV row range exceeds MLX i32 range".into()))
    };
    Ok(array.try_index((.., .., bound(start)?..bound(end)?, ..))?)
}

fn live_rows(array: &Array, rows: usize) -> Result<Array> {
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
fn dequantize_key_groups(
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
fn dequantize_value_rows(
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
fn padded_residual(geometry: PackedGeometry, rows_array: &Array, rows: usize) -> Result<Array> {
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
        if nums[5] > nums[4] {
            return Err(Error::Config(
                "snapshot capacity/logical bounds mismatch".into(),
            ));
        }
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
        self.capacity = nums[4] as usize;
        self.logical_len = nums[5] as usize;
        self.absolute_offset = nums[6] as usize;
        self.layers = restored;
        self.device_layers = vec![None; self.layers.len()];
        self.telemetry.retained_device_packed_logical_bytes = 0;
        self.reserve_storage_for_capacity();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn rejects_mlx_dimension_overflow_before_storage_allocation() {
        let oversized = usize::try_from(i32::MAX).unwrap() + 1;
        let error = PackedGroupAffineKvCache::new("overflow", 1, oversized, 1, 64, 32).unwrap_err();
        assert!(error.to_string().contains("exceeds MLX i32 range"));
    }

    #[derive(Debug)]
    struct TestPackedKernel {
        calls: AtomicUsize,
        fail_on: Option<usize>,
        cancel_on_failure: bool,
    }

    impl RetainedPackedKernel for TestPackedKernel {
        fn cache_identity(&self) -> &str {
            "test-packed"
        }
        fn backend(&self) -> &str {
            "mlx-metal"
        }
        fn retained_host_bytes_estimate(&self) -> usize {
            0
        }
        fn dispatch(&self, args: &PackedAttentionArgs<'_>) -> Result<Array> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_on == Some(call) {
                if self.cancel_on_failure {
                    return Err(Error::Canceled);
                }
                return Err(Error::Msg("injected packed dispatch fault".into()));
            }
            Ok(args.query.clone())
        }
        fn kernel_selection(
            &self,
            _args: &PackedAttentionArgs<'_>,
        ) -> Option<PackedKernelSelection> {
            Some(PackedKernelSelection {
                kernel: crate::primitives::PACKED_PER_ROW_KERNEL,
                selection: crate::primitives::PACKED_SELECTION_BELOW_MULTI_ROW,
                reason: "test reader",
                query_dtype: "float32",
            })
        }
    }

    fn hook_cache_geometry(
        layers: usize,
        batch: usize,
        kv_heads: usize,
        head_dimension: usize,
        fail_on: Option<usize>,
    ) -> (Box<dyn KvCache>, Arc<TestPackedKernel>) {
        let kernel = Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on,
            cancel_on_failure: false,
        });
        let selection = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "test-packed".into(),
                layers,
                batch,
                kv_heads,
                head_dimension,
                group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                bits: PackedCodeBits::Two,
                query_length: 1,
                has_mask: false,
            },
            CompiledKernelHandle::new(kernel.clone()),
        );
        (selection.into_cache(), kernel)
    }

    fn hook_cache(layers: usize, fail_on: Option<usize>) -> Box<dyn KvCache> {
        hook_cache_geometry(layers, 1, 1, 64, fail_on).0
    }
    fn data(step: usize, rows: usize, width: usize, bias: f32) -> Vec<f32> {
        (0..step * rows * width)
            .map(|i| bias + i as f32 * 0.25)
            .collect()
    }

    fn bhst_data(batch: usize, heads: usize, step: usize, width: usize, bias: f32) -> Vec<f32> {
        (0..batch * heads)
            .flat_map(|row| {
                (0..step).flat_map(move |token| {
                    (0..width).map(move |channel| {
                        bias + (row * 10_000 + token * 100 + channel) as f32 * 0.03125
                    })
                })
            })
            .collect()
    }

    fn pseudo_random_outliers(
        batch: usize,
        heads: usize,
        step: usize,
        width: usize,
        seed: u64,
    ) -> Vec<f32> {
        let mut state = seed;
        (0..batch * heads * step * width)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let base = ((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 20) as f32;
                match index % 29 {
                    0 => -10_000.0,
                    1 => 10_000.0,
                    2 => 0.0,
                    _ => base,
                }
            })
            .collect()
    }

    fn token_range(
        source: &[f32],
        batch: usize,
        heads: usize,
        full_step: usize,
        width: usize,
        start: usize,
        len: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(batch * heads * len * width);
        for row in 0..batch * heads {
            for token in start..start + len {
                let offset = (row * full_step + token) * width;
                out.extend_from_slice(&source[offset..offset + width]);
            }
        }
        out
    }

    fn assert_same_rows(left: &PackedGroupAffineKvCache, right: &PackedGroupAffineKvCache) {
        assert_eq!(left.logical_len(), right.logical_len());
        for token in 0..left.logical_len() {
            for row in 0..left.rows() {
                assert_eq!(
                    left.read_row(0, token, row).unwrap(),
                    right.read_row(0, token, row).unwrap()
                );
            }
        }
    }

    fn assert_same_device_layer(
        actual: &Option<DevicePackedLayer>,
        expected: &Option<DevicePackedLayer>,
    ) {
        let (Some(actual), Some(expected)) = (actual, expected) else {
            assert_eq!(actual.is_some(), expected.is_some());
            return;
        };
        let counts = |device: &DevicePackedLayer| {
            (
                device.key_packed_tokens,
                device.value_packed_tokens,
                device.key_tail_rows,
                device.value_tail_rows,
            )
        };
        assert_eq!(counts(actual), counts(expected));
        // Only the live extents are state; block padding past them is not.
        let live = |device: &DevicePackedLayer| {
            let groups = device.key_packed_tokens / PACKED_METAL_QUANT_GROUP_SIZE;
            let tokens = device.value_packed_tokens;
            [
                live_rows(&device.key_codes, groups),
                live_rows(&device.key_scales, groups),
                live_rows(&device.key_zeros, groups),
                live_rows(&device.value_codes, tokens),
                live_rows(&device.value_scales, tokens),
                live_rows(&device.value_zeros, tokens),
                live_rows(&device.key_tail, device.key_tail_rows),
                live_rows(&device.value_tail, device.value_tail_rows),
            ]
            .map(|array| {
                let array = array.unwrap().as_dtype(Dtype::Float32).unwrap();
                if array.size() == 0 {
                    return Vec::new();
                }
                array.eval().unwrap();
                array.as_slice::<f32>().to_vec()
            })
        };
        assert_eq!(live(actual), live(expected));
    }
    #[test]
    fn packed_append_is_quantized_and_non_mirroring() {
        let mut c = PackedGroupAffineKvCache::new("model", 1, 1, 2, 7, 4).unwrap();
        let k = data(3, 2, 7, -2.0);
        c.append(0, &k, &k, 3).unwrap();
        assert_eq!(c.logical_len(), 3);
        assert!(c.allocated_bytes() > 0);
        assert!(c.no_dense_mirror());
        assert_eq!(c.representation().bits, 2);
    }
    #[test]
    fn chunking_offsets_and_rollback() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 5, 4).unwrap();
        c.set_absolute_offset(19);
        let a = data(2, 1, 5, 0.0);
        c.append(0, &a, &a, 2).unwrap();
        let b = data(3, 1, 5, 9.0);
        c.append(0, &b, &b, 3).unwrap();
        assert_eq!(
            (c.logical_len(), c.allocated_len(), c.absolute_offset()),
            (5, 8, 19)
        );
        c.rollback(2).unwrap();
        assert_eq!(c.logical_len(), 2);
    }
    #[test]
    fn snapshot_rejects_identity_and_round_trips() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        let x = data(1, 1, 8, 1.0);
        c.append(0, &x, &x, 1).unwrap();
        c.packed_mlx_arguments(0).unwrap();
        assert!(c.device_layers[0].is_some());
        let bytes = c.save().unwrap();
        let mut other = PackedGroupAffineKvCache::new("wrong", 1, 1, 1, 8, 4).unwrap();
        assert!(other.restore(&bytes).is_err());
        let mut restored = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        restored.restore(&bytes).unwrap();
        assert!(restored.device_layers[0].is_none());
        // Snapshot semantics include the versioned layout and logical/token capacity, but not the
        // allocator-dependent capacity of each backing Vec. Restore must report the physical
        // allocation it actually received instead of replaying a stale byte count from another
        // process/allocation history.
        let restored_allocated = restored.allocated_payload_bytes();
        assert_eq!(
            restored.representation().allocated_bytes,
            restored_allocated
        );
        assert!(restored_allocated >= restored.logical_stored_bytes());
        assert_eq!(
            RepresentationMetadata {
                allocated_bytes: 0,
                host_allocated_payload_bytes: 0,
                retained_device_packed_logical_bytes: 0,
                ..restored.representation()
            },
            RepresentationMetadata {
                allocated_bytes: 0,
                host_allocated_payload_bytes: 0,
                retained_device_packed_logical_bytes: 0,
                ..c.representation()
            }
        );
        assert_same_rows(&restored, &c);
        restored.packed_mlx_arguments(0).unwrap();
        assert!(restored.device_layers[0].is_some());
        restored.restore(&bytes).unwrap();
        assert!(restored.device_layers[0].is_none());
        assert_eq!(
            restored
                .dispatch_telemetry()
                .retained_device_packed_logical_bytes,
            0
        );
    }

    #[test]
    fn qualified_group32_snapshot_restore_and_trim_preserve_evaluated_rows() {
        const BATCH: usize = 2;
        const HEADS: usize = 2;
        const WIDTH: usize = 64;
        const TOKENS: usize = 35;
        let keys = bhst_data(BATCH, HEADS, TOKENS, WIDTH, -11.0);
        let values = bhst_data(BATCH, HEADS, TOKENS, WIDTH, 13.0);
        let mut source = PackedGroupAffineKvCache::new(
            "group32-snapshot",
            1,
            BATCH,
            HEADS,
            WIDTH,
            PACKED_METAL_QUANT_GROUP_SIZE,
        )
        .unwrap();
        source.append(0, &keys, &values, TOKENS).unwrap();
        let expected = source.evaluated_dense_layer(0).unwrap();
        let snapshot = source.save().unwrap();

        let mut restored = PackedGroupAffineKvCache::new(
            "group32-snapshot",
            1,
            BATCH,
            HEADS,
            WIDTH,
            PACKED_METAL_QUANT_GROUP_SIZE,
        )
        .unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.representation().bits, 2);
        assert_eq!(
            restored.representation().group_size,
            PACKED_METAL_QUANT_GROUP_SIZE
        );
        assert_eq!(restored.evaluated_dense_layer(0).unwrap(), expected);

        restored.trim(33).unwrap();
        assert_eq!(restored.logical_len(), 33);
        let trimmed = restored.evaluated_dense_layer(0).unwrap();
        let trimmed_snapshot = restored.save().unwrap();
        let mut second = PackedGroupAffineKvCache::new(
            "group32-snapshot",
            1,
            BATCH,
            HEADS,
            WIDTH,
            PACKED_METAL_QUANT_GROUP_SIZE,
        )
        .unwrap();
        second.restore(&trimmed_snapshot).unwrap();
        assert_eq!(second.evaluated_dense_layer(0).unwrap(), trimmed);
        assert!(second.device_layers.iter().all(Option::is_none));
    }

    #[test]
    fn snapshot_rejects_exact_quantization_parameter_mismatch_without_mutation() {
        let mut source = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        let payload = data(3, 1, 8, -2.0);
        source.append(0, &payload, &payload, 3).unwrap();
        let snapshot = source.save().unwrap();
        let mut incompatible = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 3).unwrap();
        let before = incompatible.representation();
        assert!(incompatible.restore(&snapshot).is_err());
        assert_eq!(incompatible.representation(), before);
    }

    #[test]
    fn code_width_is_recorded_and_a_snapshot_of_another_width_is_refused_without_mutation() {
        assert_eq!(PackedCodeBits::from_bits(2).unwrap(), PackedCodeBits::Two);
        assert_eq!(PackedCodeBits::from_bits(4).unwrap(), PackedCodeBits::Four);
        for unsupported in [0, 1, 3, 8] {
            assert!(PackedCodeBits::from_bits(unsupported).is_err());
        }
        assert_eq!(
            (
                PackedCodeBits::Four.codes_per_byte(),
                PackedCodeBits::Four.max_code()
            ),
            (2, 15)
        );
        for (stored, other) in [
            (PackedCodeBits::Two, PackedCodeBits::Four),
            (PackedCodeBits::Four, PackedCodeBits::Two),
        ] {
            let payload = data(3, 1, 8, -2.0);
            let mut source =
                PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, 8, 4, stored).unwrap();
            source.append(0, &payload, &payload, 3).unwrap();
            assert_eq!(source.representation().bits, stored.bits());
            let snapshot = source.save().unwrap();
            let mut same = PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, 8, 4, stored).unwrap();
            same.restore(&snapshot).unwrap();
            assert_same_rows(&source, &same);
            let mut incompatible =
                PackedGroupAffineKvCache::with_bits("m", 1, 1, 1, 8, 4, other).unwrap();
            let resident = data(2, 1, 8, 5.0);
            incompatible.append(0, &resident, &resident, 2).unwrap();
            let (before, rows) = (
                incompatible.representation(),
                incompatible.read_row(0, 1, 0).unwrap(),
            );
            let error = incompatible.restore(&snapshot).unwrap_err().to_string();
            assert!(error.contains("quantization or shape mismatch"), "{error}");
            assert_eq!(incompatible.representation(), before);
            assert_eq!(incompatible.read_row(0, 1, 0).unwrap(), rows);
        }
    }

    /// A reader built for another code width is refused at binding, and the decoder factory falls
    /// back to dense before any packed mutation.
    #[test]
    fn a_reader_of_another_code_width_is_refused_before_mutation() {
        let two_bit_reader = CompiledKernelHandle::new(Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: None,
            cancel_on_failure: false,
        }));
        assert_eq!(two_bit_reader.code_bits(), PackedCodeBits::Two);
        let mut cache = PackedGroupAffineKvCache::with_bits(
            "test-packed",
            1,
            1,
            1,
            64,
            PACKED_METAL_QUANT_GROUP_SIZE,
            PackedCodeBits::Four,
        )
        .unwrap();
        let error = cache
            .bind_compiled_handle(two_bit_reader.clone())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("reads 2-bit codes but the cache stores 4-bit codes"),
            "{error}"
        );
        assert!(cache.compiled_handle().is_none());
        let selection = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "test-packed".into(),
                layers: 1,
                batch: 1,
                kv_heads: 1,
                head_dimension: 64,
                group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                bits: PackedCodeBits::Four,
                query_length: 1,
                has_mask: false,
            },
            two_bit_reader,
        );
        let CacheRoute::DenseFallback { reason } = selection.route() else {
            panic!("a cross-width reader must not select the packed route");
        };
        assert!(
            reason.contains("packed reader rejected before mutation"),
            "{reason}"
        );
    }

    /// Stored bytes per resident token and KV head, read from the live storage, are exactly the
    /// representation's arithmetic: `D·b/8` code bytes for K and for V, plus two f16 per channel per
    /// 32-token K group and two f16 per 32-channel V group.
    #[test]
    fn stored_bytes_per_token_are_exact_for_each_code_width() {
        let (width, group, heads, tokens) = (128, 32, 2, 4096 + 13);
        let keys = pseudo_random_outliers(1, heads, tokens, width, 0x0b17_5eed);
        let values = pseudo_random_outliers(1, heads, tokens, width, 0x5eed_0b17);
        for (bits, per_token) in [(PackedCodeBits::Two, 96), (PackedCodeBits::Four, 160)] {
            let mut cache =
                PackedGroupAffineKvCache::with_bits("m", 1, 1, heads, width, group, bits).unwrap();
            cache.append(0, &keys, &values, tokens).unwrap();
            let [key_codes, key_metadata, pending, value_codes, value_metadata] =
                cache.layer_component_bytes(0).unwrap();
            let (complete, tail) = (4096, 13);
            let code_bytes = width * usize::from(bits.bits()) / 8;
            assert_eq!(key_codes, heads * complete * code_bytes, "{bits:?}");
            assert_eq!(
                key_metadata,
                heads * (complete / group) * width * 4,
                "{bits:?}"
            );
            assert_eq!(pending, heads * tail * width * 4, "{bits:?}");
            assert_eq!(value_codes, heads * tokens * code_bytes, "{bits:?}");
            assert_eq!(
                value_metadata,
                heads * tokens * (width / group) * 4,
                "{bits:?}"
            );
            assert_eq!(
                cache.logical_stored_bytes(),
                key_codes + key_metadata + pending + value_codes + value_metadata
            );
            // Per quantized token and KV head (the dense residual excluded): K + V codes and metadata.
            let quantized = key_codes
                + key_metadata
                + value_codes / tokens * complete
                + value_metadata / tokens * complete;
            assert_eq!(quantized, per_token * complete * heads, "{bits:?}");
            #[cfg(target_os = "macos")]
            {
                // The device-resident decoder layout stores the same bytes per quantized token;
                // the incomplete group stays a dense residual of the input dtype (f16 here).
                let mut device = device_cache_bits(1, heads, width, bits);
                device
                    .append_device(
                        0,
                        &bhsd(&keys, 1, heads, tokens, width)
                            .as_dtype(Dtype::Float16)
                            .unwrap(),
                        &bhsd(&values, 1, heads, tokens, width)
                            .as_dtype(Dtype::Float16)
                            .unwrap(),
                    )
                    .unwrap();
                let (codes, metadata) = device.retained_device_component_bytes();
                assert_eq!(codes, heads * complete * 2 * code_bytes, "{bits:?}");
                assert_eq!(
                    metadata,
                    heads * (complete * 32 + tail * width * 2 * 2),
                    "{bits:?}"
                );
                assert_eq!(
                    codes + metadata - heads * tail * width * 4,
                    per_token * complete * heads
                );
                let layer = device.device_layers[0].as_ref().unwrap();
                let capacity = layer.capacity_tokens;
                assert_eq!(
                    array_bytes(&layer.key_codes) + array_bytes(&layer.value_codes),
                    (heads * capacity * 2 * code_bytes) as u64,
                    "{bits:?}: physical code arrays"
                );
            }
            // bf16 dense K+V is 4·D = 512 bytes per token and head.
            eprintln!(
                "{bits:?}: {per_token} B/token/head vs dense bf16 512 ({:.2}% reduction)",
                100.0 * (1.0 - per_token as f64 / 512.0)
            );
        }
    }
    #[test]
    fn fallback_is_explicit_and_cancel_is_safe() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 3, 2).unwrap();
        let event = c.dense_read_fallback("read", "unsupported mask");
        assert_eq!(event.reason, "unsupported mask");
        c.cancel();
        assert!(c.append(0, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1).is_err());
    }

    #[test]
    fn deterministic_extremes_zero_tail_and_non_aligned_width() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 11, 4).unwrap();
        let mut values = vec![0.0; 11];
        values[0] = -1000.0;
        values[4] = 1000.0;
        values[10] = 7.0;
        c.append(0, &values, &values, 1).unwrap();
        let (row, _) = c.read_row(0, 0, 0).unwrap();
        assert_eq!(row.len(), 11);
        assert!(row.iter().all(|value| value.is_finite()));
        assert_eq!(c.representation().logical_len, 1);
    }

    #[test]
    fn corrupted_snapshot_version_and_trailing_bytes_reject_without_mutation() {
        let mut c = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 4).unwrap();
        let x = data(2, 1, 8, -3.0);
        c.append(0, &x, &x, 2).unwrap();
        let before = c.representation();
        let mut version = c.save().unwrap();
        version[8] = 3;
        assert!(c.restore(&version).is_err());
        assert_eq!(c.representation(), before);
        let mut trailing = c.save().unwrap();
        trailing.push(0);
        assert!(c.restore(&trailing).is_err());
        assert_eq!(c.representation(), before);
    }

    #[test]
    fn atomic_layers_and_preflight_are_reachable() {
        let mut c =
            PackedGroupAffineKvCache::new("m", 2, 1, 1, 64, PACKED_METAL_QUANT_GROUP_SIZE).unwrap();
        let x = data(2, 1, 64, 0.0);
        let updates = [(&x[..], &x[..]), (&x[..x.len() - 1], &x[..x.len() - 1])];
        assert!(c.append_all_layers(&updates, 2).is_err());
        assert_eq!(c.logical_len(), 0);
        assert!(matches!(
            c.preflight("cpu", 1, false),
            crate::primitives::CacheRoute::DenseFallback { .. }
        ));
        assert!(matches!(
            c.preflight("mlx-metal", 1, false),
            crate::primitives::CacheRoute::DenseFallback { .. }
        ));
        let before_handle = c.process_visible_bytes_estimate();
        let handle = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
            "m",
            "mlx-metal",
            64,
            Arc::new(()),
        )));
        c.bind_compiled_handle(handle).unwrap();
        assert_eq!(c.compiled_handle().unwrap().backend(), "mlx-metal");
        assert_eq!(c.process_visible_bytes_estimate(), before_handle + 64);
        assert!(matches!(
            c.preflight("mlx-metal", 1, false),
            crate::primitives::CacheRoute::ExperimentalPacked
        ));
        let mismatch = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
            "other",
            "mlx-metal",
            0,
            Arc::new(()),
        )));
        assert!(c.bind_compiled_handle(mismatch).is_err());
    }

    #[test]
    fn decoder_factory_is_opt_in_and_falls_back_before_packed_mutation() {
        let mut disabled = select_decoder_cache(PackedCacheRequest::disabled(2));
        assert!(matches!(
            disabled.route(),
            CacheRoute::DenseFallback { reason } if reason.contains("disabled")
        ));
        assert!(disabled.cache.as_any_mut().is::<ContiguousKvCache>());

        let mut unsupported = select_decoder_cache(PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "m".into(),
            layers: 2,
            batch: 1,
            kv_heads: 1,
            head_dimension: 0,
            group_size: 32,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        });
        assert!(unsupported.cache.as_any_mut().is::<ContiguousKvCache>());

        let mut staged = select_decoder_cache(PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "m".into(),
            layers: 2,
            batch: 1,
            kv_heads: 2,
            head_dimension: 64,
            group_size: 32,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        });
        assert!(matches!(staged.route(), CacheRoute::DenseFallback { .. }));
        let adapter = staged
            .cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .expect("supported override stages storage through the decoder factory");
        assert_eq!(adapter.staged_representation().logical_len, 0);
        assert!(adapter.fallback_events().is_empty());
        assert!(matches!(
            adapter.preflight_packed(1, false),
            CacheRoute::DenseFallback { .. }
        ));
    }

    #[test]
    fn decoder_factory_binds_reader_before_experimental_route() {
        let handle = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
            "m",
            "mlx-metal",
            32,
            Arc::new(()),
        )));
        let selection = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "m".into(),
                layers: 1,
                batch: 1,
                kv_heads: 1,
                head_dimension: 64,
                group_size: 32,
                bits: PackedCodeBits::Two,
                query_length: 1,
                has_mask: false,
            },
            handle,
        );
        // An opaque handle is intentionally non-dispatchable, but factory binding and route
        // selection are still observable before any K/V mutation.
        assert_eq!(selection.route(), &CacheRoute::ExperimentalPacked);
        let mut cache = selection.into_cache();
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert_eq!(packed.staged_representation().logical_len, 0);
        assert!(packed
            .dispatch_packed(
                0,
                &Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]),
                crate::primitives::packed_metal::PackedMask::Causal
            )
            .is_err());
        assert_eq!(packed.staged_representation().logical_len, 0);
    }

    #[test]
    fn decoder_preflight_reports_exact_nonmutating_capability_reasons() {
        let mut cache = hook_cache(1, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert_eq!(
            packed.preflight_packed(1, false),
            CacheRoute::ExperimentalPacked
        );
        assert!(matches!(
            packed.preflight_packed(0, false),
            CacheRoute::DenseFallback { reason }
                if reason == "packed attention requires a non-empty query"
        ));
        assert!(matches!(
            packed.preflight_packed(1, true),
            CacheRoute::DenseFallback { reason }
                if reason == "additive attention masks require dense fallback"
        ));
        assert_eq!(packed.offset(), 0);
        assert!(packed.fallback_events().is_empty());
    }

    #[test]
    fn decoder_factory_rejects_mask_before_packed_mutation() {
        let handle = CompiledKernelHandle::new(Arc::new(OpaqueCompiledKernel::new(
            "m",
            "mlx-metal",
            1,
            Arc::new(()),
        )));
        let mut selection = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "m".into(),
                layers: 1,
                batch: 1,
                kv_heads: 1,
                head_dimension: 64,
                group_size: 32,
                bits: PackedCodeBits::Two,
                query_length: 1,
                has_mask: true,
            },
            handle,
        );
        assert!(matches!(
            selection.route(),
            CacheRoute::DenseFallback { .. }
        ));
        assert!(selection.cache.as_any_mut().is::<ContiguousKvCache>());
    }

    #[test]
    fn retained_metal_reader_rejects_non_qualified_quant_group_before_mutation() {
        let handle = CompiledKernelHandle::new(Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: None,
            cancel_on_failure: false,
        }));
        let selection = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "test-packed".into(),
                layers: 1,
                batch: 1,
                kv_heads: 1,
                head_dimension: 64,
                group_size: 4,
                bits: PackedCodeBits::Two,
                query_length: 1,
                has_mask: false,
            },
            handle,
        );
        assert!(matches!(
            selection.route(),
            CacheRoute::DenseFallback { reason }
                if reason.contains("quantization group size must be 32")
        ));
        assert_eq!(selection.cache.offset(), 0);
    }

    #[test]
    fn packed_hook_commits_only_after_all_layers_and_preserves_offset() {
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_some());
        assert_eq!(packed.offset(), 0);
        assert!(packed
            .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_some());
        assert_eq!(packed.offset(), 1);
        assert_eq!(packed.staged_representation().logical_len, 1);
        assert!(packed.staged_representation().allocated_bytes > 0);
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_some());
        assert_eq!(packed.offset(), 1);
        assert!(packed
            .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_some());
        assert_eq!(packed.offset(), 2);
    }

    #[test]
    fn sc20676_cloned_retained_handle_reports_one_real_cold_dispatch_across_caches() {
        let kernel = Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: None,
            cancel_on_failure: false,
        });
        let handle = CompiledKernelHandle::new(kernel);
        let request = || PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: "test-packed".into(),
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            bits: PackedCodeBits::Two,
            query_length: 1,
            has_mask: false,
        };
        let dispatch = |handle: CompiledKernelHandle| {
            let mut cache = select_decoder_cache_with_reader(request(), handle).into_cache();
            let packed = cache
                .as_any_mut()
                .downcast_mut::<DenseFallbackPackedDecoderCache>()
                .unwrap();
            let query = Array::from_slice(&[1.0_f32; 64], &[1, 1, 1, 64]);
            let kv = Array::from_slice(&[2.0_f32; 64], &[1, 1, 1, 64]);
            assert!(packed
                .try_packed_attention(
                    0,
                    &query,
                    &kv,
                    &kv,
                    PackedAttentionMask::Causal,
                    0.125,
                    false,
                )
                .unwrap()
                .is_some());
            packed.model_evidence()
        };

        let cold = dispatch(handle.clone());
        assert_eq!(cold.compile_jit_attempts, 1);
        assert_eq!(cold.cold_dispatches, 1);
        assert_eq!(cold.steady_dispatches, 0);
        let warm = dispatch(handle);
        assert_eq!(warm.compile_jit_attempts, 0);
        assert_eq!(warm.cold_dispatches, 0);
        assert_eq!(warm.steady_dispatches, 1);
        assert!(warm.kernel_warmed);
    }

    #[test]
    fn packed_hook_fault_discards_pending_step_and_lifecycle_stays_consistent() {
        let mut cache = hook_cache(2, Some(2));
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_some());
        assert_eq!(packed.offset(), 0);
        assert!(packed
            .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .is_err());
        assert_eq!(packed.offset(), 0);
        packed.reset().unwrap();
        assert_eq!(packed.offset(), 0);
        assert!(packed.retain_sequences(&[0]).is_ok());
    }

    #[test]
    fn later_layer_fault_restores_existing_device_residency_and_accepted_receipt_then_retries() {
        let (mut cache, kernel) = hook_cache_geometry(2, 1, 1, 64, Some(4));
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let query = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let values = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        for layer in 0..2 {
            assert!(packed
                .try_packed_attention(
                    layer,
                    &query,
                    &values,
                    &values,
                    PackedAttentionMask::Causal,
                    0.125,
                    false,
                )
                .unwrap()
                .is_some());
        }
        assert_eq!(packed.offset(), 1);
        let prior_devices = packed.staged.device_layers.clone();
        let prior_accepted = packed.staged.accepted_dispatch_snapshot();
        let prior_physical = packed.staged.dispatch_telemetry();
        // Kernel paths are accepted-call evidence: one per layer so far, restored with the step.
        assert_eq!(prior_accepted.kernel_paths.len(), 1);
        assert_eq!(prior_accepted.kernel_paths[0].1, 2);

        assert!(packed
            .try_packed_attention(
                0,
                &query,
                &values,
                &values,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap()
            .is_some());
        assert_eq!(
            packed
                .packed_evidence()
                .unwrap()
                .retained_device_packed_logical_bytes,
            packed
                .staged
                .dispatch_telemetry()
                .retained_device_packed_logical_bytes,
            "a step without a group flush holds nothing beside the in-place store for rollback"
        );
        assert!(packed
            .try_packed_attention(
                1,
                &query,
                &values,
                &values,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .is_err());
        assert_eq!(packed.offset(), 1);
        assert!(packed.pending_step.is_none());
        for (actual, expected) in packed.staged.device_layers.iter().zip(prior_devices.iter()) {
            assert_same_device_layer(actual, expected);
        }
        assert_eq!(packed.staged.accepted_dispatch_snapshot(), prior_accepted);
        let failed = packed.staged.dispatch_telemetry();
        assert_eq!(
            failed.dispatch_attempts,
            prior_physical.dispatch_attempts + 2
        );
        assert_eq!(
            failed.failed_dispatches,
            prior_physical.failed_dispatches + 1
        );
        assert!(failed.kernel_warmed);
        assert!(failed.uploaded_packed_bytes > prior_physical.uploaded_packed_bytes);
        assert!(
            failed.peak_packed_transient_logical_bytes
                >= prior_physical.peak_packed_transient_logical_bytes
        );
        assert_eq!(
            failed.accepted_uploaded_packed_bytes,
            prior_physical.accepted_uploaded_packed_bytes
        );

        for layer in 0..2 {
            assert!(packed
                .try_packed_attention(
                    layer,
                    &query,
                    &values,
                    &values,
                    PackedAttentionMask::Causal,
                    0.125,
                    false,
                )
                .unwrap()
                .is_some());
        }
        assert_eq!(kernel.calls.load(Ordering::Relaxed), 6);
        assert_eq!(packed.offset(), 2);
        assert_eq!(packed.staged.direct_dispatches(), 4);
        let retried = packed.staged.dispatch_telemetry();
        assert_eq!(
            retried.dispatch_attempts,
            prior_physical.dispatch_attempts + 4
        );
        assert_eq!(
            retried.failed_dispatches,
            prior_physical.failed_dispatches + 1
        );
        assert_eq!(retried.cold_dispatches, prior_physical.cold_dispatches);
        assert_eq!(
            retried.steady_dispatches,
            prior_physical.steady_dispatches + 2
        );
        assert!(
            retried.accepted_uploaded_packed_bytes > prior_physical.accepted_uploaded_packed_bytes
        );
        let evidence = packed.packed_evidence().unwrap();
        assert_eq!(evidence.dispatch_attempts, retried.dispatch_attempts);
        assert_eq!(evidence.failed_dispatches, 1);
        assert_eq!(evidence.accepted_direct_calls, 4);
        assert_eq!(
            evidence.accepted_uploaded_packed_bytes,
            retried.accepted_uploaded_packed_bytes
        );
    }

    #[test]
    fn typed_first_dispatch_cancellation_rolls_back_and_propagates_then_retry_succeeds() {
        let kernel = Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: Some(1),
            cancel_on_failure: true,
        });
        let mut selection = select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "test-packed".into(),
                layers: 1,
                batch: 1,
                kv_heads: 1,
                head_dimension: 64,
                group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                bits: PackedCodeBits::Two,
                query_length: 1,
                has_mask: false,
            },
            CompiledKernelHandle::new(kernel.clone()),
        );
        let packed = selection
            .cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let query = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let values = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        let error = packed
            .try_packed_attention(
                0,
                &query,
                &values,
                &values,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap_err();
        assert!(matches!(error, Error::Canceled));
        assert_eq!(packed.offset(), 0);
        assert!(packed.pending_step.is_none());
        assert!(packed.staged.device_layers[0].is_none());
        assert!(packed.staged.compiled_handle().is_some());
        assert!(packed.fallback_events().is_empty());
        let cancelled = packed.staged.dispatch_telemetry();
        assert_eq!(cancelled.dispatch_attempts, 1);
        assert_eq!(cancelled.failed_dispatches, 1);
        assert_eq!(cancelled.cold_dispatches, 0);
        assert_eq!(packed.staged.direct_dispatches(), 0);
        assert!(cancelled.uploaded_packed_bytes > 0);
        assert_eq!(cancelled.accepted_uploaded_packed_bytes, 0);

        assert!(packed
            .try_packed_attention(
                0,
                &query,
                &values,
                &values,
                PackedAttentionMask::Causal,
                0.125,
                false,
            )
            .unwrap()
            .is_some());
        assert_eq!(kernel.calls.load(Ordering::Relaxed), 2);
        assert_eq!(packed.offset(), 1);
        assert_eq!(packed.staged.direct_dispatches(), 1);
        let retried = packed.staged.dispatch_telemetry();
        assert_eq!(retried.dispatch_attempts, 2);
        assert_eq!(retried.failed_dispatches, 1);
        assert_eq!(retried.cold_dispatches, 1);
        assert!(retried.kernel_warmed);
        assert!(retried.uploaded_packed_bytes > retried.accepted_uploaded_packed_bytes);
        assert!(retried.accepted_uploaded_packed_bytes > 0);
    }

    #[test]
    fn retained_dispatch_fault_keeps_host_append_private_device_and_truthful_retry_evidence() {
        let kernel = Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: Some(1),
            cancel_on_failure: false,
        });
        let mut cache = PackedGroupAffineKvCache::new("test-packed", 1, 1, 1, 64, 32).unwrap();
        cache
            .bind_compiled_handle(CompiledKernelHandle::new(kernel.clone()))
            .unwrap();
        let values = vec![2.0f32; 64];
        cache.append(0, &values, &values, 1).unwrap();
        let query = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        assert!(cache
            .dispatch_packed(
                0,
                &query,
                crate::primitives::packed_metal::PackedMask::Causal,
            )
            .is_err());
        assert!(cache.device_layers[0].is_none());
        assert_eq!(cache.logical_len(), 1, "host append remains caller-owned");
        assert!(cache.layers[0].is_some());
        assert_eq!(cache.direct_dispatches(), 0);
        let failed = cache.dispatch_telemetry();
        assert_eq!(failed.dispatch_attempts, 1);
        assert_eq!(failed.failed_dispatches, 1);
        assert_eq!(failed.compile_jit_attempts, 1);
        assert!(!failed.kernel_warmed);
        assert!(failed.uploaded_packed_bytes > 0);
        assert_eq!(failed.accepted_uploaded_packed_bytes, 0);
        assert_eq!(failed.retained_device_packed_logical_bytes, 0);

        cache
            .dispatch_packed(
                0,
                &query,
                crate::primitives::packed_metal::PackedMask::Causal,
            )
            .unwrap();
        assert_eq!(kernel.calls.load(Ordering::Relaxed), 2);
        assert!(cache.device_layers[0].is_some());
        assert_eq!(cache.logical_len(), 1);
        assert_eq!(cache.direct_dispatches(), 1);
        let retried = cache.dispatch_telemetry();
        assert_eq!(retried.dispatch_attempts, 2);
        assert_eq!(retried.failed_dispatches, 1);
        assert_eq!(retried.compile_jit_attempts, 2);
        assert!(retried.kernel_warmed);
        assert!(retried.uploaded_packed_bytes > retried.accepted_uploaded_packed_bytes);
        assert!(retried.accepted_uploaded_packed_bytes > 0);
        assert!(retried.retained_device_packed_logical_bytes > 0);
    }

    #[test]
    fn host_mirror_extents_advance_only_after_a_successful_dispatch() {
        let kernel = Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: Some(2),
            cancel_on_failure: false,
        });
        let mut cache = PackedGroupAffineKvCache::new(
            "test-packed",
            1,
            2,
            2,
            64,
            PACKED_METAL_QUANT_GROUP_SIZE,
        )
        .unwrap();
        cache
            .bind_compiled_handle(CompiledKernelHandle::new(kernel))
            .unwrap();
        let query = Array::from_slice(&vec![0.5f32; 2 * 2 * 64], &[2, 2, 1, 64]);
        let first = bhst_data(2, 2, 33, 64, -1.5);
        cache.append(0, &first, &first, 33).unwrap();
        cache
            .dispatch_packed(0, &query, crate::primitives::packed_metal::PackedMask::None)
            .unwrap();
        let accepted = cache.device_layers[0].clone();
        let accepted_telemetry = cache.dispatch_telemetry();
        let extent = |cache: &PackedGroupAffineKvCache| {
            let device = cache.device_layers[0].as_ref().unwrap();
            (
                device.key_packed_tokens,
                device.value_packed_tokens,
                device.key_tail_rows,
            )
        };
        assert_eq!(extent(&cache), (32, 33, 1));

        let second = bhst_data(2, 2, 40, 64, 3.0);
        cache.append(0, &second, &second, 40).unwrap();
        assert!(cache
            .dispatch_packed(0, &query, crate::primitives::packed_metal::PackedMask::None)
            .is_err());
        assert_same_device_layer(&cache.device_layers[0], &accepted);
        let failed = cache.dispatch_telemetry();
        assert!(failed.uploaded_packed_bytes > accepted_telemetry.uploaded_packed_bytes);
        assert_eq!(
            failed.accepted_uploaded_packed_bytes,
            accepted_telemetry.accepted_uploaded_packed_bytes
        );
        assert_eq!(
            failed.retained_device_packed_logical_bytes,
            accepted_telemetry.retained_device_packed_logical_bytes
        );

        cache
            .dispatch_packed(0, &query, crate::primitives::packed_metal::PackedMask::None)
            .unwrap();
        assert_eq!(extent(&cache), (64, 73, 9));
    }

    #[test]
    fn first_packed_dispatch_fault_disables_route_and_falls_back_before_mutation() {
        let mut cache = hook_cache(2, Some(1));
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_none());
        assert_eq!(packed.offset(), 0);
        assert!(packed.staged.compiled_handle().is_none());
        assert!(packed
            .fallback_events()
            .iter()
            .any(|event| event.operation == "dispatch-fault"));
        assert!(packed
            .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_none());
        packed.update(0, &kv, &kv).unwrap();
        packed.update(1, &kv, &kv).unwrap();
        assert_eq!(packed.dense.offset(), 1);
    }

    #[test]
    fn pending_packed_step_rejects_retain_and_is_observably_discarded_by_truncate_and_reset() {
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap();
        assert!(packed.retain_sequences(&[0]).is_err());
        packed.truncate(0).unwrap();
        assert_eq!(packed.offset(), 0);
        assert!(packed
            .staged
            .fallback_events()
            .iter()
            .any(|event| event.operation == "truncate"));
        packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap();
        packed.reset().unwrap();
        assert_eq!(packed.offset(), 0);
        assert!(packed
            .staged
            .fallback_events()
            .iter()
            .any(|event| event.operation == "reset"));
    }

    #[test]
    fn committed_packed_history_reconstructs_exact_evaluated_dense_cache_before_batch_compaction() {
        const BATCH: usize = 3;
        const KV_HEADS: usize = 2;
        const QUERY_HEADS: usize = 4;
        const STEP: usize = 2;
        const WIDTH: usize = 64;
        let (mut cache, _) = hook_cache_geometry(1, BATCH, KV_HEADS, WIDTH, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        let query_values = bhst_data(BATCH, QUERY_HEADS, STEP, WIDTH, -0.5);
        let key_values = bhst_data(BATCH, KV_HEADS, STEP, WIDTH, -3.0);
        let value_values = bhst_data(BATCH, KV_HEADS, STEP, WIDTH, 7.0);
        let query = Array::from_slice(
            &query_values,
            &[BATCH as i32, QUERY_HEADS as i32, STEP as i32, WIDTH as i32],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        let keys = Array::from_slice(
            &key_values,
            &[BATCH as i32, KV_HEADS as i32, STEP as i32, WIDTH as i32],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        let values = Array::from_slice(
            &value_values,
            &[BATCH as i32, KV_HEADS as i32, STEP as i32, WIDTH as i32],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        assert!(packed
            .try_packed_attention(
                0,
                &query,
                &keys,
                &values,
                PackedAttentionMask::Causal,
                (WIDTH as f32).powf(-0.5),
                false,
            )
            .unwrap()
            .is_some());
        assert_eq!(packed.offset(), STEP as i32);

        let keep = [2, 0];
        let (_, all_evaluated_keys, all_evaluated_values) =
            packed.staged.evaluated_dense_layer(0).unwrap();
        let mut expected_keys = Vec::new();
        let mut expected_values = Vec::new();
        for &batch in &keep {
            for head in 0..KV_HEADS {
                let row = batch * KV_HEADS + head;
                for token in 0..STEP {
                    let start = (row * STEP + token) * WIDTH;
                    expected_keys.extend_from_slice(&all_evaluated_keys[start..start + WIDTH]);
                    expected_values.extend_from_slice(&all_evaluated_values[start..start + WIDTH]);
                }
            }
        }
        packed
            .retain_sequences(&keep.map(|value| value as i32))
            .unwrap();
        assert_eq!(packed.batch_size(), keep.len() as i32);
        let evidence = packed.packed_evidence().unwrap();
        assert_eq!(evidence.accepted_direct_calls, 1);
        assert_eq!(evidence.full_cache_dequantizations, 1);
        assert!(evidence.dense_active);
        assert_eq!(evidence.bits, 2);
        assert_eq!(evidence.quantization_group_size, 32);
        assert!(evidence.fallback_reasons.iter().any(|(operation, reason)| {
            operation == "retain_sequences"
                && reason == "batch compaction requires exact dense reconstruction"
        }));

        let expected_keys = Array::from_slice(
            &expected_keys,
            &[
                keep.len() as i32,
                KV_HEADS as i32,
                STEP as i32,
                WIDTH as i32,
            ],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap();
        let expected_values = Array::from_slice(
            &expected_values,
            &[
                keep.len() as i32,
                KV_HEADS as i32,
                STEP as i32,
                WIDTH as i32,
            ],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap();
        expected_keys.eval().unwrap();
        expected_values.eval().unwrap();

        let next = Array::from_slice(
            &vec![0.0f32; keep.len() * KV_HEADS * WIDTH],
            &[keep.len() as i32, KV_HEADS as i32, 1, WIDTH as i32],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        let (actual_keys, actual_values) = packed.update(0, &next, &next).unwrap();
        let actual_keys = actual_keys.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let actual_values = actual_values.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        actual_keys.eval().unwrap();
        actual_values.eval().unwrap();
        assert_eq!(
            actual_keys.shape(),
            &mlx_shape([keep.len(), KV_HEADS, STEP + 1, WIDTH]).unwrap()
        );
        for (actual, expected) in [
            (
                actual_keys.as_slice::<f32>(),
                expected_keys.as_slice::<f32>(),
            ),
            (
                actual_values.as_slice::<f32>(),
                expected_values.as_slice::<f32>(),
            ),
        ] {
            for row in 0..keep.len() * KV_HEADS {
                for token in 0..STEP {
                    let actual_start = (row * (STEP + 1) + token) * WIDTH;
                    let expected_start = (row * STEP + token) * WIDTH;
                    assert_eq!(
                        &actual[actual_start..actual_start + WIDTH],
                        &expected[expected_start..expected_start + WIDTH],
                    );
                }
            }
        }
        assert_eq!(packed.offset(), (STEP + 1) as i32);
    }

    #[test]
    fn interleaved_caches_share_a_reader_without_sharing_transaction_or_receipt_state() {
        let shared = Arc::new(TestPackedKernel {
            calls: AtomicUsize::new(0),
            fail_on: None,
            cancel_on_failure: false,
        });
        let make_cache = || {
            select_decoder_cache_with_reader(
                PackedCacheRequest {
                    enabled: true,
                    backend: "mlx-metal".into(),
                    identity: "test-packed".into(),
                    layers: 1,
                    batch: 1,
                    kv_heads: 1,
                    head_dimension: 64,
                    group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                    bits: PackedCodeBits::Two,
                    query_length: 1,
                    has_mask: false,
                },
                CompiledKernelHandle::new(shared.clone()),
            )
            .into_cache()
        };
        let mut left = make_cache();
        let mut right = make_cache();
        let q = Array::from_slice(&vec![0.25f32; 64], &[1, 1, 1, 64]);
        let left_kv = Array::from_slice(&vec![-1.0f32; 64], &[1, 1, 1, 64]);
        let right_kv = Array::from_slice(&vec![3.0f32; 64], &[1, 1, 1, 64]);
        for (cache, kv) in [(&mut left, &left_kv), (&mut right, &right_kv)] {
            assert!(cache
                .try_packed_attention(0, &q, kv, kv, PackedAttentionMask::Causal, 0.125, false,)
                .unwrap()
                .is_some());
        }
        assert_eq!(shared.calls.load(Ordering::Relaxed), 2);
        assert_eq!(left.offset(), 1);
        assert_eq!(right.offset(), 1);
        assert_eq!(left.packed_evidence().unwrap().accepted_direct_calls, 1);
        assert_eq!(right.packed_evidence().unwrap().accepted_direct_calls, 1);
        left.truncate(0).unwrap();
        assert_eq!(left.offset(), 0);
        assert_eq!(right.offset(), 1);
    }

    #[test]
    fn hard_pre_dispatch_errors_roll_back_the_pending_whole_step() {
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        let q_two = Array::from_slice(&vec![1.0f32; 128], &[1, 1, 2, 64]);
        let kv_two = Array::from_slice(&vec![2.0f32; 128], &[1, 1, 2, 64]);
        let assert_one_hard_error = |layer: usize, query: &Array, key: &Array, value: &Array| {
            let mut cache = hook_cache(2, None);
            let packed = cache
                .as_any_mut()
                .downcast_mut::<DenseFallbackPackedDecoderCache>()
                .unwrap();
            assert!(packed
                .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
                .unwrap()
                .is_some());
            assert!(packed.pending_step.is_some());
            assert_eq!(packed.offset(), 0);
            assert!(packed
                .try_packed_attention(
                    layer,
                    query,
                    key,
                    value,
                    PackedAttentionMask::Causal,
                    0.125,
                    false,
                )
                .is_err());
            assert!(packed.pending_step.is_none());
            assert_eq!(packed.offset(), 0);
            assert_eq!(packed.staged.layer_tokens(0), 0);
        };
        assert_one_hard_error(0, &q, &kv, &kv);
        assert_one_hard_error(1, &q_two, &kv_two, &kv_two);
    }

    #[test]
    fn late_capability_decline_preserves_partial_step_in_dense_cache() {
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();

        for layer in 0..2 {
            assert!(packed
                .try_packed_attention(
                    layer,
                    &q,
                    &kv,
                    &kv,
                    PackedAttentionMask::Causal,
                    0.125,
                    false,
                )
                .unwrap()
                .is_some());
        }
        assert_eq!(packed.offset(), 1);

        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
            .unwrap()
            .is_some());
        assert!(packed.pending_step.is_some());
        assert!(packed
            .try_packed_attention(1, &q, &kv, &kv, PackedAttentionMask::Additive, 0.125, false,)
            .unwrap()
            .is_none());
        assert!(packed.dense_active);
        assert!(packed.pending_step.is_none());
        assert_eq!(packed.staged.logical_len(), 0);
        assert_eq!(packed.dense.peek(0).unwrap().unwrap().0.shape()[2], 2);
        assert_eq!(packed.dense.peek(1).unwrap().unwrap().0.shape()[2], 1);

        packed.update(1, &kv, &kv).unwrap();
        assert_eq!(packed.offset(), 2);
        assert_eq!(packed.dense.peek(1).unwrap().unwrap().0.shape()[2], 2);
        assert_eq!(packed.staged.full_cache_dequantizations(), 1);
    }

    #[test]
    fn every_late_capability_branch_transitions_the_partial_step_to_dense() {
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let unsupported_q = Array::from_slice(&[1u8; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        for (query, mask, scale, retained_for_sharing) in [
            (&q, PackedAttentionMask::Causal, 0.125, true),
            (&q, PackedAttentionMask::Causal, 1.0, false),
            (&q, PackedAttentionMask::SlidingWindow(0), 0.125, false),
            (&q, PackedAttentionMask::Additive, 0.125, false),
            (&unsupported_q, PackedAttentionMask::Causal, 0.125, false),
        ] {
            let mut cache = hook_cache(2, None);
            let packed = cache
                .as_any_mut()
                .downcast_mut::<DenseFallbackPackedDecoderCache>()
                .unwrap();
            assert!(packed
                .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
                .unwrap()
                .is_some());
            assert!(packed
                .try_packed_attention(1, query, &kv, &kv, mask, scale, retained_for_sharing,)
                .unwrap()
                .is_none());
            assert!(packed.dense_active);
            assert!(packed.pending_step.is_none());
            assert_eq!(packed.dense.peek(0).unwrap().unwrap().0.shape()[2], 1);
            assert!(packed.dense.peek(1).unwrap().is_none());
            packed.update(1, &kv, &kv).unwrap();
            assert_eq!(packed.offset(), 1);
        }
    }

    #[test]
    fn direct_dense_update_after_a_packed_layer_preserves_that_partial_step() {
        let q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        let mut cache = hook_cache(2, None);
        let packed = cache
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert!(packed
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false,)
            .unwrap()
            .is_some());
        packed.update(1, &kv, &kv).unwrap();
        assert!(packed.dense_active);
        assert_eq!(packed.offset(), 1);
        assert_eq!(packed.dense.peek(0).unwrap().unwrap().0.shape()[2], 1);
        assert_eq!(packed.dense.peek(1).unwrap().unwrap().0.shape()[2], 1);
    }

    #[test]
    fn factory_rejects_zero_query_unsupported_dimension_and_group_before_staging() {
        for (query_length, head_dimension, group_size) in [(0, 64, 32), (1, 32, 32), (1, 64, 4)] {
            let mut selection = select_decoder_cache(PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: "test-packed".into(),
                layers: 2,
                batch: 1,
                kv_heads: 1,
                head_dimension,
                group_size,
                bits: PackedCodeBits::Two,
                query_length,
                has_mask: false,
            });
            assert!(selection.cache.as_any_mut().is::<ContiguousKvCache>());
            assert!(matches!(
                selection.route(),
                CacheRoute::DenseFallback { .. }
            ));
        }
    }

    #[test]
    fn unsupported_geometry_dtype_and_window_decline_before_first_append() {
        let valid_q = Array::from_slice(&vec![1.0f32; 64], &[1, 1, 1, 64]);
        let valid_kv = Array::from_slice(&vec![2.0f32; 64], &[1, 1, 1, 64]);
        let unsupported_q = Array::from_slice(&[1u8; 64], &[1, 1, 1, 64]);
        for (query, mask) in [
            (&unsupported_q, PackedAttentionMask::Causal),
            (&valid_q, PackedAttentionMask::SlidingWindow(0)),
            (&valid_q, PackedAttentionMask::SlidingWindow(usize::MAX)),
        ] {
            let mut cache = hook_cache(2, None);
            let packed = cache
                .as_any_mut()
                .downcast_mut::<DenseFallbackPackedDecoderCache>()
                .unwrap();
            assert!(packed
                .try_packed_attention(0, query, &valid_kv, &valid_kv, mask, 0.125, false)
                .unwrap()
                .is_none());
            assert_eq!(packed.offset(), 0);
            assert!(packed.pending_step.is_none());
            assert!(packed.staged.layers.iter().all(Option::is_none));
            assert!(packed.staged.compiled_handle().is_none());
        }
    }

    #[test]
    fn key_token_groups_and_value_channel_groups_are_chunk_invariant() {
        let (batch, heads, step, width, group) = (2, 2, 7, 5, 3);
        let keys = bhst_data(batch, heads, step, width, -9.0);
        let values = bhst_data(batch, heads, step, width, 4.0);
        let mut one = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        one.append(0, &keys, &values, step).unwrap();
        let mut chunks = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        for (start, len) in [(0, 1), (1, 2), (3, 1), (4, 3)] {
            chunks
                .append(
                    0,
                    &token_range(&keys, batch, heads, step, width, start, len),
                    &token_range(&values, batch, heads, step, width, start, len),
                    len,
                )
                .unwrap();
        }
        assert_same_rows(&one, &chunks);
        let store = one.layers[0].as_ref().unwrap();
        assert_eq!(store.keys.complete_tokens, 6);
        assert_eq!(store.keys.pending_tokens, 1);
        assert_eq!(store.keys.scales.len(), batch * heads * 2 * width);
        assert_eq!(
            store.values.scales.len(),
            batch * heads * step * width.div_ceil(group)
        );
        assert_eq!(
            one.representation().key_grouping,
            "token-axis groups [B,H,ceil(S/group_size),D]"
        );
    }

    #[test]
    fn packed_argument_adapter_reorders_multi_batch_multi_head_storage_row_major() {
        let (batch, heads, step, width, group) = (2, 3, 8, 8, 4);
        let keys = bhst_data(batch, heads, step, width, -17.0);
        let values = bhst_data(batch, heads, step, width, 23.0);
        let mut cache =
            PackedGroupAffineKvCache::new("layout", 1, batch, heads, width, group).unwrap();
        cache.append(0, &keys, &values, step).unwrap();
        let (kc, ks, kz, vc, vs, vz) = cache.packed_mlx_arguments(0).unwrap();
        for array in [&kc, &ks, &kz, &vc, &vs, &vz] {
            array.eval().unwrap();
        }
        let storage = cache.layers[0].as_ref().unwrap();
        let key_bytes = storage.keys.code_bytes_per_group();
        let value_bytes = cache.code_bits().code_bytes(width);
        let value_groups = width.div_ceil(group);
        let groups = step.div_ceil(group);
        let expected_kc =
            row_major_outer_rows(&storage.keys.codes, groups, batch * heads, key_bytes).unwrap();
        let expected_ks =
            row_major_outer_rows(&storage.keys.scales, groups, batch * heads, width).unwrap();
        let expected_kz =
            row_major_outer_rows(&storage.keys.zeros, groups, batch * heads, width).unwrap();
        let expected_vc =
            row_major_outer_rows(&storage.values.codes, step, batch * heads, value_bytes).unwrap();
        let expected_vs =
            row_major_outer_rows(&storage.values.scales, step, batch * heads, value_groups)
                .unwrap();
        let expected_vz =
            row_major_outer_rows(&storage.values.zeros, step, batch * heads, value_groups).unwrap();
        assert_eq!(kc.as_slice::<u8>(), expected_kc);
        assert_eq!(ks.as_slice::<f16>(), expected_ks);
        assert_eq!(kz.as_slice::<f16>(), expected_kz);
        assert_eq!(vc.as_slice::<u8>(), expected_vc);
        assert_eq!(vs.as_slice::<f16>(), expected_vs);
        assert_eq!(vz.as_slice::<f16>(), expected_vz);
        let device = cache.device_layers[0].as_ref().unwrap();
        assert_eq!(device.key_packed_tokens, groups * group);
        assert_eq!(device.value_packed_tokens, step);
        let initial = cache.dispatch_telemetry();
        assert_eq!(
            initial.uploaded_packed_bytes,
            initial.retained_device_packed_logical_bytes
        );
        let initial_representation = cache.representation();
        assert_eq!(
            initial_representation.allocated_bytes,
            initial_representation.host_allocated_payload_bytes
                + cache.retained_device_physical_bytes() as usize
        );
        assert!(
            cache.retained_device_physical_bytes()
                >= initial_representation.retained_device_packed_logical_bytes as u64,
            "block-preallocated device arrays are at least their live extents"
        );

        let extra_keys = bhst_data(batch, heads, 4, width, 101.0);
        let extra_values = bhst_data(batch, heads, 4, width, -203.0);
        cache.append(0, &extra_keys, &extra_values, 4).unwrap();
        cache.packed_mlx_arguments(0).unwrap();
        let extended = cache.dispatch_telemetry();
        assert!(
            extended.retained_device_packed_logical_bytes
                > initial.retained_device_packed_logical_bytes
        );
        assert_eq!(
            extended.uploaded_packed_bytes - initial.uploaded_packed_bytes,
            extended.retained_device_packed_logical_bytes
                - initial.retained_device_packed_logical_bytes,
            "extending a resident cache host-uploads only the new packed group/token payload"
        );

        cache.trim(5).unwrap();
        let extents = |cache: &PackedGroupAffineKvCache| {
            let device = cache.device_layers[0].as_ref().unwrap();
            (
                device.key_packed_tokens,
                device.value_packed_tokens,
                device.key_tail_rows,
            )
        };
        assert_eq!(
            extents(&cache),
            (4, 5, 0),
            "trim keeps exactly the mirror prefix that still matches the host representation"
        );
        let uploaded_before_rebuild = cache.dispatch_telemetry().uploaded_packed_bytes;
        cache.packed_mlx_arguments(0).unwrap();
        let after_trim = cache.dispatch_telemetry();
        assert_eq!(extents(&cache), (4, 5, 1));
        let pending_key_bytes = batch * heads * width * std::mem::size_of::<f32>();
        assert_eq!(
            after_trim.uploaded_packed_bytes - uploaded_before_rebuild,
            pending_key_bytes as u64,
            "after trim only the re-staged pending key row is uploaded; the kept prefix is not"
        );
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::arc_with_non_send_sync)]
    fn assert_real_metal_dense_sdpa_case(
        dtype: mlx_rs::Dtype,
        head_dimension: usize,
        kv_heads: usize,
        query_heads: usize,
        mask: PackedAttentionMask,
        family: crate::primitives::packed_metal::PackedMetalGpuFamily,
        bits: PackedCodeBits,
    ) {
        use crate::primitives::attention::{sdpa, AttnMask};
        use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
        use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};

        const BATCH: usize = 2;
        const QUERY_LEN: usize = 2;
        const KV_LEN: usize = 5;
        let identity = format!(
            "oracle-{dtype:?}-{head_dimension}-{kv_heads}-{query_heads}-{mask:?}-{family:?}-{bits:?}"
        );
        let keys = (0..BATCH * kv_heads * KV_LEN * head_dimension)
            .map(|index| ((index * 17 + 11) % 67) as f32 * 0.015625 - 0.5)
            .collect::<Vec<_>>();
        let values = (0..BATCH * kv_heads * KV_LEN * head_dimension)
            .map(|index| ((index * 29 + 7) % 79) as f32 * 0.01171875 - 0.4)
            .collect::<Vec<_>>();
        let queries = (0..BATCH * query_heads * QUERY_LEN * head_dimension)
            .map(|index| ((index * 13 + 5) % 53) as f32 * 0.0078125 - 0.2)
            .collect::<Vec<_>>();

        let mut cache = PackedGroupAffineKvCache::with_bits(
            identity.clone(),
            1,
            BATCH,
            kv_heads,
            head_dimension,
            PACKED_METAL_QUANT_GROUP_SIZE,
            bits,
        )
        .unwrap();
        cache.append(0, &keys, &values, KV_LEN).unwrap();
        let (_, dense_keys, dense_values) = cache.evaluated_dense_layer(0).unwrap();
        let query = Array::from_slice(
            &queries,
            &[
                BATCH as i32,
                query_heads as i32,
                QUERY_LEN as i32,
                head_dimension as i32,
            ],
        )
        .as_dtype(dtype)
        .unwrap();
        let dense_key = Array::from_slice(
            &dense_keys,
            &[
                BATCH as i32,
                kv_heads as i32,
                KV_LEN as i32,
                head_dimension as i32,
            ],
        )
        .as_dtype(dtype)
        .unwrap();
        let dense_value = Array::from_slice(
            &dense_values,
            &[
                BATCH as i32,
                kv_heads as i32,
                KV_LEN as i32,
                head_dimension as i32,
            ],
        )
        .as_dtype(dtype)
        .unwrap();
        let query_f32 = query.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let key_f32 = dense_key.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let value_f32 = dense_value.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        query_f32.eval().unwrap();
        key_f32.eval().unwrap();
        value_f32.eval().unwrap();
        let query_values = query_f32.as_slice::<f32>();
        let key_values = key_f32.as_slice::<f32>();
        let value_values = value_f32.as_slice::<f32>();
        let shape = PackedAttentionShape {
            batch: BATCH,
            query_heads,
            kv_heads,
            query_len: QUERY_LEN,
            kv_len: KV_LEN,
            head_dim: head_dimension,
        };
        let reference = attention_f32_masked(
            shape,
            query_values,
            |batch, head, token, channel| {
                key_values[((batch * kv_heads + head) * KV_LEN + token) * head_dimension + channel]
            },
            |batch, head, token, channel| {
                value_values
                    [((batch * kv_heads + head) * KV_LEN + token) * head_dimension + channel]
            },
            (head_dimension as f32).powf(-0.5),
            mask,
        )
        .unwrap();
        assert!(reference.iter().any(|value| value.abs() > 1e-4));

        let dense_mask = match mask {
            PackedAttentionMask::None => AttnMask::None,
            PackedAttentionMask::Causal => AttnMask::Causal,
            PackedAttentionMask::SlidingWindow(window) => AttnMask::SlidingCausal {
                window: window as i32,
            },
            PackedAttentionMask::Additive => unreachable!("additive is a deliberate fallback"),
        };
        let dense_output = sdpa(
            &query,
            &dense_key,
            &dense_value,
            (head_dimension as f32).powf(-0.5),
            dense_mask,
        )
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap();
        dense_output.eval().unwrap();

        cache
            .bind_compiled_handle(CompiledKernelHandle::new(Arc::new(
                PackedMetalKernel::for_identity_family_and_bits(identity, family, bits).unwrap(),
            )))
            .unwrap();
        let packed_mask = match mask {
            PackedAttentionMask::None => PackedMask::None,
            PackedAttentionMask::Causal => PackedMask::Causal,
            PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
            PackedAttentionMask::Additive => unreachable!("additive is a deliberate fallback"),
        };
        let packed_output = cache
            .dispatch_packed(0, &query, packed_mask)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        packed_output.eval().unwrap();
        let tolerance = match dtype {
            mlx_rs::Dtype::Float32 => 8e-3,
            mlx_rs::Dtype::Float16 => 4e-2,
            mlx_rs::Dtype::Bfloat16 => 8e-2,
            _ => unreachable!(),
        };
        for (index, ((actual, dense), reference)) in packed_output
            .as_slice::<f32>()
            .iter()
            .zip(dense_output.as_slice::<f32>())
            .zip(reference.iter())
            .enumerate()
        {
            assert!(actual.is_finite(), "non-finite packed output at {index}");
            assert!(
                (actual - dense).abs() <= tolerance,
                "packed/dense mismatch at {index}: {actual} != {dense}; dtype={dtype:?} D={head_dimension} KVH={kv_heads} QH={query_heads} mask={mask:?}"
            );
            assert!(
                (actual - reference).abs() <= tolerance,
                "packed/independent mismatch at {index}: {actual} != {reference}; dtype={dtype:?} D={head_dimension} KVH={kv_heads} QH={query_heads} mask={mask:?}"
            );
        }
        assert_eq!(cache.direct_dispatches(), 1);
        assert_eq!(cache.full_cache_dequantizations(), 0);
        let telemetry = cache.dispatch_telemetry();
        assert_eq!(telemetry.cold_dispatches, 1);
        assert_eq!(telemetry.steady_dispatches, 0);
        assert!(telemetry.uploaded_packed_bytes > 0);
        assert!(telemetry.retained_device_packed_logical_bytes > 0);
        // The residual key group is a separate argument that stays resident; nothing transient
        // (no padded or concatenated copy) is built for the reader.
        assert_eq!(
            telemetry.peak_packed_argument_logical_bytes,
            telemetry.retained_device_packed_logical_bytes
        );
        assert!(
            telemetry.peak_packed_transient_logical_bytes
                >= telemetry.retained_device_packed_logical_bytes
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn real_metal_group32_matches_dense_sdpa_and_independent_oracle_across_full_surface() {
        use crate::primitives::packed_metal::PackedMetalGpuFamily;

        for bits in PackedCodeBits::ALL {
            let dtypes = [
                mlx_rs::Dtype::Float16,
                mlx_rs::Dtype::Bfloat16,
                mlx_rs::Dtype::Float32,
            ];
            let dimensions = [64, 128, 256];
            let masks = [
                PackedAttentionMask::None,
                PackedAttentionMask::Causal,
                PackedAttentionMask::SlidingWindow(3),
            ];
            for (dtype_index, dtype) in dtypes.into_iter().enumerate() {
                for (dimension_index, dimension) in dimensions.into_iter().enumerate() {
                    let case = dtype_index * dimensions.len() + dimension_index;
                    assert_real_metal_dense_sdpa_case(
                        dtype,
                        dimension,
                        2,
                        4,
                        masks[case % masks.len()],
                        if case.is_multiple_of(2) {
                            PackedMetalGpuFamily::ConservativeUnknownApple
                        } else {
                            PackedMetalGpuFamily::Apple7OrNewer
                        },
                        bits,
                    );
                }
            }
            for (kv_heads, query_heads) in [(2, 2), (1, 4), (2, 4)] {
                assert_real_metal_dense_sdpa_case(
                    mlx_rs::Dtype::Float32,
                    64,
                    kv_heads,
                    query_heads,
                    PackedAttentionMask::Causal,
                    PackedMetalGpuFamily::Apple7OrNewer,
                    bits,
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn real_metal_adapter_dispatch_matches_nonuniform_fp32_reference_with_gqa_and_tail() {
        use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};
        for bits in PackedCodeBits::ALL {
            let (batch, kv_heads, q_heads, step, width, group) = (2, 2, 4, 5, 64, 32);
            let mut keys = Vec::with_capacity(batch * kv_heads * step * width);
            let mut values = Vec::with_capacity(batch * kv_heads * step * width);
            for row in 0..batch * kv_heads {
                for token in 0..step {
                    for channel in 0..width {
                        keys.push(
                            row as f32 * 32.0
                                + (token / group) as f32 * 8.0
                                + (token % group) as f32 * 2.0
                                + channel as f32,
                        );
                        values.push(
                            row as f32 * 32.0
                                + token as f32 * 8.0
                                + (channel / group) as f32 * 4.0
                                + (channel % group) as f32,
                        );
                    }
                }
            }
            let mut cache = PackedGroupAffineKvCache::with_bits(
                "device-layout",
                1,
                batch,
                kv_heads,
                width,
                group,
                bits,
            )
            .unwrap();
            cache.append(0, &keys, &values, step).unwrap();
            let (_, dequantized_keys, dequantized_values) = cache.evaluated_dense_layer(0).unwrap();
            let query_values = (0..batch * q_heads * width)
                .map(|index| (index as i32 % 7 - 3) as f32 * 0.01)
                .collect::<Vec<_>>();
            let query = Array::from_slice(
                &query_values,
                &[batch as i32, q_heads as i32, 1, width as i32],
            );
            let mut expected = vec![0.0f32; batch * q_heads * width];
            for b in 0..batch {
                for qh in 0..q_heads {
                    let kh = qh / (q_heads / kv_heads);
                    let row = b * kv_heads + kh;
                    let q_base = (b * q_heads + qh) * width;
                    let mut scores = Vec::with_capacity(step);
                    for token in 0..step {
                        let mut dot = 0.0f32;
                        for channel in 0..width {
                            dot += query_values[q_base + channel]
                                * dequantized_keys[(row * step + token) * width + channel];
                        }
                        scores.push(dot / (width as f32).sqrt());
                    }
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights = scores
                        .iter()
                        .map(|score| (*score - maximum).exp())
                        .collect::<Vec<_>>();
                    let norm = weights.iter().sum::<f32>();
                    for channel in 0..width {
                        let mut total = 0.0f32;
                        for token in 0..step {
                            total += weights[token]
                                * dequantized_values[(row * step + token) * width + channel];
                        }
                        expected[q_base + channel] = total / norm;
                    }
                }
            }
            cache
            .bind_compiled_handle(CompiledKernelHandle::new(Arc::new(
                PackedMetalKernel::for_identity_family_and_bits("device-layout", crate::primitives::packed_metal::PackedMetalGpuFamily::ConservativeUnknownApple, bits).unwrap(),
            )))
            .unwrap();
            let output = cache
                .dispatch_packed(0, &query, PackedMask::Causal)
                .unwrap();
            let actual = output.as_slice::<f32>();
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2e-3,
                    "index {index}: {actual} != {expected}"
                );
            }
            let steady = cache
                .dispatch_packed(0, &query, PackedMask::Causal)
                .unwrap();
            assert_eq!(steady.as_slice::<f32>(), actual);
            assert_eq!(cache.direct_dispatches(), 2);
            assert_eq!(cache.full_cache_dequantizations(), 0);
            let telemetry = cache.dispatch_telemetry();
            assert_eq!(telemetry.cold_dispatches, 1);
            assert_eq!(telemetry.steady_dispatches, 1);
        }
    }

    #[test]
    fn deterministic_pseudorandom_outliers_pack_identically_across_arbitrary_chunks() {
        for bits in PackedCodeBits::ALL {
            let (batch, heads, step, width, group) = (2, 3, 11, 7, 4);
            let keys = pseudo_random_outliers(batch, heads, step, width, 0x5eed_beef);
            let values = pseudo_random_outliers(batch, heads, step, width, 0x0123_4567);
            let mut one =
                PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits)
                    .unwrap();
            one.append(0, &keys, &values, step).unwrap();
            let mut chunks =
                PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits)
                    .unwrap();
            for (start, len) in [(0, 2), (2, 1), (3, 4), (7, 1), (8, 3)] {
                chunks
                    .append(
                        0,
                        &token_range(&keys, batch, heads, step, width, start, len),
                        &token_range(&values, batch, heads, step, width, start, len),
                        len,
                    )
                    .unwrap();
            }
            assert_same_rows(&one, &chunks);
            for token in 0..step {
                for row in 0..batch * heads {
                    let (k, v) = one.read_row(0, token, row).unwrap();
                    assert!(k.iter().chain(v.iter()).all(|value| value.is_finite()));
                }
            }
        }
    }

    #[test]
    fn rollback_reappend_clears_packed_tails_and_requantizes_cut_key_groups() {
        for bits in PackedCodeBits::ALL {
            let (batch, heads, width, group) = (1, 2, 5, 4);
            let original_keys = bhst_data(batch, heads, 6, width, -1000.0);
            let original_values = bhst_data(batch, heads, 6, width, 1000.0);
            let replacement_keys = bhst_data(batch, heads, 3, width, 700.0);
            let replacement_values = bhst_data(batch, heads, 3, width, -700.0);
            let mut rolled =
                PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits)
                    .unwrap();
            rolled
                .append(0, &original_keys, &original_values, 6)
                .unwrap();
            rolled.rollback(3).unwrap();
            rolled
                .append(0, &replacement_keys, &replacement_values, 3)
                .unwrap();

            let mut expected_keys = Vec::new();
            let mut expected_values = Vec::new();
            for row in 0..batch * heads {
                let prefix = row * 6 * width;
                let suffix = row * 3 * width;
                expected_keys.extend_from_slice(&original_keys[prefix..prefix + 3 * width]);
                expected_keys.extend_from_slice(&replacement_keys[suffix..suffix + 3 * width]);
                expected_values.extend_from_slice(&original_values[prefix..prefix + 3 * width]);
                expected_values.extend_from_slice(&replacement_values[suffix..suffix + 3 * width]);
            }
            let mut expected =
                PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits)
                    .unwrap();
            expected
                .append(0, &expected_keys, &expected_values, 6)
                .unwrap();
            assert_same_rows(&rolled, &expected);
            assert_eq!(rolled.logical_len(), 6);
        }
    }

    #[test]
    fn pending_key_groups_snapshot_and_byte_accounting_are_strict() {
        for bits in PackedCodeBits::ALL {
            let (batch, heads, step, width, group) = (2, 1, 5, 7, 3);
            let keys = bhst_data(batch, heads, step, width, 0.0);
            let mut values = bhst_data(batch, heads, step, width, 0.0);
            values[0] = -1000.0;
            let last = values.len() - 1;
            values[last] = 1000.0;
            let mut cache =
                PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits)
                    .unwrap();
            cache.append(0, &keys, &values, step).unwrap();
            let bytes_before = cache.logical_stored_bytes();
            assert!(cache.allocated_vec_bytes() >= bytes_before);
            assert!(cache.process_visible_bytes_estimate() >= cache.allocated_vec_bytes());
            let snapshot = cache.save().unwrap();
            let mut restored =
                PackedGroupAffineKvCache::with_bits("m", 1, batch, heads, width, group, bits)
                    .unwrap();
            restored.restore(&snapshot).unwrap();
            assert_same_rows(&cache, &restored);
            assert_eq!(
                cache.logical_stored_bytes(),
                restored.logical_stored_bytes()
            );
            let mut corrupt = snapshot.clone();
            let midpoint = corrupt.len() / 2;
            corrupt[midpoint] ^= 0x01;
            assert!(restored.restore(&corrupt).is_err());
            assert_same_rows(&cache, &restored);
            cache.clear();
            assert_eq!(cache.logical_stored_bytes(), 0);
        }
    }

    #[test]
    fn long_context_capacity_including_pending_tail_reduces_physical_payload_vs_dense_fp16() {
        let (step, width, group) = (4097, 128, 32);
        let keys = pseudo_random_outliers(1, 1, step, width, 0xfeed_face);
        let values = pseudo_random_outliers(1, 1, step, width, 0xface_feed);
        let mut cache = PackedGroupAffineKvCache::new("m", 1, 1, 1, width, group).unwrap();
        cache.append(0, &keys, &values, step).unwrap();
        let layer = cache.layers[0].as_ref().unwrap();
        assert_eq!(layer.keys.pending_tokens, 1);
        assert!(layer.keys.pending.len() >= width);
        assert!(cache.allocated_payload_bytes() * 4 <= cache.dense_fp16_equivalent_bytes() * 3);
        assert!(cache.process_visible_bytes_estimate() >= cache.allocated_payload_bytes());
    }

    fn bhsd(values: &[f32], batch: usize, heads: usize, tokens: usize, width: usize) -> Array {
        Array::from_slice(
            values,
            &[batch as i32, heads as i32, tokens as i32, width as i32],
        )
    }

    fn device_cache_bits(
        batch: usize,
        heads: usize,
        width: usize,
        bits: PackedCodeBits,
    ) -> PackedGroupAffineKvCache {
        PackedGroupAffineKvCache::with_bits(
            "test-packed",
            1,
            batch,
            heads,
            width,
            PACKED_METAL_QUANT_GROUP_SIZE,
            bits,
        )
        .unwrap()
    }

    fn host_readback<T: mlx_rs::ArrayElement + Copy>(array: &Array) -> Vec<T> {
        let array = materialized(array).unwrap();
        array.eval().unwrap();
        array.as_slice::<T>().to_vec()
    }

    /// GPU quantize-on-append is bit-identical to the CPU reference (codes, f16 scales and zeros),
    /// across chunked appends that complete groups mid-step, and the residual holds the exact
    /// not-yet-quantized input. Block growth past 256 tokens preserves the resident history.
    #[cfg(target_os = "macos")]
    #[test]
    fn device_append_quantizes_bit_identically_to_the_cpu_reference_across_chunks_and_growth() {
        for bits in PackedCodeBits::ALL {
            for width in [64, 128] {
                assert_device_append_matches_cpu_reference(bits, width);
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn assert_device_append_matches_cpu_reference(bits: PackedCodeBits, width: usize) {
        let (batch, heads) = (2, 2);
        let step = 301;
        let keys = pseudo_random_outliers(batch, heads, step, width, 0xabad_cafe);
        let values = pseudo_random_outliers(batch, heads, step, width, 0x1dea_f00d);
        let mut host = device_cache_bits(batch, heads, width, bits);
        host.append(0, &keys, &values, step).unwrap();
        let mut device = device_cache_bits(batch, heads, width, bits);
        for (start, len) in [(0, 1), (1, 30), (31, 5), (36, 41), (77, 224)] {
            let k = token_range(&keys, batch, heads, step, width, start, len);
            let v = token_range(&values, batch, heads, step, width, start, len);
            device
                .append_device(
                    0,
                    &bhsd(&k, batch, heads, len, width),
                    &bhsd(&v, batch, heads, len, width),
                )
                .unwrap();
        }
        assert_eq!(device.logical_len(), step);
        assert!(device.layers.iter().all(Option::is_none), "no host copy");
        assert_eq!(device.host_allocated_payload_bytes(), 0);
        let storage = host.layers[0].as_ref().unwrap();
        let layer = device.device_layers[0].as_ref().unwrap();
        assert!(layer.authoritative);
        assert_eq!(
            layer.capacity_tokens, 512,
            "grown by whole 256-token blocks"
        );
        let (rows, groups) = (batch * heads, step / 32);
        assert_eq!(layer.key_packed_tokens, groups * 32);
        assert_eq!(layer.value_packed_tokens, groups * 32);
        assert_eq!(layer.key_tail_rows, step - groups * 32);
        let key_words = 32 * width / bits.codes_per_byte();
        assert_eq!(
            layer.key_codes.shape()[3] as usize,
            key_words,
            "{bits:?}: packed key row width"
        );
        assert_eq!(
            layer.value_codes.shape()[3] as usize,
            width / bits.codes_per_byte()
        );
        assert_eq!(
            host_readback::<u8>(&live_rows(&layer.key_codes, groups).unwrap()),
            row_major_outer_rows(&storage.keys.codes, groups, rows, key_words).unwrap()
        );
        for (device_array, host_values) in [
            (&layer.key_scales, &storage.keys.scales),
            (&layer.key_zeros, &storage.keys.zeros),
        ] {
            assert_eq!(
                host_readback::<f16>(&live_rows(device_array, groups).unwrap()),
                row_major_outer_rows(host_values, groups, rows, width).unwrap()
            );
        }
        let packed = groups * 32;
        let value_rows = |source: &[u8]| {
            let words = width / bits.codes_per_byte();
            row_major_outer_rows(&source[..packed * rows * words], packed, rows, words).unwrap()
        };
        assert_eq!(
            host_readback::<u8>(&live_rows(&layer.value_codes, packed).unwrap()),
            value_rows(&storage.values.codes)
        );
        for (device_array, host_values) in [
            (&layer.value_scales, &storage.values.scales),
            (&layer.value_zeros, &storage.values.zeros),
        ] {
            let groups_per_row = width / 32;
            assert_eq!(
                host_readback::<f16>(&live_rows(device_array, packed).unwrap()),
                row_major_outer_rows(
                    &host_values[..packed * rows * groups_per_row],
                    packed,
                    rows,
                    groups_per_row
                )
                .unwrap()
            );
        }
        let tail = step - packed;
        assert_eq!(
            host_readback::<f32>(&live_rows(&layer.key_tail, tail).unwrap()),
            token_range(&keys, batch, heads, step, width, packed, tail)
        );
        assert_eq!(
            host_readback::<f32>(&live_rows(&layer.value_tail, tail).unwrap()),
            token_range(&values, batch, heads, step, width, packed, tail)
        );
    }

    /// The fused reader against the independent fp32 oracle: decode over two K groups plus a
    /// residual tail, a causal prefill chunk, and a split-KV history, at the campaign tolerance.
    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn fused_reader_matches_independent_fp32_oracle_at_campaign_tolerance() {
        use crate::primitives::packed_metal::{PackedMetalGpuFamily, PackedMetalKernel};
        for bits in PackedCodeBits::ALL {
            for family in [
                PackedMetalGpuFamily::ConservativeUnknownApple,
                PackedMetalGpuFamily::Apple7OrNewer,
            ] {
                let reader = CompiledKernelHandle::new(Arc::new(
                    PackedMetalKernel::for_identity_family_and_bits("parity", family, bits)
                        .unwrap(),
                ));
                let errors = group_affine_kernel_fp32_parity_errors(&reader).unwrap();
                assert_eq!(errors.len(), 4 * 128 + 4 * 5 * 128 + 4 * 128);
                let max = errors.iter().copied().fold(0.0, f64::max);
                assert!(
                    max <= crate::campaign::COMPRESSED_PARITY_MAX_ERROR,
                    "{family:?}: max parity error {max}"
                );
                eprintln!("campaign parity {bits:?} {family:?} D=128: max abs {max}");
                // The SC-20676 gate runs at the model's own head dimension.
                for width in [64, 256] {
                    let errors = group_affine_kernel_fp32_parity_errors_at(&reader, width).unwrap();
                    assert_eq!(errors.len(), (4 + 4 * 5 + 4) * width);
                    let max = errors.iter().copied().fold(0.0, f64::max);
                    assert!(
                        max <= crate::campaign::COMPRESSED_PARITY_MAX_ERROR,
                        "{family:?} at {width}: max parity error {max}"
                    );
                }
                assert!(group_affine_kernel_fp32_parity_errors_at(&reader, 100).is_err());
            }
        }
    }

    /// Kernel surface sweep: dtypes, head dimensions, masks, GQA, prefill rows, and forced split
    /// counts (including more splits than blocks) all agree with the fp32 oracle over the device
    /// representation; single-pass and split-KV outputs agree with each other.
    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn fused_reader_single_and_split_kv_paths_match_the_oracle_across_the_surface() {
        use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
        use crate::primitives::packed_metal::{
            PackedMask, PackedMetalGpuFamily, PackedMetalKernel,
        };
        for bits in PackedCodeBits::ALL {
            let cases = [
                (Dtype::Float32, 64, 2, 4, 77, 1, PackedAttentionMask::None),
                (
                    Dtype::Float16,
                    128,
                    2,
                    6,
                    100,
                    3,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Bfloat16,
                    256,
                    1,
                    2,
                    70,
                    1,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Float32,
                    128,
                    2,
                    4,
                    131,
                    4,
                    PackedAttentionMask::SlidingWindow(45),
                ),
                (
                    Dtype::Float32,
                    64,
                    4,
                    4,
                    32,
                    32,
                    PackedAttentionMask::Causal,
                ),
            ];
            for (index, (dtype, width, kv_heads, query_heads, tokens, query_len, mask)) in
                cases.into_iter().enumerate()
            {
                let batch = 2;
                // Clamped outliers, then a distinct range per 32-channel group and per token so every
                // K channel group and V channel-group scale/zero differs (a wrong metadata index is
                // numerically visible).
                let shaped = |seed: u64| {
                    pseudo_random_outliers(batch, kv_heads, tokens, width, seed)
                        .into_iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let (channel, token) = (i % width, (i / width) % tokens);
                            value.clamp(-4.0, 4.0) * (0.25 + (channel / 32) as f32 * 0.5)
                                + (channel / 32) as f32 * 0.75
                                - (token % 7) as f32 * 0.125
                        })
                        .collect::<Vec<_>>()
                };
                let keys = shaped(11 + index as u64);
                let values = shaped(97 + index as u64);
                let queries = (0..batch * query_heads * query_len * width)
                    .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                    .collect::<Vec<_>>();
                let mut cache = device_cache_bits(batch, kv_heads, width, bits);
                cache
                    .append_device(
                        0,
                        &bhsd(&keys, batch, kv_heads, tokens, width)
                            .as_dtype(dtype)
                            .unwrap(),
                        &bhsd(&values, batch, kv_heads, tokens, width)
                            .as_dtype(dtype)
                            .unwrap(),
                    )
                    .unwrap();
                let query = bhsd(&queries, batch, query_heads, query_len, width)
                    .as_dtype(dtype)
                    .unwrap();
                let query_f32 = host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap());
                let (_, dense_keys, dense_values) = cache.evaluated_dense_layer(0).unwrap();
                let reference = attention_f32_masked(
                    PackedAttentionShape {
                        batch,
                        query_heads,
                        kv_heads,
                        query_len,
                        kv_len: tokens,
                        head_dim: width,
                    },
                    &query_f32,
                    |b, h, t, d| dense_keys[((b * kv_heads + h) * tokens + t) * width + d],
                    |b, h, t, d| dense_values[((b * kv_heads + h) * tokens + t) * width + d],
                    (width as f32).powf(-0.5),
                    mask,
                )
                .unwrap();
                let packed_mask = match mask {
                    PackedAttentionMask::None => PackedMask::None,
                    PackedAttentionMask::Causal => PackedMask::Causal,
                    PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
                    PackedAttentionMask::Additive => unreachable!(),
                };
                let tolerance = match dtype {
                    Dtype::Float32 => 1e-4,
                    Dtype::Float16 => 4e-2,
                    _ => 8e-2,
                };
                let layer = cache.device_layers[0].as_ref().unwrap();
                let args = layer.args(&query, packed_mask);
                let mut single = None;
                let mut case_max = 0.0f32;
                for family in [
                    PackedMetalGpuFamily::ConservativeUnknownApple,
                    PackedMetalGpuFamily::Apple7OrNewer,
                ] {
                    let kernel =
                        PackedMetalKernel::for_identity_family_and_bits("sweep", family, bits)
                            .unwrap();
                    for splits in [1, 2, 3, 7, 64] {
                        let output = kernel
                            .dispatch_with_splits(&args, Some(splits))
                            .unwrap()
                            .as_dtype(Dtype::Float32)
                            .unwrap();
                        let output = host_readback::<f32>(&output);
                        for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                            case_max = case_max.max((actual - expected).abs());
                            assert!(
                            (actual - expected).abs() <= tolerance,
                            "{bits:?} case {index} {family:?} splits {splits} element {i}: {actual} != {expected}"
                        );
                        }
                        let single = single.get_or_insert_with(|| output.clone());
                        for (actual, expected) in output.iter().zip(single.iter()) {
                            assert!((actual - expected).abs() <= tolerance / 4.0);
                        }
                    }
                }
                eprintln!(
                    "per-row parity {bits:?} case {index} {dtype:?} D={width}: max abs {case_max}"
                );
            }
        }
    }

    /// Whole-step rollback across a group flush: a later-layer fault after layer 0 quantized a
    /// completed group restores both layers' extents and residuals exactly, and the retry produces
    /// the same representation as an uninterrupted cache.
    #[cfg(target_os = "macos")]
    #[test]
    fn rollback_after_a_group_flush_restores_the_exact_device_representation() {
        let width = 64;
        let history = pseudo_random_outliers(1, 1, 31, width, 5);
        let fresh = pseudo_random_outliers(1, 1, 2, width, 6);
        let run = |fail_on: Option<usize>| {
            let (mut cache, _) = hook_cache_geometry(2, 1, 1, width, fail_on);
            let packed = cache
                .as_any_mut()
                .downcast_mut::<DenseFallbackPackedDecoderCache>()
                .unwrap();
            let seed = bhsd(&history, 1, 1, 31, width);
            assert!(packed
                .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
                .unwrap());
            let before = packed.staged.device_layers.clone();
            let q = bhsd(&vec![0.1; 2 * width], 1, 1, 2, width);
            let kv = bhsd(&fresh, 1, 1, 2, width);
            let first = packed.try_packed_attention(
                0,
                &q,
                &kv,
                &kv,
                PackedAttentionMask::Causal,
                (width as f32).powf(-0.5),
                false,
            );
            assert!(first.unwrap().is_some());
            assert_eq!(
                packed.staged.device_layers[0]
                    .as_ref()
                    .unwrap()
                    .key_packed_tokens,
                32,
                "layer 0 flushed a completed group"
            );
            // The flush replaced layer 0's residual arrays; the pending step holds the originals
            // for rollback and the receipt counts them.
            let held = packed.pending_device_snapshot_overhead();
            assert!(held > 0);
            assert_eq!(
                packed.model_evidence().retained_device_metadata_bytes,
                packed.staged.retained_device_component_bytes().1 as u64 + held
            );
            let second = packed.try_packed_attention(
                1,
                &q,
                &kv,
                &kv,
                PackedAttentionMask::Causal,
                (width as f32).powf(-0.5),
                false,
            );
            (cache, before, second.map(|output| output.is_some()))
        };
        let (mut failed, before, outcome) = run(Some(2));
        assert!(outcome.is_err());
        let packed = failed
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        assert_eq!(packed.offset(), 31);
        for (actual, expected) in packed.staged.device_layers.iter().zip(before.iter()) {
            assert_same_device_layer(actual, expected);
        }
        let (mut clean, _, outcome) = run(None);
        assert!(outcome.unwrap());
        let q = bhsd(&vec![0.1; 2 * width], 1, 1, 2, width);
        let kv = bhsd(&fresh, 1, 1, 2, width);
        for layer in 0..2 {
            assert!(packed
                .try_packed_attention(
                    layer,
                    &q,
                    &kv,
                    &kv,
                    PackedAttentionMask::Causal,
                    (width as f32).powf(-0.5),
                    false,
                )
                .unwrap()
                .is_some());
        }
        let clean = clean
            .as_any_mut()
            .downcast_mut::<DenseFallbackPackedDecoderCache>()
            .unwrap();
        for (actual, expected) in packed
            .staged
            .device_layers
            .iter()
            .zip(clean.staged.device_layers.iter())
        {
            assert_same_device_layer(actual, expected);
        }
    }

    /// Trim into a completed group re-stages the kept prefix of that group from its quantized
    /// values, so the reconstructed history is exactly the kept prefix of the previous one.
    #[cfg(target_os = "macos")]
    #[test]
    fn device_trim_into_a_completed_group_restages_the_quantized_prefix() {
        for bits in PackedCodeBits::ALL {
            let (batch, heads, width, tokens) = (1, 2, 64, 77);
            let keys = pseudo_random_outliers(batch, heads, tokens, width, 21);
            let values = pseudo_random_outliers(batch, heads, tokens, width, 22);
            let mut cache = device_cache_bits(batch, heads, width, bits);
            cache
                .append_device(
                    0,
                    &bhsd(&keys, batch, heads, tokens, width),
                    &bhsd(&values, batch, heads, tokens, width),
                )
                .unwrap();
            let (_, full_keys, full_values) = cache.evaluated_dense_layer(0).unwrap();
            for keep in [70, 40, 32, 5] {
                cache.trim(keep).unwrap();
                let (resident, kept_keys, kept_values) = cache.evaluated_dense_layer(0).unwrap();
                assert_eq!(resident, keep);
                let layer = cache.device_layers[0].as_ref().unwrap();
                assert_eq!(layer.key_packed_tokens, keep / 32 * 32);
                assert_eq!(layer.key_tail_rows, keep % 32);
                for row in 0..batch * heads {
                    let prefix = |full: &[f32]| {
                        full[row * tokens * width..(row * tokens + keep) * width].to_vec()
                    };
                    let kept = |dense: &[f32]| {
                        dense[row * keep * width..(row + 1) * keep * width].to_vec()
                    };
                    assert_eq!(
                        kept(&kept_keys),
                        prefix(&full_keys),
                        "keep {keep} row {row}"
                    );
                    assert_eq!(
                        kept(&kept_values),
                        prefix(&full_values),
                        "keep {keep} row {row}"
                    );
                }
            }
            assert!(
                cache.save().is_err(),
                "a dense value residual cannot be snapshotted"
            );
        }
    }

    /// The decoder route end to end: a long first step attends densely over its own fresh K/V
    /// (exactly the dense SDPA result) and appends packed; the following decode steps use the
    /// fused reader lazily against the oracle; nothing falls back or reconstructs.
    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn decoder_first_step_is_dense_over_fresh_kv_then_decode_is_fused_and_lazy() {
        use crate::primitives::attention::{sdpa, AttnMask};
        use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
        use crate::primitives::packed_metal::PackedMetalKernel;
        for bits in PackedCodeBits::ALL {
            let (heads, query_heads, width, prompt) = (2, 4, 128, 40);
            let reader = CompiledKernelHandle::new(Arc::new(
                PackedMetalKernel::for_identity_family_and_bits(
                    "decoder-e2e",
                    crate::primitives::packed_metal::PackedMetalGpuFamily::ConservativeUnknownApple,
                    bits,
                )
                .unwrap(),
            ));
            let mut cache = select_decoder_cache_with_reader(
                PackedCacheRequest {
                    enabled: true,
                    backend: "mlx-metal".into(),
                    identity: "decoder-e2e".into(),
                    layers: 1,
                    batch: 1,
                    kv_heads: heads,
                    head_dimension: width,
                    group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                    bits,
                    query_length: 1,
                    has_mask: false,
                },
                reader,
            )
            .into_cache();
            let scale = (width as f32).powf(-0.5);
            let clamp = |values: Vec<f32>| {
                values
                    .into_iter()
                    .map(|v| v.clamp(-3.0, 3.0))
                    .collect::<Vec<_>>()
            };
            let keys = bhsd(
                &clamp(pseudo_random_outliers(1, heads, prompt, width, 31)),
                1,
                heads,
                prompt,
                width,
            );
            let values = bhsd(
                &clamp(pseudo_random_outliers(1, heads, prompt, width, 32)),
                1,
                heads,
                prompt,
                width,
            );
            let queries = bhsd(
                &(0..query_heads * prompt * width)
                    .map(|i| ((i * 13 + 1) % 41) as f32 * 0.03 - 0.6)
                    .collect::<Vec<_>>(),
                1,
                query_heads,
                prompt,
                width,
            );
            let first = cache
                .try_packed_attention(
                    0,
                    &queries,
                    &keys,
                    &values,
                    PackedAttentionMask::Causal,
                    scale,
                    false,
                )
                .unwrap()
                .unwrap();
            let dense = sdpa(&queries, &keys, &values, scale, AttnMask::Causal).unwrap();
            assert_eq!(host_readback::<f32>(&first), host_readback::<f32>(&dense));
            let evidence = cache.packed_evidence().unwrap();
            assert_eq!(
                evidence.accepted_direct_calls, 0,
                "no reader dispatch for the fresh-only step"
            );
            assert_eq!(cache.offset(), prompt as i32);

            for step in 0..3 {
                let q = bhsd(
                    &vec![0.05 * (step + 1) as f32; query_heads * width],
                    1,
                    query_heads,
                    1,
                    width,
                );
                let fresh = clamp(pseudo_random_outliers(1, heads, 1, width, 40 + step as u64));
                let kv = bhsd(&fresh, 1, heads, 1, width);
                let output = cache
                    .try_packed_attention(
                        0,
                        &q,
                        &kv,
                        &kv,
                        PackedAttentionMask::Causal,
                        scale,
                        false,
                    )
                    .unwrap()
                    .unwrap();
                let packed = cache
                    .as_any_mut()
                    .downcast_mut::<DenseFallbackPackedDecoderCache>()
                    .unwrap();
                let (tokens, dense_keys, dense_values) =
                    packed.staged.evaluated_dense_layer(0).unwrap();
                assert_eq!(tokens, prompt + step + 1);
                let reference = attention_f32_masked(
                    PackedAttentionShape {
                        batch: 1,
                        query_heads,
                        kv_heads: heads,
                        query_len: 1,
                        kv_len: tokens,
                        head_dim: width,
                    },
                    &host_readback::<f32>(&q),
                    |_, h, t, d| dense_keys[(h * tokens + t) * width + d],
                    |_, h, t, d| dense_values[(h * tokens + t) * width + d],
                    scale,
                    PackedAttentionMask::Causal,
                )
                .unwrap();
                for (actual, expected) in host_readback::<f32>(&output).iter().zip(&reference) {
                    assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
                }
            }
            let evidence = cache.packed_evidence().unwrap();
            assert_eq!(evidence.accepted_direct_calls, 3);
            assert_eq!(evidence.full_cache_dequantizations, 0);
            assert!(!evidence.dense_active && evidence.fallback_reasons.is_empty());
            assert!(evidence.accepted_uploaded_packed_bytes > 0);
            let storage = cache.compressed_storage().unwrap().unwrap();
            assert_eq!(
                storage.host_payload_bytes, 0,
                "the decoder route keeps no host copy"
            );
            assert_eq!(storage.tokens, (prompt + 3) as u64);
        }
    }

    /// Prefix import quantizes on append into the packed representation (no reconstruction); it
    /// declines without mutation when the route is not live and empty or the geometry differs.
    #[cfg(target_os = "macos")]
    #[test]
    fn prefix_import_appends_packed_and_declines_without_mutation() {
        let width = 64;
        let seed = bhsd(&pseudo_random_outliers(1, 1, 45, width, 3), 1, 1, 45, width);
        let wrong = bhsd(&pseudo_random_outliers(1, 1, 45, 128, 3), 1, 1, 45, 128);
        let mut cache = hook_cache(2, None);
        assert!(!cache
            .import_prefix(&[(seed.clone(), seed.clone())])
            .unwrap());
        assert!(!cache
            .import_prefix(&[(seed.clone(), seed.clone()), (wrong.clone(), wrong)])
            .unwrap());
        assert_eq!(cache.offset(), 0);
        assert!(cache
            .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed.clone())])
            .unwrap());
        assert_eq!(cache.offset(), 45);
        assert!(!cache
            .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
            .unwrap());
        let evidence = cache.packed_evidence().unwrap();
        assert_eq!(evidence.full_cache_dequantizations, 0);
        assert!(evidence.fallback_reasons.is_empty() && !evidence.dense_active);
        assert!(evidence.accepted_uploaded_packed_bytes > 0);
        let q = bhsd(&vec![0.2; width], 1, 1, 1, width);
        let kv = bhsd(&vec![0.3; width], 1, 1, 1, width);
        for layer in 0..2 {
            assert!(cache
                .try_packed_attention(
                    layer,
                    &q,
                    &kv,
                    &kv,
                    PackedAttentionMask::Causal,
                    0.125,
                    false
                )
                .unwrap()
                .is_some());
        }
        assert_eq!(cache.offset(), 46);
        assert!(!ContiguousKvCache::new(1)
            .import_prefix(&[(q.clone(), q)])
            .unwrap());
    }

    /// A reader that never produced an output faulting on its first dispatch over resident history
    /// (the first step went dense over fresh K/V, or a prefix was imported) rolls the step back and
    /// hands the exact packed history to the dense cache through the observable transition.
    #[cfg(target_os = "macos")]
    #[test]
    fn cold_reader_fault_over_resident_history_transitions_to_dense() {
        let width = 64;
        let (mut cache, kernel) = hook_cache_geometry(2, 1, 1, width, Some(1));
        let seed = bhsd(&pseudo_random_outliers(1, 1, 40, width, 8), 1, 1, 40, width);
        assert!(cache
            .import_prefix(&[(seed.clone(), seed.clone()), (seed.clone(), seed)])
            .unwrap());
        let q = bhsd(&vec![0.1; width], 1, 1, 1, width);
        let kv = bhsd(&vec![0.2; width], 1, 1, 1, width);
        assert!(cache
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_none());
        assert_eq!(kernel.calls.load(Ordering::Relaxed), 1);
        let evidence = cache.packed_evidence().unwrap();
        assert!(evidence.dense_active);
        assert_eq!(evidence.full_cache_dequantizations, 1);
        assert!(evidence
            .fallback_reasons
            .iter()
            .any(|(operation, _)| operation == "dispatch-fault"));
        assert_eq!(cache.offset(), 40, "the faulted step was rolled back");
        for layer in 0..2 {
            let (keys, _) = cache.update(layer, &kv, &kv).unwrap();
            assert_eq!(keys.shape()[2], 41);
        }
    }

    /// Oracle output over the device representation `cache` holds for layer 0.
    #[cfg(target_os = "macos")]
    fn tiled_oracle(
        cache: &PackedGroupAffineKvCache,
        query_f32: &[f32],
        shape: crate::primitives::packed_attention::PackedAttentionShape,
        mask: PackedAttentionMask,
    ) -> Vec<f32> {
        let (tokens, keys, values) = cache.evaluated_dense_layer(0).unwrap();
        assert_eq!(tokens, shape.kv_len);
        let (heads, width) = (shape.kv_heads, shape.head_dim);
        crate::primitives::packed_attention::attention_f32_masked(
            shape,
            query_f32,
            |b, h, t, d| keys[((b * heads + h) * tokens + t) * width + d],
            |b, h, t, d| values[((b * heads + h) * tokens + t) * width + d],
            (width as f32).powf(-0.5),
            mask,
        )
        .unwrap()
    }

    /// The tiled multi-row reader against the independent fp32 oracle at the existing tolerances:
    /// `S_q` 9/33/128/257 (partial last tiles, several tiles, rows spanning GQA heads), K and V
    /// residual tails beside packed groups, causal rows offset to the end of the KV range, GQA 1–3,
    /// a sliding window narrower than the chunk (also at D = 256), the host-mirror layout whose V rows
    /// are fully packed while K keeps a residual tail (`value_packed != key_packed`), head
    /// dimensions 64/128/256, all query dtypes, and
    /// forced KV split counts (the partial + reduction path), each agreeing with the single pass.
    #[cfg(target_os = "macos")]
    #[test]
    fn tiled_reader_matches_independent_fp32_oracle_for_multi_row_steps() {
        use crate::primitives::packed_attention::PackedAttentionShape;
        use crate::primitives::packed_metal::{
            PackedMask, PackedMetalGpuFamily, PackedMetalKernel,
        };
        for bits in PackedCodeBits::ALL {
            let cases = [
                (
                    Dtype::Float32,
                    64,
                    2,
                    6,
                    300,
                    9,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Float32,
                    128,
                    2,
                    4,
                    333,
                    33,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Float32,
                    128,
                    1,
                    3,
                    401,
                    128,
                    PackedAttentionMask::SlidingWindow(77),
                ),
                (
                    Dtype::Float32,
                    64,
                    2,
                    2,
                    530,
                    257,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Float32,
                    256,
                    1,
                    2,
                    110,
                    33,
                    PackedAttentionMask::Causal,
                ),
                (Dtype::Float32, 64, 1, 3, 150, 33, PackedAttentionMask::None),
                (
                    Dtype::Float16,
                    128,
                    2,
                    6,
                    290,
                    33,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Bfloat16,
                    64,
                    1,
                    4,
                    270,
                    128,
                    PackedAttentionMask::SlidingWindow(200),
                ),
                (
                    Dtype::Float32,
                    256,
                    1,
                    2,
                    300,
                    33,
                    PackedAttentionMask::SlidingWindow(90),
                ),
                // Host reference cache: V rows quantize per token, K keeps its incomplete group dense.
                (
                    Dtype::Float32,
                    128,
                    2,
                    4,
                    333,
                    33,
                    PackedAttentionMask::Causal,
                ),
            ];
            const HOST_MIRROR_CASE: usize = 9;
            let kernel = PackedMetalKernel::for_identity_family_and_bits(
                "tiled",
                PackedMetalGpuFamily::Apple7OrNewer,
                bits,
            )
            .unwrap();
            let mut maxima = Vec::new();
            for (index, (dtype, width, kv_heads, query_heads, tokens, query_len, mask)) in
                cases.into_iter().enumerate()
            {
                let batch = if index % 2 == 0 { 2 } else { 1 };
                // Distinct ranges per token and per 32-channel group so a wrong K group, V row, or
                // metadata index is numerically visible.
                let shaped = |seed: u64| {
                    pseudo_random_outliers(batch, kv_heads, tokens, width, seed)
                        .into_iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let (channel, token) = (i % width, (i / width) % tokens);
                            value.clamp(-4.0, 4.0) * (0.25 + (channel / 32) as f32 * 0.5)
                                + (channel / 32) as f32 * 0.75
                                - (token % 7) as f32 * 0.125
                        })
                        .collect::<Vec<_>>()
                };
                let keys = shaped(211 + index as u64);
                let values = shaped(307 + index as u64);
                let queries = (0..batch * query_heads * query_len * width)
                    .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                    .collect::<Vec<_>>();
                let mut cache = device_cache_bits(batch, kv_heads, width, bits);
                if index == HOST_MIRROR_CASE {
                    cache.append(0, &keys, &values, tokens).unwrap();
                } else {
                    cache
                        .append_device(
                            0,
                            &bhsd(&keys, batch, kv_heads, tokens, width)
                                .as_dtype(dtype)
                                .unwrap(),
                            &bhsd(&values, batch, kv_heads, tokens, width)
                                .as_dtype(dtype)
                                .unwrap(),
                        )
                        .unwrap();
                }
                let query = bhsd(&queries, batch, query_heads, query_len, width)
                    .as_dtype(dtype)
                    .unwrap();
                let reference = tiled_oracle(
                    &cache,
                    &host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap()),
                    PackedAttentionShape {
                        batch,
                        query_heads,
                        kv_heads,
                        query_len,
                        kv_len: tokens,
                        head_dim: width,
                    },
                    mask,
                );
                let packed_mask = match mask {
                    PackedAttentionMask::None => PackedMask::None,
                    PackedAttentionMask::Causal => PackedMask::Causal,
                    PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
                    PackedAttentionMask::Additive => unreachable!(),
                };
                let tolerance = match dtype {
                    Dtype::Float32 => 1e-4,
                    Dtype::Float16 => 4e-2,
                    _ => 8e-2,
                };
                let staged = cache.staged_reader_arguments(0).unwrap();
                let args = staged.args(&query, packed_mask);
                assert!(
                    args.key_packed_tokens > 0 && args.key_packed_tokens < tokens,
                    "{bits:?} case {index} exercises packed K groups and the K residual"
                );
                assert_eq!(
                    args.value_packed_tokens == tokens,
                    index == HOST_MIRROR_CASE,
                    "only the host-mirror case packs every V row ahead of K"
                );
                let mut single: Option<Vec<f32>> = None;
                let mut case_max = 0.0f32;
                for splits in [1, 2, 5] {
                    let output = host_readback::<f32>(
                        &kernel
                            .dispatch_tiled(&args, Some(splits))
                            .unwrap()
                            .as_dtype(Dtype::Float32)
                            .unwrap(),
                    );
                    assert_eq!(output.len(), reference.len());
                    for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                        let error = (actual - expected).abs();
                        case_max = case_max.max(error);
                        assert!(
                        error <= tolerance,
                        "{bits:?} case {index} splits {splits} element {i}: {actual} != {expected}"
                    );
                    }
                    let single = single.get_or_insert_with(|| output.clone());
                    for (actual, expected) in output.iter().zip(single.iter()) {
                        assert!((actual - expected).abs() <= tolerance / 4.0);
                    }
                }
                maxima.push((index, dtype, case_max));
            }
            eprintln!("tiled parity max abs error per case: {maxima:?}");
        }
    }

    /// The NAX tiled reader against the independent fp32 oracle at the tolerances the dense bf16/f16
    /// SDPA path is held to (`assert_real_metal_dense_sdpa_case`): `S_q` 16/33/128/2048, K and V
    /// residual tails (the pending K group) beside packed groups, causal rows offset to the end of
    /// the KV range, GQA 1–3, sliding windows, `None` masks, the host-mirror layout
    /// (`value_packed != key_packed`), D = 64/128, bf16 and f16 queries, and forced KV split counts
    /// (partial + reduction), each agreeing with the single pass. Each case also matches the fp32
    /// tiled kernel over the same inputs at 2e-2 (relative above 1), a bound a one-token mask or
    /// offset error in short rows exceeds. D = 256 and f32 queries are routed to the fp32 tiled
    /// kernel with the recorded reason. Requires a Neural-Accelerator Mac (ignored elsewhere).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Neural-Accelerator Mac (mlx::core::metal::is_nax_available()); run with --ignored"]
    fn nax_tiled_reader_matches_independent_fp32_oracle_for_multi_row_steps() {
        use crate::primitives::packed_attention::PackedAttentionShape;
        use crate::primitives::packed_metal::{
            mlx_nax_available, PackedKernelPath, PackedMask, PackedMetalGpuFamily,
            PackedMetalKernel,
        };
        for bits in PackedCodeBits::ALL {
            assert!(
                mlx_nax_available().unwrap(),
                "mlx::core::metal::is_nax_available() is false on this host"
            );
            let cases = [
                (
                    Dtype::Bfloat16,
                    128,
                    2,
                    6,
                    333,
                    33,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Float16,
                    64,
                    2,
                    4,
                    300,
                    16,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Bfloat16,
                    64,
                    1,
                    3,
                    401,
                    128,
                    PackedAttentionMask::SlidingWindow(77),
                ),
                (
                    Dtype::Float16,
                    128,
                    2,
                    2,
                    530,
                    128,
                    PackedAttentionMask::None,
                ),
                (
                    Dtype::Bfloat16,
                    64,
                    1,
                    2,
                    2100,
                    2048,
                    PackedAttentionMask::Causal,
                ),
                (
                    Dtype::Float16,
                    128,
                    1,
                    3,
                    290,
                    33,
                    PackedAttentionMask::SlidingWindow(200),
                ),
                (
                    Dtype::Float16,
                    128,
                    1,
                    2,
                    2090,
                    2048,
                    PackedAttentionMask::SlidingWindow(700),
                ),
                // Host reference cache: V rows quantize per token, K keeps its incomplete group dense.
                (
                    Dtype::Bfloat16,
                    128,
                    2,
                    4,
                    333,
                    33,
                    PackedAttentionMask::Causal,
                ),
            ];
            const HOST_MIRROR_CASE: usize = 7;
            let kernel = PackedMetalKernel::for_identity_family_and_bits(
                "nax",
                PackedMetalGpuFamily::Apple7OrNewer,
                bits,
            )
            .unwrap();
            assert!(kernel.nax_available());
            let mut maxima = Vec::new();
            for (index, (dtype, width, kv_heads, query_heads, tokens, query_len, mask)) in
                cases.into_iter().enumerate()
            {
                let batch = if index % 2 == 0 && query_len < 2048 {
                    2
                } else {
                    1
                };
                let shaped = |seed: u64| {
                    pseudo_random_outliers(batch, kv_heads, tokens, width, seed)
                        .into_iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let (channel, token) = (i % width, (i / width) % tokens);
                            value.clamp(-4.0, 4.0) * (0.25 + (channel / 32) as f32 * 0.5)
                                + (channel / 32) as f32 * 0.75
                                - (token % 7) as f32 * 0.125
                        })
                        .collect::<Vec<_>>()
                };
                let keys = shaped(411 + index as u64);
                let values = shaped(507 + index as u64);
                let queries = (0..batch * query_heads * query_len * width)
                    .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                    .collect::<Vec<_>>();
                let mut cache = device_cache_bits(batch, kv_heads, width, bits);
                if index == HOST_MIRROR_CASE {
                    cache.append(0, &keys, &values, tokens).unwrap();
                } else {
                    cache
                        .append_device(
                            0,
                            &bhsd(&keys, batch, kv_heads, tokens, width)
                                .as_dtype(dtype)
                                .unwrap(),
                            &bhsd(&values, batch, kv_heads, tokens, width)
                                .as_dtype(dtype)
                                .unwrap(),
                        )
                        .unwrap();
                }
                let query = bhsd(&queries, batch, query_heads, query_len, width)
                    .as_dtype(dtype)
                    .unwrap();
                let reference = tiled_oracle(
                    &cache,
                    &host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap()),
                    PackedAttentionShape {
                        batch,
                        query_heads,
                        kv_heads,
                        query_len,
                        kv_len: tokens,
                        head_dim: width,
                    },
                    mask,
                );
                let packed_mask = match mask {
                    PackedAttentionMask::None => PackedMask::None,
                    PackedAttentionMask::Causal => PackedMask::Causal,
                    PackedAttentionMask::SlidingWindow(window) => PackedMask::SlidingWindow(window),
                    PackedAttentionMask::Additive => unreachable!(),
                };
                let tolerance = match dtype {
                    Dtype::Float16 => 4e-2,
                    _ => 8e-2,
                };
                let staged = cache.staged_reader_arguments(0).unwrap();
                let args = staged.args(&query, packed_mask);
                assert!(
                    args.key_packed_tokens > 0 && args.key_packed_tokens < tokens,
                    "{bits:?} case {index} exercises packed K groups and the K residual"
                );
                assert_eq!(
                    args.value_packed_tokens == tokens,
                    index == HOST_MIRROR_CASE,
                    "only the host-mirror case packs every V row ahead of K"
                );
                assert!(
                    matches!(
                        kernel.planned_path(&args).unwrap(),
                        PackedKernelPath::NaxTiled { .. }
                    ),
                    "{bits:?} case {index} plans the NAX kernel"
                );
                let mut single: Option<Vec<f32>> = None;
                let (mut case_abs, mut case_rel) = (0.0f32, 0.0f32);
                for splits in [1, 2, 5] {
                    let output = host_readback::<f32>(
                        &kernel
                            .dispatch_nax(&args, Some(splits))
                            .unwrap()
                            .as_dtype(Dtype::Float32)
                            .unwrap(),
                    );
                    assert_eq!(output.len(), reference.len());
                    for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                        let error = (actual - expected).abs();
                        case_abs = case_abs.max(error);
                        case_rel = case_rel.max(error / expected.abs().max(1e-2));
                        assert!(
                        error <= tolerance,
                        "{bits:?} case {index} splits {splits} element {i}: {actual} != {expected}"
                    );
                    }
                    let single = single.get_or_insert_with(|| output.clone());
                    // Split and single pass round the same fp32 result to the 16-bit output, so
                    // they may differ by one output ULP, which grows with |x| (bf16 2^-5 from 4):
                    // the bound is relative above 1, the form the NAX-vs-tiled check below uses.
                    for (actual, expected) in output.iter().zip(single.iter()) {
                        assert!(
                            (actual - expected).abs() <= tolerance / 4.0 * expected.abs().max(1.0),
                            "{bits:?} case {index} splits {splits}: {actual} != single {expected}"
                        );
                    }
                }
                // Same inputs through the fp32 tiled kernel: both read the identical representation, so
                // they differ only by 16-bit tile rounding and output rounding.
                let tiled = host_readback::<f32>(
                    &kernel
                        .dispatch_tiled(&args, None)
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap(),
                );
                let mut case_tiled = 0.0f32;
                for (i, (nax, tiled)) in single.as_ref().unwrap().iter().zip(&tiled).enumerate() {
                    let error = (nax - tiled).abs() / tiled.abs().max(1.0);
                    case_tiled = case_tiled.max(error);
                    assert!(
                        error <= 2e-2,
                        "{bits:?} case {index} element {i}: NAX {nax} != tiled {tiled}"
                    );
                }
                maxima.push((
                    index, dtype, width, query_len, case_abs, case_rel, case_tiled,
                ));
            }
            eprintln!(
                "NAX parity (case, dtype, D, S_q, max abs, max rel, max vs tiled): {maxima:?}"
            );

            // D = 256 and f32 queries keep the fp32 tiled kernel, with the recorded reason.
            for (width, dtype) in [(256, Dtype::Bfloat16), (128, Dtype::Float32)] {
                let selection = kernel.nax_selection(width, dtype);
                assert!(!selection.selected, "D {width} {dtype:?}");
                let descriptor = kernel.kernel_descriptor(64, width, dtype).unwrap();
                assert_eq!(
                    descriptor.kernel,
                    "sc20676_tiled_multi_row_simdgroup_matrix"
                );
                assert_eq!(descriptor.selection, selection.selection);
                assert_eq!(descriptor.selection_reason, selection.reason);
                let mut cache = device_cache_bits(1, 1, width, bits);
                let kv = bhsd(
                    &pseudo_random_outliers(1, 1, 80, width, 3)
                        .into_iter()
                        .map(|value| value.clamp(-3.0, 3.0))
                        .collect::<Vec<_>>(),
                    1,
                    1,
                    80,
                    width,
                )
                .as_dtype(dtype)
                .unwrap();
                cache.append_device(0, &kv, &kv).unwrap();
                let query = bhsd(&vec![0.1; 2 * 64 * width], 1, 2, 64, width)
                    .as_dtype(dtype)
                    .unwrap();
                let staged = cache.staged_reader_arguments(0).unwrap();
                let args = staged.args(&query, PackedMask::Causal);
                assert!(matches!(
                    kernel.planned_path(&args).unwrap(),
                    PackedKernelPath::Tiled { .. }
                ));
                let refused = kernel.dispatch_nax(&args, None).unwrap_err().to_string();
                assert!(refused.contains(selection.reason), "{refused}");
            }
        }
    }

    /// `dispatch` runs exactly the kernel `planned_path` names: on the qualified family a tiled
    /// kernel from the per-D threshold (16 for D = 64/128, 32 for D = 256) and the per-row kernel
    /// below it — the NAX tiled kernel for bf16/f16 queries at D = 64/128 where MLX reports the
    /// Neural Accelerator, the fp32 tiled kernel otherwise (f32 queries, D = 256, or no NAX); on the
    /// conservative family always the per-row kernel (and both tiled seams are refused). The routed
    /// output is bit-identical to the named path's output, and the descriptor names the same kernel
    /// with the selection reason.
    #[cfg(target_os = "macos")]
    #[test]
    fn dispatch_runs_the_path_planned_path_names() {
        use crate::primitives::packed_metal::{
            mlx_nax_available, packed_tiled_min_query_tokens, PackedKernelPath, PackedMask,
            PackedMetalGpuFamily, PackedMetalKernel,
        };
        for bits in PackedCodeBits::ALL {
            assert_eq!(packed_tiled_min_query_tokens(64), 16);
            assert_eq!(packed_tiled_min_query_tokens(128), 16);
            assert_eq!(packed_tiled_min_query_tokens(256), 32);
            let nax = mlx_nax_available().unwrap();
            let conservative = PackedMetalKernel::for_identity_family_and_bits(
                "route",
                crate::primitives::packed_metal::PackedMetalGpuFamily::ConservativeUnknownApple,
                bits,
            )
            .unwrap();
            assert!(!conservative.nax_available());
            let recent = PackedMetalKernel::for_identity_family_and_bits(
                "route",
                PackedMetalGpuFamily::Apple7OrNewer,
                bits,
            )
            .unwrap();
            assert_eq!(recent.nax_available(), nax);
            let recent_without_nax = PackedMetalKernel::for_identity_family_and_bits(
                "route",
                PackedMetalGpuFamily::Apple7OrNewer,
                bits,
            )
            .unwrap()
            .without_nax();
            for width in [128, 256] {
                let tokens = 96;
                let data = pseudo_random_outliers(1, 2, tokens, width, 9)
                    .into_iter()
                    .map(|value| value.clamp(-3.0, 3.0))
                    .collect::<Vec<_>>();
                let mut cache = device_cache_bits(1, 2, width, bits);
                let kv = bhsd(&data, 1, 2, tokens, width);
                cache.append_device(0, &kv, &kv).unwrap();
                let layer = cache.device_layers[0].as_ref().unwrap();
                let threshold = packed_tiled_min_query_tokens(width);
                for (query_len, dtype) in [1, threshold - 1, threshold]
                    .into_iter()
                    .flat_map(|len| [(len, Dtype::Float32), (len, Dtype::Bfloat16)])
                {
                    let query = bhsd(
                        &(0..6 * query_len * width)
                            .map(|i| ((i * 7 + 1) % 23) as f32 * 0.04 - 0.4)
                            .collect::<Vec<_>>(),
                        1,
                        6,
                        query_len,
                        width,
                    )
                    .as_dtype(dtype)
                    .unwrap();
                    let args = layer.args(&query, PackedMask::Causal);
                    for (kernel, qualified, kernel_nax) in [
                        (&conservative, false, false),
                        (&recent, true, nax),
                        (&recent_without_nax, true, false),
                    ] {
                        let path = kernel.planned_path(&args).unwrap();
                        let multi_row = qualified && query_len >= threshold;
                        let nax_expected =
                            multi_row && kernel_nax && width == 128 && dtype == Dtype::Bfloat16;
                        let expected = match (multi_row, nax_expected) {
                            (false, _) => "sc20676_split_kv_simdgroup",
                            (true, false) => "sc20676_tiled_multi_row_simdgroup_matrix",
                            (true, true) => "sc20676_nax_tiled_matmul2d",
                        };
                        let context = format!(
                        "D {width} S_q {query_len} {dtype:?} qualified {qualified} nax {kernel_nax}"
                    );
                        assert_eq!(
                            (
                                matches!(path, PackedKernelPath::Tiled { .. }),
                                matches!(path, PackedKernelPath::NaxTiled { .. })
                            ),
                            (multi_row && !nax_expected, nax_expected),
                            "{context}: {path:?}"
                        );
                        let (PackedKernelPath::Tiled { splits }
                        | PackedKernelPath::PerRow { splits }
                        | PackedKernelPath::NaxTiled { splits }) = path;
                        assert_eq!(kernel.planned_splits(&args).unwrap(), splits);
                        let descriptor = kernel.kernel_descriptor(query_len, width, dtype).unwrap();
                        assert_eq!(descriptor.kernel, expected, "{context}");
                        if multi_row {
                            let nax_selection = kernel.nax_selection(width, dtype);
                            assert_eq!(descriptor.selection, nax_selection.selection, "{context}");
                            assert_eq!(
                                descriptor.selection_reason, nax_selection.reason,
                                "{context}"
                            );
                        }
                        // The dispatch-time selection the cache records is this descriptor's.
                        let recorded = kernel.kernel_selection(&args).unwrap();
                        assert_eq!(
                            (recorded.kernel, recorded.selection, recorded.reason),
                            (
                                descriptor.kernel,
                                descriptor.selection,
                                descriptor.selection_reason
                            ),
                            "{context}"
                        );
                        assert_eq!(
                            Some(recorded.query_dtype),
                            crate::primitives::packed_query_dtype_name(dtype)
                        );
                        assert!(crate::primitives::packed_kernel_path_valid(
                            kernel.gpu_family().as_str(),
                            recorded.kernel,
                            recorded.selection,
                            recorded.query_dtype
                        ));
                        let readback = |array: Array| {
                            host_readback::<f32>(&array.as_dtype(Dtype::Float32).unwrap())
                        };
                        let routed = readback(kernel.dispatch(&args).unwrap());
                        let named = readback(match path {
                            PackedKernelPath::Tiled { splits } => {
                                kernel.dispatch_tiled(&args, Some(splits)).unwrap()
                            }
                            PackedKernelPath::PerRow { splits } => {
                                kernel.dispatch_with_splits(&args, Some(splits)).unwrap()
                            }
                            PackedKernelPath::NaxTiled { splits } => {
                                kernel.dispatch_nax(&args, Some(splits)).unwrap()
                            }
                        });
                        assert_eq!(routed, named, "{context} {path:?}");
                        assert_eq!(
                            kernel.dispatch_nax(&args, None).is_ok(),
                            kernel_nax && width == 128 && dtype == Dtype::Bfloat16,
                            "{context}: the NAX seam follows nax_selection"
                        );
                    }
                    assert!(conservative.dispatch_tiled(&args, None).is_err());
                }
            }
        }
    }

    /// Chunked prefill through the decoder route: after the dense first chunk, each later
    /// multi-row chunk attends over the packed history plus its own fresh K/V through the tiled
    /// reader — the NAX tiled kernel for bf16 activations where MLX reports the Neural Accelerator,
    /// the fp32 tiled kernel for f32 — matches the fp32 oracle over the device representation, and
    /// never reconstructs the cache densely.
    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn decoder_chunked_prefill_uses_the_tiled_reader_without_dense_reconstruction() {
        use crate::primitives::packed_attention::PackedAttentionShape;
        use crate::primitives::packed_metal::{
            mlx_nax_available, PackedKernelPath, PackedMask, PackedMetalGpuFamily,
            PackedMetalKernel,
        };
        for bits in PackedCodeBits::ALL {
            let (heads, query_heads, width) = (2, 6, 128);
            let nax = mlx_nax_available().unwrap();
            for (dtype, tolerance) in [(Dtype::Float32, 1e-4), (Dtype::Bfloat16, 8e-2)] {
                let kernel = Arc::new(
                    PackedMetalKernel::for_identity_family_and_bits(
                        "decoder-tiled",
                        PackedMetalGpuFamily::Apple7OrNewer,
                        bits,
                    )
                    .unwrap(),
                );
                let reader = CompiledKernelHandle::new(kernel.clone());
                let mut cache = select_decoder_cache_with_reader(
                    PackedCacheRequest {
                        enabled: true,
                        backend: "mlx-metal".into(),
                        identity: "decoder-tiled".into(),
                        layers: 1,
                        batch: 1,
                        kv_heads: heads,
                        head_dimension: width,
                        group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                        bits,
                        query_length: 40,
                        has_mask: false,
                    },
                    reader,
                )
                .into_cache();
                let scale = (width as f32).powf(-0.5);
                let mut offset = 0;
                for (chunk, rows) in [40usize, 33, 128].into_iter().enumerate() {
                    let fresh = |seed: u64| {
                        bhsd(
                            &pseudo_random_outliers(1, heads, rows, width, seed)
                                .into_iter()
                                .map(|value| value.clamp(-3.0, 3.0))
                                .collect::<Vec<_>>(),
                            1,
                            heads,
                            rows,
                            width,
                        )
                        .as_dtype(dtype)
                        .unwrap()
                    };
                    let (keys, values) = (fresh(50 + chunk as u64), fresh(60 + chunk as u64));
                    let query = bhsd(
                        &(0..query_heads * rows * width)
                            .map(|i| ((i * 13 + chunk) % 41) as f32 * 0.03 - 0.6)
                            .collect::<Vec<_>>(),
                        1,
                        query_heads,
                        rows,
                        width,
                    )
                    .as_dtype(dtype)
                    .unwrap();
                    let query_values =
                        host_readback::<f32>(&query.as_dtype(Dtype::Float32).unwrap());
                    let output = cache
                        .try_packed_attention(
                            0,
                            &query,
                            &keys,
                            &values,
                            PackedAttentionMask::Causal,
                            scale,
                            false,
                        )
                        .unwrap()
                        .unwrap();
                    offset += rows;
                    if chunk == 0 {
                        continue;
                    }
                    let packed = cache
                        .as_any_mut()
                        .downcast_mut::<DenseFallbackPackedDecoderCache>()
                        .unwrap();
                    let reference = tiled_oracle(
                        &packed.staged,
                        &query_values,
                        PackedAttentionShape {
                            batch: 1,
                            query_heads,
                            kv_heads: heads,
                            query_len: rows,
                            kv_len: offset,
                            head_dim: width,
                        },
                        PackedAttentionMask::Causal,
                    );
                    let output = host_readback::<f32>(&output.as_dtype(Dtype::Float32).unwrap());
                    for (actual, expected) in output.iter().zip(&reference) {
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{dtype:?}: {actual} != {expected}"
                        );
                    }
                    // The step ran the planned multi-row kernel: its output is bit-identical to a
                    // direct dispatch of the kernel planned_path names over the same representation.
                    let staged = packed.staged.staged_reader_arguments(0).unwrap();
                    let args = staged.args(&query, PackedMask::Causal);
                    let direct = match kernel.planned_path(&args).unwrap() {
                        PackedKernelPath::NaxTiled { splits }
                            if dtype == Dtype::Bfloat16 && nax =>
                        {
                            kernel.dispatch_nax(&args, Some(splits)).unwrap()
                        }
                        PackedKernelPath::Tiled { splits } if dtype == Dtype::Float32 || !nax => {
                            kernel.dispatch_tiled(&args, Some(splits)).unwrap()
                        }
                        path => panic!("{dtype:?} chunk {chunk} of {rows} rows planned {path:?}"),
                    };
                    assert_eq!(
                        output,
                        host_readback::<f32>(&direct.as_dtype(Dtype::Float32).unwrap()),
                        "{dtype:?} chunk {chunk} output is not the planned kernel's"
                    );
                }
                let evidence = cache.packed_evidence().unwrap();
                assert_eq!(evidence.accepted_direct_calls, 2);
                // Both packed chunks are recorded under the path they actually ran.
                let (kernel, selection, query_dtype) = match (dtype, nax) {
                    (Dtype::Bfloat16, true) => (
                        crate::primitives::PACKED_NAX_KERNEL,
                        crate::primitives::PACKED_SELECTION_NAX,
                        "bfloat16",
                    ),
                    (Dtype::Bfloat16, false) => (
                        crate::primitives::PACKED_TILED_KERNEL,
                        crate::primitives::PACKED_SELECTION_NAX_UNAVAILABLE,
                        "bfloat16",
                    ),
                    _ => (
                        crate::primitives::PACKED_TILED_KERNEL,
                        if nax {
                            crate::primitives::PACKED_SELECTION_F32_QUERY
                        } else {
                            crate::primitives::PACKED_SELECTION_NAX_UNAVAILABLE
                        },
                        "float32",
                    ),
                };
                assert_eq!(
                    evidence
                        .kernel_paths
                        .iter()
                        .map(|path| (
                            path.kernel.as_str(),
                            path.selection.as_str(),
                            path.query_dtype.as_str(),
                            path.calls
                        ))
                        .collect::<Vec<_>>(),
                    vec![(kernel, selection, query_dtype, 2)],
                    "{dtype:?}"
                );
                assert_eq!(evidence.full_cache_dequantizations, 0);
                assert!(!evidence.dense_active && evidence.fallback_reasons.is_empty());
            }
        }
    }
}
