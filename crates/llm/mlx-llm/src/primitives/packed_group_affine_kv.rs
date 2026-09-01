//! CPU reference for the physically packed 2-bit group-affine KV representation (SC-20675).
//!
//! This module deliberately has no MLX dependency in its implementation.  It is the deterministic
//! storage/lifecycle seam that a later Metal reader can adopt: codes are four 2-bit values per byte,
//! while each group has an f16 scale and zero.  A dense reader is an explicit, instrumented fallback;
//! this type never retains a dense mirror.

use std::any::Any;
use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::primitives::kv_cache::{CacheRoute, ContiguousKvCache, KvCache, PackedCacheEvidence};
use half::f16;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

const MAGIC: &[u8; 8] = b"SW20675\0";
const VERSION: u32 = 2;
const BITS: u8 = 2;
/// Four independent 2-bit codes fit in one byte. This is a packing property, not the affine
/// quantization group selected by the qualified candidate.
pub const PACKED_CODES_PER_BYTE: usize = 4;
/// SC-20673 qualified the `b=2,g=32` group-affine candidate. K groups span tokens and V groups span
/// channels, but both use this same quantization group width.
pub const PACKED_METAL_QUANT_GROUP_SIZE: usize = 32;

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
    /// Total cache-attributable packed bytes: host vector capacity plus live device-array payload.
    /// Backend allocator overhead and shared pools require process-level measurement.
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
    /// End-to-end first successful packed dispatch latency, including packed argument sync,
    /// immutable MLX concatenation, lazy pipeline compilation, execution, and synchronization.
    /// This is deliberately not presented as compile-only time because MLX does not expose that
    /// boundary independently.
    pub cold_elapsed_ms: f64,
    /// Sum of the same end-to-end latency for successful dispatches after the first.
    pub steady_elapsed_ms: f64,
    /// Cumulative packed payload staged from host for physical attempts, including attempts whose
    /// reader later fails. Packed-domain device-to-device concatenation is not host upload traffic.
    pub uploaded_packed_bytes: u64,
    /// Subset of `uploaded_packed_bytes` belonging to accepted calls in committed model steps.
    /// Whole-step rollback restores this counter while leaving physical upload evidence intact.
    pub accepted_uploaded_packed_bytes: u64,
    /// Logical payload currently retained by the cache-owned device arrays. Backend allocator
    /// overhead and shared pools remain process-level receipt measurements.
    pub retained_device_packed_logical_bytes: u64,
    /// Largest packed argument payload presented to one kernel dispatch. This is not allocator
    /// usage; the sealed process harness measures the actual transient high-water mark.
    pub peak_packed_argument_logical_bytes: u64,
    /// Conservative cache-attributable high-water mark while immutable MLX arrays coexist during
    /// append/concatenate and padded-tail argument construction. Process-wide allocator receipts
    /// remain authoritative, but this value prevents retained bytes from being mislabeled as peak.
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
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        query: &Array,
        k_codes: &Array,
        k_scale: &Array,
        k_zero: &Array,
        v_codes: &Array,
        v_scale: &Array,
        v_zero: &Array,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> Result<Array>;
}

/// Lifetime-owned, type-erased compiled-kernel slot.  `Arc` keeps the real backend object alive
/// across cache clones/snapshots without relying on mutable device/queue/context labels.
#[derive(Clone)]
pub struct CompiledKernelHandle {
    inner: Arc<dyn RetainedPackedKernel>,
}

impl CompiledKernelHandle {
    pub fn new(inner: Arc<dyn RetainedPackedKernel>) -> Self {
        Self { inner }
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
}

impl fmt::Debug for CompiledKernelHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledKernelHandle")
            .field("cache_identity", &self.cache_identity())
            .field("backend", &self.backend())
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

    fn dispatch(
        &self,
        _query: &Array,
        _k_codes: &Array,
        _k_scale: &Array,
        _k_zero: &Array,
        _v_codes: &Array,
        _v_scale: &Array,
        _v_zero: &Array,
        _mask: crate::primitives::packed_metal::PackedMask,
    ) -> Result<Array> {
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

/// Decoder-facing owner for the experimental storage and retained fused reader. It publishes a
/// packed result only after the whole call succeeds; unsupported semantics transition the exact
/// evaluated history to dense before the caller performs its ordinary update.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AcceptedDispatchSnapshot {
    direct_dispatches: usize,
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
    device_layers: Vec<Option<DevicePackedLayer>>,
    accepted_dispatch: AcceptedDispatchSnapshot,
}

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

    fn pending_device_snapshot_overhead(&self) -> u64 {
        self.pending_step.as_ref().map_or(0, |pending| {
            self.staged
                .device_layers_logical_bytes(&pending.device_layers[..pending.next_layer])
                as u64
        })
    }

    /// Immutable receipt evidence at the same public object used by `CausalLm`.
    pub fn model_evidence(&self) -> PackedCacheEvidence {
        let telemetry = self.staged.dispatch_telemetry();
        let representation = self.staged.representation();
        PackedCacheEvidence {
            representation_identity: representation.identity,
            representation_version: representation.version,
            bits: representation.bits,
            quantization_group_size: representation.group_size,
            accepted_direct_calls: self.staged.direct_dispatches(),
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
            retained_device_packed_logical_bytes: telemetry
                .retained_device_packed_logical_bytes
                .saturating_add(self.pending_device_snapshot_overhead()),
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
        if let Err(error) = self.staged.trim(pending.original_len) {
            self.pending_step = Some(pending);
            return Err(error);
        }
        self.staged.device_layers = pending.device_layers;
        self.staged.telemetry.retained_device_packed_logical_bytes =
            self.staged.retained_device_packed_logical_bytes() as u64;
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
        let width = self.staged.head_dimension;
        let mut reconstructed = false;
        for (layer_index, layer) in self.staged.layers.iter().enumerate() {
            let Some(layer) = layer.as_ref() else {
                continue;
            };
            let tokens = layer.keys.logical_tokens();
            if tokens == 0 {
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
            // Incomplete K token groups are padded and quantized transiently for the reader. Rebuild
            // those exact evaluated values, not the denser append-friendly pending tail.
            let (evaluated_tokens, keys, values) =
                self.staged.evaluated_dense_layer(layer_index)?;
            debug_assert_eq!(evaluated_tokens, tokens);
            let shape = mlx_shape([self.staged.batch, self.staged.kv_heads, tokens, width])?;
            let keys = Array::from_slice(&keys, &shape).as_dtype(key_dtype)?;
            let values = Array::from_slice(&values, &shape).as_dtype(value_dtype)?;
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
            let keys = keys.as_dtype(mlx_rs::Dtype::Float32)?;
            let values = values.as_dtype(mlx_rs::Dtype::Float32)?;
            keys.eval()?;
            values.eval()?;

            // Mutate the packed cache in place and retain only the pre-step logical length for
            // rollback. Cloning the full packed history once per token would make host work O(S^2).
            if layer == 0 {
                self.pending_step = Some(PendingPackedStep {
                    original_len,
                    next_layer: 0,
                    step,
                    device_layers: self.staged.device_layers.clone(),
                    accepted_dispatch: self.staged.accepted_dispatch_snapshot(),
                });
            }
            self.staged.append(
                layer,
                keys.as_slice::<f32>(),
                values.as_slice::<f32>(),
                step,
            )?;
            let snapshot_overhead = self.pending_device_snapshot_overhead();
            let output = match self.staged.dispatch_packed_with_transient_overhead(
                layer,
                query,
                packed_mask,
                snapshot_overhead,
            ) {
                Ok(output) => output,
                Err(Error::Canceled) => return Err(Error::Canceled),
                Err(error) if layer == 0 && original_len == 0 => {
                    // Before any packed output is published, a device/JIT fault can still select
                    // the ordinary dense path result-equivalently for this complete model step.
                    self.rollback_pending("dispatch-fault", error.to_string())?;
                    self.staged.handle = None;
                    self.reason = format!("packed dispatch unavailable: {error}");
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            self.packed_layer_dtypes[layer] = Some((key_dtype, value_dtype));
            if layer + 1 == self.staged.layers() {
                if self.staged.logical_len() != expected_cache_len {
                    return Err(Error::Msg(
                        "packed whole-step transaction did not commit".into(),
                    ));
                }
                self.pending_step = None;
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

    fn num_layers(&self) -> usize {
        self.dense.num_layers()
    }

    fn prepare_dense_fallback(&mut self, operation: &str, reason: &str) -> Result<()> {
        self.transition_to_dense(operation, reason)
    }

    fn packed_evidence(&self) -> Option<PackedCacheEvidence> {
        Some(self.model_evidence())
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
                .trim(pending.original_len)
                .expect("pending packed rollback remains in bounds");
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
    let staged = match PackedGroupAffineKvCache::new(
        request.identity,
        request.layers,
        request.batch,
        request.kv_heads,
        request.head_dimension,
        request.group_size,
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

#[derive(Clone, Debug)]
struct PackedTensor {
    rows: usize,
    width: usize,
    groups: usize,
    codes: Vec<u8>,
    scales: Vec<f16>,
    zeros: Vec<f16>,
}

impl PackedTensor {
    fn new(rows: usize, width: usize, group_size: usize, capacity_rows: usize) -> Self {
        let groups = width.div_ceil(group_size);
        Self {
            rows,
            width,
            groups,
            codes: Vec::with_capacity(capacity_rows * width.div_ceil(PACKED_CODES_PER_BYTE)),
            scales: Vec::with_capacity(capacity_rows * groups),
            zeros: Vec::with_capacity(capacity_rows * groups),
        }
    }

    fn append(&mut self, values: &[f32], group_size: usize) -> Result<()> {
        if values.len() != self.width {
            return Err(Error::Config("packed KV row width mismatch".into()));
        }
        let code_start = self.codes.len();
        self.codes
            .resize(code_start + self.width.div_ceil(PACKED_CODES_PER_BYTE), 0);
        for group in 0..self.groups {
            let start = group * group_size;
            let end = (start + group_size).min(self.width);
            let slice = &values[start..end];
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((max - min) / 3.0).max(f32::EPSILON);
            self.scales.push(f16::from_f32(scale));
            self.zeros.push(f16::from_f32(min));
            for (i, value) in slice.iter().copied().enumerate() {
                let code = ((value - min) / scale).round().clamp(0.0, 3.0) as u8;
                let index = start + i;
                self.codes[code_start + index / PACKED_CODES_PER_BYTE] |=
                    code << ((index % PACKED_CODES_PER_BYTE) * 2);
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
        for i in 0..self.width {
            let code = (self.codes
                [row * self.width.div_ceil(PACKED_CODES_PER_BYTE) + i / PACKED_CODES_PER_BYTE]
                >> ((i % PACKED_CODES_PER_BYTE) * 2))
                & 3;
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
        self.codes
            .truncate(rows * self.width.div_ceil(PACKED_CODES_PER_BYTE));
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
            rows.saturating_mul(self.width.div_ceil(PACKED_CODES_PER_BYTE))
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
/// physically 2-bit codes plus f16 scale/zero metadata.
#[derive(Clone, Debug)]
struct TokenGroupKeyTensor {
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
    fn new(rows: usize, width: usize, group_size: usize, capacity_tokens: usize) -> Self {
        let groups = capacity_tokens.div_ceil(group_size);
        Self {
            rows,
            width,
            group_size,
            complete_tokens: 0,
            pending_tokens: 0,
            codes: Vec::with_capacity(
                rows * groups * (group_size * width).div_ceil(PACKED_CODES_PER_BYTE),
            ),
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
        (self.group_size * self.width).div_ceil(PACKED_CODES_PER_BYTE)
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
        for row in 0..self.rows {
            for channel in 0..self.width {
                let min = (0..self.group_size)
                    .map(|token| self.pending[(token * self.rows + row) * self.width + channel])
                    .fold(f32::INFINITY, f32::min);
                let max = (0..self.group_size)
                    .map(|token| self.pending[(token * self.rows + row) * self.width + channel])
                    .fold(f32::NEG_INFINITY, f32::max);
                let scale = ((max - min) / 3.0).max(f32::EPSILON);
                self.scales.push(f16::from_f32(scale));
                self.zeros.push(f16::from_f32(min));
                for token in 0..self.group_size {
                    let value = self.pending[(token * self.rows + row) * self.width + channel];
                    let code = ((value - min) / scale).round().clamp(0.0, 3.0) as u8;
                    let code_index = (token * self.width) + channel;
                    self.codes[code_start
                        + row * code_bytes_per_group
                        + code_index / PACKED_CODES_PER_BYTE] |=
                        code << ((code_index % PACKED_CODES_PER_BYTE) * 2);
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
            let code_index = local_token * self.width + channel;
            let code = (self.codes[code_base + code_index / PACKED_CODES_PER_BYTE]
                >> ((code_index % PACKED_CODES_PER_BYTE) * 2))
                & 3;
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

/// Evaluated MLX packed prefixes. The CPU representation is append-friendly and physically ordered
/// `[group_or_token,row,payload]`; the Metal contract is row-major
/// `[batch,head,group_or_token,payload]`. Monotonic extension uploads only new chunks; trim/restore
/// rebuilds the retained prefix and telemetry counts that historical upload. MLX arrays are
/// immutable, so extension uses a packed-domain concatenate whose real transient allocator peak is
/// measured by the sealed process harness.
#[derive(Clone, Debug)]
struct DevicePackedLayer {
    key_groups: usize,
    value_tokens: usize,
    key_codes: Option<Array>,
    key_scales: Option<Array>,
    key_zeros: Option<Array>,
    value_codes: Option<Array>,
    value_scales: Option<Array>,
    value_zeros: Option<Array>,
}

struct StagedPackedArguments {
    device_layer: DevicePackedLayer,
    key_codes: Array,
    key_scales: Array,
    key_zeros: Array,
    value_codes: Array,
    value_scales: Array,
    value_zeros: Array,
    uploaded_bytes: u64,
    argument_bytes: u64,
    transient_bytes: u64,
}

impl DevicePackedLayer {
    fn empty() -> Self {
        Self {
            key_groups: 0,
            value_tokens: 0,
            key_codes: None,
            key_scales: None,
            key_zeros: None,
            value_codes: None,
            value_scales: None,
            value_zeros: None,
        }
    }
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

fn joined_device_axis(prefix: Option<&Array>, chunk: Array) -> Result<Array> {
    chunk.eval()?;
    let joined = match prefix {
        Some(prefix) => concatenate_axis(&[prefix, &chunk], 2)?,
        None => chunk,
    };
    joined.eval()?;
    Ok(joined)
}

#[derive(Clone, Debug)]
pub struct PackedGroupAffineKvCache {
    identity: String,
    group_size: usize,
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
    telemetry: PackedDispatchTelemetry,
}

impl PackedGroupAffineKvCache {
    pub fn new(
        identity: impl Into<String>,
        layers: usize,
        batch: usize,
        kv_heads: usize,
        head_dimension: usize,
        group_size: usize,
    ) -> Result<Self> {
        if group_size == 0 || head_dimension == 0 || batch == 0 || kv_heads == 0 || layers == 0 {
            return Err(Error::Config("invalid packed KV shape".into()));
        }
        let _ = mlx_shape([batch, kv_heads, 1, head_dimension])?;
        Ok(Self {
            identity: identity.into(),
            group_size,
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
        let rows = self.rows();
        let slot = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| Error::Config("layer out of range".into()))?;
        if slot.is_none() {
            *slot = Some(LayerStorage {
                keys: TokenGroupKeyTensor::new(rows, width, group_size, cap),
                values: PackedTensor::new(self.logical_len * rows, width, group_size, rows * cap),
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
        for layer in self.layers.iter_mut().flatten() {
            layer.keys.truncate(len)?;
            layer.values.truncate(len * rows);
        }
        self.logical_len = len;
        for device in &mut self.device_layers {
            if device.as_ref().is_some_and(|device| {
                device.value_tokens > len || device.key_groups > len.div_ceil(self.group_size)
            }) {
                *device = None;
            }
        }
        self.telemetry.retained_device_packed_logical_bytes =
            self.retained_device_packed_logical_bytes() as u64;
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
    pub fn allocated_len(&self) -> usize {
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

    /// Actual allocated payload capacity for codes, metadata, and the pending key tail.
    pub fn host_allocated_payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.keys.allocated_bytes() + l.values.allocated_bytes())
            .sum()
    }

    /// Cache-attributable packed bytes retained across host vectors and live MLX arrays.
    pub fn allocated_payload_bytes(&self) -> usize {
        self.host_allocated_payload_bytes()
            .saturating_add(self.retained_device_packed_logical_bytes())
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
            bits: BITS,
            batch: self.batch,
            kv_heads: self.kv_heads,
            head_dimension: self.head_dimension,
            logical_len: self.logical_len,
            capacity: self.capacity,
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
        let l = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Config("layer is not resident".into()))?;
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

    /// Build and evaluate a complete six-array successor without publishing it. The caller owns
    /// the transaction boundary and may publish only after every downstream fallible operation
    /// which depends on these arrays has succeeded.
    fn stage_device_layer(&self, layer: usize) -> Result<(DevicePackedLayer, u64, u64)> {
        let storage = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Config("packed layer is not resident".into()))?;
        let rows = self.rows();
        let complete_groups = storage.keys.complete_groups();
        let value_tokens = storage.keys.logical_tokens();
        let key_bytes = storage.keys.code_bytes_per_group();
        let value_bytes = self.head_dimension.div_ceil(PACKED_CODES_PER_BYTE);
        let value_groups = self.head_dimension.div_ceil(self.group_size);
        let resident = self.device_layers.get(layer).and_then(Option::as_ref);
        let rebuild = resident.is_some_and(|device| {
            device.key_groups > complete_groups || device.value_tokens > value_tokens
        });
        let current = if rebuild {
            DevicePackedLayer::empty()
        } else {
            resident.cloned().unwrap_or_else(DevicePackedLayer::empty)
        };
        let current_bytes = self.device_layer_logical_bytes(&current);
        let (old_key_groups, old_value_tokens) = (current.key_groups, current.value_tokens);

        let key_delta = if old_key_groups < complete_groups {
            let groups = complete_groups - old_key_groups;
            let code_range = old_key_groups * rows * key_bytes..complete_groups * rows * key_bytes;
            let metadata_range = old_key_groups * rows * self.head_dimension
                ..complete_groups * rows * self.head_dimension;
            let shape_codes = mlx_shape([self.batch, self.kv_heads, groups, key_bytes])?;
            let shape_metadata =
                mlx_shape([self.batch, self.kv_heads, groups, self.head_dimension])?;
            Some((
                Array::from_slice(
                    &row_major_outer_rows(
                        &storage.keys.codes[code_range],
                        groups,
                        rows,
                        key_bytes,
                    )?,
                    &shape_codes,
                ),
                Array::from_slice(
                    &row_major_outer_rows(
                        &storage.keys.scales[metadata_range.clone()],
                        groups,
                        rows,
                        self.head_dimension,
                    )?,
                    &shape_metadata,
                ),
                Array::from_slice(
                    &row_major_outer_rows(
                        &storage.keys.zeros[metadata_range],
                        groups,
                        rows,
                        self.head_dimension,
                    )?,
                    &shape_metadata,
                ),
            ))
        } else {
            None
        };
        let value_delta = if old_value_tokens < value_tokens {
            let tokens = value_tokens - old_value_tokens;
            let code_range =
                old_value_tokens * rows * value_bytes..value_tokens * rows * value_bytes;
            let metadata_range =
                old_value_tokens * rows * value_groups..value_tokens * rows * value_groups;
            let shape_codes = mlx_shape([self.batch, self.kv_heads, tokens, value_bytes])?;
            let shape_metadata = mlx_shape([self.batch, self.kv_heads, tokens, value_groups])?;
            Some((
                Array::from_slice(
                    &row_major_outer_rows(
                        &storage.values.codes[code_range],
                        tokens,
                        rows,
                        value_bytes,
                    )?,
                    &shape_codes,
                ),
                Array::from_slice(
                    &row_major_outer_rows(
                        &storage.values.scales[metadata_range.clone()],
                        tokens,
                        rows,
                        value_groups,
                    )?,
                    &shape_metadata,
                ),
                Array::from_slice(
                    &row_major_outer_rows(
                        &storage.values.zeros[metadata_range],
                        tokens,
                        rows,
                        value_groups,
                    )?,
                    &shape_metadata,
                ),
            ))
        } else {
            None
        };

        let uploaded_key_groups = complete_groups - old_key_groups;
        let uploaded_value_tokens = value_tokens - old_value_tokens;
        let uploaded_bytes =
            uploaded_key_groups
                .saturating_mul(rows)
                .saturating_mul(
                    key_bytes.saturating_add(
                        self.head_dimension
                            .saturating_mul(2 * std::mem::size_of::<f16>()),
                    ),
                )
                .saturating_add(uploaded_value_tokens.saturating_mul(rows).saturating_mul(
                    value_bytes.saturating_add(
                        value_groups.saturating_mul(2 * std::mem::size_of::<f16>()),
                    ),
                ));

        // Build and evaluate a complete replacement off to the side. Publishing arrays one by one
        // would let a concatenate/eval fault drop or length-skew an otherwise valid resident prefix.
        let mut next = current.clone();
        if let Some((codes, scales, zeros)) = key_delta {
            let next_codes = joined_device_axis(current.key_codes.as_ref(), codes)?;
            let next_scales = joined_device_axis(current.key_scales.as_ref(), scales)?;
            let next_zeros = joined_device_axis(current.key_zeros.as_ref(), zeros)?;
            next.key_codes = Some(next_codes);
            next.key_scales = Some(next_scales);
            next.key_zeros = Some(next_zeros);
            next.key_groups = complete_groups;
        }
        if let Some((codes, scales, zeros)) = value_delta {
            let next_codes = joined_device_axis(current.value_codes.as_ref(), codes)?;
            let next_scales = joined_device_axis(current.value_scales.as_ref(), scales)?;
            let next_zeros = joined_device_axis(current.value_zeros.as_ref(), zeros)?;
            next.value_codes = Some(next_codes);
            next.value_scales = Some(next_scales);
            next.value_zeros = Some(next_zeros);
            next.value_tokens = value_tokens;
        }
        let next_bytes = self.device_layer_logical_bytes(&next);
        let transient_bytes = current_bytes
            .saturating_add(uploaded_bytes)
            .saturating_add(next_bytes) as u64;
        Ok((next, uploaded_bytes as u64, transient_bytes))
    }

    fn device_layer_logical_bytes(&self, device: &DevicePackedLayer) -> usize {
        let rows = self.rows();
        let key_bytes = (self.group_size * self.head_dimension).div_ceil(PACKED_CODES_PER_BYTE);
        let value_bytes = self.head_dimension.div_ceil(PACKED_CODES_PER_BYTE);
        let value_groups = self.head_dimension.div_ceil(self.group_size);
        device
            .key_groups
            .saturating_mul(rows)
            .saturating_mul(
                key_bytes.saturating_add(
                    self.head_dimension
                        .saturating_mul(2 * std::mem::size_of::<f16>()),
                ),
            )
            .saturating_add(
                device.value_tokens.saturating_mul(rows).saturating_mul(
                    value_bytes.saturating_add(
                        value_groups.saturating_mul(2 * std::mem::size_of::<f16>()),
                    ),
                ),
            )
    }

    pub fn retained_device_packed_logical_bytes(&self) -> usize {
        self.device_layers_logical_bytes(&self.device_layers)
    }

    fn device_layers_logical_bytes(&self, layers: &[Option<DevicePackedLayer>]) -> usize {
        layers
            .iter()
            .flatten()
            .map(|device| self.device_layer_logical_bytes(device))
            .sum()
    }

    fn padded_pending_key_tensor(
        &self,
        storage: &LayerStorage,
    ) -> Result<Option<TokenGroupKeyTensor>> {
        let pending = storage.keys.pending_tokens;
        if pending == 0 {
            return Ok(None);
        }
        let rows = self.rows();
        let width = self.head_dimension;
        let mut input = vec![0.0f32; rows * pending * width];
        for row in 0..rows {
            for token in 0..pending {
                let source = (token * rows + row) * width;
                let target = (row * pending + token) * width;
                input[target..target + width]
                    .copy_from_slice(&storage.keys.pending[source..source + width]);
            }
        }
        let mut tail = TokenGroupKeyTensor::new(rows, width, self.group_size, self.group_size);
        tail.append(&input, pending)?;
        let mut last = vec![0.0f32; rows * width];
        for row in 0..rows {
            let source = (row * pending + pending - 1) * width;
            last[row * width..(row + 1) * width].copy_from_slice(&input[source..source + width]);
        }
        for _ in pending..self.group_size {
            tail.append(&last, 1)?;
        }
        Ok(Some(tail))
    }

    /// Dense row-major values numerically equivalent to the buffers presented to the packed reader.
    /// This is used only for an observable dense transition; successful packed dispatches never call
    /// it. In particular, the incomplete K tail is reconstructed from its padded group quantization.
    fn evaluated_dense_layer(&self, layer: usize) -> Result<(usize, Vec<f32>, Vec<f32>)> {
        let storage = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Config("packed layer is not resident".into()))?;
        let tokens = storage.keys.logical_tokens();
        let rows = self.rows();
        let width = self.head_dimension;
        let pending_tail = self.padded_pending_key_tensor(storage)?;
        let mut keys = Vec::with_capacity(rows * tokens * width);
        let mut values = Vec::with_capacity(rows * tokens * width);
        for row in 0..rows {
            for token in 0..tokens {
                let key = if token < storage.keys.complete_tokens {
                    storage.keys.row(token, row)?
                } else {
                    pending_tail
                        .as_ref()
                        .ok_or_else(|| Error::Msg("missing evaluated packed key tail".into()))?
                        .row(token - storage.keys.complete_tokens, row)?
                };
                keys.extend(key);
                values.extend(storage.values.row(token * rows + row, self.group_size)?);
            }
        }
        Ok((tokens, keys, values))
    }

    fn padded_pending_key_arrays(
        &self,
        storage: &LayerStorage,
    ) -> Result<Option<(Array, Array, Array, u64)>> {
        let Some(tail) = self.padded_pending_key_tensor(storage)? else {
            return Ok(None);
        };
        let rows = self.rows();
        let width = self.head_dimension;
        let key_bytes = tail.code_bytes_per_group();
        let code_shape = mlx_shape([self.batch, self.kv_heads, 1, key_bytes])?;
        let metadata_shape = mlx_shape([self.batch, self.kv_heads, 1, width])?;
        let codes = Array::from_slice(&tail.codes, &code_shape);
        let scales = Array::from_slice(&tail.scales, &metadata_shape);
        let zeros = Array::from_slice(&tail.zeros, &metadata_shape);
        codes.eval()?;
        scales.eval()?;
        zeros.eval()?;
        let uploaded_bytes = rows.saturating_mul(
            key_bytes.saturating_add(width.saturating_mul(2 * std::mem::size_of::<f16>())),
        ) as u64;
        Ok(Some((codes, scales, zeros, uploaded_bytes)))
    }

    /// Construct every evaluated packed argument and its six-array resident successor without
    /// publishing either device state or telemetry. This keeps padding/concatenation/evaluation
    /// failures observationally pure.
    fn stage_packed_mlx_arguments(&self, layer: usize) -> Result<StagedPackedArguments> {
        let (device, persistent_uploaded_bytes, sync_transient_bytes) =
            self.stage_device_layer(layer)?;
        let storage = self.layers[layer]
            .as_ref()
            .ok_or_else(|| Error::Config("packed layer is not resident".into()))?;
        let pending = self.padded_pending_key_arrays(storage)?;
        let pending_uploaded_bytes = pending.as_ref().map_or(0, |item| item.3);
        let combine_key = |prefix: &Option<Array>, tail: Option<&Array>| -> Result<Array> {
            let result = match (prefix, tail) {
                (Some(prefix), Some(tail)) => concatenate_axis(&[prefix, tail], 2)?,
                (Some(prefix), None) => prefix.clone(),
                (None, Some(tail)) => tail.clone(),
                (None, None) => return Err(Error::Config("packed key cache is empty".into())),
            };
            result.eval()?;
            Ok(result)
        };
        let key_codes = combine_key(&device.key_codes, pending.as_ref().map(|item| &item.0))?;
        let key_scale = combine_key(&device.key_scales, pending.as_ref().map(|item| &item.1))?;
        let key_zero = combine_key(&device.key_zeros, pending.as_ref().map(|item| &item.2))?;
        let value_codes = device
            .value_codes
            .clone()
            .ok_or_else(|| Error::Config("packed value codes are empty".into()))?;
        let value_scale = device
            .value_scales
            .clone()
            .ok_or_else(|| Error::Config("packed value scales are empty".into()))?;
        let value_zero = device
            .value_zeros
            .clone()
            .ok_or_else(|| Error::Config("packed value zeros are empty".into()))?;
        let key_groups = storage.keys.logical_tokens().div_ceil(self.group_size);
        let argument_bytes = key_groups
            .saturating_mul(self.rows())
            .saturating_mul(
                storage.keys.code_bytes_per_group().saturating_add(
                    self.head_dimension
                        .saturating_mul(2 * std::mem::size_of::<f16>()),
                ),
            )
            .saturating_add(
                storage
                    .keys
                    .logical_tokens()
                    .saturating_mul(self.rows())
                    .saturating_mul(
                        self.head_dimension
                            .div_ceil(PACKED_CODES_PER_BYTE)
                            .saturating_add(
                                self.head_dimension
                                    .div_ceil(self.group_size)
                                    .saturating_mul(2 * std::mem::size_of::<f16>()),
                            ),
                    ),
            );
        let current_layer_bytes = self.device_layers[layer]
            .as_ref()
            .map_or(0, |layer| self.device_layer_logical_bytes(layer));
        let successor_retained_bytes =
            self.retained_device_packed_logical_bytes()
                .saturating_sub(current_layer_bytes)
                .saturating_add(self.device_layer_logical_bytes(&device)) as u64;
        Ok(StagedPackedArguments {
            device_layer: device,
            key_codes,
            key_scales: key_scale,
            key_zeros: key_zero,
            value_codes,
            value_scales: value_scale,
            value_zeros: value_zero,
            uploaded_bytes: persistent_uploaded_bytes.saturating_add(pending_uploaded_bytes),
            argument_bytes: argument_bytes as u64,
            transient_bytes: sync_transient_bytes.max(
                successor_retained_bytes
                    .saturating_add(pending_uploaded_bytes)
                    .saturating_add(argument_bytes as u64),
            ),
        })
    }

    fn record_staged_physical_evidence(
        &mut self,
        staged: &StagedPackedArguments,
        additional_transient_bytes: u64,
    ) {
        self.telemetry.uploaded_packed_bytes = self
            .telemetry
            .uploaded_packed_bytes
            .saturating_add(staged.uploaded_bytes);
        self.telemetry.peak_packed_argument_logical_bytes = self
            .telemetry
            .peak_packed_argument_logical_bytes
            .max(staged.argument_bytes);
        self.telemetry.peak_packed_transient_logical_bytes =
            self.telemetry.peak_packed_transient_logical_bytes.max(
                staged
                    .transient_bytes
                    .saturating_add(additional_transient_bytes),
            );
    }

    fn publish_staged_device_layer(&mut self, layer: usize, device_layer: DevicePackedLayer) {
        self.device_layers[layer] = Some(device_layer);
        self.telemetry.retained_device_packed_logical_bytes =
            self.retained_device_packed_logical_bytes() as u64;
    }

    /// Internal source-test seam: publish only after every packed argument has evaluated. Live
    /// dispatch stages privately until the reader output also evaluates successfully.
    #[cfg(test)]
    fn packed_mlx_arguments(
        &mut self,
        layer: usize,
    ) -> Result<(Array, Array, Array, Array, Array, Array)> {
        let staged = self.stage_packed_mlx_arguments(layer)?;
        self.record_staged_physical_evidence(&staged, 0);
        self.publish_staged_device_layer(layer, staged.device_layer);
        Ok((
            staged.key_codes,
            staged.key_scales,
            staged.key_zeros,
            staged.value_codes,
            staged.value_scales,
            staged.value_zeros,
        ))
    }

    /// Run the retained reader against privately staged packed buffers. Physical attempt evidence
    /// is monotonic, while the six-array successor and accepted counters publish only after the
    /// output evaluates successfully.
    pub fn dispatch_packed(
        &mut self,
        layer: usize,
        query: &Array,
        mask: crate::primitives::packed_metal::PackedMask,
    ) -> Result<Array> {
        self.dispatch_packed_with_transient_overhead(layer, query, mask, 0)
    }

    fn dispatch_packed_with_transient_overhead(
        &mut self,
        layer: usize,
        query: &Array,
        mask: crate::primitives::packed_metal::PackedMask,
        additional_transient_bytes: u64,
    ) -> Result<Array> {
        if self.handle.is_none() {
            return Err(Error::Unsupported("no retained packed reader".into()));
        }
        let layer_len = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .map_or(0, |storage| storage.keys.logical_tokens());
        if layer_len == 0 {
            return Err(Error::Config(
                "cannot dispatch an empty packed cache".into(),
            ));
        }
        let dense_dequantizations_before = self.full_cache_dequantizations;
        let started = std::time::Instant::now();
        self.telemetry.dispatch_attempts += 1;
        let staged = match self.stage_packed_mlx_arguments(layer) {
            Ok(staged) => staged,
            Err(error) => {
                self.telemetry.failed_dispatches += 1;
                self.telemetry.attempted_elapsed_ms += started.elapsed().as_secs_f64() * 1000.0;
                return Err(error);
            }
        };
        self.record_staged_physical_evidence(&staged, additional_transient_bytes);
        let was_warmed = self.telemetry.kernel_warmed;
        if !was_warmed {
            self.telemetry.compile_jit_attempts += 1;
        }
        let result = self
            .handle
            .as_ref()
            .expect("checked above")
            .inner
            .dispatch(
                query,
                &staged.key_codes,
                &staged.key_scales,
                &staged.key_zeros,
                &staged.value_codes,
                &staged.value_scales,
                &staged.value_zeros,
                mask,
            )
            .and_then(|output| {
                output.eval()?;
                Ok(output)
            });
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.attempted_elapsed_ms += elapsed_ms;
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.telemetry.failed_dispatches += 1;
                return Err(error);
            }
        };
        self.telemetry.kernel_warmed = true;
        self.publish_staged_device_layer(layer, staged.device_layer);
        debug_assert_eq!(
            self.full_cache_dequantizations, dense_dequantizations_before,
            "accepted fused packed dispatch must not reconstruct the full cache"
        );
        self.direct_dispatches += 1;
        self.telemetry.accepted_uploaded_packed_bytes = self
            .telemetry
            .accepted_uploaded_packed_bytes
            .saturating_add(staged.uploaded_bytes);
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
            cold_dispatches: self.telemetry.cold_dispatches,
            steady_dispatches: self.telemetry.steady_dispatches,
            cold_elapsed_ms: self.telemetry.cold_elapsed_ms,
            steady_elapsed_ms: self.telemetry.steady_elapsed_ms,
            accepted_uploaded_packed_bytes: self.telemetry.accepted_uploaded_packed_bytes,
        }
    }
    fn restore_accepted_dispatch(&mut self, snapshot: AcceptedDispatchSnapshot) {
        self.direct_dispatches = snapshot.direct_dispatches;
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
    /// quantization, or identity before installing any state.
    pub fn save(&self) -> Vec<u8> {
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
        out.push(BITS);
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
        out
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
            || take(&mut p, 1)?[0] != BITS
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
                    value.checked_mul(
                        (self.group_size * self.head_dimension).div_ceil(PACKED_CODES_PER_BYTE),
                    )
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
                .checked_mul(self.head_dimension.div_ceil(PACKED_CODES_PER_BYTE))
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
    use crate::primitives::kv_cache::PackedAttentionMask;
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
        #[allow(clippy::too_many_arguments)]
        fn dispatch(
            &self,
            query: &Array,
            _k_codes: &Array,
            _k_scale: &Array,
            _k_zero: &Array,
            _v_codes: &Array,
            _v_scale: &Array,
            _v_zero: &Array,
            _mask: crate::primitives::packed_metal::PackedMask,
        ) -> Result<Array> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_on == Some(call) {
                if self.cancel_on_failure {
                    return Err(Error::Canceled);
                }
                return Err(Error::Msg("injected packed dispatch fault".into()));
            }
            Ok(query.clone())
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
        assert_eq!(actual.key_groups, expected.key_groups);
        assert_eq!(actual.value_tokens, expected.value_tokens);
        for (actual, expected) in [
            (&actual.key_codes, &expected.key_codes),
            (&actual.value_codes, &expected.value_codes),
        ] {
            assert_eq!(
                actual.as_ref().map(|array| array.as_slice::<u8>()),
                expected.as_ref().map(|array| array.as_slice::<u8>())
            );
        }
        for (actual, expected) in [
            (&actual.key_scales, &expected.key_scales),
            (&actual.key_zeros, &expected.key_zeros),
            (&actual.value_scales, &expected.value_scales),
            (&actual.value_zeros, &expected.value_zeros),
        ] {
            assert_eq!(
                actual.as_ref().map(|array| array.as_slice::<f16>()),
                expected.as_ref().map(|array| array.as_slice::<f16>())
            );
        }
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
        let bytes = c.save();
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
        let snapshot = source.save();

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
        let trimmed_snapshot = restored.save();
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
        let snapshot = source.save();
        let mut incompatible = PackedGroupAffineKvCache::new("m", 1, 1, 1, 8, 3).unwrap();
        let before = incompatible.representation();
        assert!(incompatible.restore(&snapshot).is_err());
        assert_eq!(incompatible.representation(), before);
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
        let mut version = c.save();
        version[8] = 3;
        assert!(c.restore(&version).is_err());
        assert_eq!(c.representation(), before);
        let mut trailing = c.save();
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
        assert!(
            packed
                .packed_evidence()
                .unwrap()
                .retained_device_packed_logical_bytes
                > packed
                    .staged
                    .dispatch_telemetry()
                    .retained_device_packed_logical_bytes,
            "the public receipt includes the replaced pre-step arrays held for rollback"
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
    fn packed_argument_staging_is_observationally_pure_until_explicit_publication() {
        let mut cache = PackedGroupAffineKvCache::new(
            "staging-purity",
            1,
            2,
            2,
            64,
            PACKED_METAL_QUANT_GROUP_SIZE,
        )
        .unwrap();
        let values = bhst_data(2, 2, 3, 64, -1.5);
        cache.append(0, &values, &values, 3).unwrap();
        let prior_telemetry = cache.dispatch_telemetry();
        assert!(cache.device_layers[0].is_none());

        let staged = cache.stage_packed_mlx_arguments(0).unwrap();
        assert!(cache.device_layers[0].is_none());
        assert_eq!(cache.dispatch_telemetry(), prior_telemetry);
        assert_eq!(staged.device_layer.value_tokens, 3);
        assert!(staged.uploaded_bytes > 0);
        assert!(staged.argument_bytes > 0);
        assert!(staged.transient_bytes >= staged.argument_bytes);
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
            assert_eq!(
                packed.staged.layers[0]
                    .as_ref()
                    .unwrap()
                    .keys
                    .logical_tokens(),
                0
            );
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
        assert_eq!(packed.dense.peek(0).unwrap().0.shape()[2], 2);
        assert_eq!(packed.dense.peek(1).unwrap().0.shape()[2], 1);

        packed.update(1, &kv, &kv).unwrap();
        assert_eq!(packed.offset(), 2);
        assert_eq!(packed.dense.peek(1).unwrap().0.shape()[2], 2);
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
            assert_eq!(packed.dense.peek(0).unwrap().0.shape()[2], 1);
            assert!(packed.dense.peek(1).is_none());
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
        assert_eq!(packed.dense.peek(0).unwrap().0.shape()[2], 1);
        assert_eq!(packed.dense.peek(1).unwrap().0.shape()[2], 1);
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
        let value_bytes = width.div_ceil(PACKED_CODES_PER_BYTE);
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
        assert_eq!(device.key_groups, groups);
        assert_eq!(device.value_tokens, step);
        let initial = cache.dispatch_telemetry();
        assert_eq!(
            initial.uploaded_packed_bytes,
            initial.retained_device_packed_logical_bytes
        );
        let initial_representation = cache.representation();
        assert_eq!(
            initial_representation.allocated_bytes,
            initial_representation.host_allocated_payload_bytes
                + initial_representation.retained_device_packed_logical_bytes
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
        assert!(cache.device_layers[0].is_none());
        let uploaded_before_rebuild = cache.dispatch_telemetry().uploaded_packed_bytes;
        cache.packed_mlx_arguments(0).unwrap();
        let after_trim = cache.dispatch_telemetry();
        let rebuilt = cache.device_layers[0].as_ref().unwrap();
        assert_eq!(rebuilt.key_groups, 1);
        assert_eq!(rebuilt.value_tokens, 5);
        assert!(
            after_trim.peak_packed_argument_logical_bytes
                > after_trim.retained_device_packed_logical_bytes,
            "the padded pending-key argument is transient and separately accounted"
        );
        let pending_key_bytes = batch * heads * (width + width * 2 * std::mem::size_of::<f16>());
        assert_eq!(
            after_trim.uploaded_packed_bytes - uploaded_before_rebuild,
            after_trim.retained_device_packed_logical_bytes + pending_key_bytes as u64,
            "trim rebuilds retained history and uploads the transient padded key tail explicitly"
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
    ) {
        use crate::primitives::attention::{sdpa, AttnMask};
        use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
        use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};

        const BATCH: usize = 2;
        const QUERY_LEN: usize = 2;
        const KV_LEN: usize = 5;
        let identity = format!(
            "oracle-{dtype:?}-{head_dimension}-{kv_heads}-{query_heads}-{mask:?}-{family:?}"
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

        let mut cache = PackedGroupAffineKvCache::new(
            identity.clone(),
            1,
            BATCH,
            kv_heads,
            head_dimension,
            PACKED_METAL_QUANT_GROUP_SIZE,
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
                PackedMetalKernel::for_identity_and_family(identity, family).unwrap(),
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
        assert!(
            telemetry.peak_packed_argument_logical_bytes
                > telemetry.retained_device_packed_logical_bytes
        );
        assert!(
            telemetry.peak_packed_transient_logical_bytes
                >= telemetry
                    .retained_device_packed_logical_bytes
                    .saturating_add(telemetry.peak_packed_argument_logical_bytes)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn real_metal_group32_matches_dense_sdpa_and_independent_oracle_across_full_surface() {
        use crate::primitives::packed_metal::PackedMetalGpuFamily;

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
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn real_metal_adapter_dispatch_matches_nonuniform_fp32_reference_with_gqa_and_tail() {
        use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};

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
        let mut cache =
            PackedGroupAffineKvCache::new("device-layout", 1, batch, kv_heads, width, group)
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
                PackedMetalKernel::for_identity("device-layout").unwrap(),
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

    #[test]
    fn deterministic_pseudorandom_outliers_pack_identically_across_arbitrary_chunks() {
        let (batch, heads, step, width, group) = (2, 3, 11, 7, 4);
        let keys = pseudo_random_outliers(batch, heads, step, width, 0x5eed_beef);
        let values = pseudo_random_outliers(batch, heads, step, width, 0x0123_4567);
        let mut one = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        one.append(0, &keys, &values, step).unwrap();
        let mut chunks = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
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

    #[test]
    fn rollback_reappend_clears_packed_tails_and_requantizes_cut_key_groups() {
        let (batch, heads, width, group) = (1, 2, 5, 4);
        let original_keys = bhst_data(batch, heads, 6, width, -1000.0);
        let original_values = bhst_data(batch, heads, 6, width, 1000.0);
        let replacement_keys = bhst_data(batch, heads, 3, width, 700.0);
        let replacement_values = bhst_data(batch, heads, 3, width, -700.0);
        let mut rolled = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
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
            PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        expected
            .append(0, &expected_keys, &expected_values, 6)
            .unwrap();
        assert_same_rows(&rolled, &expected);
        assert_eq!(rolled.logical_len(), 6);
    }

    #[test]
    fn pending_key_groups_snapshot_and_byte_accounting_are_strict() {
        let (batch, heads, step, width, group) = (2, 1, 5, 7, 3);
        let keys = bhst_data(batch, heads, step, width, 0.0);
        let mut values = bhst_data(batch, heads, step, width, 0.0);
        values[0] = -1000.0;
        let last = values.len() - 1;
        values[last] = 1000.0;
        let mut cache = PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
        cache.append(0, &keys, &values, step).unwrap();
        let bytes_before = cache.logical_stored_bytes();
        assert!(cache.allocated_vec_bytes() >= bytes_before);
        assert!(cache.process_visible_bytes_estimate() >= cache.allocated_vec_bytes());
        let snapshot = cache.save();
        let mut restored =
            PackedGroupAffineKvCache::new("m", 1, batch, heads, width, group).unwrap();
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
}
