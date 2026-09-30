//! Key/value cache.
//!
//! The cache is the seam the throughput work (P4) plugs into: the dynamic-batch scheduler
//! (story 7167), the prefix cache (7168) and the paged cache (7169/7170) all implement the
//! [`KvCache`] trait so they swap in without the decoder changing. The decoder only ever talks to
//! the trait.
//!
//! [`ContiguousKvCache`] is the day-one implementation: per layer, one K and one V buffer
//! preallocated in blocks of [`KV_BLOCK_TOKENS`] positions along the sequence axis, written **in
//! place** as tokens arrive (the mlx-lm `KVCache` pattern). It is **batch-capable** — the batch axis
//! is real, not hardcoded to 1, so an N-sequence batch with a uniform length works today. Ragged
//! per-sequence offsets (sequences of differing lengths in one batch) need the paged cache, which is
//! exactly why the trait exists.
//!
//! Why blocks rather than a per-token concat: MLX's allocator keeps freed buffers in a cache and
//! only reuses one for a request of the same page-rounded size. A concat-per-token cache makes every
//! per-layer buffer strictly larger than the last, so no freed buffer is ever reused and the cache
//! grows by roughly `layers × 2 × heads × head_dim × context` bytes per token — tens of GB over a
//! long generation (measured 68 GB after 801 tokens of an 8B model at 5.6k context). With block
//! preallocation a layer's buffer changes size once per [`KV_BLOCK_TOKENS`] tokens and the
//! in-between steps update it in place, so the freed-buffer set stays bounded.

use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
use mlx_rs::ops::{concatenate_axis, multiply, zeros_dtype};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Layout, per layer, of the cached keys/values: `[batch, n_kv_heads, seq, head_dim]`. Keys are
/// stored already-RoPE'd; values raw. The sequence axis (2) is the one that grows each step.
pub const SEQ_AXIS: i32 = 2;

/// Experimental representation routing is decided before cache mutation.  The dense route is the
/// compatibility default until a backend advertises every requested semantic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheRoute {
    DenseFallback { reason: String },
    ExperimentalPacked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedAttentionMask {
    None,
    Causal,
    SlidingWindow(usize),
    Additive,
}

/// Immutable evidence exported by an experimental packed cache at the same trait-object boundary
/// used by the decoder.  A sealed harness must not need private-field access or a storage-only test
/// path to prove that model calls stayed compressed-domain.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct PackedCacheEvidence {
    pub representation_identity: String,
    pub representation_version: u32,
    pub bits: u8,
    pub quantization_group_size: usize,
    pub accepted_direct_calls: usize,
    /// Accepted calls per kernel path actually dispatched, sorted by (kernel, selection, query
    /// dtype); on a reader that reports its selection these sum to `accepted_direct_calls`.
    #[serde(default)]
    pub kernel_paths: Vec<PackedKernelPathEvidence>,
    pub full_cache_dequantizations: usize,
    pub dispatch_attempts: u64,
    pub failed_dispatches: u64,
    pub compile_jit_attempts: u64,
    pub kernel_warmed: bool,
    pub attempted_elapsed_ms: f64,
    pub cold_dispatches: u64,
    pub steady_dispatches: u64,
    pub cold_elapsed_ms: f64,
    pub steady_elapsed_ms: f64,
    pub uploaded_packed_bytes: u64,
    pub accepted_uploaded_packed_bytes: u64,
    /// Cache-resident packed code arrays, derived from the live device representation rather than
    /// cumulative upload traffic.
    pub retained_device_code_bytes: u64,
    /// Cache-resident scale/zero arrays, derived from the live device representation rather than
    /// dispatch arguments or allocator high-water.
    pub retained_device_metadata_bytes: u64,
    pub retained_device_packed_logical_bytes: u64,
    pub peak_packed_argument_logical_bytes: u64,
    pub peak_packed_transient_logical_bytes: u64,
    pub dense_active: bool,
    pub fallback_reasons: Vec<(String, String)>,
}
/// One kernel path of the packed reader and the accepted calls that ran it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct PackedKernelPathEvidence {
    /// Kernel name (a `PACKED_*_KERNEL` constant).
    pub kernel: String,
    /// Selection token (a `PACKED_SELECTION_*` constant).
    pub selection: String,
    /// Human-readable reason for the selection.
    pub reason: String,
    /// Dispatched query dtype (`float32`, `float16`, `bfloat16`).
    pub query_dtype: String,
    pub calls: u64,
}

impl PackedKernelPathEvidence {
    /// Evidence rows for a cache's recorded paths, sorted by (kernel, selection, query dtype).
    pub fn sorted(
        paths: &[(crate::primitives::packed_metal::PackedKernelSelection, u64)],
    ) -> Vec<Self> {
        let mut rows = paths
            .iter()
            .map(|(path, calls)| Self {
                kernel: path.kernel.into(),
                selection: path.selection.into(),
                reason: path.reason.into(),
                query_dtype: path.query_dtype.into(),
                calls: *calls,
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| {
            (&a.kernel, &a.selection, &a.query_dtype).cmp(&(
                &b.kernel,
                &b.selection,
                &b.query_dtype,
            ))
        });
        rows
    }
}

/// Measured physical storage of a live compressed KV representation, exported only to campaign
/// observers (SC-20676 compressed rows). Device bytes are the sizes of the MLX arrays the cache
/// actually retains and host bytes its allocated staging payload, never bit accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompressedCacheStorage {
    /// Retained device code arrays.
    pub device_code_bytes: u64,
    /// Retained device scale/zero (or codebook) arrays, plus any bounded dense residual of
    /// not-yet-quantized recent tokens the representation keeps on the device.
    pub device_metadata_bytes: u64,
    /// Allocated host-side staging payload.
    pub host_payload_bytes: u64,
    /// Live cached tokens.
    pub tokens: u64,
    /// Scalar width of the K/V tensors handed to the cache (the dense-equivalent element width).
    pub element_bytes: u64,
}

impl CompressedCacheStorage {
    /// MLX-resident compressed bytes: the cache's contribution to MLX active memory.
    pub fn device_bytes(&self) -> u64 {
        self.device_code_bytes
            .saturating_add(self.device_metadata_bytes)
    }
}

/// Sequence positions a [`ContiguousKvCache`] buffer grows by at a time: the buffer is
/// reallocated once per this many tokens and written in place in between.
pub const KV_BLOCK_TOKENS: i32 = 256;

/// The decoder-facing cache contract.
///
/// A decoder, for each layer, hands the cache this step's keys/values and gets back the full
/// keys/values to attend over. Positional offset bookkeeping is the cache's job — [`KvCache::offset`]
/// reports how many positions are already cached (the RoPE offset for the next step), so the
/// decoder reads it once before the step rather than threading an `index_pos` through every call.
pub trait KvCache {
    /// Preflight an experimental representation without changing the cache.  Existing dense
    /// implementations retain their feature-off behavior through this default.
    fn preflight_packed(&self, _query_length: usize, _mask: bool) -> CacheRoute {
        CacheRoute::DenseFallback {
            reason: "experimental packed representation disabled".into(),
        }
    }
    /// Optional pre-update packed attention path. Returning `Some` means the cache appended the
    /// current K/V and produced attention output directly; callers must not call `update` or
    /// dense SDPA for that layer. The default preserves all existing cache implementations.
    #[allow(clippy::too_many_arguments)]
    fn try_packed_attention(
        &mut self,
        _layer: usize,
        _query: &Array,
        _keys: &Array,
        _values: &Array,
        _mask: PackedAttentionMask,
        _scale: f32,
        _retained_for_sharing: bool,
    ) -> Result<Option<Array>> {
        Ok(None)
    }

    /// Import a reused dense prefix — one `(keys, values)` pair per layer, each
    /// `[batch, n_kv_heads, seq, head_dim]` — into an empty cache's own representation. Returns
    /// `Ok(true)` when the cache now holds the prefix; the default declines (`Ok(false)`, nothing
    /// mutated), and the caller keeps its dense seed.
    fn import_prefix(&mut self, _layers: &[(Array, Array)]) -> Result<bool> {
        Ok(false)
    }

    /// Force an explicitly-reasoned dense transition before the caller performs its ordinary
    /// update. Dense and paged caches are already dense, so their default is a no-op.
    fn prepare_dense_fallback(&mut self, _operation: &str, _reason: &str) -> Result<()> {
        Ok(())
    }

    /// Model-boundary evidence for sealed packed-cache receipts.
    fn packed_evidence(&self) -> Option<PackedCacheEvidence> {
        None
    }

    /// Campaign-only measured storage of a live compressed representation. `None` for dense
    /// caches, an empty compressed cache, or while an explicit dense fallback owns the history.
    fn compressed_storage(&self) -> Result<Option<CompressedCacheStorage>> {
        Ok(None)
    }

    /// Campaign-only: the explicit dense fallback cache owned by a compressed representation.
    fn compressed_dense_fallback(&self) -> Option<&ContiguousKvCache> {
        None
    }
    /// Append `keys`/`values` for `layer` (each `[batch, n_kv_heads, step, head_dim]`) and return
    /// the full cached `(keys, values)` to attend over, same layout with the sequence axis grown.
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)>;

    /// Number of sequence positions currently cached — i.e. the RoPE offset for the next step.
    /// `0` before the first update. Inferred from layer 0 (all layers advance in lockstep).
    fn offset(&self) -> i32;

    /// Batch size of the cached tensors, or `0` before the first update.
    fn batch_size(&self) -> i32;

    /// Number of decoder layers this cache holds slots for.
    fn num_layers(&self) -> usize;

    /// Compact the batch to keep only the rows in `keep` (indices into the current batch axis),
    /// in the given order — the seam the dynamic-batch scheduler (story 7167) retires a finished
    /// sequence through, so the next step runs a smaller batch. A contiguous cache gathers the kept
    /// rows along the batch axis; the paged cache (7169) frees the dropped sequences' pages. `keep`
    /// must be a subset of `0..batch_size`; an empty cache is a no-op.
    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()>;

    /// Drop cached positions past `len`, keeping positions `0..len` along the sequence axis — the
    /// seam speculative decoding (story 7171) rolls back rejected draft tokens through. `len` must be
    /// `<= offset()`; `len == offset()` is a no-op and an empty cache ignores it.
    fn truncate(&mut self, len: i32) -> Result<()>;

    /// Drop all cached state, returning the cache to its freshly-constructed (empty) condition.
    fn reset(&mut self) -> Result<()>;

    /// Downcast hook so a decoder can recover its concrete cache from a `&mut dyn KvCache` — the
    /// hybrid Qwen3.6 cache (recurrent linear-attention state + KV) is driven natively rather than
    /// through the softmax-only [`KvCache::update`] path.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Producer-only ownership evidence emitted by the contiguous cache.  This is intentionally a
/// value type: campaign code can serialize it without exposing MLX arrays or changing the public
/// decoder contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheEvent {
    pub layer: usize,
    pub operation: &'static str,
    pub role: &'static str,
    pub lifetime: &'static str,
    pub bytes: u64,
    pub tokens: u64,
}

fn array_bytes(array: &Array) -> Result<u64> {
    u64::try_from(array.size())
        .ok()
        .and_then(|elements| elements.checked_mul(u64::try_from(array.item_size()).ok()?))
        .ok_or_else(|| crate::error::Error::Msg("KV shape overflows byte accounting".into()))
}

/// One layer's block-allocated K/V buffers plus how many leading positions are live.
///
/// `Clone` shares the buffers (MLX arrays are refcounted). A clone taken as a snapshot stays
/// intact while the original keeps being written: the in-place update below is MLX's
/// `slice_update`, which donates the buffer only when the cache holds the sole reference and
/// otherwise writes into a fresh copy — copy-on-write at block granularity, which is what the
/// speculative loops' `cache.clone()` rollback snapshots rely on.
#[derive(Clone, Debug)]
struct LayerSlot {
    /// `[batch, n_kv_heads, capacity, k_head_dim]`; positions `offset..capacity` are padding.
    keys: Array,
    /// `[batch, n_kv_heads, capacity, v_head_dim]`; positions `offset..capacity` are padding.
    values: Array,
    /// Live positions (`<= capacity`).
    offset: i32,
    /// Whether the buffers belong to someone else — a [`ContiguousKvCache::seeded`] slot holds the
    /// prefix store's entry by reference. Growing such a slot retires nothing this cache owns: the
    /// store keeps the old buffers alive, so no transient release is recorded for them.
    borrowed: bool,
}

impl LayerSlot {
    fn capacity(&self) -> i32 {
        self.keys.shape()[SEQ_AXIS as usize]
    }

    /// The live `[.., ..offset]` views the attention must see — nothing past `offset` is real.
    fn live(&self) -> Result<(Array, Array)> {
        Ok((
            self.keys.try_index((.., .., ..self.offset, ..))?,
            self.values.try_index((.., .., ..self.offset, ..))?,
        ))
    }
}

/// A row-contiguous copy of `a` holding only `a`'s own elements. A `[.., ..offset]` view keeps
/// the whole padded block buffer alive and, while it lives, forces the next in-place update to
/// copy rather than donate; anything that outlives the step (the prefix store) takes this instead.
/// `x * 1` is bit-exact for every float (unlike `x + 0`, which folds `-0.0`) and owns its buffer,
/// but keeps a permuted-dense input's strides; [`contiguous`] then makes it row-major.
///
/// [`contiguous`]: crate::primitives::nn::contiguous
fn materialize(a: &Array) -> Result<Array> {
    crate::primitives::nn::contiguous(&multiply(a, Array::from_f32(1.0).as_dtype(a.dtype())?)?)
}

/// Block-allocated KV cache: one `(K, V)` buffer pair per layer, grown by [`KV_BLOCK_TOKENS`]
/// positions at a time and written in place between growths. Correctness-first; the paged cache
/// (P4) is the throughput replacement behind the same trait.
///
/// `Clone` is a buffer-sharing snapshot: cheap to take, and the original may keep updating in
/// place without disturbing it (the in-place write copies rather than donates while another
/// reference to the block buffer is alive — copy-on-write at block granularity).
#[derive(Clone, Debug)]
pub struct ContiguousKvCache {
    layers: Vec<Option<LayerSlot>>,
    /// Growth granularity along the sequence axis.
    block: i32,
    events: Vec<CacheEvent>,
}

impl ContiguousKvCache {
    /// A fresh cache with `num_layers` empty slots, growing by [`KV_BLOCK_TOKENS`] positions.
    pub fn new(num_layers: usize) -> Self {
        Self::with_block_tokens(num_layers, KV_BLOCK_TOKENS)
    }

    /// A fresh cache growing by `block` positions at a time (`block >= 1`). [`new`](Self::new) is
    /// the production choice; a small block lets tests cross growth boundaries with tiny tensors.
    pub fn with_block_tokens(num_layers: usize, block: i32) -> Self {
        assert!(block >= 1, "kv cache block must be at least one position");
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            events: Vec::new(),
            block,
        }
    }

    /// The currently-cached `(keys, values)` for `layer` — views over the live positions only —
    /// or `None` if the layer is empty. The views share the layer's block buffer: hold them only
    /// for the current step, or the next in-place update has to copy the block instead of
    /// donating it.
    pub fn peek(&self, layer: usize) -> Result<Option<(Array, Array)>> {
        self.layers
            .get(layer)
            .and_then(|s| s.as_ref())
            .map(LayerSlot::live)
            .transpose()
    }

    /// Ownership events since construction, for the campaign producer only.
    pub fn events(&self) -> &[CacheEvent] {
        &self.events
    }

    /// Actual scalar width of the retained key/value arrays.
    pub fn element_bytes(&self) -> Result<Option<u64>> {
        let mut observed = None;
        for slot in self.layers.iter().flatten() {
            for array in [&slot.keys, &slot.values] {
                let bytes = u64::try_from(array.item_size()).map_err(|_| {
                    crate::error::Error::Msg("KV element width overflows u64".into())
                })?;
                if bytes == 0 {
                    return Err(crate::error::Error::Msg(
                        "KV element width must be positive".into(),
                    ));
                }
                match observed {
                    Some(expected) if expected != bytes => {
                        return Err(crate::error::Error::Msg(
                            "dense KV cache contains mixed element widths".into(),
                        ));
                    }
                    None => observed = Some(bytes),
                    _ => {}
                }
            }
        }
        Ok(observed)
    }

    /// Count the resident K/V buffers, including the block padding allocated by this cache;
    /// report the live token offset and allocated capacity separately for receipt geometry.
    pub(crate) fn retained_snapshot(&self) -> Result<Option<(u64, u64, u64, u64)>> {
        if self.layers.iter().all(Option::is_none) {
            return Ok(None);
        }
        if self.layers.iter().any(Option::is_none) {
            return Err(crate::error::Error::Msg(
                "dense KV cache has incomplete layer ownership".into(),
            ));
        }
        let mut total_bytes = 0_u64;
        let mut retained_tokens = None;
        let mut allocated_tokens = None;
        for slot in self.layers.iter().flatten() {
            let key_tokens = u64::try_from(slot.offset)
                .map_err(|_| crate::error::Error::Msg("KV sequence length overflows u64".into()))?;
            if retained_tokens.is_some_and(|tokens| tokens != key_tokens) {
                return Err(crate::error::Error::Msg(
                    "dense KV cache has inconsistent retained sequence lengths".into(),
                ));
            }
            retained_tokens = Some(key_tokens);
            let capacity = u64::try_from(slot.capacity()).map_err(|_| {
                crate::error::Error::Msg("KV allocated capacity overflows u64".into())
            })?;
            if allocated_tokens.is_some_and(|tokens| tokens != capacity) {
                return Err(crate::error::Error::Msg(
                    "dense KV cache has inconsistent allocated capacities".into(),
                ));
            }
            allocated_tokens = Some(capacity);
            let pair_bytes = array_bytes(&slot.keys)?
                .checked_add(array_bytes(&slot.values)?)
                .ok_or_else(|| {
                    crate::error::Error::Msg("retained KV pair bytes overflow u64".into())
                })?;
            total_bytes = total_bytes.checked_add(pair_bytes).ok_or_else(|| {
                crate::error::Error::Msg("retained KV cache bytes overflow u64".into())
            })?;
        }
        let element_bytes = self.element_bytes()?.ok_or_else(|| {
            crate::error::Error::Msg("retained KV cache has no scalar width".into())
        })?;
        Ok(Some((
            total_bytes,
            retained_tokens.unwrap_or_default(),
            allocated_tokens.unwrap_or_default(),
            element_bytes,
        )))
    }

    /// Construct a cache pre-populated with per-layer `(keys, values)` — the seam the prefix cache
    /// (story 7168) reuses a shared prefix's KV through. Each entry is `[batch, n_kv_heads, seq,
    /// head_dim]` (keys already-RoPE'd); the cache then reports [`KvCache::offset`] equal to that
    /// seq length, so a decoder prefills only the suffix at that offset and attends over the seeded
    /// keys. Layout/length consistency across layers is the caller's responsibility.
    ///
    /// The seeded tensors are held exactly as given, so each slot starts *full*: `offset ==
    /// capacity`, no padding. That is load-bearing, not incidental — the seeded arrays are the
    /// prefix store's own entries (shared by refcount), and the first `update` must therefore grow
    /// into a fresh buffer rather than `slice_update` into one the store still holds. Growth
    /// happens exactly when `offset + n > capacity`, which a full slot guarantees for any
    /// `n >= 1`.
    pub fn seeded(layers: Vec<(Array, Array)>) -> Self {
        Self {
            layers: layers
                .into_iter()
                .map(|(keys, values)| {
                    let offset = keys.shape()[SEQ_AXIS as usize];
                    let slot = LayerSlot {
                        keys,
                        values,
                        offset,
                        borrowed: true,
                    };
                    debug_assert_eq!(
                        slot.offset,
                        slot.capacity(),
                        "seeded slot must start full so the first update grows instead of \
                         writing into the shared prefix buffer"
                    );
                    Some(slot)
                })
                .collect(),
            block: KV_BLOCK_TOKENS,
            events: Vec::new(),
        }
    }

    /// Hand every layer's live `(keys, values)` to a caller that outlives this cache — the prefix
    /// store taking a finished request's KV — without copying: the result is a `[.., ..offset]` view
    /// of each block buffer, evaluated (so it is a materialized view, not a lazy graph), or `None`
    /// if any layer is still empty.
    ///
    /// The views share the buffers, so the caller must be the last writer: reset (or drop) this
    /// cache next and never update it again, or the next in-place write would copy the block
    /// instead of donating it. Each view pins its block's padding (fewer than one block of
    /// positions: every growth sizes the buffer to the live length plus whole blocks), a small
    /// fraction of a long entry. The alternative, [`export`](Self::export), holds a second full
    /// copy of the KV next to this cache until it is reset — at long context the dominant
    /// transient of a prefix-store insert (sc-20671).
    pub fn share_live(&self) -> Result<Option<Vec<(Array, Array)>>> {
        let Some(layers) = self
            .layers
            .iter()
            .map(|slot| slot.as_ref().map(LayerSlot::live).transpose())
            .collect::<Result<Option<Vec<_>>>>()?
        else {
            return Ok(None);
        };
        mlx_rs::transforms::eval(layers.iter().flat_map(|(k, v)| [k, v]))?;
        Ok(Some(layers))
    }

    /// Snapshot every layer's cached `(keys, values)` over the live positions, or `None` if any
    /// layer is still empty. Each entry is a fresh contiguous copy, not a view: the prefix cache
    /// stores the result for as long as its LRU keeps it, and a view would pin this cache's whole
    /// padded block buffer for that long (and force every later in-place update to copy). A later
    /// shared-prefix request is [`seeded`] from the stored entries.
    ///
    /// [`seeded`]: ContiguousKvCache::seeded
    pub fn export(&self) -> Result<Option<Vec<(Array, Array)>>> {
        self.layers
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map(|s| {
                        let (k, v) = s.live()?;
                        Ok((materialize(&k)?, materialize(&v)?))
                    })
                    .transpose()
            })
            .collect()
    }

    /// Round `n` up to a whole number of blocks.
    fn blocks_for(&self, n: i32) -> i32 {
        (n + self.block - 1) / self.block * self.block
    }

    /// A zero block of `positions` along the sequence axis, matching `like` in every other axis
    /// and in dtype.
    fn zero_block(&self, like: &Array, positions: i32) -> Result<Array> {
        let mut shape = like.shape().to_vec();
        shape[SEQ_AXIS as usize] = positions;
        Ok(zeros_dtype(&shape, like.dtype())?)
    }

    /// Grow `slot` so it can take `n` more positions past `offset`: trim any padding (or a
    /// seeded/rolled-back tail) to the live length and append whole zero blocks. This is the one
    /// place a layer's buffer changes size.
    fn grow(&self, slot: &mut LayerSlot, n: i32) -> Result<()> {
        let extra = self.blocks_for(n);
        let (live_k, live_v) = if slot.offset == slot.capacity() {
            (slot.keys.clone(), slot.values.clone())
        } else {
            slot.live()?
        };
        let zk = self.zero_block(&live_k, extra)?;
        let zv = self.zero_block(&live_v, extra)?;
        slot.keys = concatenate_axis(&[&live_k, &zk], SEQ_AXIS)?;
        slot.values = concatenate_axis(&[&live_v, &zv], SEQ_AXIS)?;
        slot.borrowed = false;
        Ok(())
    }

    /// Dense compatibility route used by the experimental packed-cache planner before mutation.
    pub fn packed_preflight(&self, query_length: usize, mask: bool) -> CacheRoute {
        self.preflight_packed(query_length, mask)
    }
}

impl KvCache for ContiguousKvCache {
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let n = keys.shape()[SEQ_AXIS as usize];
        // A borrowed (seeded) slot's buffers stay owned by the prefix store, so growing out of
        // them retires nothing: only an owned buffer becomes a transient.
        let prior_bytes = self.layers[layer]
            .as_ref()
            .filter(|slot| !slot.borrowed)
            .map(|slot| {
                array_bytes(&slot.keys)?
                    .checked_add(array_bytes(&slot.values)?)
                    .ok_or_else(|| crate::error::Error::Msg("prior KV bytes overflow u64".into()))
            })
            .transpose()?;
        let mut slot = match self.layers[layer].take() {
            Some(slot) => slot,
            None => LayerSlot {
                keys: self.zero_block(keys, self.blocks_for(n))?,
                values: self.zero_block(values, self.blocks_for(n))?,
                offset: 0,
                borrowed: false,
            },
        };
        let grew = slot.offset + n > slot.capacity();
        if grew {
            self.grow(&mut slot, n)?;
        }
        if n > 0 {
            // In-place write: MLX's slice-update donates the buffer when the cache holds the only
            // reference (it does — the views handed out last step are gone by now), so this step
            // allocates nothing new for the cache.
            let end = slot.offset + n;
            slot.keys
                .try_index_mut((.., .., slot.offset..end, ..), keys)?;
            slot.values
                .try_index_mut((.., .., slot.offset..end, ..), values)?;
            slot.offset = end;
            // The write's output is this cache's own buffer (a borrowed one is copied, never
            // written through).
            slot.borrowed = false;
        }
        let retained_bytes = array_bytes(&slot.keys)?
            .checked_add(array_bytes(&slot.values)?)
            .ok_or_else(|| crate::error::Error::Msg("retained KV bytes overflow u64".into()))?;
        let tokens = u64::try_from(slot.offset)
            .map_err(|_| crate::error::Error::Msg("KV sequence length overflows u64".into()))?;
        if grew {
            // The grown buffer is the layer's new persistent cache (the `append` event below and
            // the retained snapshot). What coexists with it transiently is only the retired
            // pre-growth buffer, so that is the transient: recording old + new here would count
            // the new buffer twice against a peak floor of persistent + transient.
            if let Some(prior_bytes) = prior_bytes {
                self.events.push(CacheEvent {
                    layer,
                    operation: "dense_block_growth_retired_buffer",
                    role: "output",
                    lifetime: "transient",
                    bytes: prior_bytes,
                    tokens,
                });
            }
        }
        self.events.push(CacheEvent {
            layer,
            operation: "append",
            role: "cache",
            lifetime: "persistent",
            bytes: retained_bytes,
            tokens,
        });
        let live = slot.live()?;
        self.layers[layer] = Some(slot);
        Ok(live)
    }

    fn offset(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map_or(0, |s| s.offset)
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map_or(0, |s| s.keys.shape()[0])
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        let idx = Array::from_slice(keep, &[keep.len() as i32]);
        for slot in self.layers.iter_mut().flatten() {
            slot.keys = slot.keys.take_axis(&idx, 0)?;
            slot.values = slot.values.take_axis(&idx, 0)?;
            slot.borrowed = false;
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(crate::error::Error::Msg(format!(
                "truncate: negative len {len}"
            )));
        }
        // Rolling back is a bookkeeping change: the positions past `len` stay in the buffer as
        // padding and are overwritten by the next update.
        for slot in self.layers.iter_mut().flatten() {
            slot.offset = slot.offset.min(len);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        let bytes = self
            .layers
            .iter()
            .flatten()
            .try_fold(0_u64, |total, slot| {
                let pair = array_bytes(&slot.keys)?
                    .checked_add(array_bytes(&slot.values)?)
                    .ok_or_else(|| {
                        crate::error::Error::Msg("released KV pair bytes overflow u64".into())
                    })?;
                total.checked_add(pair).ok_or_else(|| {
                    crate::error::Error::Msg("released KV cache bytes overflow u64".into())
                })
            })?;
        if bytes > 0 {
            self.events.push(CacheEvent {
                layer: usize::MAX,
                operation: "cache_release",
                role: "cache",
                lifetime: "released",
                bytes,
                tokens: 0,
            });
        }
        for slot in &mut self.layers {
            *slot = None;
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Test support shared by the KV-cache tests here and the `AttnKv` tests in `models::qwen35`.
#[cfg(test)]
pub(crate) mod testing {
    use mlx_rs::ops::add;
    use mlx_rs::{Array, Device, Dtype};

    /// Pin every op in a test to the CPU stream, restoring the previous default device on drop.
    /// The default device is process-global and every `#[test]` in this crate shares one binary,
    /// so a one-way switch would leak into whichever test runs next. The cache is pure
    /// bookkeeping over a handful of floats and must not depend on (or dispatch to) the GPU.
    pub(crate) struct CpuStream {
        previous: Device,
    }

    impl CpuStream {
        pub(crate) fn enter() -> Self {
            let previous = Device::try_default().expect("a default device");
            Device::set_default(&Device::cpu());
            Self { previous }
        }
    }

    impl Drop for CpuStream {
        fn drop(&mut self) {
            Device::set_default(&self.previous);
        }
    }

    /// Host copy in logical (row-major) order. The cache hands out strided views over its
    /// padded buffers and `as_slice` reads raw memory ignoring strides, so materialize through an
    /// elementwise op first: its output is always a fresh row-contiguous buffer. (A `reshape` /
    /// `flatten` is *not* enough: for the toy `head_dim = 1` shapes these tests use MLX can
    /// express the flattened view with a single stride and returns another strided view.)
    pub(crate) fn host(a: &Array) -> Vec<f32> {
        add(a, Array::from_f32(0.0))
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// One position `[b=1, h=1, s=1, d=2]` carrying `(tag, tag + 0.5)`.
    pub(crate) fn tok(tag: f32) -> Array {
        Array::from_slice(&[tag, tag + 0.5], &[1, 1, 1, 2])
    }

    /// The naive growing-concat reference a block cache must be indistinguishable from.
    pub(crate) struct ConcatReference {
        pub(crate) k: Option<Array>,
        pub(crate) v: Option<Array>,
    }

    impl ConcatReference {
        pub(crate) fn new() -> Self {
            Self { k: None, v: None }
        }

        pub(crate) fn update(&mut self, k: &Array, v: &Array) -> (Array, Array) {
            use mlx_rs::ops::concatenate_axis;
            self.k = Some(match self.k.take() {
                Some(prev) => concatenate_axis(&[&prev, k], super::SEQ_AXIS).unwrap(),
                None => k.clone(),
            });
            self.v = Some(match self.v.take() {
                Some(prev) => concatenate_axis(&[&prev, v], super::SEQ_AXIS).unwrap(),
                None => v.clone(),
            });
            (self.k.clone().unwrap(), self.v.clone().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{host, tok, ConcatReference, CpuStream};
    use super::*;
    use mlx_rs::memory;

    /// `[b, h, s, d]` of sequential f32 values, for shape/equality checks.
    fn arange4(b: i32, h: i32, s: i32, d: i32) -> Array {
        let n = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Array::from_slice(&data, &[b, h, s, d])
    }

    #[test]
    fn first_update_stores_and_returns_input() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::new(2);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.batch_size(), 0);
        assert_eq!(cache.element_bytes().unwrap(), None);

        let k = arange4(1, 2, 3, 4);
        let v = arange4(1, 2, 3, 4);
        let (ka, va) = cache.update(0, &k, &v).unwrap();
        assert_eq!(ka.shape(), &[1, 2, 3, 4]);
        assert_eq!(va.shape(), &[1, 2, 3, 4]);
        assert_eq!(host(&ka), host(&k));
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.element_bytes().unwrap(), Some(4));
        assert!(cache.retained_snapshot().is_err());
        cache.update(1, &k, &v).unwrap();
        // Two physical 256-token K/V blocks, although only three positions are live.
        assert_eq!(
            cache.retained_snapshot().unwrap(),
            Some((32_768, 3, 256, 4))
        );
    }

    #[test]
    fn cache_events_account_for_block_growth_and_release() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::with_block_tokens(1, 4);
        let k0 = arange4(1, 2, 3, 4);
        cache.update(0, &k0, &k0).unwrap();
        let k1 = arange4(1, 2, 1, 4); // one new token
        let (ka, _) = cache.update(0, &k1, &k1).unwrap();
        assert_eq!(ka.shape(), &[1, 2, 4, 4]); // 3 + 1 along seq
        assert_eq!(cache.offset(), 4);
        assert_eq!(cache.retained_snapshot().unwrap(), Some((256, 4, 4, 4)));
        assert!(cache
            .events()
            .iter()
            .all(|event| event.lifetime != "transient"));

        cache.update(0, &k1, &k1).unwrap();
        assert_eq!(cache.retained_snapshot().unwrap(), Some((512, 5, 8, 4)));
        let event = cache
            .events()
            .iter()
            .find(|event| event.operation == "dense_block_growth_retired_buffer")
            .expect("block growth must publish the retired pre-growth buffer");
        assert_eq!(event.role, "output");
        assert_eq!(event.lifetime, "transient");
        // Only the 256-byte retired buffer; the 512-byte grown buffer is the persistent cache.
        assert_eq!(event.bytes, 256);
        assert_eq!(event.tokens, 5);
        cache.reset().unwrap();
        assert_eq!(cache.retained_snapshot().unwrap(), None);
        let release = cache.events().last().unwrap();
        assert_eq!(release.operation, "cache_release");
        assert_eq!(release.lifetime, "released");
        assert_eq!(release.bytes, 512);
    }

    /// A chunk that lands exactly on a block boundary, then growth: persistent + the largest
    /// transient equals the true coexistence peak (every layer's grown buffer plus one retired
    /// buffer), never the double-counted old + new.
    #[test]
    fn block_growth_at_a_boundary_is_counted_once_in_the_peak_floor() {
        let _cpu = CpuStream::enter();
        const LAYERS: usize = 2;
        let mut cache = ContiguousKvCache::with_block_tokens(LAYERS, 4);
        let chunk = arange4(1, 2, 4, 4); // lands exactly on the 4-token block boundary
        let one = arange4(1, 2, 1, 4);
        for layer in 0..LAYERS {
            cache.update(layer, &chunk, &chunk).unwrap();
        }
        let (before, ..) = cache.retained_snapshot().unwrap().unwrap();
        assert!(cache
            .events()
            .iter()
            .all(|event| event.lifetime != "transient"));
        for layer in 0..LAYERS {
            cache.update(layer, &one, &one).unwrap();
        }
        let (after, ..) = cache.retained_snapshot().unwrap().unwrap();
        let per_layer_old = before / LAYERS as u64;
        let per_layer_new = after / LAYERS as u64;
        let transient = cache
            .events()
            .iter()
            .filter(|event| event.lifetime == "transient")
            .map(|event| event.bytes)
            .max()
            .unwrap();
        // True peak: the last layer's old and new buffers coexist beside every grown layer.
        let true_peak = per_layer_new * LAYERS as u64 + per_layer_old;
        assert_eq!(after + transient, true_peak);
        assert!(after + transient < after + per_layer_old + per_layer_new);
    }

    #[test]
    fn supports_batch_greater_than_one() {
        let _cpu = CpuStream::enter();
        // The headline acceptance for story 7155: the cache is batch-capable.
        let mut cache = ContiguousKvCache::new(1);
        let k0 = arange4(4, 8, 5, 16); // batch = 4
        cache.update(0, &k0, &k0).unwrap();
        let k1 = arange4(4, 8, 2, 16);
        let (ka, va) = cache.update(0, &k1, &k1).unwrap();
        assert_eq!(ka.shape(), &[4, 8, 7, 16]);
        assert_eq!(va.shape(), &[4, 8, 7, 16]);
        assert_eq!(cache.batch_size(), 4);
        assert_eq!(cache.offset(), 7);
    }

    #[test]
    fn concatenated_values_are_in_order() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::new(1);
        // [1,1,2,2] = [[0,1],[2,3]]
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
        // [1,1,1,2] = [[10,11]]
        let b = Array::from_slice(&[10.0f32, 11.0], &[1, 1, 1, 2]);
        cache.update(0, &a, &a).unwrap();
        let (ka, _) = cache.update(0, &b, &b).unwrap();
        assert_eq!(host(&ka), vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0]);
    }

    #[test]
    fn block_cache_matches_concat_reference_across_block_boundaries() {
        let _cpu = CpuStream::enter();
        // 300 single-token updates on top of a 3-token prefill with block 256: crosses the
        // 256 -> 512 growth once. Every returned K/V must equal the concat reference's, and the
        // underlying buffer must change size only at the boundary.
        let block = 256;
        let mut cache = ContiguousKvCache::with_block_tokens(1, block);
        let mut reference = ConcatReference::new();

        let prefill_k = Array::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5], &[1, 1, 3, 2]);
        let prefill_v = Array::from_slice(
            &[100.0f32, 100.5, 101.0, 101.5, 102.0, 102.5],
            &[1, 1, 3, 2],
        );
        let (ck, cv) = cache.update(0, &prefill_k, &prefill_v).unwrap();
        let (rk, rv) = reference.update(&prefill_k, &prefill_v);
        assert_eq!(host(&ck), host(&rk));
        assert_eq!(host(&cv), host(&rv));
        assert_eq!(cache.layers[0].as_ref().unwrap().capacity(), block);

        let mut capacity_changes = Vec::new();
        let mut last_capacity = block;
        for i in 0..300 {
            let k = tok(3.0 + i as f32);
            let v = tok(103.0 + i as f32);
            let (ck, cv) = cache.update(0, &k, &v).unwrap();
            let (rk, rv) = reference.update(&k, &v);
            assert_eq!(ck.shape(), rk.shape(), "update {i}: key shape");
            assert_eq!(host(&ck), host(&rk), "update {i}: keys");
            assert_eq!(host(&cv), host(&rv), "update {i}: values");
            assert_eq!(cache.offset(), 4 + i);

            let capacity = cache.layers[0].as_ref().unwrap().capacity();
            if capacity != last_capacity {
                capacity_changes.push((cache.offset(), capacity));
                last_capacity = capacity;
            }
        }
        // The buffer grew exactly once, when position 257 needed a second block.
        assert_eq!(capacity_changes, vec![(257, 2 * block)]);
        assert_eq!(cache.offset(), 303);
    }

    #[test]
    fn multi_token_update_that_overflows_a_block_grows_by_whole_blocks() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::with_block_tokens(1, 4);
        let a = arange4(1, 1, 3, 2); // fills 3 of a 4-block
        cache.update(0, &a, &a).unwrap();
        assert_eq!(cache.layers[0].as_ref().unwrap().capacity(), 4);
        let b = arange4(1, 1, 6, 2); // needs 6 more: two whole blocks appended
        let (k, _) = cache.update(0, &b, &b).unwrap();
        assert_eq!(k.shape(), &[1, 1, 9, 2]);
        assert_eq!(cache.layers[0].as_ref().unwrap().capacity(), 3 + 8);
        assert_eq!(cache.retained_snapshot().unwrap(), Some((176, 9, 11, 4)));
        let expected: Vec<f32> = host(&a).into_iter().chain(host(&b)).collect();
        assert_eq!(host(&k), expected);
    }

    #[test]
    fn returned_view_never_exposes_padding() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::with_block_tokens(2, 8);
        let a = arange4(1, 2, 3, 4);
        for layer in 0..2 {
            let (k, v) = cache.update(layer, &a, &a).unwrap();
            assert_eq!(k.shape(), &[1, 2, 3, 4]);
            assert_eq!(v.shape(), &[1, 2, 3, 4]);
            let (pk, pv) = cache.peek(layer).unwrap().unwrap();
            assert_eq!(pk.shape(), &[1, 2, 3, 4]);
            assert_eq!(host(&pv), host(&a));
        }
        let exported = cache.export().unwrap().unwrap();
        assert_eq!(exported.len(), 2);
        assert!(exported
            .iter()
            .all(|(k, v)| k.shape() == [1, 2, 3, 4] && v.shape() == [1, 2, 3, 4]));
    }

    #[test]
    fn seeded_cache_reports_length_and_grows_on_update() {
        let _cpu = CpuStream::enter();
        // A seeded layer is exact-length (no padding); the first update grows it by a block and
        // the result equals prefix + suffix in order.
        let prefix = arange4(1, 1, 5, 2);
        let mut cache = ContiguousKvCache::seeded(vec![(prefix.clone(), prefix.clone())]);
        assert_eq!(cache.offset(), 5);
        assert_eq!(cache.batch_size(), 1);
        let suffix = tok(99.0);
        let (k, _) = cache.update(0, &suffix, &suffix).unwrap();
        assert_eq!(k.shape(), &[1, 1, 6, 2]);
        let expected: Vec<f32> = host(&prefix).into_iter().chain(host(&suffix)).collect();
        assert_eq!(host(&k), expected);
        assert_eq!(cache.offset(), 6);
        // The seeded buffer is the prefix store's, which keeps it: growing out of it retires
        // nothing (a recorded transient would raise the campaign's phase-local peak floor by a
        // whole prefix KV that is never allocated).
        assert!(!cache
            .events()
            .iter()
            .any(|event| event.operation == "dense_block_growth_retired_buffer"));
        // Once grown, the buffer is the cache's own: the next growth retires it.
        let more = arange4(1, 1, 300, 2);
        cache.update(0, &more, &more).unwrap();
        assert!(cache
            .events()
            .iter()
            .any(|event| event.operation == "dense_block_growth_retired_buffer"));
    }

    #[test]
    fn retain_sequences_compacts_batch_rows() {
        let _cpu = CpuStream::enter();
        // Batch of 3 rows; drop the middle one, keep [0, 2] in order.
        let mut cache = ContiguousKvCache::new(1);
        // Distinct per-row values so we can verify the right rows survive: row r filled with r.
        let row = |r: f32| vec![r; 2]; // [1, hkv=2, s=1, hd=1] flattened (hd=1) => 2 values/row
        let mut data = Vec::new();
        for r in 0..3 {
            data.extend(row(r as f32));
        }
        let k = Array::from_slice(&data, &[3, 2, 1, 1]);
        cache.update(0, &k, &k).unwrap();
        assert_eq!(cache.batch_size(), 3);

        cache.retain_sequences(&[0, 2]).unwrap();
        assert_eq!(cache.batch_size(), 2);
        assert_eq!(cache.offset(), 1);
        let (ka, _) = cache.peek(0).unwrap().unwrap();
        // Kept rows 0 and 2 (each 2 heads * 1 * 1 = 2 values): all 0.0 then all 2.0.
        assert_eq!(host(&ka), vec![0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn retain_sequences_on_empty_cache_is_noop() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::new(2);
        cache.retain_sequences(&[0]).unwrap();
        assert_eq!(cache.batch_size(), 0);
        assert!(cache.peek(0).unwrap().is_none());
    }

    #[test]
    fn truncate_slices_sequence_axis() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::new(1);
        // [1,1,5,1] = values 0..4 along the seq axis.
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        cache.update(0, &a, &a).unwrap();
        assert_eq!(cache.offset(), 5);
        cache.truncate(3).unwrap();
        assert_eq!(cache.offset(), 3);
        let (k, _) = cache.peek(0).unwrap().unwrap();
        assert_eq!(host(&k), vec![0.0, 1.0, 2.0]);
        cache.truncate(10).unwrap(); // no-op past the end
        assert_eq!(cache.offset(), 3);
        assert!(cache.truncate(-1).is_err());
    }

    #[test]
    fn truncate_then_update_overwrites_the_rolled_back_positions() {
        let _cpu = CpuStream::enter();
        // Speculative rollback: after truncating, the next update lands at the truncated offset and
        // the result matches a concat reference that never saw the rejected tail — in place, with
        // no buffer growth, and across a block boundary too.
        let block = 4;
        let mut cache = ContiguousKvCache::with_block_tokens(1, block);
        let mut reference = ConcatReference::new();
        let prompt = arange4(1, 1, 3, 2);
        cache.update(0, &prompt, &prompt).unwrap();
        reference.update(&prompt, &prompt);

        // Verify pass: 3 draft tokens (positions 3..6) — crosses the 4-block boundary.
        let drafts = Array::from_slice(&[7.0f32, 7.5, 8.0, 8.5, 9.0, 9.5], &[1, 1, 3, 2]);
        cache.update(0, &drafts, &drafts).unwrap();
        assert_eq!(cache.offset(), 6);
        let capacity_after_drafts = cache.layers[0].as_ref().unwrap().capacity();
        assert_eq!(capacity_after_drafts, 3 + 4);

        // Reject the last two drafts: keep the prompt + the first draft.
        cache.truncate(4).unwrap();
        assert_eq!(cache.offset(), 4);
        let accepted = drafts.try_index((.., .., 0..1, ..)).unwrap();
        reference.update(&accepted, &accepted);
        let (k, _) = cache.peek(0).unwrap().unwrap();
        assert_eq!(host(&k), host(reference.k.as_ref().unwrap()));

        // Next step overwrites the rejected positions in place.
        let next = tok(42.0);
        let (ck, cv) = cache.update(0, &next, &next).unwrap();
        let (rk, rv) = reference.update(&next, &next);
        assert_eq!(host(&ck), host(&rk));
        assert_eq!(host(&cv), host(&rv));
        assert_eq!(cache.offset(), 5);
        assert_eq!(
            cache.layers[0].as_ref().unwrap().capacity(),
            capacity_after_drafts,
            "rollback + refill must not reallocate"
        );
    }

    #[test]
    fn truncate_below_a_block_then_grow_keeps_only_live_positions() {
        let _cpu = CpuStream::enter();
        // Growth after a rollback trims the padding first, so the new buffer is live + one block.
        let block = 4;
        let mut cache = ContiguousKvCache::with_block_tokens(1, block);
        let a = arange4(1, 1, 7, 2); // capacity 8
        cache.update(0, &a, &a).unwrap();
        cache.truncate(2).unwrap();
        let big = arange4(1, 1, 7, 2); // 2 + 7 > 8: grow
        let (k, _) = cache.update(0, &big, &big).unwrap();
        assert_eq!(k.shape(), &[1, 1, 9, 2]);
        assert_eq!(cache.layers[0].as_ref().unwrap().capacity(), 2 + 8);
        let a_head = a.try_index((.., .., 0..2, ..)).unwrap();
        let expected: Vec<f32> = host(&a_head).into_iter().chain(host(&big)).collect();
        assert_eq!(host(&k), expected);
    }

    #[test]
    fn reset_clears_state() {
        let _cpu = CpuStream::enter();
        let mut cache = ContiguousKvCache::new(2);
        let k = arange4(1, 2, 3, 4);
        cache.update(0, &k, &k).unwrap();
        cache.reset().unwrap();
        assert_eq!(cache.offset(), 0);
        assert!(cache.peek(0).unwrap().is_none());
    }

    /// Shapes for the memory tests: big enough that one block buffer dwarfs the few-KB noise of
    /// the test harness, small enough to stay a CPU-only unit test.
    const MEM_HEADS: i32 = 8;
    const MEM_HEAD_DIM: i32 = 64;
    const MEM_BLOCK: i32 = 16;
    /// Bytes per cached position in ONE of K or V (f32).
    const MEM_BYTES_PER_POS: usize = (MEM_HEADS * MEM_HEAD_DIM) as usize * 4;

    fn mem_tok(tag: f32) -> Array {
        let n = (MEM_HEADS * MEM_HEAD_DIM) as usize;
        let data: Vec<f32> = (0..n).map(|i| tag + i as f32 * 1e-3).collect();
        Array::from_slice(&data, &[1, MEM_HEADS, 1, MEM_HEAD_DIM])
    }

    #[test]
    fn single_token_updates_keep_memory_bounded_to_the_live_block_buffers() {
        let _cpu = CpuStream::enter();
        // The memory claim behind this cache: 300 single-token updates allocate nothing beyond the
        // live K/V buffers (plus the old + new pair while a block is grown). If the in-place write
        // ever stopped donating — a retained view, an extra clone held across the update — every
        // step would leave a whole block buffer behind and this grows by hundreds of buffers.
        let steps = 300;
        let mut cache = ContiguousKvCache::with_block_tokens(1, MEM_BLOCK);

        // Baseline after the inputs exist, so only the cache's own allocations count. Tests run
        // one at a time (`RUST_TEST_THREADS = 1` is forced by `.cargo/config.toml`), so the
        // process-global counters are ours for the duration.
        let inputs: Vec<(Array, Array)> = (0..steps)
            .map(|i| (mem_tok(i as f32), mem_tok(1000.0 + i as f32)))
            .collect();
        for (k, v) in &inputs {
            mlx_rs::transforms::eval([k, v]).unwrap();
        }
        memory::reset_peak_memory();
        let active_base = memory::get_active_memory();
        let cache_base = memory::get_cache_memory();

        for (k, v) in &inputs {
            let (ck, cv) = cache.update(0, k, v).unwrap();
            // Force the step's graph so the accounting reflects real buffers, then drop the
            // views before the next update exactly as the attention does.
            mlx_rs::transforms::eval([&ck, &cv]).unwrap();
        }

        let capacity = cache.layers[0].as_ref().unwrap().capacity() as usize;
        assert_eq!(capacity, cache.blocks_for(steps) as usize);
        // K + V at the final capacity; growth briefly holds the previous (smaller) pair alongside
        // the new one and the zero block being appended, so "two buffers' worth" is the ceiling.
        let live_pair = 2 * capacity * MEM_BYTES_PER_POS;
        let block_pair = 2 * MEM_BLOCK as usize * MEM_BYTES_PER_POS;
        let slack = 256 * 1024; // page rounding + per-step 1-position inputs
        let bound = 2 * live_pair + block_pair + slack;

        let active_growth = memory::get_active_memory().saturating_sub(active_base);
        let peak_growth = memory::get_peak_memory().saturating_sub(active_base);
        let cache_growth = memory::get_cache_memory().saturating_sub(cache_base);
        assert!(
            active_growth <= live_pair + slack,
            "active grew {active_growth} B; live K/V is {live_pair} B"
        );
        assert!(
            peak_growth <= bound,
            "peak grew {peak_growth} B over {steps} updates; bound {bound} B (a lost donation \
             copies a block buffer per step)"
        );
        // The freed-buffer set is what blew up in production: every concat'd buffer was a new
        // size, so none was ever reused and one retired per STEP. With blocks one retires per
        // GROWTH (each a different size, so still not reused — that residue is what
        // `decode::BufferRelease` clears): the sum of the pre-growth K/V pairs, versus ~300
        // retired pairs (~180 MB here) if the cache silently went back to a per-step reallocation.
        let growths = capacity / MEM_BLOCK as usize;
        let retired: usize = (1..growths)
            .map(|g| 2 * g * MEM_BLOCK as usize * MEM_BYTES_PER_POS)
            .sum();
        assert!(
            cache_growth <= retired + block_pair + slack,
            "freed-buffer cache grew {cache_growth} B; retired pre-growth buffers total {retired} B"
        );
    }

    #[test]
    fn share_live_hands_over_the_buffers_without_a_copy() {
        let _cpu = CpuStream::enter();
        // The prefix store takes a finished request's KV through `share_live`: the entry must be
        // the cache's own buffers (a copy is a second full KV at the store's high-water) and
        // already materialized, over exactly the live positions.
        let mut cache = ContiguousKvCache::with_block_tokens(1, MEM_BLOCK);
        let inputs: Vec<(Array, Array)> = (0..5)
            .map(|i| (mem_tok(i as f32), mem_tok(50.0 + i as f32)))
            .collect();
        let mut expected = (Vec::new(), Vec::new());
        for (k, v) in &inputs {
            let (ck, cv) = cache.update(0, k, v).unwrap();
            mlx_rs::transforms::eval([&ck, &cv]).unwrap();
            expected = (host(&ck), host(&cv));
        }
        let active_base = memory::get_active_memory();
        let shared = cache.share_live().unwrap().unwrap();
        let (sk, sv) = &shared[0];
        let growth = memory::get_active_memory().saturating_sub(active_base);
        let live_pair = 2 * 5 * MEM_BYTES_PER_POS;
        assert!(
            growth < live_pair / 2,
            "share_live allocated {growth} B (a copy of the live K/V is {live_pair} B)"
        );
        assert_eq!(sk.shape(), &[1, MEM_HEADS, 5, MEM_HEAD_DIM]);
        cache.reset().unwrap();
        assert_eq!((host(sk), host(sv)), expected);
        assert!(ContiguousKvCache::new(1).share_live().unwrap().is_none());
    }

    #[test]
    fn export_copies_do_not_pin_the_block_buffer() {
        let _cpu = CpuStream::enter();
        // A stored prefix must not keep this cache's padded buffer alive: after `export`, the
        // next in-place update still donates (no fresh block-sized allocation) and the exported
        // tensors are unaffected by it.
        let mut cache = ContiguousKvCache::with_block_tokens(1, MEM_BLOCK);
        let inputs: Vec<(Array, Array)> = (0..6)
            .map(|i| (mem_tok(i as f32), mem_tok(50.0 + i as f32)))
            .collect();
        for (k, v) in &inputs[..5] {
            let (ck, cv) = cache.update(0, k, v).unwrap();
            mlx_rs::transforms::eval([&ck, &cv]).unwrap();
        }
        let exported = cache.export().unwrap().unwrap();
        let (ek, ev) = &exported[0];
        mlx_rs::transforms::eval([ek, ev]).unwrap();
        assert_eq!(ek.shape(), &[1, MEM_HEADS, 5, MEM_HEAD_DIM]);
        let before_k = host(ek);
        let before_v = host(ev);

        let active_base = memory::get_active_memory();
        let (k, v) = &inputs[5];
        let (ck, cv) = cache.update(0, k, v).unwrap();
        mlx_rs::transforms::eval([&ck, &cv]).unwrap();
        let growth = memory::get_active_memory().saturating_sub(active_base);
        let block_pair = 2 * MEM_BLOCK as usize * MEM_BYTES_PER_POS;
        assert!(
            growth < block_pair / 2,
            "update after export allocated {growth} B (a pinned buffer costs a whole block pair, \
             {block_pair} B): the export is pinning the buffer"
        );
        assert_eq!(host(ek), before_k);
        assert_eq!(host(ev), before_v);
        assert_eq!(cache.offset(), 6);
    }
}
