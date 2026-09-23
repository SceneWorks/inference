//! Key/value caches.
//!
//! Two implementations live behind the [`KvCache`] trait:
//!
//! * [`ContiguousKvCache`] is the day-one implementation: a per-layer growing concat along the
//!   sequence axis (the Candle port of `mlx-llm`'s `ContiguousKvCache`). It is **batch-capable** —
//!   the batch axis is real, not hardcoded to 1. The dynamic-batch scheduler (story 7255) retires
//!   finished sequences through [`KvCache::retain_sequences`]; the prefix cache (story 7256) seeds a
//!   fresh cache from a shared prefix's stored KV via [`ContiguousKvCache::seeded`] /
//!   [`ContiguousKvCache::export`]. The llama family still runs on it (and on the paged cache);
//!   migrating those users is S10 of epic sc-24128.
//! * [`StaticKvCache`] (epic sc-24128, story sc-24132) is the **preallocated** implementation the
//!   fast-decode path runs on: per-layer K/V buffers allocated **once** for a request's capacity,
//!   written **in place** at the current offset ([`Tensor::slice_set`], a bounded `copy2d`), and
//!   attended over as a length-bounded [`Tensor::narrow`] view — no `Tensor::cat` per step, no
//!   growing reallocation, and the buffers' device addresses never change across steps or a
//!   rollback (the property the CUDA-graph runner, S6, captures against). Rollback is an offset
//!   move; the buffers are untouched. Capacity is fixed at construction and a write past it is the
//!   typed [`Error::KvCapacityExceeded`], raised before any device write.
//!
//! GQA is handled without materializing the expanded heads: the attention primitive
//! [`sdpa_gqa_causal`](crate::primitives::attention::sdpa_gqa_causal) folds the query groups into
//! the query-sequence axis and attends over the un-expanded K/V view directly, so the per-step
//! [`repeat_kv`](crate::primitives::attention::repeat_kv) copy is gone as well.
//!
//! ## KV materialization accounting
//! [`note_kv_materialize`] / [`kv_materialize_count`] count, per thread, every KV copy the cache
//! and attention leaves issue — a growing `cat`, a [`repeat_kv`] expansion — in the same style as
//! the host-sync counter. The fast path's op-counter gate asserts the count does **not** move
//! across decode steps once the cache is warm; the growing path is what proves the counter counts.
//!
//! [`repeat_kv`]: crate::primitives::attention::repeat_kv

use std::cell::Cell;

use candle_core::{CpuStorage, DType, Device, Storage, Tensor};

use crate::error::{Error, Result};
use crate::primitives::decode_cache::tensor_bytes;

/// Layout, per layer, of the cached keys/values: `[batch, n_kv_heads, seq, head_dim]`. Keys are
/// stored already-RoPE'd; values raw. The sequence axis (2) is the one that grows each step.
pub const SEQ_AXIS: usize = 2;

/// Which KV cache implementation a decode ran on — surfaced per request through
/// [`DecodeRecord::kv_cache`](crate::decode::DecodeRecord::kv_cache) so the reference path stays
/// *visible* next to the fast one (E2), not merely present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCacheKind {
    /// A growing cache: [`ContiguousKvCache`], the paged cache, or a model's own growing slot
    /// (`AttnKv` in the Qwen3.5 decoder) — a `cat` per step.
    #[default]
    Growing,
    /// [`StaticKvCache`]: preallocated, written in place, attended over a bounded view.
    Static,
}

impl KvCacheKind {
    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            KvCacheKind::Growing => "growing",
            KvCacheKind::Static => "static",
        }
    }
}

thread_local! {
    static KV_MATERIALIZATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Record one KV materialization (a growing-cache `cat`, a `repeat_kv` expansion) on the current
/// thread. Called by the leaves that copy cached K/V; the static path never does.
#[inline]
pub fn note_kv_materialize() {
    KV_MATERIALIZATIONS.with(|c| c.set(c.get().wrapping_add(1)));
}

/// KV materializations recorded on the current thread since it started (monotone; take deltas).
pub fn kv_materialize_count() -> u64 {
    KV_MATERIALIZATIONS.with(Cell::get)
}

/// The decoder-facing cache contract.
///
/// A decoder, for each layer, hands the cache this step's keys/values and gets back the full
/// keys/values to attend over. Positional offset bookkeeping is the cache's job —
/// [`KvCache::offset`] reports how many positions are already cached (the RoPE offset for the next
/// step).
pub trait KvCache {
    /// Append `keys`/`values` for `layer` (each `[batch, n_kv_heads, step, head_dim]`) and return
    /// the full cached `(keys, values)` to attend over, same layout with the sequence axis grown.
    fn update(&mut self, layer: usize, keys: &Tensor, values: &Tensor) -> Result<(Tensor, Tensor)>;

    /// Number of sequence positions currently cached — i.e. the RoPE offset for the next step.
    /// `0` before the first update. Inferred from layer 0 (all layers advance in lockstep).
    fn offset(&self) -> i32;

    /// Batch size of the cached tensors, or `0` before the first update.
    fn batch_size(&self) -> i32;

    /// Number of decoder layers this cache holds slots for.
    fn num_layers(&self) -> usize;

    /// Compact the batch to keep only the rows in `keep` (indices into the current batch axis), in
    /// the given order — the seam the dynamic-batch scheduler (story 7255) retires a finished
    /// sequence through, so the next step runs a smaller batch. A contiguous cache gathers the kept
    /// rows along the batch axis; a paged cache (P4) would free the dropped sequences' pages. `keep`
    /// must be a subset of `0..batch_size`; an empty cache is a no-op.
    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()>;

    /// Drop cached positions past `len`, keeping positions `0..len` along the sequence axis — the
    /// seam speculative decoding (stories 7259/7260) rolls back rejected draft tokens through. `len`
    /// must be `>= 0` and `<= offset()`; `len == offset()` is a no-op and an empty cache ignores it.
    fn truncate(&mut self, len: i32) -> Result<()>;

    /// Drop all cached state, returning the cache to its freshly-constructed (empty) condition.
    fn reset(&mut self);

    /// Downcast hook for a cache driven **natively** by its model. The Qwen3.6 hybrid cache mixes
    /// recurrent (DeltaNet) and growing-KV layers advanced together, so [`Qwen35Model`] downcasts the
    /// `&mut dyn KvCache` it is handed back to the concrete [`Qwen35Cache`] rather than going through
    /// the softmax-only [`KvCache::update`].
    ///
    /// [`Qwen35Model`]: crate::models::Qwen35Model
    /// [`Qwen35Cache`]: crate::models::Qwen35Cache
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Growing-concat KV cache: one `Option<(K, V)>` slot per layer, concatenated along the sequence
/// axis each step. Correctness-first; a paged cache is the throughput replacement behind the trait.
#[derive(Debug)]
pub struct ContiguousKvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl ContiguousKvCache {
    /// A fresh cache with `num_layers` empty slots.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
        }
    }

    /// Borrow the currently-cached `(keys, values)` for `layer`, if any.
    pub fn peek(&self, layer: usize) -> Option<&(Tensor, Tensor)> {
        self.layers.get(layer).and_then(|s| s.as_ref())
    }

    /// Construct a cache pre-populated with per-layer `(keys, values)` — the seam the prefix cache
    /// (story 7256) reuses a shared prefix's KV through. Each entry is `[batch, n_kv_heads, seq,
    /// head_dim]` (keys already-RoPE'd); the cache then reports [`KvCache::offset`] equal to that
    /// seq length, so a decoder prefills only the suffix at that offset and attends over the seeded
    /// keys. Layout/length consistency across layers is the caller's responsibility.
    pub fn seeded(layers: Vec<(Tensor, Tensor)>) -> Self {
        Self {
            layers: layers.into_iter().map(Some).collect(),
        }
    }

    /// Snapshot every layer's cached `(keys, values)` as clones (Candle tensors are reference-counted,
    /// so this shares buffers rather than copying), or `None` if any layer is still empty. The prefix
    /// cache stores this after a generation so a later shared-prefix request can be [`seeded`] from it.
    ///
    /// [`seeded`]: ContiguousKvCache::seeded
    pub fn export(&self) -> Option<Vec<(Tensor, Tensor)>> {
        self.layers.iter().cloned().collect()
    }
}

impl KvCache for ContiguousKvCache {
    fn update(&mut self, layer: usize, keys: &Tensor, values: &Tensor) -> Result<(Tensor, Tensor)> {
        let merged = match self.layers[layer].take() {
            Some((pk, pv)) => {
                note_kv_materialize();
                (
                    Tensor::cat(&[&pk, keys], SEQ_AXIS)?,
                    Tensor::cat(&[&pv, values], SEQ_AXIS)?,
                )
            }
            None => (keys.clone(), values.clone()),
        };
        self.layers[layer] = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }

    fn offset(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map(|(k, _)| k.dims()[SEQ_AXIS] as i32)
            .unwrap_or(0)
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map(|(k, _)| k.dims()[0] as i32)
            .unwrap_or(0)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        for slot in &mut self.layers {
            if let Some((k, v)) = slot.take() {
                let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
                let idx = Tensor::from_vec(idx, (keep.len(),), k.device())?;
                *slot = Some((k.index_select(&idx, 0)?, v.index_select(&idx, 0)?));
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(Error::Msg(format!("truncate: negative len {len}")));
        }
        let len = len as usize;
        for slot in &mut self.layers {
            if let Some((k, v)) = slot.take() {
                if len == 0 {
                    *slot = None; // drop everything
                } else if k.dims()[SEQ_AXIS] <= len {
                    *slot = Some((k, v)); // already at/under the target length
                } else {
                    *slot = Some((k.narrow(SEQ_AXIS, 0, len)?, v.narrow(SEQ_AXIS, 0, len)?));
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Preallocated, in-place KV cache (epic sc-24128, story sc-24132).
///
/// One `[batch, n_kv_heads, capacity, head_dim]` K and V buffer per layer, allocated **once** by
/// [`StaticKvCache::new`] (zero-filled) and never reallocated. [`KvCache::update`] writes the step's
/// keys/values **in place** at the layer's current offset with [`Tensor::slice_set`] and returns the
/// `narrow(SEQ_AXIS, 0, offset + step)` views to attend over — no `cat`, no copy of the history.
/// [`KvCache::truncate`] / [`DecodeCache::rollback_to`](crate::primitives::DecodeCache::rollback_to)
/// only move the offset; the stale tail past it is overwritten by the next write. Consequently the
/// buffers' storage (and on CUDA their device pointers, see [`storage_address`]) is identical for
/// the cache's whole life — across every step and every rollback.
///
/// Capacity is the request's bound (prompt + budget), never the model's `max_position_embeddings`
/// wholesale (Qwen3.8-27B's 262 144 positions would be 16 GiB of KV); a step that would end past
/// the capacity fails with [`Error::KvCapacityExceeded`] **before** writing anything (E6).
///
/// The batch axis is real, but this cache does not compact it: [`KvCache::retain_sequences`] is
/// [`Error::Unsupported`] (the paged cache serves the continuous-batching path). Keys are stored
/// already-RoPE'd, values raw, like every cache in this module.
#[derive(Debug)]
pub struct StaticKvCache {
    /// Per-layer key buffers `[batch, n_kv_heads, capacity, head_dim]`.
    k: Vec<Tensor>,
    /// Per-layer value buffers, same layout.
    v: Vec<Tensor>,
    /// Positions written so far, per layer (all layers advance in lockstep on the decode path).
    len: Vec<usize>,
    capacity: usize,
}

impl StaticKvCache {
    /// Allocate `num_layers` zero-filled K/V buffers of `[batch, n_kv_heads, capacity, head_dim]`
    /// in `dtype` on `device`. `capacity == 0` or an empty shape is [`Error::Msg`]; an allocation
    /// failure surfaces as the device's own error (the caller admits the request against
    /// [`StaticKvCache::buffer_bytes`] first, so it should not get here without headroom).
    pub fn new(
        num_layers: usize,
        batch: usize,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::with_value_dim(
            num_layers, batch, n_kv_heads, head_dim, head_dim, capacity, dtype, device,
        )
    }

    /// [`StaticKvCache::new`] with a value head width that differs from the key's — DeepSeek-V2's
    /// materialized Multi-head Latent Attention caches `qk_nope + qk_rope`-wide keys beside
    /// `v_head_dim`-wide values (sc-24138). Keys are `[batch, n_kv_heads, capacity, key_dim]`,
    /// values `[batch, n_kv_heads, capacity, value_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_value_dim(
        num_layers: usize,
        batch: usize,
        n_kv_heads: usize,
        key_dim: usize,
        value_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if capacity == 0 || batch == 0 || n_kv_heads == 0 || key_dim == 0 || value_dim == 0 {
            return Err(Error::Msg(format!(
                "StaticKvCache: every dimension must be positive (batch {batch}, kv heads \
                 {n_kv_heads}, key dim {key_dim}, value dim {value_dim}, capacity {capacity})"
            )));
        }
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(Tensor::zeros(
                (batch, n_kv_heads, capacity, key_dim),
                dtype,
                device,
            )?);
            v.push(Tensor::zeros(
                (batch, n_kv_heads, capacity, value_dim),
                dtype,
                device,
            )?);
        }
        Ok(Self {
            k,
            v,
            len: vec![0; num_layers],
            capacity,
        })
    }

    /// Bytes [`StaticKvCache::new`] with these arguments allocates (K and V, every layer) — the
    /// number admission charges for the preallocation. Saturating.
    pub fn buffer_bytes(
        num_layers: usize,
        batch: usize,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
    ) -> usize {
        Self::buffer_bytes_with_value_dim(
            num_layers, batch, n_kv_heads, head_dim, head_dim, capacity, dtype,
        )
    }

    /// Bytes [`StaticKvCache::with_value_dim`] with these arguments allocates. Saturating.
    pub fn buffer_bytes_with_value_dim(
        num_layers: usize,
        batch: usize,
        n_kv_heads: usize,
        key_dim: usize,
        value_dim: usize,
        capacity: usize,
        dtype: DType,
    ) -> usize {
        batch
            .saturating_mul(n_kv_heads)
            .saturating_mul(key_dim.saturating_add(value_dim))
            .saturating_mul(capacity)
            .saturating_mul(dtype.size_in_bytes())
            .saturating_mul(num_layers)
    }

    /// Positions the buffers can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes held by the buffers (the full preallocation, regardless of how much is written).
    pub fn bytes(&self) -> usize {
        self.k
            .iter()
            .chain(self.v.iter())
            .fold(0usize, |acc, t| acc.saturating_add(tensor_bytes(t)))
    }

    /// The written prefix of `layer`'s buffers as `(keys, values)` views `[batch, n_kv_heads,
    /// offset, head_dim]` — a [`Tensor::narrow`], no copy. An unwritten layer yields zero-length
    /// views.
    pub fn views(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        let n = self.len[layer];
        Ok((
            self.k[layer].narrow(SEQ_AXIS, 0, n)?,
            self.v[layer].narrow(SEQ_AXIS, 0, n)?,
        ))
    }

    /// The storage addresses of `layer`'s `(keys, values)` buffers (see [`storage_address`]) —
    /// the identity the pointer-stability gate (AC3) and the CUDA-graph runner (S6) rely on.
    pub fn storage_addresses(&self, layer: usize) -> Result<(usize, usize)> {
        Ok((
            storage_address(&self.k[layer])?,
            storage_address(&self.v[layer])?,
        ))
    }

    /// The position the next write to `layer` lands at.
    pub fn layer_offset(&self, layer: usize) -> usize {
        self.len[layer]
    }
}

impl StaticKvCache {
    /// A **deep** copy: the clone gets its own buffers (a full device copy). Sharing them would let
    /// one cache's in-place writes silently corrupt the other. Deliberately not `Clone`: the copy
    /// can fail (device memory) and must surface that as an error, not a panic — the MTP loop's
    /// "restore a clone" rollback goes through this and propagates the failure.
    pub fn try_clone(&self) -> Result<Self> {
        let copy =
            |ts: &[Tensor]| -> Result<Vec<Tensor>> { ts.iter().map(|t| Ok(t.copy()?)).collect() };
        Ok(Self {
            k: copy(&self.k)?,
            v: copy(&self.v)?,
            len: self.len.clone(),
            capacity: self.capacity,
        })
    }
}

impl KvCache for StaticKvCache {
    /// In-place write at the layer's offset; returns the bounded views. Fails **before** writing
    /// when the step would end past the capacity ([`Error::KvCapacityExceeded`]).
    fn update(&mut self, layer: usize, keys: &Tensor, values: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, h, s, d) = keys.dims4()?;
        let (bb, hb, cap, db) = self.k[layer].dims4()?;
        let vd = self.v[layer].dim(3)?;
        if (b, h, d) != (bb, hb, db) || values.dims() != [b, h, s, vd] {
            return Err(Error::Msg(format!(
                "StaticKvCache: step keys {:?} / values {:?} do not fit buffers \
                 [{bb}, {hb}, {cap}, {db}] / [{bb}, {hb}, {cap}, {vd}]",
                keys.dims(),
                values.dims()
            )));
        }
        let offset = self.len[layer];
        let end = offset + s;
        if end > cap {
            return Err(Error::KvCapacityExceeded {
                requested: end,
                capacity: cap,
            });
        }
        // `slice_set` wants contiguous operands; the decoders hand over contiguous head-major
        // projections, so these are no-op clones on the decode path.
        self.k[layer].slice_set(&keys.contiguous()?, SEQ_AXIS, offset)?;
        self.v[layer].slice_set(&values.contiguous()?, SEQ_AXIS, offset)?;
        self.len[layer] = end;
        self.views(layer)
    }

    fn offset(&self) -> i32 {
        self.len.first().copied().unwrap_or(0) as i32
    }

    fn batch_size(&self) -> i32 {
        if self.offset() == 0 {
            return 0;
        }
        self.k.first().map(|k| k.dims()[0] as i32).unwrap_or(0)
    }

    fn num_layers(&self) -> usize {
        self.k.len()
    }

    fn retain_sequences(&mut self, _keep: &[i32]) -> Result<()> {
        Err(Error::Unsupported(
            "StaticKvCache: retain_sequences (batch compaction) is not supported; the paged cache \
             serves the continuous-batching path"
                .into(),
        ))
    }

    /// Moves the offset only; the buffers are untouched (that is what keeps their addresses stable).
    fn truncate(&mut self, len: i32) -> Result<()> {
        let len = usize::try_from(len)
            .map_err(|_| Error::Msg(format!("StaticKvCache: negative truncate length {len}")))?;
        for l in &mut self.len {
            if len < *l {
                *l = len;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        for l in &mut self.len {
            *l = 0;
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// The address of a tensor's first element in its backing storage: the host pointer for CPU
/// storage, the CUDA device pointer for CUDA storage (`cfg(feature = "cuda")`). Two tensors with
/// equal addresses (and dtype) alias the same bytes; a buffer whose address is unchanged across
/// operations was written in place. Storage kinds this crate does not build for are
/// [`Error::Unsupported`].
pub fn storage_address(t: &Tensor) -> Result<usize> {
    let (storage, layout) = t.storage_and_layout();
    let base = match &*storage {
        Storage::Cpu(cpu) => cpu_base_address(cpu)?,
        #[cfg(feature = "cuda")]
        Storage::Cuda(cuda) => cuda_base_address(cuda)?,
        #[allow(unreachable_patterns)]
        _ => {
            return Err(Error::Unsupported(
                "storage_address: only CPU and CUDA storage are addressable here".into(),
            ))
        }
    };
    Ok(base + layout.start_offset() * t.dtype().size_in_bytes())
}

fn cpu_base_address(cpu: &CpuStorage) -> Result<usize> {
    Ok(match cpu {
        CpuStorage::U8(v) => v.as_ptr() as usize,
        CpuStorage::U32(v) => v.as_ptr() as usize,
        CpuStorage::I64(v) => v.as_ptr() as usize,
        CpuStorage::BF16(v) => v.as_ptr() as usize,
        CpuStorage::F16(v) => v.as_ptr() as usize,
        CpuStorage::F32(v) => v.as_ptr() as usize,
        CpuStorage::F64(v) => v.as_ptr() as usize,
        _ => {
            return Err(Error::Unsupported(
                "storage_address: CPU dtype is not addressable here".into(),
            ))
        }
    })
}

#[cfg(feature = "cuda")]
fn cuda_base_address(cuda: &candle_core::CudaStorage) -> Result<usize> {
    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
    use candle_core::cuda_backend::CudaStorageSlice as S;
    fn ptr<T>(s: &CudaSlice<T>) -> usize {
        let (p, _guard) = s.device_ptr(s.stream());
        p as usize
    }
    Ok(match &cuda.slice {
        S::U8(s) => ptr(s),
        S::U32(s) => ptr(s),
        S::I64(s) => ptr(s),
        S::BF16(s) => ptr(s),
        S::F16(s) => ptr(s),
        S::F32(s) => ptr(s),
        S::F64(s) => ptr(s),
        _ => {
            return Err(Error::Unsupported(
                "storage_address: CUDA dtype is not addressable here".into(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `[b, h, s, d]` of sequential f32 values, for shape/equality checks.
    fn arange4(b: usize, h: usize, s: usize, d: usize) -> Tensor {
        let n = (b * h * s * d) as f32;
        Tensor::arange(0f32, n, &Device::Cpu)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
    }

    #[test]
    fn first_update_stores_and_returns_input() {
        let mut cache = ContiguousKvCache::new(2);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.batch_size(), 0);

        let k = arange4(1, 2, 3, 4);
        let (ka, va) = cache.update(0, &k, &k).unwrap();
        assert_eq!(ka.dims(), &[1, 2, 3, 4]);
        assert_eq!(va.dims(), &[1, 2, 3, 4]);
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.num_layers(), 2);
    }

    #[test]
    fn second_update_concatenates_on_seq_axis() {
        let mut cache = ContiguousKvCache::new(1);
        let k0 = arange4(1, 2, 3, 4);
        cache.update(0, &k0, &k0).unwrap();
        let k1 = arange4(1, 2, 1, 4); // one new token
        let (ka, _) = cache.update(0, &k1, &k1).unwrap();
        assert_eq!(ka.dims(), &[1, 2, 4, 4]); // 3 + 1 along seq
        assert_eq!(cache.offset(), 4);
    }

    #[test]
    fn supports_batch_greater_than_one() {
        let mut cache = ContiguousKvCache::new(1);
        let k0 = arange4(4, 8, 5, 16); // batch = 4
        cache.update(0, &k0, &k0).unwrap();
        let k1 = arange4(4, 8, 2, 16);
        let (ka, va) = cache.update(0, &k1, &k1).unwrap();
        assert_eq!(ka.dims(), &[4, 8, 7, 16]);
        assert_eq!(va.dims(), &[4, 8, 7, 16]);
        assert_eq!(cache.batch_size(), 4);
        assert_eq!(cache.offset(), 7);
    }

    #[test]
    fn retain_sequences_compacts_batch_rows() {
        // Batch of 3 rows; drop the middle one, keep [0, 2] in order. Each row filled with its index
        // (hkv=2, s=1, hd=1 ⇒ 2 values/row) so we can verify the right rows survive.
        let mut cache = ContiguousKvCache::new(1);
        let data: Vec<f32> = (0..3).flat_map(|r| vec![r as f32; 2]).collect();
        let k = Tensor::from_vec(data, (3, 2, 1, 1), &Device::Cpu).unwrap();
        cache.update(0, &k, &k).unwrap();
        assert_eq!(cache.batch_size(), 3);

        cache.retain_sequences(&[0, 2]).unwrap();
        assert_eq!(cache.batch_size(), 2);
        assert_eq!(cache.offset(), 1);
        let (ka, _) = cache.peek(0).unwrap();
        let host = ka.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Kept rows 0 and 2 (each 2 heads * 1 * 1 = 2 values): all 0.0 then all 2.0.
        assert_eq!(host, vec![0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn retain_sequences_on_empty_cache_is_noop() {
        let mut cache = ContiguousKvCache::new(2);
        cache.retain_sequences(&[0]).unwrap();
        assert_eq!(cache.batch_size(), 0);
        assert!(cache.peek(0).is_none());
    }

    #[test]
    fn reset_clears_state() {
        let mut cache = ContiguousKvCache::new(2);
        let k = arange4(1, 2, 3, 4);
        cache.update(0, &k, &k).unwrap();
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(cache.peek(0).is_none());
    }

    #[test]
    fn export_then_seeded_round_trips() {
        // Fill a 2-layer cache, snapshot it, and rebuild from the snapshot: offset/batch and the
        // actual values survive (the prefix cache's store → seed path).
        let mut cache = ContiguousKvCache::new(2);
        let k0 = arange4(1, 2, 3, 4);
        let k1 = arange4(1, 2, 3, 4).affine(1.0, 100.0).unwrap();
        cache.update(0, &k0, &k0).unwrap();
        cache.update(1, &k1, &k1).unwrap();

        let snapshot = cache.export().expect("all layers populated");
        assert_eq!(snapshot.len(), 2);

        let seeded = ContiguousKvCache::seeded(snapshot);
        assert_eq!(seeded.offset(), 3);
        assert_eq!(seeded.batch_size(), 1);
        assert_eq!(seeded.num_layers(), 2);
        let (sk1, _) = seeded.peek(1).unwrap();
        assert_eq!(
            sk1.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            k1.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn export_is_none_when_any_layer_empty() {
        let mut cache = ContiguousKvCache::new(2);
        let k = arange4(1, 2, 3, 4);
        cache.update(0, &k, &k).unwrap(); // layer 1 left empty
        assert!(cache.export().is_none());
    }

    #[test]
    fn truncate_slices_sequence_axis() {
        let mut cache = ContiguousKvCache::new(1);
        // [1,1,5,1] = values 0..4 along the seq axis.
        let a =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], (1, 1, 5, 1), &Device::Cpu).unwrap();
        cache.update(0, &a, &a).unwrap();
        assert_eq!(cache.offset(), 5);
        cache.truncate(3).unwrap();
        assert_eq!(cache.offset(), 3);
        let (k, _) = cache.peek(0).unwrap();
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 1.0, 2.0]
        );
        cache.truncate(10).unwrap(); // no-op past the end
        assert_eq!(cache.offset(), 3);
        cache.truncate(0).unwrap(); // drop everything
        assert_eq!(cache.offset(), 0);
        assert!(cache.peek(0).is_none());
    }

    #[test]
    fn seeded_offset_reflects_seq_length() {
        // A cache seeded to N positions prefills the suffix at offset N.
        let k = arange4(1, 2, 5, 4);
        let seeded = ContiguousKvCache::seeded(vec![(k.clone(), k)]);
        assert_eq!(seeded.offset(), 5);
        assert_eq!(seeded.num_layers(), 1);
    }
}

#[cfg(test)]
mod static_tests {
    use super::*;
    use candle_core::Device;

    fn step(b: usize, h: usize, s: usize, d: usize, phase: f32) -> Tensor {
        let n = (b * h * s * d) as f32;
        Tensor::arange(0f32, n, &Device::Cpu)
            .unwrap()
            .affine(0.5, phase as f64)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The static cache's bounded views hold exactly what the growing cache's concatenation holds,
    /// step after step, for a batch > 1 and several layers — in place, no `cat`.
    #[test]
    fn static_views_match_growing_cache_and_never_materialize() {
        let (b, h, d, cap) = (2usize, 3usize, 4usize, 12usize);
        let mut fixed = StaticKvCache::new(2, b, h, d, cap, DType::F32, &Device::Cpu).unwrap();
        let mut growing = ContiguousKvCache::new(2);
        assert_eq!(fixed.offset(), 0);
        assert_eq!(fixed.batch_size(), 0);
        assert_eq!(fixed.num_layers(), 2);
        assert_eq!(fixed.capacity(), cap);
        let before = kv_materialize_count();
        let mut phase = 0.0;
        for s in [5usize, 1, 1, 3, 1, 1] {
            for layer in 0..2 {
                let k = step(b, h, s, d, phase);
                let v = step(b, h, s, d, phase + 100.0);
                phase += 1.0;
                let (fk, fv) = fixed.update(layer, &k, &v).unwrap();
                let (gk, gv) = growing.update(layer, &k, &v).unwrap();
                assert_eq!(fk.dims(), gk.dims());
                assert_eq!(host(&fk), host(&gk));
                assert_eq!(host(&fv), host(&gv));
            }
        }
        assert_eq!(fixed.offset(), 12);
        assert_eq!(fixed.batch_size(), b as i32);
        assert_eq!(growing.offset(), 12);
        assert_eq!(
            kv_materialize_count() - before,
            2 * 5,
            "the growing cache materialized once per appended step per layer; the static cache never"
        );
    }

    #[test]
    fn write_past_capacity_is_typed_and_leaves_the_cache_untouched() {
        let mut cache = StaticKvCache::new(1, 1, 2, 4, 6, DType::F32, &Device::Cpu).unwrap();
        cache
            .update(0, &step(1, 2, 4, 4, 0.0), &step(1, 2, 4, 4, 0.0))
            .unwrap();
        let (k_before, _) = cache.views(0).unwrap();
        let snapshot = host(&k_before);
        let err = cache
            .update(0, &step(1, 2, 3, 4, 7.0), &step(1, 2, 3, 4, 7.0))
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::KvCapacityExceeded {
                    requested: 7,
                    capacity: 6
                }
            ),
            "{err}"
        );
        assert_eq!(cache.offset(), 4, "a refused write does not advance");
        assert_eq!(
            host(&cache.views(0).unwrap().0),
            snapshot,
            "a refused write does not touch the buffer"
        );
        // A step that exactly fills the capacity is fine.
        cache
            .update(0, &step(1, 2, 2, 4, 9.0), &step(1, 2, 2, 4, 9.0))
            .unwrap();
        assert_eq!(cache.offset(), 6);
        // Shape mismatches are rejected before any write.
        assert!(cache
            .update(0, &step(1, 3, 1, 4, 0.0), &step(1, 3, 1, 4, 0.0))
            .is_err());
        assert!(StaticKvCache::new(1, 1, 2, 4, 0, DType::F32, &Device::Cpu).is_err());
    }

    /// **AC3 (CPU).** The buffers' storage addresses are unchanged across 100 in-place steps and a
    /// rollback; only the offset moves, and the rolled-back positions are overwritten in place.
    #[test]
    fn buffer_addresses_are_stable_across_100_steps_and_rollback() {
        let mut cache = StaticKvCache::new(2, 1, 2, 4, 128, DType::F32, &Device::Cpu).unwrap();
        let addresses: Vec<(usize, usize)> = (0..2)
            .map(|l| cache.storage_addresses(l).unwrap())
            .collect();
        assert_ne!(addresses[0], addresses[1]);
        assert_ne!(addresses[0].0, addresses[0].1);
        for i in 0..100 {
            for (layer, expected) in addresses.iter().enumerate() {
                let (k, _) = cache
                    .update(
                        layer,
                        &step(1, 2, 1, 4, i as f32),
                        &step(1, 2, 1, 4, -(i as f32)),
                    )
                    .unwrap();
                // The view aliases the buffer (same base address, no copy).
                assert_eq!(storage_address(&k).unwrap(), expected.0);
            }
            for (layer, expected) in addresses.iter().enumerate() {
                assert_eq!(cache.storage_addresses(layer).unwrap(), *expected);
            }
        }
        assert_eq!(cache.offset(), 100);
        cache.truncate(40).unwrap();
        assert_eq!(cache.offset(), 40);
        for (layer, expected) in addresses.iter().enumerate() {
            assert_eq!(cache.storage_addresses(layer).unwrap(), *expected);
            assert_eq!(cache.views(layer).unwrap().0.dims(), &[1, 2, 40, 4]);
        }
        // Re-decoding after the rollback overwrites position 40 in place.
        let (k, _) = cache
            .update(0, &step(1, 2, 1, 4, 555.0), &step(1, 2, 1, 4, 0.0))
            .unwrap();
        assert_eq!(k.dims(), &[1, 2, 41, 4]);
        let row = k.narrow(SEQ_AXIS, 40, 1).unwrap();
        assert_eq!(host(&row), host(&step(1, 2, 1, 4, 555.0)));
        assert_eq!(cache.storage_addresses(0).unwrap(), addresses[0]);
        cache.truncate(1_000).unwrap(); // past the end: no-op
        assert_eq!(cache.offset(), 41);
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.batch_size(), 0);
        assert_eq!(cache.storage_addresses(0).unwrap(), addresses[0]);
    }

    /// **AC3 (CUDA).** The same gate reading the CUDA device pointers of the buffers.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_device_pointers_are_stable_across_100_steps_and_rollback() {
        let device = Device::new_cuda(0).expect("cuda device");
        let mk = |phase: f32| step(1, 2, 1, 8, phase).to_device(&device).unwrap();
        let mut cache = StaticKvCache::new(2, 1, 2, 8, 128, DType::F32, &device).unwrap();
        let addresses: Vec<(usize, usize)> = (0..2)
            .map(|l| cache.storage_addresses(l).unwrap())
            .collect();
        assert_ne!(addresses[0], addresses[1]);
        for i in 0..100 {
            for (layer, expected) in addresses.iter().enumerate() {
                let (k, _) = cache
                    .update(layer, &mk(i as f32), &mk(-(i as f32)))
                    .unwrap();
                assert_eq!(storage_address(&k).unwrap(), expected.0);
            }
        }
        device.synchronize().unwrap();
        cache.truncate(37).unwrap();
        for (layer, expected) in addresses.iter().enumerate() {
            assert_eq!(cache.storage_addresses(layer).unwrap(), *expected);
        }
        let (k, _) = cache.update(0, &mk(777.0), &mk(0.0)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 38, 8]);
        let row = k
            .narrow(SEQ_AXIS, 37, 1)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        assert_eq!(host(&row), host(&step(1, 2, 1, 8, 777.0)));
        assert_eq!(cache.storage_addresses(0).unwrap(), addresses[0]);
    }

    /// `try_clone` is a deep copy: the clone owns its buffers, and writing into it leaves the
    /// original's offset and bytes alone.
    #[test]
    fn bytes_are_the_full_preallocation_and_try_clone_is_deep() {
        let cache = StaticKvCache::new(3, 2, 4, 8, 16, DType::F32, &Device::Cpu).unwrap();
        let expected = StaticKvCache::buffer_bytes(3, 2, 4, 8, 16, DType::F32);
        assert_eq!(expected, 3 * 2 * 2 * 4 * 8 * 16 * 4);
        assert_eq!(cache.bytes(), expected);
        let mut original = cache;
        original
            .update(0, &step(2, 4, 1, 8, 1.0), &step(2, 4, 1, 8, 1.0))
            .unwrap();
        let mut copy = original.try_clone().unwrap();
        assert_eq!(copy.offset(), 1);
        assert_ne!(
            copy.storage_addresses(0).unwrap(),
            original.storage_addresses(0).unwrap(),
            "a clone owns its own buffers"
        );
        copy.update(0, &step(2, 4, 1, 8, 9.0), &step(2, 4, 1, 8, 9.0))
            .unwrap();
        assert_eq!(original.offset(), 1);
        assert_eq!(host(&original.views(0).unwrap().0).len(), 2 * 4 * 8);
        assert_eq!(
            host(&original.views(0).unwrap().0),
            host(&step(2, 4, 1, 8, 1.0)),
            "the clone's write must not reach the original's buffer"
        );
        assert!(matches!(
            original.retain_sequences(&[0]),
            Err(Error::Unsupported(_))
        ));
        assert!(original.truncate(-1).is_err());
    }

    /// sc-24138: MLA caches keys and values of different widths; the static cache holds both, in
    /// place, and still refuses a value that fits neither buffer.
    #[test]
    fn value_dim_may_differ_from_key_dim() {
        let mut cache =
            StaticKvCache::with_value_dim(1, 1, 2, 6, 4, 8, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            cache.bytes(),
            StaticKvCache::buffer_bytes_with_value_dim(1, 1, 2, 6, 4, 8, DType::F32)
        );
        assert_eq!(cache.bytes(), 2 * (6 + 4) * 8 * 4);
        let (k, v) = cache
            .update(0, &step(1, 2, 3, 6, 0.0), &step(1, 2, 3, 4, 1.0))
            .unwrap();
        assert_eq!(k.dims(), &[1, 2, 3, 6]);
        assert_eq!(v.dims(), &[1, 2, 3, 4]);
        assert_eq!(host(&v), host(&step(1, 2, 3, 4, 1.0)));
        assert!(cache
            .update(0, &step(1, 2, 1, 6, 0.0), &step(1, 2, 1, 6, 0.0))
            .is_err());
        assert_eq!(cache.offset(), 3);
        assert_eq!(
            StaticKvCache::buffer_bytes(2, 1, 2, 4, 8, DType::F32),
            StaticKvCache::buffer_bytes_with_value_dim(2, 1, 2, 4, 4, 8, DType::F32)
        );
    }

    #[test]
    fn kind_labels_are_stable() {
        assert_eq!(KvCacheKind::Growing.label(), "growing");
        assert_eq!(KvCacheKind::Static.label(), "static");
        assert_eq!(KvCacheKind::default(), KvCacheKind::Growing);
    }
}
