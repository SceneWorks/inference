//! The step-seam cache every softmax-attention decoder shares (epic sc-24128, story sc-24138).
//!
//! [`StepKvCache`] is the [`DecodeCache`] the llama family ([`CausalLm`]), StarCoder2 and the
//! StarVector-1B GPTBigCode decoder hand the step seam. It is one type with three backings, so no
//! decoder keeps a private cache of its own (E0):
//!
//! * **static** — one preallocated [`StaticKvCache`] per layer, sized once for the request's bound
//!   ([`StepKvCache::preallocated`]): written in place, rolled back by moving the offset, buffer
//!   addresses stable for the cache's life. The fast path's default (story sc-24132's template,
//!   generalized to per-layer shapes: Gemma 4's layer types disagree on head count and width, and
//!   DeepSeek-V2's materialized MLA caches keys and values of different widths).
//! * **growing** — the [`ContiguousKvCache`] concat, the reference oracle's arithmetic kept
//!   selectable through the seam (E2) and the backing the multimodal providers decode on.
//! * **paged** — a [`PagedKvCache`] over a shared block pool. Kept behind the seam rather than
//!   replaced: it is what the continuous-batching / prefix-sharing paths are built on (ragged
//!   batches over one pool, copy-on-write prefix blocks), which a per-request preallocation cannot
//!   express.
//!
//! The cache also carries the RoPE **position delta** (M-RoPE's `mrope_delta`): a decoder adds it
//! to [`DecodeCache::len`] to position the tokens a step feeds after a multimodal prefill.
//!
//! [`CausalLm`]: crate::models::CausalLm

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::primitives::decode_cache::{tensor_bytes, CacheMemory, DecodeCache};
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache, KvCacheKind, StaticKvCache};
use crate::primitives::paged_kv_cache::PagedKvCache;

/// One layer's cached key/value geometry.
#[derive(Clone, Debug)]
pub struct LayerKvShape {
    /// Cached K/V heads (the un-expanded GQA count; the full head count for MLA).
    pub kv_heads: usize,
    /// Key head width.
    pub key_dim: usize,
    /// Value head width (differs from `key_dim` only for MLA).
    pub value_dim: usize,
    /// The device the layer (and so its cache) lives on — per layer for a pipeline-sharded model.
    pub device: Device,
}

/// A decoder's per-layer KV geometry: what a preallocation allocates and what admission prices.
/// `None` marks a layer that caches nothing (a Gemma 4 `num_kv_shared_layers` tail layer, which
/// attends its donor's keys).
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// One entry per decoder layer.
    pub layers: Vec<Option<LayerKvShape>>,
    /// The cached tensors' dtype (the decoder's compute dtype).
    pub dtype: DType,
}

impl KvLayout {
    /// K + V bytes one sequence position costs across every caching layer.
    pub fn bytes_per_position(&self) -> usize {
        self.layers.iter().flatten().fold(0usize, |acc, l| {
            acc.saturating_add(
                l.kv_heads
                    .saturating_mul(l.key_dim.saturating_add(l.value_dim))
                    .saturating_mul(self.dtype.size_in_bytes()),
            )
        })
    }

    /// Bytes [`StepKvCache::preallocated`] allocates for `capacity` positions (batch 1). Saturating.
    pub fn static_bytes(&self, capacity: usize) -> usize {
        self.layers.iter().flatten().fold(0usize, |acc, l| {
            acc.saturating_add(StaticKvCache::buffer_bytes_with_value_dim(
                1,
                1,
                l.kv_heads,
                l.key_dim,
                l.value_dim,
                capacity,
                self.dtype,
            ))
        })
    }

    /// The largest per-layer `(kv_heads, max(key_dim, value_dim))` — the scalar pair an admission
    /// geometry must carry so `layers × kv_heads × head_dim × 2` covers every layer, whatever its
    /// type.
    pub fn widest_layer(&self) -> (usize, usize) {
        self.layers.iter().flatten().fold((0, 0), |(h, d), l| {
            (h.max(l.kv_heads), d.max(l.key_dim.max(l.value_dim)))
        })
    }
}

enum Backing {
    Growing(ContiguousKvCache),
    Static(Vec<Option<StaticKvCache>>),
    Paged(PagedKvCache),
}

/// The shared step-seam cache (see the module docs).
pub struct StepKvCache {
    backing: Backing,
    rope_delta: i32,
    bytes_per_position: usize,
}

impl std::fmt::Debug for StepKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backing = match &self.backing {
            Backing::Growing(_) => "growing",
            Backing::Static(_) => "static",
            Backing::Paged(_) => "paged",
        };
        f.debug_struct("StepKvCache")
            .field("backing", &backing)
            .field("len", &KvCache::offset(self))
            .field("rope_delta", &self.rope_delta)
            .finish()
    }
}

impl StepKvCache {
    /// An empty cache on the growing ([`ContiguousKvCache`]) backing.
    pub fn growing(layout: &KvLayout) -> Self {
        Self {
            backing: Backing::Growing(ContiguousKvCache::new(layout.layers.len())),
            rope_delta: 0,
            bytes_per_position: layout.bytes_per_position(),
        }
    }

    /// An empty cache whose caching layers are each one preallocated [`StaticKvCache`] of
    /// `capacity` positions on the layer's own device — [`KvLayout::static_bytes`] of memory,
    /// allocated here, once. `capacity == 0` is [`Error::Msg`].
    pub fn preallocated(layout: &KvLayout, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Msg(
                "a static KV cache needs a capacity of at least one position".into(),
            ));
        }
        let layers = layout
            .layers
            .iter()
            .map(|l| {
                l.as_ref()
                    .map(|l| {
                        StaticKvCache::with_value_dim(
                            1,
                            1,
                            l.kv_heads,
                            l.key_dim,
                            l.value_dim,
                            capacity,
                            layout.dtype,
                            &l.device,
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backing: Backing::Static(layers),
            rope_delta: 0,
            bytes_per_position: layout.bytes_per_position(),
        })
    }

    /// An existing paged cache behind the seam (see the module docs for why it is kept).
    pub fn paged(cache: PagedKvCache, layout: &KvLayout) -> Self {
        Self {
            backing: Backing::Paged(cache),
            rope_delta: 0,
            bytes_per_position: layout.bytes_per_position(),
        }
    }

    /// The RoPE position delta the next step's tokens are shifted by (M-RoPE's `mrope_delta`;
    /// `0` for a text prompt).
    pub fn rope_delta(&self) -> i32 {
        self.rope_delta
    }

    /// Set the RoPE position delta (after a multimodal M-RoPE prefill).
    pub fn set_rope_delta(&mut self, delta: i32) {
        self.rope_delta = delta;
    }

    /// Positions a static backing can hold (`None` for the growing and paged backings).
    pub fn kv_capacity(&self) -> Option<usize> {
        match &self.backing {
            Backing::Static(layers) => layers.iter().flatten().map(|l| l.capacity()).next(),
            _ => None,
        }
    }

    /// The storage addresses of `layer`'s static `(keys, values)` buffers — the identity the
    /// pointer-stability gate and the CUDA-graph runner rely on. `None` for a non-static backing or
    /// a layer that caches nothing.
    pub fn storage_addresses(&self, layer: usize) -> Option<Result<(usize, usize)>> {
        match &self.backing {
            Backing::Static(layers) => layers
                .get(layer)
                .and_then(Option::as_ref)
                .map(|l| l.storage_addresses(0)),
            _ => None,
        }
    }

    /// The paged backing, when that is what this cache runs on.
    pub fn as_paged(&self) -> Option<&PagedKvCache> {
        match &self.backing {
            Backing::Paged(p) => Some(p),
            _ => None,
        }
    }

    fn first_static(layers: &[Option<StaticKvCache>]) -> Option<&StaticKvCache> {
        layers.iter().flatten().next()
    }
}

impl KvCache for StepKvCache {
    fn update(&mut self, layer: usize, keys: &Tensor, values: &Tensor) -> Result<(Tensor, Tensor)> {
        match &mut self.backing {
            Backing::Growing(c) => c.update(layer, keys, values),
            Backing::Paged(c) => c.update(layer, keys, values),
            Backing::Static(layers) => layers
                .get_mut(layer)
                .and_then(Option::as_mut)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "StepKvCache: layer {layer} has no static KV slot (it caches nothing)"
                    ))
                })?
                .update(0, keys, values),
        }
    }

    fn offset(&self) -> i32 {
        match &self.backing {
            Backing::Growing(c) => c.offset(),
            Backing::Paged(c) => c.offset(),
            Backing::Static(layers) => Self::first_static(layers).map_or(0, |l| l.offset()),
        }
    }

    fn batch_size(&self) -> i32 {
        match &self.backing {
            Backing::Growing(c) => c.batch_size(),
            Backing::Paged(c) => c.batch_size(),
            Backing::Static(layers) => Self::first_static(layers).map_or(0, |l| l.batch_size()),
        }
    }

    fn num_layers(&self) -> usize {
        match &self.backing {
            Backing::Growing(c) => c.num_layers(),
            Backing::Paged(c) => c.num_layers(),
            Backing::Static(layers) => layers.len(),
        }
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        match &mut self.backing {
            Backing::Growing(c) => c.retain_sequences(keep),
            Backing::Paged(c) => c.retain_sequences(keep),
            Backing::Static(_) => Err(Error::Unsupported(
                "StepKvCache: a static backing does not compact its batch".into(),
            )),
        }
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        match &mut self.backing {
            Backing::Growing(c) => c.truncate(len),
            Backing::Paged(c) => c.truncate(len),
            Backing::Static(layers) => {
                for l in layers.iter_mut().flatten() {
                    l.truncate(len)?;
                }
                Ok(())
            }
        }
    }

    fn reset(&mut self) {
        match &mut self.backing {
            Backing::Growing(c) => c.reset(),
            Backing::Paged(c) => c.reset(),
            Backing::Static(layers) => layers.iter_mut().flatten().for_each(|l| l.reset()),
        }
        self.rope_delta = 0;
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl DecodeCache for StepKvCache {
    fn len(&self) -> i32 {
        KvCache::offset(self)
    }

    /// Exact for every backing: the growing cache narrows, the paged cache releases whole blocks
    /// (copy-on-write for a shared boundary block), the static cache moves its offset.
    fn rollback_to(&mut self, n: i32) -> Result<()> {
        let len = DecodeCache::len(self);
        if n < 0 || n > len {
            return Err(Error::Msg(format!(
                "StepKvCache: cannot roll back to {n} from {len} positions"
            )));
        }
        self.truncate(n)
    }

    fn reset(&mut self) {
        KvCache::reset(self)
    }

    fn memory(&self) -> CacheMemory {
        let live_bytes = match &self.backing {
            Backing::Growing(c) => {
                (0..c.num_layers())
                    .filter_map(|l| c.peek(l))
                    .fold(0usize, |acc, (k, v)| {
                        acc.saturating_add(tensor_bytes(k))
                            .saturating_add(tensor_bytes(v))
                    })
            }
            Backing::Static(layers) => layers
                .iter()
                .flatten()
                .fold(0usize, |acc, l| acc.saturating_add(l.bytes())),
            // The pool is shared; this sequence's share is the token slots it has reserved.
            Backing::Paged(c) => c.reserved_tokens().saturating_mul(self.bytes_per_position),
        };
        CacheMemory {
            live_bytes,
            checkpoint_bytes: 0,
        }
    }

    fn kv_kind(&self) -> KvCacheKind {
        match self.backing {
            Backing::Static(_) => KvCacheKind::Static,
            Backing::Growing(_) | Backing::Paged(_) => KvCacheKind::Growing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> KvLayout {
        let shape = |kv_heads, key_dim, value_dim| {
            Some(LayerKvShape {
                kv_heads,
                key_dim,
                value_dim,
                device: Device::Cpu,
            })
        };
        // A uniform layer, a wider (Gemma 4 full-attention-like) layer, an MLA-like layer with a
        // narrower value, and a KV-shared tail layer that caches nothing.
        KvLayout {
            layers: vec![shape(2, 4, 4), shape(1, 8, 8), shape(3, 6, 4), None],
            dtype: DType::F32,
        }
    }

    fn step(h: usize, s: usize, d: usize, phase: f32) -> Tensor {
        Tensor::arange(0f32, (h * s * d) as f32, &Device::Cpu)
            .unwrap()
            .affine(1.0, phase as f64)
            .unwrap()
            .reshape((1, h, s, d))
            .unwrap()
    }

    fn feed(cache: &mut StepKvCache, s: usize, phase: f32) -> Vec<(Tensor, Tensor)> {
        [(0usize, 2usize, 4usize, 4usize), (1, 1, 8, 8), (2, 3, 6, 4)]
            .iter()
            .map(|&(layer, h, dk, dv)| {
                cache
                    .update(layer, &step(h, s, dk, phase), &step(h, s, dv, phase + 0.5))
                    .unwrap()
            })
            .collect()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn layout_prices_every_caching_layer_and_skips_the_shared_tail() {
        let l = layout();
        assert_eq!(l.bytes_per_position(), (2 * 8 + 16 + 3 * 10) * 4);
        assert_eq!(l.static_bytes(5), 5 * l.bytes_per_position());
        assert_eq!(l.widest_layer(), (3, 8));
        let cache = StepKvCache::preallocated(&l, 5).unwrap();
        assert_eq!(cache.memory().live_bytes, l.static_bytes(5));
        assert_eq!(cache.kv_capacity(), Some(5));
        assert_eq!(cache.kv_kind(), KvCacheKind::Static);
        assert!(
            cache.storage_addresses(3).is_none(),
            "the tail caches nothing"
        );
        assert!(StepKvCache::preallocated(&l, 0).is_err());
    }

    /// The three backings hold the same K/V through writes and rollbacks; the static one never
    /// moves its buffers.
    #[test]
    fn backings_agree_through_writes_and_rollback() {
        let l = layout();
        let mut fixed = StepKvCache::preallocated(&l, 16).unwrap();
        let mut growing = StepKvCache::growing(&l);
        let addresses = fixed.storage_addresses(0).unwrap().unwrap();
        for (s, phase) in [(4usize, 0.0f32), (1, 10.0), (3, 20.0)] {
            let a = feed(&mut fixed, s, phase);
            let b = feed(&mut growing, s, phase);
            for ((ak, av), (bk, bv)) in a.iter().zip(&b) {
                assert_eq!(host(ak), host(bk));
                assert_eq!(host(av), host(bv));
            }
        }
        assert_eq!(DecodeCache::len(&fixed), 8);
        assert_eq!(DecodeCache::len(&growing), 8);
        fixed.rollback_to(5).unwrap();
        growing.rollback_to(5).unwrap();
        assert!(fixed.rollback_to(6).is_err(), "past the end");
        let a = feed(&mut fixed, 2, 30.0);
        let b = feed(&mut growing, 2, 30.0);
        for ((ak, _), (bk, _)) in a.iter().zip(&b) {
            assert_eq!(ak.dims()[2], 7);
            assert_eq!(host(ak), host(bk));
        }
        assert_eq!(fixed.storage_addresses(0).unwrap().unwrap(), addresses);
        assert_eq!(growing.kv_kind(), KvCacheKind::Growing);
        assert!(growing.memory().live_bytes > 0);
        assert!(fixed
            .update(3, &step(1, 1, 4, 0.0), &step(1, 1, 4, 0.0))
            .is_err());
        growing.set_rope_delta(-3);
        DecodeCache::reset(&mut growing);
        assert_eq!(DecodeCache::len(&growing), 0);
        assert_eq!(growing.rope_delta(), 0);
    }

    /// The paged backing sits behind the same seam: exact rollback, memory as reserved slots.
    #[test]
    fn paged_backing_rolls_back_exactly() {
        let uniform = KvLayout {
            layers: vec![
                Some(LayerKvShape {
                    kv_heads: 2,
                    key_dim: 4,
                    value_dim: 4,
                    device: Device::Cpu,
                });
                2
            ],
            dtype: DType::F32,
        };
        let mut paged = StepKvCache::paged(PagedKvCache::new(2, 4), &uniform);
        let mut growing = StepKvCache::growing(&uniform);
        for cache in [&mut paged, &mut growing] {
            for layer in 0..2 {
                cache
                    .update(layer, &step(2, 6, 4, 0.0), &step(2, 6, 4, 1.0))
                    .unwrap();
            }
            cache.rollback_to(3).unwrap();
        }
        assert!(paged.as_paged().is_some());
        assert_eq!(DecodeCache::len(&paged), 3);
        let (pk, _) = paged
            .update(0, &step(2, 1, 4, 9.0), &step(2, 1, 4, 9.0))
            .unwrap();
        let (gk, _) = growing
            .update(0, &step(2, 1, 4, 9.0), &step(2, 1, 4, 9.0))
            .unwrap();
        assert_eq!(host(&pk), host(&gk));
        assert_eq!(paged.kv_kind(), KvCacheKind::Growing);
        assert_eq!(
            paged.memory().live_bytes,
            paged.as_paged().unwrap().reserved_tokens() * uniform.bytes_per_position()
        );
    }
}
