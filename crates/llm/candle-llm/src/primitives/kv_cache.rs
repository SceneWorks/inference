//! Key/value cache.
//!
//! [`ContiguousKvCache`] is the day-one implementation: a per-layer growing concat along the
//! sequence axis (the Candle port of `mlx-llm`'s `ContiguousKvCache`). It is **batch-capable** — the
//! batch axis is real, not hardcoded to 1. The [`KvCache`] trait is the seam a paged cache (P4)
//! slots in behind so the decoder, which only ever talks to the trait, never changes. The
//! dynamic-batch scheduler (story 7255) retires finished sequences through [`KvCache::retain_sequences`];
//! the prefix cache (story 7256) seeds a fresh cache from a shared prefix's stored KV via
//! [`ContiguousKvCache::seeded`] / [`ContiguousKvCache::export`].

use candle_core::Tensor;

use crate::error::{Error, Result};

/// Layout, per layer, of the cached keys/values: `[batch, n_kv_heads, seq, head_dim]`. Keys are
/// stored already-RoPE'd; values raw. The sequence axis (2) is the one that grows each step.
pub const SEQ_AXIS: usize = 2;

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
            Some((pk, pv)) => (
                Tensor::cat(&[&pk, keys], SEQ_AXIS)?,
                Tensor::cat(&[&pv, values], SEQ_AXIS)?,
            ),
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
