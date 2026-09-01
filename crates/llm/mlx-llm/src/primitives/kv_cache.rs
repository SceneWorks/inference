//! Key/value cache.
//!
//! The cache is the seam the throughput work (P4) plugs into: the dynamic-batch scheduler
//! (story 7167), the prefix cache (7168) and the paged cache (7169/7170) all implement the
//! [`KvCache`] trait so they swap in without the decoder changing. The decoder only ever talks to
//! the trait.
//!
//! [`ContiguousKvCache`] is the day-one implementation: a per-layer growing concat along the
//! sequence axis, modelled on the working mlx-gen caches (prompt-refine `LlamaKvCache`,
//! sensenova/flux2 `Qwen3KvCache`). It is **batch-capable** — the batch axis is real, not
//! hardcoded to 1, so an N-sequence batch with a uniform length works today. Ragged per-sequence
//! offsets (sequences of differing lengths in one batch) need the paged cache, which is exactly
//! why the trait exists.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

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

fn checked_concat_overhead(prior_bytes: u64, incoming_bytes: u64) -> Result<u64> {
    prior_bytes.checked_add(incoming_bytes).ok_or_else(|| {
        crate::error::Error::Msg("dense concat transient overhead overflows u64".into())
    })
}

/// Growing-concat KV cache: one `Option<(K, V)>` slot per layer, concatenated along the sequence
/// axis each step. Correctness-first; the paged cache (P4) is the throughput replacement behind
/// the same trait.
#[derive(Debug)]
pub struct ContiguousKvCache {
    layers: Vec<Option<(Array, Array)>>,
    events: Vec<CacheEvent>,
}

impl ContiguousKvCache {
    /// A fresh cache with `num_layers` empty slots.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            events: Vec::new(),
        }
    }

    /// Borrow the currently-cached `(keys, values)` for `layer`, if any.
    pub fn peek(&self, layer: usize) -> Option<&(Array, Array)> {
        self.layers.get(layer).and_then(|s| s.as_ref())
    }

    /// Construct a cache pre-populated with per-layer `(keys, values)` — the seam the prefix cache
    /// (story 7168) reuses a shared prefix's KV through. Each entry is `[batch, n_kv_heads, seq,
    /// head_dim]` (keys already-RoPE'd); the cache then reports [`KvCache::offset`] equal to that
    /// seq length, so a decoder prefills only the suffix at that offset and attends over the seeded
    /// keys. Layout/length consistency across layers is the caller's responsibility.
    pub fn seeded(layers: Vec<(Array, Array)>) -> Self {
        Self {
            layers: layers.into_iter().map(Some).collect(),
            events: Vec::new(),
        }
    }

    /// Ownership events since construction, for the campaign producer only.
    pub fn events(&self) -> &[CacheEvent] {
        &self.events
    }

    /// Snapshot every layer's cached `(keys, values)` as clones (MLX arrays are refcounted, so this
    /// shares buffers rather than copying), or `None` if any layer is still empty. The prefix cache
    /// stores this after a generation so a later shared-prefix request can be [`seeded`] from it.
    ///
    /// [`seeded`]: ContiguousKvCache::seeded
    pub fn export(&self) -> Option<Vec<(Array, Array)>> {
        self.layers.iter().cloned().collect()
    }

    /// Dense compatibility route used by the experimental packed-cache planner before mutation.
    pub fn packed_preflight(&self, query_length: usize, mask: bool) -> CacheRoute {
        self.preflight_packed(query_length, mask)
    }
}

impl KvCache for ContiguousKvCache {
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let prior = self.layers[layer].as_ref();
        let prior_bytes = prior
            .as_ref()
            .map(|(key, value)| {
                array_bytes(key)?
                    .checked_add(array_bytes(value)?)
                    .ok_or_else(|| {
                        crate::error::Error::Msg("prior KV byte accounting overflows u64".into())
                    })
            })
            .transpose()?;
        let appended_bytes = array_bytes(keys)?
            .checked_add(array_bytes(values)?)
            .ok_or_else(|| crate::error::Error::Msg("appended KV bytes overflow u64".into()))?;
        let merged = match prior {
            Some((pk, pv)) => (
                concatenate_axis(&[pk, keys], SEQ_AXIS)?,
                concatenate_axis(&[pv, values], SEQ_AXIS)?,
            ),
            None => (keys.clone(), values.clone()),
        };
        let merged_bytes = array_bytes(&merged.0)?
            .checked_add(array_bytes(&merged.1)?)
            .ok_or_else(|| crate::error::Error::Msg("merged KV bytes overflow u64".into()))?;
        if let Some(prior_bytes) = prior_bytes {
            // The merged arrays are the retained successor and are accounted once by the
            // persistent event below.  This snapshot is only the additional high-water
            // overhead while immutable concat keeps the prior and incoming arrays alive.
            let coexistence_bytes = checked_concat_overhead(prior_bytes, appended_bytes)?;
            self.events.push(CacheEvent {
                layer,
                operation: "dense_concat_coexistence",
                role: "output",
                lifetime: "transient",
                bytes: coexistence_bytes,
                tokens: u64::try_from(merged.0.shape()[SEQ_AXIS as usize]).map_err(|_| {
                    crate::error::Error::Msg("KV sequence length overflows u64".into())
                })?,
            });
        }
        self.layers[layer] = Some((merged.0.clone(), merged.1.clone()));
        self.events.push(CacheEvent {
            layer,
            operation: "append",
            role: "cache",
            lifetime: "persistent",
            bytes: merged_bytes,
            tokens: u64::try_from(merged.0.shape()[SEQ_AXIS as usize])
                .map_err(|_| crate::error::Error::Msg("KV sequence length overflows u64".into()))?,
        });
        Ok(merged)
    }

    fn offset(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map(|(k, _)| k.shape()[SEQ_AXIS as usize])
            .unwrap_or(0)
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        let idx = Array::from_slice(keep, &[keep.len() as i32]);
        for slot in &mut self.layers {
            if let Some((k, v)) = slot.take() {
                *slot = Some((k.take_axis(&idx, 0)?, v.take_axis(&idx, 0)?));
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(crate::error::Error::Msg(format!(
                "truncate: negative len {len}"
            )));
        }
        let idx = Array::from_slice(&(0..len).collect::<Vec<_>>(), &[len]);
        for slot in &mut self.layers {
            if let Some((k, v)) = slot.take() {
                if k.shape()[SEQ_AXIS as usize] <= len {
                    *slot = Some((k, v)); // already at/under the target length
                } else {
                    *slot = Some((k.take_axis(&idx, SEQ_AXIS)?, v.take_axis(&idx, SEQ_AXIS)?));
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        let bytes = self
            .layers
            .iter()
            .flatten()
            .try_fold(0_u64, |total, (k, v)| {
                let pair = array_bytes(k)?
                    .checked_add(array_bytes(v)?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// `[b, h, s, d]` of sequential f32 values, for shape/equality checks.
    fn arange4(b: i32, h: i32, s: i32, d: i32) -> Array {
        let n = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Array::from_slice(&data, &[b, h, s, d])
    }

    #[test]
    fn first_update_stores_and_returns_input() {
        let mut cache = ContiguousKvCache::new(2);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.batch_size(), 0);

        let k = arange4(1, 2, 3, 4);
        let v = arange4(1, 2, 3, 4);
        let (ka, va) = cache.update(0, &k, &v).unwrap();
        assert_eq!(ka.shape(), &[1, 2, 3, 4]);
        assert_eq!(va.shape(), &[1, 2, 3, 4]);
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
        assert_eq!(ka.shape(), &[1, 2, 4, 4]); // 3 + 1 along seq
        assert_eq!(cache.offset(), 4);
        let event = cache
            .events()
            .iter()
            .find(|event| event.operation == "dense_concat_coexistence")
            .expect("the immutable concat must publish its real coexistence bytes");
        assert_eq!(event.role, "output");
        assert_eq!(event.lifetime, "transient");
        assert_eq!(event.bytes, 256);
    }

    #[test]
    fn supports_batch_greater_than_one() {
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
        let mut cache = ContiguousKvCache::new(1);
        // [1,1,2,2] = [[0,1],[2,3]]
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
        // [1,1,1,2] = [[10,11]]
        let b = Array::from_slice(&[10.0f32, 11.0], &[1, 1, 1, 2]);
        cache.update(0, &a, &a).unwrap();
        let (ka, _) = cache.update(0, &b, &b).unwrap();
        let host = ka
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(host, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0]);
    }

    #[test]
    fn retain_sequences_compacts_batch_rows() {
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
        let (ka, _) = cache.peek(0).unwrap();
        let host = ka
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
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
    fn truncate_slices_sequence_axis() {
        let mut cache = ContiguousKvCache::new(1);
        // [1,1,5,1] = values 0..4 along the seq axis.
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        cache.update(0, &a, &a).unwrap();
        assert_eq!(cache.offset(), 5);
        cache.truncate(3).unwrap();
        assert_eq!(cache.offset(), 3);
        let (k, _) = cache.peek(0).unwrap();
        let host = k
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(host, vec![0.0, 1.0, 2.0]);
        cache.truncate(10).unwrap(); // no-op past the end
        assert_eq!(cache.offset(), 3);
    }

    #[test]
    fn reset_clears_state() {
        let mut cache = ContiguousKvCache::new(2);
        let k = arange4(1, 2, 3, 4);
        cache.update(0, &k, &k).unwrap();
        cache.reset().unwrap();
        assert_eq!(cache.offset(), 0);
        assert!(cache.peek(0).is_none());
        let release = cache.events().last().unwrap();
        assert_eq!(release.operation, "cache_release");
        assert_eq!(release.lifetime, "released");
        assert_eq!(release.bytes, 192);
    }

    #[test]
    fn concat_overhead_fails_closed_on_overflow() {
        assert!(checked_concat_overhead(u64::MAX, 1).is_err());
    }
}
