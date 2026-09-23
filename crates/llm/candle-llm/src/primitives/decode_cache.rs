//! The decode-cache seam (epic sc-24128, story sc-24129).
//!
//! [`DecodeCache`] is the **model-agnostic** view of "everything a decoder carries between steps":
//! for a softmax-only decoder that is the growing KV cache; for the Qwen3.6/3.8 hybrid it is the
//! per-layer KV *plus* the Gated DeltaNet recurrent state. The fast-decode machinery built on top
//! (the unified speculative engine, static KV, the CUDA-graph runner) only ever needs three things
//! from a cache — how long it is, how to roll it back after a rejected draft run, and how much
//! device memory it holds — so that is the whole contract. Model files implement it; `decode/`
//! consumes it through [`StepModel`](crate::decode::StepModel).
//!
//! ## Rollback semantics
//! [`DecodeCache::rollback_to`] drops every position past `n` so the next step continues **as if
//! positions `n..` had never been decoded**. A growing KV cache can satisfy any `n` by narrowing;
//! a recurrent state cannot be inverted, so a hybrid cache may only roll back to positions it holds
//! a **checkpoint** for and must return an error otherwise (never silently approximate). The
//! contract is exact: after `rollback_to(n)` the logits for position `n` must equal a fresh decode
//! to `n` (the `Qwen35Cache` rollback test is the gate).
//!
//! ## Memory honesty
//! [`DecodeCache::memory`] reports **logical** bytes — the byte size of every tensor the cache
//! currently references, counting each reference once. Candle tensors are reference-counted, so a
//! checkpoint that shares a buffer with the live state costs no extra device memory but *is* counted
//! again under `checkpoint_bytes`; a narrowed view keeps its full backing buffer alive until the
//! next append copies it. The number is therefore the cache's own accounting, not the allocator's,
//! and is labelled as such — device-level peaks come from the allocator (`mem_get_info`) in the
//! bench harness.

use candle_core::Tensor;

use crate::error::Result;

/// The cache's own accounting of what it references, in bytes (see the module docs for what
/// "logical" means).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheMemory {
    /// Bytes referenced by the live state (KV tensors, recurrent states).
    pub live_bytes: usize,
    /// Bytes referenced by rollback checkpoints, counted separately because they usually share
    /// buffers with the live state.
    pub checkpoint_bytes: usize,
}

impl CacheMemory {
    /// `live_bytes + checkpoint_bytes` (saturating).
    pub fn total_bytes(&self) -> usize {
        self.live_bytes.saturating_add(self.checkpoint_bytes)
    }
}

/// Byte size of one tensor's elements (`elem_count × dtype size`).
pub fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count().saturating_mul(t.dtype().size_in_bytes())
}

/// What the decode machinery needs from a model's per-request state.
pub trait DecodeCache {
    /// Number of sequence positions committed so far — the position of the next token.
    fn len(&self) -> i32;

    /// `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop positions `n..`, so the next step continues from position `n`. `n == len()` is a no-op;
    /// `n > len()` is an error; a cache that cannot reconstruct position `n` exactly must return an
    /// error rather than approximate (see the module docs).
    fn rollback_to(&mut self, n: i32) -> Result<()>;

    /// Drop everything, returning to the freshly-constructed condition.
    fn reset(&mut self);

    /// The cache's logical memory accounting.
    fn memory(&self) -> CacheMemory;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn tensor_bytes_counts_elements_times_dtype() {
        let t = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(tensor_bytes(&t), 2 * 3 * 4 * 4);
        let t = Tensor::zeros((5,), DType::BF16, &Device::Cpu).unwrap();
        assert_eq!(tensor_bytes(&t), 10);
    }

    #[test]
    fn cache_memory_total_saturates() {
        let m = CacheMemory {
            live_bytes: usize::MAX,
            checkpoint_bytes: 1,
        };
        assert_eq!(m.total_bytes(), usize::MAX);
        let m = CacheMemory {
            live_bytes: 3,
            checkpoint_bytes: 4,
        };
        assert_eq!(m.total_bytes(), 7);
    }
}
