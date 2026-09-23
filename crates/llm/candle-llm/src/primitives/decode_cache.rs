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
//! `Qwen35Cache` keeps a per-token checkpoint ring (sc-24131): every position of the last verify
//! step is restorable, older ones are the typed refusal. The contract is exact: after
//! `rollback_to(n)` the logits for position `n` must equal a fresh decode to `n` (the `Qwen35Cache`
//! rollback test is the gate).
//!
//! ## Memory honesty
//! [`DecodeCache::memory`] reports **logical** bytes — the byte size of every tensor the cache
//! currently references, counting each reference once. A preallocated buffer (a static KV cache,
//! a checkpoint ring) counts in full from the moment it exists — it is what the request holds —
//! with a ring's live slot under `live_bytes` and its other slots under `checkpoint_bytes`; a
//! narrowed view keeps its full backing buffer alive until the next append copies it. The number
//! is therefore the cache's own accounting, not the allocator's, and is labelled as such —
//! device-level peaks come from the allocator (`mem_get_info`) in the bench harness.

use candle_core::Tensor;

use crate::error::Result;
use crate::primitives::kv_cache::KvCacheKind;

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

    /// Ask the cache to be able to roll back at least `n` positions from wherever it is, so a
    /// caller that takes `n` single-token steps past a position can still roll back to it — the
    /// draft-model proposer's `K + 1` draft steps before the target verifies (sc-24130). A cache
    /// that can roll back to any position without checkpoints ignores it (the default); a cache
    /// with a checkpoint ring deepens it (sc-24131), which allocates and so may fail — closed,
    /// leaving the cache as it was — and whoever admits such a request prices the extra states
    /// (E6). Never lowers an existing retention.
    fn retain_checkpoints(&mut self, n: usize) -> Result<()> {
        let _ = n;
        Ok(())
    }

    /// The cache's logical memory accounting.
    fn memory(&self) -> CacheMemory;

    /// Which KV cache implementation backs the cache — reported per request as
    /// [`DecodeRecord::kv_cache`](crate::decode::DecodeRecord::kv_cache). Defaults to
    /// [`KvCacheKind::Growing`]; a cache built on [`StaticKvCache`](crate::primitives::StaticKvCache)
    /// reports [`KvCacheKind::Static`].
    fn kv_kind(&self) -> KvCacheKind {
        KvCacheKind::Growing
    }

    /// Whether the cache can back a CUDA-graph replay (story sc-24134, E5): every tensor a step
    /// reads or writes lives at an address that does not change across steps and rollbacks, and
    /// every per-step position the kernels need is read from device data the cache stages
    /// ([`stage_positions`](Self::stage_positions)). `Err` names why not — a stable lower-case
    /// label the runner reports as the fallback reason (`cache_not_graph_capable` by default; a
    /// hybrid cache whose recurrent state is still replaced per step says so). Checked before any
    /// capture; a cache that answers `Ok` and is wrong is caught by the runner's bit-exact
    /// self-checks, so this is a declaration, not the only gate.
    fn graph_support(&self) -> std::result::Result<(), &'static str> {
        Err("cache_not_graph_capable")
    }

    /// Write the cache's current position(s) into the device tensor(s) its model's kernels read
    /// them from (a small host->device upload, issued **outside** any capture: the model calls
    /// this at the start of an eager step, the graph runner before each replay). Must be a
    /// no-op while [`graph::capturing`](crate::decode::graph::capturing) is set. The default
    /// does nothing (a cache whose model takes positions from Rust-side scalars).
    fn stage_positions(&mut self) -> Result<()> {
        Ok(())
    }

    /// The bookkeeping a step of `n` tokens would do — the capacity check, the rollback
    /// checkpoint, the per-layer offsets — when the device work for that step was produced by a
    /// graph replay instead of the model's forward. Only called on a cache whose
    /// [`graph_support`](Self::graph_support) is `Ok`; the default refuses.
    fn replay_advance(&mut self, n: usize) -> Result<()> {
        let _ = n;
        Err(crate::error::Error::Unsupported(
            "DecodeCache::replay_advance: this cache cannot back a graph replay".into(),
        ))
    }

    /// An identity for the device buffers a graph would be captured against: two caches with
    /// different identities never share graphs. The default is the cache's own address, which
    /// holds while the cache is not moved between steps (it never is: the drivers borrow it for
    /// the whole request); a cache with stable device buffers may answer with their address.
    fn graph_identity(&self) -> usize {
        self as *const Self as *const u8 as usize
    }
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
