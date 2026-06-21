//! Backend-neutral paged-KV block allocation policy (epic 7153, story 7169).
//!
//! A paged KV cache stores each sequence's keys/values in fixed-size **blocks** and tracks, per
//! sequence, which physical blocks hold its tokens. The bookkeeping that hands out block ids, reuses
//! freed ones, and **reference-counts** blocks for copy-on-write prefix sharing is identical across
//! backends — only the block *contents* (tensors) are backend-specific. Following the epic's
//! concrete-first rule, that bookkeeping lives here: a backend ([`mlx-llm`], later `candle-llm`)
//! pairs each allocator id with its own per-block tensors and lets this policy decide lifetimes.
//!
//! Blocks are immutable once written, so two sequences sharing a prompt prefix [`retain`] the same
//! ids (refcount > 1) and a block is freed only when its **last** referent [`release`]s it — no block
//! is ever copied. Ids are dense (`0..capacity`) so a backend can index its block storage by id
//! directly; a freed id is recycled by the next [`alloc`], and its slot is overwritten with fresh
//! contents.
//!
//! [`mlx-llm`]: https://github.com/SceneWorks/mlx-llm
//! [`retain`]: BlockAllocator::retain
//! [`release`]: BlockAllocator::release
//! [`alloc`]: BlockAllocator::alloc
//!
//! ```
//! use core_llm::paging::BlockAllocator;
//!
//! let mut a = BlockAllocator::new();
//! let prefix = a.alloc();          // sequence 1 writes a block
//! a.retain(prefix);                // sequence 2 shares it (copy-on-write)
//! assert_eq!(a.refcount(prefix), 2);
//! assert_eq!(a.shared_blocks(), 1);
//!
//! assert_eq!(a.release(prefix), false); // sequence 1 drops it — still referenced
//! assert_eq!(a.release(prefix), true);  // sequence 2 drops it — now freed
//! assert_eq!(a.live_blocks(), 0);
//! assert_eq!(a.alloc(), prefix);        // the freed id is recycled
//! ```

/// Reference-counted allocator of dense block ids for a paged KV cache.
///
/// Tensor-free: it owns only ids and refcounts. A backend keys its per-block tensor storage by the
/// returned ids and mirrors the lifetime decisions ([`alloc`](BlockAllocator::alloc) →
/// [`release`](BlockAllocator::release) returning `true`).
#[derive(Clone, Debug, Default)]
pub struct BlockAllocator {
    /// Refcount per block id (`0` ⇒ the id is free and on `free`).
    refcount: Vec<usize>,
    /// Freed ids available for reuse (LIFO — recycles the most recently freed first).
    free: Vec<usize>,
    /// High-water mark of simultaneously-live blocks.
    peak_live: usize,
}

impl BlockAllocator {
    /// A fresh allocator with no blocks.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a block (refcount 1), reusing a freed id when one is available, else minting the next
    /// dense id. The returned id is `< capacity()`; a recycled id's old contents must be overwritten.
    pub fn alloc(&mut self) -> usize {
        let id = if let Some(id) = self.free.pop() {
            self.refcount[id] = 1;
            id
        } else {
            self.refcount.push(1);
            self.refcount.len() - 1
        };
        let live = self.live_blocks();
        if live > self.peak_live {
            self.peak_live = live;
        }
        id
    }

    /// Add a reference to `id` (a sequence adopting a shared block for copy-on-write reuse).
    ///
    /// Panics if `id` is not currently live.
    pub fn retain(&mut self, id: usize) {
        assert!(self.is_live(id), "retain of free/unknown block {id}");
        self.refcount[id] += 1;
    }

    /// Drop a reference to `id`, returning `true` if that was the **last** reference (so the backend
    /// should free the block's tensors and the id is now recyclable).
    ///
    /// Panics if `id` is not currently live (guards against a double free).
    pub fn release(&mut self, id: usize) -> bool {
        assert!(self.is_live(id), "release of free/unknown block {id}");
        self.refcount[id] -= 1;
        if self.refcount[id] == 0 {
            self.free.push(id);
            true
        } else {
            false
        }
    }

    /// Current reference count of `id` (`0` if free/unknown).
    pub fn refcount(&self, id: usize) -> usize {
        self.refcount.get(id).copied().unwrap_or(0)
    }

    /// Whether `id` is currently allocated (refcount > 0).
    pub fn is_live(&self, id: usize) -> bool {
        self.refcount(id) > 0
    }

    /// Number of blocks currently live (refcount > 0).
    pub fn live_blocks(&self) -> usize {
        self.refcount.iter().filter(|&&r| r > 0).count()
    }

    /// Number of blocks shared by more than one sequence (refcount > 1) — the copy-on-write win.
    pub fn shared_blocks(&self) -> usize {
        self.refcount.iter().filter(|&&r| r > 1).count()
    }

    /// High-water mark of simultaneously-live blocks since construction.
    pub fn peak_live_blocks(&self) -> usize {
        self.peak_live
    }

    /// Number of distinct ids ever minted (live + free) — the size a backend's id-indexed storage
    /// must cover.
    pub fn capacity(&self) -> usize {
        self.refcount.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_mints_dense_ids() {
        let mut a = BlockAllocator::new();
        assert_eq!(a.alloc(), 0);
        assert_eq!(a.alloc(), 1);
        assert_eq!(a.alloc(), 2);
        assert_eq!(a.capacity(), 3);
        assert_eq!(a.live_blocks(), 3);
    }

    #[test]
    fn release_frees_only_on_last_reference() {
        let mut a = BlockAllocator::new();
        let id = a.alloc();
        a.retain(id);
        a.retain(id);
        assert_eq!(a.refcount(id), 3);
        assert!(!a.release(id));
        assert!(!a.release(id));
        assert!(a.release(id), "last reference frees the block");
        assert_eq!(a.refcount(id), 0);
        assert!(!a.is_live(id));
        assert_eq!(a.live_blocks(), 0);
    }

    #[test]
    fn freed_ids_are_recycled_lifo() {
        let mut a = BlockAllocator::new();
        let a0 = a.alloc();
        let a1 = a.alloc();
        assert!(a.release(a0));
        // The freed id is reused before a new one is minted.
        assert_eq!(a.alloc(), a0);
        assert_eq!(a.capacity(), 2, "no new id minted while a free one exists");
        assert!(a.is_live(a1));
    }

    #[test]
    fn shared_blocks_counts_multi_referenced() {
        let mut a = BlockAllocator::new();
        let x = a.alloc();
        let _y = a.alloc();
        assert_eq!(a.shared_blocks(), 0);
        a.retain(x);
        assert_eq!(a.shared_blocks(), 1);
        a.release(x);
        assert_eq!(a.shared_blocks(), 0);
    }

    #[test]
    fn peak_live_tracks_high_water_mark() {
        let mut a = BlockAllocator::new();
        let x = a.alloc();
        let y = a.alloc();
        let z = a.alloc();
        assert_eq!(a.peak_live_blocks(), 3);
        a.release(x);
        a.release(y);
        a.release(z);
        assert_eq!(a.live_blocks(), 0);
        assert_eq!(a.peak_live_blocks(), 3, "peak is a high-water mark, not the current count");
    }

    #[test]
    #[should_panic(expected = "release of free")]
    fn double_free_panics() {
        let mut a = BlockAllocator::new();
        let id = a.alloc();
        assert!(a.release(id));
        a.release(id); // already free
    }
}
