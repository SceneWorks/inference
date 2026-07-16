//! Paged KV cache — strategy A++ (pooled KV + batched index_select gather), epic 7253 stories 7257
//! and **7453**.
//!
//! PagedAttention-style KV management without a custom kernel. Each sequence's keys/values live in
//! fixed-size **blocks** drawn from a shared [`BlockPool`]; a per-sequence **block table** records
//! which physical blocks hold its tokens, and blocks are allocated on demand. Before attention the
//! sequence's blocks are **gathered** back into a contiguous tensor and fed to the stock
//! [`sdpa`](crate::primitives::sdpa) (single-sequence) or to one
//! `candle_flash_attn::flash_attn_varlen` over the whole batch (the continuous
//! `Throughput` path) — so the cache is a drop-in behind the [`KvCache`]
//! trait and the decoder never changes.
//!
//! ## Pooled storage (stories 7453, 7467)
//! The pool stores each layer's keys/values as **one** contiguous tensor
//! `[capacity · block_size, n_kv_heads, head_dim]` (token-major — block `id` owns the contiguous row
//! range `[id · block_size, (id+1) · block_size)`), rather than a `Vec` of per-block tensors. Three
//! consequences make both the batched gather and the batched write cheap:
//! - **A gather is one kernel.** A sequence's tokens (or, in the continuous `Throughput` step, *every*
//!   active sequence's tokens at once) gather with a single [`Tensor::index_select`] over a token-slot
//!   index tensor, instead of an `O(blocks)` `cat`. Because the rows are already token-major, the
//!   gather feeds the varlen kernel's `[Σ lₖ, n_kv_heads, head_dim]` layout with no per-sequence
//!   `squeeze`/`transpose`/`cat`. This is what removes the host-dispatch-bound gather that flatlined
//!   the continuous `Throughput` decode (sc-7258's measurement → sc-7453).
//! - **A batched write is one kernel** (story 7467). The continuous `Throughput` step scatters *every*
//!   active sequence's new token into its pooled slot with a single in-place [`Tensor::scatter_set`]
//!   per side ([`BlockPool::scatter_write`]) over a target-slot index — the write-side analogue of the
//!   gather — instead of an `O(N)` per-sequence [`Tensor::slice_set`] loop. With the gather collapsed
//!   by sc-7453 the per-sequence write was the residual launch-latency cost (sc-7453's measurement →
//!   this story); `scatter_set` touches only the written rows (no whole-pool copy, unlike the
//!   allocating `scatter`/`index_add`).
//! - **Single-sequence writes stay in place.** The `update` (`Exact`/drop-in) path writes one
//!   sequence's run into its physical slot with [`Tensor::slice_set`] — a bounded `copy2d`, no
//!   whole-pool copy and no per-block tensor churn.
//!
//! ## Why paging
//! A growing-concat cache reserves nothing it does not use, but it also cannot **share** storage. The
//! pool's two wins:
//! - **No max-context reservation**: a sequence holds `ceil(len / block_size)` blocks, never a
//!   pre-reserved `max_context` slab — [`PagedKvCache::reserved_tokens`] vs a naive allocator is the
//!   measured saving.
//! - **Copy-on-write prefix sharing**: a full block is immutable once written, so sequences sharing a
//!   prompt prefix point at the **same physical blocks** ([`PagedKvCache::new_seeded`]); only each
//!   sequence's private partial *tail* block ever diverges, so no block is ever copied mid-write.
//!
//! ## Correctness
//! The gather returns exactly the same per-position keys/values a contiguous cache holds, in the same
//! order — so a sequence decoded with a paged cache is **token-for-token identical** to one decoded
//! with [`ContiguousKvCache`](super::ContiguousKvCache). Per-sequence caches mean each sequence
//! attends only its own real keys (no padding mask), so a **ragged** batch (sequences of differing
//! lengths) is handled bit-exactly — what the left-padded contiguous batch can only approximate at
//! sub-ULP. `slice_set`/`index_select` move bytes only (no arithmetic), so the pooled layout is
//! bit-for-bit identical to the old per-block storage.
//!
//! Block id lifetimes — allocation, recycling, and the copy-on-write reference counts — are the
//! backend-neutral [`core_llm::paging::BlockAllocator`] policy; a freed id is only ever handed back out
//! once its block has no referent, so an in-place write never races a reader. The pool is
//! `Rc<RefCell<…>>`-shared and single-threaded: a cache is a transient per-decode object that lives on
//! one thread (the model itself stays `Send`/`Sync`).

use std::cell::RefCell;
use std::rc::Rc;

use candle_core::{DType, Device, Tensor};

use core_llm::paging::BlockAllocator;

use crate::error::{Error, Result};
use crate::primitives::kv_cache::{KvCache, SEQ_AXIS};

/// Smallest pooled capacity (in blocks) allocated on first use; the pool doubles past it on demand.
const MIN_POOL_BLOCKS: usize = 8;

/// The lazily-initialized pooled tensor storage, sized from the first key/value written.
#[derive(Debug)]
struct PoolStore {
    n_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
    /// Capacity in **blocks**; the per-layer tensors hold `capacity_blocks · block_size` rows.
    capacity_blocks: usize,
    /// Per layer, `[capacity_blocks · block_size, n_kv_heads, head_dim]` (token-major; keys
    /// already-RoPE'd). Block `id` owns rows `[id · block_size, (id+1) · block_size)`.
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

/// A pool of fixed-size physical KV blocks backed by **one contiguous tensor per layer**, shared by
/// the [`PagedKvCache`]s that draw from it. Block id lifetimes (allocation, recycling, copy-on-write
/// reference counts) are the backend-neutral [`core_llm::paging::BlockAllocator`] policy; this pool
/// adds the per-id Candle tensor storage and the batched gather.
#[derive(Debug)]
pub struct BlockPool {
    block_size: usize,
    alloc: BlockAllocator,
    /// Pooled per-layer storage, lazily initialized on the first write (when the head shape / dtype /
    /// device are first known).
    store: Option<PoolStore>,
}

impl BlockPool {
    /// A pool handing out `block_size`-token blocks.
    pub fn new(block_size: usize) -> Rc<RefCell<Self>> {
        assert!(block_size > 0, "block_size must be > 0");
        Rc::new(RefCell::new(Self {
            block_size,
            alloc: BlockAllocator::new(),
            store: None,
        }))
    }

    /// Token capacity of one block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks currently live (refcount > 0).
    pub fn live_blocks(&self) -> usize {
        self.alloc.live_blocks()
    }

    /// Number of blocks shared by more than one sequence (refcount > 1) — the copy-on-write win.
    pub fn shared_blocks(&self) -> usize {
        self.alloc.shared_blocks()
    }

    /// High-water mark of simultaneously-live blocks since construction.
    pub fn peak_live_blocks(&self) -> usize {
        self.alloc.peak_live_blocks()
    }

    /// Token slots reserved across all live blocks (`live_blocks · block_size`) — the apples-to-apples
    /// figure to compare against a naive `sequences · max_context` reservation.
    pub fn reserved_tokens(&self) -> usize {
        self.live_blocks() * self.block_size
    }

    /// Initialize the pooled storage on first use from the head shape / dtype / device of the first
    /// key/value written. Idempotent. `pub(crate)` so a batched prefill can seed it from the model's
    /// cfg before any write (story 7485 — see [`PagedKvCache::ensure_pool_store`]).
    pub(crate) fn ensure_store(
        &mut self,
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        if self.store.is_some() {
            return Ok(());
        }
        let capacity_blocks = MIN_POOL_BLOCKS;
        let rows = capacity_blocks * self.block_size;
        let mk = || Tensor::zeros((rows, n_kv_heads, head_dim), dtype, device);
        let k = (0..num_layers)
            .map(|_| mk())
            .collect::<candle_core::Result<Vec<_>>>()?;
        let v = (0..num_layers)
            .map(|_| mk())
            .collect::<candle_core::Result<Vec<_>>>()?;
        self.store = Some(PoolStore {
            n_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
            capacity_blocks,
            k,
            v,
        });
        Ok(())
    }

    /// Grow the pooled tensors so they cover at least `blocks` blocks (doubling past the request), in
    /// place-preserving the existing rows. A no-op when already large enough.
    fn ensure_capacity(&mut self, blocks: usize) -> Result<()> {
        let bs = self.block_size;
        let store = self
            .store
            .as_mut()
            .expect("pool store initialized before capacity growth");
        if blocks <= store.capacity_blocks {
            return Ok(());
        }
        let new_cap = blocks.max(store.capacity_blocks * 2);
        let rows = new_cap * bs;
        for l in 0..store.k.len() {
            let nk = Tensor::zeros(
                (rows, store.n_kv_heads, store.head_dim),
                store.dtype,
                &store.device,
            )?;
            let nv = Tensor::zeros(
                (rows, store.n_kv_heads, store.head_dim),
                store.dtype,
                &store.device,
            )?;
            // Carry the existing block rows forward (their physical ids are unchanged).
            nk.slice_set(&store.k[l], 0, 0)?;
            nv.slice_set(&store.v[l], 0, 0)?;
            store.k[l] = nk;
            store.v[l] = nv;
        }
        store.capacity_blocks = new_cap;
        Ok(())
    }

    /// Allocate a fresh block (refcount 1), growing the pooled tensors to cover it. The allocator
    /// reuses a freed id when available; a recycled id's rows are overwritten by the next write.
    fn alloc_block(&mut self) -> Result<usize> {
        let id = self.alloc.alloc();
        let cap = self.alloc.capacity();
        self.ensure_capacity(cap)?;
        Ok(id)
    }

    /// Add a reference to `id` (a sequence adopting a shared block).
    fn retain(&mut self, id: usize) {
        self.alloc.retain(id);
    }

    /// Drop a reference to `id`; the rows stay (to be overwritten when the id is recycled).
    fn release(&mut self, id: usize) {
        self.alloc.release(id);
    }

    /// Refcount of `id` (for copy-on-write checks).
    fn refcount(&self, id: usize) -> usize {
        self.alloc.refcount(id)
    }

    /// Write `k`/`v` (`[run, n_kv_heads, head_dim]`, token-major) into `layer`'s pooled tensors at the
    /// contiguous row range starting at `start_slot` — an in-place `copy2d`, no whole-pool copy.
    fn write_run(&self, layer: usize, start_slot: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let store = self
            .store
            .as_ref()
            .expect("pool store initialized before write");
        store.k[layer].slice_set(k, 0, start_slot)?;
        store.v[layer].slice_set(v, 0, start_slot)?;
        Ok(())
    }

    /// Gather `layer`'s keys and values at the token slots in `index` (`[Σ rows]`, u32) into
    /// `[Σ rows, n_kv_heads, head_dim]` pairs — one `index_select` kernel per side. The continuous
    /// `Throughput` step builds one `index` spanning every active sequence (each sequence's
    /// [`PagedKvCache::token_slots`] concatenated in batch order) and gathers the whole ragged batch at
    /// once.
    pub fn gather(&self, layer: usize, index: &Tensor) -> Result<(Tensor, Tensor)> {
        let store = self
            .store
            .as_ref()
            .expect("pool store initialized before gather");
        Ok((
            store.k[layer].index_select(index, 0)?,
            store.v[layer].index_select(index, 0)?,
        ))
    }

    /// **Batched fused write (story 7467).** Scatter this step's new keys/values for `layer` into the
    /// pool with **one** in-place [`Tensor::scatter_set`] per side, replacing the continuous
    /// `Throughput` path's `O(N)` per-sequence `BlockPool::write_run`/`slice_set` loop. `k`/`v` are
    /// the whole batch's new tokens token-major `[Σ rows, n_kv_heads, head_dim]` (sequence order), and
    /// `index` is the matching `[Σ rows, n_kv_heads, head_dim]` u32 target-slot tensor — element
    /// `(t, ·, ·)` is token `t`'s physical pool row, broadcast across the head/dim columns the scatter
    /// preserves. `scatter_set` writes **only** the `Σ rows` referenced rows (no whole-pool copy, unlike
    /// the allocating `scatter`/`index_add`), so this is the write-side analogue of the single
    /// `index_select` gather. Block boundaries need no run-splitting — the scattered rows may be
    /// physically non-contiguous.
    pub fn scatter_write(
        &self,
        layer: usize,
        index: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let store = self
            .store
            .as_ref()
            .expect("pool store initialized before scatter write");
        store.k[layer].scatter_set(index, k, 0)?;
        store.v[layer].scatter_set(index, v, 0)?;
        Ok(())
    }
}

/// A single sequence's paged KV cache: a block table into a [`BlockPool`] plus the token→physical-slot
/// map for the gather.
///
/// One cache holds one sequence (`batch_size == 1`); pack concurrency as separate caches over a
/// shared pool. Implements [`KvCache`] so it drops into the streaming decode loop unchanged; the
/// continuous `Throughput` step batches the gather across caches via [`PagedKvCache::reserve_step`] +
/// `PagedKvCache::write_step` + [`BlockPool`]'s gather (see `models::llama`).
#[derive(Debug)]
pub struct PagedKvCache {
    num_layers: usize,
    block_size: usize,
    pool: Rc<RefCell<BlockPool>>,
    /// Physical block ids holding this sequence's tokens, in position order. The last block is partial
    /// when `len` is not a multiple of `block_size`; the rest are full (frozen).
    block_ids: Vec<usize>,
    /// Logical token length.
    len: usize,
    /// Per token `t`, its physical pool row (`block_ids[t / block_size] · block_size + t % block_size`);
    /// `slots.len() == len`. The gather index for the whole sequence.
    slots: Vec<u32>,
    /// The slots written by the most recent [`reserve_step`](PagedKvCache::reserve_step) /
    /// `update`-step — a suffix of `slots`, the rows [`write_step`](PagedKvCache::write_step) fills.
    new_slots: Vec<u32>,
    /// Cached device tensor of `slots`, rebuilt when `len` changes — the single-sequence gather index
    /// (the `update` trait path). `None` until the first write / after a length change.
    index_dev: Option<Tensor>,
}

impl PagedKvCache {
    /// A fresh single-sequence paged cache backed by its own pool.
    pub fn new(num_layers: usize, block_size: usize) -> Self {
        Self::with_pool(BlockPool::new(block_size), num_layers)
    }

    /// A fresh single-sequence paged cache drawing from an existing (shared) pool.
    pub fn with_pool(pool: Rc<RefCell<BlockPool>>, num_layers: usize) -> Self {
        let block_size = pool.borrow().block_size;
        Self {
            num_layers,
            block_size,
            pool,
            block_ids: Vec::new(),
            len: 0,
            slots: Vec::new(),
            new_slots: Vec::new(),
            index_dev: None,
        }
    }

    /// A cache that **shares** `shared_block_ids` (a prior sequence's frozen prefix blocks) from
    /// `pool`, adopting a reference to each. The new sequence starts positioned at
    /// `shared_block_ids.len() · block_size` and recomputes only its suffix — copy-on-write prefix
    /// reuse with zero block copies.
    pub fn new_seeded(
        pool: Rc<RefCell<BlockPool>>,
        num_layers: usize,
        shared_block_ids: &[usize],
    ) -> Self {
        {
            let mut p = pool.borrow_mut();
            for &id in shared_block_ids {
                p.retain(id);
            }
        }
        let mut cache = Self::with_pool(pool, num_layers);
        cache.block_ids = shared_block_ids.to_vec();
        cache.len = shared_block_ids.len() * cache.block_size;
        cache.slots = (0..cache.len).map(|t| cache.slot_of(t)).collect();
        cache
    }

    /// The pool this cache draws from (for accounting / seeding sibling sequences / the batched gather).
    pub fn pool(&self) -> &Rc<RefCell<BlockPool>> {
        &self.pool
    }

    /// The frozen block ids covering this sequence's first `tokens` positions — the shareable prefix
    /// for [`PagedKvCache::new_seeded`]. Rounded **down** to a whole number of blocks (a partial block
    /// is private and not shareable).
    pub fn shareable_prefix_blocks(&self, tokens: usize) -> Vec<usize> {
        let n = tokens / self.block_size;
        self.block_ids[..n.min(self.block_ids.len())].to_vec()
    }

    /// Number of blocks this sequence holds (full blocks plus, if `len` is not block-aligned, its
    /// partial tail block).
    pub fn blocks(&self) -> usize {
        self.block_ids.len()
    }

    /// Token slots this sequence reserves: `blocks · block_size` — real paged allocation, at most
    /// `block_size - 1` over its true length.
    pub fn reserved_tokens(&self) -> usize {
        self.block_ids.len() * self.block_size
    }

    /// This sequence's per-token physical slots `[0, len)` — the gather index the continuous
    /// `Throughput` step concatenates across the batch.
    pub fn token_slots(&self) -> &[u32] {
        &self.slots
    }

    /// This step's newly-reserved token slots (a suffix of [`token_slots`](PagedKvCache::token_slots),
    /// length equal to the last [`reserve_step`](PagedKvCache::reserve_step)'s `s`) — the per-sequence
    /// target rows the continuous `Throughput` step concatenates across the batch into the one fused
    /// [`BlockPool::scatter_write`] index.
    pub fn new_token_slots(&self) -> &[u32] {
        &self.new_slots
    }

    /// Logical token length (the [`KvCache::offset`] for the next step).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence is empty (no tokens cached yet).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Physical pool row of logical token `t`.
    fn slot_of(&self, t: usize) -> u32 {
        (self.block_ids[t / self.block_size] * self.block_size + t % self.block_size) as u32
    }

    /// Advance the block table by `s` tokens: allocate blocks as positions cross block boundaries,
    /// extend `slots`/`new_slots`, and bump `len`. The pool store must already be initialized (the
    /// caller wrote at least one token first, or a prefill ran).
    fn advance(&mut self, s: usize) -> Result<()> {
        let base = self.len;
        self.new_slots.clear();
        for p in base..base + s {
            let bi = p / self.block_size;
            if bi == self.block_ids.len() {
                let id = self.pool.borrow_mut().alloc_block()?;
                self.block_ids.push(id);
            }
            let slot = self.slot_of(p);
            self.slots.push(slot);
            self.new_slots.push(slot);
        }
        self.len = base + s;
        self.index_dev = None;
        Ok(())
    }

    /// Write this step's `k`/`v` (`[1, n_kv_heads, s, head_dim]`, head-major as projected) into the
    /// pooled tensors at `new_slots`, splitting at block boundaries into in-place contiguous runs.
    fn write_step(&self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        // Head-major [1, kvh, s, hd] -> token-major [s, kvh, hd] for the pool's row layout.
        let s = self.new_slots.len();
        let kvh = k.dim(1)?;
        let hd = k.dim(3)?;
        let k_tm = k
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((s, kvh, hd))?
            .contiguous()?;
        let v_tm = v
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((s, kvh, hd))?
            .contiguous()?;
        self.write_step_rows(layer, &k_tm, &v_tm)
    }

    /// **Continuous `Throughput` step, part 2 (token-major).** Write this step's reserved tokens for
    /// `layer` already in token-major `[s, n_kv_heads, head_dim]` layout into the pool — the batched
    /// path transposes the whole projection to token-major **once** before the layer loop, then slices
    /// each sequence's rows here, so there is no per-sequence transpose. Splits at block boundaries into
    /// in-place contiguous `slice_set` runs.
    pub fn write_step_rows(&self, layer: usize, k_tm: &Tensor, v_tm: &Tensor) -> Result<()> {
        let pool = self.pool.borrow();
        let mut j = 0;
        while j < self.new_slots.len() {
            let start = self.new_slots[j];
            // Extend the run while slots stay physically consecutive (within one block).
            let mut run = 1;
            while j + run < self.new_slots.len() && self.new_slots[j + run] == start + run as u32 {
                run += 1;
            }
            let k_sub = k_tm.narrow(0, j, run)?.contiguous()?;
            let v_sub = v_tm.narrow(0, j, run)?.contiguous()?;
            pool.write_run(layer, start as usize, &k_sub, &v_sub)?;
            j += run;
        }
        Ok(())
    }

    /// The single-sequence gather index for `slots`, building (and caching) its device tensor lazily.
    fn gather_index(&mut self) -> Result<Tensor> {
        if let Some(t) = &self.index_dev {
            return Ok(t.clone());
        }
        let device = {
            let pool = self.pool.borrow();
            pool.store
                .as_ref()
                .expect("pool store initialized before gather")
                .device
                .clone()
        };
        let t = Tensor::from_vec(self.slots.clone(), (self.len,), &device)?;
        self.index_dev = Some(t.clone());
        Ok(t)
    }

    /// Gather this sequence's full keys/values as one head-major `[1, n_kv_heads, len, head_dim]` pair
    /// to attend over — the [`KvCache::update`] return (single-sequence / `Exact` path).
    fn gather_head_major(&mut self, layer: usize) -> Result<(Tensor, Tensor)> {
        let index = self.gather_index()?;
        let (k, v) = self.pool.borrow().gather(layer, &index)?; // [len, kvh, hd]
        let kvh = k.dim(1)?;
        let hd = k.dim(2)?;
        let to_hm = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((1, self.len, kvh, hd))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        Ok((to_hm(k)?, to_hm(v)?))
    }

    /// Ensure the shared pool's storage is initialized so a **fresh** admission wave's caches can
    /// [`reserve_step`](PagedKvCache::reserve_step) before any write. The per-sequence
    /// [`KvCache::update`] normally seeds the store lazily from the first key written, but a **batched
    /// prefill** (story 7485) reserves the whole wave's slots up front, before the first scatter — so
    /// the store must already exist. Idempotent (a no-op once any sequence has prefilled through this
    /// pool, e.g. every decode step). `n_kv_heads`/`head_dim`/`dtype`/`device` are the model's cfg head
    /// shape, compute dtype, and device — the same values the first `update` would have used.
    pub fn ensure_pool_store(
        &self,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        self.pool
            .borrow_mut()
            .ensure_store(self.num_layers, n_kv_heads, head_dim, dtype, device)
    }

    /// **Continuous `Throughput` step, part 1.** Reserve `s` new positions: allocate blocks, extend the
    /// block table, and record `new_slots` — without writing data (that is per-layer
    /// `write_step`). The pool store must already be initialized — every
    /// continuous lane prefills first (through [`KvCache::update`], or, for a batched-prefill wave,
    /// [`ensure_pool_store`](PagedKvCache::ensure_pool_store)).
    pub fn reserve_step(&mut self, s: usize) -> Result<()> {
        if self.pool.borrow().store.is_none() {
            return Err(Error::Msg(
                "PagedKvCache::reserve_step before the pool was initialized by a prefill".into(),
            ));
        }
        self.advance(s)
    }

    /// **Continuous `Throughput` step, part 2.** Write this step's reserved tokens for `layer` into the
    /// pool. `k`/`v` are head-major `[1, n_kv_heads, s, head_dim]` (the per-sequence slice of the
    /// batched projection). Call once per layer after [`reserve_step`](PagedKvCache::reserve_step).
    pub fn write_step_layer(&self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        self.write_step(layer, k, v)
    }
}

impl KvCache for PagedKvCache {
    fn update(&mut self, layer: usize, keys: &Tensor, values: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = keys.dims()[0];
        if b != 1 {
            return Err(Error::Msg(format!(
                "PagedKvCache is single-sequence (batch 1); got batch {b}"
            )));
        }
        let step = keys.dims()[SEQ_AXIS];
        // The block layout advances once per step, at the first layer (every layer adds the same
        // tokens at the same positions in lockstep), so a block carries all layers consistently.
        if layer == 0 {
            let (kvh, hd) = (keys.dims()[1], keys.dims()[3]);
            self.pool.borrow_mut().ensure_store(
                self.num_layers,
                kvh,
                hd,
                keys.dtype(),
                keys.device(),
            )?;
            self.advance(step)?;
        }
        self.write_step(layer, keys, values)?;
        self.gather_head_major(layer)
    }

    fn offset(&self) -> i32 {
        self.len as i32
    }

    fn batch_size(&self) -> i32 {
        i32::from(self.len > 0)
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        // Single-sequence: the only valid non-empty keep is `[0]` (a no-op); an empty keep drops it.
        match keep {
            [] => self.reset(),
            [0] => {}
            other => {
                return Err(Error::Msg(format!(
                    "PagedKvCache is single-sequence; retain_sequences expects [] or [0], got {other:?}"
                )))
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(Error::Msg(format!("truncate: negative len {len}")));
        }
        let len = len as usize;
        if len >= self.len {
            return Ok(()); // already at/under the target length
        }
        let keep_blocks = len.div_ceil(self.block_size); // blocks still holding a kept token
                                                         // The boundary block keeps writing past `len` on the next append; if it is shared (copy-on-write
                                                         // prefix), clone it into a fresh private block first so the sharer is never mutated.
        if !len.is_multiple_of(self.block_size) {
            let bi = keep_blocks - 1;
            let id = self.block_ids[bi];
            if self.pool.borrow().refcount(id) > 1 {
                self.cow_unshare_block(bi)?;
            }
        }
        // Release the blocks fully past the kept range and shrink the table / slot map.
        {
            let mut pool = self.pool.borrow_mut();
            for &id in &self.block_ids[keep_blocks..] {
                pool.release(id);
            }
        }
        self.block_ids.truncate(keep_blocks);
        self.slots.truncate(len);
        self.new_slots.clear();
        self.len = len;
        self.index_dev = None;
        Ok(())
    }

    fn reset(&mut self) {
        {
            let mut pool = self.pool.borrow_mut();
            for &id in &self.block_ids {
                pool.release(id);
            }
        }
        self.block_ids.clear();
        self.slots.clear();
        self.new_slots.clear();
        self.len = 0;
        self.index_dev = None;
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl PagedKvCache {
    /// Replace block `bi` with a fresh private copy of its rows so that subsequent in-place writes do
    /// not mutate a copy-on-write-shared block. Used by [`KvCache::truncate`] when the boundary block
    /// is shared.
    fn cow_unshare_block(&mut self, bi: usize) -> Result<()> {
        let old = self.block_ids[bi];
        let bs = self.block_size;
        let device = {
            let pool = self.pool.borrow();
            pool.store
                .as_ref()
                .expect("pool store initialized")
                .device
                .clone()
        };
        let idx: Vec<u32> = (0..bs).map(|o| (old * bs + o) as u32).collect();
        let idx = Tensor::from_vec(idx, (bs,), &device)?;
        // Allocate a private block (may grow the pool — that carries `old`'s rows forward), then copy
        // `old`'s rows into it layer by layer.
        let new_id = self.pool.borrow_mut().alloc_block()?;
        {
            let pool = self.pool.borrow();
            for l in 0..self.num_layers {
                let (k, v) = pool.gather(l, &idx)?;
                pool.write_run(l, new_id * bs, &k.contiguous()?, &v.contiguous()?)?;
            }
        }
        self.pool.borrow_mut().release(old); // drop the shared reference now we hold a private copy
        self.block_ids[bi] = new_id;
        // Re-point the kept slots that fall in this block.
        for (t, slot) in self.slots.iter_mut().enumerate() {
            if t / bs == bi {
                *slot = (new_id * bs + t % bs) as u32;
            }
        }
        Ok(())
    }
}

impl Drop for PagedKvCache {
    fn drop(&mut self) {
        // Release this sequence's blocks so a shared pool reclaims them (and shared prefixes survive
        // until their last referent drops).
        let mut pool = self.pool.borrow_mut();
        for &id in &self.block_ids {
            pool.release(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `[1, h, s, d]` of sequential f32 values starting at `base`, for order/equality checks.
    fn seq(h: usize, s: usize, d: usize, base: f32) -> Tensor {
        let n = h * s * d;
        let data: Vec<f32> = (0..n).map(|i| base + i as f32).collect();
        Tensor::from_vec(data, (1, h, s, d), &Device::Cpu).unwrap()
    }

    fn host(a: &Tensor) -> Vec<f32> {
        a.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn single_step_under_block_stays_in_one_block() {
        let mut c = PagedKvCache::new(1, 4);
        let k = seq(2, 3, 2, 0.0); // 3 tokens, block_size 4 -> one partial block
        let (ka, _) = c.update(0, &k, &k).unwrap();
        assert_eq!(ka.dims(), &[1, 2, 3, 2]);
        assert_eq!(c.offset(), 3);
        assert_eq!(c.blocks(), 1, "one (partial) block allocated");
        assert_eq!(c.pool().borrow().live_blocks(), 1);
    }

    #[test]
    fn crossing_block_boundary_allocates_and_gather_preserves_order() {
        let mut c = PagedKvCache::new(1, 2);
        // First step: 2 tokens -> exactly one full block.
        let k0 = seq(1, 2, 1, 0.0); // values [0, 1]
        let (g0, _) = c.update(0, &k0, &k0).unwrap();
        assert_eq!(host(&g0), vec![0.0, 1.0]);
        assert_eq!(c.blocks(), 1);
        assert_eq!(c.offset(), 2);
        // Second step: 3 tokens -> one more full block + a 1-token partial; total 5, 3 blocks.
        let k1 = seq(1, 3, 1, 2.0); // values [2, 3, 4]
        let (g1, _) = c.update(0, &k1, &k1).unwrap();
        assert_eq!(c.offset(), 5);
        assert_eq!(c.blocks(), 3);
        assert_eq!(
            host(&g1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            "gather is in position order"
        );
    }

    #[test]
    fn matches_contiguous_cache_step_by_step() {
        use crate::primitives::ContiguousKvCache;
        let mut paged = PagedKvCache::new(2, 4);
        let mut contig = ContiguousKvCache::new(2);
        let mut off = 0.0;
        for step in [5, 1, 1, 4, 1] {
            for layer in 0..2 {
                let k = seq(2, step, 3, off + layer as f32 * 100.0);
                let v = seq(2, step, 3, off + 50.0 + layer as f32 * 100.0);
                let (pk, pv) = paged.update(layer, &k, &v).unwrap();
                let (ck, cv) = contig.update(layer, &k, &v).unwrap();
                assert_eq!(host(&pk), host(&ck), "step {step} layer {layer} keys");
                assert_eq!(host(&pv), host(&cv), "step {step} layer {layer} values");
            }
            off += (step * 6) as f32;
        }
        assert_eq!(paged.offset(), contig.offset());
    }

    #[test]
    fn reserved_tokens_tracks_blocks_not_max_context() {
        let mut c = PagedKvCache::new(1, 16);
        let k = seq(1, 20, 1, 0.0); // 20 tokens -> 1 full block + 4-token partial = 2 blocks
        c.update(0, &k, &k).unwrap();
        assert_eq!(c.blocks(), 2);
        // Reserved = 2 blocks * 16 = 32; far below a naive max_context (e.g. 2048).
        assert_eq!(c.reserved_tokens(), 32);
        assert!(c.reserved_tokens() < 2048);
    }

    #[test]
    fn shared_prefix_blocks_are_refcounted_not_copied() {
        let pool = BlockPool::new(2);
        let mut a = PagedKvCache::with_pool(pool.clone(), 1);
        // 4 tokens -> 2 full shareable blocks.
        let k = seq(1, 4, 1, 0.0);
        a.update(0, &k, &k).unwrap();
        assert_eq!(pool.borrow().live_blocks(), 2);
        assert_eq!(pool.borrow().shared_blocks(), 0);

        // A sibling sequence adopts both prefix blocks (copy-on-write share).
        let shared = a.shareable_prefix_blocks(4);
        assert_eq!(shared.len(), 2);
        let mut b = PagedKvCache::new_seeded(pool.clone(), 1, &shared);
        assert_eq!(
            b.offset(),
            4,
            "seeded sequence starts past the shared prefix"
        );
        assert_eq!(
            pool.borrow().live_blocks(),
            2,
            "no new blocks: prefix is shared"
        );
        assert_eq!(pool.borrow().shared_blocks(), 2);

        // B diverges in its own private partial block; the shared full blocks are untouched.
        let bk = seq(1, 1, 1, 99.0);
        let (bg, _) = b.update(0, &bk, &bk).unwrap();
        assert_eq!(
            pool.borrow().shared_blocks(),
            2,
            "divergence touches only B's private block"
        );
        assert_eq!(
            host(&bg),
            vec![0.0, 1.0, 2.0, 3.0, 99.0],
            "shared prefix + private suffix"
        );

        // Dropping B releases its references; the shared blocks return to refcount 1 (still A's).
        drop(b);
        assert_eq!(pool.borrow().shared_blocks(), 0);
        assert_eq!(pool.borrow().live_blocks(), 2);
    }

    #[test]
    fn truncate_within_block_and_across_blocks() {
        let mut c = PagedKvCache::new(1, 4);
        let k = seq(1, 10, 1, 0.0); // values 0..9 -> blocks [0..3][4..7] + partial [8,9]
        c.update(0, &k, &k).unwrap();
        assert_eq!(c.offset(), 10);

        // Case A: within the partial block.
        c.truncate(9).unwrap();
        assert_eq!(c.offset(), 9);
        assert_eq!(
            host(&c.gather_head_major(0).unwrap().0),
            (0..9).map(|x| x as f32).collect::<Vec<_>>()
        );

        // Case B: drop into a full block (it becomes the partial boundary block).
        c.truncate(5).unwrap();
        assert_eq!(c.offset(), 5);
        assert_eq!(
            host(&c.gather_head_major(0).unwrap().0),
            (0..5).map(|x| x as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            c.pool().borrow().live_blocks(),
            2,
            "the dropped block is freed"
        );

        // Continue decoding after truncate writes into the (private) boundary block correctly.
        let nk = seq(1, 1, 1, 100.0);
        let (g, _) = c.update(0, &nk, &nk).unwrap();
        assert_eq!(host(&g), vec![0.0, 1.0, 2.0, 3.0, 4.0, 100.0]);

        // Case C: land exactly on a block boundary (no partial).
        c.truncate(4).unwrap();
        assert_eq!(c.offset(), 4);
        assert_eq!(
            host(&c.gather_head_major(0).unwrap().0),
            (0..4).map(|x| x as f32).collect::<Vec<_>>()
        );

        // No-op for len >= current length.
        c.truncate(100).unwrap();
        assert_eq!(c.offset(), 4);
    }

    #[test]
    fn reset_frees_blocks_back_to_the_pool() {
        let pool = BlockPool::new(2);
        let mut c = PagedKvCache::with_pool(pool.clone(), 1);
        let k = seq(1, 4, 1, 0.0);
        c.update(0, &k, &k).unwrap();
        assert_eq!(pool.borrow().live_blocks(), 2);
        c.reset();
        assert_eq!(pool.borrow().live_blocks(), 0);
        assert_eq!(c.offset(), 0);
    }

    #[test]
    fn batched_reserve_write_gather_matches_update() {
        // The continuous Throughput path (reserve_step + write_step_layer + pool.gather) must produce
        // the same per-sequence keys/values as the single-sequence `update`.
        let pool = BlockPool::new(4);
        let mut a = PagedKvCache::with_pool(pool.clone(), 1);
        let mut b = PagedKvCache::with_pool(pool.clone(), 1);
        // Prefill both (different lengths) through `update`.
        let pa = seq(1, 5, 2, 0.0);
        let pb = seq(1, 3, 2, 100.0);
        a.update(0, &pa, &pa).unwrap();
        b.update(0, &pb, &pb).unwrap();

        // One batched decode step over both (distinct, non-colliding new-token values).
        let ka = seq(1, 1, 2, 70.0);
        let kb = seq(1, 1, 2, 207.0);
        a.reserve_step(1).unwrap();
        b.reserve_step(1).unwrap();
        a.write_step_layer(0, &ka, &ka).unwrap();
        b.write_step_layer(0, &kb, &kb).unwrap();

        // Global gather across both sequences (in batch order), one index_select.
        let mut idx: Vec<u32> = Vec::new();
        idx.extend_from_slice(a.token_slots());
        idx.extend_from_slice(b.token_slots());
        let index = Tensor::from_vec(idx, (a.len() + b.len(),), &Device::Cpu).unwrap();
        let (k_all, _) = pool.borrow().gather(0, &index).unwrap(); // [la+lb, kvh, hd]

        // A's rows are the first la, in order: the 5 prefill tokens (values 0..9) + the new one [70,71].
        let a_rows = host(&k_all.narrow(0, 0, a.len()).unwrap());
        let expect_a: Vec<f32> = (0..10).map(|x| x as f32).chain([70.0, 71.0]).collect();
        assert_eq!(a_rows, expect_a);
        // B's rows are the next lb: 3 prefill tokens (values 100..105) + the new one [207,208].
        let b_rows = host(&k_all.narrow(0, a.len(), b.len()).unwrap());
        let expect_b: Vec<f32> = (0..6)
            .map(|x| 100.0 + x as f32)
            .chain([207.0, 208.0])
            .collect();
        assert_eq!(b_rows, expect_b);
    }

    #[test]
    fn scatter_write_matches_per_seq_loop() {
        // The story 7467 fused write (one `scatter_set` over a broadcast slot index) must land the
        // exact same bytes in the pool as the O(N) per-sequence `write_step_rows`/`slice_set` loop it
        // replaces — over a ragged batch whose new tokens cross block boundaries (non-contiguous slots).
        let bs = 4usize;
        let (kvh, hd) = (2usize, 3usize);
        // Two pools so the two write strategies start from identical (separately prefilled) state.
        let pool_loop = BlockPool::new(bs);
        let pool_scat = BlockPool::new(bs);
        // Differing prefill lengths so the per-sequence block tables (and new-token slots) diverge;
        // length 4 lands exactly on a boundary so its next token opens a fresh block (non-contiguous).
        let lens = [3usize, 4, 7];
        let mk_caches = |pool: &Rc<RefCell<BlockPool>>| -> Vec<PagedKvCache> {
            lens.iter()
                .enumerate()
                .map(|(i, &l)| {
                    let mut c = PagedKvCache::with_pool(pool.clone(), 1);
                    let k = seq(kvh, l, hd, i as f32 * 1000.0);
                    c.update(0, &k, &k).unwrap();
                    c
                })
                .collect()
        };
        let mut caches_loop = mk_caches(&pool_loop);
        let mut caches_scat = mk_caches(&pool_scat);

        // This step's new token per sequence (distinct, non-colliding values), head-major [1,kvh,1,hd].
        let new_k: Vec<Tensor> = (0..lens.len())
            .map(|i| seq(kvh, 1, hd, 50_000.0 + i as f32 * 100.0))
            .collect();
        let new_v: Vec<Tensor> = (0..lens.len())
            .map(|i| seq(kvh, 1, hd, 90_000.0 + i as f32 * 100.0))
            .collect();

        // (a) per-sequence loop: reserve + write each sequence's row via slice_set.
        for (i, c) in caches_loop.iter_mut().enumerate() {
            c.reserve_step(1).unwrap();
            c.write_step_layer(0, &new_k[i], &new_v[i]).unwrap();
        }

        // (b) fused scatter: reserve, build the broadcast slot index over all sequences' new slots, and
        // scatter the batch's new tokens (token-major [ΣS, kvh, hd]) in one call per side.
        for c in caches_scat.iter_mut() {
            c.reserve_step(1).unwrap();
        }
        let cols = kvh * hd;
        let mut wslots: Vec<u32> = Vec::new();
        for c in &caches_scat {
            for &slot in c.new_token_slots() {
                for _ in 0..cols {
                    wslots.push(slot);
                }
            }
        }
        let total_new = wslots.len() / cols;
        let index = Tensor::from_vec(wslots, (total_new, kvh, hd), &Device::Cpu).unwrap();
        // Pack the per-sequence new tokens token-major [ΣS, kvh, hd] (squeeze head-major [1,kvh,1,hd]).
        let to_tm = |t: &Tensor| {
            t.squeeze(0)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let k_tm = Tensor::cat(&new_k.iter().map(to_tm).collect::<Vec<_>>(), 0).unwrap();
        let v_tm = Tensor::cat(&new_v.iter().map(to_tm).collect::<Vec<_>>(), 0).unwrap();
        pool_scat
            .borrow()
            .scatter_write(0, &index, &k_tm, &v_tm)
            .unwrap();

        // Gather each sequence's full KV from both pools — must be bit-identical.
        for (cl, cs) in caches_loop.iter_mut().zip(caches_scat.iter_mut()) {
            let (kl, vl) = cl.gather_head_major(0).unwrap();
            let (ks, vs) = cs.gather_head_major(0).unwrap();
            assert_eq!(host(&kl), host(&ks), "keys: scatter write != per-seq loop");
            assert_eq!(
                host(&vl),
                host(&vs),
                "values: scatter write != per-seq loop"
            );
        }
    }

    #[test]
    fn pool_growth_preserves_data_bitexact() {
        // Drive a single sequence well past the initial pool capacity (MIN_POOL_BLOCKS blocks) so the
        // pooled tensors reallocate several times; the gather must stay bit-exact vs the contiguous
        // cache, proving `ensure_capacity` carries the existing block rows forward.
        use crate::primitives::ContiguousKvCache;
        let bs = 4;
        let mut paged = PagedKvCache::new(1, bs);
        let mut contig = ContiguousKvCache::new(1);
        let mut off = 0.0;
        for step in 0..40 {
            let s = if step % 3 == 0 { 5 } else { 1 }; // mix multi-token + single-token steps
            let k = seq(2, s, 3, off);
            let (pk, pv) = paged.update(0, &k, &k).unwrap();
            let (ck, cv) = contig.update(0, &k, &k).unwrap();
            assert_eq!(host(&pk), host(&ck), "keys diverged at step {step}");
            assert_eq!(host(&pv), host(&cv), "values diverged at step {step}");
            off += (s * 6) as f32;
        }
        assert!(
            paged.blocks() > MIN_POOL_BLOCKS,
            "sequence must outgrow the initial pool capacity to exercise growth (blocks={})",
            paged.blocks()
        );
    }

    #[test]
    fn single_sequence_rejects_batched_update() {
        let mut c = PagedKvCache::new(1, 4);
        let k = Tensor::from_vec(vec![0.0f32; 8], (2, 2, 1, 2), &Device::Cpu).unwrap(); // batch 2
        assert!(c.update(0, &k, &k).is_err());
    }
}
