//! Quantized paged KV storage with direct paged fused attention (epic sc-20669, story sc-20680).
//!
//! The paged counterpart of the device-resident packed group-affine cache
//! ([`DenseFallbackPackedDecoderCache`](super::DenseFallbackPackedDecoderCache)): a sequence's K/V
//! are quantized as they are appended and live in fixed-size **pages** drawn from a shared
//! [`PackedPagePool`], and the fused Metal reader walks the sequence's page table to read codes and
//! scale/zero metadata in place. No dense K/V of the history is gathered or retained on a supported
//! call; an unsupported call takes an explicit, counted dense gather fallback.
//!
//! ## Page layout (format version [`PAGED_PACKED_LAYOUT_VERSION`])
//!
//! A page holds `page_tokens` consecutive positions of one sequence for every layer, where
//! `page_tokens` is a positive multiple of the 32-token quantization group. Per layer the pool keeps
//! one array per component, indexed by physical page id:
//!
//! * K codes `[pages, Hkv, page_tokens/32, 32·D·b/8]` (token-group packed) with f16 scale and zero
//!   `[pages, Hkv, page_tokens/32, D]`;
//! * V codes `[pages, Hkv, page_tokens, D·b/8]` (channel-group packed per token) with f16 scale and
//!   zero `[pages, Hkv, page_tokens, D/32]`.
//!
//! The codes, groups and f16 rounding are exactly the contiguous representation's (the same GPU
//! quantizer writes them), so a paged and a contiguous cache fed the same K/V hold bit-identical
//! codes and differ only in where they live. Only completed 32-token groups are paged: the
//! incomplete group stays in a per-sequence dense residual (`[1, Hkv, 32, D]` for K and for V), the
//! same bounded residual the contiguous cache keeps. Page metadata is the per-sequence page table
//! (`[1, columns]` Int32, grown by doubling and updated in place one entry per new page, never
//! rebuilt per step) plus those residuals; [`PagedPackedKvCache::compressed_storage`] counts all of
//! it.
//!
//! ## Pool
//!
//! Page ids are allocated, recycled and reference-counted by the backend-neutral
//! [`BlockAllocator`]; freed ids are reused (lowest first), so a long-running pool fragments and a
//! sequence's page table is in general a non-contiguous, non-monotonic list of ids. The pool's
//! arrays grow by a bounded chunk ([`PackedPagePool::growth_chunk`], one copy per growth), a caller
//! with known demand reserves it exactly ([`PackedPagePool::reserve`]), and
//! [`PackedPagePool::trim`] returns the capacity above the highest live page (sc-20681). The held
//! capacity is reported by [`PackedPagePool::storage`].
//!
//! ## Cache semantics
//!
//! [`PagedPackedKvCache`] is a single-sequence [`KvCache`] like the dense
//! [`PagedKvCache`]. A model step is a whole-step transaction across layers:
//! a fault or cancellation at any layer restores every layer's extents and residuals and returns
//! the pages the step allocated, so the pool's live pages return to their pre-step count. Trim
//! re-stages the retained prefix of a cut group from its quantized values (as the contiguous cache
//! does), and reset/drop return every page.
//!
//! ## Serving lifecycles (story sc-20681)
//!
//! * **Batched decode.** `paged_attention_batch` serves one layer of several sequences on one
//!   pool with a single fused dispatch: a `[B, max_pages]` page table and per-sequence lengths
//!   replace the padding mask a padded batch would need, so jagged sequences decode together
//!   compressed. Each sequence's step stays its own whole-step transaction.
//! * **Prefix sharing with copy-on-write pages.** [`PagedPackedKvCache::fork_prefix`] starts a
//!   sequence on another's quantized prefix by retaining its pages (the pool's reference counts,
//!   [`PackedPagePool::refcount`]); nothing is copied. A sequence about to write into a page it
//!   shares first copies that page into a private one (all layers), so a shared page is never
//!   written in place. Inside a step the copy is provisional: the step's rollback returns the copy
//!   and restores the shared page, its commit drops the shared reference.
//! * **Exact speculative rollback.** [`KvCache::begin_speculation`] arms a window that keeps the
//!   unquantized rows appended from the current position on; the next truncation back into the
//!   window restores the residual from those rows, so the kept state is byte- and
//!   position-identical to having appended only the kept rows (quantization is per group, so
//!   every group completed inside the kept range has the codes it would have had anyway).
//! * **Identity and restore.** A [`PagedCacheIdentity`] names the model, the KV format version,
//!   the page layout version and the page geometry. A [`PagedCacheSnapshot`] carries one
//!   sequence's codes, metadata and residual under its identity, survives a process restart as
//!   bytes, and restores only into a pool of the same identity: any mismatch is refused, never
//!   reused.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use core_llm::paging::BlockAllocator;
use core_llm::{KvCacheReport, KvCompressionFormat, KvCompressionPolicy, KvModelFamily};
use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
use mlx_rs::ops::{concatenate_axis, zeros_dtype};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::attention::{sdpa, AttnMask, SDPA_MAX_FUSED_QLEN};
use crate::primitives::kv_cache::{
    CompressedCacheStorage, KvCache, PackedAttentionMask, PackedCacheEvidence,
    PackedKernelPathEvidence,
};
use crate::primitives::packed_group_affine_kv::{
    array_bytes, dequantize_key_groups, dequantize_value_rows, dtype_bytes, packed_input_shape,
    packed_metal_head_dimension_supported, padded_residual, rows_range, write_rows,
    CompiledKernelHandle, PackedCodeBits, PackedGeometry, PACKED_METAL_QUANT_GROUP_SIZE,
};
use crate::primitives::packed_metal::{
    quantize_group_affine_flush, PackedKernelSelection, PackedMask, PagedPackedAttentionArgs,
    PagedSequenceExtent,
};
use crate::primitives::paged_kv_cache::{BlockPool, PagedKvCache};

/// Version of the paged page layout described in the module docs. The codes themselves are
/// [`core_llm::KV_CACHE_FORMAT_VERSION`]'s representation; this versions where they live.
pub const PAGED_PACKED_LAYOUT_VERSION: u32 = 1;

/// Pages a pool reserves on its first allocation; growth doubles from there.
const MIN_POOL_PAGES: usize = 4;

/// Page-table columns a sequence reserves first; growth doubles from there.
const MIN_TABLE_COLUMNS: usize = 4;

/// Representation identity of a paged cache of `bits`-wide codes.
pub const fn paged_packed_identity(bits: PackedCodeBits) -> &'static str {
    match bits {
        PackedCodeBits::Two => "sc-20680-paged-group-affine-b2-v1",
        PackedCodeBits::Four => "sc-20680-paged-group-affine-b4-v1",
        PackedCodeBits::Eight => "sc-20680-paged-group-affine-b8-v1",
    }
}

fn mlx_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Config(format!("paged packed KV {what} exceeds MLX i32 range")))
}

/// One layer's page arrays (see the module docs for shapes).
#[derive(Debug)]
struct PageArrays {
    key_codes: Array,
    key_scales: Array,
    key_zeros: Array,
    value_codes: Array,
    value_scales: Array,
    value_zeros: Array,
}

impl PageArrays {
    fn all(&self) -> [&Array; 6] {
        [
            &self.key_codes,
            &self.key_scales,
            &self.key_zeros,
            &self.value_codes,
            &self.value_scales,
            &self.value_zeros,
        ]
    }
}

/// Measured storage of a [`PackedPagePool`]: the sizes of the arrays it actually holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagedPoolStorage {
    pub page_tokens: u64,
    /// Pages the pool's arrays have room for (allocated device capacity).
    pub capacity_pages: u64,
    /// Pages currently referenced by a sequence.
    pub live_pages: u64,
    /// High-water mark of live pages.
    pub peak_live_pages: u64,
    /// K and V code arrays of every layer, at capacity.
    pub code_bytes: u64,
    /// K and V scale/zero arrays of every layer, at capacity.
    pub metadata_bytes: u64,
}

/// A shared pool of packed KV pages for one decoder geometry. Single-threaded and `Rc`-shared,
/// like the dense [`BlockPool`] and the engine's MLX device.
#[derive(Debug)]
pub struct PackedPagePool {
    layers: usize,
    kv_heads: usize,
    head_dimension: usize,
    page_tokens: usize,
    bits: PackedCodeBits,
    capacity_pages: usize,
    /// One entry per layer once the pool first allocates; empty before.
    arrays: Vec<PageArrays>,
    alloc: BlockAllocator,
}

impl PackedPagePool {
    /// A pool of `page_tokens`-token pages for `layers` layers of `kv_heads × head_dimension` K/V
    /// in `bits`-wide codes. `page_tokens` must be a positive multiple of the 32-token group and
    /// the head dimension one the fused reader implements (64, 128 or 256).
    pub fn new(
        layers: usize,
        kv_heads: usize,
        head_dimension: usize,
        page_tokens: usize,
        bits: PackedCodeBits,
    ) -> Result<Rc<RefCell<Self>>> {
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        if layers == 0
            || kv_heads == 0
            || !packed_metal_head_dimension_supported(head_dimension)
            || page_tokens == 0
            || !page_tokens.is_multiple_of(group)
        {
            return Err(Error::Unsupported(format!(
                "paged packed KV needs layers > 0, KV heads > 0, head dimension 64/128/256 and a \
                 page of a positive multiple of {group} tokens (got {layers} layers, {kv_heads} \
                 heads, D {head_dimension}, {page_tokens}-token pages)"
            )));
        }
        mlx_i32(kv_heads * page_tokens * head_dimension, "page size")?;
        Ok(Rc::new(RefCell::new(Self {
            layers,
            kv_heads,
            head_dimension,
            page_tokens,
            bits,
            capacity_pages: 0,
            arrays: Vec::new(),
            alloc: BlockAllocator::new(),
        })))
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dimension(&self) -> usize {
        self.head_dimension
    }

    pub fn code_bits(&self) -> PackedCodeBits {
        self.bits
    }

    /// Pages currently referenced by a sequence.
    pub fn live_pages(&self) -> usize {
        self.alloc.live_blocks()
    }

    /// High-water mark of simultaneously live pages.
    pub fn peak_live_pages(&self) -> usize {
        self.alloc.peak_live_blocks()
    }

    /// Pages the pool's arrays currently have room for.
    pub fn capacity_pages(&self) -> usize {
        self.capacity_pages
    }

    /// Whether page `id` is currently allocated.
    pub fn is_live(&self, id: usize) -> bool {
        self.alloc.is_live(id)
    }

    /// Sequences (and prefix-store entries) referencing page `id`; `0` when it is free.
    pub fn refcount(&self, id: usize) -> usize {
        self.alloc.refcount(id)
    }

    /// Pages referenced by more than one sequence: copy-on-write prefix sharing.
    pub fn shared_pages(&self) -> usize {
        self.alloc.shared_blocks()
    }

    /// The identity a cache of this pool carries for `model` (see [`PagedCacheIdentity`]).
    pub fn identity(&self, model: &PagedModelKey) -> PagedCacheIdentity {
        PagedCacheIdentity {
            model: model.name.clone(),
            model_fingerprint: model.fingerprint.clone(),
            format_version: core_llm::KV_CACHE_FORMAT_VERSION,
            layout_version: PAGED_PACKED_LAYOUT_VERSION,
            representation: paged_packed_identity(self.bits).to_owned(),
            code_bits: self.bits.bits(),
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            page_tokens: self.page_tokens,
            layers: self.layers,
            kv_heads: self.kv_heads,
            head_dimension: self.head_dimension,
        }
    }

    fn geometry(&self) -> PackedGeometry {
        PackedGeometry {
            batch: 1,
            heads: self.kv_heads,
            dim: self.head_dimension,
            group: PACKED_METAL_QUANT_GROUP_SIZE,
            bits: self.bits,
        }
    }

    /// Device bytes one page occupies across every layer: `(codes, scale/zero metadata)`.
    pub fn page_bytes(&self) -> (u64, u64) {
        let geometry = self.geometry();
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let page_groups = self.page_tokens / group;
        let half = std::mem::size_of::<half::f16>();
        let codes = page_groups * self.bits.code_bytes(group * self.head_dimension)
            + self.page_tokens * self.bits.code_bytes(self.head_dimension);
        let metadata = 2 * half * (page_groups * self.head_dimension)
            + 2 * half * (self.page_tokens * geometry.dim.div_ceil(group));
        let per_layer = |bytes: usize| (bytes * self.kv_heads * self.layers) as u64;
        (per_layer(codes), per_layer(metadata))
    }

    /// The arrays the pool actually holds, measured.
    pub fn storage(&self) -> PagedPoolStorage {
        let (mut code_bytes, mut metadata_bytes) = (0_u64, 0_u64);
        for layer in &self.arrays {
            code_bytes += array_bytes(&layer.key_codes) + array_bytes(&layer.value_codes);
            metadata_bytes += [
                &layer.key_scales,
                &layer.key_zeros,
                &layer.value_scales,
                &layer.value_zeros,
            ]
            .into_iter()
            .map(array_bytes)
            .sum::<u64>();
        }
        PagedPoolStorage {
            page_tokens: self.page_tokens as u64,
            capacity_pages: self.capacity_pages as u64,
            live_pages: self.live_pages() as u64,
            peak_live_pages: self.peak_live_pages() as u64,
            code_bytes,
            metadata_bytes,
        }
    }

    /// Pages one growth (or a trim's rounding) adds: `MIN_POOL_PAGES`, or 1/32 of the
    /// current capacity once that is larger — a bounded over-allocation (≤ ~3 %), unlike doubling,
    /// which can hold twice the live pages.
    pub fn growth_chunk(&self) -> usize {
        MIN_POOL_PAGES.max(self.capacity_pages / 32)
    }

    /// Grow every layer's arrays so at least `pages` pages fit, by at least one
    /// [`Self::growth_chunk`]. Existing pages keep their contents.
    fn ensure_capacity(&mut self, pages: usize) -> Result<()> {
        if pages <= self.capacity_pages {
            return Ok(());
        }
        let capacity = pages.max(self.capacity_pages + self.growth_chunk());
        self.grow_to(capacity)
    }

    /// Grow to exactly `pages` pages when the pool holds fewer (sc-20681): a caller that knows
    /// its demand (a sequence about to prefill a known prompt) reserves it in one growth instead
    /// of letting the pool grow chunk by chunk under the prefill.
    pub fn reserve(&mut self, pages: usize) -> Result<()> {
        if pages <= self.capacity_pages {
            return Ok(());
        }
        self.grow_to(pages)
    }

    /// Shrink every layer's arrays to the highest live page plus one, rounded up to the growth
    /// chunk (to nothing when no page is live), returning the freed capacity's device memory.
    /// Pages are allocated lowest id first, so retired sequences' pages at the top come back.
    /// Every kept page keeps its contents; the arrays are copied, so the old buffers are freed.
    /// Call it between model steps (no sequence may have a step open on the pool).
    pub fn trim(&mut self) -> Result<()> {
        let target = match self.alloc.highest_live() {
            None => 0,
            Some(highest) => (highest + 1)
                .next_multiple_of(self.growth_chunk())
                .min(self.capacity_pages),
        };
        if target >= self.capacity_pages {
            return Ok(());
        }
        if target == 0 {
            self.arrays.clear();
            self.capacity_pages = 0;
            return Ok(());
        }
        let rows = mlx_i32(target, "pool capacity")?;
        // An elementwise copy owns a fresh buffer (a slice would keep the old one alive).
        let owned = |array: &Array| -> Result<Array> {
            let kept = array.try_index((0..rows, .., .., ..))?;
            let one = Array::from_int(1).as_dtype(array.dtype())?;
            Ok(mlx_rs::ops::multiply(&kept, &one)?)
        };
        let mut trimmed = Vec::with_capacity(self.arrays.len());
        for layer in &self.arrays {
            trimmed.push(PageArrays {
                key_codes: owned(&layer.key_codes)?,
                key_scales: owned(&layer.key_scales)?,
                key_zeros: owned(&layer.key_zeros)?,
                value_codes: owned(&layer.value_codes)?,
                value_scales: owned(&layer.value_scales)?,
                value_zeros: owned(&layer.value_zeros)?,
            });
        }
        mlx_rs::transforms::eval(trimmed.iter().flat_map(PageArrays::all))?;
        self.arrays = trimmed;
        self.capacity_pages = target;
        Ok(())
    }

    /// Page references the pool's sequences and prefix-store entries hold (summed refcounts).
    pub fn page_references(&self) -> usize {
        self.alloc.references()
    }

    /// Compact the pool (sc-20681): move the live pages above the live count into the lowest
    /// free ids, then shrink to the live count rounded up to the growth chunk — so a sequence that
    /// retires below the top returns its capacity too, not only one at the top. One gather per
    /// array builds the compacted arrays (owned, so the old buffers are freed). Returns the moves
    /// (old id → new id); the caller must apply them to every sequence and entry referencing a
    /// moved page ([`PagedPackedKvCache::remap_pages`]) before its next step. Call it between
    /// model steps, only when every holder of the pool's pages is in hand.
    pub fn compact(&mut self) -> Result<HashMap<usize, usize>> {
        let live = self.live_pages();
        let mut holes = (0..live).filter(|&id| !self.alloc.is_live(id));
        let movers = (live..self.capacity_pages)
            .filter(|&id| self.alloc.is_live(id))
            .collect::<Vec<_>>();
        let moves = movers
            .into_iter()
            .map(|from| {
                let to = holes
                    .next()
                    .ok_or_else(|| Error::Msg("page pool compaction ran out of holes".into()))?;
                Ok((from, to))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        if moves.is_empty() {
            self.trim()?;
            return Ok(moves);
        }
        let target = live
            .next_multiple_of(self.growth_chunk())
            .min(self.capacity_pages);
        let mut source = (0..target).collect::<Vec<_>>();
        for (&from, &to) in &moves {
            source[to] = from;
        }
        let index = source
            .iter()
            .map(|&id| mlx_i32(id, "page id"))
            .collect::<Result<Vec<_>>>()?;
        let index = Array::from_slice(&index, &[mlx_i32(target, "pool capacity")?]);
        let mut compacted = Vec::with_capacity(self.arrays.len());
        for layer in &self.arrays {
            let gather = |array: &Array| -> Result<Array> { Ok(array.take_axis(&index, 0)?) };
            compacted.push(PageArrays {
                key_codes: gather(&layer.key_codes)?,
                key_scales: gather(&layer.key_scales)?,
                key_zeros: gather(&layer.key_zeros)?,
                value_codes: gather(&layer.value_codes)?,
                value_scales: gather(&layer.value_scales)?,
                value_zeros: gather(&layer.value_zeros)?,
            });
        }
        mlx_rs::transforms::eval(compacted.iter().flat_map(PageArrays::all))?;
        self.arrays = compacted;
        self.capacity_pages = target;
        for (&from, &to) in &moves {
            self.alloc.relocate(from, to);
        }
        Ok(moves)
    }

    /// Resize every layer's arrays to exactly `capacity` (> the current capacity) pages.
    fn grow_to(&mut self, capacity: usize) -> Result<()> {
        let capacity = capacity.max(MIN_POOL_PAGES);
        let extra = capacity - self.capacity_pages;
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let (h, d, pt) = (self.kv_heads, self.head_dimension, self.page_tokens);
        let shapes = [
            ([pt / group, self.bits.code_bytes(group * d)], Dtype::Uint8),
            ([pt / group, d], Dtype::Float16),
            ([pt / group, d], Dtype::Float16),
            ([pt, self.bits.code_bytes(d)], Dtype::Uint8),
            ([pt, d.div_ceil(group)], Dtype::Float16),
            ([pt, d.div_ceil(group)], Dtype::Float16),
        ];
        let fresh = |rows: usize, [inner, width]: [usize; 2], dtype| -> Result<Array> {
            let shape = [
                mlx_i32(rows, "pool capacity")?,
                mlx_i32(h, "KV heads")?,
                mlx_i32(inner, "page rows")?,
                mlx_i32(width, "page width")?,
            ];
            Ok(zeros_dtype(&shape, dtype)?)
        };
        mlx_i32(capacity * h * pt, "pool rows")?;
        if self.arrays.is_empty() {
            for _ in 0..self.layers {
                let [kc, ks, kz, vc, vs, vz] = shapes;
                self.arrays.push(PageArrays {
                    key_codes: fresh(capacity, kc.0, kc.1)?,
                    key_scales: fresh(capacity, ks.0, ks.1)?,
                    key_zeros: fresh(capacity, kz.0, kz.1)?,
                    value_codes: fresh(capacity, vc.0, vc.1)?,
                    value_scales: fresh(capacity, vs.0, vs.1)?,
                    value_zeros: fresh(capacity, vz.0, vz.1)?,
                });
            }
        } else {
            for layer in &mut self.arrays {
                let grown = |array: &Array, (inner, dtype): ([usize; 2], Dtype)| -> Result<Array> {
                    Ok(concatenate_axis(&[array, &fresh(extra, inner, dtype)?], 0)?)
                };
                let [kc, ks, kz, vc, vs, vz] = shapes;
                *layer = PageArrays {
                    key_codes: grown(&layer.key_codes, kc)?,
                    key_scales: grown(&layer.key_scales, ks)?,
                    key_zeros: grown(&layer.key_zeros, kz)?,
                    value_codes: grown(&layer.value_codes, vc)?,
                    value_scales: grown(&layer.value_scales, vs)?,
                    value_zeros: grown(&layer.value_zeros, vz)?,
                };
            }
        }
        self.capacity_pages = capacity;
        Ok(())
    }

    /// Allocate a page (refcount 1), reusing the lowest freed id when one exists.
    fn alloc_page(&mut self) -> Result<usize> {
        let id = self.alloc.alloc_lowest();
        if let Err(error) = self.ensure_capacity(id + 1) {
            self.alloc.release(id);
            return Err(error);
        }
        Ok(id)
    }

    fn release(&mut self, id: usize) {
        self.alloc.release(id);
    }

    fn retain(&mut self, id: usize) {
        self.alloc.retain(id);
    }

    /// Copy page `from` into page `to` in every layer (the copy of copy-on-write).
    fn copy_page(&mut self, from: usize, to: usize) -> Result<()> {
        let (from, to) = (mlx_i32(from, "page id")?, mlx_i32(to, "page id")?);
        for layer in &mut self.arrays {
            for array in [
                &mut layer.key_codes,
                &mut layer.key_scales,
                &mut layer.key_zeros,
                &mut layer.value_codes,
                &mut layer.value_scales,
                &mut layer.value_zeros,
            ] {
                let page = array.try_index((from..from + 1, .., .., ..))?;
                array.try_index_mut((to..to + 1, .., .., ..), &page)?;
            }
        }
        Ok(())
    }

    /// `layer`'s component arrays over `pages` (in position order), each `[n, Hkv, rows, width]`
    /// page stack as one sequence `[1, Hkv, n · rows, width]`, in [`PageArrays::all`] order.
    fn sequence_arrays(&self, layer: usize, pages: &[usize]) -> Result<[Array; 6]> {
        let ids = pages
            .iter()
            .map(|&id| mlx_i32(id, "page id"))
            .collect::<Result<Vec<_>>>()?;
        let index = Array::from_slice(&ids, &[mlx_i32(ids.len(), "page count")?]);
        let heads = mlx_i32(self.kv_heads, "KV heads")?;
        let sequence = |array: &Array| -> Result<Array> {
            let shape = array.shape();
            let gathered = array.take_axis(&index, 0)?.transpose_axes(&[1, 0, 2, 3])?;
            Ok(gathered.reshape(&[1, heads, -1, shape[3]])?)
        };
        let [kc, ks, kz, vc, vs, vz] = self.arrays[layer].all();
        Ok([
            sequence(kc)?,
            sequence(ks)?,
            sequence(kz)?,
            sequence(vc)?,
            sequence(vs)?,
            sequence(vz)?,
        ])
    }

    /// Write `key` (`n` groups: codes, scales, zeros `[1, Hkv, n, ·]`) and `value` (`n · 32` rows)
    /// into `layer`'s page `page`, starting at the page's local group `group`.
    fn write(
        &mut self,
        layer: usize,
        page: usize,
        group: usize,
        key: [&Array; 3],
        value: [&Array; 3],
    ) -> Result<()> {
        let arrays = &mut self.arrays[layer];
        let page = mlx_i32(page, "page id")?;
        let group0 = mlx_i32(group, "page group")?;
        let groups = key[0].shape()[2];
        let row0 = group0 * PACKED_METAL_QUANT_GROUP_SIZE as i32;
        let rows = value[0].shape()[2];
        let pages = page..page + 1;
        arrays
            .key_codes
            .try_index_mut((pages.clone(), .., group0..group0 + groups, ..), key[0])?;
        arrays
            .key_scales
            .try_index_mut((pages.clone(), .., group0..group0 + groups, ..), key[1])?;
        arrays
            .key_zeros
            .try_index_mut((pages.clone(), .., group0..group0 + groups, ..), key[2])?;
        arrays
            .value_codes
            .try_index_mut((pages.clone(), .., row0..row0 + rows, ..), value[0])?;
        arrays
            .value_scales
            .try_index_mut((pages.clone(), .., row0..row0 + rows, ..), value[1])?;
        arrays
            .value_zeros
            .try_index_mut((pages, .., row0..row0 + rows, ..), value[2])?;
        Ok(())
    }

    /// `layer`'s first `packed` quantized tokens over `pages` (in position order), dequantized to
    /// Float32 `[1, Hkv, packed, D]` keys and values — the dense gather fallback's read.
    fn gather_dense(&self, layer: usize, pages: &[usize], packed: usize) -> Result<(Array, Array)> {
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let geometry = self.geometry();
        let [kc, ks, kz, vc, vs, vz] = self.sequence_arrays(layer, pages)?;
        let groups = packed / group;
        let keys = dequantize_key_groups(
            geometry,
            &rows_range(&kc, 0, groups)?,
            &rows_range(&ks, 0, groups)?,
            &rows_range(&kz, 0, groups)?,
        )?;
        let values = dequantize_value_rows(
            geometry,
            &rows_range(&vc, 0, packed)?,
            &rows_range(&vs, 0, packed)?,
            &rows_range(&vz, 0, packed)?,
        )?;
        Ok((keys, values))
    }
}

/// Why one attention call of a [`PagedPackedKvCache`] took the dense gather fallback. Ids are
/// stable receipt labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PagedFallbackReason {
    /// An explicit additive mask: the fused reader applies only causal/sliding/no masks.
    AdditiveMask,
    /// A K/V-sharing layer must publish dense K/V for the layers that read it.
    SharedKv,
    /// An attention scale other than `1/√D`, which the fused reader hard-codes.
    AttentionScale,
    /// Query/K/V rank, batch, heads, head dimension, dtype or sliding window outside the reader.
    Geometry,
    /// The query does not cover exactly the newly appended K/V step.
    QueryShape,
    /// No fused paged reader is bound (absent, refused, or disabled after a cold-dispatch fault).
    ReaderUnavailable,
    /// The caller asked for dense K/V directly (`update` without a fused attempt, or an explicit
    /// [`KvCache::prepare_dense_fallback`] such as an attention-score soft-cap).
    DenseCaller,
}

impl PagedFallbackReason {
    pub const ALL: [Self; 7] = [
        Self::AdditiveMask,
        Self::SharedKv,
        Self::AttentionScale,
        Self::Geometry,
        Self::QueryShape,
        Self::ReaderUnavailable,
        Self::DenseCaller,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::AdditiveMask => "additive_mask",
            Self::SharedKv => "shared_kv",
            Self::AttentionScale => "attention_scale",
            Self::Geometry => "geometry",
            Self::QueryShape => "query_shape",
            Self::ReaderUnavailable => "reader_unavailable",
            Self::DenseCaller => "dense_caller",
        }
    }
}

/// One layer of a sequence: its quantized extent (in pages) and its dense residual.
#[derive(Clone, Debug, Default)]
struct LayerState {
    /// Leading tokens quantized into pages (a multiple of the group).
    packed: usize,
    /// Rows of the residual holding the incomplete group.
    tail_rows: usize,
    /// `[1, Hkv, 32, D]` residuals, created by the layer's first append.
    key_tail: Option<Array>,
    value_tail: Option<Array>,
    /// K/V dtypes, fixed by the first append.
    dtypes: Option<(Dtype, Dtype)>,
    /// The armed speculation window (see [`KvCache::begin_speculation`]).
    window: Option<SpeculationWindow>,
}

/// One layer's unquantized rows from the group-aligned position `start` to the logical end, kept
/// while speculation is armed so a rollback into them restores the residual exactly.
#[derive(Clone, Debug)]
struct SpeculationWindow {
    start: usize,
    rows: usize,
    /// `[1, Hkv, rows, D]` keys and values; `None` while `rows == 0`.
    kv: Option<(Array, Array)>,
}

impl SpeculationWindow {
    fn extend(&mut self, keys: &Array, values: &Array) -> Result<()> {
        self.kv = Some(match self.kv.take() {
            None => (keys.clone(), values.clone()),
            Some((k, v)) => (
                concatenate_axis(&[&k, keys], 2)?,
                concatenate_axis(&[&v, values], 2)?,
            ),
        });
        self.rows += keys.shape()[2] as usize;
        Ok(())
    }
}

/// Calls accepted in committed steps; restored by a whole-step rollback.
#[derive(Clone, Debug, Default)]
struct AcceptedCalls {
    fused: u64,
    /// Of `fused`, the calls served by a batched dispatch (`paged_attention_batch`).
    batched: u64,
    kernel_paths: Vec<(PackedKernelSelection, u64)>,
    gathers: BTreeMap<PagedFallbackReason, (u64, String)>,
}

#[derive(Clone, Debug)]
struct PendingStep {
    original_len: usize,
    next_layer: usize,
    step: usize,
    marks: Vec<LayerState>,
    pages_before: usize,
    accepted: AcceptedCalls,
    /// A layer attended its fresh K/V through dense SDPA, so no output reads its pages: the commit
    /// evaluates the store, or the step's dense K/V would stay alive behind the lazy writes.
    store_unread: bool,
    /// Copy-on-write swaps of this step: `(column, shared page, private copy)`. The commit drops
    /// the shared references; a rollback returns the copies and restores the shared pages.
    cow: Vec<(usize, usize, usize)>,
}

/// A single sequence's quantized paged KV cache over a shared [`PackedPagePool`], read in place by
/// the fused paged reader (see the module docs).
#[derive(Debug)]
pub struct PagedPackedKvCache {
    pool: Rc<RefCell<PackedPagePool>>,
    handle: Option<CompiledKernelHandle>,
    /// Why no reader is bound (set when a cold dispatch faults).
    reader_refusal: Option<String>,
    /// Physical page of each `page_tokens`-token span of the sequence, in position order.
    page_ids: Vec<usize>,
    /// `[1, columns]` Int32 device copy of `page_ids` (padded), updated in place.
    page_table: Option<Array>,
    logical_len: usize,
    layers: Vec<LayerState>,
    pending: Option<PendingStep>,
    /// Reason a declined fused attempt hands to the caller's following `update`.
    pending_gather: Option<(PagedFallbackReason, String)>,
    accepted: AcceptedCalls,
    dispatch_attempts: u64,
    failed_dispatches: u64,
    /// Logical length at which speculation was armed; consumed by the next truncate.
    speculation_base: Option<usize>,
}

impl PagedPackedKvCache {
    /// A fresh sequence on `pool`, read by `reader` (which must read the pool's code width).
    pub fn with_pool(
        pool: Rc<RefCell<PackedPagePool>>,
        reader: CompiledKernelHandle,
    ) -> Result<Self> {
        let (bits, layers) = {
            let pool = pool.borrow();
            (pool.bits, pool.layers)
        };
        if reader.code_bits() != bits {
            return Err(Error::Unsupported(format!(
                "a {}-bit reader cannot read {}-bit pages",
                reader.code_bits().bits(),
                bits.bits()
            )));
        }
        Ok(Self {
            pool,
            handle: Some(reader),
            reader_refusal: None,
            page_ids: Vec::new(),
            page_table: None,
            logical_len: 0,
            layers: vec![LayerState::default(); layers],
            pending: None,
            pending_gather: None,
            accepted: AcceptedCalls::default(),
            dispatch_attempts: 0,
            failed_dispatches: 0,
            speculation_base: None,
        })
    }

    pub fn pool(&self) -> &Rc<RefCell<PackedPagePool>> {
        &self.pool
    }

    /// The sequence's page table: physical page ids in position order.
    pub fn page_ids(&self) -> &[usize] {
        &self.page_ids
    }

    /// Attention calls the fused paged reader served.
    pub fn fused_calls(&self) -> u64 {
        self.accepted.fused
    }

    /// Of [`Self::fused_calls`], those served together with other sequences in one batched
    /// dispatch (`paged_attention_batch`).
    pub fn batched_calls(&self) -> u64 {
        self.accepted.batched
    }

    /// Dense gather fallbacks, per reason.
    pub fn dense_gathers(&self) -> Vec<(PagedFallbackReason, u64)> {
        self.accepted
            .gathers
            .iter()
            .map(|(reason, (calls, _))| (*reason, *calls))
            .collect()
    }

    /// Token positions reserved by the sequence's pages beyond its quantized tokens (the
    /// partially filled last page), plus the residual's unused rows: paging's slack.
    pub fn slack_tokens(&self) -> usize {
        let page_tokens = self.pool.borrow().page_tokens;
        let state = &self.layers[0];
        self.page_ids.len() * page_tokens - state.packed
            + state
                .key_tail
                .as_ref()
                .map_or(0, |_| PACKED_METAL_QUANT_GROUP_SIZE - state.tail_rows)
    }

    /// Apply a pool compaction's moves ([`PackedPagePool::compact`]) to this sequence's page
    /// table. Refused inside an open step.
    pub fn remap_pages(&mut self, moves: &HashMap<usize, usize>) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::Config(
                "paged packed pages remapped inside an open step".into(),
            ));
        }
        let mut moved = false;
        for id in &mut self.page_ids {
            if let Some(&to) = moves.get(id) {
                *id = to;
                moved = true;
            }
        }
        if moved {
            // Same width as before (the table grows by doubling): only the moved ids change.
            let columns = self
                .page_table
                .as_ref()
                .map_or(0, |table| table.shape()[1] as usize)
                .max(self.page_ids.len())
                .max(MIN_TABLE_COLUMNS);
            let mut ids = self
                .page_ids
                .iter()
                .map(|&id| mlx_i32(id, "page id"))
                .collect::<Result<Vec<_>>>()?;
            ids.resize(columns, 0);
            self.page_table = Some(Array::from_slice(
                &ids,
                &[1, mlx_i32(columns, "page table")?],
            ));
        }
        Ok(())
    }

    /// Whether a model step is open on the sequence.
    pub fn has_open_step(&self) -> bool {
        self.pending.is_some()
    }

    /// Rebuild the device page table from `page_ids` (a fork, a restore, or a rolled-back
    /// copy-on-write swap leaves it unset).
    fn rebuild_table(&mut self) -> Result<()> {
        let columns = self.page_ids.len().max(MIN_TABLE_COLUMNS);
        let mut ids = self
            .page_ids
            .iter()
            .map(|&id| mlx_i32(id, "page id"))
            .collect::<Result<Vec<_>>>()?;
        ids.resize(columns, 0);
        self.page_table = Some(Array::from_slice(
            &ids,
            &[1, mlx_i32(columns, "page table")?],
        ));
        Ok(())
    }

    /// Write page `self.page_ids[column]` into the device page table, growing it by doubling.
    fn sync_table(&mut self, column: usize) -> Result<()> {
        let columns = self
            .page_table
            .as_ref()
            .map_or(0, |table| table.shape()[1] as usize);
        if column >= columns {
            let columns = (column + 1).max(columns * 2).max(MIN_TABLE_COLUMNS);
            let mut ids = self
                .page_ids
                .iter()
                .map(|&id| mlx_i32(id, "page id"))
                .collect::<Result<Vec<_>>>()?;
            ids.resize(columns, 0);
            self.page_table = Some(Array::from_slice(
                &ids,
                &[1, mlx_i32(columns, "page table")?],
            ));
            return Ok(());
        }
        let id = mlx_i32(self.page_ids[column], "page id")?;
        let column = mlx_i32(column, "page table column")?;
        if let Some(table) = self.page_table.as_mut() {
            table.try_index_mut(
                (0..1, column..column + 1),
                Array::from_slice(&[id], &[1, 1]),
            )?;
        }
        Ok(())
    }

    /// Allocate pages until the sequence's pages cover `tokens` quantized tokens.
    fn cover(&mut self, tokens: usize, page_tokens: usize) -> Result<()> {
        while self.page_ids.len() * page_tokens < tokens {
            let id = self.pool.borrow_mut().alloc_page()?;
            self.page_ids.push(id);
            self.sync_table(self.page_ids.len() - 1)?;
        }
        Ok(())
    }

    /// Return every page past the first `keep` to the pool.
    fn release_pages_from(&mut self, keep: usize) {
        if keep >= self.page_ids.len() {
            return;
        }
        let mut pool = self.pool.borrow_mut();
        for &id in &self.page_ids[keep..] {
            pool.release(id);
        }
        drop(pool);
        self.page_ids.truncate(keep);
    }

    /// Quantize-on-append for one layer: every completed group of `residual ++ fresh` is quantized
    /// on the GPU and written straight into the sequence's page slots (allocating pages as the
    /// quantized extent crosses a page boundary); the incomplete remainder stays in the residual.
    fn append(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<()> {
        let geometry = self.pool.borrow().geometry();
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let step = keys.shape()[2] as usize;
        let tail_shape = [
            1,
            mlx_i32(geometry.heads, "KV heads")?,
            mlx_i32(group, "group")?,
            mlx_i32(geometry.dim, "head dimension")?,
        ];
        let state = &mut self.layers[layer];
        if state
            .dtypes
            .is_some_and(|(key, value)| key != keys.dtype() || value != values.dtype())
        {
            return Err(Error::Config(
                "paged packed K/V dtype must remain stable for each layer".into(),
            ));
        }
        if let Some(window) = state.window.as_mut() {
            window.extend(keys, values)?;
        }
        let residual = state.tail_rows;
        let total = residual + step;
        let groups = total / group;
        let remainder = total % group;
        let packed = state.packed;
        let mut key_tail = match state.key_tail.take() {
            Some(tail) => tail,
            None => zeros_dtype(&tail_shape, keys.dtype())?,
        };
        let mut value_tail = match state.value_tail.take() {
            Some(tail) => tail,
            None => zeros_dtype(&tail_shape, values.dtype())?,
        };
        state.dtypes = Some((keys.dtype(), values.dtype()));
        if groups == 0 {
            write_rows(&mut key_tail, residual, keys)?;
            write_rows(&mut value_tail, residual, values)?;
            let state = &mut self.layers[layer];
            state.key_tail = Some(key_tail);
            state.value_tail = Some(value_tail);
            state.tail_rows = total;
            return Ok(());
        }
        let [key_codes, key_scales, key_zeros, value_codes, value_scales, value_zeros] =
            quantize_group_affine_flush(
                &key_tail,
                &value_tail,
                residual,
                keys,
                values,
                groups,
                geometry.bits,
            )?;
        let new_packed = packed + groups * group;
        self.write_groups(
            layer,
            packed / group,
            groups,
            [&key_codes, &key_scales, &key_zeros],
            [&value_codes, &value_scales, &value_zeros],
        )?;
        let (key_tail, value_tail) = if remainder > 0 {
            // The flush consumed the whole residual, so the remainder is the fresh suffix.
            let from = step - remainder;
            (
                padded_residual(geometry, &rows_range(keys, from, step)?, remainder)?,
                padded_residual(geometry, &rows_range(values, from, step)?, remainder)?,
            )
        } else {
            (key_tail, value_tail)
        };
        let state = &mut self.layers[layer];
        state.packed = new_packed;
        state.tail_rows = remainder;
        state.key_tail = Some(key_tail);
        state.value_tail = Some(value_tail);
        Ok(())
    }

    /// Write `groups` quantized groups (K: codes, scales, zeros `[1, Hkv, groups, ·]`; V: the
    /// `groups · 32` rows) into `layer`'s pages from sequence group `first`, allocating pages as
    /// the extent crosses a page boundary and copying a shared page before writing into it.
    fn write_groups(
        &mut self,
        layer: usize,
        first: usize,
        groups: usize,
        key: [&Array; 3],
        value: [&Array; 3],
    ) -> Result<()> {
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let page_tokens = self.pool.borrow().page_tokens;
        self.cover((first + groups) * group, page_tokens)?;
        let page_groups = page_tokens / group;
        let mut done = 0;
        while done < groups {
            let global = first + done;
            let page = self.private_page(global / page_groups)?;
            let local = global % page_groups;
            let count = (groups - done).min(page_groups - local);
            let slice = |array: &Array, unit: usize| -> Result<Array> {
                if count == groups {
                    Ok(array.clone())
                } else {
                    rows_range(array, done * unit, (done + count) * unit)
                }
            };
            self.pool.borrow_mut().write(
                layer,
                page,
                local,
                [&slice(key[0], 1)?, &slice(key[1], 1)?, &slice(key[2], 1)?],
                [
                    &slice(value[0], group)?,
                    &slice(value[1], group)?,
                    &slice(value[2], group)?,
                ],
            )?;
            done += count;
        }
        Ok(())
    }

    /// The page at `column`, made private first when another sequence (or a prefix-store entry)
    /// also references it: copy-on-write. Inside a step the swap is provisional (see
    /// [`PendingStep::cow`]); outside one the shared reference is dropped at once.
    fn private_page(&mut self, column: usize) -> Result<usize> {
        let shared = self.page_ids[column];
        if self.pool.borrow().refcount(shared) <= 1 {
            return Ok(shared);
        }
        let copy = self.pool.borrow_mut().alloc_page()?;
        if let Err(error) = self.pool.borrow_mut().copy_page(shared, copy) {
            self.pool.borrow_mut().release(copy);
            return Err(error);
        }
        self.page_ids[column] = copy;
        match self.pending.as_mut() {
            Some(pending) => pending.cow.push((column, shared, copy)),
            None => self.pool.borrow_mut().release(shared),
        }
        if let Err(error) = self.sync_table(column) {
            // The table is rebuilt from `page_ids` before its next read.
            self.page_table = None;
            return Err(error);
        }
        Ok(copy)
    }

    /// The dense gather fallback's read: `layer`'s whole sequence dequantized from its pages plus
    /// its residual rows, in the K/V dtypes. Transient: nothing dense is retained.
    fn gather(&self, layer: usize) -> Result<(Array, Array)> {
        let state = &self.layers[layer];
        let (key_dtype, value_dtype) = state
            .dtypes
            .ok_or_else(|| Error::Msg("gather on an empty paged packed layer".into()))?;
        let mut keys = Vec::with_capacity(2);
        let mut values = Vec::with_capacity(2);
        if state.packed > 0 {
            let pool = self.pool.borrow();
            let pages = state.packed.div_ceil(pool.page_tokens);
            let (k, v) = pool.gather_dense(layer, &self.page_ids[..pages], state.packed)?;
            keys.push(k.as_dtype(key_dtype)?);
            values.push(v.as_dtype(value_dtype)?);
        }
        if state.tail_rows > 0 {
            let (k, v) = (
                state.key_tail.as_ref().expect("residual present with rows"),
                state
                    .value_tail
                    .as_ref()
                    .expect("residual present with rows"),
            );
            keys.push(rows_range(k, 0, state.tail_rows)?);
            values.push(rows_range(v, 0, state.tail_rows)?);
        }
        let join = |parts: Vec<Array>| -> Result<Array> {
            match parts.as_slice() {
                [] => Err(Error::Msg("gather on an empty paged packed layer".into())),
                [one] => Ok(one.clone()),
                many => Ok(concatenate_axis(&many.iter().collect::<Vec<_>>(), 2)?),
            }
        };
        Ok((join(keys)?, join(values)?))
    }

    /// The fused paged reader over `layer`'s pages and residual. A cold dispatch (the handle never
    /// produced an output) is evaluated so a compile fault surfaces inside the step's transaction.
    fn dispatch(&mut self, layer: usize, query: &Array, mask: PackedMask) -> Result<Array> {
        let handle = self
            .handle
            .clone()
            .ok_or_else(|| Error::Unsupported("no paged packed reader is bound".into()))?;
        self.pool.borrow_mut().ensure_capacity(1)?;
        if self.page_table.is_none() {
            self.rebuild_table()?;
        }
        let pool = self.pool.clone();
        let pool = pool.borrow();
        let state = &self.layers[layer];
        let (Some(key_tail), Some(value_tail), Some(page_table)) = (
            state.key_tail.as_ref(),
            state.value_tail.as_ref(),
            self.page_table.as_ref(),
        ) else {
            return Err(Error::Msg(
                "paged dispatch before the layer's append".into(),
            ));
        };
        let arrays = &pool.arrays[layer];
        let sequences = [PagedSequenceExtent {
            packed_tokens: state.packed,
            kv_tokens: state.packed + state.tail_rows,
        }];
        let args = PagedPackedAttentionArgs {
            query,
            key_codes: &arrays.key_codes,
            key_scales: &arrays.key_scales,
            key_zeros: &arrays.key_zeros,
            key_tail,
            value_codes: &arrays.value_codes,
            value_scales: &arrays.value_scales,
            value_zeros: &arrays.value_zeros,
            value_tail,
            page_table,
            page_tokens: pool.page_tokens,
            sequences: &sequences,
            code_bits: pool.bits,
            mask,
        };
        let selection = handle.paged_kernel_selection(&args);
        let warmed = handle.is_paged_warmed();
        self.dispatch_attempts += 1;
        let output = handle.dispatch_paged(&args).and_then(|output| {
            if !warmed {
                output.eval()?;
            }
            Ok(output)
        });
        drop(pool);
        let output = match output {
            Ok(output) => output,
            Err(error) => {
                self.failed_dispatches += 1;
                return Err(error);
            }
        };
        handle.mark_paged_warmed();
        self.accepted.fused += 1;
        if let Some(selection) = selection {
            match self
                .accepted
                .kernel_paths
                .iter_mut()
                .find(|(recorded, _)| *recorded == selection)
            {
                Some((_, calls)) => *calls += 1,
                None => self.accepted.kernel_paths.push((selection, 1)),
            }
        }
        Ok(output)
    }

    /// Open (at layer 0) or continue the model step's transaction.
    fn begin_layer(&mut self, layer: usize, step: usize) -> Result<()> {
        match &self.pending {
            None if layer == 0 => {
                self.pending = Some(PendingStep {
                    original_len: self.logical_len,
                    next_layer: 0,
                    step,
                    marks: self.layers.clone(),
                    pages_before: self.page_ids.len(),
                    accepted: self.accepted.clone(),
                    store_unread: false,
                    cow: Vec::new(),
                });
                Ok(())
            }
            Some(pending) if pending.next_layer == layer && pending.step == step => Ok(()),
            _ => Err(Error::Config(
                "paged packed step layer order or token length mismatch".into(),
            )),
        }
    }

    /// Close one layer of the step; the last layer commits the step's length.
    fn finish_layer(&mut self) -> Result<()> {
        let pending = self.pending.as_mut().expect("a step is open");
        pending.next_layer += 1;
        if pending.next_layer < self.layers.len() {
            return Ok(());
        }
        let pending = self.pending.take().expect("a step is open");
        if pending.store_unread {
            let pool = self.pool.borrow();
            let mut arrays = pool
                .arrays
                .iter()
                .flat_map(PageArrays::all)
                .collect::<Vec<_>>();
            arrays.extend(
                self.layers
                    .iter()
                    .filter_map(|state| state.key_tail.as_ref()),
            );
            arrays.extend(
                self.layers
                    .iter()
                    .filter_map(|state| state.value_tail.as_ref()),
            );
            if let Some(table) = self.page_table.as_ref() {
                arrays.push(table);
            }
            if let Err(error) = mlx_rs::transforms::eval(arrays) {
                drop(pool);
                self.pending = Some(pending);
                return Err(error.into());
            }
        }
        self.logical_len = pending.original_len + pending.step;
        let mut pool = self.pool.borrow_mut();
        for &(_, shared, _) in &pending.cow {
            pool.release(shared);
        }
        Ok(())
    }

    /// Undo the open step: every layer's extents and residuals, its pages, its accepted calls.
    fn rollback(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.layers = pending.marks;
            self.release_pages_from(pending.pages_before);
            if !pending.cow.is_empty() {
                let mut pool = self.pool.borrow_mut();
                for &(column, shared, copy) in pending.cow.iter().rev() {
                    pool.release(copy);
                    self.page_ids[column] = shared;
                }
                drop(pool);
                // Rebuilt from `page_ids` before the next read.
                self.page_table = None;
            }
            self.accepted = pending.accepted;
        }
    }

    fn record_gather(&mut self, reason: PagedFallbackReason, detail: String) {
        let entry = self.accepted.gathers.entry(reason).or_default();
        entry.0 += 1;
        entry.1 = detail;
    }

    /// Decline a fused attempt before any mutation; the caller's `update` serves it by gather.
    fn decline(&mut self, reason: PagedFallbackReason, detail: impl Into<String>) -> Option<Array> {
        self.pending_gather = Some((reason, detail.into()));
        None
    }

    /// Validate one step's `[1, Hkv, step, D]` K/V against the pool geometry.
    fn check_kv(&self, layer: usize, keys: &Array, values: &Array) -> Option<&'static str> {
        let pool = self.pool.borrow();
        let float = |dtype| matches!(dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32);
        let (Ok(k), Ok(v)) = (
            packed_input_shape(keys.shape(), "key"),
            packed_input_shape(values.shape(), "value"),
        ) else {
            return Some("key and value tensors must have rank four");
        };
        if layer >= self.layers.len() {
            Some("layer index is out of range")
        } else if k != v
            || k[0] != 1
            || k[1] != pool.kv_heads
            || k[2] == 0
            || k[3] != pool.head_dimension
        {
            Some("paged K/V must be one sequence of the pool's KV heads and head dimension")
        } else if !float(keys.dtype()) || !float(values.dtype()) {
            Some("paged K/V must be f16, bf16 or f32")
        } else {
            None
        }
    }

    /// Re-stage `layer`'s quantized group containing token `keep` (exclusive end `len`) into its
    /// residual: the retained rows of a cut group, from their quantized values.
    fn restage(&mut self, layer: usize, keep: usize, len: usize) -> Result<()> {
        let rows = len - keep;
        let state = &self.layers[layer];
        let (key_dtype, value_dtype) = state.dtypes.expect("a layer with tokens has dtypes");
        let pool = self.pool.borrow();
        let geometry = pool.geometry();
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let page_groups = pool.page_tokens / group;
        let global = keep / group;
        let page = mlx_i32(self.page_ids[global / page_groups], "page id")?;
        let local = mlx_i32(global % page_groups, "page group")?;
        let row0 = local * group as i32;
        let arrays = &pool.arrays[layer];
        let pick = |array: &Array, start: i32, count: i32| -> Result<Array> {
            Ok(array.try_index((page..page + 1, .., start..start + count, ..))?)
        };
        let keys = dequantize_key_groups(
            geometry,
            &pick(&arrays.key_codes, local, 1)?,
            &pick(&arrays.key_scales, local, 1)?,
            &pick(&arrays.key_zeros, local, 1)?,
        )?;
        let rows_i32 = mlx_i32(rows, "re-staged rows")?;
        let values = dequantize_value_rows(
            geometry,
            &pick(&arrays.value_codes, row0, rows_i32)?,
            &pick(&arrays.value_scales, row0, rows_i32)?,
            &pick(&arrays.value_zeros, row0, rows_i32)?,
        )?;
        let key_tail = padded_residual(
            geometry,
            &rows_range(&keys, 0, rows)?.as_dtype(key_dtype)?,
            rows,
        )?;
        let value_tail = padded_residual(geometry, &values.as_dtype(value_dtype)?, rows)?;
        drop(pool);
        let state = &mut self.layers[layer];
        state.key_tail = Some(key_tail);
        state.value_tail = Some(value_tail);
        Ok(())
    }

    /// Close every layer's speculation window.
    fn close_windows(&mut self) {
        for state in &mut self.layers {
            state.window = None;
        }
    }

    /// Truncate to `len` inside the armed speculation window: every group completed below `len`
    /// keeps its pages (its codes came from exactly the window's rows), the residual is rebuilt
    /// from the window's unquantized rows, and the pages past the kept groups are released.
    fn truncate_into_window(&mut self, len: usize) -> Result<()> {
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let (geometry, page_tokens) = {
            let pool = self.pool.borrow();
            (pool.geometry(), pool.page_tokens)
        };
        let keep = len / group * group;
        let rows = len - keep;
        let mut next = Vec::with_capacity(self.layers.len());
        for state in &self.layers {
            let window = state
                .window
                .as_ref()
                .ok_or_else(|| Error::Msg("speculation window missing on a layer".into()))?;
            let (key_dtype, value_dtype) = state
                .dtypes
                .ok_or_else(|| Error::Msg("speculation window over an empty layer".into()))?;
            let tails = match window.kv.as_ref() {
                Some((keys, values)) if rows > 0 => {
                    let (from, to) = (keep - window.start, len - window.start);
                    (
                        padded_residual(geometry, &rows_range(keys, from, to)?, rows)?,
                        padded_residual(geometry, &rows_range(values, from, to)?, rows)?,
                    )
                }
                _ => {
                    let shape = geometry.shape(group, geometry.dim)?;
                    (
                        zeros_dtype(&shape, key_dtype)?,
                        zeros_dtype(&shape, value_dtype)?,
                    )
                }
            };
            next.push(tails);
        }
        for (state, (key_tail, value_tail)) in self.layers.iter_mut().zip(next) {
            state.packed = keep;
            state.tail_rows = rows;
            state.key_tail = Some(key_tail);
            state.value_tail = Some(value_tail);
            state.window = None;
        }
        self.release_pages_from(keep.div_ceil(page_tokens));
        self.logical_len = len;
        Ok(())
    }

    /// Whether this sequence lost its fused reader to a cold dispatch fault, so every later call
    /// is a dense gather. A prefix store never keeps such a sequence.
    pub(crate) fn reader_faulted(&self) -> bool {
        self.handle.is_none()
    }

    /// A new sequence on the same pool starting with this sequence's first `tokens` positions,
    /// sharing its pages (reference-counted, copy-on-write) instead of copying them. Returns the
    /// fork and the positions it holds: all `tokens` when they reach into this sequence's
    /// unquantized residual (whose rows the fork copies), else `tokens` rounded down to the
    /// quantization group, so the fork never re-quantizes a cut group from its codes — its state
    /// is exactly what appending the same K/V to an empty cache would have produced. The fork has
    /// its own evidence counters and no armed speculation.
    pub fn fork_prefix(&self, tokens: usize) -> Result<(Self, usize)> {
        if self.pending.is_some() {
            return Err(Error::Config(
                "paged packed fork inside an open step".into(),
            ));
        }
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let tokens = tokens.min(self.logical_len);
        let packed = self.layers.first().map_or(0, |state| state.packed);
        let (shared_packed, rows) = if tokens > packed {
            (packed, tokens - packed)
        } else {
            (tokens / group * group, 0)
        };
        let page_tokens = self.pool.borrow().page_tokens;
        let pages = self.page_ids[..shared_packed.div_ceil(page_tokens)].to_vec();
        {
            let mut pool = self.pool.borrow_mut();
            for &id in &pages {
                pool.retain(id);
            }
        }
        let layers = self
            .layers
            .iter()
            .map(|state| LayerState {
                packed: shared_packed,
                tail_rows: rows,
                key_tail: (rows > 0).then(|| state.key_tail.clone()).flatten(),
                value_tail: (rows > 0).then(|| state.value_tail.clone()).flatten(),
                dtypes: state.dtypes,
                window: None,
            })
            .collect();
        let fork = Self {
            pool: self.pool.clone(),
            handle: self.handle.clone(),
            reader_refusal: self.reader_refusal.clone(),
            page_ids: pages,
            page_table: None,
            logical_len: shared_packed + rows,
            layers,
            pending: None,
            pending_gather: None,
            accepted: AcceptedCalls::default(),
            dispatch_attempts: 0,
            failed_dispatches: 0,
            speculation_base: None,
        };
        Ok((fork, shared_packed + rows))
    }

    /// The sequence's identity under `model` (see [`PagedCacheIdentity`]).
    pub fn identity(&self, model: &PagedModelKey) -> PagedCacheIdentity {
        self.pool.borrow().identity(model)
    }

    /// A host copy of the sequence — its codes and scale/zero metadata in position order and its
    /// unquantized residual rows — under its identity for `model`. Restorable with
    /// [`Self::restore`] into a pool of the same identity, in this process or (through
    /// [`PagedCacheSnapshot::to_bytes`]) another.
    pub fn snapshot(&self, model: &PagedModelKey) -> Result<PagedCacheSnapshot> {
        if self.pending.is_some() {
            return Err(Error::Config(
                "paged packed snapshot inside an open step".into(),
            ));
        }
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let pool = self.pool.borrow();
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer, state) in self.layers.iter().enumerate() {
            let mut snapshot = LayerSnapshot {
                packed: state.packed,
                tail_rows: state.tail_rows,
                dtypes: state
                    .dtypes
                    .map(|(key, value)| (dtype_label(key), dtype_label(value))),
                ..LayerSnapshot::default()
            };
            if state.packed > 0 {
                let pages = &self.page_ids[..state.packed.div_ceil(pool.page_tokens)];
                let [kc, ks, kz, vc, vs, vz] = pool.sequence_arrays(layer, pages)?;
                let groups = state.packed / group;
                snapshot.key = [
                    host_bytes(&rows_range(&kc, 0, groups)?)?,
                    host_bytes(&rows_range(&ks, 0, groups)?)?,
                    host_bytes(&rows_range(&kz, 0, groups)?)?,
                ];
                snapshot.value = [
                    host_bytes(&rows_range(&vc, 0, state.packed)?)?,
                    host_bytes(&rows_range(&vs, 0, state.packed)?)?,
                    host_bytes(&rows_range(&vz, 0, state.packed)?)?,
                ];
            }
            if state.tail_rows > 0 {
                let (Some(keys), Some(values)) = (&state.key_tail, &state.value_tail) else {
                    return Err(Error::Msg("residual rows without a residual".into()));
                };
                snapshot.residual = [
                    host_bytes(&rows_range(keys, 0, state.tail_rows)?)?,
                    host_bytes(&rows_range(values, 0, state.tail_rows)?)?,
                ];
            }
            layers.push(snapshot);
        }
        Ok(PagedCacheSnapshot {
            identity: pool.identity(model),
            tokens: self.logical_len,
            tokens_sha256: None,
            layers,
        })
    }

    /// Rebuild a snapshot as a new sequence on `pool`, read by `reader`. Refused — nothing
    /// allocated — unless the snapshot's identity is exactly `pool`'s identity for `model`: a
    /// different model, KV format version, page layout version, code width or page geometry is
    /// never reused.
    pub fn restore(
        snapshot: &PagedCacheSnapshot,
        pool: Rc<RefCell<PackedPagePool>>,
        reader: CompiledKernelHandle,
        model: &PagedModelKey,
    ) -> Result<Self> {
        let expected = pool.borrow().identity(model);
        if let Some(mismatch) = snapshot.identity.mismatch(&expected) {
            return Err(Error::Unsupported(format!(
                "paged KV snapshot refused: {mismatch}"
            )));
        }
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let (geometry, kv_heads, width) = {
            let pool = pool.borrow();
            (pool.geometry(), pool.kv_heads, pool.head_dimension)
        };
        let bits = geometry.bits;
        let mut cache = Self::with_pool(pool, reader)?;
        // Extents are checked before any arithmetic on them: a residual within one group, whole
        // quantized groups, a non-overflowing total equal to the snapshot's positions, and dtypes
        // for every layer that holds a position.
        if snapshot.layers.len() != cache.layers.len()
            || snapshot.layers.iter().any(|layer| {
                layer.tail_rows >= group
                    || !layer.packed.is_multiple_of(group)
                    || layer.packed != snapshot.layers[0].packed
                    || layer.packed.checked_add(layer.tail_rows) != Some(snapshot.tokens)
                    || (snapshot.tokens > 0 && layer.dtypes.is_none())
            })
        {
            return Err(Error::Unsupported(
                "paged KV snapshot refused: inconsistent layer extents".into(),
            ));
        }
        let heads = mlx_i32(kv_heads, "KV heads")?;
        let array = |bytes: &[u8], rows: usize, width: usize, dtype: Dtype| -> Result<Array> {
            let element = if dtype == Dtype::Uint8 {
                1
            } else {
                dtype_bytes(dtype)
            };
            let shape = [
                1,
                heads,
                mlx_i32(rows, "snapshot rows")?,
                mlx_i32(width * element, "snapshot width")?,
            ];
            let expected = kv_heads * rows * width * element;
            if bytes.len() != expected {
                return Err(Error::Unsupported(format!(
                    "paged KV snapshot refused: a component holds {} bytes, not {expected}",
                    bytes.len()
                )));
            }
            let raw = Array::from_slice(bytes, &shape);
            Ok(if dtype == Dtype::Uint8 {
                raw
            } else {
                raw.view_dtype(dtype)?
            })
        };
        for (layer, snapshot) in snapshot.layers.iter().enumerate() {
            let dtypes = match &snapshot.dtypes {
                Some((key, value)) => Some((parse_dtype(key)?, parse_dtype(value)?)),
                None => None,
            };
            let groups = snapshot.packed / group;
            if groups > 0 {
                let key = [
                    array(
                        &snapshot.key[0],
                        groups,
                        bits.code_bytes(group * width),
                        Dtype::Uint8,
                    )?,
                    array(&snapshot.key[1], groups, width, Dtype::Float16)?,
                    array(&snapshot.key[2], groups, width, Dtype::Float16)?,
                ];
                let value = [
                    array(
                        &snapshot.value[0],
                        snapshot.packed,
                        bits.code_bytes(width),
                        Dtype::Uint8,
                    )?,
                    array(
                        &snapshot.value[1],
                        snapshot.packed,
                        width.div_ceil(group),
                        Dtype::Float16,
                    )?,
                    array(
                        &snapshot.value[2],
                        snapshot.packed,
                        width.div_ceil(group),
                        Dtype::Float16,
                    )?,
                ];
                cache.write_groups(
                    layer,
                    0,
                    groups,
                    [&key[0], &key[1], &key[2]],
                    [&value[0], &value[1], &value[2]],
                )?;
            }
            let state = &mut cache.layers[layer];
            state.packed = snapshot.packed;
            state.tail_rows = snapshot.tail_rows;
            state.dtypes = dtypes;
            if snapshot.tail_rows > 0 {
                let (key_dtype, value_dtype) = dtypes.ok_or_else(|| {
                    Error::Unsupported("paged KV snapshot refused: residual without dtype".into())
                })?;
                let keys = array(&snapshot.residual[0], snapshot.tail_rows, width, key_dtype)?;
                let values = array(
                    &snapshot.residual[1],
                    snapshot.tail_rows,
                    width,
                    value_dtype,
                )?;
                state.key_tail = Some(padded_residual(geometry, &keys, snapshot.tail_rows)?);
                state.value_tail = Some(padded_residual(geometry, &values, snapshot.tail_rows)?);
            }
        }
        cache.logical_len = snapshot.tokens;
        Ok(cache)
    }

    /// Model-boundary evidence (see [`KvCache::packed_evidence`]).
    pub fn evidence(&self) -> Result<PackedCacheEvidence> {
        let storage = self.compressed_storage()?.unwrap_or_default();
        let bits = self.pool.borrow().bits;
        Ok(PackedCacheEvidence {
            representation_identity: paged_packed_identity(bits).into(),
            representation_version: PAGED_PACKED_LAYOUT_VERSION,
            bits: bits.bits(),
            quantization_group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            accepted_direct_calls: usize::try_from(self.accepted.fused).unwrap_or(usize::MAX),
            kernel_paths: PackedKernelPathEvidence::sorted(&self.accepted.kernel_paths),
            dispatch_attempts: self.dispatch_attempts,
            failed_dispatches: self.failed_dispatches,
            kernel_warmed: self
                .handle
                .as_ref()
                .is_some_and(CompiledKernelHandle::is_paged_warmed),
            retained_device_code_bytes: storage.device_code_bytes,
            retained_device_metadata_bytes: storage.device_metadata_bytes,
            retained_device_packed_logical_bytes: storage.device_bytes(),
            dense_gather_calls: self
                .accepted
                .gathers
                .iter()
                .map(|(reason, (calls, _))| (reason.id().to_owned(), *calls))
                .collect(),
            ..PackedCacheEvidence::default()
        })
    }
}

impl KvCache for PagedPackedKvCache {
    fn try_packed_attention(
        &mut self,
        layer: usize,
        query: &Array,
        keys: &Array,
        values: &Array,
        mask: PackedAttentionMask,
        scale: f32,
        retained_for_sharing: bool,
    ) -> Result<Option<Array>> {
        use PagedFallbackReason as Reason;
        if self.handle.is_none() {
            let detail = self
                .reader_refusal
                .clone()
                .unwrap_or_else(|| "no fused paged reader is bound".into());
            return Ok(self.decline(Reason::ReaderUnavailable, detail));
        }
        if retained_for_sharing {
            return Ok(self.decline(
                Reason::SharedKv,
                "a K/V-sharing layer must publish dense K/V",
            ));
        }
        if let Some(reason) = self.check_kv(layer, keys, values) {
            return Ok(self.decline(Reason::Geometry, reason));
        }
        let q = match packed_input_shape(query.shape(), "query") {
            Ok(q) => q,
            Err(_) => return Ok(self.decline(Reason::Geometry, "the query must have rank four")),
        };
        let kv_heads = self.pool.borrow().kv_heads;
        let step = keys.shape()[2] as usize;
        if q[0] != 1
            || q[1] == 0
            || !q[1].is_multiple_of(kv_heads)
            || q[3] != keys.shape()[3] as usize
            || !matches!(
                query.dtype(),
                Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
            )
        {
            return Ok(self.decline(
                Reason::Geometry,
                "query heads, head dimension or dtype outside the fused reader",
            ));
        }
        if q[2] != step {
            return Ok(self.decline(
                Reason::QueryShape,
                "the query must cover exactly the newly appended K/V step",
            ));
        }
        let expected_scale = (q[3] as f32).powf(-0.5);
        if scale.to_bits() != expected_scale.to_bits() {
            return Ok(self.decline(
                Reason::AttentionScale,
                "attention scale is not the inverse square-root head dimension",
            ));
        }
        let packed_mask = match mask {
            PackedAttentionMask::None => PackedMask::None,
            PackedAttentionMask::Causal => PackedMask::Causal,
            PackedAttentionMask::SlidingWindow(window)
                if window > 0 && i32::try_from(window).is_ok() =>
            {
                PackedMask::SlidingWindow(window)
            }
            PackedAttentionMask::SlidingWindow(_) => {
                return Ok(self.decline(Reason::Geometry, "sliding window must be in 1..=i32::MAX"))
            }
            PackedAttentionMask::Additive => {
                return Ok(self.decline(
                    Reason::AdditiveMask,
                    "the fused paged reader applies no additive mask",
                ))
            }
        };
        let outcome = (|| -> Result<Option<Array>> {
            self.begin_layer(layer, step)?;
            let original_len = self.pending.as_ref().expect("opened").original_len;
            self.append(layer, keys, values)?;
            // An empty cache's first multi-row step has no history: dense SDPA over the step's own
            // fresh K/V is exact and reconstructs nothing (the contiguous cache's rule).
            let fresh_mask = (original_len == 0 && step > SDPA_MAX_FUSED_QLEN as usize)
                .then_some(packed_mask)
                .and_then(|mask| match mask {
                    PackedMask::None => Some(AttnMask::None),
                    PackedMask::Causal => Some(AttnMask::Causal),
                    PackedMask::SlidingWindow(window) if window >= step => Some(AttnMask::Causal),
                    _ => None,
                });
            let output = if let Some(fresh_mask) = fresh_mask {
                self.pending.as_mut().expect("opened").store_unread = true;
                sdpa(query, keys, values, scale, fresh_mask)?
            } else {
                let warmed = self
                    .handle
                    .as_ref()
                    .is_some_and(CompiledKernelHandle::is_paged_warmed);
                match self.dispatch(layer, query, packed_mask) {
                    Ok(output) => output,
                    Err(Error::Canceled) => return Err(Error::Canceled),
                    Err(error) if layer == 0 && !warmed => {
                        // The reader never produced an output (its cold compile failed): drop it,
                        // undo this layer's append, and serve this and every later call through
                        // the gather fallback. History stays compressed.
                        self.rollback();
                        let detail = format!("paged reader unavailable: {error}");
                        self.handle = None;
                        self.reader_refusal = Some(detail.clone());
                        return Ok(self.decline(Reason::ReaderUnavailable, detail));
                    }
                    Err(error) => return Err(error),
                }
            };
            self.finish_layer()?;
            Ok(Some(output))
        })();
        if outcome.is_err() {
            self.rollback();
        }
        outcome
    }

    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let (reason, detail) = self.pending_gather.take().unwrap_or((
            PagedFallbackReason::DenseCaller,
            "dense K/V requested through update".into(),
        ));
        let outcome = (|| -> Result<(Array, Array)> {
            if let Some(problem) = self.check_kv(layer, keys, values) {
                return Err(Error::Msg(format!("paged packed update: {problem}")));
            }
            let step = keys.shape()[2] as usize;
            self.begin_layer(layer, step)?;
            self.append(layer, keys, values)?;
            let gathered = self.gather(layer)?;
            self.record_gather(reason, detail);
            self.finish_layer()?;
            Ok(gathered)
        })();
        if outcome.is_err() {
            self.rollback();
        }
        outcome
    }

    fn prepare_dense_fallback(&mut self, operation: &str, reason: &str) -> Result<()> {
        self.pending_gather = Some((
            PagedFallbackReason::DenseCaller,
            format!("{operation}: {reason}"),
        ));
        Ok(())
    }

    fn packed_evidence(&self) -> Option<PackedCacheEvidence> {
        self.evidence().ok()
    }

    /// The sequence's attributable physical storage: its pages' share of the pool arrays (codes;
    /// scale/zero metadata), plus its page table and dense residuals, measured from the arrays.
    fn compressed_storage(&self) -> Result<Option<CompressedCacheStorage>> {
        if self.logical_len == 0 {
            return Ok(None);
        }
        let (page_codes, page_metadata) = self.pool.borrow().page_bytes();
        let pages = self.page_ids.len() as u64;
        let residuals = self
            .layers
            .iter()
            .flat_map(|state| {
                let window = state.window.as_ref().and_then(|window| window.kv.as_ref());
                [
                    state.key_tail.as_ref(),
                    state.value_tail.as_ref(),
                    window.map(|(keys, _)| keys),
                    window.map(|(_, values)| values),
                ]
            })
            .flatten()
            .map(array_bytes)
            .sum::<u64>();
        let table = self.page_table.as_ref().map_or(0, array_bytes);
        let element_bytes = self.layers[0]
            .dtypes
            .map_or(0, |(key, _)| dtype_bytes(key) as u64);
        Ok(Some(CompressedCacheStorage {
            device_code_bytes: pages * page_codes,
            device_metadata_bytes: pages * page_metadata + residuals + table,
            host_payload_bytes: 0,
            tokens: self.logical_len as u64,
            element_bytes,
        }))
    }

    fn offset(&self) -> i32 {
        self.logical_len as i32
    }

    fn batch_size(&self) -> i32 {
        i32::from(self.logical_len > 0 || self.pending.is_some())
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        match keep {
            [] => self.reset(),
            [0] => Ok(()),
            other => Err(Error::Msg(format!(
                "PagedPackedKvCache is single-sequence; retain_sequences expects [] or [0], got \
                 {other:?}"
            ))),
        }
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        self.rollback();
        let base = self.speculation_base.take();
        let len = usize::try_from(len)
            .map_err(|_| Error::Msg(format!("truncate: negative len {len}")))?;
        if len >= self.logical_len {
            self.close_windows();
            return Ok(());
        }
        if base.is_some_and(|base| len >= base) {
            return self.truncate_into_window(len);
        }
        self.close_windows();
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let packed = self.layers[0].packed;
        if len >= packed {
            for state in &mut self.layers {
                state.tail_rows = len - state.packed;
            }
        } else {
            let keep = len / group * group;
            for layer in 0..self.layers.len() {
                if len > keep {
                    self.restage(layer, keep, len)?;
                }
                let state = &mut self.layers[layer];
                state.packed = keep;
                state.tail_rows = len - keep;
            }
            let page_tokens = self.pool.borrow().page_tokens;
            self.release_pages_from(keep.div_ceil(page_tokens));
        }
        self.logical_len = len;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.rollback();
        self.release_pages_from(0);
        self.layers = vec![LayerState::default(); self.layers.len()];
        self.page_table = None;
        self.pending_gather = None;
        self.speculation_base = None;
        self.logical_len = 0;
        Ok(())
    }

    fn begin_speculation(&mut self) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::Config(
                "paged packed speculation armed inside an open step".into(),
            ));
        }
        for state in &mut self.layers {
            let kv = match (&state.key_tail, &state.value_tail) {
                (Some(keys), Some(values)) if state.tail_rows > 0 => Some((
                    rows_range(keys, 0, state.tail_rows)?,
                    rows_range(values, 0, state.tail_rows)?,
                )),
                _ => None,
            };
            state.window = Some(SpeculationWindow {
                start: state.packed,
                rows: state.tail_rows,
                kv,
            });
        }
        self.speculation_base = Some(self.logical_len);
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Drop for PagedPackedKvCache {
    fn drop(&mut self) {
        self.rollback();
        self.release_pages_from(0);
    }
}

/// A paged decoder cache request under the product's compressed-KV policy.
pub struct PagedCacheRequest<'a> {
    pub policy: KvCompressionPolicy,
    /// The decoder's qualification-table family (`None` when it has none).
    pub family: Option<KvModelFamily>,
    /// Tokens the sequence prefills before decoding (the qualified minimum bounds it).
    pub prompt_tokens: u64,
    /// The most tokens the sequence may generate; the qualified maximum bounds the final context
    /// `prompt_tokens + max_new_tokens` (see [`core_llm::qualify_kv_compression`]).
    pub max_new_tokens: u64,
    /// The sequences the qualification counts as decoding together: more than one is
    /// [`core_llm::KvCacheFallbackReason::BatchedDecode`] (the table's evidence is single-sequence).
    /// Only this crate's test-armed per-sequence experiment qualifies a sequence of a batch as
    /// `1`.
    pub batch: u64,
    /// The dense block pool every dense selection draws from.
    pub dense_pool: &'a Rc<RefCell<BlockPool>>,
    /// The packed page pool a compressed selection draws from.
    pub packed_pool: &'a Rc<RefCell<PackedPagePool>>,
    /// The provider's retained fused reader, if it has one.
    pub reader: Option<&'a CompiledKernelHandle>,
}

/// The paged cache chosen for one request, before any K/V mutation.
pub struct PagedCacheSelection {
    cache: Box<dyn KvCache>,
    format: Option<KvCompressionFormat>,
    dense: Option<KvCacheReport>,
    /// The page pool a compressed selection's pages live in (its held bytes are reported).
    pool: Option<Rc<RefCell<PackedPagePool>>>,
}

impl PagedCacheSelection {
    pub(crate) fn dense(cache: PagedKvCache, report: KvCacheReport) -> Self {
        Self {
            cache: Box::new(cache),
            format: None,
            dense: Some(report),
            pool: None,
        }
    }

    pub(crate) fn compressed(cache: PagedPackedKvCache, format: KvCompressionFormat) -> Self {
        Self {
            pool: Some(cache.pool().clone()),
            cache: Box::new(cache),
            format: Some(format),
            dense: None,
        }
    }

    /// Whether the request runs on quantized pages.
    pub fn is_compressed(&self) -> bool {
        self.format.is_some()
    }

    /// Run a compressed selection on `cache` instead — a sequence started on a stored prefix
    /// (sc-20681). The fresh cache it replaces holds no pages yet.
    pub(crate) fn replace_compressed(&mut self, cache: PagedPackedKvCache) -> Result<()> {
        if self.format.is_none() {
            return Err(Error::Msg(
                "a dense paged selection cannot take a compressed prefix".into(),
            ));
        }
        self.pool = Some(cache.pool().clone());
        self.cache = Box::new(cache);
        Ok(())
    }

    pub fn cache(&self) -> &dyn KvCache {
        self.cache.as_ref()
    }

    pub fn cache_mut(&mut self) -> &mut dyn KvCache {
        self.cache.as_mut()
    }

    pub fn into_cache(self) -> Box<dyn KvCache> {
        self.cache
    }

    /// The generation's KV report: the dense reason, or the compressed cache's own evidence
    /// (dense gather fallbacks are counted and named).
    pub fn report(&self) -> Result<KvCacheReport> {
        match (&self.dense, self.format) {
            (Some(report), _) => Ok(report.clone()),
            (None, Some(format)) => {
                let mut report =
                    crate::kv_policy::compressed_report(format, None, self.cache.as_ref())?;
                if let Some(pool) = &self.pool {
                    let storage = pool.borrow().storage();
                    report.counters.pool_held_bytes =
                        storage.code_bytes.saturating_add(storage.metadata_bytes);
                }
                Ok(report)
            }
            (None, None) => Err(Error::Msg("paged selection without a report".into())),
        }
    }
}

/// One layer of several sequences on one page pool through a single fused paged dispatch
/// (sc-20681): `query` `[B, Hq, S, D]`, `keys`/`values` `[B, Hkv, S, D]`, row `b` belonging to
/// `caches[b]`. Each sequence appends its row inside its own step transaction and the reader walks
/// a `[B, max_pages]` page table with per-sequence lengths, so jagged sequences need no padding
/// mask. Returns `None` — nothing mutated — when the call is outside the batched reader (a
/// sequence without a bound or warmed reader, sequences on different pools, an additive mask, a
/// scale, shape or geometry the reader does not implement, or a fresh multi-row step); the caller
/// then serves each sequence on its own. A fault rolls back every sequence's step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_attention_batch(
    caches: &mut [&mut PagedPackedKvCache],
    layer: usize,
    query: &Array,
    keys: &Array,
    values: &Array,
    mask: PackedAttentionMask,
    scale: f32,
) -> Result<Option<Array>> {
    let Some(first) = caches.first() else {
        return Ok(None);
    };
    let pool = first.pool.clone();
    let Some(handle) = first.handle.clone() else {
        return Ok(None);
    };
    let batch = caches.len();
    let (Ok(q), Ok(k), Ok(v)) = (
        packed_input_shape(query.shape(), "query"),
        packed_input_shape(keys.shape(), "key"),
        packed_input_shape(values.shape(), "value"),
    ) else {
        return Ok(None);
    };
    let packed_mask = match mask {
        PackedAttentionMask::None => PackedMask::None,
        PackedAttentionMask::Causal => PackedMask::Causal,
        PackedAttentionMask::SlidingWindow(window)
            if window > 0 && i32::try_from(window).is_ok() =>
        {
            PackedMask::SlidingWindow(window)
        }
        _ => return Ok(None),
    };
    let (kv_heads, head_dimension) = {
        let pool = pool.borrow();
        (pool.kv_heads, pool.head_dimension)
    };
    let step = k[2];
    let float = |dtype| matches!(dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32);
    let supported = handle.is_paged_warmed()
        && k == v
        && k[0] == batch
        && q[0] == batch
        && k[1] == kv_heads
        && k[3] == head_dimension
        && step != 0
        && q[1] != 0
        && q[1].is_multiple_of(kv_heads)
        && q[2] == step
        && q[3] == head_dimension
        && float(query.dtype())
        && float(keys.dtype())
        && float(values.dtype())
        && scale.to_bits() == (head_dimension as f32).powf(-0.5).to_bits()
        && caches.iter().all(|cache| {
            Rc::ptr_eq(&cache.pool, &pool)
                && cache.handle.is_some()
                && cache.pending.as_ref().is_none_or(|pending| {
                    pending.next_layer == layer && pending.step == step
                })
                && layer < cache.layers.len()
                && cache.layers[layer]
                    .dtypes
                    .is_none_or(|dtypes| dtypes == (keys.dtype(), values.dtype()))
                // A fresh multi-row step attends its own K/V densely in every layer (see
                // `try_packed_attention`).
                && !(cache
                    .pending
                    .as_ref()
                    .map_or(cache.logical_len, |pending| pending.original_len)
                    == 0
                    && step > SDPA_MAX_FUSED_QLEN as usize)
        });
    if !supported {
        return Ok(None);
    }
    let row = |array: &Array, b: usize| -> Result<Array> {
        let b = mlx_i32(b, "batch row")?;
        Ok(array.try_index((b..b + 1, .., .., ..))?)
    };
    let (mut begun, mut dispatched) = (0, false);
    let outcome = (|| -> Result<(Array, Option<PackedKernelSelection>)> {
        for (b, cache) in caches.iter_mut().enumerate() {
            cache.begin_layer(layer, step)?;
            begun = b + 1;
            cache.append(layer, &row(keys, b)?, &row(values, b)?)?;
        }
        let tails = |pick: fn(&LayerState) -> Option<&Array>| -> Result<Array> {
            let parts = caches
                .iter()
                .map(|cache| {
                    pick(&cache.layers[layer])
                        .ok_or_else(|| Error::Msg("batched dispatch before an append".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(concatenate_axis(&parts, 0)?)
        };
        let key_tail = tails(|state| state.key_tail.as_ref())?;
        let value_tail = tails(|state| state.value_tail.as_ref())?;
        let columns = caches
            .iter()
            .map(|cache| cache.page_ids.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let mut table = Vec::with_capacity(batch * columns);
        for cache in caches.iter() {
            for column in 0..columns {
                table.push(mlx_i32(
                    cache.page_ids.get(column).copied().unwrap_or(0),
                    "page id",
                )?);
            }
        }
        let page_table = Array::from_slice(
            &table,
            &[mlx_i32(batch, "batch")?, mlx_i32(columns, "page table")?],
        );
        let sequences = caches
            .iter()
            .map(|cache| {
                let state = &cache.layers[layer];
                PagedSequenceExtent {
                    packed_tokens: state.packed,
                    kv_tokens: state.packed + state.tail_rows,
                }
            })
            .collect::<Vec<_>>();
        pool.borrow_mut().ensure_capacity(1)?;
        let pool = pool.borrow();
        let arrays = &pool.arrays[layer];
        let args = PagedPackedAttentionArgs {
            query,
            key_codes: &arrays.key_codes,
            key_scales: &arrays.key_scales,
            key_zeros: &arrays.key_zeros,
            key_tail: &key_tail,
            value_codes: &arrays.value_codes,
            value_scales: &arrays.value_scales,
            value_zeros: &arrays.value_zeros,
            value_tail: &value_tail,
            page_table: &page_table,
            page_tokens: pool.page_tokens,
            sequences: &sequences,
            code_bits: pool.bits,
            mask: packed_mask,
        };
        let selection = handle.paged_kernel_selection(&args);
        dispatched = true;
        let output = handle.dispatch_paged(&args)?;
        Ok((output, selection))
    })();
    let (output, selection) = match outcome {
        Ok(done) => done,
        Err(error) => {
            if dispatched {
                for cache in caches.iter_mut() {
                    cache.dispatch_attempts += 1;
                    cache.failed_dispatches += 1;
                }
            }
            for cache in caches.iter_mut().take(begun) {
                cache.rollback();
            }
            return Err(error);
        }
    };
    for cache in caches.iter_mut() {
        cache.dispatch_attempts += 1;
        cache.accepted.fused += 1;
        cache.accepted.batched += 1;
        if let Some(selection) = selection {
            match cache
                .accepted
                .kernel_paths
                .iter_mut()
                .find(|(recorded, _)| *recorded == selection)
            {
                Some((_, calls)) => *calls += 1,
                None => cache.accepted.kernel_paths.push((selection, 1)),
            }
        }
    }
    for index in 0..batch {
        if let Err(error) = caches[index].finish_layer() {
            for cache in caches.iter_mut() {
                cache.rollback();
            }
            return Err(error);
        }
    }
    Ok(Some(output))
}

/// Which model a paged cache's K/V belong to (sc-20681): the caller's name for the checkpoint
/// (and revision) and the loaded decoder's configuration fingerprint
/// ([`crate::models::CausalLm::cache_fingerprint`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PagedModelKey {
    pub name: String,
    pub fingerprint: String,
}

impl PagedModelKey {
    pub fn new(name: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fingerprint: fingerprint.into(),
        }
    }
}

/// What a paged compressed cache is, for sharing and restore (sc-20681): the model it was computed
/// by and the exact representation its pages hold. Two caches may share pages, and a snapshot may
/// be restored, only under identical identities.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PagedCacheIdentity {
    /// The caller's model identity (checkpoint and revision); K/V are only meaningful for it.
    pub model: String,
    /// [`PagedModelKey::fingerprint`]: the loaded decoder's configuration digest, so two
    /// checkpoints named alike (or two names for differently configured decoders) never share.
    pub model_fingerprint: String,
    /// [`core_llm::KV_CACHE_FORMAT_VERSION`] of the codes.
    pub format_version: u32,
    /// [`PAGED_PACKED_LAYOUT_VERSION`] of the pages.
    pub layout_version: u32,
    /// [`paged_packed_identity`] of the code width.
    pub representation: String,
    pub code_bits: u8,
    pub group_size: usize,
    pub page_tokens: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
}

impl PagedCacheIdentity {
    /// The first field in which `self` differs from `expected`, named, or `None` when they are
    /// identical. An identity with an empty model name or fingerprint matches nothing, itself
    /// included: K/V of an unnamed model are never shared or restored.
    pub fn mismatch(&self, expected: &Self) -> Option<String> {
        if [self, expected]
            .iter()
            .any(|identity| identity.model.is_empty() || identity.model_fingerprint.is_empty())
        {
            return Some(
                "no model identity: an empty model name or fingerprint is never shared".into(),
            );
        }
        let fields: [(&str, String, String); 11] = [
            ("model", self.model.clone(), expected.model.clone()),
            (
                "model fingerprint",
                self.model_fingerprint.clone(),
                expected.model_fingerprint.clone(),
            ),
            (
                "KV format version",
                self.format_version.to_string(),
                expected.format_version.to_string(),
            ),
            (
                "page layout version",
                self.layout_version.to_string(),
                expected.layout_version.to_string(),
            ),
            (
                "representation",
                self.representation.clone(),
                expected.representation.clone(),
            ),
            (
                "code bits",
                self.code_bits.to_string(),
                expected.code_bits.to_string(),
            ),
            (
                "group size",
                self.group_size.to_string(),
                expected.group_size.to_string(),
            ),
            (
                "page tokens",
                self.page_tokens.to_string(),
                expected.page_tokens.to_string(),
            ),
            (
                "layers",
                self.layers.to_string(),
                expected.layers.to_string(),
            ),
            (
                "KV heads",
                self.kv_heads.to_string(),
                expected.kv_heads.to_string(),
            ),
            (
                "head dimension",
                self.head_dimension.to_string(),
                expected.head_dimension.to_string(),
            ),
        ];
        fields
            .into_iter()
            .find(|(_, got, want)| got != want)
            .map(|(field, got, want)| format!("{field} {got:?} does not match {want:?}"))
    }
}

/// One layer of a [`PagedCacheSnapshot`]: raw little-endian bytes of each component.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct LayerSnapshot {
    packed: usize,
    tail_rows: usize,
    dtypes: Option<(String, String)>,
    #[serde(skip)]
    key: [Vec<u8>; 3],
    #[serde(skip)]
    value: [Vec<u8>; 3],
    #[serde(skip)]
    residual: [Vec<u8>; 2],
}

impl LayerSnapshot {
    fn blobs(&self) -> [&Vec<u8>; 8] {
        [
            &self.key[0],
            &self.key[1],
            &self.key[2],
            &self.value[0],
            &self.value[1],
            &self.value[2],
            &self.residual[0],
            &self.residual[1],
        ]
    }

    fn blobs_mut(&mut self) -> [&mut Vec<u8>; 8] {
        let [k0, k1, k2] = &mut self.key;
        let [v0, v1, v2] = &mut self.value;
        let [r0, r1] = &mut self.residual;
        [k0, k1, k2, v0, v1, v2, r0, r1]
    }
}

/// A host copy of one paged compressed sequence under its [`PagedCacheIdentity`]
/// ([`PagedPackedKvCache::snapshot`] / [`PagedPackedKvCache::restore`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PagedCacheSnapshot {
    identity: PagedCacheIdentity,
    tokens: usize,
    /// SHA-256 of the token ids the positions hold (little-endian i32), when bound
    /// ([`Self::bound_to`]); a prefix store restores a snapshot only under those tokens.
    tokens_sha256: Option<String>,
    layers: Vec<LayerSnapshot>,
}

/// SHA-256 (hex) of `tokens` as little-endian i32s.
pub fn token_digest(tokens: &[i32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Leading bytes of a serialized [`PagedCacheSnapshot`].
const SNAPSHOT_MAGIC: &[u8; 8] = b"SWPKVSN1";

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotHeader {
    identity: PagedCacheIdentity,
    tokens: usize,
    tokens_sha256: Option<String>,
    layers: Vec<LayerSnapshot>,
    /// Per layer, the byte length of each component, in [`LayerSnapshot::blobs`] order.
    lengths: Vec<[usize; 8]>,
    /// SHA-256 over this header serialized with an empty digest, then the payload: every byte
    /// of the snapshot is covered (identity, extents, dtype labels, token digest, components).
    sha256: String,
}

impl SnapshotHeader {
    /// The digest of `self` (with its digest field emptied) followed by `payload`.
    fn digest(&self, payload: &[u8]) -> Result<String> {
        use sha2::{Digest, Sha256};
        let canonical = SnapshotHeader {
            identity: self.identity.clone(),
            tokens: self.tokens,
            tokens_sha256: self.tokens_sha256.clone(),
            layers: self.layers.clone(),
            lengths: self.lengths.clone(),
            sha256: String::new(),
        };
        let header = serde_json::to_vec(&canonical)
            .map_err(|error| Error::Msg(format!("paged KV snapshot header: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(&header);
        hasher.update(payload);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

impl PagedCacheSnapshot {
    pub fn identity(&self) -> &PagedCacheIdentity {
        &self.identity
    }

    /// Positions the snapshot holds.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// [`token_digest`] of the token ids the snapshot is bound to, if bound.
    pub fn tokens_sha256(&self) -> Option<&str> {
        self.tokens_sha256.as_deref()
    }

    /// Bind the snapshot to the token ids its positions hold (one per position).
    pub fn bound_to(mut self, tokens: &[i32]) -> Result<Self> {
        if tokens.len() != self.tokens {
            return Err(Error::Msg(format!(
                "a snapshot of {} positions cannot be bound to {} tokens",
                self.tokens,
                tokens.len()
            )));
        }
        self.tokens_sha256 = Some(token_digest(tokens));
        Ok(self)
    }

    #[cfg(test)]
    pub(crate) fn identity_mut(&mut self) -> &mut PagedCacheIdentity {
        &mut self.identity
    }

    /// The snapshot as bytes for storage across a process restart: a magic, a JSON header (the
    /// identity, extents, token digest, component lengths and a SHA-256 over the header and the
    /// payload), then the components.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut payload = Vec::new();
        for layer in &self.layers {
            for blob in layer.blobs() {
                payload.extend_from_slice(blob);
            }
        }
        let mut header = SnapshotHeader {
            identity: self.identity.clone(),
            tokens: self.tokens,
            tokens_sha256: self.tokens_sha256.clone(),
            layers: self.layers.clone(),
            lengths: self
                .layers
                .iter()
                .map(|layer| layer.blobs().map(Vec::len))
                .collect(),
            sha256: String::new(),
        };
        header.sha256 = header.digest(&payload)?;
        let header = serde_json::to_vec(&header)
            .map_err(|error| Error::Msg(format!("paged KV snapshot header: {error}")))?;
        let mut bytes = Vec::with_capacity(16 + header.len() + payload.len());
        bytes.extend_from_slice(SNAPSHOT_MAGIC);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    /// Parse [`Self::to_bytes`]. Refuses anything that is not an intact snapshot (wrong magic,
    /// truncated, or a header or payload that does not match the recorded digest); the identity
    /// itself is checked when the snapshot is restored.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let refuse = |why: &str| Error::Unsupported(format!("paged KV snapshot refused: {why}"));
        let rest = bytes
            .strip_prefix(SNAPSHOT_MAGIC.as_slice())
            .ok_or_else(|| refuse("not a paged KV snapshot"))?;
        let (length, rest) = rest
            .split_first_chunk::<8>()
            .ok_or_else(|| refuse("truncated header"))?;
        let length = usize::try_from(u64::from_le_bytes(*length))
            .map_err(|_| refuse("header length out of range"))?;
        if rest.len() < length {
            return Err(refuse("truncated header"));
        }
        let (header, payload) = rest.split_at(length);
        let header: SnapshotHeader =
            serde_json::from_slice(header).map_err(|_| refuse("unreadable header"))?;
        if header.digest(payload)? != header.sha256 {
            return Err(refuse("digest mismatch (header or payload altered)"));
        }
        if header.lengths.len() != header.layers.len()
            || header
                .lengths
                .iter()
                .flatten()
                .try_fold(0_usize, |total, &len| total.checked_add(len))
                != Some(payload.len())
        {
            return Err(refuse("component lengths do not match the payload"));
        }
        let mut layers = header.layers;
        let mut offset = 0;
        for (layer, lengths) in layers.iter_mut().zip(&header.lengths) {
            for (blob, &len) in layer.blobs_mut().into_iter().zip(lengths) {
                *blob = payload[offset..offset + len].to_vec();
                offset += len;
            }
        }
        Ok(Self {
            identity: header.identity,
            tokens: header.tokens,
            tokens_sha256: header.tokens_sha256,
            layers,
        })
    }
}

fn dtype_label(dtype: Dtype) -> String {
    format!("{dtype:?}")
}

fn parse_dtype(label: &str) -> Result<Dtype> {
    [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32]
        .into_iter()
        .find(|dtype| dtype_label(*dtype) == label)
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "paged KV snapshot refused: residual dtype {label:?}"
            ))
        })
}

/// The bytes of `array`'s elements in row-major order.
fn host_bytes(array: &Array) -> Result<Vec<u8>> {
    let bytes = crate::primitives::nn::contiguous(array)?.view_dtype(Dtype::Uint8)?;
    bytes.eval()?;
    Ok(bytes.as_slice::<u8>().to_vec())
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) mod tests;
