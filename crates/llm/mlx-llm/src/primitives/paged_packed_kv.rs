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
//! [`BlockAllocator`]; freed ids are reused, so a long-running pool fragments and a sequence's page
//! table is in general a non-contiguous, non-monotonic list of ids. The pool's arrays grow by
//! doubling (one copy per growth) and keep their capacity, reported by [`PackedPagePool::storage`].
//!
//! ## Cache semantics
//!
//! [`PagedPackedKvCache`] is a single-sequence [`KvCache`] like the dense
//! [`PagedKvCache`](super::PagedKvCache). A model step is a whole-step transaction across layers:
//! a fault or cancellation at any layer restores every layer's extents and residuals and returns
//! the pages the step allocated, so the pool's live pages return to their pre-step count. Trim
//! re-stages the retained prefix of a cut group from its quantized values (as the contiguous cache
//! does), and reset/drop return every page.

use std::cell::RefCell;
use std::collections::BTreeMap;
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

    /// Grow every layer's arrays so at least `pages` pages fit (doubling, from
    /// [`MIN_POOL_PAGES`]). Existing pages keep their contents.
    fn ensure_capacity(&mut self, pages: usize) -> Result<()> {
        if pages <= self.capacity_pages {
            return Ok(());
        }
        let capacity = pages.max(self.capacity_pages * 2).max(MIN_POOL_PAGES);
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

    /// Allocate a page (refcount 1), reusing a freed id when one exists.
    fn alloc_page(&mut self) -> Result<usize> {
        let id = self.alloc.alloc();
        if let Err(error) = self.ensure_capacity(id + 1) {
            self.alloc.release(id);
            return Err(error);
        }
        Ok(id)
    }

    fn release(&mut self, id: usize) {
        self.alloc.release(id);
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
        let arrays = &self.arrays[layer];
        let ids = pages
            .iter()
            .map(|&id| mlx_i32(id, "page id"))
            .collect::<Result<Vec<_>>>()?;
        let index = Array::from_slice(&ids, &[mlx_i32(ids.len(), "page count")?]);
        let heads = mlx_i32(self.kv_heads, "KV heads")?;
        // `[n, Hkv, rows, width]` pages → `[1, Hkv, n · rows, width]` in position order.
        let sequence = |array: &Array| -> Result<Array> {
            let shape = array.shape();
            let gathered = array.take_axis(&index, 0)?.transpose_axes(&[1, 0, 2, 3])?;
            Ok(gathered.reshape(&[1, heads, -1, shape[3]])?)
        };
        let groups = packed / group;
        let keys = dequantize_key_groups(
            geometry,
            &rows_range(&sequence(&arrays.key_codes)?, 0, groups)?,
            &rows_range(&sequence(&arrays.key_scales)?, 0, groups)?,
            &rows_range(&sequence(&arrays.key_zeros)?, 0, groups)?,
        )?;
        let values = dequantize_value_rows(
            geometry,
            &rows_range(&sequence(&arrays.value_codes)?, 0, packed)?,
            &rows_range(&sequence(&arrays.value_scales)?, 0, packed)?,
            &rows_range(&sequence(&arrays.value_zeros)?, 0, packed)?,
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
}

/// Calls accepted in committed steps; restored by a whole-step rollback.
#[derive(Clone, Debug, Default)]
struct AcceptedCalls {
    fused: u64,
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
        let (geometry, page_tokens) = {
            let pool = self.pool.borrow();
            (pool.geometry(), pool.page_tokens)
        };
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
        self.cover(new_packed, page_tokens)?;
        let page_groups = page_tokens / group;
        let mut done = 0;
        while done < groups {
            let global = packed / group + done;
            let page = self.page_ids[global / page_groups];
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
                [
                    &slice(&key_codes, 1)?,
                    &slice(&key_scales, 1)?,
                    &slice(&key_zeros, 1)?,
                ],
                [
                    &slice(&value_codes, group)?,
                    &slice(&value_scales, group)?,
                    &slice(&value_zeros, group)?,
                ],
            )?;
            done += count;
        }
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
            self.page_table = Some(Array::from_slice(
                &[0_i32; MIN_TABLE_COLUMNS],
                &[1, MIN_TABLE_COLUMNS as i32],
            ));
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
        let warmed = handle.is_warmed();
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
        handle.mark_warmed();
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
        Ok(())
    }

    /// Undo the open step: every layer's extents and residuals, its pages, its accepted calls.
    fn rollback(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.layers = pending.marks;
            self.release_pages_from(pending.pages_before);
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
                .is_some_and(CompiledKernelHandle::is_warmed),
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
                    .is_some_and(CompiledKernelHandle::is_warmed);
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
            .flat_map(|state| [state.key_tail.as_ref(), state.value_tail.as_ref()])
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
        let len = usize::try_from(len)
            .map_err(|_| Error::Msg(format!("truncate: negative len {len}")))?;
        if len >= self.logical_len {
            return Ok(());
        }
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
        self.logical_len = 0;
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
    /// The context the request is qualified against (see [`core_llm::qualify_kv_compression`]).
    pub context_tokens: u64,
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
}

impl PagedCacheSelection {
    pub(crate) fn dense(cache: PagedKvCache, report: KvCacheReport) -> Self {
        Self {
            cache: Box::new(cache),
            format: None,
            dense: Some(report),
        }
    }

    pub(crate) fn compressed(cache: PagedPackedKvCache, format: KvCompressionFormat) -> Self {
        Self {
            cache: Box::new(cache),
            format: Some(format),
            dense: None,
        }
    }

    /// Whether the request runs on quantized pages.
    pub fn is_compressed(&self) -> bool {
        self.format.is_some()
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
                crate::kv_policy::compressed_report(format, None, self.cache.as_ref())
            }
            (None, None) => Err(Error::Msg("paged selection without a report".into())),
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
    use crate::primitives::packed_group_affine_kv::RetainedPackedKernel;
    use crate::primitives::packed_metal::{
        PackedKernelPath, PackedMetalGpuFamily, PackedMetalKernel,
    };
    use crate::primitives::{select_decoder_cache_with_reader, PackedCacheRequest};
    use std::sync::Arc;

    /// Deterministic K/V/Q values with a distinct range per 32-channel group and per token, so a
    /// wrong page, group or metadata index is numerically visible.
    fn values(seed: u64, tokens: usize, heads: usize, width: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..heads * tokens * width)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let (channel, token) = (i % width, (i / width) % tokens);
                unit * (1.0 + (channel / 32) as f32 * 0.5) + (channel / 32) as f32 * 0.75
                    - (token % 7) as f32 * 0.125
            })
            .collect()
    }

    fn bhsd(data: &[f32], heads: usize, tokens: usize, width: usize, dtype: Dtype) -> Array {
        Array::from_slice(data, &[1, heads as i32, tokens as i32, width as i32])
            .as_dtype(dtype)
            .unwrap()
    }

    fn host<T: mlx_rs::ArrayElement + Copy>(array: &Array) -> Vec<T> {
        let array = crate::primitives::nn::contiguous(array).unwrap();
        array.eval().unwrap();
        array.as_slice::<T>().to_vec()
    }

    fn host_f32(array: &Array) -> Vec<f32> {
        host::<f32>(&array.as_dtype(Dtype::Float32).unwrap())
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn reader(bits: PackedCodeBits, family: PackedMetalGpuFamily) -> CompiledKernelHandle {
        CompiledKernelHandle::new(Arc::new(
            PackedMetalKernel::for_identity_family_and_bits("paged-test", family, bits).unwrap(),
        ))
    }

    /// Independent host dequantization of one layer of `cache`: K/V of every live token, read by
    /// walking the host page ids over host copies of the pool arrays with this test's own code
    /// unpacking (no MLX dequantization, no kernel). `[Hkv][tokens][D]` flattened.
    fn oracle_kv(cache: &PagedPackedKvCache, layer: usize) -> (Vec<f32>, Vec<f32>) {
        let pool = cache.pool.borrow();
        let (h, d, pt, bits) = (
            pool.kv_heads,
            pool.head_dimension,
            pool.page_tokens,
            pool.bits,
        );
        let group = PACKED_METAL_QUANT_GROUP_SIZE;
        let arrays = &pool.arrays[layer];
        let kc = host::<u8>(&arrays.key_codes);
        let ks = host_f32(&arrays.key_scales);
        let kz = host_f32(&arrays.key_zeros);
        let vc = host::<u8>(&arrays.value_codes);
        let vs = host_f32(&arrays.value_scales);
        let vz = host_f32(&arrays.value_zeros);
        let state = &cache.layers[layer];
        let kt = host_f32(state.key_tail.as_ref().unwrap());
        let vt = host_f32(state.value_tail.as_ref().unwrap());
        let per_byte = 8 / bits.bits() as usize;
        let mask = (1u32 << bits.bits()) - 1;
        let code = |bytes: &[u8], base: usize, index: usize| -> f32 {
            let byte = u32::from(bytes[base + index / per_byte]);
            ((byte >> ((index % per_byte) * bits.bits() as usize)) & mask) as f32
        };
        let (pg, kw, vw, vg) = (pt / group, group * d / per_byte, d / per_byte, d / group);
        let tokens = state.packed + state.tail_rows;
        let mut keys = vec![0.0; h * tokens * d];
        let mut vals = vec![0.0; h * tokens * d];
        for head in 0..h {
            for t in 0..tokens {
                for c in 0..d {
                    let out = (head * tokens + t) * d + c;
                    if t < state.packed {
                        let page = cache.page_ids[t / pt];
                        let kg = (page * h + head) * pg + (t % pt) / group;
                        keys[out] = kz[kg * d + c]
                            + ks[kg * d + c] * code(&kc, kg * kw, (t % group) * d + c);
                        let row = (page * h + head) * pt + t % pt;
                        vals[out] = vz[row * vg + c / group]
                            + vs[row * vg + c / group] * code(&vc, row * vw, c);
                    } else {
                        let r = (head * group + t - state.packed) * d + c;
                        keys[out] = kt[r];
                        vals[out] = vt[r];
                    }
                }
            }
        }
        (keys, vals)
    }

    /// Append one layer-0 chunk directly (kernel-level fixtures; no attention).
    fn append(
        cache: &mut PagedPackedKvCache,
        keys: &[f32],
        vals: &[f32],
        tokens: usize,
        dtype: Dtype,
    ) {
        let (h, d) = {
            let pool = cache.pool.borrow();
            (pool.kv_heads, pool.head_dimension)
        };
        cache
            .append(
                0,
                &bhsd(keys, h, tokens, d, dtype),
                &bhsd(vals, h, tokens, d, dtype),
            )
            .unwrap();
        cache.logical_len += tokens;
    }

    struct KernelCase {
        dtype: Dtype,
        width: usize,
        kv_heads: usize,
        query_heads: usize,
        page_tokens: usize,
        lengths: [usize; 3],
        query_len: usize,
        mask: PackedAttentionMask,
    }

    /// The fused paged reader against an independent dequantize-then-attend fp32 oracle, for
    /// several sequences of one dispatch with ragged lengths whose pages are interleaved and
    /// recycled (non-contiguous, non-monotonic page order), partially filled last pages, residual
    /// tails, a sequence with no full group, GQA, causal/sliding/no masks, decode and multi-row
    /// queries, every code width, and every kernel path (per-row single/split on both families,
    /// fp32 tiled single/split, and NAX where MLX runs it). Unread page-table columns hold an
    /// out-of-range id, so a read past a sequence's pages corrupts the output.
    #[test]
    fn paged_reader_matches_the_independent_fp32_oracle_across_paths_and_page_orders() {
        let case = |dtype, width, kv_heads, query_heads, page_tokens, lengths, query_len, mask| {
            KernelCase {
                dtype,
                width,
                kv_heads,
                query_heads,
                page_tokens,
                lengths,
                query_len,
                mask,
            }
        };
        let causal = PackedAttentionMask::Causal;
        let cases = [
            case(Dtype::Float32, 64, 2, 4, 32, [161, 70, 20], 1, causal),
            case(Dtype::Float16, 128, 2, 6, 64, [300, 129, 97], 3, causal),
            case(
                Dtype::Bfloat16,
                256,
                1,
                2,
                96,
                [200, 33, 100],
                1,
                PackedAttentionMask::None,
            ),
            case(
                Dtype::Float32,
                128,
                2,
                4,
                64,
                [257, 140, 100],
                2,
                PackedAttentionMask::SlidingWindow(45),
            ),
            case(Dtype::Bfloat16, 64, 2, 4, 32, [230, 96, 40], 32, causal),
            case(Dtype::Float16, 128, 1, 4, 128, [190, 120, 50], 16, causal),
        ];
        let mut nax_ran = false;
        for bits in PackedCodeBits::ALL {
            for (index, case) in cases.iter().enumerate() {
                nax_ran |= check_kernel_case(bits, index, case);
            }
        }
        eprintln!("NAX paged path exercised: {nax_ran}");
    }

    /// Run one [`KernelCase`]; returns whether the NAX path ran.
    fn check_kernel_case(bits: PackedCodeBits, index: usize, case: &KernelCase) -> bool {
        let (h, d, pt) = (case.kv_heads, case.width, case.page_tokens);
        let pool = PackedPagePool::new(1, h, d, pt, bits).unwrap();
        let handle = reader(bits, PackedMetalGpuFamily::Apple7OrNewer);
        let mut caches = (0..3)
            .map(|_| PagedPackedKvCache::with_pool(pool.clone(), handle.clone()).unwrap())
            .collect::<Vec<_>>();
        // Recycle: a decoy takes pages, then frees them, so the sequences reuse those ids in LIFO
        // (descending) order before fresh ones.
        {
            let mut decoy = PagedPackedKvCache::with_pool(pool.clone(), handle.clone()).unwrap();
            let data = values(999, 5 * pt, h, d);
            append(&mut decoy, &data, &data, 5 * pt, case.dtype);
        }
        assert_eq!(
            pool.borrow().live_pages(),
            0,
            "the decoy returned its pages"
        );
        // Interleave the sequences' appends in uneven chunks so their pages interleave.
        let seed = index as u64 + 1;
        let keys = (0..3)
            .map(|b| values(17 * seed + b as u64, case.lengths[b], h, d))
            .collect::<Vec<_>>();
        let vals = (0..3)
            .map(|b| values(101 * seed + b as u64, case.lengths[b], h, d))
            .collect::<Vec<_>>();
        let mut fed = [0usize; 3];
        let chunks = [37, 5, 64, 1, 29];
        let mut turn = 0;
        while (0..3).any(|b| fed[b] < case.lengths[b]) {
            for b in 0..3 {
                let chunk = chunks[(turn + b) % chunks.len()].min(case.lengths[b] - fed[b]);
                if chunk == 0 {
                    continue;
                }
                let take = |data: &[f32]| {
                    (0..h)
                        .flat_map(|head| {
                            let start = (head * case.lengths[b] + fed[b]) * d;
                            data[start..start + chunk * d].to_vec()
                        })
                        .collect::<Vec<_>>()
                };
                append(
                    &mut caches[b],
                    &take(&keys[b]),
                    &take(&vals[b]),
                    chunk,
                    case.dtype,
                );
                fed[b] += chunk;
            }
            turn += 1;
        }
        let ids = caches
            .iter()
            .map(|c| c.page_ids.clone())
            .collect::<Vec<_>>();
        assert!(
            ids.iter()
                .any(|pages| pages.windows(2).any(|w| w[1] != w[0] + 1)),
            "case {index}: some sequence's pages are non-contiguous: {ids:?}"
        );
        let query = Array::from_slice(
            &(0..3 * case.query_heads * case.query_len * d)
                .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
                .collect::<Vec<_>>(),
            &[3, case.query_heads as i32, case.query_len as i32, d as i32],
        )
        .as_dtype(case.dtype)
        .unwrap();
        let query_f32 = host_f32(&query);
        let stack =
            |parts: Vec<Array>| concatenate_axis(&parts.iter().collect::<Vec<_>>(), 0).unwrap();
        let key_tail = stack(
            caches
                .iter()
                .map(|c| c.layers[0].key_tail.clone().unwrap())
                .collect(),
        );
        let value_tail = stack(
            caches
                .iter()
                .map(|c| c.layers[0].value_tail.clone().unwrap())
                .collect(),
        );
        let columns = ids.iter().map(Vec::len).max().unwrap().max(1);
        let table = ids
            .iter()
            .flat_map(|pages| {
                let mut row = pages.iter().map(|&p| p as i32).collect::<Vec<_>>();
                row.resize(columns, i32::MAX / 2);
                row
            })
            .collect::<Vec<_>>();
        let page_table = Array::from_slice(&table, &[3, columns as i32]);
        let sequences = caches
            .iter()
            .map(|c| PagedSequenceExtent {
                packed_tokens: c.layers[0].packed,
                kv_tokens: c.layers[0].packed + c.layers[0].tail_rows,
            })
            .collect::<Vec<_>>();
        // A partially filled last page (or a sequence with no page yet) in every case whose page
        // is larger than one group.
        assert!(
            pt == PACKED_METAL_QUANT_GROUP_SIZE
                || sequences
                    .iter()
                    .any(|s| s.packed_tokens == 0 || s.packed_tokens % pt != 0)
        );
        let packed_mask = match case.mask {
            PackedAttentionMask::None => PackedMask::None,
            PackedAttentionMask::Causal => PackedMask::Causal,
            PackedAttentionMask::SlidingWindow(w) => PackedMask::SlidingWindow(w),
            PackedAttentionMask::Additive => unreachable!(),
        };
        // Independent oracle, per sequence.
        let mut reference = Vec::new();
        for (b, cache) in caches.iter().enumerate() {
            let (k, v) = oracle_kv(cache, 0);
            let kv = sequences[b].kv_tokens;
            let rows = case.query_heads * case.query_len * d;
            reference.extend(
                attention_f32_masked(
                    PackedAttentionShape {
                        batch: 1,
                        query_heads: case.query_heads,
                        kv_heads: h,
                        query_len: case.query_len,
                        kv_len: kv,
                        head_dim: d,
                    },
                    &query_f32[b * rows..(b + 1) * rows],
                    |_, hh, t, c| k[(hh * kv + t) * d + c],
                    |_, hh, t, c| v[(hh * kv + t) * d + c],
                    (d as f32).powf(-0.5),
                    case.mask,
                )
                .unwrap(),
            );
        }
        let pool_ref = pool.borrow();
        let arrays = &pool_ref.arrays[0];
        let args = PagedPackedAttentionArgs {
            query: &query,
            key_codes: &arrays.key_codes,
            key_scales: &arrays.key_scales,
            key_zeros: &arrays.key_zeros,
            key_tail: &key_tail,
            value_codes: &arrays.value_codes,
            value_scales: &arrays.value_scales,
            value_zeros: &arrays.value_zeros,
            value_tail: &value_tail,
            page_table: &page_table,
            page_tokens: pt,
            sequences: &sequences,
            code_bits: bits,
            mask: packed_mask,
        };
        let tolerance = match case.dtype {
            Dtype::Float32 => 1e-4,
            Dtype::Float16 => 4e-2,
            _ => 8e-2,
        };
        use PackedMetalGpuFamily::{Apple7OrNewer, ConservativeUnknownApple};
        let mut paths = vec![
            (
                ConservativeUnknownApple,
                PackedKernelPath::PerRow { splits: 1 },
            ),
            (
                ConservativeUnknownApple,
                PackedKernelPath::PerRow { splits: 3 },
            ),
            (Apple7OrNewer, PackedKernelPath::PerRow { splits: 1 }),
            (Apple7OrNewer, PackedKernelPath::PerRow { splits: 64 }),
            (Apple7OrNewer, PackedKernelPath::Tiled { splits: 1 }),
            (Apple7OrNewer, PackedKernelPath::Tiled { splits: 2 }),
        ];
        let qualified =
            PackedMetalKernel::for_identity_family_and_bits("paged-test", Apple7OrNewer, bits)
                .unwrap();
        let nax = qualified.nax_selection(d, case.dtype).selected;
        if nax {
            paths.push((Apple7OrNewer, PackedKernelPath::NaxTiled { splits: 1 }));
            paths.push((Apple7OrNewer, PackedKernelPath::NaxTiled { splits: 3 }));
        }
        let mut case_max = 0.0f32;
        for (family, path) in paths {
            let kernel =
                PackedMetalKernel::for_identity_family_and_bits("paged-test", family, bits)
                    .unwrap();
            let output = host_f32(&kernel.dispatch_paged_path(&args, path).unwrap());
            assert_eq!(output.len(), reference.len());
            for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                case_max = case_max.max((actual - expected).abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{bits:?} case {index} {family:?} {path:?} element {i}: {actual} != {expected}"
                );
            }
        }
        // The production planner picks one of those paths for this shape.
        let planned = host_f32(&qualified.dispatch_paged(&args).unwrap());
        for (actual, expected) in planned.iter().zip(&reference) {
            assert!((actual - expected).abs() <= tolerance);
        }
        eprintln!(
            "paged parity {bits:?} case {index} {:?} D={d}: max abs {case_max}",
            case.dtype
        );
        nax
    }

    /// A paged dispatch is refused before encoding when a sequence's quantized tokens are not
    /// covered by its page-table row, the page size disagrees with the pool arrays, or the
    /// reader's code width differs; a pool refuses a page that is not whole groups.
    #[test]
    fn paged_dispatch_validates_page_geometry_before_encoding() {
        let bits = PackedCodeBits::Eight;
        assert!(PackedPagePool::new(1, 1, 64, 48, bits).is_err());
        assert!(PackedPagePool::new(1, 1, 96, 32, bits).is_err());
        let pool = PackedPagePool::new(1, 1, 64, 32, bits).unwrap();
        let handle = reader(bits, PackedMetalGpuFamily::Apple7OrNewer);
        let mut cache = PagedPackedKvCache::with_pool(pool.clone(), handle).unwrap();
        let data = values(3, 70, 1, 64);
        append(&mut cache, &data, &data, 70, Dtype::Float32);
        assert_eq!(cache.page_ids.len(), 2);
        let query = Array::from_slice(&[0.1f32; 64], &[1, 1, 1, 64]);
        let pool_ref = pool.borrow();
        let arrays = &pool_ref.arrays[0];
        let state = &cache.layers[0];
        let short_table = Array::from_slice(&[0i32], &[1, 1]);
        let table = cache.page_table.clone().unwrap();
        let sequences = [PagedSequenceExtent {
            packed_tokens: 64,
            kv_tokens: 70,
        }];
        let kernel = PackedMetalKernel::for_identity_family_and_bits(
            "paged-test",
            PackedMetalGpuFamily::Apple7OrNewer,
            bits,
        )
        .unwrap();
        let args = |table, page_tokens, code_bits| PagedPackedAttentionArgs {
            query: &query,
            key_codes: &arrays.key_codes,
            key_scales: &arrays.key_scales,
            key_zeros: &arrays.key_zeros,
            key_tail: state.key_tail.as_ref().unwrap(),
            value_codes: &arrays.value_codes,
            value_scales: &arrays.value_scales,
            value_zeros: &arrays.value_zeros,
            value_tail: state.value_tail.as_ref().unwrap(),
            page_table: table,
            page_tokens,
            sequences: &sequences,
            code_bits,
            mask: PackedMask::Causal,
        };
        assert!(kernel.dispatch_paged(&args(&table, 32, bits)).is_ok());
        assert!(kernel
            .dispatch_paged(&args(&short_table, 32, bits))
            .is_err());
        assert!(kernel.dispatch_paged(&args(&table, 64, bits)).is_err());
        assert!(kernel
            .dispatch_paged(&args(&table, 32, PackedCodeBits::Four))
            .is_err());
    }

    const LAYERS: usize = 2;

    fn contiguous_cache(
        handle: &CompiledKernelHandle,
        layers: usize,
        kv_heads: usize,
        width: usize,
    ) -> Box<dyn KvCache> {
        select_decoder_cache_with_reader(
            PackedCacheRequest {
                enabled: true,
                backend: "mlx-metal".into(),
                identity: handle.cache_identity().to_owned(),
                layers,
                batch: 1,
                kv_heads,
                head_dimension: width,
                group_size: PACKED_METAL_QUANT_GROUP_SIZE,
                bits: handle.code_bits(),
                query_length: 1,
                has_mask: false,
            },
            handle.clone(),
        )
        .into_cache()
    }

    /// Q/K/V of one synthetic layer step.
    fn qkv(
        seed: u64,
        step: usize,
        (heads, kv_heads, width): (usize, usize, usize),
        dtype: Dtype,
    ) -> (Array, Array, Array) {
        (
            bhsd(&values(seed, step, heads, width), heads, step, width, dtype),
            bhsd(
                &values(seed + 7, step, kv_heads, width),
                kv_heads,
                step,
                width,
                dtype,
            ),
            bhsd(
                &values(seed + 13, step, kv_heads, width),
                kv_heads,
                step,
                width,
                dtype,
            ),
        )
    }

    /// One synthetic model step on a cache: per layer, fresh Q/K/V of `step` tokens attended
    /// causally; the fused route where the cache takes it, else `update` + dense SDPA.
    fn attend_step(
        cache: &mut dyn KvCache,
        seed: u64,
        step: usize,
        dims: (usize, usize, usize),
        dtype: Dtype,
    ) -> Vec<Vec<f32>> {
        let scale = (dims.2 as f32).powf(-0.5);
        (0..cache.num_layers())
            .map(|layer| {
                let (q, k, v) = qkv(seed * 31 + layer as u64, step, dims, dtype);
                let out = match cache
                    .try_packed_attention(
                        layer,
                        &q,
                        &k,
                        &v,
                        PackedAttentionMask::Causal,
                        scale,
                        false,
                    )
                    .unwrap()
                {
                    Some(out) => out,
                    None => {
                        let (keys, vals) = cache.update(layer, &k, &v).unwrap();
                        sdpa(&q, &keys, &vals, scale, AttnMask::Causal).unwrap()
                    }
                };
                host_f32(&out)
            })
            .collect()
    }

    fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| x.iter().zip(y))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn pool(kv_heads: usize, width: usize, page_tokens: usize) -> Rc<RefCell<PackedPagePool>> {
        PackedPagePool::new(LAYERS, kv_heads, width, page_tokens, PackedCodeBits::Eight).unwrap()
    }

    fn k8v8_reader() -> CompiledKernelHandle {
        crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).unwrap()
    }

    /// Cache-level differential parity over a synthetic two-layer attention stack: the paged
    /// compressed cache matches the contiguous compressed cache (same codes, same kernels) and the
    /// dense paged cache within 8-bit rounding, through a fresh prefill, decode across page
    /// boundaries, a short multi-row step (per-row kernel), and a long one (tiled/NAX kernel). The
    /// fused calls build no dense copy: no gather is ever counted.
    #[test]
    fn paged_compressed_matches_contiguous_compressed_and_dense_paged_through_a_decode() {
        let dims = (4, 2, 64);
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let handle = k8v8_reader();
            let mut paged = PagedPackedKvCache::with_pool(pool(2, 64, 64), handle.clone()).unwrap();
            let mut contiguous = contiguous_cache(&handle, LAYERS, 2, 64);
            let mut dense = PagedKvCache::new(LAYERS, 16);
            let steps = [70, 1, 1, 1, 5]
                .into_iter()
                .chain(std::iter::repeat_n(1, 30))
                .chain([20])
                .chain(std::iter::repeat_n(1, 40))
                .collect::<Vec<_>>();
            let (mut to_contiguous, mut to_dense) = (0.0f32, 0.0f32);
            let mut tokens = 0;
            for (i, &step) in steps.iter().enumerate() {
                let p = attend_step(&mut paged, i as u64, step, dims, dtype);
                let c = attend_step(contiguous.as_mut(), i as u64, step, dims, dtype);
                let d = attend_step(&mut dense, i as u64, step, dims, dtype);
                tokens += step;
                to_contiguous = to_contiguous.max(max_abs(&p, &c));
                to_dense = to_dense.max(max_abs(&p, &d));
                assert_eq!(paged.offset(), tokens as i32);
            }
            let (contiguous_bound, dense_bound) = match dtype {
                Dtype::Float32 => (1e-5, 3e-2),
                _ => (2e-2, 8e-2),
            };
            assert!(
                to_contiguous <= contiguous_bound,
                "{dtype:?}: paged vs contiguous {to_contiguous}"
            );
            assert!(
                to_dense <= dense_bound,
                "{dtype:?}: paged vs dense {to_dense}"
            );
            eprintln!("{dtype:?}: paged vs contiguous {to_contiguous}, vs dense paged {to_dense}");
            assert_eq!(paged.page_ids.len(), (tokens / 32 * 32).div_ceil(64));
            let evidence = paged.packed_evidence().unwrap();
            assert!(evidence.dense_gather_calls.is_empty(), "{evidence:?}");
            assert_eq!(evidence.full_cache_dequantizations, 0);
            assert_eq!(
                evidence.representation_identity,
                paged_packed_identity(PackedCodeBits::Eight)
            );
            // Every attention call after the fresh prefill ran fused.
            let fused = (LAYERS * (steps.len() - 1)) as u64;
            assert_eq!(paged.fused_calls(), fused);
            assert!(evidence
                .kernel_paths
                .iter()
                .any(|path| path.kernel != crate::primitives::PACKED_PER_ROW_KERNEL));
        }
    }

    /// Physical-byte accounting includes page metadata: the sequence's storage is its pages'
    /// share of the pool's measured arrays (codes; f16 scale/zero) plus its page table and dense
    /// residuals, and it stays well under the dense bf16 K/V of the same tokens despite the
    /// partially filled last page.
    #[test]
    fn physical_bytes_count_pages_page_metadata_table_and_residuals() {
        let (kv_heads, width) = (4, 128);
        let pool = pool(kv_heads, width, 64);
        let mut cache = PagedPackedKvCache::with_pool(pool.clone(), k8v8_reader()).unwrap();
        let dims = (4, kv_heads, width);
        attend_step(&mut cache, 1, 2020, dims, Dtype::Bfloat16);
        for i in 0..10 {
            attend_step(&mut cache, 2 + i, 1, dims, Dtype::Bfloat16);
        }
        let tokens = 2030_u64;
        let storage = cache.compressed_storage().unwrap().unwrap();
        assert_eq!(storage.tokens, tokens);
        let pool_storage = pool.borrow().storage();
        let (page_codes, page_metadata) = pool.borrow().page_bytes();
        assert_eq!(
            pool_storage.code_bytes,
            pool_storage.capacity_pages * page_codes
        );
        assert_eq!(
            pool_storage.metadata_bytes,
            pool_storage.capacity_pages * page_metadata
        );
        let pages = cache.page_ids.len() as u64;
        assert_eq!(
            pages, 32,
            "2016 quantized tokens: 31 full pages and a partial 32nd"
        );
        assert_eq!(pool_storage.live_pages, pages);
        assert_eq!(storage.device_code_bytes, pages * page_codes);
        let residuals = (LAYERS * 2 * kv_heads * 32 * width * 2) as u64;
        let table = array_bytes(cache.page_table.as_ref().unwrap());
        assert_eq!(table, 32 * 4, "a 32-column Int32 page table");
        assert_eq!(
            storage.device_metadata_bytes,
            pages * page_metadata + residuals + table
        );
        // Per page and layer and head: K and V codes 64·128 bytes each; metadata K 2 groups ×
        // 128 channels × (scale + zero) f16, V 64 tokens × 4 channel groups × (scale + zero) f16.
        assert_eq!(page_codes, (LAYERS * kv_heads * 64 * 128 * 2) as u64);
        assert_eq!(
            page_metadata,
            (LAYERS * kv_heads * (2 * 128 * 4 + 64 * 4 * 4)) as u64
        );
        let dense_bf16 = LAYERS as u64 * 2 * kv_heads as u64 * tokens * width as u64 * 2;
        let ratio = storage.device_bytes() as f64 / dense_bf16 as f64;
        assert!(ratio < 0.6, "paged K8V8 holds {ratio:.3} of dense bf16");
        // Slack: 32 unused rows of the last page and 18 unused residual rows.
        assert_eq!(cache.slack_tokens(), 32 * 64 - 2016 + 32 - 14);
        eprintln!(
            "paged storage {} B for {tokens} tokens = {ratio:.3} of dense bf16 (slack {} tokens)",
            storage.device_bytes(),
            cache.slack_tokens()
        );
    }

    /// Page reuse and fragmentation: sequences interleaved on one pool, one freed mid-way and its
    /// pages recycled by a third, each produce exactly the outputs they produce alone on a fresh
    /// pool, and the pool returns to zero live pages when every sequence is dropped.
    #[test]
    fn interleaved_freed_and_recycled_pages_do_not_change_any_sequence() {
        let dims = (2, 1, 64);
        let dtype = Dtype::Float32;
        let handle = k8v8_reader();
        let alone = |seed: u64, steps: &[usize]| {
            let mut cache = PagedPackedKvCache::with_pool(pool(1, 64, 32), handle.clone()).unwrap();
            steps
                .iter()
                .enumerate()
                .map(|(i, &step)| {
                    attend_step(&mut cache, seed * 1000 + i as u64, step, dims, dtype)
                })
                .collect::<Vec<_>>()
        };
        let a_steps = [40, 1, 30, 1, 1, 33];
        let b_steps = [50, 1, 1, 40, 1, 1];
        let c_steps = [100, 1, 64, 1, 1, 1];
        let shared = pool(1, 64, 32);
        let mut a = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
        let mut b = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
        let (mut a_out, mut b_out) = (Vec::new(), Vec::new());
        for i in 0..a_steps.len() {
            a_out.push(attend_step(
                &mut a,
                1000 + i as u64,
                a_steps[i],
                dims,
                dtype,
            ));
            b_out.push(attend_step(
                &mut b,
                2000 + i as u64,
                b_steps[i],
                dims,
                dtype,
            ));
        }
        assert_eq!(a_out, alone(1, &a_steps), "A interleaved with B");
        assert_eq!(b_out, alone(2, &b_steps), "B interleaved with A");
        assert!(
            a.page_ids().windows(2).any(|w| w[1] != w[0] + 1),
            "{:?}",
            a.page_ids()
        );
        let a_pages = a.page_ids().to_vec();
        drop(a);
        assert_eq!(shared.borrow().live_pages(), b.page_ids().len());
        let mut c = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
        let c_out = c_steps
            .iter()
            .enumerate()
            .map(|(i, &step)| attend_step(&mut c, 3000 + i as u64, step, dims, dtype))
            .collect::<Vec<_>>();
        assert_eq!(c_out, alone(3, &c_steps), "C on recycled pages");
        assert!(
            c.page_ids()
                .iter()
                .filter(|id| a_pages.contains(id))
                .count()
                == a_pages.len(),
            "C reused every page A freed: {:?} vs {a_pages:?}",
            c.page_ids()
        );
        assert!(
            c.page_ids().windows(2).any(|w| w[1] < w[0]),
            "non-monotonic page order: {:?}",
            c.page_ids()
        );
        drop(b);
        drop(c);
        assert_eq!(shared.borrow().live_pages(), 0);
    }

    /// A reader that cancels (or faults) on one chosen paged dispatch.
    #[derive(Debug)]
    struct Interrupting {
        inner: PackedMetalKernel,
        fail_at: std::cell::Cell<usize>,
        canceled: bool,
    }

    impl RetainedPackedKernel for Interrupting {
        fn cache_identity(&self) -> &str {
            "interrupting"
        }
        fn backend(&self) -> &str {
            "mlx-metal"
        }
        fn retained_host_bytes_estimate(&self) -> usize {
            0
        }
        fn dispatch(&self, args: &crate::primitives::PackedAttentionArgs<'_>) -> Result<Array> {
            self.inner.dispatch(args)
        }
        fn dispatch_paged(&self, args: &PagedPackedAttentionArgs<'_>) -> Result<Array> {
            let left = self.fail_at.get();
            self.fail_at.set(left.wrapping_sub(1));
            match left {
                0 if self.canceled => Err(Error::Canceled),
                0 => Err(Error::Msg("injected paged dispatch fault".into())),
                _ => self.inner.dispatch_paged(args),
            }
        }
        fn code_bits(&self) -> PackedCodeBits {
            PackedCodeBits::Eight
        }
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn interrupting(fail_at: usize, canceled: bool) -> CompiledKernelHandle {
        CompiledKernelHandle::new(Arc::new(Interrupting {
            inner: PackedMetalKernel::for_identity_family_and_bits(
                "interrupting",
                PackedMetalGpuFamily::Apple7OrNewer,
                PackedCodeBits::Eight,
            )
            .unwrap(),
            fail_at: std::cell::Cell::new(fail_at),
            canceled,
        }))
    }

    /// Cancellation or a fault at a later layer of a step that crossed a page boundary rolls back
    /// the whole step: offset, residuals, and the page it allocated (the pool's live pages return
    /// to their pre-step count); retrying produces the uninterrupted outputs. Dropping a sequence
    /// mid-generation (a cancelled request) returns every page.
    #[test]
    fn a_cancelled_or_faulted_step_returns_its_pages_and_retries_exactly() {
        let dims = (2, 1, 64);
        let dtype = Dtype::Float32;
        let steps = [63, 1, 1];
        let run = |cache: &mut PagedPackedKvCache| {
            steps
                .iter()
                .enumerate()
                .map(|(i, &s)| attend_step(cache, i as u64, s, dims, dtype))
                .collect::<Vec<_>>()
        };
        let reference =
            run(&mut PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap());
        for canceled in [true, false] {
            // Dispatch 0 is step 1's layer 0 (the 63-token prefill is fresh dense SDPA); fail
            // dispatch 1, step 1's layer 1. Step 1 completes token 64's group, whose page the
            // step allocates at layer 0.
            let shared = pool(1, 64, 32);
            let mut cache =
                PagedPackedKvCache::with_pool(shared.clone(), interrupting(1, canceled)).unwrap();
            let mut outputs = vec![attend_step(&mut cache, 0, steps[0], dims, dtype)];
            assert_eq!(cache.offset(), 63);
            let live = shared.borrow().live_pages();
            assert_eq!(live, 1);
            let scale = 0.125;
            let (q0, k0, v0) = qkv(31, 1, dims, dtype);
            assert!(cache
                .try_packed_attention(0, &q0, &k0, &v0, PackedAttentionMask::Causal, scale, false)
                .unwrap()
                .is_some());
            assert_eq!(
                shared.borrow().live_pages(),
                live + 1,
                "the step allocated a page"
            );
            let (q1, k1, v1) = qkv(32, 1, dims, dtype);
            let error = cache
                .try_packed_attention(1, &q1, &k1, &v1, PackedAttentionMask::Causal, scale, false)
                .unwrap_err();
            assert_eq!(matches!(error, Error::Canceled), canceled, "{error}");
            assert_eq!(cache.offset(), 63, "the interrupted step did not commit");
            assert_eq!(
                shared.borrow().live_pages(),
                live,
                "the step's page went back"
            );
            assert!(cache.pending.is_none());
            assert_eq!(cache.layers[0].tail_rows, 31, "layer 0's append was undone");
            // Retry the whole step, then finish: identical to the uninterrupted run.
            for (i, &s) in steps.iter().enumerate().skip(1) {
                outputs.push(attend_step(&mut cache, i as u64, s, dims, dtype));
            }
            assert_eq!(outputs, reference, "canceled={canceled}");
            assert_eq!(cache.fused_calls(), 4, "rolled-back calls are not accepted");
            drop(cache);
            assert_eq!(
                shared.borrow().live_pages(),
                0,
                "a dropped sequence returns its pages"
            );
        }
    }

    /// Truncation (speculative rollback) into a quantized group re-stages the cut group from its
    /// codes and frees the pages past it, exactly as the contiguous compressed cache trims.
    #[test]
    fn truncate_into_a_quantized_group_matches_the_contiguous_cache_and_frees_pages() {
        let dims = (2, 1, 64);
        let dtype = Dtype::Float32;
        let handle = k8v8_reader();
        let shared = pool(1, 64, 32);
        let mut paged = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
        let mut contiguous = contiguous_cache(&handle, LAYERS, 1, 64);
        for (i, step) in [100, 1, 1].into_iter().enumerate() {
            attend_step(&mut paged, i as u64, step, dims, dtype);
            attend_step(contiguous.as_mut(), i as u64, step, dims, dtype);
        }
        assert_eq!(paged.page_ids.len(), 3);
        paged.truncate(45).unwrap();
        contiguous.truncate(45).unwrap();
        assert_eq!((paged.offset(), contiguous.offset()), (45, 45));
        assert_eq!(
            paged.page_ids.len(),
            1,
            "pages past the kept group are freed"
        );
        assert_eq!(shared.borrow().live_pages(), 1);
        let mut worst = 0.0f32;
        for (i, step) in [1, 1, 30, 1].into_iter().enumerate() {
            let p = attend_step(&mut paged, 50 + i as u64, step, dims, dtype);
            let c = attend_step(contiguous.as_mut(), 50 + i as u64, step, dims, dtype);
            worst = worst.max(max_abs(&p, &c));
        }
        assert!(worst <= 1e-5, "paged vs contiguous after truncate: {worst}");
        paged.reset().unwrap();
        assert_eq!(shared.borrow().live_pages(), 0);
        assert_eq!(paged.offset(), 0);
    }

    /// An unsupported call takes the dense gather fallback: the caller's `update` returns the
    /// sequence's dequantized K/V (the independent oracle's values), the reason is counted, the
    /// history stays compressed, and the next supported call is fused again. The report names the
    /// gather.
    #[test]
    fn unsupported_calls_take_the_counted_dense_gather_and_history_stays_paged() {
        let dims = (2, 1, 64);
        let dtype = Dtype::Float32;
        let mut cache = PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap();
        attend_step(&mut cache, 0, 70, dims, dtype);
        let fused = cache.fused_calls();
        let pages = cache.page_ids.clone();
        let (q, k, v) = qkv(9, 1, dims, dtype);
        for (layer, mask, scale, reason) in [
            (
                0,
                PackedAttentionMask::Additive,
                0.125,
                PagedFallbackReason::AdditiveMask,
            ),
            (
                1,
                PackedAttentionMask::Causal,
                0.5,
                PagedFallbackReason::AttentionScale,
            ),
        ] {
            assert!(cache
                .try_packed_attention(layer, &q, &k, &v, mask, scale, false)
                .unwrap()
                .is_none());
            let (keys, vals) = cache.update(layer, &k, &v).unwrap();
            assert_eq!(keys.shape(), &[1, 1, 71, 64]);
            let (ok, ov) = oracle_kv(&cache, layer);
            assert_eq!(
                host_f32(&keys),
                ok,
                "{reason:?}: gathered keys are the paged values"
            );
            assert_eq!(host_f32(&vals), ov);
            assert!(cache.dense_gathers().contains(&(reason, 1)));
        }
        assert_eq!(cache.offset(), 71);
        assert_eq!(
            cache.page_ids[..pages.len()],
            pages[..],
            "history stayed in its pages"
        );
        assert_eq!(cache.fused_calls(), fused);
        attend_step(&mut cache, 12, 1, dims, dtype);
        assert_eq!(
            cache.fused_calls(),
            fused + LAYERS as u64,
            "fused again after the gathers"
        );
        // A soft-cap preparation and a direct `update` (a caller wanting dense K/V) count too.
        cache
            .prepare_dense_fallback("score-softcap", "tanh before softmax")
            .unwrap();
        cache.update(0, &k, &v).unwrap();
        cache.update(1, &k, &v).unwrap();
        assert!(cache
            .dense_gathers()
            .contains(&(PagedFallbackReason::DenseCaller, 2)));
        let report =
            crate::kv_policy::compressed_report(KvCompressionFormat::GroupAffineK8V8, None, &cache)
                .unwrap();
        assert_eq!(
            report.fallback,
            Some(core_llm::KvCacheFallbackReason::DenseGather)
        );
        assert_eq!(report.counters.dense_gather_fallbacks, 4);
        assert_eq!(report.counters.full_cache_dequantizations, 0);
        assert_eq!(report.counters.dense_fallback_events, 0);
        assert_eq!(report.counters.fused_attention_calls, fused + LAYERS as u64);
        let detail = report.detail.unwrap();
        for reason in ["additive_mask", "attention_scale", "dense_caller"] {
            assert!(detail.contains(reason), "{detail}");
        }
        let ids = PagedFallbackReason::ALL.map(PagedFallbackReason::id);
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            ids.len()
        );
    }

    /// A reader whose paged kernels fail on their cold dispatch is dropped before any packed output
    /// is published; that call and every later one are served by the gather fallback with the
    /// reader-unavailable reason, and the outputs stay the fused paged attention's.
    #[test]
    fn a_cold_paged_reader_fault_degrades_to_counted_gathers() {
        let dims = (2, 1, 64);
        let dtype = Dtype::Float32;
        let mut cache =
            PagedPackedKvCache::with_pool(pool(1, 64, 32), interrupting(0, false)).unwrap();
        let mut healthy = PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap();
        for (i, step) in [3, 1, 1, 40].into_iter().enumerate() {
            let a = attend_step(&mut cache, i as u64, step, dims, dtype);
            let b = attend_step(&mut healthy, i as u64, step, dims, dtype);
            // Same values either way; only the attention arithmetic differs. MLX's vector SDPA
            // (up to 8 query rows) agrees with the fused reader to f32 rounding; its multi-row
            // steel kernel differs from the fp32 tiled reader (itself within 1e-4 of the fp32
            // oracle) by up to ~1e-3.
            let bound = if step <= SDPA_MAX_FUSED_QLEN as usize {
                1e-5
            } else {
                2e-3
            };
            let worst = max_abs(&a, &b);
            assert!(
                worst <= bound,
                "step {i}: gathered vs fused paged attention {worst}"
            );
        }
        assert_eq!(cache.fused_calls(), 0);
        assert_eq!(
            cache.dense_gathers(),
            vec![(PagedFallbackReason::ReaderUnavailable, 4 * LAYERS as u64)]
        );
        assert!(cache.handle.is_none());
        assert_eq!(cache.offset(), 45);
    }

    /// Synthetic decode timing (measurement, not a gate): one decode token's attention over a
    /// long history on the fused paged reader, the fused contiguous reader (same codes), and the
    /// dense paged cache (gather + SDPA). Hq = 32, Hkv = 8, D = 128, bf16, 64-token pages.
    #[test]
    #[ignore = "GPU micro-benchmark; run explicitly with --ignored --nocapture"]
    fn paged_decode_timing_versus_contiguous_and_dense_paged() {
        let dims = (32, 8, 128);
        let dtype = Dtype::Bfloat16;
        let scale = (128f32).powf(-0.5);
        let median = |mut samples: Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        for context in [4096, 16384, 32768] {
            let handle = k8v8_reader();
            let pages = PackedPagePool::new(1, 8, 128, 64, PackedCodeBits::Eight).unwrap();
            let mut paged = PagedPackedKvCache::with_pool(pages, handle.clone()).unwrap();
            let mut contiguous = contiguous_cache(&handle, 1, 8, 128);
            let mut dense = PagedKvCache::new(1, 16);
            let chunk = 2048;
            for start in (0..context).step_by(chunk) {
                let (q, k, v) = qkv(start as u64, chunk, dims, dtype);
                for cache in [&mut paged as &mut dyn KvCache, contiguous.as_mut()] {
                    cache
                        .try_packed_attention(
                            0,
                            &q,
                            &k,
                            &v,
                            PackedAttentionMask::Causal,
                            scale,
                            false,
                        )
                        .unwrap()
                        .expect("prefill chunk accepted")
                        .eval()
                        .unwrap();
                }
                let (kk, vv) = dense.update(0, &k, &v).unwrap();
                mlx_rs::transforms::eval([&kk, &vv]).unwrap();
            }
            let (mut p, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..45 {
                let (q, k, v) = qkv(i, 1, dims, dtype);
                let timed = |run: &mut dyn FnMut() -> Array, into: &mut Vec<f64>| {
                    let started = std::time::Instant::now();
                    run().eval().unwrap();
                    into.push(started.elapsed().as_secs_f64() * 1e3);
                };
                timed(
                    &mut || {
                        paged
                            .try_packed_attention(
                                0,
                                &q,
                                &k,
                                &v,
                                PackedAttentionMask::Causal,
                                scale,
                                false,
                            )
                            .unwrap()
                            .unwrap()
                    },
                    &mut p,
                );
                timed(
                    &mut || {
                        contiguous
                            .try_packed_attention(
                                0,
                                &q,
                                &k,
                                &v,
                                PackedAttentionMask::Causal,
                                scale,
                                false,
                            )
                            .unwrap()
                            .unwrap()
                    },
                    &mut c,
                );
                timed(
                    &mut || {
                        let (kk, vv) = dense.update(0, &k, &v).unwrap();
                        sdpa(&q, &kk, &vv, scale, AttnMask::Causal).unwrap()
                    },
                    &mut d,
                );
            }
            eprintln!(
                "decode attention @ {context} tokens (median of 40 after 5 warm-up, ms): \
                 fused-paged {:.3}, fused-contiguous {:.3}, dense-paged {:.3}",
                median(p[5..].to_vec()),
                median(c[5..].to_vec()),
                median(d[5..].to_vec())
            );
        }
    }
}
