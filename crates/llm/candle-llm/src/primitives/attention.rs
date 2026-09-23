//! Attention leaves: grouped-query KV expansion and an eager scaled-dot-product-attention.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, d] -> [b, hkv*groups, s, d]`
//! broadcast (the Candle port of `mlx-llm`'s `repeat_kv`, matching `candle-gen-sensenova`).
//!
//! Candle has no portable fused causal SDPA across CPU/CUDA (flash-attn is a separate, CUDA-only
//! crate), so [`sdpa`] has an eager fallback — `softmax(scale · QKᵀ + mask) · V`. That fallback
//! processes at most `EAGER_ATTN_QUERY_CHUNK_SIZE` query rows at a time, bounding scores, masks,
//! and weights at `O(heads · query_chunk · k_len)` rather than materializing three full
//! `O(heads · q_len · k_len)` tensors. Chunk masks retain the full query's bottom-right alignment:
//! global query row `r` attends keys `0..=(k_len - total_q_len) + r`. A single-query decode still
//! skips its provably all-zero causal mask.
//!
//! With the `flash-attn` feature, [`sdpa`] first tries the fused FlashAttention-2 kernel
//! (`candle_flash_attn::flash_attn`) for the dense causal/bidirectional path and falls back to the
//! eager kernel for everything it cannot serve — Gemma-2 score soft-cap, MLA's mismatched q/v head
//! dims, an explicit additive (padded-batch) mask, or a non-f16/bf16 dtype. FlashAttention's causal
//! masking is bottom-right aligned (`window_size_right = 0`), matching `causal_mask`'s convention,
//! so cached decode stays correct. Numerics differ by a few half-precision ULPs from the eager path
//! (different reduction order), the same tolerance the batched / prefix-reuse GPU paths carry.
//!
//! [`sdpa_gqa_causal`] (epic sc-24128, story sc-24132) is the **zero-copy grouped-query** causal
//! attention the static-KV decode path runs: queries `[b, H, s, d]` against un-expanded keys/values
//! `[b, Hkv, L, d]`, with the `H / Hkv` query groups folded into the query-sequence axis so one
//! batched matmul per side serves every group — no [`repeat_kv`] expansion, and no `contiguous`
//! copy of the cache's narrowed K/V views (the matmul reads their strides directly). The causal
//! mask broadcasts over the groups from a 5-D view of the scores; a single-query decode step builds
//! none. Numerically it is the eager path's arithmetic in a different batching (`groups` query rows
//! per matmul instead of one), so it agrees with `repeat_kv` + [`sdpa`] to the backend's
//! reduction order — a few ULPs, the same tolerance the flash path carries; the real-weight
//! greedy fixture (`tests/static_kv_parity.rs`) is the token-level gate.
//!
//! The continuous-batching `Throughput` path (story 7347) decodes many sequences at once over
//! per-sequence paged caches; its attention used to be an N-call per-sequence SDPA loop, which
//! flatlined throughput at occupancy (the cost is N kernel launches + N gathers, not per-kernel
//! speed). `try_flash_attn_varlen` (story 7351) folds that loop into **one**
//! `candle_flash_attn::flash_attn_varlen` call over the ragged (gathered) KV of all active
//! sequences — no padding mask, no per-sequence launch — packed via cumulative `cu_seqlens` offsets.
//! It is grouped-query-native (K/V passed un-expanded) and bottom-right causal; the eager per-sequence
//! loop stays the fallback for the cases varlen cannot serve (soft-cap, f32/CPU, no `flash-attn`).

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;

use crate::error::{Error, Result};

/// Disallowed-attention fill for the additive mask: a large finite negative (matching the
/// candle-gen slices — avoids `-inf` propagation through the softmax kernel).
const MASK_NEG: f32 = -1e30;

/// Maximum query rows in one portable eager-attention tile.
///
/// The resource estimator imports this exact constant. At the Qwen3-VL shape (32 heads), a tile
/// keeps every score-like CUDA allocation below signed 32-bit element indexing even when the key
/// run exceeds 8K tokens, and bounds CPU prefill workspace without changing attention semantics.
/// [`sdpa_eager_with_query_chunk_size`] lowers it further when batch/head/key dimensions require
/// that to keep the flattened tile within `i32::MAX`.
pub(crate) const EAGER_ATTN_QUERY_CHUNK_SIZE: usize = 256;

/// How attention should be masked.
#[derive(Debug, Clone, Copy)]
pub enum AttnMask<'a> {
    /// No mask (fully bidirectional) — e.g. a vision tower.
    None,
    /// Bottom-right-aligned causal mask: query row `r` attends keys `0..=(k_len - q_len) + r`.
    Causal,
    /// An explicit additive mask broadcast over the score tensor (`0` keep, large-negative block).
    Additive(&'a Tensor),
    /// An explicit additive mask narrowed by a sliding causal band. This represents Gemma 4's
    /// padded-batch sliding layers without first materializing a full combined square mask.
    AdditiveSliding {
        /// Caller-provided additive mask, broadcastable over `[batch, heads, q_len, k_len]`.
        additive: &'a Tensor,
        /// Number of recent keys visible to each query, including its own position.
        window: i32,
    },
    /// **Sliding-window** causal mask (Gemma 4's `sliding_attention` layers): causal *and* limited
    /// to the `window` most recent keys, so query `q` sees key `j` iff `0 <= q - j < window` (the
    /// query's own position counts toward the window). Queries are bottom-right aligned over the
    /// cached keys like [`AttnMask::Causal`], so a cached decode step attends the tail of its cache.
    ///
    /// Built by [`sliding_causal_mask`] and applied on the eager path; the fused FlashAttention
    /// wrapper cannot express it, so [`sdpa`] falls back to eager for these layers.
    SlidingCausal {
        /// Number of most-recent keys a query may attend, inclusive of its own position. A window
        /// `>= k_len` is exactly [`AttnMask::Causal`].
        window: i32,
    },
}

/// How grouped-query attention is computed over the cached K/V (story sc-24132) — surfaced per
/// request through [`DecodeRecord::attn_formulation`](crate::decode::DecodeRecord::attn_formulation)
/// so an evidence row says which arithmetic produced its tokens.
///
/// The two are **not** bit-identical on CUDA: cuBLAS picks its kernel (and with it the fp32
/// reduction order that decides the last bf16 bit) by the GEMM's `m`, batch count and strides,
/// and the un-expanded formulation issues different GEMMs (`m = groups`, `batch = b·kv_heads`)
/// than the expanded one (`m = 1`, `batch = b·heads`). They differ by at most one bf16 ULP at
/// attention-GEMM knife-edges; see `docs/migration/evidence/sc-24132/README.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttnFormulation {
    /// [`sdpa_gqa_causal`]: the query groups folded onto the sequence axis, attending the
    /// un-expanded K/V views directly — the default for every path since S4 (the reference oracle
    /// included, so static-vs-reference parity holds by construction).
    #[default]
    Gqa,
    /// The pre-S4 arithmetic: [`repeat_kv`]-expanded K/V through [`sdpa`]. Kept selectable only as
    /// a labelled comparison row against the sealed pre-epic baseline (it reproduces that
    /// baseline's bits); it materializes the expansion every step, so it is never the fast path.
    Expanded,
}

impl AttnFormulation {
    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            AttnFormulation::Gqa => "gqa",
            AttnFormulation::Expanded => "expanded",
        }
    }
}

/// Expand grouped-query KV heads to the full query head count.
///
/// `x` is `[batch, n_kv_heads, seq, head_dim]`; the result is `[batch, n_kv_heads * groups, seq,
/// head_dim]` where `groups = n_query_heads / n_kv_heads`. `groups == 1` (MHA) is a no-op clone.
pub fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    crate::primitives::kv_cache::note_kv_materialize();
    let (b, hkv, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, hkv, groups, s, d))?
        .contiguous()?
        .reshape((b, hkv * groups, s, d))?)
}

/// Build the additive causal mask `[1, 1, q_len, k_len]` (`0` keep / [`MASK_NEG`] block) for keys
/// that include `offset = k_len - q_len` cached positions before the new queries.
fn causal_mask_chunk(
    query_start: usize,
    query_len: usize,
    total_query_len: usize,
    k_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    #[cfg(test)]
    {
        CAUSAL_MASK_BUILDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        MAX_CAUSAL_MASK_ROWS.fetch_max(query_len, std::sync::atomic::Ordering::SeqCst);
    }
    let offset = k_len.checked_sub(total_query_len).ok_or_else(|| {
        Error::Msg(format!(
            "causal attention needs k_len >= q_len, got {k_len} keys and {total_query_len} queries"
        ))
    })?;
    let mut data = vec![0f32; query_len * k_len];
    for r in 0..query_len {
        for j in 0..k_len {
            if j > offset + query_start + r {
                data[r * k_len + j] = MASK_NEG;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, query_len, k_len), device)?.to_dtype(dtype)?)
}

/// Build a complete bottom-right causal mask. Production eager attention uses bounded
/// [`causal_mask_chunk`] tiles; this wrapper remains the small-shape reference used by tests.
#[cfg(test)]
fn causal_mask(q_len: usize, k_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    causal_mask_chunk(0, q_len, q_len, k_len, dtype, device)
}

/// The additive **sliding-window** causal mask `[1, 1, q_len, k_len]` (`0` keep / a large finite
/// negative to block) — Gemma 4's `sliding_attention` layers.
///
/// Queries are bottom-right aligned over the keys (`offset = k_len - q_len` cached positions come
/// first), so query row `r` sits at absolute position `offset + r` and may attend key `j` iff
/// `0 <= (offset + r) - j < window`: causal, *and* no further back than `window - 1` positions. A
/// `window >= k_len` degenerates to the plain causal mask; a `window <= 0` is rejected rather than
/// silently producing an all-blocked row (whose softmax is a uniform distribution over garbage).
pub fn sliding_causal_mask(
    q_len: usize,
    k_len: usize,
    window: i32,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    sliding_causal_mask_chunk(0, q_len, q_len, k_len, window, dtype, device)
}

fn sliding_causal_mask_chunk(
    query_start: usize,
    query_len: usize,
    total_query_len: usize,
    k_len: usize,
    window: i32,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if window <= 0 {
        return Err(Error::Msg(format!(
            "sliding_causal_mask: window must be positive, got {window}"
        )));
    }
    let window = window as i64;
    let offset = k_len.checked_sub(total_query_len).ok_or_else(|| {
        Error::Msg(format!(
            "sliding causal attention needs k_len >= q_len, got {k_len} keys and {total_query_len} queries"
        ))
    })? as i64;
    let mut data = vec![0f32; query_len * k_len];
    for r in 0..query_len {
        let pos = offset + query_start as i64 + r as i64;
        for j in 0..k_len {
            let delta = pos - j as i64;
            if !(0..window).contains(&delta) {
                data[r * k_len + j] = MASK_NEG;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, query_len, k_len), device)?.to_dtype(dtype)?)
}

/// Number of host-side causal-mask tiles built, used to pin the decode fast path and tile bound.
#[cfg(test)]
static CAUSAL_MASK_BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static MAX_CAUSAL_MASK_ROWS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Eager scaled-dot-product attention over `[batch, heads, seq, head_dim]` tensors.
///
/// `scale` is the usual `head_dim^(-0.5)`. `softcap`, when set, applies Gemma-2's score soft-cap
/// (`c·tanh(scores/c)`) after scaling and before masking. K/V must already be GQA-expanded (see
/// [`repeat_kv`]).
pub fn sdpa(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Tensor> {
    #[cfg(feature = "flash-attn")]
    if let Some(out) = try_flash_attn(queries, keys, values, scale, softcap, mask)? {
        return Ok(out);
    }
    sdpa_eager(queries, keys, values, scale, softcap, mask)
}

/// The eager `softmax(scale · QKᵀ + mask) · V` path — the portable fallback [`sdpa`] runs when the
/// fused kernel is unavailable or ineligible. Kept as its own entry point so the flash-vs-eager
/// parity test can compare the two paths directly on one device.
fn sdpa_eager(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Tensor> {
    sdpa_eager_with_query_chunk_size(
        queries,
        keys,
        values,
        scale,
        softcap,
        mask,
        EAGER_ATTN_QUERY_CHUNK_SIZE,
    )
}

fn additive_mask_chunk(
    mask: &Tensor,
    query_start: usize,
    query_len: usize,
    total_query_len: usize,
) -> Result<Tensor> {
    let dims = mask.dims();
    if dims.len() < 2 {
        return Err(Error::Msg(format!(
            "additive attention mask must have at least two dimensions, got {dims:?}"
        )));
    }
    let query_axis = dims.len() - 2;
    match dims[query_axis] {
        1 => Ok(mask.clone()),
        len if len == total_query_len => Ok(mask.narrow(query_axis, query_start, query_len)?),
        len => Err(Error::Msg(format!(
            "additive attention mask query dimension must be 1 or {total_query_len}, got {len}"
        ))),
    }
}

fn sdpa_eager_with_query_chunk_size(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
    query_chunk_size: usize,
) -> Result<Tensor> {
    if query_chunk_size == 0 {
        return Err(Error::Msg(
            "eager attention query chunk size must be positive".into(),
        ));
    }
    let (b, h, q_len, _d) = queries.dims4()?;
    let k_len = keys.dim(2)?;
    let elements_per_query_row = b
        .checked_mul(h)
        .and_then(|elements| elements.checked_mul(k_len))
        .ok_or_else(|| Error::Msg("eager attention tile size overflow".into()))?;
    if elements_per_query_row == 0 {
        return Err(Error::Msg(
            "eager attention requires non-empty batch, heads, and keys".into(),
        ));
    }
    let indexable_query_rows = (i32::MAX as usize) / elements_per_query_row;
    if indexable_query_rows == 0 {
        return Err(Error::Msg(format!(
            "eager attention cannot index one query row with batch={b}, heads={h}, k_len={k_len}"
        )));
    }
    let query_chunk_size = query_chunk_size.min(indexable_query_rows);
    let queries = queries.contiguous()?;
    let keys_t = keys.transpose(2, 3)?.contiguous()?;
    let values = values.contiguous()?;
    let mut chunks = Vec::with_capacity(q_len.div_ceil(query_chunk_size));
    for query_start in (0..q_len).step_by(query_chunk_size) {
        let query_len = query_chunk_size.min(q_len - query_start);
        let query = queries.narrow(2, query_start, query_len)?.contiguous()?;
        let mut scores = (query.matmul(&keys_t)? * scale as f64)?;
        if let Some(c) = softcap {
            scores = crate::primitives::nn::soft_cap(&scores, c)?;
        }
        let scores = match mask {
            AttnMask::None => scores,
            // A single-query bottom-right-aligned causal mask is provably all zeros (its one row
            // blocks `j > k_len - 1`, impossible for `j < k_len`), so decode skips the mask build.
            AttnMask::Causal if q_len == 1 => scores,
            AttnMask::Causal => {
                let tile = causal_mask_chunk(
                    query_start,
                    query_len,
                    q_len,
                    k_len,
                    scores.dtype(),
                    scores.device(),
                )?;
                scores.broadcast_add(&tile)?
            }
            AttnMask::Additive(additive) => {
                let tile = additive_mask_chunk(additive, query_start, query_len, q_len)?;
                scores.broadcast_add(&tile)?
            }
            AttnMask::SlidingCausal { window } => {
                let tile = sliding_causal_mask_chunk(
                    query_start,
                    query_len,
                    q_len,
                    k_len,
                    window,
                    scores.dtype(),
                    scores.device(),
                )?;
                scores.broadcast_add(&tile)?
            }
            AttnMask::AdditiveSliding { additive, window } => {
                let additive = additive_mask_chunk(additive, query_start, query_len, q_len)?;
                let band = sliding_causal_mask_chunk(
                    query_start,
                    query_len,
                    q_len,
                    k_len,
                    window,
                    scores.dtype(),
                    scores.device(),
                )?;
                let combined = additive.broadcast_add(&band)?;
                scores.broadcast_add(&combined)?
            }
        };
        let weights = softmax_last_dim(&scores)?;
        chunks.push(weights.matmul(&values)?);
    }
    if chunks.len() == 1 {
        Ok(chunks.pop().expect("one eager attention output chunk"))
    } else {
        Ok(Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?)
    }
}

/// Convenience: causal attention with no soft-cap (the decode default).
pub fn sdpa_causal(queries: &Tensor, keys: &Tensor, values: &Tensor, scale: f32) -> Result<Tensor> {
    sdpa(queries, keys, values, scale, None, AttnMask::Causal)
}

/// Zero-copy grouped-query causal attention (see the module docs).
///
/// `queries` is `[batch, heads, q_len, head_dim]`; `keys`/`values` are the **un-expanded**
/// `[batch, kv_heads, k_len, head_dim]` (any strides the matmul can read — a static cache's
/// narrowed views included), with `heads` a multiple of `kv_heads` and `k_len >= q_len`. Head `h`
/// of the output attends KV head `h / groups`, the [`repeat_kv`] convention. Returns
/// `[batch, heads, q_len, head_dim]`, contiguous.
///
/// Always the folded eager path, on every device. The fused `flash-attn` kernel is deliberately
/// **not** tried here (unlike [`sdpa`]): it would receive un-expanded, narrowed K/V — a GQA + stride
/// combination no test covers on this repository's CUDA lane (the feature is not part of it), so
/// until a flash-vs-eager parity test exists for those views the fused kernel stays out of this
/// path. Query rows are tiled like [`sdpa`] so every score tile stays within signed 32-bit
/// element indexing.
pub fn sdpa_gqa_causal(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let (b, h, q_len, d) = queries.dims4()?;
    let (bk, hkv, k_len, dk) = keys.dims4()?;
    if bk != b || dk != d || values.dims() != keys.dims() || hkv == 0 || h % hkv != 0 {
        return Err(Error::Msg(format!(
            "sdpa_gqa_causal: queries {:?} do not group over keys {:?} / values {:?}",
            queries.dims(),
            keys.dims(),
            values.dims()
        )));
    }
    if k_len < q_len {
        return Err(Error::Msg(format!(
            "causal attention needs k_len >= q_len, got {k_len} keys and {q_len} queries"
        )));
    }
    let groups = h / hkv;
    // Same tile bound as the eager path: the folded tile has `hkv * groups * rows == h * rows`
    // score elements per batch row, so the arithmetic is unchanged.
    let elements_per_query_row = b
        .checked_mul(h)
        .and_then(|e| e.checked_mul(k_len))
        .ok_or_else(|| Error::Msg("eager attention tile size overflow".into()))?;
    if elements_per_query_row == 0 {
        return Err(Error::Msg(
            "eager attention requires non-empty batch, heads, and keys".into(),
        ));
    }
    let indexable_query_rows = (i32::MAX as usize) / elements_per_query_row;
    if indexable_query_rows == 0 {
        return Err(Error::Msg(format!(
            "eager attention cannot index one query row with batch={b}, heads={h}, k_len={k_len}"
        )));
    }
    let chunk = EAGER_ATTN_QUERY_CHUNK_SIZE.min(indexable_query_rows);
    // Keys transposed for the score matmul — a strided view, never materialized.
    let keys_t = keys.transpose(2, 3)?;
    let mut chunks = Vec::with_capacity(q_len.div_ceil(chunk));
    for query_start in (0..q_len).step_by(chunk) {
        let query_len = chunk.min(q_len - query_start);
        // [b, H, rows, d] -> [b, Hkv, groups * rows, d]: a free reshape of the contiguous query
        // (a whole-range `narrow` is the tensor itself, so the decode step copies nothing).
        let query = queries
            .narrow(2, query_start, query_len)?
            .contiguous()?
            .reshape((b, hkv, groups * query_len, d))?;
        let scores = (query.matmul(&keys_t)? * scale as f64)?; // [b, Hkv, groups * rows, L]
                                                               // A single-query bottom-right causal mask is all zeros: skip it (the decode step).
        let scores = if q_len == 1 {
            scores
        } else {
            let tile = causal_mask_chunk(
                query_start,
                query_len,
                q_len,
                k_len,
                scores.dtype(),
                scores.device(),
            )?; // [1, 1, rows, L]
            scores
                .reshape((b, hkv, groups, query_len, k_len))?
                .broadcast_add(&tile.unsqueeze(2)?)?
                .reshape((b, hkv, groups * query_len, k_len))?
        };
        let weights = softmax_last_dim(&scores)?;
        // [b, Hkv, groups * rows, d] -> [b, H, rows, d] (head h = kv * groups + g).
        chunks.push(weights.matmul(values)?.reshape((b, h, query_len, d))?);
    }
    if chunks.len() == 1 {
        Ok(chunks.pop().expect("one gqa attention output chunk"))
    } else {
        Ok(Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?)
    }
}

/// The additive length mask `[1, 1, q_len, capacity]` over a **preallocated** key buffer, built
/// from device data only (story sc-24134): query row `i` may attend key `j` iff
/// `j <= limits[i]`, where `limits` is `[q_len]` `u32` (the cache position of each query token,
/// `pos + i`) and `arange` is the cache's constant `[capacity]` `u32` ramp `0, 1, …`. Allowed
/// positions get exactly `0`, refused ones the same `MASK_NEG` the causal tiles use, so adding
/// it leaves allowed scores bit-identical and drives refused ones to an exact zero weight.
///
/// Nothing here touches the host: the position is read from `limits` by the kernels, which is
/// what lets a CUDA graph replay the same mask build at every position.
pub fn capacity_mask(arange: &Tensor, limits: &Tensor, dtype: DType) -> Result<Tensor> {
    let capacity = arange.dims1()?;
    let q_len = limits.dims1()?;
    let j = arange.unsqueeze(0)?.broadcast_as((q_len, capacity))?;
    let limit = limits.unsqueeze(1)?.broadcast_as((q_len, capacity))?;
    // `allowed` is exactly 0 or 1; `1·v − v = 0` and `0·v − v = −v` are exact in every dtype.
    let neg = f64::from(MASK_NEG);
    let mask = j.le(&limit)?.to_dtype(dtype)?.affine(-neg, neg)?;
    Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
}

/// [`sdpa_gqa_causal`] over a **full preallocated** K/V buffer `[b, hkv, capacity, d]` with an
/// explicit additive mask `[1, 1, q_len, capacity]` (from [`capacity_mask`]) in place of the
/// bounded `narrow` view plus the built-in causal tile (story sc-24134). Same folded
/// grouped-query arithmetic, same chunking; the key extent is the buffer's capacity, a constant
/// for the life of the cache, and the *length* is data (the mask), so the op sequence is
/// identical at every position — the form a CUDA graph can capture once and replay. On the
/// static-cache path this is the arithmetic both the eager step and the replayed graph run.
pub fn sdpa_gqa_masked(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, h, q_len, d) = queries.dims4()?;
    let (bk, hkv, capacity, dk) = keys.dims4()?;
    if bk != b || dk != d || values.dims() != keys.dims() || hkv == 0 || h % hkv != 0 {
        return Err(Error::Msg(format!(
            "sdpa_gqa_masked: queries {:?} do not group over keys {:?} / values {:?}",
            queries.dims(),
            keys.dims(),
            values.dims()
        )));
    }
    if mask.dims() != [1, 1, q_len, capacity] {
        return Err(Error::Msg(format!(
            "sdpa_gqa_masked: mask {:?} must be [1, 1, {q_len}, {capacity}]",
            mask.dims()
        )));
    }
    let groups = h / hkv;
    let elements_per_query_row = b
        .checked_mul(h)
        .and_then(|e| e.checked_mul(capacity))
        .ok_or_else(|| Error::Msg("eager attention tile size overflow".into()))?;
    if elements_per_query_row == 0 {
        return Err(Error::Msg(
            "eager attention requires non-empty batch, heads, and keys".into(),
        ));
    }
    let indexable_query_rows = (i32::MAX as usize) / elements_per_query_row;
    if indexable_query_rows == 0 {
        return Err(Error::Msg(format!(
            "eager attention cannot index one query row with batch={b}, heads={h}, capacity={capacity}"
        )));
    }
    let chunk = EAGER_ATTN_QUERY_CHUNK_SIZE.min(indexable_query_rows);
    let keys_t = keys.transpose(2, 3)?;
    let mut chunks = Vec::with_capacity(q_len.div_ceil(chunk));
    for query_start in (0..q_len).step_by(chunk) {
        let query_len = chunk.min(q_len - query_start);
        let query = queries
            .narrow(2, query_start, query_len)?
            .contiguous()?
            .reshape((b, hkv, groups * query_len, d))?;
        let scores = (query.matmul(&keys_t)? * scale as f64)?; // [b, Hkv, groups * rows, cap]
        let tile = mask.narrow(2, query_start, query_len)?; // [1, 1, rows, cap]
        let scores = scores
            .reshape((b, hkv, groups, query_len, capacity))?
            .broadcast_add(&tile.unsqueeze(2)?)?
            .reshape((b, hkv, groups * query_len, capacity))?;
        let weights = softmax_last_dim(&scores)?;
        chunks.push(weights.matmul(values)?.reshape((b, h, query_len, d))?);
    }
    if chunks.len() == 1 {
        Ok(chunks.pop().expect("one gqa attention output chunk"))
    } else {
        Ok(Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?)
    }
}

/// Try the fused FlashAttention-2 kernel; `Ok(None)` means "this case is not flash-eligible, use the
/// eager path". Inputs are the same `[batch, heads, seq, head_dim]` tensors [`sdpa`] takes (K/V
/// already GQA-expanded); the kernel wants `[batch, seq, heads, head_dim]`, so q/k/v are transposed
/// in and the result transposed back. Eligibility mirrors the kernel's own constraints: f16/bf16
/// only, `head_dim ≤ 512` and a multiple of 8, equal q/k/v head dims, no soft-cap, and a causal or
/// no-op mask (an explicit additive mask is left to the eager path).
#[cfg(feature = "flash-attn")]
fn try_flash_attn(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Option<Tensor>> {
    // Soft-cap (Gemma-2) and explicit additive (padded-batch) masks are not on the fused path.
    if softcap.is_some() {
        return Ok(None);
    }
    let causal = match mask {
        AttnMask::None => false,
        AttnMask::Causal => true,
        // The wrapper exposes no left-window argument, so sliding layers take the eager path.
        AttnMask::Additive(_)
        | AttnMask::AdditiveSliding { .. }
        | AttnMask::SlidingCausal { .. } => return Ok(None),
    };
    // The kernel is CUDA-only and f16/bf16-only.
    if !queries.device().is_cuda() {
        return Ok(None);
    }
    match queries.dtype() {
        DType::F16 | DType::BF16 => {}
        _ => return Ok(None),
    }
    // Equal, kernel-supported head dims (excludes MLA's q=192 / v=128 split).
    let d = queries.dim(3)?;
    if d % 8 != 0 || d > 512 || keys.dim(3)? != d || values.dim(3)? != d {
        return Ok(None);
    }
    // [b, h, s, d] -> [b, s, h, d] (last dim stays contiguous, which the kernel requires).
    let q = queries.transpose(1, 2)?;
    let k = keys.transpose(1, 2)?;
    let v = values.transpose(1, 2)?;
    let out = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?; // [b, s, h, d]
    Ok(Some(out.transpose(1, 2)?.contiguous()?))
}

/// Batched per-sequence ("ragged") attention through a **single** [`candle_flash_attn::flash_attn_varlen`]
/// call (stories 7351 + 7453) — the continuous-batching `Throughput` decode's attention without the
/// per-sequence loop.
///
/// `q` is the batched projection `[batch, heads, s, head_dim]` (every row the same query length `s`,
/// as the decode step produces). `k_ragged`/`v_ragged` are the **already-gathered** keys/values of
/// every active sequence packed token-major into `[Σ lₖ, n_kv_heads, head_dim]` (sequence `i` owning
/// the rows `cu_k[i] .. cu_k[i+1]`), as the pooled [`BlockPool`](crate::primitives::BlockPool) gather
/// produces in **one** `index_select` — no per-sequence `squeeze`/`transpose`/`cat`. The kernel
/// handles grouped-query natively (`n_kv_heads` divides `heads`), so K/V stay **un-expanded** — no
/// [`repeat_kv`]. Queries are packed to `[Σ s, heads, head_dim]`; attention is bottom-right causal
/// (`window_size_right = 0`), the decode convention (query row `r` of a sequence attends its keys
/// `0..=(lₖ - s) + r`), with no padding mask.
///
/// `cu_k` is the cumulative key-offset table `[b + 1]` (u32-valued; `cu_k[0] == 0`,
/// `cu_k[b] == Σ lₖ`) and `max_k` the longest cached key run. Returns `Some([batch, heads, s,
/// head_dim])` when varlen is eligible — mirroring [`try_flash_attn`]: CUDA, f16/bf16, `head_dim` a
/// multiple of 8 and `≤ 512`, no soft-cap — else `Ok(None)` so the caller runs the eager per-sequence
/// fallback. Numerics differ from the eager loop by a few half-precision ULPs (different reduction
/// order), the tolerance the `Throughput` path already carries.
#[cfg(feature = "flash-attn")]
pub(crate) fn try_flash_attn_varlen(
    q: &Tensor,
    k_ragged: &Tensor,
    v_ragged: &Tensor,
    cu_k: &[u32],
    max_k: usize,
    scale: f32,
    softcap: Option<f32>,
) -> Result<Option<Tensor>> {
    // Soft-cap (Gemma-2) is not on the fused path; nor is CPU / non-half-precision.
    if softcap.is_some() || !q.device().is_cuda() {
        return Ok(None);
    }
    match q.dtype() {
        DType::F16 | DType::BF16 => {}
        _ => return Ok(None),
    }
    let (b, h, s, d) = q.dims4()?;
    if d % 8 != 0 || d > 512 {
        return Ok(None);
    }
    debug_assert_eq!(
        cu_k.len(),
        b + 1,
        "cu_k is one cumulative offset per sequence + 1"
    );
    let device = q.device();

    // Sequence-major queries: [b, heads, s, d] -> [b, s, heads, d] -> [b*s, heads, d].
    let q_ragged = q.transpose(1, 2)?.contiguous()?.reshape((b * s, h, d))?;
    // Cumulative query offsets: every sequence contributes the same `s` queries (uniform decode step).
    let cu_q: Vec<u32> = (0..=b).map(|i| (i * s) as u32).collect();
    let cu_q = Tensor::from_vec(cu_q, (b + 1,), device)?;
    let cu_k_t = Tensor::from_vec(cu_k.to_vec(), (b + 1,), device)?;

    // One kernel over all sequences; bottom-right causal (window_size_right = 0). Output: [b*s, h, d].
    let out = candle_flash_attn::flash_attn_varlen(
        &q_ragged, k_ragged, v_ragged, &cu_q, &cu_k_t, s, max_k, scale, true,
    )?;
    // Back to [b, heads, s, d] for the shared output projection.
    Ok(Some(
        out.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arange4(b: usize, h: usize, s: usize, d: usize) -> Tensor {
        let n = (b * h * s * d) as f32;
        Tensor::arange(0f32, n, &Device::Cpu)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
    }

    #[test]
    fn repeat_kv_noop_for_one_group() {
        let x = arange4(1, 2, 3, 4);
        let y = repeat_kv(&x, 1).unwrap();
        assert_eq!(y.dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn repeat_kv_expands_head_axis() {
        let x = arange4(1, 2, 2, 4);
        let y = repeat_kv(&x, 4).unwrap();
        assert_eq!(y.dims(), &[1, 8, 2, 4]);
    }

    #[test]
    fn repeat_kv_duplicates_each_head() {
        // Two KV heads, head_dim 2, seq 1: head0 = [0,1], head1 = [2,3].
        let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 2, 1, 2), &Device::Cpu).unwrap();
        let y = repeat_kv(&x, 2).unwrap(); // [1, 4, 1, 2]
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // groups are adjacent: head0,head0,head1,head1
        assert_eq!(h, vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
    }

    /// sc-24132: the folded grouped-query path agrees with `repeat_kv` + eager `sdpa` — bit-identical
    /// on the decode shape (one query, no mask), and within reduction-order tolerance on a chunked
    /// prefill / multi-token verify shape — while never materializing the expanded heads, over
    /// strided (narrowed, transposed) K/V views like the static cache hands over.
    #[test]
    fn gqa_causal_matches_repeat_kv_sdpa_without_materializing() {
        let (b, hkv, groups, d) = (2usize, 2usize, 3usize, 4usize);
        let h = hkv * groups;
        let cap = 16usize;
        // A "static buffer" [b, hkv, cap, d] whose written prefix is a narrowed view.
        let k_buf = varied4(b, hkv, cap, d, 1.7);
        let v_buf = varied4(b, hkv, cap, d, 3.1);
        for (q_len, k_len) in [(1usize, 9usize), (1, 16), (5, 9), (7, 7), (3, 16)] {
            let k = k_buf.narrow(2, 0, k_len).unwrap();
            let v = v_buf.narrow(2, 0, k_len).unwrap();
            assert!(k_len == cap || !k.is_contiguous());
            let q = varied4(b, h, q_len, d, 0.4);

            let before = crate::primitives::kv_cache::kv_materialize_count();
            let got = sdpa_gqa_causal(&q, &k, &v, 0.5).unwrap();
            assert_eq!(
                crate::primitives::kv_cache::kv_materialize_count(),
                before,
                "gqa path must not expand K/V"
            );
            assert_eq!(got.dims(), &[b, h, q_len, d]);
            assert!(got.is_contiguous());

            let want = sdpa_eager(
                &q,
                &repeat_kv(&k, groups).unwrap(),
                &repeat_kv(&v, groups).unwrap(),
                0.5,
                None,
                AttnMask::Causal,
            )
            .unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(diff <= 1e-6, "q={q_len} k={k_len}: max|delta| = {diff}");
        }
        // Chunked prefill keeps the global bottom-right alignment across tiles.
        let q_len = EAGER_ATTN_QUERY_CHUNK_SIZE + 3;
        let k_len = q_len + 2;
        let q = varied4(1, 2, q_len, 2, 0.1);
        let k = varied4(1, 1, k_len, 2, 0.2);
        let v = varied4(1, 1, k_len, 2, 0.3);
        let got = sdpa_gqa_causal(&q, &k, &v, 0.5).unwrap();
        let want = sdpa_eager(
            &q,
            &repeat_kv(&k, 2).unwrap(),
            &repeat_kv(&v, 2).unwrap(),
            0.5,
            None,
            AttnMask::Causal,
        )
        .unwrap();
        assert!(max_abs_diff(&got, &want) <= 1e-6);
        // Groups == 1 (MHA) is the plain causal path.
        let q = varied4(1, 2, 3, 4, 0.9);
        let k = varied4(1, 2, 6, 4, 0.8);
        let got = sdpa_gqa_causal(&q, &k, &k, 0.5).unwrap();
        let want = sdpa_eager(&q, &k, &k, 0.5, None, AttnMask::Causal).unwrap();
        assert!(max_abs_diff(&got, &want) <= 1e-6);
        // Mis-grouped heads and k_len < q_len are rejected.
        assert!(sdpa_gqa_causal(&varied4(1, 3, 1, 4, 0.0), &k, &k, 0.5).is_err());
        assert!(sdpa_gqa_causal(&varied4(1, 2, 8, 4, 0.0), &k, &k, 0.5).is_err());
    }

    #[test]
    fn sdpa_causal_runs_and_shapes() {
        let q = arange4(1, 1, 2, 4);
        let out = sdpa_causal(&q, &q, &q, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 1, 2, 4]);
    }

    /// Reset mask accounting so a test observes only its own builds. Tests run single-threaded here
    /// (`RUST_TEST_THREADS=1` is forced), so this is race-free.
    fn reset_mask_accounting() {
        CAUSAL_MASK_BUILDS.store(0, std::sync::atomic::Ordering::SeqCst);
        MAX_CAUSAL_MASK_ROWS.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    fn mask_builds() -> usize {
        CAUSAL_MASK_BUILDS.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Bounded, varied f32 CPU tensor (cos keeps values in [-1, 1] so the softmax is well-behaved).
    fn varied4(b: usize, h: usize, s: usize, d: usize, phase: f64) -> Tensor {
        let n = (b * h * s * d) as f32;
        Tensor::arange(0f32, n, &Device::Cpu)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
            .affine(0.013, phase)
            .unwrap()
            .cos()
            .unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// sc-12458: the decode shape (`q_len == 1`, bottom-right causal) must build **no** mask — and be
    /// bit-identical to the old always-mask path (the mask it skips is provably all zeros).
    #[test]
    fn decode_step_builds_no_mask_and_matches_masked_path() {
        let (b, h, s, d) = (2, 3, 1, 4);
        let k_len = 9;
        let q = varied4(b, h, s, d, 0.0);
        let k = varied4(b, h, k_len, d, 1.7);
        let v = varied4(b, h, k_len, d, 3.1);

        reset_mask_accounting();
        let got = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Causal).unwrap();
        assert_eq!(mask_builds(), 0, "decode step must not build a causal mask");

        // Old path: the explicitly built mask, applied via the untouched Additive branch.
        let m = causal_mask(s, k_len, DType::F32, &Device::Cpu).unwrap();
        let want = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Additive(&m)).unwrap();
        assert_eq!(bits(&got), bits(&want), "decode skip must be bit-identical");
    }

    /// Production shapes at or below the bound stay on one eager operation and therefore retain the
    /// exact small-shape output of the pre-chunk implementation.
    #[test]
    fn small_prefill_is_bit_identical_to_unchunked_eager() {
        let (b, h, q_len, d) = (1, 2, 5, 4);
        let k_len = 8; // chunked/continuation prefill: cached positions ahead of the new queries
        let q = varied4(b, h, q_len, d, 0.4);
        let k = varied4(b, h, k_len, d, 2.2);
        let v = varied4(b, h, k_len, d, 4.9);
        let want =
            sdpa_eager_with_query_chunk_size(&q, &k, &v, 0.5, None, AttnMask::Causal, usize::MAX)
                .unwrap();
        let got = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Causal).unwrap();
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn causal_prefill_never_builds_more_than_the_query_chunk() {
        let q_len = EAGER_ATTN_QUERY_CHUNK_SIZE + 1;
        let q = varied4(1, 1, q_len, 2, 0.1);
        let k = varied4(1, 1, q_len + 3, 2, 0.2);
        let v = varied4(1, 1, q_len + 3, 2, 0.3);
        reset_mask_accounting();
        let out = sdpa_eager(&q, &k, &v, 0.5, None, AttnMask::Causal).unwrap();
        assert_eq!(out.dims(), &[1, 1, q_len, 2]);
        assert_eq!(mask_builds(), 2);
        assert_eq!(
            MAX_CAUSAL_MASK_ROWS.load(std::sync::atomic::Ordering::SeqCst),
            EAGER_ATTN_QUERY_CHUNK_SIZE
        );
    }

    #[test]
    fn causal_mask_blocks_future() {
        // q_len == k_len: a plain lower-triangular mask.
        let m = causal_mask(3, 3, DType::F32, &Device::Cpu)
            .unwrap()
            .reshape((3, 3))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(m[0][1], MASK_NEG); // future blocked
        assert_eq!(m[1][0], 0.0); // past attended
        assert_eq!(m[2][2], 0.0); // self attended
    }

    #[test]
    fn chunked_eager_matches_unchunked_across_masks_softcap_and_strides() {
        let (b, h, q_len, k_len, d) = (1, 2, 7, 10, 4);
        // Transposing head/query produces the production shape with a deliberately non-contiguous
        // layout; eager attention must make each query tile contiguous before matmul.
        let q = varied4(b, q_len, h, d, 0.2).transpose(1, 2).unwrap();
        let k = varied4(b, k_len, h, d, 1.4).transpose(1, 2).unwrap();
        let v = varied4(b, k_len, h, d, 3.7).transpose(1, 2).unwrap();
        assert!(!q.is_contiguous());

        // Start from `[1, 1, k, q]` and transpose to a non-contiguous `[1, 1, q, k]` additive
        // mask, exercising query-axis narrowing without requiring the caller to copy it first.
        let additive = varied4(1, 1, k_len, q_len, 0.9).transpose(2, 3).unwrap();
        assert!(!additive.is_contiguous());

        for (name, mask, softcap) in [
            ("none", AttnMask::None, None),
            ("causal", AttnMask::Causal, None),
            ("sliding", AttnMask::SlidingCausal { window: 4 }, None),
            ("additive", AttnMask::Additive(&additive), None),
            (
                "additive_sliding",
                AttnMask::AdditiveSliding {
                    additive: &additive,
                    window: 4,
                },
                None,
            ),
            ("causal_softcap", AttnMask::Causal, Some(2.5)),
        ] {
            let want = sdpa_eager_with_query_chunk_size(&q, &k, &v, 0.5, softcap, mask, usize::MAX)
                .unwrap();
            let got = sdpa_eager_with_query_chunk_size(&q, &k, &v, 0.5, softcap, mask, 3).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(
                diff <= 1e-6,
                "{name}: chunked vs unchunked max|delta| = {diff}"
            );
        }
    }

    #[test]
    fn chunked_causal_mask_keeps_global_bottom_right_alignment() {
        let total_q = EAGER_ATTN_QUERY_CHUNK_SIZE + 1;
        let k_len = total_q + 3;
        let first = causal_mask_chunk(
            0,
            EAGER_ATTN_QUERY_CHUNK_SIZE,
            total_q,
            k_len,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
        .reshape((EAGER_ATTN_QUERY_CHUNK_SIZE, k_len))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
        let last = causal_mask_chunk(
            EAGER_ATTN_QUERY_CHUNK_SIZE,
            1,
            total_q,
            k_len,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
        .reshape((1, k_len))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
        // `offset = 3`: global row 255 sees through key 258; global row 256 sees key 259 too.
        assert_eq!(first[EAGER_ATTN_QUERY_CHUNK_SIZE - 1][258], 0.0);
        assert_eq!(first[EAGER_ATTN_QUERY_CHUNK_SIZE - 1][259], MASK_NEG);
        assert_eq!(last[0][259], 0.0);
    }

    /// On CUDA in bf16 at the Qwen3.8-27B attention shape (24 query heads over 4 KV heads, head
    /// dim 256), the folded grouped-query path must agree with `repeat_kv` + eager `sdpa` — the
    /// static-KV decode step against the reference path — on the decode shape and a short verify
    /// shape, reading the K/V through narrowed views of a larger buffer as the static cache does.
    /// Prints the max |delta| and whether the bits are identical (informational; the gate is the
    /// half-precision tolerance).
    #[cfg(feature = "cuda")]
    #[test]
    fn gqa_causal_matches_repeat_kv_sdpa_on_cuda_bf16_27b_shape() {
        let device = Device::new_cuda(0).expect("cuda device");
        let (b, h, hkv, d, cap) = (1usize, 24usize, 4usize, 256usize, 400usize);
        let groups = h / hkv;
        let mk = |b, heads, s, d, phase: f64| {
            let n = (b * heads * s * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((b, heads, s, d))
                .unwrap()
                .affine(0.0137, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let scale = (d as f32).powf(-0.5);
        let k_buf = mk(b, hkv, cap, d, 1.7);
        let v_buf = mk(b, hkv, cap, d, 3.1);
        let to_bits = |t: &Tensor| -> Vec<u16> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<half::bf16>()
                .unwrap()
                .into_iter()
                .map(half::bf16::to_bits)
                .collect()
        };
        for (q_len, k_len) in [(1usize, 97usize), (1, 353), (4, 101), (97, 97)] {
            let k = k_buf.narrow(2, 0, k_len).unwrap();
            let v = v_buf.narrow(2, 0, k_len).unwrap();
            let q = mk(b, h, q_len, d, 0.4);
            let got = sdpa_gqa_causal(&q, &k, &v, scale).unwrap();
            let want = sdpa_eager(
                &q,
                &repeat_kv(&k, groups).unwrap(),
                &repeat_kv(&v, groups).unwrap(),
                scale,
                None,
                AttnMask::Causal,
            )
            .unwrap();
            let diff = (got.to_dtype(DType::F32).unwrap() - want.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let identical = to_bits(&got) == to_bits(&want);
            eprintln!(
                "[gqa-cuda] q={q_len} k={k_len}: max|delta| = {diff}, bit-identical = {identical}"
            );
            assert!(
                diff < 3e-2,
                "q={q_len} k={k_len}: gqa vs repeat_kv max|delta| = {diff}"
            );
        }
    }

    /// **sc-24132 formulation survey (evidence, not a gate).** On CUDA bf16 at the Qwen3.8-27B
    /// decode shape, how many of 131 key lengths (90..=1000 step 7) each un-expanded GQA
    /// formulation is bit-identical to the reference `repeat_kv` + eager `sdpa` for. Recorded
    /// result on RTX Pro 6000 / sm_120 (CUDA 12.9): V0 folded, strided K (the shipped path) 52
    /// mismatching lengths; V1 folded + contiguous Kᵀ 52; V2 folded + contiguous Kᵀ and V 52; V3
    /// per-KV-head stride-0 broadcast (M = 1, batch = groups) 32; V4 V3 with contiguous Kᵀ 28;
    /// V5 V4 with contiguous V 28. Every difference is a single bf16 ULP in a few elements: cuBLAS
    /// selects its kernel (and so its reduction order) by `m`, batch count and strides, and only
    /// the reference's own calls (batch = `b × H` over expanded heads, `m = 1`) reproduce the
    /// reference's bits. Bit-exact parity with the expanded reference therefore requires the
    /// expansion itself — which is why S4 moved the reference onto the un-expanded formulation
    /// instead (`AttnFormulation`); see `docs/migration/evidence/sc-24132/README.md`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "sc-24132 formulation survey; needs CUDA"]
    fn gqa_variants_bit_match_survey() {
        let device = Device::new_cuda(0).expect("cuda device");
        let (b, h, hkv, d, cap) = (1usize, 24usize, 4usize, 256usize, 1024usize);
        let groups = h / hkv;
        let mk = |b, heads, s, d, phase: f64| {
            let n = (b * heads * s * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((b, heads, s, d))
                .unwrap()
                .affine(0.0137, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let scale = (d as f32).powf(-0.5);
        let k_buf = mk(b, hkv, cap, d, 1.7);
        let v_buf = mk(b, hkv, cap, d, 3.1);
        let q = mk(b, h, 1, d, 0.4);
        let to_bits = |t: &Tensor| -> Vec<u16> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<half::bf16>()
                .unwrap()
                .into_iter()
                .map(half::bf16::to_bits)
                .collect()
        };
        let mut mism = [0usize; 6];
        let mut tested = 0usize;
        for k_len in (90..=1000).step_by(7) {
            tested += 1;
            let k = k_buf.narrow(2, 0, k_len).unwrap();
            let v = v_buf.narrow(2, 0, k_len).unwrap();
            let want = to_bits(
                &sdpa_eager(
                    &q,
                    &repeat_kv(&k, groups).unwrap(),
                    &repeat_kv(&v, groups).unwrap(),
                    scale,
                    None,
                    AttnMask::Causal,
                )
                .unwrap(),
            );
            // V0: current folded path (OP_T strided k, strided v).
            let v0 = sdpa_gqa_causal(&q, &k, &v, scale).unwrap();
            // V1: folded, kT contiguous copy (OP_N), v strided.
            let qf = q.reshape((b, hkv, groups, d)).unwrap();
            let kt = k.transpose(2, 3).unwrap().contiguous().unwrap();
            let s1 = (qf.matmul(&kt).unwrap() * scale as f64).unwrap();
            let w1 = softmax_last_dim(&s1).unwrap();
            let v1 = w1.matmul(&v).unwrap().reshape((b, h, 1, d)).unwrap();
            // V2: folded, kT contiguous and v contiguous.
            let vc = v.contiguous().unwrap();
            let v2 = w1.matmul(&vc).unwrap().reshape((b, h, 1, d)).unwrap();
            // V3: per-kv-head, stride-0 broadcast (M=1, batch=groups), OP_T k.
            let per_head = |kt_c: bool, v_c: bool| -> Tensor {
                let mut outs = Vec::new();
                for kv in 0..hkv {
                    let qg = q
                        .narrow(1, kv * groups, groups)
                        .unwrap()
                        .squeeze(0)
                        .unwrap(); // [groups,1,d]
                    let kg = k.narrow(1, kv, 1).unwrap().squeeze(0).unwrap(); // [1,L,d]
                    let kgt = kg.transpose(1, 2).unwrap(); // [1,d,L]
                    let kgt = if kt_c { kgt.contiguous().unwrap() } else { kgt };
                    let kgt = kgt.broadcast_as((groups, d, k_len)).unwrap();
                    let s = (qg.matmul(&kgt).unwrap() * scale as f64).unwrap(); // [groups,1,L]
                    let w = softmax_last_dim(&s).unwrap();
                    let vg = v.narrow(1, kv, 1).unwrap().squeeze(0).unwrap(); // [1,L,d]
                    let vg = if v_c { vg.contiguous().unwrap() } else { vg };
                    let vg = vg.broadcast_as((groups, k_len, d)).unwrap();
                    outs.push(w.matmul(&vg).unwrap()); // [groups,1,d]
                }
                Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0)
                    .unwrap()
                    .reshape((b, h, 1, d))
                    .unwrap()
            };
            let v3 = per_head(false, false);
            let v4 = per_head(true, false);
            let v5 = per_head(true, true);
            for (i, t) in [&v0, &v1, &v2, &v3, &v4, &v5].into_iter().enumerate() {
                if to_bits(t) != want {
                    mism[i] += 1;
                }
            }
        }
        eprintln!(
            "[survey] {tested} key lengths; mismatches: V0 folded/OP_T {} | V1 folded/kT-copy {} | V2 folded/kT+v copy {} | V3 per-head bcast {} | V4 per-head kT-copy {} | V5 per-head kT+v copy {}",
            mism[0], mism[1], mism[2], mism[3], mism[4], mism[5]
        );
    }

    /// Exact attention shape from the frozen Qwen3-VL campaign `context_512` request. RC2's
    /// unchunked `[1, 32, 9247, 9247]` softmax has 2,736,224,288 elements; Candle's CUDA kernel
    /// indexes it with signed `int`, so it crosses `INT_MAX` and faults. The production query tile
    /// keeps every softmax launch below that limit. This is ignored because it requires CUDA and
    /// allocates several hundred MiB, but it uses no model weights.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA; exact Qwen3-VL context_512 attention-shape regression"]
    fn cuda_qwen3vl_context_512_shape_uses_bounded_eager_attention() {
        eprintln!("stage=device");
        let device = Device::new_cuda(0).expect("cuda device");
        let (b, h, s, d) = (1usize, 32usize, 9_247usize, 128usize);
        assert!(b * h * s * s > i32::MAX as usize);
        assert!(b * h * EAGER_ATTN_QUERY_CHUNK_SIZE * s < i32::MAX as usize);
        eprintln!("stage=allocate_qkv");
        let q = Tensor::zeros((b, h, s, d), DType::BF16, &device).unwrap();
        let k = Tensor::zeros((b, h, s, d), DType::BF16, &device).unwrap();
        let v = Tensor::zeros((b, h, s, d), DType::BF16, &device).unwrap();
        device.synchronize().unwrap();
        eprintln!("stage=qkv_ready");
        eprintln!("stage=production_sdpa");
        let out = sdpa_eager(&q, &k, &v, (d as f32).powf(-0.5), None, AttnMask::Causal).unwrap();
        assert_eq!(out.dims(), &[b, h, s, d]);
        device.synchronize().unwrap();
        eprintln!("stage=sdpa_ready");
        let sum = out.sum_all().unwrap().to_scalar::<half::bf16>().unwrap();
        eprintln!("stage=readback sum={}", sum.to_f32());
        assert_eq!(sum, half::bf16::ZERO);
    }

    /// On CUDA, the fused FlashAttention-2 kernel must agree with the eager path within a few
    /// half-precision ULPs — both for full-prompt causal attention and for the bottom-right-aligned
    /// decode shape (`q_len = 1`, `k_len > 1`). Needs `--features flash-attn` (which implies `cuda`);
    /// otherwise `sdpa` is the eager path and this would compare eager to eager, so it is gated off.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_attn_matches_eager_on_cuda() {
        let device = Device::new_cuda(0).expect("cuda device");
        // Bounded, varied bf16 q/k/v (cos keeps values in [-1, 1] so the softmax doesn't saturate).
        let mk = |b, h, s, d, phase: f64| {
            let n = (b * h * s * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((b, h, s, d))
                .unwrap()
                .affine(0.013, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let (b, h, s, d) = (2, 4, 48, 64);
        let scale = (d as f32).powf(-0.5);

        let max_abs_diff = |a: &Tensor, e: &Tensor| {
            (a.to_dtype(DType::F32).unwrap() - e.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        // Full-prompt causal.
        let (q, k, v) = (
            mk(b, h, s, d, 0.0),
            mk(b, h, s, d, 1.7),
            mk(b, h, s, d, 3.1),
        );
        assert!(
            try_flash_attn(&q, &k, &v, scale, None, AttnMask::Causal)
                .unwrap()
                .is_some(),
            "test inputs must be flash-eligible, else this proves nothing"
        );
        let diff = max_abs_diff(
            &sdpa(&q, &k, &v, scale, None, AttnMask::Causal).unwrap(),
            &sdpa_eager(&q, &k, &v, scale, None, AttnMask::Causal).unwrap(),
        );
        assert!(diff < 3e-2, "full-prompt flash vs eager max|Δ| = {diff}");

        // Decode shape: one query against the full key/value run (bottom-right causal alignment).
        let q1 = mk(b, h, 1, d, 0.0);
        let diff = max_abs_diff(
            &sdpa(&q1, &k, &v, scale, None, AttnMask::Causal).unwrap(),
            &sdpa_eager(&q1, &k, &v, scale, None, AttnMask::Causal).unwrap(),
        );
        assert!(diff < 3e-2, "decode-shape flash vs eager max|Δ| = {diff}");
    }

    /// On CUDA, the batched `flash_attn_varlen` ragged path (story 7351) must agree with the eager
    /// per-sequence SDPA loop it replaces — over **differing-length** sequences (the continuous
    /// `Throughput` decode), grouped-query (`n_kv_heads < heads`), one decode query per sequence.
    /// Needs `--features flash-attn`; otherwise `try_flash_attn_varlen` is not compiled and there is
    /// nothing to compare, so it is gated off.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_attn_varlen_matches_eager_per_seq_on_cuda() {
        let device = Device::new_cuda(0).expect("cuda device");
        let (h, kvh, d) = (4usize, 2usize, 64usize); // GQA: groups = 2
        let groups = h / kvh;
        let scale = (d as f32).powf(-0.5);
        let lens = [3usize, 7, 1, 16]; // ragged per-sequence cached lengths
        let b = lens.len();

        // Bounded, varied bf16 `[1, heads, rows, d]` (cos keeps the softmax off saturation).
        let mk = |rows: usize, heads: usize, phase: f64| {
            let n = (rows * heads * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((1, heads, rows, d))
                .unwrap()
                .affine(0.011, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };

        // One decode query per sequence: q [b, heads, 1, d].
        let qs: Vec<Tensor> = (0..b).map(|i| mk(1, h, i as f64 * 0.3)).collect();
        let q = Tensor::cat(&qs.iter().collect::<Vec<_>>(), 0).unwrap();
        // Per-sequence gathered KV at the ragged lengths (kv-head count, un-expanded).
        let kv: Vec<(Tensor, Tensor)> = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| (mk(l, kvh, 1.0 + i as f64), mk(l, kvh, 5.0 + i as f64)))
            .collect();

        // Pack the per-sequence gathers into the pooled ragged layout the BlockPool gather produces:
        // each [1, kvh, lₖ, d] -> token-major [lₖ, kvh, d], concatenated into [Σ lₖ, kvh, d] with the
        // cumulative `cu_k` offsets.
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        let mut cu_k = vec![0u32];
        let mut acc = 0u32;
        let mut max_k = 0usize;
        for (k, v) in &kv {
            let l = k.dim(2).unwrap();
            acc += l as u32;
            cu_k.push(acc);
            max_k = max_k.max(l);
            ks.push(
                k.squeeze(0)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            );
            vs.push(
                v.squeeze(0)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            );
        }
        let k_ragged = Tensor::cat(&ks.iter().collect::<Vec<_>>(), 0).unwrap();
        let v_ragged = Tensor::cat(&vs.iter().collect::<Vec<_>>(), 0).unwrap();

        let got = try_flash_attn_varlen(&q, &k_ragged, &v_ragged, &cu_k, max_k, scale, None)
            .unwrap()
            .expect("ragged inputs must be varlen-eligible, else this proves nothing");

        // Eager per-sequence reference: expand GQA and run a stock causal SDPA per sequence.
        let mut outs = Vec::with_capacity(b);
        for (i, (k, v)) in kv.iter().enumerate() {
            let qi = q.narrow(0, i, 1).unwrap();
            let k = repeat_kv(k, groups).unwrap();
            let v = repeat_kv(v, groups).unwrap();
            outs.push(sdpa_eager(&qi, &k, &v, scale, None, AttnMask::Causal).unwrap());
        }
        let want = Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0).unwrap();

        let diff = (got.to_dtype(DType::F32).unwrap() - want.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 3e-2, "varlen vs eager per-seq max|Δ| = {diff}");
    }

    /// **sc-7453 → sc-7467 cost-split bench.** Within the continuous `Throughput` per-sequence
    /// attention path, split the per-layer, per-step cost into **write_loop** (the pre-7467 `O(N)`
    /// per-sequence in-place `slice_set` of the step's new token), **write_scatter** (story 7467's
    /// **one** in-place `scatter_set` per side over the same slots — the production path), **gather**
    /// (the **single** pooled `index_select` over every active sequence's token slots), **build** (the
    /// `q`-packing + `cu_seqlens` host/device prep), and the **`flash_attn_varlen` kernel**.
    ///
    /// Two collapses are visible here. sc-7453 already replaced an `O(N · blocks)` per-sequence `cat`
    /// gather (~99% of the path — the sc-7258 finding) with one `index_select`, so **gather** is now a
    /// **flat** ~70–110 µs regardless of N. That left the per-sequence **write** as the residual
    /// launch-latency cost (`O(N)` `slice_set` launches, ~86–95% of the sync-bracketed path at N=16);
    /// sc-7467 collapses it too — **write_scatter** is one `scatter_set` per side, so it should be flat
    /// in N like the gather while **write_loop** climbs. Each phase is `synchronize()`-bracketed, so
    /// these columns are launch-latency-dominated; across the real decode's many layers the writes
    /// pipeline on the stream, and the realized end-to-end throughput is `attention_bottleneck_bound`.
    /// Run at the two real test-model head shapes across occupancy (N) and context (L). Needs
    /// `--features flash-attn` + CUDA; `#[ignore]`d (a bench, not a gate).
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "sc-7453/7467 cost-split bench; needs CUDA"]
    fn paged_attention_path_cost_on_cuda() {
        use crate::primitives::kv_cache::KvCache;
        use crate::primitives::{BlockPool, PagedKvCache};
        use std::time::Instant;

        let device = Device::new_cuda(0).expect("cuda device");
        let block_size = 16usize;
        let iters = 30usize;
        let warmup = 8usize;

        // Bounded, varied bf16 `[1, heads, rows, d]` (cos keeps values in [-1, 1]).
        let mk = |rows: usize, heads: usize, d: usize, phase: f64| {
            let n = (rows * heads * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((1, heads, rows, d))
                .unwrap()
                .affine(0.011, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let cat0 = |parts: Vec<Tensor>| Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0).unwrap();

        // (label, heads, kv_heads, head_dim) for the two real test models.
        for (label, h, kvh, d) in [("SmolLM2", 9usize, 3usize, 64usize), ("Qwen3", 16, 8, 128)] {
            let scale = (d as f32).powf(-0.5);
            println!("[{label}] H={h} KVH={kvh} D={d} (per-layer, per decode step):");
            for &n in &[4usize, 8, 16] {
                for &l in &[64usize, 256] {
                    // N single-sequence paged caches over ONE shared pool (1 layer), each prefilled to
                    // length L — so the batched gather spans every sequence's blocks in one pool tensor.
                    let pool = BlockPool::new(block_size);
                    let mut caches: Vec<PagedKvCache> = (0..n)
                        .map(|_| PagedKvCache::with_pool(pool.clone(), 1))
                        .collect();
                    for (i, c) in caches.iter_mut().enumerate() {
                        for t in 0..l {
                            let p = (i * 7 + t) as f64 * 0.01;
                            c.update(0, &mk(1, kvh, d, p), &mk(1, kvh, d, p + 3.0))
                                .unwrap();
                        }
                    }
                    // This step's batched projection: q [n, H, 1, d], new k/v [n, KVH, 1, d].
                    let q = cat0((0..n).map(|i| mk(1, h, d, i as f64 * 0.3)).collect());
                    let k_step = cat0((0..n).map(|i| mk(1, kvh, d, 9.0 + i as f64)).collect());
                    let v_step = cat0((0..n).map(|i| mk(1, kvh, d, 13.0 + i as f64)).collect());

                    let cols = kvh * d;
                    let (mut t_wloop, mut t_wscat, mut t_gather, mut t_build, mut t_kernel) =
                        (0f64, 0f64, 0f64, 0f64, 0f64);
                    for it in 0..(warmup + iters) {
                        // write_loop (pre-7467): reserve the step + O(N) per-sequence in-place slice_set.
                        device.synchronize().unwrap();
                        let t0 = Instant::now();
                        for (i, c) in caches.iter_mut().enumerate() {
                            c.reserve_step(1).unwrap();
                            let ki = k_step.narrow(0, i, 1).unwrap();
                            let vi = v_step.narrow(0, i, 1).unwrap();
                            c.write_step_layer(0, &ki, &vi).unwrap();
                        }
                        device.synchronize().unwrap();
                        let t1 = Instant::now();
                        // write_scatter (story 7467): ONE in-place scatter_set per side over the same
                        // just-reserved slots — the production path. The slot index is broadcast across
                        // the kvh*d columns the scatter preserves (built once per step in reality).
                        let mut wslots: Vec<u32> = Vec::new();
                        for c in &caches {
                            for &slot in c.new_token_slots() {
                                for _ in 0..cols {
                                    wslots.push(slot);
                                }
                            }
                        }
                        let total_new = wslots.len() / cols;
                        let w_index =
                            Tensor::from_vec(wslots, (total_new, kvh, d), &device).unwrap();
                        let k_sc = k_step
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((total_new, kvh, d))
                            .unwrap();
                        let v_sc = v_step
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((total_new, kvh, d))
                            .unwrap();
                        pool.borrow()
                            .scatter_write(0, &w_index, &k_sc, &v_sc)
                            .unwrap();
                        device.synchronize().unwrap();
                        let t1s = Instant::now();
                        // gather: ONE index_select over every sequence's token slots, concatenated.
                        let mut idx: Vec<u32> = Vec::new();
                        let mut cu_k = vec![0u32];
                        let mut acc = 0u32;
                        let mut max_k = 0usize;
                        for c in &caches {
                            let ts = c.token_slots();
                            idx.extend_from_slice(ts);
                            acc += ts.len() as u32;
                            cu_k.push(acc);
                            max_k = max_k.max(ts.len());
                        }
                        let index = Tensor::from_vec(idx, (acc as usize,), &device).unwrap();
                        let (k_rag, v_rag) = pool.borrow().gather(0, &index).unwrap();
                        device.synchronize().unwrap();
                        let t2 = Instant::now();
                        // build: pack q sequence-major + cumulative offsets.
                        let q_ragged = q
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((n, h, d))
                            .unwrap();
                        let cu_q: Vec<u32> = (0..=n).map(|i| i as u32).collect();
                        let cu_q_t = Tensor::from_vec(cu_q, (n + 1,), &device).unwrap();
                        let cu_k_t = Tensor::from_vec(cu_k, (n + 1,), &device).unwrap();
                        device.synchronize().unwrap();
                        let t3 = Instant::now();
                        // kernel: the one flash_attn_varlen call.
                        let _out = candle_flash_attn::flash_attn_varlen(
                            &q_ragged, &k_rag, &v_rag, &cu_q_t, &cu_k_t, 1, max_k, scale, true,
                        )
                        .unwrap();
                        device.synchronize().unwrap();
                        let t4 = Instant::now();
                        if it >= warmup {
                            t_wloop += (t1 - t0).as_secs_f64();
                            t_wscat += (t1s - t1).as_secs_f64();
                            t_gather += (t2 - t1s).as_secs_f64();
                            t_build += (t3 - t2).as_secs_f64();
                            t_kernel += (t4 - t3).as_secs_f64();
                        }
                    }
                    let us = |s: f64| s / iters as f64 * 1e6;
                    let (wl, ws, g, bd, kn) = (
                        us(t_wloop),
                        us(t_wscat),
                        us(t_gather),
                        us(t_build),
                        us(t_kernel),
                    );
                    println!(
                        "  N={n:<2} L={l:<4} write_loop {wl:6.1}us -> write_scatter {ws:6.1}us | \
                         gather {g:6.1}us | build {bd:6.1}us | kernel {kn:6.1}us"
                    );
                }
            }
        }
    }
}
