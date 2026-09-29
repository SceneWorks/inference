//! Attention leaves: grouped-query KV expansion and a scaled-dot-product-attention wrapper.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, hd] -> [b, hkv*groups, s, hd]`
//! head expansion the mlx-gen stacks use. [`sdpa`] wraps MLX's fused `scaled_dot_product_attention`,
//! exposing the two masking modes the references need: implicit bottom-right [`AttnMask::Causal`]
//! (decode) and an explicit [`AttnMask::Additive`] mask (the block-causal / bidirectional paths).

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, matmul, multiply, softmax_axis, subtract, which,
};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::nn::soft_cap;

/// Disallowed-attention fill for the eager additive mask: a large finite negative (matching the
/// reference slices — avoids `-inf` propagating through the softmax).
const MASK_NEG: f32 = -1e30;

/// Max query rows MLX serves with its single-pass *vector* SDPA kernel, and the floor of every
/// chunked prefill tile ([`prefill_tiles`]).
///
/// `q_len > 8` shapes MLX's fused **full** kernel does not serve (head dims other than 64/80/128,
/// mismatched q/v) are chunked so no call materializes an unbounded `[heads, q_len, k_len]` score
/// tensor: to `≤ 8` rows where the vector kernel serves them, to an unfused tile budget otherwise
/// (the tile [`prefill_attention_tile_bytes`] prices for admission).
///
/// History: sc-7430 believed the full kernel returned O(1)-wrong results above 8 rows and chunked
/// every multi-head power-of-2 prefill here (sc-7455). sc-20676 bisected it against an independent
/// f64 host reference: the kernel is correct; the full kernel writes its output as a transposed
/// view (strides `[L·H·D, D, H·D, 1]`) and the tripwire read that buffer with `as_slice`, which
/// ignores strides. Production consumers transpose/reshape (stride-aware) and were never affected.
pub(crate) const SDPA_MAX_FUSED_QLEN: i32 = 8;

/// MLX 0.32 serves `q_len ≤ 8` with its vector kernel only while `q_len · gqa_factor ≤ 32`
/// (`ScaledDotProductAttention::use_fallback`); past that it falls back to unfused attention.
const SDPA_VECTOR_MAX_QUERY_HEAD_ROWS: i32 = 32;

/// Maximum query rows left in one unevaluated attention graph on the row-chunk prefill path, and
/// the prompt length above which the resident decoder stack evaluates per layer. Shorter prefills
/// keep their lazy path.
pub(crate) const SDPA_EVAL_GROUP_QLEN: i32 = 256;

/// Query rows per fused-full-kernel call in a long prefill ([`sdpa_tiled_prefill`]), sc-20676.
///
/// Measured on an M5 Max (bf16, Hq=24, Hkv=8, D=128, causal self-prefill, one layer, 3 runs):
/// versus one unchunked fused call, 2048-row blocks with an eval per block cost +1.6 ms at 8k
/// (8.8 → 10.4 ms) and +4–27% at 32k (run-to-run noise is ±50% there); the per-block eval adds
/// ≤ 6% at 2048 rows but 13–67% at 512/1024. The old 8-row path took 208 ms / 2667 ms at 8k / 32k;
/// this path 12–20 ms / 149–153 ms. Blocking bounds one Metal dispatch to one block against its
/// causal prefix (30–40 ms at 32k keys, ~4× that at 130k) instead of a single quadratic
/// multi-second dispatch at long context — the GPU-watchdog exposure — and matches mlx-lm's
/// default 2048-token prefill step.
///
/// Precision: in f32 on NAX hardware (M5 / macOS ≥ 26.2) the full kernel runs under MLX's default
/// TF32 (`MLX_ENABLE_TF32=1`), the same policy every f32 matmul in the graph already uses: f32
/// prefill at head dims 64/80/128 matches an f64 reference to rel ≤ 1.1e-3 (≤ 3.3e-6 with TF32
/// off or without NAX), where the old 8-row vector-kernel path was ~2e-7. bf16/f16 are unchanged.
pub(crate) const SDPA_PREFILL_BLOCK_QLEN: i32 = 2048;

/// How attention should be masked.
#[derive(Debug, Clone, Copy)]
pub enum AttnMask<'a> {
    /// No mask (fully bidirectional) — e.g. a vision tower.
    None,
    /// Implicit causal mask. MLX aligns the `q_len` queries to the bottom-right of the cached
    /// keys, so query `r` attends keys `0..=offset+r` — exactly right for cached decode.
    Causal,
    /// An explicit additive mask broadcast over the score tensor (`0` keep, `-inf` block).
    Additive(&'a Array),
    /// **Sliding-window** causal mask (Gemma 4's `sliding_attention` layers): causal *and* limited to
    /// the `window` most recent keys, so query `q` sees key `j` iff `0 <= q − j < window` (the
    /// query's own position counts toward the window). Queries are bottom-right aligned over the
    /// cached keys like [`AttnMask::Causal`], so a cached decode step attends the tail of its cache.
    ///
    /// MLX's fused kernel has no sliding mode, so [`sdpa`] / [`sdpa_capped`] materialize this into an
    /// additive mask ([`sliding_causal_mask`]) before dispatching.
    SlidingCausal {
        /// Number of most-recent keys a query may attend, inclusive of its own position. A window
        /// `>= k_len` is exactly [`AttnMask::Causal`].
        window: i32,
    },
}

/// Expand grouped-query KV heads to the full query head count.
///
/// `x` is `[batch, n_kv_heads, seq, head_dim]`; the result is `[batch, n_kv_heads * groups, seq,
/// head_dim]` where `groups = n_query_heads / n_kv_heads`. `groups == 1` (MHA) is a no-op clone.
/// (sc-20676 removed a >256-token gather workaround for "broadcast-reshape corruption" that did not
/// reproduce with a stride-aware readback in f32/bf16 over contiguous, transposed and sliced inputs;
/// `repeat_kv_long_sequence_preserves_finite_values` still guards it on every backend build.)
pub fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, hkv, s, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let expanded = x.expand_dims(2)?; // [b, hkv, 1, s, hd]
    let broad = broadcast_to(&expanded, &[b, hkv, groups, s, hd])?;
    Ok(broad.reshape(&[b, hkv * groups, s, hd])?)
}

/// Scaled-dot-product attention over `[batch, heads, seq, head_dim]` tensors.
///
/// `scale` is the usual `head_dim^(-0.5)`. Grouped-query attention is handled **natively** by MLX —
/// pass K/V with `n_kv_heads` (fewer than the query heads) and the fused kernel derives
/// `gqa_factor = q_heads / kv_heads`, reading K/V by head stride with no head-count materialization.
/// (Pre-expanding with [`repeat_kv`] is equivalent but allocates a `groups`× larger K/V — avoid it
/// on the hot path; see sc-7307.)
///
/// `q_len > 8` prefill on a shape MLX's fused full kernel serves (head dims 64/80/128) runs in
/// 2048-row fused blocks; every other shape is chunked to ≤ 8 rows (`prefill_tiles`,
/// `sdpa_tiled_prefill`, sc-20676). Decode and every other shape go straight to the fused op. The
/// fused full kernel's output is a transposed view of `[b, q_len, heads, hd]` storage: read it
/// through a stride-aware op (`transpose`/`reshape`), never a raw `as_slice`.
pub fn sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let (heads, q_len, head_dim) = (queries.shape()[1], queries.shape()[2], queries.shape()[3]);
    let (kv_heads, k_len, v_head_dim) = (keys.shape()[1], keys.shape()[2], values.shape()[3]);
    match prefill_tiles(q_len, heads, kv_heads, head_dim, v_head_dim, k_len) {
        Some((rows, eval_rows)) => {
            sdpa_tiled_prefill(queries, keys, values, scale, mask, rows, eval_rows)
        }
        None => sdpa_tile(queries, keys, values, scale, mask, 0, q_len),
    }
}

/// Whether MLX 0.32's fused **full** (steel / NAX) kernel serves `q_len > 8` at these head dims
/// (`sdpa_full_supported_head_dim`). sc-20676 verified it against an f64 host reference in
/// bf16/f16/f32 (`fused_full_kernel_matches_host_across_dtypes`).
fn fused_full_kernel_serves(q_head_dim: i32, v_head_dim: i32) -> bool {
    q_head_dim == v_head_dim && matches!(q_head_dim, 64 | 80 | 128)
}

/// Whether MLX 0.32's *vector* kernel serves `≤ 8`-row chunks of this shape
/// (`sdpa_vector_supported_head_dim` and `q_len · gqa ≤ 32` with at least one row).
fn vector_kernel_serves(q_head_dim: i32, v_head_dim: i32, gqa: i32) -> bool {
    let head_dim = (q_head_dim == v_head_dim && matches!(q_head_dim, 64 | 96 | 128 | 256))
        || (q_head_dim == 192 && v_head_dim == 128);
    head_dim && gqa <= SDPA_VECTOR_MAX_QUERY_HEAD_ROWS
}

/// Budget for one unfused attention tile's transient (scores, softmax and mask copies, all heads):
/// shapes neither fused kernel serves run MLX's unfused fallback, which materializes
/// `[heads, rows, k_len]` per copy. 256 MiB keeps a SigLIP/Qwen-VL tower (16 heads, head dim 72)
/// at one call up to ~1.3k tokens and bounds a long prompt to a few-hundred-MiB tile.
const SDPA_UNFUSED_TILE_BUDGET_BYTES: u64 = 256 << 20;
/// Score-set copies an unfused tile holds at once (scores, softmax, mask), at F32 width.
const SDPA_UNFUSED_TILE_COPIES: u64 = 3;
const SDPA_UNFUSED_SCORE_BYTES: u64 = 4;

/// Rows per unfused tile: as many as fit [`SDPA_UNFUSED_TILE_BUDGET_BYTES`] against `k_len` keys,
/// never fewer than [`SDPA_MAX_FUSED_QLEN`].
fn unfused_tile_rows(heads: i32, k_len: i32) -> i32 {
    let per_row = (heads.max(1) as u64)
        * (k_len.max(1) as u64)
        * SDPA_UNFUSED_SCORE_BYTES
        * SDPA_UNFUSED_TILE_COPIES;
    let rows = SDPA_UNFUSED_TILE_BUDGET_BYTES / per_row;
    rows.clamp(SDPA_MAX_FUSED_QLEN as u64, i32::MAX as u64) as i32
}

/// The `(rows, eval_rows)` tiling [`sdpa`] prefills with, or `None` for one call.
///
/// - `q_len <= 8`: `None` — one fused call (MLX's vector kernel, correct for any cache length).
/// - A shape the fused full kernel serves ([`fused_full_kernel_serves`]):
///   [`SDPA_PREFILL_BLOCK_QLEN`]-row blocks, each evaluated once the prompt spans more than one.
/// - A shape the vector kernel serves ([`vector_kernel_serves`]): `min(8, 32 / gqa)`-row chunks,
///   evaluated per [`SDPA_EVAL_GROUP_QLEN`] rows.
/// - Anything else runs MLX's unfused fallback: one call when the whole prompt fits the unfused
///   tile budget ([`unfused_tile_rows`]), otherwise budget-sized chunks, each evaluated.
fn prefill_tiles(
    q_len: i32,
    heads: i32,
    kv_heads: i32,
    q_head_dim: i32,
    v_head_dim: i32,
    k_len: i32,
) -> Option<(i32, i32)> {
    if q_len <= SDPA_MAX_FUSED_QLEN {
        return None;
    }
    if fused_full_kernel_serves(q_head_dim, v_head_dim) {
        return Some((SDPA_PREFILL_BLOCK_QLEN, SDPA_PREFILL_BLOCK_QLEN));
    }
    let gqa = (heads / kv_heads.max(1)).max(1);
    if vector_kernel_serves(q_head_dim, v_head_dim, gqa) {
        let rows = (SDPA_VECTOR_MAX_QUERY_HEAD_ROWS / gqa).clamp(1, SDPA_MAX_FUSED_QLEN);
        return Some((rows, SDPA_EVAL_GROUP_QLEN));
    }
    let rows = unfused_tile_rows(heads, k_len);
    (rows < q_len).then_some((rows, rows))
}

/// Upper bound on the attention transient one [`sdpa`] prefill tile materializes for a
/// `prompt`-token single-sequence request (keys = prompt), derived from [`prefill_tiles`]:
///
/// - full-kernel shapes: the kernel keeps scores on chip; the only per-tile allocation that scales
///   with the prompt is an explicit mask's query slice (`rows × k_len`, `rows ≤ 2048`) — priced
///   whether or not the request passes a mask;
/// - every other shape (and `prompt ≤ 8`): a score, mask and softmax set per query head for the
///   tile's rows (`rows × k_len × heads × 3`), MLX's unfused fallback where the vector kernel
///   does not serve the head dim.
///
/// `score_element_bytes` is the priced scalar width of scores/masks (F32 for MLX's fallback).
pub(crate) fn prefill_attention_tile_bytes(
    prompt: u64,
    query_heads: u64,
    kv_heads: u64,
    head_dim: u64,
    score_element_bytes: u64,
) -> Option<u64> {
    let narrow = |v: u64| i32::try_from(v).ok();
    let (heads, kv, hd) = (narrow(query_heads)?, narrow(kv_heads)?, narrow(head_dim)?);
    let q_len = i32::try_from(prompt).unwrap_or(i32::MAX);
    match prefill_tiles(q_len, heads, kv, hd, hd, q_len) {
        Some((rows, _)) if fused_full_kernel_serves(hd, hd) => prompt
            .checked_mul(prompt.min(rows as u64))?
            .checked_mul(score_element_bytes),
        tiles => {
            let rows = tiles.map_or(prompt, |(rows, _)| prompt.min(rows as u64));
            prompt
                .checked_mul(rows)?
                .checked_mul(query_heads)?
                .checked_mul(score_element_bytes)?
                .checked_mul(3)
        }
    }
}

/// The raw fused MLX kernel, one call.
fn sdpa_fused(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let m: Option<ScaledDotProductAttentionMask> = match mask {
        AttnMask::None => None,
        AttnMask::Causal => Some(ScaledDotProductAttentionMask::Causal),
        AttnMask::Additive(a) => Some(ScaledDotProductAttentionMask::Array(a)),
        // `sdpa_tile` turns the window into a per-tile additive mask before dispatching.
        AttnMask::SlidingCausal { .. } => return Err(sliding_mask_not_materialized()),
    };
    Ok(scaled_dot_product_attention(
        queries, keys, values, scale, m, None,
    )?)
}

/// The internal invariant [`sdpa_tile`] upholds: it converts [`AttnMask::SlidingCausal`] into a
/// per-tile additive mask before dispatching, so the fused call never sees the variant. Reaching
/// it with one is a bug in this module, not bad input.
fn sliding_mask_not_materialized() -> Error {
    Error::Msg(
        "sliding-window masks must be materialized by `sdpa` before dispatch (internal invariant)"
            .into(),
    )
}

/// A contiguous `[start, end)` index vector for [`Array::take_axis`].
fn range_index(start: i32, end: i32) -> Array {
    Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start])
}

/// Attention for query rows `[c0, c1)` against exactly the keys they may see — one fused call
/// (MLX's vector, full, or unfused kernel, as the shape dictates). `[0, q_len)` is the whole
/// prompt and passes the arrays through unsliced.
///
/// - **Causal**: rows `[c0, c1)` (with `offset = k_len − q_len`) attend keys `0..(offset+c1)`;
///   slice K/V to that prefix and use the implicit causal mask, which then bottom-right-aligns the
///   tile correctly (`offset' = (offset+c1) − (c1−c0) = offset+c0`). The prefix is a strided
///   **slice view**, not a `take_axis` gather — no eager copy of the growing prefix (sc-7469).
/// - **SlidingCausal**: row `r` sits at `offset + r` and sees keys `(offset+r−window, offset+r]`;
///   slice K/V to the tile's union of those windows and build that tile's additive mask on device
///   ([`sliding_mask_tile`]) — never the full `[q_len, k_len]` mask.
/// - **None**: every query attends all keys — pass the full K/V.
/// - **Additive**: the mask already encodes visibility, so pass full K/V and slice the mask's query
///   axis to `[c0, c1)` (when that axis isn't broadcast).
fn sdpa_tile(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
    c0: i32,
    c1: i32,
) -> Result<Array> {
    let (q_len, k_len) = (queries.shape()[2], keys.shape()[2]);
    let offset = k_len - q_len; // cached prefix before the new queries
    let whole = c0 == 0 && c1 == q_len;
    let q_tile = if whole {
        queries.clone()
    } else {
        queries.try_index((.., .., c0..c1, ..))? // [·, ·, c1−c0, hd] view
    };
    let key_range = |start: i32, end: i32| -> Result<(Array, Array)> {
        if start == 0 && end == k_len {
            return Ok((keys.clone(), values.clone()));
        }
        Ok((
            keys.try_index((.., .., start..end, ..))?,
            values.try_index((.., .., start..end, ..))?,
        ))
    };
    match mask {
        AttnMask::None => sdpa_fused(&q_tile, keys, values, scale, AttnMask::None),
        AttnMask::Causal => {
            let (k, v) = key_range(0, offset + c1)?;
            sdpa_fused(&q_tile, &k, &v, scale, AttnMask::Causal)
        }
        AttnMask::SlidingCausal { window } => {
            if window <= 0 {
                return Err(Error::Msg(format!(
                    "sliding-window attention: window must be positive, got {window}"
                )));
            }
            let (first, start, end) = sliding_tile_keys(offset, c0, c1, window);
            let (k, v) = key_range(start, end)?;
            let m = sliding_mask_tile(first, c1 - c0, start, end - start, window, queries.dtype())?;
            sdpa_fused(&q_tile, &k, &v, scale, AttnMask::Additive(&m))
        }
        AttnMask::Additive(a) => {
            let q_axis = a.ndim() as i32 - 2;
            if !whole && a.shape()[q_axis as usize] == q_len {
                let a_tile = a.take_axis(range_index(c0, c1), q_axis)?;
                sdpa_fused(&q_tile, keys, values, scale, AttnMask::Additive(&a_tile))
            } else {
                sdpa_fused(&q_tile, keys, values, scale, AttnMask::Additive(a))
            }
        }
    }
}

/// For sliding-window rows `[c0, c1)` after `offset` cached keys: `(first, start, end)` — the first
/// row's absolute position and the key range `[start, end)` the tile's rows can see (the union of
/// their windows, at most `window + rows − 1` keys).
fn sliding_tile_keys(offset: i32, c0: i32, c1: i32, window: i32) -> (i32, i32, i32) {
    let first = offset + c0;
    (first, (first - window + 1).max(0), offset + c1)
}

/// The additive sliding-window mask `[1, 1, rows, keys]` for queries at absolute positions
/// `first..first+rows` over keys at `key0..key0+keys`: `0` where `0 ≤ pos − j < window`, a large
/// negative elsewhere. Built on device from two position vectors, in `dtype` (the fused kernel
/// needs a mask that promotes to the output type; f16 saturates the negative to `-inf`, as the
/// f32→f16 cast of [`sliding_causal_mask`] always did).
fn sliding_mask_tile(
    first: i32,
    rows: i32,
    key0: i32,
    keys: i32,
    window: i32,
    dtype: Dtype,
) -> Result<Array> {
    let q_pos = Array::from_slice(&(first..first + rows).collect::<Vec<i32>>(), &[rows, 1]);
    let k_pos = Array::from_slice(&(key0..key0 + keys).collect::<Vec<i32>>(), &[1, keys]);
    let causal = k_pos.le(&q_pos)?;
    let recent = k_pos.gt(&subtract(&q_pos, Array::from_int(window))?)?;
    let keep = causal.logical_and(&recent)?;
    let zero = Array::from_f32(0.0).as_dtype(dtype)?;
    let blocked = Array::from_f32(MASK_NEG).as_dtype(dtype)?;
    Ok(which(&keep, &zero, &blocked)?.reshape(&[1, 1, rows, keys])?)
}

/// Tiled prefill: split the queries into `rows`-row tiles, attend each tile against exactly the
/// keys it should, and concatenate. Mathematically identical to one attention call (gated against a
/// host reference by the `*_matches_host_*` tests). [`sdpa`] runs it with
/// the rows [`prefill_tiles`] picks for the shape.
///
/// Each tile is one [`sdpa_tile`] call (which slices keys and masks per tile).
///
/// For at most `eval_rows` query rows the outputs stay one lazy graph. Longer prefills evaluate a
/// group as soon as it spans `eval_rows` rows (at most `eval_rows + rows − 1`; `rows` need not
/// divide `eval_rows`) before constructing the next group, so at most one group's
/// attention is left unevaluated and earlier tile handles can be released. The
/// group result remains live because the next layer needs every query position. The old per-8-row
/// `eval` cost roughly 95% of a 512-token, 30-layer prefill (sc-7469). This bounds graph lifetime,
/// not MLX allocator or process peak bytes. A single tile returns the fused op's output unchanged
/// (MLX's `concatenate` of one array is that array); more tiles return a fresh row-major array.
fn sdpa_tiled_prefill(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
    rows: i32,
    eval_rows: i32,
) -> Result<Array> {
    let q_len = queries.shape()[2];
    let checkpoint = q_len > eval_rows;
    let mut outs: Vec<Array> =
        Vec::with_capacity(((q_len.min(eval_rows) + rows - 1) / rows) as usize);
    let mut groups = Vec::new();
    let (mut c0, mut group_start) = (0, 0);
    while c0 < q_len {
        let c1 = (c0 + rows).min(q_len);
        let out = sdpa_tile(queries, keys, values, scale, mask, c0, c1)?;
        outs.push(out);
        if checkpoint && (c1 - group_start >= eval_rows || c1 == q_len) {
            let refs: Vec<&Array> = outs.iter().collect();
            let group = concatenate_axis(&refs, 2)?;
            // Evaluate before dropping the tile handles. Otherwise the final concat retains the
            // full prefill's lazy attention graph despite each fused call covering one tile.
            group.eval()?;
            groups.push(group);
            outs.clear();
            group_start = c1;
        }
        c0 = c1;
    }
    let refs: Vec<&Array> = if checkpoint {
        groups.iter().collect()
    } else {
        outs.iter().collect()
    };
    Ok(concatenate_axis(&refs, 2)?)
}

/// Convenience: causal attention (the decode default).
pub fn sdpa_causal(queries: &Array, keys: &Array, values: &Array, scale: f32) -> Result<Array> {
    sdpa(queries, keys, values, scale, AttnMask::Causal)
}

/// Scaled-dot-product attention with optional Gemma-2 score soft-cap, dispatching to the fused MLX
/// kernel when it can serve the case and an explicit eager path otherwise.
///
/// The fused [`sdpa`] cannot express two things the breadth architectures need: Gemma-2's
/// attention-score soft-cap (`c·tanh(scores/c)` before the softmax), and DeepSeek-V2 MLA's mismatched
/// query/key head dim (192) vs value head dim (128). When `softcap` is `None` **and** q/v share a
/// head dim, this is exactly the fused native-GQA hot path (unchanged — `softcap.is_none()` callers
/// pay nothing). Otherwise it runs the f32-precise eager `softmax(scale·QKᵀ [+softcap] +mask)·V`, with
/// K/V GQA-expanded inside (see [`repeat_kv`]).
pub fn sdpa_capped(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let q_hd = queries.shape()[3];
    let v_hd = values.shape()[3];
    if softcap.is_none() && q_hd == v_hd {
        return sdpa(queries, keys, values, scale, mask);
    }
    sdpa_eager(queries, keys, values, scale, softcap, mask)
}

/// The eager `softmax(scale · QKᵀ [+ softcap] + mask) · V` path — the portable fallback
/// [`sdpa_capped`] runs when the fused kernel can't serve the case. Computed in f32 (the scores,
/// soft-cap, and softmax all upcast), then cast back to the queries' dtype, so bf16 decoders stay
/// numerically stable through the score soft-cap. K/V are GQA-expanded here (`groups = q_heads /
/// kv_heads`); the value head dim may differ from the query/key head dim (MLA).
fn sdpa_eager(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let out_dtype = queries.dtype();
    let groups = queries.shape()[1] / keys.shape()[1];
    let q = queries.as_dtype(Dtype::Float32)?;
    let k = repeat_kv(&keys.as_dtype(Dtype::Float32)?, groups)?;
    let v = repeat_kv(&values.as_dtype(Dtype::Float32)?, groups)?;
    let q_len = q.shape()[2];
    let k_len = k.shape()[2];

    // scores = (q @ kᵀ) * scale → [b, heads, q_len, k_len], one 4-D batched matmul. (A former 3-D
    // fold worked around "wrong 4-D matmul results" that sc-20676 could not reproduce with a
    // stride-aware readback — the same `as_slice` misread as sc-7430.)
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = matmul(&q, &kt)?;
    let mut scores = multiply(&scores, Array::from_f32(scale))?;
    if let Some(c) = softcap {
        scores = soft_cap(&scores, c)?;
    }
    let scores = match mask {
        AttnMask::None => scores,
        AttnMask::Causal => add(&scores, &causal_mask(q_len, k_len)?)?,
        AttnMask::Additive(a) => add(&scores, &a.as_dtype(Dtype::Float32)?)?,
        AttnMask::SlidingCausal { window } => {
            add(&scores, &sliding_causal_mask(q_len, k_len, window)?)?
        }
    };
    let last_axis = scores.ndim() as i32 - 1;
    let weights = softmax_axis(&scores, last_axis, None)?;
    let out = matmul(&weights, &v)?; // [b, heads, q_len, v_head_dim]
    Ok(out.as_dtype(out_dtype)?)
}

/// The additive causal mask `[1, 1, q_len, k_len]` (`0` keep / [`MASK_NEG`] block) for keys that
/// include `offset = k_len - q_len` cached positions before the new queries — bottom-right aligned, so
/// query row `r` attends keys `0..=offset+r` (matching the fused kernel's implicit causal convention).
fn causal_mask(q_len: i32, k_len: i32) -> Result<Array> {
    let offset = k_len - q_len;
    let mut data = vec![0f32; (q_len * k_len) as usize];
    for r in 0..q_len {
        for j in 0..k_len {
            if j > offset + r {
                data[(r * k_len + j) as usize] = MASK_NEG;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, q_len, k_len]))
}

/// The additive **sliding-window** causal mask `[1, 1, q_len, k_len]` (`0` keep / a large finite
/// negative to block) — Gemma 4's `sliding_attention` layers.
///
/// Queries are bottom-right aligned over the keys (`offset = k_len − q_len` cached positions come
/// first), so query row `r` sits at absolute position `offset + r` and may attend key `j` iff
/// `0 <= (offset + r) − j < window`: causal, *and* no further back than `window − 1` positions. A
/// `window >= k_len` degenerates to the plain causal mask; a `window <= 0` is rejected rather than
/// silently producing an all-blocked row (whose softmax is a uniform distribution over garbage).
pub fn sliding_causal_mask(q_len: i32, k_len: i32, window: i32) -> Result<Array> {
    if window <= 0 {
        return Err(Error::Msg(format!(
            "sliding_causal_mask: window must be positive, got {window}"
        )));
    }
    let offset = k_len - q_len;
    let mut data = vec![0f32; (q_len * k_len) as usize];
    for r in 0..q_len {
        let pos = offset + r;
        for j in 0..k_len {
            let delta = pos - j;
            if !(0..window).contains(&delta) {
                data[(r * k_len + j) as usize] = MASK_NEG;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, q_len, k_len]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_kv_noop_for_one_group() {
        let x = Array::from_slice(
            &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 2, 3, 4],
        );
        let y = repeat_kv(&x, 1).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn repeat_kv_expands_head_axis() {
        let x = Array::from_slice(
            &(0..16).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 2, 2, 4],
        );
        let y = repeat_kv(&x, 4).unwrap();
        assert_eq!(y.shape(), &[1, 8, 2, 4]);
    }

    #[test]
    fn repeat_kv_duplicates_each_head() {
        // Two KV heads, head_dim 2, seq 1: head0 = [0,1], head1 = [2,3].
        let x = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 1, 2]);
        let y = repeat_kv(&x, 2).unwrap(); // [1, 4, 1, 2]
        let h = y.as_slice::<f32>().to_vec();
        // groups are adjacent: head0,head0,head1,head1
        assert_eq!(h, vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn sdpa_causal_runs_and_shapes() {
        // [b=1, heads=1, seq=2, hd=4]
        let q = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[1, 1, 2, 4],
        );
        let out = sdpa_causal(&q, &q, &q, 0.5).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 4]);
    }

    fn randf(shape: &[i32], seed: u64) -> Array {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let mut s = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        Array::from_slice(&data, shape)
    }

    /// Host readback in logical row-major order ([`crate::primitives::nn::to_f32_host`]). MLX's
    /// fused full attention kernel returns a transposed view (strides `[L·H·D, D, H·D, 1]`), and a
    /// raw `as_slice` returns the physical buffer — the misread behind sc-7430.
    fn host_f32(a: &Array) -> Vec<f32> {
        crate::primitives::nn::to_f32_host(a).unwrap()
    }

    /// Query rows a long-prompt reference checks: both ends plus interior rows around every
    /// `boundary` (tile edges), deduplicated. Short prompts check every row.
    fn check_rows(q_len: usize, boundaries: &[usize]) -> Vec<usize> {
        if q_len <= 64 {
            return (0..q_len).collect();
        }
        let mut rows: Vec<usize> = (0..4).chain(q_len - 4..q_len).collect();
        rows.extend((1..8).map(|i| i * q_len / 8));
        for &b in boundaries.iter().filter(|&&b| b > 0 && b < q_len) {
            rows.extend([b - 1, b]);
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// f64 host GQA attention for the listed query rows only, over the dtype-rounded inputs
    /// (`host_f32` of the MLX arrays), so the comparison measures kernel error, not input rounding.
    /// Returns `(logical output index, value)` pairs. Causal aligns queries bottom-right.
    fn host_attn_rows(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        causal: bool,
        rows: &[usize],
    ) -> Vec<(usize, f32)> {
        let (qs, ks) = (q.shape(), k.shape());
        let (h, ql, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let (hkv, kl) = (ks[1] as usize, ks[2] as usize);
        let (qh, kh, vh) = (host_f32(q), host_f32(k), host_f32(v));
        let mut out = Vec::with_capacity(h * rows.len() * hd);
        for head in 0..h {
            let kb = (head / (h / hkv)) * kl * hd;
            for &i in rows {
                let jmax = if causal { kl - ql + i } else { kl - 1 };
                let qrow = &qh[(head * ql + i) * hd..][..hd];
                let logits: Vec<f64> = (0..=jmax)
                    .map(|j| {
                        let krow = &kh[kb + j * hd..][..hd];
                        let dot: f64 = qrow
                            .iter()
                            .zip(krow)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum();
                        dot * scale as f64
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f64::MIN, f64::max);
                let w: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
                let denom: f64 = w.iter().sum();
                for d in 0..hd {
                    let acc: f64 = w
                        .iter()
                        .enumerate()
                        .map(|(j, wj)| wj * vh[kb + j * hd + d] as f64)
                        .sum();
                    out.push(((head * ql + i) * hd + d, (acc / denom) as f32));
                }
            }
        }
        out
    }

    /// `max|got − ref| / max|ref|` over the reference's rows (nonfinite values fail).
    fn rel_err_rows(got: &[f32], reference: &[(usize, f32)]) -> f32 {
        assert!(
            got.iter().all(|x| x.is_finite()),
            "attention output contains nonfinite values"
        );
        let maxd = reference
            .iter()
            .map(|&(i, r)| (got[i] - r).abs())
            .fold(0.0f32, f32::max);
        let maxr = reference
            .iter()
            .map(|&(_, r)| r.abs())
            .fold(0.0f32, f32::max);
        maxd / (maxr + 1e-20)
    }

    /// Random `[1, heads, len, hd]` in `dtype`.
    fn randd(heads: i32, len: i32, hd: i32, seed: u64, dtype: Dtype) -> Array {
        randf(&[1, heads, len, hd], seed).as_dtype(dtype).unwrap()
    }

    /// Passing GQA-shaped K/V straight to `sdpa` (MLX native GQA) is numerically identical to the
    /// old `repeat_kv`-then-`sdpa` path — across decode (q_len=1), prefill (q_len>8), batched, and
    /// MHA shapes, for both the implicit-causal and no-mask paths. This is the correctness gate for
    /// dropping the per-step `repeat_kv` from the model (sc-7307).
    #[test]
    fn sdpa_native_gqa_matches_repeat_kv() {
        // (b, n_heads, n_kv_heads, seq, head_dim)
        let cases = [
            (1, 8, 2, 1, 64),   // decode, groups=4 (vector path)
            (1, 32, 8, 1, 128), // decode, Qwen3-like hd=128, groups=4
            (1, 8, 2, 16, 64),  // prefill q_len=16>8 (full/steel path), groups=4
            (2, 6, 3, 5, 64),   // batched, groups=2
            (1, 4, 4, 7, 64),   // MHA groups=1 (repeat_kv is a no-op — must be unchanged)
        ];
        for (b, nh, nkv, s, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[b, nh, s, hd], 1);
            let k = randf(&[b, nkv, s, hd], 2);
            let v = randf(&[b, nkv, s, hd], 3);
            let groups = nh / nkv;

            for mask in [AttnMask::None, AttnMask::Causal] {
                let native = sdpa(&q, &k, &v, scale, mask).unwrap();
                let expanded = sdpa(
                    &q,
                    &repeat_kv(&k, groups).unwrap(),
                    &repeat_kv(&v, groups).unwrap(),
                    scale,
                    mask,
                )
                .unwrap();
                assert_eq!(native.shape(), expanded.shape());
                let (a, e) = (host_f32(&native), host_f32(&expanded));
                let maxdiff = a
                    .iter()
                    .zip(&e)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    maxdiff < 1e-4,
                    "shape {b}/{nh}/{nkv}/{s}/{hd} mask {mask:?}: max|Δ| = {maxdiff}"
                );
            }
        }
    }

    /// From-scratch host attention `softmax(scale·QKᵀ [+causal])·V` over `[1, h, ·, hd]` GQA tensors,
    /// f64 accumulation — the ground truth the chunked-prefill mitigation is
    /// gated against. Causal aligns the `ql` queries to the bottom-right of the `kl` keys
    /// (offset = kl − ql), so it also covers a cached prefix.
    fn host_attn_gqa(
        q: &Array,
        k: &Array,
        v: &Array,
        groups: i32,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let (qs, ks) = (q.shape(), k.shape());
        let (h, ql, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let kl = ks[2] as usize;
        let (qh, kh, vh) = (
            q.as_slice::<f32>(),
            k.as_slice::<f32>(),
            v.as_slice::<f32>(),
        );
        let offset = kl as isize - ql as isize;
        let mut out = vec![0f32; h * ql * hd];
        for head in 0..h {
            let (qb, kb) = (head * ql * hd, (head / groups as usize) * kl * hd);
            for i in 0..ql {
                let jmax = if causal {
                    (offset + i as isize) as usize
                } else {
                    kl - 1
                };
                let mut logits = vec![0f64; kl];
                let mut m = f64::MIN;
                for j in 0..=jmax {
                    let dot: f64 = (0..hd)
                        .map(|d| qh[qb + i * hd + d] as f64 * kh[kb + j * hd + d] as f64)
                        .sum();
                    logits[j] = dot * scale as f64;
                    m = m.max(logits[j]);
                }
                let mut denom = 0f64;
                for lj in logits.iter_mut().take(jmax + 1) {
                    *lj = (*lj - m).exp();
                    denom += *lj;
                }
                for d in 0..hd {
                    let acc: f64 = (0..=jmax)
                        .map(|j| logits[j] / denom * vh[kb + j * hd + d] as f64)
                        .sum();
                    out[(head * ql + i) * hd + d] = acc as f32;
                }
            }
        }
        out
    }

    /// The eager path (no soft-cap, equal head dims) must match the fused kernel within a small
    /// tolerance — across decode/prefill/GQA shapes and the causal + no-mask cases. This is the
    /// correctness gate that lets Gemma-2 (which forces eager via soft-cap) and MLA (mismatched dims)
    /// trust the eager softmax against the same reference the rest of the engine uses.
    /// Host reference: per-head softmax(scale·q·kᵀ [+causal])·v over `[1, h, s, *]` MHA tensors. The
    /// value head dim (`vhd`) may differ from the q/k head dim (`hd`) — the DeepSeek MLA case.
    fn host_attn(q: &Array, k: &Array, v: &Array, scale: f32, causal: bool) -> Vec<f32> {
        let (qs, vs) = (q.shape(), v.shape());
        let (h, s, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let vhd = vs[3] as usize;
        let (qh, kh, vh) = (
            q.as_slice::<f32>(),
            k.as_slice::<f32>(),
            v.as_slice::<f32>(),
        );
        let qk_at = |t: &[f32], head: usize, i: usize, d: usize| t[(head * s + i) * hd + d];
        let v_at = |head: usize, j: usize, d: usize| vh[(head * s + j) * vhd + d];
        let mut out = vec![0f32; h * s * vhd];
        for head in 0..h {
            for i in 0..s {
                let mut logits = vec![0f32; s];
                for (j, lj) in logits.iter_mut().enumerate() {
                    let dot: f32 = (0..hd)
                        .map(|d| qk_at(qh, head, i, d) * qk_at(kh, head, j, d))
                        .sum();
                    *lj = dot * scale;
                }
                let jmax = if causal { i } else { s - 1 };
                let m = (0..=jmax).map(|j| logits[j]).fold(f32::MIN, f32::max);
                let mut denom = 0f32;
                let mut w = vec![0f32; s];
                for j in 0..=jmax {
                    w[j] = (logits[j] - m).exp();
                    denom += w[j];
                }
                for d in 0..vhd {
                    let acc: f32 = (0..=jmax).map(|j| w[j] / denom * v_at(head, j, d)).sum();
                    out[(head * s + i) * vhd + d] = acc;
                }
            }
        }
        out
    }

    fn rel_err(a: &[f32], host: &[f32]) -> f32 {
        assert!(
            a.iter().chain(host).all(|x| x.is_finite()),
            "attention comparison contains nonfinite values"
        );
        let maxd = a
            .iter()
            .zip(host)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let maxh = host.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        maxd / (maxh + 1e-20)
    }

    #[test]
    fn repeat_kv_long_sequence_preserves_finite_values() {
        // Include the Llama GQA head ratio (8 KV heads -> 24 query heads) on both sides
        // of the 256-token expansion boundary, using a small head dimension.
        for (kv_heads, groups, seq, head_dim) in [
            (2, 2, 256, 64),
            (2, 2, 257, 64),
            (2, 2, 268, 64),
            (8, 3, 257, 8),
            (8, 3, 268, 8),
        ] {
            let k = randf(&[1, kv_heads, seq, head_dim], 2);
            let expanded = repeat_kv(&k, groups).unwrap();
            let source = host_f32(&k);
            let observed = host_f32(&expanded);
            let head_len = seq as usize * head_dim as usize;
            assert_eq!(observed.len(), groups as usize * source.len());
            for head in 0..kv_heads as usize {
                for copy in 0..groups as usize {
                    let start = (head * groups as usize + copy) * head_len;
                    for offset in 0..head_len {
                        assert_eq!(
                            observed[start + offset],
                            source[head * head_len + offset],
                            "KV heads {kv_heads}, groups {groups}, sequence {seq}, head {head}, copy {copy}, offset {offset}"
                        );
                    }
                }
            }
        }
    }

    /// The vector-kernel gate (sc-7430). Fused `sdpa` is numerically correct vs a host reference for
    /// `q_len <= 8`: decode (`q_len = 1`, any cache length) and short / 8-row-chunked prefill,
    /// multi-head GQA, the power-of-2 head dims 64 & 128, both masks. The full kernel (`q_len > 8`)
    /// is gated separately by `fused_full_kernel_matches_host_across_dtypes`.
    #[test]
    fn fused_sdpa_correct_vs_host_for_short_qlen() {
        // (n_heads, n_kv_heads, q_len, k_len, head_dim)
        let cases = [
            (8, 2, 1, 64, 64),    // decode, long cache, hd=64
            (32, 8, 1, 256, 128), // decode, long cache, hd=128 (Qwen3-like)
            (8, 2, 8, 8, 64),     // short prefill q_len=8 (chunk boundary)
            (8, 2, 8, 256, 64),   // chunked prefill: 8 new queries into a 256-key cache
            (2, 1, 8, 268, 64),   // chunked prefill across the 256-key boundary
            (8, 2, 8, 8, 128),    // q_len=8 at hd=128 (chunk boundary for the Qwen3 head dim)
            (4, 4, 4, 64, 128),   // MHA, hd=128
        ];
        for (nh, nkv, ql, kl, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nkv, kl, hd], 2);
            let v = randf(&[1, nkv, kl, hd], 3);
            let groups = nh / nkv;
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let out = sdpa(&q, &k, &v, scale, mask).unwrap();
                let host = host_attn_gqa(&q, &k, &v, groups, scale, causal);
                let e = rel_err(&host_f32(&out), &host);
                assert!(e < 2e-3, "fused sdpa wrong on a SAFE shape {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}");
            }
        }
    }

    /// The eager path must match a from-scratch host attention (the ground truth) — across GQA / MHA
    /// shapes, both masks, GQA expansion, and a mismatched value head dim (MLA). This is the
    /// correctness gate for Gemma-2 (forced eager by its score soft-cap) and DeepSeek-V2 MLA, pinning
    /// the eager f32 path directly to truth.
    #[test]
    fn sdpa_eager_matches_host_reference() {
        // (b, n_heads, n_kv_heads, seq, qk_head_dim, v_head_dim)
        let cases = [
            (1, 8, 2, 16, 64, 64), // prefill, groups=4
            (1, 2, 2, 16, 64, 64), // MHA big s/hd
            (1, 4, 4, 7, 32, 32),  // MHA small
            (1, 2, 2, 5, 6, 4),    // MLA-style: q/k head_dim 6, v head_dim 4
        ];
        let maxdiff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        for (b, nh, nkv, s, hd, vhd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[b, nh, s, hd], 1);
            let k = randf(&[b, nkv, s, hd], 2);
            let v = randf(&[b, nkv, s, vhd], 3);
            let groups = nh / nkv;
            let kx = repeat_kv(&k, groups).unwrap(); // [1, nh, s, hd] for the host reference
            let vx = repeat_kv(&v, groups).unwrap();
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let href = host_attn(&q, &kx, &vx, scale, causal);
                let eager = sdpa_eager(&q, &k, &v, scale, None, mask).unwrap();
                let de = maxdiff(&href, eager.as_slice::<f32>());
                assert!(
                    de < 2e-3,
                    "eager {b}/{nh}/{nkv}/{s}/{hd}/{vhd} causal={causal}: max|Δ| vs host = {de}"
                );
            }
        }
    }

    #[test]
    fn sdpa_eager_long_gqa_matches_independent_host() {
        for q_len in [1, 8] {
            let (heads, kv_heads, k_len, head_dim) = (2, 1, 268, 64);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let q = randf(&[1, heads, q_len, head_dim], 1);
            let k = randf(&[1, kv_heads, k_len, head_dim], 2);
            let v = randf(&[1, kv_heads, k_len, head_dim], 3);
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let out = sdpa_eager(&q, &k, &v, scale, None, mask).unwrap();
                let host = host_attn_gqa(&q, &k, &v, heads / kv_heads, scale, causal);
                let error = rel_err(out.as_slice::<f32>(), &host);
                assert!(
                    error < 2e-3,
                    "eager long GQA q_len={q_len} causal={causal}: rel={error}"
                );
            }
        }
    }

    /// **The full-kernel tripwire (sc-20676).** MLX's fused full (steel / NAX) kernel — the RAW op,
    /// not the [`sdpa`] wrapper — matches an f64 host reference for every dtype, head dim, and row
    /// count [`prefill_tiles`] hands it: bf16/f16/f32 × head_dim {64, 80, 128} × q_len from 9 up to
    /// one full [`SDPA_PREFILL_BLOCK_QLEN`] block × square and cached-prefix keys × causal / none,
    /// multi-head GQA. (sc-7430's "O(1)-wrong above 8 rows" was a strided-output misread; see
    /// [`SDPA_MAX_FUSED_QLEN`].) If a future MLX bump breaks any of these, this goes red — narrow
    /// [`prefill_tiles`] back to 8-row chunks for the failing cell.
    ///
    /// Tolerances are relative to `max|ref|`: f32 2e-3 covers MLX's default TF32 on NAX hardware
    /// (measured ≤ 1.1e-3; ≤ 3e-6 with `MLX_ENABLE_TF32=0` or without NAX — the same policy every
    /// f32 matmul in the graph already runs under); bf16 1e-2 and f16 2e-3 (measured ≤ 4e-3 / 8e-4).
    #[test]
    fn fused_full_kernel_matches_host_across_dtypes() {
        let (heads, kv_heads) = (6, 2);
        for (dtype, tolerance) in [
            (Dtype::Bfloat16, 1e-2f32),
            (Dtype::Float16, 2e-3),
            (Dtype::Float32, 2e-3),
        ] {
            for head_dim in [64, 80, 128] {
                let scale = 1.0 / (head_dim as f32).sqrt();
                for (q_len, k_len) in [
                    (9, 9),
                    (16, 16),
                    (64, 64),
                    (512, 512),
                    (64, 1024),
                    (2048, 2048),
                ] {
                    let q = randd(heads, q_len, head_dim, 1, dtype);
                    let k = randd(kv_heads, k_len, head_dim, 2, dtype);
                    let v = randd(kv_heads, k_len, head_dim, 3, dtype);
                    let rows = check_rows(q_len as usize, &[]);
                    for causal in [true, false] {
                        let mask = causal.then_some(ScaledDotProductAttentionMask::Causal);
                        let raw =
                            scaled_dot_product_attention(&q, &k, &v, scale, mask, None).unwrap();
                        let reference = host_attn_rows(&q, &k, &v, scale, causal, &rows);
                        let e = rel_err_rows(&host_f32(&raw), &reference);
                        assert!(
                            e < tolerance,
                            "fused full kernel {dtype:?} hd{head_dim} q{q_len} k{k_len} causal={causal}: rel={e}"
                        );
                    }
                }
            }
        }
    }

    /// The routing rule itself: only the shapes the full-kernel tripwire covers get 2048-row blocks;
    /// vector-kernel shapes get `min(8, 32 / gqa)`-row chunks; everything else runs MLX's unfused
    /// fallback — one call while the prompt fits the tile budget, budget-sized chunks beyond it.
    #[test]
    fn prefill_tiles_routes_each_kernel_family() {
        let block = Some((SDPA_PREFILL_BLOCK_QLEN, SDPA_PREFILL_BLOCK_QLEN));
        let chunk = |rows| Some((rows, SDPA_EVAL_GROUP_QLEN));
        assert_eq!(SDPA_PREFILL_BLOCK_QLEN, 2048);
        for hd in [64, 80, 128] {
            assert_eq!(prefill_tiles(9, 24, 8, hd, hd, 9), block, "hd {hd}");
            let long = prefill_tiles(131_072, 1, 1, hd, hd, 131_072);
            assert_eq!(long, block, "single head, hd {hd}");
            let short = prefill_tiles(8, 24, 8, hd, hd, 8);
            assert_eq!(short, None, "q_len 8 is the vector kernel");
        }
        // Vector-kernel head dims: rows · gqa ≤ 32.
        for (hd, heads, kv_heads, rows) in [
            (96, 32, 32, 8),
            (256, 8, 2, 8),
            (256, 16, 2, 4),
            (256, 24, 4, 5),
            (256, 32, 1, 1),
        ] {
            let tiles = prefill_tiles(4096, heads, kv_heads, hd, hd, 4096);
            assert_eq!(tiles, chunk(rows), "hd {hd} gqa {}", heads / kv_heads);
        }
        let mla = prefill_tiles(4096, 8, 8, 192, 128, 4096);
        assert_eq!(mla, chunk(8), "MLA 192/128 is vector-served");
        // Unfused shapes (head dim 72 SigLIP / Qwen-VL towers, 112, tiny dims, gqa > 32, q ≠ v
        // outside the vector set): one call while the prompt fits the tile budget.
        let budget_rows = |heads: u64, k_len: u64| {
            (SDPA_UNFUSED_TILE_BUDGET_BYTES / (heads * k_len * 4 * 3)) as i32
        };
        assert_eq!(
            prefill_tiles(1024, 16, 16, 72, 72, 1024),
            None,
            "SigLIP 1k tokens: one call"
        );
        for hd in [8, 32, 48, 112] {
            assert_eq!(prefill_tiles(40, 8, 2, hd, hd, 40), None, "short hd {hd}");
        }
        assert_eq!(prefill_tiles(40, 64, 1, 256, 256, 40), None, "gqa 64 short");
        assert_eq!(
            prefill_tiles(40, 8, 8, 128, 64, 40),
            None,
            "q/v 128/64 short"
        );
        // ... and budget-sized, per-tile-evaluated chunks past it.
        let rows = budget_rows(16, 32_768);
        assert_eq!(rows, 42);
        let long = prefill_tiles(32_768, 16, 16, 72, 72, 32_768);
        assert_eq!(long, Some((rows, rows)), "hd72 at 32k tokens");
        let rows = budget_rows(16, 4096);
        assert_eq!(
            prefill_tiles(4096, 16, 16, 72, 72, 4096),
            Some((rows, rows))
        );
        // Never below the vector-kernel floor, even when one row exceeds the budget.
        assert_eq!(
            prefill_tiles(1 << 20, 64, 64, 72, 72, 1 << 20),
            Some((SDPA_MAX_FUSED_QLEN, SDPA_MAX_FUSED_QLEN))
        );
    }

    /// Admission prices the tile the routing actually runs: a `rows × k_len` mask slice for
    /// full-kernel blocks, a per-head score/mask/softmax set for everything else (vector chunks,
    /// budgeted unfused tiles, or the whole prompt when it fits). Derived from the routing
    /// constants, not measured bytes.
    #[test]
    fn prefill_attention_tile_bytes_follows_the_routing() {
        let (prompt, heads, kv_heads, bytes) = (131_072u64, 24u64, 8u64, 4u64);
        let block = SDPA_PREFILL_BLOCK_QLEN as u64;
        for hd in [64, 80, 128] {
            assert_eq!(
                prefill_attention_tile_bytes(prompt, heads, kv_heads, hd, bytes),
                Some(prompt * block * bytes),
                "full-kernel hd {hd}"
            );
            // A prompt shorter than one block slices only its own rows.
            let short = prefill_attention_tile_bytes(100, heads, kv_heads, hd, bytes);
            assert_eq!(short, Some(100 * 100 * bytes));
        }
        for (hd, heads, kv_heads, rows) in [(96, 32, 32, 8), (256, 16, 2, 4)] {
            assert_eq!(
                prefill_attention_tile_bytes(prompt, heads, kv_heads, hd, bytes),
                Some(prompt * rows * heads * bytes * 3),
                "vector chunks hd {hd} gqa {}",
                heads / kv_heads
            );
        }
        // Unfused hd72: the budgeted tile at long context (≤ the budget), the whole prompt when
        // it fits.
        let rows = unfused_tile_rows(16, prompt as i32) as u64;
        let long = prefill_attention_tile_bytes(prompt, 16, 16, 72, bytes).unwrap();
        assert_eq!(long, prompt * rows * 16 * bytes * 3);
        assert!(long <= SDPA_UNFUSED_TILE_BUDGET_BYTES);
        let short = prefill_attention_tile_bytes(1024, 16, 16, 72, bytes);
        assert_eq!(short, Some(1024 * 1024 * 16 * bytes * 3));
        // Decode-sized prompts: one call over `prompt` rows.
        let decode = prefill_attention_tile_bytes(5, heads, kv_heads, 128, bytes);
        assert_eq!(decode, Some(5 * 5 * heads * bytes * 3));
    }

    /// [`sdpa_tiled_prefill`]'s tile/offset/eval-group bookkeeping at small tiles (many tiles, ragged
    /// tail, groups of several tiles, a cached prefix) matches the host reference for all three mask
    /// kinds — the same code path [`sdpa`] runs at 2048-row blocks and 8-row chunks.
    #[test]
    fn sdpa_tiled_prefill_matches_host_at_small_tiles() {
        let (heads, kv_heads, q_len, k_len, hd) = (6, 2, 203, 250, 64);
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randd(heads, q_len, hd, 1, Dtype::Float32);
        let k = randd(kv_heads, k_len, hd, 2, Dtype::Float32);
        let v = randd(kv_heads, k_len, hd, 3, Dtype::Float32);
        let rows: Vec<usize> = (0..q_len as usize).collect();
        let causal_ref = host_attn_rows(&q, &k, &v, scale, true, &rows);
        let none_ref = host_attn_rows(&q, &k, &v, scale, false, &rows);
        let additive = causal_mask(q_len, k_len).unwrap();
        for (tile, eval_rows) in [(16, 64), (64, 64), (100, 100), (8, 64), (6, 64)] {
            for (mask, reference, name) in [
                (AttnMask::Causal, &causal_ref, "causal"),
                (AttnMask::None, &none_ref, "none"),
                (AttnMask::Additive(&additive), &causal_ref, "additive"),
            ] {
                let out = sdpa_tiled_prefill(&q, &k, &v, scale, mask, tile, eval_rows).unwrap();
                assert_eq!(out.shape(), &[1, heads, q_len, hd]);
                let e = rel_err_rows(&host_f32(&out), reference);
                assert!(e < 2e-3, "tile {tile}/{eval_rows} {name}: rel={e}");
            }
        }
    }

    /// Production prefill for a bf16 causal GQA prompt far past one block (two block boundaries, a
    /// ragged tail, and a cached prefix) matches the f64 host reference, including the rows either
    /// side of each block edge.
    #[test]
    fn sdpa_long_bf16_causal_gqa_prefill_matches_host() {
        let (heads, kv_heads, hd) = (6, 2, 128);
        let (q_len, k_len) = (
            2 * SDPA_PREFILL_BLOCK_QLEN + 5,
            2 * SDPA_PREFILL_BLOCK_QLEN + 37,
        );
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randd(heads, q_len, hd, 1, Dtype::Bfloat16);
        let k = randd(kv_heads, k_len, hd, 2, Dtype::Bfloat16);
        let v = randd(kv_heads, k_len, hd, 3, Dtype::Bfloat16);
        let block = SDPA_PREFILL_BLOCK_QLEN as usize;
        let rows = check_rows(q_len as usize, &[block, 2 * block]);
        let out = sdpa(&q, &k, &v, scale, AttnMask::Causal).unwrap();
        assert_eq!(out.shape(), &[1, heads, q_len, hd]);
        out.eval().unwrap();
        // Three blocks concatenate into a fresh row-major array; one unblocked fused call would
        // return the kernel's transposed `[b, q_len, heads, hd]` view. Proves the prefill was tiled.
        let (heads_u, q_len_u, hd_u) = (heads as usize, q_len as usize, hd as usize);
        assert_eq!(
            out.strides(),
            &[heads_u * q_len_u * hd_u, q_len_u * hd_u, hd_u, 1],
            "a multi-block prefill must be tiled, not one quadratic dispatch"
        );
        let reference = host_attn_rows(&q, &k, &v, scale, true, &rows);
        let e = rel_err_rows(&host_f32(&out), &reference);
        assert!(e < 1e-2, "long bf16 causal GQA prefill: rel={e}");
    }

    /// [`sdpa`] is numerically correct vs the host reference for `q_len > 8` prefill on both routes —
    /// 2048-row blocks (head dims 64/128) and 8-row chunks (head dim 256) — across GQA/MHA, causal &
    /// no-mask, square prefill and a cached-prefix offset, incl. the SmolLM2-135M prefill shape.
    #[test]
    fn sdpa_prefill_matches_host_for_long_qlen() {
        // (n_heads, n_kv_heads, q_len, k_len, head_dim)
        let cases = [
            (8, 2, 16, 16, 64),   // prefill square, GQA, hd64
            (8, 8, 64, 64, 128),  // prefill square, MHA, hd128
            (32, 8, 40, 40, 128), // Qwen3-like
            (8, 2, 16, 80, 64),   // 16 new queries into a 64-key cache (offset = 64)
            (9, 3, 26, 26, 64),   // SmolLM2-135M prefill shape
            (2, 1, 264, 268, 64), // cached keys, past the 8-row path's 256-row eval group
            (4, 2, 40, 48, 256),  // hd256: row chunks (8 rows at gqa 2), cached prefix
            (16, 2, 40, 48, 256), // hd256 at gqa 8: 4-row chunks keep rows·gqa ≤ 32
            (4, 4, 40, 40, 96),   // Phi-3's hd96: row chunks, never one unfused call
            (6, 1, 30, 30, 32),   // tiny hd32 at gqa 6: unfused, one call (fits the budget)
            (8, 2, 40, 48, 72),   // SigLIP/Qwen-VL hd72 GQA: unfused, one call, cached prefix
        ];
        for (nh, nkv, ql, kl, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nkv, kl, hd], 2);
            let v = randf(&[1, nkv, kl, hd], 3);
            let groups = nh / nkv;
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let out = sdpa(&q, &k, &v, scale, mask).unwrap();
                assert_eq!(out.shape(), &[1, nh, ql, hd]);
                let host = host_attn_gqa(&q, &k, &v, groups, scale, causal);
                let e = rel_err(&host_f32(&out), &host);
                assert!(
                    e < 2e-3,
                    "prefill sdpa wrong {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}"
                );
            }
        }
    }

    /// Prefill also handles an explicit **Additive** mask (the batched-prefill block-causal mask,
    /// shape `[b,1,q_len,k_len]`) on both routes — the 8-row route slices the mask's query axis per
    /// chunk — matching a causal host reference. Guards the `decode/batch.rs` prefill path.
    #[test]
    fn sdpa_prefill_additive_mask_matches_host() {
        for hd in [64, 256] {
            let (nh, ql) = (8, 20); // q_len=20 > 8
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nh, ql, hd], 2);
            let v = randf(&[1, nh, ql, hd], 3);
            // Additive causal mask [1,1,ql,ql]: 0 on/below the diagonal, -inf above.
            let mut md = vec![0f32; (ql * ql) as usize];
            for r in 0..ql {
                for j in 0..ql {
                    if j > r {
                        md[(r * ql + j) as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            let m = Array::from_slice(&md, &[1, 1, ql, ql]);
            let out = sdpa(&q, &k, &v, scale, AttnMask::Additive(&m)).unwrap();
            let host = host_attn_gqa(&q, &k, &v, 1, scale, true);
            let e = rel_err(&host_f32(&out), &host);
            assert!(
                e < 2e-3,
                "additive-mask prefill sdpa wrong at hd {hd}: rel={e}"
            );
        }
    }

    /// sc-20676: the batched left-padded prefill (`decode_logits_masked`, `[b, 1, q, k]` bf16
    /// additive mask) on the fused full kernel — one block, and 16-row tiles whose mask slices must
    /// keep each batch row's own padding — matches an f64 host reference per batch row.
    #[test]
    fn sdpa_bf16_batched_left_pad_additive_mask_matches_host() {
        let (batch, heads, kv_heads, len) = (2usize, 4usize, 2usize, 40usize);
        let pads = [0usize, 13];
        // Row i of batch row b sees keys j ≤ i past its padding; a padding row sees only itself.
        let visible = |b: usize, i: usize, j: usize| (j <= i && j >= pads[b]) || j == i;
        let mut mask = vec![0f32; batch * len * len];
        for b in 0..batch {
            for i in 0..len {
                for j in 0..len {
                    if !visible(b, i, j) {
                        mask[(b * len + i) * len + j] = MASK_NEG;
                    }
                }
            }
        }
        let (bi, hi, kvi, li) = (batch as i32, heads as i32, kv_heads as i32, len as i32);
        let mask = Array::from_slice(&mask, &[bi, 1, li, li])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        for hd in [64usize, 128] {
            let hdi = hd as i32;
            let q = randf(&[bi, hi, li, hdi], 1)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let k = randf(&[bi, kvi, li, hdi], 2)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let v = randf(&[bi, kvi, li, hdi], 3)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let (qh, kh, vh) = (host_f32(&q), host_f32(&k), host_f32(&v));
            let scale = 1.0 / (hd as f64).sqrt();
            let mut reference = vec![0f32; batch * heads * len * hd];
            for b in 0..batch {
                for h in 0..heads {
                    let kv = b * kv_heads + h / (heads / kv_heads);
                    for i in 0..len {
                        let qrow = &qh[((b * heads + h) * len + i) * hd..][..hd];
                        let keys: Vec<usize> = (0..len).filter(|&j| visible(b, i, j)).collect();
                        let logits: Vec<f64> = keys
                            .iter()
                            .map(|&j| {
                                let krow = &kh[(kv * len + j) * hd..][..hd];
                                let dot: f64 = qrow
                                    .iter()
                                    .zip(krow)
                                    .map(|(a, c)| *a as f64 * *c as f64)
                                    .sum();
                                dot * scale
                            })
                            .collect();
                        let m = logits.iter().cloned().fold(f64::MIN, f64::max);
                        let w: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
                        let denom: f64 = w.iter().sum();
                        for d in 0..hd {
                            let acc: f64 = keys
                                .iter()
                                .zip(&w)
                                .map(|(&j, wj)| wj * vh[(kv * len + j) * hd + d] as f64)
                                .sum();
                            reference[((b * heads + h) * len + i) * hd + d] = (acc / denom) as f32;
                        }
                    }
                }
            }
            let whole = sdpa(&q, &k, &v, scale as f32, AttnMask::Additive(&mask)).unwrap();
            let tiled =
                sdpa_tiled_prefill(&q, &k, &v, scale as f32, AttnMask::Additive(&mask), 16, 16)
                    .unwrap();
            for (name, out) in [("one block", whole), ("16-row tiles", tiled)] {
                let e = rel_err(&host_f32(&out), &reference);
                assert!(e < 1e-2, "bf16 batched left-pad hd{hd} {name}: rel={e}");
            }
        }
    }

    /// f64 host GQA attention (batch 1) under an arbitrary visibility rule `visible(row, key)`.
    fn host_attn_visible(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        rows: &[usize],
        visible: impl Fn(usize, usize) -> bool,
    ) -> Vec<(usize, f32)> {
        let (qs, ks) = (q.shape(), k.shape());
        let (h, ql, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let (hkv, kl) = (ks[1] as usize, ks[2] as usize);
        let (qh, kh, vh) = (host_f32(q), host_f32(k), host_f32(v));
        let mut out = Vec::new();
        for head in 0..h {
            let kb = (head / (h / hkv)) * kl * hd;
            for &i in rows {
                let keys: Vec<usize> = (0..kl).filter(|&j| visible(i, j)).collect();
                let qrow = &qh[(head * ql + i) * hd..][..hd];
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|&j| {
                        let krow = &kh[kb + j * hd..][..hd];
                        qrow.iter()
                            .zip(krow)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f64::MIN, f64::max);
                let w: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
                let denom: f64 = w.iter().sum();
                for d in 0..hd {
                    let acc: f64 = keys
                        .iter()
                        .zip(&w)
                        .map(|(&j, wj)| wj * vh[kb + j * hd + d] as f64)
                        .sum();
                    out.push(((head * ql + i) * hd + d, (acc / denom) as f32));
                }
            }
        }
        out
    }

    /// A sliding tile sees only the union of its rows' windows, and its device-built mask is the
    /// window rule over exactly those keys.
    #[test]
    fn sliding_tile_keys_and_mask_cover_only_the_window() {
        // 100 cached keys, rows 8..12 (positions 108..111), window 5: keys 104..=111.
        assert_eq!(sliding_tile_keys(100, 8, 12, 5), (108, 104, 112));
        assert_eq!(
            sliding_tile_keys(0, 0, 4, 16),
            (0, 0, 4),
            "clamped at the first key"
        );
        let m = sliding_mask_tile(108, 4, 104, 8, 5, Dtype::Float32).unwrap();
        assert_eq!(m.shape(), &[1, 1, 4, 8]);
        let m = host_f32(&m);
        for r in 0..4 {
            for j in 0..8 {
                let (pos, key) = (108 + r, 104 + j);
                let keep = key <= pos && pos - key < 5;
                assert_eq!(m[r * 8 + j] == 0.0, keep, "row {r} key {j}");
            }
        }
    }

    /// Sliding-window prefill builds its mask per tile on device over the tile's key window
    /// (sc-20676) — never the full `[q_len, k_len]` mask — and matches the host reference: vector,
    /// full-kernel and unfused head dims, a cached prefix, windows narrower and wider than a tile.
    #[test]
    fn sdpa_sliding_window_tiles_match_host() {
        let (q_len, k_len) = (37, 50);
        let offset = k_len - q_len;
        for (hd, heads, kv_heads) in [(64, 4, 2), (256, 4, 2), (72, 4, 2)] {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randd(heads, q_len, hd, 1, Dtype::Float32);
            let k = randd(kv_heads, k_len, hd, 2, Dtype::Float32);
            let v = randd(kv_heads, k_len, hd, 3, Dtype::Float32);
            let rows: Vec<usize> = (0..q_len as usize).collect();
            for window in [1, 5, 16, 200] {
                let visible = |i: usize, j: usize| {
                    let pos = offset as usize + i;
                    j <= pos && pos - j < window as usize
                };
                let reference = host_attn_visible(&q, &k, &v, scale, &rows, visible);
                let mask = AttnMask::SlidingCausal { window };
                let whole = sdpa(&q, &k, &v, scale, mask).unwrap();
                let tiled = sdpa_tiled_prefill(&q, &k, &v, scale, mask, 8, 16).unwrap();
                for (name, out) in [("sdpa", whole), ("8-row tiles", tiled)] {
                    let e = rel_err_rows(&host_f32(&out), &reference);
                    assert!(e < 2e-3, "sliding hd{hd} window {window} {name}: rel={e}");
                }
            }
        }
        let bad = sdpa(
            &randd(2, 16, 64, 1, Dtype::Float32),
            &randd(2, 16, 64, 2, Dtype::Float32),
            &randd(2, 16, 64, 3, Dtype::Float32),
            0.125,
            AttnMask::SlidingCausal { window: 0 },
        );
        assert!(bad.is_err(), "a non-positive window is refused");
    }

    /// Unfused shapes past the tile budget run budget-sized chunks: head dim 72 GQA (8 over 2
    /// heads) against 65 536 cached keys tiles 48 rows as 42 + 6 and matches the host reference on
    /// both sides of the tile edge. (≈ 90 MiB of F32 scores per tile; 38 MiB each for K and V.)
    #[test]
    fn sdpa_unfused_budget_tiles_match_host() {
        let (heads, kv_heads, q_len, k_len, hd) = (8, 2, 48, 65_536, 72);
        let rows = unfused_tile_rows(heads, k_len);
        assert_eq!(rows, 42);
        assert_eq!(
            prefill_tiles(q_len, heads, kv_heads, hd, hd, k_len),
            Some((rows, rows))
        );
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randd(heads, q_len, hd, 1, Dtype::Float32);
        let k = randd(kv_heads, k_len, hd, 2, Dtype::Float32);
        let v = randd(kv_heads, k_len, hd, 3, Dtype::Float32);
        let check = [0usize, 41, 42, 47];
        for (causal, mask) in [(true, AttnMask::Causal), (false, AttnMask::None)] {
            let out = sdpa(&q, &k, &v, scale, mask).unwrap();
            let reference = host_attn_rows(&q, &k, &v, scale, causal, &check);
            let e = rel_err_rows(&host_f32(&out), &reference);
            assert!(e < 2e-3, "unfused budget tiles causal={causal}: rel={e}");
        }
    }

    /// `sdpa_capped` routes a mismatched query/value head dim (DeepSeek-V2 MLA: q/k=6, v=4) through the
    /// eager path and produces finite `[b, heads, s, v_head_dim]` output.
    #[test]
    fn sdpa_capped_handles_mla_head_dims() {
        let (b, h, s, qk_hd, v_hd) = (1, 2, 3, 6, 4);
        let scale = 1.0 / (qk_hd as f32).sqrt();
        let q = randf(&[b, h, s, qk_hd], 1);
        let k = randf(&[b, h, s, qk_hd], 2);
        let v = randf(&[b, h, s, v_hd], 3);
        let out = sdpa_capped(&q, &k, &v, scale, None, AttnMask::Causal).unwrap();
        assert_eq!(out.shape(), &[b, h, s, v_hd]);
        assert!(out.as_slice::<f32>().iter().all(|x| x.is_finite()));
    }

    /// A dominant soft-cap pulls extreme scores toward `±cap`, so the attention distribution over a
    /// peaked key set is flatter than the uncapped one (the Gemma-2 effect).
    #[test]
    fn softcap_flattens_attention() {
        // One query, three keys with very different alignments → uncapped attention is peaky.
        let (b, h, s) = (1, 1, 1);
        let q = Array::from_slice(&[1.0f32, 0.0], &[b, h, s, 2]);
        let k = Array::from_slice(&[10.0f32, 0.0, 0.0, 10.0, 5.0, 5.0], &[b, h, 3, 2]);
        let v = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[b, h, 3, 2]);
        let uncapped = sdpa_capped(&q, &k, &v, 1.0, None, AttnMask::None).unwrap();
        let capped = sdpa_capped(&q, &k, &v, 1.0, Some(2.0), AttnMask::None).unwrap();
        // With a tight cap the output moves toward the mean of the values (a flatter mix).
        let mean = 3.0f32; // mean of v[:,0] = (1+3+5)/3
        let u = uncapped.as_slice::<f32>()[0];
        let c = capped.as_slice::<f32>()[0];
        assert!(
            (c - mean).abs() < (u - mean).abs(),
            "capped {c} should be nearer mean {mean} than uncapped {u}"
        );
    }
}
