//! Attention leaves: grouped-query KV expansion and a scaled-dot-product-attention wrapper.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, hd] -> [b, hkv*groups, s, hd]`
//! broadcast the mlx-gen stacks use. [`sdpa`] wraps MLX's fused `scaled_dot_product_attention`,
//! exposing the two masking modes the references need: implicit bottom-right [`AttnMask::Causal`]
//! (decode) and an explicit [`AttnMask::Additive`] mask (the block-causal / bidirectional paths).

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, matmul, multiply, softmax_axis};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::nn::soft_cap;

/// Disallowed-attention fill for the eager additive mask: a large finite negative (matching the
/// reference slices — avoids `-inf` propagating through the softmax).
const MASK_NEG: f32 = -1e30;

/// Largest `q_len` MLX's fused single-pass **vector** SDPA kernel serves (MLX 0.32
/// `ScaledDotProductAttention::use_fallback`: `query_sequence_length <= 8`). Above it MLX runs its
/// fused **full** (steel) kernel for head dims 64/80/128 and its unfused, score-materializing
/// fallback for every other head dim (which [`sdpa`] row-tiles; see `sdpa_route`).
///
/// History: sc-7430 believed the full kernel returned O(1)-wrong results above 8 rows and capped
/// every fused call here (sc-7455 chunked all multi-head power-of-2 prefill into 8-row pieces).
/// sc-24442 found the cause: the full kernel writes its `[B, H, L, D]` output as a permuted-dense
/// view over `[B, L, H, D]` storage, and the tripwire read it with mlx-rs `as_slice`, which ignores
/// strides. Production consumers transpose/reshape (stride-aware) and never saw wrong numbers. The
/// kernel matches an f64 host reference (`sc7430_fused_sdpa_matches_host_at_long_qlen`).
pub(crate) const MLX_SDPA_VECTOR_MAX_QLEN: i32 = 8;

/// Largest `q_len × gqa_factor` MLX's vector kernel serves (`(query_sequence_length * gqa_factor)
/// <= 32`); a wider GQA group falls back even at `q_len <= 8`.
const MLX_SDPA_VECTOR_MAX_ROWS: i32 = 32;

/// Query rows of attention scores the chunked admission estimate prices per attention call
/// (`core_llm::estimate_chunked_request_bytes`' `max_attention_query_tokens`).
///
/// [`sdpa`] hands MLX one fused full-kernel call (flash-style: no score matrix), vector-kernel
/// tiles of at most [`MLX_SDPA_VECTOR_MAX_QLEN`] rows (no score matrix either), or — for a masked
/// call at a shape neither kernel serves — fallback tiles of exactly this many rows, each
/// materializing one `[b, heads, 8, k_len]` score tile. So an 8-row score tile bounds every masked
/// (decoder) attention call, and the admitted workspace stays byte-identical to the pre-sc-24442
/// estimate. Unmasked ([`AttnMask::None`]) calls at an unserved head dim — the vision towers — run as
/// one fallback call and are priced quadratically by their own estimates (e.g.
/// `Qwen35Vision::estimate_workspace`).
pub(crate) const SDPA_SCORE_TILE_QLEN: i32 = MLX_SDPA_VECTOR_MAX_QLEN;

/// Query rows per fused full-kernel dispatch in a long prefill. One dispatch costs
/// `O(rows × k_len)`; blocking bounds a single Metal dispatch (and, with the per-block `eval`, a
/// single command buffer) against the GPU watchdog on long contexts instead of one quadratic
/// multi-second dispatch, and matches mlx-lm's default 2048-token prefill step. Prompts up to one
/// block — every prefill and wide verify of ordinary length — stay one lazy fused call.
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
/// Every shape runs on MLX's fused kernels where MLX has one (see `sdpa_route`): prefill and wide
/// verifies at head dims 64/80/128 are **one** fused full-kernel call (blocked at
/// `SDPA_PREFILL_BLOCK_QLEN` (2048) rows only for very long prompts); head dims only the vector kernel
/// serves (96, 256) are tiled into vector-kernel row groups instead of dropping to MLX's
/// score-materializing fallback, and masked calls at a head dim neither kernel serves run that
/// fallback in 8-row tiles (`SDPA_SCORE_TILE_QLEN`). The full kernel's output is a permuted-dense
/// view of `[b, q_len, heads, hd]` storage: read it through a stride-aware op (`transpose`/`reshape`), never a raw
/// `as_slice` (the sc-7430 misread).
pub fn sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
) -> Result<Array> {
    // MLX has no fused sliding-window mode; materialize the window into an additive mask and run the
    // ordinary explicit-mask path (which the tiled path slices correctly).
    //
    // The mask is built in f32 and cast to the query dtype: the fused kernel requires the mask type
    // to *promote* to the output type, and f32 does not promote to bf16 — a cached decode in bf16
    // (every real Gemma 4 forward) fails outright without this cast.
    if let AttnMask::SlidingCausal { window } = mask {
        let m = sliding_causal_mask(queries.shape()[2], keys.shape()[2], window)?
            .as_dtype(queries.dtype())?;
        return sdpa(queries, keys, values, scale, AttnMask::Additive(&m));
    }
    let route = sdpa_route(queries, keys, values, mask);
    #[cfg(test)]
    let route = route_override::apply(queries, route);
    match route {
        SdpaRoute::Fused => sdpa_fused(queries, keys, values, scale, mask),
        SdpaRoute::Tiled { rows, eval_tiles } => {
            sdpa_tiled(queries, keys, values, scale, mask, rows, eval_tiles)
        }
    }
}

/// How [`sdpa`] hands a call to MLX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SdpaRoute {
    /// One fused call.
    Fused,
    /// Query-row tiles of `rows`, each one fused call; `eval_tiles` evaluates each tile before the
    /// next is built (the long-prefill full-kernel blocks).
    Tiled { rows: i32, eval_tiles: bool },
}

/// Pick the fused MLX kernel for a `[b, hq, q_len, qd]` × `[b, hkv, k_len, qd]` × `[…, vd]` call,
/// mirroring MLX 0.32's `ScaledDotProductAttention::use_fallback`:
///
/// - **full** kernel: `q_len > 8`, `qd == vd ∈ {64, 80, 128}` → one fused call, or
///   [`SDPA_PREFILL_BLOCK_QLEN`]-row evaluated blocks above that length;
/// - **vector** kernel: `q_len <= 8`, `q_len <= k_len`, `q_len × gqa <= 32`, `qd == vd ∈ {64, 96,
///   128, 256}` (or MLA's 192/128) → one fused call;
/// - a vector-kernel head dim at a `q_len` / GQA width the vector kernel does not take → row tiles
///   of `min(8, 32 / gqa)` so every tile is a vector-kernel call rather than MLX's unfused fallback;
/// - a shape neither kernel serves (head dims 8/32/72/…/512, or a GQA group wider than 32 below the
///   full kernel) with a **mask** (causal / additive) and `q_len >` [`SDPA_SCORE_TILE_QLEN`] →
///   [`SDPA_SCORE_TILE_QLEN`]-row tiles, so MLX's fallback materializes one 8-row score tile at a
///   time — the bound the chunked admission estimate prices for decoders;
/// - otherwise (short queries, or an unmasked vision-tower call already priced quadratically) → one
///   call; MLX runs its fallback.
fn sdpa_route(queries: &Array, keys: &Array, values: &Array, mask: AttnMask<'_>) -> SdpaRoute {
    let (hq, q_len, qd) = (queries.shape()[1], queries.shape()[2], queries.shape()[3]);
    let (hkv, k_len, vd) = (keys.shape()[1], keys.shape()[2], values.shape()[3]);
    let full_head_dim = qd == vd && matches!(qd, 64 | 80 | 128);
    if q_len > MLX_SDPA_VECTOR_MAX_QLEN && full_head_dim {
        return if q_len > SDPA_PREFILL_BLOCK_QLEN {
            SdpaRoute::Tiled {
                rows: SDPA_PREFILL_BLOCK_QLEN,
                eval_tiles: true,
            }
        } else {
            SdpaRoute::Fused
        };
    }
    if !vector_kernel_serves(hq, hkv, qd, vd) {
        let masked = !matches!(mask, AttnMask::None);
        return if masked && q_len > SDPA_SCORE_TILE_QLEN {
            SdpaRoute::Tiled {
                rows: SDPA_SCORE_TILE_QLEN,
                eval_tiles: false,
            }
        } else {
            SdpaRoute::Fused
        };
    }
    let rows = vector_verify_rows(hq, hkv, qd, vd);
    if q_len <= rows && q_len <= k_len {
        SdpaRoute::Fused
    } else {
        SdpaRoute::Tiled {
            rows,
            eval_tiles: false,
        }
    }
}

/// `gqa_factor = q_heads / kv_heads` as MLX derives it.
fn gqa_factor(hq: i32, hkv: i32) -> i32 {
    (hq / hkv.max(1)).max(1)
}

/// Whether MLX 0.32's single-pass **vector** SDPA kernel serves this head geometry at some
/// `q_len <= 8`: `qd == vd ∈ {64, 96, 128, 256}` (or MLA's 192/128) and a GQA group of at most 32.
fn vector_kernel_serves(hq: i32, hkv: i32, qd: i32, vd: i32) -> bool {
    let vector_head_dim =
        (qd == vd && matches!(qd, 64 | 96 | 128 | 256)) || (qd == 192 && vd == 128);
    vector_head_dim && gqa_factor(hq, hkv) <= MLX_SDPA_VECTOR_MAX_ROWS
}

/// The most query rows [`sdpa`] hands MLX as **one** decode-kernel call for a `[b, hq, q_len, qd]`
/// × `[b, hkv, k_len, qd]` × `[…, vd]` attention with `q_len <= k_len` — the widest speculative
/// verify (`1 + depth` rows) that costs one fused call per attention layer, like the plain
/// `q_len = 1` step it must agree with (sc-24438).
///
/// Where the vector kernel serves the head geometry it is `min(8, 32 / gqa)` — the rows
/// `sdpa_route` tiles a wider call into, so a GQA group wider than 4 narrows it (gqa 6 → 5, gqa 8
/// → 4, gqa 16 → 2). Where it does not (a head dim only the full kernel or MLX's fallback serves,
/// or a GQA group wider than 32), a masked call of up to [`SDPA_SCORE_TILE_QLEN`] (8) rows is one
/// call, so the bound is 8. Always `>= 1`.
pub(crate) fn vector_verify_rows(hq: i32, hkv: i32, qd: i32, vd: i32) -> i32 {
    if vector_kernel_serves(hq, hkv, qd, vd) {
        MLX_SDPA_VECTOR_MAX_QLEN.min(MLX_SDPA_VECTOR_MAX_ROWS / gqa_factor(hq, hkv))
    } else {
        SDPA_SCORE_TILE_QLEN
    }
}

/// Test-only routing override: replay the **pre-sc-24442** `sdpa` routing so a test can compare a
/// whole decoder's greedy output under the old and new dispatch in one process (no golden files).
///
/// Before sc-24442, `sdpa` sent `q_len > 8` × multi-head × power-of-2 `head_dim >= 64` calls through
/// 8-row tiles (the sc-7455 mitigation) and every other shape through one fused call.
#[cfg(test)]
pub(crate) mod route_override {
    use std::cell::Cell;

    use super::SdpaRoute;
    use mlx_rs::Array;

    thread_local! {
        static PRE_SC24442: Cell<bool> = const { Cell::new(false) };
        static DIFFERING_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    /// The route the pre-sc-24442 `sdpa` took for these queries.
    fn pre_sc24442_route(queries: &Array) -> SdpaRoute {
        let (heads, q_len, head_dim) = (queries.shape()[1], queries.shape()[2], queries.shape()[3]);
        if q_len > 8 && heads >= 2 && head_dim >= 64 && (head_dim as u32).is_power_of_two() {
            SdpaRoute::Tiled {
                rows: 8,
                eval_tiles: false,
            }
        } else {
            SdpaRoute::Fused
        }
    }

    /// Replace `route` with the pre-sc-24442 one inside [`with_pre_sc24442_routing`]; otherwise
    /// count the calls whose route differs from it (so a comparison can prove it exercised a change).
    pub(super) fn apply(queries: &Array, route: SdpaRoute) -> SdpaRoute {
        let old = pre_sc24442_route(queries);
        if PRE_SC24442.with(Cell::get) {
            return old;
        }
        if old != route {
            DIFFERING_CALLS.with(|c| c.set(c.get() + 1));
        }
        route
    }

    /// Run `f` with `sdpa` routed as before sc-24442 (on this thread).
    pub(crate) fn with_pre_sc24442_routing<R>(f: impl FnOnce() -> R) -> R {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                PRE_SC24442.with(|p| p.set(false));
            }
        }
        PRE_SC24442.with(|p| p.set(true));
        let _reset = Reset;
        f()
    }

    /// Calls (on this thread) whose current route differs from the pre-sc-24442 one; resets the
    /// count.
    fn take_differing_calls() -> usize {
        DIFFERING_CALLS.with(|c| c.replace(0))
    }

    /// Largest last-position logit drift the routing change may cause. The decoders compute in
    /// bf16, whose logits (|logit| ~1–3 on the fixtures) carry an ulp of 0.008–0.016; a different
    /// kernel tiling reassociates the attention sums and moves a logit by about one ulp (measured
    /// max 0.0156). A wrong attention (a mis-sliced tile, a lost mask) moves them by far more.
    const LOGIT_DRIFT_TOL: f32 = 0.03125;

    /// What one fixture comparison saw; see [`assert_greedy_matches_pre_sc24442`].
    #[derive(Debug, Default, Clone, Copy)]
    pub(crate) struct GreedyComparison {
        /// Current-run `sdpa` calls whose route differs from the pre-sc-24442 one.
        pub differing_calls: usize,
        /// Steps whose greedy token was compared (reference top-2 margin above `2 ×` the drift
        /// bound, so no in-tolerance drift can flip it).
        pub compared_steps: usize,
        /// Steps skipped as near-ties: the reference's own top-2 margin is within the drift bound,
        /// so either token is a correct greedy pick for a bf16 decoder.
        pub tie_steps: usize,
    }

    impl std::ops::AddAssign for GreedyComparison {
        fn add_assign(&mut self, o: Self) {
            self.differing_calls += o.differing_calls;
            self.compared_steps += o.compared_steps;
            self.tie_steps += o.tie_steps;
        }
    }

    /// Greedy-decode `steps` tokens (the first from the `prompt` prefill, the rest from cached
    /// single-token steps) under the pre-sc-24442 routing, then replay the same token sequence under
    /// the current routing with a fresh cache. At every step the last-position logits must agree
    /// within [`LOGIT_DRIFT_TOL`], and the current routing must pick the reference's greedy token
    /// unless the reference itself is a near-tie (top-2 margin `<= 2 × LOGIT_DRIFT_TOL`). With the
    /// inputs replayed, agreement at every step is exactly "the same greedy tokens" for a
    /// free-running decode. `decode(ids, cache, offset)` returns the last position's logits.
    pub(crate) fn assert_greedy_matches_pre_sc24442<C>(
        label: &str,
        prompt: &[i32],
        steps: usize,
        new_cache: impl Fn() -> C,
        decode: impl Fn(&Array, &mut C, i32) -> Array,
    ) -> GreedyComparison {
        let argmax = |l: &[f32]| {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as i32)
                .unwrap()
        };
        // `replay`: feed these tokens instead of the run's own greedy picks.
        let run = |replay: Option<&[i32]>| {
            let mut cache = new_cache();
            let (mut ids, mut offset) = (prompt.to_vec(), 0);
            let mut logits = Vec::new();
            for step in 0..steps {
                let input = Array::from_slice(&ids, &[1, ids.len() as i32]);
                let out = decode(&input, &mut cache, offset)
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .unwrap()
                    .reshape(&[-1])
                    .unwrap();
                let host = out.as_slice::<f32>().to_vec();
                offset += ids.len() as i32;
                ids = vec![replay.map_or_else(|| argmax(&host), |r| r[step])];
                logits.push(host);
            }
            logits
        };
        take_differing_calls();
        let old_logits = with_pre_sc24442_routing(|| run(None));
        let old_tokens: Vec<i32> = old_logits.iter().map(|l| argmax(l)).collect();
        let new_logits = run(Some(&old_tokens));
        let mut seen = GreedyComparison {
            differing_calls: take_differing_calls(),
            ..Default::default()
        };

        for (step, (old, new)) in old_logits.iter().zip(&new_logits).enumerate() {
            let mut sorted = old.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            if sorted[0] - sorted[1] <= 2.0 * LOGIT_DRIFT_TOL {
                seen.tie_steps += 1;
            } else {
                seen.compared_steps += 1;
                assert_eq!(
                    argmax(new),
                    old_tokens[step],
                    "{label}: step {step} greedy token changed with the sdpa routing"
                );
            }
            let drift = old
                .iter()
                .zip(new)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                drift <= LOGIT_DRIFT_TOL,
                "{label}: step {step} logits drifted {drift} with the sdpa routing"
            );
        }
        seen
    }
}

/// The raw fused MLX op, one call.
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
        // `sdpa` materializes the window before dispatching; nothing else reaches the fused kernel.
        AttnMask::SlidingCausal { .. } => return Err(sliding_mask_not_materialized()),
    };
    Ok(scaled_dot_product_attention(
        queries, keys, values, scale, m, None,
    )?)
}

/// The internal invariant [`sdpa`] upholds: it converts [`AttnMask::SlidingCausal`] into an explicit
/// additive mask before dispatching, so neither the fused kernel nor the tiled path ever
/// sees the variant. Reaching either with it is a bug in this module, not bad input.
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

/// Tiled attention: split the queries into `rows`-row tiles, attend each tile against exactly the
/// keys it should, and concatenate. Mathematically identical to one attention call (gated against
/// a host reference by the `*_matches_host*` tests).
///
/// - **Causal**: tile rows `[c0, c1)` (with `offset = k_len − q_len`) attend keys `0..(offset+c1)`;
///   slice K/V to that prefix and use the implicit causal mask, which then bottom-right-aligns the
///   tile correctly (`offset' = (offset+c1) − (c1−c0) = offset+c0`). The prefix is a strided
///   **slice view** (`index`, `mlx_slice`), not a `take_axis` gather — no index-vector alloc and no
///   eager copy of the growing prefix per tile (sc-7469).
/// - **None**: every query attends all keys — pass the full K/V.
/// - **Additive**: the mask already encodes visibility, so pass full K/V and slice the mask's query
///   axis to `[c0, c1)` (when that axis isn't broadcast).
///
/// Vector-kernel and fallback tiles stay **lazy** and concatenate into one graph for the caller's
/// `eval` (sc-7469 measured a per-tile `eval` at ~95% of prefill time). `eval_tiles` evaluates each
/// full-kernel prefill block before building the next, so no single command buffer carries more
/// than one block's quadratic work (see [`SDPA_PREFILL_BLOCK_QLEN`]).
fn sdpa_tiled(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
    rows: i32,
    eval_tiles: bool,
) -> Result<Array> {
    let q_len = queries.shape()[2];
    let k_len = keys.shape()[2];
    let offset = k_len - q_len; // cached prefix before the new queries; ≥ 0 for cached decode/prefill

    let mut outs: Vec<Array> = Vec::with_capacity(((q_len + rows - 1) / rows) as usize);
    let mut c0 = 0;
    while c0 < q_len {
        let c1 = (c0 + rows).min(q_len);
        let q_chunk = queries.try_index((.., .., c0..c1, ..))?; // [·, ·, c1−c0, hd] view
        let out = match mask {
            AttnMask::None => sdpa_fused(&q_chunk, keys, values, scale, AttnMask::None)?,
            AttnMask::Causal => {
                let end = offset + c1;
                let k_chunk = keys.try_index((.., .., 0..end, ..))?; // [·, ·, end, hd] prefix view
                let v_chunk = values.try_index((.., .., 0..end, ..))?;
                sdpa_fused(&q_chunk, &k_chunk, &v_chunk, scale, AttnMask::Causal)?
            }
            AttnMask::Additive(a) => {
                let q_axis = a.ndim() as i32 - 2;
                if a.shape()[q_axis as usize] == q_len {
                    let a_chunk = a.take_axis(range_index(c0, c1), q_axis)?;
                    sdpa_fused(&q_chunk, keys, values, scale, AttnMask::Additive(&a_chunk))?
                } else {
                    sdpa_fused(&q_chunk, keys, values, scale, AttnMask::Additive(a))?
                }
            }
            // `sdpa` materializes the window into `Additive` before it ever reaches here.
            AttnMask::SlidingCausal { .. } => return Err(sliding_mask_not_materialized()),
        };
        if eval_tiles {
            out.eval()?;
        }
        outs.push(out);
        c0 = c1;
    }
    let refs: Vec<&Array> = outs.iter().collect();
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
    let (b, nh) = (q.shape()[0], q.shape()[1]);
    let q_len = q.shape()[2];
    let k_len = k.shape()[2];

    // scores = (q @ kᵀ) * scale → [b, heads, q_len, k_len]. MLX's batched 4-D `matmul` misreads the
    // `[b, heads]` leading dims on this fork (wrong results for multi-head/large shapes — the fused
    // SDPA avoids raw matmul), so fold `[b, heads]` into one batch axis and run 3-D batched matmuls.
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = bmm(&q, &kt, b * nh)?; // [b, heads, q_len, k_len]
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
    let out = bmm(&weights, &v, b * nh)?; // [b, heads, q_len, v_head_dim]
    Ok(out.as_dtype(out_dtype)?)
}

/// Batched matrix multiply `a @ b` over 4-D `[lead0, lead1, m, k] @ [lead0, lead1, k, n]` tensors,
/// run as a **3-D** batched matmul over a single folded batch axis (`batch = lead0 · lead1`). MLX's
/// 4-D batched `matmul` returns wrong results on this fork for multi-batch/large shapes; folding to
/// 3-D sidesteps it. Reshape materializes any transposed/broadcast operand into row-major storage.
fn bmm(a: &Array, b: &Array, batch: i32) -> Result<Array> {
    let sa = a.shape();
    let sb = b.shape();
    let (lead0, lead1, m, k) = (sa[0], sa[1], sa[2], sa[3]);
    let n = sb[3];
    let a3 = a.reshape(&[batch, m, k])?;
    let b3 = b.reshape(&[batch, k, n])?;
    Ok(matmul(&a3, &b3)?.reshape(&[lead0, lead1, m, n])?)
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
            (1, 8, 2, 16, 64),  // prefill q_len=16>8 (fused full kernel), groups=4
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

    /// From-scratch host attention `softmax(scale·QKᵀ [+causal])·V` over `[1, h, ·, hd]` GQA tensors
    /// (kv expanded to `h`), f64 accumulation — the ground truth the chunked-prefill mitigation is
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
        let kx = repeat_kv(k, groups).unwrap();
        let vx = repeat_kv(v, groups).unwrap();
        let (qs, ks) = (q.shape(), kx.shape());
        let (h, ql, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let kl = ks[2] as usize;
        // Stride-aware: an MQA `repeat_kv` (one KV head) is a zero-stride broadcast view, which a
        // raw `as_slice` would read out of bounds.
        let (qh, kh, vh) = (host_f32(q), host_f32(&kx), host_f32(&vx));
        let offset = kl as isize - ql as isize;
        let mut out = vec![0f32; h * ql * hd];
        for head in 0..h {
            let (qb, kb) = (head * ql * hd, head * kl * hd);
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
        let (qh, kh, vh) = (host_f32(q), host_f32(k), host_f32(v));
        let qk_at = |t: &[f32], head: usize, i: usize, d: usize| t[(head * s + i) * hd + d];
        let v_at = |head: usize, j: usize, d: usize| vh[(head * s + j) * vhd + d];
        let mut out = vec![0f32; h * s * vhd];
        for head in 0..h {
            for i in 0..s {
                let mut logits = vec![0f32; s];
                for (j, lj) in logits.iter_mut().enumerate() {
                    let dot: f32 = (0..hd)
                        .map(|d| qk_at(&qh, head, i, d) * qk_at(&kh, head, j, d))
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
        let maxd = a
            .iter()
            .zip(host)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let maxh = host.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        maxd / (maxh + 1e-20)
    }

    /// The vector-kernel gate: fused `sdpa` is numerically correct vs a host reference for
    /// `q_len <= 8` — decode (`q_len = 1`, any cache length) and short prefill / verify, multi-head
    /// GQA, head dims 64 & 128, both masks. `q_len > 8` is gated by
    /// `sc7430_fused_sdpa_matches_host_at_long_qlen`.
    #[test]
    fn fused_sdpa_correct_vs_host_for_short_qlen() {
        // (n_heads, n_kv_heads, q_len, k_len, head_dim)
        let cases = [
            (8, 2, 1, 64, 64),    // decode, long cache, hd=64
            (32, 8, 1, 256, 128), // decode, long cache, hd=128 (Qwen3-like)
            (8, 2, 8, 8, 64),     // short prefill q_len=8 (vector-kernel boundary)
            (8, 2, 8, 256, 64),   // 8 new queries into a 256-key cache
            (8, 2, 8, 8, 128),    // q_len=8 at hd=128
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
                assert!(
                    e < 2e-3,
                    "fused sdpa wrong {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}"
                );
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
                let de = maxdiff(&href, &host_f32(&eager));
                assert!(
                    de < 2e-3,
                    "eager {b}/{nh}/{nkv}/{s}/{hd}/{vhd} causal={causal}: max|Δ| vs host = {de}"
                );
            }
        }
    }

    /// [`sdpa_route`] picks MLX's fused kernel for every shape one serves: the full kernel for any
    /// `q_len > 8` at head dims 64/80/128 (one call up to a block, evaluated blocks beyond it), the
    /// vector kernel at `q_len <= 8`, vector-kernel row tiles where only that kernel serves the head
    /// dim or the GQA group is too wide for the whole query length. A shape neither kernel serves
    /// runs MLX's fallback in 8-row tiles when masked (the decoders' admission bound) and as one call
    /// when unmasked (vision towers) or at `q_len <= 8`.
    #[test]
    fn sdpa_route_prefers_one_fused_call() {
        let route_masked = |mask: AttnMask<'_>, hq: i32, hkv: i32, ql: i32, kl: i32, qd, vd| {
            sdpa_route(
                &Array::zeros::<f32>(&[1, hq, ql, qd]).unwrap(),
                &Array::zeros::<f32>(&[1, hkv, kl, qd]).unwrap(),
                &Array::zeros::<f32>(&[1, hkv, kl, vd]).unwrap(),
                mask,
            )
        };
        let route =
            |hq, hkv, ql, kl, qd, vd| route_masked(AttnMask::Causal, hq, hkv, ql, kl, qd, vd);
        let unmasked =
            |hq, hkv, ql, kl, qd, vd| route_masked(AttnMask::None, hq, hkv, ql, kl, qd, vd);
        let tiled = |rows, eval_tiles| SdpaRoute::Tiled { rows, eval_tiles };
        for hd in [64, 80, 128] {
            for ql in [9, 16, 64, 512, SDPA_PREFILL_BLOCK_QLEN] {
                assert_eq!(
                    route(32, 8, ql, ql, hd, hd),
                    SdpaRoute::Fused,
                    "hd {hd} q {ql}"
                );
            }
            assert_eq!(
                route(
                    8,
                    2,
                    SDPA_PREFILL_BLOCK_QLEN + 1,
                    SDPA_PREFILL_BLOCK_QLEN + 1,
                    hd,
                    hd
                ),
                tiled(SDPA_PREFILL_BLOCK_QLEN, true)
            );
        }
        assert_eq!(route(32, 8, 1, 4096, 128, 128), SdpaRoute::Fused); // decode
        assert_eq!(route(8, 2, 8, 64, 64, 64), SdpaRoute::Fused); // 8 × gqa 4 = 32
        assert_eq!(route(32, 4, 8, 64, 128, 128), tiled(4, false)); // 8 × gqa 8 > 32
        assert_eq!(route(8, 1, 16, 16, 256, 256), tiled(4, false)); // Gemma hd 256 prefill
        assert_eq!(route(8, 4, 16, 16, 256, 256), tiled(8, false));
        assert_eq!(route(32, 32, 100, 100, 96, 96), tiled(8, false)); // Phi-3 hd 96 prefill
        assert_eq!(
            route(32, 32, 100, 100, 96, 96),
            unmasked(32, 32, 100, 100, 96, 96)
        );

        // Neither kernel serves these: masked calls above 8 rows tile the fallback 8 rows at a time.
        for hd in [8, 32, 72, 512] {
            assert_eq!(
                route(4, 2, 40, 40, hd, hd),
                tiled(8, false),
                "causal hd {hd}"
            );
            assert_eq!(
                route(8, 8, 1024, 1024, hd, hd),
                tiled(8, false),
                "causal hd {hd}"
            );
            assert_eq!(
                unmasked(4, 2, 40, 40, hd, hd),
                SdpaRoute::Fused,
                "vision hd {hd}"
            );
            assert_eq!(route(4, 2, 8, 40, hd, hd), SdpaRoute::Fused, "q 8 hd {hd}");
        }
        let additive = Array::zeros::<f32>(&[1, 1, 40, 40]).unwrap();
        assert_eq!(
            route_masked(AttnMask::Additive(&additive), 4, 2, 40, 40, 72, 72),
            tiled(8, false)
        );
        for hd in [96, 256] {
            // gqa 64 > 32: the vector kernel refuses it at any q_len.
            assert_eq!(
                route(64, 1, 40, 40, hd, hd),
                tiled(8, false),
                "gqa 64 hd {hd}"
            );
            assert_eq!(unmasked(64, 1, 40, 40, hd, hd), SdpaRoute::Fused);
        }
        assert_eq!(route(64, 1, 4, 64, 128, 128), SdpaRoute::Fused); // gqa 64, q 4: one call
        assert_eq!(route(64, 1, 40, 40, 128, 128), SdpaRoute::Fused); // full kernel takes any gqa
    }

    /// sc-24438: [`vector_verify_rows`] is the widest query `sdpa_route` keeps as one fused call —
    /// a verify of `rows` query rows over a longer cache is one call, `rows + 1` is split — for the
    /// shipped Qwen3.8-27B (24q/4kv/hd 256, gqa 6 → 5), Qwen3.5/3.6-35B-A3B (16q/2kv/hd 256, gqa 8
    /// → 4), a gqa-4 hd-128 Llama (8), a gqa-8 hd-128 one (4), and shapes the vector kernel does
    /// not serve (8: the masked one-call bound).
    #[test]
    fn vector_verify_rows_is_the_widest_single_fused_verify() {
        let route = |hq: i32, hkv: i32, ql: i32, hd: i32| {
            sdpa_route(
                &Array::zeros::<f32>(&[1, hq, ql, hd]).unwrap(),
                &Array::zeros::<f32>(&[1, hkv, 64, hd]).unwrap(),
                &Array::zeros::<f32>(&[1, hkv, 64, hd]).unwrap(),
                AttnMask::Causal,
            )
        };
        for (hq, hkv, hd, rows) in [
            (24, 4, 256, 5),
            (16, 2, 256, 4),
            (32, 8, 128, 8),
            (32, 4, 128, 4),
            (8, 4, 72, 8),
            (64, 1, 256, 8),
        ] {
            assert_eq!(vector_verify_rows(hq, hkv, hd, hd), rows, "{hq}/{hkv}/{hd}");
            assert_eq!(
                route(hq, hkv, rows, hd),
                SdpaRoute::Fused,
                "{hq}/{hkv}/{hd}"
            );
            if vector_kernel_serves(hq, hkv, hd, hd) && rows < MLX_SDPA_VECTOR_MAX_QLEN {
                assert_eq!(
                    route(hq, hkv, rows + 1, hd),
                    SdpaRoute::Tiled {
                        rows,
                        eval_tiles: false
                    },
                    "{hq}/{hkv}/{hd}: one row wider is split"
                );
            }
        }
    }

    /// Where [`sdpa`] tiles, the result still matches the host reference: vector-kernel tiles for
    /// head dims only that kernel serves (256, 96) and for a GQA group too wide for 8 rows, across
    /// causal & no-mask, square prefill and a cached-prefix offset.
    #[test]
    fn sdpa_tiled_matches_host() {
        // (n_heads, n_kv_heads, q_len, k_len, head_dim)
        let cases = [
            (8, 1, 16, 16, 256), // hd 256, gqa 8 → 4-row tiles
            (4, 4, 20, 36, 256), // hd 256 MHA, cached prefix (offset 16) → 8-row tiles
            (8, 2, 19, 19, 96),  // hd 96 → 8-row tiles
            (32, 4, 8, 64, 128), // q_len 8 × gqa 8 > 32 → 4-row tiles
            (32, 4, 5, 5, 64),   // q_len 5 × gqa 8 > 32 → 4-row tiles + a 1-row tile
            (32, 4, 5, 37, 64),  // … over a cached prefix (offset 32)
            (4, 2, 20, 36, 72),  // hd 72 (no fused kernel): causal → 8-row fallback tiles
        ];
        for (nh, nkv, ql, kl, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nkv, kl, hd], 2);
            let v = randf(&[1, nkv, kl, hd], 3);
            assert!(matches!(
                sdpa_route(&q, &k, &v, AttnMask::Causal),
                SdpaRoute::Tiled { .. }
            ));
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let out = sdpa(&q, &k, &v, scale, mask).unwrap();
                assert_eq!(out.shape(), &[1, nh, ql, hd]);
                let host = host_attn_gqa(&q, &k, &v, nh / nkv, scale, causal);
                let e = rel_err(&host_f32(&out), &host);
                assert!(
                    e < 2e-3,
                    "tiled sdpa {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}"
                );
            }
        }
    }

    /// A prompt longer than one [`SDPA_PREFILL_BLOCK_QLEN`] block runs as evaluated full-kernel
    /// blocks (causal prefix slices) and still matches the host reference — rows on both sides of
    /// the block edge included.
    #[test]
    fn sdpa_long_prefill_blocks_match_host() {
        let (nh, nkv, hd) = (4, 2, 64);
        let ql = SDPA_PREFILL_BLOCK_QLEN + 40;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randf(&[1, nh, ql, hd], 1);
        let k = randf(&[1, nkv, ql, hd], 2);
        let v = randf(&[1, nkv, ql, hd], 3);
        let edge = SDPA_PREFILL_BLOCK_QLEN as usize;
        let mut rows = check_rows(ql as usize);
        rows.extend([edge - 1, edge, edge + 1]);
        rows.sort_unstable();
        rows.dedup();
        let out = sdpa(&q, &k, &v, scale, AttnMask::Causal).unwrap();
        assert_eq!(out.shape(), &[1, nh, ql, hd]);
        let reference = host_attn_rows(&q, &k, &v, scale, true, &rows);
        let e = rel_err_rows(&host_f32(&out), &reference);
        assert!(e < 2e-3, "blocked long prefill rel={e}");
    }

    /// A masked call at a head dim neither fused kernel serves keeps MLX's score-materializing
    /// fallback to 8-row tiles: the transient beyond the output (held twice — the tiles and their
    /// concatenation) stays under ONE full `[heads, q_len, k_len]` f32 score matrix, which a single
    /// fallback call over all 1024 rows materializes (with its softmax) and the chunked admission
    /// estimate does not price.
    #[test]
    fn sdpa_unserved_head_dims_tile_the_score_transient() {
        use mlx_rs::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        let (nh, ql) = (8, 1024);
        for hd in [512, 72] {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nh, ql, hd], 2);
            let v = randf(&[1, nh, ql, hd], 3);
            mlx_rs::transforms::eval([&q, &k, &v]).unwrap();
            // Tests run one at a time (`RUST_TEST_THREADS = 1`, `.cargo/config.toml`), so the
            // process-global counters are this test's.
            reset_peak_memory();
            let base = get_active_memory();
            let out = sdpa(&q, &k, &v, scale, AttnMask::Causal).unwrap();
            out.eval().unwrap();
            let transient = get_peak_memory().saturating_sub(base);
            let out_bytes = (nh * ql * hd) as usize * 4;
            let score_matrix = (nh * ql * ql) as usize * 4;
            assert!(
                transient.saturating_sub(2 * out_bytes) < score_matrix,
                "hd {hd}: transient {transient} B − 2 × output {out_bytes} B ≥ one full score \
                 matrix {score_matrix} B (the fallback ran untiled)"
            );
        }
    }

    /// Logical row-major host readback that honours strides. MLX's fused **full** attention kernel
    /// writes its output as a permuted-dense view (`[B, L, H, D]` storage behind `[B, H, L, D]`
    /// strides), and mlx-rs `as_slice` returns the raw buffer in **physical** order — reading that
    /// view with `as_slice` is the misread behind sc-7430. A reshape of a non-row-contiguous array
    /// copies in logical order.
    fn host_f32(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[-1])
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// Whether `out` (`[B, H, L, D]`) carries the storage layout only MLX's fused full attention
    /// kernel produces: strides `[L·H·D, D, H·D, 1]`. Chunked / concatenated / unfused outputs are
    /// row-contiguous, so this proves ONE fused full-kernel call served the whole query length.
    fn is_fused_full_kernel_output(out: &Array) -> bool {
        out.eval().unwrap();
        let s = out.shape();
        let (h, l, d) = (s[1] as usize, s[2] as usize, s[3] as usize);
        out.strides() == [l * h * d, d, h * d, 1]
    }

    /// f64 host GQA attention for the listed query `rows` only (bottom-right causal alignment), as
    /// `(logical index into [1, h, ql, hd], value)` pairs — keeps a 512-row reference cheap.
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

    /// `max|got − ref| / max|ref|` over the reference rows.
    fn rel_err_rows(got: &[f32], reference: &[(usize, f32)]) -> f32 {
        assert!(
            got.iter().all(|x| x.is_finite()),
            "nonfinite attention output"
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

    /// Every row for short prompts; both ends plus seven interior rows for long ones.
    fn check_rows(q_len: usize) -> Vec<usize> {
        if q_len <= 64 {
            return (0..q_len).collect();
        }
        let mut rows: Vec<usize> = (0..4).chain(q_len - 4..q_len).collect();
        rows.extend((1..8).map(|i| i * q_len / 8));
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// **The sc-7430 tripwire, now a positive gate (sc-24442).** At `q_len` 16, 64 and 512 ×
    /// head_dim {64, 128} × GQA/MHA × causal/none — exactly the envelope sc-7430 believed broken —
    /// [`sdpa`] runs **one fused full-kernel call** (asserted from the output's storage layout, which
    /// only that kernel produces) and matches an f64 host reference. sc-7430's "O(1)-wrong above 8
    /// rows" was the stride-blind `as_slice` readback of that layout (see
    /// `sc7430_root_cause_is_a_stride_blind_readback`), never the kernel.
    #[test]
    fn sc7430_fused_sdpa_matches_host_at_long_qlen() {
        for ql in [16, 64, 512] {
            for hd in [64, 128] {
                for (nh, nkv) in [(8, 2), (4, 4)] {
                    let scale = 1.0 / (hd as f32).sqrt();
                    let q = randf(&[1, nh, ql, hd], 1);
                    let k = randf(&[1, nkv, ql, hd], 2);
                    let v = randf(&[1, nkv, ql, hd], 3);
                    let rows = check_rows(ql as usize);
                    for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                        let out = sdpa(&q, &k, &v, scale, mask).unwrap();
                        assert_eq!(out.shape(), &[1, nh, ql, hd]);
                        assert!(
                            is_fused_full_kernel_output(&out),
                            "q_len {ql} hd {hd} {nh}/{nkv} causal={causal}: not one fused call \
                             (strides {:?})",
                            out.strides()
                        );
                        let reference = host_attn_rows(&q, &k, &v, scale, causal, &rows);
                        let e = rel_err_rows(&host_f32(&out), &reference);
                        assert!(
                            e < 2e-3,
                            "fused sdpa q_len {ql} hd {hd} {nh}/{nkv} causal={causal}: rel={e}"
                        );
                    }
                }
            }
        }
    }

    /// sc-7430's root cause, pinned: the raw fused kernel's `q_len > 8` output is correct, but its
    /// storage is `[B, L, H, D]`-ordered, so a stride-blind `as_slice` returns the logical output
    /// *transposed* — the "O(1)-wrong" numbers sc-7430 saw (and why `heads == 1` looked fine).
    #[test]
    fn sc7430_root_cause_is_a_stride_blind_readback() {
        let (nh, ql, hd) = (8, 16, 64);
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randf(&[1, nh, ql, hd], 1);
        let k = randf(&[1, nh, ql, hd], 2);
        let v = randf(&[1, nh, ql, hd], 3);
        let raw = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None,
        )
        .unwrap();
        let logical = host_f32(&raw);
        let host = host_attn_gqa(&q, &k, &v, 1, scale, true);
        assert!(rel_err(&logical, &host) < 2e-3, "kernel itself is wrong");
        let physical = raw.as_slice::<f32>().to_vec();
        assert!(
            rel_err(&physical, &host) > 0.1,
            "layout is no longer permuted"
        );
        let transposed = host_f32(&raw.transpose_axes(&[0, 2, 1, 3]).unwrap());
        assert_eq!(
            physical, transposed,
            "as_slice = the [B, L, H, D] storage order"
        );
    }

    /// An explicit **Additive** mask (the batched-prefill block-causal mask, shape `[b,1,q_len,k_len]`)
    /// matches a causal host reference on both the one-call full kernel (hd 64) and the tiled path
    /// (hd 256, which slices the mask's query axis per tile). Guards the `decode/batch.rs` prefill.
    #[test]
    fn sdpa_additive_mask_matches_host() {
        let ql = 20;
        for hd in [64, 256] {
            let nh = 8;
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
            assert!(e < 2e-3, "additive-mask sdpa hd {hd} wrong: rel={e}");
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
