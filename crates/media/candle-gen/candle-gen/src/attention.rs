//! Shared **i32-overflow-safe** scaled-dot-product attention (sc-9116 / epic 8979, the F-003 sweep).
//!
//! candle's CUDA elementwise/softmax/matmul kernels index tensor elements with **i32**. A scores
//! (or probs) tensor whose element count exceeds `i32::MAX` (~2.147e9) silently corrupts its tail —
//! the trailing query rows get garbage attention (a near-zero / wrong context) with no error. For a
//! DiT operating on image latent tokens, or a VAE spatial self-attention operating on `H·W` pixels,
//! the sequence length scales with the rendered resolution, so at the advertised max render sizes
//! (2048² image → ~16k DiT tokens, or a VAE bottleneck `H·W` → ~65k) the `[…,Sq,Sk]` scores tensor
//! blows past `i32::MAX`:
//!
//! - SDXL UNet self-attn @ 2048²: `8·65536² ≈ 3.4e10`.
//! - SD3 / Z-Image / Ideogram / Krea DiT @ 2048²: `~24·16384² ≈ 6e9`.
//! - chroma / flux2 / qwen-image VAE mid-block @ 2048²: `65536² ≈ 4.3e9`.
//!
//! F-003 (sc-8983) first fixed this for the flux2 / chroma / lens / qwen-image DiT transformers with a
//! per-crate `attention_budgeted` helper that chunks over the **query rows** — each query row's softmax
//! is over all keys and is independent of the other rows, so the chunked result is *mathematically*
//! equivalent to the single pass, and the chunking only ever engages on the over-budget buckets. This
//! module hoists that pattern into the shared commons so the remaining audited sites (sc-9116) share one
//! guarded copy instead of a growing pile of near-identical per-crate copies.
//!
//! **Mathematically equivalent is not bitwise equal.** Narrowing the query axis changes the GEMM `M`
//! dimension, and both candle's CPU `gemm` and cuBLAS may select a different tiling / accumulation order
//! for `M = 541` than for `M = 4128`. So chunked and un-chunked outputs agree only to a tolerance: this
//! module's own equivalence helper compares to `1e-5`, and SC-15943 is a sibling crate
//! (`candle-gen-wan`) that asserted *exact* equality and fails by 1–2 ULP on macOS-arm64. Treat
//! per-row independence as an argument about the math, never as a bit-identity guarantee.
//!
//! Two shapes are covered:
//! - [`sdpa_budgeted_bhsd`] — the 4-D DiT shape `[B, H, Sq, D]` (heads explicit), optional additive mask.
//! - [`sdpa_budgeted_flat`] — the 3-D shape `[N, Sq, D]` where `N` folds `B·H` (SDXL, whose attention
//!   reshapes heads into the batch dim) **or** `B` for a single-head VAE spatial attention (`N=B`,
//!   `Sq=H·W`).
//!
//! Both take a caller-supplied `softmax` closure so each site keeps its exact softmax semantics — the
//! composable `softmax(_, D::Minus1)` (grad-carrying, used by the trainers), the fused `softmax_last_dim`,
//! or an f32-upcast variant — unchanged. The helpers only wrap the scores matmul + softmax + value matmul
//! in the budgeted query-row chunking; the numerics of a single block are exactly the caller's.
//!
//! ## The boundaries come from the shared rung-3 planner (SC-15796)
//!
//! Rung 3 of the SC-15448 memory ladder splits into a backend-neutral **planner** — measured budgets
//! plus pure arithmetic, no tensors — and a genuinely per-backend **kernel**. The planner lives in
//! [`gen_core::attention_budget`]; SC-15793 hoisted it there and made `mlx_gen::attention` a shim over
//! it, and this module is the candle half. This module's `query_block` is now a pure delegation to
//! [`AttentionBudget::query_block_rows`] and applies no arithmetic of its own, so a budget
//! published as `strategyParameters.chunkedAttention.attentionChunkSize` names the **same** chunking on
//! both backends and the epic's per-backend calibration evidence is comparable rather than two
//! coincidentally-equal literals.
//!
//! The public `sdpa_budgeted_*` signatures still take a plain `budget: usize`, so the ~20 consumer
//! crates need no edits — only `query_block`'s body moved.
//!
//! The kernels must NOT merge, and that is measured rather than assumed: candle's `attention_basic`
//! **materializes** `[B,H,Sq,Sk]`, so bounding it is worth **−32%** on the Z-Image denoise phase
//! (8.394 → 5.709 GB, SC-15256), while MLX's fused SDPA never builds that tensor and the same rung is
//! worth **−1.7%** on its denoise phase (SC-15615). One planner, two kernels roughly an order of
//! magnitude apart in value; neither magnitude may be inferred from the other.
//!
//! ## Why this module does not take [`gen_core::attention_budget::AttentionPlan`]
//!
//! gen-core pairs the budget with a cancel flag in `AttentionPlan` because a bounded call splits one
//! previously atomic kernel launch into N, and those boundaries are the only place a cancel can land
//! *inside* a transformer forward. That shape is deliberately **not** carried across here yet, for
//! reasons that are candle's own rather than inherited from the MLX twin:
//!
//! - **Candle's cancellation lands outside the forward, at three sites that already exist.** Per denoise
//!   step in [`crate::sampler`] (`run_flow_sampler` and the DPM/ancestral run loops each test
//!   `cancel.is_cancelled()` before the model eval), per decode tile through [`crate::check_cancel`]
//!   inside the [`crate::vae_tiling`] callback, and per load phase in [`crate::residency`]. All three
//!   raise [`crate::CandleError::Canceled`], which the `From` bridge lifts to `gen_core::Error::Canceled`
//!   — a *cancelled* job, not a failed one (sc-4481).
//! - **A between-chunk check would be a genuine granularity gain, and it is a behaviour change.**
//!   Candle's DiT forward carries no cancel check today (`candle-gen-z-image/src/packed_dit.rs` has
//!   none), so on a 4-step Z-Image-Turbo render the finest in-denoise granularity is one whole step.
//!   Adding one is worth doing, but it means threading a plan through `sdpa_budgeted_*` and every
//!   provider's `forward_with_attention_budget` (~39 call sites across ~20 crates), whereas this
//!   refactor is behaviour-preserving by construction. Tracked as **SC-16007**, not folded in here.
//!
//! [`AttentionBudget`] and [`CONSTRAINED_ATTN_SCORES_BUDGET`] are re-exported below because they *are*
//! the shared planner; `AttentionPlan` is not, because nothing on this backend consumes it yet.

use candle_core::{Result, Tensor, D};

/// The rung-3 **planner**, re-exported from its shared home ([`gen_core::attention_budget`], SC-15793)
/// so candle provider code names the same budget type — and, where it applies, the same declared value
/// — as the MLX lane. Only the kernels below are candle's.
///
/// [`CONSTRAINED_ATTN_SCORES_BUDGET`] is the **Z-Image** operating point (64 Mi), measured
/// independently on both backends. It is not a contract-wide constant: candle's Krea lane measured and
/// publishes 128 Mi (`candle_gen_krea::pipeline::CONSTRAINED_ATTN_SCORES_BUDGET`), and the
/// i32-overflow guard below runs the same planner at 1e9. What is shared is the planner, not the number.
pub use gen_core::attention_budget::{AttentionBudget, CONSTRAINED_ATTN_SCORES_BUDGET};

/// Max elements in a single attention scores tensor before the query rows are chunked. candle CUDA
/// kernels index elements with **i32**, so a scores/probs tensor exceeding `i32::MAX` (~2.147e9)
/// silently corrupts its tail. 1.0e9 keeps each chunk well under the limit while leaving every render
/// size whose single-pass scores are already `≤ 1e9` a single un-chunked pass (byte-identical to the
/// pre-guard path). Matches the per-crate F-003 budget so the two never diverge.
///
/// This is the **same planner** as the memory rung's [`CONSTRAINED_ATTN_SCORES_BUDGET`] at a different
/// setting — a correctness guard rather than a memory operating point, which is why it is much larger.
pub const ATTN_SCORES_BUDGET: usize = 1_000_000_000;

/// The largest query-block length whose `[…, block, Sk]` scores element count stays within `budget`.
/// Returns the whole `sq` (single un-chunked pass) when the full scores tensor already fits — so the
/// common sizes are the unchanged single matmul+softmax+matmul. `rows_per_query` is the product of all
/// the leading (non-query, non-key) dims times `sk` — i.e. the element count contributed by ONE query
/// row (`B·H·Sk` for the 4-D shape, `N·Sk` for the flat shape).
///
/// **Pure delegation to [`AttentionBudget::query_block_rows`] (SC-15796).** This must not apply
/// arithmetic of its own: candle and MLX would then plan different boundaries from the same declared
/// budget, which is the exact divergence the shared planner exists to prevent.
/// `the_delegation_reproduces_the_sc9116_guard_arithmetic` pins that the delegation is boundary-for-
/// boundary what this crate computed before the hoist, and
/// `candle_chunk_boundaries_match_the_shared_cross_backend_table` pins it against the shared table.
fn query_block(rows_per_query: usize, sq: usize, budget: usize) -> usize {
    // `usize::MAX` is this crate's un-chunked sentinel (the trainers and every unbounded call site pass
    // it). Map it to the planner's own `u64::MAX` sentinel explicitly rather than via `as`, which would
    // widen to 2^32-1 — a real budget — on a 32-bit target.
    let budget = if budget == usize::MAX {
        u64::MAX
    } else {
        budget as u64
    };
    // The result is bounded above by `sq`, which came from a `usize`, so the narrowing is exact.
    AttentionBudget::from_score_elements(budget, false)
        .query_block_rows(rows_per_query as u64, sq as u64) as usize
}

/// i32-overflow-safe SDPA over the 4-D shape `q,k,v: [B, H, Sq, D]` (k/v key length `Sk` may differ
/// from `Sq`), returning the attention output `[B, H, Sq, D]` (the caller does its own
/// transpose/head-merge, so this is a drop-in around an existing `matmul → softmax → matmul`).
///
/// `scale` multiplies the scores (`head_dim^-0.5` at the call sites). `mask`, if given, is an additive
/// bias broadcast onto the scores AFTER scaling; it must broadcast over the query rows (e.g. `[B,1,1,Sk]`
/// or `[B,1,Sq,Sk]` with a per-row layout that narrows consistently — the common `[B,1,1,Sk]` and the
/// full `[B,1,Sq,Sk]` both do). `softmax` is applied to each scores block over its last dim; pass the
/// exact closure the call site used (`softmax_last_dim`, composable `softmax(_, D::Minus1)`, or an
/// f32-upcast wrapper) so the numerics are unchanged.
///
/// When `budget` is large enough for the full `[B,H,Sq,Sk]` scores tensor this is a single pass,
/// byte-identical to the un-guarded `(q·kᵀ·scale) → softmax → ·v`. Otherwise it chunks over the query
/// rows; since each query row's softmax is over all keys and independent of the others, the chunked
/// result is *mathematically* equivalent — but **not** bitwise equal to the single pass. Narrowing the
/// query axis changes the GEMM `M` dimension, so candle's CPU `gemm` and cuBLAS may accumulate in a
/// different order at a different `M`; the two agree to a tolerance (~1 ULP in practice, `1e-5` in this
/// crate's tests) rather than exactly. SC-15943 is the defect a sibling crate hit by asserting `== 0.0`
/// on that difference. Do not build an exact-equality assertion on this path.
pub fn sdpa_budgeted_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> Result<Tensor>,
    budget: usize,
) -> Result<Tensor> {
    let (b, h, sq, _d) = q.dims4()?;
    let sk = k.dim(2)?;
    let q = q.contiguous()?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;

    let block = query_block(b * h * sk, sq, budget);
    if block >= sq {
        record_chunk_count(1);
        let mut scores = (q.matmul(&k_t)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax(&scores)?;
        return probs.matmul(&v);
    }

    let mut blocks = Vec::new();
    let mut start = 0;
    while start < sq {
        let len = block.min(sq - start);
        let mut scores = (q.narrow(2, start, len)?.matmul(&k_t)? * scale)?;
        if let Some(m) = mask {
            // A `[B,1,1,Sk]` mask broadcasts identically onto every query chunk; a per-query
            // `[B,1,Sq,Sk]` mask must be narrowed to the same rows so each chunk sees its own slice.
            let m = if m.dim(2)? == sq {
                m.narrow(2, start, len)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&m)?;
        }
        let probs = softmax(&scores)?;
        blocks.push(probs.matmul(&v)?); // [B,H,len,D]
        start += len;
    }
    record_chunk_count(blocks.len());
    Tensor::cat(&blocks, 2) // [B,H,Sq,D]
}

/// i32-overflow-safe SDPA over the 3-D shape `q,k,v: [N, Sq, D]`, returning `[N, Sq, D]`. `N` folds the
/// leading dims — `B·H` for SDXL's head-into-batch attention, or `B` for a single-head VAE spatial
/// self-attention where `Sq = H·W`. `scale`/`softmax` behave as in [`sdpa_budgeted_bhsd`]; there is no
/// mask parameter (none of the flat-shape call sites use one). A drop-in around an existing
/// `q.matmul(kᵀ)·scale → softmax → ·v` on the 3-D tensors.
pub fn sdpa_budgeted_flat(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    softmax: impl Fn(&Tensor) -> Result<Tensor>,
    budget: usize,
) -> Result<Tensor> {
    let (n, sq, _d) = q.dims3()?;
    let sk = k.dim(1)?;
    let q = q.contiguous()?;
    let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?;
    let v = v.contiguous()?;

    let block = query_block(n * sk, sq, budget);
    if block >= sq {
        record_chunk_count(1);
        let scores = (q.matmul(&k_t)? * scale)?;
        let probs = softmax(&scores)?;
        return probs.matmul(&v);
    }

    let mut blocks = Vec::new();
    let mut start = 0;
    while start < sq {
        let len = block.min(sq - start);
        let scores = (q.narrow(1, start, len)?.matmul(&k_t)? * scale)?;
        let probs = softmax(&scores)?;
        blocks.push(probs.matmul(&v)?); // [N,len,D]
        start += len;
    }
    record_chunk_count(blocks.len());
    Tensor::cat(&blocks, 1) // [N,Sq,D]
}

/// Test-only observation of how many chunks the last `sdpa_budgeted_*` call actually ran.
///
/// Without this, every equivalence test below is a false green: they all assert *chunked == single
/// pass*, which is trivially true when the chunking never engages, so the whole suite would keep
/// passing with the lever deleted — or with the kernel sizing its chunks from arithmetic of its own
/// rather than from the shared planner. Compiled out entirely in release; `RUST_TEST_THREADS=1` is
/// forced repo-wide (`.cargo/config.toml`), so a process-global counter is safe. Mirrors
/// `mlx_gen::attention`'s probe so the two backends' conformance tests read the same way.
#[cfg(test)]
mod chunk_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static LAST_CHUNK_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Chunks run by the most recent `sdpa_budgeted_*` call (`1` = the un-chunked fast path).
    pub fn last_chunk_count() -> usize {
        LAST_CHUNK_COUNT.load(Ordering::Relaxed)
    }

    /// Reset before a call so a stale value cannot satisfy an assertion.
    pub fn reset() {
        LAST_CHUNK_COUNT.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
use chunk_probe::{last_chunk_count, reset as reset_chunk_count};

#[cfg(test)]
fn record_chunk_count(n: usize) {
    chunk_probe::LAST_CHUNK_COUNT.store(n, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline(always)]
fn record_chunk_count(_n: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::ops::softmax_last_dim;

    fn approx_eq(a: &Tensor, b: &Tensor) {
        assert_eq!(a.dims(), b.dims());
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() < 1e-5,
                "chunked attention diverged: {x} vs {y}"
            );
        }
    }

    /// Assert the chunk count the last `sdpa_budgeted_*` call ran. Every `chunked == single pass`
    /// assertion below is vacuous without this — `chunked == single` is trivially true when the
    /// chunking never engages.
    fn assert_chunks(expected: usize) {
        assert_eq!(
            last_chunk_count(),
            expected,
            "the kernel ran {} chunks, not {expected} — the equivalence around this call would be \
             vacuous",
            last_chunk_count()
        );
    }

    #[test]
    fn bhsd_chunked_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single un-chunked pass — the i32-overflow guard invariant (sc-9116).
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        reset_chunk_count();
        let single = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, None, sm, usize::MAX).unwrap();
        // The unbounded arm must be a single un-chunked pass. Then a tiny budget → single-row chunks,
        // and a MID-SIZE budget forcing multi-row chunks + a remainder (block=3 over s=7 → chunks
        // 3,3,1) — the sc-9116 test-hardening ask.
        assert_chunks(1);
        reset_chunk_count();
        let one_row = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, None, sm, 1).unwrap();
        assert_chunks(7);
        approx_eq(&single, &one_row);
        reset_chunk_count();
        // budget = b·h·sk·block = 1·2·7·3 = 42 → block = 42/(1·2·7) = 3.
        let ragged = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, None, sm, 42).unwrap();
        assert_chunks(3);
        approx_eq(&single, &ragged);
    }

    #[test]
    fn bhsd_masked_chunked_matches_single_pass() {
        // With an additive `[B,1,1,Sk]` mask the chunked path must still match: the mask broadcasts
        // identically onto every query chunk.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let mask = Tensor::randn(0f32, 1f32, (b, 1, 1, s), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        reset_chunk_count();
        let single = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, usize::MAX).unwrap();
        assert_chunks(1);
        reset_chunk_count();
        let chunked = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, 42).unwrap();
        assert_chunks(3);
        approx_eq(&single, &chunked);
    }

    #[test]
    fn bhsd_per_query_mask_chunked_matches_single_pass() {
        // A FULL per-query `[B,1,Sq,Sk]` additive mask (ideogram's `[B,1,L,L]` shape) exercises the
        // narrow-slice branch: each query chunk must see its OWN mask rows (`narrow(2, start, len)`),
        // not the whole mask. A mid-size budget (block=3 over s=7 → 3,3,1) forces the multi-row narrow.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        // Distinct per-(query,key) bias so a wrong (un-narrowed / mis-aligned) slice would diverge.
        let mask = Tensor::randn(0f32, 1f32, (b, 1, s, s), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        reset_chunk_count();
        let single = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, usize::MAX).unwrap();
        assert_chunks(1);
        reset_chunk_count();
        let one_row = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, 1).unwrap();
        assert_chunks(7);
        approx_eq(&single, &one_row);
        reset_chunk_count();
        let ragged = sdpa_budgeted_bhsd(&q, &k, &v, 0.5, Some(&mask), sm, 42).unwrap();
        assert_chunks(3);
        approx_eq(&single, &ragged);
    }

    #[test]
    fn guard_fires_at_advertised_sizes_sc11154() {
        // sc-11154 / F-081: the five newly-swept sites overflow i32 *within* their advertised,
        // `validate`-accepted envelopes. Assert the shared budget (`ATTN_SCORES_BUDGET`) engages the
        // query-row chunking at each site's advertised over-threshold size — and, critically, does NOT
        // chunk a comfortably in-budget size (so the common path stays the byte-identical single pass).
        // `rows_per_query` is the element count contributed by ONE query row (`N·Sk` flat, `B·H·Sk` 4-D).
        let b = ATTN_SCORES_BUDGET;

        // (a) stock SDXL UNet self-attn @ 2048² (heads-into-batch flat): N = B·H = 2·10, Sk = Sq = 16384
        // → 2·10·16384² ≈ 5.4e9. Chunk length must be < Sq.
        assert!(query_block(2 * 10 * 16384, 16384, b) < 16384);
        // (b) FLUX.1 VAE mid-block @ 2048² (single-head flat): N = 1, Sk = Sq = 65536 → 65536² ≈ 4.3e9.
        assert!(query_block(65536, 65536, b) < 65536);
        // (c) boogu Qwen3-VL ViT at a ~3.0 MP reference (4-D): B·H = 1·16, Sk = Sq = 11585 → 16·11585².
        assert!(query_block(16 * 11585, 11585, b) < 11585);
        // (d) krea grounded TE at the inclusive 8192-token cap (4-D): B·H = 1·32, Sk = Sq = 8192 → 2^31.
        assert!(query_block(32 * 8192, 8192, b) < 8192);
        // (e) sensenova ~8.2k-token image prefill (4-D), heads = 32: 32·8192² > i32::MAX.
        assert!(query_block(32 * 8192, 8200, b) < 8200);

        // Below the budget every one of these families runs a SINGLE un-chunked pass (block == Sq). A
        // 512² SDXL attn (Sq = 4096, N = 20 → 20·4096² ≈ 3.4e8) and a 1024² FLUX VAE (Sq = 16384 →
        // 16384² ≈ 2.7e8) both fit, so the guard is a no-op there.
        assert_eq!(query_block(20 * 4096, 4096, b), 4096);
        assert_eq!(query_block(16384, 16384, b), 16384);
    }

    /// **The cross-backend conformance gate (SC-15793).** Candle's chunk boundaries must match the
    /// ones `gen_core::attention_budget` declares — the shared rung-3 planner the MLX backend also
    /// plans through (`mlx_gen::attention`).
    ///
    /// This is the property that makes the epic's per-backend calibration evidence comparable. The two
    /// backends' *savings* differ by an order of magnitude (−32% candle denoise vs −1.7% MLX denoise)
    /// because their kernels genuinely differ, and that is expected and correct. But a declared budget
    /// has to name the **same chunking** on both sides, or
    /// `strategyParameters.chunkedAttention.attentionChunkSize` means two different things and the two
    /// backends' numbers cannot be placed in one matrix.
    ///
    /// Since SC-15796 [`query_block`] *is* a delegation to the shared planner, comparing the two
    /// against each other would now be a tautology — it would pass with the shared arithmetic wrong.
    /// What remains load-bearing, and what this asserts, is the delegation against the table's
    /// **declared literals**: perturb the shared planner and candle goes red here, which is what makes
    /// the delegation a checked property rather than a claim. The kernel's own use of the boundary is
    /// pinned separately by [`the_kernels_chunk_exactly_as_the_shared_planner_declares`].
    #[test]
    fn candle_chunk_boundaries_match_the_shared_cross_backend_table() {
        for case in gen_core::attention_budget::CROSS_BACKEND_CHUNK_CASES {
            let rows_per_query = case.rows_per_query as usize;
            let sq = case.sq as usize;
            let budget = case.budget as usize;
            let candle = query_block(rows_per_query, sq, budget);
            assert_eq!(
                candle as u64, case.expect_rows,
                "{}: candle plans {candle} rows, the shared table declares {}",
                case.what, case.expect_rows
            );
        }
    }

    /// **The kernel-to-planner binding (SC-15796).** [`query_block`] delegating is only half of AC 1:
    /// the kernels have to actually *size their chunks from it*. A kernel that reintroduced local
    /// arithmetic — the failure this refactor exists to prevent — would leave every test above green.
    ///
    /// So for a sweep of block sizes covering the whole range (a single row, ragged multi-row splits,
    /// and the un-chunked call), both kernels must run exactly `ceil(Sq / shared_boundary)` chunks.
    /// The shared table's own rows are production geometries (Sq up to 65536) that cannot be launched
    /// in a unit test — the sibling MLX test learned that the hard way — so this sweeps runnable shapes
    /// and derives the expected count from the planner rather than from a literal.
    #[test]
    fn the_kernels_chunk_exactly_as_the_shared_planner_declares() {
        let dev = Device::Cpu;
        let sm = |x: &Tensor| softmax_last_dim(x);
        // Sq = 11 is prime, so every block size below it leaves a ragged tail — a kernel that rounded
        // the tail differently from `ceil` shows up here.
        let (b, h, s, d) = (1usize, 3usize, 11usize, 4usize);
        let q4 = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k4 = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v4 = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let (n, sf) = (3usize, 11usize);
        let q3 = Tensor::randn(0f32, 1f32, (n, sf, d), &dev).unwrap();
        let k3 = Tensor::randn(0f32, 1f32, (n, sf, d), &dev).unwrap();
        let v3 = Tensor::randn(0f32, 1f32, (n, sf, d), &dev).unwrap();

        let mut chunked_arms = 0;
        for rows in [1usize, 2, 4, 7, 10, 11] {
            // 4-D: rows_per_query = B·H·Sk.
            let rows_per_query = b * h * s;
            let budget = rows_per_query * rows;
            let block = query_block(rows_per_query, s, budget);
            assert_eq!(
                block, rows,
                "the planner did not produce the {rows}-row block"
            );
            reset_chunk_count();
            sdpa_budgeted_bhsd(&q4, &k4, &v4, 0.5, None, sm, budget).unwrap();
            assert_chunks(s.div_ceil(block));

            // Flat: rows_per_query = N·Sk. Same planner, different layout.
            let rows_per_query = n * sf;
            let budget = rows_per_query * rows;
            let block = query_block(rows_per_query, sf, budget);
            assert_eq!(block, rows, "the flat planner missed the {rows}-row block");
            reset_chunk_count();
            sdpa_budgeted_flat(&q3, &k3, &v3, 0.5, sm, budget).unwrap();
            assert_chunks(sf.div_ceil(block));

            if rows < s {
                chunked_arms += 1;
            }
        }
        assert!(
            chunked_arms >= 5,
            "only {chunked_arms} arms actually chunked — this test would pass with chunking gone"
        );
    }

    /// **AC 6 without a GPU: the delegation is boundary-for-boundary the pre-hoist arithmetic.**
    ///
    /// Why pinning boundaries also pins real-weight output. This refactor moved *only* `query_block`'s
    /// body. This test asserts the delegation returns an **identical boundary for every input** the
    /// pre-hoist closed form was given, and every downstream statement in both kernels is unchanged —
    /// the same `narrow` / `matmul` / `broadcast_add` / `softmax` / `cat` sequence in the same order,
    /// with `record_chunk_count` compiling to nothing outside `cfg(test)`. Identical boundary plus
    /// identical downstream code means the *emitted instruction sequence* is the same as the pre-hoist
    /// build's, and therefore so is the output, bit for bit.
    ///
    /// Note what that argument deliberately does **not** rely on: it never claims chunked and un-chunked
    /// results agree bitwise. **They do not.** Chunking changes the GEMM `M` dimension, so candle's CPU
    /// `gemm` and cuBLAS may choose a different tiling / accumulation order — which is why `approx_eq`
    /// above compares to `1e-5`, and why SC-15943 is a sibling crate (`candle-gen-wan`) that asserted
    /// exact equality on that difference and fails by 1–2 ULP on macOS-arm64. The bit-identity claimed
    /// here is *pre-hoist vs post-hoist at the same boundary*, never *chunked vs single pass*.
    ///
    /// The oracle below is deliberately **not** a second planner: nothing derives a production boundary
    /// from it, and it exists permanently as the i32-overflow guard's own invariant — a correctness
    /// guard, not just a memory operating point, so the boundaries it produces must never drift.
    /// SC-15793's adversarial review swept the same equivalence over ~3M grid cases plus ~4M random
    /// full-range inputs with zero divergences; this keeps a representative slice of that in CI.
    #[test]
    fn the_delegation_reproduces_the_sc9116_guard_arithmetic() {
        /// The pre-SC-15796 closed form, verbatim.
        fn pre_hoist(rows_per_query: usize, sq: usize, budget: usize) -> usize {
            if rows_per_query.saturating_mul(sq) <= budget {
                sq
            } else {
                (budget / rows_per_query.max(1)).max(1)
            }
        }

        let budgets = [
            0usize,
            1,
            42,
            63,
            4096,
            CONSTRAINED_ATTN_SCORES_BUDGET as usize,
            // Krea's own declared operating point (`candle_gen_krea::pipeline::
            // CONSTRAINED_ATTN_SCORES_BUDGET`) — the second family consuming this kernel in production.
            // Krea's crate-local pin checks 128 Mi against the *shared planner* only; without this row
            // the budget it actually ships is never compared to the pre-hoist closed form.
            128 * 1024 * 1024,
            ATTN_SCORES_BUDGET,
            usize::MAX,
        ];
        // `32 * 8192` and the `8192` sq are krea's grounded-TE geometry at its inclusive token cap
        // (B·H = 1·32, Sq = Sk = 8192), where 128 Mi plans 512 query rows and 64 Mi plans 256.
        let rows = [
            0usize,
            1,
            7,
            16384,
            30 * 4128,
            32 * 8192,
            2 * 10 * 16384,
            65536,
        ];
        let sqs = [0usize, 1, 2, 7, 32, 4128, 8192, 16384, 65536];
        let mut chunking_cases = 0;
        for &budget in &budgets {
            for &rows_per_query in &rows {
                for &sq in &sqs {
                    let want = pre_hoist(rows_per_query, sq, budget);
                    let got = query_block(rows_per_query, sq, budget);
                    assert_eq!(
                        got, want,
                        "budget {budget}, rows_per_query {rows_per_query}, sq {sq}: the shared \
                         planner moved a boundary the i32-overflow guard shipped"
                    );
                    if want < sq {
                        chunking_cases += 1;
                    }
                }
            }
        }
        // A grid where nothing ever chunks would agree trivially.
        assert!(
            chunking_cases > 100,
            "only {chunking_cases} grid cases actually chunked — the agreement is near-vacuous"
        );
    }

    /// The shared table must cover the budget this crate actually ships as its i32-overflow guard,
    /// otherwise the conformance above could pass while the production budget went unchecked.
    #[test]
    fn the_shared_table_covers_this_crates_guard_budget() {
        assert!(
            gen_core::attention_budget::CROSS_BACKEND_CHUNK_CASES
                .iter()
                .any(|c| c.budget == ATTN_SCORES_BUDGET as u64),
            "the shared cross-backend table no longer covers ATTN_SCORES_BUDGET ({ATTN_SCORES_BUDGET})"
        );
    }

    #[test]
    fn flat_chunked_matches_single_pass() {
        // The 3-D (heads-folded / single-head VAE) shape, same invariant.
        let dev = Device::Cpu;
        let (n, s, d) = (3usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let sm = |x: &Tensor| softmax_last_dim(x);
        reset_chunk_count();
        let single = sdpa_budgeted_flat(&q, &k, &v, 0.5, sm, usize::MAX).unwrap();
        assert_chunks(1);
        reset_chunk_count();
        let one_row = sdpa_budgeted_flat(&q, &k, &v, 0.5, sm, 1).unwrap();
        assert_chunks(7);
        approx_eq(&single, &one_row);
        reset_chunk_count();
        // budget = n·sk·block = 3·7·3 = 63 → block = 63/(3·7) = 3.
        let ragged = sdpa_budgeted_flat(&q, &k, &v, 0.5, sm, 63).unwrap();
        assert_chunks(3);
        approx_eq(&single, &ragged);
    }
}
