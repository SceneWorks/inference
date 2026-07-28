//! Shared **bounded attention** (rung 3 of the SC-15448 ladder) for the MLX backend — the MLX twin of
//! `candle_gen::sdpa_budgeted_bhsd` (sc-9116 / sc-9570).
//!
//! The lever is query-row chunking: every query row's softmax is over **all** keys and is independent
//! of the other rows, so running the attention over blocks of query rows leaves each query's complete
//! key/value domain intact while bounding the per-call attention scratch to `chunk_rows × Sk` scores
//! instead of `Sq × Sk`. Precision, schedule, seed, conditioning, and the output contract are
//! untouched: the same [`scaled_dot_product_attention`] is called with the same `scale`, the same
//! dtypes, and the same k/v.
//!
//! ## What this bounds on MLX — measured (SC-15615)
//!
//! candle's CPU/CUDA `attention_basic` **materializes** the whole `[B,H,Sq,Sk]` scores tensor, so
//! bounding it there is a large unconditional win — that is what bought the Candle Z-Image 8 GB fit
//! (staged Q4 denoise 8.394 → 5.709 GB, SC-15256).
//!
//! **MLX never materializes it.** `fast::scaled_dot_product_attention` dispatches to a fused Metal
//! kernel that streams the scores. Measured in isolation on Apple M5 Max at the production Z-Image
//! DiT shape (B=1, H=30, D=128, f32), the unbounded peak is exactly `4 · B·H·Sq·D · sizeof(dtype)` —
//! q, k, v and the output, nothing else (241.9 MiB at Sq=4128, 961.9 MiB at Sq=16416; a materialized
//! score tensor at Sq=16416 would alone be 30 GiB). `fused_sdpa_does_not_materialize_the_scores`
//! pins that. So there is **no score tensor for this helper to bound**, and in isolation — with q/k/v
//! already materialized and pinned alive — chunking only *adds* transients (+50% peak).
//!
//! Inside a real DiT forward the result inverts, for a different reason. There q/k/v are intermediates
//! in a deep lazy graph, and [`AttentionBudget::eval_per_chunk`] cuts that graph at every chunk
//! boundary, letting MLX free upstream transients earlier. Measured on the real Z-Image-turbo q4 DiT
//! (1024², 4 steps, staged residency):
//!
//! | Arm | Denoise peak | vs unbounded |
//! |---|---:|---:|
//! | unbounded | 4.7746 GiB | — |
//! | 64 Mi chunk, lazy (no per-chunk eval) | 4.7708 GiB | −0.08% (noise) |
//! | 64 Mi chunk + per-chunk eval | 4.6944 GiB | **−1.7%** |
//! | never-chunks budget + eval flag (control) | 4.7747 GiB | +0.00% |
//!
//! The control arm is the important one: setting `eval_per_chunk` on a budget that never chunks is
//! inert, so the saving is genuinely produced by chunking — but by the **graph cut** it forces, not by
//! bounding a score matrix. It is therefore real, quality-preserving (`sum|out|` identical to the last
//! digit across all four arms) and *small* — nothing like CUDA's 32%. End to end through the staged
//! generate it measured 4.898 → 4.653 GiB (−5.0%) with a bit-identical image.
//!
//! Whether a given family/geometry benefits stays a measurement, never an assumption:
//! [`AttentionBudget::UNBOUNDED`] is the default and every caller keeps the single-call path until a
//! measured rung selects otherwise. A family whose attention *is* materialized (a hand-rolled
//! `matmul → softmax → matmul`, or a shape the fused kernel rejects) gets the full CUDA-style saving
//! from this same seam instead of forking its own copy.
//!
//! ## Laziness
//!
//! MLX is lazy, and per the table above the laziness is where the whole MLX-side saving lives.
//! `AttentionBudget::eval_per_chunk` forces each chunk's output to be materialized before the next
//! chunk is built — the same lever `mlx_gen_flux2::chunk::MemoryConfig::eval_per_block` uses one level
//! up. It is **bit-exact** (it only forces materialization) and costs one synchronization per chunk.
//! [`AttentionBudget::CONSTRAINED`] therefore sets it; a lazily-chunked budget measured as no better
//! than unbounded.
//!
//! `eval` is not valid inside an autograd trace, so `eval_per_chunk` must stay
//! `false` on training paths. The trainers never select a bounded rung (the default is
//! [`AttentionBudget::UNBOUNDED`], whose fast path is a single unchunked call), so this is structural
//! rather than a runtime guard.

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use crate::Result;

/// The production constrained score budget: 64 Mi elements per attention call.
///
/// This is the exact parameter the Candle/CUDA Z-Image rung-3 port measured in SC-15256 ("a 64
/// Mi-element query-row attention score budget"), reused verbatim so the two backends' bounded
/// attention is the same lever with the same knob and a cross-backend comparison is meaningful.
pub const CONSTRAINED_ATTN_SCORES_BUDGET: u64 = 64 * 1024 * 1024;

/// How much attention scratch one call may build at once.
///
/// [`UNBOUNDED`](Self::UNBOUNDED) is the default everywhere and is a single
/// [`scaled_dot_product_attention`] call — byte-identical to the pre-rung forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBudget {
    /// Max elements in one chunk's `[B, H, chunk_rows, Sk]` score domain. `u64::MAX` ⇒ unbounded.
    max_score_elements: u64,
    /// Materialize each chunk's output before building the next (see the module doc). Ignored when
    /// the budget is unbounded or the whole call already fits.
    eval_per_chunk: bool,
}

impl AttentionBudget {
    /// No bound — one fused call, byte-identical to the un-guarded forward. The default for every
    /// path that has not selected the bounded-attention rung (including all training paths).
    pub const UNBOUNDED: Self = Self {
        max_score_elements: u64::MAX,
        eval_per_chunk: false,
    };

    /// The production bounded rung: [`CONSTRAINED_ATTN_SCORES_BUDGET`] scores per chunk, each chunk
    /// materialized before the next is built.
    pub const CONSTRAINED: Self = Self {
        max_score_elements: CONSTRAINED_ATTN_SCORES_BUDGET,
        eval_per_chunk: true,
    };

    /// A budget of exactly `max_score_elements` scores per chunk. `0` is clamped to `1` (one query
    /// row per chunk) rather than dividing by zero.
    pub const fn from_score_elements(max_score_elements: u64, eval_per_chunk: bool) -> Self {
        Self {
            max_score_elements,
            eval_per_chunk,
        }
    }

    /// The declared score budget — the exact value recorded as the strategy parameter.
    pub const fn max_score_elements(self) -> u64 {
        self.max_score_elements
    }

    /// `true` when this budget can never chunk (the fast path).
    pub const fn is_unbounded(self) -> bool {
        self.max_score_elements == u64::MAX
    }

    /// The largest query-block length whose `[B, H, block, Sk]` score count stays within the budget.
    /// Returns the whole `sq` (a single un-chunked call) when the full call already fits.
    pub fn query_block(self, b: i32, h: i32, sq: i32, sk: i32) -> i32 {
        if self.is_unbounded() || sq <= 1 {
            return sq;
        }
        let rows_per_query = (b.max(1) as u64)
            .saturating_mul(h.max(1) as u64)
            .saturating_mul(sk.max(1) as u64);
        if rows_per_query.saturating_mul(sq.max(0) as u64) <= self.max_score_elements {
            return sq;
        }
        let block = self.max_score_elements / rows_per_query.max(1);
        // `block` is bounded above by `sq` here (the whole call did not fit), so the cast is safe.
        block.clamp(1, sq as u64) as i32
    }
}

impl Default for AttentionBudget {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

/// Bounded scaled-dot-product attention over the 4-D DiT shape `q, k, v: [B, H, Sq, D]` (the key
/// length `Sk` may differ from `Sq`), returning `[B, H, Sq, D]` — a drop-in around an existing
/// [`scaled_dot_product_attention`] call.
///
/// When `budget` leaves the whole call within bounds (always, for
/// [`AttentionBudget::UNBOUNDED`]) this is exactly the caller's original single call. Otherwise the
/// query rows are split into blocks, each block attending over the **complete** k/v, and the block
/// outputs are concatenated back along the query axis.
///
/// `mask` is passed through to each chunk. A `[.., 1, Sk]` mask (or `None`) broadcasts identically
/// onto every chunk; a per-query `[.., Sq, Sk]` mask is narrowed to the chunk's own rows. Z-Image's
/// DiT is intentionally maskless, so the mask path exists for the other families that adopt this.
pub fn sdpa_budgeted_bhsd(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
    budget: AttentionBudget,
) -> Result<Array> {
    let qs = q.shape();
    if qs.len() != 4 {
        return Err(crate::Error::Msg(format!(
            "sdpa_budgeted_bhsd expects q as [B, H, Sq, D], got {qs:?}"
        )));
    }
    let (b, h, sq) = (qs[0], qs[1], qs[2]);
    let sk = k.shape()[2];

    let block = budget.query_block(b, h, sq, sk);
    if block >= sq {
        return Ok(scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
        )?);
    }

    let mut outs: Vec<Array> = Vec::with_capacity(((sq + block - 1) / block) as usize);
    let mut start = 0;
    while start < sq {
        let len = block.min(sq - start);
        let q_chunk = slice_axis(q, 2, start, len)?;
        // Narrow only a per-query mask; a broadcast `[.., 1, Sk]` mask is shared by every chunk.
        let chunk_mask = match mask {
            Some(m) if m.shape()[m.shape().len() - 2] == sq => {
                Some(slice_axis(m, (m.shape().len() - 2) as i32, start, len)?)
            }
            other => other.cloned(),
        };
        let out = scaled_dot_product_attention(
            &q_chunk,
            k,
            v,
            scale,
            chunk_mask
                .as_ref()
                .map(ScaledDotProductAttentionMask::Array),
            None,
        )?;
        if budget.eval_per_chunk {
            mlx_rs::transforms::eval([&out])?;
        }
        outs.push(out);
        start += len;
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 2)?)
}

/// Contiguous `[.., start..start+len, ..]` slice along `axis`, via boundary splits (no host index
/// vector and no gather, unlike `take_axis`).
fn slice_axis(a: &Array, axis: i32, start: i32, len: i32) -> Result<Array> {
    Ok(a.split_axis(&[start, start + len], axis)?.swap_remove(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    fn flat(a: &Array) -> Vec<f32> {
        a.reshape(&[-1])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    fn arange(shape: &[i32], scale: f32, offset: f32) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * scale + offset).sin())
            .collect();
        Array::from_slice(&data, shape)
    }

    /// Peak-relative max delta of the chunked forward against the unbounded one.
    fn chunked_delta(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Option<&Array>,
        budget: AttentionBudget,
    ) -> f32 {
        let full = sdpa_budgeted_bhsd(q, k, v, scale, mask, AttentionBudget::UNBOUNDED).unwrap();
        let chunked = sdpa_budgeted_bhsd(q, k, v, scale, mask, budget).unwrap();
        assert_eq!(chunked.shape(), full.shape());
        let (a, c) = (flat(&full), flat(&chunked));
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        a.iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / peak
    }

    /// The documented numerical-equivalence bound (SC-15615; the CUDA twin is `candle_gen_z_image`'s
    /// `attention_query_chunking_matches_the_unbounded_forward`).
    ///
    /// Query-row chunking does not change the math — each query still attends over the complete
    /// k/v — but it *does* change which Metal kernel specialization MLX picks for the
    /// `[.., block, Sk]` matmul, and MLX's Metal matmul runs in reduced precision (the sc-2338
    /// parity class). So the agreement is **numerical, not bit-exact in general**: measured on
    /// Apple M5 Max it is exactly `0` for some block sizes and ~1e-3 peak-relative for others, with
    /// no monotonic relationship to block size. At the *production* shape and budget it is exactly
    /// zero — see [`chunking_is_exact_at_the_production_shape_and_budget`].
    const PRODUCTION_TOLERANCE: f32 = 2e-3;

    #[test]
    fn unbounded_never_chunks() {
        let budget = AttentionBudget::UNBOUNDED;
        assert!(budget.is_unbounded());
        assert_eq!(budget.query_block(1, 30, 4128, 4128), 4128);
        assert_eq!(budget.max_score_elements(), u64::MAX);
        assert_eq!(AttentionBudget::default(), AttentionBudget::UNBOUNDED);
    }

    #[test]
    fn query_block_derives_the_declared_chunk_rows() {
        // The production Z-Image-turbo unified stack at 1024²: B=1, H=30, Sq=Sk=4128.
        // 64 Mi / (1·30·4128) = 541 rows.
        let budget = AttentionBudget::CONSTRAINED;
        assert_eq!(budget.query_block(1, 30, 4128, 4128), 541);
        // A call that already fits the budget is a single un-chunked pass.
        assert_eq!(budget.query_block(1, 30, 32, 32), 32);
        // Degenerate rows-per-query never divides by zero, and one row is the floor.
        let tiny = AttentionBudget::from_score_elements(1, false);
        assert_eq!(tiny.query_block(1, 30, 4128, 4128), 1);
        assert_eq!(tiny.query_block(1, 30, 1, 4128), 1);
        // A zero budget clamps to one row rather than dividing by zero or returning zero.
        let zero = AttentionBudget::from_score_elements(0, false);
        assert_eq!(zero.query_block(1, 30, 4128, 4128), 1);
    }

    #[test]
    fn chunked_matches_the_unbounded_forward_within_the_documented_tolerance() {
        // [B=1, H=3, Sq=Sk=37, D=8] — 37 is prime, so every chunk size leaves a ragged tail.
        let (b, h, s, d) = (1, 3, 37, 8);
        let q = arange(&[b, h, s, d], 0.017, 0.3);
        let k = arange(&[b, h, s, d], 0.013, -0.7);
        let v = arange(&[b, h, s, d], 0.011, 1.1);
        let scale = (d as f32).powf(-0.5);

        // Sweep block sizes by choosing budgets that yield 1, 5, 18, 37 (== the whole call) rows.
        for rows in [1i32, 5, 18, 37] {
            let budget = AttentionBudget::from_score_elements(
                (b * h * s * rows) as u64,
                rows % 2 == 1, // exercise both eval_per_chunk settings
            );
            assert_eq!(budget.query_block(b, h, s, s), rows, "rows {rows}");
            let max_rel = chunked_delta(&q, &k, &v, scale, None, budget);
            assert!(
                max_rel < PRODUCTION_TOLERANCE,
                "rows {rows}: max relative delta {max_rel:e} exceeds {PRODUCTION_TOLERANCE:e}"
            );
        }

        // Mutation check — the metric is not vacuously zero: perturbing `v` must move it far past
        // the tolerance, so a comparison that accidentally compared an array to itself fails here.
        let v_moved = mlx_rs::ops::add(&v, Array::from_slice(&[0.5f32], &[1])).unwrap();
        let full = sdpa_budgeted_bhsd(&q, &k, &v, scale, None, AttentionBudget::UNBOUNDED).unwrap();
        let moved =
            sdpa_budgeted_bhsd(&q, &k, &v_moved, scale, None, AttentionBudget::UNBOUNDED).unwrap();
        let (a, c) = (flat(&full), flat(&moved));
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        let moved_rel = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / peak;
        assert!(
            moved_rel > 100.0 * PRODUCTION_TOLERANCE,
            "mutation check: perturbing v moved the metric only {moved_rel:e}"
        );
    }

    /// At the **production** Z-Image-turbo DiT shape and the production 64 Mi budget the chunked and
    /// unbounded forwards agree **exactly** — the strongest form of the SC-15615 equivalence claim,
    /// pinned at the geometry that ships (1024² → 4128 unified tokens, 30 heads, head_dim 128, f32,
    /// the dtype the DiT stream runs at after `apply_pad`).
    #[test]
    fn chunking_is_exact_at_the_production_shape_and_budget() {
        let (b, h, s, d) = (1, 30, 4128, 128);
        let q = arange(&[b, h, s, d], 0.0007, 0.3);
        let k = arange(&[b, h, s, d], 0.0011, -0.7);
        let v = arange(&[b, h, s, d], 0.0013, 1.1);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::CONSTRAINED;
        assert_eq!(budget.query_block(b, h, s, s), 541);
        let max_rel = chunked_delta(&q, &k, &v, scale, None, budget);
        assert_eq!(
            max_rel, 0.0,
            "production-shape chunking must be exact, got {max_rel:e}"
        );
    }

    #[test]
    fn chunked_matches_with_a_broadcast_and_a_per_query_mask() {
        let (b, h, s, d) = (1, 2, 24, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * h * s * 7) as u64, false);
        assert_eq!(budget.query_block(b, h, s, s), 7);

        for mask in [
            arange(&[b, 1, 1, s], 0.31, 0.0),  // broadcast over query rows
            arange(&[b, 1, s, s], 0.29, -0.1), // per-query rows -> must be narrowed
        ] {
            let max_rel = chunked_delta(&q, &k, &v, scale, Some(&mask), budget);
            assert!(
                max_rel < PRODUCTION_TOLERANCE,
                "mask {:?}: max relative delta {max_rel:e}",
                mask.shape()
            );
        }
    }

    /// A per-query mask that is NOT narrowed per chunk is the classic chunking bug (every chunk would
    /// see row 0's mask). Pin that the narrowing actually happens by using a mask whose rows differ
    /// strongly: an un-narrowed implementation would blow past the tolerance.
    #[test]
    fn a_per_query_mask_is_narrowed_to_each_chunks_own_rows() {
        let (b, h, s, d) = (1, 2, 24, 8);
        let q = arange(&[b, h, s, d], 0.019, 0.2);
        let k = arange(&[b, h, s, d], 0.023, -0.4);
        let v = arange(&[b, h, s, d], 0.007, 0.9);
        let scale = (d as f32).powf(-0.5);
        let budget = AttentionBudget::from_score_elements((b * h * s * 7) as u64, false);

        // A strongly row-dependent additive mask: row i suppresses all keys except key i.
        let mut data = vec![-1.0e4f32; (b * s * s) as usize];
        for i in 0..s {
            data[(i * s + i) as usize] = 0.0;
        }
        let mask = Array::from_slice(&data, &[b, 1, s, s]);
        let max_rel = chunked_delta(&q, &k, &v, scale, Some(&mask), budget);
        assert!(
            max_rel < PRODUCTION_TOLERANCE,
            "row-dependent mask: max relative delta {max_rel:e}"
        );
    }

    /// The load-bearing MLX fact behind the whole SC-15615 finding: `fast::scaled_dot_product_attention`
    /// **streams** the `[B,H,Sq,Sk]` scores rather than materializing them, so there is no score
    /// tensor for query-row chunking to bound (unlike candle's `attention_basic`).
    ///
    /// Pinned by arithmetic, not by a magic constant: with q/k/v already evaluated, the unbounded
    /// call's peak-active bytes must be within one output's worth of `4 · B·H·Sq·D · sizeof(f32)`.
    /// A materialized score tensor would add `B·H·Sq·Sk · 4` — at this shape 1.9 GiB, ~8× the whole
    /// budget below — so the assertion cannot pass if MLX ever falls back to the decomposition.
    #[test]
    fn fused_sdpa_does_not_materialize_the_scores() {
        use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

        // The production Z-Image-turbo unified stack at 1024².
        let (b, h, s, d) = (1i64, 30i64, 4128i64, 128i64);
        let shape = [b as i32, h as i32, s as i32, d as i32];
        let q = arange(&shape, 0.0007, 0.3);
        let k = arange(&shape, 0.0011, -0.7);
        let v = arange(&shape, 0.0013, 1.1);
        mlx_rs::transforms::eval([&q, &k, &v]).unwrap();
        clear_cache();
        reset_peak_memory();

        let out = sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            (d as f32).powf(-0.5),
            None,
            AttentionBudget::UNBOUNDED,
        )
        .unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        let peak = get_peak_memory() as i64;

        let one_operand = b * h * s * d * 4; // f32
        let scores = b * h * s * s * 4;
        assert!(
            peak <= 5 * one_operand,
            "unbounded SDPA peaked at {peak} B, more than q+k+v+out ({} B) plus one operand of \
             slack — MLX appears to be materializing the {scores} B score tensor, which would make \
             query-row chunking a real memory rung on this backend. Re-measure SC-15615.",
            4 * one_operand
        );
        drop(out);
        clear_cache();
    }

    #[test]
    fn rejects_a_non_4d_query() {
        let q = arange(&[2, 3, 4], 0.1, 0.0);
        let err = sdpa_budgeted_bhsd(&q, &q, &q, 1.0, None, AttentionBudget::UNBOUNDED)
            .unwrap_err()
            .to_string();
        assert!(err.contains("[B, H, Sq, D]"), "{err}");
    }
}
