//! Rung 3 of the SC-15448 memory-strategy ladder: the backend-neutral **bounded-attention planner**.
//!
//! This is the third planner to move here, and it follows the same split the other two already use:
//!
//! | rung | planner (here) | kernel (per backend) |
//! |---|---|---|
//! | 2 bounded decode | [`crate::tiling`] | each backend's tiled VAE decode |
//! | 3 bounded attention | **this module** | `mlx_gen::attention::sdpa_budgeted_bhsd`, candle's `sdpa_budgeted_*` |
//! | 4 bounded transformer residency | [`crate::block_window`] | each backend's `BlockWindowBackend` impl |
//!
//! A rung decomposes into a **planner** — measured constants plus pure arithmetic, no tensors — and a
//! **kernel**, which is genuinely per-backend. Only the kernel needs duplicating. This module owns the
//! planner half of rung 3 so a family adopting bounded attention on any backend derives its chunk
//! boundaries from one place, and a budget declared in `strategyParameters` means the same thing in
//! every backend's calibration evidence.
//!
//! ## The lever
//!
//! Every query row's softmax is over **all** keys and is independent of the other rows. So attention
//! can run over blocks of query rows with each query's complete key/value domain intact, bounding the
//! per-call score domain to `block × Sk` instead of `Sq × Sk`. Precision, schedule, seed, conditioning
//! and the output contract are untouched — this is a quality-preserving rung, like every other one.
//!
//! [`AttentionBudget::query_block_rows`] and [`AttentionBudget::head_chunks`] are the planner. The
//! former narrows query rows; the latter narrows complete heads before a kernel considers query rows.
//! They share a score budget, but **not a numerical contract**: head chunking preserves the query
//! GEMM's `M` dimension and therefore its accumulation shape, while query-row chunking changes `M` and
//! may move results by a few ULP. A kernel may prefer heads for bit identity and use query rows only
//! when one head still exceeds the budget; callers must not treat the axes as interchangeable.
//!
//! ## The kernels must NOT merge — the value differs by an order of magnitude
//!
//! This is measured, not assumed, and it is the reason only the planner lives here:
//!
//! | backend | mechanism | denoise-phase saving | whole-request saving |
//! |---|---|---:|---:|
//! | candle (CPU/CUDA) | `attention_basic` **materializes** `[B,H,Sq,Sk]`; chunking bounds that tensor | **−32%** (8.394 → 5.709 GB, SC-15256) | not measured |
//! | MLX (Metal) | fused SDPA never builds the score tensor; the saving is the per-chunk `eval` cutting the lazy graph | **−1.7%** (4.7746 → 4.6944 GiB, SC-15615) | **−5.0%** (4.898 → 4.653 GiB) |
//!
//! **Compare like with like.** The two backends' headline numbers are quoted at different scopes in
//! the source material — candle's −32% is the denoise phase, MLX's −5.0% is the whole request — and
//! the honest comparison is the denoise column: **−32% vs −1.7%, roughly 19x**, not the ~6x a
//! cross-scope reading suggests. Note also that the units differ as published (GB vs GiB). A backend
//! adopting this planner owes its own measurement at a stated scope; neither magnitude may be inferred
//! from the other. The general rule this epic keeps re-learning: carry a twin's *shape* across, never
//! its mechanism — and never its magnitude.
//!
//! ## Shapes this planner covers
//!
//! `rows_per_query` is the score-element count contributed by ONE query row, so the same arithmetic
//! serves every attention layout in the tree without a per-shape planner:
//!
//! - 4-D DiT `[B, H, Sq, D]` → `rows_per_query = B·H·Sk` ([`AttentionBudget::query_block`])
//! - 3-D heads-folded `[N, Sq, D]` where `N = B·H` (SDXL) → `rows_per_query = N·Sk`
//! - 3-D single-head VAE spatial attention where `N = B`, `Sq = H·W` → `rows_per_query = N·Sk`
//!
//! It is also budget-agnostic: the memory rung's [`CONSTRAINED_ATTN_SCORES_BUDGET`] and candle's
//! larger `ATTN_SCORES_BUDGET` i32-overflow guard (sc-9116) are the same planner at two settings, and
//! [`CROSS_BACKEND_CHUNK_CASES`] pins both.

/// The **Z-Image** constrained score budget: 64 Mi elements per attention call.
///
/// Measured as the rung-3 operating point for Z-Image on both backends independently — the
/// Candle/CUDA port (SC-15256) and the MLX port (SC-15615) landed on the same value. It is published
/// as `strategyParameters.chunkedAttention.attentionChunkSize`.
///
/// **This is a family operating point, not a contract-wide constant.** The planner is deliberately
/// budget-agnostic ([`AttentionBudget::from_score_elements`]) and other families legitimately declare
/// other values — candle's Krea lane measured and publishes 128 Mi, and candle's i32-overflow guard
/// (sc-9116) runs the same planner at 1e9. What must be shared is the **planner**, so that a given
/// declared budget names the same chunking everywhere; the budget itself is per-family evidence.
pub const CONSTRAINED_ATTN_SCORES_BUDGET: u64 = 64 * 1024 * 1024;

/// How much attention scratch one call may build at once.
///
/// [`UNBOUNDED`](Self::UNBOUNDED) is the default everywhere and means a single un-chunked call —
/// byte-identical to the pre-rung forward. A bounded budget is only ever selected by a *measured*
/// rung, never by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBudget {
    /// Max elements in one chunk's `[.., chunk_rows, Sk]` score domain. `u64::MAX` ⇒ unbounded.
    max_score_elements: u64,
    /// Whether the kernel should materialize each chunk's output before building the next.
    ///
    /// This is a **declared hint to the kernel**, not an obligation this planner can discharge, and
    /// what it costs or buys is a per-backend measurement. On a lazy backend it cuts the graph at each
    /// chunk boundary and is where the entire MLX-side saving comes from (SC-15615: without it,
    /// chunking measured −0.08%, i.e. noise). An eager backend has nothing to force and may ignore it.
    /// Do not read MLX's reason for setting it as a general one — see the module doc.
    eval_per_chunk: bool,
}

/// The shared head-axis decision for a 4-D `[B, H, Sq, Sk]` score domain.
///
/// `chunks_heads == false` is semantically important even when `heads_per_chunk == H`: it identifies
/// the in-budget and single-head fast paths, which should delegate once instead of entering a
/// provider's head-splitting loop. Keeping that decision in the planner makes both early branches
/// mutation-testable without reintroducing local budget arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadChunkPlan {
    heads_per_chunk: u64,
    chunks_heads: bool,
}

impl HeadChunkPlan {
    /// Complete heads to send to each kernel call.
    pub const fn heads_per_chunk(self) -> u64 {
        self.heads_per_chunk
    }

    /// Whether the caller must enter its head-axis chunk loop.
    pub const fn chunks_heads(self) -> bool {
        self.chunks_heads
    }
}

impl AttentionBudget {
    /// No bound — one un-chunked call, byte-identical to the un-guarded forward. The default for every
    /// path that has not selected the bounded-attention rung, including all training paths.
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

    /// A budget of exactly `max_score_elements` scores per chunk. A `0` budget yields one query row per
    /// chunk rather than dividing by zero.
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

    /// Whether the kernel should materialize each chunk before building the next. See the field doc:
    /// this is a hint whose value is per-backend.
    pub const fn eval_per_chunk(self) -> bool {
        self.eval_per_chunk
    }

    /// `true` when this budget can never chunk (the fast path).
    pub const fn is_unbounded(self) -> bool {
        self.max_score_elements == u64::MAX
    }

    /// **The planner.** The largest query-block length whose score count stays within the budget,
    /// given `rows_per_query` — the score elements contributed by ONE query row (`B·H·Sk` for the 4-D
    /// DiT shape, `N·Sk` for a heads-folded or single-head flat shape).
    ///
    /// Returns the whole `sq` — a single un-chunked call — when the full call already fits, so an
    /// in-budget shape keeps the byte-identical unguarded path. Otherwise the result is in `1..sq`.
    ///
    /// Saturating throughout: `rows_per_query · sq` overflows `u64` only for shapes that cannot exist,
    /// but saturating there yields "chunk" rather than a wrapped comparison that yields "fits".
    pub const fn query_block_rows(self, rows_per_query: u64, sq: u64) -> u64 {
        if self.is_unbounded() || sq <= 1 {
            return sq;
        }
        if rows_per_query.saturating_mul(sq) <= self.max_score_elements {
            return sq;
        }
        // Unreachable when `rows_per_query` is 0 (the branch above returns `sq`), but the guard keeps
        // this total rather than relying on that.
        let divisor = if rows_per_query == 0 {
            1
        } else {
            rows_per_query
        };
        let block = self.max_score_elements / divisor;
        if block == 0 {
            1
        } else if block > sq {
            sq
        } else {
            block
        }
    }

    /// Plan complete-head chunks for `q/k/v: [B, H, S, D]` before query-row fallback.
    ///
    /// The largest chunk fits `B · heads_per_chunk · Sq · Sk` score elements in this budget. A call
    /// that already fits, an unbounded call, an empty score domain, or `H <= 1` returns the whole head
    /// axis with [`HeadChunkPlan::chunks_heads`] false. Otherwise the boundary is in `1..H`.
    ///
    /// This is deliberately separate from [`query_block_rows`](Self::query_block_rows). Splitting
    /// heads keeps every call's full `Sq` and the query GEMM's `M` dimension, preserving the
    /// accumulation shape and enabling bit-exact reconstruction. Query-row fallback changes `M` and
    /// is only tolerance-equivalent. Providers that need that distinction should consume this method
    /// first, then apply the query-row planner inside each planned head chunk only if necessary.
    pub const fn head_chunks(self, b: u64, h: u64, sq: u64, sk: u64) -> HeadChunkPlan {
        if self.is_unbounded() || h <= 1 {
            return HeadChunkPlan {
                heads_per_chunk: h,
                chunks_heads: false,
            };
        }
        let scores_per_head = b.saturating_mul(sq).saturating_mul(sk);
        if scores_per_head.saturating_mul(h) <= self.max_score_elements {
            return HeadChunkPlan {
                heads_per_chunk: h,
                chunks_heads: false,
            };
        }
        let divisor = if scores_per_head == 0 {
            1
        } else {
            scores_per_head
        };
        let block = self.max_score_elements / divisor;
        let heads_per_chunk = if block == 0 {
            1
        } else if block > h {
            h
        } else {
            block
        };
        HeadChunkPlan {
            heads_per_chunk,
            chunks_heads: true,
        }
    }

    /// [`query_block_rows`](Self::query_block_rows) for the 4-D DiT shape `q: [B, H, Sq, D]`,
    /// `k/v: [B, H, Sk, D]` — the layout both backends' DiT attention uses.
    ///
    /// This is a pure adapter: it computes `rows_per_query` and delegates. It **must not** apply
    /// arithmetic of its own, or a backend calling this and a backend calling the core would plan
    /// different boundaries from the same budget — the exact divergence this module exists to prevent.
    /// `the_bhsd_adapter_agrees_with_the_arithmetic_core` pins that, degenerate dims included.
    ///
    /// A non-positive dim therefore floors at `0`, not `1`: it makes the score domain genuinely empty,
    /// and the core's answer for an empty domain is the whole `sq` — a single un-chunked call, which is
    /// the correct plan for a call with nothing to bound. (Flooring at `1` instead would fabricate a
    /// non-empty domain and return a 1-row plan, which is also what a raw `B·H·Sk` caller like candle
    /// would *not* produce.)
    pub fn query_block(self, b: i32, h: i32, sq: i32, sk: i32) -> i32 {
        if self.is_unbounded() || sq <= 1 {
            return sq;
        }
        let rows_per_query = (b.max(0) as u64)
            .saturating_mul(h.max(0) as u64)
            .saturating_mul(sk.max(0) as u64);
        // `sq > 1` here, so the cast is non-negative, and the result is bounded above by `sq`.
        self.query_block_rows(rows_per_query, sq as u64) as i32
    }
}

impl Default for AttentionBudget {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

/// A budget plus the request's cancel flag — everything one bounded attention call needs that is not
/// a tensor.
///
/// Hoisted with the planner (SC-15793) for the same reason rung 4 carries its cancellation contract in
/// [`crate::block_window`] rather than per backend: **bounded attention is where cancellation becomes
/// possible at all inside a transformer forward.** A bounded call splits one previously atomic kernel
/// launch into N, and those chunk boundaries are the only place a cancel can land between steps. That
/// obligation is a property of the rung, not of a backend, so a backend adopting rung 3 should inherit
/// it rather than re-derive it.
///
/// The contract a kernel owes:
/// - check the flag **between** chunks, before launching the next one;
/// - on a tripped flag return the backend's typed cancellation error — never a stringified message,
///   which reports a cancelled render as a *failed* job (sc-4481);
/// - leave the unbounded fast path untouched. It has no chunk boundary, so it has nothing to check,
///   and cancellation there stays at step granularity exactly as before.
///
/// `Copy`, so it threads through a forward chain as cheaply as the budget alone.
#[derive(Clone, Copy, Debug)]
pub struct AttentionPlan<'a> {
    /// How much attention scratch one call may build at once.
    pub budget: AttentionBudget,
    /// Checked between chunks; `None` on paths with no request scope.
    pub cancel: Option<&'a crate::runtime::CancelFlag>,
}

impl AttentionPlan<'_> {
    /// The unbounded, uncancellable plan — the default for every path that has not selected the
    /// bounded-attention rung, and byte-identical to the pre-rung forward.
    pub const UNBOUNDED: Self = Self {
        budget: AttentionBudget::UNBOUNDED,
        cancel: None,
    };

    /// A plan at `budget` with no cancel flag (unit tests, and callers with no request scope).
    pub const fn budgeted(budget: AttentionBudget) -> Self {
        Self {
            budget,
            cancel: None,
        }
    }

    /// This plan with `cancel` attached.
    pub fn with_cancel(self, cancel: &crate::runtime::CancelFlag) -> AttentionPlan<'_> {
        AttentionPlan {
            budget: self.budget,
            cancel: Some(cancel),
        }
    }

    /// `true` when a cancel flag is attached and has been tripped — the between-chunks check a kernel
    /// makes before launching the next chunk.
    pub fn is_cancelled(self) -> bool {
        self.cancel.is_some_and(|c| c.is_cancelled())
    }
}

impl Default for AttentionPlan<'_> {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

impl From<AttentionBudget> for AttentionPlan<'_> {
    fn from(budget: AttentionBudget) -> Self {
        Self::budgeted(budget)
    }
}

/// One shape/budget row of [`CROSS_BACKEND_CHUNK_CASES`].
#[derive(Clone, Copy, Debug)]
pub struct ChunkBoundaryCase {
    /// What this row exercises, quoted in assertion failures.
    pub what: &'static str,
    /// Score elements contributed by one query row: `B·H·Sk` (4-D) or `N·Sk` (flat).
    pub rows_per_query: u64,
    /// Query rows in the call.
    pub sq: u64,
    /// The declared budget.
    pub budget: u64,
    /// The query-block length the planner must produce. `== sq` means a single un-chunked call.
    pub expect_rows: u64,
}

/// The cross-backend conformance table: **the shapes on which any two backends running rung 3 must
/// agree, and the boundary each must produce.**
///
/// This is the property that makes the epic's per-backend evidence comparable. A backend adopting the
/// rung asserts its own planner against this table, so "64 Mi budget" names one chunking everywhere
/// rather than one per implementation. This module's `the_planner_reproduces_every_cross_backend_case`
/// pins the table to the planner's real arithmetic, so the table cannot drift into asserting itself.
///
/// Deliberately spans both the memory rung's budget and candle's much larger i32-overflow guard
/// (sc-9116), and both the 4-D and heads-folded flat layouts — one planner has to cover all of it, or
/// it is not the shared planner.
pub const CROSS_BACKEND_CHUNK_CASES: &[ChunkBoundaryCase] = &[
    ChunkBoundaryCase {
        // The production Z-Image-turbo unified stack at 1024²: B=1, H=30, Sq=Sk=4128.
        // 64 Mi / (1·30·4128) = 541 rows — the boundary BOTH backends' rung-3 evidence was taken at.
        what: "z-image-turbo 1024² DiT @ 64 Mi",
        rows_per_query: 30 * 4128,
        sq: 4128,
        budget: CONSTRAINED_ATTN_SCORES_BUDGET,
        expect_rows: 541,
    },
    ChunkBoundaryCase {
        // 2048²: the same model at the top of its advertised envelope.
        what: "z-image 2048² DiT @ 64 Mi",
        rows_per_query: 30 * 16416,
        sq: 16416,
        budget: CONSTRAINED_ATTN_SCORES_BUDGET,
        expect_rows: 136,
    },
    ChunkBoundaryCase {
        // An in-budget call must stay a SINGLE un-chunked pass on every backend. Without a case like
        // this the table would tolerate a planner that chunks unconditionally.
        what: "small call inside the budget — must not chunk",
        rows_per_query: 30 * 32,
        sq: 32,
        budget: CONSTRAINED_ATTN_SCORES_BUDGET,
        expect_rows: 32,
    },
    ChunkBoundaryCase {
        // Heads-folded flat layout (SDXL folds heads into the batch dim): N = B·H = 2·10, Sq = Sk =
        // 16384. Proves the planner is layout-agnostic, not DiT-shaped.
        what: "sdxl 2048² heads-folded flat @ 1e9 i32 guard",
        rows_per_query: 2 * 10 * 16384,
        sq: 16384,
        budget: 1_000_000_000,
        expect_rows: 3051,
    },
    ChunkBoundaryCase {
        // Single-head VAE spatial attention: N = B = 1, Sq = Sk = H·W = 65536.
        what: "vae mid-block 2048² single-head @ 1e9 i32 guard",
        rows_per_query: 65536,
        sq: 65536,
        budget: 1_000_000_000,
        expect_rows: 15258,
    },
    ChunkBoundaryCase {
        // The same guard budget below its threshold: a 1024² VAE stays one un-chunked pass.
        what: "vae mid-block 1024² single-head @ 1e9 i32 guard — must not chunk",
        rows_per_query: 16384,
        sq: 16384,
        budget: 1_000_000_000,
        expect_rows: 16384,
    },
    ChunkBoundaryCase {
        // The floor: a budget too small for even one row yields one row, never zero.
        what: "budget below one row — floors at a single row",
        rows_per_query: 30 * 4128,
        sq: 4128,
        budget: 1,
        expect_rows: 1,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The table states boundaries as literals; this pins those literals to the planner's real
    /// arithmetic. Without it the table would be a second implementation asserting itself, and both
    /// backends could agree on a wrong number.
    #[test]
    fn the_planner_reproduces_every_cross_backend_case() {
        for case in CROSS_BACKEND_CHUNK_CASES {
            let budget = AttentionBudget::from_score_elements(case.budget, false);
            let got = budget.query_block_rows(case.rows_per_query, case.sq);
            assert_eq!(
                got, case.expect_rows,
                "{}: planner produced {got} rows, table declares {}",
                case.what, case.expect_rows
            );
            // Every boundary must be a usable chunk length: at least one row, never more than the call.
            assert!(
                (1..=case.sq).contains(&got),
                "{}: boundary {got} outside 1..={}",
                case.what,
                case.sq
            );
            // And the chunk it plans must actually fit the budget it was planned against — the whole
            // point of the rung. (An un-chunked in-budget call trivially satisfies this too.)
            assert!(
                case.rows_per_query.saturating_mul(got) <= case.budget.max(case.rows_per_query),
                "{}: a {got}-row chunk exceeds the {} budget",
                case.what,
                case.budget
            );
        }
    }

    /// Mutation check on the table: it must be sensitive to the budget it declares. If every row
    /// happened to be an un-chunked pass, or the planner ignored its budget, halving the budget would
    /// change nothing and the table would be worthless as a conformance gate.
    #[test]
    fn halving_the_budget_moves_the_chunking_cases() {
        for case in CROSS_BACKEND_CHUNK_CASES {
            let full = AttentionBudget::from_score_elements(case.budget, false)
                .query_block_rows(case.rows_per_query, case.sq);
            let half = AttentionBudget::from_score_elements(case.budget / 2, false)
                .query_block_rows(case.rows_per_query, case.sq);
            assert!(
                half <= full,
                "{}: halving the budget grew the block ({full} -> {half})",
                case.what
            );
            // Per-case rule rather than a hand-counted total: any row that actually chunks, and whose
            // halved budget still affords more than the one-row floor, MUST shrink. A planner that
            // ignored its budget would return the same block for both and fail here. Rows that fit
            // un-chunked, or that already floor at one row, are exempt by construction — so this stays
            // correct if the table gains or loses rows.
            let chunks = full < case.sq;
            let above_the_floor = half > 1;
            if chunks && above_the_floor {
                assert!(
                    half < full,
                    "{}: halving the budget left the block at {full} — the planner is not reading it",
                    case.what
                );
            }
        }
    }

    #[test]
    fn unbounded_never_chunks() {
        let budget = AttentionBudget::UNBOUNDED;
        assert!(budget.is_unbounded());
        assert_eq!(budget.max_score_elements(), u64::MAX);
        assert!(!budget.eval_per_chunk());
        assert_eq!(budget.query_block(1, 30, 4128, 4128), 4128);
        assert_eq!(budget.query_block_rows(30 * 4128, 4128), 4128);
        assert_eq!(
            budget.head_chunks(1, 48, 4096, 4096),
            HeadChunkPlan {
                heads_per_chunk: 48,
                chunks_heads: false,
            }
        );
        assert_eq!(AttentionBudget::default(), AttentionBudget::UNBOUNDED);
    }

    #[test]
    fn head_axis_plans_literal_krea_boundaries_and_fast_paths() {
        let krea = AttentionBudget::from_score_elements(128 * 1024 * 1024, false);

        // Krea's native 1024² image grid has 4096 tokens and 48 DiT heads: 128 Mi fits eight heads.
        let at_1024 = krea.head_chunks(1, 48, 4096, 4096);
        assert!(at_1024.chunks_heads());
        assert_eq!(at_1024.heads_per_chunk(), 8);

        // The shipping real-caption harness records about 4118 combined tokens; caption context moves
        // the literal production boundary down by one head rather than being silently ignored.
        let with_caption = krea.head_chunks(1, 48, 4118, 4118);
        assert!(with_caption.chunks_heads());
        assert_eq!(with_caption.heads_per_chunk(), 7);

        // At 2048², one 16384² head already exceeds 128 Mi, so query rows must be the fallback.
        let at_2048 = krea.head_chunks(1, 48, 16384, 16384);
        assert!(at_2048.chunks_heads());
        assert_eq!(at_2048.heads_per_chunk(), 1);

        // The two fast paths are distinct planner decisions, not consequences of a clamped boundary.
        let in_budget = krea.head_chunks(1, 48, 32, 32);
        assert!(!in_budget.chunks_heads());
        assert_eq!(in_budget.heads_per_chunk(), 48);
        let one_head_over_budget = krea.head_chunks(1, 1, 16384, 16384);
        assert!(!one_head_over_budget.chunks_heads());
        assert_eq!(one_head_over_budget.heads_per_chunk(), 1);
    }

    #[test]
    fn constrained_declares_the_production_budget_and_forces_materialization() {
        let budget = AttentionBudget::CONSTRAINED;
        assert!(!budget.is_unbounded());
        assert_eq!(budget.max_score_elements(), CONSTRAINED_ATTN_SCORES_BUDGET);
        assert_eq!(CONSTRAINED_ATTN_SCORES_BUDGET, 64 * 1024 * 1024);
        // Non-default field: a test that only asserted `UNBOUNDED`'s `false` would pass with the flag
        // ripped out.
        assert!(budget.eval_per_chunk());
    }

    /// The 4-D adapter must agree with the arithmetic core it delegates to — otherwise a backend using
    /// the BHSD convenience and one computing `B·H·Sk` itself and calling the core (which is exactly
    /// what candle's attention does) would plan different boundaries from the same budget. That is the
    /// divergence this module exists to prevent, so the degenerate rows below are the point of this
    /// test, not padding: an adapter that floored its dims at 1 would pass the ordinary rows and fail
    /// every zero row.
    #[test]
    fn the_bhsd_adapter_agrees_with_the_arithmetic_core() {
        for budget in [
            AttentionBudget::CONSTRAINED,
            AttentionBudget::from_score_elements(1_000_000_000, false),
            AttentionBudget::from_score_elements(4096, false),
            AttentionBudget::from_score_elements(1, false),
            AttentionBudget::from_score_elements(0, false),
        ] {
            for &(b, h, sq, sk) in &[
                (1i32, 30i32, 4128i32, 4128i32),
                (2, 24, 16384, 16384),
                (1, 16, 11585, 11585),
                (1, 1, 64, 4096),
                (1, 30, 32, 32),
                // Degenerate dims: an empty score domain must plan identically both ways.
                (0, 30, 4128, 4128),
                (1, 0, 4128, 4128),
                (1, 30, 4128, 0),
                (0, 0, 4128, 0),
            ] {
                let via_adapter = budget.query_block(b, h, sq, sk);
                let via_core = budget
                    .query_block_rows((b as u64) * (h as u64) * (sk as u64), sq as u64)
                    as i32;
                assert_eq!(
                    via_adapter,
                    via_core,
                    "budget {} at [B={b}, H={h}, Sq={sq}, Sk={sk}]",
                    budget.max_score_elements()
                );
            }
        }
    }

    #[test]
    fn degenerate_inputs_never_divide_by_zero_or_return_a_zero_block() {
        let tiny = AttentionBudget::from_score_elements(1, false);
        assert_eq!(tiny.query_block(1, 30, 4128, 4128), 1);
        assert_eq!(tiny.query_block(1, 30, 1, 4128), 1);
        // A zero budget clamps to one row rather than dividing by zero or returning zero.
        let zero = AttentionBudget::from_score_elements(0, false);
        assert_eq!(zero.query_block(1, 30, 4128, 4128), 1);
        // An empty score domain has nothing to bound, so the plan is the whole call — and a
        // non-positive dim reaches that same answer through the adapter, which is what keeps the
        // adapter and the core (and therefore any two backends) in agreement.
        assert_eq!(zero.query_block_rows(0, 4128), 4128);
        assert_eq!(zero.query_block(0, 0, 4128, 0), 4128);
        assert_eq!(zero.query_block(-3, 30, 4128, 4128), 4128);
        // A single query row (or none) is always the whole call.
        assert_eq!(tiny.query_block_rows(u64::MAX, 1), 1);
        assert_eq!(tiny.query_block_rows(u64::MAX, 0), 0);
        // Saturating: a product that would overflow u64 must read as "does not fit", not wrap to
        // "fits" and return an un-chunked plan.
        assert_eq!(tiny.query_block_rows(u64::MAX, u64::MAX - 1), 1);
    }
}
