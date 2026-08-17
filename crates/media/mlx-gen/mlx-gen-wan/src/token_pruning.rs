//! **Token pruning for the dense Wan DiT** — the second sc-18322 approximate mechanism, behind the
//! same contract as [`crate::feature_cache`] and under the same rules.
//!
//! # What is dropped, and what is deliberately *not*
//!
//! A pruned step evaluates each block's **per-token work** on a strided subset of the latent tokens and
//! lets the rest pass through the block unchanged. Wan's block is three residual adds
//! (`transformer::Block::forward`), every one of which is either gated by an adaLN gate or a plain
//! `add`, so a token whose sublayer contributions are all zero emerges **exactly** as it entered — the
//! pass-through is bit-exact by the residual structure, not by an approximation of it.
//!
//! What is emphatically **not** pruned is the self-attention **key/value** side. Keys and values are
//! projected from the full token stream, so every surviving query still attends over *every* token. That
//! is the structural property that bounds the degradation: pruning removes updates, never information.
//! Dropping keys as well would change what the surviving tokens see, which is a categorically stronger
//! approximation and is not what this mechanism does.
//!
//! Cross-attention is unaffected on the key side for free: its K/V come from the text context and are
//! built once per generate ([`crate::pipeline`]'s `StepCache`), independent of the latent token count.
//!
//! # No new kernel, and no data-dependent shape
//!
//! The gather is `take_axis` with a host-built index vector — the idiom this crate already uses for its
//! sequence slicing (`chunk::slice_axis0`) and its RoPE halves (`rope::rope_apply`). The restore is a
//! `concatenate_axis` + `take_axis` round-trip, the same primitive-free scatter `mlx-gen-bernini`'s
//! `mar::scatter_rows` uses. Nothing here needs a Metal kernel that does not already ship.
//!
//! The keep-set is **positional**: it comes from
//! [`TokenDropStride::keeps_token`](mlx_gen::gen_core::TokenDropStride::keeps_token) and the token
//! count, never from tensor values. So the surviving count is known before the forward, no statistic is
//! read back from the accelerator mid-forward (which would be a full GPU sync per block per step), and
//! the shape entering each block is a function of the request geometry alone. Wan compiles its
//! elementwise glue `shapeless`, so a new token count is a compile-cache hit rather than a retrace.
//!
//! # Byte-identical when off
//!
//! Threaded as `Option<&TokenKeepSet>`. With `None` — every production caller — `Block::forward` runs
//! unchanged and nothing in this module is constructed or called. The pruned path is a *separate* block
//! entry point (`Block::forward_pruned`), so the exact path's instruction sequence is not merely
//! equivalent to what it was, it is the same code.
//!
//! # Restored before the block returns, which is what keeps everything downstream valid
//!
//! Each block restores the full token count before returning. That is not incidental tidiness: the
//! packed Bernini forward unpacks its target tokens by **position** (`total - l_t .. total`),
//! `unpatchify` needs all `L` rows, `apply_head` broadcasts a modulation against the token axis, and
//! [`crate::feature_cache`]'s retained residual is shape-guarded against the live stream. Restoring
//! inside the block leaves every one of those assumptions untouched, so the two approximate mechanisms
//! compose without either knowing about the other.
//!
//! # Not reachable from a request
//!
//! Same two locks as the feature cache: the contract refuses every approximate selection until a
//! characterization artifact family exists, and [`TokenKeepSet`]'s construction from a plan is
//! `#[cfg(test)]` at the provider bridge.

use mlx_rs::ops::{concatenate_axis, zeros_dtype};
use mlx_rs::Array;

use mlx_gen::gen_core::TokenDropStride;
use mlx_gen::Error;

type Result<T> = std::result::Result<T, Error>;

/// The positional keep-set for one forward: which token rows survive, and how to put them back.
///
/// Built once per forward from the stride and the token count — cheap (two `i32` vectors of length `L`,
/// ~148 KiB at the 5B's production geometry) and shared by every block of that forward, so the gather
/// index is not rebuilt 30 times.
#[derive(Debug)]
pub struct TokenKeepSet {
    /// Total tokens in the unpruned stream.
    total: i32,
    /// Surviving row indices, `[kept]` i32 — the `take_axis` gather index.
    keep: Array,
    /// Restore map, `[total]` i32. Entry `i` is the surviving row's position for a kept token, and
    /// `kept` — one past the end, pointing at the appended pass-through row block — for a dropped one.
    /// See [`restore`](Self::restore).
    restore: Array,
    /// Surviving token count.
    kept: i32,
}

impl TokenKeepSet {
    /// The keep-set `stride` selects out of `total` tokens.
    ///
    /// Both index vectors come from the contract's single
    /// [`keeps_token`](TokenDropStride::keeps_token) decision, so the gather and the restore cannot
    /// disagree about which rows survived — a disagreement that would silently write one token's update
    /// into another's row rather than failing.
    pub fn new(stride: TokenDropStride, total: usize) -> Result<Self> {
        if total == 0 {
            return Err(Error::Msg(
                "wan: token pruning needs at least one token".into(),
            ));
        }
        let mut keep: Vec<i32> = Vec::with_capacity(stride.kept_count(total));
        let mut restore: Vec<i32> = Vec::with_capacity(total);
        for index in 0..total {
            if stride.keeps_token(index) {
                restore.push(keep.len() as i32);
                keep.push(index as i32);
            } else {
                // Sentinel: the pass-through row appended by `restore`.
                restore.push(-1);
            }
        }
        let kept = keep.len() as i32;
        if kept == 0 {
            // Unreachable through `TokenDropStride`, which refuses a stride of 1 — the only stride that
            // could empty the keep-set. Refusing anyway keeps the invariant local: an empty gather
            // would make the block the identity, and MLX would report a shape error somewhere far from
            // the cause.
            return Err(Error::Msg(format!(
                "wan: a drop stride of {} leaves no tokens out of {total}",
                stride.stride()
            )));
        }
        // Resolve the sentinel now that `kept` is known.
        for slot in restore.iter_mut() {
            if *slot < 0 {
                *slot = kept;
            }
        }
        Ok(Self {
            total: total as i32,
            keep: Array::from_slice(&keep, &[kept]),
            restore: Array::from_slice(&restore, &[total as i32]),
            kept,
        })
    }

    /// Surviving token count.
    pub fn kept(&self) -> i32 {
        self.kept
    }

    /// Total token count this keep-set was built for.
    pub fn total(&self) -> i32 {
        self.total
    }

    /// Gather the surviving rows of a `[.., total, ..]` tensor along `axis`.
    ///
    /// `take_axis` is a **clamping** gather in MLX — an out-of-range index reads the last row instead of
    /// erroring, which the workspace has been bitten by before (silently-garbage seeded KV) — so the
    /// axis length is checked here rather than trusted. The index vector itself is in range by
    /// construction.
    pub fn gather(&self, x: &Array, axis: i32) -> Result<Array> {
        let len = x.shape()[axis as usize];
        if len != self.total {
            return Err(Error::Msg(format!(
                "wan: token pruning keep-set is for {} tokens but axis {axis} of this tensor is {len}; \
                 take_axis would clamp rather than fail, so the mismatch is refused here",
                self.total
            )));
        }
        Ok(x.take_axis(&self.keep, axis)?)
    }

    /// Scatter a `[B, kept, dim]` sublayer output back to `[B, total, dim]`, with **exact zeros** in the
    /// dropped rows.
    ///
    /// Zero is the value that makes the pass-through bit-exact: every one of Wan's block residuals is
    /// `x + gate·y` or `x + y`, so a dropped row's `y` of zero leaves `x` untouched at the bit level.
    /// This is also why the scatter happens *after* each sublayer's output projection rather than before
    /// it — the projections carry biases, so a zero-filled input would emerge as the bias, not as zero.
    ///
    /// Implemented as one `concatenate_axis` + one `take_axis`, the primitive-free scatter idiom: append
    /// a single zero row to the computed block, then gather with the restore map, whose dropped entries
    /// all point at that row.
    pub fn restore(&self, y: &Array) -> Result<Array> {
        let shape = y.shape();
        if shape.len() != 3 {
            return Err(Error::Msg(format!(
                "wan: token pruning restore expects [batch, kept, dim], got {shape:?}"
            )));
        }
        if shape[1] != self.kept {
            return Err(Error::Msg(format!(
                "wan: token pruning restore expects {} kept rows, got {}",
                self.kept, shape[1]
            )));
        }
        let pad = zeros_dtype(&[shape[0], 1, shape[2]], y.dtype())?;
        let stacked = concatenate_axis(&[y, &pad], 1)?;
        Ok(stacked.take_axis(&self.restore, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{ApproximationPlan, TokenPruningPolicy};
    use mlx_rs::Dtype;

    fn keep_set(stride: u32, total: usize) -> TokenKeepSet {
        TokenKeepSet::new(TokenDropStride::new(stride).unwrap(), total).expect("keep set")
    }

    /// Logical f32 rows of an array.
    ///
    /// **Not** a bare `as_slice`: a `take_axis` result is a gather output whose logical order need not
    /// match its physical buffer, and `as_slice` reads the physical buffer. Reading a `[2, 4, 2]`
    /// restore directly returns the batch rows interleaved, which looks exactly like a scatter bug and
    /// is not one — `mlx_gen::array::contiguous` is the workspace's documented fix.
    fn rows(a: &Array) -> Vec<f32> {
        let a = mlx_gen::array::contiguous(&a.as_dtype(Dtype::Float32).expect("f32"))
            .expect("contiguous");
        a.eval().expect("eval");
        a.as_slice::<f32>().to_vec()
    }

    /// The provider bridge's `cfg(test)` shape, mirrored here so the plan → mechanism step is covered.
    fn from_plan(plan: &ApproximationPlan, total: usize) -> Option<TokenKeepSet> {
        plan.token_pruning()
            .map(|policy| TokenKeepSet::new(policy.drop_stride, total).expect("keep set"))
    }

    #[test]
    fn an_exact_plan_yields_no_keep_set() {
        assert!(from_plan(&ApproximationPlan::Exact, 8).is_none());
        assert!(from_plan(
            &ApproximationPlan::token_pruning_only(TokenPruningPolicy::new(
                TokenDropStride::EVERY_SECOND_TOKEN
            )),
            8
        )
        .is_some());
    }

    #[test]
    fn the_keep_set_matches_the_contracts_phase_and_count() {
        for stride in [2u32, 3, 5] {
            let total = 17usize;
            let set = keep_set(stride, total);
            let declared = TokenDropStride::new(stride).unwrap();
            assert_eq!(set.total(), total as i32);
            assert_eq!(set.kept(), declared.kept_count(total) as i32);
            let expected: Vec<i32> = (0..total)
                .filter(|i| declared.keeps_token(*i))
                .map(|i| i as i32)
                .collect();
            assert_eq!(set.keep.as_slice::<i32>(), expected.as_slice());
        }
        assert!(TokenKeepSet::new(TokenDropStride::EVERY_SECOND_TOKEN, 0).is_err());
    }

    #[test]
    fn a_gather_then_restore_round_trip_leaves_dropped_rows_exactly_zero() {
        // The bit-exactness of the pass-through, at tensor level. A dropped row's sublayer output must
        // be EXACTLY zero — not small — because the block's residual is `x + gate·y` and only an exact
        // zero leaves `x` alone at the bit level.
        let set = keep_set(2, 6);
        let x = Array::from_slice(&(0..12).map(|v| v as f32).collect::<Vec<f32>>(), &[1, 6, 2]);
        let gathered = set.gather(&x, 1).expect("gather");
        assert_eq!(gathered.shape(), &[1, 3, 2], "kept rows only");
        assert_eq!(
            rows(&gathered),
            vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0],
            "rows 0, 2, 4 survive a stride of 2"
        );

        let restored = set.restore(&gathered).expect("restore");
        assert_eq!(restored.shape(), x.shape(), "restore returns full length");
        assert_eq!(
            rows(&restored),
            vec![0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, 9.0, 0.0, 0.0],
            "kept rows keep their values and dropped rows are exactly zero"
        );
    }

    #[test]
    fn the_rope_tables_gather_on_their_own_axis() {
        // The query-side RoPE tables are `[L, 1, half_d]` and must be gathered with the SAME keep-set as
        // the queries: `rope_apply` derives its sequence length from the table, so a mismatch is a
        // silent padding-tail, not an error.
        let set = keep_set(3, 6);
        let cos = Array::from_slice(&(0..12).map(|v| v as f32).collect::<Vec<f32>>(), &[6, 1, 2]);
        let gathered = set.gather(&cos, 0).expect("gather rope");
        assert_eq!(gathered.shape(), &[4, 1, 2]);
        assert_eq!(
            rows(&gathered),
            vec![0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0],
            "a stride of 3 drops positions 2 and 5"
        );
    }

    #[test]
    fn a_mismatched_axis_length_is_refused_rather_than_clamped() {
        // `take_axis` clamps out-of-range indices instead of erroring, so a keep-set applied to the
        // wrong token count would silently read the last row repeatedly. That failure mode has bitten
        // this workspace before; the guard turns it into a message.
        let set = keep_set(2, 6);
        let wrong = Array::from_slice(&[0.0f32; 8], &[1, 4, 2]);
        let error = set
            .gather(&wrong, 1)
            .expect_err("a 4-token tensor must not accept a 6-token keep-set");
        assert!(error.to_string().contains("would clamp"), "{error}");

        let error = set
            .restore(&Array::from_slice(&[0.0f32; 4], &[1, 2, 2]))
            .expect_err("restore must check the kept count");
        assert!(error.to_string().contains("3 kept rows"), "{error}");
        let error = set
            .restore(&Array::from_slice(&[0.0f32; 6], &[6]))
            .expect_err("restore must check the rank");
        assert!(error.to_string().contains("batch, kept, dim"), "{error}");
    }

    #[test]
    fn restore_preserves_the_batch_width_and_the_dtype() {
        // CFG runs the Wan forward at batch 2; a restore that collapsed or broadcast the batch would
        // render one branch's update into both.
        let set = keep_set(2, 4);
        let y = Array::from_slice(&(0..8).map(|v| v as f32).collect::<Vec<f32>>(), &[2, 2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let restored = set.restore(&y).expect("restore");
        assert_eq!(restored.shape(), &[2, 4, 2]);
        assert_eq!(
            restored.dtype(),
            Dtype::Bfloat16,
            "the zero pad must adopt the value dtype rather than promoting the stream"
        );
        assert_eq!(
            rows(&restored),
            vec![0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0],
            "each batch row restores independently"
        );
    }
}
