//! Rung 4 — **bounded transformer residency**, the backend-neutral half (SC-15790, epic 15448).
//!
//! Hold a WINDOW of consecutive transformer blocks materialized, run it, release it, advance. Peak
//! transformer weight residency becomes `window_size × per-block bytes` instead of the whole stack.
//!
//! This module owns the parts that are the same everywhere: the window arithmetic, the loop order,
//! the release discipline, and the cancellation contract. The two operations that genuinely differ
//! per backend — obtaining a weights view and releasing the allocator afterwards — are behind
//! [`BlockWindowBackend`].
//!
//! ## Why it lives in gen-core
//!
//! It first shipped inside `mlx-gen` (SC-15750). That was the wrong home: [`BlockPlan`] had zero
//! backend references, and the driver touched MLX in exactly two places. Every rung of this ladder
//! decomposes the same way — a backend-neutral PLANNER (candidate ladders, window sizes, chunk
//! arithmetic: measured constants plus pure arithmetic) and a backend-specific KERNEL. Only the
//! kernel needs duplicating per backend. Leaving the planner in a backend crate guarantees the next
//! backend forks it.
//!
//! ## What the backends do NOT share: why the release ordering exists
//!
//! [`run_windowed`] forces evaluation of the carried state before releasing each window. On
//! **MLX/Metal** that is load-bearing for a specific reason — evaluation is lazy, so the carried
//! activation is an unevaluated graph node still referencing the window's weights, and dropping
//! before forcing evaluation frees nothing. Measured there on real packed-quantized weights at
//! window = 1: **8.0 MiB with the guard, 238.4 MiB without** — identical to fully resident, with
//! correct output either way. Silent, not loud.
//!
//! **Do not assume that reason transfers.** An eager backend has no lazy graph to cut, so the guard
//! may be unnecessary there — or it may be necessary for an unrelated reason, such as device memory
//! not being reclaimed until a stream synchronizes. The `materialize` hook is therefore part of the
//! contract's SHAPE (there is a point at which the carried state must be made real before the window
//! goes away), while whether a given backend needs it, and why, is that backend's question to answer
//! by measurement. Carrying a twin backend's *mechanism* across is how rung 3 came to be assumed
//! worth roughly an order of magnitude more on MLX than it actually is: measured like-for-like on the
//! denoise phase it is −32% on Candle against −1.7% here (SC-15793 — an earlier note put the gap at
//! "~6×" by comparing Candle's denoise-phase figure with MLX's whole-request one; see
//! [`crate::attention_budget`] for the corrected table).
//!
//! Likewise the MLX figures above are MLX figures. They establish that the shape works; they say
//! nothing about what any other backend will measure.
//!
//! ## What a window materialization may cost (SC-16090)
//!
//! One rung-4 render materializes every block once per denoise step, so a window's cost is multiplied
//! by `n_blocks × steps` — 240× for a 30-block stack over 8 steps. The obligation follows from that
//! arithmetic, not from taste: **`open_view` plus the tensor reads `apply` makes must be a transfer of
//! bytes already in the form the accelerator consumes.** A mapped read, and where the memory is not
//! unified, a host-to-device copy. Nothing else.
//!
//! Specifically NOT in the per-window path:
//!
//! - **Format conversion.** A quantized-container repack, a dequantize-then-requantize, a transpose,
//!   a dtype cast — anything whose cost scales with the block's weights. These are pure functions of
//!   bytes that do not change between steps, so recomputing them per window pays them 240 times for
//!   one answer. Do them once, ahead of the render, into an artifact a window can map.
//! - **A device→host round trip.** Reading the source tensors through the compute device and pulling
//!   them back to the host to convert them costs an upload plus a download per window and saves
//!   nothing. Read host-side data host-side.
//! - **Re-reading the source tier from a cold page cache.** Rung 4 exists for low-RAM hosts, which are
//!   exactly the hosts that cannot hold the tier in page cache, so a per-step re-read is disk-bound.
//!
//! This is the invariant that lets the ladder keep ONE shared cost order across backends. SC-15791
//! measured what its absence costs: Candle's rung 4 ran ~26× MLX's on the identical file, and ~100% of
//! the gap was a host-side MLX-affine → GGML-`Q4_1` repack in the per-window path, with the PCIe
//! transfer it was blamed on unresolvable against the noise floor. The contract half is
//! [`crate::memory_strategy::MemoryWindowMaterialization`], which a provider declares and
//! `conformance_errors` checks; this is the same statement addressed to whoever writes the backend.
//!
//! **The obligation binds the rung, not this trait.** It applies to any realization of bounded
//! transformer residency, including one that has not adopted this driver — `candle-gen-krea`'s shipped
//! streamed trunk folds its own block sequence and never touches [`BlockWindowBackend`], and it is the
//! realization that currently violates the obligation. That divergence is itself a defect (SC-15792:
//! a backend-local reimplementation of the driver is a review failure, because rung 4 fell through the
//! cracks originally by primitives being less enforced than the selector), but it means the obligation
//! must not be read as scoped to implementors of this trait.

use std::ops::Range;

use crate::runtime::CancelFlag;
use crate::{Error, Result};

/// How the block stack is split into windows. Pure arithmetic — no backend dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlan {
    n_blocks: usize,
    window: usize,
}

impl BlockPlan {
    /// `window` is the number of consecutive blocks held materialized at once. `1` is the tightest
    /// (lowest peak, most re-materialization); `n_blocks` degenerates to fully resident.
    ///
    /// Deliberately NOT defaulted to a constant: the right window is a measured trade between peak
    /// residency and per-step re-materialization cost, and it differs per model, per tier and per
    /// host. It is carried as `strategyParameters.streamedBlocks.transformerWindowSize` in the
    /// calibration contract so the selected value travels with the evidence that justified it.
    pub fn new(n_blocks: usize, window: usize) -> Result<Self> {
        if n_blocks == 0 {
            return Err(Error::Msg("block plan needs at least one block".to_owned()));
        }
        if window == 0 {
            return Err(Error::Msg("block window must be >= 1".to_owned()));
        }
        Ok(Self {
            n_blocks,
            window: window.min(n_blocks),
        })
    }

    /// Fully-resident equivalent — one window covering every block. Useful as the A-side of a parity
    /// test, so the windowed path is compared against the same code path rather than a second one.
    pub fn resident(n_blocks: usize) -> Result<Self> {
        Self::new(n_blocks, n_blocks)
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// Whether this plan actually bounds anything (a single all-covering window does not).
    pub fn is_bounded(&self) -> bool {
        self.window < self.n_blocks
    }

    /// The half-open block ranges, in order.
    pub fn windows(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        (0..self.n_blocks)
            .step_by(self.window)
            .map(move |start| start..(start + self.window).min(self.n_blocks))
    }

    /// How many windows a single denoise step walks.
    pub fn window_count(&self) -> usize {
        self.n_blocks.div_ceil(self.window)
    }
}

/// The backend-specific half of a windowed block schedule — deliberately only two operations.
///
/// Anything more than this belongs in [`run_windowed`], because it is the same on every backend.
pub trait BlockWindowBackend {
    /// A view of the model's block weights that a window can take its tensors out of.
    ///
    /// Whatever makes opening cheap is the backend's business — lazily-mapped handles, a deferred
    /// reader, an index. The driver opens one per window, so an implementation that eagerly reads
    /// the whole checkpoint here would defeat the entire rung.
    ///
    /// Reading the window's own tensors out of this view must also be cheap, in the specific sense the
    /// module docs give: a transfer of already-device-format bytes, never a per-window format
    /// conversion. A view that hands out tensors needing conversion satisfies the *bound* while
    /// destroying the *cost*, which is the failure SC-15791 measured at ~26×.
    type View;

    /// Open a fresh view. Called once per window.
    ///
    /// It must be FRESH: a view retained across windows keeps every materialized buffer alive
    /// through its own map, so the release below frees nothing.
    fn open_view(&mut self) -> Result<Self::View>;

    /// Release whatever the backend's allocator still holds once a window's weights have dropped.
    ///
    /// A no-op is a legitimate implementation if the backend returns memory on drop. It is called on
    /// every exit path, including cancellation and failure.
    fn release(&self);
}

/// Run one denoise step's worth of block windows, carrying `state` (the hidden activations) through.
///
/// Lifecycle per window: open a view, hand it to `apply`, force evaluation of the carried state,
/// drop the view, release the allocator.
///
/// - `apply` MUST take the tensors it uses OUT of the view rather than cloning them. Where a
///   backend's tensor handles are refcounted, a clone left behind in the view keeps the materialized
///   buffer resident and the bound silently does not hold.
/// - `materialize` makes the carried state real before the window goes away — see the module docs
///   for why this matters on a lazy backend and why that reasoning must not be assumed elsewhere.
///
/// Cancellation is checked at every window boundary and reported as [`Error::Canceled`], never as a
/// generic message: the worker and the conformance harness distinguish a cancelled render from a
/// failed one, and a stringified cancellation reports a cancelled job as a failure. The allocator is
/// released on every exit path, including the cancelled and failed ones, so a partial window cannot
/// be inherited by the next request.
pub fn run_windowed<B, S>(
    backend: &mut B,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    init: S,
    mut apply: impl FnMut(S, &mut B::View, Range<usize>) -> Result<S>,
    materialize: impl Fn(&S) -> Result<()>,
) -> Result<S>
where
    B: BlockWindowBackend,
{
    let mut state = init;
    for range in plan.windows() {
        if cancel.is_cancelled() {
            backend.release();
            return Err(Error::Canceled);
        }

        let mut view = backend.open_view()?;
        state = match apply(state, &mut view, range.clone()) {
            Ok(next) => next,
            Err(e) => {
                drop(view);
                backend.release();
                return Err(e);
            }
        };

        // LOAD-BEARING on a lazy backend: the carried state may still reference this window's
        // weights through an unevaluated graph. See the module docs.
        if let Err(e) = materialize(&state) {
            drop(view);
            backend.release();
            return Err(e);
        }

        drop(view);
        backend.release();
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn windows_cover_every_block_exactly_once() {
        let plan = BlockPlan::new(30, 4).unwrap();
        let covered: Vec<usize> = plan.windows().flat_map(|r| r.collect::<Vec<_>>()).collect();
        assert_eq!(covered, (0..30).collect::<Vec<_>>());
        assert_eq!(plan.window_count(), 8); // 7 full + 1 partial
    }

    #[test]
    fn a_ragged_tail_is_not_dropped() {
        // 30 blocks at window 7 leaves a 2-block tail; losing it would silently skip layers.
        let plan = BlockPlan::new(30, 7).unwrap();
        let last = plan.windows().last().unwrap();
        assert_eq!(last, 28..30);
        assert_eq!(
            plan.windows().flat_map(|r| r.collect::<Vec<_>>()).count(),
            30
        );
    }

    #[test]
    fn window_is_clamped_and_resident_is_unbounded() {
        let plan = BlockPlan::new(30, 999).unwrap();
        assert_eq!(plan.window(), 30);
        assert!(!plan.is_bounded(), "one all-covering window bounds nothing");
        assert!(BlockPlan::new(30, 4).unwrap().is_bounded());
        assert_eq!(BlockPlan::resident(30).unwrap().window_count(), 1);
    }

    #[test]
    fn degenerate_plans_are_rejected() {
        assert!(BlockPlan::new(0, 1).is_err());
        assert!(BlockPlan::new(30, 0).is_err());
    }

    /// A backend that records the lifecycle so the driver's ordering can be asserted without a GPU.
    #[derive(Default)]
    struct Recorder {
        log: Rc<RefCell<Vec<String>>>,
        opens: usize,
    }

    impl BlockWindowBackend for Recorder {
        type View = usize;
        fn open_view(&mut self) -> Result<usize> {
            self.opens += 1;
            self.log.borrow_mut().push("open".into());
            Ok(self.opens)
        }
        fn release(&self) {
            self.log.borrow_mut().push("release".into());
        }
    }

    #[test]
    fn a_fresh_view_is_opened_per_window_and_released_after_materialize() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut backend = Recorder {
            log: Rc::clone(&log),
            opens: 0,
        };
        let plan = BlockPlan::new(4, 2).unwrap();
        let l = Rc::clone(&log);
        run_windowed(
            &mut backend,
            &plan,
            &CancelFlag::default(),
            0usize,
            |s, _v, r| Ok(s + r.len()),
            move |_| {
                l.borrow_mut().push("materialize".into());
                Ok(())
            },
        )
        .unwrap();

        // Two windows, and within each: open BEFORE apply, materialize BEFORE release.
        assert_eq!(
            log.borrow().as_slice(),
            [
                "open",
                "materialize",
                "release",
                "open",
                "materialize",
                "release"
            ]
        );
        assert_eq!(backend.opens, 2, "a fresh view per window, not one reused");
    }

    /// Cancellation must surface as the TYPED variant. A generic message reports a cancelled render
    /// as a failed job (sc-4481), and this driver is the seam where that is easiest to get wrong.
    #[test]
    fn cancellation_is_typed_and_still_releases() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut backend = Recorder {
            log: Rc::clone(&log),
            opens: 0,
        };
        let cancel = CancelFlag::default();
        cancel.cancel();

        let err = run_windowed(
            &mut backend,
            &BlockPlan::new(4, 2).unwrap(),
            &cancel,
            0usize,
            |s, _v: &mut usize, _r| Ok(s),
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(
            matches!(err, Error::Canceled),
            "cancellation must be Error::Canceled, not a stringified message: {err:?}"
        );
        assert_eq!(log.borrow().as_slice(), ["release"]);
        assert_eq!(backend.opens, 0, "cancelled before opening anything");
    }

    /// A failing window releases too — otherwise its half-materialized weights are inherited by the
    /// next request.
    #[test]
    fn a_failed_window_still_releases() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut backend = Recorder {
            log: Rc::clone(&log),
            opens: 0,
        };
        let err = run_windowed(
            &mut backend,
            &BlockPlan::new(4, 2).unwrap(),
            &CancelFlag::default(),
            0usize,
            |_s, _v: &mut usize, _r| Err(Error::Msg("boom".into())),
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(matches!(err, Error::Msg(ref m) if m == "boom"));
        assert_eq!(log.borrow().as_slice(), ["open", "release"]);
    }

    /// A failing `materialize` must release as well — the window is still live at that point.
    #[test]
    fn a_failed_materialize_still_releases() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut backend = Recorder {
            log: Rc::clone(&log),
            opens: 0,
        };
        let err = run_windowed(
            &mut backend,
            &BlockPlan::new(2, 1).unwrap(),
            &CancelFlag::default(),
            0usize,
            |s, _v: &mut usize, _r| Ok(s),
            |_| Err(Error::Msg("no eval".into())),
        )
        .unwrap_err();

        assert!(matches!(err, Error::Msg(ref m) if m == "no eval"));
        assert_eq!(log.borrow().as_slice(), ["open", "release"]);
    }
}
