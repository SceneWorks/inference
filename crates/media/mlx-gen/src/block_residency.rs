//! Rung 4 — **bounded transformer residency** (sc-15750, epic 15448).
//!
//! The shared sibling of [`crate::residency`] (rung 1, phase-level shedding) and
//! [`crate::vae_tiling`] (rung 2, decode tiling). Where rung 1 drops the whole DiT *between* phases,
//! this bounds residency *within* the denoise phase: hold a WINDOW of consecutive transformer blocks
//! materialized, run it, release it, advance. Peak transformer weight residency becomes
//! `window_size × per-block bytes` instead of the whole stack.
//!
//! Both compose — rung 4 runs inside rung 1's `Heavy` phase, it does not replace it.
//!
//! ## Why this is a shared primitive and not per-family code
//!
//! The mechanism is identical for every block-stacked model (Z-Image, Mage-Flow, Qwen, FLUX). Rungs
//! 1 and 2 already live here; rung 3 (`sdpa_budgeted_bhsd`) joined them. Leaving rung 4 to whichever
//! family reached it first meant building it once and copying it nineteen times.
//!
//! ## Measured basis (SC-15744, real z-image-turbo q4 weights, Apple/Metal)
//!
//! - `Array::load_safetensors_with_metadata` is **lazy per tensor** — 1073 handles cost 0.0 MiB until
//!   something is evaluated. Re-opening the file per step is therefore nearly free.
//! - One block materializes at **97.3 MiB** against a header-computed 97.1 MiB — no hidden copy.
//! - Dropping returns the memory: active fell to 8.0 MiB and stayed flat across all 30 blocks.
//! - Packed-quant survives it: `quantized_matmul` works on a per-block materialized
//!   `weight`/`scales`/`biases` triple, because each is an independent file entry rather than a slice
//!   of one packed buffer.
//! - All 30 blocks resident: **2918.3 MiB**. Windowed sweep peak: **97.3 MiB** — a 30.0× reduction,
//!   the theoretical ideal.
//! - Re-materializing every block once per denoise step cost **~0.309 s/step** with a warm page
//!   cache on a 128 GB machine. That is the OPTIMISTIC bound: the hosts that need this are small
//!   Macs under pressure, where the page cache is the first thing evicted. Widen the window if a
//!   constrained-host measurement is materially worse.
//!
//! ## The trap this module exists to prevent
//!
//! MLX is lazy. Block *n*'s output is an unevaluated graph node referencing block *n*'s weights, so
//! **dropping the window before forcing evaluation frees nothing** — the graph still holds it. This
//! is the block-level twin of the hazard [`crate::residency::Residency::run_staged`] documents for
//! `materialize_mid`, and it is silent: you get correct images and no memory saving.
//!
//! [`run_windowed`] therefore materializes the carried state before releasing each window, and
//! `block_window_without_materialize_frees_nothing` pins that the guard is load-bearing rather than
//! decorative. Measured, same weights, window = 1:
//!
//! | | peak |
//! |---|---:|
//! | with the materialize guard | **8.0 MiB** |
//! | without it | **238.4 MiB** — identical to fully resident |
//!
//! i.e. omitting it costs the entire saving while still producing correct output. That is the failure
//! mode to design against: silent, not loud.
//!
//! ## Measured window sweep (`block_residency_real_weights`)
//!
//! | window | peak | vs resident |
//! |---:|---:|---:|
//! | 1 | 8.0 MiB | 29.9× |
//! | 2 | 15.9 MiB | 15.0× |
//! | 4 | 31.8 MiB | 7.5× |
//! | 8 | 63.6 MiB | 3.7× |
//! | 30 (resident) | 238.4 MiB | — |
//!
//! Linear in window size, as the contract requires. (Absolute figures are below the 97.3 MiB/block of
//! SC-15744 because that test walks one packed triple per block rather than all 31 tensors; the
//! RATIOS are the invariant, and they are exact.)

use std::ops::Range;

use gen_core::runtime::CancelFlag;

use crate::weights::Weights;
use crate::{Error, Result};

/// How the block stack is split into windows.
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
    /// residency and per-step re-materialization cost, and it differs per model and per host. It is
    /// carried as `strategyParameters.streamedBlocks.transformerWindowSize` in the calibration
    /// contract so the selected value travels with the evidence that justified it.
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

/// Run one denoise step's worth of block windows, carrying `state` (the hidden activations) through.
///
/// Lifecycle per window: open a lazy weights view, hand it to `apply` (which takes the window's
/// tensors OUT of it and runs the blocks), **force evaluation of the carried state**, then drop the
/// window's weights and release the allocator cache.
///
/// - `open` yields lazily-loaded weights. It is called once per window and must be cheap — a
///   safetensors open is ~0 bytes until something is evaluated. It is a closure rather than a stored
///   `Weights` on purpose: a `Weights` retained across windows would keep every materialized buffer
///   alive through its own map, defeating the release.
/// - `apply` MUST take ownership of the tensors it uses (`Weights::remove`), not clone them. `Array`
///   is refcounted, so a clone left behind in the map keeps the materialized buffer resident.
/// - `materialize` forces evaluation of the carried state. Without it the drop below frees nothing.
///
/// Cancellation is checked at every window boundary; the cache is released on every exit path,
/// including the early return from a cancelled or failed window.
pub fn run_windowed<S>(
    plan: &BlockPlan,
    cancel: &CancelFlag,
    init: S,
    mut open: impl FnMut() -> Result<Weights>,
    mut apply: impl FnMut(S, &mut Weights, Range<usize>) -> Result<S>,
    materialize: impl Fn(&S) -> Result<()>,
) -> Result<S> {
    let mut state = init;
    for range in plan.windows() {
        if cancel.is_cancelled() {
            mlx_rs::memory::clear_cache();
            return Err(Error::Msg("cancelled".to_owned()));
        }

        let mut view = open()?;
        // A window that fails must not leave its half-materialized weights behind for the next
        // request to inherit, so the cache is released on the error path too.
        state = match apply(state, &mut view, range.clone()) {
            Ok(next) => next,
            Err(e) => {
                drop(view);
                mlx_rs::memory::clear_cache();
                return Err(e);
            }
        };

        // LOAD-BEARING: the carried state is a lazy graph node still referencing this window's
        // weights. Dropping before this call frees nothing and the bound silently does not hold.
        if let Err(e) = materialize(&state) {
            drop(view);
            mlx_rs::memory::clear_cache();
            return Err(e);
        }

        drop(view);
        mlx_rs::memory::clear_cache();
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
