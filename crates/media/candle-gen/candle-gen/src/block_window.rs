//! Rung 4 — **bounded transformer residency**, the Candle/CUDA half (SC-15792, epic 15448).
//!
//! The schedule itself lives in [`gen_core::block_window`]: window arithmetic, loop order, release
//! discipline and the cancellation contract are identical on every backend, and SC-15790 hoisted them
//! there precisely so this crate would not fork them. This module supplies Candle's answers to the
//! two backend questions, plus the one obligation that is genuinely Candle's and has no MLX twin.
//!
//! # Candle's three answers, each measured rather than carried over
//!
//! SC-15791 measured all three on real packed-quantized weights (z-image-turbo q4, 30 blocks,
//! 97.1 MiB/block on disk) on a 95.6 GiB RTX PRO 6000 Blackwell. Where Candle differs from MLX, the
//! difference **is** the finding — this is the rule rung 3 taught, and rung 4 confirms it three times.
//!
//! ## 1. `materialize` is `Ok(())`. There is no eager-backend analogue of MLX's guard.
//!
//! On MLX the pre-release evaluation is load-bearing because an unevaluated graph node still
//! references the window's weights: 8.0 MiB with the guard against 238.4 MiB without, correct output
//! either way — silent, not loud. **That reason does not transfer, and it was ablated rather than
//! reasoned about.** Removing the per-window `Device::synchronize()` leaves both peaks unchanged:
//!
//! | | step | live peak | reserved peak |
//! |---|---|---|---|
//! | with a per-window sync | 7.99 s | 131.3 MiB | 256 MiB |
//! | without | 8.36 s | **131.3 MiB (1.00x)** | 192 MiB |
//!
//! Candle is eager, so no graph node holds a reference, and `cuMemFreeAsync` decrements the pool's
//! accounting at *enqueue* — live bytes never accumulate even with a whole step's frees outstanding.
//! The hook is therefore wired to `Ok(())` **here, once**, rather than exposed to each family: a
//! per-family hook is an invitation to re-derive MLX's answer without MLX's reason.
//!
//! ## 2. `release` is a per-window NO-OP. The synchronize is PER-FORWARD, and it is not optional.
//!
//! The obvious reading of "does the memory come back?" gives the wrong answer here, so the two
//! counters are stated separately:
//!
//! ```text
//! pool used:     live 107.9 | after drop 0.0   | after drop+sync 0.0   MiB
//! pool reserved: live 160.0 | after drop 160.0 | after drop+sync 0.0   MiB
//! DRIVER free:   93.773     | 93.773           | 93.930                GiB
//! ```
//!
//! The pool's USED counter frees on the bare drop, but **driver-visible VRAM recovers 0.0 MiB until
//! something synchronizes**, because the pool's release threshold is 0 (the driver default; neither
//! candle nor cudarc sets it). Driver-visible free is what an admission gate reads
//! (`gpu::nvidia_smi_min_free_gib`, `testkit::VramProbe` — named as text, not intra-doc links:
//! `testkit` is feature-gated, so a link would break rustdoc on any build without it), so:
//!
//! - **per window**, a no-op is correct — the bound is held by pool reuse, confirmed by the ablation
//!   above;
//! - **at teardown**, something must synchronize or the last window stays charged against the free
//!   VRAM a co-resident component and the next job's admission gate read.
//!
//! "Teardown" means once per [`run_windowed`] call, i.e. once per denoise-step forward — the sampler
//! calls the trunk `steps x count` times per request, so this is not literally request-scoped. It is
//! the tightest scope the driver can enforce without knowing which call is the last, and it takes the
//! sync count from 30 per step (one per window, the shape Krea shipped) to 1.
//!
//! `cuMemPoolTrimTo` is a **no-op** at a release threshold of 0 and must not be credited with the
//! recovery — reading only the pool counter produces the wrong answer here.
//!
//! [`run_windowed`] therefore performs the teardown synchronize itself, on **every** exit path
//! including cancellation and failure, and it is the reason this adapter is not a two-line
//! passthrough. Krea's [`MemoryRequestScope::finish`](gen_core::MemoryRequestScope) also
//! synchronizes, so the production path is doubly covered; that is deliberate rather than redundant,
//! because `finish` belongs to a lifecycle a family may not implement while this one runs for every
//! caller of the driver.
//!
//! ## 3. Freshness is discharged structurally, not by re-opening.
//!
//! [`BlockWindowBackend::open_view`] requires
//! a FRESH view per window because on MLX the view IS a map of `Array` handles — a clone left behind
//! keeps the materialized buffer resident and the bound silently does not hold.
//!
//! Candle's families hold their block weights as a read-only `MmapedSafetensors` handle that caches
//! **nothing**: every `get` reads the mapping and produces a new device tensor, so there is no
//! retained buffer for a stale view to pin. The obligation is real but vacuous here, and the honest
//! discharge is to say so rather than to pay a re-`mmap` per window for a guarantee the type already
//! gives. `open` is a caller-supplied closure so a family whose view *does* cache can return a
//! genuinely fresh one; `a_fresh_view_is_opened_per_window_and_dropped_before_the_next` pins that the
//! driver calls it once per window either way.
//!
//! Correspondingly, gen-core's "`apply` must take its tensors OUT of the view" rule is an MLX rule
//! about a `HashMap<String, Array>`. Candle's `apply` *constructs* its blocks from the mapping and
//! drops them at the end of the window; there is nothing to take out. Same obligation, different
//! discharge.
//!
//! # What this module does NOT license
//!
//! sc-12195 — the FLUX.2-dev Q4 pixel corruption fixed by a phase-boundary `Device::synchronize()`,
//! enforced at a single point in [`crate::residency::run_sequential`] — **is not explained by any of
//! the above**, and SC-15791 says so explicitly. Every arm it measured stayed on one device and one
//! stream, where the stream-ordered allocator guarantees reuse ordering a priori. Nothing here is a
//! reason to remove that sync. The nearest untested configuration is a window driver materializing
//! from a worker thread, and SC-15791 asked SC-15792 to cover it. Two arms do, at different strengths,
//! and the difference is stated rather than glossed:
//!
//! - `candle_gen_krea::transformer::tests::streamed_output_is_identical_when_driven_from_a_worker_thread`
//!   runs on CPU. It pins that the driver is `Send`-safe and order-preserving off the main thread, and
//!   it **cannot** reproduce a CUDA stream race.
//! - `rung4_block_window_real_weights::worker_thread_materialization_is_bit_identical` is the CUDA arm,
//!   on real packed weights.
//!
//! Neither is a proof that no such race exists — a negative result on one configuration is not one —
//! so sc-12195's sync stays exactly where it is.

use std::ops::Range;

use candle_core::Device;
use gen_core::block_window::BlockWindowBackend;
use gen_core::runtime::CancelFlag;

use crate::Result;

/// Re-exported so a candle family keeps a single import path, exactly as `mlx_gen::block_residency`
/// does. The type is backend-neutral arithmetic and is deliberately NOT re-declared here.
pub use gen_core::block_window::BlockPlan;

/// Binds Candle's answers to the shared driver: the caller's view opener, and a per-window release
/// that is a measured no-op (see the module docs — the synchronize is request-scoped, and
/// [`run_windowed`] owns it).
struct CandleWindowBackend<V, F> {
    open: F,
    _view: std::marker::PhantomData<fn() -> V>,
}

impl<V, F> BlockWindowBackend for CandleWindowBackend<V, F>
where
    F: FnMut() -> Result<V>,
{
    type View = V;

    fn open_view(&mut self) -> gen_core::Result<V> {
        (self.open)().map_err(Into::into)
    }

    /// Deliberately empty. Candle's per-window bound is held by pool reuse, ablated at **1.00x** on
    /// both the live and reserved counters in SC-15791; the synchronize that actually returns pages to
    /// the driver runs once per [`run_windowed`] call instead. Synchronizing here as well would cost
    /// 30 syncs per denoise step rather than 1, for a peak that is measurably unchanged. (SC-15791's
    /// two timings, 7.99 s guarded against 8.36 s unguarded, sit inside that host's run-to-run spread
    /// and are not evidence either way on speed — the claim here is about peak, which was flat.)
    fn release(&self) {}
}

/// Run one denoise step's worth of block windows over Candle weights, then return the allocator's
/// pages to the driver.
///
/// Thin adapter over [`gen_core::block_window::run_windowed`] — see there for the per-window
/// lifecycle and ordering guarantees, and the module docs above for why Candle's `materialize` is
/// `Ok(())` and its `release` a no-op.
///
/// What this adds over the shared driver is the **teardown synchronize**, which runs on every exit
/// path: success, a failed window, and cancellation. Without it the last window's pages stay charged
/// against driver-visible free VRAM — the number an admission gate reads — even though the pool's own
/// USED counter has already dropped to zero.
///
/// A synchronize failure never displaces a real one: if the window body failed or was cancelled, that
/// error is returned and the synchronize's is dropped, matching
/// [`crate::residency::run_three_stage_sequential`]'s rule that the actionable job result stays
/// primary.
pub fn run_windowed<V, S>(
    device: &Device,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    init: S,
    open: impl FnMut() -> Result<V>,
    apply: impl FnMut(S, &mut V, Range<usize>) -> Result<S>,
) -> Result<S> {
    run_windowed_with_sync(
        plan,
        cancel,
        init,
        open,
        apply,
        || Ok(device.synchronize()?),
    )
}

/// [`run_windowed`] with the teardown synchronize injected as a closure, so the "every exit path
/// synchronizes, and the primary error survives" contract is pinnable without a GPU.
///
/// Private on purpose, mirroring [`crate::residency`]'s `run_sequential_with_sync`: production
/// callers must go through [`run_windowed`], which wires the hook to [`Device::synchronize`].
fn run_windowed_with_sync<V, S>(
    plan: &BlockPlan,
    cancel: &CancelFlag,
    init: S,
    open: impl FnMut() -> Result<V>,
    mut apply: impl FnMut(S, &mut V, Range<usize>) -> Result<S>,
    teardown_sync: impl FnOnce() -> Result<()>,
) -> Result<S> {
    let mut backend = CandleWindowBackend {
        open,
        _view: std::marker::PhantomData,
    };
    let outcome = gen_core::block_window::run_windowed(
        &mut backend,
        plan,
        cancel,
        init,
        |state, view, range| apply(state, view, range).map_err(Into::into),
        // Candle is eager and `cuMemFreeAsync` decrements at enqueue: measured 1.00x with and without
        // (SC-15791). MLX's guard has no counterpart here.
        |_state| Ok(()),
    )
    .map_err(crate::CandleError::from);

    // Runs even when `outcome` is `Err` — a cancelled or failed window has still charged pages
    // against driver-visible free, and the next request's admission gate reads that number.
    let synced = teardown_sync();
    match outcome {
        Err(primary) => Err(primary),
        Ok(state) => {
            synced?;
            Ok(state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CandleError;
    use std::cell::RefCell;

    /// The teardown synchronize runs AFTER every window, on the success path. Without it the last
    /// window's pages stay charged against driver-visible free VRAM (SC-15791: the pool's USED
    /// counter drops on the bare drop, but `cuMemGetInfo` recovers 0.0 MiB until a synchronize).
    #[test]
    fn teardown_sync_runs_once_after_the_last_window() {
        let log = RefCell::new(Vec::new());

        let out: Result<usize> = run_windowed_with_sync(
            &BlockPlan::new(4, 2).unwrap(),
            &CancelFlag::default(),
            0usize,
            || {
                log.borrow_mut().push("open");
                Ok(())
            },
            |s, _v: &mut (), r| {
                log.borrow_mut().push("apply");
                Ok(s + r.len())
            },
            || {
                log.borrow_mut().push("sync");
                Ok(())
            },
        );

        assert_eq!(out.unwrap(), 4);
        assert_eq!(
            *log.borrow(),
            ["open", "apply", "open", "apply", "sync"],
            "one sync, after the last window — not one per window"
        );
    }

    /// A cancelled run still synchronizes. This is the arm that keeps a cancelled render from
    /// leaving its last window charged against the next job's admission gate, and it is the half of
    /// the AC that a success-only test would report as passing.
    #[test]
    fn a_cancelled_run_still_synchronizes_and_stays_typed() {
        let cancel = CancelFlag::default();
        cancel.cancel();
        let synced = RefCell::new(false);

        let err = run_windowed_with_sync(
            &BlockPlan::new(4, 2).unwrap(),
            &cancel,
            0usize,
            || Ok(()),
            |s, _v: &mut (), _r| Ok(s),
            || {
                *synced.borrow_mut() = true;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(
            matches!(err, CandleError::Canceled),
            "cancellation must round-trip the gen-core boundary as the typed CandleError::Canceled, \
             not a stringified Msg — a Msg reports a cancelled render as a FAILED job (sc-4481): \
             {err:?}"
        );
        assert!(*synced.borrow(), "a cancelled run must still synchronize");
    }

    /// A failing window synchronizes too, and the window's error stays primary — a synchronize
    /// failure must not overwrite the actionable one.
    #[test]
    fn a_failed_window_synchronizes_and_keeps_its_error_primary() {
        let synced = RefCell::new(false);

        let err = run_windowed_with_sync(
            &BlockPlan::new(4, 2).unwrap(),
            &CancelFlag::default(),
            0usize,
            || Ok(()),
            |_s, _v: &mut (), _r| -> Result<usize> { Err(CandleError::Msg("window boom".into())) },
            || {
                *synced.borrow_mut() = true;
                Err(CandleError::Msg("sync boom".into()))
            },
        )
        .unwrap_err();

        assert!(
            matches!(err, CandleError::Msg(ref m) if m == "window boom"),
            "the window's error is the actionable one: {err:?}"
        );
        assert!(*synced.borrow(), "a failed window must still synchronize");
    }

    /// A cancel tripped part-way through — the realistic case, since the flag is set by another
    /// thread mid-render. The driver stops at the NEXT window boundary, so the remaining windows
    /// never open, and the error is still the typed variant.
    #[test]
    fn a_mid_run_cancel_stops_at_the_next_window_boundary() {
        let cancel = CancelFlag::default();
        let opens = RefCell::new(0usize);

        let err = run_windowed_with_sync(
            &BlockPlan::new(30, 1).unwrap(),
            &cancel,
            0usize,
            || {
                *opens.borrow_mut() += 1;
                Ok(())
            },
            |s, _v: &mut (), r| {
                if r.start == 2 {
                    cancel.cancel();
                }
                Ok(s + 1)
            },
            || Ok(()),
        )
        .unwrap_err();

        assert!(matches!(err, CandleError::Canceled), "{err:?}");
        assert_eq!(
            *opens.borrow(),
            3,
            "windows 0..=2 ran; the cancel is caught at the boundary before window 3 opens, so the \
             remaining 27 blocks are never materialized"
        );
    }

    /// A successful run whose teardown synchronize fails must surface that failure rather than
    /// returning a state whose device work may not have completed.
    #[test]
    fn a_failed_teardown_sync_fails_the_run() {
        let out: Result<usize> = run_windowed_with_sync(
            &BlockPlan::new(2, 1).unwrap(),
            &CancelFlag::default(),
            0usize,
            || Ok(()),
            |s, _v: &mut (), _r| Ok(s),
            || Err(CandleError::Msg("device sync failed".into())),
        );

        assert!(matches!(out, Err(CandleError::Msg(ref m)) if m == "device sync failed"));
    }

    /// A fresh view per window, and the previous one dropped before the next opens. Candle's views
    /// cache nothing so this is not what holds the bound (see the module docs), but the driver's
    /// contract is that a family whose view *does* cache gets a genuinely fresh one — and a driver
    /// that hoisted the open out of the loop would silently break exactly that family.
    #[test]
    fn a_fresh_view_is_opened_per_window_and_dropped_before_the_next() {
        struct View<'a> {
            id: usize,
            log: &'a RefCell<Vec<String>>,
        }
        impl Drop for View<'_> {
            fn drop(&mut self) {
                self.log.borrow_mut().push(format!("drop{}", self.id));
            }
        }

        let log = RefCell::new(Vec::new());
        let next = RefCell::new(0usize);

        let out: Result<usize> = run_windowed_with_sync(
            &BlockPlan::new(6, 2).unwrap(),
            &CancelFlag::default(),
            0usize,
            || {
                let mut n = next.borrow_mut();
                *n += 1;
                log.borrow_mut().push(format!("open{n}"));
                Ok(View { id: *n, log: &log })
            },
            |s, _v: &mut View, r| Ok(s + r.len()),
            || Ok(()),
        );

        assert_eq!(out.unwrap(), 6);
        assert_eq!(
            *log.borrow(),
            ["open1", "drop1", "open2", "drop2", "open3", "drop3"],
            "each view must drop before the next opens — never two live at once"
        );
    }

    /// What survives the gen-core round trip, stated rather than assumed. `Canceled` is bit-exact
    /// because both bridges special-case it; a backend error is flattened to `Msg` on the way back
    /// (`gen_core::Error::Backend` has no `CandleError` counterpart) but keeps its text. Pinned so a
    /// future widening of `CandleError` does not silently start losing the message too.
    #[test]
    fn a_window_error_keeps_its_message_across_the_boundary() {
        let err = run_windowed_with_sync(
            &BlockPlan::new(2, 1).unwrap(),
            &CancelFlag::default(),
            0usize,
            || Ok(()),
            |_s, _v: &mut (), _r| -> Result<usize> {
                Err(CandleError::Msg(
                    "missing tensor: layers.0.attn.to_k.weight".into(),
                ))
            },
            || Ok(()),
        )
        .unwrap_err();

        assert!(
            matches!(err, CandleError::Msg(ref m) if m.ends_with("to_k.weight")),
            "{err:?}"
        );
    }

    /// A degenerate plan is rejected by `BlockPlan`, not by this adapter — pinned so a future
    /// refactor that starts constructing plans here inherits the validation rather than re-deriving
    /// it (a window of 0 would loop forever; `n_blocks` of 0 would silently skip the whole trunk).
    #[test]
    fn a_degenerate_plan_is_rejected_before_the_driver_runs() {
        assert!(BlockPlan::new(28, 0).is_err());
        assert!(BlockPlan::new(0, 1).is_err());
    }
}
