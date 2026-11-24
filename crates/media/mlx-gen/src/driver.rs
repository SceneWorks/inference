//! MLX backend bindings for the typed [`FamilyDriver`] contract (`PIPE-7`, `PIPE-14`, `B3`).
//!
//! Exposes the shared [`gen_core::driver`] types plus the process-wide MLX global-RNG trajectory
//! serializer (`B3` / `PIPE-14`): any MLX route that seeds `mlx_rs::random::seed` and draws from
//! the global MLX RNG stream acquires [`with_global_rng_trajectory`] across its whole trajectory
//! (`seed → draws → materialization`), preventing concurrent cross-thread RNG interleaving while
//! remaining reentrant on the currently owning thread.

use mlx_rs::Array;

pub use gen_core::driver::{
    conformance, run_driver_slices, DriverEvaluationKind, DriverHistoryPolicy,
    DriverRefusalReasons, DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver,
    GlobalRngTrajectoryGate, GlobalRngTrajectoryLease, SliceInput, SliceSpec, SliceTransition,
    StepHooks,
};

use crate::{CancelFlag, Result};

/// Process-wide serialization gate for MLX routes using global `mlx_rs::random::seed` /
/// `mlx_rs::random::normal` (`B3` / `PIPE-14`).
static MLX_GLOBAL_RNG_TRAJECTORY_GATE: GlobalRngTrajectoryGate = GlobalRngTrajectoryGate::new();

/// Acquire the process-wide MLX global-RNG whole-trajectory lease on the calling thread.
///
/// Reentrant on the owning thread; other threads block until the outermost trajectory finishes
/// and materializes its draws.
pub fn acquire_global_rng_trajectory() -> GlobalRngTrajectoryLease<'static> {
    MLX_GLOBAL_RNG_TRAJECTORY_GATE.acquire()
}

/// Run an entire MLX global-RNG trajectory (`seed → draws → materialization`) atomically with
/// respect to other threads (`B3` / `PIPE-14`).
pub fn with_global_rng_trajectory<T>(body: impl FnOnce() -> T) -> T {
    MLX_GLOBAL_RNG_TRAJECTORY_GATE.run_trajectory(body)
}

/// Drive an MLX [`FamilyDriver<Array>`] across one or more planned slices, automatically holding
/// [`MLX_GLOBAL_RNG_TRAJECTORY_GATE`] when `driver.support().uses_global_rng` is `true` (`B3`).
#[allow(clippy::too_many_arguments)]
pub fn run_mlx_driver_slices<D: FamilyDriver<Array> + ?Sized>(
    driver: &D,
    slices: impl IntoIterator<Item = SliceSpec>,
    initial_state: gen_core::BoundaryState<Array>,
    total_steps: usize,
    seed: u64,
    cancel: &CancelFlag,
    run: Option<&gen_core::RenderRun<'_, '_>>,
    state: &mut DriverRunState,
    hooks: &mut StepHooks<'_, Array>,
    on_boundary: impl FnMut(
        usize,
        gen_core::BoundaryState<Array>,
        &mut DriverRunState,
    ) -> gen_core::Result<gen_core::BoundaryState<Array>>,
) -> Result<gen_core::BoundaryState<Array>> {
    let execute = || {
        run_driver_slices(
            driver,
            slices,
            initial_state,
            total_steps,
            seed,
            cancel,
            run,
            state,
            hooks,
            on_boundary,
        )
        .map_err(Into::into)
    };
    if driver.support().uses_global_rng {
        with_global_rng_trajectory(execute)
    } else {
        execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, Mutex};

    #[test]
    fn mlx_global_rng_trajectory_gate_is_reentrant_and_serializes_threads() {
        let log: Arc<Mutex<Vec<(u64, &'static str)>>> = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(Barrier::new(2));

        let spawn = |seed: u64| {
            let log = Arc::clone(&log);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                with_global_rng_trajectory(|| {
                    log.lock().unwrap().push((seed, "seed"));
                    // Reentrant acquisition on same thread (e.g. InstantID -> SDXL helper).
                    with_global_rng_trajectory(|| {
                        log.lock().unwrap().push((seed, "draw"));
                        std::thread::yield_now();
                    });
                    log.lock().unwrap().push((seed, "eval"));
                });
            })
        };

        let a = spawn(11);
        let b = spawn(22);
        a.join().unwrap();
        b.join().unwrap();

        let events = log.lock().unwrap();
        assert_eq!(events.len(), 6);
        let first = events[0].0;
        let second = events[3].0;
        assert_ne!(first, second);
        assert_eq!(
            &events[0..3],
            &[(first, "seed"), (first, "draw"), (first, "eval")]
        );
        assert_eq!(
            &events[3..6],
            &[(second, "seed"), (second, "draw"), (second, "eval")]
        );
    }
}
