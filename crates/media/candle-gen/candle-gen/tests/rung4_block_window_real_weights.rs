//! **SC-15792 — the rung-4 Candle implementation, measured on real packed-quantized weights.**
//!
//! SC-15791 measured what a windowed schedule *would* cost on Candle using its own throwaway loop.
//! This file measures the shipped one: every arm drives `candle_gen::block_window::run_windowed`
//! through the real `BlockWindowBackend`, so a claim here is a claim about production code rather
//! than about a harness that resembles it.
//!
//! Deliberately the **same file** SC-15791 and SC-15744 measured —
//! `SceneWorks/z-image-turbo-mlx@bb2bc989` q4, 30 blocks, 97.1 MiB/block on disk — so the figures are
//! directly comparable to both the Candle spike and the MLX twin rather than to a fresh baseline.
//!
//! ## Run
//!
//! ```text
//! SC15792_Q4=<snapshot>/q4/transformer/model.safetensors \
//! cargo test -p candle-gen --features cuda --release --test integration rung4_block_window_real_weights:: \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release` and `--test-threads=1` are both load-bearing, for the reasons SC-15791 recorded: a
//! debug build inflates the host-side repack ~14x and so measures the wrong thing entirely, and every
//! arm reads process-global driver counters that concurrent tests would interleave.
//!
//! Optional: `SC15792_CAP_MIB` (default 1024) sizes the enforced-ceiling arm's pool.
//!
//! ## What is measured here and what is inherited
//!
//! Inherited from SC-15791, not re-derived: that the per-window `release` can be a no-op and
//! `materialize` can be `Ok(())` (both ablated there at 1.00x peak), and that the per-window cost is
//! ~100% host-side format conversion rather than PCIe. Those answers are *encoded* in the
//! implementation and documented at `candle_gen::block_window`.
//!
//! Measured here, because they are properties of the implementation rather than of the backend:
//!
//! 1. the bound actually holds when the real driver runs — peak by window, in RESERVED as well as
//!    live, since RESERVED is what an admission gate reads;
//! 2. that the bound is **load-bearing rather than incidental**, by defeating it and watching the
//!    peak return to fully-resident (the twin of MLX's `block_window_without_materialize_frees_nothing`);
//! 3. that a cancelled or failed run leaves no residual allocator growth across repeats;
//! 4. that under an ENFORCED ceiling (sc-16091's capped pool, not a balloon) the windowed path
//!    completes where the resident path cannot.
//!
//! ## Reported in RESERVED
//!
//! `nvidia-smi memory.used` is driver-**reserved** bytes, and `gpu.rs::nvidia_smi_min_free_gib` and
//! `testkit::VramProbe` both consume it. Reserved ran 48% above live for a single block in SC-15791,
//! so a bound asserted only in live is asserted in the wrong unit. Both are printed; the assertions
//! that matter are on reserved.

#![cfg(feature = "cuda")]

use std::ops::Range;
use std::time::Instant;

use candle_gen::block_window::{run_windowed, BlockPlan};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::quant::MLX_GROUP_SIZE;
use candle_gen::{CandleError, Result};

use crate::rung4_support;
use rung4_support::{
    compute_into, disclose_host, env_path_opt, env_usize, linear_fit, materialize,
    quiesce_and_reset, Block, Pool, Tier, GIB, MIB,
};

const TAG: &str = "sc-15792";
const TIER_ENV: &str = "SC15792_Q4";
/// Token count for the synthetic per-block forward. Small on purpose: this file measures WEIGHT
/// residency, and a large activation would swamp the quantity under test with attention scratch.
const TOKENS: usize = 64;

/// One denoise step's worth of block windows through the **production** driver.
///
/// Returns the accumulated on-device scalar so parity across window sizes is checkable: the schedule
/// changes when weights are read, never what arithmetic runs on them, so every window must produce
/// the identical value.
fn windowed_step(tier: &Tier, dev: &Device, window: usize, cancel: &CancelFlag) -> Result<Tensor> {
    let plan = BlockPlan::new(tier.n_blocks(), window)?;
    let init = Tensor::zeros((), DType::F32, dev)?;
    run_windowed(
        dev,
        &plan,
        cancel,
        init,
        || tier.open_view(dev).map_err(CandleError::from),
        |acc, view: &mut VarBuilder<'static>, range: Range<usize>| {
            // The whole window is materialized before any of it runs — that is what makes peak
            // linear in `window` rather than constant at one block.
            let blocks = materialize(tier, view, range)?;
            let acc = compute_into(&blocks, TOKENS, dev, &acc)?;
            Ok(acc)
        },
    )
}

fn open_tier() -> Option<(Tier, Device, Pool)> {
    let path = env_path_opt(TAG, TIER_ENV)?;
    let tier = Tier::open(&path, MLX_GROUP_SIZE).expect("open the packed tier");
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let pool = Pool::open(0).expect("default mempool");
    disclose_host(TAG, &pool);
    println!(
        "[{TAG}] TIER: {} | {} blocks, {} tensors, {:.2} GiB on disk, {:.1} MiB/block",
        tier.path.display(),
        tier.n_blocks(),
        tier.n_tensors,
        tier.file_bytes as f64 / GIB,
        tier.total_block_bytes() as f64 / tier.n_blocks() as f64 / MIB,
    );
    Some((tier, dev, pool))
}

/// **AC: measured peak-by-window on CUDA hardware, and output identical to the resident path.**
///
/// Sweeps the real driver across window sizes up to the fully-resident degenerate case (one
/// all-covering window, which `BlockPlan::resident` produces and `is_bounded()` reports as bounding
/// nothing). The resident arm is therefore the SAME code path, not a second implementation — a
/// parity test against a different loader would be comparing two things that could both be wrong.
///
/// **The wall times printed here are NOT a timing measurement, and must not be read as one.** This is
/// one sample per window, taken in ascending order, on a host SC-15791 measured at up to ±38% spread
/// — so the first row is always the cold one and any apparent trend is confounded with sweep order.
/// A timing claim needs medians of an interleaved sample, which is what SC-15791's `window_sweep_cost`
/// does. They are printed because a wildly different figure is a useful smell, not as evidence.
/// The assertions are all on memory, which is what this arm can actually establish.
#[test]
#[ignore = "real weights + CUDA; set SC15792_Q4"]
fn window_peak_sweep_is_linear_and_output_is_identical() -> Result<()> {
    let Some((tier, dev, pool)) = open_tier() else {
        return Ok(());
    };
    let cancel = CancelFlag::default();
    let n = tier.n_blocks();

    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new();
    let mut reference: Option<f32> = None;

    for window in [1usize, 2, 4, 8, 15, n] {
        quiesce_and_reset(&dev, &pool)?;
        let t0 = Instant::now();
        let acc = windowed_step(&tier, &dev, window, &cancel)?;
        let secs = t0.elapsed().as_secs_f64();
        let live = pool.used_high() as f64 / MIB;
        let reserved = pool.reserved_high() as f64 / MIB;
        let value = acc.to_scalar::<f32>()?;

        match reference {
            None => reference = Some(value),
            Some(want) => assert_eq!(
                value, want,
                "window {window} changed the arithmetic: {value} vs {want}. Rung 4 re-orders WHEN \
                 weights are read, never what runs on them, so any deviation is a defect rather \
                 than accumulated error."
            ),
        }
        rows.push((window, secs, live, reserved));
        println!(
            "[{TAG}] window {window:>2}: {secs:6.2} s | live {live:8.1} MiB | RESERVED {reserved:8.1} MiB"
        );
    }

    let resident = *rows.last().expect("the resident row");
    let w1 = rows[0];
    println!(
        "[{TAG}] window 1 vs resident: live {:.1}x, RESERVED {:.1}x (resident = {:.1} MiB reserved)",
        resident.2 / w1.2,
        resident.3 / w1.3,
        resident.3
    );

    // Linearity in the window is the contract: `peak = window x per-block bytes`. Fit on the bounded
    // rows only — the resident row is the degenerate all-covering window and is the control.
    let bounded: Vec<(usize, f64)> = rows[..rows.len() - 1]
        .iter()
        .map(|(w, _, live, _)| (*w, *live))
        .collect();
    let (slope, intercept) = linear_fit(&bounded);
    println!("[{TAG}] fitted live peak ≈ {slope:.1}·w + {intercept:.1} MiB");
    assert!(
        slope > 0.0,
        "peak must GROW with the window; a flat fit means the window is not being materialized \
         (slope {slope:.1})"
    );
    for (w, live) in &bounded {
        let predicted = slope * (*w as f64) + intercept;
        assert!(
            (predicted - live).abs() < 0.10 * predicted.max(1.0),
            "window {w}: live peak {live:.1} MiB is not on the linear fit ({predicted:.1} MiB) — \
             the bound is not `window x per-block bytes`"
        );
    }

    // The rung has to actually save something, in the unit the gate reads.
    assert!(
        resident.3 / w1.3 > 5.0,
        "window 1 must cut the RESERVED peak by more than 5x against fully resident, got {:.1}x",
        resident.3 / w1.3
    );
    Ok(())
}

/// **AC: a mutation check proves the memory bound is load-bearing, not incidental.**
///
/// This is the arm the story asks for by name, and it exists because a rung-4 implementation can look
/// *completely correct while saving nothing* — MLX's `block_window_without_materialize_frees_nothing`
/// is its twin, and there the failure was silent: 8.0 MiB with the guard, 238.4 MiB without, correct
/// output either way.
///
/// The mutation is the realistic one for this backend. Candle's trap is not MLX's lazy graph; it is
/// gen-core's other rule — *"`apply` MUST take the tensors it uses OUT of the view rather than
/// cloning them"*. Here `apply` retains each window's materialized blocks in a `Vec` that outlives
/// the window. Every window still opens, runs and releases exactly as before, the output is
/// unchanged, and no error is raised — but nothing is ever freed, so the peak must climb back to
/// fully resident. If it does NOT, the sweep above is measuring something other than the bound and
/// its green is worthless.
#[test]
#[ignore = "real weights + CUDA; set SC15792_Q4"]
fn defeating_the_release_restores_the_resident_peak() -> Result<()> {
    let Some((tier, dev, pool)) = open_tier() else {
        return Ok(());
    };
    let cancel = CancelFlag::default();
    let plan = BlockPlan::new(tier.n_blocks(), 1)?;

    quiesce_and_reset(&dev, &pool)?;
    let honest = windowed_step(&tier, &dev, 1, &cancel)?.to_scalar::<f32>()?;
    let honest_live = pool.used_high() as f64 / MIB;
    let honest_reserved = pool.reserved_high() as f64 / MIB;

    quiesce_and_reset(&dev, &pool)?;
    // The mutation: hold every window's blocks alive for the whole step.
    let mut leaked: Vec<Block> = Vec::new();
    let init = Tensor::zeros((), DType::F32, &dev)?;
    let mutated = run_windowed(
        &dev,
        &plan,
        &cancel,
        init,
        || tier.open_view(&dev).map_err(CandleError::from),
        |acc, view: &mut VarBuilder<'static>, range: Range<usize>| {
            let blocks = materialize(&tier, view, range)?;
            let acc = compute_into(&blocks, TOKENS, &dev, &acc)?;
            leaked.extend(blocks);
            Ok(acc)
        },
    )?
    .to_scalar::<f32>()?;
    let mutated_live = pool.used_high() as f64 / MIB;
    let mutated_reserved = pool.reserved_high() as f64 / MIB;
    drop(leaked);

    println!(
        "[{TAG}] MUTATION: honest live {honest_live:.1} / reserved {honest_reserved:.1} MiB → \
         retained live {mutated_live:.1} / reserved {mutated_reserved:.1} MiB \
         ({:.1}x live, {:.1}x reserved)",
        mutated_live / honest_live,
        mutated_reserved / honest_reserved
    );

    assert_eq!(
        mutated, honest,
        "the mutation must be SILENT — identical output is what makes this failure mode dangerous \
         and the mutation check necessary"
    );
    assert!(
        mutated_live > 5.0 * honest_live,
        "retaining every window's blocks did not raise the peak ({mutated_live:.1} vs \
         {honest_live:.1} MiB). The window bound is then incidental, not load-bearing, and \
         `window_peak_sweep_is_linear_and_output_is_identical` is measuring something else."
    );
    Ok(())
}

/// **AC: cancellation and injected errors leave no residual allocator growth across repeats.**
///
/// A partial window that is not released is inherited by the next request. Repeating the cancelled
/// and failing paths and comparing the pool's high-water after the first repeat against the last is
/// what distinguishes "released" from "released once".
///
/// Cancellation must also stay the TYPED `Error::Canceled` all the way out: a stringified message
/// reports a cancelled render as a failed job (sc-4481).
#[test]
#[ignore = "real weights + CUDA; set SC15792_Q4"]
fn cancel_and_injected_error_leave_no_residual_growth() -> Result<()> {
    let Some((tier, dev, pool)) = open_tier() else {
        return Ok(());
    };
    let plan = BlockPlan::new(tier.n_blocks(), 1)?;
    const REPEATS: usize = 6;

    // --- cancelled mid-run, repeatedly -------------------------------------------------------
    let mut cancel_reserved = Vec::new();
    for repeat in 0..REPEATS {
        quiesce_and_reset(&dev, &pool)?;
        let cancel = CancelFlag::default();
        let init = Tensor::zeros((), DType::F32, &dev)?;
        let err = run_windowed(
            &dev,
            &plan,
            &cancel,
            init,
            || tier.open_view(&dev).map_err(CandleError::from),
            |acc, view: &mut VarBuilder<'static>, range: Range<usize>| {
                let blocks = materialize(&tier, view, range.clone())?;
                let acc = compute_into(&blocks, TOKENS, &dev, &acc)?;
                // Trip part-way, so a window is live when the flag is read at the next boundary.
                if range.start == 3 {
                    cancel.cancel();
                }
                Ok(acc)
            },
        )
        .expect_err("a tripped flag must abort the run");
        assert!(
            matches!(err, CandleError::Canceled),
            "repeat {repeat}: cancellation must be the typed Canceled, not a stringified Msg \
             (sc-4481): {err:?}"
        );
        assert!(matches!(
            candle_gen::gen_core::Error::from(err),
            candle_gen::gen_core::Error::Canceled
        ));
        dev.synchronize()?;
        cancel_reserved.push(pool.reserved_high() as f64 / MIB);
    }

    // --- an injected error mid-run, repeatedly -----------------------------------------------
    let mut error_reserved = Vec::new();
    for repeat in 0..REPEATS {
        quiesce_and_reset(&dev, &pool)?;
        let cancel = CancelFlag::default();
        let init = Tensor::zeros((), DType::F32, &dev)?;
        let err = run_windowed(
            &dev,
            &plan,
            &cancel,
            init,
            || tier.open_view(&dev).map_err(CandleError::from),
            |acc, view: &mut VarBuilder<'static>, range: Range<usize>| {
                let blocks = materialize(&tier, view, range.clone())?;
                let acc = compute_into(&blocks, TOKENS, &dev, &acc)?;
                if range.start == 3 {
                    // Fail with the window still materialized — the state the release must survive.
                    return Err(CandleError::Msg(format!(
                        "injected at block {}",
                        range.start
                    )));
                }
                Ok(acc)
            },
        )
        .expect_err("the injected error must propagate");
        assert!(
            matches!(err, CandleError::Msg(ref m) if m.starts_with("injected at block")),
            "repeat {repeat}: the injected error must arrive intact: {err:?}"
        );
        dev.synchronize()?;
        error_reserved.push(pool.reserved_high() as f64 / MIB);
    }

    println!("[{TAG}] cancelled repeats, RESERVED high (MiB): {cancel_reserved:.1?}");
    println!("[{TAG}] failed    repeats, RESERVED high (MiB): {error_reserved:.1?}");

    // Compare the LAST repeat against the SECOND, not the first: the first pass through a given
    // shape also warms cuBLAS/kernel-module state that is charged once and never again, and folding
    // that one-off into the growth check would make this arm noisy rather than strict.
    for (label, series) in [("cancel", &cancel_reserved), ("error", &error_reserved)] {
        let baseline = series[1];
        let last = series[REPEATS - 1];
        assert!(
            last <= baseline * 1.02,
            "{label}: RESERVED high-water grew across repeats ({baseline:.1} → {last:.1} MiB). A \
             partial window is being inherited by the next run."
        );
    }

    // Driver-visible free is the number the next job's admission gate reads, and it only recovers at
    // a synchronize — which is exactly the teardown obligation `run_windowed` owns. Assert the
    // reclaim actually happened rather than trusting the pool counter (SC-15791: the pool's USED
    // frees on a bare drop while driver-visible free recovers 0.0 MiB).
    dev.synchronize()?;
    let (free_after, total) = rung4_support::mem_info();
    println!(
        "[{TAG}] after all repeats: driver free {:.2} GiB of {:.2} GiB | pool reserved {:.1} MiB",
        free_after as f64 / GIB,
        total as f64 / GIB,
        pool.reserved() as f64 / MIB
    );
    assert!(
        pool.reserved() as f64 / MIB < 64.0,
        "the pool still holds {:.1} MiB after every run ended — the teardown synchronize did not \
         return the last window's pages",
        pool.reserved() as f64 / MIB
    );
    Ok(())
}

/// **AC: the bound is real under an ENFORCED ceiling, not gate arithmetic on a 96 GiB card.**
///
/// sc-16091 established that ballooning cannot validate a tier on this host — a 3.41 GiB working set
/// completed inside 1.93 GiB of driver-visible free at 1.07x wall time, so neither completion nor
/// timing detects the spill. A capped pool IS enforced and does bind candle's allocator, and that is
/// the method used here.
///
/// The discriminating shape: under a cap sitting between the two footprints, the windowed path must
/// COMPLETE and the fully-resident path must FAIL to allocate. Both run through the same
/// `run_windowed` — the resident arm is `BlockPlan::resident`, one all-covering window — so this
/// cannot be a comparison between two different loaders.
///
/// The negative control is the load-bearing half: without asserting the resident path actually OOMs,
/// "the windowed path completed" is satisfied by a cap that binds nothing.
#[test]
#[ignore = "real weights + CUDA; set SC15792_Q4"]
fn windowed_fits_an_enforced_cap_the_resident_path_cannot() -> Result<()> {
    // `SC16091_Q4` is accepted as well: this arm is the successor to sc-16091's
    // `rung4_windowed_survives_a_cap_the_resident_path_cannot`, and anyone re-running that story's
    // documented invocation should exercise something rather than silently skip.
    let Some(path) = env_path_opt(TAG, TIER_ENV).or_else(|| env_path_opt(TAG, "SC16091_Q4")) else {
        return Ok(());
    };
    let cap_mib = env_usize("SC15792_CAP_MIB", 1024);

    // Install the cap BEFORE candle allocates anything material: the pool is consulted per
    // allocation, so anything already resident came from the previous pool and is invisible to it.
    let capped = rung4_support::CappedPool::install(0, cap_mib * 1024 * 1024)
        .expect("install a capped pool");
    // Trap 1: the counters must follow the CAPPED pool. Reading the default pool here reports ~0
    // while every allocation lands in the capped one, which would make the budget assertion below
    // trivially true (sc-16091 flagged exactly this for any harness combining the two).
    let pool = capped.counters();

    let tier = Tier::open(&path, MLX_GROUP_SIZE).expect("open the packed tier");
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let cancel = CancelFlag::default();
    let n = tier.n_blocks();
    // The host's real VRAM, disclosed here too: a cap is an ALLOCATOR ceiling on a 95.6 GiB card, not
    // a physical small card, and a report that omits the host invites the SC-15256 misreading.
    disclose_host(TAG, &pool);
    println!(
        "[{TAG}] ENFORCED CAP {cap_mib} MiB | {} blocks, {:.1} MiB/block on disk",
        n,
        tier.total_block_bytes() as f64 / n as f64 / MIB
    );

    quiesce_and_reset(&dev, &pool)?;
    let windowed = windowed_step(&tier, &dev, 1, &cancel);
    let win_used = pool.used_high() as f64 / MIB;
    let win_reserved = pool.reserved_high() as f64 / MIB;
    println!(
        "[{TAG}]   windowed (window 1) : {} | used-high {win_used:.1} MiB, reserved-high {win_reserved:.1} MiB",
        if windowed.is_ok() { "COMPLETED" } else { "FAILED" }
    );

    quiesce_and_reset(&dev, &pool)?;
    let resident = windowed_step(&tier, &dev, n, &cancel);
    let res_used = pool.used_high() as f64 / MIB;
    println!(
        "[{TAG}]   resident (all {n})    : {} | used-high {res_used:.1} MiB",
        match &resident {
            Ok(_) => "COMPLETED".to_string(),
            Err(e) => format!("FAILED — {e}"),
        }
    );

    windowed.expect("the windowed path must fit inside the enforced cap");
    let err = resident.expect_err(
        "NEGATIVE CONTROL FAILED: the fully-resident path completed under the cap, so the cap is \
         not binding this workload and the windowed arm's success proves nothing. Lower \
         SC15792_CAP_MIB.",
    );
    println!("[{TAG}]   resident error: {err}");
    assert!(
        win_reserved <= cap_mib as f64,
        "the windowed reserved high-water ({win_reserved:.1} MiB) exceeded the cap ({cap_mib} MiB) \
         — the counters are not following the capped pool"
    );
    Ok(())
}

/// **The configuration SC-15791 explicitly left untested and asked this story to cover.**
///
/// Its release-semantics arms all ran on one device and one stream, where CUDA's stream-ordered
/// allocator guarantees reuse ordering a priori — it said so, and refused to treat that as licence to
/// relax sc-12195's phase-boundary sync. It named the nearest untested shape: a window driver
/// materializing from a **worker thread**. (Its own `overlap_prefetch` arm did that, and it flagged
/// the result as unproven.)
///
/// This drives the real driver from a spawned thread over the same real packed tier and requires the
/// output to be bit-identical to the main-thread run. The window bound must also still hold there —
/// a driver that behaved differently off-thread could hold output while silently losing the bound.
///
/// **What a green here does and does not mean.** It is a negative result on one configuration, not a
/// proof that no cross-thread hazard exists: candle hands out one stream per device, so a worker
/// thread sharing this `Device` shares that stream and inherits its ordering. The genuinely untested
/// shape is two *distinct* `Device` instances for one GPU, which no candle provider builds today.
/// This does not license removing sc-12195's sync, and `residency.rs` is untouched by SC-15792.
#[test]
#[ignore = "real weights + CUDA; set SC15792_Q4"]
fn worker_thread_materialization_is_bit_identical() -> Result<()> {
    let Some((tier, dev, pool)) = open_tier() else {
        return Ok(());
    };
    let cancel = CancelFlag::default();

    quiesce_and_reset(&dev, &pool)?;
    let on_main = windowed_step(&tier, &dev, 1, &cancel)?.to_scalar::<f32>()?;
    let main_live = pool.used_high() as f64 / MIB;
    let main_reserved = pool.reserved_high() as f64 / MIB;

    quiesce_and_reset(&dev, &pool)?;
    let on_worker = std::thread::scope(|s| {
        s.spawn(|| -> Result<f32> {
            Ok(windowed_step(&tier, &dev, 1, &cancel)?.to_scalar::<f32>()?)
        })
        .join()
        .expect("the worker thread must not panic")
    })?;
    let worker_live = pool.used_high() as f64 / MIB;
    let worker_reserved = pool.reserved_high() as f64 / MIB;

    println!(
        "[{TAG}] WORKER THREAD: main {on_main} (live {main_live:.1} / reserved {main_reserved:.1} MiB) \
         | worker {on_worker} (live {worker_live:.1} / reserved {worker_reserved:.1} MiB)"
    );
    assert_eq!(
        on_worker, on_main,
        "materializing windows from a worker thread changed the output — this is the sc-12195-adjacent \
         configuration, and a mismatch here is a reuse-ordering defect, not a tolerance question"
    );
    assert!(
        (worker_live - main_live).abs() < 0.05 * main_live,
        "the window bound did not hold off the main thread: {worker_live:.1} MiB against \
         {main_live:.1} MiB on it"
    );
    Ok(())
}
