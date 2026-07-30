//! sc-16091 — **can a capped CUDA memory pool emulate a small card on this dev box?** (epic 15448)
//!
//! ## Why this exists
//!
//! SC-15791 needed to show that rung 4's window bound is real on a *small* card, and could not: this
//! host is a ~96 GiB RTX PRO 6000 under Windows/WDDM, which **does not enforce a VRAM ceiling**.
//! Ballooning to 1.93 GiB driver-visible free and then running a path whose pool RESERVED peak is
//! 3.41 GiB completed anyway — bit-identical peak, wall time 1.07×. The driver silently absorbed a
//! working set that cannot physically fit, so neither completion nor wall time detects the spill and
//! the whole balloon method is unsound here. SC-15791 correctly reported UNVERIFIED.
//!
//! A review of that story (Codex, on sc-16091) pointed out that CUDA has an *enforced* limit the
//! balloon does not: `CUmemPoolProps.maxSize` on an explicitly created pool, installed as the
//! device's **current** pool via `cuDeviceSetMemPool`. It demonstrated the primitive rejecting a raw
//! `cuMemAllocAsync` past a 64 MiB cap on this exact box.
//!
//! **That demonstration left the load-bearing question open**, which is what this file closes:
//! the primitive working says nothing about whether *candle* is subject to it. Candle could have
//! used `cuMemAllocFromPoolAsync` with a pool handle of its own, or `cuMemAlloc_v2`, either of which
//! bypasses the current-pool setting entirely and would make the cap decorative.
//!
//! It does not: cudarc 0.19.8 calls bare `cuMemAllocAsync(ptr, bytes, stream)` with no pool argument
//! (`driver/result.rs:824`), and CUDA specifies that form draws from the current pool of the stream's
//! device. `capped_pool_binds_candles_allocator` proves that end-to-end rather than by reading.
//!
//! ## The limitation that keeps this short of a physical card
//!
//! A capped pool is an **allocator** ceiling, not a device ceiling. The CUDA context, cuBLAS/cuBLASLt
//! workspaces, and any `cuMemAlloc_v2` path live outside it. `non_pool_overhead_is_measured` quantifies
//! that so a cap can be chosen as `target_card − overhead` instead of guessed.
//!
//! ```text
//! SC16091_Q4=<...>/q4/transformer/model.safetensors   (only the rung-4 arm needs it)
//! cargo test -p candle-gen --features cuda --release --test capped_pool_vram_ceiling \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is mandatory: these tests mutate a process-global device property.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Result, Tensor};

mod rung4_support;
use rung4_support::{CappedPool, GIB, MIB};

// ---------------------------------------------------------------------------------------------------
// The capped pool
// ---------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------
// A: does the cap actually bind CANDLE's allocator?
// ---------------------------------------------------------------------------------------------------

/// **The gap in the sc-16091 review's evidence.** It proved a raw `cuMemAllocAsync` respects the cap;
/// this proves *candle tensors* do, which is the only version that matters for a tier verdict.
///
/// Allocates 64 MiB candle tensors under a 512 MiB cap on a ~96 GiB card. Success would mean the cap
/// is decorative and the whole route is dead; the assertion is that allocation fails at roughly the
/// cap, not at physical VRAM.
#[test]
#[ignore = "needs a CUDA host; mutates a process-global device property"]
fn capped_pool_binds_candles_allocator() -> Result<()> {
    const CAP: usize = 512 * 1024 * 1024;
    const CHUNK_MIB: usize = 64;

    // via nvidia-smi, NOT cuMemGetInfo_v2 — the latter needs a current context and returns (0, 0)
    // before candle builds one.
    println!(
        "[sc-16091] HOST: {:.1} GiB total, {:.1} GiB free (nvidia-smi, pre-context). Installing a {} \
         MiB pool cap.",
        candle_gen::gpu::nvidia_smi_min_total_gib().unwrap_or(0.0),
        candle_gen::gpu::nvidia_smi_rendered_free_gib().unwrap_or(0.0),
        CAP / (1024 * 1024),
    );
    let capped = CappedPool::install(0, CAP).expect("install a capped pool");
    // Trap 1: the counters must follow the CAPPED pool, not `cuDeviceGetDefaultMemPool`, which
    // reports ~0 while every allocation lands here.
    let pool = capped.counters();
    let dev = Device::new_cuda(0)?;

    let chunk_elems = CHUNK_MIB * 1024 * 1024 / 4; // f32
    let mut held: Vec<Tensor> = Vec::new();
    let mut failed_at = None;
    for i in 1..=32 {
        match Tensor::zeros(chunk_elems, DType::F32, &dev) {
            Ok(t) => {
                held.push(t);
                if dev.synchronize().is_err() {
                    failed_at = Some(i * CHUNK_MIB);
                    break;
                }
            }
            Err(_) => {
                failed_at = Some(i * CHUNK_MIB);
                break;
            }
        }
    }
    let allocated_mib = held.len() * CHUNK_MIB;
    println!(
        "  allocated {allocated_mib} MiB of candle tensors before failure at {:?} MiB | pool used \
         {:.1} MiB, reserved-high {:.1} MiB, cap {} MiB",
        failed_at,
        pool.used() as f64 / MIB,
        pool.reserved_high() as f64 / MIB,
        CAP / (1024 * 1024),
    );

    assert!(
        failed_at.is_some(),
        "candle allocated {allocated_mib} MiB under a {} MiB cap without ever failing — the cap does \
         NOT bind candle's allocator, so this route is dead and sc-16091 must fall back to arithmetic",
        CAP / (1024 * 1024)
    );
    let cap_mib = CAP / (1024 * 1024);
    assert!(
        allocated_mib <= cap_mib,
        "candle allocated {allocated_mib} MiB, past the {cap_mib} MiB cap"
    );
    // And it must fail NEAR the cap, not absurdly early — otherwise the cap is not the thing binding.
    assert!(
        allocated_mib >= cap_mib / 2,
        "candle only reached {allocated_mib} MiB under a {cap_mib} MiB cap; something other than the \
         cap is binding and the emulation would be misleading"
    );
    println!(
        "  ⇒ CONFIRMED: the cap binds candle's allocator. cudarc calls bare `cuMemAllocAsync`, which \
         draws from the device's CURRENT pool, so `cuDeviceSetMemPool` redirects every candle tensor \
         through the capped pool."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// B: what lives OUTSIDE the cap?
// ---------------------------------------------------------------------------------------------------

/// A capped pool bounds pool allocations, not the process. To emulate an N GiB card the cap must be
/// `N − (everything outside the pool)`, so that overhead has to be a measured number.
///
/// The baseline MUST come from `nvidia-smi`, not `cuMemGetInfo_v2`: the driver call requires a current
/// context, so calling it before candle builds one returns `(0, 0)` — which the first version of this
/// test did, silently producing a 0 MiB "overhead". `nvidia-smi` reads the device from outside the
/// process and needs no context, so it can measure the before-state honestly.
#[test]
#[ignore = "needs a CUDA host"]
fn non_pool_overhead_is_measured() -> Result<()> {
    let smi_free = || {
        candle_gen::gpu::nvidia_smi_rendered_free_gib()
            .expect("nvidia-smi free for the rendered GPU")
    };
    // Before ANY CUDA activity in this process. Device-level, so it includes other tenants; the delta
    // below is what matters and an idle GPU keeps that honest.
    let free_before = smi_free();

    let capped = CappedPool::install(0, 256 * 1024 * 1024).expect("capped pool");
    let pool = capped.counters();
    let dev = Device::new_cuda(0)?;
    // Force the context and the usual kernel/cuBLAS machinery to materialize.
    let a = Tensor::randn(0f32, 1f32, (512, 512), &dev)?;
    let b = a.matmul(&a)?;
    let _ = b.sum_all()?.to_scalar::<f32>()?;
    dev.synchronize()?;

    let free_after = smi_free();
    let pool_used_gib = pool.used() as f64 / GIB;
    let process_footprint = free_before - free_after;
    let outside_pool = process_footprint - pool_used_gib;

    println!(
        "\n[sc-16091] NON-POOL OVERHEAD (CUDA context + libraries + kernel images)\n  \
         nvidia-smi free before any CUDA use : {free_before:.3} GiB\n  \
         after context + one matmul          : {free_after:.3} GiB\n  \
         ⇒ this process's device footprint   : {:.0} MiB\n  \
         ⇒ of which inside the capped pool   : {:.0} MiB\n  \
         ⇒ OUTSIDE the pool (uncapped)       : {:.0} MiB",
        process_footprint * 1024.0,
        pool_used_gib * 1024.0,
        outside_pool * 1024.0,
    );
    println!(
        "  ⇒ to emulate an N GiB card set the cap to N minus this overhead. Quoting the cap alone as \
         the emulated card size understates the real footprint by that much, and it is the reason a \
         capped pool is an ALLOCATOR ceiling rather than a device ceiling.\n  \
         CAVEAT: this is a FLOOR, measured against a trivial workload (one 512x512 matmul). Outside-\
         the-pool cost grows with the number of distinct kernels loaded and with cuBLAS/cuBLASLt \
         workspaces, so a real render's overhead is larger. Re-measure it for the workload being \
         emulated rather than reusing this figure."
    );
    assert!(
        process_footprint > 0.0,
        "expected a measurable CUDA context cost, saw {process_footprint:.3} GiB — the baseline is \
         not a before-state and the number is not an overhead measurement"
    );
    assert!(
        outside_pool > 0.0,
        "expected some of the footprint to sit OUTSIDE the capped pool ({outside_pool:.3} GiB). If \
         everything were inside it, the cap would be a true device ceiling and this caveat could be \
         dropped — verify before believing that."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// C: the payoff — MOVED
// ---------------------------------------------------------------------------------------------------
//
// The rung-4 verdict this file originally carried ("windowed survives a cap the resident path
// cannot") now lives in `rung4_block_window_real_weights.rs` as
// `windowed_fits_an_enforced_cap_the_resident_path_cannot`, and is STRONGER there in three ways
// (SC-15792):
//
//   * it drives `candle_gen::block_window::run_windowed` — the shipped driver — where this file's
//     version hand-rolled its own `for b in 0..n { view; materialize; sync; drop }` loop. A test that
//     proves a bound for a loop nothing ships is the fork rung 4 exists to prevent;
//   * both arms go through ONE code path (`BlockPlan` at window 1 vs at `n_blocks`), so the
//     comparison cannot be between two loaders that could each be wrong;
//   * the budget assertion is on RESERVED rather than USED — reserved is the unit the admission gate
//     reads, and it ran ~48% higher.
//
// The measured result is unchanged in shape: under a 1024 MiB enforced cap on this 95.6 GiB host,
// windowed COMPLETED at 261.4 MiB used / 416.0 MiB reserved while resident FAILED with
// `CUDA_ERROR_OUT_OF_MEMORY` at 949.0 MiB used.
//
// Arms A and B above stay here: they are about the METHOD (does the cap bind candle's allocator, and
// what sits outside the pool), which is sc-16091's finding and not rung-4-specific.
