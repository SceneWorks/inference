//! CUDA performance regression for the SANA-1.6B Mix-FFN depthwise convolution (sc-12111).
//!
//! Run on an exclusive GPU in release mode:
//! `CUDA_COMPUTE_CAP=120 cargo test --locked -j 1 -p candle-gen-sana --test depthwise_conv_gpu \
//!     --features cuda --release -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};

const CHANNELS: usize = 11_200;
const LATENT_EDGE: usize = 32;
const MEASURED_CALLS: usize = 3;
const MAX_CALL_MS: f64 = 100.0;

/// Guards the real SANA-1.6B serving shape against Candle's former one-launch-per-group path.
///
/// The pre-fix baseline on the exclusive RTX PRO 6000 rig was 982 ms/call. The deliberately loose
/// 100 ms ceiling leaves room for slower CUDA devices and allocator noise while still failing by
/// nearly an order of magnitude if the O(groups) decomposition returns.
#[test]
#[ignore = "exclusive-GPU performance gate; run in release mode"]
fn sana_1600m_depthwise_conv_is_not_launch_bound() -> candle_gen::candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let input = Tensor::ones((1, CHANNELS, LATENT_EDGE, LATENT_EDGE), DType::F32, &dev)?;
    let kernel = Tensor::ones((CHANNELS, 1, 3, 3), DType::F32, &dev)?;

    // Warm the allocator and load/compile every kernel before timing.
    for _ in 0..MEASURED_CALLS {
        let warmup = input.conv2d(&kernel, 1, 1, 1, CHANNELS)?;
        dev.synchronize()?;
        assert_eq!(warmup.dims(), input.dims());
        drop(warmup);
    }

    // sc-19556: timed PER CALL and reduced with `min`, not as one mean over the batch.
    //
    // This gate's published claim (sc-12111) genuinely IS wall-clock — "the O(groups) decomposition
    // has not returned" is a statement about launch count that Candle exposes no counter for — so
    // this keeps a duration bound rather than pretending to a clock-free instrument it cannot have.
    // What changes is which STATISTIC the bound reads. Every call here runs the identical
    // convolution, so contention can only ever push a call SLOWER; the fastest of several calls is
    // therefore a lower bound on what the hardware did, not a sample of a noisy distribution. A
    // mean folds every descheduled call straight into the number being asserted on, which is how a
    // busy runner turns this into a false red charged to an unrelated PR.
    //
    // The bound itself is unchanged and stays deliberately loose: the pre-fix baseline was 982
    // ms/call against a 100 ms ceiling, so the ~10x margin is what makes this robust rather than
    // the statistic being generous. The per-call spread is printed beside the minimum so a reader
    // can see when the host was contended instead of having to infer it.
    let mut per_call_ms = Vec::with_capacity(MEASURED_CALLS);
    for _ in 0..MEASURED_CALLS {
        let started = Instant::now();
        let output = input.conv2d(&kernel, 1, 1, 1, CHANNELS)?;
        dev.synchronize()?;
        per_call_ms.push(started.elapsed().as_secs_f64() * 1e3);
        assert_eq!(output.dims(), input.dims());
        drop(output);
    }
    let fastest_ms = per_call_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let slowest_ms = per_call_ms.iter().copied().fold(0.0f64, f64::max);

    eprintln!(
        "[sc-12111] SANA-1.6B conv_depth {CHANNELS}x{LATENT_EDGE}x{LATENT_EDGE}: \
         fastest {fastest_ms:.3} ms/call, slowest {slowest_ms:.3} ms/call \
         ({MEASURED_CALLS} measured calls; the bound reads the fastest)"
    );
    assert!(
        fastest_ms < MAX_CALL_MS,
        "SANA depthwise conv took {fastest_ms:.3} ms/call at its FASTEST; expected < \
         {MAX_CALL_MS:.1} ms. Contention cannot explain this one — the fastest call is a hardware \
         lower bound — so the O(groups) Candle path may have regressed (pre-fix: 982 ms/call)."
    );
    Ok(())
}
