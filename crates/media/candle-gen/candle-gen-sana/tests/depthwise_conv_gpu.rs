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
const MAX_MEAN_MS: f64 = 100.0;

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

    let started = Instant::now();
    for _ in 0..MEASURED_CALLS {
        let output = input.conv2d(&kernel, 1, 1, 1, CHANNELS)?;
        dev.synchronize()?;
        assert_eq!(output.dims(), input.dims());
        drop(output);
    }
    let mean_ms = started.elapsed().as_secs_f64() * 1e3 / MEASURED_CALLS as f64;

    eprintln!(
        "[sc-12111] SANA-1.6B conv_depth {CHANNELS}x{LATENT_EDGE}x{LATENT_EDGE}: \
         {mean_ms:.3} ms/call ({MEASURED_CALLS} measured calls)"
    );
    assert!(
        mean_ms < MAX_MEAN_MS,
        "SANA depthwise conv took {mean_ms:.3} ms/call; expected < {MAX_MEAN_MS:.1} ms. \
         The O(groups) Candle path may have regressed (pre-fix baseline: 982 ms/call)."
    );
    Ok(())
}
