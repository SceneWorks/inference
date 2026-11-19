//! SW-7197: a captured CUDA graph never keeps the address of the vendored quantized matmul's
//! shared, grows-only Q8_1 scratch workspace. Explicit CUDA proof only (one process per proof):
//! the workspaces are process-wide statics.
//!
//! The defect it pins: `fast_mmvq` (rows 1-8) and `fast_mmq` (rows 9+) quantized their activations
//! into one per-device workspace and freed the slice they outgrew. A graph captured over such a
//! matmul kept the old slice's address, so after a later eager call grew the workspace, every replay
//! wrote its Q8_1 activations into freed memory — over whatever the allocator had placed there
//! (on the Qwen3.8-27B, a DSpine block draft's context). The proof captures a quantized matmul,
//! grows the workspace with a far wider eager matmul, fills many fresh buffers of the freed slice's
//! sizes with a sentinel, replays the graph, and requires every sentinel intact and the replay's
//! output equal to the eager result. With the fix reverted (a capture reading the shared workspace)
//! the replays write into the freed slice, which the sentinel allocations reuse from the stream's
//! pool, so a sentinel changes; the fix gives the capture its own scratch, recorded and freed inside
//! the graph.
#![cfg(feature = "cuda")]

use candle_gen::candle_core::cuda_backend::cudarc::driver::sys;
use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{DType, Device, Error, Module, Result, Tensor};

/// splitmix64-hashed deterministic values in about [-1, 1].
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// A Q8_0 matmul of `n × k` weights on `device`.
fn q8_matmul(n: usize, k: usize, seed: u64, device: &Device) -> Result<QMatMul> {
    let w = Tensor::from_vec(pseudo_random(n * k, seed), (n, k), &Device::Cpu)?;
    QMatMul::from_qtensor(QTensor::quantize_onto(&w, GgmlDType::Q8_0, device)?)
}

/// Capture `rows` rows through a Q8_0 matmul, grow the shared workspace past it with an eager
/// matmul of `grow_rows` rows over 32,768 columns, allocate sentinels, replay: the sentinels stay
/// intact and the replay's output is the eager result.
fn a_graph_keeps_its_own_scratch(rows: usize, grow_rows: usize) -> Result<()> {
    const SENTINEL: f32 = 7.25;
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let (k, n) = (256usize, 128usize);
    let mm = q8_matmul(n, k, 0x7197_0001, &device)?;
    let x = Tensor::from_vec(pseudo_random(rows * k, 0x7197_0002), (1, rows, k), &device)?;
    let want = __model_proof_output!(mm.forward(&x))?;
    let want = want.flatten_all()?.to_vec1::<f32>()?;
    let out = Tensor::zeros((1, rows, n), DType::F32, &device)?;
    let stream = cuda.cuda_stream();

    let graph = {
        // The guard covers the warm-up and the recording, as the decode graph runner's does.
        let _htod = cuda.enable_cuda_graph_htod_cache();
        mm.forward(&x)?;
        device.synchronize()?;
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(Error::wrap)?;
        let recorded = mm.forward(&x).and_then(|y| out.slice_set(&y, 0, 0));
        let graph = stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(Error::wrap)?;
        recorded?;
        graph.expect("the capture recorded a graph")
    };

    // An eager matmul far wider than the captured one: the shared workspace grows and frees the
    // slice it held.
    let wide_k = 32_768usize;
    let wide = q8_matmul(16, wide_k, 0x7197_0003, &device)?;
    let wide_x = Tensor::from_vec(
        pseudo_random(grow_rows * wide_k, 0x7197_0004),
        (1, grow_rows, wide_k),
        &device,
    )?;
    __model_proof_output!(wide.forward(&wide_x))?;
    device.synchronize()?;

    // Fresh buffers across the freed slice's size range, each filled with the sentinel.
    let sizes = [16usize, 64, 144, 256, 576, 1024, 2304, 4096, 9216, 16_384];
    let sentinels = (0..512)
        .map(|i| Tensor::full(SENTINEL, sizes[i % sizes.len()], &device))
        .collect::<Result<Vec<_>>>()?;
    device.synchronize()?;

    for _ in 0..4 {
        graph.launch().map_err(Error::wrap)?;
    }
    device.synchronize()?;

    let got = out.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        got, want,
        "{rows}-row replay: the output is the eager result"
    );
    for (i, s) in sentinels.iter().enumerate() {
        let values = s.to_vec1::<f32>()?;
        let changed = values.iter().filter(|&&v| v != SENTINEL).count();
        assert_eq!(
            changed, 0,
            "{rows}-row replay wrote {changed} of sentinel {i}'s {} values: the graph kept a freed \
             workspace address",
            values.len()
        );
    }
    Ok(())
}

/// The vector path (`fast_mmvq`, rows 1-8) and the tiled path (`fast_mmq`, rows 9+).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_captured_quantized_matmul_keeps_its_own_scratch() -> Result<()> {
    a_graph_keeps_its_own_scratch(5, 8)?;
    a_graph_keeps_its_own_scratch(12, 64)?;

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantized_workspace_graph::a_captured_quantized_matmul_keeps_its_own_scratch"
    );
    Ok(())
}
