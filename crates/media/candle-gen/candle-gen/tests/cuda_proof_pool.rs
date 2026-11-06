//! Explicit allocator proof only. This performs raw allocation/memset, no learned computation.
//! Run in its own process: the proof pool intentionally persists until process exit.
#![cfg(feature = "cuda")]

use candle_gen::candle_core::{cuda_backend::proof_pool, DType, Device, Tensor};

#[test]
#[ignore = "explicit CUDA allocator proof; dedicated process and physical-device binding required"]
fn constructor_binds_candle_to_an_enforced_pool() -> candle_gen::candle_core::Result<()> {
    const MIB: usize = 1024 * 1024;
    const CHUNK: usize = 64 * MIB;
    let requested = std::env::var(proof_pool::ENV)
        .expect("explicit SCENEWORKS_CUDA_PROOF_POOL_BYTES is required")
        .parse::<usize>()
        .expect("positive integer pool cap");
    assert!(
        (256 * MIB..=1024 * MIB).contains(&requested),
        "primitive proof cap must be 256-1024 MiB"
    );
    assert_eq!(requested % CHUNK, 0, "use an exact chunk-aligned proof cap");

    let device = Device::new_cuda(0)?;
    let second_stream_device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let (free_bytes, _) = cuda
        .cuda_stream()
        .context()
        .mem_get_info()
        .map_err(candle_gen::candle_core::Error::wrap)?;
    assert!(
        free_bytes > requested * 2,
        "primitive proof needs physical headroom above its cap to distinguish cap enforcement from ordinary OOM"
    );
    // A SetStream call resets a user workspace. Exercise the existing matmul boundary's
    // rebind after a same-stream reset, using raw arithmetic rather than a learned layer.
    unsafe {
        candle_gen::candle_core::cuda_backend::cudarc::cublas::result::set_stream(
            *cuda.cublas_handle().handle(),
            cuda.cuda_stream().cu_stream() as _,
        )
        .map_err(candle_gen::candle_core::Error::wrap)?;
    }
    {
        let a = Tensor::ones((4, 4), DType::F32, &device)?;
        assert_eq!(a.matmul(&a)?.to_vec2::<f32>()?, vec![vec![4.; 4]; 4]);
    }
    device.synchronize()?;
    let before = proof_pool::snapshot(0)?.expect("constructor installed a proof pool");
    assert_eq!(before.cap_bytes, requested as u64);
    let mut allocations = Vec::new();
    let mut refused = false;
    for index in 0..(requested / CHUNK + 2) {
        let selected = if index % 2 == 0 {
            &device
        } else {
            &second_stream_device
        };
        match Tensor::zeros(CHUNK, DType::U8, selected) {
            Ok(tensor) => allocations.push(tensor),
            Err(_) => {
                refused = true;
                break;
            }
        }
    }
    device.synchronize()?;
    second_stream_device.synchronize()?;
    let peak = proof_pool::snapshot(0)?.expect("proof pool remains installed");
    let held_bytes = allocations.len() * CHUNK;
    assert!(refused, "Candle bypassed the pool limit");
    assert!(
        held_bytes >= requested / 2,
        "failed before the pool cap could be the limiting resource"
    );
    assert!(
        held_bytes <= requested,
        "allocated beyond the requested cap"
    );
    assert!(
        peak.used_high_bytes >= held_bytes as u64,
        "counters did not observe held tensors"
    );
    assert!(
        peak.reserved_high_bytes <= requested as u64,
        "driver rounded/reserved beyond the proof cap"
    );

    drop(allocations);
    device.synchronize()?;
    second_stream_device.synchronize()?;
    let released = proof_pool::snapshot(0)?.expect("proof pool survives tensor release");
    assert_eq!(
        released.used_bytes, before.used_bytes,
        "allocation proof leaked live pool storage"
    );
    assert!(released.reserved_high_bytes <= requested as u64);
    let successful_recovery_count =
        proof_pool::successful_recoveries(0)?.expect("proof pool remains installed");
    // Recovery is observational: the driver may satisfy freed storage directly. A zero
    // count proves no successful retry occurred, not a failure of cap enforcement.
    // No invented all-device peak: these are driver-continuous pool counters only.
    println!(
        "CUDA_PROOF_POOL_EVIDENCE {{\"logicalOrdinal\":0,\"capBytes\":{},\"heldBytes\":{},\"usedHighBytes\":{},\"reservedHighBytes\":{},\"overCapRefused\":true,\"bothConstructors\":true,\"cublasStreamResetRebound\":true,\"successfulRecoveryCount\":{},\"fragmentationRecoveryExercised\":{},\"scope\":\"stream-ordered allocator plus explicit cuBLAS workspace; context/library-internal/foreign allocations excluded\"}}",
        requested, held_bytes, released.used_high_bytes, released.reserved_high_bytes,
        successful_recovery_count, successful_recovery_count > 0
    );
    Ok(())
}
