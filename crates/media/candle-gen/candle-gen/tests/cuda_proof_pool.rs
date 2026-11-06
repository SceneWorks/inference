//! Explicit allocator proof only. This performs raw allocation/memset, no learned computation.
//! Run in its own process: the proof pool intentionally persists until process exit.
#![cfg(feature = "cuda")]

use candle_gen::candle_core::{cuda_backend::proof_pool, DType, Device, Tensor};

/// Force a real cross-stream cached-backing OOM inside the SAME capped pool. The fixture
/// alone disables CUDA's normal reuse shortcuts; production pool attributes are unchanged.
fn prove_cross_stream_recovery(
    device: &Device,
    other_stream: &Device,
) -> candle_gen::candle_core::Result<()> {
    use candle_gen::candle_core::cuda_backend::cudarc::driver::sys;
    use sys::CUmemPool_attribute as A;
    let cuda = device.as_cuda_device()?;
    let mut pool = std::ptr::null_mut();
    let wrap = candle_gen::candle_core::Error::wrap;
    // SAFETY: the constructor installed and retains this device's pool until process exit.
    unsafe {
        sys::cuDeviceGetMemPool(&mut pool, cuda.cuda_stream().context().cu_device()).result()
    }
    .map_err(wrap)?;
    let reuse = [
        A::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        A::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        A::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    ];
    let mut saved_reuse = [0i32; 3];
    let mut saved_threshold = 0u64;
    // SAFETY: each attribute receives its documented int or uint64_t out pointer.
    unsafe {
        for (attr, value) in reuse.iter().zip(&mut saved_reuse) {
            sys::cuMemPoolGetAttribute(pool, *attr, (value as *mut i32).cast())
                .result()
                .map_err(wrap)?;
        }
        sys::cuMemPoolGetAttribute(
            pool,
            A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut saved_threshold as *mut u64).cast(),
        )
        .result()
        .map_err(wrap)?;
    }
    let mut after_priming = None;
    // No assertions until all temporary attributes have been restored, including error exits.
    let exercise = (|| -> candle_gen::candle_core::Result<()> {
        let mut disabled = 0i32;
        let mut retained = u64::MAX;
        // SAFETY: the same live pool, with documented attribute types. Cap is never modified.
        unsafe {
            for attr in reuse {
                sys::cuMemPoolSetAttribute(pool, attr, (&mut disabled as *mut i32).cast())
                    .result()
                    .map_err(wrap)?;
            }
            sys::cuMemPoolSetAttribute(
                pool,
                A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut retained as *mut u64).cast(),
            )
            .result()
            .map_err(wrap)?;
        }
        let cached = Tensor::zeros(320 * 1024 * 1024, DType::U8, other_stream)?;
        after_priming = proof_pool::successful_recoveries(0)?;
        // Queue a free on the nonblocking other stream. Fixture-only host observation
        // makes unused backing trimmable; recovery itself must never fence a context.
        drop(cached);
        other_stream.synchronize()?;
        let recovered = Tensor::zeros(320 * 1024 * 1024, DType::U8, device)?;
        device.synchronize()?;
        drop(recovered);
        Ok(())
    })();
    let mut restoration_error = None;
    // SAFETY: restore every saved attribute with its original type even if setup/allocation failed.
    unsafe {
        for (attr, value) in reuse.iter().zip(&mut saved_reuse) {
            if let Err(error) =
                sys::cuMemPoolSetAttribute(pool, *attr, (value as *mut i32).cast()).result()
            {
                restoration_error.get_or_insert(error);
            }
        }
        if let Err(error) = sys::cuMemPoolSetAttribute(
            pool,
            A::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut saved_threshold as *mut u64).cast(),
        )
        .result()
        {
            restoration_error.get_or_insert(error);
        }
    }
    if let Some(error) = restoration_error {
        return Err(wrap(error));
    }
    exercise?;
    device.synchronize()?;
    other_stream.synchronize()?;
    assert!(
        proof_pool::successful_recoveries(0)?.expect("installed pool")
            > after_priming.expect("priming completed"),
        "fixture did not exercise a real allocation OOM recovered by same-pool trim/retry"
    );
    Ok(())
}

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
    assert_eq!(
        requested,
        512 * MIB,
        "recovery proof uses the dedicated512MiB cap"
    );
    prove_cross_stream_recovery(&device, &second_stream_device)?;
    let final_pool = proof_pool::snapshot(0)?.expect("pool remains installed");
    assert_eq!(final_pool.used_bytes, before.used_bytes);
    assert!(final_pool.reserved_high_bytes <= requested as u64);
    // No invented all-device peak: these are driver-continuous pool counters only.
    println!(
        "CUDA_PROOF_POOL_EVIDENCE {{\"logicalOrdinal\":0,\"capBytes\":{},\"heldBytes\":{},\"usedHighBytes\":{},\"reservedHighBytes\":{},\"overCapRefused\":true,\"bothConstructors\":true,\"cublasStreamResetRebound\":true,\"fragmentationRecoveryExercised\":true,\"scope\":\"stream-ordered allocator plus explicit cuBLAS workspace; context/library-internal/foreign allocations excluded\"}}",
        requested, held_bytes, final_pool.used_high_bytes, final_pool.reserved_high_bytes
    );
    Ok(())
}
