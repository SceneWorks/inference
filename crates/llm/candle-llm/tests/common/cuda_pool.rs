use candle_core::cuda_backend::cudarc::driver::{result, sys, CudaContext};
use serde_json::{json, Value};

/// The device's current memory pool (the one `cuMemAllocAsync` on the default stream serves from).
pub(super) fn pool(ctx: &CudaContext) -> sys::CUmemoryPool {
    ctx.bind_to_thread().unwrap();
    // SAFETY: the live context is bound to this thread and `cu_device` is owned by it.
    unsafe { result::device::get_mem_pool(ctx.cu_device()) }.unwrap()
}

fn pool_attr(ctx: &CudaContext, attr: sys::CUmemPool_attribute) -> u64 {
    let mut value = 0u64;
    // SAFETY: every attribute read here is a u64 counter of the live pool.
    unsafe { result::mem_pool::get_attribute(pool(ctx), attr, (&mut value as *mut u64).cast()) }
        .unwrap();
    value
}

/// Reset the pool's high watermarks (a set to zero resets them to the current value).
pub(super) fn reset_pool_peaks(ctx: &CudaContext) {
    for attr in [
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    ] {
        let mut zero = 0u64;
        // SAFETY: both high-watermark attributes take a u64; zero is the documented reset.
        unsafe { result::mem_pool::set_attribute(pool(ctx), attr, (&mut zero as *mut u64).cast()) }
            .unwrap();
    }
}

/// One synchronized sample: device-wide used, and the pool's reserved / used and their peaks.
pub(super) fn sample(ctx: &CudaContext) -> Value {
    use sys::CUmemPool_attribute as A;
    ctx.synchronize().unwrap();
    let (free, total) = ctx.mem_get_info().unwrap();
    json!({
        "device_used_bytes": (total - free) as u64,
        "pool_reserved_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        "pool_used_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
        "pool_reserved_peak_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
        "pool_used_peak_bytes": pool_attr(ctx, A::CU_MEMPOOL_ATTR_USED_MEM_HIGH),
    })
}
