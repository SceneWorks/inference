//! Opt-in allocator enforcement for explicit CUDA memory proofs (SW-6925).
//!
//! `SCENEWORKS_CUDA_PROOF_POOL_BYTES` installs a private, size-limited current pool before
//! Candle creates streams or library handles. This caps stream-ordered allocations, not the
//! device: contexts, module images, library-internal and foreign allocations remain outside it.
//! Acceptance must reserve for those allocations and observe the whole device separately.
//! This is independent of SceneWorks' admission-budget override. With the variable absent,
//! construction does not change any pool. Run a proof in a dedicated short-lived process;
//! the installed pool and primary context intentionally live until process exit, so no tensor
//! or outstanding stream can outlive its allocator. Removing/changing the cap needs a restart.

use super::WrapErr;
use crate::Result;
use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub const ENV: &str = "SCENEWORKS_CUDA_PROOF_POOL_BYTES";

struct Installed {
    _context: Arc<CudaContext>,
    pool: usize,
    cap_bytes: usize,
    // Retain until process exit, including when callers retain an exported cuBLAS handle after
    // dropping its CudaDevice. The small proof process declares these live buffers in its cap.
    blas_workspaces: HashMap<usize, Arc<CudaSlice<u8>>>,
}

static INSTALLED: OnceLock<Mutex<HashMap<usize, Installed>>> = OnceLock::new();

/// The installed pool's continuous CUDA-driver counters. These are not sampled process peaks.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub logical_ordinal: usize,
    pub cap_bytes: u64,
    pub used_bytes: u64,
    pub used_high_bytes: u64,
    pub reserved_bytes: u64,
    pub reserved_high_bytes: u64,
}

fn requested_cap(raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw else { return Ok(None) };
    let cap = raw.parse::<usize>().ok().filter(|cap| *cap > 0);
    match cap {
        Some(cap) => Ok(Some(cap)),
        None => crate::bail!(
            "{ENV} must be a positive integer byte count; restart with a valid proof cap"
        ),
    }
}

fn attribute(pool: sys::CUmemoryPool, attr: sys::CUmemPool_attribute) -> Result<u64> {
    let mut value = 0u64;
    // SAFETY: the caller holds INSTALLED's lock and the installed pool lives until process exit;
    // all attributes read here use a uint64_t out parameter.
    unsafe { sys::cuMemPoolGetAttribute(pool, attr, (&mut value as *mut u64).cast()).result() }
        .w()?;
    Ok(value)
}

fn active_pool(device: sys::CUdevice) -> Result<sys::CUmemoryPool> {
    let mut pool = std::ptr::null_mut();
    // SAFETY: device belongs to a retained CUDA context; pool is a live out pointer.
    unsafe { sys::cuDeviceGetMemPool(&mut pool, device).result() }.w()?;
    Ok(pool)
}

pub(super) fn configure(context: &Arc<CudaContext>) -> Result<()> {
    let raw = match std::env::var(ENV) {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => crate::bail!("{ENV} is not valid Unicode"),
    };
    let requested = requested_cap(raw.as_deref())?;
    if requested.is_none() && INSTALLED.get().is_none() {
        return Ok(());
    }
    let mut installed = INSTALLED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
    let ordinal = context.ordinal();
    context.bind_to_thread().w()?;
    if let Some(previous) = installed.get(&ordinal) {
        if requested != Some(previous.cap_bytes) {
            crate::bail!("CUDA proof cap changed or disappeared on logical device {ordinal}; restart the proof process");
        }
        if active_pool(context.cu_device())? as usize != previous.pool {
            crate::bail!("CUDA current pool changed after proof-cap installation on logical device {ordinal}");
        }
        return Ok(());
    }
    let Some(cap_bytes) = requested else {
        return Ok(());
    };
    if !context.has_async_alloc() {
        crate::bail!("CUDA proof cap requires stream-ordered allocation support; refusing an unenforced legacy allocator");
    }
    context.synchronize().w()?;
    let previous = active_pool(context.cu_device())?;
    let mut default = std::ptr::null_mut();
    // SAFETY: all out pointers are live; the context retains this valid device.
    unsafe { sys::cuDeviceGetDefaultMemPool(&mut default, context.cu_device()).result() }.w()?;
    if previous != default {
        crate::bail!("CUDA proof cap must be installed before any other custom pool");
    }
    if attribute(
        previous,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
    )? != 0
    {
        crate::bail!("CUDA proof cap was requested after existing pool allocations; restart before loading tensors");
    }
    // SAFETY: properties are initialized as required by cuMemPoolCreate; the pool is device-local,
    // non-exported, and is retained for this process's lifetime after installation.
    let pool = unsafe {
        let mut properties: sys::CUmemPoolProps = std::mem::zeroed();
        properties.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        properties.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = context.cu_device();
        properties.maxSize = cap_bytes;
        let mut pool = std::ptr::null_mut();
        sys::cuMemPoolCreate(&mut pool, &properties).result().w()?;
        if let Err(error) = sys::cuDeviceSetMemPool(context.cu_device(), pool)
            .result()
            .w()
        {
            let _ = sys::cuMemPoolDestroy(pool);
            return Err(error);
        }
        pool
    };
    installed.insert(
        ordinal,
        Installed {
            _context: context.clone(),
            pool: pool as usize,
            cap_bytes,
            blas_workspaces: HashMap::new(),
        },
    );
    eprintln!("CUDA_PROOF_POOL installed logical_ordinal={ordinal} cap_bytes={cap_bytes} scope=stream_ordered_pool outside_pool=unbounded_requires_separate_reserve_and_observation");
    Ok(())
}

/// Keep cuBLAS's explicit workspace inside the proof pool. Called after its constructor's
/// initial SetStream and immediately before every Candle matmul: SetStream resets a user
/// workspace, even when it selects the same stream. H3 has no direct handle/stream mutation.
/// An externally changed stream is refused rather than applying a buffer with wrong ownership.
pub(super) fn bind_cublas_workspace(
    stream: &Arc<CudaStream>,
    blas: &cudarc::cublas::CudaBlas,
) -> Result<()> {
    let Some(state) = INSTALLED.get() else {
        return Ok(());
    };
    let mut state = state
        .lock()
        .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
    let Some(installed) = state.get_mut(&stream.context().ordinal()) else {
        return Ok(());
    };
    installed._context.bind_to_thread().w()?;
    if active_pool(installed._context.cu_device())? as usize != installed.pool {
        crate::bail!("CUDA current pool changed before proof cuBLAS allocation");
    }
    let mut current = std::ptr::null_mut();
    // SAFETY: the live cuBLAS handle owns its stream; current is a valid out pointer.
    unsafe { cudarc::cublas::sys::cublasGetStream_v2(*blas.handle(), &mut current).result() }
        .w()?;
    if current as usize != stream.cu_stream() as usize {
        crate::bail!("CUDA proof cuBLAS handle was moved to another stream; refusing untracked workspace ownership");
    }
    let handle = *blas.handle() as usize;
    if let std::collections::hash_map::Entry::Vacant(entry) =
        installed.blas_workspaces.entry(handle)
    {
        // NVIDIA recommends 32 MiB for sm90+; explicit storage prevents fallback to the
        // library's default outside-pool workspace. This is proof-only, not a math-mode change.
        entry.insert(Arc::new(stream.alloc_zeros::<u8>(32 * 1024 * 1024).w()?));
    }
    let workspace = installed
        .blas_workspaces
        .get(&handle)
        .expect("inserted workspace");
    let (pointer, _guard) = workspace.device_ptr(stream);
    // SAFETY: process-lifetime pool storage outlives every use of this handle. Same-stream
    // ordering protects initialization; ordinary H3 execution serializes cuBLAS operations.
    unsafe {
        cudarc::cublas::sys::cublasSetWorkspace_v2(
            *blas.handle(),
            pointer as *mut _,
            workspace.len(),
        )
        .result()
    }
    .w()?;
    Ok(())
}

/// Read the actual current capped pool. A replaced pool is an error, never a plausible zero peak.
/// `None` means no proof cap was installed for this logical device.
pub fn snapshot(ordinal: usize) -> Result<Option<Snapshot>> {
    let Some(state) = INSTALLED.get() else {
        return Ok(None);
    };
    let state = state
        .lock()
        .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
    let Some(installed) = state.get(&ordinal) else {
        return Ok(None);
    };
    installed._context.bind_to_thread().w()?;
    let pool = active_pool(installed._context.cu_device())?;
    if pool as usize != installed.pool {
        crate::bail!(
            "CUDA current pool changed after proof-cap installation on logical device {ordinal}"
        );
    }
    use sys::CUmemPool_attribute as A;
    Ok(Some(Snapshot {
        logical_ordinal: ordinal,
        cap_bytes: installed.cap_bytes as u64,
        used_bytes: attribute(pool, A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
        used_high_bytes: attribute(pool, A::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
        reserved_bytes: attribute(pool, A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        reserved_high_bytes: attribute(pool, A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::requested_cap;

    #[test]
    fn proof_cap_is_explicit_and_invalid_values_fail_closed() {
        assert_eq!(requested_cap(None).unwrap(), None);
        assert_eq!(
            requested_cap(Some("23622320128")).unwrap(),
            Some(23622320128)
        );
        for invalid in ["", "0", "-1", "1.5", "24GiB", " 24", "18446744073709551616"] {
            assert!(requested_cap(Some(invalid)).is_err(), "{invalid}");
        }
    }
}
