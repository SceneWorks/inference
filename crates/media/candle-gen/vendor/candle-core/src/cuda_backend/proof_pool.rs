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

mod recovery;

pub const ENV: &str = "SCENEWORKS_CUDA_PROOF_POOL_BYTES";

struct Installed {
    _context: Arc<CudaContext>,
    pool: usize,
    cap_bytes: usize,
    // Retain until process exit, including when callers retain an exported cuBLAS handle after
    // dropping its CudaDevice. The small proof process declares these live buffers in its cap.
    blas_workspaces: HashMap<usize, Arc<CudaSlice<u8>>>,
    successful_recoveries: u64,
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
            successful_recoveries: 0,
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
    if !std::ptr::eq(
        current.cast::<std::ffi::c_void>(),
        stream.cu_stream().cast::<std::ffi::c_void>(),
    ) {
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

/// Positive allocator-recovery witness for the dedicated raw-allocation proof. Never reset.
pub fn successful_recoveries(ordinal: usize) -> Result<Option<u64>> {
    let Some(state) = INSTALLED.get() else {
        return Ok(None);
    };
    let state = state
        .lock()
        .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
    Ok(state.get(&ordinal).map(|entry| entry.successful_recoveries))
}

pub(super) fn is_installed(ordinal: usize) -> Result<bool> {
    let Some(state) = INSTALLED.get() else {
        return Ok(false);
    };
    let state = state
        .lock()
        .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
    Ok(state.contains_key(&ordinal))
}

fn is_allocation_oom(error: cudarc::driver::DriverError) -> bool {
    error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
}

/// Preserve ordinary allocations exactly. Only an installed proof pool's allocation OOM
/// may trim unused backing in that same pool and retry the original request once.
pub(super) fn allocate<T>(
    stream: &Arc<CudaStream>,
    requested_bytes: Option<usize>,
    operation: &'static str,
    caller: &std::panic::Location<'_>,
    mut allocation: impl FnMut() -> std::result::Result<T, cudarc::driver::DriverError>,
) -> Result<T> {
    let error = match allocation() {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };
    if !is_allocation_oom(error) {
        return Err(error).w();
    }
    let ordinal = stream.context().ordinal();
    let Some(before) = snapshot(ordinal)? else {
        return Err(error).w();
    };
    let requested_description = requested_bytes
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "overflow".into());
    let report = |phase: &str, stats: &Snapshot| {
        eprintln!("CUDA_PROOF_POOL_ALLOCATION phase={phase} operation={operation} requested_bytes={requested_description} logical_ordinal={ordinal} stream=0x{:x} cap_bytes={} used_bytes={} used_high_bytes={} reserved_bytes={} reserved_high_bytes={} caller={caller}",
            stream.cu_stream() as usize, stats.cap_bytes, stats.used_bytes, stats.used_high_bytes, stats.reserved_bytes, stats.reserved_high_bytes);
    };
    report("oom", &before);
    let Some(requested_bytes) = requested_bytes else {
        return Err(error).w();
    };
    if stream.capture_status().w()? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err(error).w();
    }
    // The installed entry retains the pool/context until process exit. Do not hold its mutex
    // across allocation; CUDA slice drops can complete on other streams.
    let (pool, context) = {
        let state = INSTALLED
            .get()
            .expect("snapshot found installed pool")
            .lock()
            .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
        let installed = state
            .get(&ordinal)
            .expect("installed pool is process-lifetime");
        (installed.pool, installed._context.clone())
    };
    struct DriverReclaim<'a, F> {
        ordinal: usize,
        pool: usize,
        context: Arc<CudaContext>,
        report: &'a F,
    }
    impl<F: Fn(&str, &Snapshot)> recovery::Reclaim for DriverReclaim<'_, F> {
        fn snapshot(&mut self, phase: &str) -> Result<recovery::State> {
            let stats = snapshot(self.ordinal)?.ok_or_else(|| {
                crate::Error::Msg("CUDA proof pool disappeared during OOM recovery".into())
            })?;
            (self.report)(phase, &stats);
            Ok(recovery::State {
                cap_bytes: stats.cap_bytes,
                used_bytes: stats.used_bytes,
            })
        }
        fn trim(&mut self) -> Result<()> {
            self.context.bind_to_thread().w()?;
            if active_pool(self.context.cu_device())? as usize != self.pool {
                crate::bail!("CUDA current pool changed before OOM trim");
            }
            // SAFETY: the process-lifetime installed pool is still current on this retained
            // context. TrimTo never releases live or outstanding allocations and performs no
            // context/stream synchronization that could interfere with capture elsewhere.
            unsafe { sys::cuMemPoolTrimTo(self.pool as sys::CUmemoryPool, 0).result() }.w()
        }
    }
    let mut reclaim = DriverReclaim {
        ordinal,
        pool,
        context,
        report: &report,
    };
    match recovery::reclaim_and_retry(
        requested_bytes as u64,
        before.cap_bytes,
        &mut reclaim,
        &mut allocation,
    )? {
        None => Err(error).w(),
        Some(result) => {
            if result.is_ok() {
                let mut state = INSTALLED
                    .get()
                    .expect("installed pool")
                    .lock()
                    .map_err(|_| crate::Error::Msg("CUDA proof-pool state was poisoned".into()))?;
                let entry = state.get_mut(&ordinal).expect("process-lifetime pool");
                entry.successful_recoveries =
                    entry.successful_recoveries.checked_add(1).ok_or_else(|| {
                        crate::Error::Msg("CUDA proof recovery counter overflow".into())
                    })?;
            }
            let after = snapshot(ordinal)?.ok_or_else(|| {
                crate::Error::Msg("CUDA proof pool disappeared after OOM retry".into())
            })?;
            report(
                if result.is_ok() {
                    "recovered"
                } else {
                    "retry_failed"
                },
                &after,
            );
            result.w()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_uncapped_state_does_not_install_or_recover_anything() {
        assert!(!is_installed(0).unwrap());
        assert!(snapshot(0).unwrap().is_none());
        assert!(successful_recoveries(0).unwrap().is_none());
    }

    #[test]
    fn allocation_only_recovers_the_driver_oom_error_class() {
        use cudarc::driver::DriverError;
        assert!(is_allocation_oom(DriverError(
            sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
        )));
        for status in [
            sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            sys::CUresult::CUDA_ERROR_INVALID_CONTEXT,
            sys::CUresult::CUDA_ERROR_LAUNCH_FAILED,
            sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
        ] {
            assert!(!is_allocation_oom(DriverError(status)));
        }
    }

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
