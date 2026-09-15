//! Driver stream-ordered memory-pool introspection — the accurate VRAM counters for a CUDA render
//! (sc-12818, hoisted and widened by SC-15792).
//!
//! candle 0.10 allocates every tensor through `cuMemAllocAsync` on the device's stream, which CUDA
//! specifies draws from that device's **current** pool. The driver tracks that pool's occupancy
//! continuously, so its attributes are candle's real memory behaviour rather than a sample of it.
//!
//! # Why not `nvidia-smi` sampling
//!
//! `crate::testkit::PeakSampler` polls `nvidia-smi memory.used` every ~40 ms and therefore misses
//! sub-poll transients — the brief im2col / VAE-decode / attention spikes the pool allocates and
//! releases between polls. That understated the Wan A14B peak ~2x. `USED_MEM_HIGH` is a continuous
//! high-water and does not.
//!
//! # The four counters are not interchangeable, and the difference decides answers
//!
//! | counter | what it is | MLX analogue |
//! |---|---|---|
//! | [`used`](MemPool::used) | bytes currently live in the pool | `get_active_memory` |
//! | [`used_high`](MemPool::used_high) | high-water of concurrently-live bytes | `get_peak_memory` |
//! | [`reserved`](MemPool::reserved) | bytes the pool holds FROM THE DRIVER (live + cached-free) | — |
//! | [`reserved_high`](MemPool::reserved_high) | high-water of driver-reserved bytes | — |
//!
//! **RESERVED is the unit an admission gate reads.** `nvidia-smi memory.used` reports driver-reserved
//! bytes, so `gpu::nvidia_smi_min_free_gib` and `testkit::VramProbe` both consume
//! RESERVED. It runs materially above USED — measured at 48% higher for a single 107.9 MiB rung-4
//! block (SC-15791). A peak reported in USED and compared against a gate reading RESERVED is an
//! apples-to-oranges pass.
//!
//! Reading only USED also produces the wrong answer to "did the memory come back?". Measured on a
//! bare drop with no synchronize: USED falls to 0 while RESERVED stays at 160.0 MiB and
//! driver-visible free recovers **0.0 MiB**. See [`release_threshold`](MemPool::release_threshold).
//!
//! # Two traps, both of which have already produced a plausible-looking wrong number
//!
//! 1. **[`device_default`](MemPool::device_default) is the wrong handle under a custom pool.** A
//!    capped pool installed as the device's *current* pool via `cuDeviceSetMemPool` (the sc-16091
//!    small-card method) receives every allocation, while the *default* pool reports ~0 — so
//!    "the peak stayed within budget" becomes trivially true. Use [`MemPool::from_raw`] with the
//!    capped pool's own handle.
//! 2. **Reset the watermarks only after the device and pool are quiesced.** The high-water attributes
//!    are write-to-zero, and zeroing one while the pool still physically holds pages makes it snap
//!    straight back to the current reserved value — so the next measurement silently inherits the
//!    previous phase's. SC-15791's first sweep reported window 1's reserved peak as the fully-resident
//!    control's figure for exactly this reason.
//!
//! Every accessor returns `Option`/`bool` rather than a defaulted number: a silent zero lets a broken
//! probe print a plausible report and bank a green. In a measurement context `unwrap_or(0)` re-creates
//! that hazard — prefer `expect`.

use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys;

/// A handle to a CUDA stream-ordered memory pool.
///
/// `Copy`-cheap and inert: it borrows nothing and owns nothing, so dropping it never destroys the
/// pool. Obtain the one candle actually allocates from with [`device_default`](Self::device_default),
/// or wrap an explicitly created pool with [`from_raw`](Self::from_raw) — read trap 1 in the module
/// docs before assuming the former.
#[derive(Clone, Copy)]
pub struct MemPool(sys::CUmemoryPool);

impl MemPool {
    /// The **default** stream-ordered pool for logical device `ordinal`, or `None` on any driver
    /// error.
    ///
    /// `cuInit` is idempotent (candle calls it too), so this is safe to run before candle builds its
    /// context — the default pool is a stable per-device handle, unaffected by context retain.
    ///
    /// The ordinal is candle's **logical** device: the driver API honours `CUDA_VISIBLE_DEVICES`, so
    /// logical 0 is the card candle renders on, NOT `testkit::probe_gpu`'s physical
    /// nvidia-smi ordinal.
    ///
    /// **This is not the right handle if something has installed a custom current pool** — see trap 1
    /// in the module docs.
    pub fn device_default(ordinal: i32) -> Option<Self> {
        unsafe {
            if sys::cuInit(0) != sys::CUresult::CUDA_SUCCESS {
                return None;
            }
            let mut dev: sys::CUdevice = 0;
            if sys::cuDeviceGet(&mut dev, ordinal) != sys::CUresult::CUDA_SUCCESS {
                return None;
            }
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            if sys::cuDeviceGetDefaultMemPool(&mut pool, dev) != sys::CUresult::CUDA_SUCCESS {
                return None;
            }
            Some(Self(pool))
        }
    }

    /// Wrap a pool handle the caller already owns — an explicitly created pool, typically one carrying
    /// a `CUmemPoolProps.maxSize` cap and installed as the device's current pool to emulate a smaller
    /// card (sc-16091).
    pub fn from_raw(pool: sys::CUmemoryPool) -> Self {
        Self(pool)
    }

    /// The raw handle, for driver calls this type does not wrap.
    pub fn raw(&self) -> sys::CUmemoryPool {
        self.0
    }

    fn attr(&self, which: sys::CUmemPool_attribute) -> Option<u64> {
        let mut value: u64 = 0;
        let ok = unsafe {
            sys::cuMemPoolGetAttribute(self.0, which, (&mut value as *mut u64).cast::<c_void>())
                == sys::CUresult::CUDA_SUCCESS
        };
        ok.then_some(value)
    }

    fn set_attr(&self, which: sys::CUmemPool_attribute, mut value: u64) -> bool {
        unsafe {
            sys::cuMemPoolSetAttribute(self.0, which, (&mut value as *mut u64).cast::<c_void>())
                == sys::CUresult::CUDA_SUCCESS
        }
    }

    /// Bytes currently **live** in the pool.
    pub fn used(&self) -> Option<u64> {
        self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
    }

    /// High-water of concurrently-live pool bytes — the true concurrent-live peak.
    pub fn used_high(&self) -> Option<u64> {
        self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
    }

    /// Bytes the pool currently holds from the driver (live + cached-free).
    pub fn reserved(&self) -> Option<u64> {
        self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
    }

    /// High-water of driver-reserved bytes — **the peak in the admission gate's own unit.**
    pub fn reserved_high(&self) -> Option<u64> {
        self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)
    }

    /// Bytes the pool may retain across a synchronization before returning them to the driver.
    ///
    /// Load-bearing for any "does the memory come back?" claim: neither candle nor cudarc sets this,
    /// so it sits at the driver default of **0** — release everything on every synchronize. That is
    /// why a bare drop decrements USED immediately while driver-visible free recovers nothing until
    /// the next synchronize, and why [`trim`](Self::trim) has nothing to do and must not be credited
    /// with the recovery (SC-15791).
    pub fn release_threshold(&self) -> Option<u64> {
        self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)
    }

    /// Reset both high-water marks to zero. Returns whether both writes landed.
    ///
    /// Quiesce first — see trap 2 in the module docs.
    pub fn reset_high_water(&self) -> bool {
        self.set_attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH, 0)
            && self.set_attr(
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                0,
            )
    }

    /// Return cached-free pool pages to the driver. Success is NOT the same as having freed anything:
    /// at a [`release_threshold`](Self::release_threshold) of 0 there is no retained cache to return,
    /// so on a stock candle setup this is a **no-op**.
    pub fn trim(&self) -> bool {
        unsafe { sys::cuMemPoolTrimTo(self.0, 0) == sys::CUresult::CUDA_SUCCESS }
    }
}

/// Driver-level `(free, total)` bytes — what `nvidia-smi` reports, i.e. what a smaller card's VRAM
/// gate would actually see. `None` on any driver error.
///
/// **Requires a current context.** Called before candle builds one, `cuMemGetInfo_v2` returns
/// `(0, 0)`, which reads as a plausible measurement and is pure artifact — it silently reported a
/// 0 MiB per-process overhead in sc-16091's first draft. For a baseline taken before any CUDA use,
/// read `nvidia-smi` from outside the process instead (`testkit::used_mib`).
pub fn mem_info() -> Option<(u64, u64)> {
    let (mut free, mut total) = (0usize, 0usize);
    let ok = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) == sys::CUresult::CUDA_SUCCESS };
    ok.then_some((free as u64, total as u64))
}

/// Reset logical device `ordinal`'s **default** pool `USED_MEM_HIGH` watermark so a later
/// [`cuda_mempool_used_high_bytes`] reads the peak of just the work since this call. Returns whether
/// the reset landed.
///
/// Retained at its original signature for the provider real-weight harnesses that already drive it
/// (`candle-gen-wan`'s `vram_probe` / `vae16_decode_sweep`, `candle-gen-svd`'s `real_weights_smoke`).
/// New code wanting RESERVED, the release threshold, or a non-default pool should use [`MemPool`].
pub fn reset_cuda_mempool_high_water(ordinal: i32) -> bool {
    MemPool::device_default(ordinal)
        .map(|pool| pool.set_attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH, 0))
        .unwrap_or(false)
}

/// Logical device `ordinal`'s **default** pool `USED_MEM_HIGH` watermark in bytes, or `None` on any
/// driver error. See [`reset_cuda_mempool_high_water`] for why this narrower pair still exists.
pub fn cuda_mempool_used_high_bytes(ordinal: i32) -> Option<u64> {
    MemPool::device_default(ordinal)?.used_high()
}
