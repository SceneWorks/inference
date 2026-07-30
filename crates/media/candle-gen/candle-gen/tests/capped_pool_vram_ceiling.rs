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

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::{lin, QLinear, MLX_GROUP_SIZE};

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------------
// The capped pool
// ---------------------------------------------------------------------------------------------------

mod capped {
    use candle_gen::candle_core::cuda::cudarc::driver::sys;
    use std::ffi::c_void;

    /// An explicitly created, size-capped stream-ordered pool installed as device `ordinal`'s
    /// **current** pool. Restores the device's original pool on drop, because this is a
    /// process-global device property and leaving it installed would silently cap every later test.
    pub struct CappedPool {
        pool: sys::CUmemoryPool,
        previous: sys::CUmemoryPool,
        ordinal: i32,
    }

    fn device(ordinal: i32) -> Option<sys::CUdevice> {
        unsafe {
            if sys::cuInit(0) != sys::CUresult::CUDA_SUCCESS {
                return None;
            }
            let mut dev: sys::CUdevice = 0;
            (sys::cuDeviceGet(&mut dev, ordinal) == sys::CUresult::CUDA_SUCCESS).then_some(dev)
        }
    }

    impl CappedPool {
        /// Create a pool limited to `cap_bytes` and make it the device's current pool.
        ///
        /// MUST be installed before candle allocates anything material. The pool is consulted per
        /// allocation, so a later install would still cap subsequent allocations — but anything
        /// already resident came from the previous pool and is invisible to the cap.
        pub fn install(ordinal: i32, cap_bytes: usize) -> Option<Self> {
            let dev = device(ordinal)?;
            unsafe {
                // The device's existing current pool, so it can be restored.
                let mut previous: sys::CUmemoryPool = std::ptr::null_mut();
                if sys::cuDeviceGetMemPool(&mut previous, dev) != sys::CUresult::CUDA_SUCCESS {
                    return None;
                }

                let mut props: sys::CUmemPoolProps = std::mem::zeroed();
                props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
                props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
                props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                props.location.id = ordinal;
                // The whole point: an enforced ceiling, unlike a balloon.
                props.maxSize = cap_bytes;

                let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
                if sys::cuMemPoolCreate(&mut pool, &props) != sys::CUresult::CUDA_SUCCESS {
                    return None;
                }
                if sys::cuDeviceSetMemPool(dev, pool) != sys::CUresult::CUDA_SUCCESS {
                    sys::cuMemPoolDestroy(pool);
                    return None;
                }
                Some(Self {
                    pool,
                    previous,
                    ordinal,
                })
            }
        }

        fn attr(&self, which: sys::CUmemPool_attribute) -> u64 {
            let mut v: u64 = 0;
            unsafe {
                if sys::cuMemPoolGetAttribute(
                    self.pool,
                    which,
                    (&mut v as *mut u64).cast::<c_void>(),
                ) != sys::CUresult::CUDA_SUCCESS
                {
                    return 0;
                }
            }
            v
        }

        /// **Read the CAPPED pool, not the default one.** SC-15791's probe reads
        /// `cuDeviceGetDefaultMemPool`; under a custom current pool that is the wrong handle and would
        /// report ~0 while the real allocations happen elsewhere. Any harness combining the two must
        /// switch to the current pool.
        pub fn used(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
        }

        pub fn used_high(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
        }

        pub fn reserved_high(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)
        }

        pub fn reset_high(&self) {
            for which in [
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            ] {
                let mut zero: u64 = 0;
                unsafe {
                    sys::cuMemPoolSetAttribute(
                        self.pool,
                        which,
                        (&mut zero as *mut u64).cast::<c_void>(),
                    );
                }
            }
        }
    }

    impl Drop for CappedPool {
        fn drop(&mut self) {
            unsafe {
                if let Some(dev) = device(self.ordinal) {
                    sys::cuDeviceSetMemPool(dev, self.previous);
                }
                sys::cuMemPoolDestroy(self.pool);
            }
        }
    }
}

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
    let pool = capped::CappedPool::install(0, CAP).expect("install a capped pool");
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

    let pool = capped::CappedPool::install(0, 256 * 1024 * 1024).expect("capped pool");
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
// C: the payoff — does rung 4 survive a cap the resident path cannot?
// ---------------------------------------------------------------------------------------------------

/// Minimal tier plumbing, duplicated from `rung4_block_streaming_spike.rs` rather than shared: these
/// are separate test binaries and the spike is deliberately throwaway. SC-15792 will own the real one.
struct Tier {
    path: PathBuf,
    packed: Vec<Vec<String>>,
    dense: Vec<Vec<String>>,
    dims: HashMap<String, (usize, usize)>,
}

impl Tier {
    fn open(path: &Path) -> Result<Self> {
        // SAFETY: an immutable HF-cache blob; nothing rewrites it mid-test.
        let st = unsafe { MmapedSafetensors::new(path)? };
        let views = st.tensors();
        let n = views
            .iter()
            .filter_map(|(k, _)| k.strip_prefix("layers."))
            .filter_map(|r| r.split('.').next())
            .filter_map(|i| i.parse::<usize>().ok())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        assert!(n > 0, "no `layers.N.` blocks in {}", path.display());

        let shapes: HashMap<&str, &[usize]> =
            views.iter().map(|(k, v)| (k.as_str(), v.shape())).collect();
        let (mut packed, mut dense) = (vec![Vec::new(); n], vec![Vec::new(); n]);
        let mut dims = HashMap::new();
        for (k, v) in &views {
            let Some(b) = k
                .strip_prefix("layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
            else {
                continue;
            };
            if let Some(base) = k.strip_suffix(".scales") {
                let w = shapes[format!("{base}.weight").as_str()];
                packed[b].push(base.to_string());
                dims.insert(base.to_string(), (w[0], v.shape()[1] * MLX_GROUP_SIZE));
            } else if let Some(base) = k.strip_suffix(".weight") {
                if !shapes.contains_key(format!("{base}.scales").as_str()) {
                    dense[b].push(k.to_string());
                }
            } else if !k.ends_with(".biases") {
                dense[b].push(k.to_string());
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            packed,
            dense,
            dims,
        })
    }

    fn n_blocks(&self) -> usize {
        self.packed.len()
    }

    fn view(&self, dev: &Device) -> Result<VarBuilder<'static>> {
        // SAFETY: immutable HF-cache blob; a fresh mmap per view.
        let st = unsafe { MmapedSafetensors::new(&self.path)? };
        Ok(VarBuilder::from_backend(
            Box::new(st),
            DType::F32,
            dev.clone(),
        ))
    }
}

fn materialize(
    tier: &Tier,
    view: &VarBuilder,
    range: Range<usize>,
) -> Result<Vec<(Vec<QLinear>, Vec<Tensor>)>> {
    let mut out = Vec::new();
    for b in range {
        let mut lins = Vec::new();
        for base in &tier.packed[b] {
            let (o, i) = tier.dims[base];
            lins.push(lin(view, base, i, o, false)?);
        }
        let mut dense = Vec::new();
        for key in &tier.dense[b] {
            dense.push(view.get_unchecked_dtype(key, DType::F32)?);
        }
        out.push((lins, dense));
    }
    Ok(out)
}

/// **The verdict SC-15791 could not reach.** Under an enforced cap sitting between the two
/// footprints, the windowed path must complete and the fully-resident path must fail to allocate.
///
/// That is a genuine discriminating result rather than gate arithmetic — the thing the balloon could
/// not deliver, because on WDDM the balloon is not a ceiling and this is.
#[test]
#[ignore = "needs a CUDA host and the hosted z-image q4 tier (SC16091_Q4 env)"]
fn rung4_windowed_survives_a_cap_the_resident_path_cannot() -> Result<()> {
    let Ok(q4) = std::env::var("SC16091_Q4") else {
        println!("[sc-16091] SKIP: SC16091_Q4 not set");
        return Ok(());
    };
    // SC-15791 measured window 1 at ~256 MiB reserved and the resident stack at ~3488 MiB. A 1 GiB
    // cap sits between them with wide margin on both sides.
    const CAP: usize = 1024 * 1024 * 1024;

    let pool = capped::CappedPool::install(0, CAP).expect("install a capped pool");
    let dev = Device::new_cuda(0)?;
    let tier = Tier::open(Path::new(&q4))?;
    println!(
        "\n[sc-16091] ENFORCED CAP {} MiB, {} blocks. window 1 needs ~256 MiB; resident needs ~3488 MiB.",
        CAP / (1024 * 1024),
        tier.n_blocks(),
    );

    // Windowed: one block at a time. Must succeed.
    pool.reset_high();
    let mut windowed_err = None;
    for b in 0..tier.n_blocks() {
        let view = tier.view(&dev)?;
        match materialize(&tier, &view, b..b + 1) {
            Ok(blocks) => {
                if let Err(e) = dev.synchronize() {
                    windowed_err = Some(e.to_string());
                    break;
                }
                drop(blocks);
            }
            Err(e) => {
                windowed_err = Some(e.to_string());
                break;
            }
        }
    }
    let windowed_peak = pool.used_high();
    println!(
        "  windowed (window 1): {} | peak used {:.1} MiB, reserved-high {:.1} MiB",
        match &windowed_err {
            None => "COMPLETED".to_string(),
            Some(e) => format!("FAILED — {e}"),
        },
        windowed_peak as f64 / MIB,
        pool.reserved_high() as f64 / MIB,
    );

    // Resident: all blocks at once. Must fail under the same cap.
    pool.reset_high();
    let view = tier.view(&dev)?;
    let resident = materialize(&tier, &view, 0..tier.n_blocks())
        .and_then(|blocks| dev.synchronize().map(|_| blocks));
    let resident_err = match &resident {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    };
    println!(
        "  resident (all {} blocks): {} | peak used {:.1} MiB",
        tier.n_blocks(),
        match &resident_err {
            None => "COMPLETED".to_string(),
            Some(e) => format!("FAILED — {e}"),
        },
        pool.used_high() as f64 / MIB,
    );
    drop(resident);

    assert!(
        windowed_err.is_none(),
        "the windowed path must survive a {} MiB cap: {windowed_err:?}",
        CAP / (1024 * 1024)
    );
    assert!(
        resident_err.is_some(),
        "the resident path COMPLETED under a {} MiB cap despite needing ~3488 MiB — the cap is not \
         binding this path either, so it is no better than the balloon and must not be reported as a \
         small-card verdict",
        CAP / (1024 * 1024)
    );
    assert!(
        windowed_peak as usize <= CAP,
        "the windowed peak {} exceeded the cap it supposedly ran under",
        windowed_peak
    );
    println!(
        "  ⇒ DISCRIMINATING RESULT: rung 4's window bound lets a model run inside an ENFORCED budget \
         that the resident path cannot fit. This is measured behaviour under a real ceiling, not gate \
         arithmetic on a 96 GiB card."
    );
    Ok(())
}
