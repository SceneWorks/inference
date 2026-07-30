//! sc-15791 — **SPIKE: what does host-to-device block streaming cost on Candle?** (rung 4, epic 15448)
//!
//! The CUDA twin of the MLX rung-4 spike (SC-15744). Its result deliberately does **not** transfer:
//! MLX's mechanism is page-faulting an mmap into unified memory; Candle's is reading a packed tier
//! off disk, repacking it on the **host CPU**, and copying the result across PCIe into a device
//! allocation. Different cost model, different failure modes — so this measures rather than reasons.
//!
//! Run on a CUDA host with the hosted z-image-turbo tiers cached (the SAME snapshot SC-15744
//! measured, so the two backends are like-for-like):
//!
//! ```text
//! SC15791_Q4=<...>/q4/transformer/model.safetensors     REQUIRED by every test
//! SC15791_Q8=<...>/q8/transformer/model.safetensors     optional; q8_tier_cost SKIPs without it
//! SC15791_TARGET_FREE_GIB=1                             optional; constrained_budget_sweep SKIPs without it
//! SC15791_TOKENS=1024                                   optional; per-block compute width for Q5
//! SC15791_REPEATS=3                                     optional; timing samples per window size
//! SC15791_RACE_TOKENS=4096 SC15791_RACE_REPEATS=64      optional; depth of the Q3 race queue
//! SC15791_RACE_CLAIM_MIB=256                            optional; bytes re-claimed in the Q3b probe
//! SC16096_Q4=<...>/q4/transformer/model.safetensors     SC-16096 before/after (required together)
//! SC16096_Q8=<...>/q8/transformer/model.safetensors     SC-16096 before/after (required together)
//! SC16096_REPEATS=3                                     SC-16096 samples (minimum three)
//! cargo test -p candle-gen --features cuda --release --test rung4_block_streaming_spike \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release` is not optional: the dominant cost measured here is host-side Rust, which a debug
//! build inflates ~14× (a first pass measured 3.8 s/block against 0.27 s/block optimized). Any rung-4
//! figure taken from a debug build is meaningless. `--test-threads=1` because every arm reads
//! process-global driver counters.
//!
//! ## What each test answers
//!
//! | Test | Story question |
//! |---|---|
//! | `window_sweep_cost` | Q1 cost per window and its scaling, the loader-overhead decomposition, and the release-guard ablation |
//! | `overlap_prefetch` | Q5 overlap |
//! | `release_semantics` | Q2 does VRAM come back / need a sync; Q3 must `release` be non-trivial |
//! | `packed_quant_per_block` | Q4 the packed-quant triple, per block, bit-exact |
//! | `q8_tier_cost` | whether SC-15744's "q8 ≈ 2× q4" carries to Candle |
//! | `device_format_sidecars_before_after` | SC-16096 q4/q8 cost, parity, host memory, and load-bearing window bound |
//! | `constrained_budget_sweep` | the small-card disclosure |
//!
//! Throwaway measurement code by the story's own terms — the answer is the deliverable. Kept as an
//! `#[ignore]` real-weight test so SC-15792 can re-run it against its implementation. It drives the
//! production packed loader but deliberately does **not** drive
//! `gen_core::block_window::run_windowed`: there is no candle `BlockWindowBackend` yet, and building
//! one is SC-15792's job, not this spike's.
//!
//! The pool accessors below duplicate `candle_gen::testkit::cuda_mempool`'s `default_pool` because
//! testkit exports only `USED_MEM_HIGH` and this spike needs the CURRENT/RESERVED/RELEASE_THRESHOLD
//! set. Extending testkit instead would force this file behind the `testkit` feature, and CI compiles
//! and Clippies candle-gen's cuda-gated tests with `--features cuda` only (ci.yml — a cuda-gated lint
//! "sat red on main until found by hand", sc-12379), so gating it would drop it from that coverage.
//! Hoisting these into testkit belongs in SC-15792, which will have a non-throwaway consumer.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::{
    dequant_mlx_q4_reference, lin, repack_packed_weight, PackedConfig, PackedWeightSidecars,
    QLinear, MLX_GROUP_SIZE,
};

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------------
// Driver memory-pool probe
//
// The mapping to the MLX spike's counters, which is what makes the two comparable at all:
//   MLX get_active_memory  ↔  CU_MEMPOOL_ATTR_USED_MEM_CURRENT
//   MLX get_peak_memory    ↔  CU_MEMPOOL_ATTR_USED_MEM_HIGH
//   MLX get_cache_memory   ↔  RESERVED_MEM_CURRENT − USED_MEM_CURRENT
//   MLX clear_cache()      ↔  cuMemPoolTrimTo(pool, 0)
//
// RESERVED is not a footnote: `nvidia-smi memory.used` reports driver-RESERVED bytes, so RESERVED is
// the unit a VRAM gate consumes, and it exceeds USED by the pool's allocation granularity.
// ---------------------------------------------------------------------------------------------------
mod pool {
    use candle_gen::candle_core::cuda::cudarc::driver::sys;
    use std::ffi::c_void;

    pub struct Pool(sys::CUmemoryPool);

    impl Pool {
        /// The default stream-ordered pool `cuMemAllocAsync` draws from for **logical** device
        /// `ordinal` — the pool candle 0.10 allocates every tensor from.
        pub fn open(ordinal: i32) -> Option<Self> {
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

        /// Read one attribute. Panics rather than returning 0 on a driver error: a silent zero would
        /// let a broken probe print a plausible report and bank a green.
        fn attr(&self, which: sys::CUmemPool_attribute) -> u64 {
            let mut v: u64 = 0;
            let ok = unsafe {
                sys::cuMemPoolGetAttribute(self.0, which, (&mut v as *mut u64).cast::<c_void>())
                    == sys::CUresult::CUDA_SUCCESS
            };
            assert!(ok, "cuMemPoolGetAttribute({which:?}) failed");
            v
        }

        /// Bytes currently **live** in the pool — the MLX `get_active_memory` analogue.
        pub fn used(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
        }

        /// High-water of concurrently-live pool bytes — the MLX `get_peak_memory` analogue.
        pub fn used_high(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)
        }

        /// Bytes the pool holds from the driver (live + cached-free).
        pub fn reserved(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
        }

        /// High-water of driver-reserved bytes — **the peak in the VRAM gate's own unit.**
        pub fn reserved_high(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)
        }

        /// Bytes the pool may retain across a synchronization before returning them to the driver.
        ///
        /// Load-bearing for any "does the memory come back?" claim: neither candle nor cudarc sets
        /// this, so it sits at the driver default of **0** = release everything on every synchronize.
        /// That is why a drop decrements USED immediately while driver-visible free only recovers at
        /// the next synchronize, and why `trim` has nothing to do.
        pub fn release_threshold(&self) -> u64 {
            self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)
        }

        /// Reset both high-water marks (write-to-zero per the driver ABI).
        pub fn reset_high(&self) {
            for which in [
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            ] {
                let mut zero: u64 = 0;
                let ok = unsafe {
                    sys::cuMemPoolSetAttribute(
                        self.0,
                        which,
                        (&mut zero as *mut u64).cast::<c_void>(),
                    ) == sys::CUresult::CUDA_SUCCESS
                };
                assert!(ok, "resetting {which:?} failed — peaks would be stale");
            }
        }

        /// Return cached-free pool pages to the driver — the `clear_cache()` analogue. Success is NOT
        /// the same as having freed anything: at a release threshold of 0 there is no retained cache.
        pub fn trim(&self) {
            let ok = unsafe { sys::cuMemPoolTrimTo(self.0, 0) == sys::CUresult::CUDA_SUCCESS };
            assert!(ok, "cuMemPoolTrimTo failed");
        }
    }

    /// Driver-level `(free, total)` bytes — what `nvidia-smi` reports, i.e. what a smaller card's
    /// VRAM gate would actually see. Panics on driver error rather than reporting `(0, 0)`.
    pub fn mem_info() -> (u64, u64) {
        let (mut free, mut total) = (0usize, 0usize);
        let ok =
            unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) == sys::CUresult::CUDA_SUCCESS };
        assert!(ok, "cuMemGetInfo_v2 failed");
        (free as u64, total as u64)
    }
}

// Windows process-memory probe used by SC-16096. `WorkingSetSize` includes mapped/file-backed pages;
// `PrivateUsage` is private commit and exposes the q8 dense-grid transient the CUDA pool cannot see.
#[cfg(windows)]
mod host_memory {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Peak {
        pub working_set_start: u64,
        pub working_set_peak: u64,
        pub working_set_end: u64,
        pub private_start: u64,
        pub private_peak: u64,
        pub private_end: u64,
    }

    fn counters() -> (u64, u64) {
        let mut counters = ProcessMemoryCountersEx {
            cb: std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
            private_usage: 0,
        };
        let ok = unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
            )
        };
        assert_ne!(ok, 0, "K32GetProcessMemoryInfo failed");
        (
            counters.working_set_size as u64,
            counters.private_usage as u64,
        )
    }

    fn update_max(max: &AtomicU64, value: u64) {
        let mut old = max.load(Ordering::Relaxed);
        while value > old {
            match max.compare_exchange_weak(old, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => old = actual,
            }
        }
    }

    pub fn sample<T>(f: impl FnOnce() -> T) -> (T, Peak) {
        let (working_set_start, private_start) = counters();
        let stop = Arc::new(AtomicBool::new(false));
        let working_set_peak = Arc::new(AtomicU64::new(working_set_start));
        let private_peak = Arc::new(AtomicU64::new(private_start));
        let sampler = {
            let stop = Arc::clone(&stop);
            let working_set_peak = Arc::clone(&working_set_peak);
            let private_peak = Arc::clone(&private_peak);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let (working_set, private) = counters();
                    update_max(&working_set_peak, working_set);
                    update_max(&private_peak, private);
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };
        let value = f();
        stop.store(true, Ordering::Relaxed);
        sampler.join().expect("host-memory sampler");
        let (working_set_end, private_end) = counters();
        (
            value,
            Peak {
                working_set_start,
                working_set_peak: working_set_peak.load(Ordering::Relaxed),
                working_set_end,
                private_start,
                private_peak: private_peak.load(Ordering::Relaxed),
                private_end,
            },
        )
    }
}

#[cfg(not(windows))]
mod host_memory {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Peak {
        pub working_set_start: u64,
        pub working_set_peak: u64,
        pub working_set_end: u64,
        pub private_start: u64,
        pub private_peak: u64,
        pub private_end: u64,
    }

    pub fn sample<T>(f: impl FnOnce() -> T) -> (T, Peak) {
        (f(), Peak::default())
    }
}

// ---------------------------------------------------------------------------------------------------
// The tier under test
// ---------------------------------------------------------------------------------------------------

/// Everything about a packed transformer tier that can be read from the safetensors **header** alone.
struct Tier {
    path: PathBuf,
    packed: Vec<Vec<String>>,
    dense: Vec<Vec<String>>,
    dims: HashMap<String, (usize, usize)>,
    bytes: Vec<usize>,
    n_tensors: usize,
    file_bytes: u64,
}

impl Tier {
    fn open(path: &Path, group_size: usize) -> Result<Self> {
        // SAFETY: an immutable HF-cache blob; nothing rewrites it mid-test.
        let st = unsafe { MmapedSafetensors::new(path)? };
        let views = st.tensors();
        let n_tensors = views.len();

        let n_blocks = views
            .iter()
            .filter_map(|(k, _)| k.strip_prefix("layers."))
            .filter_map(|rest| rest.split('.').next())
            .filter_map(|i| i.parse::<usize>().ok())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        assert!(n_blocks > 0, "no `layers.N.` blocks in {}", path.display());

        let mut packed = vec![Vec::new(); n_blocks];
        let mut dense = vec![Vec::new(); n_blocks];
        let mut bytes = vec![0usize; n_blocks];
        let mut dims = HashMap::new();

        // `.data().len()` is a length on the mmap view, not a read, so no page is faulted here.
        let mut shape: HashMap<&str, (&[usize], usize)> = HashMap::new();
        for (k, v) in &views {
            shape.insert(k.as_str(), (v.shape(), v.data().len()));
        }

        let block_of = |k: &str| -> Option<usize> {
            k.strip_prefix("layers.")?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        };

        for (k, v) in &views {
            let Some(b) = block_of(k) else { continue };
            bytes[b] += v.data().len();
            if let Some(base) = k.strip_suffix(".scales") {
                let w = shape
                    .get(format!("{base}.weight").as_str())
                    .unwrap_or_else(|| panic!("{base}.scales has no {base}.weight sibling"));
                packed[b].push(base.to_string());
                dims.insert(base.to_string(), (w.0[0], v.shape()[1] * group_size));
            } else if let Some(base) = k.strip_suffix(".weight") {
                if !shape.contains_key(format!("{base}.scales").as_str()) {
                    dense[b].push(k.to_string());
                }
            } else if !k.ends_with(".biases") {
                dense[b].push(k.to_string());
            }
        }
        for p in &mut packed {
            p.sort();
        }
        for d in &mut dense {
            d.sort();
        }

        // Per-block figures below are block 0's; assert the stack is uniform so that is honest.
        let (lo, hi) = (*bytes.iter().min().unwrap(), *bytes.iter().max().unwrap());
        assert!(
            hi - lo < hi / 100,
            "blocks are not uniform ({lo}..{hi} bytes) — per-block figures taken from block 0 would \
             misrepresent the stack"
        );

        Ok(Self {
            path: path.to_path_buf(),
            packed,
            dense,
            dims,
            bytes,
            n_tensors,
            file_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        })
    }

    fn n_blocks(&self) -> usize {
        self.packed.len()
    }

    fn all_block_bytes(&self) -> usize {
        self.bytes.iter().sum()
    }

    /// A **fresh** weights view — the `BlockWindowBackend::open_view` analogue. Header-only.
    fn open_view(&self, dev: &Device) -> Result<VarBuilder<'static>> {
        // SAFETY: immutable HF-cache blob; a fresh mmap per view, never mutated behind the mapping.
        let st = unsafe { MmapedSafetensors::new(&self.path)? };
        Ok(VarBuilder::from_backend(
            Box::new(st),
            DType::F32,
            dev.clone(),
        ))
    }

    /// A raw mmap for the corrected-loader arm, which reads the triple on the host.
    fn open_raw(&self) -> Result<MmapedSafetensors> {
        // SAFETY: as above.
        unsafe { MmapedSafetensors::new(&self.path) }
    }
}

/// One materialized transformer block.
struct Block {
    /// `(name, projection, in_dim)`.
    lins: Vec<(String, QLinear, usize)>,
    dense: Vec<Tensor>,
}

impl Block {
    fn dense_bytes(&self) -> usize {
        self.dense
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// Materialize `range` through the **production** packed loader (`candle_gen::quant::lin`) onto the
/// view's device — what a naive `BlockWindowBackend` would do.
///
/// Since SC-16096, `lin_gs` reads its source triple directly on CPU, so this path no longer includes
/// the historical CUDA→CPU round trip. It still performs the invariant format conversion on every
/// call; [`materialize_sidecars`] is the no-conversion shipping window path.
fn materialize(tier: &Tier, view: &VarBuilder, range: Range<usize>) -> Result<Vec<Block>> {
    let mut out = Vec::with_capacity(range.len());
    for b in range {
        let mut lins = Vec::with_capacity(tier.packed[b].len());
        for base in &tier.packed[b] {
            let (out_dim, in_dim) = tier.dims[base];
            lins.push((
                base.clone(),
                lin(view, base, in_dim, out_dim, false)?,
                in_dim,
            ));
        }
        let mut dense = Vec::with_capacity(tier.dense[b].len());
        for key in &tier.dense[b] {
            dense.push(view.get_unchecked_dtype(key, DType::F32)?);
        }
        out.push(Block { lins, dense });
    }
    Ok(out)
}

/// Manual raw-mmap twin of the current production loader: read the packed triple on the host, repack
/// there, and upload only the resulting device-format blocks. It remains in the older SC-15791
/// decomposition as a cross-check; it is not SC-16096's final path because it still converts per call.
fn materialize_host_repack(
    tier: &Tier,
    raw: &MmapedSafetensors,
    dev: &Device,
    range: Range<usize>,
) -> Result<Vec<Block>> {
    let cpu = Device::Cpu;
    let mut out = Vec::with_capacity(range.len());
    for b in range {
        let mut lins = Vec::with_capacity(tier.packed[b].len());
        for base in &tier.packed[b] {
            let (_, in_dim) = tier.dims[base];
            let wq = raw.load(&format!("{base}.weight"), &cpu)?;
            let scales = raw.load(&format!("{base}.scales"), &cpu)?;
            let biases = raw.load(&format!("{base}.biases"), &cpu)?;
            let ql = QLinear::from_packed_gs(&wq, &scales, &biases, None, MLX_GROUP_SIZE, dev)?;
            lins.push((base.clone(), ql, in_dim));
        }
        let mut dense = Vec::with_capacity(tier.dense[b].len());
        for key in &tier.dense[b] {
            dense.push(raw.load(key, dev)?.to_dtype(DType::F32)?);
        }
        out.push(Block { lins, dense });
    }
    Ok(out)
}

/// Reconstruct the exact pre-SC-16096 Krea packed path for the before measurement: source tensors
/// first load on CUDA, then `from_packed_gs` pulls them back to CPU for repacking.
fn materialize_prechange(
    tier: &Tier,
    raw: &MmapedSafetensors,
    dev: &Device,
    range: Range<usize>,
) -> Result<Vec<Block>> {
    let mut out = Vec::with_capacity(range.len());
    for b in range {
        let mut lins = Vec::with_capacity(tier.packed[b].len());
        for base in &tier.packed[b] {
            let (_, in_dim) = tier.dims[base];
            let wq = raw.load(&format!("{base}.weight"), dev)?;
            let scales = raw
                .load(&format!("{base}.scales"), dev)?
                .to_dtype(DType::F32)?;
            let biases = raw
                .load(&format!("{base}.biases"), dev)?
                .to_dtype(DType::F32)?;
            let ql = QLinear::from_packed_gs(&wq, &scales, &biases, None, MLX_GROUP_SIZE, dev)?;
            lins.push((base.clone(), ql, in_dim));
        }
        let mut dense = Vec::with_capacity(tier.dense[b].len());
        for key in &tier.dense[b] {
            dense.push(raw.load(key, dev)?.to_dtype(DType::F32)?);
        }
        out.push(Block { lins, dense });
    }
    Ok(out)
}

/// SC-16096's shipping window path: map already-GGML bytes and transfer them to the target device.
/// The API deliberately accepts neither the MLX source nor quantization parameters.
fn materialize_sidecars(
    tier: &Tier,
    sidecars: &PackedWeightSidecars,
    raw: &MmapedSafetensors,
    dev: &Device,
    range: Range<usize>,
) -> Result<Vec<Block>> {
    let mut out = Vec::with_capacity(range.len());
    for b in range {
        let mut lins = Vec::with_capacity(tier.packed[b].len());
        for base in &tier.packed[b] {
            let (_, in_dim) = tier.dims[base];
            let weight = sidecars.load(base, dev)?;
            lins.push((
                base.clone(),
                QLinear::from_qtensor_dequant(Arc::new(weight), None),
                in_dim,
            ));
        }
        let mut dense = Vec::with_capacity(tier.dense[b].len());
        for key in &tier.dense[b] {
            dense.push(raw.load(key, dev)?.to_dtype(DType::F32)?);
        }
        out.push(Block { lins, dense });
    }
    Ok(out)
}

/// A plausible per-block forward. Accumulates into an on-DEVICE scalar and never reads it back: a
/// `to_scalar` per projection would synchronize the stream on every call and serialize exactly the
/// overlap the Q5 arm is trying to detect.
fn compute(blocks: &[Block], tokens: usize, dev: &Device) -> Result<Tensor> {
    let mut acc = Tensor::zeros((), DType::F32, dev)?;
    for b in blocks {
        for (_, ql, in_dim) in &b.lins {
            let x = Tensor::ones((tokens, *in_dim), DType::F32, dev)?;
            acc = (acc + ql.forward(&x)?.sum_all()?)?;
        }
    }
    Ok(acc)
}

fn env_path_req(key: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(key).unwrap_or_else(|_| panic!("{key} not set — see the module docstring")),
    )
}

/// An OPTIONAL env path: `None` prints a SKIP rather than panicking, so the documented
/// `-- --ignored` invocation does not fail arms whose inputs were not supplied (the house pattern —
/// `mlx_repack_real_weights.rs` uses `.ok()` for its optional tiers).
fn env_path_opt(key: &str) -> Option<PathBuf> {
    match std::env::var(key) {
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            println!("[sc-15791] SKIP: {key} not set");
            None
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// The host's VRAM, disclosed in every report — SC-15256's closing note is that an acceptance
/// measured as gate arithmetic on a 97.9 GB card is not evidence about an 8 GB one.
fn disclose_host(pool: &pool::Pool) {
    let (free, total) = pool::mem_info();
    assert!(
        total > 0 && free > 0,
        "the driver reported no memory — a zeroed probe must not be allowed to satisfy the \
         host-VRAM-disclosure requirement"
    );
    println!(
        "[sc-15791] HOST: CUDA device 0 total {:.1} GiB, free now {:.1} GiB | pool used {:.1} MiB / \
         reserved {:.1} MiB / release-threshold {} bytes",
        total as f64 / GIB,
        free as f64 / GIB,
        pool.used() as f64 / MIB,
        pool.reserved() as f64 / MIB,
        pool.release_threshold(),
    );
}

/// Quiesce the device and the pool, THEN reset the watermarks.
///
/// Order is load-bearing for `RESERVED_MEM_HIGH`. Resetting the watermark to 0 while the pool still
/// physically holds pages makes it snap straight back to the current reserved value, so the next
/// measurement inherits the previous one. That is not hypothetical: the first run of this sweep
/// reported window 1's reserved peak as 3488.0 MiB — exactly the fully-resident control's figure —
/// because the control's pages had not been returned when the reset ran.
fn quiesce_and_reset(dev: &Device, pool: &pool::Pool) -> Result<()> {
    dev.synchronize()?;
    pool.trim();
    pool.reset_high();
    Ok(())
}

fn windows(n_blocks: usize, window: usize) -> impl Iterator<Item = Range<usize>> {
    (0..n_blocks).step_by(window).map(move |s| {
        let e = (s + window).min(n_blocks);
        s..e
    })
}

/// Least-squares fit of `peak ≈ a·window + b` over the sweep, in MiB. `b` is fixed allocator/loader
/// overhead that does not scale with the window width.
fn linear_fit(points: &[(usize, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|(w, _)| *w as f64).sum();
    let sy: f64 = points.iter().map(|(_, p)| *p).sum();
    let sxx: f64 = points.iter().map(|(w, _)| (*w as f64) * (*w as f64)).sum();
    let sxy: f64 = points.iter().map(|(w, p)| (*w as f64) * p).sum();
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (a, (sy - a * sx) / n)
}

/// Median of a small sample. Every headline timing is reported as a median with its range, because a
/// single sample of this quantity varies ~10% run to run.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------------------------------
// Q1 — cost per window, its scaling, the loader overhead, and the release-guard ablation
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn window_sweep_cost() -> Result<()> {
    let q4 = env_path_req("SC15791_Q4");
    let repeats = env_usize("SC15791_REPEATS", 3);
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);

    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;
    println!(
        "[sc-15791] TIER {}: {:.2} GiB on disk, {} tensors, {} blocks, {:.2} GiB of block weights, \
         {} packed triples + {} dense per block",
        q4.display(),
        tier.file_bytes as f64 / GIB,
        tier.n_tensors,
        tier.n_blocks(),
        tier.all_block_bytes() as f64 / GIB,
        tier.packed[0].len(),
        tier.dense[0].len(),
    );

    // ── Fully-resident control, doubling as the page-cache warm-up. Everything below is a WARM
    // page-cache measurement, exactly as SC-15744's 0.309 s/step was; see the caveat in the closeout.
    quiesce_and_reset(&dev, &pool)?;
    let t_res = Instant::now();
    let (resident_secs, resident_used, resident_reserved) = {
        let view = tier.open_view(&dev)?;
        let b = materialize(&tier, &view, 0..tier.n_blocks())?;
        dev.synchronize()?;
        let secs = t_res.elapsed().as_secs_f64();
        let r = (secs, pool.used_high(), pool.reserved_high());
        drop(b);
        drop(view);
        r
    };
    println!(
        "\n[sc-15791] CONTROL — all {} blocks resident: {resident_secs:.3} s to load, {:.1} MiB live \
         peak, {:.1} MiB RESERVED peak (the gate's unit)",
        tier.n_blocks(),
        resident_used as f64 / MIB,
        resident_reserved as f64 / MIB,
    );

    // ── The sweep. The per-window `dev.synchronize()` is INSIDE the timed region because it is part
    // of the candidate design (it is the candle analogue of MLX's materialize guard); its cost is
    // isolated by the ablation arm further down rather than left confounded.
    println!(
        "\n[sc-15791] Q1 — one denoise step (all {} blocks re-materialized), median of {repeats}",
        tier.n_blocks()
    );
    println!(
        "  {:>6} {:>8} {:>10} {:>10} {:>9} {:>10} {:>10} {:>11}",
        "window",
        "windows",
        "step med s",
        "range s",
        "ms/block",
        "live MiB",
        "resvd MiB",
        "on-disk MiB",
    );
    let mut results: Vec<(usize, f64, f64, f64)> = Vec::new();
    for window in [1usize, 2, 4, 8, 15, 30] {
        let mut samples = Vec::new();
        let (mut peak_used, mut peak_reserved) = (0u64, 0u64);
        for _ in 0..repeats {
            quiesce_and_reset(&dev, &pool)?;
            let base = pool.used();
            let t0 = Instant::now();
            for range in windows(tier.n_blocks(), window) {
                let view = tier.open_view(&dev)?;
                let blocks = materialize(&tier, &view, range)?;
                dev.synchronize()?;
                drop(blocks);
                drop(view);
            }
            dev.synchronize()?;
            samples.push(t0.elapsed().as_secs_f64());
            peak_used = peak_used.max(pool.used_high().saturating_sub(base));
            peak_reserved = peak_reserved.max(pool.reserved_high());
            pool.trim();
        }
        let med = median(samples.clone());
        let (lo, hi) = (
            samples.iter().cloned().fold(f64::MAX, f64::min),
            samples.iter().cloned().fold(0.0, f64::max),
        );
        let on_disk: usize = (0..window.min(tier.n_blocks()))
            .map(|b| tier.bytes[b])
            .sum();
        println!(
            "  {window:>6} {:>8} {med:>10.3} {:>10} {:>9.1} {:>10.1} {:>10.1} {:>11.1}",
            tier.n_blocks().div_ceil(window),
            format!("{lo:.2}-{hi:.2}"),
            med * 1000.0 / tier.n_blocks() as f64,
            peak_used as f64 / MIB,
            peak_reserved as f64 / MIB,
            on_disk as f64 / MIB,
        );
        results.push((
            window,
            med,
            peak_used as f64 / MIB,
            peak_reserved as f64 / MIB,
        ));
    }

    // The bound must actually be a bound, and it must scale — a flat or non-monotone peak column
    // would mean the window is not controlling residency at all.
    for pair in results.windows(2) {
        assert!(
            pair[1].2 > pair[0].2,
            "peak must grow with window size, else the window bounds nothing: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }
    let w1 = results[0];
    assert!(
        w1.2 < resident_used as f64 / MIB / 10.0,
        "window 1 peak {:.1} MiB is not materially below the resident control {:.1} MiB",
        w1.2,
        resident_used as f64 / MIB
    );

    let (slope, intercept) = linear_fit(
        &results
            .iter()
            .map(|(w, _, p, _)| (*w, *p))
            .collect::<Vec<_>>(),
    );
    println!(
        "  peak(window) ≈ {slope:.1}·w + {intercept:.1} MiB. The {intercept:.1} MiB intercept is \
         fixed allocator/loader overhead, not block weight. At the {slope:.1} MiB/block slope alone \
         the reduction would be {:.1}x.",
        resident_used as f64 / MIB / slope
    );

    // ── Where the time goes. The current production loader vs its raw-mmap host-repack twin, plus
    // the CPU-only leg to separate read+repack from all transfer.
    // All three legs run on the SAME block (0) and are interleaved, so neither page-cache state nor
    // block-to-block variation is charged to one leg. More repeats than the sweep, because the
    // quantities being differenced here are close to the noise floor.
    let b_repeats = (repeats * 3).max(7);
    println!(
        "\n[sc-15791] Q1b — loader overhead, one 1-block window (median of {b_repeats}, block 0)"
    );
    let raw = tier.open_raw()?;
    let cpu = Device::Cpu;
    let (mut prod, mut corrected, mut host_only) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..b_repeats {
        let view = tier.open_view(&cpu)?;
        let t = Instant::now();
        drop(materialize(&tier, &view, 0..1)?);
        host_only.push(t.elapsed().as_secs_f64() * 1000.0);
        drop(view);

        let view = tier.open_view(&dev)?;
        let t = Instant::now();
        let blk = materialize(&tier, &view, 0..1)?;
        dev.synchronize()?;
        prod.push(t.elapsed().as_secs_f64() * 1000.0);
        drop(blk);
        drop(view);
        quiesce_and_reset(&dev, &pool)?;

        let t = Instant::now();
        let blk = materialize_host_repack(&tier, &raw, &dev, 0..1)?;
        dev.synchronize()?;
        corrected.push(t.elapsed().as_secs_f64() * 1000.0);
        drop(blk);
        quiesce_and_reset(&dev, &pool)?;
    }
    let spread = |v: &[f64]| {
        let lo = v.iter().cloned().fold(f64::MAX, f64::min);
        let hi = v.iter().cloned().fold(0.0, f64::max);
        hi - lo
    };
    let (h, p, c) = (
        median(host_only.clone()),
        median(prod.clone()),
        median(corrected.clone()),
    );
    // The noise floor for a DIFFERENCE of two of these is at least the larger spread.
    let noise = spread(&host_only)
        .max(spread(&corrected))
        .max(spread(&prod));
    let block_mib = tier.bytes[0] as f64 / MIB;
    println!(
        "  host read+repack only (CPU target)  {h:8.1} ms  (spread {:.1})\n  \
         raw-mmap host-repack twin            {c:8.1} ms  (spread {:.1})  ⇒ one step = {:.2} s\n  \
         production loader (`quant::lin`)     {p:8.1} ms  (spread {:.1})  ⇒ one step = {:.2} s",
        spread(&host_only),
        spread(&corrected),
        c * tier.n_blocks() as f64 / 1000.0,
        spread(&prod),
        p * tier.n_blocks() as f64 / 1000.0,
    );
    println!(
        "  ⇒ uploading the repacked {block_mib:.1} MiB block costs {:.1} ms, which is {} the \
         ±{noise:.1} ms noise floor.\n  \
         ⇒ the production VarBuilder seam differs from the raw-mmap twin by {:.1} ms/block = {:.2} \
         s/step; this is loader/framework overhead, not the historical device round trip.\n  \
         ⇒ **the host read+repack is {:.0}% of the raw-mmap path.** The story's premise that \"PCIe \
         bandwidth is the analogous bound here\" is NOT supported on this backend: transfer is a \
         rounding error next to the format conversion, and removing only the round trip still leaves \
         {:.2} s/step.",
        c - h,
        if (c - h).abs() < noise { "BELOW (i.e. unresolvable against)" } else { "above" },
        p - c,
        (p - c) * tier.n_blocks() as f64 / 1000.0,
        h / c * 100.0,
        c * tier.n_blocks() as f64 / 1000.0,
    );
    println!(
        "  CAUTION: none of these is a clean PCIe figure. Each device-target leg also contains a \
         device `to_dtype`, the padded QTensor `alloc_zeros`, the per-block synchronize, and — because \
         the pool release threshold is 0 — pages returned to the driver at that synchronize and \
         re-acquired next block."
    );

    // ── The release-guard ablation. THE evidence for whether `release`/the per-window synchronize is
    // load-bearing. MLX's twin (`block_window_without_materialize_frees_nothing`) exists because a
    // rung-4 path can look correct while saving nothing; this is the candle equivalent.
    println!("\n[sc-15791] Q1c — ABLATION: the same window-1 sweep with NO per-window synchronize");
    quiesce_and_reset(&dev, &pool)?;
    let base = pool.used();
    let t0 = Instant::now();
    for range in windows(tier.n_blocks(), 1) {
        let view = tier.open_view(&dev)?;
        let blocks = materialize(&tier, &view, range)?;
        drop(blocks); // no synchronize: frees are enqueued, allocations keep coming
        drop(view);
    }
    let unguarded_secs = t0.elapsed().as_secs_f64();
    let unguarded_used = pool.used_high().saturating_sub(base);
    let unguarded_reserved = pool.reserved_high();
    dev.synchronize()?;
    println!(
        "  guarded (per-window sync): {:.3} s, {:.1} MiB live peak, {:.1} MiB reserved peak\n  \
         UNGUARDED (no sync):        {unguarded_secs:.3} s, {:.1} MiB live peak, {:.1} MiB reserved \
         peak\n  ⇒ the guard is {} for the BOUND ({:.2}x peak without it)",
        w1.1,
        w1.2,
        w1.3,
        unguarded_used as f64 / MIB,
        unguarded_reserved as f64 / MIB,
        if unguarded_used as f64 / MIB > w1.2 * 1.5 {
            "LOAD-BEARING"
        } else {
            "NOT load-bearing"
        },
        unguarded_used as f64 / MIB / w1.2,
    );
    pool.trim();

    println!(
        "\n[sc-15791] Q1 SUMMARY (window, median step s, live peak MiB, reserved peak MiB):\n  {:?}",
        results
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Q5 — overlap
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn overlap_prefetch() -> Result<()> {
    let q4 = env_path_req("SC15791_Q4");
    let tokens = env_usize("SC15791_TOKENS", 1024);
    let repeats = env_usize("SC15791_REPEATS", 3);
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = std::sync::Arc::new(Tier::open(&q4, MLX_GROUP_SIZE)?);

    // Warm.
    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..tier.n_blocks())?);
        dev.synchronize()?;
    }
    pool.trim();

    let window = 1usize;
    let mut seq_samples = Vec::new();
    let mut pre_samples = Vec::new();
    let (mut seq_peak, mut pre_peak) = (0u64, 0u64);

    for _ in 0..repeats {
        // Sequential.
        quiesce_and_reset(&dev, &pool)?;
        let base = pool.used();
        let t = Instant::now();
        let mut acc = Vec::new();
        for range in windows(tier.n_blocks(), window) {
            let view = tier.open_view(&dev)?;
            let blocks = materialize(&tier, &view, range)?;
            acc.push(compute(&blocks, tokens, &dev)?);
            drop(blocks);
            drop(view);
        }
        dev.synchronize()?;
        seq_samples.push(t.elapsed().as_secs_f64());
        std::hint::black_box(acc.last().unwrap().to_scalar::<f32>()?);
        seq_peak = seq_peak.max(pool.used_high().saturating_sub(base));
        pool.trim();

        // Prefetched on a worker thread. The `Tier` header census is SHARED via `Arc` — re-parsing it
        // per prefetch would charge this arm for work the sequential arm never does.
        quiesce_and_reset(&dev, &pool)?;
        let base = pool.used();
        let t = Instant::now();
        type Prefetch = Result<(Vec<Block>, VarBuilder<'static>)>;
        let mut pending: Option<std::thread::JoinHandle<Prefetch>> = None;
        let ranges: Vec<Range<usize>> = windows(tier.n_blocks(), window).collect();
        let mut acc = Vec::new();
        for (i, range) in ranges.iter().enumerate() {
            let (blocks, view) = match pending.take() {
                Some(h) => h.join().expect("prefetch thread")?,
                None => {
                    let view = tier.open_view(&dev)?;
                    let blocks = materialize(&tier, &view, range.clone())?;
                    (blocks, view)
                }
            };
            if let Some(next) = ranges.get(i + 1).cloned() {
                let dev2 = dev.clone();
                let tier2 = std::sync::Arc::clone(&tier);
                pending = Some(std::thread::spawn(move || -> Prefetch {
                    let view = tier2.open_view(&dev2)?;
                    let blocks = materialize(&tier2, &view, next)?;
                    Ok((blocks, view))
                }));
            }
            acc.push(compute(&blocks, tokens, &dev)?);
            drop(blocks);
            drop(view);
        }
        dev.synchronize()?;
        pre_samples.push(t.elapsed().as_secs_f64());
        std::hint::black_box(acc.last().unwrap().to_scalar::<f32>()?);
        // The cost side of the trade: prefetching holds TWO windows plus two dequant transients.
        pre_peak = pre_peak.max(pool.used_high().saturating_sub(base));
        pool.trim();
    }

    let (s, p) = (median(seq_samples.clone()), median(pre_samples.clone()));
    let spread = |v: &[f64]| {
        let (lo, hi) = (
            v.iter().cloned().fold(f64::MAX, f64::min),
            v.iter().cloned().fold(0.0, f64::max),
        );
        (hi - lo) / lo * 100.0
    };
    println!(
        "\n[sc-15791] Q5 — overlap, tokens={tokens} window={window}, median of {repeats}\n  \
         sequential  {s:.3} s (spread {:.1}%), live peak {:.1} MiB\n  \
         prefetched  {p:.3} s (spread {:.1}%), live peak {:.1} MiB\n  \
         ⇒ saving {:.1}%, against a run-to-run spread of {:.1}%",
        spread(&seq_samples),
        seq_peak as f64 / MIB,
        spread(&pre_samples),
        pre_peak as f64 / MIB,
        (s - p) / s * 100.0,
        spread(&seq_samples).max(spread(&pre_samples)),
    );
    println!(
        "  Structural answer, from `release_semantics` Q3c rather than from this delta: candle issues \
         H2D as `memcpy_htod_async` from PAGEABLE host memory on the single per-device stream, so a \
         copy submitted while compute is queued BLOCKS THE SUBMITTING THREAD behind that queue. The \
         transfer therefore cannot overlap the compute, and the prefetch thread cannot get ahead. \
         Real overlap needs pinned host staging plus a dedicated copy stream — neither of which \
         candle exposes today."
    );
    println!(
        "  Prefetch also costs {:.2}x the live peak (two windows plus two dequant transients \
         co-resident), so on this backend it worsens the very quantity rung 4 exists to bound.",
        pre_peak as f64 / seq_peak as f64
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Q2 / Q3 — does the memory come back, and must `release` be non-trivial?
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn release_semantics() -> Result<()> {
    let q4 = env_path_req("SC15791_Q4");
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;

    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..2)?);
        dev.synchronize()?;
    }
    pool.trim();

    println!("\n[sc-15791] Q2 — does dropping a window return the memory?");
    let threshold = pool.release_threshold();
    let (free_before, _) = pool::mem_info();
    let (used_before, reserved_before) = (pool.used(), pool.reserved());

    let view = tier.open_view(&dev)?;
    let blocks = materialize(&tier, &view, 0..1)?;
    let used_live_nosync = pool.used();
    dev.synchronize()?;
    let (used_live, reserved_live) = (pool.used(), pool.reserved());
    let (free_live, _) = pool::mem_info();

    // Drop with NO synchronize.
    drop(blocks);
    drop(view);
    let (used_drop_nosync, reserved_drop_nosync) = (pool.used(), pool.reserved());
    let (free_drop_nosync, _) = pool::mem_info();

    dev.synchronize()?;
    let (used_sync, reserved_sync) = (pool.used(), pool.reserved());
    let (free_sync, _) = pool::mem_info();

    // Only now the trim, with `reserved_sync` already captured so its effect is attributable.
    pool.trim();
    let (free_trim, _) = pool::mem_info();
    let reserved_trim = pool.reserved();

    let block_mib = tier.bytes[0] as f64 / MIB;
    println!(
        "  one block = {block_mib:.1} MiB on disk | pool release threshold = {threshold} bytes\n  \
         pool used:     before {:.1} | live(no sync) {:.1} | live {:.1} | drop(no sync) {:.1} | \
         drop+sync {:.1}  MiB\n  \
         pool reserved: before {:.1} | live {:.1} | drop(no sync) {:.1} | drop+sync {:.1} | trim \
         {:.1}  MiB\n  \
         DRIVER free:   before {:.3} | live {:.3} | drop(no sync) {:.3} | drop+sync {:.3} | trim \
         {:.3}  GiB",
        used_before as f64 / MIB,
        used_live_nosync as f64 / MIB,
        used_live as f64 / MIB,
        used_drop_nosync as f64 / MIB,
        used_sync as f64 / MIB,
        reserved_before as f64 / MIB,
        reserved_live as f64 / MIB,
        reserved_drop_nosync as f64 / MIB,
        reserved_sync as f64 / MIB,
        reserved_trim as f64 / MIB,
        free_before as f64 / GIB,
        free_live as f64 / GIB,
        free_drop_nosync as f64 / GIB,
        free_sync as f64 / GIB,
        free_trim as f64 / GIB,
    );

    let live_delta = used_live.saturating_sub(used_before) as f64 / MIB;
    assert!(
        live_delta > block_mib * 0.5,
        "materializing a block must show as live pool bytes (saw {live_delta:.1} MiB for a \
         {block_mib:.1} MiB block) — else the probe is not watching candle's allocator"
    );

    // ── Q3a. THE answer, and it must be read off the DRIVER, not the pool counter. A VRAM gate reads
    // driver-visible free (`gpu.rs::nvidia_smi_min_free_gib`, `testkit::VramProbe`), so "the pool
    // says 0" is not "the card got its memory back".
    println!("\n[sc-15791] Q3a — is a stream synchronize required for the memory to come back?");
    let pool_freed_nosync = used_live.saturating_sub(used_drop_nosync) as f64 / MIB;
    let driver_recovered_nosync = free_drop_nosync.saturating_sub(free_live) as f64 / MIB;
    let driver_recovered_sync = free_sync.saturating_sub(free_live) as f64 / MIB;
    let driver_held_live = free_before.saturating_sub(free_live) as f64 / MIB;
    println!(
        "  the drop took {driver_held_live:.1} MiB of driver-visible VRAM while live.\n  \
         pool USED freed by the bare drop:          {pool_freed_nosync:.1} MiB\n  \
         DRIVER free recovered by the bare drop:    {driver_recovered_nosync:.1} MiB\n  \
         DRIVER free recovered after a synchronize: {driver_recovered_sync:.1} MiB\n  \
         additionally recovered by an explicit trim: {:.1} MiB",
        free_trim.saturating_sub(free_sync) as f64 / MIB,
    );
    let sync_required = driver_recovered_sync > driver_recovered_nosync + driver_held_live * 0.25;
    println!(
        "  ⇒ driver-visible VRAM {} a synchronize to come back.\n  \
         Mechanism: `cuMemFreeAsync` decrements the pool's USED counter at ENQUEUE, so the pool-level \
         view frees on the bare drop and the NEXT WINDOW CAN ALLOCATE IMMEDIATELY. The physical pages \
         return to the driver only at a synchronization, because the pool's release threshold is \
         {threshold} (the driver default — neither candle nor cudarc sets it), i.e. \"release \
         everything on every synchronize\".",
        if sync_required { "REQUIRES" } else { "does not require" },
    );
    println!(
        "  What that does and does NOT imply for `BlockWindowBackend::release`:\n    \
         · PER-WINDOW it can be a NO-OP. The bound is held by pool reuse, not by returning pages to \
           the driver — and `window_sweep_cost`'s Q1c ablation confirms it directly: with the \
           per-window synchronize removed, the live AND reserved peaks both stay at one window.\n    \
         · AT TEARDOWN something must synchronize, or the request's last window stays charged against \
           driver-visible free — which is what a co-resident component, the next job's admission gate \
           (`gpu.rs::nvidia_smi_min_free_gib`), and `testkit::VramProbe` all read. That is a \
           request-scoped obligation, not a per-window one."
    );
    assert!(
        sync_required,
        "expected driver-visible VRAM to require a synchronize; if this backend ever starts \
         returning pages on the bare drop, the SC-15792 release contract must be revisited rather \
         than silently inheriting a stale answer"
    );
    // `trim` is a no-op at threshold 0 and must not be reported as the thing that worked.
    assert_eq!(
        threshold, 0,
        "the pool release threshold is no longer 0 — `release` now additionally needs a trim, and \
         the Q3a conclusion above inverts"
    );

    // ── Q3b/Q3c. Both arms below reuse the freed pages from the SAME per-device stream that owns the
    // outstanding compute, and CUDA's stream-ordered allocator guarantees ordering there a priori. So
    // they are recorded as CONSISTENCY CHECKS, not as discriminating experiments: no setting of the
    // race knobs could make a same-stream reuse corrupt a same-stream matmul. Read them that way.
    println!(
        "\n[sc-15791] Q3b/Q3c — same-stream reuse under outstanding compute (CONSISTENCY CHECK, not \
         a discriminating test: stream order guarantees this case)"
    );
    let tokens = env_usize("SC15791_RACE_TOKENS", 4096);
    let repeats = env_usize("SC15791_RACE_REPEATS", 64);
    let chain = |blocks: &[Block], dev: &Device| -> Result<Tensor> {
        let mut acc = Tensor::zeros((), DType::F32, dev)?;
        for _ in 0..repeats {
            acc = (acc + compute(blocks, tokens, dev)?)?;
        }
        Ok(acc)
    };
    let reference = {
        let view = tier.open_view(&dev)?;
        let blocks = materialize(&tier, &view, 0..1)?;
        let y = chain(&blocks, &dev)?;
        dev.synchronize()?;
        let v = y.to_scalar::<f32>()?;
        drop(blocks);
        drop(view);
        v
    };
    // (b) reuse by a `ones` memset — a live weight buffer overwritten with 1.0f32 would move the
    // result enormously if the free were not ordered behind the kernels.
    let racy_memset = {
        let view = tier.open_view(&dev)?;
        let blocks = materialize(&tier, &view, 0..1)?;
        let t = Instant::now();
        let y = chain(&blocks, &dev)?;
        let launch = t.elapsed().as_secs_f64() * 1000.0;
        drop(blocks);
        drop(view);
        let claim_mib = env_usize("SC15791_RACE_CLAIM_MIB", 256);
        let chunk = 8 * 1024 * 1024 / 4;
        let claims: Vec<Tensor> = (0..(claim_mib / 8).max(1))
            .map(|_| Tensor::ones(chunk, DType::F32, &dev))
            .collect::<Result<_>>()?;
        let reuse = t.elapsed().as_secs_f64() * 1000.0;
        let v = y.to_scalar::<f32>()?;
        let total = t.elapsed().as_secs_f64() * 1000.0;
        drop(claims);
        println!(
            "  (b) memset reuse: enqueue {launch:.1} ms | pages re-claimed {reuse:.1} ms | first \
             sync {total:.1} ms ⇒ {:.0} ms of work was still outstanding at the reuse",
            total - reuse
        );
        v
    };
    // (c) reuse by a real H2D weight materialization — the sc-12195 shape.
    let (racy_upload, stall, idle_ms) = {
        let idle_ms = {
            let view = tier.open_view(&dev)?;
            drop(materialize(&tier, &view, 1..2)?);
            dev.synchronize()?;
            let view = tier.open_view(&dev)?;
            let t = Instant::now();
            let b = materialize(&tier, &view, 1..2)?;
            dev.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            drop(b);
            ms
        };
        // NOTE: no `pool.trim()` between the baseline and the arm below — trimming would force the
        // arm to re-acquire pages from the driver that the baseline already had, inflating the ratio.
        let view = tier.open_view(&dev)?;
        let blocks = materialize(&tier, &view, 0..1)?;
        let t = Instant::now();
        let y = chain(&blocks, &dev)?;
        let launch = t.elapsed().as_secs_f64() * 1000.0;
        drop(blocks);
        drop(view);
        let view2 = tier.open_view(&dev)?;
        let next = materialize(&tier, &view2, 1..2)?;
        let reuse = t.elapsed().as_secs_f64() * 1000.0;
        let v = y.to_scalar::<f32>()?;
        drop(next);
        drop(view2);
        (v, reuse - launch, idle_ms)
    };
    println!(
        "  (c) H2D reuse: the upload took {stall:.1} ms against {idle_ms:.1} ms on an idle stream \
         ({:.1}x). That stall is the finding: candle submits H2D on the SAME stream as the compute, \
         from pageable host memory, so the submitting thread blocks until the queue drains. It is \
         also why (c) cannot discriminate — the reuse physically cannot precede the kernels.",
        stall / idle_ms,
    );
    println!(
        "  reference {reference:e} | memset-reuse {racy_memset:e} | H2D-reuse {racy_upload:e} ⇒ {}",
        if reference == racy_memset && reference == racy_upload {
            "all IDENTICAL (as stream ordering requires)"
        } else {
            "DIVERGED — stream-ordered freeing is NOT holding, which would be a candle bug"
        }
    );
    assert_eq!(
        reference, racy_memset,
        "same-stream memset reuse corrupted an in-flight result"
    );
    assert_eq!(
        reference, racy_upload,
        "same-stream H2D reuse corrupted an in-flight result"
    );

    println!(
        "\n[sc-15791] Q3 — UNRESOLVED, and deliberately not claimed either way:\n  \
         sc-12195 (`residency.rs:24-32`) records that dropping a phase and letting the next loader \
         reuse the freed pool DETERMINISTICALLY corrupted FLUX.2-dev Q4 pixels until a \
         `Device::synchronize()` was added, and that seam still performs it as \"the single point of \
         enforcement\". Nothing above reproduces that: every arm here stays on one device and one \
         stream, where ordering is guaranteed. So this spike does NOT explain sc-12195 and must not \
         be read as licence to remove that sync. Whatever crossed streams or contexts there is \
         untested here — and note `overlap_prefetch` materializes from a WORKER THREAD, which is the \
         closest thing to the untested configuration and should be treated as unproven until \
         SC-15792 covers it."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Q4 — packed-quant, per block
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn packed_quant_per_block() -> Result<()> {
    let q4 = env_path_req("SC15791_Q4");
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;
    let st = tier.open_raw()?;

    println!("\n[sc-15791] Q4 — does the packed triple materialize per block, faithfully?");
    println!(
        "  block 0: {} packed triples + {} dense tensors, {:.1} MiB on disk",
        tier.packed[0].len(),
        tier.dense[0].len(),
        tier.bytes[0] as f64 / MIB
    );

    // Every packed base is three INDEPENDENT file entries — the property that makes per-block
    // materialization per-tensor rather than a slice out of a larger packed buffer. This was
    // SC-15744's identified risk on MLX; confirmed absent here against the same file.
    let names: std::collections::HashSet<String> =
        st.tensors().into_iter().map(|(k, _)| k).collect();
    for base in &tier.packed[0] {
        for suffix in [".weight", ".scales", ".biases"] {
            assert!(
                names.contains(&format!("{base}{suffix}")),
                "{base}{suffix} must be its own file entry for per-block materialization to work"
            );
        }
    }
    println!(
        "  all {} triples are three independent safetensors entries ⇒ per-block materialization is \
         per-tensor, not a sub-slice",
        tier.packed[0].len()
    );

    // Fidelity. Compared against the dense MLX affine grid — the ground truth both the windowed and
    // the resident path claim to represent. (`mlx_repack_real_weights.rs` already pins the resident
    // loader against the same grid, so agreeing with it here is equivalent to agreeing with the
    // resident path, without this spike having to build one.)
    let view = tier.open_view(&dev)?;
    let blocks = materialize(&tier, &view, 0..1)?;
    let mut worst = 0f32;
    for (base, ql, in_dim) in &blocks[0].lins {
        assert!(
            ql.is_quantized(),
            "{base}: `.scales` present but the loader took the dense path — the window would stream \
             dequantized weights and the whole rung-4 saving would be fictional"
        );
        let wq = st.load(&format!("{base}.weight"), &Device::Cpu)?;
        let scales = st.load(&format!("{base}.scales"), &Device::Cpu)?;
        let biases = st.load(&format!("{base}.biases"), &Device::Cpu)?;
        let grid = dequant_mlx_q4_reference(&wq, &scales, &biases)?.to_device(&dev)?;
        let dense = QLinear::Dense(candle_gen::quant::DenseLinear::Linear(
            candle_gen::candle_nn::Linear::new(grid, None),
        ));
        let x = Tensor::randn(0f32, 1f32, (2, *in_dim), &dev)?;
        let d = (ql.forward(&x)?.sub(&dense.forward(&x)?))?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        worst = worst.max(d);
    }
    println!(
        "  windowed packed forward vs the dense MLX grid, all {} projections of block 0: max \
         deviation {worst}",
        blocks[0].lins.len()
    );
    assert_eq!(
        worst, 0.0,
        "a windowed block must be bit-exact against the MLX grid"
    );
    drop(blocks);
    drop(view);
    pool.trim();

    // Resident cost of one packed block, in BOTH units.
    quiesce_and_reset(&dev, &pool)?;
    let before = pool.used();
    let view2 = tier.open_view(&dev)?;
    let b2 = materialize(&tier, &view2, 0..1)?;
    dev.synchronize()?;
    let (live, reserved) = (pool.used().saturating_sub(before), pool.reserved_high());
    println!(
        "  one packed block resident: {:.1} MiB live ({:.1} MiB of it the dense norm half), {:.1} \
         MiB RESERVED (the gate's unit — {:.0}% above live) | {:.1} MiB on disk\n  \
         The {:.1}x on-disk inflation is exactly Q4_1's 0.625 B/elem against the MLX pack's 0.5625, \
         i.e. the repack container — a permanent, structural cost of Candle's realization that MLX \
         does not pay.",
        live as f64 / MIB,
        b2[0].dense_bytes() as f64 / MIB,
        reserved as f64 / MIB,
        (reserved as f64 / live as f64 - 1.0) * 100.0,
        tier.bytes[0] as f64 / MIB,
        live as f64 / tier.bytes[0] as f64,
    );
    drop(b2);
    drop(view2);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// The q8 tier — measured, because Candle's q8 path is not Candle's q4 path
// ---------------------------------------------------------------------------------------------------

/// SC-15744 extrapolated "a q8 transformer is roughly 2x the q4 figure, so ~190 MiB/block". That is a
/// **memory** claim, and this arm exists to check it rather than to assume it — on Candle the two
/// tiers take structurally different code paths. Q4 repacks losslessly into GGML `Q4_1` (a byte
/// shuffle). Q8 has no affine GGML container, so `repack_packed_weight` (`quant/mod.rs:930-934`)
/// materializes the FULL dense f32 grid on the host and re-quantizes it to `Q8_0`, per projection,
/// per window, per step — a host transient the q4 path never pays.
#[test]
#[ignore = "optional: needs the hosted z-image q8 tier (SC15791_Q8 env)"]
fn q8_tier_cost() -> Result<()> {
    let Some(q8) = env_path_opt("SC15791_Q8") else {
        return Ok(());
    };
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q8, MLX_GROUP_SIZE)?;
    println!(
        "\n[sc-15791] Q8 TIER: {:.2} GiB on disk, {} blocks, {:.1} MiB/block on disk",
        tier.file_bytes as f64 / GIB,
        tier.n_blocks(),
        tier.bytes[0] as f64 / MIB
    );

    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..2)?);
        dev.synchronize()?;
    }
    pool.trim();

    // Windows 1 and 2, so the scaling question is answered for q8 too and not silently generalized
    // from q4.
    for window in [1usize, 2] {
        quiesce_and_reset(&dev, &pool)?;
        let base = pool.used();
        let t = Instant::now();
        for range in windows(tier.n_blocks(), window) {
            let view = tier.open_view(&dev)?;
            let blocks = materialize(&tier, &view, range)?;
            dev.synchronize()?;
            drop(blocks);
            drop(view);
        }
        dev.synchronize()?;
        let step = t.elapsed().as_secs_f64();
        println!(
            "  window {window}: step {step:.3} s ({:.1} ms/block) | live peak {:.1} MiB | RESERVED \
             peak {:.1} MiB",
            step * 1000.0 / tier.n_blocks() as f64,
            pool.used_high().saturating_sub(base) as f64 / MIB,
            pool.reserved_high() as f64 / MIB,
        );
        pool.trim();
    }
    println!(
        "  Compare against the q4 arm. SC-15744's extrapolation was about MEMORY and should be judged \
         on the peak column, not the time column — the two diverge sharply here.\n  \
         NOT MEASURED: the HOST transient. The dense f32 grid this path materializes per projection \
         is the real q8 risk (~225 MiB of host f32 for a 3840x15360 projection) and nothing above \
         watches host RSS. Treat q8 host memory as UNVERIFIED."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// SC-16096 — content-addressed device-format windows, before/after, host memory, and bound mutation
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs CUDA plus SC16096_Q4 and SC16096_Q8 real packed tiers"]
fn device_format_sidecars_before_after() -> Result<()> {
    let repeats = env_usize("SC16096_REPEATS", 3).max(3);
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);

    for (label, path, bits) in [
        ("q4", env_path_req("SC16096_Q4"), 4i32),
        ("q8", env_path_req("SC16096_Q8"), 8i32),
    ] {
        let tier = Tier::open(&path, MLX_GROUP_SIZE)?;
        let raw = tier.open_raw()?;
        let component_dir = path.parent().expect("tier file has a component directory");
        println!(
            "\n[sc-16096] {label}: {} | {:.2} GiB source mmap, {} blocks, {} packed projections/block",
            path.display(),
            tier.file_bytes as f64 / GIB,
            tier.n_blocks(),
            tier.packed[0].len()
        );

        let prepare_started = Instant::now();
        let (prepared, build_host) = host_memory::sample(|| {
            PackedWeightSidecars::prepare(
                &raw,
                component_dir,
                PackedConfig {
                    bits,
                    group_size: MLX_GROUP_SIZE as i32,
                },
                &dev,
            )
        });
        let sidecars = prepared?;
        dev.synchronize()?;
        println!(
            "  prepare once: {:.3} s | created {} reused {} | source hashed {:.1} MiB | \
             sidecars {:.1} MiB | host working-set peak Δ {:.1} MiB, private-commit peak Δ {:.1} \
             MiB (end Δ {:.1}/{:.1} MiB)",
            prepare_started.elapsed().as_secs_f64(),
            sidecars.created_count(),
            sidecars.reused_count(),
            sidecars.source_bytes_hashed() as f64 / MIB,
            sidecars.sidecar_bytes() as f64 / MIB,
            build_host
                .working_set_peak
                .saturating_sub(build_host.working_set_start) as f64
                / MIB,
            build_host
                .private_peak
                .saturating_sub(build_host.private_start) as f64
                / MIB,
            build_host
                .working_set_end
                .saturating_sub(build_host.working_set_start) as f64
                / MIB,
            build_host
                .private_end
                .saturating_sub(build_host.private_start) as f64
                / MIB,
        );

        // Every projection in one real block must preserve the exact GGML bytes produced by the old
        // CUDA-target conversion. Q8 is especially important: its target-device quantizer, not a CPU
        // approximation, defines the pre-change bytes.
        let mut checked_bytes = 0usize;
        let mut checked_outputs = 0usize;
        for base in &tier.packed[0] {
            let (_, in_dim) = tier.dims[base];
            let wq = raw.load(&format!("{base}.weight"), &dev)?;
            let scales = raw
                .load(&format!("{base}.scales"), &dev)?
                .to_dtype(DType::F32)?;
            let biases = raw
                .load(&format!("{base}.biases"), &dev)?
                .to_dtype(DType::F32)?;
            let old = repack_packed_weight(&wq, &scales, &biases, MLX_GROUP_SIZE, &dev)?;
            let new = sidecars.load(base, &dev)?;
            let old_bytes = old.data()?.into_owned();
            let new_bytes = new.data()?.into_owned();
            assert_eq!(
                old_bytes, new_bytes,
                "{label} {base}: sidecar bytes differ from the pre-change CUDA path"
            );
            checked_bytes += old_bytes.len();

            // Drive the exact shared QLinear compute seam used by Krea on both weights. Unchanged
            // bytes alone already imply an unchanged forward, but this makes that implication an
            // executable real-CUDA output assertion for every projection in the sampled block.
            let input = Tensor::ones((1, in_dim), DType::F32, &dev)?;
            let old_output = QLinear::from_qtensor_dequant(Arc::new(old), None).forward(&input)?;
            let new_output = QLinear::from_qtensor_dequant(Arc::new(new), None).forward(&input)?;
            assert_eq!(
                old_output.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
                new_output.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
                "{label} {base}: sidecar-backed QLinear output differs from pre-change"
            );
            checked_outputs += 1;
        }
        println!(
            "  parity: {} real projections / {:.1} MiB and {} real QLinear CUDA forwards are \
             BIT-EXACT to pre-change",
            tier.packed[0].len(),
            checked_bytes as f64 / MIB,
            checked_outputs,
        );

        let measure =
            |after: bool, range: Range<usize>| -> Result<(f64, u64, u64, host_memory::Peak)> {
                quiesce_and_reset(&dev, &pool)?;
                let base = pool.used();
                let started = Instant::now();
                let (materialized, host) = host_memory::sample(|| {
                    if after {
                        materialize_sidecars(&tier, &sidecars, &raw, &dev, range)
                    } else {
                        materialize_prechange(&tier, &raw, &dev, range)
                    }
                });
                let blocks = materialized?;
                dev.synchronize()?;
                let elapsed = started.elapsed().as_secs_f64();
                let used = pool.used_high().saturating_sub(base);
                let reserved = pool.reserved_high();
                drop(blocks);
                dev.synchronize()?;
                pool.trim();
                Ok((elapsed, used, reserved, host))
            };

        for (name, after) in [("before", false), ("after", true)] {
            let mut times = Vec::with_capacity(repeats);
            let mut vram_peak = 0u64;
            let mut host_ws_peak = 0u64;
            let mut host_private_peak = 0u64;
            let mut host_ws_end = 0u64;
            let mut host_private_end = 0u64;
            for _ in 0..repeats {
                let (secs, used, _, host) = measure(after, 0..1)?;
                times.push(secs);
                vram_peak = vram_peak.max(used);
                host_ws_peak =
                    host_ws_peak.max(host.working_set_peak.saturating_sub(host.working_set_start));
                host_private_peak =
                    host_private_peak.max(host.private_peak.saturating_sub(host.private_start));
                host_ws_end =
                    host_ws_end.max(host.working_set_end.saturating_sub(host.working_set_start));
                host_private_end =
                    host_private_end.max(host.private_end.saturating_sub(host.private_start));
            }
            let med = median(times.clone());
            let lo = times.iter().copied().fold(f64::MAX, f64::min);
            let hi = times.iter().copied().fold(0.0, f64::max);
            println!(
                "  {name:>6} window-1: median {med:.4} s/block, spread {lo:.4}-{hi:.4} s | \
                 VRAM live peak {:.1} MiB | host WS/private peak Δ {:.1}/{:.1} MiB, end Δ \
                 {:.1}/{:.1} MiB",
                vram_peak as f64 / MIB,
                host_ws_peak as f64 / MIB,
                host_private_peak as f64 / MIB,
                host_ws_end as f64 / MIB,
                host_private_end as f64 / MIB,
            );
        }

        // Peak-by-window remeasurement. Mutating only the window width must raise the live CUDA
        // peak; otherwise the purported bound is decorative rather than load-bearing.
        let mut bound = Vec::new();
        for window in [1usize, 2, 4] {
            let width = window.min(tier.n_blocks());
            let (secs, used, reserved, host) = measure(true, 0..width)?;
            let mapped_peak = tier.packed[0]
                .iter()
                .filter_map(|base| sidecars.path_for(base))
                .filter_map(|path| std::fs::metadata(path).ok())
                .map(|metadata| metadata.len())
                .max()
                .unwrap_or(0);
            println!(
                "  bound mutation window {width}: {secs:.4} s | live {:.1} MiB, reserved {:.1} MiB | \
                 host WS/private peak Δ {:.1}/{:.1} MiB | at most one {:.1} MiB sidecar mapping \
                 held (file-backed/reclaimable; 0 retained after transfer)",
                used as f64 / MIB,
                reserved as f64 / MIB,
                host.working_set_peak.saturating_sub(host.working_set_start) as f64 / MIB,
                host.private_peak.saturating_sub(host.private_start) as f64 / MIB,
                mapped_peak as f64 / MIB,
            );
            bound.push((width, used));
        }
        for pair in bound.windows(2) {
            assert!(
                pair[1].1 > pair[0].1,
                "{label}: mutating window {} -> {} did not increase the live CUDA peak ({} -> {})",
                pair[0].0,
                pair[1].0,
                pair[0].1,
                pair[1].1
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// The small-card disclosure
// ---------------------------------------------------------------------------------------------------

/// The constrained-budget arm. This host is a ~96 GiB RTX PRO 6000, so an unconstrained number says
/// nothing about the 8 GB card rung 4 exists for.
///
/// **It must be a DISCRIMINATING constraint.** Ballooning to 8 GiB free and then measuring a 131 MiB
/// peak proves only that 131 MiB fits in 8 GiB — 58x headroom, no possible spill, a vacuous pass. So
/// this squeezes to a budget where the RESIDENT path cannot fit and the WINDOWED path can, and
/// reports both. That is the only arrangement in which "the bound is real" is evidence rather than
/// arithmetic.
///
/// **Read the caveat in the output.** On Windows/WDDM this host SPILLS to shared system memory rather
/// than hard-OOMing, so "it completed" is never proof of fit; the signals are an unchanged live peak
/// and an unchanged wall time, and a wall-time inflation IS the spill.
#[test]
#[ignore = "optional: needs SC15791_Q4 and SC15791_TARGET_FREE_GIB"]
fn constrained_budget_sweep() -> Result<()> {
    let q4 = env_path_req("SC15791_Q4");
    let target_free = env_usize("SC15791_TARGET_FREE_GIB", 0);
    if target_free == 0 {
        println!("[sc-15791] SKIP: SC15791_TARGET_FREE_GIB not set");
        return Ok(());
    }
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;

    // window 30 == one all-covering window == the resident path, through the same code, so the
    // comparison is not against a second implementation.
    // `resvd_base` is the reserved floor to subtract: 0 unconstrained, the balloon's own reservation
    // once it is held. RESERVED_HIGH is absolute and snaps back to whatever the pool currently holds
    // when the watermark is reset, so without this the constrained rows would just report the balloon.
    let run = |label: &str, window: usize, resvd_base: u64| -> Result<(f64, f64, f64)> {
        quiesce_and_reset(&dev, &pool)?;
        let base = pool.used();
        let t0 = Instant::now();
        for range in windows(tier.n_blocks(), window) {
            let view = tier.open_view(&dev)?;
            let blocks = materialize(&tier, &view, range)?;
            dev.synchronize()?;
            drop(blocks);
            drop(view);
        }
        dev.synchronize()?;
        let secs = t0.elapsed().as_secs_f64();
        let live = pool.used_high().saturating_sub(base) as f64 / MIB;
        let resvd = pool.reserved_high().saturating_sub(resvd_base) as f64 / MIB;
        println!(
            "  [{label}] window {window}: {secs:.3} s | live {live:.1} MiB | resvd {resvd:.1} MiB"
        );
        Ok((secs, live, resvd))
    };

    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..tier.n_blocks())?);
        dev.synchronize()?;
    }
    pool.trim();
    println!("\n[sc-15791] BASELINE (unconstrained)");
    let base_w1 = run("free", 1, 0)?;
    let base_w30 = run("free", 30, 0)?;

    // Balloon ADAPTIVELY: a fixed GiB count overshoots badly (86 GiB on a 93.9 GiB-free card left
    // 0.00 GiB and measured spilling, not fit).
    let mut balloon: Vec<Tensor> = Vec::new();
    let chunk_elems = 256usize * 1024 * 1024; // 1 GiB of f32
    loop {
        let (free, _) = pool::mem_info();
        if free as f64 / GIB <= target_free as f64 {
            break;
        }
        match Tensor::zeros(chunk_elems, DType::F32, &dev) {
            Ok(t) => balloon.push(t),
            Err(_) => break,
        }
        dev.synchronize()?;
    }
    let (free_under, total) = pool::mem_info();
    let achieved = free_under as f64 / GIB;
    println!(
        "\n[sc-15791] CONSTRAINED: {} GiB balloon; driver free now {achieved:.2} GiB of {:.1} GiB \
         total (target {target_free} GiB)",
        balloon.len(),
        total as f64 / GIB,
    );
    assert!(
        achieved <= target_free as f64 * 1.5,
        "the balloon under-shot ({achieved:.2} GiB free vs a {target_free} GiB target) — an \
         unconstrained run must not be reported as a constrained one"
    );
    // The constraint has to be tight enough that the RESIDENT path genuinely does not fit, or the
    // comparison proves nothing.
    println!(
        "  resident path needs {:.2} GiB reserved; {achieved:.2} GiB is available ⇒ resident {} fit",
        base_w30.2 / 1024.0,
        if base_w30.2 / 1024.0 > achieved { "must NOT" } else { "DOES" }
    );

    // The balloon's own reservation, captured after it settles, so the window figures below are the
    // window's and not the balloon's.
    dev.synchronize()?;
    let balloon_reserved = pool.reserved();
    // Window 1 must survive the squeeze — if it does not, rung 4 buys nothing and that is the story.
    let con_w1 = run("constrained", 1, balloon_reserved)?;
    // Window 30 (the resident path) is EXPECTED to struggle. An allocation failure here is the
    // discriminating result, not a harness bug, so it is caught and reported rather than propagated.
    let con_w30 = match run("constrained", 30, balloon_reserved) {
        Ok(v) => Some(v),
        Err(e) => {
            println!("  [constrained] window 30: FAILED TO ALLOCATE — {e}");
            None
        }
    };

    println!("\n[sc-15791] SMALL-CARD VERDICT");
    for (label, base, con) in [
        ("window 1 ", base_w1, Some(con_w1)),
        ("window 30", base_w30, con_w30),
    ] {
        match con {
            Some(con) => {
                let slow = con.0 / base.0;
                println!(
                    "  {label}: {:.3} s → {:.3} s ({slow:.2}x) | live {:.1} → {:.1} MiB | resvd \
                     {:.1} → {:.1} MiB{}",
                    base.0,
                    con.0,
                    base.1,
                    con.1,
                    base.2,
                    con.2,
                    if slow > 1.25 { "   ← SPILLED" } else { "" },
                );
            }
            None => println!(
                "  {label}: {:.3} s unconstrained, {:.1} MiB resvd → DID NOT FIT under the squeeze",
                base.0, base.2
            ),
        }
    }
    // The honest reading. `resident_overcommit` is the case that settles it: if the resident path's
    // reserved footprint EXCEEDED the driver-visible free VRAM and it completed anyway, unpenalized,
    // then this host absorbed an impossible working set and cannot emulate a hard VRAM ceiling at all.
    let resident_overcommit = con_w30
        .map(|w30| w30.2 / 1024.0 > achieved)
        .unwrap_or(false);
    let w30_slow = con_w30.map(|w| w.0 / base_w30.0).unwrap_or(f64::INFINITY);
    println!(
        "  ⇒ {}",
        match con_w30 {
            None => "the windowed path RAN where the resident path could not allocate — the bound is \
                     load-bearing on this host",
            Some(_) if resident_overcommit && w30_slow <= 1.25 =>
                "THIS HOST CANNOT DISCRIMINATE. The resident path's reserved footprint EXCEEDED the \
                 driver-visible free VRAM and it completed anyway with an identical peak and NO \
                 wall-time penalty — i.e. the driver silently absorbed a working set that cannot \
                 physically fit. Squeezing harder will not help: the ceiling is not enforced.",
            Some(_) if w30_slow > 1.25 =>
                "the resident path spilled (wall time inflated) while the windowed path did not — the \
                 bound is load-bearing on this host",
            Some(_) => "both paths fit this budget, so this run does not discriminate; lower \
                        SC15791_TARGET_FREE_GIB",
        }
    );
    if resident_overcommit && w30_slow <= 1.25 {
        println!(
            "  Note for the record, because it contradicts the prior WDDM-spill playbook: that \
             playbook (sc-13174) held that a spill announces itself as a 2x+ wall-time inflation. \
             Here the resident path over-committed {:.2} GiB into {achieved:.2} GiB of free VRAM and \
             ran {w30_slow:.2}x — FASTER. So wall time is NOT a reliable spill detector on this host, \
             and neither is completion.",
            con_w30.map(|w| w.2 / 1024.0).unwrap_or(0.0),
        );
    }
    println!(
        "  CAVEAT (load-bearing): this host does NOT hard-OOM — Windows/WDDM spills to shared system \
         memory. Completion is never proof of fit, and per the line above wall time is not either. \
         What this arm CAN establish is that the allocator's own accounting of the window bound is \
         unchanged under pressure; what it CANNOT establish is behaviour on a PHYSICAL small card. \
         Report that half as UNVERIFIED, which is what the story's acceptance criterion permits."
    );
    drop(balloon);
    Ok(())
}
