//! Shared plumbing for the rung-4 real-weight harnesses (SC-15792).
//!
//! `rung4_block_streaming_spike.rs` (SC-15791) and `capped_pool_vram_ceiling.rs` (SC-16091) had grown
//! near-identical copies of the tier header parser and the block materializer, and both said so:
//! *"duplicated ... rather than shared: these are separate test binaries and the spike is deliberately
//! throwaway. SC-15792 will own the real one."* This is that one.
//!
//! It lives under `tests/` rather than in `candle_gen::testkit` on purpose. The CUDA compile/Clippy
//! lane enables `cuda` and **not** `testkit` (ci.yml), so anything these cuda-gated test binaries
//! import from a `testkit`-gated module would drop them out of the only lane that can build them —
//! the same constraint that produced the forks in the first place. A `tests/` submodule is compiled
//! into each binary from one source file, which satisfies both. (The *driver-level* probe did move
//! into the crate, at `candle_gen::cuda_mempool`, because `cuda` alone gates it.)

#![allow(dead_code)]

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::{lin, QLinear};

pub const MIB: f64 = 1024.0 * 1024.0;
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------------
// Driver memory-pool probe
//
// The mapping to the MLX spike's counters, which is what makes the two backends comparable at all:
//   MLX get_active_memory  ↔  CU_MEMPOOL_ATTR_USED_MEM_CURRENT
//   MLX get_peak_memory    ↔  CU_MEMPOOL_ATTR_USED_MEM_HIGH
//   MLX get_cache_memory   ↔  RESERVED_MEM_CURRENT − USED_MEM_CURRENT
//   MLX clear_cache()      ↔  cuMemPoolTrimTo(pool, 0)
// ---------------------------------------------------------------------------------------------------

/// Panicking adapter over [`candle_gen::cuda_mempool`].
///
/// The only thing it adds is the unwrap policy: every accessor **panics** on a driver error rather
/// than returning `Option`, because in a measurement context a silent `unwrap_or(0)` lets a broken
/// probe print a plausible report and bank a green. The driver calls, the counter semantics and the
/// two traps are documented once, in the shared module.
pub struct Pool(candle_gen::cuda_mempool::MemPool);

impl Pool {
    /// The default pool candle allocates from. Under a custom **current** pool this is the wrong
    /// handle and reports ~0 — see `candle_gen::cuda_mempool`'s trap 1. Use [`Pool::wrap`] there.
    pub fn open(ordinal: i32) -> Option<Self> {
        candle_gen::cuda_mempool::MemPool::device_default(ordinal).map(Self)
    }

    /// Wrap a pool handle the caller installed (e.g. a capped one), so the counters follow the pool
    /// the allocations actually land in.
    pub fn wrap(pool: candle_gen::cuda_mempool::MemPool) -> Self {
        Self(pool)
    }

    pub fn used(&self) -> u64 {
        self.0.used().expect("CU_MEMPOOL_ATTR_USED_MEM_CURRENT")
    }

    pub fn used_high(&self) -> u64 {
        self.0.used_high().expect("CU_MEMPOOL_ATTR_USED_MEM_HIGH")
    }

    pub fn reserved(&self) -> u64 {
        self.0
            .reserved()
            .expect("CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT")
    }

    pub fn reserved_high(&self) -> u64 {
        self.0
            .reserved_high()
            .expect("CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH")
    }

    pub fn release_threshold(&self) -> u64 {
        self.0
            .release_threshold()
            .expect("CU_MEMPOOL_ATTR_RELEASE_THRESHOLD")
    }

    pub fn reset_high(&self) {
        assert!(
            self.0.reset_high_water(),
            "resetting the high-water marks failed — peaks would be stale"
        );
    }

    /// A no-op at a release threshold of 0, which is where candle leaves it. Kept because the
    /// spike's finding is precisely that it recovers nothing.
    pub fn trim(&self) {
        assert!(self.0.trim(), "cuMemPoolTrimTo failed");
    }
}

/// Driver-level `(free, total)` bytes — what a smaller card's VRAM gate would see. Needs a current
/// context; see `candle_gen::cuda_mempool::mem_info`.
pub fn mem_info() -> (u64, u64) {
    candle_gen::cuda_mempool::mem_info().expect("cuMemGetInfo_v2 (needs a current context)")
}

/// The host's VRAM, disclosed in every report — SC-15256's closing note is that an acceptance
/// measured as gate arithmetic on a 97.9 GB card is not evidence about an 8 GB one.
pub fn disclose_host(tag: &str, pool: &Pool) {
    let (free, total) = mem_info();
    assert!(
        total > 0 && free > 0,
        "the driver reported no memory — a zeroed probe must not be allowed to satisfy the \
         host-VRAM-disclosure requirement"
    );
    println!(
        "[{tag}] HOST: CUDA device 0 total {:.1} GiB, free now {:.1} GiB | pool used {:.1} MiB / \
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
/// measurement inherits the previous one. That is not hypothetical: SC-15791's first sweep reported
/// window 1's reserved peak as 3488.0 MiB — exactly the fully-resident control's figure — because the
/// control's pages had not been returned when the reset ran.
pub fn quiesce_and_reset(dev: &Device, pool: &Pool) -> Result<()> {
    dev.synchronize()?;
    pool.trim();
    pool.reset_high();
    Ok(())
}

pub fn env_path_req(key: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(key).unwrap_or_else(|_| panic!("{key} not set — see the module docstring")),
    )
}

/// An OPTIONAL env path: `None` prints a SKIP rather than panicking, so the documented
/// `-- --ignored` invocation does not fail arms whose inputs were not supplied (the house pattern —
/// `mlx_repack_real_weights.rs` uses `.ok()` for its optional tiers).
pub fn env_path_opt(tag: &str, key: &str) -> Option<PathBuf> {
    match std::env::var(key) {
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            println!("[{tag}] SKIP: {key} not set");
            None
        }
    }
}

pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// Least-squares fit of `peak ≈ a·window + b`, in whatever unit the points carry.
pub fn linear_fit(points: &[(usize, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|(x, _)| *x as f64).sum();
    let sy: f64 = points.iter().map(|(_, y)| *y).sum();
    let sxx: f64 = points.iter().map(|(x, _)| (*x as f64) * (*x as f64)).sum();
    let sxy: f64 = points.iter().map(|(x, y)| (*x as f64) * *y).sum();
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (a, (sy - a * sx) / n)
}

/// Median of a small sample. Every headline timing is reported as a median with its range, because a
/// single sample of this quantity varies ~10% run to run on this host (and up to 38% at the tail).
pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------------------------------
// The tier under test
// ---------------------------------------------------------------------------------------------------

/// Everything about a packed transformer tier that can be read from the safetensors **header** alone.
pub struct Tier {
    pub path: PathBuf,
    /// Per block, the base names of the packed `weight`/`scales`/`biases` triples.
    pub packed: Vec<Vec<String>>,
    /// Per block, the keys of the dense (unpacked) tensors.
    pub dense: Vec<Vec<String>>,
    /// `base -> (out_dim, in_dim)`, recovered from the packed shapes plus the group size.
    pub dims: HashMap<String, (usize, usize)>,
    pub bytes: Vec<usize>,
    pub n_tensors: usize,
    pub file_bytes: u64,
}

impl Tier {
    pub fn open(path: &Path, group_size: usize) -> Result<Self> {
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

        Ok(Self {
            path: path.to_path_buf(),
            packed,
            dense,
            dims,
            bytes,
            n_tensors,
            file_bytes: std::fs::metadata(path)?.len(),
        })
    }

    pub fn n_blocks(&self) -> usize {
        self.packed.len()
    }

    pub fn block_bytes(&self, b: usize) -> usize {
        self.bytes[b]
    }

    pub fn total_block_bytes(&self) -> usize {
        self.bytes.iter().sum()
    }

    /// A **fresh** weights view — what the caller hands to
    /// `candle_gen::block_window::run_windowed`'s `open`. Header-only: mapping a file faults no page.
    pub fn open_view(&self, dev: &Device) -> Result<VarBuilder<'static>> {
        // SAFETY: immutable HF-cache blob; a fresh mmap per view, never mutated behind the mapping.
        let st = unsafe { MmapedSafetensors::new(&self.path)? };
        Ok(VarBuilder::from_backend(
            Box::new(st),
            DType::F32,
            dev.clone(),
        ))
    }
}

/// One materialized transformer block.
pub struct Block {
    /// `(name, projection, in_dim)`.
    pub lins: Vec<(String, QLinear, usize)>,
    pub dense: Vec<Tensor>,
}

impl Block {
    pub fn dense_bytes(&self) -> usize {
        self.dense
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// Materialize `range` through the **production** packed loader (`candle_gen::quant::lin`) onto the
/// view's device.
///
/// NOTE the round trip this incurs when the view is on CUDA: `lin_gs` loads `wq`/`scales`/`biases`
/// onto the view's device (`quant/mod.rs:973-975`), then `repack::q4_parts` immediately pulls all
/// three back with `to_device(&Cpu)` (`repack.rs:133-144`) to do the host repack, then uploads the
/// `Q4_1` bytes. SC-15791 measured that waste at 62.5 ms/block; hoisting it is SC-16096, not this.
pub fn materialize(tier: &Tier, view: &VarBuilder, range: Range<usize>) -> Result<Vec<Block>> {
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

/// A plausible per-block forward. Accumulates into an on-DEVICE scalar and never reads it back: a
/// `to_scalar` per projection would synchronize the stream on every call, serializing exactly the
/// asynchrony these harnesses are measuring.
pub fn compute(blocks: &[Block], tokens: usize, dev: &Device, acc: &Tensor) -> Result<Tensor> {
    let mut acc = acc.clone();
    for b in blocks {
        for (_, ql, in_dim) in &b.lins {
            let x = Tensor::ones((tokens, *in_dim), DType::F32, dev)?;
            acc = (acc + ql.forward(&x)?.sum_all()?)?;
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------------------------------
// The enforced ceiling (sc-16091)
// ---------------------------------------------------------------------------------------------------

/// An explicitly created, size-capped stream-ordered pool installed as device `ordinal`'s **current**
/// pool — the only route to a genuine small-card verdict on this hardware.
///
/// sc-16091 established why a balloon is not a substitute: this host absorbed a 3.41 GiB working set
/// into 1.93 GiB of driver-visible free VRAM at 1.07x wall time, so neither completion nor timing
/// detects the spill. `CUmemPoolProps.maxSize` IS enforced, and it binds candle because cudarc calls
/// bare `cuMemAllocAsync`, which draws from the device's *current* pool.
///
/// Restores the device's original pool on drop: this is a process-global device property, and leaving
/// it installed would silently cap every later test in the binary.
///
/// **Honest limit.** This is an *allocator* ceiling, not a physical card. The CUDA context and library
/// workspaces sit outside the pool — measured at a 40 MiB floor on a trivial workload, and it grows
/// with kernel count and cuBLASLt workspaces, so `cap = target_card − overhead` must be re-measured
/// per workload rather than inherited.
pub struct CappedPool {
    pool: candle_gen::candle_core::cuda::cudarc::driver::sys::CUmemoryPool,
    previous: candle_gen::candle_core::cuda::cudarc::driver::sys::CUmemoryPool,
    ordinal: i32,
}

fn cuda_device(
    ordinal: i32,
) -> Option<candle_gen::candle_core::cuda::cudarc::driver::sys::CUdevice> {
    use candle_gen::candle_core::cuda::cudarc::driver::sys;
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
    /// allocation, so a later install still caps subsequent allocations — but anything already
    /// resident came from the previous pool and is invisible to the cap.
    pub fn install(ordinal: i32, cap_bytes: usize) -> Option<Self> {
        use candle_gen::candle_core::cuda::cudarc::driver::sys;
        let dev = cuda_device(ordinal)?;
        unsafe {
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

    /// Counters for the CAPPED pool. Reading `Pool::open` (the *default* pool) while this is
    /// installed is `candle_gen::cuda_mempool`'s trap 1: it reports ~0 while every allocation lands
    /// here, which makes "the peak stayed within budget" trivially true.
    pub fn counters(&self) -> Pool {
        Pool::wrap(candle_gen::cuda_mempool::MemPool::from_raw(self.pool))
    }
}

impl Drop for CappedPool {
    fn drop(&mut self) {
        use candle_gen::candle_core::cuda::cudarc::driver::sys;
        unsafe {
            if let Some(dev) = cuda_device(self.ordinal) {
                sys::cuDeviceSetMemPool(dev, self.previous);
            }
            sys::cuMemPoolDestroy(self.pool);
        }
    }
}
