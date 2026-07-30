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
//! SC15791_Q4=<...>/q4/transformer/model.safetensors     (required)
//! SC15791_Q8=<...>/q8/transformer/model.safetensors     (optional: the q8 tier arm)
//! SC15791_TOKENS=1024                                   (optional: per-block compute width)
//! SC15791_BALLOON_GIB=88                                (optional: constrained-budget arm)
//! cargo test -p candle-gen --features cuda --test rung4_block_streaming_spike -- --ignored --nocapture
//! ```
//!
//! ## What each test answers
//!
//! | Test | Story question |
//! |---|---|
//! | `q1_window_sweep_cost` | Q1 cost per window and its scaling; Q5 overlap |
//! | `q2_q3_release_semantics` | Q2 does VRAM come back / need a sync; Q3 must `release` be non-trivial |
//! | `q4_packed_quant_per_block` | Q4 the packed-quant triple, per block, bit-exact |
//! | `constrained_budget_sweep` | the small-card disclosure (opt-in balloon) |
//!
//! Throwaway measurement code — the answer is the deliverable, not the implementation. The
//! implementation lands in SC-15792 against `gen_core::block_window::BlockWindowBackend`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::quant::{dequant_mlx_q4_reference, lin, QLinear, MLX_GROUP_SIZE};

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------------
// Driver memory-pool probe
//
// `candle_gen::testkit` exposes only `USED_MEM_HIGH`; this spike needs the CURRENT/RESERVED pair too
// (the MLX `get_active_memory` / `get_cache_memory` analogues) plus `cuMemPoolTrimTo`, so the probe is
// local. SC-15792 should hoist whichever of these the implementation ends up needing.
//
// The mapping to the MLX spike's counters, which is what makes the two comparable at all:
//   MLX get_active_memory  ↔  CU_MEMPOOL_ATTR_USED_MEM_CURRENT
//   MLX get_peak_memory    ↔  CU_MEMPOOL_ATTR_USED_MEM_HIGH
//   MLX get_cache_memory   ↔  RESERVED_MEM_CURRENT − USED_MEM_CURRENT
//   MLX clear_cache()      ↔  cuMemPoolTrimTo(pool, 0)
// ---------------------------------------------------------------------------------------------------
mod pool {
    use candle_gen::candle_core::cuda::cudarc::driver::sys;
    use std::ffi::c_void;

    pub struct Pool(sys::CUmemoryPool);

    // SAFETY: a `CUmemoryPool` is an opaque driver handle for a *device*, not a context-bound or
    // thread-bound resource; the driver API is thread-safe. Needed because the Q5 overlap arm reads
    // the pool from a worker thread.
    unsafe impl Send for Pool {}
    unsafe impl Sync for Pool {}

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

        fn attr(&self, attr: sys::CUmemPool_attribute) -> u64 {
            let mut v: u64 = 0;
            unsafe {
                if sys::cuMemPoolGetAttribute(self.0, attr, (&mut v as *mut u64).cast::<c_void>())
                    != sys::CUresult::CUDA_SUCCESS
                {
                    return 0;
                }
            }
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

        /// Cached-but-free bytes: reserved − used. The MLX `get_cache_memory` analogue.
        pub fn cached(&self) -> u64 {
            self.reserved().saturating_sub(self.used())
        }

        /// Reset the `USED_MEM_HIGH` watermark (write-to-zero per the driver ABI).
        pub fn reset_high(&self) -> bool {
            let mut zero: u64 = 0;
            unsafe {
                sys::cuMemPoolSetAttribute(
                    self.0,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                    (&mut zero as *mut u64).cast::<c_void>(),
                ) == sys::CUresult::CUDA_SUCCESS
            }
        }

        /// Return cached-free pool pages to the driver — the `clear_cache()` analogue.
        pub fn trim(&self) -> bool {
            unsafe { sys::cuMemPoolTrimTo(self.0, 0) == sys::CUresult::CUDA_SUCCESS }
        }
    }

    /// Driver-level `(free, total)` bytes for the current context — what `nvidia-smi` reports, i.e.
    /// what a VRAM gate on a smaller card would actually see.
    pub fn mem_info() -> (u64, u64) {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe {
            if sys::cuMemGetInfo_v2(&mut free, &mut total) != sys::CUresult::CUDA_SUCCESS {
                return (0, 0);
            }
        }
        (free as u64, total as u64)
    }
}

// ---------------------------------------------------------------------------------------------------
// The tier under test
// ---------------------------------------------------------------------------------------------------

/// Everything about a packed transformer tier that can be read from the safetensors **header** alone —
/// no tensor bytes are touched, which is what makes `open_view` cheap enough for rung 4.
struct Tier {
    path: PathBuf,
    /// Packed bases (`{base}.scales` present) per block index.
    packed: Vec<Vec<String>>,
    /// Non-packed tensors (norm weights, dense biases) per block index.
    dense: Vec<Vec<String>>,
    /// `base` → `(out_dim, in_dim)` recovered from the packed shapes.
    dims: HashMap<String, (usize, usize)>,
    /// On-disk bytes per block.
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

        // How many blocks: the max `layers.{i}.` index.
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

        // Shape/size census straight off the header — `.data().len()` is a length on the mmap view,
        // not a read, so no page is faulted here.
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
                // out_dim from the u32 code matrix `[out, in/(32/bits)]`; in_dim from the scales'
                // group axis `[out, in/group_size]`. A `.scales` with no `.weight` sibling is a
                // malformed tier, not something to paper over.
                let w = shape
                    .get(format!("{base}.weight").as_str())
                    .unwrap_or_else(|| panic!("{base}.scales has no {base}.weight sibling"));
                packed[b].push(base.to_string());
                dims.insert(base.to_string(), (w.0[0], v.shape()[1] * group_size));
            } else if let Some(base) = k.strip_suffix(".weight") {
                // A `.weight` is packed iff it has a `.scales` sibling; otherwise it is a dense
                // norm weight the window must also carry.
                if !shape.contains_key(format!("{base}.scales").as_str()) {
                    dense[b].push(k.to_string());
                }
            } else if !k.ends_with(".biases") {
                // Dense biases and anything else block-scoped. `.biases` is the packed triple's own
                // member and is loaded through `lin`, so it must NOT be double-counted here.
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
            file_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        })
    }

    fn n_blocks(&self) -> usize {
        self.packed.len()
    }

    fn block_bytes(&self, range: Range<usize>) -> usize {
        range.map(|b| self.bytes[b]).sum()
    }

    fn all_block_bytes(&self) -> usize {
        self.bytes.iter().sum()
    }

    /// A **fresh** weights view — the `BlockWindowBackend::open_view` analogue. Header-only: a new
    /// mmap plus the safetensors index, no tensor bytes.
    fn open_view(&self, dev: &Device) -> Result<VarBuilder<'static>> {
        // SAFETY: immutable HF-cache blob; a fresh mmap per view, never mutated behind the mapping.
        let st = unsafe { MmapedSafetensors::new(&self.path)? };
        Ok(VarBuilder::from_backend(
            Box::new(st),
            DType::F32,
            dev.clone(),
        ))
    }
}

/// One materialized transformer block: its packed projections plus the dense norm weights.
///
/// The dense half is small but must be carried: a window that materializes only the packed
/// projections is not a runnable block, and leaving it out would understate per-window residency.
struct Block {
    /// `(name, projection, in_dim)`.
    lins: Vec<(String, QLinear, usize)>,
    dense: Vec<Tensor>,
}

impl Block {
    /// Device bytes held by the dense (non-packed) half — reported so the residency figure is
    /// visibly the whole block, not just its projections.
    fn dense_bytes(&self) -> usize {
        self.dense
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// Materialize `range`'s blocks out of `view` onto the view's device — the host-to-device block
/// materialization whose cost this spike exists to measure. Goes through the **production** packed
/// loader (`candle_gen::quant::lin`), not a bespoke read, so the number includes the real
/// mmap-read → host repack → H2D upload chain a rung-4 implementation would pay.
fn materialize(tier: &Tier, view: &VarBuilder, range: Range<usize>) -> Result<Vec<Block>> {
    let mut out = Vec::with_capacity(range.len());
    for b in range {
        let mut lins = Vec::with_capacity(tier.packed[b].len());
        for base in &tier.packed[b] {
            let (out_dim, in_dim) = tier.dims[base];
            let ql = lin(view, base, in_dim, out_dim, false)?;
            lins.push((base.clone(), ql, in_dim));
        }
        let mut dense = Vec::with_capacity(tier.dense[b].len());
        for key in &tier.dense[b] {
            dense.push(view.get_unchecked_dtype(key, DType::F32)?);
        }
        out.push(Block { lins, dense });
    }
    Ok(out)
}

/// A plausible per-block forward: push an activation through every packed projection. Not the real
/// DiT graph — the point is to give the transfer something to overlap WITH and to keep the
/// materialized weights genuinely referenced, so the drop below is a real drop.
///
/// Accumulates into an on-DEVICE scalar and never reads it back: a `to_scalar` per projection would
/// synchronize the stream on every call, which would serialize exactly the overlap the Q5 arm is
/// trying to detect. The caller reads the accumulator once, at the end.
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

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(key).unwrap_or_else(|_| panic!("{key} not set — see the module docstring")),
    )
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
    println!(
        "[sc-15791] HOST: CUDA device 0 total {:.1} GiB, free at start {:.1} GiB | pool reserved \
         {:.1} MiB / used {:.1} MiB",
        total as f64 / GIB,
        free as f64 / GIB,
        pool.reserved() as f64 / MIB,
        pool.used() as f64 / MIB
    );
}

fn windows(n_blocks: usize, window: usize) -> impl Iterator<Item = Range<usize>> {
    (0..n_blocks).step_by(window).map(move |s| {
        let e = (s + window).min(n_blocks);
        s..e
    })
}

// ---------------------------------------------------------------------------------------------------
// Q1 (cost per window, scaling) + Q5 (overlap)
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn q1_window_sweep_cost() -> Result<()> {
    let q4 = env_path("SC15791_Q4");
    let tokens = env_usize("SC15791_TOKENS", 1024);
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);

    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;
    println!(
        "[sc-15791] TIER {}: {:.2} GiB on disk, {} tensors, {} blocks, {:.2} GiB of block weights, \
         {} packed triples/block",
        q4.display(),
        tier.file_bytes as f64 / GIB,
        tier.n_tensors,
        tier.n_blocks(),
        tier.all_block_bytes() as f64 / GIB,
        tier.packed[0].len(),
    );

    // ── The fully-resident control, which doubles as the page-cache warm-up. This is the A-side the
    // windowed numbers below are a saving *against*: one pass materializing every block at once.
    //
    // Everything after this is a WARM page-cache measurement, exactly as SC-15744's 0.309 s/step was.
    // The cold-start cost is NOT isolated here: Windows offers no supported way to drop the page
    // cache, and this host has enough RAM to hold the whole 3.23 GiB tier indefinitely. Treat every
    // figure below as the optimistic bound and the cold cost as UNVERIFIED.
    pool.reset_high();
    let t_res = Instant::now();
    let (resident_secs, resident_peak) = {
        let view = tier.open_view(&dev)?;
        let b = materialize(&tier, &view, 0..tier.n_blocks())?;
        dev.synchronize()?;
        let secs = t_res.elapsed().as_secs_f64();
        let peak = pool.used();
        drop(b);
        drop(view);
        (secs, peak)
    };
    println!(
        "\n[sc-15791] CONTROL — all {} blocks resident at once: {:.3} s to load, {:.1} MiB live",
        tier.n_blocks(),
        resident_secs,
        resident_peak as f64 / MIB
    );
    pool.trim();

    // ── Q1: one full denoise step's worth of re-materialization, per window size.
    println!(
        "\n[sc-15791] Q1 — cost of ONE denoise step (all {} blocks re-materialized), by window size",
        tier.n_blocks()
    );
    println!(
        "  {:>6} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "window", "windows", "step (s)", "per-block ms", "peak live MiB", "hdr bytes MiB"
    );
    let mut results: Vec<(usize, f64, f64)> = Vec::new();
    for window in [1usize, 2, 3, 5, 6, 10, 15, 30] {
        pool.reset_high();
        let base_used = pool.used();
        let t0 = Instant::now();
        for range in windows(tier.n_blocks(), window) {
            let view = tier.open_view(&dev)?;
            let blocks = materialize(&tier, &view, range)?;
            dev.synchronize()?;
            drop(blocks);
            drop(view);
        }
        dev.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();
        let peak = pool.used_high().saturating_sub(base_used);
        let hdr = tier.block_bytes(0..window.min(tier.n_blocks()));
        println!(
            "  {window:>6} {:>9} {elapsed:>12.3} {:>12.1} {:>12.1} {:>12.1}",
            tier.n_blocks().div_ceil(window),
            elapsed * 1000.0 / tier.n_blocks() as f64,
            peak as f64 / MIB,
            hdr as f64 / MIB,
        );
        results.push((window, elapsed, peak as f64 / MIB));
        pool.trim();
    }

    // ── Where the time actually goes. PCIe is the assumed bound; measure whether it is.
    println!("\n[sc-15791] Q1b — cost decomposition for one 1-block window (mean of 5)");
    let mut host_only = Duration::ZERO;
    let mut full = Duration::ZERO;
    let cpu = Device::Cpu;
    for i in 0..5 {
        let b = i % tier.n_blocks();
        // Host leg: mmap read + Q4→Q4_1 repack, landing on the CPU. No PCIe, no device alloc.
        let view = tier.open_view(&cpu)?;
        let t = Instant::now();
        let blocks = materialize(&tier, &view, b..b + 1)?;
        host_only += t.elapsed();
        drop(blocks);
        drop(view);

        // Full leg: the same work landing on CUDA.
        let view = tier.open_view(&dev)?;
        let t = Instant::now();
        let blocks = materialize(&tier, &view, b..b + 1)?;
        dev.synchronize()?;
        full += t.elapsed();
        drop(blocks);
        drop(view);
    }
    let host_ms = host_only.as_secs_f64() * 1000.0 / 5.0;
    let full_ms = full.as_secs_f64() * 1000.0 / 5.0;
    let block_mib = tier.bytes[0] as f64 / MIB;
    println!(
        "  host read+repack {host_ms:.1} ms | host+H2D {full_ms:.1} ms | H2D-attributable \
         {:.1} ms ({:.0}% of total) | block {block_mib:.1} MiB on disk ⇒ apparent H2D \
         {:.1} GiB/s",
        full_ms - host_ms,
        (full_ms - host_ms) / full_ms * 100.0,
        (block_mib / 1024.0) / ((full_ms - host_ms).max(1e-3) / 1000.0),
    );

    // ── Q5: can the next window's materialization overlap the current window's compute?
    // The host leg (mmap read + repack) is pure CPU and can genuinely run concurrently; the H2D copy
    // and the compute kernels share candle's single per-device stream, so those cannot. Measure the
    // net rather than reasoning about it.
    println!("\n[sc-15791] Q5 — overlap of the next window's transfer with this window's compute");
    let tier = std::sync::Arc::new(tier);
    let window = 1usize;
    pool.reset_high();
    let seq_base = pool.used();
    let (seq, seq_peak) = {
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
        let e = t.elapsed().as_secs_f64();
        std::hint::black_box(acc.last().unwrap().to_scalar::<f32>()?);
        (e, pool.used_high().saturating_sub(seq_base))
    };

    // Prefetch arm: a worker thread materializes window i+1 while the main thread computes window i.
    // The `Tier` (header census) is SHARED via `Arc` — re-parsing the header per prefetch would
    // charge the overlap arm for work the sequential arm never does and manufacture a fake loss.
    let overlapped = {
        let t = Instant::now();
        type Prefetch = Result<(Vec<Block>, VarBuilder<'static>)>;
        let mut pending: Option<std::thread::JoinHandle<Prefetch>> = None;
        let ranges: Vec<Range<usize>> = windows(tier.n_blocks(), window).collect();
        let mut acc = Vec::new();
        for (i, range) in ranges.iter().enumerate() {
            // Take whatever the previous iteration prefetched, else materialize inline.
            let (blocks, view) = match pending.take() {
                Some(h) => h.join().expect("prefetch thread")?,
                None => {
                    let view = tier.open_view(&dev)?;
                    let blocks = materialize(&tier, &view, range.clone())?;
                    (blocks, view)
                }
            };
            // Kick off the NEXT window's materialization before computing this one.
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
        let e = t.elapsed().as_secs_f64();
        std::hint::black_box(acc.last().unwrap().to_scalar::<f32>()?);
        e
    };
    println!(
        "  tokens={tokens} window={window}: sequential {seq:.3} s | prefetched {overlapped:.3} s | \
         saving {:.1}%",
        (seq - overlapped) / seq * 100.0
    );
    // The transient the packed forward itself allocates. `MatmulStrategy::DequantDense` (the sc-7702
    // fix) dequantizes each packed weight to a DENSE tensor per forward, so a rung-4 window bounds
    // the STORED weights while the compute still materializes one dense projection at a time. If that
    // transient dominates, bounding residency alone does not move the request peak — which the epic
    // explicitly says is not a saving.
    println!(
        "  peak live across materialize+compute: {:.1} MiB (vs {:.1} MiB for materialize alone at \
         window 1) — the excess is the per-forward dequant transient",
        seq_peak as f64 / MIB,
        results
            .iter()
            .find(|(w, _, _)| *w == 1)
            .map(|(_, _, p)| *p)
            .unwrap_or(0.0),
    );

    println!("\n[sc-15791] Q1 SUMMARY (window, step seconds, peak live MiB): {results:?}");
    println!(
        "[sc-15791] REDUCTION vs the resident control ({:.1} MiB): window 1 = {:.1}x, and each \
         denoise step costs {:.3} s more than the resident path's zero",
        resident_peak as f64 / MIB,
        resident_peak as f64
            / results
                .iter()
                .find(|(w, _, _)| *w == 1)
                .map(|(_, _, p)| p * MIB)
                .unwrap_or(f64::INFINITY),
        results
            .iter()
            .find(|(w, _, _)| *w == 1)
            .map(|(_, s, _)| *s)
            .unwrap_or(0.0),
    );
    Ok(())
}

/// The q8 tier, measured rather than extrapolated.
///
/// SC-15744 closed by reasoning that "a q8 transformer is roughly 2x the q4 figure, so ~190 MiB/block"
/// — true on MLX, where both tiers are the same affine pack and materialization is the same page
/// fault. **On Candle the two tiers take structurally different code paths.** Q4 repacks losslessly
/// into GGML `Q4_1` (a byte shuffle). Q8 has no affine GGML container, so `repack_packed_weight`
/// materializes the FULL dense f32 grid on the host and re-quantizes it to symmetric `Q8_0`
/// (`quant/mod.rs:930`). That is a per-window host transient the q4 path never pays, so the 2x
/// extrapolation is exactly the kind of cross-backend carry-over this spike exists to refuse.
#[test]
#[ignore = "needs a CUDA host with the hosted z-image q8 tier (SC15791_Q8 env)"]
fn q1b_q8_tier_cost() -> Result<()> {
    let q8 = env_path("SC15791_Q8");
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

    // Warm, then measure one window-1 pass over the whole stack.
    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..2)?);
        dev.synchronize()?;
    }
    pool.trim();

    pool.reset_high();
    let base = pool.used();
    let t = Instant::now();
    for range in windows(tier.n_blocks(), 1) {
        let view = tier.open_view(&dev)?;
        let blocks = materialize(&tier, &view, range)?;
        dev.synchronize()?;
        drop(blocks);
        drop(view);
    }
    dev.synchronize()?;
    let step = t.elapsed().as_secs_f64();
    let peak = pool.used_high().saturating_sub(base);
    println!(
        "  window 1: one denoise step re-materializes all {} blocks in {step:.3} s ({:.1} ms/block); \
         peak live {:.1} MiB/block",
        tier.n_blocks(),
        step * 1000.0 / tier.n_blocks() as f64,
        peak as f64 / MIB
    );
    println!(
        "  Compare against the q4 arm before repeating SC-15744's \"q8 is roughly 2x q4\" — on \
         Candle the q8 path additionally materializes a dense f32 grid per projection on the host."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Q2 (does the memory come back / does it need a sync) + Q3 (must `release` be non-trivial)
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn q2_q3_release_semantics() -> Result<()> {
    let q4 = env_path("SC15791_Q4");
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;

    // Warm, then trim, so the numbers below are this window's and not the warm-up's.
    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..1)?);
        dev.synchronize()?;
    }
    pool.trim();

    println!("\n[sc-15791] Q2 — does dropping a window return the memory?");
    let (free_before, _) = pool::mem_info();
    let used_before = pool.used();
    let reserved_before = pool.reserved();

    let view = tier.open_view(&dev)?;
    let blocks = materialize(&tier, &view, 0..1)?;
    // Deliberately NOT synchronized yet: the first read is the "is it even visible" question.
    let used_live_nosync = pool.used();
    dev.synchronize()?;
    let used_live = pool.used();
    let reserved_live = pool.reserved();
    let (free_live, _) = pool::mem_info();

    // Drop with NO synchronize — the question the story asks directly.
    drop(blocks);
    drop(view);
    let used_after_drop_nosync = pool.used();
    let (free_after_drop_nosync, _) = pool::mem_info();

    dev.synchronize()?;
    let used_after_sync = pool.used();
    let (free_after_sync, _) = pool::mem_info();
    let cached_after_sync = pool.cached();

    // And the explicit pool trim — the `clear_cache()` analogue.
    pool.trim();
    let (free_after_trim, _) = pool::mem_info();
    let reserved_after_trim = pool.reserved();

    let block_mib = tier.bytes[0] as f64 / MIB;
    println!(
        "  one block = {block_mib:.1} MiB on disk\n  \
         pool used:     before {:.1} | live(no sync) {:.1} | live(synced) {:.1} | after drop(no \
         sync) {:.1} | after drop+sync {:.1}  MiB\n  \
         driver free:   before {:.2} | live {:.2} | after drop(no sync) {:.2} | after drop+sync \
         {:.2} | after trim {:.2}  GiB\n  \
         pool reserved: before {:.1} | while live {:.1} | after trim {:.1}  MiB (cached-but-free \
         after the drop: {:.1} MiB)",
        used_before as f64 / MIB,
        used_live_nosync as f64 / MIB,
        used_live as f64 / MIB,
        used_after_drop_nosync as f64 / MIB,
        used_after_sync as f64 / MIB,
        free_before as f64 / GIB,
        free_live as f64 / GIB,
        free_after_drop_nosync as f64 / GIB,
        free_after_sync as f64 / GIB,
        free_after_trim as f64 / GIB,
        reserved_before as f64 / MIB,
        reserved_live as f64 / MIB,
        reserved_after_trim as f64 / MIB,
        cached_after_sync as f64 / MIB,
    );

    // The load-bearing claim: a dropped window's bytes stop being LIVE. Whether they return to the
    // driver or stay in the pool as reusable cache is the next line's question, not this one.
    let live_delta = used_live.saturating_sub(used_before) as f64 / MIB;
    let residual = used_after_sync.saturating_sub(used_before) as f64 / MIB;
    assert!(
        live_delta > block_mib * 0.5,
        "materializing a block must show up as live pool bytes (saw {live_delta:.1} MiB for a \
         {block_mib:.1} MiB block) — if it does not, the probe is not watching candle's allocator"
    );
    assert!(
        residual < live_delta * 0.1,
        "dropping the window must return its bytes to the pool: {residual:.1} MiB still live out \
         of {live_delta:.1} MiB"
    );

    // ── Q3, the memory half: does the release need a synchronize to be observable?
    println!("\n[sc-15791] Q3a — is a stream synchronize required for the memory to come back?");
    let freed_without_sync = used_live.saturating_sub(used_after_drop_nosync) as f64 / MIB;
    let freed_with_sync = used_live.saturating_sub(used_after_sync) as f64 / MIB;
    println!(
        "  freed by the bare drop: {freed_without_sync:.1} MiB | after an added synchronize: \
         {freed_with_sync:.1} MiB  ⇒ sync {} for RECLAIM",
        if freed_with_sync - freed_without_sync > block_mib * 0.05 {
            "IS REQUIRED"
        } else {
            "is NOT required"
        }
    );

    // ── Q3, the correctness half. This is the trap the story predicts: not laziness (candle is
    // eager) but device memory being recycled under kernels that are still in flight. Compute with
    // window N's weights, drop WITHOUT reading the result back, immediately materialize window N+1
    // into the pages just freed, and only then read window N's output. If the stream-ordered
    // allocator did not protect us, the result would be corrupt.
    println!(
        "\n[sc-15791] Q3b — is a synchronize required for CORRECTNESS across a window boundary?"
    );
    // The compute must still be IN FLIGHT when the drop happens, or the probe is vacuous: a single
    // small matmul finishes in well under a millisecond while materializing the next window takes
    // tens, so the race would never be armed. `compute` chains every projection in the block at a
    // deliberately large token count — hundreds of GFLOP, tens of ms of queued stream work — and
    // reads nothing back, so the free below genuinely lands under running kernels.
    let tokens = env_usize("SC15791_RACE_TOKENS", 4096);
    // `repeats` deepens the queue until the outstanding work outlasts the next window's
    // materialization. Without it the probe silently does not discriminate — the first run of this
    // spike enqueued 1.8 ms of work and then spent 3.8 s materializing, so the kernels had long
    // drained by the time the pages were recycled and the "IDENTICAL" result meant nothing.
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
    let (racy, inflight_ms) = {
        let view = tier.open_view(&dev)?;
        let blocks = materialize(&tier, &view, 0..1)?;
        let t = Instant::now();
        let y = chain(&blocks, &dev)?; // launched, NOT awaited
        let launch = t.elapsed();
        drop(blocks); // window N's weights freed while its kernels may still reference them
        drop(view);
        // Claim the pages the drop just released, and claim them with `ones` — a memset to 1.0f32.
        //
        // The first version of this probe re-materialized the next window here, which took 1.6 s
        // (cold read + host repack) and let the kernels drain long before the pages were touched, so
        // the probe never armed. A bare allocation is ~1 ms, which keeps the recycled-page window
        // tight, and filling it with 1.0 rather than 0.0 makes corruption maximally visible: if the
        // stream-ordered free were NOT ordered behind the still-running matmuls, those matmuls would
        // read 1.0 where a packed weight used to be and the result would move enormously.
        let claim_mib = env_usize("SC15791_RACE_CLAIM_MIB", 256);
        let chunk = 8 * 1024 * 1024 / 4; // 8 MiB of f32
        let claims: Vec<Tensor> = (0..(claim_mib / 8).max(1))
            .map(|_| Tensor::ones(chunk, DType::F32, &dev))
            .collect::<Result<_>>()?;
        let reuse_done = t.elapsed();
        let v = y.to_scalar::<f32>()?; // only NOW is the result read (this is the first sync)
        let total = t.elapsed();
        drop(claims);
        // If the whole thing took barely longer than the enqueue, the kernels had already drained
        // and the race was never armed — report that rather than claiming a clean result.
        (
            v,
            (
                launch.as_secs_f64() * 1000.0,
                reuse_done.as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0,
            ),
        )
    };
    let (launch_ms, reuse_ms, total_ms) = inflight_ms;
    println!(
        "  enqueue returned at {launch_ms:.1} ms | freed pages re-claimed and memset to 1.0 at \
         {reuse_ms:.1} ms | first sync completed at {total_ms:.1} ms\n  \
         reference {reference:e} | drop-without-sync then reuse {racy:e} ⇒ {}",
        if reference == racy {
            "IDENTICAL"
        } else {
            "DIVERGED — release MUST synchronize"
        }
    );
    let armed = total_ms > reuse_ms * 1.05;
    println!(
        "  (armed = the reuse happened before the kernels drained: {})",
        if armed {
            "YES — work was still outstanding when the pages were recycled"
        } else {
            "NO — kernels had already drained; this run does not discriminate. Raise \
             SC15791_RACE_REPEATS / SC15791_RACE_TOKENS."
        }
    );
    assert!(
        armed,
        "the race probe did not arm ({repeats}x{tokens} tokens enqueued in {launch_ms:.1} ms, \
         drained before the {reuse_ms:.1} ms reuse) — an IDENTICAL result here would prove nothing, \
         so fail loudly rather than bank a vacuous green"
    );
    assert_eq!(
        reference, racy,
        "a window dropped without a synchronize, whose pages were immediately reused by the next \
         window, changed the in-flight result — `release` must synchronize on this backend"
    );

    // ── The same probe, but the reuse is a REAL host-to-device weight upload rather than a memset.
    //
    // This matters because the precedent that motivated the question — sc-12195, where FLUX.2-dev Q4
    // pixels were deterministically corrupted until `Device::synchronize()` was added at the end of
    // the text-encode phase — reused the pool via a *loader*, not an allocation. If candle's H2D copy
    // path were not ordered on the same stream as the compute, the memset arm above could pass while
    // the real thing still corrupted. So drive the actual shape: drop the window, materialize the
    // NEXT window into the freed pages, and only then read the in-flight result.
    println!("\n[sc-15791] Q3c — same probe, but the reuse is a real H2D weight materialization");
    // Baseline: what does materializing block 1 cost with an IDLE stream? Needed because the racy
    // arm's own timing is the evidence — see below.
    let idle_upload_ms = {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 1..2)?); // warm its pages
        dev.synchronize()?;
        let view = tier.open_view(&dev)?;
        let t = Instant::now();
        let b = materialize(&tier, &view, 1..2)?;
        dev.synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        drop(b);
        ms
    };
    pool.trim();
    let racy_upload = {
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
        let total = t.elapsed().as_secs_f64() * 1000.0;
        drop(next);
        drop(view2);
        // The timing IS the evidence. Materializing block 1 on an idle stream takes
        // `idle_upload_ms`; here it takes far longer, because its host-to-device copies are enqueued
        // on the SAME per-device stream that still holds seconds of outstanding compute and must wait
        // for capacity. That serialization is precisely why the stream-ordered free cannot hand these
        // pages to the copy before the matmuls reading them have retired — so the arming heuristic
        // used in Q3b ("was work still outstanding when the reuse call returned?") is inexpressible
        // here: the reuse call itself cannot return until the stream drains.
        let stall = reuse - launch;
        println!(
            "  enqueue {launch:.1} ms | next window uploaded into the freed pages by {reuse:.1} ms \
             (took {stall:.1} ms, vs {idle_upload_ms:.1} ms on an idle stream = {:.1}x) | first sync \
             {total:.1} ms",
            stall / idle_upload_ms.max(1e-3)
        );
        println!(
            "  ⇒ the H2D copies were serialized behind the outstanding compute, which is the \
             mechanism that makes the un-synchronized free safe rather than lucky"
        );
        assert!(
            stall > idle_upload_ms * 2.0,
            "expected the reuse upload to stall behind the queued compute (took {stall:.1} ms vs \
             {idle_upload_ms:.1} ms idle); without that stall this arm does not demonstrate \
             same-stream ordering"
        );
        v
    };
    println!(
        "  reference {reference:e} | drop-without-sync then H2D-reuse {racy_upload:e} ⇒ {}",
        if reference == racy_upload {
            "IDENTICAL — candle's H2D copy is stream-ordered with the compute too"
        } else {
            "DIVERGED — release MUST synchronize before the next window loads"
        }
    );
    assert_eq!(
        reference, racy_upload,
        "a real host-to-device materialization into a just-freed window's pages changed the \
         in-flight result — this is the sc-12195 shape and `release` must synchronize"
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Q4 (packed-quant, per block)
// ---------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA host with the hosted z-image q4 tier (SC15791_Q4 env)"]
fn q4_packed_quant_per_block() -> Result<()> {
    let q4 = env_path("SC15791_Q4");
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;

    // SAFETY: immutable HF-cache blob.
    let st = unsafe { MmapedSafetensors::new(&q4)? };

    println!("\n[sc-15791] Q4 — does the packed triple materialize per block, faithfully?");
    println!(
        "  block 0 carries {} packed triples + {} dense tensors, {:.1} MiB",
        tier.packed[0].len(),
        tier.dense[0].len(),
        tier.bytes[0] as f64 / MIB
    );

    // Every packed base is three INDEPENDENT file entries — the property that makes per-block
    // materialization per-tensor rather than a slice out of a larger packed buffer (the MLX spike's
    // identified risk, confirmed here on the candle side against the same file).
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

    // Faithfulness: a block materialized inside a WINDOW must be bit-identical to the same weights
    // loaded resident. Compared against the dense MLX affine grid, which is the ground truth both
    // paths claim to represent.
    let view = tier.open_view(&dev)?;
    let blocks = materialize(&tier, &view, 0..1)?;
    let mut worst = 0f32;
    for (base, ql, in_dim) in &blocks[0].lins {
        assert!(
            ql.is_quantized(),
            "{base}: `.scales` present but the loader took the dense path — the window would be \
             streaming dequantized weights and the whole rung-4 saving would be fictional"
        );
        let wq = st.load(&format!("{base}.weight"), &Device::Cpu)?;
        let scales = st.load(&format!("{base}.scales"), &Device::Cpu)?;
        let biases = st.load(&format!("{base}.biases"), &Device::Cpu)?;
        let grid = dequant_mlx_q4_reference(&wq, &scales, &biases)?.to_device(&dev)?;
        let dense = candle_gen::quant::QLinear::Dense(candle_gen::quant::DenseLinear::Linear(
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
        "a block materialized inside a window must be bit-identical to its resident twin"
    );

    // Resident cost of one packed block, measured rather than computed.
    pool.reset_high();
    let before = pool.used();
    let view2 = tier.open_view(&dev)?;
    let b2 = materialize(&tier, &view2, 0..1)?;
    dev.synchronize()?;
    let resident = pool.used().saturating_sub(before);
    println!(
        "  one packed block resident on device: {:.1} MiB ({:.1} MiB of it the dense norm half) vs \
         {:.1} MiB on disk | all {} blocks resident would be {:.2} GiB",
        resident as f64 / MIB,
        b2[0].dense_bytes() as f64 / MIB,
        tier.bytes[0] as f64 / MIB,
        tier.n_blocks(),
        resident as f64 * tier.n_blocks() as f64 / GIB,
    );
    drop(b2);
    drop(view2);
    drop(blocks);
    drop(view);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// The small-card disclosure (opt-in)
// ---------------------------------------------------------------------------------------------------

/// The constrained-budget arm. This host is a 97.9 GiB RTX PRO 6000, so an unconstrained number here
/// says nothing about the 8 GB card rung 4 exists for. A balloon shrinks the driver-visible free pool
/// to the target before the sweep runs.
///
/// **Read the caveat in the output.** On Windows/WDDM this box SPILLS to shared system memory rather
/// than hard-OOMing, so "it completed" is not proof of fit; the honest signals are (a) the live peak
/// being unchanged and (b) wall time not inflating, which is what a spill would show.
#[test]
#[ignore = "opt-in: needs SC15791_Q4 and SC15791_TARGET_FREE_GIB"]
fn constrained_budget_sweep() -> Result<()> {
    let q4 = env_path("SC15791_Q4");
    let target_free = env_usize("SC15791_TARGET_FREE_GIB", 0);
    assert!(
        target_free > 0,
        "set SC15791_TARGET_FREE_GIB to the free VRAM to emulate (e.g. 8 for an 8 GiB card)"
    );
    let dev = Device::new_cuda(0)?;
    let pool = pool::Pool::open(0).expect("default mempool");
    disclose_host(&pool);
    let tier = Tier::open(&q4, MLX_GROUP_SIZE)?;

    let sweep = |label: &str| -> Result<Vec<(usize, f64, f64)>> {
        let mut out = Vec::new();
        for window in [1usize, 2, 4] {
            pool.reset_high();
            let base_used = pool.used();
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
            let peak = pool.used_high().saturating_sub(base_used) as f64 / MIB;
            println!("  [{label}] window {window}: step {secs:.3} s | peak live {peak:.1} MiB");
            out.push((window, secs, peak));
        }
        Ok(out)
    };

    // Warm, then take the UNCONSTRAINED baseline in this same process — comparing against a figure
    // from a different run would confound the spill signal with run-to-run variance.
    {
        let view = tier.open_view(&dev)?;
        drop(materialize(&tier, &view, 0..tier.n_blocks())?);
        dev.synchronize()?;
    }
    pool.trim();
    println!("\n[sc-15791] BASELINE (unconstrained)");
    let baseline = sweep("free")?;
    pool.trim();

    // Balloon ADAPTIVELY to the target: a fixed GiB count overshoots, because the pool reserves in
    // granular chunks and the CUDA context itself is not free. The first run of this arm asked for
    // 86 GiB on a 93.9 GiB-free card and left 0.00 GiB — which measures spilling, not an 8 GiB card.
    let mut balloon: Vec<Tensor> = Vec::new();
    let chunk_elems = 256usize * 1024 * 1024; // 1 GiB of f32
    loop {
        let (free, _) = pool::mem_info();
        if free as f64 / GIB <= target_free as f64 {
            break;
        }
        match Tensor::zeros(chunk_elems, DType::F32, &dev) {
            Ok(t) => balloon.push(t),
            Err(_) => break, // the driver refused before we hit the target — stop and report
        }
        dev.synchronize()?;
    }
    let (free_under, total) = pool::mem_info();
    println!(
        "\n[sc-15791] CONSTRAINED: {} GiB balloon held; driver free now {:.2} GiB of {:.1} GiB total \
         (target {target_free} GiB)",
        balloon.len(),
        free_under as f64 / GIB,
        total as f64 / GIB
    );
    let constrained = sweep("constrained")?;

    println!("\n[sc-15791] SMALL-CARD VERDICT");
    let mut spilled = false;
    for ((w, b_s, b_p), (_, c_s, c_p)) in baseline.iter().zip(&constrained) {
        let slow = c_s / b_s;
        if slow > 1.25 {
            spilled = true;
        }
        println!(
            "  window {w}: {b_s:.3} s → {c_s:.3} s ({slow:.2}x) | peak {b_p:.1} → {c_p:.1} MiB"
        );
    }
    println!(
        "  CAVEAT (load-bearing): this host does NOT hard-OOM — Windows/WDDM spills to shared system \
         memory, so completion alone is never proof of fit. The signals that do mean something are \
         an UNCHANGED live peak and an UNCHANGED wall time; a wall-time inflation IS the spill.\n  \
         ⇒ {}",
        if spilled {
            "wall time inflated >1.25x under the constraint — this run SPILLED, so small-card \
             behaviour remains UNVERIFIED at this budget"
        } else {
            "live peak and wall time both held — the window bound is real at this budget, on a card \
             that still cannot be made to hard-OOM"
        }
    );
    drop(balloon);
    Ok(())
}
