//! sc-17155: **the AdaLN weights really leave memory on candle** — measured, not inferred from a
//! dropped handle.
//!
//! # Why this is its own test binary, with one `#[test]` in it
//!
//! The measurement is a process-global counter. `cargo test` runs a binary's tests on parallel
//! worker threads, so any other test allocating concurrently corrupts the reading. Cargo gives each
//! `tests/*.rs` its own process, and this file holds a **single** test that runs every arm
//! sequentially, so the measurement is uncontended without needing `--test-threads=1`.
//!
//! # What is measured, and why it is not what the MLX lane measures
//!
//! sc-17145 measured MLX's `get_active_memory` / `get_cache_memory` and had to defeat two MLX
//! properties: lazy evaluation (a pending graph node keeps the weight alive) and an allocator cache
//! (freed buffers migrate active → cache rather than leaving). **Neither exists here.** candle is
//! eager, so there is no graph node; and candle keeps no user-visible allocator cache of its own.
//!
//! Carrying the MLX shape across would have produced ceremony that measures nothing. What candle is
//! actually exposed to is different, and this file measures *that*:
//!
//! | device | what actually releases | what proves it here |
//! |---|---|---|
//! | **CPU** | the last `Tensor` handle dropping — candle's CPU storage is a plain Rust allocation | a counting [`#[global_allocator]`](std::alloc::GlobalAlloc). **Exact**: it counts the bytes the allocator was asked for and gave back, with no sampling window, so it cannot miss a transient the way an RSS poll can |
//! | **CUDA** | the drop **plus a `Device::synchronize`** — candle allocates from the stream-ordered pool, whose release threshold is 0, so pages stay charged to the process until something synchronizes | `candle_gen::cuda_mempool::MemPool`'s `used` and **`reserved`**. Written below, **not executed by any lane this story can reach** — see "What this file does NOT establish" |
//!
//! # The candle trap, which is the control arm
//!
//! A candle `Tensor` is an `Arc` over its storage. `Weights::require` hands out a clone **sharing
//! that storage**, and `Tensor::to_dtype` is a no-op clone when the dtype already matches. So a
//! loader map still held anywhere keeps all 26.02 GB resident while every block cheerfully reports
//! itself evicted, `released_bytes` returns the full figure, and nothing fails.
//!
//! That is the exact analogue of MLX's lazy-eval trap — a memory defect that is invisible to every
//! correctness gate — and arm 2 below performs it deliberately and requires the release **not** to
//! happen. Without that arm, arm 1 would be satisfied by a port that never released anything in
//! production, because production is precisely where a loader map tends to outlive the load.
//!
//! # What this file does NOT establish
//!
//! Stated plainly rather than left to inference:
//!
//! * **No CUDA measurement was taken.** This Mac has no CUDA device and `cargo check --features
//!   cuda` is unreachable here (`cudarc`'s build script panics on a missing toolkit root).
//!
//!   The CUDA arm is **not** dead code, though, and it is worth being precise about which half of
//!   "declared vs reached" applies. The `CI gate`'s `windows-cuda-check` job runs
//!   `cargo test --lib --tests -p 'candle-gen*' --features cuda --no-run`, plus Clippy
//!   `--all-targets --features cuda -- -D warnings` and rustdoc, on every PR touching these paths.
//!   So this arm is **compiled, linted and documented** by CI — what it is not is **executed**:
//!   that job has no GPU and passes `--no-run`, and it is additionally `#[ignore]`d. Running it
//!   needs the self-hosted CUDA box, which is sc-17156's lane.
//! * The CUDA *recipe* — drop, then synchronize, and read `reserved` rather than `used` — is not
//!   guesswork: it is the measured result of SC-15791, recorded in `candle_gen::cuda_mempool` and
//!   `candle_gen::block_window` (`pool reserved: live 160.0 │ after drop 160.0 │ after drop+sync
//!   0.0 MiB`). What is unverified is that *this* eviction follows that established behaviour, not
//!   the behaviour itself.
//! * Nothing here is measured at the real 26.02 GB. The stack is synthetic and AdaLN-heavy at the
//!   real *ratio*; the 26_020_915_200 B figure is arithmetic over the published shapes, asserted as
//!   such in `dit::adaln`'s unit tests.
//!
//! # A measured candle property this test had to be built around
//!
//! **candle's CPU matmul takes ~48 MiB of process-lifetime scratch on first use and never returns
//! it.** Measured here while building this test: a cold process that builds the 18.98 MiB stack,
//! evicts, and then drops *everything* still holds 48.14 MiB. That is `gemm`'s thread-pool and
//! per-thread packing buffers, not a leak in this crate — it is allocated once and reused — but it
//! is far larger than the quantity under test, so a cold measurement reports the eviction as
//! freeing **0.00 MiB** while the release is in fact happening.
//!
//! [`warm_up`] therefore pays that cost before any baseline is taken. On the warm process the same
//! eviction frees **17.22 of 18.07 MiB**, the 0.85 MiB difference being exactly the newly allocated
//! cache table. Taking the baseline cold would have produced a red test for a correct port — and,
//! worse, a green one for an incorrect port under any bound loose enough to absorb 48 MiB.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;

use candle_gen_minimax_h3::{
    AdaLnCache, AdaLnResidency, DitBlock, MiniMaxH3DitConfig, MmRope, MODALITY_NUM,
};

// ---------------------------------------------------------------------------------------------
// The counting allocator
// ---------------------------------------------------------------------------------------------

/// Net live bytes the process has taken from the system allocator.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// A pass-through allocator that keeps a running total of live bytes.
///
/// This is the CPU device's memory counter, and it is **exact** in a way a sampler is not: it
/// observes every `alloc`/`dealloc` rather than polling RSS, so a buffer that is freed cannot be
/// missed and a transient cannot slip between two samples. (SC memory: RSS sampling undercounts
/// transients; a `#[global_allocator]` is the tool that does not.)
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_add(new_size, Ordering::Relaxed);
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// ---------------------------------------------------------------------------------------------
// A synthetic AdaLN-dominated stack
// ---------------------------------------------------------------------------------------------

/// Blocks in the synthetic stack.
const LAYERS: usize = 8;

/// An AdaLN-dominated geometry mirroring the shipped model's *ratio*: `adaln_proj` is `[2304, 256]`
/// — 2.25 MiB per block against ~0.13 MiB of block body — where the real model has 26.02 GB of
/// `adaln_proj` in ~62 GB.
fn cfg() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        num_attention_heads: 2,
        attention_head_dim: 16,
        hidden_size: 128,
        num_layers: LAYERS,
        ffn_dim: 32,
        time_embed_dim: 256,
        rope_freq_dim: 2,
        ..MiniMaxH3DitConfig::default()
    }
}

/// A deterministic tensor, host-built so the stack is reproducible.
fn tensor(shape: &[usize], device: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| ((i % 977) as f32 * 0.001) - 0.4).collect();
    Tensor::from_vec(v, shape, device).unwrap()
}

fn stack_map(c: &MiniMaxH3DitConfig, device: &Device) -> HashMap<String, Tensor> {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for i in 0..LAYERS {
        let p = format!("transformer_blocks.{i}");
        map.insert(
            format!("{p}.norm1.weight"),
            tensor(&[c.hidden_size], device),
        );
        map.insert(
            format!("{p}.norm2.weight"),
            tensor(&[c.hidden_size], device),
        );
        for proj in ["to_q", "to_k", "to_v"] {
            map.insert(
                format!("{p}.attn.{proj}.weight"),
                tensor(&[c.inner_dim(), c.hidden_size], device),
            );
        }
        map.insert(
            format!("{p}.attn.to_out.0.weight"),
            tensor(&[c.hidden_size, c.inner_dim()], device),
        );
        map.insert(
            format!("{p}.attn.norm_q.weight"),
            tensor(&[c.attention_head_dim], device),
        );
        map.insert(
            format!("{p}.attn.norm_k.weight"),
            tensor(&[c.attention_head_dim], device),
        );
        map.insert(
            format!("{p}.ff.net.0.proj.weight"),
            tensor(&[2 * c.ffn_dim, c.hidden_size], device),
        );
        map.insert(
            format!("{p}.ff.net.2.weight"),
            tensor(&[c.hidden_size, c.ffn_dim], device),
        );
        map.insert(
            format!("{p}.adaln_proj.linear.weight"),
            tensor(&[c.adaln_out_features(), c.time_embed_dim], device),
        );
        map.insert(
            format!("{p}.adaln_proj.linear.bias"),
            tensor(&[c.adaln_out_features()], device),
        );
    }
    map
}

/// Build the stack and **drop the loader map**, so the blocks are the only holders.
///
/// The drop is the whole point: `Weights::require` returns a storage-sharing clone, so a surviving
/// map would pin every projection. Arm 2 below is the same construction with the map deliberately
/// kept.
fn stack(c: &MiniMaxH3DitConfig, device: &Device) -> Vec<DitBlock> {
    let w = Weights::from_map(stack_map(c, device));
    let blocks: Vec<DitBlock> = (0..LAYERS)
        .map(|i| {
            DitBlock::from_weights(&w, &format!("transformer_blocks.{i}"), c, DType::F32).unwrap()
        })
        .collect();
    drop(w);
    blocks
}

/// A 6-evaluation joint schedule: video and audio row classes on their own sigma shifts, the
/// conditioning rows at 0.999 and the reference-audio rows at 1.0.
fn schedule() -> candle_gen_minimax_h3::TimestepSchedule {
    let shift = |sigma: f32, s: f32| s * sigma / (1.0 + (s - 1.0) * sigma);
    candle_gen_minimax_h3::TimestepSchedule::new(
        (0..6)
            .map(|i| {
                let sigma = 1.0 - (i as f32) / 6.0;
                let v = 1.0 - shift(sigma, 12.0);
                vec![v, 1.0 - shift(sigma, 3.0), v.max(0.999), 1.0]
            })
            .collect(),
    )
    .unwrap()
}

fn embed<'a>(
    c: &'a MiniMaxH3DitConfig,
    device: &Device,
) -> impl Fn(&[f32]) -> candle_gen::Result<Tensor> + 'a {
    let device = device.clone();
    move |ts: &[f32]| {
        let d = c.time_embed_dim;
        let mut v = Vec::with_capacity(ts.len() * d);
        for &t in ts {
            for j in 0..d {
                v.push((t * 0.37 * (j as f32 + 1.0)).sin() * 0.7);
            }
        }
        Ok(Tensor::from_vec(v, (ts.len(), d), &device)?)
    }
}

/// One real block forward against the cached table — the post-eviction denoise phase in miniature.
fn one_forward(c: &MiniMaxH3DitConfig, blocks: &[DitBlock], cache: &AdaLnCache, device: &Device) {
    const SEQ: usize = 48;
    let rope = MmRope::new(c.rope_freq_dim, c.rope_theta).unwrap();
    let pos: Vec<f32> = (0..SEQ * 3).map(|i| (i % 7) as f32).collect();
    let tables = rope
        .tables(
            &Tensor::from_vec(pos, (SEQ, 3), device).unwrap(),
            DType::F32,
        )
        .unwrap();
    let classes: Vec<u32> = (0..SEQ).map(|i| (i % 4) as u32).collect();
    let tags: Vec<u32> = (0..SEQ).map(|i| (i % MODALITY_NUM) as u32).collect();
    let idx = cache
        .schedule()
        .adaln_indices(0, &classes, &tags, device)
        .unwrap();

    let v: Vec<f32> = (0..SEQ * c.hidden_size)
        .map(|i| (i as f32 * 0.013).sin() * 0.4)
        .collect();
    let mut x = Tensor::from_vec(v, (1, SEQ, c.hidden_size), device).unwrap();
    for (layer, block) in blocks.iter().enumerate() {
        x = block
            .forward(&x, cache.modulation(layer).unwrap(), &idx, &rope, &tables)
            .unwrap();
    }
    let sum: f32 = x
        .flatten_all()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(sum.is_finite(), "the post-eviction forward produced {sum}");
}

// ---------------------------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------------------------

/// Pay candle's one-time CPU-matmul scratch before any baseline is taken.
///
/// See the module docs: `gemm` allocates ~48 MiB of thread-pool and packing buffers on first use
/// and keeps them for the life of the process. That is ~2.7× the quantity under test here, so a
/// baseline taken before it is paid makes a correct eviction read as freeing nothing.
///
/// Runs a complete build → precompute → evict → forward cycle and discards it, so every allocation
/// path the measured arms take has already been walked once.
fn warm_up(c: &MiniMaxH3DitConfig, device: &Device) {
    let mut blocks = stack(c, device);
    let (cache, _) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        device,
        embed(c, device),
    )
    .expect("warm-up eviction");
    one_forward(c, &blocks, &cache, device);
}

#[test]
fn the_evicted_adaln_weights_leave_memory() {
    let c = cfg();
    let device = Device::Cpu;

    // The one-time cost, paid and discarded before anything is measured.
    let cold = live();
    warm_up(&c, &device);
    let warm = live();
    println!(
        "\n  warm-up: process-lifetime candle/gemm scratch {:.2} MiB (cold baseline {:.2} MiB). \
         Measuring before paying this would report a correct eviction as freeing 0.00 MiB.",
        mib(warm.saturating_sub(cold)),
        mib(cold)
    );

    let per_block = (c.adaln_out_features() * c.time_embed_dim + c.adaln_out_features()) * 4;
    let expected_release = per_block * LAYERS;
    println!(
        "\n  synthetic stack: {LAYERS} blocks, adaln_proj [{}, {}] f32 = {:.2} MiB/block, \
         {:.2} MiB total",
        c.adaln_out_features(),
        c.time_embed_dim,
        mib(per_block),
        mib(expected_release)
    );

    // ── arm 1: the real path ────────────────────────────────────────────────────────────────
    let mut blocks = stack(&c, &device);
    let body_bytes: usize = blocks.iter().map(DitBlock::body_nbytes).sum();
    let live_before = live();

    let (cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        &device,
        embed(&c, &device),
    )
    .unwrap();

    let live_after = live();
    let freed = live_before.saturating_sub(live_after);

    println!(
        "  arm 1 (map dropped): live {:.2} -> {:.2} MiB, freed {:.2} MiB of the {:.2} MiB evicted; \
         cache table {:.2} MiB, block bodies {:.2} MiB",
        mib(live_before),
        mib(live_after),
        mib(freed),
        mib(released),
        mib(cache.bytes()),
        mib(body_bytes),
    );

    assert_eq!(released, expected_release, "released bytes");
    assert!(
        !blocks.iter().any(DitBlock::holds_adaln),
        "every block must have given up its projection"
    );

    // (a) THE RELEASE. The projections' bytes really went back to the system allocator. The cache
    //     table is newly allocated in the same window, so the net fall is `released - cache.bytes()`
    //     plus whatever else the precompute holds; requiring 90 % of that net figure is the gate.
    let net_expected = released.saturating_sub(cache.bytes());
    assert!(
        freed as f64 >= 0.90 * net_expected as f64,
        "live memory fell by only {:.2} MiB against a {:.2} MiB eviction less a {:.2} MiB cache \
         table — the projections are still referenced by something",
        mib(freed),
        mib(released),
        mib(cache.bytes())
    );

    // (b) THE PRODUCT CLAIM, in the form the acceptance criterion states it: what remains resident
    //     for the denoise phase is materially below what was resident before the eviction.
    //
    //     Measured **net of the warm baseline**, because the counter is process-wide and candle's
    //     50 MiB of matmul scratch is neither attributable to the model nor releasable. Comparing
    //     the gross figures would be comparing "the model plus a fixed 50 MiB" against itself and
    //     would fail for a correct port at any realistic stack size.
    let before_net = live_before.saturating_sub(warm);
    let after_net = live_after.saturating_sub(warm);
    println!(
        "  model-attributable resident: {:.2} -> {:.2} MiB ({:.1}x reduction), net of the {:.2} \
         MiB process-lifetime scratch",
        mib(before_net),
        mib(after_net),
        before_net as f64 / after_net.max(1) as f64,
        mib(warm)
    );
    assert!(
        (after_net as f64) < 0.5 * before_net as f64,
        "post-eviction model-attributable resident {:.2} MiB is not meaningfully below the \
         pre-eviction {:.2} MiB",
        mib(after_net),
        mib(before_net)
    );

    // (c) …and the post-eviction stack still runs, against the cached table, producing finite
    //     output. An eviction that broke the forward would be a memory win nobody could use.
    one_forward(&c, &blocks, &cache, &device);
    println!(
        "  post-eviction forward over {} blocks: finite; peak live during it {:.2} MiB",
        blocks.len(),
        mib(live())
    );

    drop(cache);
    drop(blocks);

    // ── arm 2 (control): the SAME eviction with the loader map still alive ───────────────────
    // The candle trap. `Weights::require` returns a storage-sharing clone, so while the map lives
    // the projections' buffers are pinned no matter what the blocks do. Every correctness gate in
    // this crate stays green through this; only a memory measurement can see it.
    let map = stack_map(&c, &device);
    let w = Weights::from_map(map);
    let mut blocks: Vec<DitBlock> = (0..LAYERS)
        .map(|i| {
            DitBlock::from_weights(&w, &format!("transformer_blocks.{i}"), &c, DType::F32).unwrap()
        })
        .collect();
    let live_before = live();
    let (cache, released_pinned) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        &device,
        embed(&c, &device),
    )
    .unwrap();
    let live_after = live();
    let freed_pinned = live_before.saturating_sub(live_after);

    println!(
        "  arm 2 (map HELD):    live {:.2} -> {:.2} MiB, freed {:.2} MiB of the {:.2} MiB the API \
         reported as evicted",
        mib(live_before),
        mib(live_after),
        mib(freed_pinned),
        mib(released_pinned),
    );

    // The API reports the same number in both arms — that is exactly the point.
    assert_eq!(
        released_pinned, expected_release,
        "`released_bytes` is the blocks' arithmetic, not a measurement: it reports the full figure \
         even when nothing was freed. That is why this file exists."
    );
    assert!(
        !blocks.iter().any(DitBlock::holds_adaln),
        "the blocks report themselves evicted in this arm too"
    );
    assert!(
        (freed_pinned as f64) < 0.5 * released_pinned as f64,
        "the pinned eviction freed {:.2} MiB of {:.2} MiB. If a surviving `Weights` map no longer \
         holds the projections resident, candle's storage sharing has changed and `dit::adaln`'s \
         documented precondition — drop the loader map before evicting — is no longer load-bearing",
        mib(freed_pinned),
        mib(released_pinned)
    );
    drop(cache);
    drop(blocks);
    drop(w);

    // ── arm 3: the cache is NUMERICALLY IDENTICAL to the naive per-step projection ───────────
    // The memory arms above would be satisfied by an eviction that cached the wrong numbers, so the
    // identity is asserted over the same synthetic stack rather than only at fixture scale
    // (`dit_io.rs::the_cached_modulation_path_reproduces_the_resident_one` is the whole-model twin).
    let blocks_resident = stack(&c, &device);
    let s = schedule();
    let temb = embed(&c, &device)(s.distinct_timesteps()).unwrap();
    let naive: Vec<_> = blocks_resident
        .iter()
        .map(|b| b.modulation(&temb).unwrap())
        .collect();

    let mut blocks_evicted = stack(&c, &device);
    let (cache, _) = AdaLnCache::precompute_and_evict(
        &mut blocks_evicted,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        &device,
        embed(&c, &device),
    )
    .unwrap();

    for (layer, want) in naive.iter().enumerate() {
        let cached = cache.modulation(layer).unwrap();
        for (i, (a, b)) in cached.tables().zip(want.tables()).enumerate() {
            let x: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let y: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(
                x, y,
                "layer {layer} table {i}: the cached modulation must be BITWISE the naive \
                 projection — both run the same AdaLnProjection over the same temb, so any \
                 difference means the cache is not the thing it claims to be"
            );
        }
    }
    println!(
        "  arm 3: {} layers x 6 tables bitwise identical between the cached and naive paths",
        LAYERS
    );

    // …and the projection is genuinely gone afterwards: asking for it is a typed error naming the
    // way out, not a panic or a silently fabricated table.
    let e = blocks_evicted[0].modulation(&temb).unwrap_err().to_string();
    assert!(e.contains("evicted"), "{e}");
    assert!(
        e.contains("Resident"),
        "the error must name the way out: {e}"
    );

    // A `Resident` load must refuse to build a cache at all — it is the choice NOT to evict, and
    // silently evicting anyway would drop weights a run-state-dependent sampler still needs.
    let mut fresh = stack(&c, &device);
    let e = AdaLnCache::precompute_and_evict(
        &mut fresh,
        schedule(),
        AdaLnResidency::Resident,
        &device,
        embed(&c, &device),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("does not precompute"), "{e}");
    assert!(
        fresh.iter().all(DitBlock::holds_adaln),
        "a refused precompute must not have evicted anything"
    );
}

// ---------------------------------------------------------------------------------------------
// The CUDA arm — written, never executed
// ---------------------------------------------------------------------------------------------

/// The same eviction measured against the **driver's** view of the stream-ordered pool.
///
/// **This has never run.** It compiles only under `--features cuda`, which no lane reachable from
/// this story builds: the dev machine has no CUDA toolkit (`cudarc`'s build script panics on a
/// missing toolkit root) and the inference `CI gate` has no CUDA job. It is committed because the
/// alternative — leaving the CUDA answer as prose — is what makes a memory claim unfalsifiable, and
/// because sc-17156 runs on the self-hosted CUDA box and can execute it as-is.
///
/// The two counters are deliberately both read, because they answer different questions and only
/// one of them is the product-relevant one:
///
/// * `used` falls on the bare drop. On its own it proves nothing — SC-15791 measured `used` at 0
///   while `nvidia-smi` had recovered **0.0 MiB**;
/// * `reserved` is what the driver has actually handed back, i.e. what another process or another
///   allocation can now take, and what every VRAM admission gate in the product reads. It falls
///   only after the synchronize `precompute_and_evict` performs.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device: compiled/linted by windows-cuda-check, never executed (sc-17155)"]
fn the_evicted_adaln_weights_leave_the_cuda_pool() {
    use candle_gen::cuda_mempool::MemPool;

    let device = Device::new_cuda(0).expect("a CUDA device");
    let pool = MemPool::device_default(0).expect("the default stream-ordered pool");
    let c = cfg();

    let mut blocks = stack(&c, &device);
    device.synchronize().expect("quiesce before measuring");
    let used_before = pool.used().expect("used");
    let reserved_before = pool.reserved().expect("reserved");

    let (cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        &device,
        embed(&c, &device),
    )
    .unwrap();

    let used_after = pool.used().expect("used");
    let reserved_after = pool.reserved().expect("reserved");
    println!(
        "  CUDA pool: used {:.2} -> {:.2} MiB, reserved {:.2} -> {:.2} MiB, against a {:.2} MiB \
         eviction and a {:.2} MiB cache table",
        mib(used_before as usize),
        mib(used_after as usize),
        mib(reserved_before as usize),
        mib(reserved_after as usize),
        mib(released),
        mib(cache.bytes()),
    );

    let net_expected = released.saturating_sub(cache.bytes());
    // USED: nothing live references the projections any more.
    assert!(
        used_before.saturating_sub(used_after) as f64 >= 0.90 * net_expected as f64,
        "pool `used` did not fall by the evicted bytes"
    );
    // RESERVED: and the driver got them back — the claim `used` alone cannot support.
    assert!(
        reserved_before.saturating_sub(reserved_after) as f64 >= 0.90 * net_expected as f64,
        "pool `reserved` did not fall: the pages are still charged to this process, so \
         `nvidia-smi` and every VRAM admission gate still see them. The synchronize in \
         `precompute_and_evict` is what is supposed to return them (SC-15791)"
    );
    one_forward(&c, &blocks, &cache, &device);
}
