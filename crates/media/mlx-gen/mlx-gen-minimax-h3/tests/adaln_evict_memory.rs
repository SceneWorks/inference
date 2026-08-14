//! sc-17145: **the AdaLN weights really leave device memory**, measured — not inferred from a
//! dropped Rust handle.
//!
//! # Why this is its own test binary, with one `#[test]` in it
//!
//! MLX's allocator counters are **process-global**. `cargo test` runs the tests inside one binary
//! on parallel worker threads, so any other test allocating concurrently corrupts the reading.
//! Cargo gives each `tests/*.rs` its own process, and this file holds a single test that runs both
//! arms sequentially, so the measurement is uncontended without needing `--test-threads=1`.
//!
//! # What each counter can and cannot prove
//!
//! | counter | proves | does NOT prove |
//! |---|---|---|
//! | `get_active_memory` falling by the projections' bytes | nothing live — no Rust handle and no pending graph node — references those buffers | that the memory left the process: MLX's allocator keeps freed buffers in its own cache, so active can fall while RSS does not move |
//! | `get_cache_memory` still ~0 after `clear_cache` | the freed bytes went back to the system allocator instead of migrating active → cache | anything about *which* buffers were released |
//! | `get_peak_memory` after `reset_peak_memory` | the high-water of the post-eviction phase | nothing about the cache, which it does not count at all |
//!
//! The spike's torch/MPS result — a 62 GB conditioner that `.to("cpu")`, `del`, `gc.collect()` and
//! `torch.mps.empty_cache()` all failed to release inside a process — is why this is measured on
//! MLX rather than assumed from the API's shape.
//!
//! # The control arm
//!
//! The second half of the test performs the eviction **without** forcing evaluation first, and
//! shows the release does not happen: the modulation is still an un-evaluated graph node holding a
//! reference to `adaln_proj.weight`, so dropping the projection frees nothing. That is the trap
//! `AdaLnCache::precompute` exists to close, demonstrated rather than described.
//!
//! Dimensions here are synthetic and deliberately AdaLN-heavy — the real ratio, at a size that runs
//! in a second. `tests/adaln_evict_real_weights.rs` is the same measurement on the real 62 GB
//! `transformer/`.
//!
//! # The two phase peaks, and the envelope between them (sc-19449)
//!
//! Both peaks used to be measured, printed, and asserted against nothing. They are now pinned by
//! [`common::assert_adaln_phase_envelope`], which both this file and the real-weight file call, so
//! the synthetic and the real measurement cannot drift into pinning different things — see that
//! function for the direction, its derivation, and why it is an envelope rather than a law.
//!
//! Arm 1b holds a ballast of exactly half the evicted bytes live across the *same* forward on the
//! *same* stack, and shows the gap falls by it. That turns the envelope's geometry dependence from
//! an argument about render-geometry numbers into a measurement, and it is also what keeps the
//! `3/4` bound honest: a denoise working set of half the eviction breaks it, so the bound is not a
//! constant that anything would satisfy.

mod common;

use std::collections::HashMap;

use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{AdaLnCache, AdaLnResidency, DitBlock, MiniMaxH3DitConfig, MODALITY_NUM};

use common::{AdaLnPhases, ADALN_GAP_DEN, ADALN_GAP_NUM};

/// Blocks in the synthetic stack.
const LAYERS: i32 = 8;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// An AdaLN-dominated geometry: `adaln_proj` is `[9216, 512]` — 18.0 MiB per block against ~0.9 MiB
/// of block body — mirroring the shipped model, where 26.02 GB of 62 GB is `adaln_proj`.
fn cfg() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        num_attention_heads: 2,
        attention_head_dim: 32,
        hidden_size: 512,
        num_layers: LAYERS,
        ffn_dim: 64,
        time_embed_dim: 512,
        rope_freq_dim: 4,
        ..MiniMaxH3DitConfig::default()
    }
}

/// A deterministic tensor. Host-built so the stack is reproducible and needs no GPU RNG stream.
fn tensor(shape: &[i32]) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let v: Vec<f32> = (0..n).map(|i| ((i % 977) as f32 * 0.001) - 0.4).collect();
    Array::from_slice(&v, shape)
}

/// The synthetic stack, materialized: every tensor is evaluated before the blocks are built, so the
/// "before" reading counts real buffers rather than pending graph nodes.
fn stack(c: &MiniMaxH3DitConfig) -> Vec<DitBlock> {
    let mut map: HashMap<String, Array> = HashMap::new();
    for i in 0..LAYERS {
        let p = format!("transformer_blocks.{i}");
        map.insert(format!("{p}.norm1.weight"), tensor(&[c.hidden_size]));
        map.insert(format!("{p}.norm2.weight"), tensor(&[c.hidden_size]));
        for proj in ["to_q", "to_k", "to_v"] {
            map.insert(
                format!("{p}.attn.{proj}.weight"),
                tensor(&[c.inner_dim(), c.hidden_size]),
            );
        }
        map.insert(
            format!("{p}.attn.to_out.0.weight"),
            tensor(&[c.hidden_size, c.inner_dim()]),
        );
        map.insert(
            format!("{p}.attn.norm_q.weight"),
            tensor(&[c.attention_head_dim]),
        );
        map.insert(
            format!("{p}.attn.norm_k.weight"),
            tensor(&[c.attention_head_dim]),
        );
        map.insert(
            format!("{p}.ff.net.0.proj.weight"),
            tensor(&[2 * c.ffn_dim, c.hidden_size]),
        );
        map.insert(
            format!("{p}.ff.net.2.weight"),
            tensor(&[c.hidden_size, c.ffn_dim]),
        );
        map.insert(
            format!("{p}.adaln_proj.linear.weight"),
            tensor(&[c.adaln_out_features(), c.time_embed_dim]),
        );
        map.insert(
            format!("{p}.adaln_proj.linear.bias"),
            tensor(&[c.adaln_out_features()]),
        );
    }
    let values: Vec<Array> = map.values().cloned().collect();
    mlx_rs::transforms::eval(values.iter()).expect("materialize the stack");

    let mut w = Weights::from_map(map);
    let blocks: Vec<DitBlock> = (0..LAYERS)
        .map(|i| {
            DitBlock::from_weights(
                &mut w,
                &format!("transformer_blocks.{i}"),
                c,
                Dtype::Float32,
            )
            .unwrap()
        })
        .collect();
    // Drop the loader's own references, so the blocks are the ONLY holders.
    drop(w);
    clear_cache();
    blocks
}

/// A 6-evaluation joint schedule: video and audio row classes on their own sigma shifts, the
/// conditioning rows at 0.999 and the text rows at 1.0.
fn schedule() -> mlx_gen_minimax_h3::TimestepSchedule {
    let shift = |sigma: f32, s: f32| s * sigma / (1.0 + (s - 1.0) * sigma);
    mlx_gen_minimax_h3::TimestepSchedule::new(
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

fn embed(c: &MiniMaxH3DitConfig) -> impl Fn(&[f32]) -> mlx_gen::Result<Array> + '_ {
    move |ts: &[f32]| {
        let d = c.time_embed_dim;
        let mut v = Vec::with_capacity(ts.len() * d as usize);
        for &t in ts {
            for j in 0..d {
                v.push((t * 0.37 * (j as f32 + 1.0)).sin() * 0.7);
            }
        }
        Ok(Array::from_slice(&v, &[ts.len() as i32, d]))
    }
}

/// One real block forward against the cached table, evaluated, returning the MLX peak that phase
/// reached. This is the post-eviction denoise phase in miniature.
fn one_forward(c: &MiniMaxH3DitConfig, blocks: &[DitBlock], cache: &AdaLnCache) -> usize {
    const SEQ: i32 = 48;
    let rope = mlx_gen_minimax_h3::MmRope::new(c.rope_freq_dim, c.rope_theta).unwrap();
    let pos: Vec<f32> = (0..SEQ * 3).map(|i| (i % 7) as f32).collect();
    let tables = rope
        .tables(&Array::from_slice(&pos, &[SEQ, 3]), Dtype::Float32)
        .unwrap();
    // Row classes 0..3 and modality tags 0..2, cycling — a stand-in packed sequence.
    let classes: Vec<i32> = (0..SEQ).map(|i| i % 4).collect();
    let tags: Vec<i32> = (0..SEQ).map(|i| i % MODALITY_NUM).collect();
    let idx = cache
        .schedule()
        .adaln_indices(
            0,
            &Array::from_slice(&classes, &[SEQ]),
            &Array::from_slice(&tags, &[SEQ]),
        )
        .unwrap();

    let v: Vec<f32> = (0..SEQ * c.hidden_size)
        .map(|i| (i as f32 * 0.013).sin() * 0.4)
        .collect();
    let mut x = Array::from_slice(&v, &[1, SEQ, c.hidden_size]);
    for (layer, block) in blocks.iter().enumerate() {
        x = block
            .forward(&x, cache.modulation(layer).unwrap(), &idx, &rope, &tables)
            .unwrap();
    }
    mlx_rs::transforms::eval([&x]).unwrap();
    assert!(x.sum(None).unwrap().item::<f32>().is_finite());
    get_peak_memory()
}

#[test]
fn the_evicted_adaln_weights_leave_device_memory() {
    let c = cfg();
    let per_block = (c.adaln_out_features() as usize * c.time_embed_dim as usize
        + c.adaln_out_features() as usize)
        * 4;
    let expected_release = per_block * LAYERS as usize;
    println!(
        "\n  synthetic stack: {LAYERS} blocks, adaln_proj [{}, {}] f32 = {:.1} MiB/block, \
         {:.1} MiB total",
        c.adaln_out_features(),
        c.time_embed_dim,
        mib(per_block),
        mib(expected_release)
    );

    // ── arm 1: the real path ────────────────────────────────────────────────────────────────
    let mut blocks = stack(&c);
    clear_cache();
    let active_before = get_active_memory();
    let cache_before = get_cache_memory();
    reset_peak_memory();

    let (cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        embed(&c),
    )
    .unwrap();

    let active_after = get_active_memory();
    let cache_after = get_cache_memory();
    // The transient: the precompute holds the projections AND the new table at the same time, so
    // this is the high-water of the whole precompute+evict phase, not of the denoise that follows.
    let peak_precompute = get_peak_memory();

    // The number the acceptance criterion is about — the high-water of the phase that runs AFTER
    // the eviction. Measured over a real block forward against the cached table.
    reset_peak_memory();
    let peak_denoise = one_forward(&c, &blocks, &cache);

    println!(
        "  active {:.1} -> {:.1} MiB (-{:.1}), mlx cache {:.1} -> {:.1} MiB",
        mib(active_before),
        mib(active_after),
        mib(active_before.saturating_sub(active_after)),
        mib(cache_before),
        mib(cache_after),
    );
    println!(
        "  peak: precompute+evict {:.1} MiB, post-evict denoise {:.1} MiB, pre-evict resident \
         {:.1} MiB",
        mib(peak_precompute),
        mib(peak_denoise),
        mib(active_before)
    );
    println!(
        "  cache table {:.2} MiB for {} rows x {} hidden x 6 x {LAYERS} layers",
        mib(cache.bytes()),
        cache.schedule().modulation_rows(),
        c.hidden_size
    );

    assert_eq!(released, expected_release, "released bytes");

    // (a) ACTIVE: nothing live references the projections any more. Alone this would not
    //     distinguish "released" from "moved into MLX's cache".
    let freed = active_before.saturating_sub(active_after);
    assert!(
        freed as f64 >= 0.90 * released as f64,
        "active memory fell by only {:.1} MiB of the {:.1} MiB evicted — the projections are \
         still referenced",
        mib(freed),
        mib(released)
    );

    // (b) THE DRAIN: and they did not merely migrate into the allocator cache. Without the
    //     `clear_cache` inside `precompute_and_evict`, this is where the bytes would be sitting.
    assert!(
        cache_after <= released / 8,
        "MLX's allocator cache holds {:.1} MiB after the drain — the {:.1} MiB moved from active \
         into cache rather than being released",
        mib(cache_after),
        mib(released)
    );

    // (c) TOTAL FOOTPRINT: active + cache, the accounting that actually says "the memory went
    //     away". The cache table is the only thing that grew.
    let before_total = active_before + cache_before;
    let after_total = active_after + cache_after;
    assert!(
        after_total + (released * 9 / 10) <= before_total + cache.bytes(),
        "total MLX footprint went {:.1} -> {:.1} MiB against a {:.1} MiB eviction and a {:.1} MiB \
         cache table",
        mib(before_total),
        mib(after_total),
        mib(released),
        mib(cache.bytes())
    );

    // (d) THE PRODUCT CLAIM: the peak of the phase that runs after the eviction is bounded well
    //     below what was resident before it — "denoise-resident drops", in measurable form.
    assert!(
        (peak_denoise as f64) < 0.5 * active_before as f64,
        "the post-eviction denoise peak {:.1} MiB is not meaningfully below the pre-eviction \
         resident {:.1} MiB",
        mib(peak_denoise),
        mib(active_before)
    );

    // (e)/(f)/(g) THE PHASE PAIR — sc-19449. Everything above compares a peak to a *residency*;
    //     this compares the two phase peaks to each other, which is the relationship a retracted
    //     `convert.rs` claim got backwards while both of these tests were measuring it and
    //     throwing it away.
    let phases = AdaLnPhases {
        scale: "synthetic 8-block stack, f32, adaln_proj [9216, 512], denoise SEQ=48",
        active_before,
        active_after,
        peak_precompute,
        peak_denoise,
        released,
        table: cache.bytes(),
    };
    common::assert_adaln_phase_envelope(&phases);

    // ── arm 1b (envelope): the gap narrows one-for-one with the denoise phase's working set ──
    // sc-19449 AC6. `gap ≈ released − denoise_working_set` is an identity, not an inequality, so
    // the relationship is geometry- and tier-dependent and crosses at `denoise_working_set =
    // released`. Hold a ballast of exactly half the evicted bytes live across the SAME forward on
    // the SAME stack — everything else identical, so the difference is attributable — and the gap
    // has to fall by it.
    let ballast_bytes = released / 2;
    let ballast = Array::zeros::<f32>(&[(ballast_bytes / 4) as i32]).unwrap();
    mlx_rs::transforms::eval([&ballast]).unwrap();
    reset_peak_memory();
    let peak_denoise_ballasted = one_forward(&c, &blocks, &cache);
    let gap = peak_precompute.saturating_sub(peak_denoise);
    let gap_ballasted = peak_precompute.saturating_sub(peak_denoise_ballasted);
    println!(
        "  envelope: a {:.1} MiB denoise working set moves the denoise peak {:.1} -> {:.1} MiB \
         and the gap {:.1} -> {:.1} MiB",
        mib(ballast_bytes),
        mib(peak_denoise),
        mib(peak_denoise_ballasted),
        mib(gap),
        mib(gap_ballasted)
    );

    // The identity itself. Also the proof that the ballast landed at all: MLX materializes
    // `zeros` through `full`, but a lazily-broadcast one would leave this at ~0.
    let narrowed = gap.saturating_sub(gap_ballasted);
    assert!(
        narrowed >= ballast_bytes / 5 * 4,
        "a {:.1} MiB denoise working set narrowed the phase gap by only {:.1} MiB — the gap is \
         not `released − denoise_working_set`, so the envelope documented on \
         `assert_adaln_phase_envelope` is wrong",
        mib(ballast_bytes),
        mib(narrowed)
    );

    // …and therefore the `3/4` bound is not a constant that anything would satisfy: half an
    // eviction's worth of denoise working set breaks it. If this ever passes, the bound has
    // stopped measuring the phase relationship and is being met by something else.
    let want_gap = released / ADALN_GAP_DEN * ADALN_GAP_NUM;
    assert!(
        gap_ballasted < want_gap,
        "the {ADALN_GAP_NUM}/{ADALN_GAP_DEN} gap bound ({:.1} MiB) still holds with a {:.1} MiB \
         denoise working set in the way — it is not bounding the phase relationship",
        mib(want_gap),
        mib(ballast_bytes)
    );

    // The direction survives at half, because the crossing is at a FULL eviction's worth. This is
    // what says the q4 + full-render-geometry case is near the crossing rather than past it.
    assert!(
        peak_precompute > peak_denoise_ballasted,
        "the precompute peak {:.1} MiB fell to or below the ballasted denoise peak {:.1} MiB at \
         half an eviction's working set — the crossing is not where the envelope says it is",
        mib(peak_precompute),
        mib(peak_denoise_ballasted)
    );

    drop(ballast);
    drop(cache);
    drop(blocks);
    clear_cache();

    // ── arm 2 (control): the SAME eviction, without forcing evaluation first ────────────────
    // The lazy-eval trap. Every table below is an un-evaluated graph node still referencing its
    // block's `adaln_proj.weight`, so taking the projection out of the block and draining the
    // allocator releases nothing.
    let mut blocks = stack(&c);
    clear_cache();
    let active_before = get_active_memory();

    let s = schedule();
    let temb = embed(&c)(s.distinct_timesteps()).unwrap();
    let unforced: Vec<_> = blocks
        .iter()
        .map(|b| b.modulation(&temb).unwrap())
        .collect();
    // NO `eval` here — that is the whole point.
    for block in blocks.iter_mut() {
        drop(block.evict_adaln());
    }
    clear_cache();
    let active_unforced = get_active_memory();
    let freed_unforced = active_before.saturating_sub(active_unforced);

    println!(
        "  control (no eval before the evict): active {:.1} -> {:.1} MiB, freed {:.1} MiB of \
         {:.1} MiB",
        mib(active_before),
        mib(active_unforced),
        mib(freed_unforced),
        mib(expected_release)
    );
    assert!(
        (freed_unforced as f64) < 0.5 * expected_release as f64,
        "the un-forced eviction freed {:.1} MiB of {:.1} MiB — if this now releases, the lazy-eval \
         ordering in AdaLnCache::precompute is no longer load-bearing and its documentation is \
         wrong",
        mib(freed_unforced),
        mib(expected_release)
    );

    // …and the tables are still correct once something does force them, so this is a MEMORY trap
    // and not a correctness one — which is exactly why it can ship unnoticed.
    let tables: Vec<&Array> = unforced.iter().flat_map(|m| m.tables()).collect();
    mlx_rs::transforms::eval(tables).unwrap();
    assert_eq!(
        unforced[0].scale_msa.shape(),
        &[s.num_distinct_timesteps() * MODALITY_NUM, c.hidden_size]
    );
    assert!(unforced[0]
        .scale_msa
        .sum(None)
        .unwrap()
        .item::<f32>()
        .is_finite());
}
