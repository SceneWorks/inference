//! sc-17145: the **26.02 GB** AdaLN eviction, measured on the real `MiniMaxAI/MiniMax-H3`
//! `transformer/` — 50 blocks, `adaln_proj [96768, 2688]` each, bf16.
//!
//! `#[ignore]`d and env-gated on `MINIMAX_H3_SNAPSHOT`; needs ~62 GB of unified memory and Metal.
//!
//! ```text
//! MINIMAX_H3_SNAPSHOT=/path/to/MiniMaxAI--MiniMax-H3/snapshot \
//!   cargo test --release -p mlx-gen-minimax-h3 --test adaln_evict_real_weights \
//!   -- --ignored --nocapture
//! ```
//!
//! The path is supplied, never derived: inference does not resolve artifact locations itself
//! (epic 13657), and `scripts/check-workspace.py` enforces that no source here names one.
//!
//! # A skipped run must not read as a passing one
//!
//! `common::snapshot()` asserts rather than returning early, and every assertion below is on
//! evidence only a real run produces: the published `[96768, 2688]` shape, the exact 26_020_915_200
//! bytes that arithmetic predicts, and allocator counters that only move if 62 GB were actually
//! resident.
//!
//! # What is measured, and what each measurement is worth
//!
//! Both counters, because neither alone is evidence — see `dit::adaln`'s module docs. In summary:
//! `get_active_memory` falling proves nothing live references the weights, but cannot distinguish
//! "released" from "moved into MLX's allocator cache"; `get_cache_memory` staying flat after
//! `clear_cache` is what rules the second out. The peak is reported for two distinct phases,
//! because the precompute transiently holds the projections *and* the new table, while the number
//! the product cares about is the high-water of the denoise that follows.
//!
//! # The two phase peaks are compared to each other (sc-19449)
//!
//! Both were measured and printed here and asserted against nothing; assertions (a)-(d) below all
//! pin the post-evict drop against the pre-evict *resident*, which is a residency and not a phase
//! peak. So this — the one test positioned to measure the precompute/denoise relationship on real
//! weights — measured it and let it go, which is how `convert.rs` came to carry the backwards
//! claim sc-18659 retracted. [`common::assert_adaln_phase_envelope`] now pins it, shared verbatim
//! with the synthetic `tests/adaln_evict_memory.rs`; see that function for the identity both
//! bounds are stated against, and why that leaves them free of the tier, the geometry and the
//! schedule.
//!
//! The run prints one `ADALN PHASE PAIR [...]` line carrying the measured pair together with the
//! tier, the denoise geometry and the cache schedule. **That line is the record sc-19449 asks
//! for** — copy it onto the story. Measured at bf16 on the real `transformer/`: `gap` is 0.995x
//! the 26.02 GB evicted and the precompute's own transient 1.331x the table it retains.

mod common;

use std::time::Instant;

use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{
    AdaLnCache, AdaLnResidency, DitBlock, MiniMaxH3DitConfig, MmRope, TimestepSchedule,
    MODALITY_NUM,
};

use common::{snapshot, AdaLnPhases};

/// `50 × (96768·2688 + 96768) × 2 B` — the whole point of the story, as an arithmetic constant the
/// run has to reproduce from the loaded tensors.
const EXPECTED_EVICTED_BYTES: usize = 26_020_915_200;

/// Model evaluations in the measured schedule. Small on purpose: the eviction is what is being
/// measured, and the table's size is a schedule property already pinned by `adaln_cache.rs`.
const EVALS: usize = 8;

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

/// `σ' = s·σ / (1 + (s−1)·σ)`, video 12.0 / audio 3.0.
fn shift(sigma: f32, s: f32) -> f32 {
    s * sigma / (1.0 + (s - 1.0) * sigma)
}

fn schedule() -> TimestepSchedule {
    TimestepSchedule::new(
        (0..EVALS)
            .map(|i| {
                let sigma = 1.0 - (i as f32) / (EVALS as f32);
                let v = 1.0 - shift(sigma, 12.0);
                vec![v, 1.0 - shift(sigma, 3.0), v.max(0.999), 1.0]
            })
            .collect(),
    )
    .unwrap()
}

/// A stand-in for the timestep MLP (sc-17147). f32, as `_keep_in_fp32_modules` keeps the real one —
/// `AdaLnProjection::forward` runs the activation at this precision and only then casts down.
fn embed(cfg: &MiniMaxH3DitConfig) -> impl Fn(&[f32]) -> mlx_gen::Result<Array> + '_ {
    move |ts: &[f32]| {
        let d = cfg.time_embed_dim;
        let mut v = Vec::with_capacity(ts.len() * d as usize);
        for &t in ts {
            for j in 0..d {
                v.push((t * 0.37 * (j as f32 + 1.0)).sin() * 0.7);
            }
        }
        Ok(Array::from_slice(&v, &[ts.len() as i32, d]))
    }
}

#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~62 GB resident"]
fn real_weight_adaln_evict_releases_26_gb() {
    let root = snapshot();
    let dir = root.join("transformer");
    let cfg = MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg, MiniMaxH3DitConfig::default());
    assert_eq!(cfg.adaln_out_features(), 96_768);
    assert_eq!(cfg.num_layers, 50);

    // ── load the whole 62 GB component and MATERIALIZE it ───────────────────────────────────
    // MLX's safetensors load is lazy. Without forcing evaluation the "before" reading would be
    // near zero and the eviction would look free — the same lazy-eval trap the precompute has to
    // close, one level up.
    let started = Instant::now();
    let mut w = Weights::from_dir(&dir).unwrap();
    let keys: Vec<String> = w.keys().map(str::to_owned).collect();
    assert_eq!(
        keys.len(),
        638,
        "the published transformer/ tensor set changed"
    );
    let all: Vec<Array> = keys
        .iter()
        .map(|k| w.get(k).expect("key from keys()").clone())
        .collect();
    mlx_rs::transforms::eval(all.iter()).unwrap();
    drop(all);
    let load_s = started.elapsed().as_secs_f64();

    let blocks_started = Instant::now();
    let mut blocks: Vec<DitBlock> = (0..cfg.num_layers)
        .map(|i| {
            DitBlock::from_weights(
                &mut w,
                &format!("transformer_blocks.{i}"),
                &cfg,
                Dtype::Bfloat16,
            )
            .unwrap()
        })
        .collect();
    let build_s = blocks_started.elapsed().as_secs_f64();

    // The published shape and dtype, read off the real tensors rather than assumed.
    let proj = blocks[0]
        .adaln_proj()
        .expect("block 0 holds its projection");
    assert_eq!(proj.out_features(), 96_768);
    assert_eq!(proj.time_embed_dim(), 2_688);
    let per_block = proj.nbytes();
    assert_eq!(
        per_block * cfg.num_layers as usize,
        EXPECTED_EVICTED_BYTES,
        "the 50-block adaln footprint is not the 26.02 GB the story is sized on"
    );

    // Release the loader's references so the blocks are the only holders, then settle.
    drop(w);
    clear_cache();
    let active_before = get_active_memory();
    let cache_before = get_cache_memory();
    println!(
        "\n  loaded transformer/ in {load_s:.1}s ({} tensors), built 50 blocks in {build_s:.1}s\n  \
         resident before the evict: active {:.2} GB, mlx cache {:.2} GB",
        keys.len(),
        gb(active_before),
        gb(cache_before)
    );
    assert!(
        active_before > 30_000_000_000,
        "only {:.2} GB active — the 62 GB component did not actually materialize, so nothing \
         below is a measurement",
        gb(active_before)
    );

    // ── precompute + evict ──────────────────────────────────────────────────────────────────
    reset_peak_memory();
    let evict_started = Instant::now();
    let (cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule(),
        AdaLnResidency::PrecomputeAndEvict,
        embed(&cfg),
    )
    .unwrap();
    let evict_s = evict_started.elapsed().as_secs_f64();
    let peak_precompute = get_peak_memory();
    let active_after = get_active_memory();
    let cache_after = get_cache_memory();

    // ── the post-eviction denoise phase ─────────────────────────────────────────────────────
    reset_peak_memory();
    let forward_started = Instant::now();
    one_forward(&cfg, &blocks, &cache);
    let forward_s = forward_started.elapsed().as_secs_f64();
    let peak_denoise = get_peak_memory();

    println!(
        "  precompute+evict in {evict_s:.1}s: released {:.3} GB, cache table {:.3} GB \
         ({} rows x {} hidden x 6 x 50)",
        gb(released),
        gb(cache.bytes()),
        cache.schedule().modulation_rows(),
        cfg.hidden_size
    );
    println!(
        "  active {:.2} -> {:.2} GB (-{:.2}), mlx cache {:.2} -> {:.2} GB",
        gb(active_before),
        gb(active_after),
        gb(active_before.saturating_sub(active_after)),
        gb(cache_before),
        gb(cache_after)
    );
    println!(
        "  peak: precompute+evict {:.2} GB, post-evict block forward {:.2} GB ({forward_s:.1}s), \
         pre-evict resident {:.2} GB",
        gb(peak_precompute),
        gb(peak_denoise),
        gb(active_before)
    );

    assert_eq!(released, EXPECTED_EVICTED_BYTES);

    // (a) ACTIVE — nothing live references the projections. Cannot distinguish released from
    //     cached on its own.
    let freed = active_before.saturating_sub(active_after);
    assert!(
        freed as f64 >= 0.95 * released as f64,
        "active memory fell by only {:.2} GB of the {:.2} GB evicted",
        gb(freed),
        gb(released)
    );

    // (b) THE DRAIN — and they did not migrate into MLX's allocator cache instead.
    assert!(
        cache_after <= released / 10,
        "MLX's allocator cache holds {:.2} GB after the drain — the bytes moved from active into \
         cache rather than being released",
        gb(cache_after)
    );

    // (c) TOTAL FOOTPRINT — active + cache, the accounting that says "the memory went away".
    assert!(
        active_after + cache_after + (released * 95 / 100)
            <= active_before + cache_before + cache.bytes(),
        "total MLX footprint {:.2} -> {:.2} GB against a {:.2} GB eviction",
        gb(active_before + cache_before),
        gb(active_after + cache_after),
        gb(released)
    );

    // (d) THE PRODUCT CLAIM — denoise-resident really is ~26 GB smaller.
    assert!(
        peak_denoise + released / 2 < active_before,
        "the post-eviction denoise peak {:.2} GB is not below the pre-eviction resident {:.2} GB \
         by anything like the {:.2} GB evicted",
        gb(peak_denoise),
        gb(active_before),
        gb(released)
    );

    // (e)/(f)/(g) THE PHASE PAIR — sc-19449. (a)-(d) all compare something to a *residency*; this
    //     compares the two phase peaks to each other. The label names the tier, the denoise
    //     geometry and the cache schedule **separately**, because they are three different things:
    //     the denoise measurement is one block's forward at SEQ=512 (`one_forward` below), while
    //     the `EVALS`-evaluation schedule is a property of the cache the forward reads, not of the
    //     forward. The bounds themselves no longer depend on any of the three — see
    //     `assert_adaln_phase_envelope` — but the record has to say what was measured.
    common::assert_adaln_phase_envelope(&AdaLnPhases {
        scale: "real transformer/, bf16, adaln_proj [96768, 2688] x 50, denoise: 1 block @ \
                SEQ=512, cache: 8-evaluation schedule",
        active_before,
        active_after,
        peak_precompute,
        peak_denoise,
        released,
        table: cache.bytes(),
    });

    println!(
        "\n  REAL-WEIGHT ADALN EVICT: 50 x adaln_proj [96768, 2688] bf16 = {:.2} GB released; \
         denoise-resident {:.2} -> {:.2} GB; replaced by a {:.3} GB modulation table over {} \
         distinct timesteps of an {EVALS}-evaluation schedule.\n",
        gb(released),
        gb(active_before),
        gb(peak_denoise),
        gb(cache.bytes()),
        cache.schedule().num_distinct_timesteps()
    );
}

/// One block forward at the published geometry against the cached table, evaluated.
fn one_forward(cfg: &MiniMaxH3DitConfig, blocks: &[DitBlock], cache: &AdaLnCache) {
    const SEQ: i32 = 512;
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let pos: Vec<f32> = (0..SEQ * 3).map(|i| (i % 13) as f32).collect();
    let tables = rope.tables(&Array::from_slice(&pos, &[SEQ, 3])).unwrap();
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

    let v: Vec<f32> = (0..SEQ * cfg.hidden_size)
        .map(|i| (i as f32 * 0.013).sin() * 0.4)
        .collect();
    let x = Array::from_slice(&v, &[1, SEQ, cfg.hidden_size])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let out = blocks[0]
        .forward(&x, cache.modulation(0).unwrap(), &idx, &rope, &tables)
        .unwrap();
    mlx_rs::transforms::eval([&out]).unwrap();
    let total: f32 = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .sum(None)
        .unwrap()
        .item();
    assert!(
        total.is_finite(),
        "the post-eviction forward produced a non-finite output"
    );
    assert_eq!(out.shape(), &[1, SEQ, cfg.hidden_size]);
}
