//! sc-23108: exercise AdaLN precomputation with newly loaded, unevaluated real weights.
//!
//! The older eviction gate evaluates the entire source model before precomputation, which hides
//! command buffers waiting on a long sequence of lazy file reads. This gate deliberately starts
//! at the production load boundary. "Cold" means fresh arrays, not an asserted cold OS file cache.
//!
//! Run separately for bf16/q4/q8 with `MINIMAX_H3_DIT` pointing at that tier's transformer directory
//! and `MINIMAX_H3_ADAPTER_FILE` pointing at the Turbo 4-step 768p file. No assets are downloaded.

use std::{path::PathBuf, time::Instant};

use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen_minimax_h3::{
    adapters::apply_minimax_h3_adapters,
    denoise::{adaln_schedule, JointSchedule},
    AdaLnCache, AdaLnResidency, MiniMaxH3Dit,
};
use mlx_rs::{Array, Dtype};

#[test]
#[ignore = "requires real H3 transformer and Turbo adapter files plus exclusive Metal access"]
fn fresh_adaln_tables_match_warm_projection_with_turbo() {
    let root = PathBuf::from(std::env::var("MINIMAX_H3_DIT").expect("MINIMAX_H3_DIT"));
    let adapter =
        PathBuf::from(std::env::var("MINIMAX_H3_ADAPTER_FILE").expect("MINIMAX_H3_ADAPTER_FILE"));
    let started = Instant::now();
    let mut dit = MiniMaxH3Dit::load_dir(root, Dtype::Bfloat16).expect("fresh lazy transformer");
    let report = apply_minimax_h3_adapters(
        &mut dit,
        &[AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)],
    )
    .expect("install the production Turbo adapter");
    assert_eq!(report.applied, 312);
    assert!(report.unmatched_paths.is_empty());
    assert_eq!(dit.num_layers(), 50);

    let schedule = adaln_schedule(&JointSchedule::with_shifts(4, 6.0, 3.0).unwrap()).unwrap();
    let cache = AdaLnCache::precompute(dit.blocks(), schedule.clone(), |ts| {
        dit.projections().time_embedder.forward(ts)
    })
    .expect("cold AdaLN precompute must complete without a Metal timeout");
    assert_eq!(cache.num_layers(), 50);
    assert!(cache.is_current_for(&schedule));
    let temb = dit
        .projections()
        .time_embedder
        .forward(schedule.distinct_timesteps())
        .unwrap();
    let mut checked = 0;
    for (index, block) in dit.blocks().iter().enumerate() {
        let reference = block.modulation(&temb).unwrap();
        for (got, expected) in cache
            .modulation(index)
            .unwrap()
            .tables()
            .zip(reference.tables())
        {
            assert_eq!(got.shape(), expected.shape());
            let finite: bool = got.is_finite().unwrap().all(None).unwrap().item();
            assert!(finite, "layer {index}: non-finite modulation");
            let delta: Array = got
                .subtract(expected)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap();
            assert_eq!(delta.item::<f32>(), 0.0, "layer {index}: cold/warm drift");
            checked += 1;
        }
    }
    assert_eq!(checked, 300);
    let mut blocks = dit.blocks().to_vec();
    drop(dit);
    let projection_bytes: usize = blocks
        .iter()
        .map(|block| block.adaln_proj().unwrap().nbytes())
        .sum();
    let before = mlx_rs::memory::get_active_memory();
    let (evicted_cache, released) = AdaLnCache::precompute_and_evict(
        &mut blocks,
        schedule,
        AdaLnResidency::PrecomputeAndEvict,
        |_| Ok(temb.clone()),
    )
    .expect("warm precompute and eviction");
    let after = mlx_rs::memory::get_active_memory();
    assert_eq!(released, projection_bytes);
    assert_eq!(evicted_cache.bytes(), cache.bytes());
    assert!(blocks.iter().all(|block| block.adaln_proj().is_none()));
    // Array nbytes excludes Metal allocation rounding. Allow 1% for the second cache and its
    // allocation overhead, still less than one of the 50 equal-sized projections (2%).
    assert!(
        before.saturating_sub(after) >= released * 99 / 100,
        "projections stayed live: before={before} after={after} released={released}"
    );
    assert!(mlx_rs::memory::get_cache_memory() < released / 100);
    println!(
        "SC23108 cold/warm AdaLN: {checked} finite matching tables; released_bytes={released}; \
         active_before={before}; active_after={after}; elapsed={:?}; peak_bytes={}",
        started.elapsed(),
        mlx_rs::memory::get_peak_memory()
    );
}
