//! Shared helpers for the Z-Image integration tests (F-045): the HF-snapshot discovery and the
//! relative-error metric, previously copy-pasted across many test files (and already drifting — the
//! `perf` bench wanted an `Option`-returning variant while the rest panicked). Included per test
//! binary via `mod common;`. `#![allow(dead_code)]` because no single test uses every helper.
#![allow(dead_code)]

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

/// The `Tongyi-MAI/Z-Image-Turbo` TORCH-ORIGINAL snapshot root, from `ZIMAGE_SNAPSHOT`. This is
/// the name the wired real-weight lanes export after `verify_model_snapshot.py --model
/// z-image-turbo` (memory-evidence-v1 and the MLX media conformance step), so it is pinned to that
/// artifact and must not be handed a `SceneWorks/z-image-turbo-mlx` tier dir — one env name per
/// verifier pin (sc-18213; same split sc-17284 made for Qwen with `MLX_GEN_QWEN_SNAPSHOT`).
/// Returns `None` when unset — for tests/benches that skip rather than panic.
pub fn snapshot_opt() -> Option<PathBuf> {
    let p = std::env::var("ZIMAGE_SNAPSHOT").ok()?;
    Some(PathBuf::from(p))
}

/// [`snapshot_opt`] but panicking with a clear message when no snapshot is found — for the
/// `#[ignore]`d weight-gated tests that need real weights to run at all.
pub fn snapshot() -> PathBuf {
    snapshot_opt()
        .expect("a Z-Image-Turbo snapshot (set ZIMAGE_SNAPSHOT or populate the HF hub cache)")
}

/// A `SceneWorks/z-image-turbo-mlx` RE-HOST tier dir (`…/bf16`, `…/q8`, `…/q4` — or, for the
/// tests that sweep tiers, the multi-tier snapshot root), from `MLX_GEN_ZIMAGE_SNAPSHOT`. A
/// separate name from [`snapshot_opt`]'s `ZIMAGE_SNAPSHOT` because the two are verified against
/// different pins (`z-image-turbo-mlx-*` vs `z-image-turbo` in `release/real-weight-models.toml`)
/// and one name cannot be verified against two pins (sc-18213; the sc-17284 Qwen precedent).
/// Deliberately no fallback to `ZIMAGE_SNAPSHOT` — a fallback would re-merge the names.
pub fn tier_snapshot_opt() -> Option<PathBuf> {
    let p = std::env::var("MLX_GEN_ZIMAGE_SNAPSHOT").ok()?;
    Some(PathBuf::from(p))
}

/// [`tier_snapshot_opt`] but panicking with a clear message — for the `#[ignore]`d weight-gated
/// tests that need a real re-host tier on disk to run at all.
pub fn tier_snapshot() -> PathBuf {
    tier_snapshot_opt().expect(
        "a SceneWorks/z-image-turbo-mlx tier dir (set MLX_GEN_ZIMAGE_SNAPSHOT; \
         inference never self-fetches or derives a cache location, epic 13657)",
    )
}

/// `(max|a-b| / peak|b|, mean|a-b| / mean|b|)` over the full tensors (cast to f32, flattened) — the
/// peak- and mean-relative error used by the parity gates.
pub fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_diff = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_diff / peak, mean_diff / mabs)
}
