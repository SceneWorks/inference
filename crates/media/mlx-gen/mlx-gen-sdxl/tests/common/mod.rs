//! Shared test helpers (F-118). The `snapshot()` / `snapshot_opt()` resolvers were copy-pasted
//! byte-for-byte into ~20 of this crate's `#[ignore]`d real-weight test files; they live here now,
//! keyed off the single canonical `SDXL_SNAPSHOT` env var (falling back to the standard HF cache
//! `stabilityai/stable-diffusion-xl-base-1.0` snapshots dir). Behavior is byte-identical to the
//! per-file copies — the panicking `snapshot()` and the `Option`-returning `snapshot_opt()` are the
//! two shapes that existed inline.
//!
//! `tests/common/mod.rs` is compiled once into each integration-test binary that declares `mod
//! common;`; every binary uses only a subset, so `#![allow(dead_code)]` suppresses the otherwise
//! unavoidable dead-code warnings under `-D warnings`.
#![allow(dead_code)]

use std::path::PathBuf;

/// The `SDXL_SNAPSHOT` override, else the newest HF-cache snapshot dir; **panics** if neither exists.
/// The form used by the parity/real-weight tests that unconditionally need weights.
pub fn snapshot() -> PathBuf {
    let p = std::env::var("SDXL_SNAPSHOT").unwrap_or_else(|_| panic!("set SDXL_SNAPSHOT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

/// The `SDXL_SNAPSHOT` override, else the newest HF-cache snapshot dir, or `None` if unavailable —
/// the graceful `skip:` form used by the smoke/perf tests that self-skip when weights are absent.
pub fn snapshot_opt() -> Option<PathBuf> {
    let p = std::env::var("SDXL_SNAPSHOT").ok()?;
    Some(PathBuf::from(p))
}
