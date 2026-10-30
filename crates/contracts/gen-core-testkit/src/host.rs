//! Backend-neutral, host-side test helpers (sc-27323): comparison metrics over plain slices, box and
//! mask IoU, and env-var resolvers.
//!
//! Provider tests in both the MLX and Candle families had re-declared these with bodies that differ
//! in substance — f32 vs f64 accumulation, `+ε` vs `max(ε)` vs no guard on a denominator, floored
//! vs unfloored peaks. Each distinct computation lives here under its own name (or behind an
//! explicit [`Denom`] / floor argument the caller spells out), so a parity gate keeps computing
//! exactly the number its tolerance was tuned against. `mlx_gen::testing` and
//! `candle_gen::testkit` re-export the module, adding tensor-level wrappers that read back to host
//! and call into it.
//!
//! Every slice comparison asserts its two inputs have equal length: a zip over mismatched buffers
//! compares a prefix and proves nothing.
//!
//! Nothing here runs a model, and nothing here derives a cache location: [`env_path_or_home`] reads
//! its override first and only falls back to a caller-named `$HOME` path (epic 13657).

use std::path::PathBuf;

use gen_core::Image;

/// How a cosine's norm product is guarded before the division. The guard is part of the metric —
/// `x / (n + ε)` and `x / max(n, ε)` differ for every input, not only near zero — so callers name it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Denom {
    /// `dot / (|a|·|b|)`: no guard (a zero vector yields NaN or ±inf).
    Exact,
    /// `dot / (|a|·|b| + ε)`.
    PlusEps(f64),
    /// `dot / max(|a|·|b|, ε)`.
    FloorAt(f64),
}

impl Denom {
    /// Apply the guard in f64.
    pub fn apply_f64(self, norms: f64) -> f64 {
        match self {
            Denom::Exact => norms,
            Denom::PlusEps(eps) => norms + eps,
            Denom::FloorAt(eps) => norms.max(eps),
        }
    }

    /// Apply the guard in f32 (`ε` rounded to f32, the literal the f32 copies spelled).
    pub fn apply_f32(self, norms: f32) -> f32 {
        match self {
            Denom::Exact => norms,
            Denom::PlusEps(eps) => norms + eps as f32,
            Denom::FloorAt(eps) => norms.max(eps as f32),
        }
    }
}

fn assert_same_len(a: usize, b: usize, what: &str) {
    assert_eq!(a, b, "{what} over buffers of different lengths");
}

/// Cosine similarity accumulated **in f32**: `Σxy / guard(√Σx² · √Σy²)`, each sum a sequential f32
/// fold.
pub fn cosine_f32_sums(a: &[f32], b: &[f32], denom: Denom) -> f32 {
    assert_same_len(a.len(), b.len(), "cosine");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / denom.apply_f32(na * nb)
}

/// Cosine similarity accumulated **in f64** (each element widened before multiplying):
/// `Σxy / guard(√Σx² · √Σy²)`. Generic over the element so u8 pixels and f32 values share it.
pub fn cosine_f64_sums<T: Copy + Into<f64>>(a: &[T], b: &[T], denom: Denom) -> f64 {
    assert_same_len(a.len(), b.len(), "cosine");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y): (f64, f64) = (x.into(), y.into());
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / denom.apply_f64(na.sqrt() * nb.sqrt())
}

/// `max|a − b|` over two equal-length slices: each difference taken in f32, folded from `0` with
/// [`f32::max`] (so a NaN difference is skipped, unlike a device `max` reduction).
pub fn max_abs_diff_slices(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len(a.len(), b.len(), "max_abs_diff");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// `max|a − b|` over two equal-length u8 buffers, exact in integers (`0` when empty).
pub fn max_abs_diff_u8(a: &[u8], b: &[u8]) -> u32 {
    assert_same_len(a.len(), b.len(), "max_abs_diff_u8");
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// `max|a|` over a slice, folded from `0` with [`f32::max`] (NaN-skipping).
pub fn max_abs_slice(a: &[f32]) -> f32 {
    a.iter().fold(0f32, |m, v| m.max(v.abs()))
}

/// Peak relative error in f32: [`max_abs_diff_slices`]`(got, want)` over [`max_abs_slice`]`(want)`,
/// the denominator raised to `floor` when one is given (`None` divides by the raw peak).
pub fn peak_rel_slices(got: &[f32], want: &[f32], floor: Option<f32>) -> f32 {
    let diff = max_abs_diff_slices(got, want);
    let peak = max_abs_slice(want);
    diff / floor.map_or(peak, |f| peak.max(f))
}

/// [`peak_rel_slices`] with the division (and the floor) in **f64**: the same two f32 maxima,
/// widened, then `diff / max(peak, floor)`.
pub fn peak_rel_slices_f64(got: &[f32], want: &[f32], floor: Option<f64>) -> f64 {
    let diff = f64::from(max_abs_diff_slices(got, want));
    let peak = f64::from(max_abs_slice(want));
    diff / floor.map_or(peak, |f| peak.max(f))
}

/// Worst per-channel distance between paired u8 frames over the **first** argument's peak value
/// (`max |a − b| / max a`, both in f64); `+inf` when `a` is entirely black.
pub fn peak_rel_u8_frames(a: &[Image], b: &[Image]) -> f64 {
    assert_same_len(a.len(), b.len(), "peak_rel_u8_frames");
    let mut worst = 0.0f64;
    let mut denom = 0.0f64;
    for (fa, fb) in a.iter().zip(b) {
        assert_same_len(fa.pixels.len(), fb.pixels.len(), "peak_rel_u8_frames");
        for (&pa, &pb) in fa.pixels.iter().zip(&fb.pixels) {
            worst = worst.max((f64::from(pa) - f64::from(pb)).abs());
            denom = denom.max(f64::from(pa));
        }
    }
    if denom == 0.0 {
        f64::INFINITY
    } else {
        worst / denom
    }
}

/// Intersection-over-union of two axis-aligned `[x1, y1, x2, y2]` boxes, in f32. Disjoint boxes
/// give `0`; no guard on a zero union.
pub fn box_iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (x1, y1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x2, y2) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter)
}

/// IoU of two already-thresholded masks (pairwise over the shorter of the two iterators — callers
/// pass equal-length masks); two empty masks are a perfect match (`1.0`). The threshold that makes
/// a value "on" is the caller's, so it stays visible at the gate.
pub fn iou_bool(a: impl IntoIterator<Item = bool>, b: impl IntoIterator<Item = bool>) -> f32 {
    let (mut inter, mut union) = (0u64, 0u64);
    for (x, y) in a.into_iter().zip(b) {
        if x && y {
            inter += 1;
        }
        if x || y {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

/// The value of `key`, or `default` when it is unset (or not UTF-8).
pub fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

/// [`env_or`], also treating a set-but-blank (whitespace-only) value as unset.
pub fn env_or_nonempty(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

/// The value of `key`, else of `fallback_key`; panics naming both when neither is set.
pub fn env_or_env(key: &str, fallback_key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| {
        std::env::var(fallback_key).unwrap_or_else(|_| panic!("set {key} or {fallback_key}"))
    })
}

/// `key`, whitespace-trimmed, parsed as `T`; `default` when it is unset, blank or unparsable.
pub fn env_parse_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// An optional env-var path — `None` when unset (or not UTF-8), for rows that skip gracefully.
pub fn env_path_opt(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

/// The path named by `var`, else `$HOME/<home_relative>` — the override-first shape for a **derived
/// cache** a test builds itself (a converted-weights directory). A provided input must hard-fail
/// instead (`require_env_path` / `env_path`), never default (epic 13657).
pub fn env_path_or_home(var: &str, home_relative: &str) -> PathBuf {
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").expect("HOME")).join(home_relative)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denom_guards_are_distinct_and_spell_the_f32_literal() {
        assert_eq!(Denom::Exact.apply_f64(2.0), 2.0);
        assert_eq!(Denom::PlusEps(0.5).apply_f64(2.0), 2.5);
        assert_eq!(Denom::FloorAt(0.5).apply_f64(2.0), 2.0);
        assert_eq!(Denom::FloorAt(0.5).apply_f64(0.25), 0.5);
        // The f32 copies spelled these literals directly; the f64 → f32 rounding must agree.
        for (wide, narrow) in [(1e-12f64, 1e-12f32), (1e-6, 1e-6), (1e-9, 1e-9), (1e-30, 1e-30)] {
            assert_eq!((wide as f32).to_bits(), narrow.to_bits());
        }
    }

    #[test]
    fn cosine_sums_pin_accumulator_and_guard() {
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let c = [0.0f32, 1.0, 1.0, 0.0];
        assert!((cosine_f32_sums(&a, &a, Denom::Exact) - 1.0).abs() < 1e-6);
        assert_eq!(cosine_f32_sums(&a, &c, Denom::Exact), 0.0);
        // |a|·|a| = 2: `+1` and `max(4)` are visibly different guards.
        assert_eq!(cosine_f32_sums(&a, &a, Denom::PlusEps(1.0)), 2.0 / 3.0);
        assert_eq!(cosine_f32_sums(&a, &a, Denom::FloorAt(4.0)), 0.5);
        assert!(cosine_f32_sums(&[0.0], &[0.0], Denom::Exact).is_nan());
        assert_eq!(cosine_f32_sums(&[0.0], &[0.0], Denom::PlusEps(1e-12)), 0.0);
        // f64 accumulation: 1e8 + 1 − 1e8 survives in f64 but rounds away in f32.
        let x = [1e8f32, 1.0, -1e8];
        let ones = [1.0f32, 1.0, 1.0];
        let wide = cosine_f64_sums(&x, &ones, Denom::Exact);
        let narrow = cosine_f32_sums(&x, &ones, Denom::Exact);
        assert!(wide > 0.0 && narrow == 0.0);
        let px = [0u8, 255, 0];
        assert!((cosine_f64_sums(&px, &px, Denom::PlusEps(1e-12)) - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "different lengths")]
    fn slice_comparisons_refuse_mismatched_lengths() {
        max_abs_diff_slices(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn max_and_peak_rel_pin_nan_skipping_and_floor() {
        assert_eq!(max_abs_diff_slices(&[1.0, -2.0], &[0.5, 2.0]), 4.0);
        assert_eq!(max_abs_diff_slices(&[f32::NAN, 1.0], &[0.0, 0.0]), 1.0);
        assert_eq!(max_abs_slice(&[-3.0, 2.0]), 3.0);
        assert_eq!(max_abs_diff_u8(&[0, 255], &[255, 250]), 255);
        assert_eq!(max_abs_diff_u8(&[], &[]), 0);
        let (got, want) = ([2.5f32, -4.0], [2.0f32, -4.0]);
        assert_eq!(peak_rel_slices(&got, &want, None), 0.125);
        assert_eq!(peak_rel_slices(&got, &want, Some(8.0)), 0.0625);
        assert!(peak_rel_slices(&[1.0], &[0.0], None).is_infinite());
        assert_eq!(peak_rel_slices(&[1.0], &[0.0], Some(0.5)), 2.0);
        assert_eq!(peak_rel_slices_f64(&got, &want, Some(1e-12)), 0.125);
        assert_eq!(peak_rel_slices_f64(&[1.0], &[0.0], Some(0.25)), 4.0);
    }

    #[test]
    fn frame_peak_rel_divides_by_the_first_arguments_peak() {
        let img = |pixels: Vec<u8>| Image {
            width: 1,
            height: 1,
            pixels,
        };
        let a = [img(vec![200, 100, 0])];
        let b = [img(vec![150, 100, 0])];
        assert_eq!(peak_rel_u8_frames(&a, &b), 0.25);
        assert_eq!(peak_rel_u8_frames(&b, &a), 50.0 / 150.0);
        let black = [img(vec![0, 0, 0])];
        assert_eq!(peak_rel_u8_frames(&black, &b), f64::INFINITY);
    }

    #[test]
    fn iou_helpers_pin_boxes_and_empty_masks() {
        let a = [0.0f32, 0.0, 2.0, 2.0];
        let b = [1.0f32, 1.0, 3.0, 3.0];
        assert_eq!(box_iou_xyxy(&a, &a), 1.0);
        assert_eq!(box_iou_xyxy(&a, &b), 1.0 / 7.0);
        assert_eq!(box_iou_xyxy(&a, &[5.0, 5.0, 6.0, 6.0]), 0.0);
        assert_eq!(iou_bool([true, true, false], [true, false, false]), 0.5);
        assert_eq!(iou_bool([false, false], [false, false]), 1.0);
    }

    #[test]
    fn env_resolvers_read_the_override_first() {
        // Unique, never-set names: no other test touches them, so no env lock is needed.
        let unset = "GEN_CORE_TESTKIT_HOST_UNSET_8f2c";
        assert_eq!(env_or(unset, "d"), "d");
        assert_eq!(env_or_nonempty(unset, "d"), "d");
        assert_eq!(env_parse_or(unset, 7u32), 7);
        assert_eq!(env_path_opt(unset), None);
        if let Ok(home) = std::env::var("HOME") {
            assert_eq!(
                env_path_or_home(unset, "x/y"),
                PathBuf::from(home).join("x/y")
            );
        }
        // PATH is set on every CI host; it is read, never written.
        let path = std::env::var("PATH").expect("PATH");
        assert_eq!(env_or("PATH", "d"), path);
        assert_eq!(env_or_env(unset, "PATH"), path);
        assert_eq!(env_path_or_home("PATH", "x"), PathBuf::from(&path));
    }
}
