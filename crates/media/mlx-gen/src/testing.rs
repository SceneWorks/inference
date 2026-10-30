//! Shared **test-only** comparison helpers (sc-27323), behind the `test-support` feature.
//!
//! ~100 provider test files each re-declared their own `cosine` / `max_abs_diff` / `iou` /
//! `max_abs` / `peak_rel` / `save_png` / snapshot-path helpers with subtly different bodies (flatten
//! or not, device vs host `sqrt`, f32 cast or not). New and touched tests use these instead; enable
//! them with a dev-dependency `mlx-gen = { path = "..", features = ["test-support"] }`. None of
//! them runs a model.
//!
//! The copies did not all compute the same number, and each parity gate's tolerance is tuned to its
//! own metric, so every distinct computation has its own name here rather than one "close enough"
//! body: whether the inputs are cast to f32 **before** the arithmetic (a bf16 difference or sum
//! rounds differently), whether the reduction runs on the device (NaN-propagating) or as a host
//! fold over a read-back slice (NaN-skipping), and how a denominator is guarded ([`Denom`], or an
//! explicit peak floor). The backend-neutral slice metrics and env resolvers come from
//! [`gen_core_testkit::host`] and are re-exported here, so a test needs one import path.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{abs, max, multiply, sqrt, subtract, sum};
use mlx_rs::{Array, Dtype};

use crate::media::Image;

pub use gen_core_testkit::host::*;

/// Read a (0-D or one-element) array back as an `f32`, casting first so a bf16/f16/integer
/// reduction never mis-reads.
pub fn item_f32(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32).unwrap().item::<f32>()
}

/// Cosine similarity of two same-sized arrays over **all** elements (both flattened): the device
/// dot product over the device product of the two L2 norms.
pub fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.reshape(&[-1]).unwrap();
    let b = b.reshape(&[-1]).unwrap();
    let dot = item_f32(&sum(multiply(&a, &b).unwrap(), None).unwrap());
    let na = item_f32(&sqrt(sum(multiply(&a, &a).unwrap(), None).unwrap()).unwrap());
    let nb = item_f32(&sqrt(sum(multiply(&b, &b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

/// `max|a − b|` over all elements.
pub fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    item_f32(&max(abs(subtract(a, b).unwrap()).unwrap(), None).unwrap())
}

/// `max|a|` over all elements.
pub fn max_abs(a: &Array) -> f32 {
    item_f32(&max(abs(a).unwrap(), None).unwrap())
}

/// Peak relative error `max|got − want| / max|want|` (the denominator floored at `1e-12`), both
/// read in f32.
pub fn peak_rel(got: &Array, want: &Array) -> f32 {
    peak_rel_f32(got, want, Some(1e-12))
}

/// [`peak_rel`] with an explicit denominator floor: both inputs cast to f32 first, then device
/// `max|got − want|` over device `max|want|` (raised to `floor` when given).
pub fn peak_rel_f32(got: &Array, want: &Array, floor: Option<f32>) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    peak_rel_native(&got, &want, floor)
}

/// Peak relative error with **no** f32 cast first: the difference is taken (and both maxima
/// reduced) on the device in the inputs' own dtype, each maximum read back as f32, the
/// denominator raised to `floor` when given.
pub fn peak_rel_native(got: &Array, want: &Array, floor: Option<f32>) -> f32 {
    let peak = max_abs(want);
    max_abs_diff(got, want) / floor.map_or(peak, |f| peak.max(f))
}

/// `max|a − b| / max|b|` (both cast to f32, device reductions), except that an all-zero `b`
/// returns the raw `max|a − b|` instead of dividing by zero.
pub fn peak_rel_or_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let (diff, scale) = (max_abs_diff(&a, &b), max_abs(&b));
    if scale == 0.0 {
        diff
    } else {
        diff / scale
    }
}

/// Peak relative error as a **host** fold over the f32 read-backs ([`peak_rel_slices`]): NaN
/// differences are skipped, the denominator raised to `floor` when given.
pub fn peak_rel_host(got: &Array, want: &Array, floor: Option<f32>) -> f32 {
    peak_rel_slices(&host_f32(got), &host_f32(want), floor)
}

/// The array cast to f32 and flattened in logical (row-major) order, read back to the host.
pub fn host_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// `max|a − b|` with both inputs cast to f32 **before** the device subtraction (a bf16 difference
/// would otherwise round in bf16).
pub fn max_abs_diff_f32(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    max_abs_diff(&a, &b)
}

/// `max|a − b|` as a host fold over the f32 read-backs ([`max_abs_diff_slices`], NaN-skipping).
pub fn max_abs_diff_host(a: &Array, b: &Array) -> f32 {
    max_abs_diff_slices(&host_f32(a), &host_f32(b))
}

/// `max|a|` as a host fold over the f32 read-back ([`max_abs_slice`], NaN-skipping).
pub fn max_abs_host(a: &Array) -> f32 {
    max_abs_slice(&host_f32(a))
}

/// Cosine with both inputs cast to f32 and **every** step on the device — the three sums, both
/// square roots and the final division — reading back only the quotient.
pub fn cosine_f32_device(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), None).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), None).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), None).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

/// Cosine from three device sums in the inputs' **own** dtype, each read back as f32; the square
/// roots, the guard and the division then run on the host.
pub fn cosine_device_sums(a: &Array, b: &Array, denom: Denom) -> f32 {
    let dot = item_f32(&sum(multiply(a, b).unwrap(), None).unwrap());
    let na = item_f32(&sum(multiply(a, a).unwrap(), None).unwrap()).sqrt();
    let nb = item_f32(&sum(multiply(b, b).unwrap(), None).unwrap()).sqrt();
    dot / denom.apply_f32(na * nb)
}

/// [`cosine_device_sums`] with both inputs cast to f32 first (the sums accumulate in f32 even for
/// bf16 inputs).
pub fn cosine_device_sums_f32(a: &Array, b: &Array, denom: Denom) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    cosine_device_sums(&a, &b, denom)
}

/// Cosine accumulated in f64 on the host over the f32 read-backs ([`cosine_f64_sums`]).
pub fn cosine_host_f64(a: &Array, b: &Array, denom: Denom) -> f64 {
    cosine_f64_sums(&host_f32(a), &host_f32(b), denom)
}

/// Cosine accumulated in f32 on the host over the f32 read-backs ([`cosine_f32_sums`]).
pub fn cosine_host_f32(a: &Array, b: &Array, denom: Denom) -> f32 {
    cosine_f32_sums(&host_f32(a), &host_f32(b), denom)
}

/// Write an RGB8 [`Image`] as a PNG; panics on failure (a test artefact).
pub fn save_png(img: &Image, path: impl AsRef<Path>) {
    ::image::save_buffer(
        path.as_ref(),
        &img.pixels,
        img.width,
        img.height,
        ::image::ColorType::Rgb8,
    )
    .unwrap();
}

/// Write an `[1, 3, H, W]` array holding values in `[-1, 1]` as an RGB8 PNG: `(x + 1) · 127.5`,
/// clamped to `[0, 255]` and **truncated** (not rounded) to u8.
pub fn save_nchw_signed_unit_png(out: &Array, path: impl AsRef<Path>) {
    let sh = out.shape();
    let (h, w) = (sh[2], sh[3]);
    let hwc = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[3, h, w])
        .unwrap()
        .transpose_axes(&[1, 2, 0])
        .unwrap()
        .reshape(&[h * w * 3])
        .unwrap();
    let buf: Vec<u8> = hwc
        .as_slice::<f32>()
        .iter()
        .map(|x| (((x + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect();
    ::image::save_buffer(
        path.as_ref(),
        &buf,
        w as u32,
        h as u32,
        ::image::ColorType::Rgb8,
    )
    .unwrap();
}

/// The directory or file named by the environment variable `var` — the caller-provisioned
/// real-weight input of an `#[ignore]`d test. Panics naming `what` when unset: inference never
/// self-fetches or derives a cache location (epic 13657).
pub fn require_env_path(var: &str, what: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(path) => PathBuf::from(path),
        None => panic!(
            "set {var} to {what}; inference never self-fetches or derives a cache location \
             (epic 13657)"
        ),
    }
}

/// Intersection-over-union of two binary (0/1) masks of the same shape; two empty masks are a
/// perfect match (`1.0`).
pub fn iou(a: &Array, b: &Array) -> f32 {
    let af = a.as_dtype(Dtype::Float32).unwrap();
    let bf = b.as_dtype(Dtype::Float32).unwrap();
    let inter = item_f32(&sum(multiply(&af, &bf).unwrap(), None).unwrap());
    let sa = item_f32(&sum(&af, None).unwrap());
    let sb = item_f32(&sum(&bf, None).unwrap());
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The helpers are plain reductions (no model): pin their arithmetic on tiny host arrays.
    #[test]
    fn comparison_helpers_reduce_over_all_elements() {
        let a = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let b = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[4]);
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        let c = Array::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[2, 2]);
        assert!(cosine(&a, &c).abs() < 1e-6);
        assert_eq!(max_abs_diff(&a, &c), 1.0);
        assert_eq!(iou(&a, &a), 1.0);
        assert_eq!(iou(&a, &c), 0.0);
        let empty = Array::from_slice(&[0u8; 4], &[2, 2]);
        assert_eq!(iou(&empty, &empty), 1.0);
        assert_eq!(item_f32(&Array::from_slice(&[3i32], &[1])), 3.0);
        let want = Array::from_slice(&[2.0f32, -4.0], &[2]);
        let got = Array::from_slice(&[2.5f32, -4.0], &[2]);
        assert_eq!(max_abs(&want), 4.0);
        assert_eq!(peak_rel(&got, &want), 0.125);
    }

    /// The named variants differ exactly where their names say: cast placement, device vs host
    /// reduction, and the denominator guard.
    #[test]
    fn semantic_variants_pin_cast_reduction_and_guard() {
        let a = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let c = Array::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[2, 2]);
        assert_eq!(host_f32(&a), vec![1.0, 0.0, 0.0, 1.0]);
        // A transposed view reads back in logical order, not storage order.
        let t = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .transpose_axes(&[1, 0])
            .unwrap();
        assert_eq!(host_f32(&t), vec![1.0, 3.0, 2.0, 4.0]);

        for cos in [
            cosine_f32_device(&a, &a),
            cosine_device_sums(&a, &a, Denom::Exact),
            cosine_device_sums_f32(&a, &a, Denom::Exact),
            cosine_host_f32(&a, &a, Denom::Exact),
            cosine_host_f64(&a, &a, Denom::Exact) as f32,
        ] {
            assert!((cos - 1.0).abs() < 1e-6, "{cos}");
        }
        assert_eq!(cosine_device_sums(&a, &c, Denom::Exact), 0.0);
        // |v|·|v| = 25 exactly, so each guard's effect is exact.
        let v = Array::from_slice(&[3.0f32, 4.0], &[2]);
        assert_eq!(cosine_device_sums(&v, &v, Denom::FloorAt(50.0)), 0.5);
        assert_eq!(cosine_host_f64(&v, &v, Denom::PlusEps(25.0)), 0.5);
        assert_eq!(cosine_host_f32(&v, &v, Denom::PlusEps(25.0)), 0.5);

        // bf16 cast placement: 256 − 1.5 = 254.5 needs 9 significant bits, so a bf16 difference
        // rounds it (ties-to-even) to 254 while the f32-first difference keeps 254.5.
        let x = Array::from_slice(&[256.0f32], &[1]).as_dtype(Dtype::Bfloat16).unwrap();
        let y = Array::from_slice(&[1.5f32], &[1]).as_dtype(Dtype::Bfloat16).unwrap();
        assert_eq!(max_abs_diff(&x, &y), 254.0);
        assert_eq!(max_abs_diff_f32(&x, &y), 254.5);
        assert_eq!(max_abs_diff_host(&x, &y), 254.5);

        // Device max propagates NaN; the host fold skips it.
        let nan = Array::from_slice(&[f32::NAN, -3.0], &[2]);
        assert!(max_abs(&nan).is_nan());
        assert_eq!(max_abs_host(&nan), 3.0);

        let want = Array::from_slice(&[2.0f32, -4.0], &[2]);
        let got = Array::from_slice(&[2.5f32, -4.0], &[2]);
        assert_eq!(peak_rel_f32(&got, &want, None), 0.125);
        assert_eq!(peak_rel_native(&got, &want, Some(8.0)), 0.0625);
        assert_eq!(peak_rel_host(&got, &want, None), 0.125);
        let zero = Array::from_slice(&[0.0f32, 0.0], &[2]);
        assert_eq!(peak_rel_or_abs(&got, &zero), 4.0);
        assert_eq!(peak_rel_or_abs(&got, &want), 0.125);
    }

    #[test]
    fn signed_unit_png_truncates_and_reorders_channels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.png");
        // One pixel, channel-planar [1, 3, 1, 1]: R = -1, G = 0, B = 1.
        let out = Array::from_slice(&[-1.0f32, 0.0, 1.0], &[1, 3, 1, 1]);
        save_nchw_signed_unit_png(&out, &path);
        let png = ::image::open(&path).unwrap().to_rgb8();
        assert_eq!(png.get_pixel(0, 0).0, [0, 127, 255]);
    }
}
