//! Shared **test-only** comparison helpers (sc-27323), behind the `test-support` feature.
//!
//! ~100 provider test files each re-declared their own `cosine` / `max_abs_diff` / `iou` with
//! subtly different bodies (flatten or not, device vs host `sqrt`, f32 cast or not). New and touched
//! tests use these instead; enable them with a dev-dependency
//! `mlx-gen = { path = "..", features = ["test-support"] }`. Every helper reduces in f32 and reads
//! back one host scalar; none of them runs a model.

use mlx_rs::ops::{abs, max, multiply, sqrt, subtract, sum};
use mlx_rs::{Array, Dtype};

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
    }
}
