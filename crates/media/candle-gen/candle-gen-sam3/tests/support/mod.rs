//! Shared tensor comparison helpers for the consolidated integration binary (sc-27381).

use candle_gen::candle_core::{DType, Tensor};

/// `sum_all` of `t` read back as an `f32` scalar.
pub(crate) fn sum_scalar(t: Tensor) -> f32 {
    t.sum_all().unwrap().to_scalar::<f32>().unwrap()
}

/// Cosine similarity between two tensors (flattened), accumulated in the tensors' own `f32` sums.
pub(crate) fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = sum_scalar((&a * &b).unwrap());
    let na = sum_scalar((&a * &a).unwrap()).sqrt();
    let nb = sum_scalar((&b * &b).unwrap()).sqrt();
    dot / (na * nb)
}

/// Largest absolute elementwise difference between two tensors (flattened).
pub(crate) fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let d = (a.flatten_all().unwrap() - b.flatten_all().unwrap()).unwrap();
    d.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

/// Intersection-over-union of two binary masks; two empty masks count as a perfect match.
pub(crate) fn iou(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).unwrap();
    let bf = b.to_dtype(DType::F32).unwrap();
    let inter = sum_scalar((&af * &bf).unwrap());
    let sa = sum_scalar(af);
    let sb = sum_scalar(bf);
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}
