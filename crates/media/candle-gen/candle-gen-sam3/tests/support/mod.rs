//! Shared tensor comparison helpers for the consolidated integration binary (sc-27381).

use candle_gen::candle_core::Tensor;

/// Largest absolute elementwise difference between two flattened tensors (the shared testkit form).
pub(crate) fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    candle_gen::testkit::tensor_max_abs_diff(&a.flatten_all().unwrap(), &b.flatten_all().unwrap())
}
