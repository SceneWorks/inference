//! Lower-level MLX linear, quantization, and residual adapter primitives shared below media and LLM
//! engines (`SW-6955` / `SW-F335`).
//!
//! This crate depends only on tensor-free contracts (`sceneworks-gen-core`, `core-llm`) and the
//! pinned MLX bindings (`mlx-rs`, `mlx-sys`), with zero upward dependency on `mlx-gen*`, `mlx-llm*`,
//! `candle-*`, `runtime-*`, or training crates.

pub use ::gen_core;

#[macro_use]
#[path = "../../../llm/test-support/model_proof.rs"]
mod model_proof;

pub mod adapters;
pub mod convrot_int8;
pub mod error;
pub mod int8_gemm;
pub mod llm_projection;
pub mod prism;
pub mod quant;
pub mod w4a8_int8;

pub use adapters::{
    AdaptableConv2d, AdaptableHost, AdaptableLinear, Adapter, DiffPatchPart, Linear, LinearBase,
    LinearFacts, LokrFactors, ScaleContext,
};
pub use error::{Error, Result};
pub use quant::TokenEmbedding;

use mlx_rs::Array;
use std::sync::OnceLock;

/// Minimal tensor-lookup contract over a loaded MLX weight table so [`convrot_int8::ConvRotLinear`],
/// [`w4a8_int8::W4A8Linear`], [`quant::lin`], and [`quant::embedding`] construct directly in
/// `mlx-layers` without depending on `mlx-gen`'s loader/weight types.
pub trait MlxWeightSource {
    /// Look up an optional tensor by exact key.
    fn get(&self, key: &str) -> Option<&Array>;

    /// Look up a required tensor by exact key.
    fn require(&self, key: &str) -> Result<&Array>;

    /// Return the cached `[group, group]` orthogonal Walsh-Hadamard rotation matrix on the target
    /// stream, building it via `build` on first request for `group`.
    fn convrot_rotation(&self, group: usize, build: impl FnOnce() -> Array) -> Array;
}

/// Scalar `f32` `Array` constructor matching `mlx_gen::array::scalar`.
#[inline]
pub fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

/// Contiguous `[.., start..start+len, ..]` slice along `axis` via boundary splits.
#[inline]
pub fn slice_axis(a: &Array, axis: i32, start: i32, len: i32) -> Result<Array> {
    Ok(a.split_axis(&[start, start + len], axis)?.swap_remove(1))
}

struct ExactEpilogueHooks {
    enabled: fn() -> bool,
    record_applied: fn(&'static str),
    record_fallback: fn(&'static str),
}

static EXACT_EPILOGUE_HOOKS: OnceLock<ExactEpilogueHooks> = OnceLock::new();

/// Register process-wide exact-epilogue capability and diagnostic hooks from `mlx-gen` without an
/// upward crate dependency. Safe to call repeatedly; the first registration wins.
pub fn register_exact_epilogue_hooks(
    enabled: fn() -> bool,
    record_applied: fn(&'static str),
    record_fallback: fn(&'static str),
) {
    let _ = EXACT_EPILOGUE_HOOKS.set(ExactEpilogueHooks {
        enabled,
        record_applied,
        record_fallback,
    });
}

#[inline]
pub(crate) fn exact_epilogues_enabled() -> bool {
    EXACT_EPILOGUE_HOOKS.get().is_none_or(|h| (h.enabled)())
}

#[inline]
pub(crate) fn record_exact_epilogue_applied(op: &'static str) {
    if let Some(h) = EXACT_EPILOGUE_HOOKS.get() {
        (h.record_applied)(op);
    }
}

#[inline]
pub(crate) fn record_exact_epilogue_fallback(op: &'static str) {
    if let Some(h) = EXACT_EPILOGUE_HOOKS.get() {
        (h.record_fallback)(op);
    }
}
