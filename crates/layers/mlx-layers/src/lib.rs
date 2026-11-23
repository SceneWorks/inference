//! Lower-level MLX linear, quantization, and residual adapter primitives shared below media and LLM
//! engines (`SW-6955` / `SW-F335`).
//!
//! This crate depends only on tensor-free contracts (`sceneworks-gen-core`, `core-llm`) and the
//! pinned MLX bindings (`mlx-rs`, `mlx-sys`), with zero upward dependency on `mlx-gen*`, `mlx-llm*`,
//! `candle-*`, `runtime-*`, or training crates.

pub use ::gen_core;

#[allow(unused_imports, unused_macros)]
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
pub use llm_projection::{
    quantize_on_stored_grid, KvProjection, Projection, QuantGrid, QuantSpec, QuantizedEmbedding,
    QuantizedLinear,
};
pub use prism::{PrismEmbedding, PrismLinear};
pub use quant::TokenEmbedding;

use mlx_rs::Array;
use std::sync::OnceLock;

/// Source operands and load-scoped rotation cache. Both engines mark operands through their
/// normal access-tracking weight map so block eviction and materialization keep their semantics.
pub trait MlxWeightSource {
    fn require(&self, key: &str) -> Result<&Array> {
        self.get(key)
            .ok_or_else(|| crate::Error::MissingTensor(key.to_owned()))
    }
    fn planned_fp8(&self, _base: &str) -> Option<&crate::fp8::Fp8Linear> {
        None
    }
    /// Ordered floating-point adapter deltas retained alongside a source-owned native matrix.
    fn native_projection_deltas(&self, _base: &str) -> &[Array] {
        &[]
    }
    fn convrot_rotation(&self, group: usize, build: impl FnOnce() -> Array) -> Array;
    fn get(&self, key: &str) -> Option<&Array>;
    /// Inspect an optional operand without claiming it for a block's access set.
    fn peek(&self, key: &str) -> Option<&Array> {
        self.get(key)
    }
    /// A constructor successfully bound a native operator, including for in-memory sources
    /// without a file plan. Materialization and resident bytes are accounted separately.
    fn native_bound(&self) {}
    fn observation(&self, _key: &str) -> Option<crate::source::Observation> {
        None
    }
    fn planned_geometry(&self, _base: &str) -> Option<gen_core::convrot_int8::ConvRotInt8> {
        None
    }
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

pub mod fp8;
pub mod source;
pub use gen_core::comfy_quant::ComfyQuantFormat;

#[derive(Clone, Debug)]
pub enum NativeProjection {
    ConvRot(convrot_int8::ConvRotLinear),
    Fp8(fp8::Fp8Linear),
}

impl NativeProjection {
    /// Probe the source without adding a plain tensor to a block's tracked accesses.
    pub fn is_declared(w: &(impl MlxWeightSource + ?Sized), base: &str, key: &str) -> bool {
        w.planned_fp8(base).is_some()
            || w.planned_geometry(base).is_some()
            || w.peek(&format!("{base}.comfy_quant")).is_some()
            || w.peek(key)
                .is_some_and(|v| v.dtype() == mlx_rs::Dtype::Int8)
    }

    pub fn from_weights(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        Self::from_key(w, base, &format!("{base}.weight"), bias)
    }
    pub fn from_key(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        key: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        let mut native = Self::bind_key(w, base, key, bias)?;
        if let (Some(bound), Some(observation)) = (native.as_mut(), w.observation(key)) {
            match bound {
                Self::Fp8(p) => p.observe(observation)?,
                Self::ConvRot(p) => p.observe(observation)?,
            }
        }
        if native.is_some() {
            w.native_bound();
        }
        Ok(native)
    }
    fn bind_key(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        key: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        if let Some(p) = w.planned_fp8(base) {
            let mut p = p.clone();
            w.require(key)?;
            w.get(&format!("{base}.comfy_quant"));
            w.get(&format!("{base}.weight_scale"));
            w.get(&format!("{base}.input_scale"));
            if bias {
                p.set_bias(w.require(&format!("{base}.bias"))?.clone())?;
            }
            return Ok(Some(Self::Fp8(p)));
        }
        if let Some(blob) = w.get(&format!("{base}.comfy_quant")) {
            let desc = convrot_int8::descriptor(blob, base)?;
            use gen_core::comfy_quant::ComfyQuantFormat::*;
            if matches!(desc.format, Float8E4M3Fn | Float8E5M2 | Fp8Rowwise | Mxfp8) {
                let codes = w.require(key)?.clone();
                let shape: Vec<_> = codes.shape().iter().map(|&n| n as usize).collect();
                let logical = if desc.format == Mxfp8 {
                    desc.orig_shape
                        .as_deref()
                        .ok_or("MXFP8 requires architecture-declared logical dimensions")?
                } else {
                    &shape
                };
                let scale = w.require(&format!("{base}.weight_scale"))?.clone();
                let sx = w.get(&format!("{base}.input_scale")).cloned();
                let bias = bias
                    .then(|| w.require(&format!("{base}.bias")).cloned())
                    .transpose()?;
                return Ok(Some(Self::Fp8(fp8::Fp8Linear::from_source(
                    desc.format,
                    codes,
                    Some(scale),
                    sx,
                    logical,
                    bias,
                    desc.full_precision_matrix_mult,
                )?)));
            }
            if desc.format != ConvRotInt8 {
                return Ok(None);
            }
        }
        Ok(convrot_int8::ConvRotLinear::from_key(w, base, key, bias)?.map(Self::ConvRot))
    }
    pub fn shape(&self) -> [i32; 2] {
        match self {
            Self::ConvRot(p) => [
                p.descriptor().out_features as i32,
                p.descriptor().in_features as i32,
            ],
            Self::Fp8(p) => p.shape(),
        }
    }
    pub fn rows(&self, ids: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        self.rows_as(ids, mlx_rs::Dtype::Bfloat16)
    }
    pub fn rows_as(&self, ids: &mlx_rs::Array, dtype: mlx_rs::Dtype) -> Result<mlx_rs::Array> {
        match self {
            Self::ConvRot(p) => p.rows_as(ids, dtype),
            Self::Fp8(p) => p.rows_as(ids, dtype),
        }
    }
    pub fn materialize(&self) -> Result<()> {
        match self {
            Self::ConvRot(p) => p.materialize(),
            Self::Fp8(p) => p.materialize(),
        }
    }
    pub fn operands(&self) -> Vec<(&'static str, &mlx_rs::Array)> {
        match self {
            Self::ConvRot(p) => vec![
                ("weight", p.codes()),
                ("weight_scale", p.scales()),
                ("rotation", p.rotation()),
            ],
            Self::Fp8(p) => vec![
                ("weight", p.codes()),
                ("weight_scale", p.scales()),
                ("input_scale", p.input_scale()),
            ],
        }
    }
    pub fn set_bias(&mut self, b: mlx_rs::Array) -> Result<()> {
        match self {
            Self::ConvRot(p) => p.set_bias(b),
            Self::Fp8(p) => p.set_bias(b),
        }
    }
    pub fn slice_rows(&self, lo: i32, hi: i32) -> Result<Self> {
        Ok(match self {
            Self::ConvRot(p) => Self::ConvRot(p.slice_rows(lo, hi)?),
            Self::Fp8(p) => Self::Fp8(p.slice_rows(lo, hi)?),
        })
    }
    pub fn push_arrays(&self, out: &mut Vec<mlx_rs::Array>) {
        match self {
            Self::ConvRot(p) => p.push_arrays(out),
            Self::Fp8(p) => p.push_arrays(out),
        }
    }
    pub fn forward_with_bias(
        &self,
        x: &mlx_rs::Array,
        b: Option<&mlx_rs::Array>,
    ) -> Result<mlx_rs::Array> {
        match self {
            Self::ConvRot(p) => p.forward_with_bias(x, b),
            Self::Fp8(p) => p.forward_with_bias(x, b),
        }
    }
    pub fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        match self {
            Self::ConvRot(p) => p.forward(x),
            Self::Fp8(p) => p.forward(x),
        }
    }
}

pub mod quantization_cache;

pub use source::contiguous_weight;
