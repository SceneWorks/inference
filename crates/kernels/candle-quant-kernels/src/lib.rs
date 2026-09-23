//! Shared low-precision Candle compute kernels (sc-24135, epic sc-24128).
//!
//! The one NVFP4 implementation in the workspace, plus the cuBLASLt wrapper it runs on. Lifted
//! verbatim out of `candle-gen::quant` so the LLM lane (`candle-llm`) can serve NVFP4 projections
//! without depending on the media engine; `candle-gen` re-exports every module here at its old
//! `candle_gen::quant::…` path, so media callers are unchanged.
//!
//! - [`nvfp4`] — the NVFP4 container, offline packer and CPU dequant reference (sc-11040).
//! - [`cublaslt`] — the cuBLASLt GEMM wrapper: fp8 / int8 (sc-9299) and the NVFP4 block-scaled FP4
//!   GEMM with its fused on-device activation quantizer (sc-11039 / sc-12078). The handle is
//!   `cuda`-only; the small quant helpers and capability floors build everywhere.
//! - [`nvfp4_linear`] — `Nvfp4Linear`, the media lane's FP4 linear layer with its W4A16 policy.
//! - [`nvfp4_outlier`] — activation-outlier sparsity instrumentation (sc-11044).
//! - [`nvfp4_weight`] — the LLM lane's load-time NVFP4 weight and its strict, typed capability
//!   gate (sc-24135): quantized on-device at load, refused (never downgraded) where FP4 cannot run.
//!
//! Device code is behind `cfg(feature = "cuda")`; a CPU or Metal build compiles the codec and the
//! capability floors only.

pub mod cublaslt;
pub mod nvfp4;
pub mod nvfp4_linear;
pub mod nvfp4_outlier;
pub mod nvfp4_weight;

pub use cublaslt::{
    compute_cap_meets_fp8_floor, compute_cap_meets_nvfp4_floor, quantize_activation_fp8,
    quantize_activation_int8, quantize_weight_fp8, quantize_weight_int8,
    quantize_weight_int8_per_channel, Int8Context, PerChannelInt8Weight, QuantizedActivation,
    F8E4M3_MAX, FP8_COMPUTE_CAP_FLOOR, I8_MAX, NVFP4_COMPUTE_CAP_FLOOR, NVFP4_K_ALIGN,
    NVFP4_N_ALIGN,
};
#[cfg(feature = "cuda")]
pub use cublaslt::{CublasLt, DevNvfp4};
pub use nvfp4::{
    e2m1_from_f32, e4m3_from_f32, e4m3_to_f32, Nvfp4Tensor, E2M1_LUT, E2M1_MAX, E4M3_MAX,
    NVFP4_BLOCK,
};
pub use nvfp4_linear::{
    ActPrecision, Nvfp4Context, Nvfp4Fallback, Nvfp4Linear, Nvfp4Partition, Nvfp4Regime,
    NVFP4_M_ALIGN,
};
pub use nvfp4_outlier::{OutlierClass, OutlierSparsity};
pub use nvfp4_weight::{
    nvfp4_refusal_for_compute_cap, nvfp4_shape_refusal, Nvfp4Refusal, Nvfp4Weight, NVFP4_CAPABILITY,
};

/// Poison-tolerant `Mutex` lock for the handle's overwrite-on-miss caches — the same recovery
/// `candle_gen::lock_recover` documents (sc-9015): every cache on [`cublaslt::CublasLt`] is a
/// shape-keyed memo with no cross-field invariant, so a poisoned lock is safe to keep serving.
#[cfg(feature = "cuda")]
pub(crate) fn lock_recover<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
