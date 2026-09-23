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
//! - [`nvrtc`] — the shared nvrtc **compile-once** seam (sc-23990, landed by sc-24137): a
//!   [`KernelSource`] is compiled per device at first use, and the outcome — success *or* failure —
//!   is cached, so every runtime-compiled kernel in the workspace shares one mechanism.
//! - [`fused_decode`] — the fused decode primitives (sc-24137): RMSNorm(+residual), SwiGLU and
//!   QK-norm+RoPE as single launches that are bit-identical to candle's op chain, each with a typed
//!   shape/dtype refusal so the caller can fall back to the reference visibly.
//!
//! - [`nvfp4_gemv`] — the fused NVFP4 decode GEMV (sc-24136): a resident [`Nvfp4Weight`] times an
//!   unquantized bf16 activation of 1..=8 rows in one tensor-core launch (exact bf16 dequant,
//!   f32 accumulate), the
//!   decode-sized alternative to the W4A4 cuBLASLt forward, with a typed refusal for everything
//!   else so the caller falls back to cuBLASLt visibly.
//!
//! Device code is behind `cfg(feature = "cuda")`; a CPU or Metal build compiles the codec, the
//! capability floors, the kernel descriptors and the fused primitives' input checks only.

pub mod cublaslt;
pub mod fused_decode;
pub mod nvfp4;
pub mod nvfp4_gemv;
pub mod nvfp4_linear;
pub mod nvfp4_outlier;
pub mod nvfp4_weight;
pub mod nvrtc;

pub use cublaslt::{
    compute_cap_meets_fp8_floor, compute_cap_meets_nvfp4_floor, quantize_activation_fp8,
    quantize_activation_int8, quantize_weight_fp8, quantize_weight_int8,
    quantize_weight_int8_per_channel, Int8Context, PerChannelInt8Weight, QuantizedActivation,
    F8E4M3_MAX, FP8_COMPUTE_CAP_FLOOR, I8_MAX, NVFP4_COMPUTE_CAP_FLOOR, NVFP4_K_ALIGN,
    NVFP4_N_ALIGN,
};
#[cfg(feature = "cuda")]
pub use cublaslt::{CublasLt, DevNvfp4};
pub use fused_decode::{
    check_rms_norm, check_rms_norm_rope, check_swiglu, FusedError, FusedRefusal, RmsNormPlan,
    RopePlan, FUSED_DECODE_SRC, FUSED_ROPE_MAX_HEAD_DIM,
};
pub use nvfp4::{
    e2m1_from_f32, e4m3_from_f32, e4m3_to_f32, Nvfp4Tensor, E2M1_LUT, E2M1_MAX, E4M3_MAX,
    NVFP4_BLOCK,
};
pub use nvfp4_gemv::{
    check_nvfp4_gemv, gemv_abs_bound, Nvfp4GemvError, Nvfp4GemvPlan, Nvfp4GemvRefusal,
    GEMV_REL_RMS_TOL, NVFP4_GEMV_FUNCTION, NVFP4_GEMV_MAX_ROWS, NVFP4_GEMV_ROWS_PER_BLOCK,
    NVFP4_GEMV_SRC, NVFP4_GEMV_THREADS,
};
pub use nvfp4_linear::{
    ActPrecision, Nvfp4Context, Nvfp4Fallback, Nvfp4Linear, Nvfp4Partition, Nvfp4Regime,
    NVFP4_M_ALIGN,
};
pub use nvfp4_outlier::{OutlierClass, OutlierSparsity};
pub use nvfp4_weight::{
    nvfp4_refusal_for_compute_cap, nvfp4_shape_refusal, Nvfp4Refusal, Nvfp4Weight, NVFP4_CAPABILITY,
};
#[cfg(feature = "cuda")]
pub use nvrtc::{device_compute_cap, CompiledKernel};
pub use nvrtc::{nvrtc_arch_for, KernelCompileError, KernelSource};

/// Poison-tolerant `Mutex` lock for the handle's overwrite-on-miss caches — the same recovery
/// `candle_gen::lock_recover` documents (sc-9015): every cache on [`cublaslt::CublasLt`] is a
/// shape-keyed memo with no cross-field invariant, so a poisoned lock is safe to keep serving.
#[cfg(feature = "cuda")]
pub(crate) fn lock_recover<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
