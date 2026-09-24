//! Backend-owned tensor primitives (epic 7153).
//!
//! These are the decode leaves `candle-llm` owns — the Candle reimplementation of the `mlx-llm`
//! foundation: the batch-capable KV caches (growing and preallocated), the sampler, the RoPE family, GQA attention helpers,
//! group-wise quantization (Candle's `QTensor`/`QMatMul`), the `nn` leaves (linear / RMSNorm /
//! activations / embedding), and a safetensors weights loader. They own Candle `Tensor`s directly.
//!
//! The RMSNorm / SwiGLU / QK-norm+RoPE leaves have a fused CUDA implementation behind the same
//! entry points (sc-24137): bit-identical to the op chain, on by default in a `cuda` build, with
//! which path ran recorded per thread in [`fused`] and per request in the decode record.
//!
//! NVFP4 projections pick the fused decode GEMV (≤ 8 rows) or the cuBLASLt W4A4 GEMM per call
//! (sc-24136), with the path recorded per thread in [`nvfp4_path`] and per request in the decode
//! record.
//!
//! Shapes are **batch-capable from day one**: the batch axis is a real dimension everywhere, even
//! though the first decoders run batch-1. The [`KvCache`] trait is the seam a paged cache slots in
//! behind without touching decoders.

pub mod attention;
pub mod decode_cache;
pub mod fused;
pub mod gated_delta;
pub mod host_sync;
pub mod kv_cache;
pub mod nn;
pub mod nvfp4_path;
pub mod paged_kv_cache;
pub mod prism;
pub mod projection;
pub mod quant;
pub mod rope;
pub mod sampler;
pub mod step_kv_cache;
pub mod switch;
pub mod weights;

pub use attention::{
    repeat_kv, sdpa, sdpa_causal, sdpa_gqa_causal, sliding_causal_mask, AttnFormulation, AttnMask,
};
pub use decode_cache::{tensor_bytes, CacheMemory, DecodeCache};
pub use fused::{
    fused_kernels_enabled, fused_tally, set_fused_kernels, FusedTally, FUSED_KERNELS_ENV,
};
#[doc(hidden)]
pub use fused::{fused_policy_guard, FusedPolicyGuard};
pub use gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
pub use host_sync::{
    host_sync_count, last_host_reason, note_host_sync, note_logits_to_host, note_sampler_path,
    sampler_counters, SamplerCounters,
};
pub use kv_cache::{
    kv_materialize_count, note_kv_materialize, storage_address, ContiguousKvCache, KvCache,
    KvCacheKind, StaticKvCache,
};
pub use nn::{
    conv2d, embed, gelu, gelu_erf, input_ids, input_ids_batch, layer_norm, linear, rms_norm,
    rms_norm_reference, rms_norm_residual, rms_norm_unscaled, silu, soft_cap, swiglu,
};
pub use nvfp4_path::{
    nvfp4_gemv_enabled, nvfp4_path_tally, set_nvfp4_gemv, Nvfp4PathTally, NVFP4_GEMV_ENV,
};
#[doc(hidden)]
pub use nvfp4_path::{nvfp4_gemv_policy_guard, Nvfp4GemvPolicyGuard};
pub use paged_kv_cache::{BlockPool, PagedKvCache};
pub use prism::{GdnRowMap, PrismPackedWeight, PrismRegistry};
pub use projection::{
    KvProjection, Projection, ProjectionCensus, ProjectionFormat, ProjectionKind, ProjectionTally,
    QuantSpec, WeightCensus,
};
pub use quant::QuantizedLinear;
pub use rope::{apply_rope, rms_norm_rope, Rope};
pub use sampler::{
    device_sampler_available, sample, sample_device, sample_host, sampler_path, shaped_candidates,
    uniform_device, with_reference_sampler, HostSampleReason, SamplerPath, SamplingParams,
    SplitMix64, TokenRng,
};
pub use step_kv_cache::{KvLayout, LayerKvShape, StepKvCache};
pub use switch::{ProcessSwitch, SwitchGuard};
pub use weights::Weights;
