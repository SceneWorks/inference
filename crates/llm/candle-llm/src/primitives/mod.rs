//! Backend-owned tensor primitives (epic 7153).
//!
//! These are the decode leaves `candle-llm` owns — the Candle reimplementation of the `mlx-llm`
//! foundation: a batch-capable KV cache, the sampler, the RoPE family, GQA attention helpers,
//! group-wise quantization (Candle's `QTensor`/`QMatMul`), the `nn` leaves (linear / RMSNorm /
//! activations / embedding), and a safetensors weights loader. They own Candle `Tensor`s directly.
//!
//! Shapes are **batch-capable from day one**: the batch axis is a real dimension everywhere, even
//! though the first decoders run batch-1. The [`KvCache`] trait is the seam a paged cache slots in
//! behind without touching decoders.

pub mod attention;
pub mod gated_delta;
pub mod kv_cache;
pub mod nn;
pub mod paged_kv_cache;
pub mod projection;
pub mod quant;
pub mod rope;
pub mod sampler;
pub mod weights;

pub use attention::{repeat_kv, sdpa, sdpa_causal, AttnMask};
pub use gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
pub use kv_cache::{ContiguousKvCache, KvCache};
pub use nn::{
    conv2d, embed, gelu, gelu_erf, input_ids, input_ids_batch, layer_norm, linear, rms_norm, silu,
    soft_cap,
};
pub use paged_kv_cache::{BlockPool, PagedKvCache};
pub use projection::{Projection, QuantSpec};
pub use quant::QuantizedLinear;
pub use rope::{apply_rope, Rope};
pub use sampler::{sample, shaped_candidates, SamplingParams, SplitMix64, TokenRng};
pub use weights::Weights;
