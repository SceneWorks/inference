//! Backend-owned tensor primitives (epic 7153, story 7155).
//!
//! These are the decode leaves `mlx-llm` owns because it must not pull them from the gen-ai core:
//! a batch-capable KV cache, the sampler, the RoPE family, GQA attention helpers, group-wise
//! quantization, the `nn` leaves (linear / norms / activations / embedding), and a safetensors
//! weights loader. They are modelled faithfully on the working mlx-gen implementations
//! (prompt-refine's Llama decoder, JoyCaption's VLM, sensenova/flux2's Qwen3 stacks) so that the
//! generating stacks can later adopt them at parity — the accepted small leaf-duplication that
//! buys this crate its independence. The Candle backend (`candle-llm`) reimplements the
//! equivalents on candle.
//!
//! Shapes are **batch-capable from day one**: the batch axis is a real dimension everywhere, even
//! though the first decoders run batch-1. The [`KvCache`] trait is the seam the P4 paged cache
//! (story 7169) slots in behind without touching decoders.

pub mod activation;
pub mod attention;
pub mod coherence;
pub mod fused;
pub mod gated_delta;
pub mod kv_cache;
pub mod moe;
pub mod nn;
pub mod paged_kv_cache;
pub mod prism;
pub mod projection;
pub mod quant;
pub mod rope;
pub mod sampler;
pub mod weights;

pub use attention::{repeat_kv, sdpa, sdpa_capped, sdpa_causal, sliding_causal_mask, AttnMask};
pub use coherence::verify_gpu_view;
pub use gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_chunked, gated_delta_kernel,
    gated_delta_recurrence, gated_delta_recurrence_ops, rms_norm_gated, DeltaNetCache,
    CHUNKED_PREFILL_MIN_TOKENS, KERNEL_MAX_STEPS,
};
pub use kv_cache::{ContiguousKvCache, KvCache};
pub use moe::{GateUp, MoeRouting, SparseMoe, SwiGlu, SwitchLinear};
pub use nn::{
    conv2d, embed, input_ids, input_ids_batch, layer_norm, linear, rms_norm, rms_norm_unscaled,
    soft_cap,
};
pub use paged_kv_cache::{BlockPool, PagedKvCache};
pub use projection::{KvProjection, Projection, QuantSpec};
pub use quant::QuantizedLinear;
pub use rope::{apply_rope, Rope};
pub use sampler::{sample, shaped_candidates, SamplingParams, SplitMix64, TokenRng};
pub use weights::{Materialized, Weights};

/// Whether `stream` is a GPU stream — the only place a custom Metal kernel can run. Every fused
/// kernel resolves [`Stream::task_local_or_default`](mlx_rs::Stream::task_local_or_default) once,
/// gates on this, and dispatches on that same stream: the task-local stream need not be on the
/// process default device.
pub(crate) fn stream_is_gpu(stream: &mlx_rs::Stream) -> bool {
    // SAFETY: `dev` is created and freed here; `stream` outlives both calls.
    unsafe {
        let mut dev = mlx_sys::mlx_device_new();
        let mut ty: mlx_sys::mlx_device_type = mlx_sys::mlx_device_type__MLX_CPU;
        let ok = mlx_sys::mlx_stream_get_device(&mut dev, stream.as_ptr()) == 0
            && mlx_sys::mlx_device_get_type(&mut ty, dev) == 0;
        mlx_sys::mlx_device_free(dev);
        ok && ty == mlx_sys::mlx_device_type__MLX_GPU
    }
}
