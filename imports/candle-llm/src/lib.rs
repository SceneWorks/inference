//! `candle-llm` — on-device text LLM serving engine (Candle backend).
//!
//! The crate is built bottom-up, mirroring `mlx-llm`'s structure on Candle tensors (epic 7153):
//!
//! 1. [`primitives`] — the backend-owned tensor leaves the engine needs: a batch-capable
//!    [`KvCache`](primitives::KvCache), a pluggable [`sample`](primitives::sample)r, the
//!    [`Rope`](primitives::Rope) family, GQA attention helpers, group-wise quantization (via
//!    Candle's `QTensor`/`QMatMul`), the `nn` leaves, and a safetensors
//!    [`Weights`](primitives::Weights) loader. These own Candle `Tensor`s directly.
//! 2. [`config`] + [`models`] — model configuration ([`LlamaConfig`]) and the generic Llama-family
//!    decoder ([`LlamaModel`]), `&self` forward + `from_weights`, with architecture dispatch
//!    (Llama / Mistral / Qwen3).
//! 3. [`decode`] — the streaming, cancellable decode loop ([`generate`]) that drives any
//!    [`Decode`](decode::Decode) model, emitting a [`StreamEvent`] per token.
//! 4. [`provider`] — implements the backend-neutral [`core_llm::TextLlm`] contract over the engine
//!    and registers it (`candle-llama`), so consumers stream a generation entirely through
//!    `core-llm`. Passing the `core-llm-testkit` conformance suite as a second backend is what
//!    de-provisionalizes the contract (story 7237).
//!
//! Compute runs in `bf16` on the GPU backends (CUDA / Metal) and `f32` on CPU. Candle `Tensor`s are
//! `Send`/`Sync`, so a loaded model is freely shareable across threads.

pub mod config;
pub mod decode;
pub mod device;
pub mod error;
pub mod gguf;
pub mod image;
pub mod llava;
pub mod models;
pub mod primitives;
pub mod provider;

// Re-export the contract crate so consumers can reach it as `candle_llm::core_llm::…`.
pub use core_llm;

pub use config::{Architecture, LlamaConfig, RopeScaling};
pub use decode::{
    generate, generate_batch, generate_cached, generate_draft_speculative, generate_prompt_lookup,
    generate_with, generate_with_cache, BatchRequest, CancelFlag, FinishReason, GenerationConfig,
    GenerationOutput, PrefixCache, PrefixStats, SpeculativeConfig, SpeculativeStats, StreamEvent,
};
pub use device::{compute_dtype, select_device};
pub use error::{Error, Result};
pub use llava::{LlavaConfig, LlavaModel, LlavaProvider};
pub use models::LlamaModel;
pub use provider::LlamaProvider;
