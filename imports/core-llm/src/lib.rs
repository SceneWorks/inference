//! `core-llm` — the backend-neutral contract, host policy, and provider registry for an on-device
//! LLM serving engine.
//!
//! This crate is deliberately **tensor-free** and **gen-ai-free**: it builds standalone on Linux,
//! Windows, and macOS, and depends on nothing from any tensor backend or image-generation stack.
//! Tensor backends — [`mlx-llm`](https://github.com/SceneWorks/mlx-llm) (Apple MLX) and
//! [`candle-llm`](https://github.com/SceneWorks/candle-llm) (Candle) — implement [`TextLlm`] and
//! register through the [`registry`]; consumers select a provider and stream a generation entirely
//! through this contract.
//!
//! The contract was **extracted from the working mlx-llm engine** (epic 7153, story 7154), not
//! designed in a vacuum, and is provisional until `candle-llm` validates it.
//!
//! # Surface
//! - [`TextLlm`] — the streaming, cancellable, multimodal (text + vision) provider trait.
//! - [`TextLlmRequest`] / [`Message`] / [`Content`] — the multimodal, multi-turn request model.
//! - [`StreamEvent`] / [`TextLlmOutput`] / [`Usage`] / [`FinishReason`] — streaming + result types.
//! - [`Sampling`] — backend-neutral sampling policy.
//! - [`Constraint`] + [`JsonState`] — constrained-decoding policy (generic JSON grammar).
//! - [`Tokenizer`] + [`ChatTemplate`] — host-side text policy.
//! - [`StopMatcher`] — backend-neutral request stop-string matching over decoded text.
//! - [`ThinkingSegmenter`] — backend-neutral reasoning/answer segmentation (`<think>…</think>`),
//!   paired with the [`ThinkingMode`] request control and `supports_thinking` capability.
//! - [`ToolSpec`] / [`ToolCall`] / [`ToolCallSegmenter`] — backend-neutral tool ("function") calling:
//!   offered tools render into the chat template (`tools` context), and the model's `<tool_call>`
//!   output (Qwen3.6 XML or JSON/Hermes) is parsed back into structure; paired with the request
//!   [`tools`](TextLlmRequest::tools) field and the `supports_tools` capability.
//! - [`Scheduler`] — backend-neutral continuous-batching policy (admission + per-sequence retire).
//! - [`PrefixIndex`] — backend-neutral shared-prefix KV-reuse policy (longest-match + LRU).
//! - [`BlockAllocator`] — backend-neutral paged-KV block allocation policy (refcounts + free list).
//! - [`speculative`] — backend-neutral speculative-decoding policy (n-gram proposer + distribution-
//!   preserving acceptance sampler).
//! - [`registry`] — link-time provider registration, id-based routing, and **model-first**
//!   resolution ([`load_for_model`] / [`ModelRequirements`] over a weightless `can_load` probe).
//! - [`prepare_snapshot`] — persisted, backend-neutral snapshot preparation: turn a downloaded
//!   HF-safetensors or GGUF source into a loadable, optionally-quantized snapshot, delegating the
//!   tensor work to the linked backend ([`PrepareSpec`] / [`SnapshotPreparerRegistration`]).

pub mod cancel;
pub mod capabilities;
pub mod constraint;
pub mod error;
pub mod message;
pub mod output;
pub mod paging;
pub mod prefix;
pub mod prepare;
pub mod registry;
pub mod request;
pub mod schedule;
pub mod speculative;
pub mod stop;
pub mod template;
pub mod text_llm;
pub mod thinking;
pub mod tokenizer;
pub mod tool;

pub use cancel::CancelFlag;
pub use capabilities::{TextLlmCapabilities, TextLlmDescriptor};
pub use constraint::{Constraint, ConstraintDecodeTable, JsonConstraint, JsonState};
pub use error::{Error, Result};
pub use message::{Content, ImageRef, Message, Role};
pub use output::{Channel, FinishReason, StreamEvent, TextLlmOutput, Usage};
pub use paging::BlockAllocator;
pub use prefix::{InsertOutcome, PrefixId, PrefixIndex, PrefixMatch};
pub use prepare::{
    detect_format, prepare_snapshot, snapshot_preparers, ModelFormat, PrepareReport, PrepareSpec,
    SnapshotPreparerRegistration,
};
pub use registry::{
    load_for_model, load_for_model_with, load_textllm, textllms, ModelRequirements,
    TextLlmRegistration,
};
pub use request::{LoadSpec, Quantize, Sampling, TextLlmRequest, ThinkingMode};
pub use schedule::{Scheduler, SeqId, SeqSpec};
pub use speculative::{accept_greedy_run, accept_token, ngram_propose, Acceptance};
pub use stop::{StopChunk, StopMatcher};
pub use template::{ChatMlTemplate, ChatTemplate, JinjaChatTemplate, Llama3Template, RenderOptions};
pub use text_llm::TextLlm;
pub use thinking::{ThinkingSegmenter, ThinkingSpan};
pub use tokenizer::Tokenizer;
pub use tool::{ToolCall, ToolCallSegmenter, ToolSpec};

/// The crate version, surfaced in conformance / diagnostic messages.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
