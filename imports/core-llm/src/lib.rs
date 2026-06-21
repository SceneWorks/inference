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
//! - [`Scheduler`] — backend-neutral continuous-batching policy (admission + per-sequence retire).
//! - [`PrefixIndex`] — backend-neutral shared-prefix KV-reuse policy (longest-match + LRU).
//! - [`BlockAllocator`] — backend-neutral paged-KV block allocation policy (refcounts + free list).
//! - [`registry`] — link-time provider registration and id-based routing.

pub mod cancel;
pub mod capabilities;
pub mod constraint;
pub mod error;
pub mod message;
pub mod output;
pub mod paging;
pub mod prefix;
pub mod registry;
pub mod request;
pub mod schedule;
pub mod template;
pub mod text_llm;
pub mod tokenizer;

pub use cancel::CancelFlag;
pub use capabilities::{TextLlmCapabilities, TextLlmDescriptor};
pub use constraint::{Constraint, ConstraintDecodeTable, JsonConstraint, JsonState};
pub use error::{Error, Result};
pub use message::{Content, ImageRef, Message, Role};
pub use output::{FinishReason, StreamEvent, TextLlmOutput, Usage};
pub use paging::BlockAllocator;
pub use prefix::{InsertOutcome, PrefixId, PrefixIndex, PrefixMatch};
pub use registry::{load_textllm, textllms, TextLlmRegistration};
pub use request::{LoadSpec, Quantize, Sampling, TextLlmRequest};
pub use schedule::{Scheduler, SeqId, SeqSpec};
pub use template::{ChatMlTemplate, ChatTemplate, JinjaChatTemplate, Llama3Template};
pub use text_llm::TextLlm;
pub use tokenizer::Tokenizer;

/// The crate version, surfaced in conformance / diagnostic messages.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
