//! `core-llm` — the backend-neutral contract, host policy, and explicit provider registry for an on-device
//! LLM serving engine.
//!
//! This crate is deliberately **tensor-free** and **gen-ai-free**: it builds standalone on Linux,
//! Windows, and macOS, and depends on nothing from any tensor backend or image-generation stack.
//! Tensor backends — [`mlx-llm`](https://github.com/SceneWorks/mlx-llm) (Apple MLX) and
//! [`candle-llm`](https://github.com/SceneWorks/candle-llm) (Candle) — implement [`TextLlm`] and
//! expose registrations that a runtime bundle composes through the [`registry`]; consumers select a
//! provider and stream a generation entirely through this contract.
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
//! - [`IncrementalDetok`] — backend-neutral streaming-detokenization delta guard (holds back
//!   lossy U+FFFD placeholders for multi-byte characters split across BPE tokens).
//! - [`ThinkingSegmenter`] — backend-neutral reasoning/answer segmentation (`<think>…</think>`),
//!   paired with the [`ThinkingMode`] request control and `supports_thinking` capability. Qwen-specific
//!   `reasoning_effort` and `preserve_thinking` controls require their own advertised capabilities.
//! - [`ToolSpec`] / [`ToolCall`] / [`ToolCallSegmenter`] — backend-neutral tool ("function") calling:
//!   offered tools render into the chat template (`tools` context), and the model's `<tool_call>`
//!   output (Qwen3.6 XML or JSON/Hermes) is parsed back into structure; paired with the request
//!   [`tools`](TextLlmRequest::tools) field and the `supports_tools` capability.
//! - [`Scheduler`] — backend-neutral continuous-batching policy (admission + per-sequence retire).
//! - [`PrefixIndex`] — backend-neutral shared-prefix KV-reuse policy (longest-match + LRU).
//! - [`BlockAllocator`] — backend-neutral paged-KV block allocation policy (refcounts + free list).
//! - [`speculative`] — backend-neutral speculative-decoding policy (n-gram proposer + distribution-
//!   preserving acceptance sampler).
//! - [`registry`] — explicit provider composition, id-based routing, and **model-first** resolution
//!   ([`TextLlmRegistry::load_for_model`] / [`ModelRequirements`] over a weightless `can_load`
//!   probe).
//! - [`SnapshotPreparerRegistry::prepare_snapshot`] — persisted, backend-neutral snapshot
//!   preparation: turn a downloaded HF-safetensors or GGUF source into a loadable,
//!   optionally-quantized snapshot, delegating tensor work to a selected backend.

pub mod cancel;
pub mod capabilities;
pub mod constraint;
pub mod detok;
pub mod error;
pub mod message;
pub mod output;
pub mod paging;
pub mod prefix;
pub mod prepare;
pub mod prism;
pub mod registry;
pub mod request;
pub mod resource;
pub mod schedule;
pub mod speculative;
pub mod starvector;
pub mod stop;
pub mod template;
pub mod text_llm;
pub mod thinking;
pub mod tokenizer;
pub mod tool;

pub use cancel::CancelFlag;
pub use capabilities::{
    ModelSamplingDefaults, MtpCapabilities, TextLlmCapabilities, TextLlmDescriptor,
};
pub use constraint::{
    Constraint, ConstraintDecodeTable, ConstraintKind, JsonConstraint, JsonState,
};
pub use detok::IncrementalDetok;
pub use error::{Error, RequestResourceExhausted, Result};
pub use message::{AudioRef, Content, ImageRef, Message, Role, VideoRef};
pub use output::{
    Channel, FinishReason, GenerationTimings, MtpStats, StreamEvent, TextLlmOutput, Usage,
};
pub use paging::BlockAllocator;
pub use prefix::{InsertOutcome, PrefixId, PrefixIndex, PrefixMatch};
pub use prepare::{
    detect_format, ModelFormat, PrepareReport, PrepareSpec, SnapshotPreparerRegistration,
    SnapshotPreparerRegistry, SnapshotPreparerRegistryBuilder,
};
pub use prism::{
    apply_hadamard_forward_in_place, apply_hadamard_inverse_in_place, decode_block_into,
    gdn_reorder_last_axis_in_place, gguf_weight_name, is_gdn_ssm_out_weight,
    normalized_fwht_in_place, transcode_block_to_affine, GdnLayout, PrismError,
    PrismHadamardMetadata, PrismPackedKind, PrismPackedMatrixRef, PrismTransformRole,
    PrismWeightTransform, PRISM_AFFINE_WORDS_PER_BLOCK, PRISM_GROUP_SIZE, PRISM_PQ2_0_GGML_TYPE,
    PRISM_PTQ1_0_GGML_TYPE,
};
pub use registry::{
    ModelRequirements, TextLlmRegistration, TextLlmRegistry, TextLlmRegistryBuilder,
};
pub use request::{
    LoadSpec, MtpMode, Quantize, ReasoningEffort, Sampling, TextLlmRequest, ThinkingMode,
};
pub use resource::{
    admit_request_memory, admit_request_memory_with_geometry, available_host_memory_bytes,
    checkpoint_payload_bytes, checkpoint_staging_bytes, effective_memory_budget,
    estimate_chunked_request_bytes, estimate_request_bytes, operational_memory_override,
    LlmMemoryGeometry, AVAILABLE_MEMORY_OVERRIDE,
};
pub use schedule::{Scheduler, SeqId, SeqSpec};
pub use speculative::{
    accept_greedy_run, accept_token, greedy_commit, ngram_propose, resolve_mtp_plan, Acceptance,
    MtpPlan, ProposerKind,
};
pub use starvector::{
    generated_token_budget, validate_advertised_generated_token_cap,
    validate_generated_token_budget, DecoderArchitecture, ImagePreprocessing, ProjectionMetadata,
    StarVectorBoundedStream, StarVectorDescriptor, StarVectorFinishReason, StarVectorOutput,
    StarVectorProvider, StarVectorRequest, StarVectorStreamEvent, StarVectorStreamStatus,
    StarVectorTier, VisionEncoderArchitecture,
};
pub use stop::{StopChunk, StopMatcher};
pub use template::{
    ChatMlTemplate, ChatTemplate, JinjaChatTemplate, Llama3Template, RenderOptions,
};
pub use text_llm::TextLlm;
pub use thinking::{ThinkingSegmenter, ThinkingSpan};
pub use tokenizer::{Tokenizer, TokenizerDecodeStream};
pub use tool::{ToolCall, ToolCallSegmenter, ToolSpec};

/// The crate version, surfaced in conformance / diagnostic messages.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
