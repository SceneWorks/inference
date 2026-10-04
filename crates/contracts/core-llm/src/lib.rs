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
//! - [`defaults`] — the per-backend decode defaults table (epic sc-24432 E5): every decode
//!   optimization's default on MLX / Candle CUDA / Candle Metal / Candle CPU, with its
//!   justification; [`switch`] is the process switch the backends' runtime toggles share.
//! - [`speculative`] — backend-neutral speculative-decoding policy (n-gram proposer + distribution-
//!   preserving acceptance sampler).
//! - [`report`] — backend-neutral evidence a product renders: [`DecodeReport`] (which decode path
//!   served a generation, on [`TextLlmOutput::decode`]), [`LoadReport`] (what a load produced, via
//!   [`TextLlm::load_report`]) and [`BackendCapabilities`] (what the host can serve — NVFP4, CUDA
//!   graphs — with the refusal reason when it cannot).
//! - [`kv_compression`] — the opt-in compressed-KV policy ([`KvCompressionPolicy`] on
//!   [`TextLlmRequest::kv_compression`]), its one qualification table
//!   ([`KV_COMPRESSION_QUALIFICATIONS`]) and the per-generation [`KvCacheReport`] on
//!   [`TextLlmOutput::kv_cache`], with a [`KvCacheFallbackReason`] whenever the cache ran dense.
//! - [`registry`] — explicit provider composition, id-based routing, and **model-first** resolution
//!   ([`TextLlmRegistry::load_for_model`] / [`ModelRequirements`] over a weightless `can_load`
//!   probe).
//! - [`SnapshotPreparerRegistry::prepare_snapshot`] — persisted, backend-neutral snapshot
//!   preparation: turn a downloaded HF-safetensors or GGUF source into a loadable,
//!   optionally-quantized snapshot, delegating tensor work to a selected backend.

pub mod cancel;
pub mod capabilities;
pub mod constraint;
pub mod defaults;
pub mod detok;
pub mod error;
pub mod kv_compression;
pub mod message;
pub mod mtp_head;
pub mod output;
pub mod paging;
pub mod prefix;
pub mod prepare;
pub mod prism;
pub mod registry;
pub mod report;
pub mod request;
pub mod resource;
pub mod schedule;
pub mod speculative;
pub mod starvector;
pub mod stop;
pub mod switch;
pub mod template;
pub mod text_llm;
pub mod thinking;
pub mod tokenizer;
pub mod tool;

pub use cancel::CancelFlag;
pub use capabilities::{
    ModelSamplingDefaults, MtpCapabilities, ProposerCapabilities, TextLlmCapabilities,
    TextLlmDescriptor,
};
pub use constraint::{
    Constraint, ConstraintDecodeTable, ConstraintKind, JsonConstraint, JsonState,
};
pub use defaults::{speculative_default, DecodeBackend, DecodeDefaults, RecommendedDepths};
pub use detok::IncrementalDetok;
pub use error::{Error, RequestResourceExhausted, Result};
pub use kv_compression::{
    compressed_kv_cache_bytes, kv_model_family, plan_kv_cache, plan_kv_cache_without_reader,
    qualified_kv_model_family, qualify_kv_compression, CompressedKvAllocation, KvAttentionGeometry,
    KvCacheCounters, KvCacheFallbackReason, KvCachePlan, KvCacheReport, KvCacheRequest,
    KvCacheShape, KvCompressionFormat, KvCompressionPolicy, KvModelArchitecture, KvModelFamily,
    KvQualification, KV_CACHE_FORMAT_VERSION, KV_COMPRESSION_QUALIFICATIONS,
    KV_FUSED_READER_HEAD_DIMS, LLAMA_3_2_3B_ARCHITECTURE, QWEN3_1_7B_ARCHITECTURE,
};
pub use message::{AudioRef, Content, ImageRef, Message, Role, VideoRef};
pub use mtp_head::{
    admit_companion_head, check_companion_unused, companion_head_fallback,
    companion_head_payload_bytes, companion_mtp_prefix, native_mtp_plan, read_companion_mtp_config,
    CompanionMtpGeometry, NativeMtp, COMPANION_MTP_ALREADY_NATIVE, COMPANION_MTP_DROPPED,
    COMPANION_MTP_FAMILY_REFUSAL, COMPANION_MTP_MODEL_TYPE, COMPANION_MTP_MOE_REFUSAL,
};
pub use output::{
    Channel, FinishReason, GenerationTimings, MtpStats, StreamEvent, TextLlmOutput, Usage,
};
pub use paging::BlockAllocator;
pub use prefix::{
    prefix_cache_budget, prefix_path_before_lookup, requested_prefix_cache_bytes, InsertOutcome,
    PrefixAdmission, PrefixHit, PrefixId, PrefixIndex, PrefixInsert, PrefixMatch, PrefixReuse,
    PrefixStats, PrefixStore, PREFIX_COMPRESSED_IMPORT_DECLINED, PREFIX_COMPRESSED_NOT_STORED,
    PREFIX_COPY_FAILED, PREFIX_MULTIMODAL_BYPASS, PREFIX_NOT_ADMITTED,
    PREFIX_PAGED_NOT_SNAPSHOTTED,
};
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
pub use report::{
    prefix_budget_fallback, BackendCapabilities, CudaGraphsReport, DecodeReport, DraftReport,
    FeatureSupport, FusedTally, LoadReport, PathReport, ProjectionReport,
};
pub use request::{
    HostSampleReason, LoadSpec, MtpMode, Quantize, ReasoningEffort, SamplerPath, Sampling,
    Speculative, SpeculativeProposer, TextLlmRequest, ThinkingMode,
};
pub use resource::{
    admit_draft_load, admit_load_memory, admit_request_memory, admit_request_memory_with_geometry,
    available_host_memory_bytes, checkpoint_payload_bytes, checkpoint_staging_bytes,
    effective_memory_budget, estimate_chunked_request_bytes,
    estimate_chunked_request_bytes_with_recurrent_copies, estimate_request_bytes,
    estimate_request_bytes_with_recurrent_copies, estimate_tiled_request_bytes_with_kv_bytes,
    estimate_tiled_request_bytes_with_recurrent_copies, operational_memory_override,
    tiled_prefill_activation_bytes, LlmMemoryGeometry, AVAILABLE_MEMORY_OVERRIDE,
};
pub use schedule::{Scheduler, SeqId, SeqSpec};
pub use speculative::{
    accept_greedy_run, accept_token, decode_clock, demotion_threshold, draft_compatibility,
    draft_load_refusal, draft_model_capabilities, draft_refusal, draft_tokenizer_refusal,
    draft_unpriced_refusal, fit_draft_context, greedy_commit, ngram_propose, no_proposer_fallback,
    prompt_lookup_capabilities, resolve_speculative, settle_draft, verify_depth_bound,
    with_decode_clock, Acceptance, AcceptanceMonitor, DecodeClock, DemotionBasis, MonitorDecision,
    PlainDecode, ProposerKind, SpeculativePlan, SpeculativeResolution, StepObservation, WallClock,
    ACCEPTANCE_PROBE_VERIFIES, CANDLE_MTP_DEMOTE_BELOW_PER_EXTRA_DRAFT,
    CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_AT_ONE_DRAFT,
    CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_PER_EXTRA_DRAFT, CAPTIONER_NO_PREFIX_CACHE,
    CAPTIONER_NO_PROPOSER, CLEAR_LOSS_CONFIDENCE_Z, CLEAR_LOSS_GAIN, CLEAR_LOSS_MIN_TIMED_STEPS,
    DRAFT_MODEL_NOT_LOADED, DRAFT_MODEL_RECOMMENDED_DEPTH, MEASURED_GAIN_MARGIN,
    MIN_TIMED_WINDOW_STEPS, MTP_DEMOTE_BELOW_AT_ONE_DRAFT, MTP_DEMOTE_BELOW_PER_EXTRA_DRAFT,
    MTP_RECOMMENDED_DEPTH, PLAIN_PROBE_MAX_STEPS, PLAIN_PROBE_STEPS, PROMPT_LOOKUP_DEMOTE_BELOW,
    PROMPT_LOOKUP_RECOMMENDED_DEPTH, SHAPE_WARMUP_STEPS,
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
