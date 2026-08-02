//! # gen-core
//!
//! The **backend-neutral contract layer** for SceneWorks generative inference. gen-core has
//! **zero tensor dependencies**: it owns the `Generator` / `Trainer` / `Captioner` / `Transform`
//! contracts, the request/output/conditioning/progress/cancel/error types, the explicit provider
//! registry, and the pure host-side policy math (tokenization, PIL-compatible resize, tiling,
//! LR schedule, audio mixdown + BS.1770-4 loudness/true-peak). The tensor backends — `mlx-gen` (Apple MLX) and the forthcoming `candle-gen`
//! (Windows/CUDA) — implement these contracts and re-export this crate at their own paths.
//!
//! Numeric types here are restricted to `f32`/`f64`/`Vec<f32>`/`Vec<i32>`/`&[u8]` — never an
//! `mlx_rs::Array` or candle tensor. See epic 3720 (the unified-contract roadmap, Phase 0).

pub mod attention_budget;
pub mod audio_dsp;
pub mod audio_embed;
pub mod audio_transform;
pub mod block_window;
pub mod caption;
pub mod control;
pub mod error;
pub mod face;
pub mod generator;
pub mod guidance;
pub mod image_embed;
pub mod imageops;
pub mod json_constraint;
pub mod license;
mod macros;
pub mod media;
pub mod memory_strategy;
pub mod registry;
pub mod residency;
pub mod runtime;
pub mod sampling;
pub mod sdxl_ldm;
pub mod text_embed;
pub mod tier_integrity;
pub mod tiling;
pub mod tokenizer;
pub mod train;
pub mod transcribe;
pub mod transform;
pub mod voice_embed;
pub mod weightsmeta;

pub use audio_dsp::{
    db_to_linear, measure_loudness, measure_track_loudness, mixdown, LoudnessStats, MixClip,
    MixRequest, SILENCE_FLOOR_LUFS,
};
pub use audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
pub use audio_transform::{
    AudioTarget, AudioTransform, AudioTransformCapabilities, AudioTransformDescriptor,
    AudioTransformKind, AudioTransformRequest,
};
pub use caption::{
    CaptionCapabilities, CaptionFinishReason, CaptionOptions, CaptionOutput, CaptionRequest,
    CaptionSampling, Captioner, CaptionerDescriptor,
};
pub use control::{
    reject_unknown_components, require_base_dir, require_component, require_control,
    AcceptedControlKinds, ControlBranch,
};
pub use error::{Error, Result};
pub use face::{DetectedFace, FaceEmbedder, FaceEmbedderDescriptor};
pub use generator::{
    default_seed, effective_component_quant, ActivationMemoryAnchor, AudioEditMode, AudioEditRef,
    AudioParams, Capabilities, ComponentPrecisionFloor, Conditioning, ConditioningKind,
    ControlClipRef, ControlKind, ConversationRole, ConversationSession, ConversationTurn,
    GenerationMemory, GenerationOutput, GenerationPhase, GenerationRequest, Generator, KeyframeRef,
    Modality, ModelDescriptor, PhaseAdapter, PrecisionFloorComponent, ReplacementMode, SizeFloor,
    SpeechSegment, TimeRegion, VideoClipRef,
};
pub use image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
pub use json_constraint::JsonState;
pub use license::components::MEDIA_COMPONENT_LICENSES;
pub use license::families::LICENSE_FAMILIES;
pub use license::{
    component_licenses_manifest_json, license_table_conformance_errors, provider_terms,
    resolve_component, resolve_family, CeilingBoundary, ComponentLicense, LicenseFamily,
    LicenseTerm, ProviderComponents,
};
pub use media::{AudioChunk, AudioStem, AudioTrack, Image};
pub use memory_strategy::{
    adapter_stack_resident_bytes, default_memory_strategy_safety_check,
    default_registered_memory_strategy_safety_check, standard_memory_behavior_context,
    standard_memory_strategy_safety_check, validate_calibration_fingerprint, AdapterResidencyMode,
    MemoryAssetFacts, MemoryBackend, MemoryBackendRealization, MemoryBehaviorRoute, MemoryBudget,
    MemoryCacheSemantics, MemoryCacheState, MemoryCalibrationIdentity, MemoryCleanupSemantics,
    MemoryComponentKind, MemoryConformanceState, MemoryDecodeRouteDomain, MemoryEvidence,
    MemoryEvidenceDimension, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryEvidenceLogRecord,
    MemoryEvidenceVerdict, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges,
    MemoryParityContract, MemoryParityResult, MemoryPeakBreakdown, MemoryPhase,
    MemoryPidDecodeRoutes, MemoryPrerequisiteScope, MemoryProviderContract, MemoryRejection,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyEngagementExclusion, MemoryStrategyParameters,
    MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryWarmRunSemantics,
    MemoryWindowMaterialization, ResidentRequestMemory, TransformerComponent,
    MEMORY_CALIBRATION_ABI, MEMORY_EVIDENCE_V1_PREFIX,
};
pub use registry::{
    ActivationMemoryRegistration, AudioEmbedderRegistration, AudioTransformRegistration,
    CaptionerRegistration, ImageEmbedderRegistration, MemoryBehaviorBeginRequest,
    MemoryBehaviorFixture, MemoryBehaviorRegistration, MemoryContractFixtureRegistration,
    MemoryRegistration, ModelRegistration, PerComponentBytes, ProviderRegistry,
    ProviderRegistryBuilder, TextEmbedderRegistration, TrainerRegistration,
    TranscriberRegistration, TransformRegistration, VoiceEmbedderRegistration,
};
pub use residency::{Residency, ResidencyRuntime, StagedHeavy};
pub use runtime::{
    AdapterApplyReport, AdapterKind, AdapterSpec, CancelFlag, IdentityWeights, LoadPhase,
    LoadShape, LoadSpec, MoeExpert, OffloadPolicy, PidWeights, Precision, PreviewFrame,
    PreviewSink, Progress, Quant, WeightsSource,
};
pub use text_embed::{TextEmbedder, TextEmbedderDescriptor};
pub use tier_integrity::{control_branch_tier, is_above_selected_tier};
pub use tiling::{TilingConfig, VaeTiling};
pub use voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};
pub use weightsmeta::{
    safetensors_dir_bytes, safetensors_path_bytes, safetensors_path_tensor_headers,
    SafetensorsTensorHeader,
};

// The independent LLM-serving library, re-exported at `gen_core::core_llm` (epic 7153, sc-7189). The
// dependency is INVERTED: gen-core CONSUMES `core-llm` — the same way mlx-gen re-exports gen-core via
// `pub use ::gen_core` — so a consumer that already pins gen-core reaches the unified
// `core_llm::TextLlm` engine (and `core_llm::load_for_model` model-first resolution) through this one
// path, with no separate core-llm pin. core-llm is itself tensor-free, preserving gen-core's invariant.
//
// `core_llm::TextLlm` is now the SOLE text-LLM contract: the legacy `gen_core::TextLlm` trait + its
// `load_textllm`/`TextLlmRegistration` registry plumbing were removed in sc-7189 Phase 3 once every
// provider had migrated (prompt-refine mac sc-7158 / candle sc-7404; JoyCaption sc-7265 / candle
// sc-7692). All text-LLM serving — including model-first resolution via `core_llm::load_for_model` —
// goes through this path.
pub use ::core_llm;
// NOTE: `TrainOptimizer` is intentionally NOT re-exported here — it wraps an mlx-rs optimizer and
// lives in mlx-gen (`mlx_gen::train::optim`). `LrSchedule` is pure policy and lives here.
pub use train::{
    LrSchedule, NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingItem,
    TrainingOutput, TrainingProgress, TrainingRequest,
};
pub use transcribe::{
    TimestampGranularity, TranscribeCapabilities, TranscribeFinishReason, TranscribeOptions,
    TranscribeRequest, TranscribeSampling, TranscribeTask, Transcriber, TranscriberDescriptor,
    TranscriptOutput, TranscriptSegment, TranscriptWord,
};
pub use transform::{
    TargetSize, Transform, TransformCapabilities, TransformDescriptor, TransformRequest,
};

/// gen-core's package version, for the version-skew runtime guard (sc-4482).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
