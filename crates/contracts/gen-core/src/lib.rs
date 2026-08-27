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

pub mod approximation;
pub mod attention_budget;
pub mod audio_dsp;
pub mod audio_embed;
pub mod audio_transform;
pub mod block_window;
pub mod caption;
pub mod checkpoint_codec;
pub mod checkpoint_facts;
pub mod comfy_quant;
pub mod control;
pub mod duration_head;
pub mod encoder_contract;
pub mod error;
pub mod execution_domains;
pub mod exr_io;
pub mod face;
pub mod gemma_assets;
pub mod generator;
pub mod guidance;
pub mod hdr;
pub mod image_embed;
pub mod imageops;
pub mod json_constraint;
pub mod latent;
pub mod license;
pub mod ltx_checkpoint;
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
pub mod vision_encoder_contract;
pub mod voice_embed;
pub mod wan_i2v_memory;
pub mod weightsmeta;

pub use approximation::{
    ApproximationPlan, ApproximationRequest, ApproximationSurface, CacheReuseInterval,
    CacheWarmupSteps, CharacterizationBinding, CharacterizationFamily, CharacterizationRef,
    FeatureCacheDomain, FeatureCachePolicy, TokenDropStride, TokenPruningDomain,
    TokenPruningPolicy, MIN_DROP_STRIDE, MIN_REUSE_INTERVAL,
};
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
    apply_caption_trigger_words, CaptionCapabilities, CaptionFinishReason, CaptionOptions,
    CaptionOutput, CaptionRequest, CaptionSampling, CaptionTriggerWordConformanceCase, Captioner,
    CaptionerDescriptor, CAPTION_TRIGGER_WORD_CONFORMANCE,
};
pub use checkpoint_codec::{
    compile_logical_weight_plan, compile_logical_weight_plan_with_metadata,
    CheckpointCodecRegistration, CheckpointCodecRegistry, CodecResidencyPolicy,
    CodecResidencyReport, CompanionRole, CompanionTensorPlan, DenseResidencyPolicy,
    IdentityKeyMapping, LogicalKeyMapping, LogicalReadMaterialization, LogicalTensorPlan,
    LogicalWeightPlan, LogicalWeightPlanError, LogicalWeightReceipt, PlannedResidency,
    ResidencyMode, ResidentTensorHeadersError, ScalarScaleSource, StoredTensorFormat,
    TensorCodecSpec, WeightEncoding, COMFY_QUANT_CODECS, DENSE_BF16_CODEC, DENSE_CODECS,
    DENSE_F16_CODEC, DENSE_F32_CODEC, FP8_E4M3_SCALAR_CODEC, FP8_E5M2_SCALAR_CODEC,
    GGUF_CONTAINER_CODEC, INT8_PER_ROW_CODEC, MXFP8_CODEC, NVFP4_CODEC,
};
pub use checkpoint_facts::{
    CheckpointFactsSink, CheckpointWeightFacts, CheckpointWeightFactsError,
    ExecutionRepresentation, NativeExecutionCapability, SourceBinding, SourceCodecEntry,
    SourceCodecSummary,
};
pub use comfy_quant::{
    blocked_scale_index, blocked_scale_shape, decode_fp8_e4m3fn_scalar, decode_fp8_e5m2_scalar,
    decode_int8_per_row, decode_mxfp8, decode_nvfp4, descriptor_from_json, e2m1_to_f32,
    e8m0_to_f32, fp8_e4m3fn_to_f32, fp8_e5m2_to_f32, mxfp8_padded_shape, mxfp8_scale_shape,
    mxfp8_swizzled_scale_index, nvfp4_padded_shape, nvfp4_scale_shape, nvfp4_swizzled_scale_index,
    parse_comfy_quant_descriptor, parse_quantization_metadata, partial_descriptor_from_json,
    validate_mxfp8_geometry, validate_nvfp4_block_scale_payload, validate_nvfp4_geometry,
    ComfyQuantDescriptor, ComfyQuantDescriptorError, ComfyQuantFormat, Mxfp8GeometryError,
    Nvfp4GeometryError, PartialComfyQuantDescriptor, QuantizationMetadataError, E2M1_LUT,
    MXFP8_BLOCK, NVFP4_BLOCK, NVFP4_PAD,
};
pub use control::{
    reject_unknown_components, require_base_dir, require_base_snapshot, require_component,
    require_component_file, require_control, AcceptedControlKinds, ControlBranch,
};
pub use encoder_contract::{
    read_text_encoder_source_unchanged, text_encoder_packed_quant_bits, text_encoder_source_bytes,
    EncoderConfigBool, EncoderConfigFloat, EncoderContract, EncoderPackingContract,
    EncoderPromptExecutionContract, EncoderPromptLengthPolicy, EncoderPromptPadding,
    EncoderPromptTemplate, EncoderRequiredToken, EncoderTokenizerBinding, EncoderTokenizerContract,
    EncoderTokenizerDisposition, ValidatedEncoderSource, ValidatedTokenizerSource,
};
pub use error::{Error, Result};
pub use execution_domains::{
    CfgBatching, CfgBatchingDomain, ExecutionSurface, ExecutionValueDomain, FfnChunk,
    GraphEvalCadence,
};
pub use face::{DetectedFace, FaceEmbedder, FaceEmbedderDescriptor};
pub use generator::{
    default_seed, effective_component_quant, reject_unsupported_adapters, ActivationMemoryAnchor,
    AudioEditMode, AudioEditRef, AudioParams, Capabilities, ComponentPrecisionFloor, Conditioning,
    ConditioningKind, ControlClipRef, ControlKind, ConversationRole, ConversationSession,
    ConversationTurn, GenerationMemory, GenerationOutput, GenerationPhase, GenerationRequest,
    Generator, KeyframeRef, Modality, ModelDescriptor, PhaseAdapter, PrecisionFloorComponent,
    ReplacementMode, SizeFloor, SpeechSegment, StagedResidencyAvailability, StepSupport,
    TimeRegion, VideoClipRef,
};
pub use image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
pub use json_constraint::JsonState;
pub use latent::{
    latent_spaces_compatible, normalization_vectors_hash, DecoderOption, LatentNormalization,
    LatentNormalizationStats, LatentPatchLayout, LatentSpace, LatentTemporalLaw,
    SpatialCompression, DECODER_OPTIONS, FLUX1_LATENT_SPACE, FLUX2_PACKED_LATENT_SPACE,
    LTX_VIDEO_LATENT_SPACE, MAGE_LATENT_SPACE, MOCHI_VIDEO_LATENT_SPACE,
    QWEN_KREA_Z16_LATENT_SPACE, QWEN_WAN_Z16_MEAN, QWEN_WAN_Z16_NORMALIZATION, QWEN_WAN_Z16_STD,
    SANA_LATENT_SPACE, SD3_LATENT_SPACE, SDXL_LATENT_SPACE, SEEDVR2_VIDEO_LATENT_SPACE,
    SVD_LATENT_SPACE, WAN_2_1_VAE_DECODER_ID, WAN_Z16_LATENT_SPACE, WAN_Z16_VIDEO_LATENT_SPACE,
    WAN_Z48_LATENT_SPACE, WAN_Z48_MEAN, WAN_Z48_NORMALIZATION, WAN_Z48_STD,
};
pub use license::components::MEDIA_COMPONENT_LICENSES;
pub use license::families::LICENSE_FAMILIES;
pub use license::{
    component_licenses_manifest_json, decoder_license_conformance_errors,
    license_table_conformance_errors, provider_terms, resolve_component, resolve_family,
    CeilingBoundary, ComponentLicense, LicenseFamily, LicenseTerm, ProviderComponents,
};
pub use media::{AudioChunk, AudioStem, AudioTrack, Image};
pub use memory_strategy::{
    adapter_stack_identity, adapter_stack_resident_bytes, default_memory_strategy_safety_check,
    default_registered_memory_strategy_safety_check, standard_memory_behavior_context,
    standard_memory_strategy_safety_check, validate_calibration_fingerprint, AdapterResidencyMode,
    MemoryAssetFacts, MemoryBackend, MemoryBackendRealization, MemoryBehaviorRoute, MemoryBudget,
    MemoryCacheSemantics, MemoryCacheState, MemoryCalibrationIdentity, MemoryCleanupSemantics,
    MemoryComponentKind, MemoryComponentResidency, MemoryConformanceState,
    MemoryDecodeArtifactIdentity, MemoryDecodeGeometryPolicy, MemoryDecodePolicyQuery,
    MemoryDecodeQualityDisposition, MemoryDecodeQualityFixture, MemoryDecodeQualityRuntimeIdentity,
    MemoryDecodeRouteDomain, MemoryEvidence, MemoryEvidenceDimension, MemoryEvidenceDimensions,
    MemoryEvidenceKey, MemoryEvidenceLogRecord, MemoryEvidenceVerdict, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryOptimizationAuthority, MemoryParameterRanges, MemoryParityContract,
    MemoryParityResult, MemoryPeakBreakdown, MemoryPhase, MemoryPidDecodeRoutes,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryReferenceShape, MemoryRejection,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyEngagementExclusion, MemoryStrategyParameters,
    MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryStructuralResidentEvidence,
    MemoryStructuralResidentRequestIdentity, MemoryWarmRunSemantics, MemoryWindowMaterialization,
    ResidentRequestMemory, TransformerComponent, MEMORY_CALIBRATION_ABI, MEMORY_DECODE_QUALITY_ABI,
    MEMORY_EVIDENCE_SCHEMA_VERSION, MEMORY_EVIDENCE_V1_PREFIX,
    MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
};
pub use registry::{
    candle_memory_contract_surface_specs, candle_nvfp4_memory_contract_surface_specs,
    mlx_memory_contract_surface_specs, ActivationMemoryRegistration, AudioEmbedderRegistration,
    AudioTransformRegistration, CaptionerRegistration, CheckpointAdapterCapabilityRegistration,
    CheckpointAdapterRegistration, CheckpointBackend, CheckpointBackendBindingRegistration,
    CheckpointBaseCompatibilityRegistration, CheckpointCanonicalMappingRegistration,
    CheckpointComponentRegistration, CheckpointConfigRecoveryRegistration,
    CheckpointDialectRegistration, CheckpointSignatureRegistration,
    EncoderContractRouteRegistration, ImageEmbedderRegistration,
    ImportedModelCompatibilityProjectionRegistration, ImportedModelOperation,
    ImportedModelRegistration, ImportedModelSource, MemoryBehaviorBeginRequest,
    MemoryBehaviorFixture, MemoryBehaviorRegistration, MemoryContractFixtureRegistration,
    MemoryContractSurface, MemoryContractSurfaceResolverRegistration,
    MemoryContractSurfaceSelector, MemoryContractSurfaceSpec, MemoryContractSurfaceTier,
    MemoryRegistration, ModelRegistration, PerComponentBytes, ProviderRegistry,
    ProviderRegistryBuilder, ResidentOnlyMemoryContractRegistration, TextEmbedderRegistration,
    TrainerRegistration, TranscriberRegistration, TransformRegistration, VoiceEmbedderRegistration,
    FLUX2_CHECKPOINT_ADAPTER, KREA_2_CHECKPOINT_ADAPTER, MAGE_FLOW_CHECKPOINT_ADAPTER,
    QWEN_IMAGE_CHECKPOINT_ADAPTER, SDXL_CHECKPOINT_ADAPTER, WAN_CHECKPOINT_ADAPTER,
    Z_IMAGE_CHECKPOINT_ADAPTER,
};
pub use residency::{Residency, ResidencyRuntime, StagedHeavy};
pub use runtime::{
    AdapterApplyReport, AdapterKind, AdapterSpec, CancelFlag, FileStatFingerprint, IdentityWeights,
    LoadPhase, LoadShape, LoadShapeDeclarationResult, LoadSpec, MoeExpert, OffloadPolicy,
    PidWeights, PinnedWeightsFile, Precision, PreparedFilePins, PreviewFrame, PreviewSink,
    Progress, PromptEnhancementOutcome, PromptEnhancementReport, PromptEnhancementSink, Quant,
    WeightsSource, BASE_SNAPSHOT_COMPONENT, COMFYUI_TEXT_ENCODER_COMPONENT, COMFYUI_VAE_COMPONENT,
    KREA_CONVROT_DIT_COMPONENT, LTX_SPATIAL_UPSCALER_COMPONENT, VAE_COMPONENT,
};
pub use text_embed::{TextEmbedder, TextEmbedderDescriptor};
pub use tier_integrity::{control_branch_tier, is_above_selected_tier};
pub use tiling::{TilingConfig, VaeTiling};
pub use vision_encoder_contract::{VisionEncoderArchitecture, VisionEncoderContract};
pub use voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};
pub use weightsmeta::{
    read_safetensors_tensor_payloads, safetensors_dir_bytes, safetensors_file_metadata,
    safetensors_file_tensor_locations, safetensors_path_bytes,
    safetensors_path_quantization_metadata, safetensors_path_tensor_headers, SafetensorsFileLayout,
    SafetensorsTensorHeader, SafetensorsTensorLocation,
};
// The LTX split-checkpoint component resolver (sc-18757), shared verbatim by mlx-gen-ltx and
// candle-gen-ltx: layout selection keyed on `model_version`, per-component config isolation, and the
// `gemma_source_checkpoint` ⇄ text-encoder `gemma_version` assertion. Pure paths + JSON, so it keeps
// gen-core's zero-tensor invariant.
pub use ltx_checkpoint::{
    caption_feature_version, check_gemma_version, declared_layout, declared_model_version,
    discover_split_bundle, discover_split_bundle_skipping, layout_for_declared_version,
    layout_for_version, parse_model_version, CaptionFeatureVersion, GemmaEncoderIdentity,
    GemmaSourceCheckpoint, GemmaVersionCheck, LtxBundle, LtxBundleBuilder, LtxCheckpointLayout,
    LtxCheckpointMetadata, LtxComponent, LtxConfigRoot, LtxResolvedComponent,
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
