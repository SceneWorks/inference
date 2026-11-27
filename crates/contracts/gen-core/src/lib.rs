//! # gen-core
//!
//! The **backend-neutral contract layer** for SceneWorks generative inference. gen-core has
//! **zero tensor dependencies**: it owns the `Generator` / `Trainer` / `Captioner` / `Transform`
//! contracts, the request/output/conditioning/progress/cancel/error types, the explicit provider
//! registry, and the pure host-side policy math (PIL-compatible resize, tiling, LR schedule,
//! audio mixdown + BS.1770-4 loudness/true-peak). The tensor backends — `mlx-gen` (Apple MLX) and the forthcoming `candle-gen`
//! (Windows/CUDA) — implement these contracts and re-export this crate at their own paths.
//! Family- and feature-specific host code (trainer helpers, tokenization, per-family planners,
//! checkpoint readers, EXR/HDR I/O) lives in the sibling `gen-core-*` crates, which depend on this
//! one (sc-27316).
//!
//! Numeric types here are restricted to `f32`/`f64`/`Vec<f32>`/`Vec<i32>`/`&[u8]` — never an
//! `mlx_rs::Array` or candle tensor. See epic 3720 (the unified-contract roadmap, Phase 0).

mod model_input_macros;

#[cfg(feature = "model-inference-proof")]
pub mod model_input_proof;

// Observe only the original successful result; feature OFF is the original
// expression, without a result temporary, new block or reader/control change.
#[cfg(feature = "model-inference-proof")]
macro_rules! __observe_input_success {
    ($value:expr, $observe:expr) => {
        ($value).inspect($observe)
    };
}
#[cfg(not(feature = "model-inference-proof"))]
macro_rules! __observe_input_success {
    ($value:expr, $observe:expr) => {
        ($value)
    };
}
const _: fn() = || {
    let _: std::result::Result<(), ()> = __observe_input_success!(Ok(()), |_| {});
};

pub mod approximation;
pub mod attention_budget;
pub mod audio_dsp;
pub mod audio_embed;
pub mod audio_transform;
pub mod block_window;
pub mod caption;
pub mod checkpoint_codec;
pub mod checkpoint_facts;
pub mod chunk;
pub use chunk::{
    emit_terminal_chunk, BackpressureError, BackpressurePolicy, BoundedChunkSink,
    CausalCodecStatePolicy, ChunkResidencyPolicy, ChunkSink, OutputChunk, TerminalChunkCollector,
};
mod read_accounting;
mod receipt_scope;
pub use read_accounting::CheckpointReadAccounting;
pub use receipt_scope::CheckpointReceiptScope;
pub mod comfy_quant;
pub mod control;
pub mod convrot_int8;
pub mod driver;
pub mod multistream;
pub use driver::{
    run_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver, GlobalRngTrajectoryGate,
    GlobalRngTrajectoryLease, SliceInput, SliceSpec, SliceTransition, StepHooks, StreamObserveHook,
};
pub use multistream::{
    build_multistream_phase_live_set, build_stream_transition_live_set, LearnedUpsamplerComponent,
    MultiStreamStepInfo, PerTokenTimePolicy, ResolvedSchedule, ScheduleSource, ScheduleSourceKind,
    StreamBoundarySemantics, StreamBundleSpec, StreamGeometry, StreamId, StreamLatentDescriptor,
    StreamMemorySpec, StreamSpec, StreamStepSigma, StreamTransition, StreamTransitionKind,
    TransitionBoundaryGuard,
};
pub mod duration_head;
pub mod encoder_contract;
pub mod env_flag;
pub mod error;
pub mod execution_domains;
pub mod face;
pub mod feature;
pub use feature::{
    apply_destination_noise_law_f32, bilinear_half_pixel_clamp_coord, dump_hook_map_json, gcd_u32,
    hook_map, hook_map_for_registry, hook_point_metadata, lcm_u32, low_res_start_legal_quantum,
    low_res_start_transition_noise_f32, low_res_start_transition_seed,
    qualify_low_res_start_estimators, repack_unpacked_latents_f32,
    resize_bilinear_half_pixel_clamp_f32, resolve_img2img_init_time_step,
    resolve_img2img_truncated_schedule, transition_clean_estimate_f32,
    transition_clean_estimate_with_noise_f32, transition_packed_clean_estimate_f32,
    transition_packed_clean_estimate_with_noise_f32, unpack_packed_latents_f32,
    validate_low_res_start_request, BoundaryState, ComposedDenoisePhase, ComposedPassPlan,
    ComposedPhaseTransition, ComposedRenderPlan, EstimateProvenance, EstimateSource,
    EvaluationKind, FeatureApplicability, FeatureClass, FeatureHookBinding,
    FeatureInapplicabilityCode, FeatureInapplicabilityReason, FeatureInvocation,
    FeatureMemoryEffect, FeatureParameterDomain, FeatureParameterSchema, FeatureParameterSpec,
    FeatureParameters, FeatureRegistry, FeatureRelations, FeatureRouteStructure, FeatureSpec,
    HiresFixFeature, HookPoint, HookPointMetadata, HookTiming, InertPhaseSplitFeature,
    KreaMultiphaseFeature, LowResStartExecutionReport, LowResStartFeature,
    LowResStartMemoryProfile, LowResStartQualificationReport, LowResStartQualityMetrics,
    LowResStartSpec, LowResStartTraceSummary, LowResStartWorkProfile, PagFeature,
    Pass0ConditioningRequirement, PhaseTransitionAction, ResolvedLowResStartPlan,
    TrajectoryObserverFeature, TransitionResidency, WanExpertSwitchFeature,
    LOW_RES_START_ENDPOINT_ESTIMATOR_ID, LOW_RES_START_PREVIOUS_ESTIMATOR_ID,
    LOW_RES_START_RECIPE_ID, LOW_RES_START_TRANSITION_ARITHMETIC_ID,
    LOW_RES_START_TRANSITION_DOMAIN,
};
pub mod flux3;
pub mod generator;
pub mod guidance;
pub mod h3_adapter_memory;
pub mod h3_conditioning_memory;
pub mod h3_image;
pub mod h3_memory;
pub mod h3_reference;
pub mod hdr;
mod headwise;
pub mod image_embed;
pub mod imageops;
pub mod json_constraint;
pub mod latent;
pub mod license;
pub mod load_activity;
mod macros;
pub mod media;
pub mod memory_phases;
pub mod memory_strategy;
pub mod moss_image;
mod moss_image_assets;
pub mod moss_image_edit;
pub use headwise::{HeadwiseProjection, HeadwiseWeights};
pub mod numeric;
pub mod phase;
pub mod registry;
pub mod render;
pub use render::video_plan::{
    VideoOutputConversion, VideoPackedComponent, VideoPhaseLayout, VideoPlanLayout,
    VideoPlanRegistration, VideoReferenceLayout, VideoRenderPlan,
};
pub use render::{
    MultiPassRenderPlan, PlannedPassRequest, ProducedReference, RenderLoadBinding, RenderPlan,
    RenderPlanRegistration, RenderRun, RenderStage,
};
pub mod residency;
pub mod route;
pub mod runtime;
pub use load_activity::{
    LoadActivity, LoadActivityContext, LoadActivityFile, LoadActivityObserver, LoadActivityPhase,
    LoadActivityScope,
};
pub mod safetensors_shards;
pub mod sampling;
pub mod text_embed;
pub mod tier_integrity;
pub mod tiling;
pub mod train;
pub mod transcribe;
pub mod transform;
pub mod vision_encoder_contract;
pub mod voice_embed;
pub mod w4a8_int8;
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
    AudioTransformKind, AudioTransformRequest, CompletedAudioTransform,
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
    DENSE_F16_CODEC, DENSE_F32_CODEC, FP8_E4M3_ROWWISE_CODEC, FP8_E4M3_SCALAR_CODEC,
    FP8_E5M2_SCALAR_CODEC, GGUF_CONTAINER_CODEC, INT8_PER_ROW_CODEC, MXFP8_CODEC, NVFP4_CODEC,
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
    remap_mxfp8_scales, validate_mxfp8_geometry, validate_nvfp4_block_scale_payload,
    validate_nvfp4_geometry, ComfyQuantDescriptor, ComfyQuantDescriptorError, ComfyQuantFormat,
    Mxfp8GeometryError, Nvfp4GeometryError, PartialComfyQuantDescriptor, QuantizationMetadataError,
    E2M1_LUT, MXFP8_BLOCK, NVFP4_BLOCK, NVFP4_PAD,
};
pub use control::{
    reject_unknown_components, require_base_dir, require_base_snapshot, require_component,
    require_component_file, require_control, AcceptedControlKinds, ControlBranch,
};
pub use encoder_contract::{
    hf_cache_discovery_roots, read_text_encoder_source_unchanged,
    resolve_encoder_load_time_quant_bits, text_encoder_packed_quant_bits,
    text_encoder_planning_facts_for_discovery, text_encoder_source_bytes, EncoderConfigBool,
    EncoderConfigFloat, EncoderContract, EncoderPackingContract, EncoderPromptExecutionContract,
    EncoderPromptLengthPolicy, EncoderPromptPadding, EncoderPromptTemplate, EncoderRequiredToken,
    EncoderTokenizerBinding, EncoderTokenizerContract, EncoderTokenizerDisposition,
    TextEncoderPlanningFacts, TextEncoderSourceLayout, ValidatedEncoderSource,
    ValidatedTokenizerSource,
};
pub use env_flag::env_flag;
pub use error::{
    run_with_cleanup_policy, run_with_cleanup_policy_with_outcome, CleanupPolicyOutcome, Error,
    ErrorClass, Result,
};
pub use execution_domains::{
    CfgBatching, CfgBatchingDomain, ExecutionSurface, ExecutionValueDomain, FfnChunk,
    GraphEvalCadence, PrefixKvCacheDomain, PrefixKvCacheMode, PrefixKvCacheOverflow,
    PrefixKvCacheReport,
};
pub use face::{DetectedFace, FaceEmbedder, FaceEmbedderDescriptor};
pub use generator::{
    default_seed, effective_component_quant, effective_reference_image_short_edge,
    reject_unsupported_adapters, validate_reference_image_short_edge, ActivationMemoryAnchor,
    ArtifactRecord, AudioArtifacts, AudioEditMode, AudioEditRef, AudioParams, Capabilities,
    CaptionOverflowPolicy, ComponentPrecisionFloor, Conditioning, ConditioningKind, ControlClipRef,
    ControlKind, ConversationRole, ConversationSession, ConversationTurn, GenerationMemory,
    GenerationOutput, GenerationPhase, GenerationReport, GenerationRequest, GenerationWarning,
    Generator, HdrRequest, KeyframeRef, Modality, ModelDescriptor, OutputChannels, OutputLimiter,
    PhaseAdapter, PrecisionFloorComponent, ReplacementMode, SavedPlan, SizeFloor, SongCover,
    SongCoverMode, SongCoverVoice, SongDecoder, SongParams, SongPlanning, SpeechSegment,
    StagedResidencyAvailability, StepSupport, TimeRegion, TokenSampling, VideoClipRef,
    REFERENCE_IMAGE_SHORT_EDGE_DEFAULT, REFERENCE_IMAGE_SHORT_EDGE_MAX,
    REFERENCE_IMAGE_SHORT_EDGE_MIN,
};
pub use h3_reference::{effective_reference_image_resize_mode, ReferenceImageResizeMode};
pub use hdr::{HdrColorSpace, HdrTransfer, Primaries};
pub use image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
pub use json_constraint::JsonState;
pub use latent::{
    latent_spaces_compatible, normalization_vectors_hash, DecoderOption, LatentNormalization,
    LatentNormalizationStats, LatentPatchLayout, LatentSpace, LatentTemporalLaw,
    SpatialCompression, DECODER_OPTIONS, FLUX1_LATENT_SPACE, FLUX2_PACKED_LATENT_SPACE,
    LENS_LATENT_SPACE, LTX_VIDEO_LATENT_SPACE, MAGE_LATENT_SPACE, MOCHI_VIDEO_LATENT_SPACE,
    QWEN_IMAGE_2_1_Z64_LATENT_SPACE, QWEN_IMAGE_2_1_Z64_MEAN, QWEN_IMAGE_2_1_Z64_NORMALIZATION,
    QWEN_IMAGE_2_1_Z64_STD, QWEN_KREA_Z16_LATENT_SPACE, QWEN_WAN_Z16_MEAN,
    QWEN_WAN_Z16_NORMALIZATION, QWEN_WAN_Z16_STD, SANA_LATENT_SPACE, SD3_LATENT_SPACE,
    SDXL_LATENT_SPACE, SEEDVR2_VIDEO_LATENT_SPACE, SVD_LATENT_SPACE, WAN_2_1_VAE_DECODER_ID,
    WAN_Z16_LATENT_SPACE, WAN_Z16_VIDEO_LATENT_SPACE, WAN_Z48_LATENT_SPACE, WAN_Z48_MEAN,
    WAN_Z48_NORMALIZATION, WAN_Z48_STD,
};
pub use license::components::MEDIA_COMPONENT_LICENSES;
pub use license::families::LICENSE_FAMILIES;
pub use license::{
    component_licenses_manifest_json, decoder_license_conformance_errors,
    license_table_conformance_errors, provider_terms, resolve_component, resolve_family,
    CeilingBoundary, ComponentLicense, LicenseFamily, LicenseTerm, ProviderComponents,
};
pub use media::{AudioChunk, AudioStem, AudioTrack, HdrFrame, Image, RgbaImage};
pub use memory_phases::{
    DecoderTilingRealization, DecoderWorkspaceFacts, ImagePipelineArchitecture, MemoryPhaseFacts,
    StagedWeightSchedule, StreamedWeightFacts,
};
pub use memory_strategy::{
    adapter_stack_identity, adapter_stack_resident_bytes, adapter_stack_upcast_resident_bytes,
    admit_memory_safety, checked_byte_total, default_memory_strategy_safety_check,
    default_registered_memory_strategy_safety_check, evaluate_low_res_transition_admission,
    qualify_flux1_low_res_transition_bound, select_equivalent_only_admission,
    standard_memory_behavior_context, standard_memory_strategy_safety_check,
    sum_deduplicated_live_bytes, trace_planned_low_res_scope_lifecycle,
    validate_bounded_decode_params, validate_calibration_fingerprint,
    validate_equivalent_admission_candidate, AdapterResidencyMode, AdmissionAdjustmentCandidate,
    EquivalentAdmissionRefusal, EvidencePeakBasis, LokrKroneckerDims,
    LowResTransitionAdmissionReceipt, LowResTransitionAdmissionRefusal,
    LowResTransitionCoverageTrace, LowResTransitionPricingPosture,
    LowResTransitionSixBufferCeiling, MemoryArchitectureFacts, MemoryAssetFacts, MemoryBackend,
    MemoryBackendRealization, MemoryBehaviorRoute, MemoryBudget, MemoryBudgetDomain,
    MemoryCacheSemantics, MemoryCacheState, MemoryCalibrationIdentity, MemoryCleanupSemantics,
    MemoryComponentKind, MemoryComponentResidency, MemoryConformanceState,
    MemoryDecodeArtifactIdentity, MemoryDecodeGeometryPolicy, MemoryDecodePolicyQuery,
    MemoryDecodeQualityDisposition, MemoryDecodeQualityFixture, MemoryDecodeQualityRuntimeIdentity,
    MemoryDecodeRouteDomain, MemoryEvidence, MemoryEvidenceDimension, MemoryEvidenceDimensions,
    MemoryEvidenceKey, MemoryEvidenceLogRecord, MemoryEvidenceVerdict, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryLowResStartContext,
    MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority, MemoryParameterRanges,
    MemoryParityContract, MemoryParityResult, MemoryPeakBreakdown, MemoryPhase,
    MemoryPidDecodeRoutes, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryReferenceResourceFacts, MemoryReferenceShape, MemoryRejection, MemoryRequestScope,
    MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome, MemoryRuntimeSemantics,
    MemorySafetyDecision, MemorySelection, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategyEngagementExclusion, MemoryStrategyParameters, MemoryStrategyPrerequisite,
    MemoryStrategySupport, MemoryStructuralResidentEvidence,
    MemoryStructuralResidentRequestIdentity, MemoryWarmRunSemantics, MemoryWindowMaterialization,
    PhaseOperationLiveSet, PricedAllocation, PricedDomainPeak, RecordedScopeLifecycleEvent,
    RequestLiveSetPlan, ResidentRequestMemory, ScopeLifecycleRecorder, StreamOutputMode,
    StreamPhaseGeometry, TransformerComponent, TransitionMechanismKind, TransitionOperationLiveSet,
    UpcastLoraCopy, CANDLE_FLUX1_DEV_Q4_1024_CONDITIONAL_LOW_RES_PEAK_BYTES,
    CANDLE_FLUX1_DEV_Q4_1024_ORDINARY_PEAK_BYTES, FLUX1_FULL_RES_1024_FP32_LATENT_BYTES,
    FLUX1_LOW_RES_1024_SIX_BUFFER_CEILING_BYTES, FLUX1_LOW_RES_512_FP32_LATENT_BYTES,
    MEMORY_CALIBRATION_ABI, MEMORY_DECODE_QUALITY_ABI, MEMORY_EVIDENCE_SCHEMA_VERSION,
    MEMORY_EVIDENCE_V1_PREFIX, MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI,
};
pub use memory_strategy::{
    resolved_snapshot_root, standard_architecture_facts, standard_begin_request,
    standard_registered_safety_check, standard_registered_valid_fixture, standard_safety_check,
    standard_validate_context, MemoryArchitectureSource, MemoryFingerprintPin,
    MemoryFingerprintTable, MemoryFixtureMode, MemoryFixtureOverlay, MemoryFixturePhases,
    MemoryFixturePid, MemoryFixtureRoute, MemoryProfileFixtures, MemoryProfileGate,
    MemoryProfileGates, MemoryProfileScope, MemoryProfileTier, MemoryProfileTierResolver,
    MemoryRejectionRendering, ProviderMemoryProfile,
};
pub use registry::{
    candle_memory_contract_surface_specs, candle_nvfp4_memory_contract_surface_specs,
    candle_publisher_memory_contract_surface_specs, mlx_memory_contract_surface_specs,
    ActivationMemoryRegistration, AudioEmbedderRegistration, AudioTransformRegistration,
    CaptionerRegistration, CheckpointAdapterCapabilityRegistration, CheckpointAdapterRegistration,
    CheckpointBackend, CheckpointBackendBindingRegistration,
    CheckpointBaseCompatibilityRegistration, CheckpointCanonicalMappingRegistration,
    CheckpointComponentRegistration, CheckpointConfigRecoveryRegistration,
    CheckpointDialectRegistration, CheckpointLoaderReuseRegistration,
    CheckpointSignatureRegistration, EncoderContractRouteRegistration, FeatureRouteRegistration,
    ImageEmbedderRegistration, ImportedModelCompatibilityProjectionRegistration,
    ImportedModelOperation, ImportedModelRegistration, ImportedModelSource,
    MemoryBehaviorBeginRequest, MemoryBehaviorFixture, MemoryBehaviorRegistration,
    MemoryContractFixtureRegistration, MemoryContractSurface,
    MemoryContractSurfaceResolverRegistration, MemoryContractSurfaceSelector,
    MemoryContractSurfaceSpec, MemoryContractSurfaceTier, MemoryRegistration, ModelRegistration,
    PerComponentBytes, ProviderRegistry, ProviderRegistryBuilder, ReferenceWorkspaceRegistration,
    ResidentOnlyMemoryContractRegistration, TextEmbedderRegistration, TrainerRegistration,
    TranscriberRegistration, TransformRegistration, VoiceEmbedderRegistration, CHECKPOINT_ADAPTERS,
    FLUX2_CHECKPOINT_ADAPTER, KREA_2_CHECKPOINT_ADAPTER, MAGE_FLOW_CHECKPOINT_ADAPTER,
    MOSS_IMAGE_CHECKPOINT_ADAPTER, QWEN_IMAGE_CHECKPOINT_ADAPTER, SDXL_CHECKPOINT_ADAPTER,
    WAN_CHECKPOINT_ADAPTER, Z_IMAGE_CHECKPOINT_ADAPTER,
};
pub use residency::{Residency, ResidencyRuntime, StagedHeavy};
pub use route::{
    route_record_conformance_errors, verify_route_conformance, AdapterInstallPolicy,
    AdapterSurface, AudioSurface, ConditioningSurface, FamilyDriverSpec, FamilyModelSpec,
    LatentCodecDeclaration, ModelBinding, OutputSurface, PreviewSupport, ProductLimits,
    RenderPlanCompiler, ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface, RouteBindings,
    RouteDefaults, RouteFeatureBinding, RouteGeneratorBinding, RouteMemoryBindings, RouteRecord,
    RouteStructure, SamplingSurface, SeedAndBatchSemantics, SeedScheme, TextEncoderDeclaration,
    TextEncoderSurface, UpsamplerSurface, VideoSurface, WeightPlanDeclaration, WeightSourceKind,
};
pub use runtime::{
    with_prompt_encoding, AdapterApplyReport, AdapterKind, AdapterSpec, ArtifactDigestAttestation,
    ArtifactSeal, CancelFlag, FileStatFingerprint, HdrFrameSink, HdrOutputFrame, IdentityWeights,
    LoadPhase, LoadShape, LoadShapeDeclarationResult, LoadSpec, MoeExpert, OffloadPolicy,
    PidWeights, PinnedWeightsFile, Precision, PreparedFilePins, PreviewFrame, PreviewSink,
    Progress, PromptEnhancementOutcome, PromptEnhancementReport, PromptEnhancementSink, Quant,
    ResolvedFileIdentity, SingleFlight, SingleFlightGuard, WeightsSource, Yue2ArMode,
    Yue2ComputePolicy, BASE_SNAPSHOT_COMPONENT, COMFYUI_TEXT_ENCODER_COMPONENT,
    COMFYUI_VAE_COMPONENT, KREA_CONVROT_DIT_COMPONENT, LTX_SPATIAL_UPSCALER_COMPONENT,
    VAE_COMPONENT,
};
pub use text_embed::{TextEmbedder, TextEmbedderDescriptor};
pub use tier_integrity::{control_branch_tier, is_above_selected_tier};
pub use tiling::{TilingConfig, VaeTiling};
pub use vision_encoder_contract::{VisionEncoderArchitecture, VisionEncoderContract};
pub use voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};
pub use weightsmeta::{
    materialized_header_bytes, materialized_path_bytes, read_safetensors_tensor_payloads,
    safetensors_dir_bytes, safetensors_file_metadata, safetensors_file_tensor_locations,
    safetensors_path_bytes, safetensors_path_quantization_metadata,
    safetensors_path_tensor_headers, SafetensorsFileLayout, SafetensorsTensorHeader,
    SafetensorsTensorLocation, CANDLE_DEVICE_FORMAT_CACHE_DIR,
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
    AspectRatioBucket, AuxLossSchedule, BodyLossesConfig, BucketSchedule, DepthAnchoringConfig,
    DepthModelSize, FaceLandmarkLossConfig, IdentityLossConfig, IdentityReferenceMode,
    LatentLpipsConfig, LatentLpipsFamily, LrSchedule, NetworkType, ResolutionBucket,
    SubjectMaskLoss, Trainer, TrainerDescriptor, TrainingConfig, TrainingItem, TrainingOutput,
    TrainingProgress, TrainingReport, TrainingRequest, TrainingTechniques, VaeAnchorConfig,
    BODY_SURFACE_SCHEDULE, LATENT_PERCEPTUAL_SCHEDULE, MAX_RESOLUTION_BUCKETS,
    PIXEL_PERCEPTUAL_SCHEDULE, RESOLUTION_BUCKET_STRIDE,
};
pub use transcribe::{
    TimestampGranularity, TranscribeCapabilities, TranscribeFinishReason, TranscribeOptions,
    TranscribeRequest, TranscribeSampling, TranscribeTask, Transcriber, TranscriberDescriptor,
    TranscriptOutput, TranscriptSegment, TranscriptWord,
};
pub use transform::{
    CompletedTransform, InputBudget, InputSizing, TargetSize, Transform, TransformCapabilities,
    TransformDescriptor, TransformRequest,
};

/// gen-core's package version, for the version-skew runtime guard (sc-4482).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod pag;
pub use guidance::{pag_apply, pag_apply_slice};
pub use pag::{
    resolve_pag, PagCapability, PagConfig, PagExecutionReport, PagPolicyError, PagPolicyErrorKind,
    PagResolved, DEFAULT_PAG_MAX_SCALE,
};

pub mod veda;
pub use veda::{VedaBudget, VedaExecutionReport, VedaRequest};

mod source_inventory;
pub use source_inventory::SourceInventory;

// One snapshot-inventory walker + receipt hashing helpers shared by the Candle providers (sc-27253).
pub mod snapshot_inventory;

mod ios_build_identity;
#[doc(hidden)]
pub use ios_build_identity::IOS_ENGINEERING_FEATURES_SUPPORTED;

/// Durable cache interface shared by native weight quantizers.
pub mod quantization_cache;
pub use quantization_cache::{
    QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey,
    QuantizationCacheContext,
};
