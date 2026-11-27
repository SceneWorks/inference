//! # gen-core-ltx
//!
//! The tensor-free LTX helpers that only the LTX providers need, split out of gen-core in
//! sc-27316: the split-checkpoint component resolver (sc-18757: layout selection keyed on
//! `model_version`, per-component config isolation, and the `gemma_source_checkpoint` ⇄
//! text-encoder `gemma_version` assertion), the Gemma encoder assets, the DFR keyframe policy and
//! the duration head. The auto-duration range a request carries stays in
//! `gen_core::duration_head` and is re-exported from [`duration_head`].

pub mod duration_head;
pub mod gemma_assets;
pub mod ltx_checkpoint;
pub mod ltx_dfr;
pub mod ltx_plan;

pub use ltx_plan::{
    compute_audio_frames, ltx_dev_sigmas, LtxConditioningProfile, LtxDomainMemorySummary,
    LtxMemoryFootprintSpec, LtxModelFamily, LtxOrchestrationRoute, LtxRenderPlan,
    LtxRenderPlanInput, LtxStagePlanSpec, LtxStageScheduleEntry, LtxVariant, MemoryBudgetDomainKey,
    LTX_AUDIO_LATENT_CHANNELS, LTX_AUDIO_MEL_BINS, LTX_DEV_STEPS, LTX_LATENT_CHANNELS,
    LTX_MAX_FRAMES, LTX_NATIVE_DISTILLED_STEPS, LTX_SIZE_MULTIPLE, LTX_SPATIAL_SCALE,
    LTX_SPATIAL_UPSAMPLER_COMPONENT_NAME, LTX_STAGE1_SIGMAS, LTX_STAGE2_SIGMAS, LTX_STAGE2_STEPS,
    LTX_TEMPORAL_SCALE, LTX_TEMPORAL_SIGMAS, LTX_TEMPORAL_UPSAMPLER_COMPONENT_NAME,
    LTX_UPSAMPLER_NORMALIZATION,
};

pub use ltx_checkpoint::{
    caption_feature_version, check_gemma_version, declared_layout, declared_model_version,
    discover_split_bundle, discover_split_bundle_skipping, layout_for_declared_version,
    layout_for_version, parse_model_version, CaptionFeatureVersion, GemmaEncoderIdentity,
    GemmaSourceCheckpoint, GemmaVersionCheck, LtxBundle, LtxBundleBuilder, LtxCheckpointLayout,
    LtxCheckpointMetadata, LtxComponent, LtxConfigRoot, LtxResolvedComponent,
};
