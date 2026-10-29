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

pub use ltx_checkpoint::{
    caption_feature_version, check_gemma_version, declared_layout, declared_model_version,
    discover_split_bundle, discover_split_bundle_skipping, layout_for_declared_version,
    layout_for_version, parse_model_version, CaptionFeatureVersion, GemmaEncoderIdentity,
    GemmaSourceCheckpoint, GemmaVersionCheck, LtxBundle, LtxBundleBuilder, LtxCheckpointLayout,
    LtxCheckpointMetadata, LtxComponent, LtxConfigRoot, LtxResolvedComponent,
};
