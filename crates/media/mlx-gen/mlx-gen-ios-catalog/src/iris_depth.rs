//! Callable Iris depth surface for a catalog-only consumer.
//!
//! Depth is a provider-specific task, not a gen-core generator. Its composed memory registration
//! is metadata/admission plumbing, not a loader. Loads go through the shared iOS seam
//! ([`crate::ios_admission`]): [`crate::Admission::load_depth`] for an evidence-admitted load, or a
//! `crate::capture_permit` (feature `device-capture`) for a bounded device capture. There is no unadmitted loader.

pub use mlx_gen::gen_core::iris::depth::{
    DepthMap, DepthMetadata, DepthOutput, DepthRequest, DepthResolution, IrisDepthEstimator,
};
pub use mlx_gen_iris::depth::IrisDepth;
