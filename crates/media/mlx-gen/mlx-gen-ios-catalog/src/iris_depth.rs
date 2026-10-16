//! Callable Iris depth surface for a catalog-only consumer.
//!
//! Depth is a provider-specific task, not a gen-core generator. Its composed memory registration
//! is metadata/admission plumbing, not a loader. This module exposes the actual task without a
//! direct dependency on `mlx-gen-iris` and without exposing that crate's training/other loaders.
//!
//! No Iris device footprint has been captured. The diagnostic loader is explicitly unadmitted;
//! registration and safetensors headers are not device-fit evidence. A product must not treat
//! this diagnostic entry point as an admitted device load.

pub use mlx_gen::gen_core::iris::depth::{
    DepthMap, DepthMetadata, DepthOutput, DepthRequest, DepthResolution, IrisDepthEstimator,
};
pub use mlx_gen_iris::depth::IrisDepth;

/// Load the Iris depth export for explicitly unadmitted diagnostics. Every Iris spec validation
/// still applies, but this does not authorize a device footprint or bypass a safe admitted facade.
pub fn load_depth_unadmitted_for_diagnostics(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::Result<IrisDepth> {
    mlx_gen_iris::depth::load_depth(spec)
}
