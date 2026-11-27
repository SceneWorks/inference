//! Primary typed route declarations, one-call registration records, computed `Capabilities`,
//! `HOOK-4` structural applicability derivation, and Tier 0 conformance checks
//! (`SW-6979`: `REG-1`, `REG-2`, `HOOK-4`, `AP5`, `AP12`, `AP18`).
//!
//! A [`RouteRecord`] is the single primary source of truth for one executable route:
//! - its structural components (text encoders, conditioning modules, denoiser/model binding,
//!   latent codec, upsamplers, adapter surface & install policy, residency lifetime graph,
//!   weight plan, execution & approximation domains, preview/output, video/audio surfaces),
//! - its [`ProductLimits`], [`SeedAndBatchSemantics`], and [`RouteDefaults`],
//! - its executable/registry [`RouteBindings`] (generator factory, footprint, render/video plan,
//!   feature sub-route, encoder contract alias, memory contract/fixtures/behavior, activation
//!   memory, and reference workspace).
//!
//! [`crate::ModelDescriptor`], [`crate::Capabilities`], and [`crate::FeatureRouteStructure`] are **computed** from the
//! primary declarations rather than hand-maintained in parallel tables (`AP5`).

mod compute;
mod conformance;
mod types;

pub use conformance::{route_record_conformance_errors, verify_route_conformance};
pub use types::*;

#[cfg(test)]
mod tests;
