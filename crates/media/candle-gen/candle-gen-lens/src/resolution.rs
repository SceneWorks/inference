//! Lens resolution buckets (sc-5115) — a faithful port of `_vendor/lens/resolution.py`.
//!
//! The bucket table is backend-neutral and lives in [`gen_core_family::lens_resolution`]
//! (sc-27329), shared with the MLX twin; this module re-exports it so `resolution::*` paths
//! stay stable.

pub use gen_core_family::lens_resolution::*;
