//! Lens resolution buckets (sc-3173) — a faithful port of `_vendor/lens/resolution.py`.
//!
//! The bucket table is backend-neutral and lives in [`mlx_gen::gen_core::lens_resolution`]
//! (sc-27329), shared with the Candle twin; this module re-exports it so `resolution::*` paths
//! stay stable.

pub use mlx_gen::gen_core::lens_resolution::*;
