//! Multi-phase Krea denoise primitive (epic 13879): one trajectory over one global sigma schedule.
//!
//! The host math is backend-neutral and lives in [`mlx_gen::gen_core::krea_multiphase`] (sc-27329), shared
//! with the Candle twin; this module re-exports it so `multiphase::*` paths stay stable.

pub use mlx_gen::gen_core::krea_multiphase::*;
