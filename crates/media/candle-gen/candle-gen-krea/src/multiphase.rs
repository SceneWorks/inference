//! Multi-phase Krea denoise primitive (epic 13879): one trajectory over one global sigma schedule.
//!
//! The host math is backend-neutral and lives in [`gen_core_family::krea_multiphase`] (sc-27329), shared
//! with the MLX twin; this module re-exports it so `multiphase::*` paths stay stable.

pub use gen_core_family::krea_multiphase::*;
