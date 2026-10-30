//! Krea 2's rectified-flow (v-parameterization) timestep schedule (reference `sampling.py`).
//!
//! The host math is backend-neutral and lives in [`gen_core_family::krea_schedule`] (sc-27329), shared
//! with the MLX twin; this module re-exports it so `schedule::*` paths stay stable.

pub use gen_core_family::krea_schedule::*;
