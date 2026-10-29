//! Ideogram 4's logit-normal flow-matching schedule + Euler sampler timing.
//!
//! The schedule math is backend-neutral and lives in [`candle_gen::gen_core::ideogram_scheduler`]
//! (sc-27256), shared with the MLX twin; this module re-exports it so `scheduler::*` paths stay
//! stable.

pub use candle_gen::gen_core::ideogram_scheduler::*;
