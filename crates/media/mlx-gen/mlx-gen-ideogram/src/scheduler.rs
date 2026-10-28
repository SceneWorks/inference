//! Ideogram 4's logit-normal flow-matching schedule + Euler sampler timing.
//!
//! The schedule math is backend-neutral and lives in [`mlx_gen::gen_core::ideogram_scheduler`]
//! (sc-27256), shared with the Candle twin; this module re-exports it so `scheduler::*` paths stay
//! stable.

pub use mlx_gen::gen_core::ideogram_scheduler::*;
