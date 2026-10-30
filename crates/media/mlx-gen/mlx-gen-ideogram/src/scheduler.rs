//! Ideogram 4's logit-normal flow-matching schedule + Euler sampler timing.
//!
//! The schedule math is backend-neutral and lives in [`gen_core_family::ideogram_scheduler`]
//! (sc-27256), shared with the Candle twin; this module re-exports it so `scheduler::*` paths stay
//! stable.

pub use gen_core_family::ideogram_scheduler::*;
