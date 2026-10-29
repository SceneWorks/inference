//! The Qwen-Image 2.1 flow-match schedule (resolution-shifted base, saved-sigma Turbo).
//!
//! The schedule math is backend-neutral and lives in
//! [`mlx_gen::gen_core::qwen_image_2_1::scheduler`] (sc-27329), shared with the Candle twin; this
//! module re-exports it so `scheduler::*` paths stay stable.

pub use mlx_gen::gen_core::qwen_image_2_1::scheduler::*;
