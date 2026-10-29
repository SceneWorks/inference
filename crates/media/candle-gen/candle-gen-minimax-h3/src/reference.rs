//! **`ref2va` multi-modal references**: the ordered, heterogeneous condition list that
//! MiniMax-H3's reference-to-video-and-audio task consumes.
//!
//! Every rule is a property of the model, not of a backend, so the module is backend-neutral and
//! lives in [`candle_gen::gen_core::h3_reference`] (sc-27256), shared with the MLX twin; this module
//! re-exports it so `reference::*` paths stay stable.

pub use candle_gen::gen_core::h3_reference::*;
