//! Streaming, cancellable decoding.
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to be
//! driven by it. [`StreamEvent`]s are emitted per token through a callback. The Candle port of
//! `mlx-llm`'s `decode` module.

pub mod batch;
pub mod cancel;
pub mod prefix;
pub mod stream;

pub use batch::{generate_batch, BatchRequest};
pub use cancel::CancelFlag;
pub use prefix::{generate_cached, PrefixCache, PrefixStats};
pub use stream::{
    generate, generate_with, generate_with_cache, ConstraintMask, Decode, FinishReason,
    GenerationConfig, GenerationOutput, StreamEvent,
};
