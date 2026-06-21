//! Streaming, cancellable decoding.
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to be
//! driven by it. [`StreamEvent`]s are emitted per token through a callback. The Candle port of
//! `mlx-llm`'s `decode` module.

pub mod batch;
pub mod cancel;
pub mod stream;

pub use batch::{generate_batch, BatchRequest};
pub use cancel::CancelFlag;
pub use stream::{
    generate, generate_with, ConstraintMask, Decode, FinishReason, GenerationConfig,
    GenerationOutput, StreamEvent,
};
