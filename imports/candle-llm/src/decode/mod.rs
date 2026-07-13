//! Streaming, cancellable decoding.
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to be
//! driven by it. [`StreamEvent`]s are emitted per token through a callback. The Candle port of
//! `mlx-llm`'s `decode` module.

pub mod batch;
pub mod cancel;
pub mod continuous;
pub mod prefix;
pub mod speculative;
pub mod stream;

pub use batch::{generate_batch, BatchRequest};
pub use cancel::CancelFlag;
pub use continuous::{generate_continuous, BatchExactness, ContinuousConfig};
pub use prefix::{generate_cached, PrefixCache, PrefixStats};
pub use speculative::{
    generate_draft_speculative, generate_prompt_lookup, SpeculativeConfig, SpeculativeStats,
};
pub use stream::{
    generate, generate_from_prefill, generate_with, generate_with_cache, ConstraintMask, Decode,
    FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
