//! Concrete model decoders built on the [`crate::primitives`].
//!
//! The generic Llama decoder uses an **immutable `&self` forward** and a `from_weights`
//! constructor, so a single loaded model can be shared and driven concurrently in the batch
//! dimension later. The only mutable state in a forward pass is the KV cache, threaded in as
//! `&mut dyn KvCache`.

pub mod llama;

pub use llama::LlamaModel;
