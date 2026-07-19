//! `candle-llm` — on-device text LLM serving engine (Candle backend).
//!
//! The crate is built bottom-up, mirroring `mlx-llm`'s structure on Candle tensors (epic 7153):
//!
//! 1. [`primitives`] — the backend-owned tensor leaves the engine needs: a batch-capable
//!    [`KvCache`](primitives::KvCache), a pluggable [`sample`](primitives::sample)r, the
//!    [`Rope`](primitives::Rope) family, GQA attention helpers, group-wise quantization (via
//!    Candle's `QTensor`/`QMatMul`), the `nn` leaves, and a safetensors
//!    [`Weights`](primitives::Weights) loader. These own Candle `Tensor`s directly.
//! 2. [`config`] + [`models`] — model configuration ([`ModelConfig`]) and the generic Llama-family
//!    decoder ([`CausalLm`]), `&self` forward + `from_weights`, with architecture dispatch
//!    (Llama / Mistral / Qwen3).
//! 3. [`decode`] — the streaming, cancellable decode loop ([`generate`]) that drives any
//!    [`Decode`](decode::Decode) model, emitting a [`StreamEvent`] per token.
//! 4. [`provider`] — implements the backend-neutral [`core_llm::TextLlm`] contract over the engine
//!    and exposes it (`candle-llama`) for explicit runtime composition. Passing the
//!    `core-llm-testkit` conformance suite as a second backend is what
//!    de-provisionalizes the contract (story 7237).
//! 5. [`prepare`] — exposes a [`core_llm::SnapshotPreparerRegistration`]: convert an HF snapshot
//!    or a `*.gguf` into a persisted, loadable snapshot, optionally baking in Q4/Q8 (story 7662).
//!
//! Compute runs in `bf16` on the GPU backends (CUDA / Metal) and `f32` on CPU. Candle `Tensor`s are
//! `Send`/`Sync`, so a loaded model is freely shareable across threads.

pub mod config;
pub mod decode;
pub mod device;
pub mod error;
pub mod gguf;
pub mod image;
pub mod llava;
pub mod models;
pub mod prepare;
pub mod primitives;
pub mod provider;

// Re-export the contract crate so consumers can reach it as `candle_llm::core_llm::…`.
pub use core_llm;

pub use config::{Architecture, ModelConfig, RopeScaling};
pub use decode::{
    generate, generate_batch, generate_cached, generate_draft_speculative, generate_prompt_lookup,
    generate_with, generate_with_cache, BatchRequest, CancelFlag, FinishReason, GenerationConfig,
    GenerationOutput, PrefixCache, PrefixStats, SpeculativeConfig, SpeculativeStats, StreamEvent,
};
pub use device::{compute_dtype, select_device};
pub use error::{Error, Result};
pub use llava::{LlavaConfig, LlavaModel, LlavaProvider};
pub use models::CausalLm;
pub use provider::LlamaProvider;

/// Add every Candle LLM provider to an explicit registry builder.
pub fn register_text_providers(
    registry: core_llm::TextLlmRegistryBuilder,
) -> core_llm::TextLlmRegistryBuilder {
    registry
        .register(provider::REGISTRATION)
        .register(llava::REGISTRATION)
}

/// Build the complete, explicit Candle LLM provider catalog.
pub fn text_registry() -> core_llm::Result<core_llm::TextLlmRegistry> {
    register_text_providers(core_llm::TextLlmRegistryBuilder::new()).build()
}

/// Add the Candle snapshot preparer to an explicit registry builder.
pub fn register_snapshot_preparers(
    registry: core_llm::SnapshotPreparerRegistryBuilder,
) -> core_llm::SnapshotPreparerRegistryBuilder {
    registry.register(prepare::REGISTRATION)
}

/// Build the complete, explicit Candle snapshot-preparer catalog.
pub fn snapshot_preparer_registry() -> core_llm::Result<core_llm::SnapshotPreparerRegistry> {
    register_snapshot_preparers(core_llm::SnapshotPreparerRegistryBuilder::new()).build()
}

/// Load a bundled Candle provider by descriptor id.
pub fn load_textllm(
    id: &str,
    spec: &core_llm::LoadSpec,
) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
    text_registry()?.load_textllm(id, spec)
}

/// Select and load the bundled Candle provider that accepts `spec`.
pub fn load_for_model(spec: &core_llm::LoadSpec) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
    text_registry()?.load_for_model(spec)
}

/// Select and load a bundled Candle provider with explicit capability requirements.
pub fn load_for_model_with(
    spec: &core_llm::LoadSpec,
    requirements: &core_llm::ModelRequirements,
) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
    text_registry()?.load_for_model_with(spec, requirements)
}

/// Prepare a snapshot through the bundled Candle preparer.
pub fn prepare_snapshot(spec: &core_llm::PrepareSpec) -> core_llm::Result<core_llm::PrepareReport> {
    snapshot_preparer_registry()?.prepare_snapshot(spec)
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_is_complete_and_stable() {
        let explicit: Vec<String> = super::text_registry()
            .unwrap()
            .registrations()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(explicit, ["candle-llama", "candle-llava"]);

        let preparers = super::snapshot_preparer_registry().unwrap();
        assert_eq!(
            preparers
                .registrations()
                .map(|registration| (registration.backend)())
                .collect::<Vec<_>>(),
            ["candle"]
        );
    }
}
