//! `mlx-llm` — on-device text + vision LLM serving engine for Apple MLX.
//!
//! The crate is built bottom-up, mirroring the dependency order of epic 7153:
//!
//! 1. [`primitives`] — the backend-owned tensor leaves the engine needs: a batch-capable
//!    [`KvCache`](primitives::KvCache), a pluggable [`Sampler`](primitives::sampler), the
//!    [`Rope`](primitives::Rope) family, GQA attention helpers, quantization, the `nn` leaves,
//!    and a safetensors [`Weights`](primitives::Weights) loader. These own MLX `Array`s directly
//!    and deliberately depend on nothing from the gen-ai side.
//! 2. [`config`] + [`models`] — model configuration ([`ModelConfig`]) and the
//!    generic Llama decoder ([`CausalLm`]), `&self` forward + `from_weights`.
//! 3. [`decode`] — the streaming, cancellable decode loop ([`generate`]) that
//!    drives any [`Decode`](decode::Decode) model, emitting a [`StreamEvent`]
//!    per token. This internal streaming API is what the backend-neutral `core-llm` contract
//!    (story 7154) is extracted from. [`generate_batch`] packs many requests
//!    into one forward step (story 7167), driven by `core_llm`'s backend-neutral scheduler policy;
//!    [`generate_cached`] reuses a shared prompt prefix's KV across requests
//!    (story 7168), driven by `core_llm`'s backend-neutral prefix-index policy;
//!    [`generate_prompt_lookup`] is n-gram speculative decoding
//!    (story 7171) and [`generate_draft_speculative`] is
//!    draft-model speculative decoding (story 7172), both driven by `core_llm`'s backend-neutral
//!    proposer + distribution-preserving acceptance sampler.
//! 4. [`provider`] — implements the backend-neutral [`core_llm::TextLlm`] contract over the engine
//!    and exposes it (`mlx-llama`) for explicit runtime composition.
//!
//! [`gguf`] is a side door into step 2: it converts a llama.cpp `*.gguf` (incl. k-quants) into the
//! `{config.json, model.safetensors}` snapshot the loader consumes (story 7165), optionally
//! re-quantizing to MLX Q4/Q8.
//!
//! [`snapshot`] is the shared persisted-snapshot writer both ingest paths funnel through
//! ([`snapshot::write_snapshot`], story 7660): it (optionally) re-quantizes the attention/MLP
//! projections and writes the loadable snapshot directory. [`snapshot::write_hf_snapshot`] is the
//! Hugging Face leaf — load a dense HF safetensors directory and persist it as a snapshot, dense or
//! Q4/Q8 (the backend half of `core_llm::SnapshotPreparer`'s HF case).
//!
//! [`joycaption`] is the **text + vision** path (story 7157): a SigLIP vision tower
//! ([`models::SiglipVisionTower`]) + LLaVA projector + image splice in front of the reused
//! [`CausalLm`] decode, served as a multimodal [`core_llm::TextLlm`] provider (`mlx-joycaption`).
//!
//! MLX's default Metal device is single-threaded; engine instances hold MLX `Array`s and are
//! therefore neither `Send` nor `Sync`. Drive one engine from one thread (or behind a mutex).

pub mod config;
pub mod decode;
pub mod error;
pub mod gguf;
pub mod image;
pub mod joycaption;
pub mod models;
pub mod prepare;
pub mod primitives;
pub mod provider;
pub mod residency;
pub mod snapshot;
pub mod starvector_1b;
pub mod starvector_8b;

// Self-removing temp fixtures for the crate's unit suites (sc-17768). This is the SAME file the
// integration suites pull in as `mod common;` — included by path rather than copied, so the two
// contexts cannot drift. `#[cfg(test)]` keeps it out of the shipped surface and `tempfile` in
// `[dev-dependencies]`.
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_fixture;

// Re-export the contract crate so consumers can reach it as `mlx_llm::core_llm::…`.
pub use core_llm;

pub use config::ModelConfig;
pub use decode::{
    generate, generate_batch, generate_cached, generate_cached_with, generate_continuous,
    generate_draft_speculative, generate_prompt_lookup, generate_with_cache, BatchExactness,
    BatchRequest, CancelFlag, ContinuousConfig, FinishReason, GenerationConfig, GenerationOutput,
    PrefixCache, PrefixStats, SpeculativeConfig, SpeculativeStats, StreamEvent,
};
pub use error::{Error, Result};
pub use joycaption::{JoyCaptionModel, JoyCaptionProvider};
pub use models::CausalLm;
pub use provider::LlamaProvider;
pub use residency::{EncoderResidency, StreamObservation};
pub use snapshot::{write_hf_snapshot, write_snapshot, SnapshotReport, SnapshotTokenizer};
pub use starvector_1b::StarVector1bProvider;
pub use starvector_8b::StarVector8bProvider;

/// Add every MLX LLM provider to an explicit registry builder.
pub fn register_text_providers(
    registry: core_llm::TextLlmRegistryBuilder,
) -> core_llm::TextLlmRegistryBuilder {
    registry
        .register(provider::REGISTRATION)
        .register(joycaption::REGISTRATION)
        .register(starvector_1b::REGISTRATION)
        .register(starvector_8b::REGISTRATION)
}

/// Build the complete, explicit MLX LLM provider catalog.
pub fn text_registry() -> core_llm::Result<core_llm::TextLlmRegistry> {
    register_text_providers(core_llm::TextLlmRegistryBuilder::new()).build()
}

/// Add only providers admitted to a shipped runtime catalog.
///
/// StarVector-8B is intentionally absent until terminal sc-22261 records its one real-weight
/// quality/admission receipt. It remains available through [`text_registry`] for explicit local
/// loading and the shared conformance harness; this keeps discovery and catalog eligibility from
/// being conflated.
pub fn register_catalog_text_providers(
    registry: core_llm::TextLlmRegistryBuilder,
) -> core_llm::TextLlmRegistryBuilder {
    registry
        .register(provider::REGISTRATION)
        .register(joycaption::REGISTRATION)
        .register(starvector_1b::REGISTRATION)
}

/// Build the explicit MLX LLM catalog whose providers have admission evidence.
pub fn catalog_text_registry() -> core_llm::Result<core_llm::TextLlmRegistry> {
    register_catalog_text_providers(core_llm::TextLlmRegistryBuilder::new()).build()
}

/// Add the MLX snapshot preparer to an explicit registry builder.
pub fn register_snapshot_preparers(
    registry: core_llm::SnapshotPreparerRegistryBuilder,
) -> core_llm::SnapshotPreparerRegistryBuilder {
    registry.register(prepare::REGISTRATION)
}

/// Build the complete, explicit MLX snapshot-preparer catalog.
pub fn snapshot_preparer_registry() -> core_llm::Result<core_llm::SnapshotPreparerRegistry> {
    register_snapshot_preparers(core_llm::SnapshotPreparerRegistryBuilder::new()).build()
}

/// Load a bundled MLX provider by descriptor id.
pub fn load_textllm(
    id: &str,
    spec: &core_llm::LoadSpec,
) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
    text_registry()?.load_textllm(id, spec)
}

/// Select and load the bundled MLX provider that accepts `spec`.
pub fn load_for_model(spec: &core_llm::LoadSpec) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
    text_registry()?.load_for_model(spec)
}

/// Select and load a bundled MLX provider with explicit capability requirements.
pub fn load_for_model_with(
    spec: &core_llm::LoadSpec,
    requirements: &core_llm::ModelRequirements,
) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
    text_registry()?.load_for_model_with(spec, requirements)
}

/// Prepare a snapshot through the bundled MLX preparer.
pub fn prepare_snapshot(spec: &core_llm::PrepareSpec) -> core_llm::Result<core_llm::PrepareReport> {
    snapshot_preparer_registry()?.prepare_snapshot(spec)
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_registry_keeps_8b_available_but_catalog_admission_fail_closed() {
        let explicit: Vec<String> = super::text_registry()
            .unwrap()
            .registrations()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(
            explicit,
            [
                "mlx-llama",
                "mlx-joycaption",
                "mlx-starvector-1b",
                "mlx-starvector-8b",
            ]
        );

        let admitted: Vec<String> = super::catalog_text_registry()
            .unwrap()
            .registrations()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(
            admitted,
            ["mlx-llama", "mlx-joycaption", "mlx-starvector-1b"],
            "StarVector-8B must stay out of runtime catalogs until sc-22261 admission evidence"
        );

        let preparers = super::snapshot_preparer_registry().unwrap();
        assert_eq!(
            preparers
                .registrations()
                .map(|registration| (registration.backend)())
                .collect::<Vec<_>>(),
            ["mlx"]
        );
    }
}
