//! # candle-gen-iris
//!
//! The **Iris-3B** provider crate for [`candle-gen`](candle_gen) (epic sc-25678, story sc-25680) —
//! the Candle (Windows/Linux CUDA) sibling of `mlx-gen-iris` (sc-25679) and a native port of the
//! frozen upstream `speridlabs/iris-3b` text-to-image pipeline: Qwen3-VL-4B conditioning over 12
//! intermediate layers, the layerwise text adapter, the hybrid dual-/single-stream pixel-space DiT
//! with its PiT pixel head, and the FlowDPM-Solver++ on the shifted rectified-flow schedule —
//! registered as [`MODEL_ID`] (`iris_3b`). No Python runtime path.
//!
//! ## Tasks and resources (E4)
//!
//! The backend-neutral task identity, resource layout, `config.yaml` reader, conditioning window,
//! solver plan and control surface live in [`gen_core::iris`] (shared with the MLX twin and the
//! SceneWorks worker). The generation task is **backbone + text encoder**: `LoadSpec::weights` is
//! the backbone directory and `LoadSpec::components["text_encoder"]` the Qwen3-VL snapshot.
//! [`IrisDiT`] and [`load_backbone`] never touch the text encoder, so the depth and restoration
//! tasks reuse them with their own (encoder-free) conditioning.
//!
//! ## Frozen upstream and parity
//!
//! The coverage table, revisions, fixtures and tolerances live in the MLX twin's `UPSTREAM.md`
//! (`crates/media/mlx-gen/mlx-gen-iris/UPSTREAM.md`), which carries a Candle column. This crate's
//! parity tests read the **same** committed fixtures the MLX twin reads
//! (`crates/media/mlx-gen/mlx-gen-iris/tests/fixtures/`, produced by `tools/dump_iris_*.py` from the
//! pinned revisions), so both backends are held to one numeric reference.
//!
//! ## Training (sc-25686)
//!
//! [`train`] registers the `iris_3b` [`Trainer`](candle_gen::gen_core::train::Trainer): full
//! training (random or weights init) and LoRA / LoKr adapters over the one backend-neutral contract
//! `gen_core::iris::train` the MLX trainer reads, with its checkpoint layout, resume semantics,
//! artifact schemas and metadata stamps. Adapters are forward-time residuals in training **and** in
//! [`load_backbone_with_adapters`], so a training preview renders exactly what the exported file
//! renders here; artifacts load across backends (`tests/train_cross_backend.rs`).
//!
//! ## Memory strategy (sc-25690)
//!
//! [`memory_strategy`] publishes one request-memory contract per inference task through the shared
//! ladder seams. Generation stages its text encoder per request (rung 1) and bounds its attention
//! scores (rung 3); depth and restoration are one resident backbone forward, and restoration's
//! 50 %-overlap tiling is an unconditional part of that forward rather than a selectable rung.
//!
//! ## Deliberate differences from the MLX twin
//!
//! * `backend = "candle"`, `mac_only = false`.
//! * Noise is drawn from the shared launch-portable CPU `StdRng` (`candle_gen::seed`), not MLX's
//!   RNG, so a seed reproduces within a backend but not across the two (upstream's `torch.randn`
//!   matches neither; parity is measured with injected noise on both).
//!
//! [`gen_core::iris`]: candle_gen::gen_core::iris

pub mod adapters;
pub mod depth;
pub mod dit;
pub mod memory_strategy;
pub mod model;
pub mod nn;
pub mod pipeline;
pub mod restoration;
pub mod solver;
pub mod text_encoder;
pub mod train;

pub use candle_gen::gen_core::iris::{
    IrisTask, TEXT_ENCODER_COMPONENT, TEXT_ENCODER_REPO, TEXT_ENCODER_REVISION,
    UPSTREAM_CODE_REVISION, UPSTREAM_WEIGHTS_REPO, UPSTREAM_WEIGHTS_REVISION,
};
pub use dit::{IrisDiT, TextBatch};
pub use model::{
    compute_dtype, descriptor, load, load_backbone, load_backbone_with_adapters, Iris3b, MODEL_ID,
};
pub use pipeline::{
    denoise, encode, noise, noise_batch, preview_image, to_image, to_images, Conditioning,
};
pub use restoration::IrisRestorer;
pub use text_encoder::{IrisTextEncoder, TextConditioning};

/// Add the Candle Iris-3B generator, its generation (sc-25686) and depth (sc-25687) trainers and the
/// restoration transform to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    register_memory_contract_surfaces(
        registry
            .register_generator(model::REGISTRATION)
            .register_trainer(train::REGISTRATION)
            .register_trainer(train::depth::REGISTRATION)
            .register_trainer(train::restore::REGISTRATION)
            .register_transform(restoration::REGISTRATION),
    )
}

/// The shared-ladder registrations of the three inference tasks (sc-25690): strategy, weights-free
/// contract fixture and behavior for each. Generation pairs with its registered generator; depth
/// (an `IrisDepthEstimator`) and restoration (a `Transform`) are not gen-core generators, so they
/// use the composed-route seam. Split out the way the sibling Candle providers do, so a catalog can
/// append the contract surface to a registry that already carries the generator.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR_REGISTRATION)
        .register_composed_memory_strategy(memory_strategy::DEPTH_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::DEPTH_MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::DEPTH_MEMORY_BEHAVIOR_REGISTRATION)
        .register_composed_memory_strategy(memory_strategy::RESTORE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::RESTORE_MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::RESTORE_MEMORY_BEHAVIOR_REGISTRATION)
}

/// Build the complete explicit Candle Iris provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    /// The shared coverage table (in the MLX twin's crate) records the pinned revisions and carries
    /// the Candle column this crate is accountable to.
    #[test]
    fn upstream_md_records_the_pinned_revisions_and_the_candle_column() {
        let doc = include_str!("../../../mlx-gen/mlx-gen-iris/UPSTREAM.md");
        for sha in [
            super::UPSTREAM_CODE_REVISION,
            super::UPSTREAM_WEIGHTS_REVISION,
            super::TEXT_ENCODER_REVISION,
        ] {
            assert!(doc.contains(sha), "UPSTREAM.md must record {sha}");
        }
        assert!(
            doc.contains("candle-gen-iris"),
            "UPSTREAM.md must carry the Candle column"
        );
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let ids: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, ["iris_3b"]);
        let transforms: Vec<_> = registry
            .transforms()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(transforms, ["iris_3b_restore"]);
        let trainers: Vec<_> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(trainers, ["iris_3b", "iris_3b_depth", "iris_3b_restore"]);
        assert!(registry.descriptor_conformance_errors().is_empty());
    }
}
