//! # mlx-gen-iris
//!
//! The **Iris-3B** provider crate for [`mlx-gen`](mlx_gen) (epic sc-25678, story sc-25679): a
//! native MLX port of the frozen upstream `speridlabs/iris-3b` text-to-image pipeline — Qwen3-VL-4B
//! conditioning over 12 intermediate layers, the layerwise text adapter, the hybrid
//! dual-/single-stream pixel-space DiT with its PiT pixel head, and the FlowDPM-Solver++ on the
//! shifted rectified-flow schedule — registered as [`MODEL_ID`] (`iris_3b`).
//!
//! ## Tasks and resources (E4)
//!
//! The backend-neutral task identity, resource layout, `config.yaml` reader, conditioning window
//! and solver plan live in [`gen_core_iris`] (shared with a Candle twin and the SceneWorks
//! worker). The generation task is **backbone + text encoder**: `LoadSpec::weights` is the backbone
//! directory and `LoadSpec::components["text_encoder"]` the Qwen3-VL snapshot. [`IrisDiT`] and
//! [`load_backbone`] never touch the text encoder, so the depth and restoration tasks reuse them
//! with their own (encoder-free) conditioning.
//!
//! The monocular-depth task ([`depth`], [`depth::DEPTH_MODEL_ID`] = `iris_3b_depth`, sc-25682) is a
//! provider-specific API (like the catalog's other depth estimator), not a registry generator: it
//! returns a float32 relative-log-depth map, not an image. [`depth::load`] resolves only the
//! `depth/` export and returns a `gen_core_iris::depth::IrisDepthEstimator`.
//!
//! ## Quantized tiers (sc-26266)
//!
//! `iris_3b` loads at bf16 (the release's autocast policy) and at the affine `q8` / `q4` tiers,
//! which are quantized **at load** from the same dense release — there is no converted artifact.
//! A tier is a whole-pipeline contract: the Qwen3-VL tower's attention/MLP projections
//! ([`IrisTextEncoder::load_quantized`]) and every backbone projection ([`load_backbone_quantized`],
//! [`nn::ProjectionLayout`]) pack at the selected width. Embeddings, norms, the text position table
//! and the sub-group-wide pixel heads stay dense. The depth and restoration tasks and every trainer
//! ship the dense tier only.
//!
//! ## Frozen upstream
//!
//! See `UPSTREAM.md` (revisions, resource layout, the generation coverage table, fixtures and
//! tolerances). The committed fixtures under `tests/fixtures/` were produced by
//! `tools/dump_iris_*.py` from the pinned revisions on miniature configs.
//!
//! [`gen_core_iris`]: gen_core_iris

pub mod adapters;
pub mod depth;
pub mod dit;
pub mod ios_pricing;
pub mod memory_strategy;
pub mod model;
pub mod nn;
pub mod packed;
pub mod pipeline;
pub mod restoration;
pub(crate) mod retention;
pub mod solver;
pub mod stream;
pub mod text_encoder;
#[cfg(feature = "training")]
pub mod train;

#[cfg(all(test, target_os = "macos"))]
mod anchor_tests;
#[cfg(all(test, target_os = "macos"))]
mod packed_tests;
#[cfg(all(test, target_os = "macos"))]
mod quantum_proofs;
#[cfg(all(test, target_os = "macos", debug_assertions))]
mod quantum_tests;
#[cfg(test)]
mod reachable_sizes;
#[cfg(all(test, target_os = "macos"))]
mod stream_tests;
#[cfg(all(test, target_os = "macos"))]
#[path = "../tests/common/metal_guard.rs"]
pub(crate) mod test_metal_guard;

pub use dit::{Execution, IrisDiT, TextBatch};
pub use gen_core_iris::{
    IrisTask, TEXT_ENCODER_COMPONENT, TEXT_ENCODER_REPO, TEXT_ENCODER_REVISION,
    UPSTREAM_CODE_REVISION, UPSTREAM_WEIGHTS_REPO, UPSTREAM_WEIGHTS_REVISION,
};
pub use model::{
    compute_dtype, descriptor, load, load_backbone, load_backbone_quantized,
    load_backbone_with_adapters, Iris3b, MODEL_ID,
};
pub use nn::{BackboneQuant, ProjectionLayout, QUANT_GROUP_SIZE};
pub use pipeline::{
    denoise, encode, noise, noise_batch, preview_image, to_image, to_images, Conditioning,
};
pub use restoration::IrisRestorer;
pub use text_encoder::{
    text_encoder_quant_group, text_encoder_quant_spec, IrisTextEncoder, TextConditioning,
};

/// Add the MLX Iris-3B generator, its generation (sc-25685) and depth (sc-25687) trainers and the
/// restoration transform to an explicit media registry builder, with the memory-strategy contract,
/// weights-free contract fixture and behavior seam of each of the three inference tasks
/// (sc-25690, [`memory_strategy`]).
///
/// `iris_3b` is a registered generator, so its contract registers through the ordinary seam. The
/// depth task is a provider-specific API and restoration is a transform — neither is a gen-core
/// [`Generator`](mlx_gen::Generator) — so their contracts register as composed routes, the seam
/// gen-core provides for a real route that no generator registration represents.
///
/// Deliberately registers **no** `ActivationMemoryRegistration`: that carrier publishes only real
/// on-device measurements, and the epic's terminal campaign owns the anchors.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    // The three trainers and their training-only deps (MLX distributed / JACCL, dataset image
    // decode, checkpoint state) are the default-on `training` feature (sc-26265), so an
    // inference-only graph compiles none of them.
    #[cfg(feature = "training")]
    let registry = registry
        .register_trainer(train::REGISTRATION)
        .register_trainer(train::depth::REGISTRATION)
        .register_trainer(train::restore::REGISTRATION);
    register_inference_providers(registry)
}

/// Register every Iris inference route, never trainers, even if Cargo unifies `training` on.
/// Registration does not claim a device fit: admission owns that decision for each mode.
pub fn register_inference_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_transform(restoration::REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_composed_memory_strategy(memory_strategy::DEPTH_MEMORY_REGISTRATION)
        .register_composed_memory_strategy(memory_strategy::RESTORE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_CONTRACT_FIXTURE)
        .register_memory_contract_fixture(memory_strategy::DEPTH_MEMORY_CONTRACT_FIXTURE)
        .register_memory_contract_fixture(memory_strategy::RESTORE_MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_behavior(memory_strategy::DEPTH_MEMORY_BEHAVIOR_REGISTRATION)
        .register_memory_behavior(memory_strategy::RESTORE_MEMORY_BEHAVIOR_REGISTRATION)
}

/// Build the complete explicit MLX Iris provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    #[test]
    fn upstream_md_records_the_pinned_revisions() {
        let doc = include_str!("../UPSTREAM.md");
        for sha in [
            super::UPSTREAM_CODE_REVISION,
            super::UPSTREAM_WEIGHTS_REVISION,
            super::TEXT_ENCODER_REVISION,
        ] {
            assert!(doc.contains(sha), "UPSTREAM.md must record {sha}");
        }
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
        let expected_trainers: &[&str] = if cfg!(feature = "training") {
            &["iris_3b", "iris_3b_depth", "iris_3b_restore"]
        } else {
            &[]
        };
        assert_eq!(trainers, expected_trainers);
        assert!(registry.descriptor_conformance_errors().is_empty());
    }
}

mod ios_build_identity;
#[doc(hidden)]
pub use ios_build_identity::{IosCompiledDependencyAnchor, IOS_ENGINEERING_FEATURES_SUPPORTED};
