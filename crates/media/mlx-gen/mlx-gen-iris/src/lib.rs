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
//! and solver plan live in [`gen_core::iris`] (shared with a Candle twin and the SceneWorks
//! worker). The generation task is **backbone + text encoder**: `LoadSpec::weights` is the backbone
//! directory and `LoadSpec::components["text_encoder"]` the Qwen3-VL snapshot. [`IrisDiT`] and
//! [`load_backbone`] never touch the text encoder, so the depth and restoration tasks reuse them
//! with their own (encoder-free) conditioning.
//!
//! The monocular-depth task ([`depth`], [`depth::DEPTH_MODEL_ID`] = `iris_3b_depth`, sc-25682) is a
//! provider-specific API (like the catalog's other depth estimator), not a registry generator: it
//! returns a float32 relative-log-depth map, not an image. [`depth::load`] resolves only the
//! `depth/` export and returns a `gen_core::iris::depth::IrisDepthEstimator`.
//!
//! ## Frozen upstream
//!
//! See `UPSTREAM.md` (revisions, resource layout, the generation coverage table, fixtures and
//! tolerances). The committed fixtures under `tests/fixtures/` were produced by
//! `tools/dump_iris_*.py` from the pinned revisions on miniature configs.
//!
//! [`gen_core::iris`]: mlx_gen::gen_core::iris

pub mod depth;
pub mod dit;
pub mod model;
pub mod nn;
pub mod pipeline;
pub mod restoration;
pub mod solver;
pub mod text_encoder;

pub use dit::{IrisDiT, TextBatch};
pub use mlx_gen::gen_core::iris::{
    IrisTask, TEXT_ENCODER_COMPONENT, TEXT_ENCODER_REPO, TEXT_ENCODER_REVISION,
    UPSTREAM_CODE_REVISION, UPSTREAM_WEIGHTS_REPO, UPSTREAM_WEIGHTS_REVISION,
};
pub use model::{compute_dtype, descriptor, load, load_backbone, Iris3b, MODEL_ID};
pub use pipeline::{denoise, encode, noise, to_image, Conditioning};
pub use restoration::IrisRestorer;
pub use text_encoder::{IrisTextEncoder, TextConditioning};

/// Add the MLX Iris-3B generator and restoration transform to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_transform(restoration::REGISTRATION)
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
        assert!(registry.descriptor_conformance_errors().is_empty());
    }
}
