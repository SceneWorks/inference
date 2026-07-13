//! `mlx-gen-pulid` — PuLID-FLUX face-identity provider (epic 3069).
//!
//! Ports the PuLID-FLUX stack to MLX/Rust on top of the existing FLUX.1-dev backbone:
//!   * [`eva_clip`] — the EVA02-CLIP-L-14-336 visual tower (sc-3070) producing `id_cond_vit` + 5
//!     hidden states from the aligned face crop.
//!   * IDFormer perceiver-resampler (sc-3071) — fuses ArcFace + EVA features into the id_embedding.
//!   * PerceiverAttentionCA ×20 injected into the FLUX DiT (sc-3072).
//!   * end-to-end `pulid_flux` generate (sc-3074).
//!
//! Face analysis (ArcFace embedding + `face_features_image`) is the native `mlx-gen-face` stack
//! (epic 3079) — no Python/onnx sidecar.

pub mod ca;
pub mod eva_clip;
pub mod idformer;
pub mod pulid_flux;

/// Add the MLX PuLID-FLUX generator to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(pulid_flux::REGISTRATION)
}

/// Build the complete explicit MLX PuLID provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["pulid_flux"]);
    }
}
