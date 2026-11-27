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
pub mod memory_strategy;
pub mod pulid_flux;

/// Add the MLX PuLID-FLUX generator to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(pulid_flux::route_pulid_flux())
        .register_route(pulid_flux::route_flux1_dev_pulid())
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

    #[test]
    fn pulid_routes_export_inapplicability_and_refuse_phases() {
        use mlx_gen::gen_core::{
            Conditioning, Error, FeatureInapplicabilityCode, GenerationPhase, GenerationRequest,
            IdentityWeights, Image, LoadSpec, OffloadPolicy, WeightsSource,
        };

        let registry = super::provider_registry().unwrap();
        let all = registry.all_route_feature_applicability();
        for route_id in ["pulid_flux", "flux1_dev_pulid"] {
            let app = &all[route_id]["low_res_start"];
            assert!(!app.applicable, "{route_id} requires pass-0 reference");
            assert!(app
                .reasons
                .iter()
                .any(|r| r.code
                    == FeatureInapplicabilityCode::ConflictsWithPass0ReferenceConditioning));
        }

        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File("/nonexistent/pulid.safetensors".into())),
            eva: Some(WeightsSource::File("/nonexistent/eva.safetensors".into())),
            face_dir: Some(WeightsSource::Dir("/nonexistent/face".into())),
        });
        let phased_req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 16,
                    height: 16,
                    pixels: vec![0; 16 * 16 * 3],
                },
                strength: Some(1.0),
            }],
            features: mlx_gen::gen_core::FeatureParams {
                phases: Some(vec![GenerationPhase {
                    steps: 4,
                    guidance: Some(4.0),
                    adapters: vec![],
                }]),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            registry.compile_render_plan("pulid_flux", &spec, &phased_req),
            Err(Error::Unsupported(_))
        ));
    }
}

#[cfg(test)]
pub(crate) mod model_inference_device;
