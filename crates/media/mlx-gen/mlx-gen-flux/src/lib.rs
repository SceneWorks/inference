//! # mlx-gen-flux
//!
//! FLUX.1 provider crate for [`mlx-gen`](mlx_gen). This crate establishes the provider
//! boundary for FLUX.1-schnell and FLUX.1-dev: registry ids, fork-derived variant
//! configuration, tokenizer loading contracts, text encoders, MMDiT transformer, VAE loading, and
//! the base txt2img generation path.

pub mod adapters;
pub(crate) mod artifact_inventory;
pub(crate) mod block_stream;
pub mod config;
pub mod control_transformer;
pub mod convert;
pub mod image_encoder;
pub mod ip_adapter;
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod model_control;
pub mod pipeline;
pub mod preview;
pub mod text_encoder;
pub mod transformer;

pub use adapters::apply_flux_adapters;
pub use config::{
    FluxTokenizerKind, FluxVariant, DEFAULT_GUIDANCE, DEFAULT_HEIGHT, DEFAULT_WIDTH,
    FLUX1_DEV_CONTROL_ID, FLUX1_DEV_ID, FLUX1_SCHNELL_ID,
};
pub use control_transformer::{FluxControlNet, FluxControlNetConfig, FluxControlTransformer};
pub use image_encoder::FluxIpImageEncoder;
pub use ip_adapter::{FluxIpAdapter, FluxIpInjector};
pub use loader::{
    load_clip_encoder, load_clip_tokenizer, load_control_transformer_dev, load_t5_encoder,
    load_t5_tokenizer, load_transformer, load_vae, load_vae_from_weights,
};
pub use model::{
    compile_trajectory_for, descriptor_dev, descriptor_for, descriptor_schnell,
    execute_flux_denoise_trajectory, load_dev, load_schnell, transition_mlx_flux_boundary_state,
    Flux1, SIZE_MULTIPLE,
};
pub use model_control::{descriptor_dev_control, load_dev_control, Flux1DevControl};
pub use pipeline::{
    build_linear_sigmas, create_noise, image_seq_len, pack_latents, unpack_latents,
    LATENT_CHANNELS, LATENT_PATCH_SIZE, PACKED_TOKEN_WIDTH,
};
pub use text_encoder::{
    ClipTextEncoder, FluxTextEncoders, T5BlockStream, T5TextEncoder, T5_BLOCKS,
};
pub use transformer::{FluxTransformer, FluxTransformerConfig};

/// sc-16209 Apple-Silicon warm sweep: FLUX.1 Dev bf16 peaked below 14.06 GiB at 1024².
pub const DEV_ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: FLUX1_DEV_ID,
        anchor: mlx_gen::ActivationMemoryAnchor {
            bytes_1024: 15_096_810_046,
        },
    };

/// Add all MLX FLUX.1 providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_render_plan(mlx_gen::gen_core::RenderPlanRegistration {
            provider_id: FLUX1_SCHNELL_ID,
            compile: |spec, req| model::compile_base(FluxVariant::Schnell, spec, req),
        })
        .register_render_plan(mlx_gen::gen_core::RenderPlanRegistration {
            provider_id: FLUX1_DEV_ID,
            compile: |spec, req| model::compile_base(FluxVariant::Dev, spec, req),
        })
        .register_render_plan(mlx_gen::gen_core::RenderPlanRegistration {
            provider_id: FLUX1_DEV_CONTROL_ID,
            compile: model_control::compile_control,
        })
        .register_feature_route(mlx_gen::gen_core::FeatureRouteRegistration {
            route_id: "flux1_schnell_ip_adapter",
            parent_provider_id: Some(FLUX1_SCHNELL_ID),
            descriptor: model::descriptor_schnell_ip_adapter,
            render_plan: Some(mlx_gen::gen_core::RenderPlanRegistration {
                provider_id: "flux1_schnell_ip_adapter",
                compile: |spec, req| model::compile_ip_adapter(FluxVariant::Schnell, spec, req),
            }),
        })
        .register_feature_route(mlx_gen::gen_core::FeatureRouteRegistration {
            route_id: "flux1_dev_ip_adapter",
            parent_provider_id: Some(FLUX1_DEV_ID),
            descriptor: model::descriptor_dev_ip_adapter,
            render_plan: Some(mlx_gen::gen_core::RenderPlanRegistration {
                provider_id: "flux1_dev_ip_adapter",
                compile: |spec, req| model::compile_ip_adapter(FluxVariant::Dev, spec, req),
            }),
        })
        .register_feature_route(mlx_gen::gen_core::FeatureRouteRegistration {
            route_id: FLUX1_DEV_CONTROL_ID,
            parent_provider_id: Some(FLUX1_DEV_ID),
            descriptor: model_control::descriptor_dev_control,
            render_plan: Some(mlx_gen::gen_core::RenderPlanRegistration {
                provider_id: FLUX1_DEV_CONTROL_ID,
                compile: model_control::compile_control,
            }),
        })
        .register_generator(model::SCHNELL_REGISTRATION)
        .register_memory_strategy(model::SCHNELL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: FLUX1_SCHNELL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(FLUX1_SCHNELL_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: FLUX1_SCHNELL_ID,
                contract: |surface| {
                    memory_strategy::weights_free_memory_surface_contract(FLUX1_SCHNELL_ID, surface)
                },
            },
        )
        .register_memory_behavior(model::SCHNELL_MEMORY_BEHAVIOR)
        .register_generator(model::DEV_REGISTRATION)
        .register_memory_strategy(model::DEV_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: FLUX1_DEV_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(FLUX1_DEV_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: FLUX1_DEV_ID,
                contract: |surface| {
                    memory_strategy::weights_free_memory_surface_contract(FLUX1_DEV_ID, surface)
                },
            },
        )
        .register_memory_behavior(model::DEV_MEMORY_BEHAVIOR)
        .register_activation_memory(DEV_ACTIVATION_MEMORY_REGISTRATION)
        .register_generator(model_control::DEV_CONTROL_REGISTRATION)
        .register_memory_strategy(model_control::DEV_CONTROL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
            provider_id: FLUX1_DEV_CONTROL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(FLUX1_DEV_CONTROL_ID, spec)
            },
        })
        .register_memory_contract_surface_resolver(
            mlx_gen::gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: FLUX1_DEV_CONTROL_ID,
                contract: |surface| {
                    memory_strategy::weights_free_memory_surface_contract(
                        FLUX1_DEV_CONTROL_ID,
                        surface,
                    )
                },
            },
        )
        .register_memory_behavior(model_control::DEV_CONTROL_MEMORY_BEHAVIOR)
}

/// Build the complete explicit MLX FLUX.1 provider catalog.
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

        assert_eq!(
            explicit,
            ["flux1_schnell", "flux1_dev", "flux1_dev_control"]
        );
    }

    #[test]
    fn migrated_flux1_routes_export_feature_applicability_and_refuse_phases() {
        use mlx_gen::gen_core::{
            Conditioning, ControlKind, Error, FeatureInapplicabilityCode, GenerationPhase,
            GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource,
        };

        let registry = super::provider_registry().unwrap();
        let all = registry.all_route_feature_applicability();

        for (route_id, expected_default_steps, expected_max_k) in
            [("flux1_schnell", 4u32, 3u64), ("flux1_dev", 25u32, 24u64)]
        {
            let app = &all[route_id]["low_res_start"];
            assert!(app.applicable, "{route_id} must support low_res_start");
            let json = app.to_json();
            assert_eq!(json["class"], "recipe");
            assert_eq!(json["version"], 1);
            assert_eq!(json["parameter_schema"]["version"], 1);
            assert_eq!(json["parameter_schema"]["k"]["min"], 1);
            assert_eq!(
                json["parameter_schema"]["k"]["default_executed_steps"],
                expected_default_steps
            );
            assert_eq!(json["parameter_schema"]["k"]["max"], expected_max_k);
            assert_eq!(json["parameter_schema"]["scale"]["min"], 0.25);
            assert_eq!(json["parameter_schema"]["scale"]["max"], 0.9);
            assert_eq!(json["parameter_schema"]["scale"]["default"], 0.5);
        }

        let control_app = &all["flux1_dev_control"]["low_res_start"];
        assert!(!control_app.applicable);
        assert!(control_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning));

        for ip_route in ["flux1_dev_ip_adapter", "flux1_schnell_ip_adapter"] {
            let ip_app = &all[ip_route]["low_res_start"];
            assert!(!ip_app.applicable, "{ip_route} must refuse low_res_start");
            assert!(ip_app
                .reasons
                .iter()
                .any(|r| r.code
                    == FeatureInapplicabilityCode::ConflictsWithPass0ReferenceConditioning));
        }

        // Unsupported `phases` must return a typed refusal (`Error::Unsupported`) on every migrated route.
        let base_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let phased_req = GenerationRequest {
            prompt: "a phased test".into(),
            width: 512,
            height: 512,
            phases: Some(vec![GenerationPhase {
                steps: 4,
                guidance: Some(3.5),
                adapters: vec![],
            }]),
            ..Default::default()
        };
        for id in ["flux1_schnell", "flux1_dev"] {
            assert!(matches!(
                registry.compile_render_plan(id, &base_spec, &phased_req),
                Err(Error::Unsupported(_))
            ));
        }
        let img = Image {
            width: 16,
            height: 16,
            pixels: vec![0; 16 * 16 * 3],
        };
        let ip_spec = base_spec
            .clone()
            .with_ip_adapter(WeightsSource::Dir("/nonexistent/ip".into()));
        let ip_phased = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img.clone(),
                strength: Some(0.7),
            }],
            ..phased_req.clone()
        };
        for variant in [super::FluxVariant::Schnell, super::FluxVariant::Dev] {
            assert!(matches!(
                super::model::compile_ip_adapter(variant, &ip_spec, &ip_phased),
                Err(Error::Unsupported(_))
            ));
        }
        let control_spec = base_spec.with_control(WeightsSource::File(
            "/nonexistent/control.safetensors".into(),
        ));
        let control_phased = GenerationRequest {
            conditioning: vec![Conditioning::Control {
                image: img,
                kind: ControlKind::Pose,
                scale: Some(0.7),
            }],
            ..phased_req
        };
        assert!(matches!(
            registry.compile_render_plan("flux1_dev_control", &control_spec, &control_phased),
            Err(Error::Unsupported(_))
        ));
    }
}

#[cfg(test)]
pub(crate) mod model_inference_device;
