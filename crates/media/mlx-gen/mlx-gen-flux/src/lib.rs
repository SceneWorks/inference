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

/// Add all MLX FLUX.1 providers to an explicit media registry builder via one-call
/// [`ProviderRegistryBuilder::register_route`](mlx_gen::gen_core::ProviderRegistryBuilder::register_route)
/// (`SW-6979`, `REG-1`).
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(model::route_schnell())
        .register_route(model::route_dev())
        .register_route(model_control::route_dev_control())
        .register_route(model::route_schnell_ip_adapter())
        .register_route(model::route_dev_ip_adapter())
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
            features: mlx_gen::gen_core::FeatureParams {
                phases: Some(vec![GenerationPhase {
                    steps: 4,
                    guidance: Some(3.5),
                    adapters: vec![],
                }]),
                ..Default::default()
            },
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

    #[test]
    fn route_records_preserve_exact_pre_migration_descriptor_and_capability_parity() {
        use mlx_gen::gen_core::{
            AcceptedControlKinds, ConditioningKind, ControlKind, Modality, Quant, SizeFloor,
            StepSupport, FLUX1_LATENT_SPACE,
        };

        let registry = super::provider_registry().unwrap();
        assert_eq!(registry.route_records().len(), 5);
        assert!(
            registry.route_conformance_errors().is_empty(),
            "route conformance errors: {:?}",
            registry.route_conformance_errors()
        );
        assert!(
            registry.descriptor_conformance_errors().is_empty(),
            "descriptor conformance errors: {:?}",
            registry.descriptor_conformance_errors()
        );

        let schnell = registry.generator_descriptor("flux1_schnell").unwrap();
        assert_eq!(schnell.family, "flux");
        assert_eq!(schnell.backend, "mlx");
        assert_eq!(schnell.modality, Modality::Image);
        assert_eq!(
            schnell.denoiser_output_latent_space,
            Some(&FLUX1_LATENT_SPACE)
        );
        assert!(schnell.encoder_contract.is_none());
        assert!(schnell.control_kinds.is_none());
        assert!(schnell.required_components.is_empty());
        assert!(!schnell.capabilities.supports_negative_prompt);
        assert!(!schnell.capabilities.supports_guidance);
        assert!(!schnell.capabilities.supports_true_cfg);
        assert_eq!(
            schnell.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(
            schnell.capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        assert!(schnell.capabilities.supports_lora);
        assert!(schnell.capabilities.supports_lokr);
        assert!(schnell.capabilities.samplers.contains(&"flow_match"));
        assert!(!schnell.capabilities.samplers.contains(&"hyper"));
        assert!(schnell.capabilities.schedulers.contains(&"linear"));
        assert_eq!(schnell.capabilities.min_size, 256);
        assert_eq!(schnell.capabilities.max_size, 2048);
        assert_eq!(schnell.capabilities.max_count, 8);
        assert_eq!(schnell.capabilities.size_floor, SizeFloor::RangeChecked);
        assert_eq!(
            schnell.capabilities.supported_steps,
            StepSupport::Unconstrained
        );
        assert!(schnell.capabilities.mac_only);
        assert!(!schnell.capabilities.requires_sigma_shift);
        assert!(schnell.capabilities.supports_sequential_offload);
        assert!(schnell.capabilities.supports_preview);
        assert!(schnell.capabilities.supports_low_res_start);
        assert!(!schnell.capabilities.supports_phases);
        assert_eq!(
            schnell.capabilities.pag,
            Some(super::FluxVariant::Schnell.pag_capability())
        );

        let dev = registry.generator_descriptor("flux1_dev").unwrap();
        assert!(dev.capabilities.supports_guidance);
        assert!(dev.capabilities.requires_sigma_shift);
        assert!(dev.capabilities.samplers.contains(&"hyper"));
        assert!(dev.capabilities.supports_low_res_start);
        assert_eq!(
            dev.capabilities.pag,
            Some(super::FluxVariant::Dev.pag_capability())
        );
        assert_eq!(
            registry.activation_memory_bytes_1024("flux1_dev").unwrap(),
            Some(15_096_810_046)
        );
        let dev_route = registry.route_record("flux1_dev").unwrap();
        assert_eq!(dev_route.defaults.steps, Some(25));
        assert_eq!(dev_route.defaults.guidance, Some(3.5));

        let ctrl = registry.generator_descriptor("flux1_dev_control").unwrap();
        assert_eq!(
            ctrl.capabilities.conditioning,
            vec![ConditioningKind::Control]
        );
        assert_eq!(
            ctrl.control_kinds,
            Some(AcceptedControlKinds::Only(vec![
                ControlKind::Pose,
                ControlKind::Canny,
                ControlKind::Depth,
            ]))
        );
        assert!(ctrl.capabilities.supports_guidance);
        assert!(ctrl.capabilities.supports_preview);
        assert!(!ctrl.capabilities.supports_low_res_start);

        for ip_id in ["flux1_schnell_ip_adapter", "flux1_dev_ip_adapter"] {
            let ip_route = registry.route_record(ip_id).unwrap();
            assert!(!ip_route.compute_capabilities().supports_low_res_start);
        }
    }
}

#[cfg(test)]
pub(crate) mod model_inference_device;
