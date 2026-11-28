//! # mlx-gen-flux2
//!
//! The **FLUX.2-klein** provider crate for [`mlx-gen`](mlx_gen). FLUX.2 shares no crate-local
//! code with the FLUX.1 family (`mlx-gen-flux`): its text encoder is a **Qwen3** dense LLM
//! (not T5+CLIP), its VAE is the 32-channel `AutoencoderKL-Flux2` with 2×2 patchify +
//! BatchNorm-stats normalization (not the 16-ch BFL VAE), and its MMDiT uses fused parallel
//! single blocks with a 4-axis RoPE. Everything shared lives in the core `mlx-gen` crate
//! (nn primitives, the `FlowMatchEuler` scheduler, adapters, quant, the `Generator` contract,
//! the registry), so this is a clean sibling crate, decoupled from FLUX.1.
//!
//! Ported from the frozen Python mflux fork (`~/repos/mflux/src/mflux/models/flux2/`) and
//! parity-gated against it. Target: **FLUX.2-klein-9b** (txt2img + edit); the config is
//! dimension-parametric so the 4b variant is a near-free follow-on.
//!
//! Slice status: **S0** — scaffold, registry ids, dimension-parametric config, the flow-match
//! schedule (reused from core), 2×2 latent pack/unpack/patchify, the 4-axis RoPE table, and the
//! latent/text id builders. `load()`/`generate()` are guarded with a clear error until the
//! text-encoder (S1), VAE (S2), and transformer (S3) modules land.

pub mod adapters;
pub mod admission_estimate;
mod artifact_inventory;
mod block_stream;
pub mod caption_upsample;
pub mod chunk;
pub mod config;
pub mod convert;
pub mod kv_cache;
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod model_control;
pub mod pipeline;
pub mod pos_embed;
pub mod preview;
pub mod product_load;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;

pub use adapters::apply_flux2_adapters;
pub use caption_upsample::{
    build_upsample_input_ids, expand_pixtral_image_tokens, upsample_prompt,
    SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I,
};
pub use chunk::{map_seq_chunks, MemoryConfig};
pub use config::{
    Flux2Config, Flux2Quant, Flux2Variant, DEFAULT_GUIDANCE, DEFAULT_GUIDANCE_DEV, DEFAULT_HEIGHT,
    DEFAULT_STEPS, DEFAULT_STEPS_DEV, DEFAULT_WIDTH, FLUX2_DEV_CONTROL_ID, FLUX2_DEV_EDIT_ID,
    FLUX2_DEV_ID, FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID, FLUX2_KLEIN_9B_KV_EDIT_ID,
    SIZE_MULTIPLE,
};
pub use convert::{
    build_target_state_dict, convert_and_assemble, quantize_flux2_dit, quantize_flux2_text_encoder,
    quantize_flux2_text_encoder_dir, quantize_flux2_transformer,
};
pub use kv_cache::{CacheMode, CfgBranch, Flux2KvCache, Flux2KvCfgCaches, Stream};
pub use loader::{
    load_control_transformer_dev, load_multimodal_projector_dev, load_text_encoder,
    load_text_encoder_dev, load_tokenizer, load_tokenizer_dev, load_transformer,
    load_transformer_dev, load_vae, load_vision_tower_dev,
};
pub use model::{
    compile_trajectory_for, descriptor_dev, descriptor_dev_edit, descriptor_klein_9b,
    descriptor_klein_9b_edit, descriptor_klein_9b_kv_edit, load_dev, load_dev_edit, load_klein_9b,
    load_klein_9b_edit, load_klein_9b_kv_edit,
};
pub use model_control::{
    compile_control_trajectory, descriptor_dev_control, load_dev_control, Flux2DevControl,
};
pub use pipeline::{
    add_noise_by_interpolation, create_noise, init_time_step, pack_latents, patchify_latents,
    prepare_grid_ids, prepare_text_ids, preprocess_ref_image, schedule, timesteps_x1000,
    unpack_latents,
};
pub use pos_embed::Flux2PosEmbed;
pub use text_encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
pub use transformer::{
    Flux2ControlBranch, Flux2ControlTransformer, Flux2ForwardInputs, Flux2Transformer,
    Flux2TransformerConfig, CONTROL_IN_DIM,
};
pub use vae::Flux2Vae;
pub use vision::{Mistral3Projector, PixtralVisionConfig, PixtralVisionTower};

/// sc-16209 Apple-Silicon warm sweep: FLUX.2 Klein 9B bf16 peaked below 14.07 GiB at 1024².
pub const KLEIN_ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: FLUX2_KLEIN_9B_ID,
        anchor: mlx_gen::ActivationMemoryAnchor {
            bytes_1024: 15_107_547_464,
        },
    };

/// Add all MLX FLUX.2 providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(model::route_klein())
        .register_route(model::route_klein_edit())
        .register_route(model::route_klein_kv_edit())
        .register_route(model::route_dev())
        .register_route(model::route_dev_edit())
        .register_route(model_control::route_dev_control())
}

/// Build the complete explicit MLX FLUX.2 provider catalog.
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
            [
                "flux2_klein_9b",
                "flux2_klein_9b_edit",
                "flux2_klein_9b_kv_edit",
                "flux2_dev",
                "flux2_dev_edit",
                "flux2_dev_control",
            ]
        );

        let preview_ids: Vec<_> = registry
            .generators()
            .filter_map(|registration| {
                let descriptor = (registration.descriptor)();
                descriptor
                    .capabilities
                    .supports_preview
                    .then_some(descriptor.id)
            })
            .collect();
        assert_eq!(
            preview_ids, explicit,
            "every and only FLUX.2 route previews"
        );
        assert_eq!(registry.route_conformance_errors(), Vec::<String>::new());
    }

    #[test]
    fn dev_edit_exposes_the_provider_memory_safety_contract() {
        let registry = super::provider_registry().unwrap();
        let contract = registry
            .memory_strategy_contract(
                super::FLUX2_DEV_EDIT_ID,
                &mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(Default::default())),
            )
            .unwrap()
            .expect("FLUX.2-dev edit memory contract");
        assert_eq!(contract.provider_id, super::FLUX2_DEV_EDIT_ID);
        assert!(contract.conformance_errors().is_empty());
    }

    #[test]
    fn dev_t2i_exposes_its_own_provider_memory_safety_contract() {
        let registry = super::provider_registry().unwrap();
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(Default::default()))
            .with_quant(mlx_gen::Quant::Q4);
        let t2i = registry
            .memory_strategy_contract(super::FLUX2_DEV_ID, &spec)
            .unwrap()
            .expect("FLUX.2-dev T2I memory contract");
        let edit = registry
            .memory_strategy_contract(super::FLUX2_DEV_EDIT_ID, &spec)
            .unwrap()
            .expect("FLUX.2-dev edit memory contract");

        assert_eq!(t2i.provider_id, super::FLUX2_DEV_ID);
        assert_eq!(edit.provider_id, super::FLUX2_DEV_EDIT_ID);
        assert_ne!(t2i.calibration, edit.calibration);
        assert!(t2i.conformance_errors().is_empty());
        assert_eq!(
            registry
                .memory_strategy_registrations()
                .filter(|registration| registration.provider_id == super::FLUX2_DEV_ID)
                .count(),
            1,
            "the T2I provider must have exactly one memory registration"
        );
    }

    #[test]
    fn klein_shared_ladder_passes_the_weights_free_behavior_oracle() {
        let registry = super::provider_registry().unwrap();
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_pid(
                mlx_gen::WeightsSource::File("/nonexistent/pid.safetensors".into()),
                mlx_gen::WeightsSource::Dir("/nonexistent/gemma".into()),
            );
        gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(&registry, &spec);
        assert!(registry
            .memory_strategy_registrations()
            .any(|registration| registration.provider_id == super::FLUX2_KLEIN_9B_KV_EDIT_ID));
        assert!(registry
            .memory_behavior_registrations()
            .any(|registration| registration.provider_id == super::FLUX2_KLEIN_9B_KV_EDIT_ID));
    }

    #[test]
    fn sw6945_flux2_render_plan_and_low_res_start_conformance() {
        use mlx_gen::gen_core::{
            Conditioning, FeatureInapplicabilityCode, GenerationRequest, Image, LoadSpec,
            LowResStartSpec, WeightsSource,
        };

        let registry = super::provider_registry().unwrap();
        let all_features = registry.all_route_feature_applicability();
        for t2i_id in [super::FLUX2_KLEIN_9B_ID, super::FLUX2_DEV_ID] {
            let app = &all_features[t2i_id]["low_res_start"];
            assert!(app.applicable, "mlx {t2i_id} must support low_res_start");
        }
        for edit_id in [
            super::FLUX2_KLEIN_9B_EDIT_ID,
            super::FLUX2_KLEIN_9B_KV_EDIT_ID,
            super::FLUX2_DEV_EDIT_ID,
        ] {
            let app = &all_features[edit_id]["low_res_start"];
            assert!(!app.applicable, "mlx {edit_id} must refuse low_res_start");
            assert!(app.reasons.iter().any(|r| {
                r.code == FeatureInapplicabilityCode::ConflictsWithPass0EditConditioning
            }));
        }
        let ctrl_app = &all_features[super::FLUX2_DEV_CONTROL_ID]["low_res_start"];
        assert!(!ctrl_app.applicable);
        assert!(ctrl_app.reasons.iter().any(|r| {
            r.code == FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning
        }));

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            seed: Some(42),
            features: mlx_gen::gen_core::FeatureParams {
                low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = registry
            .compile_render_plan(super::FLUX2_KLEIN_9B_ID, &spec, &req)
            .unwrap()
            .expect("flux2_klein_9b render plan");
        assert_eq!(plan.denoise_spans(), vec![0..2, 2..4]);
        assert_eq!(plan.phase_output_size(0), (256, 256));
        assert_eq!(plan.phase_output_size(1), (512, 512));

        // Plain single-source img2img on flux2_klein_9b via Conditioning::Reference with strength.
        let src_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };
        let img2img_req = GenerationRequest {
            steps: Some(10),
            conditioning: vec![Conditioning::Reference {
                image: src_img.clone(),
                strength: Some(0.6),
            }],
            features: mlx_gen::gen_core::FeatureParams {
                low_res_start: Some(LowResStartSpec::new(3, Some(0.5))),
                ..Default::default()
            },
            ..req.clone()
        };
        let img2img_plan = registry
            .compile_render_plan(super::FLUX2_KLEIN_9B_ID, &spec, &img2img_req)
            .unwrap()
            .unwrap();
        assert_eq!(img2img_plan.executed_steps(), 4);
        assert_eq!(img2img_plan.denoise_spans(), vec![0..3, 3..4]);

        // Edit routes refuse low_res_start with a typed error.
        let edit_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: src_img,
                strength: None,
            }],
            ..req
        };
        assert!(registry
            .compile_render_plan(super::FLUX2_KLEIN_9B_EDIT_ID, &spec, &edit_req)
            .is_err());
    }

    #[test]
    fn sw6946_flux2_m3_non_square_plain_img2img_low_res_start_fixture_and_pricing() {
        use mlx_gen::gen_core::{
            memory_strategy::trace_planned_low_res_scope_lifecycle,
            sampling::{compute_mu, image_seq_len, init_time_step},
            CancelFlag, Conditioning, GenerationRequest, Image, LoadShape, LoadSpec,
            LowResStartSpec, MemoryBehaviorRoute, MemoryMode, MemorySafetyDecision, MemoryStrategy,
            OffloadPolicy, WeightsSource, FLUX2_PACKED_LATENT_SPACE,
        };

        let registry = super::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);

        // M3 non-square strength fixture (§6.13):
        // 1200x700 source -> 1024x768 full target, scale=0.5, steps=10, strength=0.4
        // => start_step=4, M=6, K=3, phase-0 512x384 (64x48 latent, 32x24 = 768 packed tokens).
        let source_1200x700 = Image {
            width: 1200,
            height: 700,
            pixels: vec![96; 1200 * 700 * 3],
        };
        let steps = 10_usize;
        let strength = 0.4_f32;
        assert_eq!(init_time_step(steps, Some(strength)), 4);

        // Full-pass schedule at full target geometry (1024x768, seq_len=3072), NOT 512x384 (seq_len=768).
        let full_seq_len = image_seq_len(1024, 768);
        let low_seq_len = image_seq_len(512, 384);
        assert_eq!(full_seq_len, 3072);
        assert_eq!(low_seq_len, 768);
        let full_mu = compute_mu(full_seq_len, steps);
        let low_mu = compute_mu(low_seq_len, steps);
        assert!((full_mu - low_mu).abs() > 1e-4);
        let base_sched =
            crate::pipeline::schedule_with(steps, 1024, 768, None).expect("full schedule");
        let full_sigmas = mlx_gen::resolve_flow_schedule_with_shift(
            None,
            None,
            full_mu,
            steps,
            &base_sched.sigmas,
        )
        .expect("shifted full sigmas");
        let low_base_sched =
            crate::pipeline::schedule_with(steps, 512, 384, None).expect("low schedule");
        let wrong_low_sigmas = mlx_gen::resolve_flow_schedule_with_shift(
            None,
            None,
            low_mu,
            steps,
            &low_base_sched.sigmas,
        )
        .expect("shifted low sigmas");
        assert_eq!(full_sigmas.len(), 11);

        // Native FLUX.2 packed latent space is 32-ch /8 -> 128-ch /16 packed with BatchNorm stats.
        assert_eq!(FLUX2_PACKED_LATENT_SPACE.channels, 128);
        assert_eq!(
            FLUX2_PACKED_LATENT_SPACE.spatial_compression,
            mlx_gen::gen_core::latent::SpatialCompression {
                height: 16,
                width: 16
            }
        );
        assert_eq!(
            FLUX2_PACKED_LATENT_SPACE.patch_layout,
            mlx_gen::gen_core::LatentPatchLayout::Packed {
                patch_height: 2,
                patch_width: 2,
            }
        );

        for provider_id in [super::FLUX2_KLEIN_9B_ID, super::FLUX2_DEV_ID] {
            for via_source_image in [false, true] {
                let mut m3_req = GenerationRequest {
                    prompt: "a brass astrolabe on a wooden desk".into(),
                    width: 1024,
                    height: 768,
                    steps: Some(steps as u32),
                    seed: Some(77),
                    features: mlx_gen::gen_core::FeatureParams {
                        low_res_start: Some(LowResStartSpec::new(3, Some(0.5))),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                if via_source_image {
                    m3_req.source_image = Some(source_1200x700.clone());
                    m3_req.strength = Some(strength);
                } else {
                    m3_req.conditioning = vec![Conditioning::Reference {
                        image: source_1200x700.clone(),
                        strength: Some(strength),
                    }];
                }

                let plan = registry
                    .compile_render_plan(provider_id, &spec, &m3_req)
                    .unwrap_or_else(|e| panic!("{provider_id} (source={via_source_image}): {e}"))
                    .expect("compiled RenderPlan");

                assert_eq!(plan.executed_steps(), 6, "{provider_id}");
                assert_eq!(plan.denoise_spans(), vec![0..3, 3..6], "{provider_id}");
                assert_eq!(plan.phase_output_size(0), (512, 384), "{provider_id}");
                assert_eq!(plan.phase_output_size(1), (1024, 768), "{provider_id}");
                // 512x384 -> 64x48 latent -> 32x24 packed = 768 tokens; 1024x768 -> 64x48 packed = 3072 tokens.
                assert_eq!(plan.phase_packed_tokens(0), 32 * 24, "{provider_id}");
                assert_eq!(plan.phase_packed_tokens(1), 64 * 48, "{provider_id}");

                // Denoise sigmas must slice the full-pass 1024x768 schedule at start_step=4:
                // denoise_sigmas[0] == full_sigmas[4], denoise_sigmas[3] == full_sigmas[7], NOT low_mu sigmas!
                let denoise_sigmas = plan.denoise_sigmas();
                assert_eq!(denoise_sigmas.len(), 7, "{provider_id}");
                assert_eq!(denoise_sigmas, &full_sigmas[4..=10], "{provider_id}");
                assert_eq!(denoise_sigmas[0], full_sigmas[4], "{provider_id}");
                assert_eq!(denoise_sigmas[3], full_sigmas[7], "{provider_id}");
                assert!(
                    (denoise_sigmas[0] - wrong_low_sigmas[4]).abs() > 1e-5,
                    "{provider_id} must use full-pass mu from 1024x768, not 512x384"
                );

                // Initial codec pricing covers raw 1200x700 RGB8 + fitted 512x384 f32 + VAE encode +
                // retained clean phase-0 latent (393,216 B) + noise/blend + phase-0 first forward + node-7 transition ceiling.
                let contract = if provider_id == super::FLUX2_DEV_ID {
                    crate::memory_strategy::registered_dev_t2i_contract(&spec).unwrap()
                } else {
                    crate::memory_strategy::weights_free_klein_contract(provider_id, &spec).unwrap()
                };
                let low_res_plan = plan.low_res_start().expect("low_res_start active");
                let priced = crate::memory_strategy::price_flux2_low_res_img2img_live_set_plan(
                    &contract,
                    low_res_plan,
                    source_1200x700.width,
                    source_1200x700.height,
                    &full_sigmas,
                    4,
                )
                .expect("FLUX.2 initial codec live-set pricing");
                assert_eq!(priced.raw_source_rgb8_bytes, 1200 * 700 * 3);
                assert_eq!(priced.fitted_phase0_input_f32_bytes, 512 * 384 * 3 * 4);
                assert_eq!(priced.retained_clean_init_latent_bytes, 32 * 24 * 128 * 4);
                assert_eq!(
                    priced.transition_ceiling.total_six_buffer_ceiling_bytes,
                    7_077_888
                );
                assert_eq!(priced.init_sigma, full_sigmas[4]);
                assert_eq!(priced.transition_sigma, full_sigmas[7]);
            }
        }

        // Verify dev_t2i_safety_check accepts plain single-source ImageToImage (1 ref),
        // refuses multi-ref Edit, and trace_planned_low_res_scope_lifecycle succeeds on Klein9b img2img.
        let dev_eager_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let dev_contract =
            crate::memory_strategy::registered_dev_t2i_contract(&dev_eager_spec).unwrap();
        let dev_tier =
            crate::model::effective_dev_memory_numeric_tier(&dev_eager_spec, super::FLUX2_DEV_ID)
                .unwrap();
        let mut dev_img2img_ctx = mlx_gen::gen_core::standard_memory_behavior_context(
            &dev_contract,
            MemoryStrategy::StagedResidency,
            dev_tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::ImageToImage,
                reference_count: 1,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        dev_img2img_ctx.geometry.width = 1024;
        dev_img2img_ctx.geometry.height = 768;
        assert_eq!(
            crate::memory_strategy::dev_t2i_safety_check(&dev_contract, &dev_img2img_ctx, dev_tier),
            MemorySafetyDecision::Accept
        );

        let mut dev_multi_ref_edit = dev_img2img_ctx.clone();
        dev_multi_ref_edit.mode = MemoryMode::Edit;
        dev_multi_ref_edit.geometry.reference_count = 2;
        assert!(matches!(
            crate::memory_strategy::dev_t2i_safety_check(
                &dev_contract,
                &dev_multi_ref_edit,
                dev_tier
            ),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut dev_scope = crate::memory_strategy::registered_dev_begin_request(
            &dev_eager_spec,
            &dev_contract,
            &dev_img2img_ctx,
        )
        .unwrap()
        .expect("staged Dev img2img scope");
        let dev_m3_req = GenerationRequest {
            prompt: "a brass astrolabe on a wooden desk".into(),
            width: 1024,
            height: 768,
            steps: Some(10),
            seed: Some(77),
            conditioning: vec![Conditioning::Reference {
                image: source_1200x700,
                strength: Some(0.4),
            }],
            features: mlx_gen::gen_core::FeatureParams {
                low_res_start: Some(LowResStartSpec::new(3, Some(0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let dev_plan = registry
            .compile_render_plan(super::FLUX2_DEV_ID, &dev_eager_spec, &dev_m3_req)
            .unwrap()
            .unwrap();
        let events = trace_planned_low_res_scope_lifecycle(
            &dev_plan,
            &CancelFlag::new(),
            dev_scope.as_mut(),
            None,
            None,
            None,
        )
        .expect("Dev img2img low_res_start scope lifecycle");
        assert!(!events.is_empty());
    }
}

#[cfg(test)]
pub(crate) mod model_inference_device;
