//! Kolors provider for mlx-gen — a bilingual (Chinese/English) SDXL-family T2I model.
//!
//! Kolors keeps the SDXL U-Net + SDXL VAE but swaps dual-CLIP conditioning for a **ChatGLM3-6B**
//! text encoder (penultimate hidden state = context, last-token last-layer state = pooled). This
//! crate is built up across epic 3090:
//!
//!  - [`chatglm3`] — the ChatGLM3-6B encoder-only forward (sc-3091).
//!  - [`tokenizer`] — the ChatGLM3 SentencePiece tokenizer (sc-3092).
//!  - [`unet`] — the SDXL U-Net + ChatGLM3 context-projection wiring (sc-3093).
//!  - the T2I / img2img pipelines (sc-3094/3095), quant (sc-3096), ControlNet / IP-Adapter-Plus
//!    (sc-3097/98).

pub(crate) mod block_stream;
pub mod chatglm3;
pub mod convert;
pub mod ip_adapter;
pub mod memory_strategy;
pub mod model;
pub mod registry;
pub mod sampler;
pub mod tokenizer;
pub mod training;
pub mod unet;

pub use model::Kolors;
pub use registry::{descriptor, KolorsGenerator, MODEL_ID, SIZE_MULTIPLE};
pub use training::{load_trainer, KolorsTrainer};

/// Add the MLX Kolors generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(crate::registry::route())
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX Kolors provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit_generators, ["kolors"]);
        assert_eq!(explicit_trainers, ["kolors"]);
        assert_eq!(registry.route_conformance_errors(), Vec::<String>::new());
    }

    /// Weights-free behavioral oracle for the shared memory ladder (SC-15521).
    ///
    /// This is the check that makes the declaration non-vacuous without weights: for **every**
    /// declared rung it builds the provider's own representative selection, drives the whole request
    /// scope through it (`configure_request` → phases → `configure_decode` / `configure_attention` /
    /// `materialize_transformer_window` → `finish`), and proves the safety check is not blind to an
    /// impossible budget.
    #[test]
    fn shared_ladder_registrations_pass_the_weights_free_behavior_oracle() {
        let registry = super::provider_registry().unwrap();
        for shape in [
            mlx_gen::LoadShape::DeferredMaterialization,
            mlx_gen::LoadShape::EagerMaterialization,
        ] {
            let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
                .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
                .with_load_shape(shape);
            gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(
                &registry, &spec,
            );
        }
    }

    #[test]
    fn kolors_curated_shared_driver_render_plan_and_eng5_refusals() {
        use mlx_gen::gen_core::{
            BoundaryState, Conditioning, ControlKind, EstimateSource, GenerationRequest, Image,
            LoadSpec, LowResStartSpec, WeightsSource,
        };
        use mlx_gen::{AlphaSchedule, DiscreteModelSampling};

        let registry = super::provider_registry().unwrap();
        let all_features = registry.all_route_feature_applicability();
        assert!(
            all_features[super::MODEL_ID]["low_res_start"].applicable,
            "kolors curated route must advertise low_res_start applicability"
        );

        let spec = LoadSpec::new(WeightsSource::Dir("/tmp/kolors".into()));
        let t2i_req = GenerationRequest {
            prompt: "a misty mountain landscape".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            sampler: Some("euler".into()),
            scheduler: Some("normal".into()),
            low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
            ..Default::default()
        };
        let plan = registry
            .compile_render_plan(super::MODEL_ID, &spec, &t2i_req)
            .unwrap()
            .expect("compiled curated plan");
        assert_eq!(plan.executed_steps(), 8);
        assert_eq!(plan.phase_output_size(0), (512, 512));
        assert_eq!(plan.phase_output_size(1), (1024, 1024));
        assert_eq!(plan.denoise_spans(), vec![0..2, 2..8]);

        let img2img_req = GenerationRequest {
            prompt: "a misty mountain landscape".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            sampler: Some("dpmpp_2m".into()),
            scheduler: Some("karras".into()),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![128; 64 * 64 * 3],
                },
                strength: Some(0.5),
            }],
            low_res_start: Some(LowResStartSpec::new(1, Some(0.5))),
            ..Default::default()
        };
        let img_plan = registry
            .compile_render_plan(super::MODEL_ID, &spec, &img2img_req)
            .unwrap()
            .expect("compiled curated img2img plan");
        assert_eq!(img_plan.executed_steps(), 4);
        assert_eq!(img_plan.phase_output_size(0), (512, 512));
        assert_eq!(img_plan.denoise_spans(), vec![0..1, 1..4]);

        let low_res_plan = *img_plan.low_res_start().unwrap();
        let sigma_k = img_plan.denoise_sigmas()[low_res_plan.k];
        let alpha = AlphaSchedule::scaled_linear(
            crate::sampler::NUM_TRAIN_TIMESTEPS,
            crate::sampler::BETA_START,
            crate::sampler::BETA_END,
        );
        let ms = DiscreteModelSampling::sdxl(&alpha);
        let low_latent = vec![0.25f32; 4 * 64 * 64];
        let transitioned = mlx_gen::gen_core::transition_clean_estimate_f32(
            BoundaryState::CleanEstimate {
                latent: &low_latent,
                source: EstimateSource::EndpointEvaluation,
            },
            1,
            4,
            8,
            &low_res_plan,
            sigma_k,
            &ms,
            img_plan.base_seed(),
            0,
            0,
            0,
        )
        .unwrap();
        assert_eq!(transitioned.len(), 4 * 128 * 128);
        assert!(transitioned.iter().all(|v| v.is_finite()));

        // Native euler_discrete/discrete, ControlNet-pose, and IP-Adapter refuse low_res_start
        // with typed Error::Unsupported.
        let dummy_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };
        for (bad_spec, bad_req) in [
            (
                spec.clone(),
                GenerationRequest {
                    sampler: None,
                    scheduler: None,
                    ..t2i_req.clone()
                },
            ),
            (
                spec.clone(),
                GenerationRequest {
                    sampler: Some("euler_discrete".into()),
                    scheduler: Some("discrete".into()),
                    ..t2i_req.clone()
                },
            ),
            (
                spec.clone()
                    .with_control(WeightsSource::Dir("/tmp/ctrl".into())),
                GenerationRequest {
                    conditioning: vec![Conditioning::Control {
                        image: dummy_img,
                        kind: ControlKind::Pose,
                        scale: Some(0.8),
                    }],
                    ..t2i_req.clone()
                },
            ),
            (
                spec.clone()
                    .with_ip_adapter(WeightsSource::Dir("/tmp/ip".into())),
                t2i_req.clone(),
            ),
        ] {
            let err = registry
                .compile_render_plan(super::MODEL_ID, &bad_spec, &bad_req)
                .unwrap_err();
            assert!(
                matches!(err, mlx_gen::gen_core::Error::Unsupported(_)),
                "expected typed Unsupported error, got {err:?}"
            );
        }

        // Curated PiD early-stop (`use_pid = true`, `pid_capture_sigma = Some(0.35)`) compiles a
        // valid RenderPlan preserving the full EDM schedule ending at 0.0 and expressing the
        // early stop through `executed_steps`.
        let pid_req = GenerationRequest {
            prompt: "a misty mountain landscape".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            sampler: Some("euler".into()),
            scheduler: Some("karras".into()),
            use_pid: true,
            pid_capture_sigma: Some(0.35),
            ..Default::default()
        };
        let pid_plan = registry
            .compile_render_plan(super::MODEL_ID, &spec, &pid_req)
            .unwrap()
            .expect("compiled curated PiD early-stop plan");
        assert_eq!(pid_plan.sigmas().last(), Some(&0.0));
        assert_eq!(pid_plan.sigmas().len(), 9);
        assert!(pid_plan.executed_steps() < 8);
        assert_eq!(
            pid_plan.denoise_sigmas().len(),
            pid_plan.executed_steps() + 1
        );
        assert_eq!(
            pid_plan.capture_sigma(),
            pid_plan.sigmas()[pid_plan.executed_steps()]
        );

        // Planned execution rejects a mutated request or mismatched load binding before
        // materializing heavy components.
        let tmp = tempfile::tempdir().unwrap();
        for rel in [
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ] {
            let path = tmp.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![4], &[0u8; 8])
                    .unwrap();
            std::fs::write(path, safetensors::serialize([("w", view)], &None).unwrap()).unwrap();
        }
        let seq_spec = LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential);
        let gen = crate::registry::load(&seq_spec).expect("sequential load defers all weights");
        let base_req = GenerationRequest {
            prompt: "a misty mountain landscape".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            sampler: Some("euler".into()),
            scheduler: Some("karras".into()),
            ..Default::default()
        };
        let bound_plan = registry
            .compile_render_plan(super::MODEL_ID, &seq_spec, &base_req)
            .unwrap()
            .expect("compiled bound plan");
        let mut mutated_req = base_req.clone();
        mutated_req.steps = Some(12);
        let req_err = gen
            .generate_planned(&mutated_req, &bound_plan, None, &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(
            req_err.contains("compiled render plan does not match"),
            "unexpected error: {req_err}"
        );
        let other_plan = registry
            .compile_render_plan(super::MODEL_ID, &spec, &base_req)
            .unwrap()
            .expect("compiled plan for other spec");
        let load_err = gen
            .generate_planned(&base_req, &other_plan, None, &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(
            load_err.contains("different model or adapter stack"),
            "unexpected error: {load_err}"
        );
    }

    #[test]
    fn native_kolors_routes_compile_render_plans() {
        use mlx_gen::gen_core::{
            Conditioning, ControlKind, GenerationRequest, Image, LoadSpec, OffloadPolicy,
            WeightsSource,
        };

        let registry = super::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/tmp/kolors-native".into()));
        let dummy_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };

        let native_req = GenerationRequest {
            prompt: "a misty mountain landscape".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            seed: Some(17),
            ..Default::default()
        };
        let plan = registry
            .compile_render_plan(super::MODEL_ID, &spec, &native_req)
            .unwrap()
            .expect("native Kolors txt2img must compile a RenderPlan");
        assert_eq!(plan.sampler(), "euler_discrete");
        assert_eq!(plan.executed_steps(), 8);
        assert_eq!(plan.sigmas().last(), Some(&0.0));
        let rebound = plan
            .rebind_load_placement(&spec.clone().with_offload_policy(OffloadPolicy::Sequential))
            .unwrap();
        assert_eq!(rebound.executed_steps(), 8);

        let img2img_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: dummy_img.clone(),
                strength: Some(0.5),
            }],
            ..native_req.clone()
        };
        let img_plan = registry
            .compile_render_plan(super::MODEL_ID, &spec, &img2img_req)
            .unwrap()
            .expect("native Kolors img2img must compile a RenderPlan");
        assert_eq!(img_plan.sampler(), "euler_discrete");
        assert_eq!(img_plan.executed_steps(), 4);

        let ctrl_spec = spec
            .clone()
            .with_control(WeightsSource::Dir("/tmp/kolors-pose".into()));
        let ctrl_req = GenerationRequest {
            conditioning: vec![Conditioning::Control {
                image: dummy_img.clone(),
                kind: ControlKind::Pose,
                scale: Some(0.8),
            }],
            ..native_req.clone()
        };
        let ctrl_plan = registry
            .compile_render_plan(super::MODEL_ID, &ctrl_spec, &ctrl_req)
            .unwrap()
            .expect("native Kolors ControlNet-pose must compile a RenderPlan");
        assert_eq!(ctrl_plan.executed_steps(), 8);

        let ip_spec = spec
            .clone()
            .with_ip_adapter(WeightsSource::Dir("/tmp/kolors-ip".into()));
        let ip_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: dummy_img,
                strength: Some(0.6),
            }],
            ..native_req
        };
        let ip_plan = registry
            .compile_render_plan(super::MODEL_ID, &ip_spec, &ip_req)
            .unwrap()
            .expect("native Kolors IP-Adapter must compile a RenderPlan");
        assert_eq!(ip_plan.executed_steps(), 8);
    }
}

#[cfg(test)]
pub(crate) mod model_inference_device;
