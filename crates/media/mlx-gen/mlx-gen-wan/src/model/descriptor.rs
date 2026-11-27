//! The TI2V-5B, T2V-A14B and I2V-A14B model route declarations and descriptors.

use super::*;
use mlx_gen::gen_core;

/// Primary typed route declaration for Wan2.2 TI2V-5B (`REG-1`, `AP5`).
pub(crate) fn route_ti2v() -> gen_core::RouteRecord {
    gen_core::RouteRecord {
        route_id: MODEL_ID,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        mac_only: true,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![gen_core::TextEncoderDeclaration::new("umt5_xxl", Some(512))],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![ConditioningKind::Reference, ConditioningKind::Keyframe],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: Vec::new(),
            },
            sampling: gen_core::SamplingSurface {
                samplers: wan_samplers(),
                schedulers: Vec::new(),
                guidance_methods: Vec::new(),
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new(
                "wan_dit_5b",
            )),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("wan22_z48_vae"),
                output_latent_space: Some(&mlx_gen::gen_core::WAN_Z48_LATENT_SPACE),
                pixel_stride: Some(32),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            upsamplers: gen_core::UpsamplerSurface::default(),
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::FusedMerge,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["umt5_xxl"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["wan_dit_5b"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["wan22_z48_vae"],
                        true,
                    ),
                ],
                supports_sequential_offload: true,
                unconditionally_engages_staged_residency: true,
                supports_kv_cache: true,
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: crate::memory_strategy::COMPONENT_PRECISION_FLOORS,
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            approximation: approximation_surface(),
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: MIN_SIZE,
            max_size: 1280,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::BackendGlobalSeeded,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(50),
            guidance: Some(5.0),
            sampler: Some("uni_pc"),
            schedule: None,
            shift: Some(5.0),
            strength: None,
            dimensions: Some((1280, 704)),
            ..Default::default()
        },
        bindings: gen_core::RouteBindings::generator(descriptor, |spec| {
            load(spec).map_err(Into::into)
        })
        .with_video_plan(gen_core::VideoPlanRegistration {
            provider_id: MODEL_ID,
            compile: |s, r| video_plan::compile_for(descriptor(), s, r),
        })
        .with_memory_registration(crate::memory_strategy::MEMORY_REGISTRATION)
        .with_memory_fixture_registration(crate::memory_strategy::MEMORY_FIXTURE)
        .with_memory_behavior_registration(crate::memory_strategy::MEMORY_BEHAVIOR),
    }
}

/// Primary typed route declaration for Wan2.2 T2V-A14B (`REG-1`, `AP5`).
pub(crate) fn route_t2v_14b() -> gen_core::RouteRecord {
    let mut route = route_ti2v();
    route.route_id = MODEL_ID_T2V_14B;
    route.structure.conditioning.kinds = Vec::new();
    route.structure.denoiser = gen_core::ModelBinding::family_model(
        gen_core::FamilyModelSpec::new("wan_dit_14b_moe").with_expert_switching(),
    );
    route.structure.latent_codec = gen_core::LatentCodecDeclaration {
        codec_id: Some("wan_z16_vae"),
        output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        pixel_stride: Some(16),
        token_width: None,
        supports_diffusion_decoder: false,
    };
    route.structure.residency.lifetime_graph = vec![
        gen_core::ResidencyPhaseNode::new(
            gen_core::ResidencyPhaseKind::PromptEncode,
            ["umt5_xxl"],
            true,
        ),
        gen_core::ResidencyPhaseNode::new(
            gen_core::ResidencyPhaseKind::Denoise,
            ["wan_dit_14b_moe"],
            true,
        ),
        gen_core::ResidencyPhaseNode::new(
            gen_core::ResidencyPhaseKind::Decode,
            ["wan_z16_vae"],
            true,
        ),
    ];
    route.structure.weight_plan.component_precision_floors = &[];
    route.structure.approximation = Default::default();
    route.limits.min_size = 16;
    route.defaults = gen_core::RouteDefaults {
        steps: Some(40),
        guidance: Some(4.0),
        sampler: Some("uni_pc"),
        schedule: None,
        shift: Some(12.0),
        strength: None,
        dimensions: Some((832, 480)),
        ..Default::default()
    };
    route.bindings = gen_core::RouteBindings::generator(descriptor_t2v_14b, |spec| {
        load_t2v_14b(spec).map_err(Into::into)
    })
    .with_video_plan(gen_core::VideoPlanRegistration {
        provider_id: MODEL_ID_T2V_14B,
        compile: |s, r| video_plan::compile_for(descriptor_t2v_14b(), s, r),
    })
    .with_memory_registration(crate::i2v_memory_strategy::t2v_14b::MEMORY_REGISTRATION)
    .with_memory_fixture_registration(crate::i2v_memory_strategy::t2v_14b::MEMORY_FIXTURE)
    .with_memory_behavior_registration(crate::i2v_memory_strategy::t2v_14b::MEMORY_BEHAVIOR);
    route
}

/// Primary typed route declaration for Wan2.2 I2V-A14B (`REG-1`, `AP5`).
pub(crate) fn route_i2v_14b() -> gen_core::RouteRecord {
    let mut route = route_t2v_14b();
    route.route_id = MODEL_ID_I2V_14B;
    route.structure.conditioning.kinds = vec![ConditioningKind::Reference];
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresReference;
    route.defaults.guidance = Some(3.5);
    route.defaults.shift = Some(5.0);
    route.bindings = gen_core::RouteBindings::generator(descriptor_i2v_14b, |spec| {
        load_i2v_14b(spec).map_err(Into::into)
    })
    .with_video_plan(gen_core::VideoPlanRegistration {
        provider_id: MODEL_ID_I2V_14B,
        compile: |s, r| video_plan::compile_for(descriptor_i2v_14b(), s, r),
    })
    .with_memory_registration(crate::i2v_memory_strategy::i2v_14b::MEMORY_REGISTRATION)
    .with_memory_fixture_registration(crate::i2v_memory_strategy::i2v_14b::MEMORY_FIXTURE)
    .with_memory_behavior_registration(crate::i2v_memory_strategy::i2v_14b::MEMORY_BEHAVIOR);
    route
}

/// Stable identity + advertised capabilities for the Wan2.2 TI2V-5B (dense text+image→video).
pub fn descriptor() -> ModelDescriptor {
    route_ti2v().compute_descriptor()
}

/// Stable identity + advertised capabilities for the Wan2.2 T2V-A14B (dual-expert MoE text→video).
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    route_t2v_14b().compute_descriptor()
}

/// Stable identity + advertised capabilities for the Wan2.2 I2V-A14B (dual-expert MoE image→video).
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    route_i2v_14b().compute_descriptor()
}
